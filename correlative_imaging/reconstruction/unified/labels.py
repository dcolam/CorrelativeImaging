"""Step 5, your biological labels. The first pass (steps 2-4) gives channel positivity only; cell types come from you.

  sheet       interventions/labels.csv + labels.html: holes drawn at random from every positivity pattern (dim ones
              included) and surrounding cells from every pattern, numbered, each with its composite and one grey image
              per channel. You write a label in your own words (astrocyte, interneuron, ...) or "not a cell", and,
              where you disagree with the first pass or it says dim, the channels you see positive
  your labels set the dim cut per channel (call.py), feed the shape model (cell / not a cell), and train the label
              model: LDA (shrinkage) on the per-channel signals relative to a typical positive cell of the same
              population (holes are dimmed by the hole), holes and surrounding cells together. Leave-one-out accuracy
              on the hand labels. Labels with fewer than 3 examples are left out.
  outputs     two separate columns on every hole and surrounding cell: predicted_label (the model's most likely label,
              "not sure" below 0.7) with predicted_p, and hand_label (the label given by hand, empty if none).
              label_model.csv: every hand-labelled cell with its two discriminant coordinates (LD1, LD2), its hand
              label and the leave-one-out prediction (what the report and figures draw)
"""
from __future__ import annotations

import html
import re
from pathlib import Path

import numpy as np
import pandas as pd

MIN_PER_LABEL = 3
SURE = 0.7
NOT_SURE = "not sure"
NOT_A_CELL = ("not a cell", "no cell", "halo", "plug", "debris", "no", "plug / halo", "empty", "nothing")
NOT_CELL = "no cell"     # one label for every hole without a cell: empty, or an object that is not a cell (plug, halo,
                         # debris); learnt like any cell type. The reason stays in status_cell
UNSURE = ("unsure", "not sure", "?", "skip", "skip (not clean or can't tell)")


def _norm(s) -> str:
    return str(s or "").strip()


def _key(s) -> str:
    """Spelling-tolerant key: case, spaces, underscores and dashes do not matter ("GFP_ROMK" = "gfp romk")."""
    return re.sub(r"[\s_\-]+", " ", _norm(s).lower()).strip()


def tidy(labels: pd.Series) -> pd.Series:
    """One spelling per label (the most used one); every "not a cell" synonym becomes NOT_CELL; "unsure" stays."""
    k = labels.map(_key)
    best = {}
    for kk, grp in labels.groupby(k):
        best[kk] = grp.value_counts().index[0] if kk else ""
    out = k.map(best)
    out[k.isin(NOT_A_CELL)] = NOT_CELL
    out[k.isin(UNSURE)] = "unsure"
    return out


def read(f: Path, e: dict) -> pd.DataFrame:
    """labels.csv with the channels parsed: positive = set of raw channel keys, or None when left empty."""
    if not f.exists():
        return pd.DataFrame()
    R = pd.read_csv(f, dtype=str).fillna("")
    by_name = {v["name"].lower(): c for c, v in e["channels"].items()}
    by_name.update({c.lower(): c for c in e["channels"]})

    def chans(v):
        v = v.strip().lower()
        if not v:
            return None
        if v in ("none", "-", "0"):
            return set()
        out = set()
        for tok in v.replace(",", " ").replace("+", " ").split():
            out |= {by_name[tok]} if tok in by_name else set()
        # names with spaces ("tx red")
        for n, c in by_name.items():
            if " " in n and n in v:
                out.add(c)
        return out
    R["positive"] = R.positive_channels.map(chans)
    R["your_label"] = tidy(R.your_label.map(_norm))
    R["not_cell"] = R.your_label == NOT_CELL
    return R


def hole_marks(R: pd.DataFrame, chans: list = ()) -> tuple[dict, dict]:
    """(shape marks {(plate, well): "yes"/"no"}, channel marks {(plate, well): {channel: positive?}})."""
    shape_m, chan_m = {}, {}
    if not len(R):
        return shape_m, chan_m
    for r in R[R.kind == "hole"].to_dict("records"):
        k = (r["plate_id"], r["well_id"])
        if r["not_cell"]:
            shape_m[k] = "no"
        elif r["your_label"] or r["positive"] is not None:
            shape_m[k] = "yes"
        if r["positive"] is not None and not r["not_cell"]:
            chan_m[k] = {c: c in r["positive"] for c in chans}
    return shape_m, chan_m


NOT_CLEAN = ("possible clump", "neighbour of another type", "glow from neighbour")


def clean(H: pd.DataFrame) -> pd.Series:
    """Holes worth a biological label: one cell in the hole, no neighbour of another type touching it, not a clump.
    Clumps and neighbours are the tool's job (flags, questions), never the Label tab's."""
    fl = H["flags"].astype(str) if "flags" in H else pd.Series("", index=H.index)
    ok = (H.status_cell.astype(str) == "cell") if "status_cell" in H else pd.Series(True, index=H.index)
    ok &= ~fl.str.contains("|".join(NOT_CLEAN))      # clump: the flag (learnt size factor), not a fixed size rule
    return ok


def suggest(H: pd.DataFrame, O: pd.DataFrame, n_holes: int = 40, n_field: int = 20, seed: int = 0) -> pd.DataFrame:
    """Random clean holes (see clean) from every positivity pattern (sqrt of its size, at least 2) and surrounding
    cells likewise."""
    rng = np.random.default_rng(seed)

    def draw(D, key, n):
        if not len(D):
            return D
        cnt = D[key].value_counts()
        w = np.sqrt(cnt)
        k = np.maximum(2, np.round(n * w / w.sum())).astype(int)
        parts = [D[D[key] == p].sample(min(int(k[p]), int(cnt[p])), random_state=int(rng.integers(1e9)))
                 for p in cnt.index]
        return pd.concat(parts)
    h = draw(H[H.valid & clean(H)], "pattern", n_holes)
    h = pd.DataFrame(dict(kind="hole", plate_id=h.plate_id, well_id=h.well_id, image_id=h.image_id, obj="",
                          first_pass=h.pattern))
    f = draw(O[O.learn & O.exemplar], "pattern", n_field) if "exemplar" in O else draw(O[O.learn], "pattern", n_field)
    f = pd.DataFrame(dict(kind="field", plate_id=f.plate_id, well_id=f.well_id, image_id=f.image_id,
                          obj=f.channel.astype(str) + ":" + f.label.astype(str), first_pass=f.pattern))
    S = pd.concat([h, f]).sample(frac=1, random_state=seed).reset_index(drop=True)      # interspersed
    S.insert(0, "number", np.arange(1, len(S) + 1))
    return S.assign(your_label="", positive_channels="")


def sheet(S: pd.DataFrame, H: pd.DataFrame, O: pd.DataFrame, out: Path, e: dict, crops, top_hole: dict,
          path: Path) -> None:
    """labels.html: per item its number, what the first pass says, the composite and one grey image per channel."""
    from .report import _b64, hole_rgb
    import matplotlib.pyplot as plt
    chans = list(e["channels"])
    rows = []
    Oi = O.assign(obj=O.channel.astype(str) + ":" + O.label.astype(str)).set_index(["plate_id", "image_id", "obj"])
    for r in S.to_dict("records"):
        try:
            if r["kind"] == "hole":
                z = np.load(out / "wells" / f"{r['plate_id']}_{r['well_id']}.npz")
                names = [str(n) for n in z["names"]]
                comp = hole_rgb(out / "wells" / f"{r['plate_id']}_{r['well_id']}.npz", e, top_hole)
                grey = {}
                for c in chans:
                    if c in names:
                        k = names.index(c)
                        from .view import display_range
                        lo, hi = display_range(z["img"][k])            # one channel alone: its own range
                        grey[c] = np.clip((z["img"][k] - lo) / max(hi - lo, 1e-6), 0, 1)
            else:
                o = Oi.loc[(r["plate_id"], r["image_id"], r["obj"])]
                o = o.iloc[0] if isinstance(o, pd.DataFrame) else o
                o = pd.Series(o).rename(None)
                comp = crops.rgb(_Row(o))
                grey = {c: crops.rgb(_Row(o), which=[c])[..., 0] for c in chans}
        except Exception:                                                        # noqa: BLE001
            continue
        fig, ax = plt.subplots(1, 1 + len(chans), figsize=(1.9 * (1 + len(chans)), 2.1), facecolor="black")
        ax[0].imshow(comp, interpolation="nearest"); ax[0].set_title("composite", color="w", fontsize=7)
        for a, c in zip(ax[1:], chans):
            if c in grey:
                a.imshow(grey[c], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            a.set_title(e["channels"][c]["name"], color="w", fontsize=7)
        for a in ax:
            a.axis("off")
        rows.append(f"<tr><td><b>#{r['number']}</b><br>{html.escape(r['kind'])}<br>{html.escape(r['plate_id'])} "
                    f"{html.escape(r['well_id'])}<br>first pass: <b>{html.escape(r['first_pass'])}</b></td>"
                    f"<td><img src=\"{_b64(fig)}\"></td></tr>")
    css = ("body{font-family:system-ui,sans-serif;background:#f6f7f9;color:#1c2230;margin:16px}td{vertical-align:top;"
           "padding:6px 10px;border-bottom:1px solid #dde}img{max-width:100%}code{background:#e8eaf0;padding:1px 4px}")
    names = ", ".join(f"<code>{html.escape(v['name'])}</code>" for v in e["channels"].values())
    path.write_text(encoding="utf-8", data=
        f"<!doctype html><meta charset='utf-8'><title>Label sheet</title><style>{css}</style><h1>Label sheet</h1>"
        "<p>Holes (yellow = hole, cyan = hole cell) and surrounding cells (centred), drawn at random from every "
        "channel-positivity pattern of the first pass. For each number, in <code>labels.csv</code>:</p><ul>"
        "<li><code>your_label</code>: what the cell is, in your own words (e.g. astrocyte, interneuron, excitatory "
        "neuron), or <code>no cell</code> (empty, halo, plug, debris, glow). Leave empty if you cannot tell.</li>"
        f"<li><code>positive_channels</code>: only where you disagree with the first pass or it says dim: the "
        f"channels you see positive ({names}), or <code>none</code>.</li></ul>"
        "<p>About 10 per label you use is enough. Then run the same command again.</p>"
        f"<table>{''.join(rows)}</table>")


class _Row:
    """Attribute access to a surrounding-cell row, for Crops.rgb."""

    def __init__(self, s: pd.Series):
        self.__dict__.update({k: s[k] for k in s.index})


def features(D: pd.DataFrame, chans: list, scale: dict, kind: str) -> np.ndarray:
    """Per channel signal relative to a typical positive cell of that population, compressed (sign*log(1+|v|))."""
    cols = []
    for c in chans:
        cut = scale[c][1] if (scale[c][1] is not None and np.isfinite(scale[c][1]) and scale[c][1] > 0) else 1.0
        v = D[scale[c][0]].astype(float) / cut
        cols.append(np.sign(v) * np.log1p(np.abs(v)))
    return np.column_stack(cols) if cols else np.zeros((len(D), 0))


def _typical(D, col, pos, sel):
    """Median of a column over the positive cells of a population (None with fewer than 5)."""
    if col not in D or pos not in D:
        return None
    v = D.loc[sel & D[pos].fillna(False).astype(bool), col].astype(float)
    v = v[np.isfinite(v) & (v > 0)]
    return float(v.median()) if len(v) >= 5 else None


def scales(H: pd.DataFrame, O: pd.DataFrame, F: dict) -> tuple[dict, dict]:
    """{channel: (column, typical positive value)} for hole cells and for surrounding cells. Each channel is taken
    relative to a typical positive cell of the same population: hole cells are dimmed by the hole, surrounding cells
    are not, so raw signals or cut units would not be comparable between the two."""
    colour, marker = F["colour"], F["marker"]
    typical = _typical
    hcol = {c: f"x_{c}" for c in colour}
    hcol.update({m: f"hsig_{m}" if f"hsig_{m}" in H else f"own_{m}" for m in marker})
    fcol = {c: f"k_{c}" for c in colour}
    fcol.update({m: f"y_{m}" for m in marker})
    hole = {c: (hcol[c], typical(H, hcol[c], f"pos_{c}", H.valid)) for c in colour + marker}
    field = {c: (fcol[c], typical(O, fcol[c], f"pos_{c}", O.learn)) for c in colour + marker}
    return hole, field


def _answers_not_cell(R: pd.DataFrame, reviewed: dict) -> pd.DataFrame:
    """Holes answered "not a cell" in the tool's questions (channel, clump, particle answers) that have no row in
    labels.csv: extra "not a cell" examples, so every hand input counts once."""
    have = {(str(p), str(w)) for p, w in zip(R.plate_id, R.well_id)} if len(R) else set()
    rows = [dict(kind="hole", plate_id=str(p), well_id=str(w), image_id="", obj="", your_label=NOT_CELL,
                 positive=None, not_cell=True)
            for (p, w), v in (reviewed or {}).items() if v == "no" and (str(p), str(w)) not in have]
    return pd.DataFrame(rows)


def fit(R: pd.DataFrame, H: pd.DataFrame, O: pd.DataFrame, F: dict, cal: dict, reviewed: dict | None = None) -> dict:
    """The label model, in two steps. 1) cell or no cell: the first pass (shape model and size; your "no cell" labels
    and answers train the shape model). 2) the type, among cells only: LDA on the colours, learnt from your type labels
    ("no cell" is not a type). Adds predicted_label / predicted_p / hand_label to H and O; every hole gets a
    predicted_label: a type, "not sure", "no cell" (empty hole, an object that is not a cell, or your "no cell" label;
    the reason is in status_cell) or "hole detection failed"."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import LeaveOneOut, cross_val_predict
    colour, marker = F["colour"], F["marker"]
    chans = colour + marker
    hole_scale, field_scale = scales(H, O, F)
    # a second view of each colour: in holes the bleed-corrected own signal (sees a dim colour next to a bright one, e.g.
    # FITC under bleed-through into TX RED); surrounding cells have one measure (already corrected), used for both
    hole_own = {c: (f"own_{c}", _typical(H, f"own_{c}", f"pos_{c}", H.valid)) for c in colour}
    field_own = {c: field_scale[c] for c in colour}
    # a third view: the particle's colour (the object in the hole itself). The grown cell can follow a bright
    # neighbour and lose the patched cell's colour (Olympus PNC: blue particle, grown cell red); the particle keeps it.
    # Surrounding cells are particles already: their one measure again
    hole_part = {c: (f"part_{c}", _typical(H, f"part_{c}", f"pos_{c}", H.valid)) if f"part_{c}" in H
                 else hole_own[c] for c in colour}
    field_part = {c: field_scale[c] for c in colour}

    def design(D, a, b, p, field):
        # the type is a colour question: colour only (cell or not was decided before, by shape and size)
        # nothing grown in the hole = no own signal: 0, not missing (a hole valid by its particle alone is still a cell)
        # plus the first pass's call per colour (positive or not, 0 or 2: the range of the log signals). Without it a
        # dim but positive cell (e.g. an astrocyte with weak BFP) sat nearer the other type's average and was lost
        pos = np.column_stack([D[f"pos_{c}"].fillna(False).astype(float).to_numpy() * 2.0 if f"pos_{c}" in D
                               else np.zeros(len(D)) for c in colour])
        return np.nan_to_num(np.hstack([features(D, chans, a, ""), features(D, colour, b, ""),
                                        features(D, colour, p, ""), pos]), nan=0.0)
    info = dict(used=False, n=0, labels={}, why="")
    for D in (H, O):
        D["predicted_label"], D["predicted_p"], D["hand_label"] = "", np.nan, ""
    # holes with nothing to classify get their state as the label: never an empty cell
    st = H.status_cell.astype(str)
    H.loc[st == "hole detection failed", "predicted_label"] = "hole detection failed"
    H.loc[st.isin(["no cell", "speck (< 1/10 cell)", "mostly outside the hole"]), "predicted_label"] = "no cell"
    H.loc[st == "no cell (shape)", "predicted_label"] = NOT_CELL
    extra = _answers_not_cell(R, reviewed)
    R = pd.concat([R, extra], ignore_index=True) if len(extra) else R
    _hand(R, H, O)
    if not len(R):
        info["why"] = "no labels yet"
        return info
    L = R[(R.your_label != "") & (R.your_label != "unsure")]                 # unsure is not a type
    # two steps: cell or no cell is decided by the first pass (shape model and size, which your "no cell" labels and
    # answers train); the type only among cells, from your type labels. "no cell" is not a type: as one of the
    # classes it took every red, cell-like object to "not sure" (red debris looks like a red cell in colour)
    info["no_cell_labels"] = int(L.not_cell.sum())
    L = L[~L.not_cell]
    # the sheet is read as text: match on text keys
    L = L.drop(columns=["image_id"]).assign(image_id=L.image_id.astype(str)) if "image_id" in L else L
    L = L.assign(plate_id=L.plate_id.astype(str), well_id=L.well_id.astype(str))
    hk = L[L.kind == "hole"].drop(columns=["image_id"]).drop_duplicates(["plate_id", "well_id"]).merge(
        H.assign(plate_id=H.plate_id.astype(str), well_id=H.well_id.astype(str)), on=["plate_id", "well_id"],
        how="inner", suffixes=("", "_h"))
    hk = hk[hk.status_cell.astype(str) != "hole detection failed"]
    Oo = O.assign(obj=O.channel.astype(str) + ":" + O.label.astype(str), image_id=O.image_id.astype(str),
                  plate_id=O.plate_id.astype(str))
    fk = L[L.kind != "hole"].merge(Oo, on=["plate_id", "image_id", "obj"], how="inner", suffixes=("", "_o"))
    X = np.vstack([design(hk, hole_scale, hole_own, hole_part, False),
                   design(fk, field_scale, field_own, field_part, True)])
    y = np.concatenate([hk.your_label.to_numpy(), fk.your_label.to_numpy()])
    kind = np.array(["hole"] * len(hk) + ["surrounding"] * len(fk))
    keys = pd.concat([hk[["plate_id", "well_id"]].assign(image_id="", obj=""),
                      fk[["plate_id", "well_id", "image_id", "obj"]]], ignore_index=True)
    ok = np.isfinite(X).all(1)
    X, y, kind, keys = X[ok], y[ok], kind[ok], keys[ok].reset_index(drop=True)
    cnt = pd.Series(y).value_counts()
    keep = cnt[cnt >= MIN_PER_LABEL].index
    m = np.isin(y, keep)
    info.update(labels=cnt.to_dict(), left_out=[k for k in cnt.index if k not in keep],
                from_answers=int(len(extra)))
    if len(keep) < 2:
        info["why"] = f"at least 2 labels with {MIN_PER_LABEL}+ examples needed (have: {cnt.to_dict()})"
        return info
    X, y, kind, keys = X[m], y[m], kind[m], keys[m].reset_index(drop=True)
    # every label weighs the same, however many examples it has
    pri = np.full(len(np.unique(y)), 1 / len(np.unique(y)))
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=pri)
    # leave-one-out: each hand label predicted by a model trained on all the others (probabilities kept for ROCs)
    cvp = cross_val_predict(LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=pri), X, y,
                            cv=LeaveOneOut(), method="predict_proba")
    lda.fit(X, y)
    cv = np.asarray(lda.classes_)[cvp.argmax(1)]
    info.update(used=True, n=int(len(y)), acc=float(np.mean(cv == y)),
                acc_hole=float(np.mean(cv[kind == "hole"] == y[kind == "hole"])) if (kind == "hole").any() else np.nan,
                confusion=pd.crosstab(pd.Series(y, name="hand label"), pd.Series(cv, name="predicted")),
                classes=list(lda.classes_), X=X, y=y)
    # the model as a picture: the hand-labelled cells on the discriminant axes (the eigen solver gives the same
    # shrinkage LDA with a projection; lsqr has none)
    try:
        Z = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto", priors=pri).fit(X, y).transform(X)
    except Exception:                                                        # noqa: BLE001
        Z = np.full((len(y), 1), np.nan)
    Z = np.column_stack([Z, np.full((len(y), 2), np.nan)])[:, :2]
    info["model_table"] = keys.assign(kind=kind, hand_label=y, ld1=Z[:, 0], ld2=Z[:, 1], loo_predicted=cv,
                                      **{"loo_p_" + re.sub(r"\W+", "_", str(c)).strip("_"): cvp[:, j]
                                         for j, c in enumerate(lda.classes_)})
    # step 2 on every hole the first pass calls a cell, and every surrounding cell; a hole you labelled "no cell"
    # keeps it (your label wins over the first pass)
    no_hand = (H.hand_label.astype(str) == NOT_CELL).to_numpy()
    for D, scale, own, part, sel, field in [(H, hole_scale, hole_own, hole_part, H.valid & ~no_hand, False),
                                            (O, field_scale, field_own, field_part, O.learn, True)]:
        Xd = design(D, scale, own, part, field)
        good = sel.to_numpy() & np.isfinite(Xd).all(1)
        if good.any():
            p = lda.predict_proba(Xd[good])
            cls = np.asarray(lda.classes_)
            lab = cls[p.argmax(1)]
            # not sure: name the two candidates (alphabetical, so each pair is one category), e.g. "excitatory or
            # inhibitory" = a neuron whose type is unclear
            o = np.argsort(-p, axis=1)
            two = np.array([" or ".join(sorted((str(cls[a]), str(cls[b])))) if len(cls) > 1 else NOT_SURE
                            for a, b in zip(o[:, 0], o[:, min(1, len(cls) - 1)])], dtype=object)
            for j, c in enumerate(cls):             # every label's probability: curves of certainty vs calls
                col = "p_" + re.sub(r"\W+", "_", str(c)).strip("_")
                if col not in D:
                    D[col] = np.nan
                D.loc[good, col] = p[:, j]
            D.loc[good, "predicted_p"] = p.max(1)
            D.loc[good, "predicted_label"] = np.where(p.max(1) >= SURE, lab, two)
    H.loc[no_hand & (H.status_cell.astype(str) != "hole detection failed"), "predicted_label"] = NOT_CELL
    return info


def _hand(R: pd.DataFrame, H: pd.DataFrame, O: pd.DataFrame) -> None:
    """hand_label on the holes and surrounding cells labelled by hand (any answer: a type, not a cell, unsure)."""
    if not len(R):
        return
    R = R[R.your_label != ""]
    hole = {(str(r["plate_id"]), str(r["well_id"])): r["your_label"] for r in R[R.kind == "hole"].to_dict("records")}
    H["hand_label"] = [hole.get((str(p), str(w)), "") for p, w in zip(H.plate_id, H.well_id)]
    fld = {(str(r["plate_id"]), str(r["image_id"]), str(r["obj"])): r["your_label"]
           for r in R[R.kind != "hole"].to_dict("records")}
    if fld and "label" in O:
        O["hand_label"] = [fld.get((str(p), str(i), f"{c}:{l}"), "")
                           for p, i, c, l in zip(O.plate_id, O.image_id, O.channel, O.label)]
