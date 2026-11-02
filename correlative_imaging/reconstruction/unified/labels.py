"""Step 5, your biological labels. The first pass (steps 2-4) gives channel positivity only; cell types come from you.

  sheet       interventions/labels.csv + labels.html: holes drawn at random from every positivity pattern (dim ones
              included) and surrounding cells from every pattern, numbered, each with its composite and one grey image
              per channel. You write a label in your own words (astrocyte, interneuron, ...) or "not a cell", and,
              where you disagree with the first pass or it says dim, the channels you see positive
  your labels set the dim cut per channel (call.py), feed the shape model (cell / not a cell), and train the label
              model: LDA (shrinkage) on the per-channel signals relative to a typical positive cell of the same
              population (holes are dimmed by the hole), holes and surrounding cells together. Leave-one-out accuracy on your labels; every hole and surrounding cell gets the most
              likely label and its probability ("unsure" below 0.7). Labels with fewer than 3 examples are left out
"""
from __future__ import annotations

import html
from pathlib import Path

import numpy as np
import pandas as pd

MIN_PER_LABEL = 3
SURE = 0.7
NOT_A_CELL = ("not a cell", "no cell", "halo", "plug", "debris", "no")


def _norm(s) -> str:
    return str(s or "").strip()


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
    R["your_label"] = R.your_label.map(_norm)
    R["not_cell"] = R.your_label.str.lower().isin(NOT_A_CELL)
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


def suggest(H: pd.DataFrame, O: pd.DataFrame, n_holes: int = 40, n_field: int = 20, seed: int = 0) -> pd.DataFrame:
    """Random holes from every positivity pattern (sqrt of its size, at least 2) and surrounding cells likewise."""
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
    h = draw(H[H.valid], "pattern", n_holes)
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
        "neuron), or <code>not a cell</code> (halo, plug, debris, glow). Leave empty if you cannot tell.</li>"
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


def fit(R: pd.DataFrame, H: pd.DataFrame, O: pd.DataFrame, F: dict, cal: dict) -> dict:
    """The label model from your labels. Adds bio_label / bio_p to H and O; returns what the report shows."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import LeaveOneOut, cross_val_predict
    colour, marker = F["colour"], F["marker"]
    chans = colour + marker
    hole_scale, field_scale = scales(H, O, F)
    # a second view of each colour: in holes the bleed-corrected own signal (sees a dim colour next to a bright one, e.g.
    # FITC under bleed-through into TX RED); surrounding cells have one measure (already corrected), used for both
    hole_own = {c: (f"own_{c}", _typical(H, f"own_{c}", f"pos_{c}", H.valid)) for c in colour}
    field_own = {c: field_scale[c] for c in colour}

    def design(D, a, b):
        return np.hstack([features(D, chans, a, ""), features(D, colour, b, "")])
    info = dict(used=False, n=0, labels={}, why="")
    H["bio_label"], H["bio_p"] = "", np.nan
    O["bio_label"], O["bio_p"] = "", np.nan
    if not len(R):
        info["why"] = "no labels yet"
        return info
    L = R[(R.your_label != "") & ~R.not_cell & (R.your_label.str.lower() != "unsure")]   # unsure is not a type
    # the sheet is read as text: match on text keys
    L = L.drop(columns=["image_id"]).assign(image_id=L.image_id.astype(str)) if "image_id" in L else L
    hk = L[L.kind == "hole"].drop(columns=["image_id"]).merge(
        H.assign(plate_id=H.plate_id.astype(str), well_id=H.well_id.astype(str)), on=["plate_id", "well_id"],
        how="inner", suffixes=("", "_h"))
    Oo = O.assign(obj=O.channel.astype(str) + ":" + O.label.astype(str), image_id=O.image_id.astype(str),
                  plate_id=O.plate_id.astype(str))
    fk = L[L.kind != "hole"].merge(Oo, on=["plate_id", "image_id", "obj"], how="inner", suffixes=("", "_o"))
    X = np.vstack([design(hk, hole_scale, hole_own), design(fk, field_scale, field_own)])
    y = np.concatenate([hk.your_label.to_numpy(), fk.your_label.to_numpy()])
    kind = np.array(["hole"] * len(hk) + ["surrounding"] * len(fk))
    ok = np.isfinite(X).all(1)
    X, y, kind = X[ok], y[ok], kind[ok]
    cnt = pd.Series(y).value_counts()
    keep = cnt[cnt >= MIN_PER_LABEL].index
    m = np.isin(y, keep)
    info.update(labels=cnt.to_dict(), left_out=[k for k in cnt.index if k not in keep])
    if len(keep) < 2:
        info["why"] = f"at least 2 labels with {MIN_PER_LABEL}+ examples needed (have: {cnt.to_dict()})"
        return info
    X, y, kind = X[m], y[m], kind[m]
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    cv = cross_val_predict(LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"), X, y, cv=LeaveOneOut())
    lda.fit(X, y)
    info.update(used=True, n=int(len(y)), acc=float(np.mean(cv == y)),
                acc_hole=float(np.mean(cv[kind == "hole"] == y[kind == "hole"])) if (kind == "hole").any() else np.nan,
                confusion=pd.crosstab(pd.Series(y, name="your label"), pd.Series(cv, name="predicted")),
                classes=list(lda.classes_), X=X, y=y)
    for D, scale, own, sel in [(H, hole_scale, hole_own, H.valid), (O, field_scale, field_own, O.learn)]:
        Xd = design(D, scale, own)
        good = sel.to_numpy() & np.isfinite(Xd).all(1)
        if good.any():
            p = lda.predict_proba(Xd[good])
            lab = lda.classes_[p.argmax(1)]
            D.loc[good, "bio_p"] = p.max(1)
            D.loc[good, "bio_label"] = np.where(p.max(1) >= SURE, lab, "unsure")
    # your own labels win for the holes you labelled
    own = {(r["plate_id"], r["well_id"]): r["your_label"] for r in R[R.kind == "hole"].to_dict("records")
           if r["your_label"]}
    k = [own.get((str(p), str(w))) for p, w in zip(H.plate_id, H.well_id)]
    sel = [v is not None for v in k]
    H.loc[sel, "bio_label"] = [f"{v} (you)" for v in k if v is not None]
    H.loc[sel, "bio_p"] = 1.0
    return info
