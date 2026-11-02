"""Step 5, the report (<out>/report.html, self-contained) and the user's input sheets (<out>/interventions/).

Every section: what was done, the number it gave, a diagnostic plot, example images. No bar plots.
"""
from __future__ import annotations

import base64
import html
import io
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from skimage.segmentation import find_boundaries  # noqa: E402

from .vocab import TERMS  # noqa: E402

RNG = np.random.default_rng(0)


def _b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=95, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _rgb(colour_hex: str) -> np.ndarray:
    h = colour_hex.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])


# --------------------------------------------------------------------------------------------------------- crops
class Crops:
    """Cuts surrounding-cell crops from the whole-image TIFs (one read per image, cached)."""

    def __init__(self, e, path_map, top):
        self.e, self.path_map, self.top, self.cache = e, path_map, top, {}

    def planes(self, diag, well_raw):
        import tifffile
        from .data import local_path
        k = (diag, well_raw)
        if k not in self.cache:
            if len(self.cache) > 40:
                self.cache.pop(next(iter(self.cache)))
            f = local_path(str(Path(diag) / f"{well_raw}_whole_channels.tif"), self.path_map)
            with tifffile.TiffFile(f) as tf:
                self.cache[k] = {c: tf.pages[i].asarray().astype(float) for i, c in enumerate(self.e["channels"])}
        return self.cache[k]

    def rgb(self, r, half_um=12.0, which=None):
        p = self.planes(r.diag, r.well_raw)
        h = int(round(half_um / r.px_um))
        y, x = int(r.centroid_row), int(r.centroid_col)
        sl = (slice(max(y - h, 0), y + h), slice(max(x - h, 0), x + h))
        chans = which or [c for c, v in self.e["channels"].items() if v["role"] == "colour"]
        out = None
        from .view import limits
        # colour: minimum to maximum of the crop, channels together (view.limits); one channel alone (grey): its own range
        L = limits({c: p[c][sl] for c in chans}, self.top if which is None else None, together=which is None)
        for c in chans:
            a = p[c][sl]
            lo, hi = L[c]
            v = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
            col = _rgb(self.e["channels"][c]["color"]) if which is None else np.ones(3)
            out = v[..., None] * col if out is None else out + v[..., None] * col
        return np.clip(out, 0, 1)


def hole_rgb(npz: Path, e: dict, white: dict):
    """The hole crop as in the tool (view.py): minimum to maximum, channels stretched together."""
    from .view import limits
    z = np.load(npz)
    names = [str(n) for n in z["names"]]
    out = None
    use = [c for c, v in e["channels"].items() if v["role"] == "colour" and c in names]
    L = limits({c: z["img"][names.index(c)] for c in use}, white, together=True)
    for c, v in e["channels"].items():
        if c not in use:
            continue
        a = z["img"][names.index(c)]
        lo, hi = L[c]
        d = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
        out = d[..., None] * _rgb(v["color"]) if out is None else out + d[..., None] * _rgb(v["color"])
    out = np.clip(out, 0, 1)
    out[find_boundaries(z["hole"])] = (1, 1, 0)
    out[find_boundaries(z["cell"], mode="inner")] = (0, 1, 1)
    return out


def grid(images, labels, title, ncol=8, size=1.35):
    n = max(len(images), 1)
    nrow = int(np.ceil(n / ncol))
    fig, ax = plt.subplots(nrow, ncol, figsize=(size * ncol, size * nrow + 0.4), squeeze=False, facecolor="black")
    for i in range(nrow * ncol):
        a = ax[i // ncol, i % ncol]
        a.axis("off")
        if i < len(images):
            a.imshow(images[i], interpolation="nearest")
            a.set_title(labels[i], fontsize=6, color="w", pad=1)
    fig.suptitle(title, color="w", fontsize=8)
    return fig


# ---------------------------------------------------------------------------------------------------------- write
def write(out: Path, T, P, W, F, C, size, H, cal, e, path_map, summary, linfo=None, R=None, calib=None):
    colour, marker, nm = F["colour"], F["marker"], F["nm"]
    O = F["O"]
    S = []                                          # (title, html)
    col = {c: e["channels"][c]["color"] for c in colour + marker}

    # ---- 1 checks ----
    st = W.status.value_counts()
    few = W[W.status.str.startswith("few cells")]
    txt = (f"<p>{len(W)} images. " + ", ".join(f"{html.escape(k)}: {v}" for k, v in st.items()) + ".</p>")
    if W.flag_name.any():
        bad = W[W.flag_name].groupby(["plate_id", "file_barcode"]).size()
        txt += "<p>File names name another plate than their folder: " + ", ".join(
            f"folder {p}, files {f} ({n})" for (p, f), n in bad.items()) + ". The folder is used.</p>"
    plates = sorted(W.plate_id.unique())
    W2 = W.copy()
    W2["row"], W2["col"] = W2.well_id.str[0], W2.well_id.str[1:].astype(int)
    rows, cols = sorted(W2.row.unique()), sorted(W2.col.unique())
    fig, ax = plt.subplots(1, len(plates), figsize=(3.6 * len(plates), 2.8), squeeze=False)
    for a, p in zip(ax[0], plates):
        g = W2[W2.plate_id == p]
        M = np.full((len(rows), len(cols)), np.nan)
        for r in g.itertuples():
            M[rows.index(r.row), cols.index(r.col)] = r.n_objects
        a.imshow(M, aspect="auto", cmap="viridis", vmin=0, vmax=np.nanpercentile(W2.n_objects, 98))
        f = g[g.status != "ok"]
        a.scatter([cols.index(c) for c in f.col], [rows.index(r) for r in f.row], marker="x", c="r", s=10)
        a.set_title(f"{p}", fontsize=8); a.set_xticks([]); a.set_yticks(range(len(rows))[::3], rows[::3], fontsize=6)
    fig.suptitle("surrounding objects per well (red x = flagged or excluded)", fontsize=9)
    S.append(("1 Well checks", txt + f'<img src="{_b64(fig)}">'))

    # ---- 2 surrounding cells ----
    ex = O[O.exemplar & O.learn]
    t = [f"<p>{int(O.learn.sum())} surrounding objects learnt from ({len(O)} in all), {len(ex)} clean exemplars.</p>"]
    t.append("<table><tr><th>channel</th><th>debris below (µm²)</th><th>threshold outline, typical (µm²)</th>"
             "<th>one cell body (µm²)</th><th>typical cell (noise units)</th><th>positive cut (noise units)</th><th>how</th></tr>"
             + "".join(
        f"<tr><td>{nm[c]}</td><td>{F['debris'][c]:.0f}</td><td>{F['exemplar_area'].get(c, np.nan):.0f}</td>"
        f"<td>{size[c]:.0f}</td><td>{F['typical'][c]:.0f}</td><td>{max(F['cuts'][c], 0.2 * F['typical'][c]):.1f}</td>"
        f"<td>{F['how'][c]}; at least a fifth of a typical cell</td></tr>" for c in colour) + "</table>")
    t.append("<p>Bleed-through measured on the exemplars (copy of the row channel seen in the column channel, % of "
             "its signal): " + ", ".join(f"{k} {100 * v:.1f} %" for k, v in summary["bleed"].items()) + ". "
             + ("A value at the 30 % cap is not bleed-through but cells really carrying both colours (co-expression): "
                "the copy removed is then too large and that colour is under-called in those cells. "
                if any(v >= 0.299 for v in summary["bleed"].values()) else "") + "</p>")
    # size: outline vs core
    D = C["table"]
    fig, ax = plt.subplots(1, len(colour), figsize=(3.6 * len(colour), 3.0), squeeze=False)
    for a, c in zip(ax[0], colour):
        d = D[D.channel == c] if len(D) else D
        if len(d):
            a.scatter(d.area_outline, d.area_core, s=5, alpha=0.4, c=col[c], linewidths=0)
            lim = [1, max(d.area_outline.max(), 10)]
            a.plot(lim, lim, "k:", lw=0.8)
            a.axhline(size[c], color="k", lw=1)
            a.set_xscale("log"); a.set_yscale("log")
        a.set_xlabel("threshold outline (µm²)"); a.set_ylabel("cell body: above half its peak (µm²)")
        a.set_title(f"{nm[c]}: one cell = {size[c]:.0f} µm²", fontsize=8)
    t.append("<p>Cell size. The threshold outline grows with brightness; the body (area above half of the cell's own "
             f"peak) does not. One cell = the median body (line).</p><img src=\"{_b64(fig)}\">")
    # positive cuts: distributions
    fig, ax = plt.subplots(1, len(colour), figsize=(3.6 * len(colour), 3.0), squeeze=False)
    for a, c in zip(ax[0], colour):
        x = np.arcsinh(O.loc[O.learn, f"k_{c}"].dropna() / 2)
        h, edges = np.histogram(x, bins=80)
        a.step(edges[:-1], h, where="post", color=col[c])
        a.axvline(np.arcsinh(F["cuts"][c] / 2), color="k", ls=":")
        a.set_yscale("log")
        tk = [0, 5, 20, 100, 500, 2000]
        a.set_xticks(np.arcsinh(np.array(tk) / 2), tk, fontsize=7)
        a.set_xlabel(f"{nm[c]} signal after bleed-through (noise units)"); a.set_ylabel("objects (log)")
        a.set_title(f"cut {F['cuts'][c]:.1f}: {F['how'][c]}", fontsize=8)
    t.append(f"<p>Positive or not, per colour (dotted = cut).</p><img src=\"{_b64(fig)}\">")
    # patterns
    pt = O[O.learn].pattern.value_counts()
    cl = O[O.learn].cls.value_counts()
    t.append("<table><tr><th>pattern</th><th>surrounding cells</th><th>share</th></tr>" + "".join(
        f"<tr><td>{html.escape(k)}</td><td>{v}</td><td>{100 * v / pt.sum():.1f} %</td></tr>" for k, v in pt.items())
        + "</table>")
    for mk in marker:
        t.append(f"<p>{nm[mk]} (marker) cut {F['mk_cut'][mk]:.1f}: {html.escape(F['mk_how'][mk])}. Share positive "
                 "per pattern: " + ", ".join(f"{k} {100 * g[f'pos_{mk}'].mean():.0f} % (n {len(g)})"
                                          for k, g in O[O.learn].groupby('cls') if len(g) >= 20) + ".</p>")
    # class shares per well, plate maps (colour mix)
    sh = O[O.learn].groupby(["plate_id", "well_id"]).apply(
        lambda g: pd.Series({c: (g[f"pos_{c}"]).mean() for c in colour}), include_groups=False).reset_index()
    sh["row"], sh["col"] = sh.well_id.str[0], sh.well_id.str[1:].astype(int)
    fig, ax = plt.subplots(1, len(plates), figsize=(3.6 * len(plates), 2.8), squeeze=False)
    for a, p in zip(ax[0], plates):
        g = sh[sh.plate_id == p]
        M = np.full((len(rows), len(cols), 3), 0.85)
        for r in g.to_dict("records"):
            M[rows.index(r["row"]), cols.index(r["col"])] = np.clip(sum(_rgb(col[c]) * r[c] for c in colour), 0, 1)
        a.imshow(M, aspect="auto", interpolation="nearest")
        a.set_title(p, fontsize=8); a.set_xticks([]); a.set_yticks(range(len(rows))[::3], rows[::3], fontsize=6)
    fig.suptitle("share of surrounding cells positive in each colour, mixed as colours (pure colour = one type only)",
                 fontsize=9)
    t.append(f"<img src=\"{_b64(fig)}\">")
    # gallery of typical surrounding cells per class
    top = {}
    for c in colour + marker:
        v = (O.loc[O.channel == c, f"med_{c}"] - O.loc[O.channel == c, f"ring_{c}"]).dropna() \
            if f"med_{c}" in O else pd.Series([1.0])
        top[c] = float(np.percentile(v, 90)) if len(v) else 1.0
    white = (summary.get("display_top") or {}).get("field_max") or {}
    cr = Crops(e, path_map, white)
    imgs, labs = [], []
    for k in cl.index[:6]:
        g = ex[ex.cls == k]
        for r in g.iloc[RNG.choice(len(g), min(8, len(g)), replace=False)].itertuples() if len(g) else []:
            try:
                imgs.append(cr.rgb(r)); labs.append(f"{k}")
            except Exception:                                                  # noqa: BLE001
                pass
    if imgs:
        t.append("<p>Typical surrounding cells per pattern (random clean exemplars, 24 µm). A double colour among the "
                 "surrounding cells is often two touching cells rather than one cell with both colours: judge the "
                 "double patterns by eye here.</p>"
                 f"<img src=\"{_b64(grid(imgs, labs, 'surrounding cells by pattern'))}\">")
    S.append(("2 Surrounding cells", "".join(t)))

    # ---- 3/4 hole cells ----
    V = H[H.valid]
    t = [f"<p>{len(H)} holes measured; valid hole cells {len(V)}. Not valid: " + ", ".join(
        f"{html.escape(k)} {v}" for k, v in H[~H.valid].status_cell.value_counts().items()) + ".</p>"]
    t.append("<table><tr><th>colour</th><th>noise SD</th><th>cut (own signal)</th><th>how</th>"
             "<th>brightest empty hole (99th pct)</th><th>positive</th></tr>"
             + "".join(f"<tr><td>{nm[c]}</td><td>{cal[c]['noise_sd']:.1f}</td><td>{cal[c]['cut']:.1f}</td>"
                       f"<td>{cal[c]['measure']}: {cal[c]['how']} (separation {cal[c]['separation']:.1f}; "
                       f"tried {cal[c]['separations']})"
                       + (f"; by your marks: {cal[c]['cut_from_marks']['measure']}, cut x{cal[c]['cut_from_marks']['factor']} "
                          f"({cal[c]['marked']['positive']} seen positive, {cal[c]['marked']['negative']} negative or no cell; "
                          f"balanced accuracy {cal[c]['cut_from_marks']['balanced_accuracy']})"
                          if 'cut_from_marks' in cal[c] else "")
                       + (f"; your marks NOT used ({cal[c]['marked']['positive']} positive, "
                          f"{cal[c]['marked']['negative']} negative; on this measure they separate only to balanced "
                          f"accuracy {cal[c]['marks_not_used']['balanced_accuracy']}, the learnt cut "
                          f"{cal[c]['marks_not_used']['before']}: the measure does not see what you see)"
                          if 'marks_not_used' in cal[c] else "") + "</td>"
                       f"<td>{cal[c]['empty_cut']:.1f} ({cal[c]['n_empty']} holes)</td><td>{cal[c]['n_pos']}</td></tr>"
                       for c in colour) + "</table>")
    sh = cal.get("_shape", {})
    if sh.get("used"):
        t.append(f"<p>Cell or not by shape (no brightness), learnt from {sh['no_cell']} 'no cell' and {sh['cell']} "
                 f"'cell' particles of this dataset: separation on plates the model did not see, AUC "
                 f"{sh.get('auc', float('nan')):.2f}. A colour dimmer than the brightest empty hole counts only when "
                 f"the shape says 'cell'.</p>")
    else:
        t.append("<p>No 'no cell' examples (empty wells or holes you marked), so cell vs glow is judged by brightness "
                 "alone. See 'What you need to do'.</p>")
    for mk in marker:
        t.append(f"<p>{nm[mk]}: cut {cal[mk]['cut']:.1f}, {html.escape(cal[mk]['how'])}; positive {cal[mk]['n_pos']}.</p>")
    fig, ax = plt.subplots(1, len(colour), figsize=(4.2 * len(colour), 3.0), squeeze=False)
    for a, c in zip(ax[0], colour):
        x = np.log10(V[f"x_{c}"].clip(lower=0.01).dropna())
        h, edges = np.histogram(x, bins=60)
        a.step(edges[:-1], h, where="post", color=col[c])
        a.axvline(0, color="k", ls=":")
        a.set_xticks([-2, -1, 0, 1, 2], ["0.01", "0.1", "1 = cut", "10", "100"], fontsize=7)
        a.set_xlabel(f"{nm[c]} signal of the hole cell, in units of its cut"); a.set_ylabel("hole cells")
        a.set_title(nm[c], fontsize=8)
    t.append(f"<p>The hole cells' signal per colour, in units of the learnt cut (dotted = 1). Each hole is measured "
             f"with the best-ranked measurement it has: " + "; ".join(
                 f"{nm[c]}: " + ", ".join(f"{k} {v}" for k, v in cal[c]["used"].items() if v) for c in colour)
             + f".</p><img src=\"{_b64(fig)}\">")
    cc = H.cls.value_counts()
    t.append("<table><tr><th>pattern</th><th>holes</th><th>of which clear (no flag)</th></tr>" + "".join(
        f"<tr><td>{html.escape(k)}</td><td>{v}</td><td>{int(((H.cls == k) & (H.confidence == 'clear')).sum())}</td></tr>"
        for k, v in cc.items()) + "</table>")
    fl = pd.Series([f for s in H["flags"] for f in s.split("; ") if f]).value_counts()
    t.append("<p>Flags: " + ", ".join(f"{html.escape(k)} {v}" for k, v in fl.items()) + ".</p>")
    # bias check: field vs hole
    fp = O[O.learn].cls.value_counts(normalize=True)
    hp = V.cls.value_counts(normalize=True)
    t.append("<table><tr><th>pattern</th><th>surrounding cells</th><th>hole cells</th></tr>" + "".join(
        f"<tr><td>{html.escape(k)}</td><td>{100 * fp.get(k, 0):.1f} %</td><td>{100 * hp.get(k, 0):.1f} %</td></tr>"
        for k in sorted(set(fp.index) | set(hp.index), key=lambda k: -fp.get(k, 0))) + "</table>"
        "<p>Large differences point at the hole: dimming, neighbours, or which cells get patched.</p>")
    topH = {c: float(np.nanpercentile(V.loc[V[f"pos_{c}"], f"snr_{c}"], 90)) if V[f"pos_{c}"].any() else 50.0
            for c in colour}
    dims = [k for k in cc.index if k.endswith("(dim)")]
    for title, sel in [(f"pattern {k}", V[(V.cls == k) & (V.confidence == "clear")])
                       for k in [k for k in cc.index if not k.endswith("(dim)")][:6]] + \
                      [(f"{k} (below the cut: check)", V[V.cls == k]) for k in dims] + \
                      [(f"flag: {f}", H[H["flags"].str.contains(f, regex=False)])
                       for f in [f for f in fl.index if not f.startswith("dim: below")][:4]]:
        imgs, labs = [], []
        for r in sel.iloc[RNG.choice(len(sel), min(8, len(sel)), replace=False)].itertuples() if len(sel) else []:
            f = out / "wells" / f"{r.plate_id}_{r.well_id}.npz"
            if f.exists():
                imgs.append(hole_rgb(f, e, white)); labs.append(f"{r.plate_id[-5:]} {r.well_id}")
        if imgs:
            t.append(f"<p>{html.escape(title)} (random; yellow = hole, cyan = hole cell; each channel linear from its background to the brightest pixels, at least 15 noise SDs)</p>"
                     f"<img src=\"{_b64(grid(imgs, labs, title, ncol=6, size=2.2))}\">")
    S.append(("3 Hole cells", "".join(t)))

    # ---- 4 hand labels and predicted labels ----
    li = linfo or {}
    t = []
    if not li.get("used"):
        t.append("<p>No label model yet: the calls above are channel positivity only. Cell types come only from hand "
                 "labels (see 'What you need to do'). " + html.escape(li.get("why", "")) + "</p>")
    else:
        t.append(f"<p>Learnt from {li['n']} hand labels (" + ", ".join(
            f"{html.escape(str(k))} {v}" for k, v in li["labels"].items()) + ")" + (
            f"; left out (fewer than 3): {html.escape(', '.join(map(str, li['left_out'])))}" if li.get("left_out")
            else "") + ". Model: LDA on each channel's signal relative to a typical positive cell of the same "
            "population (hole cells are dimmed by the hole, surrounding cells are not), holes and surrounding cells "
            f"together. Each hand label predicted by a model trained on all the others (leave-one-out): "
            f"<b>{100 * li['acc']:.0f} %</b> right"
            + (f", on holes {100 * li['acc_hole']:.0f} %" if np.isfinite(li.get("acc_hole", np.nan)) else "")
            + ". Every cell gets a predicted label when its probability is at least 0.7, otherwise 'not sure'. "
              "Two separate columns in hole_calls.csv and field_cells.parquet: <code>predicted_label</code> (with "
              "<code>predicted_p</code>) and <code>hand_label</code>.</p>")
        cf = li["confusion"]
        t.append("<table><tr><th>hand label \\ predicted (left out)</th>" + "".join(
            f"<th>{html.escape(str(c))}</th>" for c in cf.columns) + "</tr>" + "".join(
            f"<tr><td>{html.escape(str(i))}</td>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>"
            for i, row in zip(cf.index, cf.to_numpy())) + "</table>")
        M_ = li.get("model_table")
        if M_ is not None and np.isfinite(M_.ld1).any():
            two = np.isfinite(M_.ld2).any()
            fig, a = plt.subplots(figsize=(5.2, 3.8))
            cmap = plt.get_cmap("tab10")
            for n_, c_ in enumerate(sorted(M_.hand_label.unique())):
                for kind_, mk in (("hole", "o"), ("surrounding", "^")):
                    m = (M_.hand_label == c_) & (M_.kind == kind_)
                    a.scatter(M_.ld1[m], M_.ld2[m] if two else np.zeros(int(m.sum())), s=22, marker=mk,
                              color=cmap(n_ % 10), edgecolor="white", lw=0.5,
                              label=f"{c_} ({kind_})")
            a.set_xlabel("LD1"); a.set_ylabel("LD2" if two else "")
            a.legend(fontsize=7, frameon=False, ncol=2)
            t.append("<p>The label model: every hand-labelled cell on the two discriminant axes the model learnt "
                     "(circles: holes, triangles: surrounding cells). Coordinates in label_model.csv.</p>"
                     f"<img src=\"{_b64(fig)}\">")
        from .vocab import shown
        ct = pd.crosstab(V.pattern, V.predicted_label.map(shown))
        t.append("<p>Hole cells: first-pass pattern vs predicted label.</p><table><tr><th>pattern</th>" + "".join(
            f"<th>{html.escape(str(c))}</th>" for c in ct.columns) + "</tr>" + "".join(
            f"<tr><td>{html.escape(str(i))}</td>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>"
            for i, row in zip(ct.index, ct.to_numpy())) + "</table>")
        fo = O[O.learn].predicted_label.map(shown).value_counts()
        t.append("<p>Surrounding cells by predicted label: " + ", ".join(
            f"{html.escape(str(k))} {v} ({100 * v / fo.sum():.0f} %)" for k, v in fo.items()) + ".</p>")
        for lab in [c for c in V.predicted_label.value_counts().index if c and c not in ("unsure", "not sure")][:6]:
            sel = V[(V.predicted_label == lab)]
            imgs, labs = [], []
            for r in sel.iloc[RNG.choice(len(sel), min(8, len(sel)), replace=False)].itertuples():
                f = out / "wells" / f"{r.plate_id}_{r.well_id}.npz"
                if f.exists():
                    imgs.append(hole_rgb(f, e, white)); labs.append(f"{r.plate_id[-5:]} {r.well_id} p {r.predicted_p:.2f}")
            if imgs:
                t.append(f"<p>Hole cells predicted {html.escape(str(lab))} (random)</p>"
                         f"<img src=\"{_b64(grid(imgs, labs, str(lab), ncol=6, size=2.2))}\">")
    S.append(("4 Hand labels and predicted labels", "".join(t)))

    # ---- 5 cell colours ----
    if "hue" in H:
        from . import hue as hue_mod
        t = ["<p>Each cell's colour as you see it on screen: the channels' display colours mixed by each channel's "
             "signal relative to a typical positive cell (hole cells and surrounding cells each against their own). "
             "The line is the share of cells per 10 degrees of hue; the ring shows the hues and where each channel's "
             "own colour sits. Cells with several channels about equally strong mix to near white and are counted "
             f"below the plot, not placed on the circle; cells weaker than {hue_mod.WEAK:.2f} of a typical positive "
             "cell in every channel have no colour and are left out.</p>"]
        for i, (title, fig) in enumerate(hue_mod.figures(H, O, e)):
            if i == 0:
                fig.savefig(out / "cell_colours.png", dpi=110)       # the tool shows it next to the plate
            t.append(f"<h3>{html.escape(title)}</h3><img style='max-width:520px' src=\"{_b64(fig)}\">")
        S.append(("5 Cell colours", "".join(t)))

    # ---- 6 does more data help ----
    if calib:
        from . import calibration as cb
        t = ["<p>Every quantity learnt above, learnt again on random subsets of growing size (" + str(cb.DRAWS) +
             " draws per size): the line is the median, the band 5-95 % of the draws, the last point all the data. "
             "<b>Flat and narrow at the right end: enough data.</b> Still moving or wide: more data would change it. "
             "For the label model: rising at the right end = labelling more still helps (accuracy on the labels left "
             "out of each draw).</p>"]
        for g, curves in calib.items():
            fig = cb.figure(g, curves, nm, col, cb.XLABEL.get(g, "n"))
            if g.startswith("label model"):
                fig.savefig(out / "calib_labels.png", dpi=100)                 # the Learn tab shows it
            t.append(f"<h3>{html.escape(g)}</h3><img src=\"{_b64(fig)}\"><ul>" + "".join(
                f"<li>{html.escape(' -> '.join(nm.get(x, x) for x in k.split(' -> ')))}: "
                f"{html.escape(cb.verdict(D))}</li>" for k, D in curves.items()) + "</ul>")
        S.append(("6 Does more data help?", "".join(t)))

    # ---- interventions ----
    iv = out / "interventions"
    iv.mkdir(exist_ok=True)
    todo = []
    if len(few) and not (iv / "wells_to_check.csv").exists():
        few[["plate_id", "well_id", "n_objects", "plate_median"]].assign(decision="").to_csv(
            iv / "wells_to_check.csv", index=False)
    if not (iv / "cell_size.csv").exists() and len(C["table"]):
        cand = []
        for c in colour:
            g = ex[ex.channel == c]
            cand.append(g.iloc[RNG.choice(len(g), min(12, len(g)), replace=False)] if len(g) else g)
        cand = pd.concat(cand).reset_index(drop=True)
        imgs, labs = [], []
        for i, r in enumerate(cand.itertuples()):
            try:
                imgs.append(cr.rgb(r)); labs.append(f"#{i + 1} {nm[r.channel]}")
            except Exception:                                                  # noqa: BLE001
                imgs.append(np.zeros((4, 4, 3))); labs.append(f"#{i + 1}")
        grid(imgs, labs, "cell size check: is the centred object ONE cell? (24 um crops)").savefig(
            iv / "cell_size_sheet.png", dpi=110, facecolor="black")
        plt.close("all")
        cand.assign(number=np.arange(1, len(cand) + 1), single_cell="",
                    area_core_um2=np.nan)[["number", "plate_id", "well_id", "channel", "area_um2", "area_core_um2",
                                           "single_cell"]].to_csv(iv / "cell_size.csv", index=False)
    # the label sheet: random holes and surrounding cells from every positivity pattern, interspersed; rewritten while
    # you have not labelled anything in it yet
    from . import labels as labels_mod
    f_lab = iv / "labels.csv"
    untouched = R is None or not len(R) or ((R.your_label == "") & R.positive.isna()).all()
    if f_lab.exists() and untouched:                # never overwrite a sheet with any entry in it
        Rd = pd.read_csv(f_lab, dtype=str).fillna("")
        untouched = not ((Rd.get("your_label", "") != "") | (Rd.get("positive_channels", "") != "")).any()
    if untouched:
        Sh = labels_mod.suggest(H, O)
        Sh.to_csv(f_lab, index=False)
        labels_mod.sheet(Sh, H, O, out, e, Crops(e, path_map, white), white, iv / "labels.html")
    # what to do: the first pass's open questions (the same list as in the tool), then your labels
    linfo = linfo or {}
    try:
        from .questions import open_questions
        for q in open_questions(out):
            if q["state"] == "open":
                todo.append(f"<li><b>{html.escape(q['title'])}</b>: {html.escape(q['why'])} Needed: "
                            f"{html.escape(str(q['needed']))}.</li>")
    except Exception as ex:                                                  # noqa: BLE001
        todo.append(f"<li>(questions not available: {html.escape(str(ex))})</li>")
    if not linfo.get("used"):
        todo.append("<li><b>Your labels</b>: what kind of cell, in your words, on holes and surrounding cells drawn at "
                    "random from every channel pattern. About 10 per label." +
                    (f" Now: {html.escape(linfo.get('why', ''))}." if linfo.get("why") else "") + "</li>")
    S.insert(0, ("What you need to do", "<p>In <b>ci reconstruct v2</b> (<code>ci reconstruct --v2</code>): the First pass tab asks the questions, the Label "
                 "tab takes your labels; then run again. Without the tool: interventions/answers.csv and labels.csv."
                 "</p><ol>" + "".join(todo) + "</ol>"
                 + ("<p>Applied this run: " + "; ".join(summary["applied"]) + ".</p>" if summary["applied"] else "")))
    S.append(("Vocabulary", "<dl>" + "".join(f"<dt>{html.escape(k)}</dt><dd>{html.escape(v)}</dd>" for k, v in TERMS)
              + "</dl>"))
    css = ("body{font-family:system-ui,sans-serif;max-width:1100px;margin:auto;padding:16px;line-height:1.5;color:#1c2230;"
           "background:#f6f7f9}h1{margin:0}h2{margin-top:32px;border-bottom:1px solid #ccd}img{max-width:100%;display:block;"
           "margin:8px 0}table{border-collapse:collapse;font-size:14px;margin:8px 0}td,th{border-bottom:1px solid #dde;"
           "padding:4px 10px;text-align:left}dt{font-weight:600;margin-top:8px}dd{margin-left:16px}code{background:#e8eaf0;"
           "padding:1px 4px}")
    body = "".join(f"<h2>{html.escape(k)}</h2>{v}" for k, v in S)
    (out / "report.html").write_text(encoding="utf-8", data=
        f"<!doctype html><meta charset='utf-8'><title>Cell classification report</title><style>{css}</style>"
        f"<h1>Cell classification</h1><p>{html.escape(summary['run'])} - {html.escape(summary['results'])}</p>{body}")
