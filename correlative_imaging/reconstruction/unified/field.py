"""Step 2, the surrounding cells: what cell populations this dataset holds. Everything is learnt from this dataset only.

  objects     every object the pipeline segmented on the whole image, in the colour channels; not touching the frame,
              not debris (smaller than the valley between the speck peak and the cell peak of its channel's size
              curve); one cell seen in two channels with the same centre is kept once
  exemplars   clean surrounding cells: compact (solidity >= 0.9, circularity >= 0.6), typical size for their channel,
              not saturated, and every other colour channel either on the same cell (>= 80 % overlap) or absent
              (<= 5 %). The reference for everything below
  bleed       per channel pair a -> b: the copy of a seen in b, from the ratios signal_b / signal_a of the brighter half
              of the exemplars of a: one group of ratios = all copy (median); two groups (some cells really carry b)
              = the median of the lower group; capped at 30 %. Removed from every object
  positive    per colour channel, from the corrected signal of all objects: where the channel has two groups (two-group
              fit, separation >= 2), the point where both are equally likely; otherwise the mirrored-noise cut (the
              spread of the values below 0, 1 % expected above). Never below 5 noise units, and at least a fifth of a
              typical cell of that colour (a bright cell's spill into a dim channel is not that colour)
  marker      positive where the marker signal splits into two groups (never below 5 noise units); or, if you declare
              a negative pattern (experiment.json: marker_negative, e.g. {"640nm": "561nm+"}), above the 99th
              percentile of the marker signal of the cells with that pattern
  pattern     channel positivity only, named as in the run: "FITC+ TX RED+", "none". No cell types: biological
              labels come only from your labels (interventions/labels.csv)
  cell size   area of one cell body: the area above half of each exemplar's own peak (independent of brightness,
              unlike the threshold outline, which grows with brightness), median per colour; or from cells the user
              marked (interventions/cell_size.csv)
  wells       counts and shares per well; neighbours of each hole (objects whose centre lies within 2 hole radii)
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import label as nd_label
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde, norm

from .experiment import colour_channels, marker_channels, name

GRID = np.linspace(0, 4, 400)
REL = 0.2            # a colour counts from a fifth of a typical cell of that colour


def positivity(D: pd.DataFrame, chans: list, nm: dict) -> pd.Series:
    """Channel positivity named as in the run, e.g. "FITC+ TX RED+"; "none" when no channel is positive."""
    pos = np.column_stack([D[f"pos_{c}"].fillna(False).to_numpy(bool) for c in chans]) if chans else None
    return pd.Series([" ".join(f"{nm[c]}+" for c, p in zip(chans, row) if p) or "none" for row in pos], index=D.index)


def debris_cut(area: np.ndarray) -> tuple[float, float]:
    """(one cell from the area-weighted size curve, debris cut at the valley of the plain curve)."""
    a = np.log10(area[area > 0])
    s = a if len(a) <= 60000 else np.random.default_rng(0).choice(a, 60000, replace=False)
    plain = gaussian_kde(s, bw_method=0.08)(GRID)
    w = gaussian_kde(s, bw_method=0.08, weights=10 ** s)(GRID)
    cell = 10 ** GRID[np.argmax(w)]
    pk, _ = find_peaks(plain, prominence=plain.max() * 0.03)
    small = [i for i in pk if 10 ** GRID[i] < cell / 5]
    if small:
        i0, i1 = small[0], int(np.argmin(np.abs(GRID - np.log10(cell))))
        return float(cell), float(10 ** GRID[i0 + int(np.argmin(plain[i0:i1]))])
    return float(cell), float(cell / 4)


def two_groups(x: np.ndarray):
    """Two-group fit on asinh(x / 2): (cut in x units or None, separation D)."""
    from sklearn.mixture import GaussianMixture
    z = np.arcsinh(x[np.isfinite(x)] / 2).reshape(-1, 1)
    if len(z) < 50:
        return None, 0.0
    gm = GaussianMixture(2, random_state=0).fit(z)
    mu, sd = gm.means_.ravel(), np.sqrt(gm.covariances_.ravel())
    D = abs(mu[1] - mu[0]) / np.sqrt((sd[0] ** 2 + sd[1] ** 2) / 2)
    hi = int(np.argmax(mu))
    grid = np.linspace(mu.min(), mu.max(), 2000).reshape(-1, 1)
    pr = gm.predict_proba(grid)[:, hi]
    return float(2 * np.sinh(grid[np.argmax(pr >= 0.5), 0])), float(D)


def mirrored_cut(neg_side: np.ndarray, fdr: float = 0.01) -> float:
    v = neg_side[np.isfinite(neg_side)]
    s = float(np.median(-v[v < 0]) / 0.6745) if (v < 0).sum() >= 20 else 1.0
    return float(norm.ppf(1 - fdr) * s)


def copy_fraction(r: np.ndarray) -> float:
    """The copy of a in b from the ratios b / a of bright cells of a. One group of ratios: all of it is copy (median).
    Two groups (some cells really carry b): the median of the lower group. Capped at 30 %."""
    from sklearn.mixture import GaussianMixture
    r = r[np.isfinite(r)]
    if len(r) < 30:
        return 0.0
    k = float(np.median(r))
    z = np.log10(np.clip(r, 1e-5, None)).reshape(-1, 1)
    if len(z) >= 100:
        gm = GaussianMixture(2, random_state=0).fit(z)
        mu, sd = gm.means_.ravel(), np.sqrt(gm.covariances_.ravel())
        if abs(mu[1] - mu[0]) / np.sqrt((sd[0] ** 2 + sd[1] ** 2) / 2) >= 2:
            lo = gm.predict(z) == int(np.argmin(mu))
            k = float(np.median(r[lo]))
    return float(np.clip(k, 0, 0.3))


def frame_shape(T: dict, path_map: dict):
    import tifffile
    from .data import local_path
    for r in T["images"].itertuples():
        f = local_path(str(Path(r.diag) / f"{r.well_raw}_whole_channels.tif"), path_map)
        if f.exists():
            with tifffile.TiffFile(f) as tf:
                return tf.pages[0].shape
    return None


def learn(T: dict, P: pd.DataFrame, W: pd.DataFrame, e: dict, path_map: dict, log=print) -> dict:
    colour, marker = colour_channels(e), marker_channels(e)
    nm = {c: name(e, c) for c in colour + marker}
    learn_img = set(zip(W.plate_id[W.use_for_learning], W.image_id[W.use_for_learning]))
    O = P[(P.roi_mask == "whole_image") & P.channel.isin(colour)].copy()
    O["learn"] = [k in learn_img for k in zip(O.plate_id, O.image_id)]
    shp = frame_shape(T, path_map)
    O["r_px"] = np.sqrt(O.area_px / np.pi)
    if shp is not None:
        O["edge"] = ((O.centroid_row - O.r_px < 1) | (O.centroid_col - O.r_px < 1) |
                     (O.centroid_row + O.r_px > shp[0] - 2) | (O.centroid_col + O.r_px > shp[1] - 2))
    else:
        O["edge"] = O.cut_at_edge.astype(bool)
    sizes, cuts_debris = {}, {}
    for c in colour:
        a = O.loc[(O.channel == c) & ~O.edge & O.learn, "area_um2"].to_numpy(float)
        sizes[c], cuts_debris[c] = debris_cut(a) if len(a) > 100 else (np.nan, 20.0)
    O = O[~O.edge & (O.area_um2 >= O.channel.map(cuts_debris))]
    # one cell seen in several channels with the same centre: kept once (the largest outline)
    O = O.sort_values("area_um2", ascending=False)
    keep = []
    for _, g in O.groupby(["plate_id", "image_id"]):
        y, x, r = g.centroid_row.to_numpy(), g.centroid_col.to_numpy(), g.r_px.to_numpy()
        taken = np.zeros(len(g), bool)
        for i in range(len(g)):
            if taken[i]:
                continue
            dup = (np.hypot(y - y[i], x - x[i]) < 0.3 * np.minimum(r, r[i])) & ~taken
            dup[i] = False
            taken |= dup
        keep.append(g.index[~taken])
    O = O.loc[np.concatenate(keep)] if keep else O
    # exemplars
    ok = O.learn & (O.saturated_px == 0) & (O.solidity >= 0.9) & (O.circularity >= 0.6)
    med = O[ok].groupby("channel").area_um2.median()
    ok &= O.area_um2.between(0.5 * O.channel.map(med), 2 * O.channel.map(med))
    for c in colour:
        ov = O[f"ov_{c}"].fillna(0) if f"ov_{c}" in O else pd.Series(0, index=O.index)
        ok &= (O.channel == c) | (ov >= 0.8) | (ov <= 0.05)
    O["exemplar"] = ok
    Y0 = np.clip(O[[f"y_{c}" for c in colour]].to_numpy(float), 0, None)
    seg = O.channel.map({c: i for i, c in enumerate(colour)}).to_numpy()
    K = np.zeros((len(colour), len(colour)))
    ex = O.exemplar.to_numpy()
    for a in range(len(colour)):
        m = ex & (seg == a) & np.isfinite(Y0).all(1)
        if m.sum() < 30:
            continue
        m &= Y0[:, a] > np.median(Y0[m, a])
        for b in range(len(colour)):
            if a != b and m.sum() >= 30:
                K[a, b] = copy_fraction(Y0[m, b] / Y0[m, a])
    Yc = np.where(np.isfinite(Y0), Y0 - np.nan_to_num(Y0) @ K, np.nan)
    for i, c in enumerate(colour):
        O[f"k_{c}"] = Yc[:, i]
    cuts, how = {}, {}
    for i, c in enumerate(colour):
        x = Yc[O.learn.to_numpy(), i]
        cut, D = two_groups(x)
        if cut is not None and D >= 2:
            cuts[c], how[c] = max(5.0, cut), f"two groups (separation {D:.1f})"
        else:
            neg = Yc[O.learn.to_numpy() & (seg != i), i]
            cuts[c], how[c] = max(5.0, mirrored_cut(neg)), f"noise (one group, separation {D:.1f})"
    # a colour also has to reach a fifth of a typical cell of that colour (median of the exemplars segmented in it):
    # a bright cell's small spill into a dim channel (skirt, neighbour, residual copy) passes that channel's noise cut
    typical = {}
    for i, c in enumerate(colour):
        t_ = Yc[ex & (seg == i), i]
        typical[c] = float(np.nanmedian(t_)) if np.isfinite(t_).sum() >= 30 else np.nan
        rel = O[f"k_{c}"] / typical[c] if np.isfinite(typical[c]) else pd.Series(np.inf, index=O.index)
        O[f"rel_{c}"] = rel
        O[f"pos_{c}"] = (O[f"k_{c}"] >= cuts[c]) & (rel >= REL)
    O["colours"] = positivity(O, colour, nm)
    # markers
    mk_cut, mk_how = {}, {}
    for mk in marker:
        O[f"y_{mk}"] = O[f"y_{mk}"].astype(float)
        neg_t = (e.get("marker_negative") or {}).get(nm[mk])
        if neg_t:
            ref = O[O.learn & (O.colours == neg_t)][f"y_{mk}"].dropna()
            if len(ref) >= 50:
                mk_cut[mk], mk_how[mk] = float(np.percentile(ref, 99)), f"99th percentile of {len(ref)} {neg_t} cells"
        if mk not in mk_cut:
            cut, D = two_groups(O.loc[O.learn, f"y_{mk}"].to_numpy(float))
            mk_cut[mk] = max(5.0, cut) if cut is not None else np.nan
            mk_how[mk] = f"two groups (separation {D:.1f})"
        O[f"pos_{mk}"] = O[f"y_{mk}"] >= mk_cut[mk]
    O["pattern"] = positivity(O, colour + marker, nm)
    O["cls"] = O.pattern
    O["large"] = O.area_um2 > 2.5 * O.channel.map(med)
    return dict(O=O, K=K, cuts=cuts, how=how, typical=typical, mk_cut=mk_cut, mk_how=mk_how, colour=colour, marker=marker, nm=nm,
                outline_cell=sizes, debris=cuts_debris, exemplar_area=med.to_dict())


def core_sizes(F: dict, T: dict, e: dict, path_map: dict, n_wells: int = 40, seed: int = 0) -> dict:
    """Area above half of the cell's own peak, on exemplars of a sample of wells (images read once per well)."""
    import tifffile
    from .data import local_path
    O, colour = F["O"], F["colour"]
    lv = T["levels"].set_index(["plate_id", "image_id", "channel"]).thr_whole
    ex = O[O.exemplar]
    wells = ex[["plate_id", "image_id", "diag", "well_raw"]].drop_duplicates(["plate_id", "image_id"])
    wells = wells.sample(min(n_wells, len(wells)), random_state=seed)
    chans = list(e["channels"])
    rows = []
    for w in wells.itertuples():
        f = local_path(str(Path(w.diag) / f"{w.well_raw}_whole_channels.tif"), path_map)
        try:
            with tifffile.TiffFile(f) as tf:
                planes = {c: tf.pages[k].asarray().astype(float) for k, c in enumerate(chans)}
        except Exception:                                                    # noqa: BLE001
            continue
        for c in colour:
            t = lv.get((w.plate_id, w.image_id, c))
            if t is None or not np.isfinite(t):
                continue
            lab, _ = nd_label(planes[c] > t)
            for o in ex[(ex.plate_id == w.plate_id) & (ex.image_id == w.image_id) & (ex.channel == c)].itertuples():
                y, x = int(round(o.centroid_row)), int(round(o.centroid_col))
                if not (0 <= y < lab.shape[0] and 0 <= x < lab.shape[1]) or lab[y, x] == 0:
                    continue
                m = lab == lab[y, x]
                peak = float(np.percentile(planes[c][m], 99))
                rows.append(dict(channel=c, plate_id=o.plate_id, well_id=o.well_id, image_id=o.image_id,
                                 label=o.label, diag=o.diag, well_raw=o.well_raw, px_um=o.px_um,
                                 centroid_row=o.centroid_row, centroid_col=o.centroid_col, area_outline=o.area_um2,
                                 area_core=float((m & (planes[c] >= 0.5 * peak)).sum()) * o.px_um ** 2))
    D = pd.DataFrame(rows)
    out = {}
    for c in colour:
        d = D[D.channel == c] if len(D) else D
        out[c] = float(d.area_core.median()) if len(d) else np.nan
    return dict(core=out, table=D)


def neighbours(F: dict, T: dict, P: pd.DataFrame) -> pd.DataFrame:
    """Per hole: surrounding objects whose centre lies 1-2 hole radii from the hole centre (outside the hole)."""
    O = F["O"].merge(T["holes"][["plate_id", "image_id", "hole_um2", "hole_row", "hole_col"]], on=["plate_id", "image_id"])
    R = np.sqrt(O.hole_um2 / np.pi)
    d = np.hypot(O.centroid_row - O.hole_row, O.centroid_col - O.hole_col) * O.px_um
    O = O.assign(d_hole_r=d / R, rim_um=d - R - np.sqrt(O.area_um2 / np.pi))
    near = O[(O.d_hole_r > 1.0) & (O.d_hole_r <= 2.0)]
    rows = []
    for (p, i), g in near.groupby(["plate_id", "image_id"]):
        g = g.sort_values("rim_um")
        rows.append(dict(plate_id=p, image_id=i, nb_n=len(g), nb_nearest_um=float(g.rim_um.iloc[0]),
                         nb_nearest_cls=g.colours.iloc[0], nb_classes=" / ".join(sorted(set(g.cls)))))
    return pd.DataFrame(rows)
