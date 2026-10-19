"""Particles of both thresholds in a well's crop.

As pipeline.segment.AutoThreshold + ParticleAnalysis: threshold with the value the run stored, then
remove_small_objects and label.
  seg "whole" (stringent): the whole-image threshold; EVERY object in the crop (field cells + those at the hole)
  seg "hole"  (lenient):   the threshold computed on the hole pixels, applied to the whole crop so a cell
                           crossing the rim stays whole; only objects touching the hole

Per object touching the hole: area inside / in full, share inside, coverage of the hole (fill), mean/max inside,
every channel's mean over the inside footprint (x_<ch>), centroids, an edge flag (cut by the crop border), the
inside footprint (flat pixel indices, for grouping into cells) and `ref` = the stringent object of the same
channel it overlaps most (the whole cell behind a piece).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from skimage.measure import label
from skimage.morphology import remove_small_objects

from .config import Config


def extract_particles(chans: np.ndarray, names: list, hole: np.ndarray, thr: dict, px_um: float, cfg: Config,
                      plate_id: str = "", well_id: str = ""):
    """chans: (C, H, W) fluorescence in the order of `names`; thr[(seg, ch)] -> threshold value.
    Returns (particles DataFrame, label maps {(seg, ch): label image})."""
    S = hole.shape
    hole_px = int(hole.sum())
    hy, hx = np.where(hole)
    hr, hc, R = hy.mean(), hx.mean(), np.sqrt(hole_px / np.pi)
    objs, maps = [], {}
    for seg in ("whole", "hole"):
        for k, ch in enumerate(names):
            t = thr.get((seg, ch))
            if t is None or not np.isfinite(t):
                continue
            L = label(remove_small_objects(chans[k] > t, max_size=cfg.min_object_px))
            maps[(seg, ch)] = L
            touching = set(np.unique(L[hole])) - {0}
            ids = (set(np.unique(L)) - {0}) if seg == "whole" else touching
            for lb in sorted(ids):
                m = L == lb
                ins = m & hole
                if lb in touching and ins.sum() * px_um ** 2 < cfg.min_inside_um2:
                    continue
                y, x = np.where(m)
                o = dict(plate_id=plate_id, well_id=well_id, oid=f"{seg[0]}{k}_{lb}", seg=seg, ch=ch, label=int(lb),
                         touch=lb in touching, full_px=int(m.sum()), in_px=int(ins.sum()),
                         share=ins.sum() / m.sum(), fill=ins.sum() / hole_px,
                         edge=bool(y.min() == 0 or x.min() == 0 or y.max() == S[0] - 1 or x.max() == S[1] - 1),
                         full_mean=float(chans[k][m].mean()), cy=float(y.mean()), cx=float(x.mean()),
                         d_full=float(np.hypot(y.mean() - hr, x.mean() - hc) / R))
                if lb in touching:
                    iy, ix = np.where(ins)
                    o.update(mean=float(chans[k][ins].mean()), max=float(chans[k][ins].max()),
                             foot=np.flatnonzero(ins), d_in=float(np.hypot(iy.mean() - hr, ix.mean() - hc) / R),
                             **{f"x_{c2}": float(chans[j][ins].mean()) for j, c2 in enumerate(names)})
                objs.append(o)
    p = pd.DataFrame(objs)
    if len(p):
        p["ref"] = None
        whole = {(ch): maps[("whole", ch)] for ch in names if ("whole", ch) in maps}
        for i in p.index[p.touch]:
            ch = p.at[i, "ch"]
            if p.at[i, "seg"] == "whole":
                p.at[i, "ref"] = p.at[i, "oid"]
                continue
            if ch not in whole:
                continue
            k = names.index(ch)
            under = whole[ch].ravel()[p.at[i, "foot"]]
            under = under[under > 0]
            if len(under):
                p.at[i, "ref"] = f"w{k}_{np.bincount(under).argmax()}"
    return p, maps
