"""Particles of one well, at two thresholds, with raw measurements (no model, no calibration yet).

As pipeline.segment.AutoThreshold + ParticleAnalysis: threshold with the run's value, remove_small_objects, label.
  level "whole" (stringent): the whole-image threshold. Every object is kept: the ones touching the hole (pool),
                             the cells around it in the context disc (context, competitors) and every other cell of
                             the detected area (`outer`: neighbours by position, shown and clickable, not classified
                             and left out of every feature).
  level "hole"  (lenient):   the threshold computed on the hole pixels, applied to the whole window so a cell crossing
                             the rim stays whole. Only objects touching the hole.

The detected area is the whole hole crop (or +- view_R of the frame); objects reaching into the analysis window
(`WellData.box`, +- work_R hole radii) are measured clipped to it, so the model's features do not depend on how much
of the crop is detected.

Per object: full-object shape and intensity (every channel), and for objects touching the hole the same inside the
hole (share, fill, distance to the hole centre), the number of peaks (h-maxima of the smoothed channel, height
`peak_h` hole-noise SDs: a peak that does not rise that far above its saddle does not count) and `parent`, the
stringent object of the same channel it overlaps most (the whole cell behind a lenient piece).

Well statistics: the frame background per channel (from the source) and the hole's own background (median / robust
SD of hole pixels outside every object), which judges growth and peaks inside the brighter, noisier aperture.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.measure import label, regionprops
from skimage.morphology import h_maxima, remove_small_objects

from .config import Config
from .sources import WellData

LEVELS = ("whole", "hole")


def label_maps(wd: WellData, cfg: Config) -> dict:
    maps = {}
    for lev in LEVELS:
        for k, ch in enumerate(wd.names):
            t = wd.thr.get((lev, ch))
            if t is None or not np.isfinite(t):
                continue
            maps[(lev, ch)] = label(remove_small_objects(wd.chans[k] > t, max_size=cfg.min_object_px)).astype(np.int32)
    return maps


def hole_background(wd: WellData, maps: dict) -> dict:
    """Median and robust SD of hole pixels outside every object, per channel ((frame mean, sd) if < 30 pixels)."""
    occupied = np.zeros(wd.hole.shape, bool)
    for L in maps.values():
        occupied |= L > 0
    free = wd.hole & ~occupied
    out = {}
    for k, ch in enumerate(wd.names):
        if free.sum() >= 30:
            v = wd.chans[k][free]
            m = float(np.median(v))
            s = float(1.4826 * np.median(np.abs(v - m)))
            out[ch] = (m, s if s > 0 else float(v.std() or 1.0), int(free.sum()))
        else:
            out[ch] = (*wd.bg[ch], 0)
    return out


def smoothed(wd: WellData, cfg: Config) -> np.ndarray:
    if not cfg.smooth_px:
        return wd.chans
    return np.stack([ndi.gaussian_filter(c, cfg.smooth_px) for c in wd.chans])


def peak_maps(sm: np.ndarray, names: list, hbg: dict, cfg: Config) -> dict:
    """Labelled h-maxima per channel, height = peak_h x hole-noise SD."""
    out = {}
    for k, ch in enumerate(names):
        h = cfg.peak_h * max(hbg[ch][1], 1e-6)
        out[ch] = label(h_maxima(sm[k].astype(np.float64), h))
    return out


def edit_maps(wd: WellData, maps: dict, sm: np.ndarray, hbg: dict, cfg: Config, edits: dict | None):
    """Hand cuts, splits at dips and merges (edits.py), in place on `maps`. Returns the origin maps (labels before
    any change) and adds maps[("splits", "")]: 1 = applied split between pieces of one object, 2 = hand cut,
    3 = proposed split (Detect found a dip; not applied unless `split_particles` or accepted for that particle)."""
    from .edits import apply_cuts, apply_merges, auto_split, cut_mask, split_lines, watershed_lines
    edits = edits or {}
    S = wd.hole.shape
    origin = {k: L.copy() for k, L in maps.items()}
    proposals = {}
    cut = cut_mask(S, edits.get("cuts", []), wd.offset) if edits.get("cuts") else None
    lines = np.zeros(S, np.uint8)
    max_px = int(cfg.split_max_x_hole * wd.hole.sum())
    oy, ox = wd.offset
    # on-demand watersheds: their dividing lines join the hand cuts (all channels, both levels)
    ws = {}
    for a in edits.get("watersheds", []):
        ws.setdefault((a["level"], a["channel"]), []).append((int(round(a["y"] - oy)), int(round(a["x"] - ox))))
    for (lev, ch), picks in ws.items():
        if (lev, ch) not in maps:
            continue
        L0 = apply_cuts(maps[(lev, ch)], cut) if cut is not None else maps[(lev, ch)]
        k = wd.names.index(ch)
        wl, _ = watershed_lines(L0, sm[k] - hbg[ch][0], picks, cfg.peak_h * max(hbg[ch][1], 1e-6), cfg.min_object_px)
        if wl.any():
            cut = wl if cut is None else (cut | wl)
    for (lev, ch), L in list(maps.items()):
        if cut is not None:
            L = apply_cuts(L, cut)
        k = wd.names.index(ch)
        prop = auto_split(L, sm[k] - hbg[ch][0], wd.hole, cfg.peak_h * max(hbg[ch][1], 1e-6), cfg.split_dip_ratio,
                          cfg.min_object_px, max_px)
        proposals[(lev, ch)] = prop
        if cfg.split_particles:
            L = prop
        else:
            acc = [(int(round(a["y"] - oy)), int(round(a["x"] - ox))) for a in edits.get("splits", [])
                   if a["channel"] == ch and a["level"] == lev]
            labs = {int(L[y, x]) for y, x in acc if 0 <= y < S[0] and 0 <= x < S[1] and L[y, x]}
            if labs:
                L = L.copy()
                nxt = int(L.max()) + 1
                for lb in labs:
                    m = L == lb
                    for i, pl in enumerate(sorted(set(np.unique(prop[m])) - {0})):
                        if i:                                   # first piece keeps the old label
                            L[m & (prop == pl)] = nxt
                            nxt += 1
            proposed = split_lines(prop, L)
            lines[proposed] = 3
        mg = [((m["y1"] - oy, m["x1"] - ox), (m["y2"] - oy, m["x2"] - ox)) for m in edits.get("merges", [])
              if m["channel"] == ch and m["level"] == lev]
        if mg:
            L, _ = apply_merges(L, origin[(lev, ch)], mg)
        maps[(lev, ch)] = L
        lines[split_lines(L, origin[(lev, ch)])] = 1
    if cut is not None:
        anyfg = np.zeros(S, bool)
        for L in origin.values():
            anyfg |= L > 0
        lines[cut & anyfg] = 2
    maps[("splits", "")] = lines
    origin["_proposals"] = proposals
    return origin


def detect_well(wd: WellData, cfg: Config, edits: dict | None = None):
    """Returns (particles, maps, well_row). particles: one row per object (see module doc).
    edits: {"cuts": [polyline, ...], "merges": [{channel, level, y1, x1, y2, x2}, ...],
            "splits": [{channel, level, y, x}, ...], "watersheds": [{channel, level, y, x}, ...]} in frame
            coordinates (splits = accepted proposals, watersheds = split on demand)."""
    S = wd.hole.shape
    hole_px = int(wd.hole.sum())
    hy, hx = np.nonzero(wd.hole)
    hr, hc, R = hy.mean(), hx.mean(), np.sqrt(hole_px / np.pi)
    yy, xx = np.indices(S)
    context = np.hypot(yy - hr, xx - hc) <= cfg.context_R * R
    maps = label_maps(wd, cfg)
    hbg = hole_background(wd, maps)
    sm = smoothed(wd, cfg)
    origin = edit_maps(wd, maps, sm, hbg, cfg, edits)
    proposals = origin.pop("_proposals")
    peaks = peak_maps(sm, wd.names, hbg, cfg)
    stack = np.moveaxis(wd.chans, 0, -1)            # (H, W, C) for regionprops
    objs = []
    bs = wd.box_slices
    inbox = np.zeros(S, bool)
    inbox[bs] = True
    for (lev, ch), L in maps.items():
        if lev == "splits":
            continue
        O = origin[(lev, ch)]
        fg = (L > 0) & (O > 0)
        pairs = np.unique(np.c_[O[fg], L[fg]], axis=0) if fg.any() else np.zeros((0, 2), int)
        parts_of = pd.Series(pairs[:, 1]).groupby(pairs[:, 0]).nunique().to_dict() if len(pairs) else {}
        k = wd.names.index(ch)
        touching = set(np.unique(L[wd.hole])) - {0}
        o_touching = set(np.unique(O[wd.hole])) - {0}
        L_in = np.where(wd.hole, L, 0)
        inside = {p.label: p for p in regionprops(L_in, intensity_image=stack)}
        Lb = np.where(inbox, L, 0)                      # measured in the analysis window (the model's features)
        in_box = {p.label: p for p in regionprops(Lb, intensity_image=stack)}
        full = {p.label: p for p in regionprops(L, intensity_image=stack)} if lev == "whole" else {}
        for lb in sorted(set(in_box) | set(full)):
            p = in_box.get(lb)
            outer = p is None
            touch = lb in touching
            m = L[p.slice] == lb if p is not None else None
            if p is not None:
                ol = O[p.slice][m]
                ol = ol[ol > 0]
                o_lab = int(np.bincount(ol).argmax()) if len(ol) else 0
            else:
                o_lab = 0
            # a piece cut off an object that touched the hole stays (as context) so it does not vanish from the view
            cut_piece = parts_of.get(o_lab, 1) > 1 and o_lab in o_touching
            if touch and inside[lb].area * wd.px_um ** 2 < cfg.min_inside_um2:
                if not cut_piece:
                    continue
                touch = False
            if not outer and not touch and not cut_piece and (
                    lev == "hole" or not context[int(p.centroid[0]), int(p.centroid[1])]):
                if lev == "hole":
                    continue
                outer = True                            # a cell of the area beyond the context disc
            if outer:
                p = full[lb]                            # shown and picked by its whole outline
                m = L[p.slice] == lb
                ol = O[p.slice][m]
                ol = ol[ol > 0]
                o_lab = int(np.bincount(ol).argmax()) if len(ol) else 0
            r0, c0, r1, c1 = p.bbox
            b = (bs[0].start, bs[1].start, bs[0].stop, bs[1].stop) if not outer else (0, 0, S[0], S[1])
            per = p.perimeter
            o = dict(plate_id=wd.plate_id, well_id=wd.well_id, oid=f"{lev[0]}{ch}_{lb}", level=lev, ch=ch,
                     lab=int(lb), pool=touch, outer=outer, full_px=int(p.area),
                     full_mean=float(p.intensity_mean[k]), full_max=float(p.intensity_max[k]),
                     cy=float(p.centroid[0]), cx=float(p.centroid[1]),
                     d_full=float(np.hypot(p.centroid[0] - hr, p.centroid[1] - hc) / R),
                     edge=bool(r0 == b[0] or c0 == b[1] or r1 == b[2] or c1 == b[3]),
                     circ=float(min(4 * np.pi * p.area / per ** 2, 1.0)) if per > 0 else 1.0,
                     ecc=float(p.eccentricity), solidity=float(p.solidity),
                     **{f"full_{c2}": float(p.intensity_mean[j]) for j, c2 in enumerate(wd.names)})
            mm = p.image
            o["n_peaks"] = int(len(set(np.unique(peaks[ch][p.slice][mm])) - {0}))
            o["origin"] = f"{lev[0]}{ch}_o{o_lab}"
            o["n_parts"] = int(parts_of.get(o_lab, 1))          # pieces of the original object (incl. outside)
            pr = proposals[(lev, ch)][p.slice][mm]
            o["n_proposed"] = int(len(set(np.unique(pr)) - {0}))       # pieces a split at dips would give
            if touch:
                q = inside[lb]
                o.update(in_px=int(q.area), share=q.area / p.area, fill=q.area / hole_px,
                         in_mean=float(q.intensity_mean[k]), in_max=float(q.intensity_max[k]),
                         iy=float(q.centroid[0]), ix=float(q.centroid[1]),
                         d_in=float(np.hypot(q.centroid[0] - hr, q.centroid[1] - hc) / R),
                         **{f"in_{c2}": float(q.intensity_mean[j]) for j, c2 in enumerate(wd.names)})
            objs.append(o)
    p = pd.DataFrame(objs)
    if len(p):
        p["parent"] = np.where(p.level == "whole", p.oid, None)
        for i in p.index[(p.level == "hole")]:
            ch = p.at[i, "ch"]
            W = maps.get(("whole", ch))
            if W is None:
                continue
            under = W[maps[("hole", ch)] == p.at[i, "lab"]]
            under = under[under > 0]
            if len(under):
                p.at[i, "parent"] = f"w{ch}_{np.bincount(under).argmax()}"
    row = dict(plate_id=wd.plate_id, well_id=wd.well_id, px_um=wd.px_um, hole_px=hole_px, R_px=R,
               off_r=wd.offset[0], off_c=wd.offset[1], win_h=S[0], win_w=S[1], hole_free_px=hbg[wd.names[0]][2],
               notes="; ".join(dict.fromkeys(wd.notes)))
    for ch in wd.names:
        row.update({f"bg_mean_{ch}": wd.bg[ch][0], f"bg_sd_{ch}": wd.bg[ch][1], f"hbg_med_{ch}": hbg[ch][0],
                    f"hbg_sd_{ch}": hbg[ch][1], f"thr_whole_{ch}": wd.thr.get(("whole", ch)),
                    f"thr_hole_{ch}": wd.thr.get(("hole", ch))})
    return p, maps, row
