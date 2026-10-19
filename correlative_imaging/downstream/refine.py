"""Refinement: the full outline of the patched cell, with neighbours claiming their own pixels.

The patched cell is dim (0.2-0.35x a field cell) and the thresholds cut it into a small core. Here every candidate
cell grows from its seed, while the neighbours grow at the same time from theirs:

  seeds      candidate cells: their particles' pieces inside the hole (already filtered: P(cell), not neighbour,
             not bleed-through). Competitors: neighbour particles (whole extent) and every stringent field object
             in the crop that does not belong to a candidate.
  signal     S = max over the body channels of the lightly smoothed image, in units of the HOLE's own background
             (median and robust SD of the hole pixels outside every particle). The hole is brighter and noisier
             than the rest of the frame (aperture), so growth judged against the whole-image background would
             flood it. (Scoring keeps the whole-image background: there the in-hole one lets halo through.)
  growth     pixels with S >= floor are shared out between all seeds by flooding from the seeds over -S (each
             pixel goes to the seed it is best connected to through bright pixels). A candidate then keeps only
             pixels >= `refine_frac` x the median S of its own seed (a cut relative to the seed, so a dim patched
             cell grows by its own standard), within `refine_margin_um` of the hole and connected to its seed.
             If it is still larger than `refine_max_area_x_hole` x the hole, the relative cut is raised in steps.

This is not the watershed splitting that is switched off in the pipeline: nothing is split by shape; the seeds
are particles that were already classified, and only the dim pixels between them are assigned.

Detection and tier stay on the seed (the P(cell) model was trained on seeds); the colour is measured on the
grown body, every channel over the same footprint.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.segmentation import watershed

from .config import Config


def refine_well(wd, particles: pd.DataFrame, cells: list, maps: dict, bg: dict, cfg: Config):
    """wd: WellData; particles: scored particles of this well (oid, seg, ch, role, foot, touch);
    cells: from cells_of_well (kind, members); maps: label images from extract_particles;
    bg: {ch: (mean, sd)}. Returns (list of refined-cell dicts with 'mask', competitor label image)."""
    H, W = wd.hole.shape
    names = wd.names
    body = [c for c in cfg.body if c in names]
    # hole background: hole pixels outside every particle of any channel/threshold
    occupied = np.zeros((H, W), bool)
    for L in maps.values():
        occupied |= L > 0
    free = wd.hole & ~occupied
    z = []
    for c in body:
        k = names.index(c)
        img = ndi.gaussian_filter(wd.chans[k], cfg.refine_smooth_px) if cfg.refine_smooth_px else wd.chans[k]
        if free.sum() >= 30:
            v = img[free]
            m = float(np.median(v))
            s = float(1.4826 * np.median(np.abs(v - m)))
        else:                                               # cell fills the hole: whole-image background
            m, s = bg[c]
        z.append((img - m) / (s if s and s > 0 else 1.0))
    S = np.max(z, axis=0)

    cand = [c for c in cells if c["kind"] == "candidate"]
    if not cand:
        return [], None
    markers = np.zeros((H, W), np.int32)
    seed_of = {}
    for i, c in enumerate(cand, start=1):
        seed = np.zeros(H * W, bool)
        seed[c["foot"]] = True
        seed = seed.reshape(H, W)
        markers[seed] = i
        seed_of[i] = seed
    # competitors: neighbour particles (whole extent) + stringent field objects not belonging to a candidate
    cand_oids = {o for c in cand for o in c["members"]}
    nxt = len(cand) + 1
    comp = particles[(particles.role == "neighbour") | ((particles.seg == "whole") & ~particles.oid.isin(cand_oids))]
    any_seed = markers > 0
    for r in comp.itertuples():
        L = maps.get((r.seg, r.ch))
        if L is None:
            continue
        m = L == r.label
        if r.role != "neighbour" and (m & any_seed).any():   # the stringent object of a candidate itself
            continue
        m &= markers == 0
        if m.any():
            markers[m] = nxt
            nxt += 1
    eligible = (S >= cfg.refine_floor_snr) | (markers > 0)
    flood = watershed(-S, markers, mask=eligible)

    margin = int(round(cfg.refine_margin_um / wd.px_um))
    reach = ndi.binary_dilation(wd.hole, iterations=margin) if margin > 0 else wd.hole
    cap = int(cfg.refine_max_area_x_hole * wd.hole.sum())
    out = []
    for i, c in enumerate(cand, start=1):
        seed = seed_of[i]
        med = float(np.median(S[seed]))
        frac = cfg.refine_frac
        while True:
            m = ((flood == i) & reach & (S >= max(frac * med, cfg.refine_floor_snr))) | seed
            lab, _ = ndi.label(m)
            keep = np.unique(lab[seed])
            m = np.isin(lab, keep[keep > 0])
            if m.sum() <= cap or frac >= 1:
                break
            frac = min(frac + .1, 1.0)                     # too big: raise the relative cut
        border = ndi.binary_dilation(m) & ~m
        touch_nb = bool(((flood > len(cand)) & border).any())
        meas = {f"body_mean_{c2}": float(wd.chans[names.index(c2)][m].mean()) for c2 in names}
        out.append(dict(cell_id=c["cell_id"], mask=m, seed_px=int(seed.sum()), body_px=int(m.sum()),
                        body_um2=float(m.sum() * wd.px_um ** 2), grow_x=float(m.sum() / max(seed.sum(), 1)),
                        body_in_hole=float((m & wd.hole).sum() / max(m.sum(), 1)),
                        touches_neighbour=touch_nb, seed_median_snr=med, cut_frac=frac, **meas))
    return out, flood
