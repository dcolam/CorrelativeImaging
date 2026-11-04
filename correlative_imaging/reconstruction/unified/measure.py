"""Step 3, the cell in each hole: grow it and measure it (the method validated on the PNC runs, made general).

Inputs per well: the hole and the particle maps Detect cached (<results>/cache), the processed channels of the hole crop
and the pipeline's ChannelRelation residual (<well>_roi_hole_crop_channels.tif / _residual.tif), and the band background
and noise the residual was computed with (run database). Nothing is written next to the run.

  seeds      per-hole particles of any colour channel lying >= 50 % inside the hole (judged within hole + 10 um), except
             the particles the shape model calls "not a cell" (plug, glow; shape.py): those are barriers
  barriers   objects lying mostly outside the hole (neighbours), and the pixel next to them
  grow       from the seeds while a pixel is above both the band background + 3 noise and half of the median of the
             region so far, never more than 2 noise uphill (so the glow of a neighbour is not climbed), at most hole +
             10 um; thin bridges (< ~2 um) cut. Every colour channel grows from the seeds of ALL colour channels (a cell
             dim in one colour has no particle of that colour; grown only from its own seeds, a red-weak cell with a
             green dot got an outline that was only the dot); the cell = the union
  signal     per channel: median signal-to-noise over the cell ((value - band background) / noise), the own signal
             = median of the ChannelRelation residual over the cell inside the hole (what is left after removing the
             copies of the other channels), saturated pixels left out
  hole sig   per channel: median over the cell inside the hole minus the hole's own background (hole pixels outside
             every object), / noise. Removes the aperture's own glow, which the band outside the hole does not see
  glow       per channel: the part of the cell near (< 3 um) a bright neighbour of that channel vs far (> 6 um) from it;
             near > 3 and far < half of near = the signal comes from the neighbour's glow
  neighbour  nearest object of each channel outside the hole: distance and brightness
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation, binary_opening, distance_transform_edt, generate_binary_structure, \
    label as nd_label
from skimage.morphology import disk

CROSS = generate_binary_structure(2, 1)
K_NOISE, F_MEDIAN, UPHILL = 3.0, 0.5, 2.0


def neighbour_max(plane, region):
    v = np.where(region, plane, -np.inf)
    out = np.full(plane.shape, -np.inf)
    out[1:] = np.maximum(out[1:], v[:-1]); out[:-1] = np.maximum(out[:-1], v[1:])
    out[:, 1:] = np.maximum(out[:, 1:], v[:, :-1]); out[:, :-1] = np.maximum(out[:, :-1], v[:, 1:])
    return out


def grow(plane, seeds, bg_stop, allowed, uphill):
    region = seeds.copy()
    while True:
        stop = max(bg_stop, F_MEDIAN * float(np.median(plane[region]))) if region.any() else bg_stop
        new = binary_dilation(region, CROSS) & ~region & allowed & (plane >= stop) & \
            (plane <= neighbour_max(plane, region) + uphill)
        if not new.any():
            return region
        region |= new


def clean(region, seeds, r):
    if not region.any():
        return region
    opened = binary_opening(region, disk(r)) | (seeds & region)
    lab, _ = nd_label(opened)
    keep = np.unique(lab[seeds & opened]); keep = keep[keep > 0]
    return np.isin(lab, keep)


def cache_prefix(plates: pd.DataFrame, cache_dir: Path) -> dict:
    """plate_id -> the name Detect used in its cache (the barcode, or the plate folder's name at Detect time)."""
    pre = {p.stem.rsplit("_", 1)[0] for p in Path(cache_dir).glob("*.npz")}
    out = {}
    for r in plates.itertuples():
        folder = Path(r.plate_dir).name
        for cand in (r.plate_id, folder):
            if cand in pre:
                out[r.plate_id] = cand
                break
        else:
            stem = folder.rsplit("_", 1)[0]
            hit = [p for p in pre if p.rsplit("_", 1)[0] == stem or folder.startswith(p)]
            out[r.plate_id] = hit[0] if len(hit) == 1 else None
    return out


def _one(job):
    import tifffile
    out = dict(plate_id=job["plate_id"], well_id=job["well_id"], image_id=job["image_id"], error="")
    try:
        cz = np.load(job["cache"])
        hole = cz["hole"].astype(bool); H, W = hole.shape
        px = float(cz["px_um"]); names = [str(n) for n in cz["names"]]
        colour = [c for c in names if c in job["colour"]]
        allc = [c for c in names if c in job["colour"] + job["marker"]]
        um = lambda d: max(int(round(d / px)), 1)
        diag = Path(job["diag"])
        with tifffile.TiffFile(diag / f"{job['well_raw']}_roi_hole_crop_channels.tif") as tf:
            img = np.stack([tf.pages[i].asarray() for i in range(len(names))]).astype(np.float32)[:, :H, :W]
        with tifffile.TiffFile(diag / f"{job['well_raw']}_roi_hole_crop_residual.tif") as tf:
            res = np.stack([tf.pages[i].asarray() for i in range(len(names))]).astype(np.float32)[:, :H, :W]
        if img.shape[1:] != hole.shape:            # hole at the frame border: whole frame + residual crop at its offset
            with tifffile.TiffFile(diag / f"{job['well_raw']}_whole_channels.tif") as tf:
                img = np.stack([tf.pages[i].asarray() for i in range(len(names))]).astype(np.float32)[:, :H, :W]
            ys, xs = np.nonzero(hole)
            ext = lambda lo, hi, n, m: (max(lo - m, 0), min(hi + 1 + m, n))
            m = next(m for m in range(0, 1000) if all(b - a == L for (a, b), L in
                     ((ext(ys.min(), ys.max(), H, m), res.shape[1]), (ext(xs.min(), xs.max(), W, m), res.shape[2]))))
            r0, c0 = ext(ys.min(), ys.max(), H, m)[0], ext(xs.min(), xs.max(), W, m)[0]
            full = np.zeros((len(names), H, W), np.float32)
            full[:, r0:r0 + res.shape[1], c0:c0 + res.shape[2]] = res
            res = full
        idx = {c: names.index(c) for c in names}
        bg = {c: job["bg"].get(c, (np.nan, np.nan))[0] for c in names}
        noise = {c: job["bg"].get(c, (np.nan, np.nan))[1] for c in names}
        sat = hole & np.all(res[[idx[c] for c in allc]] == 0, axis=0)
        whole = {c: cz[f"map|whole|{c}"] > 0 for c in names if f"map|whole|{c}" in cz.files}
        own = {c: cz[f"map|hole|{c}"] > 0 for c in names if f"map|hole|{c}" in cz.files}
        safety = binary_dilation(hole, iterations=um(10))
        # particles the shape model calls "not a cell" (plug, glow): not seeds, but barriers
        # per channel: a dropped particle removes only its own channel's seed (a cell bright in all colours with one
        # dropped colour particle kept no seed in any colour and lost its outline)
        notcell = np.zeros(hole.shape, bool)
        notc = {}
        for c, labs in (job.get("drop") or {}).items():
            notc[c] = np.zeros(hole.shape, bool)
            for lev in ("hole", "whole"):
                key = f"map|{lev}|{c}"
                if key in cz.files and labs.get(lev):
                    notc[c] |= np.isin(cz[key], labs[lev])
            notcell |= notc[c]
        own = {c: m_ & ~notc.get(c, np.zeros(hole.shape, bool)) for c, m_ in own.items()}
        seeds, near_obj = {}, {}
        for c in colour:
            lab, n = nd_label(own.get(c, np.zeros(hole.shape, bool)) & safety)
            tot = np.bincount(lab.ravel(), minlength=n + 1); ins = np.bincount(lab[hole], minlength=n + 1)
            seeds[c] = np.isin(lab, np.nonzero((ins / np.maximum(tot, 1) >= 0.5) & (np.arange(n + 1) > 0))[0])
        seeds_any = np.any([seeds[c] for c in colour], axis=0) if colour else np.zeros(hole.shape, bool)
        # a marker helps only where no colour found anything: its particles >= 50 % inside the hole seed the outline
        # (grown in the marker channel); it never decides the colour or the type
        mark_seed = {}
        if not seeds_any.any():
            for c in [m for m in allc if m not in colour]:
                lab, n = nd_label(own.get(c, np.zeros(hole.shape, bool)) & safety)
                tot = np.bincount(lab.ravel(), minlength=n + 1); ins = np.bincount(lab[hole], minlength=n + 1)
                mark_seed[c] = np.isin(lab, np.nonzero((ins / np.maximum(tot, 1) >= 0.5) & (np.arange(n + 1) > 0))[0])
            if mark_seed:
                seeds_any = np.any(list(mark_seed.values()), axis=0)
        out["outline_from_marker"] = bool(seeds_any.any() and mark_seed)
        neigh = np.zeros(hole.shape, bool)
        for c in allc:
            near_obj[c] = np.zeros(hole.shape, bool)
            for mk in (own.get(c), whole.get(c)):
                if mk is None:
                    continue
                lab, n = nd_label(mk & safety)
                tot = np.bincount(lab.ravel(), minlength=n + 1); ins = np.bincount(lab[hole], minlength=n + 1)
                has_seed = np.nonzero(np.bincount(lab[seeds_any], minlength=n + 1) > 0)[0]
                other = np.nonzero((ins / np.maximum(tot, 1) < 0.5) & (np.arange(n + 1) > 0) & (tot > 0))[0]
                o = np.isin(lab, other) & ~(np.isin(lab, has_seed[has_seed > 0]) & hole)
                if c in colour:
                    neigh |= o
                near_obj[c] |= o
        neigh |= notcell
        allowed = safety & ~(binary_dilation(neigh, CROSS) & ~seeds_any)
        regions = {}
        for c in colour:
            # from every colour's seeds (2026-11-04): the seed pixels stay in the region even where this channel is
            # dark, so a channel without signal adds nothing but the seeds themselves to the union
            st = grow(img[idx[c]], seeds_any & allowed, bg[c] + K_NOISE * noise[c], allowed, UPHILL * noise[c])
            regions[c] = clean(st, seeds_any & allowed, 2) if st.any() else st
        for c, sd in mark_seed.items():
            st = grow(img[idx[c]], sd & allowed, bg[c] + K_NOISE * noise[c], allowed, UPHILL * noise[c])
            regions[c] = clean(st, sd & allowed, 2) if st.any() else st
        cell = np.any(list(regions.values()), axis=0) if regions else np.zeros(hole.shape, bool)
        out.update(hole_um2=hole.sum() * px ** 2, cell_um2=cell.sum() * px ** 2,
                   cell_in_hole=float((cell & hole).sum() / max(cell.sum(), 1)),
                   hole_cover=float((cell & hole).sum() / max(hole.sum(), 1)), sat_px_cell=int((cell & sat).sum()),
                   n_seed_px=int(seeds_any.sum()), px_um=px)
        use = cell & ~sat
        inh = cell & hole & ~sat
        # the hole's own background: hole pixels outside every object of every channel and outside the cell (the
        # aperture can glow; the band outside the hole does not see that). Too few free pixels: the band
        anyobj = np.zeros(hole.shape, bool)
        for mk_ in list(whole.values()) + list(own.values()):
            anyobj |= mk_
        free = hole & ~binary_dilation(anyobj | cell, iterations=um(1)) & ~sat
        out["hole_free_px"] = int(free.sum())
        for c in allc:
            k = idx[c]
            snr = (img[k] - bg[c]) / noise[c]
            hb = float(np.median(img[k][free])) if free.sum() >= 30 else bg[c]
            out[f"hbg_{c}"] = hb
            out[f"hsig_{c}"] = float(np.median((img[k][inh] - hb) / noise[c])) if inh.sum() >= 5 else np.nan
            out[f"snr_{c}"] = float(np.median(snr[use])) if use.any() else np.nan
            out[f"own_{c}"] = float(np.median(res[k][inh])) if inh.sum() >= 5 else np.nan
            out[f"raw_{c}"] = out[f"snr_{c}"] * noise[c] if np.isfinite(out[f"snr_{c}"]) else np.nan
            d = distance_transform_edt(~near_obj[c]) * px
            near, far = use & (d < 3), use & (d > 6)
            out[f"near_{c}"] = float(np.median(snr[near])) if near.sum() >= 20 else np.nan
            out[f"far_{c}"] = float(np.median(snr[far])) if far.sum() >= 20 else np.nan
            out[f"glow_{c}"] = bool(np.isfinite(out[f"near_{c}"]) and np.isfinite(out[f"far_{c}"])
                                    and out[f"near_{c}"] > 3 and out[f"far_{c}"] < 0.5 * out[f"near_{c}"])
            nb = whole.get(c, np.zeros(hole.shape, bool)) & ~binary_dilation(hole, iterations=um(2))
            if nb.any() and cell.any():
                dist = distance_transform_edt(~cell) * px
                lab, _ = nd_label(nb)
                j = lab[nb][np.argmin(dist[nb])]
                out[f"nb_dist_um_{c}"] = float(dist[nb].min())
                out[f"nb_snr_{c}"] = float(np.median(snr[lab == j]))
            else:
                out[f"nb_dist_um_{c}"], out[f"nb_snr_{c}"] = np.nan, np.nan
            if c in colour:
                out[f"region_um2_{c}"] = regions[c].sum() * px ** 2
        ys, xs = np.nonzero(hole); pad = um(8)
        sl = (slice(max(ys.min() - pad, 0), ys.max() + pad), slice(max(xs.min() - pad, 0), xs.max() + pad))
        np.savez_compressed(Path(job["wells_dir"]) / f"{job['plate_id']}_{job['well_id']}.npz",
                            img=img[:, sl[0], sl[1]], cell=cell[sl], hole=hole[sl], neigh=neigh[sl],
                            names=np.array(names), bg=np.array([bg[c] for c in names]),
                            noise=np.array([noise[c] for c in names]), px=px)
    except Exception as ex:                                                  # noqa: BLE001
        out["error"] = f"{type(ex).__name__}: {ex}"
    return out


def measure(T: dict, W: pd.DataFrame, e: dict, results: Path, out: Path, path_map: dict, workers: int = 8,
            log=print, drop: dict | None = None) -> pd.DataFrame:
    from .data import local_path
    from .experiment import colour_channels, marker_channels
    wells_dir = Path(out) / "wells"
    wells_dir.mkdir(parents=True, exist_ok=True)
    pre = cache_prefix(T["images"][["plate_id", "plate_dir"]].drop_duplicates(), Path(results) / "cache")
    lv = T["levels"]
    bgd = {}
    for r in lv.itertuples():
        b = (r.band_bg, r.band_noise) if np.isfinite(getattr(r, "band_bg", np.nan)) else (r.img_bg, r.img_noise)
        bgd.setdefault((r.plate_id, r.image_id), {})[r.channel] = (float(b[0]), float(b[1]))
    jobs = []
    for r in T["images"].merge(W[["plate_id", "image_id", "call"]], on=["plate_id", "image_id"]).itertuples():
        if not r.call or pre.get(r.plate_id) is None:
            continue
        cf = Path(results) / "cache" / f"{pre[r.plate_id]}_{r.well_id}.npz"
        if not cf.exists():
            continue
        jobs.append(dict(plate_id=r.plate_id, well_id=r.well_id, image_id=r.image_id, well_raw=r.well_raw,
                         diag=str(local_path(r.diag, path_map)), cache=str(cf), bg=bgd.get((r.plate_id, r.image_id), {}),
                         colour=colour_channels(e), marker=marker_channels(e), wells_dir=str(wells_dir),
                         drop=(drop or {}).get((r.plate_id, r.well_id), {})))
    log(f"  measuring {len(jobs)} holes")
    rows = []
    workers = max(1, min(int(workers), 61 if os.name == "nt" else 256, len(jobs) or 1))   # Windows allows at most 61
    with ProcessPoolExecutor(workers) as ex:
        for i, o in enumerate(ex.map(_one, jobs, chunksize=4)):
            rows.append(o)
            if (i + 1) % 500 == 0:
                log(f"  {i + 1}/{len(jobs)}")
    return pd.DataFrame(rows)
