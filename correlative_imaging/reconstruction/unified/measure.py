"""Step 3, the cell in each hole: grow it and measure it (the method validated on the PNC runs, made general).

Inputs per well: the hole and the particle maps Detect cached (<results>/cache), the processed channels of the hole crop
and the pipeline's ChannelRelation residual (<well>_roi_hole_crop_channels.tif / _residual.tif), and the band background
and noise the residual was computed with (run database). Nothing is written next to the run.

  seeds      per-hole particles of any colour channel lying >= 50 % inside the hole (judged within hole + 10 um), except
             the particles the shape model calls "not a cell" (plug, glow; shape.py): those are barriers
  barriers   objects lying mostly outside the hole (neighbours), and the pixel next to them
  grow       from the seeds while a pixel is above both the band background + 3 noise and half of the median of the
             region so far, never more than 2 noise uphill (so the glow of a neighbour is not climbed), at most hole +
             10 um; thin bridges (< ~2 um) cut. Every colour channel grows from its own seeds (growing every channel from
             all seeds, tried 2026-11-04, let a channel climb into a neighbour's glow: the red of a red neighbour became
             the cell's red; reverted. An outline that is only a speck is handled by the type step instead:
             smaller than the smallest cell body learnt from the particle review (labels.soma_cut) gets no type);
             the cell = the union
  signal     per channel: median signal-to-noise over the cell ((value - band background) / noise), the own signal
             = median of the ChannelRelation residual over the cell inside the hole (what is left after removing the
             copies of the other channels), saturated pixels left out
  hole sig   per channel: median over the cell inside the hole minus the hole's own background (hole pixels outside
             every object), / noise. Removes the aperture's own glow, which the band outside the hole does not see
  glow       per channel: the part of the cell near (< 3 um) a bright neighbour of that channel vs far (> 6 um) from it;
             near > 3 and far < half of near = the signal comes from the neighbour's glow
  neighbour  nearest object of each channel outside the hole: distance and brightness

New first pass (run(new=True), job["new"]): the cell outline is the regularized one (regular_outline; the grown one
above is kept as cell_old in wells/*.npz for comparison) and every hole gets the measures of hole_features (g_*), which
the learnt hole gate (call.py) uses. From the exploration of 2026-11-06 (figures/v2/explore/outline.py, features.py).
Remaining hand-set constants of the regularized outline (px units of the run): smoothing 2 px (cut) and 3 px (seeds),
inner hole = hole eroded by 2 px, seeds >= 3 px apart (at most 6), neighbour barriers >= 4 um2 and < 50 % in the hole,
background markers > 4 px outside the hole, a region kept when >= 50 % of it lies in the hole.
Measurement in the outline (new first pass): the pixels of ruled-out particles (your particle answers plug / halo,
debris, neighbour; the particle model's P < 0.2) are left out of every channel's measure (own signal, hole signal,
signal-to-noise, saturation), so a plug or a speck of debris inside the outline adds no colour (2026-11-06).
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


def regular_outline(img: np.ndarray, hole: np.ndarray, neigh: np.ndarray, bg: np.ndarray, noise: np.ndarray,
                    px: float) -> np.ndarray:
    """The regularized outline: the channels' signal-to-noise summed (clipped at 0, so a channel without signal adds
    only its noise), smoothed and log-scaled; seeds at the local maxima inside the hole on a coarser smoothing (a broad
    cell wins over a speck; several seeds, so a dim cell made of patches is one cell); a watershed from the seeds, the
    neighbours (objects mostly outside the hole, >= 4 um2, holding no seed) and the background; in each seed's basin the
    cell is cut at Otsu's threshold of that basin's own pixels; the regions mostly inside the hole, merged, filled,
    opened; the part with the most area in the hole. It does not decide whether a cell is there (an empty hole's glow
    gets an outline too): the hole gate does. Ruled-out particles (plug, debris, ...) are not handled here but left
    out of the measurement inside the outline (_one)."""
    from scipy import ndimage as ndi
    from skimage import measure as skm, morphology as skmo, segmentation
    from skimage.feature import peak_local_max
    from skimage.filters import threshold_otsu
    snr = np.maximum((img - bg[:, None, None]) / np.maximum(noise[:, None, None], 1e-6), 0)
    snr = np.nan_to_num(snr)
    s0 = snr.sum(axis=0)
    s = np.log1p(ndi.gaussian_filter(s0, 2.0))
    sb = ndi.gaussian_filter(s0, 3.0)
    inner = ndi.binary_erosion(hole, skmo.disk(2), border_value=1)
    if not inner.any():
        inner = hole
    pk = peak_local_max(np.where(inner, sb, 0), min_distance=3, num_peaks=6, exclude_border=False)
    pk = [tuple(p) for p in pk if inner[tuple(p)]]
    if not pk:
        pk = [np.unravel_index(np.argmax(np.where(inner, sb, -np.inf)), s.shape)]
    markers = np.zeros(s.shape, int)
    nl = skm.label(neigh)
    for rp in skm.regionprops(nl):
        m = nl == rp.label
        if rp.area * px * px >= 4 and (m & hole).sum() < 0.5 * m.sum() and not any(m[p] for p in pk):
            markers[m] = 2
    far = ~ndi.binary_dilation(hole, skmo.disk(4))
    if far.any():
        markers[far & (s < np.median(s[far])) & (markers == 0)] = 3
    for i, (y, x) in enumerate(pk):
        markers[max(y - 1, 0):y + 2, max(x - 1, 0):x + 2] = 10 + i
    ws = segmentation.watershed(-s, markers)
    cell = np.zeros(hole.shape, bool)
    for i in range(len(pk)):
        basin = ws == 10 + i
        if not basin.any():
            continue
        ctx = ndi.binary_dilation(basin, skmo.disk(4))
        try:
            cut = threshold_otsu(s[ctx])
        except ValueError:
            continue
        r = basin & (s >= cut)
        if r.sum() and (r & hole).sum() >= 0.5 * r.sum():
            cell |= r
    d1 = skmo.disk(1)                                    # opening as skimage did it (erosion with the border True)
    cell = ndi.binary_dilation(ndi.binary_erosion(ndi.binary_fill_holes(cell), d1, border_value=1), d1)
    lab = skm.label(cell)
    if lab.max() > 1:
        best = max(range(1, lab.max() + 1), key=lambda k: ((lab == k) & hole).sum())
        cell = lab == best
    return cell


def hole_features(img: np.ndarray, hole: np.ndarray, cell: np.ndarray, neigh: np.ndarray, bg: np.ndarray,
                  noise: np.ndarray, px: float, names: list) -> dict:
    """Measures of a hole for the learnt hole gate (g_*): outline shape, per channel the signal and texture in the
    hole, the outline and the ring around the hole, and the context (neighbour / barrier specks)."""
    from scipy import ndimage as ndi
    from skimage import measure as skm, morphology as skmo
    snr = (img - bg[:, None, None]) / np.maximum(noise[:, None, None], 1e-6)
    f = {}
    ring = ndi.binary_dilation(hole, skmo.disk(max(1, int(round(4 / px))))) & ~hole
    a_um = px * px
    f["cell_frac_hole"] = (cell & hole).sum() / max(hole.sum(), 1)
    if cell.any():
        lab = skm.label(cell)
        f["cell_parts"] = lab.max()
        rp = max(skm.regionprops(lab), key=lambda r: r.area)
        f["solidity"], f["eccentricity"] = rp.solidity, rp.eccentricity
        f["jagged"] = skm.perimeter(cell) ** 2 / (4 * np.pi * cell.sum())
    else:
        f.update(cell_parts=0, solidity=np.nan, eccentricity=np.nan, jagged=np.nan)
    nb = neigh & ~cell
    f["neigh_frac_crop"] = nb.mean()
    f["neigh_frac_ring"] = (nb & ring).sum() / max(ring.sum(), 1)
    lab = skm.label(nb)
    sizes = np.bincount(lab.ravel())[1:] * a_um if lab.max() else np.array([])
    f["neigh_n_specks"] = int((sizes < 5).sum())
    f["neigh_n_big"] = int((sizes >= 20).sum())
    if cell.any():
        f["outline_touch_neigh"] = (ndi.binary_dilation(cell) & nb).sum() / max(skm.perimeter(cell), 1)
        dist = ndi.distance_transform_edt(~nb) * px if nb.any() else np.full(cell.shape, 99.0)
        f["neigh_min_dist_um"] = float(dist[cell].min())
    else:
        f["outline_touch_neigh"] = f["neigh_min_dist_um"] = np.nan
    rest = ~hole & ~neigh
    for k, c in enumerate(names):
        x = snr[k]
        f[f"{c}_hole_med"], f[f"{c}_hole_p90"] = np.median(x[hole]), np.percentile(x[hole], 90)
        f[f"{c}_ring_med"] = np.median(x[ring]) if ring.any() else np.nan
        f[f"{c}_cell_med"] = np.median(x[cell]) if cell.any() else np.nan
        f[f"{c}_cell_p90"] = np.percentile(x[cell], 90) if cell.any() else np.nan
        f[f"{c}_cell_frac3"] = float(np.mean(x[cell] > 3)) if cell.any() else np.nan
        lg = ndi.gaussian_laplace(x, 1.0)
        f[f"{c}_tex_hole"] = np.std(lg[hole])
        f[f"{c}_tex_ring"] = np.std(lg[ring]) if ring.any() else np.nan
        f[f"{c}_bg_noise_crop"] = np.median(np.abs(x[rest])) if rest.any() else np.nan
    return {f"g_{k}": float(v) for k, v in f.items()}


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
            st = grow(img[idx[c]], seeds[c] & allowed, bg[c] + K_NOISE * noise[c], allowed, UPHILL * noise[c])
            regions[c] = clean(st, seeds[c] & allowed, 2) if st.any() else st
        for c, sd in mark_seed.items():
            st = grow(img[idx[c]], sd & allowed, bg[c] + K_NOISE * noise[c], allowed, UPHILL * noise[c])
            regions[c] = clean(st, sd & allowed, 2) if st.any() else st
        cell = np.any(list(regions.values()), axis=0) if regions else np.zeros(hole.shape, bool)
        ys, xs = np.nonzero(hole); pad = um(8)
        sl = (slice(max(ys.min() - pad, 0), ys.max() + pad), slice(max(xs.min() - pad, 0), xs.max() + pad))
        bga, nsa = np.array([bg[c] for c in names], float), np.array([noise[c] for c in names], float)
        cell_old = cell
        if job.get("new"):                         # the regularized outline, on the hole crop (as in the exploration)
            k_ = [idx[c] for c in allc]
            reg = regular_outline(img[k_][:, sl[0], sl[1]], hole[sl], neigh[sl], bga[k_], nsa[k_], px)
            cell = np.zeros(hole.shape, bool)
            cell[sl] = reg
            out.update(hole_features(img[:, sl[0], sl[1]], hole[sl], cell[sl], neigh[sl], bga, nsa, px, names))
            out["cell_um2_old"] = cell_old.sum() * px ** 2
        out.update(hole_um2=hole.sum() * px ** 2, cell_um2=cell.sum() * px ** 2,
                   cell_in_hole=float((cell & hole).sum() / max(cell.sum(), 1)),
                   hole_cover=float((cell & hole).sum() / max(hole.sum(), 1)), sat_px_cell=int((cell & sat).sum()),
                   n_seed_px=int(seeds_any.sum()), px_um=px)
        # new first pass: the ruled-out particles (plug, debris, neighbour; notcell) are not measured, in any channel
        excl = notcell if job.get("new") else np.zeros(hole.shape, bool)
        use = cell & ~sat & ~excl
        inh = cell & hole & ~sat & ~excl
        out["excluded_px"] = int((cell & excl).sum())
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
        np.savez_compressed(Path(job["wells_dir"]) / f"{job['plate_id']}_{job['well_id']}.npz",
                            img=img[:, sl[0], sl[1]], cell=cell[sl], hole=hole[sl], neigh=neigh[sl],
                            cell_old=cell_old[sl],
                            names=np.array(names), bg=np.array([bg[c] for c in names]),
                            noise=np.array([noise[c] for c in names]), px=px)
    except Exception as ex:                                                  # noqa: BLE001
        out["error"] = f"{type(ex).__name__}: {ex}"
    return out


def measure(T: dict, W: pd.DataFrame, e: dict, results: Path, out: Path, path_map: dict, workers: int = 8,
            log=print, drop: dict | None = None, new: bool = False) -> pd.DataFrame:
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
                         drop=(drop or {}).get((r.plate_id, r.well_id), {}), new=new))
    log(f"  measuring {len(jobs)} holes")
    rows = []
    workers = max(1, min(int(workers), 61 if os.name == "nt" else 256, len(jobs) or 1))   # Windows allows at most 61
    with ProcessPoolExecutor(workers) as ex:
        for i, o in enumerate(ex.map(_one, jobs, chunksize=4)):
            rows.append(o)
            if (i + 1) % 500 == 0:
                log(f"  {i + 1}/{len(jobs)}")
    return pd.DataFrame(rows)
