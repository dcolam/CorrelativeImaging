"""Read a finished CorrelativeImaging run, read-only, into a few tables (cached as parquet in <out>/cache).

Per plate database:
  objects   every particle the pipeline segmented: whole image (the surrounding cells) and inside the hole; per object
            its shape, its own channel's median, the local ring background, and every other channel's median and
            ring inside the same object (colocalization table), saturation, overlap with the other channels' masks
  images    image -> plate, well (from the image metadata), pixel size, barcode in the file name
  levels    per image and channel: image background and noise (whole image outside all objects), the band background
            and noise around the hole (ChannelRelation), whole-image and hole thresholds
  holes     the hole of each image: area, centre, crop origin and crop files
Plate id = the barcode in the plate folder name (folders are trusted; file names can be wrong - a mismatch is
reported by the checks).
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from .._discovery import discover_runs

BARCODE = re.compile(r"(\d{2}[A-Z]\d{4,5})")
TABLES = ("objects", "images", "levels", "holes")


def plate_id_of(folder: str) -> str:
    m = BARCODE.search(folder)
    return m.group(1) if m else folder


def pad_well(w: str) -> str:
    m = re.match(r"([A-Z]+)0*(\d+)$", str(w))
    return f"{m[1]}{int(m[2]):02d}" if m else str(w)


def _ro(db):
    return sqlite3.connect(f"file:{db}?mode=ro&immutable=1", uri=True)


def _plate(pr) -> dict:
    con = _ro(pr.db_path)
    plate = plate_id_of(pr.plate)
    im = pd.read_sql("select id as image_id, filename, metadata, pixel_size_um from images", con)
    im["well_id"] = [pad_well(json.loads(m or "{}").get("well_id", "")) for m in im.metadata]
    im["well_raw"] = [json.loads(m or "{}").get("well_id", "") for m in im.metadata]
    im["file_barcode"] = im.filename.str.extract(r"(\d{2}[A-Z]\d{4,5})")[0]
    im = im.drop(columns="metadata").assign(plate_id=plate, plate_dir=str(pr.db_path.parent),
                                             diag=str(pr.diagnostics_dir) if pr.diagnostics_dir else "")
    P = pd.read_sql("select image_id, channel, roi_mask, label, area_um2, area_px, circularity, solidity, eccentricity, "
                    "median_intensity, ring_median_intensity, ring_px, centroid_row, centroid_col, cut_at_edge, "
                    "saturated_px, share_inside from particle_measurements "
                    "where roi_mask in ('whole_image', 'roi_hole')", con)
    C = pd.read_sql("select image_id, roi_mask, primary_channel, primary_label, secondary_channel, "
                    "secondary_median_intensity, secondary_ring_median_intensity, mask_overlap_fraction "
                    "from colocalization_per_particle where roi_mask in ('whole_image', 'roi_hole')", con)
    key = ["image_id", "roi_mask", "channel", "label"]
    C = C.rename(columns={"primary_channel": "channel", "primary_label": "label"})
    for ch in sorted(C.secondary_channel.unique()):
        s = C[C.secondary_channel == ch][key + ["secondary_median_intensity", "secondary_ring_median_intensity",
                                                "mask_overlap_fraction"]]
        s = s.rename(columns={"secondary_median_intensity": f"med_{ch}", "secondary_ring_median_intensity": f"ring_{ch}",
                              "mask_overlap_fraction": f"ov_{ch}"})
        P = P.merge(s, on=key, how="left")
    for ch in P.channel.unique():
        own = P.channel == ch
        P.loc[own, f"med_{ch}"] = P.loc[own, "median_intensity"]
        P.loc[own, f"ring_{ch}"] = P.loc[own, "ring_median_intensity"]
        P.loc[own, f"ov_{ch}"] = 1.0
    P["plate_id"] = plate
    I = pd.read_sql("select image_id, channel, roi_mask, median_intensity, robust_sd_intensity "
                    "from intensity_measurements where roi_mask = 'whole_image:outside_particles'", con)
    I = I.rename(columns={"median_intensity": "img_bg", "robust_sd_intensity": "img_noise"}).drop(columns="roi_mask")
    tabs = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
    if "channel_relations" in tabs:
        R = pd.read_sql("select image_id, channel, background as band_bg, noise as band_noise, background_source "
                        "from channel_relations where roi_mask = 'roi_hole' and object_label = 0", con)
        R = R.drop_duplicates(["image_id", "channel"])
        I = I.merge(R, on=["image_id", "channel"], how="left")
    L = pd.read_sql("select image_id, channel, roi_mask, threshold_value from channel_levels", con)
    L = L.pivot_table(index=["image_id", "channel"], columns="roi_mask", values="threshold_value").reset_index()
    L = L.rename(columns={"whole_image": "thr_whole", "roi_hole": "thr_hole"})
    I = I.merge(L, on=["image_id", "channel"], how="left").assign(plate_id=plate)
    S = pd.read_sql("select image_id, area_um2 as hole_um2, centroid_row as hole_row, centroid_col as hole_col, "
                    "crop_row0, crop_col0, crop_files from selections where roi_mask = 'roi_hole'", con)
    S = S.drop_duplicates("image_id").assign(plate_id=plate)
    con.close()
    return dict(objects=P, images=im, levels=I, holes=S)


def _local_dirs(T: dict) -> dict:
    """Folder columns as paths of this system (the cache may have been written on Windows or in WSL)."""
    for t in T.values():
        for c in ("diag", "plate_dir"):
            if c in t:
                foreign = (lambda v: v.startswith("/mnt/")) if os.name == "nt" else \
                    (lambda v: re.match(r"^[A-Za-z]:\\", v) is not None)
                t[c] = [str(local_path(v, {})) if foreign(v) else v for v in t[c].astype(str)]
    return T


def load_run(root, run_tag, cache: Path, log=print) -> dict:
    cache = Path(cache)
    if all((cache / f"{t}.parquet").exists() for t in TABLES):
        return _local_dirs({t: pd.read_parquet(cache / f"{t}.parquet") for t in TABLES})
    runs = {g.run_tag: g for g in discover_runs(root)}
    if run_tag not in runs:
        raise KeyError(f"run {run_tag} not found under {root}")
    parts = {t: [] for t in TABLES}
    for pr in runs[run_tag].plates:
        d = _plate(pr)
        for t in TABLES:
            parts[t].append(d[t])
        log(f"  read {pr.plate}")
    cache.mkdir(parents=True, exist_ok=True)
    out = {}
    for t in TABLES:
        out[t] = pd.concat(parts[t], ignore_index=True)
        out[t].to_parquet(cache / f"{t}.parquet", index=False)
    return out


def local_path(p: str, path_map: dict) -> Path:
    from pathlib import PureWindowsPath
    if Path(p).exists():
        return Path(p)
    for win, loc in (path_map or {}).items():
        if p.startswith(win):
            return Path(loc) / PureWindowsPath(p[len(win):]).as_posix()
    if re.match(r"^[A-Za-z]:\\", p) and os.name != "nt":     # Windows path read from WSL
        return Path("/mnt") / p[0].lower() / PureWindowsPath(p[3:]).as_posix()
    m = re.match(r"^/mnt/([a-z])/(.*)$", p)
    if m and os.name == "nt":                                  # WSL path read on Windows
        return Path(f"{m.group(1).upper()}:/{m.group(2)}")
    return Path(p)


def signals(T: dict, colour: list, marker: list) -> pd.DataFrame:
    """Objects with, per channel, the signal above the local background in noise units ("contrast"):
    (median inside - ring around; the image background where the ring is too crowded) / image noise."""
    P = T["objects"].merge(T["images"][["plate_id", "image_id", "well_id", "plate_dir", "diag", "well_raw"]],
                           on=["plate_id", "image_id"])
    lv = T["levels"].set_index(["plate_id", "image_id", "channel"])
    for ch in colour + marker:
        k = list(zip(P.plate_id, P.image_id, [ch] * len(P)))
        bg = lv.img_bg.reindex(k).to_numpy()
        nz = lv.img_noise.reindex(k).to_numpy()
        ring = P[f"ring_{ch}"].to_numpy(float) if f"ring_{ch}" in P else np.full(len(P), np.nan)
        P[f"y_{ch}"] = (P[f"med_{ch}"].to_numpy(float) - np.where(np.isfinite(ring), ring, bg)) / nz
    P["px_um"] = np.sqrt(P.area_um2 / P.area_px)
    return P
