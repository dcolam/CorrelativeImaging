"""Where wells come from. Everything else works on :class:`WellData`, never on files.

RunSource       a finished CorrelativeImaging run (all plate databases sharing one run tag). Per well it reads, in
                order of preference:
                  1. ``diagnostics/<well>_whole_channels.tif`` (the last pre-threshold image of the whole frame, + BF):
                     thresholds and background can then be computed exactly as the pipeline does when the database
                     does not store them (older runs);
                  2. ``diagnostics/<well>_roi_hole_crop_channels.tif`` with its origin from the ``selections`` table,
                     thresholds from ``channel_levels`` and background from ``whole_image:outside_particles``.
                The hole is the ImageJ .roi rasterised exactly as pipeline/segment.py (int coordinates clipped to the
                frame, skimage.draw.polygon). Channels are matched by name (the TIF's ImageJ labels), never by position.
JsonpCropSource development adapter for viewer exports (``<plate>_<well>.js`` crops + a wells table with stored
                thresholds/backgrounds + the plate databases). Used to import the first hand-labelled particle set,
                whose coordinates refer to those crops.

Every source also gives the field cells of each well (whole-image particles from the database): the reference for
what a typical, unpatched cell looks like on that plate.
"""
from __future__ import annotations

import base64
import json
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd

from .config import Config


def _given(v) -> bool:
    """True for a real value (not None / NaN / empty): table columns of wells without a row come back as NaN."""
    if v is None:
        return False
    if isinstance(v, float):
        return v == v
    return v != ""


def pad_well(w: str) -> str:
    return w[0] + f"{int(w[1:]):02d}"


@dataclass
class WellData:
    plate_id: str
    well_id: str
    px_um: float
    chans: np.ndarray                 # (C, H, W) fluorescence of the analysis window, order = names
    names: list                       # channel keys (raw names, e.g. "488nm")
    hole: np.ndarray                  # (H, W) bool
    thr: dict                         # ("whole" | "hole", ch) -> threshold
    bg: dict                          # ch -> (mean, sd) of the whole frame outside particles
    bf: list = field(default_factory=list)   # [(label, (H, W) array)]
    offset: tuple = (0, 0)            # window origin in the frame (row, col)
    frame_shape: tuple | None = None  # shape of the full frame, if known
    notes: list = field(default_factory=list)   # how thresholds / background were obtained
    view_src: dict | None = None      # image file the window came from (display: a larger view around the hole)

    @property
    def R_px(self) -> float:
        return float(np.sqrt(self.hole.sum() / np.pi))


# ---------------------------------------------------------------------------------------------------------------
# helpers shared by the sources

def _triangle(v: np.ndarray) -> float:
    from skimage.filters import threshold_triangle
    v = v[np.isfinite(v)]
    if v.size == 0 or v.min() == v.max():
        return float("nan")
    return float(threshold_triangle(v))


def _stringent_mask(plane: np.ndarray, thr: float, min_px: int) -> np.ndarray:
    from skimage.morphology import remove_small_objects
    if not np.isfinite(thr):
        return np.zeros(plane.shape, bool)
    return remove_small_objects(plane > thr, max_size=min_px)


def _rasterise_roi(path: Path, shape: tuple, scale: float = 1.0) -> np.ndarray:
    """As pipeline/segment.py LoadROI._from_imagej: roifile coordinates (x scale) -> int, clipped -> polygon."""
    import roifile
    from skimage.draw import polygon
    h, w = shape
    coords = roifile.ImagejRoi.fromfile(str(path)).coordinates() * scale
    rows = np.clip(coords[:, 1].astype(int), 0, h - 1)
    cols = np.clip(coords[:, 0].astype(int), 0, w - 1)
    m = np.zeros(shape, bool)
    rr, cc = polygon(rows, cols, shape=shape)
    m[rr, cc] = True
    return m


def _window(hole: np.ndarray, work_R: float):
    ys, xs = np.nonzero(hole)
    R = np.sqrt(len(ys) / np.pi)
    half = int(np.ceil(work_R * R))
    cy, cx = int(round(ys.mean())), int(round(xs.mean()))
    H, W = hole.shape
    return max(cy - half, 0), min(cy + half + 1, H), max(cx - half, 0), min(cx + half + 1, W)


def field_from_db(db: Path, plate_id: str, min_um2: float) -> pd.DataFrame:
    """Field cells: whole-image particles >= min_um2 (one row per object; the pipeline's own measurements)."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    im = pd.read_sql("select id, metadata from images", con)
    wm = {i: pad_well(json.loads(m)["well_id"]) for i, m in zip(im.id, im.metadata)}
    pm = pd.read_sql("select image_id, channel, area_um2, mean_intensity, circularity, solidity, eccentricity "
                     "from particle_measurements where roi_mask = 'whole_image' and area_um2 >= ?", con,
                     params=(min_um2,))
    con.close()
    pm.insert(0, "well_id", pm.image_id.map(wm))
    pm.insert(0, "plate_id", plate_id)
    return pm.rename(columns={"channel": "ch", "mean_intensity": "mean"}).drop(columns="image_id")


# ---------------------------------------------------------------------------------------------------------------

class RunSource:
    """A finished CorrelativeImaging run under `root` (see module doc)."""

    def __init__(self, root: str | Path, run_tag: str, cfg: Config):
        from ._discovery import discover_runs
        runs = {g.run_tag: g for g in discover_runs(root)}
        if run_tag not in runs:
            raise KeyError(f"run {run_tag} not under {root}; found {sorted(runs)}")
        self.cfg, self.run, self.run_tag = cfg, runs[run_tag], run_tag
        rows, self._db = [], {}
        for pr in self.run.plates:
            rows += self._read_plate(pr)
        self.meta = pd.DataFrame(rows)
        if not len(self.meta):
            raise ValueError(f"run {run_tag}: no wells with a hole ROI and diagnostics images")

    # -- paths
    def _local(self, p: str | None) -> Path | None:
        if not _given(p):
            return None
        if Path(p).exists():
            return Path(p)
        for win, loc in self.cfg.path_map.items():
            if p.startswith(win):
                return Path(loc) / PureWindowsPath(p[len(win):]).as_posix()
        return Path(p)

    @staticmethod
    def plate_id_of(folder: str) -> str:
        m = re.search(r"(\d{2}[A-Z]\d{5})", folder)          # plate barcode, if the folder name holds one
        return m.group(1) if m else folder

    def _read_plate(self, pr) -> list:
        con = sqlite3.connect(f"file:{pr.db_path}?mode=ro", uri=True)
        tables = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
        im = pd.read_sql("select * from images", con)
        sel = (pd.read_sql("select * from selections where roi_mask = 'roi_hole'", con)
               if "selections" in tables else pd.DataFrame(columns=["image_id"]))
        lv = (pd.read_sql("select image_id, channel, roi_mask, threshold_value from channel_levels", con)
              if "channel_levels" in tables else pd.DataFrame(columns=["image_id", "channel", "roi_mask", "threshold_value"]))
        bg = pd.read_sql("select image_id, channel, mean_intensity, std_intensity from intensity_measurements "
                         "where roi_mask = 'whole_image:outside_particles'", con)
        chans = [r[0] for r in con.execute("select distinct channel from particle_measurements")]
        px = con.execute("select area_um2 / area_px from particle_measurements where area_px > 0 limit 500").fetchall()
        con.close()
        px_db = float(np.sqrt(np.median([p[0] for p in px]))) if px else np.nan   # the analysis grid, binned or not
        plate_id = self.plate_id_of(pr.plate)
        self._db[plate_id] = pr.db_path
        diag = pr.diagnostics_dir
        rows = []
        for img in im.itertuples():
            md = json.loads(img.metadata or "{}")
            w = md.get("well_id")
            if not w or diag is None:
                continue
            s = sel[sel.image_id == img.id]
            s = s.iloc[0] if len(s) else None
            rec = dict(plate_id=plate_id, well_id=pad_well(w), well_raw=w, image_id=int(img.id), px_um=px_db,
                       raw_shape=tuple((md.get("shape") or [None, None])[-2:]),
                       raw_px=getattr(img, "pixel_size_um", None), db=str(pr.db_path), diag=str(diag),
                       plate_dir=str(pr.db_path.parent), channels_db=chans,
                       roi_path=None if s is None else s.roi_path,
                       crop_row0=None if s is None else s.crop_row0, crop_col0=None if s is None else s.crop_col0,
                       crop_files=None if s is None else s.crop_files)
            l = lv[lv.image_id == img.id]
            rec["thr"] = {("whole" if r.roi_mask == "whole_image" else "hole", r.channel): r.threshold_value
                          for r in l.itertuples() if r.roi_mask in ("whole_image", "roi_hole")}
            b = bg[bg.image_id == img.id]
            rec["bg"] = {r.channel: (r.mean_intensity, r.std_intensity) for r in b.itertuples()}
            rows.append(rec)
        return rows

    def _roi(self, r) -> Path | None:
        p = self._local(r.roi_path)
        if p is not None and p.exists():
            return p
        hits = sorted((Path(r.plate_dir) / "bf_pipeline" / "rois").glob(f"_{r.well_raw}-*_hole.roi"))
        return hits[0] if hits else None

    # -- public
    def wells(self) -> pd.DataFrame:
        return self.meta[["plate_id", "well_id", "px_um"]].copy()

    def plates(self) -> list:
        return sorted(self.meta.plate_id.unique())

    def load(self, plate_id: str, well_id: str) -> WellData | None:
        """Prefers the hole crop the pipeline already wrote (``<well>_roi_hole_crop_channels.tif``), trimmed to the
        analysis window, when its origin is known (``selections``), it covers the whole window and the database
        stores thresholds and backgrounds. Otherwise the window is cut from ``<well>_whole_channels.tif`` (needed for
        older runs: no crop origin, thresholds/background recomputed on the full frame)."""
        import tifffile
        r = self.meta[(self.meta.plate_id == plate_id) & (self.meta.well_id == well_id)]
        if not len(r):
            return None
        r = r.iloc[0]
        cfg = self.cfg
        roi = self._roi(r)
        if roi is None:
            return None
        whole = Path(r.diag) / f"{r.well_raw}_whole_channels.tif"
        crop = None
        if _given(r.crop_files):
            crop = next((Path(r.diag) / f for f in json.loads(r.crop_files) if f.endswith("_channels.tif")), None)
        crop = crop or Path(r.diag) / f"{r.well_raw}_roi_hole_crop_channels.tif"
        stored = all(np.isfinite(r.thr.get(("whole", ch), np.nan)) for ch in r.channels_db) and \
            all(ch in r.bg and np.isfinite(r.bg[ch][0]) for ch in r.channels_db)
        scale = float(r.raw_px) / r.px_um if (_given(r.raw_px) and np.isfinite(r.px_um)) else 1.0
        fs = (int(round(r.raw_shape[0] * scale)), int(round(r.raw_shape[1] * scale))) if r.raw_shape[0] else None
        notes, origin, image, frame_shape = [], (0, 0), None, None
        has_origin = _given(r.crop_row0) and _given(r.crop_col0)
        if crop.exists() and has_origin and stored and fs is not None:
            hole_f = _rasterise_roi(roi, fs, scale)
            if not hole_f.any():
                return None
            w0, w1, v0, v1 = _window(hole_f, cfg.work_R)
            with tifffile.TiffFile(crop) as tf:
                ch_, h_, w_ = tf.series[0].shape[-3:]
            o = (int(r.crop_row0), int(r.crop_col0))
            if o[0] <= w0 and o[1] <= v0 and o[0] + h_ >= w1 and o[1] + w_ >= v1:
                image, origin, frame_shape = crop, o, fs
                notes.append("window from the hole crop")
            else:
                notes.append("hole crop smaller than the window: window from the whole frame")
        if image is None and whole.exists():
            image, origin = whole, (0, 0)
        if image is None and crop.exists() and has_origin:
            image, origin = crop, (int(r.crop_row0), int(r.crop_col0))
            notes.append("no whole-frame TIF: window clipped to the hole crop")
        if image is None:
            return None
        with tifffile.TiffFile(image) as tf:
            stack = tf.asarray().astype(np.float32)
            labels = list((tf.imagej_metadata or {}).get("Labels") or [])
        if image == whole:
            frame_shape = stack.shape[-2:]
            if r.raw_shape[-1]:
                scale = frame_shape[-1] / float(r.raw_shape[-1])
            fs = frame_shape
        fl = [k for k, lab in enumerate(labels) if lab in r.channels_db]
        if not fl:
            fl = list(range(len(r.channels_db)))
            labels = list(r.channels_db) + labels[len(r.channels_db):]
            notes.append("no channel labels in the TIF: channel order taken from the database")
        names = [labels[k] for k in fl]
        if cfg.channels:
            keep = [i for i, n in enumerate(names) if n in cfg.channels]
            fl, names = [fl[i] for i in keep], [names[i] for i in keep]
        bf = [(labels[k] if k < len(labels) else f"BF {k}", stack[k]) for k in range(stack.shape[0])
              if k not in fl and (k >= len(labels) or labels[k] not in r.channels_db)]
        fluo = stack[fl]
        H, W = fluo.shape[1:]
        # hole on the frame grid (as segment.py: ROI drawn on the raw image, scaled to the analysis grid), then into
        # this image's coordinates
        grid = fs or (origin[0] + H, origin[1] + W)
        hole_f = _rasterise_roi(roi, (max(grid[0], origin[0] + H), max(grid[1], origin[1] + W)), scale)
        hole_f = hole_f[origin[0]:origin[0] + H, origin[1]:origin[1] + W]
        if not hole_f.any():
            return None
        on_frame = image == whole
        thr, bg = {}, {}
        for k, ch in enumerate(names):
            t_w, t_h = r.thr.get(("whole", ch)), r.thr.get(("hole", ch))
            if t_w is None or not np.isfinite(t_w):
                t_w = _triangle(fluo[k])
                notes.append(f"{ch}: whole-image threshold computed (triangle){'' if on_frame else ' on the crop'}")
            if t_h is None or not np.isfinite(t_h):
                t_h = _triangle(fluo[k][hole_f])
            thr[("whole", ch)], thr[("hole", ch)] = float(t_w), float(t_h)
            b = r.bg.get(ch)
            if b is None or not np.isfinite(b[0]):
                out = fluo[k][~_stringent_mask(fluo[k], t_w, cfg.min_object_px)]
                b = (float(out.mean()), float(out.std()))
                notes.append(f"{ch}: background computed outside particles{'' if on_frame else ' of the crop'}")
            bg[ch] = (float(b[0]), float(b[1]))
        r0, r1, c0, c1 = _window(hole_f, cfg.work_R)
        cut = (slice(r0, r1), slice(c0, c1))
        bf_idx = [k for k in range(stack.shape[0]) if k not in fl and (k >= len(labels) or labels[k] not in r.channels_db)]
        view = dict(file=str(image), r0=int(origin[0]), c0=int(origin[1]), fl=[int(k) for k in fl],
                    bf=[int(k) for k in bf_idx], bf_labels=[labels[k] if k < len(labels) else f"BF {k}" for k in bf_idx])
        return WellData(plate_id, well_id, float(r.px_um), np.ascontiguousarray(fluo[(slice(None),) + cut]), names,
                        hole_f[cut].copy(), thr, bg, [(lab, a[cut].copy()) for lab, a in bf],
                        (origin[0] + r0, origin[1] + c0), frame_shape, notes, view)

    def field_table(self) -> pd.DataFrame:
        return pd.concat([field_from_db(db, pl, self.cfg.field_min_um2) for pl, db in self._db.items()],
                         ignore_index=True)

    @property
    def dataset(self) -> str:
        return self.run_tag


class JsonpCropSource:
    """Viewer-export crops (``wells/<plate>_<well>.js``, ``DV_WELL({...})``) + a wells table (pickle or CSV with
    plate_id, well_id, px_um, thr_whole_<name>, thr_hole_<name>, bg_mean_<name>, bg_sd_<name>) + the plate
    databases (``*_<plate>.db``) for the field cells. `names` maps the crop's channel labels to channel keys."""

    def __init__(self, crops_dir, wells_table, db_dir, names: dict, cfg: Config, wells: list | None = None,
                 dataset: str = "jsonp"):
        self.dir, self.dbdir, self.names, self.cfg, self.dataset = Path(crops_dir), Path(db_dir), names, cfg, dataset
        wt = Path(wells_table)
        self.meta = pd.read_pickle(wt) if wt.suffix == ".pkl" else pd.read_csv(wt, dtype={"plate_id": str})
        self.meta = self.meta.drop(columns=["win_shape"], errors="ignore")
        if wells is not None:
            keys = set(map(tuple, wells))
            self.meta = self.meta[[k in keys for k in zip(self.meta.plate_id, self.meta.well_id)]]
        self.meta = self.meta.reset_index(drop=True)

    def wells(self) -> pd.DataFrame:
        return self.meta[["plate_id", "well_id", "px_um"]].copy()

    def plates(self) -> list:
        return sorted(self.meta.plate_id.unique())

    def load(self, plate_id: str, well_id: str) -> WellData | None:
        from skimage.draw import polygon
        f = self.dir / f"{plate_id}_{well_id}.js"
        r = self.meta[(self.meta.plate_id == plate_id) & (self.meta.well_id == well_id)]
        if not f.exists() or not len(r):
            return None
        r = r.iloc[0]
        d = json.loads(f.read_text()[len("DV_WELL("):-2])
        arr = []
        for c, q in zip(d["channels"], d["q"]):
            u = np.frombuffer(base64.b64decode(c), "<u2").astype(np.float32).reshape(d["h"], d["w"])
            arr.append(q[0] + u * q[1])
        labels = d["labels"]
        fl = [k for k, lab in enumerate(labels) if lab in self.names]
        names = [self.names[labels[k]] for k in fl]
        disp = {self.names[labels[k]]: labels[k] for k in fl}
        hole = np.zeros((d["h"], d["w"]), bool)
        for xs, ys in d["hole"]:
            rr, cc = polygon(np.asarray(ys) - .5, np.asarray(xs) - .5, hole.shape)
            hole[rr, cc] = True
        thr = {(s, ch): float(r[f"thr_{s}_{disp[ch]}"]) for s in ("whole", "hole") for ch in names}
        bg = {ch: (float(r[f"bg_mean_{disp[ch]}"]), float(r[f"bg_sd_{disp[ch]}"])) for ch in names}
        bf = [(lab, arr[k]) for k, lab in enumerate(labels) if k not in fl and k < len(arr)]
        return WellData(plate_id, well_id, float(r.px_um), np.stack([arr[k] for k in fl]), names, hole, thr, bg, bf,
                        notes=["viewer crop: neighbours beyond the crop are cut"])

    def field_table(self) -> pd.DataFrame:
        out = []
        for pl in self.plates():
            dbs = sorted(self.dbdir.glob(f"*_{pl}.db"))
            if dbs:
                out.append(field_from_db(dbs[0], pl, self.cfg.field_min_um2))
        return pd.concat(out, ignore_index=True) if out else pd.DataFrame()
