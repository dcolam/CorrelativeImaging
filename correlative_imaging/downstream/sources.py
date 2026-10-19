"""Where wells come from. Everything downstream works on :class:`WellData`, never on files.

RunSource  a finished CorrelativeImaging run: per-plate .db + diagnostics/<well>_roi_hole_crop_channels.tif
           (fluorescence + BF planes), crop offsets in the `selections` table, hole = the ImageJ .roi rasterised
           exactly as pipeline/segment.py (roifile coordinates -> int -> skimage.draw.polygon).
DevV5Source  the local copies of ROMK run v5 (z 12) in correlative_imaging_reports: thresholds/backgrounds from
           workflow_comparison/cache/wells.pkl, pixels from the viewer crops (viewer/wells/*.js, +-2.2 hole radii),
           field cells from romk/imaging/v5/*.db. For development while the share is offline.

Both also give the field-cell statistics per well x channel from the DB (calibration reference). Well ids are
padded (B05) everywhere.
"""
from __future__ import annotations

import base64
import glob
import json
import pickle
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd

from .config import Config

pad_well = lambda w: w[0] + f"{int(w[1:]):02d}"


@dataclass
class WellData:
    plate_id: str
    well_id: str
    px_um: float
    chans: np.ndarray                 # (C, H, W) fluorescence, order = names
    names: list
    hole: np.ndarray                  # (H, W) bool
    thr: dict                         # (seg, ch) -> threshold
    bf: list = field(default_factory=list)   # [(label, (H, W) array)]
    offset: tuple = (0, 0)            # crop origin in the frame (rows, cols)


def _field_from_db(db: Path, plate_id: str, cfg: Config, min_um2: float) -> pd.DataFrame:
    """Field cells (whole-image particles >= min_um2) per well x channel: the calibration reference."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    im = pd.read_sql("select id, metadata from images", con)
    wm = {i: pad_well(json.loads(m)["well_id"]) for i, m in zip(im.id, im.metadata)}
    pm = pd.read_sql("select image_id, channel, label, area_um2, mean_intensity from particle_measurements "
                     "where roi_mask = 'whole_image'", con)
    bg = pd.read_sql("select image_id, channel, mean_intensity from intensity_measurements "
                     "where roi_mask = 'whole_image:outside_particles'", con)
    con.close()
    for t in (pm, bg):
        t["well_id"] = t.image_id.map(wm)
        t["ch"] = t.channel.map(cfg.channels)
    pm["plate_id"] = plate_id
    bgm = bg.set_index(["image_id", "ch"]).mean_intensity
    pm["bg"] = [bgm.get((i, c), np.nan) for i, c in zip(pm.image_id, pm.ch)]
    fld = pm[pm.area_um2 >= min_um2]
    out = fld.groupby(["plate_id", "well_id", "ch"]).agg(field_n=("mean_intensity", "size"),
                                                          field_median=("mean_intensity", "median"),
                                                          field_max=("mean_intensity", "max"),
                                                          bg_mean=("bg", "first")).reset_index()
    return out, pd.DataFrame()   # bleed ratios are measured on the crop pixels (pipeline.field_bleed)


class DevV5Source:
    """Local v5 copies (see module doc). Hole and pixels come from the viewer crops, so neighbours reaching beyond
    the +-2.2 hole-radius window are cut (edge flag); use it for development, validate on RunSource."""

    def __init__(self, wc_dir: str | Path, v5_db_dir: str | Path, cfg: Config):
        self.wc, self.dbdir, self.cfg = Path(wc_dir), Path(v5_db_dir), cfg
        self.meta = pickle.load(open(self.wc / "cache/wells.pkl", "rb")).drop(columns=["win_shape"], errors="ignore")

    def wells(self) -> pd.DataFrame:
        return self.meta

    def plates(self) -> list:
        return sorted(self.meta.plate_id.unique())

    def load(self, plate_id: str, well_id: str) -> WellData | None:
        from skimage.draw import polygon
        f = self.wc / f"viewer/wells/{plate_id}_{well_id}.js"
        if not f.exists():
            return None
        d = json.loads(f.read_text()[len("DV_WELL("):-2])
        arr = []
        for c, q in zip(d["channels"], d["q"]):
            u = np.frombuffer(base64.b64decode(c), "<u2").astype(np.float32).reshape(d["h"], d["w"])
            arr.append(q[0] + u * q[1])
        names = list(self.cfg.channels.values())
        hole = np.zeros((d["h"], d["w"]), bool)
        for xs, ys in d["hole"]:
            rr, cc = polygon(np.asarray(ys) - .5, np.asarray(xs) - .5, hole.shape)
            hole[rr, cc] = True
        r = self.meta[(self.meta.plate_id == plate_id) & (self.meta.well_id == well_id)].iloc[0]
        thr = {(s, ch): r[f"thr_{s}_{ch}"] for s in ("whole", "hole") for ch in names}
        bf = [(lab, a) for lab, a in zip(d["labels"][len(names):], arr[len(names):])]
        return WellData(plate_id, well_id, float(r.px_um), np.stack(arr[:len(names)]), names, hole, thr, bf)

    def field(self):
        outs, rats = [], []
        for pl in self.plates():
            db = glob.glob(str(self.dbdir / f"*_{pl}.db"))
            if db:
                o, r = _field_from_db(Path(db[0]), pl, self.cfg, self.cfg.calib_min_field_um2)
                outs.append(o); rats.append(r)
        return pd.concat(outs), pd.concat(rats) if rats else pd.DataFrame()


class RunSource:
    """A finished CorrelativeImaging run (all plate databases sharing one run tag under `root`)."""

    def __init__(self, root: str | Path, run_tag: str, cfg: Config):
        from ._discovery import discover_runs
        runs = {g.run_tag: g for g in discover_runs(root)}
        if run_tag not in runs:
            raise KeyError(f"run {run_tag} not under {root}; found {sorted(runs)}")
        self.cfg, self.run = cfg, runs[run_tag]
        self._meta, self._plate_of = [], {}
        for pr in self.run.plates:
            self._read_plate(pr)
        self.meta = pd.DataFrame(self._meta)

    def _local(self, p: str | None) -> Path | None:
        if not p:
            return None
        if Path(p).exists():                                # native path (Windows) works as stored
            return Path(p)
        for win, loc in self.cfg.path_map.items():
            if p.startswith(win):
                return Path(loc) / PureWindowsPath(p[len(win):]).as_posix()
        return Path(p)

    def _read_plate(self, pr):
        cfg = self.cfg
        con = sqlite3.connect(f"file:{pr.db_path}?mode=ro", uri=True)
        im = pd.read_sql("select * from images", con)
        tables = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
        if "selections" not in tables:
            con.close()
            raise ValueError(f"{pr.db_path.name} has no 'selections' table: this run was made before the pipeline "
                             "stored the hole crops. Rerun it with the current pipeline (diagnostics + crops on).")
        sel = pd.read_sql("select * from selections where roi_mask = 'roi_hole'", con)
        lv = pd.read_sql("select image_id, channel, roi_mask, threshold_value from channel_levels", con)
        it = pd.read_sql("select image_id, channel, mean_intensity, std_intensity from intensity_measurements "
                         "where roi_mask = 'whole_image:outside_particles'", con)
        pm = pd.read_sql("select area_px, area_um2 from particle_measurements limit 2000", con)
        con.close()
        plate_id = pr.plate                                   # plate folder name; barcode extracted below if present
        import re
        m = re.search(r"(\d{2}[A-Z]\d{5})", pr.plate)
        plate_id = m.group(1) if m else plate_id
        px_fallback = float(np.sqrt((pm.area_um2 / pm.area_px).median())) if len(pm) else np.nan
        for t in (lv, it):
            t["ch"] = t.channel.map(cfg.channels)
        for s in sel.itertuples():
            img = im[im.id == s.image_id].iloc[0]
            md = json.loads(img.metadata or "{}")
            w = md.get("well_id")
            if not w:
                continue
            px = img.get("analysis_pixel_size_um")
            px = float(px) if px is not None and np.isfinite(px) else px_fallback
            rec = dict(plate_id=plate_id, well_raw=w, well_id=pad_well(w), px_um=px, db=str(pr.db_path),
                       diag=str(pr.diagnostics_dir) if pr.diagnostics_dir else None, plate_dir=str(pr.db_path.parent),
                       roi_path=s.roi_path, crop_row0=s.crop_row0, crop_col0=s.crop_col0,
                       crop_files=s.crop_files, hole_px_db=s.area_px)
            for ch in cfg.channels.values():
                l = lv[(lv.image_id == s.image_id) & (lv.ch == ch)].set_index("roi_mask").threshold_value
                rec[f"thr_whole_{ch}"], rec[f"thr_hole_{ch}"] = l.get("whole_image"), l.get("roi_hole")
                b = it[(it.image_id == s.image_id) & (it.ch == ch)]
                rec[f"bg_mean_{ch}"] = b.mean_intensity.iloc[0] if len(b) else np.nan
                rec[f"bg_sd_{ch}"] = b.std_intensity.iloc[0] if len(b) else np.nan
            self._meta.append(rec)
        self._plate_of[plate_id] = pr

    def wells(self) -> pd.DataFrame:
        return self.meta

    def plates(self) -> list:
        return sorted(self.meta.plate_id.unique())

    def load(self, plate_id: str, well_id: str) -> WellData | None:
        import roifile
        import tifffile
        from skimage.draw import polygon
        r = self.meta[(self.meta.plate_id == plate_id) & (self.meta.well_id == well_id)]
        if not len(r) or not r.iloc[0].diag:
            return None
        r = r.iloc[0]
        files = json.loads(r.crop_files or "[]")
        stack = next((f for f in files if f.endswith("_channels.tif")), f"{r.well_raw}_roi_hole_crop_channels.tif")
        path = Path(r.diag) / stack
        roi = self._local(r.roi_path)
        if roi is None or not roi.exists():
            roi = Path(r.plate_dir) / "bf_pipeline" / "rois" / f"_{r.well_raw}-1_hole.roi"
        try:
            with tifffile.TiffFile(path) as tf:
                c = tf.asarray().astype(np.float32)
                labels = list((tf.imagej_metadata or {}).get("Labels") or [])
            pts = roifile.ImagejRoi.fromfile(roi).coordinates()
        except Exception:
            return None
        names = list(self.cfg.channels.values())
        r0, c0 = int(r.crop_row0), int(r.crop_col0)
        H, W = c.shape[-2:]
        # exactly as segment.py: int coordinates, polygon on the frame; only the crop window is kept
        rows, cols = pts[:, 1].astype(int) - r0, pts[:, 0].astype(int) - c0
        hole = np.zeros((H, W), bool)
        rr, cc = polygon(rows, cols, (H, W))
        hole[rr, cc] = True
        if not hole.any():
            return None
        thr = {(s, ch): r[f"thr_{s}_{ch}"] for s in ("whole", "hole") for ch in names}
        nf = len(names)
        bf_labels = labels[nf:] if len(labels) >= c.shape[0] else [f"BF {k}" for k in range(c.shape[0] - nf)]
        bf = list(zip(bf_labels, list(c[nf:])))
        return WellData(plate_id, well_id, float(r.px_um), c[:nf], names, hole, thr, bf, (r0, c0))

    def field(self):
        outs, rats = [], []
        for pl, pr in self._plate_of.items():
            o, r = _field_from_db(pr.db_path, pl, self.cfg, self.cfg.calib_min_field_um2)
            outs.append(o); rats.append(r)
        return pd.concat(outs), pd.concat(rats) if rats else pd.DataFrame()
