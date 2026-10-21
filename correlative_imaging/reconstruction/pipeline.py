"""One run of the tool, in two stages, writing only into its own output folder (never into the run's .db).

detect_run    wells -> particles (both thresholds) -> field reference -> features. Needs no model, so a new dataset
              can be detected, labelled and trained on before anything is classified.
classify_run  model -> P(class) per particle -> neighbour-aware reconstruction -> calibrated identity per cell.
              Reads the per-well cache written by detect_run, so changing the model or the reconstruction settings
              does not re-read the images.

Output folder:
  run.json            source, dataset, config, model provenance, timings
  wells.csv           one row per well: backgrounds, thresholds, hole, window position
  reference.csv       field reference per plate x channel
  particles.parquet   every particle (pool + context) with measurements, features and (after classify) P(class)
  cells.csv           reconstructed cells with body measurements, n_<ch>, p_<ch>, combination call
  calls.csv           one row per well (keyed by well_plate = B05.18T39383)
  identity.json       per-channel mixture of the identity call
  job.json            the last job (detect / classify / train): started, finished or failed (workflow.py)
  cache/<plate>_<well>.npz   window pixels (float32, so a re-detect from the cache equals Detect), hole, BF planes
                             (float16), label maps (after splits / edits) and the split lines
  recon/<plate>_<well>.npz   reconstruction label image (cells > 0, neighbours < 0)
"""
from __future__ import annotations

import datetime
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config, load_config
from .detect import detect_well, smoothed
from .features import build_features
from .reference import field_reference
from .sources import WellData


def _now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def _key(plate_id, well_id):
    return f"{plate_id}_{well_id}"


# ---------------------------------------------------------------------------------------------------------------
# cache

_PATH_MAP: dict = {}


def set_path_map(path_map: dict | None):
    """Stored path prefix -> local prefix (e.g. {"Z:\\": "/mnt/z/"}), for reading the run's TIFs back from the cache."""
    _PATH_MAP.clear()
    _PATH_MAP.update(path_map or {})


def _local_file(f: str) -> Path:
    from pathlib import PureWindowsPath
    if Path(f).exists():
        return Path(f)
    for win, loc in _PATH_MAP.items():
        if f.startswith(win):
            return Path(loc) / PureWindowsPath(f[len(win):]).as_posix()
    return Path(f)


def save_cache(out: Path, wd: WellData, maps: dict):
    """Per well: hole, label maps, thresholds, backgrounds and where the pixels are. A well read from a TIF of the
    run (the hole crop or the whole frame) stores no pixels: they are read back from that TIF (`view_src`), so the
    cache stays small (~0.01 MB per well instead of ~4.5 MB for a whole PNC crop). Other sources keep their pixels."""
    d = out / "cache"
    d.mkdir(parents=True, exist_ok=True)
    src = wd.view_src or {}
    from_tif = bool(src.get("file"))
    arrs = dict(hole=wd.hole, names=np.array(wd.names),
                bf_labels=np.array([l for l, _ in wd.bf] or [""]), offset=np.array(wd.offset), px_um=wd.px_um,
                thr=json.dumps({f"{k[0]}|{k[1]}": v for k, v in wd.thr.items()}), bg=json.dumps(wd.bg),
                view_src=json.dumps(src), box=np.array(wd.box if wd.box else [-1]),
                pixels=np.array("tif" if from_tif else "cache"))
    if not from_tif:
        arrs["chans"] = wd.chans.astype(np.float32)
        arrs["bf"] = np.stack([a for _, a in wd.bf]).astype(np.float16) if wd.bf else \
            np.zeros((0,) + wd.hole.shape, np.float16)
    for (lev, ch), L in maps.items():
        arrs[f"map|{lev}|{ch}"] = L.astype(np.uint16 if L.max() < 65535 else np.int32)
    np.savez_compressed(d / f"{_key(wd.plate_id, wd.well_id)}.npz", **arrs)


def _read_pixels(src: dict, offset: tuple, shape: tuple):
    """The well's area from the run's TIF: (fluorescence (C, H, W) float32 in channel order, [(label, BF plane)])."""
    import tifffile
    f = _local_file(src["file"])
    if not f.exists():
        raise FileNotFoundError(f"the run's image is not reachable: {src['file']} (Settings: path map)")
    y0, x0 = offset[0] - src["r0"], offset[1] - src["c0"]
    sl = (slice(y0, y0 + shape[0]), slice(x0, x0 + shape[1]))
    try:
        arr = tifffile.memmap(str(f), mode="r")
    except Exception:                                   # compressed / not memory-mappable: read it all
        arr = tifffile.imread(str(f))
    arr = arr.reshape((-1,) + arr.shape[-2:])
    fl = np.stack([np.asarray(arr[k][sl], np.float32) for k in src["fl"]])
    bf = [(lab, np.asarray(arr[k][sl], np.float32)) for lab, k in zip(src.get("bf_labels", []), src.get("bf", []))]
    return fl, bf


def load_cache(out: Path, plate_id: str, well_id: str):
    """Returns (WellData, maps) from the cache, or None. Pixels come from the run's TIF when the cache holds none
    (FileNotFoundError if that TIF cannot be reached)."""
    f = Path(out) / "cache" / f"{_key(plate_id, well_id)}.npz"
    if not f.exists():
        return None
    z = np.load(f, allow_pickle=False)
    names = [str(n) for n in z["names"]]
    thr = {tuple(k.split("|")): v for k, v in json.loads(str(z["thr"])).items()}
    bg = {k: tuple(v) for k, v in json.loads(str(z["bg"])).items()}
    view = json.loads(str(z["view_src"])) if "view_src" in z.files else {}
    box = tuple(int(v) for v in z["box"]) if "box" in z.files and len(z["box"]) == 4 else None
    offset = tuple(int(v) for v in z["offset"])
    hole = z["hole"]
    if "chans" in z.files:
        bf_labels = [str(l) for l in z["bf_labels"]]
        bf = [(l, z["bf"][i].astype(np.float32)) for i, l in enumerate(bf_labels) if i < len(z["bf"])]
        chans = z["chans"].astype(np.float32)
    else:
        chans, bf = _read_pixels(view, offset, hole.shape)
    wd = WellData(plate_id, well_id, float(z["px_um"]), chans, names, hole, thr, bg, bf, offset,
                  view_src=view or None, box=box)
    maps = {(k.split("|")[1], k.split("|")[2]): z[k].astype(np.int32) for k in z.files if k.startswith("map|")}
    return wd, maps


def load_recon(out: Path, plate_id: str, well_id: str):
    f = Path(out) / "recon" / f"{_key(plate_id, well_id)}.npz"
    if not f.exists():
        return None
    z = np.load(f)
    return z["G"].astype(np.int32)


def plate_hole_background(W: pd.DataFrame, names: list, min_free: int = 30) -> pd.DataFrame:
    """Wells whose hole is (nearly) filled by objects have no free hole pixels to measure the hole background on:
    they take the plate's median hole background (of the wells that have one) instead of the frame background,
    which is much darker than the aperture. `hbg_source` records which one a well uses."""
    W = W.copy()
    fb = W.hole_free_px < min_free
    W["hbg_source"] = np.where(fb, "plate median", "own")
    missing = pd.Series(False, index=W.index)
    for ch in names:
        for col in (f"hbg_med_{ch}", f"hbg_sd_{ch}"):
            fill = W.loc[fb, "plate_id"].map(W[~fb].groupby("plate_id")[col].median()).dropna()
            W.loc[fill.index, col] = fill
            missing |= fb & ~W.index.isin(fill.index)
    W.loc[missing, "hbg_source"] = "frame"     # no well of that plate has a free hole: keep the frame background
    return W


def well_hbg(W_row, names: list) -> dict:
    return {ch: (float(W_row[f"hbg_med_{ch}"]), float(W_row[f"hbg_sd_{ch}"])) for ch in names}


# ---------------------------------------------------------------------------------------------------------------

def _atomic(path: Path, write):
    """Write via a temp file and replace (results may live on a network share)."""
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    os.replace(tmp, path)


def _job(step: str, out_dir, fn):
    """Run a stage with its record in job.json (started / finished / failed), so an interrupted or failed run shows
    as such in the workflow status (workflow.py), from the GUI and the command line alike."""
    from .workflow import record_job
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    record_job(out_dir, step, "started")
    try:
        r = fn()
    except BaseException as e:
        record_job(out_dir, step, "failed", f"{type(e).__name__}: {e}")
        raise
    record_job(out_dir, step, "finished")
    return r


def detect_run(src, cfg: Config, out_dir, wells: list | None = None, workers: int = 8, progress=None,
               source_spec: dict | None = None, edits: dict | None = None, cancel=None,
               labels_path: str | None = None) -> dict:
    """Stage 1 (see _detect_run), recorded in job.json and workflow.json (labels_path: the label library the hand
    edits came from)."""
    r = _job("detect", out_dir, lambda: _detect_run(src, cfg, out_dir, wells, workers, progress, source_spec,
                                                      edits, cancel))
    from .provenance import record, sha1_obj
    record(out_dir, "detect", config=json.loads(cfg.to_json()), source=source_spec or {}, wells=wells or None,
           n_wells=int(len(r["wells"])), n_edited_wells=len(edits or {}),
           edits_sha1=sha1_obj(sorted((f"{k[0]}_{k[1]}", v) for k, v in (edits or {}).items())) if edits else None,
           labels=str(labels_path) if labels_path else None)
    return r


def _detect_run(src, cfg: Config, out_dir, wells: list | None = None, workers: int = 8, progress=None,
                source_spec: dict | None = None, edits: dict | None = None, cancel=None) -> dict:
    """Stage 1. `wells`: optional list of "PLATE_WELL" keys. progress(done, total, message). cancel: a
    threading.Event; set, the wells not started yet are skipped and RuntimeError is raised.
    edits: {(plate_id, well_id): {"cuts": [...], "merges": [...]}} from the label library (LabelStore.edits)."""
    say = progress or (lambda i, n, m="": print(m or f"{i}/{n}", flush=True) if (m or i % 100 == 0 or i == n) else None)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    meta = src.wells()
    if wells:
        keys = {tuple(w.split("_", 1)) for w in wells}
        meta = meta[[(p, w) in keys for p, w in zip(meta.plate_id, meta.well_id)]]
    rows = list(meta.itertuples())

    def one(r):
        if cancel is not None and cancel.is_set():          # cancelled: the remaining wells are skipped
            return None
        wd = src.load(r.plate_id, r.well_id)
        if wd is None:
            return None
        p, maps, row = detect_well(wd, cfg, (edits or {}).get((r.plate_id, r.well_id)))
        save_cache(out, wd, maps)
        return p, row, wd.names

    res, names = [], []
    with ThreadPoolExecutor(workers) as ex:
        for i, r in enumerate(ex.map(one, rows)):
            if r is not None:
                res.append(r)
                names = names or r[2]
            say(i + 1, len(rows), "")
    if cancel is not None and cancel.is_set():
        raise RuntimeError("Detect cancelled")
    if not res:
        raise ValueError("no well could be loaded (missing diagnostics TIFs or hole ROIs?)")
    skipped = len(rows) - len(res)
    if skipped:
        say(len(rows), len(rows), f"{skipped} of {len(rows)} wells skipped: no hole ROI or no diagnostics image")
    W = plate_hole_background(pd.DataFrame([r[1] for r in res]), names)
    P = pd.concat([r[0] for r in res], ignore_index=True)
    say(len(rows), len(rows), "field reference + features")
    ref = field_reference(src.field_table(), W, names)
    F = build_features(P, W, ref, names)
    from .preview import provisional_classes, well_previews
    F["dclass"] = provisional_classes(F)
    identity_ch = cfg.identity_channels(names)
    W = W.merge(well_previews(F, W, identity_ch, cfg), on=["plate_id", "well_id"], how="left")
    dataset = getattr(src, "dataset", "") or out.name
    F.insert(0, "dataset", dataset)
    W.insert(0, "dataset", dataset)
    # a new Detect invalidates what Classify derived from the old particles (cells, calls, calibration, bodies)
    stale = [f for f in ("calls.csv", "cells.csv", "identity.json") if (out / f).exists()]
    for f in stale:
        (out / f).unlink()
    if (out / "recon").exists():
        import shutil
        shutil.rmtree(out / "recon", ignore_errors=True)
        stale.append("recon/")
    if stale:
        say(len(rows), len(rows), f"removed the previous Classify outputs ({', '.join(stale)}): run Classify again")
    _atomic(out / "wells.csv", lambda t: W.to_csv(t, index=False))
    _atomic(out / "reference.csv", lambda t: ref.to_csv(t, index=False))
    _atomic(out / "particles.parquet", lambda t: F.to_parquet(t, index=False))
    run = dict(dataset=dataset, channels=names, source=source_spec or {}, config=json.loads(cfg.to_json()),
               detected=_now(), detect_s=round(time.time() - t0, 1), n_wells=len(W), n_particles=len(F),
               n_pool=int(F.pool.sum()), n_skipped=skipped, n_edited_wells=len(edits or {}))
    (out / "run.json").write_text(json.dumps(run, indent=2))
    try:
        from .diagnostics import detect_diagnostics
        say(len(rows), len(rows), "diagnostic plots")
        detect_diagnostics(out, cfg)
    except Exception as e:                              # plots never fail Detect
        say(len(rows), len(rows), f"diagnostic plots failed: {e}")
    return dict(wells=W, particles=F, reference=ref)


class Results:
    """An output folder, read back (GUI, training, R export)."""

    def __init__(self, out_dir):
        self.dir = Path(out_dir)
        self.run = json.loads((self.dir / "run.json").read_text())
        self.wells = pd.read_csv(self.dir / "wells.csv", dtype={"plate_id": str, "well_id": str})
        self.reference = pd.read_csv(self.dir / "reference.csv", dtype={"plate_id": str})
        self.particles = pd.read_parquet(self.dir / "particles.parquet")
        self.cells = pd.read_csv(self.dir / "cells.csv", dtype={"plate_id": str}) if (self.dir / "cells.csv").exists() else None
        self.calls = pd.read_csv(self.dir / "calls.csv", dtype={"plate_id": str}) if (self.dir / "calls.csv").exists() else None

    @property
    def dataset(self) -> str:
        return self.run["dataset"]

    @property
    def channels(self) -> list:
        return self.run["channels"]

    @property
    def config(self) -> Config:
        from .config import LEGACY_FEATURE_VALUES
        cfg = Config()
        stored = self.run.get("config", {})
        for k, v in LEGACY_FEATURE_VALUES.items():          # detected before the setting existed
            if k not in stored:
                setattr(cfg, k, v)
        for k, v in stored.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg

    def well(self, plate_id, well_id):
        return load_cache(self.dir, plate_id, well_id)

    def recon(self, plate_id, well_id):
        """The well's reconstruction, or None when there is none or it no longer fits the well (made before the
        well was detected again: run Classify)."""
        G = load_recon(self.dir, plate_id, well_id)
        if G is None or "win_h" not in self.wells:
            return G
        w = self.wells[(self.wells.plate_id == plate_id) & (self.wells.well_id == well_id)]
        if len(w) and tuple(G.shape) != (int(w.win_h.iloc[0]), int(w.win_w.iloc[0])):
            return None
        return G


def classify_run(out_dir, cfg: Config | None = None, model=None, workers: int = 8, progress=None,
                 cell_labels: pd.DataFrame | None = None, labels_path: str | None = None) -> dict:
    """Stage 2 (see _classify_run), recorded in job.json and workflow.json (labels_path: the label library the cell
    labels came from)."""
    r = _job("classify", out_dir, lambda: _classify_run(out_dir, cfg, model, workers, progress, cell_labels))
    from .provenance import record, sha1_file, sha1_obj
    run = json.loads((Path(out_dir) / "run.json").read_text())
    used = Config(**{k: v for k, v in run.get("config", {}).items() if hasattr(Config(), k)})
    mfile = getattr(r.get("model"), "path", None)            # the file the model actually came from
    if mfile is None and (run.get("model") or {}).get("arm") != "preview":
        mp = used.model_path()
        mfile = str(mp) if (mp is not None and mp.exists()) else None
    n_cl = int(len(cell_labels)) if cell_labels is not None else 0
    record(out_dir, "classify", config=run.get("config", {}), model_file=mfile,
           model_sha1=sha1_file(mfile) if mfile else None, model_arm=(run.get("model") or {}).get("arm"),
           model_fitted=(run.get("model") or {}).get("fitted"), labels=str(labels_path) if labels_path else None,
           n_cell_labels=n_cl,
           cell_labels_sha1=sha1_obj(cell_labels[[c for c in ("plate_id", "well_id", "y", "x", "verdict", "colour")
                                                  if c in cell_labels]].to_dict("records")) if n_cl else None)
    return r


def _classify_run(out_dir, cfg: Config | None = None, model=None, workers: int = 8, progress=None,
                  cell_labels: pd.DataFrame | None = None) -> dict:
    """Stage 2 on a detected output folder. model: a learn.ParticleModel (default: cfg.model; none: the Detect
    preview). cell_labels: LabelStore.cell_frame rows of this dataset, test wells removed (verdicts and colours:
    identify.py 2b)."""
    from .identify import identify, well_calls
    from .learn import ParticleModel
    say = progress or (lambda i, n, m="": print(m or f"{i}/{n}", flush=True) if (m or i % 100 == 0 or i == n) else None)
    res = Results(out_dir)
    out = res.dir
    cfg = cfg or res.config
    _PATH_MAP.update(cfg.path_map or {})
    t0 = time.time()
    if model is None:
        mp = cfg.model_path()
        if mp is None or not mp.exists():
            if cfg.model:
                raise ValueError(f"model {cfg.model} not found")
            from .preview import PreviewModel
            model = PreviewModel()                     # clean slate: the Detect preview stands in for a model
            say(0, 0, "no model: classifying with the Detect preview (train a model once you have labels)")
        else:
            model = ParticleModel.load(mp)
    need = [f for f in model.spec.get("features", []) if f not in res.particles or (
        f == "f_coloc" and len(res.channels) > 1 and not any(c.startswith("r_") for c in res.particles.columns))]
    if need:
        raise ValueError(f"the model needs features these results lack ({', '.join(need)}): they were detected "
                         "before the feature existed. Run Detect again")
    from .config import complete_feature_config
    fc_model = complete_feature_config(model.spec.get("feature_config") or {})
    fc_run = res.config.feature_config()
    diff = {k: (fc_run.get(k), v) for k, v in fc_model.items() if fc_run.get(k) != v} \
        if model.spec.get("feature_config") else {}
    warn = (f"detection settings differ from the model's training ({', '.join(f'{k}: {a} vs {b}' for k, (a, b) in diff.items())}):"
            " features are not comparable; detect again with the model's settings or retrain") if diff else ""
    if warn:
        say(0, 0, "WARNING " + warn)
    F = model.predict(res.particles)
    names = res.channels
    identity_ch = cfg.identity_channels(names)
    body = identity_ch or names
    groups = {k: g for k, g in F.groupby(["plate_id", "well_id"])}
    Wi = res.wells.set_index(["plate_id", "well_id"])
    # a fixed order (edited wells are appended to the tables): the identity fit must not depend on edit history
    rows = list(res.wells[["plate_id", "well_id"]].sort_values(["plate_id", "well_id"]).itertuples(index=False))
    (out / "recon").mkdir(exist_ok=True)

    def one(pw):
        got = load_cache(out, *pw)
        g = groups.get(tuple(pw))
        if got is None or g is None:
            return []
        return _classify_well(out, *got, g, well_hbg(Wi.loc[tuple(pw)], got[0].names), body, cfg)

    cell_rows = []
    with ThreadPoolExecutor(workers) as ex:
        for i, r in enumerate(ex.map(one, rows)):
            cell_rows += r
            say(i + 1, len(rows), "")
    cells = pd.DataFrame(cell_rows)
    say(len(rows), len(rows), "identity call")
    user = None
    if cell_labels is not None and len(cell_labels) and len(cells):
        from .labels import match_cells
        m = match_cells(cell_labels, res)                # by point, on the reconstruction just written
        if len(m):
            user = m[["plate_id", "well_id", "cell_id", "verdict", "colour"]].drop_duplicates(
                ["plate_id", "well_id", "cell_id"], keep="last")
            cells = cells.merge(user[["plate_id", "well_id", "cell_id", "verdict"]], how="left")
    cells, info = identify(cells, res.wells, res.reference, identity_ch, cfg, user=user)
    calls = well_calls(cells, res.wells, identity_ch)
    # particle -> cell membership (for the viewer)
    mem = {(r.plate_id, r.well_id, o): r.cell_id for r in cells.itertuples() for o in str(r.members).split(";") if o} \
        if len(cells) else {}
    F["cell_id"] = [mem.get(k) for k in zip(F.plate_id, F.well_id, F.oid)]
    _atomic(out / "particles.parquet", lambda t: F.to_parquet(t, index=False))
    _atomic(out / "cells.csv", lambda t: cells.to_csv(t, index=False))
    _atomic(out / "calls.csv", lambda t: calls.to_csv(t, index=False))
    (out / "identity.json").write_text(json.dumps(info, indent=2, default=float))
    run = res.run
    run.update(classified=_now(), classify_s=round(time.time() - t0, 1), model=model.spec,
               config=json.loads(cfg.to_json()), identity_channels=identity_ch, warning=warn, edited_wells=[])
    (out / "run.json").write_text(json.dumps(run, indent=2, default=str))
    try:
        from .diagnostics import classify_diagnostics
        classify_diagnostics(out, info, cfg)
    except Exception:                                   # plots never fail Classify
        pass
    return dict(cells=cells, calls=calls, particles=F, identity=info, model=model)


def _classify_well(out: Path, wd, maps, g, hbg, body, cfg) -> list:
    """Reconstruct one well from its classified particles; saves recon/<well>.npz, returns cell rows."""
    from .reconstruct import reconstruct_from_particles
    cells, G, _ = reconstruct_from_particles(wd, smoothed(wd, cfg), g, maps, hbg, body, cfg)
    (out / "recon").mkdir(exist_ok=True)
    np.savez_compressed(out / "recon" / f"{_key(wd.plate_id, wd.well_id)}.npz", G=G.astype(np.int16))
    rows = []
    for k, c in enumerate(cells, start=1):
        c.pop("mask")
        c["members"] = ";".join(c["members"])
        rows.append(dict(plate_id=wd.plate_id, well_id=wd.well_id, cell_id=k, **c))
    return rows


def redetect_well(out_dir, plate_id: str, well_id: str, edits: dict | None, cfg: Config | None = None,
                  model=None) -> dict:
    """Detect one well again from the cache with its hand edits (cuts / merges), using the run's own field reference
    and plate backgrounds, and replace its rows. If the run was classified, the well is classified and
    reconstructed again too; its identity uses the stored per-channel calibration (identity.json), so no other
    well's call changes. The run is marked `edited_wells` until the next full Classify."""
    from .preview import provisional_classes, well_previews
    res = Results(out_dir)
    out = res.dir
    cfg = cfg or res.config
    _PATH_MAP.update(cfg.path_map or {})
    got = load_cache(out, plate_id, well_id)
    if got is None:
        raise ValueError(f"{plate_id} {well_id}: not in the cache")
    wd, _ = got
    p, maps, row = detect_well(wd, cfg, edits)
    names = res.channels
    W_all = res.wells
    others = W_all[~((W_all.plate_id == plate_id) & (W_all.well_id == well_id))]
    keep_cols = [c for c in W_all.columns if c in row or c in ("dataset",)]
    W_new = plate_hole_background(pd.concat([others[keep_cols].drop(columns="dataset", errors="ignore"),
                                             pd.DataFrame([row])], ignore_index=True), names)
    w_row = W_new[(W_new.plate_id == plate_id) & (W_new.well_id == well_id)]
    F = build_features(p, w_row, res.reference, names)
    F["dclass"] = provisional_classes(F)
    w_row = w_row.merge(well_previews(F, w_row, cfg.identity_channels(names), cfg), on=["plate_id", "well_id"],
                        how="left")
    F.insert(0, "dataset", res.dataset)
    w_row.insert(0, "dataset", res.dataset)
    save_cache(out, wd, maps)
    P_all = res.particles
    P_all = P_all[~((P_all.plate_id == plate_id) & (P_all.well_id == well_id))]
    W_out = pd.concat([W_all[~((W_all.plate_id == plate_id) & (W_all.well_id == well_id))], w_row],
                      ignore_index=True)
    run = res.run
    if res.calls is not None:
        from .identify import apply_identity, well_calls
        from .learn import ParticleModel
        if model is None and cfg.model_path() and cfg.model_path().exists():
            model = ParticleModel.load(cfg.model_path())
        if model is None and (run.get("model") or {}).get("arm") == "preview":
            from .preview import PreviewModel
            model = PreviewModel()
        if model is not None:
            F = model.predict(F)
            identity_ch = run.get("identity_channels") or cfg.identity_channels(names)
            body = identity_ch or names
            rows = _classify_well(out, wd, maps, F, well_hbg(w_row.iloc[0], names), body, cfg)
            info = json.loads((out / "identity.json").read_text()) if (out / "identity.json").exists() else {}
            new_cells = apply_identity(pd.DataFrame(rows), w_row, res.reference, identity_ch, cfg, info) \
                if rows else pd.DataFrame()
            mem = {o: r.cell_id for r in new_cells.itertuples() for o in str(r.members).split(";") if o} \
                if len(new_cells) else {}
            F["cell_id"] = [mem.get(o) for o in F.oid]
            cells = res.cells if res.cells is not None else pd.DataFrame()
            cells = pd.concat([cells[~((cells.plate_id == plate_id) & (cells.well_id == well_id))], new_cells],
                              ignore_index=True) if len(cells) else new_cells
            new_call = well_calls(new_cells, w_row, identity_ch)
            calls = pd.concat([res.calls[~((res.calls.plate_id == plate_id) & (res.calls.well_id == well_id))],
                               new_call], ignore_index=True)
            _atomic(out / "cells.csv", lambda t: cells.to_csv(t, index=False))
            _atomic(out / "calls.csv", lambda t: calls.to_csv(t, index=False))
            run["edited_wells"] = sorted(set(run.get("edited_wells", [])) | {f"{plate_id}_{well_id}"})
    P_out = pd.concat([P_all, F], ignore_index=True)
    _atomic(out / "particles.parquet", lambda t: P_out.to_parquet(t, index=False))
    _atomic(out / "wells.csv", lambda t: W_out.to_csv(t, index=False))
    (out / "run.json").write_text(json.dumps(run, indent=2, default=str))
    from .provenance import record
    record(out, "edit", well=f"{plate_id}_{well_id}", n_edits=sum(len(v) for v in (edits or {}).values()),
           reclassified=bool(res.calls is not None))
    return dict(particles=F, well=w_row)


def source_from_spec(spec: dict, cfg: Config):
    from .sources import JsonpCropSource, RunSource
    if spec.get("kind") == "jsonp":
        wells = None
        if spec.get("wells_csv"):
            w = pd.read_csv(spec["wells_csv"], dtype=str)
            wells = set(zip(w.plate_id, w.well_id))
        return JsonpCropSource(spec["crops"], spec["wells_table"], spec["db_dir"], spec["names"], cfg, wells,
                               spec.get("dataset", "jsonp"))
    return RunSource(spec["root"], spec["run_tag"], cfg)


def run_all(spec: dict, cfg: Config, out_dir, wells=None, workers=8, progress=None, classify=True):
    src = source_from_spec(spec, cfg)
    r = detect_run(src, cfg, out_dir, wells, workers, progress, spec)
    if classify and cfg.model_path() and cfg.model_path().exists():
        r.update(classify_run(out_dir, cfg, workers=workers, progress=progress))
    return r
