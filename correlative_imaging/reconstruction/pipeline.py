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

def save_cache(out: Path, wd: WellData, maps: dict):
    d = out / "cache"
    d.mkdir(parents=True, exist_ok=True)
    arrs = dict(chans=wd.chans.astype(np.float32), hole=wd.hole, names=np.array(wd.names),
                bf=np.stack([a for _, a in wd.bf]).astype(np.float16) if wd.bf else np.zeros((0,) + wd.hole.shape, np.float16),
                bf_labels=np.array([l for l, _ in wd.bf] or [""]), offset=np.array(wd.offset), px_um=wd.px_um,
                thr=json.dumps({f"{k[0]}|{k[1]}": v for k, v in wd.thr.items()}), bg=json.dumps(wd.bg),
                view_src=json.dumps(wd.view_src or {}))
    for (lev, ch), L in maps.items():
        arrs[f"map|{lev}|{ch}"] = L.astype(np.uint16 if L.max() < 65535 else np.int32)
    np.savez_compressed(d / f"{_key(wd.plate_id, wd.well_id)}.npz", **arrs)


def load_cache(out: Path, plate_id: str, well_id: str):
    """Returns (WellData, maps) from the cache, or None."""
    f = Path(out) / "cache" / f"{_key(plate_id, well_id)}.npz"
    if not f.exists():
        return None
    z = np.load(f, allow_pickle=False)
    names = [str(n) for n in z["names"]]
    thr = {tuple(k.split("|")): v for k, v in json.loads(str(z["thr"])).items()}
    bg = {k: tuple(v) for k, v in json.loads(str(z["bg"])).items()}
    bf_labels = [str(l) for l in z["bf_labels"]]
    bf = [(l, z["bf"][i].astype(np.float32)) for i, l in enumerate(bf_labels) if i < len(z["bf"])]
    view = json.loads(str(z["view_src"])) if "view_src" in z.files else {}
    wd = WellData(plate_id, well_id, float(z["px_um"]), z["chans"].astype(np.float32), names, z["hole"], thr, bg, bf,
                  tuple(int(v) for v in z["offset"]), view_src=view or None)
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


def detect_run(src, cfg: Config, out_dir, wells: list | None = None, workers: int = 8, progress=None,
               source_spec: dict | None = None, edits: dict | None = None) -> dict:
    """Stage 1. `wells`: optional list of "PLATE_WELL" keys. progress(done, total, message).
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
    _atomic(out / "wells.csv", lambda t: W.to_csv(t, index=False))
    _atomic(out / "reference.csv", lambda t: ref.to_csv(t, index=False))
    _atomic(out / "particles.parquet", lambda t: F.to_parquet(t, index=False))
    run = dict(dataset=dataset, channels=names, source=source_spec or {}, config=json.loads(cfg.to_json()),
               detected=_now(), detect_s=round(time.time() - t0, 1), n_wells=len(W), n_particles=len(F),
               n_pool=int(F.pool.sum()), n_skipped=skipped, n_edited_wells=len(edits or {}))
    (out / "run.json").write_text(json.dumps(run, indent=2))
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
        return load_recon(self.dir, plate_id, well_id)


def classify_run(out_dir, cfg: Config | None = None, model=None, workers: int = 8, progress=None) -> dict:
    """Stage 2 on a detected output folder. model: a learn.ParticleModel (default: cfg.model)."""
    from .identify import identify, well_calls
    from .learn import ParticleModel
    say = progress or (lambda i, n, m="": print(m or f"{i}/{n}", flush=True) if (m or i % 100 == 0 or i == n) else None)
    res = Results(out_dir)
    out = res.dir
    cfg = cfg or res.config
    t0 = time.time()
    if model is None:
        mp = cfg.model_path()
        if mp is None or not mp.exists():
            raise ValueError("no particle model: train one (Train tab / `train`) or set `model` in the settings")
        model = ParticleModel.load(mp)
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
    rows = list(res.wells[["plate_id", "well_id"]].itertuples(index=False))
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
    cells, info = identify(cells, res.wells, res.reference, identity_ch, cfg)
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
    return dict(cells=cells, calls=calls, particles=F, identity=info)


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
