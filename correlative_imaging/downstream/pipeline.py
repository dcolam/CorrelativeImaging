"""One downstream run: wells -> particles -> scores -> cells + tier -> refined outlines -> calibrated colour.

Writes to `out_dir` (never into the run's .db):
  particles.csv   every particle touching the hole, both thresholds, with SNR, P(cell), P(neighbour), role, cell
  cells.csv       candidate / neighbour / marker-only cells; candidates with their refined body and calibrated colour
  calls.csv       one row per well: tier, simple call, calibrated call + P, keyed by well_plate (ephacRTools key)
  calibration.json, config.json
  viewer/         the browser viewer (viewer.py)
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .calibrate import calibrate_cells, references
from .cells import cells_of_well
from .config import Config
from .models import ParticleModels
from .particles import extract_particles
from .refine import refine_well
from .score import score_particles
from .viewer import well_payload, write_payload, write_viewer


def field_bleed(wd, p: pd.DataFrame, maps: dict, bg: dict, cfg: Config) -> list:
    """Bleed-through ratios measured on the field cells of the crop: for every stringent object of the source
    colour (>= calib_min_field_um2), (into - bg) / (source - bg) over its own footprint."""
    out = []
    if not len(p):
        return out
    for b in cfg.bleed:
        if b["from"] not in wd.names or b["into"] not in wd.names:
            continue
        kf, ki = wd.names.index(b["from"]), wd.names.index(b["into"])
        f = p[(p.seg == "whole") & (p.ch == b["from"]) & (p.full_px * wd.px_um ** 2 >= cfg.calib_min_field_um2)]
        for r in f.itertuples():
            m = maps[("whole", b["from"])] == r.label
            out.append(dict(plate_id=wd.plate_id, src=b["from"], into=b["into"],
                            own=float(wd.chans[kf][m].mean() - bg[b["from"]][0]),
                            sec=float(wd.chans[ki][m].mean() - bg[b["into"]][0])))
    return out


def process_well(src, meta_row: pd.Series, cfg: Config, models: ParticleModels, viewer_dir: Path):
    wd = src.load(meta_row.plate_id, meta_row.well_id)
    if wd is None:
        return None
    p, maps = extract_particles(wd.chans, wd.names, wd.hole, wd.thr, wd.px_um, cfg, wd.plate_id, wd.well_id)
    meta = pd.DataFrame([meta_row]).assign(px_um=wd.px_um)
    t = p[p.touch] if len(p) else p
    t = score_particles(t, meta, cfg, models) if len(t) else t.assign(role=pd.Series(dtype=str))
    L, S = (t[t.seg == "hole"], t[t.seg == "whole"]) if len(t) else (t, t)
    row, cells, member = cells_of_well(L, S, wd.hole.shape, wd.px_um, cfg, wd.names)
    bg = {c: (meta_row[f"bg_mean_{c}"], meta_row[f"bg_sd_{c}"]) for c in wd.names}
    refined, flood = refine_well(wd, t, cells, maps, bg, cfg) if len(t) else ([], None)
    if len(t):
        t = t.assign(cell_id=t.oid.map(member))
    write_payload(viewer_dir, well_payload(wd, p, t, refined, maps))
    bleed = field_bleed(wd, p, maps, bg, cfg)
    for r in refined:
        r.pop("mask")
    cell_rows = []
    rmap = {r["cell_id"]: r for r in refined}
    for c in cells:
        d = {k: v for k, v in c.items() if k not in ("foot", "members")}
        d.update(plate_id=wd.plate_id, well_id=wd.well_id, n_members=len(c["members"]), **rmap.get(c["cell_id"], {}))
        cell_rows.append(d)
    return dict(well=dict(plate_id=wd.plate_id, well_id=wd.well_id, px_um=wd.px_um, hole_px=int(wd.hole.sum()), **row),
                particles=t.drop(columns=["foot"], errors="ignore"), cells=cell_rows, bleed=bleed)


def run_downstream(src, cfg: Config, out_dir: str | Path, wells: list | None = None, workers: int = 8,
                   col_data: pd.DataFrame | None = None, title: str = "Patched-cell identity",
                   progress=None) -> dict:
    """progress(done, total, message) is called after every well and at each later stage (for the GUI)."""
    def _print(i, n, msg=""):
        if msg:
            print(msg, flush=True)
        elif i % 100 == 0 or i == n:
            print(f"{i}/{n} wells", flush=True)
    say = progress or _print
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    models = ParticleModels.load(cfg.models)
    meta = src.wells()
    if wells:
        keys = {tuple(w.split("_", 1)) for w in wells}
        meta = meta[[(p, w) in keys for p, w in zip(meta.plate_id, meta.well_id)]]
    res = []
    with ThreadPoolExecutor(workers) as ex:
        for i, r in enumerate(ex.map(lambda r: process_well(src, r, cfg, models, out / "viewer"), [r for _, r in meta.iterrows()])):
            if r is not None:
                res.append(r)
            say(i + 1, len(meta), "")
    calls = pd.DataFrame([r["well"] for r in res])
    parts = pd.concat([r["particles"] for r in res], ignore_index=True)
    cells = pd.DataFrame([c for r in res for c in r["cells"]])

    say(len(meta), len(meta), "calibrating colours over the run")
    # calibration over the whole run (field cells + all refined candidate cells)
    field, _ = src.field()
    ratios = pd.DataFrame([b for r in res for b in r["bleed"]])
    ref, bleed = references(field, ratios, cfg)
    cand = cells[(cells.kind == "candidate") & cells.body_px.notna()] if len(cells) else cells
    cal, info = calibrate_cells(cand, meta, ref, bleed, cfg)
    cal_cols = [c for c in cal.columns if c.startswith(("a_", "n_", "p_")) and c not in ("p_cell", "p_nb")]
    cal_cols += ["call_cal"] + (["log_ratio"] if "log_ratio" in cal else [])
    cells = cells.merge(cal[["plate_id", "well_id", "cell_id"] + cal_cols], on=["plate_id", "well_id", "cell_id"],
                        how="left")
    # well-level calibrated call = the best candidate cell (highest P(cell) x (1 - P(nb)))
    best = (cal.assign(rank=cal.p_cell * (1 - cal.p_nb)).sort_values("rank", ascending=False)
            .drop_duplicates(["plate_id", "well_id"]))
    keep = ["plate_id", "well_id", "call_cal"] + [f"p_{r}" for r in cfg.reporters] + \
           ["body_um2", "grow_x", "body_in_hole", "touches_neighbour"]
    calls = calls.merge(best[keep], on=["plate_id", "well_id"], how="left")
    calls["call_cal"] = calls.call_cal.fillna("none")
    calls.insert(0, "well_plate", calls.well_id + "." + calls.plate_id)

    parts.to_csv(out / "particles.csv", index=False)
    cells.to_csv(out / "cells.csv", index=False)
    calls.to_csv(out / "calls.csv", index=False)
    (out / "config.json").write_text(cfg.to_json())
    if col_data is not None:                         # kept for the viewer / GUI (display + evaluation only)
        col_data.to_csv(out / "col_data.csv", index=False)
    (out / "calibration.json").write_text(json.dumps(dict(mixture=info, references=ref.to_dict("records"),
                                                          bleed=bleed.to_dict("records")), indent=2, default=float))
    if col_data is not None and {"Condition", "Induction"} <= set(col_data.columns):
        from .evaluate import report
        (out / "evaluation.txt").write_text(report(calls, col_data), encoding="utf-8")
    say(len(meta), len(meta), "writing tables and viewer")
    write_viewer(out / "viewer", calls, cells, cfg, col_data=col_data, title=title)
    return dict(calls=calls, cells=cells, particles=parts)
