"""Run the unified workflow on one Detect results folder.

  python -m correlative_imaging.reconstruction unified --results <Detect results folder> [--out <folder>]

Steps: 1 checks, 2 surrounding cells, 3 hole cells measured, 4 calls, 5 report. Re-running after editing
experiment.json or filling an interventions/*.csv file applies the change (steps 2-5 are recomputed; the run's tables
and the hole measurements are reused from <out>/cache and <out>/measurements.parquet).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import call as call_mod
from . import data, experiment, field, labels, measure, qc, questions, report


def _interventions(out: Path, e: dict, W: pd.DataFrame):
    """Apply what the user filled in interventions/*.csv."""
    d = out / "interventions"
    applied = []
    # wells: your answers in the tool (answers.csv), or the older wells_to_check.csv
    A = questions.Answers(out)
    dec = questions.wells(A)
    f = d / "wells_to_check.csv"
    if f.exists():
        R = pd.read_csv(f, dtype=str).fillna("")
        for r in R.itertuples():
            dec.setdefault((r.plate_id, r.well_id), r.decision.strip().lower())
    for (p, w), v in dec.items():
        spec = f"{p}:{w}"
        if v == "exclude" and spec not in e["exclude_wells"]:
            e["exclude_wells"].append(spec); applied.append(f"excluded {spec}")
        if v == "empty" and spec not in e["empty_wells"]:
            e["empty_wells"].append(spec); applied.append(f"empty {spec}")
    # your labels (labels.csv): "not a cell" and labelled holes feed the shape model, positive_channels the dim cuts
    R = labels.read(d / "labels.csv", e)
    reviewed, chan_marks = labels.hole_marks(R, list(e["channels"]))
    # the first pass's questions: channel answers and "not a cell" on holes
    cm, no = questions.channels(A)
    for k, v in cm.items():
        chan_marks.setdefault(k, {}).update(v)
    reviewed.update(no)
    for k, v in questions.clumps(A).items():
        if v == "not a cell":
            reviewed[k] = "no"
    n = len(A.df)
    if n:
        applied.append(f"{n} answers to the first pass's questions")
    if len(R):
        n = R[(R.your_label != "") | R.positive.notna()]
        if len(n):
            applied.append(f"{len(n)} items labelled by you ({int(n.not_cell.sum())} not a cell, "
                           f"{int(R.positive.notna().sum())} with channels)")
    return applied, reviewed, chan_marks, R, A


def _display_top(F: dict, H: pd.DataFrame, colour: list, marker: list) -> dict:
    """Brightness that fills the display, per channel: surrounding cells (raw units above the ring) and holes (noise
    units) - the 90th percentile of the positive ones."""
    O = F["O"]
    field_ = {}
    for c in colour + marker:
        v = (O.loc[O.channel == c, f"med_{c}"] - O.loc[O.channel == c, f"ring_{c}"]).dropna() if f"med_{c}" in O \
            else pd.Series(dtype=float)
        field_[c] = float(np.percentile(v, 90)) if len(v) else 1.0
    hole = {c: float(np.nanpercentile(H.loc[H[f"pos_{c}"].astype(bool), f"snr_{c}"], 90))
            if f"pos_{c}" in H and H[f"pos_{c}"].astype(bool).any() else 50.0 for c in colour + marker}
    # the channel balance of the thumbnails: the brightest pixel of a typical positive surrounding cell of each channel
    # (median over its unsaturated positive cells; a channel without objects of its own: its brightest cell median).
    # Each thumbnail is then stretched from its minimum to its maximum (view.limits), so only the balance comes from here
    field_max = {}
    for c in colour + marker:
        own = O[(O.channel == c) & O.get("learn", True)] if "channel" in O else O.iloc[:0]
        if "max_intensity" in own:
            ok = own.max_intensity.notna() & (own.get("saturated_px", 0) == 0)
            if f"pos_{c}" in own and own[f"pos_{c}"].fillna(False).astype(bool)[ok].sum() >= 20:
                ok &= own[f"pos_{c}"].fillna(False).astype(bool)
            v = own.loc[ok, "max_intensity"]
        else:
            v = pd.Series(dtype=float)
        if len(v):
            field_max[c] = float(v.median())
        else:
            m = O[f"med_{c}"].dropna() if f"med_{c}" in O else pd.Series(dtype=float)
            field_max[c] = float(m.quantile(0.9)) if len(m) else None
    return dict(field=field_, hole=hole, field_max=field_max)


def run(results, out=None, workers: int = 8, log=print) -> Path:
    t0 = time.time()
    results = Path(results)
    if not ((results / "run.json").exists() and (results / "particles.parquet").exists()):
        raise SystemExit(f"{results} holds no Detect results (run.json, particles.parquet). Run Detect first: tick "
                         "'Then detect particles for Cell reconstruction' in ci gui, or Detect in ci reconstruct, and "
                         "point --results at its reconstruction_<run tag> folder.")
    out = Path(out) if out else results / "unified"
    out.mkdir(parents=True, exist_ok=True)
    rj = json.loads((results / "run.json").read_text(encoding="utf-8"))
    path_map = (rj.get("config") or {}).get("path_map") or {}
    src = rj["source"]
    root = data.local_path(src["root"], path_map)
    e = experiment.load(out, rj)
    colour, marker = experiment.colour_channels(e), experiment.marker_channels(e)
    log(f"1/5 reading the run ({root})")
    T = data.load_run(root, src["run_tag"], out / "cache", log)
    P = data.signals(T, colour, marker)
    W = qc.check(T, P, e)
    applied, reviewed, chan_marks, R, A = _interventions(out, e, W)
    if applied:
        W = qc.check(T, P, e)
    log(f"   checks: " + ", ".join(f"{k} {v}" for k, v in W.status.value_counts().items()))
    log("2/5 surrounding cells")
    F = field.learn(T, P, W, e, path_map, log)
    C = field.core_sizes(F, T, e, path_map)
    size = {c: (e.get("cell_size_um2") or {}).get(F["nm"][c], C["core"].get(c)) for c in colour}
    C["table"].to_parquet(out / "cell_cores.parquet", index=False)
    for c, v in questions.sizes(A, C["table"]).items():
        size[c] = v; applied.append(f"cell size {F['nm'][c]} from the cells you called one cell")
    marked = out / "interventions" / "cell_size.csv"
    if marked.exists():
        CS = pd.read_csv(marked).fillna("")
        CS = CS[CS.single_cell.astype(str).str.lower().isin(["y", "yes", "1", "x"])]
        for c in colour:
            a = CS[CS.channel == c].area_core_um2
            if len(a) >= 3:
                size[c] = float(a.median()); applied.append(f"cell size {F['nm'][c]} from {len(a)} cells you marked")
    nb = field.neighbours(F, T, P)
    log("3/5 hole cells")
    from . import shape
    prefix_of = measure.cache_prefix(T["images"][["plate_id", "plate_dir"]].drop_duplicates(), results / "cache")
    FP, sinfo = shape.particle_model(results, prefix_of, colour, W, reviewed, questions.particles(A))
    keepP = [c for c in ["plate_id", "well_id", "ch", "level", "lab", "p_cell", "hsnr", "in_um2", "cy", "cx", "share"]
             if c in FP]
    FP[keepP].assign(level=FP.level.astype(str)).to_parquet(out / "hole_particles.parquet", index=False)
    drop = {}
    if sinfo["used"]:
        d = FP[FP.p_cell < 0.2]                     # only clearly not a cell; uncertain particles may still seed
        for r in d.itertuples():
            drop.setdefault((r.plate_id, r.well_id), {}).setdefault(r.ch, {}).setdefault(r.level, []).append(int(r.lab))
        log(f"   particles: cell or not by shape, learnt from {sinfo['no_cell']} no-cell and {sinfo['cell']} cell "
            f"examples (AUC {sinfo['auc']:.2f}); {sinfo['dropped']} particles are not seeds")
    else:
        log("   particles: no 'no cell' examples, every particle can seed (see report)")
    tag = f"{sinfo.get('no_cell', 0)}-{sinfo.get('cell', 0)}-{sinfo.get('dropped', 0)}"
    mf, mt = out / "measurements.parquet", out / "measurements.tag"
    if mf.exists() and mt.exists() and mt.read_text(encoding="utf-8") == tag:
        M = pd.read_parquet(mf)
    else:
        M = measure.measure(T, W, e, results, out, path_map, workers, log, drop)
        M.to_parquet(mf, index=False)
        mt.write_text(encoding="utf-8", data=tag)
    log(f"   measured {int((M.error == '').sum())} holes, errors {int((M.error != '').sum())}")
    log("4/5 calls")
    H, cal = call_mod.call(M, W, F, size, nb, e, results, prefix_of, reviewed, FP, chan_marks, questions.clumps(A))
    if reviewed:
        k = [reviewed.get((p, w)) == "no" for p, w in zip(H.plate_id, H.well_id)]
        H.loc[k, "cls"] = "not a cell (hand label)"
    log("5/5 your labels and the report")
    linfo = labels.fit(R, H, F["O"], F, cal, reviewed)
    if linfo["used"]:
        log(f"   label model from {linfo['n']} labels {linfo['labels']}: leave-one-out accuracy {linfo['acc']:.2f}")
    else:
        log(f"   no label model: {linfo['why']}")
    from . import hue
    hue.add(H, F["O"], F, e)                              # each cell's colour as seen on screen
    keep = ["plate_id", "well_id", "image_id", "cls", "pattern", "confidence", "flags", "status_cell", "cell_um2", "cells_in_body",
            "cell_in_hole", "hole_um2"] + [f"{p}_{c}" for c in colour + marker for p in ("own", "snr")] + \
        [f"q_{c}" for c in colour] + [f"pos_{c}" for c in colour + marker if f"pos_{c}" in H] + \
        [f"shape_only_{c}" for c in colour if f"shape_only_{c}" in H] + \
        ["nb_nearest_um", "nb_nearest_cls", "p_shape", "part_p", "part_um2", "predicted_label", "predicted_p", "hand_label"] + [f"part_{c}" for c in colour] + \
        [f"x_{c}" for c in colour] + ["hue", "purity", "strength", "mix"]
    H[[k for k in keep if k in H]].to_csv(out / "hole_calls.csv", index=False)
    O = F["O"]
    O.drop(columns=[c for c in O.columns if c.startswith(("med_", "ring_"))]).to_parquet(out / "field_cells.parquet",
                                                                                          index=False)
    FW = O.groupby(["plate_id", "well_id", "cls"]).size().unstack(fill_value=0)
    if linfo["used"]:
        FW = FW.join(O[O.learn].groupby(["plate_id", "well_id", "predicted_label"]).size().unstack(fill_value=0)
                     .add_prefix("label: "))
    FW.to_csv(out / "field_cells_per_well.csv")
    W.to_csv(out / "checks.csv", index=False)
    summary = dict(results=str(results), run=src["run_tag"], minutes=round((time.time() - t0) / 60, 1),
                   applied=applied, cell_size=size, cal={k: v for k, v in cal.items() if k != '_shape'},
                   shape={k: v for k, v in cal['_shape'].items() if k != 'feats'}, cuts=F["cuts"], how=F["how"],
                   bleed={f"{F['nm'][a]}->{F['nm'][b]}": float(F["K"][i, j]) for i, a in enumerate(colour)
                          for j, b in enumerate(colour) if i != j})
    summary["labels"] = {k: v for k, v in linfo.items() if k not in ("X", "y", "confusion", "model_table")}
    if linfo.get("model_table") is not None:
        linfo["model_table"].to_csv(out / "label_model.csv", index=False)       # the label model as a picture
    if linfo.get("used"):
        cf = linfo["confusion"]
        summary["labels"]["confusion"] = {str(i): {str(c): int(v) for c, v in row.items()} for i, row in cf.iterrows()}
    # what the tool needs to draw and ask
    summary.update(names=F["nm"], colour=colour, marker=marker, prefix_of=prefix_of,
                   particle_shape={k: v for k, v in sinfo.items() if k != "feats"},
                   display_top=_display_top(F, H, colour, marker))
    # does more data still help? every learnt quantity again on random subsets (calibration.py)
    from . import calibration
    try:
        calib = calibration.compute(H, F["O"], C["table"], cal, colour, linfo)
    except Exception as ex:                                                  # noqa: BLE001
        calib = {}
        log(f"   calibration curves skipped: {type(ex).__name__}: {ex}")
    summary["calibration"] = {g: {k: D.to_dict("list") for k, D in v.items()} for g, v in calib.items()}
    (out / "summary.json").write_text(encoding="utf-8", data=json.dumps(summary, indent=2, default=float))   # the report's questions read it
    report.write(out, T, P, W, F, C, size, H, cal, e, path_map, summary, linfo, R, calib=calib)
    (out / "summary.json").write_text(encoding="utf-8", data=json.dumps(summary, indent=2, default=float))
    log(f"done in {summary['minutes']} min -> {out / 'report.html'}")
    return out
