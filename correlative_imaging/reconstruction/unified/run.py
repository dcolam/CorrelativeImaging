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


NEW_OUT = "unified_new"          # the new first pass writes here by default, next to the old unified/


def prepare_new(results: Path, out: Path, log=print) -> None:
    """The new first pass starts from your inputs of the old one: copies (never moves) experiment.json and
    interventions/ from <results>/unified when <out> has none yet. Answers given later go to <out> only."""
    import shutil
    old = Path(results) / "unified"
    out = Path(out)
    if old.resolve() == out.resolve() or not old.exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    if (old / "experiment.json").exists() and not (out / "experiment.json").exists():
        shutil.copy2(old / "experiment.json", out / "experiment.json")
        log(f"   copied experiment.json from {old}")
    if (old / "interventions").exists() and not (out / "interventions" / "answers.csv").exists():
        shutil.copytree(old / "interventions", out / "interventions", dirs_exist_ok=True)
        log(f"   copied your answers and labels (interventions/) from {old}")


def _plug_holes(A, reviewed: dict, chan_marks: dict) -> dict:
    """New first pass: a hole with a particle you answered "plug / halo" has no cell, unless you said there is a
    cell in it (a particle answered cell / several cells, a channel answer, clump "one cell" or a type label)."""
    P = A.get("particle")
    plug = {(r["plate_id"], r["well_id"]) for r in P.to_dict("records") if r["answer"] == "plug / halo"}
    cell = {(r["plate_id"], r["well_id"]) for r in P.to_dict("records") if r["answer"] in ("cell", "several cells")}
    cell |= set(chan_marks) | {k for k, v in reviewed.items() if v == "yes"}
    cell |= {(r["plate_id"], r["well_id"]) for r in A.get("clump").to_dict("records")
             if r["answer"] in ("one cell", "several cells")}
    return {k: "no" for k in plug - cell}


def _verdicts(A, R: pd.DataFrame, reviewed: dict, chan_marks: dict, W: pd.DataFrame) -> dict:
    """New first pass: your verdict per hole for the learnt hole gate, {(plate, well): 1 cell / 0 no cell}.
    cell: a type label, a channel answer positive / negative, a particle answered cell, clump "one cell".
    no cell: a "no cell" label or answer, a plug hole (_plug_holes), every hole of a well you called empty.
    No verdict: "several cells" (a clump is no call), skip / unsure / exclude, the surrounding-cell size answers;
    a hole with both is left out."""
    from .labels import EXCLUDE_WORDS
    yes, no = set(), set()
    if len(R):
        lab = R.your_label.fillna("").astype(str)
        T = R[(R.kind == "hole") & (lab != "") & ~R.not_cell & ~lab.str.lower().str.startswith(("skip", "unsure"))
              & ~lab.str.lower().isin(EXCLUDE_WORDS)]
        yes |= set(zip(T.plate_id.astype(str), T.well_id.astype(str)))
    yes |= set(chan_marks)
    yes |= {(r["plate_id"], r["well_id"]) for r in A.get("particle").to_dict("records") if r["answer"] == "cell"}
    yes |= {(r["plate_id"], r["well_id"]) for r in A.get("clump").to_dict("records") if r["answer"] == "one cell"}
    no |= {k for k, v in reviewed.items() if v == "no"}
    no |= set(zip(W.plate_id[W.user_empty], W.well_id[W.user_empty]))
    sev = {(r["plate_id"], r["well_id"]) for q in ("clump", "particle") for r in A.get(q).to_dict("records")
           if r["answer"] == "several cells"}
    out = {k: 1 for k in yes - no - sev}
    out.update({k: 0 for k in no - yes - sev})
    return out


def _interventions(out: Path, e: dict, W: pd.DataFrame, new: bool = False):
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
        if v in ("not a cell", "no cell"):
            reviewed[k] = "no"
    if new:
        pl = _plug_holes(A, reviewed, chan_marks)
        reviewed.update(pl)
        if pl:
            applied.append(f"{len(pl)} holes with a particle you called plug / halo: no cell")
    n = len(A.df)
    if n:
        applied.append(f"{n} answers to the first pass's questions")
    if len(R):
        n = R[(R.your_label != "") | R.positive.notna()]
        if len(n):
            applied.append(f"{len(n)} items labelled by you ({int(n.not_cell.sum())} no cell, "
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


def run(results, out=None, workers: int = 8, log=print, new: bool = False) -> Path:
    """new=True: the new first pass. Your answers are final for the particle / hole you answered; a plug answer empties
    its hole; the regularized cell outline (measure.regular_outline); the hole gate learnt from your verdicts and the
    Empty wells (call.learnt_gate) instead of the shape gate; clumps learnt from the whole-image particles and your
    clump answers, flagged only (the type is kept). Writes to <results>/unified_new unless --out says otherwise;
    starts from the old run's inputs."""
    t0 = time.time()
    results = Path(results)
    if not ((results / "run.json").exists() and (results / "particles.parquet").exists()):
        raise SystemExit(f"{results} holds no Detect results (run.json, particles.parquet). Run Detect first: tick "
                         "'Then detect particles for Cell reconstruction' in ci gui, or Detect in ci reconstruct, and "
                         "point --results at its reconstruction_<run tag> folder.")
    out = Path(out) if out else results / (NEW_OUT if new else "unified")
    out.mkdir(parents=True, exist_ok=True)
    if new:
        log("new first pass: your answers are final, plugs empty their hole, regularized outline, learnt hole gate, clumps flagged")
        prepare_new(results, out, log)
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
    applied, reviewed, chan_marks, R, A = _interventions(out, e, W, new)
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
    FP, sinfo = shape.particle_model(results, prefix_of, colour, W, reviewed, questions.particles(A), new=new)
    keepP = [c for c in ["plate_id", "well_id", "ch", "level", "lab", "p_cell", "hsnr", "in_um2", "cy", "cx", "share"]
             if c in FP]
    FP[keepP].assign(level=FP.level.astype(str)).to_parquet(out / "hole_particles.parquet", index=False)
    drop = {}
    if sinfo["used"] or (new and FP.p_cell.notna().any()):
        d = FP[FP.p_cell < 0.2]                     # only clearly not a cell; uncertain particles may still seed
        for r in d.itertuples():
            drop.setdefault((r.plate_id, r.well_id), {}).setdefault(r.ch, {}).setdefault(r.level, []).append(int(r.lab))
        log(f"   particles: cell or not by shape, learnt from {sinfo['no_cell']} no-cell and {sinfo['cell']} cell "
            f"examples (AUC {sinfo.get('auc', float('nan')):.2f}); {sinfo['dropped']} particles are not seeds"
            + (" (your particle answers are final)" if new else ""))
    else:
        log("   particles: no 'no cell' examples, every particle can seed (see report)")
    # "s5": each colour grows from its own seeds again (the all-seeds growth of "s4", 2026-11-04, climbed into
    # neighbours' glow; reverted); "s3": + a marker seeds the outline where no colour does (2026-11-03); "s2": seeds
    # dropped per channel; a new tag re-measures instead of reusing old results
    tag = f"s5-{sinfo.get('no_cell', 0)}-{sinfo.get('cell', 0)}-{sinfo.get('dropped', 0)}" + ("-new3" if new else "")
    mf, mt = out / "measurements.parquet", out / "measurements.tag"
    if mf.exists() and mt.exists() and mt.read_text(encoding="utf-8") == tag:
        M = pd.read_parquet(mf)
    else:
        M = measure.measure(T, W, e, results, out, path_map, workers, log, drop, new=new)
        M.to_parquet(mf, index=False)
        mt.write_text(encoding="utf-8", data=tag)
    log(f"   measured {int((M.error == '').sum())} holes, errors {int((M.error != '').sum())}")
    log("4/5 calls")
    gate_y = _verdicts(A, R, reviewed, chan_marks, W) if new else None
    H, cal = call_mod.call(M, W, F, size, nb, e, results, prefix_of, reviewed, FP, chan_marks, questions.clumps(A),
                           gate_y=gate_y)
    if new:
        g = cal["_shape"].get("learnt_gate", {})
        log(f"   hole gate learnt from {g.get('cell', 0)} cell / {g.get('no_cell', 0)} no-cell verdicts: "
            + (f"AUC {g['auc']:.2f}, cut {g['cut']:.2f}; {g['gate']}" if g.get("used") else f"not used ({g.get('why', '')}), shape gate kept"))
    if reviewed:
        k = [reviewed.get((p, w)) == "no" for p, w in zip(H.plate_id, H.well_id)]
        H.loc[k, "cls"] = "no cell (hand label)"
    log("5/5 your labels and the report")
    soma = labels.soma_cut(FP, A.get("particle"))          # the smallest cell body, from your particle review
    linfo = labels.fit(R, H, F["O"], F, cal, reviewed, soma, new=new)
    sm = linfo.get("soma") or {}
    log(f"   smallest cell outline: {sm['cut_um2']:.1f} um2 ({sm['source']})" if sm.get("cut_um2") else
        f"   smallest cell outline: {sm.get('source') or 'no cut'}")
    if linfo["used"]:
        log(f"   label model from {linfo['n']} labels {linfo['labels']}: leave-one-out accuracy {linfo['acc']:.2f}")
        if new:
            log(f"   no colour: a type needs one of its colours on (own signal above the plate's noise); type colours from "
                f"your labels {linfo.get('type_colours')}; {linfo.get('no_colour_rule', 0)} typed holes set to no colour")
    else:
        log(f"   no label model: {linfo['why']}")
    cinfo = None
    if new:                                               # clumps: whole-image particles + your clump answers
        from . import clumps
        isc, cinfo = clumps.rule(FP, H, clumps.answers(A, R))
        ans = clumps.answers(A)
        typed = ~H.predicted_label.astype(str).isin(["no cell", labels.NOT_CELL, "hole detection failed", labels.EXCLUDE, ""])
        own = H.hand_label.fillna("").astype(str) != "" if "hand_label" in H else pd.Series(False, index=H.index)
        hit = isc & typed & ~own                          # a type label you gave yourself stays
        why = ["clump (your answer)" if ans.get(k) == 1 else "clump (whole image)" for k in zip(H.plate_id, H.well_id)]
        H.loc[isc, "flags"] = [(f + "; " if isinstance(f, str) and f else "") + w
                               for f, w in zip(H.loc[isc, "flags"], np.array(why)[isc.to_numpy()])]
        # flagged only: the type stays (sensitivity first); a figure can leave the flagged holes out
        cinfo["typed"] = int(hit.sum())
        log(f"   clumps: {cinfo['n_clump']} holes flagged ({cinfo['typed']} of them typed; the type is kept)"
            + (f"; rule learnt from {cinfo['several']} several / {cinfo['one']} one-cell answers, "
               f"{cinfo['validation']} balanced accuracy {cinfo['balanced_accuracy']:.2f}" if cinfo["used"]
               else f"; no rule: {cinfo.get('why', '')}"))
        applied.append(f"clumps: {cinfo['n_clump']} holes flagged")
    from . import hue
    hue.add(H, F["O"], F, e)                              # each cell's colour as seen on screen
    keep = ["plate_id", "well_id", "image_id", "cls", "pattern", "confidence", "flags", "status_cell", "cell_um2", "cells_in_body",
            "cell_in_hole", "hole_um2"] + [f"{p}_{c}" for c in colour + marker for p in ("own", "snr")] + \
        [f"q_{c}" for c in colour] + [f"pos_{c}" for c in colour + marker if f"pos_{c}" in H] + \
        [f"shape_only_{c}" for c in colour if f"shape_only_{c}" in H] + \
        ["nb_nearest_um", "nb_nearest_cls", "p_shape", "part_p", "part_um2", "predicted_label", "predicted_p", "hand_label", "cell_um2_old"] + [f"part_{c}" for c in colour] + \
        [f"x_{c}" for c in colour] + ["hue", "purity", "strength", "mix"] + \
        [c for c in H.columns if c.startswith("p_") and c != "p_shape"] + \
        [c for c in H.columns if c.startswith(("g_", "on_"))]           # new first pass: the type step's measures
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
    summary["new"] = dict(on=bool(new), clumps=cinfo)
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
