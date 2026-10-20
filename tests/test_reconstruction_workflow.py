"""Workflow status (reconstruction/workflow.py): every state, from hand-written results folders (no images)."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from correlative_imaging.reconstruction.labels import LabelStore
from correlative_imaging.reconstruction.workflow import (MIN_TRAIN, STALE_LABELS, next_step, record_job, summary,
                                                         workflow_state)

T0 = "2026-10-01T10:00:00"
T1 = "2026-10-01T11:00:00"
T2 = "2026-10-01T12:00:00"


def _detected(d, t=T0, **run):
    d.mkdir(parents=True, exist_ok=True)
    for f in ("wells.csv", "reference.csv"):
        (d / f).write_text("plate_id,well_id\nP1,A01\n")
    (d / "particles.parquet").write_bytes(b"")
    r = dict(dataset="ds", channels=["c1"], n_wells=1, n_pool=3, **run)
    if t is not None:
        r["detected"] = t
    (d / "run.json").write_text(json.dumps(r))
    return d


def _classified(d, t=T1, model=None, **extra):
    run = json.loads((d / "run.json").read_text())
    run.update(model=model or dict(arm="preview", fitted=""), **extra)
    if t is not None:
        run["classified"] = t
    (d / "run.json").write_text(json.dumps(run))
    pd.DataFrame(dict(plate_id=["P1", "P1"], well_id=["A01", "A02"], status=["one cell", "no cell"],
                      call=["c1", "no cell"])).to_csv(d / "calls.csv", index=False)
    (d / "cells.csv").write_text("plate_id,well_id,cell_id\nP1,A01,1\n")
    return d


def _model(fitted=T1, arm="rf"):
    return SimpleNamespace(spec=dict(arm=arm, fitted=fitted, n_train=40, datasets=["ds"]))


def _store(tmp_path, n_train=0, t=T1, test=(("P1", "T01"),)):
    st = LabelStore(tmp_path / "ci_labels.db")
    if test:
        st.seed_test_wells("ds", list(test))
    for k in range(n_train):
        st.set("ds", "P1", "A01", "c1", "any", k, k, "cell")
    if n_train:                                         # timestamps as given (the store writes "now")
        with st._con() as con:
            con.execute("UPDATE particle_labels SET t=?", (t,))
    return st


def test_fresh_folder(tmp_path):
    S = workflow_state(tmp_path / "nothing")
    assert S["detect"].state == "needed"
    assert all(S[k].state == "blocked" for k in ("label", "classify", "results"))
    assert next_step(S) == "detect"


def test_detected_only(tmp_path):
    S = workflow_state(_detected(tmp_path / "r"))
    assert S["detect"].state == "done" and "3 particles" in S["detect"].detail
    assert S["classify"].state == "needed" and "preview" in S["classify"].detail
    assert S["label"].state == "needed" and "library" in S["label"].reason       # no store yet
    assert S["train"].state == "blocked" and S["results"].state == "blocked"
    assert next_step(S) == "classify"                  # the preview is enough to start labelling cells


def test_classified_with_preview_is_done(tmp_path):
    d = _classified(_detected(tmp_path / "r"))
    st = _store(tmp_path)
    S = workflow_state(d, st)
    assert S["classify"].state == "done" and "preview" in S["classify"].detail
    assert S["results"].state == "done" and "1 one cell" in S["results"].detail
    assert S["label"].state == "needed" and f"/ {MIN_TRAIN} training labels" in S["label"].reason
    assert S["train"].state == "blocked" and not S["train"].can_run
    assert next_step(S) == "label"


def test_enough_labels_ask_for_a_model(tmp_path):
    d = _classified(_detected(tmp_path / "r"))
    S = workflow_state(d, _store(tmp_path, n_train=MIN_TRAIN))
    assert S["label"].state == "done" and S["train"].state == "needed" and S["train"].can_run
    assert next_step(S) == "train"


def test_classified_with_the_chosen_model(tmp_path):
    m = _model()
    d = _classified(_detected(tmp_path / "r"), model=m.spec)
    S = workflow_state(d, _store(tmp_path, n_train=MIN_TRAIN, t=T0), m)
    assert S["train"].state == "done" and S["classify"].state == "done"
    assert next_step(S) is None


def test_other_model_chosen_makes_classify_stale(tmp_path):
    d = _classified(_detected(tmp_path / "r"), model=_model(fitted=T0).spec)
    S = workflow_state(d, _store(tmp_path, n_train=MIN_TRAIN, t=T0), _model(fitted=T2))
    assert S["classify"].state == "stale" and "not the one used" in S["classify"].reason
    assert S["results"].state == "stale"
    assert next_step(S) == "classify"


def test_labels_newer_than_the_model_make_train_stale(tmp_path):
    m = _model(fitted=T1)
    d = _classified(_detected(tmp_path / "r"), model=m.spec)
    S = workflow_state(d, _store(tmp_path, n_train=STALE_LABELS, t=T2), m)
    assert S["train"].state == "stale" and f"{STALE_LABELS} new" in S["train"].reason
    few = workflow_state(d, _store(tmp_path / "x", n_train=STALE_LABELS - 1, t=T2), m)
    assert few["train"].state == "done" and "new labels since" in few["train"].detail


def test_legacy_timestamp_formats_compare(tmp_path):
    m = _model(fitted=T1)
    d = _classified(_detected(tmp_path / "r"), model=m.spec)
    st = _store(tmp_path, n_train=STALE_LABELS, t="2026-10-01 14:00:00+02:00")        # 12:00 UTC, after T1
    assert workflow_state(d, st, m)["train"].state == "stale"
    with st._con() as con:
        con.execute("UPDATE particle_labels SET t='not a time'")
    assert workflow_state(d, st, m)["train"].state == "done"                       # unreadable: never counted


def test_detect_again_removes_classify(tmp_path):
    d = _classified(_detected(tmp_path / "r"))
    run = json.loads((d / "run.json").read_text())
    run["detected"] = T2                                 # Detect ran after Classify and removed its outputs
    (d / "run.json").write_text(json.dumps(run))
    for f in ("calls.csv", "cells.csv"):
        (d / f).unlink()
    S = workflow_state(d)
    assert S["classify"].state == "needed" and "Detect ran again" in S["classify"].reason


def test_crash_between_removal_and_run_json(tmp_path):
    """Detect removed the classify outputs, then died before writing run.json: the old run.json still says
    classified, the job record says Detect never finished."""
    d = _classified(_detected(tmp_path / "r"))
    for f in ("calls.csv", "cells.csv"):
        (d / f).unlink()
    record_job(d, "detect", "started")
    S = workflow_state(d)
    assert S["detect"].state == "incomplete" and "did not finish" in S["detect"].reason
    assert S["classify"].state == "blocked" and not S["classify"].can_run
    assert next_step(S) == "detect"


def test_interrupted_detect_blocks_later_steps(tmp_path):
    d = _classified(_detected(tmp_path / "r"))
    record_job(d, "detect", "started")
    S = workflow_state(d)
    assert S["detect"].state == "incomplete"
    assert S["classify"].state == "blocked" and S["results"].state == "blocked"


def test_failed_and_finished_jobs(tmp_path):
    d = _detected(tmp_path / "r")
    record_job(d, "classify", "failed", "ValueError: boom")
    S = workflow_state(d)
    assert S["classify"].state == "failed" and "boom" in S["classify"].reason
    assert next_step(S) == "classify"
    record_job(d, "classify", "finished")
    assert workflow_state(d)["classify"].state == "needed"


def test_pipeline_records_failures(tmp_path):
    from correlative_imaging.reconstruction.pipeline import classify_run
    d = tmp_path / "r"
    d.mkdir()
    with pytest.raises(Exception):
        classify_run(d)                                  # no run.json: fails, and says so
    assert workflow_state(d)["classify"].state == "failed"


def test_edited_wells_make_classify_stale(tmp_path):
    d = _classified(_detected(tmp_path / "r"), edited_wells=["P1_A01"])
    S = workflow_state(d)
    assert S["classify"].state == "stale" and "hand-edited" in S["classify"].reason


def test_cell_labels_after_classify_make_it_stale(tmp_path):
    d = _classified(_detected(tmp_path / "r"), t=T0)
    st = _store(tmp_path)
    st.set_cell("ds", "P1", "A01", 5, 5, verdict="one cell", colour="c1")
    S = workflow_state(d, st)
    assert S["classify"].state == "stale" and "cell label" in S["classify"].reason
    st2 = _store(tmp_path / "x", test=(("P1", "A01"),))   # judged in a test well: never calibrates
    st2.set_cell("ds", "P1", "A01", 5, 5, verdict="one cell", colour="c1")
    assert workflow_state(d, st2)["classify"].state == "done"


def test_old_version_results_never_raise(tmp_path):
    d = _detected(tmp_path / "r", t=None)
    S = workflow_state(d)
    assert S["detect"].state == "unknown" and S["classify"].state == "needed"
    _classified(d, t=None)
    S = workflow_state(d)
    assert S["classify"].state == "unknown" and S["results"].state == "done"
    (d / "run.json").write_text("{not json")
    assert workflow_state(d)["detect"].state == "needed"


def test_running_overrides_and_summary(tmp_path):
    d = _detected(tmp_path / "r")
    S = workflow_state(d, running={"classify"})
    assert S["classify"].state == "running"
    text = summary(S)
    assert "Detect" in text and "running" in text


def test_a_later_classify_never_hides_an_unfinished_detect(tmp_path):
    d = _classified(_detected(tmp_path / "r"))
    record_job(d, "detect", "failed", "OSError: share offline")
    record_job(d, "classify", "started")
    record_job(d, "classify", "finished")
    S = workflow_state(d)
    assert S["detect"].state == "failed" and "share offline" in S["detect"].reason
    for k in ("label", "train", "classify", "results"):
        assert S[k].state == "blocked" and not S[k].can_run and "Detect did not finish" in S[k].reason
    assert next_step(S) == "detect"
    record_job(d, "detect", "finished")
    assert workflow_state(d)["classify"].state == "done"


def test_cancelled_detect_is_unfinished_not_failed(tmp_path):
    d = _detected(tmp_path / "r")
    record_job(d, "detect", "failed", "RuntimeError: Detect cancelled")
    assert workflow_state(d)["detect"].state == "incomplete"


def test_old_single_record_job_file(tmp_path):
    d = _detected(tmp_path / "r")
    (d / "job.json").write_text(json.dumps(dict(step="classify", status="failed", t=T1, error="boom")))
    assert workflow_state(d)["classify"].state == "failed"
    record_job(d, "train", "failed", "x")                    # converted, the classify record kept
    S = workflow_state(d)
    assert S["classify"].state == "failed" and S["train"].state == "failed"
    (d / "job.json").write_text("[1, 2]")                    # garbage: ignored
    assert workflow_state(d)["classify"].state == "needed"
