"""ci gui workflow status (viewer/workflow.py): a session snapshot + hand-made output folders."""
from __future__ import annotations

import json

from correlative_imaging.viewer.workflow import changed_since, ci_workflow_state, find_runs, next_step

PL = {"name": "pipeline", "steps": [{"type": "AutoThreshold", "channel": 0, "method": "otsu"}],
      "channel_names": ["GFP", "RFP"], "per_well_selections": [], "channel_colors": ["green", "red"]}


def _snap(out, **kw):
    s = dict(input_dir="in", output_dir=str(out), n_plates=1, n_wells=10, n_complete=10, channels=["GFP", "RFP"],
             channel_source="well B3", selections=[("whole", "whole")], needs_bf=False, n_roi_ok=0,
             pipeline_now=dict(PL), running=False)
    s.update(kw)
    return s


def _run(folder, tag, end="Done ✓   Results → x.db", db=True, pl=PL, log_dir=None):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{tag}_pipeline.json").write_text(json.dumps(pl))
    if db:
        (folder / f"{tag}.db").write_bytes(b"")
    if end is not None:
        (log_dir or folder).mkdir(parents=True, exist_ok=True)
        (log_dir or folder).joinpath(f"{tag}.log").write_text(f"Starting batch\n{end}\n", encoding="utf-8")


def test_nothing_scanned(tmp_path):
    S = ci_workflow_state(dict(input_dir="", output_dir=str(tmp_path / "missing")))
    assert S["scan"].state == "needed" and S["channels"].state == "blocked" and S["run"].state == "blocked"
    assert next_step(S) == "scan"


def test_scanned_without_channels(tmp_path):
    S = ci_workflow_state(_snap(tmp_path, channels=[], selections=[]))
    assert S["scan"].state == "done" and S["channels"].state == "needed"
    assert S["selections"].state == "blocked" and S["run"].state == "blocked" and not S["run"].can_run
    assert next_step(S) == "channels"


def test_ready_to_run_and_session_note(tmp_path):
    S = ci_workflow_state(_snap(tmp_path))
    assert S["run"].state == "needed" and S["run"].can_run
    assert "this session" in S["channels"].detail
    assert next_step(S) == "run"


def test_hole_rois_never_block(tmp_path):
    assert ci_workflow_state(_snap(tmp_path))["hole"].state == "not_used"
    S = ci_workflow_state(_snap(tmp_path, needs_bf=True, n_roi_ok=4))
    assert S["hole"].state == "needed" and "6 missing" in S["hole"].reason
    assert next_step(S) == "run"                                    # the run makes them itself
    assert ci_workflow_state(_snap(tmp_path, needs_bf=True, n_roi_ok=10))["hole"].state == "done"


def test_finished_run_one_plate(tmp_path):
    _run(tmp_path, "exp_20261021_101010")
    S = ci_workflow_state(_snap(tmp_path))
    assert S["run"].state == "done" and "exp_20261021_101010" in S["run"].detail
    assert S["results"].state == "needed" and next_step(S) == "results"


def test_several_plates_log_in_the_output_folder(tmp_path):
    for p in ("plateA", "plateB"):
        _run(tmp_path / p, "exp_20261021_101010", end=None)
    (tmp_path / "exp_20261021_101010.log").write_text("Done ✓   Results → a.db;b.db\n", encoding="utf-8")
    runs = find_runs(tmp_path)
    assert len(runs) == 1 and len(runs[0].dbs) == 2 and runs[0].end == "done"
    assert runs[0].base == tmp_path                 # several plates: reconstruction_<tag> in the output folder
    assert ci_workflow_state(_snap(tmp_path))["run"].state == "done"


def test_log_endings_and_no_log(tmp_path):
    _run(tmp_path / "a", "exp_20261021_101010", end="Finished (with errors).")
    assert ci_workflow_state(_snap(tmp_path / "a"))["run"].state == "failed"
    _run(tmp_path / "b", "exp_20261021_101010", end="well B03 …")
    assert ci_workflow_state(_snap(tmp_path / "b"))["run"].state == "incomplete"
    _run(tmp_path / "c", "exp_20261021_101010", end=None)
    assert ci_workflow_state(_snap(tmp_path / "c"))["run"].state == "unknown"          # older version: no log


def test_latest_run_by_its_timestamp(tmp_path):
    _run(tmp_path, "exp_20261021_101010", end="Finished (with errors).")
    _run(tmp_path, "exp_20261022_090000")
    S = ci_workflow_state(_snap(tmp_path))
    assert S["run"].state == "done" and "20261022" in S["run"].detail


def test_changed_set_up_makes_run_stale_colours_do_not(tmp_path):
    _run(tmp_path, "exp_20261021_101010", pl={k: v for k, v in PL.items() if k != "channel_colors"})
    assert ci_workflow_state(_snap(tmp_path))["run"].state == "done"                  # colours never count
    other = dict(PL, steps=[{"type": "AutoThreshold", "channel": 0, "method": "li"}])
    S = ci_workflow_state(_snap(tmp_path, pipeline_now=other))
    assert S["run"].state == "stale" and next_step(S) == "run"
    assert changed_since(find_runs(tmp_path)[-1], None) is None


def test_reconstruction_state(tmp_path):
    out = tmp_path / "output"                       # one plate: reconstruction_<tag> sits beside the output folder
    _run(out, "exp_20261021_101010")
    rec = tmp_path / "reconstruction_exp_20261021_101010"
    rec.mkdir()
    (rec / "run.json").write_text("{}")
    assert "not classified" in ci_workflow_state(_snap(out))["results"].detail
    (rec / "calls.csv").write_text("x\n")
    S = ci_workflow_state(_snap(out))
    assert S["results"].state == "done" and next_step(S) is None


def test_running_and_missing_output_never_raise(tmp_path):
    assert ci_workflow_state(_snap(tmp_path, running=True))["run"].state == "running"
    assert ci_workflow_state(_snap(tmp_path / "does_not_exist"))["run"].state == "needed"
    _run(tmp_path, "exp_20261021_101010")
    (tmp_path / "exp_20261021_101010_pipeline.json").write_text("{broken")
    assert ci_workflow_state(_snap(tmp_path))["run"].state in ("done", "stale")
