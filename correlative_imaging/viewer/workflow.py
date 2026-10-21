"""Where ci gui stands in its workflow (the strip on top of the main panel). No Qt here: the GUI passes a snapshot of
the session (what was scanned and set up; most of it lives only in this session until a run writes it), the run
outputs are read from the output folder.

  Scan        the wells of the input folder (BF + FL files, ROI files)
  Channels    each channel's preprocessing and threshold (from a sample well or a loaded pipeline JSON)
  Hole ROIs   the brightfield pipeline's hole ROI per well (only when a selection uses it; a run makes missing ones)
  Selections  the regions that are measured (whole image, the hole, imported ROIs)
  Run         the pipeline on every well: <tag>.db, <tag>_pipeline.json, <tag>.log, diagnostics/
  Results     the Results Explorer and Cell reconstruction on the latest run

States as in the reconstruction (done / needed / stale / incomplete / failed / blocked / running / unknown), plus
not_used (a step this set-up does not need).
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

STEPS = ["scan", "channels", "hole", "selections", "run", "results"]
TITLES = {"scan": "Scan", "channels": "Channels", "hole": "Hole ROIs", "selections": "Selections", "run": "Run",
          "results": "Results"}
PURPOSE = {
    "scan": "finds the wells in the input folder: brightfield + fluorescence files and ROI files",
    "channels": "sets how each channel is cleaned up and thresholded (Preprocessing and Channels tabs)",
    "hole": "finds the hole in each brightfield image and saves it as an ROI (BF Pipeline tab)",
    "selections": "chooses the regions that are measured: the whole image, the hole, imported ROIs",
    "run": "runs the pipeline on every well and writes the results database",
    "results": "explore the results, and reconstruct the cell in each hole (Cell reconstruction)",
}
OUTPUTS = {
    "scan": [],
    "channels": ["saved with the run: <tag>_pipeline.json"],
    "hole": ["bf_pipeline/rois/<well>_<class>.roi"],
    "selections": ["saved with the run: <tag>_pipeline.json"],
    "run": ["<tag>.db", "<tag>_pipeline.json", "<tag>.log", "diagnostics/"],
    "results": ["reconstruction_<tag>/"],
}
ACTIONS = {"scan": "Scan folder", "channels": "Set up channels", "hole": "Open BF pipeline",
           "selections": "Edit selections", "run": "Go to Run", "results": "Cell reconstruction…"}
DONE_LINE, ERROR_LINE = "Done ✓", "Finished (with errors)"
# what changes the results (colours, the BF-pipeline config and the note do not)
COMPARED = ("steps", "channel_names", "per_well_selections")
_TAG_RE = re.compile(r"_(\d{8}_\d{6})$")


@dataclass
class Step:
    key: str
    state: str
    detail: str = ""
    reason: str = ""
    can_run: bool = True
    facts: dict = field(default_factory=dict)

    @property
    def title(self) -> str:
        return TITLES[self.key]

    @property
    def purpose(self) -> str:
        return PURPOSE[self.key]

    @property
    def outputs(self) -> list:
        return OUTPUTS[self.key]


# ------------------------------------------------------------------------------------------------------------------
# runs on disk

@dataclass
class Run:
    tag: str
    jsons: list                   # one <tag>_pipeline.json per plate
    dbs: list                     # the <tag>.db that exist
    log: Path | None
    end: str                      # "done" / "errors" / "open" (log without an end line) / "no log"
    base: Path                    # where reconstruction_<tag> goes: the folder above the plate folders (as ci gui's
                                  # _run_of_dbs: one plate = the output folder's parent)

    @property
    def stamp(self) -> str:
        m = _TAG_RE.search(self.tag)
        return m.group(1) if m else ""


def _pipeline_jsons(folder: Path, max_depth: int = 2) -> list:
    """*_pipeline.json at most max_depth folders below (diagnostics and VSI companion folders skipped)."""
    out = []
    root = str(folder)
    base = root.rstrip("/\\").count(os.sep)
    for dirpath, dirnames, filenames in os.walk(root):
        depth = dirpath.count(os.sep) - base
        dirnames[:] = [] if depth >= max_depth else [
            d for d in dirnames if d not in ("diagnostics", "bf_pipeline", "cache", "recon")
            and not d.startswith(("reconstruction_", "identity_", ".")) and not (d.startswith("_") and d.endswith("_"))]
        out += [Path(dirpath) / f for f in filenames if f.endswith("_pipeline.json")]
    return out


def _log_end(path: Path) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            n = f.tell()
            f.seek(max(0, n - 4096))
            tail = f.read().decode("utf-8", errors="replace")
    except OSError:
        return "no log"
    if DONE_LINE in tail:
        return "done"
    if ERROR_LINE in tail:
        return "errors"
    return "open"


def find_runs(output_dir) -> list:
    """The runs in an output folder (one plate: everything in it; several: a sub-folder per plate, the log in the
    output folder), oldest first."""
    d = Path(output_dir) if output_dir else None
    if d is None or not d.is_dir():
        return []
    by_tag = {}
    for j in _pipeline_jsons(d):
        tag = j.name[:-len("_pipeline.json")]
        by_tag.setdefault(tag, []).append(j)
    runs = []
    for tag, jsons in by_tag.items():
        dbs = [j.parent / f"{tag}.db" for j in jsons if (j.parent / f"{tag}.db").exists()]
        log = next((p for p in [d / f"{tag}.log"] + [j.parent / f"{tag}.log" for j in jsons] if p.exists()), None)
        root = Path(os.path.commonpath([str(p.parent.parent) for p in (dbs or jsons)]))
        runs.append(Run(tag, sorted(jsons), dbs, log, _log_end(log) if log else "no log", root))
    runs.sort(key=lambda r: (r.stamp, max((j.stat().st_mtime for j in r.jsons), default=0)))
    return runs


def _compared(pl: dict) -> str:
    return json.dumps({k: pl.get(k) for k in COMPARED}, sort_keys=True, default=str)


def changed_since(run: Run, pipeline_now: dict | None) -> bool | None:
    """Whether the set-up on screen would give other results than this run (None: unknown)."""
    if not pipeline_now or not run.jsons:
        return None
    try:
        saved = json.loads(run.jsons[0].read_text())
    except (OSError, ValueError):
        return None
    return _compared(saved) != _compared(pipeline_now)


# ------------------------------------------------------------------------------------------------------------------
# the state

def ci_workflow_state(snap: dict) -> dict:
    """{step: Step}. snap (from the GUI): input_dir, output_dir, n_plates, n_wells, n_complete, channels (names),
    channel_source ("pipeline X" / "well B3" / ""), selections ([(label, source)]), needs_bf, bf_classes, n_roi_ok
    (wells with every BF class ROI), pipeline_now (build_pipeline_dict), running (bool), runs (find_runs result, may
    be passed to avoid reading the folder again)."""
    S = {}
    n_wells = int(snap.get("n_wells") or 0)
    scanned = n_wells > 0
    session = " (this session; saved with the next run)"

    # Scan
    if not snap.get("input_dir"):
        S["scan"] = Step("scan", "needed", "no input folder", "choose the input / plate folder (Setup tab)")
    elif not scanned:
        S["scan"] = Step("scan", "needed", str(snap["input_dir"]), "scan the folder (Setup tab)")
    else:
        S["scan"] = Step("scan", "done", f"{snap.get('n_plates', 1)} plate(s) · {n_wells} wells · "
                                         f"{snap.get('n_complete', 0)} with BF + FL")

    # Channels
    chans = list(snap.get("channels") or [])
    if not scanned:
        S["channels"] = Step("channels", "blocked", reason="scan first", can_run=False)
    elif not chans:
        S["channels"] = Step("channels", "needed", reason="load a sample well or a pipeline JSON")
    else:
        src = snap.get("channel_source") or ""
        S["channels"] = Step("channels", "done", f"{len(chans)} channels: {', '.join(chans)}" +
                             (f" · from {src}" if src else "") + session)

    # Hole ROIs
    needs_bf = bool(snap.get("needs_bf"))
    n_complete = int(snap.get("n_complete") or 0)
    n_ok = int(snap.get("n_roi_ok") or 0)
    if not scanned:
        S["hole"] = Step("hole", "blocked", reason="scan first", can_run=False)
    elif not needs_bf:
        S["hole"] = Step("hole", "not_used", "no selection uses the brightfield hole")
    elif n_complete and n_ok >= n_complete:
        S["hole"] = Step("hole", "done", f"hole ROIs for all {n_complete} wells")
    else:
        S["hole"] = Step("hole", "needed", f"hole ROIs for {n_ok} of {n_complete} wells",
                         f"the run makes the {n_complete - n_ok} missing ones itself (or run the BF pipeline first "
                         "to check them)")

    # Selections
    sels = list(snap.get("selections") or [])
    if not chans:
        S["selections"] = Step("selections", "blocked", reason="set up the channels first", can_run=False)
    elif not sels:
        S["selections"] = Step("selections", "needed", reason="add at least one region to measure")
    else:
        S["selections"] = Step("selections", "done", " · ".join(f"{l} ({src})" for l, src in sels) + session)

    # Run (from disk)
    runs = snap.get("runs")
    if runs is None:
        runs = find_runs(snap.get("output_dir"))
    last = runs[-1] if runs else None
    ready = scanned and chans and sels
    if snap.get("running"):
        S["run"] = Step("run", "running", "the pipeline is running")
    elif last is None:
        S["run"] = Step("run", "needed" if ready else "blocked", "no run in the output folder yet",
                        "" if ready else "scan, set up the channels and the selections first", can_run=bool(ready))
    else:
        detail = f"last run {last.tag} · {len(last.dbs)} database(s)"
        diff = changed_since(last, snap.get("pipeline_now")) if chans else None
        if last.end == "errors":
            S["run"] = Step("run", "failed", detail, "the last run finished with errors (see its log)")
        elif last.end == "open":
            S["run"] = Step("run", "incomplete", detail, "the last run did not finish (aborted, crashed, or still "
                                                         "running in another window): run again")
        elif not last.dbs:
            S["run"] = Step("run", "incomplete", detail, "the last run wrote no database")
        elif diff:
            S["run"] = Step("run", "stale", detail, "the set-up on screen differs from this run's (steps, channels "
                                                    "or selections): run again to apply it")
        else:
            S["run"] = Step("run", "done" if last.end == "done" else "unknown", detail,
                            "" if last.end == "done" else "written by an older version (no log)")
        S["run"].can_run = bool(ready)
        S["run"].facts = dict(tag=last.tag, base=str(last.base))

    # Results
    if last is None or not last.dbs:
        S["results"] = Step("results", "blocked", reason="run the pipeline first", can_run=False)
    else:
        rec = last.base / f"reconstruction_{last.tag}"
        if (rec / "calls.csv").exists():
            S["results"] = Step("results", "done", f"Cell reconstruction classified ({rec.name})")
        elif (rec / "run.json").exists():
            S["results"] = Step("results", "needed", f"particles detected ({rec.name}), not classified yet",
                                "open Cell reconstruction to label and classify")
        else:
            S["results"] = Step("results", "needed", f"results of {last.tag}",
                                "open Cell reconstruction (or the Results Explorer)")
        S["results"].facts = dict(tag=last.tag, base=str(last.base))
    return S


def next_step(S: dict) -> str | None:
    """Scan → Channels → Selections → (Hole ROIs, never blocking) → Run → Results."""
    bad = ("needed", "stale", "incomplete", "failed")
    for k in ("scan", "channels", "selections", "run", "results"):
        if S[k].state in bad:
            return k
    return None
