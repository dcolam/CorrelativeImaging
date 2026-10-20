"""Where a results folder stands in the workflow, read from disk only (the GUI's strip and `status` on the command line
use this one function, so they always agree).

  Detect   particles, field reference, features, a first look per well (no model needed)
  Label    hand labels in the label library: frozen test wells, training labels, judged cells
  Train    a particle classifier fitted on the training labels
  Classify classes per particle, reconstructed cells, colour call per well (with a model, or the Detect preview)
  Results  the calls per well

Label, Train and Classify form a loop: Detect -> Classify (preview) -> Label -> Train + classify -> Label -> ...
Classify never waits for a model; a model only makes it better.

A step's state:
  done         its outputs exist and nothing they depend on changed since
  needed       not run yet (and nothing blocks it)
  stale        its outputs exist but an input changed afterwards (named in `detail`): run it again
  incomplete   it started but did not finish (cancelled or crashed): run it again
  blocked      an earlier step is missing (`reason` says which)
  running      a job is working on it (GUI only)
  failed       the last job for it raised an error (job.json in the results folder)
  unknown      the results were written by an older version that kept no record of it
"""
from __future__ import annotations

import datetime
import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

STEPS = ["detect", "label", "train", "classify", "results"]
TITLES = {"detect": "Detect", "label": "Label", "train": "Train", "classify": "Classify", "results": "Results"}
PURPOSE = {
    "detect": "finds the particles in and around each hole and measures them (no model needed)",
    "label": "you label particles and cells; test wells are labelled blind to score models fairly",
    "train": "learns the particle classes from your training labels",
    "classify": "classifies every particle, reconstructs the cell(s) per hole and calls their colour",
    "results": "one call per well: one cell / several cells / no cell, and its colour",
}
OUTPUTS = {
    "detect": ["run.json", "wells.csv", "reference.csv", "particles.parquet", "cache/", "diagnostics/"],
    "label": ["ci_labels.db (label library, beside the results)"],
    "train": ["models/<name>.joblib"],
    "classify": ["cells.csv", "calls.csv", "identity.json", "recon/"],
    "results": ["calls.csv"],
}
MIN_TRAIN = 30          # training labels before a model can be trained (the Retrain button's condition)
STALE_LABELS = 50       # new training labels since the model was fitted that make it worth retraining
JOB_FILE = "job.json"


@dataclass
class Step:
    key: str
    state: str
    detail: str = ""                    # one line: what is there (counts, which model, ...)
    reason: str = ""                    # why it is blocked / stale / failed
    can_run: bool = True                # the step's main action is possible now
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


def _ts(v):
    """A timestamp from run.json / the label library, or None. Naive and aware times both end up naive (UTC for
    aware ones), so legacy imports with a time zone compare without raising."""
    if v is None or v == "":
        return None
    t = pd.to_datetime(str(v), errors="coerce", utc=True)
    return None if pd.isna(t) else t.tz_localize(None)


def _ts_series(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s.astype(str), errors="coerce", utc=True, format="mixed").dt.tz_localize(None)


def jobs(results_dir) -> dict:
    """{step: {status, t, error}}: the last job of each step (job.json). An older single-record file is read as the
    record of its one step."""
    try:
        J = json.loads((Path(results_dir) / JOB_FILE).read_text())
    except (OSError, ValueError, TypeError):
        return {}
    if not isinstance(J, dict):
        return {}
    if "step" in J and "status" in J:
        return {J["step"]: J}
    return {k: v for k, v in J.items() if isinstance(v, dict)}


def record_job(results_dir, step: str, status: str, error: str = ""):
    """Remember the last job of one step (status: started / finished / failed). One record per step: a later job of
    another step never hides that this one did not finish."""
    d = Path(results_dir)
    if not d.exists():
        return
    J = jobs(d)
    J[step] = dict(status=status, t=datetime.datetime.now().isoformat(timespec="seconds"), error=str(error)[:2000])
    try:
        tmp = d / (JOB_FILE + ".tmp")
        tmp.write_text(json.dumps(J, indent=2))
        tmp.replace(d / JOB_FILE)
    except OSError:
        pass


def _unfinished(rec: dict) -> str | None:
    """'incomplete' (started, never finished; or cancelled), 'failed', or None."""
    st = (rec or {}).get("status")
    if st == "started" or (st == "failed" and "cancelled" in str(rec.get("error", "")).lower()):
        return "incomplete"
    return "failed" if st == "failed" else None


def _read_run(d: Path) -> dict | None:
    try:
        return json.loads((d / "run.json").read_text())
    except (OSError, ValueError):
        return None


def _training_labels(store, dataset: str, test: set) -> pd.DataFrame:
    L = store.frame(dataset)
    if not len(L):
        return L
    return L[[(p, w) not in test for p, w in zip(L.plate_id, L.well_id)]]


def workflow_state(results_dir, store=None, model=None, progress: dict | None = None,
                   running: set | None = None) -> dict:
    """{step: Step} for a results folder. store: the LabelStore (labels live beside the results, not in them);
    model: the model chosen for Classify (a ParticleModel, or None for the Detect preview); progress: a
    tiers.label_progress result if one is at hand (labels matched to particles, as the batches count them), else the
    library's rows are counted; running: steps a job is working on now."""
    d = Path(results_dir) if results_dir else None
    run = _read_run(d) if d is not None else None
    running = set(running or ())
    S = {}

    # -- Detect
    det_files = all((d / f).exists() for f in ("wells.csv", "reference.csv", "particles.parquet")) if d else False
    t_det = _ts((run or {}).get("detected"))
    if d is None or not d.exists():
        S["detect"] = Step("detect", "needed", "no results folder yet")
    elif run is None or not det_files:
        S["detect"] = Step("detect", "needed")
    else:
        detail = f"{run.get('n_wells', '?')} wells · {run.get('n_pool', '?')} particles in the holes"
        if t_det is None:
            S["detect"] = Step("detect", "unknown", detail, "written by an older version (no Detect time)")
        else:
            S["detect"] = Step("detect", "done", detail)
    detected = S["detect"].state in ("done", "unknown")
    dataset = (run or {}).get("dataset", "")

    # -- Label
    test = store.test_wells(dataset) if (store is not None and dataset) else set()
    if not detected:
        S["label"] = Step("label", "blocked", reason="run Detect first", can_run=False)
        n_train, Ltr = 0, pd.DataFrame()
    elif store is None:
        S["label"] = Step("label", "needed", reason="choose a label library (ci_labels.db, Label → Setup)")
        n_train, Ltr = 0, pd.DataFrame()
    else:
        Ltr = _training_labels(store, dataset, test)
        if progress:
            n_train = sum(n for n, _ in progress.get("classes", {}).values())
            n_test = progress.get("test", {}).get("labelled", 0)
        else:
            n_train = len(Ltr)
            L = store.frame(dataset)
            n_test = len(L) - n_train
        C = store.cell_frame(dataset)
        n_cells = len(C)
        detail = (f"{len(test)} test wells · {n_train} training labels · {n_test} test labels · "
                  f"{n_cells} cells judged")
        facts = dict(n_train=n_train, n_test=n_test, n_cells=n_cells, n_test_wells=len(test))
        if not test:
            S["label"] = Step("label", "needed", detail, "freeze the test wells first (Label → Setup)", facts=facts)
        elif n_train < MIN_TRAIN:
            S["label"] = Step("label", "needed", detail,
                              f"{n_train} / {MIN_TRAIN} training labels (needed before a model can be trained)", facts=facts)
        else:
            S["label"] = Step("label", "done", detail, facts=facts)
    n_train = S["label"].facts.get("n_train", 0)

    # -- Train
    spec = getattr(model, "spec", None) or {}
    real_model = bool(spec) and spec.get("arm") != "preview"
    if real_model:
        fitted = _ts(spec.get("fitted"))
        new = 0
        if fitted is not None and len(Ltr):
            new = int((_ts_series(Ltr.t) > fitted).sum())
        on = spec.get("datasets") or []
        detail = f"model '{spec.get('arm', '?')}' · {spec.get('n_train', '?')} labels" + \
                 (f" · trained on {', '.join(on)}" if on else "")
        facts = dict(new_labels=new, model_fitted=spec.get("fitted"))
        if new >= STALE_LABELS:
            S["train"] = Step("train", "stale", detail, f"{new} new training labels since this model was trained",
                              can_run=detected and n_train >= MIN_TRAIN, facts=facts)
        else:
            S["train"] = Step("train", "done", detail + (f" · {new} new labels since" if new else ""),
                              can_run=detected and n_train >= MIN_TRAIN, facts=facts)
    elif not detected:
        S["train"] = Step("train", "blocked", reason="run Detect first", can_run=False)
    elif n_train < MIN_TRAIN:
        S["train"] = Step("train", "blocked", "no model: Classify uses the Detect preview",
                          f"needs {MIN_TRAIN} training labels ({n_train} so far)", can_run=False)
    else:
        S["train"] = Step("train", "needed", "no model yet: Classify uses the Detect preview")

    # -- Classify
    t_cls = _ts((run or {}).get("classified"))
    have_calls = d is not None and (d / "calls.csv").exists()
    used = (run or {}).get("model") or {}
    if not detected:
        S["classify"] = Step("classify", "blocked", reason="run Detect first", can_run=False)
    elif not have_calls:
        lost = t_cls is not None and t_det is not None and t_cls < t_det
        S["classify"] = Step("classify", "needed", "with " + ("the chosen model" if real_model else
                                                               "the Detect preview (no model yet)"),
                             "Detect ran again since the last Classify" if lost else "")
    else:
        used_txt = "the Detect preview" if used.get("arm") == "preview" else f"model '{used.get('arm', '?')}'"
        detail = f"classified with {used_txt}"
        why = []
        if t_cls is None:
            why.append("no Classify time recorded (older version)")
        elif t_det is not None and t_cls < t_det:
            why.append("Detect ran again afterwards")
        if real_model and used.get("fitted") != spec.get("fitted"):
            why.append("the chosen model is not the one used")
        if not real_model and used and used.get("arm") != "preview":
            why.append("no model chosen now, but it was classified with one")
        ed = (run or {}).get("edited_wells") or []
        if ed:
            why.append(f"{len(ed)} hand-edited well(s) re-classified alone (colour calibration not refitted)")
        if store is not None and dataset and t_cls is not None:
            C = store.cell_frame(dataset)
            if len(C):
                C = C[[(p, w) not in test for p, w in zip(C.plate_id, C.well_id)]]
                k = int((_ts_series(C.t) > t_cls).sum()) if len(C) else 0
                if k:
                    why.append(f"{k} cell label(s) since (they calibrate the colour call)")
        state = "unknown" if (t_cls is None and len(why) == 1) else ("stale" if why else "done")
        S["classify"] = Step("classify", state, detail, "; ".join(why))

    # -- Results
    if S["classify"].state in ("done", "stale", "unknown"):
        try:
            calls = pd.read_csv(d / "calls.csv")
            st = calls["status"].value_counts().to_dict() if "status" in calls else {}
            unc = int((calls["call"].astype(str) == "uncertain").sum()) if "call" in calls else 0
            detail = " · ".join(f"{v} {k}" for k, v in st.items()) + (f" · {unc} uncertain" if unc else "")
            S["results"] = Step("results", S["classify"].state if S["classify"].state != "unknown" else "done",
                                detail or f"{len(calls)} wells",
                                "Classify is out of date" if S["classify"].state == "stale" else "")
        except Exception as e:                                      # unreadable calls.csv
            S["results"] = Step("results", "incomplete", reason=f"calls.csv unreadable: {e}")
    else:
        S["results"] = Step("results", "blocked", reason="run Classify first", can_run=False)

    # -- jobs that did not finish (job.json, one record per step)
    J = jobs(d) if d is not None else {}
    for k in ("detect", "train", "classify"):
        bad = _unfinished(J.get(k))
        if bad is None:
            continue
        rec = J[k]
        S[k].state = bad
        S[k].reason = (f"{TITLES[k]} started {rec.get('t', '')} and did not finish (cancelled, crashed, or still "
                       "running in another window): run it again" if bad == "incomplete" else
                       f"the last run failed: {rec.get('error', '')}")
        S[k].can_run = True
    bad_det = _unfinished(J.get("detect"))
    if bad_det:                                   # the cache may no longer match the particles: Detect first
        why = "Detect did not finish: run Detect again first"
        for j in ("label", "train", "classify", "results"):
            S[j].can_run = False
            S[j].reason = why
            if S[j].state in ("done", "stale", "unknown", "needed"):
                S[j].state = "blocked"
    for k in running:
        if k in S:
            S[k].state = "running"
    return S


def next_step(S: dict) -> str | None:
    """The step to do next: Detect, then a first Classify (the preview is enough to label cells), then labelling,
    a model, and Classify again whenever it is out of date. None: everything is done."""
    bad = ("needed", "stale", "incomplete", "failed")
    if S["detect"].state in bad:
        return "detect"
    if S["classify"].state in ("needed", "incomplete", "failed"):
        return "classify"
    if S["label"].state in bad:
        return "label"
    if S["train"].state in bad:
        return "train"
    if S["classify"].state == "stale":
        return "classify"
    return None


def summary(S: dict) -> str:
    """Plain text (the `status` command)."""
    out = []
    nxt = next_step(S)
    for k in STEPS:
        s = S[k]
        line = f"{'>' if k == nxt else ' '} {s.title:<9} {s.state:<10} {s.detail}"
        if s.reason:
            line += f"\n{'':13}{s.reason}"
        out.append(line)
    # plain ASCII: any console can print it (a Windows console cannot print "→")
    return "\n".join(out).replace(" · ", " | ").replace("→", "->").replace("–", "-")
