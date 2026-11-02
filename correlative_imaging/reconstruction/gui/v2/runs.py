"""What a ci gui output folder holds for this tool (no Qt): every pipeline run, its Detect results and its first pass.

  scan(root)        one row per Detect results folder of each run, plus one row per run without Detect yet
  fresh_out(root, tag)   where a new Detect writes: reconstruction_<tag>, or with date and time when that exists
  carry_over(old, new)   your setup, answers and labels from an older Detect to a new one. Answers about single
                         particles ("cell or not": ids of Detect's objects) are left behind: a new Detect numbers its
                         objects anew, so they would point at other particles
"""
from __future__ import annotations

import datetime
import json
import shutil
from pathlib import Path

import pandas as pd

PARTICLE_QUESTIONS = {"particle"}          # answers keyed by Detect's object ids


def _when(p: Path) -> str:
    return f"{datetime.datetime.fromtimestamp(p.stat().st_mtime):%d.%m.%Y %H:%M}" if p.exists() else ""


def detect_folders(root: Path) -> list[dict]:
    out = []
    for d in sorted(Path(root).glob("reconstruction_*")):
        rj = d / "run.json"
        if not (d.is_dir() and rj.exists() and (d / "particles.parquet").exists()):
            continue
        try:
            r = json.loads(rj.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        tag = (r.get("source") or {}).get("run_tag") or r.get("dataset", "")
        fp = d / "unified" / "summary.json"
        out.append(dict(run_tag=tag, detect=str(d), detect_when=_when(d / "particles.parquet"),
                        n_wells=r.get("n_wells"), first_pass_when=_when(fp)))
    return out


def _runs(root: Path) -> dict:
    """{run tag: number of plates}: the plate databases ci gui writes (<root>/<plate>/run_<tag>.db, or in the root).
    Two folder levels only (a full search of a network share takes minutes); falls back to the full search."""
    import re
    from ..._discovery import _RUN_TAG_RE, discover_runs
    found = {}
    for db in list(root.glob("*.db")) + list(root.glob("*/*.db")):
        if re.search(r"\.part\d+$", db.stem) or not db.stem.startswith("run_"):
            continue
        m = _RUN_TAG_RE.match(db.stem)
        tag = m.group("tag") if m else db.stem
        found.setdefault(tag, set()).add(db.parent)
    if found:
        return {t: len(v) for t, v in found.items()}
    return {r.run_tag: r.n_plates for r in discover_runs(root)}


def scan(root) -> pd.DataFrame:
    """Rows: run_tag, n_plates, detect (folder or ""), detect_when, n_wells, first_pass_when, text."""
    root = Path(root)
    runs = _runs(root)
    rows = detect_folders(root)
    have = {r["run_tag"] for r in rows}
    for r in rows:
        r["n_plates"] = runs.get(r["run_tag"])
    for tag, n in runs.items():
        if tag not in have:
            rows.append(dict(run_tag=tag, n_plates=n, detect="", detect_when="", n_wells=None, first_pass_when=""))
    D = pd.DataFrame(rows, columns=["run_tag", "n_plates", "detect", "detect_when", "n_wells", "first_pass_when"])
    if not len(D):
        return D.assign(text=[])
    D["text"] = [
        f"{r.run_tag} · {int(r.n_plates) if pd.notna(r.n_plates) else '?'} plates · " +
        (f"Detect {r.detect_when}" + (f" ({Path(r.detect).name})" if Path(r.detect).name != f"reconstruction_{r.run_tag}"
                                      else "") + (f" · first pass {r.first_pass_when}" if r.first_pass_when
                                                  else " · no first pass yet")
         if r.detect else "no Detect yet") for r in D.itertuples()]
    return D.sort_values(["run_tag", "detect_when"]).reset_index(drop=True)


def fresh_out(root, tag) -> Path:
    base = Path(root) / f"reconstruction_{tag}"
    return base if not base.exists() else base.with_name(f"{base.name}_{datetime.datetime.now():%Y%m%d_%H%M}")


def carry_over(old_detect, new_detect) -> list[str]:
    """Copies <old>/unified/experiment.json and interventions (labels, answers without the particle answers)."""
    old, new = Path(old_detect) / "unified", Path(new_detect) / "unified"
    done = []
    if not old.exists():
        return done
    (new / "interventions").mkdir(parents=True, exist_ok=True)
    if (old / "experiment.json").exists():
        shutil.copy2(old / "experiment.json", new / "experiment.json")
        done.append("setup (experiment.json)")
    oi = old / "interventions"
    if (oi / "labels.csv").exists():
        shutil.copy2(oi / "labels.csv", new / "interventions" / "labels.csv")
        done.append("your labels")
    if (oi / "answers.csv").exists():
        A = pd.read_csv(oi / "answers.csv", dtype=str).fillna("")
        keep = A[~A.question.isin(PARTICLE_QUESTIONS)] if "question" in A else A
        keep.to_csv(new / "interventions" / "answers.csv", index=False)
        done.append(f"{len(keep)} answers ({len(A) - len(keep)} about single particles left behind: they refer to the "
                    "old Detect's objects)")
    return done
