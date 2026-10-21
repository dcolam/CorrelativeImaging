"""Which label library a results folder uses, decided once per folder and written into it (library.json).

  own      <results>/ci_labels.db: labels, test wells, hand edits and cell judgements of this results folder only.
           Every folder Detect creates from now on gets its own library, so a new run starts clean (no labels, no
           hand edits, no test wells) even when an older folder of the same dataset (run tag) sits next to it.
  shared   a library outside the folder, e.g. the project library next to the results folders (older layout:
           <output>/ci_labels.db, which the Results Explorer also uses for its channel names). Kept for folders that
           already rely on it.

An existing folder without a decision gets it once: keep the shared library beside it when that holds labels of its
dataset (the default: nothing is lost), start clean, or copy this dataset's rows into an own library. Libraries
outside the folder are only ever probed and copied read-only (no schema is added to them).
"""
from __future__ import annotations

import datetime
import json
import sqlite3
from pathlib import Path

DECISION = "library.json"
OWN = "ci_labels.db"
TABLES = ["particle_labels", "test_wells", "particle_cuts", "particle_merges", "particle_splits",
          "particle_watersheds", "unsure_particles", "cell_labels", "well_verdicts"]
EDIT_TABLES = ["particle_cuts", "particle_merges", "particle_splits", "particle_watersheds"]


def own_path(results) -> Path:
    return Path(results) / OWN


def beside_path(results) -> Path:
    return Path(results).parent / OWN


def _ro(path: Path):
    from urllib.request import pathname2url
    return sqlite3.connect(f"file:{pathname2url(str(Path(path).resolve()))}?mode=ro&immutable=1", uri=True)


def probe(path, dataset: str) -> dict:
    """What a library holds for one dataset (read-only): {labels, test_wells, edits, cells}; {} when there is no file,
    {"error": ...} when it exists but cannot be read (then the folder is asked, never silently given a new one)."""
    p = Path(path)
    if not p.exists():
        return {}
    try:
        con = _ro(p)
        have = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
        n = lambda t: (con.execute(f"select count(*) from {t} where dataset=?", (dataset,)).fetchone()[0]
                       if t in have else 0)
        out = dict(labels=n("particle_labels"), test_wells=n("test_wells"), cells=n("cell_labels"),
                   edits=sum(n(t) for t in EDIT_TABLES))
        con.close()
        return out
    except sqlite3.Error as e:
        return {"error": str(e)}


def has_rows(info: dict) -> bool:
    """Rows for the dataset, or unreadable (counts as possibly holding them)."""
    return bool(info) and ("error" in info or any(v > 0 for v in info.values() if isinstance(v, int)))


def decision(results) -> dict | None:
    try:
        return json.loads((Path(results) / DECISION).read_text())
    except (OSError, ValueError):
        return None


def remember(results, kind: str, path=None):
    """Write the folder's decision: kind "own" or "shared" (with the library's path)."""
    d = dict(kind=kind, path=str(Path(path).resolve()) if path else None,
             t=datetime.datetime.now().isoformat(timespec="seconds"))
    Path(results).mkdir(parents=True, exist_ok=True)
    (Path(results) / DECISION).write_text(json.dumps(d, indent=2))


def resolve(results, dataset: str) -> dict:
    """{path, kind ("own" / "shared"), decided (bool), shared (probe of the library beside, if it holds this
    dataset)}. Not decided = the folder needs the one-time choice (an existing folder whose dataset has rows in the
    library beside it); its suggested default is to keep that shared library."""
    d = decision(results)
    if d:
        if d.get("kind") == "shared" and d.get("path"):
            return dict(path=Path(d["path"]), kind="shared", decided=True, shared=None)
        return dict(path=own_path(results), kind="own", decided=True, shared=None)
    if own_path(results).exists():
        return dict(path=own_path(results), kind="own", decided=True, shared=None)
    b = beside_path(results)
    info = probe(b, dataset)
    if has_rows(info):
        return dict(path=b, kind="shared", decided=False, shared=info)
    return dict(path=own_path(results), kind="own", decided=True, shared=None)


def copy_dataset(src, dst, dataset: str) -> dict:
    """Copy one dataset's rows of every table from src (read-only) into dst (created with the full schema)."""
    from .labels import LabelStore
    LabelStore(dst)                                         # schema
    counts = {}
    s = _ro(Path(src))
    have = {r[0] for r in s.execute("select name from sqlite_master where type='table'")}
    with sqlite3.connect(Path(dst)) as d:
        for t in TABLES:
            if t not in have:
                continue
            cols_src = [r[1] for r in s.execute(f"pragma table_info({t})")]
            cols_dst = {r[1] for r in d.execute(f"pragma table_info({t})")}
            cols = [c for c in cols_src if c in cols_dst and c != "cut_id"]
            rows = s.execute(f"select {', '.join(cols)} from {t} where dataset=?", (dataset,)).fetchall()
            if rows:
                d.executemany(f"insert or ignore into {t} ({', '.join(cols)}) values ({', '.join('?' * len(cols))})",
                              rows)
            counts[t] = len(rows)
    s.close()
    return counts
