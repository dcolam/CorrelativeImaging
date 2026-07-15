"""Discover finished pipeline runs under an output folder.

A batch run writes one SQLite ``.db`` per plate, each inside that plate's own
output subfolder, all sharing one *run tag* (the ``<experiment>_<timestamp>``
basename — see ``viewer.gui._run_basename``). So the databases of a single run
are spread across sibling plate folders, and one plate folder can hold the
databases of several runs. This module untangles that:

* :func:`discover_runs` walks a root, groups every ``.db`` by run tag, and
  returns one :class:`RunGroup` per run — its per-plate databases, the plate
  each belongs to, and (when present) that plate's ``diagnostics`` folder.

Everything here is plain ``pathlib`` + ``sqlite3`` — no Qt, no pandas — so it
can be run and tested headless.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

# The three required result tables. A file must contain all of them to count as
# a pipeline results database (guards against picking up unrelated .db files).
_REQUIRED_TABLES = {"images", "particle_measurements", "intensity_measurements"}

# Run-tag pattern: "<experiment>_<YYYYMMDD>_<HHMMSS>" (see _run_basename). The
# timestamp suffix is the stable part; the experiment prefix is free-form.
_RUN_TAG_RE = re.compile(r"^(?P<tag>.+_\d{8}_\d{6})(?:\.part\d+)?$")


@dataclass
class PlateResult:
    """One plate's database within a run."""
    plate: str          # plate identity = the db's parent folder name
    db_path: Path
    diagnostics_dir: Path | None = None   # <plate folder>/diagnostics, if it exists


@dataclass
class RunGroup:
    """All plate databases sharing one run tag."""
    run_tag: str
    plates: list[PlateResult] = field(default_factory=list)

    @property
    def n_plates(self) -> int:
        return len(self.plates)


def _is_results_db(path: Path) -> bool:
    """True if *path* is a SQLite file with the expected result tables."""
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error:
        return False
    try:
        names = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    except sqlite3.DatabaseError:
        return False   # not a valid SQLite file
    finally:
        con.close()
    return _REQUIRED_TABLES.issubset(names)


def _run_tag_of(db_path: Path) -> str:
    """Extract the run tag from a database filename, ignoring any ``.partN``
    sub-database suffix (parallel-batch shards). Falls back to the plain stem
    when the name doesn't match the timestamped convention."""
    m = _RUN_TAG_RE.match(db_path.stem)
    return m.group("tag") if m else db_path.stem


def discover_runs(root: str | Path) -> list[RunGroup]:
    """Find every pipeline results database under *root* and group by run tag.

    Recurses *root* for ``.db`` files, keeps only genuine result databases,
    skips ``.partN`` sub-databases (they're merged into the main db and would
    double-count), groups the rest by run tag, and attaches each database's
    plate identity (its parent folder name) and diagnostics folder.

    Returns runs sorted by tag (so the newest timestamp sorts last); each run's
    plates are sorted by name. An empty list means nothing was found.
    """
    root = Path(root)
    if not root.is_dir():
        return []

    groups: dict[str, RunGroup] = {}
    for db_path in sorted(root.rglob("*.db")):
        # Skip parallel-batch sub-DBs — the merged main db holds their rows.
        if re.search(r"\.part\d+$", db_path.stem):
            continue
        if not _is_results_db(db_path):
            continue
        tag = _run_tag_of(db_path)
        plate_dir = db_path.parent
        diag = plate_dir / "diagnostics"
        pr = PlateResult(
            plate=plate_dir.name,
            db_path=db_path,
            diagnostics_dir=diag if diag.is_dir() else None,
        )
        groups.setdefault(tag, RunGroup(run_tag=tag)).plates.append(pr)

    runs = sorted(groups.values(), key=lambda g: g.run_tag)
    for g in runs:
        g.plates.sort(key=lambda p: p.plate)
    return runs


def find_diagnostic_images(diagnostics_dir: Path | None, well_id: str) -> dict[str, Path]:
    """Return the diagnostic images on disk for one well, keyed by kind.

    Keys: ``"whole"`` (whole-image composite) and ``"<roi>_crop"`` for each
    per-ROI crop (e.g. ``"hole_crop"``, ``"background_crop"``). Prefers ``.jpg``
    for display, falling back to ``.tif``. Missing files are simply absent from
    the dict. Empty dict if *diagnostics_dir* is None or nothing matches.

    Filenames follow ``diagnostics._save_well_diagnostics``:
    ``<well>_whole.<ext>`` and ``<well>_<roi>_crop.<ext>``.
    """
    if diagnostics_dir is None or not diagnostics_dir.is_dir():
        return {}
    out: dict[str, Path] = {}
    for f in sorted(diagnostics_dir.glob(f"{well_id}_*")):
        stem = f.name[len(well_id) + 1:].rsplit(".", 1)[0]  # strip "<well>_" and ext
        if stem.startswith("particles_"):
            continue  # particle label maps aren't display composites
        # Prefer .jpg; only take .tif if no .jpg already recorded for this kind.
        if stem in out and out[stem].suffix.lower() == ".jpg":
            continue
        if f.suffix.lower() in (".jpg", ".jpeg", ".tif", ".tiff"):
            out[stem] = f
    return out
