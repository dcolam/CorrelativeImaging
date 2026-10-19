"""Hand labels for particles, independent of any particular run of this tool.

A label is stored by position, not by particle id: (dataset, plate, well, channel, level, y, x) in frame coordinates
(the particle's centroid inside the hole). Rerunning detection with other settings changes particle ids, not
positions, so labels survive; :func:`match_labels` finds the particle a label belongs to.

Classes (keys 1-5 in the GUI):
  cell       a piece of a cell in the hole (the patched cell, or a cell sitting in an unrecorded hole)
  neighbour  signal of a cell next to the hole that reaches in
  merged     one particle holding two cells (e.g. the patched cell fused with a neighbour)
  debris     not a cell: plug, crystal, dirt (typically small and bright in several channels)
  noise      background, halo, haze

The store is a SQLite file (default ``ci_labels.db`` in the run's output folder, next to the Results Explorer's
well labels). Tables: ``particle_labels``, ``test_wells`` (the frozen held-out set, never trained on) and
``well_verdicts``.
"""
from __future__ import annotations

import datetime
import getpass
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ["cell", "neighbour", "merged", "debris", "noise"]
# carousel / first labelling session -> classes (None = not used for training)
LEGACY_MAP = {"patched cell": "cell", "maybe cell": "cell", "neighbour": "neighbour", "two fused cells": "merged",
              "background_noise": "noise", "halo / haze": "noise", "plug / debris": "debris",
              "debris/unsure": None}

_DDL = """
CREATE TABLE IF NOT EXISTS particle_labels (
    dataset TEXT, plate_id TEXT, well_id TEXT, channel TEXT, level TEXT, y REAL, x REAL,
    label TEXT, orig_label TEXT, source TEXT, labeller TEXT, t TEXT,
    PRIMARY KEY (dataset, plate_id, well_id, channel, level, y, x));
CREATE TABLE IF NOT EXISTS test_wells (dataset TEXT, plate_id TEXT, well_id TEXT, t TEXT,
    PRIMARY KEY (dataset, plate_id, well_id));
CREATE TABLE IF NOT EXISTS well_verdicts (dataset TEXT, plate_id TEXT, well_id TEXT, verdict TEXT, note TEXT,
    labeller TEXT, t TEXT, PRIMARY KEY (dataset, plate_id, well_id));
"""


def _now():
    return datetime.datetime.now().isoformat(timespec="seconds")


class LabelStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._con() as con:
            con.executescript(_DDL)

    def _con(self):
        return sqlite3.connect(self.path)

    # -- particle labels
    def set(self, dataset, plate_id, well_id, channel, level, y, x, label, source="hand", orig_label=None):
        if label not in CLASSES:
            raise ValueError(f"unknown class {label!r}")
        with self._con() as con:
            con.execute("INSERT OR REPLACE INTO particle_labels VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                        (dataset, plate_id, well_id, channel, level, round(float(y), 2), round(float(x), 2), label,
                         orig_label, source, getpass.getuser(), _now()))

    def remove(self, dataset, plate_id, well_id, channel, level, y, x):
        with self._con() as con:
            con.execute("DELETE FROM particle_labels WHERE dataset=? AND plate_id=? AND well_id=? AND channel=? "
                        "AND level=? AND y=? AND x=?",
                        (dataset, plate_id, well_id, channel, level, round(float(y), 2), round(float(x), 2)))

    def frame(self, dataset: str | None = None) -> pd.DataFrame:
        with self._con() as con:
            q, a = "SELECT * FROM particle_labels", ()
            if dataset:
                q, a = q + " WHERE dataset=?", (dataset,)
            return pd.read_sql(q, con, params=a)

    def counts(self) -> pd.DataFrame:
        d = self.frame()
        if not len(d):
            return pd.DataFrame(columns=["dataset"] + CLASSES)
        return d.pivot_table(index="dataset", columns="label", values="y", aggfunc="size", fill_value=0) \
                .reindex(columns=CLASSES, fill_value=0).reset_index()

    # -- frozen test wells
    def test_wells(self, dataset: str) -> set:
        with self._con() as con:
            return {(p, w) for p, w in con.execute("SELECT plate_id, well_id FROM test_wells WHERE dataset=?",
                                                    (dataset,))}

    def freeze_test_wells(self, dataset: str, frac: float = 0.2, seed: int = 0) -> set:
        """Pick the held-out wells of a dataset once (a fraction of its labelled wells); later calls return the
        same set. Wells labelled after freezing are never added, so the test set stays untouched by training."""
        have = self.test_wells(dataset)
        if have:
            return have
        d = self.frame(dataset)
        wells = sorted(set(zip(d.plate_id, d.well_id)))
        if len(wells) < 5:
            return set()
        rng = np.random.default_rng(seed)
        pick = [wells[i] for i in rng.choice(len(wells), max(1, int(round(frac * len(wells)))), replace=False)]
        with self._con() as con:
            con.executemany("INSERT OR IGNORE INTO test_wells VALUES (?,?,?,?)", [(dataset, p, w, _now()) for p, w in pick])
        return set(pick)

    # -- well verdicts
    def set_verdict(self, dataset, plate_id, well_id, verdict, note=""):
        with self._con() as con:
            con.execute("INSERT OR REPLACE INTO well_verdicts VALUES (?,?,?,?,?,?,?)",
                        (dataset, plate_id, well_id, verdict, note, getpass.getuser(), _now()))

    def verdicts(self, dataset: str | None = None) -> pd.DataFrame:
        with self._con() as con:
            q, a = "SELECT * FROM well_verdicts", ()
            if dataset:
                q, a = q + " WHERE dataset=?", (dataset,)
            return pd.read_sql(q, con, params=a)

    # -- import of the first labelling session
    def import_legacy(self, csv: str | Path, dataset: str, names: dict, wells: set | None = None) -> dict:
        """csv: plate_id, well_id, channel (display name), x_px, y_px, particle_label. `names` maps display names to
        channel keys. Only identity-neutral columns are read (no plate map, no ephys, no earlier call). Coordinates
        are those of the crops of the source the labels were made on; level is unknown ("any")."""
        d = pd.read_csv(csv, dtype={"plate_id": str})
        n0 = len(d)
        if wells is not None:
            d = d[[(p, w) in wells for p, w in zip(d.plate_id, d.well_id)]]
        n_wells = len(d)
        d = d.assign(label=d.particle_label.map(LEGACY_MAP), channel=d.channel.map(names))
        d = d[d.label.notna() & d.channel.notna()]
        rows = [(dataset, r.plate_id, r.well_id, r.channel, "any", round(float(r.y_px), 2), round(float(r.x_px), 2),
                 r.label, r.particle_label, "legacy", "", str(r.labelled_at) if "labelled_at" in d else _now())
                for r in d.itertuples()]
        with self._con() as con:
            con.executemany("INSERT OR REPLACE INTO particle_labels VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
        return dict(rows_in_file=n0, in_selected_wells=n_wells, imported=len(rows),
                    per_class=d.label.value_counts().to_dict())


def match_labels(labels: pd.DataFrame, particles: pd.DataFrame, wells: pd.DataFrame, tol_um: float = 2.0,
                 legacy_offset: float = 0.0) -> pd.DataFrame:
    """Attach each label to the pool particle(s) it refers to: same well and channel (and level, unless "any"),
    nearest centroid inside the hole within `tol_um`. With level "any" the nearest particle of each level is taken
    (one physical object seen at both thresholds); `weight` = 1 / number of particles a label went to.
    Returns one row per (label, particle) with the label columns + oid, level, dist_um, weight."""
    W = wells.set_index(["plate_id", "well_id"])
    pool = particles[particles.pool]
    out = []
    groups = {k: g for k, g in pool.groupby(["plate_id", "well_id", "ch"])}
    for r in labels.itertuples():
        g = groups.get((r.plate_id, r.well_id, r.channel))
        if g is None or (r.plate_id, r.well_id) not in W.index:
            continue
        w = W.loc[(r.plate_id, r.well_id)]
        y = g.iy.to_numpy() + w.off_r + legacy_offset
        x = g.ix.to_numpy() + w.off_c + legacy_offset
        d = np.hypot(y - r.y, x - r.x) * w.px_um
        cand = g.assign(dist_um=d)
        cand = cand[cand.dist_um <= tol_um]
        if r.level != "any":
            cand = cand[cand.level == r.level]
        if not len(cand):
            continue
        best = cand.sort_values("dist_um").drop_duplicates("level")
        for b in best.itertuples():
            out.append(dict(dataset=r.dataset, plate_id=r.plate_id, well_id=r.well_id, oid=b.oid, level=b.level,
                            label=r.label, orig_label=r.orig_label, source=r.source, dist_um=b.dist_um,
                            weight=1.0 / len(best), ly=r.y, lx=r.x))
    return pd.DataFrame(out, columns=["dataset", "plate_id", "well_id", "oid", "level", "label", "orig_label",
                                      "source", "dist_um", "weight", "ly", "lx"])
