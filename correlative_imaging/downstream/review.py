"""Review store: your verdicts per well and labels per particle, in <out>/review.db (plain sqlite3, no Qt).

Particle labels use the carousel-session set, so old and new labels train together:
  patched cell, maybe cell, neighbour, two fused cells, background_noise, debris/unsure

`training_table` joins the labels with the particle measurements of the run (particles.csv) into the format that
models.fit_models expects (particle_label, snr, area_um2, channel, share_inside, fill), optionally together with
earlier label files.
"""
from __future__ import annotations

import datetime
import json
import sqlite3
from pathlib import Path

import pandas as pd

PARTICLE_LABELS = ["patched cell", "maybe cell", "neighbour", "two fused cells", "background_noise", "debris/unsure"]
VERDICTS = ["agree", "wrong colour", "wrong cell", "patched not visible", "unsure"]

_DDL = """
CREATE TABLE IF NOT EXISTS verdicts (
    plate_id TEXT, well_id TEXT, verdict TEXT, note TEXT, blind INTEGER, reviewer TEXT, t TEXT,
    PRIMARY KEY (plate_id, well_id));
CREATE TABLE IF NOT EXISTS particle_labels (
    plate_id TEXT, well_id TEXT, oid TEXT, seg TEXT, channel TEXT, particle_label TEXT, blind INTEGER,
    reviewer TEXT, t TEXT, PRIMARY KEY (plate_id, well_id, oid));
"""


class ReviewStore:
    def __init__(self, out_dir: str | Path, reviewer: str = ""):
        self.path = Path(out_dir) / "review.db"
        self.reviewer = reviewer
        with self._con() as con:
            con.executescript(_DDL)

    def _con(self):
        return sqlite3.connect(self.path)

    @staticmethod
    def _now():
        return datetime.datetime.now().isoformat(timespec="seconds")

    def save_verdict(self, d: dict) -> None:
        with self._con() as con:
            con.execute("INSERT OR REPLACE INTO verdicts VALUES (?,?,?,?,?,?,?)",
                        (d["plate"], d["well"], d["v"], d.get("note", ""), int(bool(d.get("blind"))),
                         self.reviewer, self._now()))

    def save_label(self, d: dict) -> None:
        with self._con() as con:
            if not d.get("label"):
                con.execute("DELETE FROM particle_labels WHERE plate_id=? AND well_id=? AND oid=?",
                            (d["plate"], d["well"], d["oid"]))
                return
            if d["label"] not in PARTICLE_LABELS:
                raise ValueError(f"unknown label {d['label']}")
            con.execute("INSERT OR REPLACE INTO particle_labels VALUES (?,?,?,?,?,?,?,?,?)",
                        (d["plate"], d["well"], d["oid"], d.get("seg"), d.get("ch"), d["label"],
                         int(bool(d.get("blind"))), self.reviewer, d.get("t") or self._now()))

    def verdicts(self) -> pd.DataFrame:
        with self._con() as con:
            return pd.read_sql("SELECT * FROM verdicts", con)

    def labels(self) -> pd.DataFrame:
        with self._con() as con:
            return pd.read_sql("SELECT * FROM particle_labels", con)

    def state_json(self) -> str:
        """Everything the viewer needs to show what was already reviewed."""
        v = {f"{r.plate_id}_{r.well_id}": dict(v=r.verdict, note=r.note, blind=bool(r.blind), t=r.t)
             for r in self.verdicts().itertuples()}
        lab = {}
        for r in self.labels().itertuples():
            lab.setdefault(f"{r.plate_id}_{r.well_id}", {})[r.oid] = r.particle_label
        return json.dumps(dict(verdicts=v, labels=lab))


def training_table(out_dir: str | Path, extra_label_files: list | None = None) -> pd.DataFrame:
    """Labels of this run joined with its particle measurements, plus earlier label files (same columns)."""
    out = Path(out_dir)
    lab = ReviewStore(out).labels()
    parts = pd.read_csv(out / "particles.csv", dtype={"plate_id": str, "well_id": str})
    t = lab.merge(parts, on=["plate_id", "well_id", "oid"], how="inner")
    t = pd.DataFrame(dict(particle_label=t.particle_label, snr=t.snr, area_um2=t.in_um2, channel=t.ch,
                          share_inside=t.share, fill=t.fill, source=str(out)))
    frames = [t]
    for f in extra_label_files or []:
        e = pd.read_csv(f)
        frames.append(e[["particle_label", "snr", "area_um2", "channel", "share_inside", "fill"]].assign(source=str(f)))
    return pd.concat(frames, ignore_index=True)
