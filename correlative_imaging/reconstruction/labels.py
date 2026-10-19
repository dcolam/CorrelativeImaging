"""Hand labels for particles, independent of any particular run of this tool.

A label is stored by position, not by particle id: (dataset, plate, well, channel, level, y, x) in frame coordinates
(the particle's centroid inside the hole). Rerunning detection with other settings changes particle ids, not
positions, so labels survive; :func:`match_labels` finds the particle a label belongs to.

Classes (keys 1-5 in the GUI):
  cell       a piece of a cell in the hole (the patched cell, or a cell sitting in an unrecorded hole)
  neighbour  signal of a cell next to the hole that reaches in
  merged     one particle holding two cells (e.g. the patched cell fused with a neighbour)
  debris     not a cell: crystal, dirt (typically small, dimmer than a cell)
  plug       not a cell: the bright plug of material over the aperture (very bright, often in every channel; common
             in wells without a cell, e.g. the "Empty" condition)
  noise      background, halo, haze

The store is a SQLite file (default ``ci_labels.db`` in the run's output folder, next to the Results Explorer's
well labels). Tables: ``particle_labels``, ``test_wells`` (the frozen held-out set, never trained on),
``particle_cuts`` (hand-drawn lines that split particles; frame coordinates, all channels and levels),
``particle_merges`` (two positions whose particles are joined; one channel and level), ``particle_splits``
(accepted split proposals: the particle at that position is split at its dips), ``particle_watersheds`` (particles
split on demand; their dividing lines act as cuts), ``cell_labels``, ``unsure_particles`` (marked unsure in the
batch labeller: never trained on, not shown again) and ``well_verdicts``. A cell marked unsure is a cell label with
source "unsure" and no verdict or colour.

``source`` of a particle label: "hand" (clicked in the canvas), "blind" (labelled with every suggestion hidden: the
test wells always), "accepted" / "corrected" (a suggestion shown in the label queue was kept / changed), "legacy"
(imported). Only human decisions are stored: Detect's pre-labels and the model's classes never are.

Cell labels (one per reconstructed cell, stored at a point inside one of its seed particles, frame coordinates):
``verdict`` one cell / several cells (inseparable: two or more cells that cannot be split by eye, either could be
the patched one) / not a cell, and ``colour``: the identity channels the cell is positive for ("+"-joined raw
channel names, "none" for none, "" = not judged).
"""
from __future__ import annotations

import datetime
import getpass
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ["cell", "neighbour", "merged", "debris", "noise", "plug"]   # new classes only ever go at the end
# (a model stores class indices; a model trained before "plug" existed simply never predicts it)
VERDICTS = ["one cell", "several cells", "not a cell"]
SOURCES = ["hand", "blind", "accepted", "corrected", "legacy"]
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
CREATE TABLE IF NOT EXISTS particle_cuts (dataset TEXT, plate_id TEXT, well_id TEXT, cut_id INTEGER PRIMARY KEY
    AUTOINCREMENT, points TEXT, labeller TEXT, t TEXT);
CREATE TABLE IF NOT EXISTS particle_merges (dataset TEXT, plate_id TEXT, well_id TEXT, channel TEXT, level TEXT,
    y1 REAL, x1 REAL, y2 REAL, x2 REAL, labeller TEXT, t TEXT);
CREATE TABLE IF NOT EXISTS particle_splits (dataset TEXT, plate_id TEXT, well_id TEXT, channel TEXT, level TEXT,
    y REAL, x REAL, labeller TEXT, t TEXT);
CREATE TABLE IF NOT EXISTS particle_watersheds (dataset TEXT, plate_id TEXT, well_id TEXT, channel TEXT,
    level TEXT, y REAL, x REAL, labeller TEXT, t TEXT);
CREATE TABLE IF NOT EXISTS unsure_particles (dataset TEXT, plate_id TEXT, well_id TEXT, channel TEXT, level TEXT,
    y REAL, x REAL, labeller TEXT, t TEXT, PRIMARY KEY (dataset, plate_id, well_id, channel, level, y, x));
CREATE TABLE IF NOT EXISTS cell_labels (dataset TEXT, plate_id TEXT, well_id TEXT, y REAL, x REAL,
    verdict TEXT, colour TEXT, source TEXT, labeller TEXT, t TEXT, PRIMARY KEY (dataset, plate_id, well_id, y, x));
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

    def seed_test_wells(self, dataset: str, wells) -> int:
        """Add held-out wells explicitly (e.g. copied from another library, or chosen before any labelling)."""
        with self._con() as con:
            n0 = con.execute("SELECT count(*) FROM test_wells WHERE dataset=?", (dataset,)).fetchone()[0]
            con.executemany("INSERT OR IGNORE INTO test_wells VALUES (?,?,?,?)",
                            [(dataset, p, w, _now()) for p, w in wells])
            return con.execute("SELECT count(*) FROM test_wells WHERE dataset=?", (dataset,)).fetchone()[0] - n0

    def freeze_test_wells_from(self, dataset: str, wells: pd.DataFrame, frac: float = 0.2, seed: int = 0,
                               other: str | Path | None = None) -> set:
        """Clean slate: pick the held-out wells before any labelling, from the results' wells that hold something
        in the hole (preview status other than "no particles"), stratified by plate. With `other` (an older label
        library of the same dataset), its test wells are taken over and the rest is drawn only from wells it never
        labelled, so a model trained on that library can be scored fairly on them."""
        have = self.test_wells(dataset)
        if have:
            return have
        w = wells
        if "preview_status" in w:
            w = w[w.preview_status != "no particles"]
        pick = set()
        if other:
            o = LabelStore(other)
            pick |= o.test_wells(dataset)
            seen = set(zip(o.frame(dataset).plate_id, o.frame(dataset).well_id))
            w = w[[k not in seen for k in zip(w.plate_id, w.well_id)]]
        rng = np.random.default_rng(seed)
        for _, g in w.groupby("plate_id"):
            n = int(round(frac * len(g)))
            if n:
                idx = rng.choice(len(g), n, replace=False)
                pick |= set(zip(g.plate_id.iloc[idx], g.well_id.iloc[idx]))
        self.seed_test_wells(dataset, sorted(pick))
        return pick

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

    # -- particle edits (applied by Detect: edits.py)
    def add_cut(self, dataset, plate_id, well_id, points):
        import json
        with self._con() as con:
            con.execute("INSERT INTO particle_cuts (dataset, plate_id, well_id, points, labeller, t) VALUES (?,?,?,?,?,?)",
                        (dataset, plate_id, well_id, json.dumps([[round(float(y), 2), round(float(x), 2)]
                                                                 for y, x in points]), getpass.getuser(), _now()))

    def add_merge(self, dataset, plate_id, well_id, channel, level, p1, p2):
        with self._con() as con:
            con.execute("INSERT INTO particle_merges VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                        (dataset, plate_id, well_id, channel, level, float(p1[0]), float(p1[1]), float(p2[0]),
                         float(p2[1]), getpass.getuser(), _now()))

    def accept_split(self, dataset, plate_id, well_id, channel, level, y, x):
        with self._con() as con:
            con.execute("INSERT INTO particle_splits VALUES (?,?,?,?,?,?,?,?,?)",
                        (dataset, plate_id, well_id, channel, level, float(y), float(x), getpass.getuser(), _now()))

    def add_watershed(self, dataset, plate_id, well_id, channel, level, y, x):
        with self._con() as con:
            con.execute("INSERT INTO particle_watersheds VALUES (?,?,?,?,?,?,?,?,?)",
                        (dataset, plate_id, well_id, channel, level, float(y), float(x), getpass.getuser(), _now()))

    def clear_edits(self, dataset, plate_id, well_id):
        with self._con() as con:
            for t in ("particle_cuts", "particle_merges", "particle_splits", "particle_watersheds"):
                con.execute(f"DELETE FROM {t} WHERE dataset=? AND plate_id=? AND well_id=?", (dataset, plate_id, well_id))

    def edits(self, dataset: str) -> dict:
        """{(plate_id, well_id): {"cuts": [...], "merges": [...], "splits": [...], "watersheds": [...]}} for
        detect_well."""
        import json
        out = {}
        new = lambda: {"cuts": [], "merges": [], "splits": [], "watersheds": []}
        with self._con() as con:
            for p, w, pts in con.execute("SELECT plate_id, well_id, points FROM particle_cuts WHERE dataset=?",
                                         (dataset,)):
                out.setdefault((p, w), new())["cuts"].append(json.loads(pts))
            for p, w, ch, lev, y1, x1, y2, x2 in con.execute(
                    "SELECT plate_id, well_id, channel, level, y1, x1, y2, x2 FROM particle_merges WHERE dataset=?",
                    (dataset,)):
                out.setdefault((p, w), new())["merges"].append(
                    dict(channel=ch, level=lev, y1=y1, x1=x1, y2=y2, x2=x2))
            for p, w, ch, lev, y, x in con.execute(
                    "SELECT plate_id, well_id, channel, level, y, x FROM particle_splits WHERE dataset=?", (dataset,)):
                out.setdefault((p, w), new())["splits"].append(
                    dict(channel=ch, level=lev, y=y, x=x))
            for p, w, ch, lev, y, x in con.execute(
                    "SELECT plate_id, well_id, channel, level, y, x FROM particle_watersheds WHERE dataset=?",
                    (dataset,)):
                out.setdefault((p, w), new())["watersheds"].append(dict(channel=ch, level=lev, y=y, x=x))
        return out

    # -- unsure (batch labeller: not trained on, not shown again)
    def mark_unsure(self, dataset, plate_id, well_id, channel, level, y, x):
        with self._con() as con:
            con.execute("INSERT OR REPLACE INTO unsure_particles VALUES (?,?,?,?,?,?,?,?,?)",
                        (dataset, plate_id, well_id, channel, level, round(float(y), 2), round(float(x), 2),
                         getpass.getuser(), _now()))

    def unsure_frame(self, dataset: str) -> pd.DataFrame:
        """In the shape of `frame` (label "unsure"), so match_labels can place them."""
        with self._con() as con:
            d = pd.read_sql("SELECT * FROM unsure_particles WHERE dataset=?", con, params=(dataset,))
        return d.assign(label="unsure", orig_label=None, source="unsure")

    # -- cell labels
    def set_cell(self, dataset, plate_id, well_id, y, x, verdict=None, colour=None, source="hand"):
        """Store or update the label of the cell at (y, x); None keeps what is stored."""
        y, x = round(float(y), 2), round(float(x), 2)
        with self._con() as con:
            old = con.execute("SELECT verdict, colour FROM cell_labels WHERE dataset=? AND plate_id=? AND well_id=? "
                              "AND y=? AND x=?", (dataset, plate_id, well_id, y, x)).fetchone()
            v = verdict if verdict is not None else (old[0] if old else "")
            c = colour if colour is not None else (old[1] if old else "")
            if verdict is not None and verdict not in VERDICTS + [""]:
                raise ValueError(f"unknown verdict {verdict!r}")
            con.execute("INSERT OR REPLACE INTO cell_labels VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (dataset, plate_id, well_id, y, x, v, c, source, getpass.getuser(), _now()))

    def remove_cell(self, dataset, plate_id, well_id, y, x):
        with self._con() as con:
            con.execute("DELETE FROM cell_labels WHERE dataset=? AND plate_id=? AND well_id=? AND y=? AND x=?",
                        (dataset, plate_id, well_id, round(float(y), 2), round(float(x), 2)))

    def cell_frame(self, dataset: str | None = None) -> pd.DataFrame:
        with self._con() as con:
            q, a = "SELECT * FROM cell_labels", ()
            if dataset:
                q, a = q + " WHERE dataset=?", (dataset,)
            return pd.read_sql(q, con, params=a)

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
    out = []
    if labels is None or not len(labels):
        return pd.DataFrame(out, columns=["dataset", "plate_id", "well_id", "oid", "level", "label", "orig_label",
                                          "source", "dist_um", "weight", "ly", "lx"])
    off = {(p, w): (float(r0), float(c0), float(px)) for p, w, r0, c0, px in
           zip(wells.plate_id, wells.well_id, wells.off_r, wells.off_c, wells.px_um)}
    # only the labelled wells' particles (a full groupby over every well is the slow part)
    lk = set(zip(labels.plate_id, labels.well_id))
    pk = [k in lk for k in zip(particles.plate_id.to_numpy(object), particles.well_id.to_numpy(object))]
    pool = particles[np.asarray(pk, bool) & particles.pool.astype(bool).to_numpy()]
    groups = {k: g for k, g in pool.groupby(["plate_id", "well_id", "ch"])}
    for (pl, wl, ch), lg in labels.groupby(["plate_id", "well_id", "channel"]):
        g = groups.get((pl, wl, ch))
        if g is None or (pl, wl) not in off:
            continue
        r0, c0, px = off[(pl, wl)]
        y = g.iy.to_numpy(float) + r0 + legacy_offset
        x = g.ix.to_numpy(float) + c0 + legacy_offset
        D = np.hypot(y[None] - lg.y.to_numpy(float)[:, None], x[None] - lg.x.to_numpy(float)[:, None]) * px
        lev = g.level.to_numpy()
        oid = g.oid.to_numpy()
        for r, d in zip(lg.itertuples(index=False), D):
            ok = d <= tol_um
            if r.level != "any":
                ok &= lev == r.level
            if not ok.any():
                continue
            best = {}
            for k in np.flatnonzero(ok)[np.argsort(d[ok])]:          # nearest particle of each level
                best.setdefault(lev[k], k)
            for k in best.values():
                out.append(dict(dataset=r.dataset, plate_id=pl, well_id=wl, oid=oid[k], level=lev[k],
                                label=r.label, orig_label=r.orig_label, source=r.source, dist_um=float(d[k]),
                                weight=1.0 / len(best), ly=r.y, lx=r.x))
    return pd.DataFrame(out, columns=["dataset", "plate_id", "well_id", "oid", "level", "label", "orig_label",
                                      "source", "dist_um", "weight", "ly", "lx"])


def match_cells(cell_labels: pd.DataFrame, res) -> pd.DataFrame:
    """Attach each cell label to the reconstructed cell under its point (the run's recon/<well>.npz). Returns the
    label rows + cell_id (labels on no cell are dropped) and the cell's call / members if Classify wrote them."""
    from .pipeline import load_recon
    cols = list(cell_labels.columns) + ["cell_id"]
    if cell_labels is None or not len(cell_labels):
        return pd.DataFrame(columns=cols)
    W = res.wells.set_index(["plate_id", "well_id"])
    out = []
    for (p, w), g in cell_labels.groupby(["plate_id", "well_id"]):
        if (p, w) not in W.index:
            continue
        G = res.recon(p, w)
        if G is None:
            continue
        r0, c0 = int(W.at[(p, w), "off_r"]), int(W.at[(p, w), "off_c"])
        for r in g.itertuples(index=False):
            y, x = int(round(r.y)) - r0, int(round(r.x)) - c0
            if 0 <= y < G.shape[0] and 0 <= x < G.shape[1] and G[y, x] > 0:
                out.append(dict(r._asdict(), cell_id=int(G[y, x])))
    m = pd.DataFrame(out, columns=cols)
    if len(m) and res.cells is not None and len(res.cells):
        keep = [c for c in ("call", "members", "status") if c in res.cells]
        m = m.merge(res.cells[["plate_id", "well_id", "cell_id"] + keep], on=["plate_id", "well_id", "cell_id"],
                    how="left")
    return m
