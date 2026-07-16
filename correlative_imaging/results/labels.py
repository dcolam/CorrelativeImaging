"""Persisted ground-truth labels for hole colour (multi-label, per colour).

The user hand-labels a well by eye: which colours are truly present in the hole
(blue / green / red), independently — a hole can be several colours, or none
(a negative / empty hole). These labels are the ground truth that per-channel
thresholds are calibrated to and that any later Random Forest trains on.

Storage is a small sidecar SQLite file (``ci_labels.db`` by default) in the
results folder the user browsed — the run result databases are opened read-only
and one folder can hold several runs, so labels live outside them and are keyed
by ``(run_tag, plate, well_id)``.

The label schema mirrors the classifier output verbatim (per-colour booleans),
so validating the classifier against the labels is a plain per-colour join.
Labels are keyed by *colour name*, not channel, so they survive a change of
fluorophore/channel naming; :data:`~.analysis.CHANNEL_COLOR_NAME` maps a run's
channels onto these colours.

Everything here is plain ``sqlite3`` — no Qt — so it is testable headless.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .analysis import CHANNEL_COLOR_NAME

DEFAULT_LABELS_FILENAME = "ci_labels.db"

# The colour vocabulary, derived from the fixed channel→colour map so it stays
# in sync with the classifier. Sorted for a stable column order.
LABEL_COLORS: list[str] = sorted(set(CHANNEL_COLOR_NAME.values()))


def _pos_col(color: str) -> str:
    return f"pos_{color}"


class LabelStore:
    """Read/write hand labels in a sidecar SQLite file.

    One row per (run_tag, plate, well_id). A row's presence means the well was
    labelled; ``is_negative`` (all colours 0) records a deliberate negative-hole
    label, distinct from "never labelled" (no row)."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self._ensure_schema()

    @classmethod
    def for_folder(cls, folder: str | Path,
                   filename: str = DEFAULT_LABELS_FILENAME) -> "LabelStore":
        """A label store living beside a browsed results folder."""
        return cls(Path(folder) / filename)

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        return con

    def _ensure_schema(self) -> None:
        cols = ", ".join(f"{_pos_col(c)} INTEGER NOT NULL DEFAULT 0" for c in LABEL_COLORS)
        con = self._connect()
        try:
            con.execute(
                f"""
                CREATE TABLE IF NOT EXISTS well_labels (
                    run_tag   TEXT NOT NULL,
                    plate     TEXT NOT NULL,
                    well_id   TEXT NOT NULL,
                    {cols},
                    is_negative INTEGER NOT NULL DEFAULT 0,
                    notes     TEXT,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (run_tag, plate, well_id)
                )
                """
            )
            con.commit()
        finally:
            con.close()

    def set_label(self, run_tag: str, plate: str, well_id: str,
                  colors, notes: str | None = None) -> None:
        """Upsert the label for one well. ``colors`` is the set/iterable of
        positive colour names; an empty set records a negative hole."""
        colors = {c for c in colors if c in LABEL_COLORS}
        vals = {_pos_col(c): (1 if c in colors else 0) for c in LABEL_COLORS}
        is_negative = 1 if not colors else 0
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")

        pos_cols = list(vals.keys())
        col_names = ["run_tag", "plate", "well_id", *pos_cols,
                     "is_negative", "notes", "updated_at"]
        placeholders = ", ".join("?" for _ in col_names)
        updates = ", ".join(f"{c}=excluded.{c}"
                            for c in [*pos_cols, "is_negative", "notes", "updated_at"])
        params = [run_tag, plate, well_id, *[vals[c] for c in pos_cols],
                  is_negative, notes, now]
        con = self._connect()
        try:
            con.execute(
                f"""
                INSERT INTO well_labels ({", ".join(col_names)})
                VALUES ({placeholders})
                ON CONFLICT(run_tag, plate, well_id) DO UPDATE SET {updates}
                """,
                params,
            )
            con.commit()
        finally:
            con.close()

    def delete_label(self, run_tag: str, plate: str, well_id: str) -> None:
        con = self._connect()
        try:
            con.execute(
                "DELETE FROM well_labels WHERE run_tag=? AND plate=? AND well_id=?",
                (run_tag, plate, well_id),
            )
            con.commit()
        finally:
            con.close()

    def get_label(self, run_tag: str, plate: str, well_id: str) -> dict | None:
        """The stored label for one well as a dict (with a ``colors`` set), or
        ``None`` if the well was never labelled."""
        con = self._connect()
        try:
            row = con.execute(
                "SELECT * FROM well_labels WHERE run_tag=? AND plate=? AND well_id=?",
                (run_tag, plate, well_id),
            ).fetchone()
        finally:
            con.close()
        if row is None:
            return None
        d = dict(row)
        d["colors"] = {c for c in LABEL_COLORS if d.get(_pos_col(c))}
        return d

    def load_frame(self, run_tag: str | None = None) -> pd.DataFrame:
        """All labels (optionally for one run) as a DataFrame. Empty if none."""
        con = self._connect()
        try:
            if run_tag is None:
                df = pd.read_sql("SELECT * FROM well_labels", con)
            else:
                df = pd.read_sql(
                    "SELECT * FROM well_labels WHERE run_tag=?", con, params=(run_tag,)
                )
        finally:
            con.close()
        return df

    def count(self, run_tag: str | None = None) -> int:
        df = self.load_frame(run_tag)
        return 0 if df.empty else len(df)
