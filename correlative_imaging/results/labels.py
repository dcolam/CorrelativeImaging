"""Persisted ground-truth labels + per-run channel-display config (sidecar DB).

Two things live in a small sidecar SQLite file (``ci_labels.db`` by default) in
the results folder the user browsed — the run result databases are opened
read-only and one folder can hold several runs, so both live outside them:

1. **Ground-truth labels** (``well_labels``). The user hand-labels a well by eye:
   which *channels* are truly positive in the hole, independently (multi-label) —
   a hole can be several channels, or none (a negative / empty hole). These are
   the ground truth that per-channel thresholds are calibrated to and that any
   later Random Forest trains on. Keyed by ``(run_tag, plate, well_id)``; the
   positive channels are a comma-joined string, so ANY channel names / counts
   work with no schema change.

2. **Channel display config** (``channel_display``). Channel↔colour assignment
   varies between experiments, so colour/label is a pure *display* concern, kept
   per run and editable: a display name and a tint colour per channel. Defaults
   come from the wavelength map (:data:`~.analysis.CHANNEL_HEX`) with a fallback
   palette for unknown channels.

Both label store and classifier key by CHANNEL NAME, so validating the
classifier against the labels is a direct per-channel comparison — no colour
indirection. Everything here is plain ``sqlite3`` — no Qt — so it is testable
headless.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .analysis import CHANNEL_COLOR_NAME, CHANNEL_HEX

DEFAULT_LABELS_FILENAME = "ci_labels.db"

# Cycled for channels with no known wavelength colour, so every channel still
# gets a distinct, stable tint.
_FALLBACK_PALETTE = [
    "#3b6bff", "#2fbf3c", "#ff3b3b", "#e8a33d", "#9b5de5", "#00b4d8",
    "#f15bb5", "#8ac926",
]


def default_display(channel: str, index: int = 0) -> dict:
    """The default display name + colour for a channel: the channel name itself
    as the label, and the wavelength colour if known else a palette fallback
    (by *index*, so a run's channels get distinct tints)."""
    hexc = CHANNEL_HEX.get(channel) or _FALLBACK_PALETTE[index % len(_FALLBACK_PALETTE)]
    return {"display_name": channel, "color_hex": hexc}


class LabelStore:
    """Read/write hand labels and channel-display config in a sidecar SQLite file.

    One ``well_labels`` row per (run_tag, plate, well_id). A row's presence means
    the well was labelled; an empty ``pos_channels`` records a deliberate
    negative-hole label, distinct from "never labelled" (no row)."""

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
        con = self._connect()
        try:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS well_labels (
                    run_tag      TEXT NOT NULL,
                    plate        TEXT NOT NULL,
                    well_id      TEXT NOT NULL,
                    pos_channels TEXT NOT NULL DEFAULT '',
                    is_negative  INTEGER NOT NULL DEFAULT 0,
                    notes        TEXT,
                    updated_at   TEXT NOT NULL,
                    PRIMARY KEY (run_tag, plate, well_id)
                )
                """
            )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS channel_display (
                    run_tag      TEXT NOT NULL,
                    channel      TEXT NOT NULL,
                    display_name TEXT NOT NULL,
                    color_hex    TEXT NOT NULL,
                    PRIMARY KEY (run_tag, channel)
                )
                """
            )
            self._migrate_legacy_labels(con)
            con.commit()
        finally:
            con.close()

    def _migrate_legacy_labels(self, con) -> None:
        """Upgrade a ``well_labels`` table written by the earlier per-COLOUR
        schema (``pos_blue``/``pos_green``/``pos_red``) to the channel-keyed
        ``pos_channels`` column, backfilling existing labels by mapping each
        stored colour back to its channel via :data:`~.analysis.CHANNEL_COLOR_NAME`.
        A no-op on a fresh (already channel-keyed) database."""
        cols = [r[1] for r in con.execute("PRAGMA table_info(well_labels)")]
        if "pos_channels" in cols:
            return
        con.execute("ALTER TABLE well_labels ADD COLUMN pos_channels TEXT NOT NULL DEFAULT ''")
        legacy = [c for c in cols if c.startswith("pos_") and c != "pos_channels"]
        if not legacy:
            return
        colour_to_channel = {v: k for k, v in CHANNEL_COLOR_NAME.items()}
        rows = con.execute(
            f"SELECT rowid, {', '.join(legacy)} FROM well_labels"
        ).fetchall()
        for row in rows:
            chans = [
                colour_to_channel.get(c[len("pos_"):], c[len("pos_"):])
                for c in legacy if row[c]
            ]
            con.execute(
                "UPDATE well_labels SET pos_channels=? WHERE rowid=?",
                (self._join(chans), row["rowid"]),
            )

    # ── Ground-truth labels ──────────────────────────────────────────
    @staticmethod
    def _join(channels) -> str:
        # Stable order, de-duplicated, no blanks.
        return ",".join(sorted({c for c in channels if c}))

    def set_label(self, run_tag: str, plate: str, well_id: str,
                  channels, notes: str | None = None) -> None:
        """Upsert the label for one well. ``channels`` is the set/iterable of
        positive channel names; an empty set records a negative hole."""
        joined = self._join(channels)
        is_negative = 0 if joined else 1
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        con = self._connect()
        try:
            con.execute(
                """
                INSERT INTO well_labels
                    (run_tag, plate, well_id, pos_channels, is_negative, notes, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_tag, plate, well_id) DO UPDATE SET
                    pos_channels=excluded.pos_channels,
                    is_negative=excluded.is_negative,
                    notes=excluded.notes,
                    updated_at=excluded.updated_at
                """,
                (run_tag, plate, well_id, joined, is_negative, notes, now),
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
        """The stored label for one well as a dict (with a ``channels`` set), or
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
        d["channels"] = {c for c in (d.get("pos_channels") or "").split(",") if c}
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

    # ── Channel display config (per run) ─────────────────────────────
    def get_channel_display(self, run_tag: str, channels: list[str]) -> dict:
        """``{channel: {"display_name", "color_hex"}}`` for *channels*, merging
        any stored overrides over the wavelength/palette defaults. Always returns
        an entry for every channel asked for."""
        con = self._connect()
        try:
            rows = con.execute(
                "SELECT channel, display_name, color_hex FROM channel_display WHERE run_tag=?",
                (run_tag,),
            ).fetchall()
        finally:
            con.close()
        stored = {r["channel"]: {"display_name": r["display_name"],
                                 "color_hex": r["color_hex"]} for r in rows}
        out = {}
        for i, ch in enumerate(channels):
            out[ch] = stored.get(ch, default_display(ch, i))
        return out

    def set_channel_display(self, run_tag: str, channel: str,
                            display_name: str, color_hex: str) -> None:
        con = self._connect()
        try:
            con.execute(
                """
                INSERT INTO channel_display (run_tag, channel, display_name, color_hex)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(run_tag, channel) DO UPDATE SET
                    display_name=excluded.display_name,
                    color_hex=excluded.color_hex
                """,
                (run_tag, channel, display_name, color_hex),
            )
            con.commit()
        finally:
            con.close()
