"""SQLite-based results store with optional parquet export."""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

_DDL = """
CREATE TABLE IF NOT EXISTS images (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    path            TEXT    NOT NULL,
    filename        TEXT    NOT NULL,
    experiment      TEXT,
    n_channels      INTEGER,
    pixel_size_um   REAL,
    run_timestamp   TEXT,
    metadata        TEXT    -- JSON blob
);

CREATE TABLE IF NOT EXISTS particle_measurements (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id        INTEGER NOT NULL REFERENCES images(id),
    channel         TEXT    NOT NULL,
    roi_mask        TEXT,
    roi_path        TEXT,
    label           INTEGER,
    area_px         REAL,
    area_um2        REAL,
    perimeter_px    REAL,
    circularity     REAL,
    eccentricity    REAL,
    solidity        REAL,
    mean_intensity  REAL,
    max_intensity   REAL,
    min_intensity   REAL,
    centroid_row    REAL,
    centroid_col    REAL
);

CREATE TABLE IF NOT EXISTS colocalization_results (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id                INTEGER NOT NULL REFERENCES images(id),
    primary_channel         TEXT    NOT NULL,
    secondary_channel       TEXT    NOT NULL,
    roi_mask                TEXT,
    manders_m1              REAL,
    manders_m2              REAL,
    pearson_r               REAL,
    manders_m1_random       REAL,
    manders_m2_random       REAL,
    n_primary_particles     INTEGER
);

CREATE TABLE IF NOT EXISTS colocalization_per_particle (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id                INTEGER NOT NULL REFERENCES images(id),
    primary_channel         TEXT    NOT NULL,
    secondary_channel       TEXT    NOT NULL,
    roi_mask                TEXT,
    primary_label           INTEGER,
    primary_area_um2        REAL,
    n_secondary_pixels      INTEGER,
    secondary_mean_intensity REAL,
    secondary_max_intensity REAL,
    overlap_fraction        REAL
);

CREATE TABLE IF NOT EXISTS intensity_measurements (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id        INTEGER NOT NULL REFERENCES images(id),
    channel         TEXT    NOT NULL,
    roi_mask        TEXT,
    roi_path        TEXT,
    mean_intensity  REAL,
    sum_intensity   REAL,
    std_intensity   REAL,
    area_px         INTEGER,
    area_um2        REAL
);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id        INTEGER REFERENCES images(id),
    pipeline_name   TEXT,
    pipeline_json   TEXT,
    run_timestamp   TEXT,
    status          TEXT,
    error_message   TEXT
);
"""


class ResultsDB:
    """Wraps a SQLite database for storing pipeline results.

    Parameters
    ----------
    path:   Path to the ``.db`` file.  Created if it does not exist.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._con = sqlite3.connect(str(self.path))
        self._con.row_factory = sqlite3.Row
        self._con.executescript(_DDL)
        self._con.commit()
        self._migrate()
        log.debug("Database opened at %s", self.path)

    def _migrate(self) -> None:
        """Add columns introduced in later versions to existing databases."""
        new_columns = [
            ("colocalization_results",      "roi_mask TEXT"),
            ("colocalization_per_particle", "roi_mask TEXT"),
            ("particle_measurements",       "roi_mask TEXT"),
            ("particle_measurements",       "roi_path TEXT"),
            ("intensity_measurements",      "roi_path TEXT"),
        ]
        for table, col_def in new_columns:
            try:
                self._con.execute(f"ALTER TABLE {table} ADD COLUMN {col_def}")
                self._con.commit()
            except Exception:
                pass  # column already exists

    # ------------------------------------------------------------------
    # Image registration
    # ------------------------------------------------------------------

    def register_image(
        self,
        image_path: Path,
        experiment: str = "",
        n_channels: int = 0,
        pixel_size_um: float = 1.0,
        metadata: dict | None = None,
    ) -> int:
        """Insert an image record and return its row id."""
        cur = self._con.execute(
            """
            INSERT INTO images (path, filename, experiment, n_channels,
                                pixel_size_um, run_timestamp, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(image_path),
                image_path.name,
                experiment,
                n_channels,
                pixel_size_um,
                datetime.utcnow().isoformat(),
                json.dumps(metadata or {}),
            ),
        )
        self._con.commit()
        return cur.lastrowid  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Particle measurements
    # ------------------------------------------------------------------

    def save_particle_measurements(
        self,
        image_id: int,
        df: pd.DataFrame,
    ) -> None:
        if df.empty:
            return
        rows = []
        for _, r in df.iterrows():
            rows.append(
                (
                    image_id,
                    str(r.get("channel", "")),
                    str(r.get("roi_mask", "whole_image")),
                    str(r.get("roi_path", "") or ""),
                    int(r.get("label", 0)),
                    float(r.get("area", 0)),
                    float(r.get("area_um2", 0)),
                    float(r.get("perimeter", 0)),
                    float(r.get("circularity", 0)),
                    float(r.get("eccentricity", 0)),
                    float(r.get("solidity", 0)),
                    float(r.get("mean_intensity", 0)),
                    float(r.get("max_intensity", 0)),
                    float(r.get("min_intensity", 0)),
                    float(r.get("centroid_row", 0)),
                    float(r.get("centroid_col", 0)),
                )
            )
        self._con.executemany(
            """
            INSERT INTO particle_measurements
                (image_id, channel, roi_mask, roi_path, label, area_px, area_um2, perimeter_px,
                 circularity, eccentricity, solidity, mean_intensity,
                 max_intensity, min_intensity, centroid_row, centroid_col)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        self._con.commit()

    # ------------------------------------------------------------------
    # Bulk intensity measurements
    # ------------------------------------------------------------------

    def save_intensity_measurements(self, image_id: int, df: pd.DataFrame) -> None:
        if df.empty:
            return
        rows = [
            (
                image_id,
                str(r.get("channel", "")),
                str(r.get("roi_mask", "whole_image")),
                str(r.get("roi_path", "") or ""),
                float(r.get("mean_intensity", 0)),
                float(r.get("sum_intensity", 0)),
                float(r.get("std_intensity", 0)),
                int(r.get("area_px", 0)),
                float(r.get("area_um2", 0)),
            )
            for _, r in df.iterrows()
        ]
        self._con.executemany(
            """
            INSERT INTO intensity_measurements
                (image_id, channel, roi_mask, roi_path,
                 mean_intensity, sum_intensity, std_intensity, area_px, area_um2)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        self._con.commit()

    # ------------------------------------------------------------------
    # Colocalization results
    # ------------------------------------------------------------------

    def save_colocalization(
        self,
        image_id: int,
        per_particle_df: pd.DataFrame,
        global_stats: dict,
    ) -> None:
        if per_particle_df.empty:
            return
        first = per_particle_df.iloc[0]
        p_ch = str(first.get("primary_channel", ""))
        s_ch = str(first.get("secondary_channel", ""))
        roi  = str(first.get("roi_mask", "whole_image"))

        self._con.execute(
            """
            INSERT INTO colocalization_results
                (image_id, primary_channel, secondary_channel, roi_mask,
                 manders_m1, manders_m2, pearson_r,
                 manders_m1_random, manders_m2_random, n_primary_particles)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                image_id, p_ch, s_ch, roi,
                global_stats.get("manders_m1"),
                global_stats.get("manders_m2"),
                global_stats.get("pearson_r"),
                global_stats.get("manders_m1_random"),
                global_stats.get("manders_m2_random"),
                len(per_particle_df),
            ),
        )

        rows = [
            (
                image_id, p_ch, s_ch, roi,
                int(r.get("primary_label", 0)),
                float(r.get("primary_area_um2", 0)),
                int(r.get("n_secondary_pixels", 0)),
                float(r.get("secondary_mean_intensity", 0)),
                float(r.get("secondary_max_intensity", 0)),
                float(r.get("overlap_fraction", 0)),
            )
            for _, r in per_particle_df.iterrows()
        ]
        self._con.executemany(
            """
            INSERT INTO colocalization_per_particle
                (image_id, primary_channel, secondary_channel, roi_mask, primary_label,
                 primary_area_um2, n_secondary_pixels, secondary_mean_intensity,
                 secondary_max_intensity, overlap_fraction)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        self._con.commit()

    # ------------------------------------------------------------------
    # Pipeline run log
    # ------------------------------------------------------------------

    def log_run(
        self,
        image_id: int | None,
        pipeline_name: str,
        pipeline_json: str,
        status: str = "ok",
        error: str = "",
    ) -> None:
        self._con.execute(
            """
            INSERT INTO pipeline_runs
                (image_id, pipeline_name, pipeline_json,
                 run_timestamp, status, error_message)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                image_id,
                pipeline_name,
                pipeline_json,
                datetime.utcnow().isoformat(),
                status,
                error,
            ),
        )
        self._con.commit()

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def export_parquet(self, output_dir: str | Path) -> None:
        """Dump all result tables as parquet files next to the database."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        tables = [
            "images",
            "particle_measurements",
            "intensity_measurements",
            "colocalization_results",
            "colocalization_per_particle",
        ]
        for table in tables:
            df = pd.read_sql_query(f"SELECT * FROM {table}", self._con)  # noqa: S608
            if not df.empty:
                out = output_dir / f"{table}.parquet"
                df.to_parquet(out, index=False)
                log.info("Exported %s → %s", table, out)

    def close(self) -> None:
        self._con.close()

    def __enter__(self) -> ResultsDB:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
