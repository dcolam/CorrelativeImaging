"""SQLite-based results store with optional parquet export."""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


def _as_float(v) -> float | None:
    """None/NaN-preserving float cast — a metric the user didn't select to
    save is stored as SQL NULL, not 0, so "not measured" stays distinguishable
    from "measured and happened to be zero"."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    return float(v)


def _as_int(v) -> int | None:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    return int(v)


_DDL = """
CREATE TABLE IF NOT EXISTS images (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    path            TEXT    NOT NULL,
    filename        TEXT    NOT NULL,
    experiment      TEXT,
    n_channels      INTEGER,
    pixel_size_um   REAL,   -- of the file as read (unbinned)
    run_timestamp   TEXT,
    metadata        TEXT,   -- JSON blob
    -- Pixel size of the grid the pipeline measured on, and the binning that
    -- produced it: every *_row/*_col/bbox pixel value in the other tables is
    -- on this grid. NULL = run from before these were recorded.
    analysis_pixel_size_um REAL,
    binning_factor  REAL
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
    centroid_col    REAL,
    area_full_um2   REAL,
    cut_at_edge     INTEGER,
    share_inside    REAL,
    dist_centre_um  REAL,
    dist_centre_rel REAL,
    dist_rim_um     REAL,
    centroid_row_um REAL,   -- absolute, µm from the frame's top-left corner
    centroid_col_um REAL,
    crop_row        REAL,   -- position in this selection's diagnostic crop
    crop_col        REAL,   -- (crop pixels); NULL if no crop was written
    -- robust intensity, spread and local background (ring around the object);
    -- NULL = run from before they were measured
    median_intensity REAL,
    std_intensity   REAL,
    p25_intensity   REAL,
    p75_intensity   REAL,
    ring_median_intensity REAL,
    ring_sd_intensity REAL,
    ring_px         INTEGER
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
    n_primary_particles     INTEGER,
    pearson_mode            TEXT
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
    overlap_fraction        REAL,
    mask_overlap_fraction   REAL,
    -- NULL = run from before they were measured
    secondary_median_intensity REAL,
    secondary_std_intensity REAL,
    secondary_p25_intensity REAL,
    secondary_p75_intensity REAL,
    secondary_ring_median_intensity REAL,
    secondary_ring_sd_intensity REAL
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
    area_um2        REAL,
    median_intensity REAL
);

-- Per-image, per-channel levels the pipeline actually computed: the black
-- level BlackLevelNormalization subtracted and the threshold AutoThreshold
-- used (both computed once per channel on the whole frame). NULL = that step
-- isn't in the pipeline for this channel.
CREATE TABLE IF NOT EXISTS channel_levels (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id                INTEGER NOT NULL REFERENCES images(id),
    channel                 TEXT    NOT NULL,
    channel_index           INTEGER,
    roi_mask                TEXT,   -- 'whole_image' = global; else per-selection threshold
    black_level             REAL,
    black_method            TEXT,
    threshold_value         REAL,
    threshold_method        TEXT,
    threshold_z_projection  TEXT
);

-- One row per selection per image: where the selection lies in the frame
-- and, if a diagnostic crop was written for it, the crop's box. All pixel
-- values are on the analysis grid (images.analysis_pixel_size_um), rows/cols
-- from the top-left corner, *_row1/*_col1 exclusive. A particle's position in
-- the crop is centroid_row - crop_row0 (stored as particle_measurements.crop_row).
CREATE TABLE IF NOT EXISTS selections (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id        INTEGER NOT NULL REFERENCES images(id),
    roi_mask        TEXT    NOT NULL,
    roi_path        TEXT,
    area_px         INTEGER,
    area_um2        REAL,
    centroid_row    REAL,
    centroid_col    REAL,
    bbox_row0       INTEGER,
    bbox_row1       INTEGER,
    bbox_col0       INTEGER,
    bbox_col1       INTEGER,
    crop_row0       INTEGER,
    crop_row1       INTEGER,
    crop_col0       INTEGER,
    crop_col1       INTEGER,
    crop_files      TEXT    -- JSON list of the crop files written
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
            ("particle_measurements",       "area_full_um2 REAL"),
            ("particle_measurements",       "cut_at_edge INTEGER"),
            ("particle_measurements",       "share_inside REAL"),
            ("particle_measurements",       "dist_centre_um REAL"),
            ("particle_measurements",       "dist_centre_rel REAL"),
            ("particle_measurements",       "dist_rim_um REAL"),
            # NULL = run from before Pearson was restricted to ROI pixels.
            ("colocalization_results",      "pearson_mode TEXT"),
            ("colocalization_per_particle", "mask_overlap_fraction REAL"),
            ("intensity_measurements",      "median_intensity REAL"),
            ("channel_levels",              "roi_mask TEXT"),
            ("images",                      "analysis_pixel_size_um REAL"),
            ("images",                      "binning_factor REAL"),
            ("particle_measurements",       "centroid_row_um REAL"),
            ("particle_measurements",       "centroid_col_um REAL"),
            ("particle_measurements",       "crop_row REAL"),
            ("particle_measurements",       "crop_col REAL"),
            ("particle_measurements",       "median_intensity REAL"),
            ("colocalization_per_particle", "secondary_median_intensity REAL"),
            ("particle_measurements",       "std_intensity REAL"),
            ("particle_measurements",       "p25_intensity REAL"),
            ("particle_measurements",       "p75_intensity REAL"),
            ("particle_measurements",       "ring_median_intensity REAL"),
            ("particle_measurements",       "ring_sd_intensity REAL"),
            ("particle_measurements",       "ring_px INTEGER"),
            ("colocalization_per_particle", "secondary_std_intensity REAL"),
            ("colocalization_per_particle", "secondary_p25_intensity REAL"),
            ("colocalization_per_particle", "secondary_p75_intensity REAL"),
            ("colocalization_per_particle", "secondary_ring_median_intensity REAL"),
            ("colocalization_per_particle", "secondary_ring_sd_intensity REAL"),
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
        analysis_pixel_size_um: float | None = None,
        binning_factor: float | None = None,
    ) -> int:
        """Insert an image record and return its row id."""
        cur = self._con.execute(
            """
            INSERT INTO images (path, filename, experiment, n_channels,
                                pixel_size_um, run_timestamp, metadata,
                                analysis_pixel_size_um, binning_factor)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(image_path),
                image_path.name,
                experiment,
                n_channels,
                pixel_size_um,
                datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
                json.dumps(metadata or {}),
                _as_float(analysis_pixel_size_um),
                _as_float(binning_factor),
            ),
        )
        self._con.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def set_analysis_scale(self, image_id: int, analysis_pixel_size_um: float,
                           binning_factor: float) -> None:
        """Record the analysis grid for an image registered before its run."""
        self._con.execute(
            "UPDATE images SET analysis_pixel_size_um = ?, binning_factor = ? WHERE id = ?",
            (_as_float(analysis_pixel_size_um), _as_float(binning_factor), image_id),
        )
        self._con.commit()

    # ------------------------------------------------------------------
    # Selections (ROI geometry + diagnostic crop boxes)
    # ------------------------------------------------------------------

    def save_selections(self, image_id: int, rows: list[dict]) -> None:
        if not rows:
            return
        self._con.executemany(
            """
            INSERT INTO selections
                (image_id, roi_mask, roi_path, area_px, area_um2,
                 centroid_row, centroid_col,
                 bbox_row0, bbox_row1, bbox_col0, bbox_col1,
                 crop_row0, crop_row1, crop_col0, crop_col1, crop_files)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (image_id, str(r["roi_mask"]), str(r.get("roi_path") or ""),
                 _as_int(r.get("area_px")), _as_float(r.get("area_um2")),
                 _as_float(r.get("centroid_row")), _as_float(r.get("centroid_col")),
                 _as_int(r.get("bbox_row0")), _as_int(r.get("bbox_row1")),
                 _as_int(r.get("bbox_col0")), _as_int(r.get("bbox_col1")),
                 _as_int(r.get("crop_row0")), _as_int(r.get("crop_row1")),
                 _as_int(r.get("crop_col0")), _as_int(r.get("crop_col1")),
                 json.dumps(r["crop_files"]) if r.get("crop_files") else None)
                for r in rows
            ],
        )
        self._con.commit()

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
                    _as_int(r.get("label")),
                    _as_float(r.get("area")),
                    _as_float(r.get("area_um2")),
                    _as_float(r.get("perimeter")),
                    _as_float(r.get("circularity")),
                    _as_float(r.get("eccentricity")),
                    _as_float(r.get("solidity")),
                    _as_float(r.get("mean_intensity")),
                    _as_float(r.get("max_intensity")),
                    _as_float(r.get("min_intensity")),
                    _as_float(r.get("centroid_row")),
                    _as_float(r.get("centroid_col")),
                    _as_float(r.get("area_full_um2")),
                    _as_int(r.get("cut_at_edge")),
                    _as_float(r.get("share_inside")),
                    _as_float(r.get("dist_centre_um")),
                    _as_float(r.get("dist_centre_rel")),
                    _as_float(r.get("dist_rim_um")),
                    _as_float(r.get("centroid_row_um")),
                    _as_float(r.get("centroid_col_um")),
                    _as_float(r.get("crop_row")),
                    _as_float(r.get("crop_col")),
                    _as_float(r.get("median_intensity")),
                    _as_float(r.get("std_intensity")),
                    _as_float(r.get("p25_intensity")),
                    _as_float(r.get("p75_intensity")),
                    _as_float(r.get("ring_median_intensity")),
                    _as_float(r.get("ring_sd_intensity")),
                    _as_int(r.get("ring_px")),
                )
            )
        self._con.executemany(
            """
            INSERT INTO particle_measurements
                (image_id, channel, roi_mask, roi_path, label, area_px, area_um2, perimeter_px,
                 circularity, eccentricity, solidity, mean_intensity,
                 max_intensity, min_intensity, centroid_row, centroid_col,
                 area_full_um2, cut_at_edge, share_inside,
                 dist_centre_um, dist_centre_rel, dist_rim_um,
                 centroid_row_um, centroid_col_um, crop_row, crop_col, median_intensity,
                 std_intensity, p25_intensity, p75_intensity,
                 ring_median_intensity, ring_sd_intensity, ring_px)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                _as_float(r.get("mean_intensity")),
                _as_float(r.get("sum_intensity")),
                _as_float(r.get("std_intensity")),
                _as_int(r.get("area_px")),
                _as_float(r.get("area_um2")),
                _as_float(r.get("median_intensity")),
            )
            for _, r in df.iterrows()
        ]
        self._con.executemany(
            """
            INSERT INTO intensity_measurements
                (image_id, channel, roi_mask, roi_path,
                 mean_intensity, sum_intensity, std_intensity, area_px, area_um2,
                 median_intensity)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        self._con.commit()

    # ------------------------------------------------------------------
    # Per-channel black level / threshold
    # ------------------------------------------------------------------

    def save_channel_levels(self, image_id: int, rows: list[dict]) -> None:
        if not rows:
            return
        self._con.executemany(
            """
            INSERT INTO channel_levels
                (image_id, channel, channel_index, roi_mask, black_level, black_method,
                 threshold_value, threshold_method, threshold_z_projection)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (image_id, str(r.get("channel", "")), _as_int(r.get("channel_index")),
                 str(r.get("roi_mask") or "whole_image"),
                 _as_float(r.get("black_level")), r.get("black_method"),
                 _as_float(r.get("threshold_value")), r.get("threshold_method"),
                 r.get("threshold_z_projection"))
                for r in rows
            ],
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
                 manders_m1_random, manders_m2_random, n_primary_particles,
                 pearson_mode)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                image_id, p_ch, s_ch, roi,
                global_stats.get("manders_m1"),
                global_stats.get("manders_m2"),
                global_stats.get("pearson_r"),
                global_stats.get("manders_m1_random"),
                global_stats.get("manders_m2_random"),
                len(per_particle_df),
                global_stats.get("pearson_mode"),
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
                _as_float(r.get("mask_overlap_fraction")),
                _as_float(r.get("secondary_median_intensity")),
                _as_float(r.get("secondary_std_intensity")),
                _as_float(r.get("secondary_p25_intensity")),
                _as_float(r.get("secondary_p75_intensity")),
                _as_float(r.get("secondary_ring_median_intensity")),
                _as_float(r.get("secondary_ring_sd_intensity")),
            )
            for _, r in per_particle_df.iterrows()
        ]
        self._con.executemany(
            """
            INSERT INTO colocalization_per_particle
                (image_id, primary_channel, secondary_channel, roi_mask, primary_label,
                 primary_area_um2, n_secondary_pixels, secondary_mean_intensity,
                 secondary_max_intensity, overlap_fraction, mask_overlap_fraction,
                 secondary_median_intensity, secondary_std_intensity,
                 secondary_p25_intensity, secondary_p75_intensity,
                 secondary_ring_median_intensity, secondary_ring_sd_intensity)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
                status,
                error,
            ),
        )
        self._con.commit()

    # ------------------------------------------------------------------
    # Merge (parallel-batch per-worker sub-DBs → one combined database)
    # ------------------------------------------------------------------

    def merge_from(self, other_db_path: str | Path) -> int:
        """Copy every row from *other_db_path* (a separate ``ResultsDB`` file,
        e.g. a per-worker sub-database from a parallel batch run) into this
        database. ``image_id`` foreign keys are remapped since each source
        database has its own independent AUTOINCREMENT sequence — copying
        ``images`` rows first and recording old→new id, then rewriting every
        other table's ``image_id`` through that mapping as it's copied.

        The source file is left untouched — callers keep sub-DBs alongside
        the merged one for inspection/debugging.

        Returns the number of images merged.
        """
        other_db_path = Path(other_db_path)
        src = sqlite3.connect(str(other_db_path))
        src.row_factory = sqlite3.Row
        try:
            id_map: dict[int, int] = {}
            # Copy whichever image columns the source has — an older source
            # lacks the newer ones, which then stay NULL here.
            img_cols = [c[1] for c in src.execute("PRAGMA table_info(images)") if c[1] != "id"]
            for row in src.execute("SELECT * FROM images ORDER BY id"):
                d = {c: row[c] for c in img_cols}
                cur = self._con.execute(
                    f"INSERT INTO images ({', '.join(img_cols)}) "  # noqa: S608
                    f"VALUES ({', '.join(':' + c for c in img_cols)})",
                    d,
                )
                id_map[row["id"]] = cur.lastrowid

            for table in (
                "particle_measurements", "intensity_measurements",
                "colocalization_results", "colocalization_per_particle",
                "channel_levels", "selections", "pipeline_runs",
            ):
                cols = [c[1] for c in src.execute(f"PRAGMA table_info({table})") if c[1] != "id"]
                if not cols:
                    continue   # table predates the source database
                col_list = ", ".join(cols)
                placeholders = ", ".join(f":{c}" for c in cols)
                for row in src.execute(f"SELECT * FROM {table}"):  # noqa: S608
                    d = {c: row[c] for c in cols}
                    if "image_id" in d and d["image_id"] is not None:
                        d["image_id"] = id_map.get(d["image_id"])
                    self._con.execute(
                        f"INSERT INTO {table} ({col_list}) VALUES ({placeholders})",  # noqa: S608
                        d,
                    )

            self._con.commit()
            return len(id_map)
        finally:
            src.close()

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
            "channel_levels",
            "selections",
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
