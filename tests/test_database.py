"""ResultsDB: new columns round-trip, and old databases are migrated."""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from correlative_imaging.storage.database import ResultsDB


def _particle_df():
    return pd.DataFrame([{
        "channel": "GFP", "roi_mask": "roi_hole", "label": 4, "area": 53, "area_um2": 26.5,
        "area_full_um2": 173.0, "cut_at_edge": True, "share_inside": 0.15,
        "dist_centre_um": 12.2, "dist_centre_rel": 1.34, "dist_rim_um": -3.1,
    }])


def _coloc_df():
    return pd.DataFrame([{
        "primary_channel": "GFP", "secondary_channel": "mScarlet", "roi_mask": "roi_hole",
        "primary_label": 1, "primary_area_um2": 10.0, "n_secondary_pixels": 5,
        "secondary_mean_intensity": 1.0, "secondary_max_intensity": 2.0,
        "overlap_fraction": 1.0, "mask_overlap_fraction": np.nan,
    }])


def _save_all(db):
    image_id = db.register_image(Path("x.tif"))
    db.save_particle_measurements(image_id, _particle_df())
    db.save_colocalization(image_id, _coloc_df(), {"pearson_r": 0.5, "pearson_mode": "roi_pixels"})
    db.save_intensity_measurements(image_id, pd.DataFrame([{
        "channel": "GFP", "roi_mask": "roi_hole:outside_particles",
        "mean_intensity": np.nan, "median_intensity": 12.0, "area_px": 0,
    }]))


def _check(path):
    con = sqlite3.connect(path)
    row = con.execute("SELECT area_full_um2, cut_at_edge, share_inside, dist_centre_um, "
                      "dist_centre_rel, dist_rim_um FROM particle_measurements").fetchone()
    assert row == (173.0, 1, 0.15, 12.2, 1.34, -3.1)
    assert con.execute("SELECT pearson_mode FROM colocalization_results").fetchone() == ("roi_pixels",)
    assert con.execute("SELECT mask_overlap_fraction FROM colocalization_per_particle").fetchone() == (None,)
    assert con.execute("SELECT mean_intensity, median_intensity, area_px "
                       "FROM intensity_measurements").fetchone() == (None, 12.0, 0)
    con.close()


def test_new_columns_round_trip(tmp_path):
    path = tmp_path / "new.db"
    with ResultsDB(path) as db:
        _save_all(db)
    _check(path)


def test_old_database_is_migrated(tmp_path):
    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE images (id INTEGER PRIMARY KEY AUTOINCREMENT, path TEXT NOT NULL,
            filename TEXT NOT NULL, experiment TEXT, n_channels INTEGER, pixel_size_um REAL,
            run_timestamp TEXT, metadata TEXT);
        CREATE TABLE particle_measurements (id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id INTEGER NOT NULL, channel TEXT NOT NULL, roi_mask TEXT, roi_path TEXT,
            label INTEGER, area_px REAL, area_um2 REAL, perimeter_px REAL, circularity REAL,
            eccentricity REAL, solidity REAL, mean_intensity REAL, max_intensity REAL,
            min_intensity REAL, centroid_row REAL, centroid_col REAL);
        CREATE TABLE colocalization_results (id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id INTEGER NOT NULL, primary_channel TEXT NOT NULL, secondary_channel TEXT NOT NULL,
            roi_mask TEXT, manders_m1 REAL, manders_m2 REAL, pearson_r REAL,
            manders_m1_random REAL, manders_m2_random REAL, n_primary_particles INTEGER);
        CREATE TABLE colocalization_per_particle (id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id INTEGER NOT NULL, primary_channel TEXT NOT NULL, secondary_channel TEXT NOT NULL,
            roi_mask TEXT, primary_label INTEGER, primary_area_um2 REAL, n_secondary_pixels INTEGER,
            secondary_mean_intensity REAL, secondary_max_intensity REAL, overlap_fraction REAL);
        CREATE TABLE intensity_measurements (id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id INTEGER NOT NULL, channel TEXT NOT NULL, roi_mask TEXT, roi_path TEXT,
            mean_intensity REAL, sum_intensity REAL, std_intensity REAL, area_px INTEGER, area_um2 REAL);
    """)
    con.close()
    with ResultsDB(path) as db:
        _save_all(db)
    _check(path)


def test_channel_levels_saved_and_merged(tmp_path):
    from correlative_imaging.batch import _save_channel_levels
    from correlative_imaging.pipeline.base import PipelineContext
    from correlative_imaging.pipeline.preprocess import BlackLevelNormalization
    from correlative_imaging.pipeline.segment import AutoThreshold

    img = np.full((2, 20, 20), 7.0); img[:, 5:15, 5:15] = 100.0
    ctx = PipelineContext(channel_names=["GFP", "RFP"])
    black = BlackLevelNormalization(0)
    r_black = black.process(img, ctx)
    thr0, thr1 = AutoThreshold(0, min_size=0), AutoThreshold(1, method="li", min_size=0)
    pairs = [(black, r_black), (thr0, thr0.process(r_black.image, ctx)),
             (thr1, thr1.process(img, ctx))]

    sub = tmp_path / "sub.db"
    with ResultsDB(sub) as db:
        image_id = db.register_image(Path("x.tif"))
        _save_channel_levels(db, image_id, pairs)
    with ResultsDB(tmp_path / "merged.db") as db:
        db.register_image(Path("other.tif"))         # shifts ids, exercises remapping
        db.merge_from(sub)
    con = sqlite3.connect(tmp_path / "merged.db")
    rows = con.execute("SELECT image_id, channel, channel_index, black_level, black_method, "
                       "threshold_value, threshold_method, threshold_z_projection "
                       "FROM channel_levels ORDER BY channel_index").fetchall()
    con.close()
    assert len(rows) == 2
    gfp, rfp = rows
    assert gfp[:5] == (2, "GFP", 0, 7.0, "mode")
    assert 0 < gfp[5] < 93 and gfp[6:] == ("otsu", "max")
    assert rfp[3] is None and rfp[4] is None        # no black-level step on that channel
    assert rfp[6] == "li"
