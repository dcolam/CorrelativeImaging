"""Particle positions: absolute (analysis-grid px and µm) and relative to the
diagnostic crop, plus the selection/crop geometry and analysis scale in the DB."""

import json
import sqlite3

import numpy as np
import pytest
import tifffile

from correlative_imaging import batch
from correlative_imaging.io.plate import WellInfo
from correlative_imaging.io.reader import ImageData
from correlative_imaging.storage import ResultsDB

PX = 0.65            # file pixel size; the pipeline bins 2x → 1.3 µm


@pytest.fixture
def run(tmp_path, monkeypatch):
    data = np.full((2, 3, 120, 120), 10, dtype=np.uint16)          # C, Z, Y, X
    data[0, :, 40:60, 50:70] = 1000                                   # one particle
    monkeypatch.setattr(batch, "read_image", lambda p: ImageData(
        data=data, channel_names=["GFP", "RFP"], pixel_size_um=PX, z_step_um=2.0))

    roi = np.zeros((120, 120), np.uint8); roi[20:80, 30:90] = 1        # unbinned grid
    roi_path = tmp_path / "hole.tif"; tifffile.imwrite(roi_path, roi)

    pl_dict = {"name": "t", "steps": [
        {"type": "Binning", "factor": 2, "method": "mean"},
        {"type": "ZProjection", "channel": -1, "method": "max"},
        {"type": "LoadROI", "path": str(roi_path), "roi_name": "roi_hole"},
        {"type": "AutoThreshold", "channel": 0, "method": "otsu"},
        {"type": "ParticleAnalysis", "channel": 0, "roi_mask": "roi_hole",
         "metrics": ["area"]},                       # no "centroid" selected
    ]}
    diag = {"output_dir": str(tmp_path / "diagnostics"), "formats": {"jpg"},
            "whole": False, "crops": True, "crop_pad_um": 0.0, "colors": [],
            "multichannel_tif": True, "zstack_crops": True}
    well = WellInfo(row="A", col=1, field=1, fl_path=tmp_path / "A01.tif")
    result = batch._run_pipeline_for_well(well, pl_dict, [], diag)
    assert result.error is None, result.error

    db_path = tmp_path / "r.db"
    with ResultsDB(db_path) as db:
        batch.WellBatchRunner(db_path)._write_result(db, result)
    con = sqlite3.connect(db_path); con.row_factory = sqlite3.Row
    yield con, tmp_path / "diagnostics"
    con.close()


def test_image_records_analysis_scale(run):
    con, _ = run
    img = con.execute("SELECT * FROM images").fetchone()
    assert img["pixel_size_um"] == pytest.approx(PX)
    assert img["analysis_pixel_size_um"] == pytest.approx(2 * PX)
    assert img["binning_factor"] == 2


def test_particle_absolute_and_crop_positions(run):
    con, _ = run
    p = con.execute("SELECT * FROM particle_measurements").fetchone()
    # binned particle rows 20:30, cols 25:35 → centroid (24.5, 29.5)
    assert (p["centroid_row"], p["centroid_col"]) == pytest.approx((24.5, 29.5))
    assert p["centroid_row_um"] == pytest.approx(24.5 * 2 * PX)
    assert p["centroid_col_um"] == pytest.approx(29.5 * 2 * PX)
    # hole on the binned grid starts at (10, 15); crop has no padding
    assert (p["crop_row"], p["crop_col"]) == pytest.approx((14.5, 14.5))


def test_selection_and_crop_box(run):
    con, diag = run
    s = con.execute("SELECT * FROM selections").fetchone()
    assert s["roi_mask"] == "roi_hole" and s["roi_path"].endswith("hole.tif")
    assert (s["bbox_row0"], s["bbox_row1"], s["bbox_col0"], s["bbox_col1"]) == (10, 40, 15, 45)
    assert (s["crop_row0"], s["crop_row1"], s["crop_col0"], s["crop_col1"]) == (10, 40, 15, 45)
    assert s["area_px"] == 30 * 30
    files = json.loads(s["crop_files"])
    assert [f.split("_roi_hole_crop")[1] for f in files] == [".jpg", "_channels.tif", "_zstack.tif"]
    for f in files:
        assert (diag / f).exists()


def test_crop_tifs_carry_origin(run):
    _, diag = run
    for f in diag.glob("*_crop_*.tif"):
        with tifffile.TiffFile(f) as tf:
            info = tf.imagej_metadata.get("Info", "")
        assert "crop_row0 = 10" in info and "crop_col0 = 15" in info, f.name
