"""ParticleAnalysis edge columns and IntensityMeasurement background outside particles."""

import numpy as np
import pytest

from correlative_imaging.pipeline.analyze import IntensityMeasurement, ParticleAnalysis
from correlative_imaging.pipeline.base import PipelineContext


def _particles(img2d, mask, roi=None, px=1.0, **kw):
    masks = {"mask_ch0": mask}
    if roi is not None:
        masks["sel"] = roi
    ctx = PipelineContext(channel_names=["a"], pixel_size_um=px, masks=masks)
    step = ParticleAnalysis(0, min_size_um2=0, max_size_um2=0,
                            roi_mask="sel" if roi is not None else "", **kw)
    return step.process(img2d[None], ctx).measurements


# ── item 3: where each particle sits relative to its selection ──────────

def test_object_cut_at_selection_edge():
    mask = np.zeros((60, 60), bool); mask[20:40, 20:40] = True       # 20×20 object
    roi = np.zeros((60, 60), bool); roi[:, :30] = True               # covers its left half
    df = _particles(mask.astype(float), mask, roi)
    row = df.iloc[0]
    assert row["area_um2"] == 200
    assert row["area_full_um2"] == 400
    assert row["share_inside"] == pytest.approx(0.5)
    assert bool(row["cut_at_edge"]) is True


def test_object_fully_inside():
    mask = np.zeros((60, 60), bool); mask[20:40, 20:40] = True
    roi = np.zeros((60, 60), bool); roi[10:50, 10:50] = True
    row = _particles(mask.astype(float), mask, roi).iloc[0]
    assert row["share_inside"] == 1.0
    assert bool(row["cut_at_edge"]) is False


def test_no_selection_leaves_distances_empty():
    mask = np.zeros((60, 60), bool); mask[20:40, 20:40] = True
    row = _particles(mask.astype(float), mask).iloc[0]
    assert row["share_inside"] == 1.0
    assert bool(row["cut_at_edge"]) is False
    assert np.isnan(row["dist_centre_um"]) and np.isnan(row["dist_rim_um"])


def test_edge_columns_kept_when_metrics_selected():
    mask = np.zeros((60, 60), bool); mask[20:40, 20:40] = True
    df = _particles(mask.astype(float), mask, metrics=["area_um2"])
    assert {"area_full_um2", "cut_at_edge", "share_inside",
            "dist_centre_um", "dist_centre_rel", "dist_rim_um"} <= set(df.columns)


def _disc(shape, centre, r):
    yy, xx = np.indices(shape)
    return np.hypot(yy - centre[0], xx - centre[1]) <= r


@pytest.mark.parametrize("px", [1.0, 0.5])
def test_distances_to_disc_selection(px):
    shape, c = (256, 256), (128, 128)
    roi = _disc(shape, c, 50)
    centred = _disc(shape, c, 8)
    outside = _disc(shape, (128, 188), 15)    # 60 px out, reaches 5 px into the disc
    mask = centred | outside
    df = _particles(mask.astype(float), mask, roi, px=px).sort_values("centroid_col")
    inner, outer = df.iloc[0], df.iloc[1]

    assert inner["dist_centre_rel"] == pytest.approx(0, abs=0.02)
    assert inner["dist_rim_um"] == pytest.approx(50 * px, abs=1.5 * px)

    # the whole object's centroid, not that of its piece inside the disc (~0.95)
    assert bool(outer["cut_at_edge"]) is True
    assert outer["dist_centre_um"] == pytest.approx(60 * px, abs=0.5 * px)
    assert outer["dist_centre_rel"] == pytest.approx(1.2, abs=0.02)
    assert outer["dist_rim_um"] == pytest.approx(-10 * px, abs=1.5 * px)


# ── item 4: background outside all particles ─────────────────────────────

def _bg_rows(margin_um, roi=None):
    plane = np.full((40, 40), 10.0); plane[15:25, 15:25] = 100.0
    masks = {"mask_ch0": plane > 50}
    if roi is not None:
        masks["sel"] = roi
    ctx = PipelineContext(channel_names=["a"], pixel_size_um=1.0, masks=masks)
    step = IntensityMeasurement(0, roi_mask="sel" if roi is not None else "",
                                outside_particles=True, outside_margin_um=margin_um)
    return step.process(plane[None], ctx).measurements.set_index("roi_mask")


def test_outside_particles_background():
    df = _bg_rows(0)
    out = df.loc["whole_image:outside_particles"]
    assert out["mean_intensity"] == 10.0
    assert out["area_px"] == 1600 - 100
    assert df.loc["whole_image", "area_px"] == 1600   # the normal row is unchanged


def test_outside_particles_margin_removes_ring():
    a0 = _bg_rows(0).loc["whole_image:outside_particles", "area_px"]
    a1 = _bg_rows(1).loc["whole_image:outside_particles", "area_px"]
    assert a0 - a1 == 40     # one cross-dilation of a 10×10 square


def test_outside_particles_selection_fully_covered_is_nan():
    roi = np.zeros((40, 40), bool); roi[17:23, 17:23] = True
    out = _bg_rows(0, roi).loc["sel:outside_particles"]
    assert out["area_px"] == 0
    assert np.isnan(out["mean_intensity"]) and np.isnan(out["std_intensity"])


def test_outside_particles_off_by_default():
    plane = np.full((40, 40), 10.0)
    ctx = PipelineContext(channel_names=["a"], masks={"mask_ch0": plane > 50})
    df = IntensityMeasurement(0).process(plane[None], ctx).measurements
    assert list(df["roi_mask"]) == ["whole_image"]


def test_outside_particles_median():
    plane = np.full((40, 40), 10.0); plane[15:25, 15:25] = 100.0
    plane[0, :5] = 1000.0                       # a few unsegmented bright pixels
    ctx = PipelineContext(channel_names=["a"], masks={"mask_ch0": plane == 100.0})
    df = (IntensityMeasurement(0, outside_particles=True, outside_margin_um=0)
          .process(plane[None], ctx).measurements.set_index("roi_mask"))
    out = df.loc["whole_image:outside_particles"]
    assert out["median_intensity"] == 10.0
    assert out["mean_intensity"] > 10.0
