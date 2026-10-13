"""ColocalizationAnalysis: Pearson inside an ROI, overlap against the secondary mask."""

import numpy as np
import pytest

from correlative_imaging.pipeline.base import PipelineContext
from correlative_imaging.pipeline.colocalize import ColocalizationAnalysis


def test_pearson_uses_roi_pixels_only():
    rng = np.random.default_rng(0)
    a = rng.random((64, 64)); b = 1 - a            # anti-correlated everywhere ...
    roi = np.zeros((64, 64), bool); roi[20:30, 20:30] = True
    b[roi] = a[roi]                                 # ... except identical inside the ROI
    img = np.stack([a, b])
    ctx = PipelineContext(channel_names=["p", "s"], masks={"r": roi,
          "particles_ch0": roi.astype(np.int32)})
    res = ColocalizationAnalysis(0, 1, roi_mask="r").process(img, ctx)
    assert res.info["global_stats"]["pearson_r"] == pytest.approx(1.0, abs=1e-6)
    assert res.info["global_stats"]["pearson_mode"] == "roi_pixels"


def test_pearson_whole_image_unchanged():
    rng = np.random.default_rng(0)
    a = rng.random((64, 64)); b = 1 - a
    img = np.stack([a, b])
    ctx = PipelineContext(channel_names=["p", "s"],
                          masks={"particles_ch0": np.ones((64, 64), np.int32)})
    res = ColocalizationAnalysis(0, 1).process(img, ctx)
    assert res.info["global_stats"]["pearson_r"] == pytest.approx(-1.0, abs=1e-6)
    assert res.info["global_stats"]["pearson_mode"] == "whole_image"


def _half_covered():
    """A 10×10 primary particle, its left half inside a secondary object;
    the secondary channel is +1 everywhere (blur/black-level residue)."""
    p = np.zeros((40, 40)); s = np.ones((40, 40))
    particle = np.zeros((40, 40), np.int32); particle[10:20, 10:20] = 1
    sec_obj = np.zeros((40, 40), bool); sec_obj[5:25, 5:15] = True
    p[particle > 0] = 50; s[sec_obj] = 80
    return np.stack([p, s]), particle, sec_obj


def test_mask_overlap_fraction():
    img, particle, sec_obj = _half_covered()
    ctx = PipelineContext(channel_names=["p", "s"],
                          masks={"particles_ch0": particle, "mask_ch1": sec_obj})
    row = ColocalizationAnalysis(0, 1).process(img, ctx).measurements.iloc[0]
    assert row["overlap_fraction"] == pytest.approx(1.0)
    assert row["mask_overlap_fraction"] == pytest.approx(0.5)


def test_mask_overlap_fraction_restricted_to_roi():
    img, particle, sec_obj = _half_covered()
    roi = np.zeros((40, 40), bool); roi[:, 12:] = True   # keeps cols 12–19 of the particle
    ctx = PipelineContext(channel_names=["p", "s"],
                          masks={"particles_ch0": particle, "mask_ch1": sec_obj, "r": roi})
    row = ColocalizationAnalysis(0, 1, roi_mask="r").process(img, ctx).measurements.iloc[0]
    # 8 particle columns in the ROI, 3 of them (12–14) in the secondary object
    assert row["mask_overlap_fraction"] == pytest.approx(3 / 8)


def test_mask_overlap_fraction_nan_without_secondary_mask():
    img, particle, _ = _half_covered()
    ctx = PipelineContext(channel_names=["p", "s"], masks={"particles_ch0": particle})
    row = ColocalizationAnalysis(0, 1).process(img, ctx).measurements.iloc[0]
    assert np.isnan(row["mask_overlap_fraction"])
