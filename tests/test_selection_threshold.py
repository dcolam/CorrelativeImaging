"""Per-selection thresholding: threshold from the selection, applied to the whole frame."""

import numpy as np
import pytest

from correlative_imaging.pipeline import Pipeline
from correlative_imaging.pipeline.analyze import IntensityMeasurement, ParticleAnalysis
from correlative_imaging.pipeline.base import PipelineContext, Step
from correlative_imaging.pipeline.colocalize import ColocalizationAnalysis
from correlative_imaging.pipeline.segment import AutoThreshold, WatershedSplit


def _frame():
    """A bright cell far away and a dim cell in the hole. The global (Otsu)
    threshold sits between dim and bright, so the dim cell is lost; the
    hole's own threshold finds it. The dim cell crosses the hole edge."""
    a = np.full((120, 120), 5.0, np.float32)
    a[5:55, 5:55] = 1000.0             # bright cell, outside the hole
    a[70:90, 80:100] = 10.0            # dim cell, half inside the hole
    hole = np.zeros_like(a, bool); hole[60:100, 50:90] = True
    return a, hole


def _ctx(hole):
    return PipelineContext(channel_names=["a"], masks={"roi_hole": hole})


def _pipeline(own: bool):
    p = Pipeline().add(AutoThreshold(0, min_size=0))
    if own:
        p.add(AutoThreshold(0, min_size=0, roi_mask="roi_hole"))
    return (p.add(ParticleAnalysis(0, min_size_um2=0, max_size_um2=0, roi_mask="roi_hole"))
             .add(ParticleAnalysis(0, min_size_um2=0, max_size_um2=0)))


def test_global_threshold_misses_dim_cell():
    a, hole = _frame()
    _, res = _pipeline(own=False).run(a[None], _ctx(hole))
    assert res[1].measurements.empty


def test_own_threshold_finds_dim_cell_and_keeps_it_whole():
    a, hole = _frame()
    ctx = _ctx(hole)
    _, res = _pipeline(own=True).run(a[None], ctx)
    thr_global, thr_hole = res[0].info["threshold_value"], res[1].info["threshold_value"]
    assert thr_hole < 10 < thr_global
    assert "mask_ch0@roi_hole" in ctx.masks and "mask_ch0" in ctx.masks

    hole_df = res[2].measurements
    assert len(hole_df) == 1
    row = hole_df.iloc[0]
    assert row["area"] == 200                      # cols 80–89 of the 20×20 cell
    assert row["area_full_um2"] == 400
    assert bool(row["cut_at_edge"])

    # the whole-image selection keeps the global threshold
    whole_df = res[3].measurements
    assert len(whole_df) == 1 and whole_df.iloc[0]["area"] == 2500


def test_empty_selection_gives_empty_mask():
    a, _ = _frame()
    ctx = PipelineContext(channel_names=["a"], masks={"roi_x": np.zeros_like(a, bool)})
    r = AutoThreshold(0, roi_mask="roi_x").process(a[None], ctx)
    assert np.isnan(r.info["threshold_value"]) and not r.masks["mask_ch0@roi_x"].any()


def test_missing_selection_is_an_error():
    a, _ = _frame()
    with pytest.raises(RuntimeError, match="roi_missing"):
        AutoThreshold(0, roi_mask="roi_missing").process(a[None], PipelineContext(channel_names=["a"]))


def test_watershed_and_downstream_use_selection_mask():
    a, hole = _frame()
    b = np.zeros_like(a); b[70:90, 80:85] = 50.0   # secondary object over part of the dim cell
    b[5:55, 5:55] = 5000.0
    img = np.stack([a, b])
    ctx = PipelineContext(channel_names=["a", "b"], masks={"roi_hole": hole})
    pl = Pipeline()
    for c in (0, 1):
        pl.add(AutoThreshold(c, min_size=0)).add(AutoThreshold(c, min_size=0, roi_mask="roi_hole"))
    pl.add(WatershedSplit(0, roi_mask="roi_hole"))
    pl.add(ParticleAnalysis(0, min_size_um2=0, max_size_um2=0, roi_mask="roi_hole"))
    pl.add(ColocalizationAnalysis(0, 1, roi_mask="roi_hole"))
    pl.add(IntensityMeasurement(0, roi_mask="roi_hole", outside_particles=True, outside_margin_um=0))
    _, res = pl.run(img, ctx)

    assert ctx.masks["mask_ch0@roi_hole"].max() >= 1      # watershed wrote labels to the selection mask
    coloc = res[-2].measurements.iloc[0]
    assert coloc["mask_overlap_fraction"] == pytest.approx(0.5)   # cols 80–84 of 80–89
    out = res[-1].measurements.set_index("roi_mask").loc["roi_hole:outside_particles"]
    # the dim cell's 200 px inside the hole are excluded from the background
    assert out["area_px"] == hole.sum() - 200


def test_old_json_without_roi_mask_loads():
    s = Step.from_dict({"type": "AutoThreshold", "channel": 0, "method": "otsu",
                        "z_projection": "max", "min_size": 50})
    assert s.roi_mask == "" and s.name == "auto_threshold_ch0"
