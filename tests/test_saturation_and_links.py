"""Saturation flags (raw -> binned -> projected), their use in thresholds and measurements, the robust background
SD, and the link from a selection's own-threshold particles to the whole-image threshold."""
import numpy as np
import pandas as pd

from correlative_imaging.pipeline import Binning, Pipeline, PipelineContext, ZProjection
from correlative_imaging.pipeline.analyze import IntensityMeasurement, ParticleAnalysis, resolve_intensity_metrics
from correlative_imaging.pipeline.colocalize import ColocalizationAnalysis
from correlative_imaging.pipeline.saturation import SaturationMask
from correlative_imaging.pipeline.segment import AutoThreshold
from correlative_imaging.storage.database import ResultsDB


def _stack():
    """(C=2, Z=6, 16, 16) uint16; channel 0 clipped in plane 5 at (4:6, 4:6) and in plane 1 at (10:12, 10:12)."""
    a = np.full((2, 6, 16, 16), 100, np.uint16)
    a[0, 4, 4:6, 4:6] = 65535
    a[0, 0, 10:12, 10:12] = 65535
    return a


def _run(steps, img, px=0.5):
    ctx = PipelineContext(channel_names=["a", "b"], pixel_size_um=px)
    pl = Pipeline("t")
    for s in steps:
        pl.add(s)
    out, res = pl.run(img, ctx)
    return out, res, ctx


def test_saturation_follows_binning_and_the_z_range():
    _, _, ctx = _run([SaturationMask(grow_px=0), Binning(factor=2), ZProjection(method="max", z_start=3, z_stop=6)], _stack())
    m = ctx.masks["saturated_ch0"]
    assert m.shape == (8, 8)
    assert m[2:3, 2:3].all() and m.sum() == 1          # plane 5 is in the range, binned 2x2 -> one pixel
    assert not m[5, 5]                                  # plane 1 is outside the range
    assert not ctx.masks["saturated_ch1"].any()


def test_saturation_grow_and_old_pipelines_unchanged():
    _, _, ctx = _run([SaturationMask(grow_px=1), ZProjection(method="max")], _stack())
    assert ctx.masks["saturated_ch0"].sum() > 8          # 2 clipped 2x2 spots, each grown by 1 px
    _, _, ctx2 = _run([ZProjection(method="max")], _stack())
    assert not any(k.startswith("saturated") for k in ctx2.masks)


def test_clipped_pixels_do_not_set_the_threshold():
    from skimage.filters import threshold_otsu
    rng = np.random.default_rng(1)
    img = rng.normal(100, 10, (1, 40, 40)).astype(np.float32)
    img[0, 5:15, 5:15] += 300.0                          # an object
    img[0, 20:40, 20:40] = 65535.0                       # a large clipped area
    ctx = PipelineContext(channel_names=["a"], pixel_size_um=1.0)
    ctx.masks.update(SaturationMask(grow_px=0).process(img, ctx).masks)
    sat = ctx.masks["saturated_ch0"]
    with_sat = AutoThreshold(channel=0, method="otsu", min_size=0).process(img, ctx)
    ctx2 = PipelineContext(channel_names=["a"], pixel_size_um=1.0)
    without = AutoThreshold(channel=0, method="otsu", min_size=0).process(img, ctx2)
    assert with_sat.info["threshold_value"] == float(threshold_otsu(img[0][~sat]))   # only unclipped pixels
    assert with_sat.info["threshold_value"] != without.info["threshold_value"]
    assert with_sat.info["saturated_px"] == 400
    assert with_sat.masks["mask_ch0"][sat].all()          # clipped pixels are still foreground


def test_particle_and_bulk_values_leave_clipped_pixels_out():
    img = np.zeros((2, 30, 30), np.float32)
    img[0, 5:15, 5:15] = 100.0
    img[0, 5:7, 5:7] = 65535.0                           # 4 clipped px inside the particle
    img[1, 5:15, 5:15] = 50.0
    img[1, 8:10, 8:10] = 65535.0                         # 4 clipped px of the other channel
    ctx = PipelineContext(channel_names=["a", "b"], pixel_size_um=1.0)
    m = np.zeros((30, 30), np.int32); m[5:15, 5:15] = 1
    ctx.masks["mask_ch0"] = m
    ctx.masks["saturated_ch0"] = img[0] >= 65535
    ctx.masks["saturated_ch1"] = img[1] >= 65535
    df = ParticleAnalysis(channel=0, min_size_um2=0).process(img, ctx).measurements
    r = df.iloc[0]
    assert r.saturated_px == 4 and r.mean_intensity == 100.0 and r.median_intensity == 100.0 and r.std_intensity == 0
    assert r.max_intensity == 65535.0                    # max keeps it: the particle reached the detector limit
    ctx.masks["particles_ch0"] = m
    per = ColocalizationAnalysis(primary_channel=0, secondary_channel=1).process(img, ctx).measurements
    assert per.secondary_saturated_px.iloc[0] == 4 and per.secondary_mean_intensity.iloc[0] == 50.0
    bulk = IntensityMeasurement(channel=0).process(img, ctx).measurements.iloc[0]
    assert bulk.saturated_px == 4
    assert bulk.sum_intensity == float(img[0].sum())     # the sum keeps every pixel


def test_robust_sd_ignores_bright_outliers_and_old_lists_get_it():
    rng = np.random.default_rng(0)
    img = rng.normal(0, 5, (1, 100, 100)).astype(np.float32)
    img[0, :10, :10] += 400                              # a dim unsegmented cell inflates the plain SD
    ctx = PipelineContext(channel_names=["a"], pixel_size_um=1.0)
    ctx.masks["mask_ch0"] = np.zeros((100, 100), bool)
    rows = IntensityMeasurement(channel=0, metrics=["mean_intensity", "std_intensity"], outside_particles=True
                                ).process(img, ctx).measurements
    bg = rows[rows.roi_mask == "whole_image:outside_particles"].iloc[0]
    assert bg.std_intensity > 30 and 4 < bg.robust_sd_intensity < 6
    assert "robust_sd_intensity" in resolve_intensity_metrics(["mean_intensity"])
    assert resolve_intensity_metrics(["robust_sd_intensity"]) == ["robust_sd_intensity"]


def test_selection_particles_are_linked_to_the_whole_image_threshold():
    img = np.zeros((1, 40, 40), np.float32)
    ctx = PipelineContext(channel_names=["a"], pixel_size_um=1.0)
    roi = np.zeros((40, 40), bool); roi[0:20, 0:20] = True
    glob = np.zeros((40, 40), bool); glob[2:10, 2:10] = True          # found by the whole-image threshold
    own = np.zeros((40, 40), bool); own[2:12, 2:10] = True; own[14:18, 14:18] = True   # bigger + one dim extra
    img[0][own] = 10
    ctx.masks.update({"roi": roi, "mask_ch0": glob, "mask_ch0@roi": own})
    df = ParticleAnalysis(channel=0, min_size_um2=0, roi_mask="roi").process(img, ctx).measurements.sort_values("area")
    small, big = df.iloc[0], df.iloc[1]
    assert small.whole_image_iou == 0.0                  # not found by the whole-image threshold
    assert big.whole_image_label == 1 and abs(big.whole_image_iou - 64 / 80) < 1e-9
    whole = ParticleAnalysis(channel=0, min_size_um2=0).process(img, ctx).measurements
    assert whole.whole_image_iou.isna().all()            # whole-image particles have nothing to link to


def test_new_columns_reach_the_database(tmp_path):
    img = np.zeros((2, 30, 30), np.float32)
    img[0, 5:15, 5:15] = 100.0; img[0, 5:7, 5:7] = 65535.0
    ctx = PipelineContext(channel_names=["a", "b"], pixel_size_um=1.0)
    m = np.zeros((30, 30), np.int32); m[5:15, 5:15] = 1
    roi = np.ones((30, 30), bool)
    ctx.masks.update({"mask_ch0": m, "mask_ch0@roi": m, "roi": roi, "particles_ch0": m,
                      "saturated_ch0": img[0] >= 65535, "saturated_ch1": np.zeros((30, 30), bool)})
    db = ResultsDB(tmp_path / "r.db")
    iid = db.register_image(tmp_path / "x.tif", n_channels=2, pixel_size_um=1.0)
    db.save_particle_measurements(iid, ParticleAnalysis(channel=0, min_size_um2=0, roi_mask="roi").process(img, ctx).measurements)
    db.save_intensity_measurements(iid, IntensityMeasurement(channel=0, outside_particles=True).process(img, ctx).measurements)
    res = ColocalizationAnalysis(primary_channel=0, secondary_channel=1).process(img, ctx)
    db.save_colocalization(iid, res.measurements, res.info["global_stats"])
    p = pd.read_sql("select saturated_px, whole_image_label, whole_image_iou from particle_measurements", db._con).iloc[0]
    assert p.saturated_px == 4 and p.whole_image_label == 1 and p.whole_image_iou == 1.0
    i = pd.read_sql("select robust_sd_intensity, saturated_px from intensity_measurements", db._con)
    assert i.saturated_px.iloc[0] == 4 and i.robust_sd_intensity.notna().all()
    c = pd.read_sql("select secondary_saturated_px from colocalization_per_particle", db._con)
    assert c.secondary_saturated_px.iloc[0] == 0
