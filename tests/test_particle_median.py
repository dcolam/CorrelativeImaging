"""Per-particle median (robust to specks) in ParticleAnalysis, ColocalizationAnalysis and the database."""
import numpy as np
import pandas as pd

from correlative_imaging.pipeline.analyze import ParticleAnalysis
from correlative_imaging.pipeline.base import PipelineContext
from correlative_imaging.pipeline.colocalize import ColocalizationAnalysis
from correlative_imaging.storage.database import ResultsDB


def _image():
    img = np.zeros((2, 40, 40), np.float32)
    img[0, 5:15, 5:15] = 10.0                       # a cell of 100 px at 10 ...
    img[0, 6, 6] = img[0, 7, 7] = 1000.0            # ... with two bright specks
    img[1, 5:15, 5:15] = 4.0
    img[1, 8, 8] = 500.0
    return img


def _context(img):
    ctx = PipelineContext(pixel_size_um=0.5, channel_names=["A", "B"])
    mask = np.zeros(img.shape[1:], np.int32)
    mask[5:15, 5:15] = 1
    ctx.masks["mask_ch0"] = mask
    ctx.masks["mask_ch1"] = mask
    return ctx


def test_particle_median_ignores_specks_and_is_kept_with_an_old_metric_list():
    img = _image()
    df = ParticleAnalysis(channel=0, min_size_um2=0, metrics=["area", "mean_intensity"]).process(img, _context(img)).measurements
    assert df.mean_intensity.iloc[0] > 25                       # the specks pull the mean up
    assert df.median_intensity.iloc[0] == 10.0                  # not the median
    assert "ring_median_intensity" in df and "p75_intensity" in df


def test_robust_metrics_are_selectable():
    img = _image()
    df = ParticleAnalysis(channel=0, min_size_um2=0, metrics=["area", "median_intensity"]).process(
        img, _context(img)).measurements
    assert "median_intensity" in df
    assert not {"std_intensity", "p25_intensity", "ring_median_intensity", "mean_intensity"} & set(df.columns)


def test_colocalization_median_and_database_columns(tmp_path):
    img = _image()
    ctx = _context(img)
    ctx.masks["particles_ch0"] = ctx.masks["mask_ch0"]
    res = ColocalizationAnalysis(primary_channel=0, secondary_channel=1).process(img, ctx)
    per = res.measurements
    assert per.secondary_median_intensity.iloc[0] == 4.0
    db = ResultsDB(tmp_path / "r.db")
    iid = db.register_image(tmp_path / "x.tif", n_channels=2, pixel_size_um=0.5)
    pa = ParticleAnalysis(channel=0, min_size_um2=0).process(img, ctx).measurements
    db.save_particle_measurements(iid, pa)
    db.save_colocalization(iid, per, res.info["global_stats"])
    got = pd.read_sql("select median_intensity, std_intensity, p25_intensity, p75_intensity, ring_px "
                      "from particle_measurements", db._con)
    assert got.median_intensity.iloc[0] == 10.0 and got.p25_intensity.iloc[0] == 10.0
    assert got.std_intensity.iloc[0] > 0 and got.ring_px.iloc[0] > 0
    got = pd.read_sql("select secondary_median_intensity, secondary_p75_intensity, secondary_ring_median_intensity "
                      "from colocalization_per_particle", db._con)
    assert got.secondary_median_intensity.iloc[0] == 4.0 and got.secondary_p75_intensity.iloc[0] == 4.0
    assert got.secondary_ring_median_intensity.iloc[0] == 0.0


def test_spread_percentiles_and_local_ring():
    img = np.zeros((2, 60, 60), np.float32)
    img[0, 20:30, 20:30] = 50.0                        # a cell
    img[0, 15:35, 15:35] += 0.0
    img[0, 10:40, 10:40] += 7.0                         # glow of a bright neighbour around it
    img[0, 20:30, 20:30] = 50.0
    img[0, 22:24, 22:24] = 400.0                        # a speck
    img[1, 20:30, 20:30] = 3.0
    img[1, 10:40, 10:40] += 9.0                         # the other channel also glows around the cell
    img[1, 20:30, 20:30] = 3.0
    ctx = PipelineContext(pixel_size_um=0.5, channel_names=["A", "B"])
    m = np.zeros(img.shape[1:], np.int32)
    m[20:30, 20:30] = 1
    ctx.masks["mask_ch0"] = m
    df = ParticleAnalysis(channel=0, min_size_um2=0).process(img, ctx).measurements
    r = df.iloc[0]
    assert r.median_intensity == 50.0 and r.p25_intensity == 50.0 and r.p75_intensity == 50.0
    assert r.std_intensity > 50                         # the speck
    assert r.ring_median_intensity == 7.0 and r.ring_px > 100     # the glow right next to the cell
    ctx.masks["particles_ch0"] = m
    per = ColocalizationAnalysis(primary_channel=0, secondary_channel=1).process(img, ctx).measurements
    assert per.secondary_median_intensity.iloc[0] == 3.0
    assert per.secondary_ring_median_intensity.iloc[0] == 9.0
    # inside a selection the ring stays inside it: a cell filling the selection has no ring
    ctx.masks["roi_hole"] = m.astype(bool)
    df = ParticleAnalysis(channel=0, min_size_um2=0, roi_mask="roi_hole").process(img, ctx).measurements
    assert np.isnan(df.ring_median_intensity.iloc[0]) and df.ring_px.iloc[0] == 0
