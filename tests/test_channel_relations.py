"""ChannelRelation: own signal vs a bounded copy of the other channels inside a selection."""
import json

import numpy as np
import pandas as pd
import pytest

from correlative_imaging.pipeline import Pipeline, PipelineContext, Step
from correlative_imaging.pipeline.analyze import ParticleAnalysis
from correlative_imaging.pipeline.relations import ChannelRelation, residual_key, topography_contours
from correlative_imaging.storage.database import ResultsDB


def _scene(copy=0.03, seed=0):
    """3 channels, 80x80, noise SD 2 around a background of 10. A cell of ch0 (+400) in the selection; ch1 holds
    ``copy`` x ch0's signal plus a blob of its own (+60); ch2 is noise only. Every channel is thresholded on the whole
    image (cell + blob), so the background band and the noise come from outside them."""
    rng = np.random.default_rng(seed)
    img = 10 + rng.normal(0, 2, (3, 80, 80))
    yy, xx = np.mgrid[:80, :80]
    cell = (yy - 40) ** 2 + (xx - 34) ** 2 <= 8 ** 2
    blob = (yy - 40) ** 2 + (xx - 48) ** 2 <= 3 ** 2
    img[0][cell] += (400 + 20 * (xx - 34))[cell]          # a cell with structure (a gradient)
    img[1] += copy * (img[0] - 10)
    img[1][blob] += 60
    sel = (yy - 40) ** 2 + (xx - 40) ** 2 <= 18 ** 2
    ctx = PipelineContext(channel_names=["GFP", "mCherry", "BFP"], pixel_size_um=1.0)
    ctx.masks.update({"roi_hole": sel, "mask_ch0": cell, "mask_ch1": blob | cell, "mask_ch2": np.zeros_like(cell)})
    return img.astype(np.float32), ctx, cell, blob, sel


def _row(df, ch, other, label=0):
    return df[(df.channel == ch) & (df.other_channel == other) & (df.object_label == label)].iloc[0]


def test_a_copy_is_recovered_and_the_own_blob_is_the_residual():
    img, ctx, cell, blob, sel = _scene(copy=0.03)
    res = ChannelRelation(roi_mask="roi_hole").process(img, ctx)
    df = res.measurements
    assert _row(df, "mCherry", "GFP").copy_fraction == pytest.approx(0.03, abs=0.004)
    r = res.masks[residual_key(1, "roi_hole")]
    noise = _row(df, "mCherry", "GFP").noise
    assert np.median(r[blob]) == pytest.approx(60 / noise, rel=0.15)      # the blob is mCherry's own signal
    assert abs(np.median(r[cell & ~blob])) < 1.0                           # the copy of GFP is explained away
    # GFP fitted as a copy of the 33x dimmer mCherry may use the full bound, but that explains next to nothing
    assert _row(df, "GFP", "mCherry").explained_fraction < 0.05
    assert not r[~sel].any()                                               # 0 outside the selection
    assert df.background_source.iloc[0].startswith("band 2-6")


def test_a_copy_above_the_bound_is_capped_and_leaves_a_positive_residual():
    img, ctx, cell, blob, _ = _scene(copy=0.10)
    res = ChannelRelation(roi_mask="roi_hole", max_copy=0.05).process(img, ctx)
    assert _row(res.measurements, "mCherry", "GFP").copy_fraction == pytest.approx(0.05, abs=1e-6)
    r = res.masks[residual_key(1, "roi_hole")]
    assert np.median(r[cell & ~blob]) > 5                                  # the other 5 % stays as own signal
    # a per-pair limit overrides the general one
    res2 = ChannelRelation(roi_mask="roi_hole", max_copy=0.05, max_copy_pairs=[[1, 0, 0.01]]).process(img, ctx)
    assert _row(res2.measurements, "mCherry", "GFP").copy_fraction == pytest.approx(0.01, abs=1e-6)


def test_clipped_pixels_are_left_out():
    img, ctx, cell, blob, sel = _scene(copy=0.03)
    base = ChannelRelation(roi_mask="roi_hole").process(img, ctx).measurements
    clip = np.zeros(cell.shape, bool); clip[38:42, 30:34] = True
    img2 = img.copy(); img2[1][clip] = 65535
    ctx.masks["saturated_ch1"] = clip
    res = ChannelRelation(roi_mask="roi_hole").process(img2, ctx)
    assert not res.masks[residual_key(1, "roi_hole")][clip].any()
    assert _row(res.measurements, "mCherry", "GFP").copy_fraction == pytest.approx(
        _row(base, "mCherry", "GFP").copy_fraction, abs=0.002)


def test_particles_of_this_selection_only_and_their_pixel_relations():
    img, ctx, cell, blob, _ = _scene(copy=0.03)
    ctx.masks["particles_ch1"] = np.where(blob, 7, 0)
    ctx.metadata["particles_roi"] = {1: "elsewhere"}                       # belongs to another selection
    df = ChannelRelation(roi_mask="roi_hole").process(img, ctx).measurements
    assert set(df.object_label) == {0}
    ParticleAnalysis(channel=0, min_size_um2=0, roi_mask="roi_hole").process(img, ctx)   # claims particles_ch0
    ctx.masks["particles_ch0"] = np.where(cell, 3, 0)
    df = ChannelRelation(roi_mask="roi_hole").process(img, ctx).measurements
    p = _row(df, "mCherry", "GFP", label=3)
    assert p.object_channel == "GFP" and p.n_px == int(cell.sum())
    assert p.slope == pytest.approx(0.03, abs=0.01) and p.pearson_r > 0.3   # mCherry follows GFP inside the cell


def test_pipeline_json_round_trip_and_old_pipelines_unchanged(tmp_path):
    pl = Pipeline("t").add(ChannelRelation(roi_mask="roi_hole", max_copy_pairs=[[0, 1, 0.02]]))
    pl.save(tmp_path / "p.json")
    s = Pipeline.load(tmp_path / "p.json").steps[0]
    assert isinstance(s, ChannelRelation) and s.bound(0, 1) == 0.02 and s.bound(1, 0) == 0.05
    assert s.band_um == [2.0, 6.0]
    assert ChannelRelation().band_um is not ChannelRelation().band_um        # no shared mutable default
    assert "ChannelRelation" not in json.dumps(Step.from_dict({"type": "ParticleAnalysis", "channel": 0}).to_dict())


def test_relations_reach_the_database(tmp_path):
    img, ctx, *_ = _scene()
    df = ChannelRelation(roi_mask="roi_hole").process(img, ctx).measurements
    db = ResultsDB(tmp_path / "r.db")
    iid = db.register_image(tmp_path / "x.tif", n_channels=3, pixel_size_um=1.0)
    db.save_channel_relations(iid, df)
    out = pd.read_sql("select * from channel_relations", db._con)
    assert len(out) == 6 and set(out.channel) == {"GFP", "mCherry", "BFP"}
    assert out.copy_fraction.notna().all() and (out.background_source == df.background_source.iloc[0]).all()


def test_topography_lines_lie_inside_the_selection():
    img, ctx, cell, blob, sel = _scene()
    lines = topography_contours(img[0], sel, 7)
    assert lines and all(0 < lv < 1 for lv, _ in lines)
    pts = np.vstack([p for _, p in lines]).round().astype(int)
    assert sel[pts[:, 0], pts[:, 1]].mean() > 0.95


def test_gui_relation_controls(qapp):
    from correlative_imaging.viewer.gui import CombineTab

    t = CombineTab()
    t.set_channels(["GFP", "mCherry", "BFP"])
    assert t.get_relation_step("") == []                                   # never for the whole image
    t._add_rel_pair(0, 1, 2.0)
    s = t.get_relation_step("roi_hole")[0]
    assert s["type"] == "ChannelRelation" and s["max_copy"] == 0.05 and s["max_copy_pairs"] == [[0, 1, 0.02]]
    Step.from_dict(s)                                                       # a valid step
    t.set_relations({"steps": [{**s, "max_copy": 0.03, "max_copy_pairs": [[2, 1, 0.01]]}]})
    s2 = t.get_relation_step("roi_hole")[0]
    assert s2["max_copy"] == pytest.approx(0.03) and s2["max_copy_pairs"] == [[2, 1, pytest.approx(0.01)]]
    t.set_relations({"steps": []})                                          # an old pipeline: off
    assert t.get_relation_step("roi_hole") == []


def test_diagnostic_crops_get_the_residual_tif_and_the_relations_png(tmp_path):
    import types

    import tifffile

    from correlative_imaging.batch import _save_well_diagnostics

    img, ctx, *_ = _scene()
    res = ChannelRelation(roi_mask="roi_hole").process(img, ctx)
    ctx.masks.update(res.masks)
    cfg = {"output_dir": str(tmp_path), "formats": {"jpg"}, "colors": ["green", "red", "blue"], "crops": True,
           "crop_pad_um": 2.0, "crop_selections": None, "multichannel_tif": True}
    crops = _save_well_diagnostics(types.SimpleNamespace(well_id="A01"), img, 1.0,
                                   {"steps": [{"type": "LoadROI", "roi_name": "roi_hole"}]}, ctx, cfg)
    files = crops["roi_hole"]["files"]
    assert "A01_roi_hole_crop_residual.tif" in files and "A01_roi_hole_crop_relations.png" in files
    r0, r1, c0, c1 = crops["roi_hole"]["bbox"]
    stack = tifffile.imread(tmp_path / "A01_roi_hole_crop_residual.tif")
    assert stack.shape == (3, r1 - r0, c1 - c0)
    assert np.allclose(stack[1], res.masks[residual_key(1, "roi_hole")][r0:r1, c0:c1])
    assert (tmp_path / "A01_roi_hole_crop_relations.png").stat().st_size > 1000
