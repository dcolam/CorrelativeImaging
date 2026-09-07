"""Binning: reduce X/Y before the pipeline runs, without changing the physics."""

import numpy as np
import pytest

from correlative_imaging.pipeline import (
    AutoThreshold, Binning, ParticleAnalysis, Pipeline, ZProjection,
)
from correlative_imaging.pipeline.base import PipelineContext


def ctx(px=0.65):
    return PipelineContext(channel_names=["a"], pixel_size_um=px)


@pytest.fixture
def img():
    """One 40×40 px square on a dim background, in a (C, Z, Y, X) stack."""
    a = np.full((1, 3, 128, 128), 100.0, dtype="float32")
    a[:, :, 40:80, 40:80] = 900.0
    return a


# ── mechanics ─────────────────────────────────────────────────────────

def test_factor_1_is_a_noop(img):
    assert Binning(factor=1).process(img, ctx()).image is None


def test_shape_and_pixel_size_scale_together(img):
    c = ctx(0.65)
    out = Binning(factor=2).process(img, c).image
    assert out.shape == (1, 3, 64, 64)
    assert c.pixel_size_um == pytest.approx(1.30)


def test_mean_is_the_block_mean():
    a = np.arange(2 * 1 * 8 * 8, dtype="float32").reshape(2, 1, 8, 8)
    out = Binning(factor=2, method="mean").process(a, ctx()).image
    assert out[0, 0, 0, 0] == pytest.approx(a[0, 0, :2, :2].mean())


def test_non_divisible_shape_is_cropped_not_padded():
    """Padding would make edge pixels averages of fewer than f×f inputs."""
    a = np.ones((1, 1, 9, 7), dtype="float32")
    assert Binning(factor=2).process(a, ctx()).image.shape == (1, 1, 4, 3)


def test_works_without_a_z_axis():
    assert Binning(factor=2).process(np.ones((3, 8, 8), dtype="float32"),
                                     ctx()).image.shape == (3, 4, 4)


def test_z_is_not_binned(img):
    c = ctx()
    out = Binning(factor=2).process(img, c).image
    assert out.shape[1] == img.shape[1]        # Z untouched
    assert c.z_step_um == 1.0                  # and its spacing unchanged


def test_sum_does_not_overflow_an_integer_input():
    u = np.full((1, 1, 8, 8), 60000, dtype="uint16")
    out = Binning(factor=4, method="sum").process(u, ctx()).image
    assert out.max() == pytest.approx(60000 * 16)   # would wrap in uint16


def test_bad_factor_and_method_are_refused(img):
    with pytest.raises(ValueError):
        Binning(factor=999).process(img, ctx())
    with pytest.raises(ValueError):
        Binning(factor=2, method="median").process(img, ctx())


# ── the constraint that actually matters ──────────────────────────────

@pytest.mark.parametrize("factor", [1, 2, 4])
def test_area_in_um2_is_invariant_to_binning(img, factor):
    """A 40×40 px object at 0.65 µm/px is 676 µm². Binned 2× it is 20×20 px at
    1.3 µm/px — still 676 µm². If the pixel-size update fails to reach
    ParticleAnalysis, this is where it shows up."""
    pl = Pipeline()
    if factor > 1:
        pl.add(Binning(factor=factor))
    (pl.add(ZProjection(channel=-1, method="max"))
       .add(AutoThreshold(channel=0, method="otsu"))
       .add(ParticleAnalysis(channel=0)))

    c = ctx(0.65)
    _, res = pl.run(img, c)
    df = res[-1].measurements
    assert len(df) == 1
    assert float(df["area_um2"].iloc[0]) == pytest.approx(40 * 40 * 0.65 ** 2, rel=0.05)
    assert c.pixel_size_um == pytest.approx(0.65 * factor)


def test_binning_shrinks_the_array_the_pipeline_works_on(img):
    pl = Pipeline().add(Binning(factor=4)).add(ZProjection(channel=-1, method="max"))
    final, _ = pl.run(img, ctx())
    assert final.shape == (1, 32, 32)          # 128/4, Z collapsed


# ── Binned FL vs unbinned BF diagnostics ──────────────────────────────

def _write_bf_projection(root, well_id, shape):
    """Stand in for what the BF pipeline writes: an unbinned projection TIF."""
    import tifffile
    d = root / "bf_pipeline" / "projections"
    d.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(d / f"{well_id}_proj.tif"),
                     np.full(shape, 128, dtype="uint8"))
    return root / "diagnostics"


def test_bf_projection_is_binned_onto_the_pipeline_grid(tmp_path):
    """The BF pipeline never bins (its .ilp expects full scale), so with a
    binned analysis run the projection is larger than the planes it is stacked
    with. Unmatched, np.stack raises and the ROI crop boxes index the wrong
    region of the BF."""
    from correlative_imaging.batch import _load_bf_projection

    diag = _write_bf_projection(tmp_path, "A1", (128, 128))
    bf = _load_bf_projection(diag, "A1", target_shape=(32, 32))
    assert bf is not None and bf.shape == (32, 32)      # 4× binned to match


def test_bf_projection_unbinned_run_is_unchanged(tmp_path):
    from correlative_imaging.batch import _load_bf_projection

    diag = _write_bf_projection(tmp_path, "A1", (128, 128))
    assert _load_bf_projection(diag, "A1", target_shape=(128, 128)).shape == (128, 128)
    assert _load_bf_projection(diag, "A1").shape == (128, 128)   # no target given


def test_bf_projection_dropped_when_grids_do_not_match_wholly(tmp_path):
    """A misaligned overlay is worse than no overlay."""
    from correlative_imaging.batch import _load_bf_projection

    diag = _write_bf_projection(tmp_path, "A1", (100, 100))
    # 100 → 30 is no whole factor (100//30 = 3, but 100//3 = 33): refuse.
    assert _load_bf_projection(diag, "A1", target_shape=(30, 30)) is None
    # Nor is a BF projection smaller than the planes it would join.
    assert _load_bf_projection(diag, "A1", target_shape=(200, 200)) is None


def test_bf_binning_matches_what_the_pipeline_did_to_the_planes(tmp_path):
    """When the source size is not divisible, the pipeline crops the remainder —
    the BF plane must be reduced the same way, or it lands half a bin off."""
    from correlative_imaging.batch import _load_bf_projection
    from correlative_imaging.pipeline.preprocess import bin_array

    diag = _write_bf_projection(tmp_path, "A1", (100, 100))
    # A 100-px FL image binned 3× becomes 33 px (99 kept, 1 row cropped).
    assert bin_array(np.zeros((100, 100), dtype="float32"), 3).shape == (33, 33)
    assert _load_bf_projection(diag, "A1", target_shape=(33, 33)).shape == (33, 33)


def test_multichannel_tif_stacks_binned_fl_with_binned_bf(tmp_path):
    """End to end: the composite the binned run actually writes."""
    import tifffile
    from correlative_imaging.batch import _load_bf_projection, _save_multichannel_tif

    diag = _write_bf_projection(tmp_path, "A1", (128, 128))
    planes = [np.zeros((32, 32), dtype="float32"), np.ones((32, 32), dtype="float32")]
    bf = _load_bf_projection(diag, "A1", target_shape=planes[0].shape)
    out = tmp_path / "A1_whole_channels.tif"
    _save_multichannel_tif(planes, ["ch0", "ch1"], bf, out)
    assert tifffile.imread(str(out)).shape == (3, 32, 32)     # 2 FL + 1 BF


# ── ROI placement under binning ───────────────────────────────────────

@pytest.fixture
def roi_files(tmp_path):
    """The three ROI sources the pipeline accepts, all describing the same
    200–400 px square drawn on an unbinned 512×512 image."""
    import json
    import roifile
    import tifffile
    from skimage.measure import find_contours

    m = np.zeros((512, 512), bool)
    m[200:400, 200:400] = True
    contour = find_contours(np.pad(m, 1), 0.5)[0] - 1
    roi = roifile.ImagejRoi.frompoints(np.stack([contour[:, 1], contour[:, 0]], axis=1))
    roi.tofile(str(tmp_path / "with_sidecar.roi"))
    roi.tofile(str(tmp_path / "no_sidecar.roi"))
    (tmp_path / "with_sidecar.roi.json").write_text(json.dumps({"pixel_size_um": 0.65}))
    tifffile.imwrite(str(tmp_path / "mask.tif"), m.astype("uint8") * 255)
    return tmp_path


def _roi_rows(roi_path, factor):
    """Row extent of the loaded ROI after binning by *factor*."""
    from correlative_imaging.pipeline.segment import LoadROI

    pl = Pipeline()
    if factor > 1:
        pl.add(Binning(factor=factor))
    pl.add(ZProjection(channel=-1, method="max"))
    pl.add(LoadROI(path=str(roi_path), roi_name="roi"))

    c = ctx(0.65)
    _, res = pl.run(np.zeros((1, 1, 512, 512), dtype="float32"), c)
    ys, _ = np.where(res[-1].masks["roi"])
    return int(ys.min()), int(ys.max())


@pytest.mark.parametrize("factor", [1, 2, 4])
@pytest.mark.parametrize("name", ["with_sidecar.roi", "no_sidecar.roi", "mask.tif"])
def test_roi_lands_in_the_right_place_when_binned(roi_files, name, factor):
    """A square at rows 200–400 of the unbinned image must appear at
    200/f–400/f after binning, whichever way the ROI was supplied:

    * .roi + sidecar — rescaled by recorded pixel size / context pixel size
    * .roi, no sidecar — rescaled by the binning factor recorded on the context
      (without this it keeps full-resolution coordinates and is clipped into
      the corner: at 4× it collapsed to a single row)
    * .tif mask — resized to the image shape, so it never depended on pixel size
    """
    lo, hi = _roi_rows(roi_files / name, factor)
    assert lo == pytest.approx(200 // factor, abs=2)
    assert hi == pytest.approx(400 // factor, abs=2)


def test_binning_factor_is_cumulative_on_the_context():
    """Two Binning steps must compose, or a later ROI is scaled by only one."""
    c = ctx(0.5)
    img = np.zeros((1, 64, 64), dtype="float32")
    img = Binning(factor=2).process(img, c).image
    Binning(factor=2).process(img, c)
    assert c.metadata["binning_factor"] == 4
    assert c.pixel_size_um == pytest.approx(2.0)


# ── GUI wiring after the Preprocessing/Channels split ─────────────────

def test_whole_image_controls_still_drive_the_pipeline(qapp):
    """Binning and Z-projection moved to their own tab page; they must still
    be the first two steps and still round-trip through the pipeline dict."""
    from correlative_imaging.viewer.gui import CorrelativeImagingWidget

    w = CorrelativeImagingWidget()
    ch = w._channels_tab
    ch.set_channels(["DAPI", "GFP"])
    ch._bin_combo.setCurrentIndex(ch._bin_combo.findText("4×"))
    ch._zproj_from.setValue(2)
    ch._zproj_to.setValue(9)

    steps = w.build_pipeline_dict()["steps"]
    assert [s["type"] for s in steps[:2]] == ["Binning", "ZProjection"]
    assert steps[0]["factor"] == 4
    assert (steps[1]["z_start"], steps[1]["z_stop"]) == (2, 9)
    # and they live on the Preprocessing page, not the Channels page
    assert ch._bin_combo.parent() is not None
    assert w._tabs.tabText(3) == "Preprocessing"


def test_loading_another_sample_keeps_channel_settings(qapp):
    """'Load selected well as sample' used to destroy and rebuild every channel
    panel, silently discarding all configuration."""
    from correlative_imaging.viewer.gui import CorrelativeImagingWidget

    ch = CorrelativeImagingWidget()._channels_tab
    ch.set_channels(["DAPI", "GFP", "mCherry"])
    p0 = ch.get_panels()[0]
    p0._blur_sigma.setValue(7.5)
    p0._name_edit.setText("MyName")
    before = p0.get_preprocess_steps()

    ch.set_channels(["w1", "w2", "w3"])          # another well, same count
    assert ch.get_panels()[0] is p0
    assert p0.get_preprocess_steps() == before
    assert p0.display_name == "MyName"           # user's name survives

    ch.set_channels(["a", "b"])                  # different count → rebuild
    assert len(ch.get_panels()) == 2


def test_all_colocalization_pairs_are_ordered_and_deduped(qapp):
    from correlative_imaging.viewer.gui import CorrelativeImagingWidget

    c = CorrelativeImagingWidget()._combine_tab
    c.set_channels(["a", "b", "c"])
    c._add_all_pairs()
    steps = c.get_coloc_steps("")
    assert len(steps) == 6                        # n*(n-1), ordered
    assert all(s["primary_channel"] != s["secondary_channel"] for s in steps)
    assert all(s["dilation_um"] == 0.0 for s in steps)
    c._add_all_pairs()
    assert len(c.get_coloc_steps("")) == 6        # idempotent
