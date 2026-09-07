"""Z sub-stack selection: the Channels-tab projection step and the BF pipeline."""

import numpy as np
import pytest
import tifffile

from correlative_imaging.pipeline import Pipeline, ZProjection
from correlative_imaging.pipeline.base import PipelineContext
from correlative_imaging.pipeline.ilastik import select_z_range


@pytest.fixture
def stack():
    """(C, Z, Y, X) with a distinct constant per plane, so a projection's value
    identifies exactly which planes went into it."""
    arr = np.zeros((2, 8, 4, 4), dtype="uint16")
    for z in range(8):
        arr[:, z] = (z + 1) * 10
    return arr


def _ctx():
    return PipelineContext(channel_names=["a", "b"])


def _write_stack(path):
    """An 8-plane Z-stack, one constant value per plane. ImageJ axis tags are
    required for bioio to report it as (C, Z, Y, X) rather than collapsing it."""
    arr = np.zeros((8, 4, 4), dtype="uint16")
    for z in range(8):
        arr[z] = (z + 1) * 10
    tifffile.imwrite(path, arr, imagej=True, metadata={"axes": "ZYX"})
    return path


# ── select_z_range ────────────────────────────────────────────────────

def test_range_is_1_based_inclusive(stack):
    sub = select_z_range(stack, 3, 5, axis=1)
    assert sub.shape[1] == 3
    assert sub[0, 0, 0, 0] == 30 and sub[0, -1, 0, 0] == 50


def test_zero_means_unbounded(stack):
    assert select_z_range(stack, 0, 0, axis=1) is stack
    assert select_z_range(stack, 0, 3, axis=1).shape[1] == 3
    assert select_z_range(stack, 6, 0, axis=1).shape[1] == 3


def test_range_wider_than_stack_is_clamped(stack):
    assert select_z_range(stack, 3, 99, axis=1).shape[1] == 6


def test_empty_selection_is_refused_not_silently_zero(stack):
    """Projecting zero planes would yield an all-zero image that looks real."""
    with pytest.raises(ValueError):
        select_z_range(stack, 9, 12, axis=1)
    with pytest.raises(ValueError):
        select_z_range(stack, 5, 2, axis=1)


# ── ZProjection step ──────────────────────────────────────────────────

def test_projection_honours_the_range(stack):
    out = ZProjection(method="max", z_start=3, z_stop=5).process(stack, _ctx()).image
    assert out.shape == (2, 4, 4) and out[0, 0, 0] == 50      # plane 5, not 8


def test_single_plane_range(stack):
    out = ZProjection(method="max", z_start=4, z_stop=4).process(stack, _ctx()).image
    assert out[0, 0, 0] == 40


def test_no_range_projects_the_whole_stack(stack):
    """Pipelines saved before sub-stacks existed must be unaffected."""
    plain = ZProjection(method="max").process(stack, _ctx()).image
    assert plain[0, 0, 0] == 80
    assert np.array_equal(plain, ZProjection(method="max", z_start=0, z_stop=0)
                          .process(stack, _ctx()).image)


def test_range_on_a_2d_image_is_a_noop(stack):
    """Already-projected input must not raise just because a range is set."""
    assert ZProjection(z_start=2, z_stop=3).process(np.zeros((2, 4, 4)), _ctx()).image is None


def test_dtype_and_channel_count_survive(stack):
    out = ZProjection(method="mean", z_start=2, z_stop=6).process(stack, _ctx()).image
    assert out.dtype == stack.dtype and out.shape[0] == stack.shape[0]


# ── Pipeline JSON round-trip (this is what "headless" reads) ──────────

def test_range_survives_pipeline_json(tmp_path, stack):
    pl = Pipeline(name="t").add(ZProjection(method="max", z_start=3, z_stop=5))
    f = tmp_path / "p.json"
    pl.save(f)
    step = Pipeline.load(f).steps[0]
    assert (step.z_start, step.z_stop) == (3, 5)
    assert step.process(stack, _ctx()).image[0, 0, 0] == 50


def test_old_pipeline_json_without_range_still_loads(tmp_path, stack):
    import json
    f = tmp_path / "old.json"
    f.write_text(json.dumps({
        "name": "old",
        "steps": [{"type": "ZProjection", "channel": -1, "method": "max"}],
    }))
    step = Pipeline.load(f).steps[0]
    assert (step.z_start, step.z_stop) == (0, 0)
    assert step.process(stack, _ctx()).image[0, 0, 0] == 80


# ── BF pipeline projection worker ─────────────────────────────────────

def test_bf_project_one_applies_the_range(tmp_path):
    """The ProcessPool worker must project the same sub-stack the GUI's
    inline path does — the two are kept in sync by hand."""
    from correlative_imaging.batch import _bf_project_one
    import h5py

    src = _write_stack(tmp_path / "well.tif")

    in_dir = tmp_path / "h5"
    in_dir.mkdir()
    task = (str(src), "A1", 0, "max", str(in_dir), None, 3, 5)
    stem, well_id, px, dtype, err = _bf_project_one(task)
    assert err is None, err
    with h5py.File(in_dir / f"{stem}.h5") as f:
        data = f["data"][:]
    assert data.max() == 50          # plane 5, not the whole stack's 80


def test_bf_project_one_without_a_range_is_unchanged(tmp_path):
    from correlative_imaging.batch import _bf_project_one
    import h5py

    src = _write_stack(tmp_path / "well.tif")

    in_dir = tmp_path / "h5"
    in_dir.mkdir()
    stem, _, _, _, err = _bf_project_one((str(src), "A1", 0, "max", str(in_dir), None, 0, 0))
    assert err is None, err
    with h5py.File(in_dir / f"{stem}.h5") as f:
        assert f["data"][:].max() == 80


def test_bf_project_one_on_an_already_2d_image_ignores_the_range(tmp_path):
    """A BF file that is already a single plane must project fine with a range
    set — the range applies to stacks, and a 2-D input is not an error."""
    from correlative_imaging.batch import _bf_project_one
    import h5py

    src = tmp_path / "flat.tif"
    tifffile.imwrite(src, np.full((4, 4), 42, dtype="uint16"))

    in_dir = tmp_path / "h5"
    in_dir.mkdir()
    stem, _, _, _, err = _bf_project_one((str(src), "A1", 0, "max", str(in_dir), None, 3, 5))
    assert err is None, err
    with h5py.File(in_dir / f"{stem}.h5") as f:
        assert f["data"][:].max() == 42


def test_bf_project_one_reports_an_impossible_range(tmp_path):
    """Out-of-range on the worker path surfaces as a per-well error rather than
    a silently empty projection (the GUI pre-flights it once before fanning out)."""
    from correlative_imaging.batch import _bf_project_one

    src = _write_stack(tmp_path / "well.tif")
    in_dir = tmp_path / "h5"
    in_dir.mkdir()
    stem, _, _, _, err = _bf_project_one((str(src), "A1", 0, "max", str(in_dir), None, 20, 30))
    assert stem is None and err and "selects no planes" in err


# ── Viewer: unprojected (scroll-Z) mode ───────────────────────────────

def test_unprojected_mode_keeps_the_z_axis_and_scales_it(tmp_path, qapp):
    """'none' must hand napari a 3-D array per channel with a Z-first scale —
    the fallback layer path has its own projection call that would otherwise
    silently collapse the stack."""
    from correlative_imaging.io.reader import ImageData
    from correlative_imaging.viewer.gui import CorrelativeImagingWidget, _view_projection

    w = CorrelativeImagingWidget()

    class FakeViewer:
        def __init__(self):
            self.calls = []

        def add_image(self, data, **kw):
            self.calls.append((data.shape, kw.get("scale")))

    img = ImageData(data=np.zeros((2, 8, 16, 16), dtype="float32"),
                    channel_names=["a", "b"], pixel_size_um=0.65, z_step_um=2.0,
                    source_path=None, metadata={})

    w._viewer = FakeViewer()
    w._add_image_layers(img, "FL", "none")
    assert [c[0] for c in w._viewer.calls] == [(8, 16, 16), (8, 16, 16)]
    assert w._viewer.calls[0][1] == [2.0, 0.65, 0.65]

    w._viewer = FakeViewer()
    w._add_image_layers(img, "FL", "max")
    assert [c[0] for c in w._viewer.calls] == [(16, 16), (16, 16)]

    # The combo's spelled-out label must reach the layer code as a plain token.
    combo = w._plate_tab._view_proj_combo
    combo.setCurrentIndex(combo.count() - 1)
    assert combo.currentText().startswith("none")
    assert _view_projection(combo) == "none"


def test_stack_depth_label_never_rewrites_the_range(qapp):
    """A pipeline is reused across wells of differing depth, so loading a
    shallower sample must not silently shrink a range the user typed."""
    from correlative_imaging.viewer.gui import CorrelativeImagingWidget

    ch = CorrelativeImagingWidget()._channels_tab
    ch.set_stack_depth(21)
    assert ch._zdepth_lbl.text() == "of 21"

    ch._zproj_from.setValue(2)
    ch._zproj_to.setValue(19)
    ch.set_stack_depth(5)
    assert ch.get_zprojection_step()["z_stop"] == 19       # value kept, not clamped to 5
    assert "projects 2–5" in ch._zdepth_lbl.text()         # but says what will happen

    # A range starting past the end of this sample selects nothing — warn
    # rather than quietly rewriting what the user typed.
    ch._zproj_from.setValue(7)
    assert "no slices" in ch._zdepth_lbl.text()
    assert ch.get_zprojection_step()["z_start"] == 7
