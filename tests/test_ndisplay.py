"""napari 3-D display mode must not break 2-D layer display."""

import numpy as np
import pytest

from correlative_imaging.viewer.napari_viewer import ensure_display_dims


class FakeDims:
    def __init__(self, ndisplay=2):
        self.ndisplay = ndisplay


class FakeViewer:
    """Stands in for napari.Viewer: records adds and refuses 2-D data in 3-D
    mode exactly as vispy's Volume visual does."""

    def __init__(self, ndisplay=2):
        self.dims = FakeDims(ndisplay)
        self.calls = []

    def add_image(self, data, **kw):
        if self.dims.ndisplay == 3 and np.ndim(data) < 3:
            raise ValueError("Volume visual needs a 3D array.")
        self.calls.append(np.shape(data))
        return object()

    add_labels = add_image

    class _Layers:
        def clear(self):
            pass

    layers = _Layers()


def test_switches_out_of_3d_for_2d_data():
    v = FakeViewer(ndisplay=3)
    assert ensure_display_dims(v, 2) is True
    assert v.dims.ndisplay == 2
    v.add_image(np.zeros((8, 8)))          # would have raised before


def test_leaves_2d_mode_alone():
    v = FakeViewer(ndisplay=2)
    assert ensure_display_dims(v, 2) is False
    assert v.dims.ndisplay == 2


def test_never_forces_3d_for_3d_data():
    """A stack is fine to view in 2-D slice mode; don't hijack the display."""
    v = FakeViewer(ndisplay=2)
    assert ensure_display_dims(v, 3) is False
    assert v.dims.ndisplay == 2

    v3 = FakeViewer(ndisplay=3)
    assert ensure_display_dims(v3, 3) is False
    assert v3.dims.ndisplay == 3


def test_tolerates_a_viewer_without_dims():
    assert ensure_display_dims(None, 2) is False
    assert ensure_display_dims(object(), 2) is False


def test_projected_layers_render_from_3d_mode(qapp):
    """The reported failure: viewer in 3-D, preview adds projected 2-D layers,
    vispy raises 'Volume visual needs a 3D array' from inside the insert."""
    from correlative_imaging.io.reader import ImageData
    from correlative_imaging.viewer.gui import CorrelativeImagingWidget

    w = CorrelativeImagingWidget()
    w._viewer = FakeViewer(ndisplay=3)
    img = ImageData(data=np.zeros((2, 4, 16, 16), dtype="float32"),
                    channel_names=["a", "b"], pixel_size_um=0.5, z_step_um=2.0,
                    source_path=None, metadata={})

    w._add_image_layers(img, "FL", "max")          # projected → 2-D
    assert w._viewer.dims.ndisplay == 2
    assert [s for s in w._viewer.calls] == [(16, 16), (16, 16)]


def test_unprojected_layers_may_stay_in_3d(qapp):
    from correlative_imaging.io.reader import ImageData
    from correlative_imaging.viewer.gui import CorrelativeImagingWidget

    w = CorrelativeImagingWidget()
    w._viewer = FakeViewer(ndisplay=3)
    img = ImageData(data=np.zeros((2, 4, 16, 16), dtype="float32"),
                    channel_names=["a", "b"], pixel_size_um=0.5, z_step_um=2.0,
                    source_path=None, metadata={})

    w._add_image_layers(img, "FL", "none")         # keeps Z → 3-D layers
    assert w._viewer.dims.ndisplay == 3
    assert w._viewer.calls == [(4, 16, 16), (4, 16, 16)]


# ── Recovering from a viewer left broken by a failed add ──────────────

class BrokenClearViewer(FakeViewer):
    """Reproduces the reported second failure: one layer is an orphan (in the
    model, no vispy counterpart), so napari's clear() raises KeyError and every
    later clear fails too — the viewer stays broken until napari restarts."""

    def __init__(self):
        super().__init__(ndisplay=2)
        self._layers = _LayerList(orphan="raw/FITC")
        self.layers = self._layers


class _LayerList(list):
    def __init__(self, orphan):
        super().__init__(["raw/DAPI", orphan, "raw/mCherry"])
        self._orphan = orphan

    def clear(self):
        raise KeyError(self._orphan)      # as napari._vispy does

    def remove(self, layer):
        if layer == self._orphan:
            raise KeyError(layer)
        super().remove(layer)


def test_clear_recovers_from_an_orphaned_layer():
    from correlative_imaging.viewer.napari_viewer import safe_clear_layers

    v = BrokenClearViewer()
    safe_clear_layers(v)                   # must not raise
    assert list(v.layers) == ["raw/FITC"]  # only the unremovable orphan remains


def test_clear_is_a_plain_clear_when_nothing_is_broken():
    v = FakeViewer()
    calls = []
    v.layers = type("L", (), {"clear": lambda self: calls.append("clear")})()
    from correlative_imaging.viewer.napari_viewer import safe_clear_layers

    safe_clear_layers(v)
    assert calls == ["clear"]


def test_clear_tolerates_no_viewer():
    from correlative_imaging.viewer.napari_viewer import safe_clear_layers
    safe_clear_layers(None)
