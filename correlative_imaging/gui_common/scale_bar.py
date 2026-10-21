"""Scale bar in µm (ci gui and ci reconstruct), the napari >= 0.7 way: the unit comes from the layers (Layer.units),
not from ScaleBar.unit (deprecated, ignored from napari 0.8, removed in 0.9).

Layers added with a physical scale (any spatial scale other than 1: the pixel size in µm, as both tools add them) get
µm units. Layers without one (scale 1, e.g. an image whose pixel size is unknown) stay in pixels, so the bar never
claims µm for data that is not calibrated.
"""
from __future__ import annotations


def _calibrated(layer) -> bool:
    try:
        return any(abs(float(s) - 1.0) > 1e-9 for s in layer.scale[-2:])
    except Exception:
        return False


def set_um(layer):
    if _calibrated(layer):
        try:
            layer.units = ("um",) * layer.ndim
        except Exception:                                  # a layer type or napari version without units
            pass


def enable(viewer):
    """Show the scale bar; every calibrated layer (now and later) gets µm units."""
    viewer.scale_bar.visible = True
    for L in viewer.layers:
        set_um(L)
    viewer.layers.events.inserted.connect(lambda e: set_um(e.value))
