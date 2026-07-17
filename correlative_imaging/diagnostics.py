"""Diagnostic-image compositing helpers — shared by the GUI (Run tab config)
and the headless batch runner (actual per-well image generation).

Deliberately has no PyQt/qtpy/napari import at module scope so ``batch.py``
(and the ``ci batch`` CLI) can use it without pulling in GUI dependencies.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def auto_contrast_limits(data: np.ndarray, low: float = 1.0, high: float = 99.5) -> tuple[float, float]:
    """Robust percentile-based display range for an image layer.

    napari's own default (min/max of the data at add-time) is easily skewed
    dim by a handful of outlier hot/saturated pixels — common in microscopy
    images with a stray calibration mark or thumbnail artifact. A percentile
    stretch is far more representative of the actual signal. Display-only;
    does not modify the underlying data.

    Canonical home for this function is here (not ``viewer/napari_viewer.py``)
    so it stays importable without pulling in napari/qtpy — ``batch.py``
    needs it for headless diagnostic-image compositing.
    """
    finite = data[np.isfinite(data)] if np.issubdtype(data.dtype, np.floating) else data.ravel()
    if finite.size == 0:
        return (0.0, 1.0)
    lo, hi = (float(v) for v in np.percentile(finite, [low, high]))
    if hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
        if hi <= lo:
            hi = lo + 1.0
    return (lo, hi)


# User-assignable per-channel colors (Channels tab "Channel identity" box) —
# used for both the preview display and diagnostic image export. Deliberately
# a small, named set (not a full color picker) so the same name can double
# as a napari colormap name and an RGB tint for exported composites.
CHANNEL_COLOR_CHOICES = ["gray", "red", "green", "blue", "cyan", "magenta", "yellow"]
CHANNEL_COLOR_RGB = {
    "gray": (1.0, 1.0, 1.0), "red": (1.0, 0.0, 0.0), "green": (0.0, 1.0, 0.0),
    "blue": (0.0, 0.0, 1.0), "cyan": (0.0, 1.0, 1.0), "magenta": (1.0, 0.0, 1.0),
    "yellow": (1.0, 1.0, 0.0),
}


def composite_rgb(channel_planes: list, colors: list[str]) -> np.ndarray:
    """Combine 2-D (Y,X) arrays, one per channel, into one RGB uint8 image.
    Each channel is auto-contrast-stretched (same percentile stretch used
    for viewer display), tinted by its assigned color, and additively
    combined (clipped to [0,255]) — a display-only composite, matching what
    the preview/viewer already does per-channel, just flattened to one RGB
    image for saving to disk.
    """
    h, w = channel_planes[0].shape
    out = np.zeros((h, w, 3), dtype=np.float32)
    for data, color in zip(channel_planes, colors):
        lo, hi = auto_contrast_limits(data)
        norm = np.clip((data.astype(np.float32) - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
        r, g, b = CHANNEL_COLOR_RGB.get(color, (1.0, 1.0, 1.0))
        out[..., 0] += norm * r
        out[..., 1] += norm * g
        out[..., 2] += norm * b
    return (np.clip(out, 0.0, 1.0) * 255).astype(np.uint8)


def load_channel_stack(path) -> tuple[np.ndarray, list[str]]:
    """Read a multi-channel diagnostic TIF written by the batch runner
    (``_save_multichannel_tif``): axes CYX float, one slice per channel, with the
    channel names in the ImageJ ``Labels`` metadata. Returns ``(stack, labels)``
    where ``stack`` is ``(C, Y, X)`` and ``labels`` has one name per channel
    (falls back to ``ch0, ch1, …`` if the labels are missing/mismatched)."""
    import tifffile

    arr = tifffile.imread(str(path))
    if arr.ndim == 2:
        arr = arr[None]
    labels = None
    try:
        with tifffile.TiffFile(str(path)) as tf:
            ij = tf.imagej_metadata or {}
        labels = ij.get("Labels")
    except Exception:
        labels = None
    if not labels or len(labels) != arr.shape[0]:
        labels = [f"ch{i}" for i in range(arr.shape[0])]
    return np.asarray(arr), list(labels)


def render_channels(
    planes: list[np.ndarray],
    colors: list[tuple[float, float, float]],
    gains: list[float] | None = None,
    limits: list[tuple[float, float]] | None = None,
) -> np.ndarray | None:
    """Composite selected channel planes into an RGB uint8 image for display.

    Like :func:`composite_rgb` but per-channel: each plane is auto-contrast
    stretched, multiplied by a display ``gain`` (brightness; 1.0 = the plain
    auto-contrast), tinted by an ``(r,g,b)`` colour in 0–1, and additively
    combined. Display-only — never touches the values the classifier reads.

    ``limits`` optionally supplies the per-channel ``(lo, hi)`` contrast bounds;
    pass them precomputed so dragging a gain slider (which only rescales the
    already-normalised image) doesn't recompute a percentile over the full plane
    each tick. ``None`` → computed here. ``None`` return if no planes given."""
    if not planes:
        return None
    gains = gains if gains is not None else [1.0] * len(planes)
    limits = limits if limits is not None else [auto_contrast_limits(p) for p in planes]
    h, w = planes[0].shape
    out = np.zeros((h, w, 3), dtype=np.float32)
    for data, (r, g, b), gain, (lo, hi) in zip(planes, colors, gains, limits):
        norm = np.clip((data.astype(np.float32) - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
        norm = np.clip(norm * float(gain), 0.0, 1.0)
        out[..., 0] += norm * r
        out[..., 1] += norm * g
        out[..., 2] += norm * b
    return (np.clip(out, 0.0, 1.0) * 255).astype(np.uint8)


def hex_to_rgb01(hexc: str) -> tuple[float, float, float]:
    """'#rrggbb' → (r,g,b) floats in 0–1. Falls back to white on a bad string."""
    try:
        h = hexc.lstrip("#")
        return (int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255)
    except Exception:
        return (1.0, 1.0, 1.0)


def save_diagnostic_image(rgb: np.ndarray, path_no_ext: Path, formats: set[str]) -> None:
    """Save an RGB uint8 composite in whichever of {"tiff", "jpg"} formats
    are requested."""
    if "tiff" in formats:
        import tifffile
        tifffile.imwrite(str(path_no_ext.with_suffix(".tif")), rgb)
    if "jpg" in formats:
        import imageio.v3 as iio
        iio.imwrite(str(path_no_ext.with_suffix(".jpg")), rgb)


def stamp_outlines(
    rgb: np.ndarray, masks: dict[str, np.ndarray],
    color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Draw a 1px inner-boundary outline of each mask onto an RGB composite —
    lets a saved diagnostic image (including a lossy JPG) show exactly where
    an ROI selection landed without needing to reopen it alongside a
    separate mask file. Returns a new array; ``rgb`` itself is untouched.
    """
    from skimage.segmentation import find_boundaries

    out = rgb.copy()
    for mask in masks.values():
        if mask is None or not np.any(mask):
            continue
        boundary = find_boundaries(mask.astype(bool), mode="inner")
        out[boundary] = color
    return out


def bbox_from_mask(mask: np.ndarray, pad_px: int = 0):
    """Return (r0, r1, c0, c1) bounding box of a mask's True pixels, padded
    by pad_px and clipped to the mask's own shape. None if mask is empty."""
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    h, w = mask.shape
    r0 = max(0, int(ys.min()) - pad_px)
    r1 = min(h, int(ys.max()) + 1 + pad_px)
    c0 = max(0, int(xs.min()) - pad_px)
    c1 = min(w, int(xs.max()) + 1 + pad_px)
    return r0, r1, c0, c1
