"""Saturation: find clipped camera pixels on the raw image and keep them out of thresholds and measurements.

A clipped pixel is at the detector's maximum; its true value is unknown, so it must not set a threshold or enter a
mean, median, spread or background. It still belongs to its object (it is certainly signal), so masks keep it.

``SaturationMask`` runs first, on the raw stack, and records a per-channel boolean array in
``context.metadata["saturation_raw"]`` (same layout as the image: (C, Z, Y, X) or (C, Y, X)). Binning reduces it with
"any of the bin", ZProjection with "any plane of the projected range" - so it always follows the image's own grid and
z-range. Whenever it is down to one plane per channel, the step that got it there also returns
``saturated_ch{c}`` masks (grown by ``grow_px`` to cover the blur), which land in ``context.masks`` like any other mask
(and show up as layers in the single-image preview). Measuring steps read them with :func:`saturated`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation

from .base import PipelineContext, Step, StepResult, register_step

RAW_KEY = "saturation_raw"
GROW_KEY = "saturation_grow_px"


def saturation_level(image: np.ndarray, level: float = 0.0) -> float | None:
    """The value at which a pixel counts as clipped: ``level`` if given, else the integer dtype's maximum, or 65535
    for float data that holds 16-bit camera values (max <= 65535). None when it cannot be decided (float data
    beyond the 16-bit range) - then nothing is flagged."""
    if level > 0:
        return float(level)
    if np.issubdtype(image.dtype, np.integer):
        return float(np.iinfo(image.dtype).max)
    return 65535.0 if float(np.nanmax(image)) <= 65535.5 else None


def bin_any(arr: np.ndarray, factor: int) -> np.ndarray:
    """Bin the last two axes of a boolean array: a bin is set if any of its pixels is (crop as in bin_array)."""
    f = int(factor)
    if f <= 1:
        return arr
    h, w = arr.shape[-2:]
    hc, wc = (h // f) * f, (w // f) * f
    c = arr[..., :hc, :wc]
    return c.reshape(*c.shape[:-2], hc // f, f, wc // f, f).any(axis=(-3, -1))


def emit_masks(context: PipelineContext) -> dict[str, np.ndarray]:
    """``saturated_ch{c}`` masks once the raw saturation array is one plane per channel (else {})."""
    raw = context.metadata.get(RAW_KEY)
    if raw is None or raw.ndim != 3:
        return {}
    r = int(context.metadata.get(GROW_KEY, 0))
    return {f"saturated_ch{c}": (binary_dilation(m, iterations=r) if (r > 0 and m.any()) else m.copy())
            for c, m in enumerate(raw)}


def saturated(context: PipelineContext, channel: int, shape) -> np.ndarray | None:
    """This channel's saturation mask on the current grid, or None (no SaturationMask step, or a grid mismatch)."""
    m = context.masks.get(f"saturated_ch{channel}")
    if m is None or tuple(m.shape) != tuple(shape):
        return None
    return m.astype(bool)


def any_saturated(context: PipelineContext, shape) -> np.ndarray | None:
    """Union of every channel's saturation mask on the current grid, or None."""
    ms = [m.astype(bool) for k, m in context.masks.items()
          if k.startswith("saturated_ch") and tuple(m.shape) == tuple(shape)]
    if not ms:
        return None
    out = ms[0].copy()
    for m in ms[1:]:
        out |= m
    return out


@dataclass
@register_step
class SaturationMask(Step):
    """Flag clipped camera pixels on the raw image (must be the first step).

    Parameters
    ----------
    level:    Value at which a pixel counts as clipped; 0 = the detector maximum (integer dtype max, or 65535 for
              float data holding 16-bit values).
    grow_px:  Grow the final per-channel masks by this many pixels on the analysis grid, so the blur halo of a
              clipped spot is left out too (default 2).
    """
    level: float = 0.0
    grow_px: int = 2

    @property
    def name(self) -> str:
        return "saturation_mask"

    def process(self, image: np.ndarray, context: PipelineContext) -> StepResult:
        lvl = saturation_level(image, self.level)
        raw = image >= lvl if lvl is not None else np.zeros(image.shape, bool)
        context.metadata[RAW_KEY] = raw
        context.metadata[GROW_KEY] = int(self.grow_px)
        names = context.channel_names
        counts = {(names[c] if c < len(names) else f"ch{c}"): int(raw[c].sum()) for c in range(raw.shape[0])}
        return StepResult(masks=emit_masks(context),
                          info={"saturation_level": lvl, "saturated_raw_px": counts})
