"""Preprocessing steps: background subtraction, blur, normalization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

from .base import PipelineContext, Step, StepResult, register_step


def _apply_per_slice(fn, arr: np.ndarray, **kwargs) -> np.ndarray:
    """Apply fn slice-by-slice on (Z, Y, X) or (Y, X) arrays."""
    if arr.ndim == 2:
        return fn(arr, **kwargs)
    return np.stack([fn(z, **kwargs) for z in arr])


# ------------------------------------------------------------------
# Background subtraction — rolling-ball (scikit-image) or tophat
# ------------------------------------------------------------------

@dataclass
@register_step
class BackgroundSubtraction(Step):
    """Subtract uneven background using the rolling-ball algorithm.

    Parameters
    ----------
    channel:    Channel index to process.
    radius:     Rolling-ball radius in pixels.
    method:     'rolling_ball' (default) or 'tophat' (morphological top-hat).
    """
    channel: int
    radius: float = 50.0
    method: str = "rolling_ball"

    @property
    def name(self) -> str:
        return f"background_subtraction_ch{self.channel}"

    def process(self, image: np.ndarray, context: PipelineContext) -> StepResult:
        ch = image[self.channel].copy()

        if self.method == "rolling_ball":
            try:
                from skimage.restoration import rolling_ball as _rb
                corrected = _apply_per_slice(
                    lambda s: np.clip(s - _rb(s, radius=self.radius), 0, None), ch
                )
            except ImportError:
                raise ImportError("scikit-image >=0.19 required for rolling_ball")

        elif self.method == "tophat":
            from skimage.morphology import white_tophat, disk
            selem = disk(int(self.radius))
            corrected = _apply_per_slice(lambda s: white_tophat(s, selem), ch)

        else:
            raise ValueError(f"Unknown background method: {self.method!r}")

        result_image = image.copy()
        result_image[self.channel] = corrected.astype(image.dtype)
        return StepResult(image=result_image)


# ------------------------------------------------------------------
# Gaussian blur
# ------------------------------------------------------------------

@dataclass
@register_step
class GaussianBlur(Step):
    """Convolve a channel with a Gaussian kernel.

    ``sigma`` is in pixels.  For Z-stacks, blurring is applied per slice (2-D).
    """
    channel: int
    sigma: float = 2.0

    @property
    def name(self) -> str:
        return f"gaussian_blur_ch{self.channel}"

    def process(self, image: np.ndarray, context: PipelineContext) -> StepResult:
        ch = image[self.channel]
        blurred = _apply_per_slice(gaussian_filter, ch, sigma=self.sigma)
        result_image = image.copy()
        result_image[self.channel] = blurred.astype(image.dtype)
        return StepResult(image=result_image)


# ------------------------------------------------------------------
# Intensity normalization
# ------------------------------------------------------------------

@dataclass
@register_step
class Normalize(Step):
    """Rescale channel intensity to [0, 1].

    Parameters
    ----------
    channel:      Channel index.
    method:       'minmax' (default) or 'percentile'.
    low_pct:      Lower percentile clip (only for method='percentile').
    high_pct:     Upper percentile clip (only for method='percentile').
    """
    channel: int
    method: str = "minmax"
    low_pct: float = 1.0
    high_pct: float = 99.0

    @property
    def name(self) -> str:
        return f"normalize_ch{self.channel}"

    def process(self, image: np.ndarray, context: PipelineContext) -> StepResult:
        ch = image[self.channel].astype(np.float32)

        if self.method == "minmax":
            lo, hi = ch.min(), ch.max()
        elif self.method == "percentile":
            lo = float(np.percentile(ch, self.low_pct))
            hi = float(np.percentile(ch, self.high_pct))
        else:
            raise ValueError(f"Unknown normalization method: {self.method!r}")

        span = hi - lo
        normalized = np.clip((ch - lo) / (span if span > 0 else 1), 0.0, 1.0)

        result_image = image.copy()
        result_image[self.channel] = normalized
        return StepResult(image=result_image, info={"lo": float(lo), "hi": float(hi)})


# ------------------------------------------------------------------
# Manual brightness / contrast (linear stretch)
# ------------------------------------------------------------------

@dataclass
@register_step
class BrightnessContrast(Step):
    """Clip and linearly rescale a channel to [min_val, max_val].

    Mirrors ImageJ's "Set Min/Max" brightness-contrast control.
    """
    channel: int
    min_val: float = 0.0
    max_val: float = 65535.0

    @property
    def name(self) -> str:
        return f"brightness_contrast_ch{self.channel}"

    def process(self, image: np.ndarray, context: PipelineContext) -> StepResult:
        ch = image[self.channel].astype(np.float32)
        span = self.max_val - self.min_val
        adjusted = np.clip((ch - self.min_val) / (span if span > 0 else 1), 0.0, 1.0)
        result_image = image.copy()
        result_image[self.channel] = adjusted
        return StepResult(image=result_image)
