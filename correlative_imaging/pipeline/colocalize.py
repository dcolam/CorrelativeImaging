"""Colocalization analysis step."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation
from skimage.measure import regionprops_table

from .base import PipelineContext, Step, StepResult, register_step
from .segment import _project

_COLOC_PROPS = [
    "label", "area", "mean_intensity", "max_intensity", "centroid",
]


@dataclass
@register_step
class ColocalizationAnalysis(Step):
    """Measure secondary-channel particles that fall within primary-channel particle masks.

    Mirrors the original plugin's colocalization workflow:
    1. Use primary-channel particle label image as a mask.
    2. Optionally dilate each particle mask by ``dilation_um`` µm.
    3. Detect secondary-channel signal within those masks.
    4. Compute Manders' M1/M2 and Pearson correlation.
    5. Estimate random colocalization by rotating the secondary channel 90°.

    Results are stored as two DataFrames:
    - ``measurements``:         per-primary-particle colocalization stats
    - ``info['global_stats']``: image-level Manders + Pearson values

    Parameters
    ----------
    primary_channel:    Channel index with already-analysed particles.
    secondary_channel:  Channel index to probe for colocalization.
    dilation_um:        Expand each primary mask by this many µm (0 = no dilation).
    z_projection:       Z-projection for both channels.
    """
    primary_channel: int
    secondary_channel: int
    dilation_um: float = 0.0
    z_projection: str = "max"
    roi_mask: str = ""   # empty = whole image; matches the key used by ExtractROI/LoadROI

    @property
    def name(self) -> str:
        roi_suffix = f"_{self.roi_mask}" if self.roi_mask else ""
        return f"colocalization_ch{self.primary_channel}_ch{self.secondary_channel}{roi_suffix}"

    def process(self, image: np.ndarray, context: PipelineContext) -> StepResult:
        px = context.pixel_size_um
        primary_key = f"particles_ch{self.primary_channel}"
        primary_labels = context.masks.get(primary_key)
        if primary_labels is None:
            raise RuntimeError(
                f"No particle mask '{primary_key}' found. "
                "Run ParticleAnalysis for the primary channel first."
            )

        ch_p = _project(image[self.primary_channel], self.z_projection)
        ch_s = _project(image[self.secondary_channel], self.z_projection)

        # Restrict to ROI when requested.
        # Manders/Pearson denominators are computed within the ROI only,
        # giving meaningful colocalization values for a specific tissue region.
        roi = context.masks.get(self.roi_mask) if self.roi_mask else None
        if roi is not None:
            roi_bool = roi.astype(bool)
            ch_p = ch_p * roi_bool
            ch_s = ch_s * roi_bool
            primary_labels = primary_labels * roi_bool.astype(primary_labels.dtype)

        # --- Global Manders + Pearson (within ROI or whole image) ---
        m1, m2 = _manders(ch_p, ch_s, primary_labels > 0)
        pearson = _pearson(ch_p, ch_s)
        ch_s_rot = np.rot90(ch_s)
        m1_rand, m2_rand = _manders(ch_p, ch_s_rot, primary_labels > 0)

        global_stats = {
            "manders_m1": m1,
            "manders_m2": m2,
            "pearson_r": pearson,
            "manders_m1_random": m1_rand,
            "manders_m2_random": m2_rand,
        }

        # --- Per-particle colocalization ---
        dilation_px = int(round(self.dilation_um / px)) if self.dilation_um > 0 else 0
        rows: list[dict] = []
        for particle_label in np.unique(primary_labels):
            if particle_label == 0:
                continue
            particle_mask = primary_labels == particle_label
            if dilation_px > 0:
                particle_mask = binary_dilation(particle_mask, iterations=dilation_px)
            secondary_in_mask = ch_s * particle_mask
            n_secondary_px = int((secondary_in_mask > 0).sum())
            rows.append({
                "primary_label": int(particle_label),
                "primary_area_um2": float(particle_mask.sum()) * px ** 2,
                "n_secondary_pixels": n_secondary_px,
                "secondary_mean_intensity": float(secondary_in_mask[particle_mask].mean())
                    if particle_mask.any() else 0.0,
                "secondary_max_intensity": float(ch_s[particle_mask].max())
                    if particle_mask.any() else 0.0,
                "overlap_fraction": n_secondary_px / max(particle_mask.sum(), 1),
            })

        df = pd.DataFrame(rows)
        ch_names = context.channel_names
        p_name = ch_names[self.primary_channel] if self.primary_channel < len(ch_names) else f"ch{self.primary_channel}"
        s_name = ch_names[self.secondary_channel] if self.secondary_channel < len(ch_names) else f"ch{self.secondary_channel}"
        df.insert(0, "primary_channel",  p_name)
        df.insert(1, "secondary_channel", s_name)
        df.insert(2, "roi_mask", self.roi_mask or "whole_image")

        return StepResult(measurements=df, info={"global_stats": global_stats})


# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------

def _manders(
    ch1: np.ndarray,
    ch2: np.ndarray,
    overlap_mask: np.ndarray,
) -> tuple[float, float]:
    """Manders' overlap coefficients M1 and M2."""
    ch1f, ch2f = ch1.astype(float), ch2.astype(float)
    m1 = float(ch1f[overlap_mask].sum() / (ch1f.sum() + 1e-10))
    m2 = float(ch2f[overlap_mask].sum() / (ch2f.sum() + 1e-10))
    return m1, m2


def _pearson(ch1: np.ndarray, ch2: np.ndarray) -> float:
    """Pearson correlation coefficient between two channels."""
    a, b = ch1.ravel().astype(float), ch2.ravel().astype(float)
    a -= a.mean()
    b -= b.mean()
    denom = np.sqrt((a ** 2).sum() * (b ** 2).sum()) + 1e-10
    return float((a * b).sum() / denom)
