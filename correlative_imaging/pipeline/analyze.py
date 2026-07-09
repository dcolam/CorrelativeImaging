"""Particle analysis step — wraps skimage.measure.regionprops."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from skimage.measure import label as sk_label, regionprops_table

from .base import PipelineContext, Step, StepResult, register_step
from .segment import _project

_REGION_PROPS = [
    "label",
    "area",
    "perimeter",
    "eccentricity",
    "solidity",
    "mean_intensity",
    "max_intensity",
    "min_intensity",
    "centroid",
    "bbox",
]


@dataclass
@register_step
class ParticleAnalysis(Step):
    """Detect and measure particles within an optional mask region.

    Reads the label image from ``context.masks['mask_ch{channel}']`` if it
    exists; otherwise thresholds the channel on-the-fly with the requested
    method.  Measurements are returned as a :class:`pandas.DataFrame` and the
    particle label image is stored in the context as ``'particles_ch{channel}'``.

    Parameters
    ----------
    channel:        Channel index to analyse.
    min_size_um2:   Minimum particle area in µm² (0 = no filter).
    max_size_um2:   Maximum particle area in µm² (0 = no filter).
    min_circularity:Minimum circularity [0–1].
    z_projection:   How to collapse Z before analysis.
    roi_mask:       Name of an ROI mask in context to restrict analysis area.
                    Typically ``'roi'`` from an ExtractROI step.
    """
    channel: int
    min_size_um2: float = 0.5
    max_size_um2: float = 5000.0
    min_circularity: float = 0.0
    z_projection: str = "max"
    roi_mask: str = ""   # empty = no restriction

    @property
    def name(self) -> str:
        roi_suffix = f"_{self.roi_mask}" if self.roi_mask else ""
        return f"particle_analysis_ch{self.channel}{roi_suffix}"

    def process(self, image: np.ndarray, context: PipelineContext) -> StepResult:
        ch = image[self.channel]
        intensity = _project(ch, self.z_projection)
        px = context.pixel_size_um

        # Resolve label image
        mask_key = f"mask_ch{self.channel}"
        label_image: np.ndarray = context.masks.get(mask_key)  # type: ignore[assignment]
        if label_image is None:
            # No prior segmentation — fall back to a simple Otsu threshold
            from skimage.filters import threshold_otsu
            binary = intensity > threshold_otsu(intensity)
            label_image = sk_label(binary)
        elif label_image.max() == 1:
            # Binary mask from AutoThreshold (no watershed yet)
            label_image = sk_label(label_image.astype(bool))

        # Restrict to ROI if requested
        roi = context.masks.get(self.roi_mask) if self.roi_mask else None
        if roi is not None:
            label_image = label_image * roi.astype(label_image.dtype)

        if label_image.max() == 0:
            return StepResult(
                measurements=pd.DataFrame(),
                masks={f"particles_ch{self.channel}": label_image},
                info={"n_particles": 0},
            )

        props = regionprops_table(
            label_image, intensity_image=intensity, properties=_REGION_PROPS
        )
        df = pd.DataFrame(props)

        # Derived metrics
        df["circularity"] = (
            4 * np.pi * df["area"] / (df["perimeter"].replace(0, np.nan) ** 2)
        ).fillna(0.0)
        df["area_um2"] = df["area"] * px ** 2

        # Rename centroid columns for clarity
        df = df.rename(
            columns={"centroid-0": "centroid_row", "centroid-1": "centroid_col"},
        )

        # Filtering
        if self.min_size_um2 > 0:
            df = df[df["area_um2"] >= self.min_size_um2]
        if self.max_size_um2 > 0:
            df = df[df["area_um2"] <= self.max_size_um2]
        if self.min_circularity > 0:
            df = df[df["circularity"] >= self.min_circularity]

        ch_name = context.channel_names[self.channel] if self.channel < len(context.channel_names) else f"ch{self.channel}"
        df.insert(0, "channel", ch_name)
        df.insert(1, "roi_mask", self.roi_mask or "whole_image")
        df.insert(2, "roi_path", context.mask_paths.get(self.roi_mask, "") if self.roi_mask else "")
        df = df.reset_index(drop=True)

        # Keep only particles that survived filtering in the label image
        surviving_labels = set(df["label"])
        filtered_labels = np.where(
            np.isin(label_image, list(surviving_labels)), label_image, 0
        )

        return StepResult(
            measurements=df,
            masks={f"particles_ch{self.channel}": filtered_labels},
            info={"n_particles": len(df)},
        )


@dataclass
@register_step
class IntensityMeasurement(Step):
    """Measure bulk fluorescence intensity of a channel within an ROI.

    Unlike :class:`ParticleAnalysis`, this measures *all* pixels in the region
    regardless of segmentation — useful for tracking overall expression level
    or comparing signal strength between conditions and samples.

    Intensity is measured on the current (preprocessed) image, so background
    subtraction is already applied.

    Parameters
    ----------
    channel:       Channel index to measure.
    z_projection:  How to collapse Z ('max', 'mean', 'sum').
    roi_mask:      Key of an ROI mask in context.masks.  Empty = whole image.
    """
    channel: int
    z_projection: str = "max"
    roi_mask: str = ""

    @property
    def name(self) -> str:
        return f"intensity_ch{self.channel}"

    def process(self, image: np.ndarray, context: PipelineContext) -> StepResult:
        ch_arr = image[self.channel]
        plane  = _project(ch_arr, self.z_projection)

        if self.roi_mask and self.roi_mask in context.masks:
            mask    = context.masks[self.roi_mask].astype(bool)
            pixels  = plane[mask]
            area_px = int(mask.sum())
        else:
            pixels  = plane.ravel()
            area_px = int(plane.size)

        px       = context.pixel_size_um
        area_um2 = area_px * px * px

        ch_name = (context.channel_names[self.channel]
                   if self.channel < len(context.channel_names)
                   else f"ch{self.channel}")

        vals = {
            "channel":        ch_name,
            "roi_mask":       self.roi_mask or "whole_image",
            "roi_path":       context.mask_paths.get(self.roi_mask, "") if self.roi_mask else "",
            "mean_intensity": float(pixels.mean()) if pixels.size > 0 else 0.0,
            "sum_intensity":  float(pixels.sum())  if pixels.size > 0 else 0.0,
            "std_intensity":  float(pixels.std())  if pixels.size > 0 else 0.0,
            "area_px":        area_px,
            "area_um2":       area_um2,
        }
        return StepResult(
            measurements=pd.DataFrame([vals]),
            info={"mean_intensity": vals["mean_intensity"]},
        )
