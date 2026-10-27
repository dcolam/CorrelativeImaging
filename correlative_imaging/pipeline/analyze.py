"""Particle analysis step — wraps skimage.measure.regionprops."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation, center_of_mass, distance_transform_edt
from scipy.ndimage import labeled_comprehension, median as ndi_median, standard_deviation as ndi_std
from skimage.measure import label as sk_label, regionprops_table
from skimage.segmentation import expand_labels

from .base import PipelineContext, Step, StepResult, register_step
from .segment import _project, channel_mask

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

# User-selectable particle-level metrics (Channels tab). "circularity" and
# "area_um2" are derived, not raw regionprops columns; "centroid" and "bbox"
# each expand to more than one output column. Always computed internally
# (min/max-size and min-circularity filtering depend on area_um2/circularity
# regardless of what's selected for output) — a None ``metrics`` list means
# "all of them", keeping old saved pipeline JSONs behaving identically.
PARTICLE_METRIC_CHOICES = [
    "area", "area_um2", "perimeter", "circularity", "eccentricity", "solidity",
    "mean_intensity", "median_intensity", "max_intensity", "min_intensity",
    "std_intensity", "p25_intensity", "p75_intensity", "local_background",
    "centroid", "bbox",
]
_PARTICLE_METRIC_COLUMNS = {
    "centroid": ["centroid_row", "centroid_col"],
    "bbox": ["bbox-0", "bbox-1", "bbox-2", "bbox-3"],
    "local_background": ["ring_median_intensity", "ring_sd_intensity", "ring_px"],
}

# User-selectable gross/bulk metrics (Channels tab). "area_px"/"area_um2" are
# the selection's own area — independently toggleable from mean/sum/std so a
# selection's area can be recorded without also measuring bulk intensity.
INTENSITY_METRIC_CHOICES = [
    "mean_intensity", "median_intensity", "sum_intensity", "std_intensity", "area_px", "area_um2",
]

# Where each particle sits in the frame: centroid in pixels of the grid the
# pipeline ran on (binned if it bins) and in µm from the frame's top-left
# corner (independent of binning). Always output — positions are needed to
# place particles on the image or a diagnostic crop.
PARTICLE_POSITION_COLUMNS = ["centroid_row", "centroid_col", "centroid_row_um", "centroid_col_um"]

# Robust intensity and spread inside the particle, and its local background:
# the median is robust to a few very bright pixels (a speck pulls the mean up,
# not the median); SD and the 25th/75th percentiles describe how even the
# particle is; the ring is a band around the whole object (a gap, then
# ``ring_width_um`` wide) without any channel's segmented objects, inside the
# selection when there is one, so glow of a bright neighbour or uneven
# background is measured right next to the particle. Selectable like the
# others; a saved metric list that predates them (contains none of them) gets
# all of them, so re-running an old pipeline JSON does not silently drop them.
PARTICLE_ROBUST_METRICS = ["median_intensity", "std_intensity", "p25_intensity", "p75_intensity",
                           "local_background"]


def resolve_particle_metrics(metrics: list[str] | None) -> list[str]:
    """The particle metrics to save: None = all; an old list without any robust metric gets them added."""
    if metrics is None:
        return list(PARTICLE_METRIC_CHOICES)
    if not any(m in metrics for m in PARTICLE_ROBUST_METRICS):
        return list(metrics) + PARTICLE_ROBUST_METRICS
    return list(metrics)


def foreground_mask(context: PipelineContext, roi_mask: str, shape) -> np.ndarray:
    """Every channel's threshold mask (this selection's own where it has per-selection thresholding)."""
    fg = np.zeros(shape, bool)
    for c in sorted({int(k[len("mask_ch"):]) for k in context.masks
                     if k.startswith("mask_ch") and k[len("mask_ch"):].isdigit()}):
        m = channel_mask(context, c, roi_mask)
        if m is not None:
            fg |= m.astype(bool)
    return fg


def ring_stats(plane: np.ndarray, labels: np.ndarray, index, fg: np.ndarray, px: float,
               gap_um: float = 1.0, width_um: float = 3.0, within: np.ndarray | None = None) -> pd.DataFrame:
    """Local background of each labelled object: median, robust SD (1.4826 MAD) and pixel count of a ring
    ``gap_um`` .. ``gap_um + width_um`` around it, without ``fg`` (other objects of any channel) and, with
    ``within``, only inside that selection. Neighbouring rings do not overlap (each pixel goes to the
    nearest object). NaN for an object with fewer than 10 ring pixels."""
    index = np.asarray(index, dtype=np.int64)
    out = pd.DataFrame({"ring_median_intensity": np.nan, "ring_sd_intensity": np.nan, "ring_px": 0}, index=index)
    if not len(index):
        return out
    lab = labels.astype(np.int64)
    gap = max(int(round(gap_um / px)), 0)
    outer = gap + max(int(round(width_um / px)), 1)
    near = expand_labels(lab, gap) if gap > 0 else lab
    ring = np.where((near == 0) & ~fg, expand_labels(lab, outer), 0)
    if within is not None:
        ring = np.where(within.astype(bool), ring, 0)
    n = np.bincount(ring.ravel(), minlength=int(lab.max()) + 1)[index] if ring.any() else np.zeros(len(index), int)
    out["ring_px"] = n
    ok = index[n >= 10]
    if len(ok):
        med = np.asarray(ndi_median(plane, ring, ok), float)
        mad = labeled_comprehension(plane, ring, ok, lambda v: np.median(np.abs(v - np.median(v))), float, np.nan)
        out.loc[ok, "ring_median_intensity"] = med
        out.loc[ok, "ring_sd_intensity"] = 1.4826 * np.asarray(mad, float)
    return out

# Where each particle sits relative to its selection, computed from the WHOLE
# whole-image object it was cut from (not the piece inside the selection).
# Always output — not user-selectable — so a particle cut at the selection
# edge can always be told apart from one lying fully inside it.
PARTICLE_EDGE_COLUMNS = [
    "area_full_um2", "cut_at_edge", "share_inside",
    "dist_centre_um", "dist_centre_rel", "dist_rim_um",
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
    metrics:        Which per-particle metrics to keep in the output, from
                    ``PARTICLE_METRIC_CHOICES``. ``None`` = all of them
                    (backward-compatible default). ``channel``/``roi_mask``/
                    ``roi_path``/``label`` are always kept regardless.
    """
    channel: int
    min_size_um2: float = 0.5
    max_size_um2: float = 5000.0
    min_circularity: float = 0.0
    z_projection: str = "max"
    roi_mask: str = ""   # empty = no restriction
    metrics: list[str] | None = None   # None = all of PARTICLE_METRIC_CHOICES
    ring_gap_um: float = 1.0           # local background: gap to the object ...
    ring_width_um: float = 3.0         # ... and width of the ring

    @property
    def name(self) -> str:
        roi_suffix = f"_{self.roi_mask}" if self.roi_mask else ""
        return f"particle_analysis_ch{self.channel}{roi_suffix}"

    def process(self, image: np.ndarray, context: PipelineContext) -> StepResult:
        ch = image[self.channel]
        intensity = _project(ch, self.z_projection)
        px = context.pixel_size_um

        # Resolve label image — the selection's own threshold mask if it has
        # one (per-selection thresholding), else the global one.
        label_image: np.ndarray = channel_mask(context, self.channel, self.roi_mask)  # type: ignore[assignment]
        if label_image is None:
            # No prior segmentation — fall back to a simple Otsu threshold
            from skimage.filters import threshold_otsu
            binary = intensity > threshold_otsu(intensity)
            label_image = sk_label(binary)
        elif label_image.max() == 1:
            # Binary mask from AutoThreshold (no watershed yet)
            label_image = sk_label(label_image.astype(bool))

        # Restrict to ROI if requested. Objects crossing the selection edge
        # are cut there — keep the whole-image labels to describe the object
        # each piece was cut from (see _edge_columns).
        full_labels = label_image
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
        idx = df["label"].to_numpy()
        df["median_intensity"] = ndi_median(intensity, label_image, idx)
        with np.errstate(invalid="ignore", divide="ignore"):    # scipy divides over every label number up to the
            df["std_intensity"] = ndi_std(intensity, label_image, idx)   # max, also those cut away (0/0, not used)
        df["p25_intensity"] = labeled_comprehension(intensity, label_image, idx, lambda v: np.percentile(v, 25), float, np.nan)
        df["p75_intensity"] = labeled_comprehension(intensity, label_image, idx, lambda v: np.percentile(v, 75), float, np.nan)
        rs = ring_stats(intensity, full_labels, idx, foreground_mask(context, self.roi_mask, intensity.shape), px,
                        self.ring_gap_um, self.ring_width_um, within=roi)
        for col in rs.columns:
            df[col] = rs[col].to_numpy()

        # Derived metrics
        df["circularity"] = (
            4 * np.pi * df["area"] / (df["perimeter"].replace(0, np.nan) ** 2)
        ).fillna(0.0)
        df["area_um2"] = df["area"] * px ** 2

        # Rename centroid columns for clarity
        df = df.rename(
            columns={"centroid-0": "centroid_row", "centroid-1": "centroid_col"},
        )
        df["centroid_row_um"] = df["centroid_row"] * px
        df["centroid_col_um"] = df["centroid_col"] * px

        # Filtering
        if self.min_size_um2 > 0:
            df = df[df["area_um2"] >= self.min_size_um2]
        if self.max_size_um2 > 0:
            df = df[df["area_um2"] <= self.max_size_um2]
        if self.min_circularity > 0:
            df = df[df["circularity"] >= self.min_circularity]

        df = _edge_columns(df, full_labels, roi, px)

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

        # Trim to user-selected metrics — "channel"/"roi_mask"/"roi_path"/
        # "label" are bookkeeping/identity columns and always kept. Filtering
        # above already used the full internal set (area_um2/circularity)
        # regardless of what's selected for output.
        keep = ["channel", "roi_mask", "roi_path", "label"]
        for m in resolve_particle_metrics(self.metrics):
            keep.extend(_PARTICLE_METRIC_COLUMNS.get(m, [m]))
        keep.extend(c for c in PARTICLE_POSITION_COLUMNS if c not in keep)
        keep.extend(PARTICLE_EDGE_COLUMNS)
        df = df[[c for c in keep if c in df.columns]]

        return StepResult(
            measurements=df,
            masks={f"particles_ch{self.channel}": filtered_labels},
            info={"n_particles": len(df)},
        )


def _edge_columns(
    df: pd.DataFrame, full_labels: np.ndarray, roi: np.ndarray | None, px: float,
) -> pd.DataFrame:
    """Add PARTICLE_EDGE_COLUMNS to *df* (one row per particle piece inside
    the selection, ``label`` matching *full_labels*).

    - ``area_full_um2``:   area of the whole-image object the piece was cut from
    - ``cut_at_edge``:     part of that object lies outside the selection
    - ``share_inside``:    piece area / whole object area
    - ``dist_centre_um``:  selection centre of mass → whole object's centroid
    - ``dist_centre_rel``: the same / the selection's equal-area radius
                           (0 = centre, 1 = rim, > 1 = centred outside)
    - ``dist_rim_um``:     whole object's centroid → selection outline,
                           positive inside, negative outside

    Without a selection the pieces are the whole objects: share_inside = 1,
    cut_at_edge = False, distances NaN.
    """
    df = df.copy()
    labels = df["label"].to_numpy().astype(np.int64)
    full_area = np.bincount(full_labels.ravel().astype(np.int64))
    full_px = full_area[labels] if len(labels) else np.zeros(0, dtype=np.int64)
    df["area_full_um2"] = full_px * px ** 2
    df["share_inside"] = df["area"].to_numpy() / np.maximum(full_px, 1)
    # Compare pixel counts, not the float ratio.
    df["cut_at_edge"] = df["area"].to_numpy() < full_px
    for c in ("dist_centre_um", "dist_centre_rel", "dist_rim_um"):
        df[c] = np.nan
    if roi is None or df.empty:
        return df

    rb = roi.astype(bool)
    # Centroids of the WHOLE objects that have a piece in the selection —
    # the piece's own centroid is pulled inwards.
    fc = pd.DataFrame(regionprops_table(
        np.where(np.isin(full_labels, labels), full_labels, 0).astype(np.int64),
        properties=["label", "centroid"],
    )).set_index("label").loc[labels]
    fr, fcl = fc["centroid-0"].to_numpy(), fc["centroid-1"].to_numpy()
    cr, cc = center_of_mass(rb)
    r_eq = np.sqrt(rb.sum() / np.pi)
    d = np.hypot(fr - cr, fcl - cc)
    df["dist_centre_um"] = d * px
    df["dist_centre_rel"] = d / r_eq
    rim = distance_transform_edt(rb) - distance_transform_edt(~rb)   # + inside, - outside, px
    ri = np.clip(np.round(fr).astype(int), 0, rb.shape[0] - 1)
    ci = np.clip(np.round(fcl).astype(int), 0, rb.shape[1] - 1)
    df["dist_rim_um"] = rim[ri, ci] * px
    return df


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
    metrics:       Which of ``INTENSITY_METRIC_CHOICES`` to keep in the
                    output. ``None`` = all of them (backward-compatible
                    default). ``area_px``/``area_um2`` are independent of
                    mean/sum/std — a selection's area can be recorded without
                    also measuring bulk intensity.
    outside_particles: Also report the background: the selection's pixels
                    that belong to no segmented object of any channel
                    (every ``mask_ch*`` AutoThreshold mask, whole frame).
                    Written as a second row with
                    ``roi_mask = '<selection>:outside_particles'``. Must run
                    after all channels' AutoThreshold steps.
    outside_margin_um: Dilate the particle masks by this margin (blur halo)
                    before taking the outside pixels.
    """
    channel: int
    z_projection: str = "max"
    roi_mask: str = ""
    metrics: list[str] | None = None   # None = all of INTENSITY_METRIC_CHOICES
    outside_particles: bool = False
    outside_margin_um: float = 1.0

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
            mask    = None
            pixels  = plane.ravel()
            area_px = int(plane.size)

        px       = context.pixel_size_um
        area_um2 = area_px * px * px

        ch_name = (context.channel_names[self.channel]
                   if self.channel < len(context.channel_names)
                   else f"ch{self.channel}")

        all_vals = {
            "mean_intensity": float(pixels.mean()) if pixels.size > 0 else 0.0,
            "median_intensity": float(np.median(pixels)) if pixels.size > 0 else 0.0,
            "sum_intensity":  float(pixels.sum())  if pixels.size > 0 else 0.0,
            "std_intensity":  float(pixels.std())  if pixels.size > 0 else 0.0,
            "area_px":        area_px,
            "area_um2":       area_um2,
        }
        selected = self.metrics if self.metrics is not None else INTENSITY_METRIC_CHOICES
        roi_name = self.roi_mask or "whole_image"
        roi_path = context.mask_paths.get(self.roi_mask, "") if self.roi_mask else ""
        rows = [{
            "channel":        ch_name,
            "roi_mask":       roi_name,
            "roi_path":       roi_path,
            **{m: all_vals[m] for m in selected if m in all_vals},
        }]

        if self.outside_particles:
            # Every channel's threshold mask — this selection's own where it
            # has per-selection thresholding, else the global one.
            fg = foreground_mask(context, self.roi_mask, plane.shape)
            r = int(round(self.outside_margin_um / px)) if self.outside_margin_um > 0 else 0
            if r > 0:   # iterations=0 would dilate until nothing changes
                fg = binary_dilation(fg, iterations=r)
            sel = mask if mask is not None else np.ones(plane.shape, bool)
            bg = plane[sel & ~fg]
            # A selection fully covered by particles stays NaN — not filled in.
            bg_vals = {
                "mean_intensity": float(bg.mean()) if bg.size else float("nan"),
                # robust choice for background: a few unsegmented dim cells skew the mean
                "median_intensity": float(np.median(bg)) if bg.size else float("nan"),
                "sum_intensity":  float(bg.sum()),
                "std_intensity":  float(bg.std()) if bg.size else float("nan"),
                "area_px":        int(bg.size),
                "area_um2":       bg.size * px * px,
            }
            rows.append({
                "channel":        ch_name,
                "roi_mask":       f"{roi_name}:outside_particles",
                "roi_path":       roi_path,
                **{m: bg_vals[m] for m in selected if m in bg_vals},
            })

        return StepResult(
            measurements=pd.DataFrame(rows),
            info={"mean_intensity": all_vals["mean_intensity"]},
        )
