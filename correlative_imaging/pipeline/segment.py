"""Segmentation steps: thresholding, watershed, ROI extraction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt, label as ndi_label
from skimage.filters import (
    threshold_isodata,
    threshold_li,
    threshold_otsu,
    threshold_triangle,
    threshold_yen,
)
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed

from .base import PipelineContext, Step, StepResult, register_step

_THRESHOLD_METHODS: dict[str, object] = {
    "otsu": threshold_otsu,
    "li": threshold_li,
    "yen": threshold_yen,
    "triangle": threshold_triangle,
    "isodata": threshold_isodata,
}


def _project(arr: np.ndarray, method: str = "max") -> np.ndarray:
    """Collapse a (Z, Y, X) array to (Y, X) for 2-D analysis."""
    if arr.ndim == 2:
        return arr
    match method:
        case "max":
            return arr.max(axis=0)
        case "mean":
            return arr.mean(axis=0)
        case "sum":
            return arr.sum(axis=0)
        case _:
            raise ValueError(f"Unknown projection method: {method!r}")


def mask_key(channel: int, roi_mask: str = "") -> str:
    """Context key of a channel's threshold mask: ``mask_ch{c}`` for the
    global (whole-frame) threshold, ``mask_ch{c}@{roi}`` for one computed
    from a selection's pixels."""
    return f"mask_ch{channel}@{roi_mask}" if roi_mask else f"mask_ch{channel}"


def channel_mask(context: PipelineContext, channel: int, roi_mask: str = "") -> np.ndarray | None:
    """The threshold mask to use for *channel* within selection *roi_mask*:
    the selection's own (per-selection thresholding) if there is one, else
    the global one."""
    if roi_mask:
        m = context.masks.get(mask_key(channel, roi_mask))
        if m is not None:
            return m
    return context.masks.get(mask_key(channel))


# ------------------------------------------------------------------
# Auto-threshold  →  binary mask
# ------------------------------------------------------------------

@dataclass
@register_step
class AutoThreshold(Step):
    """Threshold a channel using an automated method.

    Produces a binary mask stored in the context as ``'mask_ch{channel}'``.

    With ``roi_mask`` set (per-selection thresholding), the threshold is
    computed from that selection's pixels only, then applied to the whole
    frame — so objects crossing the selection edge stay whole and
    ParticleAnalysis cuts them at the edge exactly as with the global mask.
    Stored as ``'mask_ch{channel}@{roi_mask}'``; ParticleAnalysis,
    ColocalizationAnalysis and IntensityMeasurement for that selection use it
    in place of the global mask. An empty selection gives an empty mask and
    a NaN threshold.

    Parameters
    ----------
    channel:        Channel index to threshold.
    method:         One of 'otsu', 'li', 'yen', 'triangle', 'isodata'.
    z_projection:   How to flatten Z before thresholding ('max', 'mean', 'sum').
    min_size:       Remove objects smaller than this many pixels after thresholding.
    roi_mask:       Selection whose pixels set the threshold. Empty = whole frame.
    """
    channel: int
    method: str = "otsu"
    z_projection: str = "max"
    min_size: int = 50
    roi_mask: str = ""

    @property
    def name(self) -> str:
        roi_suffix = f"_{self.roi_mask}" if self.roi_mask else ""
        return f"auto_threshold_ch{self.channel}{roi_suffix}"

    def process(self, image: np.ndarray, context: PipelineContext) -> StepResult:
        if self.method not in _THRESHOLD_METHODS:
            raise ValueError(
                f"Unknown threshold method {self.method!r}. "
                f"Choose from: {sorted(_THRESHOLD_METHODS)}"
            )
        ch = image[self.channel]
        plane = _project(ch, self.z_projection)
        thresh_fn = _THRESHOLD_METHODS[self.method]
        key = mask_key(self.channel, self.roi_mask)
        ch_name = (context.channel_names[self.channel]
                   if self.channel < len(context.channel_names)
                   else f"ch{self.channel}")
        info = {"mask_key": key, "channel": ch_name, "roi_mask": self.roi_mask or "whole_image"}

        if self.roi_mask:
            roi = context.masks.get(self.roi_mask)
            if roi is None:
                raise RuntimeError(
                    f"No selection mask '{self.roi_mask}' in context — the per-selection "
                    "AutoThreshold must run after that selection's ROI step."
                )
            pixels = plane[roi.astype(bool)]
            if pixels.size == 0 or pixels.min() == pixels.max():
                # Nothing to threshold: no objects in this selection.
                return StepResult(masks={key: np.zeros(plane.shape, bool)},
                                  info={**info, "threshold_value": float("nan")})
        else:
            pixels = plane
        threshold_value = float(thresh_fn(pixels))  # type: ignore[operator]
        binary = plane > threshold_value
        if self.min_size > 0:
            binary = remove_small_objects(binary, max_size=self.min_size)

        return StepResult(masks={key: binary}, info={**info, "threshold_value": threshold_value})


# ------------------------------------------------------------------
# Watershed split  →  label image
# ------------------------------------------------------------------

@dataclass
@register_step
class WatershedSplit(Step):
    """Separate touching objects in a binary mask using the watershed algorithm.

    Reads the mask produced by a prior :class:`AutoThreshold` step for the same
    channel and replaces it with a label image (each object has a unique integer).

    Parameters
    ----------
    channel:        Must match the channel used in the preceding AutoThreshold step.
    min_distance:   Minimum distance in pixels between object centres.
    roi_mask:       Split the per-selection mask of this selection instead of
                    the global one (see AutoThreshold.roi_mask).
    """
    channel: int
    min_distance: int = 5
    roi_mask: str = ""

    @property
    def name(self) -> str:
        roi_suffix = f"_{self.roi_mask}" if self.roi_mask else ""
        return f"watershed_split_ch{self.channel}{roi_suffix}"

    def process(self, image: np.ndarray, context: PipelineContext) -> StepResult:
        key = mask_key(self.channel, self.roi_mask)
        binary = context.masks.get(key)
        if binary is None:
            raise RuntimeError(
                f"No mask '{key}' found in context. "
                "Run AutoThreshold for this channel first."
            )

        labels = _watershed_labels(binary.astype(bool), self.min_distance)
        return StepResult(masks={key: labels})


def _watershed_labels(binary: np.ndarray, min_distance: int) -> np.ndarray:
    from skimage.feature import peak_local_max

    distance = distance_transform_edt(binary)
    coords = peak_local_max(
        distance, min_distance=min_distance, labels=binary
    )
    local_max = np.zeros_like(distance, dtype=bool)
    if coords.size:
        local_max[tuple(coords.T)] = True
    markers, _ = ndi_label(local_max)
    return watershed(-distance, markers, mask=binary)


# ------------------------------------------------------------------
# ROI extraction — detect tissue boundary / region of interest
# ------------------------------------------------------------------

@dataclass
@register_step
class ExtractROI(Step):
    """Detect a region-of-interest boundary by heavily blurring a channel.

    Useful for isolating the tissue section / slice boundary before running
    particle analysis only within that region.

    Produces mask ``'roi'`` in the context.

    Parameters
    ----------
    channel:        Channel used as ROI reference (typically DAPI/nuclear).
    blur_sigma:     Heavy Gaussian blur sigma in pixels to smooth the outline.
    method:         Threshold method applied after blurring.
    """
    channel: int
    blur_sigma: float = 20.0
    method: str = "otsu"
    roi_name: str = "roi"   # key stored in context.masks; use unique names for multiple ROIs

    @property
    def name(self) -> str:
        return f"extract_roi_{self.roi_name}_ch{self.channel}"

    def process(self, image: np.ndarray, context: PipelineContext) -> StepResult:
        from scipy.ndimage import gaussian_filter, binary_fill_holes

        ch = image[self.channel]
        plane = _project(ch, "max")
        blurred = gaussian_filter(plane.astype(np.float32), sigma=self.blur_sigma)
        thresh_fn = _THRESHOLD_METHODS.get(self.method, threshold_otsu)
        roi = blurred > float(thresh_fn(blurred))  # type: ignore[operator]
        roi = binary_fill_holes(roi).astype(bool)
        return StepResult(masks={self.roi_name: roi})


# ------------------------------------------------------------------
# Load ROI from file — ImageJ .roi or binary mask image
# ------------------------------------------------------------------

@dataclass
@register_step
class LoadROI(Step):
    """Load an ROI mask from a file and inject it into the pipeline context.

    Supported formats
    -----------------
    - ImageJ ``.roi`` file (single polygon/rectangle ROI)
    - Binary image (``.tif``, ``.tiff``, ``.png``) — any non-zero pixel = ROI

    For batch runs the **same file is applied to every image**.  If per-image
    ROI files are needed, place a matching ``.roi`` file alongside each image
    and use ``roi_name`` matching the image stem (future feature).

    Parameters
    ----------
    path:       Absolute path to the ROI file.
    roi_name:   Key stored in ``context.masks`` (default ``'roi'``).
    """
    path: str
    roi_name: str = "roi"

    @property
    def name(self) -> str:
        return f"load_roi_{self.roi_name}"

    def process(self, image: np.ndarray, context: PipelineContext) -> StepResult:
        from pathlib import Path as _Path

        src = _Path(self.path)
        if not src.exists():
            raise FileNotFoundError(f"ROI file not found: {src}")

        h, w = image.shape[-2], image.shape[-1]
        suffix = src.suffix.lower()
        scale = self._pixel_scale(src, context.pixel_size_um, context)

        if suffix == ".roi":
            mask = self._from_imagej(src, h, w, scale=scale)
        elif suffix in {".tif", ".tiff", ".png", ".bmp"}:
            mask = self._from_image(src, h, w)
        else:
            raise ValueError(
                f"Unsupported ROI file format '{suffix}'. "
                "Use .roi (ImageJ) or .tif / .png (binary mask)."
            )

        return StepResult(masks={self.roi_name: mask}, mask_paths={self.roi_name: str(src)})

    # ── Private helpers ──────────────────────────────────────────────

    @staticmethod
    def _pixel_scale(src, target_pixel_size_um: float, context=None) -> float:
        """Ratio to convert pixel coordinates stored in *src* into the
        target image's pixel grid.

        Two sources of truth, in order:

        1. A ``<src>.json`` sidecar (written by the BF Pipeline tab, see
           ``gui._save_roi``) recording the physical pixel size of the image
           the ROI was drawn on. If that differs from the image the ROI is now
           being applied to — a brightfield ROI on a differently-scaled
           fluorescence image, or any image the pipeline has binned — this
           gives the exact ratio.
        2. Failing that, the cumulative ``Binning`` factor recorded on the
           context. An ROI file with no provenance was necessarily drawn on the
           unbinned image, so its coordinates must shrink by the same factor;
           without this it would keep full-resolution coordinates on a smaller
           canvas and be clipped into the corner.

        Returns ``1.0`` when neither applies — an unbinned run with an ROI file
        that has no sidecar, which is the historical behaviour.
        """
        import json
        from pathlib import Path as _Path

        sidecar = _Path(str(src) + ".json")
        source_px = None
        if sidecar.exists() and target_pixel_size_um:
            try:
                source_px = json.loads(sidecar.read_text()).get("pixel_size_um")
            except Exception:
                source_px = None
        if source_px:
            return source_px / target_pixel_size_um

        binning = float(getattr(context, "metadata", {}).get("binning_factor", 1.0) or 1.0)
        return 1.0 / binning if binning > 1 else 1.0

    @staticmethod
    def _from_imagej(src, h: int, w: int, scale: float = 1.0) -> np.ndarray:
        try:
            import roifile
        except ImportError as exc:
            raise ImportError(
                "roifile is required for ImageJ .roi files: "
                "pip install roifile"
            ) from exc

        from skimage.draw import polygon as draw_polygon

        rois = [roifile.ImagejRoi.fromfile(str(src))]

        mask = np.zeros((h, w), dtype=bool)
        for roi in rois:
            try:
                coords = roi.coordinates()
                if coords is not None and len(coords) >= 3:
                    if scale != 1.0:
                        coords = coords * scale
                    rows = np.clip(coords[:, 1].astype(int), 0, h - 1)
                    cols = np.clip(coords[:, 0].astype(int), 0, w - 1)
                    rr, cc = draw_polygon(rows, cols, shape=(h, w))
                    mask[rr, cc] = True
                    continue
            except Exception:
                pass  # polygon coords unavailable — fall through to bounding-box
            # Fallback: use bounding box attributes
            try:
                r1 = max(0, int(roi.top * scale))
                c1 = max(0, int(roi.left * scale))
                r2 = min(h, int(roi.bottom * scale))
                c2 = min(w, int(roi.right * scale))
                mask[r1:r2, c1:c2] = True
            except Exception:
                pass  # ROI has neither polygon coords nor a valid bounding box — skip

        return mask

    @staticmethod
    def _from_image(src, h: int, w: int) -> np.ndarray:
        import tifffile
        from skimage.transform import resize

        img = tifffile.imread(str(src))
        # Collapse to 2-D
        while img.ndim > 2:
            img = img[0]
        mask = img > 0
        if mask.shape != (h, w):
            mask = resize(mask.astype(np.float32), (h, w), order=0) > 0.5
        return mask.astype(bool)
