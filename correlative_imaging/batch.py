"""Headless batch runner — process a directory of images with a saved pipeline."""

from __future__ import annotations

import json
import logging
import threading
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

# Two Dask LocalClusters must not spin up at the SAME instant in one process —
# they race on shared asyncio/IOLoop state and one dies ("Cluster closed
# without starting up"). When several plates run concurrently, each builds its
# own cluster; this lock serializes just the (fast) startup, after which the
# clusters run fully in parallel. Module-level so all plate threads share it.
_DASK_CLUSTER_STARTUP_LOCK = threading.Lock()

from correlative_imaging.diagnostics import bbox_from_mask, composite_rgb, save_diagnostic_image, stamp_outlines
from correlative_imaging.io import read_image, supported_extensions
from correlative_imaging.io.plate import WellInfo
from correlative_imaging.pipeline.base import Pipeline, PipelineContext, Step
from correlative_imaging.pipeline.analyze import IntensityMeasurement, ParticleAnalysis
from correlative_imaging.pipeline.colocalize import ColocalizationAnalysis
from correlative_imaging.pipeline.preprocess import BlackLevelNormalization
from correlative_imaging.pipeline.segment import AutoThreshold
from correlative_imaging.storage import ResultsDB

log = logging.getLogger(__name__)

# Signature: (current_index, total, filename, n_particles_or_None)
ProgressFn = Callable[[int, int, str, int | None], None]

# Signature: (well) -> (pipeline dict, missing_selection_labels). The pipeline
# dict is JSON-serializable with any ROI steps already resolved to this
# well's own file(s); missing_selection_labels lists the label of every
# per-well-dynamic ROI selection that had no matching file for this well
# (so it was left out of the pipeline dict, not silently analyzed unrestricted).
PipelineDictFn = Callable[[WellInfo], tuple[dict, list[str]]]


class BatchRunner:
    """Run a :class:`Pipeline` over every image in a directory.

    Parameters
    ----------
    pipeline:       Loaded :class:`Pipeline` instance.
    db_path:        Path to the output SQLite database.
    experiment:     Experiment label stored in the database.
    extra_extensions: Additional file extensions to process beyond the defaults.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        db_path: str | Path,
        experiment: str = "",
        extra_extensions: list[str] | None = None,
    ):
        self.pipeline = pipeline
        self.db_path = Path(db_path)
        self.experiment = experiment
        self._extensions = supported_extensions | set(extra_extensions or [])

    def _collect_files(
        self,
        input_dir: Path,
        pattern: str = "**/*",
        recursive: bool = True,
    ) -> list[Path]:
        glob = input_dir.rglob(pattern) if recursive else input_dir.glob(pattern)
        return sorted(
            p for p in glob
            if p.is_file() and p.suffix.lower() in self._extensions
        )

    def run_directory(
        self,
        input_dir: str | Path,
        output_dir: str | Path | None = None,
        pattern: str = "**/*",
        recursive: bool = True,
        export_parquet: bool = False,
        progress_fn: ProgressFn | None = None,
    ) -> None:
        """Process all matching images under ``input_dir``.

        Parameters
        ----------
        input_dir:      Root folder to search for images.
        output_dir:     Where to write the database (defaults to input_dir).
        pattern:        Glob pattern for file discovery.
        recursive:      Whether to recurse into subdirectories.
        export_parquet: Also export result tables as .parquet files.
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir) if output_dir else input_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        files = self._collect_files(input_dir, pattern, recursive)
        if not files:
            log.warning("No matching images found in %s", input_dir)
            return

        log.info("Found %d images to process", len(files))
        pipeline_json = json.dumps(
            {"name": self.pipeline.name, "steps": [s.to_dict() for s in self.pipeline.steps]},
            indent=2,
        )

        with ResultsDB(output_dir / "results.db") as db:
            iterator = tqdm(files, unit="img", desc="Batch") if progress_fn is None else files
            for idx, image_path in enumerate(iterator):
                image_id = None
                try:
                    image_data = read_image(image_path)
                    image_id = db.register_image(
                        image_path,
                        experiment=self.experiment,
                        n_channels=image_data.n_channels,
                        pixel_size_um=image_data.pixel_size_um,
                        metadata=image_data.metadata,
                    )
                    context = PipelineContext(
                        channel_names=image_data.channel_names,
                        pixel_size_um=image_data.pixel_size_um,
                        z_step_um=image_data.z_step_um,
                    )
                    _, results = self.pipeline.run(image_data.data, context)

                    db.set_analysis_scale(image_id, context.pixel_size_um,
                                          context.metadata.get("binning_factor", 1.0))
                    db.save_selections(image_id, _selection_rows(
                        _selection_names(self.pipeline.steps), context))
                    _save_channel_levels(db, image_id, zip(self.pipeline.steps, results))
                    for step, result in zip(self.pipeline.steps, results):
                        if result.measurements is not None and not result.measurements.empty:
                            if isinstance(step, ParticleAnalysis):
                                db.save_particle_measurements(image_id, result.measurements)
                            elif isinstance(step, IntensityMeasurement):
                                db.save_intensity_measurements(image_id, result.measurements)
                            elif isinstance(step, ColocalizationAnalysis):
                                db.save_colocalization(
                                    image_id,
                                    result.measurements,
                                    result.info.get("global_stats", {}),
                                )

                    db.log_run(image_id, self.pipeline.name, pipeline_json)

                    n_particles = sum(
                        r.info.get("n_particles", 0)
                        for step, r in zip(self.pipeline.steps, results)
                        if isinstance(step, ParticleAnalysis)
                    )
                    if progress_fn:
                        progress_fn(idx + 1, len(files), image_path.name, n_particles)

                except Exception:
                    err = traceback.format_exc()
                    log.error("Failed on %s:\n%s", image_path.name, err)
                    db.log_run(
                        image_id,
                        self.pipeline.name,
                        pipeline_json,
                        status="error",
                        error=err,
                    )
                    if progress_fn:
                        progress_fn(idx + 1, len(files), image_path.name, None)

            if export_parquet:
                db.export_parquet(output_dir)

        log.info("Batch complete. Results → %s", output_dir / "results.db")


@dataclass
class _WellComputeResult:
    """Output of running one well's pipeline — pure compute, no DB access.

    ``steps_and_results`` is ``None`` on failure (see ``error``).
    """
    well: WellInfo
    pl_name: str
    pipeline_json: str
    steps_and_results: list | None = None
    image_meta: dict | None = None   # n_channels, pixel_size_um, metadata
    error: str | None = None
    missing_selections: list[str] = field(default_factory=list)  # unmatched per-well selection labels
    selections: list[dict] = field(default_factory=list)   # rows for the `selections` table


def _project_planes(arr) -> "np.ndarray":
    """Collapse a (C, Z, Y, X) pipeline image to (C, Y, X) via max-projection;
    no-op for an already-2-D-per-channel (C, Y, X) array."""
    return arr.max(axis=1) if arr.ndim == 4 else arr


def _load_bf_projection(diag_dir, well_id: str, target_shape=None):
    """Load this well's brightfield projection (written by the BF pipeline to
    ``<output>/bf_pipeline/projections/<well>_proj.tif``) for inclusion as an
    extra channel. Returns a 2-D array, or None if not found.

    *target_shape* is the (Y, X) shape of the pipeline's own planes. The BF
    pipeline deliberately does **not** bin — a trained Ilastik ``.ilp`` expects
    the original scale — so when the analysis pipeline bins, this projection is
    larger than the planes it is being stacked with. It is binned down here by
    the same integer factor to match; without that, ``np.stack`` in
    :func:`_save_multichannel_tif` raises on mismatched shapes and the ROI-crop
    bounding boxes (computed on the binned grid) would index the wrong region.

    Returns None, with a warning, when the two grids are not related by a whole
    factor — a resampled-but-misaligned overlay is worse than no overlay.
    """
    import numpy as np
    import tifffile

    proj = Path(diag_dir).parent / "bf_pipeline" / "projections" / f"{well_id}_proj.tif"
    if not proj.exists():
        return None
    try:
        arr = tifffile.imread(str(proj))
        if arr.ndim == 3:                 # collapse any stray channel/z axis
            arr = arr.max(axis=0)
        arr = np.asarray(arr, dtype=np.float32)
    except Exception:
        return None
    return _match_grid(arr, target_shape, well_id, "BF projection")


def _load_bf_focus(diag_dir, well_id: str, n_around: int, target_shape=None) -> dict:
    """The sharpest BF plane and up to *n_around* planes either side of it,
    from ``<well>_focus.tif`` (see :func:`_write_bf_focus`), as
    ``{"BF focus-1": plane, "BF focus": plane, "BF focus+1": plane, …}``.
    Neighbours past the end of the stack are left out. An older single-plane
    file gives just ``"BF focus"``. Empty dict if there is no file."""
    import re

    import numpy as np
    import tifffile

    path = Path(diag_dir).parent / "bf_pipeline" / "projections" / f"{well_id}_focus.tif"
    if not path.exists():
        return {}
    try:
        with tifffile.TiffFile(str(path)) as tf:
            arr = tf.asarray()
            info = (tf.imagej_metadata or {}).get("Info", "")
    except Exception:
        return {}
    if arr.ndim == 2:
        arr, k = arr[None], 0
    else:
        m = re.search(r"focus_index_in_file\s*=\s*(\d+)", info)
        k = int(m.group(1)) if m else arr.shape[0] // 2
    out = {}
    for off in range(-n_around, n_around + 1):
        if 0 <= k + off < arr.shape[0]:
            plane = _match_grid(np.asarray(arr[k + off], dtype=np.float32),
                                target_shape, well_id, "BF focal plane")
            if plane is not None:
                out["BF focus" + (f"{off:+d}" if off else "")] = plane
    return out


def _match_grid(arr, target_shape, well_id: str, what: str):
    """Bin a BF plane onto the pipeline's (possibly binned) grid; None, with a
    warning, if the two are not a whole binning factor apart."""
    from correlative_imaging.pipeline.preprocess import bin_array

    if target_shape is None or tuple(arr.shape) == tuple(target_shape):
        return arr
    th, tw = target_shape
    fy, fx = arr.shape[0] // th if th else 0, arr.shape[1] // tw if tw else 0
    if fy >= 1 and fy == fx and arr.shape[0] // fy == th and arr.shape[1] // fx == tw:
        return bin_array(arr, fy, "mean")
    log.warning(
        "Well %s: %s is %s but the pipeline planes are %s — not a whole "
        "binning factor apart, so it is omitted from diagnostics.",
        well_id, what, tuple(arr.shape), tuple(target_shape),
    )
    return None


_IJ_LUT_RGB = {
    "gray": (255, 255, 255), "grey": (255, 255, 255), "white": (255, 255, 255),
    "red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255),
    "cyan": (0, 255, 255), "magenta": (255, 0, 255), "yellow": (255, 255, 0),
    "orange": (255, 128, 0),
}


def _imagej_luts(colors, n: int):
    """Per-channel ImageJ LUTs (3×256 uint8 ramps), or None if no usable colors.

    Without these the TIF opens in grayscale and the channel colours chosen in
    the Channels tab are lost — the composite JPG had them, the real data did not.
    """
    import numpy as np

    if not colors:
        return None
    ramp = np.arange(256, dtype="uint8")
    luts = []
    for i in range(n):
        name = str(colors[i]).lower() if i < len(colors) else "gray"
        r, g, b = _IJ_LUT_RGB.get(name, (255, 255, 255))
        luts.append(np.stack([(ramp * (c / 255)).astype("uint8") for c in (r, g, b)]))
    return luts


def _save_multichannel_tif(planes, channel_names, bf_planes, path, colors=None,
                           pixel_size_um: float | None = None, info: str | None = None) -> None:
    """Write real per-channel data as an ImageJ-readable multi-channel TIF
    (axes CYX, one slice per channel, channel labels preserved). Fluorescence
    planes are saved as float32 as-is (lossless, comparable across wells); the
    optional BF plane is min-max scaled to [0,1] and appended as a labelled
    channel so it shares the stack's float range without distorting the FL
    values."""
    import numpy as np
    import tifffile

    chans = [np.asarray(p, dtype=np.float32) for p in planes]
    labels = list(channel_names[:len(chans)]) or [f"ch{i}" for i in range(len(chans))]
    while len(labels) < len(chans):
        labels.append(f"ch{len(labels)}")

    # bf_planes: {label: plane}, e.g. {"BF": projection, "BF focus": plane};
    # a bare array (older callers) is labelled "BF".
    if bf_planes is not None and not isinstance(bf_planes, dict):
        bf_planes = {"BF": bf_planes}
    n_fl = len(chans)
    for label, plane in (bf_planes or {}).items():
        bf = np.asarray(plane, dtype=np.float32)
        mn, mx = float(bf.min()), float(bf.max())
        bf = (bf - mn) / (mx - mn) if mx > mn else np.zeros_like(bf)
        chans.append(bf)
        labels.append(label)

    stack = np.stack(chans, axis=0)   # (C, Y, X)
    meta = {"axes": "CYX", "Labels": labels, "mode": "composite"}
    # BF channels, when appended, are always last and grayscale.
    lut_colors = list(colors or [])
    if len(chans) > n_fl:
        fl_colors = (lut_colors + ["gray"] * n_fl)[:n_fl]
        lut_colors = fl_colors + ["gray"] * (len(chans) - n_fl)
    luts = _imagej_luts(lut_colors, len(chans))
    if luts is not None:
        meta["LUTs"] = luts
    kw = {}
    if pixel_size_um:
        meta["unit"] = "um"
        kw["resolution"] = (1.0 / pixel_size_um, 1.0 / pixel_size_um)
    if info:
        meta["Info"] = info
    tifffile.imwrite(str(path), stack, imagej=True, metadata=meta, **kw)


def _crop_info(well_id: str, roi_name: str, bbox, pixel_size_um: float,
               bin_factor: int) -> str:
    """ImageJ 'Info' text (Image > Show Info in Fiji) placing a crop in its
    frame: origin in pixels of the analysis grid and in µm."""
    r0, r1, c0, c1 = bbox
    px = float(pixel_size_um or 1.0)
    return "\n".join([
        f"well = {well_id}", f"selection = {roi_name}",
        f"crop_row0 = {r0}", f"crop_row1 = {r1}", f"crop_col0 = {c0}", f"crop_col1 = {c1}",
        f"origin_row_um = {r0 * px:.4f}", f"origin_col_um = {c0 * px:.4f}",
        f"pixel_size_um = {px:.6g}", f"binning_factor = {bin_factor}",
        "Pixel positions are on the analysis grid (binned if the pipeline bins).",
        "Position in this crop = whole-frame position - origin.",
    ])


def _save_zstack_crop(raw_stack, bbox, bin_factor: int, bin_method: str,
                      channel_names, colors, pixel_size_um: float,
                      z_step_um: float | None, path, info: str | None = None) -> bool:
    """Write the *un-projected* (C, Z, Y, X) input cropped to *bbox* as an
    ImageJ hyperstack (axes ZCYX, raw dtype, calibrated XY/Z, channel LUTs).

    *bbox* is on the pipeline's grid, which is binned when the pipeline bins;
    the raw stack is not, so the matching raw region is cut first and then
    binned by the same factor/method — only the crop is binned, not the frame.
    Raw (pre-preprocessing) values, so it opens like the source file in Fiji.
    Returns False (nothing written) if the image has no Z axis.
    """
    import numpy as np
    import tifffile

    from correlative_imaging.pipeline.preprocess import bin_array

    arr = np.asarray(raw_stack)
    if arr.ndim != 4:
        return False
    f = max(int(bin_factor or 1), 1)
    r0, r1, c0, c1 = bbox
    crop = arr[:, :, r0 * f:r1 * f, c0 * f:c1 * f]
    if f > 1:
        crop = bin_array(crop, f, bin_method).astype(arr.dtype, copy=False)
    zcyx = np.ascontiguousarray(crop.transpose(1, 0, 2, 3))
    n_c = zcyx.shape[1]
    meta = {"axes": "ZCYX", "mode": "composite", "unit": "um",
            "spacing": float(z_step_um or 1.0)}
    luts = _imagej_luts(colors, n_c)
    if luts is not None:
        meta["LUTs"] = luts
    names = list(channel_names[:n_c]) + [f"ch{i}" for i in range(len(channel_names), n_c)]
    meta["Labels"] = [names[c] for _ in range(zcyx.shape[0]) for c in range(n_c)]
    if info:
        meta["Info"] = info
    px = float(pixel_size_um or 1.0)
    tifffile.imwrite(str(path), zcyx, imagej=True, metadata=meta,
                     resolution=(1.0 / px, 1.0 / px))
    return True


def _save_well_diagnostics(
    well, final_image, pixel_size_um: float, pl_dict: dict,
    context: PipelineContext, diag_cfg: dict, raw_stack=None,
    z_step_um: float | None = None,
) -> dict:
    """Write whole-image and/or per-ROI-crop composite diagnostic images for
    one well, per ``diag_cfg`` (see ``RunTab._on_run`` for the dict shape).

    ``raw_stack`` is the unprocessed (C, Z, Y, X) input; with
    ``diag_cfg["zstack_crops"]`` each crop is also written un-projected from it
    as ``<well>_<roi>_crop_zstack.tif``.

    Returns ``{roi_name: {"bbox": (r0, r1, c0, c1), "files": [...]}}`` for
    every crop written, the box on the analysis grid (r1/c1 exclusive).

    ``final_image`` is the array ``Pipeline.run()`` returns (not the raw
    ``image_data.data``) — since ``AutoThreshold``/``WatershedSplit`` never
    set ``result.image`` (only ``result.masks``) and every preprocessing step
    only overwrites its own channel's slice, this is exactly each channel's
    *last pre-threshold* image, matching what the single-image preview shows
    by default (#23) — a channel with no preprocessing steps configured just
    keeps its raw data unchanged, which is still the right fallback.

    Only the actual ROI *selections* configured in this well's pipeline are
    cropped — ``context.masks`` also holds intermediate segmentation/particle
    label masks (``mask_ch{c}``, ``particles_ch{c}``) that aren't selections
    and must not be emitted as crops.
    """
    out_dir = Path(diag_cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    formats = diag_cfg["formats"]
    colors = diag_cfg.get("colors") or []
    stamp = diag_cfg.get("stamp_rois", False)
    planes = list(_project_planes(final_image))

    # Real multi-channel TIFs (opt-in): the actual per-channel pixel data (not
    # a baked-in RGB composite), so the TIFs are comparable across wells and
    # each channel is separable in the viewer. The stamped RGB composite is
    # kept for the human-facing JPG only. Optionally append the BF projection
    # as an extra labelled channel.
    multichannel = diag_cfg.get("multichannel_tif", False)
    channel_names = list(getattr(context, "channel_names", []) or [])
    # Pass the pipeline's own plane shape so a binned run still lines up.
    target = planes[0].shape[-2:] if planes else None
    bf_planes = {}
    if diag_cfg.get("include_bf"):
        plane = _load_bf_projection(out_dir, well.well_id, target_shape=target)
        if plane is not None:
            bf_planes["BF"] = plane
    if diag_cfg.get("include_bf_focus"):
        focus = _load_bf_focus(out_dir, well.well_id,
                               int(diag_cfg.get("bf_focus_around", 1) or 0), target)
        if not focus:
            log.info("Well %s: no BF focal-plane image (%s_focus.tif) — rerun the "
                     "BF pipeline to create it.", well.well_id, well.well_id)
        bf_planes.update(focus)

    roi_masks = _selection_masks(_selection_names(pl_dict.get("steps", [])), context)
    crops: dict = {}
    px_an = context.pixel_size_um

    if diag_cfg.get("whole"):
        rgb = composite_rgb(planes, colors)
        if stamp and roi_masks:
            rgb = stamp_outlines(rgb, roi_masks)
        save_diagnostic_image(rgb, out_dir / f"{well.well_id}_whole", formats)
        if multichannel:
            _save_multichannel_tif(
                planes, channel_names, bf_planes,
                out_dir / f"{well.well_id}_whole_channels.tif", colors=colors,
            )

    if diag_cfg.get("crops"):
        pad_px = round(diag_cfg.get("crop_pad_um", 0.0) / (pixel_size_um or 1.0))
        # None = every selection (the historical behaviour). A list restricts
        # crops to those mask keys — a background/whole selection's "crop" is
        # the full frame, so cropping it only doubles the output.
        wanted = diag_cfg.get("crop_selections")
        zstack = diag_cfg.get("zstack_crops", False)
        bin_factor = int(context.metadata.get("binning_factor", 1) or 1)
        bin_method = next((s.get("method", "mean") for s in pl_dict.get("steps", [])
                           if s.get("type") == "Binning"), "mean")
        to_crop = (roi_masks if wanted is None
                   else {k: v for k, v in roi_masks.items() if k in set(wanted)})
        if wanted is not None and not to_crop and roi_masks:
            log.warning(
                "Well %s: crops are on but none of the chosen selections %s matched "
                "this well's ROIs %s — no crops written.",
                well.well_id, sorted(wanted), sorted(roi_masks),
            )
        for roi_name, mask in to_crop.items():
            bbox = bbox_from_mask(mask, pad_px)
            if bbox is None:
                continue
            r0, r1, c0, c1 = bbox
            info = _crop_info(well.well_id, roi_name, bbox, px_an, bin_factor)
            stem = f"{well.well_id}_{roi_name}_crop"
            files = [f"{stem}{ext}" for fmt, ext in (("tiff", ".tif"), ("jpg", ".jpg"))
                     if fmt in formats]
            cropped_planes = [p[r0:r1, c0:c1] for p in planes]
            rgb = composite_rgb(cropped_planes, colors)
            if stamp:
                rgb = stamp_outlines(rgb, {roi_name: mask[r0:r1, c0:c1]})
            save_diagnostic_image(rgb, out_dir / stem, formats)
            if multichannel:
                bf_crop = {k: v[r0:r1, c0:c1] for k, v in bf_planes.items()}
                _save_multichannel_tif(
                    cropped_planes, channel_names, bf_crop,
                    out_dir / f"{stem}_channels.tif",
                    colors=colors, pixel_size_um=px_an, info=info,
                )
                files.append(f"{stem}_channels.tif")
            if zstack and raw_stack is not None:
                if _save_zstack_crop(
                    raw_stack, bbox, bin_factor, bin_method, channel_names, colors,
                    px_an, z_step_um, out_dir / f"{stem}_zstack.tif", info=info,
                ):
                    files.append(f"{stem}_zstack.tif")
                else:
                    log.info("Well %s: image has no Z axis — no Z-stack crop written.",
                             well.well_id)
                    zstack = False
            crops[roi_name] = {"bbox": bbox, "files": files}

    if diag_cfg.get("save_particle_labels"):
        import tifffile
        for key, mask in context.masks.items():
            if key.startswith("particles_ch") and mask is not None and mask.max() > 0:
                tifffile.imwrite(str(out_dir / f"{well.well_id}_{key}.tif"), mask.astype("uint16"))

    return crops


def _selection_names(steps) -> list[str]:
    """Mask keys of the ROI selections in a pipeline — step dicts or Step
    objects. Other masks in the context (thresholds, particle labels) aren't
    selections."""
    names = []
    for st in steps:
        typ = st.get("type") if isinstance(st, dict) else type(st).__name__
        name = st.get("roi_name") if isinstance(st, dict) else getattr(st, "roi_name", None)
        if typ in ("LoadROI", "ExtractROI") and name and name not in names:
            names.append(name)
    return names


def _selection_masks(names, context: PipelineContext) -> dict:
    return {n: context.masks[n] for n in names if context.masks.get(n) is not None}


def _selection_rows(names, context: PipelineContext, crops: dict | None = None) -> list[dict]:
    """Rows for the ``selections`` table: each selection's geometry on the
    analysis grid plus, if one was written, its diagnostic crop box."""
    import numpy as np

    px = float(context.pixel_size_um or 1.0)
    rows = []
    for name, mask in _selection_masks(names, context).items():
        m = np.asarray(mask, dtype=bool)
        ys, xs = np.nonzero(m)
        row = {"roi_mask": name, "roi_path": context.mask_paths.get(name, ""),
               "area_px": int(ys.size), "area_um2": ys.size * px * px}
        if ys.size:
            row.update(centroid_row=float(ys.mean()), centroid_col=float(xs.mean()),
                       bbox_row0=int(ys.min()), bbox_row1=int(ys.max()) + 1,
                       bbox_col0=int(xs.min()), bbox_col1=int(xs.max()) + 1)
        crop = (crops or {}).get(name)
        if crop:
            r0, r1, c0, c1 = crop["bbox"]
            row.update(crop_row0=r0, crop_row1=r1, crop_col0=c0, crop_col1=c1,
                       crop_files=crop["files"])
        rows.append(row)
    return rows


def _with_crop_positions(df, selection_rows: list[dict]):
    """Add ``crop_row``/``crop_col`` — the particle's centroid in its own
    selection's diagnostic crop. NaN where that selection has no crop."""
    import numpy as np

    origins = {r["roi_mask"]: (r["crop_row0"], r["crop_col0"])
               for r in selection_rows if r.get("crop_row0") is not None}
    if not origins or "centroid_row" not in df.columns:
        return df
    df = df.copy()
    o = df["roi_mask"].map(origins)
    df["crop_row"] = df["centroid_row"] - o.map(lambda t: t[0] if isinstance(t, tuple) else np.nan)
    df["crop_col"] = df["centroid_col"] - o.map(lambda t: t[1] if isinstance(t, tuple) else np.nan)
    return df


def bf_focus_plane(stack) -> tuple[int, list[float]]:
    """Index of the sharpest plane of a (Z, Y, X) stack and every plane's score.

    Score = variance of the Laplacian of the lightly smoothed plane (sigma 1,
    so pixel noise doesn't decide it); large planes are subsampled 2x for
    speed, which doesn't change which plane wins. Unstained cells in
    brightfield can be faintest exactly in focus; this picks the plane with
    the most fine edge detail, which for well/hole outlines is the focal one.
    """
    import numpy as np
    from scipy.ndimage import gaussian_filter, laplace

    scores = []
    for plane in stack:
        a = np.asarray(plane, dtype=np.float32)
        if min(a.shape) > 1024:
            a = a[::2, ::2]
        scores.append(float(laplace(gaussian_filter(a, 1.0)).var()))
    return int(np.argmax(scores)), scores


def bf_project_for_ilastik(ch_data, z_method: str, z_start: int, z_stop: int,
                           well_id: str = "", on_substack=None):
    """One BF channel ((Z, Y, X) or (Y, X)) → the uint16 image Ilastik gets:
    the Z range selected, projected with *z_method*, then
    :func:`bf_ilastik_input`. *on_substack* sees the selected sub-stack before
    projection (used to pick the focal plane). Shared by the batch projection
    and the training-image export, so a training image is exactly what the
    pipeline will later hand Ilastik."""
    import numpy as np

    from correlative_imaging.pipeline.ilastik import select_z_range

    ops = {"min": np.min, "max": np.max, "mean": np.mean, "sum": np.sum}
    if ch_data.ndim == 3:
        ch_data = select_z_range(ch_data, z_start, z_stop, axis=0, label=well_id)
        if on_substack is not None:
            on_substack(ch_data)
        ch_data = ops.get(z_method, np.min)(ch_data, axis=0)
    return bf_ilastik_input(ch_data, well_id)


def export_bf_training_images(wells, cfg: dict, out_dir, n: int, seed: int | None = None,
                              log_fn=print, should_abort=lambda: False) -> list:
    """Write *n* randomly chosen wells' BF projections — exactly the uint16
    images the BF pipeline hands Ilastik, same channel / Z range / projection
    as *cfg* — to *out_dir* for training a model on. *wells* is a list of
    ``(name, WellInfo)``; *name* becomes the file name (``<name>_proj.tif``).
    Also writes ``training_images.txt`` recording sources and settings.
    Returns the paths written."""
    import random

    import tifffile

    pool = [(name, w) for name, w in wells if w.bf_path is not None]
    if not pool:
        log_fn("No wells with a brightfield image.")
        return []
    rng = random.Random(seed)
    chosen = rng.sample(pool, min(n, len(pool)))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    z_start = int(cfg.get("z_start", 0) or 0)
    z_stop = int(cfg.get("z_stop", 0) or 0)
    lines = [f"BF channel {cfg.get('bf_channel', 0)}, projection {cfg.get('z_method', 'min')}, "
             f"Z range {z_start or 'first'}–{z_stop or 'last'}, raw values as uint16 "
             f"(identical to the BF pipeline's Ilastik input)", ""]
    written = []
    for i, (name, w) in enumerate(chosen, 1):
        if should_abort():
            log_fn("Aborted.")
            break
        try:
            img = read_image(w.bf_path)
            arr = bf_project_for_ilastik(img.data[int(cfg.get("bf_channel", 0))],
                                         cfg.get("z_method", "min"), z_start, z_stop, name)
            dest = out_dir / f"{name}_proj.tif"
            tifffile.imwrite(str(dest), arr)
            written.append(dest)
            lines.append(f"{dest.name}\t{w.bf_path}")
            log_fn(f"  [{i}/{len(chosen)}] {dest.name}  (uint16, {int(arr.min())}–{int(arr.max())})")
        except Exception as exc:
            log_fn(f"  [{i}/{len(chosen)}] {name}: failed — {exc}")
    (out_dir / "training_images.txt").write_text("\n".join(lines) + "\n")
    return written


def bf_ilastik_input(proj, well_id: str = ""):
    """The exact array handed to Ilastik — and saved as ``<well>_proj.tif`` —
    for a BF projection: the raw camera values as uint16.

    The reader gives float32, but a min/max projection of camera data holds
    whole numbers, so this is lossless for those. A mean projection is
    rounded; values outside 0–65535 (e.g. a sum projection) are clipped, with
    a warning, since uint16 cannot hold them. Never rescaled: what Ilastik was
    trained on (raw 16-bit projections) is what it gets, and the saved file is
    byte-for-byte that input, so a run can be reproduced from it.
    """
    import numpy as np

    a = np.asarray(proj)
    lo, hi = float(a.min()), float(a.max())
    if lo < 0 or hi > 65535:
        log.warning("Well %s: BF projection spans %.0f–%.0f, outside uint16 — clipped "
                    "for Ilastik (use min/max/mean projection).", well_id, lo, hi)
    return np.clip(np.rint(a), 0, 65535).astype(np.uint16)


# Planes saved either side of the sharpest BF plane; the diagnostics export
# includes 0 up to this many of them (Run tab → Diagnostic images).
BF_FOCUS_SAVED = 3


def _write_bf_vis(plane, path, info: str | None = None) -> None:
    """uint8 min–max scaled copy of a BF plane (or a Z, Y, X stack, scaled as
    one so the planes stay comparable), for display and diagnostics."""
    import numpy as np
    import tifffile

    mn, mx = plane.min(), plane.max()
    vis = (((plane - mn) / (mx - mn) * 255).astype(np.uint8)
           if mx > mn else np.zeros(plane.shape, dtype=np.uint8))
    if vis.ndim == 3:
        tifffile.imwrite(str(path), vis, imagej=True, compression="zlib",
                         metadata={"axes": "ZYX", **({"Info": info} if info else {})})
    else:
        tifffile.imwrite(str(path), vis, imagej=bool(info),
                         metadata={"Info": info} if info else None)


def _write_bf_focus(sub_stack, z_start: int, well_id: str, proj_dir) -> int:
    """Write ``<well>_focus.tif`` next to the projection: the sharpest plane of
    the BF stack (within the configured Z range) and up to
    :data:`BF_FOCUS_SAVED` planes either side, as one ImageJ Z-stack. Its Info
    says which plane in the file is the sharpest. Returns that plane's 1-based
    number in the full stack."""
    import numpy as np

    idx, scores = bf_focus_plane(sub_stack)
    first = max(1, z_start or 1)                      # plane number of sub_stack[0]
    lo, hi = max(0, idx - BF_FOCUS_SAVED), min(len(sub_stack), idx + BF_FOCUS_SAVED + 1)
    info = "\n".join([f"well = {well_id}", f"focus_plane = {first + idx}",
                       f"focus_index_in_file = {idx - lo}",
                       f"planes_in_file = {first + lo}-{first + hi - 1}",
                       f"planes_scored = {first}-{first + len(scores) - 1}",
                       "metric = variance of Laplacian (sigma 1)",
                       "scores = " + ", ".join(f"{v:.4g}" for v in scores)])
    _write_bf_vis(np.asarray(sub_stack[lo:hi]), Path(proj_dir) / f"{well_id}_focus.tif", info)
    return first + idx


def _bf_project_one(task: tuple) -> tuple:
    """Read one brightfield image, z-project the chosen channel, write the
    Ilastik HDF5 input, and (optionally) a uint8 projection TIFF.

    MODULE-LEVEL and Qt-free on purpose: the BF-pipeline projection phase
    fans these out across a ``ProcessPoolExecutor`` (spawn), so this must be
    importable in a worker process without dragging in the GUI/Qt. Mirrors
    the per-well projection that ``_BFWorker.run`` does inline for its
    sequential/test path — kept in sync with it by hand (small, stable).

    ``task`` = (bf_path, well_id, bf_channel, z_method, in_dir, proj_dir|None,
    z_start, z_stop). The Z range is 1-based inclusive with 0 = unbounded, the
    same convention as ``ZProjection``.
    Returns (stem, well_id, pixel_size_um, dtype_str, error|None); on failure
    ``stem`` is None and ``error`` holds the traceback.
    """
    import h5py
    import numpy as np
    import tifffile

    (bf_path_str, well_id, bf_channel, z_method, in_dir_str, proj_dir_str,
     z_start, z_stop) = task
    try:
        bf_path = Path(bf_path_str)
        stem = bf_path.stem
        img = read_image(bf_path)
        on_sub = ((lambda sub: _write_bf_focus(sub, z_start, well_id, proj_dir_str))
                  if proj_dir_str else None)
        ilastik_in = bf_project_for_ilastik(img.data[bf_channel], z_method, z_start, z_stop,
                                            well_id, on_substack=on_sub)
        h5_in = Path(in_dir_str) / f"{stem}.h5"
        with h5py.File(h5_in, "w") as f:
            f.create_dataset("data", data=ilastik_in[np.newaxis, np.newaxis, np.newaxis])
        if proj_dir_str:
            # Exactly the Ilastik input — reproducible, not a display copy.
            tifffile.imwrite(str(Path(proj_dir_str) / f"{well_id}_proj.tif"), ilastik_in)
        return (stem, well_id, float(img.pixel_size_um), str(ilastik_in.dtype), None)
    except Exception:
        return (None, well_id, None, None, traceback.format_exc())


def _run_pipeline_for_well(
    well: WellInfo,
    pl_dict: dict,
    missing: list[str],
    diag_cfg: dict | None = None,
    on_step_fn: Callable[[WellInfo, int, int, str], None] | None = None,
    trim_arrays: bool = False,
) -> _WellComputeResult:
    """Pure per-well compute: build the pipeline from an already-resolved
    ``pl_dict``, read the FL image, run, optionally write diagnostics, and
    return a :class:`_WellComputeResult`. Touches no shared state and never
    the database.

    This is a MODULE-LEVEL function (not a method or closure) on purpose: the
    Dask engine ships it to spawned worker processes, where it must be
    importable and where only picklable, plain-data arguments (``WellInfo``,
    a JSON-able ``pl_dict``, a list of labels) may cross the boundary — hence
    the pipeline dict is resolved by the caller, not a ``pipeline_dict_fn``
    closure here.

    ``trim_arrays`` (used only by the Dask path): after diagnostics have been
    written, drop each ``StepResult``'s big full-frame ``image``/``masks``
    arrays, since ``_write_result`` consumes only ``measurements``/``info`` —
    this keeps the result light when it's pickled back across the process
    boundary. Left False for the in-process (sequential/thread) paths, where
    there is no IPC and the result is used and discarded immediately.
    """
    pipeline_json = json.dumps(pl_dict)
    pl_name = pl_dict.get("name", "pipeline")
    try:
        pl = Pipeline(name=pl_name)
        for step_data in pl_dict.get("steps", []):
            pl.steps.append(Step.from_dict(step_data))

        image_data = read_image(well.fl_path)
        context = PipelineContext(
            channel_names=image_data.channel_names,
            pixel_size_um=image_data.pixel_size_um,
            z_step_um=image_data.z_step_um,
        )
        total_steps = len(pl.steps)
        on_step = None
        if on_step_fn:
            step_counter = [0]

            def on_step(step, result, current):
                step_counter[0] += 1
                on_step_fn(well, step_counter[0], total_steps, step.name)

        final_image, results = pl.run(image_data.data, context, on_step=on_step)

        crops: dict = {}
        if diag_cfg:
            try:
                crops = _save_well_diagnostics(
                    well, final_image, image_data.pixel_size_um, pl_dict, context, diag_cfg,
                    raw_stack=image_data.data, z_step_um=image_data.z_step_um,
                )
            except Exception:
                log.warning(
                    "Diagnostic image export failed for well %s:\n%s",
                    well.well_id, traceback.format_exc(),
                )

        # Before trimming: needs the selection masks.
        selections = _selection_rows(_selection_names(pl_dict.get("steps", [])), context, crops)

        if trim_arrays:
            for r in results:
                r.image = None
                r.masks = {}
                r.mask_paths = {}

        return _WellComputeResult(
            well=well, pl_name=pl_name, pipeline_json=pipeline_json,
            steps_and_results=list(zip(pl.steps, results)),
            image_meta={
                "n_channels": image_data.n_channels,
                "pixel_size_um": image_data.pixel_size_um,
                "analysis_pixel_size_um": context.pixel_size_um,
                "binning_factor": context.metadata.get("binning_factor", 1.0),
                "metadata": {
                    **image_data.metadata,
                    "well_id": well.well_id,
                    "row": well.row,
                    "col": well.col,
                    "field": well.field,
                    "missing_selections": missing,
                },
            },
            missing_selections=missing,
            selections=selections,
        )
    except Exception:
        return _WellComputeResult(
            well=well, pl_name=pl_name, pipeline_json=pipeline_json,
            error=traceback.format_exc(), missing_selections=missing,
        )


class WellBatchRunner:
    """Run a per-well pipeline over a list of :class:`WellInfo`.

    Unlike :class:`BatchRunner`, which processes every file in a directory
    independently, this iterates well-paired brightfield/fluorescence
    acquisitions and analyzes each well's fluorescence image against ROI(s)
    resolved specifically for that well (e.g. a per-well hole ROI detected by
    the BF pipeline, or a pre-existing per-well ROI file) — the pipeline dict
    itself is supplied per well via ``pipeline_dict_fn`` rather than being
    fixed once for the whole batch.
    """

    def __init__(self, db_path: str | Path, experiment: str = ""):
        self.db_path = Path(db_path)
        self.experiment = experiment

    # ------------------------------------------------------------------
    # Per-well compute — safe to call concurrently from multiple threads.
    # Touches no shared state: builds its own Pipeline/PipelineContext and
    # only reads well.fl_path, never the database.
    # ------------------------------------------------------------------

    def _process_well(
        self, well: WellInfo, pipeline_dict_fn: PipelineDictFn,
        diag_cfg: dict | None = None,
        on_step_fn: Callable[[WellInfo, int, int, str], None] | None = None,
    ) -> _WellComputeResult:
        pl_dict, missing = pipeline_dict_fn(well)
        return _run_pipeline_for_well(
            well, pl_dict, missing, diag_cfg, on_step_fn, trim_arrays=False,
        )

    # ------------------------------------------------------------------
    # DB write — must only ever be called from one thread (the orchestrating
    # thread that owns ``db``); SQLite connections aren't safe to share
    # across threads.
    # ------------------------------------------------------------------

    def _write_result(self, db: ResultsDB, result: _WellComputeResult) -> int | None:
        well = result.well
        if result.error is not None:
            log.error("Failed on well %s:\n%s", well.well_id, result.error)
            db.log_run(None, result.pl_name, result.pipeline_json,
                       status="error", error=result.error)
            return None

        image_id = db.register_image(
            well.fl_path,
            experiment=self.experiment,
            n_channels=result.image_meta["n_channels"],
            pixel_size_um=result.image_meta["pixel_size_um"],
            metadata=result.image_meta["metadata"],
            analysis_pixel_size_um=result.image_meta.get("analysis_pixel_size_um"),
            binning_factor=result.image_meta.get("binning_factor"),
        )
        n_particles = 0
        _save_channel_levels(db, image_id, result.steps_and_results)
        db.save_selections(image_id, result.selections)
        for step, step_result in result.steps_and_results:
            if step_result.measurements is None or step_result.measurements.empty:
                continue
            if isinstance(step, ParticleAnalysis):
                db.save_particle_measurements(
                    image_id, _with_crop_positions(step_result.measurements, result.selections))
                n_particles += step_result.info.get("n_particles", 0)
            elif isinstance(step, IntensityMeasurement):
                db.save_intensity_measurements(image_id, step_result.measurements)
            elif isinstance(step, ColocalizationAnalysis):
                db.save_colocalization(
                    image_id, step_result.measurements,
                    step_result.info.get("global_stats", {}),
                )
        db.log_run(image_id, result.pl_name, result.pipeline_json)
        return n_particles

    # ------------------------------------------------------------------
    # Dask engine — process-based, true multi-core. Compute in workers,
    # DB writes here on the main process (SQLite stays single-writer).
    # ------------------------------------------------------------------

    def _run_wells_dask(
        self, wells_with_fl, pipeline_dict_fn, dask_workers, progress_fn,
        should_abort, diag_cfg, note_missing, total, warn_fn, dashboard_fn=None,
    ) -> None:
        try:
            import dask
            from dask.distributed import Client, LocalCluster
            from dask.distributed import as_completed as dask_as_completed
        except ImportError as e:
            raise RuntimeError(
                "Dask parallel batch was requested but 'dask[distributed]' is not "
                "installed in this environment. Install it with:\n"
                "    pip install 'dask[distributed]'\n"
                "…or turn off the Dask option to use the thread-based path instead."
            ) from e

        # Force spawn (not fork): matches Windows semantics everywhere, and
        # avoids forking a process that may hold a live Bio-Formats JVM.
        dask.config.set({"distributed.worker.multiprocessing-method": "spawn"})

        # Resolve every well's pipeline dict HERE on the main process — cheap
        # (dict-building + ROI-file matching, no image I/O) — so the possibly
        # non-picklable pipeline_dict_fn closure never has to reach a worker;
        # only plain, picklable data crosses the process boundary.
        resolved = [(w, *pipeline_dict_fn(w)) for w in wells_with_fl]

        import os
        n_workers = dask_workers or max(1, (os.cpu_count() or 2) - 1)
        log.info("Starting Dask LocalCluster: %d process worker(s), 1 thread each …", n_workers)

        # Serialize startup only (see _DASK_CLUSTER_STARTUP_LOCK) — concurrent
        # plates each build their own cluster, and two starting at once race.
        with _DASK_CLUSTER_STARTUP_LOCK:
            cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1, processes=True)
            client = Client(cluster)
        try:
            dash = cluster.dashboard_link
            log.info("Dask dashboard: %s", dash)
            if warn_fn:
                warn_fn(f"Dask dashboard: {dash}")
            if dashboard_fn:
                dashboard_fn(dash)

            # Surface unmatched selections up front (compute is remote, so we
            # can't rely on doing it as results arrive — same counts either way).
            for (w, _pl, missing) in resolved:
                note_missing(w, missing)

            futures = [
                client.submit(
                    _run_pipeline_for_well, w, pl_dict, missing, diag_cfg, None, True,
                    pure=False,
                )
                for (w, pl_dict, missing) in resolved
            ]

            done = 0
            with ResultsDB(self.db_path) as db:
                for fut in dask_as_completed(futures):
                    if should_abort and should_abort():
                        log.info("Dask batch aborted — cancelling remaining tasks.")
                        client.cancel(futures)
                        break
                    result = fut.result()
                    n_particles = self._write_result(db, result)
                    done += 1
                    if progress_fn:
                        progress_fn(done, total, result.well.well_id, n_particles)
        finally:
            client.close()
            cluster.close()

    # ------------------------------------------------------------------

    def run_wells(
        self,
        wells: list[WellInfo],
        pipeline_dict_fn: PipelineDictFn,
        progress_fn: ProgressFn | None = None,
        should_abort: Callable[[], bool] | None = None,
        max_workers: int = 1,
        warn_fn: Callable[[str], None] | None = None,
        diag_cfg: dict | None = None,
        on_step_fn: Callable[[WellInfo, int, int, str], None] | None = None,
        use_dask: bool = False,
        dask_workers: int | None = None,
        dashboard_fn: Callable[[str], None] | None = None,
    ) -> None:
        """Process every well with an FL image, using a pipeline built per well.

        Wells without ``fl_path`` are skipped (logged, not an error).

        Any per-well-dynamic ROI selection (BF-pipeline hole/background,
        existing project ROI) that has no matching file for a given well is
        never silently dropped: it's reported immediately via ``warn_fn`` (if
        given) and via the standard logger either way, and the well's
        ``images.metadata`` JSON records which selection(s) were unmatched. A
        summary count per selection is emitted once at the end of the run.

        diag_cfg: optional diagnostic-image export config (see
        ``RunTab._on_run`` for the dict shape) — when given, ``_process_well``
        also writes composite diagnostic image(s) per well.

        on_step_fn: optional callback ``(well, step_index, total_steps,
        step_name)`` invoked after each pipeline step within a well — drives
        a per-image step-level progress indicator, distinct from the
        well-level ``progress_fn``. Called from whichever thread is running
        that well (may not be the calling thread when ``max_workers > 1``).

        max_workers: EXPERIMENTAL. 1 (default) = original sequential
        behavior, byte-for-byte, writing directly to ``db_path``. >1 gives
        each worker thread its own sub-database (``<db_path>.partN<ext>``) —
        a whole contiguous slice of wells per thread, compute *and* write —
        so SQLite is never touched by more than one thread at a time and
        writes are never serialized onto a single connection. All sub-DBs
        are merged into ``db_path`` at the end (remapping ``image_id``
        foreign keys, since each sub-DB has its own independent autoincrement
        sequence) and kept on disk afterward, not deleted.

        use_dask: when True, the per-well compute runs in a Dask
        ``LocalCluster`` of separate *processes* (true multi-core, unlike the
        thread pool which is GIL-limited for the Python-heavy parts). Only
        plain data is shipped to workers; DB writes stay on this (main)
        process, so SQLite is single-writer as always. ``max_workers`` (the
        thread path) is ignored when this is set. Requires
        ``dask[distributed]`` — raises a clear error if not installed.
        dask_workers: process-worker count for the Dask cluster (default:
        CPU count − 1).
        """
        wells_with_fl = [w for w in wells if w.fl_path is not None]
        if not wells_with_fl:
            log.warning("No wells with an FL image to process")
            return
        total = len(wells_with_fl)

        from collections import Counter
        import threading
        missing_counts: Counter = Counter()
        missing_lock = threading.Lock()

        def _note_missing(well: WellInfo, missing: list[str]) -> None:
            if not missing:
                return
            with missing_lock:
                missing_counts.update(missing)
            msg = f"⚠ {well.well_id}: no match for {', '.join(missing)} — skipped for this well"
            log.warning(msg)
            if warn_fn:
                warn_fn(msg)

        if use_dask:
            self._run_wells_dask(
                wells_with_fl, pipeline_dict_fn, dask_workers, progress_fn,
                should_abort, diag_cfg, _note_missing, total, warn_fn,
                dashboard_fn,
            )
        elif max_workers <= 1:
            with ResultsDB(self.db_path) as db:
                iterator = (
                    tqdm(wells_with_fl, unit="well", desc="Batch")
                    if progress_fn is None else wells_with_fl
                )
                for idx, well in enumerate(iterator):
                    if should_abort and should_abort():
                        log.info("Batch aborted at well %s", well.well_id)
                        break
                    result = self._process_well(well, pipeline_dict_fn, diag_cfg, on_step_fn)
                    _note_missing(well, result.missing_selections)
                    n_particles = self._write_result(db, result)
                    if progress_fn:
                        progress_fn(idx + 1, total, well.well_id, n_particles)
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            log.info(
                "Running well batch with %d worker threads, one sub-DB each (experimental)",
                max_workers,
            )
            n_chunks = min(max_workers, len(wells_with_fl))
            chunks = [wells_with_fl[i::n_chunks] for i in range(n_chunks)]
            progress_lock = threading.Lock()
            done = [0]

            def _run_chunk(chunk_idx: int, chunk: list[WellInfo]) -> Path:
                sub_db_path = self.db_path.with_name(
                    f"{self.db_path.stem}.part{chunk_idx}{self.db_path.suffix}"
                )
                with ResultsDB(sub_db_path) as db:
                    for well in chunk:
                        if should_abort and should_abort():
                            log.info("Batch aborted at well %s (chunk %d)", well.well_id, chunk_idx)
                            break
                        result = self._process_well(well, pipeline_dict_fn, diag_cfg, on_step_fn)
                        _note_missing(well, result.missing_selections)
                        n_particles = self._write_result(db, result)
                        with progress_lock:
                            done[0] += 1
                            current = done[0]
                        if progress_fn:
                            progress_fn(current, total, well.well_id, n_particles)
                return sub_db_path

            sub_db_paths: list[Path] = []
            with ThreadPoolExecutor(max_workers=n_chunks) as executor:
                futures = {
                    executor.submit(_run_chunk, i, chunk): i
                    for i, chunk in enumerate(chunks)
                }
                for future in as_completed(futures):
                    sub_db_paths.append(future.result())

            log.info("Merging %d sub-DB(s) into %s …", len(sub_db_paths), self.db_path)
            with ResultsDB(self.db_path) as db:
                for sub_path in sorted(sub_db_paths):
                    n = db.merge_from(sub_path)
                    log.info("  merged %d image(s) from %s", n, sub_path.name)

        if missing_counts:
            summary = "Unmatched ROI selections: " + ", ".join(
                f"'{label}' missing for {n}/{total} well(s)" for label, n in missing_counts.items()
            )
            log.warning(summary)
            if warn_fn:
                warn_fn(summary)

        log.info("Well batch complete. Results → %s", self.db_path)


def _save_channel_levels(db: ResultsDB, image_id: int, steps_and_results) -> None:
    """Record, per channel, the black level and threshold this image got —
    one ``channel_levels`` row per channel that has either step, plus one per
    (channel, selection) with per-selection thresholding."""
    rows: dict[tuple, dict] = {}
    for step, result in steps_and_results:
        if isinstance(step, BlackLevelNormalization):
            r = rows.setdefault((step.channel, ""), {"channel_index": step.channel})
            r["channel"] = result.info.get("channel", f"ch{step.channel}")
            r["black_level"] = result.info.get("black_level")
            r["black_method"] = (step.method if step.method != "percentile"
                                 else f"percentile {step.percentile}")
        elif isinstance(step, AutoThreshold):
            r = rows.setdefault((step.channel, step.roi_mask),
                                {"channel_index": step.channel, "roi_mask": step.roi_mask})
            r["channel"] = result.info.get("channel", f"ch{step.channel}")
            r["threshold_value"] = result.info.get("threshold_value")
            r["threshold_method"] = step.method
            r["threshold_z_projection"] = step.z_projection
    db.save_channel_levels(image_id, [rows[c] for c in sorted(rows)])
