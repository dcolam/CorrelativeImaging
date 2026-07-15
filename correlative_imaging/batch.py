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

                    for step, result in zip(self.pipeline.steps, results):
                        if result.measurements is not None and not result.measurements.empty:
                            if isinstance(step, ParticleAnalysis):
                                db.save_particle_measurements(image_id, result.measurements)
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


def _project_planes(arr) -> "np.ndarray":
    """Collapse a (C, Z, Y, X) pipeline image to (C, Y, X) via max-projection;
    no-op for an already-2-D-per-channel (C, Y, X) array."""
    return arr.max(axis=1) if arr.ndim == 4 else arr


def _save_well_diagnostics(
    well, final_image, pixel_size_um: float, pl_dict: dict,
    context: PipelineContext, diag_cfg: dict,
) -> None:
    """Write whole-image and/or per-ROI-crop composite diagnostic images for
    one well, per ``diag_cfg`` (see ``RunTab._on_run`` for the dict shape).

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

    roi_names = [
        s["roi_name"] for s in pl_dict.get("steps", [])
        if s.get("type") in ("LoadROI", "ExtractROI") and "roi_name" in s
    ]
    roi_masks = {name: context.masks[name] for name in roi_names if context.masks.get(name) is not None}

    if diag_cfg.get("whole"):
        rgb = composite_rgb(planes, colors)
        if stamp and roi_masks:
            rgb = stamp_outlines(rgb, roi_masks)
        save_diagnostic_image(rgb, out_dir / f"{well.well_id}_whole", formats)

    if diag_cfg.get("crops"):
        pad_px = round(diag_cfg.get("crop_pad_um", 0.0) / (pixel_size_um or 1.0))
        for roi_name, mask in roi_masks.items():
            bbox = bbox_from_mask(mask, pad_px)
            if bbox is None:
                continue
            r0, r1, c0, c1 = bbox
            cropped_planes = [p[r0:r1, c0:c1] for p in planes]
            rgb = composite_rgb(cropped_planes, colors)
            if stamp:
                rgb = stamp_outlines(rgb, {roi_name: mask[r0:r1, c0:c1]})
            save_diagnostic_image(rgb, out_dir / f"{well.well_id}_{roi_name}_crop", formats)

    if diag_cfg.get("save_particle_labels"):
        import tifffile
        for key, mask in context.masks.items():
            if key.startswith("particles_ch") and mask is not None and mask.max() > 0:
                tifffile.imwrite(str(out_dir / f"{well.well_id}_{key}.tif"), mask.astype("uint16"))


def _bf_project_one(task: tuple) -> tuple:
    """Read one brightfield image, z-project the chosen channel, write the
    Ilastik HDF5 input, and (optionally) a uint8 projection TIFF.

    MODULE-LEVEL and Qt-free on purpose: the BF-pipeline projection phase
    fans these out across a ``ProcessPoolExecutor`` (spawn), so this must be
    importable in a worker process without dragging in the GUI/Qt. Mirrors
    the per-well projection that ``_BFWorker.run`` does inline for its
    sequential/test path — kept in sync with it by hand (small, stable).

    ``task`` = (bf_path, well_id, bf_channel, z_method, in_dir, proj_dir|None).
    Returns (stem, well_id, pixel_size_um, dtype_str, error|None); on failure
    ``stem`` is None and ``error`` holds the traceback.
    """
    import h5py
    import numpy as np
    import tifffile

    bf_path_str, well_id, bf_channel, z_method, in_dir_str, proj_dir_str = task
    try:
        ops = {"min": np.min, "max": np.max, "mean": np.mean, "sum": np.sum}
        proj_fn = ops.get(z_method, np.min)
        bf_path = Path(bf_path_str)
        stem = bf_path.stem
        img = read_image(bf_path)
        ch_data = img.data[bf_channel]              # (Z,H,W) or (H,W)
        if ch_data.ndim == 3:
            ch_data = proj_fn(ch_data, axis=0)      # → (H,W), original dtype
        h5_in = Path(in_dir_str) / f"{stem}.h5"
        with h5py.File(h5_in, "w") as f:
            f.create_dataset("data", data=ch_data[np.newaxis, np.newaxis, np.newaxis])
        if proj_dir_str:
            mn, mx = ch_data.min(), ch_data.max()
            ch_vis = (((ch_data - mn) / (mx - mn) * 255).astype(np.uint8)
                      if mx > mn else np.zeros_like(ch_data, dtype=np.uint8))
            tifffile.imwrite(str(Path(proj_dir_str) / f"{well_id}_proj.tif"), ch_vis)
        return (stem, well_id, float(img.pixel_size_um), str(ch_data.dtype), None)
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

        if diag_cfg:
            try:
                _save_well_diagnostics(
                    well, final_image, image_data.pixel_size_um, pl_dict, context, diag_cfg,
                )
            except Exception:
                log.warning(
                    "Diagnostic image export failed for well %s:\n%s",
                    well.well_id, traceback.format_exc(),
                )

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
        )
        n_particles = 0
        for step, step_result in result.steps_and_results:
            if step_result.measurements is None or step_result.measurements.empty:
                continue
            if isinstance(step, ParticleAnalysis):
                db.save_particle_measurements(image_id, step_result.measurements)
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
