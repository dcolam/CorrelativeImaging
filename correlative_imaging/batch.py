"""Headless batch runner — process a directory of images with a saved pipeline."""

from __future__ import annotations

import json
import logging
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

from correlative_imaging.diagnostics import bbox_from_mask, composite_rgb, save_diagnostic_image
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


def _save_well_diagnostics(well, image_data, pl_dict: dict, context: PipelineContext, diag_cfg: dict) -> None:
    """Write whole-image and/or per-ROI-crop composite diagnostic images for
    one well, per ``diag_cfg`` (see ``RunTab._on_run`` for the dict shape).

    Only the actual ROI *selections* configured in this well's pipeline are
    cropped — ``context.masks`` also holds intermediate segmentation/particle
    label masks (``mask_ch{c}``, ``particles_ch{c}``) that aren't selections
    and must not be emitted as crops.
    """
    out_dir = Path(diag_cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    formats = diag_cfg["formats"]
    colors = diag_cfg.get("colors") or []
    planes = list(image_data.project("max"))

    if diag_cfg.get("whole"):
        rgb = composite_rgb(planes, colors)
        save_diagnostic_image(rgb, out_dir / f"{well.well_id}_whole", formats)

    if diag_cfg.get("crops"):
        pixel_size_um = image_data.pixel_size_um or 1.0
        pad_px = round(diag_cfg.get("crop_pad_um", 0.0) / pixel_size_um)
        roi_names = [
            s["roi_name"] for s in pl_dict.get("steps", [])
            if s.get("type") in ("LoadROI", "ExtractROI") and "roi_name" in s
        ]
        for roi_name in roi_names:
            mask = context.masks.get(roi_name)
            if mask is None:
                continue
            bbox = bbox_from_mask(mask, pad_px)
            if bbox is None:
                continue
            r0, r1, c0, c1 = bbox
            cropped_planes = [p[r0:r1, c0:c1] for p in planes]
            rgb = composite_rgb(cropped_planes, colors)
            save_diagnostic_image(rgb, out_dir / f"{well.well_id}_{roi_name}_crop", formats)


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

            _, results = pl.run(image_data.data, context, on_step=on_step)

            if diag_cfg:
                try:
                    _save_well_diagnostics(well, image_data, pl_dict, context, diag_cfg)
                except Exception:
                    log.warning(
                        "Diagnostic image export failed for well %s:\n%s",
                        well.well_id, traceback.format_exc(),
                    )

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

        if max_workers <= 1:
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
