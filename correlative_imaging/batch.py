"""Headless batch runner — process a directory of images with a saved pipeline."""

from __future__ import annotations

import json
import logging
import traceback
from collections.abc import Callable
from pathlib import Path

from tqdm import tqdm

from correlative_imaging.io import read_image, supported_extensions
from correlative_imaging.io.plate import WellInfo
from correlative_imaging.pipeline.base import Pipeline, PipelineContext, Step
from correlative_imaging.pipeline.analyze import IntensityMeasurement, ParticleAnalysis
from correlative_imaging.pipeline.colocalize import ColocalizationAnalysis
from correlative_imaging.storage import ResultsDB

log = logging.getLogger(__name__)

# Signature: (current_index, total, filename, n_particles_or_None)
ProgressFn = Callable[[int, int, str, int | None], None]

# Signature: (well) -> pipeline dict (JSON-serializable), with any ROI steps
# already resolved to this well's own file(s).
PipelineDictFn = Callable[[WellInfo], dict]


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

    def run_wells(
        self,
        wells: list[WellInfo],
        pipeline_dict_fn: PipelineDictFn,
        progress_fn: ProgressFn | None = None,
        should_abort: Callable[[], bool] | None = None,
    ) -> None:
        """Process every well with an FL image, using a pipeline built per well.

        Wells without ``fl_path`` are skipped (logged, not an error).
        """
        wells_with_fl = [w for w in wells if w.fl_path is not None]
        if not wells_with_fl:
            log.warning("No wells with an FL image to process")
            return

        with ResultsDB(self.db_path) as db:
            iterator = (
                tqdm(wells_with_fl, unit="well", desc="Batch")
                if progress_fn is None else wells_with_fl
            )
            for idx, well in enumerate(iterator):
                if should_abort and should_abort():
                    log.info("Batch aborted at well %s", well.well_id)
                    break

                image_id = None
                pl_dict = pipeline_dict_fn(well)
                pipeline_json = json.dumps(pl_dict)
                try:
                    pl = Pipeline(name=pl_dict.get("name", "pipeline"))
                    for step_data in pl_dict.get("steps", []):
                        pl.steps.append(Step.from_dict(step_data))

                    image_data = read_image(well.fl_path)
                    image_id = db.register_image(
                        well.fl_path,
                        experiment=self.experiment,
                        n_channels=image_data.n_channels,
                        pixel_size_um=image_data.pixel_size_um,
                        metadata={
                            **image_data.metadata,
                            "well_id": well.well_id,
                            "row": well.row,
                            "col": well.col,
                            "field": well.field,
                        },
                    )
                    context = PipelineContext(
                        channel_names=image_data.channel_names,
                        pixel_size_um=image_data.pixel_size_um,
                        z_step_um=image_data.z_step_um,
                    )
                    _, results = pl.run(image_data.data, context)

                    n_particles = 0
                    for step, result in zip(pl.steps, results):
                        if result.measurements is None or result.measurements.empty:
                            continue
                        if isinstance(step, ParticleAnalysis):
                            db.save_particle_measurements(image_id, result.measurements)
                            n_particles += result.info.get("n_particles", 0)
                        elif isinstance(step, IntensityMeasurement):
                            db.save_intensity_measurements(image_id, result.measurements)
                        elif isinstance(step, ColocalizationAnalysis):
                            db.save_colocalization(
                                image_id,
                                result.measurements,
                                result.info.get("global_stats", {}),
                            )

                    db.log_run(image_id, pl.name, pipeline_json)
                    if progress_fn:
                        progress_fn(idx + 1, len(wells_with_fl), well.well_id, n_particles)

                except Exception:
                    err = traceback.format_exc()
                    log.error("Failed on well %s:\n%s", well.well_id, err)
                    db.log_run(
                        image_id,
                        pl_dict.get("name", "pipeline"),
                        pipeline_json,
                        status="error",
                        error=err,
                    )
                    if progress_fn:
                        progress_fn(idx + 1, len(wells_with_fl), well.well_id, None)

        log.info("Well batch complete. Results → %s", self.db_path)
