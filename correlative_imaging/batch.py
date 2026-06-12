"""Headless batch runner — process a directory of images with a saved pipeline."""

from __future__ import annotations

import json
import logging
import traceback
from collections.abc import Callable
from pathlib import Path

from tqdm import tqdm

from correlative_imaging.io import read_image, supported_extensions
from correlative_imaging.pipeline.base import Pipeline, PipelineContext
from correlative_imaging.pipeline.analyze import ParticleAnalysis
from correlative_imaging.pipeline.colocalize import ColocalizationAnalysis
from correlative_imaging.storage import ResultsDB

log = logging.getLogger(__name__)

# Signature: (current_index, total, filename, n_particles_or_None)
ProgressFn = Callable[[int, int, str, int | None], None]


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
