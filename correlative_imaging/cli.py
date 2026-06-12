"""Command-line interface for the correlative imaging pipeline."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

console = Console()


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ------------------------------------------------------------------
# CLI group
# ------------------------------------------------------------------

@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging.")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """Correlative Imaging — microscopy analysis pipeline."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    _setup_logging(verbose)


# ------------------------------------------------------------------
# ci info  — inspect an image without running any analysis
# ------------------------------------------------------------------

@cli.command()
@click.argument("image_path", type=click.Path(exists=True, path_type=Path))
@click.option("--scene", default=0, show_default=True, help="Scene/series index.")
def info(image_path: Path, scene: int) -> None:
    """Print metadata for IMAGE_PATH (dimensions, channels, pixel size)."""
    from correlative_imaging.io import read_image

    img = read_image(image_path, scene=scene)
    t = Table(title=str(image_path.name), show_header=True, header_style="bold cyan")
    t.add_column("Property")
    t.add_column("Value")
    t.add_row("Shape", str(img.data.shape))
    t.add_row("Channels", ", ".join(img.channel_names))
    t.add_row("Pixel size XY", f"{img.pixel_size_um:.4f} µm")
    t.add_row("Z step", f"{img.z_step_um:.4f} µm")
    t.add_row("Is Z-stack", str(img.is_zstack))
    t.add_row("dtype", str(img.data.dtype))
    console.print(t)


# ------------------------------------------------------------------
# ci run  — single image
# ------------------------------------------------------------------

@cli.command()
@click.argument("image_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--pipeline", "-p",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to a pipeline JSON file.",
)
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory (default: image folder).",
)
@click.option("--preview", is_flag=True, help="Open napari after processing.")
@click.option("--scene", default=0, show_default=True)
@click.pass_context
def run(
    ctx: click.Context,
    image_path: Path,
    pipeline: Path,
    output: Path | None,
    preview: bool,
    scene: int,
) -> None:
    """Run PIPELINE on a single IMAGE_PATH."""
    from correlative_imaging.io import read_image
    from correlative_imaging.pipeline import Pipeline
    from correlative_imaging.pipeline.base import PipelineContext
    from correlative_imaging.pipeline.analyze import ParticleAnalysis, IntensityMeasurement
    from correlative_imaging.pipeline.colocalize import ColocalizationAnalysis
    from correlative_imaging.storage import ResultsDB

    output_dir = output or image_path.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold]Loading[/bold] {image_path.name}")
    image_data = read_image(image_path, scene=scene)
    pl = Pipeline.load(pipeline)

    console.print(f"[bold]Running[/bold] pipeline '{pl.name}' ({len(pl.steps)} steps)")
    context = PipelineContext(
        channel_names=image_data.channel_names,
        pixel_size_um=image_data.pixel_size_um,
        z_step_um=image_data.z_step_um,
    )
    _, results = pl.run(image_data.data, context)

    with ResultsDB(output_dir / "results.db") as db:
        image_id = db.register_image(
            image_path,
            n_channels=image_data.n_channels,
            pixel_size_um=image_data.pixel_size_um,
            metadata=image_data.metadata,
        )
        for step, result in zip(pl.steps, results):
            if result.measurements is not None and not result.measurements.empty:
                if isinstance(step, ParticleAnalysis):
                    db.save_particle_measurements(image_id, result.measurements)
                    console.print(
                        f"  [green]✓[/green] {step.name}: "
                        f"{result.info.get('n_particles', '?')} particles"
                    )
                elif isinstance(step, IntensityMeasurement):
                    db.save_intensity_measurements(image_id, result.measurements)
                    console.print(
                        f"  [green]✓[/green] {step.name}: "
                        f"mean={result.info.get('mean_intensity', 0):.1f}"
                    )
                elif isinstance(step, ColocalizationAnalysis):
                    db.save_colocalization(
                        image_id,
                        result.measurements,
                        result.info.get("global_stats", {}),
                    )
                    gs = result.info.get("global_stats", {})
                    console.print(
                        f"  [green]✓[/green] {step.name}: "
                        f"M1={gs.get('manders_m1', 0):.3f}  "
                        f"M2={gs.get('manders_m2', 0):.3f}  "
                        f"Pearson={gs.get('pearson_r', 0):.3f}"
                    )
        pipeline_json = json.dumps(
            {"name": pl.name, "steps": [s.to_dict() for s in pl.steps]}
        )
        db.log_run(image_id, pl.name, pipeline_json)

    console.print(f"[bold green]Done.[/bold green] Results → {output_dir}/results.db")

    if preview:
        from correlative_imaging.viewer import show_pipeline_preview
        console.print("[bold]Opening napari preview…[/bold]")
        # Re-run so we can pass the on_step callback to the viewer
        show_pipeline_preview(image_data, pl)


# ------------------------------------------------------------------
# ci batch  — directory of images
# ------------------------------------------------------------------

@cli.command()
@click.argument("input_dir", type=click.Path(exists=True, path_type=Path))
@click.option("--pipeline", "-p", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None)
@click.option("--pattern", default="**/*", show_default=True, help="Glob pattern.")
@click.option("--no-recursive", is_flag=True, help="Do not recurse into subdirectories.")
@click.option("--experiment", default="", help="Experiment label for the database.")
@click.option("--parquet", is_flag=True, help="Also export results as parquet files.")
def batch(
    input_dir: Path,
    pipeline: Path,
    output: Path | None,
    pattern: str,
    no_recursive: bool,
    experiment: str,
    parquet: bool,
) -> None:
    """Run PIPELINE on all images in INPUT_DIR."""
    from correlative_imaging.pipeline import Pipeline
    from correlative_imaging.batch import BatchRunner

    pl = Pipeline.load(pipeline)
    output_dir = output or input_dir / "output"
    runner = BatchRunner(pl, db_path=output_dir / "results.db", experiment=experiment)
    runner.run_directory(
        input_dir,
        output_dir=output_dir,
        pattern=pattern,
        recursive=not no_recursive,
        export_parquet=parquet,
    )


# ------------------------------------------------------------------
# ci preview  — open napari without saving results
# ------------------------------------------------------------------

@cli.command()
@click.argument("image_path", type=click.Path(exists=True, path_type=Path))
@click.option("--pipeline", "-p", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--scene", default=0, show_default=True)
def preview(image_path: Path, pipeline: Path, scene: int) -> None:
    """Interactively preview the pipeline on IMAGE_PATH using napari."""
    from correlative_imaging.io import read_image
    from correlative_imaging.pipeline import Pipeline
    from correlative_imaging.viewer import show_pipeline_preview

    img = read_image(image_path, scene=scene)
    pl = Pipeline.load(pipeline)
    console.print(f"[bold]Opening napari for[/bold] {image_path.name}")
    show_pipeline_preview(img, pl)


# ------------------------------------------------------------------
# ci gui  — full GUI with folder picker and batch runner
# ------------------------------------------------------------------

@cli.command()
def gui() -> None:
    """Open the napari GUI with folder browser and batch runner."""
    from correlative_imaging.viewer.gui import launch_gui
    launch_gui()


if __name__ == "__main__":
    cli()
