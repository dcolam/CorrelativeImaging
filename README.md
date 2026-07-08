# CorrelativeImaging

A Python pipeline for correlative microscopy image analysis: use one channel (e.g. brightfield) to find a region of interest, then quantify signal from another channel (e.g. fluorescence) within that region.

The core idea is a two-stage correlative workflow — segment a "where to look" image, then measure a "what's there" image inside that mask. ROI detection can be done with classical thresholding/watershed, or with an [Ilastik](https://www.ilastik.org/) pixel-classification model for cases where a trained classifier does a better job of picking out the object of interest than a fixed threshold.

This package is a modern Python rewrite of an older ImageJ/Fiji macro pipeline (see `fiji/` for the legacy scripts it replaces) and is part of a larger project on automated 384-well-plate imaging and analysis, [dcolam/ephacRTools](https://github.com/dcolam/ephacRTools).

## Features

- **Multi-format image I/O** via [bioio](https://github.com/bioio-devs/bioio) — CZI, OME-TIFF, LIF, ND2, OME-Zarr, and common formats (PNG/JPG/BMP), plus Bio-Formats support (VSI, SCN, ICS, …) as an optional extra
- **Composable pipeline steps** — background subtraction, Gaussian blur, normalization, auto-thresholding, watershed splitting, ROI extraction (classical or Ilastik-based), particle analysis, intensity measurement, and colocalization (Manders' coefficients, Pearson's r)
- **Pipelines defined as JSON** — build a pipeline once, reuse it across single images or whole batches (see `example_pipeline.json`)
- **Well-plate aware** — discovers and pairs brightfield/fluorescence acquisitions across a plate folder (built for Olympus VSI plate scans)
- **Batch processing** with results written to a SQLite database, with optional Parquet export
- **napari-based GUI** — folder picker, plate grid, per-channel pipeline configuration, and multiple ROI selections, in addition to a scriptable CLI

## Installation

Requires Python 3.11–3.13 (see the version pins in `pyproject.toml` — some dependencies don't yet ship wheels for newer Python versions).

```bash
conda create -n corrimg python=3.12
conda activate corrimg
pip install -e ".[viewer]"
```

Optional extras:

```bash
pip install -e ".[gpu]"         # CUDA-accelerated array ops (requires a matching CUDA toolkit)
pip install -e ".[bioformats]"  # Bio-Formats support for VSI/SCN/ICS/etc. — bundles its own JDK (jdk4py), no system Java required
pip install -e ".[dev]"         # pytest, ruff
```

Ilastik-based ROI extraction requires a separate [Ilastik](https://www.ilastik.org/download.html) installation; point the pipeline at it via the `ILASTIK_PATH` environment variable or a standard install location.

## Usage

```bash
# Inspect an image's metadata
ci info path/to/image.czi

# Run a pipeline on a single image
ci run path/to/image.czi --pipeline example_pipeline.json --preview

# Run a pipeline over a directory of images
ci batch path/to/plate_folder --pipeline example_pipeline.json --parquet

# Interactively preview a pipeline in napari
ci preview path/to/image.czi --pipeline example_pipeline.json

# Launch the full GUI (folder browser, plate grid, batch runner)
ci gui
```

Results are stored in a `results.db` SQLite database in the output directory, with tables for particle measurements, intensity measurements, colocalization statistics, and run metadata.

## Pipeline steps

Pipelines are a named, ordered list of steps, each configured with its own parameters:

| Step | Purpose |
|---|---|
| `BackgroundSubtraction` | Rolling-ball background removal |
| `GaussianBlur` | Smoothing |
| `Normalize` / `BrightnessContrast` | Intensity rescaling |
| `ZProjection` | Collapse a Z-stack to 2-D |
| `AutoThreshold` | Automated thresholding (e.g. Otsu) |
| `WatershedSplit` | Split touching objects |
| `ExtractROI` | Detect an ROI boundary via heavy blur + threshold |
| `IlastikROI` | Detect an ROI using a trained Ilastik pixel classifier |
| `LoadROI` | Load a pre-made ROI mask from file |
| `ParticleAnalysis` | Detect and measure particles (region properties) within an optional ROI mask |
| `IntensityMeasurement` | Measure bulk intensity within an ROI |
| `ColocalizationAnalysis` | Measure overlap between two channels (Manders', Pearson's) |

See `example_pipeline.json` for a complete pipeline definition.

## Project layout

```
correlative_imaging/
├── cli.py            # click-based CLI (info / run / batch / preview / gui)
├── batch.py           # headless batch runner
├── io/                # multi-format readers, well-plate discovery/pairing
├── pipeline/          # Step/Pipeline core + preprocessing, segmentation, Ilastik, analysis, colocalization
├── storage/           # SQLite results database
└── viewer/            # napari-based GUI and preview
fiji/                  # legacy ImageJ/Fiji macros this package replaces
```
