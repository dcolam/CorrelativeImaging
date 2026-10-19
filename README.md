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

### 1. Get the code

```bash
git clone https://github.com/dcolam/CorrelativeImaging.git
cd CorrelativeImaging
```

(or copy the folder). Everything below runs from this folder.

### 2. Create an environment

Requires **Python 3.11–3.13**; some dependencies don't yet ship wheels for newer versions. Conda (Miniconda or
Anaconda) is the simplest:

```bash
conda create -n corrimg python=3.13
conda activate corrimg
```

A plain virtual environment works too: `python -m venv .venv`, then activate it (`.venv\Scripts\activate` on
Windows, `source .venv/bin/activate` on Linux/macOS).

### 3. Install the package and its dependencies

Pick the extras you need. `-e` installs in editable mode, so code changes apply without reinstalling, and it puts
the `ci` command on the path.

```bash
pip install -e ".[viewer]"                # pipeline + GUIs (ci gui, ci preview, ci results)
pip install -e ".[viewer,reconstruction]" # + Cell reconstruction (ci reconstruct)
pip install -e ".[viewer,reconstruction,ml,dask,bioformats]"   # everything used day-to-day
```

| extra | what it adds | needed for |
|---|---|---|
| *(none)* | numpy, scipy, scikit-image ≥ 0.26, pandas, bioio + readers (TIFF, OME-TIFF, CZI, LIF, ND2, OME-Zarr, PNG/JPG), tifffile, roifile, pyarrow, pyyaml, click, rich, tqdm | `ci info / run / batch` (headless) |
| `viewer` | napari with a Qt backend | `ci gui`, `ci preview`, `ci results` |
| `reconstruction` | scikit-learn (+ joblib), napari ≥ 0.5 | `ci reconstruct` and `python -m correlative_imaging.reconstruction …` (detect, classify, train) |
| `ml` | scikit-learn, umap-learn, matplotlib, pyqtgraph | Results Explorer: LDA, Clusters tab |
| `dask` | dask[distributed], bokeh | "Use Dask" multi-process batch engine + dashboard |
| `bioformats` | bioio-bioformats + a bundled JDK 17 (jdk4py) | Olympus VSI, SCN, ICS, … (no system Java needed) |
| `gpu` | cupy, cucim (CUDA 12) | GPU-accelerated array ops; needs a matching CUDA toolkit |
| `test` / `dev` | pytest (+ pytest-cov, ruff) | running `tests/` |
| `analysis` | jupyter, matplotlib | `notebooks/` |

`napari[all]` installs PyQt5 by default. If the environment already has another Qt binding (e.g. PyQt6), napari
uses that one; don't mix bindings by hand.

Versions this was last tested with (Windows, conda env `corrimg`, Python 3.13): numpy 2.4, scipy 1.17,
scikit-image 0.26, pandas 3.0, pyarrow 24, scikit-learn 1.9, napari 0.7 (PyQt6 6.11).

### 4. Check the installation

```bash
ci --help                                         # lists info, run, batch, preview, gui, results, reconstruct
python -c "import correlative_imaging, napari, sklearn; print('ok')"
pip install -e ".[test]" && pytest tests          # optional
```

### External tools (optional)

- **Ilastik.** Ilastik-based ROI extraction needs a separate [Ilastik](https://www.ilastik.org/download.html)
  installation. Point the pipeline to it with the `ILASTIK_PATH` environment variable, or install it in a standard
  location.
- **Cell-reconstruction models.** A trained model (`.joblib`) is tied to the scikit-learn version it was saved with
  (the version is recorded in the `.json` next to it). After a scikit-learn upgrade, retrain in the Train tab or
  with `python -m correlative_imaging.reconstruction train …`.

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

# Inspect a finished run (well grid, diagnostics, hole-colour classifier)
ci results path/to/output_folder

# Cell reconstruction: find the cell in each hole, label particles, train, call each cell's channels
# (also reachable from ci gui: Run tab → "Cell reconstruction …", or the bar shown when a run finishes)
ci reconstruct [path/to/reconstruction_results_folder]
```

Cell reconstruction also has a command line for headless runs
(`python -m correlative_imaging.reconstruction detect | classify | train | import-labels`). See
`correlative_imaging/reconstruction/README.md`.

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
├── cli.py            # click-based CLI (info / run / batch / preview / gui / results / reconstruct)
├── batch.py           # headless batch runner
├── io/                # multi-format readers, well-plate discovery/pairing
├── pipeline/          # Step/Pipeline core + preprocessing, segmentation, Ilastik, analysis, colocalization
├── results/           # Results Explorer (ci results)
├── reconstruction/    # Cell reconstruction (ci reconstruct): particle classifier, cell reconstruction, napari GUI
├── storage/           # SQLite results database
└── viewer/            # napari-based GUI and preview
fiji/                  # legacy ImageJ/Fiji macros this package replaces
```
