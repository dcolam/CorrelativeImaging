"""Results Explorer — inspect, classify and eyeball a finished pipeline run.

Split into a Qt-free data layer (discovery + analysis, importable and testable
headless) and a thin GUI shell (explorer) that only renders it:

* :mod:`correlative_imaging.results.discovery` — find output folders, group the
  per-plate result databases of one run together, locate diagnostic images.
* :mod:`correlative_imaging.results.analysis` — load a run's databases into a
  tidy per-well table with derived per-channel intensities, all pandas + sqlite.
* :mod:`correlative_imaging.results.classify` — multi-label hole-colour call:
  each channel positive/negative independently, tunable + inspectable.
* :mod:`correlative_imaging.results.labels` — persisted hand ground-truth labels
  (sidecar SQLite), schema matching the classifier output for direct validation.

The GUI window (``explorer``) imports these; they never import Qt or napari, so
the analysis can be exercised and validated without a display.
"""
