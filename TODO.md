# GUI redesign — agenda + progress

Queued during a design pass through every GUI tab. Checked items are built and
verified (real execution, not just compile-checks unless noted).

## STATUS as of this overnight session

Every numbered item is done except the two explicitly deferred by you:
**#10** (multi-plate folders — "later, not now") and **#26** (low-priority
summary/histogram tab — "not yet urgent"). Neither was started; stopped here
deliberately rather than adding a risky new tab wiring on top of an
uncommitted tree overnight.

**IMPORTANT — commit before anything else next session.** Git identity
(`user.name`/`user.email`) isn't configured in this dev environment, so
nothing from this session could be committed (and per my instructions I
won't set git config myself). All edits are saved to disk and durable, but
there is currently **no commit checkpoint** for tonight's work — run
`git config user.email/user.name` (once) and commit before doing anything
destructive (`checkout`/`reset`/branch switches). Changed/new files:
`correlative_imaging/viewer/gui.py`, `correlative_imaging/batch.py`,
`correlative_imaging/pipeline/analyze.py`, `correlative_imaging/storage/database.py`,
`correlative_imaging/viewer/napari_viewer.py`, `correlative_imaging/diagnostics.py` (new).

**One behavior change worth knowing about:** `save_particle_measurements`/
`save_intensity_measurements` now write SQL `NULL` (not `0`) for any metric
you didn't select to save (#17/#18). Correct and more honest, but if
`notebooks/plate_results_analysis.ipynb` or anything else downstream assumes
non-null numeric columns, a `.sum()`/`.mean()` will now see `NaN` for
selectively-saved runs. Runs with every metric selected (the default) are
unaffected.

**Testing note:** `tests/` is empty and `pytest` isn't installed in this
environment, so there's no pre-existing suite to regress-check against.
Every item below was verified with a real, purpose-built script exercising
the actual code path (not just `py_compile`) — synthetic wells through the
real `WellBatchRunner`/`ResultsDB`, a headless-constructed GUI cycling all
tabs, a fake `napari.qt.threading` harness driving the real preview code
(napari itself isn't installed here either) — details noted per item below.

## Recent (not part of the original numbered list)

- [x] Split "Setup" tab: it now has folders, the plate-scan controls
  (extension/filter/recursive/Scan button — moved back here per your
  follow-up), and a "Load existing pipeline JSON" section (moved here from
  Channels tab). Only the plate grid + selected-well detail live on the new
  **Plate** tab now, so neither tab overflows a short window. Pipeline
  Summary tab now scrolls internally too (its bottom controls could get
  clipped before). Verified headlessly: all 8 tabs build, cycle, JSON-load
  signal/round-trip still works, scan controls confirmed on Setup page.
- [x] **Fixed a real bug found while extending diagnostics**: `RunTab` was
  constructed without `get_channel_colors=`, so `self._get_channel_colors`
  was always `None` — every diagnostic image was composited with an empty
  color list (silently landing all-black) and the single-image preview
  always fell back to rotating step-colors instead of the user's assigned
  channel colors, regardless of what was set in the Channels tab's "Channel
  identity" box. Fixed by wiring `get_channel_colors=lambda: [p.display_color
  for p in self._channels_tab.get_panels()]` into the `RunTab(...)`
  constructor call. Verified with a regression test that builds the *real*
  `CorrelativeImagingWidget` (not a hand-fed `RunTab`), sets colors through
  the actual `ChannelPanel` combo, and confirms both the diagnostic `diag_cfg`
  and the preview layers pick them up — this is the exact path the earlier,
  narrower tests never exercised (they always supplied colors directly).
- [x] Diagnostic images now use each channel's **last pre-threshold
  processed image** (e.g. post-Normalize) as the composite base, not raw
  values — reuses `Pipeline.run()`'s own returned final image, which (since
  `AutoThreshold`/`WatershedSplit` never touch `result.image`) already holds
  exactly that per channel. Verified with a GaussianBlur step: the saved
  composite showed the blur spread, proving the processed image was used,
  not raw.
- [x] Added an optional **ROI-outline stamp** on diagnostic images (new
  `diagnostics.stamp_outlines()`, white 1px inner boundary) — draws exactly
  where each ROI selection landed directly onto the composite, including
  JPGs. New "Stamp ROI outline on image" checkbox on the Run tab.
- [x] Added an optional **particle label co-save**: one extra
  `<well>_particles_ch<N>.tif` per channel with particle analysis enabled,
  saving the exact integer-labeled particle mask from `context.masks`
  (TIFF-only, never JPG, since lossy compression would corrupt label
  values). New "Also save particle label images (TIFF)" checkbox, off by
  default given it's a heavier, more specialized output.
- [x] Added a **"Assign channel colors …" popup dialog** (`ChannelColorDialog`)
  on the Channels tab — lists every channel with its own color/swatch picker
  in one place, applying back to each `ChannelPanel`'s existing color combo,
  instead of having to open each channel's own panel to change its color.
- [x] Decoupled loading a pipeline JSON from having a sample image loaded
  first. Previously `_on_load_pipeline_json` refused outright ("load a
  sample image first") because channel count/names aren't in the JSON
  itself. Now derives the channel count from the highest channel/
  primary_channel/secondary_channel index referenced anywhere in the JSON's
  steps, and creates that many generic channel panels (ch0, ch1, …) on the
  spot if fewer (or none) exist yet — so clicking Load on the Setup tab's
  JSON dropdown populates Channels/ROI & Selections/Colocalization
  immediately, no image load required. If a sample was already loaded with
  real channel names and it already has enough channels, those real names
  are left alone (verified both paths explicitly).
- [x] Extracted diagnostic-image compositing helpers (`composite_rgb`,
  `save_diagnostic_image`, `bbox_from_mask`, `auto_contrast_limits`) out of
  `gui.py` into a new `correlative_imaging/diagnostics.py` with zero Qt/napari
  import at module scope, so `batch.py` (and the headless `ci batch` CLI) can
  use them without pulling in GUI dependencies. Fixed a circular-import bug
  this surfaced (`diagnostics.py` → `viewer.napari_viewer` → `viewer/__init__`
  → `gui.py` → `diagnostics.py`) by making `diagnostics.py` the canonical home
  for `auto_contrast_limits`, with `napari_viewer.py` re-exporting it.
- Investigated a reported "generated ROI is in a totally different place"
  bug: renaming the BF-pipeline class made it work again, which points to a
  stale `.roi` file on disk being reused (predating the circularity-scoring
  fix), not a coordinate/placement bug — consistent with what **#19** already
  suspected. Save/load round-trip re-verified empirically (IoU ~0.95, off by
  <1px from contour tracing) — no bug found there. Nothing further to fix
  unless it recurs after a fresh run.

## Setup / Plate

- [x] **[#9]** Output folder auto-updates to sit inside the new folder
  whenever a new input/plate folder is chosen.
- [ ] **[#10]** (later, not now) Support a folder containing several plates.
- [x] **[#11]** DB + pipeline JSON named by experiment name + run-start
  date/time, so runs stop overwriting each other. Pipeline JSON auto-saves
  every run. Run log also written to a file in the output folder.
- [x] **[#12]** Load a previously-saved pipeline JSON to auto-populate
  Channels / ROI & Selections / Colocalization / Pipeline Summary. Dropdown
  of JSONs found in the output folder, now living on the Setup tab.
- [x] **[#13]** One button to load BF + FL + ROI matching together.
- [x] **[#14]** When the plate scan already matches an existing ROI file to a
  well, auto-add the corresponding "existing project ROI" selection in ROI
  & Selections (removable, but present by default).
- [x] **[#15]** Diagnostic image export: whole image + one crop per ROI
  selection, auto-saved every run; user picks which to save, per-channel
  color, format (TIFF/JPG/both). Wired into `WellBatchRunner._process_well`
  via a new headless-safe `diagnostics.py` module — verified end-to-end
  (synthetic wells, real `run_wells()`, both sequential and 4-worker parallel
  paths) that whole/crop files land on disk with correct size and per-channel
  color compositing, and that crops only ever come from actual ROI selections
  (`LoadROI`/`ExtractROI` steps), never from segmentation/particle masks that
  also live in `context.masks`.

## BF Pipeline

- [x] **[#16]** Generalized from hardcoded hole/background to N user-named
  classes, each scored/gated independently (area/max_area × circularity,
  matching the original Fiji algorithm). "Import to ROI Selections" button
  auto-creates one selection per class.

## Channels

- [x] **[#17]** Let the user choose which particle-level metrics (area,
  perimeter, eccentricity, solidity, intensity stats, centroid, circularity,
  etc.) and which gross/bulk metrics get measured and saved. Added
  `metrics: list[str] | None` to `ParticleAnalysis`/`IntensityMeasurement`
  (`None` = all, backward-compatible with old saved JSONs); Channels tab now
  has a compact checkable list per group ("Save metrics:" under both the
  particle-filters box and a new "D. Gross/bulk metrics" box). DB layer now
  writes SQL `NULL` (not `0`) for a metric that wasn't selected, so "not
  measured" stays distinguishable from "measured and exactly zero" — matters
  for the trustworthiness of saved results.
- [x] **[#18]** Selection area (px + µm²) is now its own entry in the
  gross/bulk metrics checklist, independent of mean/sum/std — checking only
  `area_px`/`area_um2` still emits an `IntensityMeasurement` step (with only
  those two columns saved) even with all intensity-stat checkboxes off.
  Verified end-to-end: default all-checked, area-only emission, all-unchecked
  → no step at all, JSON round-trip (both with and without an explicit
  `metrics` key for backward compatibility).

## ROI & Selections

- [x] **[#19]** Verify/fix ROI-to-image dimension matching — scale-safety
  sidecar (`<roi>.json` recording source pixel size) added; `.roi` polygon
  loading now rescales like the `.tif` mask loader already did.
- [x] **[#20]** Dropped the untested `.zip` ROI-import option.

## Colocalization (was "Combine")

- [x] **[#21]** Renamed tab to "Colocalization". Defaults to every pairwise
  combination of the configured channels automatically.

## Pipeline Summary (was "Advanced")

- [x] **[#22]** Renamed tab to "Pipeline Summary". Shows the exact pipeline
  that will actually run — for single-image preview and for one image in
  batch — including the BF-pipeline/Ilastik step and every configured ROI,
  organized into readable sections.

## Run

- [x] **[#8]** Preview ("Test on one image") now resolves an actual well and
  runs the exact same pipeline batch would run for it, including always
  re-running Ilastik rather than reusing stale files (a passing preview now
  actually proves the batch pipeline works).
- [x] **[#23]** Default preview display: shows the last not-thresholded
  per-channel image (e.g. post-Normalize if that's the last preprocessing
  step), in that channel's user-specified color from the Channels tab. Works
  because `AutoThreshold`/`WatershedSplit` never set `result.image` (only
  `result.masks`) — so any per-channel image layer added during preview is
  definitionally pre-threshold; the last one added per channel is the one
  made visible, with its color overridden from the rotating step-color to the
  user's assigned channel color. Raw/unprocessed image is always loaded too,
  same color, hidden by default — except for a channel with *no*
  preprocessing steps configured at all, where raw becomes the (only
  sensible) default-visible layer. Verified with a fake `napari.qt.threading`
  harness (real napari isn't installed in this dev environment) driving the
  actual preview code path synchronously — confirmed hidden/visible states
  and color assignment for both a channel with preprocessing and one without.
- [x] **[#24]** Per-image step-level progress indicator during batch: a
  small thin progress bar + label under the well-level one, showing
  "well_id: step N/M — step_name" as each pipeline step finishes, using
  `Pipeline.run()`'s existing `on_step` hook. Threaded a new `on_step_fn`
  callback through `WellBatchRunner.run_wells()`/`_process_well()` down to
  `pl.run(..., on_step=...)`, and a new `_WellBatchWorker.step_progress` Qt
  signal to `RunTab`. Verified end-to-end for both sequential and 3-worker
  parallel batch paths (3 wells × 3 steps → 9 correctly-ordered events per
  well). Known limitation: under `max_workers > 1` several wells' step
  updates interleave on the one shared bar/label, so it can jump between
  wells rather than tracking one at a time — acceptable since parallel batch
  is already labeled experimental elsewhere in the code.
- [x] **[#25]** Parallel batch: each worker thread now gets its own
  sub-database (`<name>.partN.db`, one per thread, wells split round-robin)
  instead of funneling all writes through one serialized connection —
  compute *and* write happen together on the same thread against that
  thread's own file, so SQLite is never touched by more than one thread at a
  time. New `ResultsDB.merge_from()` copies every sub-DB into the final
  `results.db` afterward, remapping `image_id` foreign keys through an
  old→new id map (each sub-DB has its own independent AUTOINCREMENT
  sequence). Sub-DBs are kept on disk, not deleted. Verified with the
  strongest check available — same synthetic wells run through
  `max_workers=1` and `max_workers=3`, normalized and compared for exact
  content equality — plus a targeted FK-remap check that follows a merged
  row's `image_id` back into `images` and confirms it resolves to the
  *correct* well (via an exact, dilution-free seeded pixel value), not just
  a non-dangling one. Also verified: aborting mid-run still merges whatever
  partial per-worker results exist.

## Later — not urgent

- [ ] **[#26]** A final "Summary" tab with per-channel histograms and a
  channel-identification/dominant-color-per-well strategy, similar to
  `notebooks/plate_results_analysis.ipynb` but built into the GUI. Explicitly
  lower priority than everything above.
