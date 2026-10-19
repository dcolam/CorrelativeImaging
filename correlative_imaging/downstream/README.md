# Downstream analyzer (prototype, 2026-10-16)

Which cell sits in the patch-clamp hole, and what colour is it? This folder runs on the **output** of a finished
CorrelativeImaging run: the per-plate `.db`, the hole crops in `diagnostics/`, the stored thresholds. It never
writes into the run.

It is self-contained:
-   It doesn't import `results/`. The two helpers it needs are copied in `_discovery.py`.
-   `pyproject.toml`, `cli.py` and `batch.py` are untouched.
-   It runs from the repo root without installing:

```
python -m correlative_imaging.downstream run --root <output folder> --run-tag <run_YYYYMMDD_HHMMSS> --out <dir> \
       [--col-data plate_map.csv] [--config my.json] [--wells 18T39412_N11 ...]
python -m correlative_imaging.downstream fit-models --labels particle_labels.csv --out models/<name>.json
```

Open `<dir>/viewer/index.html` in a browser (works from file://).

## GUI

Double-click `start_gui.bat` (conda env `corrimg`), or run `python -m correlative_imaging.downstream gui` from the
repo root. The embedded viewer needs PyQt6-WebEngine, installed in `corrimg` on 2026-10-16 together with
scikit-learn. There are four tabs:

1.  **Run.** Pick the CorrelativeImaging output folder; its runs are found automatically. Then pick:
    -   the run;
    -   an optional plate map (CSV with `plate_id`, `well_id` + any columns; shown and evaluated, never used for
        a call);
    -   the results folder (default `<output>/downstream_<run>`);
    -   optional wells.

    "Run the whole routine" shows progress and a log. "Open existing results ..." reloads an earlier run.
2.  **Settings.** Every setting of `config.py`, grouped, with an explanation on hover. Presets per dataset are
    JSON files in `presets/` (`romk.json` = the defaults).
3.  **Review.** The particle viewer, embedded. Verdicts (keys 1–5) and particle labels go to
    `<results>/review.db`.
    -   **Label mode** (button or `L`): click a particle in any panel and pick one of the six carousel labels
        (keys 1–6 for the first particle listed). Labelled particles carry a yellow tag.
    -   Blind mode hides the plate map and shuffles the wells.
4.  **Results.** Tier × call summaries, the evaluation against the plate map (if it has Condition/Induction),
    the review counts, and "Export calls.csv". "Fit models" refits P(cell)/P(neighbour) from the labels of this
    run plus earlier label files (e.g. `neighbour_filter/labels_geom.csv`). It shows the cross-validated AUC,
    saves `models/<name>.json` and puts it into Settings; then rerun.

Tested headless in `corrimg`:
-   all three runs under `Z:\...\romk\output_v5` were found;
-   two wells of `run_newbf_20261015_133155` ran end to end through the Run tab;
-   in the embedded viewer, a verdict and a label from a simulated click were saved to review.db.

Runs made before the pipeline stored the hole crops (no `selections` table, e.g. the 2026-10-13 runs) are refused
with an explanation.

## What it does, in order (one module each)

| step | module | what |
|---|---|---|
| 1 | `sources.py` | load a well: fluorescence + BF planes of the hole crop, the hole (ImageJ .roi rasterised like `segment.py`), both thresholds, backgrounds. `RunSource` = a real run, `DevV5Source` = local ROMK v5 copies |
| 2 | `particles.py` | particles of both thresholds in the crop (stringent = whole-image, lenient = hole threshold), measured inside the hole and in full |
| 3 | `score.py` + `models.py` | SNR vs the whole-image background, P(cell), P(neighbour) (logistic, hand labels), roles: neighbour → bleed → signal |
| 4 | `cells.py` | particles → cells (overlap or ≤ 2 µm), colour-blind tier, simple 3-fold call |
| 5 | `refine.py` | seeded, competing growth: the patched cell grows from its seed, neighbours from theirs; judged against the hole's own background, ≤ 1.5 µm beyond the hole |
| 6 | `calibrate.py` | per plate × channel reference from field cells, bleed coefficient from field cells in the crop, colour of each refined cell in "typical cell" units, P(colour) from a 2-component mixture over all cells of the run |
| 7 | `viewer.py` + `www/` | browser viewer (ephacRTools `diagviewer.js`, copied unchanged), with overlays: particles by role/threshold, neighbours greyed, refined outline coloured by the calibrated call, BF plane slider (raw min–max), blind mode, verdicts + CSV export |
| – | `evaluate.py` | the only module that reads the experiment table (col_data). Evaluation and display only |

Outputs in `--out`:
-   `particles.csv`, `cells.csv`;
-   `calls.csv`: one row per well, keyed by `well_plate` = `B05.18T39383`, ready for `ephacRTools` merging;
-   `calibration.json`, `config.json`, `viewer/`, `evaluation.txt`.

All dataset-specific choices are in `config.py` (`Config`, JSON-overridable). These include channels, reporters
vs markers, bleed pairs, every cut, and the model file. The default is the ROMK setting.

## Ground rules built in

-   Nothing that decides a call sees col_data (plate map, ephys).
-   No cut is fitted on Empty or pure wells. Calibration uses field cells and all holes.
-   No watershed splitting. Refinement only assigns dim pixels between particles that were already classified.
-   Detection and tier come from the seed (the models were trained on seeds). Colour comes from the grown body,
    with every channel measured on the same footprint.

## Validation so far (ROMK v5, dev source; the share was offline)

-   **Scoring → roles → cells → tier**, on the reference particles (`workflow_comparison/cache/objects.pkl`):
    identical to `tiers.csv` in **1232/1232** wells (tier and call). The models equal the reference to machine
    precision.
-   **Extraction from pixels** (viewer crops): only 639/1232 crops hold the v5 image. The other 593 hold the
    pixels of `run_newbf_20261015_133155` (z 10). That run overwrote the diagnostics before the viewer export,
    confirmed: thresholded with that run's own thresholds, they match its particles to a few pixels. The list of
    good wells is in `scratch_analysis/downstream_v5/wells_crop_matches_v5.csv`.
    -   On the 639 good wells: tier 627/639 and call 638/639 equal to `tiers.csv`.
    -   The differences come from the smaller viewer window (±2.2 hole radii), which cuts big neighbours.
-   **Refinement:** median growth ×1.2, 93 % of the body inside the hole. The settings (hole background,
    1.5 µm margin, relative cut 0.35) were set by eye on ~9 wells, not tuned on the ephys.
-   **Calibrated call vs simple call,** 639 good wells, tiers high/middle.
    -   **Hyperpolarized Mixed wells (42):** the simple call has 10 wrong red calls. The calibrated call fixes 1,
        abstains on 2 ("dim") and keeps 7. It also abstains on 3 of the 32 right green calls.
    -   **Pure recorded wells:** it abstains on 1 wrong and 3 right calls, and makes no new error.
    -   **So it is not a fix for the main error.** It mostly adds abstentions ("dim" / "uncertain"). A fair
        comparison has to be made at equal coverage.
    -   Note: the calibrated well call uses the best cell only; the simple call sums over all candidate cells.
-   **Bleed coefficient:** measured from field cells, it is ≈ 0 (< 0.1 %) after background subtraction. The old
    "2–3 %" was mostly background in the raw ratio.

## Not done / next

1.  **`RunSource` smoke-tested (2026-10-16)** on 6 wells of `run_newbf_20261015_133155`. Its in-hole particle
    areas equal the reference crop extraction (`extract2.py`) exactly in every channel. Still to do:
    -   the full run;
    -   validation on a z ≥ 12 rerun with crops (`--reference tiers.csv`);
    -   ROIs crossing the frame edge (segment.py clips to the frame first; here the polygon is cut at the crop).
2.  **Review loop:** a particle-label mode in the viewer (click a particle → label), exported to CSV →
    `fit-models`. Hold out a labelled set that is never trained on.
3.  **Refinement artefacts seen:** straight-edged bodies where the signal is flat (18T39383 M13); rim halo when
    the margin is large (3 µm took it in, 1.5 µm mostly not).
4.  **BF:** focus-plane choice and a cell-body mask over the hole, as an outer limit for the growth. This is the
    route to the remaining error: a red cell over a green patched cell.
5.  **Generalisation:**
    -   `call_mode = "multilabel"` for PNC/iPSC;
    -   the calibrated call for more than 2 reporters (placeholder now);
    -   writing the tables into the run DB / an R reader (see `HANDOVER_identity_stage.md`).
