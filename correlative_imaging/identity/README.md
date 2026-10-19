# Particle identity (rebuild of `downstream/`, 2026-10-19)

What is in the hole, and what is it positive for? The tool runs on the **output** of a finished CorrelativeImaging run
(per-plate `.db`, `diagnostics/` TIFs, hole ROIs). It never writes into the run.

It is a particle classifier first:
1. every particle at the hole is described by features that mean the same thing for 1, 2, 3 or 4 channels;
2. a classifier trained on hand labels (and pretrained on all particles) says cell / neighbour / merged / debris /
   noise;
3. cells are rebuilt from the cell particles, neighbour-aware: they split where the signal dips and never fuse with
   a neighbour;
4. each rebuilt cell gets a calibrated call over all channel combinations.

`downstream/` (the validated ROMK prototype) and `results/` are untouched. `downstream/` stays as the reference.

## Start

```
start_gui.bat                                     (double-click; conda env corrimg)
python -m correlative_imaging.identity gui [--results <folder>]
```

The GUI is napari with two docks:
- **Wells** (left): search, a filter (one cell / several / no cell / uncertain / labelled / test wells) and the well
  table. Click a well to show it. `n` and `p` step through the list.
- **Particle identity** (right), with four tabs:
  - **Run:** pick the output folder and run, a preset and the results folder. Then **Detect** (needs no model),
    **Classify**, or both. Also "Open existing results…", progress and a log.
  - **Settings:** every setting, grouped, with explanations on hover; channel display names and colours; presets.
  - **Train:** the label library, label mode, training with the comparison tables, and "Label next".
  - **Results:** counts, calls per plate, per-channel mixture quality, and the plate map (evaluation only).

In the canvas, each channel is its own napari layer (additive, run colours). The other layers:
- **BF:** all planes, raw min–max, hidden by default;
- **hole:** the hole outline;
- **particles (lenient / stringent):** outlines coloured by predicted class (cell yellow, neighbour grey, merged
  orange, debris violet, noise dark), with hand labels as tags;
- **cells:** the reconstructed bodies. Cells are coloured by their call (blended channel colours, amber =
  uncertain); neighbours are grey.

Layer visibility and contrast are kept from one well to the next.

**Labelling:** turn on label mode (`L`), click a particle and press `1`–`5` (cell, neighbour, merged, debris, noise).
`Del` removes a label. Where several particles overlap (channels, two thresholds), a menu lists them.

## Command line

```
python -m correlative_imaging.identity detect   --root <output folder> --run-tag <tag> --out <dir> [--config ipsc]
python -m correlative_imaging.identity classify --out <dir> [--model <file.joblib>]
python -m correlative_imaging.identity train    --results <dir> [<dir> …] --labels ci_labels.db --model <name>
python -m correlative_imaging.identity import-labels --csv labels_geom.csv --labels ci_labels.db --dataset romk_v5 \
       --names GFP=488nm mScarlet=561nm ROMK=640nm [--wells-csv wells_crop_matches_v5.csv]
```

`--config` takes a JSON file or a preset name (`presets/romk.json`, `presets/ipsc.json`). The neutral default
treats every fluorescence channel of the run as an identity channel.

## Pipeline (one module each)

| step | module | what |
|---|---|---|
| 1 | `sources.py` | load a well into `WellData` (window ±`work_R` = 2.2 hole radii). Prefers `<well>_whole_channels.tif` (the last pre-threshold image of the frame), else the hole crop with its `selections` origin. Channels are matched by the TIF labels, never by position. Thresholds and background come from the DB when stored; otherwise they're computed as the pipeline does (triangle; mean/SD outside particles). The hole is the .roi rasterised like `segment.py`. `JsonpCropSource` reads viewer-export crops (used to import the first ROMK labels). |
| 2 | `detect.py` | particles at two thresholds (stringent whole-image; lenient hole threshold applied to the window). Measured in full and inside the hole, every channel over the same footprint; peaks; parent object; the hole's own background. |
| 3 | `reference.py` | field reference per plate × channel (whole-image objects ≥ 30 µm²): typical brightness (dynamic-range unit), size, shape. |
| 4 | `features.py` | 27 features that don't depend on the number of channels (own signal in calibrated units, shape, position, context, summaries of the other channels; see the module doc). The channel itself is never a feature. |
| 5 | `labels.py` | label library (`ci_labels.db`): labels stored by position, so they survive reruns; frozen test wells; legacy import. |
| 6 | `learn.py` | classifier (gradient-boosted trees), pretraining on all particles (label spreading, self-training), grouped CV + frozen test, "label next" queue. |
| 7 | `reconstruct.py` | neighbour-aware cells: a basin per significant peak, split at dips (saddle/lower peak < 0.6), cell and neighbour basins never joined, growth relative to the seed within 1.5 µm of the hole. |
| 8 | `identify.py` | n_c in typical-cell units over the body (hole background subtracted). Per-channel mixture over all cells in the holes of the run; P over all 2^K channel combinations; "uncertain" below 0.8. |
| – | `pipeline.py` | `detect_run` (stage 1, no model) and `classify_run` (stage 2, from the per-well cache). |
| – | `gui/` | napari docks (`app.py`, `panel.py`, `display.py`), theme from napari. |

Outputs (results folder):
- `run.json`, `wells.csv`, `reference.csv`;
- `particles.parquet` (features + P(class) + cell membership);
- `cells.csv`, `calls.csv` (one row per well, key `well_plate` = `B05.18T39383`), `identity.json`;
- `cache/` (window pixels float16, hole, BF, label maps; about 0.25 MB per well) and `recon/`.

## Ground rules built in

- **No plate map or ephys in a call.** The plate map is read only in the Results tab (evaluation and display).
- **No cut fitted on control wells.**
  - The identity reference is the population of all cells in the holes.
  - The classifier learns from hand labels.
  - Weak rules exist only as one arm of the comparison.
- **Flagged particles are kept.** Every particle stays in `particles.parquet` with its P(class).
- **Wells are matched by name and padded (`B05`); channels are matched by name.**
- **Splitting at dips replaces "watershed stays off".** This is on request, and it's switchable (`split_dips`).
  Splitting needs a real dip between two significant peaks (synthetic test: a noisy plateau is not cut up).
- **The training numbers are honest.**
  - Model choice uses CV grouped by well.
  - A frozen set of test wells per dataset is never trained on, never pseudo-labelled and never in the spreading
    graph.
  - Coverage is reported next to accuracy.

## Pretraining on all particles

Labels are few; particles are many. The Train tab compares, on the same folds:

| arm | what |
|---|---|
| legacy | logistic regression on the reference workflow's 4 features |
| hand | gradient-boosted trees, all features, hand labels only |
| spread | hand labels spread over a kNN graph of all particles (PCA embedding), then a kNN classifier |
| self-train | hand labels + confident pseudo-labels (trees P ≥ 0.9 **and** agreeing with label spreading), weight 0.3, 3 rounds |
| self-train+rules | the same, plus a few colour-blind weak rules (no signal → noise; mostly outside and field-cell-like → neighbour; small and bright in every channel → debris) |

**"Label next"** lists the unlabelled particles the model is least sure about, one per well (test wells excluded).
Labelling those is the fastest way to grow the set where it matters.

## Validation (2026-10-19)

Results, labels and source spec are in `correlative_imaging_reports/.../scratch_analysis/identity_v5/` (ROMK) and
`output2/identity_iPSC-n1_newrun_20260716_115952/` (iPSC). The run to regenerate ROMK is
`detect --spec identity_v5/source_romk_v5_crops.json`, then `import-labels`, `train` and `classify` (see Command line).

The analysis window is ±2.2 hole radii everywhere (`work_R`). That is the size of the ROMK viewer crops the first
labels were made on, so whole-object features (full area, edge, parent share) mean the same thing across datasets.
If the model is retrained on full-frame sources, a larger window can be used again.

### ROMK v5, 639 wells whose viewer crops hold the v5 pixels (the share was offline)

- **Legacy labels:** 862 imported in those wells; 837 matched to a new particle (97 %). After the label-name
  mapping: cell 388, noise 339, neighbour 81, merged 20, debris 9. That makes 901 training rows and 248 test rows
  (50 frozen test wells).
- **Classifier:**

  | arm | CV macro-F1 | CV AUC(cell) | test macro-F1 | test AUC(cell) | test recall cell | coverage |
  |---|---|---|---|---|---|---|
  | legacy (4 features) | 0.53 | 0.88 | 0.46 | 0.78 | 0.56 | 0.15 |
  | hand | 0.60 | 0.93 | 0.54 | **0.91** | 0.81 | 0.84 |
  | spread | 0.61 | 0.91 | 0.56 | 0.89 | 0.89 | 0.77 |
  | self-train | **0.64** | 0.93 | **0.57** | 0.90 | 0.81 | 0.92 |
  | self-train+rules | 0.62 | 0.93 | 0.53 | 0.90 | 0.82 | 0.92 |

  - The new features are a clear gain over the legacy set (test AUC for cell 0.90 vs 0.78).
  - Pretraining on all particles raises coverage (0.84 → 0.92) and the CV score.
  - On the frozen test, the order of the arms changed between two runs that differed only slightly (a run before the
    window/background fix had hand 0.59 vs self-train 0.54). With 248 test particles the arms are within noise of
    each other.
  - Debris (7 training labels) is not learnable yet, and merged only barely.
  - The legacy arm is the old feature set, not the old model (which had channel dummies and was binary).
  - Shipped model: `models/romk_v5_selftrain_20261019.joblib` (self-train, chosen by CV as the protocol says), set
    in `presets/romk.json`.
- **Calls against the reference workflow, at equal coverage** (plate map and ephys used for evaluation only):

  | | reference workflow | this tool |
  |---|---|---|
  | Hyperpolarized Mixed (79 wells, all GFP by ephys): single-colour calls | 46 (58 %) | 33 (42 %) |
  | … of which red (= wrong) | 11 (24 %) | 3 (9 %) |
  | … wells where both tools give a single colour (n = 29): red | 5 | 3 |
  | … the reference's confident tiers (high/middle, 41 wells): red | 9 / 41 | 2 / 25 called |
  | pure recorded wells (110): any colour call | 55 % | 58 % |
  | … right colour / wrong colour / both | 58 / 2 / 1 | 62 / 0 / 2 |
  | Empty wells (82) coloured | 8 | 3 |

  - On the pure wells, at the same coverage, the tool is at least as good and makes no cross-colour error.
  - In the Mixed wells it calls fewer wells. On the wells both tools call, the gain is small (5 → 3 red of 29), so
    part of the lower error rate is abstention. The remaining Mixed problem (6 "both", 3 red) is a red cell stacked
    over the green patched cell: there is no spatial dip to split on, so this needs BF.
  - **18T39412 N11 (a regression well in the handover) changed** from GFP (reference) to "uncertain" (P = 0.76 for
    GFP). The red neighbour's signal reaches into the body.
  - Subtracting the hole background under the body, instead of the frame background, halves the "both" calls
    (`identity_bg`, default "hole"). Wells whose hole is filled by a cell (20 % of ROMK, 8 % of iPSC) take the
    plate's median hole background.

### iPSC n1 (`output2`, 4 plates, 3 reporters, no labels yet)

- **Detection:** 1198 wells in about 4 min. Thresholds and backgrounds were not stored in this older run; they were
  recomputed and reproduce the DB field particles exactly (area and mean ratio 1.000).
- **ROMK model applied unchanged:** it calls Dlx-GFP speckle in empty holes "cell" (e.g. 18T39263 O13 → several
  tiny cells). In holes filled with aperture glow (e.g. 18T39263 C08, 18T39607 O08), pieces of the glow become
  "cells".
  - 70 of 1040 wells with cells have ≥ 2 cells, one of them > 50 µm².
  - In the ones inspected, the extra cells are glow fragments separated by gaps below the growth floor. They are
    not a real cell cut at a dip.

  This is a classification problem (iPSC needs its own noise/cell labels), not a splitting problem. Over-splitting
  of real cells is only covered by the synthetic test so far.
- **Identity:** per channel the hole cells don't separate into negatives and positives (Ashman's D 1.9–2.3; 23–38 %
  of cells with P between 0.2 and 0.8). The combination call is therefore mostly "uncertain" (745 / 1198), and
  195 are triple-positive. Part of this is the ROMK model accepting glow and speckle as cells. Whether the rest is
  biology (co-expression), aperture glow or bodies grown too large can only be decided with iPSC labels. The earlier
  Results Explorer scores showed the same pattern (707 / 1198 triple-positive holes).
- **Training on ROMK labels + the unlabelled iPSC pool** works (7872 pseudo-labels, coverage 0.95). Without iPSC
  labels it cannot show any accuracy on iPSC, and it did not raise the ROMK test score.

Tests: `tests/test_identity_reconstruct.py` covers dip splitting, no split without a dip, no over-splitting of a
noisy plateau, a dim cell stopping at the saddle, and debris as a barrier. `tests/test_identity_features.py` covers
identical features for 1–4 channels and labels matching their particle. All pass. pytest is not installed in
`corrimg`; they were run with a minimal runner.

## Not done / next

1. **Label iPSC.** About 300 particles, starting with "Label next", then retrain on ROMK + iPSC. The cross-dataset
   number (ROMK → iPSC) needs these labels; the frozen iPSC test wells are drawn on first training.
2. **Debris and merged need more labels:** 7 and 27 now.
3. **Validate on the full ROMK v5 run once the share is online.** `RunSource` reads `whole_channels.tif` if v5 wrote
   it (all 1232 wells, all 1796 labels, full context instead of ±2.2 R crops).
4. **Brightfield** (no Cellpose yet): BF planes are in the cache and the viewer. A BF body mask as an outer limit for
   growth is the route to the remaining red-over-green error.
5. **Write `calls.csv` / cells into the run DB and an R reader** (`HANDOVER_identity_stage.md` §3).
