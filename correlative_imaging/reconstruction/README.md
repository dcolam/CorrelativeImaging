# Cell reconstruction (rebuild of `downstream/`, 2026-10-19; first called "identity")

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
ci reconstruct [<results folder>]                 (or: start_gui.bat; ci identity still works)
```

From `ci gui`, use the Run tab button **"Cell reconstruction …"**, or the bar that appears when a run finishes. Both
open the tool in its own window. If the run already has a `reconstruction_<run>` (or older `identity_<run>`) folder,
it opens that. Otherwise the Run tab is filled in for that run, ready for Detect.

The GUI is napari with two docks.

**Wells (bottom)** has three tabs:
- **Plate:** a plate view (96 / 384 / 1536, taken from the well ids). Wells are coloured by status, dominant channel
  or P. A white frame means the well is labelled; a dashed amber frame means it is a test well.
- **List:** a searchable, filterable list of wells.
- **Particles (PCA):** a PCA of the 27 features of every particle in the holes (what the classifier sees). Colour by
  preview class, model class, hand label, channel, plate or threshold level. Hover for details, click to open the
  well with that particle picked. The current well's particles are ringed.

Before Classify the colours are the Detect preview; afterwards they are the calls. The dock says which. `n` and `p`
step through the list.

**Cell reconstruction (right)** has four tabs:
1. **Run:**
   - pick the output folder and run (a preset whose channels match is suggested) and the results folder;
   - set each channel's **name and colour** (display only) and **role** (identity or marker, or untick it to ignore
     it). Names and colours from the Results Explorer are used when present;
   - pick the model;
   - then **Detect** (needs no model), **Classify**, or both.
2. **Train:**
   - the label library and label mode;
   - **Random well** (unlabelled, not a test well) and **Next well**;
   - which outlines to show (stringent, lenient, cells around the hole, noise; noise outlines are hidden by default
     because background speckle at the lenient threshold covers the whole window);
   - training with the comparison tables, and "Label next".
3. **Results:** counts, calls per plate, per-channel mixture quality, and the plate map (evaluation only).
4. **Settings** (last; the defaults work):
   - **recommendations** from the detected run, applied only on click. Example: a channel with signal in nearly every
     hole that holds anything is suggested as a marker;
   - **safe settings** (reconstruction and call; change them, then Classify again);
   - **detection settings**, which define the features. They are locked, shown next to the model's values (✓/✗),
     and can be unlocked.

**Canvas layers:**
- **BF:** all planes, raw min–max, hidden by default.
- **each channel:** its image (additive, its colour) plus an outline layer in the same colour. Stringent particles
  are solid, lenient (hole-threshold) particles dashed. The outlines show and hide with their channel.
- **hole:** the hole outline.
- **cells:** the reconstructed bodies (after Classify). Cells are coloured by their call (blended channel colours,
  amber = uncertain); neighbours are grey.
- **tags:** one per particle in the hole. The model's class: C cell, N neighbour, M merged, D debris, · noise.
  Before Classify, the provisional class instead: in, N?, D?, ·?. A hand label is white in brackets, e.g. [C].

The layers are created once and only their data changes from one well to the next (about 0.4 s per well switch, was
4.5 s), so visibility stays as you set it. Each channel is auto-contrasted (napari's "once") when another well
opens; re-detecting the same well after an edit keeps your contrast.

**Labelling:** turn on label mode (`L`), click a particle and press `1`–`5` (cell, neighbour, merged, debris, noise).
`Del` removes a label. Where several particles overlap (channels, two thresholds), a menu lists them.

## Splitting and editing particles

**Proposed splits.** Detect finds, for every particle in the hole (up to 4 hole areas), where it would split at a
real dip between two significant peaks. The split lines are drawn dashed orange, and `n_proposed` counts the pieces.
They are only proposals: applied to every particle (`split_particles`), the split cut about 20 % of the
hand-labelled cells on ROMK, while catching only 50–70 % of the "merged" particles. No peak height or dip depth
separated the two better (table below). So splitting is off by default, and you accept proposals per particle.

| peak height (hole-noise SDs) | dip (saddle / lower peak) | cells split | merged split | neighbours split |
|---|---|---|---|---|
| 3 | 0.6 | 22 % | 52 % | 29 % |
| 3 | 0.45 | 11 % | 19 % | 14 % |
| 3 | 0.3 | 5 % | 3 % | 5 % |
| 12 | 0.6 | 20 % | 71 % | 29 % |

**Edits** (Train tab, stored by position in the label library):
- **Accept proposed split:** for the picked particle.
- **Watershed picked:** splits the picked particle in two on demand, with or without a proposal. It splits at its
  deepest dip between significant peaks, or, with a single peak, at the neck of its shape (distance-map watershed).
  Watershed a piece again to split it further. Pieces under 10 % of the particle join their neighbour. The dividing
  line is stored and applied like a drawn cut. Splitting at every dip instead gave 4–21 pieces on noisy ROMK
  particles, which is why it splits only once.
- **Draw cut** (`X`): drag a line across a particle, then Apply. The cut applies to every channel and both
  thresholds. It is drawn 3 px wide (so a diagonal line also separates), and the cut pixels go back to the nearest
  piece.
- **Merge with…:** joins the picked particle with another one. They must have the same channel and threshold and
  either touch or come from one object, so merge mainly undoes a split. Pieces of one cell that don't touch are
  joined by the reconstruction anyway.
- **Clear edits in this well.**

**Cuts hold in the reconstruction.** The lines of hand cuts, watersheds and applied splits are walls that no cell
body crosses. Seed pieces of one edited object are never joined into one cell, whether by the no-dip rule or the
`cell_gap_um` join. Earlier, the two pieces of a cut fused back into one cell. The wall pixels go back to the cell
beside them. A piece cut off outside the hole stays as a visible, clickable neighbour: it takes no label but can be
merged back. Before, it was dropped and vanished from the view. Whether each piece seeds a cell is still the model's
call: a piece classified neighbour or noise starts no cell. Wells without edits are unchanged (639/639 calls on ROMK).

**After each edit** that well is detected again from the cache, which is float32 so the result equals a full Detect.
If the run was classified, the well is also classified again; its identity uses the stored per-channel calibration,
so no other well's call changes (tested: 638/638 unchanged). The run lists the edited wells until the next full
Classify, and a full Detect applies all stored edits (`detect --labels`).

**Labels on split objects.** Particles keep `origin` (the object before edits) and `n_parts`. A legacy "merged"
label on an object that now splits is left out of training, and the training report counts these.

## Detect gives a first look (no model)

`preview.py` gives each particle in the hole a provisional class:
- **noise:** no clear signal against either background;
- **debris:** small and bright in every channel;
- **neighbour:** mostly outside, in the outer half of the hole, and its whole object is a bright or typical-looking
  cell next to the hole;
- **inside:** everything else.

Per well it gives a status (no particles / neighbours only / one object / several objects) and the **dominant
channel**: signal above the hole background, in typical-cell units, times area. When the top two channels are within
1.5×, it's shown as "a+b".

Measured on ROMK against the hand labels and the plate map:
- 85 % of hand-labelled particles get the matching provisional class (noise rule: SNR < 3 or < 5 % of a typical
  cell);
- 2 of 82 Empty wells get a colour;
- the dominant channel is right in 103 of the 109 pure wells that get one;
- the well status agrees with the full Classify in 82 % of wells.

The Results tab shows this distribution right after Detect: wells per status, dominant channel per plate, particles
per channel and class.

These are previews. They are kept apart from the model's classes and are **never** written to the label store.
Labelling a particle overrides its tag; the classifier learns only from hand labels.

## Which images are read

Detect works on the **whole hole crop** the pipeline wrote (`<well>_roi_hole_crop_channels.tif`, e.g. 511 × 513 px
on PNC) when three things hold:
- its position in the frame is known (`selections` table);
- it covers the analysis window;
- the database stores thresholds and backgrounds.

Otherwise (older runs: crops but no crop positions and no stored thresholds) it works on ±`view_R` (default 6) hole
radii of `<well>_whole_channels.tif`.

**The analysis window stays ±2.2 hole radii** (`work_R`) around the hole. Particles in the hole are measured clipped to
it, the context statistics count only the window's objects, and the reconstruction runs on it. So the model's
features do not depend on how much of the crop is detected: on ROMK and PNC every hole particle has identical features
and classes, and all 639 ROMK calls are unchanged.

**Every other cell of the crop is a neighbour by position** (`outer`). It is outlined (stringent threshold, with the
context toggle), can be clicked, cut, watershed or merged, but is not classified and takes no label: the classifier
only judges particles that reach into the hole.

**The cache holds no pixels** for wells read from a TIF of the run. It keeps the hole, the particle label maps,
thresholds, backgrounds and where the pixels are. Opening or re-detecting a well reads its area back from the crop
TIF (memory-mapped, bit-identical to Detect). On PNC that is ~16 KB per well instead of ~4.5 MB (19 wells: 0.3 MB vs
83 MB; about 50 MB for the whole run of 3,071 wells). Detect takes 16 s for 19 wells, well switching about 0.6 s and
an edit plus re-detect about 1.2 s over Z:. The TIFs must be reachable from the computer that opens the results
(Settings → path map if they sit under another drive letter or mount). Viewer-export crops (the ROMK set) have no TIF
to go back to and keep their pixels in the cache.

Results detected before keep their small window and show ±`view_R` read from the TIF around it; run Detect again to
get the whole crop.

On PNC the pixels are on the pipeline's 2× binned grid (0.427 µm; the hole is ~44 px across). That is the pipeline's
`Binning` step, not this tool. Only an unbinned run gives more pixels.

## Command line

```
python -m correlative_imaging.reconstruction detect   --root <output folder> --run-tag <tag> --out <dir> [--config romk]
python -m correlative_imaging.reconstruction classify --out <dir> [--model <file.joblib>]
python -m correlative_imaging.reconstruction train    --results <dir> [<dir> …] --labels ci_labels.db --model <name>
python -m correlative_imaging.reconstruction import-labels --csv labels_geom.csv --labels ci_labels.db --dataset romk_v5 \
       --names GFP=488nm mScarlet=561nm ROMK=640nm [--wells-csv wells_crop_matches_v5.csv]
```

`--config` takes a JSON file or a preset name (`presets/romk.json`, `presets/ipsc.json`). The neutral default
treats every fluorescence channel of the run as an identity channel.

## Pipeline (one module each)

| step | module | what |
|---|---|---|
| 1 | `sources.py` | load a well into `WellData` (window ±`work_R` = 2.2 hole radii). Uses the hole crop when its origin is known and it covers the window, else `<well>_whole_channels.tif` (see above). Channels are matched by the TIF labels, never by position. Thresholds and background come from the DB when stored; otherwise they're computed as the pipeline does (triangle; mean/SD outside particles). The hole is the .roi rasterised like `segment.py`. `JsonpCropSource` reads viewer-export crops (used to import the first ROMK labels). |
| 2 | `detect.py` | particles at two thresholds (stringent whole-image; lenient hole threshold applied to the window). Measured in full and inside the hole, every channel over the same footprint; peaks; parent object; the hole's own background. |
| 3 | `reference.py` | field reference per plate × channel (whole-image objects ≥ 30 µm²): typical brightness (dynamic-range unit), size, shape. |
| 4 | `features.py` | 27 features that don't depend on the number of channels (own signal in calibrated units, shape, position, context, summaries of the other channels; see the module doc). The channel itself is never a feature. |
| 5 | `labels.py` | label library (`ci_labels.db`): labels stored by position, so they survive reruns; frozen test wells; legacy import. |
| 6 | `learn.py` | classifier (gradient-boosted trees), pretraining on all particles (label spreading, self-training), grouped CV + frozen test, "label next" queue. |
| 7 | `reconstruct.py` | neighbour-aware cells: a basin per significant peak, split at dips (saddle/lower peak < 0.6), cell and neighbour basins never joined, growth relative to the seed within 1.5 µm of the hole. |
| – | `preview.py` | Detect-stage provisional classes, well status and dominant channel (no model). |
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

Results, labels and source spec are in `correlative_imaging_reports/.../scratch_analysis/identity_v5/` (ROMK). The run to regenerate ROMK is
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
    in `presets/romk.json`. Models are saved as plain sklearn objects + arrays (no package classes), so they survive
    renaming the package. Each model records the detection settings it was trained on (`feature_config`); Classify
    warns when a run was detected with other values.
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
    (`identity_bg`, default "hole"). Wells whose hole is filled by a cell (20 % of ROMK) take the
    plate's median hole background.

### Detect preview (no model), ROMK

- **Dominant channel:** in pure wells it matches the plate map (GFP wells: 76 of 84 coloured GFP; mScarlet wells:
  93 of 104 mScarlet).
- **Empty wells:** they still get a colour (47 of 82). Without a model the noise rule is weak. That is what Classify
  is for.
- **Provisional neighbour mark vs the hand labels:** it catches 107 of 130 labelled neighbours and wrongly marks 32
  of 620 labelled cells.

Tests: `tests/test_reconstruction_reconstruct.py` covers dip splitting, no split without a dip, no over-splitting of a
noisy plateau, a dim cell stopping at the saddle, and debris as a barrier. `tests/test_reconstruction_features.py` covers
identical features for 1–4 channels, labels matching their particle, and the Detect preview marking a neighbour. All
pass. pytest is not installed in
`corrimg`; they were run with a minimal runner.

## Not done / next

1. **Label a second dataset** (a current run, e.g. PNC or a new iPSC run). About 300 particles, starting with
   "Label next", then retrain on both. The cross-dataset number needs these labels; that dataset's frozen test wells
   are drawn on first training.
2. **Debris and merged need more labels:** 7 and 27 now.
3. **Validate on the full ROMK v5 run once the share is online.** `RunSource` reads `whole_channels.tif` if v5 wrote
   it (all 1232 wells, all 1796 labels, full context instead of ±2.2 R crops).
4. **Brightfield** (no Cellpose yet): BF planes are in the cache and the viewer. A BF body mask as an outer limit for
   growth is the route to the remaining red-over-green error.
5. **Write `calls.csv` / cells into the run DB and an R reader** (`HANDOVER_identity_stage.md` §3).
