# Unified cell classification

Classifies the cell in every hole, and every surrounding cell, by colour, for any dataset: any number of colour
channels, any markers, any microscope. Everything is learnt from the dataset itself; nothing is carried over from
another dataset. Each step writes what it did, the number it got, a diagnostic plot and example images into one
report. Where the data cannot decide, the report asks you for a small, specific input.

## Run it

**In the tool (recommended):** `ci reconstruct --v2 [<Detect results folder>]` (or start_gui_v2.bat, or
`python -m correlative_imaging.reconstruction gui --v2 [--results <folder>]`). `ci reconstruct` without `--v2` is the
old tool, unchanged.

One dock on the right: the workflow strip (Detect → First pass → Label → Learn → Report; the next step is framed)
and four tabs. The canvas shows whatever you open (Space or double-click on a thumbnail): one layer per channel in its
colour, the hole (yellow), the grown cell (cyan), the asked particle (white), or a surrounding cell on its whole image.

| tab | what you do there |
|---|---|
| First pass | Setup (channel names and roles, excluded / empty wells), **Run first pass**, then answer its **questions**: only where this dataset cannot decide on its own, each with the reason and how many answers it needs (wells with few cells; cell or not for hole particles; one cell? for the cell size; is a channel positive near its cut; one cell or several for large hole cells). Answers are saved at once; run again to apply them. "What it found": patterns, cuts, flags |
| Label | Your **biological labels** in your own words (+ new label), on holes and surrounding cells drawn at random from every channel pattern, interspersed; correct the positive channels where you see it differently. "More suggestions" adds items |
| Learn | **Learn from my labels**: the LDA, how often it agrees with you (each label predicted by a model that did not see it), the confusion table, pattern × label; "Label the unsure ones" sends uncertain cells back to Label |
| Report | the full report (also "Open in browser") |

Keys in the thumbnail grids: 1-9 answer, 0 clear, arrows move, Space open in the canvas, PgDn/PgUp page.

**Without the tool:** `python -m correlative_imaging.reconstruction unified --results <Detect results folder>`; the
answers and labels are then `interventions/answers.csv` and `interventions/labels.csv`.

Output goes to `<results>/unified/` (or `--out`). Nothing else is changed. The first run reads every hole image once
(15-40 min); later runs reuse the measurements (2-6 min); new "cell or not" answers measure the holes again.

## The steps

| step | what it does | learnt from |
|---|---|---|
| 1 Well checks | flags wells with far fewer surrounding cells than their plate (lost focus? no cells seeded?), file names naming another plate than their folder, images without a hole | the plate itself |
| 2 Surrounding cells | clean example cells, bleed-through per channel pair, the size of one cell body, where "positive" starts in each channel, the channel positivity of every surrounding cell, shares per well | the surrounding cells of this dataset |
| 3 Hole cells | per particle: cell or not by shape (once you gave "no cell" examples); grows the cell from the cell particles (neighbours and plugs are barriers); three measurements per channel | the hole images, Detect's particles |
| 4 First pass | **channel positivity only**, named as in the run (`FITC+ TX RED+`, `none`, `FITC? (dim)`): per channel the measurement whose hole cells split most clearly, and its cut; flags. No cell types, no priority between channels | the hole cells of this dataset |
| 5 Your labels | the label sheet (random holes and surrounding cells from every pattern, interspersed); from what you write: the dim cut per channel, the shape model, and an LDA that learns your labels from the channel signals and labels every cell with a probability | your labels |

## Your inputs (only where needed)

- **interventions/labels.html + labels.csv** (the main input): numbered holes and surrounding cells, each with its
  composite and one grey image per channel. Per number write `your_label` (what the cell is, in your words, e.g.
  astrocyte, interneuron; or `not a cell` for halo, plug, debris, glow) and, only where you disagree with the first
  pass or it says dim, `positive_channels` (the channels you see positive, or `none`). About 10 per label. The sheet
  is never overwritten once it has an entry; add rows yourself for more holes.
- **interventions/wells_to_check.csv**: wells with few cells; write `exclude`, `empty` or `keep`.
- **interventions/cell_size.csv** (optional): mark which numbered surrounding cells in `cell_size_sheet.png` are
  exactly one cell, if the learnt cell size looks wrong.
- **experiment.json** (written on the first run): channel names and display colours (from the run), which channel is
  a `marker` (measured without bleed correction), `exclude_wells` / `empty_wells` (`"*:col 6"`,
  `"18T40668:rows C-O cols 1-8"`, `"18T39383:B04"`; excluded = imaging failure, empty = no cells seeded, a reference
  for "no cell"). Optional `marker_negative` ({"640nm": "561nm+"}): the marker's "no" reference is the cells with that
  pattern.

## Outputs

- `report.html`: everything, readable top to bottom.
- `hole_calls.csv`: one row per hole: channel positivity (`pattern`), your label or the predicted one (`bio_label`, `bio_p`), flags, the signals behind the call.
- `field_cells.parquet`, `field_cells_per_well.csv`: every surrounding cell and the counts per well.
- `checks.csv`, `summary.json`, `measurements.parquet`, `wells/` (crops for the example images).

## Why it is built this way (what the test datasets showed)

- The threshold outline of a cell grows with its brightness (max projection keeps the out-of-focus skirt; bright
  cells merge with processes and neighbours): areas from outlines are wrong. One cell body = area above half of the
  cell's own peak.
- Bleed-through goes both ways and depends on the microscope; a tiny copy of a very bright channel can cross a dim
  channel's threshold. It is measured per channel pair in every dataset, never assumed.
- Mask overlap fakes co-expression; signals are compared in noise units after removing the copies.
- The hole dims each colour differently, and the aperture itself can glow: the cut for the hole is learnt from the
  hole cells, the surrounding cells provide the references (bleed-through, cell size, types, neighbours).
- A well without cells is only empty when it is in focus: the workflow flags, you decide.

## Vocabulary

See the end of `report.html` (`unified/vocab.py`).
