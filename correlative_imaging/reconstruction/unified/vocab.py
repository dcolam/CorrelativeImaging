"""The words the workflow uses, one definition each (shown in the report and in README.md)."""

MODEL_UNSURE = "not sure (model)"     # shown for bio_label == "unsure" (the stored value stays "unsure")


def shown(label) -> str:
    """A label as shown: the label model's "unsure" is told apart from your own answer "unsure"."""
    return MODEL_UNSURE if label == "unsure" else str(label)


TERMS = [
    ("hole", "The aperture of the patch-clamp chip in the image; the patched cell sits on it. Drawn by the pipeline "
             "(hole ROI)."),
    ("hole cell", "What grew from the particles inside the hole: the candidate patched cell."),
    ("surrounding cells", "Every cell the pipeline segmented on the whole image outside the hole. Unpatched, usually "
                          "brighter (no aperture), used to learn what each cell type looks like in this dataset."),
    ("particle / object", "A connected area above a channel's threshold. One cell can be one particle in several "
                          "channels; one particle can hold several touching cells."),
    ("colour channel", "A reporter that defines the cell type (e.g. GFP, mCherry, BFP, mScarlet)."),
    ("marker channel", "A stain that is measured and reported but does not define the type (e.g. the ROMK antibody)."),
    ("background", "Surrounding cells: the median of a ring 1-4 um around the object outside all objects (the image's "
                   "background where the ring is too crowded). Hole cells: the band 2-6 um outside the hole."),
    ("noise", "Robust spread (1.4826 x median absolute deviation) of the image outside all objects."),
    ("signal / contrast / SNR", "(median inside - background) / noise: how many noise units a cell stands out. Means "
                                "the same in every channel and image."),
    ("own signal", "The part of a channel's signal in the hole cell that is not a copy of the other channels "
                   "(the pipeline's ChannelRelation residual), in noise units."),
    ("bleed-through (copy)", "Light of one fluorophore recorded in another channel. Usually from the shorter- into the "
                             "longer-wavelength channel, but the reverse happens (cross-excitation, filters). Measured "
                             "per channel pair in every dataset; what makes a false object is copy x brightness of the "
                             "source against the receiving channel's threshold."),
    ("co-expression", "A cell really carrying two colours. Looks like bleed-through in a scatter plot; a 'copy' "
                      "larger than 30 % of the source cannot be bleed-through."),
    ("exemplar", "A clean surrounding cell used as reference: compact, typical size, not saturated, other colours "
                 "either fully on it or absent."),
    ("pattern (channel positivity)", "The channels a cell is positive in, named as in the run, e.g. 'FITC+ TX RED+'; "
     "'none' when no channel is. The first pass gives only this: no cell types, no priority between channels."),
    ("not sure (model)", "The label model's best guess for a cell has a probability below 0.7. Not something you "
                         "have to fill in; labelling some of these cells (Learn: 'Label cells the model is not sure "
                         "about') teaches the model where it is unsure."),
    ("your label / label model", "What a cell is, in your words (astrocyte, interneuron, ...), from the label sheet. "
     "An LDA learns your labels from the channel signals (relative to a typical positive cell of the same population) "
     "of the holes and "
     "surrounding cells you labelled, and labels every cell with a probability; 'unsure' below 0.7."),
    ("cell size (one cell body)", "Area of the part of a cell above half of its own peak brightness. The threshold "
                                  "outline is not used for size: it grows with brightness (out-of-focus skirt of the "
                                  "max projection, processes, touching neighbours)."),
    ("debris", "Objects smaller than the valley between the speck peak and the cell peak of the size distribution."),
    ("clump / several cells", "A hole cell of 2 or more cell bodies: either cell could be the patched one."),
    ("neighbour", "A surrounding cell reaching into or lying next to the hole; a barrier when the hole cell is grown."),
    ("glow", "Light of a bright neighbour spilling into the hole cell: bright near the neighbour, dim far from it."),
    ("halo / plug", "The aperture itself glowing (often in every channel), filling the hole like a disc: not a cell."),
    ("dim (e.g. 'FITC? (dim)')", "A hole cell with no channel above its cut whose strongest channel reaches half its "
     "cut and at least twice the next (or is above the cut but no brighter than an empty hole). Probably dim and "
     "positive in that channel; the channels you see positive in the label sheet set where dim starts."),
    ("cell colour (hue)", "A cell's colour as seen on screen: the channels' display colours mixed by each channel's "
                          "signal relative to a typical positive cell (capped at one). Hue = the angle on the colour "
                          "circle; purity = how much one hue dominates (several channels equally strong mix to near "
                          "white); cells below a quarter of a typical positive cell in every channel have no colour."),
    ("fragment", "A hole cell smaller than a quarter of a cell body."),
    ("q-value", "Mirrored false-discovery rate: among cells this bright in a channel, the expected share that are "
                "only noise (from the cells below 0, mirrored). Positive at q < 0.05."),
    ("two groups / separation", "A channel whose cells split into a negative and a positive group; separation >= 2 = "
                                "clearly two groups, and the cut sits where both are equally likely."),
    ("well checks", "Few cells (< a quarter of the plate median: focus lost or empty well), barcode mismatch between "
                    "file names and plate folder, no hole."),
]
