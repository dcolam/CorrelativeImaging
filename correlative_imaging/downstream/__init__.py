"""Downstream analyzer: which cell sits in the patch-clamp hole, and what colour is it?

Runs on the OUTPUT of a finished CorrelativeImaging run (per-plate .db + diagnostic hole crops) and never
modifies it. Steps, one module each:

    sources    where wells come from (a finished run, or the local v5 dev cache)
    particles  particles of both thresholds in the crop (stringent = whole-image, lenient = hole threshold)
    score      SNR, P(cell), P(neighbour) (models.py), roles in a fixed order
    cells      particles -> cells, colour-blind confidence tier
    refine     seeded, competing growth: the full outline of the patched cell, neighbours claim their pixels
    calibrate  field-cell calibration per plate x channel, bleed-through, colour probability per cell
    viewer     the browser viewer (diagviewer.js) with particle / cell overlays
    evaluate   joins the experiment table (col_data) -- the only module that sees it

Rule: nothing that decides a call ever sees col_data (plate map, ephys). It is joined only for evaluation and
display. No cut is fitted on control (Empty / pure) wells.

Run:  python -m correlative_imaging.downstream --help
"""
