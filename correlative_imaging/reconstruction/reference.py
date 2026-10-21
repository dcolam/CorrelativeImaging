"""What a typical unpatched cell looks like, per plate x channel: the field reference.

Field cells are the whole-image particles >= `field_min_um2` (the pipeline's own measurements in the database).
Their signal is the object mean minus the frame background of their well. Per plate x channel:

  ref_med         median signal: one "typical cell" of that channel. All brightness features and the identity call
                  are expressed in these units, which calibrates the very different dynamic ranges of the channels.
  ref_q90         90th percentile (dynamic range)
  mu/sd_lsig      log10 signal          mu/sd_larea   log10 area (um2)
  mu/sd_circ      circularity           mu/sd_sol     solidity
  cell_um2        the area of ONE cell: the main peak of the field objects' area distribution (density of log10
                  area). Single cells pile up in a sharp peak, clumps form a tail to the right, so the peak is not
                  pulled up by clumps the way a mean or median is. Every particle's area divided by it gives its size
                  in cells (features.py: ~1 a single cell, ~0.2 debris, 2+ a clump).

A plate x channel with fewer than 20 field cells falls back to the whole run for that channel (`fallback` = True).
A channel without any field cell has no reference (NaN): its brightness features are then missing, which the
classifier tolerates, and the identity call for it relies on the hole population alone.

This reference describes neighbours well (they are unpatched cells). It does not describe the patched cell, which is
dimmer and deformed by the aperture; the identity call therefore calibrates against the population of cells found
in all holes of the run (identify.py), with this reference only as the unit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_FIELD = 20
STATS = [("lsig", "lsig"), ("larea", "larea"), ("circ", "circularity"), ("sol", "solidity")]


def single_cell_area(area_um2) -> float:
    """The main peak of the log10 area density (Gaussian kernel, bandwidth 0.12 of the data's SD scale): the area of
    one typical cell. NaN with fewer than MIN_FIELD objects."""
    a = np.asarray(area_um2, float)
    a = a[np.isfinite(a) & (a > 0)]
    if len(a) < MIN_FIELD:
        return float("nan")
    x = np.log10(a)
    if np.ptp(x) == 0:
        return float(a[0])
    from scipy.stats import gaussian_kde
    grid = np.linspace(x.min(), x.max(), 400)
    k = gaussian_kde(x, bw_method=0.12)(grid)
    return float(10 ** grid[np.argmax(k)])


def _summ(d: pd.DataFrame) -> dict:
    out = dict(n_field=len(d), ref_med=float(d.sig.median()) if len(d) else np.nan,
               ref_q90=float(d.sig.quantile(.9)) if len(d) else np.nan,
               cell_um2=single_cell_area(d.area_um2) if len(d) else np.nan)
    for name, col in STATS:
        v = d[col].astype(float)
        out[f"mu_{name}"] = float(v.mean()) if len(v) else np.nan
        out[f"sd_{name}"] = float(v.std()) if len(v) > 1 else np.nan
    return out


def field_reference(field: pd.DataFrame, wells: pd.DataFrame, channels: list) -> pd.DataFrame:
    """field: plate_id, well_id, ch, area_um2, mean, circularity, solidity; wells: plate_id, well_id, bg_mean_<ch>.
    Returns one row per plate x channel (all plates of `wells`)."""
    plates = sorted(wells.plate_id.unique())
    f = field[field.ch.isin(channels) & field.plate_id.isin(plates)].copy()
    if len(f):
        bg = wells.melt(id_vars=["plate_id", "well_id"], value_vars=[f"bg_mean_{c}" for c in channels
                                                                       if f"bg_mean_{c}" in wells],
                        var_name="ch", value_name="bg")
        bg["ch"] = bg.ch.str[len("bg_mean_"):]
        f = f.merge(bg, on=["plate_id", "well_id", "ch"], how="left")
        plate_bg = bg.groupby(["plate_id", "ch"]).bg.median()     # wells not processed: the plate's median background
        miss = f.bg.isna()
        f.loc[miss, "bg"] = [plate_bg.get((p, c), np.nan) for p, c in zip(f.plate_id[miss], f.ch[miss])]
        f["sig"] = f["mean"] - f.bg
        f = f[f.sig > 0]
        f["lsig"] = np.log10(f.sig)
        f["larea"] = np.log10(f.area_um2)
    rows = []
    for ch in channels:
        d_ch = f[f.ch == ch] if len(f) else f
        run = _summ(d_ch) if len(d_ch) else _summ(pd.DataFrame(columns=["sig"] + [c for _, c in STATS]))
        for pl in plates:
            d = d_ch[d_ch.plate_id == pl] if len(d_ch) else d_ch
            fb = len(d) < MIN_FIELD
            rows.append(dict(plate_id=pl, ch=ch, fallback=fb, **(run if fb else _summ(d))))
    return pd.DataFrame(rows)


def field_likeness(lsig, larea, circ, sol, ref_rows: pd.DataFrame) -> np.ndarray:
    """Squared standardised distance of an object from the typical field cell (diagonal Mahalanobis, 4 terms).
    Small = looks like a whole unpatched cell of that plate x channel. NaN terms are skipped."""
    z = []
    for v, name in ((lsig, "lsig"), (larea, "larea"), (circ, "circ"), (sol, "sol")):
        mu, sd = ref_rows[f"mu_{name}"].to_numpy(float), ref_rows[f"sd_{name}"].to_numpy(float)
        z.append(((np.asarray(v, float) - mu) / np.where(sd > 0, sd, np.nan)) ** 2)
    z = np.vstack(z)
    n = np.isfinite(z).sum(0)
    return np.where(n > 0, np.nansum(z, 0) * 4 / np.maximum(n, 1), np.nan)
