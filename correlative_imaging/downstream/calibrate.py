"""Calibrated colour of each refined cell: how green / red is it, in units of a typical cell of that colour?

Nothing here is fitted on control wells; everything comes from the field cells and from all holes of the run.

1 reference  per plate x channel: median over wells of (median field-cell intensity - background), i.e. what a
             typical cell of that colour looks like on that plate. Field cells = whole-image particles >= 30 um2.
2 bleed      per plate and bleed pair: the `calib_bleed_quantile` of (into - bg) / (from - bg) over bright field
             cells of the source colour (low quantile: robust to genuinely double-positive cells).
3 per cell   a_c = body mean - background; a_into -= k x a_from; n_c = max(a_c, 0) / reference_c
4 call       two reporters: x = ln((n_1 + eps) / (n_2 + eps)); a two-component Gaussian mixture over all refined
             cells of the run gives P(reporter 1); the call needs P >= `calib_p_call`, otherwise "uncertain".
             Both n below eps -> "dim". More reporters: P_c = n_c / sum(n) (placeholder until multilabel mode).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config


def references(field: pd.DataFrame, ratios: pd.DataFrame, cfg: Config):
    f = field.assign(sig=field.field_median - field.bg_mean)
    ref = f.groupby(["plate_id", "ch"]).sig.median().rename("ref").reset_index()
    rows = []
    if len(ratios):
        for (pl, src, into), d in ratios.groupby(["plate_id", "src", "into"]):
            d = d[(d.own > d.own.median()) & (d.own > 0)]
            k = float(np.quantile(d.sec / d.own, cfg.calib_bleed_quantile)) if len(d) >= 10 else 0.0
            rows.append(dict(plate_id=pl, src=src, into=into, k=max(k, 0.0), n=len(d)))
    return ref, pd.DataFrame(rows, columns=["plate_id", "src", "into", "k", "n"])


def calibrate_cells(cells: pd.DataFrame, wells: pd.DataFrame, ref: pd.DataFrame, bleed: pd.DataFrame, cfg: Config):
    """cells: refined candidate cells (plate_id, well_id, body_mean_<ch>). Adds n_<rep>, p_<rep>, call_cal."""
    c = cells.copy()
    if not len(c):
        return c, {}
    W = wells.set_index(["plate_id", "well_id"])
    R = ref.set_index(["plate_id", "ch"]).ref
    for ch in cfg.reporters:
        bg = np.array([W.at[(p, w), f"bg_mean_{ch}"] for p, w in zip(c.plate_id, c.well_id)], float)
        c[f"a_{ch}"] = c[f"body_mean_{ch}"] - bg
    for b in cfg.bleed:
        if b["into"] not in cfg.reporters or b["from"] not in cfg.reporters:
            continue
        kk = bleed.set_index(["plate_id", "src", "into"]).k if len(bleed) else pd.Series(dtype=float)
        k = np.array([kk.get((p, b["from"], b["into"]), 0.0) for p in c.plate_id])
        c[f"a_{b['into']}"] -= k * c[f"a_{b['from']}"].clip(lower=0)
    for ch in cfg.reporters:
        ref_c = np.array([R.get((p, ch), np.nan) for p in c.plate_id])
        c[f"n_{ch}"] = c[f"a_{ch}"].clip(lower=0) / ref_c
    info = {}
    reps = cfg.reporters
    if len(reps) == 2:
        from sklearn.mixture import GaussianMixture
        a, b = reps
        x = np.log((c[f"n_{a}"] + cfg.calib_eps) / (c[f"n_{b}"] + cfg.calib_eps)).to_numpy()
        c["log_ratio"] = x
        ok = np.isfinite(x)
        gm = GaussianMixture(2, means_init=[[-2.0], [2.0]], random_state=0).fit(x[ok, None])
        hi = int(np.argmax(gm.means_.ravel()))
        p = np.full(len(c), np.nan)
        p[ok] = gm.predict_proba(x[ok, None])[:, hi]
        c[f"p_{a}"], c[f"p_{b}"] = p, 1 - p
        info = dict(means=gm.means_.ravel().tolist(), sds=np.sqrt(gm.covariances_.ravel()).tolist(),
                    weights=gm.weights_.tolist(), reporter_hi=a)
    else:
        tot = sum(c[f"n_{r}"] for r in reps)
        for r in reps:
            c[f"p_{r}"] = c[f"n_{r}"] / tot.replace(0, np.nan)
    P = c[[f"p_{r}" for r in reps]].to_numpy()
    top = np.nanargmax(np.nan_to_num(P, nan=-1), axis=1)
    dim = np.all(c[[f"n_{r}" for r in reps]].to_numpy() < cfg.calib_eps, axis=1)
    c["call_cal"] = np.where(dim, "dim", np.where(np.nanmax(np.nan_to_num(P, nan=0), axis=1) >= cfg.calib_p_call,
                                                  np.array(reps)[top], "uncertain"))
    return c, info
