"""Particle features that mean the same thing for 1, 2, 3 or 4 channels.

Every feature is computed in the particle's own channel, in calibrated units (frame background, hole background,
the plate x channel field reference). Other channels enter only through summaries that do not depend on how many
there are or on their order (max, mean, fraction positive, spectral flatness). The channel itself is never a
feature, so one model serves every channel and every dataset.

  own signal     f_snr      asinh(SNR / 2), SNR = (mean inside - frame bg) / frame bg SD
                 f_hsnr     the same against the hole's own background (aperture-local contrast)
                 f_rel      log10 brightness inside in typical-cell units (dynamic-range calibration)
                 f_relmax   log10 peak brightness inside, typical-cell units
                 f_field_pct  where the whole object's brightness sits in the plate's field-cell distribution (0-1)
                 f_rel_ctx  log10 brightness relative to the median object of that channel in the window
  shape          f_area_in, f_area_full (log10 um2), f_area_z (vs field cells), f_circ, f_sol, f_ecc, f_peaks
                 f_cells    log2 of the whole object's size in cells: area / the area of one cell of that plate x
                            channel (reference.cell_um2, the peak of the field objects' areas). 0 = one cell, -2 = a
                            quarter (debris), 1 = two cells, 2 = four (a clump). Also as columns cells_full and
                            cells_in (the part inside the hole)
  position       f_share (logit share inside), f_fill (log coverage of the hole), f_d_in, f_d_full (hole radii),
                 f_edge, f_lenient (lenient piece), f_parent_share, f_field_d2 (log1p distance of the whole parent
                 object from the typical field cell: small = looks like a complete neighbour)
  context        f_density  log1p number of objects (any channel) in the context disc
  other channels f_oth_max, f_oth_mean (log10 typical-cell units over the same footprint), f_oth_pos (fraction of
                 other channels with SNR > 3), f_flat (min / max brightness over all channels: 1 = equally bright
                 everywhere, the debris signature), f_dom (own minus brightest other, log10),
                 f_coloc (the highest pixel-wise Pearson r with another channel inside the particle: co-varying
                 pixels = one structure seen in two channels, e.g. a double-positive cell or autofluorescent debris;
                 bleed-through gives r near 1 with a much dimmer other channel)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .reference import field_likeness

FEATURES = ["f_snr", "f_hsnr", "f_rel", "f_relmax", "f_field_pct", "f_rel_ctx",
            "f_area_in", "f_area_full", "f_area_z", "f_circ", "f_sol", "f_ecc", "f_peaks",
            "f_share", "f_fill", "f_d_in", "f_d_full", "f_edge", "f_lenient", "f_parent_share", "f_field_d2",
            "f_density",
            "f_oth_max", "f_oth_mean", "f_oth_pos", "f_flat", "f_dom", "f_coloc", "f_cells"]
# the reference workflow's feature set, kept for the ablation in learn.py
LEGACY_FEATURES = ["f_snr", "f_area_in", "f_share", "f_fill"]

_EPS = 1e-3


def _logit(p):
    p = np.clip(np.asarray(p, float), _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def _log_rel(sig, ref):
    return np.log10(np.clip(np.asarray(sig, float) / np.asarray(ref, float), _EPS, None))


def _norm_cdf(z):
    from scipy.special import ndtr
    return ndtr(z)


def build_features(p: pd.DataFrame, wells: pd.DataFrame, ref: pd.DataFrame, channels: list) -> pd.DataFrame:
    """p: particles of detect.detect_well (any number of wells); wells: their well rows; ref: field_reference.
    Returns p with SNRs, calibrated brightness and the FEATURES columns. Context objects get the features that
    make sense for them (no 'inside' values)."""
    if not len(p):
        return p.assign(**{f: pd.Series(dtype=float) for f in FEATURES})
    p = p.copy()
    W = wells.set_index(["plate_id", "well_id"])
    key = list(zip(p.plate_id, p.well_id))
    px = W.px_um.reindex(key).to_numpy(float)

    def wcol(prefix, chs):
        return np.array([W.at[k, f"{prefix}_{c}"] for k, c in zip(key, chs)], float)

    R_ = ref.set_index(["plate_id", "ch"])
    rr = R_.reindex(list(zip(p.plate_id, p.ch)))
    bg, bsd = wcol("bg_mean", p.ch), wcol("bg_sd", p.ch)
    hm, hs = wcol("hbg_med", p.ch), wcol("hbg_sd", p.ch)
    inside = p.pool.to_numpy(bool)
    mean_in = np.where(inside, p.get("in_mean", np.nan), p.full_mean).astype(float)
    p["snr"] = (mean_in - bg) / np.where(bsd > 0, bsd, np.nan)
    p["hsnr"] = (mean_in - hm) / np.where(hs > 0, hs, np.nan)
    p["in_um2"] = p.get("in_px", np.nan) * px ** 2
    p["full_um2"] = p.full_px * px ** 2
    ref_med = rr.ref_med.to_numpy(float)
    p["ref_med"] = ref_med
    p["rel"] = (mean_in - bg) / ref_med
    sig_full = p.full_mean.to_numpy(float) - bg
    p["rel_full"] = sig_full / ref_med
    lsig_full = np.log10(np.clip(sig_full, _EPS, None))
    larea_full = np.log10(p.full_um2.to_numpy(float))

    p["f_snr"] = np.arcsinh(p.snr / 2)
    p["f_hsnr"] = np.arcsinh(p.hsnr / 2)
    p["f_rel"] = _log_rel(mean_in - bg, ref_med)
    p["f_relmax"] = _log_rel(np.where(inside, p.get("in_max", np.nan), p.full_max).astype(float) - bg, ref_med)
    p["f_field_pct"] = _norm_cdf((lsig_full - rr.mu_lsig.to_numpy(float)) / rr.sd_lsig.to_numpy(float))
    p["f_area_in"] = np.log10(np.clip(p.in_um2.to_numpy(float), .05, None))
    p["f_area_full"] = larea_full
    p["f_area_z"] = (larea_full - rr.mu_larea.to_numpy(float)) / rr.sd_larea.to_numpy(float)
    one = rr.cell_um2.to_numpy(float) if "cell_um2" in rr else np.full(len(p), np.nan)   # area of one cell
    p["cell_um2"] = one
    p["cells_full"] = p.full_um2.to_numpy(float) / one
    p["cells_in"] = p.in_um2.to_numpy(float) / one
    p["f_cells"] = np.log2(np.clip(p.cells_full.to_numpy(float), 1 / 64, 64))
    p["f_circ"], p["f_sol"], p["f_ecc"] = p.circ, p.solidity, p.ecc
    p["f_peaks"] = np.log1p(p.n_peaks)
    p["f_share"] = _logit(p.get("share", np.nan))
    p["f_fill"] = np.log(np.clip(p.get("fill", np.nan).astype(float), _EPS, None))
    p["f_d_in"] = p.get("d_in", np.nan)
    p["f_d_full"] = p.d_full
    p["f_edge"] = p.edge.astype(float)
    p["f_lenient"] = (p.level == "hole").astype(float)
    p["fd2"] = field_likeness(lsig_full, larea_full, p.circ, p.solidity, rr)

    # parent (whole stringent object behind a lenient piece): its share and its field-likeness
    outer = p.outer.fillna(False).to_numpy(bool) if "outer" in p else np.zeros(len(p), bool)
    by_oid = p[~outer].set_index(["plate_id", "well_id", "oid"])     # cells far out in the crop are no parents
    pk = list(zip(p.plate_id, p.well_id, p.parent))
    par = by_oid.reindex(pk)
    p["f_parent_share"] = par.get("share", pd.Series(np.nan, index=par.index)).to_numpy(float)
    pd2 = par.fd2.to_numpy(float)
    p["f_field_d2"] = np.log1p(np.where(np.isfinite(pd2), pd2, p.fd2))

    # context: brightness relative to the window's objects of that channel, object density
    ctx = (p.level == "whole").to_numpy() & ~outer                  # the window's objects, as the model knows them
    whole = p[ctx]
    s_whole = pd.Series(sig_full, index=p.index)[ctx]
    med = s_whole.groupby([p.plate_id[s_whole.index], p.well_id[s_whole.index], p.ch[s_whole.index]]).median()
    m = med.reindex(list(zip(p.plate_id, p.well_id, p.ch))).to_numpy(float)
    p["f_rel_ctx"] = np.log10(np.clip((mean_in - bg) / np.where(m > 0, m, np.nan), _EPS, None))
    dens = whole.groupby(["plate_id", "well_id"]).size()
    p["f_density"] = np.log1p(dens.reindex(key).fillna(0).to_numpy(float))

    # other channels over the same footprint (inside for pool particles, full object otherwise)
    rel_all = {}
    snr_all = {}
    for c in channels:
        col_in, col_full = f"in_{c}", f"full_{c}"
        if col_full not in p:
            continue
        v = np.where(inside, p.get(col_in, np.nan), p[col_full]).astype(float)
        b = np.array([W.at[k, f"bg_mean_{c}"] for k in key], float)
        s = np.array([W.at[k, f"bg_sd_{c}"] for k in key], float)
        rm = R_.ref_med.reindex([(pl, c) for pl in p.plate_id]).to_numpy(float)
        rel_all[c] = (v - b) / rm
        snr_all[c] = (v - b) / np.where(s > 0, s, np.nan)
    chs = list(rel_all)
    ch_idx = np.array([chs.index(c) if c in chs else -1 for c in p.ch])
    REL = np.vstack([rel_all[c] for c in chs]).T if chs else np.zeros((len(p), 0))
    SNR = np.vstack([snr_all[c] for c in chs]).T if chs else np.zeros((len(p), 0))
    others = np.ones_like(REL, bool)
    others[np.arange(len(p)), ch_idx] = False
    n_oth = others.sum(1)
    import warnings
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        R_o = np.where(others, REL, np.nan)
        p["f_oth_max"] = np.where(n_oth > 0, np.log10(np.clip(np.nanmax(np.where(others, REL, -np.inf), 1), _EPS, None)), np.nan)
        p["f_oth_mean"] = np.where(n_oth > 0, np.log10(np.clip(np.nanmean(R_o, 1), _EPS, None)), np.nan)
        p["f_oth_pos"] = np.where(n_oth > 0, np.nansum(np.where(others, SNR > 3, False), 1) / np.maximum(n_oth, 1), np.nan)
        Rc = np.clip(REL, _EPS, None)
        p["f_flat"] = np.where(n_oth > 0, np.nanmin(Rc, 1) / np.nanmax(Rc, 1), np.nan)
    p["f_dom"] = p.f_rel - p.f_oth_max
    rcols = [f"r_{c}" for c in channels if f"r_{c}" in p]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        p["f_coloc"] = np.nanmax(p[rcols].to_numpy(float), 1) if rcols else np.nan   # NaN: one channel / detected before
    return p.rename(columns={"fd2": "field_d2"})
