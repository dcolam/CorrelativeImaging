"""Calibrated identity of each reconstructed cell: which combination of channels is it positive in?

1 units      n_c = (body mean - background) / typical field cell of that plate x channel (reference.py).
             Every channel is measured over the same body. Background (`identity_bg`): "hole" = median of the hole
             pixels outside every object (removes the aperture's own glow, which differs between wells and runs and
             otherwise makes every body a little positive in every channel); falls back to the frame background
             when the cell fills the hole. "frame" = the whole frame outside particles.
2 per channel  P(positive in c) from a two-component Gaussian mixture of asinh(n_c / 0.05) over ALL cells
             reconstructed in the run's holes (signed: noise around 0 stays a continuous negative component instead of
             piling up at a clip; log-like above ~5 % of a typical cell). The hole population is the reference here, not the field cells:
             cells in holes are dimmer than field cells (aperture), whether they were patched or not. Nothing is
             fitted on control wells. Mixture separation (Ashman's D) is reported per channel; D < `min_mixture_sep`
             flags a channel whose positives and negatives do not separate well (e.g. nearly all cells positive).
             Fewer than 20 cells: no mixture; P from the body SNR against the hole background (flagged).
3 combination  P(S) = prod_{c in S} P_c x prod_{c not in S} (1 - P_c) over every subset S of the identity channels
             (2^K, K = 1..4 in practice). The call is the most likely subset ("none" = negative in all), or
             "uncertain" when its probability is below `p_call`. Marker channels are measured, never called.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from .config import Config

S0 = 0.05          # asinh scale: 5 % of a typical field cell
MIN_CELLS = 20


def _mixture(x: np.ndarray):
    from sklearn.mixture import GaussianMixture
    ok = np.isfinite(x)
    if ok.sum() < MIN_CELLS:
        return None
    gm = GaussianMixture(2, means_init=[[np.nanpercentile(x, 10)], [np.nanpercentile(x, 90)]],
                         random_state=0).fit(x[ok, None])
    mu, sd = gm.means_.ravel(), np.sqrt(gm.covariances_.ravel())
    hi = int(np.argmax(mu))
    p = np.full(len(x), np.nan)
    p[ok] = gm.predict_proba(x[ok, None])[:, hi]
    D = float(np.sqrt(2) * abs(mu[0] - mu[1]) / np.sqrt(sd[0] ** 2 + sd[1] ** 2))
    return p, dict(means=mu.tolist(), sds=sd.tolist(), weights=gm.weights_.tolist(), ashman_d=D, n=int(ok.sum()))


def combos(channels: list) -> list[tuple]:
    out = [()]
    for k in range(1, len(channels) + 1):
        out += list(combinations(channels, k))
    return out


def identify(cells: pd.DataFrame, wells: pd.DataFrame, ref: pd.DataFrame, identity: list, cfg: Config,
             stored: dict | None = None):
    """cells: reconstructed cells (plate_id, well_id, body_mean_<ch>). Returns (cells + n_<ch>, p_<ch>, call,
    p_call_best, dominant), info per channel. With `stored` (a previous run's info) the per-channel mixtures are
    not refitted but applied as stored, so changing a few cells does not shift any other call."""
    c = cells.copy()
    info = {}
    if not len(c):
        return c, info
    W = wells.set_index(["plate_id", "well_id"])
    Rf = ref.set_index(["plate_id", "ch"]).ref_med
    key = list(zip(c.plate_id, c.well_id))
    for ch in identity:
        bg = np.array([W.at[k, f"bg_mean_{ch}"] for k in key], float)
        if cfg.identity_bg == "hole":
            bg = np.array([W.at[k, f"hbg_med_{ch}"] for k in key], float)
        rm = np.array([Rf.get((p, ch), np.nan) for p in c.plate_id], float)
        c[f"n_{ch}"] = (c[f"body_mean_{ch}"] - bg) / rm
        x = np.arcsinh(c[f"n_{ch}"].to_numpy(float) / S0)
        st = (stored or {}).get(ch)
        if stored is not None and st and st.get("means"):
            from scipy.stats import norm
            mu, sd, w = np.asarray(st["means"]), np.asarray(st["sds"]), np.asarray(st["weights"])
            hi = int(np.argmax(mu))
            dens = np.vstack([w[j] * norm.pdf(x, mu[j], sd[j]) for j in range(len(mu))])
            c[f"p_{ch}"] = dens[hi] / np.where(dens.sum(0) > 0, dens.sum(0), np.nan)
            info[ch] = st
            continue
        mx = None if stored is not None else _mixture(x)
        if mx is None:
            hm = np.array([W.at[k, f"hbg_med_{ch}"] for k in key], float)
            hs = np.array([W.at[k, f"hbg_sd_{ch}"] for k in key], float)
            snr = (c[f"body_mean_{ch}"].to_numpy(float) - hm) / np.where(hs > 0, hs, np.nan)
            c[f"p_{ch}"] = 1 / (1 + np.exp(-(snr - 5) / 1.5))
            info[ch] = dict(mixture=None, note=f"fewer than {MIN_CELLS} cells: P from body SNR vs hole background")
        else:
            c[f"p_{ch}"], info[ch] = mx
            info[ch]["weak_separation"] = info[ch]["ashman_d"] < cfg.min_mixture_sep
    P = c[[f"p_{ch}" for ch in identity]].to_numpy(float)
    P = np.where(np.isfinite(P), P, 0.5)
    allc = combos(identity)
    L = np.zeros((len(c), len(allc)))
    for j, S in enumerate(allc):
        m = np.array([ch in S for ch in identity])
        L[:, j] = np.prod(np.where(m, P, 1 - P), axis=1)
    best = L.argmax(1)
    pbest = L.max(1)
    name = ["none" if not S else "+".join(S) for S in allc]
    c["combo"] = [name[i] for i in best]
    c["p_combo"] = pbest
    c["call"] = np.where(pbest >= cfg.p_call, c.combo, "uncertain")
    N = c[[f"n_{ch}" for ch in identity]].to_numpy(float)
    c["dominant"] = [identity[i] if np.isfinite(r).any() else "" for i, r in
                     zip(np.nanargmax(np.where(np.isfinite(N), N, -np.inf), axis=1), N)]
    return c, info


def apply_identity(cells, wells, ref, identity, cfg, stored: dict) -> pd.DataFrame:
    """Identity of a few cells with the run's stored calibration (no refit)."""
    return identify(cells, wells, ref, identity, cfg, stored=stored or {})[0]


def well_calls(cells: pd.DataFrame, wells: pd.DataFrame, identity: list) -> pd.DataFrame:
    """One row per well: the best cell in the hole (highest P(cell) x share of its body inside the hole)."""
    base = wells[["plate_id", "well_id"]].copy()
    if not len(cells):
        return base.assign(n_cells=0, call="no cell", status="no cell")
    c = cells.assign(rank=cells.p_cell.fillna(0) * cells.body_in_hole)
    best = c.sort_values("rank", ascending=False).drop_duplicates(["plate_id", "well_id"])
    n = cells.groupby(["plate_id", "well_id"]).size().rename("n_cells")
    keep = ["plate_id", "well_id", "cell_id", "call", "combo", "p_combo", "dominant", "p_cell", "body_um2", "grow_x",
            "body_in_hole", "hole_cover", "touches_neighbour", "n_dips"] + \
           [f"{p}_{ch}" for ch in identity for p in ("n", "p")]
    out = base.merge(n.reset_index(), how="left").merge(best[[k for k in keep if k in best]], how="left")
    out["n_cells"] = out.n_cells.fillna(0).astype(int)
    out["call"] = out.call.fillna("no cell")
    out["status"] = np.select([out.n_cells == 0, out.n_cells == 1], ["no cell", "one cell"], "several cells")
    out.insert(0, "well_plate", out.well_id + "." + out.plate_id)
    return out
