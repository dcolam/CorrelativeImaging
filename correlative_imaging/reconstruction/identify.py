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
2b your colour  With cell labels (outside the test wells) that judge a channel at least MIN_USER times positive and
             MIN_USER times negative, P(positive in c) comes from a logistic fit of those answers on the same
             asinh scale instead of the mixture (the mixture is still fitted and reported, with its agreement with
             your answers). Cells judged "several cells" / "not a cell" are never part of that fit.
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
MIN_USER = 5       # judged cells per side (positive / negative) before your colour calibration takes over


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


def _user_fit(x: np.ndarray, pos: np.ndarray):
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=10.0).fit(x[:, None], pos.astype(int))
    a, b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
    return a, b


def identify(cells: pd.DataFrame, wells: pd.DataFrame, ref: pd.DataFrame, identity: list, cfg: Config,
             stored: dict | None = None, user: pd.DataFrame | None = None):
    """cells: reconstructed cells (plate_id, well_id, body_mean_<ch>). Returns (cells + n_<ch>, p_<ch>, call,
    p_call_best, dominant), info per channel. With `stored` (a previous run's info) the per-channel mixtures are
    not refitted but applied as stored, so changing a few cells does not shift any other call. user: judged cells
    (plate_id, well_id, cell_id, verdict, colour), test wells already removed (see module doc, 2b)."""
    c = cells.copy().reset_index(drop=True)
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
        if stored is not None and st and st.get("user"):
            a, b = st["user"]["a"], st["user"]["b"]
            c[f"p_{ch}"] = 1 / (1 + np.exp(-(a * x + b)))
            info[ch] = st
            continue
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
        if user is not None and len(user) and stored is None:
            u = c[["plate_id", "well_id", "cell_id"]].reset_index().merge(user, on=["plate_id", "well_id", "cell_id"])
            u = u[(u.verdict.fillna("one cell").isin(["one cell", ""])) & (u.colour.fillna("") != "")]
            if len(u):
                pos = np.array([ch in str(col).split("+") for col in u.colour])
                xu = x[u["index"].to_numpy()]
                ok = np.isfinite(xu)
                agree = float(((c[f"p_{ch}"].to_numpy(float)[u["index"].to_numpy()] >= 0.5) == pos).mean())
                info[ch]["user_agreement_before"] = agree
                info[ch]["user_n"] = dict(pos=int(pos[ok].sum()), neg=int((~pos[ok]).sum()))
                if pos[ok].sum() >= MIN_USER and (~pos[ok]).sum() >= MIN_USER:
                    a, b = _user_fit(xu[ok], pos[ok])
                    c[f"p_{ch}"] = 1 / (1 + np.exp(-(a * x + b)))
                    info[ch]["user"] = dict(a=a, b=b, n_pos=int(pos[ok].sum()), n_neg=int((~pos[ok]).sum()),
                                            cut_n=float(S0 * np.sinh(-b / a)) if a else None)
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
    if "verdict" in c:                                  # judged by hand: "not a cell" never wins the well
        c.loc[c.verdict == "not a cell", "rank"] = -1
    best = c.sort_values("rank", ascending=False).drop_duplicates(["plate_id", "well_id"])
    n = cells.groupby(["plate_id", "well_id"]).size().rename("n_cells")
    keep = ["plate_id", "well_id", "cell_id", "call", "combo", "p_combo", "dominant", "p_cell", "body_um2", "grow_x",
            "body_in_hole", "hole_cover", "touches_neighbour", "n_dips"] + \
           [f"{p}_{ch}" for ch in identity for p in ("n", "p")]
    out = base.merge(n.reset_index(), how="left").merge(best[[k for k in keep if k in best]], how="left")
    out["n_cells"] = out.n_cells.fillna(0).astype(int)
    out["call"] = out.call.fillna("no cell")
    out["status"] = np.select([out.n_cells == 0, out.n_cells == 1], ["no cell", "one cell"], "several cells")
    if "verdict" in best:
        v = out[["plate_id", "well_id"]].merge(best[["plate_id", "well_id", "verdict"]], how="left").verdict
        sev = (v == "several cells").to_numpy()
        out.loc[sev, "status"] = "several cells (by eye)"     # cannot be split by eye: either could be patched
        out.loc[sev, "call"] = "ambiguous"
        notc = (v == "not a cell").to_numpy()
        out.loc[notc, "status"] = "no cell"
        out.loc[notc, "call"] = "no cell"
    out.insert(0, "well_plate", out.well_id + "." + out.plate_id)
    return out


def advice(info: dict, cells: pd.DataFrame | None, calls: pd.DataFrame | None, cfg: Config) -> dict:
    """Plain-language hints per identity channel: why the call may be weak and what would help."""
    out = {}
    for ch, d in (info or {}).items():
        tips = []
        if not d.get("means"):
            tips.append(f"fewer than {MIN_CELLS} cells: no mixture, the call uses the body's SNR. Classify more wells, "
                        "or judge cell colours (Bulk: cells) to calibrate it")
            out[ch] = tips
            continue
        mu, w = np.asarray(d["means"]), np.asarray(d["weights"])
        w_pos = float(w[int(np.argmax(mu))])
        D = d.get("ashman_d", np.nan)
        if d.get("user"):
            u = d["user"]
            tips.append(f"your colour labels set the cut ({u['n_pos']} positive / {u['n_neg']} negative cells)")
        if w_pos > 0.85:
            tips.append(f"{w_pos:.0%} of the cells come out positive: if this channel is on (nearly) every cell, e.g. "
                        "a stain or a co-expressed marker, make it a marker (Run tab), or the hole background may be "
                        "under-estimated (Settings: identity background)")
        elif w_pos < 0.05:
            tips.append(f"only {w_pos:.0%} of the cells come out positive: a dim fluorophore or a channel that is "
                        "rarely expressed; check the Identity calibration plot (Diagnostics)")
        if np.isfinite(D) and D < cfg.min_mixture_sep:
            tips.append(f"positives and negatives overlap (Ashman's D {D:.1f} < {cfg.min_mixture_sep:g}): dim positives "
                        "blur into background. Judge the colour of ~10 positive and ~10 negative cells (Bulk: cells): "
                        "with ≥ 5 of each your answers set the cut")
        ag = d.get("user_agreement_before")
        if ag is not None and ag < 0.85:
            tips.append(f"the unsupervised mixture agreed with your colour labels on only {ag:.0%} of the cells you "
                        "judged: more colour labels make the calibration take over")
        if not tips:
            tips.append(f"separates well (D {D:.1f}, {w_pos:.0%} positive)")
        out[ch] = tips
    if calls is not None and len(calls):
        unc = float((calls["call"] == "uncertain").mean())
        if unc > 0.1:
            out["_all"] = [f"{unc:.0%} of the wells are 'uncertain' (no combination reaches P {cfg.p_call:g}): "
                           "usually one channel above separates badly; fix that one first"]
    return out
