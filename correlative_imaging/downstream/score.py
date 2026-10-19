"""Per-particle signal, probabilities and role.

SNR = (particle mean inside the hole - background mean) / background SD, background = the whole image outside
particles (the in-hole background passes the halo of bright neighbours and doubles the colour error).

Roles, in this order:
  1 neighbour  share inside < 0.5 AND covers < 0.25 of the hole   -> "neighbour" (P(cell) >= candidate) / "rim noise"
  2 bleed      e.g. a GFP particle with green < 10 % of red over its own footprint
  3 signal     P(cell) >= 0.5 "candidate", >= 0.2 "weak", else "noise"
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .models import ParticleModels


def score_particles(p: pd.DataFrame, wells: pd.DataFrame, cfg: Config, models: ParticleModels) -> pd.DataFrame:
    """p: particles touching the hole (columns ch, mean, in_px, share, fill, x_<ch>); wells: one row per well with
    px_um, bg_mean_<ch>, bg_sd_<ch>. Returns p with snr, in_um2, p_cell, p_nb, neighbour, bleed, role."""
    p = p.copy()
    W = wells.set_index(["plate_id", "well_id"])
    idx = list(zip(p.plate_id, p.well_id))
    px = W.px_um.reindex(idx).to_numpy()
    p["in_um2"] = p.in_px * px ** 2
    p["full_um2"] = p.full_px * px ** 2
    bm = np.array([W.at[k, f"bg_mean_{c}"] for k, c in zip(idx, p.ch)], float)
    bs = np.array([W.at[k, f"bg_sd_{c}"] for k, c in zip(idx, p.ch)], float)
    p["snr"] = (p["mean"].to_numpy() - bm) / np.where(bs > 0, bs, np.nan)
    p["p_cell"] = models.p_cell(p.snr, p.in_um2, p.ch)
    p["p_nb"] = models.p_nb(p.share, p.fill)
    p["neighbour"] = (p.share < cfg.neighbour_share_max) & (p.fill < cfg.neighbour_fill_max)
    p["bleed"] = False
    for b in cfg.bleed:
        lr = np.log10(np.clip(p[f"x_{b['into']}"], .1, None) / np.clip(p[f"x_{b['from']}"], .1, None))
        p.loc[p.ch == b["into"], "bleed_lr"] = lr[p.ch == b["into"]]
        p["bleed"] |= (p.ch == b["into"]) & (lr < b["max_log_ratio"])
    p["role"] = np.select(
        [p.neighbour & (p.p_cell >= cfg.p_candidate), p.neighbour, p.bleed,
         p.p_cell >= cfg.p_candidate, p.p_cell >= cfg.p_weak],
        ["neighbour", "rim noise", "bleed", "candidate", "weak"], "noise")
    return p
