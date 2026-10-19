"""Particles -> cells, the colour-blind confidence tier, and the simple colour call.

Cells: candidate particles of all channels are joined when their footprints overlap or their centroids lie within
`cell_gap_um` (connected components). Neighbour particles are grouped the same way. A cell needs a reporter channel
to count as identity evidence; marker-only cells (e.g. ROMK) are counted separately.

Tier (colour does not enter, defined before looking at any outcome):
  high             one candidate cell, P(cell) >= 0.95, P(nb) <= 0.2 (area-weighted), overlaps a stringent candidate,
                   <= 1 neighbour cell
  middle           one candidate cell, P(cell) >= 0.8, P(nb) <= 0.5
  low              one candidate cell otherwise
  several cells    >= 2 candidate cells
  many neighbours  one candidate cell but >= 2 neighbour cells
  neighbours only  no candidate, >= 1 neighbour cell
  marker only      no reporter cell, only marker objects
  low SNR          only weak particles
  empty            nothing

Simple call (exclusive): summed SNR+ x area per reporter over the candidate cells; the top reporter wins when it is
>= `call_fold` x the next one, otherwise "both" (two reporters) / "mixed:<a>+<b>".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components

from .config import Config

TIERS = ["high", "middle", "low", "several cells", "many neighbours", "neighbours only", "marker only",
         "low SNR", "empty"]


def _centroids(feet, shape):
    yx = [np.unravel_index(f, shape) for f in feet]
    return np.array([a[0].mean() for a in yx]), np.array([a[1].mean() for a in yx])


def group_cells(g: pd.DataFrame, shape, px_um: float, gap_um: float) -> np.ndarray:
    """Connected components over particles g (column foot = flat pixel indices). Returns a label per row."""
    n = len(g)
    if n == 0:
        return np.zeros(0, int)
    feet = list(g.foot)
    cy, cx = _centroids(feet, shape)
    A = np.zeros((n, n), bool)
    for i in range(n):
        for j in range(i + 1, n):
            A[i, j] = A[j, i] = (np.hypot(cy[i] - cy[j], cx[i] - cx[j]) * px_um <= gap_um
                                 or len(np.intersect1d(feet[i], feet[j], assume_unique=True)) > 0)
    return connected_components(A, directed=False)[1]


def colour_call(sig: dict, reporters: list, fold: float) -> str:
    s = sorted(((sig.get(r, 0.0), r) for r in reporters), reverse=True)
    if s[0][0] <= 0:
        return "none"
    if len(s) == 1 or (s[0][0] + 1) / (s[1][0] + 1) >= fold:
        return s[0][1]
    return "both" if len(reporters) == 2 else f"mixed:{s[0][1]}+{s[1][1]}"


def _summarise(m: pd.DataFrame, cfg: Config, channels: list) -> dict:
    sig = {c: float((m[m.ch == c].snr.clip(lower=0) * m[m.ch == c].in_um2).sum()) for c in channels}
    return dict(p_cell=float(m.p_cell.max()), p_nb=float(np.average(m.p_nb, weights=m.in_px)),
                chans="+".join(sorted(set(m.ch))), reporter=bool(m.ch.isin(cfg.reporters).any()),
                foot=np.unique(np.concatenate(list(m.foot))), members=list(m.oid),
                **{f"sig_{c}": v for c, v in sig.items()})


def cells_of_well(L: pd.DataFrame, S: pd.DataFrame, shape, px_um: float, cfg: Config, channels: list):
    """L: scored lenient particles of one well, S: scored stringent ones. Returns (well_row, cells, particle->cell)."""
    def build(d, kind):
        d = d.reset_index(drop=True)
        lab = group_cells(d, shape, px_um, cfg.cell_gap_um)
        out = []
        for k in range(lab.max() + 1 if len(lab) else 0):
            c = _summarise(d[lab == k], cfg, channels)
            c["kind"] = kind
            out.append(c)
        return out

    allc = build(L[L.role == "candidate"], "candidate")
    cand = [c for c in allc if c["reporter"]]
    marker_only = [c for c in allc if not c["reporter"]]
    nbs = build(L[L.role == "neighbour"], "neighbour")
    scand = [c for c in build(S[S.role == "candidate"], "stringent") if c["reporter"]]
    for c in cand:
        c["agrees_stringent"] = any(len(np.intersect1d(c["foot"], s["foot"])) > 0 for s in scand)
    n_weak = int((L.role == "weak").sum())
    hi, mid = cfg.tier_high, cfg.tier_middle
    if len(cand) == 1:
        c = cand[0]
        if len(nbs) >= 2:
            tier = "many neighbours"
        elif c["p_cell"] >= hi["p_cell"] and c["p_nb"] <= hi["p_nb"] and (c["agrees_stringent"] or not hi["need_stringent"]):
            tier = "high"
        elif c["p_cell"] >= mid["p_cell"] and c["p_nb"] <= mid["p_nb"]:
            tier = "middle"
        else:
            tier = "low"
    elif len(cand) >= 2:
        tier = "several cells"
    else:
        tier = ("neighbours only" if nbs else "marker only" if marker_only else "low SNR" if n_weak else "empty")
    sig = {r: sum(c[f"sig_{r}"] for c in cand) for r in cfg.reporters}
    call = colour_call(sig, cfg.reporters, cfg.call_fold) if cand else "none"
    best = max(cand, key=lambda c: c["p_cell"] * (1 - c["p_nb"]), default=None)
    row = dict(tier=tier, call=call, n_cand=len(cand), n_marker_only=len(marker_only), n_nb=len(nbs), n_weak=n_weak,
               p_cell=best["p_cell"] if best else np.nan, p_nb=best["p_nb"] if best else np.nan,
               stringent_agrees=best["agrees_stringent"] if best else None,
               cand_chans=" | ".join(c["chans"] for c in cand), **{f"sig_{r}": round(v, 1) for r, v in sig.items()})
    cells = cand + marker_only + nbs
    for k, c in enumerate(cells):
        c["cell_id"] = k
        c["kind"] = "marker only" if (c["kind"] == "candidate" and not c["reporter"]) else c["kind"]
    member = {oid: c["cell_id"] for c in cells for oid in c["members"]}
    return row, cells, member
