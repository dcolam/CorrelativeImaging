"""Evaluation against the experiment table (col_data). The ONLY module that reads it; nothing here feeds back
into a call.

col_data: one row per well with plate_id, well_id (padded, B05) and free columns. For the ROMK benchmark the
ground truth is `Condition` (plate map: GFP / mScarlet / Mixed / Empty) and `Induction` (ephys: Hyperpolarized /
Depolarized / no-cell). Every hyperpolarized cell is ROMK-induced = GFP, so in hyperpolarized Mixed wells every
mScarlet call is an error.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def read_col_data(path) -> pd.DataFrame:
    cd = pd.read_csv(path, dtype={"plate_id": str, "well_id": str})
    cd["well_id"] = cd.well_id.map(lambda w: w[0] + f"{int(w[1:]):02d}")
    return cd


def _wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def romk_summary(calls: pd.DataFrame, col_data: pd.DataFrame, call_col: str, reporters=("GFP", "mScarlet"),
                 tiers=None) -> str:
    d = calls.merge(col_data, on=["plate_id", "well_id"], how="inner")
    if tiers:
        d = d[d.tier.isin(tiers)]
    rec = d.Induction.isin(["Hyperpolarized", "Depolarized"])
    pu = d[rec & d.Condition.isin(reporters) & d[call_col].isin(reporters)]
    mh = d[(d.Condition == "Mixed") & (d.Induction == "Hyperpolarized")]
    g, m = (mh[call_col] == reporters[0]).sum(), (mh[call_col] == reporters[1]).sum()
    lo, hi = _wilson(m, g + m)
    e = d[d.Condition == "Empty"]
    return (f"{call_col:9s} {'all tiers' if not tiers else '+'.join(tiers):12s} | pure recorded right "
            f"{(pu[call_col] == pu.Condition).sum()}/{len(pu)} | Mixed hyper {reporters[0]} {g} {reporters[1]} {m} "
            f"(errors {m / max(g + m, 1):.0%} [{lo:.0%}-{hi:.0%}], called {(g + m) / max(len(mh), 1):.0%}) | "
            f"Empty coloured {e[call_col].isin(reporters).sum()}/{len(e)}")


def report(calls: pd.DataFrame, col_data: pd.DataFrame, reference: pd.DataFrame | None = None) -> str:
    L = [romk_summary(calls, col_data, c, tiers=t) for c in ("call", "call_cal") for t in (None, ["high", "middle"])]
    d = calls.merge(col_data, on=["plate_id", "well_id"])
    d["group"] = np.where(d.Condition == "Empty", "Empty", d.Condition + " " + d.Induction.str[:5])
    L.append("\ncalibrated vs simple call (all wells):")
    L.append(pd.crosstab(d.call, d.call_cal).to_string())
    L.append("\ncalibrated call per group:")
    L.append(pd.crosstab(d.group, d.call_cal).to_string())
    if reference is not None:
        m = calls.merge(reference, on=["plate_id", "well_id"], suffixes=("", "_ref"))
        L.append(f"\nagainst reference ({len(m)} wells): tier same {(m.tier == m.tier_ref).sum()}, "
                 f"simple call same {(m.call == m.call_ref).sum()}")
        L.append(pd.crosstab(m.tier_ref, m.tier).to_string())
    return "\n".join(L)
