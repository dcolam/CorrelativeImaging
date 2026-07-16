"""Holistic hole-colour classification from all available pipeline measurements.

Phase 2 (rule-based, transparent, tunable). For each well it combines several
INDEPENDENT lines of evidence — per channel — that a bright cell of that colour
sits in the hole, rather than relying on a single raw intensity:

* intensity enrichment   hole_mean / bg_mean   (ratio, cancels per-channel gain)
* extent enrichment      hole_sum  / bg_sum     (a bigger/brighter object scores more)
* particle evidence      brightest detected particle in the hole vs background

Each signal is turned into a non-negative log-enrichment, combined with tunable
weights into a per-channel score; the top score is the call, and a hole with no
channel clearing ``min_score`` is a negative / empty hole. All component scores
are returned so the decision is inspectable — and this same feature table is
what the Phase-3 Random Forest will train on (plus image/BF features).

Ratios, not differences (per the user): a channel bright everywhere shouldn't
win the hole, and per-channel gain differences must cancel.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .analysis import CHANNEL_COLOR_NAME, RunTable

_EPS = 1e-6


@dataclass
class ClassifierParams:
    """Tunable knobs for the holistic classifier (sensible defaults)."""
    w_intensity: float = 1.0     # weight on hole/bg mean-intensity enrichment
    w_sum: float = 1.0           # weight on hole/bg sum-intensity enrichment
    w_particle: float = 1.0      # weight on brightest-particle enrichment
    min_score: float = 0.10      # top channel score below this → negative/empty hole
    ambiguous_margin: float = 0.15   # confidence margin below this → flagged ambiguous


def _log_enrich(hole, bg) -> float:
    """Non-negative log enrichment of hole over background; 0 when not enriched
    or data missing. log keeps ratios symmetric and bounded."""
    if hole is None or bg is None or not np.isfinite(hole) or not np.isfinite(bg):
        return 0.0
    ratio = (float(hole) + _EPS) / (float(bg) + _EPS)
    return float(max(0.0, np.log(ratio)))


def _hole_particle_brightness(particles: pd.DataFrame) -> dict:
    """{(plate, well, channel): brightest particle mean-intensity in the hole}."""
    if particles is None or particles.empty:
        return {}
    sub = particles[particles["roi_mask"] == "roi_hole"]
    if sub.empty:
        return {}
    g = sub.groupby(["plate", "well_id", "channel"])["mean_intensity"].max()
    return g.to_dict()


def _roi_sum(intensity: pd.DataFrame, roi: str) -> dict:
    """{(plate, well, channel): sum_intensity in *roi*} — the real extent signal."""
    if intensity is None or intensity.empty or "sum_intensity" not in intensity.columns:
        return {}
    sub = intensity[intensity["roi_mask"] == roi]
    if sub.empty:
        return {}
    g = sub.groupby(["plate", "well_id", "channel"])["sum_intensity"].mean()
    return g.to_dict()


def classify_wells(rt: RunTable, params: ClassifierParams | None = None) -> pd.DataFrame:
    """Return a per-well classification table with the dominant colour, a
    confidence margin, a negative/empty-hole flag, and every per-channel
    component score (for inspection). One row per (plate, well)."""
    params = params or ClassifierParams()
    channels = rt.channels
    wells = rt.wells
    if wells.empty or not channels:
        return pd.DataFrame()

    pb = _hole_particle_brightness(rt.particles)         # brightest hole particle
    hole_sum = _roi_sum(rt.intensity, "roi_hole")        # extent signal
    bg_sum = _roi_sum(rt.intensity, "roi_background")

    rows = []
    for r in wells.itertuples(index=False):
        rec = {"plate": r.plate, "well_id": r.well_id,
               "row": getattr(r, "row", None), "col": getattr(r, "col", None),
               "experiment": getattr(r, "experiment", None)}
        scores = {}
        has_hole = False
        for ch in channels:
            hole = getattr(r, f"hole_{ch}", None)
            bg = getattr(r, f"bg_{ch}", None)
            if hole is not None and np.isfinite(hole):
                has_hole = True
            key = (r.plate, r.well_id, ch)
            # three genuinely independent enrichment signals (all ratios):
            s_int = _log_enrich(hole, bg)                                  # mean intensity
            s_sum = _log_enrich(hole_sum.get(key), bg_sum.get(key))        # extent (sum)
            s_part = _log_enrich(pb.get(key), bg)                          # brightest particle
            score = (params.w_intensity * s_int
                     + params.w_sum * s_sum
                     + params.w_particle * s_part)
            scores[ch] = score
            rec[f"score_{ch}"] = score

        if not has_hole:
            rec.update(dominant_channel=None, dominant_color=None,
                       confidence=None, is_negative_hole=False, is_ambiguous=False,
                       hole_present=False)
            rows.append(rec)
            continue

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top_ch, top = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        negative = top < params.min_score
        margin = (top - second) / top if top > 0 else 0.0
        rec.update(
            hole_present=True,
            dominant_channel=None if negative else top_ch,
            dominant_color=None if negative else CHANNEL_COLOR_NAME.get(top_ch, top_ch),
            dominant_score=top,
            confidence=None if negative else margin,
            is_negative_hole=bool(negative),
            is_ambiguous=bool(not negative and margin < params.ambiguous_margin),
        )
        rows.append(rec)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(
            ["plate", "row", "col"],
            key=lambda s: s if s.name != "col" else s.astype("Int64"),
        ).reset_index(drop=True)
    return df
