"""Holistic hole-colour classification from all available pipeline measurements.

Phase 2 (rule-based, transparent, tunable). The call is **multi-label**: each
channel is decided *independently* — a hole can be green AND red, or none. This
matches the biology (GFAP/Dlx/Camk2a are largely distinct cell-type markers, and
a hole may hold several cells of different types *or* a single co-expressing
cell — the hole-level call does not try to distinguish those). It replaces the
earlier argmax "one dominant colour" model, which structurally could not
represent a multi-positive hole.

For each (well, channel) it combines several INDEPENDENT lines of evidence that
a bright cell of that colour sits in the hole, rather than one raw intensity:

* intensity enrichment   hole_mean / bg_mean   (ratio, cancels per-channel gain)
* extent enrichment      hole_sum  / bg_sum     (a bigger/brighter object scores more)
* particle evidence      brightest detected particle in the hole vs background

Each signal is a non-negative log-enrichment; the weighted sum is that channel's
``score``. A channel is called **positive** when its score clears that channel's
threshold. A hole present but positive in NO channel is a negative / empty hole.
Every component score and the per-channel distance-from-threshold are returned so
the decision is inspectable — and this same table is what calibration fits its
thresholds to, and what the Phase-3 Random Forest will train on.

Ratios, not differences (per the user): a channel bright everywhere shouldn't
win the hole, and per-channel gain differences must cancel.

Output schema (one row per plate+well):

    hole_present : bool
    score_<ch>   : float      combined enrichment score
    pos_<ch>     : bool        score_<ch> >= threshold_<ch>
    margin_<ch>  : float       score_<ch> - threshold_<ch> (signed confidence)
    colors       : str         comma-joined positive colour names, "" if none
    n_positive   : int
    is_negative_hole : bool    hole_present and n_positive == 0

Note the classifier keys its per-channel columns by CHANNEL (``pos_405nm``),
while the label store (:mod:`.labels`) keys by COLOUR (``pos_blue``). They are
NOT positionally interchangeable — the bridge is the ``colors`` field / the
:data:`~.analysis.CHANNEL_COLOR_NAME` map. Validation and calibration must join
channel→colour through that map, never by column position.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .analysis import CHANNEL_COLOR_NAME, RunTable

_EPS = 1e-6


@dataclass
class ClassifierParams:
    """Tunable knobs for the multi-label classifier (sensible defaults).

    ``pos_threshold`` is the default score a channel must clear to be called
    positive; ``thresholds`` overrides it per channel (this is what calibration
    against the hand labels fills in — one threshold per channel)."""
    w_intensity: float = 1.0     # weight on hole/bg mean-intensity enrichment
    w_sum: float = 1.0           # weight on hole/bg sum-intensity enrichment
    w_particle: float = 1.0      # weight on brightest-particle enrichment
    pos_threshold: float = 0.10  # default per-channel positive-call threshold
    thresholds: dict[str, float] = field(default_factory=dict)  # per-channel overrides

    def threshold_for(self, channel: str) -> float:
        return float(self.thresholds.get(channel, self.pos_threshold))


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


def channel_score(hole, bg, hole_sum, bg_sum, part, params: ClassifierParams) -> dict:
    """The three enrichment components and their weighted sum for one channel.

    Returns the components too, so the score is fully inspectable and calibration
    can, if wanted, re-weight without recomputing enrichments."""
    s_int = _log_enrich(hole, bg)                 # mean intensity
    s_sum = _log_enrich(hole_sum, bg_sum)          # extent (sum)
    s_part = _log_enrich(part, bg)                 # brightest particle
    score = (params.w_intensity * s_int
             + params.w_sum * s_sum
             + params.w_particle * s_part)
    return {"s_int": s_int, "s_sum": s_sum, "s_part": s_part, "score": score}


def classify_wells(rt: RunTable, params: ClassifierParams | None = None) -> pd.DataFrame:
    """Return a per-well **multi-label** classification table: one independent
    positive/negative call per channel, an explicit negative-hole flag, and every
    per-channel component score + distance-from-threshold (for inspection and
    calibration). One row per (plate, well)."""
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
        has_hole = False
        positives = []
        for ch in channels:
            hole = getattr(r, f"hole_{ch}", None)
            bg = getattr(r, f"bg_{ch}", None)
            if hole is not None and np.isfinite(hole):
                has_hole = True
            key = (r.plate, r.well_id, ch)
            comp = channel_score(
                hole, bg, hole_sum.get(key), bg_sum.get(key), pb.get(key), params,
            )
            thr = params.threshold_for(ch)
            score = comp["score"]
            pos = score >= thr
            rec[f"score_{ch}"] = score
            rec[f"pos_{ch}"] = bool(pos)
            rec[f"margin_{ch}"] = score - thr        # signed distance from threshold
            if pos:
                positives.append(CHANNEL_COLOR_NAME.get(ch, ch))

        if not has_hole:
            # No hole ROI at all — not a negative hole, just no data.
            for ch in channels:
                rec[f"pos_{ch}"] = False
            rec.update(hole_present=False, colors="", n_positive=0,
                       is_negative_hole=False)
            rows.append(rec)
            continue

        rec.update(
            hole_present=True,
            colors=",".join(positives),
            n_positive=len(positives),
            is_negative_hole=(len(positives) == 0),
        )
        rows.append(rec)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(
            ["plate", "row", "col"],
            key=lambda s: s if s.name != "col" else s.astype("Int64"),
        ).reset_index(drop=True)
    return df
