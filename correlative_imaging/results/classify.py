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

Everything here is keyed by CHANNEL NAME (the wavelength string in the DB, e.g.
``405nm``), which is the stable identity — channel↔colour assignment varies
between experiments, so colour is a pure display concern handled elsewhere
(:mod:`.labels` channel-display config + the Explorer). This module never names a
colour. The label store keys by the same channel names, so validation/calibration
is a direct per-channel join — no colour indirection.

Output schema (one row per plate+well):

    hole_present : bool
    score_<ch>   : float      combined enrichment score (population-scaled)
    occ_<ch>     : float       occupancy: fraction of hole ROI covered (0–1)
    pos_<ch>     : bool        score_<ch> >= threshold AND occ_<ch> >= min_area
    margin_<ch>  : float       score_<ch> - threshold_<ch> (signed confidence)
    pos_channels : str         comma-joined positive channel names, "" if none
    n_positive   : int
    is_negative_hole : bool    hole_present and n_positive == 0
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .analysis import RunTable

_EPS = 1e-6


@dataclass
class ClassifierParams:
    """Tunable knobs for the multi-label classifier (sensible defaults).

    ``pos_threshold`` is the default score a channel must clear to be called
    positive; ``thresholds`` overrides it per channel (what calibration against
    the hand labels fills in). ``min_area`` is an independent occupancy floor
    (fraction of the hole ROI covered by this channel's particles) — an
    intensity bump with no real object in the hole is rejected regardless of
    score, which is scale-robust by construction (occupancy is already 0–1).

    ``population_scale`` divides each channel's score by that channel's SD
    (uncentered) over the wells in ``scale_group``, so a fixed threshold means
    the same thing across a dim and a bright channel and across plates.
    ``scale_per_plate`` scopes that SD to each plate (batch-effect sanity check);
    the default (False) pools across all plates of the run — "across experiment",
    which is what preserves cross-plate/timepoint differences."""
    w_intensity: float = 1.0     # weight on hole/bg mean-intensity enrichment
    w_sum: float = 1.0           # weight on hole/bg sum-intensity enrichment
    w_particle: float = 1.0      # weight on brightest-particle enrichment
    pos_threshold: float = 0.10  # default per-channel positive-call threshold
    thresholds: dict[str, float] = field(default_factory=dict)  # per-channel overrides
    min_area: float = 0.0        # occupancy floor (0 = off); fraction of hole covered
    population_scale: bool = True   # divide score by per-channel SD (cross-well comparability)
    scale_per_plate: bool = False   # SD per (channel, plate) instead of per channel across the run

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


def _hole_occupancy(particles: pd.DataFrame, intensity: pd.DataFrame) -> dict:
    """{(plate, well, channel): occupancy} — fraction of the hole ROI covered by
    that channel's detected particles: Σ(particle area_px) / hole-ROI area_px.

    A 0–1 fraction (may slightly exceed 1 if particle masks overlap the ROI edge),
    inherently comparable across wells/plates. Missing pieces → occupancy absent
    (treated as 0 by the caller)."""
    if (particles is None or particles.empty
            or intensity is None or intensity.empty
            or "area_px" not in particles.columns or "area_px" not in intensity.columns):
        return {}
    p = particles[particles["roi_mask"] == "roi_hole"]
    roi = intensity[intensity["roi_mask"] == "roi_hole"]
    if p.empty or roi.empty:
        return {}
    part_area = p.groupby(["plate", "well_id", "channel"])["area_px"].sum()
    # ROI area is the same object per well, recorded per channel row — take the mean.
    roi_area = roi.groupby(["plate", "well_id", "channel"])["area_px"].mean()
    out = {}
    for key, num in part_area.items():
        den = roi_area.get(key)
        if den is not None and np.isfinite(den) and den > 0:
            out[key] = float(num) / float(den)
    return out


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


def _channel_scale_factors(raw: dict, params: ClassifierParams) -> dict:
    """Per-channel (or per channel×plate) uncentered SD used to population-scale
    scores. ``raw`` maps (plate, well, channel) → raw score. Returns a dict keyed
    by the same grouping as scaling: channel, or (channel, plate)."""
    if not params.population_scale or not raw:
        return {}
    groups: dict = {}
    for (plate, _well, ch), score in raw.items():
        key = (ch, plate) if params.scale_per_plate else ch
        groups.setdefault(key, []).append(score)
    factors = {}
    for key, vals in groups.items():
        arr = np.asarray(vals, dtype=float)
        sd = float(np.nanstd(arr))          # uncentered scale (R-style ÷SD)
        factors[key] = sd if np.isfinite(sd) and sd > 0 else 1.0
    return factors


def classify_wells(rt: RunTable, params: ClassifierParams | None = None) -> pd.DataFrame:
    """Return a per-well **multi-label** classification table: one independent
    positive/negative call per channel, an explicit negative-hole flag, and every
    per-channel score / occupancy / distance-from-threshold (for inspection and
    calibration). One row per (plate, well).

    A channel is positive when its (optionally population-scaled) score clears the
    threshold AND its occupancy clears ``min_area`` — two independent gates, so a
    bright-but-empty hole is rejected. Population scaling divides each channel's
    score by that channel's SD across the run (or per plate), so one threshold is
    comparable across dim/bright channels and across plates."""
    params = params or ClassifierParams()
    channels = rt.channels
    wells = rt.wells
    if wells.empty or not channels:
        return pd.DataFrame()

    pb = _hole_particle_brightness(rt.particles)         # brightest hole particle
    hole_sum = _roi_sum(rt.intensity, "roi_hole")        # extent signal
    bg_sum = _roi_sum(rt.intensity, "roi_background")
    occ = _hole_occupancy(rt.particles, rt.intensity)    # occupancy gate

    # ── Pass 1: raw per-(well, channel) scores + occupancy, and hole presence.
    raw_score: dict = {}
    occupancy: dict = {}
    has_hole: dict = {}
    for r in wells.itertuples(index=False):
        wkey = (r.plate, r.well_id)
        present = False
        for ch in channels:
            hole = getattr(r, f"hole_{ch}", None)
            bg = getattr(r, f"bg_{ch}", None)
            if hole is not None and np.isfinite(hole):
                present = True
            key = (r.plate, r.well_id, ch)
            comp = channel_score(
                hole, bg, hole_sum.get(key), bg_sum.get(key), pb.get(key), params,
            )
            raw_score[key] = comp["score"]
            occupancy[key] = float(occ.get(key, 0.0))
        has_hole[wkey] = present

    # ── Population scaling factors (per channel across the run, or per plate).
    factors = _channel_scale_factors(raw_score, params)

    def _scaled(plate, ch, score):
        if not params.population_scale:
            return score
        key = (ch, plate) if params.scale_per_plate else ch
        return score / factors.get(key, 1.0)

    # ── Pass 2: apply threshold + occupancy gates, assemble rows.
    rows = []
    for r in wells.itertuples(index=False):
        rec = {"plate": r.plate, "well_id": r.well_id,
               "row": getattr(r, "row", None), "col": getattr(r, "col", None),
               "experiment": getattr(r, "experiment", None)}
        positives = []
        for ch in channels:
            key = (r.plate, r.well_id, ch)
            score = _scaled(r.plate, ch, raw_score[key])
            occ_ch = occupancy[key]
            thr = params.threshold_for(ch)
            pos = (score >= thr) and (occ_ch >= params.min_area)
            rec[f"score_{ch}"] = score
            rec[f"occ_{ch}"] = occ_ch
            rec[f"pos_{ch}"] = bool(pos)
            rec[f"margin_{ch}"] = score - thr        # signed distance from threshold
            if pos:
                positives.append(ch)                 # channel name, never a colour

        if not has_hole[(r.plate, r.well_id)]:
            # No hole ROI at all — not a negative hole, just no data.
            for ch in channels:
                rec[f"pos_{ch}"] = False
            rec.update(hole_present=False, pos_channels="", n_positive=0,
                       is_negative_hole=False)
            rows.append(rec)
            continue

        rec.update(
            hole_present=True,
            pos_channels=",".join(positives),
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
