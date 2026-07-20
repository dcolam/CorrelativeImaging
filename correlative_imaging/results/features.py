"""Per-well feature catalog — the shared feature source for clustering (and,
later, supervised RF/LDA).

A "cell" is a well's hole. Per-particle measurements can't be features directly
for a per-well embedding, so each raw DB column is summarised into named columns.
The catalog is grouped by source so the UI can offer a curated pick-list rather
than an unbounded column × stat × ROI cross-product:

* ``bulk intensity``   — ``intensity_measurements``: mean / sum / std per channel
* ``occupancy``        — Σ(hole particle area) / hole-ROI area, per channel (0–1)
* ``particle count``   — # detected particles in the hole, per channel
* ``particle shape``   — mean + std of area/circularity/eccentricity/solidity
* ``particle intensity`` — mean + std of per-particle mean/max intensity
* ``colocalization``   — Pearson + background-corrected Manders (``m1 − m1_random``)
                         per channel PAIR (the point of correlative imaging)

``std`` is included because dispersion carries heterogeneity signal (a hole with
mixed cell sizes vs uniform) that the mean discards. Columns are named
``<prefix>_<attr>_<channel>`` (per channel) or ``coloc_<metric>_<a>-<b>`` (per
pair). ROI defaults to the hole; background columns are offered too so callers
can build hole-vs-background enrichment if wanted.

Everything here is plain pandas — no Qt — so it is testable headless.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .analysis import RunTable

# Per-particle attributes to summarise, split into shape vs intensity groups.
_SHAPE_ATTRS = ["area_um2", "perimeter_px", "circularity", "eccentricity", "solidity"]
_PARTINT_ATTRS = ["mean_intensity", "max_intensity"]
_PARTICLE_STATS = ["mean", "std"]     # count is added separately


@dataclass
class FeatureCatalog:
    """Wide per-well feature table + column metadata.

    ``df`` has id columns (``run_tag``, ``plate``, ``well_id``) plus one column
    per feature; ``groups`` maps each feature column → its source group (for the
    pick-list); ``columns`` is the ordered feature-column list."""
    df: pd.DataFrame
    groups: dict[str, str]
    columns: list[str]

    def default_selection(self) -> list[str]:
        """A sane starting subset (R-matching): bulk hole mean + occupancy per
        channel, plus the Pearson coloc metrics. Everything else is opt-in."""
        keep = []
        for c in self.columns:
            g = self.groups.get(c)
            if g == "occupancy":
                keep.append(c)
            elif g == "bulk intensity" and c.startswith("int_mean_"):
                keep.append(c)
            elif g == "colocalization" and c.startswith("coloc_pearson_"):
                keep.append(c)
        return keep or list(self.columns)


def _pivot_channel(df: pd.DataFrame, value_cols: list[str], prefix: str,
                   groups: dict, group_name: str) -> pd.DataFrame:
    """Pivot a per-(plate,well,channel) frame to wide ``<prefix>_<val>_<ch>``
    columns, recording each column's group."""
    if df.empty:
        return pd.DataFrame()
    wide = df.pivot_table(index=["plate", "well_id"], columns="channel",
                          values=value_cols)
    wide.columns = [f"{prefix}_{val}_{ch}" for val, ch in wide.columns]
    for c in wide.columns:
        groups[c] = group_name
    return wide.reset_index()


def filter_particles(particles: pd.DataFrame, roi: str, method: str = "zscore",
                     threshold: float | None = None,
                     group_vars=("channel", "plate")) -> pd.DataFrame:
    """Drop background / out-of-focus particles before aggregation, ported from
    ephacRTools ``filterParticles``. Scales each particle's ``mean_intensity``
    within ``group_vars`` and keeps those above a cut-off:

    * ``"zscore"``       — standardise (mean 0, sd 1); keep ``≥ threshold``
      (default 0, i.e. above the group mean).
    * ``"median_ratio"`` — uncentered ÷SD; keep ``≥ threshold``; ``threshold``
      ``None`` → each group's ``median(scaled)/3``.

    Returns the roi-subset with rejected particles removed."""
    p = particles[particles["roi_mask"] == roi].copy()
    if p.empty or "mean_intensity" not in p.columns:
        return p
    gv = [g for g in group_vars if g in p.columns]

    def _sd(x):
        s = x.std(ddof=0)
        return s if s and np.isfinite(s) else 1.0

    if method == "median_ratio":
        scaled = p.groupby(gv)["mean_intensity"].transform(lambda x: x / _sd(x))
        if threshold is None:
            thr = p.assign(_s=scaled).groupby(gv)["_s"].transform("median") / 3.0
            keep = scaled >= thr
        else:
            keep = scaled >= threshold
    else:  # zscore
        scaled = p.groupby(gv)["mean_intensity"].transform(lambda x: (x - x.mean()) / _sd(x))
        keep = scaled >= (0.0 if threshold is None else threshold)
    return p[keep.fillna(False)]


def build_catalog(rt: RunTable, roi: str = "roi_hole",
                  particle_filter: dict | None = None) -> FeatureCatalog:
    """Build the per-well feature catalog for one run (all its plates).

    ``particle_filter`` (e.g. ``{"method": "zscore", "threshold": 0.0}``) applies
    :func:`filter_particles` before the particle-based aggregates (occupancy,
    count, shape, intensity), so those features are computed over accepted
    particles only — as in the R figures. Bulk intensity + colocalization are not
    particle-based and are unaffected."""
    channels = rt.channels
    groups: dict[str, str] = {}
    base = None

    # ── bulk intensity (mean/sum/std) per channel ────────────────────
    if rt.intensity is not None and not rt.intensity.empty:
        sub = rt.intensity[rt.intensity["roi_mask"] == roi]
        if not sub.empty:
            agg = (sub.groupby(["plate", "well_id", "channel"])
                   .agg(mean=("mean_intensity", "mean"),
                        sum=("sum_intensity", "mean"),
                        std=("std_intensity", "mean")).reset_index())
            w = _pivot_channel(agg, ["mean", "sum", "std"], "int", groups, "bulk intensity")
            base = w if base is None else base.merge(w, on=["plate", "well_id"], how="outer")

    # ── particle aggregates: occupancy, count, shape, intensity ──────
    if rt.particles is not None and not rt.particles.empty:
        if particle_filter:
            p = filter_particles(rt.particles, roi,
                                 particle_filter.get("method", "zscore"),
                                 particle_filter.get("threshold"))
        else:
            p = rt.particles[rt.particles["roi_mask"] == roi]
        roi_area = None
        if rt.intensity is not None and not rt.intensity.empty:
            ri = rt.intensity[rt.intensity["roi_mask"] == roi]
            if not ri.empty:
                roi_area = (ri.groupby(["plate", "well_id", "channel"])["area_px"]
                            .mean().rename("roi_area").reset_index())
        if not p.empty:
            # occupancy + count
            occ = (p.groupby(["plate", "well_id", "channel"])
                   .agg(part_area_sum=("area_px", "sum"),
                        n_particles=("label", "count")).reset_index())
            if roi_area is not None:
                occ = occ.merge(roi_area, on=["plate", "well_id", "channel"], how="left")
                occ["occ"] = occ["part_area_sum"] / occ["roi_area"].replace(0, np.nan)
            else:
                occ["occ"] = np.nan
            w = _pivot_channel(occ[["plate", "well_id", "channel", "occ"]],
                               ["occ"], "occ", groups, "occupancy")
            # rename occ_occ_<ch> → occ_<ch> for readability
            w.columns = [c.replace("occ_occ_", "occ_") for c in w.columns]
            groups = {(k.replace("occ_occ_", "occ_")): v for k, v in groups.items()}
            base = w if base is None else base.merge(w, on=["plate", "well_id"], how="outer")
            wc = _pivot_channel(occ[["plate", "well_id", "channel", "n_particles"]],
                                ["n_particles"], "part", groups, "particle count")
            wc.columns = [c.replace("part_n_particles_", "part_count_") for c in wc.columns]
            groups = {(k.replace("part_n_particles_", "part_count_")): v for k, v in groups.items()}
            base = base.merge(wc, on=["plate", "well_id"], how="outer")

            # shape + intensity aggregates (mean + std)
            for attrs, grp, pref in ((_SHAPE_ATTRS, "particle shape", "shape"),
                                     (_PARTINT_ATTRS, "particle intensity", "pint")):
                present = [a for a in attrs if a in p.columns]
                if not present:
                    continue
                agg = p.groupby(["plate", "well_id", "channel"])[present].agg(_PARTICLE_STATS)
                agg.columns = [f"{stat}__{attr}" for attr, stat in agg.columns]
                agg = agg.reset_index()
                w = _pivot_channel(agg, list(agg.columns[3:]), pref, groups, grp)
                # column names: <pref>_<stat>__<attr>_<ch>
                base = base.merge(w, on=["plate", "well_id"], how="outer")

    # ── colocalization per channel PAIR (background-corrected Manders) ─
    if rt.coloc is not None and not rt.coloc.empty:
        cc = rt.coloc[rt.coloc["roi_mask"] == roi].copy()
        if not cc.empty:
            cc["pair"] = cc["primary_channel"].astype(str) + "-" + cc["secondary_channel"].astype(str)
            cc["manders1_corr"] = cc["manders_m1"] - cc.get("manders_m1_random", 0)
            cc["manders2_corr"] = cc["manders_m2"] - cc.get("manders_m2_random", 0)
            metrics = {"pearson": "pearson_r", "manders1_corr": "manders1_corr",
                       "manders2_corr": "manders2_corr"}
            g = cc.groupby(["plate", "well_id", "pair"]).agg(
                {v: "mean" for v in metrics.values()}).reset_index()
            wide = g.pivot_table(index=["plate", "well_id"], columns="pair",
                                 values=list(metrics.values()))
            rev = {v: k for k, v in metrics.items()}
            wide.columns = [f"coloc_{rev[val]}_{pair}" for val, pair in wide.columns]
            for c in wide.columns:
                groups[c] = "colocalization"
            wide = wide.reset_index()
            base = wide if base is None else base.merge(wide, on=["plate", "well_id"], how="outer")

    if base is None or base.empty:
        return FeatureCatalog(pd.DataFrame(columns=["run_tag", "plate", "well_id"]), {}, [])

    base.insert(0, "run_tag", rt.run_tag)
    feat_cols = [c for c in base.columns if c not in ("run_tag", "plate", "well_id")]
    # order columns by group for a tidy pick-list
    order = ["bulk intensity", "occupancy", "particle count", "particle shape",
             "particle intensity", "colocalization"]
    feat_cols.sort(key=lambda c: (order.index(groups.get(c, "")) if groups.get(c) in order else 99, c))
    base = base[["run_tag", "plate", "well_id"] + feat_cols]
    return FeatureCatalog(df=base, groups=groups, columns=feat_cols)
