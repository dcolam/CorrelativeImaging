"""Load a run's per-plate databases into one tidy per-well table.

Pure pandas + sqlite — no Qt, no napari — so it's testable headless. The GUI
shell renders what this returns; it holds none of the answers itself.

Channel → colour is fixed by the acquisition (excitation wavelength), never
inferred: 405 nm = blue (e.g. BFP), 488 nm = green (e.g. GFP), 561 nm = red
(e.g. mCherry). Adjust :data:`CHANNEL_COLOR_NAME` if a project's fluorophores
differ.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .discovery import RunGroup

# Excitation wavelength → human colour name / hex. Lifted from the analysis
# notebook (notebooks/plate_results_analysis.ipynb) so GUI and notebook agree.
CHANNEL_COLOR_NAME = {"405nm": "blue", "488nm": "green", "561nm": "red"}
CHANNEL_HEX = {"405nm": "#3b6bff", "488nm": "#2fbf3c", "561nm": "#ff3b3b"}

# The ROI whose intensities decide "what colour is the cell in the hole".
HOLE_ROI = "roi_hole"
# The ROI for "cells around the hole".
BACKGROUND_ROI = "roi_background"


@dataclass
class RunTable:
    """A loaded run: one row per (plate, well) in ``wells``, plus the raw
    long-form measurement frames for drill-down."""
    run_tag: str
    wells: pd.DataFrame          # one row per plate+well, derived columns below
    intensity: pd.DataFrame      # long: plate, well, channel, roi_mask, mean/sum/...
    particles: pd.DataFrame      # long: plate, well, channel, roi_mask, per-particle
    channels: list[str]          # distinct channel names seen, e.g. ["405nm","488nm","561nm"]
    coloc: pd.DataFrame = field(default_factory=pd.DataFrame)  # long: plate, well, channel-pair, roi_mask, manders/pearson


def _read_table(con, name: str) -> pd.DataFrame:
    try:
        return pd.read_sql(f"SELECT * FROM {name}", con)
    except Exception:
        return pd.DataFrame()


def load_run(run: RunGroup) -> RunTable:
    """Load every plate database of *run* into unified long-form frames plus a
    derived per-well summary table.

    Each row is tagged with ``plate`` (from :class:`~.discovery.PlateResult`)
    and ``experiment`` (from the images table), so tables from several plates —
    or later, several experiments — concatenate cleanly and stay distinguishable
    (the basis for cross-plate/experiment aggregation and batch correction).
    """
    import sqlite3

    img_frames, int_frames, part_frames, coloc_frames = [], [], [], []
    for pr in run.plates:
        con = sqlite3.connect(f"file:{pr.db_path}?mode=ro", uri=True)
        try:
            images = _read_table(con, "images")
            intensity = _read_table(con, "intensity_measurements")
            particles = _read_table(con, "particle_measurements")
            coloc = _read_table(con, "colocalization_results")
        finally:
            con.close()
        if images.empty:
            continue

        # Expand each image's JSON metadata into well_id/row/col columns.
        meta = images["metadata"].apply(_safe_json)
        images = images.assign(
            plate=pr.plate,
            well_id=meta.apply(lambda m: m.get("well_id")),
            row=meta.apply(lambda m: m.get("row")),
            col=meta.apply(lambda m: m.get("col")),
        )
        # Map measurement rows (keyed by image_id) to plate + well.
        id2well = dict(zip(images["id"], images["well_id"]))
        id2exp = dict(zip(images["id"], images.get("experiment", pd.Series(dtype=str))))
        for frame in (intensity, particles, coloc):
            if not frame.empty:
                frame["plate"] = pr.plate
                frame["well_id"] = frame["image_id"].map(id2well)
                frame["experiment"] = frame["image_id"].map(id2exp)

        img_frames.append(images)
        if not intensity.empty:
            int_frames.append(intensity)
        if not particles.empty:
            part_frames.append(particles)
        if not coloc.empty:
            coloc_frames.append(coloc)

    images_all = pd.concat(img_frames, ignore_index=True) if img_frames else pd.DataFrame()
    intensity_all = pd.concat(int_frames, ignore_index=True) if int_frames else pd.DataFrame()
    particles_all = pd.concat(part_frames, ignore_index=True) if part_frames else pd.DataFrame()
    coloc_all = pd.concat(coloc_frames, ignore_index=True) if coloc_frames else pd.DataFrame()

    channels = (
        sorted(intensity_all["channel"].dropna().unique().tolist())
        if not intensity_all.empty else []
    )
    wells = _build_well_table(images_all, intensity_all, particles_all, channels)
    return RunTable(
        run_tag=run.run_tag,
        wells=wells,
        intensity=intensity_all,
        particles=particles_all,
        channels=channels,
        coloc=coloc_all,
    )


def _safe_json(s) -> dict:
    import json
    if not s:
        return {}
    try:
        return json.loads(s)
    except (ValueError, TypeError):
        return {}


def _build_well_table(
    images: pd.DataFrame, intensity: pd.DataFrame,
    particles: pd.DataFrame, channels: list[str],
) -> pd.DataFrame:
    """One row per (plate, well) with derived per-channel hole/background
    intensities, particle counts, and a *naive* dominant colour + confidence
    margin (idxmax over hole intensities). The naive call is a provisional
    sanity view only — the tunable, negative-hole-aware classifier is a later
    phase; this just answers "does the plate look sane?".

    Vectorised: all per-(plate, well, roi, channel) values come from a handful
    of groupby/pivot passes over the whole frame, not per-well filtering (which
    is quadratic — ~30 s for 1200 wells; this is sub-second).
    """
    if images.empty:
        return pd.DataFrame()

    # Base: one row per (plate, well) from the images table.
    base = (
        images.dropna(subset=["well_id"])
        .groupby(["plate", "well_id"], as_index=False)
        .agg(row=("row", "first"), col=("col", "first"),
             experiment=("experiment", "first"))
    )

    # Mean intensity pivoted to columns hole_<ch> / bg_<ch> in one pass each.
    def _intensity_pivot(roi: str, prefix: str) -> pd.DataFrame:
        if intensity.empty:
            return pd.DataFrame()
        sub = intensity[intensity["roi_mask"] == roi]
        if sub.empty:
            return pd.DataFrame()
        piv = sub.pivot_table(
            index=["plate", "well_id"], columns="channel",
            values="mean_intensity", aggfunc="mean",
        )
        piv.columns = [f"{prefix}{c}" for c in piv.columns]
        return piv.reset_index()

    # Particle counts per (plate, well[, channel]) via groupby size.
    def _particle_counts(roi: str, by_channel: bool):
        if particles.empty:
            return pd.DataFrame()
        sub = particles[particles["roi_mask"] == roi]
        if sub.empty:
            return pd.DataFrame()
        if by_channel:
            g = (sub.groupby(["plate", "well_id", "channel"]).size()
                 .unstack("channel", fill_value=0))
            g.columns = [f"nbg_{c}" for c in g.columns]
            return g.reset_index()
        # total particles in the hole ROI, counting each label once
        g = (sub.groupby(["plate", "well_id"])["label"].nunique()
             .rename("n_particles_hole"))
        return g.reset_index()

    df = base
    for frame in (
        _intensity_pivot(HOLE_ROI, "hole_"),
        _intensity_pivot(BACKGROUND_ROI, "bg_"),
        _particle_counts(HOLE_ROI, by_channel=False),
        _particle_counts(BACKGROUND_ROI, by_channel=True),
    ):
        if not frame.empty:
            df = df.merge(frame, on=["plate", "well_id"], how="left")

    # Ensure expected columns exist even if a ROI/channel was absent.
    for ch in channels:
        for col in (f"hole_{ch}", f"bg_{ch}", f"nbg_{ch}"):
            if col not in df.columns:
                df[col] = pd.NA
    if "n_particles_hole" not in df.columns:
        df["n_particles_hole"] = 0
    df["n_particles_hole"] = df["n_particles_hole"].fillna(0).astype(int)
    for ch in channels:
        df[f"nbg_{ch}"] = df[f"nbg_{ch}"].fillna(0).astype(int)

    # ── Enrichment over background ──────────────────────────────────
    # The cell colour is decided by how much brighter the hole is than its
    # surroundings PER CHANNEL, not by raw hole brightness — a channel that's
    # bright everywhere (autofluorescence, uneven illumination) would otherwise
    # win the hole spuriously. enrich_<ch> = hole_<ch> - bg_<ch> (local, this
    # well's own background ROI). A hole not enriched over background in ANY
    # channel is a negative / empty hole.
    for ch in channels:
        hole = pd.to_numeric(df[f"hole_{ch}"], errors="coerce")
        bg = pd.to_numeric(df[f"bg_{ch}"], errors="coerce")
        df[f"enrich_{ch}"] = hole - bg

    enrich_cols = [f"enrich_{ch}" for ch in channels]
    evals = df[enrich_cols]
    top = evals.max(axis=1)                        # NaN when no hole ROI at all
    valid = evals.notna().any(axis=1)
    # A real colour call needs positive enrichment over background; otherwise
    # it's a negative/empty hole (no cell brighter than surroundings).
    is_colored = valid & top.notna() & (top > 0)

    dom_ch = pd.Series(pd.NA, index=df.index, dtype=object)
    if valid.any():
        dom_ch.loc[valid] = (
            evals.loc[valid].idxmax(axis=1).str.replace("enrich_", "", regex=False)
        )
    # second-highest enrichment per row (for the confidence margin)
    second = evals.apply(
        lambda r: r.nlargest(2).iloc[-1] if r.notna().sum() >= 2 else 0.0, axis=1
    )
    df["dominant_channel"] = dom_ch.where(is_colored)
    df["dominant_color"] = df["dominant_channel"].map(CHANNEL_COLOR_NAME).where(is_colored)
    df["dominant_enrichment"] = top.where(is_colored)
    df["margin"] = ((top - second) / top).where(is_colored & (top > 0))
    # Explicit negative-hole flag: a hole exists but no channel is enriched.
    df["is_negative_hole"] = valid & ~is_colored

    df = df.sort_values(
        ["plate", "row", "col"],
        key=lambda s: s if s.name != "col" else s.astype("Int64"),
    ).reset_index(drop=True)
    return df
