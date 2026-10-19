"""Settings of the downstream analyzer. Every dataset-specific choice lives here, nothing is hard-coded elsewhere.

Load from JSON with :func:`load_config`; missing keys keep the defaults below (the ROMK settings, 2026-10-15).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Config:
    # raw channel name in the DB (excitation) -> display name; order = channel order in the crop tif
    channels: dict = field(default_factory=lambda: {"488nm": "GFP", "561nm": "mScarlet", "640nm": "ROMK"})
    reporters: list = field(default_factory=lambda: ["GFP", "mScarlet"])     # identity evidence
    markers: list = field(default_factory=lambda: ["ROMK"])                  # measured, never identity evidence
    colors: dict = field(default_factory=lambda: {"GFP": "#00ff00", "mScarlet": "#ff00ff", "ROMK": "#3a7bff",
                                                  "BF": "#ffffff"})
    # bleed-through: a particle of `into` is dropped when into < 10**max_log_ratio x `from` over its footprint
    bleed: list = field(default_factory=lambda: [{"from": "mScarlet", "into": "GFP", "max_log_ratio": -1.0}])

    # particles (as pipeline.segment.AutoThreshold / ParticleAnalysis)
    min_object_px: int = 50          # remove_small_objects before labelling
    min_inside_um2: float = 0.5      # a piece inside the hole must be at least this big

    # roles, in this order: neighbour -> bleed -> signal
    neighbour_share_max: float = 0.5   # less than half of the particle inside the hole ...
    neighbour_fill_max: float = 0.25   # ... AND covers less than a quarter of the hole
    p_candidate: float = 0.5
    p_weak: float = 0.2

    # cells + tier
    cell_gap_um: float = 2.0
    tier_high: dict = field(default_factory=lambda: {"p_cell": 0.95, "p_nb": 0.2, "need_stringent": True})
    tier_middle: dict = field(default_factory=lambda: {"p_cell": 0.8, "p_nb": 0.5})
    call_fold: float = 3.0           # exclusive call: top reporter >= fold x the next
    call_mode: str = "exclusive"     # "exclusive" (ROMK) | "multilabel" (not implemented yet)

    # refinement (seeded, competing growth)
    refine_frac: float = 0.35        # a grown pixel must reach this fraction of its seed's median signal ...
    refine_floor_snr: float = 2.0    # ... and at least this SNR above background
    refine_margin_um: float = 1.5    # the patched cell may grow this far beyond the hole (3 um took in the rim halo)
    refine_max_area_x_hole: float = 2.0
    refine_smooth_px: float = 1.0
    body_channels: list | None = None   # channels that define the cell body; None = reporters

    # calibrated colour call
    calib_min_field_um2: float = 30.0   # field cells used as the per plate x channel reference
    calib_p_call: float = 0.8           # P(colour) needed for a calibrated call
    calib_bleed_quantile: float = 0.25  # bleed coefficient = this quantile of into/from over bright field cells
                                        # (low quantile: robust to real double-positive cells)
    calib_eps: float = 0.02             # floor in field-cell units (2 % of a typical cell) for the colour ratio

    # particle models (P(cell), P(neighbour)); relative paths are resolved next to this file
    models: str = "models/romk_2026-10-14.json"
    # Windows drive -> local mount, for roi paths stored in the DB
    path_map: dict = field(default_factory=lambda: {"Z:\\": "/mnt/z/"})

    @property
    def body(self) -> list:
        return self.body_channels or self.reporters

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def load_config(path: str | Path | None = None) -> Config:
    cfg = Config()
    if path:
        for k, v in json.loads(Path(path).read_text()).items():
            if not hasattr(cfg, k):
                raise KeyError(f"unknown config key: {k}")
            setattr(cfg, k, v)
    return cfg
