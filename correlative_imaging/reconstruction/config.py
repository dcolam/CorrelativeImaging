"""Settings of the particle-identity tool. Nothing dataset-specific is hard-coded anywhere else.

Channels are keyed by their raw name in the run (the excitation wavelength, e.g. ``488nm``). Display names and colours
are a display layer only (``display``); no decision ever looks at them. The defaults below are neutral: every
fluorescence channel of the run is an identity channel, none is a marker. Dataset presets live in ``presets/``.

Load from JSON with :func:`load_config`; missing keys keep the defaults.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

PRESETS = Path(__file__).parent / "presets"
# settings that define the particle features: a model is only valid for detections made with the same values
FEATURE_KEYS = ["work_R", "context_R", "min_object_px", "min_inside_um2", "field_min_um2", "smooth_px", "peak_h",
                "split_particles", "split_dip_ratio", "split_max_x_hole"]
MODELS = Path(__file__).parent / "models"


@dataclass
class Config:
    # --- channels -----------------------------------------------------------------------------------------------
    channels: list = field(default_factory=list)   # fluorescence channels to use; empty = all found in the run
    markers: list = field(default_factory=list)    # measured, never identity evidence (e.g. a channel on every cell)
    display: dict = field(default_factory=dict)    # channel -> {"name": str, "color": "#rrggbb"}; display only

    # --- detection (as pipeline.segment: threshold, remove_small_objects, label) --------------------------------
    work_R: float = 2.2           # analysis window: +- this many hole radii (clipped to the frame). Whole-object
                                  # features (full area, edge, parent share) depend on it: keep it equal across the
                                  # datasets a model is trained and used on (the first ROMK labels: 2.2)
    context_R: float = 2.2        # context features (neighbour density, ranks) use a disc of this many hole radii
    min_object_px: int = 50       # remove_small_objects(max_size=...) before labelling, as the pipeline
    min_inside_um2: float = 0.5   # a particle must have at least this much area inside the hole
    field_min_um2: float = 30.0   # whole-image objects at least this big describe a typical (unpatched) cell
    smooth_px: float = 1.0        # light smoothing for peaks and growth (not for measurements)
    split_particles: bool = False # split every particle at signal dips at Detect. Off by default: on ROMK it split
                                  # ~20 % of hand-labelled cells (vs ~50-70 % of "merged"). Detect always PROPOSES the
                                  # split lines; accept them per particle (Train tab) or draw cuts
    split_dip_ratio: float = 0.6  # ... where the saddle between two peaks is below this fraction of the lower peak
    split_max_x_hole: float = 4.0 # only particles touching the hole and at most this many hole areas are split
                                  # (a lenient-threshold speckle "particle" spanning the window is left whole)

    # --- particle classifier -------------------------------------------------------------------------------------
    model: str = ""               # trained model (.joblib); relative paths resolve in identity/models/
    p_seed: float = 0.5           # P(cell) a particle needs to seed a cell

    # --- reconstruction (neighbour aware) ------------------------------------------------------------------------
    split_dips: bool = True       # split at signal dips between peaks (off = every connected bright area is one cell)
    peak_h: float = 3.0           # a peak must rise this many hole-noise SDs above its surroundings
    dip_ratio: float = 0.6        # saddle / lower peak below this = two objects
    cell_gap_um: float = 2.0      # seeds of different channels closer than this may join (if no dip between them)
    grow_floor: float = 2.0       # grown pixels must be at least this many hole-noise SDs above the hole background
    grow_frac: float = 0.35       # ... and at least this fraction of the seed's median signal
    grow_margin_um: float = 1.5   # the cell may grow this far beyond the hole
    grow_max_x_hole: float = 2.0  # area cap (x hole area); the relative cut is raised in steps until it fits

    # --- identity call -------------------------------------------------------------------------------------------
    p_call: float = 0.8           # P(best channel combination) needed for a call, otherwise "uncertain"
    identity_bg: str = "hole"     # background under the body: "hole" (the hole's own, removes aperture glow) | "frame"
    min_mixture_sep: float = 2.0  # per-channel mixture separation (Ashman's D) below which the channel is flagged

    # --- display ------------------------------------------------------------------------------------------------
    view_R: float = 6.0           # runs without a usable hole crop: Detect covers +- this many hole radii of the
                                  # whole frame (with a crop: all of the crop). Features stay on the work_R window.

    # --- local --------------------------------------------------------------------------------------------------
    path_map: dict = field(default_factory=dict)   # stored path prefix -> local prefix, e.g. {"Z:\\": "/mnt/z/"}

    def identity_channels(self, available: list) -> list:
        chans = [c for c in (self.channels or available) if c in available]
        return [c for c in chans if c not in self.markers]

    def display_name(self, ch: str) -> str:
        return self.display.get(ch, {}).get("name") or ch

    def color(self, ch: str) -> str:
        return self.display.get(ch, {}).get("color") or default_color(ch)

    def feature_config(self) -> dict:
        return {k: getattr(self, k) for k in FEATURE_KEYS}


    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def model_path(self) -> Path | None:
        if not self.model:
            return None
        p = Path(self.model)
        return p if p.is_absolute() else MODELS / p


# what a run or model saved before a feature setting existed was made with
LEGACY_FEATURE_VALUES = {"split_particles": False}


def complete_feature_config(fc: dict) -> dict:
    """Fill settings missing from an older run.json / model spec with the value they implicitly had."""
    d = Config()
    return {k: fc.get(k, LEGACY_FEATURE_VALUES.get(k, getattr(d, k))) for k in FEATURE_KEYS}

_NAME_COLORS = [   # dye / filter names (display only), checked before the wavelength
    (("cy5", "alexa647", "af647", "far red", "farred"), "#38d9e8"),       # before "red"
    (("dapi", "hoechst", "bfp", "blue"), "#4f7dff"),
    (("fitc", "gfp", "alexa488", "af488", "green"), "#3ddc5a"),
    (("tx red", "texas", "tritc", "mcherry", "mscarlet", "cy3", "rfp", "tdtomato", "red"), "#ff4fd8"),
]


def default_color(ch: str) -> str:
    """Colour by dye/filter name or excitation wavelength (display only)."""
    import re
    low = str(ch).lower()
    for keys, col in _NAME_COLORS:
        if any(k in low for k in keys):
            return col
    m = re.search(r"(\d{3})", low)
    if not m:
        return "#bfbfbf"
    nm = int(m.group(1))
    return "#4f7dff" if nm < 450 else "#3ddc5a" if nm < 520 else "#ff4fd8" if nm < 600 else "#38d9e8"


def load_config(path: str | Path | None = None) -> Config:
    cfg = Config()
    if path:
        p = Path(path)
        if not p.exists() and (PRESETS / f"{path}.json").exists():
            p = PRESETS / f"{path}.json"
        known = {f.name for f in fields(Config)}
        for k, v in json.loads(p.read_text()).items():
            if k.startswith("_"):
                continue
            if k not in known:
                raise KeyError(f"unknown config key: {k}")
            setattr(cfg, k, v)
    return cfg


def presets() -> list[str]:
    return sorted(p.stem for p in PRESETS.glob("*.json"))
