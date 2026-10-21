"""Plate view of the reconstruction (the shared gui_common.plate.PlateGrid): wells coloured by status, dominant channel,
P or the plate map's active column (Results tab); labelled wells framed white, test wells framed dashed amber."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ...gui_common.plate import PlateGrid
from .theme import blend, hex_rgb

STATUS_COLORS = {
    "one cell": "#2bb3a3", "several cells": "#f08a24", "no cell": "#3a3f47", "uncertain": "#e0a526",
    "one object": "#2bb3a3", "several objects": "#f08a24", "neighbours only": "#56708c", "no particles": "#3a3f47",
}


# plate-map values: a value named like a channel (its display name, e.g. "GFP") gets that channel's colour, so the map
# and the "dominant channel" colours can be compared directly; other values get these, in sorted order
MAP_PALETTE = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#59a14f",
               "#bab0ac"]
MAP_EMPTY = {"empty", "none", "no cell", "-"}


def map_colors(values: pd.Series, channels: list, color_of, display_name) -> dict | tuple:
    """{value: hex} for a categorical column, ("range", lo, hi) for a numeric one with many values."""
    v = values.dropna()
    if pd.api.types.is_numeric_dtype(v) and v.nunique() > len(MAP_PALETTE):
        return ("range", float(v.min()), float(v.max()))
    named = {display_name(c).lower(): color_of(c) for c in channels} | {str(c).lower(): color_of(c) for c in channels}
    out, k = {}, 0
    for x in sorted(v.astype(str).unique()):
        if x.lower() in named:
            out[x] = named[x.lower()]
        elif x.lower() in MAP_EMPTY:
            out[x] = "#56606b"
        else:
            out[x] = MAP_PALETTE[k % len(MAP_PALETTE)]
            k += 1
    return out


def _gradient(p: float) -> str:
    if p is None or not np.isfinite(p):
        return "#2a2e35"
    a, b = hex_rgb("#2a2e35"), hex_rgb("#4fc3f7")
    c = a + (b - a) * float(np.clip(p, 0, 1))
    return "#" + "".join(f"{int(round(v * 255)):02x}" for v in c)


class PlateView(PlateGrid):
    def __init__(self, color_of, display_name):
        self.color_of, self.display_name = color_of, display_name
        super().__init__({"status": self._status, "dominant channel": self._dominant,
                          "P (call)": lambda r: _gradient(r.get("P", np.nan)), "plate map": self._map},
                         legend_fn=self._legend, frame_fn=self._frame, tip_fn=self._tip)
        self.map_col, self.map_cols = "", {}

    def set_map(self, col: str, colors):
        """The plate map's active column and its colours (map_colors, over all plates: stable between plates)."""
        self.map_col, self.map_cols = col or "", colors if col else {}

    def _map(self, r):
        x = r.get("map_value")
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "#2a2e35"
        if isinstance(self.map_cols, tuple):
            _, lo, hi = self.map_cols
            return _gradient((float(x) - lo) / (hi - lo) if hi > lo else 1.0)
        return self.map_cols.get(str(x), "#2a2e35")

    @staticmethod
    def _status(r):
        return STATUS_COLORS.get(r.get("status_shown", ""), "#3a3f47")

    def _dominant(self, r):
        dom = r.get("dominant_shown", "")
        if not isinstance(dom, str) or dom in ("", "none", "uncertain", "no cell"):
            return "#3a3f47" if dom != "uncertain" else "#e0a526"
        return blend([self.color_of(c) for c in dom.split("+")])

    @staticmethod
    def _frame(r):
        if (r.get("labels") or 0) > 0:
            return "2px solid #ffffff"
        if r.get("test") == "✓":
            return "2px dashed #e0a526"
        return None

    def _tip(self, pl, well, r):
        tip = [f"{pl} {well}", f"status: {r.get('status_shown', '')}"]
        if r.get("dominant_shown"):
            tip.append("dominant: " + "+".join(self.display_name(c) for c in str(r["dominant_shown"]).split("+")))
        if isinstance(r.get("P"), float) and r["P"] == r["P"]:
            tip.append(f"P: {r['P']:.2f}")
        if self.map_col and r.get("map_value") is not None and r.get("map_value") == r.get("map_value"):
            tip.append(f"{self.map_col}: {r['map_value']}")
        if r.get("labels"):
            tip.append(f"{r['labels']} hand labels")
        return "\n".join(tip)

    def _legend(self, mode, d):
        if mode == "status":
            keys = sorted(set(d.get("status_shown", pd.Series(dtype=str)).dropna()))
            return " ".join(f'<span style="color:{STATUS_COLORS.get(k, "#888")}">■</span> {k}'
                            for k in keys) + ' &nbsp; <b>□</b> white frame = labelled'
        if mode == "dominant channel":
            chans = sorted({c for s in d.get("dominant_shown", pd.Series(dtype=str)).dropna() for c in str(s).split("+")
                            if c and c not in ("none", "uncertain", "no cell")})
            return " ".join(f'<span style="color:{self.color_of(c)}">■</span> {self.display_name(c)}'
                            for c in chans) + " · blends = several"
        if mode == "plate map":
            if not self.map_col:
                return "load a plate map in the Results tab; its selected column colours the wells"
            if isinstance(self.map_cols, tuple):
                return f"{self.map_col}: dark → light blue = {self.map_cols[1]:g} → {self.map_cols[2]:g}"
            here = set(d.get("map_value", pd.Series(dtype=object)).dropna().astype(str))
            return f"<b>{self.map_col}</b>: " + " ".join(
                f'<span style="color:{c}">■</span> {k}' for k, c in self.map_cols.items() if k in here) + \
                " · dark = not in the map (or blind)"
        return "dark → light blue: P of the call (after Classify)"
