"""Plate view of the reconstruction (the shared gui_common.plate.PlateGrid): wells coloured by status, dominant channel
or P; labelled wells framed white, test wells framed dashed amber."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ...gui_common.plate import PlateGrid
from .theme import blend, hex_rgb

STATUS_COLORS = {
    "one cell": "#2bb3a3", "several cells": "#f08a24", "no cell": "#3a3f47", "uncertain": "#e0a526",
    "one object": "#2bb3a3", "several objects": "#f08a24", "neighbours only": "#56708c", "no particles": "#3a3f47",
}


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
                          "P (call)": lambda r: _gradient(r.get("P", np.nan))},
                         legend_fn=self._legend, frame_fn=self._frame, tip_fn=self._tip)

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
        return "dark → light blue: P of the call (after Classify)"
