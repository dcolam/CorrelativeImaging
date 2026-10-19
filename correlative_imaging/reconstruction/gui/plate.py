"""Plate view: one button per well, coloured by status, dominant channel or P; labelled wells framed white, test
wells framed dashed amber. The plate format (96 / 384 / 1536) follows from the well ids."""
from __future__ import annotations

import numpy as np
import pandas as pd
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (QComboBox, QGridLayout, QHBoxLayout, QLabel, QScrollArea, QSizePolicy, QToolButton,
                            QVBoxLayout, QWidget)

from .theme import blend, hex_rgb

STATUS_COLORS = {
    "one cell": "#2bb3a3", "several cells": "#f08a24", "no cell": "#3a3f47", "uncertain": "#e0a526",
    "one object": "#2bb3a3", "several objects": "#f08a24", "neighbours only": "#56708c", "no particles": "#3a3f47",
}
FORMATS = [(8, 12), (16, 24), (32, 48)]


def _rc(well: str):
    row = 0
    letters = "".join(c for c in well if c.isalpha())
    for c in letters:
        row = row * 26 + (ord(c.upper()) - 64)
    return row - 1, int("".join(c for c in well if c.isdigit())) - 1


def _row_name(i: int) -> str:
    return chr(65 + i) if i < 26 else chr(64 + i // 26) + chr(65 + i % 26)


def _gradient(p: float) -> str:
    if p is None or not np.isfinite(p):
        return "#2a2e35"
    a, b = hex_rgb("#2a2e35"), hex_rgb("#4fc3f7")
    c = a + (b - a) * float(np.clip(p, 0, 1))
    return "#" + "".join(f"{int(round(v * 255)):02x}" for v in c)


class PlateView(QWidget):
    well_clicked = Signal(str, str)

    def __init__(self, color_of, display_name):
        super().__init__()
        self.color_of, self.display_name = color_of, display_name
        v = QVBoxLayout(self)
        v.setContentsMargins(4, 4, 4, 4)
        h = QHBoxLayout()
        self.plate = QComboBox()
        self.plate.currentTextChanged.connect(self.draw)
        self.mode = QComboBox()
        self.mode.addItems(["status", "dominant channel", "P (call)"])
        self.mode.currentTextChanged.connect(self.draw)
        h.addWidget(self.plate, 1)
        h.addWidget(self.mode)
        v.addLayout(h)
        self.stage = QLabel("")
        self.stage.setProperty("role", "hint")
        v.addWidget(self.stage)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.grid_w = QWidget()
        self.grid = QGridLayout(self.grid_w)
        self.grid.setSpacing(1)
        self.grid.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.scroll.setWidget(self.grid_w)
        v.addWidget(self.scroll, 1)
        self.legend = QLabel("")
        self.legend.setWordWrap(True)
        self.legend.setTextFormat(Qt.RichText)
        v.addWidget(self.legend)
        self.df = pd.DataFrame()
        self.current = None

    def set(self, df: pd.DataFrame, stage: str):
        self.df = df
        cur = self.plate.currentText()
        self.plate.blockSignals(True)
        self.plate.clear()
        plates = sorted(df.plate_id.unique()) if len(df) else []
        self.plate.addItems(plates)
        if cur in plates:
            self.plate.setCurrentText(cur)
        self.plate.blockSignals(False)
        self.stage.setText(f"colours: {stage}")
        self.draw()

    def mark(self, plate_id, well_id):
        self.current = (plate_id, well_id)
        if self.plate.currentText() != plate_id:
            self.plate.setCurrentText(plate_id)
        else:
            self.draw()

    def _color(self, r) -> str:
        mode = self.mode.currentText()
        if mode == "status":
            return STATUS_COLORS.get(r.get("status_shown", ""), "#3a3f47")
        if mode == "dominant channel":
            dom = r.get("dominant_shown", "")
            if not isinstance(dom, str) or dom in ("", "none", "uncertain", "no cell"):
                return "#3a3f47" if dom != "uncertain" else "#e0a526"
            return blend([self.color_of(c) for c in dom.split("+")])
        return _gradient(r.get("P", np.nan))

    def draw(self, *_):
        while self.grid.count():
            w = self.grid.takeAt(0).widget()
            if w:
                w.setParent(None)
                w.deleteLater()
        pl = self.plate.currentText()
        if not len(self.df) or not pl:
            return
        d = self.df[self.df.plate_id == pl]
        rc = [_rc(w) for w in d.well_id]
        nr, nc = max(r for r, _ in rc) + 1, max(c for _, c in rc) + 1
        R, C = next(((a, b) for a, b in FORMATS if nr <= a and nc <= b), (nr, nc))
        for j in range(C):
            lab = QLabel(str(j + 1))
            lab.setAlignment(Qt.AlignCenter)
            lab.setProperty("role", "hint")
            self.grid.addWidget(lab, 0, j + 1)
        for i in range(R):
            lab = QLabel(_row_name(i))
            lab.setProperty("role", "hint")
            self.grid.addWidget(lab, i + 1, 0)
        size = 11 if C > 24 else 15 if C > 12 else 22
        by = {w: r for w, r in zip(d.well_id, d.to_dict("records"))}
        for i in range(R):
            for j in range(C):
                well = f"{_row_name(i)}{j + 1:02d}"
                r = by.get(well)
                b = QToolButton()
                b.setFixedSize(size, size)
                b.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
                if r is None:
                    b.setEnabled(False)
                    b.setStyleSheet("QToolButton { background: transparent; border: 1px solid #2a2e35; }")
                    self.grid.addWidget(b, i + 1, j + 1)
                    continue
                col = self._color(r)
                border = "1px solid #20232a"
                if r.get("test") == "✓":
                    border = "2px dashed #e0a526"
                if (r.get("labels") or 0) > 0:
                    border = "2px solid #ffffff"
                if self.current == (pl, well):
                    border = "2px solid #4fc3f7"
                b.setStyleSheet(f"QToolButton {{ background: {col}; border: {border}; border-radius: 3px; }}")
                tip = [f"{pl} {well}", f"status: {r.get('status_shown', '')}"]
                if r.get("dominant_shown"):
                    tip.append("dominant: " + "+".join(self.display_name(c) for c in str(r["dominant_shown"]).split("+")))
                if r.get("P") == r.get("P"):
                    tip.append(f"P: {r['P']:.2f}" if isinstance(r.get("P"), float) else "")
                if r.get("labels"):
                    tip.append(f"{r['labels']} hand labels")
                b.setToolTip("\n".join(t for t in tip if t))
                b.clicked.connect(lambda _=False, w=well: self.well_clicked.emit(pl, w))
                self.grid.addWidget(b, i + 1, j + 1)
        mode = self.mode.currentText()
        if mode == "status":
            keys = sorted(set(d.get("status_shown", pd.Series(dtype=str)).dropna()))
            self.legend.setText(" ".join(f'<span style="color:{STATUS_COLORS.get(k, "#888")}">■</span> {k}'
                                         for k in keys) + ' &nbsp; <b>□</b> white frame = labelled')
        elif mode == "dominant channel":
            chans = sorted({c for s in d.get("dominant_shown", pd.Series(dtype=str)).dropna() for c in str(s).split("+")
                            if c and c not in ("none", "uncertain", "no cell")})
            self.legend.setText(" ".join(f'<span style="color:{self.color_of(c)}">■</span> {self.display_name(c)}'
                                         for c in chans) + " · blends = several")
        else:
            self.legend.setText("dark → light blue: P of the call (after Classify)")
