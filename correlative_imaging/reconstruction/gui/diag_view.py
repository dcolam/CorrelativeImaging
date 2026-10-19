"""Diagnostics tab (bottom dock): the plots Detect and Classify write to <results>/diagnostics, one at a time, scaled
to the dock's width (scroll for the rest; undock the Wells dock for a large view)."""
from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import Qt
from qtpy.QtGui import QPixmap
from qtpy.QtWidgets import QHBoxLayout, QLabel, QListWidget, QPushButton, QScrollArea, QVBoxLayout, QWidget

TITLES = {"1_snr": "SNR per channel", "2_brightness": "Brightness (typical-cell units)",
          "3_dynamic_range": "Dynamic range per well", "4_particles_per_well": "Particles per well",
          "5_background": "Backgrounds", "6_cross_channel": "Cross-channel brightness",
          "7_coloc": "Colocalisation in particles", "8_identity": "Identity calibration (after Classify)"}


class DiagnosticsView(QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app
        h = QHBoxLayout(self)
        h.setContentsMargins(4, 4, 4, 4)
        left = QVBoxLayout()
        self.list = QListWidget()
        self.list.setMaximumWidth(230)
        self.list.currentRowChanged.connect(self.show_plot)
        left.addWidget(self.list, 1)
        b = QPushButton("Re-draw plots")
        b.setToolTip("draw the plots again from the results (e.g. after renaming channels or re-detecting wells)")
        b.clicked.connect(self.redraw)
        left.addWidget(b)
        self.hint = QLabel("")
        self.hint.setProperty("role", "hint")
        self.hint.setWordWrap(True)
        self.hint.setMaximumWidth(230)
        left.addWidget(self.hint)
        h.addLayout(left)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.img = QLabel("no diagnostics yet: run Detect")
        self.img.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.scroll.setWidget(self.img)
        h.addWidget(self.scroll, 1)
        self.files = []
        self._pix = None

    def set_results(self, res):
        keep = self.list.currentRow()
        self.files = sorted((Path(res.dir) / "diagnostics").glob("*.png")) if res is not None else []
        self.list.blockSignals(True)
        self.list.clear()
        for f in self.files:
            self.list.addItem(TITLES.get(f.stem, f.stem))
        self.list.blockSignals(False)
        self.hint.setText("written by Detect / Classify into the results' diagnostics folder" if self.files else
                          "no plots in this results folder (detected before plots existed): Re-draw plots")
        if self.files:
            i = keep if 0 <= keep < len(self.files) else 0
            self.list.setCurrentRow(i)
            self.show_plot(i)
        else:
            self.img.setText("no diagnostics yet")

    def show_plot(self, i):
        if not (0 <= i < len(self.files)):
            return
        self._pix = QPixmap(str(self.files[i]))
        self._fit()

    def _fit(self):
        if self._pix is None or self._pix.isNull():
            return
        w = max(self.scroll.viewport().width() - 4, 200)
        self.img.setPixmap(self._pix.scaledToWidth(min(w, self._pix.width()), Qt.SmoothTransformation))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._fit()

    def redraw(self):
        res = self.app.res
        if res is None:
            return
        from ..diagnostics import classify_diagnostics, detect_diagnostics
        detect_diagnostics(res.dir, self.app.cfg)
        if res.calls is not None:
            classify_diagnostics(res.dir, cfg=self.app.cfg)
        self.set_results(res)
