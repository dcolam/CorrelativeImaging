"""Diagnostics tab: the plots Detect and Classify write to <results>/diagnostics, one at a time, scaled to the panel's
width (undock the panel or open the folder for a large view)."""
from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import Qt
from qtpy.QtGui import QPixmap
from qtpy.QtWidgets import QComboBox, QHBoxLayout, QLabel, QPushButton, QScrollArea, QVBoxLayout, QWidget

TITLES = {"1_snr": "SNR per channel", "2_brightness": "Brightness (typical-cell units)",
          "3_dynamic_range": "Dynamic range per well", "4_particles_per_well": "Particles per well",
          "5_background": "Backgrounds", "6_cross_channel": "Cross-channel brightness",
          "7_coloc": "Colocalisation in particles", "8_identity": "Identity calibration (after Classify)",
          "9_cell_size": "Size of one cell (cells per particle)", "10_clumps": "Clump scores per hole (after Classify)"}


class DiagnosticsView(QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app
        v = QVBoxLayout(self)
        v.setContentsMargins(4, 4, 4, 4)
        h = QHBoxLayout()
        self.list = QComboBox()
        self.list.currentIndexChanged.connect(self.show_plot)
        h.addWidget(self.list, 1)
        b = QPushButton("Re-draw")
        b.setToolTip("draw the plots again from the results (e.g. after renaming channels or re-detecting wells)")
        b.clicked.connect(self.redraw)
        h.addWidget(b)
        b = QPushButton("Open large")
        b.setToolTip("the plots in their own window, at full size (resizable, keeps the choice in sync)")
        b.clicked.connect(self.open_large)
        h.addWidget(b)
        b = QPushButton("Folder")
        b.clicked.connect(self.open_folder)
        h.addWidget(b)
        v.addLayout(h)
        self.hint = QLabel("")
        self.hint.setProperty("role", "hint")
        self.hint.setWordWrap(True)
        v.addWidget(self.hint)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.img = QLabel("no diagnostics yet: run Detect")
        self.img.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.scroll.setWidget(self.img)
        v.addWidget(self.scroll, 1)
        self.files = []
        self._pix = None

    def open_large(self):
        """A separate window: the same plot list, the plot at full size (scroll for more)."""
        from qtpy.QtWidgets import QDialog
        if getattr(self, "_big", None) is None:
            dlg = QDialog(self)
            dlg.setWindowTitle("Diagnostics")
            dlg.setWindowFlag(Qt.WindowMaximizeButtonHint, True)
            lv = QVBoxLayout(dlg)
            self._big_combo = QComboBox()
            self._big_combo.currentIndexChanged.connect(self._show_big)
            lv.addWidget(self._big_combo)
            sc = QScrollArea()
            sc.setWidgetResizable(True)
            self._big_img = QLabel()
            self._big_img.setAlignment(Qt.AlignTop | Qt.AlignLeft)
            sc.setWidget(self._big_img)
            lv.addWidget(sc, 1)
            dlg.resize(1400, 900)
            self._big = dlg
        self._big_combo.blockSignals(True)
        self._big_combo.clear()
        self._big_combo.addItems([TITLES.get(f.stem, f.stem) for f in self.files])
        self._big_combo.setCurrentIndex(max(self.list.currentIndex(), 0))
        self._big_combo.blockSignals(False)
        self._show_big(self._big_combo.currentIndex())
        self._big.show()
        self._big.raise_()

    def _show_big(self, i):
        if 0 <= i < len(self.files):
            self._big_img.setPixmap(QPixmap(str(self.files[i])))

    def open_folder(self):
        import os
        if self.app.res is not None:
            d = Path(self.app.res.dir) / "diagnostics"
            if d.exists():
                os.startfile(str(d)) if hasattr(os, "startfile") else None

    def set_results(self, res):
        keep = self.list.currentIndex()
        import re
        num = lambda p: (int(m.group(1)) if (m := re.match(r"(\d+)", p.stem)) else 999, p.stem)   # 1, 2, … 10
        self.files = sorted((Path(res.dir) / "diagnostics").glob("*.png"), key=num) if res is not None else []
        self.list.blockSignals(True)
        self.list.clear()
        for f in self.files:
            self.list.addItem(TITLES.get(f.stem, f.stem))
        self.list.blockSignals(False)
        self.hint.setText("written by Detect / Classify into the results' diagnostics folder" if self.files else
                          "no plots in this results folder (detected before plots existed): Re-draw plots")
        if self.files:
            i = keep if 0 <= keep < len(self.files) else 0
            self.list.setCurrentIndex(i)
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
        w = max(self.width() - 30, 200)                 # the panel's width (the label would keep the old size)
        self.img.setPixmap(self._pix.scaledToWidth(min(w, self._pix.width()), Qt.SmoothTransformation))

    def showEvent(self, e):
        super().showEvent(e)
        self._fit()

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
