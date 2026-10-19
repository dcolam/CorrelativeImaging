"""Channels dock (left): every channel at once, as in the HTML viewer. Per channel: show, its histogram (log counts
of the well on screen, the shown range shaded in the channel's colour), min / max sliders, gamma and "auto" (the
contrast rule). On top: auto-contrast on well change; below: switches for the overlay layers."""
from __future__ import annotations

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor, QImage, QPixmap
from qtpy.QtWidgets import (QCheckBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton, QSlider, QVBoxLayout,
                            QWidget)

from .display import contrast_rule

HW, HH = 220, 34          # histogram size
STEPS = 1000

OVERLAYS = [("neighbours", "neighbours (grey, N)"), ("cells", "reconstructed cell (outline)"),
            ("tags", "class tags"), ("splits", "edits / proposed splits"), ("bf", "brightfield")]


def _hex(c):
    h = (c or "#ffffff").lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


class ChannelRow(QFrame):
    def __init__(self, panel, ch, layer, color, name):
        super().__init__()
        self.panel, self.ch, self.L, self.color = panel, ch, layer, color
        self._lock = False
        g = QGridLayout(self)
        g.setContentsMargins(4, 2, 4, 2)
        g.setVerticalSpacing(1)
        self.vis = QCheckBox(name)
        self.vis.setStyleSheet(f"QCheckBox {{ color: {color}; font-weight: bold; }}")
        self.vis.setChecked(layer.visible)
        self.vis.toggled.connect(lambda on: setattr(self.L, "visible", on))
        g.addWidget(self.vis, 0, 0, 1, 2)
        self.rng = QLabel("")
        self.rng.setProperty("role", "hint")
        g.addWidget(self.rng, 0, 2)
        b = QPushButton("auto")
        b.setToolTip("minimum → 99.9th percentile, at least minimum + 100")
        b.clicked.connect(self.auto)
        g.addWidget(b, 0, 3)
        self.hist = QLabel()
        self.hist.setFixedSize(HW, HH)
        g.addWidget(self.hist, 1, 0, 1, 4)
        self.smin, self.smax, self.sgam = (QSlider(Qt.Horizontal) for _ in range(3))
        for s_ in (self.smin, self.smax):
            s_.setRange(0, STEPS)
            s_.valueChanged.connect(self.from_sliders)
        self.sgam.setRange(20, 300)
        self.sgam.valueChanged.connect(lambda v: setattr(self.L, "gamma", v / 100))
        for r, (lab, w) in enumerate((("min", self.smin), ("max", self.smax), ("γ", self.sgam)), start=2):
            g.addWidget(QLabel(lab), r, 0)
            g.addWidget(w, r, 1, 1, 3)
        layer.events.contrast_limits.connect(lambda e: self.sync())
        layer.events.visible.connect(lambda e: self.vis.setChecked(self.L.visible))
        self.data_changed()

    # data range of the well on screen -> slider positions
    def data_changed(self):
        a = np.asarray(self.L.data, float)
        a = a[np.isfinite(a)]
        self.lo, self.hi = (float(a.min()), float(a.max())) if a.size else (0.0, 1.0)
        if self.hi <= self.lo:
            self.hi = self.lo + 1
        sub = a[:: max(1, a.size // 200000)]
        pos = np.clip(((sub - self.lo) / (self.hi - self.lo) * (HW - 1)).astype(int), 0, HW - 1)
        self.bins = np.log1p(np.bincount(pos, minlength=HW).astype(float))
        self.sync()

    def _pos(self, v):
        return int(round((v - self.lo) / (self.hi - self.lo) * STEPS))

    def _val(self, p):
        return self.lo + p / STEPS * (self.hi - self.lo)

    def sync(self):
        lo, hi = self.L.contrast_limits
        self._lock = True
        self.smin.setValue(max(0, min(STEPS, self._pos(lo))))
        self.smax.setValue(max(0, min(STEPS, self._pos(hi))))
        self.sgam.setValue(int(round(self.L.gamma * 100)))
        self._lock = False
        self.rng.setText(f"{lo:.0f} – {hi:.0f}  (data {self.lo:.0f} – {self.hi:.0f})")
        self.draw_hist(lo, hi)

    def from_sliders(self, *_):
        if self._lock:
            return
        lo, hi = self._val(self.smin.value()), self._val(self.smax.value())
        if hi <= lo:
            hi = lo + (self.hi - self.lo) / STEPS
        self.L.contrast_limits = (lo, hi)

    def auto(self):
        self.L.contrast_limits = contrast_rule(np.asarray(self.L.data, float))
        self.L.gamma = 1.0

    def draw_hist(self, lo, hi):
        img = np.zeros((HH, HW, 4), np.uint8)
        mx = self.bins.max() or 1
        r, g, b = _hex(self.color)
        x0 = int(np.clip((lo - self.lo) / (self.hi - self.lo) * (HW - 1), 0, HW - 1))
        x1 = int(np.clip((hi - self.lo) / (self.hi - self.lo) * (HW - 1), 0, HW - 1))
        img[:, x0:x1 + 1] = (r, g, b, 60)                       # the shown range, in the channel's colour
        for x in range(HW):
            h = int(round(self.bins[x] / mx * (HH - 2)))
            if h:
                img[HH - h:, x] = (150, 155, 160, 255)
        q = QImage(img.data, HW, HH, 4 * HW, QImage.Format_RGBA8888)
        self.hist.setPixmap(QPixmap.fromImage(q.copy()))


class ChannelPanel(QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app
        v = QVBoxLayout(self)
        v.setContentsMargins(4, 4, 4, 4)
        self.auto_cb = QCheckBox("auto-contrast on well change")
        self.auto_cb.setToolTip("every channel: black point = its minimum, white point = 99.9th percentile but at least "
                                "minimum + 100")
        self.auto_cb.setChecked(True)
        self.auto_cb.toggled.connect(lambda on: setattr(app.display, "auto_contrast", on))
        v.addWidget(self.auto_cb)
        self.rows_box = QVBoxLayout()
        v.addLayout(self.rows_box)
        self.rows = {}
        self.parts = QVBoxLayout()
        lab = QLabel("particles in the hole (full = both thresholds agree)")
        lab.setProperty("role", "hint")
        v.addWidget(lab)
        v.addLayout(self.parts)
        self.part_cb = {}
        lab = QLabel("overlays")
        lab.setProperty("role", "hint")
        v.addWidget(lab)
        self.ov = {}
        for role, text in OVERLAYS:
            cb = QCheckBox(text)
            cb.toggled.connect(lambda on, role=role: self._set_vis(role, on))
            self.ov[role] = cb
            v.addWidget(cb)
        v.addStretch(1)
        self._sig = None

    def _set_vis(self, role, on):
        L = self.app.display.L.get(role)
        if L is not None:
            L.visible = on

    def refresh(self):
        """After a well is shown: rebuild rows if the channels changed, else only their histograms."""
        d = self.app.display
        chans = [r[4:] for r in d.L if r.startswith("img:")]
        sig = tuple((c, id(d.L[f"img:{c}"])) for c in chans)
        if sig != self._sig:
            self._sig = sig
            for w in list(self.rows.values()) + list(self.part_cb.values()):
                w.setParent(None)
            self.rows, self.part_cb = {}, {}
            cfg = self.app.cfg
            for c in chans:
                row = ChannelRow(self, c, d.L[f"img:{c}"], cfg.color(c), cfg.display_name(c))
                self.rows[c] = row
                self.rows_box.addWidget(row)
                P = d.L.get(f"part:{c}")
                cb = QCheckBox(cfg.display_name(c))
                cb.setStyleSheet(f"QCheckBox {{ color: {cfg.color(c)}; }}")
                cb.setChecked(bool(P is not None and P.visible))
                cb.toggled.connect(lambda on, c=c: self._set_vis(f"part:{c}", on))
                if P is not None:
                    P.events.visible.connect(lambda e, cb=cb, P=P: cb.setChecked(P.visible))
                self.part_cb[c] = cb
                self.parts.addWidget(cb)
            for role, cb in self.ov.items():
                L = d.L.get(role)
                cb.setEnabled(L is not None)
                if L is not None:
                    cb.blockSignals(True)
                    cb.setChecked(L.visible)
                    cb.blockSignals(False)
                    L.events.visible.connect(lambda e, cb=cb, L=L: cb.setChecked(L.visible))
        else:
            for row in self.rows.values():
                row.data_changed()
