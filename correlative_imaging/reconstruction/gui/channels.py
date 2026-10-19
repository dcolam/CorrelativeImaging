"""Control view (left, tabbed with napari's layer list and layer controls): the main place to steer the canvas.

  plate          the plate view (plate + colouring dropdowns); click a well to open it; "Label this well" puts
                 every object of the well on screen onto a bulk-labeller page (labelled ones too, to relabel them)
  channels       one row per channel: show it, pick it for adjusting; ONE histogram (log counts of the well on
                 screen, the shown range shaded in the channel's colour) with min / max / gamma / auto for the
                 picked channel. Auto-contrast on well change: minimum -> 99.9th percentile, at least minimum + 100
  particles      the per-channel particle fills (full = both thresholds agree)
  overlays       neighbours, reconstructed cell, class tags, edits, brightfield
"""
from __future__ import annotations

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtGui import QImage, QPixmap
from qtpy.QtWidgets import (QButtonGroup, QCheckBox, QGridLayout, QHBoxLayout, QLabel, QPushButton, QRadioButton,
                            QScrollArea, QFrame, QSizePolicy, QSlider, QVBoxLayout, QWidget)

from .display import contrast_rule

HH = 46                   # histogram height (width follows the panel)
STEPS = 1000

OVERLAYS = [("neighbours", "neighbours (grey, N)"), ("cells", "cell outline"), ("tags", "class tags"),
            ("splits", "edits / splits"), ("bf", "brightfield")]


def _hex(c):
    h = (c or "#ffffff").lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _section(text):
    l = QLabel(text)
    l.setProperty("role", "section")
    return l


class ControlView(QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setFrameShape(QFrame.NoFrame)
        sc.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        body = QWidget()
        v = QVBoxLayout(body)
        v.setContentsMargins(4, 4, 4, 4)
        sc.setWidget(body)
        outer.addWidget(sc)

        # plate
        v.addWidget(_section("Plate"))
        self.plate_box = QVBoxLayout()
        v.addLayout(self.plate_box)
        b = QPushButton("Label this well")
        b.setToolTip("every object of the well on screen onto a bulk-labeller page (Label tab), labelled ones too: "
                     "a key overwrites their label")
        b.clicked.connect(app.label_this_well)
        v.addWidget(b)

        # channels: rows + one histogram
        v.addWidget(_section("Channels"))
        self.auto_cb = QCheckBox("auto-contrast on well change")
        self.auto_cb.setToolTip("every channel: black point = its minimum, white point = 99.9th percentile but at least "
                                "minimum + 100")
        self.auto_cb.setChecked(True)
        self.auto_cb.toggled.connect(lambda on: setattr(app.display, "auto_contrast", on))
        v.addWidget(self.auto_cb)
        self.rows = QGridLayout()
        self.rows.setVerticalSpacing(2)
        v.addLayout(self.rows)
        self.pick = QButtonGroup(self)
        self.pick.buttonToggled.connect(lambda b, on: on and self._pick(b.property("ch")))
        self.hist = QLabel()
        self.hist.setMinimumHeight(HH)
        self.hist.setMaximumHeight(HH)
        # never sized by its own picture (that grew the panel on every redraw): the panel decides its width
        self.hist.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.hist.setMinimumWidth(10)
        v.addWidget(self.hist)
        g = QGridLayout()
        self.smin, self.smax, self.sgam = (QSlider(Qt.Horizontal) for _ in range(3))
        for s_ in (self.smin, self.smax):
            s_.setRange(0, STEPS)
            s_.valueChanged.connect(self._from_sliders)
        self.sgam.setRange(20, 300)
        self.sgam.valueChanged.connect(self._gamma)
        for r, (lab, w) in enumerate((("min", self.smin), ("max", self.smax), ("γ", self.sgam))):
            g.addWidget(QLabel(lab), r, 0)
            g.addWidget(w, r, 1)
        self.rng = QLabel("")
        self.rng.setProperty("role", "hint")
        self.rng.setWordWrap(True)
        self.rng.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        g.addWidget(self.rng, 3, 0, 1, 2)
        b = QPushButton("auto")
        b.setToolTip("minimum → 99.9th percentile, at least minimum + 100")
        b.clicked.connect(self._auto)
        g.addWidget(b, 0, 2, 3, 1)
        v.addLayout(g)

        # particles, overlays
        v.addWidget(_section("Particles in the hole"))
        lab = QLabel("filled: full where both thresholds agree, half where only one covers it")
        lab.setProperty("role", "hint")
        lab.setWordWrap(True)
        v.addWidget(lab)
        self.parts = QHBoxLayout()
        v.addLayout(self.parts)
        v.addWidget(_section("Overlays"))
        og = QGridLayout()
        self.ov = {}
        for i, (role, text) in enumerate(OVERLAYS):
            cb = QCheckBox(text)
            cb.toggled.connect(lambda on, role=role: self._set_vis(role, on))
            self.ov[role] = cb
            og.addWidget(cb, i // 2, i % 2)
        v.addLayout(og)
        v.addStretch(1)
        self._sig = None
        self.cur = None
        self._lock = False
        self.vis_cb, self.part_cb = {}, {}

    # -- plate
    def set_plate(self, plate_widget):
        self.plate_box.addWidget(plate_widget)

    # -- layers
    def _set_vis(self, role, on):
        L = self.app.display.L.get(role)
        if L is not None:
            L.visible = on

    def refresh(self):
        """After a well is shown: rebuild the rows if the channels changed, else only the histogram."""
        d = self.app.display
        chans = [r[4:] for r in d.L if r.startswith("img:")]
        sig = tuple((c, id(d.L[f"img:{c}"])) for c in chans)
        if sig != self._sig:
            self._sig = sig
            self._build(chans)
        self._data_changed()

    def _build(self, chans):
        d, cfg = self.app.display, self.app.cfg
        while self.rows.count():
            w = self.rows.takeAt(0).widget()
            if w is not None:
                w.setParent(None)
        for b in self.pick.buttons():
            self.pick.removeButton(b)
        for w in self.part_cb.values():
            w.setParent(None)
        self.vis_cb, self.part_cb = {}, {}
        for i, c in enumerate(chans):
            L = d.L[f"img:{c}"]
            cb = QCheckBox(cfg.display_name(c))
            cb.setStyleSheet(f"QCheckBox {{ color: {cfg.color(c)}; font-weight: bold; }}")
            cb.setChecked(L.visible)
            cb.toggled.connect(lambda on, L=L: setattr(L, "visible", on))
            L.events.visible.connect(lambda e, cb=cb, L=L: cb.setChecked(L.visible))
            rb = QRadioButton("adjust")
            rb.setProperty("ch", c)
            self.pick.addButton(rb)
            lim = QLabel("")
            lim.setProperty("role", "hint")
            L.events.contrast_limits.connect(lambda e, c=c, lim=lim, L=L: (
                lim.setText(f"{L.contrast_limits[0]:.0f} – {L.contrast_limits[1]:.0f}"),
                self._sync() if c == self.cur else None))
            lim.setText(f"{L.contrast_limits[0]:.0f} – {L.contrast_limits[1]:.0f}")
            self.rows.addWidget(cb, i, 0)
            self.rows.addWidget(lim, i, 1)
            self.rows.addWidget(rb, i, 2)
            self.vis_cb[c] = cb
            P = d.L.get(f"part:{c}")
            pc = QCheckBox(cfg.display_name(c))
            pc.setStyleSheet(f"QCheckBox {{ color: {cfg.color(c)}; }}")
            pc.setChecked(bool(P is not None and P.visible))
            pc.toggled.connect(lambda on, c=c: self._set_vis(f"part:{c}", on))
            if P is not None:
                P.events.visible.connect(lambda e, pc=pc, P=P: pc.setChecked(P.visible))
            self.part_cb[c] = pc
            self.parts.addWidget(pc)
        for role, cb in self.ov.items():
            L = d.L.get(role)
            cb.setEnabled(L is not None)
            if L is not None:
                cb.blockSignals(True)
                cb.setChecked(L.visible)
                cb.blockSignals(False)
                L.events.visible.connect(lambda e, cb=cb, L=L: cb.setChecked(L.visible))
        if chans:
            self.cur = chans[0]
            self.pick.buttons()[0].setChecked(True)

    # -- the picked channel
    def _layer(self):
        return self.app.display.L.get(f"img:{self.cur}") if self.cur else None

    def _pick(self, ch):
        self.cur = ch
        self._data_changed()

    def _data_changed(self):
        L = self._layer()
        if L is None:
            return
        a = np.asarray(L.data, float)
        a = a[np.isfinite(a)]
        self.lo, self.hi = (float(a.min()), float(a.max())) if a.size else (0.0, 1.0)
        if self.hi <= self.lo:
            self.hi = self.lo + 1
        self._sample = a[:: max(1, a.size // 200000)]
        self._sync()

    def _pos(self, v):
        return int(round((v - self.lo) / (self.hi - self.lo) * STEPS))

    def _val(self, p):
        return self.lo + p / STEPS * (self.hi - self.lo)

    def _sync(self):
        L = self._layer()
        if L is None or not hasattr(self, "lo"):
            return
        lo, hi = L.contrast_limits
        self._lock = True
        self.smin.setValue(max(0, min(STEPS, self._pos(lo))))
        self.smax.setValue(max(0, min(STEPS, self._pos(hi))))
        self.sgam.setValue(int(round(L.gamma * 100)))
        self._lock = False
        self.rng.setText(f"{self.app.cfg.display_name(self.cur)}: {lo:.0f} – {hi:.0f}  (data {self.lo:.0f} – "
                         f"{self.hi:.0f}), γ {L.gamma:.2f}")
        self._draw_hist(lo, hi)

    def _from_sliders(self, *_):
        L = self._layer()
        if self._lock or L is None:
            return
        lo, hi = self._val(self.smin.value()), self._val(self.smax.value())
        if hi <= lo:
            hi = lo + (self.hi - self.lo) / STEPS
        L.contrast_limits = (lo, hi)

    def _gamma(self, v):
        L = self._layer()
        if not self._lock and L is not None:
            L.gamma = v / 100
            self._sync()

    def _auto(self):
        L = self._layer()
        if L is not None:
            L.contrast_limits = contrast_rule(np.asarray(L.data, float))
            L.gamma = 1.0

    def _draw_hist(self, lo, hi):
        W = max(self.width() - 24, 120)                     # the panel's width, not the label's
        a = getattr(self, "_sample", np.zeros(0))
        pos = np.clip(((a - self.lo) / (self.hi - self.lo) * (W - 1)).astype(int), 0, W - 1) if a.size else \
            np.zeros(0, int)
        bins = np.log1p(np.bincount(pos, minlength=W).astype(float))
        img = np.zeros((HH, W, 4), np.uint8)
        r, g, b = _hex(self.app.cfg.color(self.cur))
        x0 = int(np.clip((lo - self.lo) / (self.hi - self.lo) * (W - 1), 0, W - 1))
        x1 = int(np.clip((hi - self.lo) / (self.hi - self.lo) * (W - 1), 0, W - 1))
        img[:, x0:x1 + 1] = (r, g, b, 70)
        mx = bins.max() or 1
        for x in range(W):
            h = int(round(bins[x] / mx * (HH - 2)))
            if h:
                img[HH - h:, x] = (160, 165, 170, 255)
        q = QImage(img.data, W, HH, 4 * W, QImage.Format_RGBA8888)
        self.hist.setPixmap(QPixmap.fromImage(q.copy()))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        L = self._layer()
        if L is not None and hasattr(self, "lo"):
            self._draw_hist(*L.contrast_limits)
