"""Control view (left, tabbed with napari's layer list and layer controls): the main place to steer the canvas.

  plate          the plate view (plate + colouring dropdowns); click a well to open it; "Label this well" puts
                 every object of the well on screen onto a bulk-labeller page (labelled ones too, to relabel them)
  channels       the shared channel histogram (gui_common/histogram.py): one button per channel as the legend (click
                 the name to adjust it, the dot to show / hide it), every channel's histogram overlaid, min / max as
                 handles inside the plot, gamma, auto. Auto-contrast on well change: minimum -> 99.9th percentile, at
                 least minimum + 100
  particles      the per-channel particle fills (full = both thresholds agree)
  overlays       neighbours, reconstructed cell, class tags, edits, brightfield
"""
from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (QCheckBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea,
                            QVBoxLayout, QWidget)

from ...gui_common.histogram import ChannelHistogram
from .display import contrast_rule

OVERLAYS = [("neighbours", "neighbours (grey, N)"), ("cells", "cell outline"), ("tags", "class tags"),
            ("splits", "edits / splits"), ("bf", "brightfield")]


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

        # channels: the shared histogram
        v.addWidget(_section("Channels"))
        self.auto_cb = QCheckBox("auto-contrast on well change")
        self.auto_cb.setToolTip("every channel: black point = its minimum, white point = 99.9th percentile but at least "
                                "minimum + 100")
        self.auto_cb.setChecked(True)
        self.auto_cb.toggled.connect(lambda on: setattr(app.display, "auto_contrast", on))
        v.addWidget(self.auto_cb)
        self.chans = []
        self.hist = ChannelHistogram(
            app.v, layers_fn=lambda: [(c, app.display.L.get(f"img:{c}")) for c in self.chans],
            name_fn=lambda c, L: app.cfg.display_name(c), color_fn=lambda c, L: app.cfg.color(c),
            auto_rule=contrast_rule)
        v.addWidget(self.hist)

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
        self.part_cb = {}

    # -- plate
    def set_plate(self, plate_widget):
        self.plate_box.addWidget(plate_widget)

    # -- layers
    def _set_vis(self, role, on):
        L = self.app.display.L.get(role)
        if L is not None:
            L.visible = on

    def refresh(self):
        """After a well is shown: rebuild the buttons if the channels changed, else only redraw."""
        d = self.app.display
        chans = [r[4:] for r in d.L if r.startswith("img:")]
        sig = tuple((c, id(d.L[f"img:{c}"])) for c in chans)
        if sig != self._sig:
            self._sig = sig
            self.chans = chans
            self._build(chans)
            self.hist.rebuild()
        else:
            self.hist._redraw()

    def _build(self, chans):
        """Particle switches per channel; overlay switches follow their layers."""
        d, cfg = self.app.display, self.app.cfg
        for w in self.part_cb.values():
            w.setParent(None)
        self.part_cb = {}
        for c in chans:
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
