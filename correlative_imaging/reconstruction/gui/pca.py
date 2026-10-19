"""Particle map: every particle in the holes, in one of two spaces.

  features (PCA)       a PCA of the features the classifier sees (median-imputed, standardised). Classes overlap
                       here: the classifier separates them with non-linear rules a linear map cannot show.
  model probabilities  each class is a corner of a pentagon; a particle sits at the probability-weighted mean of the
                       corners. Sure particles sit at a corner, unsure ones between corners: where to label.

Colour by the Detect preview class, the model's class, the hand label, the channel, the plate or the threshold
level. Hover shows the particle, a click opens its well with the particle picked; the particles of the well on screen
are ringed. Shift-drag a rectangle to select particles, then "Queue selected" puts them on a label-queue page."""
from __future__ import annotations

import numpy as np
import pandas as pd
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from ..features import FEATURES
from .theme import CLASS_COLORS

DCOL = {"inside": "#ffd23f", "neighbour": "#9aa5b1", "debris": "#c77dff", "noise": "#5c6670"}
PALETTE = ["#4fc3f7", "#f08a24", "#2bb3a3", "#e05a8a", "#a3d977", "#c77dff", "#ffd23f", "#8a929c"]


class ParticleMap(QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app
        v = QVBoxLayout(self)
        v.setContentsMargins(4, 4, 4, 4)
        h = QHBoxLayout()
        h.addWidget(QLabel("colour by"))
        self.by = QComboBox()
        self.by.addItems(["preview class", "model class", "hand label", "channel", "plate", "threshold level"])
        self.by.currentTextChanged.connect(self.draw)
        h.addWidget(self.by)
        h.addWidget(QLabel("space"))
        self.space = QComboBox()
        self.space.addItems(["features (PCA)", "model probabilities"])
        self.space.currentTextChanged.connect(self._place)
        h.addWidget(self.space)
        self.b_queue = QPushButton("Queue selected")
        self.b_queue.setEnabled(False)
        self.b_queue.setToolTip("shift-drag a rectangle over points first; they become a page of the label queue")
        self.b_queue.clicked.connect(self.queue_selected)
        h.addWidget(self.b_queue)
        self.only_lab = QCheckBox("hand-labelled only")
        self.only_lab.toggled.connect(self.draw)
        h.addWidget(self.only_lab)
        h.addStretch(1)
        self.info = QLabel("")
        self.info.setProperty("role", "hint")
        h.addWidget(self.info)
        v.addLayout(h)
        try:
            import pyqtgraph as pg
        except ImportError:
            self.plot = None
            v.addWidget(QLabel("pyqtgraph is not installed: pip install pyqtgraph"))
            return
        pg.setConfigOptions(antialias=True)
        owner = self

        class _VB(pg.ViewBox):                       # shift-drag: select a rectangle of points
            def mouseDragEvent(self, ev, axis=None):
                if ev.modifiers() & Qt.ShiftModifier:
                    ev.accept()
                    a, b = self.mapToView(ev.buttonDownPos()), self.mapToView(ev.pos())
                    r = (min(a.x(), b.x()), min(a.y(), b.y()), abs(a.x() - b.x()), abs(a.y() - b.y()))
                    owner._rect.setRect(*r)
                    owner._rect.show()
                    if ev.isFinish():
                        owner._select(r)
                        owner._rect.hide()
                else:
                    super().mouseDragEvent(ev, axis)
        self.plot = pg.PlotWidget(background=None, viewBox=_VB())
        from qtpy.QtWidgets import QGraphicsRectItem
        self._rect = QGraphicsRectItem()
        self._rect.setPen(pg.mkPen("#ffffff", width=1, style=Qt.DashLine))
        self._rect.hide()
        self.plot.addItem(self._rect)
        self.sel = None
        self.sel_item = pg.ScatterPlotItem(size=9, pen=pg.mkPen("#4fc3f7", width=1.5), brush=None)
        self.plot.addItem(self.sel_item)
        self.plot.showGrid(x=False, y=False)
        self.plot.setMenuEnabled(False)
        self.scatter = pg.ScatterPlotItem(size=6, pen=None, hoverable=True, tip=self._tip)
        self.scatter.sigClicked.connect(self._clicked)
        self.ring = pg.ScatterPlotItem(size=11, pen=pg.mkPen("#ffffff", width=1.5), brush=None)
        self.plot.addItem(self.scatter)
        self.plot.addItem(self.ring)
        self.legend = self.plot.addLegend(offset=(-10, 10))
        v.addWidget(self.plot, 1)
        self.df = pd.DataFrame()
        self.var = None

    # -- data
    def set_results(self, res, labels: pd.DataFrame | None):
        if self.plot is None or res is None:
            return
        from sklearn.decomposition import PCA
        P = res.particles
        P = P[P.pool].reset_index(drop=True)
        if len(P) < 3:
            self.df = pd.DataFrame()
            self.draw()
            return
        X = P[FEATURES].to_numpy(float)
        med = np.nanmedian(X, 0)
        X = np.where(np.isfinite(X), X, np.where(np.isfinite(med), med, 0))
        sd = X.std(0)
        X = (X - X.mean(0)) / np.where(sd > 0, sd, 1)
        pca = PCA(2, random_state=0).fit(X)
        Z = pca.transform(X)
        self.var = pca.explained_variance_ratio_
        cols = ["plate_id", "well_id", "oid", "ch", "level"] + [c for c in ("dclass", "pclass") if c in P]
        self.df = P[cols].assign(f1=Z[:, 0], f2=Z[:, 1])
        from ..labels import CLASSES
        pc = [f"p_{c}" for c in CLASSES]
        if all(c in P for c in pc):
            ang = np.pi / 2 + 2 * np.pi * np.arange(len(CLASSES)) / len(CLASSES)
            V = np.c_[np.cos(ang), np.sin(ang)]
            Pr = np.nan_to_num(P[pc].to_numpy(float))
            Pr = Pr / np.clip(Pr.sum(1, keepdims=True), 1e-9, None)
            jit = np.random.default_rng(0).normal(0, 0.015, (len(P), 2))
            Q = Pr @ V + jit
            self.df = self.df.assign(q1=Q[:, 0], q2=Q[:, 1])
            self._corners = (V, CLASSES)
        else:
            self._corners = None
        self._place()
        lab = {} if labels is None or not len(labels) else \
            {(r.plate_id, r.well_id, r.oid): r.label for r in labels.itertuples()}
        self.df["hand"] = [lab.get(k, "") for k in zip(self.df.plate_id, self.df.well_id, self.df.oid)]
        self.draw()

    def _place(self, *_):
        """Coordinates of the chosen space -> pc1 / pc2 (what draw / mark use)."""
        import pyqtgraph as pg
        if self.plot is None or not len(self.df):
            return
        prob = self.space.currentText() == "model probabilities"
        if prob and "q1" not in self.df:
            self.info.setText("model probabilities: Classify first")
            prob = False
        self.df["pc1"], self.df["pc2"] = (self.df.q1, self.df.q2) if prob else (self.df.f1, self.df.f2)
        for it in getattr(self, "_corner_items", []):
            self.plot.removeItem(it)
        self._corner_items = []
        if prob and self._corners is not None:
            V, names = self._corners
            for (x, y), n in zip(V, names):
                t = pg.TextItem(n, color=CLASS_COLORS.get(n, "#ffffff"), anchor=(0.5, 0.5))
                t.setPos(x * 1.12, y * 1.12)
                self.plot.addItem(t)
                self._corner_items.append(t)
            self.plot.setLabel("bottom", "")
            self.plot.setLabel("left", "")
        else:
            self.plot.setLabel("bottom", f"PC1 ({self.var[0]:.0%})" if self.var is not None else "")
            self.plot.setLabel("left", f"PC2 ({self.var[1]:.0%})" if self.var is not None else "")
        self.sel = None
        self.sel_item.setData([])
        self.b_queue.setEnabled(False)
        self.draw()
        self.plot.autoRange()

    def _select(self, r):
        x0, y0, w, h = r
        d = getattr(self, "shown", pd.DataFrame())
        if not len(d):
            return
        m = (d.pc1 >= x0) & (d.pc1 <= x0 + w) & (d.pc2 >= y0) & (d.pc2 <= y0 + h)
        self.sel = d[m]
        self.sel_item.setData(x=self.sel.pc1.to_numpy(), y=self.sel.pc2.to_numpy())
        self.b_queue.setEnabled(len(self.sel) > 0)
        self.b_queue.setText(f"Queue selected ({len(self.sel)})")

    def queue_selected(self):
        if self.sel is None or not len(self.sel):
            return
        self.app.browser.queue.load_selection(self.sel[["plate_id", "well_id", "oid"]])

    def _colors(self, d) -> tuple[list, dict]:
        by = self.by.currentText()
        if by == "preview class" and "dclass" in d:
            keys, cmap = d.dclass.fillna(""), DCOL
        elif by == "model class" and "pclass" in d:
            keys, cmap = d.pclass.fillna(""), CLASS_COLORS
        elif by == "hand label":
            keys, cmap = d.hand.replace("", "unlabelled"), {**CLASS_COLORS, "unlabelled": "#3a3f47"}
        elif by == "channel":
            keys = d.ch
            cmap = {c: self.app.cfg.color(c) for c in keys.unique()}
        elif by == "plate":
            keys = d.plate_id
            cmap = {p: PALETTE[i % len(PALETTE)] for i, p in enumerate(sorted(keys.unique()))}
        else:
            keys, cmap = d.level.map({"hole": "lenient", "whole": "stringent"}), {"lenient": "#4fc3f7",
                                                                                 "stringent": "#f08a24"}
        keys = list(keys)
        if by in ("preview class", "model class"):        # blind: no class for hidden wells
            t = self.app.panel.train
            tw = self.app.store.test_wells(self.app.res.dataset) if self.app.store is not None and \
                self.app.res is not None else set()
            for i, (p, w) in enumerate(zip(d.plate_id, d.well_id)):
                if t.blind.isChecked() or (t.blind_test.isChecked() and (p, w) in tw):
                    keys[i] = "blind"
            cmap = {**cmap, "blind": "#3a3f47"}
        return keys, cmap

    def draw(self, *_):
        if self.plot is None:
            return
        import pyqtgraph as pg
        self.legend.clear()
        d = self.df
        if self.only_lab.isChecked() and len(d):
            d = d[d.hand != ""]
        self.shown = d.reset_index(drop=True)
        if not len(d):
            self.scatter.setData([])
            self.info.setText("no particles")
            return
        keys, cmap = self._colors(self.shown)
        brushes = [pg.mkBrush(cmap.get(k, "#808080")) for k in keys]
        self.scatter.setData(x=self.shown.pc1.to_numpy(), y=self.shown.pc2.to_numpy(), brush=brushes,
                             data=np.arange(len(self.shown)))
        names = self.app.cfg.display_name
        for k in sorted(set(keys)):
            item = pg.ScatterPlotItem(size=8, pen=None, brush=pg.mkBrush(cmap.get(k, "#808080")))
            self.legend.addItem(item, names(k) if self.by.currentText() == "channel" else str(k))
        self.info.setText(f"{len(self.shown):,} particles in the holes · click one to open its well")
        self.mark(*(getattr(self.app, "_current_well", None) or (None, None)))

    def mark(self, plate_id, well_id, oid=None):
        if self.plot is None or not len(getattr(self, "shown", [])):
            return
        d = self.shown
        sel = d[(d.plate_id == plate_id) & (d.well_id == well_id)]
        if oid is not None:
            sel = sel[sel.oid == oid]
        self.ring.setData(x=sel.pc1.to_numpy(), y=sel.pc2.to_numpy())

    # -- interaction
    def _tip(self, x, y, data):
        r = self.shown.iloc[int(data)]
        cls = "" if self.app.is_blind(r.plate_id, r.well_id) else (r.get("pclass") or r.get("dclass") or "")
        return (f"{r.plate_id} {r.well_id} · {self.app.cfg.display_name(r.ch)} "
                f"{'lenient' if r.level == 'hole' else 'stringent'}\n{cls}" + (f" · hand: {r.hand}" if r.hand else ""))

    def _clicked(self, item, points, ev=None):
        if not len(points):
            return
        r = self.shown.iloc[int(points[0].data())]
        self.app.show_well(r.plate_id, r.well_id)
        P = self.app.display.current["P"] if self.app.display.current else None
        if P is not None:
            m = P[P.oid == r.oid]
            if len(m):
                self.app.pick(m.iloc[0])
        self.mark(r.plate_id, r.well_id, r.oid)
