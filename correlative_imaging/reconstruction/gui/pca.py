"""Particle map: a PCA of the features of every particle in the holes (the same 27 numbers the classifier sees).

Each point is a particle; colour by the Detect preview class, the model's class, the hand label, the channel or the
plate. Hover shows the particle, a click opens its well with the particle picked; the particles of the well on
screen are ringed. The PCA is fitted on the open results (median-imputed, standardised features)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

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
        self.plot = pg.PlotWidget(background=None)
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
        self.df = P[cols].assign(pc1=Z[:, 0], pc2=Z[:, 1])
        lab = {} if labels is None or not len(labels) else \
            {(r.plate_id, r.well_id, r.oid): r.label for r in labels.itertuples()}
        self.df["hand"] = [lab.get(k, "") for k in zip(self.df.plate_id, self.df.well_id, self.df.oid)]
        self.plot.setLabel("bottom", f"PC1 ({self.var[0]:.0%})")
        self.plot.setLabel("left", f"PC2 ({self.var[1]:.0%})")
        self.draw()

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
        return list(keys), cmap

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
        cls = r.get("pclass") or r.get("dclass") or ""
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
