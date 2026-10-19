"""Batch labeller (Label tab: one for particles, one for cells): pages of thumbnails labelled from the keyboard,
built by tiers.py. Each has two sets:
  Test wells (blind)  the frozen held-out wells, whole wells in turn, no suggestion anywhere: the honest score
  Training            outside the test wells; particles: mixed tiers (ask / confirm / random) with the suggestion
                      (the model's class, or the Detect preview's before any model) unless "hide suggestions" is on;
                      cells: verdict (one cell / several cells that cannot be split by eye / not a cell) and colour

Keys (click the grid first)  ←→↑↓ move · Enter accept the suggestion · Space open the well in the canvas ·
  Backspace skip.  Particles: 1–5 cell / neighbour / merged / debris / noise.
  Cells: 1 one cell · 2 several cells · 3 not a cell · Q W E R toggle identity channel 1–4 · 0 colour none.

Stored source: "blind" (no suggestion was visible), "accepted" (suggestion kept), "corrected" (changed). Thumbnails
are drawn in a worker thread (the image is read back from the run's TIF) and the next page is prepared ahead.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import QImage, QPixmap
from qtpy.QtWidgets import (QCheckBox, QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea,
                            QSizePolicy, QVBoxLayout, QWidget)

from ..labels import CLASSES, VERDICTS
from .theme import CLASS_COLORS, CLASS_KEYS

TILE = 120
VERDICT_KEYS = {Qt.Key_1: "one cell", Qt.Key_2: "several cells", Qt.Key_3: "not a cell"}
CH_KEYS = [Qt.Key_Q, Qt.Key_W, Qt.Key_E, Qt.Key_R]
VERDICT_COLORS = {"one cell": "#2bb3a3", "several cells": "#e0a526", "not a cell": "#8a929c"}
DESCRIPTION = {
    "particle": ("<b>What is this particle?</b> 1 cell · 2 neighbour · 3 merged · 4 debris · 5 noise · U unsure",
                 "One tile per object in a hole; trains the particle classifier.\n"
                 "1 cell: the patched cell or part of it\n2 neighbour: a cell next to the hole reaching in\n"
                 "3 merged: one particle holding two cells\n4 debris: dirt, plug, crystal\n"
                 "5 noise: background, haze"),
    "cell": ("<b>What is this cell?</b> 1 one cell · 2 several cells · 3 not a cell · U unsure — colour: Q W E R, 0 none",
             "One tile per reconstructed cell (needs Classify; before any model the Detect preview stands in).\n"
             "2 several cells: two or more cells that cannot be split by eye, either could be the patched one.\n"
             "Colour: the identity channels the cell is positive for. Checks and calibrates the colour call and "
             "lends labels to the cell's particles."),
}
SETS = {
    "test": ("Test wells (blind)", "held-out wells, no suggestions: the honest score. Label these first (≥ 30)."),
    "train": ("Training", "all other wells; the suggestion under each tile: Enter keeps it, a key changes it."),
}


def _rgb(hex_color: str) -> np.ndarray:
    h = (hex_color or "#ffffff").lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], float) / 255


class _Wells:
    """Small LRU of (WellData, maps, G, per-channel display limits) per well, shared by the render threads."""

    def __init__(self, res, n=24):
        self.res, self.n, self.d, self.lock = res, n, OrderedDict(), threading.Lock()

    def get(self, plate_id, well_id):
        k = (plate_id, well_id)
        with self.lock:
            if k in self.d:
                self.d.move_to_end(k)
                return self.d[k]
        got = self.res.well(plate_id, well_id)
        if got is None:
            return None
        wd, maps = got
        b = wd.box_slices
        # from the frame background up to the brightest pixels, but never closer than 15 noise SDs: an empty hole
        # stays dark instead of its noise being stretched into "signal"
        lim = []
        for c, ch in zip(wd.chans, wd.names):
            bm, bs = wd.bg.get(ch, (float(np.percentile(c[b], 5)), float(np.std(c[b]))))
            lim.append((float(bm), max(float(np.percentile(c[b], 99.8)), float(bm) + 15 * float(bs))))
        G = self.res.recon(plate_id, well_id)
        v = (wd, maps, G, lim)
        with self.lock:
            self.d[k] = v
            while len(self.d) > self.n:
                self.d.popitem(last=False)
        return v


def render(wells: _Wells, row, kind: str, cfg, grey: bool = False) -> np.ndarray | None:
    """RGB uint8 TILE x TILE: the channels in their colours (well-wide limits), the object's outline white, the hole
    grey. grey: only the particle's own channel, in greyscale (limits from that channel alone), outline yellow."""
    from skimage.segmentation import find_boundaries
    got = wells.get(row.plate_id, row.well_id)
    if got is None:
        return None
    wd, maps, G, lim = got
    R = wd.R_px
    if kind == "cell":
        if G is None:
            return None
        mask = G == int(row.cell_id)
        if not mask.any():
            return None
        ys, xs = np.nonzero(mask)
        cy, cx = ys.mean(), xs.mean()
        half = int(max(1.6 * R, 0.6 * max(np.ptp(ys), np.ptp(xs)) + 6))
    else:
        L = maps.get((row.level, row.ch))
        if L is None:
            return None
        mask = L == int(row.lab)
        cy, cx = float(row.iy), float(row.ix)
        ys, xs = np.nonzero(mask)
        ext = max(np.ptp(ys), np.ptp(xs)) if len(ys) else 0
        half = int(max(1.3 * R, 0.6 * ext + 6))
    H, W = mask.shape
    y0, y1 = int(max(cy - half, 0)), int(min(cy + half + 1, H))
    x0, x1 = int(max(cx - half, 0)), int(min(cx + half + 1, W))
    img = np.zeros((y1 - y0, x1 - x0, 3))
    one = grey and kind == "particle" and row.ch in wd.names
    for k, ch in enumerate(wd.names):
        if one and ch != row.ch:
            continue
        lo, hi = lim[k]
        a = np.clip((wd.chans[k][y0:y1, x0:x1] - lo) / max(hi - lo, 1e-6), 0, 1)
        img += a[..., None] * (np.ones(3) if one else _rgb(cfg.color(ch)))
    img = np.clip(img, 0, 1)
    hole = find_boundaries(wd.hole[y0:y1, x0:x1], mode="inner")
    img[hole] = 0.55 * img[hole] + 0.45 * (np.array([0.3, 0.55, 0.95]) if one else np.array([0.6, 0.6, 0.6]))
    edge = find_boundaries(mask[y0:y1, x0:x1], mode="inner")
    img[edge] = np.array([1.0, 0.82, 0.25]) if one else 1.0
    out = (img * 255).astype(np.uint8)
    # nearest-neighbour up/down to a square tile
    hh, ww = out.shape[:2]
    s = TILE / max(hh, ww)
    yi = np.minimum((np.arange(int(hh * s)) / s).astype(int), hh - 1)
    xi = np.minimum((np.arange(int(ww * s)) / s).astype(int), ww - 1)
    out = out[yi][:, xi]
    tile = np.zeros((TILE, TILE, 3), np.uint8)
    tile[:out.shape[0], :out.shape[1]] = out
    return np.ascontiguousarray(tile)


class Tile(QFrame):
    def __init__(self, queue, i):
        super().__init__()
        self.q, self.i = queue, i
        self.setFrameShape(QFrame.Box)
        self.setLineWidth(2)
        v = QVBoxLayout(self)
        v.setContentsMargins(3, 3, 3, 3)
        v.setSpacing(2)
        self.img = QLabel("…")
        self.img.setFixedSize(TILE, TILE)
        self.img.setAlignment(Qt.AlignCenter)
        v.addWidget(self.img)
        self.cap = QLabel("")
        self.cap.setTextFormat(Qt.RichText)
        self.cap.setWordWrap(True)
        self.cap.setFixedWidth(TILE)
        v.addWidget(self.cap)
        self.state = QLabel("")
        self.state.setTextFormat(Qt.RichText)
        self.state.setFixedWidth(TILE)
        v.addWidget(self.state)
        self.setFixedSize(TILE + 12, TILE + 64)          # rows never squeezed: the page scrolls instead
        self.set_focus(False)

    def set_focus(self, on, selected=False):
        col = "#4fc3f7" if on else ("#f08a24" if selected else "transparent")
        bg = "rgba(240,138,36,0.18)" if selected else "transparent"
        self.setStyleSheet("Tile { border: 2px solid %s; border-radius: 4px; background: %s; }" % (col, bg))

    def mousePressEvent(self, e):
        self.q.click(self.i, e.modifiers())
        self.q.grid_host.setFocus()

    def mouseDoubleClickEvent(self, e):
        self.q.focus(self.i)
        self.q.open_focused()


class LabelQueue(QWidget):
    def __init__(self, app, kind: str = "particle"):
        super().__init__()
        self.app = app
        self._kind = kind
        outer = QVBoxLayout(self)
        outer.setContentsMargins(2, 4, 2, 2)
        d = QLabel(DESCRIPTION[kind][0] + "  <span style='color:#8a929c'>(hover for more)</span>")
        d.setWordWrap(True)
        d.setTextFormat(Qt.RichText)
        d.setToolTip(DESCRIPTION[kind][1])
        outer.addWidget(d)
        bar = QHBoxLayout()
        from qtpy.QtWidgets import QButtonGroup, QRadioButton
        self.set_group = QButtonGroup(self)
        self.set_btn = {}
        for k in ("test", "train"):
            b = QRadioButton(SETS[k][0])
            b.setToolTip(SETS[k][1])
            self.set_group.addButton(b)
            self.set_btn[k] = b
            bar.addWidget(b)
        self.set_btn["test"].setChecked(True)
        self.set_group.buttonToggled.connect(lambda b, on: on and self._set_changed())
        bar.addStretch(1)
        self.b_next = QPushButton("New page  (N)")
        self.b_next.clicked.connect(self.new_page)
        bar.addWidget(self.b_next)
        outer.addLayout(bar)
        self.set_hint = QLabel(SETS["test"][1])
        self.set_hint.setWordWrap(True)
        self.set_hint.setProperty("role", "hint")
        outer.addWidget(self.set_hint)
        bar = QHBoxLayout()
        self.only = QComboBox()
        self.only.addItems(["mixed"] + ([f"suggested {c}" for c in CLASSES] if kind == "particle" else []))
        self.only.setToolTip("training pages: a mix of unsure, probable and random objects (recommended), or only the "
                             "objects suggested as one class, least sure first (e.g. confirm all noise in one go)")
        self.only.currentIndexChanged.connect(lambda *_: (not self.test) and self.new_page())
        self.only.setVisible(kind == "particle")
        bar.addWidget(self.only)
        self.grey = QCheckBox("this channel only (grey)")
        self.grey.setToolTip("show only the particle's own channel in greyscale, scaled to that channel alone (other "
                             "channels' noise does not distract)")
        self.grey.toggled.connect(lambda *_: self._rerender())
        self.grey.setVisible(kind == "particle")
        bar.addWidget(self.grey)
        bar.addStretch(1)
        outer.addLayout(bar)
        bar = QHBoxLayout()
        self.blind = QCheckBox("hide suggestions")
        self.blind.setToolTip("label the training pages blind too (test wells are always blind)")
        self.blind.toggled.connect(lambda *_: self.redraw_captions())
        bar.addWidget(self.blind)
        self.b_accept = QPushButton("Accept all suggestions")
        self.b_accept.setToolTip("store the suggestion of every tile on this page that you have not labelled or skipped")
        self.b_accept.clicked.connect(self.accept_all)
        bar.addWidget(self.b_accept)
        bar.addStretch(1)
        self.src = QLabel("")
        self.src.setProperty("role", "hint")
        bar.addWidget(self.src)
        outer.addLayout(bar)
        self.info = QLabel("")
        self.info.setProperty("role", "hint")
        outer.addWidget(self.info)
        self.keys = QLabel("")
        self.keys.setProperty("role", "hint")
        self.keys.setWordWrap(True)
        outer.addWidget(self.keys)
        self.grid_host = QWidget()
        self.grid_host.setFocusPolicy(Qt.StrongFocus)
        self.grid_host.keyPressEvent = self.key
        self.grid = QGridLayout(self.grid_host)
        self.grid.setSpacing(6)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(self.grid_host)
        outer.addWidget(sc, 1)
        self.rows = pd.DataFrame()
        self.tiles = []
        self.state = {}                    # tile index -> dict(label / verdict / colour / skipped / unsure)
        self.cur = 0
        self.sel = set()                   # multi-selected tiles (keys act on all of them)
        self.done = set()                  # keys shown this session (labelled or skipped)
        self.pool = ThreadPoolExecutor(4)
        self.futs = {}
        self.wells = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._poll)
        self.n = 24

    # -- data
    @property
    def kind(self):
        return self._kind

    @property
    def test(self):
        return self.set_btn["test"].isChecked()

    def _set_changed(self):
        k = "test" if self.test else "train"
        self.set_hint.setText(SETS[k][1])
        self.blind.setEnabled(k == "train")
        self.b_accept.setVisible(k == "train")
        if self.app.res is not None:
            self.new_page()

    def set_results(self, res):
        self.wells = _Wells(res) if res is not None else None
        self.rows = pd.DataFrame()
        self.done = set()
        self._clear()
        self.info.setText("press New page to start" if res is not None else "open results first")

    def new_page(self):
        app = self.app
        if app.res is None or app.store is None:
            self.info.setText("open results and choose a label library first")
            return
        from .. import tiers
        if self.test and not app.store.test_wells(app.res.dataset):
            app.freeze_test_wells()
        try:
            if self.kind == "cell":
                if app.res.cells is None:
                    self.info.setText("cells need Classify first (with a model, or the Detect preview: Run tab → "
                                      "Classify)")
                    self._clear()
                    return
                rows = tiers.cell_page(app.res, app.store, n=max(self.n // 2, 6), mode="test" if self.test else "train",
                                       done=self.done)
            else:
                self._m = tiers.matched(app.res, app.store)
                only = None if self.test or self.only.currentIndex() == 0 else CLASSES[self.only.currentIndex() - 1]
                rows = tiers.particle_page(app.res, app.store, app.model, n=self.n,
                                           mode="test" if self.test else "train", done=self.done, m=self._m,
                                           only=only)
        except Exception as e:
            self.info.setText(f"queue failed: {e}")
            return
        m = app.model
        arm = getattr(m, "spec", {}).get("arm") if m is not None else None
        self.src.setText("suggestions: " + ("Detect preview (no model)" if arm in (None, "preview")
                                            else f"model {app.model_file or arm}")
                         if self.kind == "particle" else "suggestions: the call")
        self.rows = rows.reset_index(drop=True)
        self.state = {}
        self.sel = set()
        self.cur = 0
        self._build()
        self.update_info()
        self._prefetch()

    def _prefetch(self):
        """Read the wells of a likely next page ahead (they stay in the well cache)."""
        app = self.app
        if self.kind != "particle" or self.wells is None:
            return
        from .. import tiers
        done = self.done | {self._key(r) for r in self.rows.itertuples(index=False)}
        res, store, model, mode, n, wells = app.res, app.store, app.model, "test" if self.test else "train", self.n, \
            self.wells

        def job():                                      # in the background: never blocks the keys
            nxt = tiers.particle_page(res, store, model, n=n, mode=mode, done=done)
            for r in nxt.drop_duplicates(["plate_id", "well_id"]).itertuples(index=False):
                wells.get(r.plate_id, r.well_id)
        self.pool.submit(job)

    def load_selection(self, keys: pd.DataFrame):
        """A page of hand-picked particles (from the particle map); test wells are left out (blind elsewhere)."""
        app = self.app
        from .. import tiers
        T = tiers.particle_tiers(app.res, app.model)
        test = app.store.test_wells(app.res.dataset) if app.store is not None else set()
        k = set(map(tuple, keys.to_numpy()))
        T = T[[x in k and (x[0], x[1]) not in test for x in zip(T.plate_id, T.well_id, T.oid)]]
        if not len(T):
            self.info.setText("nothing to queue (test wells and lenient/stringent duplicates are left out)")
            return
        if self.test:
            self.set_group.blockSignals(True)
            self.set_btn["train"].setChecked(True)
            self.set_group.blockSignals(False)
            self._set_changed_quiet()
        self.rows = T.head(60).assign(blind=False).reset_index(drop=True)
        self.state, self.cur, self.sel = {}, 0, set()
        self._build()
        self.update_info()
        self.app.show_batch("particle")

    def _set_changed_quiet(self):
        k = "test" if self.test else "train"
        self.set_hint.setText(SETS[k][1])
        self.blind.setEnabled(k == "train")
        self.b_accept.setVisible(k == "train")

    def _key(self, r):
        return (r.plate_id, r.well_id, r.cell_id) if self.kind == "cell" else (r.plate_id, r.well_id, r.oid)

    # -- tiles
    def _clear(self):
        for t in self.tiles:
            t.setParent(None)
        self.tiles = []
        self.futs = {}

    def _build(self):
        self._clear()
        cols = self._cols()
        for i, r in enumerate(self.rows.itertuples(index=False)):
            t = Tile(self, i)
            self.tiles.append(t)
            self.grid.addWidget(t, i // cols, i % cols)
            self.futs[i] = self.pool.submit(render, self.wells, r, self.kind, self.app.cfg, self.grey.isChecked())
        self.redraw_captions()
        self.keys.setText(
            "Click a tile, then: ←→↑↓ move · Enter keep the suggestion · U unsure · Backspace skip · Space show the "
            "well in the canvas · N new page. Ctrl+click / Shift+click / Ctrl+A select several: a key then labels "
            "them all, Enter keeps all their suggestions" + (" · " + " · ".join(f"{k} {self.app.cfg.display_name(c)}"
                                                               for k, c in zip("QWER", self._chans()))
                                             if self.kind == "cell" else ""))
        if self.tiles:
            self.focus(0)
        self.timer.start(80)
        self.grid_host.setFocus()

    def _rerender(self):
        for i, r in enumerate(self.rows.itertuples(index=False)):
            if i < len(self.tiles):
                self.futs[i] = self.pool.submit(render, self.wells, r, self.kind, self.app.cfg, self.grey.isChecked())
        self.timer.start(80)

    # -- selection: click focuses; Ctrl+click toggles, Shift+click extends; keys act on the selection
    def click(self, i, mods):
        if mods & Qt.ControlModifier:
            self.sel ^= {i}
        elif mods & Qt.ShiftModifier:
            a, b = sorted((self.cur, i))
            self.sel |= set(range(a, b + 1))
        else:
            self.sel = set()
        self.focus(i)

    def targets(self) -> list:
        return sorted(self.sel) if self.sel else [self.cur]

    def _paint(self):
        for j, t in enumerate(self.tiles):
            t.set_focus(j == self.cur, j in self.sel)

    def _poll(self):
        for i, f in list(self.futs.items()):
            if f.done():
                del self.futs[i]
                if i >= len(self.tiles):
                    continue
                try:
                    a = f.result()
                except Exception as e:
                    self.tiles[i].img.setText(f"image not readable:\n{e}"[:120])
                    continue
                if a is None:
                    self.tiles[i].img.setText("no image")
                    continue
                q = QImage(a.data, a.shape[1], a.shape[0], 3 * a.shape[1], QImage.Format_RGB888)
                self.tiles[i].img.setPixmap(QPixmap.fromImage(q.copy()))
        if not self.futs:
            self.timer.stop()

    def _chans(self):
        res = self.app.res
        return (res.run.get("identity_channels") or res.channels) if res is not None else []

    def _is_blind(self, r):
        return bool(getattr(r, "blind", False)) or self.blind.isChecked()

    def _suggest(self, r):
        if self.kind == "cell":
            call = str(getattr(r, "call", "") or "")
            return call if call and call not in ("uncertain", "no cell", "nan") else ""
        return r.suggest if isinstance(r.suggest, str) else ""

    def redraw_captions(self):
        disp = self.app.cfg.display_name
        for i, r in enumerate(self.rows.itertuples(index=False)):
            if i >= len(self.tiles):
                break
            blind = self._is_blind(r)
            if self.kind == "cell":
                head = f"{r.plate_id} {r.well_id} · cell {r.cell_id}"
                sug = "" if blind else (f"call: <b>{'+'.join(disp(c) for c in self._suggest(r).split('+'))}</b>"
                                        if self._suggest(r) else f"call: {getattr(r, 'call', '')}")
            else:
                head = f"{r.plate_id} {r.well_id} · {disp(r.ch)} {'lenient' if r.level == 'hole' else 'stringent'}"
                s = self._suggest(r)
                sug = "" if blind or not s else (f"<span style='color:{CLASS_COLORS.get(s, '#ccc')}'>■</span> "
                                                 f"{s} <span style='color:#8a929c'>({r.tier}, "
                                                 f"{r.conf:.2f})</span>")
            self.tiles[i].cap.setText(f"<span style='font-size:9px'>{head}</span><br>{sug or ('blind' if blind else '')}")
            self._draw_state(i)
        self.b_accept.setEnabled(not self.test and not self.blind.isChecked())

    def _draw_state(self, i):
        st = self.state.get(i, {})
        disp = self.app.cfg.display_name
        if st.get("skipped"):
            txt = "<span style='color:#8a929c'>skipped</span>"
        elif st.get("unsure"):
            txt = "<span style='color:#e0a526'>? unsure</span>"
        elif self.kind == "cell":
            v, c = st.get("verdict"), st.get("colour")
            txt = (f"<span style='color:{VERDICT_COLORS.get(v, '#ccc')}'>■ {v}</span>" if v else "") + \
                  (f" · <b>{'+'.join(disp(x) for x in c.split('+')) if c and c != 'none' else 'none'}</b>"
                   if c is not None else "")
        else:
            lab = st.get("label")
            txt = f"<span style='color:{CLASS_COLORS.get(lab, '#ccc')}'>■ <b>{lab}</b></span>" if lab else ""
        self.tiles[i].state.setText(txt)

    def focus(self, i):
        if not self.tiles:
            return
        self.cur = max(0, min(i, len(self.tiles) - 1))
        self._paint()

    # -- labelling
    def _row(self, i=None):
        return self.rows.iloc[self.cur if i is None else i]

    def _source(self, r, value, suggestion):
        if self._is_blind(r):
            return "blind"
        return "accepted" if suggestion and value == suggestion else "corrected"

    def set_class(self, cls, i=None):
        i = self.cur if i is None else i
        r = self._row(i)
        app = self.app
        y, x = app.frame_point(r)
        app.store.set(app.res.dataset, r.plate_id, r.well_id, r.ch, "any", y, x, cls,
                      source=self._source(r, cls, self._suggest(r)))
        self.state[i] = {"label": cls}
        self.done.add(self._key(r))
        self._draw_state(i)

    def set_cell(self, verdict=None, colour=None, i=None):
        i = self.cur if i is None else i
        r = self._row(i)
        app = self.app
        st = self.state.setdefault(i, {})
        st.pop("skipped", None)
        if verdict is not None:
            st["verdict"] = verdict
        if colour is not None:
            st["colour"] = colour
        y, x = app.cell_point(r)
        if y is None:
            self.info.setText("this cell has no seed pixel to store the label at")
            return
        sug = self._suggest(r)
        src = "blind" if self._is_blind(r) else ("accepted" if (st.get("colour") == sug and
                                                                st.get("verdict") in (None, "one cell"))
                                                 else "corrected")
        app.store.set_cell(app.res.dataset, r.plate_id, r.well_id, y, x, verdict=st.get("verdict"),
                           colour=st.get("colour"), source=src)
        self.done.add(self._key(r))
        self._draw_state(i)

    def toggle_colour(self, k, i=None):
        i = self.cur if i is None else i
        chans = self._chans()
        if k >= len(chans):
            return
        st = self.state.get(i, {})
        cur = [c for c in (st.get("colour") or "").split("+") if c and c != "none"]
        ch = chans[k]
        cur = [c for c in cur if c != ch] if ch in cur else cur + [ch]
        self.set_cell(colour="+".join(c for c in chans if c in cur) or "none", i=i)

    def accept(self, i=None):
        i = self.cur if i is None else i
        r = self._row(i)
        if self._is_blind(r):
            self.info.setText("blind: no suggestion to keep, choose the class (or U unsure)")
            return
        s = self._suggest(r)
        if self.kind == "cell":
            self.set_cell(verdict="one cell", colour=s or "none", i=i)
        elif s:
            self.set_class(s, i)

    def accept_all(self):
        if self.test or self.blind.isChecked():
            return
        for i in range(len(self.tiles)):
            if i not in self.state:
                self.accept(i)
        self.update_info()

    def skip(self, i=None):
        i = self.cur if i is None else i
        r = self._row(i)
        self.state[i] = {"skipped": True}
        self.done.add(self._key(r))
        self._draw_state(i)

    def unsure(self, i=None):
        """Not sure what it is: recorded (never trained on) and not shown again."""
        i = self.cur if i is None else i
        r = self._row(i)
        app = self.app
        if self.kind == "cell":
            y, x = app.cell_point(r)
            if y is not None:
                app.store.set_cell(app.res.dataset, r.plate_id, r.well_id, y, x, verdict="", colour="",
                                   source="unsure")
        else:
            y, x = app.frame_point(r)
            app.store.mark_unsure(app.res.dataset, r.plate_id, r.well_id, r.ch, "any", y, x)
        self.state[i] = {"unsure": True}
        self.done.add(self._key(r))
        self._draw_state(i)

    def open_focused(self):
        if not len(self.rows):
            return
        r = self._row()
        self.app.show_well(r.plate_id, r.well_id)
        if self.kind == "particle":
            P = self.app.display.current["P"] if self.app.display.current else None
            if P is not None:
                m = P[P.oid == r.oid]
                if len(m):
                    self.app.pick(m.iloc[0])

    def _after(self, last=None):
        self._n_since = getattr(self, "_n_since", 0) + 1
        if self._n_since >= 10:                          # the full progress (all labels matched) now and then
            self._n_since = 0
            self.update_info()
        else:
            self.info.setText(f"page {len(self.state)}/{len(self.tiles)}" + self.info.text()[self.info.text().find(" ·"):]
                              if " ·" in self.info.text() else f"page {len(self.state)}/{len(self.tiles)}")
        self.sel = set()
        if self.tiles and len(self.state) >= len(self.tiles):
            QTimer.singleShot(250, self.new_page)
        else:
            self.focus((self.cur if last is None else last) + 1)

    def key(self, e):
        k = e.key()
        cols = self._cols()
        if not self.tiles:
            if k == Qt.Key_N:
                self.new_page()
            return
        T = self.targets()
        last = max(T)
        ctrl = bool(e.modifiers() & Qt.ControlModifier)
        if k == Qt.Key_A and ctrl:
            self.sel = set(range(len(self.tiles)))
            self._paint()
        elif k == Qt.Key_Escape:
            self.sel = set()
            self._paint()
        elif k == Qt.Key_Right:
            self.focus(self.cur + 1)
        elif k == Qt.Key_Left:
            self.focus(self.cur - 1)
        elif k == Qt.Key_Down:
            self.focus(self.cur + cols)
        elif k == Qt.Key_Up:
            self.focus(self.cur - cols)
        elif k in (Qt.Key_Return, Qt.Key_Enter):
            for i in T:
                self.accept(i)
            self._after(last)
        elif k == Qt.Key_Space:
            self.open_focused()
        elif k == Qt.Key_Backspace:
            for i in T:
                self.skip(i)
            self._after(last)
        elif k == Qt.Key_U:
            for i in T:
                self.unsure(i)
            self._after(last)
        elif k == Qt.Key_N:
            self.new_page()
        elif self.kind == "cell" and k in VERDICT_KEYS:
            for i in T:
                self.set_cell(verdict=VERDICT_KEYS[k], i=i)
            st = self.state.get(self.cur, {})
            if len(T) > 1 or (st.get("verdict") and (st.get("colour") or st.get("verdict") == "not a cell")):
                self._after(last)
        elif self.kind == "cell" and k in CH_KEYS:
            for i in T:
                self.toggle_colour(CH_KEYS.index(k), i)
        elif self.kind == "cell" and k == Qt.Key_0:
            for i in T:
                self.set_cell(colour="none", i=i)
        elif self.kind == "particle" and Qt.Key_1 <= k <= Qt.Key_5:
            cls = {CLASS_KEYS[c]: c for c in CLASSES}.get(chr(k))
            if cls:
                for i in T:
                    self.set_class(cls, i)
                self._after(last)
        else:
            return
        e.accept()

    def update_info(self):
        app = self.app
        n_done = len(self.state)
        if app.res is None or app.store is None:
            self.info.setText("open results and choose a label library first")
            app.update_status()
            return
        try:
            from ..tiers import label_progress
            m = getattr(self, "_m", None)
            self._m = None                                 # one shared match per page, fresh afterwards
            pr = label_progress(app.res, app.store, m=m)
            t = pr["test"]
            self.info.setText(f"page {n_done}/{len(self.tiles)} · test wells: {t['labelled']}/{t['objects']} objects "
                              f"labelled in {t['wells']} wells")
            app.panel.train.show_progress(pr)
            app.update_status(pr)
        except Exception as e:
            self.info.setText(f"page {n_done}/{len(self.tiles)} ({e})")

    def _cols(self):
        return max(1, (self.width() - 40) // (TILE + 20))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self.tiles:
            cols = self._cols()
            for i, t in enumerate(self.tiles):
                self.grid.addWidget(t, i // cols, i % cols)
