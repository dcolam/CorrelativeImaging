"""Label queue (bottom dock): a page of thumbnails to label from the keyboard, built by tiers.py.

Modes
  Particles             objects outside the test wells, mixed tiers (ask / confirm / random), with the suggestion
                        (the model's class, or the Detect preview's before any model) unless "blind" is on
  Particles · test      the frozen test wells, whole wells in turn, always blind (no suggestion anywhere)
  Cells                 reconstructed cells (after Classify): verdict (one cell / several cells that cannot be split
                        by eye / not a cell) and colour (which identity channels it is positive for)
  Cells · test          the same in the test wells, blind

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

MODES = ["Particles", "Particles · test wells (blind)", "Cells", "Cells · test wells (blind)"]
TILE = 150
VERDICT_KEYS = {Qt.Key_1: "one cell", Qt.Key_2: "several cells", Qt.Key_3: "not a cell"}
CH_KEYS = [Qt.Key_Q, Qt.Key_W, Qt.Key_E, Qt.Key_R]
VERDICT_COLORS = {"one cell": "#2bb3a3", "several cells": "#e0a526", "not a cell": "#8a929c"}


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


def render(wells: _Wells, row, kind: str, cfg) -> np.ndarray | None:
    """RGB uint8 TILE x TILE: the channels in their colours (well-wide limits), the object's outline white, the hole
    grey."""
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
    for k, ch in enumerate(wd.names):
        lo, hi = lim[k]
        a = np.clip((wd.chans[k][y0:y1, x0:x1] - lo) / max(hi - lo, 1e-6), 0, 1)
        img += a[..., None] * _rgb(cfg.color(ch))
    img = np.clip(img, 0, 1)
    hole = find_boundaries(wd.hole[y0:y1, x0:x1], mode="inner")
    img[hole] = 0.55 * img[hole] + 0.45 * np.array([0.6, 0.6, 0.6])
    edge = find_boundaries(mask[y0:y1, x0:x1], mode="inner")
    img[edge] = 1.0
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
        self.set_focus(False)

    def set_focus(self, on):
        self.setStyleSheet("Tile { border: 2px solid %s; border-radius: 4px; }" % ("#4fc3f7" if on else "transparent"))

    def mousePressEvent(self, e):
        self.q.focus(self.i)
        self.q.grid_host.setFocus()

    def mouseDoubleClickEvent(self, e):
        self.q.focus(self.i)
        self.q.open_focused()


class LabelQueue(QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        bar = QHBoxLayout()
        self.mode = QComboBox()
        self.mode.addItems(MODES)
        self.mode.currentIndexChanged.connect(lambda *_: self.new_page())
        bar.addWidget(self.mode)
        self.blind = QCheckBox("blind")
        self.blind.setToolTip("hide the suggestions on training pages too (test wells are always blind)")
        self.blind.toggled.connect(lambda *_: self.redraw_captions())
        bar.addWidget(self.blind)
        self.b_next = QPushButton("New page  (N)")
        self.b_next.clicked.connect(self.new_page)
        bar.addWidget(self.b_next)
        self.b_accept = QPushButton("Accept all suggestions")
        self.b_accept.setToolTip("store the suggestion of every tile on this page that you have not labelled or skipped")
        self.b_accept.clicked.connect(self.accept_all)
        bar.addWidget(self.b_accept)
        self.b_retrain = QPushButton("Retrain + classify")
        self.b_retrain.setToolTip("train a model on all labels so far (Train tab settings), save it, classify these "
                                  "results with it and rebuild the queue")
        self.b_retrain.clicked.connect(app.retrain_and_classify)
        bar.addWidget(self.b_retrain)
        self.src = QLabel("")
        self.src.setProperty("role", "hint")
        bar.addWidget(self.src)
        bar.addStretch(1)
        self.info = QLabel("")
        self.info.setProperty("role", "hint")
        bar.addWidget(self.info)
        outer.addLayout(bar)
        self.keys = QLabel("")
        self.keys.setProperty("role", "hint")
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
        self.state = {}                    # tile index -> dict(label / verdict / colour / skipped)
        self.cur = 0
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
        return "cell" if self.mode.currentText().startswith("Cells") else "particle"

    @property
    def test(self):
        return "test" in self.mode.currentText()

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
                rows = tiers.particle_page(app.res, app.store, app.model, n=self.n,
                                           mode="test" if self.test else "train", done=self.done, m=self._m)
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
        if self.mode.currentText() != MODES[0]:
            self.mode.blockSignals(True)
            self.mode.setCurrentIndex(0)
            self.mode.blockSignals(False)
        self.rows = T.head(60).assign(blind=False).reset_index(drop=True)
        self.state, self.cur = {}, 0
        self._build()
        self.update_info()
        self.app.browser.tabs.setCurrentWidget(self)

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
        cols = max(1, (self.width() - 30) // (TILE + 14))
        for i, r in enumerate(self.rows.itertuples(index=False)):
            t = Tile(self, i)
            self.tiles.append(t)
            self.grid.addWidget(t, i // cols, i % cols)
            self.futs[i] = self.pool.submit(render, self.wells, r, self.kind, self.app.cfg)
        self.redraw_captions()
        self.keys.setText(
            "Keys (click the grid first): ←→↑↓ move · Enter accept suggestion · Space open in canvas · Backspace "
            "skip · " + ("1 one cell · 2 several cells (cannot be split by eye) · 3 not a cell · " +
                         " · ".join(f"{k} {self.app.cfg.display_name(c)}" for k, c in zip("QWER", self._chans())) +
                         " · 0 colour none" if self.kind == "cell" else
                         " · ".join(f"{CLASS_KEYS[c]} {c}" for c in CLASSES)))
        if self.tiles:
            self.focus(0)
        self.timer.start(80)
        self.grid_host.setFocus()

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
        i = max(0, min(i, len(self.tiles) - 1))
        self.tiles[self.cur].set_focus(False) if self.cur < len(self.tiles) else None
        self.cur = i
        self.tiles[i].set_focus(True)

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

    def toggle_colour(self, k):
        chans = self._chans()
        if k >= len(chans):
            return
        st = self.state.get(self.cur, {})
        cur = [c for c in (st.get("colour") or "").split("+") if c and c != "none"]
        ch = chans[k]
        cur = [c for c in cur if c != ch] if ch in cur else cur + [ch]
        self.set_cell(colour="+".join(c for c in chans if c in cur) or "none")

    def accept(self, i=None):
        i = self.cur if i is None else i
        r = self._row(i)
        if self._is_blind(r):
            self.info.setText("blind: no suggestion to accept, choose the class")
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

    def skip(self):
        r = self._row()
        self.state[self.cur] = {"skipped": True}
        self.done.add(self._key(r))
        self._draw_state(self.cur)

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

    def _after(self):
        self._n_since = getattr(self, "_n_since", 0) + 1
        if self._n_since >= 10:                          # the full progress (all labels matched) now and then
            self._n_since = 0
            self.update_info()
        else:
            self.info.setText(f"page {len(self.state)}/{len(self.tiles)}" + self.info.text()[self.info.text().find(" ·"):]
                              if " ·" in self.info.text() else f"page {len(self.state)}/{len(self.tiles)}")
        if self.tiles and len(self.state) >= len(self.tiles):
            QTimer.singleShot(250, self.new_page)
        else:
            self.focus(self.cur + 1)

    def key(self, e):
        k = e.key()
        cols = max(1, (self.width() - 30) // (TILE + 14))
        if not self.tiles:
            if k == Qt.Key_N:
                self.new_page()
            return
        if k == Qt.Key_Right:
            self.focus(self.cur + 1)
        elif k == Qt.Key_Left:
            self.focus(self.cur - 1)
        elif k == Qt.Key_Down:
            self.focus(self.cur + cols)
        elif k == Qt.Key_Up:
            self.focus(self.cur - cols)
        elif k in (Qt.Key_Return, Qt.Key_Enter):
            self.accept()
            self._after()
        elif k == Qt.Key_Space:
            self.open_focused()
        elif k == Qt.Key_Backspace:
            self.skip()
            self._after()
        elif k == Qt.Key_N:
            self.new_page()
        elif self.kind == "cell" and k in VERDICT_KEYS:
            self.set_cell(verdict=VERDICT_KEYS[k])
            st = self.state.get(self.cur, {})
            if st.get("verdict") and (st.get("colour") or st.get("verdict") == "not a cell"):
                self._after()
        elif self.kind == "cell" and k in CH_KEYS:
            self.toggle_colour(CH_KEYS.index(k))
        elif self.kind == "cell" and k == Qt.Key_0:
            self.set_cell(colour="none")
        elif self.kind == "particle" and Qt.Key_1 <= k <= Qt.Key_5:
            cls = {CLASS_KEYS[c]: c for c in CLASSES}.get(chr(k))
            if cls:
                self.set_class(cls)
                self._after()
        else:
            return
        e.accept()

    def update_info(self):
        app = self.app
        n_done = len(self.state)
        try:
            from ..tiers import label_progress
            m = getattr(self, "_m", None)
            self._m = None                                 # one shared match per page, fresh afterwards
            pr = label_progress(app.res, app.store, m=m)
            t = pr["test"]
            self.info.setText(f"page {n_done}/{len(self.tiles)} · test wells: {t['labelled']}/{t['objects']} objects "
                              f"labelled in {t['wells']} wells")
            app.panel.train.show_progress(pr)
        except Exception as e:
            self.info.setText(f"page {n_done}/{len(self.tiles)} ({e})")

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self.tiles:
            cols = max(1, (self.width() - 30) // (TILE + 14))
            for i, t in enumerate(self.tiles):
                self.grid.addWidget(t, i // cols, i % cols)
