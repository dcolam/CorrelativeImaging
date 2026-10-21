"""Channel histogram: every channel's histogram overlaid on one axis, with the channel buttons as its legend and the
contrast handles inside the plot.

  buttons    one per channel, in its colour. Click the name: adjust that channel. Click the dot: show / hide the
             channel (the napari layer's visibility, kept in sync both ways). Hidden channels are not drawn.
  plot       log counts per channel, each scaled to its own peak (so a dim channel stays readable next to a bright
             one); the picked channel filled, the others outlined. Intensity axis: asinh (default, readable across
             channels whose ranges differ by orders of magnitude, copes with negative values) or linear.
  handles    the picked channel's min and max as vertical lines: drag one; drag the shaded band between them to move
             both; double-click for auto. The thin curve is the display mapping (with gamma).
  bottom     the range in numbers, gamma, auto, asinh / linear.

Works on any napari image layers (the caller says which): only a strided sample of at most SAMPLE pixels is read,
so 4D stacks and dask arrays stay cheap. `auto_rule(sample) -> (lo, hi)` is the caller's auto-contrast rule.
"""
from __future__ import annotations

import numpy as np
from qtpy.QtCore import QPointF, QRectF, QSize, Qt, QTimer, Signal
from qtpy.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from qtpy.QtWidgets import (QHBoxLayout, QLabel, QLayout, QPushButton, QSizePolicy, QSlider, QToolButton,
                            QVBoxLayout, QWidget)

SAMPLE = 1_000_000
BINS = 220
PLOT_H = 120
GRIP = 7                     # px: how close to a handle a press grabs it


# --------------------------------------------------------------------------------------------------------------
# helpers

def layer_color(layer) -> str:
    """The brightest colour of an image layer's colormap, as hex."""
    try:
        c = np.asarray(layer.colormap.colors)[-1][:3]
        return "#" + "".join(f"{int(round(float(v) * 255)):02x}" for v in np.clip(c, 0, 1))
    except Exception:
        return "#cccccc"


def is_channel_layer(layer) -> bool:
    """A plain intensity image (not RGB, not a mask / label image)."""
    from napari.layers import Image
    if not isinstance(layer, Image) or getattr(layer, "rgb", False):
        return False
    if (getattr(layer, "metadata", None) or {}).get("mask"):       # a 0/1 mask shown as an image
        return False
    try:
        d = layer.data[-1] if layer.multiscale else layer.data
        return d.dtype != bool
    except Exception:
        return False


def sample(layer, n: int = SAMPLE) -> np.ndarray:
    """At most ~n finite pixel values of a layer (strided; the lowest resolution of a multiscale layer)."""
    d = layer.data[-1] if getattr(layer, "multiscale", False) else layer.data
    shape = tuple(int(s) for s in d.shape)
    total = int(np.prod(shape)) if shape else 0
    if total == 0:
        return np.zeros(0)
    steps = [1] * len(shape)
    if total > n:                                            # stride the outer axes first, then the image axes
        f = total / n
        for ax in range(len(shape)):
            if f <= 1:
                break
            s = int(min(shape[ax], np.ceil(f ** (1 / (len(shape) - ax)))))
            steps[ax] = max(1, s)
            f /= steps[ax]
    a = np.asarray(d[tuple(slice(None, None, s) for s in steps)], dtype=float).ravel()
    return a[np.isfinite(a)]


class _Axis:
    """Intensity <-> 0..1 along the plot, linear or asinh."""

    def __init__(self, lo: float, hi: float, log: bool):
        self.log = log
        self.s = max((hi - lo) / 2000.0, 1e-6)              # asinh: linear within ~s of zero, logarithmic beyond
        self.a, self.b = self._f(lo), self._f(hi)
        if self.b <= self.a:
            self.b = self.a + 1

    def _f(self, v):
        return np.arcsinh(np.asarray(v, float) / self.s) if self.log else np.asarray(v, float)

    def to01(self, v):
        return (self._f(v) - self.a) / (self.b - self.a)

    def from01(self, t):
        u = self.a + np.asarray(t, float) * (self.b - self.a)
        return np.sinh(u) * self.s if self.log else u


# --------------------------------------------------------------------------------------------------------------
# legend buttons

class _FlowLayout(QLayout):
    """Buttons left to right, wrapping to a new row when the panel is narrow."""

    def __init__(self, parent=None, spacing=4):
        super().__init__(parent)
        self._items, self._sp = [], spacing

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, i):
        return self._items[i] if 0 <= i < len(self._items) else None

    def takeAt(self, i):
        return self._items.pop(i) if 0 <= i < len(self._items) else None

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, w):
        return self._do(QRectF(0, 0, w, 0), True)

    def setGeometry(self, r):
        super().setGeometry(r)
        self._do(QRectF(r), False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        s = QSize(0, 0)
        for it in self._items:
            s = s.expandedTo(it.minimumSize())
        return s

    def _do(self, r, test):
        x, y, row_h = r.x(), r.y(), 0
        for it in self._items:
            sh = it.sizeHint()
            if x + sh.width() > r.right() and row_h > 0:
                x, y, row_h = r.x(), y + row_h + self._sp, 0
            if not test:
                from qtpy.QtCore import QRect
                it.setGeometry(QRect(int(x), int(y), sh.width(), sh.height()))
            x += sh.width() + self._sp
            row_h = max(row_h, sh.height())
        return int(y + row_h - r.y())


class ChannelButton(QWidget):
    """A pill in the channel's colour: dot = show / hide, name = pick for adjusting."""
    picked = Signal(object)
    toggled = Signal(object)

    def __init__(self, key, text: str, color: str):
        super().__init__()
        self.key, self.text, self.color = key, text, QColor(color)
        self.active, self.shown = False, True
        self.setCursor(Qt.PointingHandCursor)
        self._fm = self.fontMetrics()
        self.setFixedHeight(22)
        self.setToolTip(f"{text}: click to adjust its contrast · click the dot to show / hide it")
        self._w = self._fm.horizontalAdvance(text) + 34
        self.setFixedWidth(self._w)

    def sizeHint(self):
        return QSize(self._w, 22)

    def set(self, text=None, color=None, active=None, shown=None):
        if text is not None and text != self.text:
            self.text = text
            self._w = self._fm.horizontalAdvance(text) + 34
            self.setFixedWidth(self._w)
        if color is not None:
            self.color = QColor(color)
        if active is not None:
            self.active = active
        if shown is not None:
            self.shown = shown
        self.update()

    def mousePressEvent(self, e):
        if e.button() != Qt.LeftButton:
            return
        if e.pos().x() < 22:
            self.toggled.emit(self.key)
        else:
            self.picked.emit(self.key)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(0.5, 0.5, self.width() - 1, self.height() - 1)
        c = QColor(self.color)
        if self.active:
            fill = QColor(c)
            fill.setAlpha(70)
            p.setBrush(fill)
            p.setPen(QPen(c, 1.6))
        else:
            p.setBrush(Qt.NoBrush)
            q = QColor(c)
            q.setAlpha(110 if self.shown else 60)
            p.setPen(QPen(q, 1.0))
        p.drawRoundedRect(r, 11, 11)
        dot = QPointF(11.5, self.height() / 2)               # the visibility dot
        if self.shown:
            p.setPen(Qt.NoPen)
            p.setBrush(c)
            p.drawEllipse(dot, 5, 5)
        else:
            q = QColor(c)
            q.setAlpha(150)
            p.setPen(QPen(q, 1.4))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(dot, 4.5, 4.5)
        t = self.palette().windowText().color()
        if not self.shown:
            t.setAlpha(110)
        p.setPen(t)
        f = p.font()
        f.setBold(self.active)
        p.setFont(f)
        p.drawText(QRectF(22, 0, self.width() - 26, self.height()), Qt.AlignVCenter | Qt.AlignLeft, self.text)
        if not self.shown:                                  # struck through: hidden
            y = self.height() / 2
            p.drawLine(QPointF(22, y), QPointF(22 + self._fm.horizontalAdvance(self.text), y))
        p.end()


# --------------------------------------------------------------------------------------------------------------
# the plot

class HistPlot(QWidget):
    limits_dragged = Signal(float, float)                   # new (lo, hi) of the picked channel while dragging
    auto_requested = Signal()

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(PLOT_H)
        self.setMaximumHeight(PLOT_H)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)     # never sized by its content
        self.setMouseTracking(True)
        self.curves = {}          # key -> (counts 0..1 over BINS, colour)
        self.active = None
        self.lim = None           # (lo, hi) of the picked channel
        self.gamma = 1.0
        self.axis = _Axis(0, 1, True)
        self._drag = None

    # geometry: a margin for the value labels at the bottom
    def _plot_rect(self) -> QRectF:
        return QRectF(6, 6, max(self.width() - 12, 10), self.height() - 22)

    def _x(self, v) -> float:
        r = self._plot_rect()
        return r.left() + float(np.clip(self.axis.to01(v), -0.02, 1.02)) * r.width()

    def _v(self, x) -> float:
        r = self._plot_rect()
        return float(self.axis.from01((x - r.left()) / r.width()))

    def set_data(self, curves: dict, axis: _Axis, active, lim, gamma):
        self.curves, self.axis, self.active, self.lim, self.gamma = curves, axis, active, lim, gamma
        self.update()

    def set_limits(self, lim, gamma=None):
        self.lim = lim
        if gamma is not None:
            self.gamma = gamma
        self.update()

    # -- mouse
    def _hit(self, x):
        if self.lim is None:
            return None
        x0, x1 = self._x(self.lim[0]), self._x(self.lim[1])
        if abs(x - x0) <= GRIP:
            return "lo"
        if abs(x - x1) <= GRIP:
            return "hi"
        if x0 < x < x1:
            return "band"
        return None

    def mousePressEvent(self, e):
        if e.button() != Qt.LeftButton or self.lim is None:
            return
        x = e.pos().x()
        h = self._hit(x)
        if h is None:                                       # a press outside the band: the nearer handle jumps there
            h = "lo" if abs(x - self._x(self.lim[0])) < abs(x - self._x(self.lim[1])) else "hi"
            self._move(h, x)
        self._drag = (h, x, self._x(self.lim[0]), self._x(self.lim[1]))

    def mouseMoveEvent(self, e):
        x = e.pos().x()
        if self._drag is None:
            h = self._hit(x)
            self.setCursor(Qt.SizeHorCursor if h in ("lo", "hi") else Qt.OpenHandCursor if h == "band"
                           else Qt.ArrowCursor)
            return
        h, x_start, x0, x1 = self._drag
        if h == "band":
            dx = x - x_start
            self._emit(self._v(x0 + dx), self._v(x1 + dx))
        else:
            self._move(h, x)

    def _move(self, h, x):
        lo, hi = self.lim
        if h == "lo":
            lo = min(self._v(x), hi - 1e-9)
        else:
            hi = max(self._v(x), lo + 1e-9)
        self._emit(lo, hi)

    def _emit(self, lo, hi):
        if hi <= lo:
            return
        self.lim = (lo, hi)
        self.update()
        self.limits_dragged.emit(lo, hi)

    def mouseReleaseEvent(self, e):
        self._drag = None

    def mouseDoubleClickEvent(self, e):
        self.auto_requested.emit()

    # -- drawing
    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = self._plot_rect()
        pal = self.palette()
        txt = pal.windowText().color()
        grid = QColor(txt)
        grid.setAlpha(28)
        p.setPen(QPen(grid, 1))
        p.drawLine(QPointF(r.left(), r.bottom()), QPointF(r.right(), r.bottom()))
        xs = np.linspace(r.left(), r.right(), BINS)

        def path_of(counts, closed):
            nz = np.flatnonzero(counts)                      # from the first to the last non-empty bin
            i0, i1 = (int(nz[0]), int(nz[-1])) if nz.size else (0, 0)
            path = QPainterPath()
            path.moveTo(float(xs[i0]), r.bottom())
            for x, c in zip(xs[i0:i1 + 1], counts[i0:i1 + 1]):
                path.lineTo(float(x), r.bottom() - float(c) * (r.height() - 2))
            path.lineTo(float(xs[i1]), r.bottom())
            if closed:
                path.closeSubpath()
            return path

        # band of the picked channel
        if self.lim is not None and self.active in self.curves:
            col = QColor(self.curves[self.active][1])
            band = QColor(col)
            band.setAlpha(34)
            x0, x1 = self._x(self.lim[0]), self._x(self.lim[1])
            p.fillRect(QRectF(x0, r.top(), max(x1 - x0, 1), r.height()), band)
        # the other channels outlined, the picked one filled on top
        for k, (counts, color) in self.curves.items():
            if k == self.active:
                continue
            c = QColor(color)
            c.setAlpha(150)
            p.setPen(QPen(c, 1.2))
            p.setBrush(Qt.NoBrush)
            p.drawPath(path_of(counts, False))
        if self.active in self.curves:
            counts, color = self.curves[self.active]
            c = QColor(color)
            fill = QColor(c)
            fill.setAlpha(90)
            p.setBrush(fill)
            p.setPen(QPen(c, 1.6))
            p.drawPath(path_of(counts, True))
        # display mapping and handles
        if self.lim is not None and self.active in self.curves:
            lo, hi = self.lim
            x0, x1 = self._x(lo), self._x(hi)
            curve = QPainterPath()
            curve.moveTo(x0, r.bottom())
            for t in np.linspace(0, 1, 40):
                x = self._x(lo + t * (hi - lo))
                curve.lineTo(x, r.bottom() - (t ** self.gamma) * (r.height() - 2))
            m = QColor(txt)
            m.setAlpha(120)
            p.setPen(QPen(m, 1, Qt.DashLine))
            p.setBrush(Qt.NoBrush)
            p.drawPath(curve)
            hc = QColor(self.curves[self.active][1])
            for x in (x0, x1):
                p.setPen(QPen(txt, 1.4))
                p.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))
                grip = QPainterPath()                        # a small grip on top of each handle
                grip.addRoundedRect(QRectF(x - 4, r.top() - 3, 8, 12), 2, 2)
                p.setBrush(hc)
                p.setPen(QPen(txt, 1))
                p.drawPath(grip)
            f = QFont(p.font())
            f.setPointSizeF(max(f.pointSizeF() - 1.5, 7))
            p.setFont(f)
            p.setPen(txt)
            fm = p.fontMetrics()
            for x, v, left in ((x0, lo, True), (x1, hi, False)):
                s = _fmt(v)
                w = fm.horizontalAdvance(s)
                tx = x - w - 3 if left else x + 3
                tx = min(max(tx, 0), self.width() - w)
                p.drawText(QPointF(tx, self.height() - 4), s)
        elif not self.curves:
            p.setPen(txt)
            p.drawText(r, Qt.AlignCenter, "no channel shown")
        p.end()


def _grey(hex_color: str) -> bool:
    c = QColor(hex_color)
    return max(c.red(), c.green(), c.blue()) - min(c.red(), c.green(), c.blue()) < 20


def _fmt(v: float) -> str:
    a = abs(v)
    return f"{v:.0f}" if a >= 100 else f"{v:.1f}" if a >= 1 else f"{v:.3g}"


# --------------------------------------------------------------------------------------------------------------
# the widget

class ChannelHistogram(QWidget):
    """Legend buttons + overlaid histograms + handles for a set of napari image layers.

    layers_fn() -> [(key, layer)] in display order (default: every channel-like image layer of the viewer);
    name_fn(key, layer) / color_fn(key, layer): text and colour of a channel (default: layer name / colormap);
    auto_rule(sample) -> (lo, hi)."""

    def __init__(self, viewer, layers_fn=None, name_fn=None, color_fn=None, auto_rule=None):
        super().__init__()
        self.v = viewer
        self.layers_fn = layers_fn or (lambda: [(L.name, L) for L in viewer.layers if is_channel_layer(L)])
        self.name_fn = name_fn or (lambda k, L: L.name)
        self.color_fn = color_fn or (lambda k, L: layer_color(L))
        self.auto_rule = auto_rule or (lambda a: (float(np.nanpercentile(a, 1)), float(np.nanpercentile(a, 99.5))))
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(3)
        self.btn_box = QWidget()
        self.flow = _FlowLayout(self.btn_box)
        self.btn_box.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        v.addWidget(self.btn_box)
        self.plot = HistPlot()
        self.plot.limits_dragged.connect(self._dragged)
        self.plot.auto_requested.connect(self.auto)
        v.addWidget(self.plot)
        h = QHBoxLayout()
        h.setSpacing(6)
        self.rng = QLabel("")
        self.rng.setProperty("role", "hint")
        self.rng.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        h.addWidget(self.rng, 1)
        h.addWidget(QLabel("γ"))
        self.gam = QSlider(Qt.Horizontal)
        self.gam.setRange(20, 300)
        self.gam.setFixedWidth(80)
        self.gam.setToolTip("gamma of the picked channel (double-click the plot: auto)")
        self.gam.valueChanged.connect(self._gamma)
        h.addWidget(self.gam)
        self.b_auto = QPushButton("auto")
        self.b_auto.setToolTip("auto-contrast the picked channel (or double-click the plot)")
        self.b_auto.clicked.connect(self.auto)
        h.addWidget(self.b_auto)
        self.b_log = QToolButton()
        self.b_log.setText("log")
        self.b_log.setCheckable(True)
        self.b_log.setChecked(True)
        self.b_log.setToolTip("intensity axis: log-like (asinh: every channel readable, copes with negative values) "
                              "or linear")
        self.b_log.toggled.connect(lambda on: (self.b_log.setText("log" if on else "lin"), self._redraw()))
        h.addWidget(self.b_log)
        v.addLayout(h)
        self.entries = []         # [(key, layer)]
        self.buttons = {}
        self.cur = None
        self._samples = {}        # key -> (id(data), sample)
        self._hooks = []          # (emitter, callback) to disconnect
        self._lock = False
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.rebuild)
        try:
            for ev in ("inserted", "removed", "moved", "reordered"):
                em = getattr(viewer.layers.events, ev, None)
                if em is not None:
                    em.connect(self._later)
        except Exception:
            pass
        self.rebuild()

    # -- layer bookkeeping
    def _later(self, *_):
        self._timer.start(0)

    def _unhook(self):
        for em, cb in self._hooks:
            try:
                em.disconnect(cb)
            except Exception:
                pass
        self._hooks = []

    def _hook(self, em, cb):
        em.connect(cb)
        self._hooks.append((em, cb))

    def rebuild(self):
        """Re-read which layers are channels (after layers were added / removed)."""
        self._unhook()
        try:
            entries = [(k, L) for k, L in self.layers_fn() if L is not None]
        except Exception:
            entries = []
        live = {id(L) for _, L in entries}
        self._samples = {k: s for k, s in self._samples.items() if s[2] in live}
        self.entries = entries
        while self.flow.count():
            it = self.flow.takeAt(0)
            if it.widget():
                it.widget().setParent(None)
        self.buttons = {}
        for k, L in entries:
            b = ChannelButton(k, self.name_fn(k, L), self.color_fn(k, L))
            b.picked.connect(self.pick)
            b.toggled.connect(self._toggle)
            self.flow.addWidget(b)
            self.buttons[k] = b
            e = L.events
            self._hook(e.contrast_limits, lambda ev, k=k: self._limits_changed(k))
            self._hook(e.gamma, lambda ev, k=k: self._limits_changed(k))
            self._hook(e.visible, lambda ev, k=k: self._vis_changed(k))
            self._hook(e.data, lambda ev, k=k: self._data_changed(k))
            self._hook(e.colormap, lambda ev, k=k: self._style_changed(k))
            self._hook(e.name, lambda ev, k=k: self._style_changed(k))
        keys = [k for k, _ in entries]
        if self.cur not in keys:                     # start on a shown, coloured channel (not brightfield)
            shown = [k for k, L in entries if L.visible]
            coloured = [k for k in shown if not _grey(self.color_fn(k, self._layer_of(entries, k)))]
            self.cur = (coloured or shown or keys or [None])[0]
        self.btn_box.setMinimumHeight(self.flow.heightForWidth(max(self.btn_box.width(), 200)))
        self._redraw()

    @staticmethod
    def _layer_of(entries, key):
        return next((L for k, L in entries if k == key), None)

    def _layer(self, key=None):
        key = self.cur if key is None else key
        for k, L in self.entries:
            if k == key:
                return L
        return None

    def _sample(self, k, L):
        got = self._samples.get(k)
        if got is None or got[0] != id(L.data):
            try:
                a = sample(L)
            except Exception:
                a = np.zeros(0)
            got = (id(L.data), a, id(L))
            self._samples[k] = got
        return got[1]

    # -- events from the layers
    def _limits_changed(self, k):
        if self._lock or k != self.cur:
            return
        L = self._layer()
        if L is not None:
            lo, hi = L.contrast_limits
            if not (self._ax_lo <= lo and hi <= self._ax_hi):          # outside the axis: widen it
                self._redraw()
            else:
                self.plot.set_limits((lo, hi), L.gamma)
                self._range_text(L)

    def _vis_changed(self, k):
        self._redraw()

    def _data_changed(self, k):
        self._samples.pop(k, None)
        self._later_redraw()

    def _later_redraw(self):
        QTimer.singleShot(0, self._redraw)

    def _style_changed(self, k):
        L = self._layer(k)
        if L is not None and k in self.buttons:
            self.buttons[k].set(text=self.name_fn(k, L), color=self.color_fn(k, L))
        self._redraw()

    # -- user actions
    def pick(self, key):
        self.cur = key
        self._redraw()

    def _toggle(self, key):
        L = self._layer(key)
        if L is not None:
            L.visible = not L.visible                         # napari's Layers list follows (and vice versa)

    def _dragged(self, lo, hi):
        L = self._layer()
        if L is None:
            return
        self._lock = True
        try:
            set_limits(L, lo, hi)
        finally:
            self._lock = False
        self._range_text(L)

    def _gamma(self, v):
        L = self._layer()
        if L is not None and not self._lock:
            L.gamma = v / 100

    def auto(self):
        L = self._layer()
        if L is None:
            return
        a = self._sample(self.cur, L)
        if a.size:
            lo, hi = self.auto_rule(a)
            set_limits(L, lo, hi)
            L.gamma = 1.0
            self._redraw()

    # -- drawing
    def _range_text(self, L):
        lo, hi = L.contrast_limits
        self.rng.setText(f"{_fmt(lo)} – {_fmt(hi)}")
        self.rng.setToolTip(f"{self.name_fn(self.cur, L)}: {lo:g} – {hi:g}")

    def _redraw(self):
        shown = [(k, L) for k, L in self.entries if L.visible]
        for k, L in self.entries:
            if k in self.buttons:
                self.buttons[k].set(active=(k == self.cur), shown=bool(L.visible))
        S = {k: self._sample(k, L) for k, L in shown}
        cur_L = self._layer()
        lims = [tuple(cur_L.contrast_limits)] if cur_L is not None else []
        vals = [a for a in S.values() if a.size]
        if self.cur not in S and cur_L is not None:                  # picked but hidden: its axis still counts
            vals.append(self._sample(self.cur, cur_L))
        if vals or lims:
            lo = min([float(a.min()) for a in vals if a.size] + [l[0] for l in lims])
            hi = max([float(a.max()) for a in vals if a.size] + [l[1] for l in lims])
        else:
            lo, hi = 0.0, 1.0
        if hi <= lo:
            hi = lo + 1
        self._ax_lo, self._ax_hi = lo, hi
        ax = _Axis(lo, hi, self.b_log.isChecked())
        curves = {}
        for k, a in S.items():
            if not a.size:
                continue
            t = np.clip(ax.to01(a), 0, 1)
            c = np.bincount(np.minimum((t * (BINS - 1)).astype(int), BINS - 1), minlength=BINS).astype(float)
            c = np.log1p(c)
            c = c / (c.max() or 1)
            L = self._layer(k)
            curves[k] = (c, self.color_fn(k, L))
        if cur_L is not None and self.cur not in curves:            # hidden picked channel: handles, no curve
            curves_draw = curves
            active = None
        else:
            curves_draw, active = curves, self.cur
        self.plot.set_data(curves_draw, ax, active, tuple(cur_L.contrast_limits) if cur_L is not None else None,
                           float(cur_L.gamma) if cur_L is not None else 1.0)
        if cur_L is not None:
            self._lock = True
            self.gam.setValue(int(round(cur_L.gamma * 100)))
            self._lock = False
            self._range_text(cur_L)
        else:
            self.rng.setText("")

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.btn_box.setMinimumHeight(self.flow.heightForWidth(max(self.btn_box.width(), 50)))

    def closeEvent(self, e):
        self._unhook()
        super().closeEvent(e)


def set_limits(L, lo: float, hi: float):
    """Contrast limits, widening the layer's slider range first when they fall outside it (napari would clip)."""
    r0, r1 = L.contrast_limits_range
    if lo < r0 or hi > r1:
        L.contrast_limits_range = (min(lo, r0), max(hi, r1))
    L.contrast_limits = (float(lo), float(hi))
