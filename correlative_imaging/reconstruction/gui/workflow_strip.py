"""The workflow strip on top of the right panel: Detect → Label → Train → Classify → Results, each with a pictogram,
its state (read from disk by workflow.workflow_state) and, for the picked step, what it does, why it is blocked or out
of date, what it writes, and its one main action. The step to do next is marked; clicking a step shows its card and
opens its tab."""
from __future__ import annotations

import html

from qtpy.QtCore import QPointF, QRectF, QSize, Qt, Signal
from qtpy.QtGui import QColor, QPainter, QPainterPath, QPen
from qtpy.QtWidgets import (QFrame, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget)

from ..workflow import STEPS, next_step

STATE_COLOR = {"done": "#4caf6e", "needed": "#4a90d9", "stale": "#e0a526", "incomplete": "#e0a526",
               "failed": "#e05252", "blocked": "#6b737c", "running": "#4a90d9", "unknown": "#8a929c"}
STATE_TEXT = {"done": "done", "needed": "to do", "stale": "out of date", "incomplete": "unfinished",
              "failed": "failed", "blocked": "not yet", "running": "running…", "unknown": "done (old)"}
ACTION = {"detect": "Run Detect", "label": "Open the labeller", "train": "Train + classify",
          "classify": "Run Classify", "results": "Show results"}
ICON = 30


def _paint_icon(p: QPainter, key: str, r: QRectF, fg: QColor, acc: QColor):
    """Simple pictograms: what the step does to a hole."""
    c = r.center()
    s = r.width()
    pen = QPen(fg, 1.6)
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)
    if key == "detect":                                  # the hole with particles in and around it
        p.drawEllipse(c, s * 0.30, s * 0.30)
        p.setPen(Qt.NoPen)
        p.setBrush(acc)
        for dx, dy, k in ((-0.08, -0.05, 0.09), (0.10, 0.08, 0.07), (0.38, -0.30, 0.06), (-0.36, 0.34, 0.06)):
            p.drawEllipse(QPointF(c.x() + dx * s, c.y() + dy * s), k * s, k * s)
    elif key == "label":                                 # a tag
        path = QPainterPath()
        x0, y0 = r.left() + s * 0.14, r.top() + s * 0.26
        path.moveTo(x0, y0)
        path.lineTo(x0 + s * 0.48, y0)
        path.lineTo(x0 + s * 0.72, y0 + s * 0.24)
        path.lineTo(x0 + s * 0.48, y0 + s * 0.48)
        path.lineTo(x0, y0 + s * 0.48)
        path.closeSubpath()
        p.drawPath(path)
        p.setBrush(acc)
        p.setPen(Qt.NoPen)
        p.drawEllipse(QPointF(x0 + s * 0.14, y0 + s * 0.24), s * 0.06, s * 0.06)
    elif key == "train":                                 # a tiny network: inputs -> class
        a = [QPointF(r.left() + s * 0.2, r.top() + s * (0.25 + 0.25 * i)) for i in range(3)]
        b = [QPointF(r.left() + s * 0.8, r.top() + s * (0.37 + 0.26 * i)) for i in range(2)]
        p.setPen(QPen(fg, 1.0))
        for u in a:
            for w in b:
                p.drawLine(u, w)
        p.setPen(Qt.NoPen)
        p.setBrush(fg)
        for u in a:
            p.drawEllipse(u, s * 0.07, s * 0.07)
        p.setBrush(acc)
        for w in b:
            p.drawEllipse(w, s * 0.08, s * 0.08)
    elif key == "classify":                              # a reconstructed cell: outline grown from a seed
        path = QPainterPath()
        path.moveTo(c.x() - s * 0.30, c.y())
        path.cubicTo(c.x() - s * 0.30, c.y() - s * 0.36, c.x() + s * 0.20, c.y() - s * 0.38, c.x() + s * 0.30,
                     c.y() - s * 0.05)
        path.cubicTo(c.x() + s * 0.40, c.y() + s * 0.30, c.x() - s * 0.10, c.y() + s * 0.40, c.x() - s * 0.30, c.y())
        p.setBrush(QColor(acc.red(), acc.green(), acc.blue(), 90))
        p.drawPath(path)
        p.setPen(Qt.NoPen)
        p.setBrush(acc)
        p.drawEllipse(QPointF(c.x() - s * 0.02, c.y() - s * 0.02), s * 0.08, s * 0.08)
    elif key == "results":                               # a plate: wells coloured by their call
        cols = [acc, fg, QColor("#8a929c")]
        p.setPen(Qt.NoPen)
        for i in range(3):
            for j in range(3):
                p.setBrush(cols[(i * 2 + j) % 3])
                p.drawEllipse(QPointF(r.left() + s * (0.22 + 0.28 * j), r.top() + s * (0.22 + 0.28 * i)),
                              s * 0.09, s * 0.09)


class StepTile(QWidget):
    clicked = Signal(str)

    def __init__(self, key: str, title: str, fg: str, acc: str):
        super().__init__()
        self.key, self.title = key, title
        self.state, self.is_next, self.picked = "blocked", False, False
        self.fg, self.acc = QColor(fg), QColor(acc)
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumSize(QSize(64, ICON + 34))
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set(self, state: str, is_next: bool, picked: bool, tip: str):
        self.state, self.is_next, self.picked = state, is_next, picked
        self.setToolTip(tip)
        self.update()

    def mousePressEvent(self, e):
        self.clicked.emit(self.key)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        col = QColor(STATE_COLOR.get(self.state, "#6b737c"))
        frame = QRectF(1.5, 1.5, w - 3, h - 3)
        if self.picked or self.is_next:
            p.setPen(QPen(col if self.is_next else self.fg, 2.0 if self.is_next else 1.0))
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(frame, 5, 5)
        dim = self.state == "blocked"
        fg = QColor(self.fg)
        acc = QColor(self.acc)
        if dim:
            fg.setAlpha(110)
            acc.setAlpha(110)
        _paint_icon(p, self.key, QRectF((w - ICON) / 2, 4, ICON, ICON), fg, acc)
        f = p.font()
        f.setBold(self.is_next)
        p.setFont(f)
        p.setPen(fg)
        p.drawText(QRectF(0, ICON + 4, w, 15), Qt.AlignHCenter, self.title)
        f.setBold(False)
        f.setPointSizeF(max(f.pointSizeF() - 1.5, 7))
        p.setFont(f)
        p.setPen(col)
        p.drawText(QRectF(0, ICON + 18, w, 14), Qt.AlignHCenter, ("● " if self.state != "blocked" else "") +
                   STATE_TEXT.get(self.state, self.state))
        p.end()


class WorkflowStrip(QFrame):
    """The strip and the picked step's card."""

    def __init__(self, app, theme: dict):
        super().__init__()
        self.app = app
        self.setProperty("role", "card")
        v = QVBoxLayout(self)
        v.setContentsMargins(6, 4, 6, 6)
        v.setSpacing(4)
        row = QHBoxLayout()
        row.setSpacing(2)
        self.tiles = {}
        for i, k in enumerate(STEPS):
            if i:
                a = QLabel("→")
                a.setProperty("role", "hint")
                row.addWidget(a)
            t = StepTile(k, k.capitalize(), theme.get("text", "#f0f1f2"), theme.get("current", "#4a90d9"))
            t.clicked.connect(self.pick)
            self.tiles[k] = t
            row.addWidget(t, 1)
        v.addLayout(row)
        loop = QLabel("Label, Train and Classify form a loop: label → train + classify → label the new suggestions")
        loop.setProperty("role", "hint")
        loop.setWordWrap(True)
        loop.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        v.addWidget(loop)
        card = QHBoxLayout()
        self.text = QLabel("")
        self.text.setWordWrap(True)
        self.text.setTextFormat(Qt.RichText)
        self.text.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        card.addWidget(self.text, 1)
        self.btn = QPushButton("")
        self.btn.setProperty("accent", True)
        self.btn.clicked.connect(self._act)
        card.addWidget(self.btn, 0, Qt.AlignTop)
        v.addLayout(card)
        self.S, self.cur, self._user_pick = None, None, False

    # -- state
    def set_state(self, S: dict):
        self.S = S
        nxt = next_step(S)
        if not self._user_pick or self.cur not in S:
            self.cur = nxt or "results"
        for k, t in self.tiles.items():
            s = S[k]
            tip = f"{s.title}: {s.purpose}\n{STATE_TEXT.get(s.state, s.state)}" + (f": {s.reason}" if s.reason else "")
            t.set(s.state, k == nxt, k == self.cur, tip)
        self._card()

    def pick(self, key: str):
        self._user_pick = True
        self.cur = key
        if self.S is not None:
            self.set_state(self.S)
        self.app.show_step(key)

    def follow_next(self):
        """After a job: show the card of the step to do next again."""
        self._user_pick = False

    def _card(self):
        s = self.S[self.cur]
        col = STATE_COLOR.get(s.state, "#6b737c")
        e = html.escape
        parts = [f"<b>{s.title}</b> — {e(s.purpose)}.",
                 f"<span style='color:{col}'>● {STATE_TEXT.get(s.state, s.state)}</span>"
                 + (f" · {e(s.detail)}" if s.detail else "")]
        if s.reason:
            parts.append(f"<i>{e(s.reason)}</i>")
        parts.append(f"<span style='font-size:8pt'>writes: {e(', '.join(s.outputs))}</span>")
        self.text.setText("<br>".join(parts))
        self.btn.setText(ACTION[self.cur])
        runnable = s.can_run and s.state not in ("running", "blocked")
        self.btn.setEnabled(runnable and not getattr(self.app, "_job_running", False))
        self.btn.setToolTip("" if runnable else (s.reason or "not possible yet"))

    def _act(self):
        self.app.run_step(self.cur)
