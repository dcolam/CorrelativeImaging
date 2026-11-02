"""The thumbnail labeller shared by the first pass's questions and the Label tab: pages of items, one answer per item
from buttons or keys, saved at once by the caller.

Keys (click the grid first): arrows move · 1-9 the answer buttons in order · 0 clears · Space opens the item large in
the canvas · PgDn / PgUp next / previous page.
Thumbnails are drawn in one background thread (images come from the run's files) and painted as they arrive.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from qtpy.QtCore import Qt, QTimer, Signal
from qtpy.QtGui import QImage, QPixmap
from qtpy.QtWidgets import (QCheckBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea,
                            QSizePolicy, QVBoxLayout, QWidget)

TILE = 128
COLS, ROWS = 4, 3
ANSWER_COLORS = ["#2bb3a3", "#e0a526", "#4a90d9", "#c45ab3", "#8a929c", "#e05252", "#6fbf4a", "#d98a4a", "#5ac4e0"]


def pixmap(a: np.ndarray | None, size: int = TILE) -> QPixmap:
    if a is None:
        p = QPixmap(size, size)
        p.fill(Qt.black)
        return p
    a = np.ascontiguousarray(a[..., :3].astype(np.uint8))
    h, w, _ = a.shape
    img = QImage(a.data, w, h, 3 * w, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(img).scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)


class Tile(QFrame):
    clicked = Signal(int)
    double = Signal(int)

    def __init__(self, i: int):
        super().__init__()
        self.i = i
        self.setFrameShape(QFrame.Box)
        self.setLineWidth(2)
        self.setFixedSize(TILE + 10, TILE + 52)
        v = QVBoxLayout(self)
        v.setContentsMargins(3, 3, 3, 3)
        v.setSpacing(1)
        self.img = QLabel()
        self.img.setFixedSize(TILE, TILE)
        self.img.setAlignment(Qt.AlignCenter)
        self.cap = QLabel("")
        self.cap.setWordWrap(True)
        self.cap.setFixedWidth(TILE)
        self.cap.setStyleSheet("font-size: 8pt")
        self.ans = QLabel("")
        self.ans.setStyleSheet("font-weight: 600")
        v.addWidget(self.img)
        v.addWidget(self.cap)
        v.addWidget(self.ans)

    def mousePressEvent(self, e):
        self.clicked.emit(self.i)

    def mouseDoubleClickEvent(self, e):
        self.double.emit(self.i)


class Asker(QWidget):
    """set_items(items, choices, on_answer, ...). items: DataFrame with kind, plate_id, well_id, image_id, obj, caption,
    answer. on_answer(row: dict, answer: str) stores it (empty answer = cleared)."""
    answered = Signal()

    def __init__(self, views_fn, open_fn):
        super().__init__()
        self.views_fn, self.open_fn = views_fn, open_fn
        self.items = pd.DataFrame()
        self.choices, self.on_answer, self.page, self.cur = [], None, 0, 0
        self.channels, self.on_channels, self.chan_state = [], None, {}
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.futs, self.pix = {}, {}
        self.setFocusPolicy(Qt.StrongFocus)
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        self.title = QLabel("")
        self.title.setWordWrap(True)
        self.title.setTextFormat(Qt.RichText)
        v.addWidget(self.title)
        self.btn_row = QHBoxLayout()
        v.addLayout(self.btn_row)
        self.chan_row = QHBoxLayout()
        v.addLayout(self.chan_row)
        grid_w = QWidget()
        self.grid = QGridLayout(grid_w)
        self.grid.setSpacing(4)
        self.tiles = []
        for i in range(COLS * ROWS):
            t = Tile(i)
            t.clicked.connect(self.focus)
            t.double.connect(lambda i: self.open_fn(self._row(i)) if self._row(i) else None)
            self.grid.addWidget(t, i // COLS, i % COLS)
            self.tiles.append(t)
        sc = QScrollArea()
        sc.setWidget(grid_w)
        sc.setWidgetResizable(True)
        v.addWidget(sc, 1)
        nav = QHBoxLayout()
        self.prev_b = QPushButton("◀ page")
        self.prev_b.clicked.connect(lambda: self.set_page(self.page - 1))
        self.next_b = QPushButton("page ▶")
        self.next_b.clicked.connect(lambda: self.set_page(self.page + 1))
        self.info = QLabel("")
        nav.addWidget(self.prev_b)
        nav.addWidget(self.info, 1)
        nav.addWidget(self.next_b)
        v.addLayout(nav)
        hint = QLabel("Click a thumbnail, then: 1-9 answer · 0 clear · arrows move · Space / double-click: open in the "
                      "canvas · PgDn/PgUp page")
        hint.setProperty("role", "hint")
        hint.setWordWrap(True)
        v.addWidget(hint)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._poll)
        self.timer.start(120)

    # ------------------------------------------------------------------------------------------------- setup
    def set_items(self, items: pd.DataFrame, choices: list, on_answer, title: str = "", channels=None,
                  on_channels=None, chan_state=None):
        self.items = items.reset_index(drop=True)
        if "answer" not in self.items:
            self.items["answer"] = ""
        self.choices, self.on_answer = list(choices), on_answer
        self.channels, self.on_channels = list(channels or []), on_channels
        self.chan_state = chan_state or {}
        self.title.setText(title)
        self._buttons()
        self.page, self.cur = 0, 0
        self.futs, self.pix = {}, {}
        first = (self.items.answer == "").to_numpy().nonzero()[0]
        self.set_page(int(first[0]) // (COLS * ROWS) if len(first) else 0)

    def _clear_layout(self, lay):
        while lay.count():
            w = lay.takeAt(0).widget()
            if w is not None:
                w.deleteLater()

    def _buttons(self):
        self._clear_layout(self.btn_row)
        for k, c in enumerate(self.choices):
            b = QPushButton(f"{k + 1}  {c}" if k < 9 else c)
            b.setStyleSheet(f"border-left: 4px solid {ANSWER_COLORS[k % len(ANSWER_COLORS)]}")
            b.clicked.connect(lambda _=False, c=c: self.answer(c))
            self.btn_row.addWidget(b)
        self.btn_row.addStretch(1)
        self._clear_layout(self.chan_row)
        self.chan_boxes = {}
        if self.channels:
            lab = QLabel("positive channels:")
            self.chan_row.addWidget(lab)
            for c, name in self.channels:
                cb = QCheckBox(name)
                cb.toggled.connect(lambda on, c=c: self._chan_toggled(c, on))
                self.chan_boxes[c] = cb
                self.chan_row.addWidget(cb)
            self.chan_row.addStretch(1)

    # ------------------------------------------------------------------------------------------------- paging
    def _row(self, i=None) -> dict | None:
        k = self.page * COLS * ROWS + (self.cur if i is None else i)
        return self.items.iloc[k].to_dict() if 0 <= k < len(self.items) else None

    def set_page(self, p: int):
        n_pages = max(1, int(np.ceil(len(self.items) / (COLS * ROWS))))
        self.page = int(np.clip(p, 0, n_pages - 1))
        self.prev_b.setEnabled(self.page > 0)
        self.next_b.setEnabled(self.page < n_pages - 1)
        V = self.views_fn()
        for i, t in enumerate(self.tiles):
            k = self.page * COLS * ROWS + i
            if k < len(self.items):
                t.show()
                r = self.items.iloc[k].to_dict()
                t.cap.setText(str(r.get("caption", "")))
                if k in self.pix:
                    t.img.setPixmap(self.pix[k])
                else:
                    t.img.setPixmap(pixmap(None))
                    if k not in self.futs and V is not None:
                        self.futs[k] = self.pool.submit(V.render, r)
            else:
                t.hide()
        # the next page ahead
        for k in range((self.page + 1) * COLS * ROWS, min((self.page + 2) * COLS * ROWS, len(self.items))):
            if k not in self.futs and k not in self.pix and V is not None:
                self.futs[k] = self.pool.submit(V.render, self.items.iloc[k].to_dict())
        self.focus(min(self.cur, max(0, min(COLS * ROWS, len(self.items) - self.page * COLS * ROWS) - 1)))
        self._paint()

    def _poll(self):
        done = [k for k, f in self.futs.items() if f.done()]
        for k in done:
            f = self.futs.pop(k)
            try:
                self.pix[k] = pixmap(f.result())
            except Exception:                                                # noqa: BLE001
                self.pix[k] = pixmap(None)
            i = k - self.page * COLS * ROWS
            if 0 <= i < len(self.tiles):
                self.tiles[i].img.setPixmap(self.pix[k])

    def _paint(self):
        n = len(self.items)
        done = int((self.items.answer != "").sum()) if n else 0
        self.info.setText(f"page {self.page + 1} of {max(1, int(np.ceil(n / (COLS * ROWS))))} · answered {done} of {n}")
        for i, t in enumerate(self.tiles):
            k = self.page * COLS * ROWS + i
            if k >= n:
                continue
            a = str(self.items.answer.iloc[k] or "")
            col = ANSWER_COLORS[self.choices.index(a) % len(ANSWER_COLORS)] if a in self.choices else "#8a929c"
            t.ans.setText(a)
            t.ans.setStyleSheet(f"font-weight: 600; color: {col}")
            t.setStyleSheet(f"Tile {{ border: 2px solid {'#ffffff' if i == self.cur else (col if a else '#3a3f46')}; }}")

    def focus(self, i: int):
        self.cur = i
        r = self._row()
        if r is not None and self.chan_boxes:
            st = self.chan_state.get(self._key(r), set())
            for c, cb in self.chan_boxes.items():
                cb.blockSignals(True)
                cb.setChecked(c in st)
                cb.blockSignals(False)
        self._paint()
        self.setFocus()

    @staticmethod
    def _key(r: dict) -> tuple:
        return tuple(str(r.get(k, "")) for k in ("kind", "plate_id", "well_id", "image_id", "obj"))

    # ------------------------------------------------------------------------------------------------- answers
    def answer(self, a: str):
        r = self._row()
        if r is None:
            return
        k = self.page * COLS * ROWS + self.cur
        self.items.loc[k, "answer"] = a
        if self.on_answer:
            self.on_answer(r, a)
        self.answered.emit()
        # on to the next unanswered item
        nxt = [j for j in range(k + 1, len(self.items)) if not self.items.answer.iloc[j]]
        if a and nxt:
            j = nxt[0]
            if j // (COLS * ROWS) != self.page:
                self.cur = j % (COLS * ROWS)
                self.set_page(j // (COLS * ROWS))
            else:
                self.focus(j % (COLS * ROWS))
        else:
            self._paint()

    def _chan_toggled(self, c, on):
        r = self._row()
        if r is None:
            return
        st = set(self.chan_state.get(self._key(r), set()))
        st = st | {c} if on else st - {c}
        self.chan_state[self._key(r)] = st
        if self.on_channels:
            self.on_channels(r, st)

    def keyPressEvent(self, e):
        k = e.key()
        if Qt.Key_1 <= k <= Qt.Key_9 and k - Qt.Key_1 < len(self.choices):
            self.answer(self.choices[k - Qt.Key_1])
        elif k == Qt.Key_0:
            self.answer("")
        elif k == Qt.Key_Space:
            r = self._row()
            if r is not None:
                self.open_fn(r)
        elif k in (Qt.Key_Right, Qt.Key_Left, Qt.Key_Up, Qt.Key_Down):
            d = {Qt.Key_Right: 1, Qt.Key_Left: -1, Qt.Key_Up: -COLS, Qt.Key_Down: COLS}[k]
            j = self.page * COLS * ROWS + self.cur + d
            if 0 <= j < len(self.items):
                if j // (COLS * ROWS) != self.page:
                    self.cur = j % (COLS * ROWS)
                    self.set_page(j // (COLS * ROWS))
                else:
                    self.focus(j % (COLS * ROWS))
        elif k == Qt.Key_PageDown:
            self.set_page(self.page + 1)
        elif k == Qt.Key_PageUp:
            self.set_page(self.page - 1)
        else:
            super().keyPressEvent(e)
