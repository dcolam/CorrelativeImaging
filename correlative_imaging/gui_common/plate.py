"""Quick plate view: one small button per well, filling the panel's width, coloured by a mode the caller defines.
The plate format (96 / 384 / 1536) follows from the well ids ("A1" and "A01" both work).

The caller passes
  modes      {mode name: colour_fn(record) -> hex}
  legend_fn  (mode, rows of the shown plate) -> rich text under the plate
  frame_fn   record -> CSS border of a well (None: the default thin border)
  tip_fn     (plate label, well id, record) -> tooltip
and feeds set(df, stage) with one row per well (columns plate_id, well_id and whatever the functions read).
Click a well: well_clicked(plate_id, well_id); double-click: well_double_clicked; another plate picked:
plate_changed(plate_id).
"""
from __future__ import annotations

import pandas as pd
from qtpy.QtCore import Qt, QTimer, Signal
from qtpy.QtWidgets import (QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QScrollArea, QSizePolicy,
                            QToolButton, QVBoxLayout, QWidget)

FORMATS = [(8, 12), (16, 24), (32, 48)]
CURRENT = "2px solid #4fc3f7"
DEFAULT_BORDER = "1px solid #20232a"


def rc(well: str):
    """(row, column), 0-based, of a well id like B7, B07 or AA12."""
    row = 0
    for c in "".join(c for c in str(well) if c.isalpha()):
        row = row * 26 + (ord(c.upper()) - 64)
    return row - 1, int("".join(c for c in str(well) if c.isdigit())) - 1


def row_name(i: int) -> str:
    return chr(65 + i) if i < 26 else chr(64 + i // 26) + chr(65 + i % 26)


class _Well(QToolButton):
    double = Signal()

    def mouseDoubleClickEvent(self, e):
        self.double.emit()


class PlateGrid(QWidget):
    well_clicked = Signal(str, str)
    well_double_clicked = Signal(str, str)
    plate_changed = Signal(str)

    def __init__(self, modes: dict, legend_fn=None, frame_fn=None, tip_fn=None):
        super().__init__()
        self.modes = modes
        self.legend_fn = legend_fn or (lambda mode, d: "")
        self.frame_fn = frame_fn or (lambda r: None)
        self.tip_fn = tip_fn or (lambda pl, w, r: f"{pl} {w}")
        v = QVBoxLayout(self)
        v.setContentsMargins(4, 4, 4, 4)
        h = QHBoxLayout()
        self.plate = QComboBox()
        self.plate.currentIndexChanged.connect(self._plate_picked)
        self.mode = QComboBox()
        self.mode.addItems(list(modes))
        self.mode.currentTextChanged.connect(self.draw)
        self.mode.setVisible(len(modes) > 1)
        h.addWidget(self.plate, 1)
        h.addWidget(self.mode)
        v.addLayout(h)
        self.stage = QLabel("")
        self.stage.setProperty("role", "hint")
        v.addWidget(self.stage)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.grid_w = QWidget()
        self.grid = QGridLayout(self.grid_w)
        self.grid.setSpacing(1)
        self.grid.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.scroll.setWidget(self.grid_w)
        v.addWidget(self.scroll, 1)
        self.legend = QLabel("")
        self.legend.setWordWrap(True)
        self.legend.setTextFormat(Qt.RichText)
        v.addWidget(self.legend)
        self.df = pd.DataFrame()
        self.current = None
        self.btns, self.recs = {}, {}
        self._all = []

    # -- data
    def set(self, df: pd.DataFrame, stage: str = "", labels: dict | None = None):
        """df: one row per well; labels: {plate_id: text shown in the plate list} (default: the id)."""
        self.df = df
        cur = self.plate_id()
        self.plate.blockSignals(True)
        self.plate.clear()
        plates = list(dict.fromkeys(df.plate_id)) if len(df) else []
        for p in sorted(plates) if labels is None else plates:
            self.plate.addItem((labels or {}).get(p, str(p)), p)
        i = self.plate.findData(cur)
        if i >= 0:
            self.plate.setCurrentIndex(i)
        self.plate.blockSignals(False)
        self.stage.setText(f"colours: {stage}" if stage else "")
        self.stage.setVisible(bool(stage))
        self.draw()

    def plate_id(self):
        return self.plate.currentData()

    def show_plate(self, plate_id):
        i = self.plate.findData(plate_id)
        if i >= 0 and i != self.plate.currentIndex():
            self.plate.setCurrentIndex(i)

    def _plate_picked(self, *_):
        self.draw()
        if self.plate_id() is not None:
            self.plate_changed.emit(self.plate_id())

    def mark(self, plate_id, well_id):
        """Frame the open well (the plate switches if needed)."""
        prev, self.current = self.current, (plate_id, rc(well_id))
        if self.plate_id() != plate_id:
            self.show_plate(plate_id)                       # redraws the plate
            return
        for key in (prev, self.current):                    # same plate: restyle just the two buttons
            if key and key[0] == plate_id and key[1] in self.btns:
                self._style(key[1])

    # -- drawing
    def _style(self, pos):
        r, b = self.recs[pos], self.btns[pos]
        border = self.frame_fn(r) or DEFAULT_BORDER
        if self.current == (self.plate_id(), pos):
            border = CURRENT
        col = self.modes[self.mode.currentText()](r)
        b.setStyleSheet(f"QToolButton {{ background: {col}; border: {border}; border-radius: 3px; }}")

    def _size(self) -> int:
        """Well button size: the plate fills the available width (9 to 30 px per well)."""
        C = getattr(self, "_C", 12)
        w = self.width() if self.width() > 120 else 0
        if not w:
            return 11 if C > 24 else 15 if C > 12 else 22
        return int(max(9, min(30, (w - 34) // (C + 1) - 1)))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._fit()

    def _fit(self):
        s = self._size()
        for b in self._all:
            if b.width() != s:
                b.setFixedSize(s, s)
        if getattr(self, "_R", None):
            h = (self._R + 1) * (max(s, 12) + 2) + 14     # rows are at least as tall as their labels
            self.scroll.setMinimumHeight(h)               # the whole plate, no inner scrolling
            self.scroll.setMaximumHeight(h)

    def draw(self, *_):
        self.btns, self.recs = {}, {}
        self._all = []
        while self.grid.count():
            w = self.grid.takeAt(0).widget()
            if w:
                w.setParent(None)
                w.deleteLater()
        pl = self.plate_id()
        if not len(self.df) or pl is None:
            self.legend.setText("")
            return
        label = self.plate.currentText()
        d = self.df[self.df.plate_id == pl]
        pos = [rc(w) for w in d.well_id]
        nr, nc = max(r for r, _ in pos) + 1, max(c for _, c in pos) + 1
        R, C = next(((a, b) for a, b in FORMATS if nr <= a and nc <= b), (nr, nc))
        small = C > 12
        for j in range(C):
            lab = QLabel(str(j + 1) if not small or j % 2 == 0 else "")   # every other number on big plates
            lab.setAlignment(Qt.AlignCenter)
            lab.setProperty("role", "hint")
            lab.setStyleSheet("font-size: 8px;" if small else "")
            lab.setMinimumWidth(0)
            self.grid.addWidget(lab, 0, j + 1)
        for i in range(R):
            lab = QLabel(row_name(i))
            lab.setProperty("role", "hint")
            lab.setStyleSheet("font-size: 8px;" if small else "")
            self.grid.addWidget(lab, i + 1, 0)
        self._C, self._R = C, R
        size = self._size()
        self.scroll.setMinimumHeight((R + 1) * (max(size, 12) + 2) + 14)     # the whole plate, no scrolling
        by = {p: (w, r) for p, w, r in zip(pos, d.well_id, d.to_dict("records"))}
        for i in range(R):
            for j in range(C):
                b = _Well()
                b.setFixedSize(size, size)
                b.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
                self._all.append(b)
                got = by.get((i, j))
                if got is None:
                    b.setEnabled(False)
                    b.setStyleSheet("QToolButton { background: transparent; border: 1px solid #2a2e35; }")
                    self.grid.addWidget(b, i + 1, j + 1)
                    continue
                well, r = got
                self.btns[(i, j)], self.recs[(i, j)] = b, r
                self._style((i, j))
                b.setToolTip(self.tip_fn(label, well, r))
                b.clicked.connect(lambda _=False, w=well: self.well_clicked.emit(pl, w))
                b.double.connect(lambda w=well: self.well_double_clicked.emit(pl, w))
                self.grid.addWidget(b, i + 1, j + 1)
        QTimer.singleShot(0, self._fit)                    # once the panel has its width
        self.legend.setText(self.legend_fn(self.mode.currentText(), d))
