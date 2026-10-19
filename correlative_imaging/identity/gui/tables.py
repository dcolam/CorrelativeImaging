"""A read-only pandas table model with optional per-cell colour chips, and a filterable table view."""
from __future__ import annotations

import numpy as np
import pandas as pd
from qtpy.QtCore import QAbstractTableModel, QModelIndex, QSortFilterProxyModel, Qt
from qtpy.QtGui import QBrush, QColor
from qtpy.QtWidgets import QAbstractItemView, QHeaderView, QTableView


class FrameModel(QAbstractTableModel):
    def __init__(self, df: pd.DataFrame | None = None, colors: dict | None = None, fmt: dict | None = None):
        super().__init__()
        self.df = pd.DataFrame() if df is None else df.reset_index(drop=True)
        self.colors = colors or {}      # column -> callable(value) -> hex | None (background)
        self.fmt = fmt or {}            # column -> format string

    def set(self, df: pd.DataFrame):
        self.beginResetModel()
        self.df = df.reset_index(drop=True)
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return len(self.df)

    def columnCount(self, parent=QModelIndex()):
        return self.df.shape[1]

    def data(self, idx, role=Qt.DisplayRole):
        if not idx.isValid():
            return None
        col = self.df.columns[idx.column()]
        v = self.df.iat[idx.row(), idx.column()]
        if role == Qt.DisplayRole:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return ""
            if col in self.fmt and isinstance(v, (int, float, np.number)):
                return self.fmt[col].format(v)
            if isinstance(v, (float, np.floating)):
                return f"{v:.3g}"
            return str(v)
        if role == Qt.UserRole:                      # sort key
            return v if isinstance(v, (int, float, np.number)) and not pd.isna(v) else str(v)
        if role == Qt.BackgroundRole and col in self.colors:
            c = self.colors[col](v)
            if c:
                return QBrush(QColor(c))
        if role == Qt.ForegroundRole and col in self.colors:
            c = self.colors[col](v)
            if c:
                q = QColor(c)
                return QBrush(QColor("#111111" if q.lightnessF() > 0.55 else "#f0f1f2"))
        if role == Qt.TextAlignmentRole and isinstance(v, (int, float, np.number)):
            return int(Qt.AlignRight | Qt.AlignVCenter)
        return None

    def headerData(self, s, orient, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        return str(self.df.columns[s]) if orient == Qt.Horizontal else str(s + 1)

    def row(self, i: int) -> pd.Series:
        return self.df.iloc[i]


class FrameView(QTableView):
    def __init__(self, model: FrameModel, parent=None, stretch_last=True, stretch=False):
        super().__init__(parent)
        self.src = model
        self.proxy = QSortFilterProxyModel(self)
        self.proxy.setSourceModel(model)
        self.proxy.setSortRole(Qt.UserRole)
        self.proxy.setFilterKeyColumn(-1)
        self.proxy.setFilterCaseSensitivity(Qt.CaseInsensitive)
        self.setModel(self.proxy)
        self.setSortingEnabled(True)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.verticalHeader().setVisible(False)
        self.verticalHeader().setDefaultSectionSize(22)
        self.horizontalHeader().setHighlightSections(False)
        self.horizontalHeader().setStretchLastSection(stretch_last)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch if stretch else QHeaderView.ResizeToContents)
        self.setAlternatingRowColors(False)
        self.setShowGrid(False)

    def current_row(self) -> pd.Series | None:
        idx = self.currentIndex()
        if not idx.isValid():
            return None
        return self.src.row(self.proxy.mapToSource(idx).row())
