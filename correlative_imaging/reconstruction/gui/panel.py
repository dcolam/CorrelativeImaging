"""The right-hand dock: Run, Train, Results, Settings. Each tab is a plain QWidget driven by the app controller."""
from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QFrame, QGridLayout,
                            QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit, QProgressBar, QPushButton,
                            QScrollArea, QSpinBox, QTabWidget, QToolButton, QVBoxLayout, QWidget, QSizePolicy)

from ..config import MODELS, Config, load_config, presets
from ..labels import CLASSES
from .tables import FrameModel, FrameView
from .theme import CLASS_COLORS, CLASS_KEYS, call_color, chip

# plain-language label, explanation (tooltip), group
SETTINGS = {
    # safe to change: only Classify has to run again
    "p_seed": ("P(cell) to start a cell", "a particle needs P(cell) + P(merged) at least this high to seed a cell", "safe"),
    "split_dips": ("split at signal dips", "split a bright area where the signal dips between two peaks; off = "
                   "connected bright areas stay one cell", "safe"),
    "dip_ratio": ("dip that splits (saddle / lower peak)", "lower = only deep dips split; higher = split more "
                  "readily", "safe"),
    "cell_gap_um": ("join pieces closer than (µm)", "seeds of different channels closer than this join into one "
                    "cell (the same cell seen in two channels)", "safe"),
    "grow_floor": ("growth floor (hole-noise SDs)", "grown pixels must be at least this far above the hole "
                   "background", "safe"),
    "grow_frac": ("growth cut (fraction of the seed)", "grown pixels must reach this fraction of the seed's median "
                  "signal (a dim cell grows by its own standard)", "safe"),
    "grow_margin_um": ("max growth beyond the hole (µm)", "how far a cell may grow past the hole rim", "safe"),
    "grow_max_x_hole": ("max cell area (× hole area)", "if larger, the growth cut is raised until the cell fits",
                        "safe"),
    "p_call": ("P needed for a call", "P of the best channel combination below this = 'uncertain'", "safe"),
    "identity_bg": ("background under the cell", "hole = the hole's own background (removes aperture glow); "
                    "frame = the whole frame outside particles", "safe"),
    "min_mixture_sep": ("flag channels separating less than", "per-channel positive/negative separation "
                        "(Ashman's D) below which a channel is flagged", "safe"),
    # define the features: locked to the model
    "work_R": ("analysis window (hole radii)", "window around the hole; whole-object features depend on it", "locked"),
    "context_R": ("context disc (hole radii)", "disc for neighbour density and brightness ranks", "locked"),
    "min_object_px": ("smallest object (px)", "objects up to this size are removed, as in the pipeline", "locked"),
    "min_inside_um2": ("min area inside the hole (µm²)", "a particle needs at least this much area in the hole",
                       "locked"),
    "field_min_um2": ("typical-cell objects from (µm²)", "whole-image objects this big describe a typical "
                      "unpatched cell (the brightness unit)", "locked"),
    "smooth_px": ("smoothing (px)", "light smoothing for peaks and growth (measurements are unsmoothed)", "locked"),
    "peak_h": ("peak height (hole-noise SDs)", "a peak must rise this far above its surroundings", "locked"),
    "view_R": ("area without a hole crop (hole radii)", "runs whose hole crop cannot be used: how much of the whole "
               "frame Detect covers and shows (with a crop, all of it). Cells out there are neighbours; the model's "
               "features stay on the analysis window", "local"),
    "path_map": ("path map (JSON)", 'stored path prefix -> local prefix, e.g. {"Z:\\\\": "/mnt/z/"}', "local"),
}


def section(text):
    l = QLabel(text)
    l.setProperty("role", "section")
    return l


def hint(text):
    l = QLabel(text)
    l.setProperty("role", "hint")
    l.setWordWrap(True)
    return l


def accent(btn: QPushButton) -> QPushButton:
    btn.setProperty("accent", True)
    return btn


def path_row(edit: QLineEdit, on_browse) -> QWidget:
    w = QWidget()
    h = QHBoxLayout(w)
    h.setContentsMargins(0, 0, 0, 0)
    h.addWidget(edit, 1)
    b = QToolButton()
    b.setText("…")
    b.clicked.connect(on_browse)
    h.addWidget(b)
    return w


def card(title: str) -> tuple[QFrame, QLabel]:
    f = QFrame()
    f.setProperty("role", "card")
    v = QVBoxLayout(f)
    v.setContentsMargins(10, 6, 10, 6)
    val = QLabel("–")
    val.setProperty("role", "metric")
    v.addWidget(val)
    v.addWidget(hint(title))
    return f, val


# ---------------------------------------------------------------------------------------------------------------

class RunTab(QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        v = QVBoxLayout(body)
        v.addWidget(section("Source"))
        f = QFormLayout()
        self.root = QLineEdit()
        self.root.setPlaceholderText("CorrelativeImaging output folder (holds the plate folders)")
        self.root.editingFinished.connect(self.find_runs)
        f.addRow("output folder", path_row(self.root, self.browse_root))
        self.run = QComboBox()
        self.run.currentTextChanged.connect(self.default_out)
        f.addRow("run", self.run)
        self.preset = QComboBox()
        self.preset.addItems(["(neutral)"] + presets())
        self.preset.currentTextChanged.connect(self.load_preset)
        f.addRow("preset", self.preset)
        self.out = QLineEdit()
        f.addRow("results folder", path_row(self.out, self.browse_out))
        self.wells = QLineEdit()
        self.wells.setPlaceholderText("optional: 18T39412_N11 18T39383_M13 …")
        f.addRow("wells", self.wells)
        v.addLayout(f)
        v.addWidget(section("Channels"))
        v.addWidget(hint("Names and colours are display only (decisions use the raw channel). A marker channel is "
                         "measured but never counts as identity (e.g. a channel on every cell). Untick to ignore a "
                         "channel."))
        self.chan_grid = QGridLayout()
        self.chan_rows = {}
        v.addLayout(self.chan_grid)
        f2 = QFormLayout()
        self.model = QComboBox()
        self.model.setEditable(True)
        self.model.setToolTip("trained particle classifier; empty = Detect only (a first look without a model)")
        f2.addRow("model", self.model)
        v.addLayout(f2)
        h = QHBoxLayout()
        self.b_detect = QPushButton("Detect")
        self.b_detect.setToolTip("particles, field reference, features and a first look per well: provisional "
                                 "classes (inside / neighbour / debris / noise) and the dominant channel. No model "
                                 "needed")
        self.b_classify = QPushButton("Classify")
        self.b_classify.setToolTip("classify particles, reconstruct cells, call identity (needs a model)")
        self.b_all = accent(QPushButton("Detect + classify"))
        for b in (self.b_detect, self.b_classify, self.b_all):
            h.addWidget(b)
        self.b_detect.clicked.connect(lambda: app.run_stage("detect"))
        self.b_classify.clicked.connect(lambda: app.run_stage("classify"))
        self.b_all.clicked.connect(lambda: app.run_stage("all"))
        v.addLayout(h)
        b = QPushButton("Open existing results…")
        b.clicked.connect(self.open_results)
        v.addWidget(b)
        v.addWidget(section("Current results"))
        g = QGridLayout()
        self.cards = {}
        for i, (k, t) in enumerate([("wells", "wells"), ("pool", "particles in holes"), ("labels", "hand labels"),
                                    ("model", "model")]):
            c, val = card(t)
            self.cards[k] = val
            g.addWidget(c, i // 2, i % 2)
        v.addLayout(g)
        self.info = hint("")
        v.addWidget(self.info)
        v.addWidget(section("Progress"))
        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        v.addWidget(self.progress)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setProperty("role", "log")
        self.log.setMaximumBlockCount(2000)
        self.log.setMinimumHeight(160)
        v.addWidget(self.log, 1)
        scroll.setWidget(body)
        outer.addWidget(scroll)

    def browse_root(self):
        d = QFileDialog.getExistingDirectory(self, "CorrelativeImaging output folder", self.root.text())
        if d:
            self.root.setText(d)
            self.find_runs()

    def browse_out(self):
        d = QFileDialog.getExistingDirectory(self, "Results folder", self.out.text())
        if d:
            self.out.setText(d)

    # -- channels
    def set_channels(self, channels: list, cfg: Config):
        from qtpy.QtGui import QColor
        while self.chan_grid.count():
            w = self.chan_grid.takeAt(0).widget()
            if w:
                w.setParent(None)
                w.deleteLater()
        self.chan_rows = {}
        for i, t in enumerate(["use", "channel", "name", "colour", "role"]):
            self.chan_grid.addWidget(hint(t), 0, i)
        for i, ch in enumerate(channels, start=1):
            use = QCheckBox()
            use.setChecked(not cfg.channels or ch in cfg.channels)
            name = QLineEdit(cfg.display_name(ch))
            col = QToolButton()
            col.setFixedWidth(46)
            col.setProperty("hex", cfg.color(ch))
            col.setStyleSheet(f"QToolButton {{ background: {cfg.color(ch)}; border-radius: 3px; }}")

            def pick(_=False, b=col, ch=ch):
                from qtpy.QtWidgets import QColorDialog
                c = QColorDialog.getColor(QColor(b.property("hex")), self, f"Colour of {ch}")
                if c.isValid():
                    b.setProperty("hex", c.name())
                    b.setStyleSheet(f"QToolButton {{ background: {c.name()}; border-radius: 3px; }}")
            col.clicked.connect(pick)
            role = QComboBox()
            role.addItems(["identity", "marker"])
            role.setCurrentText("marker" if ch in cfg.markers else "identity")
            for j, w in enumerate([use, QLabel(ch), name, col, role]):
                self.chan_grid.addWidget(w, i, j)
            self.chan_rows[ch] = (use, name, col, role)

    def apply_channels(self, cfg: Config) -> Config:
        if not self.chan_rows:
            return cfg
        used = [ch for ch, (u, *_rest) in self.chan_rows.items() if u.isChecked()]
        cfg.channels = used if len(used) < len(self.chan_rows) else []
        cfg.markers = [ch for ch, (u, n, c, r) in self.chan_rows.items() if r.currentText() == "marker"]
        cfg.display = {ch: {"name": n.text().strip() or ch, "color": c.property("hex")}
                       for ch, (u, n, c, r) in self.chan_rows.items()}
        return cfg

    def set_models(self, current: str):
        self.model.blockSignals(True)
        self.model.clear()
        self.model.addItems([""] + sorted(p.name for p in MODELS.glob("*.joblib")))
        self.model.setCurrentText(current or "")
        self.model.blockSignals(False)

    def run_channels(self, tag: str) -> list:
        """Fluorescence channels of a run, from its first plate database."""
        import sqlite3
        r = self._runs.get(tag) if hasattr(self, "_runs") else None
        if r is None or not r.plates:
            return []
        con = sqlite3.connect(f"file:{r.plates[0].db_path}?mode=ro", uri=True)
        try:
            return [c for (c,) in con.execute("select distinct channel from particle_measurements order by channel")]
        finally:
            con.close()

    def explorer_display(self, tag: str) -> dict:
        """Channel names/colours the Results Explorer stored for this run (ci_labels.db in the output folder)."""
        import sqlite3
        db = Path(self.root.text().strip()) / "ci_labels.db"
        if not db.exists():
            return {}
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            rows = con.execute("select channel, display_name, color_hex from channel_display where run_tag=?",
                               (tag,)).fetchall()
        except sqlite3.Error:
            rows = []
        finally:
            con.close()
        return {c: {"name": n, "color": h} for c, n, h in rows if n or h}

    def find_runs(self):
        from .._discovery import discover_runs
        self.run.clear()
        root = self.root.text().strip()
        if not root:
            return
        runs = discover_runs(root)
        self._runs = {r.run_tag: r for r in runs}
        self.run.addItems([r.run_tag for r in runs])
        self.app.say(f"{len(runs)} run(s) under {root}")
        if runs:
            self.run.setCurrentIndex(len(runs) - 1)

    def default_out(self, tag):
        if not tag or not self.root.text().strip():
            return
        self.out.setText(str(Path(self.root.text().strip()) / f"reconstruction_{tag}"))
        chans = self.run_channels(tag)
        if not chans:
            return
        # a preset whose channels are exactly this run's channels is suggested
        from ..config import load_config as _lc
        if self.preset.currentText() == "(neutral)":
            for name in presets():
                if set(_lc(name).channels) == set(chans):
                    self.app.say(f"preset '{name}' matches the channels of this run")
                    self.preset.setCurrentText(name)
                    break
        cfg = self.app.cfg
        shown = self.explorer_display(tag)
        if shown:
            cfg.display = {**cfg.display, **shown}
            self.app.say("channel names/colours taken from the Results Explorer")
        self.set_channels(chans, cfg)

    def load_preset(self, name):
        cfg = Config() if name == "(neutral)" else load_config(name)
        cfg.model = cfg.model or self.app.cfg.model
        cfg.path_map = self.app.cfg.path_map
        self.app.set_config(cfg)

    def open_results(self):
        d = QFileDialog.getExistingDirectory(self, "Results folder of this tool", self.out.text())
        if d:
            self.app.open_results(d)

    def update(self, res, store, model_spec):
        self.cards["wells"].setText(str(len(res.wells)) if res else "–")
        self.cards["pool"].setText(f"{int(res.particles.pool.sum()):,}" if res else "–")
        n = len(store.frame(res.dataset)) if (res and store) else 0
        self.cards["labels"].setText(f"{n:,}")
        self.cards["model"].setText(model_spec.get("arm", "–") if model_spec else "none")
        if res:
            src = res.run.get("source") or {}
            if src.get("kind") == "run" and not self.root.text():
                self.root.blockSignals(True)
                self.root.setText(src.get("root", ""))
                self.root.blockSignals(False)
                self.find_runs()
                self.run.setCurrentText(src.get("run_tag", ""))
                self.out.setText(str(res.dir))
            self.info.setText(f"dataset <b>{res.dataset}</b> · channels {', '.join(res.channels)} · "
                              f"{'classified' if res.calls is not None else 'detected, not classified'}")


class SettingsTab(QWidget):
    """Last tab, for fine-tuning. Defaults work; change them only with a reason.
      safe      reconstruction and call: change, then Classify again
      locked    detection settings that define the features; a model only fits detections made with the same values,
                so they are locked to the model (unlock to change, then Detect again and retrain)
      recommendations from the detected run (applied only on click)"""

    def __init__(self, app):
        super().__init__()
        self.app = app
        outer = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        v = QVBoxLayout(body)
        v.addWidget(hint("The defaults work. Change a setting only with a reason; hover for what it does."))
        self.widgets = {}
        self.model_val = {}
        box = QGroupBox("Recommendations from this run")
        self.rec = QVBoxLayout(box)
        self.rec_none = hint("Detect a run to get recommendations.")
        self.rec.addWidget(self.rec_none)
        v.addWidget(box)
        for grp, title in (("safe", "Reconstruction and call · safe to change, then Classify again"),
                           ("locked", "Detection · defines the features, locked to the model"),
                           ("local", "Display and this computer")):
            box = QGroupBox(title)
            form = QFormLayout(box)
            if grp == "locked":
                self.unlock = QCheckBox("unlock (then Detect again and retrain)")
                self.unlock.toggled.connect(self._lock)
                form.addRow(self.unlock)
            for k, (label, tip, g) in SETTINGS.items():
                if g != grp:
                    continue
                w = self._widget(k)
                w.setToolTip(tip)
                lab = QLabel(label)
                lab.setToolTip(tip)
                if grp == "locked":
                    row = QWidget()
                    h = QHBoxLayout(row)
                    h.setContentsMargins(0, 0, 0, 0)
                    h.addWidget(w, 1)
                    mv = QLabel("")
                    mv.setProperty("role", "hint")
                    self.model_val[k] = mv
                    h.addWidget(mv)
                    form.addRow(lab, row)
                else:
                    d = getattr(Config(), k)
                    row = QWidget()
                    h = QHBoxLayout(row)
                    h.setContentsMargins(0, 0, 0, 0)
                    h.addWidget(w, 1)
                    dl = QLabel(f"default {d}" if k != "path_map" else "")
                    dl.setProperty("role", "hint")
                    dl.setMinimumWidth(80)
                    h.addWidget(dl)
                    form.addRow(lab, row)
                self.widgets[k] = w
            v.addWidget(box)
        v.addStretch(1)
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)
        h = QHBoxLayout()
        b = accent(QPushButton("Apply"))
        b.clicked.connect(self.apply)
        r = QPushButton("Reset to defaults")
        r.clicked.connect(self.reset)
        s_ = QPushButton("Save as preset…")
        s_.clicked.connect(self.save)
        for x in (b, r, s_):
            h.addWidget(x)
        outer.addLayout(h)
        self._lock(False)

    def _widget(self, k):
        default = getattr(Config(), k)
        if k == "identity_bg":
            w = QComboBox()
            w.addItems(["hole", "frame"])
            return w
        if isinstance(default, bool):
            return QCheckBox()
        if isinstance(default, int):
            w = QSpinBox()
            w.setRange(0, 100000)
            return w
        if isinstance(default, float):
            w = QDoubleSpinBox()
            w.setDecimals(3)
            w.setRange(0, 1000)
            w.setSingleStep(0.05 if default < 2 else 0.5)
            return w
        return QLineEdit()

    def _lock(self, unlocked: bool):
        for k, (_, _, g) in SETTINGS.items():
            if g == "locked":
                self.widgets[k].setEnabled(bool(unlocked))

    def set_config(self, cfg: Config, channels: list | None = None):
        for k, w in self.widgets.items():
            v = getattr(cfg, k)
            if isinstance(w, QCheckBox):
                w.setChecked(bool(v))
            elif isinstance(w, (QSpinBox, QDoubleSpinBox)):
                w.setValue(v)
            elif isinstance(w, QComboBox):
                w.setCurrentText(str(v))
            elif isinstance(v, dict):
                w.setText(json.dumps(v) if v else "")
            else:
                w.setText(str(v))
        self.show_model_match()

    def show_model_match(self):
        from ..config import complete_feature_config
        m = self.app.model
        fc = complete_feature_config(m.spec.get("feature_config") or {}) \
            if m is not None and m.spec.get("feature_config") else {}
        for k, lab in self.model_val.items():
            if k not in fc:
                lab.setText("" if m is None else "model: ?")
                continue
            cur = getattr(self.app.cfg, k)
            ok = cur == fc[k]
            lab.setText(f"model {fc[k]} {'✓' if ok else '✗'}")
            lab.setStyleSheet("" if ok else "color: #e0a526;")

    def read(self) -> Config:
        import copy
        cfg = copy.deepcopy(self.app.cfg)
        for k, w in self.widgets.items():
            d = getattr(Config(), k)
            if isinstance(w, QCheckBox):
                v = w.isChecked()
            elif isinstance(w, (QSpinBox, QDoubleSpinBox)):
                v = type(d)(w.value())
            elif isinstance(w, QComboBox):
                v = w.currentText()
            elif isinstance(d, dict):
                v = json.loads(w.text()) if w.text().strip() else {}
            else:
                v = w.text()
            setattr(cfg, k, v)
        return cfg

    def apply(self):
        try:
            self.app.set_config(self.read(), from_form=True)
            self.app.say("settings applied")
        except Exception as e:
            self.app.say(f"settings not applied: {e}")

    def reset(self):
        import copy
        cfg = copy.deepcopy(self.app.cfg)
        for k in SETTINGS:
            if k != "path_map":
                setattr(cfg, k, getattr(Config(), k))
        self.app.set_config(cfg)

    def recommend(self, items: list):
        """items: [(text, callable or None)]"""
        while self.rec.count():
            w = self.rec.takeAt(0).widget()
            if w:
                w.setParent(None)
                w.deleteLater()
        if not items:
            self.rec.addWidget(hint("Nothing to recommend: the current settings fit this run."))
            return
        for text, fn in items:
            row = QWidget()
            h = QHBoxLayout(row)
            h.setContentsMargins(0, 0, 0, 0)
            l = QLabel(text)
            l.setWordWrap(True)
            h.addWidget(l, 1)
            if fn is not None:
                b = QPushButton("Apply")
                b.clicked.connect(fn)
                h.addWidget(b)
            self.rec.addWidget(row)

    def save(self):
        from ..config import PRESETS
        p, _ = QFileDialog.getSaveFileName(self, "Save preset", str(PRESETS / "my_dataset.json"), "JSON (*.json)")
        if p:
            cfg = self.app.read_config()
            d = json.loads(cfg.to_json())
            d.pop("path_map", None)
            Path(p).write_text(json.dumps(d, indent=2))
            self.app.say(f"preset saved: {p}")


def _scroll(body: QWidget) -> QScrollArea:
    sc = QScrollArea()
    sc.setWidgetResizable(True)
    sc.setFrameShape(QFrame.NoFrame)
    sc.setWidget(body)
    return sc


class Collapsible(QWidget):
    """A section that opens on click (advanced options stay out of the way)."""

    def __init__(self, title: str, body: QWidget, open_: bool = False):
        super().__init__()
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        self.btn = QToolButton()
        self.btn.setText(title)
        self.btn.setCheckable(True)
        self.btn.setChecked(open_)
        self.btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.btn.setArrowType(Qt.DownArrow if open_ else Qt.RightArrow)
        self.btn.setStyleSheet("QToolButton { border: none; font-weight: bold; }")
        self.btn.toggled.connect(self._toggle)
        v.addWidget(self.btn)
        self.body = body
        body.setVisible(open_)
        v.addWidget(body)

    def _toggle(self, on):
        self.btn.setArrowType(Qt.DownArrow if on else Qt.RightArrow)
        self.body.setVisible(on)


class TrainTab(QWidget):
    """Holds the widgets of the labelling and model steps; they are shown on three pages:
    `setup_page` and `canvas_page` (Label tab) and `model_page` (Model tab). Never shown itself."""

    def __init__(self, app):
        super().__init__()
        self.app = app

        # ---- Label tab, top: library, test wells, blind -------------------------------------------------------
        self.setup_page = QWidget()
        v = QVBoxLayout(self.setup_page)
        v.setContentsMargins(4, 4, 4, 4)
        h = QHBoxLayout()
        h.addWidget(QLabel("label library"))
        self.store_path = QLineEdit()
        self.store_path.setToolTip("one SQLite file per project; a ci_labels.db next to the results folder is opened "
                                   "automatically")
        self.store_path.editingFinished.connect(lambda: app.set_store(self.store_path.text()))
        h.addWidget(path_row(self.store_path, self.browse_store), 1)
        v.addLayout(h)
        h = QHBoxLayout()
        self.b_freeze = QPushButton("Freeze test wells")
        self.b_freeze.setToolTip("20 % of the wells with something in the hole, per plate, chosen once and before "
                                 "any labelling: held out, labelled blind, never trained on")
        self.b_freeze.clicked.connect(lambda: app.freeze_test_wells())
        h.addWidget(self.b_freeze)
        b = QPushButton("…take over")
        b.setToolTip("use the test wells of an older label library of this dataset and draw the rest only from wells "
                     "it never labelled (a model trained on it can then be compared fairly)")
        b.clicked.connect(self.freeze_from_other)
        h.addWidget(b)
        self.test_info = QLabel("")
        self.test_info.setProperty("role", "hint")
        h.addWidget(self.test_info, 1)
        v.addLayout(h)
        h = QHBoxLayout()
        self.blind_test = QCheckBox("always blind in test wells")
        self.blind_test.setChecked(True)
        self.blind_test.setToolTip("in test wells, hide every class, call, probability and reconstructed cell "
                                   "(canvas, plate, lists, particle map)")
        self.blind = QCheckBox("blind everywhere")
        self.blind.setToolTip("the same for every well")
        for cb in (self.blind_test, self.blind):
            cb.toggled.connect(app.blind_changed)
            h.addWidget(cb)
        h.addStretch(1)
        v.addLayout(h)

        # ---- Label tab, "In the canvas": click labelling, cell labels, particle edits -------------------------
        self.canvas_page = QWidget()
        v = QVBoxLayout(self.canvas_page)
        v.addWidget(hint("Label any particle where you see it: turn on label mode, click a particle in the canvas "
                         "(several overlap: pick from the menu), press 1–6 or a button; Del removes the label. "
                         "Labels are stored by position, so they survive re-detection. Use this for particles the "
                         "batches did not show you, and to fix segmentation (below)."))
        h = QHBoxLayout()
        self.mode = QPushButton("Label mode  (L)")
        self.mode.setCheckable(True)
        self.mode.toggled.connect(app.set_label_mode)
        h.addWidget(self.mode, 2)
        rnd = QPushButton("Random well")
        rnd.setToolTip("a random unlabelled well (test wells excluded) from the current well list")
        rnd.clicked.connect(app.random_well)
        nxt = QPushButton("Next well  (N)")
        nxt.clicked.connect(lambda: app.browser.step(1))
        h.addWidget(rnd, 1)
        h.addWidget(nxt, 1)
        v.addLayout(h)
        h = QHBoxLayout()
        self.show_whole = QCheckBox("stringent")
        self.show_hole = QCheckBox("lenient")
        self.show_ctx = QCheckBox("cells around")
        for cb in (self.show_whole, self.show_hole, self.show_ctx):   # kept for the display; both levels always drawn
            cb.setChecked(True)
            cb.setVisible(False)
        self.show_noise = QCheckBox("include noise particles in the particle layers")
        self.show_noise.setToolTip("particles classed noise (often background speckle at the lenient threshold) are "
                                   "left out of the particle fills by default")
        self.show_noise.toggled.connect(app.redraw_levels)
        h.addWidget(self.show_noise)
        h.addStretch(1)
        v.addLayout(h)
        v.addWidget(hint("Layers (channels, particles, neighbours, cell outline, tags, edits): Channels dock on the "
                         "left. Tags show each particle's class; turn them on for labelling here."))
        grid = QGridLayout()
        for i, c in enumerate(CLASSES):
            btn = QPushButton(f"{CLASS_KEYS[c]}  {c}")
            btn.setStyleSheet(f"QPushButton {{ border-left: 6px solid {CLASS_COLORS[c]}; text-align: left; "
                              f"padding-left: 8px; }}")
            btn.clicked.connect(lambda _=False, c=c: app.set_label(c))
            grid.addWidget(btn, i // 3, i % 3)
        rm = QPushButton("Del  remove")
        rm.clicked.connect(app.remove_label)
        grid.addWidget(rm, (len(CLASSES) + 2) // 3, 0)
        v.addLayout(grid)
        self.picked = QLabel("no particle picked")
        self.picked.setWordWrap(True)
        self.picked.setTextFormat(Qt.RichText)
        v.addWidget(self.picked)

        v.addWidget(section("The cell of the picked particle"))
        v.addWidget(hint("After Classify: verdict and colour of the reconstructed cell under the picked particle "
                         "(the same as the Cells batches)."))
        h = QHBoxLayout()
        for vd in ("one cell", "several cells", "not a cell"):
            b = QPushButton(vd)
            b.clicked.connect(lambda _=False, vd=vd: app.label_cell(verdict=vd))
            h.addWidget(b)
        v.addLayout(h)
        self.colour_row = QHBoxLayout()
        v.addLayout(self.colour_row)
        self.colour_boxes = {}
        self.cell_info = hint("")
        v.addWidget(self.cell_info)

        v.addWidget(section("Fix the segmentation"))
        v.addWidget(hint("Dashed orange lines are splits Detect proposes at signal dips (not applied). Accept one for "
                         "the picked particle, watershed it in two, draw your own cut (X, drag a line across, then "
                         "Apply), or merge it with a touching piece. Cuts hold in the reconstruction too; a piece cut "
                         "off outside the hole stays visible as a neighbour. Each edit re-detects the well."))
        g = QGridLayout()
        self.b_accept = QPushButton("Accept proposed split")
        self.b_accept.clicked.connect(app.accept_split)
        self.b_ws = QPushButton("Watershed picked")
        self.b_ws.setToolTip("split the picked particle in two: at its deepest dip between peaks, or, with one peak, "
                             "at the neck of its shape. Watershed a piece again to split further. Acts like a drawn "
                             "cut (every channel, both thresholds)")
        self.b_ws.clicked.connect(app.watershed)
        self.b_cut = QPushButton("Draw cut  (X)")
        self.b_cut.setCheckable(True)
        self.b_cut.toggled.connect(app.draw_cut)
        self.b_apply_cut = QPushButton("Apply cuts")
        self.b_apply_cut.clicked.connect(app.apply_cuts)
        self.b_merge = QPushButton("Merge with…")
        self.b_merge.setCheckable(True)
        self.b_merge.setToolTip("pick a particle, press this, then click the piece to join it with (same channel and "
                                "threshold; it must touch or come from the same object)")
        self.b_merge.toggled.connect(app.merge_mode)
        self.b_clear_edits = QPushButton("Clear edits in this well")
        self.b_clear_edits.clicked.connect(app.clear_edits)
        for i, b in enumerate([self.b_accept, self.b_ws, self.b_cut, self.b_apply_cut, self.b_merge, self.b_clear_edits]):
            g.addWidget(b, i // 3, i % 3)
        v.addLayout(g)
        v.addStretch(1)
        self.canvas_page = _scroll(self.canvas_page)

        # ---- Model tab ------------------------------------------------------------------------------------------
        body = QWidget()
        v = QVBoxLayout(body)
        v.addWidget(section("Labelling progress"))
        v.addWidget(hint("Training labels per class (target 30 each), test objects labelled blind, and judged cells "
                         "per colour. Rare classes (merged, debris) are usually the bottleneck: the particle map "
                         "(bottom) helps find them."))
        self.prog_box = QVBoxLayout()
        v.addLayout(self.prog_box)
        self.prog_bars = {}
        v.addWidget(section("Learning curve"))
        self.curve_lbl = hint("Drawn after each Retrain: the score on the test wells against the number of training "
                              "labels. Still rising = keep labelling; flat = label rarer classes or cells instead.")
        v.addWidget(self.curve_lbl)
        try:
            import pyqtgraph as pg
            self.curve = pg.PlotWidget(background=None)
            self.curve.setMinimumHeight(140)
            self.curve.setMaximumHeight(170)
            self.curve.setLabel("bottom", "training labels")
            self.curve.setLabel("left", "macro F1 (test wells)")
            self.curve.setMenuEnabled(False)
            v.addWidget(self.curve)
        except ImportError:
            self.curve = None

        v.addWidget(section("Compare models"))
        v.addWidget(hint("Scores saved models (and the Detect preview) on the blind labels of the open results' "
                         "test wells: the fair comparison. Open another dataset's results to compare across datasets."))
        h = QHBoxLayout()
        self.cmp_models = QComboBox()
        h.addWidget(self.cmp_models, 1)
        b = QPushButton("Add")
        b.clicked.connect(self.add_cmp)
        h.addWidget(b)
        b = QPushButton("Clear")
        b.clicked.connect(lambda: self.cmp_list.setText("preview"))
        h.addWidget(b)
        b = accent(QPushButton("Compare"))
        b.clicked.connect(app.compare_models)
        h.addWidget(b)
        v.addLayout(h)
        self.cmp_list = QLabel("preview")
        self.cmp_list.setProperty("role", "hint")
        v.addWidget(self.cmp_list)
        h = QHBoxLayout()
        h.addWidget(QLabel("no-cell check: wells where"))
        self.chk_col = QComboBox()
        self.chk_val = QComboBox()
        self.chk_col.currentTextChanged.connect(self._fill_chk_values)
        h.addWidget(self.chk_col, 1)
        h.addWidget(QLabel("="))
        h.addWidget(self.chk_val, 1)
        v.addLayout(h)
        v.addWidget(hint("Needs a plate map (Results tab). Everything detected in those wells (e.g. the Empty "
                         "condition) should be debris, plug or noise: the table adds the share each model calls "
                         "cell-like there, and the share of those wells that would get a cell. No labels needed."))
        self.cmp = FrameModel()
        cmv = FrameView(self.cmp)
        cmv.setMinimumHeight(120)
        v.addWidget(cmv)

        # advanced: the full training (compares all arms), label counts, CSV import
        adv = QWidget()
        a = QVBoxLayout(adv)
        a.setContentsMargins(0, 0, 0, 0)
        a.addWidget(hint("Retrain (Label tab) refits one arm quickly. This compares every arm by cross-validation "
                         "grouped by well and reports the test wells: use it for the final model. It uses every "
                         "labelled dataset of the results folders listed, plus all their unlabelled particles."))
        self.folders = QPlainTextEdit()
        self.folders.setPlaceholderText("results folders to train on, one per line (the open one is added)")
        self.folders.setMaximumHeight(60)
        a.addWidget(self.folders)
        h = QHBoxLayout()
        self.arms = {}
        from ..learn import ARMS
        for arm in ARMS:
            cb = QCheckBox(arm)
            cb.setChecked(True)
            self.arms[arm] = cb
            h.addWidget(cb)
        a.addLayout(h)
        h = QHBoxLayout()
        h.addWidget(QLabel("rounds"))
        self.rounds = QSpinBox()
        self.rounds.setRange(0, 10)
        self.rounds.setValue(3)
        h.addWidget(self.rounds)
        h.addWidget(QLabel("pseudo-label P ≥"))
        self.tau = QDoubleSpinBox()
        self.tau.setRange(0.5, 0.999)
        self.tau.setSingleStep(0.01)
        self.tau.setValue(0.9)
        h.addWidget(self.tau)
        h.addStretch(1)
        self.b_train = accent(QPushButton("Train (compare arms)"))
        self.b_train.clicked.connect(app.train)
        h.addWidget(self.b_train)
        a.addLayout(h)
        a.addWidget(QLabel("grouped cross-validation (by well)"))
        self.cv = FrameModel(fmt={})
        cvv = FrameView(self.cv)
        cvv.setMinimumHeight(120)
        a.addWidget(cvv)
        a.addWidget(QLabel("frozen test wells"))
        self.test = FrameModel()
        tv = FrameView(self.test)
        tv.setMinimumHeight(120)
        a.addWidget(tv)
        self.notes = hint("")
        a.addWidget(self.notes)
        h = QHBoxLayout()
        self.arm_pick = QComboBox()
        h.addWidget(QLabel("use arm"))
        h.addWidget(self.arm_pick, 1)
        self.b_save = accent(QPushButton("Save model + classify"))
        self.b_save.setEnabled(False)
        self.b_save.clicked.connect(app.save_model)
        h.addWidget(self.b_save)
        a.addLayout(h)
        a.addWidget(QLabel("labels in this library, per dataset"))
        self.counts = FrameModel()
        cv = FrameView(self.counts, stretch=True)
        cv.setMaximumHeight(110)
        a.addWidget(cv)
        b = QPushButton("Import a first labelling session (CSV)…")
        b.setToolTip("CSV with plate_id, well_id, channel (display name), x_px, y_px, particle_label; "
                     "imported for the dataset of the open results")
        b.clicked.connect(self.import_csv)
        a.addWidget(b)
        # the old "label next" list (the batches replace it); kept for open_queue_item
        self.queue = FrameModel()
        self.qv = FrameView(self.queue)
        self.qv.setVisible(False)
        a.addWidget(self.qv)
        v.addWidget(Collapsible("Advanced: full training, label counts, import", adv))
        v.addStretch(1)
        self.model_page = _scroll(body)

    def freeze_from_other(self):
        p, _ = QFileDialog.getOpenFileName(self, "Older label library", "", "SQLite (*.db)")
        if p:
            self.app.freeze_test_wells(other=p)

    def set_colour_boxes(self, chans: list):
        while self.colour_row.count():
            w = self.colour_row.takeAt(0).widget()
            if w is not None:
                w.setParent(None)
        self.colour_boxes = {}
        self.colour_row.addWidget(QLabel("positive for"))
        for ch in chans:
            cb = QCheckBox(self.app.cfg.display_name(ch))
            cb.setStyleSheet(f"QCheckBox {{ color: {self.app.cfg.color(ch)}; }}")
            self.colour_boxes[ch] = cb
            self.colour_row.addWidget(cb)
        b = QPushButton("set colour")
        b.clicked.connect(lambda: self.app.label_cell(colour="+".join(c for c, cb in self.colour_boxes.items()
                                                                      if cb.isChecked()) or "none"))
        self.colour_row.addWidget(b)
        self.colour_row.addStretch(1)

    def set_plate_map(self, m):
        self._map = m
        self.chk_col.blockSignals(True)
        self.chk_col.clear()
        self.chk_col.addItems(["(none)"] + [c for c in m.columns if c not in ("plate_id", "well_id")])
        self.chk_col.blockSignals(False)
        cols = [c for c in m.columns if c.lower() == "condition"]
        self.chk_col.setCurrentText(cols[0] if cols else "(none)")
        self._fill_chk_values(self.chk_col.currentText())

    def _fill_chk_values(self, col):
        self.chk_val.clear()
        m = getattr(self, "_map", None)
        if m is None or col not in m:
            return
        vals = sorted(m[col].dropna().astype(str).unique())
        self.chk_val.addItems(vals)
        for v in vals:
            if v.lower() == "empty":
                self.chk_val.setCurrentText(v)

    def check_wells(self):
        m, col, val = getattr(self, "_map", None), self.chk_col.currentText(), self.chk_val.currentText()
        if m is None or col not in m or not val:
            return None, "check"
        d = m[m[col].astype(str) == val]
        return set(zip(d.plate_id, d.well_id)), val

    def add_cmp(self):
        m = self.cmp_models.currentText()
        cur = [x for x in self.cmp_list.text().split(" · ") if x]
        if m and m not in cur:
            self.cmp_list.setText(" · ".join(cur + [m]))

    def compare_list(self) -> list:
        return [x for x in self.cmp_list.text().split(" · ") if x]

    def show_progress(self, pr: dict):
        """Per class (target), test wells, cells per colour: progress bars (counts, not plots)."""
        items = [(f"{c}", n, t) for c, (n, t) in pr.get("classes", {}).items()]
        t = pr.get("test", {})
        if t:
            items.append(("test objects (blind)", t.get("labelled", 0), max(t.get("objects", 0), 1)))
        for col, (n, tg) in pr.get("cells", {}).items():
            items.append((f"cells: {'+'.join(self.app.cfg.display_name(x) for x in col.split('+'))}", n, tg))
        for name, n, tg in items:
            bar = self.prog_bars.get(name)
            if bar is None:
                row = QHBoxLayout()
                lab = QLabel(name)
                lab.setMinimumWidth(150)
                row.addWidget(lab)
                bar = QProgressBar()
                bar.setTextVisible(True)
                row.addWidget(bar, 1)
                self.prog_box.addLayout(row)
                self.prog_bars[name] = bar
            bar.setMaximum(max(int(tg), 1))
            bar.setValue(min(int(n), int(tg)))
            bar.setFormat(f"{n} / {tg}")

    def show_curve(self, curve: pd.DataFrame | None, text: str = ""):
        if self.curve is None:
            return
        self.curve.clear()
        if curve is not None and len(curve):
            import pyqtgraph as pg
            self.curve.plot(curve.n.to_numpy(float), curve.macro_f1.to_numpy(float), pen=pg.mkPen("#4fc3f7", width=2),
                            symbol="o", symbolSize=6, symbolBrush="#4fc3f7")
        self.curve_lbl.setText(text or "Learning curve after training: score on the test wells against the number "
                                       "of training labels.")

    def browse_store(self):
        p, _ = QFileDialog.getSaveFileName(self, "Label library", self.store_path.text() or "ci_labels.db",
                                           "SQLite (*.db)", options=QFileDialog.DontConfirmOverwrite)
        if p:
            self.store_path.setText(p)
            self.app.set_store(p)

    def import_csv(self):
        p, _ = QFileDialog.getOpenFileName(self, "Labels CSV", "", "CSV (*.csv)")
        if p:
            self.app.import_labels(p)

    def show_result(self, res):
        cols = ["arm", "macro_f1", "bal_acc", "auc_cell", "recall_cell", "recall_neighbour", "recall_merged",
                "recall_debris", "recall_noise", "coverage_unlabelled", "n_pseudo", "n"]
        self.cv.set(res.cv[[c for c in cols if c in res.cv]].round(3))
        self.test.set(res.test[[c for c in cols if c in res.test]].round(3))
        self.notes.setText(" · ".join(res.notes) or "")
        self.arm_pick.clear()
        self.arm_pick.addItems(list(res.cv.arm))
        self.arm_pick.setCurrentText(res.model.spec["arm"])
        self.b_save.setEnabled(True)
        self.queue.set(res.queue.head(100).round(3) if len(res.queue) else pd.DataFrame())


class ResultsTab(QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app
        outer = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        v = QVBoxLayout(body)
        v.addWidget(section("Wells"))
        self.wells_box = QVBoxLayout()
        v.addLayout(self.wells_box)
        v.addWidget(section("Detect preview (no model)"))
        v.addWidget(hint("A first look straight after Detect: provisional particle classes (inside / neighbour / "
                         "debris / noise) and per well the status and the dominant channel. On ROMK it agrees with "
                         "the hand labels for 85 % of particles and with the classified well status for 82 % of "
                         "wells."))
        g = QGridLayout()
        self.pcards = {}
        for i, (k, t) in enumerate([("one object", "one object"), ("several objects", "several objects"),
                                    ("neighbours only", "neighbours only"), ("no particles", "no particles")]):
            c, val = card(t)
            self.pcards[k] = val
            g.addWidget(c, 0, i)
        v.addLayout(g)
        self.prev_plate = FrameModel()
        ppv = FrameView(self.prev_plate)
        ppv.setMinimumHeight(130)
        v.addWidget(QLabel("dominant channel per plate (wells with something inside the hole)"))
        v.addWidget(ppv)
        self.prev_cls = FrameModel()
        pcv = FrameView(self.prev_cls, stretch=True)
        pcv.setMaximumHeight(130)
        v.addWidget(QLabel("particles in the holes per channel, by provisional class"))
        v.addWidget(pcv)
        v.addWidget(section("After Classify"))
        g = QGridLayout()
        self.cards = {}
        for i, (k, t) in enumerate([("one", "one cell"), ("several", "several cells"), ("none", "no cell"),
                                    ("uncertain", "uncertain call")]):
            c, val = card(t)
            self.cards[k] = val
            g.addWidget(c, 0, i)
        v.addLayout(g)
        v.addWidget(section("Calls per plate"))
        self.per_plate = FrameModel()
        pv = FrameView(self.per_plate)
        pv.setMinimumHeight(140)
        v.addWidget(pv)
        v.addWidget(section("Identity channels"))
        v.addWidget(hint("Per channel: a two-component mixture over all cells found in the holes of the run. "
                         "Ashman's D < 2 = positives and negatives do not separate well."))
        self.mix = FrameModel()
        mv = FrameView(self.mix, stretch=True)
        mv.setMaximumHeight(130)
        v.addWidget(mv)
        self.advice = QLabel("")
        self.advice.setWordWrap(True)
        self.advice.setTextFormat(Qt.RichText)
        v.addWidget(self.advice)
        v.addWidget(section("Plate map (evaluation and display only)"))
        v.addWidget(hint("Never used for a call. Any CSV with plate_id, well_id and columns of your choice."))
        h = QHBoxLayout()
        b = QPushButton("Load plate map…")
        b.clicked.connect(self.load_map)
        h.addWidget(b)
        self.col = QComboBox()
        self.col.currentTextChanged.connect(self.crosstab)
        h.addWidget(self.col, 1)
        v.addLayout(h)
        self.ct = FrameModel()
        cv = FrameView(self.ct)
        cv.setMinimumHeight(160)
        v.addWidget(cv)
        h = QHBoxLayout()
        o = QPushButton("Open results folder")
        o.clicked.connect(app.open_folder)
        h.addWidget(o)
        v.addLayout(h)
        v.addStretch(1)
        scroll.setWidget(body)
        outer.addWidget(scroll)
        self.map = None

    def set_well_list(self, w):
        w.setMinimumHeight(260)
        self.wells_box.addWidget(w)

    def update_preview(self, res):
        disp = self.app.cfg.display_name
        if res is None or "preview_status" not in res.wells:
            for c in self.pcards.values():
                c.setText("–")
            self.prev_plate.set(pd.DataFrame())
            self.prev_cls.set(pd.DataFrame())
            return
        W = res.wells
        for k, c in self.pcards.items():
            c.setText(str(int((W.preview_status == k).sum())))
        d = W[W.preview_dominant.fillna("") != ""]
        dom = d.preview_dominant.map(lambda s: "+".join(disp(x) for x in str(s).split("+")))
        self.prev_plate.set(pd.crosstab(d.plate_id, dom).reset_index() if len(d) else pd.DataFrame())
        P = res.particles
        P = P[P.pool]
        if "dclass" in P and len(P):
            ct = pd.crosstab(P.ch.map(disp), P.dclass).reset_index().rename(columns={"ch": "channel"})
            self.prev_cls.set(ct)

    def update(self, res):
        self.update_preview(res)
        if res is None or res.calls is None:
            for c in self.cards.values():
                c.setText("–")
            self.per_plate.set(pd.DataFrame())
            self.mix.set(pd.DataFrame())
            self.advice.setText("")
            return
        c = res.calls
        self.cards["one"].setText(str(int((c.status == "one cell").sum())))
        self.cards["several"].setText(str(int((c.status == "several cells").sum())))
        self.cards["none"].setText(str(int((c.status == "no cell").sum())))
        self.cards["uncertain"].setText(str(int((c["call"] == "uncertain").sum())))
        disp = self.app.cfg.display_name
        lab = c["call"].map(lambda s: "+".join(disp(x) for x in str(s).split("+")) if s not in
                            ("none", "uncertain", "no cell") else s)
        pp = pd.crosstab(c.plate_id, lab).reset_index()
        self.per_plate.set(pp)
        info = json.loads((res.dir / "identity.json").read_text()) if (res.dir / "identity.json").exists() else {}
        rows = []
        for ch, d in info.items():
            rows.append(dict(channel=disp(ch), ashman_D=d.get("ashman_d"), n_cells=d.get("n"),
                             flag="weak separation" if d.get("weak_separation") else (d.get("note") or ""),
                             means=", ".join(f"{m:.2f}" for m in d.get("means", []))))
        self.mix.set(pd.DataFrame(rows))
        from ..identify import advice
        adv = advice(info, res.cells, res.calls, self.app.cfg)
        self.advice.setText("<br>".join(f"<b>{disp(ch) if ch != '_all' else 'all channels'}</b>: " + "; ".join(t)
                                        for ch, t in adv.items()))
        self.crosstab()

    def load_map(self):
        p, _ = QFileDialog.getOpenFileName(self, "Plate map", "", "CSV (*.csv)")
        if not p:
            return
        m = pd.read_csv(p, dtype={"plate_id": str, "well_id": str})
        m["well_id"] = m.well_id.map(lambda w: w[0] + f"{int(w[1:]):02d}")
        self.map = m
        self.col.clear()
        self.col.addItems([c for c in m.columns if c not in ("plate_id", "well_id")])
        self.app.set_plate_map(m)

    def crosstab(self, *_):
        res = self.app.res
        col = self.col.currentText()
        if self.map is None or res is None or res.calls is None or not col:
            self.ct.set(pd.DataFrame())
            return
        d = res.calls.merge(self.map, on=["plate_id", "well_id"], how="inner")
        disp = self.app.cfg.display_name
        d["call_"] = d["call"].map(lambda s: "+".join(disp(x) for x in str(s).split("+")) if s not in
                                   ("none", "uncertain", "no cell") else s)
        self.ct.set(pd.crosstab(d[col].astype(str), d.call_, margins=True).reset_index())


class LabelTab(QWidget):
    """Setup (library, test wells, blind) on top; sub-tabs Particles / Cells (batch labellers) / In the canvas; the
    retrain bar at the bottom."""

    def __init__(self, app, train: TrainTab):
        super().__init__()
        from .queue import LabelQueue
        v = QVBoxLayout(self)
        v.setContentsMargins(2, 2, 2, 2)
        self.setup = Collapsible("Setup: label library and test wells (once per dataset)", train.setup_page, True)
        v.addWidget(self.setup)
        self.sub = QTabWidget()
        self.particles = LabelQueue(app, "particle")
        self.cells = LabelQueue(app, "cell")
        self.sub.addTab(self.particles, "Bulk: particles")
        self.sub.addTab(self.cells, "Bulk: cells")
        self.sub.addTab(train.canvas_page, "Detailed view")
        v.addWidget(self.sub, 1)
        bar = QFrame()
        bar.setProperty("role", "card")
        h = QHBoxLayout(bar)
        h.setContentsMargins(8, 4, 8, 4)
        self.retrain_info = QLabel("")
        self.retrain_info.setWordWrap(True)
        self.retrain_info.setTextFormat(Qt.RichText)
        h.addWidget(self.retrain_info, 1)
        self.b_retrain = accent(QPushButton("Retrain + classify"))
        self.b_retrain.setToolTip("train one model on all labels so far (outside the test wells), save it, classify "
                                  "these results with it; the batches then suggest the model's classes (~1 min)")
        self.b_retrain.clicked.connect(app.retrain_and_classify)
        h.addWidget(self.b_retrain)
        v.addWidget(bar)


def _add_map(self, w):
    self.sub.addTab(w, "Particle map")


LabelTab.add_map = _add_map


class Panel(QTabWidget):
    def __init__(self, app):
        super().__init__()
        self.setMinimumWidth(520)
        self.run = RunTab(app)
        self.settings = SettingsTab(app)
        self.train = TrainTab(app)
        self.label = LabelTab(app, self.train)
        self.results = ResultsTab(app)
        from .diag_view import DiagnosticsView
        self.diag = DiagnosticsView(app)
        self.addTab(self.run, "1 Run")
        self.addTab(self.label, "2 Label")
        self.addTab(self.train.model_page, "3 Model")
        self.addTab(self.results, "Results")
        self.addTab(self.diag, "Diagnostics")
        self.addTab(self.settings, "Settings")
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)


class PanelHost(QWidget):
    """The right dock: the tabs, with a progress bar on top while a job runs."""

    def __init__(self, panel: Panel):
        super().__init__()
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        self.bar = QProgressBar()
        self.bar.setTextVisible(True)
        self.bar.setFixedHeight(16)
        self.bar.setVisible(False)
        v.addWidget(self.bar)
        v.addWidget(panel, 1)
        self._what = ""

    def busy(self, on: bool, what: str = ""):
        """Running: an indeterminate bar with the job's name until real progress (i of n) arrives."""
        self._what = what or self._what
        self.bar.setVisible(on)
        if on:
            self.bar.setRange(0, 0)
            self.bar.setFormat(f"{self._what}…")

    def progress(self, i, n, msg=""):
        if not self.bar.isVisible():
            return
        if n:
            self.bar.setRange(0, n)
            self.bar.setValue(i)
            self.bar.setFormat(f"{self._what}: {i} / {n}")
        elif msg:
            self.bar.setRange(0, 0)
            self.bar.setFormat(f"{self._what}: {msg[:60]}")
