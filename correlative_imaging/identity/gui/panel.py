"""The right-hand dock: Run, Settings, Train, Results. Each tab is a plain QWidget driven by the app controller."""
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

HELP = {
    "channels": "fluorescence channels to use (raw names, e.g. 488nm); empty = all channels of the run",
    "markers": "channels that are measured but never count as identity (e.g. a channel on every cell)",
    "work_R": "analysis window around the hole, in hole radii",
    "context_R": "disc for context features (neighbour density, brightness ranks), in hole radii; keep it the same "
                 "across datasets so features stay comparable",
    "min_object_px": "objects up to this size (px) are removed before labelling, as in the pipeline",
    "min_inside_um2": "a particle needs at least this much area inside the hole",
    "field_min_um2": "whole-image objects at least this big describe a typical unpatched cell",
    "smooth_px": "light smoothing (px) for peaks and growth; measurements use the unsmoothed image",
    "model": "trained particle classifier (.joblib in identity/models or a full path)",
    "p_seed": "P(cell) + P(merged) a particle needs to seed a cell",
    "split_dips": "split bright areas where the signal dips between two peaks (off = connected areas stay whole)",
    "peak_h": "a peak must rise this many hole-noise SDs above its surroundings to count",
    "dip_ratio": "saddle / lower peak below this = two objects",
    "cell_gap_um": "seeds closer than this (um) join into one cell (the same cell in two channels)",
    "grow_floor": "grown pixels must be this many hole-noise SDs above the hole background",
    "grow_frac": "grown pixels must reach this fraction of the seed's median signal",
    "grow_margin_um": "how far (um) a cell may grow beyond the hole",
    "grow_max_x_hole": "area cap in hole areas; the relative cut is raised until the cell fits",
    "p_call": "P(best channel combination) needed for a call, otherwise 'uncertain'",
    "min_mixture_sep": "per-channel mixture separation (Ashman's D) below which the channel is flagged",
    "path_map": 'stored path prefix -> local prefix, JSON, e.g. {"Z:\\\\": "/mnt/z/"}',
}
GROUPS = {
    "Channels": ["channels", "markers"],
    "Detection": ["work_R", "context_R", "min_object_px", "min_inside_um2", "field_min_um2", "smooth_px"],
    "Classifier": ["model", "p_seed"],
    "Reconstruction": ["split_dips", "peak_h", "dip_ratio", "cell_gap_um", "grow_floor", "grow_frac",
                       "grow_margin_um", "grow_max_x_hole"],
    "Identity call": ["p_call", "min_mixture_sep"],
    "Local": ["path_map"],
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
        v = QVBoxLayout(self)
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
        h = QHBoxLayout()
        self.b_detect = QPushButton("Detect")
        self.b_detect.setToolTip("particles, field reference and features (no model needed)")
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
        v.addWidget(self.log, 1)

    def browse_root(self):
        d = QFileDialog.getExistingDirectory(self, "CorrelativeImaging output folder", self.root.text())
        if d:
            self.root.setText(d)
            self.find_runs()

    def browse_out(self):
        d = QFileDialog.getExistingDirectory(self, "Results folder", self.out.text())
        if d:
            self.out.setText(d)

    def find_runs(self):
        from .._discovery import discover_runs
        self.run.clear()
        root = self.root.text().strip()
        if not root:
            return
        runs = discover_runs(root)
        self.run.addItems([r.run_tag for r in runs])
        self.app.say(f"{len(runs)} run(s) under {root}")
        if runs:
            self.run.setCurrentIndex(len(runs) - 1)

    def default_out(self, tag):
        if tag and self.root.text().strip():
            self.out.setText(str(Path(self.root.text().strip()) / f"identity_{tag}"))

    def load_preset(self, name):
        cfg = Config() if name == "(neutral)" else load_config(name)
        cfg.model = self.app.cfg.model or cfg.model
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
    def __init__(self, app):
        super().__init__()
        self.app = app
        outer = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        v = QVBoxLayout(body)
        self.widgets = {}
        ftypes = {f.name: f.type for f in fields(Config)}
        for group, keys in GROUPS.items():
            box = QGroupBox(group)
            form = QFormLayout(box)
            for k in keys:
                w = self._widget(k, ftypes[k])
                w.setToolTip(HELP.get(k, ""))
                lab = QLabel(k.replace("_", " "))
                lab.setToolTip(HELP.get(k, ""))
                form.addRow(lab, w)
                self.widgets[k] = w
            v.addWidget(box)
        box = QGroupBox("Channel display (names and colours, display only)")
        self.disp = QFormLayout(box)
        v.addWidget(box)
        v.addStretch(1)
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)
        h = QHBoxLayout()
        b = accent(QPushButton("Apply"))
        b.clicked.connect(self.apply)
        s = QPushButton("Save as preset…")
        s.clicked.connect(self.save)
        h.addWidget(b)
        h.addWidget(s)
        outer.addLayout(h)

    def _widget(self, k, t):
        default = getattr(Config(), k)
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
        if k == "model":
            w = QComboBox()
            w.setEditable(True)
            return w
        return QLineEdit()

    def set_config(self, cfg: Config, channels: list | None = None):
        for k, w in self.widgets.items():
            v = getattr(cfg, k)
            if isinstance(w, QCheckBox):
                w.setChecked(bool(v))
            elif isinstance(w, (QSpinBox, QDoubleSpinBox)):
                w.setValue(v)
            elif isinstance(w, QComboBox):
                w.clear()
                w.addItems(sorted(p.name for p in MODELS.glob("*.joblib")))
                w.setCurrentText(v or "")
            elif isinstance(v, list):
                w.setText(", ".join(v))
            elif isinstance(v, dict):
                w.setText(json.dumps(v) if v else "")
            else:
                w.setText(str(v))
        while self.disp.rowCount():
            self.disp.removeRow(0)
        self._disp = {}
        for ch in channels or cfg.channels or list(cfg.display):
            name = QLineEdit(cfg.display_name(ch))
            col = QLineEdit(cfg.color(ch))
            col.setMaximumWidth(90)
            row = QWidget()
            h = QHBoxLayout(row)
            h.setContentsMargins(0, 0, 0, 0)
            h.addWidget(name, 1)
            h.addWidget(col)
            self.disp.addRow(ch, row)
            self._disp[ch] = (name, col)

    def read(self) -> Config:
        cfg = Config()
        for k, w in self.widgets.items():
            d = getattr(cfg, k)
            if isinstance(w, QCheckBox):
                v = w.isChecked()
            elif isinstance(w, (QSpinBox, QDoubleSpinBox)):
                v = type(d)(w.value())
            elif isinstance(w, QComboBox):
                v = w.currentText().strip()
            elif isinstance(d, list):
                v = [s.strip() for s in w.text().split(",") if s.strip()]
            elif isinstance(d, dict):
                v = json.loads(w.text()) if w.text().strip() else {}
            else:
                v = w.text()
            setattr(cfg, k, v)
        cfg.display = {ch: {"name": n.text().strip() or ch, "color": c.text().strip()}
                       for ch, (n, c) in getattr(self, "_disp", {}).items()}
        return cfg

    def apply(self):
        try:
            self.app.set_config(self.read(), from_form=True)
        except Exception as e:
            self.app.say(f"settings not applied: {e}")

    def save(self):
        from ..config import PRESETS
        p, _ = QFileDialog.getSaveFileName(self, "Save preset", str(PRESETS / "my_dataset.json"), "JSON (*.json)")
        if p:
            cfg = self.read()
            d = json.loads(cfg.to_json())
            d.pop("path_map", None)
            Path(p).write_text(json.dumps(d, indent=2))
            self.app.say(f"preset saved: {p}")


class TrainTab(QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app
        outer = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        v = QVBoxLayout(body)

        v.addWidget(section("Label library"))
        self.store_path = QLineEdit()
        self.store_path.editingFinished.connect(lambda: app.set_store(self.store_path.text()))
        v.addWidget(path_row(self.store_path, self.browse_store))
        self.counts = FrameModel()
        cv = FrameView(self.counts, stretch=True)
        cv.setMaximumHeight(110)
        v.addWidget(cv)
        b = QPushButton("Import a first labelling session (CSV)…")
        b.setToolTip("CSV with plate_id, well_id, channel (display name), x_px, y_px, particle_label; "
                     "imported for the dataset of the open results")
        b.clicked.connect(self.import_csv)
        v.addWidget(b)

        v.addWidget(section("Label particles"))
        self.mode = QPushButton("Label mode  (L)")
        self.mode.setCheckable(True)
        self.mode.toggled.connect(app.set_label_mode)
        v.addWidget(self.mode)
        v.addWidget(hint("In label mode, click a particle in the canvas (several overlap: pick from the menu), then "
                         "press 1–5 or a button. Del removes the label. Labels are stored by position, so they "
                         "survive reruns with other settings."))
        grid = QGridLayout()
        for i, c in enumerate(CLASSES):
            btn = QPushButton(f"{CLASS_KEYS[c]}  {c}")
            btn.setStyleSheet(f"QPushButton {{ border-left: 6px solid {CLASS_COLORS[c]}; text-align: left; "
                              f"padding-left: 8px; }}")
            btn.clicked.connect(lambda _=False, c=c: app.set_label(c))
            grid.addWidget(btn, i // 3, i % 3)
        rm = QPushButton("Del  remove")
        rm.clicked.connect(app.remove_label)
        grid.addWidget(rm, 1, 2)
        v.addLayout(grid)
        self.picked = QLabel("no particle picked")
        self.picked.setWordWrap(True)
        self.picked.setTextFormat(Qt.RichText)
        v.addWidget(self.picked)

        v.addWidget(section("Train the classifier"))
        v.addWidget(hint("Uses every labelled dataset of the results folders listed below, plus all their unlabelled "
                         "particles (semi-supervised arms). A frozen set of test wells per dataset is never trained "
                         "on; choose by the grouped cross-validation, read the test row as the honest check."))
        self.folders = QPlainTextEdit()
        self.folders.setPlaceholderText("results folders to train on, one per line (the open one is added)")
        self.folders.setMaximumHeight(60)
        v.addWidget(self.folders)
        h = QHBoxLayout()
        self.arms = {}
        from ..learn import ARMS
        for a in ARMS:
            cb = QCheckBox(a)
            cb.setChecked(True)
            self.arms[a] = cb
            h.addWidget(cb)
        v.addLayout(h)
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
        self.b_train = accent(QPushButton("Train"))
        self.b_train.clicked.connect(app.train)
        h.addWidget(self.b_train)
        v.addLayout(h)
        v.addWidget(QLabel("grouped cross-validation (by well)"))
        self.cv = FrameModel(fmt={})
        cvv = FrameView(self.cv)
        cvv.setMinimumHeight(130)
        v.addWidget(cvv)
        v.addWidget(QLabel("frozen test wells"))
        self.test = FrameModel()
        tv = FrameView(self.test)
        tv.setMinimumHeight(130)
        v.addWidget(tv)
        self.notes = hint("")
        v.addWidget(self.notes)
        h = QHBoxLayout()
        self.arm_pick = QComboBox()
        h.addWidget(QLabel("use arm"))
        h.addWidget(self.arm_pick, 1)
        self.b_save = accent(QPushButton("Save model + classify"))
        self.b_save.setEnabled(False)
        self.b_save.clicked.connect(app.save_model)
        h.addWidget(self.b_save)
        v.addLayout(h)

        v.addWidget(section("Label next"))
        v.addWidget(hint("Unlabelled particles the model is least sure about (one per well, test wells excluded). "
                         "Double-click to open."))
        self.queue = FrameModel()
        self.qv = FrameView(self.queue)
        self.qv.setMinimumHeight(160)
        self.qv.doubleClicked.connect(lambda _: app.open_queue_item(self.qv.current_row()))
        v.addWidget(self.qv)
        v.addStretch(1)
        scroll.setWidget(body)
        outer.addWidget(scroll)

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

    def update(self, res):
        if res is None or res.calls is None:
            for c in self.cards.values():
                c.setText("–")
            self.per_plate.set(pd.DataFrame())
            self.mix.set(pd.DataFrame())
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


class Panel(QTabWidget):
    def __init__(self, app):
        super().__init__()
        self.setMinimumWidth(460)
        self.run = RunTab(app)
        self.settings = SettingsTab(app)
        self.train = TrainTab(app)
        self.results = ResultsTab(app)
        self.addTab(self.run, "Run")
        self.addTab(self.settings, "Settings")
        self.addTab(self.train, "Train")
        self.addTab(self.results, "Results")
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
