"""GUI for the downstream analyzer: run a finished CorrelativeImaging run through the whole routine yourself.

Tabs
  Run       pick the output folder + run, an optional plate map (experiment table), start; progress + log
  Settings  every setting of config.Config, grouped, with presets (JSON) per dataset
  Review    the particle viewer embedded (Qt WebEngine): verdicts per well and particle labels are saved to
            <results>/review.db; label mode: click a particle, pick its label
  Results   summary (and evaluation against the plate map, if given), export calls.csv, refit the particle models
            from your labels

Launch:  python -m correlative_imaging.downstream gui
Only this module imports Qt; everything it runs is the Qt-free code of this folder.
"""
from __future__ import annotations

import dataclasses
import json
import sys
import traceback
from pathlib import Path

import pandas as pd
from qtpy.QtCore import QObject, QSettings, Qt, QThread, QUrl, Signal, Slot
from qtpy.QtGui import QFont
from qtpy.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
                            QGroupBox, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QListWidget, QMainWindow,
                            QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QSpinBox,
                            QTabWidget, QVBoxLayout, QWidget)

try:                                     # must be imported before the QApplication exists
    from qtpy.QtWebChannel import QWebChannel
    from qtpy.QtWebEngineCore import QWebEngineSettings
    from qtpy.QtWebEngineWidgets import QWebEngineView
    HAVE_WEB = True
except Exception:                        # pragma: no cover - depends on the install
    HAVE_WEB = False

from .config import Config, load_config

PRESETS = Path(__file__).parent / "presets"
MONO = QFont("Consolas")
MONO.setStyleHint(QFont.StyleHint.Monospace)

# settings shown per group, with a short explanation (tooltip)
GROUPS = {
    "Channels": {
        "channels": "raw channel name in the DB -> display name, in the order of the crop tif",
        "reporters": "channels that carry identity (the colour call is made between these)",
        "markers": "measured but never identity evidence (e.g. ROMK, on almost every cell)",
        "colors": "display colour per channel",
        "bleed": "bleed-through pairs: a particle of 'into' is dropped when into < 10^max_log_ratio x 'from'",
    },
    "Particles": {
        "min_object_px": "remove_small_objects before labelling (as the pipeline)",
        "min_inside_um2": "a piece inside the hole must be at least this big",
    },
    "Roles (in this order: neighbour, bleed, signal)": {
        "neighbour_share_max": "neighbour: less than this share of the particle inside the hole ...",
        "neighbour_fill_max": "... AND it covers less than this fraction of the hole",
        "p_candidate": "P(cell) for a candidate",
        "p_weak": "P(cell) for a weak particle",
    },
    "Cells and tier": {
        "cell_gap_um": "particles closer than this (or overlapping) form one cell",
        "tier_high": "tier high: P(cell) >=, P(neighbour) <=, must overlap a stringent candidate",
        "tier_middle": "tier middle: P(cell) >=, P(neighbour) <=",
        "call_fold": "simple call: top reporter must be >= this x the next one",
        "call_mode": "exclusive (one colour per cell) | multilabel (not implemented yet)",
    },
    "Refinement (grow the patched cell)": {
        "refine_frac": "a grown pixel must reach this fraction of its seed's median signal",
        "refine_floor_snr": "... and this SNR above the hole's own background",
        "refine_margin_um": "how far the patched cell may grow beyond the hole",
        "refine_max_area_x_hole": "largest body, in hole areas (the cut is raised until it fits)",
        "refine_smooth_px": "Gaussian smoothing before growing (pixels)",
        "body_channels": "channels that define the cell body (null = reporters)",
    },
    "Calibrated colour call": {
        "calib_min_field_um2": "field cells at least this big are the per plate x channel reference",
        "calib_p_call": "P(colour) needed for a call; below = uncertain",
        "calib_bleed_quantile": "bleed coefficient = this quantile over bright field cells",
        "calib_eps": "floor in field-cell units for the colour ratio; both below = dim",
    },
    "Models and paths": {
        "models": "particle models (JSON); relative paths are next to the code",
        "path_map": "Windows drive -> local mount, only used when a stored path does not exist",
    },
}


# ---------------------------------------------------------------------------------------------- workers
class RunWorker(QThread):
    progress = Signal(int, int, str)
    done = Signal(str)
    failed = Signal(str)

    def __init__(self, root, run_tag, out, cfg, col_data, wells, workers, title):
        super().__init__()
        self.a = (root, run_tag, out, cfg, col_data, wells, workers, title)

    def run(self):
        root, tag, out, cfg, cd, wells, workers, title = self.a
        try:
            from .pipeline import run_downstream
            from .sources import RunSource
            self.progress.emit(0, 0, f"reading run {tag} ...")
            src = RunSource(root, tag, cfg)
            self.progress.emit(0, len(src.wells()), f"{len(src.wells())} wells in {len(src.plates())} plates")
            run_downstream(src, cfg, out, wells=wells, workers=workers, col_data=cd, title=title,
                           progress=lambda i, n, m="": self.progress.emit(i, n, m))
            self.done.emit(str(out))
        except Exception:
            self.failed.emit(traceback.format_exc())


class DiscoverWorker(QThread):
    found = Signal(list)

    def __init__(self, root):
        super().__init__()
        self.root = root

    def run(self):
        from ._discovery import discover_runs
        try:
            self.found.emit([(g.run_tag, g.n_plates) for g in discover_runs(self.root)])
        except Exception:
            self.found.emit([])


# ---------------------------------------------------------------------------------------------- tabs
class SettingsTab(QWidget):
    def __init__(self):
        super().__init__()
        self.widgets = {}
        self.types = {f.name: f.type for f in dataclasses.fields(Config)}
        lay = QVBoxLayout(self)
        bar = QHBoxLayout()
        self.preset = QLabel("preset: ROMK defaults")
        for txt, fn in (("Load preset ...", self.load_preset), ("Save preset as ...", self.save_preset),
                        ("Reset to ROMK defaults", lambda: self.set_config(Config(), "ROMK defaults"))):
            b = QPushButton(txt)
            b.clicked.connect(fn)
            bar.addWidget(b)
        bar.addStretch()
        bar.addWidget(self.preset)
        lay.addLayout(bar)
        inner = QWidget()
        il = QVBoxLayout(inner)
        for g, keys in GROUPS.items():
            box = QGroupBox(g)
            fl = QFormLayout(box)
            for k, tip in keys.items():
                w = self._widget(k)
                w.setToolTip(tip)
                lab = QLabel(k)
                lab.setToolTip(tip)
                fl.addRow(lab, w)
                self.widgets[k] = w
            il.addWidget(box)
        il.addStretch()
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(inner)
        lay.addWidget(sc)
        self.set_config(Config(), "ROMK defaults")

    def _widget(self, k):
        default = getattr(Config(), k)
        if isinstance(default, bool):
            return QCheckBox()
        if isinstance(default, float):
            w = QDoubleSpinBox()
            w.setDecimals(3)
            w.setRange(-1e6, 1e6)
            w.setSingleStep(0.05)
            return w
        if isinstance(default, int):
            w = QSpinBox()
            w.setRange(0, 10 ** 6)
            return w
        if k == "models":
            w = QWidget()
            h = QHBoxLayout(w)
            h.setContentsMargins(0, 0, 0, 0)
            e = QLineEdit()
            b = QPushButton("...")
            b.clicked.connect(lambda: self._pick_model(e))
            h.addWidget(e)
            h.addWidget(b)
            w.edit = e
            return w
        if k == "call_mode":
            w = QComboBox()
            w.addItems(["exclusive", "multilabel"])
            return w
        return QLineEdit()                                   # dict / list / None -> JSON

    def _pick_model(self, e):
        f, _ = QFileDialog.getOpenFileName(self, "Particle models", str(Path(__file__).parent / "models"), "JSON (*.json)")
        if f:
            e.setText(f)

    def set_config(self, cfg: Config, name: str):
        for k, w in self.widgets.items():
            v = getattr(cfg, k)
            if isinstance(w, QDoubleSpinBox):
                w.setValue(float(v))
            elif isinstance(w, QSpinBox):
                w.setValue(int(v))
            elif isinstance(w, QComboBox):
                w.setCurrentText(str(v))
            elif hasattr(w, "edit"):
                w.edit.setText(str(v))
            else:
                w.setText(json.dumps(v))
        self.preset.setText(f"preset: {name}")

    def config(self) -> Config:
        cfg = Config()
        for k, w in self.widgets.items():
            if isinstance(w, (QDoubleSpinBox, QSpinBox)):
                setattr(cfg, k, w.value())
            elif isinstance(w, QComboBox):
                setattr(cfg, k, w.currentText())
            elif hasattr(w, "edit"):
                setattr(cfg, k, w.edit.text().strip())
            else:
                try:
                    setattr(cfg, k, json.loads(w.text()))
                except json.JSONDecodeError as e:
                    raise ValueError(f"setting '{k}' is not valid JSON: {e}") from None
        return cfg

    def load_preset(self):
        f, _ = QFileDialog.getOpenFileName(self, "Load preset", str(PRESETS), "JSON (*.json)")
        if f:
            self.set_config(load_config(f), Path(f).stem)

    def save_preset(self):
        try:
            cfg = self.config()
        except ValueError as e:
            QMessageBox.warning(self, "Settings", str(e))
            return
        PRESETS.mkdir(exist_ok=True)
        f, _ = QFileDialog.getSaveFileName(self, "Save preset", str(PRESETS / "my_dataset.json"), "JSON (*.json)")
        if f:
            Path(f).write_text(cfg.to_json())
            self.preset.setText(f"preset: {Path(f).stem}")


class RunTab(QWidget):
    finished = Signal(str)

    def __init__(self, settings: SettingsTab, qs: QSettings):
        super().__init__()
        self.settings, self.qs, self.worker = settings, qs, None
        lay = QVBoxLayout(self)
        form = QFormLayout()
        self.root = QLineEdit(qs.value("root", ""))
        form.addRow("CorrelativeImaging output folder", self._row(self.root, self._pick_root, "Find runs"))
        self.run = QComboBox()
        self.run.currentTextChanged.connect(self._default_out)
        form.addRow("Run", self.run)
        self.coldata = QLineEdit(qs.value("coldata", ""))
        self.coldata.setPlaceholderText("optional CSV: plate_id, well_id + any columns (e.g. Condition, Induction)")
        form.addRow("Plate map (experiment table)", self._row(self.coldata, self._pick_coldata))
        self.out = QLineEdit()
        form.addRow("Results folder", self._row(self.out, self._pick_out))
        self.wells = QLineEdit()
        self.wells.setPlaceholderText("empty = all wells, or e.g. 18T39412_N11 18T39383_B03")
        form.addRow("Wells", self.wells)
        self.title = QLineEdit("Patched-cell identity")
        form.addRow("Title (viewer)", self.title)
        self.nwork = QSpinBox()
        self.nwork.setRange(1, 32)
        self.nwork.setValue(8)
        form.addRow("Parallel wells", self.nwork)
        lay.addLayout(form)
        note = QLabel("The plate map is only used to display and evaluate; nothing that decides a call reads it. "
                      "Settings come from the Settings tab.")
        note.setWordWrap(True)
        note.setStyleSheet("color:#666")
        lay.addWidget(note)
        h = QHBoxLayout()
        self.go = QPushButton("Run the whole routine")
        self.go.setStyleSheet("font-weight:bold;padding:6px 14px")
        self.go.clicked.connect(self.start)
        self.open_btn = QPushButton("Open existing results ...")
        self.open_btn.clicked.connect(self._open_existing)
        h.addWidget(self.go)
        h.addWidget(self.open_btn)
        h.addStretch()
        lay.addLayout(h)
        self.bar = QProgressBar()
        lay.addWidget(self.bar)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(MONO)
        lay.addWidget(self.log)
        if self.root.text():
            self._discover()

    def _row(self, edit, fn, label="..."):
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        b = QPushButton(label)
        b.clicked.connect(fn)
        h.addWidget(edit)
        h.addWidget(b)
        return w

    def _pick_root(self):
        d = QFileDialog.getExistingDirectory(self, "CorrelativeImaging output folder", self.root.text())
        if d:
            self.root.setText(d)
        self._discover()

    def _discover(self):
        root = self.root.text().strip()
        if not root:
            return
        self.qs.setValue("root", root)
        self.run.clear()
        self.log.appendPlainText(f"searching runs under {root} ...")
        self._dw = DiscoverWorker(root)
        self._dw.found.connect(self._runs_found)
        self._dw.start()

    def _runs_found(self, runs):
        for tag, n in runs:
            self.run.addItem(tag, tag)
            self.run.setItemText(self.run.count() - 1, f"{tag}   ({n} plates)")
        self.log.appendPlainText(f"{len(runs)} runs found" if runs else "no result databases found there")
        if runs:
            self.run.setCurrentIndex(len(runs) - 1)

    def _default_out(self, *_):
        tag = self.run.currentData()
        if tag and self.root.text():
            self.out.setText(str(Path(self.root.text()) / f"downstream_{tag}"))

    def _pick_coldata(self):
        f, _ = QFileDialog.getOpenFileName(self, "Plate map", self.coldata.text(), "CSV (*.csv)")
        if f:
            self.coldata.setText(f)

    def _pick_out(self):
        d = QFileDialog.getExistingDirectory(self, "Results folder", self.out.text())
        if d:
            self.out.setText(d)

    def _open_existing(self):
        d = QFileDialog.getExistingDirectory(self, "Results folder of an earlier run", self.out.text())
        if d and (Path(d) / "calls.csv").exists():
            self.finished.emit(d)
        elif d:
            QMessageBox.warning(self, "Open results", "No calls.csv in that folder.")

    def start(self):
        try:
            cfg = self.settings.config()
        except ValueError as e:
            QMessageBox.warning(self, "Settings", str(e))
            return
        tag = self.run.currentData()
        if not tag:
            QMessageBox.warning(self, "Run", "Pick an output folder and a run first.")
            return
        cd = None
        if self.coldata.text().strip():
            from .evaluate import read_col_data
            try:
                cd = read_col_data(self.coldata.text().strip())
                self.qs.setValue("coldata", self.coldata.text().strip())
            except Exception as e:
                QMessageBox.warning(self, "Plate map", f"Could not read it (needs plate_id, well_id): {e}")
                return
        wells = self.wells.text().split() or None
        out = self.out.text().strip()
        Path(out).mkdir(parents=True, exist_ok=True)
        self.go.setEnabled(False)
        self.bar.setRange(0, 0)
        self.log.appendPlainText(f"\n=== run {tag} -> {out}")
        self.worker = RunWorker(self.root.text().strip(), tag, out, cfg, cd, wells, self.nwork.value(),
                                self.title.text())
        self.worker.progress.connect(self._progress)
        self.worker.done.connect(self._done)
        self.worker.failed.connect(self._failed)
        self.worker.start()

    def _progress(self, i, n, msg):
        if n:
            self.bar.setRange(0, n)
            self.bar.setValue(i)
        if msg:
            self.log.appendPlainText(msg)
        elif n and (i % 50 == 0 or i == n):
            self.log.appendPlainText(f"{i}/{n} wells")

    def _done(self, out):
        self.go.setEnabled(True)
        self.log.appendPlainText(f"done: {out}")
        self.finished.emit(out)

    def _failed(self, tb):
        self.go.setEnabled(True)
        self.bar.setRange(0, 1)
        self.log.appendPlainText(tb)
        QMessageBox.critical(self, "Run failed", tb.strip().splitlines()[-1])


class Bridge(QObject):
    """Called from the viewer page (QWebChannel)."""
    changed = Signal()

    def __init__(self):
        super().__init__()
        self.store = None

    @Slot(result=str)
    def getState(self):
        return self.store.state_json() if self.store else '{"verdicts":{},"labels":{}}'

    @Slot(str)
    def saveVerdict(self, js):
        if self.store:
            self.store.save_verdict(json.loads(js))
            self.changed.emit()

    @Slot(str)
    def saveLabel(self, js):
        if self.store:
            self.store.save_label(json.loads(js))
            self.changed.emit()


class ReviewTab(QWidget):
    def __init__(self, qs: QSettings):
        super().__init__()
        self.qs, self.out = qs, None
        lay = QVBoxLayout(self)
        bar = QHBoxLayout()
        self.who = QLineEdit(qs.value("reviewer", ""))
        self.who.setPlaceholderText("your name")
        self.who.setMaximumWidth(160)
        self.who.editingFinished.connect(self._reviewer)
        self.info = QLabel("no results loaded")
        rl = QPushButton("Reload")
        rl.clicked.connect(self.reload)
        ob = QPushButton("Open in browser")
        ob.clicked.connect(self._browser)
        bar.addWidget(QLabel("Reviewer"))
        bar.addWidget(self.who)
        bar.addWidget(rl)
        bar.addWidget(ob)
        bar.addStretch()
        bar.addWidget(self.info)
        lay.addLayout(bar)
        self.bridge = Bridge()
        self.bridge.changed.connect(self._count)
        if HAVE_WEB:
            self.web = QWebEngineView()
            s = self.web.settings()
            s.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)
            s.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
            self.channel = QWebChannel()
            self.channel.registerObject("bridge", self.bridge)
            self.web.page().setWebChannel(self.channel)
            lay.addWidget(self.web)
        else:
            self.web = None
            lay.addWidget(QLabel("Qt WebEngine is not installed: use 'Open in browser' "
                                 "(verdicts/labels then come back via their CSV export)."))

    def _reviewer(self):
        self.qs.setValue("reviewer", self.who.text())
        if self.bridge.store:
            self.bridge.store.reviewer = self.who.text()

    def load(self, out):
        from .review import ReviewStore
        self.out = Path(out)
        self.bridge.store = ReviewStore(self.out, self.who.text())
        self._count()
        self.reload()

    def reload(self):
        if self.out and self.web:
            self.web.load(QUrl.fromLocalFile(str(self.out / "viewer" / "index.html")))

    def _browser(self):
        if self.out:
            import webbrowser
            webbrowser.open((self.out / "viewer" / "index.html").as_uri())

    def _count(self):
        if self.bridge.store:
            v, l = self.bridge.store.verdicts(), self.bridge.store.labels()
            self.info.setText(f"{self.out.name}: {len(v)} verdicts, {len(l)} particle labels (review.db)")


class ResultsTab(QWidget):
    models_fitted = Signal(str)

    def __init__(self, settings: SettingsTab):
        super().__init__()
        self.settings, self.out = settings, None
        lay = QVBoxLayout(self)
        bar = QHBoxLayout()
        for txt, fn in (("Refresh", self.refresh), ("Export calls.csv ...", self.export)):
            b = QPushButton(txt)
            b.clicked.connect(fn)
            bar.addWidget(b)
        bar.addStretch()
        lay.addLayout(bar)
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setFont(MONO)
        lay.addWidget(self.text, 3)
        box = QGroupBox("Refit the particle models from your labels")
        bl = QVBoxLayout(box)
        bl.addWidget(QLabel("Your particle labels of this run (review.db), plus earlier label files (same columns: "
                            "particle_label, snr, area_um2, channel, share_inside, fill). Cross-validated AUC is shown; "
                            "the new model file is set in Settings, then rerun."))
        self.extra = QListWidget()
        self.extra.setMaximumHeight(70)
        h = QHBoxLayout()
        add = QPushButton("Add label file ...")
        add.clicked.connect(self._add_extra)
        rm = QPushButton("Remove")
        rm.clicked.connect(lambda: [self.extra.takeItem(self.extra.row(i)) for i in self.extra.selectedItems()])
        fit = QPushButton("Fit models")
        fit.clicked.connect(self.fit)
        h.addWidget(add)
        h.addWidget(rm)
        h.addStretch()
        h.addWidget(fit)
        bl.addWidget(self.extra)
        bl.addLayout(h)
        lay.addWidget(box, 1)

    def load(self, out):
        self.out = Path(out)
        self.refresh()

    def refresh(self):
        if not self.out:
            return
        calls = pd.read_csv(self.out / "calls.csv", dtype={"plate_id": str})
        L = [f"{self.out}\n{len(calls)} wells\n", "tier x simple call:",
             pd.crosstab(calls.tier, calls.call, margins=True).to_string(), "\ntier x calibrated call:",
             pd.crosstab(calls.tier, calls.call_cal, margins=True).to_string()]
        ev = self.out / "evaluation.txt"
        if ev.exists():
            L += ["\nagainst the plate map (evaluation only):", ev.read_text()]
        rv = self.out / "review.db"
        if rv.exists():
            from .review import ReviewStore
            st = ReviewStore(self.out)
            v, lab = st.verdicts(), st.labels()
            L.append(f"\nreview: {len(v)} verdicts, {len(lab)} particle labels")
            if len(v):
                L.append(v.verdict.value_counts().to_string())
            if len(lab):
                L.append(lab.particle_label.value_counts().to_string())
        self.text.setPlainText("\n".join(L))

    def export(self):
        if not self.out:
            return
        f, _ = QFileDialog.getSaveFileName(self, "Export calls", str(self.out / "calls.csv"), "CSV (*.csv)")
        if f:
            pd.read_csv(self.out / "calls.csv", dtype={"plate_id": str}).to_csv(f, index=False)

    def _add_extra(self):
        fs, _ = QFileDialog.getOpenFileNames(self, "Earlier label files", "", "CSV (*.csv)")
        for f in fs:
            self.extra.addItem(f)

    def fit(self):
        if not self.out:
            return
        from .models import fit_models
        from .review import training_table
        extra = [self.extra.item(i).text() for i in range(self.extra.count())]
        try:
            t = training_table(self.out, extra)
            cfg = self.settings.config()
            if len(t) < 30:
                raise ValueError(f"only {len(t)} labelled particles; label more first")
            name, ok = QInputDialog.getText(self, "Model name", "Save the models as", text=f"{self.out.name}_models")
            if not ok:
                return
            m = fit_models(t, list(cfg.channels.values()), source=f"{self.out} + {extra}")
            path = Path(__file__).parent / "models" / f"{name}.json"
            m.save(path)
        except Exception as e:
            QMessageBox.warning(self, "Fit models", str(e))
            return
        QMessageBox.information(self, "Fit models",
                                f"{len(t)} labelled particles\nP(cell) cv AUC {m.spec['cell']['cv_auc']:.3f}\n"
                                f"P(neighbour) cv AUC {m.spec['nb']['cv_auc']:.3f}\nsaved {path.name}; set in Settings.")
        self.models_fitted.emit(str(path))


class MainWindow(QMainWindow):
    def __init__(self, root: str | None = None):
        super().__init__()
        self.setWindowTitle("Patched-cell identity: downstream analyzer")
        self.resize(1500, 950)
        qs = QSettings("CorrelativeImaging", "downstream")
        if root:
            qs.setValue("root", root)
        self.tabs = QTabWidget()
        self.settings = SettingsTab()
        self.runtab = RunTab(self.settings, qs)
        self.review = ReviewTab(qs)
        self.results = ResultsTab(self.settings)
        self.tabs.addTab(self.runtab, "1  Run")
        self.tabs.addTab(self.settings, "2  Settings")
        self.tabs.addTab(self.review, "3  Review")
        self.tabs.addTab(self.results, "4  Results")
        self.setCentralWidget(self.tabs)
        self.runtab.finished.connect(self.load_results)
        self.results.models_fitted.connect(self._set_models)

    def load_results(self, out):
        from .viewer import rebuild_index
        try:
            rebuild_index(out)                       # page from the current template, data unchanged
        except Exception as e:
            QMessageBox.warning(self, "Viewer", f"could not rebuild the viewer page: {e}")
        self.review.load(out)
        self.results.load(out)
        cfgf = Path(out) / "config.json"
        if cfgf.exists():
            self.settings.set_config(load_config(cfgf), f"settings of {Path(out).name}")
        self.tabs.setCurrentWidget(self.review)

    def _set_models(self, path):
        self.settings.widgets["models"].edit.setText(path)
        self.tabs.setCurrentWidget(self.settings)


def launch(root: str | None = None, results: str | None = None):
    app = QApplication.instance() or QApplication(sys.argv)
    w = MainWindow(root)
    w.show()
    if results:
        w.load_results(results)
    return app.exec()
