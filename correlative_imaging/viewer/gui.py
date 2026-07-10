"""Napari-based guided GUI — folder picker, channel sidebar, multiple ROI selections, batch runner."""

from __future__ import annotations

import datetime
import json
import logging
import re
import traceback
from dataclasses import dataclass
from pathlib import Path

from qtpy.QtCore import Qt, QSettings, QThread, QTimer, Signal
from qtpy.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from correlative_imaging.diagnostics import CHANNEL_COLOR_CHOICES as _CHANNEL_COLOR_CHOICES
from correlative_imaging.io import supported_extensions
from correlative_imaging.pipeline.analyze import INTENSITY_METRIC_CHOICES, PARTICLE_METRIC_CHOICES
from correlative_imaging.viewer.napari_viewer import auto_contrast_limits

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# Advanced-tab form definitions
# ──────────────────────────────────────────────────────────────────

_STEP_FORMS: dict[str, list[tuple]] = {
    "BackgroundSubtraction": [
        ("channel", "int",                        "Channel index",   0,    (0, 32)),
        ("radius",  "float",                      "Ball radius (px)", 50.0, (1.0, 500.0)),
        ("method",  "choice:rolling_ball,tophat", "Method",          "rolling_ball", None),
    ],
    "GaussianBlur": [
        ("channel", "int",   "Channel index", 0,   (0, 32)),
        ("sigma",   "float", "Sigma (px)",    2.0, (0.1, 50.0)),
    ],
    "Normalize": [
        ("channel",  "int",                      "Channel index",   0,    (0, 32)),
        ("method",   "choice:minmax,percentile", "Method",          "minmax", None),
        ("low_pct",  "float",                    "Low percentile",  1.0,  (0.0, 49.0)),
        ("high_pct", "float",                    "High percentile", 99.0, (51.0, 100.0)),
    ],
    "ExtractROI": [
        ("channel",    "int",                                "Channel index",  0,    (0, 32)),
        ("blur_sigma", "float",                              "Blur sigma (px)", 20.0, (1.0, 200.0)),
        ("method",     "choice:otsu,li,yen,triangle,isodata", "Threshold",    "otsu", None),
        ("roi_name",   "str",                                "Mask key",       "roi", None),
    ],
    "LoadROI": [
        ("path",     "str", "ROI file path", "", None),
        ("roi_name", "str", "Mask key",      "roi", None),
    ],
    "AutoThreshold": [
        ("channel",      "int",                                "Channel index",       0,  (0, 32)),
        ("method",       "choice:otsu,li,yen,triangle,isodata", "Method",           "otsu", None),
        ("z_projection", "choice:max,mean,sum",               "Z projection",        "max", None),
        ("min_size",     "int",                               "Min object size (px)", 50,  (0, 100000)),
    ],
    "WatershedSplit": [
        ("channel",      "int", "Channel index",     0, (0, 32)),
        ("min_distance", "int", "Min distance (px)", 5, (1, 200)),
    ],
    "ParticleAnalysis": [
        ("channel",         "int",                "Channel index",   0,      (0, 32)),
        ("min_size_um2",    "float",              "Min size (µm²)",  0.5,    (0.0, 1e6)),
        ("max_size_um2",    "float",              "Max size (µm²)",  5000.0, (0.0, 1e9)),
        ("min_circularity", "float",              "Min circularity", 0.0,    (0.0, 1.0)),
        ("z_projection",    "choice:max,mean,sum", "Z projection",   "max",  None),
        ("roi_mask",        "str",                "ROI mask key",    "",     None),
    ],
    "IntensityMeasurement": [
        ("channel",      "int",                "Channel index", 0,     (0, 32)),
        ("z_projection", "choice:max,mean,sum", "Z projection", "max", None),
        ("roi_mask",     "str",                "ROI mask key",  "",    None),
    ],
    "ColocalizationAnalysis": [
        ("primary_channel",   "int",                "Primary channel",   0,   (0, 32)),
        ("secondary_channel", "int",                "Secondary channel", 1,   (0, 32)),
        ("dilation_um",       "float",              "Mask dilation (µm)", 0.0, (0.0, 20.0)),
        ("z_projection",      "choice:max,mean,sum", "Z projection",     "max", None),
        ("roi_mask",          "str",                "ROI mask key",      "",   None),
    ],
}


# ──────────────────────────────────────────────────────────────────
# Background workers
# ──────────────────────────────────────────────────────────────────

class _ScanWorker(QThread):
    found  = Signal(list)
    status = Signal(str)

    def __init__(self, folder: Path):
        super().__init__()
        self.folder = folder

    def run(self) -> None:
        self.status.emit(f"Scanning {self.folder} …")
        files = sorted(
            str(p) for p in self.folder.rglob("*")
            if p.is_file() and p.suffix.lower() in supported_extensions
        )
        self.found.emit(files)
        self.status.emit(f"{len(files)} image(s) found.")


class _LoadImageWorker(QThread):
    loaded = Signal(object)
    error  = Signal(str)

    def __init__(self, path: Path):
        super().__init__()
        self.path = path

    def run(self) -> None:
        try:
            from correlative_imaging.io import read_image
            self.loaded.emit(read_image(self.path))
        except Exception:
            self.error.emit(traceback.format_exc())


class _LoadWellWorker(QThread):
    """Load a well's BF and FL images together (for the combined overview)."""
    loaded = Signal(object, object)  # (bf_data | None, fl_data | None)
    error  = Signal(str)

    def __init__(self, well):
        super().__init__()
        self._well = well

    def run(self) -> None:
        try:
            from correlative_imaging.io import read_image
            bf = read_image(self._well.bf_path) if self._well.bf_path else None
            fl = read_image(self._well.fl_path) if self._well.fl_path else None
            self.loaded.emit(bf, fl)
        except Exception:
            self.error.emit(traceback.format_exc())


def _run_basename(experiment: str) -> str:
    """Base filename (no extension) for one run's DB/pipeline-JSON/log,
    shared across all three so they're obviously grouped and never
    overwrite an earlier run's outputs."""
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp = re.sub(r"[^A-Za-z0-9_.-]+", "_", experiment.strip()) or "run"
    return f"{exp}_{ts}"


class _BatchWorker(QThread):
    progress = Signal(int, int, str, object)
    log_msg  = Signal(str)
    finished = Signal(str)

    def __init__(self, pipeline_dict: dict, input_dir: Path, output_dir: Path, experiment: str,
                 db_path: Path | None = None):
        super().__init__()
        self.pipeline_dict = pipeline_dict
        self.input_dir  = input_dir
        self.output_dir = output_dir
        self.experiment = experiment
        self.db_path = Path(db_path) if db_path else (output_dir / "results.db")
        self._abort = False

    def abort(self) -> None:
        self._abort = True

    def run(self) -> None:
        try:
            import os, tempfile
            from correlative_imaging.pipeline import Pipeline
            from correlative_imaging.batch import BatchRunner

            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                json.dump(self.pipeline_dict, f)
                tmp = f.name
            pl = Pipeline.load(tmp)
            os.unlink(tmp)

            runner = BatchRunner(pl, db_path=self.db_path, experiment=self.experiment)

            def _cb(current, total, name, n_particles):
                if self._abort:
                    return
                self.progress.emit(current, total, name, n_particles)
                icon   = "✓" if n_particles is not None else "✗"
                detail = f"({n_particles} particles)" if n_particles is not None else "(error)"
                self.log_msg.emit(f"{icon}  {name}  {detail}")

            runner.run_directory(self.input_dir, output_dir=self.output_dir, progress_fn=_cb)
            self.finished.emit(str(self.db_path))
        except Exception:
            self.log_msg.emit(traceback.format_exc())
            self.finished.emit("")


class _WellBatchWorker(QThread):
    """Batch runner for plate wells: reuse-or-generate each well's BF-pipeline
    class ROIs, then run the FL analysis pipeline per well (see
    WellBatchRunner in correlative_imaging.batch).

    ``pipeline_dict_fn`` must be safe to call from this background thread —
    build it with ``CorrelativeImagingWidget.make_well_pipeline_dict_fn()``,
    which snapshots all Qt-widget-derived config on the main thread first.
    """
    progress = Signal(int, int, str, object)
    step_progress = Signal(str, int, int, str)   # well_id, step_index, total_steps, step_name
    log_msg  = Signal(str)
    finished = Signal(str)

    def __init__(self, wells: list, pipeline_dict_fn, bf_cfg: dict | None,
                 output_dir: Path, experiment: str, max_workers: int = 1,
                 db_path: Path | None = None, force_regen: bool = False,
                 diag_cfg: dict | None = None):
        super().__init__()
        self._wells = wells
        self._pipeline_dict_fn = pipeline_dict_fn
        self._bf_cfg = bf_cfg
        self._output_dir = Path(output_dir)
        self._experiment = experiment
        self._max_workers = max_workers
        self._db_path = Path(db_path) if db_path else (self._output_dir / "results.db")
        self._force_regen = force_regen
        self._diag_cfg = diag_cfg
        self._abort = False

    def abort(self) -> None:
        self._abort = True

    def run(self) -> None:
        try:
            from correlative_imaging.batch import WellBatchRunner

            total = len(self._wells)

            # ── Ensure every configured BF-pipeline class ROI exists for
            # every well, reusing from disk — a single BF-pipeline pass
            # produces all configured classes at once, so a well missing
            # any one of them gets every class (re)generated together.
            if self._bf_cfg:
                class_names = [c["name"] for c in self._bf_cfg.get("classes", [])]
                if self._force_regen:
                    missing = list(self._wells)
                else:
                    missing = [
                        w for w in self._wells
                        if any(_find_well_roi(w, name) is None for name in class_names)
                    ]
                if missing:
                    reason = "forced regeneration" if self._force_regen else \
                        f"({total - len(missing)} reused from disk)"
                    self.log_msg.emit(
                        f"Generating BF-pipeline ROI(s) for {len(missing)}/{total} well(s) {reason} …"
                    )
                    bf_worker = _BFWorker(missing, self._bf_cfg, test_mode=False)
                    bf_worker.log_msg.connect(lambda m: self.log_msg.emit(f"[BF] {m}"))
                    bf_worker.run()   # synchronous call — reuses its logic on this thread
                else:
                    self.log_msg.emit(f"All {total} well(s) already have their BF-pipeline ROI(s) on disk.")

            if self._abort:
                self.log_msg.emit("Aborted before FL analysis.")
                self.finished.emit("")
                return

            db_path = self._db_path
            runner = WellBatchRunner(db_path=db_path, experiment=self._experiment)

            def _cb(current, total_, name, n_particles):
                self.progress.emit(current, total_, name, n_particles)
                icon   = "✓" if n_particles is not None else "✗"
                detail = f"({n_particles} particles)" if n_particles is not None else "(error)"
                self.log_msg.emit(f"{icon}  {name}  {detail}")

            def _on_step(well, step_idx, total_steps, step_name):
                self.step_progress.emit(well.well_id, step_idx, total_steps, step_name)

            runner.run_wells(
                self._wells,
                pipeline_dict_fn=self._pipeline_dict_fn,
                progress_fn=_cb,
                should_abort=lambda: self._abort,
                max_workers=self._max_workers,
                warn_fn=self.log_msg.emit,
                diag_cfg=self._diag_cfg,
                on_step_fn=_on_step,
            )
            self.finished.emit("" if self._abort else str(db_path))
        except Exception:
            self.log_msg.emit(traceback.format_exc())
            self.finished.emit("")


# ──────────────────────────────────────────────────────────────────
# Tab 1 – Setup  (folder + sample loader only)
# ──────────────────────────────────────────────────────────────────

class SetupTab(QWidget):
    channels_ready = Signal(list)   # list[str] raw channel names from the image file

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scan_worker: _ScanWorker | None = None
        self._load_worker: _LoadImageWorker | None = None
        self._image_data = None
        self._scan_timer = QTimer(self)
        self._scan_timer.setSingleShot(True)
        self._scan_timer.timeout.connect(self._start_scan)
        self._build()

    def _build(self) -> None:
        lay = QVBoxLayout(self)

        folder_box = QGroupBox("1. Select folders")
        fl = QFormLayout(folder_box)
        self._input_edit, r1  = _path_row("Input folder …",  is_dir=True)
        self._output_edit, r2 = _path_row("Output folder …", is_dir=True)
        self._exp_edit = QLineEdit()
        self._exp_edit.setPlaceholderText("optional experiment label")
        fl.addRow("Input folder:",  r1)
        fl.addRow("Output folder:", r2)
        fl.addRow("Experiment:",    self._exp_edit)
        self._input_edit.textChanged.connect(self._on_input_changed)
        lay.addWidget(folder_box)

        img_box = QGroupBox("2. Images found")
        il = QVBoxLayout(img_box)
        self._scan_status = QLabel("No folder selected.")
        il.addWidget(self._scan_status)
        self._img_list = QListWidget()
        self._img_list.setMaximumHeight(130)
        il.addWidget(self._img_list)
        self._load_btn = QPushButton("Load sample image → configure Channels & ROI tabs")
        self._load_btn.setEnabled(False)
        self._load_btn.clicked.connect(self._on_load_sample)
        il.addWidget(self._load_btn)
        lay.addWidget(img_box)

        hint = QLabel(
            "After loading a sample:\n"
            "  • Channels tab — rename channels, set preprocessing & segmentation\n"
            "  • ROI & Selections tab — define one or more analysis regions\n"
            "  • Combine tab — colocalization pairs\n"
            "  • Run tab — preview on sample, then batch"
        )
        hint.setWordWrap(True)
        lay.addWidget(hint)
        lay.addStretch()

    def _on_input_changed(self, text: str) -> None:
        self._scan_timer.start(600)
        if not self._output_edit.text():
            p = Path(text)
            if p.is_dir():
                self._output_edit.setText(str(p / "output"))

    def _start_scan(self) -> None:
        folder = Path(self._input_edit.text())
        if not folder.is_dir():
            return
        self._scan_status.setText("Scanning …")
        self._img_list.clear()
        self._load_btn.setEnabled(False)
        if self._scan_worker and self._scan_worker.isRunning():
            self._scan_worker.terminate()
        self._scan_worker = _ScanWorker(folder)
        self._scan_worker.found.connect(self._on_scan_done)
        self._scan_worker.status.connect(self._scan_status.setText)
        self._scan_worker.start()

    def _on_scan_done(self, paths: list[str]) -> None:
        self._img_list.clear()
        folder = Path(self._input_edit.text())
        for p in paths:
            item = QListWidgetItem(str(Path(p).relative_to(folder)))
            item.setData(Qt.UserRole, p)
            self._img_list.addItem(item)
        self._scan_status.setText(f"{len(paths)} image(s) found.")
        self._load_btn.setEnabled(bool(paths))
        if paths:
            self._img_list.setCurrentRow(0)

    def _on_load_sample(self) -> None:
        item = self._img_list.currentItem() or self._img_list.item(0)
        if item is None:
            return
        self._load_btn.setEnabled(False)
        self._load_btn.setText("Loading …")
        self._load_worker = _LoadImageWorker(Path(item.data(Qt.UserRole)))
        self._load_worker.loaded.connect(self._on_image_loaded)
        self._load_worker.error.connect(self._on_load_error)
        self._load_worker.start()

    def _on_load_error(self, msg: str) -> None:
        QMessageBox.critical(self, "Load error", msg)
        self._load_btn.setEnabled(True)
        self._load_btn.setText("Load sample image → configure Channels & ROI tabs")

    def _on_image_loaded(self, image_data) -> None:
        self._image_data = image_data
        self._load_btn.setEnabled(True)
        self._load_btn.setText("Load sample image → configure Channels & ROI tabs")
        self.channels_ready.emit(image_data.channel_names)

    # ── Accessors ──────────────────────────────────────────────────

    @property
    def input_dir(self) -> Path | None:
        p = Path(self._input_edit.text())
        return p if p.is_dir() else None

    @property
    def output_dir(self) -> Path:
        t = self._output_edit.text()
        return Path(t) if t else (self.input_dir or Path(".")) / "output"

    @property
    def experiment(self) -> str:
        return self._exp_edit.text().strip()

    @property
    def image_data(self):
        return self._image_data

    @property
    def image_paths(self) -> list[Path]:
        return [
            Path(self._img_list.item(i).data(Qt.UserRole))
            for i in range(self._img_list.count())
        ]


# ──────────────────────────────────────────────────────────────────
# Tab 2 – Plate  (384-well grid, BF/FL scanner)
# ──────────────────────────────────────────────────────────────────

_PLATE_ROWS = list("ABCDEFGHIJKLMNOP")   # 16 rows
_PLATE_COLS = list(range(1, 25))          # 24 columns

_WELL_STATUS_COLORS = {
    "empty":    "#4a4a4a",
    "complete": "#4CAF50",
    "bf_only":  "#FF9800",
    "selected": "#2196F3",
}

# One napari colormap per pipeline step in the preview, cycling if there are more steps than colors.
_STEP_COLORMAPS = ["green", "cyan", "magenta", "yellow", "red", "blue", "orange", "gray"]

_ROI_BADGE_COLOR = "#FFEB3B"  # border color marking a well with ROI file(s)


class _WellButton(QPushButton):
    """Single cell in the plate grid."""

    def __init__(self, well_id: str, parent=None):
        super().__init__(parent)
        self.well_id = well_id
        self.setFixedSize(24, 24)
        self.setToolTip(well_id)
        self.setText("")
        self._has_roi = False
        self._apply("empty")

    def set_status(self, status: str, has_roi: bool | None = None) -> None:
        if has_roi is not None:
            self._has_roi = has_roi
        self._apply(status)
        self.setToolTip(f"{self.well_id}  (has ROI)" if self._has_roi else self.well_id)

    def _apply(self, status: str) -> None:
        c = _WELL_STATUS_COLORS.get(status, _WELL_STATUS_COLORS["empty"])
        border = f"2px solid {_ROI_BADGE_COLOR}" if self._has_roi else "1px solid #222"
        self.setStyleSheet(
            f"background-color:{c}; border:{border}; border-radius:2px;"
        )


class _PlateGrid(QWidget):
    """16×24 clickable well grid."""

    well_clicked = Signal(str)   # emits well_id

    def __init__(self, parent=None):
        super().__init__(parent)
        self._btns: dict[str, _WellButton] = {}
        self._selected: str | None = None
        self._prev_status: dict[str, str] = {}   # to restore colour on deselect
        self._has_roi: dict[str, bool] = {}
        self._build()

    def _build(self) -> None:
        grid = QVBoxLayout(self)
        grid.setSpacing(2)
        grid.setContentsMargins(4, 4, 4, 4)

        # Column header row
        hdr = QHBoxLayout(); hdr.setSpacing(2)
        lbl = QLabel(""); lbl.setFixedWidth(18)
        hdr.addWidget(lbl)
        for col in _PLATE_COLS:
            l = QLabel(str(col)); l.setFixedWidth(24); l.setAlignment(Qt.AlignCenter)
            l.setStyleSheet("font-size:9px; color:#aaa;")
            hdr.addWidget(l)
        grid.addLayout(hdr)

        for row in _PLATE_ROWS:
            row_lay = QHBoxLayout(); row_lay.setSpacing(2)
            lbl = QLabel(row); lbl.setFixedWidth(18); lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("font-size:9px; color:#aaa;")
            row_lay.addWidget(lbl)
            for col in _PLATE_COLS:
                wid = f"{row}{col}"
                btn = _WellButton(wid)
                btn.clicked.connect(lambda _checked, w=wid: self._on_click(w))
                self._btns[wid] = btn
                row_lay.addWidget(btn)
            grid.addLayout(row_lay)

    def _on_click(self, well_id: str) -> None:
        if self._selected and self._selected != well_id:
            self._btns[self._selected].set_status(
                self._prev_status.get(self._selected, "empty"),
                self._has_roi.get(self._selected, False),
            )
        self._selected = well_id
        self._prev_status[well_id] = self._prev_status.get(well_id, "empty")
        self._btns[well_id].set_status("selected", self._has_roi.get(well_id, False))
        self.well_clicked.emit(well_id)

    def set_well_status(self, well_id: str, status: str, has_roi: bool = False) -> None:
        if well_id in self._btns:
            self._prev_status[well_id] = status
            self._has_roi[well_id] = has_roi
            if well_id != self._selected:
                self._btns[well_id].set_status(status, has_roi)

    def clear_all(self) -> None:
        self._selected = None
        self._prev_status.clear()
        self._has_roi.clear()
        for btn in self._btns.values():
            btn.set_status("empty")


class PlateTab(QWidget):
    """Tab 1: folder setup + 384-well plate scanner + BF/FL pairing.

    Replaces the old SetupTab — exposes the same interface (input_dir,
    output_dir, experiment, image_data, image_paths, channels_ready) so
    RunTab and CorrelativeImagingWidget work without changes.
    """

    channels_ready = Signal(list)   # list[str] channel names after sample load
    well_selected  = Signal(object) # WellInfo when a well is clicked
    view_requested = Signal(object, str, str)  # (WellInfo, "bf" | "fl", projection)
    overview_requested = Signal(object, str)  # (WellInfo, projection) — BF + FL + ROI
    existing_rois_detected = Signal(list)  # list[str] distinct tags found across the scan
    load_pipeline_requested = Signal(dict)  # parsed pipeline dict, ready to apply everywhere

    def __init__(self, parent=None):
        super().__init__(parent)
        self._wells: dict[str, object] = {}
        self._image_data = None
        self._load_worker: _LoadImageWorker | None = None
        self._restoring_settings = False
        self._build()

    def _build(self) -> None:
        # PlateTab itself is never shown directly — it's a controller that
        # owns two separate page widgets (setup_page / plate_page) so the
        # GUI can show them as two distinct tabs instead of one tall tab
        # that overflows the window (folders+JSON belong together and are
        # short; the scan/grid/well-detail UI is tall and grows with the
        # plate size).
        self.setup_page = QWidget()
        self.plate_page = QWidget()

        setup_lay = QVBoxLayout(self.setup_page)

        # ── 1. Folders & experiment ──────────────────────────────────
        folder_box = QGroupBox("1. Folders & experiment")
        ff = QFormLayout(folder_box)

        self._folder_edit, folder_row = _path_row("Input / plate folder …", is_dir=True)
        self._folder_edit.textChanged.connect(self._on_folder_changed)
        self._output_edit, output_row = _path_row("Output folder …", is_dir=True)
        self._exp_edit = QLineEdit()
        self._exp_edit.setPlaceholderText("optional experiment label")

        ff.addRow("Input folder:",  folder_row)
        ff.addRow("Output folder:", output_row)
        ff.addRow("Experiment:",    self._exp_edit)
        setup_lay.addWidget(folder_box)

        # ── 2. Plate scan options ────────────────────────────────────
        scan_box = QGroupBox("2. Plate scan")
        sl = QFormLayout(scan_box)

        self._ext_edit = QLineEdit(".vsi")
        sl.addRow("Extension:", self._ext_edit)

        self._contains_edit = QLineEdit("")
        self._contains_edit.setPlaceholderText("optional substring filter")
        sl.addRow("Name contains:", self._contains_edit)

        self._recursive_cb = QCheckBox("Search subfolders recursively")
        sl.addRow(self._recursive_cb)

        scan_btn = QPushButton("Scan folder")
        scan_btn.clicked.connect(self._scan)
        sl.addRow(scan_btn)

        self._status_lbl = QLabel("")
        self._status_lbl.setWordWrap(True)
        sl.addRow(self._status_lbl)
        setup_lay.addWidget(scan_box)

        # ── 3. Load an existing pipeline JSON (optional) ─────────────
        json_box = QGroupBox("3. Load existing pipeline (optional)")
        jl = QVBoxLayout(json_box)
        json_row = QHBoxLayout()
        json_row.addWidget(QLabel("Pipeline JSON:"))
        self._json_combo = QComboBox()
        self._json_combo.addItem("(none found in output folder)")
        json_row.addWidget(self._json_combo, stretch=1)
        load_json_btn = QPushButton("Load")
        load_json_btn.clicked.connect(self._on_load_json_clicked)
        json_row.addWidget(load_json_btn)
        browse_json_btn = QPushButton("Browse …")
        browse_json_btn.clicked.connect(self._on_browse_json_clicked)
        json_row.addWidget(browse_json_btn)
        jl.addLayout(json_row)
        setup_lay.addWidget(json_box)
        setup_lay.addStretch()

        # ── Plate page: grid + selected-well detail ───────────────────
        plate_outer = QVBoxLayout(self.plate_page)
        plate_outer.setContentsMargins(0, 0, 0, 0)
        plate_content = QWidget()
        lay = QVBoxLayout(plate_content)
        plate_scroll = QScrollArea()
        plate_scroll.setWidgetResizable(True)
        plate_scroll.setWidget(plate_content)
        plate_outer.addWidget(plate_scroll)

        # ── 1. Plate grid ────────────────────────────────────────────
        grid_box = QGroupBox("1. Plate layout  (click a well to select)")
        gl = QVBoxLayout(grid_box)

        legend = QHBoxLayout()
        for color, text in [
            (_WELL_STATUS_COLORS["empty"],    "Empty"),
            (_WELL_STATUS_COLORS["complete"], "BF + FL"),
            (_WELL_STATUS_COLORS["bf_only"],  "BF only"),
            (_WELL_STATUS_COLORS["selected"], "Selected"),
        ]:
            dot = QLabel("■"); dot.setStyleSheet(f"color:{color}; font-size:14px;")
            legend.addWidget(dot)
            lbl = QLabel(text); lbl.setStyleSheet("font-size:10px;")
            legend.addWidget(lbl)
            legend.addSpacing(6)
        roi_dot = QLabel("▢")
        roi_dot.setStyleSheet(f"color:{_ROI_BADGE_COLOR}; font-size:14px; font-weight:bold;")
        legend.addWidget(roi_dot)
        roi_lbl = QLabel("Has ROI"); roi_lbl.setStyleSheet("font-size:10px;")
        legend.addWidget(roi_lbl)
        legend.addStretch()
        gl.addLayout(legend)

        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        self._grid = _PlateGrid()
        self._grid.well_clicked.connect(self._on_well_clicked)
        scroll.setWidget(self._grid)
        scroll.setMinimumHeight(220)
        gl.addWidget(scroll)
        lay.addWidget(grid_box)

        # ── 4. Selected well / load sample ───────────────────────────
        detail_box = QGroupBox("2. Selected well")
        dl = QFormLayout(detail_box)
        self._d_well = QLabel("—")
        self._d_bf   = QLabel("—"); self._d_bf.setWordWrap(True)
        self._d_fl   = QLabel("—"); self._d_fl.setWordWrap(True)
        self._d_roi  = QLabel("—"); self._d_roi.setWordWrap(True)
        dl.addRow("Well:",      self._d_well)
        dl.addRow("BF path:",   self._d_bf)
        dl.addRow("FL path:",   self._d_fl)
        dl.addRow("ROI files:", self._d_roi)

        self._auto_roi_cb = QCheckBox(
            "Auto-assign detected ROI files to ROI & Selections tab on well click"
        )
        self._auto_roi_cb.setChecked(True)
        dl.addRow(self._auto_roi_cb)

        proj_row = QHBoxLayout()
        proj_row.addWidget(QLabel("Projection:"))
        self._view_proj_combo = QComboBox()
        self._view_proj_combo.addItems(["max", "min", "mean", "sum"])
        self._view_proj_combo.setToolTip(
            "Z-projection method used when showing images below (no effect on 2-D images)."
        )
        proj_row.addWidget(self._view_proj_combo)
        proj_row.addStretch()
        dl.addRow(proj_row)

        view_row = QHBoxLayout()
        self._view_bf_btn  = QPushButton("Show BF in viewer")
        self._view_fl_btn  = QPushButton("Show FL in viewer")
        self._view_all_btn = QPushButton("Show BF + FL + ROI overview")
        self._view_all_btn.setStyleSheet("font-weight:bold")
        self._view_bf_btn.setEnabled(False)
        self._view_fl_btn.setEnabled(False)
        self._view_all_btn.setEnabled(False)
        self._view_bf_btn.clicked.connect(lambda: self._on_view("bf"))
        self._view_fl_btn.clicked.connect(lambda: self._on_view("fl"))
        self._view_all_btn.clicked.connect(self._on_view_all)
        view_row.addWidget(self._view_bf_btn)
        view_row.addWidget(self._view_fl_btn)
        dl.addRow(view_row)
        dl.addRow(self._view_all_btn)

        self._load_btn = QPushButton(
            "Load selected well as sample → configure Channels & ROI tabs"
        )
        self._load_btn.setEnabled(False)
        self._load_btn.clicked.connect(self._on_load_sample)
        dl.addRow(self._load_btn)
        lay.addWidget(detail_box)

        lay.addStretch()

        # Restore saved paths/settings and wire auto-save
        self._load_settings()
        self._connect_persistence()

    # ── Persistence ──────────────────────────────────────────────────

    def _load_settings(self) -> None:
        # Guards _on_folder_changed against auto-overwriting the restored
        # output folder with the setText() calls below — it should only
        # follow the input folder on real (user-driven) changes.
        self._restoring_settings = True
        s = QSettings("CorrelativeImaging", "CorrelativeImaging")
        s.beginGroup("plate")
        self._output_edit.setText(s.value("output_folder", ""))
        self._folder_edit.setText(s.value("input_folder", ""))
        self._exp_edit.setText(s.value("experiment", ""))
        ext = s.value("extension", ".vsi")
        self._ext_edit.setText(ext if ext else ".vsi")
        self._contains_edit.setText(s.value("contains", ""))
        self._recursive_cb.setChecked(s.value("recursive", False, type=bool))
        s.endGroup()
        self._restoring_settings = False

    def _save_settings(self) -> None:
        s = QSettings("CorrelativeImaging", "CorrelativeImaging")
        s.beginGroup("plate")
        s.setValue("input_folder",  self._folder_edit.text())
        s.setValue("output_folder", self._output_edit.text())
        s.setValue("experiment",    self._exp_edit.text())
        s.setValue("extension",     self._ext_edit.text())
        s.setValue("contains",      self._contains_edit.text())
        s.setValue("recursive",     self._recursive_cb.isChecked())
        s.endGroup()

    def _connect_persistence(self) -> None:
        for w in (self._folder_edit, self._output_edit, self._exp_edit,
                  self._ext_edit, self._contains_edit):
            w.textChanged.connect(self._save_settings)
        self._recursive_cb.toggled.connect(self._save_settings)

    # ── Load pipeline JSON ────────────────────────────────────────────

    def refresh_json_dropdown(self, output_dir) -> None:
        """Repopulate the dropdown with *_pipeline.json files found in
        output_dir (the naming convention from RunTab's auto-save, see
        _run_basename). Call whenever output_dir might have changed."""
        self._json_combo.blockSignals(True)
        current = self._json_combo.currentText()
        self._json_combo.clear()
        found = sorted(Path(output_dir).glob("*_pipeline.json")) if output_dir else []
        if found:
            for p in found:
                self._json_combo.addItem(p.name, str(p))
        else:
            self._json_combo.addItem("(none found in output folder)")
        idx = self._json_combo.findText(current)
        self._json_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self._json_combo.blockSignals(False)

    def _on_load_json_clicked(self) -> None:
        path = self._json_combo.currentData()
        if not path:
            QMessageBox.information(self, "No file", "No pipeline JSON selected — use Browse instead.")
            return
        self._load_json_path(Path(path))

    def _on_browse_json_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load pipeline JSON", "", "JSON (*.json)")
        if path:
            self._load_json_path(Path(path))

    def _load_json_path(self, path: Path) -> None:
        try:
            pl_dict = json.loads(path.read_text())
        except Exception as exc:
            QMessageBox.critical(self, "Load failed", str(exc))
            return
        self.load_pipeline_requested.emit(pl_dict)

    # ── Slots ────────────────────────────────────────────────────────

    def _on_folder_changed(self, text: str) -> None:
        """Output folder always follows the input/plate folder — every time
        a new one is chosen, not just the first time. Skipped while
        restoring saved settings on startup (see _load_settings) so a
        previously customized output path isn't silently overwritten."""
        if self._restoring_settings:
            return
        p = Path(text)
        if p.is_dir():
            self._output_edit.setText(str(p / "output"))

    def _scan(self) -> None:
        from correlative_imaging.io.plate import scan_plate_folder
        folder = self._folder_edit.text().strip()
        if not folder or not Path(folder).is_dir():
            QMessageBox.warning(self, "No folder", "Set a valid input folder first.")
            return
        ext       = self._ext_edit.text().strip() or ".vsi"
        contains  = self._contains_edit.text().strip()
        recursive = self._recursive_cb.isChecked()
        bf_roi_dir = self.output_dir / "bf_pipeline" / "rois"
        try:
            wells = scan_plate_folder(folder, extension=ext,
                                      contains=contains, recursive=recursive,
                                      extra_roi_dirs=[bf_roi_dir])
        except Exception as exc:
            QMessageBox.critical(self, "Scan failed", str(exc))
            return

        self._wells = {}
        self._grid.clear_all()
        for w in wells:
            self._wells[w.well_id] = w
            self._grid.set_well_status(w.well_id,
                                       "complete" if w.is_complete else "bf_only",
                                       has_roi=bool(w.roi_paths))

        n_complete = sum(1 for w in wells if w.is_complete)
        n_roi      = sum(1 for w in wells if w.roi_paths)
        msg = f"{len(wells)} wells found — {n_complete} complete BF+FL pairs"
        if len(wells) > n_complete:
            msg += f", {len(wells) - n_complete} BF only"
        msg += f".  {n_roi} well(s) have ROI files." if n_roi else "."
        self._status_lbl.setText(msg)

        tags: set[str] = set()
        for w in wells:
            for p in w.roi_paths:
                tag = _guess_existing_tag(p)
                if tag:
                    tags.add(tag)
        if tags:
            self.existing_rois_detected.emit(sorted(tags))

    def _on_well_clicked(self, well_id: str) -> None:
        w = self._wells.get(well_id)
        if not w:
            return
        self._d_well.setText(w.well_id)
        self._d_bf.setText(str(w.bf_path) if w.bf_path else "—")
        self._d_fl.setText(str(w.fl_path) if w.fl_path else "—")
        if w.roi_paths:
            self._d_roi.setText("\n".join(p.name for p in w.roi_paths))
        else:
            self._d_roi.setText("none detected")
        self._view_bf_btn.setEnabled(w.bf_path is not None)
        self._view_fl_btn.setEnabled(w.fl_path is not None)
        self._view_all_btn.setEnabled(w.bf_path is not None or w.fl_path is not None)
        self._load_btn.setEnabled(True)
        self.well_selected.emit(w)

    def _on_view(self, which: str) -> None:
        wid = self._grid._selected
        w = self._wells.get(wid) if wid else None
        if w:
            self.view_requested.emit(w, which, self._view_proj_combo.currentText())

    def _on_view_all(self) -> None:
        wid = self._grid._selected
        w = self._wells.get(wid) if wid else None
        if w:
            self.overview_requested.emit(w, self._view_proj_combo.currentText())

    def _on_load_sample(self) -> None:
        wid = self._grid._selected
        w = self._wells.get(wid) if wid else None
        if w is None:
            return
        path = w.fl_path or w.bf_path   # prefer FL for channel discovery
        if path is None:
            return
        self._load_btn.setEnabled(False)
        self._load_btn.setText("Loading …")
        self._load_worker = _LoadImageWorker(path)
        self._load_worker.loaded.connect(self._on_image_loaded)
        self._load_worker.error.connect(self._on_load_error)
        self._load_worker.start()

    def _on_load_error(self, msg: str) -> None:
        QMessageBox.critical(self, "Load error", msg)
        self._load_btn.setEnabled(True)
        self._load_btn.setText(
            "Load selected well as sample → configure Channels & ROI tabs"
        )

    def _on_image_loaded(self, image_data) -> None:
        self._image_data = image_data
        self._load_btn.setEnabled(True)
        self._load_btn.setText(
            "Load selected well as sample → configure Channels & ROI tabs"
        )
        self.channels_ready.emit(image_data.channel_names)

    # ── Accessors (same interface as old SetupTab) ───────────────────

    def get_selected_well(self):
        wid = self._grid._selected
        return self._wells.get(wid) if wid else None

    def get_all_wells(self) -> list:
        return list(self._wells.values())

    @property
    def input_dir(self) -> Path | None:
        p = Path(self._folder_edit.text())
        return p if p.is_dir() else None

    @property
    def output_dir(self) -> Path:
        t = self._output_edit.text().strip()
        return Path(t) if t else (self.input_dir or Path(".")) / "output"

    @property
    def experiment(self) -> str:
        return self._exp_edit.text().strip()

    @property
    def image_data(self):
        return self._image_data

    @property
    def image_paths(self) -> list[Path]:
        paths = []
        for w in self._wells.values():
            if w.fl_path:
                paths.append(w.fl_path)
            elif w.bf_path:
                paths.append(w.bf_path)
        return paths


# ──────────────────────────────────────────────────────────────────
# Tab 3 – BF Pipeline  (Ilastik ROI extraction from brightfield)
# ──────────────────────────────────────────────────────────────────

class BFPipelineTab(QWidget):
    """Configure and run the brightfield Ilastik ROI pipeline across all plate wells."""

    def __init__(self, viewer=None, parent=None):
        super().__init__(parent)
        self._viewer = viewer
        self._worker = None
        self._roi_tab = None
        self._build()

    def set_roi_tab(self, roi_tab) -> None:
        """Wire the ROI & Selections tab so "Import to ROI Selections" can add to it."""
        self._roi_tab = roi_tab

    def _build(self) -> None:
        lay = QVBoxLayout(self)

        # ── 1. Ilastik setup ─────────────────────────────────────────
        il_box = QGroupBox("1. Ilastik")
        fl = QFormLayout(il_box)

        self._exe_edit, exe_row = _path_row("ilastik.exe / run_ilastik.sh …", is_dir=False)
        self._exe_edit.setPlaceholderText("leave empty to auto-detect")
        fl.addRow("Executable:", exe_row)

        self._ilp_edit, ilp_row = _path_row("trained .ilp project …", is_dir=False)
        fl.addRow("Project (.ilp):", ilp_row)

        lay.addWidget(il_box)

        # ── 2. Projection (shared/general — not per-class) ────────────
        param_box = QGroupBox("2. Parameters")
        pfl = QFormLayout(param_box)

        self._proj_combo = QComboBox()
        self._proj_combo.addItems(["min", "max", "mean", "sum"])
        self._proj_combo.setToolTip("Min projection works best for transmitted-light BF")
        pfl.addRow("Z-projection:", self._proj_combo)

        self._bf_ch_spin = QSpinBox()
        self._bf_ch_spin.setRange(0, 15); self._bf_ch_spin.setValue(0)
        pfl.addRow("BF channel index:", self._bf_ch_spin)

        lay.addWidget(param_box)

        # ── 2b. Classes ────────────────────────────────────────────────
        # Ilastik's "Simple Segmentation" output is an integer label image
        # (1, 2, 3, ...) — which label means what (hole? background? debris?)
        # is decided once, when the Ilastik project was trained, and is NOT
        # inferred here by comparing shapes across labels. Each class is
        # cleaned up independently: if its pixels form more than one
        # disconnected blob, keep only the single best one (scored by
        # (area/max_area) × circularity, gated by that class's own min
        # area/circularity) — guaranteeing exactly one ROI per class.
        class_box = QGroupBox("2b. Classes  (one ROI saved per class, per well)")
        cbl = QVBoxLayout(class_box)

        n_row = QHBoxLayout()
        n_row.addWidget(QLabel("Number of classes:"))
        self._n_classes_spin = QSpinBox()
        self._n_classes_spin.setRange(1, 9)
        self._n_classes_spin.setValue(2)
        self._n_classes_spin.valueChanged.connect(self._on_n_classes_changed)
        n_row.addWidget(self._n_classes_spin)
        n_row.addStretch()
        cbl.addLayout(n_row)

        self._classes_table = QTableWidget(0, 3)
        self._classes_table.setHorizontalHeaderLabels(["Name", "Min area (px)", "Min circularity"])
        self._classes_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._classes_table.setMinimumHeight(90)
        cbl.addWidget(self._classes_table)

        import_row = QHBoxLayout()
        self._import_to_roi_btn = QPushButton("Import to ROI Selections →")
        self._import_to_roi_btn.setToolTip(
            "Create one ROI & Selections entry per class above, matched "
            "per well — one BF-pipeline run gives you as many ROI "
            "selections as classes configured, with nothing to retype."
        )
        self._import_to_roi_btn.clicked.connect(self._on_import_to_roi_selections)
        import_row.addWidget(self._import_to_roi_btn)
        import_row.addStretch()
        cbl.addLayout(import_row)

        lay.addWidget(class_box)
        self._set_n_class_rows(2, defaults=[("hole", 500, 0.1), ("background", 500, 0.1)])

        # ── 3. Output ────────────────────────────────────────────────
        out_box = QGroupBox("3. Output")
        ofl = QFormLayout(out_box)

        self._save_dir_edit, save_row = _path_row("same folder as images", is_dir=True)
        self._save_dir_edit.setPlaceholderText("leave empty → save alongside each VSI")
        ofl.addRow("Save ROIs to:", save_row)

        lay.addWidget(out_box)

        # ── 4. Run ───────────────────────────────────────────────────
        run_box = QGroupBox("4. Run")
        rl = QVBoxLayout(run_box)

        btn_row = QHBoxLayout()
        self._test_btn = QPushButton("Test on selected well")
        self._test_btn.setToolTip("Run on the well currently selected in the plate grid")
        self._test_btn.clicked.connect(self._on_test)
        self._run_btn = QPushButton("Run on all scanned wells")
        self._run_btn.setStyleSheet("font-weight:bold")
        self._run_btn.clicked.connect(self._on_run)
        self._abort_btn = QPushButton("Abort")
        self._abort_btn.setEnabled(False)
        self._abort_btn.clicked.connect(self._on_abort)
        btn_row.addWidget(self._test_btn)
        btn_row.addWidget(self._run_btn)
        btn_row.addWidget(self._abort_btn)
        rl.addLayout(btn_row)

        self._bar = QProgressBar()
        self._prog_lbl = QLabel("")
        rl.addWidget(self._bar)
        rl.addWidget(self._prog_lbl)

        self._log_edit = QTextEdit()
        self._log_edit.setReadOnly(True)
        self._log_edit.setMaximumHeight(140)
        self._log_edit.setStyleSheet("font-family:monospace; font-size:11px;")
        rl.addWidget(self._log_edit)
        lay.addWidget(run_box)

        lay.addStretch()

        # Restore saved paths/settings and wire auto-save
        self._load_settings()
        self._connect_persistence()

    # ── Persistence ──────────────────────────────────────────────────

    def _load_settings(self) -> None:
        s = QSettings("CorrelativeImaging", "CorrelativeImaging")
        s.beginGroup("bf")
        self._exe_edit.setText(s.value("ilastik_exe", ""))
        self._ilp_edit.setText(s.value("ilp_path", ""))
        idx = self._proj_combo.findText(s.value("z_method", "min"))
        if idx >= 0:
            self._proj_combo.setCurrentIndex(idx)
        self._bf_ch_spin.setValue(int(s.value("bf_channel", 0)))
        self._save_dir_edit.setText(s.value("save_dir", ""))

        n = int(s.value("n_classes", 2))
        defaults = []
        for i in range(1, n + 1):
            defaults.append((
                s.value(f"class_{i}_name", f"class{i}"),
                int(s.value(f"class_{i}_min_area", 500)),
                float(s.value(f"class_{i}_min_circ", 0.1)),
            ))
        s.endGroup()
        self._n_classes_spin.blockSignals(True)
        self._n_classes_spin.setValue(n)
        self._n_classes_spin.blockSignals(False)
        self._set_n_class_rows(n, defaults=defaults)

    def _save_settings(self) -> None:
        s = QSettings("CorrelativeImaging", "CorrelativeImaging")
        s.beginGroup("bf")
        s.setValue("ilastik_exe",    self._exe_edit.text())
        s.setValue("ilp_path",       self._ilp_edit.text())
        s.setValue("z_method",       self._proj_combo.currentText())
        s.setValue("bf_channel",     self._bf_ch_spin.value())
        s.setValue("save_dir",       self._save_dir_edit.text())

        classes = self.get_classes_config()
        s.setValue("n_classes", len(classes))
        for c in classes:
            i = c["index"]
            s.setValue(f"class_{i}_name",      c["name"])
            s.setValue(f"class_{i}_min_area",  c["min_area_px"])
            s.setValue(f"class_{i}_min_circ",  c["min_circularity"])
        s.endGroup()

    def _connect_persistence(self) -> None:
        for w in (self._exe_edit, self._ilp_edit, self._save_dir_edit):
            w.textChanged.connect(self._save_settings)
        self._bf_ch_spin.valueChanged.connect(self._save_settings)
        self._proj_combo.currentTextChanged.connect(self._save_settings)

    # ── Classes table ──────────────────────────────────────────────────

    def _on_n_classes_changed(self, n: int) -> None:
        self._set_n_class_rows(n)
        self._save_settings()

    def _set_n_class_rows(self, n: int, defaults: list[tuple] | None = None) -> None:
        """Resize the class table to exactly `n` rows, preserving already-entered
        values for rows that still exist. `defaults` (name, min_area, min_circ)
        seeds new rows when there's nothing saved yet."""
        table = self._classes_table
        table.blockSignals(True)
        current = self.get_classes_config() if table.rowCount() else []
        table.setRowCount(n)
        for i in range(n):
            table.setVerticalHeaderItem(i, QTableWidgetItem(f"Class {i + 1}"))
            if i < len(current):
                name, min_area, min_circ = current[i]["name"], current[i]["min_area_px"], current[i]["min_circularity"]
            elif defaults and i < len(defaults):
                name, min_area, min_circ = defaults[i]
            else:
                name, min_area, min_circ = f"class{i + 1}", 500, 0.1

            if table.cellWidget(i, 0) is None:
                name_edit = QLineEdit(str(name))
                name_edit.textChanged.connect(self._save_settings)
                table.setCellWidget(i, 0, name_edit)

                area_spin = QSpinBox()
                area_spin.setRange(0, 1000000); area_spin.setValue(int(min_area)); area_spin.setSuffix(" px")
                area_spin.valueChanged.connect(self._save_settings)
                table.setCellWidget(i, 1, area_spin)

                circ_spin = QDoubleSpinBox()
                circ_spin.setRange(0.0, 1.0); circ_spin.setValue(float(min_circ)); circ_spin.setSingleStep(0.05)
                circ_spin.valueChanged.connect(self._save_settings)
                table.setCellWidget(i, 2, circ_spin)
        table.blockSignals(False)

    def get_classes_config(self) -> list[dict]:
        """Return the configured classes as
        ``[{"index": 1, "name": ..., "min_area_px": ..., "min_circularity": ...}, ...]``.
        ``index`` is the Ilastik Simple Segmentation pixel value (1-based, fixed
        by class position) — not inferred from any shape/geometry."""
        table = self._classes_table
        out = []
        for i in range(table.rowCount()):
            name_edit = table.cellWidget(i, 0)
            area_spin = table.cellWidget(i, 1)
            circ_spin = table.cellWidget(i, 2)
            name = (name_edit.text().strip() if name_edit else "") or f"class{i + 1}"
            out.append({
                "index": i + 1,
                "name": name,
                "min_area_px": area_spin.value() if area_spin else 500,
                "min_circularity": circ_spin.value() if circ_spin else 0.1,
            })
        return out

    def _on_import_to_roi_selections(self) -> None:
        if self._roi_tab is None:
            return
        classes = self.get_classes_config()
        existing_names = {s.class_name for s in self._roi_tab.get_selections() if s.source == "well_class"}
        added = 0
        for c in classes:
            if c["name"] in existing_names:
                continue
            self._roi_tab.add_selection(_ROISel(label=c["name"], source="well_class", class_name=c["name"]))
            added += 1
        self._log(
            f"Imported {added} class(es) to ROI & Selections "
            f"({len(classes) - added} already present)."
        )

    # ── Config accessors ─────────────────────────────────────────────

    def get_config(self) -> dict:
        return {
            "ilastik_exe":     self._exe_edit.text().strip(),
            "ilp_path":        self._ilp_edit.text().strip(),
            "z_method":        self._proj_combo.currentText(),
            "bf_channel":      self._bf_ch_spin.value(),
            "classes":         self.get_classes_config(),
            "save_dir":        self._save_dir_edit.text().strip(),
        }

    # ── Run logic ────────────────────────────────────────────────────

    def run_on_wells(self, wells: list, test_mode: bool = False) -> None:
        """Called externally with the list of WellInfo from PlateTab."""
        if not wells:
            QMessageBox.information(self, "No wells", "Scan a plate folder first.")
            return
        cfg = self.get_config()
        if not cfg["ilp_path"] or not Path(cfg["ilp_path"]).exists():
            QMessageBox.warning(self, "No project", "Set a valid Ilastik .ilp project file.")
            return
        # Inject output dir from plate tab so worker can organise its outputs
        plate = self._get_plate_tab()
        if plate and not cfg["save_dir"]:
            cfg["output_dir"] = str(plate.output_dir)

        self._test_btn.setEnabled(False)
        self._run_btn.setEnabled(False)
        self._abort_btn.setEnabled(True)
        self._bar.setRange(0, len(wells))
        self._bar.setValue(0)
        self._log_edit.clear()

        self._worker = _BFWorker(wells, cfg, test_mode=test_mode)
        self._worker.progress.connect(self._on_progress)
        self._worker.log_msg.connect(self._log)
        self._worker.finished.connect(self._on_finished)
        if test_mode:
            self._worker.result_ready.connect(self._show_test_result)
        self._worker.start()

    def _show_test_result(self, ch_vis, seg, masks: dict, well_id: str, pixel_size_um: float = 1.0) -> None:
        if self._viewer is None:
            self._log("[napari] No viewer attached — cannot display result.")
            return
        # Append to whatever's already in the viewer (e.g. the Setup tab's
        # BF + FL + ROI overview) instead of wiping it — only replace this
        # well's own previous test-result layers, so re-testing the same
        # well doesn't pile up duplicates.
        stale_names = {f"BF proj/{well_id}", f"Segmentation/{well_id}",
                       *(f"{kind}/{well_id}" for kind in masks)}
        for layer in list(self._viewer.layers):
            if layer.name in stale_names:
                self._viewer.layers.remove(layer)

        # scale MUST match the already-loaded image's scale (see
        # _add_image_layers / NapariViewer.show_image) — without it, these
        # layers render at raw pixel scale while the loaded sample renders
        # at physical (µm) scale, making them appear far larger/misaligned
        # whenever pixel_size_um != 1.
        scale = [pixel_size_um, pixel_size_um]
        if ch_vis is not None:
            self._viewer.add_image(ch_vis, name=f"BF proj/{well_id}", colormap="gray", scale=scale)
        self._viewer.add_labels(seg, name=f"Segmentation/{well_id}", opacity=0.45, scale=scale)
        for i, (kind, mask) in enumerate(masks.items(), start=1):
            if mask.max() > 0:
                self._viewer.add_labels(
                    mask.astype("int32") * i,
                    name=f"{kind}/{well_id}", opacity=0.5, scale=scale,
                )
        self._log(
            f"  → napari: BF projection + segmentation map + {len(masks)} ROI(s) appended."
        )

    def _get_plate_tab(self):
        p = self.parent()
        while p is not None:
            if hasattr(p, "_plate_tab"):
                return p._plate_tab
            p = p.parent()
        return None

    def _on_test(self) -> None:
        plate = self._get_plate_tab()
        if plate is None:
            QMessageBox.warning(self, "Not connected", "Cannot find plate scan results.")
            return
        well = plate.get_selected_well()
        if well is None:
            QMessageBox.information(self, "No well selected",
                                    "Click a well in the plate grid first.")
            return
        self.run_on_wells([well], test_mode=True)

    def _on_run(self) -> None:
        plate = self._get_plate_tab()
        if plate is None:
            QMessageBox.warning(self, "Not connected", "Cannot find plate scan results.")
            return
        wells = list(plate._wells.values())
        self.run_on_wells(wells, test_mode=False)

    def _on_abort(self) -> None:
        if self._worker:
            self._worker.abort()

    def _on_progress(self, current: int, total: int, msg: str) -> None:
        self._bar.setValue(current)
        self._prog_lbl.setText(f"{current}/{total}  {msg}")

    def _on_finished(self, n_ok: int, n_err: int) -> None:
        self._test_btn.setEnabled(True)
        self._run_btn.setEnabled(True)
        self._abort_btn.setEnabled(False)
        self._log(f"Done — {n_ok} ROIs saved, {n_err} errors.")

        # Refresh the plate grid's ROI badge immediately, without requiring a rescan.
        plate = self._get_plate_tab()
        if plate is not None and self._worker is not None:
            for well in self._worker._wells:
                plate._grid.set_well_status(
                    well.well_id,
                    "complete" if well.is_complete else "bf_only",
                    has_roi=bool(well.roi_paths),
                )

    def _log(self, msg: str) -> None:
        self._log_edit.append(msg)


class _BFWorker(QThread):
    progress     = Signal(int, int, str)
    log_msg      = Signal(str)
    finished     = Signal(int, int)          # n_ok, n_err
    result_ready = Signal(object, object, object, str, float)  # ch_u8, prob(H,W,C), masks{c:mask}, well_id, pixel_size_um

    def __init__(self, wells: list, cfg: dict, test_mode: bool = False):
        super().__init__()
        self._wells     = wells
        self._cfg       = cfg
        self._abort     = False
        self._test_mode = test_mode

    def abort(self) -> None:
        self._abort = True

    def run(self) -> None:
        import subprocess
        import tempfile
        import traceback as _tb

        import h5py
        import numpy as np
        import tifffile

        from correlative_imaging.io import read_image
        from correlative_imaging.pipeline.ilastik import _find_ilastik_exe

        cfg   = self._cfg
        total = len(self._wells)
        n_ok = n_err = 0

        exe = cfg["ilastik_exe"] or _find_ilastik_exe()
        if not exe:
            self.log_msg.emit("ERROR: Ilastik executable not found.")
            self.finished.emit(0, total)
            return

        ops = {"min": np.min, "max": np.max, "mean": np.mean, "sum": np.sum}
        proj_fn = ops.get(cfg["z_method"], np.min)
        ch = cfg["bf_channel"]

        # ── Output directories ────────────────────────────────────────
        if cfg.get("save_dir"):
            out_root = Path(cfg["save_dir"])
        elif cfg.get("output_dir"):
            out_root = Path(cfg["output_dir"]) / "bf_pipeline"
        else:
            out_root = None

        proj_dir = (out_root / "projections")  if out_root else None
        seg_dir  = (out_root / "segmentation") if out_root else None
        roi_dir  = (out_root / "rois")         if out_root else None
        for d in (proj_dir, seg_dir, roi_dir):
            if d:
                d.mkdir(parents=True, exist_ok=True)

        if out_root:
            self.log_msg.emit(f"Output root: {out_root}")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp    = Path(tmpdir)
            in_dir = tmp / "inputs";  in_dir.mkdir()
            out_dir= tmp / "outputs"; out_dir.mkdir()

            # ── Phase 1: load + project + write HDF5 for Ilastik ─────
            # Critical: pass the original 16-bit (or whatever) pixel values —
            # Ilastik was trained on those, not on uint8-normalised data.
            # Shape matches what Fiji's ilastik4ij plugin sends: (t,z,c,y,x).
            self.log_msg.emit("Phase 1/3 — loading and projecting BF images …")
            well_map:   dict[str, object]     = {}
            h5_stems:   list[str]             = []
            h5_in_args: list[str]             = []
            ch_vis_map: dict[str, np.ndarray] = {}  # uint8 display copies
            px_map:     dict[str, float]      = {}  # BF pixel_size_um, for ROI scale sidecars

            for i, well in enumerate(self._wells):
                if self._abort:
                    self.log_msg.emit("Aborted.")
                    self.finished.emit(n_ok, n_err)
                    return

                self.progress.emit(i, total * 3, f"loading {well.well_id}")

                if well.bf_path is None:
                    self.log_msg.emit(f"  ⚠ {well.well_id}: no BF — skipped")
                    n_err += 1
                    continue

                try:
                    img     = read_image(well.bf_path)
                    ch_data = img.data[ch]          # (Z,H,W) or (H,W)
                    if ch_data.ndim == 3:
                        ch_data = proj_fn(ch_data, axis=0)   # → (H,W), original dtype

                    stem = well.bf_path.stem
                    well_map[stem] = well
                    px_map[stem] = img.pixel_size_um

                    # HDF5 for Ilastik — original bit depth, tzcyx shape
                    h5_in = in_dir / f"{stem}.h5"
                    with h5py.File(h5_in, "w") as f:
                        f.create_dataset(
                            "data",
                            data=ch_data[np.newaxis, np.newaxis, np.newaxis],
                        )
                    h5_stems.append(stem)
                    h5_in_args.append(f"{h5_in}/data")

                    # uint8 copy only for display / saved projection TIFF
                    mn, mx = ch_data.min(), ch_data.max()
                    ch_vis = ((ch_data - mn) / (mx - mn) * 255).astype(np.uint8) \
                             if mx > mn else np.zeros_like(ch_data, dtype=np.uint8)
                    ch_vis_map[stem] = ch_vis

                    if proj_dir:
                        proj_out = proj_dir / f"{well.well_id}_proj.tif"
                        tifffile.imwrite(str(proj_out), ch_vis)
                        self.log_msg.emit(
                            f"  ✓ {well.well_id}: projected "
                            f"(dtype={ch_data.dtype}) → {proj_out.name}"
                        )
                    else:
                        self.log_msg.emit(
                            f"  ✓ {well.well_id}: projected (dtype={ch_data.dtype})"
                        )

                except Exception:
                    self.log_msg.emit(f"  ✗ {well.well_id}: {_tb.format_exc()}")
                    n_err += 1

            if not h5_in_args:
                self.log_msg.emit("No images to process.")
                self.finished.emit(n_ok, n_err)
                return

            # ── Phase 2: single Ilastik call ─────────────────────────
            # Same flags as Fiji's ilastik4ij:
            #   --input_axes=tzcyx  (matches our HDF5 shape)
            #   --export_source=Simple Segmentation  (integer labels, no threshold needed)
            self.log_msg.emit(
                f"Phase 2/3 — running Ilastik on {len(h5_in_args)} images …"
            )
            self.progress.emit(total, total * 3, "Ilastik running …")

            out_pattern = str(out_dir / "{nickname}.h5")
            cmd = [
                exe, "--headless",
                f"--project={cfg['ilp_path']}",
                "--export_source=Simple Segmentation",
                "--output_format=hdf5",
                "--output_axis_order=tzcyx",
                "--input_axes=tzcyx",
                "--readonly=1",
                "--output_internal_path=exported_data",
                f"--output_filename_format={out_pattern}",
            ] + h5_in_args

            res = subprocess.run(cmd, capture_output=True, text=True,
                                 timeout=60 + 120 * len(h5_in_args))
            if res.returncode != 0:
                self.log_msg.emit(f"Ilastik failed:\n{res.stderr[-2000:]}")
                self.finished.emit(n_ok, n_err + len(h5_in_args))
                return
            self.log_msg.emit("  Ilastik finished.")

            # Map each well (in order) to its output HDF5.
            # Don't reconstruct the filename — Ilastik's {nickname} derivation
            # varies by version. Scan what's actually there and sort by name.
            output_h5s = sorted(out_dir.glob("*.h5"))
            self.log_msg.emit(
                f"  Output files ({len(output_h5s)}): "
                + ", ".join(f.name for f in output_h5s)
            )
            if len(output_h5s) != len(h5_stems):
                self.log_msg.emit(
                    f"  ERROR: expected {len(h5_stems)} output(s), "
                    f"got {len(output_h5s)} — aborting Phase 3."
                )
                self.finished.emit(n_ok, n_err + len(h5_stems))
                return

            # ── Phase 3: read segmentation labels, extract one ROI per class ──
            # Simple Segmentation → integer label per pixel. Which label means
            # what (hole? background? something else?) is fixed by how the
            # Ilastik project was trained — cfg["classes"] maps each expected
            # label value (class index, 1-based) to a user-given name. Each
            # class is handled independently: if its pixels form more than one
            # disconnected blob, keep only the single best one, scored by
            # (area/max_area) × circularity and gated by that class's own min
            # area/circularity — guaranteeing at most one ROI per class.
            from skimage.measure import label as _cc_label, regionprops as _regionprops

            classes = cfg["classes"]

            self.log_msg.emit(
                f"Phase 3/3 — extracting {len(classes)} class ROI(s) per well …"
            )
            for j, (stem, h5_out) in enumerate(zip(h5_stems, output_h5s)):
                if self._abort:
                    self.log_msg.emit("Aborted.")
                    break

                well = well_map[stem]
                self.progress.emit(total * 2 + j, total * 3, well.well_id)

                try:
                    with h5py.File(h5_out, "r") as f:
                        seg = f["exported_data"][()]
                    seg = np.squeeze(seg).astype(np.int32)   # (H, W)

                    # Save segmentation map for reference
                    if seg_dir:
                        seg_out = seg_dir / f"{well.well_id}_seg.tif"
                        tifffile.imwrite(str(seg_out), seg.astype(np.uint8))
                        self.log_msg.emit(f"    seg map → {seg_out.name}")

                    masks: dict[str, np.ndarray] = {}
                    well_ok = False
                    for class_cfg in classes:
                        idx       = class_cfg["index"]
                        name      = class_cfg["name"]
                        min_area  = class_cfg["min_area_px"]
                        min_circ  = class_cfg["min_circularity"]

                        comps = _regionprops(_cc_label(seg == idx))
                        if not comps:
                            self.log_msg.emit(f"    {name} (class {idx}): no pixels")
                            continue

                        areas = [c.area for c in comps]
                        max_area = max(areas) if max(areas) > 0 else 1.0
                        best = None
                        best_score = -1.0
                        for c in comps:
                            if c.area < min_area or c.perimeter <= 0:
                                continue
                            circ = 4 * np.pi * c.area / c.perimeter ** 2
                            if circ < min_circ:
                                continue
                            # Same formula as the original Fiji pipeline's
                            # best_sub_roi: pick the best part *within this
                            # one class's own selection* — never compared
                            # against other classes.
                            score = (c.area / max_area) * circ
                            if score > best_score:
                                best_score = score
                                best = c
                        if best is None:
                            self.log_msg.emit(
                                f"    ✗ {name} (class {idx}): no component passed "
                                f"area (≥{min_area}px) / circularity (≥{min_circ})"
                            )
                            continue

                        mask = (_cc_label(seg == idx) == best.label).astype(np.uint8)
                        masks[name] = mask

                        fname = _roi_filename_for_well(well, name, ext=".roi")
                        roi_path = (roi_dir / fname) if roi_dir else well.bf_path.parent / fname
                        try:
                            _save_roi(mask, roi_path, pixel_size_um=px_map.get(stem))
                            if roi_path not in well.roi_paths:
                                well.roi_paths.append(roi_path)
                            self.log_msg.emit(
                                f"    {name} → {roi_path.name}  "
                                f"area={best.area:.0f}px score={best_score:.2f}"
                            )
                            well_ok = True
                        except Exception as e:
                            self.log_msg.emit(f"    {name}: {e}")

                    if well_ok:
                        n_ok += 1
                        self.log_msg.emit(f"  ✓ {well.well_id}")
                    else:
                        n_err += 1
                        self.log_msg.emit(f"  ✗ {well.well_id}: no ROIs saved")

                    if self._test_mode:
                        self.result_ready.emit(
                            ch_vis_map.get(stem), seg, masks, well.well_id,
                            px_map.get(stem, 1.0),
                        )

                except Exception:
                    self.log_msg.emit(f"  ✗ {well.well_id}: {_tb.format_exc()}")
                    n_err += 1

        self.progress.emit(total * 3, total * 3, "done")
        self.finished.emit(n_ok, n_err)


def _find_ilastik_exe() -> str | None:
    from correlative_imaging.pipeline.ilastik import _find_ilastik_exe as _f
    return _f()


def _roi_filename_for_well(well, kind: str, ext: str = ".roi") -> str:
    """Build an ROI filename that embeds the well coordinate so
    ``io.plate._WELL_COORD_RE`` can find it on a fresh plate scan.
    """
    return f"_{well.row}{well.col}-{well.field}_{kind}{ext}"


# Matches this module's own BF-pipeline naming convention (see
# _roi_filename_for_well): a leading well coordinate with no other prefix.
_BF_OWNED_ROI_RE = re.compile(r"^_[A-Pa-p]\d{1,2}-\d+_")


def _guess_existing_tag(path: Path) -> str | None:
    """Best-effort guess at a distinguishing tag for a pre-existing
    (non-BF-pipeline) ROI file, used to auto-add a "well_existing" ROI &
    Selections entry after a plate scan.

    Returns ``None`` — skip auto-adding rather than guess wrong — for files
    that look BF-pipeline-owned (leading well coordinate, e.g.
    ``_G10-1_hole.roi``; those are handled by the BF Pipeline tab's own
    "Import to ROI Selections" button instead) or that have no recognizable
    ``<tag>__...`` prefix to use as a tag.
    """
    stem = Path(path).stem
    if _BF_OWNED_ROI_RE.match(stem):
        return None
    if "__" in stem:
        tag = stem.split("__", 1)[0].strip()
        return tag or None
    return None


def _find_well_roi(well, tag: str = "") -> Path | None:
    """Return one of a well's ROI files whose filename contains *tag*
    (case-insensitive), or the first ROI file at all if *tag* is empty.

    No hardcoded notion of "hole"/"background"/"existing" — a BF-pipeline
    class ROI (see ``_roi_filename_for_well``) and a pre-existing project
    ROI are matched the exact same way, by whatever name/tag was configured
    for that selection. Multiple distinct selections (e.g. "class1" vs
    "class2", or "hole" vs "debris") each match their own file per well by
    using a distinct tag, instead of all grabbing whichever file sorts first.
    """
    for p in well.roi_paths:
        if not tag or tag.lower() in Path(p).stem.lower():
            return p
    return None


def _load_roi_mask(path: Path, h: int, w: int, pixel_size_um: float = 1.0) -> np.ndarray | None:
    """Load an ROI file as a boolean (h, w) mask for viewer overlay, reusing
    LoadROI's own format handling. Returns None (logged) on failure rather
    than raising, since this is a best-effort display overlay.

    ``pixel_size_um``: the pixel size of the image this mask will be
    overlaid on — passed through to ``LoadROI``'s scale-safety sidecar
    lookup so a polygon ROI drawn on a differently-scaled source image
    (e.g. brightfield) still lands in the right place here.
    """
    from correlative_imaging.pipeline.segment import LoadROI

    path = Path(path)
    try:
        suffix = path.suffix.lower()
        if suffix == ".roi":
            scale = LoadROI._pixel_scale(path, pixel_size_um)
            return LoadROI._from_imagej(path, h, w, scale=scale)
        if suffix in {".tif", ".tiff", ".png", ".bmp"}:
            return LoadROI._from_image(path, h, w)
    except Exception:
        log.warning("Could not load ROI overlay %s", path, exc_info=True)
    return None


def _save_roi(mask: np.ndarray, path: Path, pixel_size_um: float | None = None) -> None:
    """Save a binary mask as an ImageJ .roi file (bounding-box + polygon outline).

    Raises RuntimeError when the mask is empty so the caller can log it clearly.

    ``pixel_size_um``: the physical pixel size of the image the mask was
    computed on (e.g. the brightfield image, for a BF-pipeline ROI). When
    given, a small sidecar JSON (``<path>.json``) is written alongside the
    ROI recording it plus the mask's own pixel shape — this lets ``LoadROI``
    correctly rescale the polygon if it's later applied to a *different*
    image (e.g. the fluorescence image) with a different pixel size, instead
    of blindly drawing the same raw pixel coordinates onto a differently
    scaled canvas.
    """
    if mask.max() == 0:
        raise RuntimeError(
            "Mask is empty — no foreground pixels after thresholding. "
            "Try lowering the probability threshold or check the foreground "
            "class index in the BF Pipeline tab."
        )
    try:
        import roifile
        from skimage.measure import find_contours
        import numpy as np
        # Pad with zeros so find_contours can trace masks that touch the image edge
        padded = np.pad(mask, 1, mode="constant", constant_values=0)
        contours = find_contours(padded, 0.5)
        if not contours:
            raise RuntimeError(
                "No contour found in mask even after padding — mask may be degenerate."
            )
        # Take the longest contour and subtract the 1-px padding offset
        coords = max(contours, key=len) - 1
        # roifile expects (x, y) i.e. (col, row)
        roi = roifile.ImagejRoi.frompoints(coords[:, ::-1])
        roi.tofile(str(path))
    except ImportError:
        # roifile not installed — fall back to saving as a binary TIFF
        import tifffile
        tiff_path = path.with_suffix(".tif")
        tifffile.imwrite(str(tiff_path), mask)
        log.warning("roifile not installed — saved mask as TIFF: %s", tiff_path.name)
        path = tiff_path

    if pixel_size_um is not None:
        import json as _json
        sidecar = Path(str(path) + ".json")
        sidecar.write_text(_json.dumps({
            "pixel_size_um": pixel_size_um,
            "shape": list(mask.shape),
        }))


def _build_metric_checklist(choices: list[str]) -> QListWidget:
    """Compact checkable list for picking a subset of metric names — used for
    both particle-level and gross/bulk metric selection. All checked by
    default (backward-compatible with the old always-everything behavior)."""
    lst = QListWidget()
    lst.setFixedHeight(110)
    for name in choices:
        item = QListWidgetItem(name)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Checked)
        lst.addItem(item)
    return lst


def _checked_metrics(lst: QListWidget) -> list[str]:
    return [lst.item(i).text() for i in range(lst.count())
            if lst.item(i).checkState() == Qt.Checked]


def _set_checked_metrics(lst: QListWidget, selected: list[str] | None) -> None:
    """selected=None means "all" (backward-compat default)."""
    for i in range(lst.count()):
        item = lst.item(i)
        checked = selected is None or item.text() in selected
        item.setCheckState(Qt.Checked if checked else Qt.Unchecked)


# ──────────────────────────────────────────────────────────────────
# Tab 4 – Channels  (sidebar list + per-channel settings panel)
# ──────────────────────────────────────────────────────────────────

class ChannelPanel(QWidget):
    """Settings panel for one channel: name, preprocessing, segmentation, analysis."""

    name_changed   = Signal(str)
    copy_requested = Signal()   # emitted when user wants to copy settings to all channels

    def __init__(self, ch_index: int, raw_name: str, parent=None):
        super().__init__(parent)
        self._ch_index = ch_index
        self._build(raw_name)

    def _build(self, raw_name: str) -> None:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        lay   = QVBoxLayout(inner)
        lay.setSpacing(6)

        # ── Name ───────────────────────────────────────────────────
        id_box = QGroupBox("Channel identity")
        ifl = QFormLayout(id_box)
        self._name_edit = QLineEdit(raw_name)
        self._name_edit.textChanged.connect(self.name_changed)
        ifl.addRow("Display name:", self._name_edit)
        self._color_combo = QComboBox()
        self._color_combo.addItems(_CHANNEL_COLOR_CHOICES)
        self._color_combo.setCurrentText(_CHANNEL_COLOR_CHOICES[self._ch_index % len(_CHANNEL_COLOR_CHOICES)])
        self._color_combo.setToolTip(
            "Used for the preview display and for diagnostic image export "
            "(Run tab) — not copied by 'Copy settings to all other channels' "
            "since channels usually want distinct colors."
        )
        ifl.addRow("Color:", self._color_combo)
        copy_btn = QPushButton("Copy settings to all other channels")
        copy_btn.clicked.connect(self.copy_requested)
        ifl.addRow(copy_btn)
        lay.addWidget(id_box)

        # ── A. Preprocessing ───────────────────────────────────────
        pre_box = QGroupBox("A.  Preprocessing  (runs once on the full image)")
        pfl = QFormLayout(pre_box)
        pfl.setLabelAlignment(Qt.AlignRight)

        self._bg_en     = QCheckBox("Background subtraction")
        self._bg_en.setChecked(True)
        self._bg_method = QComboBox(); self._bg_method.addItems(["rolling_ball", "tophat"])
        self._bg_radius = QDoubleSpinBox()
        self._bg_radius.setRange(1, 500); self._bg_radius.setValue(50); self._bg_radius.setSuffix(" px")

        self._blur_en    = QCheckBox("Gaussian blur")
        self._blur_en.setChecked(True)
        self._blur_sigma = QDoubleSpinBox()
        self._blur_sigma.setRange(0.1, 50); self._blur_sigma.setValue(2); self._blur_sigma.setSuffix(" px")

        self._norm_en     = QCheckBox("Normalize")
        self._norm_method = QComboBox(); self._norm_method.addItems(["minmax", "percentile"])

        pfl.addRow(self._bg_en)
        pfl.addRow("  Method:",  self._bg_method)
        pfl.addRow("  Radius:",  self._bg_radius)
        pfl.addRow(self._blur_en)
        pfl.addRow("  Sigma:",   self._blur_sigma)
        pfl.addRow(self._norm_en)
        pfl.addRow("  Method:",  self._norm_method)

        self._bg_en.toggled.connect(lambda c: [self._bg_method.setEnabled(c), self._bg_radius.setEnabled(c)])
        self._blur_en.toggled.connect(self._blur_sigma.setEnabled)
        self._norm_en.toggled.connect(self._norm_method.setEnabled)
        lay.addWidget(pre_box)

        # ── B. Segmentation ────────────────────────────────────────
        seg_box = QGroupBox("B.  Segmentation  (runs once, all ROIs share the same mask)")
        sfl = QFormLayout(seg_box)
        sfl.setLabelAlignment(Qt.AlignRight)

        self._thresh = QComboBox(); self._thresh.addItems(["otsu", "li", "yen", "triangle", "isodata"])
        self._z_proj = QComboBox(); self._z_proj.addItems(["max", "mean", "sum"])
        self._min_obj = QSpinBox()
        self._min_obj.setRange(0, 100000); self._min_obj.setValue(50); self._min_obj.setSuffix(" px")
        self._ws_en   = QCheckBox("Watershed split")
        self._ws_en.setChecked(True)
        self._ws_dist = QSpinBox()
        self._ws_dist.setRange(1, 200); self._ws_dist.setValue(5); self._ws_dist.setSuffix(" px")

        sfl.addRow("Threshold:", self._thresh)
        sfl.addRow("Z proj.:",   self._z_proj)
        sfl.addRow("Min obj. size:", self._min_obj)
        sfl.addRow(self._ws_en)
        sfl.addRow("  Min distance:", self._ws_dist)

        self._ws_en.toggled.connect(self._ws_dist.setEnabled)
        lay.addWidget(seg_box)

        # ── C. Particle analysis filters ───────────────────────────
        an_box = QGroupBox("C.  Particle analysis filters  (applied per ROI selection)")
        afl = QFormLayout(an_box)
        afl.setLabelAlignment(Qt.AlignRight)

        self._particle_en = QCheckBox("Run particle analysis")
        self._particle_en.setChecked(True)
        self._particle_en.setToolTip(
            "Segment + measure individual particles (thresholded, filtered by\n"
            "size/circularity below). Turn off to skip particle detection\n"
            "entirely and only run the threshold-free bulk intensity\n"
            "measurement below, for this channel."
        )
        self._min_area = QDoubleSpinBox()
        self._min_area.setRange(0, 1e6); self._min_area.setValue(0.5); self._min_area.setSuffix(" µm²")
        self._max_area = QDoubleSpinBox()
        self._max_area.setRange(0, 1e9); self._max_area.setValue(5000); self._max_area.setSuffix(" µm²")
        self._min_circ = QDoubleSpinBox()
        self._min_circ.setRange(0, 1); self._min_circ.setValue(0); self._min_circ.setSingleStep(0.05)
        self._an_z_proj = QComboBox(); self._an_z_proj.addItems(["max", "mean", "sum"])

        self._particle_metrics = _build_metric_checklist(PARTICLE_METRIC_CHOICES)
        self._particle_metrics.setToolTip(
            "Which per-particle columns to save (unchecked ones are computed\n"
            "for filtering above regardless, just not saved to the results table)."
        )

        afl.addRow(self._particle_en)
        afl.addRow("Min area:", self._min_area)
        afl.addRow("Max area:", self._max_area)
        afl.addRow("Min circularity:", self._min_circ)
        afl.addRow("Z proj.:", self._an_z_proj)
        afl.addRow("Save metrics:", self._particle_metrics)
        lay.addWidget(an_box)

        for w in (self._min_area, self._max_area, self._min_circ, self._an_z_proj, self._particle_metrics):
            self._particle_en.toggled.connect(w.setEnabled)

        # ── D. Gross/bulk metrics (independent of particle detection) ──
        gross_box = QGroupBox("D.  Gross/bulk metrics  (whole selection, no segmentation)")
        gfl = QFormLayout(gross_box)
        gfl.setLabelAlignment(Qt.AlignRight)
        self._gross_metrics = _build_metric_checklist(INTENSITY_METRIC_CHOICES)
        self._gross_metrics.setToolTip(
            "mean/sum/std_intensity and area_px/area_um2 are independently\n"
            "toggleable — e.g. record a selection's area without also\n"
            "measuring its bulk intensity."
        )
        gfl.addRow("Save metrics:", self._gross_metrics)
        lay.addWidget(gross_box)

        lay.addStretch()
        scroll.setWidget(inner)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    def apply_from(self, other: "ChannelPanel") -> None:
        """Copy all settings (except display name) from *other* into this panel."""
        self._bg_en.setChecked(other._bg_en.isChecked())
        self._bg_method.setCurrentText(other._bg_method.currentText())
        self._bg_radius.setValue(other._bg_radius.value())
        self._blur_en.setChecked(other._blur_en.isChecked())
        self._blur_sigma.setValue(other._blur_sigma.value())
        self._norm_en.setChecked(other._norm_en.isChecked())
        self._norm_method.setCurrentText(other._norm_method.currentText())
        self._thresh.setCurrentText(other._thresh.currentText())
        self._z_proj.setCurrentText(other._z_proj.currentText())
        self._min_obj.setValue(other._min_obj.value())
        self._ws_en.setChecked(other._ws_en.isChecked())
        self._ws_dist.setValue(other._ws_dist.value())
        self._particle_en.setChecked(other._particle_en.isChecked())
        self._min_area.setValue(other._min_area.value())
        self._max_area.setValue(other._max_area.value())
        self._min_circ.setValue(other._min_circ.value())
        self._an_z_proj.setCurrentText(other._an_z_proj.currentText())
        _set_checked_metrics(self._particle_metrics, _checked_metrics(other._particle_metrics))
        _set_checked_metrics(self._gross_metrics, _checked_metrics(other._gross_metrics))

    # ── Step generators ────────────────────────────────────────────

    @property
    def display_name(self) -> str:
        t = self._name_edit.text().strip()
        return t if t else f"ch{self._ch_index}"

    @property
    def display_color(self) -> str:
        return self._color_combo.currentText()

    def get_preprocess_steps(self) -> list[dict]:
        steps: list[dict] = []
        if self._bg_en.isChecked():
            steps.append({"type": "BackgroundSubtraction", "channel": self._ch_index,
                          "radius": self._bg_radius.value(), "method": self._bg_method.currentText()})
        if self._blur_en.isChecked():
            steps.append({"type": "GaussianBlur", "channel": self._ch_index,
                          "sigma": self._blur_sigma.value()})
        if self._norm_en.isChecked():
            steps.append({"type": "Normalize", "channel": self._ch_index,
                          "method": self._norm_method.currentText()})
        return steps

    def get_segment_steps(self) -> list[dict]:
        steps = [{"type": "AutoThreshold", "channel": self._ch_index,
                  "method": self._thresh.currentText(), "z_projection": self._z_proj.currentText(),
                  "min_size": self._min_obj.value()}]
        if self._ws_en.isChecked():
            steps.append({"type": "WatershedSplit", "channel": self._ch_index,
                          "min_distance": self._ws_dist.value()})
        return steps

    def get_analysis_steps(self, roi_mask_key: str) -> list[dict]:
        steps: list[dict] = []
        if self._particle_en.isChecked():
            steps.append({"type": "ParticleAnalysis", "channel": self._ch_index,
                          "min_size_um2": self._min_area.value(), "max_size_um2": self._max_area.value(),
                          "min_circularity": self._min_circ.value(), "z_projection": self._an_z_proj.currentText(),
                          "roi_mask": roi_mask_key, "metrics": _checked_metrics(self._particle_metrics)})
        gross_metrics = _checked_metrics(self._gross_metrics)
        if gross_metrics:
            steps.append({"type": "IntensityMeasurement", "channel": self._ch_index,
                          "z_projection": self._an_z_proj.currentText(), "roi_mask": roi_mask_key,
                          "metrics": gross_metrics})
        return steps

    def set_from_steps(self, steps: list[dict]) -> None:
        """Reverse of get_preprocess_steps/get_segment_steps/get_analysis_steps —
        populate this channel's widgets from a loaded pipeline JSON's steps
        for this channel. A step type absent from `steps` is treated as
        disabled (matches how it's built: the step is simply omitted when
        its checkbox is off)."""
        by_type: dict[str, dict] = {}
        for s in steps:
            by_type.setdefault(s["type"], s)  # first occurrence wins (e.g. across ROI selections)

        bg = by_type.get("BackgroundSubtraction")
        self._bg_en.setChecked(bg is not None)
        if bg:
            self._bg_method.setCurrentText(bg.get("method", "rolling_ball"))
            self._bg_radius.setValue(bg.get("radius", 50))

        blur = by_type.get("GaussianBlur")
        self._blur_en.setChecked(blur is not None)
        if blur:
            self._blur_sigma.setValue(blur.get("sigma", 2.0))

        norm = by_type.get("Normalize")
        self._norm_en.setChecked(norm is not None)
        if norm:
            self._norm_method.setCurrentText(norm.get("method", "minmax"))

        thresh = by_type.get("AutoThreshold")
        if thresh:
            self._thresh.setCurrentText(thresh.get("method", "otsu"))
            self._z_proj.setCurrentText(thresh.get("z_projection", "max"))
            self._min_obj.setValue(thresh.get("min_size", 50))

        ws = by_type.get("WatershedSplit")
        self._ws_en.setChecked(ws is not None)
        if ws:
            self._ws_dist.setValue(ws.get("min_distance", 5))

        particle = by_type.get("ParticleAnalysis")
        self._particle_en.setChecked(particle is not None)
        if particle:
            self._min_area.setValue(particle.get("min_size_um2", 0.5))
            self._max_area.setValue(particle.get("max_size_um2", 5000.0))
            self._min_circ.setValue(particle.get("min_circularity", 0.0))
            self._an_z_proj.setCurrentText(particle.get("z_projection", "max"))
            _set_checked_metrics(self._particle_metrics, particle.get("metrics"))

        intensity = by_type.get("IntensityMeasurement")
        _set_checked_metrics(
            self._gross_metrics,
            intensity.get("metrics") if intensity else [],
        )


class ChannelsTab(QWidget):
    """Sidebar: list of channels on the left, settings panel on the right."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._panels: list[ChannelPanel] = []
        self._build()

    def _build(self) -> None:
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(4)

        # Left: channel list
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(4, 8, 0, 4)
        ll.addWidget(QLabel("Channels:"))
        self._ch_list = QListWidget()
        self._ch_list.setFixedWidth(130)
        self._ch_list.currentRowChanged.connect(self._on_row_changed)
        ll.addWidget(self._ch_list)

        # Right: stacked panels
        self._stack = QStackedWidget()
        placeholder = QLabel("Load a sample image in the Setup tab first.")
        placeholder.setAlignment(Qt.AlignCenter)
        self._stack.addWidget(placeholder)

        splitter.addWidget(left)
        splitter.addWidget(self._stack)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        disable_btn = QPushButton("Disable all steps (all channels)")
        disable_btn.clicked.connect(self._disable_all_steps)
        root.addWidget(disable_btn)
        root.addWidget(splitter)

    def _on_row_changed(self, row: int) -> None:
        if 0 <= row < len(self._panels):
            self._stack.setCurrentWidget(self._panels[row])

    def set_channels(self, raw_names: list[str]) -> None:
        for panel in self._panels:
            self._stack.removeWidget(panel)
        self._panels.clear()
        self._ch_list.clear()

        for i, name in enumerate(raw_names):
            panel = ChannelPanel(i, name)
            panel.name_changed.connect(lambda text, idx=i: self._update_label(idx, text))
            panel.copy_requested.connect(lambda p=panel: self._copy_to_all(p))
            self._panels.append(panel)
            self._stack.addWidget(panel)
            self._ch_list.addItem(name)

        if self._panels:
            self._ch_list.setCurrentRow(0)
            self._stack.setCurrentWidget(self._panels[0])

    def _update_label(self, idx: int, text: str) -> None:
        item = self._ch_list.item(idx)
        if item:
            item.setText(text or f"ch{idx}")

    def _copy_to_all(self, source: ChannelPanel) -> None:
        for panel in self._panels:
            if panel is not source:
                panel.apply_from(source)

    def _disable_all_steps(self) -> None:
        for panel in self._panels:
            panel._bg_en.setChecked(False)
            panel._blur_en.setChecked(False)
            panel._norm_en.setChecked(False)
            panel._ws_en.setChecked(False)
            _set_checked_metrics(panel._gross_metrics, [])

    def get_panels(self) -> list[ChannelPanel]:
        return self._panels

    def get_channel_names(self) -> list[str]:
        return [p.display_name for p in self._panels]

    def populate_from_pipeline(self, pl_dict: dict) -> None:
        """Apply a loaded pipeline JSON's steps to each already-existing
        channel panel (set_channels must have been called first — i.e. a
        sample image already loaded, so channel count/names are known)."""
        steps = pl_dict.get("steps", [])
        per_ch_types = {"BackgroundSubtraction", "GaussianBlur", "Normalize",
                         "AutoThreshold", "WatershedSplit", "ParticleAnalysis", "IntensityMeasurement"}
        for i, panel in enumerate(self._panels):
            ch_steps = [s for s in steps if s.get("channel") == i and s.get("type") in per_ch_types]
            panel.set_from_steps(ch_steps)


# ──────────────────────────────────────────────────────────────────
# Tab 4 – ROI & Selections  (master-detail: multiple ROI configs)
# ──────────────────────────────────────────────────────────────────

@dataclass
class _ROISel:
    """Data object holding one ROI selection's configuration."""
    label: str     = "Whole image"
    # "whole" | "auto" | "file" | "well_class" | "well_existing"
    source: str    = "whole"
    channel: int   = 0
    blur: float    = 20.0
    method: str    = "otsu"
    path: str      = ""
    fill_holes: bool = True
    dilation_um: float = 0.0
    # "well_existing" only: restrict matches to files whose stem contains this
    # substring (case-insensitive), e.g. "class1" vs "class2" — required
    # whenever a well can have more than one pre-existing ROI file, otherwise
    # every "well_existing" selection would grab the same (first-found) file.
    existing_tag: str = ""
    # "well_class" only: which BF-pipeline class (by name, see BFPipelineTab's
    # class table) to match — matched the same way as existing_tag, just
    # against this project's own generated ROI files instead of imported ones.
    class_name: str = ""

    # Sources that need a `well` to resolve against, and which tag field
    # each one uses for matching (see _find_well_roi).
    _WELL_DYNAMIC_SOURCES = {"well_class": "class_name", "well_existing": "existing_tag"}

    @property
    def mask_key(self) -> str:
        """Unique key used in context.masks. Empty string = no restriction."""
        if self.source == "whole":
            return ""
        slug = re.sub(r"[^a-z0-9_]", "", self.label.lower().replace(" ", "_"))
        return f"roi_{slug}" if slug else "roi"

    def list_label(self) -> str:
        src = {"whole":        "whole image",
               "auto":         f"auto ch{self.channel}",
               "file":         Path(self.path).name if self.path else "(no file)",
               "well_class":   f"BF-pipeline ROI (per well){f' — {self.class_name}' if self.class_name else ''}",
               "well_existing":f"existing project ROI (per well){f' — {self.existing_tag}' if self.existing_tag else ''}",
               }.get(self.source, self.source)
        return f"{self.label}  [{src}]"

    def get_roi_step(self, well=None) -> dict | None:
        """Build this selection's ROI-extraction step dict.

        ``well`` resolves the two per-well-dynamic sources against that
        well's ``roi_paths`` (see ``_find_well_roi``). It's ``None`` during
        single-image preview, or when a well genuinely has no matching ROI —
        in both cases this returns ``None`` and the caller (see
        ``CorrelativeImagingWidget.build_pipeline_dict``) skips this
        selection's analysis steps entirely rather than silently analyzing
        the whole image under a mask_key that implies it's ROI-restricted.
        """
        if self.source == "whole":
            return None
        if self.source == "auto":
            return {"type": "ExtractROI", "channel": self.channel,
                    "blur_sigma": self.blur, "method": self.method, "roi_name": self.mask_key}
        if self.source == "file" and self.path:
            return {"type": "LoadROI", "path": self.path, "roi_name": self.mask_key}
        if self.source in self._WELL_DYNAMIC_SOURCES and well is not None:
            tag = getattr(self, self._WELL_DYNAMIC_SOURCES[self.source])
            resolved = _find_well_roi(well, tag)
            if resolved is not None:
                return {"type": "LoadROI", "path": str(resolved), "roi_name": self.mask_key}
            log.warning(
                "ROI selection '%s': no ROI matching '%s' found for well %s — skipped",
                self.label, tag or "(any)", getattr(well, "well_id", "?"),
            )
        return None


class ROISelectionsTab(QWidget):
    """Add / remove ROI selections; each is run across all channels."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._sels: list[_ROISel] = []
        self._channel_names: list[str] = []
        self._viewer = None
        self._loading = False   # suppress form-change callbacks while loading
        self._build()
        # Default: one "Whole image" selection
        self._add_sel(_ROISel("Whole image", "whole"))

    # ── Public API ─────────────────────────────────────────────────

    def set_channels(self, names: list[str]) -> None:
        self._channel_names = names
        self._ch_combo.clear()
        self._ch_combo.addItems([f"{i}: {n}" for i, n in enumerate(names)])

    def set_viewer(self, v) -> None:
        self._viewer = v

    def get_selections(self) -> list[_ROISel]:
        return list(self._sels)

    def add_selection(self, sel: _ROISel) -> None:
        """Public entry point for other tabs (e.g. "Import to ROI Selections")."""
        self._add_sel(sel)

    def populate_from_pipeline(self, pl_dict: dict) -> list[str]:
        """Replace all selections with ones reconstructed from a loaded
        pipeline JSON's flat step list. Returns a list of warnings — a JSON
        only records each selection's *resolved* absolute path, not whether
        it was originally a per-well "well_class"/"well_existing" match or a
        genuinely fixed file, so any LoadROI-backed selection is
        reconstructed as a fixed "file" source. If per-well matching is
        wanted, reconfigure it manually after loading.
        """
        steps = pl_dict.get("steps", [])
        groups: dict[str, dict] = {}
        order: list[str] = []
        for s in steps:
            if s["type"] in ("LoadROI", "ExtractROI"):
                key = s["roi_name"]
                groups.setdefault(key, {})["roi_step"] = s
                if key not in order:
                    order.append(key)

        self._sels.clear()
        self._sel_list.clear()
        self._add_sel(_ROISel(label="Whole image", source="whole"))

        warnings: list[str] = []
        for key in order:
            rt = groups[key].get("roi_step")
            if rt is None:
                continue
            label = key[4:] if key.startswith("roi_") else key
            label = label or key
            if rt["type"] == "ExtractROI":
                sel = _ROISel(label=label, source="auto", channel=rt.get("channel", 0),
                               blur=rt.get("blur_sigma", 20.0), method=rt.get("method", "otsu"))
            else:
                sel = _ROISel(label=label, source="file", path=rt.get("path", ""))
                warnings.append(
                    f"'{label}' imported as a fixed file ({rt.get('path', '?')}) — "
                    "not per-well matched. Reconfigure its source manually if you "
                    "need per-well matching (well_class / existing project ROI)."
                )
            self._add_sel(sel)
        return warnings

    def load_detected_rois(self, roi_paths: list) -> None:
        """Replace any existing file-based selections with the detected ROI paths."""
        # Remove existing file-based selections (keep whole/auto ones)
        to_remove = [i for i, s in enumerate(self._sels) if s.source == "file"]
        for i in reversed(to_remove):
            self._sels.pop(i)
            self._sel_list.takeItem(i)
        # Add one selection per detected file
        for path in roi_paths:
            self._add_sel(_ROISel(
                label=Path(path).stem,
                source="file",
                path=str(path),
            ))

    # ── Build ──────────────────────────────────────────────────────

    def _build(self) -> None:
        lay = QVBoxLayout(self)

        # ── List + buttons ─────────────────────────────────────────
        list_box = QGroupBox("Selections  (each is analyzed independently on all channels)")
        lb = QVBoxLayout(list_box)
        self._sel_list = QListWidget()
        self._sel_list.setMaximumHeight(100)
        self._sel_list.currentRowChanged.connect(self._on_row_changed)
        lb.addWidget(self._sel_list)

        btn_row = QHBoxLayout()
        add_btn = QPushButton("+ Add selection")
        add_btn.clicked.connect(self._on_add_clicked)
        rm_btn  = QPushButton("− Remove")
        rm_btn.clicked.connect(self._on_remove_clicked)
        btn_row.addWidget(add_btn)
        btn_row.addWidget(rm_btn)
        lb.addLayout(btn_row)
        lay.addWidget(list_box)

        # ── Settings form (detail panel) ───────────────────────────
        self._detail = QGroupBox("Settings for selected")
        dfl = QFormLayout(self._detail)
        dfl.setLabelAlignment(Qt.AlignRight)

        self._lbl_edit = QLineEdit()
        dfl.addRow("Name:", self._lbl_edit)

        # Source radios
        self._src_grp  = QButtonGroup(self)
        self._rb_whole = QRadioButton("Whole image  (no restriction)")
        self._rb_auto  = QRadioButton("Auto-detect from channel")
        self._rb_file  = QRadioButton("Import from file  (.roi / .tif)")
        self._rb_well_class    = QRadioButton("BF-pipeline ROI  (per well, batch only)")
        self._rb_well_existing = QRadioButton("Existing project ROI  (per well, batch only)")
        self._rb_whole.setChecked(True)
        for rb in [self._rb_whole, self._rb_auto, self._rb_file,
                   self._rb_well_class, self._rb_well_existing]:
            self._src_grp.addButton(rb)
            dfl.addRow(rb)
        note2 = QLabel(
            "The two “per well” sources are resolved separately for each "
            "well during a batch run — always matched to that well specifically, "
            "never a fixed file shared across wells: BF-pipeline ROI is reused "
            "from disk if present, else (re)generated on the fly for whichever "
            "classes are configured in the BF Pipeline tab; existing project "
            "ROI is matched by well coordinate. Use the name/tag field below to "
            "disambiguate if a well has more than one match (e.g. hole vs "
            "background, or class1 vs class2). Any well with no match for a "
            "selection is reported at the end of the batch run, not silently "
            "skipped. During single-image preview these contribute no "
            "restriction."
        )
        note2.setWordWrap(True)
        note2.setStyleSheet("color: gray; font-size: 10px;")
        dfl.addRow(note2)

        self._class_name_edit = QLineEdit()
        self._class_name_edit.setPlaceholderText("e.g. hole — must match a BF Pipeline class name")
        self._class_name_row_label = QLabel("Match class name:")
        dfl.addRow(self._class_name_row_label, self._class_name_edit)
        self._class_name_row_label.setVisible(False)
        self._class_name_edit.setVisible(False)

        self._existing_tag_edit = QLineEdit()
        self._existing_tag_edit.setPlaceholderText("e.g. class1 — leave empty to match any")
        self._existing_tag_row_label = QLabel("Match filename containing:")
        dfl.addRow(self._existing_tag_row_label, self._existing_tag_edit)
        self._existing_tag_row_label.setVisible(False)
        self._existing_tag_edit.setVisible(False)

        # Auto-detect sub-group
        self._auto_grp = QGroupBox("Auto-detect options")
        afl = QFormLayout(self._auto_grp)
        self._ch_combo  = QComboBox()
        self._blur_spin = QDoubleSpinBox()
        self._blur_spin.setRange(1, 500); self._blur_spin.setValue(20); self._blur_spin.setSuffix(" px")
        self._thresh_cb = QComboBox(); self._thresh_cb.addItems(["otsu", "li", "yen", "triangle", "isodata"])
        afl.addRow("Reference channel:", self._ch_combo)
        afl.addRow("Blur sigma:",        self._blur_spin)
        afl.addRow("Threshold method:",  self._thresh_cb)
        self._auto_grp.setVisible(False)
        dfl.addRow(self._auto_grp)

        # File sub-group
        self._file_grp = QGroupBox("File import")
        ffl = QFormLayout(self._file_grp)
        self._path_edit, path_row = _path_row("Select .roi / .tif …", is_dir=False)
        ffl.addRow("ROI file:", path_row)
        note = QLabel("Same file used for all images in batch.")
        note.setWordWrap(True)
        ffl.addRow(note)
        self._file_grp.setVisible(False)
        dfl.addRow(self._file_grp)

        # Post-processing
        self._fill_cb   = QCheckBox("Fill holes in mask")
        self._fill_cb.setChecked(True)
        self._dil_spin  = QDoubleSpinBox()
        self._dil_spin.setRange(-50, 50); self._dil_spin.setValue(0); self._dil_spin.setSuffix(" µm")
        self._dil_spin.setToolTip("Positive = dilate outward,  negative = erode inward")
        dfl.addRow(self._fill_cb)
        dfl.addRow("Dilation / erosion:", self._dil_spin)

        self._detail.setEnabled(False)
        lay.addWidget(self._detail)
        lay.addStretch()

        # Wire visibility toggles
        self._rb_auto.toggled.connect(self._auto_grp.setVisible)
        self._rb_file.toggled.connect(self._file_grp.setVisible)
        self._rb_well_class.toggled.connect(self._class_name_row_label.setVisible)
        self._rb_well_class.toggled.connect(self._class_name_edit.setVisible)
        self._rb_well_existing.toggled.connect(self._existing_tag_row_label.setVisible)
        self._rb_well_existing.toggled.connect(self._existing_tag_edit.setVisible)

        # Wire auto-save on any field change
        for sig in [
            self._lbl_edit.textChanged,
            self._ch_combo.currentIndexChanged,
            self._blur_spin.valueChanged,
            self._thresh_cb.currentIndexChanged,
            self._path_edit.textChanged,
            self._fill_cb.toggled,
            self._dil_spin.valueChanged,
            self._class_name_edit.textChanged,
            self._existing_tag_edit.textChanged,
        ]:
            sig.connect(self._on_form_changed)
        self._src_grp.buttonToggled.connect(self._on_form_changed)

    # ── Slots ──────────────────────────────────────────────────────

    def _on_row_changed(self, row: int) -> None:
        if 0 <= row < len(self._sels):
            self._load_form(self._sels[row])
            self._detail.setEnabled(True)
        else:
            self._detail.setEnabled(False)

    def _on_add_clicked(self) -> None:
        n = len(self._sels) + 1
        self._add_sel(_ROISel(label=f"Selection {n}", source="whole"))

    def _on_remove_clicked(self) -> None:
        row = self._sel_list.currentRow()
        if row < 0:
            return
        if len(self._sels) == 1:
            QMessageBox.information(self, "Cannot remove", "At least one selection is required.")
            return
        self._sels.pop(row)
        self._sel_list.takeItem(row)
        self._sel_list.setCurrentRow(min(row, len(self._sels) - 1))

    def _on_form_changed(self, *_) -> None:
        if self._loading:
            return
        row = self._sel_list.currentRow()
        if 0 <= row < len(self._sels):
            self._save_form(self._sels[row])
            self._sel_list.item(row).setText(self._sels[row].list_label())

    # ── Internal helpers ───────────────────────────────────────────

    def _add_sel(self, sel: _ROISel) -> None:
        self._sels.append(sel)
        self._sel_list.addItem(sel.list_label())
        self._sel_list.setCurrentRow(len(self._sels) - 1)

    def _load_form(self, sel: _ROISel) -> None:
        self._loading = True
        self._lbl_edit.setText(sel.label)
        rb_map = {
            "whole": self._rb_whole, "auto": self._rb_auto, "file": self._rb_file,
            "well_class": self._rb_well_class, "well_existing": self._rb_well_existing,
        }
        rb_map.get(sel.source, self._rb_whole).setChecked(True)
        self._auto_grp.setVisible(sel.source == "auto")
        self._file_grp.setVisible(sel.source == "file")
        self._class_name_row_label.setVisible(sel.source == "well_class")
        self._class_name_edit.setVisible(sel.source == "well_class")
        self._existing_tag_row_label.setVisible(sel.source == "well_existing")
        self._existing_tag_edit.setVisible(sel.source == "well_existing")
        idx = min(sel.channel, self._ch_combo.count() - 1) if self._ch_combo.count() else 0
        self._ch_combo.setCurrentIndex(idx)
        self._blur_spin.setValue(sel.blur)
        self._thresh_cb.setCurrentText(sel.method)
        self._path_edit.setText(sel.path)
        self._fill_cb.setChecked(sel.fill_holes)
        self._dil_spin.setValue(sel.dilation_um)
        self._class_name_edit.setText(sel.class_name)
        self._existing_tag_edit.setText(sel.existing_tag)
        self._loading = False

    def _save_form(self, sel: _ROISel) -> None:
        sel.label  = self._lbl_edit.text().strip() or sel.label
        sel.source = ("whole"         if self._rb_whole.isChecked() else
                      "auto"          if self._rb_auto.isChecked()  else
                      "well_class"    if self._rb_well_class.isChecked() else
                      "well_existing" if self._rb_well_existing.isChecked() else
                      "file")
        sel.channel    = self._ch_combo.currentIndex()
        sel.blur       = self._blur_spin.value()
        sel.method     = self._thresh_cb.currentText()
        sel.path       = self._path_edit.text().strip()
        sel.fill_holes = self._fill_cb.isChecked()
        sel.dilation_um = self._dil_spin.value()
        sel.class_name = self._class_name_edit.text().strip()
        sel.existing_tag = self._existing_tag_edit.text().strip()


# ──────────────────────────────────────────────────────────────────
# Tab 5 – Colocalization  (channel-pair combinations)
# ──────────────────────────────────────────────────────────────────

class CombineTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._channel_names: list[str] = []
        self._build()

    def set_channels(self, names: list[str]) -> None:
        self._channel_names = names
        for row in range(self._table.rowCount()):
            for col in [0, 1]:
                combo = self._table.cellWidget(row, col)
                if isinstance(combo, QComboBox):
                    prev = combo.currentIndex()
                    combo.clear()
                    combo.addItems([f"{i}: {n}" for i, n in enumerate(names)])
                    combo.setCurrentIndex(min(prev, len(names) - 1))

    def _build(self) -> None:
        lay = QVBoxLayout(self)
        box = QGroupBox("Colocalization pairs")
        cl  = QVBoxLayout(box)

        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["Primary ch.", "Secondary ch.", "Dilation (µm)", "Z proj."])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._table.setMinimumHeight(120)
        cl.addWidget(self._table)

        btn_row = QHBoxLayout()
        add_btn = QPushButton("+ Add pair")
        add_btn.clicked.connect(self._add_pair)
        rm_btn  = QPushButton("− Remove selected")
        rm_btn.clicked.connect(self._remove_pair)
        btn_row.addWidget(add_btn)
        btn_row.addWidget(rm_btn)
        cl.addLayout(btn_row)
        lay.addWidget(box)

        note = QLabel(
            "Each pair computes Manders M1 & M2, Pearson r, 90° rotation control, "
            "and per-particle overlap fraction.\n"
            "Colocalization is run independently for each ROI selection defined in the "
            "ROI & Selections tab. Results are tagged with the ROI name in the database."
        )
        note.setWordWrap(True)
        lay.addWidget(note)
        lay.addStretch()

    def _add_pair(self) -> None:
        r      = self._table.rowCount()
        labels = [f"{i}: {n}" for i, n in enumerate(self._channel_names)] or ["0", "1"]
        self._table.insertRow(r)
        primary   = QComboBox(); primary.addItems(labels)
        secondary = QComboBox(); secondary.addItems(labels); secondary.setCurrentIndex(min(1, len(labels) - 1))
        dilation  = QDoubleSpinBox(); dilation.setRange(0, 20); dilation.setValue(0.5); dilation.setSuffix(" µm")
        z_proj    = QComboBox(); z_proj.addItems(["max", "mean", "sum"])
        self._table.setCellWidget(r, 0, primary)
        self._table.setCellWidget(r, 1, secondary)
        self._table.setCellWidget(r, 2, dilation)
        self._table.setCellWidget(r, 3, z_proj)

    def _remove_pair(self) -> None:
        row = self._table.currentRow()
        if row >= 0:
            self._table.removeRow(row)

    def auto_populate(self, channel_names: list[str]) -> None:
        """Default to every pairwise combination of the given channels —
        e.g. 3 channels -> (0,1), (0,2), (1,2) — not just consecutive pairs."""
        from itertools import combinations

        self._table.setRowCount(0)
        for i, j in combinations(range(len(channel_names)), 2):
            self._add_pair()
            n = self._table.rowCount() - 1
            for col, idx in [(0, i), (1, j)]:
                c = self._table.cellWidget(n, col)
                if isinstance(c, QComboBox):
                    c.setCurrentIndex(idx)

    def get_coloc_steps(self, roi_mask_key: str = "") -> list[dict]:
        steps = []
        for row in range(self._table.rowCount()):
            p  = self._table.cellWidget(row, 0)
            s  = self._table.cellWidget(row, 1)
            d  = self._table.cellWidget(row, 2)
            zp = self._table.cellWidget(row, 3)
            if isinstance(p, QComboBox) and isinstance(s, QComboBox):
                steps.append({"type": "ColocalizationAnalysis",
                               "primary_channel":   p.currentIndex(),
                               "secondary_channel": s.currentIndex(),
                               "dilation_um":       d.value() if isinstance(d, QDoubleSpinBox) else 0.5,
                               "z_projection":      zp.currentText() if isinstance(zp, QComboBox) else "max",
                               "roi_mask":          roi_mask_key})
        return steps

    def populate_from_pipeline(self, pl_dict: dict) -> None:
        """Rebuild the pair table from a loaded pipeline JSON — dedupes
        across ROI selections (the same pair repeats once per selection in
        the flat step list) since primary/secondary/dilation/z-projection
        are shared config, not per-selection."""
        steps = pl_dict.get("steps", [])
        self._table.setRowCount(0)
        seen: set[tuple] = set()
        for s in steps:
            if s.get("type") != "ColocalizationAnalysis":
                continue
            key = (s.get("primary_channel"), s.get("secondary_channel"))
            if key in seen:
                continue
            seen.add(key)
            self._add_pair()
            n = self._table.rowCount() - 1
            p, sec, d, zp = (self._table.cellWidget(n, c) for c in range(4))
            if isinstance(p, QComboBox):
                p.setCurrentIndex(s.get("primary_channel", 0))
            if isinstance(sec, QComboBox):
                sec.setCurrentIndex(s.get("secondary_channel", 0))
            if isinstance(d, QDoubleSpinBox):
                d.setValue(s.get("dilation_um", 0.5))
            if isinstance(zp, QComboBox):
                zp.setCurrentText(s.get("z_projection", "max"))


def _metrics_suffix(selected: list[str] | None, all_choices: list[str]) -> str:
    """", metrics=..." suffix for _step_repr — empty when all metrics are
    saved (the common case), so the summary stays uncluttered."""
    if selected is None or set(selected) == set(all_choices):
        return ""
    if not selected:
        return ", metrics=none"
    return f", metrics={','.join(selected)}"


def _step_repr(s: dict) -> str:
    """One-line human-readable summary of a single pipeline step dict."""
    t = s.get("type", "?")
    if t == "BackgroundSubtraction":
        return f"BackgroundSubtraction({s.get('method')}, r={s.get('radius')}px)"
    if t == "GaussianBlur":
        return f"GaussianBlur(sigma={s.get('sigma')})"
    if t == "Normalize":
        return f"Normalize({s.get('method')})"
    if t == "AutoThreshold":
        return f"AutoThreshold({s.get('method')}, {s.get('z_projection')}, min_size={s.get('min_size')}px)"
    if t == "WatershedSplit":
        return f"WatershedSplit(min_dist={s.get('min_distance')}px)"
    if t == "ParticleAnalysis":
        return (f"ParticleAnalysis(ch{s.get('channel')}, area {s.get('min_size_um2')}-{s.get('max_size_um2')}µm², "
                f"circ≥{s.get('min_circularity')}, {s.get('z_projection')}"
                f"{_metrics_suffix(s.get('metrics'), PARTICLE_METRIC_CHOICES)})")
    if t == "IntensityMeasurement":
        return (f"IntensityMeasurement(ch{s.get('channel')}, {s.get('z_projection')}"
                f"{_metrics_suffix(s.get('metrics'), INTENSITY_METRIC_CHOICES)})")
    if t == "ColocalizationAnalysis":
        return (f"ColocalizationAnalysis(ch{s.get('primary_channel')}↔ch{s.get('secondary_channel')}, "
                f"dilation={s.get('dilation_um')}µm, {s.get('z_projection')})")
    if t == "LoadROI":
        return f"LoadROI({s.get('path')})"
    if t == "ExtractROI":
        return f"ExtractROI(ch{s.get('channel')}, blur={s.get('blur_sigma')}, {s.get('method')})"
    return f"{t}(" + ", ".join(f"{k}={v}" for k, v in s.items() if k not in ("type", "channel")) + ")"


def _format_pipeline_summary(pl_dict: dict, well=None, bf_cfg: dict | None = None,
                              missing: list[str] | None = None) -> str:
    """Render a flat pipeline-step dict (see build_pipeline_dict /
    make_well_pipeline_dict_fn) as a readable, sectioned summary — the
    same pipeline that actually runs for `well` (or the bare preview
    pipeline if `well` is None), not a re-derived approximation.
    """
    steps = pl_dict.get("steps", [])
    lines = [f"Pipeline: {pl_dict.get('name', 'pipeline')}"]
    if well is not None:
        lines.append(f"Well: {getattr(well, 'well_id', '?')}  — exactly what batch runs for this well.")
    else:
        lines.append("No well selected — whole-image pipeline (per-well ROI selections not resolved).")
    lines.append("")

    pre_seg_types = {"BackgroundSubtraction", "GaussianBlur", "Normalize", "AutoThreshold", "WatershedSplit"}
    pre_seg = [s for s in steps if s["type"] in pre_seg_types]
    rest    = [s for s in steps if s["type"] not in pre_seg_types]

    if bf_cfg and bf_cfg.get("classes"):
        lines.append("=== BF-Pipeline / Ilastik ===")
        lines.append(f"Project: {bf_cfg.get('ilp_path') or '(not set)'}")
        for c in bf_cfg["classes"]:
            status = ""
            if well is not None:
                resolved = _find_well_roi(well, c["name"])
                status = "  →  " + (f"reused: {resolved.name}" if resolved else "MISSING, would be generated")
            lines.append(f"  class {c['index']} \"{c['name']}\" "
                          f"(min area {c['min_area_px']}px, min circ {c['min_circularity']:.2f}){status}")
        lines.append("")

    if pre_seg:
        lines.append("=== Preprocessing & segmentation (per channel) ===")
        by_ch: dict = {}
        for s in pre_seg:
            by_ch.setdefault(s.get("channel"), []).append(s)
        for ch in sorted(by_ch, key=lambda x: (x is None, x)):
            desc = " → ".join(_step_repr(s) for s in by_ch[ch])
            lines.append(f"  ch{ch}: {desc}")
        lines.append("")

    groups: dict = {}
    order: list = []
    for s in rest:
        if s["type"] in ("LoadROI", "ExtractROI"):
            key = s["roi_name"]
            g = groups.setdefault(key, {"roi_step": None, "steps": []})
            g["roi_step"] = s
            if key not in order:
                order.append(key)
        else:
            key = s.get("roi_mask", "")
            g = groups.setdefault(key, {"roi_step": None, "steps": []})
            if key not in order:
                order.append(key)
            g["steps"].append(s)

    lines.append("=== ROI selections ===")
    if not order:
        lines.append("  (none)")
    for key in order:
        g = groups[key]
        lines.append(f"[{key or 'whole image'}]")
        if g["roi_step"] is not None:
            lines.append(f"    {_step_repr(g['roi_step'])}")
        for s in g["steps"]:
            lines.append(f"    {_step_repr(s)}")
        lines.append("")

    if missing:
        lines.append("=== NOT matched for this well (skipped) ===")
        for m in missing:
            lines.append(f"  ⚠ {m}")
        lines.append("")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────
# Tab 7 – Pipeline Summary  (read-only "what actually runs" view +
# manual step builder)
# ──────────────────────────────────────────────────────────────────

class AdvancedPipelineTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._steps: list[dict] = []
        self._enabled: list[bool] = []   # parallel to _steps; False = skipped at run time
        self._get_guided_pipeline = None   # set by CorrelativeImagingWidget
        self._get_setup = None                    # -> PlateTab
        self._make_well_pipeline_dict_fn = None   # -> (fn, needs_bf)
        self._get_bf_config = None                # -> BFPipelineTab.get_config()
        self._build()

    def _build(self) -> None:
        # Wrapped in a scroll area so the editor's step list/splitter never
        # pushes the bottom controls (Save JSON, etc.) off the visible tab
        # when the window/dock is shorter than this tab's natural content
        # height.
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        content = QWidget()
        lay = QVBoxLayout(content)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        outer.addWidget(scroll)

        summary_box = QGroupBox("What actually runs")
        sl = QVBoxLayout(summary_box)
        well_row = QHBoxLayout()
        well_row.addWidget(QLabel("Well:"))
        self._summary_well_combo = QComboBox()
        self._summary_well_combo.addItem("(none — whole-image preview)")
        well_row.addWidget(self._summary_well_combo)
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._on_refresh_summary)
        well_row.addWidget(refresh_btn)
        well_row.addStretch()
        sl.addLayout(well_row)
        self._summary_view = QTextEdit()
        self._summary_view.setReadOnly(True)
        self._summary_view.setStyleSheet("font-family:monospace; font-size:11px;")
        self._summary_view.setMinimumHeight(220)
        sl.addWidget(self._summary_view)
        lay.addWidget(summary_box)

        editor_box = QGroupBox("Manual step editor (advanced — builds a separate, independent pipeline)")
        ebl = QVBoxLayout(editor_box)

        top = QHBoxLayout()
        imp_btn  = QPushButton("Import from guided tabs")
        imp_btn.clicked.connect(self._import_guided)
        load_btn = QPushButton("Load JSON …")
        load_btn.clicked.connect(self._load_json)
        top.addWidget(imp_btn)
        top.addWidget(load_btn)
        ebl.addLayout(top)

        splitter = QSplitter(Qt.Horizontal)

        left = QWidget(); ll = QVBoxLayout(left); ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(QLabel("Steps (check to enable):"))
        self._step_list = QListWidget()
        self._step_list.currentRowChanged.connect(self._on_selected)
        self._step_list.itemChanged.connect(self._on_item_changed)
        ll.addWidget(self._step_list)
        btn_row = QHBoxLayout()
        for label, fn in [("↑", lambda: self._move(-1)), ("↓", lambda: self._move(1)), ("✕", self._remove), ("✕ All", self._clear_all)]:
            b = QPushButton(label); b.clicked.connect(fn); btn_row.addWidget(b)
        ll.addLayout(btn_row)
        en_row = QHBoxLayout()
        en_btn  = QPushButton("✓ Enable all"); en_btn.clicked.connect(lambda: self._set_all_enabled(True))
        dis_btn = QPushButton("○ Disable all"); dis_btn.clicked.connect(lambda: self._set_all_enabled(False))
        en_row.addWidget(en_btn); en_row.addWidget(dis_btn)
        ll.addLayout(en_row)
        splitter.addWidget(left)

        right = QWidget(); rl = QVBoxLayout(right); rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(QLabel("Add / edit step:"))
        tr = QHBoxLayout(); tr.addWidget(QLabel("Type:"))
        self._type_combo = QComboBox(); self._type_combo.addItems(sorted(_STEP_FORMS.keys()))
        self._type_combo.currentTextChanged.connect(self._refresh_form)
        tr.addWidget(self._type_combo); rl.addLayout(tr)
        self._form_container = QWidget()
        self._form_layout = QFormLayout(self._form_container)
        self._form_widgets: dict[str, QWidget] = {}
        sc = QScrollArea(); sc.setWidgetResizable(True); sc.setWidget(self._form_container)
        rl.addWidget(sc)
        add_b = QPushButton("Add step ↓"); add_b.clicked.connect(self._add_step); rl.addWidget(add_b)
        splitter.addWidget(right)
        ebl.addWidget(splitter)

        bot = QHBoxLayout()
        bot.addWidget(QLabel("Name:"))
        self._name_edit = QLineEdit("pipeline")
        bot.addWidget(self._name_edit)
        save_btn = QPushButton("Save JSON …"); save_btn.clicked.connect(self._save_json); bot.addWidget(save_btn)
        ebl.addLayout(bot)
        lay.addWidget(editor_box)
        self._refresh_form(self._type_combo.currentText())

    # ── Pipeline Summary (read-only, batch-equivalent view) ───────────

    def refresh_well_list(self) -> None:
        """Repopulate the well combo from the current plate scan. Call this
        whenever the plate is (re)scanned."""
        self._summary_well_combo.blockSignals(True)
        current = self._summary_well_combo.currentText()
        self._summary_well_combo.clear()
        self._summary_well_combo.addItem("(none — whole-image preview)")
        setup = self._get_setup() if self._get_setup else None
        wells = getattr(setup, "_wells", {}) if setup is not None else {}
        for well_id in sorted(wells):
            self._summary_well_combo.addItem(well_id)
        idx = self._summary_well_combo.findText(current)
        self._summary_well_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self._summary_well_combo.blockSignals(False)

    def _on_refresh_summary(self) -> None:
        self.refresh_well_list()
        well = None
        well_id = self._summary_well_combo.currentText()
        setup = self._get_setup() if self._get_setup else None
        wells = getattr(setup, "_wells", {}) if setup is not None else {}
        well = wells.get(well_id)

        bf_cfg = self._get_bf_config() if self._get_bf_config else None

        if well is not None and self._make_well_pipeline_dict_fn is not None:
            pipeline_dict_fn, _needs_bf = self._make_well_pipeline_dict_fn()
            pl_dict, missing = pipeline_dict_fn(well)
        else:
            pl_dict = self._get_guided_pipeline() if self._get_guided_pipeline else {"steps": []}
            missing = []

        self._summary_view.setPlainText(
            _format_pipeline_summary(pl_dict, well=well, bf_cfg=bf_cfg, missing=missing)
        )

    def _clear_form(self) -> None:
        while self._form_layout.rowCount(): self._form_layout.removeRow(0)
        self._form_widgets.clear()

    def _refresh_form(self, step_type: str) -> None:
        self._clear_form()
        for param, wtype, label, default, extra in _STEP_FORMS.get(step_type, []):
            w = _make_widget(wtype, default, extra)
            self._form_widgets[param] = w
            self._form_layout.addRow(label + ":", w)

    def _read_form(self) -> dict:
        st = self._type_combo.currentText()
        d: dict = {"type": st}
        for param, wtype, *_ in _STEP_FORMS.get(st, []):
            d[param] = _read_widget(self._form_widgets[param], wtype)
        return d

    def _add_step(self) -> None:
        self._steps.append(self._read_form())
        self._enabled.append(True)
        self._refresh_list()

    def _remove(self) -> None:
        r = self._step_list.currentRow()
        if 0 <= r < len(self._steps):
            self._steps.pop(r); self._enabled.pop(r); self._refresh_list()

    def _clear_all(self) -> None:
        self._steps.clear(); self._enabled.clear(); self._refresh_list()

    def _move(self, delta: int) -> None:
        r = self._step_list.currentRow(); n = r + delta
        if 0 <= n < len(self._steps):
            self._steps[r], self._steps[n] = self._steps[n], self._steps[r]
            self._enabled[r], self._enabled[n] = self._enabled[n], self._enabled[r]
            self._refresh_list(); self._step_list.setCurrentRow(n)

    def _on_item_changed(self, item) -> None:
        from qtpy.QtCore import Qt as _Qt
        row = self._step_list.row(item)
        if 0 <= row < len(self._enabled):
            self._enabled[row] = (item.checkState() == _Qt.Checked)

    def _set_all_enabled(self, state: bool) -> None:
        self._enabled = [state] * len(self._steps)
        self._refresh_list()

    def _on_selected(self, row: int) -> None:
        if 0 <= row < len(self._steps):
            s = self._steps[row]
            self._type_combo.setCurrentText(s["type"]); self._refresh_form(s["type"])
            for param, wtype, *_ in _STEP_FORMS.get(s["type"], []):
                if param in s and param in self._form_widgets:
                    _set_widget(self._form_widgets[param], wtype, s[param])

    def _refresh_list(self) -> None:
        from qtpy.QtCore import Qt as _Qt
        self._step_list.blockSignals(True)
        self._step_list.clear()
        for i, s in enumerate(self._steps):
            params = "  ".join(f"{k}={v}" for k, v in s.items() if k != "type")
            item = QListWidgetItem(f"{i+1}. {s['type']}  {params}")
            item.setFlags(item.flags() | _Qt.ItemIsUserCheckable)
            en = self._enabled[i] if i < len(self._enabled) else True
            item.setCheckState(_Qt.Checked if en else _Qt.Unchecked)
            self._step_list.addItem(item)
        self._step_list.blockSignals(False)

    def _import_guided(self) -> None:
        if self._get_guided_pipeline is None:
            QMessageBox.information(self, "Not available", "No guided pipeline connected."); return
        pl = self._get_guided_pipeline()
        self._steps = pl.get("steps", [])
        self._enabled = [True] * len(self._steps)
        self._name_edit.setText(pl.get("name", "pipeline"))
        self._refresh_list()

    def _save_json(self) -> None:
        if not self._steps: QMessageBox.information(self, "Empty", "Add steps first."); return
        path, _ = QFileDialog.getSaveFileName(self, "Save pipeline JSON", "", "JSON (*.json)")
        if path:
            Path(path).write_text(json.dumps({"name": self._name_edit.text(), "steps": self._steps}, indent=2))

    def _load_json(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load pipeline JSON", "", "JSON (*.json)")
        if path:
            data = json.loads(Path(path).read_text())
            self._name_edit.setText(data.get("name", "pipeline"))
            self._steps = data.get("steps", [])
            self._enabled = [True] * len(self._steps)
            self._refresh_list()

    def get_pipeline_dict(self) -> dict:
        active = [s for s, en in zip(self._steps, self._enabled) if en]
        return {"name": self._name_edit.text(), "steps": active}

    def set_channel_names(self, names: list[str]) -> None:
        pass   # kept for API compatibility


# ──────────────────────────────────────────────────────────────────
# Tab 6 – Run
# ──────────────────────────────────────────────────────────────────

class RunTab(QWidget):
    def __init__(self, get_setup, get_pipeline_dict, get_bf_config=None,
                 make_well_pipeline_dict_fn=None, napari_viewer=None,
                 get_channel_colors=None, parent=None):
        super().__init__(parent)
        self._get_setup         = get_setup
        self._get_pipeline_dict = get_pipeline_dict
        self._get_bf_config     = get_bf_config
        self._make_well_pipeline_dict_fn = make_well_pipeline_dict_fn
        self._get_channel_colors = get_channel_colors  # () -> list[str], one per channel
        self._viewer            = napari_viewer
        self._worker: _BatchWorker | _WellBatchWorker | None = None
        self._current_log_path: Path | None = None
        self._build()

    def _build(self) -> None:
        lay = QVBoxLayout(self)

        prev_box = QGroupBox("Preview on sample image")
        pl = QVBoxLayout(prev_box)
        self._preview_btn = QPushButton("Run pipeline on sample image (shows in napari)")
        self._preview_btn.clicked.connect(self._on_preview)
        pl.addWidget(self._preview_btn)
        layer_row = QHBoxLayout()
        show_all_btn = QPushButton("Show all layers")
        show_all_btn.clicked.connect(lambda: self._set_all_layers_visible(True))
        hide_all_btn = QPushButton("Hide all layers")
        hide_all_btn.clicked.connect(lambda: self._set_all_layers_visible(False))
        layer_row.addWidget(show_all_btn)
        layer_row.addWidget(hide_all_btn)
        pl.addLayout(layer_row)

        gamma_row = QHBoxLayout()
        gamma_row.addWidget(QLabel("Gamma (display):"))
        self._gamma_slider = QSlider(Qt.Horizontal)
        self._gamma_slider.setRange(10, 300)   # represents 0.10 – 3.00
        self._gamma_slider.setValue(100)        # 1.00 = no correction
        self._gamma_slider.setTickInterval(10)
        self._gamma_lbl = QLabel("1.00")
        self._gamma_lbl.setFixedWidth(32)
        self._gamma_slider.valueChanged.connect(self._on_gamma_changed)
        gamma_row.addWidget(self._gamma_slider)
        gamma_row.addWidget(self._gamma_lbl)
        pl.addLayout(gamma_row)

        contrast_row = QHBoxLayout()
        contrast_row.addWidget(QLabel("Contrast min/max:"))
        self._contrast_min = QDoubleSpinBox()
        self._contrast_max = QDoubleSpinBox()
        for spin in (self._contrast_min, self._contrast_max):
            spin.setRange(-1e9, 1e9)
            spin.setDecimals(2)
        self._contrast_max.setValue(1.0)
        auto_contrast_btn = QPushButton("Auto")
        auto_contrast_btn.setToolTip(
            "Reset to a 1st–99.5th percentile stretch, computed from the "
            "current layer's data (same default used when layers are added)."
        )
        auto_contrast_btn.clicked.connect(self._on_auto_contrast)
        apply_contrast_btn = QPushButton("Apply")
        apply_contrast_btn.clicked.connect(self._on_apply_contrast)
        contrast_row.addWidget(self._contrast_min)
        contrast_row.addWidget(self._contrast_max)
        contrast_row.addWidget(auto_contrast_btn)
        contrast_row.addWidget(apply_contrast_btn)
        pl.addLayout(contrast_row)
        lay.addWidget(prev_box)

        # No manual "Save/Show JSON" here anymore — every batch run now
        # auto-saves its pipeline JSON (and log) next to the database,
        # named after the experiment + run start time (see _on_run). The
        # Setup tab and Pipeline Summary tab cover inspecting/reusing it.

        batch_box = QGroupBox("Batch run")
        bl = QVBoxLayout(batch_box)

        par_row = QHBoxLayout()
        self._parallel_cb = QCheckBox("Parallel batch (experimental)")
        self._parallel_cb.setToolTip(
            "Runs multiple wells' read + pipeline concurrently in a thread "
            "pool instead of one at a time. Off by default — sequential "
            "behavior (the well-tested default) is unchanged either way. "
            "Database writes always happen one at a time regardless."
        )
        self._parallel_cb.toggled.connect(self._on_parallel_toggled)
        self._worker_spin = QSpinBox()
        self._worker_spin.setRange(2, 32)
        self._worker_spin.setValue(4)
        self._worker_spin.setEnabled(False)
        self._worker_spin.setPrefix("workers: ")
        par_row.addWidget(self._parallel_cb)
        par_row.addWidget(self._worker_spin)
        par_row.addStretch()
        bl.addLayout(par_row)

        self._force_regen_cb = QCheckBox("Force regenerate BF-pipeline ROI(s) (ignore existing files)")
        self._force_regen_cb.setToolTip(
            "Off (default): a well's BF-pipeline ROI(s) are reused from disk "
            "if already present, only generated for wells missing one. "
            "On: every well's ROI(s) are regenerated from scratch this run, "
            "overwriting whatever's already there."
        )
        bl.addWidget(self._force_regen_cb)

        diag_box = QGroupBox("Diagnostic images (saved into output_dir/diagnostics/)")
        dfl = QVBoxLayout(diag_box)
        diag_check_row = QHBoxLayout()
        self._diag_whole_cb = QCheckBox("Whole image")
        self._diag_crops_cb = QCheckBox("Crop around each ROI")
        diag_check_row.addWidget(self._diag_whole_cb)
        diag_check_row.addWidget(self._diag_crops_cb)
        diag_check_row.addStretch()
        dfl.addLayout(diag_check_row)

        diag_opt_row = QHBoxLayout()
        diag_opt_row.addWidget(QLabel("Crop padding:"))
        self._diag_crop_pad = QDoubleSpinBox()
        self._diag_crop_pad.setRange(0, 2000); self._diag_crop_pad.setValue(50); self._diag_crop_pad.setSuffix(" µm")
        diag_opt_row.addWidget(self._diag_crop_pad)
        diag_opt_row.addSpacing(12)
        diag_opt_row.addWidget(QLabel("Format:"))
        self._diag_tiff_cb = QCheckBox("TIFF"); self._diag_tiff_cb.setChecked(True)
        self._diag_jpg_cb  = QCheckBox("JPG")
        diag_opt_row.addWidget(self._diag_tiff_cb)
        diag_opt_row.addWidget(self._diag_jpg_cb)
        diag_opt_row.addStretch()
        dfl.addLayout(diag_opt_row)

        diag_note = QLabel(
            "Colors come from each channel's \"Color\" setting in the Channels tab."
        )
        diag_note.setStyleSheet("color: gray; font-size: 10px;")
        dfl.addWidget(diag_note)
        bl.addWidget(diag_box)

        ctrl = QHBoxLayout()
        self._run_btn   = QPushButton("Run batch"); self._run_btn.setStyleSheet("font-weight:bold")
        self._run_btn.clicked.connect(self._on_run)
        self._abort_btn = QPushButton("Abort"); self._abort_btn.setEnabled(False)
        self._abort_btn.clicked.connect(self._on_abort)
        ctrl.addWidget(self._run_btn); ctrl.addWidget(self._abort_btn)
        bl.addLayout(ctrl)
        self._bar   = QProgressBar()
        self._prog_lbl = QLabel("")
        bl.addWidget(self._bar); bl.addWidget(self._prog_lbl)
        self._step_bar = QProgressBar()
        self._step_bar.setFixedHeight(10)
        self._step_bar.setTextVisible(False)
        self._step_lbl = QLabel("")
        self._step_lbl.setStyleSheet("color: gray; font-size: 10px;")
        bl.addWidget(self._step_bar); bl.addWidget(self._step_lbl)
        lay.addWidget(batch_box)

        log_box = QGroupBox("Log")
        ll = QVBoxLayout(log_box)
        self._log_edit = QTextEdit(); self._log_edit.setReadOnly(True)
        self._log_edit.setStyleSheet("font-family:monospace; font-size:11px;")
        ll.addWidget(self._log_edit)
        lay.addWidget(log_box)

    def _set_all_layers_visible(self, visible: bool) -> None:
        if self._viewer is None:
            return
        for layer in self._viewer.layers:
            layer.visible = visible

    def _on_gamma_changed(self, value: int) -> None:
        gamma = value / 100.0
        self._gamma_lbl.setText(f"{gamma:.2f}")
        if self._viewer is None:
            return
        from napari.layers import Image
        for layer in self._viewer.layers:
            if isinstance(layer, Image):
                layer.gamma = gamma

    def _on_auto_contrast(self) -> None:
        """Reset every visible Image layer to a percentile-stretched
        contrast range and reflect the (first layer's) values in the spinboxes.
        """
        if self._viewer is None:
            return
        from napari.layers import Image
        from correlative_imaging.viewer.napari_viewer import auto_contrast_limits

        first = True
        for layer in self._viewer.layers:
            if isinstance(layer, Image):
                lo, hi = auto_contrast_limits(layer.data)
                layer.contrast_limits = (lo, hi)
                if first:
                    self._contrast_min.setValue(lo)
                    self._contrast_max.setValue(hi)
                    first = False

    def _on_apply_contrast(self) -> None:
        """Apply the spinbox min/max to every Image layer currently in the viewer."""
        if self._viewer is None:
            return
        lo, hi = self._contrast_min.value(), self._contrast_max.value()
        if hi <= lo:
            QMessageBox.information(self, "Invalid range", "Max must be greater than min.")
            return
        from napari.layers import Image
        for layer in self._viewer.layers:
            if isinstance(layer, Image):
                layer.contrast_limits = (lo, hi)

    def _on_preview(self) -> None:
        """Preview on the loaded sample image.

        If a well is selected (the normal plate workflow), this runs the
        EXACT SAME pipeline batch would run for that well — including
        resolving/generating BF-pipeline class ROIs and per-well "existing
        project ROI" selections — not a separate, simplified, well-less
        pipeline. Only falls back to the simplified whole-image pipeline
        when there's genuinely no well to resolve against.
        """
        setup    = self._get_setup()
        img_data = setup.image_data
        if img_data is None:
            QMessageBox.information(self, "No image", "Load a sample image in the Setup tab first.")
            return

        well = setup.get_selected_well() if hasattr(setup, "get_selected_well") else None

        if well is not None and self._make_well_pipeline_dict_fn is not None:
            pipeline_dict_fn, needs_bf = self._make_well_pipeline_dict_fn()
            bf_cfg = self._get_bf_config() if (needs_bf and self._get_bf_config) else None
            if bf_cfg:
                # Unlike batch (hundreds of wells, reuse-from-disk matters for
                # time), preview is a single well on demand — the whole point
                # is to verify the pipeline actually works right now,
                # including Ilastik, not to shortcut past it using
                # possibly-stale files from an earlier run. Always regenerate.
                self._log(f"Running BF pipeline for {well.well_id} before preview …")
                self._preview_btn.setEnabled(False)
                self._preview_btn.setText("Running BF pipeline …")
                self._bf_preview_worker = _BFWorker([well], bf_cfg, test_mode=False)
                self._bf_preview_worker.log_msg.connect(lambda m: self._log(f"[BF] {m}"))
                self._bf_preview_worker.finished.connect(
                    lambda n_ok, n_err: self._run_well_preview(img_data, well, pipeline_dict_fn)
                )
                self._bf_preview_worker.start()
                return
            self._run_well_preview(img_data, well, pipeline_dict_fn)
        else:
            if well is None:
                self._log(
                    "No well selected — previewing the simplified whole-image "
                    "pipeline (per-well ROI selections are skipped; select a "
                    "well in Setup for a full, batch-equivalent preview)."
                )
            self._run_preview_with_dict(img_data, self._get_pipeline_dict())

    def _run_well_preview(self, img_data, well, pipeline_dict_fn) -> None:
        pl_dict, missing = pipeline_dict_fn(well)
        if missing:
            self._log(
                f"⚠ {well.well_id}: no match for {', '.join(missing)} — "
                "those selection(s) skipped for this preview."
            )
        self._run_preview_with_dict(img_data, pl_dict)

    def _run_preview_with_dict(self, img_data, pl_dict: dict) -> None:
        if not pl_dict.get("steps"):
            QMessageBox.information(self, "Empty pipeline",
                "Load a sample image and configure at least one channel.")
            return
        try:
            import os, tempfile
            from correlative_imaging.pipeline import Pipeline
            from correlative_imaging.pipeline.base import PipelineContext
            from napari.qt.threading import thread_worker

            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                json.dump(pl_dict, f); tmp = f.name
            pl = Pipeline.load(tmp); os.unlink(tmp)

            channel_colors = self._get_channel_colors() if self._get_channel_colors else []
            def _color_for(ch_idx: int) -> str:
                return channel_colors[ch_idx] if ch_idx < len(channel_colors) else \
                    _STEP_COLORMAPS[ch_idx % len(_STEP_COLORMAPS)]

            # last_preproc_layer[channel_idx] = the most recently added
            # per-channel preprocessing layer — since AutoThreshold/
            # WatershedSplit never set result.image (only result.masks), any
            # layer added below is definitionally "pre-threshold"; the last
            # one added per channel is what ends up visible by default (#23).
            last_preproc_layer: dict[int, object] = {}
            raw_layer_by_ch: dict[int, object] = {}

            if self._viewer is not None:
                self._viewer.layers.clear()
                px = img_data.pixel_size_um
                scale = [px, px]
                raw_mip = img_data.project("max")
                for i, ch in enumerate(img_data.channel_names):
                    raw_layer_by_ch[i] = self._viewer.add_image(
                        raw_mip[i], name=f"raw/{ch}", colormap=_color_for(i),
                        visible=False, blending="additive", scale=scale,
                        contrast_limits=auto_contrast_limits(raw_mip[i]),
                    )

            context = PipelineContext(channel_names=img_data.channel_names,
                                      pixel_size_um=img_data.pixel_size_um,
                                      z_step_um=img_data.z_step_um)
            self._preview_btn.setEnabled(False); self._preview_btn.setText("Running …")

            step_counter = [0]   # mutable so the closure can increment it

            @thread_worker
            def _run():
                current = img_data.data.copy()
                for step in pl.steps:
                    result = step.process(current, context)
                    if result.image is not None: current = result.image.copy()
                    context.masks.update(result.masks)
                    yield step.name, step, result, current.copy()

            def _on_step(args):
                step_name, step_obj, result, current_image = args
                if self._viewer is None: return
                px   = img_data.pixel_size_um
                scale = [px, px]
                cmap  = _STEP_COLORMAPS[step_counter[0] % len(_STEP_COLORMAPS)]
                step_counter[0] += 1
                mip = current_image.max(axis=1) if current_image.ndim == 4 else current_image
                if result.image is not None:
                    step_ch = getattr(step_obj, "channel", None)
                    if step_ch is not None:
                        # Per-channel step — only show the channel it actually modified
                        ch = img_data.channel_names[step_ch]
                        lyr = self._viewer.add_image(
                            mip[step_ch], name=f"{step_name}/{ch}",
                            visible=False, blending="additive",
                            colormap=cmap, scale=scale,
                            contrast_limits=auto_contrast_limits(mip[step_ch]),
                        )
                        # AutoThreshold/WatershedSplit never set result.image
                        # (only result.masks), so any layer added here is
                        # definitionally still "pre-threshold" — the last one
                        # per channel is what _on_done makes visible (#23).
                        last_preproc_layer[step_ch] = lyr
                    else:
                        # Multi-channel step — show all channels
                        for i, ch in enumerate(img_data.channel_names):
                            lyr = self._viewer.add_image(
                                mip[i], name=f"{step_name}/{ch}",
                                visible=False, blending="additive",
                                colormap=cmap, scale=scale,
                                contrast_limits=auto_contrast_limits(mip[i]),
                            )
                            last_preproc_layer[i] = lyr
                for mk, mask in result.masks.items():
                    if int(mask.max()) > 1:
                        self._viewer.add_labels(
                            mask.astype(int), name=f"{step_name}/{mk}", scale=scale,
                        )
                    elif mk.startswith("roi"):
                        lyr = self._viewer.add_labels(
                            mask.astype(int), name=f"{step_name}/{mk}", scale=scale,
                        )
                        lyr.contour = 1
                    else:
                        mask_f = mask.astype(float)
                        self._viewer.add_image(
                            mask_f, name=f"{step_name}/{mk}",
                            colormap=cmap, blending="additive", opacity=0.4, scale=scale,
                            contrast_limits=auto_contrast_limits(mask_f),
                        )
                self._log(f"  ✓ {step_name}")

            def _on_done():
                # Default display (#23): the last pre-threshold per-channel
                # layer (or, if a channel had no preprocessing steps at all,
                # its raw layer) becomes visible, in that channel's
                # user-assigned color. Everything else stays available but
                # hidden, for manual toggling/comparison.
                for i in range(len(img_data.channel_names)):
                    lyr = last_preproc_layer.get(i) or raw_layer_by_ch.get(i)
                    if lyr is not None:
                        lyr.colormap = _color_for(i)
                        lyr.visible = True
                self._preview_btn.setEnabled(True)
                self._preview_btn.setText("Run pipeline on sample image (shows in napari)")
                self._log("Preview complete.")

            def _on_err(exc_info):
                self._preview_btn.setEnabled(True)
                self._preview_btn.setText("Run pipeline on sample image (shows in napari)")
                msg = str(exc_info[1]) if isinstance(exc_info, tuple) else str(exc_info)
                self._log(f"Error: {msg}")

            worker = _run()
            worker.yielded.connect(_on_step)
            worker.finished.connect(_on_done)
            worker.errored.connect(_on_err); worker.start()
        except Exception:
            self._log(traceback.format_exc())
            self._preview_btn.setEnabled(True)
            self._preview_btn.setText("Run pipeline on sample image (shows in napari)")

    def _on_run(self) -> None:
        setup = self._get_setup()
        if setup.input_dir is None:
            QMessageBox.warning(self, "No folder", "Select an input folder in the Setup tab."); return
        pl_dict = self._get_pipeline_dict()
        if not pl_dict.get("steps"):
            QMessageBox.warning(self, "No pipeline", "Configure channels first."); return
        setup.output_dir.mkdir(parents=True, exist_ok=True)

        # DB, pipeline JSON, and log all share one base name (experiment +
        # run-start time) so a run's outputs are obviously grouped and never
        # silently overwrite an earlier run's.
        base = _run_basename(setup.experiment)
        db_path   = setup.output_dir / f"{base}.db"
        json_path = setup.output_dir / f"{base}_pipeline.json"
        self._current_log_path = setup.output_dir / f"{base}.log"
        json_path.write_text(json.dumps(pl_dict, indent=2))

        self._bar.setValue(0); self._run_btn.setEnabled(False); self._abort_btn.setEnabled(True)
        self._log(f"Starting batch — outputs: {base}.db / {base}_pipeline.json / {base}.log")

        diag_cfg = None
        if self._diag_whole_cb.isChecked() or self._diag_crops_cb.isChecked():
            formats = set()
            if self._diag_tiff_cb.isChecked():
                formats.add("tiff")
            if self._diag_jpg_cb.isChecked():
                formats.add("jpg")
            if formats:
                colors = self._get_channel_colors() if self._get_channel_colors else []
                diag_cfg = {
                    "whole": self._diag_whole_cb.isChecked(),
                    "crops": self._diag_crops_cb.isChecked(),
                    "crop_pad_um": self._diag_crop_pad.value(),
                    "formats": formats,
                    "colors": colors,
                    "output_dir": str(setup.output_dir / "diagnostics"),
                }
            else:
                self._log("Diagnostic images enabled but no format checked — skipping.")

        wells = setup.get_all_wells() if hasattr(setup, "get_all_wells") else []
        if wells and self._make_well_pipeline_dict_fn is not None:
            pipeline_dict_fn, needs_bf = self._make_well_pipeline_dict_fn()
            bf_cfg = self._get_bf_config() if (needs_bf and self._get_bf_config) else None
            if bf_cfg and not bf_cfg.get("save_dir"):
                # Mirror BFPipelineTab.run_on_wells: without this, a missing
                # "Save ROIs to" falls back to writing ROI files next to the
                # source BF image — often a read-only/slow network data share.
                bf_cfg["output_dir"] = str(setup.output_dir)
            max_workers = self._worker_spin.value() if self._parallel_cb.isChecked() else 1
            if max_workers > 1:
                self._log(f"Parallel batch (experimental): {max_workers} workers.")
            self._worker = _WellBatchWorker(
                wells, pipeline_dict_fn, bf_cfg, setup.output_dir, setup.experiment,
                max_workers=max_workers, db_path=db_path,
                force_regen=self._force_regen_cb.isChecked(),
                diag_cfg=diag_cfg,
            )
        else:
            self._worker = _BatchWorker(pl_dict, setup.input_dir, setup.output_dir, setup.experiment,
                                         db_path=db_path)
        self._worker.progress.connect(self._on_progress)
        self._worker.log_msg.connect(self._log)
        self._worker.finished.connect(self._on_finished)
        if hasattr(self._worker, "step_progress"):
            self._worker.step_progress.connect(self._on_step_progress)
        self._step_bar.setValue(0); self._step_lbl.setText("")
        self._worker.start()

    def _on_parallel_toggled(self, checked: bool) -> None:
        self._worker_spin.setEnabled(checked)

    def _on_abort(self) -> None:
        if self._worker: self._worker.abort(); self._log("Abort requested …")

    def _on_progress(self, current: int, total: int, name: str, _) -> None:
        self._bar.setValue(int(100 * current / max(total, 1)))
        self._prog_lbl.setText(f"{current}/{total}  —  {name}")

    def _on_step_progress(self, well_id: str, step_idx: int, total_steps: int, step_name: str) -> None:
        self._step_bar.setRange(0, max(total_steps, 1))
        self._step_bar.setValue(step_idx)
        self._step_lbl.setText(f"{well_id}: step {step_idx}/{total_steps} — {step_name}")

    def _on_finished(self, db_path: str) -> None:
        self._run_btn.setEnabled(True); self._abort_btn.setEnabled(False)
        self._bar.setValue(100)
        self._step_bar.setValue(0); self._step_lbl.setText("")
        self._log(f"Done ✓   Results → {db_path}" if db_path else "Finished (with errors).")
        self._prog_lbl.setText("Finished." if db_path else "Finished (with errors).")
        self._current_log_path = None

    def _log(self, msg: str) -> None:
        self._log_edit.append(str(msg))
        self._log_edit.verticalScrollBar().setValue(self._log_edit.verticalScrollBar().maximum())
        log_path = getattr(self, "_current_log_path", None)
        if log_path is not None:
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(str(msg) + "\n")
            except OSError:
                pass  # best-effort — never let log-file writing break the run


# ──────────────────────────────────────────────────────────────────
# Main container
# ──────────────────────────────────────────────────────────────────

class CorrelativeImagingWidget(QWidget):
    def __init__(self, napari_viewer=None, parent=None):
        super().__init__(parent)
        self._viewer = napari_viewer
        self._tabs   = QTabWidget()

        self._plate_tab    = PlateTab()
        self._bf_tab       = BFPipelineTab(viewer=napari_viewer)
        self._channels_tab = ChannelsTab()
        self._roi_tab      = ROISelectionsTab()
        self._combine_tab  = CombineTab()
        self._advanced_tab = AdvancedPipelineTab()
        self._run_tab      = RunTab(
            get_setup         = lambda: self._plate_tab,
            get_pipeline_dict = self.build_pipeline_dict,
            get_bf_config     = self._bf_tab.get_config,
            make_well_pipeline_dict_fn = self.make_well_pipeline_dict_fn,
            napari_viewer     = napari_viewer,
        )

        self._roi_tab.set_viewer(napari_viewer)
        self._bf_tab.set_roi_tab(self._roi_tab)
        self._plate_tab.existing_rois_detected.connect(self._on_existing_rois_detected)
        self._plate_tab.load_pipeline_requested.connect(self._on_load_pipeline_json)
        self._advanced_tab._get_guided_pipeline = self.build_pipeline_dict
        self._advanced_tab._get_setup = lambda: self._plate_tab
        self._advanced_tab._make_well_pipeline_dict_fn = self.make_well_pipeline_dict_fn
        self._advanced_tab._get_bf_config = self._bf_tab.get_config

        self._tabs.addTab(self._plate_tab.setup_page, "Setup")
        self._tabs.addTab(self._plate_tab.plate_page, "Plate")
        self._tabs.addTab(self._bf_tab,       "BF Pipeline")
        self._tabs.addTab(self._channels_tab, "Channels")
        self._tabs.addTab(self._roi_tab,      "ROI & Selections")
        self._tabs.addTab(self._combine_tab,  "Colocalization")
        self._tabs.addTab(self._advanced_tab, "Pipeline Summary")
        self._tabs.addTab(self._run_tab,      "Run")
        self._tabs.currentChanged.connect(self._on_tab_changed)

        self._plate_tab.channels_ready.connect(self._on_channels_ready)
        self._plate_tab.well_selected.connect(self._on_well_selected)
        self._plate_tab.view_requested.connect(self._on_view_requested)
        self._plate_tab.overview_requested.connect(self._on_overview_requested)

        self._view_cache: dict[tuple, object] = {}  # (well_id, "bf"/"fl") → image_data
        self._view_worker = None

        root = QVBoxLayout(self)
        root.addWidget(self._tabs)
        self.setMinimumWidth(420)

    def _add_image_layers(self, image_data, label: str, projection: str,
                          blending: str = "additive") -> None:
        try:
            from correlative_imaging.viewer.napari_viewer import NapariViewer
            nv = NapariViewer.__new__(NapariViewer); nv._viewer = self._viewer
            nv.show_image(image_data, group=label, projection=projection, blending=blending)
        except Exception:
            px = image_data.pixel_size_um; scale = [px, px]
            # ImageData.data always keeps the channel axis first (C,Y,X) or
            # (C,Z,Y,X) per its contract, so .project() always returns (C,Y,X).
            mip = image_data.project(projection)
            for i, ch in enumerate(image_data.channel_names):
                self._viewer.add_image(mip[i], name=f"{label}/{ch}",
                                       blending=blending, scale=scale,
                                       contrast_limits=auto_contrast_limits(mip[i]))

    def _add_roi_overlays(self, well, shape_yx: tuple, pixel_size_um: float = 1.0) -> None:
        # Separate Labels layers per ROI file — napari's own layer-visibility
        # checkboxes double as the "toggle one ROI vs another" control. All
        # visible by default; no hardcoded per-name special-casing — every
        # class is just a user-named region now, none more "special" than
        # another.
        #
        # scale MUST match the image layers' scale (see _add_image_layers /
        # NapariViewer.show_image, which scale by pixel_size_um) — without
        # it, napari renders this mask at raw pixel scale while the image
        # renders at physical (µm) scale, so the ROI appears shifted/outside
        # the image whenever pixel_size_um != 1.
        if well is None or not well.roi_paths:
            return
        h, w = shape_yx
        scale = [pixel_size_um, pixel_size_um]
        for p in well.roi_paths:
            mask = _load_roi_mask(p, h, w, pixel_size_um=pixel_size_um)
            if mask is None or not mask.any():
                continue
            self._viewer.add_labels(
                mask.astype(int), name=f"roi_{Path(p).stem}/{well.well_id}", opacity=0.4,
                scale=scale,
            )

    def _show_in_viewer(self, image_data, label: str, well=None, projection: str = "max") -> None:
        self._viewer.layers.clear()
        self._add_image_layers(image_data, label, projection)
        if well is not None:
            self._add_roi_overlays(well, image_data.data.shape[-2:], image_data.pixel_size_um)

    def _show_well_overview(self, well, bf_data, fl_data, projection: str) -> None:
        self._viewer.layers.clear()
        shape_yx = None
        pixel_size_um = 1.0
        if bf_data is not None:
            # BF as a translucent (not additive) base layer — additively
            # summing it with FL would wash the whole view out to white,
            # since BF is often bright across most of the frame.
            self._add_image_layers(bf_data, "BF", projection, blending="translucent")
            shape_yx = bf_data.data.shape[-2:]
            pixel_size_um = bf_data.pixel_size_um
        if fl_data is not None:
            self._add_image_layers(fl_data, "FL", projection)
            shape_yx = shape_yx or fl_data.data.shape[-2:]
            if bf_data is None:
                pixel_size_um = fl_data.pixel_size_um
        if shape_yx is not None:
            self._add_roi_overlays(well, shape_yx, pixel_size_um)

    def _on_view_requested(self, well, which: str, projection: str = "max") -> None:
        if self._viewer is None:
            return
        path = well.bf_path if which == "bf" else well.fl_path
        if path is None:
            return
        label = which.upper()
        cache_key = (well.well_id, which)

        if cache_key in self._view_cache:
            self._show_in_viewer(self._view_cache[cache_key], label, well, projection)
            return

        def _loaded(image_data):
            self._view_cache[cache_key] = image_data
            self._show_in_viewer(image_data, label, well, projection)

        self._view_worker = _LoadImageWorker(path)
        self._view_worker.loaded.connect(_loaded)
        self._view_worker.error.connect(lambda msg: None)
        self._view_worker.start()
        self._viewer.status = f"Loading {label} for well {well.well_id} …"

    def _on_overview_requested(self, well, projection: str = "max") -> None:
        """Show BF + FL + this well's ROI(s) together, for the Setup tab's
        combined "Show BF + FL + ROI overview" button.
        """
        if self._viewer is None:
            return
        bf_key, fl_key = (well.well_id, "bf"), (well.well_id, "fl")
        cached_bf = self._view_cache.get(bf_key)
        cached_fl = self._view_cache.get(fl_key)
        bf_ready = well.bf_path is None or cached_bf is not None
        fl_ready = well.fl_path is None or cached_fl is not None
        if bf_ready and fl_ready:
            self._show_well_overview(well, cached_bf, cached_fl, projection)
            return

        def _loaded(bf_data, fl_data):
            if bf_data is not None:
                self._view_cache[bf_key] = bf_data
            if fl_data is not None:
                self._view_cache[fl_key] = fl_data
            self._show_well_overview(well, bf_data, fl_data, projection)

        self._view_worker = _LoadWellWorker(well)
        self._view_worker.loaded.connect(_loaded)
        self._view_worker.error.connect(
            lambda msg: QMessageBox.critical(self, "Load error", msg)
        )
        self._view_worker.start()
        self._viewer.status = f"Loading BF + FL for well {well.well_id} …"

    def _on_well_selected(self, well) -> None:
        if self._plate_tab._auto_roi_cb.isChecked() and well.roi_paths:
            self._roi_tab.load_detected_rois(well.roi_paths)

    def _on_channels_ready(self, channel_names: list[str]) -> None:
        img = self._plate_tab.image_data
        if img is not None and self._viewer is not None:
            try:
                from correlative_imaging.viewer.napari_viewer import NapariViewer
                v = NapariViewer.__new__(NapariViewer); v._viewer = self._viewer
                self._viewer.layers.clear()
                v.show_image(img, group="sample")
            except Exception:
                pass

        self._channels_tab.set_channels(channel_names)
        self._roi_tab.set_channels(channel_names)
        self._combine_tab.set_channels(channel_names)
        self._combine_tab.auto_populate(channel_names)
        self._advanced_tab.set_channel_names(channel_names)
        self._tabs.setCurrentWidget(self._channels_tab)

    def _on_existing_rois_detected(self, tags: list[str]) -> None:
        """Auto-add a "well_existing" ROI & Selections entry for every
        distinct pre-existing ROI tag found across the plate scan — the
        user can remove any they don't want, but since a match already
        exists on disk it should show up by default, not require manually
        re-adding and re-typing it. Skips tags already present as a
        selection, and tags that match a configured BF-pipeline class name
        (those are added via the BF Pipeline tab's own "Import to ROI
        Selections" button instead, not here).
        """
        existing = {s.existing_tag for s in self._roi_tab.get_selections() if s.source == "well_existing"}
        bf_class_names = {c["name"] for c in self._bf_tab.get_classes_config()}
        added = 0
        for tag in tags:
            if tag in existing or tag in bf_class_names:
                continue
            self._roi_tab.add_selection(_ROISel(label=tag, source="well_existing", existing_tag=tag))
            added += 1
        if added:
            log.info("Auto-added %d existing-project ROI selection(s) from plate scan: %s",
                      added, ", ".join(tags))

    def _on_tab_changed(self, index: int) -> None:
        if self._tabs.widget(index) is self._plate_tab.setup_page:
            self._plate_tab.refresh_json_dropdown(self._plate_tab.output_dir)

    def _on_load_pipeline_json(self, pl_dict: dict) -> None:
        """Apply a loaded pipeline JSON to Channels, ROI & Selections, and
        Colocalization all at once. Channels must already be set (a sample
        image loaded) since channel count/names aren't recoverable from the
        JSON alone (it only has channel indices)."""
        if not self._channels_tab.get_panels():
            QMessageBox.information(
                self, "Load a sample image first",
                "Load a sample image in the Setup tab first, so channel "
                "count/names are known, then load the pipeline JSON."
            )
            return
        self._channels_tab.populate_from_pipeline(pl_dict)
        warnings = self._roi_tab.populate_from_pipeline(pl_dict)
        self._combine_tab.populate_from_pipeline(pl_dict)
        msg = f"Loaded pipeline '{pl_dict.get('name', 'pipeline')}' into Channels, ROI & Selections, Colocalization."
        if warnings:
            msg += "\n\n" + "\n".join(warnings)
            QMessageBox.warning(self, "Loaded with caveats", msg)
        log.info(msg)

    def build_pipeline_dict(self, well=None) -> dict:
        """
        Pipeline order:
          1. Preprocessing   — once per channel  (BGSub, Blur, Normalize)
          2. Segmentation    — once per channel  (AutoThreshold, WatershedSplit)
          3. Per ROI selection:
               a. ROI extraction (ExtractROI or LoadROI) — skipped for "whole image"
               b. ParticleAnalysis per channel, restricted to that ROI
               c. ColocalizationAnalysis per pair, restricted to that ROI

        ``well`` resolves per-well-dynamic ROI selections ("BF-pipeline hole
        ROI" / "existing project ROI") against that well's own ROI files —
        pass it when building a pipeline for one specific well in a batch
        run. Leave it ``None`` for single-image preview, where those two
        selections can't resolve and are skipped entirely (not run
        unrestricted) so they never produce misleadingly-labeled results.
        """
        panels = self._channels_tab.get_panels()
        sels   = self._roi_tab.get_selections()
        steps: list[dict] = []

        # ── 1. Preprocessing ──
        for p in panels:
            steps.extend(p.get_preprocess_steps())

        # ── 2. Segmentation ──
        for p in panels:
            steps.extend(p.get_segment_steps())

        # ── 3. Per ROI selection: ROI mask + particle analysis + colocalization ──
        for sel in sels:
            roi_step = sel.get_roi_step(well)
            if sel.source != "whole" and roi_step is None:
                continue  # per-well or file source didn't resolve — skip this selection
            if roi_step:
                steps.append(roi_step)
            for p in panels:
                steps.extend(p.get_analysis_steps(sel.mask_key))
            for coloc in self._combine_tab.get_coloc_steps(sel.mask_key):
                steps.append(coloc)

        return {"name": "pipeline", "steps": steps}

    def make_well_pipeline_dict_fn(self):
        """Snapshot the current Channels/ROI/Combine config into a plain,
        thread-safe closure ``(well) -> (pipeline dict, missing_labels)``.

        ``missing_labels`` lists the ``label`` of every per-well-dynamic
        selection (``well_class`` / ``well_existing``) that had no matching
        file for this specific well — the caller (``WellBatchRunner``)
        surfaces these instead of silently dropping the selection for that
        well. "Whole image" / "auto" / "file" selections can't go missing
        this way and are never listed.

        Returns ``(fn, needs_bf)`` — ``needs_bf`` is True iff at least one
        selection uses the "well_class" (BF-pipeline) source, so the caller
        only bothers running Ilastik detection when actually needed.

        ``_WellBatchWorker`` calls ``fn`` once per well from a background
        QThread — it must not touch any QWidget. This method reads all
        Qt-widget-derived config *here*, on the calling (main GUI) thread,
        and captures only plain dicts / ``_ROISel`` dataclasses / ``WellInfo``
        dataclasses, which are safe to read from any thread.
        """
        panels = self._channels_tab.get_panels()
        preprocess_steps: list[dict] = []
        segment_steps: list[dict] = []
        for p in panels:
            preprocess_steps.extend(p.get_preprocess_steps())
            segment_steps.extend(p.get_segment_steps())

        sels = list(self._roi_tab.get_selections())
        needs_bf = any(s.source == "well_class" for s in sels)
        per_well_dynamic = {"well_class", "well_existing"}
        per_sel_tail: list[tuple] = []
        for sel in sels:
            tail: list[dict] = []
            for p in panels:
                tail.extend(p.get_analysis_steps(sel.mask_key))
            tail.extend(self._combine_tab.get_coloc_steps(sel.mask_key))
            per_sel_tail.append((sel, tail))

        def _fn(well) -> tuple[dict, list[str]]:
            steps = list(preprocess_steps) + list(segment_steps)
            missing: list[str] = []
            for sel, tail in per_sel_tail:
                roi_step = sel.get_roi_step(well)
                if sel.source != "whole" and roi_step is None:
                    if sel.source in per_well_dynamic:
                        missing.append(sel.label)
                    continue
                if roi_step:
                    steps.append(roi_step)
                steps.extend(tail)
            return {"name": "pipeline", "steps": steps}, missing

        return _fn, needs_bf


# ──────────────────────────────────────────────────────────────────
# Widget / form helpers
# ──────────────────────────────────────────────────────────────────

def _path_row(placeholder: str, is_dir: bool) -> tuple[QLineEdit, QHBoxLayout]:
    row  = QHBoxLayout()
    edit = QLineEdit(); edit.setPlaceholderText(placeholder)
    btn  = QPushButton("…"); btn.setFixedWidth(28)

    def _browse():
        p = (QFileDialog.getExistingDirectory(None, placeholder)
             if is_dir
             else QFileDialog.getOpenFileName(None, placeholder, "", "All files (*)")[0])
        if p: edit.setText(p)

    btn.clicked.connect(_browse)
    row.addWidget(edit); row.addWidget(btn)
    return edit, row


def _make_widget(wtype: str, default, extra) -> QWidget:
    if wtype == "int":
        w = QSpinBox(); lo, hi = extra or (0, 100000); w.setRange(lo, hi); w.setValue(int(default)); return w
    if wtype == "float":
        w = QDoubleSpinBox(); lo, hi = extra or (0.0, 1e9); w.setRange(lo, hi)
        w.setDecimals(3); w.setValue(float(default)); return w
    if wtype.startswith("choice:"):
        choices = wtype.split(":", 1)[1].split(",")
        w = QComboBox(); w.addItems(choices)
        if default in choices: w.setCurrentText(str(default))
        return w
    if wtype == "bool":
        w = QCheckBox(); w.setChecked(bool(default)); return w
    return QLineEdit(str(default))


def _read_widget(w: QWidget, wtype: str):
    if isinstance(w, QSpinBox):       return w.value()
    if isinstance(w, QDoubleSpinBox): return w.value()
    if isinstance(w, QComboBox):      return w.currentText()
    if isinstance(w, QCheckBox):      return w.isChecked()
    if isinstance(w, QLineEdit):
        v = w.text()
        if wtype == "int":   return int(v)   if v.strip() else 0
        if wtype == "float": return float(v) if v.strip() else 0.0
        return v
    return None


def _set_widget(w: QWidget, wtype: str, value) -> None:
    if isinstance(w, (QSpinBox, QDoubleSpinBox)): w.setValue(value)
    elif isinstance(w, QComboBox):                w.setCurrentText(str(value))
    elif isinstance(w, QCheckBox):                w.setChecked(bool(value))
    elif isinstance(w, QLineEdit):                w.setText(str(value))


# ──────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────

def launch_gui() -> None:
    import napari
    viewer = napari.Viewer(title="Correlative Imaging")
    widget = CorrelativeImagingWidget(napari_viewer=viewer)
    viewer.window.add_dock_widget(widget, name="Correlative Imaging", area="right")
    napari.run()
