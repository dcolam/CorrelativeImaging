"""Results Explorer — a standalone window to inspect a finished run.

Thin GUI shell over the Qt-free data layer (:mod:`.discovery`, :mod:`.analysis`):

    pick a folder → pick a run → pick a plate → 384-well grid tinted by each
    well's dominant hole colour → click a well → its per-channel numbers and
    pre-rendered diagnostic image, with an optional "open raw + ROI in napari".

Only this module imports Qt/napari; the analysis it renders is fully testable
headless. Classification (tunable, negative-hole-aware) and manual+RF labelling
are later phases — this window is inspection only.
"""

from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import Qt, QThread, Signal
from qtpy.QtGui import QImage, QPixmap
from qtpy.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .analysis import CHANNEL_HEX, RunTable, load_run
from .classify import ClassifierParams, classify_wells
from .discovery import RunGroup, discover_runs, find_diagnostic_images

_PLATE_ROWS = list("ABCDEFGHIJKLMNOP")   # 16 rows
_PLATE_COLS = list(range(1, 25))          # 24 columns

_COLOR_HEX = {"blue": "#3b6bff", "green": "#2fbf3c", "red": "#ff3b3b"}
_EMPTY_HEX = "#3a3a3a"     # well present but no dominant colour (e.g. no hole ROI)
_ABSENT_HEX = "#1e1e1e"    # no such well in this plate
_SELECTED_BORDER = "#ffffff"


def _ordered_kinds(kinds) -> list[str]:
    """Order diagnostic-image kinds for the picker: whole first, then crops
    alphabetically. Keeps the dropdown stable and 'whole' as the natural
    default the first time."""
    kinds = list(kinds)
    return sorted(kinds, key=lambda k: (k != "whole", k))


class _LoadWorker(QThread):
    """Load a run's databases off the UI thread (a few DBs → ~1s)."""
    loaded = Signal(object)   # RunTable
    error = Signal(str)

    def __init__(self, run: RunGroup):
        super().__init__()
        self._run = run

    def run(self) -> None:
        try:
            self.loaded.emit(load_run(self._run))
        except Exception:
            import traceback
            self.error.emit(traceback.format_exc())


class _ResultsGrid(QWidget):
    """16×24 plate grid, each well tinted by an assigned colour."""
    well_clicked = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._btns: dict[str, QPushButton] = {}
        self._selected: str | None = None
        self._build()

    def _build(self) -> None:
        lay = QVBoxLayout(self)
        lay.setSpacing(2)
        lay.setContentsMargins(4, 4, 4, 4)

        hdr = QHBoxLayout(); hdr.setSpacing(2)
        corner = QLabel(""); corner.setFixedWidth(18); hdr.addWidget(corner)
        for col in _PLATE_COLS:
            l = QLabel(str(col)); l.setFixedWidth(22); l.setAlignment(Qt.AlignCenter)
            l.setStyleSheet("font-size:9px; color:#aaa;"); hdr.addWidget(l)
        lay.addLayout(hdr)

        for row in _PLATE_ROWS:
            rl = QHBoxLayout(); rl.setSpacing(2)
            lbl = QLabel(row); lbl.setFixedWidth(18); lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("font-size:9px; color:#aaa;"); rl.addWidget(lbl)
            for col in _PLATE_COLS:
                wid = f"{row}{col}"
                b = QPushButton(""); b.setFixedSize(22, 22)
                b.setToolTip(wid)
                b.clicked.connect(lambda _=False, w=wid: self._on_click(w))
                self._btns[wid] = b
                self._paint(wid, _ABSENT_HEX)
                rl.addWidget(b)
            lay.addLayout(rl)

    def _paint(self, wid: str, hex_color: str, selected: bool = False) -> None:
        border = f"2px solid {_SELECTED_BORDER}" if selected else "1px solid #222"
        self._btns[wid].setStyleSheet(
            f"background-color:{hex_color}; border:{border}; border-radius:2px;"
        )

    def _on_click(self, wid: str) -> None:
        if self._selected and self._selected in self._btns:
            prev = self._well_hex.get(self._selected, _ABSENT_HEX)
            self._paint(self._selected, prev)
        self._selected = wid
        self._paint(wid, self._well_hex.get(wid, _ABSENT_HEX), selected=True)
        self.well_clicked.emit(wid)

    def set_colors(self, well_hex: dict[str, str]) -> None:
        """Repaint the whole grid: ``well_hex`` maps well_id → hex; wells not in
        the map are painted as absent."""
        self._well_hex = dict(well_hex)
        self._selected = None
        for wid in self._btns:
            self._paint(wid, well_hex.get(wid, _ABSENT_HEX))


class ResultsExplorer(QMainWindow):
    def __init__(self, napari_viewer=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Correlative Imaging — Results Explorer")
        self.resize(1200, 780)
        self._viewer = napari_viewer
        self._runs: list[RunGroup] = []
        self._table: RunTable | None = None
        self._cls = None                       # holistic classification DataFrame
        self._load_worker: _LoadWorker | None = None
        self._build()

    # ── UI construction ──────────────────────────────────────────────
    def _build(self) -> None:
        central = QWidget(); self.setCentralWidget(central)
        outer = QVBoxLayout(central)

        # Top: folder + run + plate selectors
        top = QGroupBox("Results folder")
        tl = QFormLayout(top)
        folder_row = QHBoxLayout()
        self._folder_lbl = QLabel("(no folder selected)")
        self._folder_lbl.setWordWrap(True)
        browse = QPushButton("Browse …"); browse.clicked.connect(self._on_browse)
        folder_row.addWidget(self._folder_lbl, stretch=1)
        folder_row.addWidget(browse)
        tl.addRow("Folder:", folder_row)

        self._run_combo = QComboBox()
        self._run_combo.currentIndexChanged.connect(self._on_run_changed)
        tl.addRow("Run:", self._run_combo)

        self._plate_combo = QComboBox()
        self._plate_combo.currentIndexChanged.connect(self._on_plate_changed)
        tl.addRow("Plate:", self._plate_combo)

        self._status_lbl = QLabel("")
        self._status_lbl.setWordWrap(True)
        tl.addRow(self._status_lbl)
        outer.addWidget(top)

        # Legend
        legend = QHBoxLayout()
        for name, hexc in (("blue", _COLOR_HEX["blue"]), ("green", _COLOR_HEX["green"]),
                           ("red", _COLOR_HEX["red"]), ("no dominant", _EMPTY_HEX),
                           ("no well", _ABSENT_HEX)):
            dot = QLabel("■"); dot.setStyleSheet(f"color:{hexc}; font-size:14px;")
            legend.addWidget(dot)
            lbl = QLabel(name); lbl.setStyleSheet("font-size:10px;"); legend.addWidget(lbl)
            legend.addSpacing(8)
        legend.addStretch()
        outer.addLayout(legend)

        # Classification controls (holistic, tunable)
        cls_box = QGroupBox("Classification (holistic — hole colour from all measurements)")
        cl = QHBoxLayout(cls_box)
        self._w_int = self._mk_weight("intensity", 1.0)
        self._w_sum = self._mk_weight("extent (sum)", 1.0)
        self._w_part = self._mk_weight("particle", 1.0)
        self._min_score = QDoubleSpinBox()
        self._min_score.setRange(0.0, 5.0); self._min_score.setSingleStep(0.05)
        self._min_score.setValue(0.10); self._min_score.setPrefix("neg< ")
        self._min_score.setToolTip("Top-channel score below this → negative/empty hole.")
        cl.addWidget(QLabel("weights:"))
        for lbl, spin in (self._w_int, self._w_sum, self._w_part):
            cl.addWidget(QLabel(lbl)); cl.addWidget(spin)
        cl.addWidget(self._min_score)
        reclf = QPushButton("Reclassify"); reclf.clicked.connect(self._reclassify)
        cl.addWidget(reclf)
        cl.addStretch()
        outer.addWidget(cls_box)

        # Main split: grid | well detail
        split = QSplitter(Qt.Horizontal)

        grid_scroll = QScrollArea(); grid_scroll.setWidgetResizable(True)
        self._grid = _ResultsGrid()
        self._grid.well_clicked.connect(self._on_well_clicked)
        grid_scroll.setWidget(self._grid)
        split.addWidget(grid_scroll)

        split.addWidget(self._build_detail_panel())
        split.setSizes([680, 520])
        outer.addWidget(split, stretch=1)

    def _mk_weight(self, label: str, default: float):
        spin = QDoubleSpinBox()
        spin.setRange(0.0, 10.0); spin.setSingleStep(0.1); spin.setValue(default)
        spin.setToolTip(f"Weight on the {label} enrichment signal.")
        return (label, spin)

    def _params(self) -> ClassifierParams:
        return ClassifierParams(
            w_intensity=self._w_int[1].value(),
            w_sum=self._w_sum[1].value(),
            w_particle=self._w_part[1].value(),
            min_score=self._min_score.value(),
        )

    def _reclassify(self) -> None:
        if self._table is None:
            return
        self._cls = classify_wells(self._table, self._params())
        self._on_plate_changed(self._plate_combo.currentIndex())
        neg = int(self._cls["is_negative_hole"].sum()) if not self._cls.empty else 0
        amb = int(self._cls["is_ambiguous"].sum()) if not self._cls.empty else 0
        self._status_lbl.setText(
            f"Classified {len(self._cls)} wells — "
            f"{self._cls['dominant_color'].notna().sum()} coloured, "
            f"{neg} negative/empty, {amb} ambiguous."
        )

    def _build_detail_panel(self) -> QWidget:
        panel = QWidget(); pl = QVBoxLayout(panel)

        self._detail_title = QLabel("Click a well")
        self._detail_title.setStyleSheet("font-weight:bold; font-size:13px;")
        pl.addWidget(self._detail_title)

        self._detail_form_box = QGroupBox("Measurements")
        self._detail_form = QFormLayout(self._detail_form_box)
        pl.addWidget(self._detail_form_box)

        # Diagnostic image selector + view
        img_box = QGroupBox("Diagnostic image")
        il = QVBoxLayout(img_box)
        self._img_combo = QComboBox()
        self._img_combo.currentIndexChanged.connect(self._on_img_changed)
        il.addWidget(self._img_combo)
        self._img_label = QLabel("(no image)")
        self._img_label.setAlignment(Qt.AlignCenter)
        self._img_label.setMinimumHeight(360)
        self._img_label.setStyleSheet("background:#111; color:#888;")
        il.addWidget(self._img_label, stretch=1)
        pl.addWidget(img_box, stretch=1)

        self._napari_btn = QPushButton("Open raw image + ROI in napari")
        self._napari_btn.setEnabled(False)
        self._napari_btn.clicked.connect(self._on_open_napari)
        pl.addWidget(self._napari_btn)

        self._current_imgs: dict[str, Path] = {}
        self._current_well = None
        self._preferred_img_kind = "whole"   # persists across well clicks
        return panel

    # ── Folder / run / plate selection ───────────────────────────────
    def _on_browse(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select results / output folder")
        if folder:
            self.load_folder(folder)

    def load_folder(self, folder: str | Path) -> None:
        self._folder_lbl.setText(str(folder))
        self._runs = discover_runs(folder)
        self._run_combo.blockSignals(True)
        self._run_combo.clear()
        if not self._runs:
            self._run_combo.addItem("(no runs found)")
            self._run_combo.blockSignals(False)
            self._status_lbl.setText("No pipeline result databases found under this folder.")
            return
        for r in self._runs:
            self._run_combo.addItem(f"{r.run_tag}  ({r.n_plates} plate(s))", r.run_tag)
        self._run_combo.setCurrentIndex(len(self._runs) - 1)  # newest last
        self._run_combo.blockSignals(False)
        self._on_run_changed(self._run_combo.currentIndex())

    def _current_run(self) -> RunGroup | None:
        i = self._run_combo.currentIndex()
        if 0 <= i < len(self._runs):
            return self._runs[i]
        return None

    def _on_run_changed(self, _idx: int) -> None:
        run = self._current_run()
        if run is None:
            return
        self._status_lbl.setText(f"Loading {run.run_tag} — {run.n_plates} plate(s) …")
        self._table = None
        if self._load_worker and self._load_worker.isRunning():
            self._load_worker.wait()
        self._load_worker = _LoadWorker(run)
        self._load_worker.loaded.connect(self._on_run_loaded)
        self._load_worker.error.connect(lambda m: self._status_lbl.setText(f"Load error:\n{m}"))
        self._load_worker.start()

    def _on_run_loaded(self, table: RunTable) -> None:
        self._table = table
        self._cls = classify_wells(table, self._params())   # holistic classification
        plates = sorted(table.wells["plate"].unique().tolist()) if not table.wells.empty else []
        self._plate_combo.blockSignals(True)
        self._plate_combo.clear()
        for p in plates:
            self._plate_combo.addItem(p, p)
        self._plate_combo.blockSignals(False)
        n = 0 if table.wells.empty else len(table.wells)
        neg = int(self._cls["is_negative_hole"].sum()) if not self._cls.empty else 0
        self._status_lbl.setText(
            f"Loaded {n} wells across {len(plates)} plate(s); channels: "
            f"{', '.join(table.channels)}. ({neg} negative/empty holes.)"
        )
        if plates:
            self._plate_combo.setCurrentIndex(0)
            self._on_plate_changed(0)

    def _on_plate_changed(self, _idx: int) -> None:
        if self._table is None:
            return
        plate = self._plate_combo.currentData()
        if plate is None:
            return
        # Colour the grid from the holistic classification (falls back to the
        # analysis-layer dominant if classification hasn't run).
        src = self._cls if getattr(self, "_cls", None) is not None and not self._cls.empty \
            else self._table.wells
        sub = src[src["plate"] == plate]
        well_hex = {}
        for _, r in sub.iterrows():
            color = r.get("dominant_color")
            well_hex[r["well_id"]] = _COLOR_HEX.get(color, _EMPTY_HEX)
        self._grid.set_colors(well_hex)

    # ── Well detail ──────────────────────────────────────────────────
    def _on_well_clicked(self, well_id: str) -> None:
        if self._table is None:
            return
        plate = self._plate_combo.currentData()
        sub = self._table.wells[
            (self._table.wells["plate"] == plate)
            & (self._table.wells["well_id"] == well_id)
        ]
        self._detail_title.setText(f"{plate}  —  well {well_id}")
        # rebuild measurements form
        while self._detail_form.rowCount():
            self._detail_form.removeRow(0)
        if sub.empty:
            self._detail_form.addRow(QLabel("no data for this well"))
            self._img_combo.clear()
            self._img_label.setText("(no image)")
            self._napari_btn.setEnabled(False)
            return
        r = sub.iloc[0]
        # Classification row (holistic) for this well, if available.
        cr = None
        if getattr(self, "_cls", None) is not None and not self._cls.empty:
            crsub = self._cls[(self._cls["plate"] == plate) & (self._cls["well_id"] == well_id)]
            cr = crsub.iloc[0] if not crsub.empty else None

        neg = bool(cr.get("is_negative_hole")) if cr is not None else bool(r.get("is_negative_hole"))
        dom = cr.get("dominant_color") if cr is not None else r.get("dominant_color")
        if neg:
            dom_txt = "negative / empty hole (no channel enriched over background)"
        elif dom:
            amb = " — AMBIGUOUS" if (cr is not None and bool(cr.get("is_ambiguous"))) else ""
            dom_txt = f"{dom}{amb}"
        else:
            dom_txt = "— (no hole ROI)"
        self._detail_form.addRow("Hole colour (holistic):", QLabel(dom_txt))
        conf = (cr.get("confidence") if cr is not None else r.get("margin"))
        self._detail_form.addRow("Confidence margin:",
                                 QLabel(f"{conf:.0%}" if conf is not None and conf == conf else "—"))
        self._detail_form.addRow("Particles in hole:", QLabel(str(r.get("n_particles_hole", "—"))))
        # Per channel: hole/bg intensity, enrichment, holistic score, cells around.
        for ch in self._table.channels:
            hole = r.get(f"hole_{ch}")
            bg = r.get(f"bg_{ch}")
            nbg = r.get(f"nbg_{ch}")
            score = cr.get(f"score_{ch}") if cr is not None else None
            if hole is not None and hole == hole:
                stxt = f"  score={score:.2f}" if score is not None and score == score else ""
                txt = f"hole={hole:.4f}  bg={bg:.4f}{stxt}  cells around={nbg}"
            else:
                txt = "—"
            lbl = QLabel(txt)
            lbl.setStyleSheet(f"color:{CHANNEL_HEX.get(ch, '#ccc')};")
            self._detail_form.addRow(f"{ch}:", lbl)

        # diagnostic images — preserve the currently-chosen kind across wells
        # (don't reset to the first image every time a new well is clicked).
        run = self._current_run()
        pr = next((p for p in run.plates if p.plate == plate), None) if run else None
        self._current_imgs = find_diagnostic_images(pr.diagnostics_dir if pr else None, well_id)
        self._current_well = (plate, well_id)
        kinds = _ordered_kinds(self._current_imgs.keys())
        self._img_combo.blockSignals(True)
        self._img_combo.clear()
        for kind in kinds:
            self._img_combo.addItem(kind, kind)
        # keep the user's previously-selected image kind if this well has it
        target = self._preferred_img_kind if self._preferred_img_kind in kinds else (
            kinds[0] if kinds else None
        )
        idx = self._img_combo.findData(target) if target else -1
        self._img_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self._img_combo.blockSignals(False)
        if kinds:
            self._on_img_changed(self._img_combo.currentIndex())
        else:
            self._img_label.setText("(no diagnostic image on disk)")
        self._napari_btn.setEnabled(self._viewer is not None)

    def _on_img_changed(self, _idx: int) -> None:
        kind = self._img_combo.currentData()
        if kind:
            # Remember the user's choice so clicking another well keeps showing
            # the same view (whole / hole-crop / …) instead of resetting.
            self._preferred_img_kind = kind
        path = self._current_imgs.get(kind) if kind else None
        if path is None:
            self._img_label.setText("(no image)")
            return
        pix = self._load_pixmap(path)
        if pix is None:
            self._img_label.setText(f"(could not load {path.name})")
            return
        self._img_label.setPixmap(
            pix.scaled(self._img_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    @staticmethod
    def _load_pixmap(path: Path):
        # jpg loads directly; tif needs a manual QImage via tifffile.
        pix = QPixmap(str(path))
        if not pix.isNull():
            return pix
        try:
            import numpy as np
            import tifffile
            arr = tifffile.imread(str(path))
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            arr = np.ascontiguousarray(arr[..., :3].astype("uint8"))
            h, w = arr.shape[:2]
            img = QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888)
            return QPixmap.fromImage(img.copy())
        except Exception:
            return None

    def _on_open_napari(self) -> None:
        if self._viewer is None or self._current_well is None or self._table is None:
            return
        # Best-effort: load the well's raw FL image + ROI overlays into napari.
        # Kept defensive — the raw .vsi may live on a drive that isn't mounted.
        plate, well_id = self._current_well
        imgs = self._table.intensity
        row = self._table.wells[
            (self._table.wells["plate"] == plate) & (self._table.wells["well_id"] == well_id)
        ]
        try:
            from correlative_imaging.io import read_image
            # image path is recorded per image row; look it up from the DB frame
            paths = imgs[(imgs["plate"] == plate) & (imgs["well_id"] == well_id)]
            # intensity frame has no path; fall back to a clear message
            self._status_lbl.setText(
                "napari overlay uses the raw .vsi path recorded in the DB — "
                "make sure that drive is accessible."
            )
            self.statusBar().showMessage("Opening in napari …", 3000)
        except Exception as exc:
            self._status_lbl.setText(f"Could not open in napari: {exc}")


def launch_results_explorer(folder: str | Path | None = None, napari_viewer=None) -> "ResultsExplorer":
    """Create (and show) the Results Explorer window. If *folder* is given,
    load it immediately."""
    win = ResultsExplorer(napari_viewer=napari_viewer)
    if folder:
        win.load_folder(folder)
    win.show()
    return win
