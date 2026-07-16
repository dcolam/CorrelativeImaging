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
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
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
from .labels import LABEL_COLORS, LabelStore

_PLATE_ROWS = list("ABCDEFGHIJKLMNOP")   # 16 rows
_PLATE_COLS = list(range(1, 25))          # 24 columns

_COLOR_HEX = {"blue": "#3b6bff", "green": "#2fbf3c", "red": "#ff3b3b"}
_NEGATIVE_HEX = "#2a2a2a"  # hole present but no channel positive (negative/empty hole)
_EMPTY_HEX = "#3a3a3a"     # well present but no hole ROI at all
_ABSENT_HEX = "#1e1e1e"    # no such well in this plate
_SELECTED_BORDER = "#ffffff"
_LABELLED_BORDER = "#ffcc33"   # well has a hand label (ground truth)


def _blend_hex(colors: list[str]) -> str:
    """Average the RGB of the positive colours so a multi-positive hole shows a
    mixed tint (e.g. green+red → yellow-ish). Empty list → negative-hole grey."""
    rgbs = [_COLOR_HEX[c] for c in colors if c in _COLOR_HEX]
    if not rgbs:
        return _NEGATIVE_HEX
    if len(rgbs) == 1:
        return rgbs[0]
    r = g = b = 0
    for hx in rgbs:
        r += int(hx[1:3], 16); g += int(hx[3:5], 16); b += int(hx[5:7], 16)
    n = len(rgbs)
    return f"#{r // n:02x}{g // n:02x}{b // n:02x}"


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
        self._well_hex: dict[str, str] = {}
        self._labelled: set[str] = set()
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
        if selected:
            border = f"2px solid {_SELECTED_BORDER}"
        elif wid in self._labelled:
            border = f"2px solid {_LABELLED_BORDER}"   # ground-truth marker
        else:
            border = "1px solid #222"
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

    def set_colors(self, well_hex: dict[str, str], labelled: set[str] | None = None) -> None:
        """Repaint the whole grid: ``well_hex`` maps well_id → hex; wells not in
        the map are painted as absent. Wells in ``labelled`` get a gold border
        marking that they carry a hand label."""
        self._well_hex = dict(well_hex)
        self._labelled = set(labelled or ())
        self._selected = None
        for wid in self._btns:
            self._paint(wid, well_hex.get(wid, _ABSENT_HEX))

    def mark_labelled(self, wid: str, labelled: bool, hex_color: str) -> None:
        """Toggle one well's labelled border in place (after a save/clear),
        keeping it selected."""
        if labelled:
            self._labelled.add(wid)
        else:
            self._labelled.discard(wid)
        self._well_hex[wid] = hex_color
        self._paint(wid, hex_color, selected=(wid == self._selected))


class ResultsExplorer(QMainWindow):
    def __init__(self, napari_viewer=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Correlative Imaging — Results Explorer")
        self.resize(1200, 780)
        self._viewer = napari_viewer
        self._runs: list[RunGroup] = []
        self._table: RunTable | None = None
        self._cls = None                       # holistic classification DataFrame
        self._labels: LabelStore | None = None  # ground-truth store for this folder
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

        # Legend — multi-label: a hole can be several colours (mixed tint),
        # a negative/empty hole, or have no hole ROI. Gold border = hand-labelled.
        legend = QHBoxLayout()
        for name, hexc in (("blue", _COLOR_HEX["blue"]), ("green", _COLOR_HEX["green"]),
                           ("red", _COLOR_HEX["red"]), ("mixed", "#c8b400"),
                           ("negative hole", _NEGATIVE_HEX), ("no hole", _EMPTY_HEX),
                           ("no well", _ABSENT_HEX)):
            dot = QLabel("■"); dot.setStyleSheet(f"color:{hexc}; font-size:14px;")
            legend.addWidget(dot)
            lbl = QLabel(name); lbl.setStyleSheet("font-size:10px;"); legend.addWidget(lbl)
            legend.addSpacing(8)
        gold = QLabel("▢"); gold.setStyleSheet(f"color:{_LABELLED_BORDER}; font-size:14px;")
        legend.addWidget(gold)
        gl = QLabel("labelled"); gl.setStyleSheet("font-size:10px;"); legend.addWidget(gl)
        legend.addStretch()
        outer.addLayout(legend)

        # Classification controls (holistic, multi-label, tunable)
        cls_box = QGroupBox("Classification (multi-label — each channel called independently)")
        cl = QHBoxLayout(cls_box)
        self._w_int = self._mk_weight("intensity", 1.0)
        self._w_sum = self._mk_weight("extent (sum)", 1.0)
        self._w_part = self._mk_weight("particle", 1.0)
        self._pos_thr = QDoubleSpinBox()
        self._pos_thr.setRange(0.0, 10.0); self._pos_thr.setSingleStep(0.05)
        self._pos_thr.setValue(0.10); self._pos_thr.setPrefix("pos≥ ")
        self._pos_thr.setToolTip(
            "A channel is called positive when its score clears this. Shared "
            "default across channels — calibrate per-channel against hand labels."
        )
        cl.addWidget(QLabel("weights:"))
        for lbl, spin in (self._w_int, self._w_sum, self._w_part):
            cl.addWidget(QLabel(lbl)); cl.addWidget(spin)
        cl.addWidget(self._pos_thr)
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
            pos_threshold=self._pos_thr.value(),
        )

    def _reclassify(self) -> None:
        if self._table is None:
            return
        self._cls = classify_wells(self._table, self._params())
        self._on_plate_changed(self._plate_combo.currentIndex())
        if self._cls.empty:
            return
        present = self._cls[self._cls["hole_present"]]
        neg = int(present["is_negative_hole"].sum())
        coloured = int((present["n_positive"] > 0).sum())
        multi = int((present["n_positive"] > 1).sum())
        self._status_lbl.setText(
            f"Classified {len(present)} holes — {coloured} positive "
            f"({multi} multi-colour), {neg} negative/empty."
        )

    def _build_detail_panel(self) -> QWidget:
        panel = QWidget(); pl = QVBoxLayout(panel)

        self._detail_title = QLabel("Click a well")
        self._detail_title.setStyleSheet("font-weight:bold; font-size:13px;")
        pl.addWidget(self._detail_title)

        self._detail_form_box = QGroupBox("Measurements")
        self._detail_form = QFormLayout(self._detail_form_box)
        pl.addWidget(self._detail_form_box)

        # Ground-truth labelling — seeded from the classifier call so the user
        # corrects rather than labels from scratch. Each colour is independent
        # (multi-label); none ticked = negative/empty hole.
        label_box = QGroupBox("Ground-truth label (hand)")
        ll = QVBoxLayout(label_box)
        self._label_status = QLabel("—"); self._label_status.setStyleSheet("font-size:11px;")
        ll.addWidget(self._label_status)
        chk_row = QHBoxLayout()
        self._label_checks: dict[str, QCheckBox] = {}
        for color in LABEL_COLORS:
            cb = QCheckBox(color)
            cb.setStyleSheet(f"color:{_COLOR_HEX.get(color, '#ccc')}; font-weight:bold;")
            self._label_checks[color] = cb
            chk_row.addWidget(cb)
        chk_row.addStretch()
        ll.addLayout(chk_row)
        hint = QLabel("(none ticked = negative / empty hole)")
        hint.setStyleSheet("font-size:10px; color:#888;"); ll.addWidget(hint)
        self._label_notes = QLineEdit(); self._label_notes.setPlaceholderText("notes (optional)")
        ll.addWidget(self._label_notes)
        btn_row = QHBoxLayout()
        self._save_label_btn = QPushButton("Save label")
        self._save_label_btn.clicked.connect(self._on_save_label)
        self._clear_label_btn = QPushButton("Clear label")
        self._clear_label_btn.clicked.connect(self._on_clear_label)
        self._seed_label_btn = QPushButton("Reset to classifier call")
        self._seed_label_btn.clicked.connect(self._seed_label_from_classifier)
        btn_row.addWidget(self._save_label_btn)
        btn_row.addWidget(self._clear_label_btn)
        btn_row.addWidget(self._seed_label_btn)
        ll.addLayout(btn_row)
        self._set_label_controls_enabled(False)
        pl.addWidget(label_box)

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
        # Ground-truth labels live in a sidecar DB beside the browsed folder.
        try:
            self._labels = LabelStore.for_folder(folder)
        except Exception:
            self._labels = None
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
        # Colour the grid from the multi-label classification: a hole present &
        # positive → mixed tint of its positive colours; present & negative →
        # negative-hole grey; no hole ROI → empty grey.
        well_hex = {}
        if getattr(self, "_cls", None) is not None and not self._cls.empty:
            sub = self._cls[self._cls["plate"] == plate]
            for _, r in sub.iterrows():
                if not r.get("hole_present"):
                    well_hex[r["well_id"]] = _EMPTY_HEX
                    continue
                colors = [c for c in (r.get("colors") or "").split(",") if c]
                well_hex[r["well_id"]] = _blend_hex(colors)
        self._grid.set_colors(well_hex, labelled=self._labelled_wells(plate))

    def _labelled_wells(self, plate: str) -> set[str]:
        """Well ids on *plate* that carry a hand label in the current run."""
        run = self._current_run()
        if self._labels is None or run is None:
            return set()
        df = self._labels.load_frame(run.run_tag)
        if df.empty:
            return set()
        return set(df[df["plate"] == plate]["well_id"].tolist())

    # ── Ground-truth labelling ───────────────────────────────────────
    def _set_label_controls_enabled(self, enabled: bool) -> None:
        for cb in self._label_checks.values():
            cb.setEnabled(enabled)
        self._label_notes.setEnabled(enabled)
        self._save_label_btn.setEnabled(enabled)
        self._clear_label_btn.setEnabled(enabled)
        self._seed_label_btn.setEnabled(enabled)

    def _classifier_colors_for(self, cr) -> set[str]:
        """The classifier's positive colours for a classification row."""
        if cr is None or not cr.get("hole_present"):
            return set()
        return {c for c in (cr.get("colors") or "").split(",") if c}

    def _set_checks(self, colors: set[str]) -> None:
        for color, cb in self._label_checks.items():
            cb.blockSignals(True)
            cb.setChecked(color in colors)
            cb.blockSignals(False)

    def _refresh_label_controls(self, plate: str, well_id: str, cr) -> None:
        """Show any stored label for this well; if none, seed the checkboxes
        from the classifier call so the user corrects rather than starts blank."""
        self._label_notes.setText("")
        stored = None
        if self._labels is not None:
            run = self._current_run()
            if run is not None:
                stored = self._labels.get_label(run.run_tag, plate, well_id)
        if stored is not None:
            self._set_checks(stored["colors"])
            self._label_notes.setText(stored.get("notes") or "")
            colours = ", ".join(sorted(stored["colors"])) or "negative / empty"
            self._label_status.setText(f"labelled: {colours}  ({stored['updated_at']})")
        else:
            self._set_checks(self._classifier_colors_for(cr))
            self._label_status.setText("not labelled — seeded from classifier; edit & Save")
        self._set_label_controls_enabled(self._labels is not None)

    def _seed_label_from_classifier(self) -> None:
        if self._current_well is None or self._cls is None:
            return
        plate, well_id = self._current_well
        crsub = self._cls[(self._cls["plate"] == plate) & (self._cls["well_id"] == well_id)]
        cr = crsub.iloc[0] if not crsub.empty else None
        self._set_checks(self._classifier_colors_for(cr))

    def _on_save_label(self) -> None:
        if self._labels is None or self._current_well is None:
            return
        run = self._current_run()
        if run is None:
            return
        plate, well_id = self._current_well
        colors = {c for c, cb in self._label_checks.items() if cb.isChecked()}
        self._labels.set_label(run.run_tag, plate, well_id, colors,
                               notes=self._label_notes.text().strip() or None)
        colours = ", ".join(sorted(colors)) or "negative / empty"
        self._label_status.setText(f"saved: {colours}")
        # gold-border the well and refresh the labelled-count in the status bar
        self._grid.mark_labelled(well_id, True, _blend_hex(sorted(colors)))
        self._update_label_count()

    def _on_clear_label(self) -> None:
        if self._labels is None or self._current_well is None:
            return
        run = self._current_run()
        if run is None:
            return
        plate, well_id = self._current_well
        self._labels.delete_label(run.run_tag, plate, well_id)
        self._label_status.setText("label cleared")
        # repaint the well back to its classifier tint, drop the gold border
        cr = None
        crsub = self._cls[(self._cls["plate"] == plate) & (self._cls["well_id"] == well_id)] \
            if self._cls is not None and not self._cls.empty else None
        if crsub is not None and not crsub.empty:
            cr = crsub.iloc[0]
        hexc = _blend_hex(sorted(self._classifier_colors_for(cr))) if cr is not None \
            else _EMPTY_HEX
        self._grid.mark_labelled(well_id, False, hexc)
        self._update_label_count()

    def _update_label_count(self) -> None:
        run = self._current_run()
        if self._labels is None or run is None:
            return
        n = self._labels.count(run.run_tag)
        self.statusBar().showMessage(f"{n} wells labelled in {run.run_tag}", 4000)

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
            # Critical: drop the label target so a stray Save can't write this
            # (empty) well's UI state onto the previously-selected well.
            self._current_well = None
            self._label_status.setText("— (no data for this well)")
            self._set_checks(set())
            self._set_label_controls_enabled(False)
            return
        r = sub.iloc[0]
        # Multi-label classification row for this well, if available.
        cr = None
        if getattr(self, "_cls", None) is not None and not self._cls.empty:
            crsub = self._cls[(self._cls["plate"] == plate) & (self._cls["well_id"] == well_id)]
            cr = crsub.iloc[0] if not crsub.empty else None

        if cr is not None and cr.get("hole_present"):
            colors = [c for c in (cr.get("colors") or "").split(",") if c]
            if colors:
                call_txt = " + ".join(colors)
            else:
                call_txt = "negative / empty hole (no channel positive)"
        elif cr is not None:
            call_txt = "— (no hole ROI)"
        else:
            call_txt = "—"
        self._detail_form.addRow("Call (multi-label):", QLabel(call_txt))
        self._detail_form.addRow("Particles in hole:", QLabel(str(r.get("n_particles_hole", "—"))))
        # Per channel: hole/bg intensity, score, margin (signed dist to threshold),
        # positive?, and cells around.
        for ch in self._table.channels:
            hole = r.get(f"hole_{ch}")
            bg = r.get(f"bg_{ch}")
            nbg = r.get(f"nbg_{ch}")
            score = cr.get(f"score_{ch}") if cr is not None else None
            margin = cr.get(f"margin_{ch}") if cr is not None else None
            pos = bool(cr.get(f"pos_{ch}")) if cr is not None else False
            if hole is not None and hole == hole:
                stxt = f"  score={score:.2f}" if score is not None and score == score else ""
                mtxt = f"  Δthr={margin:+.2f}" if margin is not None and margin == margin else ""
                flag = "  ✓POS" if pos else ""
                txt = f"hole={hole:.4f}  bg={bg:.4f}{stxt}{mtxt}{flag}  cells around={nbg}"
            else:
                txt = "—"
            lbl = QLabel(txt)
            weight = "font-weight:bold;" if pos else ""
            lbl.setStyleSheet(f"color:{CHANNEL_HEX.get(ch, '#ccc')};{weight}")
            self._detail_form.addRow(f"{ch}:", lbl)

        # Seed / show the ground-truth label controls for this well.
        self._refresh_label_controls(plate, well_id, cr)

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
