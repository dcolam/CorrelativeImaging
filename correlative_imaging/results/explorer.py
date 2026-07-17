"""Results Explorer — a standalone window to inspect and label a finished run.

Thin GUI shell over the Qt-free data layer (:mod:`.discovery`, :mod:`.analysis`,
:mod:`.classify`, :mod:`.labels`):

    pick a folder → pick a run → pick a plate → 384-well grid tinted by each
    hole's multi-label channel call → click a well → its per-channel numbers,
    pre-rendered diagnostic image, and a ground-truth labelling panel.

Everything is keyed by CHANNEL NAME; colour is a per-run display concern, edited
via the "Channels…" dialog and persisted in the sidecar label DB. Only this
module imports Qt/napari; the analysis it renders is fully testable headless.
"""

from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import Qt, QThread, Signal
from qtpy.QtGui import QColor, QImage, QPixmap
from qtpy.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from . import lda as _lda
from .analysis import RunTable, load_run
from .classify import ClassifierParams, classify_wells
from .discovery import RunGroup, discover_runs, find_diagnostic_images
from .labels import LabelStore, default_display

_PLATE_ROWS = list("ABCDEFGHIJKLMNOP")   # 16 rows
_PLATE_COLS = list(range(1, 25))          # 24 columns

_NEGATIVE_HEX = "#2a2a2a"  # hole present but no channel positive (negative/empty hole)
_EMPTY_HEX = "#3a3a3a"     # well present but no hole ROI at all
_ABSENT_HEX = "#1e1e1e"    # no such well in this plate
_SELECTED_BORDER = "#ffffff"
_LABELLED_BORDER = "#ffcc33"   # well has a hand label (ground truth)


def _blend_hex(hexes: list[str]) -> str:
    """Average the RGB of the given tints so a multi-positive hole shows a mixed
    colour. Empty list → negative-hole grey. Inputs are ``#rrggbb`` strings (the
    per-channel display colours)."""
    hexes = [h for h in hexes if isinstance(h, str) and h.startswith("#") and len(h) == 7]
    if not hexes:
        return _NEGATIVE_HEX
    if len(hexes) == 1:
        return hexes[0]
    r = g = b = 0
    for hx in hexes:
        r += int(hx[1:3], 16); g += int(hx[3:5], 16); b += int(hx[5:7], 16)
    n = len(hexes)
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
    """16×24 plate grid, each well tinted by an assigned colour.

    Laid out as one rigid :class:`QGridLayout` (row/column headers + wells) with a
    trailing row/column stretch, so the matrix stays tight in the top-left and the
    wells always line up — rather than drifting apart when the scroll area is
    wider than the plate."""
    well_clicked = Signal(str)

    _CELL = 19          # px per well button (compact, keeps the whole plate small)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._btns: dict[str, QPushButton] = {}
        self._selected: str | None = None
        self._well_hex: dict[str, str] = {}
        self._labelled: set[str] = set()
        self._build()

    def _build(self) -> None:
        grid = QGridLayout(self)
        grid.setSpacing(1)
        grid.setContentsMargins(4, 4, 4, 4)

        # Column headers (1..24) in row 0; row headers (A..P) in column 0.
        for c, col in enumerate(_PLATE_COLS, start=1):
            l = QLabel(str(col)); l.setAlignment(Qt.AlignCenter)
            l.setStyleSheet("font-size:9px; color:#aaa;")
            grid.addWidget(l, 0, c)
        for r, row in enumerate(_PLATE_ROWS, start=1):
            rl = QLabel(row); rl.setAlignment(Qt.AlignCenter); rl.setFixedWidth(14)
            rl.setStyleSheet("font-size:9px; color:#aaa;")
            grid.addWidget(rl, r, 0)
            for c, col in enumerate(_PLATE_COLS, start=1):
                wid = f"{row}{col}"
                b = QPushButton(""); b.setFixedSize(self._CELL, self._CELL)
                b.setToolTip(wid)
                b.clicked.connect(lambda _=False, w=wid: self._on_click(w))
                self._btns[wid] = b
                self._paint(wid, _ABSENT_HEX)
                grid.addWidget(b, r, c)

        # Pool any extra space bottom-right so the plate stays compact top-left.
        grid.setRowStretch(len(_PLATE_ROWS) + 1, 1)
        grid.setColumnStretch(len(_PLATE_COLS) + 1, 1)

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

    def select(self, wid: str) -> None:
        """Assert the white selection border on *wid* (idempotent) — used when a
        well is re-rendered programmatically, e.g. after a channel-display edit
        repainted the grid and cleared the selection."""
        if wid not in self._btns:
            return
        if self._selected and self._selected != wid and self._selected in self._btns:
            self._paint(self._selected, self._well_hex.get(self._selected, _ABSENT_HEX))
        self._selected = wid
        self._paint(wid, self._well_hex.get(wid, _ABSENT_HEX), selected=True)

    def mark_labelled(self, wid: str, labelled: bool, hex_color: str) -> None:
        """Toggle one well's labelled border in place (after a save/clear),
        keeping it selected."""
        if labelled:
            self._labelled.add(wid)
        else:
            self._labelled.discard(wid)
        self._well_hex[wid] = hex_color
        self._paint(wid, hex_color, selected=(wid == self._selected))


class _ChannelDisplayDialog(QDialog):
    """Edit the per-run display name + tint colour of each channel.

    Channel↔colour varies between experiments, so this is where the user maps a
    run's channels (e.g. ``405nm``) to a meaningful label (e.g. ``GFAP-BFP``) and
    a tint. Returns ``{channel: {"display_name", "color_hex"}}`` on accept."""

    def __init__(self, channels: list[str], current: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Channel display")
        self._colors = {ch: current[ch]["color_hex"] for ch in channels}
        self._names: dict[str, QLineEdit] = {}
        self._swatches: dict[str, QPushButton] = {}

        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Display name and tint colour per channel (this run):"))
        grid = QGridLayout()
        grid.addWidget(QLabel("channel"), 0, 0)
        grid.addWidget(QLabel("display name"), 0, 1)
        grid.addWidget(QLabel("colour"), 0, 2)
        for i, ch in enumerate(channels, start=1):
            grid.addWidget(QLabel(ch), i, 0)
            name = QLineEdit(current[ch]["display_name"])
            self._names[ch] = name
            grid.addWidget(name, i, 1)
            sw = QPushButton()
            sw.setFixedWidth(60)
            self._swatches[ch] = sw
            self._paint_swatch(ch)
            sw.clicked.connect(lambda _=False, c=ch: self._pick_color(c))
            grid.addWidget(sw, i, 2)
        lay.addLayout(grid)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

    def _paint_swatch(self, ch: str) -> None:
        self._swatches[ch].setStyleSheet(
            f"background-color:{self._colors[ch]}; border:1px solid #222;"
        )

    def _pick_color(self, ch: str) -> None:
        col = QColorDialog.getColor(QColor(self._colors[ch]), self, f"Colour for {ch}")
        if col.isValid():
            self._colors[ch] = col.name()
            self._paint_swatch(ch)

    def result_config(self) -> dict:
        return {ch: {"display_name": self._names[ch].text().strip() or ch,
                     "color_hex": self._colors[ch]}
                for ch in self._names}


class ResultsExplorer(QMainWindow):
    def __init__(self, napari_viewer=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Correlative Imaging — Results Explorer")
        self.resize(1200, 780)
        self._viewer = napari_viewer
        self._runs: list[RunGroup] = []
        self._table: RunTable | None = None
        self._cls = None                       # ACTIVE classification frame (grid/detail read this)
        self._cls_rule = None                  # rule-based classification
        self._cls_lda = None                   # LDA-predicted classification (after Evaluate LDA)
        self._labels: LabelStore | None = None  # ground-truth store for this folder
        self._chan_display: dict[str, dict] = {}  # {channel: {display_name, color_hex}}
        self._load_worker: _LoadWorker | None = None
        self._build()

    def _channel_hex(self, channel: str) -> str:
        d = self._chan_display.get(channel) or default_display(channel)
        return d["color_hex"]

    def _channel_name(self, channel: str) -> str:
        d = self._chan_display.get(channel) or default_display(channel)
        return d["display_name"]

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

        # Legend — rebuilt per run from the channel-display config (a swatch per
        # channel), plus the fixed states. Gold border = hand-labelled. A
        # "Channels…" button opens the per-run display editor.
        legend_row = QHBoxLayout()
        self._legend = QHBoxLayout()
        legend_row.addLayout(self._legend)
        legend_row.addStretch()
        self._channels_btn = QPushButton("Channels …")
        self._channels_btn.setToolTip("Edit per-channel display name and tint colour (this run).")
        self._channels_btn.clicked.connect(self._edit_channels)
        self._channels_btn.setEnabled(False)
        legend_row.addWidget(self._channels_btn)
        outer.addLayout(legend_row)
        self._rebuild_legend()

        # Classification controls (holistic, multi-label, tunable)
        cls_box = QGroupBox("Classification (multi-label — each channel called independently)")
        cl = QHBoxLayout(cls_box)
        self._w_int = self._mk_weight("intensity", 1.0)
        self._w_sum = self._mk_weight("extent (sum)", 1.0)
        self._w_part = self._mk_weight("particle", 1.0)
        self._pos_thr = QDoubleSpinBox()
        self._pos_thr.setRange(0.0, 10.0); self._pos_thr.setSingleStep(0.05)
        self._pos_thr.setValue(0.50); self._pos_thr.setPrefix("pos≥ ")
        self._pos_thr.setToolTip(
            "A channel is positive when its score clears this. With scaling on, "
            "the score is in per-channel SD units. Calibrate against hand labels."
        )
        self._min_area = QDoubleSpinBox()
        self._min_area.setRange(0.0, 1.0); self._min_area.setSingleStep(0.01)
        self._min_area.setValue(0.05); self._min_area.setPrefix("occ≥ ")
        self._min_area.setToolTip(
            "Occupancy floor: fraction of the hole ROI that this channel's objects "
            "must cover to be called positive. Rejects bright-but-empty holes."
        )
        self._scale_combo = QComboBox()
        self._scale_combo.addItem("scale: across run", "run")
        self._scale_combo.addItem("scale: per plate", "plate")
        self._scale_combo.addItem("scale: off", "off")
        self._scale_combo.setToolTip(
            "Population scaling of scores. 'across run' pools all plates (preserves "
            "cross-plate/timepoint differences); 'per plate' removes plate batch drift."
        )
        cl.addWidget(QLabel("weights:"))
        for lbl, spin in (self._w_int, self._w_sum, self._w_part):
            cl.addWidget(QLabel(lbl)); cl.addWidget(spin)
        cl.addWidget(self._pos_thr)
        cl.addWidget(self._min_area)
        cl.addWidget(self._scale_combo)
        reclf = QPushButton("Reclassify"); reclf.clicked.connect(self._reclassify)
        cl.addWidget(reclf)
        cl.addSpacing(12)
        self._lda_btn = QPushButton("Evaluate LDA")
        self._lda_btn.setToolTip(
            "Train one LDA per channel on the hand labels and report honest "
            "cross-validated accuracy/kappa vs the rule-based baseline."
        )
        self._lda_btn.clicked.connect(self._evaluate_lda)
        cl.addWidget(self._lda_btn)
        self._use_lda = QCheckBox("use LDA calls")
        self._use_lda.setToolTip("Colour the grid by the LDA prediction instead of the rule.")
        self._use_lda.setEnabled(False)
        self._use_lda.toggled.connect(self._toggle_call_source)
        cl.addWidget(self._use_lda)
        cl.addStretch()
        outer.addWidget(cls_box)

        # Main split: plate grid | well detail — each in its own scroll area so a
        # small window scrolls instead of clipping.
        split = QSplitter(Qt.Horizontal)

        grid_scroll = QScrollArea(); grid_scroll.setWidgetResizable(True)
        grid_scroll.setAlignment(Qt.AlignTop | Qt.AlignLeft)   # keep plate top-left
        self._grid = _ResultsGrid()
        self._grid.well_clicked.connect(self._on_well_clicked)
        grid_scroll.setWidget(self._grid)
        split.addWidget(grid_scroll)

        detail_scroll = QScrollArea(); detail_scroll.setWidgetResizable(True)
        detail_scroll.setWidget(self._build_detail_panel())
        split.addWidget(detail_scroll)

        split.setStretchFactor(0, 0)     # plate keeps its size
        split.setStretchFactor(1, 1)     # detail takes the extra width
        split.setSizes([520, 680])
        outer.addWidget(split, stretch=1)

    def _mk_weight(self, label: str, default: float):
        spin = QDoubleSpinBox()
        spin.setRange(0.0, 10.0); spin.setSingleStep(0.1); spin.setValue(default)
        spin.setToolTip(f"Weight on the {label} enrichment signal.")
        return (label, spin)

    def _params(self) -> ClassifierParams:
        scale = self._scale_combo.currentData()
        return ClassifierParams(
            w_intensity=self._w_int[1].value(),
            w_sum=self._w_sum[1].value(),
            w_particle=self._w_part[1].value(),
            pos_threshold=self._pos_thr.value(),
            min_area=self._min_area.value(),
            population_scale=(scale != "off"),
            scale_per_plate=(scale == "plate"),
        )

    def _reclassify(self) -> None:
        if self._table is None:
            return
        self._cls_rule = classify_wells(self._table, self._params())
        # Params changed → any prior LDA prediction is stale; drop back to rule.
        self._cls_lda = None
        self._use_lda.blockSignals(True)
        self._use_lda.setChecked(False)
        self._use_lda.setEnabled(False)
        self._use_lda.blockSignals(False)
        self._cls = self._cls_rule
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
        self._detail_title.setWordWrap(True)   # long plate names wrap, not overflow
        pl.addWidget(self._detail_title)

        self._detail_form_box = QGroupBox("Measurements")
        self._detail_form = QFormLayout(self._detail_form_box)
        pl.addWidget(self._detail_form_box)

        # Ground-truth labelling — one checkbox per CHANNEL (built per run),
        # seeded from the classifier call so the user corrects rather than labels
        # from scratch. Each channel is independent (multi-label); none ticked =
        # negative/empty hole.
        label_box = QGroupBox("Ground-truth label (hand)")
        ll = QVBoxLayout(label_box)
        self._label_status = QLabel("—"); self._label_status.setStyleSheet("font-size:11px;")
        ll.addWidget(self._label_status)
        self._label_check_row = QHBoxLayout()
        self._label_checks: dict[str, QCheckBox] = {}   # {channel: checkbox}
        ll.addLayout(self._label_check_row)
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
        self._cls_rule = classify_wells(table, self._params())   # rule-based classification
        self._cls_lda = None
        self._cls = self._cls_rule
        self._use_lda.blockSignals(True)
        self._use_lda.setChecked(False); self._use_lda.setEnabled(False)
        self._use_lda.blockSignals(False)
        self._load_channel_display()          # per-run display names + tints
        self._rebuild_legend()
        self._rebuild_label_checks()
        self._channels_btn.setEnabled(bool(table.channels) and self._labels is not None)
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
        # positive → mixed tint of its positive channels' display colours;
        # present & negative → negative-hole grey; no hole ROI → empty grey.
        well_hex = {}
        if getattr(self, "_cls", None) is not None and not self._cls.empty:
            sub = self._cls[self._cls["plate"] == plate]
            for _, r in sub.iterrows():
                if not r.get("hole_present"):
                    well_hex[r["well_id"]] = _EMPTY_HEX
                    continue
                chans = [c for c in (r.get("pos_channels") or "").split(",") if c]
                well_hex[r["well_id"]] = _blend_hex([self._channel_hex(c) for c in chans])
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

    # ── Channel display (name + tint), per run ───────────────────────
    def _load_channel_display(self) -> None:
        run = self._current_run()
        channels = self._table.channels if self._table else []
        if self._labels is not None and run is not None and channels:
            self._chan_display = self._labels.get_channel_display(run.run_tag, channels)
        else:
            self._chan_display = {ch: default_display(ch, i)
                                  for i, ch in enumerate(channels)}

    def _clear_layout(self, layout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
            elif item.layout() is not None:
                self._clear_layout(item.layout())

    def _rebuild_legend(self) -> None:
        """Rebuild the legend from the current channel-display config plus the
        fixed grid states."""
        self._clear_layout(self._legend)
        channels = self._table.channels if self._table else []
        entries = [(self._channel_name(ch), self._channel_hex(ch)) for ch in channels]
        entries += [("mixed", "#c8b400"), ("negative hole", _NEGATIVE_HEX),
                    ("no hole", _EMPTY_HEX), ("no well", _ABSENT_HEX)]
        for name, hexc in entries:
            dot = QLabel("■"); dot.setStyleSheet(f"color:{hexc}; font-size:14px;")
            self._legend.addWidget(dot)
            lbl = QLabel(name); lbl.setStyleSheet("font-size:10px;")
            self._legend.addWidget(lbl)
            self._legend.addSpacing(8)
        gold = QLabel("▢"); gold.setStyleSheet(f"color:{_LABELLED_BORDER}; font-size:14px;")
        self._legend.addWidget(gold)
        gl = QLabel("labelled"); gl.setStyleSheet("font-size:10px;")
        self._legend.addWidget(gl)

    def _rebuild_label_checks(self) -> None:
        """One label checkbox per channel, styled by its display name + tint."""
        self._clear_layout(self._label_check_row)
        self._label_checks = {}
        channels = self._table.channels if self._table else []
        for ch in channels:
            cb = QCheckBox(self._channel_name(ch))
            cb.setStyleSheet(f"color:{self._channel_hex(ch)}; font-weight:bold;")
            cb.setToolTip(f"channel {ch}")
            self._label_checks[ch] = cb
            self._label_check_row.addWidget(cb)
        self._label_check_row.addStretch()

    def _edit_channels(self) -> None:
        run = self._current_run()
        if self._labels is None or run is None or not (self._table and self._table.channels):
            return
        channels = self._table.channels
        dlg = _ChannelDisplayDialog(channels, self._chan_display, self)
        if dlg.exec_() if hasattr(dlg, "exec_") else dlg.exec():
            for ch, cfg in dlg.result_config().items():
                self._labels.set_channel_display(run.run_tag, ch,
                                                  cfg["display_name"], cfg["color_hex"])
            self._load_channel_display()
            self._rebuild_legend()
            self._rebuild_label_checks()
            self._on_plate_changed(self._plate_combo.currentIndex())
            # re-render the open well so its per-channel rows + checkbox styling refresh
            if self._current_well is not None:
                self._on_well_clicked(self._current_well[1])

    # ── Ground-truth labelling ───────────────────────────────────────
    def _set_label_controls_enabled(self, enabled: bool) -> None:
        for cb in self._label_checks.values():
            cb.setEnabled(enabled)
        self._label_notes.setEnabled(enabled)
        self._save_label_btn.setEnabled(enabled)
        self._clear_label_btn.setEnabled(enabled)
        self._seed_label_btn.setEnabled(enabled)

    def _classifier_channels_for(self, cr) -> set[str]:
        """The classifier's positive CHANNELS for a classification row."""
        if cr is None or not cr.get("hole_present"):
            return set()
        return {c for c in (cr.get("pos_channels") or "").split(",") if c}

    def _blend_channels(self, channels) -> str:
        """Grid tint for a set/iterable of positive channel names."""
        return _blend_hex([self._channel_hex(c) for c in channels])

    def _set_checks(self, channels: set[str]) -> None:
        for ch, cb in self._label_checks.items():
            cb.blockSignals(True)
            cb.setChecked(ch in channels)
            cb.blockSignals(False)

    def _label_summary(self, channels) -> str:
        """Human summary of a channel set using display names."""
        chans = sorted(channels)
        return ", ".join(self._channel_name(c) for c in chans) if chans \
            else "negative / empty"

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
            self._set_checks(stored["channels"])
            self._label_notes.setText(stored.get("notes") or "")
            self._label_status.setText(
                f"labelled: {self._label_summary(stored['channels'])}  ({stored['updated_at']})"
            )
        else:
            self._set_checks(self._classifier_channels_for(cr))
            self._label_status.setText("not labelled — seeded from classifier; edit & Save")
        self._set_label_controls_enabled(self._labels is not None)

    def _seed_label_from_classifier(self) -> None:
        if self._current_well is None or self._cls is None:
            return
        plate, well_id = self._current_well
        crsub = self._cls[(self._cls["plate"] == plate) & (self._cls["well_id"] == well_id)]
        cr = crsub.iloc[0] if not crsub.empty else None
        self._set_checks(self._classifier_channels_for(cr))

    def _on_save_label(self) -> None:
        if self._labels is None or self._current_well is None:
            return
        run = self._current_run()
        if run is None:
            return
        plate, well_id = self._current_well
        channels = {c for c, cb in self._label_checks.items() if cb.isChecked()}
        self._labels.set_label(run.run_tag, plate, well_id, channels,
                               notes=self._label_notes.text().strip() or None)
        self._label_status.setText(f"saved: {self._label_summary(channels)}")
        # gold-border the well and refresh the labelled-count in the status bar
        self._grid.mark_labelled(well_id, True, self._blend_channels(channels))
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
        hexc = self._blend_channels(self._classifier_channels_for(cr)) if cr is not None \
            else _EMPTY_HEX
        self._grid.mark_labelled(well_id, False, hexc)
        self._update_label_count()

    def _update_label_count(self) -> None:
        run = self._current_run()
        if self._labels is None or run is None:
            return
        n = self._labels.count(run.run_tag)
        self.statusBar().showMessage(f"{n} wells labelled in {run.run_tag}", 4000)

    # ── LDA (supervised) ─────────────────────────────────────────────
    def _evaluate_lda(self) -> None:
        """Train one LDA per channel on the hand labels, show the honest CV report,
        and cache the whole-run prediction so it can tint the grid."""
        run = self._current_run()
        if self._table is None or self._labels is None or run is None:
            return
        params = self._params()
        try:
            reports = _lda.evaluate(self._table, self._labels, run.run_tag, params)
            text = _lda.format_report(reports, self._chan_display)
        except ImportError as exc:
            self._show_text_dialog("LDA — scikit-learn missing", str(exc))
            return
        except ValueError as exc:      # no labels yet
            self._show_text_dialog("LDA", str(exc))
            return
        except Exception:
            import traceback
            self._show_text_dialog("LDA — error", traceback.format_exc())
            return
        # Cache whole-run prediction for the grid toggle (best-effort).
        try:
            self._cls_lda = _lda.predict(self._table, self._labels, run.run_tag, params)
            self._use_lda.setEnabled(self._cls_lda is not None and not self._cls_lda.empty)
        except Exception:
            self._cls_lda = None
            self._use_lda.setEnabled(False)
        # If the grid is already showing LDA calls, repoint it at the freshly
        # trained model instead of leaving stale predictions on screen.
        if self._use_lda.isChecked():
            self._toggle_call_source(True)
        self._show_text_dialog("LDA — per-channel evaluation", text)

    def _toggle_call_source(self, use_lda: bool) -> None:
        """Switch the grid/detail between the rule-based call and the LDA prediction."""
        if use_lda and self._cls_lda is not None and not self._cls_lda.empty:
            self._cls = self._cls_lda
        else:
            self._cls = self._cls_rule
        self._on_plate_changed(self._plate_combo.currentIndex())
        if self._current_well is not None:
            self._on_well_clicked(self._current_well[1])

    def _show_text_dialog(self, title: str, text: str) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        dlg.resize(620, 560)
        lay = QVBoxLayout(dlg)
        box = QTextEdit()
        box.setReadOnly(True)
        box.setStyleSheet("font-family: monospace; font-size: 12px;")
        box.setPlainText(text)
        lay.addWidget(box)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(dlg.reject)
        buttons.accepted.connect(dlg.accept)
        lay.addWidget(buttons)
        dlg.exec_() if hasattr(dlg, "exec_") else dlg.exec()

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
            chans = [c for c in (cr.get("pos_channels") or "").split(",") if c]
            if chans:
                call_txt = " + ".join(self._channel_name(c) for c in chans)
            else:
                call_txt = "negative / empty hole (no channel positive)"
        elif cr is not None:
            call_txt = "— (no hole ROI)"
        else:
            call_txt = "—"
        self._detail_form.addRow("Call (multi-label):", QLabel(call_txt))
        self._detail_form.addRow("Particles in hole:", QLabel(str(r.get("n_particles_hole", "—"))))
        # Per channel: hole/bg intensity, score, margin (signed dist to threshold),
        # positive?, and cells around. Row label = display name (channel in tooltip).
        for ch in self._table.channels:
            hole = r.get(f"hole_{ch}")
            bg = r.get(f"bg_{ch}")
            nbg = r.get(f"nbg_{ch}")
            score = cr.get(f"score_{ch}") if cr is not None else None
            margin = cr.get(f"margin_{ch}") if cr is not None else None
            occ = cr.get(f"occ_{ch}") if cr is not None else None
            proba = cr.get(f"proba_{ch}") if cr is not None else None
            pos = bool(cr.get(f"pos_{ch}")) if cr is not None else False
            if hole is not None and hole == hole:
                stxt = f"  score={score:.2f}" if score is not None and score == score else ""
                otxt = f"  occ={occ:.0%}" if occ is not None and occ == occ else ""
                mtxt = f"  Δthr={margin:+.2f}" if margin is not None and margin == margin else ""
                ptxt = f"  P={proba:.0%}" if proba is not None and proba == proba else ""
                flag = "  ✓POS" if pos else ""
                txt = f"hole={hole:.4f}  bg={bg:.4f}{stxt}{otxt}{ptxt}{mtxt}{flag}  cells around={nbg}"
            else:
                txt = "—"
            lbl = QLabel(txt)
            weight = "font-weight:bold;" if pos else ""
            lbl.setStyleSheet(f"color:{self._channel_hex(ch)};{weight}")
            row_lbl = QLabel(f"{self._channel_name(ch)}:")
            row_lbl.setToolTip(f"channel {ch}")
            self._detail_form.addRow(row_lbl, lbl)

        # Seed / show the ground-truth label controls for this well.
        self._refresh_label_controls(plate, well_id, cr)
        # Keep the well highlighted even when re-rendered programmatically (e.g.
        # after a Channels… edit repainted the grid).
        self._grid.select(well_id)

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
