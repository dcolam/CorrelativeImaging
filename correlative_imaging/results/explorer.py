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
    QSlider,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..diagnostics import (
    auto_contrast_limits,
    hex_to_rgb01,
    load_channel_stack,
    render_channels,
)
from . import cluster as _cluster
from . import lda as _lda
from .analysis import RunTable, load_run
from .classify import ClassifierParams, classify_wells
from .discovery import (
    RunGroup,
    diagnostic_view_sources,
    discover_runs,
    find_bf_projection,
    find_channel_stack,
    find_diagnostic_images,
)
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


class _ClusterWorker(QThread):
    """Compute the feature matrix + PCA/UMAP embeddings + HDBSCAN labels off the
    UI thread (UMAP on thousands of cells takes seconds). Loads whatever runs are
    requested itself so the GUI stays responsive."""
    done = Signal(object)     # dict of results
    error = Signal(str)

    def __init__(self, run_groups, params, fit_plate, want_umap, min_cluster_size,
                 omit_empty=False, families=("score",), scaling="run_plate",
                 cluster_method="hdbscan", n_clusters=5):
        super().__init__()
        self._run_groups = run_groups
        self._params = params
        self._fit_plate = fit_plate
        self._want_umap = want_umap
        self._min_cluster_size = min_cluster_size
        self._omit_empty = omit_empty
        self._families = families
        self._scaling = scaling
        self._cluster_method = cluster_method
        self._n_clusters = n_clusters

    def run(self) -> None:
        try:
            import numpy as np
            tables = {}
            for rg in self._run_groups:
                try:
                    tables[rg.run_tag] = load_run(rg)
                except Exception:
                    continue
            fm = _cluster.build_feature_matrix(
                tables, self._params, self._families, self._scaling, self._omit_empty)
            if fm is None or len(fm) < 3:
                self.done.emit({"empty": True})
                return
            if self._fit_plate is not None:
                mask = (fm.meta["plate"] == self._fit_plate).to_numpy()
            else:
                mask = np.ones(len(fm), dtype=bool)
            if mask.sum() < 3:
                self.done.emit({"empty": True})
                return
            Xf = fm.X[mask]
            meta_f = fm.meta[mask].reset_index(drop=True)
            coords = {"pca": _cluster.embed(Xf, "pca")}
            if self._want_umap:
                coords["umap"] = _cluster.embed(Xf, "umap")
            labels = _cluster.cluster_labels(
                Xf, self._cluster_method, self._min_cluster_size, self._n_clusters)
            self.done.emit({
                "empty": False, "meta": meta_f, "coords": coords,
                "labels": labels, "channels": fm.channels, "n_total": len(fm),
                "X": Xf, "feature_names": fm.feature_names,
            })
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


class _DiagnosticViewer(QWidget):
    """Reusable diagnostic-image viewer: a *kind* selector (whole / hole crop /
    background crop), a *view* selector (per-channel / composite jpg|tiff /
    brightfield), per-channel toggle + brightness controls, and the rendered
    image. Dropped into BOTH the Plate-tab detail panel and the Clusters-tab
    preview so a clicked cluster point shows the same rich view as a clicked well.

    Channel display (name + tint) is injected via :meth:`set_channel_display`, so
    the widget doesn't depend on the run's channel-display config directly."""

    def __init__(self, parent=None, min_height: int = 300):
        super().__init__(parent)
        self._channel_name_fn = lambda ch: ch
        self._channel_hex_fn = lambda ch: "#cccccc"
        self._well_id: str | None = None
        self._diag_dir: Path | None = None
        self._current_imgs: dict[str, Path] = {}
        self._view_sources: dict[str, Path] = {}
        self._chan_rows: dict[str, tuple] = {}
        self._preferred_img_kind = "whole"
        self._preferred_view = "per-channel"
        self._stack_cache: dict = {}
        self._bf_cache: dict = {}
        self._limits_cache: dict = {}
        self._build(min_height)

    def _build(self, min_height: int) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        box = QGroupBox("Diagnostic image")
        outer.addWidget(box)
        il = QVBoxLayout(box)
        sel_row = QHBoxLayout()
        self._img_combo = QComboBox()
        self._img_combo.currentIndexChanged.connect(self._on_img_changed)
        sel_row.addWidget(self._img_combo, stretch=1)
        self._view_combo = QComboBox()
        self._view_combo.setToolTip(
            "How to view this image: per-channel (raw channels + BF, adjustable), "
            "or the baked RGB composite (jpg/tiff — carries any ROI outline)."
        )
        self._view_combo.currentIndexChanged.connect(self._on_view_changed)
        sel_row.addWidget(self._view_combo, stretch=1)
        il.addLayout(sel_row)

        self._chan_ctrl = QWidget()
        self._chan_ctrl_lay = QVBoxLayout(self._chan_ctrl)
        self._chan_ctrl_lay.setContentsMargins(0, 0, 0, 0)
        self._chan_ctrl_lay.setSpacing(1)
        il.addWidget(self._chan_ctrl)
        self._diag_hint = QLabel("")
        self._diag_hint.setStyleSheet("font-size:10px; color:#888;")
        self._diag_hint.setWordWrap(True)
        il.addWidget(self._diag_hint)
        self._img_label = QLabel("(no cell selected)")
        self._img_label.setAlignment(Qt.AlignCenter)
        self._img_label.setMinimumHeight(min_height)
        self._img_label.setStyleSheet("background:#111; color:#888;")
        il.addWidget(self._img_label, stretch=1)

    def set_channel_display(self, name_fn, hex_fn) -> None:
        self._channel_name_fn = name_fn
        self._channel_hex_fn = hex_fn

    def clear(self) -> None:
        self._well_id = None
        self._img_combo.blockSignals(True); self._img_combo.clear(); self._img_combo.blockSignals(False)
        self._view_combo.blockSignals(True); self._view_combo.clear(); self._view_combo.blockSignals(False)
        self._clear_controls(); self._diag_hint.setText("")
        self._img_label.setPixmap(QPixmap()); self._img_label.setText("(no cell selected)")

    def show_well(self, diag_dir, well_id: str) -> None:
        """Load and render the diagnostic image(s) for one well, preserving the
        user's chosen kind/view across wells."""
        self._diag_dir = diag_dir
        self._well_id = well_id
        self._current_imgs = find_diagnostic_images(diag_dir, well_id)
        self._stack_cache = {}          # per-well; multichannel stacks are large
        self._bf_cache = {}
        self._limits_cache = {}
        kinds = _ordered_kinds(self._current_imgs.keys())
        self._img_combo.blockSignals(True)
        self._img_combo.clear()
        for kind in kinds:
            self._img_combo.addItem(kind, kind)
        target = self._preferred_img_kind if self._preferred_img_kind in kinds else (
            kinds[0] if kinds else None)
        idx = self._img_combo.findData(target) if target else -1
        self._img_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self._img_combo.blockSignals(False)
        if kinds:
            self._on_img_changed(self._img_combo.currentIndex())
        else:
            self._clear_controls()
            self._diag_hint.setText("")
            self._img_label.setPixmap(QPixmap())
            self._img_label.setText("(no diagnostic image on disk)")

    # ── kind → view → per-channel controls ───────────────────────────
    def _on_img_changed(self, _idx: int) -> None:
        kind = self._img_combo.currentData()
        if kind:
            self._preferred_img_kind = kind
        self._rebuild_view_combo(kind)

    def _rebuild_view_combo(self, kind) -> None:
        self._view_combo.blockSignals(True)
        self._view_combo.clear()
        self._view_sources = {}
        if self._well_id and kind:
            sources = diagnostic_view_sources(self._diag_dir, self._well_id, kind)
            if "per-channel" not in sources and self._get_bf(self._well_id) is not None:
                sources["brightfield"] = None
            self._view_sources = sources
            order = ["per-channel", "composite (tiff)", "composite (jpg)", "brightfield"]
            views = [v for v in order if v in sources]
            for v in views:
                self._view_combo.addItem(v, v)
            target = (self._preferred_view if self._preferred_view in views
                      else (views[0] if views else None))
            idx = self._view_combo.findData(target) if target else -1
            self._view_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self._view_combo.blockSignals(False)
        if self._view_combo.count():
            self._on_view_changed(self._view_combo.currentIndex())
        else:
            self._clear_controls()
            self._diag_hint.setText("")
            self._img_label.setPixmap(QPixmap())
            self._img_label.setText("(no diagnostic image on disk)")

    def _on_view_changed(self, _idx: int) -> None:
        view = self._view_combo.currentData()
        if view:
            self._preferred_view = view
        self._rebuild_channel_controls(view)
        self._render_diag()

    def _get_stack(self, well_id: str, kind: str):
        key = (well_id, kind)
        if key not in self._stack_cache:
            p = find_channel_stack(self._diag_dir, well_id, kind)
            try:
                self._stack_cache[key] = load_channel_stack(p) if p else None
            except Exception:
                self._stack_cache[key] = None
        return self._stack_cache[key]

    def _get_bf(self, well_id: str):
        if well_id not in self._bf_cache:
            p = find_bf_projection(self._diag_dir, well_id)
            arr = None
            if p is not None:
                try:
                    import numpy as np
                    import tifffile
                    arr = tifffile.imread(str(p))
                    if arr.ndim == 3:
                        arr = arr.max(axis=0)
                    arr = np.asarray(arr, dtype="float32")
                except Exception:
                    arr = None
            self._bf_cache[well_id] = arr
        return self._bf_cache[well_id]

    def _clear_controls(self) -> None:
        while self._chan_ctrl_lay.count():
            item = self._chan_ctrl_lay.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
            elif item.layout() is not None:
                self._clear_sublayout(item.layout())
        self._chan_rows = {}

    @staticmethod
    def _clear_sublayout(layout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
            elif item.layout() is not None:
                _DiagnosticViewer._clear_sublayout(item.layout())

    def _add_channel_row(self, label: str, source: tuple, checked: bool = True) -> None:
        row = QHBoxLayout(); row.setSpacing(4)
        name = "BF (brightfield)" if label == "BF" else self._channel_name_fn(label)
        hexc = "#cccccc" if label == "BF" else self._channel_hex_fn(label)
        cb = QCheckBox(name); cb.setChecked(checked)
        cb.setStyleSheet(f"color:{hexc}; font-weight:bold; font-size:11px;")
        cb.toggled.connect(self._render_diag)
        sld = QSlider(Qt.Horizontal); sld.setRange(0, 300); sld.setValue(100)
        sld.setFixedWidth(110)
        sld.setToolTip("display brightness (gain) — display-only, does not affect classification")
        sld.valueChanged.connect(self._render_diag)
        row.addWidget(cb); row.addWidget(sld); row.addStretch()
        self._chan_ctrl_lay.addLayout(row)
        self._chan_rows[label] = (cb, sld, source)

    def _rebuild_channel_controls(self, view) -> None:
        self._clear_controls()
        if not self._well_id:
            self._diag_hint.setText("")
            return
        well = self._well_id
        kind = self._img_combo.currentData()
        if view == "per-channel":
            stack = self._get_stack(well, kind)
            if stack is not None:
                _arr, labels = stack
                for i, lab in enumerate(labels):
                    self._add_channel_row(lab, source=("stack", i), checked=(lab != "BF"))
            self._diag_hint.setText(
                "Per-channel view: untick channels to view singly; sliders adjust "
                "display brightness (display-only). The ROI outline is only on the "
                "composite views."
            )
        elif view == "brightfield":
            self._add_channel_row("BF", source=("bf", None), checked=True)
            self._diag_hint.setText("Brightfield projection — slider adjusts display brightness.")
        else:
            self._diag_hint.setText(
                "Baked RGB composite (carries the stamped ROI outline, if the run enabled it)."
            )

    def _render_diag(self, *_args) -> None:
        if not self._well_id:
            return
        well = self._well_id
        kind = self._img_combo.currentData()
        view = self._view_combo.currentData()
        if view == "per-channel":
            stack = self._get_stack(well, kind)
            if stack is not None and self._chan_rows:
                arr, _labels = stack
                lkey = (well, kind)
                if lkey not in self._limits_cache:
                    self._limits_cache[lkey] = [auto_contrast_limits(arr[i])
                                                for i in range(arr.shape[0])]
                all_limits = self._limits_cache[lkey]
                planes, colors, gains, limits = [], [], [], []
                for lab, (cb, sld, source) in self._chan_rows.items():
                    if source[0] != "stack" or not cb.isChecked():
                        continue
                    planes.append(arr[source[1]])
                    colors.append((1.0, 1.0, 1.0) if lab == "BF"
                                  else hex_to_rgb01(self._channel_hex_fn(lab)))
                    gains.append(sld.value() / 100.0)
                    limits.append(all_limits[source[1]])
                rgb = render_channels(planes, colors, gains, limits)
                if rgb is None:
                    self._img_label.setPixmap(QPixmap())
                    self._img_label.setText("(all channels hidden)")
                    return
                self._show_rgb(rgb)
                return
        if view == "brightfield":
            bf = self._get_bf(well)
            row = self._chan_rows.get("BF")
            gain = (row[1].value() / 100.0) if row else 1.0
            if bf is not None:
                self._show_rgb(render_channels([bf], [(1.0, 1.0, 1.0)], [gain]))
                return
        path = self._view_sources.get(view)
        if path is None:
            self._img_label.setPixmap(QPixmap()); self._img_label.setText("(no image)")
            return
        pix = self._load_pixmap(path)
        if pix is None:
            self._img_label.setText(f"(could not load {path.name})")
            return
        self._img_label.setText("")
        self._img_label.setPixmap(
            pix.scaled(self._img_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _show_rgb(self, rgb) -> None:
        import numpy as np
        rgb = np.ascontiguousarray(rgb.astype("uint8"))
        h, w = rgb.shape[:2]
        img = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(img.copy())
        self._img_label.setText("")
        self._img_label.setPixmap(
            pix.scaled(self._img_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    @staticmethod
    def _load_pixmap(path: Path):
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


class _ClusterConfigDialog(QDialog):
    """Choose which feature families to include and how to standardise them for
    the Clusters tab. Returns ``(families, scaling)`` on accept."""

    _SCALING_LABELS = [
        ("per run × plate", "run_plate"),
        ("per run", "run"),
        ("global (no batch grouping)", "global"),
        ("per plate  — ⚠ erases DIV-timepoint biology", "plate"),
    ]

    def __init__(self, families, scaling, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cluster features & scaling")
        lay = QVBoxLayout(self)

        lay.addWidget(QLabel("Feature families to include:"))
        self._checks: dict[str, QCheckBox] = {}
        for fam in _cluster.FEATURE_FAMILIES:
            cb = QCheckBox(fam)
            cb.setChecked(fam in families)
            self._checks[fam] = cb
            lay.addWidget(cb)

        lay.addWidget(QLabel("Standardise (z-score each feature within):"))
        self._scaling = QComboBox()
        for label, key in self._SCALING_LABELS:
            self._scaling.addItem(label, key)
        idx = self._scaling.findData(scaling)
        self._scaling.setCurrentIndex(idx if idx >= 0 else 0)
        lay.addWidget(self._scaling)

        note = QLabel(
            "Per-column standardisation always happens (so no feature dominates by "
            "raw scale); the grouping only sets global-vs-within-batch. Per-run is "
            "the safe default; per-plate removes plate/DIV-timepoint differences."
        )
        note.setWordWrap(True); note.setStyleSheet("font-size:10px; color:#888;")
        lay.addWidget(note)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

    def result_config(self):
        fams = [f for f, cb in self._checks.items() if cb.isChecked()]
        return (fams or list(_cluster.DEFAULT_FAMILIES)), self._scaling.currentData()


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
        # Clusters tab state
        self._cluster_worker: _ClusterWorker | None = None
        self._cluster_result: dict | None = None   # last worker payload
        self._cl_families = list(_cluster.DEFAULT_FAMILIES)   # feature families (pop-up)
        self._cl_scaling = _cluster.DEFAULT_SCALING           # standardisation grouping
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

        # Folder/run/plate selectors are shared above; the rest lives in tabs:
        # "Plate" (grid + detail) and "Clusters" (feature-space embedding).
        self._tabs = QTabWidget()
        outer.addWidget(self._tabs, stretch=1)
        plate_tab = QWidget()
        ptl = QVBoxLayout(plate_tab)

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
        ptl.addLayout(legend_row)
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
        ptl.addWidget(cls_box)

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
        ptl.addWidget(split, stretch=1)

        self._tabs.addTab(plate_tab, "Plate")
        self._tabs.addTab(self._build_clusters_tab(), "Clusters")

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

        # Diagnostic image viewer (reusable widget — same one used in the
        # Clusters-tab preview). Per-channel/composite/brightfield + brightness.
        self._diag_viewer = _DiagnosticViewer(min_height=320)
        pl.addWidget(self._diag_viewer, stretch=1)

        self._napari_btn = QPushButton("Open raw image + ROI in napari")
        self._napari_btn.setEnabled(False)
        self._napari_btn.clicked.connect(self._on_open_napari)
        pl.addWidget(self._napari_btn)

        self._current_well = None
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
        # keep the cluster scatter's plate filter in sync (redraw only, no refit)
        if getattr(self, "_cluster_result", None) is not None:
            self._cluster_draw()

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
            self._diag_viewer.clear()
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

        # diagnostic images — the reusable viewer preserves the chosen kind/view
        # across wells and shares behaviour with the Clusters-tab preview.
        self._current_well = (plate, well_id)
        run = self._current_run()
        pr = next((p for p in run.plates if p.plate == plate), None) if run else None
        self._diag_viewer.set_channel_display(self._channel_name, self._channel_hex)
        self._diag_viewer.show_well(pr.diagnostics_dir if pr else None, well_id)
        self._napari_btn.setEnabled(self._viewer is not None)

    # ── Clusters tab: feature-space embedding + unbiased clustering ───
    def _build_clusters_tab(self) -> QWidget:
        tab = QWidget()
        lay = QVBoxLayout(tab)

        # Row 1: view + colour-by + scope toggles
        row1 = QHBoxLayout()
        self._cl_method = QComboBox()
        self._cl_method.addItem("PCA", "pca")
        self._cl_method.addItem("UMAP", "umap")
        self._cl_method.addItem("LDA (supervised)", "lda")
        self._cl_method.setToolTip(
            "PCA/UMAP are unsupervised; LDA is a supervised LD1-vs-LD2 projection "
            "fit on your hand labels (needs ≥3 labelled classes)."
        )
        self._cl_method.currentIndexChanged.connect(self._cluster_draw)
        self._cl_colorby = QComboBox()
        for label, key in (("cluster", "cluster"), ("classifier call", "call"),
                           ("hand label", "label"), ("plate", "plate"),
                           ("feature…", "feature")):
            self._cl_colorby.addItem(f"colour: {label}", key)
        self._cl_colorby.setToolTip(
            "Colour points by posthoc cluster, the classifier's call, your hand "
            "labels, plate (batch check), or a chosen feature."
        )
        self._cl_colorby.currentIndexChanged.connect(self._on_colorby_changed)
        self._cl_feature = QComboBox()
        self._cl_feature.setToolTip("Feature to colour by (continuous).")
        self._cl_feature.setEnabled(False)
        self._cl_feature.currentIndexChanged.connect(self._cluster_draw)
        # Scope: NEVER mix runs. Either all plates of the current run, or just the
        # selected plate. (One control, replacing the old all-runs/fit-plate pair.)
        self._cl_scope = QComboBox()
        self._cl_scope.addItem("all plates in this run", "run")
        self._cl_scope.addItem("selected plate only", "plate")
        self._cl_scope.setToolTip(
            "What to embed — all plates of the current run, or only the selected "
            "plate. Runs are never mixed. Changing this needs Recompute.")
        self._cl_omit_empty = QCheckBox("omit empty wells")
        self._cl_omit_empty.setToolTip(
            "Exclude wells the classifier calls negative/empty. Needs Recompute.")
        row1.addWidget(QLabel("view:")); row1.addWidget(self._cl_method)
        row1.addWidget(self._cl_colorby); row1.addWidget(self._cl_feature)
        row1.addWidget(QLabel("scope:")); row1.addWidget(self._cl_scope)
        row1.addWidget(self._cl_omit_empty)
        row1.addStretch()
        lay.addLayout(row1)

        # Row 2: features/scaling + clustering + recompute/export
        row2 = QHBoxLayout()
        cfg_btn = QPushButton("Features / scaling …")
        cfg_btn.setToolTip("Choose which feature families to include and how to standardise them.")
        cfg_btn.clicked.connect(self._edit_cluster_config)
        self._cl_algo = QComboBox()
        for label, key in (("HDBSCAN", "hdbscan"), ("KMeans", "kmeans"),
                           ("Agglomerative", "agglomerative")):
            self._cl_algo.addItem(label, key)
        self._cl_algo.setToolTip("Posthoc clustering of the FEATURE space (not the 2-D coords).")
        self._cl_algo.currentIndexChanged.connect(self._on_algo_changed)
        self._cl_k = QSpinBox()
        self._cl_k.setRange(2, 50); self._cl_k.setValue(5); self._cl_k.setPrefix("k ")
        self._cl_k.setToolTip("Number of clusters (KMeans / Agglomerative).")
        self._cl_k.setEnabled(False)
        self._cl_minsize = QSpinBox()
        self._cl_minsize.setRange(2, 500); self._cl_minsize.setValue(15)
        self._cl_minsize.setPrefix("min clust ")
        self._cl_minsize.setToolTip("HDBSCAN minimum cluster size.")
        recompute = QPushButton("Recompute")
        recompute.clicked.connect(self._cluster_recompute)
        self._cl_export = QPushButton("Export CSV")
        self._cl_export.setToolTip("Save coords + cluster + label + features to CSV.")
        self._cl_export.clicked.connect(self._export_clusters)
        row2.addWidget(cfg_btn)
        row2.addWidget(QLabel("cluster:")); row2.addWidget(self._cl_algo)
        row2.addWidget(self._cl_k); row2.addWidget(self._cl_minsize)
        row2.addWidget(recompute); row2.addWidget(self._cl_export)
        row2.addStretch()
        lay.addLayout(row2)

        self._cl_status = QLabel("Load a run, then press Recompute to embed the cells.")
        self._cl_status.setStyleSheet("font-size:11px; color:#aaa;")
        self._cl_status.setWordWrap(True)
        lay.addWidget(self._cl_status)

        split = QSplitter(Qt.Horizontal)
        self._cl_canvas = None
        try:
            import matplotlib
            matplotlib.use("qtagg")
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure
            self._cl_fig = Figure(figsize=(5, 5), layout="tight")
            self._cl_ax = self._cl_fig.add_subplot(111)
            self._cl_canvas = FigureCanvasQTAgg(self._cl_fig)
            self._cl_canvas.mpl_connect("pick_event", self._on_cluster_pick)
            split.addWidget(self._cl_canvas)
        except Exception as exc:
            split.addWidget(QLabel(f"matplotlib unavailable — clustering plot disabled:\n{exc}"))

        prev = QWidget(); pv = QVBoxLayout(prev)
        self._cl_prev_title = QLabel("Click a point to see its cell.")
        self._cl_prev_title.setWordWrap(True)
        self._cl_prev_title.setStyleSheet("font-weight:bold; font-size:12px;")
        pv.addWidget(self._cl_prev_title)
        self._cl_diag_viewer = _DiagnosticViewer(min_height=260)
        pv.addWidget(self._cl_diag_viewer, stretch=1)
        split.addWidget(prev)
        split.setStretchFactor(0, 1); split.setStretchFactor(1, 0)
        split.setSizes([640, 340])
        lay.addWidget(split, stretch=1)

        self._cl_disp_idx = None      # scatter-point index → meta row index
        return tab

    def _cluster_recompute(self) -> None:
        if not self._runs:
            self._cl_status.setText("No runs loaded.")
            return
        run = self._current_run()
        plate = self._plate_combo.currentData()
        # Runs are NEVER mixed — always embed within the current run only.
        groups = [run] if run else []
        if not groups:
            self._cl_status.setText("No run selected.")
            return
        plate_only = self._cl_scope.currentData() == "plate"
        fit_plate = plate if plate_only else None
        umap_err = _cluster.umap_import_error()
        want_umap = umap_err is None
        scope = f"plate {plate}" if plate_only else "all plates in run"
        empty_note = "; empty wells omitted" if self._cl_omit_empty.isChecked() else ""
        msg = f"Computing embedding ({scope}{empty_note}) …"
        # If UMAP is the chosen view but unavailable, say WHY (the real import error).
        if self._cl_method.currentData() == "umap" and umap_err is not None:
            msg = f"UMAP unavailable — {umap_err}. Showing PCA. {msg}"
        self._cl_status.setText(msg)
        if self._cluster_worker and self._cluster_worker.isRunning():
            self._cluster_worker.wait()
        self._cluster_worker = _ClusterWorker(
            groups, self._params(), fit_plate, want_umap, self._cl_minsize.value(),
            self._cl_omit_empty.isChecked(), tuple(self._cl_families), self._cl_scaling,
            self._cl_algo.currentData(), self._cl_k.value())
        self._cluster_worker.done.connect(self._on_cluster_done)
        self._cluster_worker.error.connect(
            lambda m: self._cl_status.setText(f"Cluster error:\n{m}"))
        self._cluster_worker.start()

    def _on_cluster_done(self, result: dict) -> None:
        if result.get("empty"):
            self._cluster_result = None
            self._cl_status.setText("Not enough cells with holes to embed (need ≥3).")
            self._cluster_draw()
            return
        import numpy as np
        self._cluster_result = result
        labs = result["labels"]
        n_clusters = len({int(x) for x in labs} - {-1})
        n_noise = int((labs == -1).sum())
        umap_note = "" if "umap" in result["coords"] else " (UMAP unavailable — PCA only)"
        scale_note = "; ⚠ per-plate scaling removes DIV-timepoint biology" \
            if self._cl_scaling in ("plate", "run_plate") else ""
        self._cl_status.setText(
            f"{len(result['meta'])} cells embedded of {result['n_total']} total (this run) — "
            f"{self._cl_algo.currentText()}: {n_clusters} clusters, {n_noise} unclustered. "
            f"Features: {', '.join(self._cl_families)}; scaled {self._cl_scaling}"
            f"{umap_note}{scale_note}."
        )
        # Populate the colour-by-feature dropdown from this result's features.
        self._cl_feature.blockSignals(True)
        self._cl_feature.clear()
        for name in result.get("feature_names", []):
            self._cl_feature.addItem(name, name)
        self._cl_feature.blockSignals(False)
        self._cluster_draw()

    def _cluster_point_colors(self, disp):
        """Return (colors list, title suffix, legend pairs) for the displayed
        points under the current colour-by mode."""
        import numpy as np
        import matplotlib
        res = self._cluster_result
        meta = res["meta"]
        mode = self._cl_colorby.currentData()
        tab = matplotlib.colormaps["tab10"]

        def categorical(values, label):
            cats = sorted({str(v) for v in values})
            cmap = {c: tab(i % 10) for i, c in enumerate(cats)}
            colors = [cmap[str(v)] for v in values]
            legend = [(c, cmap[c]) for c in cats][:10]
            return colors, label, legend

        if mode == "cluster":
            labs = res["labels"][disp]
            colors = ["#666666" if int(l) == -1 else tab(int(l) % 10) for l in labs]
            cats = sorted({int(l) for l in labs})
            legend = [("noise" if c == -1 else f"c{c}",
                       "#666666" if c == -1 else tab(c % 10)) for c in cats][:11]
            return colors, "cluster", legend
        if mode == "call":
            vals = [v or "negative" for v in meta["pos_channels"][disp]]
            return categorical(vals, "classifier call")
        if mode == "run":
            return categorical(list(meta["run_tag"][disp]), "run")
        if mode == "plate":
            return categorical(list(meta["plate"][disp]), "plate")
        if mode == "label":
            vals = [self._hand_label_str(meta.iloc[i]) for i in np.where(disp)[0]]
            return categorical(vals, "hand label")
        if mode == "feature":
            fname = self._cl_feature.currentData()
            X = res.get("X"); names = res.get("feature_names", [])
            if fname in names and X is not None:
                col = X[disp, names.index(fname)]
                norm = matplotlib.colors.Normalize()
                colors = matplotlib.colormaps["viridis"](norm(col))
                return colors, f"feature {fname}", None
        return ["#4477aa"] * int(disp.sum()), "", None

    def _hand_label_str(self, meta_row) -> str:
        """The hand label for a meta row as a channel-set string, or 'unlabelled'."""
        v = self._hand_label_or_none(meta_row)
        return v if v is not None else "unlabelled"

    def _hand_label_or_none(self, meta_row):
        """Hand-label class string for a meta row, or ``None`` if unlabelled
        (so the multi-class LDA fits only on genuinely labelled cells)."""
        if self._labels is None:
            return None
        lab = self._labels.get_label(meta_row["run_tag"], meta_row["plate"], meta_row["well_id"])
        if lab is None:
            return None
        chans = lab.get("channels") or set()
        return ",".join(sorted(chans)) if chans else "negative"

    def _cluster_draw(self, *_args) -> None:
        if self._cl_canvas is None:
            return
        import numpy as np
        ax = self._cl_ax
        ax.clear()
        res = self._cluster_result
        if not res:
            ax.text(0.5, 0.5, "press Recompute", ha="center", va="center",
                    transform=ax.transAxes, color="#888")
            ax.set_xticks([]); ax.set_yticks([])
            self._cl_canvas.draw_idle()
            return
        method = self._cl_method.currentData()
        # LDA is a supervised embedding computed lazily from the current hand
        # labels (fast; no worker). PCA/UMAP come from the worker payload.
        if method == "lda" and "lda" not in res["coords"]:
            classes = [self._hand_label_or_none(res["meta"].iloc[i])
                       for i in range(len(res["meta"]))]
            lda_coords, note = _cluster.lda_embedding(res["X"], classes)
            res["coords"]["lda"] = lda_coords
            res["lda_note"] = note
        coords = res["coords"].get(method)
        if coords is None:
            reason = res.get("lda_note", "") if method == "lda" \
                else "(install umap-learn, then Recompute)"
            ax.text(0.5, 0.5, f"{method.upper()} not available\n{reason}",
                    ha="center", va="center", transform=ax.transAxes, color="#888", wrap=True)
            ax.set_xticks([]); ax.set_yticks([])
            self._cl_canvas.draw_idle()
            return
        meta = res["meta"]
        # The worker already scoped the embedding (all plates of the run, or the
        # selected plate), so every computed cell is shown.
        disp = np.ones(len(meta), dtype=bool)
        self._cl_disp_idx = np.where(disp)[0]
        if disp.sum() == 0:
            ax.text(0.5, 0.5, "no cells", ha="center", va="center",
                    transform=ax.transAxes, color="#888")
            ax.set_xticks([]); ax.set_yticks([])
            self._cl_canvas.draw_idle()
            return
        colors, by, legend = self._cluster_point_colors(disp)
        sc = ax.scatter(coords[disp, 0], coords[disp, 1], c=colors, s=18,
                        picker=5, linewidths=0)
        if legend:
            from matplotlib.lines import Line2D
            handles = [Line2D([0], [0], marker="o", linestyle="", markersize=6,
                              markerfacecolor=c, markeredgecolor="none", label=str(name))
                       for name, c in legend]
            ax.legend(handles=handles, fontsize=7, loc="best", framealpha=0.8)
        ax.set_title(f"{method.upper()} — {int(disp.sum())} cells, coloured by {by}")
        if method == "lda":
            ax.set_xlabel("LD1"); ax.set_ylabel("LD2")
        ax.set_xticks([]); ax.set_yticks([])
        self._cl_canvas.draw_idle()

    def _on_colorby_changed(self, *_a) -> None:
        self._cl_feature.setEnabled(self._cl_colorby.currentData() == "feature")
        self._cluster_draw()

    def _on_algo_changed(self, *_a) -> None:
        algo = self._cl_algo.currentData()
        self._cl_k.setEnabled(algo in ("kmeans", "agglomerative"))
        self._cl_minsize.setEnabled(algo == "hdbscan")

    def _edit_cluster_config(self) -> None:
        dlg = _ClusterConfigDialog(self._cl_families, self._cl_scaling, self)
        if dlg.exec_() if hasattr(dlg, "exec_") else dlg.exec():
            self._cl_families, self._cl_scaling = dlg.result_config()
            self._cl_status.setText(
                f"Features: {', '.join(self._cl_families)}; scaled {self._cl_scaling}. "
                "Press Recompute to apply.")

    def _export_clusters(self) -> None:
        res = self._cluster_result
        if not res:
            self._cl_status.setText("Nothing to export — Recompute first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export cluster table", "clusters.csv",
                                              "CSV (*.csv)")
        if not path:
            return
        import numpy as np
        import pandas as pd
        df = res["meta"].copy()
        df["cluster"] = res["labels"]
        for m, coords in res["coords"].items():
            df[f"{m}1"] = coords[:, 0]; df[f"{m}2"] = coords[:, 1]
        X = res.get("X"); names = res.get("feature_names", [])
        if X is not None:
            for i, name in enumerate(names):
                df[name] = X[:, i]
        df["hand_label"] = [self._hand_label_str(df.iloc[i]) for i in range(len(df))]
        try:
            df.to_csv(path, index=False)
            self._cl_status.setText(f"Exported {len(df)} cells → {path}")
        except Exception as exc:
            self._cl_status.setText(f"Export failed: {exc}")

    def _on_cluster_pick(self, event) -> None:
        if self._cluster_result is None or self._cl_disp_idx is None:
            return
        ind = list(getattr(event, "ind", []))
        if not ind:
            return
        meta_idx = int(self._cl_disp_idx[int(ind[0])])
        row = self._cluster_result["meta"].iloc[meta_idx]
        self._cluster_show_preview(row["run_tag"], row["plate"], row["well_id"])

    def _diag_dir_for(self, run_tag: str, plate: str):
        for rg in self._runs:
            if rg.run_tag == run_tag:
                for pr in rg.plates:
                    if pr.plate == plate:
                        return pr.diagnostics_dir
        return None

    def _cluster_show_preview(self, run_tag: str, plate: str, well_id: str) -> None:
        self._cl_prev_title.setText(f"{plate} — well {well_id}\n({run_tag})")
        diag = self._diag_dir_for(run_tag, plate)
        # Same rich viewer as the Plate tab: kind/view selectors, per-channel
        # toggles + brightness, jpg/tiff, BF.
        self._cl_diag_viewer.set_channel_display(self._channel_name, self._channel_hex)
        self._cl_diag_viewer.show_well(diag, well_id)

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
