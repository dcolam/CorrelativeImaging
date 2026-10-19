"""Cell reconstruction inside napari: the viewer is the canvas, the tool lives in two docks.

  bottom Wells: a plate view (coloured by status / dominant channel / P; labelled wells framed) and a searchable list.
         Before Classify the colours are the Detect preview, afterwards the calls; the dock says which.
  right  Run · Train · Results · Settings (panel.py)

Long work (detect, classify, train) runs in a napari thread worker; progress lands in the Run tab.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from qtpy.QtCore import QObject, Qt, Signal
from qtpy.QtWidgets import QComboBox, QHBoxLayout, QLabel, QLineEdit, QMenu, QTabWidget, QVBoxLayout, QWidget
from qtpy.QtGui import QCursor

from ..config import MODELS, Config
from ..labels import CLASSES, LabelStore, match_labels
from .display import WellDisplay
from .panel import Panel
from .plate import PlateView
from .tables import FrameModel, FrameView
from .theme import CLASS_COLORS, CLASS_KEYS, call_color, chip, stylesheet, theme


class _Bus(QObject):
    progress = Signal(int, int, str)


class WellBrowser(QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget()
        outer.addWidget(self.tabs)
        self.plate = PlateView(lambda c: app.cfg.color(c), lambda c: app.cfg.display_name(c))
        self.plate.well_clicked.connect(app.show_well)
        self.tabs.addTab(self.plate, "Plate")
        lw = QWidget()
        self.tabs.addTab(lw, "List")
        from .pca import ParticleMap
        self.pca = ParticleMap(app)
        self.tabs.addTab(self.pca, "Particles (PCA)")
        v = QVBoxLayout(lw)
        v.setContentsMargins(4, 4, 4, 4)
        h = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("search plate / well / call…")
        self.search.textChanged.connect(self.filter)
        h.addWidget(self.search, 1)
        self.status = QComboBox()
        self.status.addItems(["all", "one cell", "several cells", "no cell", "uncertain", "one object",
                              "several objects", "neighbours only", "no particles", "labelled", "unlabelled",
                              "test wells"])
        self.status.currentTextChanged.connect(self.filter)
        h.addWidget(self.status)
        v.addLayout(h)
        from .plate import STATUS_COLORS
        chan = lambda c: call_color(app.call_key(c), app.cfg.color) if isinstance(c, str) and c else None
        self.model = FrameModel(colors={"call": chan, "dominant": chan,
                                        "status": lambda c: STATUS_COLORS.get(c) if isinstance(c, str) else None})
        self.view = FrameView(self.model)
        self.view.clicked.connect(self.clicked)
        self.view.activated.connect(self.clicked)
        v.addWidget(self.view, 1)
        self.count = QLabel("")
        self.count.setProperty("role", "hint")
        v.addWidget(self.count)
        self.full = pd.DataFrame()

    def set(self, df: pd.DataFrame, stage: str = ""):
        self.full = df
        self.filter()
        self.plate.set(df, stage)

    def filter(self, *_):
        d = self.full
        s = self.status.currentText()
        if len(d) and s != "all":
            if s == "uncertain":
                d = d[d.get("call", pd.Series(index=d.index, dtype=str)) == "uncertain"]
            elif s == "labelled":
                d = d[d.labels > 0]
            elif s == "unlabelled":
                d = d[d.labels == 0]
            elif s == "test wells":
                d = d[d.test == "✓"]
            elif "status" in d:
                d = d[d.status == s]
        self.model.set(d.drop(columns=["status_shown", "dominant_shown"], errors="ignore"))
        self.view.proxy.setFilterFixedString(self.search.text())
        self.count.setText(f"{self.view.proxy.rowCount()} of {len(self.full)} wells")

    def clicked(self, idx):
        r = self.view.current_row()
        if r is not None:
            self.app.show_well(r.plate_id, r.well_id)

    def step(self, k: int):
        i = self.view.currentIndex().row() + k
        if 0 <= i < self.view.proxy.rowCount():
            self.view.selectRow(i)
            self.clicked(None)


class ReconstructionApp:
    def __init__(self, viewer):
        self.v = viewer
        self.cfg = Config()
        self.res = None
        self.store = None
        self.model = None
        self.model_file = None
        self.trained = None
        self._merge_from = None
        self.label_mode = False
        self.plate_map = None
        self.bus = _Bus()
        self.bus.progress.connect(self._progress)
        self.display = WellDisplay(viewer)
        self.panel = Panel(self)
        self.browser = WellBrowser(self)
        t = theme(viewer)
        for w in (self.panel, self.browser):
            w.setStyleSheet(stylesheet(t))
        viewer.window.add_dock_widget(self.panel, area="right", name="Cell reconstruction")
        dw = viewer.window.add_dock_widget(self.browser, area="bottom", name="Wells")
        try:                                   # a 384-well plate fits across the bottom
            dw.parentWidget().resizeDocks([dw], [330], Qt.Vertical)
        except Exception:
            pass
        viewer.mouse_drag_callbacks.append(self._on_click)
        for c in CLASSES:
            viewer.bind_key(CLASS_KEYS[c], lambda v, c=c: self.set_label(c), overwrite=True)
        viewer.bind_key("Delete", lambda v: self.remove_label(), overwrite=True)
        viewer.bind_key("l", lambda v: self.panel.train.mode.toggle(), overwrite=True)
        viewer.bind_key("x", lambda v: self.panel.train.b_cut.toggle(), overwrite=True)
        viewer.bind_key("n", lambda v: self.browser.step(1), overwrite=True)
        viewer.bind_key("p", lambda v: self.browser.step(-1), overwrite=True)
        viewer.scale_bar.visible = True
        viewer.scale_bar.unit = "um"
        viewer.text_overlay.visible = True
        viewer.text_overlay.position = "top_left"
        viewer.text_overlay.font_size = 11
        self.panel.settings.set_config(self.cfg)

    # -- messages
    def say(self, msg: str):
        self.panel.run.log.appendPlainText(msg)
        try:
            self.v.status = msg
        except Exception:
            pass

    def _progress(self, i, n, msg):
        if n:
            self.panel.run.progress.setMaximum(n)
            self.panel.run.progress.setValue(i)
        if msg:
            self.say(msg)

    # -- names
    def call_text(self, call) -> str:
        if not isinstance(call, str) or call in ("none", "uncertain", "no cell"):
            return call
        return "+".join(self.cfg.display_name(c) for c in call.split("+"))

    def call_key(self, text) -> str:
        if not isinstance(text, str) or text in ("none", "uncertain", "no cell"):
            return text
        back = {self.cfg.display_name(c): c for c in (self.res.channels if self.res else [])}
        return "+".join(back.get(t, t) for t in text.split("+"))

    # -- config / store / results
    def set_config(self, cfg: Config, from_form: bool = False):
        self.cfg = cfg
        from ..pipeline import set_path_map
        set_path_map(cfg.path_map)                     # the run's TIFs are read back through it
        self.model = None
        if cfg.model and cfg.model_path() and cfg.model_path().exists():
            from ..learn import ParticleModel
            try:
                self.model = ParticleModel.load(cfg.model_path())
                self.model_file = cfg.model
            except Exception as e:
                self.say(f"model not loaded: {e}")
        if not from_form:
            self.panel.settings.set_config(cfg)
        self.panel.run.set_models(cfg.model)
        chans = self.res.channels if self.res else list(self.panel.run.chan_rows) or cfg.channels
        if chans:
            self.panel.run.set_channels(chans, cfg)
        self.panel.settings.show_model_match()
        self.refresh()

    def read_config(self) -> Config:
        """Settings tab + the Run tab's channel table and model."""
        cfg = self.panel.settings.read()
        cfg = self.panel.run.apply_channels(cfg)
        cfg.model = self.panel.run.model.currentText().strip()
        return cfg

    def set_store(self, path: str):
        if not path:
            return
        self.store = LabelStore(path)
        from qtpy.QtCore import QSettings
        QSettings("CorrelativeImaging", "reconstruction").setValue("label_library", str(self.store.path))
        self.panel.train.store_path.setText(str(self.store.path))
        self.refresh()

    def _last_store(self) -> str:
        from qtpy.QtCore import QSettings
        for app in ("reconstruction", "identity"):          # "identity": the tool's earlier name
            last = QSettings("CorrelativeImaging", app).value("label_library", "")
            if last and Path(last).exists():
                return last
        return ""

    def open_results(self, d: str):
        from ..pipeline import Results
        try:
            self.res = Results(d)
        except Exception as e:
            self.say(f"cannot open {d}: {e}")
            return
        cfg = self.res.config
        if not cfg.model and self.cfg.model:
            cfg.model = self.cfg.model
        if self.res.run.get("model"):
            self.say(f"classified with model arm '{self.res.run['model'].get('arm')}'")
        if self.res.run.get("warning"):
            self.say("WARNING " + self.res.run["warning"])
        if self.store is None:
            self.set_store(self._last_store() or str(Path(d).parent / "ci_labels.db"))
        folders = self.panel.train.folders
        if str(self.res.dir) not in folders.toPlainText():
            folders.setPlainText((folders.toPlainText().strip() + "\n" + str(self.res.dir)).strip())
        self.panel.run.out.setText(str(self.res.dir))
        self.say(f"opened {d}: {len(self.res.wells)} wells, dataset {self.res.dataset}")
        self.set_config(cfg)
        self.recommend()
        self.update_pca()

    def all_labels(self):
        """Hand labels of the open dataset matched to its particles (for the particle map)."""
        if self.store is None or self.res is None:
            return None
        L = self.store.frame(self.res.dataset)
        return match_labels(L, self.res.particles, self.res.wells) if len(L) else None

    def update_pca(self):
        try:
            self.browser.pca.set_results(self.res, self.all_labels())
        except Exception as e:
            self.say(f"particle map not drawn: {e}")

    def prefill(self, root: str, run_tag: str):
        """Called from `ci gui` after a run: open the run's existing results, or prepare Detect for it."""
        for name in (f"reconstruction_{run_tag}", f"identity_{run_tag}"):
            d = Path(root) / name
            if (d / "run.json").exists():
                self.open_results(str(d))
                return
        self.panel.run.root.setText(str(root))
        self.panel.run.find_runs()
        self.panel.run.run.setCurrentText(run_tag)
        self.panel.setCurrentIndex(0)
        self.say(f"run {run_tag}: check the channel names, colours and roles, then Detect")

    def recommend(self):
        """Data-driven suggestions for the Settings tab (applied only on click)."""
        from ..preview import suggest_markers
        items = []
        res = self.res
        if res is not None and "n_inside" in res.wells:
            ident = self.cfg.identity_channels(res.channels)
            for ch in suggest_markers(res.wells, ident):
                def apply(_=False, ch=ch):
                    cfg = self.read_config()
                    cfg.markers = sorted(set(cfg.markers) | {ch})
                    self.set_config(cfg)
                    self.say(f"{ch} is now a marker channel: Classify again")
                items.append((f"{self.cfg.display_name(ch)} has signal in nearly every hole that holds anything: "
                              f"make it a marker (measured, not identity)?", apply))
        if self.model is not None and res is not None:
            from ..config import complete_feature_config
            fc = complete_feature_config(self.model.spec.get("feature_config") or {})
            diff = [k for k, v in fc.items() if res.config.feature_config().get(k) != v] \
                if self.model.spec.get("feature_config") else []
            if diff:
                items.append((f"This run was detected with other feature settings than the model was trained on "
                              f"({', '.join(diff)}). Detect again with the model's settings, or retrain.", None))
        self.panel.settings.recommend(items)

    def refresh(self):
        res = self.res
        self.panel.run.update(res, self.store, self.model.spec if self.model else None)
        if self.store is not None:
            self.panel.train.counts.set(self.store.counts())
        self.panel.results.update(res)
        if res is None:
            return
        W = res.wells
        cols = [c for c in ("preview_status", "preview_dominant", "n_inside") if c in W]
        w = W[["plate_id", "well_id"] + cols].copy()
        classified = res.calls is not None
        if classified:
            keep = [c for c in ("status", "call", "dominant", "p_combo", "n_cells", "p_cell") if c in res.calls]
            w = w.merge(res.calls[["plate_id", "well_id"] + keep], how="left")
            w["status_shown"] = np.where(w["call"] == "uncertain", "uncertain", w["status"])
            w["dominant_shown"] = w["call"].where(~w["call"].isin(["no cell"]), "")
        else:
            w["status_shown"] = w.get("preview_status", "")
            w["dominant_shown"] = w.get("preview_dominant", "")
        nl = pd.Series(dtype=int)
        tw = set()
        if self.store is not None:
            L = self.store.frame(res.dataset)
            if len(L):
                nl = L.groupby(["plate_id", "well_id"]).size()
            tw = self.store.test_wells(res.dataset)
        w["labels"] = [int(nl.get((p, q), 0)) for p, q in zip(w.plate_id, w.well_id)]
        w["test"] = ["✓" if (p, q) in tw else "" for p, q in zip(w.plate_id, w.well_id)]
        if self.plate_map is not None:
            extra = [c for c in self.plate_map.columns if c not in ("plate_id", "well_id")][:3]
            w = w.merge(self.plate_map[["plate_id", "well_id"] + extra], how="left")
        if "call" in w:
            w["call"] = w["call"].map(self.call_text)
        w = w.rename(columns={"p_combo": "P"})
        w["status"] = w["status_shown"]
        extra = [c for c in (self.plate_map.columns if self.plate_map is not None else []) if c not in ("plate_id", "well_id")][:3]
        if classified:
            show = ["plate_id", "well_id", "status", "call", "P", "n_cells", "labels", "test"] + extra
        else:
            w["dominant"] = w["dominant_shown"].map(lambda d: self.call_text(d) if isinstance(d, str) and d else "")
            show = ["plate_id", "well_id", "status", "dominant", "n_inside", "labels", "test"] + extra
        show = [c for c in show if c in w]
        stage = "calls (after Classify)" if classified else "Detect preview (no model yet)"
        self.browser.set(w[show].assign(status_shown=w.status_shown, dominant_shown=w.dominant_shown), stage)

    def set_plate_map(self, m):
        self.plate_map = m
        self.refresh()

    # -- one well
    def show_well(self, plate_id, well_id):
        if self.res is None:
            return
        try:
            got = self.res.well(plate_id, well_id)
        except (FileNotFoundError, OSError) as e:
            self.say(f"{plate_id} {well_id}: {e}")
            return
        if got is None:
            self.say(f"{plate_id} {well_id}: not in the cache")
            return
        wd, maps = got
        P = self.res.particles
        P = P[(P.plate_id == plate_id) & (P.well_id == well_id)]
        G = self.res.recon(plate_id, well_id)
        cells = self.res.cells
        cells = cells[(cells.plate_id == plate_id) & (cells.well_id == well_id)] if cells is not None else None
        if self.model is not None and "pclass" not in P:
            P = self.model.predict(P)
        labels = self._well_labels(P)
        self.display.show(wd, maps, P, G, self.cfg, cells, labels)
        call = ""
        if self.res.calls is not None:
            c = self.res.calls[(self.res.calls.plate_id == plate_id) & (self.res.calls.well_id == well_id)]
            if len(c):
                r = c.iloc[0]
                call = f"   {self.call_text(r['call'])}" + (f"  P={r.p_combo:.2f}" if pd.notna(r.get("p_combo")) else "")
        elif "preview_status" in self.res.wells:
            w = self.res.wells[(self.res.wells.plate_id == plate_id) & (self.res.wells.well_id == well_id)]
            if len(w):
                r = w.iloc[0]
                dom = r.preview_dominant if isinstance(r.preview_dominant, str) and r.preview_dominant else ""
                call = f"   preview: {r.preview_status}" + (f", mostly {self.call_text(dom)}" if dom else "")
        self.browser.plate.mark(plate_id, well_id)
        self._current_well = (plate_id, well_id)
        self.browser.pca.mark(plate_id, well_id)
        chans = [L for L in self.v.layers if L.name.startswith(self.display.PREFIX) and "channel" in L.metadata]
        if chans:
            self.v.layers.selection.active = chans[0]
        self.v.text_overlay.text = f"{plate_id}  {well_id}{call}"
        self.v.title = f"Cell reconstruction · {plate_id} {well_id}"
        self.panel.train.picked.setText("no particle picked")

    def _well_labels(self, P):
        if self.store is None or self.res is None or not len(P):
            return None
        L = self.store.frame(self.res.dataset)
        L = L[(L.plate_id == P.plate_id.iloc[0]) & (L.well_id == P.well_id.iloc[0])]
        if not len(L):
            return None
        W = self.res.wells
        W = W[(W.plate_id == P.plate_id.iloc[0]) & (W.well_id == P.well_id.iloc[0])]
        return match_labels(L, P, W)

    def random_well(self):
        d = self.browser.model.df
        if not len(d):
            return
        pool = d[(d.get("labels", 0) == 0) & (d.get("test", "") != "✓")] if "labels" in d else d
        pool = pool if len(pool) else d
        r = pool.sample(1).iloc[0]
        self.show_well(r.plate_id, r.well_id)

    def redraw_levels(self, *_):
        t = self.panel.train
        self.display.show_levels = {"whole": t.show_whole.isChecked(), "hole": t.show_hole.isChecked()}
        self.display.show_context = t.show_ctx.isChecked()
        self.display.show_noise = t.show_noise.isChecked()
        if getattr(self, "_current_well", None):
            picked = self.display.picked
            self.show_well(*self._current_well)
            if picked is not None:
                self.pick(picked)

    # -- labelling
    def set_label_mode(self, on: bool):
        self.label_mode = on
        self.say("label mode on: click a particle, then 1–5" if on else "label mode off")

    def _on_click(self, viewer, event):
        """napari drag generator: only records press and release. The pick (and its menu, which runs its own event
        loop) happens after the generator has finished, so mouse moves over an open menu cannot re-enter it."""
        if not self.label_mode or event.button != 1:
            return
        start = np.array(event.position)
        yield
        while event.type == "mouse_move":
            yield
        end = np.array(event.position)
        if np.linalg.norm(end - start) > 2 * (self.display.current or {}).get("scale", (1,))[0]:
            return                                   # a drag (pan), not a click
        from qtpy.QtCore import QTimer
        QTimer.singleShot(0, lambda: self.pick_at(end, QCursor.pos()))

    def pick_at(self, world_pos, screen_pos=None):
        hits = self.display.particles_at(world_pos)
        if not len(hits):
            self.pick(None)
            return
        if len(hits) == 1:
            self.pick(hits.iloc[0])
            return
        menu = QMenu(self.panel)
        for _, r in hits.iterrows():
            cls = r.get("pclass", "") if isinstance(r.get("pclass", ""), str) else ""
            text = f"{self.cfg.display_name(r.ch)} · {'lenient' if r.level == 'hole' else 'stringent'}" + \
                   (f" · {cls} {r.get('p_' + cls, np.nan):.2f}" if cls else "") + \
                   (f" · {r.in_um2:.0f} µm²" if r.pool else " · outside the hole (neighbour)")
            a = menu.addAction(text)
            a.setData(r.name)
        self._menu = menu
        menu.triggered.connect(lambda act: self.pick(hits.loc[act.data()]))
        menu.popup(screen_pos or QCursor.pos())

    # -- particle edits (cuts, accepted splits, merges): stored in the label library, applied by Detect
    def _inside_point(self, row):
        """A frame position inside the particle (the centroid can fall outside a concave particle)."""
        cur = self.display.current
        pts = np.argwhere(cur["maps"][(row.level, row.ch)] == row.lab)
        y, x = pts[len(pts) // 2]
        return float(y + cur["wd"].offset[0]), float(x + cur["wd"].offset[1])

    def _redetect(self, msg: str):
        from qtpy.QtWidgets import QApplication
        from qtpy.QtCore import Qt as _Qt
        from ..pipeline import Results, redetect_well
        pl, wl = self._current_well
        edits = self.store.edits(self.res.dataset).get((pl, wl))
        QApplication.setOverrideCursor(_Qt.WaitCursor)
        try:
            redetect_well(self.res.dir, pl, wl, edits, model=self.model)
            self.res = Results(self.res.dir)
        except Exception as e:
            self.say(f"re-detect failed: {e}")
        finally:
            QApplication.restoreOverrideCursor()
        self.refresh()
        self.update_pca()
        self.show_well(pl, wl)
        P = self.res.particles
        n = P[(P.plate_id == pl) & (P.well_id == wl) & P.pool]
        self.say(f"{msg}: {pl} {wl} re-detected ({len(n)} particles in the hole)" +
                 ("; calls of this well updated with the stored calibration" if self.res.calls is not None else ""))

    def _need_edit_target(self):
        if self.res is None or self.store is None or not getattr(self, "_current_well", None):
            self.say("open results and a well first")
            return False
        return True

    def accept_split(self):
        row = self.display.picked
        if not self._need_edit_target() or row is None:
            self.say("pick a particle with a proposed split (dashed orange line) first")
            return
        if getattr(row, "n_proposed", 2) < 2:
            self.say("Detect proposes no split for this particle (no real dip inside it); draw a cut instead")
            return
        y, x = self._inside_point(row)
        self.store.accept_split(self.res.dataset, row.plate_id, row.well_id, row.ch, row.level, y, x)
        self._redetect("split accepted")

    def watershed(self):
        """Split the picked particle on demand (at any dip between its peaks, else at a neck in its shape). The
        dividing lines are stored like a hand cut: every channel and both levels, undone by Clear edits."""
        row = self.display.picked
        if not self._need_edit_target() or row is None:
            self.say("pick the particle to watershed first")
            return
        from ..detect import smoothed
        from ..edits import watershed_pieces
        from scipy import ndimage as ndi
        from ..pipeline import well_hbg
        cur = self.display.current
        wd, L = cur["wd"], cur["maps"][(row.level, row.ch)]
        m = L == row.lab
        sl = tuple(slice(max(a.start - 1, 0), a.stop + 1) for a in
                   ndi.find_objects(m.astype(np.uint8))[0])
        w = self.res.wells
        hbg = well_hbg(w[(w.plate_id == row.plate_id) & (w.well_id == row.well_id)].iloc[0], self.res.channels)
        sm = smoothed(wd, self.cfg)[wd.names.index(row.ch)] - hbg[row.ch][0]
        n = int(watershed_pieces(m[sl], sm[sl], self.cfg.peak_h * max(hbg[row.ch][1], 1e-6),
                                 self.cfg.min_object_px).max())
        if n < 2:
            self.say("watershed finds no second peak or neck in this particle; draw a cut instead (X)")
            return
        y, x = self._inside_point(row)
        self.store.add_watershed(self.res.dataset, row.plate_id, row.well_id, row.ch, row.level, y, x)
        self._redetect(f"watershed: {n} pieces")

    def draw_cut(self, on: bool):
        if not self._need_edit_target():
            self.panel.train.b_cut.setChecked(False)
            return
        name = "✂ cut"
        if on:
            cur = self.display.current
            tr = (cur["delta"][0] * cur["scale"][0], cur["delta"][1] * cur["scale"][1])
            if name not in self.v.layers:
                self.v.add_shapes([], shape_type="line", name=name, edge_color="#ffffff", edge_width=1.0,
                                  scale=cur["scale"])
            L = self.v.layers[name]
            L.translate = tr                                   # same placement as the overlays: window pixels
            self.v.layers.selection.active = L
            L.mode = "add_line"
            self.say("draw cut: drag a line across the particle (several are fine), then Apply cuts")
        else:
            if name in self.v.layers:
                self.v.layers[name].mode = "pan_zoom"
            self._activate_channel()

    def _activate_channel(self):
        chans = [L for L in self.v.layers if L.name.startswith(self.display.PREFIX) and "channel" in L.metadata]
        if chans:
            self.v.layers.selection.active = chans[0]

    def apply_cuts(self):
        name = "✂ cut"
        if not self._need_edit_target() or name not in self.v.layers or not len(self.v.layers[name].data):
            self.say("draw a cut first (Draw cut, then drag across the particle)")
            return
        L = self.v.layers[name]
        oy, ox = self.display.current["wd"].offset
        pl, wl = self._current_well
        for line_ in L.data:
            pts = np.asarray(line_)[:, -2:] + np.array([oy, ox])
            self.store.add_cut(self.res.dataset, pl, wl, pts.tolist())
        n = len(L.data)
        self.v.layers.remove(L)
        self.panel.train.b_cut.setChecked(False)
        self._activate_channel()
        self._redetect(f"{n} cut(s) applied")

    def merge_mode(self, on: bool):
        if not on:
            self._merge_from = None
            return
        row = self.display.picked
        if not self._need_edit_target() or row is None:
            self.say("pick the first particle, then press Merge with…")
            self.panel.train.b_merge.setChecked(False)
            return
        self._merge_from = row
        if not self.panel.train.mode.isChecked():
            self.panel.train.mode.setChecked(True)
        self.say("now click the piece to join it with (same channel and threshold, touching or same object)")

    def _do_merge(self, a, b):
        self.panel.train.b_merge.setChecked(False)
        if a.ch != b.ch or a.level != b.level:
            self.say("merge needs two particles of the same channel and threshold")
            return
        if a.origin != b.origin:
            Lm = self.display.current["maps"][(a.level, a.ch)]
            from scipy import ndimage as ndi
            if not (ndi.binary_dilation(Lm == a.lab) & (Lm == b.lab)).any():
                self.say("these particles neither touch nor come from the same object: not merged "
                         "(pieces of one cell that do not touch are joined by the reconstruction anyway)")
                return
        p1, p2 = self._inside_point(a), self._inside_point(b)
        self.store.add_merge(self.res.dataset, a.plate_id, a.well_id, a.ch, a.level, p1, p2)
        self._redetect("merged")

    def clear_edits(self):
        if not self._need_edit_target():
            return
        pl, wl = self._current_well
        self.store.clear_edits(self.res.dataset, pl, wl)
        self._redetect("edits cleared")

    def pick(self, row):
        if getattr(self, "_merge_from", None) is not None and row is not None and row.oid != self._merge_from.oid:
            a, self._merge_from = self._merge_from, None
            self._do_merge(a, row)
            return
        self.display.select(row)
        if row is None:
            self.panel.train.picked.setText("no particle picked")
            return
        lab = self._well_labels(self.display.current["P"])
        hand = ""
        if lab is not None and len(lab):
            m = lab[lab.oid == row.oid]
            hand = m.label.iloc[0] if len(m) else ""
        ps = " ".join(chip(f"{c} {row.get('p_' + c, np.nan):.2f}", CLASS_COLORS[c]) for c in CLASSES
                      if f"p_{c}" in row)
        prop = int(row.get("n_proposed", 1) or 1) if hasattr(row, "get") else 1
        self.panel.train.b_accept.setEnabled(prop > 1)
        if not bool(row.pool):
            self.panel.train.picked.setText(
                f"<b>{self.cfg.display_name(row.ch)}</b> · {'lenient' if row.level == 'hole' else 'stringent'} · "
                f"outside the hole ({row.full_px * self.res.wells.px_um.iloc[0] ** 2:.0f} µm²): a neighbour by "
                "its position, not classified" + (" (cut off a particle in the hole: merge it back or clear the "
                                                  "edits to undo)" if row.get("n_parts", 1) > 1 else "") + ".")
            return
        self.panel.train.picked.setText(
            (f"<i>Detect proposes splitting this particle into {prop} pieces</i><br>" if prop > 1 else "") +
            f"<b>{self.cfg.display_name(row.ch)}</b> · {'lenient' if row.level == 'hole' else 'stringent'} · "
            f"{row.in_um2:.1f} µm² inside · share {row.share:.2f} · SNR {row.snr:.1f}<br>{ps}"
            + (f"<br>hand label: {chip(hand, CLASS_COLORS[hand])}" if hand else ""))

    def _label_target(self):
        row = self.display.picked
        if row is None or self.res is None or self.store is None:
            self.say("pick a particle first (label mode, click)")
            return None
        if not bool(row.pool):
            self.say("this particle does not reach into the hole: it counts as a neighbour by its position and "
                     "is not classified, so it takes no label (cut, watershed and merge work on it)")
            return None
        w = self.res.wells
        w = w[(w.plate_id == row.plate_id) & (w.well_id == row.well_id)].iloc[0]
        return row, float(row.iy + w.off_r), float(row.ix + w.off_c)

    def set_label(self, cls: str):
        t = self._label_target()
        if t is None:
            return
        row, y, x = t
        self.store.set(self.res.dataset, row.plate_id, row.well_id, row.ch, row.level, y, x, cls)
        self.say(f"{row.plate_id} {row.well_id} {row.oid}: {cls}")
        self._redraw_keep_pick(row)

    def remove_label(self):
        t = self._label_target()
        if t is None:
            return
        row, y, x = t
        self.store.remove(self.res.dataset, row.plate_id, row.well_id, row.ch, row.level, y, x)
        L = self.store.frame(self.res.dataset)
        # labels imported without a level ("any") are removed by position too
        m = L[(L.plate_id == row.plate_id) & (L.well_id == row.well_id) & (L.channel == row.ch)]
        for r in m.itertuples():
            if np.hypot(r.y - y, r.x - x) * self.res.wells.px_um.iloc[0] <= 2.0:
                self.store.remove(r.dataset, r.plate_id, r.well_id, r.channel, r.level, r.y, r.x)
        self._redraw_keep_pick(row)

    def _redraw_keep_pick(self, row):
        self.show_well(row.plate_id, row.well_id)
        self.pick(row)
        self.panel.train.counts.set(self.store.counts())
        pm = self.browser.pca
        if len(getattr(pm, "df", [])):                      # keep the map's hand labels current
            labs = self.all_labels()
            lab = {} if labs is None else {(r.plate_id, r.well_id, r.oid): r.label for r in labs.itertuples()}
            pm.df["hand"] = [lab.get(k, "") for k in zip(pm.df.plate_id, pm.df.well_id, pm.df.oid)]
            pm.draw()

    def import_labels(self, path: str):
        if self.res is None or self.store is None:
            self.say("open results first: labels are imported for their dataset")
            return
        names = {self.cfg.display_name(ch): ch for ch in self.res.channels}
        names.update({ch: ch for ch in self.res.channels})
        rep = self.store.import_legacy(path, self.res.dataset, names)
        self.say(f"imported: {rep}")
        self.refresh()

    def open_queue_item(self, r):
        if r is None:
            return
        self.show_well(r.plate_id, r.well_id)
        P = self.display.current["P"]
        m = P[P.oid == r.oid]
        if len(m):
            self.pick(m.iloc[0])
            if not self.panel.train.mode.isChecked():
                self.panel.train.mode.setChecked(True)

    # -- long work
    def _worker(self, fn, done):
        from napari.qt.threading import create_worker
        w = create_worker(fn)
        w.returned.connect(done)
        w.errored.connect(lambda e: self.say(f"error: {e}"))
        w.finished.connect(lambda: self._busy(False))
        self._busy(True)
        w.start()
        self._w = w

    def _busy(self, on):
        for b in (self.panel.run.b_detect, self.panel.run.b_classify, self.panel.run.b_all, self.panel.train.b_train):
            b.setEnabled(not on)

    def run_stage(self, stage: str):
        from ..pipeline import classify_run, detect_run, source_from_spec
        cfg = self.read_config()
        self.cfg = cfg
        if stage != "detect" and cfg.model and (self.model is None or Path(str(cfg.model)).name !=
                                                Path(str(self.model_file or "")).name):
            from ..learn import ParticleModel
            try:
                self.model = ParticleModel.load(cfg.model_path())
                self.model_file = cfg.model
            except Exception as e:
                self.say(f"model not loaded: {e}")
        out = self.panel.run.out.text().strip()
        if not out:
            self.say("choose a results folder")
            return
        wells = self.panel.run.wells.text().split() or None
        prog = lambda i, n, m="": self.bus.progress.emit(i, n, m)
        if stage in ("detect", "all"):
            root, tag = self.panel.run.root.text().strip(), self.panel.run.run.currentText()
            if not root or not tag:
                self.say("choose an output folder and a run")
                return
            spec = dict(kind="run", root=root, run_tag=tag)

            store = self.store

            def job():
                src = source_from_spec(spec, cfg)
                edits = store.edits(getattr(src, "dataset", "")) if store is not None else None
                detect_run(src, cfg, out, wells, progress=prog, source_spec=spec, edits=edits)
                if stage == "all":
                    if self.model is None and not (cfg.model_path() and cfg.model_path().exists()):
                        prog(0, 0, "detected; no model yet: label particles and train, then Classify")
                    else:
                        classify_run(out, cfg, self.model, progress=prog)
                return out
        else:
            def job():
                classify_run(out, cfg, self.model, progress=prog)
                return out
        self.say(f"{stage} → {out}")
        self._worker(job, lambda d: (self.say("done"), self.open_results(d)))

    def train(self):
        from ..learn import assemble, train
        if self.store is None:
            self.say("choose a label library first")
            return
        folders = [l.strip() for l in self.panel.train.folders.toPlainText().splitlines() if l.strip()]
        if not folders:
            self.say("no results folder to train on")
            return
        arms = [a for a, cb in self.panel.train.arms.items() if cb.isChecked()]
        rounds, tau = self.panel.train.rounds.value(), self.panel.train.tau.value()
        prog = lambda m: self.bus.progress.emit(0, 0, m)
        store = self.store

        def job():
            T, tg, rep = assemble(folders, store)
            prog(rep.to_string(index=False))
            if rep.attrs.get("feature_config_differs"):
                prog("WARNING: the results folders were detected with different feature settings")
            return train(T, tg, arms=arms, rounds=rounds, tau=tau, progress=prog,
                         feature_config=rep.attrs.get("feature_config"))

        def done(res):
            self.trained = res
            self.panel.train.show_result(res)
            self.say(f"trained; best by grouped CV: {res.model.spec['arm']}")

        self._worker(job, done)

    def save_model(self):
        import datetime
        from ..learn import ParticleModel
        res = self.trained
        if res is None:
            return
        arm = self.panel.train.arm_pick.currentText()
        model = res.model
        if arm != model.spec["arm"]:
            self.say(f"refitting arm '{arm}' on all training labels…")
            from ..learn import assemble, train
            folders = [l.strip() for l in self.panel.train.folders.toPlainText().splitlines() if l.strip()]
            T, tg, rep = assemble(folders, self.store)
            model = train(T, tg, arms=[arm], folds=2, final_arm=arm,
                          feature_config=rep.attrs.get("feature_config")).model
        name = f"{'_'.join(model.spec.get('datasets', ['model']))[:60]}_{arm}_{datetime.date.today():%Y%m%d}.joblib"
        path = MODELS / name
        model.save(path)
        self.model = ParticleModel.load(path)
        self.model_file = name
        self.cfg.model = name
        self.panel.run.set_models(name)
        self.panel.settings.show_model_match()
        self.say(f"model saved: {path}")
        if self.res is not None:
            self.panel.run.out.setText(str(self.res.dir))
            self.run_stage("classify")

    def open_folder(self):
        if self.res is None:
            return
        p = str(self.res.dir)
        if sys.platform.startswith("win"):
            os.startfile(p)  # noqa: S606
        else:
            import subprocess
            subprocess.Popen(["xdg-open", p])


def launch(results: str | None = None, viewer=None, run: bool = True):
    import napari
    v = viewer or napari.Viewer(title="Cell reconstruction")
    app = ReconstructionApp(v)
    v.window._ci_reconstruction = app            # keep a reference
    if results:
        app.open_results(results)
    if run:
        napari.run()
        return 0
    return app
