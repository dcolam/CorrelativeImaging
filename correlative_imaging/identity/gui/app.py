"""The particle-identity tool inside napari: the viewer is the canvas, the tool lives in two docks.

  left   Wells: search + filter + table (plate, well, status, call, P, cells, labels). Click a well to show it.
  right  Run · Settings · Train · Results (panel.py)

Long work (detect, classify, train) runs in a napari thread worker; progress lands in the Run tab.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from qtpy.QtCore import QObject, Qt, Signal
from qtpy.QtWidgets import QComboBox, QHBoxLayout, QLabel, QLineEdit, QMenu, QVBoxLayout, QWidget
from qtpy.QtGui import QCursor

from ..config import MODELS, Config
from ..labels import CLASSES, LabelStore, match_labels
from .display import WellDisplay
from .panel import Panel
from .tables import FrameModel, FrameView
from .theme import CLASS_COLORS, CLASS_KEYS, call_color, chip, stylesheet, theme


class _Bus(QObject):
    progress = Signal(int, int, str)


class WellBrowser(QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app
        v = QVBoxLayout(self)
        v.setContentsMargins(4, 4, 4, 4)
        h = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("search plate / well / call…")
        self.search.textChanged.connect(self.filter)
        h.addWidget(self.search, 1)
        self.status = QComboBox()
        self.status.addItems(["all", "one cell", "several cells", "no cell", "uncertain", "labelled", "test wells"])
        self.status.currentTextChanged.connect(self.filter)
        h.addWidget(self.status)
        v.addLayout(h)
        self.model = FrameModel(colors={"call": lambda c: call_color(app.call_key(c), app.cfg.color)
                                        if isinstance(c, str) else None})
        self.view = FrameView(self.model)
        self.view.clicked.connect(self.clicked)
        self.view.activated.connect(self.clicked)
        v.addWidget(self.view, 1)
        self.count = QLabel("")
        self.count.setProperty("role", "hint")
        v.addWidget(self.count)
        self.full = pd.DataFrame()

    def set(self, df: pd.DataFrame):
        self.full = df
        self.filter()

    def filter(self, *_):
        d = self.full
        s = self.status.currentText()
        if len(d) and s != "all":
            if s == "uncertain":
                d = d[d.get("call", pd.Series(index=d.index, dtype=str)) == "uncertain"]
            elif s == "labelled":
                d = d[d.labels > 0]
            elif s == "test wells":
                d = d[d.test == "✓"]
            elif "status" in d:
                d = d[d.status == s]
        self.model.set(d)
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


class IdentityApp:
    def __init__(self, viewer):
        self.v = viewer
        self.cfg = Config()
        self.res = None
        self.store = None
        self.model = None
        self.trained = None
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
        dw = viewer.window.add_dock_widget(self.browser, area="left", name="Wells")
        viewer.window.add_dock_widget(self.panel, area="right", name="Particle identity")
        try:                                   # give the well list room under napari's layer docks
            dw.parentWidget().resizeDocks([dw], [460], Qt.Vertical)
        except Exception:
            pass
        viewer.mouse_drag_callbacks.append(self._on_click)
        for c in CLASSES:
            viewer.bind_key(CLASS_KEYS[c], lambda v, c=c: self.set_label(c), overwrite=True)
        viewer.bind_key("Delete", lambda v: self.remove_label(), overwrite=True)
        viewer.bind_key("l", lambda v: self.panel.train.mode.toggle(), overwrite=True)
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
        if not from_form:
            self.panel.settings.set_config(cfg, self.res.channels if self.res else None)
        if cfg.model and cfg.model_path() and cfg.model_path().exists():
            from ..learn import ParticleModel
            try:
                self.model = ParticleModel.load(cfg.model_path())
            except Exception as e:
                self.say(f"model not loaded: {e}")
        self.refresh()

    def set_store(self, path: str):
        if not path:
            return
        self.store = LabelStore(path)
        from qtpy.QtCore import QSettings
        QSettings("CorrelativeImaging", "identity").setValue("label_library", str(self.store.path))
        self.panel.train.store_path.setText(str(self.store.path))
        self.refresh()

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
        self.cfg = cfg
        self.panel.settings.set_config(cfg, self.res.channels)
        if self.res.run.get("model"):
            self.say(f"classified with model arm '{self.res.run['model'].get('arm')}'")
        if cfg.model_path() and cfg.model_path().exists():
            from ..learn import ParticleModel
            self.model = ParticleModel.load(cfg.model_path())
        if self.store is None:
            from qtpy.QtCore import QSettings
            last = QSettings("CorrelativeImaging", "identity").value("label_library", "")
            self.set_store(last if last and Path(last).exists() else str(Path(d).parent / "ci_labels.db"))
        folders = self.panel.train.folders
        if str(self.res.dir) not in folders.toPlainText():
            folders.setPlainText((folders.toPlainText().strip() + "\n" + str(self.res.dir)).strip())
        self.panel.run.out.setText(str(self.res.dir))
        self.say(f"opened {d}: {len(self.res.wells)} wells, dataset {self.res.dataset}")
        self.refresh()

    def refresh(self):
        res = self.res
        self.panel.run.update(res, self.store, self.model.spec if self.model else None)
        if self.store is not None:
            self.panel.train.counts.set(self.store.counts())
        self.panel.results.update(res)
        if res is None:
            return
        w = res.wells[["plate_id", "well_id"]].copy()
        if res.calls is not None:
            keep = [c for c in ("status", "call", "p_combo", "n_cells", "p_cell") if c in res.calls]
            w = w.merge(res.calls[["plate_id", "well_id"] + keep], how="left")
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
        self.browser.set(w.rename(columns={"p_combo": "P"}))

    def set_plate_map(self, m):
        self.plate_map = m
        self.refresh()

    # -- one well
    def show_well(self, plate_id, well_id):
        if self.res is None:
            return
        got = self.res.well(plate_id, well_id)
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
        chans = [L for L in self.v.layers if L.name.startswith(self.display.PREFIX) and "channel" in L.metadata]
        if chans:
            self.v.layers.selection.active = chans[0]
        self.v.text_overlay.text = f"{plate_id}  {well_id}{call}"
        self.v.title = f"Particle identity · {plate_id} {well_id}"
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

    # -- labelling
    def set_label_mode(self, on: bool):
        self.label_mode = on
        self.say("label mode on: click a particle, then 1–5" if on else "label mode off")

    def _on_click(self, viewer, event):
        if not self.label_mode or event.button != 1:
            return
        start = np.array(event.position)
        yield
        while event.type == "mouse_move":
            yield
        if np.linalg.norm(np.array(event.position) - start) > 2 * (self.display.current or {}).get("scale", (1,))[0]:
            return                                   # a drag (pan), not a click
        hits = self.display.particles_at(event.position)
        if not len(hits):
            self.pick(None)
            return
        if len(hits) == 1:
            self.pick(hits.iloc[0])
            return
        menu = QMenu()
        for _, r in hits.iterrows():
            cls = r.get("pclass", "")
            text = f"{self.cfg.display_name(r.ch)} · {'lenient' if r.level == 'hole' else 'stringent'}" + \
                   (f" · {cls} {r.get('p_' + cls, np.nan):.2f}" if isinstance(cls, str) and cls else "") + \
                   f" · {r.in_um2:.0f} µm²"
            a = menu.addAction(text)
            a.setData(r.name)
        act = menu.exec_(QCursor.pos())
        if act is not None:
            self.pick(hits.loc[act.data()])

    def pick(self, row):
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
        self.panel.train.picked.setText(
            f"<b>{self.cfg.display_name(row.ch)}</b> · {'lenient' if row.level == 'hole' else 'stringent'} · "
            f"{row.in_um2:.1f} µm² inside · share {row.share:.2f} · SNR {row.snr:.1f}<br>{ps}"
            + (f"<br>hand label: {chip(hand, CLASS_COLORS[hand])}" if hand else ""))

    def _label_target(self):
        row = self.display.picked
        if row is None or self.res is None or self.store is None:
            self.say("pick a particle first (label mode, click)")
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
        cfg = self.panel.settings.read()
        self.cfg = cfg
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

            def job():
                src = source_from_spec(spec, cfg)
                detect_run(src, cfg, out, wells, progress=prog, source_spec=spec)
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
            return train(T, tg, arms=arms, rounds=rounds, tau=tau, progress=prog)

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
            T, tg, _ = assemble(folders, self.store)
            model = train(T, tg, arms=[arm], folds=2, final_arm=arm).model
        name = f"{'_'.join(model.spec.get('datasets', ['model']))[:60]}_{arm}_{datetime.date.today():%Y%m%d}.joblib"
        path = MODELS / name
        model.save(path)
        self.model = ParticleModel.load(path)
        self.cfg.model = name
        self.panel.settings.set_config(self.cfg, self.res.channels if self.res else None)
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
    v = viewer or napari.Viewer(title="Particle identity")
    app = IdentityApp(v)
    v.window._ci_identity = app            # keep a reference
    if results:
        app.open_results(results)
    if run:
        napari.run()
        return 0
    return app
