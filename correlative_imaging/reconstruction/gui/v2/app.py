"""Cell reconstruction v2: the canvas is napari; the tool is one dock on the right with the workflow strip and four tabs.

  Detect       (done before: ci gui runs it after a pipeline run, or the classic ci reconstruct)
  First pass   setup, run, the first pass's questions (only where the data cannot decide), what it found
  Label        your biological labels on random holes and surrounding cells from every channel pattern
  Learn        the label model learnt from your labels: how well it agrees with you, uncertain cells back to Label
  Report       the full report, with plots and example images

Everything is computed by correlative_imaging.reconstruction.unified (also runnable without the tool:
python -m correlative_imaging.reconstruction unified --results <Detect results>). The tool writes only the run's
output folder (default <Detect results>/unified): experiment.json, interventions/answers.csv, interventions/labels.csv.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from qtpy.QtCore import QObject, Qt, QUrl, Signal
from qtpy.QtGui import QColor, QDesktopServices
from qtpy.QtWidgets import (QComboBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QInputDialog,
                            QLabel, QLineEdit, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QStackedWidget,
                            QTableWidget, QTableWidgetItem, QTabWidget, QTextBrowser, QVBoxLayout, QWidget)

from . import canvas
from .asker import Asker
from .control import ControlView

STEPS = {"detect": "Detect", "first": "First pass", "label": "Label", "learn": "Learn", "report": "Report"}
ACTIONS = {"detect": "Run Detect", "first": "Run first pass", "label": "Open Label", "learn": "Learn from my labels",
           "report": "Show report"}
STATE_DOT = {"open": "#4a90d9", "done": "#4caf6e", "optional": "#8a929c", "not needed": "#8a929c"}


def local(p: str | Path) -> Path:
    """A path written on Windows or in WSL, usable here."""
    p = str(p)
    if Path(p).exists():
        return Path(p)
    m = re.match(r"^/mnt/([a-z])/(.*)$", p)
    if m and os.name == "nt":
        return Path(f"{m.group(1).upper()}:/{m.group(2)}")
    m = re.match(r"^([A-Za-z]):[\\/](.*)$", p)
    if m and os.name != "nt":
        return Path(f"/mnt/{m.group(1).lower()}/{m.group(2).replace(chr(92), '/')}")
    return Path(p)


class _Bus(QObject):
    log = Signal(str)


def _mtime(p: Path) -> float:
    return p.stat().st_mtime if p.exists() else 0.0


# ------------------------------------------------------------------------------------------------------- tabs
class FirstPassTab(QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.stack = QStackedWidget()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.addWidget(self.stack)
        # page 0: overview
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(0, 0, 0, 0)
        self.setup = QGroupBox("Setup (experiment.json)")
        self.setup.setCheckable(True)
        self.setup.setChecked(False)
        self.setup.toggled.connect(lambda on: self.setup_body.setVisible(on))
        sv = QVBoxLayout(self.setup)
        self.setup_body = QWidget()
        self.setup_body.setVisible(False)
        self.form = QFormLayout(self.setup_body)
        sv.addWidget(self.setup_body)
        v.addWidget(self.setup)
        row = QHBoxLayout()
        self.run_b = QPushButton("Run first pass")
        self.run_b.setProperty("accent", True)
        self.run_b.clicked.connect(lambda: app.run())
        row.addWidget(self.run_b)
        self.last = QLabel("")
        self.last.setProperty("role", "hint")
        row.addWidget(self.last, 1)
        v.addLayout(row)
        self.report_link = QLabel("<a href='report' style='color:#6cb6ff'>Open the report in the browser</a>")
        self.report_link.setTextInteractionFlags(Qt.TextBrowserInteraction)
        self.report_link.linkActivated.connect(lambda _: app.open_report())
        self.report_link.setVisible(False)
        v.addWidget(self.report_link)
        self.bar = QProgressBar()
        self.bar.setRange(0, 5)
        self.bar.setFormat("%v/5")
        v.addWidget(self.bar)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(90)
        v.addWidget(self.log)
        q = QLabel("Questions: the first pass asks only where this dataset cannot decide on its own")
        q.setProperty("role", "section")
        q.setWordWrap(True)
        v.addWidget(q)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["question", "why", "needed", "state"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setWordWrap(True)
        self.table.cellDoubleClicked.connect(lambda r, c: self.ask(r))
        v.addWidget(self.table, 2)
        ab = QPushButton("Answer the selected question")
        ab.clicked.connect(lambda: self.ask(self.table.currentRow()))
        v.addWidget(ab)
        f = QLabel("What it found")
        f.setProperty("role", "section")
        v.addWidget(f)
        self.found = QTextBrowser()
        v.addWidget(self.found, 2)
        self.stack.addWidget(page)
        # page 1: answering one question
        page2 = QWidget()
        v2 = QVBoxLayout(page2)
        v2.setContentsMargins(0, 0, 0, 0)
        back = QPushButton("◀ Back to the questions")
        back.clicked.connect(self.back)
        v2.addWidget(back)
        self.asker = Asker(app.views, app.open_item)
        v2.addWidget(self.asker, 1)
        self.stack.addWidget(page2)
        self.Q = []

    # setup
    def fill_setup(self):
        while self.form.rowCount():
            self.form.removeRow(0)
        e = self.app.experiment()
        if e is None:
            self.form.addRow(QLabel("written by the first run"))
            return
        self.role, self.cname = {}, {}
        for c, d in e["channels"].items():
            w = QWidget()
            h = QHBoxLayout(w)
            h.setContentsMargins(0, 0, 0, 0)
            n = QLineEdit(d.get("name", c))
            r = QComboBox()
            r.addItems(["colour", "marker"])
            r.setCurrentText(d.get("role", "colour"))
            r.setToolTip("colour: a reporter (bleed-through corrected, part of the pattern)\nmarker: a stain measured "
                         "on its own (e.g. an antibody)")
            h.addWidget(n, 1)
            h.addWidget(r)
            self.cname[c], self.role[c] = n, r
            self.form.addRow(c, w)
        self.excl = QLineEdit("; ".join(e.get("exclude_wells") or []))
        self.excl.setToolTip("imaging failures, never used: '*:col 6', '18T40668:rows C-O cols 1-8', '18T39383:B04'")
        self.empty = QLineEdit("; ".join(e.get("empty_wells") or []))
        self.empty.setToolTip("no cells seeded: the reference for 'no cell' (same syntax)")
        self.form.addRow("excluded wells", self.excl)
        self.form.addRow("empty wells", self.empty)
        sb = QPushButton("Save setup")
        sb.clicked.connect(self.save_setup)
        self.form.addRow("", sb)

    def save_setup(self):
        f = self.app.out / "experiment.json"
        e = json.loads(f.read_text(encoding="utf-8"))
        for c in e["channels"]:
            e["channels"][c]["name"] = self.cname[c].text().strip() or c
            e["channels"][c]["role"] = self.role[c].currentText()
        split = lambda s: [x.strip() for x in re.split(r"[;\n]", s) if x.strip()]
        e["exclude_wells"], e["empty_wells"] = split(self.excl.text()), split(self.empty.text())
        f.write_text(encoding="utf-8", data=json.dumps(e, indent=2))
        self.app.say("setup saved: run the first pass again to apply it")
        self.app.refresh()

    # questions
    def fill(self):
        self.fill_setup()
        S = self.app.summary()
        self.report_link.setVisible(bool(self.app.out and (self.app.out / "report.html").exists()))
        self.last.setText(f"last run: {S.get('minutes', '?')} min" if S else "not run yet: the first run reads every "
                          "hole image once (15-40 min); later runs take 2-6 min")
        try:
            from ...unified.questions import open_questions
            self.Q = open_questions(self.app.out) if S else []
        except Exception as ex:                                             # noqa: BLE001
            self.Q = []
            self.app.say(f"questions: {ex}")
        self.table.setRowCount(len(self.Q))
        for i, q in enumerate(self.Q):
            for j, txt in enumerate([q["title"], q["why"], f"{q['needed']} · answered {q['answered']}", q["state"]]):
                it = QTableWidgetItem(txt)
                it.setToolTip(txt)
                if j == 3:
                    it.setForeground(Qt.GlobalColor.white)
                    it.setBackground(QColor(STATE_DOT.get(q["state"], "#8a929c")))
                self.table.setItem(i, j, it)
        self.table.resizeRowsToContents()
        self.found.setHtml(self.app.found_html())

    def ask(self, i: int):
        if not (0 <= i < len(self.Q)):
            return
        q = self.Q[i]
        from ...unified.questions import Answers
        A = Answers(self.app.out)

        def on_answer(r, a, q=q):
            A.set(A.key(q["question"], r["plate_id"], r["well_id"], r.get("image_id", ""), r.get("obj", "")), a)
        self.asker.set_items(q["items"], q["choices"], on_answer,
                             f"<b>{q['title']}</b><br>{q['why']}<br><i>Needed: {q['needed']}. Answers are saved at "
                             f"once; 'Run first pass' applies them.</i>")
        self.stack.setCurrentIndex(1)
        self.asker.setFocus()

    def back(self):
        self.stack.setCurrentIndex(0)
        self.app.refresh()


class LabelTab(QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app
        v = QVBoxLayout(self)
        v.setContentsMargins(4, 4, 4, 4)
        self.head = QLabel("")
        self.head.setWordWrap(True)
        self.head.setTextFormat(Qt.RichText)
        v.addWidget(self.head)
        row = QHBoxLayout()
        nb = QPushButton("+ new label")
        nb.clicked.connect(self.new_label)
        row.addWidget(nb)
        mb = QPushButton("More suggestions")
        mb.setToolTip("add random holes and surrounding cells from every pattern that are not on the sheet yet")
        mb.clicked.connect(self.more)
        row.addWidget(mb)
        row.addStretch(1)
        self.counts = QLabel("")
        row.addWidget(self.counts)
        v.addLayout(row)
        self.asker = Asker(app.views, app.open_item)
        self.asker.answered.connect(self.update_counts)
        v.addWidget(self.asker, 1)
        self.labels = []

    def _file(self) -> Path:
        return self.app.out / "interventions" / "labels.csv"

    def _read(self) -> pd.DataFrame:
        return pd.read_csv(self._file(), dtype=str).fillna("") if self._file().exists() else pd.DataFrame()

    def fill(self):
        R = self._read()
        e = self.app.experiment()
        if not len(R) or e is None:
            self.head.setText("Run the first pass first: it suggests what to label.")
            self.asker.set_items(pd.DataFrame(columns=["kind", "plate_id", "well_id", "image_id", "obj"]), [], None)
            return
        self.head.setText(
            "<b>What is this cell?</b> In your own words (+ new label: astrocyte, interneuron, …), or <i>not a cell</i>. "
            "Holes and surrounding cells are drawn at random from every channel pattern of the first pass. Correct the "
            "positive channels only where you see it differently. About 10 per label; then <b>Learn</b>.")
        own = [x for x in R.your_label.unique() if x and x.lower() not in ("not a cell", "unsure")]
        self.labels = list(dict.fromkeys(self.labels + own))
        names = {c: d.get("name", c) for c, d in e["channels"].items()}
        by_name = {n: c for c, n in names.items()}
        items = R.assign(answer=R.your_label,
                         caption=[f"#{n} {'hole' if k == 'hole' else 'surrounding cell'} · {fp}"
                                  for n, k, fp in zip(R.number, R.kind, R.first_pass)])
        chan_state = {}
        for r in items.to_dict("records"):
            src = r["positive_channels"] if r["positive_channels"] else r["first_pass"]
            st = set()
            if src.strip().lower() != "none":
                for n, c in by_name.items():
                    if f"{n}+" in src or (r["positive_channels"] and n in src):
                        st.add(c)
            chan_state[Asker._key(r)] = st
        self._fp = {Asker._key(r): r["first_pass"] for r in items.to_dict("records")}
        self.asker.set_items(items, self.labels + ["not a cell", "unsure"], self.on_answer, "",
                             channels=list(names.items()), on_channels=self.on_channels, chan_state=chan_state)
        self.update_counts()

    def _write(self, r: dict, col: str, value: str):
        R = self._read()
        m = R.number == str(r["number"])
        R.loc[m, col] = value
        R.to_csv(self._file(), index=False)

    def on_answer(self, r, a):
        self._write(r, "your_label", a)

    def on_channels(self, r, st):
        e = self.app.experiment()
        names = [e["channels"][c].get("name", c) for c in e["channels"] if c in st]
        self._write(r, "positive_channels", ", ".join(names) if names else "none")

    def new_label(self):
        t, ok = QInputDialog.getText(self, "New label", "What kind of cell (your words):")
        t = t.strip()
        if ok and t and t not in self.labels:
            self.labels.append(t)
            items = self.asker.items
            self.asker.choices = self.labels + ["not a cell", "unsure"]
            self.asker._buttons()
            self.asker.items = items
            self.asker._paint()

    def more(self):
        from ...unified import labels as L
        out = self.app.out
        H = pd.read_csv(out / "hole_calls.csv", dtype={"plate_id": str, "well_id": str})
        H["valid"] = H.status_cell == "cell"
        O = pd.read_parquet(out / "field_cells.parquet")
        R = self._read()
        seed = int(R.number.astype(int).max()) if len(R) else 0
        S = L.suggest(H, O, n_holes=20, n_field=10, seed=seed + 1)
        have = set(zip(R.kind, R.plate_id, R.well_id, R.obj.astype(str))) if len(R) else set()
        S = S[[(k, p, w, str(o)) not in have for k, p, w, o in zip(S.kind, S.plate_id, S.well_id, S.obj)]]
        S["number"] = np.arange(seed + 1, seed + 1 + len(S))
        pd.concat([R, S.astype(str)]).to_csv(self._file(), index=False)
        self.app.say(f"{len(S)} more items to label")
        self.fill()

    def update_counts(self):
        R = self._read()
        if not len(R):
            self.counts.setText("")
            return
        c = R.your_label[R.your_label != ""].value_counts()
        self.counts.setText(" · ".join(f"{k} {v}" for k, v in c.items()) + f" · left {int((R.your_label == '').sum())}")
        self.app.update_strip()
        try:
            self.app.control.fill()                       # the white frames: what you labelled
        except Exception:                                                   # noqa: BLE001
            pass


class LearnTab(QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app
        v = QVBoxLayout(self)
        v.setContentsMargins(4, 4, 4, 4)
        row = QHBoxLayout()
        b = QPushButton("Learn from my labels")
        b.setProperty("accent", True)
        b.setToolTip("runs the first pass again with your answers and labels, then learns the labels (LDA)")
        b.clicked.connect(lambda: app.run())
        row.addWidget(b)
        u = QPushButton("Label the unsure ones")
        u.clicked.connect(self.unsure)
        row.addWidget(u)
        row.addStretch(1)
        v.addLayout(row)
        self.text = QTextBrowser()
        v.addWidget(self.text, 1)

    def fill(self):
        S = self.app.summary() or {}
        L = S.get("labels", {})
        e = html_escape
        if not L.get("used"):
            self.text.setHtml(f"<p>No label model yet: {e(str(L.get('why', 'run the first pass first')))}.</p>"
                              "<p>Label at least 3 cells of 2 labels in the Label tab (about 10 per label is better), "
                              "then press <b>Learn from my labels</b>.</p>")
            return
        cf = L.get("confusion", {})
        cols = sorted({c for r in cf.values() for c in r})
        t = (f"<p>Learnt from <b>{L['n']}</b> of your labels: " + ", ".join(f"{e(k)} {v}" for k, v in L["labels"].items())
             + f".</p><p>Each of your labels predicted by a model trained on all the others: <b>{100 * L['acc']:.0f} %"
             f"</b> right" + (f" (holes {100 * L['acc_hole']:.0f} %)" if L.get("acc_hole") == L.get("acc_hole") else "")
             + ".</p><table border='1' cellspacing='0' cellpadding='3'><tr><th>your label → predicted</th>"
             + "".join(f"<th>{e(c)}</th>" for c in cols) + "</tr>"
             + "".join(f"<tr><td>{e(i)}</td>" + "".join(f"<td>{r.get(c, 0)}</td>" for c in cols) + "</tr>"
                       for i, r in cf.items()) + "</table>")
        H = pd.read_csv(self.app.out / "hole_calls.csv")
        if "bio_label" in H:
            c = H.bio_label.fillna("").replace("", "—").value_counts()
            t += "<p>Hole cells per label: " + ", ".join(f"{e(str(k))} {v}" for k, v in c.items()) + "</p>"
            Hv = H[H.status_cell == "cell"] if "status_cell" in H else H
            ct = pd.crosstab(Hv.pattern, Hv.bio_label.fillna(""))
            t += ("<p>First-pass pattern × label:</p><table border='1' cellspacing='0' cellpadding='3'><tr><th></th>"
                  + "".join(f"<th>{e(str(c))}</th>" for c in ct.columns) + "</tr>"
                  + "".join(f"<tr><td>{e(str(i))}</td>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>"
                            for i, row in zip(ct.index, ct.to_numpy())) + "</table>")
        self.text.setHtml(t)

    def unsure(self):
        out = self.app.out
        H = pd.read_csv(out / "hole_calls.csv", dtype={"plate_id": str, "well_id": str})
        U = H[H.get("bio_label", pd.Series("", index=H.index)).fillna("") == "unsure"]
        if not len(U):
            self.app.say("no unsure hole cells")
            return
        f = out / "interventions" / "labels.csv"
        R = pd.read_csv(f, dtype=str).fillna("") if f.exists() else pd.DataFrame()
        have = set(zip(R.plate_id, R.well_id)) if len(R) else set()
        U = U[[(p, w) not in have for p, w in zip(U.plate_id, U.well_id)]].sample(min(20, len(U)), random_state=0)
        n0 = int(R.number.astype(int).max()) if len(R) else 0
        S = pd.DataFrame(dict(number=np.arange(n0 + 1, n0 + 1 + len(U)), kind="hole", plate_id=U.plate_id,
                              well_id=U.well_id, image_id=U.get("image_id", ""), obj="", first_pass=U.pattern,
                              your_label="", positive_channels="")).astype(str)
        pd.concat([R, S]).to_csv(f, index=False)
        self.app.say(f"{len(S)} unsure hole cells added to the Label tab")
        self.app.tabs.setCurrentWidget(self.app.label_tab)
        self.app.label_tab.fill()


class ReportTab(QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app
        v = QVBoxLayout(self)
        v.setContentsMargins(4, 4, 4, 4)
        row = QHBoxLayout()
        ob = QPushButton("Open in browser")
        ob.clicked.connect(lambda: self.app.open_report())
        row.addWidget(ob)
        fb = QPushButton("Open output folder")
        fb.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.app.out))))
        row.addWidget(fb)
        row.addStretch(1)
        v.addLayout(row)
        self.text = QTextBrowser()
        self.text.setOpenExternalLinks(True)
        self.text.setStyleSheet("QTextBrowser { background: #f6f7f9; color: #1c2230; }")   # the report's own colours
        v.addWidget(self.text, 1)
        self._shown = 0.0

    def fill(self):
        f = self.app.out / "report.html" if self.app.out else None
        if f is None or not f.exists():
            self.text.setHtml("<p>The report is written by every first-pass run.</p>")
            return
        if _mtime(f) != self._shown:
            self.text.setHtml(f.read_text(encoding="utf-8", errors="replace"))
            self._shown = _mtime(f)


def html_escape(s) -> str:
    import html
    return html.escape(str(s))


# -------------------------------------------------------------------------------------------------------- app
class AppV2:
    def __init__(self, viewer, results=None, out=None, workers: int | None = None):
        from ....gui_common.workflow_strip import WorkflowStrip
        from ..theme import stylesheet, theme
        self.v = viewer
        self.results, self.out, self._V, self._S = None, None, None, None
        self.workers = workers or min(max(2, (os.cpu_count() or 4) - 2), 16)   # reading from the share limits it anyway
        self.busy = False
        self.bus = _Bus()
        self.bus.log.connect(self._log)
        self.host = QWidget()
        lay = QVBoxLayout(self.host)
        lay.setContentsMargins(4, 4, 4, 4)
        top = QHBoxLayout()
        self.where = QLabel("no results open")
        self.where.setWordWrap(True)
        self.where.setProperty("role", "hint")
        top.addWidget(self.where, 1)
        ob = QPushButton("Open Detect results…")
        ob.clicked.connect(self.choose)
        top.addWidget(ob)
        lay.addLayout(top)
        t = theme(viewer)
        self.strip = WorkflowStrip(STEPS, ACTIONS, self._next, self._pick, self._act, t,
                                   loop_text="First pass → answer its questions → Label → Learn → label the unsure ones")
        lay.addWidget(self.strip)
        self.tabs = QTabWidget()
        self.first_tab, self.label_tab = FirstPassTab(self), LabelTab(self)
        self.learn_tab, self.report_tab = LearnTab(self), ReportTab(self)
        for w, n in [(self.first_tab, "First pass"), (self.label_tab, "Label"), (self.learn_tab, "Learn"),
                     (self.report_tab, "Report")]:
            self.tabs.addTab(w, n)
        self.tabs.currentChanged.connect(lambda i: self.tabs.widget(i).fill() if self.out else None)
        lay.addWidget(self.tabs, 1)
        self.host.setStyleSheet(stylesheet(t))
        self.host.setMinimumWidth(640)
        dock = viewer.window.add_dock_widget(self.host, area="right", name="Cell reconstruction v2")
        from ....gui_common.docks import add_control_view
        self.control = ControlView(self)
        add_control_view(viewer, self.control, right_dock=dock, right_width=640)
        viewer.text_overlay.visible = True
        viewer.text_overlay.position = "top_left"
        if results:
            self.open(results, out)
        else:
            self.update_strip()

    # ------------------------------------------------------------------------------------------- files
    def choose(self):
        d = QFileDialog.getExistingDirectory(self.host, "Detect results (reconstruction_<run tag>)")
        if d:
            self.open(d)

    def open(self, results, out=None):
        results = local(results)
        if not (results / "particles.parquet").exists():
            QMessageBox.warning(self.host, "Not Detect results",
                                f"{results} holds no Detect results (particles.parquet). Run Detect first: ci gui does "
                                "it after a pipeline run ('Then detect particles…'), or the classic ci reconstruct.")
            return
        self.results = results
        self.out = local(out) if out else results / "unified"
        self.out.mkdir(parents=True, exist_ok=True)
        self.where.setText(f"Detect results: {results}\noutput: {self.out}")
        self.refresh()

    def summary(self) -> dict | None:
        f = self.out / "summary.json" if self.out else None
        if f is None or not f.exists():
            return None
        if self._S is None or self._S[0] != _mtime(f):
            self._S = (_mtime(f), json.loads(f.read_text(encoding="utf-8")))
        return self._S[1]

    def experiment(self) -> dict | None:
        f = self.out / "experiment.json" if self.out else None
        return json.loads(f.read_text(encoding="utf-8")) if f and f.exists() else None

    def views(self):
        if self._V is None and self.summary() is not None:
            from ...unified.view import Views
            try:
                self._V = Views(self.out)
            except Exception as ex:                                         # noqa: BLE001
                self.say(f"cannot draw: {ex}")
        return self._V

    def found_html(self) -> str:
        """What the first pass found, in numbers."""
        S = self.summary()
        if not S:
            return "<p>Not run yet.</p>"
        e = html_escape
        nm = S.get("names", {})
        t = []
        try:
            H = pd.read_csv(self.out / "hole_calls.csv")
            V = H[H.status_cell == "cell"]
            t.append(f"<p><b>Holes</b>: {len(H)} measured, {len(V)} with a cell. Channel patterns: " + ", ".join(
                f"{e(k)} {v}" for k, v in V.pattern.value_counts().items()) + ".</p>")
            O = pd.read_parquet(self.out / "field_cells.parquet", columns=["pattern", "learn"])
            c = O[O.learn].pattern.value_counts()
            t.append(f"<p><b>Surrounding cells</b>: {int(c.sum())}. " + ", ".join(
                f"{e(k)} {v} ({100 * v / c.sum():.0f} %)" for k, v in c.items()) + ".</p>")
            fl = pd.Series([f for s in H["flags"].fillna("") for f in str(s).split("; ") if f]).value_counts()
            if len(fl):
                t.append("<p><b>Flags</b>: " + ", ".join(f"{e(k)} {v}" for k, v in fl.items()) + ".</p>")
        except Exception as ex:                                             # noqa: BLE001
            t.append(f"<p>{e(ex)}</p>")
        cal = S.get("cal", {})
        rows = "".join(f"<tr><td>{e(nm.get(c, c))}</td><td>{cal[c].get('measure', '')}</td>"
                       f"<td>{cal[c].get('cut', float('nan')):.1f}</td><td>{e(cal[c].get('how', ''))}</td>"
                       f"<td>{cal[c].get('n_pos', '')}</td></tr>" for c in S.get("colour", []) + S.get("marker", [])
                       if c in cal)
        t.append("<p><b>Per channel</b> (hole cells)</p><table border='1' cellspacing='0' cellpadding='3'><tr><th>"
                 "channel</th><th>measure</th><th>cut</th><th>how</th><th>positive</th></tr>" + rows + "</table>")
        sh = S.get("particle_shape", {})
        t.append("<p><b>Cell or not by shape</b>: " + (f"learnt from {sh.get('no_cell')} no-cell and {sh.get('cell')} "
                 f"cell particles, AUC {sh.get('auc', float('nan')):.2f}." if sh.get("used") else
                 "off (no 'no cell' examples yet: see the questions).") + "</p>")
        if S.get("applied"):
            t.append("<p><b>Applied</b>: " + e("; ".join(S["applied"])) + "</p>")
        return "".join(t)

    def refresh(self):
        self._V = None
        self._S = None
        w = self.tabs.currentWidget()
        if self.out:
            w.fill()
            if w is not self.first_tab:
                self.first_tab.fill()
            try:
                self.control.fill()
            except Exception as ex:                                         # noqa: BLE001
                self.say(f"plate: {ex}")
        self.update_strip()

    # ------------------------------------------------------------------------------------------- canvas
    def open_item(self, item: dict):
        V = self.views()
        if V is None:
            return
        try:
            self.v.text_overlay.text = canvas.show(self.v, V, item)
            for name, cb in self.control.ov.items():         # keep the overlays as switched in the Control view
                self.control._set_vis(name, cb.isChecked())
            self.control.mark(item["plate_id"], item["well_id"])
        except Exception as ex:                                             # noqa: BLE001
            self.say(f"cannot show {item.get('plate_id')} {item.get('well_id')}: {ex}")

    def open_report(self):
        f = self.out / "report.html" if self.out else None
        if f is not None and f.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(f)))
        else:
            self.say("no report yet: it is written by every first-pass run")

    def say(self, msg: str):
        self.first_tab.log.appendPlainText(msg)
        try:
            self.v.status = msg
        except Exception:                                                   # noqa: BLE001
            pass

    def _log(self, msg: str):
        self.first_tab.log.appendPlainText(msg)
        m = re.match(r"^(\d)/5", msg.strip())
        if m:
            self.first_tab.bar.setValue(int(m.group(1)))

    # ------------------------------------------------------------------------------------------- running
    def run(self):
        if self.busy or self.results is None:
            return
        from napari.qt.threading import create_worker
        from ...unified.run import run as run_unified
        self.busy = True
        self.first_tab.run_b.setEnabled(False)
        self.first_tab.bar.setValue(0)
        self.tabs.setCurrentWidget(self.first_tab)
        self.first_tab.stack.setCurrentIndex(0)
        self.say("running … (the first run reads every hole image once: 15-40 min; later runs 2-6 min; with new "
                 "'cell or not' answers the holes are measured again)")
        res, out, workers, log = str(self.results), str(self.out), self.workers, self.bus.log.emit

        def job():
            return run_unified(res, out, workers, log=log)
        w = create_worker(job)

        def done(*_):
            self.busy = False
            self.first_tab.run_b.setEnabled(True)
            self.say(f"done. Report: {self.out / 'report.html'} (link above, or the Report tab)")
            self.refresh()

        def failed(ex):
            self.busy = False
            self.first_tab.run_b.setEnabled(True)
            self.say(f"FAILED: {type(ex).__name__}: {ex}")
            self.update_strip()
        w.returned.connect(done)
        w.errored.connect(failed)
        w.start()
        self.update_strip()

    # ------------------------------------------------------------------------------------------- the strip
    def state(self) -> dict:
        S = self.summary() if self.out else None
        step = lambda **k: SimpleNamespace(**{**dict(purpose="", outputs=[], state="needed", detail="", reason="",
                                                    can_run=True), **k})
        st = {}
        has_detect = self.results is not None and (self.results / "particles.parquet").exists()
        st["detect"] = step(title="Detect", purpose="find the particles of every hole (no model needed)",
                            state="done" if has_detect else "needed", can_run=False,
                            reason="" if has_detect else "run Detect in ci gui (after a run) or in the classic ci "
                                                         "reconstruct", outputs=["particles.parquet"])
        inter = self.out / "interventions" if self.out else None
        if S is None:
            st["first"] = step(title="First pass", purpose="channel positivity of every hole and surrounding cell, "
                               "learnt from this dataset; questions where it is unsure",
                               state="needed" if has_detect else "blocked", outputs=["report.html", "hole_calls.csv"])
        else:
            newer = max(_mtime(self.out / "experiment.json"), _mtime(inter / "answers.csv")) > _mtime(self.out / "summary.json")
            open_q = [q for q in self.first_tab.Q if q["state"] == "open"]
            state = "running" if self.busy else ("stale" if newer else ("incomplete" if open_q else "done"))
            st["first"] = step(title="First pass", purpose="channel positivity, learnt from this dataset",
                               state=state, detail=f"{len(open_q)} questions open" if open_q else "",
                               reason="new answers or setup: run again to apply them" if newer else "",
                               outputs=["report.html", "hole_calls.csv"])
        R = pd.read_csv(inter / "labels.csv", dtype=str).fillna("") if inter and (inter / "labels.csv").exists() \
            else pd.DataFrame()
        lab = R.your_label[(R.your_label != "") & ~R.your_label.str.lower().isin(["not a cell", "unsure"])] \
            if len(R) else pd.Series(dtype=str)
        c = lab.value_counts()
        enough = int((c >= 3).sum()) >= 2
        st["label"] = step(title="Label", purpose="your biological labels on random cells of every pattern",
                           state="blocked" if S is None else ("done" if enough else "needed"),
                           detail=", ".join(f"{k} {v}" for k, v in c.items()), outputs=["interventions/labels.csv"],
                           reason="" if S is not None else "run the first pass first")
        L = (S or {}).get("labels", {})
        lab_newer = inter is not None and _mtime(inter / "labels.csv") > _mtime(self.out / "summary.json")
        st["learn"] = step(title="Learn", purpose="learn your labels from the channel signals (LDA)",
                           state="blocked" if not enough else ("stale" if (lab_newer and L.get("used")) else
                                                               ("done" if L.get("used") else "needed")),
                           detail=f"{100 * L['acc']:.0f} % agree with you" if L.get("used") else "",
                           reason="" if enough else "needs 2 labels with 3+ cells each", outputs=["hole_calls.csv"])
        st["report"] = step(title="Report", purpose="everything, with plots and example images",
                            state="done" if self.out and (self.out / "report.html").exists() else "blocked",
                            outputs=["report.html"])
        if self.busy:
            for k in st.values():
                k.can_run = False
        return st

    def update_strip(self):
        try:
            self.strip.set_state(self.state())
        except Exception as ex:                                             # noqa: BLE001
            self.say(f"strip: {ex}")

    def _next(self, S: dict):
        for k in STEPS:
            if S[k].state not in ("done", "not_used"):
                return k
        return None

    def _pick(self, key: str):
        tab = {"first": self.first_tab, "label": self.label_tab, "learn": self.learn_tab, "report": self.report_tab}
        if key in tab:
            self.tabs.setCurrentWidget(tab[key])

    def _act(self, key: str):
        if key in ("first", "learn"):
            self.run()
        else:
            self._pick(key)


def launch(results: str | None = None, out: str | None = None, viewer=None, run: bool = True):
    import napari
    v = viewer or napari.Viewer(title="Cell reconstruction v2")
    try:
        from ....gui_common import themes
        themes.apply(v, "ci_reconstruct")
    except Exception:                                                       # noqa: BLE001
        pass
    app = AppV2(v, results, out)
    v.window._ci_reconstruction_v2 = app
    if run:
        napari.run()
        return 0
    return app
