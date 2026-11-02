"""Control view of v2 (left, tabbed with napari's layer list and layer controls), as in ci gui / ci reconstruct v1.

  plate          one button per well (gui_common.plate.PlateGrid), coloured by
                   cell colour       the hole cell's colour as seen on screen (channels mixed, unified/hue.py)
                   channel pattern   the first pass's positive channels, mixed in their colours (dim: darker)
                   your label        the label model's call (your own label where you gave one)
                   checks            ok / flagged / no cell / excluded / empty well / no hole
                 white frame: you answered a question or labelled something in that well. Click: the hole on the
                 canvas; double-click: the whole image.
  channels       the shared channel histogram (gui_common/histogram.py) on the canvas layers: name = adjust, dot =
                 show / hide, right-click = colour (saved in experiment.json, used everywhere)
  overlays       hole, hole cell, asked particle
  cell colours   the hue histogram of the last first pass (surrounding vs hole cells; the report has more)
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from qtpy.QtCore import Qt
from qtpy.QtGui import QPixmap
from qtpy.QtWidgets import QCheckBox, QFrame, QHBoxLayout, QLabel, QScrollArea, QVBoxLayout, QWidget

from ....gui_common.histogram import ChannelHistogram
from ....gui_common.plate import PlateGrid
from .canvas import PREFIX

NO_CELL, EMPTY, EXCLUDED, NO_HOLE, OK, FLAGGED = "#3a3f47", "#8a929c", "#20232a", "#b5473a", "#4caf6e", "#e0a030"
LABEL_COLOURS = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948", "#b07aa1", "#ff9da7", "#9c755f"]
OVERLAYS = [("hole", "hole (yellow)"), ("hole cell", "hole cell (cyan)"), ("particle", "asked particle (white)")]


def _section(text):
    l = QLabel(text)
    l.setProperty("role", "section")
    return l


def _hex(rgb) -> str:
    return "#" + "".join(f"{int(round(float(v) * 255)):02x}" for v in np.clip(rgb, 0, 1))


def _rgb(hexc: str) -> np.ndarray:
    h = (hexc or "#ffffff").lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])


def auto_rule(a: np.ndarray) -> tuple:
    """As the first contrast of every item (unified/view.py display_range): minimum to maximum, no clipping."""
    from ...unified.view import display_range
    return display_range(a)


class ControlView(QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setFrameShape(QFrame.NoFrame)
        sc.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        body = QWidget()
        v = QVBoxLayout(body)
        v.setContentsMargins(4, 4, 4, 4)
        sc.setWidget(body)
        outer.addWidget(sc)

        v.addWidget(_section("Plate"))
        self.plate = PlateGrid({"cell colour": self._c_mix, "channel pattern": self._c_pattern,
                                "your label": self._c_label, "checks": self._c_check},
                               legend_fn=self._legend, frame_fn=self._frame, tip_fn=self._tip)
        self.plate.well_clicked.connect(lambda p, w: app.open_item({"kind": "hole", "plate_id": p, "well_id": w}))
        self.plate.well_double_clicked.connect(
            lambda p, w: app.open_item({"kind": "well", "plate_id": p, "well_id": w}))
        v.addWidget(self.plate)
        hint = QLabel("click a well: its hole; double-click: the whole image. White frame: you answered or labelled "
                      "something there.")
        hint.setProperty("role", "hint")
        hint.setWordWrap(True)
        v.addWidget(hint)

        v.addWidget(_section("Channels"))
        self.hist = ChannelHistogram(
            app.v, layers_fn=self._layers, name_fn=lambda c, L: self._V().name.get(c, c) if self._V() else c,
            color_fn=lambda c, L: self._V().color.get(c, "#ffffff") if self._V() else "#ffffff",
            auto_rule=auto_rule, set_color=self._set_color)
        v.addWidget(self.hist)

        v.addWidget(_section("Overlays"))
        row = QHBoxLayout()
        self.ov = {}
        for name, text in OVERLAYS:
            cb = QCheckBox(text)
            cb.setChecked(True)
            cb.toggled.connect(lambda on, name=name: self._set_vis(name, on))
            self.ov[name] = cb
            row.addWidget(cb)
        v.addLayout(row)

        v.addWidget(_section("Cell colours"))
        self.hue = QLabel("written by the first pass")
        self.hue.setProperty("role", "hint")
        self.hue.setAlignment(Qt.AlignCenter)
        v.addWidget(self.hue)
        v.addStretch(1)
        self.names, self.colors, self.D = {}, {}, pd.DataFrame()

    # ------------------------------------------------------------------------------------------------ data
    def _V(self):
        return self.app.views()

    def fill(self):
        """Re-read the wells after a run, an answer or a label."""
        out = self.app.out
        e = self.app.experiment()
        if out is None or e is None or not (out / "checks.csv").exists():
            self.plate.set(pd.DataFrame())
            return
        self.names = {c: d.get("name", c) for c, d in e["channels"].items()}
        self.colors = {c: d.get("color", "#ffffff") for c, d in e["channels"].items()}
        W = pd.read_csv(out / "checks.csv", dtype={"plate_id": str, "well_id": str}, usecols=lambda c: c in (
            "plate_id", "well_id", "status", "call"))
        W = W.drop_duplicates(["plate_id", "well_id"])
        hc = out / "hole_calls.csv"
        if hc.exists():
            H = pd.read_csv(hc, dtype={"plate_id": str, "well_id": str})
            keep = [c for c in ("plate_id", "well_id", "pattern", "status_cell", "flags", "bio_label", "mix", "cls")
                    if c in H]
            W = W.merge(H[keep].drop_duplicates(["plate_id", "well_id"]), on=["plate_id", "well_id"], how="left")
        touched = set()
        iv = out / "interventions"
        for f in ("answers.csv", "labels.csv"):
            if (iv / f).exists():
                A = pd.read_csv(iv / f, dtype=str).fillna("")
                if f == "labels.csv" and "your_label" in A:
                    mine = A[(A.your_label != "") & (A.kind == "hole")]
                    W = W.merge(mine[["plate_id", "well_id", "your_label"]].drop_duplicates(["plate_id", "well_id"]),
                                on=["plate_id", "well_id"], how="left")
                    A = A[(A.your_label != "") | (A.get("positive_channels", "") != "")]
                touched |= set(zip(A.plate_id, A.well_id))
        W["touched"] = [(p, w) in touched for p, w in zip(W.plate_id, W.well_id)]
        for c in ("pattern", "status_cell", "flags", "bio_label", "mix", "your_label"):
            if c not in W:
                W[c] = ""
        W = W.fillna("")
        labs = [l for l in pd.concat([W.your_label, W.bio_label]).value_counts().index if l and l != "unsure"]
        self.label_colour = {l: LABEL_COLOURS[i % len(LABEL_COLOURS)] for i, l in enumerate(labs)}
        self.D = W
        self.plate.set(W, stage=f"first pass of {pd.Timestamp.fromtimestamp((out / 'summary.json').stat().st_mtime):%d.%m. %H:%M}"
                       if (out / "summary.json").exists() else "")
        png = out / "cell_colours.png"
        if png.exists():
            pm = QPixmap(str(png))
            pm = pm.scaledToWidth(min(320, pm.width()), Qt.SmoothTransformation)
            self.hue.setPixmap(pm)
            self.hue.setMinimumHeight(pm.height())
        else:
            self.hue.setMinimumHeight(0)
            self.hue.setText("written by the next first pass")

    def mark(self, plate_id, well_id):
        try:
            self.plate.mark(plate_id, well_id)
        except Exception:                                                  # noqa: BLE001
            pass

    # ------------------------------------------------------------------------------------------- colouring
    def _status(self, r) -> str | None:
        st = str(r.get("status", ""))
        if st.startswith("excluded"):
            return EXCLUDED
        if st.startswith("empty well"):
            return EMPTY
        if st.startswith("no hole"):
            return NO_HOLE
        if r.get("status_cell", "") not in ("cell",):
            return NO_CELL
        return None

    def _c_check(self, r) -> str:
        s = self._status(r)
        return s or (FLAGGED if r.get("flags") else OK)

    def _c_mix(self, r) -> str:
        return self._status(r) or (r.get("mix") or NO_CELL)

    def _pattern_colour(self, pattern: str) -> str:
        if not pattern or pattern.strip().lower() == "none":
            return NO_CELL
        rgb, dim = np.zeros(3), False
        for c, n in self.names.items():
            if f"{n}+" in pattern:
                rgb += _rgb(self.colors[c])
            elif f"{n}?" in pattern:
                rgb += _rgb(self.colors[c])
                dim = True
        if not rgb.any():
            return NO_CELL
        rgb = rgb / max(rgb.max(), 1)
        return _hex(rgb * (0.5 if dim else 1))

    def _c_pattern(self, r) -> str:
        return self._status(r) or self._pattern_colour(str(r.get("pattern", "")))

    def _c_label(self, r) -> str:
        lab = r.get("your_label") or r.get("bio_label")
        return self._status(r) or self.label_colour.get(lab, NO_CELL)

    def _frame(self, r):
        return "2px solid #ffffff" if r.get("touched") else None

    def _tip(self, plate, well, r) -> str:
        t = [f"{plate} {well}", str(r.get("status", ""))]
        if r.get("status_cell"):
            t.append(f"{r.get('pattern', '')} ({r.get('status_cell')})")
        if r.get("bio_label"):
            from ...unified.vocab import shown
            t.append(f"label model: {shown(r['bio_label'])}")
        if r.get("your_label"):
            t.append(f"your label: {r['your_label']}")
        if r.get("flags"):
            t.append(f"flags: {r['flags']}")
        return "\n".join(x for x in t if x)

    def _legend(self, mode, d) -> str:
        sw = lambda col: f"<span style='background:{col}'>&nbsp;&nbsp;&nbsp;</span>"
        if mode == "checks":
            k = d.apply(self._c_check, axis=1) if len(d) else pd.Series(dtype=str)
            names = {OK: "ok", FLAGGED: "flagged", NO_CELL: "no cell", EXCLUDED: "excluded", EMPTY: "empty well",
                     NO_HOLE: "no hole"}
            return " ".join(f"{sw(c)} {names[c]} {n}" for c, n in k.value_counts().items() if c in names)
        if mode == "channel pattern":
            c = d.pattern.where(d.status_cell == "cell", "").value_counts()
            return " ".join(f"{sw(self._pattern_colour(p))} {p} {n}" for p, n in c.items() if p)
        if mode == "your label":
            lab = d.your_label.where(d.your_label != "", d.bio_label)
            c = lab[d.status_cell == "cell"].value_counts()
            return (" ".join(f"{sw(self.label_colour.get(l, NO_CELL))} {l} {n}" for l, n in c.items() if l)
                    or "no labels yet (Label tab, then Learn)")
        return ("the hole cell's colour as seen on screen; grey: no cell / too dim; "
                f"{sw(EXCLUDED)} excluded {sw(EMPTY)} empty well {sw(NO_HOLE)} no hole")

    # ---------------------------------------------------------------------------------------------- layers
    def _layers(self):
        V = self._V()
        if V is None:
            return []
        by = {L.name: L for L in self.app.v.layers}
        return [(c, by[PREFIX + V.name[c]]) for c in V.chans if PREFIX + V.name.get(c, c) in by]

    def _set_vis(self, name, on):
        for L in self.app.v.layers:
            if L.name == PREFIX + name:
                L.visible = on

    def _set_color(self, c, L, hexc):
        """A new colour for a channel: the layer, the thumbnails and experiment.json (the report on the next run)."""
        from napari.utils.colormaps import Colormap
        if not (isinstance(hexc, str) and hexc.startswith("#")):
            return
        L.colormap = Colormap([[0, 0, 0, 1], list(_rgb(hexc)) + [1]], name=f"v2_{c}_{hexc.lstrip('#')}")
        f = self.app.out / "experiment.json"
        e = json.loads(f.read_text(encoding="utf-8"))
        e["channels"][c]["color"] = hexc
        f.write_text(encoding="utf-8", data=json.dumps(e, indent=2))
        V = self._V()
        if V is not None:
            V.color[c] = hexc
            V.e["channels"][c]["color"] = hexc
        self.colors[c] = hexc
        self.plate.draw()
        self.app.say(f"{self.names.get(c, c)}: colour {hexc} saved (the report follows on the next first pass)")
