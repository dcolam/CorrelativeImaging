"""Control view of ci gui (left, tabbed with napari's layer list and layer controls):

  plate      a quick view of the current plate (gui_common.plate): wells with BF + FL files, BF only, a yellow
             frame where ROI files exist. Click: select the well, exactly as in the Plate tab (both stay in sync);
             double-click: view the well (the Plate tab's "View all").
  channels   the shared channel histogram (gui_common.histogram) over the image layers on screen; hovering over
             the image shows every channel's pixel value. With
             "auto-contrast on well change" on (default), every channel layer that appears (a well, a sample, a
             preview) gets the shared rule at once: minimum -> 99.9th percentile, at least minimum + 100 (as in
             Cell reconstruction). Mask layers keep their own range.
  threshold  the threshold preview of the picked channel (viewer/threshold_preview.py).
"""
from __future__ import annotations

import pandas as pd
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import QCheckBox, QFrame, QLabel, QScrollArea, QVBoxLayout, QWidget

from correlative_imaging.gui_common.contrast import contrast_rule
from correlative_imaging.gui_common.histogram import ChannelHistogram, is_channel_layer, sample, set_limits
from correlative_imaging.gui_common.plate import PlateGrid

COLORS = {"complete": "#4CAF50", "bf_only": "#FF9800"}     # as the Plate tab
ROI_FRAME = "2px solid #FFEB3B"


def _section(text):
    l = QLabel(text)
    l.setStyleSheet("font-weight: 600; font-size: 11pt; padding-top: 6px;")
    return l


def _hint(text):
    l = QLabel(text)
    l.setWordWrap(True)
    l.setStyleSheet("font-size: 9pt; color: palette(mid);")
    return l


class CIControlView(QWidget):
    def __init__(self, viewer, plate_tab, main=None):
        super().__init__()
        self.v, self.pt, self.main = viewer, plate_tab, main
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
        self.plate = PlateGrid(
            {"files": lambda r: COLORS["complete" if r.get("complete") else "bf_only"]},
            legend_fn=lambda mode, d: (f'<span style="color:{COLORS["complete"]}">■</span> BF + FL &nbsp;'
                                       f'<span style="color:{COLORS["bf_only"]}">■</span> BF only &nbsp;'
                                       f'<span style="color:#FFEB3B">□</span> ROI files'),
            frame_fn=lambda r: ROI_FRAME if r.get("roi") else None,
            tip_fn=self._tip)
        self.plate.well_clicked.connect(self._clicked)
        self.plate.well_double_clicked.connect(self._double)
        self.plate.plate_changed.connect(self._plate_picked)
        v.addWidget(self.plate)
        v.addWidget(_hint("click: select the well (as in the Plate tab) · double-click: view it"))

        v.addWidget(_section("Channels"))
        self.auto_cb = QCheckBox("auto-contrast on well change")
        self.auto_cb.setToolTip("every channel that is loaded: black point = its minimum, white point = 99.9th "
                                "percentile but at least minimum + 100")
        self.auto_cb.setChecked(True)
        v.addWidget(self.auto_cb)
        from correlative_imaging.diagnostics import CHANNEL_COLOR_CHOICES, CHANNEL_COLOR_RGB
        choices = {c: "#" + "".join(f"{int(v * 255):02x}" for v in CHANNEL_COLOR_RGB[c]) for c in CHANNEL_COLOR_CHOICES}
        # right-click: the Channels tab's colour choices (the layer's colormap; the Channels tab follows it)
        # names without the group ("sample/", "FL/"); brightfield keeps its full name
        short = lambda k, L: (L.name.split("/", 1)[1] if L.name.split("/", 1)[0] in ("sample", "FL")
                              and "/" in L.name else L.name)
        self.hist = ChannelHistogram(viewer, auto_rule=contrast_rule, color_choices=choices, name_fn=short)
        v.addWidget(self.hist)
        self.threshold = None
        if main is not None:
            from correlative_imaging.viewer.threshold_preview import ThresholdPreview
            v.addWidget(_section("Threshold preview"))
            self.threshold = ThresholdPreview(viewer, self.hist, main)
            v.addWidget(self.threshold)
        v.addStretch(1)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.refresh)
        viewer.layers.events.inserted.connect(self._layer_added)
        grid = plate_tab._grid
        grid.changed.connect(lambda: self._timer.start(0))
        grid.well_clicked.connect(self._from_plate_tab)
        self.refresh()

    # -- channels
    def _layer_added(self, e):
        L = e.value
        if self.auto_cb.isChecked() and is_channel_layer(L):
            QTimer.singleShot(0, lambda L=L: self._auto(L))     # once the layer is set up

    def _auto(self, L):
        if L not in self.v.layers:
            return
        try:
            a = sample(L)
            if a.size:
                set_limits(L, *contrast_rule(a))
        except Exception:                                    # unreadable data: napari's own range stays
            pass

    # -- plate
    @staticmethod
    def _tip(plate, well, r):
        t = [f"{plate} {well}", "BF + FL" if r.get("complete") else "BF only"]
        if r.get("roi"):
            t.append(f"{r['roi']} ROI file(s)")
        return "\n".join(t)

    def refresh(self):
        """Re-read the plates of the Plate tab (after a scan, a plate switch or new ROIs)."""
        rows, labels = [], {}
        for key, e in (getattr(self.pt, "_plates", None) or {}).items():
            labels[key] = e.display_name
            for w in e.wells.values():
                rows.append(dict(plate_id=key, well_id=w.well_id, complete=bool(w.is_complete),
                                 roi=len(w.roi_paths or [])))
        df = pd.DataFrame(rows, columns=["plate_id", "well_id", "complete", "roi"])
        self.plate.blockSignals(True)
        self.plate.set(df, labels=labels)
        cur = getattr(self.pt, "_current_plate_key", None)
        if cur is not None:
            self.plate.show_plate(cur)
        self.plate.blockSignals(False)
        sel = getattr(self.pt._grid, "_selected", None)
        if cur is not None and sel:
            self.plate.mark(cur, sel)

    def _plate_picked(self, key):
        if key != getattr(self.pt, "_current_plate_key", None):
            self.pt._select_plate(key)                   # the Plate tab follows (and refreshes this view)

    def _clicked(self, key, well):
        if key != getattr(self.pt, "_current_plate_key", None):
            self.pt._select_plate(key)
        self.pt._grid._on_click(well)                    # the Plate tab's own path: details, ROI auto-load, …

    def _double(self, key, well):
        self._clicked(key, well)
        self.pt._on_view_all()

    def _from_plate_tab(self, well):
        key = getattr(self.pt, "_current_plate_key", None)
        if key is not None:
            self.plate.mark(key, well)
