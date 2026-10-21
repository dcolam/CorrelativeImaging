"""Threshold preview (ci gui, Control view): what a threshold does to the picked channel of the image on screen.

Two modes:
  quick (default)    the raw channel thresholded directly with the method picked in the dropdown ("Channels tab" = the
                     channel's own method there), with the channel's Z-projection and minimum object size. Fast.
  preprocess first   exactly the steps the pipeline runs before its threshold, on that channel alone: binning and
                     Z-projection (Preprocessing tab), the channel's preprocessing in its order, then AutoThreshold.
                     Slower (e.g. background subtraction), but the threshold the run will use.
Shown:

  objects    the thresholded objects as outlines in the channel's colour (layer "threshold preview: <ch>")
  image      when preprocessing changes the values, the preprocessed image the threshold was computed on (layer
             "<ch> preprocessed"; the raw layer is hidden meanwhile and shown again when the preview is off)
  histogram  the threshold as a dashed line on the histogram of that image ("otsu 523")
  numbers    method, threshold, share of pixels above it, number of objects

It runs again by itself when the picked channel, the image on screen or a Channels / Preprocessing setting changes.
Only whole-frame thresholds: a selection's own threshold ("threshold within this selection") needs that selection's
ROI and is not previewed yet.
"""
from __future__ import annotations

import numpy as np
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import (QAbstractButton, QCheckBox, QComboBox, QDoubleSpinBox, QLabel, QSpinBox, QVBoxLayout,
                            QWidget)

PREFIX = "threshold preview: "
PRE_SUFFIX = " preprocessed"


def _compute(img, idx: int, steps: list[dict]):
    """Run the steps on channel idx alone. Returns (threshold, method, plane thresholded, mask, n objects)."""
    from scipy.ndimage import label as ndi_label

    from correlative_imaging.pipeline.base import Pipeline, PipelineContext, Step
    from correlative_imaging.pipeline.segment import _project, mask_key
    one = []
    for s in steps:
        s = dict(s)
        if s.get("channel") == idx:
            s["channel"] = 0                                 # the channel alone, as channel 0
        one.append(s)
    pl = Pipeline("threshold preview")
    for s in one:
        pl.add(Step.from_dict(s))
    data = np.asarray(img.data[idx:idx + 1], dtype=np.float32)
    names = list(img.channel_names)
    ctx = PipelineContext(channel_names=[names[idx]], pixel_size_um=float(img.pixel_size_um or 1.0),
                          z_step_um=float(getattr(img, "z_step_um", None) or 1.0))
    current, results = pl.run(data, ctx)
    thr = next((r for r in reversed(results) if "threshold_value" in r.info), None)
    t = float(thr.info["threshold_value"]) if thr is not None else float("nan")
    seg = one[-1]
    plane = _project(np.asarray(current[0]), seg.get("z_projection", "max"))
    mask = ctx.masks.get(mask_key(0))
    lab, n = ndi_label(mask) if mask is not None else (np.zeros(plane.shape, np.int32), 0)
    return t, seg.get("method", ""), plane, lab.astype(np.int32), int(n)


class ThresholdPreview(QWidget):
    def __init__(self, viewer, hist, main):
        super().__init__()
        self.v, self.hist, self.main = viewer, hist, main
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        self.on = QCheckBox("show the threshold of the picked channel")
        self.on.setToolTip("the pipeline's threshold for the channel picked above, with its Channels-tab settings, "
                           "on the image on screen: objects as outlines, the threshold as a line in the histogram")
        self.on.toggled.connect(self._toggled)
        v.addWidget(self.on)
        from qtpy.QtWidgets import QHBoxLayout
        from correlative_imaging.pipeline.segment import _THRESHOLD_METHODS
        row = QHBoxLayout()
        row.addWidget(QLabel("method"))
        self.method = QComboBox()
        self.method.addItem("Channels tab", "")
        for m in _THRESHOLD_METHODS:
            self.method.addItem(m, m)
        self.method.setToolTip("the threshold method; \"Channels tab\" = the channel's own method there")
        self.method.currentIndexChanged.connect(lambda *_: self.schedule())
        row.addWidget(self.method, 1)
        self.pre = QCheckBox("preprocess first")
        self.pre.setToolTip("off: the raw channel, thresholded directly (fast). On: binning, Z-projection and the "
                            "channel's preprocessing first, exactly as the pipeline (slower)")
        self.pre.toggled.connect(lambda *_: self.schedule())
        row.addWidget(self.pre)
        v.addLayout(row)
        self.info = QLabel("")
        self.info.setWordWrap(True)
        self.info.setStyleSheet("font-size: 9pt;")
        v.addWidget(self.info)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(350)
        self._timer.timeout.connect(self.run)
        self._gen = 0
        self._target = None          # (raw layer key, channel index)
        self._hidden_raw = None      # raw layer hidden while its preprocessed image is shown
        self._watched = set()
        self._picking = False
        hist.picked_changed.connect(self._picked)
        viewer.layers.events.inserted.connect(self._layers_changed)

    # -- triggers
    def schedule(self):
        if self.on.isChecked():
            self._timer.start()

    def _picked(self, key):
        if self.on.isChecked() and not self._picking:
            self.schedule()

    def _layers_changed(self, e):
        name = str(getattr(e.value, "name", ""))
        try:
            self.on.isChecked()
        except RuntimeError:                                   # this view is gone: stop listening
            self.v.layers.events.inserted.disconnect(self._layers_changed)
            return
        if self.on.isChecked() and not name.startswith(PREFIX) and not name.endswith(PRE_SUFFIX):
            self.schedule()                                   # another image on screen

    def _watch_settings(self):
        """Any change on the Channels / Preprocessing tabs re-runs the preview."""
        tab = self.main._channels_tab
        for page in (tab.preprocess_page, tab.channels_page):
            for w in page.findChildren(QWidget):
                if id(w) in self._watched:
                    continue
                sig = None
                if isinstance(w, QComboBox):
                    sig = w.currentIndexChanged
                elif isinstance(w, (QSpinBox, QDoubleSpinBox)):
                    sig = w.valueChanged
                elif isinstance(w, QAbstractButton) and w.isCheckable():
                    sig = w.toggled
                if sig is not None:
                    sig.connect(lambda *_: self.schedule())
                    self._watched.add(id(w))

    def _toggled(self, on):
        if on:
            self._watch_settings()
            self.run()
        else:
            self._gen += 1
            self.clear()
            self.info.setText("")

    # -- what to threshold
    def _source(self):
        """(raw layer, its key, image data, channel index) for the picked channel, or a reason string."""
        img = getattr(self.main, "_shown_image", None)
        key = self.hist.cur
        if key is not None and str(key).endswith(PRE_SUFFIX) and self._target is not None:
            key = self._target[0]                             # the preprocessed image is picked: same channel
        L = self.hist._layer(key)
        if img is None or L is None:
            return "load an image (a sample or a well) first"
        group, _, ch = str(L.name).partition("/")
        if group.startswith("BF"):
            return "pick a fluorescence channel in the histogram"
        names = list(img.channel_names)
        if ch not in names:
            return f"'{L.name}' is not a channel of the image on screen"
        return L, key, img, names.index(ch)

    def _steps(self, idx):
        tab = self.main._channels_tab
        panels = tab.get_panels()
        if idx >= len(panels):
            return None, False
        p = panels[idx]
        seg = dict(p.get_segment_steps()[0])                  # AutoThreshold (a watershed split does not move it)
        self.method.setItemText(0, f"Channels tab ({seg.get('method', '')})")
        if self.method.currentData():
            seg["method"] = self.method.currentData()
        if not self.pre.isChecked():                          # quick: the raw channel, thresholded directly
            return [seg], False
        pre = [s for s in (tab.get_binning_step(), tab.get_zprojection_step()) if s]
        own = p.get_preprocess_steps()
        return pre + own + [seg], bool(own or any(s.get("type") == "Binning" for s in pre))

    # -- run
    def run(self):
        if not self.on.isChecked():
            return
        src = self._source()
        if isinstance(src, str):
            self.info.setText(src)
            return
        L, key, img, idx = src
        steps, changes_values = self._steps(idx)
        if steps is None:
            self.info.setText("the Channels tab has no settings for this channel yet")
            return
        self._gen += 1
        gen = self._gen
        self.info.setText(f"computing the threshold of {L.name.partition('/')[2]} …")
        from napari.qt.threading import create_worker
        w = create_worker(_compute, img, idx, steps)
        w.returned.connect(lambda r, gen=gen: self._show(gen, r, L, key, idx, changes_values))
        w.errored.connect(lambda e, gen=gen: gen == self._gen and self.info.setText(f"preview failed: {e}"))
        w.start()
        self._w = w

    def _show(self, gen, r, L, key, idx, changes_values):
        if gen != self._gen or not self.on.isChecked() or L not in self.v.layers:
            return                                            # a newer run, or switched off
        t, method, plane, lab, n = r
        ch = L.name.partition("/")[2]
        color = self.hist.color_fn(key, L)
        sy, sx = (np.asarray(L.data.shape[-2:], float) / np.asarray(lab.shape[-2:], float))
        scale = (float(L.scale[-2]) * sy, float(L.scale[-1]) * sx)
        tr = tuple(float(x) for x in L.translate[-2:])
        self._drop_others(ch)
        target_key = key
        if changes_values:                                    # the threshold lives on the preprocessed values
            P = self._layer(ch + PRE_SUFFIX, "image")
            if P is None:
                P = self.v.add_image(plane, name=ch + PRE_SUFFIX, colormap=L.colormap, blending="additive",
                                     scale=scale, translate=tr)
            else:
                P.data, P.scale, P.translate = plane, scale, tr
            if self._hidden_raw is None and L.visible:
                L.visible = False
                self._hidden_raw = L
            target_key = P.name
        else:
            P = self._layer(ch + PRE_SUFFIX, "image")
            if P is not None:
                self.v.layers.remove(P)
            self._restore_raw()
        from napari.utils.colormaps import DirectLabelColormap
        M = self._layer(PREFIX + ch, "labels")
        if M is None:
            M = self.v.add_labels(lab, name=PREFIX + ch, scale=scale, translate=tr, opacity=1.0,
                                  metadata={"mask": True})
        else:
            M.data, M.scale, M.translate = lab, scale, tr
        M.contour = 1
        M.colormap = DirectLabelColormap(color_dict={None: color, 0: "transparent"})
        self._target = (key, idx)
        self.hist.rebuild()
        if self.hist.cur != target_key:                       # adjust the image the threshold was computed on
            self._picking = True
            try:
                self.hist.pick(target_key)
            finally:
                self._picking = False
        self.hist.set_markers(target_key, [(t, f"{method} {t:.4g}")])
        above = float((plane > t).mean() * 100) if np.isfinite(t) else float("nan")
        self.info.setText(f"<b>{ch}</b>: {method} threshold <b>{t:.4g}</b> · {above:.1f} % of pixels above · "
                          f"<b>{n}</b> objects (min. size applied) · " +
                          ("on the preprocessed image" if changes_values else
                           "preprocessed first (it does not change the values)" if self.pre.isChecked() else
                           "raw channel, no preprocessing"))

    # -- layers
    def _layer(self, name, kind):
        from napari.layers import Image, Labels
        cls = Image if kind == "image" else Labels
        return next((l for l in self.v.layers if l.name == name and isinstance(l, cls)), None)

    def _restore_raw(self):
        if self._hidden_raw is not None:
            if self._hidden_raw in self.v.layers:
                self._hidden_raw.visible = True
            self._hidden_raw = None

    def _is_preview(self, l) -> bool:
        return l.name.startswith(PREFIX) or l.name.endswith(PRE_SUFFIX)

    def _drop_others(self, ch: str):
        """Preview layers of another channel go (and that channel's raw image comes back)."""
        for l in list(self.v.layers):
            if self._is_preview(l) and ch not in (l.name[len(PREFIX):], l.name[:-len(PRE_SUFFIX)]):
                self.v.layers.remove(l)
        if self._hidden_raw is not None and self._hidden_raw.name.partition("/")[2] != ch:
            self._restore_raw()
        for k in list(self.hist.markers):
            self.hist.set_markers(k, [])

    def clear(self):
        """Remove the preview's layers and lines, and show the raw channel again."""
        for l in list(self.v.layers):
            if self._is_preview(l):
                self.v.layers.remove(l)
        self._restore_raw()
        for k in list(self.hist.markers):
            self.hist.set_markers(k, [])
