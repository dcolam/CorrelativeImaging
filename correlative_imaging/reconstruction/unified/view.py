"""What the tool draws: holes (with the grown cell and Detect's particles), single particles, surrounding cells and
whole wells, as RGB thumbnails or as raw per-channel images for the canvas. No Qt here.

Display range, linear, never clipped: every image (surrounding cell, whole image, hole) runs from its own minimum to its
own maximum. The channels are stretched together, each relative to a typical positive surrounding cell of that channel
(summary.json display_top.field_max), so the brightest channel just fills the range and the others keep their share:
a dim cell is visible and its colour is the colour it has relative to the dataset's cells.
The canvas shows the raw pixel values with these ranges as contrast limits (adjustable in the Control view).
"""
from __future__ import annotations

import json
import os
import re
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.segmentation import find_boundaries

from .data import local_path


def anypath(p) -> Path:
    """A path written on Windows or in WSL, usable on this system."""
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


def _rgb(hexc: str) -> np.ndarray:
    h = (hexc or "#ffffff").lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])


def limits(planes: dict, white: dict | None = None, together: bool = False) -> dict:
    """{channel: (low, high)} for raw planes {channel: image}: low = the image's minimum, high = white[channel] (else
    the image's maximum). together: one common stretch so the brightest channel just reaches its high."""
    out = {}
    for c, a in planes.items():
        lo, hi = display_range(a)
        w = (white or {}).get(c)
        if w is not None and np.isfinite(w) and w > lo:
            hi = float(w)
        out[c] = (lo, hi)
    if together and out:
        f = max((display_range(planes[c])[1] - lo) / (hi - lo) for c, (lo, hi) in out.items())
        f = max(f, 1e-6) if white else 1.0          # the brightest channel ends exactly at its maximum: no clipping
        out = {c: (lo, lo + f * (hi - lo)) for c, (lo, hi) in out.items()}
    return out


def display_range(a: np.ndarray, bg: float = 0.0, sd: float = 0.0) -> tuple:
    """(low, high), linear: the image's minimum to its maximum (no clipping). bg / sd are no longer used."""
    v = np.asarray(a, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return 0.0, 1.0
    lo, hi = float(v.min()), float(v.max())
    return lo, hi if hi > lo else lo + 1.0


class Views:
    def __init__(self, out: Path):
        self.out = Path(out)
        self.S = json.loads((self.out / "summary.json").read_text(encoding="utf-8"))
        self.results = anypath(self.S["results"])
        rj = json.loads((self.results / "run.json").read_text(encoding="utf-8"))
        from .experiment import load as load_experiment
        self.e = load_experiment(self.out, rj)            # also repairs channels saved without a colour
        self.path_map = (rj.get("config") or {}).get("path_map") or {}
        self.prefix_of = self.S.get("prefix_of") or {}
        top = self.S.get("display_top") or {}
        self.top_hole, self.top_field = top.get("hole", {}), top.get("field", {})
        self.white = top.get("field_max") or {}
        # holes on one scale for the whole run (default): (signal - background) / noise against a typical bright
        # positive hole cell of each channel (display_top.hole). Off: each hole stretched to its own brightest channel,
        # which makes a faint colour in a dim hole look bright (labels were set on that view)
        self.absolute = True
        self.chans = list(self.e["channels"])
        self.color = {c: v.get("color", "#ffffff") for c, v in self.e["channels"].items()}
        self.name = {c: v.get("name", c) for c, v in self.e["channels"].items()}
        self._fc = None
        self._cache = OrderedDict()
        self._lock = threading.RLock()            # the thumbnails are drawn in a background thread

    # ------------------------------------------------------------------------------------------------- data
    @property
    def field_cells(self) -> pd.DataFrame:
        if self._fc is None:
            with self._lock:
                fc = pd.read_parquet(self.out / "field_cells.parquet")
                fc["image_id"] = fc.image_id.astype(str)
                self._fc = fc
        return self._fc

    def _memo(self, k, fn):
        with self._lock:
            if k in self._cache:
                self._cache.move_to_end(k)
                return self._cache[k]
            v = fn()
            self._cache[k] = v
            while len(self._cache) > 48:
                self._cache.popitem(last=False)
            return v

    def hole(self, plate_id: str, well_id: str) -> dict | None:
        """The hole crop: img (C, H, W), names, bg, noise, hole, cell, neigh, px, maps {(level, ch): labels}."""
        def load():
            f = self.out / "wells" / f"{plate_id}_{well_id}.npz"
            if not f.exists():
                return None
            z = np.load(f)
            h = {k: z[k] for k in z.files}
            h["names"] = [str(n) for n in z["names"]]
            h["maps"] = {}
            pre = self.prefix_of.get(plate_id) or plate_id
            cf = self.results / "cache" / f"{pre}_{well_id}.npz"
            if cf.exists():
                cz = np.load(cf, allow_pickle=False)
                hole = cz["hole"].astype(bool)
                px = float(cz["px_um"])
                pad = max(int(round(8 / px)), 1)               # as measure.py cut the crop
                ys, xs = np.nonzero(hole)
                sl = (slice(max(ys.min() - pad, 0), ys.max() + pad), slice(max(xs.min() - pad, 0), xs.max() + pad))
                for k in cz.files:
                    if k.startswith("map|"):
                        _, lev, ch = k.split("|")
                        h["maps"][(lev, ch)] = cz[k][sl].astype(np.int32)
            return h
        return self._memo(("hole", plate_id, well_id), load)

    def raw_hole(self, h: dict) -> dict:
        """{channel: (raw image, (low, high))}."""
        P = {c: h["img"][k] for k, c in enumerate(h["names"])}
        L = limits(P, self.white, together=True)
        return {c: (P[c], L[c]) for c in P}

    def planes_hole(self, h: dict) -> dict:
        """{channel: (H, W) in [0, 1]}, linear within the display range: one scale for the run (absolute) or the hole's
        own stretch."""
        if self.absolute and self.top_hole and all((self.top_hole.get(c) or 0) > 0 for c in h["names"]):
            return {c: np.clip((h["img"][k] - h["bg"][k]) / max(float(h["noise"][k]), 1e-6) / float(self.top_hole[c]), 0, 1)
                    for k, c in enumerate(h["names"])}
        return {c: np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1) for c, (a, (lo, hi)) in self.raw_hole(h).items()}

    def whole(self, diag: str, well_raw: str) -> dict:
        import tifffile

        def load():
            raw = str(diag).rstrip("/\\") + ("/" if "/" in str(diag) else "\\") + f"{well_raw}_whole_channels.tif"
            f = anypath(raw)                    # written on Windows or in WSL
            if not f.exists():
                f = local_path(raw, self.path_map)
            with tifffile.TiffFile(f) as tf:
                return {c: tf.pages[i].asarray().astype(np.float32) for i, c in enumerate(self.chans)}
        return self._memo(("whole", diag, well_raw), load)

    def raw_field(self, P: dict, sl=None) -> dict:
        """{channel: (raw image or crop, (low, high))}: the crop's minimum to the dataset's brightest surrounding cell."""
        B = {c: (a[sl] if sl is not None else a) for c, a in P.items()}
        L = limits(B, self.white, together=True)
        return {c: (B[c], L[c]) for c in B}

    def planes_field(self, P: dict, sl=None) -> dict:
        """{channel: (H, W) in [0, 1]}, linear within the display range."""
        return {c: np.clip((b - lo) / max(hi - lo, 1e-6), 0, 1) for c, (b, (lo, hi)) in self.raw_field(P, sl).items()}

    # ---------------------------------------------------------------------------------------------- drawing
    def compose(self, planes: dict, chans=None) -> np.ndarray:
        chans = [c for c in (chans or self.chans) if c in planes]
        shape = next(iter(planes.values())).shape
        out = np.zeros(shape + (3,))
        for c in chans:
            out += planes[c][..., None] * _rgb(self.color.get(c))
        return np.clip(out, 0, 1)

    def hole_rgb(self, plate_id, well_id, particle: tuple | None = None, chans=None) -> np.ndarray | None:
        """Composite of the hole: hole yellow, grown cell cyan; with particle = (ch, level, lab) that particle white."""
        h = self.hole(plate_id, well_id)
        if h is None:
            return None
        out = self.compose(self.planes_hole(h), chans)
        out[find_boundaries(h["hole"] > 0)] = (1, 1, 0)
        if particle is None:
            out[find_boundaries(h["cell"] > 0, mode="inner")] = (0, 1, 1)
        else:
            ch, lev, lab = particle
            L = h["maps"].get((str(lev), ch))
            if L is not None and L.shape == h["hole"].shape:
                out[find_boundaries(L == int(lab), mode="inner")] = (1, 1, 1)
        return (out * 255).astype(np.uint8)

    def field_row(self, plate_id, image_id, obj) -> pd.Series | None:
        ch, lab = str(obj).split(":", 1)
        FC = self.field_cells
        r = FC[(FC.plate_id == plate_id) & (FC.image_id == str(image_id)) & (FC.channel == ch) &
               (FC.label.astype(str) == lab)]
        if not len(r):
            C = self.out / "cell_cores.parquet"
            if C.exists():
                Cc = pd.read_parquet(C)
                r = Cc[(Cc.plate_id == plate_id) & (Cc.image_id.astype(str) == str(image_id)) & (Cc.channel == ch) &
                       (Cc.label.astype(str) == lab)]
        return r.iloc[0] if len(r) else None

    def field_rgb(self, r: pd.Series, half_um: float = 12.0, chans=None) -> np.ndarray | None:
        P = self.whole(r.diag, r.well_raw)
        h = int(round(half_um / r.px_um))
        y, x = int(r.centroid_row), int(r.centroid_col)
        sl = (slice(max(y - h, 0), y + h), slice(max(x - h, 0), x + h))
        out = self.compose(self.planes_field(P, sl), chans)
        c = out.shape[0] // 2, out.shape[1] // 2
        out[max(c[0] - 1, 0):c[0] + 2, max(c[1] - 1, 0):c[1] + 2] = (1, 1, 1)
        return (out * 255).astype(np.uint8)

    def well_rgb(self, plate_id, well_id, size: int = 256) -> np.ndarray | None:
        FC = self.field_cells
        r = FC[(FC.plate_id == plate_id) & (FC.well_id == well_id)]
        if not len(r):
            return None
        r = r.iloc[0]
        P = self.whole(r.diag, r.well_raw)
        step = max(1, max(next(iter(P.values())).shape) // size)
        small = {c: a[::step, ::step] for c, a in P.items()}
        return (self.compose(self.planes_field(small)) * 255).astype(np.uint8)

    def _has_cells(self, item: dict) -> bool:
        FC = self.field_cells
        return bool(((FC.plate_id == item["plate_id"]) & (FC.well_id == item["well_id"])).any())

    def render(self, item: dict, chans=None) -> np.ndarray | None:
        """A thumbnail for a question or label item (kind: hole, particle, field, well)."""
        k = item.get("kind")
        try:
            if k == "hole":
                return self.hole_rgb(item["plate_id"], item["well_id"], chans=chans)
            if k == "particle":
                ch, lev, lab = str(item["obj"]).split("|")
                return self.hole_rgb(item["plate_id"], item["well_id"], (ch, lev, lab), chans=chans)
            if k == "field":
                r = self.field_row(item["plate_id"], item["image_id"], item["obj"])
                return None if r is None else self.field_rgb(r, chans=chans)
            if k == "well":                     # the whole image; the hole when the well has no surrounding cells
                return self.well_rgb(item["plate_id"], item["well_id"]) if self._has_cells(item) else \
                    self.hole_rgb(item["plate_id"], item["well_id"])
        except Exception:                                                    # noqa: BLE001
            return None
        return None
