"""One well in the napari canvas.

Layers (rebuilt per well; visibility and contrast are remembered per layer name):
  BF                    brightfield planes (raw min-max), hidden by default; several planes -> slider
  <channel>             fluorescence, additive, the run's colour for that channel
  <channel> outlines    particles of that channel in the channel's colour: stringent threshold solid, lenient
                        (hole) threshold dashed. Shown and hidden together with the channel's image.
  hole                  hole outline
  cells                 reconstructed bodies (after Classify): cells in their call's colour, neighbours grey
  tags                  one tag per particle in the hole: the model's class (C cell, N neighbour, M merged,
                        D debris, · noise), or before Classify the provisional class (in, N?, D?, ·?). A hand
                        label shows as e.g. "[C]" in white.
  selection             the picked particle
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from skimage.measure import find_contours

from .theme import CLASS_COLORS, call_color

SHORT = {"cell": "C", "neighbour": "N", "merged": "M", "debris": "D", "noise": "·"}
DSHORT = {"inside": "in", "neighbour": "N?", "debris": "D?", "noise": "·?"}
DCOLORS = {"inside": "#ffd23f", "neighbour": "#9aa5b1", "debris": "#c77dff", "noise": "#5c6670"}


def _outline(mask: np.ndarray):
    """Largest contour of a mask as an (N, 2) polygon in pixel coordinates."""
    m = np.pad(mask, 1)
    cs = find_contours(m.astype(float), 0.5)
    if not cs:
        return None
    c = max(cs, key=len) - 1
    return c if len(c) >= 3 else None


def _dashes(poly: np.ndarray, dash: float = 2.5, gap: float = 2.0) -> list:
    """Split a closed polygon into dash segments (napari has no dashed edges)."""
    p = np.vstack([poly, poly[:1]])
    seg = np.hypot(*np.diff(p, axis=0).T)
    s = np.r_[0, np.cumsum(seg)]
    out, t, period = [], 0.0, dash + gap
    while t < s[-1]:
        a, b = t, min(t + dash, s[-1])
        ts = np.r_[a, s[(s > a) & (s < b)], b]
        pts = np.c_[np.interp(ts, s, p[:, 0]), np.interp(ts, s, p[:, 1])]
        if len(pts) >= 2:
            out.append(pts)
        t += period
    return out


def _colormap(hex_color: str, name: str):
    from napari.utils.colormaps import Colormap
    h = hex_color.lstrip("#")
    rgb = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    return Colormap([[0, 0, 0, 1], rgb + [1]], name=f"ci_{name}_{h}")


class WellDisplay:
    PREFIX = "◆ "      # our layers carry this prefix so user layers are never touched

    def __init__(self, viewer):
        self.v = viewer
        self.state = {}          # layer name -> dict(visible, contrast_limits)
        self.current = None      # dict(plate_id, well_id, wd, maps, P, scale)
        self.picked = None
        self.show_levels = {"whole": True, "hole": True}
        self.show_context = True

    def _name(self, s):
        return self.PREFIX + s

    def _remember(self):
        for L in list(self.v.layers):
            if L.name.startswith(self.PREFIX):
                st = dict(visible=L.visible)
                if hasattr(L, "contrast_limits"):
                    st["contrast_limits"] = tuple(L.contrast_limits)
                self.state[L.name] = st

    def _restore(self, L, default_visible=True):
        st = self.state.get(L.name)
        L.visible = st["visible"] if st else default_visible
        return st

    def clear(self):
        self._remember()
        for L in [L for L in self.v.layers if L.name.startswith(self.PREFIX)]:
            self.v.layers.remove(L)

    def show(self, wd, maps: dict, P: pd.DataFrame, G: np.ndarray | None, cfg, cells: pd.DataFrame | None,
             labels: pd.DataFrame | None):
        """P: this well's particles (with dclass, and pclass/p_* once classified). labels: matched hand labels."""
        self.clear()
        scale = (wd.px_um, wd.px_um)
        self.current = dict(plate_id=wd.plate_id, well_id=wd.well_id, wd=wd, maps=maps, P=P, scale=scale)
        self.picked = None
        if wd.bf:
            bf = np.stack([a for _, a in wd.bf])
            L = self.v.add_image(bf if len(bf) > 1 else bf[0], name=self._name("BF"), colormap="gray", scale=scale,
                                 contrast_limits=(float(bf.min()), float(bf.max()) or 1.0))
            self._restore(L, default_visible=False)
        img_layers = {}
        for k, ch in enumerate(wd.names):
            img = wd.chans[k]
            bg = wd.bg.get(ch, (float(np.percentile(img, 5)), 1))[0]
            hi = float(np.percentile(img, 99.7))
            L = self.v.add_image(img, name=self._name(cfg.display_name(ch)), colormap=_colormap(cfg.color(ch), ch),
                                 blending="additive", scale=scale, contrast_limits=(float(bg), max(hi, float(bg) + 1)))
            st = self._restore(L)
            if st and "contrast_limits" in st:
                try:
                    L.contrast_limits = st["contrast_limits"]
                except Exception:
                    pass
            L.metadata["channel"] = ch
            img_layers[ch] = L
        # outlines per channel: stringent solid, lenient dashed, channel colour
        for ch in wd.names:
            col = cfg.color(ch)
            shapes, types, widths = [], [], []
            d = P[P.ch == ch] if len(P) else P
            for r in d.itertuples():
                if not self.show_levels.get(r.level, True) or (not r.pool and not self.show_context):
                    continue
                L_ = maps.get((r.level, r.ch))
                if L_ is None:
                    continue
                pg = _outline(L_ == r.lab)
                if pg is None:
                    continue
                if r.level == "whole":
                    shapes.append(pg); types.append("polygon"); widths.append(0.9 if r.pool else 0.5)
                else:
                    for seg in _dashes(pg):
                        shapes.append(seg); types.append("path"); widths.append(0.9)
            if not shapes:
                continue
            L = self.v.add_shapes(shapes, shape_type=types, name=self._name(f"{cfg.display_name(ch)} outlines"),
                                  edge_color=col, face_color="transparent", edge_width=widths, scale=scale,
                                  opacity=0.95)
            img = img_layers[ch]
            L.visible = img.visible
            img.events.visible.connect(lambda e, L=L, img=img: setattr(L, "visible", img.visible))
        poly = _outline(wd.hole)
        if poly is not None:
            L = self.v.add_shapes([poly], shape_type="polygon", name=self._name("hole"), edge_color="#ffffff",
                                  face_color="transparent", edge_width=0.6, opacity=0.8, scale=scale)
            self._restore(L)
        if G is not None and G.any():
            from napari.utils.colormaps import DirectLabelColormap
            R = np.where(G > 0, G, np.where(G < 0, 1000 - G, 0)).astype(np.int32)
            colors = {None: "transparent", 0: "transparent"}
            calls = {} if cells is None or not len(cells) else dict(zip(cells.cell_id, cells["call"]))
            for k in np.unique(G[G > 0]):
                colors[int(k)] = call_color(calls.get(int(k), "uncertain"), cfg.color)
            for k in np.unique(G[G < 0]):
                colors[int(1000 - k)] = "#8a929c"
            L = self.v.add_labels(R, name=self._name("cells"), scale=scale, opacity=0.95)
            L.colormap = DirectLabelColormap(color_dict=colors)
            L.contour = 2
            self._restore(L)
        self._tags(P, labels, scale)
        self.v.add_shapes([], shape_type="polygon", name=self._name("selection"), edge_color="#ffffff",
                          face_color="transparent", edge_width=2.0, scale=scale)
        self.v.reset_view()

    def _tags(self, P, labels, scale):
        if not len(P):
            return
        pool = P[P.pool]
        # one tag per object: the lenient piece if there is one, else the stringent particle
        covered = set(pool.loc[pool.level == "hole", "parent"].dropna())
        pool = pool[(pool.level == "hole") | ~pool.oid.isin(covered)]
        if not len(pool):
            return
        hand = {} if labels is None or not len(labels) else dict(zip(labels.oid, labels.label))
        txt, col = [], []
        for r in pool.itertuples():
            h = hand.get(r.oid) or hand.get(getattr(r, "parent", None))
            pc = getattr(r, "pclass", None)
            if h:
                txt.append(f"[{SHORT[h]}]"); col.append("#ffffff")
            elif isinstance(pc, str) and pc:
                txt.append(SHORT[pc]); col.append(CLASS_COLORS[pc])
            else:
                dc = getattr(r, "dclass", None) or "inside"
                txt.append(DSHORT.get(dc, "")); col.append(DCOLORS.get(dc, "#d0d0d0"))
        pts = np.c_[pool.iy.to_numpy(float), pool.ix.to_numpy(float)]
        L = self.v.add_points(pts, name=self._name("tags"), scale=scale, size=0.1, face_color="transparent",
                              border_width=0, features=pd.DataFrame(dict(t=txt, c=col)),
                              text=dict(string="{t}", color={"feature": "c"}, size=11, anchor="center"))
        self._restore(L)

    # -- picking
    def particles_at(self, world_pos) -> pd.DataFrame:
        """Pool particles whose mask contains the clicked pixel (any channel, both thresholds)."""
        if self.current is None:
            return pd.DataFrame()
        s = self.current["scale"]
        y, x = int(round(world_pos[-2] / s[0])), int(round(world_pos[-1] / s[1]))
        maps, P = self.current["maps"], self.current["P"]
        H, W = self.current["wd"].hole.shape
        if not (0 <= y < H and 0 <= x < W) or not len(P):
            return pd.DataFrame()
        hits = []
        for (lev, ch), L in maps.items():
            v = int(L[y, x])
            if v:
                m = P[(P.level == lev) & (P.ch == ch) & (P.lab == v) & P.pool]
                if len(m):
                    hits.append(m.index[0])
        return P.loc[hits]

    def select(self, row: pd.Series | None):
        self.picked = row
        name = self._name("selection")
        if name not in self.v.layers:
            return
        L = self.v.layers[name]
        L.data = []
        if row is None:
            return
        pg = _outline(self.current["maps"][(row.level, row.ch)] == row.lab)
        if pg is not None:
            L.add_polygons([pg])
            L.edge_color = "#ffffff"
            L.edge_width = 2.0
