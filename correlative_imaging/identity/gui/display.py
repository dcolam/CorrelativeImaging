"""One well in the napari canvas: channels, BF planes, hole, particles (by class), reconstructed cells, and the
particle picker used for labelling.

Layers (rebuilt per well; visibility and contrast are remembered per layer name):
  <channel display name>   fluorescence, additive, run colour
  BF                       brightfield planes (raw min-max), hidden by default; several planes -> slider
  hole                     hole outline
  particles (lenient)      outlines of the particles in the hole, coloured by predicted class; hand labels as tags
  particles (stringent)    the same for the stringent threshold, plus the cells around the hole (hidden by default)
  cells                    reconstructed bodies: cells coloured by their call, neighbours grey (contours)
  selection                the picked particle
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from skimage.measure import find_contours

from .theme import CLASS_COLORS, call_color

SHORT = {"cell": "C", "neighbour": "N", "merged": "M", "debris": "D", "noise": "·"}


def _outline(mask: np.ndarray):
    """Largest contour of a mask as an (N, 2) polygon in pixel coordinates."""
    m = np.pad(mask, 1)
    cs = find_contours(m.astype(float), 0.5)
    if not cs:
        return None
    c = max(cs, key=len) - 1
    return c if len(c) >= 3 else None


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
        self.current = None      # dict(plate_id, well_id, wd, maps, P)
        self.picked = None       # particle row (Series)

    # -- helpers
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

    # -- main
    def show(self, wd, maps: dict, P: pd.DataFrame, G: np.ndarray | None, cfg, cells: pd.DataFrame | None,
             labels: pd.DataFrame | None):
        """P: this well's particles (with pclass/p_* if classified). labels: matched hand labels (oid, label)."""
        self.clear()
        scale = (wd.px_um, wd.px_um)
        self.current = dict(plate_id=wd.plate_id, well_id=wd.well_id, wd=wd, maps=maps, P=P, scale=scale)
        self.picked = None
        # brightfield first (below fluorescence)
        if wd.bf:
            bf = np.stack([a for _, a in wd.bf])
            L = self.v.add_image(bf if len(bf) > 1 else bf[0], name=self._name("BF"), colormap="gray", scale=scale,
                                 contrast_limits=(float(bf.min()), float(bf.max()) or 1.0))
            self._restore(L, default_visible=False)
            if len(bf) > 1:
                L.metadata["planes"] = [l for l, _ in wd.bf]
        for k, ch in enumerate(wd.names):
            img = wd.chans[k]
            bg = wd.bg.get(ch, (float(np.percentile(img, 5)), 1))[0]
            hi = float(np.percentile(img, 99.7))
            name = self._name(cfg.display_name(ch))
            L = self.v.add_image(img, name=name, colormap=_colormap(cfg.color(ch), ch), blending="additive",
                                 scale=scale, contrast_limits=(float(bg), max(hi, float(bg) + 1)))
            st = self._restore(L)
            if st and "contrast_limits" in st:
                try:
                    L.contrast_limits = st["contrast_limits"]
                except Exception:
                    pass
            L.metadata["channel"] = ch
        # hole
        poly = _outline(wd.hole)
        if poly is not None:
            L = self.v.add_shapes([poly], shape_type="polygon", name=self._name("hole"), edge_color="#ffffff",
                                  face_color="transparent", edge_width=0.6, opacity=0.8, scale=scale)
            self._restore(L)
        # particles
        lab = {} if labels is None or not len(labels) else dict(zip(labels.oid, labels.label))
        for level, title, vis in (("hole", "particles (lenient)", True), ("whole", "particles (stringent)", False)):
            d = P[P.level == level] if len(P) else P
            polys, feats = [], []
            for r in d.itertuples():
                L_ = maps.get((r.level, r.ch))
                if L_ is None:
                    continue
                pg = _outline(L_ == r.lab)
                if pg is None:
                    continue
                cls = getattr(r, "pclass", "") or ""
                hand = lab.get(r.oid, "")
                polys.append(pg)
                feats.append(dict(oid=r.oid, channel=cfg.display_name(r.ch), pool=bool(r.pool),
                                  pclass=cls, p_cell=float(getattr(r, "p_cell", np.nan)),
                                  label=hand, tag=SHORT.get(hand, "") if hand else "",
                                  edge=CLASS_COLORS.get(cls, "#d0d0d0") if r.pool else "#6b7280"))
            if not polys:
                continue
            f = pd.DataFrame(feats)
            L = self.v.add_shapes(polys, shape_type="polygon", name=self._name(title), features=f,
                                  edge_color=list(f.edge), face_color="transparent",
                                  edge_width=[1.4 if t else 0.7 for t in f.tag], scale=scale,
                                  text=dict(string="{tag}", color="#ffffff", size=10, anchor="upper_left"))
            self._restore(L, default_visible=vis)
        # reconstruction
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
        self.v.add_shapes([], shape_type="polygon", name=self._name("selection"), edge_color="#ffffff",
                          face_color="transparent", edge_width=2.0, scale=scale)
        self.v.reset_view()

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
