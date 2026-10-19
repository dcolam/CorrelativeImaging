"""One well in the napari canvas.

Layers (created once, then their data is swapped per well; visibility and contrast stay as you set them):
  BF                    brightfield planes (raw min-max), hidden by default; several planes -> slider
  <channel>             fluorescence, additive, the run's colour for that channel
  <channel> outlines    particle edges of that channel in the channel's colour (a pixel image, fast to swap):
                        stringent threshold solid, lenient (hole) threshold dashed. Shown and hidden together with
                        the channel's image.
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


def _colormap(hex_color: str, name: str):
    from napari.utils.colormaps import Colormap
    h = hex_color.lstrip("#")
    rgb = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    return Colormap([[0, 0, 0, 1], rgb + [1]], name=f"ci_{name}_{h}")


class WellDisplay:
    """Keeps one set of layers and swaps their data per well (creating and removing napari layers is what made
    switching wells slow). The set is rebuilt only when the channels, the BF planes or the display settings change."""
    PREFIX = "◆ "      # our layers carry this prefix so user layers are never touched

    def __init__(self, viewer):
        self.v = viewer
        self.L = {}              # role -> layer
        self.sig = None          # layout the current layers were built for
        self.current = None      # dict(plate_id, well_id, wd, maps, P, scale)
        self.picked = None
        self.show_levels = {"whole": True, "hole": True}
        self.show_context = True
        self.show_noise = False

    def _name(self, s):
        return self.PREFIX + s

    def clear(self):
        for L in [L for L in self.v.layers if L.name.startswith(self.PREFIX)]:
            self.v.layers.remove(L)
        self.L, self.sig = {}, None

    # -- layer set
    def _build(self, wd, cfg, scale):
        """Create the layers (bottom to top) with this well's shapes; visibility of rebuilt layers is kept."""
        vis = {L.name: L.visible for L in self.v.layers if L.name.startswith(self.PREFIX)}
        self.clear()
        H, W = wd.hole.shape
        z = np.zeros((H, W), np.uint8)

        def keep(L, default=True):
            L.visible = vis.get(L.name, default)
            return L
        if wd.bf:
            bf = np.stack([a for _, a in wd.bf])
            self.L["bf"] = keep(self.v.add_image(bf if len(bf) > 1 else bf[0], name=self._name("BF"), colormap="gray",
                                                 scale=scale), False)
        for k, ch in enumerate(wd.names):
            img = wd.chans[k]
            bg = wd.bg.get(ch, (float(np.percentile(img, 5)), 1))[0]
            hi = float(np.percentile(img, 99.7))
            L = keep(self.v.add_image(img, name=self._name(cfg.display_name(ch)),
                                      colormap=_colormap(cfg.color(ch), ch), blending="additive", scale=scale,
                                      contrast_limits=(float(bg), max(hi, float(bg) + 1))))
            L.metadata["channel"] = ch
            self.L[f"img:{ch}"] = L
        for ch in wd.names:
            O = self.v.add_image(z, name=self._name(f"{cfg.display_name(ch)} outlines"),
                                 colormap=_colormap(cfg.color(ch), ch + "o"), blending="additive", scale=scale,
                                 contrast_limits=(0, 1), interpolation2d="nearest")
            img = self.L[f"img:{ch}"]
            O.visible = img.visible
            img.events.visible.connect(lambda e, O=O, img=img: setattr(O, "visible", img.visible))
            self.L[f"out:{ch}"] = O
        self.L["hole"] = keep(self.v.add_image(z, name=self._name("hole"), colormap="gray", blending="additive",
                                               scale=scale, contrast_limits=(0, 1), opacity=0.8,
                                               interpolation2d="nearest"))
        self.L["cells"] = keep(self.v.add_labels(z.astype(np.int32), name=self._name("cells"), scale=scale,
                                                 opacity=0.95))
        self.L["cells"].contour = 2
        self.L["tags"] = keep(self.v.add_points(np.zeros((1, 2)), name=self._name("tags"), scale=scale, size=0.1,
                                                face_color="transparent", border_width=0,
                                                features=pd.DataFrame(dict(t=[""], c=["#ffffff"])),
                                                text=dict(string="{t}", color={"feature": "c"}, size=11,
                                                          anchor="center")))
        self.L["selection"] = self.v.add_shapes([], shape_type="polygon", name=self._name("selection"),
                                                edge_color="#ffffff", face_color="transparent", edge_width=2.0,
                                                scale=scale)

    def show(self, wd, maps: dict, P: pd.DataFrame, G: np.ndarray | None, cfg, cells: pd.DataFrame | None,
             labels: pd.DataFrame | None):
        """P: this well's particles (with dclass, and pclass/p_* once classified). labels: matched hand labels."""
        scale = (wd.px_um, wd.px_um)
        sig = (tuple(wd.names), tuple(l for l, _ in wd.bf), tuple((cfg.display_name(c), cfg.color(c))
                                                                  for c in wd.names))
        first = self.sig is None
        if sig != self.sig or any(L not in self.v.layers for L in self.L.values()):
            self._build(wd, cfg, scale)
            self.sig = sig
        shape_changed = self.current is None or self.current["wd"].hole.shape != wd.hole.shape
        self.current = dict(plate_id=wd.plate_id, well_id=wd.well_id, wd=wd, maps=maps, P=P, scale=scale)
        self.picked = None
        for L in self.L.values():
            if tuple(L.scale[-2:]) != scale:
                L.scale = scale
        if "bf" in self.L and wd.bf:
            bf = np.stack([a for _, a in wd.bf])
            self.L["bf"].data = bf if len(bf) > 1 else bf[0]
            self.L["bf"].contrast_limits = (float(bf.min()), float(bf.max()) or 1.0)
        for k, ch in enumerate(wd.names):
            self.L[f"img:{ch}"].data = wd.chans[k]
            self.L[f"out:{ch}"].data = self._outline_raster(wd, maps, P, ch)
        from skimage.segmentation import find_boundaries
        self.L["hole"].data = find_boundaries(wd.hole, mode="inner").astype(np.uint8)
        self._cells(G, cells, cfg, wd.hole.shape)
        self._tags(P, labels)
        if self.L["selection"].nshapes:
            self.L["selection"].data = []
        if first or shape_changed:
            self.v.reset_view()

    def _outline_raster(self, wd, maps, P, ch) -> np.ndarray:
        """One image per channel: stringent particle edges solid, lenient (hole-threshold) edges dashed."""
        from skimage.segmentation import find_boundaries
        out = np.zeros(wd.hole.shape, np.uint8)
        if not len(P):
            return out
        d = P[P.ch == ch]
        from scipy import ndimage as ndi
        yy, xx = np.indices(out.shape)
        dash = ((yy // 3 + xx // 3) % 2) == 0          # 3-px dashes along straight edges
        for level in ("whole", "hole"):
            if not self.show_levels.get(level, True) or (level, ch) not in maps:
                continue
            g = d[d.level == level]
            if not self.show_context:
                g = g[g.pool]
            if not self.show_noise:
                cls = g["pclass"] if "pclass" in g else g.get("dclass")
                if cls is not None:
                    g = g[~cls.isin(["noise"]).to_numpy()]
            if not len(g):
                continue
            L = maps[(level, ch)]
            sel = np.where(np.isin(L, g.lab.to_numpy()), L, 0)
            # outer edge only: fill each particle's holes so internal speckle gaps draw no edges
            filled = np.zeros_like(sel)
            for i, sl in enumerate(ndi.find_objects(sel), start=1):
                if sl is not None:
                    m = ndi.binary_fill_holes(sel[sl] == i)
                    filled[sl][m] = i
            edge = find_boundaries(filled, mode="inner")
            out |= (edge & dash if level == "hole" else edge).astype(np.uint8)
        return out

    def _cells(self, G, cells, cfg, shape):
        from napari.utils.colormaps import DirectLabelColormap
        L = self.L["cells"]
        if G is None or not G.any():
            L.data = np.zeros(shape, np.int32)
            return
        R = np.where(G > 0, G, np.where(G < 0, 1000 - G, 0)).astype(np.int32)
        colors = {None: "transparent", 0: "transparent"}
        calls = {} if cells is None or not len(cells) else dict(zip(cells.cell_id, cells["call"]))
        for k in np.unique(G[G > 0]):
            colors[int(k)] = call_color(calls.get(int(k), "uncertain"), cfg.color)
        for k in np.unique(G[G < 0]):
            colors[int(1000 - k)] = "#8a929c"
        L.data = R
        L.colormap = DirectLabelColormap(color_dict=colors)

    def _tags(self, P, labels):
        L = self.L["tags"]
        pool = P[P.pool] if len(P) else P
        if len(pool):
            # one tag per object: the lenient piece if there is one, else the stringent particle
            covered = set(pool.loc[pool.level == "hole", "parent"].dropna())
            pool = pool[(pool.level == "hole") | ~pool.oid.isin(covered)]
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
        pts = np.c_[pool.iy.to_numpy(float), pool.ix.to_numpy(float)] if len(pool) else np.zeros((0, 2))
        with L.events.blocker_all():
            L.data = pts
        L.features = pd.DataFrame(dict(t=txt, c=col))
        L.refresh_text()
        L.refresh()

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
        L = self.L.get("selection")
        if L is None:
            return
        L.data = []
        if row is None:
            return
        pg = _outline(self.current["maps"][(row.level, row.ch)] == row.lab)
        if pg is not None:
            L.add_polygons([pg])
            L.edge_color = "#ffffff"
            L.edge_width = 2.0
