"""The napari canvas: an item large, one layer per channel in its colour (toggle them in the layer list), the hole
(yellow), the grown cell (cyan), the asked particle (white), or a surrounding cell's whole image with the cell marked."""
from __future__ import annotations

import numpy as np
from skimage.segmentation import find_boundaries

PREFIX = "· "


def _cmap(hexc: str, name: str):
    from napari.utils.colormaps import Colormap
    h = (hexc or "#ffffff").lstrip("#")
    rgb = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    return Colormap([[0, 0, 0, 1], rgb + [1]], name=f"v2_{name}_{h}")


def clear(viewer):
    for L in [L for L in viewer.layers if L.name.startswith(PREFIX)]:
        viewer.layers.remove(L)


def _outline(viewer, mask, name, color, scale):
    m = find_boundaries(mask, mode="inner") if mask.any() else mask
    viewer.add_image(m.astype(np.float32), name=PREFIX + name, colormap=_cmap(color, name), blending="additive",
                     contrast_limits=(0, 1), scale=scale)


def show(viewer, V, item: dict) -> str:
    """Draws the item; returns a one-line description for the canvas overlay."""
    clear(viewer)
    kind = item.get("kind")
    if kind == "well" and not V._has_cells(item):
        kind = "hole"                                   # no surrounding cells to find the whole image by: the hole
    if kind in ("hole", "particle"):
        h = V.hole(item["plate_id"], item["well_id"])
        if h is None:
            return f"{item['plate_id']} {item['well_id']}: no hole crop"
        px = float(h.get("px", 1.0))
        sc = (px, px)
        P = V.raw_hole(h)                                # raw pixel values; the range only sets the contrast
        for c in V.chans:
            if c in P:
                a, lim = P[c]
                viewer.add_image(a, name=PREFIX + V.name[c], colormap=_cmap(V.color[c], c), blending="additive",
                                 contrast_limits=lim, scale=sc)
        _outline(viewer, h["hole"] > 0, "hole", "#ffff00", sc)
        _outline(viewer, h["cell"] > 0, "hole cell", "#00ffff", sc)
        if kind == "particle":
            ch, lev, lab = str(item["obj"]).split("|")
            L = h["maps"].get((lev, ch))
            if L is not None and L.shape == h["hole"].shape:
                _outline(viewer, L == int(lab), "particle", "#ffffff", sc)
        viewer.reset_view()
        return f"{item['plate_id']} {item['well_id']} · hole" + (f" · {item.get('caption', '')}" if item.get("caption") else "")
    if kind in ("field", "well"):
        if kind == "field":
            r = V.field_row(item["plate_id"], item["image_id"], item["obj"])
        else:
            FC = V.field_cells
            g = FC[(FC.plate_id == item["plate_id"]) & (FC.well_id == item["well_id"])]
            r = g.iloc[0] if len(g) else None
        if r is None:
            return f"{item['plate_id']} {item['well_id']}: image not found"
        Pw = V.whole(r.diag, r.well_raw)
        P = V.raw_field(Pw)
        px = float(r.px_um)
        sc = (px, px)
        for c in V.chans:
            if c in P:
                a, lim = P[c]
                viewer.add_image(a, name=PREFIX + V.name[c], colormap=_cmap(V.color[c], c), blending="additive",
                                 contrast_limits=lim, scale=sc)
        if kind == "field":
            viewer.add_points([[r.centroid_row * px, r.centroid_col * px]], name=PREFIX + "cell", size=12,
                              face_color="transparent", border_color="white", symbol="ring")
            viewer.camera.center = (r.centroid_row * px, r.centroid_col * px)
            viewer.camera.zoom = 6
        else:
            viewer.reset_view()
        return f"{item['plate_id']} {item['well_id']} · " + ("surrounding cell" if kind == "field" else "whole well")
    return ""
