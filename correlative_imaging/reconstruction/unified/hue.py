"""The colour of each cell as you would see it on screen, and its distribution on the hue circle.

Per cell, every channel's display colour is mixed, weighted by that channel's signal relative to a typical positive
cell of the same population (hole cells and surrounding cells each against their own; see labels.scales), capped at 1
so one very bright channel does not drown the others. Of the mix:
  hue         the angle on the colour circle (0-360)
  purity      how much one hue dominates (0: equal parts of colours that add up to white, 1: one channel alone)
  strength    the strongest channel's weight (below WEAK: too dim to have a colour; left out of the circle)
Works for any channels and colours (experiment.json); markers take part like any channel.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WEAK = 0.25          # strongest channel below a quarter of a typical positive cell: no colour
PALE = 0.25          # purity below this: the mix is close to white (several channels about equally)
BINS = 36            # 10 degrees each


def _rgb(hexc: str) -> np.ndarray:
    h = (hexc or "#ffffff").lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])


def hue_of(hexc: str) -> float:
    from matplotlib.colors import rgb_to_hsv
    return float(rgb_to_hsv(_rgb(hexc))[0] * 360)


def mix(D: pd.DataFrame, scale: dict, color: dict) -> pd.DataFrame:
    """hue, purity, strength and the mixed colour (hex) of every row of D. scale: {channel: (column, typical)}."""
    from matplotlib.colors import rgb_to_hsv
    rgb = np.zeros((len(D), 3))
    strength = np.zeros(len(D))
    for c, (col, typ) in scale.items():
        if col not in D or not typ or not np.isfinite(typ) or typ <= 0:
            continue
        w = np.clip(D[col].astype(float).to_numpy() / typ, 0, 1)
        w = np.nan_to_num(w)
        rgb += w[:, None] * _rgb(color.get(c))
        strength = np.maximum(strength, w)
    shown = rgb / np.maximum(rgb.max(axis=1, keepdims=True), 1e-9)
    hsv = rgb_to_hsv(np.clip(shown, 0, 1))
    hue = hsv[:, 0] * 360
    hue[strength < WEAK] = np.nan
    hexes = ["#" + "".join(f"{int(round(v * 255)):02x}" for v in row) for row in np.clip(shown * np.minimum(strength, 1)[:, None], 0, 1)]
    return pd.DataFrame({"hue": hue, "purity": hsv[:, 1], "strength": strength, "mix": hexes}, index=D.index)


def add(H: pd.DataFrame, O: pd.DataFrame, F: dict, e: dict) -> None:
    """Adds hue / purity / strength / mix to the hole cells (H) and the surrounding cells (O)."""
    from .labels import scales
    color = {c: v.get("color", "#ffffff") for c, v in e["channels"].items()}
    hs, fs = scales(H, O, F)
    for D, s in ((H, hs), (O, fs)):
        m = mix(D, s, color)
        for k in m:
            D[k] = m[k]


def _curve(h: np.ndarray) -> np.ndarray:
    """Share of cells per 10-degree bin, lightly smoothed around the circle."""
    h = h[np.isfinite(h)]
    n, _ = np.histogram(h, bins=BINS, range=(0, 360))
    n = n / max(len(h), 1)
    return (np.roll(n, 1) + 2 * n + np.roll(n, -1)) / 4


def figure(groups: list, e: dict, title: str = ""):
    """A circular histogram: one closed line per group [(name, hue, purity, line style)], the hue ring outside,
    each channel's own colour marked on the ring. Pale (near-white) mixes are counted, not placed on the circle."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import hsv_to_rgb
    fig = plt.figure(figsize=(5.2, 5.2))
    ax = fig.add_subplot(projection="polar")
    th = np.deg2rad(np.arange(BINS) * 360 / BINS + 180 / BINS)
    top = 0.0
    for name, hue, purity, ls in groups:
        keep = np.isfinite(hue) & (purity >= PALE)
        y = _curve(hue[keep])
        top = max(top, float(y.max()) if len(y) else 0)
    top = top or 1.0
    ring = top * 1.12
    for i in range(360):                                         # the hue ring: where each colour sits
        ax.bar(np.deg2rad(i + 0.5), top * 0.07, width=np.deg2rad(1.02), bottom=ring,
               color=hsv_to_rgb([i / 360, 0.85, 0.9]), linewidth=0)
    for c, v in e["channels"].items():                           # each channel's own colour on the ring
        a = np.deg2rad(hue_of(v.get("color", "#ffffff")))
        ax.plot([a, a], [ring * 0.97, ring + top * 0.1], color="k", lw=1.5)
        ax.text(a, ring + top * 0.2, v.get("name", c), ha="center", va="center", fontsize=8)
    lines = []
    for name, hue, purity, ls in groups:
        n_all = int(np.isfinite(hue).sum())
        keep = np.isfinite(hue) & (purity >= PALE)
        y = _curve(hue[keep])
        tt, yy = np.append(th, th[0]), np.append(y, y[0])
        ln, = ax.plot(tt, yy, ls=ls[0], color=ls[1], lw=2)
        ax.fill(tt, yy, color=ls[1], alpha=0.08)
        pale = n_all - int(keep.sum())
        lines.append((ln, f"{name}: {int(keep.sum())} on the circle, {pale} near white (mixed)"))
    ax.set_ylim(0, ring + top * 0.28)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines["polar"].set_visible(False)
    ax.legend([l for l, _ in lines], [t for _, t in lines], loc="upper center", bbox_to_anchor=(0.5, -0.02),
              fontsize=8, frameon=False)
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


STYLES = [("-", "#1c2230"), ("--", "#7a8294"), ("-", "#d1495b"), ("-", "#2e86ab"), ("-", "#edae49"),
          ("-", "#66a182"), ("--", "#8e5ea2"), ("--", "#d1495b")]


def figures(H: pd.DataFrame, O: pd.DataFrame, e: dict) -> list:
    """[(title, figure)]: surrounding vs hole cells; with your labels, one line per label (hole and surrounding)."""
    out = []
    V = H[H.status_cell == "cell"] if "status_cell" in H else H
    Ol = O[O.learn] if "learn" in O else O
    g = [("surrounding cells", Ol.hue.to_numpy(), Ol.purity.to_numpy(), STYLES[0]),
         ("hole cells", V.hue.to_numpy(), V.purity.to_numpy(), STYLES[1])]
    out.append(("Surrounding and hole cells", figure(g, e)))
    if "predicted_label" in V and (V.predicted_label.fillna("") != "").any():
        both = pd.concat([V[["hue", "purity", "predicted_label"]], Ol[["hue", "purity", "predicted_label"]]])
        both = both.rename(columns={"predicted_label": "lab"})
        labs = [l for l in both.lab.fillna("").value_counts().index if l and " or " not in l and l not in ("unsure", "not sure")][:len(STYLES)]
        g = [(l, both.hue[both.lab == l].to_numpy(), both.purity[both.lab == l].to_numpy(), STYLES[i])
             for i, l in enumerate(labs)]
        out.append(("Per label (the label model's call, hole and surrounding cells)", figure(g, e)))
    return out
