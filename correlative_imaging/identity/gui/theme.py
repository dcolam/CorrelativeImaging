"""Colours from napari's own theme (so the docks look like napari in dark and light), plus the class palette and
call colours. Channel colours come from the run's display settings, never from here."""
from __future__ import annotations

import numpy as np

# class palette (particle classes), chosen to stay apart from typical channel colours
CLASS_COLORS = {"cell": "#ffd23f", "neighbour": "#9aa5b1", "merged": "#ff8c42", "debris": "#c77dff",
                "noise": "#5c6670"}
CLASS_KEYS = {"cell": "1", "neighbour": "2", "merged": "3", "debris": "4", "noise": "5"}


def theme(viewer=None) -> dict:
    """napari theme colours as hex strings (background, foreground, primary, secondary, highlight, text, icon,
    warning, error, current)."""
    from napari.utils.theme import get_theme
    name = getattr(viewer, "theme", None) or "dark"
    try:
        t = get_theme(name)
    except Exception:
        t = get_theme("dark")
    out = {}
    for k in ("background", "foreground", "primary", "secondary", "highlight", "text", "icon", "warning", "error",
              "current"):
        v = getattr(t, k, None)
        out[k] = v.as_hex() if hasattr(v, "as_hex") else str(v)
    return out


def hex_rgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])


def blend(hexes: list[str]) -> str:
    if not hexes:
        return "#808080"
    c = np.clip(np.sum([hex_rgb(h) for h in hexes], 0), 0, 1)
    return "#" + "".join(f"{int(round(v * 255)):02x}" for v in c)


def call_color(call: str, color_of) -> str:
    """color_of(channel) -> hex. Combinations blend their channels; none/uncertain/no cell are neutral."""
    if call in (None, "", "no cell") or (isinstance(call, float) and np.isnan(call)):
        return "#3a3f47"
    if call == "none":
        return "#8a929c"
    if call == "uncertain":
        return "#e0a526"
    return blend([color_of(c) for c in str(call).split("+")])


def chip(text: str, color: str, fg: str | None = None) -> str:
    """Small rounded label (rich text) for tables and headers."""
    rgb = hex_rgb(color)
    fg = fg or ("#111111" if rgb @ [0.299, 0.587, 0.114] > 0.55 else "#f0f1f2")
    return (f'<span style="background-color:{color}; color:{fg}; border-radius:3px; '
            f'padding:1px 5px;">&nbsp;{text}&nbsp;</span>')


def stylesheet(t: dict) -> str:
    """Extra rules on top of napari's stylesheet: accent buttons, section headers, compact tables."""
    return f"""
    QPushButton[accent="true"] {{ background-color: {t['current']}; color: {t['text']}; font-weight: 600;
                                  padding: 4px 10px; border-radius: 3px; }}
    QPushButton[accent="true"]:hover {{ background-color: {t['highlight']}; }}
    QPushButton[accent="true"]:disabled {{ background-color: {t['foreground']}; color: {t['secondary']}; }}
    QLabel[role="section"] {{ color: {t['text']}; font-weight: 600; font-size: 11pt; padding-top: 6px; }}
    QLabel[role="hint"] {{ color: {t['secondary'] if t['secondary'] != t['background'] else t['icon']};
                           font-size: 9pt; }}
    QLabel[role="metric"] {{ font-size: 16pt; font-weight: 600; color: {t['text']}; }}
    QFrame[role="card"] {{ background-color: {t['foreground']}; border-radius: 4px; }}
    QTableView {{ gridline-color: {t['background']}; }}
    QPlainTextEdit[role="log"] {{ font-family: Consolas, monospace; font-size: 9pt; }}
    """
