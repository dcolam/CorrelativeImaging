"""The experiment file: what the user tells the workflow, in one place (<out>/experiment.json).

Written with defaults on the first run (channel roles from the Detect run); the user edits it and runs again. Every
key is optional; anything not given is learnt from the data or left out.

  channels           {raw channel: {"name": str, "role": "colour" | "marker", "color": "#rrggbb"}}
                     colour = decides the cell type; marker = measured and reported (e.g. an antibody stain)
  colour_priority    list of colour names, first wins (e.g. ["BFP", "GFP", "mCherry"]): the class of a cell positive
                     in several colours. Empty = the class lists every positive colour.
  marker_negative    {marker name: colour name}: a cell type known NOT to carry the marker; its marker signal is the
                     "no marker" reference. Without it the marker is reported as a strength, not as +/-.
  exclude_wells      ["<plate>:<well>", "<plate>:col 6", "<plate>:rows C-O cols 1-8", "*:col 6"]: imaging failures
                     (lost focus, empty images). Left out of learning and calls.
  empty_wells        same syntax: wells known to hold no cells (controls). Used to learn what a hole without a cell
                     looks like. Optional.
  cell_size_um2      {colour name: area}: the area of one cell body, if known. Otherwise learnt (see README).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

GREY = "#bbbbbb"          # the old fallback: channels saved with it get their colour resolved again
UNKNOWN = "#bfbfbf"       # config.default_color when neither the name nor the wavelength says a colour
FREE = {"#3b6fff": 225, "#38d9e8": 186, "#f2c12e": 45, "#ff8c1a": 30, "#e8413c": 2, "#e0398a": 330, "#2fbf4a": 130}
DEFAULT_COLORS = {"GFP": "#2fbf4a", "mCherry": "#e8413c", "mScarlet": "#e8413c", "BFP": "#3b6fff"}


def _stored_display(run_json: dict) -> dict:
    """Channel names / colours you set in ci gui (Results Explorer: ci_labels.db in the run's output folder)."""
    import sqlite3
    from .data import local_path
    src = run_json.get("source") or {}
    if not src.get("root"):
        return {}
    db = local_path(str(src["root"]), (run_json.get("config") or {}).get("path_map") or {}) / "ci_labels.db"
    if not db.exists():
        return {}
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        rows = con.execute("select channel, display_name, color_hex from channel_display where run_tag=?",
                           (src.get("run_tag", ""),)).fetchall()
        con.close()
    except sqlite3.Error:
        return {}
    return {c: {"name": n, "color": h} for c, n, h in rows if n or h}


def _colour(c: str, name: str, *sources) -> str:
    """The first colour set anywhere (Detect's settings, ci gui), else from the dye name or wavelength."""
    from ..config import default_color
    for d in sources:
        col = (d.get(c) or {}).get("color")
        if col and col.lower() != GREY:
            return col
    guess = default_color(name) if name != c else "#bfbfbf"
    return guess if guess != "#bfbfbf" else default_color(c)


def _hue(hexc: str) -> float:
    import colorsys
    h = hexc.lstrip("#")
    return colorsys.rgb_to_hsv(*(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)))[0] * 360


def _spread(chans: dict) -> None:
    """Channels without a known colour (e.g. named 'ROMK') get the free colour farthest on the hue circle from the
    colours already taken, so every channel stays distinguishable on any setup."""
    for c, v in chans.items():
        if str(v.get("color", "")).lower() not in (UNKNOWN, GREY, ""):
            continue
        taken = [_hue(w["color"]) for k, w in chans.items() if k != c and str(w.get("color", "")).lower()
                 not in (UNKNOWN, GREY, "")]
        dist = lambda h: min((min(abs(h - t), 360 - abs(h - t)) for t in taken), default=360)
        v["color"] = max(FREE, key=lambda k: dist(FREE[k]))


def default(run_json: dict) -> dict:
    cfg = run_json.get("config", {})
    chans = run_json.get("channels", [])
    markers = set(cfg.get("markers") or [])
    disp = cfg.get("display") or {}
    stored = _stored_display(run_json)
    out = {}
    for c in chans:
        name = (disp.get(c) or {}).get("name") or (stored.get(c) or {}).get("name") or c
        out[c] = {"name": name, "role": "marker" if c in markers else "colour",
                  "color": _colour(c, name, disp, stored)}
    _spread(out)
    return {"channels": out, "colour_priority": [], "marker_negative": {}, "exclude_wells": [], "empty_wells": [],
            "cell_size_um2": {}}


def load(out: Path, run_json: dict) -> dict:
    f = Path(out) / "experiment.json"
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(encoding="utf-8", data=json.dumps(default(run_json), indent=2))
    e = default(run_json)
    d = e["channels"]
    e.update(json.loads(f.read_text(encoding="utf-8")))
    # colours saved as the old grey fallback: resolve them again (Detect's settings, ci gui, dye name / wavelength)
    fixed = False
    for c, v in e["channels"].items():
        if str(v.get("color", "")).lower() in (GREY, UNKNOWN, "") and c in d:
            v["color"] = d[c]["color"]
            fixed = True
    if fixed:
        f.write_text(encoding="utf-8", data=json.dumps(e, indent=2))
    return e


def _rows(spec: str) -> set | None:
    m = re.search(r"rows\s+([A-P])\s*-\s*([A-P])", spec)
    return {chr(c) for c in range(ord(m[1]), ord(m[2]) + 1)} if m else None


def _cols(spec: str) -> set | None:
    m = re.search(r"cols?\s+(\d+)(?:\s*-\s*(\d+))?", spec)
    return set(range(int(m[1]), int(m[2] or m[1]) + 1)) if m else None


def matches(specs: list, plate: str, well: str) -> bool:
    """'<plate>:<well>', '<plate>:col 6', '<plate>:rows C-O cols 1-8', '*:col 21-23'."""
    row, col = well[0], int(well[1:])
    for s in specs or []:
        p, _, w = s.partition(":")
        if p not in ("*", plate):
            continue
        w = w.strip()
        if re.fullmatch(r"[A-P]\d{1,2}", w):
            if w[0] == row and int(w[1:]) == col:
                return True
            continue
        rs, cs = _rows(w), _cols(w)
        if (rs is None or row in rs) and (cs is None or col in cs) and (rs or cs):
            return True
    return False


def colour_channels(e: dict) -> list:
    return [c for c, v in e["channels"].items() if v["role"] == "colour"]


def marker_channels(e: dict) -> list:
    return [c for c, v in e["channels"].items() if v["role"] == "marker"]


def name(e: dict, c: str) -> str:
    return e["channels"][c]["name"]
