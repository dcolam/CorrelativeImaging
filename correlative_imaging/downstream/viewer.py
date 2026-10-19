"""Browser viewer for a downstream run (static files, works from file://; built on ephacRTools' diagviewer.js).

viewer/index.html          the page (www/template.html with the per-well summary table embedded)
viewer/wells/<plate>_<well>.js   JSONP per well: the crop around the hole (fluorescence + BF planes, quantised to
                           uint16 per channel), hole outline, every particle (outline inside the hole and whole
                           extent, role, P(cell), P(neighbour)) and the refined outline of each candidate cell.
Per-well values that are only known after the run-wide calibration (calibrated call, P) live in the summary table.
"""
from __future__ import annotations

import base64
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.measure import approximate_polygon, find_contours

WWW = Path(__file__).parent / "www"
DISP = 2.6        # display window: hole centre +- DISP hole radii (clipped to the crop)


def _polys(mask, off):
    out = []
    for c in find_contours(np.pad(mask, 1).astype(float), .5):
        c = approximate_polygon(c - 1, tolerance=.6)
        if len(c) >= 3:
            out.append([np.round(c[:, 1] + .5 - off[1], 1).tolist(), np.round(c[:, 0] + .5 - off[0], 1).tolist()])
    return out


def _q16(a):
    lo, hi = float(a.min()), float(a.max())
    sc = (hi - lo) / 65535 if hi > lo else 1.0
    return base64.b64encode(np.round((a - lo) / sc).astype("<u2").tobytes()).decode(), [lo, sc]


def _r(v, n=3):
    return None if v is None or (isinstance(v, float) and not np.isfinite(v)) else round(float(v), n)


def well_payload(wd, p_all: pd.DataFrame, scored: pd.DataFrame, refined: list, maps: dict) -> dict:
    """Pixels + overlays for one well. p_all: every extracted object (incl. field cells); scored: the objects
    touching the hole with role etc.; refined: dicts with 'mask'; maps: label images from extract_particles."""
    H, W = wd.hole.shape
    hy, hx = np.where(wd.hole)
    R = np.sqrt(wd.hole.sum() / np.pi)
    h = int(DISP * R)
    y1, x1 = max(int(hy.mean()) - h, 0), max(int(hx.mean()) - h, 0)
    y2, x2 = min(int(hy.mean()) + h, H), min(int(hx.mean()) + h, W)
    planes = list(wd.chans) + [a for _, a in wd.bf]
    chans, q = zip(*[_q16(np.asarray(a, np.float32)[y1:y2, x1:x2]) for a in planes])
    sc = scored.set_index("oid") if len(scored) else None
    ov = []
    for r in p_all.itertuples():
        full = maps[(r.seg, r.ch)] == r.label
        if not full[y1:y2, x1:x2].any():
            continue
        e = dict(oid=r.oid, seg=r.seg, ch=r.ch, touch=bool(r.touch), full=_polys(full[y1:y2, x1:x2], (0, 0)))
        if r.touch:
            ins = np.zeros(H * W, bool)
            ins[r.foot] = True
            ins = ins.reshape(H, W)
            iy, ix = np.where(ins[y1:y2, x1:x2])
            e["ins"] = _polys(ins[y1:y2, x1:x2], (0, 0))
            e["at_in"] = [float(np.median(ix)) + .5, float(np.median(iy)) + .5] if len(ix) else None
            oy, ox = np.where((full & ~wd.hole)[y1:y2, x1:x2])
            e["at_out"] = [float(np.median(ox)) + .5, float(np.median(oy)) + .5] if len(ox) else None
            if sc is not None and r.oid in sc.index:
                s = sc.loc[r.oid]
                e.update(role=s.role, p_cell=_r(s.p_cell), p_nb=_r(s.p_nb), snr=_r(s.snr, 1), area=_r(s.in_um2, 1),
                         share=_r(s.share, 2), fill=_r(s.fill, 2), bleed_lr=_r(s.get("bleed_lr"), 2),
                         cell=None if pd.isna(s.get("cell_id")) else int(s.cell_id), ref=s.get("ref"))
        else:
            e["role"] = "field"
        ov.append(e)
    ref_cells = [dict(cell=r["cell_id"], outline=_polys(r["mask"][y1:y2, x1:x2], (0, 0))) for r in refined]
    return dict(plate=wd.plate_id, well=wd.well_id, w=int(x2 - x1), h=int(y2 - y1), px_um=wd.px_um,
                labels=list(wd.names) + [lab for lab, _ in wd.bf], n_fluo=len(wd.names), q=list(q),
                channels=list(chans), hole=_polys(wd.hole[y1:y2, x1:x2], (0, 0)), ov=ov, refined=ref_cells)


def write_payload(viewer_dir: Path, payload: dict) -> None:
    d = Path(viewer_dir) / "wells"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{payload['plate']}_{payload['well']}.js").write_text(
        "DV_WELL(" + json.dumps(payload, separators=(",", ":")) + ");", encoding="utf-8")


def write_viewer(viewer_dir: Path, calls: pd.DataFrame, cells: pd.DataFrame, cfg, col_data=None,
                 title: str = "Patched-cell identity") -> Path:
    vd = Path(viewer_dir)
    vd.mkdir(parents=True, exist_ok=True)
    for f in ("diagviewer.js", "diagviewer.css"):
        shutil.copy(WWW / f, vd / f)
    cd = None
    if col_data is not None:
        cd = col_data.set_index(["plate_id", "well_id"])
    cand = cells[cells.kind == "candidate"] if len(cells) else cells
    rows = []
    for r in calls.itertuples():
        d = dict(plate=r.plate_id, well=r.well_id, tier=r.tier, call=r.call, call_cal=r.call_cal,
                 n_cand=int(r.n_cand), n_nb=int(r.n_nb), n_weak=int(r.n_weak), n_marker=int(r.n_marker_only),
                 p_cell=_r(r.p_cell), p_nb=_r(r.p_nb), agree=None if pd.isna(r.stringent_agrees) else bool(r.stringent_agrees),
                 chans=r.cand_chans if isinstance(r.cand_chans, str) else "",
                 P={c: _r(getattr(r, f"p_{c}", None)) for c in cfg.reporters},
                 body=_r(getattr(r, "body_um2", None), 1), grow=_r(getattr(r, "grow_x", None), 2),
                 in_hole=_r(getattr(r, "body_in_hole", None), 2))
        cc = cand[(cand.plate_id == r.plate_id) & (cand.well_id == r.well_id)] if len(cand) else cand
        d["cells"] = [dict(cell=int(c.cell_id), chans=c.chans, call=c.get("call_cal"), body=_r(c.get("body_um2"), 1),
                           grow=_r(c.get("grow_x"), 2), **{f"n_{x}": _r(c.get(f"n_{x}"), 2) for x in cfg.reporters},
                           **{f"p_{x}": _r(c.get(f"p_{x}"), 3) for x in cfg.reporters})
                      for _, c in cc.iterrows()]
        if cd is not None and (r.plate_id, r.well_id) in cd.index:
            e = cd.loc[(r.plate_id, r.well_id)]
            d["info"] = {k: (None if pd.isna(v) else str(v)) for k, v in e.items()}
        rows.append(d)
    cfgv = dict(title=title, reporters=cfg.reporters, markers=cfg.markers, colors=cfg.colors,
                group_by=[c for c in ("Condition", "Induction") if cd is not None and c in cd.columns])
    html = (WWW / "template.html").read_text(encoding="utf-8")
    html = html.replace("/*DATA*/[]", json.dumps(rows, separators=(",", ":"))).replace("/*CFG*/{}", json.dumps(cfgv))
    html = html.replace("<title>Patched-cell identity</title>", f"<title>{title}</title>")
    (vd / "index.html").write_text(html, encoding="utf-8")
    return vd / "index.html"


def rebuild_index(out_dir: str | Path) -> Path:
    """Rewrite viewer/index.html of an existing results folder from the current template (no rerun)."""
    from .config import load_config
    out = Path(out_dir)
    cfg = load_config(out / "config.json") if (out / "config.json").exists() else None
    if cfg is None:
        from .config import Config
        cfg = Config()
    calls = pd.read_csv(out / "calls.csv", dtype={"plate_id": str, "well_id": str})
    cells = pd.read_csv(out / "cells.csv", dtype={"plate_id": str, "well_id": str})
    cd = pd.read_csv(out / "col_data.csv", dtype={"plate_id": str, "well_id": str}) if (out / "col_data.csv").exists() else None
    title = "Patched-cell identity"
    old = out / "viewer" / "index.html"
    if old.exists():
        import re
        m = re.search(r"<title>(.*?)</title>", old.read_text(encoding="utf-8"))
        title = m.group(1) if m else title
    return write_viewer(out / "viewer", calls, cells, cfg, col_data=cd, title=title)


def check_diagviewer(ephac_www: str | Path) -> bool:
    """True when the bundled diagviewer files equal the ephacRTools copies."""
    md5 = lambda p: hashlib.md5(Path(p).read_bytes()).hexdigest()
    return all(md5(WWW / f) == md5(Path(ephac_www) / f) for f in ("diagviewer.js", "diagviewer.css"))
