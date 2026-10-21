"""Is there more than one cell in the hole? Two measures per reconstructed cell, in units of one cell, after Classify.

  body_cells    the cell's body area / the area of ONE patched cell of that plate. That unit is the smaller of
                  * the area of one field cell (reference.cell_um2: the peak of the field objects' areas, averaged
                    over the identity channels): in a large hole the patched cell is seen whole;
                  * the plate's typical body (median over its wells' best cells): in a small hole the aperture
                    squeezes the patched cell to a part of a field cell.
                Two touching cells fused into one body give ~2.
  others_cells  cell-like objects inside the hole that are NOT part of this cell: the in-hole part of every
                stringent particle of an identity channel classified cell, merged or neighbour and not a member,
                in field-cell units (cells_in), summed per channel; the largest channel counts (a double-positive
                cell is seen in two channels). Debris, noise and plugs never count; marker channels never count.

A cell is a possible clump when body_cells >= cfg.clump_body_x or others_cells >= cfg.clump_others. The call does not
change: the well gets `possible_clump` and the reason (`clump_reason`) so the flagged wells can be judged first
(Bulk: cells) and the line set from those judgements.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def body_units(cells: pd.DataFrame, calls: pd.DataFrame, ref: pd.DataFrame, identity_ch: list) -> dict:
    """{plate: area of one patched cell (um2)} (see module doc)."""
    out = {}
    if cells is None or not len(cells):
        return out
    best = cells
    if calls is not None and len(calls) and "cell_id" in calls:
        key = calls[["plate_id", "well_id", "cell_id"]].dropna()
        key = key.assign(cell_id=key.cell_id.astype(int))
        best = cells.merge(key, on=["plate_id", "well_id", "cell_id"])
    typ = best.groupby("plate_id").body_um2.median() if len(best) else pd.Series(dtype=float)
    field = {}
    if ref is not None and "cell_um2" in ref:
        r = ref[ref.ch.isin(identity_ch)]
        field = r.groupby("plate_id").cell_um2.median().to_dict()
    for pl in set(typ.index) | set(field):
        vals = [v for v in (typ.get(pl, np.nan), field.get(pl, np.nan)) if np.isfinite(v) and v > 0]
        out[str(pl)] = float(min(vals)) if vals else float("nan")
    return out


def others_in_hole(cells: pd.DataFrame, F: pd.DataFrame, identity_ch: list) -> pd.Series:
    """Per cell (index of `cells`): cell-equivalents of the other cell-like objects inside the hole."""
    if cells is None or not len(cells) or "cells_in" not in F:
        return pd.Series(np.nan, index=getattr(cells, "index", None), dtype=float)
    p = F[F.pool.astype(bool) & (F.level == "whole") & F.ch.isin(identity_ch)
          & F.pclass.isin(["cell", "merged", "neighbour"])]
    p = p[["plate_id", "well_id", "oid", "ch", "cells_in"]]
    by_well = {k: g for k, g in p.groupby(["plate_id", "well_id"])}
    out = []
    for c in cells.itertuples():
        g = by_well.get((c.plate_id, c.well_id))
        if g is None:
            out.append(0.0)
            continue
        mem = set(str(c.members).split(";")) if isinstance(c.members, str) else set()
        o = g[~g.oid.isin(mem)]
        out.append(float(o.groupby("ch").cells_in.sum().max()) if len(o) else 0.0)
    return pd.Series(out, index=cells.index, dtype=float)


def add_clump_scores(cells: pd.DataFrame, calls: pd.DataFrame, F: pd.DataFrame, ref: pd.DataFrame,
                     identity_ch: list, cfg, units: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Adds body_cells, others_cells, possible_clump, clump_reason to cells and (for the well's best cell) to calls.
    units: {plate: um2} to reuse (re-detecting one well keeps the run's units). Returns (cells, calls, units)."""
    if cells is None or not len(cells):
        if calls is not None:
            calls = calls.assign(body_cells=np.nan, others_cells=np.nan, possible_clump=False, clump_reason="")
        return cells, calls, units or {}
    units = units if units is not None else body_units(cells, calls, ref, identity_ch)
    cells = cells.copy()
    u = np.array([units.get(str(p), np.nan) for p in cells.plate_id], float)
    cells["body_cells"] = cells.body_um2.to_numpy(float) / u
    cells["others_cells"] = others_in_hole(cells, F, identity_ch).to_numpy(float)
    big = cells.body_cells >= cfg.clump_body_x
    oth = cells.others_cells >= cfg.clump_others
    cells["possible_clump"] = (big | oth).to_numpy(bool)
    cells["clump_reason"] = [
        "; ".join(([f"body {b:.1f} cells"] if bb else []) + ([f"others in the hole {o:.1f} cells"] if oo else []))
        for b, o, bb, oo in zip(cells.body_cells, cells.others_cells, big, oth)]
    if calls is not None and len(calls):
        cols = ["plate_id", "well_id", "cell_id", "body_cells", "others_cells", "possible_clump", "clump_reason"]
        calls = calls.drop(columns=[c for c in cols[3:] if c in calls], errors="ignore")
        key = cells[cols].copy()
        calls = calls.merge(key, on=["plate_id", "well_id", "cell_id"], how="left")
        calls["possible_clump"] = calls.possible_clump.fillna(False).astype(bool)
        calls["clump_reason"] = calls.clump_reason.fillna("")
    return cells, calls, units
