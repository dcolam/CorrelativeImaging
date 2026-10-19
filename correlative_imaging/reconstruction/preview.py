"""A first look right after Detect, without any model: provisional particle classes and a preview per well.

Everything here is a PREVIEW, kept in its own columns (`dclass`, `preview_*`), never written into the label store
and never used by the classifier, so the model is not trained on these rules.

Provisional class of each particle in the hole (`dclass`), first rule that applies:
  noise      no clear signal: SNR < 3 against the frame background, or dimmer than 5 % of a typical cell
             (on the ROMK hand labels: 316/354 noise caught, 47/620 cells lost)
  debris     (2+ channels) small (< 5 um2 inside) and bright in every other channel too (flatness > 0.5, all other
             channels SNR > 3)
  neighbour  mostly outside (share < 0.5), sitting in the outer half of the hole (inside centroid > 0.5 hole radii
             from the centre), and its whole object looks like a cell next to the hole: at least 0.3 typical
             cells bright, or close to the typical field cell in brightness, size and shape. A whole object cut by
             the window edge has an unknown size, so only its brightness counts.
  inside     everything else

Well preview, from the `inside` particles of the identity channels (markers excluded):
  preview_status    no particles | neighbours only | one object | several objects (inside particles grouped when they
                    overlap or lie within `cell_gap_um` of each other)
  preview_dominant  the channel with the most signal (sum over inside particles of brightness above the hole
                    background in typical-cell units x area); "a+b" when the top two are within 1.5x
  preview_sig_<ch>  that signal per channel
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components

from .config import Config

DCLASSES = ["inside", "neighbour", "debris", "noise"]
NB_MIN_REL = 0.3       # whole object at least 0.3 typical cells bright
NB_MAX_D2 = 8.0        # field-likeness distance (4 squared z-scores; ~4 = typical)
DOMINANCE = 1.5
NOISE_SNR = 3.0        # below 3 frame-noise SDs
NOISE_REL = 0.05       # or below 5 % of a typical cell of that channel


def provisional_classes(F: pd.DataFrame) -> pd.Series:
    """F: build_features output (pool + context). Returns dclass for pool particles (None for context objects)."""
    pool = F.pool.to_numpy(bool)
    by = (F[~F.outer.fillna(False).astype(bool)] if "outer" in F else F).set_index(["plate_id", "well_id", "oid"])
    par = by.reindex(list(zip(F.plate_id, F.well_id, F.parent)))
    par_rel = par.rel_full.to_numpy(float)
    par_d2 = par.field_d2.to_numpy(float)
    par_edge = par.edge.to_numpy(object)
    par_edge = np.array([bool(e) if e is not None and e == e else False for e in par_edge])
    size_ok = np.where(par_edge, False, np.nan_to_num(par_d2, nan=np.inf) <= NB_MAX_D2)
    bright = np.nan_to_num(par_rel, nan=0) >= NB_MIN_REL
    noise = (F.snr < NOISE_SNR) | (F.rel < NOISE_REL)
    debris = (F.in_um2 < 5) & (F.f_flat > 0.5) & (F.f_oth_pos >= 1)
    nb = (F.share < 0.5) & (F.d_in > 0.5) & (bright | size_ok)
    out = np.select([~pool, noise, debris, nb], [None, "noise", "debris", "neighbour"], "inside")
    return pd.Series(out, index=F.index, dtype=object)


def well_previews(F: pd.DataFrame, W: pd.DataFrame, identity: list, cfg: Config) -> pd.DataFrame:
    rows = []
    Wi = W.set_index(["plate_id", "well_id"])
    groups = {k: g for k, g in F[F.pool].groupby(["plate_id", "well_id"])}
    for (pl, wl), w in Wi.iterrows():
        g = groups.get((pl, wl))
        row = dict(plate_id=pl, well_id=wl)
        sig = {ch: 0.0 for ch in identity}
        if g is None or not len(g):
            rows.append(dict(row, preview_status="no particles", preview_dominant="", n_inside=0, n_nb=0,
                             **{f"preview_sig_{c}": 0.0 for c in identity}))
            continue
        ins = g[(g.dclass == "inside") & g.ch.isin(identity)]
        n_nb = int((g.dclass == "neighbour").sum())
        for ch in identity:
            d = ins[ins.ch == ch]
            if not len(d):
                continue
            ref = d.ref_med.to_numpy(float)
            s = np.clip((d.in_mean.to_numpy(float) - float(w[f"hbg_med_{ch}"])) / ref, 0, None) * d.in_um2.to_numpy(float)
            per_level = pd.Series(s, index=d.index).groupby(d.level).sum()
            sig[ch] = float(np.nan_to_num(per_level.max()))       # one threshold level per channel (no double count)
        # objects: inside particles that overlap / lie within the gap, any channel
        if len(ins):
            px = float(w.px_um)
            y, x = ins.iy.to_numpy(float), ins.ix.to_numpy(float)
            r = np.sqrt(ins.in_px.to_numpy(float) / np.pi)
            D = np.hypot(y[:, None] - y[None], x[:, None] - x[None])
            A = D <= (r[:, None] + r[None]) * 0.8 + cfg.cell_gap_um / px
            n_obj = connected_components(A, directed=False)[0]
        else:
            n_obj = 0
        status = ("one object" if n_obj == 1 else "several objects" if n_obj > 1 else
                  "neighbours only" if n_nb else "no particles")
        top = sorted(sig.items(), key=lambda t: -t[1])
        if not top or top[0][1] <= 0:
            dom = ""
        elif len(top) > 1 and top[1][1] > 0 and top[0][1] / top[1][1] < DOMINANCE:
            dom = f"{top[0][0]}+{top[1][0]}"
        else:
            dom = top[0][0]
        rows.append(dict(row, preview_status=status, preview_dominant=dom, n_inside=len(ins), n_nb=n_nb,
                         **{f"preview_sig_{c}": round(v, 3) for c, v in sig.items()}))
    return pd.DataFrame(rows)


def suggest_markers(previews: pd.DataFrame, identity: list, min_share: float = 0.8, min_sig: float = 2.0) -> list:
    """Channels with clear signal (> `min_sig` typical-cell um2) in nearly every hole that holds anything (>= 80 %),
    while at least one other identity channel is clearly not everywhere (< 50 %): candidates for `markers`
    (measured, not identity evidence). When every channel is everywhere (e.g. aperture glow in all channels), the
    preview cannot tell a marker apart, and nothing is suggested. A suggestion only; nothing changes by itself."""
    p = previews[previews.n_inside > 0]
    if len(p) < 20 or len(identity) < 2:
        return []
    share = {ch: float((p[f"preview_sig_{ch}"] > min_sig).mean()) for ch in identity}
    cand = [ch for ch, v in share.items() if v >= min_share]
    if not cand or len(cand) == len(identity) or not any(v < 0.5 for v in share.values()):
        return []
    return cand
