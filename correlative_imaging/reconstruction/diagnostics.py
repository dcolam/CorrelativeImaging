"""Diagnostic plots written after Detect (and after Classify): what the data look like before anything is trained.

All are distributions (ECDFs, step histograms, density hexbins, heatmaps), never bar plots. Classes before Classify are
the Detect preview's *provisional* classes. Files go to <results>/diagnostics/*.png, plus diagnostics.json with the
numbers behind them. Rendering uses matplotlib's Agg canvas only (no pyplot), so it runs in worker threads and on
the command line, and a failing plot never fails Detect.

  1_snr                 per channel: ECDF of particle SNR = (mean inside - frame background) / frame background SD,
                        one curve per provisional class; the noise rule's cut (3) marked
  2_brightness          per channel: ECDF of brightness in typical-cell units ((mean - frame bg) / median field cell
                        of that plate and channel); the noise rule's cut (0.05) marked
  3_dynamic_range       per channel and plate: step histogram over wells of log10 dynamic range = (brightest particle
                        in the hole - hole background median) / hole background SD
  4_particles_per_well  ECDF over wells of the number of particles in the hole per provisional class, and of the
                        "inside" particles per channel
  5_background          per channel: ECDF over wells of the hole background median (solid) and the frame background
                        (dashed), one colour per plate: plate effects and aperture glow
  6_cross_channel       channel x channel: per particle, brightness of every channel over the same footprint
                        (asinh of typical-cell units / 0.05, density); diagonal = its distribution; upper = Spearman rho over particles
                        and over wells (brightest particle per well and channel)
  7_coloc               within-particle colocalisation: pixel-wise Pearson r between a particle's channel and each other
                        channel, ECDF per pair, and the median per pair as a heatmap
  8_identity            after Classify: per identity channel, the distribution of the cells' brightness (asinh of
                        typical-cell units) with the calibration's two components and the call's cut
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

CLASS_COLORS = {"plug": "#e05a8a", "inside": "#d4a017", "neighbour": "#6c7a89", "debris": "#9b5de5", "noise": "#b0b7bf",
                "cell": "#2bb3a3", "merged": "#e05a8a"}
PLATE_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
                "#bcbd22", "#17becf"]


def _fig(nrows, ncols, w=3.6, h=3.0):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    fig = Figure(figsize=(w * ncols, h * nrows), dpi=110, layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(nrows, ncols, squeeze=False)
    return fig, axes


def _ecdf(ax, v, label=None, color=None, ls="-", lw=1.5):
    v = np.sort(np.asarray(v, float)[np.isfinite(v)])
    if not len(v):
        return
    ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", label=f"{label} (n={len(v)})" if label else None,
            color=color, ls=ls, lw=lw)


def _style(ax, xlabel, ylabel="fraction ≤ x"):
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25, lw=0.5)


def _color(cfg, ch):
    try:
        return cfg.color(ch)
    except Exception:
        return None


def _name(cfg, ch):
    try:
        return cfg.display_name(ch)
    except Exception:
        return ch


def _save(fig, path: Path, title: str):
    import textwrap
    width = max(int(fig.get_figwidth() * 14), 60)
    fig.suptitle("\n".join(textwrap.wrap(title, width)), fontsize=9)
    fig.savefig(path, facecolor="white")
    return path


def _pos(x, lo=1e-2):
    x = np.asarray(x, float)
    return np.where(np.isfinite(x), np.clip(x, lo, None), np.nan)


def detect_diagnostics(out_dir, cfg=None) -> list:
    """Writes the Detect plots (1-7) and diagnostics.json; returns the files written."""
    from .pipeline import Results
    res = Results(out_dir)
    cfg = cfg or res.config
    d = res.dir / "diagnostics"
    d.mkdir(exist_ok=True)
    P, W, names = res.particles, res.wells, res.channels
    pool = P[P.pool.astype(bool)].copy()
    cls = pool["dclass"] if "dclass" in pool else pd.Series("", index=pool.index)
    written, summary = [], {"channels": {}}
    plates = sorted(W.plate_id.unique())
    pcol = {p: PLATE_COLORS[i % len(PLATE_COLORS)] for i, p in enumerate(plates)}
    nc = len(names)
    jobs = [_p_snr, _p_brightness, _p_dynamic, _p_counts, _p_background, _p_cross, _p_coloc, _p_cell_size]
    for job in jobs:
        try:
            f = job(d, cfg, P, pool, cls, W, names, nc, plates, pcol, summary)
            if f:
                written.append(f)
        except Exception as e:                           # a plot never fails Detect
            summary.setdefault("errors", []).append(f"{job.__name__}: {e}")
    (d / "diagnostics.json").write_text(json.dumps(summary, indent=2, default=float))
    return written


def _p_snr(d, cfg, P, pool, cls, W, names, nc, plates, pcol, summary):
    fig, ax = _fig(1, nc)
    for k, ch in enumerate(names):
        g = pool[pool.ch == ch]
        a = ax[0, k]
        for c in ("inside", "merged", "neighbour", "debris", "noise"):
            _ecdf(a, _pos(g.snr[cls[g.index] == c]), c, CLASS_COLORS[c])
        _ecdf(a, _pos(g.snr), "all", "black", ls=":", lw=1)
        a.axvline(3, color="#c0392b", lw=0.8, ls="--")
        a.set_xscale("log")
        a.set_title(_name(cfg, ch), fontsize=9, color=_color(cfg, ch))
        _style(a, "SNR (frame background SDs)")
        a.legend(fontsize=6)
        ins = g.snr[cls[g.index] == "inside"]
        summary["channels"].setdefault(ch, {}).update(
            n_particles=int(len(g)), n_inside=int(len(ins)), median_snr_inside=float(np.nanmedian(ins)) if len(ins) else None,
            share_noise=float((cls[g.index] == "noise").mean()) if len(g) else None)
    return _save(fig, d / "1_snr.png", "Particles in the hole: SNR = (mean inside − frame background) / frame "
                                         "background SD, by provisional class (dashed: noise cut 3)")


def _p_brightness(d, cfg, P, pool, cls, W, names, nc, plates, pcol, summary):
    fig, ax = _fig(1, nc)
    for k, ch in enumerate(names):
        g = pool[pool.ch == ch]
        a = ax[0, k]
        for c in ("inside", "merged", "neighbour", "debris", "noise"):
            _ecdf(a, _pos(g.rel[cls[g.index] == c], 1e-3), c, CLASS_COLORS[c])
        a.axvline(0.05, color="#c0392b", lw=0.8, ls="--")
        a.axvline(1, color="black", lw=0.6, ls=":")
        a.set_xscale("log")
        a.set_title(_name(cfg, ch), fontsize=9, color=_color(cfg, ch))
        _style(a, "brightness (typical field cell = 1)")
        a.legend(fontsize=6)
        ins = g.rel[cls[g.index] == "inside"]
        summary["channels"].setdefault(ch, {}).update(
            median_rel_inside=float(np.nanmedian(ins)) if len(ins) else None)
    return _save(fig, d / "2_brightness.png", "Brightness in typical-cell units = (mean − frame background) / "
                                                "median field cell of that plate and channel (dashed: noise cut 0.05)")


def _p_dynamic(d, cfg, P, pool, cls, W, names, nc, plates, pcol, summary):
    fig, ax = _fig(1, nc)
    for k, ch in enumerate(names):
        a = ax[0, k]
        g = pool[pool.ch == ch]
        mx = g.groupby(["plate_id", "well_id"]).in_max.max()
        w = W.set_index(["plate_id", "well_id"]).reindex(mx.index)
        dr = (mx.to_numpy(float) - w[f"hbg_med_{ch}"].to_numpy(float)) / w[f"hbg_sd_{ch}"].to_numpy(float)
        ldr = np.log10(np.clip(dr, 1e-1, None))
        s = pd.Series(ldr, index=mx.index)
        bins = np.linspace(np.nanmin(ldr) if np.isfinite(ldr).any() else 0, np.nanmax(ldr) if np.isfinite(ldr).any() else 1, 30)
        for pl in plates:
            v = s[s.index.get_level_values(0) == pl].dropna()
            if len(v):
                a.hist(v, bins=bins, histtype="step", color=pcol[pl], label=f"{pl} (n={len(v)})", lw=1.3)
        a.set_title(_name(cfg, ch), fontsize=9, color=_color(cfg, ch))
        _style(a, "log10 dynamic range", "wells")
        a.legend(fontsize=6)
        summary["channels"].setdefault(ch, {}).update(
            median_log10_dynamic_range=float(np.nanmedian(ldr)) if len(ldr) else None,
            wells_without_particles=int(len(W) - len(mx)))
    return _save(fig, d / "3_dynamic_range.png", "Dynamic range per well = (brightest particle in the hole − hole "
                                                   "background median) / hole background SD, one curve per plate")


def _p_counts(d, cfg, P, pool, cls, W, names, nc, plates, pcol, summary):
    fig, ax = _fig(1, 2, w=4.6)
    keys = W[["plate_id", "well_id"]].apply(tuple, axis=1)
    a = ax[0, 0]
    for c in ("inside", "merged", "neighbour", "debris", "noise"):
        n = pool[cls == c].groupby(["plate_id", "well_id"]).size().reindex(keys, fill_value=0)
        _ecdf(a, n.to_numpy(), c, CLASS_COLORS[c])
        summary.setdefault("per_well", {})[c] = dict(median=float(n.median()), share_wells_with_any=float((n > 0).mean()))
    _style(a, "particles per well")
    a.set_title("by provisional class", fontsize=9)
    a.legend(fontsize=6)
    a = ax[0, 1]
    for ch in names:
        n = pool[(cls == "inside") & (pool.ch == ch)].groupby(["plate_id", "well_id"]).size().reindex(keys, fill_value=0)
        _ecdf(a, n.to_numpy(), _name(cfg, ch), _color(cfg, ch))
    _style(a, "'inside' particles per well")
    a.set_title("by channel (both thresholds)", fontsize=9)
    a.legend(fontsize=6)
    return _save(fig, d / "4_particles_per_well.png", f"Particles in the hole per well ({len(W)} wells)")


def _p_background(d, cfg, P, pool, cls, W, names, nc, plates, pcol, summary):
    fig, ax = _fig(1, nc)
    for k, ch in enumerate(names):
        a = ax[0, k]
        for pl in plates:
            w = W[W.plate_id == pl]
            _ecdf(a, w[f"hbg_med_{ch}"], f"{pl} hole", pcol[pl])
            _ecdf(a, w[f"bg_mean_{ch}"], None, pcol[pl], ls="--", lw=1)
        a.set_title(_name(cfg, ch), fontsize=9, color=_color(cfg, ch))
        _style(a, "background (grey value)")
        a.legend(fontsize=6)
    return _save(fig, d / "5_background.png", "Background per well: hole background median (solid) and frame "
                                                "background outside particles (dashed), one colour per plate")


def _footprint_rel(pool, W, names, ref):
    """Brightness of every channel over each particle's footprint (inside the hole), typical-cell units."""
    Wi = W.set_index(["plate_id", "well_id"])
    key = list(zip(pool.plate_id, pool.well_id))
    R_ = ref.set_index(["plate_id", "ch"])
    out = {}
    for c in names:
        if f"in_{c}" not in pool:
            continue
        b = Wi[f"bg_mean_{c}"].reindex(key).to_numpy(float)
        rm = R_.ref_med.reindex([(p, c) for p in pool.plate_id]).to_numpy(float)
        out[c] = (pool[f"in_{c}"].to_numpy(float) - b) / rm
    return pd.DataFrame(out, index=pool.index)


def _p_cross(d, cfg, P, pool, cls, W, names, nc, plates, pcol, summary):
    from scipy.stats import spearmanr
    from .pipeline import Results
    if nc < 2:
        return None
    ref = Results(d.parent).reference
    g = pool[(pool.level == "hole")]                       # one footprint per object (the lenient piece)
    R = _footprint_rel(g, W, names, ref)
    L = np.arcsinh(R / 0.05)                             # like the identity call: linear near 0, log-like above
    per_well = R.assign(plate_id=g.plate_id, well_id=g.well_id).groupby(["plate_id", "well_id"]).max()
    fig, ax = _fig(nc, nc, w=2.6, h=2.4)
    rho = {}
    for i, a_ in enumerate(names):
        for j, b_ in enumerate(names):
            a = ax[i, j]
            a.tick_params(labelsize=6)
            if i == j:
                a.hist(L[a_].dropna(), bins=40, histtype="step", color=_color(cfg, a_) or "black")
                a.set_title(_name(cfg, a_), fontsize=8, color=_color(cfg, a_))
            elif i > j:
                ok = L[a_].notna() & L[b_].notna()
                if ok.sum() > 5:
                    a.hexbin(L[b_][ok], L[a_][ok], gridsize=30, bins="log", cmap="viridis", mincnt=1)
                a.set_xlabel(f"asinh({_name(cfg, b_)} / 0.05)", fontsize=6)
                a.set_ylabel(f"asinh({_name(cfg, a_)} / 0.05)", fontsize=6)
            else:
                ok = R[a_].notna() & R[b_].notna()
                r_p = spearmanr(R[a_][ok], R[b_][ok]).statistic if ok.sum() > 5 else np.nan
                okw = per_well[a_].notna() & per_well[b_].notna()
                r_w = spearmanr(per_well[a_][okw], per_well[b_][okw]).statistic if okw.sum() > 5 else np.nan
                rho[f"{a_}|{b_}"] = dict(particles=float(r_p), wells=float(r_w))
                a.axis("off")
                a.text(0.5, 0.5, f"{_name(cfg, a_)} vs {_name(cfg, b_)}\nρ particles {r_p:.2f}\nρ wells {r_w:.2f}",
                       ha="center", va="center", fontsize=8, transform=a.transAxes,
                       color="#c0392b" if np.nanmax([abs(r_p), abs(r_w)]) >= 0.5 else "black")
    summary["cross_channel_spearman"] = rho
    return _save(fig, d / "6_cross_channel.png", "Cross-channel: brightness of each channel over the same particle "
                                                   "footprint, asinh(typical-cell units / 0.05). Lower: density (log "
                                                   "counts). Upper: Spearman ρ over particles and over wells (brightest "
                                                   "particle per well and channel)")


def _p_coloc(d, cfg, P, pool, cls, W, names, nc, plates, pcol, summary):
    if nc < 2 or not any(f"r_{c}" in pool for c in names):
        return None
    ins = pool[cls == "inside"] if (cls == "inside").any() else pool
    fig, ax = _fig(1, nc + 1, w=3.2)
    M = np.full((nc, nc), np.nan)
    for i, a_ in enumerate(names):
        a = ax[0, i]
        g = ins[ins.ch == a_]
        for j, b_ in enumerate(names):
            if i == j or f"r_{b_}" not in g:
                continue
            _ecdf(a, g[f"r_{b_}"], _name(cfg, b_), _color(cfg, b_))
            M[i, j] = np.nanmedian(g[f"r_{b_}"]) if g[f"r_{b_}"].notna().any() else np.nan
        a.set_xlim(-1, 1)
        a.set_title(f"{_name(cfg, a_)} particles", fontsize=9, color=_color(cfg, a_))
        _style(a, "pixel Pearson r with …")
        a.legend(fontsize=6)
    a = ax[0, nc]
    im = a.imshow(M, vmin=-1, vmax=1, cmap="RdBu_r")
    a.set_xticks(range(nc), [_name(cfg, c) for c in names], fontsize=7, rotation=45)
    a.set_yticks(range(nc), [_name(cfg, c) for c in names], fontsize=7)
    for i in range(nc):
        for j in range(nc):
            if np.isfinite(M[i, j]):
                a.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7)
    a.set_xlabel("other channel", fontsize=8)
    a.set_ylabel("particle's channel", fontsize=8)
    a.figure.colorbar(im, ax=a, shrink=0.7)
    summary["coloc_median_r"] = {f"{names[i]}|{names[j]}": float(M[i, j]) for i in range(nc) for j in range(nc)
                                 if np.isfinite(M[i, j])}
    return _save(fig, d / "7_coloc.png", "Within-particle colocalisation: pixel-wise Pearson r of a particle's "
                                           "channel with each other channel ('inside' particles; right: median)")


def _density(ax, v, label=None, color=None, lw=1.5, bw=0.12, grid=None):
    """Density of log10 values as a line (never bars)."""
    from scipy.stats import gaussian_kde
    v = np.asarray(v, float)
    v = v[np.isfinite(v) & (v > 0)]
    if len(v) < 10 or np.ptp(np.log10(v)) == 0:
        return
    g = grid if grid is not None else np.linspace(np.log10(v.min()), np.log10(v.max()), 300)
    ax.plot(10 ** g, gaussian_kde(np.log10(v), bw_method=bw)(g), lw=lw, color=color, label=label)


def _p_cell_size(d, cfg, P, pool, cls, W, names, nc, plates, pcol, summary):
    """The size of one cell per plate x channel (the peak of the field objects' areas) and what the particles in the
    holes measure in cells (area / that size)."""
    f_path = d.parent / "field.parquet"
    ref_path = d.parent / "reference.csv"
    if not f_path.exists() or not ref_path.exists():
        return None
    F = pd.read_parquet(f_path)
    ref = pd.read_csv(ref_path, dtype={"plate_id": str})
    if "cell_um2" not in ref:
        return None
    fig, ax = _fig(2, nc, w=3.8, h=3.1)
    grid = np.linspace(np.log10(max(cfg.field_min_um2, 1)), 3.6, 300)
    summary["cell_um2"] = {}
    for j, ch in enumerate(names):
        a = ax[0, j]
        for pl in plates:
            g = F[(F.ch == ch) & (F.plate_id.astype(str) == str(pl))]
            r = ref[(ref.plate_id.astype(str) == str(pl)) & (ref.ch == ch)]
            one = float(r.cell_um2.iloc[0]) if len(r) else np.nan
            share2 = float((g.area_um2 >= 2 * one).mean()) if len(g) and np.isfinite(one) else np.nan
            _density(a, g.area_um2, f"{pl}: 1 cell = {one:.0f} µm², ≥2 cells {share2:.0%}", pcol[pl], grid=grid)
            if np.isfinite(one):
                a.axvline(one, color=pcol[pl], ls="--", lw=0.9)
                a.axvline(2 * one, color=pcol[pl], ls=":", lw=0.7)
            summary["cell_um2"][f"{pl}|{ch}"] = one
        a.set_xscale("log")
        a.set_title(f"{_name(cfg, ch)}: field objects (area density)", fontsize=9, color=_color(cfg, ch))
        _style(a, "area (µm², log)", "density of log10 area")
        a.legend(fontsize=6)
        b = ax[1, j]
        g = pool[(pool.ch == ch) & (pool.level == "whole")] if "level" in pool else pool[pool.ch == ch]
        if "cells_full" in g:
            for c in sorted(set(cls[g.index].dropna())):
                _density(b, g.cells_full[cls[g.index] == c], f"{c} (n={int((cls[g.index] == c).sum())})", None,
                         bw=0.2, grid=np.linspace(-2.5, 1.6, 300))
            b.axvline(1, color="grey", ls="--", lw=0.8)
            b.axvline(2, color="grey", ls=":", lw=0.7)
            b.set_xscale("log")
            b.legend(fontsize=6)
        b.set_title(f"{_name(cfg, ch)}: particles in the holes, by preview class", fontsize=9)
        _style(b, "size in cells (area / 1 cell, log)", "density")
    return _save(fig, d / "9_cell_size.png", "Size of one cell = the peak of the field objects' areas (dashed; dotted "
                                               "= 2 cells). Below: particles touching a hole measured in cells "
                                               "(whole object): ~1 one cell, < 0.3 debris-sized, ≥ 2 a clump")


def _p_clumps(d, cfg, res) -> Path | None:
    """10_clumps: the two clump scores of each well's best cell, per plate, with the thresholds."""
    calls = res.calls
    if calls is None or "body_cells" not in calls:
        return None
    C = calls[calls.status.astype(str) != "no cell"] if "status" in calls else calls
    plates = sorted(C.plate_id.astype(str).unique())
    pcol = {p: PLATE_COLORS[i % len(PLATE_COLORS)] for i, p in enumerate(plates)}
    units = res.run.get("clump_units") or {}
    fig, ax = _fig(1, 2, w=4.6, h=3.3)
    for pl in plates:
        g = C[C.plate_id.astype(str) == pl]
        n_fl = int(g.possible_clump.sum()) if "possible_clump" in g else 0
        _density(ax[0, 0], g.body_cells, f"{pl}: 1 cell = {units.get(pl, float('nan')):.0f} µm², flagged {n_fl}/{len(g)}",
                 pcol[pl], bw=0.2, grid=np.linspace(-1.3, 1.0, 300))
        _ecdf(ax[0, 1], g.others_cells, pl, pcol[pl])
    ax[0, 0].axvline(cfg.clump_body_x, color="#c0392b", lw=0.9)
    ax[0, 0].set_xscale("log")
    _style(ax[0, 0], "body / one patched cell (log)", "density")
    ax[0, 0].legend(fontsize=6)
    ax[0, 0].set_title("Body size in cells", fontsize=9)
    ax[0, 1].axvline(cfg.clump_others, color="#c0392b", lw=0.9)
    _style(ax[0, 1], "other cell-like objects inside the hole (cells)")
    ax[0, 1].legend(fontsize=6)
    ax[0, 1].set_title("Others in the hole", fontsize=9)
    n = int(C.possible_clump.sum()) if "possible_clump" in C else 0
    return _save(fig, d / "10_clumps.png", f"Possible clumps: {n} of {len(C)} wells with a cell (red: the thresholds; "
                                            "the call does not change, the wells are flagged)")


def classify_diagnostics(out_dir, info: dict | None = None, cfg=None) -> list:
    """After Classify: 8_identity (cells' brightness per identity channel with the calibration) and 10_clumps."""
    from .pipeline import Results
    res = Results(out_dir)
    cfg = cfg or res.config
    d = res.dir / "diagnostics"
    d.mkdir(exist_ok=True)
    C = res.cells
    if C is None or not len(C):
        return []
    extra = []
    try:
        f = _p_clumps(d, cfg, res)
        if f:
            extra.append(f)
    except Exception:                                   # a plot never fails Classify
        pass
    info = info or (json.loads((res.dir / "identity.json").read_text()) if (res.dir / "identity.json").exists() else {})
    chans = [c for c in res.run.get("identity_channels") or res.channels if f"n_{c}" in C]
    if not chans:
        return extra
    try:
        fig, ax = _fig(1, len(chans))
        for k, ch in enumerate(chans):
            a = ax[0, k]
            x = np.arcsinh(C[f"n_{ch}"].to_numpy(float) / 0.05)
            x = x[np.isfinite(x)]
            a.hist(x, bins=50, histtype="step", color=_color(cfg, ch) or "black", density=True, lw=1.4,
                   label=f"cells (n={len(x)})")
            m = info.get(ch) or {}
            if m.get("means"):
                xs = np.linspace(x.min(), x.max(), 300) if len(x) else np.linspace(0, 1, 2)
                for mu, sd, w in zip(m["means"], m["sds"], m["weights"]):
                    a.plot(xs, w * np.exp(-(xs - mu) ** 2 / (2 * sd ** 2)) / (sd * np.sqrt(2 * np.pi)), lw=1,
                           color="#555555", ls="--")
            if f"p_{ch}" in C:
                p = C[f"p_{ch}"].to_numpy(float)
                ok = np.isfinite(p) & np.isfinite(np.arcsinh(C[f"n_{ch}"].to_numpy(float) / 0.05))
                xx = np.arcsinh(C[f"n_{ch}"].to_numpy(float)[ok] / 0.05)
                if ok.any() and (p[ok] >= 0.5).any() and (p[ok] < 0.5).any():
                    a.axvline(0.5 * (xx[p[ok] < 0.5].max() + xx[p[ok] >= 0.5].min()), color="#c0392b", lw=0.8)
            a.set_title(_name(cfg, ch), fontsize=9, color=_color(cfg, ch))
            _style(a, "asinh(brightness / 0.05), typical-cell units", "density")
            a.legend(fontsize=6)
        return [_save(fig, d / "8_identity.png", "Cells' brightness per identity channel with the calibration's two "
                                                  "components (dashed) and the call's cut (red)")] + extra
    except Exception:
        return extra
