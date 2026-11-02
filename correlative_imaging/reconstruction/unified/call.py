"""Step 4, the call for every hole cell. Learnt from the hole cells of this dataset; the surrounding cells (step 2)
supply the bleed-through check, the cell size and the neighbours.

  valid cell   a grown cell (> 1/10 cell body, at least half inside the hole) or a particle mostly inside the hole that
               is not clearly a non-cell (shape); the hole not larger than 3x the dataset's median hole
  measure      per colour, of three: the grown cell's own signal (bleed-corrected, band background), the grown cell's
               hole signal (the hole's own background, removes aperture glow), or the particle signal (the most
               cell-like particle, hole background; best for small, dim cells). Taken: the one whose hole cells split
               most clearly into two groups
  positive     per colour channel, on that measure (noise units) of all valid hole cells:
               the noise is the spread of the values below 0, mirrored (a cell without that colour scatters around 0);
               q = mirrored false-discovery rate. Positive when q < 0.05 and the own signal >= the cut: the point where
               the two groups are equally likely when the channel has two groups, otherwise 5 noise units.
               Not positive when that channel's signal is glow from a neighbour (step 3)
  pattern      channel positivity, named as in the run ("FITC+ TX RED+", "none"); no cell types, no priority: those
               come only from your labels (interventions/labels.csv, step 5)
  marker       where its hole signal splits into two groups (never below 5 noise units); or above the 99th percentile
               of the hole cells with a declared negative pattern (experiment.json: marker_negative)
  flags        possible clump (the grown cell >= 2x the dataset's median hole cell), copy (a positive colour weaker than 3x
               the copy the other colours would put there, from the bleed-through measured on the surrounding cells),
               neighbour of another type within 3 um, glow from neighbour, check (q 0.05-0.2: borderline), dim (no
               brighter than an empty hole; with the shape model: cell by shape only). Halos are recognised only
               through the shape model, i.e. after you give "no cell" examples
  dim          a hole with no colour above its cut, whose strongest colour reaches half its cut and at least twice the
               next colour (or is above the cut but no brighter than an empty hole): "FITC? (dim)" etc., for review
  your marks   the channels you saw positive in holes (labels.csv, positive_channels) set each channel's cut: the point
               that best separates the holes where you saw it from those where you did not or saw no cell (5 of each;
               the cut can move between 0.3x and 3x the learnt one)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .field import positivity, two_groups


def qvalues(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, float)
    q = np.ones(len(z))
    pos = np.nonzero(z > 0)[0]
    if not len(pos):
        return q
    order = pos[np.argsort(-z[pos])]
    neg = np.sort(-z[z < 0])
    called = np.arange(1, len(order) + 1)
    false = len(neg) - np.searchsorted(neg, z[order], side="left")
    fdr = np.minimum(false / called, 1.0)
    q[order] = np.minimum.accumulate(fdr[::-1])[::-1]
    return q


def particle_signal(FP: pd.DataFrame, results, prefix_of: dict, colour: list) -> pd.DataFrame:
    """Per hole, the most cell-like particle (highest P(cell) when the shape model ran, else the brightest particle
    lying mostly inside): its mean in the hole minus the hole's own background, / the image noise, per colour."""
    from pathlib import Path
    Wr = pd.read_csv(Path(results) / "wells.csv", dtype={"plate_id": str, "well_id": str})
    back = {v: k for k, v in (prefix_of or {}).items() if v}
    Wr["plate_id"] = Wr.plate_id.map(lambda p: back.get(p, p))
    F = FP.merge(Wr, on=["plate_id", "well_id"], suffixes=("", "_w"))       # every particle touching the hole
    rank = F.p_cell.fillna(0) * 1e6 + F.hsnr.fillna(0) if F.p_cell.notna().any() else F.hsnr.fillna(0)
    F = F.assign(_r=rank).sort_values("_r", ascending=False).drop_duplicates(["plate_id", "well_id"])
    out = F[["plate_id", "well_id"]].copy()
    out["part_p"] = F.p_cell.to_numpy()
    out["part_um2"] = F.in_um2.to_numpy()
    for c in colour:
        out[f"part_{c}"] = ((F[f"in_{c}"] - F[f"hbg_med_{c}"]) / F[f"bg_sd_{c}"]).to_numpy()
    return out


MARKS_MIN_ACC = 0.75     # your channel marks move a cut only when they separate this well (balanced accuracy)


def call(M: pd.DataFrame, W: pd.DataFrame, F: dict, cell_size: dict, nb: pd.DataFrame, e: dict, results=None,
         prefix_of=None, reviewed=None, FP=None, chan_marks=None, clump_marks=None) -> tuple:
    colour, marker, nm, K = F["colour"], F["marker"], F["nm"], F["K"]
    H = M[M.error == ""].copy()
    H = H.merge(W[["plate_id", "image_id", "status"]], on=["plate_id", "image_id"], how="left")
    one = float(np.nanmedian([cell_size[c] for c in colour]))
    H["cells_in_body"] = H.cell_um2 / one
    med_hole = H.hole_um2.median()
    frag = 0.1 * one                    # size does not separate a hole cell from an empty hole: only specks are left out
    if FP is not None and results is not None:
        H = H.merge(particle_signal(FP, results, prefix_of, colour), on=["plate_id", "well_id"], how="left")
    part_ok = H.part_um2.notna() & ((H.part_p >= 0.2) | H.part_p.isna()) if "part_um2" in H else False
    grown_ok = (H.cell_um2 > frag) & (H.cell_in_hole >= 0.5)
    H["valid"] = (grown_ok | part_ok) & (H.hole_um2 <= 3 * med_hole)
    H["status_cell"] = np.select(
        [H.hole_um2 > 3 * med_hole, H.valid, H.cell_um2 <= 0, H.cell_um2 <= frag, H.cell_in_hole < 0.5],
        ["hole detection failed", "cell", "no cell", "speck (< 1/10 cell)", "mostly outside the hole"], "no cell")
    empty = H[[(p, w) in set(zip(W.plate_id[W.user_empty], W.well_id[W.user_empty]))
               for p, w in zip(H.plate_id, H.well_id)]]
    V = H.valid.to_numpy()
    cal = {}
    emp_mask = np.array([(p, w) in set(zip(empty.plate_id, empty.well_id)) for p, w in zip(H.plate_id, H.well_id)])
    bright_empty = {}
    measures = {}                                   # per colour: every measurement with its cut (your marks pick one)
    for c in colour:
        # three measurements of the same colour: the grown cell's own signal (bleed-corrected, band background), its
        # hole signal (the hole's own background, removes aperture glow), the most cell-like particle's signal. Each
        # gets its own noise, cut and q; they are ranked by how clearly their hole cells split into two groups, and
        # every hole uses the best-ranked one it has (a hole where nothing grew still has its particle)
        cand = {"own signal": H[f"own_{c}"].to_numpy(float)}
        if f"hsig_{c}" in H:
            cand["hole signal"] = H[f"hsig_{c}"].to_numpy(float)
        if f"part_{c}" in H:
            cand["particle signal"] = H[f"part_{c}"].to_numpy(float)
        res = {}
        for k, v in cand.items():
            ok = V & np.isfinite(v)
            if ok.sum() < 50:
                continue
            neg = v[ok]
            # noise from the values below 0, mirrored; a colour in nearly every cell has none: measures are in noise
            # units already, so 1
            sd = float(np.median(-neg[neg < 0]) / 0.6745) if (neg < 0).sum() >= 20 else 1.0
            q = np.ones(len(H))
            q[ok] = qvalues(v[ok] / sd)
            cut, D = two_groups(v[ok])
            floor = max(5.0, cut) if (cut is not None and D >= 2) else 5.0
            xe = v[emp_mask & np.isfinite(v)]
            ce = float(np.percentile(xe, 99)) if len(xe) >= 20 else np.nan
            res[k] = dict(v=v, ok=ok, sd=sd, q=q, cut=floor, D=D, ce=ce,
                          how="two groups" if (cut is not None and D >= 2) else "noise (one group)")
        order = sorted(res, key=lambda k: -res[k]["D"])
        measures[c] = res
        used = np.array([""] * len(H), dtype=object)
        xq, xr, br = np.ones(len(H)), np.full(len(H), np.nan), np.zeros(len(H), bool)
        for k in order[::-1]:                          # best-ranked last, so it overwrites where it exists
            r = res[k]
            m = r["ok"]
            used[m], xq[m] = k, r["q"][m]
            xr[m] = r["v"][m] / r["cut"]               # in units of the cut: >= 1 = above it
            br[m] = (r["v"][m] >= r["ce"]) if np.isfinite(r["ce"]) else True
        glow = H[f"glow_{c}"].astype(bool).to_numpy()
        H[f"x_{c}"], H[f"q_{c}"], H[f"measure_{c}"] = xr, xq, used
        H[f"pos_{c}"] = V & (xq < 0.05) & (xr >= 1) & ~glow
        H[f"check_{c}"] = V & (xq >= 0.05) & (xq < 0.2)
        H[f"glowonly_{c}"] = V & (xq < 0.05) & (xr >= 1) & glow
        bright_empty[c] = br
        best = order[0] if order else ""
        cal[c] = dict(noise_sd=res[best]["sd"] if best else np.nan, cut=res[best]["cut"] if best else np.nan,
                      separation=res[best]["D"] if best else 0.0, how=res[best]["how"] if best else "",
                      empty_cut=res[best]["ce"] if best else np.nan, n_empty=int(emp_mask.sum()), measure=best,
                      separations={k: round(res[k]["D"], 2) for k in order},
                      used={k: int((used == k).sum()) for k in order})
    # your marks (labels.csv, holes): per channel, the measurement and cut that best separate the holes where you saw
    # that channel positive from those where you saw it negative or no cell. The learnt measurement is the one whose
    # hole cells split most clearly into two groups; it need not be the one that sees what you see (e.g. bleed-through
    # from a bright channel lifts the particle signal but not the bleed-corrected own signal)
    reviewed, chan_marks = reviewed or {}, chan_marks or {}
    rkey = list(zip(H.plate_id, H.well_id))
    for c in colour:
        x = H[f"x_{c}"].to_numpy(float)
        seen = np.array([chan_marks.get(k, {}).get(c) is True for k in rkey])
        unseen = np.array([chan_marks.get(k, {}).get(c) is False or reviewed.get(k) == "no" for k in rkey])
        yes, no = seen & np.isfinite(x), unseen & np.isfinite(x)
        cal[c]["marked"] = dict(positive=int(yes.sum()), negative=int(no.sum()))
        if yes.sum() < 5 or no.sum() < 5:
            continue
        now = (np.mean(x[yes] >= 1) + np.mean(x[no] < 1)) / 2
        best = None                                  # (balanced accuracy, measure, factor)
        for k, r in measures.get(c, {}).items():
            xk = r["v"] / r["cut"]
            y, n = seen & r["ok"], unseen & r["ok"]
            if y.sum() < 5 or n.sum() < 5:
                continue
            cand = np.unique(np.clip(np.concatenate([xk[y], xk[n]]), 0.3, 3.0))
            acc = [(np.mean(xk[y] >= t) + np.mean(xk[n] < t)) / 2 for t in cand]
            i = int(np.argmax(acc))
            if best is None or acc[i] > best[0]:
                best = (float(acc[i]), k, float(cand[i]))
        if best is None:
            continue
        acc, k, t = best
        info = dict(measure=k, factor=round(t, 2), balanced_accuracy=round(acc, 2), before=round(float(now), 2),
                    measure_before=cal[c].get("measure", ""))
        # used only where your marks clearly separate and better than the learnt call; otherwise the measurements cannot
        # see what you see, and moving the cut would only shift the error
        if acc >= MARKS_MIN_ACC and acc - now >= 0.05:
            r = measures[c][k]
            xk = r["v"] / (r["cut"] * t)
            x = np.where(r["ok"], xk, x)
            H[f"x_{c}"] = x
            H[f"measure_{c}"] = np.where(r["ok"], k, H[f"measure_{c}"])
            glow = H[f"glow_{c}"].astype(bool)
            H[f"pos_{c}"] = H.valid & (H[f"x_{c}"] >= 1) & ~glow
            cal[c]["cut_from_marks"] = info
        else:
            cal[c]["marks_not_used"] = info
    # holes without a cell (your empty wells) and the shape model: a colour counts when it is brighter than any empty
    # hole, or when the shape model says the hole holds a cell
    H["p_shape"] = np.nan
    shape_info = dict(used=False)
    if results is not None and (len(empty) or reviewed):
        from . import shape
        Fp = shape.particles(results, prefix_of or {}, colour)
        H["p_shape"], shape_info = shape.model(Fp, H, W, {c: cal[c]["empty_cut"] for c in colour},
                                               {k: "no" if v == "no" else "yes" for k, v in reviewed.items()})
    for c in colour:
        ce = cal[c]["empty_cut"]
        if np.isfinite(ce):
            bright = pd.Series(bright_empty[c], index=H.index)
            by_shape = (H.p_shape >= 0.5) & ~bright if shape_info.get("used") else pd.Series(False, index=H.index)
            H[f"shape_only_{c}"] = H[f"pos_{c}"] & by_shape
            H[f"dim_unsure_{c}"] = H[f"pos_{c}"] & ~bright & ~by_shape
            H[f"pos_{c}"] = H[f"pos_{c}"] & (bright | by_shape)
        else:
            H[f"dim_unsure_{c}"] = False
            H[f"shape_only_{c}"] = False
        cal[c]["n_pos"] = int(H[f"pos_{c}"].sum())
    cal["_shape"] = shape_info
    # copies: a positive colour weaker than 3 x the copy the other colours would put there
    for j, b in enumerate(colour):
        copy = sum(K[i, j] * H[f"snr_{a}"].clip(lower=0).fillna(0) for i, a in enumerate(colour) if i != j)
        H[f"copy_{b}"] = H[f"pos_{b}"] & (H[f"snr_{b}"] < 3 * copy)
    H["colours"] = positivity(H, colour, nm)
    # dim: no colour above its cut, but one colour clearly leading (or above the cut and only blocked by the empty holes)
    xs = H[[f"x_{c}" for c in colour]].fillna(-np.inf).to_numpy(float)
    order_ = np.argsort(-xs, axis=1)
    best, second = xs[np.arange(len(H)), order_[:, 0]], (xs[np.arange(len(H)), order_[:, 1]] if len(colour) > 1
                                                         else np.full(len(H), -np.inf))
    lean = np.array([nm[colour[i]] for i in order_[:, 0]], dtype=object)
    blocked = np.column_stack([H[f"dim_unsure_{c}"].to_numpy(bool) for c in colour]).any(axis=1)
    clear_lead = (best >= 0.5) & (second < 0.5 * best)
    dim = H.valid.to_numpy() & (H.colours == "none").to_numpy() & (clear_lead | blocked)
    H["dim"] = dim
    for mk in marker:
        neg_t = (e.get("marker_negative") or {}).get(nm[mk])
        # the marker by its hole signal (the residual would remove a marker that sits on the same cells as a colour)
        mcol = f"hsig_{mk}" if f"hsig_{mk}" in H else f"own_{mk}"
        ref = H[H.valid & (H.colours == neg_t)][mcol].dropna() if neg_t else pd.Series(dtype=float)
        if len(ref) >= 20:
            cut, how = float(np.percentile(ref, 99)), f"99th percentile of {len(ref)} hole cells {neg_t}"
        else:
            g, D = two_groups(H.loc[H.valid, mcol].dropna().to_numpy(float))
            cut = max(5.0, g) if (g is not None and D >= 2) else 5.0
            how = f"two groups (separation {D:.1f})" if (g is not None and D >= 2) else "5 noise units (one group)"
        H[f"pos_{mk}"] = H.valid & (H[mcol] >= cut)
        cal[mk] = dict(cut=cut, how=how, n_pos=int(np.sum(H[f"pos_{mk}"])))
    # channel positivity, named as in the run; no cell types (those come only from your labels)
    H["pattern"] = positivity(H, colour + marker, nm)
    H.loc[dim, "pattern"] = [f"{n}? (dim)" for n in lean[dim]]
    H.loc[~H.valid, "pattern"] = H.loc[~H.valid, "status_cell"]
    H["cls"] = H.pattern
    # flags
    H = H.merge(nb, on=["plate_id", "image_id"], how="left") if len(nb) else H.assign(nb_nearest_um=np.nan,
                                                                                         nb_nearest_cls=None)
    med_cell = float(H.loc[H.valid & (H.cell_um2 > 0), "cell_um2"].median())   # the dataset's typical hole cell
    # the clump rule: 2x the typical hole cell, or where your one / several answers separate best (3 of each)
    clump_x = 2.0
    if clump_marks:
        r_ = H.cell_um2 / med_cell
        m_ = [clump_marks.get((p, w)) for p, w in zip(H.plate_id, H.well_id)]
        one = r_[[v == "one cell" for v in m_]].dropna().to_numpy()
        sev = r_[[v == "several cells" for v in m_]].dropna().to_numpy()
        if len(one) >= 3 and len(sev) >= 3:
            cand = np.unique(np.concatenate([one, sev]))
            acc = [(np.mean(sev >= t) + np.mean(one < t)) / 2 for t in cand]
            clump_x = float(cand[int(np.argmax(acc))])
    cal["_clump"] = dict(factor=clump_x, answers=len(clump_marks or {}))
    flags = []
    for r in H.to_dict("records"):                 # dict rows: channel names may hold spaces ("TX RED")
        f = []
        if r["valid"]:
            if r["cell_um2"] >= clump_x * med_cell:
                f.append("possible clump")
            if any(r[f"copy_{c}"] for c in colour):
                f.append("copy")
            nbc = r.get("nb_nearest_cls")
            if isinstance(nbc, str) and r.get("nb_nearest_um", np.inf) <= 3 and nbc != r["colours"]:
                f.append("neighbour of another type")
            if any(r[f"check_{c}"] for c in colour):
                f.append("check (borderline)")
            if any(r[f"glowonly_{c}"] for c in colour):
                f.append("glow from neighbour")
            if any(r.get(f"shape_only_{c}", False) for c in colour):
                f.append("dim: cell by shape only")
            if any(r.get(f"dim_unsure_{c}", False) for c in colour):
                f.append("dim: no brighter than an empty hole")
            if r["dim"]:
                f.append("dim: below the cut, check")
        if isinstance(r.get("status"), str) and r["status"].startswith("few cells"):
            f.append("well: few cells")
        flags.append("; ".join(f))
    H["flags"] = flags
    H["confidence"] = np.where(~H.valid, "", np.where(H["flags"] == "", "clear", "see flags"))
    return H, cal
