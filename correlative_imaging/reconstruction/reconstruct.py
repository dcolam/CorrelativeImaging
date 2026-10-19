"""Reconstruct the cell(s) in the hole from classified particles, neighbour aware.

Signal  S = max over the identity channels of the lightly smoothed image in hole-noise units
        ((image - hole background median) / hole background SD). Marker channels do not shape bodies.

1 basins   Every significant peak of S (h-maxima, height `peak_h`) gets its own basin: watershed on -S over the
           pixels above `grow_floor`. A seed particle without such a peak (a dim, flat patched cell) still gets
           one marker at its brightest pixel, so it is never swallowed by a brighter neighbour's basin.
2 dips     Between adjacent basins the saddle is the highest pass between them. saddle / lower peak < `dip_ratio`
           = a real dip: two objects. Above it, the two basins are one object (no dip = no split; a noisy
           plateau has no significant peaks inside it, so it stays whole).
3 types    A basin is "cell" if it holds more seed pixels (particles with P(cell) + P(merged) >= p_seed, inside
           the hole) than neighbour pixels (particles classified neighbour, and every stringent object of the window
           that is not a seed), "neighbour" if the reverse, else free. Debris particles are barriers.
4 joining  Adjacent cell basins without a dip join; free basins join the typed basin they connect to without a dip
           (the highest saddle wins). A cell basin and a neighbour basin never join: the classification wins where
           there is no dip, the dip wins where there is one. Cell groups whose seeds come within `cell_gap_um` of
           each other also join (the same cell seen in two channels with a small gap).
5 growth   Each cell group keeps the pixels of its basins with S >= max(grow_frac x median S of its seed,
           grow_floor), within `grow_margin_um` of the hole and connected to its seed. Larger than
           `grow_max_x_hole` x the hole: the relative cut is raised in steps of 0.1.

With `split_dips` off, step 2 joins all adjacent basins of the same type (connected bright areas stay whole).

Hand edits are kept: the lines of hand cuts and applied splits (Detect's splits raster, 1 and 2) are walls no basin
crosses, and seed pieces of one edited object never end up in the same cell (none of the joins above unites them).
The wall pixels go back to the cell on their side afterwards.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.measure import label
from skimage.morphology import h_maxima
from skimage.segmentation import watershed

from .config import Config


class _UF:
    """Union-find; with `parts` (basin -> set of piece ids) and `conflicts` ({frozenset(id1, id2)}) a union that would
    put two conflicting pieces into one group is refused."""

    def __init__(self, n, parts: dict | None = None, conflicts: set | None = None):
        self.p = list(range(n))
        self.parts = {i: set(parts.get(i, ())) for i in range(n)} if parts else None
        self.conflicts = conflicts or set()

    def can(self, a, b) -> bool:
        if not self.parts or not self.conflicts:
            return True
        pa, pb = self.parts[self.find(a)], self.parts[self.find(b)]
        return not any(frozenset((x, y)) in self.conflicts for x in pa for y in pb)

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b) -> bool:
        a, b = self.find(a), self.find(b)
        if a == b:
            return True
        if not self.can(a, b):
            return False
        self.p[max(a, b)] = min(a, b)
        if self.parts:
            self.parts[min(a, b)] |= self.parts[max(a, b)]
        return True


def signal_map(sm: np.ndarray, names: list, body: list, hbg: dict) -> np.ndarray:
    z = [(sm[names.index(c)] - hbg[c][0]) / max(hbg[c][1], 1e-6) for c in body if c in names]
    return np.max(z, axis=0) if z else np.zeros(sm.shape[1:])


def saddles(B: np.ndarray, S: np.ndarray) -> dict:
    """{(a, b): saddle} for adjacent basins a < b (4-connectivity): the highest min(S) over touching pixel pairs."""
    out = {}
    for (la, lb, sa, sb) in ((B[:, :-1], B[:, 1:], S[:, :-1], S[:, 1:]), (B[:-1, :], B[1:, :], S[:-1, :], S[1:, :])):
        m = (la != lb) & (la > 0) & (lb > 0)
        if not m.any():
            continue
        a, b = np.minimum(la[m], lb[m]), np.maximum(la[m], lb[m])
        v = np.minimum(sa[m], sb[m])
        df = pd.DataFrame(dict(a=a, b=b, v=v)).groupby(["a", "b"]).v.max()
        for k, x in df.items():
            out[k] = max(out.get(k, -np.inf), x)
    return out


def basins(S: np.ndarray, floor: float, h: float, seeds: list[np.ndarray]):
    """Watershed basins of S above `floor`, one per h-maximum, plus one marker per seed lacking a peak."""
    allowed = S >= floor
    for s in seeds:
        allowed |= s
    pk = h_maxima(S.astype(np.float64), h).astype(bool) & allowed
    markers = label(pk)
    nxt = markers.max() + 1
    for s in seeds:
        for comp in range(1, (lab := label(s)).max() + 1):
            c = lab == comp
            if not (markers[c] > 0).any():
                idx = np.argmax(np.where(c, S, -np.inf))
                markers.flat[idx] = nxt
                nxt += 1
    B = watershed(-S, markers, mask=allowed)
    return B


def reconstruct_well(S: np.ndarray, hole: np.ndarray, px_um: float, seed_mask: np.ndarray, nb_mask: np.ndarray,
                     barrier: np.ndarray, cfg: Config, walls: np.ndarray | None = None, parts: np.ndarray | None = None,
                     conflicts: set | None = None):
    """Core, on arrays (testable): S signal (noise units), hole, seed / neighbour / barrier masks.
    walls: hand-edit lines (barriers whose pixels go back to the cell beside them); parts: piece id per seed pixel
    (0 = none) with `conflicts` {frozenset(id1, id2)}: pieces that must stay in different cells.
    Returns (cells: list of dicts with 'mask', group label image G (cells 1..n, neighbours -1..-m))."""
    if walls is not None and walls.any():
        barrier = barrier | walls
    else:
        walls = None
    S = np.where(barrier, -np.inf, S).astype(float)
    S_f = np.where(np.isfinite(S), S, cfg.grow_floor - 1)
    seed_comps = [label(seed_mask & ~barrier) == k for k in range(1, label(seed_mask & ~barrier).max() + 1)]
    B = basins(S_f, cfg.grow_floor, cfg.peak_h, seed_comps)
    B[barrier] = 0
    n = int(B.max())
    if n == 0:
        return [], np.zeros(S.shape, int)
    idx = np.arange(1, n + 1)
    peak = np.asarray(ndi.maximum(S_f, B, idx), float)
    n_seed = np.asarray(ndi.sum(seed_mask, B, idx), float)
    n_nb = np.asarray(ndi.sum(nb_mask & ~seed_mask, B, idx), float)
    typ = np.where((n_seed > 0) & (n_seed >= n_nb), 1, np.where(n_nb > 0, -1, 0))   # 1 cell, -1 neighbour, 0 free
    sad = saddles(B, S_f)
    joined = lambda a, b, s: (not cfg.split_dips) or s / max(min(peak[a - 1], peak[b - 1]), 1e-6) >= cfg.dip_ratio
    bparts = {}
    if parts is not None and conflicts:
        pm = (parts > 0) & (B > 0)
        if pm.any():
            cnt = pd.DataFrame(dict(b=B[pm], p=parts[pm])).value_counts()
            tot = cnt.groupby(level=0).sum()
            for (b, pid), c in cnt.items():
                if c >= 0.2 * tot[b]:                       # pieces that make up a real share of the basin's seed
                    bparts.setdefault(int(b), set()).add(int(pid))
    uf = _UF(n + 1, bparts, conflicts)
    n_dips = np.zeros(n + 1, int)
    # typed-typed of the same type
    for (a, b), s in sorted(sad.items(), key=lambda t: -t[1]):
        if typ[a - 1] == typ[b - 1] != 0:
            if not (joined(a, b, s) and uf.union(a, b)):          # a dip, or two pieces kept apart by hand
                n_dips[a] += 1
                n_dips[b] += 1
    # free basins: to the typed basin with the highest saddle among those reached without a dip, in rounds
    for _ in range(4):
        best = {}
        for (a, b), s in sad.items():
            for f, t in ((a, b), (b, a)):
                rt = uf.find(t)
                tt = typ[rt - 1] if typ[t - 1] == 0 else typ[t - 1]
                if typ[f - 1] == 0 and uf.find(f) == f and tt != 0 and joined(f, t, s) and uf.can(f, t):
                    if s > best.get(f, (-np.inf, 0))[0]:
                        best[f] = (s, t)
        if not best:
            break
        for f, (s, t) in best.items():
            if not uf.union(f, t):
                continue
            typ[f - 1] = typ[uf.find(t) - 1] if typ[uf.find(t) - 1] != 0 else typ[t - 1]
    root = np.array([uf.find(i) for i in range(n + 1)])
    G0 = root[B]
    G0[B == 0] = 0
    # cell groups also join across a small gap (same cell, two channels)
    cell_roots = sorted({root[i] for i in idx if typ[i - 1] == 1})
    gap = max(int(round(cfg.cell_gap_um / px_um)), 0)
    if gap and len(cell_roots) > 1:
        dil = {r: ndi.binary_dilation((G0 == r) & seed_mask, iterations=gap) for r in cell_roots}
        for i, a in enumerate(cell_roots):
            for b in cell_roots[i + 1:]:
                if (dil[a] & (G0 == b) & seed_mask).any():
                    uf.union(a, b)
        root = np.array([uf.find(i) for i in range(n + 1)])
        G0 = root[B]
        G0[B == 0] = 0

    margin = int(round(cfg.grow_margin_um / px_um))
    reach = ndi.binary_dilation(hole, iterations=margin) if margin > 0 else hole
    cap = int(cfg.grow_max_x_hole * hole.sum())
    nb_groups = sorted({root[i] for i in idx if typ[root[i] - 1] == -1 or typ[i - 1] == -1} - set(cell_roots))
    G = np.zeros(S.shape, int)
    for k, r in enumerate(nb_groups, start=1):
        G[G0 == r] = -k
    cells = []
    for r in sorted({root[c] for c in cell_roots}):
        region = G0 == r
        seed = region & seed_mask
        if not seed.any():
            continue
        med = float(np.median(S_f[seed]))
        frac = cfg.grow_frac
        while True:
            m = (region & reach & (S_f >= max(frac * med, cfg.grow_floor))) | seed
            lab = label(m)
            keep = np.unique(lab[seed])
            m = np.isin(lab, keep[keep > 0])
            if m.sum() <= cap or frac >= 1:
                break
            frac = min(frac + .1, 1.0)
        border = ndi.binary_dilation(m) & ~m
        members = [i for i in idx if root[i] == r]
        cells.append(dict(mask=m, seed_px=int(seed.sum()), body_px=int(m.sum()), body_um2=float(m.sum() * px_um ** 2),
                          grow_x=float(m.sum() / max(seed.sum(), 1)), body_in_hole=float((m & hole).sum() / max(m.sum(), 1)),
                          hole_cover=float((m & hole).sum() / max(hole.sum(), 1)),
                          touches_neighbour=bool((G[border] < 0).any()), n_basins=len(members),
                          n_dips=int(sum(n_dips[i] for i in members)), seed_median_S=med, cut_frac=frac,
                          peak_S=float(S_f[m].max())))
    for k, c in enumerate(cells, start=1):
        G[c["mask"]] = k
    if walls is not None and cells:
        # wall pixels inside a seed go back to the nearest cell (bodies do not keep a gap where the line was drawn)
        back = walls & seed_mask & (G == 0)
        if back.any():
            _, (iy, ix) = ndi.distance_transform_edt(G <= 0, return_indices=True)
            near = G[iy, ix]
            give = back & (near > 0)
            G[give] = near[give]
            for k, c in enumerate(cells, start=1):
                c["mask"] = G == k
                c["body_px"] = int(c["mask"].sum())
                c["body_um2"] = float(c["body_px"] * px_um ** 2)
    return cells, G


def reconstruct_from_particles(wd, sm: np.ndarray, P: pd.DataFrame, maps: dict, hbg: dict, body: list, cfg: Config):
    """Seeds and competitors from the classified particles of one well, then :func:`reconstruct_well`.
    P: this well's particles with p_<class> and pclass (pool) — context objects count as neighbours.
    Adds per cell: members (seed oids), p_cell (max over seeds), body_mean_<ch> (every channel)."""
    H, W = wd.hole.shape
    S = signal_map(sm, wd.names, body, hbg)
    seed = np.zeros((H, W), bool)
    nb = np.zeros((H, W), bool)
    barrier = np.zeros((H, W), bool)
    seed_rows = []
    for r in P.itertuples():
        L = maps.get((r.level, r.ch))
        if L is None:
            continue
        m = L == r.lab
        if r.pool:
            if r.ch in body and (r.p_cell + r.p_merged) >= cfg.p_seed and r.pclass in ("cell", "merged"):
                seed |= m & wd.hole
                seed_rows.append(r)
            elif r.pclass == "neighbour":
                nb |= m
            elif r.pclass == "debris":
                barrier |= m & ~seed
        elif getattr(r, "outer", False) == True:            # noqa: E712 (NaN on old rows)
            continue                                    # cells far out in the crop: beyond the window it works on
        elif r.level == "whole" or getattr(r, "n_parts", 1) > 1:   # context objects and pieces cut off outside
            nb |= m
    # hand edits: cut / applied-split lines are walls; pieces of one edited object must stay in different cells
    sp = maps.get(("splits", ""))
    walls = None
    if sp is not None and ((sp == 1) | (sp == 2)).any():
        walls = (sp == 2) | ndi.binary_dilation(sp == 1)          # split lines are 1 px: widen so 8-neighbours part
    parts = np.zeros((H, W), np.int32)
    conflicts = set()
    if walls is not None:
        by_origin = {}
        for i, r in enumerate(seed_rows, start=1):
            if getattr(r, "n_parts", 1) and getattr(r, "n_parts", 1) > 1 and isinstance(getattr(r, "origin", None), str):
                parts[(maps[(r.level, r.ch)] == r.lab) & wd.hole & (parts == 0)] = i
                by_origin.setdefault((r.level, r.ch, r.origin), []).append(i)
        for ids in by_origin.values():
            conflicts |= {frozenset((a, b)) for j, a in enumerate(ids) for b in ids[j + 1:]}
    # on the analysis window (+- work_R), as the model was made for; bodies and G are pasted back into the area
    b = wd.box_slices if hasattr(wd, "box_slices") else (slice(None), slice(None))
    cells, Gb = reconstruct_well(S[b], wd.hole[b], wd.px_um, seed[b], (nb & ~seed)[b], (barrier & ~seed)[b], cfg,
                                 walls=None if walls is None else walls[b], parts=parts[b], conflicts=conflicts)
    G = np.zeros((H, W), int)
    G[b] = Gb
    for c in cells:
        full = np.zeros((H, W), bool)
        full[b] = c["mask"]
        c["mask"] = full
    for k, c in enumerate(cells, start=1):
        mem = []
        for r in seed_rows:
            m = (maps[(r.level, r.ch)] == r.lab) & wd.hole
            if (G[m] == k).mean() > 0.5:
                mem.append(r)
        c["members"] = [r.oid for r in mem]
        c["p_cell"] = float(max((r.p_cell + r.p_merged for r in mem), default=np.nan))
        c["seed_chans"] = "+".join(sorted({r.ch for r in mem}))
        for j, ch in enumerate(wd.names):
            c[f"body_mean_{ch}"] = float(wd.chans[j][c["mask"]].mean())
    return cells, G, S
