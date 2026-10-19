"""Changing particles at Detect: automatic splits at signal dips, hand-drawn cuts, and merges.

Order inside detect_well, per label map (channel x threshold level):
  0 watershed    a particle picked for "Watershed" (one channel and level, by position) is split in two at its
                 deepest dip (or at the neck of its shape) by :func:`watershed_pieces`; its dividing lines are added to the hand cuts below, so they act like a
                 drawn cut (every channel, both levels) and are undone the same way.
  1 hand cuts        every cut line (frame coordinates, from the label library) is drawn 3 px wide and removed from
                     the mask, the mask is relabelled, and the cut pixels are given back to the nearest piece (areas do
                     not shrink). A cut applies to every channel and both threshold levels.
  2 split at dips    proposed for every particle touching the hole (and applied to all of them with `split_particles`,
                     or only to the particles whose proposal was accepted in the Train tab): each particle touching the hole and at most `split_max_x_hole` hole areas
                     is cut along the valleys between its significant peaks: h-maxima of the smoothed channel
                     (height `peak_h` x the hole-noise SD), watershed basins inside the particle, and basins whose
                     saddle is at least `split_dip_ratio` of the lower peak are joined again (no real dip = no split).
                     Pieces smaller than `min_object_px` go back to the neighbour they share the highest saddle with.
                     Signal for the ratios: smoothed image minus the hole background median.
  3 merges           a stored merge joins the two particles at two positions (same channel and level), but only when
                     they touch or came from the same object (undoing a split); otherwise it is skipped.

Every particle keeps `origin`, the label of the object it came from before 1-3, so splits can be drawn and labels
made on the unsplit object can be traced.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.draw import line
from skimage.measure import label
from skimage.morphology import h_maxima
from skimage.segmentation import watershed


def cut_mask(shape, cuts: list, offset=(0, 0), width: int = 3) -> np.ndarray:
    """cuts: list of polylines [[y, x], ...] in frame coordinates. Returns a bool mask in window coordinates."""
    m = np.zeros(shape, bool)
    for pts in cuts:
        pts = np.asarray(pts, float) - np.asarray(offset, float)
        for (y0, x0), (y1, x1) in zip(pts[:-1], pts[1:]):
            rr, cc = line(int(round(y0)), int(round(x0)), int(round(y1)), int(round(x1)))
            ok = (rr >= 0) & (rr < shape[0]) & (cc >= 0) & (cc < shape[1])
            m[rr[ok], cc[ok]] = True
    if width > 1 and m.any():
        m = ndi.binary_dilation(m, iterations=(width - 1) // 2)
    return m


def apply_cuts(L: np.ndarray, cut: np.ndarray) -> np.ndarray:
    fg = L > 0
    if not (cut & fg).any():
        return L
    L2 = label(fg & ~cut).astype(np.int32)
    back = fg & cut
    if L2.max():
        # give the cut pixels back to the nearest piece of the same original object
        _, (iy, ix) = ndi.distance_transform_edt(L2 == 0, return_indices=True)
        give = back & (L[iy, ix] == L)
        L2[give] = L2[iy, ix][give]
    return L2


def _saddles(B: np.ndarray, S: np.ndarray) -> dict:
    out = {}
    for la, lb, sa, sb in ((B[:, :-1], B[:, 1:], S[:, :-1], S[:, 1:]), (B[:-1, :], B[1:, :], S[:-1, :], S[1:, :])):
        m = (la != lb) & (la > 0) & (lb > 0)
        if not m.any():
            continue
        a, b, v = np.minimum(la[m], lb[m]), np.maximum(la[m], lb[m]), np.minimum(sa[m], sb[m])
        for x, y, s in zip(a, b, v):
            k = (int(x), int(y))
            if s > out.get(k, -np.inf):
                out[k] = float(s)
    return out


def split_object(mask: np.ndarray, S: np.ndarray, h: float, dip_ratio: float, min_px: int) -> np.ndarray:
    """Pieces of one object (labels 1..n inside `mask`), split only at real dips of S."""
    pk = h_maxima(np.where(mask, S, S[mask].min() if mask.any() else 0).astype(np.float64), h).astype(bool) & mask
    markers = label(pk)
    if markers.max() < 2:
        return mask.astype(np.int32)
    B = watershed(-S, markers, mask=mask)
    n = int(B.max())
    peak = np.asarray(ndi.maximum(S, B, np.arange(1, n + 1)), float)
    size = np.asarray(ndi.sum(mask, B, np.arange(1, n + 1)), float)
    sad = _saddles(B, S)
    parent = list(range(n + 1))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for (a, b), s in sorted(sad.items(), key=lambda t: -t[1]):
        low = max(min(peak[a - 1], peak[b - 1]), 1e-6)
        if s / low >= dip_ratio:
            parent[find(b)] = find(a)
    # crumbs: pieces below min_px join the neighbour with the highest saddle
    for _ in range(n):
        roots = np.array([find(i) for i in range(n + 1)])
        G = roots[B] * mask
        sizes = {r: int((G == r).sum()) for r in set(roots[1:])}
        small = [r for r, v in sizes.items() if v < min_px]
        if not small or len(sizes) < 2:
            break
        changed = False
        for r in small:
            best = max(((s, a if find(b) == r else b) for (a, b), s in sad.items()
                        if (find(a) == r) != (find(b) == r) and r in (find(a), find(b))), default=None)
            if best is not None:
                parent[r] = find(best[1])
                changed = True
        if not changed:
            break
    roots = np.array([find(i) for i in range(n + 1)])
    G = roots[B] * mask
    out = np.zeros(mask.shape, np.int32)
    for k, r in enumerate(sorted(set(np.unique(G)) - {0}), start=1):
        out[G == r] = k
    return out


def watershed_pieces(mask: np.ndarray, S: np.ndarray, h: float, min_px: int) -> np.ndarray:
    """On-demand split of one object into two, at its deepest dip: watershed basins of S at its significant peaks
    (h-maxima), small basins (< max(min_px, 10 % of the object)) first join the neighbour they share the highest
    saddle with, then basins join from the shallowest dip on until two pieces are left. Without two peaks, by shape:
    watershed of the distance map from its two thickest points (a neck between two round parts). Watershed a piece
    again to split it further."""
    pk = h_maxima(np.where(mask, S, S[mask].min() if mask.any() else 0).astype(np.float64), h).astype(bool) & mask
    markers = label(pk)
    if markers.max() >= 2:
        B = watershed(-S, markers, mask=mask)
        n = int(B.max())
        peak = np.asarray(ndi.maximum(S, B, np.arange(1, n + 1)), float)
        sad = _saddles(B, S)
        parent = list(range(n + 1))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def roots():
            r = np.array([find(i) for i in range(n + 1)])
            G = r[B] * mask
            return {int(x): int((G == x).sum()) for x in set(r[1:])}
        small = max(min_px, int(0.1 * mask.sum()))
        for _ in range(n):                                       # crumbs join their best neighbour
            sizes = roots()
            tiny = [x for x, v in sizes.items() if v < small]
            if not tiny or len(sizes) < 2:
                break
            r = min(tiny, key=lambda x: sizes[x])
            best = max(((v, b if find(a) == r else a) for (a, b), v in sad.items()
                        if (find(a) == r) != (find(b) == r)), default=None)
            if best is None:
                break
            parent[r] = find(best[1])
        # join from the shallowest dip (highest saddle / lower peak) on until two pieces are left
        for (a, b), v in sorted(sad.items(), key=lambda t: -t[1] / max(min(peak[t[0][0] - 1], peak[t[0][1] - 1]), 1e-6)):
            if len(roots()) <= 2:
                break
            if find(a) != find(b):
                parent[find(b)] = find(a)
        sizes = roots()
        if len(sizes) == 2 and min(sizes.values()) >= small:
            r = np.array([find(i) for i in range(n + 1)])
            G = r[B] * mask
            out = np.zeros(mask.shape, np.int32)
            for k, x in enumerate(sorted(sizes), start=1):
                out[G == x] = k
            return out
    from skimage.feature import peak_local_max
    dt = ndi.distance_transform_edt(mask)
    pk = peak_local_max(dt, min_distance=max(int(dt.max() * 0.5), 2), labels=mask.astype(int), num_peaks=2,
                        exclude_border=False)
    if len(pk) < 2:
        return mask.astype(np.int32)
    markers = np.zeros(mask.shape, np.int32)
    for i, (y, x) in enumerate(pk, start=1):
        markers[y, x] = i
    out = watershed(-dt, markers, mask=mask).astype(np.int32)
    sizes = np.bincount(out.ravel())[1:]
    return out if len(sizes) == 2 and sizes.min() >= max(min_px, int(0.1 * mask.sum())) else mask.astype(np.int32)


def watershed_lines(L: np.ndarray, S: np.ndarray, picks: list, h: float, min_px: int) -> tuple[np.ndarray, list]:
    """picks: (y, x) window positions. Returns (dividing-line mask for the cut list, number of pieces per pick)."""
    lines = np.zeros(L.shape, bool)
    n = []
    for y, x in picks:
        if not (0 <= y < L.shape[0] and 0 <= x < L.shape[1]) or not L[y, x]:
            n.append(0)
            continue
        sl = ndi.find_objects((L == L[y, x]).astype(np.uint8))[0]
        m = L[sl] == L[y, x]
        pieces = watershed_pieces(m, S[sl], h, min_px)
        n.append(int(pieces.max()))
        if pieces.max() >= 2:
            sub = lines[sl]
            sub |= ndi.binary_dilation(split_lines(pieces, m.astype(np.int32))) & m
    return lines, n


def auto_split(L: np.ndarray, S: np.ndarray, hole: np.ndarray, h: float, dip_ratio: float, min_px: int,
               max_px: int) -> np.ndarray:
    """Split the particles of label map L that touch the hole and are at most max_px big. Returns a new label map."""
    out = np.zeros_like(L)
    nxt = 1
    touching = set(np.unique(L[hole])) - {0}
    for lb, sl in enumerate(ndi.find_objects(L), start=1):
        if sl is None:
            continue
        m = L[sl] == lb
        if lb in touching and m.sum() <= max_px:
            pieces = split_object(m, S[sl], h, dip_ratio, min_px)
        else:
            pieces = m.astype(np.int32)
        k = int(pieces.max())
        sub = out[sl]
        sub[pieces > 0] = pieces[pieces > 0] + nxt - 1
        nxt += max(k, 1)
    return out


def apply_merges(L: np.ndarray, origin: np.ndarray, merges: list) -> tuple[np.ndarray, int]:
    """merges: list of ((y1, x1), (y2, x2)) in window coordinates. Joins the particles there if they touch or share
    an origin. Returns (map, number applied)."""
    L = L.copy()
    done = 0
    for (y1, x1), (y2, x2) in merges:
        H, W = L.shape
        if not (0 <= y1 < H and 0 <= x1 < W and 0 <= y2 < H and 0 <= x2 < W):
            continue
        a, b = int(L[int(y1), int(x1)]), int(L[int(y2), int(x2)])
        if not a or not b or a == b:
            continue
        touch = (ndi.binary_dilation(L == a) & (L == b)).any()
        same = origin[int(y1), int(x1)] == origin[int(y2), int(x2)] != 0
        if touch or same:
            L[L == b] = a
            done += 1
    return L, done


def split_lines(L: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Pixels where two pieces of the same origin object meet (for display)."""
    out = np.zeros(L.shape, bool)
    h = (L[:, :-1] != L[:, 1:]) & (L[:, :-1] > 0) & (L[:, 1:] > 0) & (origin[:, :-1] == origin[:, 1:])
    v = (L[:-1, :] != L[1:, :]) & (L[:-1, :] > 0) & (L[1:, :] > 0) & (origin[:-1, :] == origin[1:, :])
    out[:, :-1] |= h
    out[:-1, :] |= v
    return out
