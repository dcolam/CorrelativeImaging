"""Particle edits at Detect (reconstruction/edits.py): automatic split at dips, hand cuts, merges."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("skimage")

from correlative_imaging.reconstruction.edits import (apply_cuts, apply_merges, auto_split, cut_mask, split_lines,
                                                      split_object)

N = 80
yy, xx = np.indices((N, N))


def blob(cy, cx, s, a):
    return a * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * s ** 2))


def test_two_cells_with_dip_split_one_plateau_not():
    S = blob(40, 28, 6, 20) + blob(40, 52, 6, 20)            # saddle ~ 0.1 of the peaks
    mask = S > 2
    assert split_object(mask, S, h=3, dip_ratio=0.6, min_px=20).max() == 2
    rng = np.random.default_rng(0)
    from scipy.ndimage import gaussian_filter
    P = 10.0 * ((yy - 40) ** 2 + (xx - 40) ** 2 <= 15 ** 2) + gaussian_filter(rng.normal(0, 1, (N, N)), 1)
    assert split_object(P > 5, P, h=3, dip_ratio=0.6, min_px=20).max() == 1


def test_auto_split_only_hole_particles_below_cap():
    S = blob(40, 28, 6, 20) + blob(40, 52, 6, 20)
    L = (S > 2).astype(np.int32)
    hole = (yy - 40) ** 2 + (xx - 40) ** 2 <= 20 ** 2
    assert auto_split(L, S, hole, 3, 0.6, 20, max_px=10_000).max() == 2
    assert auto_split(L, S, hole, 3, 0.6, 20, max_px=10).max() == 1          # bigger than the cap: left whole
    assert auto_split(L, S, np.zeros_like(hole), 3, 0.6, 20, 10_000).max() == 1   # not touching the hole


def test_hand_cut_splits_and_keeps_area():
    L = ((yy - 40) ** 2 + (xx - 40) ** 2 <= 20 ** 2).astype(np.int32)
    cut = cut_mask(L.shape, [[[10, 40], [70, 40]]], offset=(0, 0))
    L2 = apply_cuts(L, cut)
    assert L2.max() == 2
    assert (L2 > 0).sum() == (L > 0).sum()                    # cut pixels given back to a piece
    # a diagonal cut must also separate (8-connectivity)
    L3 = apply_cuts(L, cut_mask(L.shape, [[[15, 15], [65, 65]]]))
    assert L3.max() == 2


def test_merge_undoes_split_but_not_separate_particles():
    S = blob(40, 28, 6, 20) + blob(40, 52, 6, 20)
    L0 = (S > 2).astype(np.int32)
    L = auto_split(L0, S, np.ones_like(L0, bool), 3, 0.6, 20, 10_000)
    assert split_lines(L, L0).any()
    M, n = apply_merges(L, L0, [((40, 28), (40, 52))])
    assert n == 1 and M.max() == M[40, 28] == M[40, 52]
    far = np.zeros((N, N), np.int32)
    far[5:10, 5:10], far[60:70, 60:70] = 1, 2
    M2, n2 = apply_merges(far, far.copy(), [((7, 7), (65, 65))])
    assert n2 == 0


def test_watershed_on_demand_splits_at_a_neck_without_a_dip():
    from correlative_imaging.reconstruction.edits import watershed_lines, watershed_pieces
    yy, xx = np.indices((80, 120))
    m = ((yy - 40) ** 2 + (xx - 40) ** 2 <= 18 ** 2) | ((yy - 40) ** 2 + (xx - 76) ** 2 <= 18 ** 2)
    S = np.where(m, 10.0, 0.0)                                  # flat: no peak, so the shape decides
    pieces = watershed_pieces(m, S, 3.0, 20)
    assert pieces.max() == 2
    assert pieces[40, 40] != pieces[40, 76]
    L = m.astype(np.int32)
    lines, n = watershed_lines(L, S, [(40, 40)], 3.0, 20)
    assert n == [2] and lines.any() and not lines[40, 40] and not lines[40, 76]
    cut = apply_cuts(L, lines)
    assert cut.max() == 2 and (cut > 0).sum() == m.sum()      # split, area kept
    disc_only = ((yy - 40) ** 2 + (xx - 40) ** 2 <= 18 ** 2)
    assert watershed_pieces(disc_only, np.where(disc_only, 10.0, 0.0), 3.0, 20).max() == 1   # a round cell stays
