"""Neighbour-aware reconstruction (reconstruction/reconstruct.py) on synthetic signal maps.

S is already in hole-noise units, so peak_h = 3 means "3 noise SDs". The cases:
  * two blobs with a deep valley between them split into two objects;
  * one noisy plateau stays one object (no over-splitting of a dim, flat patched cell);
  * a dim cell next to a bright neighbour: the boundary sits at the saddle, the dim cell does not grow into it;
  * with split_dips off the two blobs of the first case stay one object.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("skimage")

from correlative_imaging.reconstruction.config import Config
from correlative_imaging.reconstruction.reconstruct import reconstruct_well, saddles

N = 120
yy, xx = np.indices((N, N))


def blob(cy, cx, sigma, amp):
    return amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))


def disc(cy, cx, r):
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2


CFG = Config(grow_margin_um=30.0, grow_max_x_hole=10.0, cell_gap_um=0.0)


def two_blobs():
    # saddle ~ 7 noise SDs (above the growth floor, so the area is connected), peaks 20: saddle / peak ~ 0.35
    return blob(60, 45, 8, 20) + blob(60, 75, 8, 20), disc(60, 45, 4) | disc(60, 75, 4)


def test_two_blobs_with_valley_split():
    S, seed = two_blobs()
    hole = disc(60, 60, 35)
    assert S[60, 60] > CFG.grow_floor                      # connected above the floor: only the dip separates them
    cells, G = reconstruct_well(S, hole, 1.0, seed, np.zeros_like(seed), np.zeros_like(seed), CFG)
    assert len(cells) == 2
    assert sum(c["n_dips"] for c in cells) >= 2


def test_two_blobs_join_without_dip_splitting():
    S = blob(60, 50, 8, 20) + blob(60, 70, 8, 20)        # saddle ~ 0.9 of the peaks: no real dip
    hole = disc(60, 60, 35)
    seed = disc(60, 50, 4) | disc(60, 70, 4)
    cells, _ = reconstruct_well(S, hole, 1.0, seed, np.zeros_like(seed), np.zeros_like(seed), CFG)
    assert len(cells) == 1
    S2, seed2 = two_blobs()
    cfg = Config(**{**CFG.__dict__, "split_dips": False})
    cells2, _ = reconstruct_well(S2, hole, 1.0, seed2, np.zeros_like(seed2), np.zeros_like(seed2), cfg)
    assert len(cells2) == 1


def test_noisy_plateau_not_split():
    rng = np.random.default_rng(0)
    from scipy.ndimage import gaussian_filter
    S = 8.0 * disc(60, 60, 25) + gaussian_filter(rng.normal(0, 1, (N, N)), 1) * 1.0
    hole = disc(60, 60, 35)
    seed = disc(60, 60, 20)
    cells, _ = reconstruct_well(S, hole, 1.0, seed, np.zeros_like(seed), np.zeros_like(seed), CFG)
    assert len(cells) == 1
    assert cells[0]["body_px"] >= disc(60, 60, 20).sum()


def test_dim_cell_stops_at_saddle_with_bright_neighbour():
    S = blob(60, 45, 7, 6) + blob(60, 75, 7, 40)          # dim patched cell left, bright neighbour right
    hole = disc(60, 45, 14)
    seed = disc(60, 45, 4)
    nb = disc(60, 78, 6)                                   # the neighbour's own particle
    cells, G = reconstruct_well(S, hole, 1.0, seed, nb, np.zeros_like(seed), CFG)
    assert len(cells) == 1
    m = cells[0]["mask"]
    sad = saddles(np.where(xx < 60, 1, 2) * (S > 0), S)
    assert m[:, 62:].sum() == 0                           # never past the valley into the neighbour
    assert (G[:, 70:] < 0).any()                          # the neighbour keeps its own pixels
    assert sad                                             # (sanity: the helper finds the pass)


def test_debris_is_a_barrier():
    S = blob(60, 60, 10, 15)
    hole = disc(60, 60, 30)
    seed = disc(60, 55, 3)
    barrier = disc(60, 70, 4)
    cells, _ = reconstruct_well(S, hole, 1.0, seed, np.zeros_like(seed), barrier, CFG)
    assert not (cells[0]["mask"] & barrier).any()


def test_hand_cut_keeps_two_cells_apart():
    # one bright plateau (no dip) covering two touching seeds: without edits it is one cell; with the cut line as a
    # wall and the two seeds marked as pieces of one cut object it stays two cells, even with a gap join
    S = blob(60, 60, 14, 20)
    left, right = disc(60, 52, 6) & (xx < 60), disc(60, 68, 6) & (xx > 60)
    seed = left | right | (disc(60, 60, 6) & (xx == 60))
    hole = disc(60, 60, 30)
    cfg = Config(grow_margin_um=30.0, grow_max_x_hole=10.0, cell_gap_um=3.0)
    z = np.zeros_like(seed)
    cells, _ = reconstruct_well(S, hole, 1.0, seed, z, z, cfg)
    assert len(cells) == 1
    walls = (xx >= 59) & (xx <= 61)
    parts = np.where(left, 1, np.where(right, 2, 0)).astype(np.int32)
    cells, G = reconstruct_well(S, hole, 1.0, seed, z, z, cfg, walls=walls, parts=parts,
                                conflicts={frozenset((1, 2))})
    assert len(cells) == 2
    assert G[60, 52] != G[60, 68] and G[60, 52] > 0 and G[60, 68] > 0
    assert G[60, 60] > 0                     # wall pixels inside a seed go back to a cell
