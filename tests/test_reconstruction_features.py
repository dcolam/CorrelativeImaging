"""The reconstruction tool's detection + features work for any number of channels (1-4), and labels find their particle.

Synthetic wells: a hole with a cell in it and a neighbour reaching in, drawn into 1, 2, 3 or 4 channels. The same
feature columns must come out every time (other-channel summaries are NaN with a single channel), and a label
stored at a particle's position must be matched back to that particle.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("skimage")

from correlative_imaging.reconstruction.config import Config
from correlative_imaging.reconstruction.detect import detect_well
from correlative_imaging.reconstruction.features import FEATURES, build_features
from correlative_imaging.reconstruction.labels import match_labels
from correlative_imaging.reconstruction.reference import field_reference
from correlative_imaging.reconstruction.sources import WellData

N = 160
yy, xx = np.indices((N, N))


def _blob(cy, cx, s, a):
    return a * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * s ** 2))


def _well(n_ch: int, seed=0) -> WellData:
    rng = np.random.default_rng(seed)
    names = ["405nm", "488nm", "561nm", "640nm"][:n_ch]
    hole = (yy - 80) ** 2 + (xx - 80) ** 2 <= 25 ** 2
    chans = []
    for k in range(n_ch):
        img = rng.normal(5, 1, (N, N)) + 15 * hole                  # bright aperture
        img += _blob(80, 80, 8, 60 if k == 0 else 5)                  # patched cell, channel 0
        img += _blob(80, 118, 12, 200 if k == n_ch - 1 else 3)        # neighbour, last channel
        chans.append(img.astype(np.float32))
    chans = np.stack(chans)
    thr = {(lev, ch): (40.0 if lev == "whole" else 30.0) for lev in ("whole", "hole") for ch in names}
    bg = {ch: (5.0, 1.0) for ch in names}
    return WellData("P1", "B02", 0.325, chans, names, hole, thr, bg)


def _field(names):
    rng = np.random.default_rng(1)
    rows = [dict(plate_id="P1", well_id="B02", ch=ch, area_um2=a, mean=m, circularity=0.8, solidity=0.9,
                 eccentricity=0.5) for ch in names for a, m in zip(rng.uniform(40, 200, 50), rng.uniform(80, 300, 50))]
    return pd.DataFrame(rows)


def _features_for(n_ch):
    cfg = Config(context_R=2.5)
    wd = _well(n_ch)
    p, maps, row = detect_well(wd, cfg)
    W = pd.DataFrame([row])
    ref = field_reference(_field(wd.names), W, wd.names)
    F = build_features(p, W, ref, wd.names)
    return wd, p, W, F


def test_same_features_for_1_to_4_channels():
    cols = None
    for n in (1, 2, 3, 4):
        _, p, _, F = _features_for(n)
        assert F.pool.any(), f"{n} channels: no particle in the hole"
        assert set(FEATURES) <= set(F.columns)
        if cols is None:
            cols = [c for c in F.columns if c.startswith("f_")]
        assert [c for c in F.columns if c.startswith("f_")] == cols
        pool = F[F.pool]
        if n == 1:
            assert pool.f_oth_max.isna().all() and pool.f_flat.isna().all()
        else:
            assert pool.f_oth_max.notna().all()
        assert np.isfinite(pool.f_snr).all() and np.isfinite(pool.f_rel).all()


def test_neighbour_reaching_in_has_low_share():
    wd, p, W, F = _features_for(3)
    pool = F[F.pool & (F.level == "whole")]
    nb = pool[pool.ch == wd.names[-1]]
    cell = pool[pool.ch == wd.names[0]]
    assert len(nb) and len(cell)
    assert nb.share.max() < cell.share.min()


def test_label_matches_its_particle():
    wd, p, W, F = _features_for(2)
    target = F[F.pool].iloc[0]
    L = pd.DataFrame([dict(dataset="d", plate_id="P1", well_id="B02", channel=target.ch, level=target.level,
                           y=target.iy + 0.4, x=target.ix - 0.3, label="cell", orig_label=None, source="hand")])
    m = match_labels(L, F.assign(dataset="d"), W)
    assert list(m.oid) == [target.oid]


def test_detect_preview_marks_the_neighbour():
    """Without any model, the neighbour reaching into the hole is marked 'neighbour' and the cell 'inside';
    the well preview finds one object, dominated by the cell's channel."""
    from correlative_imaging.reconstruction.preview import provisional_classes, well_previews
    wd, p, W, F = _features_for(3)
    F["dclass"] = provisional_classes(F)
    pool = F[F.pool]
    assert set(pool[pool.ch == wd.names[-1]].dclass) == {"neighbour"}
    assert "inside" in set(pool[pool.ch == wd.names[0]].dclass)
    pv = well_previews(F, W, wd.names, Config()).iloc[0]
    assert pv.preview_status == "one object"
    assert pv.preview_dominant == wd.names[0]
