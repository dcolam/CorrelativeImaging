"""Cells per particle and possible clumps (reference.single_cell_area, features f_cells, clumps.py, cell pages)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("scipy")

from correlative_imaging.reconstruction.clumps import add_clump_scores, body_units, others_in_hole
from correlative_imaging.reconstruction.config import Config
from correlative_imaging.reconstruction.reference import single_cell_area


def test_single_cell_area_is_the_peak_not_pulled_up_by_clumps():
    rng = np.random.default_rng(0)
    single = 10 ** rng.normal(np.log10(150), 0.06, 3000)          # one cell = 150 um2
    clumps = 150 * rng.integers(2, 7, 1500) * 10 ** rng.normal(0, 0.06, 1500)
    debris = 10 ** rng.normal(np.log10(8), 0.15, 800)
    a = np.concatenate([single, clumps, debris])
    one = single_cell_area(a[a >= 30])
    assert 135 < one < 165
    assert np.median(a[a >= 30]) > 1.05 * one                      # the median would be pulled up
    assert np.isnan(single_cell_area([100, 120]))                    # too few objects


def _cells():
    return pd.DataFrame(dict(plate_id=["A", "A", "A", "B"], well_id=["w1", "w2", "w3", "w4"], cell_id=1,
                             body_um2=[100.0, 90.0, 260.0, 150.0], members=["a1", "b1", "c1", "d1"]))


def _calls():
    return pd.DataFrame(dict(plate_id=["A", "A", "A", "B"], well_id=["w1", "w2", "w3", "w4"], cell_id=1.0,
                             status="one cell", call="561nm"))


def test_unit_is_the_smaller_of_field_cell_and_typical_body():
    ref = pd.DataFrame(dict(plate_id=["A", "A", "B"], ch=["488nm", "561nm", "561nm"], cell_um2=[200.0, 220.0, 140.0]))
    u = body_units(_cells(), _calls(), ref, ["488nm", "561nm"])
    assert u["A"] == pytest.approx(100.0)                            # small hole: the typical body
    assert u["B"] == pytest.approx(140.0)                            # large hole: one field cell


def _particles():
    rows = [
        # w1: only its own seed
        ("A", "w1", "a1", "561nm", "cell", 0.5),
        # w2: another mScarlet cell mostly inside the hole (neighbour) + debris + the marker channel
        ("A", "w2", "b1", "561nm", "cell", 0.4), ("A", "w2", "b2", "561nm", "neighbour", 0.7),
        ("A", "w2", "b3", "561nm", "debris", 0.9), ("A", "w2", "b4", "640nm", "cell", 2.0),
        # w3: a double-positive other cell seen in two channels: counts once (largest channel)
        ("A", "w3", "c1", "488nm", "cell", 0.9), ("A", "w3", "c2", "488nm", "neighbour", 0.3),
        ("A", "w3", "c3", "561nm", "neighbour", 0.35),
    ]
    F = pd.DataFrame(rows, columns=["plate_id", "well_id", "oid", "ch", "pclass", "cells_in"])
    return F.assign(pool=True, level="whole")


def test_others_in_hole_counts_cell_like_objects_of_identity_channels_only():
    o = others_in_hole(_cells(), _particles(), ["488nm", "561nm"])
    assert list(o.round(2)) == [0.0, 0.7, 0.35, 0.0]


def test_flags_and_reasons_reach_the_well_call():
    ref = pd.DataFrame(dict(plate_id=["A", "B"], ch="561nm", cell_um2=[220.0, 140.0]))
    cells, calls, units = add_clump_scores(_cells(), _calls(), _particles(), ref, ["488nm", "561nm"], Config())
    assert list(cells.possible_clump) == [False, True, True, False]
    assert "others in the hole 0.7" in cells.clump_reason[1]
    assert "body 2.6 cells" in cells.clump_reason[2]
    assert list(calls.possible_clump) == [False, True, True, False]
    # re-detecting one well reuses the run's units
    one, _, _ = add_clump_scores(_cells().iloc[[2]], _calls().iloc[[2]], _particles(), ref, ["488nm", "561nm"],
                                 Config(), units=units)
    assert one.body_cells.iloc[0] == pytest.approx(2.6)


def test_old_results_without_cell_sizes_never_raise():
    F = _particles().drop(columns="cells_in")
    cells, calls, _ = add_clump_scores(_cells(), _calls(), F, pd.DataFrame(columns=["plate_id", "ch"]),
                                       ["561nm"], Config())
    assert cells.others_cells.isna().all() and "possible_clump" in calls
