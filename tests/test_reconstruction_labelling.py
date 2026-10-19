"""Labelling loop pieces (reconstruction): colocalisation, the preview stand-in model, test-well freezing, the user
colour calibration and the "several cells" verdict."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from correlative_imaging.reconstruction.config import Config
from correlative_imaging.reconstruction.detect import _coloc
from correlative_imaging.reconstruction.identify import identify, well_calls
from correlative_imaging.reconstruction.labels import LabelStore
from correlative_imaging.reconstruction.preview import PreviewModel


def test_coloc_pearson_per_other_channel():
    rng = np.random.default_rng(0)
    a = rng.normal(size=200)
    v = np.vstack([a, 2 * a + 1, rng.normal(size=200)])
    r = _coloc(v, 0, ["A", "B", "C"])
    assert r["r_B"] == pytest.approx(1.0) or abs(r["r_B"] - 1) < 1e-9
    assert abs(r["r_C"]) < 0.3 and "r_A" not in r
    assert np.isnan(_coloc(v[:, :3], 0, ["A", "B", "C"])["r_B"])          # too few pixels


def test_preview_model_maps_provisional_classes():
    F = pd.DataFrame(dict(dclass=["inside", "neighbour", "noise", None], pool=[True, True, True, False]))
    M = PreviewModel().predict(F)
    assert list(M.pclass[:3]) == ["cell", "neighbour", "noise"] and pd.isna(M.pclass.iloc[3])
    assert M.p_cell.iloc[0] == 1 and M.p_merged.sum() == 0


def test_freeze_test_wells_takes_over_and_avoids_old_labels(tmp_path):
    old = LabelStore(tmp_path / "old.db")
    for k in range(10):
        old.set("ds", "P1", f"A{k:02d}", "c1", "any", 1, 1, "cell")
    old.seed_test_wells("ds", [("P1", "A00"), ("P1", "A01")])
    W = pd.DataFrame(dict(plate_id="P1", well_id=[f"A{k:02d}" for k in range(40)], preview_status="one object"))
    new = LabelStore(tmp_path / "new.db")
    pick = new.freeze_test_wells_from("ds", W, frac=0.25, other=tmp_path / "old.db")
    assert {("P1", "A00"), ("P1", "A01")} <= pick
    assert not any(("P1", f"A{k:02d}") in pick for k in range(2, 10))       # old training wells never test
    assert new.freeze_test_wells_from("ds", W) == pick                     # frozen once


def _cells(n=60, seed=0):
    rng = np.random.default_rng(seed)
    pos = rng.random(n) < 0.5
    body = np.where(pos, rng.normal(1.0, 0.2, n), rng.normal(0.02, 0.02, n))
    return pd.DataFrame(dict(plate_id="P1", well_id=[f"A{k:02d}" for k in range(n)], cell_id=1,
                             body_mean_c1=body, p_cell=1.0, body_in_hole=1.0)), pos


def _wells(n):
    return pd.DataFrame(dict(plate_id="P1", well_id=[f"A{k:02d}" for k in range(n)], bg_mean_c1=0.0, bg_sd_c1=0.01,
                             hbg_med_c1=0.0, hbg_sd_c1=0.01))


def test_user_colour_calibration_takes_over_with_enough_answers():
    C, pos = _cells()
    W = _wells(len(C))
    ref = pd.DataFrame(dict(plate_id=["P1"], ch=["c1"], ref_med=[1.0]))
    # the user calls positive only clearly bright cells (> 0.9): a stricter cut than the mixture's
    user = C[["plate_id", "well_id", "cell_id"]].assign(verdict="one cell",
                                                        colour=np.where(C.body_mean_c1 > 0.9, "c1", "none"))
    c0, info0 = identify(C, W, ref, ["c1"], Config())
    c1, info1 = identify(C, W, ref, ["c1"], Config(), user=user.head(40))
    assert "user" in info1["c1"] and "user" not in info0["c1"]
    mid = (C.body_mean_c1 > 0.5) & (C.body_mean_c1 < 0.8)
    if mid.any():                                                           # cells the user would call negative
        assert (c1.p_c1[mid.to_numpy()] < c0.p_c1[mid.to_numpy()]).all()
    few = user.head(4)
    assert "user" not in identify(C, W, ref, ["c1"], Config(), user=few)[1]["c1"]


def test_several_cells_verdict_makes_the_well_ambiguous():
    C, _ = _cells(30)
    W = _wells(len(C))
    ref = pd.DataFrame(dict(plate_id=["P1"], ch=["c1"], ref_med=[1.0]))
    c, _ = identify(C, W, ref, ["c1"], Config())
    c["verdict"] = None
    c.loc[0, "verdict"] = "several cells"
    calls = well_calls(c, W, ["c1"])
    r = calls[calls.well_id == "A00"].iloc[0]
    assert r.status == "several cells (by eye)" and r["call"] == "ambiguous"
