"""Plumbing tests for the per-channel LDA machinery (:mod:`.results.lda`).

Deliberately NOT tested against the rule-based classifier's own calls as labels
— that would be circular (the LDA would just relearn ``score≥thr AND occ≥min``
and post ~100%). Instead:

* synthetic labels with a *known* clean boundary → LDA must recover it (high κ);
* random labels → cross-validated κ must sit near 0 (no false confidence).

These use :func:`_report_channel` directly on synthetic feature frames, so they
need no result database and run anywhere.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from correlative_imaging.results import lda
from correlative_imaging.results.classify import ClassifierParams


def _synthetic_channel_df(n=60, n_plates=4, rule=None, seed=0):
    """A per-channel training frame (score, occ, y, plate, channel). ``rule`` maps
    (score, occ) → bool label; None → random labels."""
    rng = np.random.default_rng(seed)
    score = rng.uniform(0, 4, n)
    occ = rng.uniform(0, 1, n)
    plate = np.array([f"P{i % n_plates}" for i in range(n)])
    if rule is None:
        y = rng.integers(0, 2, n)
    else:
        y = np.array([int(rule(s, o)) for s, o in zip(score, occ)])
    return pd.DataFrame({"score": score, "occ": occ, "y": y,
                         "plate": plate, "channel": "488nm"})


def test_binary_metrics_counts():
    y_true = np.array([1, 1, 0, 0, 1])
    y_pred = np.array([1, 0, 0, 1, 1])
    m = lda._binary_metrics(y_true, y_pred)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 1)
    assert m["accuracy"] == pytest.approx(3 / 5)


def test_recovers_clean_boundary():
    # Label positive iff occupancy > 0.3 — a boundary in the feature plane.
    df = _synthetic_channel_df(rule=lambda s, o: o > 0.3)
    r = lda._report_channel(df, ClassifierParams())
    assert r.trainable
    assert r.kfold_lda["kappa"] > 0.7      # recovers the separation
    assert r.logo_lda["kappa"] > 0.6


def test_random_labels_kappa_near_zero():
    # No learnable structure → cross-validated agreement should be ~chance.
    kappas = []
    for seed in range(5):
        df = _synthetic_channel_df(rule=None, seed=seed)
        r = lda._report_channel(df, ClassifierParams())
        if r.trainable and r.kfold_lda is not None:
            kappas.append(r.kfold_lda["kappa"])
    assert kappas, "expected at least one trainable random fold"
    assert abs(np.mean(kappas)) < 0.25     # scattered around zero, no false signal


def test_not_trainable_single_class():
    df = _synthetic_channel_df(rule=lambda s, o: True)   # all positive
    r = lda._report_channel(df, ClassifierParams())
    assert not r.trainable
    assert "class" in r.reason
