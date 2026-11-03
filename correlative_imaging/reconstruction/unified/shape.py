"""Cell or not, by shape and position (never brightness) - learnt within the dataset when the user gives "no cell"
examples.

Why: the aperture itself can glow, about as brightly as a dim patched cell (on ROMK, holes of empty wells reach an own
GFP signal of ~30 noise units while half of the patched GFP cells are below ~6). Brightness cannot separate them; the
shape of the object in the hole can (glow is a centred disc filling the hole, the same in every channel).

Examples (all from this dataset):
  no cell   every particle in the holes of the empty wells (experiment.json: empty_wells) and of the holes you marked
            "not a cell" (interventions/hole_review.csv)
  cell      particles of holes you marked "cell", and particles in the other holes clearly brighter (hole SNR) than
            95 % of the empty-well particles of their channel
Used twice: before the hole cell is grown (particles that are not cells - plugs, glow - are not seeds but barriers)
and in the call (a colour no brighter than an empty hole counts only in a cell-shaped hole).
Model: gradient-boosted trees on the particle measures of Detect (size, shape, position in the hole, peaks, pattern
across channels; brightness measures left out), predictions for every hole from a model that did not see its plate.
Needs >= 15 examples of each (your answers count one each); otherwise skipped (the report says so).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BRIGHT = ["f_snr", "f_hsnr", "f_rel", "f_relmax", "f_field_pct", "f_rel_ctx", "f_oth_max", "f_oth_mean"]


def particles(results: Path, prefix_of: dict, colour: list) -> pd.DataFrame:
    F = pd.read_parquet(Path(results) / "particles.parquet")
    F = F[F.pool.astype(bool) & F.ch.isin(colour)].copy()
    back = {v: k for k, v in prefix_of.items() if v}
    F["plate_id"] = F.plate_id.astype(str).map(lambda p: back.get(p, p))
    return F


def model(F: pd.DataFrame, H: pd.DataFrame, W: pd.DataFrame, own_cut: dict, reviewed: dict) -> tuple:
    """Returns (per hole best P(cell) as a Series indexed like H, info dict)."""
    feats = [c for c in F.columns if c.startswith("f_") and c not in BRIGHT and F[c].notna().any()]
    empty = set(zip(W.plate_id[W.user_empty], W.well_id[W.user_empty]))
    no = {k for k, v in reviewed.items() if v == "no"}
    yes = {k for k, v in reviewed.items() if v == "yes"}
    bright = set()
    for c, cut in own_cut.items():
        if np.isfinite(cut):
            b = H[H.valid & (H[f"own_{c}"] > cut)]
            bright |= {(p, w, c) for p, w in zip(b.plate_id, b.well_id)}
    key = list(zip(F.plate_id, F.well_id))
    y = np.full(len(F), np.nan)
    y[[k in empty or k in no for k in key]] = 0
    y[[k in yes for k in key]] = 1
    y[[(p, w, c) in bright and (p, w) not in empty and (p, w) not in no for p, w, c in zip(F.plate_id, F.well_id, F.ch)]] = 1
    n0, n1 = int((y == 0).sum()), int((y == 1).sum())
    info = dict(no_cell=n0, cell=n1, used=False, feats=feats)
    if n0 < 30 or n1 < 30:
        return pd.Series(np.nan, index=H.index), info
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    X = F[feats].to_numpy(float)
    lab = ~np.isnan(y)
    p = np.full(len(F), np.nan)
    plates = F.plate_id.unique()
    for pl in plates:
        tr = lab & (F.plate_id != pl).to_numpy()
        if len(plates) == 1:
            tr = lab
        if len(np.unique(y[tr])) < 2:
            continue
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20,
                                           l2_regularization=1.0, class_weight="balanced", random_state=0)
        m.fit(X[tr], y[tr].astype(int))
        te = (F.plate_id == pl).to_numpy()
        p[te] = m.predict_proba(X[te])[:, 1]
    F = F.assign(p_cell=p)
    k = lab & np.isfinite(p)
    info.update(used=True, auc=float(roc_auc_score(y[k], p[k])) if len(np.unique(y[k])) == 2 else np.nan)
    best = F.groupby(["plate_id", "well_id"]).p_cell.max()
    return pd.Series(best.reindex(list(zip(H.plate_id, H.well_id))).to_numpy(), index=H.index), info


def particle_model(results: Path, prefix_of: dict, colour: list, W: pd.DataFrame, reviewed: dict,
                   answers: dict | None = None) -> tuple:
    """Every hole particle with an out-of-fold P(cell) (by plate). (particles, info); P is NaN when there are too few
    examples."""
    F = particles(results, prefix_of, colour)
    feats = [c for c in F.columns if c.startswith("f_") and c not in BRIGHT and F[c].notna().any()]
    empty = set(zip(W.plate_id[W.user_empty], W.well_id[W.user_empty]))
    key = list(zip(F.plate_id, F.well_id))
    neg = np.array([k in empty or reviewed.get(k) == "no" for k in key])
    yes = np.array([reviewed.get(k) == "yes" for k in key])
    y = np.full(len(F), np.nan)
    y[neg] = 0
    # your answers on single particles (questions.py)
    a = np.array([answers.get((p, w, c, str(l), int(b)), np.nan) for p, w, c, l, b in
                  zip(F.plate_id, F.well_id, F.ch, F.level, F.lab)], float) if answers else np.full(len(F), np.nan)
    ref = neg | (a == 0)                    # what "no cell" looks like: empty / not-a-cell holes and your answers
    if ref.any():
        cut = {c: float(np.percentile(F.loc[ref & (F.ch == c).to_numpy(), "hsnr"], 95))
               if (ref & (F.ch == c).to_numpy()).sum() >= 5 else np.inf for c in colour}
        bright = (F.hsnr > F.ch.map(cut)).to_numpy() & ~ref
        y[bright] = 1
    y[yes & ~neg] = 1
    y = np.where(np.isfinite(a), a, y)      # your answers win
    F["p_cell"] = np.nan
    info = dict(no_cell=int((y == 0).sum()), cell=int((y == 1).sum()), used=False, feats=feats,
                answered=int(np.isfinite(a).sum()))
    if info["no_cell"] < 15 or info["cell"] < 15:
        return F, info
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    X = F[feats].to_numpy(float)
    lab = ~np.isnan(y)
    plates = F.plate_id.unique()
    p = np.full(len(F), np.nan)
    for pl in plates:
        tr = lab & ((F.plate_id != pl).to_numpy() if len(plates) > 1 else True)
        if len(np.unique(y[tr])) < 2:
            continue
        # leaves scale with the smaller class: a few dozen answers still fit a shallow model
        leaf = int(np.clip(min((y[tr] == 0).sum(), (y[tr] == 1).sum()) // 5, 3, 20))
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=leaf,
                                           l2_regularization=1.0, class_weight="balanced", random_state=0)
        m.fit(X[tr], y[tr].astype(int))
        te = (F.plate_id == pl).to_numpy()
        p[te] = m.predict_proba(X[te])[:, 1]
    F["p_cell"] = p
    k = lab & np.isfinite(p)
    info.update(used=True, auc=float(roc_auc_score(y[k], p[k])) if len(np.unique(y[k])) == 2 else np.nan,
                dropped=int((F.p_cell < 0.2).sum()))
    return F, info
