"""Clumps from the whole-image particles (the "new" first pass, run(new=True)).

A clump engulfs the hole: one big whole-image particle reaches into the hole and far out of it, so what is in the
hole cannot be told apart. The hole cell outline alone misses it when the outline inside the hole stays small.

Features per hole (the largest whole-image particle reaching into the hole, any colour channel):
  log area inside the hole, log area outside the hole, its share of the hole, log hole cell outline / typical one.
Learnt from your inputs on holes only: clump question "several cells" / particle "several cells" (clump) vs clump
question "one cell" and the holes you gave a type in the Label tab (one cell). Logistic regression (balanced); the
cut is where the leave-one-out predictions separate your inputs best (balanced accuracy). Leave-one-out, not
leave-one-plate-out: on ROMK 20 of 24 "several cells" answers are on one plate (checked 2026-11-06: holding plates
out gave balanced accuracy 0.65, leave-one-out with the type labels 0.85). Needs >= 5 of each; otherwise no clump
rule (the report says so).
Your answer on a hole always wins over the rule.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_ANSWERS = 5
FEATS = ["log_in", "log_out", "hole_share", "log_cell_rel"]


def answers(A, R: pd.DataFrame | None = None) -> dict:
    """{(plate, well): 1 several cells / 0 one cell} from the hole questions (never the surrounding-cell size answers)
    and, as one cell, the holes you labelled with a type (labels.csv)."""
    from .labels import EXCLUDE_WORDS
    out = {}
    if R is not None and len(R):
        lab = R.your_label.fillna("").astype(str)
        t = R[(R.kind == "hole") & (lab != "") & ~R.not_cell.fillna(False).astype(bool)
              & ~lab.str.lower().str.startswith(("skip", "unsure")) & ~lab.str.lower().isin(EXCLUDE_WORDS)]
        out.update({(str(p), str(w)): 0 for p, w in zip(t.plate_id, t.well_id)})
    for r in A.get("clump").to_dict("records"):
        if r["answer"] == "several cells":
            out[(r["plate_id"], r["well_id"])] = 1
        elif r["answer"] == "one cell":
            out[(r["plate_id"], r["well_id"])] = 0
    for r in A.get("particle").to_dict("records"):
        if r["answer"] == "several cells":
            out[(r["plate_id"], r["well_id"])] = 1
    return out


def features(FP: pd.DataFrame, H: pd.DataFrame) -> pd.DataFrame:
    W = FP[(FP.level.astype(str) == "whole") & (FP.in_um2 > 0)].copy()
    W["out_um2"] = W.in_um2 * (1 / W.share.clip(lower=1e-3) - 1)
    W["tot"] = W.in_um2 + W.out_um2
    big = W.sort_values("tot").groupby(["plate_id", "well_id"]).tail(1).set_index(["plate_id", "well_id"])
    X = H[["plate_id", "well_id", "hole_um2", "cell_um2", "valid"]].copy()
    k = list(zip(X.plate_id, X.well_id))
    X["w_in"] = big.in_um2.reindex(k).fillna(0).to_numpy()
    X["w_out"] = big.out_um2.reindex(k).fillna(0).to_numpy()
    med = float(X.loc[X.valid & (X.cell_um2 > 0), "cell_um2"].median())
    X["log_in"] = np.log1p(X.w_in)
    X["log_out"] = np.log1p(X.w_out)
    X["hole_share"] = (X.w_in / X.hole_um2).clip(0, 1).fillna(0)
    X["log_cell_rel"] = np.log1p(X.cell_um2.fillna(0) / med) if med > 0 else 0.0
    return X


def rule(FP: pd.DataFrame, H: pd.DataFrame, ans: dict) -> tuple[pd.Series, dict]:
    """(clump: True / False per hole of H, NaN-free; info). Answers win."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X = features(FP, H)
    key = list(zip(X.plate_id, X.well_id))
    y = np.array([ans.get(k, np.nan) for k in key], float)
    lab = np.isfinite(y)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    info = dict(used=False, several=n1, one=n0, feats=FEATS)
    out = pd.Series(False, index=H.index)
    if n1 < MIN_ANSWERS or n0 < MIN_ANSWERS:
        info["why"] = f"needs {MIN_ANSWERS} 'several cells' and {MIN_ANSWERS} 'one cell' answers on holes"
    else:
        M = X[FEATS].to_numpy(float)
        mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", C=1.0, max_iter=1000))
        idx = np.nonzero(lab)[0]
        oof = np.full(len(X), np.nan)
        for i in idx:                                  # leave-one-out
            tr = idx[idx != i]
            oof[i] = mk().fit(M[tr], y[tr].astype(int)).predict_proba(M[[i]])[0, 1]
        ok = idx[np.isfinite(oof[idx])]
        cand = np.unique(oof[ok])
        bal = [(np.mean(oof[ok][y[ok] == 1] >= t) + np.mean(oof[ok][y[ok] == 0] < t)) / 2 for t in cand]
        cut = float(cand[int(np.argmax(bal))])
        p = mk().fit(M[idx], y[idx].astype(int)).predict_proba(M)[:, 1]
        out[:] = (p >= cut) & X.valid.to_numpy()
        H["p_clump"] = p
        info.update(used=True, cut=cut, balanced_accuracy=float(max(bal)),
                    validation="leave-one-out")
    out[lab & (y == 1)] = True                      # your answers win
    out[lab & (y == 0)] = False
    info["n_clump"] = int(out.sum())
    return out, info
