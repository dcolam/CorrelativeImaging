"""What to label next, in tiers, so a few hundred decisions go where they help most.

Every particle in a hole gets a suggested class and a tier:

  auto     the suggestion is almost surely right (no human needed): with a model, P >= 0.97 and the Detect preview
           agrees; without one, the colour-blind rules of learn.weak_labels (no signal; a complete neighbour lying
           mostly outside; small debris bright in every channel). Never stored as labels; the "+rules" training arms
           use the rules as weak labels anyway.
  confirm  probably right: shown with the suggestion, one key to accept or correct.
  ask      unsure: the model and the preview disagree, or the model's top class is below 0.7, or (no model) the
           particle sits near a cut of the preview's rules. These carry the most information per click.

A page mixes the tiers (default half ask, 30 % confirm, 20 % random from all tiers, so the acceptance rate and the
training set are not biased toward the hard cases only) and spreads them over wells, plates and channels.

Test wells (the frozen, held-out set) get their own queue: every object of the well in turn, shown blind (no
suggestion, no model output), so the score of any model on them, the old one included, is fair.

One tile per physical object: the lenient piece in the hole, or a stringent particle no lenient piece covers. Its
label is stored with level "any" (it applies to both threshold levels, as matched by position).

Cells (after Classify, with the preview or a model): one tile per reconstructed cell, suggested colour = the call;
tier ask when the call is uncertain, several seeds or basins went into the cell, or a channel's probability sits
between 0.2 and 0.8.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .labels import CLASSES

TIERS = ["ask", "confirm", "auto"]
MIX = {"ask": 0.5, "confirm": 0.3, "random": 0.2}


def objects(P: pd.DataFrame) -> pd.DataFrame:
    """One row per physical object in the hole: lenient pieces + stringent particles no lenient piece covers."""
    pool = P[P.pool.astype(bool)]
    covered = set(pool.loc[pool.level == "hole", "parent"].dropna())
    return pool[(pool.level == "hole") | ~pool.oid.isin(covered)]


def matched(res, store) -> pd.DataFrame:
    """The dataset's particle labels matched to the open results (shared by a page and the progress)."""
    from .labels import match_labels
    L = store.frame(res.dataset)
    return match_labels(L, res.particles, res.wells)


def _labelled_oids(res, store, m: pd.DataFrame | None = None) -> set:
    """Objects already labelled, or marked unsure (neither is shown again)."""
    from .labels import match_labels
    m = matched(res, store) if m is None else m
    out = set(zip(m.plate_id, m.well_id, m.oid)) if len(m) else set()
    U = store.unsure_frame(res.dataset)
    if len(U):
        u = match_labels(U, res.particles, res.wells)
        out |= set(zip(u.plate_id, u.well_id, u.oid))
    return out


def particle_tiers(res, model=None) -> pd.DataFrame:
    """Suggested class, confidence and tier for every object in the holes (see module doc)."""
    from .learn import weak_labels
    F = res.particles
    O = objects(F).copy()
    prev = O["dclass"].map({"inside": "cell"}).fillna(O["dclass"]) if "dclass" in O else pd.Series(None, index=O.index)
    use_model = model is not None and getattr(model, "spec", {}).get("arm") != "preview"
    if use_model:
        M = model.predict(O)
        P = M[[f"p_{c}" for c in CLASSES]].to_numpy(float)
        sug = M.pclass
        conf = P.max(1)
        agree = (sug == prev).to_numpy()
        tier = np.where((conf >= 0.97) & agree, "auto", np.where(~agree | (conf < 0.7), "ask", "confirm"))
    else:
        sug = prev
        weak = weak_labels(O)
        snr, rel = O.snr.to_numpy(float), O.rel.to_numpy(float)
        share = O.get("share", pd.Series(np.nan, index=O.index)).to_numpy(float)
        near = ((snr > 2) & (snr < 5)) | ((rel > 0.03) & (rel < 0.08)) | ((share > 0.35) & (share < 0.65))
        tier = np.where(weak.notna().to_numpy() & (weak == sug).to_numpy(), "auto", np.where(near, "ask", "confirm"))
        conf = np.where(tier == "auto", 0.95, np.where(tier == "ask", 0.5, 0.8))
    O["suggest"] = sug.to_numpy()
    O["conf"] = conf
    O["tier"] = tier
    O["source_of_suggestion"] = "model" if use_model else "preview"
    return O


def particle_page(res, store, model=None, n: int = 24, mode: str = "train", seed: int | None = None,
                  done: set | None = None, m: pd.DataFrame | None = None, only: str | None = None) -> pd.DataFrame:
    """The next page of objects to label. mode "train": outside the test wells, mixed tiers; "test": the test wells,
    whole wells in turn, blind. `done`: (plate, well, oid) already shown this session (skipped ones too).
    only: a class: training objects suggested as that class only (e.g. confirm all "noise" in one go), spread over
    plates / channels / wells, least sure first."""
    rng = np.random.default_rng(seed)
    T = particle_tiers(res, model)
    test = store.test_wells(res.dataset)
    is_test = np.array([(p, w) in test for p, w in zip(T.plate_id, T.well_id)], bool)
    skip = _labelled_oids(res, store, m) | (done or set())
    T = T[[k not in skip for k in zip(T.plate_id, T.well_id, T.oid)]]
    is_test = np.array([(p, w) in test for p, w in zip(T.plate_id, T.well_id)], bool)
    if mode == "test":
        d = T[is_test].sort_values(["plate_id", "well_id", "ch", "level"])
        wells = d[["plate_id", "well_id"]].drop_duplicates()
        if len(wells):                                           # random well order, whole wells
            order = {k: i for i, k in enumerate(map(tuple, wells.sample(frac=1, random_state=int(rng.integers(1 << 31)))
                                                     .to_numpy()))}
            d = d.assign(_o=[order[(p, w)] for p, w in zip(d.plate_id, d.well_id)]).sort_values(["_o", "ch", "level"])
        return d.head(n).drop(columns="_o", errors="ignore").assign(blind=True)
    d = T[~is_test]
    if only:
        d = d[d.suggest == only]
    if not len(d):
        return d.assign(blind=False)
    if only:
        d = d.sort_values("conf")
        d = d.groupby(["plate_id", "well_id"], sort=False).head(2)       # spread over wells, least sure first
        return d.head(n).assign(blind=False)
    want = {k: int(round(v * n)) for k, v in MIX.items()}
    picked = []
    used_wells: dict = {}

    def take(pool, k):
        out = []
        pool = pool.sample(frac=1, random_state=int(rng.integers(1 << 31)))
        # spread: round-robin over plate x channel, at most 2 objects of one well per page
        groups = [g for _, g in pool.groupby(["plate_id", "ch"])]
        rng.shuffle(groups)
        its = [iter(g.itertuples()) for g in groups]
        while len(out) < k and its:
            for it in list(its):
                r = next(it, None)
                if r is None:
                    its.remove(it)
                    continue
                key = (r.plate_id, r.well_id)
                if used_wells.get(key, 0) >= 2 or r.Index in picked:
                    continue
                used_wells[key] = used_wells.get(key, 0) + 1
                out.append(r.Index)
                picked.append(r.Index)
                if len(out) >= k:
                    break
        return out
    take(d[d.tier == "ask"], want["ask"])
    take(d[d.tier == "confirm"], want["confirm"])
    take(d, n - len(picked))                                    # random from every tier (fills shortfalls too)
    return d.loc[picked].assign(blind=False)


def cell_page(res, store, n: int = 12, mode: str = "train", seed: int | None = None, done: set | None = None):
    """Reconstructed cells to judge (verdict + colour). Needs Classify (with a model or the preview)."""
    from .labels import match_cells
    C = res.cells
    if C is None or not len(C):
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    C = C.copy()
    test = store.test_wells(res.dataset)
    C["is_test"] = [(p, w) in test for p, w in zip(C.plate_id, C.well_id)]
    m = match_cells(store.cell_frame(res.dataset), res)
    have = set(zip(m.plate_id, m.well_id, m.cell_id)) if len(m) else set()
    skip = have | (done or set())
    C = C[[k not in skip for k in zip(C.plate_id, C.well_id, C.cell_id)]]
    chans = res.run.get("identity_channels") or res.channels
    pc = C[[f"p_{c}" for c in chans if f"p_{c}" in C]].to_numpy(float)
    mid = ((pc > 0.2) & (pc < 0.8)).any(1) if pc.size else np.zeros(len(C), bool)
    unsure = (C["call"].astype(str) == "uncertain").to_numpy() if "call" in C else np.zeros(len(C), bool)
    multi = (C.get("n_basins", pd.Series(1, index=C.index)).fillna(1).to_numpy() > 1)
    C["tier"] = np.where(unsure | mid | multi, "ask", "confirm")
    if mode == "test":
        d = C[C.is_test]
        return d.sample(frac=1, random_state=int(rng.integers(1 << 31))).head(n).assign(blind=True)
    d = C[~C.is_test]
    k_ask = int(round(0.6 * n))
    a = d[d.tier == "ask"].sample(frac=1, random_state=int(rng.integers(1 << 31))).head(k_ask)
    rest = d.drop(a.index).sample(frac=1, random_state=int(rng.integers(1 << 31))).head(n - len(a))
    return pd.concat([a, rest]).assign(blind=False)


def label_progress(res, store, targets: dict | None = None, m: pd.DataFrame | None = None) -> dict:
    """Counts against simple targets: per class (training, outside the test wells), test objects labelled, and
    cells judged per colour combination."""
    from .labels import match_cells, match_labels
    targets = targets or {}
    t_cls = targets.get("class", 30)
    L = store.frame(res.dataset)
    test = store.test_wells(res.dataset)
    out = {"classes": {}, "test": {}, "cells": {}}
    if m is None:
        m = match_labels(L, res.particles, res.wells) if len(L) else pd.DataFrame(
            columns=["plate_id", "well_id", "label", "ly", "lx"])
    if len(m):
        m = m.drop_duplicates(["plate_id", "well_id", "ly", "lx"])
    is_t = np.array([(p, w) in test for p, w in zip(m.plate_id, m.well_id)], bool) if len(m) else np.zeros(0, bool)
    for c in CLASSES:
        out["classes"][c] = (int(((m.label == c).to_numpy() & ~is_t).sum()) if len(m) else 0, t_cls)
    O = objects(res.particles)
    in_test = np.array([(p, w) in test for p, w in zip(O.plate_id, O.well_id)], bool)
    out["test"] = dict(labelled=int(is_t.sum()), objects=int(in_test.sum()), wells=len(test))
    cm = match_cells(store.cell_frame(res.dataset), res)
    if len(cm):
        cm = cm[cm.colour.astype(str) != ""]
        for col, g in cm.groupby("colour"):
            out["cells"][col] = (len(g), targets.get("cells", 20))
    return out
