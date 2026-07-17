"""Per-channel supervised classification (LDA) of hole colour.

Phase 3, supervised. One **binary LDA per channel** (multi-label, matching the
rule-based classifier and the "a hole can be several channels" biology), trained
on the hand labels (:mod:`.labels`). Two features per channel, deliberately:

* ``score`` — the combined enrichment score (same as the rule-based classifier,
  but RAW / unscaled here so scaling can be fit inside each CV fold)
* ``occ``   — occupancy, fraction of the hole ROI covered by that channel

LDA on those two axes learns the optimal *oriented* boundary in the plane whose
axes are the rule-based classifier's two gates — i.e. it generalises the
axis-aligned ``score ≥ thr AND occ ≥ min_area`` box into a learned line that can
trade the two off (modest score + high occupancy → positive). At the tiny label
budget this is the honest model; morphology features and Random Forest come later
once labels reach the hundreds.

**Honest evaluation is the point** (this data has ~hundreds of wells but only the
handful the user labels): :func:`evaluate` reports, per channel, TP/FP/FN, Cohen's
kappa, the apparent-vs-CV overfit gap, N per fold, and — crucially — how the LDA
compares to the incumbent rule-based box on the *same held-out* wells. If the LDA
doesn't beat the rule, the rule wins.

Two CV schemes, because the plates are DIV timepoints:

* **leave-one-plate-out (LOGO)** — generalisation ACROSS developmental stage; the
  honest, hard test (and possibly pessimistic if only within-timepoint calls are
  wanted).
* **stratified k-fold** — pooled, within-condition generalisation.

Scaling (per-feature standardisation, and the rule's ÷SD) is fit on the TRAIN
fold only and applied to the test fold, so no test well leaks into the scale.

scikit-learn is an optional dependency (``pip install 'correlative_imaging[ml]'``);
importing this module is fine without it — only fitting/evaluating needs it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .analysis import RunTable
from .classify import ClassifierParams, classify_wells

FEATURES = ("score", "occ")


def _require_sklearn():
    try:
        import sklearn  # noqa: F401
    except ImportError as e:  # pragma: no cover - trivial guard
        raise ImportError(
            "scikit-learn is required for LDA classification. Install it with:\n"
            "    pip install 'correlative_imaging[ml]'\n"
            "  (or: pip install scikit-learn)"
        ) from e


# ── Feature / label assembly ─────────────────────────────────────────────────
def build_features(rt: RunTable) -> pd.DataFrame:
    """Long per-(plate, well, channel) feature table: RAW (unscaled) ``score`` +
    ``occ`` for every well with a hole. Population scaling is intentionally NOT
    applied — CV fits it inside each fold to avoid leakage."""
    cls = classify_wells(rt, ClassifierParams(population_scale=False, min_area=0.0))
    cols = ["plate", "well_id", "channel", "score", "occ", "hole_present"]
    if cls.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for r in cls.itertuples(index=False):
        for ch in rt.channels:
            rows.append({
                "plate": r.plate, "well_id": r.well_id, "channel": ch,
                "score": float(getattr(r, f"score_{ch}")),
                "occ": float(getattr(r, f"occ_{ch}")),
                "hole_present": bool(r.hole_present),
            })
    return pd.DataFrame(rows, columns=cols)


def build_labels(store, run_tag: str, channels: list[str]) -> pd.DataFrame:
    """Long per-(plate, well, channel) binary label from the hand labels:
    ``y = 1`` if the channel is in that well's positive set, else 0. Only labelled
    wells appear."""
    cols = ["plate", "well_id", "channel", "y"]
    lab = store.load_frame(run_tag) if store is not None else None
    if lab is None or lab.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for r in lab.itertuples(index=False):
        pos = {c for c in (getattr(r, "pos_channels", "") or "").split(",") if c}
        for ch in channels:
            rows.append({"plate": r.plate, "well_id": r.well_id,
                         "channel": ch, "y": int(ch in pos)})
    return pd.DataFrame(rows, columns=cols)


def training_table(rt: RunTable, store, run_tag: str) -> pd.DataFrame:
    """Features joined to labels, labelled wells only — the per-channel training
    data. Columns: plate, well_id, channel, score, occ, y."""
    feat = build_features(rt)
    lab = build_labels(store, run_tag, rt.channels)
    if feat.empty or lab.empty:
        return pd.DataFrame(columns=["plate", "well_id", "channel", "score", "occ", "y"])
    return lab.merge(feat, on=["plate", "well_id", "channel"], how="inner")


# ── Metrics ──────────────────────────────────────────────────────────────────
def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """TP/FP/FN/TN, accuracy and Cohen's kappa for binary (1=positive) calls."""
    from sklearn.metrics import cohen_kappa_score
    y_true = np.asarray(y_true, int)
    y_pred = np.asarray(y_pred, int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    n = tp + fp + fn + tn
    acc = (tp + tn) / n if n else float("nan")
    # kappa is undefined when one label is entirely absent from both → report 0.
    kappa = float(cohen_kappa_score(y_true, y_pred, labels=[0, 1])) if n else float("nan")
    if np.isnan(kappa):
        kappa = 0.0
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": n,
            "accuracy": acc, "kappa": kappa, "precision": prec, "recall": rec}


# ── Scaling + fit/predict primitives (fold-local) ────────────────────────────
def _zscore_fit(X: np.ndarray):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    return mu, sd


def _fit_predict(X_tr, y_tr, X_te, method: str, params: ClassifierParams):
    """Fit on train, predict test. ``method`` = 'lda' or 'rule' (the incumbent
    box). Scaling is derived from the TRAIN fold only."""
    if method == "lda":
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        mu, sd = _zscore_fit(X_tr)
        clf = LinearDiscriminantAnalysis()
        clf.fit((X_tr - mu) / sd, y_tr)
        return clf.predict((X_te - mu) / sd).astype(int)
    if method == "rule":
        # Mirror classify_wells: uncentered ÷SD scaling of score (train SD),
        # then score ≥ pos_threshold AND occ ≥ min_area. occ is raw (0–1).
        sd_score = X_tr[:, 0].std()
        sd_score = sd_score if sd_score > 0 else 1.0
        scaled = X_te[:, 0] / sd_score
        return ((scaled >= params.pos_threshold)
                & (X_te[:, 1] >= params.min_area)).astype(int)
    raise ValueError(f"unknown method {method!r}")


def _make_folds(y: np.ndarray, plates: np.ndarray, scheme: str):
    """Yield (train_idx, test_idx). 'logo' = one fold per plate; 'kfold' =
    stratified k-fold over all rows."""
    idx = np.arange(len(y))
    if scheme == "logo":
        for p in pd.unique(plates):
            te = idx[plates == p]
            tr = idx[plates != p]
            if len(te) and len(tr):
                yield tr, te
    elif scheme == "kfold":
        from sklearn.model_selection import StratifiedKFold
        npos, nneg = int(np.sum(y == 1)), int(np.sum(y == 0))
        k = max(2, min(5, npos, nneg))
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)
        yield from skf.split(idx, y)
    else:
        raise ValueError(f"unknown scheme {scheme!r}")


def _grouped_cv(X, y, plates, params, method, scheme):
    """Run a CV scheme, pooling test-fold predictions, then score once.
    Returns metrics dict (with n_folds) or None if no usable fold."""
    y_true, y_pred, n_folds = [], [], 0
    for tr, te in _make_folds(y, plates, scheme):
        if len(np.unique(y[tr])) < 2:
            continue                        # can't fit a discriminant on one class
        preds = _fit_predict(X[tr], y[tr], X[te], method, params)
        y_true.append(y[te])
        y_pred.append(preds)
        n_folds += 1
    if not y_true:
        return None
    m = _binary_metrics(np.concatenate(y_true), np.concatenate(y_pred))
    m["n_folds"] = n_folds
    return m


# ── Per-channel report ───────────────────────────────────────────────────────
@dataclass
class ChannelReport:
    channel: str
    n: int
    n_pos: int
    n_neg: int
    trainable: bool
    reason: str = ""
    apparent: dict | None = None        # in-sample LDA (overfit-optimistic)
    logo_lda: dict | None = None        # leave-one-plate-out LDA
    logo_rule: dict | None = None       # leave-one-plate-out rule baseline
    kfold_lda: dict | None = None       # stratified k-fold LDA
    kfold_rule: dict | None = None      # stratified k-fold rule baseline
    overfit_gap: float | None = None    # apparent_acc - logo_acc


def _report_channel(df_ch: pd.DataFrame, params: ClassifierParams,
                    min_per_class: int = 2) -> ChannelReport:
    X = df_ch[["score", "occ"]].to_numpy(float)
    y = df_ch["y"].to_numpy(int)
    plates = df_ch["plate"].to_numpy()
    n, n_pos = len(y), int(y.sum())
    n_neg = n - n_pos
    ch = df_ch["channel"].iloc[0] if n else "?"

    if n_pos < min_per_class or n_neg < min_per_class:
        return ChannelReport(
            channel=ch, n=n, n_pos=n_pos, n_neg=n_neg, trainable=False,
            reason=f"need ≥{min_per_class} of each class (have {n_pos}+ / {n_neg}-)",
        )

    apparent = _binary_metrics(y, _fit_predict(X, y, X, "lda", params))
    logo_lda = _grouped_cv(X, y, plates, params, "lda", "logo")
    logo_rule = _grouped_cv(X, y, plates, params, "rule", "logo")
    kfold_lda = _grouped_cv(X, y, plates, params, "lda", "kfold")
    kfold_rule = _grouped_cv(X, y, plates, params, "rule", "kfold")
    gap = (apparent["accuracy"] - logo_lda["accuracy"]
           if logo_lda is not None else None)
    return ChannelReport(
        channel=ch, n=n, n_pos=n_pos, n_neg=n_neg, trainable=True,
        apparent=apparent, logo_lda=logo_lda, logo_rule=logo_rule,
        kfold_lda=kfold_lda, kfold_rule=kfold_rule, overfit_gap=gap,
    )


def evaluate(rt: RunTable, store, run_tag: str,
             params: ClassifierParams | None = None) -> dict[str, ChannelReport]:
    """Per-channel honest evaluation of the LDA vs the rule-based baseline on the
    hand labels. Returns ``{channel: ChannelReport}``. Raises if scikit-learn is
    missing or there are no labels."""
    _require_sklearn()
    params = params or ClassifierParams()
    tt = training_table(rt, store, run_tag)
    if tt.empty:
        raise ValueError("No hand labels found for this run — label some wells first.")
    reports = {}
    for ch in rt.channels:
        df_ch = tt[tt["channel"] == ch]
        if df_ch.empty:
            reports[ch] = ChannelReport(ch, 0, 0, 0, False, "no labelled wells")
        else:
            reports[ch] = _report_channel(df_ch, params)
    return reports


# ── Predict every well ───────────────────────────────────────────────────────
def predict(rt: RunTable, store, run_tag: str,
            params: ClassifierParams | None = None) -> pd.DataFrame:
    """Fit one LDA per channel on ALL hand labels, then predict every well.
    Returns a classification frame in the same shape as :func:`classify_wells`
    (``pos_channels``, ``pos_<ch>``, ``is_negative_hole`` …) plus ``proba_<ch>``,
    so it can drop into the Explorer alongside the rule-based call. Channels that
    aren't trainable fall back to the rule-based positive call for that channel."""
    _require_sklearn()
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    params = params or ClassifierParams()

    feat = build_features(rt)
    tt = training_table(rt, store, run_tag)
    rule = classify_wells(rt, params)     # fallback + population-scaled reference
    if feat.empty:
        return pd.DataFrame()

    # Fit a per-channel model (or mark the channel as fallback).
    models = {}
    for ch in rt.channels:
        df_ch = tt[tt["channel"] == ch]
        y = df_ch["y"].to_numpy(int) if not df_ch.empty else np.array([], int)
        if df_ch.empty or int(y.sum()) < 2 or int((y == 0).sum()) < 2:
            models[ch] = None            # fall back to rule for this channel
            continue
        X = df_ch[["score", "occ"]].to_numpy(float)
        mu, sd = _zscore_fit(X)
        clf = LinearDiscriminantAnalysis().fit((X - mu) / sd, y)
        models[ch] = (clf, mu, sd)

    # Predict per well.
    feat_by_key = {(r.plate, r.well_id, r.channel): (r.score, r.occ, r.hole_present)
                   for r in feat.itertuples(index=False)}
    rule_by_key = {(r.plate, r.well_id): r for r in rule.itertuples(index=False)}

    wells = rule[["plate", "well_id", "row", "col", "experiment", "hole_present"]] \
        if not rule.empty else feat[["plate", "well_id", "hole_present"]].drop_duplicates()

    rows = []
    for w in wells.itertuples(index=False):
        rec = {"plate": w.plate, "well_id": w.well_id,
               "row": getattr(w, "row", None), "col": getattr(w, "col", None),
               "experiment": getattr(w, "experiment", None),
               "hole_present": bool(w.hole_present)}
        rr = rule_by_key.get((w.plate, w.well_id))
        positives = []
        for ch in rt.channels:
            fk = feat_by_key.get((w.plate, w.well_id, ch))
            if models[ch] is not None and fk is not None:
                clf, mu, sd = models[ch]
                x = (np.array([[fk[0], fk[1]]], float) - mu) / sd
                proba = float(clf.predict_proba(x)[0, 1])
                pos = bool(clf.predict(x)[0])
                rec[f"proba_{ch}"] = proba
            else:
                # fallback: rule-based per-channel call
                pos = bool(getattr(rr, f"pos_{ch}")) if rr is not None else False
                rec[f"proba_{ch}"] = float("nan")
            rec[f"pos_{ch}"] = pos
            if pos and rec["hole_present"]:
                positives.append(ch)
        if not rec["hole_present"]:
            for ch in rt.channels:
                rec[f"pos_{ch}"] = False
            rec.update(pos_channels="", n_positive=0, is_negative_hole=False)
        else:
            rec.update(pos_channels=",".join(positives), n_positive=len(positives),
                       is_negative_hole=(len(positives) == 0))
        rows.append(rec)
    return pd.DataFrame(rows)


# ── Text report ──────────────────────────────────────────────────────────────
def format_report(reports: dict[str, ChannelReport],
                  display: dict | None = None) -> str:
    """Human-readable per-channel report. ``display`` optionally maps channel →
    ``{"display_name": ...}`` so channels show under their friendly names."""
    def name(ch):
        return display.get(ch, {}).get("display_name", ch) if display else ch

    def line(tag, m):
        if m is None:
            return f"    {tag}: (no usable fold)"
        return (f"    {tag}: acc={m['accuracy']:.0%} κ={m['kappa']:+.2f} "
                f"TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']} "
                f"(n={m['n']}, folds={m.get('n_folds', '—')})")

    out = ["Per-channel LDA vs rule-based baseline (hand labels)", ""]
    for ch, r in reports.items():
        out.append(f"■ {name(ch)}  —  {r.n} labelled ({r.n_pos}+ / {r.n_neg}-)")
        if not r.trainable:
            out.append(f"    not trainable: {r.reason}")
            out.append("")
            continue
        out.append(f"    apparent (in-sample, optimistic): "
                   f"acc={r.apparent['accuracy']:.0%} κ={r.apparent['kappa']:+.2f}")
        out.append("  leave-one-plate-out (across DIV timepoints):")
        out.append(line("LDA ", r.logo_lda))
        out.append(line("rule", r.logo_rule))
        out.append("  stratified k-fold (pooled, within-condition):")
        out.append(line("LDA ", r.kfold_lda))
        out.append(line("rule", r.kfold_rule))
        if r.overfit_gap is not None:
            out.append(f"    overfit gap (apparent − LOGO acc): {r.overfit_gap:+.0%}")
        if r.logo_lda is not None and r.logo_rule is not None:
            verdict = ("LDA beats rule" if r.logo_lda["kappa"] > r.logo_rule["kappa"] + 1e-9
                       else "rule ≥ LDA (keep the simpler rule)")
            out.append(f"    → {verdict} on held-out (LOGO κ).")
        out.append("")
    return "\n".join(out)
