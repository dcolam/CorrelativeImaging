"""Particle classifier: trained on hand labels, extended to all particles, and checked honestly.

Pretraining on all particles (semi-supervised): most particles are never labelled, but they show the structure of the
data. Three ways to use them are compared side by side ("arms"), on the same folds:

  legacy        logistic regression on the reference workflow's 4 features (SNR, area, share, fill)
  hand          gradient-boosted trees on all features, hand labels only
  spread        label spreading: hand labels diffuse over a k-nearest-neighbour graph of ALL particles (in a PCA
                embedding of the standardised features), then a kNN classifier on the spread labels
  self-train    hand labels + confident pseudo-labels on the unlabelled particles, refitted in rounds. A pseudo-label
                is only taken when the trees (P >= tau) and label spreading agree, and counts with a lower weight
  self-train+rules  the same, starting from a few colour-blind rules as extra weak labels (see `weak_labels`)

Honesty rules, built in:
  * a frozen test set of wells (labels.LabelStore.freeze_test_wells) is never trained on, never pseudo-labelled and
    never part of the spreading graph. Its numbers are reported, never used to choose anything;
  * model choice uses cross-validation grouped by well (particles of one well never sit in train and validation);
    pseudo-labels are made inside each fold, without the validation wells;
  * coverage (the share of all particles the model is confident about, P >= 0.9) is reported next to accuracy,
    because pseudo-labelling raises coverage much more easily than it raises accuracy.

Prediction needs only the stored estimator and the feature list (joblib file + JSON provenance next to it).
"""
from __future__ import annotations

import datetime
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .features import FEATURES, LEGACY_FEATURES
from .labels import CLASSES

ARMS = ["legacy", "hand", "spread", "self-train", "self-train+rules"]


def group_key(df: pd.DataFrame) -> np.ndarray:
    ds = df["dataset"] if "dataset" in df else ""
    return (ds.astype(str) + "|" + df.plate_id.astype(str) + "|" + df.well_id.astype(str)).to_numpy() \
        if isinstance(ds, pd.Series) else (df.plate_id.astype(str) + "|" + df.well_id.astype(str)).to_numpy()


def weak_labels(F: pd.DataFrame) -> pd.Series:
    """A few colour-blind rules that are almost never wrong; used only as weak labels in one arm.
      noise      no signal: SNR < 1 against both the frame and the hole background
      neighbour  a lenient or stringent piece of an object lying mostly outside (share < 0.2) whose whole object looks
                 like a typical field cell (field distance in the closest 25 %)
      debris     small (< 5 um2 inside) and bright in every other channel (flatness > 0.5, all others SNR > 3)"""
    lab = pd.Series(None, index=F.index, dtype=object)
    noise = (F.snr < 1) & (F.hsnr < 1)
    d2q = F.f_field_d2.quantile(.25)
    nb = (F.share < 0.2) & (F.f_field_d2 <= d2q)
    deb = (F.in_um2 < 5) & (F.f_flat > 0.5) & (F.f_oth_pos >= 1)
    lab[noise] = "noise"
    lab[nb & lab.isna()] = "neighbour"
    lab[deb & lab.isna()] = "debris"
    return lab


# ----------------------------------------------------------------------------------------------------------------
# estimators

def _hgb():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=8,
                                          l2_regularization=1.0, class_weight="balanced", random_state=0)


def _legacy():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(max_iter=2000, class_weight="balanced"))


class Embedding:
    """Median-impute, standardise, PCA: the space label spreading and the kNN classifier work in."""

    def __init__(self, n_components: int = 8):
        self.n = n_components

    def fit(self, X):
        from sklearn.decomposition import PCA
        self.med = np.nanmedian(X, 0)
        Xi = np.where(np.isfinite(X), X, self.med)
        self.mu, self.sd = Xi.mean(0), Xi.std(0)
        self.sd[self.sd == 0] = 1
        self.pca = PCA(min(self.n, X.shape[1]), random_state=0).fit((Xi - self.mu) / self.sd)
        return self

    def transform(self, X):
        Xi = np.where(np.isfinite(X), X, self.med)
        return self.pca.transform((Xi - self.mu) / self.sd)


def _spread(Z_lab, y_lab, Z_unl, k=10, alpha=0.2):
    """Label spreading over labelled + unlabelled; returns class distributions for the unlabelled points."""
    from sklearn.semi_supervised import LabelSpreading
    Z = np.vstack([Z_lab, Z_unl])
    y = np.r_[y_lab, -np.ones(len(Z_unl), int)]
    ls = LabelSpreading(kernel="knn", n_neighbors=k, alpha=alpha, max_iter=200).fit(Z, y)
    D = np.zeros((len(Z_unl), len(CLASSES)))
    D[:, ls.classes_] = ls.label_distributions_[len(Z_lab):]
    return D


def _proba(est, X) -> np.ndarray:
    P = np.zeros((len(X), len(CLASSES)))
    if len(X):
        P[:, est.classes_] = est.predict_proba(X)
    return P


@dataclass
class ArmModel:
    arm: str
    est: object
    emb: Embedding | None = None
    n_pseudo: int = 0

    def predict_proba(self, X) -> np.ndarray:
        if self.arm == "legacy":
            return _proba(self.est, X[:, [FEATURES.index(f) for f in LEGACY_FEATURES]])
        if self.arm == "spread":
            return _proba(self.est, self.emb.transform(X))
        return _proba(self.est, X)


def fit_arm(arm: str, X, y, w, X_unl, rounds: int = 3, tau: float = 0.9, w_pseudo: float = 0.3,
            weak: np.ndarray | None = None, log: list | None = None) -> ArmModel:
    """X, y, w: labelled features / class index / weight; X_unl: unlabelled pool (same feature columns).
    weak: class index or -1 per unlabelled row (arm self-train+rules)."""
    if arm == "legacy":
        cols = [FEATURES.index(f) for f in LEGACY_FEATURES]
        return ArmModel(arm, _legacy().fit(X[:, cols], y, logisticregression__sample_weight=w))
    if arm == "hand":
        return ArmModel(arm, _hgb().fit(X, y, sample_weight=w))
    emb = Embedding().fit(np.vstack([X, X_unl]) if len(X_unl) else X)
    Zl, Zu = emb.transform(X), emb.transform(X_unl) if len(X_unl) else np.zeros((0, emb.pca.n_components_))
    D = _spread(Zl, y, Zu) if len(Zu) else np.zeros((0, len(CLASSES)))
    if arm == "spread":
        from sklearn.neighbors import KNeighborsClassifier
        ok = D.max(1) >= tau if len(D) else np.zeros(0, bool)
        Z = np.vstack([Zl, Zu[ok]])
        yy = np.r_[y, D[ok].argmax(1)]
        return ArmModel(arm, KNeighborsClassifier(15, weights="distance").fit(Z, yy), emb, int(ok.sum()))
    # self-training
    Xa, ya, wa = X, y, w
    if arm == "self-train+rules" and weak is not None:
        k = weak >= 0
        Xa, ya, wa = np.vstack([X, X_unl[k]]), np.r_[y, weak[k]], np.r_[w, np.full(k.sum(), w_pseudo)]
    est = _hgb().fit(Xa, ya, sample_weight=wa)
    n_ps = 0
    for r in range(rounds):
        if not len(X_unl):
            break
        P = _proba(est, X_unl)
        take = (P.max(1) >= tau) & (P.argmax(1) == D.argmax(1))
        if arm == "self-train+rules" and weak is not None:
            take &= ~(weak >= 0)
        n_ps = int(take.sum())
        Xa = np.vstack([X, X_unl[take]])
        ya = np.r_[y, P[take].argmax(1)]
        wa = np.r_[w, w_pseudo * P[take].max(1)]
        if arm == "self-train+rules" and weak is not None:
            k = weak >= 0
            Xa, ya, wa = np.vstack([Xa, X_unl[k]]), np.r_[ya, weak[k]], np.r_[wa, np.full(k.sum(), w_pseudo)]
        est = _hgb().fit(Xa, ya, sample_weight=wa)
        if log is not None:
            log.append(dict(arm=arm, round=r + 1, n_pseudo=n_ps))
    return ArmModel(arm, est, emb, n_ps)


# ----------------------------------------------------------------------------------------------------------------
# evaluation

def metrics(y, P, w=None) -> dict:
    import warnings
    from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
    warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
    pred = P.argmax(1)
    present = sorted(set(y))
    out = dict(n=len(y), macro_f1=f1_score(y, pred, labels=present, average="macro", sample_weight=w),
               bal_acc=balanced_accuracy_score(y, pred, sample_weight=w))
    cell = CLASSES.index("cell")
    yc = (y == cell).astype(int)
    out["auc_cell"] = roc_auc_score(yc, P[:, cell], sample_weight=w) if 0 < yc.sum() < len(yc) else np.nan
    for c in present:
        m = y == c
        out[f"recall_{CLASSES[c]}"] = float(np.average(pred[m] == c, weights=None if w is None else w[m]))
    return out


@dataclass
class TrainResult:
    model: "ParticleModel"
    cv: pd.DataFrame              # arm x metrics, grouped CV (mean over folds)
    test: pd.DataFrame            # arm x metrics on the frozen test wells
    rounds: pd.DataFrame
    queue: pd.DataFrame           # unlabelled particles to label next
    notes: list = field(default_factory=list)


def training_frame(F: pd.DataFrame, matched: pd.DataFrame) -> pd.DataFrame:
    """Pool particles with their label (NaN = unlabelled) and weight."""
    keys = ["dataset", "plate_id", "well_id", "oid"]
    pool = F[F.pool].copy()
    m = matched.groupby(keys).agg(label=("label", "first"), weight=("weight", "sum")).reset_index()
    return pool.merge(m, on=keys, how="left")


def train(T: pd.DataFrame, test_groups: set, arms=ARMS, folds: int = 5, rounds: int = 3, tau: float = 0.9,
          final_arm: str | None = None, progress=None, feature_config: dict | None = None,
          cv: bool = True) -> TrainResult:
    """T: training_frame (all pool particles of the loaded datasets, labelled or not). test_groups: group keys
    (dataset|plate|well) of the frozen test wells. Returns the final model (fitted on all non-test labels with the
    chosen arm) + the reports."""
    from sklearn.model_selection import GroupKFold
    say = progress or (lambda m: None)
    g = group_key(T)
    is_test = np.isin(g, list(test_groups))
    lab = T.label.notna().to_numpy()
    X_all = T[FEATURES].to_numpy(float)
    y_all = np.array([CLASSES.index(c) if isinstance(c, str) else -1 for c in T.label])
    w_all = T.weight.fillna(0).to_numpy(float)
    weak_all = np.array([CLASSES.index(c) if isinstance(c, str) else -1 for c in weak_labels(T)])
    tr = lab & ~is_test
    notes = []
    if tr.sum() < 30:
        raise ValueError(f"only {int(tr.sum())} labelled particles outside the test wells: label more first")
    counts = pd.Series([CLASSES[i] for i in y_all[tr]]).value_counts()
    for c in CLASSES:
        if counts.get(c, 0) < 15:
            notes.append(f"class '{c}' has only {counts.get(c, 0)} training labels: its recall is not reliable")

    # grouped CV over the training wells
    rows, rlog = [], []
    gtr = g[tr]
    nf = min(folds, len(set(gtr)))
    splits = list(GroupKFold(nf).split(X_all[tr], y_all[tr], gtr))
    idx_tr = np.flatnonzero(tr)
    for arm in (arms if cv else []):                    # cv=False: the labelling loop's quick refit, no comparison
        say(f"cross-validating '{arm}'")
        fold_m = []
        for k, (a, b) in enumerate(splits):
            fit_i, val_i = idx_tr[a], idx_tr[b]
            val_groups = set(g[val_i])
            unl = (~lab) & ~is_test & ~np.isin(g, list(val_groups))
            m = fit_arm(arm, X_all[fit_i], y_all[fit_i], w_all[fit_i], X_all[unl], rounds, tau,
                        weak=weak_all[unl], log=rlog if k == 0 else None)
            fold_m.append(metrics(y_all[val_i], m.predict_proba(X_all[val_i]), w_all[val_i]))
        cvm = pd.DataFrame(fold_m).mean(numeric_only=True)
        rows.append(dict(arm=arm, **cvm.to_dict()))
    cv = pd.DataFrame(rows, columns=["arm", "macro_f1"]) if not rows else pd.DataFrame(rows)
    if not len(cv):
        cv = pd.DataFrame(dict(arm=list(arms), macro_f1=np.nan))

    # final fits on all training labels; the frozen test wells only reported
    unl = (~lab) & ~is_test
    test_rows, models = [], {}
    for arm in arms:
        say(f"fitting '{arm}' on all training labels")
        m = fit_arm(arm, X_all[tr], y_all[tr], w_all[tr], X_all[unl], rounds, tau, weak=weak_all[unl])
        models[arm] = m
        P_unl = m.predict_proba(X_all[unl]) if unl.any() else np.zeros((0, len(CLASSES)))
        cov = float((P_unl.max(1) >= 0.9).mean()) if len(P_unl) else np.nan
        te = lab & is_test
        tm = metrics(y_all[te], m.predict_proba(X_all[te]), w_all[te]) if te.sum() >= 10 else {"n": int(te.sum())}
        test_rows.append(dict(arm=arm, coverage_unlabelled=cov, n_pseudo=m.n_pseudo, **tm))
    test = pd.DataFrame(test_rows)
    cv = cv.merge(test[["arm", "coverage_unlabelled", "n_pseudo"]], on="arm")
    best = final_arm or (cv.sort_values("macro_f1", ascending=False).arm.iloc[0] if cv.macro_f1.notna().any()
                         else list(arms)[0])
    chosen = models[best]

    # what to label next: the most uncertain unlabelled particles, one per well, test wells excluded
    queue = pd.DataFrame()
    if unl.any():
        P = chosen.predict_proba(X_all[unl])
        ent = -(np.where(P > 0, P * np.log(P), 0)).sum(1)
        q = T.loc[unl, ["dataset", "plate_id", "well_id", "oid", "ch", "level"]].assign(
            entropy=ent, pred=[CLASSES[i] for i in P.argmax(1)], p_max=P.max(1))
        queue = q.sort_values("entropy", ascending=False).drop_duplicates(["dataset", "plate_id", "well_id"]).head(200)

    pm = ParticleModel(chosen, dict(
        arm=best, classes=CLASSES, features=FEATURES, fitted=datetime.datetime.now().isoformat(timespec="seconds"),
        n_train=int(tr.sum()), n_test=int((lab & is_test).sum()), n_unlabelled=int(unl.sum()),
        train_counts=counts.to_dict(), datasets=sorted(T.dataset.unique().tolist()) if "dataset" in T else [],
        cv=cv.set_index("arm").round(4).to_dict("index"), test=test.set_index("arm").round(4).to_dict("index"),
        labels_sha1=hashlib.sha1(pd.util.hash_pandas_object(T.loc[lab, ["plate_id", "well_id", "oid", "label"]])
                                 .values).hexdigest(),
        sklearn=__import__("sklearn").__version__, notes=notes, feature_config=feature_config or {}))
    return TrainResult(pm, cv, test, pd.DataFrame(rlog), queue, notes)


class ParticleModel:
    """A fitted arm + provenance. predict(F) adds p_<class> and `pclass` columns."""

    def __init__(self, arm_model: ArmModel, spec: dict):
        self.m, self.spec = arm_model, spec

    def predict(self, F: pd.DataFrame) -> pd.DataFrame:
        F = F.copy()
        P = self.m.predict_proba(F[self.spec["features"]].to_numpy(float)) if len(F) else np.zeros((0, len(CLASSES)))
        for i, c in enumerate(CLASSES):
            F[f"p_{c}"] = P[:, i]
        F["pclass"] = [CLASSES[i] for i in P.argmax(1)] if len(F) else []
        return F

    def save(self, path: str | Path):
        """Stored as plain data (sklearn estimators, arrays, spec), never as this package's own classes, so a
        model survives renaming or moving this package."""
        import joblib
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        e = self.m.emb
        joblib.dump(dict(format=2, arm=self.m.arm, est=self.m.est, n_pseudo=self.m.n_pseudo,
                         emb=None if e is None else dict(n=e.n, med=e.med, mu=e.mu, sd=e.sd, pca=e.pca),
                         spec=self.spec), path)
        path.with_suffix(".json").write_text(json.dumps(self.spec, indent=2, default=str))

    @classmethod
    def load(cls, path: str | Path) -> "ParticleModel":
        import joblib
        import sys
        # models saved before the package was renamed (identity -> reconstruction) pickled its classes by path
        if "correlative_imaging.identity" not in sys.modules:
            from .. import reconstruction as _pkg
            from . import learn as _learn
            sys.modules.setdefault("correlative_imaging.identity", _pkg)
            sys.modules.setdefault("correlative_imaging.identity.learn", _learn)
        d = joblib.load(path)
        if d.get("format") == 2:
            emb = None
            if d["emb"] is not None:
                emb = Embedding(d["emb"]["n"])
                emb.med, emb.mu, emb.sd, emb.pca = d["emb"]["med"], d["emb"]["mu"], d["emb"]["sd"], d["emb"]["pca"]
            return cls(ArmModel(d["arm"], d["est"], emb, d.get("n_pseudo", 0)), d["spec"])
        return cls(d["model"], d["spec"])


def learning_curve(T: pd.DataFrame, test_groups: set, arm: str = "hand", fracs=(0.25, 0.5, 0.75, 1.0), seed: int = 0,
                   min_test: int = 30) -> pd.DataFrame | None:
    """Score on the test wells against the number of training labels (random subsets of the training wells). None
    when the test wells hold fewer than `min_test` labels (the curve would be noise)."""
    rng = np.random.default_rng(seed)
    g = group_key(T)
    is_test = np.isin(g, list(test_groups))
    lab = T.label.notna().to_numpy()
    te = lab & is_test
    if te.sum() < min_test:
        return None
    X = T[FEATURES].to_numpy(float)
    y = np.array([CLASSES.index(c) if isinstance(c, str) else -1 for c in T.label])
    w = T.weight.fillna(0).to_numpy(float)
    tr_wells = np.array(sorted(set(g[lab & ~is_test])))
    unl = (~lab) & ~is_test
    weak = np.array([CLASSES.index(c) if isinstance(c, str) else -1 for c in weak_labels(T)])
    rows = []
    for f in fracs:
        pick = set(rng.choice(tr_wells, max(1, int(round(f * len(tr_wells)))), replace=False))
        tr = lab & ~is_test & np.isin(g, list(pick))
        if tr.sum() < 20 or len(set(y[tr])) < 2:
            continue
        m = fit_arm(arm, X[tr], y[tr], w[tr], X[unl], rounds=1, weak=weak[unl])
        mt = metrics(y[te], m.predict_proba(X[te]), w[te])
        mt["n_test"] = mt.pop("n")
        rows.append(dict(n=int(tr.sum()), **mt))
    return pd.DataFrame(rows)


def compare_models(results_dir, store, models: list, tol_um: float = 2.0) -> pd.DataFrame:
    """Every model (a .joblib path / name in models/, or "preview") scored on the labelled particles of the frozen test
    wells of one results folder (its dataset's test wells in `store`)."""
    from .config import MODELS
    from .pipeline import Results
    from .preview import PreviewModel
    T, tg, _ = assemble([results_dir], store, tol_um)
    g = group_key(T)
    te = T.label.notna().to_numpy() & np.isin(g, list(tg))
    y = np.array([CLASSES.index(c) for c in T.label[te]])
    w = T.weight[te].fillna(1).to_numpy(float)
    rows = []
    for name in models:
        try:
            if name == "preview":
                mdl = PreviewModel()
            else:
                p = Path(name)
                mdl = ParticleModel.load(p if p.exists() else MODELS / name)
            miss = [f for f in getattr(mdl, "spec", {}).get("features", []) if f not in T]
            if miss:
                rows.append(dict(model=name, n=int(te.sum()), note=f"results lack {', '.join(miss)}: detect again"))
                continue
            M = mdl.predict(T[te])
            P = M[[f"p_{c}" for c in CLASSES]].to_numpy(float)
            sp = getattr(mdl, "spec", {})
            rows.append(dict(model=name, arm=sp.get("arm"), trained_on=",".join(sp.get("datasets", []) or []),
                             n_train=sp.get("n_train"), **{k: (round(v, 3) if isinstance(v, float) else v)
                                                           for k, v in metrics(y, P, w).items()}))
        except Exception as e:
            rows.append(dict(model=name, note=str(e)))
    return pd.DataFrame(rows)


def assemble(results_dirs: list, store, tol_um: float = 2.0, test_frac: float = 0.2):
    """Training frame over several output folders: their pool particles, the labels of their datasets matched to
    particles, and the frozen test wells per dataset (frozen on first use). Returns (T, test_groups, report);
    report.attrs carries the detection feature settings (and whether the folders differ in them)."""
    from .labels import match_labels
    from .pipeline import Results
    frames, tg, rep, fcs = [], set(), [], []
    for d in results_dirs:
        r = Results(d)
        F = r.particles
        L = store.frame(r.dataset)
        m = match_labels(L, F, r.wells, tol_um)
        n_dropped = 0
        if len(m) and "n_parts" in F:
            # a "merged" label made on an object that Detect now splits no longer describes one particle
            parts = F.set_index(["plate_id", "well_id", "oid"]).n_parts
            np_ = parts.reindex(list(zip(m.plate_id, m.well_id, m.oid))).to_numpy(float)
            drop = (m.label == "merged").to_numpy() & (np.nan_to_num(np_, nan=1) > 1)
            n_dropped = int(drop.sum())
            m = m[~drop]
        # cell labels (outside the test wells) lend weak labels to their seed particles: "one cell" -> cell,
        # "several cells" -> merged (weight 0.5; a particle's own label wins)
        from .labels import match_cells
        tw_ = store.test_wells(r.dataset)
        cm = match_cells(store.cell_frame(r.dataset), r)
        n_cell_weak = 0
        if len(cm) and "members" in cm:
            cm = cm[[(p, w) not in tw_ for p, w in zip(cm.plate_id, cm.well_id)]]
            add = []
            for c in cm.itertuples():
                lab = {"one cell": "cell", "several cells": "merged"}.get(c.verdict)
                if not lab or not isinstance(c.members, str):
                    continue
                for o in c.members.split(";"):
                    if o:
                        add.append(dict(dataset=r.dataset, plate_id=c.plate_id, well_id=c.well_id, oid=o, label=lab,
                                        source="cell label", weight=0.5))
            if add:
                A = pd.DataFrame(add)
                have = set(zip(m.plate_id, m.well_id, m.oid)) if len(m) else set()
                A = A[[k not in have for k in zip(A.plate_id, A.well_id, A.oid)]]
                n_cell_weak = len(A)
                m = pd.concat([m, A], ignore_index=True) if len(m) else A
        frames.append(training_frame(F, m))
        fcs.append(r.config.feature_config())
        tw = store.freeze_test_wells(r.dataset, test_frac)
        tg |= {f"{r.dataset}|{p}|{w}" for p, w in tw}
        u = m.drop_duplicates(["plate_id", "well_id", "ly", "lx"]) if len(m) else m
        rep.append(dict(dataset=r.dataset, folder=str(d), labels=len(L), matched=len(u), test_wells=len(tw),
                        pool=int(F.pool.sum()), merged_on_split_dropped=n_dropped, from_cell_labels=n_cell_weak))
    T = pd.concat(frames, ignore_index=True)
    rep = pd.DataFrame(rep)
    rep.attrs["feature_config"] = fcs[0] if fcs else {}
    rep.attrs["feature_config_differs"] = any(f != fcs[0] for f in fcs)
    return T, tg, rep
