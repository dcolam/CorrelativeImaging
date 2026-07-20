"""Feature-space dimensionality reduction + unbiased clustering of wells.

Qt-free layer for the Explorer's Clusters tab. One row per (run_tag, plate,
well) — a "cell" is a well's hole — described by a configurable set of per-channel
features. Produces PCA (always) and UMAP (optional dep) 2-D embeddings and
posthoc cluster labels (HDBSCAN / KMeans / Agglomerative), so the GUI can scatter
the cells, colour them, and map a clicked point back to its well.

Design decisions baked in here (see the Clusters tab for the interaction):

* **Features** come from the shared catalog (:mod:`.features`) — the caller picks
  which catalog columns to embed on; runs are NEVER mixed (one run at a time).
* **Scaling** (:data:`SCALING_GROUPS`) z-scores each selected column within a
  row-group. Within one run "run" == whole-run per channel across plates (matches
  the R figures); "run_plate" is per-plate (removes between-plate differences).
  Centered-vs-uncentered ÷SD is irrelevant to PCA/UMAP/clustering (distances are
  unchanged); the grouping is the only thing that matters.
* **Clustering is done on the FEATURE matrix, not on the 2-D coords** (clustering
  on UMAP/PCA coordinates is an anti-pattern — those distances aren't faithful).
* UMAP gets a fixed ``random_state`` so a refit is reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .analysis import RunTable
from .classify import ClassifierParams, classify_wells

# Scaling grouping name → row-group columns to z-score each feature within.
# Runs are never mixed, so within a single run "run" == global. The meaningful
# choice is whole-run (per channel across plates, matches the R figures) vs
# per-plate (removes between-plate differences).
SCALING_GROUPS: dict[str, list[str]] = {
    "global": [],
    "run": ["run_tag"],
    "run_plate": ["run_tag", "plate"],
    "plate": ["plate"],
}
DEFAULT_SCALING = "run"


@dataclass
class FeatureMatrix:
    """Per-cell feature matrix for embedding/clustering.

    ``X`` is ``[n_cells × n_features]`` (standardised per the chosen grouping);
    ``meta`` has one row per cell with ``run_tag``, ``plate``, ``well_id``,
    ``pos_channels`` (the classifier call, for colour-by) and ``is_negative``;
    ``feature_names`` labels the columns; ``channels`` is the shared channel set."""
    X: np.ndarray
    meta: pd.DataFrame
    feature_names: list[str]
    channels: list[str]

    def __len__(self) -> int:
        return self.X.shape[0]


def umap_import_error() -> str | None:
    """``None`` if ``umap`` imports cleanly, else a short description of WHY it
    failed (numba/numpy ABI, wrong env, …) — a bare "not installed" hides that."""
    try:
        import umap  # noqa: F401
        return None
    except ModuleNotFoundError as e:
        return f"not installed ({e})"
    except Exception as e:
        return f"installed but failed to import — {type(e).__name__}: {e}"


def umap_available() -> bool:
    return umap_import_error() is None


# ── Feature assembly ─────────────────────────────────────────────────────────
# Standardisation methods (per feature column, within the chosen row-group).
# Mirrors the options in ephacRTools (scoreParticles center T/F, the median
# threshold variant, and the ldaTools plate-centering step).
SCALE_METHODS = {
    "zscore": "z-score (centre + ÷SD)",
    "sd": "÷SD only (uncentred)",
    "robust": "robust (median + MAD)",
    "center": "centre only (subtract mean)",
    "none": "none (raw)",
}
DEFAULT_SCALE_METHOD = "zscore"


def _standardize_within(df: pd.DataFrame, feat_cols: list[str],
                        group_cols: list[str], method: str = "zscore") -> pd.DataFrame:
    """Standardise each feature column within row-groups (``group_cols`` empty =
    global) by ``method`` (see :data:`SCALE_METHODS`). Zero-variance / single-row
    groups collapse to 0."""
    out = df.copy()

    def _apply(block: pd.DataFrame) -> pd.DataFrame:
        b = block.astype(float)
        if method == "none":
            return b.fillna(0.0)
        if method == "center":
            return (b - b.mean()).fillna(0.0)
        if method == "sd":
            sd = b.std(ddof=0).replace(0.0, 1.0)
            return (b / sd).fillna(0.0)
        if method == "robust":
            med = b.median()
            mad = (b - med).abs().median() * 1.4826
            scale = mad.where(mad > 0, b.std(ddof=0)).replace(0.0, 1.0)
            return ((b - med) / scale).fillna(0.0)
        mu = b.mean(); sd = b.std(ddof=0).replace(0.0, 1.0)      # zscore
        return ((b - mu) / sd).fillna(0.0)

    if group_cols:
        for _key, idx in df.groupby(group_cols).groups.items():
            out.loc[idx, feat_cols] = _apply(df.loc[idx, feat_cols])
    else:
        out[feat_cols] = _apply(df[feat_cols])
    return out


def build_feature_matrix(
    rt: RunTable,
    selected_columns: list[str] | None = None,
    scaling: str = DEFAULT_SCALING,
    omit_empty: bool = False,
    params: ClassifierParams | None = None,
    roi: str = "roi_hole",
    scale_method: str = DEFAULT_SCALE_METHOD,
    particle_filter: dict | None = None,
) -> FeatureMatrix | None:
    """Assemble the per-cell feature matrix for ONE run (all its plates — runs are
    never mixed) from the shared feature catalog (:mod:`.features`).

    ``selected_columns`` picks catalog columns (``None`` → the catalog's default
    subset). ``scaling`` sets the row-group (:data:`SCALING_GROUPS`) and
    ``scale_method`` the per-column standardisation (:data:`SCALE_METHODS`); within
    one run "run" == whole-run per channel. ``particle_filter`` (e.g.
    ``{"method": "zscore", "threshold": 0.0}``) drops background particles before
    the particle-based features. ``omit_empty`` drops classifier-negative wells.
    The classifier call is carried into ``meta`` for colour-by."""
    from .features import build_catalog

    if rt is None or not rt.channels:
        return None
    cat = build_catalog(rt, roi=roi, particle_filter=particle_filter)
    if cat.df.empty:
        return None
    df = cat.df.copy()

    # Classifier call + negativity per well (for colour-by and omit-empty).
    params = params or ClassifierParams()
    cls = classify_wells(rt, params)
    if not cls.empty:
        call = cls[["plate", "well_id", "pos_channels", "is_negative_hole"]]
        df = df.merge(call, on=["plate", "well_id"], how="left")
    else:
        df["pos_channels"] = ""; df["is_negative_hole"] = False
    if omit_empty:
        df = df[~df["is_negative_hole"].fillna(False).astype(bool)]
    if df.empty:
        return None

    cols = [c for c in (selected_columns or cat.default_selection()) if c in cat.columns]
    if not cols:
        cols = cat.default_selection()
    if not cols:
        return None

    df[cols] = df[cols].fillna(0.0)
    scaled = _standardize_within(df, cols, SCALING_GROUPS.get(scaling, []), scale_method)
    X = np.nan_to_num(scaled[cols].to_numpy(dtype=float), nan=0.0)
    meta = df[["run_tag", "plate", "well_id", "pos_channels", "is_negative_hole"]] \
        .reset_index(drop=True)
    return FeatureMatrix(X=X, meta=meta, feature_names=cols, channels=rt.channels)


def embed(X: np.ndarray, method: str = "pca", random_state: int = 0) -> np.ndarray:
    """2-D embedding of ``X``. ``method`` = ``"pca"`` (sklearn) or ``"umap"``
    (optional dep; raises ImportError with an install hint if absent). Fewer than
    3 rows → the raw first two columns padded to 2-D."""
    n = X.shape[0]
    if n < 3:
        pad = np.zeros((n, 2), dtype=float)
        pad[:, : min(2, X.shape[1])] = X[:, : min(2, X.shape[1])]
        return pad
    if method == "pca":
        from sklearn.decomposition import PCA
        return PCA(n_components=2).fit_transform(X)
    if method == "umap":
        try:
            import umap
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "UMAP needs umap-learn — install with: pip install "
                "'correlative_imaging[ml]'  (or: pip install umap-learn)"
            ) from e
        n_neighbors = min(15, max(2, n - 1))
        return umap.UMAP(n_components=2, random_state=random_state,
                         n_neighbors=n_neighbors).fit_transform(X)
    raise ValueError(f"unknown method {method!r}")


def pca_embed(X: np.ndarray):
    """PCA embedding that also returns the loadings (``components_``, shape
    ``[2 × n_features]``) and per-PC explained-variance fraction, so the GUI can
    show which features drive PC1/PC2. Returns ``(coords, loadings, var_ratio)``;
    loadings/var are ``None`` for <3 rows."""
    n = X.shape[0]
    if n < 3:
        pad = np.zeros((n, 2), dtype=float)
        pad[:, : min(2, X.shape[1])] = X[:, : min(2, X.shape[1])]
        return pad, None, None
    from sklearn.decomposition import PCA
    p = PCA(n_components=2).fit(X)
    return p.transform(X), p.components_, p.explained_variance_ratio_


def lda_embedding(X: np.ndarray, class_labels, min_per_class: int = 2):
    """Supervised 2-D embedding: fit a MULTI-CLASS LDA on the hand-labelled cells
    (``class_labels[i]`` is the class string, or ``None`` for unlabelled), then
    project ALL cells into LD space → LD1 vs LD2.

    This is distinct from the per-channel binary LDA in :mod:`.lda` (that one is
    the calibration/prediction tool and has only LD1). Returns ``(coords, note)``:
    ``coords`` is ``[n × 2]`` or ``None`` when there aren't ≥3 classes with
    ≥``min_per_class`` labelled cells each; ``note`` explains why when ``None``."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    y = np.asarray(class_labels, dtype=object)
    labelled = np.array([v is not None for v in y])
    n_lab = int(labelled.sum())
    if n_lab < 3 * min_per_class:
        return None, f"LDA needs ≥3 labelled classes; only {n_lab} labelled cells."
    classes, counts = np.unique(y[labelled].astype(str), return_counts=True)
    keep = classes[counts >= min_per_class]
    if len(keep) < 3:
        return None, (f"LDA needs ≥3 classes with ≥{min_per_class} labels; "
                      f"have {len(keep)} ({', '.join(keep)}).")
    fit_mask = labelled & np.isin(y.astype(str), keep)
    lda = LinearDiscriminantAnalysis(n_components=2)
    lda.fit(X[fit_mask], y[fit_mask].astype(str))
    coords = lda.transform(X)
    if coords.shape[1] < 2:      # shouldn't happen with ≥3 classes, guard anyway
        coords = np.column_stack([coords, np.zeros(len(coords))])
    return coords, f"LDA fit on {int(fit_mask.sum())} labelled cells, {len(keep)} classes."


def cluster_labels(X: np.ndarray, method: str = "hdbscan",
                   min_cluster_size: int = 15, n_clusters: int = 5) -> np.ndarray:
    """Posthoc clustering of the FEATURE matrix (not the 2-D coords). ``method`` =
    ``"hdbscan"`` (density, no preset k; noise = -1), ``"kmeans"`` or
    ``"agglomerative"`` (both use ``n_clusters``)."""
    n = X.shape[0]
    if n < 3:
        return np.zeros(n, dtype=int)
    if method == "hdbscan":
        from sklearn.cluster import HDBSCAN
        mcs = int(max(2, min(min_cluster_size, max(2, n // 2))))
        return HDBSCAN(min_cluster_size=mcs, copy=True).fit_predict(X)
    if method == "kmeans":
        from sklearn.cluster import KMeans
        k = int(max(2, min(n_clusters, n)))
        return KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)
    if method == "agglomerative":
        from sklearn.cluster import AgglomerativeClustering
        k = int(max(2, min(n_clusters, n)))
        return AgglomerativeClustering(n_clusters=k).fit_predict(X)
    raise ValueError(f"unknown cluster method {method!r}")
