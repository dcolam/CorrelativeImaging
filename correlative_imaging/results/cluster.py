"""Feature-space dimensionality reduction + unbiased clustering of wells.

Qt-free layer for the Explorer's Clusters tab. One row per (run_tag, plate,
well) — a "cell" is a well's hole — described by the SAME features the classifier
uses: per-channel population-scaled ``score`` + ``occ`` (occupancy). Produces PCA
(always) and UMAP (optional dep) 2-D embeddings and HDBSCAN cluster labels, so the
GUI can scatter the cells, colour them by unbiased cluster, and map a clicked
point back to its well.

Design decisions baked in here (see the Clusters tab for the interaction):

* **Aligned by channel NAME**, never position — channel↔order varies between
  experiments. Multiple runs are combined on their *shared* channels; if runs
  share no channel, the caller should fall back to a single run.
* **Standardised per run** before embedding. The score is already per-run
  population-scaled, but per-run z-scoring each feature further limits the
  "all-runs" view from clustering by BATCH rather than biology. It cannot remove
  batch structure entirely — flag that to the user.
* The **fit-set vs display** split lives in the GUI: fit PCA/UMAP/HDBSCAN once on
  the chosen fit-set (all cells, or one plate), then filter which points are
  drawn. UMAP is given a fixed ``random_state`` so a refit is reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .analysis import RunTable
from .classify import ClassifierParams, classify_wells


@dataclass
class FeatureMatrix:
    """Per-cell feature matrix for embedding/clustering.

    ``X`` is ``[n_cells × n_features]`` (per-run standardised); ``meta`` has one
    row per cell with ``run_tag``, ``plate``, ``well_id`` (same order as ``X``);
    ``feature_names`` labels the columns; ``channels`` is the shared channel set."""
    X: np.ndarray
    meta: pd.DataFrame
    feature_names: list[str]
    channels: list[str]

    def __len__(self) -> int:
        return self.X.shape[0]


def umap_available() -> bool:
    try:
        import umap  # noqa: F401
        return True
    except Exception:
        return False


def _standardize_per_run(df: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    """z-score each feature column within each run (batch mitigation). A run with
    a single cell, or a zero-variance column, is left centred at 0."""
    out = df.copy()
    for _run, idx in df.groupby("run_tag").groups.items():
        sub = df.loc[idx, feat_cols].astype(float)
        mu = sub.mean()
        sd = sub.std(ddof=0).replace(0.0, 1.0)
        out.loc[idx, feat_cols] = ((sub - mu) / sd).fillna(0.0)
    return out


def build_feature_matrix(
    run_tables: dict[str, RunTable],
    params: ClassifierParams | None = None,
) -> FeatureMatrix | None:
    """Assemble the per-cell feature matrix across one or more loaded runs.

    Features per cell = ``score_<ch>`` (population-scaled) + ``occ_<ch>`` for each
    channel *shared* by all runs, aligned by name. Only wells with a hole are
    included. Returns ``None`` if there is nothing to embed."""
    params = params or ClassifierParams()
    frames: list[tuple[str, pd.DataFrame]] = []
    channel_sets: list[set] = []
    for run_tag, rt in run_tables.items():
        if rt is None or not rt.channels:
            continue
        cls = classify_wells(rt, params)
        if cls.empty:
            continue
        cls = cls[cls["hole_present"]]
        if cls.empty:
            continue
        frames.append((run_tag, cls))
        channel_sets.append(set(rt.channels))
    if not frames:
        return None

    shared = sorted(set.intersection(*channel_sets)) if channel_sets else []
    if not shared:
        return None
    feat_cols: list[str] = []
    for ch in shared:
        feat_cols += [f"score_{ch}", f"occ_{ch}"]

    rows = []
    for run_tag, cls in frames:
        for r in cls.itertuples(index=False):
            rec = {"run_tag": run_tag, "plate": r.plate, "well_id": r.well_id}
            for ch in shared:
                rec[f"score_{ch}"] = float(getattr(r, f"score_{ch}"))
                rec[f"occ_{ch}"] = float(getattr(r, f"occ_{ch}"))
            rows.append(rec)
    df = pd.DataFrame(rows)
    dfz = _standardize_per_run(df, feat_cols)
    X = np.nan_to_num(dfz[feat_cols].to_numpy(dtype=float), nan=0.0)
    meta = df[["run_tag", "plate", "well_id"]].reset_index(drop=True)
    return FeatureMatrix(X=X, meta=meta, feature_names=feat_cols, channels=shared)


def embed(X: np.ndarray, method: str = "pca", random_state: int = 0) -> np.ndarray:
    """2-D embedding of ``X``. ``method`` = ``"pca"`` (sklearn) or ``"umap"``
    (optional dep; raises ImportError with an install hint if absent). Fewer than
    3 rows → returns the raw first two columns padded to 2-D."""
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
        reducer = umap.UMAP(n_components=2, random_state=random_state,
                            n_neighbors=n_neighbors)
        return reducer.fit_transform(X)
    raise ValueError(f"unknown method {method!r}")


def cluster_labels(X: np.ndarray, min_cluster_size: int = 15) -> np.ndarray:
    """Unbiased HDBSCAN clustering (no preset cluster count). Noise points get
    label ``-1``. ``min_cluster_size`` is clamped to a sane range for small n."""
    from sklearn.cluster import HDBSCAN
    n = X.shape[0]
    if n < 3:
        return np.zeros(n, dtype=int)
    mcs = int(max(2, min(min_cluster_size, max(2, n // 2))))
    return HDBSCAN(min_cluster_size=mcs).fit_predict(X)
