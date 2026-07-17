"""Feature-space dimensionality reduction + unbiased clustering of wells.

Qt-free layer for the Explorer's Clusters tab. One row per (run_tag, plate,
well) — a "cell" is a well's hole — described by a configurable set of per-channel
features. Produces PCA (always) and UMAP (optional dep) 2-D embeddings and
posthoc cluster labels (HDBSCAN / KMeans / Agglomerative), so the GUI can scatter
the cells, colour them, and map a clicked point back to its well.

Design decisions baked in here (see the Clusters tab for the interaction):

* **Feature families** (:data:`FEATURE_FAMILIES`) are chosen in the pop-up; each
  contributes per-channel columns. Everything is aligned by channel NAME (channel
  order varies between experiments) on the runs' *shared* channels.
* **Scaling** (:data:`SCALING_GROUPS`) z-scores each feature column within a
  row-group — global / per run / per (run×plate) / per plate. Per-run is the sane
  default for cross-run work; per-plate removes DIV-timepoint biology (warn).
  Per-column unit-variance standardisation always happens so no feature dominates
  PCA by raw scale; the grouping only decides global-vs-within-batch.
* **Clustering is done on the FEATURE matrix, not on the 2-D coords** (clustering
  on UMAP/PCA coordinates is an anti-pattern — those distances aren't faithful).
* The **fit-set vs display** split lives in the GUI: fit once on the chosen
  fit-set, then filter which points are drawn. UMAP gets a fixed ``random_state``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .analysis import RunTable
from .classify import ClassifierParams, classify_wells

# Feature family → per-channel base column names it contributes.
FEATURE_FAMILIES: dict[str, list[str]] = {
    "score": ["score"],                 # population-scaled enrichment (classifier's signal)
    "occupancy": ["occ"],               # fraction of hole covered by objects (0–1)
    "morphology": ["area", "circularity", "eccentricity", "solidity"],
    "intensity": ["hole_mean", "bg_mean", "n_particles"],
}
DEFAULT_FAMILIES = ("score",)

# Scaling grouping name → row-group columns to z-score each feature within.
SCALING_GROUPS: dict[str, list[str]] = {
    "global": [],
    "run": ["run_tag"],
    "run_plate": ["run_tag", "plate"],
    "plate": ["plate"],
}
DEFAULT_SCALING = "run_plate"


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
def _hole_particle_aggregates(particles: pd.DataFrame) -> pd.DataFrame:
    """Per (plate, well_id, channel) hole-ROI aggregates: mean morphology +
    particle count. Empty frame if no particle data."""
    cols = ["plate", "well_id", "channel", "area", "circularity",
            "eccentricity", "solidity", "n_particles"]
    if particles is None or particles.empty or "roi_mask" not in particles.columns:
        return pd.DataFrame(columns=cols)
    sub = particles[particles["roi_mask"] == "roi_hole"]
    if sub.empty:
        return pd.DataFrame(columns=cols)
    g = sub.groupby(["plate", "well_id", "channel"])
    out = g.agg(
        area=("area_px", "mean"),
        circularity=("circularity", "mean"),
        eccentricity=("eccentricity", "mean"),
        solidity=("solidity", "mean"),
        n_particles=("label", "count"),
    ).reset_index()
    return out


def _roi_mean(intensity: pd.DataFrame, roi: str, name: str) -> pd.DataFrame:
    """Per (plate, well_id, channel) mean intensity in *roi*, column *name*."""
    if intensity is None or intensity.empty or "roi_mask" not in intensity.columns:
        return pd.DataFrame(columns=["plate", "well_id", "channel", name])
    sub = intensity[intensity["roi_mask"] == roi]
    if sub.empty:
        return pd.DataFrame(columns=["plate", "well_id", "channel", name])
    g = sub.groupby(["plate", "well_id", "channel"])["mean_intensity"].mean()
    return g.rename(name).reset_index()


def _run_long_features(rt: RunTable, params: ClassifierParams, omit_empty: bool) -> pd.DataFrame:
    """Long per-(plate, well, channel) base-feature table for one run: score, occ,
    hole_mean, bg_mean, n_particles, area, circularity, eccentricity, solidity.
    Only wells with a hole (optionally only non-empty) are kept."""
    cls = classify_wells(rt, params)
    if cls.empty:
        return pd.DataFrame()
    cls = cls[cls["hole_present"]]
    if omit_empty and "is_negative_hole" in cls.columns:
        cls = cls[~cls["is_negative_hole"].astype(bool)]
    if cls.empty:
        return pd.DataFrame()

    # score + occ (per-channel columns from classify) → long, plus the classifier
    # call carried along for colour-by.
    rows = []
    for r in cls.itertuples(index=False):
        for ch in rt.channels:
            rows.append({
                "plate": r.plate, "well_id": r.well_id, "channel": ch,
                "score": float(getattr(r, f"score_{ch}")),
                "occ": float(getattr(r, f"occ_{ch}")),
            })
    base = pd.DataFrame(rows)

    morph = _hole_particle_aggregates(rt.particles)
    hole = _roi_mean(rt.intensity, "roi_hole", "hole_mean")
    bg = _roi_mean(rt.intensity, "roi_background", "bg_mean")
    for extra in (morph, hole, bg):
        if not extra.empty:
            base = base.merge(extra, on=["plate", "well_id", "channel"], how="left")
    # missing morphology/intensity → 0 (well/channel with no detected objects)
    for col in ("area", "circularity", "eccentricity", "solidity", "n_particles",
                "hole_mean", "bg_mean"):
        if col not in base.columns:
            base[col] = 0.0
        base[col] = base[col].fillna(0.0)

    call = cls[["plate", "well_id", "pos_channels", "is_negative_hole"]].copy()
    base = base.merge(call, on=["plate", "well_id"], how="left")
    base.insert(0, "run_tag", rt.run_tag)
    return base


def _standardize_within(df: pd.DataFrame, feat_cols: list[str],
                        group_cols: list[str]) -> pd.DataFrame:
    """z-score each feature column within row-groups (``group_cols`` empty =
    global). Zero-variance / single-row groups collapse to 0."""
    out = df.copy()

    def _z(block: pd.DataFrame) -> pd.DataFrame:
        mu = block.mean()
        sd = block.std(ddof=0).replace(0.0, 1.0)
        return ((block - mu) / sd).fillna(0.0)

    if group_cols:
        for _key, idx in df.groupby(group_cols).groups.items():
            out.loc[idx, feat_cols] = _z(df.loc[idx, feat_cols].astype(float))
    else:
        out[feat_cols] = _z(df[feat_cols].astype(float))
    return out


def build_feature_matrix(
    run_tables: dict[str, RunTable],
    params: ClassifierParams | None = None,
    families: tuple[str, ...] | list[str] = DEFAULT_FAMILIES,
    scaling: str = DEFAULT_SCALING,
    omit_empty: bool = False,
) -> FeatureMatrix | None:
    """Assemble the per-cell feature matrix across one or more loaded runs.

    ``families`` selects which feature families (:data:`FEATURE_FAMILIES`) to
    include; ``scaling`` picks the standardisation grouping (:data:`SCALING_GROUPS`).
    Features are taken on the channels *shared* by all runs, aligned by name.
    Returns ``None`` if there is nothing to embed or no family resolves to a
    column."""
    params = params or ClassifierParams()
    families = [f for f in families if f in FEATURE_FAMILIES] or list(DEFAULT_FAMILIES)
    bases = [b for fam in families for b in FEATURE_FAMILIES[fam]]

    longs, channel_sets = [], []
    for _run_tag, rt in run_tables.items():
        if rt is None or not rt.channels:
            continue
        lf = _run_long_features(rt, params, omit_empty)
        if lf.empty:
            continue
        longs.append(lf)
        channel_sets.append(set(rt.channels))
    if not longs:
        return None
    shared = sorted(set.intersection(*channel_sets)) if channel_sets else []
    if not shared:
        return None

    long = pd.concat(longs, ignore_index=True)
    long = long[long["channel"].isin(shared)]

    # long → wide: one column per (base, channel).
    wide = long.pivot_table(
        index=["run_tag", "plate", "well_id"], columns="channel", values=bases,
    )
    wide.columns = [f"{base}_{ch}" for base, ch in wide.columns]
    wide = wide.reset_index()
    feat_cols = [c for c in wide.columns if c not in ("run_tag", "plate", "well_id")]
    if not feat_cols:
        return None

    # classifier call per (run,plate,well) for colour-by (one value per well).
    call = (long[["run_tag", "plate", "well_id", "pos_channels", "is_negative_hole"]]
            .drop_duplicates(["run_tag", "plate", "well_id"]))
    wide = wide.merge(call, on=["run_tag", "plate", "well_id"], how="left")

    wide[feat_cols] = wide[feat_cols].fillna(0.0)
    scaled = _standardize_within(wide, feat_cols, SCALING_GROUPS.get(scaling, []))
    X = np.nan_to_num(scaled[feat_cols].to_numpy(dtype=float), nan=0.0)
    meta = wide[["run_tag", "plate", "well_id", "pos_channels", "is_negative_hole"]] \
        .reset_index(drop=True)
    return FeatureMatrix(X=X, meta=meta, feature_names=feat_cols, channels=shared)


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
