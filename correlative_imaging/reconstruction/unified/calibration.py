"""Does more data still help? Every quantity the first pass learns is learnt again on random subsets of growing size
(DRAWS draws per size); the median and the 5-95 % band per size show whether the estimate has settled.

  channel cut      per colour channel, hole cells (x = number of hole cells): the cut on the measurement the call uses
  bleed-through    per channel pair, surrounding exemplars (x = number of images)
  cell size        per colour channel, cells measured for one cell body (x = number of cells)
  label model      your labels (x = number of labels): accuracy on the labels left out of each draw; at least 3 per
                   label in every draw. Rising at the right end = labelling more still helps

Flat and narrow at the right end = enough data. The text under each plot gives the change over the last doubling and
the spread at the largest subset (both in the quantity's own units).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DRAWS = 15
MEASURE_COL = {"own signal": "own_{}", "hole signal": "hsig_{}", "particle signal": "part_{}"}


def sizes(n: int, start: int) -> list:
    """start, 2 x start, 4 x start, ... below n, then n."""
    out, k = [], start
    while k < n:
        out.append(k)
        k *= 2
    return out + [n] if n >= start else []


def curve(n: int, fn, start: int, draws: int = DRAWS, seed: int = 0, top: int | None = None) -> pd.DataFrame:
    """fn(indices) -> value. Rows: n, median, p5, p95 (the full set once; draws without replacement)."""
    rng = np.random.default_rng(seed)
    rows = []
    for k in sizes(top or n, start):
        vals = []
        for _ in range(1 if k == n else draws):
            idx = np.arange(n) if k == n else rng.choice(n, k, replace=False)
            try:
                v = fn(idx)
            except Exception:                                                # noqa: BLE001
                v = np.nan
            if v is not None and np.isfinite(v):
                vals.append(float(v))
        if vals:
            rows.append(dict(n=k, median=float(np.median(vals)), p5=float(np.percentile(vals, 5)),
                             p95=float(np.percentile(vals, 95)), draws=len(vals)))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ quantities
def cut_curves(H: pd.DataFrame, cal: dict, colour: list) -> dict:
    from .field import two_groups
    out = {}
    V = H[H.valid]
    for c in colour:
        col = MEASURE_COL.get(cal.get(c, {}).get("measure", ""), "").format(c)
        if col not in V:
            continue
        v = V[col].to_numpy(float)
        v = v[np.isfinite(v)]

        def fn(idx, v=v):
            cut, D = two_groups(v[idx])
            return max(5.0, cut) if (cut is not None and D >= 2) else 5.0
        out[c] = curve(len(v), fn, 50)
    return out


def bleed_curves(O: pd.DataFrame, colour: list) -> dict:
    from .field import copy_fraction
    imgs = O[["plate_id", "image_id"]].drop_duplicates().reset_index(drop=True)
    key = pd.MultiIndex.from_frame(O[["plate_id", "image_id"]])
    pos = pd.Series(np.arange(len(imgs)), index=pd.MultiIndex.from_frame(imgs))
    img_of = pos.reindex(key).to_numpy()
    Y0 = np.clip(O[[f"y_{c}" for c in colour]].to_numpy(float), 0, None)
    seg = O.channel.map({c: i for i, c in enumerate(colour)}).to_numpy()
    ex = O.exemplar.to_numpy(bool)
    out = {}
    for a in range(len(colour)):
        for b in range(len(colour)):
            if a == b:
                continue

            def fn(idx, a=a, b=b):
                m = ex & (seg == a) & np.isfinite(Y0).all(1) & np.isin(img_of, idx)
                if m.sum() < 30:
                    return np.nan
                m &= Y0[:, a] > np.median(Y0[m, a])
                return 100 * copy_fraction(Y0[m, b] / Y0[m, a]) if m.sum() >= 30 else np.nan
            out[(colour[a], colour[b])] = curve(len(imgs), fn, 25)
    return out


def size_curves(C: pd.DataFrame, colour: list) -> dict:
    out = {}
    for c in colour:
        a = C.loc[C.channel == c, "area_core"].dropna().to_numpy(float) if len(C) else np.array([])
        if len(a) >= 10:
            out[c] = curve(len(a), lambda idx, a=a: float(np.median(a[idx])), 5)
    return out


def label_curve(X: np.ndarray, y: np.ndarray, seed: int = 0) -> pd.DataFrame:
    """Accuracy on the labels left out, for growing numbers of training labels (stratified, >= 3 per label)."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    y = np.asarray(y)
    labs = np.unique(y)
    n = len(y)
    hold = max(10, n // 5)                        # always keep some labels out to test on
    if n - hold < 3 * len(labs):
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    rows = []
    for k in sizes(n - hold, 3 * len(labs)):
        acc = []
        for _ in range(DRAWS):
            tr = np.concatenate([rng.choice(np.nonzero(y == l)[0], 3, replace=False) for l in labs])
            rest = np.setdiff1d(np.arange(n), tr)
            tr = np.concatenate([tr, rng.choice(rest, k - len(tr), replace=False)]) if k > len(tr) else tr
            te = np.setdiff1d(np.arange(n), tr)
            m = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(X[tr], y[tr])
            acc.append(100 * float(np.mean(m.predict(X[te]) == y[te])))
        rows.append(dict(n=k, median=float(np.median(acc)), p5=float(np.percentile(acc, 5)),
                         p95=float(np.percentile(acc, 95)), draws=len(acc)))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------------ text
def verdict(D: pd.DataFrame, unit: str = "") -> str:
    """Change of the median over the last doubling and the spread at the largest subset."""
    if len(D) < 2:
        return "too little data for a curve"
    last, prev = D.iloc[-1], D.iloc[-2]
    sub = D[D.draws > 1]
    spread = (sub.iloc[-1].p95 - sub.iloc[-1].p5) if len(sub) else np.nan
    return (f"last step {int(prev.n)} -> {int(last.n)}: {prev['median']:.3g} -> {last['median']:.3g}{unit}; "
            f"spread at {int(sub.iloc[-1].n) if len(sub) else '-'}: {spread:.3g}{unit}")


def compute(H, O, C, cal, colour, linfo) -> dict:
    """{group: {name: DataFrame}} for the report and summary.json."""
    out = {"channel cut (noise units)": {}, "bleed-through (%)": {}, "cell size (um2)": {}, "label model (% right)": {}}
    for c, D in cut_curves(H, cal, colour).items():
        out["channel cut (noise units)"][c] = D
    for (a, b), D in bleed_curves(O, colour).items():
        if len(D) and D["median"].iloc[-1] > 0.5:          # only pairs with a copy worth showing
            out["bleed-through (%)"][f"{a} -> {b}"] = D
    for c, D in size_curves(C, colour).items():
        out["cell size (um2)"][c] = D
    if linfo.get("used") and linfo.get("X") is not None:
        D = label_curve(linfo["X"], linfo["y"])
        if len(D):
            out["label model (% right)"]["your labels"] = D
    return {g: v for g, v in out.items() if v}


def figure(group: str, curves: dict, names: dict, colors: dict, xlabel: str):
    """Small multiples, one per curve: median line, 5-95 % band, log x."""
    import matplotlib.pyplot as plt
    k = len(curves)
    ncol = min(k, 3)
    nrow = int(np.ceil(k / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 2.7 * nrow), squeeze=False)
    for a, (name, D) in zip(axes.ravel(), curves.items()):
        first = name.split(" -> ")[0]
        col = colors.get(first, "#1c2230")
        a.fill_between(D.n, D.p5, D.p95, color=col, alpha=0.18, linewidth=0)
        a.plot(D.n, D["median"], color=col, lw=2, marker="o", ms=3)
        a.set_xscale("log")
        a.set_title(" -> ".join(names.get(x, x) for x in name.split(" -> ")), fontsize=9)
        a.set_xlabel(xlabel, fontsize=8)
        a.tick_params(labelsize=7)
        a.grid(alpha=0.25)
        last = D.iloc[-1]
        a.annotate(f"{last['median']:.3g}", (last.n, last["median"]), textcoords="offset points", xytext=(-4, 6),
                   ha="right", fontsize=8)
    for a in axes.ravel()[k:]:
        a.axis("off")
    fig.suptitle(group, fontsize=10)
    fig.tight_layout()
    return fig


XLABEL = {"channel cut (noise units)": "hole cells", "bleed-through (%)": "images", "cell size (um2)": "cells",
          "label model (% right)": "your labels used for learning"}
