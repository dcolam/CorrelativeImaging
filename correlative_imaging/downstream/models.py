"""Particle models: P(cell vs noise) and P(neighbour), logistic, trained on hand-labelled particles.

Features (colour-neutral, so the colour call stays independent of them):
  P(cell)       asinh(SNR/2), log10(area inside the hole), channel dummies, asinh(SNR/2) x channel
  P(neighbour)  logit(share of the particle inside the hole), log(coverage of the hole)

Fitting uses sklearn's LogisticRegression with its defaults (as the reference workflow_comparison/models.py);
the coefficients are stored as JSON with their provenance, and prediction needs only numpy.

Label file columns: particle_label, snr, area_um2, channel, share_inside, fill.
"""
from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

CELL_LABELS = ["patched cell", "maybe cell", "neighbour", "two fused cells"]
NOISE_LABEL = "background_noise"
NB_FIT_LABELS = ["neighbour", "patched cell", "maybe cell"]


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


def cell_features(snr, area_um2, ch, channels: list) -> np.ndarray:
    a = np.arcsinh(np.asarray(snr, float) / 2)
    la = np.log10(np.clip(np.asarray(area_um2, float), .5, None))
    ch = np.asarray(ch)
    dummies = [(ch == c) * 1.0 for c in channels[1:]]
    return np.c_[a, la, *dummies, *[a * d for d in dummies]]


def nb_features(share, fill) -> np.ndarray:
    return np.c_[_logit(share), np.log(np.clip(np.asarray(fill, float), 1e-3, None))]


class ParticleModels:
    def __init__(self, spec: dict):
        self.spec = spec
        self.channels = spec["channels"]

    @staticmethod
    def _p(m, X):
        return 1 / (1 + np.exp(-(m["intercept"] + X @ np.asarray(m["coef"]))))

    def p_cell(self, snr, area_um2, ch):
        return self._p(self.spec["cell"], cell_features(snr, area_um2, ch, self.channels))

    def p_nb(self, share, fill):
        return self._p(self.spec["nb"], nb_features(share, fill))

    @classmethod
    def load(cls, path: str | Path) -> "ParticleModels":
        p = Path(path)
        if not p.is_absolute():
            p = Path(__file__).parent / p
        return cls(json.loads(p.read_text()))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.spec, indent=2))


def fit_models(labels: pd.DataFrame | str | Path, channels: list, cv: int = 5, source: str = "") -> ParticleModels:
    """Fit both models on hand labels; reports cross-validated AUC (never on training data)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    if not isinstance(labels, pd.DataFrame):
        source = source or str(labels)
        labels = pd.read_csv(labels)
    lab = labels
    nc = lab[lab.particle_label.isin(CELL_LABELS + [NOISE_LABEL])]
    Xc, yc = cell_features(nc.snr, nc.area_um2, nc.channel, channels), nc.particle_label.isin(CELL_LABELS).values
    nb = lab[lab.particle_label.isin(NB_FIT_LABELS)]
    Xn, yn = nb_features(nb.share_inside, nb.fill), (nb.particle_label == "neighbour").values
    mc = LogisticRegression(max_iter=1000).fit(Xc, yc)
    mn = LogisticRegression().fit(Xn, yn)
    auc = lambda m, X, y: float(np.mean(cross_val_score(type(m)(**m.get_params()), X, y, cv=cv, scoring="roc_auc")))
    spec = dict(
        channels=list(channels),
        cell=dict(intercept=float(mc.intercept_[0]), coef=mc.coef_[0].tolist(), n_cell=int(yc.sum()),
                  n_noise=int((~yc).sum()), cv_auc=auc(mc, Xc, yc),
                  features="asinh(snr/2), log10(area_um2), dummies(channels[1:]), asinh x dummies"),
        nb=dict(intercept=float(mn.intercept_[0]), coef=mn.coef_[0].tolist(), n_neighbour=int(yn.sum()),
                n_other=int((~yn).sum()), cv_auc=auc(mn, Xn, yn), features="logit(share_inside), log(fill)"),
        provenance=dict(labels=source, labels_sha1=hashlib.sha1(pd.util.hash_pandas_object(lab).values).hexdigest(),
                        fitted=datetime.datetime.now().isoformat(timespec="seconds"),
                        sklearn=__import__("sklearn").__version__))
    return ParticleModels(spec)
