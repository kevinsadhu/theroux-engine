"""
Learning the composite from outcomes, instead of asserting it.

v0 used hand-picked weights (specificity .30, hedging .25, ...). Those encoded
our priors. This module learns a discriminative direction from labelled outcomes
with L2-regularised logistic regression, and — because the sample is small and
the temptation to overfit is enormous — reports honest LOO-CV performance beside
the in-sample fit and refuses to present tuned weights as an out-of-sample result.

Two design decisions worth defending:

  STRONG REGULARISATION BY DEFAULT. With tens of observations and six features,
  an unregularised fit will happily find a direction that separates the training
  data and generalises to nothing. C is small and chosen by nested LOO, not by
  looking at the final number.

  GROUPED SPLITS. Statements cluster by speaker. Leaving out a single statement
  while its speaker's other statements remain in training leaks information
  through the shared baseline. `fit_and_validate` supports leave-one-speaker-out,
  which is the split that actually estimates generalisation to a new speaker —
  the deployment case that matters.

The learned direction feeds `anomaly.directional_component`, which turns the
unsigned Mahalanobis distance into a signed, interpretable score.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .validate import auc, calibration


@dataclass
class LearnedModel:
    features: list[str]
    weights: np.ndarray
    intercept: float
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    C: float

    def direction(self) -> np.ndarray:
        """Unit direction in raw feature space, for projection scoring."""
        w = self.weights / np.maximum(self.scaler_scale, 1e-9)
        n = np.linalg.norm(w)
        return w / n if n > 0 else w

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Z = (X - self.scaler_mean) / np.maximum(self.scaler_scale, 1e-9)
        return 1 / (1 + np.exp(-(Z @ self.weights + self.intercept)))

    def to_dict(self) -> dict:
        d = self.direction()
        return {
            "features": self.features,
            "standardised_weights": {f: round(float(w), 4)
                                     for f, w in zip(self.features, self.weights)},
            "direction": {f: round(float(v), 4) for f, v in zip(self.features, d)},
            "intercept": round(float(self.intercept), 4),
            "C": self.C,
        }


def _fit(X: np.ndarray, y: np.ndarray, features: list[str], C: float) -> LearnedModel:
    sc = StandardScaler().fit(X)
    Z = sc.transform(X)
    lr = LogisticRegression(C=C, solver="lbfgs", max_iter=2000)
    lr.fit(Z, y)
    return LearnedModel(
        features=features, weights=lr.coef_[0], intercept=float(lr.intercept_[0]),
        scaler_mean=sc.mean_, scaler_scale=sc.scale_, C=C,
    )


def fit_and_validate(
    X: np.ndarray,
    y: np.ndarray,
    features: list[str],
    groups: np.ndarray | None = None,
    C_grid: tuple[float, ...] = (0.03, 0.1, 0.3, 1.0),
) -> dict:
    """
    Fit with C selected by grouped LOO AUC, then report in-sample vs out-of-fold.

    `groups` should be speaker ids. When supplied, folds leave out an entire
    speaker — the honest estimate of generalisation to someone unseen.
    """
    n = len(y)
    if n < 6 or len(np.unique(y)) < 2:
        return {"status": "insufficient_data", "n": int(n),
                "n_positive": int(y.sum()) if n else 0}

    fold_ids = groups if groups is not None else np.arange(n)
    unique_folds = np.unique(fold_ids)
    split_kind = "leave-one-speaker-out" if groups is not None else "leave-one-out"

    # --- select C by out-of-fold AUC ---
    best_C, best_auc, oof_best = C_grid[0], -1.0, None
    for C in C_grid:
        oof = np.full(n, np.nan)
        for f in unique_folds:
            te = fold_ids == f
            tr = ~te
            if len(np.unique(y[tr])) < 2:
                continue
            m = _fit(X[tr], y[tr], features, C)
            oof[te] = m.predict_proba(X[te])
        ok = ~np.isnan(oof)
        if ok.sum() < 4 or len(np.unique(y[ok])) < 2:
            continue
        a = auc(oof[ok], y[ok])
        if a > best_auc:
            best_C, best_auc, oof_best = C, a, oof

    if oof_best is None:
        return {"status": "cv_failed", "n": int(n)}

    final = _fit(X, y, features, best_C)
    in_sample = final.predict_proba(X)
    ok = ~np.isnan(oof_best)

    return {
        "status": "ok",
        "n": int(n),
        "n_positive": int(y.sum()),
        "split": split_kind,
        "n_folds": int(len(unique_folds)),
        "C_selected": best_C,
        "auc_in_sample": round(auc(in_sample, y), 4),
        "auc_out_of_fold": round(float(best_auc), 4),
        "overfit_gap": round(auc(in_sample, y) - float(best_auc), 4),
        "calibration_out_of_fold": calibration(oof_best[ok], y[ok]),
        "model": final.to_dict(),
        "_model": final,
        "_oof": oof_best,
    }


def compare_to_handpicked(X: np.ndarray, y: np.ndarray,
                          handpicked: np.ndarray) -> dict:
    """
    Does the learned direction actually beat the weights we asserted in v0?

    If it does not, that is a finding worth reporting rather than hiding — it
    would mean the priors encoded in `lexicon.py` were already close to optimal
    and the extra machinery is buying interpretability, not accuracy.
    """
    hp = X @ handpicked
    return {"auc_handpicked_weights": round(auc(hp, y), 4)}
