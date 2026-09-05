"""
Validation that is honest at small n.

At n in the dozens, asymptotic p-values are wrong and in-sample fit is
meaningless. This module implements the three things that keep us from fooling
ourselves, and it is deliberately the harshest code in the repo.

  PERMUTATION TESTING. Shuffle the outcome labels many times, recompute the
    statistic, and read the observed value against that null distribution. Exact
    under the null, makes no distributional assumption, and is the only credible
    significance test at this sample size.

  LEAVE-ONE-OUT CROSS-VALIDATION. Any weights learned from outcomes will fit the
    data they were learned on. LOO refits without each observation and predicts
    it, which is the smallest honest out-of-sample estimate available. We report
    in-sample and LOO side by side; a large gap is the overfitting alarm.

  CALIBRATION. A score is only useful if its magnitude means something. We bin
    predictions and compare predicted against observed rates, and report Brier
    score and expected calibration error. A model that discriminates well but is
    badly calibrated should not be shipped to an analyst as a probability.

Blocked permutation note: statements from the same speaker are not independent.
`permutation_test` therefore supports permuting *within speaker blocks*, which
preserves speaker structure under the null and avoids the inflated significance
you get from naive shuffling of clustered data.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return 0.0
    r = stats.spearmanr(x, y).statistic
    return 0.0 if np.isnan(r) else float(r)


def permutation_test(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None = None,
    n_perm: int = 20000,
    statistic=spearman,
    seed: int = 7,
) -> dict:
    """
    Two-sided permutation test. `groups` enables within-block shuffling so
    clustered observations (multiple statements per speaker) do not inflate
    significance.
    """
    rng = np.random.default_rng(seed)
    observed = statistic(x, y)
    null = np.empty(n_perm)

    if groups is None:
        for i in range(n_perm):
            null[i] = statistic(x, rng.permutation(y))
    else:
        idx_by_group = [np.where(groups == g)[0] for g in np.unique(groups)]
        for i in range(n_perm):
            yp = y.copy()
            for idx in idx_by_group:
                if len(idx) > 1:
                    yp[idx] = rng.permutation(y[idx])
            null[i] = statistic(x, yp)

    # +1 correction: never report p = 0 from a finite permutation sample
    p = (np.sum(np.abs(null) >= abs(observed)) + 1) / (n_perm + 1)
    return {
        "observed": round(float(observed), 4),
        "p_value": round(float(p), 5),
        "n_perm": n_perm,
        "null_mean": round(float(null.mean()), 4),
        "null_sd": round(float(null.std()), 4),
        "blocked": groups is not None,
        "significant_at_05": bool(p < 0.05),
    }


def bootstrap_ci(x: np.ndarray, y: np.ndarray, n_boot: int = 10000,
                 alpha: float = 0.05, seed: int = 7) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(x)
    stats_ = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        stats_[i] = spearman(x[idx], y[idx])
    return (round(float(np.quantile(stats_, alpha / 2)), 4),
            round(float(np.quantile(stats_, 1 - alpha / 2)), 4))


def loo_predict(fit_fn, predict_fn, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Leave-one-out predictions. fit_fn(X,y) -> model; predict_fn(model,x) -> float."""
    n = len(y)
    preds = np.empty(n)
    for i in range(n):
        mask = np.ones(n, bool)
        mask[i] = False
        model = fit_fn(X[mask], y[mask])
        preds[i] = predict_fn(model, X[i])
    return preds


def calibration(probs: np.ndarray, labels: np.ndarray, n_bins: int = 5) -> dict:
    """Reliability curve, Brier score, expected calibration error."""
    probs = np.clip(probs, 0, 1)
    edges = np.linspace(0, 1, n_bins + 1)
    bins = []
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (probs >= lo) & (probs < hi if hi < 1 else probs <= hi)
        if m.sum() == 0:
            continue
        pred, obs, w = probs[m].mean(), labels[m].mean(), m.sum() / len(probs)
        ece += w * abs(pred - obs)
        bins.append({"lo": round(float(lo), 2), "hi": round(float(hi), 2),
                     "n": int(m.sum()), "predicted": round(float(pred), 3),
                     "observed": round(float(obs), 3)})
    return {
        "bins": bins,
        "brier": round(float(np.mean((probs - labels) ** 2)), 4),
        "ece": round(float(ece), 4),
        "base_rate": round(float(labels.mean()), 3),
    }


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC AUC via the Mann-Whitney identity — no sklearn dependency, exact."""
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    ranks = stats.rankdata(np.concatenate([pos, neg]))
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))
