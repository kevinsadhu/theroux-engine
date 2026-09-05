"""
Multivariate anomaly scoring with calibrated probabilities.

v0 reported a weighted sum of independent z-scores. That has three defects:

  - it ignores correlation between dimensions, double-counting shared variance;
  - "6.8 sigma" is not interpretable when the dimensions are not independent;
  - the weights were hand-picked, so the composite encoded our priors, not data.

This module replaces it with squared Mahalanobis distance against the speaker's
own shrunk covariance:

    D^2(x) = (x - mu_s)^T  Sigma_s^-1  (x - mu_s)

Under approximate multivariate normality D^2 ~ chi^2_k, which converts the
distance into a **p-value**: the probability of observing language at least this
far from the speaker's own norm by chance. That is the number an analyst can
actually act on — "99.4th percentile of this speaker's own distribution" beats
"6.8 sigma" because it is calibrated and unit-free.

We also decompose the distance. Mahalanobis is a single scalar, which is useless
for the audit trail, so we attribute it back to dimensions two ways:

  - marginal z: (x_i - mu_i) / sd_i, the naive per-dimension view
  - whitened contribution: the element-wise share of D^2 after decorrelation,
    which answers "which dimension actually drove the distance once correlation
    is removed" — often a different answer from the marginal z.

The chi-square assumption is not free. With small baselines the tail is heavier
than chi-square and p-values will be optimistic. `validate.py` therefore also
computes a permutation-based empirical p-value, and the two are reported side by
side. When they disagree, believe the permutation.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from .estimator import SpeakerBaseline


def mahalanobis(x: np.ndarray, base: SpeakerBaseline) -> float:
    d = x - base.mean
    return float(d @ base.precision @ d)


def score(x: np.ndarray, base: SpeakerBaseline) -> dict:
    """Full anomaly record for one statement against one speaker baseline."""
    d = x - base.mean
    d2 = float(d @ base.precision @ d)
    k = len(base.mean)

    # chi-square tail probability: how unusual is this, under normality
    p_chi2 = float(stats.chi2.sf(d2, df=k))

    # Whitened contributions: L^T d where Sigma^-1 = L L^T (Cholesky of precision)
    try:
        L = np.linalg.cholesky(base.precision)
        wcontrib = (L.T @ d) ** 2
    except np.linalg.LinAlgError:
        wcontrib = (d ** 2) * np.diag(base.precision)
    share = wcontrib / wcontrib.sum() if wcontrib.sum() > 0 else np.zeros_like(wcontrib)

    sd = np.sqrt(np.diag(base.cov))
    marginal_z = d / np.maximum(sd, 1e-6)

    return {
        "d2": round(d2, 4),
        "mahalanobis": round(float(np.sqrt(d2)), 4),
        "p_chi2": float(f"{p_chi2:.6g}"),
        "percentile": round(float((1 - p_chi2) * 100), 3),
        "marginal_z": {f: round(float(z), 3) for f, z in zip(base.features, marginal_z)},
        "attribution": {f: round(float(s), 4) for f, s in zip(base.features, share)},
        "dominant_feature": base.features[int(np.argmax(share))],
        "k": k,
    }


def directional_component(x: np.ndarray, base: SpeakerBaseline,
                          direction: np.ndarray) -> float:
    """
    Signed projection of the deviation onto a learned direction, whitened.

    Mahalanobis distance is unsigned — it cannot distinguish "unusually evasive"
    from "unusually forthcoming". Once `model.py` learns a discriminative
    direction from outcomes, this projects the deviation onto it to recover sign
    and magnitude in a single interpretable number.
    """
    d = x - base.mean
    w = base.precision @ direction
    denom = float(np.sqrt(direction @ base.precision @ direction))
    return float(d @ w / denom) if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# A-priori signed score.
#
# Mahalanobis is unsigned: a speaker who becomes unusually *direct* scores as
# high as one who becomes unusually evasive. For a directional hypothesis that
# is the wrong statistic.
#
# This projects the deviation onto a direction fixed in advance by the research
# literature — more hedging, less specificity, more distancing, more deflection
# — whitened by the speaker's own covariance so correlated dimensions are not
# double counted. Because the direction is asserted a priori rather than fitted,
# it cannot overfit, and it is the honest primary signal at small n.
# ---------------------------------------------------------------------------

EVASION_DIRECTION = {
    "hedging": 1.0,
    "specificity_avoidance": 1.0,
    "pronoun_distancing": 0.8,
    "topic_deflection": 0.9,
    "confidence_language": 0.3,
    "surprisal_z": 0.6,
}


def evasion_score(x: np.ndarray, base: SpeakerBaseline) -> float:
    """Signed, whitened projection onto the a-priori evasion direction."""
    v = np.array([EVASION_DIRECTION.get(f, 0.0) for f in base.features])
    return directional_component(x, base, v)
