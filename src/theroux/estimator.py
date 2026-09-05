"""
Baseline estimation under thin per-speaker data.

Two problems v0 papered over, both of which matter once the corpus is real:

1. CORRELATED DIMENSIONS. Hedging and specificity-avoidance co-move. Treating
   them as independent z-scores double-counts shared variance and inflates the
   composite. The fix is a covariance-aware distance (see `anomaly.py`), which
   requires a covariance estimate that is actually invertible.

2. THIN SAMPLES. A speaker with three baseline statements cannot support a
   6x6 sample covariance — it is singular, and the sample mean is high-variance.
   Two standard corrections:

   - Ledoit-Wolf shrinkage of the covariance toward a structured target
     (Ledoit & Wolf, 2004). Same estimator used for portfolio covariance in
     finance for exactly the same reason: p is large relative to n.

   - Empirical-Bayes / James-Stein shrinkage of each speaker's *mean* toward
     the population mean, with weight determined by how much of the observed
     between-speaker spread is real signal versus sampling noise. A speaker with
     20 baselines keeps their own mean; a speaker with 2 borrows strength from
     the population. This is the principled version of "trust thin baselines
     less" — it is not a heuristic, it minimises expected squared error.

Both shrinkage intensities are reported per speaker so the analyst can see how
much of a baseline is the speaker's own history versus borrowed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.covariance import LedoitWolf


@dataclass
class SpeakerBaseline:
    speaker: str
    n: int
    features: list[str]
    raw_mean: np.ndarray
    mean: np.ndarray                 # empirical-Bayes shrunk
    cov: np.ndarray                  # Ledoit-Wolf shrunk
    precision: np.ndarray            # inverse covariance
    cov_shrinkage: float             # Ledoit-Wolf intensity, 0-1
    mean_shrinkage: float            # EB weight on the population mean, 0-1
    diag: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "speaker": self.speaker,
            "n": self.n,
            "features": self.features,
            "mean": [round(float(v), 4) for v in self.mean],
            "raw_mean": [round(float(v), 4) for v in self.raw_mean],
            "sd": [round(float(v), 4) for v in np.sqrt(np.diag(self.cov))],
            "cov_shrinkage": round(float(self.cov_shrinkage), 3),
            "mean_shrinkage": round(float(self.mean_shrinkage), 3),
            **self.diag,
        }


def _ledoit_wolf(X: np.ndarray) -> tuple[np.ndarray, float]:
    """Shrunk covariance. Falls back to a ridged diagonal when n is tiny."""
    n, p = X.shape
    if n < 3:
        var = X.var(axis=0, ddof=0) if n > 1 else np.ones(p)
        cov = np.diag(np.maximum(var, 1e-3))
        return cov, 1.0
    lw = LedoitWolf(assume_centered=False).fit(X)
    cov = lw.covariance_.copy()
    # Ridge for numerical safety — the shrinkage target can still be near-singular
    cov += np.eye(p) * 1e-4
    return cov, float(lw.shrinkage_)


def empirical_bayes_weight(n: int, within_var: np.ndarray,
                           between_var: np.ndarray) -> float:
    """
    James-Stein / hierarchical weight on the speaker's own mean.

        w = tau^2 / (tau^2 + sigma^2 / n)

    tau^2 = between-speaker variance of true means (signal)
    sigma^2 = within-speaker variance (noise)

    n large or between-speaker spread large  -> trust the speaker's own mean.
    n small or speakers all alike            -> shrink toward the population.
    """
    tau2 = np.maximum(between_var - within_var / max(n, 1), 1e-6)
    w = tau2 / (tau2 + within_var / max(n, 1))
    return float(np.clip(np.mean(w), 0.0, 1.0))


def fit_baselines(
    rows_by_speaker: dict[str, np.ndarray],
    feature_names: list[str],
    min_n: int = 2,
) -> dict[str, SpeakerBaseline]:
    """
    rows_by_speaker: speaker -> (n_baseline_statements, n_features) matrix.
    Returns fitted baselines with both shrinkage corrections applied.
    """
    eligible = {s: X for s, X in rows_by_speaker.items() if len(X) >= min_n}
    if not eligible:
        return {}

    # Population statistics for the hierarchical prior
    speaker_means = np.vstack([X.mean(axis=0) for X in eligible.values()])
    pop_mean = speaker_means.mean(axis=0)
    between_var = speaker_means.var(axis=0, ddof=1) if len(speaker_means) > 1 \
        else np.ones(speaker_means.shape[1])
    within_var = np.mean(
        [X.var(axis=0, ddof=1) if len(X) > 1 else np.ones(X.shape[1])
         for X in eligible.values()], axis=0
    )

    out: dict[str, SpeakerBaseline] = {}
    for sp, X in eligible.items():
        n = len(X)
        raw_mean = X.mean(axis=0)
        w = empirical_bayes_weight(n, within_var, between_var)
        mean = w * raw_mean + (1 - w) * pop_mean

        cov, lw_intensity = _ledoit_wolf(X)
        precision = np.linalg.pinv(cov)

        out[sp] = SpeakerBaseline(
            speaker=sp, n=n, features=feature_names,
            raw_mean=raw_mean, mean=mean, cov=cov, precision=precision,
            cov_shrinkage=lw_intensity, mean_shrinkage=round(1 - w, 3),
            diag={"pop_mean": [round(float(v), 4) for v in pop_mean]},
        )
    return out
