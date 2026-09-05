"""
Confound control.

Before any feature is compared against a baseline, it has to be purged of
variation that has nothing to do with disclosure constraint. Three confounds
matter here and all three are mechanical rather than behavioural:

  LENGTH.       Rate features are per-100-words, but they are not length-neutral.
                Short media answers have fewer opportunities for numeric markers
                than a long prepared Q&A, so specificity-avoidance rises with
                brevity for reasons that have nothing to do with what the speaker
                knows. Uncontrolled, the model learns "media interviews are
                short" and calls it a signal.

  EVENT TYPE — DELIBERATELY EXCLUDED. Formats differ in register, and the naive
                move is to residualise them away. We do not, and the reason is
                important: in the corporate corpus, format is almost perfectly
                collinear with the treatment. Constrained windows ARE media hits;
                baselines ARE earnings calls. Residualising on format therefore
                removes the effect we are trying to measure along with the
                nuisance — we verified this empirically, and it drove the signal
                to chance. Format is a confound we cannot control by regression;
                it has to be controlled by DESIGN, with a corpus where both
                windows share a format. Federal Reserve speeches do exactly that:
                the same speaker gives the same kind of speech inside and outside
                the FOMC blackout. That is a scientific argument for the Fed
                corpus, not merely a convenience one.

  TIME.         Corpora drift. Vocabulary and disclosure norms move over years,
                and speaker baselines built from older statements inherit that.

We residualise each feature on these covariates by OLS and score the residual.
Any signal that survives is variation the covariates cannot explain.

This is the least glamorous module in the repo and the one most likely to be the
difference between a real result and a spurious one.
"""

from __future__ import annotations

import numpy as np


def build_design(rows: list[dict], event_types: list[str],
                 t0: float | None = None) -> tuple[np.ndarray, list[str]]:
    """Design matrix: intercept, log length, event-type dummies, time trend."""
    n = len(rows)
    log_len = np.array([np.log(max(r["meta"]["words"], 1)) for r in rows])
    log_len = (log_len - log_len.mean()) / max(log_len.std(), 1e-6)

    cols = [np.ones(n), log_len]
    names = ["intercept", "log_words"]

    # NOTE: event-type dummies are intentionally absent. See module docstring —
    # format is collinear with the treatment in this corpus, so regressing it out
    # destroys the signal. Control it by corpus design instead.

    ts = np.array([r.get("_t", 0.0) for r in rows], dtype=float)
    if t0 is None:
        t0 = ts.mean()
    trend = (ts - t0) / max(ts.std(), 1e-6)
    cols.append(trend)
    names.append("time")

    return np.column_stack(cols), names


def fit_residualiser(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """OLS coefficients (ridge-stabilised) mapping covariates -> each feature."""
    XtX = X.T @ X + np.eye(X.shape[1]) * 1e-6
    return np.linalg.solve(XtX, X.T @ Y)


def residualise(X: np.ndarray, Y: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """
    Feature residuals, re-centred on the corpus mean.

    Recentring keeps residuals on roughly the original scale so downstream
    numbers stay interpretable — a residual of 0.4 on specificity-avoidance
    still reads as "fairly vague", not as an abstract deviation.
    """
    fitted = X @ beta
    return Y - fitted + Y.mean(axis=0)


def variance_explained(Y: np.ndarray, resid: np.ndarray) -> np.ndarray:
    """Share of each feature's variance the confounds accounted for."""
    total = Y.var(axis=0)
    left = (resid - resid.mean(axis=0)).var(axis=0)
    return np.clip(1 - left / np.maximum(total, 1e-9), 0, 1)
