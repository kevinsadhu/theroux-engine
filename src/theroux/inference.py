"""
Simulation-based inference: from an observed divergence to a distribution over what
is being withheld.

The dashboard currently answers "how unusual is this statement". The question an
analyst actually has is "how much is being held back, and what is the spread of
outcomes consistent with that". Those are different questions and only the second
supports a decision.

We cannot answer the second with a regression at n=42 — the out-of-fold calibration
of our fitted model is Brier 0.28 / ECE 0.20, which means its probabilities are not
probabilities. So we do the thing you do when you trust your generative model more
than you trust your sample: **approximate Bayesian computation**. Simulate a large
population of speakers from `world.simulate`, keep the ones whose measured
divergence looks like the one in front of you, and read off the distribution of
their hidden state.

    prior over theta  ->  simulate  ->  measure  ->  keep the near-matches
                                                     ->  posterior over theta

THE HONEST CAVEAT, WHICH MUST TRAVEL WITH EVERY NUMBER THIS MODULE PRODUCES:
this posterior is conditional on the world model being right. It is not an
empirical probability. Its value is that the assumptions are explicit and
attackable — a reviewer can change `GameParams` and watch the answer move, which is
strictly more honest than a fitted probability whose assumptions are buried in a
training set. Where it surfaces in the UI it is labelled "model-conditional", and
the reference table's parameters are shown alongside it.
"""

from __future__ import annotations

import numpy as np

from . import features as F, wargame as WG, world as W


class ReferenceTable:
    """
    A simulated population, measured the same way real statements are measured.

    Built once and reused: the expensive part is scoring, not matching. Holds the
    per-window feature z-vector, the signed evasion projection, the latent state
    that produced them, and the realised move.
    """

    def __init__(self, Z, evasion, theta_held, move, params: W.GameParams, n_runs: int):
        self.Z, self.evasion = Z, evasion
        self.theta_held, self.move = theta_held, move
        self.params, self.n_runs = params, n_runs
        self.feats = F.FEATURES
        # Scale each feature by its own spread so the distance metric is not
        # dominated by whichever feature happens to have the largest variance.
        self.scale = np.maximum(Z.std(axis=0), 1e-6)

    @property
    def n(self) -> int:
        return len(self.theta_held)


def build_reference(params: W.GameParams | None = None, n_runs: int = 14,
                    seed: int = 2200) -> ReferenceTable:
    """Simulate and score a reference population. Minutes, not seconds."""
    params = params or W.GameParams()
    Z, ev, th, mv = [], [], [], []
    for rep in range(n_runs):
        s = WG.score_run(WG._prep(W.simulate(params, seed + rep * 37)))
        if not len(s["held"]):
            continue
        Z.append(s["Z"]); ev.append(s["evasion"])
        th.append(s["held"]); mv.append(s["move"])
    return ReferenceTable(np.vstack(Z), np.concatenate(ev), np.concatenate(th),
                          np.concatenate(mv), params, n_runs)


def posterior(observed_z: dict[str, float], ref: ReferenceTable,
              keep_frac: float = 0.02, material_pct: float = 6.0) -> dict:
    """
    Posterior over withheld information, given one window's measured divergence.

    Kernel-weighted ABC: distance in scaled feature space, keep the closest
    `keep_frac`, weight by an Epanechnikov kernel. Returns the posterior over
    `theta_held` plus the implied distribution over the realised move — the latter
    being the thing that was asked for as "a distribution of outcomes", and the
    thing that must never be shown without the model-conditional label.
    """
    x = np.array([observed_z.get(f, 0.0) for f in ref.feats], float)
    d = np.linalg.norm((ref.Z - x) / ref.scale, axis=1)

    k = max(40, int(keep_frac * ref.n))
    idx = np.argsort(d)[:k]
    dk = d[idx]
    h = dk.max() if dk.max() > 1e-9 else 1.0
    wgt = np.clip(1.0 - (dk / h) ** 2, 1e-9, None)      # Epanechnikov
    wgt = wgt / wgt.sum()

    th, mv = ref.theta_held[idx], ref.move[idx]
    finite = np.isfinite(mv)

    def wq(v, w, q):
        o = np.argsort(v); v, w = v[o], w[o]
        c = np.cumsum(w) / w.sum()
        return float(np.interp(q, c, v))

    p_material = float(wgt[finite] @ (np.abs(mv[finite]) >= material_pct)) \
        if finite.any() else float("nan")
    p_null = float(wgt @ (th <= 1e-6))

    return {
        "n_matched": int(k),
        "match_quality": round(float(dk.mean()), 4),
        "theta_held": {
            "mean": round(float(wgt @ th), 4),
            "q10": round(wq(th, wgt, .10), 4),
            "q50": round(wq(th, wgt, .50), 4),
            "q90": round(wq(th, wgt, .90), 4),
        },
        "p_nothing_withheld": round(p_null, 4),
        "move": {
            "q10": round(wq(mv[finite], wgt[finite], .10), 2) if finite.any() else None,
            "q50": round(wq(mv[finite], wgt[finite], .50), 2) if finite.any() else None,
            "q90": round(wq(mv[finite], wgt[finite], .90), 2) if finite.any() else None,
        },
        "p_material_move": round(p_material, 4),
        "conditional_on": "world.GameParams as shown; NOT an empirical probability",
    }


def histogram(observed_z: dict[str, float], ref: ReferenceTable,
              bins: int = 12, keep_frac: float = 0.02) -> list[dict]:
    """Posterior over theta_held as a histogram, for plotting."""
    x = np.array([observed_z.get(f, 0.0) for f in ref.feats], float)
    d = np.linalg.norm((ref.Z - x) / ref.scale, axis=1)
    k = max(40, int(keep_frac * ref.n))
    idx = np.argsort(d)[:k]
    th = ref.theta_held[idx]
    counts, edges = np.histogram(th, bins=bins, range=(0.0, max(0.35, float(ref.theta_held.max()))))
    total = counts.sum() or 1
    return [{"lo": round(float(edges[i]), 3), "hi": round(float(edges[i + 1]), 3),
             "p": round(float(counts[i] / total), 4)} for i in range(len(counts))]
