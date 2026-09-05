"""
Experiments that test the method rather than showcase it.

Three studies, each designed to be able to embarrass us:

  ABLATION      Strip the pipeline back stage by stage and re-measure. If a stage
                does not improve discrimination, we say so. The point is to find
                out which of our design decisions actually earn their place, not
                to produce a ladder that happens to go up.

  POWER         Subsample speakers, recompute the statistic and its blocked
                permutation p-value, repeat. This converts "nothing is
                significant at n=42" from an apology into a number: how many
                speakers do we need before this question is answerable at all.

  RECOVERY      Generate corpora at known true effect sizes and measure what the
                pipeline recovers. At a true effect of zero it must return chance
                — a method that finds signal in null data is worse than useless.
                This is the only study here that can validate the machinery
                itself, because it is the only one where we know the answer.

Together these answer the question a technical reviewer actually asks, which is
not "does it work on your data" but "would you know if it didn't".
"""

from __future__ import annotations

import numpy as np

from . import anomaly, confounds, features as F, validate
from .estimator import fit_baselines
from .lexicon import DIMENSIONS

# ---------------------------------------------------------------------------
# Ablation
# ---------------------------------------------------------------------------

ABLATION_LADDER = [
    ("absolute",
     "Raw lexical scores, no baseline",
     "What transcript-sentiment vendors do: score the statement in isolation."),
    ("baseline_naive",
     "+ per-speaker baseline (independent z)",
     "Divergence from the speaker's own norm, treating features as independent."),
    ("confounds",
     "+ confound residualisation",
     "Length and time regressed out before scoring."),
    ("shrinkage",
     "+ shrinkage estimation",
     "Ledoit-Wolf covariance, empirical-Bayes pooled means."),
    ("mahalanobis",
     "+ correlation-aware distance",
     "Mahalanobis instead of independent z-sums."),
    ("surprisal",
     "+ information-theoretic feature",
     "Cross-entropy against the speaker's own language model."),
    ("signed",
     "+ signed a-priori projection",
     "Direction, not just magnitude."),
]

APRIORI = {
    "hedging": 1.0, "specificity_avoidance": 1.0, "pronoun_distancing": 0.8,
    "topic_deflection": 0.9, "confidence_language": 0.3, "surprisal_z": 0.6,
}


def _score_at_level(rows: list[dict], level: str, feats: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (scores, labels, groups) for constrained statements at one ablation level."""
    use_feats = [f for f in feats if f != "surprisal_z"] \
        if level in ("absolute", "baseline_naive", "confounds", "shrinkage", "mahalanobis") else feats

    Y = np.array([[r["scores"][f] for f in use_feats] for r in rows])

    # confound control from the 'confounds' level onward
    if level in ("absolute", "baseline_naive"):
        Yr = Y
    else:
        X, _ = confounds.build_design(rows, ["x"])
        Yr = confounds.residualise(X, Y, confounds.fit_residualiser(X, Y))

    w = np.array([APRIORI[f] for f in use_feats])

    cons = [i for i, r in enumerate(rows) if r["window"] == "constrained"
            and r.get("outcome_move_pct") is not None]

    # ---- absolute: no baseline at all ----
    if level == "absolute":
        s = Yr[cons] @ (w / np.linalg.norm(w))
    else:
        by_speaker: dict[str, np.ndarray] = {}
        for sp in sorted({r["speaker"] for r in rows}):
            M = np.array([Yr[i] for i, r in enumerate(rows)
                          if r["speaker"] == sp and r["window"] == "baseline"])
            if len(M) >= 2:
                by_speaker[sp] = M

        if level in ("baseline_naive", "confounds"):
            # naive: raw mean/sd, independent, unsigned magnitude
            stats = {sp: (M.mean(0), np.maximum(M.std(0, ddof=1), 0.12))
                     for sp, M in by_speaker.items()}
            s = []
            for i in cons:
                sp = rows[i]["speaker"]
                if sp not in stats:
                    s.append(0.0); continue
                mu, sd = stats[sp]
                z = (Yr[i] - mu) / sd
                s.append(float(np.abs(z) @ (w / w.sum())))
            s = np.array(s)
        else:
            bl = fit_baselines(by_speaker, use_feats, min_n=2)
            s = []
            for i in cons:
                b = bl.get(rows[i]["speaker"])
                if b is None:
                    s.append(0.0); continue
                if level == "shrinkage":
                    sd = np.sqrt(np.diag(b.cov))
                    z = (Yr[i] - b.mean) / np.maximum(sd, 1e-6)
                    s.append(float(np.abs(z) @ (w / w.sum())))
                elif level == "mahalanobis":
                    s.append(anomaly.mahalanobis(Yr[i], b))
                elif level == "surprisal":
                    s.append(anomaly.mahalanobis(Yr[i], b))
                else:  # signed
                    s.append(anomaly.directional_component(Yr[i], b, w))
            s = np.array(s)

    labels = np.array([1 if abs(rows[i]["outcome_move_pct"]) >= 6.0 else 0 for i in cons])
    groups = np.array([rows[i]["speaker"] for i in cons])
    return s, labels, groups


def ablation(rows: list[dict]) -> list[dict]:
    """Run the full ladder. Reports AUC and blocked permutation p at each rung."""
    out = []
    prev = None
    for level, label, why in ABLATION_LADDER:
        s, y, g = _score_at_level(rows, level, F.FEATURES)
        a = validate.auc(s, y)
        perm = validate.permutation_test(s, y.astype(float), groups=g, n_perm=4000)
        out.append({
            "level": level, "label": label, "rationale": why,
            "auc": round(a, 4),
            "delta": None if prev is None else round(a - prev, 4),
            "p_value": perm["p_value"],
            "n": int(len(y)),
        })
        prev = a
    return out


# ---------------------------------------------------------------------------
# Power
# ---------------------------------------------------------------------------

def power_curve(rows: list[dict], speaker_counts: tuple[int, ...] = (4, 6, 8, 10, 12, 15),
                n_draws: int = 40, seed: int = 3) -> list[dict]:
    """
    Subsample speakers, recompute AUC and blocked permutation p, repeat.

    Answers: at how many speakers does this question become answerable?
    """
    rng = np.random.default_rng(seed)
    speakers = sorted({r["speaker"] for r in rows})
    out = []

    for k in speaker_counts:
        if k > len(speakers):
            continue
        aucs, ps, ns = [], [], []
        for _ in range(n_draws):
            pick = set(rng.choice(speakers, size=k, replace=False))
            sub = [r for r in rows if r["speaker"] in pick]
            try:
                s, y, g = _score_at_level(sub, "signed", F.FEATURES)
            except Exception:
                continue
            if len(y) < 6 or len(np.unique(y)) < 2:
                continue
            aucs.append(validate.auc(s, y))
            ps.append(validate.permutation_test(s, y.astype(float), groups=g,
                                                n_perm=1200)["p_value"])
            ns.append(len(y))
        if not aucs:
            continue
        out.append({
            "n_speakers": k,
            "median_n_windows": int(np.median(ns)),
            "auc_mean": round(float(np.mean(aucs)), 4),
            "auc_lo": round(float(np.quantile(aucs, .1)), 4),
            "auc_hi": round(float(np.quantile(aucs, .9)), 4),
            "median_p": round(float(np.median(ps)), 4),
            "power_at_05": round(float(np.mean(np.array(ps) < .05)), 3),
        })
    return out


def speakers_needed(curve: list[dict], target_power: float = 0.8) -> dict:
    """
    Extrapolate the speaker count needed for `target_power`.

    Power grows roughly with sqrt(n) for a fixed effect, so we fit
    power ~ a + b*sqrt(n) and invert. Crude, and labelled as such — the point
    is an order of magnitude for planning, not a precise sample-size table.
    """
    if len(curve) < 3:
        return {"status": "insufficient_points"}
    x = np.sqrt([c["n_speakers"] for c in curve])
    y = np.array([c["power_at_05"] for c in curve])
    if y.max() < 0.02:
        return {"status": "no_measurable_power_in_range",
                "max_power_observed": float(y.max()),
                "note": "Effect is too small to detect at any n tested; "
                        "extrapolation would be fabrication."}
    b, a = np.polyfit(x, y, 1)
    if b <= 0:
        return {"status": "power_not_increasing"}
    need = ((target_power - a) / b) ** 2
    return {"status": "ok", "target_power": target_power,
            "speakers_needed": int(np.ceil(need)),
            "fit_slope": round(float(b), 4), "fit_intercept": round(float(a), 4),
            "caveat": "sqrt-n extrapolation from a small grid; order of magnitude only"}


# ---------------------------------------------------------------------------
# Recovery
# ---------------------------------------------------------------------------

def recovery_study(generator, effect_sizes=(0.0, 0.1, 0.2, 0.3, 0.45, 0.6),
                   n_reps: int = 6, seed: int = 5) -> list[dict]:
    """
    Generate corpora at known true effect sizes; measure what we recover.

    `generator(effect, seed) -> rows` must produce a scored-ready corpus in which
    every speaker's constrained statements are shifted by `effect`.

    The critical row is effect = 0.0. If the pipeline reports AUC meaningfully
    above 0.5 on null data, the method manufactures signal and everything else
    here is void.
    """
    out = []
    for eff in effect_sizes:
        aucs = []
        for rep in range(n_reps):
            rows = generator(eff, seed * 100 + rep)
            F.score_corpus(rows)
            lms, _, _ = F.fit_language_models(rows)
            F.attach_surprisal(rows, lms)
            F.add_time_index(rows)
            try:
                s, y, _ = _score_at_level(rows, "signed", F.FEATURES)
            except Exception:
                continue
            if len(np.unique(y)) < 2:
                continue
            aucs.append(validate.auc(s, y))
        if not aucs:
            continue
        out.append({
            "true_effect": eff,
            "auc_mean": round(float(np.mean(aucs)), 4),
            "auc_sd": round(float(np.std(aucs)), 4),
            "auc_lo": round(float(np.min(aucs)), 4),
            "auc_hi": round(float(np.max(aucs)), 4),
            "n_reps": len(aucs),
        })
    return out
