"""
Baselines and divergence — the methodological core.

A speaker's baseline is the mean and standard deviation of each dimension across
their ordinary-window statements. A constrained-window statement is scored as a
per-dimension z-score against that speaker's OWN baseline, then combined into a
weighted composite.

This is what separates Theroux from transcript-level sentiment scoring: we measure
divergence from self, not deviation from an absolute standard.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from .lexicon import DIMENSIONS

# Starting hypothesis. Tune against real outcome data with leave-one-out checks,
# and never present tuned weights as an out-of-sample result.
WEIGHTS = {
    "specificity_avoidance": 0.30,
    "hedging": 0.25,
    "topic_deflection": 0.20,
    "pronoun_distancing": 0.15,
    "confidence_language": 0.10,
}

# With few baseline statements per speaker the sample std is unstable and can
# explode the z-score. Floor it. At fellowship scale (20+ baselines/speaker)
# this floor stops binding and the distribution becomes properly estimated.
STD_FLOOR = 0.12


@dataclass
class Baseline:
    speaker: str
    n: int
    means: dict = field(default_factory=dict)
    stds: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"speaker": self.speaker, "n": self.n,
                "means": self.means, "stds": self.stds}


def build_baseline(speaker: str, scored_rows: list[dict]) -> Baseline:
    """scored_rows: records carrying a 'scores' dict, ordinary windows only."""
    if len(scored_rows) < 2:
        raise ValueError(
            f"{speaker}: need >=2 baseline statements, got {len(scored_rows)}"
        )
    means, stds = {}, {}
    for d in DIMENSIONS:
        vals = [r["scores"][d] for r in scored_rows]
        means[d] = round(statistics.fmean(vals), 4)
        stds[d] = round(max(statistics.stdev(vals), STD_FLOOR), 4)
    return Baseline(speaker=speaker, n=len(scored_rows), means=means, stds=stds)


def divergence(base: Baseline, scores: dict) -> dict:
    """Per-dimension z-scores against the speaker's own baseline, plus composite."""
    z = {d: round((scores[d] - base.means[d]) / base.stds[d], 3) for d in DIMENSIONS}
    composite = round(sum(WEIGHTS[d] * z[d] for d in DIMENSIONS), 3)
    return {"z": z, "composite": composite}


def spearman(xs: list[float], ys: list[float]) -> float:
    """Rank correlation. Reported alongside n — at small n nothing is significant."""
    if len(xs) < 3:
        return 0.0

    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        rk = [0.0] * len(v)
        for pos, i in enumerate(order):
            rk[i] = pos + 1
        return rk

    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    den = (
        sum((rx[i] - mx) ** 2 for i in range(n))
        * sum((ry[i] - my) ** 2 for i in range(n))
    ) ** 0.5
    return round(num / den, 3) if den else 0.0
