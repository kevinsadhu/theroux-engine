"""
Information-theoretic divergence — the lexicon-free signal.

WHY THIS EXISTS
---------------
The lexical scorer in `scorer.py` is transparent and defensible, but it has a
structural weakness: it can be gamed. A media-trained executive who is told
"stop saying 'I think'" will score lower on hedging without becoming any more
forthcoming. Every hand-authored word list has this problem.

This module measures something a speaker cannot easily suppress: how *unlikely*
their current vocabulary is under a language model fitted to their own history.
It uses no word lists at all. A speaker avoiding a topic reaches for different
words than they normally use, and that shows up as elevated cross-entropy
against their own baseline distribution regardless of which words they pick.

METHOD
------
For each speaker we fit a unigram distribution over their baseline statements,
smoothed toward a global background distribution by a Dirichlet prior:

    p_speaker(w) = (c_s(w) + alpha * p_bg(w)) / (N_s + alpha)

Dirichlet smoothing (Zhai & Lafferty, 2001) rather than add-one, because the
prior mass should follow the background distribution, not be uniform — and
because the effective smoothing scales with document length, which matters when
baseline sample sizes differ across speakers.

We then report, for a new statement q:

  cross_entropy       H(q, p_speaker)  — bits per token under the speaker's own model
  baseline_ce         mean H over the speaker's own baseline statements (their normal)
  surprisal_z         standardised deviation of cross_entropy from baseline_ce
  kl_from_self        KL(q || p_speaker) restricted to the statement's support
  novel_rate          fraction of tokens absent from the speaker's baseline vocabulary
  jsd_from_self       Jensen-Shannon divergence — symmetric, bounded, robust to
                      the support mismatch that makes raw KL unstable on short texts

`surprisal_z` is the feature that enters the model. The rest are diagnostics
that an analyst can inspect, because every number in this system has to be
explainable.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

# Function words carry the signal in stylometry — deliberately NOT removed.
# Authorship-attribution work (Mosteller & Wallace onward) shows function-word
# frequencies are the most speaker-stable and hardest-to-consciously-control
# features in text. Stripping them would discard exactly what we want.
_TOKEN = re.compile(r"[a-z']+")

ALPHA = 300.0          # Dirichlet concentration; tuned for ~1-3k token statements
MIN_TOKENS = 40        # below this, surprisal estimates are too noisy to report


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class SpeakerLanguageModel:
    """Unigram LM over one speaker's baseline statements, Dirichlet-smoothed."""

    def __init__(self, speaker: str, baseline_texts: list[str],
                 background: Counter, bg_total: int, alpha: float = ALPHA):
        self.speaker = speaker
        self.alpha = alpha
        self.counts = Counter()
        for t in baseline_texts:
            self.counts.update(tokenize(t))
        self.total = sum(self.counts.values())
        self.background = background
        self.bg_total = max(bg_total, 1)
        self.vocab = set(self.counts)

        # Each baseline statement's own cross-entropy, so we know what "normal"
        # surprisal looks like for this speaker rather than assuming a constant.
        self._baseline_ces = [self.cross_entropy(t) for t in baseline_texts]
        self.baseline_ce = float(np.mean(self._baseline_ces)) if self._baseline_ces else 0.0
        self.baseline_ce_sd = (
            float(np.std(self._baseline_ces, ddof=1)) if len(self._baseline_ces) > 1 else 0.0
        )

    def prob(self, w: str) -> float:
        p_bg = (self.background.get(w, 0) + 0.5) / (self.bg_total + 0.5 * len(self.background) + 1)
        return (self.counts.get(w, 0) + self.alpha * p_bg) / (self.total + self.alpha)

    def cross_entropy(self, text: str) -> float:
        """Bits per token of `text` under this speaker's model."""
        toks = tokenize(text)
        if not toks:
            return 0.0
        return -sum(math.log2(self.prob(w)) for w in toks) / len(toks)

    def analyse(self, text: str) -> dict:
        toks = tokenize(text)
        n = len(toks)
        if n < MIN_TOKENS:
            return {"insufficient_tokens": True, "n_tokens": n}

        ce = self.cross_entropy(text)
        # Floor the SD: with 2-3 baselines the sample SD is unstable and can
        # manufacture enormous z-scores out of nothing.
        sd = max(self.baseline_ce_sd, 0.15)
        surprisal_z = (ce - self.baseline_ce) / sd

        q = Counter(toks)
        support = list(q)
        pq = np.array([q[w] / n for w in support])
        pp = np.array([self.prob(w) for w in support])
        pp = pp / pp.sum()

        kl = float(np.sum(pq * np.log2(pq / pp)))
        m = 0.5 * (pq + pp)
        jsd = float(0.5 * np.sum(pq * np.log2(pq / m)) + 0.5 * np.sum(pp * np.log2(pp / m)))
        novel = sum(1 for w in toks if w not in self.vocab) / n

        return {
            "insufficient_tokens": False,
            "n_tokens": n,
            "cross_entropy": round(ce, 4),
            "baseline_ce": round(self.baseline_ce, 4),
            "surprisal_z": round(surprisal_z, 4),
            "kl_from_self": round(kl, 4),
            "jsd_from_self": round(jsd, 4),
            "novel_rate": round(novel, 4),
        }


def build_background(all_texts: list[str]) -> tuple[Counter, int]:
    """Corpus-wide unigram counts — the prior every speaker model shrinks toward."""
    bg = Counter()
    for t in all_texts:
        bg.update(tokenize(t))
    return bg, sum(bg.values())
