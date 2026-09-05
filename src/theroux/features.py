"""
Feature assembly — the vector every downstream stage operates on.

Six features in two families, deliberately chosen to fail differently:

  LEXICAL (5)   hedging, specificity_avoidance, pronoun_distancing,
                topic_deflection, confidence_language
                Transparent, auditable, explainable to a non-technical analyst.
                Weakness: hand-authored word lists can be coached against.

  INFORMATION-THEORETIC (1)  surprisal_z
                Cross-entropy of the statement under a language model fitted to
                the speaker's own history, standardised against their normal.
                Uses no word lists at all, so it survives exactly the coaching
                that defeats the lexical family.

Combining families whose failure modes are uncorrelated is the point. A speaker
drilled to avoid "I think" lowers hedging while their vocabulary still shifts
away from their own baseline — surprisal catches what the lexicon loses.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from .lexicon import DIMENSIONS
from .scorer import score_text
from .surprisal import SpeakerLanguageModel, build_background

FEATURES = DIMENSIONS + ["surprisal_z"]


def score_corpus(rows: list[dict]) -> None:
    """Attach lexical scores, evidence and meta to every row, in place."""
    for r in rows:
        out = score_text(r["text"])
        r["scores"] = out["scores"]
        r["evidence"] = out["evidence"]
        r["meta"] = out["meta"]


def fit_language_models(rows: list[dict]) -> tuple[dict[str, SpeakerLanguageModel], Counter, int]:
    """One unigram LM per speaker, fitted on their baseline statements only."""
    bg, bg_total = build_background([r["text"] for r in rows])
    models: dict[str, SpeakerLanguageModel] = {}
    for sp in sorted({r["speaker"] for r in rows}):
        base_texts = [r["text"] for r in rows
                      if r["speaker"] == sp and r["window"] == "baseline"]
        if len(base_texts) >= 2:
            models[sp] = SpeakerLanguageModel(sp, base_texts, bg, bg_total)
    return models, bg, bg_total


def attach_surprisal(rows: list[dict], models: dict[str, SpeakerLanguageModel]) -> None:
    """Attach information-theoretic divergence against the speaker's own model."""
    for r in rows:
        lm = models.get(r["speaker"])
        if lm is None:
            r["surprisal"] = {"insufficient_tokens": True, "surprisal_z": 0.0}
            r["scores"]["surprisal_z"] = 0.0
            continue
        a = lm.analyse(r["text"])
        r["surprisal"] = a
        r["scores"]["surprisal_z"] = float(a.get("surprisal_z", 0.0))


def matrix(rows: list[dict], features: list[str] = FEATURES) -> np.ndarray:
    return np.array([[r["scores"][f] for f in features] for r in rows], dtype=float)


def add_time_index(rows: list[dict]) -> None:
    """Numeric time coordinate for the trend confound."""
    for r in rows:
        y, m, d = (r.get("date") or "2024-01-01").split("-")
        r["_t"] = int(y) * 12 + int(m) + int(d) / 31.0
