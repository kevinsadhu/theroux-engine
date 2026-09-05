"""
Scoring layer.

`score_text()` is the single interface the rest of the system depends on.
v0 is lexical: deterministic, instant, no API key. v1 is an LLM call that
returns the same five dimensions plus quote-level citations.

Swapping v0 for v1 must not require changes anywhere else.
"""

from __future__ import annotations

import re

from . import lexicon as LX

_NUM = re.compile(LX.NUMERIC_PATTERN, re.I)
_PERIOD = re.compile(LX.PERIOD_PATTERN, re.I)


def _rate(text: str, phrases: list[str], per: int = 100) -> float:
    """Occurrences of any phrase per `per` words."""
    haystack = " " + text.lower() + " "
    hits = sum(haystack.count(p) for p in phrases)
    words = max(len(text.split()), 1)
    return hits / words * per


def _clip(x: float) -> float:
    return round(max(0.0, min(1.0, x)), 4)


def find_evidence(text: str, phrases: list[str], limit: int = 6) -> list[str]:
    """Return the matched phrases actually present, for the audit trail."""
    low = " " + text.lower() + " "
    found = [p.strip() for p in phrases if p in low]
    return sorted(set(found), key=len, reverse=True)[:limit]


def score_text(text: str) -> dict:
    """
    Five dimensions on 0-1, computed from the text itself.

    Returns {"scores": {...}, "evidence": {...}} — evidence is the provenance
    trail. Every score path in this system must be able to cite itself.
    """
    words = max(len(text.split()), 1)
    N = LX.NORMALISERS

    hedging = _clip(_rate(text, LX.HEDGES) / N["hedging"])

    concrete = (len(_NUM.findall(text)) + len(_PERIOD.findall(text))) / words * 100
    specificity_avoidance = _clip(
        1.0 - (concrete / N["specificity_density_full_marks"])
    )

    fp = _rate(text, LX.FIRST_PERSON)
    imp = _rate(text, LX.IMPERSONAL)
    pronoun_distancing = _clip(imp / max(fp + imp, 1e-6))

    topic_deflection = _clip(_rate(text, LX.DEFLECTORS) / N["topic_deflection"])
    confidence_language = _clip(_rate(text, LX.CONFIDENCE) / N["confidence_language"])

    return {
        "scores": {
            "hedging": hedging,
            "specificity_avoidance": specificity_avoidance,
            "pronoun_distancing": pronoun_distancing,
            "topic_deflection": topic_deflection,
            "confidence_language": confidence_language,
        },
        "evidence": {
            "hedging": find_evidence(text, LX.HEDGES),
            "topic_deflection": find_evidence(text, LX.DEFLECTORS),
            "confidence_language": find_evidence(text, LX.CONFIDENCE),
            "pronoun_distancing": find_evidence(text, LX.IMPERSONAL),
            "specificity_avoidance": [
                f"{len(_NUM.findall(text))} numeric markers",
                f"{len(_PERIOD.findall(text))} period markers",
                f"{concrete:.1f} concrete markers / 100 words",
            ],
        },
        "meta": {"words": words, "concrete_per_100w": round(concrete, 2)},
    }


# ---------------------------------------------------------------------------
# v1 — LLM scorer. Same signature, same return shape.
# Highest-value next task. See CLAUDE.md.
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM_PROMPT = """\
You are scoring a transcript of public remarks by a speaker who may hold material
information they are constrained from disclosing directly.

You are NOT detecting lies. Score linguistic patterns associated with constrained
or reluctant disclosure. Cite the exact quote behind every score.

Score 0.0-1.0 on each dimension:
1. hedging — qualifiers where a direct answer was available
2. specificity_avoidance — vagueness where a number, date or fact was expected
3. pronoun_distancing — first-person shifting to impersonal at moments of risk
4. topic_deflection — redirecting from the question asked
5. confidence_language — overt certainty markers (scored, not assumed directional)

Do not speculate about what the speaker knows. Score only the language present.
"""

EXTRACTION_TOOL = {
    "name": "record_scores",
    "description": "Record scored linguistic dimensions with quote citations.",
    "input_schema": {
        "type": "object",
        "properties": {
            **{
                d: {"type": "number", "minimum": 0, "maximum": 1}
                for d in LX.DIMENSIONS
            },
            "citations": {
                "type": "object",
                "description": "dimension -> list of exact quotes justifying the score",
            },
        },
        "required": LX.DIMENSIONS + ["citations"],
    },
}


def score_text_llm(text: str, model: str = "claude-opus-4-5") -> dict:
    """
    Not implemented. Wire this to the Anthropic API with forced tool use so
    the return shape matches score_text(). Anonymize speaker and company in
    the prompt — the model may recall real outcomes and leak them into scores.
    """
    raise NotImplementedError(
        "LLM scorer not yet wired. See CLAUDE.md — this is the next task."
    )
