"""Behavioural tests for the scoring layer — run: python -m pytest tests/ -q"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "src"))

from theroux.scorer import score_text
from theroux.baseline import build_baseline, divergence

DIRECT = ("Revenue was 412 million, up 19 percent year over year. Gross margin 74 percent, "
          "up 200 basis points sequentially. We added 340 logos in the first quarter. "
          "I'll tell you exactly where that came from.")
EVASIVE = ("I think it's fair to say the business is generally tracking. Stepping back, "
           "the bigger picture is that conditions in the industry have been somewhat mixed. "
           "The organization has done a lot of work. We'll see how it develops over time.")


def test_evasive_scores_higher_on_hedging_and_vagueness():
    d, e = score_text(DIRECT)["scores"], score_text(EVASIVE)["scores"]
    assert e["hedging"] > d["hedging"]
    assert e["specificity_avoidance"] > d["specificity_avoidance"]
    assert e["topic_deflection"] > d["topic_deflection"]


def test_scores_bounded():
    for txt in (DIRECT, EVASIVE):
        for v in score_text(txt)["scores"].values():
            assert 0.0 <= v <= 1.0


def test_every_score_carries_evidence():
    out = score_text(EVASIVE)
    assert out["evidence"]["hedging"], "hedging score must cite matched phrases"
    assert out["meta"]["words"] > 0


def test_divergence_is_relative_to_own_baseline():
    """A consistently vague speaker should NOT score as divergent when vague."""
    rows = [{"scores": score_text(EVASIVE)["scores"]} for _ in range(3)]
    base = build_baseline("Habitually Vague", rows)
    same = divergence(base, score_text(EVASIVE)["scores"])
    shift = divergence(base, score_text(DIRECT)["scores"])
    assert abs(same["composite"]) < abs(shift["composite"])
