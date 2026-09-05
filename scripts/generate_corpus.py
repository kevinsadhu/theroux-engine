"""
Synthetic corpus generator — with deliberate, documented noise.

READ THIS BEFORE TRUSTING ANY NUMBER THE PIPELINE PRODUCES ON THIS DATA.

Any model evaluated on data we generated is circular: we wrote the pattern in,
so of course it can be found. The purpose of this corpus is to exercise the
statistical machinery end to end — shrinkage, Mahalanobis, permutation testing,
grouped cross-validation — on data with realistic structure, so that when a real
corpus arrives the pipeline is already correct.

To keep it from being trivially separable we build in four kinds of honesty:

  1. NULL SPEAKERS. ~30% of speakers have signal_strength = 0. Their constrained
     statements are drawn from the same distribution as their baselines. A model
     that "detects" them is overfitting.

  2. HETEROGENEOUS EFFECT SIZE. The rest are drawn from a wide distribution, so
     some speakers leak heavily and some barely.

  3. UNPREDICTABLE OUTCOMES. Outcome magnitude is signal * beta + large Gaussian
     noise, and ~20% of events are "shocks" — big moves with no linguistic
     antecedent, because in reality most large moves come from things nobody was
     hiding.

  4. WITHIN-SPEAKER DRIFT. Baseline statements vary; speakers are not constants.

Expected result: a real but modest effect, out-of-fold AUC in the 0.6-0.8 range,
not 0.95. If this pipeline ever reports near-perfect separation, something is
wrong with the pipeline, not right with the idea.
"""

from __future__ import annotations

import json
import pathlib
import numpy as np

RNG = np.random.default_rng(20260902)

FIRST = ["D. Okonjo", "M. Ferreira", "S. Ambrose", "T. Lindqvist", "R. Vasquez",
         "A. Whitfield", "K. Nakamura", "L. Bergström", "P. Adeyemi", "J. Kovač",
         "H. Castellanos", "N. Farouk", "E. Thorsen", "C. Mbeki", "V. Ivanova"]

ORGS = ["Vantree Systems", "Halden Robotics", "Corveth Health", "Brightmoor Retail",
        "Nortrail Energy", "Calderon Software", "Kestrel Semiconductor",
        "Ardent Logistics", "Pallas Financial", "Riverton Materials",
        "Sable Networks", "Occitan Foods", "Braemar Aviation",
        "Tessaly Biosciences", "Nordvik Marine"]

# ---------------------------------------------------------------- text banks
# Each bank is graded from concrete (index 0) to evasive (index -1).

OPENERS_DIRECT = [
    "Revenue was {a} million, up {b} percent year over year.",
    "We finished the quarter at {a} million with margin of {b} percent.",
    "Volume was {a} thousand units against a plan of {c} thousand.",
    "ARR closed at {a} million, growing {b} percent.",
    "We delivered {a} million in free cash flow this quarter.",
]
OPENERS_VAGUE = [
    "I think the quarter developed broadly in line with how we've been describing it.",
    "It's fair to say the business is generally tracking to our expectations.",
    "The environment has been, I would say, somewhat mixed.",
    "Conditions in the segment have been reasonably constructive.",
    "Perhaps the useful framing is that the trajectory is intact.",
]

DETAIL_DIRECT = [
    "Gross margin was {b} percent, up {c} basis points sequentially.",
    "Net retention held at {b} percent versus {c} a year ago.",
    "Operating expense was {a} million, or {b} percent of revenue.",
    "We added {c} net new accounts, and {a} of those came from the enterprise tier.",
    "Capital expenditure will be {a} million for the full year.",
    "Our backlog stands at {a} million, covering {b} percent of next year's plan.",
]
DETAIL_VAGUE = [
    "There are a number of dynamics at play across the category.",
    "The organization has done considerable work in this area.",
    "There is always some variability quarter to quarter.",
    "The business has performed consistent with the model.",
    "Conditions have been somewhat dynamic in recent months.",
]

OWNERSHIP_FIRST = [
    "I'll take the churn question directly: we lost two accounts and both were consolidation.",
    "I want to be clear that we missed on the utilization line, and here is why.",
    "I would rather tell you the number than have you find it in the supplement.",
    "We made that call and I'd make it again at these levels.",
    "I'm not going to pretend the labor market is easy, because it is not.",
]
OWNERSHIP_DISTANT = [
    "The company continues to evaluate its position on that.",
    "It was decided that the prior approach would be revisited.",
    "The team is executing against the framework that was established.",
    "There is a process underway and the business will update in due course.",
    "The market will ultimately determine how conditions resolve.",
]

DEFLECT = [
    "Stepping back, the bigger picture is that the multi-year opportunity is unchanged.",
    "What I would focus on is the durability of the platform over time.",
    "More broadly, the structural story here has not shifted.",
    "The way I think about it is over a longer horizon than any single period.",
    "I'd point you to the long-term algorithm rather than any one print.",
]

CONFIDENT = [
    "We are absolutely thrilled with how this has developed.",
    "Clearly the demand environment has been tremendous.",
    "Without a doubt this is the strongest period we have had.",
    "I'm extremely confident in what this team can deliver.",
    "Certainly the fundamentals have never been stronger.",
]

CLOSERS_DIRECT = [
    "We are holding the full year guide at {a} billion.",
    "We are raising the range to {a} to {c} million.",
    "Net debt to EBITDA is {d} times.",
    "Our hedge book covers {b} percent of next year at {a} dollars.",
]
CLOSERS_VAGUE = [
    "We'll see how it develops from here.",
    "I wouldn't want to get ahead of where the company lands.",
    "It would be premature to characterize the outcome.",
    "Hopefully that gives you a sense of the shape of things.",
]

EVENTS_BASE = ["earnings_call_qa", "analyst_day_qa"]
EVENTS_CONS = ["media_interview", "conference_fireside", "investor_conference"]


def _nums() -> dict:
    return {
        "a": int(RNG.integers(80, 990)),
        "b": int(RNG.integers(3, 78)),
        "c": int(RNG.integers(10, 460)),
        "d": round(float(RNG.uniform(0.4, 3.8)), 1),
    }


def compose(evasive: float, deflect_p: float, confident_p: float,
            distant: float, n_sent: int) -> str:
    """Build a statement whose surface features track the requested propensities."""
    out = []
    out.append((OPENERS_VAGUE if RNG.random() < evasive else OPENERS_DIRECT)[
        RNG.integers(0, 5)].format(**_nums()))

    for _ in range(n_sent):
        r = RNG.random()
        if r < deflect_p:
            out.append(DEFLECT[RNG.integers(0, len(DEFLECT))])
        elif r < deflect_p + confident_p:
            out.append(CONFIDENT[RNG.integers(0, len(CONFIDENT))])
        elif RNG.random() < distant:
            out.append(OWNERSHIP_DISTANT[RNG.integers(0, len(OWNERSHIP_DISTANT))])
        elif RNG.random() < evasive:
            out.append(DETAIL_VAGUE[RNG.integers(0, len(DETAIL_VAGUE))])
        else:
            bank = DETAIL_DIRECT if RNG.random() > 0.35 else OWNERSHIP_FIRST
            out.append(bank[RNG.integers(0, len(bank))].format(**_nums()))

    out.append((CLOSERS_VAGUE if RNG.random() < evasive else CLOSERS_DIRECT)[
        RNG.integers(0, 4)].format(**_nums()))
    return " ".join(out)


def main() -> None:
    out_dir = pathlib.Path("data/corpus")
    for f in out_dir.glob("*.json"):
        f.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)

    records, manifest = [], []

    for i, (spk, org) in enumerate(zip(FIRST, ORGS)):
        # --- speaker style: their own normal ---
        style = {
            "evasive": float(np.clip(RNG.beta(2, 5), .03, .75)),
            "deflect": float(np.clip(RNG.beta(1.6, 8), .01, .35)),
            "confident": float(np.clip(RNG.beta(1.5, 9), .01, .30)),
            "distant": float(np.clip(RNG.beta(2, 7), .02, .55)),
        }

        # --- 30% of speakers leak nothing at all ---
        null_speaker = RNG.random() < 0.30
        strength = 0.0 if null_speaker else float(np.clip(RNG.gamma(2.2, 0.16), 0, 0.85))

        n_base = int(RNG.integers(4, 9))
        n_cons = int(RNG.integers(2, 5))

        for j in range(n_base):
            drift = RNG.normal(0, 0.05)          # speakers are not constants
            records.append({
                "id": f"{org.split()[0].lower()}-b{j}",
                "speaker": spk, "org": org, "domain": "corporate",
                "event": "Quarterly earnings call Q&A",
                "event_type": EVENTS_BASE[RNG.integers(0, 2)],
                "date": f"202{3 + j // 4}-{(j % 4) * 3 + 2:02d}-15",
                "window": "baseline",
                "days_to_disclosure": None,
                "text": compose(
                    np.clip(style["evasive"] + drift, 0, 1),
                    np.clip(style["deflect"] + drift / 2, 0, 1),
                    np.clip(style["confident"] + drift / 2, 0, 1),
                    np.clip(style["distant"] + drift, 0, 1),
                    int(RNG.integers(7, 13))),
                "outcome_move_pct": None, "outcome": None,
                "synthetic": True, "source_url": None,
            })

        for j in range(n_cons):
            shift = strength * float(RNG.uniform(0.6, 1.4))
            noise = RNG.normal(0, 0.05)
            text = compose(
                np.clip(style["evasive"] + shift + noise, 0, 1),
                np.clip(style["deflect"] + shift * 0.7 + noise, 0, 1),
                np.clip(style["confident"] + shift * 0.35 + noise, 0, 1),
                np.clip(style["distant"] + shift * 0.6 + noise, 0, 1),
                int(RNG.integers(6, 11)))

            # Outcome: partly driven by the leak, mostly by everything else.
            shock = RNG.random() < 0.20        # big move, no linguistic antecedent
            base_move = shift * 16.0 * RNG.uniform(0.5, 1.3)
            move = base_move + RNG.normal(0, 4.2) + (RNG.normal(0, 11) if shock else 0)
            sign = -1 if RNG.random() < (0.5 + 0.22 * (shift > 0.18)) else 1
            move = round(float(sign * abs(move)), 1)
            outcome = "miss" if move < -2.5 else "beat" if move > 2.5 else "inline"

            records.append({
                "id": f"{org.split()[0].lower()}-c{j}",
                "speaker": spk, "org": org, "domain": "corporate",
                "event": ["Media interview", "Conference fireside",
                          "Investor conference"][RNG.integers(0, 3)],
                "event_type": EVENTS_CONS[RNG.integers(0, 3)],
                "date": f"202{4 + j // 3}-{(j % 4) * 3 + 1:02d}-{RNG.integers(5, 27):02d}",
                "window": "constrained",
                "days_to_disclosure": int(RNG.integers(6, 24)),
                "text": text,
                "outcome_move_pct": move, "outcome": outcome,
                "synthetic": True, "source_url": None,
            })

        manifest.append({"speaker": spk, "org": org, "null_speaker": null_speaker,
                         "true_signal_strength": round(strength, 3),
                         "n_baseline": n_base, "n_constrained": n_cons})

    for r in records:
        (out_dir / f"{r['id']}.json").write_text(json.dumps(r, indent=1))

    pathlib.Path("data/ground_truth.json").write_text(json.dumps(manifest, indent=1))

    n_c = sum(1 for r in records if r["window"] == "constrained")
    n_null = sum(1 for m in manifest if m["null_speaker"])
    print(f"{len(records)} statements · {len(manifest)} speakers · "
          f"{n_c} constrained · {n_null} null speakers (no signal by construction)")
    print("ground truth (for evaluation only) → data/ground_truth.json")


if __name__ == "__main__":
    main()
