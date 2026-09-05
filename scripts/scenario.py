"""
Define and run a scenario. This is the analyst-facing entry point to the simulator.

    # what does my coverage universe look like today?
    python scripts/scenario.py --name today

    # what if everyone gets media-trained?
    python scripts/scenario.py --name coached --adoption 1.0 --coaching-skill 0.9

    # my universe is mostly central bankers under a published blackout
    python scripts/scenario.py --name fed \
        --mix career_official=0.7,genuinely_uncertain=0.2,null=0.1 \
        --constraint-base 0.5 --n-speakers 22

    # stress the false-positive case: a universe of honest habitual hedgers
    python scripts/scenario.py --name skeptic --mix genuinely_uncertain=0.8,null=0.2

    python scripts/scenario.py --list                 # archetypes and presets
    python scripts/scenario.py --preset litigation    # a named starting point

Each run writes dashboard/scenarios/<name>.json and refreshes the index the
dashboard reads, so scenarios accumulate and can be compared side by side. A run
takes well under a minute, which is the point — a scenario you cannot iterate on
in a meeting is a report, not a tool.

WHAT THIS IS FOR. Not prediction. An analyst uses it to answer three questions
before trusting a flag:

  1. Given who I actually cover, how often will this cry wolf? (`false_positive`)
  2. Given my roster size and history depth, is this even answerable? (`power`)
  3. If my speakers get coached, what do I lose? (run it twice and diff)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from theroux import wargame as WG, world as W  # noqa: E402

OUT_DIR = Path("dashboard/scenarios")

#: Named starting points. Each is a (population, param overrides, blurb) triple.
PRESETS: dict[str, dict] = {
    "today": {
        "label": "Today",
        "blurb": "A mixed corporate universe in which nobody knows they are being read.",
        "mix": None, "params": {},
    },
    "coached": {
        "label": "Fully coached market",
        "blurb": "Every speaker is media-trained and optimising against the measurement.",
        "mix": {"coached_executive": 0.6, "naive_executive": 0.2,
                "genuinely_uncertain": 0.1, "null": 0.1},
        "params": {"adoption": 1.0, "coaching_skill": 0.9},
    },
    "fed": {
        "label": "Central bank",
        "blurb": "Career officials under a published blackout calendar. Low style "
                 "variance, real constraint, modest private information — the corpus "
                 "that holds format constant by design.",
        "mix": {"career_official": 0.7, "genuinely_uncertain": 0.2, "null": 0.1},
        "params": {"constraint_base": 0.5, "heterogeneity": 0.35, "n_speakers": 22},
    },
    "litigation": {
        "label": "Under litigation",
        "blurb": "Counsel has told most of the universe to decline rather than talk "
                 "around it. Tests whether flat refusal is still legible.",
        "mix": {"legally_muzzled": 0.6, "naive_executive": 0.2,
                "genuinely_uncertain": 0.2},
        "params": {},
    },
    "skeptic": {
        "label": "False-positive stress test",
        "blurb": "A universe of honest habitual hedgers who are withholding nothing. "
                 "Every flag here is a false positive. Run this before believing any "
                 "other number.",
        "mix": {"genuinely_uncertain": 0.8, "null": 0.2},
        "params": {},
    },
}


def parse_mix(s: str | None) -> dict[str, float] | None:
    """`a=0.6,b=0.4` -> {'a': 0.6, 'b': 0.4}. Weights need not sum to 1."""
    if not s:
        return None
    out = {}
    for part in s.split(","):
        if "=" not in part:
            raise SystemExit(f"bad --mix term {part!r}; expected archetype=weight")
        k, v = part.split("=", 1)
        k = k.strip()
        if k not in W.ARCHETYPES:
            raise SystemExit(f"unknown archetype {k!r}. available: {sorted(W.ARCHETYPES)}")
        out[k] = float(v)
    return out


def list_options() -> None:
    print("\nARCHETYPES\n")
    for a in W.ARCHETYPES.values():
        print(f"  {a.name:<22} {a.label}")
        for line in _wrap(a.description, 74):
            print(f"  {'':<22} {line}")
        bits = []
        if a.theta_mult != 1.0:
            bits.append(f"theta x{a.theta_mult}")
        if a.always_adapted:
            bits.append("always coached")
        if a.overrides:
            bits.append(", ".join(f"{k}={v}" for k, v in a.overrides.items()))
        if bits:
            print(f"  {'':<22} \033[2m[{' · '.join(bits)}]\033[0m")
        print()
    print("PRESETS\n")
    for k, v in PRESETS.items():
        print(f"  {k:<14} {v['label']}")
        for line in _wrap(v["blurb"], 74):
            print(f"  {'':<14} {line}")
        print()
    print("Default population (used when --mix is omitted):")
    for k, v in W.DEFAULT_POPULATION.items():
        print(f"  {k:<22} {v:.0%}")
    print()


def _wrap(text: str, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur); cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


def run(name: str, label: str, blurb: str, mix: dict | None,
        params: dict, n_reps: int, flag: float) -> dict:
    base = W.GameParams(**{**W.GameParams().as_dict(), **params})
    pop = W.resolve_population(mix)

    print(f"\n  scenario: {label}")
    print(f"  population: " + ", ".join(f"{k} {v:.0%}" for k, v in pop.items()))
    print(f"  n_speakers={base.n_speakers} baseline_n={base.baseline_n} "
          f"adoption={base.adoption} constraint_base={base.constraint_base}\n")

    ar = WG.archetype_rates(population=mix, flag_threshold=flag,
                            n_reps=n_reps, base=base)

    # measurement quality on this population
    rhos = []
    for rep in range(max(3, n_reps // 2)):
        s = WG.score_run(WG._prep(W.simulate(base, 6100 + rep * 19, population=mix)))
        if len(s["held"]) >= 8:
            rhos.append(WG._rho(s["evasion"], s["held"]))
    rho = float(np.nanmedian(rhos)) if rhos else float("nan")

    m = ar["matched_budget"]
    # A universe in which nobody withholds anything has no true-positive rate to
    # report — there is nothing to detect. That is not a missing value to paper over:
    # it is the whole content of the skeptic scenario, where the flag rate simply IS
    # the false-positive rate. NaN is also not valid JSON, so it must not reach disk.
    no_signal = not any(a["withholds"] for a in ar["archetypes"])
    print(f"  {'archetype':<22}{'n':>5}{'holds':>7}{'meanDiv':>9}{'flag':>8}")
    for a in ar["archetypes"]:
        print(f"  {a['label']:<22}{a['n_windows']:>5}{'yes' if a['withholds'] else 'no':>7}"
              f"{a['mean_divergence']:>9.3f}{a['flag_rate_relative']:>8.3f}")
    print(f"\n  matched alarm budget {m['alarm_budget']:.1%} of windows")
    print(f"    false positives   vs-own-baseline {m['false_positive_relative']:.3f}"
          f"   absolute {m['false_positive_absolute']:.3f}")
    if no_signal:
        print("    true positives    n/a — nobody in this population withholds anything,")
        print("                      so every alarm above is by definition a false one")
    else:
        print(f"    true positives    vs-own-baseline {m['true_positive_relative']:.3f}"
              f"   absolute {m['true_positive_absolute']:.3f}")
        print(f"  rho(divergence, information withheld) {rho:+.3f}")

    def clean(o):
        """NaN is not valid JSON. Nulls are honest; NaN on disk is a landmine."""
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [clean(v) for v in o]
        if isinstance(o, float) and (np.isnan(o) or np.isinf(o)):
            return None
        return o

    return clean({
        "name": name, "label": label, "blurb": blurb,
        "population": pop, "params": base.as_dict(),
        "flag_threshold": flag,
        "no_signal_population": no_signal,
        "archetypes": ar["archetypes"],
        "summary": ar["summary"],
        "matched_budget": m,
        "rho_held": round(rho, 4) if not np.isnan(rho) else None,
        "n_reps": n_reps,
        "model_conditional": True,
    })


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Define and run a Theroux simulation scenario.")
    ap.add_argument("--list", action="store_true", help="show archetypes and presets")
    ap.add_argument("--preset", choices=sorted(PRESETS))
    ap.add_argument("--name", help="scenario id (filename). defaults to the preset name")
    ap.add_argument("--label", help="display name")
    ap.add_argument("--mix", help="archetype=weight,archetype=weight")
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--flag-threshold", type=float, default=1.5)
    # any GameParams field can be set from the command line
    for f, v in W.GameParams().as_dict().items():
        ap.add_argument(f"--{f.replace('_', '-')}",
                        type=type(v), default=None, help=f"GameParams.{f} (default {v})")
    args = ap.parse_args()

    if args.list or (not args.preset and not args.name):
        list_options()
        return

    preset = PRESETS.get(args.preset or "", {})
    name = args.name or args.preset
    label = args.label or preset.get("label") or name
    blurb = preset.get("blurb", "Custom scenario.")

    params = dict(preset.get("params", {}))
    for f in W.GameParams().as_dict():
        v = getattr(args, f, None)
        if v is not None:
            params[f] = v

    mix = parse_mix(args.mix) or preset.get("mix")

    out = run(name, label, blurb, mix, params, args.reps, args.flag_threshold)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{name}.json").write_text(json.dumps(out, indent=1))

    # refresh the index the dashboard reads
    scenarios = []
    for f in sorted(OUT_DIR.glob("*.json")):
        if f.name == "index.json":
            continue
        scenarios.append(json.loads(f.read_text()))
    (OUT_DIR / "index.json").write_text(json.dumps({"scenarios": scenarios}, indent=1))

    print(f"\n  → {OUT_DIR / (name + '.json')}")
    print(f"  → {len(scenarios)} scenario(s) in the index; run `python run.py` to "
          f"rebuild the dashboard\n")


if __name__ == "__main__":
    main()
