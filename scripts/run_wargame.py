"""
Run the war-game and write dashboard/wargame.json.

    python scripts/run_wargame.py            # full run, a few minutes
    python scripts/run_wargame.py --fast     # coarse, for iterating on the UI

Five outputs:

  adoption          how much signal survives as speakers learn they are being read
  survival          which features carry it once coaching is applied
  dose_predicted    the model's T-to-disclosure gradient
  dose_measured     the same gradient measured on the actual corpus, blocked-permuted
  design            speakers x baseline depth -> power, for planning acquisition
  posteriors        per-window model-conditional distribution over withheld info

The measured dose-response is the only item here that touches real measurement. It
is also the most important, because it is the prediction the Federal Reserve corpus
will confirm or kill — and it needs no outcome data, so it is testable the day the
transcripts land.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from theroux import inference, wargame as WG, world as W  # noqa: E402
from theroux.pipeline import load_corpus  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--results", default="dashboard/results.json")
    ap.add_argument("--out", default="dashboard/wargame.json")
    args = ap.parse_args()

    reps = 3 if args.fast else 10
    base = W.GameParams()

    print("\n── ADOPTION ──────────────────────────────────────────────────")
    print("  what survives as speakers learn they are being measured")
    adoption = WG.adoption_sweep(n_reps=reps, base=base)
    print(f"  {'adoption':>9} {'rho(held)':>10} {'AUC(move)':>10} "
          f"{'register':>9}   behaviour")
    for a in adoption:
        reg = a["mean_register"]
        note = "hedged evasion" if reg > .66 else "mixed" if reg > .33 else "flat refusal"
        print(f"  {a['adoption']:>9.2f} {a['rho_held']:>10.3f} "
              f"{(a['auc_move'] if a['auc_move'] is not None else float('nan')):>10.3f} "
              f"{reg:>9.3f}   {note}")

    print("\n── FEATURE SURVIVAL ──────────────────────────────────────────")
    survival = WG.feature_survival(n_reps=reps, base=base)
    print(f"  {'feature':<26} {'coachable':>9} {'naive':>7} {'adapted':>8} {'retained':>9}")
    for s in survival:
        r = s["retained"]
        print(f"  {s['feature']:<26} {s['coachability']:>9.2f} {s['rho_naive']:>7.3f} "
              f"{s['rho_adapted']:>8.3f} {(r if r is not None else float('nan')):>9.2f}")

    print("\n── DOSE-RESPONSE (predicted by the model) ────────────────────")
    dose_pred = WG.dose_response_predicted(n_reps=max(4, reps - 2), base=base)
    for d in dose_pred:
        m = d["mean_divergence"]
        bar = "" if m is None else "█" * max(0, int(round((m + 1) * 8)))
        print(f"  T−{d['days_lo']:>2}..{d['days_hi']:<2}d  "
              f"{(m if m is not None else float('nan')):>6.3f}  n={d['n']:<4} {bar}")

    res = json.loads(Path(args.results).read_text())
    ev_by_id = {r["id"]: r["evasion"] for r in res["results"]}
    corpus = load_corpus("data/corpus")
    is_synth = any(r.get("synthetic") for r in corpus)

    suppression = {"status": "not_applicable"}
    if not is_synth:
        print("\n── DOES THE CONSTRAINT BIND? (no linguistics required) ───────")
        # Share of the calendar under blackout, computed from the published rule.
        from datetime import date, timedelta
        from theroux.sources import fed
        d, days, cons_days = date(2024, 1, 1), 0, 0
        while d < date(2026, 12, 31):
            cons_days += fed.label_window(d)[0] == "constrained"
            days += 1
            d += timedelta(days=1)
        suppression = WG.speech_suppression(corpus, cons_days / days)
        if suppression.get("status") == "ok":
            print(f"  {suppression['n_constrained']} of {suppression['n_statements']} "
                  f"speeches fall in blackout  =  "
                  f"{suppression['share_of_speeches']:.1%} of speeches")
            print(f"  blackout covers                        "
                  f"{suppression['share_of_calendar']:.1%} of the calendar")
            print(f"  → speech suppressed {suppression['suppression_ratio']}x while the "
                  f"constraint is in force")
            print("  That is evidence the treatment is real, established before a single")
            print("  word is scored. It is also why the treatment group is small.")

    print("\n── DOSE-RESPONSE (measured on the corpus) ────────────────────")
    dose_meas = WG.dose_response_measured(corpus, ev_by_id)
    if dose_meas.get("status") == "ok":
        mw = dose_meas["median_within_speaker_rho"]
        print(f"  n={dose_meas['n']} windows across {dose_meas['n_speakers']} speakers")
        print(f"  pooled rho            {dose_meas['rho_pooled']:+.3f}   "
              f"p {dose_meas['p_pooled']:.4f}   (the wrong statistic — see below)")
        print(f"  median WITHIN-speaker {mw:+.3f}   p {dose_meas['p_within']:.4f}"
              f"   over {dose_meas['n_speakers_with_gradient']} speakers, "
              f"{dose_meas['frac_positive']:.0%} positive")
        print(f"  {dose_meas['direction']}")
        if not dose_meas["significant_at_05"]:
            if is_synth:
                print("  NOT significant — and expected: generate_corpus.py does not make")
                print("  the effect a function of time, so a null here is correct. This")
                print("  test earns its keep on the Fed corpus, where the blackout")
                print("  calendar supplies real variation in how binding the constraint is.")
            else:
                ns = dose_meas["n_speakers_with_gradient"]
                print(f"  NOT significant. Before reading anything into that: only {ns} "
                      f"speaker(s)\n  have the 3+ constrained windows a within-speaker "
                      "gradient requires, so this\n  is a statement about sample size, "
                      "not about the world. The blackout\n  suppresses speaking, so "
                      "constrained windows are structurally scarce —\n  widen the corpus "
                      "(--since 2021-01-01) before treating this as a result.")
    else:
        print(f"  {dose_meas.get('status')}")

    print("\n── DESIGN GRID ───────────────────────────────────────────────")
    design = WG.design_grid(n_reps=max(8, reps + 4), base=base)
    print(f"  {'speakers':>9} {'baseline_n':>11} {'rho':>7} {'power':>7}")
    for d in design:
        print(f"  {d['n_speakers']:>9} {d['baseline_n']:>11} {d['rho_held']:>7.3f} "
              f"{d['power_at_05']:>7.2f}")

    print("\n── REFERENCE TABLE + POSTERIORS ──────────────────────────────")
    ref = inference.build_reference(base, n_runs=6 if args.fast else 14)
    print(f"  {ref.n} simulated windows in the reference population")
    posteriors = {}
    for r in res["results"]:
        z = r.get("marginal_z") or {}
        posteriors[r["id"]] = {
            **inference.posterior(z, ref),
            "hist": inference.histogram(z, ref),
        }
    print(f"  posteriors for {len(posteriors)} observed windows")

    out = {
        "params": base.as_dict(),
        "coachability": W.COACHABILITY,
        "adoption": adoption,
        "survival": survival,
        "dose_predicted": dose_pred,
        "dose_measured": dose_meas,
        "design": design,
        "posteriors": posteriors,
        "reference_n": int(ref.n),
        "disclaimer": ("Every figure here except dose_measured is generated by "
                       "theroux.world, a model we specified. It describes the "
                       "implications of our assumptions, not evidence about the world."),
    }
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"\n  → {args.out}\n")


if __name__ == "__main__":
    main()
