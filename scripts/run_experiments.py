"""
Method-validation studies. Writes dashboard/experiments.json.

    python scripts/run_experiments.py

Four studies. Two of them exist because the first version of this file produced
misleading results and the fixes are worth keeping visible:

  ABLATION       Does each pipeline stage earn its place?

  HETEROGENEITY  THE key experiment. Baseline-relative scoring should only beat
                 absolute scoring when speakers actually differ from each other.
                 We sweep speaker heterogeneity and measure both. If the two
                 curves never separate, our core design claim is wrong.

  POWER          How many speakers before the question is answerable at all?

  RECOVERY       Plant a known per-speaker effect; measure whether the pipeline
                 recovers the RANKING of speakers by true leakiness. Ranking,
                 not classification — the first version used a binary
                 material-move label which becomes degenerate at high effect
                 (70% of windows material), so AUC had nothing to discriminate
                 and the method looked broken when the harness was.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from theroux import anomaly, confounds, experiments as EX, features as F, validate  # noqa: E402
from theroux.estimator import fit_baselines  # noqa: E402
from theroux.pipeline import load_corpus  # noqa: E402

import generate_corpus as G  # noqa: E402
from generate_corpus import compose, FIRST, ORGS, EVENTS_BASE, EVENTS_CONS  # noqa: E402

APRIORI = EX.APRIORI


def draw_style(rng, heterogeneity: float) -> dict:
    """
    Speaker's baseline linguistic style.

    `heterogeneity` controls how much speakers differ from one another. At 0 every
    speaker is the population average; at 1 they span nearly the full range —
    which is closer to reality, where a habitually vague CEO and a
    quote-the-basis-points CFO are genuinely different people.
    """
    mid = {"evasive": .34, "deflect": .16, "confident": .14, "distant": .28}
    span = {"evasive": .40, "deflect": .22, "confident": .22, "distant": .34}
    return {k: float(np.clip(mid[k] + heterogeneity * span[k] * rng.normal(), .02, .88))
            for k in mid}


def make_corpus(effect, seed: int, n_speakers: int = 14,
                heterogeneity: float = 0.9, per_speaker_effect: bool = False) -> list[dict]:
    """
    Corpus with a known true effect and controllable speaker heterogeneity.

    `effect` may be a scalar (all speakers shift equally) or, with
    per_speaker_effect=True, is ignored in favour of a per-speaker draw recorded
    on every row as `true_effect` — which is what the recovery study needs.
    """
    rng = np.random.default_rng(seed)
    G.RNG = rng
    rows = []

    for i in range(n_speakers):
        spk, org = f"{FIRST[i % len(FIRST)]}_{i}", ORGS[i % len(ORGS)]
        style = draw_style(rng, heterogeneity)
        eff = float(rng.uniform(0.0, 0.7)) if per_speaker_effect else float(effect)

        for j in range(6):
            d = rng.normal(0, .05)
            rows.append({
                "id": f"{i}-b{j}", "speaker": spk, "org": org, "domain": "corporate",
                "event": "Earnings call", "event_type": EVENTS_BASE[j % 2],
                "date": f"2024-{(j % 12) + 1:02d}-15", "window": "baseline",
                "days_to_disclosure": None, "outcome_move_pct": None, "outcome": None,
                "synthetic": True, "source_url": None, "true_effect": eff,
                "text": compose(np.clip(style["evasive"] + d, 0, 1),
                                np.clip(style["deflect"] + d / 2, 0, 1),
                                np.clip(style["confident"] + d / 2, 0, 1),
                                np.clip(style["distant"] + d, 0, 1),
                                int(rng.integers(7, 12))),
            })

        for j in range(3):
            shift = eff * float(rng.uniform(.6, 1.4))
            nz = rng.normal(0, .05)
            # Outcome kept deliberately noisy so the label never becomes degenerate.
            move = shift * 11 * rng.uniform(.5, 1.3) + rng.normal(0, 5.0)
            move = round(float((-1 if rng.random() < .5 + .2 * (shift > .18) else 1) * abs(move)), 1)
            rows.append({
                "id": f"{i}-c{j}", "speaker": spk, "org": org, "domain": "corporate",
                "event": "Media interview", "event_type": EVENTS_CONS[j % 3],
                "date": f"2025-{(j % 12) + 1:02d}-10", "window": "constrained",
                "days_to_disclosure": int(rng.integers(6, 24)),
                "outcome_move_pct": move,
                "outcome": "miss" if move < -2.5 else "beat" if move > 2.5 else "inline",
                "synthetic": True, "source_url": None, "true_effect": eff,
                "text": compose(np.clip(style["evasive"] + shift + nz, 0, 1),
                                np.clip(style["deflect"] + shift * .7 + nz, 0, 1),
                                np.clip(style["confident"] + shift * .35 + nz, 0, 1),
                                np.clip(style["distant"] + shift * .6 + nz, 0, 1),
                                int(rng.integers(6, 11))),
            })
    return rows


def prep(rows):
    F.score_corpus(rows)
    lms, _, _ = F.fit_language_models(rows)
    F.attach_surprisal(rows, lms)
    F.add_time_index(rows)
    return rows


def _absolute_and_relative(rows, want_truth: bool = False):
    """Score constrained windows two ways: against a shared standard, and against self."""
    feats = F.FEATURES
    Y = np.array([[r["scores"][f] for f in feats] for r in rows])
    X, _ = confounds.build_design(rows, ["x"])
    Yr = confounds.residualise(X, Y, confounds.fit_residualiser(X, Y))
    w = np.array([APRIORI[f] for f in feats]); w = w / np.linalg.norm(w)

    by_sp = {}
    for sp in sorted({r["speaker"] for r in rows}):
        M = np.array([Yr[i] for i, r in enumerate(rows)
                      if r["speaker"] == sp and r["window"] == "baseline"])
        if len(M) >= 2:
            by_sp[sp] = M
    bl = fit_baselines(by_sp, feats, min_n=2)

    idx = [i for i, r in enumerate(rows)
           if r["window"] == "constrained" and r.get("outcome_move_pct") is not None]
    absolute, relative, y, g, truth = [], [], [], [], []
    for i in idx:
        b = bl.get(rows[i]["speaker"])
        if b is None:
            continue
        absolute.append(float(Yr[i] @ w))
        relative.append(anomaly.directional_component(Yr[i], b, w))
        y.append(1 if abs(rows[i]["outcome_move_pct"]) >= 6.0 else 0)
        g.append(rows[i]["speaker"])
        truth.append(rows[i].get("true_effect", 0.0))
    res = (np.array(absolute), np.array(relative), np.array(y), np.array(g))
    return res + (np.array(truth),) if want_truth else res


def heterogeneity_study(levels=(0.0, 0.25, 0.5, 0.75, 1.0, 1.3),
                        n_reps: int = 5, effect: float = 0.35) -> list[dict]:
    """
    The core design claim, tested.

    Measured against the TRUE planted shift, not the outcome. That is deliberate:
    the outcome is a noisy downstream proxy, and mixing outcome noise into this
    test would hide the very thing we are trying to isolate — whether baselining
    improves our ability to MEASURE the linguistic shift at all.

    Prediction: at zero heterogeneity the two are equivalent (every speaker is the
    average speaker, so "vs the average" and "vs yourself" are the same
    comparison). As speakers diverge, absolute scoring degrades because a
    habitually-vague speaker looks evasive even at baseline, while
    baseline-relative scoring should hold.
    """
    out = []
    for h in levels:
        a_rho, r_rho = [], []
        for rep in range(n_reps):
            rows = prep(make_corpus(None, 900 + rep * 17, heterogeneity=h,
                                    per_speaker_effect=True))
            a, r, _, _, t = _absolute_and_relative(rows, want_truth=True)
            if len(a) < 8 or np.std(t) < 1e-9:
                continue
            a_rho.append(stats.spearmanr(a, t).statistic)
            r_rho.append(stats.spearmanr(r, t).statistic)
        if not a_rho:
            continue
        out.append({
            "heterogeneity": h,
            "rho_absolute": round(float(np.mean(a_rho)), 4),
            "rho_baseline_relative": round(float(np.mean(r_rho)), 4),
            "advantage": round(float(np.mean(r_rho) - np.mean(a_rho)), 4),
            "n_reps": len(a_rho),
        })
    return out


def recovery_ranking(n_reps: int = 8) -> list[dict]:
    """
    Plant a different true effect per speaker; measure whether the pipeline
    recovers the RANKING. Reports Spearman between each speaker's mean recovered
    score and their true planted effect.
    """
    out = []
    for rep in range(n_reps):
        rows = prep(make_corpus(None, 400 + rep * 13, per_speaker_effect=True))
        _, rel, _, g = _absolute_and_relative(rows)
        truth = {r["speaker"]: r["true_effect"] for r in rows}
        by = {}
        for s, sp in zip(rel, g):
            by.setdefault(sp, []).append(s)
        sp_list = sorted(by)
        rec = [float(np.mean(by[s])) for s in sp_list]
        tru = [truth[s] for s in sp_list]
        rho = stats.spearmanr(rec, tru).statistic
        out.append({"rep": rep, "n_speakers": len(sp_list),
                    "spearman_recovered_vs_true": round(float(rho), 4)})
    return out


def main() -> None:
    rows = prep(load_corpus("data/corpus"))

    print("\n── ABLATION ──────────────────────────────────────────────────")
    abl = EX.ablation(rows)
    for a in abl:
        d = f"{a['delta']:+.3f}" if a["delta"] is not None else "  —  "
        print(f"  {a['label']:<42} AUC {a['auc']:.3f}  Δ{d}")

    print("\n── HETEROGENEITY (the core design claim) ─────────────────────")
    het = heterogeneity_study()
    print("  measured against the TRUE planted shift (Spearman), not the noisy outcome")
    print(f"  {'speaker spread':>15}   {'absolute':>9} {'vs-own-baseline':>16} {'advantage':>10}")
    for h in het:
        print(f"  {h['heterogeneity']:>15.2f}   {h['rho_absolute']:>9.3f} "
              f"{h['rho_baseline_relative']:>16.3f} {h['advantage']:>+10.3f}")

    print("\n── POWER ─────────────────────────────────────────────────────")
    pw = EX.power_curve(rows)
    for p in pw:
        print(f"  {p['n_speakers']:>3} speakers ({p['median_n_windows']:>3} windows)  "
              f"AUC {p['auc_mean']:.3f}  median p {p['median_p']:.3f}  "
              f"power {p['power_at_05']:.2f}")
    need = EX.speakers_needed(pw)
    print(f"  → {need.get('status')}"
          + (f": {need.get('speakers_needed')} speakers for 80% power"
             if need.get("status") == "ok" else ""))

    print("\n── RECOVERY (does it rank speakers by true leakiness?) ───────")
    rec = recovery_ranking()
    rhos = [r["spearman_recovered_vs_true"] for r in rec]
    print(f"  Spearman(recovered, true) over {len(rec)} corpora: "
          f"median {np.median(rhos):+.3f}, range [{min(rhos):+.3f}, {max(rhos):+.3f}]")

    out = {"ablation": abl, "heterogeneity": het, "power": pw,
           "speakers_needed": need, "recovery_ranking": rec,
           "recovery_summary": {
               "median_spearman": round(float(np.median(rhos)), 4),
               "min": round(float(min(rhos)), 4), "max": round(float(max(rhos)), 4),
               "n_corpora": len(rec)}}
    Path("dashboard/experiments.json").write_text(json.dumps(out, indent=1))
    print("\n  → dashboard/experiments.json\n")


if __name__ == "__main__":
    main()
