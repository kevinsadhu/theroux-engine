"""
Property tests for the generative game and the war-game sweeps.

These assert PROPERTIES, not pinned numbers — the same discipline as
test_statistics.py. A simulator that cannot be tested is a simulator you cannot
trust, and every one of these exists because getting it wrong would have produced a
plausible-looking chart that meant nothing.

    python tests/test_world.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from theroux import wargame as WG, world as W  # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'pass' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))


# ---------------------------------------------------------------------------

def test_nothing_to_hide_produces_no_register():
    """A speaker holding nothing has no reason to paper over anything."""
    p = W.GameParams()
    br = W.best_response(0.0, 0.9, p)
    check("theta=0 gives zero register and zero held",
          br["register"] == 0.0 and br["theta_held"] == 0.0)


def test_constraint_rises_as_disclosure_approaches():
    """The dose-response is the whole falsifiable prediction; it must be monotone."""
    p = W.GameParams()
    cs = [W.constraint_at(t, p) for t in (30, 25, 20, 15, 10, 5, 1)]
    check("constraint is monotone increasing as T -> 0",
          all(b >= a - 1e-9 for a, b in zip(cs, cs[1:])),
          f"{[round(c, 2) for c in cs]}")


def test_more_binding_constraint_means_more_withheld():
    """Intensive margin: the same speaker withholds more when the constraint bites."""
    p = W.GameParams()
    held = [W.best_response(1.0, c, p)["theta_held"] for c in (0.1, 0.3, 0.5, 0.7, 0.9)]
    check("held information rises with constraint at every step",
          all(b > a for a, b in zip(held, held[1:])),
          f"{[round(h, 3) for h in held]}")


def test_adoption_pushes_speakers_from_hedging_to_refusal():
    """The behavioural prediction of the game."""
    p0 = W.GameParams(adoption=0.0)
    p1 = W.GameParams(adoption=1.0)
    r0 = W.best_response(0.5, 0.8, p0)["register"]
    r1 = W.best_response(0.5, 0.8, p1)["register"]
    check("register falls as adoption rises", r1 < r0, f"{r0:.2f} -> {r1:.2f}")
    check("register is 1.0 when nobody is watching", abs(r0 - 1.0) < 1e-9)


def test_null_corpus_produces_no_recoverable_signal():
    """
    The single most important test here. With theta_scale = 0 nobody is withholding
    anything, so any correlation the pipeline reports is manufactured. A simulator
    that finds signal in its own null is worse than useless.
    """
    p = W.GameParams(theta_scale=0.0, p_null=1.0, n_speakers=12)
    rows = WG._prep(W.simulate(p, 4242))
    s = WG.score_run(rows)
    check("null world: nothing is withheld",
          float(np.max(s["held"])) < 1e-9)
    check("null world: divergence has no systematic sign",
          abs(float(np.mean(s["evasion"]))) < 0.6,
          f"mean evasion {np.mean(s['evasion']):+.3f}")


def test_signal_is_recovered_when_it_is_planted():
    """
    The converse of the null test: with a real effect the pipeline must find it.

    Across seeds, deliberately. A single-seed version of this asserted rho > 0.2 and
    failed at +0.183 — not because anything was broken but because rho on one corpus
    of 16 speakers has sd ~0.16. The tempting fix was to lower the threshold until
    that seed passed, which is how a test stops testing anything. The honest fix is
    to assert the property that actually holds: the effect is positive on every draw,
    and its median clears the bar.
    """
    p = W.GameParams(theta_scale=0.7, p_null=0.1, n_speakers=16)
    rhos = [WG._rho(*[WG.score_run(WG._prep(W.simulate(p, s)))[k]
                      for k in ("evasion", "held")])
            for s in (909, 101, 202, 303, 404)]
    rhos = np.array(rhos)
    check("planted signal is recovered on every seed", bool((rhos > 0).all()),
          f"min {rhos.min():+.3f}")
    check("median recovery clears 0.2", float(np.median(rhos)) > 0.2,
          f"median {np.median(rhos):+.3f} over {len(rhos)} corpora")


def test_window_lengths_match_across_conditions():
    """
    Regression test for a real bug. Baseline windows once drew 7-12 sentences and
    constrained windows 6-11; that one-sentence difference changed concrete-marker
    density and manufactured a -0.95 sigma divergence on windows where nothing at all
    was being withheld. Confound residualisation did not absorb it.
    """
    p = W.GameParams(theta_scale=0.0, p_null=1.0, n_speakers=10)
    rows = W.simulate(p, 77)
    base = [len(r["text"].split()) for r in rows if r["window"] == "baseline"]
    cons = [len(r["text"].split()) for r in rows if r["window"] == "constrained"]
    rel = abs(np.mean(base) - np.mean(cons)) / np.mean(base)
    check("baseline and constrained windows have comparable length",
          rel < 0.10, f"{np.mean(base):.0f} vs {np.mean(cons):.0f} words ({rel:.1%})")


def test_coaching_degrades_lexical_features_more_than_structural_ones():
    """The moat argument, as a test. If this inverts, the mechanism is wrong."""
    p = W.GameParams()
    sv = {d["feature"]: d for d in WG.feature_survival(n_reps=4, base=p)}
    check("hedging degrades under coaching",
          sv["hedging"]["retained"] < 0.85,
          f"retained {sv['hedging']['retained']}")
    check("hedging is hit harder than specificity avoidance",
          sv["hedging"]["retained"] < sv["specificity_avoidance"]["retained"],
          f"{sv['hedging']['retained']} vs {sv['specificity_avoidance']['retained']}")


def test_within_speaker_test_beats_pooled_on_a_planted_gradient():
    """
    Why dose_response_measured blocks. theta is a speaker-level property, so the
    between-speaker spread swamps the within-speaker movement of the constraint. The
    pooled statistic should be much weaker than the blocked one on the same data.
    """
    # A population of speakers who ACTUALLY WITHHOLD. The default mix is ~half
    # zero-theta speakers (genuinely_uncertain, null, plus p_null), and a speaker
    # holding nothing has no gradient to detect, so including them measures population
    # composition rather than the statistical property under test. This assertion is
    # about blocking versus pooling, not about how many honest people are in the room.
    p = W.GameParams(n_speakers=20, constrained_n=4, p_null=0.0)
    rows = WG._prep(W.simulate(p, 3131, population={"naive_executive": 1.0}))
    s = WG.score_run(rows)
    ev_by_id = {}
    k = 0
    for r in rows:
        if r["window"] == "constrained" and k < len(s["evasion"]):
            ev_by_id[r["id"]] = float(s["evasion"][k]); k += 1
    out = WG.dose_response_measured(rows, ev_by_id, n_perm=800)
    ok = (out.get("status") == "ok"
          and out["median_within_speaker_rho"] is not None
          and out["median_within_speaker_rho"] > abs(out["rho_pooled"]))
    check("within-speaker gradient exceeds the pooled one", ok,
          f"within {out.get('median_within_speaker_rho')} vs pooled {out.get('rho_pooled')}")


def test_posterior_is_wider_when_the_match_is_poor():
    """An inference layer that never expresses doubt is a liability."""
    from theroux import inference
    ref = inference.build_reference(W.GameParams(n_speakers=10), n_runs=3, seed=55)
    typical = {f: 0.0 for f in ref.feats}
    extreme = {f: 9.0 for f in ref.feats}
    a = inference.posterior(typical, ref)
    b = inference.posterior(extreme, ref)
    check("an out-of-distribution window reports a worse match",
          b["match_quality"] > a["match_quality"],
          f"{a['match_quality']} vs {b['match_quality']}")


if __name__ == "__main__":
    print("\nworld / wargame property tests\n")
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        fn()
    print(f"\n{len(PASS)}/{len(PASS) + len(FAIL)} passed")
    if FAIL:
        print("failed: " + ", ".join(FAIL))
        sys.exit(1)
