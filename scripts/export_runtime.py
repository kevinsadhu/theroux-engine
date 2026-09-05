"""
Export everything needed to score a NEW statement, client-side, with no server.

    python scripts/export_runtime.py     # writes dashboard/runtime.json

WHY THIS EXISTS
---------------
Up to now the dashboard has been a report: it renders scores that Python computed
at build time. That makes it a document, not a tool. An analyst cannot point it at
a speaker who said something this morning.

This ships the *model itself* — lexicon, per-speaker baselines, per-speaker language
models, the residualiser, the evasion direction — so the page can score arbitrary
text against any speaker in the roster, instantly, offline, with the same provenance
trail the batch pipeline produces.

THE DRIFT PROBLEM, AND HOW IT IS HANDLED
----------------------------------------
Two implementations of one scorer is how you get two answers. So:

  * The word lists are NOT reimplemented in JavaScript. They are exported from
    `lexicon.py` — the single source of truth — and the JS scorer is a pure
    interpreter of this file. Editing the lexicon changes both implementations at
    once, because there is only one lexicon.
  * `tests/test_parity.py` scores the same texts through Python and through the
    JavaScript (under node) and asserts they agree to 4 decimal places. If the two
    ever diverge, that test fails and this file is why.

WHAT IS SHIPPED, AND WHAT THAT MEANS FOR PRIVACY
------------------------------------------------
Per-speaker unigram counts are a language model of that speaker's public remarks.
On this corpus that is synthetic text. On a real deployment it is derived from
published transcripts, but it is still a redistributable model of a named person's
speech, so treat `runtime.json` as an artefact with a data-rights question attached
rather than as a build output. If that is a problem for a customer, the same scorer
runs server-side against the same file and only the scores leave the building.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from theroux import anomaly, confounds, features as F, lexicon as LX, surprisal  # noqa: E402
from theroux.estimator import fit_baselines  # noqa: E402
from theroux.pipeline import load_corpus, MATERIAL_MOVE_PCT  # noqa: E402

#: Vocabulary caps.
#:
#: These were set for a 136-statement synthetic corpus and were badly wrong for a real
#: one: the Fed corpus has a 15,838-word background vocabulary, so a 6,000 cap silently
#: dropped 62% of it. Every word above the cap then has count 0 in the browser but a
#: real count in Python, so the two cross-entropies diverge — the browser scorer was
#: quietly wrong on the very corpus it was meant to serve. tests/test_parity.py caught
#: it, which is the entire reason that test exists.
#:
#: Truncation is now effectively off. A full background is a few hundred KB, which is
#: nothing next to being wrong, and any truncation that does happen is reported loudly
#: rather than noted in a field nobody reads.
MAX_VOCAB_PER_SPEAKER = 60000
MAX_BG_VOCAB = 200000


def main() -> None:
    corpus = load_corpus("data/corpus")
    F.score_corpus(corpus)
    lms, bg, bg_total = F.fit_language_models(corpus)
    F.attach_surprisal(corpus, lms)
    F.add_time_index(corpus)

    feats = F.FEATURES

    # ---- residualiser, fitted on the whole corpus ----
    Y = np.array([[r["scores"][f] for f in feats] for r in corpus])
    X, cov_names = confounds.build_design(corpus, ["x"])
    beta = confounds.fit_residualiser(X, Y)
    Yr = confounds.residualise(X, Y, beta)
    # A pasted statement has no position in time. We use the corpus mean, and say so
    # in the UI, because pretending to know when something was said would silently
    # change the score.
    mean_design = X.mean(axis=0)
    # build_design STANDARDISES log-length against the corpus, so scoring a new
    # statement needs the corpus moments, not just the beta matrix. Omitting these
    # would silently score every pasted statement as average-length.
    log_len_raw = np.array([np.log(max(r["meta"]["words"], 1)) for r in corpus])
    log_len_mean, log_len_sd = float(log_len_raw.mean()), float(max(log_len_raw.std(), 1e-6))

    # ---- per-speaker baselines on residualised features ----
    by_speaker: dict[str, np.ndarray] = {}
    for sp in sorted({r["speaker"] for r in corpus}):
        M = np.array([Yr[i] for i, r in enumerate(corpus)
                      if r["speaker"] == sp and r["window"] == "baseline"])
        if len(M) >= 2:
            by_speaker[sp] = M
    baselines = fit_baselines(by_speaker, feats, min_n=2)

    speakers = {}
    for sp, b in baselines.items():
        lm = lms.get(sp)
        if lm is None:
            continue
        top = dict(Counter(lm.counts).most_common(MAX_VOCAB_PER_SPEAKER))
        org = next((r.get("org", "") for r in corpus if r["speaker"] == sp), "")
        speakers[sp] = {
            "speaker": sp,
            "org": org,
            "n_baseline": int(b.n),
            "features": list(b.features),
            "mean": [round(float(v), 6) for v in b.mean],
            "sd": [round(float(v), 6) for v in np.sqrt(np.diag(b.cov))],
            "precision": [[round(float(v), 6) for v in row] for row in b.precision],
            "mean_shrinkage": round(float(b.mean_shrinkage), 4),
            "cov_shrinkage": round(float(b.cov_shrinkage), 4),
            "lm": {
                "counts": top,
                # `total` stays the TRUE total, not the truncated sum — the Dirichlet
                # denominator must match what Python used or every cross-entropy shifts.
                "total": int(lm.total),
                "vocab_size": int(len(lm.vocab)),
                "baseline_ce": round(float(lm.baseline_ce), 6),
                "baseline_ce_sd": round(float(lm.baseline_ce_sd), 6),
            },
        }

    bg_top = dict(Counter(bg).most_common(MAX_BG_VOCAB))

    runtime = {
        "generated_from": "data/corpus",
        "features": feats,
        "feature_labels": {**LX.DIMENSION_LABELS,
                           "surprisal_z": "Surprisal (vs own language model)"},
        "lexicon": {
            "HEDGES": LX.HEDGES,
            "DEFLECTORS": LX.DEFLECTORS,
            "CONFIDENCE": LX.CONFIDENCE,
            "FIRST_PERSON": LX.FIRST_PERSON,
            "IMPERSONAL": LX.IMPERSONAL,
            "NUMERIC_PATTERN": LX.NUMERIC_PATTERN,
            "PERIOD_PATTERN": LX.PERIOD_PATTERN,
            "NORMALISERS": LX.NORMALISERS,
            "DIMENSIONS": LX.DIMENSIONS,
        },
        "surprisal": {
            "alpha": surprisal.ALPHA,
            "min_tokens": surprisal.MIN_TOKENS,
            "token_pattern": "[a-z']+",
            "background": {"counts": bg_top, "total": int(bg_total),
                           "vocab_size": int(len(bg)),
                           "truncated": bool(len(bg_top) < len(bg))},
        },
        "residualiser": {
            "covariates": cov_names,
            "beta": [[round(float(v), 8) for v in row] for row in beta],
            "mean_design": [round(float(v), 8) for v in mean_design],
            "log_len_mean": round(log_len_mean, 8),
            "log_len_sd": round(log_len_sd, 8),
            # residualise() re-centres on the corpus feature means, so those are part
            # of the transform and have to travel with beta.
            "y_mean": [round(float(v), 8) for v in Y.mean(axis=0)],
        },
        "evasion_direction": anomaly.EVASION_DIRECTION,
        "speakers": speakers,
        "thresholds": {"flag": 1.5, "material_move_pct": MATERIAL_MOVE_PCT},
    }

    out = Path("dashboard/runtime.json")
    out.write_text(json.dumps(runtime, separators=(",", ":")))
    kb = out.stat().st_size / 1024
    print(f"\n  {len(speakers)} speakers · {len(feats)} features · "
          f"bg vocab {len(bg_top)} of {len(bg)}")

    truncated_speakers = [sp for sp, s in speakers.items()
                          if len(s["lm"]["counts"]) < s["lm"]["vocab_size"]]
    if len(bg_top) < len(bg) or truncated_speakers:
        print("\n  *** WARNING: VOCABULARY TRUNCATED ***")
        print("  The browser scorer will DISAGREE with the Python scorer, because a")
        print("  dropped token has count 0 in one and a real count in the other.")
        if len(bg_top) < len(bg):
            print(f"    background: kept {len(bg_top)} of {len(bg)} "
                  f"— raise MAX_BG_VOCAB")
        if truncated_speakers:
            print(f"    speakers:   {len(truncated_speakers)} truncated "
                  f"— raise MAX_VOCAB_PER_SPEAKER")
        print("  Run tests/test_parity.py to see the damage.\n")

    print(f"  → {out}  ({kb:.0f} KB)\n")


if __name__ == "__main__":
    main()
