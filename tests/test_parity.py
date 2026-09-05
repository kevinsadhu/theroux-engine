"""
Python/JavaScript parity.

There are now two scorers: `theroux.scorer` + `theroux.surprisal` in Python, and
`dashboard/scorer.js` running in the analyst's browser. Two implementations of one
method is how a system starts giving two answers to the same question, and the
customer finds out before you do.

This test forecloses that. It scores the same statements through both and fails on
any disagreement. Run it whenever `lexicon.py`, `scorer.py`, `surprisal.py`,
`confounds.py` or `scorer.js` changes.

    python tests/test_parity.py

Requires node and a current dashboard/runtime.json:

    python scripts/export_runtime.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from theroux import anomaly, confounds, features as F, scorer, surprisal  # noqa: E402
from theroux.estimator import fit_baselines  # noqa: E402
from theroux.pipeline import load_corpus  # noqa: E402

# Rounding differs at the last bit: Python's round() is banker's rounding, JS's
# Math.round is half-up. On a 4-decimal score that is a 1e-4 disagreement at worst,
# so the tolerance is set just above it rather than pretending the two agree exactly.
TOL = 2e-4

PROBES = [
    # deliberately spans the range: concrete, evasive, refusing, and short
    "Revenue was 482 million, up 17 percent year over year. Gross margin was 61 "
    "percent, up 240 basis points sequentially. I'll take the churn question "
    "directly: we lost two accounts and both were consolidation. We are holding "
    "the full year guide at 2 billion. Our backlog stands at 951 million, "
    "covering 68 percent of next year's plan. I want to be clear that we missed "
    "on the utilization line, and here is why. We added 274 net new accounts.",

    "I think the quarter developed broadly in line with how we've been describing "
    "it. The way I think about it is over a longer horizon than any single period. "
    "There is a process underway and the business will update in due course. "
    "Conditions have been somewhat dynamic in recent months. Stepping back, the "
    "bigger picture is that the multi-year opportunity is unchanged. It would be "
    "premature to characterize the outcome. Perhaps the useful framing is that the "
    "trajectory is intact. The market will ultimately determine how conditions resolve.",

    "I'm not going to give you that number today. That's not something I can get "
    "into on this call, and I'll leave it there. We'll address that at the "
    "appropriate time and not before. You'll have to wait for the filing on that "
    "one. I won't speculate on that, and I'd rather say so plainly than talk "
    "around it. I'm going to decline that question rather than give you a half "
    "answer. We are absolutely thrilled with how this has developed.",

    "Absolutely clearly tremendous. Without a doubt this is the strongest period "
    "we have had, and certainly the fundamentals have never been stronger.",

    "Short answer.",
]

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'pass' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))


def python_side(runtime: dict) -> dict:
    """Score the probes through the real Python pipeline, per speaker."""
    corpus = load_corpus("data/corpus")
    F.score_corpus(corpus)
    lms, bg, bg_total = F.fit_language_models(corpus)
    F.attach_surprisal(corpus, lms)
    F.add_time_index(corpus)

    feats = F.FEATURES
    Y = np.array([[r["scores"][f] for f in feats] for r in corpus])
    X, _ = confounds.build_design(corpus, ["x"])
    beta = confounds.fit_residualiser(X, Y)
    Yr = confounds.residualise(X, Y, beta)
    y_mean = Y.mean(axis=0)
    log_raw = np.array([np.log(max(r["meta"]["words"], 1)) for r in corpus])
    ll_mean, ll_sd = float(log_raw.mean()), float(max(log_raw.std(), 1e-6))
    t_mean = float(X.mean(axis=0)[2])

    by_speaker = {}
    for sp in sorted({r["speaker"] for r in corpus}):
        M = np.array([Yr[i] for i, r in enumerate(corpus)
                      if r["speaker"] == sp and r["window"] == "baseline"])
        if len(M) >= 2:
            by_speaker[sp] = M
    baselines = fit_baselines(by_speaker, feats, min_n=2)

    speakers = sorted(runtime["speakers"])[:4]
    out = {}
    for sp in speakers:
        b = baselines[sp]
        lm = lms[sp]
        for pi, text in enumerate(PROBES):
            lex = scorer.score_text(text)
            sur = lm.analyse(text)
            raw = np.array([
                (0.0 if sur.get("insufficient_tokens") else sur["surprisal_z"])
                if f == "surprisal_z" else lex["scores"][f]
                for f in feats])

            words = lex["meta"]["words"]
            design = np.array([1.0, (np.log(max(words, 1)) - ll_mean) / ll_sd, t_mean])
            resid = raw - design @ beta + y_mean

            d = resid - b.mean
            dirv = np.array([anomaly.EVASION_DIRECTION.get(f, 0.0) for f in feats])
            out[f"{sp}||{pi}"] = {
                "hedging": lex["scores"]["hedging"],
                "specificity_avoidance": lex["scores"]["specificity_avoidance"],
                "pronoun_distancing": lex["scores"]["pronoun_distancing"],
                "topic_deflection": lex["scores"]["topic_deflection"],
                "confidence_language": lex["scores"]["confidence_language"],
                "surprisal_z": (None if sur.get("insufficient_tokens")
                                else sur["surprisal_z"]),
                "cross_entropy": (None if sur.get("insufficient_tokens")
                                  else sur["cross_entropy"]),
                "novel_rate": (None if sur.get("insufficient_tokens")
                               else sur["novel_rate"]),
                "mahalanobis": float(np.sqrt(max(d @ b.precision @ d, 0))),
                "evasion": anomaly.directional_component(resid, b, dirv),
                "words": words,
                "n_evidence_hedging": len(lex["evidence"]["hedging"]),
            }
    return out, speakers


JS_HARNESS = """
const fs = require('fs');
const S = require(process.argv[2]);
const RT = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
const probes = JSON.parse(fs.readFileSync(process.argv[4], 'utf8'));
const speakers = JSON.parse(fs.readFileSync(process.argv[5], 'utf8'));
const out = {};
for (const sp of speakers) {
  probes.forEach((text, pi) => {
    const r = S.scoreAgainstSpeaker(text, sp, RT);
    const sur = r.surprisal || {};
    out[sp + '||' + pi] = {
      hedging: r.scores.hedging,
      specificity_avoidance: r.scores.specificity_avoidance,
      pronoun_distancing: r.scores.pronoun_distancing,
      topic_deflection: r.scores.topic_deflection,
      confidence_language: r.scores.confidence_language,
      surprisal_z: sur.insufficient_tokens ? null : sur.surprisal_z,
      cross_entropy: sur.insufficient_tokens ? null : sur.cross_entropy,
      novel_rate: sur.insufficient_tokens ? null : sur.novel_rate,
      mahalanobis: r.mahalanobis,
      evasion: r.evasion,
      words: r.words,
      n_evidence_hedging: r.evidence.hedging.length
    };
  });
}
process.stdout.write(JSON.stringify(out));
"""


def js_side(runtime_path: Path, speakers: list[str]) -> dict:
    tmp = Path(tempfile.mkdtemp())
    (tmp / "h.js").write_text(JS_HARNESS)
    (tmp / "probes.json").write_text(json.dumps(PROBES))
    (tmp / "speakers.json").write_text(json.dumps(speakers))
    res = subprocess.run(
        ["node", str(tmp / "h.js"), str((ROOT / "dashboard" / "scorer.js").resolve()),
         str(runtime_path.resolve()), str(tmp / "probes.json"), str(tmp / "speakers.json")],
        capture_output=True, text=True)
    shutil.rmtree(tmp, ignore_errors=True)
    if res.returncode != 0:
        print(res.stderr[:2000])
        raise SystemExit("node harness failed")
    return json.loads(res.stdout)


def main() -> None:
    print("\npython / javascript scorer parity\n")

    if shutil.which("node") is None:
        print("  SKIP  node not available")
        return
    rt_path = ROOT / "dashboard" / "runtime.json"
    if not rt_path.exists():
        raise SystemExit("run scripts/export_runtime.py first")
    runtime = json.loads(rt_path.read_text())

    # Vocabulary truncation would silently break parity, so assert it did not happen.
    check("background vocabulary was not truncated",
          not runtime["surprisal"]["background"].get("truncated", False))

    py, speakers = python_side(runtime)
    js = js_side(rt_path, speakers)

    check("same number of scored cells", len(py) == len(js),
          f"py {len(py)} js {len(js)}")

    fields = ["hedging", "specificity_avoidance", "pronoun_distancing",
              "topic_deflection", "confidence_language", "surprisal_z",
              "cross_entropy", "novel_rate", "mahalanobis", "evasion"]
    worst = {f: 0.0 for f in fields}
    worst_cell = {f: "" for f in fields}
    mismatched_ints = []

    for key, pv in py.items():
        jv = js.get(key)
        if jv is None:
            mismatched_ints.append(f"{key}: missing in JS")
            continue
        if pv["words"] != jv["words"]:
            mismatched_ints.append(f"{key}: words {pv['words']} vs {jv['words']}")
        if pv["n_evidence_hedging"] != jv["n_evidence_hedging"]:
            mismatched_ints.append(
                f"{key}: hedging evidence {pv['n_evidence_hedging']} vs {jv['n_evidence_hedging']}")
        for f in fields:
            a, b = pv[f], jv[f]
            if a is None or b is None:
                if (a is None) != (b is None):
                    mismatched_ints.append(f"{key}: {f} null mismatch")
                continue
            delta = abs(float(a) - float(b))
            if delta > worst[f]:
                worst[f], worst_cell[f] = delta, key

    check("word counts and evidence counts agree", not mismatched_ints,
          "; ".join(mismatched_ints[:3]))

    for f in fields:
        check(f"{f} agrees to {TOL}", worst[f] <= TOL,
              f"max delta {worst[f]:.2e} at {worst_cell[f]}")

    print(f"\n  {len(py)} cells compared across {len(speakers)} speakers "
          f"x {len(PROBES)} statements")
    print(f"\n{len(PASS)}/{len(PASS) + len(FAIL)} passed")
    if FAIL:
        print("failed: " + ", ".join(FAIL))
        sys.exit(1)


if __name__ == "__main__":
    main()
