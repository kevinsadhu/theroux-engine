# Theroux

**Measuring constrained disclosure: the linguistic residue left when a speaker holds
material information they cannot state directly.**

Every statement is scored against a longitudinal model of *that speaker's own
language*, not against an absolute standard. The question is never "is this language
evasive" but "is this language evasive **for this person**".

This is not lie detection. No output claims any speaker was untruthful, and that is a
scientific and legal commitment rather than a style preference.

Grounding: Larcker & Zakolyukina, *Detecting Deceptive Discussions in Conference
Calls*, Journal of Accounting Research (2012).

---

## The current corpus is real

**472 Federal Reserve speeches · 11 speakers with fitted baselines · 23 constrained
windows · Feb 2021 – Sep 2026 · 1.18M words.**

The Fed corpus was chosen for one reason: **we do not get to decide who is
constrained.** The FOMC publishes a blackout calendar in advance — no policy comment
from the second Saturday before a meeting through the day after — for institutional
reasons unrelated to what any individual official knows. Treatment assignment is
arithmetic on a public calendar, not our judgement.

It also solves a confound we could not fix by regression. In a corporate corpus,
constrained windows *are* media interviews and baselines *are* earnings calls, so
format is nearly collinear with treatment; residualising on event type drove the
signal to chance. Fed officials give the same kind of speech inside and outside
blackout. Format is held by design.

---

## What we found

### 1. The constraint is real, and this needs no linguistics at all

The blackout covers **28.4% of the calendar** and contains **4.9% of speeches** —
a **5.8× suppression** of public speaking while the constraint is in force.

That is unarguable evidence the treatment bites, established from a published
calendar and a count. It is also the structural limit of this corpus: a constraint
that stops people speaking necessarily yields few constrained observations.

### 2. The dose-response prediction FAILED

The generative model in `world.py` predicts that divergence should rise as the
constraint binds harder — that is, as the meeting approaches. Tested within-speaker,
which is the correct level (between-speaker variation swamps the gradient; pooled
ρ ≈ 0.12 against within-speaker ≈ 0.50 in simulation):

| | |
|---|---|
| Median within-speaker ρ | **−0.80** |
| Blocked permutation p | 0.99 |
| Speakers with a testable gradient | 5 |
| Fraction pointing the predicted way | 20% |

**Divergence falls as disclosure approaches.** That is the opposite of the
prediction, consistent across 4 of 5 speakers.

We cannot yet separate two explanations, and we are not going to pretend otherwise:

- The hypothesis is wrong.
- **Selection.** The 5% who speak during blackout are self-selected, and their
  remarks are the safest and most pre-cleared they have. Heavier vetting closer to a
  meeting would *reduce* divergence-from-self, because clearance regresses everyone
  toward a house style. This predicts exactly the sign we observe.

The second is testable and is the next piece of work: score Q&A and press-conference
transcripts, where remarks are extempore, rather than prepared speeches largely
drafted by staff.

### 3. Two of four method-validation studies fail

Run `python3 scripts/run_experiments.py`.

| Study | Result | |
|---|---|---|
| Recovery of true ranking | median ρ **+0.51** over 8 corpora | passes |
| Baselining advantage | flat at +0.02 to +0.05, no trend | **fails** |
| Stage-by-stage ablation | full ladder gains +0.03 over raw lexical | **fails** |
| Power | 0.00 at every n tested | underpowered |

The estimator measures what it claims to. The claim that per-speaker baselining beats
absolute scoring — our core design argument — is **not demonstrated** on synthetic
data. It shows a modest advantage on false positives (see below) and nothing more.

---

## The statistical spine

Six stages, each fixing a specific failure in the one before it.

| # | Stage | Module | The failure it fixes |
|---|---|---|---|
| 1 | Features | `scorer.py`, `surprisal.py` | Word lists can be coached against, so we add a lexicon-free information-theoretic feature that fails differently |
| 2 | Confounds | `confounds.py` | Length drives lexical rates mechanically; uncontrolled, the model learns "short answers are evasive" |
| 3 | Shrinkage | `estimator.py` | With few baselines and six features the sample covariance is singular and the mean is high-variance |
| 4 | Anomaly | `anomaly.py` | Independent z-scores double-count correlated dimensions |
| 5 | Projection | `anomaly.evasion_score` | Mahalanobis is unsigned — it cannot tell evasive from unusually candid |
| 6 | Validation | `validate.py`, `model.py` | In-sample fit and naive p-values are meaningless at this n |

Ledoit–Wolf covariance shrinkage · James–Stein mean pooling · Mahalanobis with χ²
tail and whitened per-feature attribution · Dirichlet-smoothed unigram models
(Zhai & Lafferty 2001) · blocked permutation testing · leave-one-speaker-out CV.

**Every score carries provenance.** No score path exists that cannot cite itself.

---

## Run it

```bash
pip install -r requirements.txt

python3 run.py                       # score, validate, rebuild the dashboard
open dashboard/theroux.html          # the whole product, one file, no server
```

Rebuild the real corpus (needs unproxied network):

```bash
python3 scripts/fetch_fed.py --since 2021-01-01 --dry-run
python3 scripts/fetch_fed.py --since 2021-01-01
python3 run.py && python3 scripts/export_runtime.py && python3 run.py
python3 scripts/run_wargame.py
```

The heavier analyses:

```bash
python3 scripts/run_experiments.py   # four validation studies
python3 scripts/run_wargame.py       # adoption, feature survival, dose-response
python3 scripts/scenario.py --list   # speaker archetypes and presets
python3 scripts/scenario.py --preset skeptic   # run this before believing anything
```

## Tests

```bash
python3 tests/test_statistics.py     # 15 — the statistical spine
python3 tests/test_scorer.py         #  4 — scoring
python3 tests/test_fed_windows.py    # 16 — FOMC blackout labelling and the parser
python3 tests/test_parity.py         # 13 — Python and browser scorers agree (needs node)
python3 tests/test_world.py          # 14 — the generative game model (~10 min)
```

`test_parity.py` matters more than it looks. The dashboard scores text live in the
browser, which means two implementations of one scorer — the classic way to end up
with two answers. The JS file contains no methodology at all: every word list,
coefficient and baseline is read from `runtime.json`, generated from `lexicon.py`.
The test scores identical statements through both and fails above 2e-4. It has
already caught one real bug, when a vocabulary cap tuned for a synthetic corpus
silently dropped 62% of the real one.

## Bugs we found and kept in the source

Documented rather than deleted, because each is a way to manufacture a finding:

- **Silent mislabelling.** The FOMC calendar covered 2024–2026, so a `--since 2021`
  pull labelled every earlier speech "baseline" by default — 7 of them wrongly.
  A contaminated control group looks exactly like a clean one and yields a confident,
  well-formed, wrong null. `label_window` now refuses out-of-range dates.
- **A one-sentence length difference** between baseline and constrained windows in the
  simulator changed concrete-marker density and manufactured a −0.95σ effect where
  nothing was withheld.
- **Blocking a constant.** A within-speaker permutation run against a speaker-level
  quantity shuffled a constant, the null went degenerate, and power came back flat at
  every n — which looked like the method failing to scale when it was the test being
  inapplicable.

## Layout

| You want to… | File |
|---|---|
| Change what counts as hedging | `src/theroux/lexicon.py` |
| Change the a-priori evasion direction | `src/theroux/anomaly.py` |
| Change the game's assumptions | `src/theroux/world.py` |
| Add a speaker archetype | `world.ARCHETYPES` |
| Add a data source | `src/theroux/sources/` |
| Change the dashboard | `dashboard/index.html`, then `python3 run.py` |

`CLAUDE.md` is the full working guide. Read it before changing anything.

## Data

Federal Reserve speeches are US Government works in the public domain. `runtime.json`
ships per-speaker unigram counts — a redistributable language model of a named
person's public remarks — so treat it as an artefact with a data-rights question
attached rather than as a build output.

## Licence

MIT. See `LICENSE`.
