# CLAUDE.md — working on this repo

Read this before changing anything.

## What this is

Theroux measures **constrained disclosure**: the linguistic residue left when a
speaker holds material information they cannot state directly. Every statement is
scored against a model of **that speaker's own language**, not an absolute standard.

This is not lie detection. We never claim a speaker is lying. Keep that distinction
in every user-facing string — it is a legal and scientific commitment, not a style
preference.

Grounding: Larcker & Zakolyukina, *Detecting Deceptive Discussions in Conference
Calls*, Journal of Accounting Research (2012).

## The statistical spine

Six stages, each fixing a specific failure in the one before it. Understand why each
exists before you change it.

| # | Stage | Module | The failure it fixes |
|---|---|---|---|
| 1 | Features | `features.py`, `scorer.py`, `surprisal.py` | Word lists can be coached against, so we add a lexicon-free information-theoretic feature that fails differently |
| 2 | Confounds | `confounds.py` | Length and time drive lexical rates mechanically; uncontrolled, the model learns "media hits are short" |
| 3 | Shrinkage | `estimator.py` | With 4 baselines and 6 features the sample covariance is singular and the mean is high-variance |
| 4 | Anomaly | `anomaly.py` | Independent z-scores double-count correlated dimensions; "6σ" is uninterpretable when features co-move |
| 5 | Projection | `anomaly.evasion_score` | Mahalanobis is unsigned — it cannot tell evasive from unusually candid |
| 6 | Validation | `validate.py`, `model.py` | In-sample fit and naive p-values are meaningless at this n |

### Specifics worth knowing

**Ledoit–Wolf shrinkage** (`estimator._ledoit_wolf`) — the finance covariance
estimator, here for the same reason it exists there: p large relative to n. Reports
its shrinkage intensity so you can see how much structure was borrowed.

**Empirical-Bayes pooling** (`estimator.empirical_bayes_weight`) — James–Stein
weight `tau²/(tau² + sigma²/n)`. A speaker with 20 baselines keeps their own mean;
one with 2 borrows from the population. Not a heuristic — it minimises expected
squared error.

**Mahalanobis + χ²** (`anomaly.score`) — `D² = (x-µ)ᵀΣ⁻¹(x-µ)`, and under
approximate normality `D² ~ χ²_k`, which converts distance into a **percentile**.
Also decomposes into whitened per-feature attribution, because a scalar distance is
useless for an audit trail.

**Blocked permutation** (`validate.permutation_test`) — statements cluster by
speaker, so shuffling naively inflates significance. `groups=` shuffles within
speaker. When blocked and bootstrap disagree, believe the blocked test.

**Leave-one-speaker-out** (`model.fit_and_validate`) — leaving out one *statement*
leaks through the shared baseline. Whole speakers are held out, which estimates
generalisation to someone unseen — the deployment case.

## What the current run actually shows

Run `python run.py`. On the synthetic corpus:

- A-priori evasion projection: **AUC 0.71**, ρ 0.38, blocked permutation **p = 0.22**
- Unsigned Mahalanobis: **AUC 0.51** — chance
- Learned direction: in-sample 0.75, **out-of-fold 0.48** — overfits by 0.27

Three findings, all of which should survive into any presentation:

1. **The a-priori direction beats the fitted one.** At n=42 across 15 speakers,
   learning the weights is worse than asserting them from the literature.
2. **Direction is the signal, not novelty.** Unsigned distance is chance.
3. **Nothing is significant.** The bootstrap CI excludes zero; the blocked
   permutation does not.

**Never present these numbers as evidence the thesis works.** The corpus is
synthetic — we generated the pattern, so finding it is circular. The purpose is to
prove the machinery is correct before real data arrives. `scripts/generate_corpus.py`
deliberately includes null speakers, heterogeneous effect sizes and unpredictable
shocks so the pipeline cannot trivially win.

## The method-validation studies, and the two that fail

`python scripts/run_experiments.py` writes `dashboard/experiments.json`, which the
Method view of the dashboard renders. Four studies, chosen because each can return a
negative answer:

| Study | Result | What it means |
|---|---|---|
| **Recovery of ranking** | median ρ **+0.51** over 8 corpora | **Passes.** Plant a different true leakiness per speaker; the pipeline recovers their ordering. The estimator measures what it claims to. |
| **Heterogeneity** | advantage flat at +0.02 to +0.05, trend +0.01 | **Fails.** Baseline-relative scoring should beat absolute scoring *and the gap should widen as speakers differ*. It doesn't. |
| **Ablation** | full ladder gains +0.03 over raw lexical | **Fails.** Only the signed projection earns its rung (+0.20). Confounds, Mahalanobis and surprisal each cost AUC here. |
| **Power** | 0.00 at every n tested | **Underpowered.** `power_not_increasing`. We refuse to extrapolate a required n from a flat curve. |

The heterogeneity result is the one that matters, because it is our moat claim.
Two readings are still open: the synthetic generator may not produce the kind of
between-speaker variation baselining exploits (`draw_style` samples style
independently of effect, so "who you are" and "what you're hiding" are uncorrelated
in a way real speakers are not), or the claim is wrong. **Do not resolve this with
more synthetic data.** It is resolvable only on the Fed corpus.

Two harness bugs were found and fixed while building these; both are documented in
`scripts/run_experiments.py` docstrings rather than deleted, because "the method
looked broken when the harness was" is the failure mode to stay alert to:

1. Recovery originally used a binary material-move label that goes degenerate at high
   planted effect (70% material) — AUC had nothing to discriminate. Now rank recovery.
2. Speaker style was drawn from `beta(2,5)`, making speakers nearly interchangeable,
   so baselining could not possibly help. Now `draw_style(rng, heterogeneity)`.

## Running it on real speakers

The dashboard is not a report any more. `scripts/export_runtime.py` ships the fitted
model — lexicon, per-speaker baselines, per-speaker language models, residualiser,
evasion direction — into `dashboard/runtime.json`, and `dashboard/scorer.js` scores
arbitrary text against any speaker **in the browser**, offline, with no server and no
transcript leaving the machine.

```bash
# 1. put real statements in data/corpus/ as JSON  (see sources/fed.py for the shape)
python run.py                       # fit baselines, validate, rebuild the dashboard
python scripts/export_runtime.py    # ship the model to the browser
python run.py                       # re-inject
```

Minimum viable roster: **2 baseline statements per speaker** for a baseline to be fitted
at all, 5+ before the UI stops labelling it thin, and the design grid says ~22 speakers ×
6 baselines to reach 0.8 power. Statements under 40 tokens skip the surprisal feature.

**Two scorers now exist** — Python for batch, JavaScript for live. That is a real hazard
and it is contained two ways: the JS file contains no methodology at all (every word
list, coefficient and baseline is read from `runtime.json`, generated from `lexicon.py`),
and `tests/test_parity.py` scores the same statements through both under node and fails
on any disagreement above 2e-4. Run it after touching `lexicon.py`, `scorer.py`,
`surprisal.py`, `confounds.py` or `scorer.js`.

**`runtime.json` is not an ordinary build artefact.** Per-speaker unigram counts are a
redistributable language model of a named person's speech. Synthetic here; on real
transcripts it carries a data-rights question. If a customer objects, the same
`scorer.js` runs server-side against the same file and only scores leave the building.

### Defining scenarios without touching code

`scripts/scenario.py` is the analyst-facing entry point. Populations are weighted mixes
of `world.ARCHETYPES`; any `GameParams` field is a CLI flag.

```bash
python scripts/scenario.py --list
python scripts/scenario.py --preset fed
python scripts/scenario.py --name my-roster \
    --mix career_official=0.7,genuinely_uncertain=0.2,null=0.1 \
    --constraint-base 0.5 --n-speakers 22
python run.py     # scenarios appear in the dashboard's scenario console
```

The one to run before believing anything else is `--preset skeptic`: a universe of honest
habitual hedgers who withhold nothing, where every alarm is by definition a false one.

## The simulation layer — what it is for, and what it is not

`world.py` is the only module here that does not measure. It models the speaker as an
agent holding private information `theta` under a constraint `c`, choosing how much to
disclose and — having withheld — whether to paper over the withholding (hedge, deflect)
or refuse flatly. `wargame.py` sweeps it; `scripts/run_wargame.py` writes
`dashboard/wargame.json`; the dashboard's Simulation view renders it.

**Everything it produces is model-conditional.** It cannot validate the thesis. It can
only say what our assumptions imply and how much data would settle them. Label it that
way in every user-facing string, exactly as we do with "not lie detection".

What it bought us, in order of value:

| Output | Finding |
|---|---|
| **Design grid** | **~22 speakers × 6 baseline statements reaches 0.86 power.** Depth is the better purchase than roster width — 3→6 baselines beats 14→22 speakers. That reorders data acquisition. |
| **Dose-response** | Divergence should rise as the disclosure date nears. Cheapest strong test we have: **needs no outcome data**, is within-speaker, and a published blackout calendar supplies the treatment. |
| **Adoption** | Register collapses 1.00 → 0.13 as speakers learn they are read; ρ with withheld information only falls 0.43 → 0.27. Behaviour changes far more than measurability. |
| **Feature migration** | Hedging collapses (retained 0.27); specificity avoidance holds (0.76); surprisal *gains*, because refusal is its own register. Partly built in — see below. |

Three cautions that must travel with these:

1. **The surprisal gain is partly an artifact.** We gave refusals their own text bank, so
   they read as novel vocabulary by construction. The model shows the mechanism is
   coherent, not that it is true of real speakers.
2. **Best response, not equilibrium.** The speaker optimises against a *fixed* analyst.
   We do not solve a fixed point, because we could not estimate one at this n.
3. **The two decay curves are two businesses.** Falling AUC-on-move degrades a trading
   product. Falling register *is* a compliance product working — it pushes speakers from
   talking around a subject to declining it on the record.

### Three bugs found building it, kept in the source rather than deleted

- **Binary withholding.** `d = 1 if kappa > c else 0` made the dose-response purely an
  extensive margin flipping at one date. Replaced with `d = kappa/(kappa+c)`, which never
  saturates. The intermediate fix, `clip((kappa-c)/kappa)`, saturated instead and dropped
  rho from 0.40 to 0.09 — worth knowing that the obvious repair was worse than the bug.
- **Length artifact.** Baseline windows drew 7–12 sentences, constrained 6–11. That one
  sentence changed concrete-marker density and manufactured a −0.95σ divergence on windows
  where *nothing was withheld*. Residualising on log-words did not absorb it.
  `tests/test_world.py` now guards it.
- **Blocking a constant.** The design grid ran a within-speaker permutation against
  `theta_held`, which is a speaker-level property. Shuffling inside a block shuffled a
  constant, the null went degenerate, and power came back flat at every n — which looked
  like the method failing to scale when it was the test being inapplicable. Blocking is
  right when the quantity varies within the block and wrong when it does not.

The mirror-image lesson sits in `dose_response_measured`: there the *pooled* statistic is
the wrong one, because between-speaker spread in theta swamps within-speaker movement in
the constraint (pooled ρ ~0.12 vs within-speaker ~0.50 on the same simulated data). Pick
the level of the test to match the level at which the quantity actually varies.

## The format problem, and why the Fed corpus is the answer

We tried residualising on event type and it drove the signal to chance. The reason
matters: in the corporate corpus, format is nearly collinear with the treatment.
Constrained windows *are* media hits; baselines *are* earnings calls. Regressing out
format removes the effect along with the nuisance.

Format cannot be controlled by regression here. It has to be controlled by **design** —
a corpus where both windows share a format. Federal Reserve speeches do exactly that:
the same speaker gives the same kind of speech inside and outside the FOMC blackout,
and the blackout is a *published calendar* rather than our judgement.

That is a scientific argument for the Fed corpus, not a convenience one. It is the
single highest-value next task. `sources/fed.py` is written and needs a network run.

## Where to make each kind of edit

| You want to… | Edit |
|---|---|
| Change what counts as hedging/deflection | `lexicon.py` |
| Swap lexical scoring for an LLM | `scorer.py` → `score_text_llm()` |
| Change the a-priori evasion direction | `anomaly.EVASION_DIRECTION` |
| Adjust shrinkage behaviour | `estimator.py` |
| Add/relax a confound | `confounds.build_design` |
| Change the material-move threshold | `pipeline.MATERIAL_MOVE_PCT` |
| Add transcripts | drop JSON in `data/corpus/` |
| Add a data source | new module in `sources/` |
| Change the dashboard | `dashboard/index.html`, then `python run.py` |
| Add/change a validation study | `experiments.py` + `scripts/run_experiments.py` |
| Change the game's assumptions | `world.GameParams`, `world.COACHABILITY` |
| Add a speaker archetype | `world.ARCHETYPES` |
| Add/change a war-game sweep | `wargame.py` + `scripts/run_wargame.py` |
| Define a scenario | `scripts/scenario.py` (CLI — no code change needed) |
| Change what ships to the browser | `scripts/export_runtime.py` |

## Domain extension

The ontology is `Speaker → Statement → Event → Outcome`, domain-agnostic by
construction.

1. **Corporate** (now) — executives pre-earnings; outcome = price reaction.
2. **Central banks** (next) — FOMC blackout; outcome = decision + yield move.
   Solves the format confound.
3. **Geopolitical** (later) — leader rhetoric before policy events. Outcomes rarely
   verifiable, which is precisely why 1 and 2 exist.

Adding a domain should require a `sources/` module and a corpus — nothing else. If a
new domain would force changes to `scorer.py` or `estimator.py`, the abstraction is
wrong. Say so rather than special-casing.

## Palantir AIP mapping

| Local | AIP |
|---|---|
| `data/corpus/` + schema | Ontology: Speaker, Statement, Event, Outcome objects |
| `estimator.py` baselines | Longitudinal properties on the Speaker object |
| `scorer.py` | AIP Agent Studio extraction agents with citations |
| `pipeline.py` | AIP Logic functions |
| `dashboard/` | Workshop application |
| `sources/` | Pipeline Builder scheduled ingestion |

Keep this true as the code changes — it is load-bearing for the fellowship application.

## Conventions

- Python 3.11+. Core numerics: numpy, scipy, scikit-learn. Network code lives in
  `sources/` and degrades gracefully offline.
- **Every score carries provenance.** No score path that cannot cite itself.
- **Never fabricate a quote attributed to a real person.** Synthetic records carry
  `"synthetic": true`; real ones carry `source_url`.
- **Always report n beside any statistic**, and say plainly when nothing is significant.
- Tests assert *properties*, not pinned numbers. `python tests/test_statistics.py`.

## Highest-value next tasks, in order

1. **Run `sources/fed.py`** against a network. Real speakers, real published
   constraint windows, format held constant. Everything else is downstream of this.
2. **Wire `score_text_llm()`.** Anonymise speaker and company in the prompt — a
   frontier model may recall the actual outcome and leak it into the score.
3. **Prospective validation.** Score statements before outcomes land. This is the
   only design that fully rules out contamination, and it is a first-quarter priority.
4. **Sequential change detection.** CUSUM over each speaker's statement stream to
   detect *when* a linguistic regime shifted, rather than scoring windows in isolation.
5. **Hierarchical model.** Partial pooling of the evasion direction across speakers
   (currently pooled only in the mean) — the natural next step once n supports it.
