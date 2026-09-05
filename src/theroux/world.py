"""
The disclosure game — a generative model in which the speaker is a strategic agent.

Everything else in this repo MEASURES. This module SIMULATES, and the distinction
matters: nothing here is evidence about the world. A simulator can only tell you
what a set of assumptions implies, and how much data you would need to tell those
assumptions apart from the alternative. Any number produced by this module is
model-conditional and must be labelled that way wherever it surfaces.

Why bother, then. Three reasons, in order of how much they earn their place:

  1. IT DERIVES A SHARP PREDICTION. Our validation so far tests divergence against
     an *outcome*, which is noisy, late, and confounded. The game model implies
     something stronger and much cheaper to test: divergence should scale with how
     BINDING the constraint is, and the constraint binds harder as the disclosure
     date approaches. That is a within-speaker dose-response gradient requiring no
     outcome data at all. See `constraint_at`.

  2. IT ANSWERS THE COACHING QUESTION QUANTITATIVELY. "What happens when speakers
     know you are measuring them" is the first question any serious reviewer asks,
     and rhetoric is not an answer. Here it is a parameter (`adoption`) and the
     answer is a curve.

  3. IT MAKES THE MOAT ARGUMENT MECHANICAL RATHER THAN RHETORICAL. See COACHABILITY.

---------------------------------------------------------------------------
THE GAME
---------------------------------------------------------------------------

A speaker holds private information of magnitude `theta` (0 = nothing material).
A constraint `c` in [0,1] says how prohibited disclosure currently is — quiet
period, MNPI rules, active litigation, competitive harm.

The speaker chooses two things:

    d   how much of theta to actually disclose
    r   REGISTER: having withheld, how much to paper over the withholding with
        hedging and deflection, versus refusing flatly and visibly

Costs:

    legal        c * d * theta            revealing prohibited material is costly
    credibility  kappa * (1-d) * theta * (1-r)
                                          bare silence is itself informative — this
                                          is the Grossman/Milgrom unravelling force,
                                          and papering over is what blunts it
    detection    delta * adoption * r^2   if the market reads register, papering over
                                          is itself costly

Best responses (`best_response`), not an equilibrium — we solve the speaker's
problem against a FIXED analyst, we do not find a fixed point. Saying so is the
difference between game theory and the word "game theory".

    r* = kappa(1-d)theta / (kappa(1-d)theta + 2 delta * adoption)

    adoption = 0  ->  r* = 1     paper over completely; it is free. This is today.
    adoption ↑    ->  r* ↓       speakers abandon hedged evasion for flat refusal.
    theta   = 0   ->  r* = 0     nothing to paper over.

THE PREDICTION THAT FALLS OUT is worth stating plainly, because it is not the one
we expected and it cuts against a naive reading of the product: **under adoption,
the market does not become unreadable — it becomes blunter.** Speakers stop
hedging and start saying "I won't answer that." For a trading customer that
degrades the alpha. For a compliance or regulatory customer it is the product
working: it induces cleaner, more legible disclosure. Those are different
businesses and they decay in opposite directions.

---------------------------------------------------------------------------
WHY THE SIGNAL SHOULD EXIST AT ALL
---------------------------------------------------------------------------

Grossman (1981) and Milgrom (1981): if disclosure were costless and verifiable,
markets would unravel to full disclosure — silence proves bad news, so everyone
with good news speaks, and the pool of silent speakers collapses. Real markets do
not unravel, because disclosure is *constrained*: Reg FD, MNPI, litigation
exposure, competitive harm.

Theroux measures the residue in exactly the gap where unravelling fails. That is a
theoretical reason to expect the signal to exist, which is worth more than another
AUC point — and it also predicts *where* to look: wherever disclosure is legally
prohibited on a published calendar. Which is the FOMC blackout, which is
`sources/fed.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field

import numpy as np

# ---------------------------------------------------------------------------
# Differential coachability — the mechanical moat argument
# ---------------------------------------------------------------------------
#
# A prepared speaker can defeat a word list. They cannot defeat a constraint on
# the information itself. These numbers are ASSERTED from the mechanism, not
# fitted — same epistemic status as anomaly.EVASION_DIRECTION — but the ordering
# is defensible and it is the ordering that carries the argument:
#
#   hedging / confidence   pure word choice. A media trainer fixes this in an
#                          afternoon. Nearly fully coachable.
#   deflection / distance  structural but paraphrasable. Partly coachable.
#   specificity avoidance  YOU CANNOT STATE A NUMBER YOU ARE LEGALLY BARRED FROM
#                          STATING. No amount of training produces the figure.
#   surprisal              defeating this means permanently changing how you speak,
#                          including on the days you have nothing to hide — which
#                          forfeits the credibility you need on those days, and is
#                          detectable longitudinally as a baseline shift.
#
# So the two features that survive coaching are precisely the two that are not
# about word choice. That is the moat, stated as a mechanism rather than a claim.

COACHABILITY = {
    "hedging": 0.85,
    "confidence_language": 0.80,
    "pronoun_distancing": 0.60,
    "topic_deflection": 0.55,
    "specificity_avoidance": 0.15,
    "surprisal_z": 0.10,
}

T_MAX_DAYS = 30.0   # horizon over which constraint bindingness ramps


@dataclass
class GameParams:
    """Every knob the war-game exposes. Defaults are today's world."""

    # --- population ---
    n_speakers: int = 14
    baseline_n: int = 6
    constrained_n: int = 3
    heterogeneity: float = 0.9      # how much speakers differ from one another
    p_null: float = 0.30            # speakers holding nothing at all

    # --- private information ---
    theta_scale: float = 0.55       # magnitude of material non-public information

    # --- the constraint ---
    constraint_base: float = 0.15   # bindingness far from the disclosure date
    dose_response: float = 0.85     # how much harder it binds as T -> 0

    # --- the adversarial dimension ---
    adoption: float = 0.0           # fraction of speakers who know they are read
    coaching_skill: float = 0.75    # how well an adapted speaker suppresses tells

    # --- payoffs ---
    kappa: float = 0.60             # credibility cost of bare silence (unravelling)
    delta: float = 0.45             # cost of being *detected* papering over

    # --- outcome process ---
    outcome_beta: float = 15.0      # how much held information moves the price
    outcome_noise: float = 4.2
    p_shock: float = 0.20           # big moves with no linguistic antecedent

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Text banks the base generator does not have
# ---------------------------------------------------------------------------
#
# `generate_corpus.compose` entangles hedging with specificity: its vague banks are
# simultaneously hedged AND number-free, so there is no way to express the register
# an adapted speaker actually adopts. We need one more bank — the flat refusal.
#
# It is direct, first-person, unhedged, confident in tone, and contains no figures.
# That combination is the whole point: it is what a coached speaker under a real
# legal constraint sounds like, and it defeats every lexical feature we have while
# leaving specificity avoidance untouched.

REFUSE = [
    "I'm not going to give you that number today.",
    "That's not something I can get into on this call, and I'll leave it there.",
    "We'll address that at the appropriate time and not before.",
    "I won't speculate on that, and I'd rather say so plainly than talk around it.",
    "You'll have to wait for the filing on that one.",
    "I'm going to decline that question rather than give you a half answer.",
]


# ---------------------------------------------------------------------------
# Speaker archetypes — the unit an analyst actually thinks in
# ---------------------------------------------------------------------------
#
# Nobody covers "a speaker with theta_scale 0.55". They cover a media-trained CFO, a
# general counsel under active litigation, a founder who genuinely does not know yet.
# These are those people, as parameter bundles.
#
# `theta_mult`     scales how much material information this type actually holds
# `style_bias`     shifts their BASELINE register — who they are when nothing is at stake
# `overrides`      GameParams fields this type replaces
# `always_adapted` this type is coached regardless of the population adoption rate
#
# The load-bearing one is `genuinely_uncertain`. It is the objection every reviewer
# raises — "aren't you just detecting anxious people?" — and it is the case
# per-speaker baselining exists to handle. A habitual hedger who is withholding
# nothing should NOT flag, because their hedging is their own norm and we score
# divergence from that norm rather than from an absolute standard. If this archetype
# flags at the population rate, the entire premise of the product is wrong, and
# `wargame.archetype_rates` is the test that says so.

@dataclass
class Archetype:
    name: str
    label: str
    description: str
    theta_mult: float = 1.0
    style_bias: dict = field(default_factory=dict)
    overrides: dict = field(default_factory=dict)
    always_adapted: bool = False


ARCHETYPES: dict[str, Archetype] = {
    "naive_executive": Archetype(
        "naive_executive", "Naive executive",
        "Holds material information, not media-trained. Papers over freely because "
        "doing so is costless when nobody is reading register. The base case."),
    "coached_executive": Archetype(
        "coached_executive", "Coached executive",
        "Same information, professionally media-trained. Suppresses hedging and "
        "confidence tells; cannot suppress the numbers they are barred from giving.",
        always_adapted=True, overrides={"coaching_skill": 0.9}),
    "legally_muzzled": Archetype(
        "legally_muzzled", "Legally muzzled",
        "Under quiet period or active litigation. Counsel has told them to decline "
        "rather than talk around it, so they refuse flatly even when unobserved — "
        "low credibility pressure, very high constraint.",
        overrides={"constraint_base": 0.65, "kappa": 0.22}),
    "genuinely_uncertain": Archetype(
        "genuinely_uncertain", "Genuinely uncertain",
        "Withholding NOTHING, but habitually hedges and rarely commits — they simply "
        "do not know yet. The false-positive case the whole method must survive.",
        theta_mult=0.0,
        style_bias={"evasive": +0.26, "confident": -0.09, "deflect": +0.10}),
    "career_official": Archetype(
        "career_official", "Career official",
        "Central banker or senior civil servant. Formal, low-variance, tightly "
        "scripted register, real published constraint, modest private information. "
        "Tests whether the method works when baseline variance is small.",
        theta_mult=0.65,
        style_bias={"evasive": +0.10, "confident": -0.06},
        overrides={"heterogeneity": 0.35, "constraint_base": 0.45}),
    "null": Archetype(
        "null", "Null speaker",
        "Nothing withheld, unremarkable style. The control: any signal found here "
        "is manufactured.",
        theta_mult=0.0),
}

#: A population is a weighted mix of archetypes — an analyst's actual coverage universe.
DEFAULT_POPULATION = {"naive_executive": 0.45, "coached_executive": 0.15,
                      "legally_muzzled": 0.10, "genuinely_uncertain": 0.15,
                      "career_official": 0.05, "null": 0.10}


def resolve_population(pop: dict[str, float] | None) -> dict[str, float]:
    """Validate and normalise a population mix to weights summing to 1."""
    pop = dict(pop or DEFAULT_POPULATION)
    unknown = set(pop) - set(ARCHETYPES)
    if unknown:
        raise ValueError(f"unknown archetype(s): {sorted(unknown)}. "
                         f"available: {sorted(ARCHETYPES)}")
    total = sum(pop.values())
    if total <= 0:
        raise ValueError("population weights must sum to something positive")
    return {k: v / total for k, v in pop.items()}


def constraint_at(days_to_disclosure: float | None, p: GameParams) -> float:
    """
    How binding the prohibition is, as a function of time to the disclosure event.

    THIS FUNCTION IS THE TESTABLE PREDICTION. If divergence is driven by constraint
    rather than by personality or format, it must rise as T -> 0. That gradient is
    within-speaker, needs no outcome data, and is exactly what a published blackout
    calendar delivers cleanly.
    """
    if days_to_disclosure is None:
        return 0.0
    t = float(np.clip(days_to_disclosure, 0.0, T_MAX_DAYS))
    return float(np.clip(p.constraint_base + p.dose_response * (1.0 - t / T_MAX_DAYS),
                         0.0, 1.0))


def best_response(theta: float, c: float, p: GameParams) -> dict:
    """
    The speaker's choice of disclosure `d` and register `r`.

    A best response against a fixed analyst, not an equilibrium. We do not solve
    for a fixed point in which the analyst also re-optimises, because we would be
    asserting an equilibrium concept we cannot estimate at this n. The honest claim
    is: given how much the speaker believes they are being read, this is what they
    should do.
    """
    if theta <= 1e-9:
        return {"disclose": 1.0, "register": 0.0, "theta_held": 0.0}

    # Disclose to the point where the credibility cost of further silence no longer
    # outruns the legal cost of speech.
    #
    # Two earlier versions of this line were wrong in opposite directions and both
    # are worth remembering:
    #
    #   d = 1 if kappa > c else 0   made withholding binary. The whole dose-response
    #                               gradient then came from the EXTENSIVE margin —
    #                               whether you withhold at all — flipping at a single
    #                               threshold date. A Fed governor does not answer
    #                               freely at T−22 and go silent at T−20.
    #   d = clip((kappa-c)/kappa)   fixed that but saturated at d=0 for every c above
    #                               kappa, which is most of the range, so the gradient
    #                               vanished entirely (rho fell from 0.40 to 0.09).
    #
    # The ratio form never saturates: disclosure falls smoothly across the whole range
    # of bindingness, so there is a genuine intensive margin at every T.
    d = float(p.kappa / (p.kappa + max(c, 1e-9)))
    held = theta * (1.0 - d)

    # Register: paper over, unless being seen to paper over is expensive.
    num = p.kappa * held
    den = num + 2.0 * p.delta * p.adoption
    r = 1.0 if den <= 1e-12 else float(np.clip(num / den, 0.0, 1.0))

    return {"disclose": d, "register": r, "theta_held": float(held)}


def draw_style(rng: np.random.Generator, heterogeneity: float,
               bias: dict | None = None) -> dict:
    """
    A speaker's baseline register — who they are when nothing is at stake.

    `bias` shifts the centre of the distribution for an archetype. This is what makes
    the genuinely-uncertain case a real test rather than a relabelling: that speaker
    is *actually* a habitual hedger, in their baselines as much as their constrained
    windows, so an absolute scorer would flag them and a baseline-relative one should
    not.
    """
    mid = {"evasive": .34, "deflect": .16, "confident": .14, "distant": .28}
    span = {"evasive": .40, "deflect": .22, "confident": .22, "distant": .34}
    b = bias or {}
    return {k: float(np.clip(mid[k] + b.get(k, 0.0) + heterogeneity * span[k] * rng.normal(),
                             .02, .88))
            for k in mid}


def _compose(rng, banks, hedge, deflect, confident, distant, refuse, n_sent) -> str:
    """
    Compose a statement from the shared text banks plus the refusal bank.

    Mirrors generate_corpus.compose so the two corpora are directly comparable, with
    one addition: a `refuse` propensity that emits direct, number-free refusals. That
    is the register an adapted speaker moves to, and it is the reason the war-game
    can show lexical features collapsing while specificity avoidance holds.
    """
    G = banks
    pick = lambda b: b[int(rng.integers(0, len(b)))]
    nums = lambda: {"a": int(rng.integers(80, 990)), "b": int(rng.integers(3, 78)),
                    "c": int(rng.integers(10, 460)),
                    "d": round(float(rng.uniform(0.4, 3.8)), 1)}

    out = [pick(G.OPENERS_VAGUE if rng.random() < hedge else G.OPENERS_DIRECT)
           .format(**nums())]

    for _ in range(n_sent):
        u = rng.random()
        if u < refuse:
            out.append(pick(REFUSE))
        elif u < refuse + deflect:
            out.append(pick(G.DEFLECT))
        elif u < refuse + deflect + confident:
            out.append(pick(G.CONFIDENT))
        elif rng.random() < distant:
            out.append(pick(G.OWNERSHIP_DISTANT))
        elif rng.random() < hedge:
            out.append(pick(G.DETAIL_VAGUE))
        else:
            bank = G.DETAIL_DIRECT if rng.random() > 0.35 else G.OWNERSHIP_FIRST
            out.append(pick(bank).format(**nums()))

    # The closer is where specificity lives or dies: a speaker who cannot give the
    # number cannot give it here either, however well coached they are.
    out.append(pick(G.CLOSERS_VAGUE if rng.random() < hedge else G.CLOSERS_DIRECT)
               .format(**nums()))
    return " ".join(out)


def _suppress(value: float, feature: str, adapted: bool, p: GameParams) -> float:
    """Apply coaching to one feature's expressed magnitude."""
    if not adapted:
        return value
    return value * (1.0 - p.coaching_skill * COACHABILITY[feature])


def simulate(p: GameParams, seed: int, banks=None,
             population: dict[str, float] | None = None) -> list[dict]:
    """
    Run the game and emit a corpus in the pipeline's row format.

    Every row carries its own latent truth (`theta`, `theta_held`, `register`,
    `constraint`, `adapted`) so downstream studies can score against what actually
    happened rather than against a noisy proxy. Those fields are for evaluation
    only — nothing in `theroux.pipeline` reads them.
    """
    if banks is None:
        import importlib.util
        import pathlib
        spec = importlib.util.spec_from_file_location(
            "_gc", pathlib.Path(__file__).parents[2] / "scripts" / "generate_corpus.py")
        banks = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(banks)

    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    # Assign archetypes across the roster. Deterministic allocation rather than
    # per-speaker sampling, so that a 15% coached share actually yields ~15% coached
    # speakers at n=14 instead of whatever the coin flips gave us — an analyst who
    # says "a fifth of my universe is media-trained" should get that, not a draw.
    mix = resolve_population(population)
    names, weights = list(mix), np.array([mix[k] for k in mix], float)
    counts = np.floor(weights * p.n_speakers).astype(int)
    while counts.sum() < p.n_speakers:                    # distribute the remainder
        counts[int(np.argmax(weights * p.n_speakers - counts))] += 1
    assignment = [names[i] for i, c in enumerate(counts) for _ in range(int(c))]
    rng.shuffle(assignment)

    for i in range(p.n_speakers):
        spk = f"{banks.FIRST[i % len(banks.FIRST)]}_{i}"
        org = banks.ORGS[i % len(banks.ORGS)]

        arch = ARCHETYPES[assignment[i]]
        # Per-speaker parameters: the population defaults with this type's overrides.
        sp_p = GameParams(**{**p.as_dict(), **arch.overrides})

        style = draw_style(rng, sp_p.heterogeneity, arch.style_bias)
        adapted = bool(arch.always_adapted or rng.random() < sp_p.adoption)

        theta = 0.0 if rng.random() < sp_p.p_null else \
            float(np.clip(rng.gamma(2.2, sp_p.theta_scale / 2.2), 0.0, 1.0))
        theta *= arch.theta_mult

        # ---- baseline window: no constraint, so no held information ----
        for j in range(p.baseline_n):
            drift = rng.normal(0, .05)
            rows.append({
                "id": f"{i}-b{j}", "speaker": spk, "org": org, "domain": "corporate",
                "event": "Quarterly earnings call Q&A",
                "event_type": banks.EVENTS_BASE[j % 2],
                "date": f"2024-{(j % 12) + 1:02d}-15", "window": "baseline",
                "days_to_disclosure": None, "outcome_move_pct": None, "outcome": None,
                "synthetic": True, "simulated": True, "source_url": None,
                "theta": theta, "theta_held": 0.0, "register": 0.0,
                "constraint": 0.0, "adapted": adapted, "archetype": arch.name,
                "text": _compose(rng, banks,
                                 np.clip(style["evasive"] + drift, 0, 1),
                                 np.clip(style["deflect"] + drift / 2, 0, 1),
                                 np.clip(style["confident"] + drift / 2, 0, 1),
                                 np.clip(style["distant"] + drift, 0, 1),
                                 0.0, int(rng.integers(7, 12))),
                # NB: baseline and constrained windows draw statement length from the
                # SAME range. An earlier version drew 7–12 and 6–11, and the one-sentence
                # difference produced a spurious −0.95σ divergence on windows where
                # nothing was withheld: short statements carry the numeric opener and
                # closer as a larger share of their words, so concrete-marker density
                # rose and specificity avoidance fell. Confound residualisation on
                # log-words did not fully absorb it. Worth remembering as the cheapest
                # possible way to manufacture a finding.
            })

        # ---- constrained windows: the game is played here ----
        for j in range(p.constrained_n):
            T = float(rng.integers(2, 28))
            c = constraint_at(T, sp_p)
            br = best_response(theta, c, sp_p)
            held, r = br["theta_held"], br["register"]
            nz = rng.normal(0, .05)

            # Held information expresses itself two ways. The papered-over channel
            # (scaled by r) is lexical and coachable. The structural channel is not:
            # specificity avoidance is driven by held * c directly, because the
            # constraint is on the information rather than on the words.
            hedge_amt = _suppress(held * r * 1.00, "hedging", adapted, sp_p)
            deflect_amt = _suppress(held * r * 0.70, "topic_deflection", adapted, sp_p)
            conf_amt = _suppress(held * r * 0.35, "confidence_language", adapted, sp_p)
            distant_amt = _suppress(held * r * 0.60, "pronoun_distancing", adapted, sp_p)
            vague_amt = _suppress(held * c * 0.90, "specificity_avoidance", adapted, sp_p)
            refuse_amt = float(np.clip(held * (1.0 - r) * 0.85, 0.0, 0.6))

            # `evasive` in the base generator drives hedging and number-suppression
            # together, so it takes the larger of the two channels.
            evasive_knob = max(hedge_amt, vague_amt)

            move = held * sp_p.outcome_beta * rng.uniform(.5, 1.3) + rng.normal(0, sp_p.outcome_noise)
            if rng.random() < sp_p.p_shock:
                move += rng.normal(0, 11.0)
            sign = -1 if rng.random() < (.5 + .22 * (held > .18)) else 1
            move = round(float(sign * abs(move)), 1)

            rows.append({
                "id": f"{i}-c{j}", "speaker": spk, "org": org, "domain": "corporate",
                "event": ["Media interview", "Conference fireside",
                          "Investor conference"][j % 3],
                "event_type": banks.EVENTS_CONS[j % 3],
                "date": f"2025-{(j % 12) + 1:02d}-10", "window": "constrained",
                "days_to_disclosure": int(T),
                "outcome_move_pct": move,
                "outcome": "miss" if move < -2.5 else "beat" if move > 2.5 else "inline",
                "synthetic": True, "simulated": True, "source_url": None,
                "theta": theta, "theta_held": held, "register": r,
                "constraint": c, "adapted": adapted, "archetype": arch.name,
                "text": _compose(rng, banks,
                                 np.clip(style["evasive"] + evasive_knob + nz, 0, 1),
                                 np.clip(style["deflect"] + deflect_amt + nz, 0, 1),
                                 np.clip(style["confident"] + conf_amt + nz, 0, 1),
                                 np.clip(style["distant"] + distant_amt + nz, 0, 1),
                                 refuse_amt, int(rng.integers(7, 12))),
            })

    return rows
