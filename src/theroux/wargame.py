"""
War-gaming the disclosure game.

Runs `world.simulate` under varied assumptions and reports what survives. Four
studies, each answering a question that a reviewer will ask and that no amount of
measurement on 42 windows can answer:

  ADOPTION      What happens when speakers know they are being read? This is the
                first question anyone asks and the one the field answers with
                rhetoric. Here it is a curve.

  SURVIVAL      Which features carry the signal once coaching is applied? The moat
                argument stands or falls on this decomposition.

  DOSE-RESPONSE The model's sharp prediction: divergence should scale with how
                binding the constraint is, and bindingness rises as the disclosure
                date approaches. This is what the Fed corpus will falsify.

  DESIGN        How many speakers and how deep a baseline before the question is
                answerable. Turns "we are underpowered" into a procurement number.

EVERY NUMBER IN THIS MODULE IS MODEL-CONDITIONAL. It describes a world we specified,
not the one we are selling into. The value is in the shape of the curves and in the
sample sizes, not in any single figure.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from . import anomaly, confounds, features as F, validate, world as W
from .estimator import fit_baselines
from .experiments import APRIORI

MATERIAL_PCT = 6.0


# ---------------------------------------------------------------------------
# One scored run
# ---------------------------------------------------------------------------

def _prep(rows: list[dict]) -> list[dict]:
    F.score_corpus(rows)
    lms, _, _ = F.fit_language_models(rows)
    F.attach_surprisal(rows, lms)
    F.add_time_index(rows)
    return rows


def score_run(rows: list[dict]) -> dict:
    """
    Score a simulated corpus and return everything the sweeps need.

    Returns per-constrained-window: signed evasion projection, the marginal z on
    every feature, the latent truth, and the outcome. Scoring against `theta_held`
    — the information actually withheld — rather than against the price move is
    deliberate: the move is a noisy downstream proxy, and mixing its noise into a
    measurement question would hide the thing being measured.
    """
    feats = F.FEATURES
    Y = np.array([[r["scores"][f] for f in feats] for r in rows])
    X, _ = confounds.build_design(rows, ["x"])
    Yr = confounds.residualise(X, Y, confounds.fit_residualiser(X, Y))
    w = np.array([APRIORI[f] for f in feats])

    by_sp = {}
    for sp in sorted({r["speaker"] for r in rows}):
        M = np.array([Yr[i] for i, r in enumerate(rows)
                      if r["speaker"] == sp and r["window"] == "baseline"])
        if len(M) >= 2:
            by_sp[sp] = M
    bl = fit_baselines(by_sp, feats, min_n=2)

    ev, Z, held, mv, grp, days = [], [], [], [], [], []
    for i, r in enumerate(rows):
        if r["window"] != "constrained":
            continue
        b = bl.get(r["speaker"])
        if b is None:
            continue
        sd = np.sqrt(np.diag(b.cov))
        ev.append(anomaly.directional_component(Yr[i], b, w))
        Z.append((Yr[i] - b.mean) / np.maximum(sd, 1e-6))
        held.append(r.get("theta_held", 0.0))
        mv.append(r.get("outcome_move_pct"))
        grp.append(r["speaker"])
        days.append(r.get("days_to_disclosure"))

    return {"evasion": np.array(ev), "Z": np.array(Z), "held": np.array(held),
            "move": np.array([m if m is not None else np.nan for m in mv], float),
            "groups": np.array(grp), "days": np.array(days, float),
            "features": feats}


def _rho(a, b, min_n: int = 6) -> float:
    """Spearman with an explicit minimum n — within-speaker calls legitimately have 3."""
    if len(a) < min_n or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(stats.spearmanr(a, b).statistic)


# ---------------------------------------------------------------------------
# 1. Adoption
# ---------------------------------------------------------------------------

def adoption_sweep(levels=(0.0, 0.25, 0.5, 0.75, 1.0), n_reps: int = 10,
                   base: W.GameParams | None = None, seed: int = 700) -> list[dict]:
    """
    How much of the signal survives as speakers adapt.

    Reports two things that decay differently and are sold to different customers:

      rho_held   can we still MEASURE what is being withheld
      auc_move   can we still PREDICT the market reaction

    Also reports mean register, which is the behavioural prediction: as adoption
    rises, speakers abandon hedged evasion for flat refusal. A falling register with
    a holding rho_held is the product working rather than failing — the information
    is still legible, it is simply being declined rather than talked around.
    """
    base = base or W.GameParams()
    out = []
    for a in levels:
        rhos, aucs, regs, held_mean = [], [], [], []
        for rep in range(n_reps):
            p = W.GameParams(**{**base.as_dict(), "adoption": float(a)})
            rows = _prep(W.simulate(p, seed + rep * 31))
            s = score_run(rows)
            if len(s["evasion"]) < 8:
                continue
            rhos.append(_rho(s["evasion"], s["held"]))
            y = (np.abs(s["move"]) >= MATERIAL_PCT).astype(int)
            if len(np.unique(y)) > 1:
                aucs.append(validate.auc(s["evasion"], y))
            # Register is only meaningful for speakers who actually hold something —
            # a speaker with nothing to withhold has r* = 0 by construction, and
            # averaging them in dilutes the behavioural signal we are trying to read.
            holding = [r.get("register", 0.0) for r in rows
                       if r["window"] == "constrained" and r.get("theta_held", 0) > 1e-6]
            regs.append(float(np.mean(holding)) if holding else float("nan"))
            held_mean.append(float(np.mean(s["held"])))
        if not rhos:
            continue
        out.append({
            "adoption": float(a),
            "rho_held": round(float(np.nanmean(rhos)), 4),
            "rho_held_sd": round(float(np.nanstd(rhos)), 4),
            "auc_move": round(float(np.mean(aucs)), 4) if aucs else None,
            "mean_register": round(float(np.nanmean(regs)), 4),
            "mean_theta_held": round(float(np.mean(held_mean)), 4),
            "n_reps": len(rhos),
        })
    return out


# ---------------------------------------------------------------------------
# 2. Feature survival
# ---------------------------------------------------------------------------

def feature_survival(n_reps: int = 10, base: W.GameParams | None = None,
                     seed: int = 810) -> list[dict]:
    """
    Per-feature correlation with the withheld information, at zero adoption and at
    full adoption.

    The prediction from COACHABILITY was that hedging and confidence language would
    collapse, deflection and distancing degrade, and specificity avoidance and
    surprisal hold — because the last two are not about word choice.

    WHAT ACTUALLY HAPPENS IS MORE INTERESTING AND WE SHOULD NOT SAND IT DOWN.
    The lexical evasion features do degrade as predicted. But surprisal does not
    merely hold, it RISES — often above 1.0 retention. The reason is mechanical: an
    adapted speaker abandons hedged evasion for flat refusal, and refusal is its own
    distinctive register. It is not what this speaker sounds like on an earnings
    call either. So the signature does not disappear under coaching, it MIGRATES —
    out of the lexicon and into the language model.

    Two honest caveats, both load-bearing:

      * This is partly built in. We put refusals in their own text bank, so of
        course they read as novel vocabulary. The model demonstrates that the
        mechanism is coherent, NOT that it is true of real speakers.
      * A retention ratio above 1.0 is not "better detection". It means the
        detector is now keying on a different thing, which has to be re-validated
        against outcomes before anyone leans on it.

    The claim this licenses is narrow and defensible: to escape divergence-from-self
    you cannot simply change your words, because the baseline is what you are being
    compared to. Changing your words IS the divergence.
    """
    base = base or W.GameParams()
    feats = F.FEATURES
    acc = {f: {"naive": [], "adapted": []} for f in feats}

    for rep in range(n_reps):
        for tag, a in (("naive", 0.0), ("adapted", 1.0)):
            p = W.GameParams(**{**base.as_dict(), "adoption": a})
            s = score_run(_prep(W.simulate(p, seed + rep * 47)))
            if len(s["held"]) < 8:
                continue
            for k, f in enumerate(feats):
                acc[f][tag].append(abs(_rho(s["Z"][:, k], s["held"])))

    out = []
    for f in feats:
        n_ = float(np.nanmean(acc[f]["naive"])) if acc[f]["naive"] else float("nan")
        a_ = float(np.nanmean(acc[f]["adapted"])) if acc[f]["adapted"] else float("nan")
        ret = float(a_ / n_) if n_ and n_ > 1e-6 else None
        out.append({
            "feature": f,
            "coachability": W.COACHABILITY[f],
            "rho_naive": round(n_, 4),
            "rho_adapted": round(a_, 4),
            "retained": round(ret, 4) if ret is not None else None,
            "verdict": (None if ret is None else
                        "gains — keys on the refusal register" if ret > 1.15 else
                        "holds" if ret >= 0.80 else
                        "degrades" if ret >= 0.55 else "collapses"),
        })
    return sorted(out, key=lambda d: -(d["retained"] or 0))


# ---------------------------------------------------------------------------
# 3. Dose-response — the falsifiable prediction
# ---------------------------------------------------------------------------

def dose_response_predicted(n_reps: int = 8, bins=(0, 5, 10, 15, 20, 25, 30),
                            base: W.GameParams | None = None,
                            seed: int = 920) -> list[dict]:
    """What the model says the T-to-disclosure gradient should look like."""
    base = base or W.GameParams()
    acc = {i: [] for i in range(len(bins) - 1)}
    for rep in range(n_reps):
        s = score_run(_prep(W.simulate(base, seed + rep * 53)))
        for e, d in zip(s["evasion"], s["days"]):
            for i in range(len(bins) - 1):
                if bins[i] <= d < bins[i + 1]:
                    acc[i].append(float(e))
                    break
    return [{"days_lo": bins[i], "days_hi": bins[i + 1],
             "mean_divergence": round(float(np.mean(v)), 4) if v else None,
             "n": len(v)}
            for i, v in acc.items()]


def dose_response_measured(rows: list[dict], evasion_by_id: dict[str, float],
                           n_perm: int = 20000) -> dict:
    """
    The same gradient, measured on a real (or corpus) run, with a blocked
    permutation test.

    WITHIN-SPEAKER by construction: the permutation shuffles days-to-disclosure
    inside each speaker, so the test asks whether a speaker diverges MORE as their
    own disclosure approaches — not whether leaky speakers happen to be scheduled
    differently. Requires no outcome data at all, which is why it is the cheapest
    strong test available to us and why it should run first on the Fed corpus.
    """
    ev, days, grp = [], [], []
    for r in rows:
        if r["window"] != "constrained":
            continue
        d = r.get("days_to_disclosure")
        e = evasion_by_id.get(r["id"])
        if d is None or e is None:
            continue
        ev.append(float(e)); days.append(float(d)); grp.append(r["speaker"])
    ev, days, grp = np.array(ev), np.array(days), np.array(grp)

    if len(ev) < 10 or np.std(days) < 1e-9:
        return {"status": "insufficient_variation", "n": int(len(ev))}

    def _within_median(e, d, g) -> float:
        """Median of the per-speaker rho(divergence, -days). The right statistic."""
        vals = []
        for sp in np.unique(g):
            m = g == sp
            if m.sum() >= 3 and np.std(d[m]) > 1e-9 and np.std(e[m]) > 1e-9:
                r = _rho(e[m], -d[m], min_n=3)
                if not np.isnan(r):
                    vals.append(r)
        return float(np.median(vals)) if vals else float("nan")

    # The pooled correlation is the WRONG statistic here and we report it only to show
    # why. Between-speaker variation in how much anyone is holding swamps the
    # within-speaker movement of the constraint over time — in simulation the pooled
    # rho is ~0.12 while the median within-speaker rho is ~0.50 on the same data. A
    # pooled test would call a real gradient a null. The headline is the within-speaker
    # statistic, tested against a null that shuffles days INSIDE each speaker, so the
    # only thing destroyed is the timing relationship and each speaker keeps their own
    # level and spread.
    pooled = validate.permutation_test(ev, -days, groups=grp, n_perm=min(n_perm, 20000))

    obs = _within_median(ev, days, grp)
    rng = np.random.default_rng(11)
    null = []
    n_draw = min(n_perm, 4000)
    for _ in range(n_draw):
        d_sh = days.copy()
        for sp in np.unique(grp):
            m = grp == sp
            d_sh[m] = rng.permutation(d_sh[m])
        v = _within_median(ev, d_sh, grp)
        if not np.isnan(v):
            null.append(v)
    null = np.array(null)
    p_within = float((np.sum(null >= obs) + 1) / (len(null) + 1)) if len(null) else float("nan")

    per_speaker = []
    for sp in np.unique(grp):
        m = grp == sp
        if m.sum() >= 3 and np.std(days[m]) > 1e-9 and np.std(ev[m]) > 1e-9:
            r = _rho(ev[m], -days[m], min_n=3)
            if not np.isnan(r):
                per_speaker.append(r)

    return {
        "status": "ok",
        "n": int(len(ev)), "n_speakers": int(len(np.unique(grp))),
        "rho_pooled": round(float(pooled["observed"]), 4),
        "p_pooled": pooled["p_value"],
        "median_within_speaker_rho": round(obs, 4) if not np.isnan(obs) else None,
        "p_within": round(p_within, 5) if not np.isnan(p_within) else None,
        "n_speakers_with_gradient": len(per_speaker),
        "frac_positive": round(float(np.mean(np.array(per_speaker) > 0)), 3) if per_speaker else None,
        "n_perm": int(len(null)),
        "blocked": True,
        "significant_at_05": bool(not np.isnan(p_within) and p_within < .05),
        "headline_statistic": "median within-speaker rho(divergence, -days_to_disclosure)",
        "direction": "divergence rises as disclosure approaches" if obs > 0
                     else "divergence falls as disclosure approaches",
    }


# ---------------------------------------------------------------------------
# 3b. Archetypes — who gets flagged, and who gets flagged wrongly
# ---------------------------------------------------------------------------

def archetype_rates(population: dict[str, float] | None = None,
                    flag_threshold: float = 1.5, n_reps: int = 10,
                    base: W.GameParams | None = None, seed: int = 1300) -> dict:
    """
    Flag rate per speaker archetype, scored two ways.

    THIS IS THE FALSE-POSITIVE TEST and it is the one an analyst should look at
    first. `genuinely_uncertain` speakers withhold nothing — theta is identically
    zero — but they are habitual hedgers who rarely commit. An absolute scorer, which
    is what every transcript-sentiment vendor ships, must flag them: they look evasive
    against any fixed standard. A baseline-relative scorer should not, because their
    hedging is their own norm and we measure divergence from that norm.

    The gap between `flag_rate_relative` and `flag_rate_absolute` on that one row is
    the entire product argument, stated as a number an analyst can check rather than
    a claim they have to accept. If the two are equal, per-speaker baselining is
    buying nothing and we should say so.

    Also reports the true-positive side, so nobody reads a low false-positive rate as
    success when the method is simply flagging no one.
    """
    base = base or W.GameParams()
    feats = F.FEATURES
    w_ap = np.array([APRIORI[f] for f in feats])
    acc: dict[str, dict[str, list]] = {}

    for rep in range(n_reps):
        rows = _prep(W.simulate(base, seed + rep * 41, population=population))

        # --- baseline-relative scoring (what we ship) ---
        s = score_run(rows)
        cons = [r for r in rows if r["window"] == "constrained"]
        # score_run drops windows whose speaker lacks a baseline; realign by speaker
        keep = []
        seen: dict[str, int] = {}
        for r in cons:
            seen[r["speaker"]] = seen.get(r["speaker"], 0)
        idx = 0
        for r in cons:
            if idx < len(s["groups"]) and s["groups"][idx] == r["speaker"]:
                keep.append((r, float(s["evasion"][idx]))); idx += 1

        # --- absolute scoring (what a transcript vendor does) ---
        Y = np.array([[r["scores"][f] for f in feats] for r in rows])
        X, _ = confounds.build_design(rows, ["x"])
        Yr = confounds.residualise(X, Y, confounds.fit_residualiser(X, Y))
        wn = w_ap / np.linalg.norm(w_ap)
        abs_all = Yr @ wn
        # Standardise absolute scores across the corpus so the same nominal threshold
        # means "unusual relative to the population" — the fairest possible version of
        # the competing method, rather than a straw man.
        abs_z = (abs_all - abs_all.mean()) / max(abs_all.std(), 1e-9)
        abs_by_id = {r["id"]: float(abs_z[i]) for i, r in enumerate(rows)}

        for r, ev in keep:
            a = acc.setdefault(r["archetype"], {"rel": [], "abs": [], "held": []})
            a["rel"].append(ev)
            a["abs"].append(abs_by_id[r["id"]])
            a["held"].append(float(r.get("theta_held", 0.0)))

    out = []
    for name, a in acc.items():
        arch = W.ARCHETYPES[name]
        rel, ab, held = np.array(a["rel"]), np.array(a["abs"]), np.array(a["held"])
        truly_holding = held > 1e-9
        out.append({
            "archetype": name,
            "label": arch.label,
            "description": arch.description,
            "n_windows": int(len(rel)),
            "withholds": bool(truly_holding.any()),
            "frac_withholding": round(float(truly_holding.mean()), 3),
            "mean_divergence": round(float(rel.mean()), 4),
            "flag_rate_relative": round(float((rel >= flag_threshold).mean()), 4),
            "flag_rate_absolute": round(float((ab >= flag_threshold).mean()), 4),
            # For a type that withholds nothing, every flag is a false positive.
            "false_positive_relative": None if truly_holding.any()
                else round(float((rel >= flag_threshold).mean()), 4),
            "false_positive_absolute": None if truly_holding.any()
                else round(float((ab >= flag_threshold).mean()), 4),
        })
    out.sort(key=lambda d: -d["flag_rate_relative"])

    honest = [d for d in out if not d["withholds"]]
    fp_rel = float(np.mean([d["false_positive_relative"] for d in honest])) if honest else float("nan")
    fp_abs = float(np.mean([d["false_positive_absolute"] for d in honest])) if honest else float("nan")
    holders = [d for d in out if d["withholds"]]
    tp_rel = float(np.mean([d["flag_rate_relative"] for d in holders])) if holders else float("nan")

    # ---- matched alarm budget: the only fair comparison ----
    #
    # Comparing two detectors at the same NOMINAL threshold is not a comparison, because
    # the two scores are in different units — one is in a speaker's own SDs, the other in
    # population SDs — so they raise different numbers of alarms and whichever is more
    # conservative wins on false positives for free. The honest version fixes the thing an
    # analyst actually has a fixed supply of: attention. Set each scorer's threshold so
    # both flag the SAME number of windows, then ask which flags the wrong ones.
    all_rel = np.concatenate([np.array(a["rel"]) for a in acc.values()])
    all_abs = np.concatenate([np.array(a["abs"]) for a in acc.values()])
    all_held = np.concatenate([np.array(a["held"]) for a in acc.values()])
    budget = float((all_rel >= flag_threshold).mean())          # alarms we can afford
    q = 1.0 - budget
    t_rel = float(np.quantile(all_rel, q)) if 0 < q < 1 else flag_threshold
    t_abs = float(np.quantile(all_abs, q)) if 0 < q < 1 else flag_threshold
    dishonest = all_held > 1e-9
    m_fp_rel = float((all_rel[~dishonest] >= t_rel).mean()) if (~dishonest).any() else float("nan")
    m_fp_abs = float((all_abs[~dishonest] >= t_abs).mean()) if (~dishonest).any() else float("nan")
    m_tp_rel = float((all_rel[dishonest] >= t_rel).mean()) if dishonest.any() else float("nan")
    m_tp_abs = float((all_abs[dishonest] >= t_abs).mean()) if dishonest.any() else float("nan")

    return {
        "flag_threshold": flag_threshold,
        "population": W.resolve_population(population),
        "archetypes": out,
        "summary": {
            "false_positive_rate_relative": round(fp_rel, 4),
            "false_positive_rate_absolute": round(fp_abs, 4),
            "true_positive_rate_relative": round(tp_rel, 4),
            "advantage": round(fp_abs - fp_rel, 4),
            "reading": ("Baseline-relative scoring flags honest-but-hedging speakers "
                        f"{round(fp_rel * 100, 1)}% of the time versus "
                        f"{round(fp_abs * 100, 1)}% for absolute scoring, at the same "
                        "nominal threshold."),
        },
        "matched_budget": {
            "alarm_budget": round(budget, 4),
            "threshold_relative": round(t_rel, 3),
            "threshold_absolute": round(t_abs, 3),
            "false_positive_relative": round(m_fp_rel, 4),
            "false_positive_absolute": round(m_fp_abs, 4),
            "true_positive_relative": round(m_tp_rel, 4),
            "true_positive_absolute": round(m_tp_abs, 4),
            "n_windows": int(len(all_rel)),
            "reading": ("Both scorers tuned to raise the same number of alarms. "
                        "Compare false-positive rates: this is the comparison that "
                        "survives a hostile reviewer."),
        },
    }


# ---------------------------------------------------------------------------
# 3c. Does the constraint actually bind? (no linguistics required)
# ---------------------------------------------------------------------------

def speech_suppression(rows: list[dict], blackout_share_of_calendar: float) -> dict:
    """
    Compare the share of SPEECHES that fall in the constrained window against the share
    of the CALENDAR the window covers.

    This is the cheapest real finding available and it needs no linguistic analysis at
    all. If the constraint is genuine, people subject to it should speak less while it
    is in force. On the Fed corpus the blackout covers ~28% of the calendar but contains
    ~5% of speeches — a suppression ratio above 5x.

    Two things follow, and they point in opposite directions:

      * It is direct evidence the treatment is real and binding, established before a
        single word is scored. Worth leading with, because it is unarguable.
      * It is also why the treatment group is small. A constraint that stops people
        speaking necessarily produces few constrained observations, which is a
        structural limit on this corpus rather than a fixable oversight. The answer is
        more years, not a cleverer estimator.
    """
    cons = [r for r in rows if r.get("window") == "constrained"]
    n = len(rows)
    if not n or not 0 < blackout_share_of_calendar < 1:
        return {"status": "insufficient_data"}
    observed = len(cons) / n
    expected = blackout_share_of_calendar
    return {
        "status": "ok",
        "n_statements": n,
        "n_constrained": len(cons),
        "share_of_speeches": round(observed, 4),
        "share_of_calendar": round(expected, 4),
        "suppression_ratio": round(expected / observed, 2) if observed > 0 else None,
        "reading": (f"{observed:.1%} of speeches fall in a window covering "
                    f"{expected:.1%} of the calendar — speech is suppressed roughly "
                    f"{expected / observed:.1f}x while the constraint is in force."
                    if observed > 0 else "No constrained windows."),
    }


# ---------------------------------------------------------------------------
# 4. Design — how much data do we actually need
# ---------------------------------------------------------------------------

def design_grid(speaker_counts=(8, 14, 22, 34), baseline_depths=(3, 6, 10),
                n_reps: int = 6, base: W.GameParams | None = None,
                seed: int = 1040) -> list[dict]:
    """
    Power over the two dimensions we can actually buy: more speakers, or more
    history per speaker.

    Worth knowing which one to spend on. Baseline depth is usually cheaper — it is
    archival — while speakers are a licensing and coverage problem. If the grid says
    depth dominates, that changes the data-acquisition plan.
    """
    base = base or W.GameParams()
    out = []
    rng = np.random.default_rng(seed)
    for k in speaker_counts:
        for b in baseline_depths:
            rhos, ps = [], []
            for rep in range(n_reps):
                p = W.GameParams(**{**base.as_dict(), "n_speakers": int(k),
                                    "baseline_n": int(b)})
                s = score_run(_prep(W.simulate(p, seed + rep * 29)))
                if len(s["evasion"]) < 8:
                    continue

                # AT THE SPEAKER LEVEL, and this correction matters. An earlier
                # version ran a within-speaker blocked permutation against
                # theta_held — but theta is a property OF THE SPEAKER, nearly
                # constant across their windows, so shuffling inside a speaker
                # shuffled a constant. The null was degenerate and power came back
                # flat at every n, which looked like the method failing to scale
                # when it was the test being inapplicable. Blocking is right when
                # the quantity varies within the block and wrong when it does not.
                sp = np.unique(s["groups"])
                x = np.array([s["evasion"][s["groups"] == u].mean() for u in sp])
                t = np.array([s["held"][s["groups"] == u].mean() for u in sp])
                if len(sp) < 5 or np.std(t) < 1e-9 or np.std(x) < 1e-9:
                    continue
                obs = _rho(x, t, min_n=5)
                null = np.array([_rho(x, rng.permutation(t), min_n=5)
                                 for _ in range(1500)])
                null = null[~np.isnan(null)]
                rhos.append(obs)
                ps.append(float((np.sum(null >= obs) + 1) / (len(null) + 1)))
            if not rhos:
                continue
            out.append({
                "n_speakers": int(k), "baseline_n": int(b),
                "rho_held": round(float(np.nanmean(rhos)), 4),
                "median_p": round(float(np.median(ps)), 4),
                "power_at_05": round(float(np.mean(np.array(ps) < .05)), 3),
                "n_reps": len(rhos),
            })
    return out
