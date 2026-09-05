"""
Tests for the statistical layer.

These assert *properties* rather than fixed numbers — a test that pins an AUC to
three decimals breaks on every legitimate change and teaches you nothing. What
these check is that the estimators behave the way the theory says they must.

Run: python tests/test_statistics.py
"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "src"))

from theroux import anomaly, confounds, validate  # noqa: E402
from theroux.estimator import empirical_bayes_weight, fit_baselines  # noqa: E402
from theroux.surprisal import SpeakerLanguageModel, build_background  # noqa: E402

RNG = np.random.default_rng(11)
FEATS = ["a", "b", "c"]


# ---------------------------------------------------------------- shrinkage

def test_ledoit_wolf_covariance_is_invertible_at_tiny_n():
    """The whole point of shrinkage: n < p must still yield a usable precision."""
    for n in (2, 3, 4):
        X = RNG.normal(size=(n, 6))
        b = fit_baselines({"s": X}, [f"f{i}" for i in range(6)])["s"]
        assert np.all(np.isfinite(b.precision))
        assert np.linalg.cond(b.cov) < 1e12, "covariance is numerically singular"


def test_empirical_bayes_shrinks_more_when_n_is_small():
    within, between = np.array([1.0, 1.0]), np.array([1.0, 1.0])
    w_small = empirical_bayes_weight(2, within, between)
    w_large = empirical_bayes_weight(50, within, between)
    assert w_small < w_large, "thin baselines must borrow more strength"
    assert 0 <= w_small <= 1 and 0 <= w_large <= 1


def test_thin_baseline_is_pulled_toward_population():
    """A speaker with 2 observations should end up closer to the pack than their raw mean."""
    pop = {f"s{i}": RNG.normal(0, 1, size=(9, 3)) for i in range(7)}
    pop["thin"] = RNG.normal(6.0, 1, size=(2, 3))      # far-out speaker, tiny n
    b = fit_baselines(pop, FEATS)["thin"]
    assert np.linalg.norm(b.mean) < np.linalg.norm(b.raw_mean)
    assert b.mean_shrinkage > 0


# ---------------------------------------------------------------- anomaly

def test_mahalanobis_accounts_for_correlation():
    """
    Two features correlated at 0.9. A deviation ALONG that correlation is ordinary;
    the same-sized deviation ACROSS it is not. Independent z-scores cannot tell
    these apart — Mahalanobis must.
    """
    n = 400
    z = RNG.normal(size=n)
    X = np.column_stack([z + RNG.normal(0, .25, n),
                         z + RNG.normal(0, .25, n),
                         RNG.normal(0, 1, n)])
    b = fit_baselines({"s": X}, FEATS)["s"]
    along = b.mean + np.array([1.5, 1.5, 0.0])     # with the correlation
    across = b.mean + np.array([1.5, -1.5, 0.0])   # against it
    assert anomaly.mahalanobis(across, b) > anomaly.mahalanobis(along, b) * 2


def test_chi2_percentile_is_calibrated_under_the_null():
    """Draws from the fitted distribution should be uniform in p — roughly."""
    X = RNG.normal(size=(600, 3))
    b = fit_baselines({"s": X}, FEATS)["s"]
    ps = np.array([anomaly.score(RNG.normal(size=3), b)["p_chi2"] for _ in range(600)])
    assert 0.02 < (ps < 0.05).mean() < 0.15, "chi-square tail is badly miscalibrated"


def test_attribution_sums_to_one():
    X = RNG.normal(size=(30, 3))
    b = fit_baselines({"s": X}, FEATS)["s"]
    a = anomaly.score(b.mean + np.array([2.0, .3, -1.0]), b)
    assert abs(sum(a["attribution"].values()) - 1.0) < 1e-6


def test_evasion_score_is_signed():
    """Unsigned distance cannot separate evasive from unusually candid. This must."""
    feats = ["hedging", "specificity_avoidance", "pronoun_distancing",
             "topic_deflection", "confidence_language", "surprisal_z"]
    X = RNG.normal(size=(40, 6))
    b = fit_baselines({"s": X}, feats)["s"]
    more = b.mean + np.ones(6) * 1.2
    less = b.mean - np.ones(6) * 1.2
    assert anomaly.evasion_score(more, b) > 0 > anomaly.evasion_score(less, b)
    assert anomaly.mahalanobis(more, b) == float(
        f"{anomaly.mahalanobis(less, b):.10f}"
    ) or abs(anomaly.mahalanobis(more, b) - anomaly.mahalanobis(less, b)) < 1e-6


# ---------------------------------------------------------------- validation

def test_permutation_test_finds_nothing_in_noise():
    x, y = RNG.normal(size=90), RNG.normal(size=90)
    r = validate.permutation_test(x, y, n_perm=3000)
    assert r["p_value"] > 0.05, "false positive on pure noise"


def test_permutation_test_finds_a_real_effect():
    x = RNG.normal(size=90)
    y = x * 1.4 + RNG.normal(0, .55, 90)
    r = validate.permutation_test(x, y, n_perm=3000)
    assert r["p_value"] < 0.01 and r["significant_at_05"]


def test_blocked_permutation_is_more_conservative_on_clustered_data():
    """
    Clustered data with a between-group effect and no within-group effect.
    Naive shuffling sees significance; block shuffling should not.
    """
    g = np.repeat(np.arange(12), 8)
    offs = RNG.normal(0, 2.2, 12)
    x = offs[g] + RNG.normal(0, .25, 96)
    y = offs[g] + RNG.normal(0, .25, 96)
    naive = validate.permutation_test(x, y, n_perm=3000)
    blocked = validate.permutation_test(x, y, groups=g, n_perm=3000)
    assert blocked["p_value"] > naive["p_value"]


def test_auc_matches_known_values():
    assert validate.auc(np.array([1., 2, 3, 4]), np.array([0, 0, 1, 1])) == 1.0
    assert validate.auc(np.array([4., 3, 2, 1]), np.array([0, 0, 1, 1])) == 0.0
    assert abs(validate.auc(np.array([1., 2, 3, 4]), np.array([0, 1, 0, 1])) - 0.75) < 1e-9


def test_calibration_reports_perfect_and_terrible_correctly():
    labels = np.array([0] * 50 + [1] * 50)
    good = validate.calibration(np.array([0.05] * 50 + [0.95] * 50), labels)
    bad = validate.calibration(np.array([0.95] * 50 + [0.05] * 50), labels)
    assert good["brier"] < 0.02 and good["ece"] < 0.1
    assert bad["brier"] > 0.8


# ---------------------------------------------------------------- surprisal

def test_surprisal_rises_for_out_of_character_text():
    base = ["we grew revenue twenty percent and margin improved two hundred basis points"] * 4
    bg, tot = build_background(base + ["completely different vocabulary about maritime shipping"])
    lm = SpeakerLanguageModel("s", base, bg, tot)
    normal = lm.cross_entropy("we grew revenue twenty percent and margin improved again")
    odd = lm.cross_entropy("maritime shipping vessels harbour tonnage freight cargo logistics")
    assert odd > normal, "vocabulary shift must raise cross-entropy"


def test_surprisal_needs_no_word_lists():
    """The point of this feature: it works on text containing zero lexicon terms."""
    import theroux.lexicon as LX
    base = ["alpha beta gamma delta epsilon zeta eta theta iota kappa"] * 4
    bg, tot = build_background(base)
    lm = SpeakerLanguageModel("s", base, bg, tot)
    text = "lambda mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega"
    assert not any(h in text for h in LX.HEDGES)
    assert lm.cross_entropy(text) > lm.cross_entropy(base[0])


# ---------------------------------------------------------------- confounds

def test_residualisation_removes_a_planted_length_effect():
    n = 220
    words = RNG.integers(120, 2400, n)
    rows = [{"meta": {"words": int(w)}, "event_type": "x", "_t": float(i)}
            for i, w in enumerate(words)]
    signal = RNG.normal(size=n)
    Y = np.column_stack([signal + 2.2 * np.log(words), signal])
    X, _ = confounds.build_design(rows, ["x"])
    beta = confounds.fit_residualiser(X, Y)
    ve = confounds.variance_explained(Y, confounds.residualise(X, Y, beta))
    assert ve[0] > 0.6, "planted length confound was not absorbed"
    assert ve[1] < 0.25, "clean feature was over-corrected"


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  pass  {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
