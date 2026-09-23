import subprocess
import sys
import warnings

import numpy as np
import pytest

from spectral_connectivity.statistics import (
    Benjamini_Hochberg_procedure,
    Bonferroni_correction,
    adjust_for_multiple_comparisons,
    coherence_bias,
    coherence_fisher_z_transform,
    coherence_rate_adjustment,
    coherence_significance_pvalue,
    get_normal_distribution_p_values,
    jackknife_confidence_interval,
    power_bias,
    power_confidence_intervals,
    power_fisher_z_transform,
    power_variance,
)


@pytest.mark.parametrize("correction", [Benjamini_Hochberg_procedure, Bonferroni_correction])
@pytest.mark.parametrize("bad_alpha", [0, 1, -0.1, np.nan, np.inf, True, "0.05"])
def test_multiple_comparison_corrections_validate_alpha(correction, bad_alpha):
    with pytest.raises(ValueError, match="alpha must be a finite number"):
        correction(np.array([0.01, 0.2]), alpha=bad_alpha)


def test_bonferroni_handles_empty_and_nonfinite_families():
    assert Bonferroni_correction(np.array([])).shape == (0,)
    result = Bonferroni_correction(np.array([0.01, np.nan, np.inf]), alpha=0.05)
    np.testing.assert_array_equal(result, [True, False, False])


def test_bonferroni_warns_when_whole_family_undefined():
    """A fully non-finite Bonferroni family warns, matching Benjamini-Hochberg.

    Both corrections exclude undefined tests; the all-undefined case must be
    equally loud, or "nothing significant" hides that nothing was testable.
    """
    with pytest.warns(UserWarning, match="every p-value is non-finite"):
        result = Bonferroni_correction(np.array([np.nan, np.inf]), alpha=0.05)
    assert not result.any()
    # A finite value in the family must NOT warn; an empty family must NOT warn.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Bonferroni_correction(np.array([0.01, np.nan]), alpha=0.05)
        Bonferroni_correction(np.array([]), alpha=0.05)


def test_adjust_for_multiple_comparisons_rejects_unknown_method():
    with pytest.raises(ValueError, match="Unknown multiple-comparisons method"):
        adjust_for_multiple_comparisons(np.array([0.1]), method="not-a-method")


def test_get_normal_distribution_p_values():
    # approximate 97.5 percentile of the standard normal distribution
    zscore = 1.95996
    assert np.allclose(get_normal_distribution_p_values(zscore), 0.025)


def test_fisher_z_transform_two_sample_matches_analytic():
    """Two-sample statistic for unequal coherences and unequal sample sizes.

    z = ((atanh|C1| - b1) - (atanh|C2| - b2)) / sqrt(b1 + b2), with the
    documented bias b = 1 / (2 * n_obs - 2) (Bokil et al. 2007).
    """
    coherency1 = np.array([0.6 * np.exp(0.3j), 0.2 * np.exp(-2.0j)])
    coherency2 = np.array([0.3 * np.exp(-1.0j), 0.7 * np.exp(1.5j)])
    n_obs1, n_obs2 = 20, 50
    bias1 = 1 / (2 * n_obs1 - 2)
    bias2 = 1 / (2 * n_obs2 - 2)
    expected = (
        (np.arctanh(np.abs(coherency1)) - bias1) - (np.arctanh(np.abs(coherency2)) - bias2)
    ) / np.sqrt(bias1 + bias2)

    z = coherence_fisher_z_transform(coherency1, n_obs1, coherency2=coherency2, n_obs2=n_obs2)
    np.testing.assert_allclose(z, expected, rtol=1e-12)
    # Swapping the samples negates the statistic.
    swapped = coherence_fisher_z_transform(
        coherency2, n_obs2, coherency2=coherency1, n_obs2=n_obs1
    )
    np.testing.assert_allclose(swapped, -expected, rtol=1e-12)
    # Identical samples give zero difference.
    np.testing.assert_allclose(
        coherence_fisher_z_transform(coherency1, n_obs1, coherency2=coherency1, n_obs2=n_obs1),
        0.0,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    ("p_values", "expected_is_significant"),
    [
        (np.ones((10, 2)), np.zeros((10, 2), dtype=bool)),
        (np.zeros((10, 2)), np.ones((10, 2), dtype=bool)),
        (np.array([0.03, 0.01, 0.04, 0.06]), np.array([False, True, False, False])),
        (np.array([0.03, 0.01, 0.04, 0.05]), np.ones((4,))),
    ],
)
def test_Benjamini_Hochberg_procedure(p_values, expected_is_significant):
    alpha = 0.05
    assert np.allclose(Benjamini_Hochberg_procedure(p_values, alpha), expected_is_significant)


def test_Benjamini_Hochberg_excludes_nan_from_family():
    """Undefined (NaN) tests must not count toward the FDR family.

    A NaN p-value marks an undefined test (e.g. a coherence pair with a
    dead/zero-power channel). It must be returned as not-significant and must
    not tighten the threshold for the valid tests: the decision for the finite
    p-values is identical whether or not NaN padding is present.
    """
    alpha = 0.05
    valid = np.array([0.001, 0.02, 0.04, 0.3])
    padded = np.array([0.001, 0.02, 0.04, 0.3, np.nan, np.nan])

    result_valid = Benjamini_Hochberg_procedure(valid, alpha)
    result_padded = Benjamini_Hochberg_procedure(padded, alpha)

    # NaN entries are never significant.
    assert not result_padded[4:].any()
    # The finite entries are unaffected by the presence of NaN.
    assert np.array_equal(result_padded[:4], result_valid)
    # All-NaN input yields all-False (no valid tests), same input shape, and
    # warns because the whole family is undefined (see the dedicated test below).
    with pytest.warns(UserWarning, match="every p-value is non-finite"):
        all_nan = Benjamini_Hochberg_procedure(np.full((2, 3), np.nan), alpha)
    assert all_nan.shape == (2, 3)
    assert not all_nan.any()


def test_Benjamini_Hochberg_warns_when_whole_family_undefined():
    """An all-non-finite family returns all-False but must warn, not fail silently.

    Otherwise "nothing significant" is indistinguishable from a valid family
    with no true effects, when in fact every test was undefined (e.g. every
    tested pair involves a dead/zero-power channel).
    """
    with pytest.warns(UserWarning, match="every p-value is non-finite"):
        result = Benjamini_Hochberg_procedure(np.array([np.nan, np.inf]), alpha=0.05)
    assert not result.any()
    # A family with at least one finite p-value must NOT warn.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Benjamini_Hochberg_procedure(np.array([0.01, np.nan]), alpha=0.05)
    # An empty family is not "undefined"; it must not warn either.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        empty = Benjamini_Hochberg_procedure(np.array([]), alpha=0.05)
    assert empty.shape == (0,)


@pytest.mark.parametrize(
    ("p_values", "n_outside"),
    [([0.1, 1.5, -0.2, 0.3], 2), ([0.1, 1.5, 0.2], 1)],
    ids=["above_and_below", "above_only"],
)
def test_Benjamini_Hochberg_out_of_range_error_names_values(p_values, n_outside):
    """A finite p-value outside [0, 1] raises, named by this function's param.

    The delegated SciPy error refers to its own parameter ``ps``; the message is
    restated in terms of ``p_values``, reports how many values are out of range,
    and keeps a domain hint so a caller who passed, e.g., coherence magnitudes is
    pointed at the real fix.
    """
    with pytest.raises(ValueError, match=r"p_values must all be in \[0, 1\]") as excinfo:
        Benjamini_Hochberg_procedure(np.array(p_values), alpha=0.05)
    message = str(excinfo.value)
    assert f"{n_outside} value(s) outside" in message
    assert "coherence_significance_pvalue" in message  # keeps the domain hint


def test_Benjamini_Hochberg_missing_scipy_raises_clear_error(monkeypatch):
    """A SciPy too old for false_discovery_control must fail with a named error.

    The project requires scipy>=1.11, but an environment can resolve an older
    SciPy; without the guard the call raises a bare AttributeError. Simulate the
    missing attribute and require an actionable RuntimeError instead.
    """
    import scipy.stats

    monkeypatch.delattr(scipy.stats, "false_discovery_control", raising=False)
    with pytest.raises(RuntimeError, match=r"scipy>=1\.11"):
        Benjamini_Hochberg_procedure(np.array([0.01, 0.2, 0.5]), alpha=0.05)


@pytest.mark.parametrize(
    ("p_values", "expected_is_significant"),
    [
        (np.ones((10, 2)), np.zeros((10, 2), dtype=bool)),
        (np.zeros((10, 2)), np.ones((10, 2), dtype=bool)),
        (np.array([0.03, 0.01, 0.04, 0.06]), np.array([False, True, False, False])),
        (np.array([0.03, 0.01, 0.04, 0.05]), np.array([False, True, False, False])),
    ],
)
def test_Bonferroni_correction(p_values, expected_is_significant):
    alpha = 0.05
    assert np.allclose(Bonferroni_correction(p_values, alpha), expected_is_significant)


def test_coherence_bias():
    n_observations = 10
    expected_bias = 1.0 / 18
    assert coherence_bias(n_observations) == expected_bias


def test_coherence_fisher_z_transform_one_sample_is_finite():
    """One-sample test against zero (n_obs2=0) must not produce NaN."""
    rng = np.random.default_rng(0)
    coherency = 0.4 * np.exp(1j * rng.uniform(0, 2 * np.pi, size=8))
    z = coherence_fisher_z_transform(coherency, n_obs1=50)
    assert np.all(np.isfinite(z))
    # Higher coherence -> larger positive z-score against the null of zero.
    high = coherence_fisher_z_transform(np.array([0.8 + 0j]), 50)[0]
    low = coherence_fisher_z_transform(np.array([0.1 + 0j]), 50)[0]
    assert high > low


def test_coherence_fisher_z_transform_one_sample_matches_analytic():
    """One-sample statistic is (arctanh|C| - bias) / sqrt(bias)."""
    coh = np.array([0.5 + 0j])
    n_obs = 20
    bias = coherence_bias(n_obs)
    expected = (np.arctanh(0.5) - bias) / np.sqrt(bias)
    assert np.allclose(coherence_fisher_z_transform(coh, n_obs), expected)


def test_coherence_significance_pvalue_is_well_calibrated():
    """Under the null (zero true coherence) the test rejects at ~the nominal rate.

    The Fisher one-sample transform is badly miscalibrated at the zero boundary
    (rejects ~16-22% at a nominal 5%); the exact Beta(1, n-1) null must be close
    to nominal.
    """
    rng = np.random.default_rng(0)
    alpha = 0.05
    n_rep = 200_000
    for n_obs in (5, 20, 50):
        # Two independent complex-Gaussian signals -> true coherence is zero.
        x = rng.standard_normal((n_rep, n_obs)) + 1j * rng.standard_normal((n_rep, n_obs))
        y = rng.standard_normal((n_rep, n_obs)) + 1j * rng.standard_normal((n_rep, n_obs))
        cross = (x * np.conj(y)).mean(axis=1)
        coherency = cross / np.sqrt(
            (np.abs(x) ** 2).mean(axis=1) * (np.abs(y) ** 2).mean(axis=1)
        )
        rejection_rate = np.mean(coherence_significance_pvalue(coherency, n_obs) <= alpha)
        assert np.isclose(rejection_rate, alpha, atol=0.01)


def test_coherence_significance_pvalue_matches_beta_null():
    """p-value equals (1 - |C|^2)^(n-1)."""
    coherency = np.array([0.0 + 0j, 0.3 + 0.1j, 0.9 + 0j])
    n_obs = 12
    expected = (1 - np.clip(np.abs(coherency) ** 2, 0, 1)) ** (n_obs - 1)
    assert np.allclose(coherence_significance_pvalue(coherency, n_obs), expected)
    # Zero coherence -> p-value of 1 (never significant); high coherence -> small p.
    assert coherence_significance_pvalue(np.array([0.0 + 0j]), n_obs)[0] == 1.0
    assert coherence_significance_pvalue(np.array([0.95 + 0j]), n_obs)[0] < 0.05


def test_coherence_rate_adjustment_valid_inputs():
    """A rate decrease with 1/f power gives finite, in-range adjustment factors."""
    freqs = np.linspace(1, 100, 50)
    power_spec = 1 / (1 + freqs**2)
    adjustment = coherence_rate_adjustment(15.0, 10.0, power_spec)
    assert np.all(np.isfinite(adjustment))
    assert np.all((adjustment > 0) & (adjustment <= 1))


def test_coherence_rate_adjustment_warns_when_undefined():
    """A large rate increase drives the argument negative -> NaN with a warning."""
    freqs = np.linspace(1, 100, 50)
    power_spec = 1 / (1 + freqs**2)
    with pytest.warns(UserWarning, match="undefined"):
        adjustment = coherence_rate_adjustment(10.0, 15.0, power_spec)
    assert np.any(np.isnan(adjustment))


def test_coherence_rate_adjustment_rejects_zero_rate():
    with pytest.raises(ValueError, match="firing_rate_condition1 must be a finite"):
        coherence_rate_adjustment(0.0, 10.0, np.array([1.0, 2.0]))


def test_coherence_rate_adjustment_zero_power_is_nan_without_runtime_warning():
    """A zero spike-power bin must return NaN (not 0) and emit no RuntimeWarning.

    The division and sqrt run under a scoped errstate; only the documented
    UserWarning is raised, and the undefined bin is NaN, not a silent 0.
    """
    power_spec = np.array([1.0, 0.0, 2.0])  # bin 1 has zero spike power
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)  # no leaked divide warning
        with pytest.warns(UserWarning, match="undefined"):
            adjustment = coherence_rate_adjustment(15.0, 10.0, power_spec)
    assert np.isnan(adjustment[1])
    assert np.all(np.isfinite(adjustment[[0, 2]]))


def test_coherence_rate_adjustment_nonpositive_argument_is_nan():
    """A non-positive argument (argument <= 0) returns NaN, not inf/0."""
    # Choose spike power so that 1 + adjusted_rate / power == 0 for one bin.
    # adjusted_rate is negative for a rate increase; pick power = -adjusted_rate.
    dt = 1.0
    rate1, rate2, noise = 10.0, 15.0, 0.0
    ratio = rate2 / rate1
    adjusted = ((1 / ratio - 1) * rate1 + noise / ratio**2) * dt**2
    # First bin -> argument == 0 (undefined); second bin large enough that
    # argument = 1 + adjusted/power > 0 (adjusted is negative for a rate rise).
    power_spec = np.array([-adjusted, 100.0])
    with pytest.warns(UserWarning, match="undefined"):
        adjustment = coherence_rate_adjustment(
            rate1, rate2, power_spec, dt=dt, homogeneous_poisson_noise=noise
        )
    assert np.isnan(adjustment[0])
    assert np.isfinite(adjustment[1])


def test_power_bias_and_variance_match_log_chi2_moments():
    """Multitaper power S_hat/S ~ chi2_nu / nu with nu = 2 * n_obs.

    bias = E[log(S_hat/S)], variance = Var[log(S_hat/S)], checked by
    simulation against the closed-form digamma/trigamma expressions.
    """
    rng = np.random.default_rng(1)
    for n_obs in (5, 25, 100):
        nu = 2 * n_obs
        log_samples = np.log(rng.chisquare(nu, size=1_000_000) / nu)
        assert np.allclose(power_bias(n_obs), log_samples.mean(), atol=2e-3)
        assert np.allclose(power_variance(n_obs), log_samples.var(), rtol=5e-2)


def test_power_confidence_intervals_coverage():
    """A nominal 95% CI must actually cover the true power ~95% of the time."""
    rng = np.random.default_rng(2)
    n_tapers = 8
    nu = 2 * n_tapers
    true_power = 3.0
    estimates = true_power * rng.chisquare(nu, size=200_000) / nu
    lower, upper = power_confidence_intervals(n_tapers, estimates, ci=0.95)
    coverage = np.mean((lower <= true_power) & (true_power <= upper))
    assert np.isclose(coverage, 0.95, atol=0.01)


def test_power_fisher_z_transform_one_sample_is_finite():
    """One-sample power test against a positive baseline must be finite."""
    z = power_fisher_z_transform(np.array([0.5, 1.0, 2.0]), n_obs1=50, spectrum2=1.0)
    assert np.all(np.isfinite(z))


def test_power_fisher_z_transform_one_sample_matches_analytic():
    """One-sample statistic is (log(spectrum1) - bias1 - log(baseline)) / sqrt(var1).

    bias = digamma(n) - log(n) and variance = trigamma(n) (log of a
    chi2_{2n} / 2n variable). A baseline other than 1 is used so that
    ``log(baseline)`` is not identically zero.
    """
    from scipy.special import polygamma, psi

    spectrum1 = np.array([0.5, 2.0, 6.0])
    baseline = 2.5
    n_obs = 30
    bias = psi(n_obs) - np.log(n_obs)
    variance = polygamma(1, n_obs)
    expected = (np.log(spectrum1) - bias - np.log(baseline)) / np.sqrt(variance)
    result = power_fisher_z_transform(spectrum1, n_obs1=n_obs, spectrum2=baseline)
    np.testing.assert_allclose(result, expected, rtol=1e-12)
    # Power above the baseline gives a positive z-score, below gives negative.
    assert result[2] > 0 > result[1]


def test_power_fisher_z_transform_two_sample_matches_analytic():
    """Two-sample statistic subtracts both biases and pools both variances.

    z = ((log S1 - b1) - (log S2 - b2)) / sqrt(v1 + v2) with
    b = digamma(n) - log(n) and v = trigamma(n) for each sample.
    """
    from scipy.special import polygamma, psi

    spectrum1 = np.array([0.5, 2.0, 6.0])
    spectrum2 = np.array([1.5, 1.0, 3.0])
    n_obs1, n_obs2 = 30, 8
    bias1, bias2 = psi(n_obs1) - np.log(n_obs1), psi(n_obs2) - np.log(n_obs2)
    variance1, variance2 = polygamma(1, n_obs1), polygamma(1, n_obs2)
    expected = ((np.log(spectrum1) - bias1) - (np.log(spectrum2) - bias2)) / np.sqrt(
        variance1 + variance2
    )
    result = power_fisher_z_transform(spectrum1, n_obs1, spectrum2, n_obs2)
    np.testing.assert_allclose(result, expected, rtol=1e-12)


def test_power_fisher_z_transform_rejects_nonpositive_power():
    """Non-positive power would give silent -inf/nan; must raise instead."""
    with pytest.raises(ValueError, match="spectrum1 must be finite and strictly positive"):
        power_fisher_z_transform(np.array([1.0, 0.0]), n_obs1=30, spectrum2=1.0)
    with pytest.raises(ValueError, match="spectrum2 must be finite and strictly positive"):
        power_fisher_z_transform(np.array([1.0, 2.0]), n_obs1=30, spectrum2=0.0)


@pytest.mark.parametrize("bad_ci", [0.3, 1.0, 1.5, -0.1])
def test_power_confidence_intervals_rejects_out_of_range_ci(bad_ci):
    """ci must be in [0.5, 1.0); out-of-range values raise instead of inverting."""
    with pytest.raises(
        ValueError, match=r"Confidence level `ci` must be in the range \[0\.5, 1\.0\)"
    ):
        power_confidence_intervals(n_tapers=5, power=1.0, ci=bad_ci)


def test_import_does_not_change_numpy_error_state():
    """Importing the package must not globally suppress NumPy warnings.

    A module-level ``np.seterr`` would leak into all caller code, silently
    turning invalid-operation warnings into NaNs process-wide.
    """
    code = (
        "import numpy as np; before = np.geterr()['invalid'];"
        "import spectral_connectivity;"
        "print(before, np.geterr()['invalid'])"
    )
    out = subprocess.check_output([sys.executable, "-c", code], text=True).split()
    assert out == ["warn", "warn"]


def test_get_normal_distribution_p_values_survival_precision():
    """Far-tail p-values use the survival function, not 1 - cdf.

    ``1 - norm.cdf(8.3)`` underflows to exactly 0; ``norm.sf`` keeps precision.
    """
    p = get_normal_distribution_p_values(8.3)
    assert p > 0
    assert np.allclose(p, 5.2055697448902465e-17, rtol=1e-6)


def test_coherence_rate_adjustment_rejects_zero_second_rate():
    """firing_rate_condition2 <= 0 must raise, not ZeroDivisionError."""
    with pytest.raises(ValueError, match="firing_rate_condition2 must be a finite"):
        coherence_rate_adjustment(10.0, 0.0, np.array([1.0, 2.0]))


def test_coherence_rate_adjustment_negative_power_is_masked():
    """Non-positive spike power is invalid and must be returned as NaN."""
    power_spec = np.array([1.0, -2.0, 3.0])  # bin 1 has invalid negative power
    with pytest.warns(UserWarning, match="non-positive spike power"):
        adjustment = coherence_rate_adjustment(12.0, 10.0, power_spec)
    assert np.isnan(adjustment[1])
    assert np.all(np.isfinite(adjustment[[0, 2]]))


@pytest.mark.parametrize("bad_n_obs", [0, 1])
def test_coherence_significance_pvalue_rejects_small_n_observations(bad_n_obs):
    """n_observations < 2 would give values outside [0, 1]; must raise."""
    with pytest.raises(ValueError, match="n_observations must be a finite integer >= 2"):
        coherence_significance_pvalue(np.array([0.5 + 0j]), bad_n_obs)


@pytest.mark.parametrize("bad_rate", [np.inf, np.nan])
def test_coherence_rate_adjustment_rejects_nonfinite_rate(bad_rate):
    """A non-finite firing rate must raise, not pass and hit ZeroDivisionError."""
    with pytest.raises(ValueError, match="firing_rate_condition1 must be a finite"):
        coherence_rate_adjustment(bad_rate, 10.0, np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="firing_rate_condition2 must be a finite"):
        coherence_rate_adjustment(10.0, bad_rate, np.array([1.0, 2.0]))


@pytest.mark.parametrize("bad_n_obs", [np.nan, np.inf, 2.5])
def test_coherence_significance_pvalue_rejects_nonfinite_or_noninteger(bad_n_obs):
    """Non-finite or non-integer observation counts must raise, not give NaN."""
    with pytest.raises(ValueError, match="finite integer >= 2"):
        coherence_significance_pvalue(np.array([0.5 + 0j]), bad_n_obs)


@pytest.mark.parametrize("bad_n_obs", [np.nan, np.inf, 2.5])
def test_power_fisher_z_transform_rejects_nonfinite_or_noninteger_counts(bad_n_obs):
    with pytest.raises(ValueError, match="n_obs1 must be a finite integer"):
        power_fisher_z_transform(np.array([1.0]), n_obs1=bad_n_obs, spectrum2=1.0)


@pytest.mark.parametrize("bad_spectrum", [np.nan, np.inf])
def test_power_fisher_z_transform_rejects_nonfinite_spectrum(bad_spectrum):
    with pytest.raises(ValueError, match="spectrum1 must be finite"):
        power_fisher_z_transform(np.array([1.0, bad_spectrum]), n_obs1=50, spectrum2=1.0)


@pytest.mark.parametrize("bad_n_tapers", [0, -3, np.nan, np.inf, 2.5])
def test_power_confidence_intervals_rejects_invalid_n_tapers(bad_n_tapers):
    with pytest.raises(ValueError, match="n_tapers must be a finite positive integer"):
        power_confidence_intervals(n_tapers=bad_n_tapers, power=1.0, ci=0.95)


@pytest.mark.parametrize("bad_n_obs1", [1, 0, np.nan, np.inf, 2.5])
def test_coherence_fisher_z_transform_rejects_invalid_n_obs1(bad_n_obs1):
    """n_obs1 must be a finite integer >= 2 (n_obs=1 hit ZeroDivisionError)."""
    with pytest.raises(ValueError, match="n_obs1 must be a finite integer >= 2"):
        coherence_fisher_z_transform(np.array([0.5 + 0j]), n_obs1=bad_n_obs1)


@pytest.mark.parametrize("bad_n_obs2", [1, -2, np.nan, 3.5])
def test_coherence_fisher_z_transform_rejects_invalid_n_obs2(bad_n_obs2):
    """n_obs2 must be a finite integer equal to 0 or >= 2."""
    with pytest.raises(ValueError, match="n_obs2 must be a finite integer"):
        coherence_fisher_z_transform(
            np.array([0.5 + 0j]),
            n_obs1=10,
            coherency2=np.array([0.3 + 0j]),
            n_obs2=bad_n_obs2,
        )


@pytest.mark.parametrize("bad_power", [-1.0, np.nan, np.inf])
def test_power_confidence_intervals_rejects_invalid_power(bad_power):
    """Negative or non-finite power must raise, not return reversed/NaN bounds."""
    with pytest.raises(ValueError, match="power must be finite and non-negative"):
        power_confidence_intervals(n_tapers=5, power=bad_power, ci=0.95)


# Two-sided 95% critical values: the standard normal, norm.ppf(0.975), and
# Student t with n - 1 degrees of freedom for n = 3 and n = 5 leave-one-out
# replicates, t.ppf(0.975, df).
_Z_975 = 1.959963984540054
_T_975_DF2 = 4.302652729749462
_T_975_DF3 = 3.182446305284263
_T_975_DF4 = 2.7764451051977934


def test_jackknife_confidence_interval_matches_mean_example():
    """Identity transform: estimate +/- t_{n-1} * jackknife SE.

    Replicates [2.5, 2.0, 1.5] have mean 2 and jackknife variance
    (n - 1) / n * sum((r - mean)**2) = 2/3 * 0.5 = 1/3, so with the Student t
    critical value on 2 degrees of freedom the 95% interval is
    2 -/+ 4.302653 * sqrt(1/3) = [-0.4841, 4.4841].
    """
    result = jackknife_confidence_interval(
        np.array(2.0), np.array([2.5, 2.0, 1.5]), confidence_level=0.95
    )

    assert result.estimate == pytest.approx(2.0)
    assert result.bias_corrected == pytest.approx(2.0)
    assert result.standard_error == pytest.approx(np.sqrt(1 / 3))
    half_width = _T_975_DF2 * np.sqrt(1 / 3)
    np.testing.assert_allclose(
        result.confidence_interval, (2.0 - half_width, 2.0 + half_width), rtol=1e-12
    )
    np.testing.assert_allclose(result.confidence_interval, (-0.4841, 4.4841), atol=1e-4)


def test_jackknife_interval_uses_student_t_not_normal_critical_value():
    """The half-width is t_{n-1} * SE, which is wider than z * SE at small n.

    With n leave-one-out replicates the jackknife variance has n - 1 degrees
    of freedom (Thomson & Chave 1991; Efron & Tibshirani 1993); the normal
    quantile under-covers (about 0.88 instead of 0.95 at n = 5).
    """
    replicates = np.array([1.0, 1.4, 0.7, 1.2, 0.9])
    n = replicates.size
    standard_error = np.sqrt((n - 1) / n * np.sum((replicates - replicates.mean()) ** 2))
    result = jackknife_confidence_interval(np.array(1.05), replicates, confidence_level=0.95)

    lower, upper = result.confidence_interval
    half_width = (upper - lower) / 2
    np.testing.assert_allclose(half_width, _T_975_DF4 * standard_error, rtol=1e-12)
    # The normal critical value would give a visibly narrower interval.
    assert half_width > 1.3 * _Z_975 * standard_error


@pytest.mark.slow
def test_jackknife_interval_coverage_for_gaussian_mean():
    """A nominal 95% jackknife interval for a plain mean covers ~95% of the time.

    For the sample mean the jackknife SE equals s / sqrt(n) exactly, so with a
    Student t critical value the coverage is exactly nominal; the normal
    quantile would give ~0.91 at n = 8.
    """
    rng = np.random.default_rng(7)
    n_repetitions, n = 1000, 8
    samples = rng.standard_normal((n_repetitions, n))
    estimate = samples.mean(axis=1)
    # Leave-one-out means, stacked on the first axis: shape (n, n_repetitions).
    leave_one_out = (samples.sum(axis=1)[None, :] - samples.T) / (n - 1)
    result = jackknife_confidence_interval(estimate, leave_one_out, confidence_level=0.95)

    lower, upper = result.confidence_interval
    coverage = np.mean((lower <= 0.0) & (upper >= 0.0))
    assert 0.92 <= coverage <= 0.98


def test_jackknife_log_and_circular_transforms_return_original_scale():
    replicates = np.array([1.8, 2.0, 2.2])
    log_result = jackknife_confidence_interval(
        np.array(2.0),
        replicates,
        transformation="log",
    )
    assert log_result.transformation == "log"
    # The interval is formed on log scale and exponentiated back, so it is
    # asymmetric about the estimate (unlike the identity interval).
    log_replicates = np.log(replicates)
    n = replicates.size
    log_standard_error = np.sqrt(
        (n - 1) / n * np.sum((log_replicates - log_replicates.mean()) ** 2)
    )
    expected_interval = np.exp(
        np.log(2.0) + np.array([-1, 1]) * _T_975_DF2 * log_standard_error
    )
    np.testing.assert_allclose(log_result.confidence_interval, expected_interval, rtol=1e-12)
    np.testing.assert_allclose(
        log_result.bias_corrected,
        np.exp(n * np.log(2.0) - (n - 1) * log_replicates.mean()),
        rtol=1e-12,
    )
    # Delta-method standard error back on the original scale: estimate * log SE.
    np.testing.assert_allclose(log_result.standard_error, 2.0 * log_standard_error, rtol=1e-12)

    phases = np.array([np.pi - 0.1, -np.pi + 0.1, np.pi - 0.05])
    circular = jackknife_confidence_interval(
        np.array(np.pi), phases, transformation="circular"
    )
    assert abs(circular.bias_corrected) > 3.0


def test_jackknife_circular_interval_wider_than_the_circle_warns():
    """A circular half-width >= pi is the whole circle, not a narrow interval.

    The circular SE is a linear SE of the unwrapped replicates and is
    unbounded; wrapping bounds that are more than 2*pi apart would report a
    whole-circle interval as e.g. (-1.65, 1.65). Such bins must be pinned to
    (-pi, pi) with a warning, while resolved bins keep their wrapped bounds.
    """
    wide = np.array([-2.0, 0.0, 2.0, -2.5, 2.5])
    tight = np.array([0.9, 1.0, 1.1, 0.95, 1.05])
    with pytest.warns(UserWarning, match="whole circle") as record:
        result = jackknife_confidence_interval(
            np.array([0.0, 1.0]), np.stack([wide, tight], axis=1), transformation="circular"
        )
    assert len(record) == 1
    assert "1 value(s)" in str(record[0].message)
    lower, upper = result.confidence_interval
    # The unbounded standard error is still reported (sqrt(4/5 * 20.5) = 4.05).
    np.testing.assert_allclose(result.standard_error[0], np.sqrt(0.8 * 20.5), rtol=1e-12)
    assert (lower[0], upper[0]) == (-np.pi, np.pi)
    # The resolved bin is untouched: a wrapped interval bracketing the estimate.
    assert -np.pi < lower[1] < 1.0 < upper[1] <= np.pi
    n = tight.size
    tight_half_width = _T_975_DF4 * np.sqrt((n - 1) / n * np.sum((tight - tight.mean()) ** 2))
    np.testing.assert_allclose(
        (lower[1], upper[1]), (1.0 - tight_half_width, 1.0 + tight_half_width), rtol=1e-12
    )
    # A scalar estimate takes the same path.
    with pytest.warns(UserWarning, match="whole circle"):
        scalar = jackknife_confidence_interval(np.array(0.0), wide, transformation="circular")
    assert scalar.confidence_interval == (-np.pi, np.pi)


def test_jackknife_circular_interval_crossing_pi_has_lower_above_upper():
    """Bounds are wrapped to (-pi, pi]; lower > upper marks a crossing of +/-pi.

    Estimate 3.0 with replicates near +/-pi: the replicate at -3.1 is unwrapped
    to 3.183 before the SE is formed, and the upper bound 3.0 + half-width
    exceeds pi so it wraps negative. The interval is [lower, pi] U (-pi, upper].
    """
    replicates = np.array([3.1, 2.9, -3.1, 3.0])
    estimate = 3.0
    unwrapped = estimate + np.angle(np.exp(1j * (replicates - estimate)))
    n = replicates.size
    half_width = _T_975_DF3 * np.sqrt(
        (n - 1) / n * np.sum((unwrapped - unwrapped.mean()) ** 2)
    )
    assert half_width < np.pi  # a resolved interval, so no warning is expected

    result = jackknife_confidence_interval(
        np.array(estimate), replicates, transformation="circular"
    )

    lower, upper = result.confidence_interval
    assert lower > upper
    np.testing.assert_allclose(lower, estimate - half_width, rtol=1e-12)
    np.testing.assert_allclose(upper, estimate + half_width - 2 * np.pi, rtol=1e-12)
    # The estimate lies in the [lower, pi] arm of the wrapped interval.
    assert lower <= estimate <= np.pi


def test_jackknife_fisher_squared_matches_atanh_of_magnitude():
    # fisher_squared applies the atanh(sqrt(.)) variance-stabilizing transform
    # for magnitude-squared coherence. Its confidence interval must equal the
    # squared plain-fisher interval computed on the unsquared magnitude.
    magnitude_estimate = 0.6
    magnitude_replicates = np.array([0.55, 0.6, 0.65])
    squared = jackknife_confidence_interval(
        np.array(magnitude_estimate**2),
        np.array(magnitude_replicates**2),
        transformation="fisher_squared",
    )
    magnitude = jackknife_confidence_interval(
        np.array(magnitude_estimate),
        magnitude_replicates,
        transformation="fisher",
    )
    assert squared.transformation == "fisher_squared"
    # Bounds and interval map through the square of the magnitude interval.
    np.testing.assert_allclose(
        squared.confidence_interval[0], magnitude.confidence_interval[0] ** 2
    )
    np.testing.assert_allclose(
        squared.confidence_interval[1], magnitude.confidence_interval[1] ** 2
    )
    # The interval stays ordered and brackets the estimate.
    assert squared.confidence_interval[0] <= magnitude_estimate**2
    assert squared.confidence_interval[1] >= magnitude_estimate**2


def test_jackknife_fisher_squared_interval_is_monotonic_near_zero():
    # Small estimate with wide spread: the lower atanh bound maps below zero, and
    # squaring must not fold it back above the estimate.
    result = jackknife_confidence_interval(
        np.array(0.01),
        np.array([0.0, 0.02, 0.05]),
        transformation="fisher_squared",
    )
    assert result.confidence_interval[0] <= 0.01
    assert result.confidence_interval[1] >= 0.01
    assert result.confidence_interval[0] >= 0.0


def test_jackknife_log_warns_on_non_positive_values():
    with pytest.warns(UserWarning, match="non-positive"):
        jackknife_confidence_interval(
            np.array(1.0),
            np.array([1.0, -0.5, 2.0]),
            transformation="log",
        )


def test_jackknife_fisher_warns_at_saturated_coherence():
    with pytest.warns(UserWarning, match="saturated coherence"):
        jackknife_confidence_interval(
            np.array(1.0),
            np.array([0.99, 1.0, 0.995]),
            transformation="fisher",
        )


def test_jackknife_rejects_unknown_transformation():
    with pytest.raises(ValueError, match="transformation must be"):
        jackknife_confidence_interval(
            np.array(1.0), np.array([1.0, 2.0]), transformation="bogus"
        )
