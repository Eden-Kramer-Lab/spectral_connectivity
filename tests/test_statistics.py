import subprocess
import sys
import warnings

import numpy as np
import pytest
from pytest import mark

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


@pytest.mark.parametrize(
    "correction", [Benjamini_Hochberg_procedure, Bonferroni_correction]
)
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


def test_fisher_z_transform():
    coherency = 0.5 * np.exp(1j * np.pi / 2) * np.ones((2, 2))
    n_obs1, n_obs2 = 6, 6
    expected_difference_z = np.zeros((2, 2))
    assert np.allclose(
        coherence_fisher_z_transform(
            coherency, n_obs1, coherency2=coherency, n_obs2=n_obs2
        ),
        expected_difference_z,
    )


@mark.parametrize(
    "p_values, expected_is_significant",
    [
        (np.ones((10, 2)), np.zeros((10, 2), dtype=bool)),
        (np.zeros((10, 2)), np.ones((10, 2), dtype=bool)),
        (np.array([0.03, 0.01, 0.04, 0.06]), np.array([False, True, False, False])),
        (np.array([0.03, 0.01, 0.04, 0.05]), np.ones((4,))),
    ],
)
def test_Benjamini_Hochberg_procedure(p_values, expected_is_significant):
    alpha = 0.05
    assert np.allclose(
        Benjamini_Hochberg_procedure(p_values, alpha), expected_is_significant
    )


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


def test_Benjamini_Hochberg_out_of_range_error_names_values():
    """The out-of-range error reports how many values and their min/max."""
    with pytest.raises(ValueError) as excinfo:
        Benjamini_Hochberg_procedure(np.array([0.1, 1.5, -0.2, 0.3]), alpha=0.05)
    message = str(excinfo.value)
    assert "2 value(s) outside" in message
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


def test_Benjamini_Hochberg_rejects_out_of_range_pvalues():
    """A finite p-value outside [0, 1] must raise, named by this function's param.

    The delegated SciPy error refers to its own parameter ``ps``; the message is
    restated in terms of ``p_values`` with a domain hint so a caller who passed,
    e.g., coherence magnitudes is pointed at the real fix.
    """
    with pytest.raises(ValueError, match="p_values must all be in"):
        Benjamini_Hochberg_procedure(np.array([0.1, 1.5, 0.2]), alpha=0.05)


@mark.parametrize(
    "p_values, expected_is_significant",
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
        x = rng.standard_normal((n_rep, n_obs)) + 1j * rng.standard_normal(
            (n_rep, n_obs)
        )
        y = rng.standard_normal((n_rep, n_obs)) + 1j * rng.standard_normal(
            (n_rep, n_obs)
        )
        cross = (x * np.conj(y)).mean(axis=1)
        coherency = cross / np.sqrt(
            (np.abs(x) ** 2).mean(axis=1) * (np.abs(y) ** 2).mean(axis=1)
        )
        rejection_rate = np.mean(
            coherence_significance_pvalue(coherency, n_obs) <= alpha
        )
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
    """One-sample statistic is (log(spectrum1) - bias1) / sqrt(variance1)."""
    spectrum1 = np.array([0.5, 2.0])
    n_obs = 30
    expected = (np.log(spectrum1) - power_bias(n_obs)) / np.sqrt(power_variance(n_obs))
    result = power_fisher_z_transform(spectrum1, n_obs1=n_obs, spectrum2=1.0)
    assert np.allclose(result, expected)
    # Power above the baseline gives a positive z-score, below gives negative.
    assert result[1] > 0 > result[0]


def test_power_fisher_z_transform_rejects_nonpositive_power():
    """Non-positive power would give silent -inf/nan; must raise instead."""
    with pytest.raises(
        ValueError, match="spectrum1 must be finite and strictly positive"
    ):
        power_fisher_z_transform(np.array([1.0, 0.0]), n_obs1=30, spectrum2=1.0)
    with pytest.raises(
        ValueError, match="spectrum2 must be finite and strictly positive"
    ):
        power_fisher_z_transform(np.array([1.0, 2.0]), n_obs1=30, spectrum2=0.0)


@mark.parametrize("bad_ci", [0.3, 1.0, 1.5, -0.1])
def test_power_confidence_intervals_rejects_out_of_range_ci(bad_ci):
    """ci must be in [0.5, 1.0); out-of-range values raise instead of inverting."""
    with pytest.raises(ValueError, match="ci"):
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
    with pytest.raises(
        ValueError, match="n_observations must be a finite integer >= 2"
    ):
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
        power_fisher_z_transform(
            np.array([1.0, bad_spectrum]), n_obs1=50, spectrum2=1.0
        )


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


def test_jackknife_confidence_interval_matches_mean_example():
    result = jackknife_confidence_interval(
        np.array(2.0), np.array([2.5, 2.0, 1.5]), confidence_level=0.95
    )

    assert result.estimate == pytest.approx(2.0)
    assert result.bias_corrected == pytest.approx(2.0)
    assert result.standard_error == pytest.approx(np.sqrt(1 / 3))
    assert result.confidence_interval[0] < result.estimate
    assert result.confidence_interval[1] > result.estimate


def test_jackknife_log_and_circular_transforms_return_original_scale():
    log_result = jackknife_confidence_interval(
        np.array(2.0),
        np.array([1.8, 2.0, 2.2]),
        transformation="log",
    )
    assert log_result.transformation == "log"
    assert log_result.confidence_interval[0] > 0

    phases = np.array([np.pi - 0.1, -np.pi + 0.1, np.pi - 0.05])
    circular = jackknife_confidence_interval(
        np.array(np.pi), phases, transformation="circular"
    )
    assert abs(circular.bias_corrected) > 3.0


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
