import subprocess
import sys
import warnings

import numpy as np
import pytest
from pytest import mark

from spectral_connectivity.statistics import (
    Benjamini_Hochberg_procedure,
    Bonferroni_correction,
    coherence_bias,
    coherence_fisher_z_transform,
    coherence_rate_adjustment,
    coherence_significance_pvalue,
    get_normal_distribution_p_values,
    power_bias,
    power_confidence_intervals,
    power_fisher_z_transform,
    power_variance,
)


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
    # All-NaN input yields all-False (no valid tests), same input shape.
    all_nan = Benjamini_Hochberg_procedure(np.full((2, 3), np.nan), alpha)
    assert all_nan.shape == (2, 3)
    assert not all_nan.any()


def test_Benjamini_Hochberg_rejects_out_of_range_pvalues():
    """A finite p-value outside [0, 1] is invalid input and must raise."""
    with pytest.raises(ValueError):
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
