"""Statistical procedures for connectivity analysis.

This module provides statistical functions for testing significance of
connectivity measures, including multiple comparison corrections and
transforms for coherence-based measures. Functions support both parametric
and non-parametric approaches for statistical inference in frequency domain
connectivity analysis.
"""

from collections.abc import Callable
from typing import Literal

import numpy as np
import scipy.special
import scipy.stats
from numpy.typing import NDArray

np.seterr(invalid="ignore")


def Benjamini_Hochberg_procedure(
    p_values: NDArray[np.floating], alpha: float = 0.05
) -> NDArray[np.bool_]:
    """Control false discovery rate using Benjamini-Hochberg procedure.

    Corrects for multiple comparisons and returns significant p-values by
    controlling the false discovery rate at level `alpha` using the
    Benjamini-Hochberg procedure.

    Parameters
    ----------
    p_values : NDArray[floating], shape (...,)
        P-values from statistical tests to be corrected.
    alpha : float, default=0.05
        Expected proportion of false positive tests (false discovery rate).

    Returns
    -------
    is_significant : NDArray[bool], shape (...,)
        Boolean array same shape as `p_values` indicating whether the
        null hypothesis has been rejected (True) or failed to reject (False).

    Examples
    --------
    >>> import numpy as np
    >>> p_vals = np.array([0.001, 0.02, 0.04, 0.3, 0.8])
    >>> significant = Benjamini_Hochberg_procedure(p_vals, alpha=0.05)
    >>> significant
    array([ True,  True, False, False, False])
    """
    p_values = np.array(p_values)
    threshold_line = np.linspace(0, alpha, num=p_values.size + 1, endpoint=True)[1:]
    sorted_p_values = np.sort(p_values.flatten())
    try:
        threshold_ind: int = np.max(np.where(sorted_p_values <= threshold_line)[0])
        threshold = sorted_p_values[threshold_ind]
    except ValueError:  # There are no values below threshold
        threshold = -1
    return p_values <= threshold


def Bonferroni_correction(
    p_values: NDArray[np.floating], alpha: float = 0.05
) -> NDArray[np.bool_]:
    """Control family-wise error rate using Bonferroni correction.

    Corrects for multiple comparisons by dividing the significance level
    by the number of tests. This is a conservative method that controls
    the family-wise error rate.

    Parameters
    ----------
    p_values : NDArray[floating], shape (...,)
        P-values from statistical tests to be corrected.
    alpha : float, default=0.05
        Critical threshold for significance testing.

    Returns
    -------
    is_significant : NDArray[bool], shape (...,)
        Boolean array indicating significant tests after Bonferroni correction.

    Examples
    --------
    >>> import numpy as np
    >>> p_vals = np.array([0.001, 0.02, 0.04, 0.3, 0.8])
    >>> significant = Bonferroni_correction(p_vals, alpha=0.05)
    >>> significant
    array([ True, False, False, False, False])
    """
    p_values = np.asarray(p_values)
    return p_values <= alpha / p_values.size


MULTIPLE_COMPARISONS: dict[str, Callable] = {
    "Benjamini_Hochberg_procedure": Benjamini_Hochberg_procedure,
    "Bonferroni_correction": Bonferroni_correction,
}


def adjust_for_multiple_comparisons(
    p_values: NDArray[np.floating],
    alpha: float = 0.05,
    method: Literal[
        "Benjamini_Hochberg_procedure", "Bonferroni_correction"
    ] = "Benjamini_Hochberg_procedure",
) -> NDArray[np.bool_]:
    """Apply multiple comparison correction to p-values.

    Wrapper function that applies the specified multiple comparison correction
    method to control either false discovery rate or family-wise error rate.

    Parameters
    ----------
    p_values : NDArray[floating], shape (...,)
        P-values from statistical tests to be corrected.
    alpha : float, default=0.05
        Significance threshold for the correction method.
    method : {"Benjamini_Hochberg_procedure", "Bonferroni_correction"},
        default="Benjamini_Hochberg_procedure"
        Multiple comparison correction method to apply.

    Returns
    -------
    is_significant : NDArray[bool], shape (...,)
        Boolean array indicating which tests remain significant after correction.

    Examples
    --------
    >>> import numpy as np
    >>> p_vals = np.array([0.001, 0.02, 0.04, 0.3, 0.8])
    >>> # Using Benjamini-Hochberg (default)
    >>> bh_sig = adjust_for_multiple_comparisons(p_vals)
    >>> # Using Bonferroni
    >>> bonf_sig = adjust_for_multiple_comparisons(
    ...     p_vals, method="Bonferroni_correction"
    ... )
    """
    # TODO: add axis keyword?
    return MULTIPLE_COMPARISONS[method](p_values, alpha=alpha)


def coherence_fisher_z_transform(
    coherency1: NDArray[np.complexfloating],
    n_obs1: int,
    coherency2: NDArray[np.complexfloating] | float = 0,
    n_obs2: int = 0,
) -> NDArray[np.floating]:
    """Transform coherence magnitude to approximately normal distribution.

    Applies Fisher's z-transformation to coherence magnitudes, which
    approximately normalizes the distribution for statistical testing.
    Can compute single-sample test against zero or two-sample test.

    Parameters
    ----------
    coherency1 : NDArray[complexfloating], shape (...,)
        Complex coherency values between signals.
    n_obs1 : int
        Number of observations for coherency1 (n_tapers * n_trials).
    coherency2 : NDArray[complexfloating] or float, default=0
        Second coherency for comparison. If 0, tests against null hypothesis.
    n_obs2 : int, default=0
        Number of observations for coherency2 (n_tapers * n_trials).

    Returns
    -------
    fisher_z_transform : NDArray[floating], shape (...,)
        Z-scores for statistical testing. If coherency2=0, tests coherency1
        against zero. Otherwise, tests difference coherency1 - coherency2.

    Examples
    --------
    >>> import numpy as np
    >>> # Test single coherence against zero
    >>> coherence = np.array([0.1 + 0.05j, 0.3 + 0.2j, 0.8 + 0.1j])
    >>> z_scores = coherence_fisher_z_transform(coherence, n_obs1=100)
    >>>
    >>> # Compare two coherences
    >>> coh1 = np.array([0.5 + 0.2j, 0.3 + 0.1j])
    >>> coh2 = np.array([0.3 + 0.15j, 0.4 + 0.05j])
    >>> diff_z = coherence_fisher_z_transform(coh1, 100, coh2, 120)

    Notes
    -----
    The transformation uses bias correction based on the number of observations
    to improve the normal approximation for small sample sizes.
    """
    coherence_magnitude1 = np.abs(coherency1)
    coherence_magnitude1[coherence_magnitude1 >= 1] = 1 - np.finfo(float).eps

    coherence_magnitude2 = np.array(np.abs(coherency2))
    coherence_magnitude2[coherence_magnitude2 >= 1] = 1 - np.finfo(float).eps

    bias1, bias2 = coherence_bias(n_obs1), coherence_bias(n_obs2)

    z1 = np.arctanh(coherence_magnitude1) - bias1
    z2 = np.arctanh(coherence_magnitude2) - bias2
    return (z1 - z2) / np.sqrt(bias1 + bias2)


def get_normal_distribution_p_values(
    data: NDArray[np.floating],
    mean: float = 0,
    std_deviation: float = 1,
) -> NDArray[np.floating]:
    """Compute p-values for normal distribution test.

    Given data values, returns the probability that each value was generated
    from a normal distribution with specified mean and standard deviation.
    This computes one-tailed p-values (upper tail).

    Parameters
    ----------
    data : NDArray[floating], shape (...,)
        Data values to test.
    mean : float, default=0
        Mean of the null hypothesis normal distribution.
    std_deviation : float, default=1
        Standard deviation of the null hypothesis normal distribution.

    Returns
    -------
    p_values : NDArray[floating], shape (...,)
        One-tailed p-values (upper tail) for each data point.

    Examples
    --------
    >>> import numpy as np
    >>> z_scores = np.array([-1.96, 0, 1.96, 2.58])
    >>> p_vals = get_normal_distribution_p_values(z_scores)
    >>> p_vals
    array([0.975, 0.5, 0.025, 0.005])

    Notes
    -----
    This function handles both NumPy and CuPy arrays automatically,
    falling back to NumPy computation if CuPy fails.
    """
    try:
        return 1 - scipy.stats.norm.cdf(data, loc=mean, scale=std_deviation)
    except TypeError:
        return 1 - scipy.stats.norm.cdf(data.get(), loc=mean, scale=std_deviation)  # type: ignore[attr-defined]


def coherence_bias(n_observations: int) -> float:
    """Estimate bias correction for coherence estimates.

    Coherence estimates are biased by finite sample size. This function
    computes the bias correction factor that can be subtracted from
    Fisher z-transformed coherence estimates.

    Parameters
    ----------
    n_observations : int
        Number of observations used in coherence estimation (n_tapers * n_trials).

    Returns
    -------
    bias : float
        Bias correction factor for Fisher z-transform of coherence.

    Examples
    --------
    >>> bias_100 = coherence_bias(100)
    >>> bias_1000 = coherence_bias(1000)
    >>> print(f"Bias with 100 obs: {bias_100:.6f}")
    >>> print(f"Bias with 1000 obs: {bias_1000:.6f}")
    Bias with 100 obs: 0.005051
    Bias with 1000 obs: 0.000501

    References
    ----------
    .. [1] Enochson, L.D., and Goodman, N.R. (1965). Gaussian approximations
           to the distribution
           of sample coherence (Measurement analysis corp Los Angeles CA).
    .. [2] Bokil, H., Purpura, K., Schoffelen, J.-M., Thomson, D., and Mitra, P.
           (2007). Comparing
           spectra and coherences for groups of unequal size.
           Journal of Neuroscience Methods 159,
           337–345. 10.1016/j.jneumeth.2006.07.011.
    """
    degrees_of_freedom = 2 * n_observations
    return 1.0 / (degrees_of_freedom - 2)


def coherence_rate_adjustment(
    firing_rate_condition1: float,
    firing_rate_condition2: float,
    spike_power_spectrum: NDArray[np.floating],
    homogeneous_poisson_noise: float = 0,
    dt: float = 1,
) -> NDArray[np.floating]:
    """Adjust spike-field coherence for different firing rates between conditions.

    When comparing coherence between conditions with different firing rates,
    rate differences can cause coherence changes independent of coupling strength.
    This function computes adjustment factors to correct for firing rate differences.

    Parameters
    ----------
    firing_rate_condition1 : float
        Average firing rate in first condition (spikes/sec).
    firing_rate_condition2 : float
        Average firing rate in second condition (spikes/sec).
    spike_power_spectrum : NDArray[floating], shape (n_frequencies,)
        Power spectrum of spike train in condition 1.
    homogeneous_poisson_noise : float, default=0
        Homogeneous Poisson noise parameter (beta in reference).
    dt : float, default=1
        Time step size for discretization.

    Returns
    -------
    rate_adjustment_factor : NDArray[floating], shape (n_frequencies,)
        Multiplicative factors to adjust coherence from condition 1 to
        account for firing rate difference.

    Examples
    --------
    >>> import numpy as np
    >>> # Simulate power spectrum and firing rates
    >>> freqs = np.linspace(1, 100, 50)
    >>> power_spec = 1 / (1 + freqs**2)  # 1/f-like spectrum
    >>> rate1, rate2 = 10.0, 15.0  # Different firing rates
    >>>
    >>> adjustment = coherence_rate_adjustment(rate1, rate2, power_spec)
    >>> print(f"Adjustment range: {adjustment.min():.3f} to {adjustment.max():.3f}")

    Notes
    -----
    For spike-spike coherence comparisons, apply this adjustment twice,
    once for each spike train.

    References
    ----------
    .. [1] Aoi, M.C., Lepage, K.Q., Kramer, M.A., and Eden, U.T. (2015).
           Rate-adjusted spike-LFP coherence comparisons from spike-train
           statistics. Journal of Neuroscience Methods 240, 141-153.
    """
    # alpha in [1]
    firing_rate_ratio = firing_rate_condition2 / firing_rate_condition1
    adjusted_firing_rate = (
        (1 / firing_rate_ratio - 1) * firing_rate_condition1
        + homogeneous_poisson_noise / firing_rate_ratio**2
    ) * dt**2
    return 1 / np.sqrt(1 + (adjusted_firing_rate / spike_power_spectrum))


def power_confidence_intervals(
    n_tapers: int,
    power: NDArray[np.floating] | float = 1,
    ci: float = 0.95,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Compute confidence intervals for multitaper power spectrum estimates.

    Uses chi-squared distribution to compute confidence bounds for power
    spectral density estimates from multitaper analysis.

    Parameters
    ----------
    n_tapers : int
        Number of tapers used in multitaper estimation.
    power : NDArray[floating] or float, default=1
        Power spectrum estimates. Can be array of values or scalar.
    ci : float, default=0.95
        Confidence level, must be in range [0.5, 1.0].

    Returns
    -------
    lower_bound : NDArray[floating]
        Lower confidence bounds for power estimates.
    upper_bound : NDArray[floating]
        Upper confidence bounds for power estimates.

    Examples
    --------
    >>> import numpy as np
    >>> # Single power estimate with 5 tapers
    >>> lower, upper = power_confidence_intervals(n_tapers=5, power=1.0, ci=0.95)
    >>> print(f"95% CI: [{lower:.3f}, {upper:.3f}]")
    >>>
    >>> # Multiple power estimates
    >>> power_vals = np.array([0.5, 1.0, 2.0, 5.0])
    >>> lower, upper = power_confidence_intervals(5, power_vals, 0.95)

    References
    ----------
    .. [1] Kramer, M.A., and Eden, U.T. (2016). Case studies in neural
           data analysis: a guide for the practicing neuroscientist (MIT Press).
    """
    upper_bound = 2 * n_tapers / scipy.stats.chi2.ppf(1 - ci, 2 * n_tapers) * power
    lower_bound = 2 * n_tapers / scipy.stats.chi2.ppf(ci, 2 * n_tapers) * power

    return lower_bound, upper_bound


def power_bias(n_observations: int) -> float:
    """Bias of the power spectrum.

    Parameters
    ----------
    n_observations : int
        n_observations is n_tapers * n_trials

    Returns
    -------
    bias : float
    """
    degrees_of_freedom = 2 * n_observations
    return scipy.special.psi(degrees_of_freedom) - np.log(degrees_of_freedom)


def power_variance(n_observations: int) -> float:
    """Compute variance of log-power spectrum estimates.

    Calculates the variance of log-transformed power spectrum estimates
    for use in statistical testing and confidence interval computation.

    Parameters
    ----------
    n_observations : int
        Number of observations used in power estimation (n_tapers * n_trials).

    Returns
    -------
    variance : float
        Variance of log-power estimates.

    Examples
    --------
    >>> var_100 = power_variance(100)
    >>> var_1000 = power_variance(1000)
    >>> print(f"Variance with 100 obs: {var_100:.6f}")
    >>> print(f"Variance with 1000 obs: {var_1000:.6f}")
    Variance with 100 obs: 0.005051
    Variance with 1000 obs: 0.000501
    """
    degrees_of_freedom = 2 * n_observations
    return scipy.special.polygamma(1, degrees_of_freedom)


def power_fisher_z_transform(
    spectrum1: NDArray[np.floating],
    n_obs1: int,
    spectrum2: NDArray[np.floating] | float = 0,
    n_obs2: int = 0,
) -> NDArray[np.floating]:
    """Transform power spectrum estimates for statistical testing.

    Applies log-transformation with bias correction to power spectrum estimates,
    enabling approximately normal distributions for hypothesis testing.
    Can perform one-sample test against baseline or two-sample comparison.

    Parameters
    ----------
    spectrum1 : NDArray[floating], shape (...,)
        Power spectrum estimates from first condition.
    n_obs1 : int
        Number of observations for spectrum1 (n_tapers * n_trials).
    spectrum2 : NDArray[floating] or float, default=0
        Power spectrum estimates from second condition for comparison.
        If 0, performs one-sample test.
    n_obs2 : int, default=0
        Number of observations for spectrum2 (n_tapers * n_trials).

    Returns
    -------
    z_scores : NDArray[floating], shape (...,)
        Z-scores for statistical testing of power differences.

    Examples
    --------
    >>> import numpy as np
    >>> # One-sample test against baseline
    >>> power1 = np.array([0.5, 1.0, 2.0, 0.8])
    >>> z_one = power_fisher_z_transform(power1, n_obs1=100)
    >>>
    >>> # Two-sample comparison
    >>> power2 = np.array([0.3, 0.8, 1.5, 0.9])
    >>> z_two = power_fisher_z_transform(power1, 100, power2, 120)

    Notes
    -----
    Uses bias correction based on sample size to improve the normal
    approximation for statistical testing.
    """
    bias1, bias2 = power_bias(n_obs1), power_bias(n_obs2)
    variance1, variance2 = power_variance(n_obs1), power_variance(n_obs2)

    # Bias correction
    z1 = np.log(spectrum1) - bias1
    z2 = np.log(spectrum2) - bias2

    return (z1 - z2) / np.sqrt(variance1 + variance2)
