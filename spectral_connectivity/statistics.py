"""Statistical procedures for connectivity analysis.

This module provides statistical functions for testing significance of
connectivity measures, including multiple comparison corrections and
transforms for coherence-based measures. Functions support both parametric
and non-parametric approaches for statistical inference in frequency domain
connectivity analysis.
"""

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import scipy.special
import scipy.stats
from numpy.typing import NDArray

from spectral_connectivity.utils import to_numpy


@dataclass(frozen=True)
class JackknifeResult:
    """Leave-one-observation-out estimate and normal-approximation interval."""

    estimate: NDArray[np.floating]
    bias_corrected: NDArray[np.floating]
    standard_error: NDArray[np.floating]
    confidence_interval: tuple[NDArray[np.floating], NDArray[np.floating]]
    n_observations: int
    transformation: str


def _identity(value: NDArray[np.floating]) -> NDArray[np.floating]:
    return value


def _wrap_phase(value: NDArray[np.floating]) -> NDArray[np.floating]:
    return np.angle(np.exp(1j * value))


def _exponential(value: NDArray[np.floating]) -> NDArray[np.floating]:
    return np.exp(value)


def _hyperbolic_tangent(value: NDArray[np.floating]) -> NDArray[np.floating]:
    return np.tanh(value)


def _hyperbolic_tangent_squared(value: NDArray[np.floating]) -> NDArray[np.floating]:
    # Back-transform for magnitude-squared coherence: recover |coherence| =
    # tanh(.), clamp it to the physical [0, 1] range (a lower confidence bound in
    # atanh space can map to a negative magnitude), then square. Clamping before
    # squaring keeps the mapping monotonic, so squaring cannot fold a negative
    # magnitude back above the estimate.
    return np.clip(np.tanh(value), 0, 1) ** 2


def _warn_fisher_boundary(at_boundary: NDArray[np.bool_]) -> None:
    """Warn that a saturated coherence yields a delta-method standard error of 0.

    At ``|coherence| == 1`` the Fisher delta-method derivative is exactly zero,
    so the back-transformed standard error is reported as ``0`` -- implying
    perfect certainty rather than a degenerate boundary. Surface it so the zero
    is not mistaken for a genuinely tight estimate.
    """
    if bool(np.any(at_boundary)):
        warnings.warn(
            f"Fisher jackknife: {int(np.count_nonzero(at_boundary))} value(s) sit "
            "at saturated coherence (|coherence| == 1), where the delta-method "
            "standard error is exactly 0. This reflects a boundary, not perfect "
            "certainty; interpret those standard errors with care.",
            UserWarning,
            stacklevel=3,
        )


def jackknife_confidence_interval(
    estimate: NDArray[np.floating],
    leave_one_out: NDArray[np.floating],
    *,
    confidence_level: float = 0.95,
    transformation: Literal[
        "identity", "log", "fisher", "fisher_squared", "circular"
    ] = "identity",
) -> JackknifeResult:
    """Summarize leave-one-out replicates with a jackknife confidence interval.

    ``leave_one_out`` must have replicate on its first axis. Log transformation
    is appropriate for positive spectra, Fisher's ``atanh`` for magnitude
    coherence in ``[-1, 1]``, ``fisher_squared`` (``atanh(sqrt(.))``) for
    magnitude-squared coherence in ``[0, 1]``, and circular transformation for
    angles in radians. Reported confidence bounds and bias-corrected estimates
    are returned on the original scale; the standard error is converted back
    with the local delta method.
    """
    if not np.isfinite(confidence_level) or not 0 < confidence_level < 1:
        raise ValueError(
            "confidence_level must be finite and strictly between 0 and 1."
        )
    valid_transformations = {"identity", "log", "fisher", "fisher_squared", "circular"}
    if transformation not in valid_transformations:
        raise ValueError(
            "transformation must be 'identity', 'log', 'fisher', "
            f"'fisher_squared', or 'circular'; got {transformation!r}."
        )
    estimate_array = np.asarray(estimate)
    replicates = np.asarray(leave_one_out)
    if np.iscomplexobj(estimate_array) or np.iscomplexobj(replicates):
        raise TypeError("Jackknife intervals require a real-valued measure.")
    if (
        replicates.ndim != estimate_array.ndim + 1
        or replicates.shape[1:] != estimate_array.shape
    ):
        raise ValueError(
            "leave_one_out must have shape (n_observations, *estimate.shape)."
        )
    n_observations = replicates.shape[0]
    if n_observations < 2:
        raise ValueError("Jackknife inference requires at least 2 observations.")

    if transformation == "identity":
        transformed_estimate = estimate_array
        transformed_replicates = replicates
        inverse = _identity
        derivative = np.ones_like(estimate_array)
    elif transformation == "log":
        n_nonpositive = int(
            np.count_nonzero(estimate_array <= 0) + np.count_nonzero(replicates <= 0)
        )
        if n_nonpositive:
            warnings.warn(
                f"Jackknife log transform: {n_nonpositive} non-positive value(s) "
                "map to NaN and propagate to the affected bins' confidence bounds "
                "(a single non-positive replicate makes its whole bin NaN). This "
                "usually indicates a spectral null or a non-power measure.",
                UserWarning,
                stacklevel=2,
            )
        with np.errstate(divide="ignore", invalid="ignore"):
            transformed_estimate = np.where(
                estimate_array > 0, np.log(estimate_array), np.nan
            )
            transformed_replicates = np.where(
                replicates > 0, np.log(replicates), np.nan
            )
        inverse = _exponential
        derivative = estimate_array
    elif transformation == "fisher":
        _warn_fisher_boundary(np.abs(estimate_array) >= 1)
        epsilon = np.finfo(float).eps
        transformed_estimate = np.arctanh(
            np.clip(estimate_array, -1 + epsilon, 1 - epsilon)
        )
        transformed_replicates = np.arctanh(
            np.clip(replicates, -1 + epsilon, 1 - epsilon)
        )
        inverse = _hyperbolic_tangent
        derivative = 1 - np.clip(estimate_array, -1, 1) ** 2
    elif transformation == "fisher_squared":
        # Variance-stabilizing transform for magnitude-squared coherence in
        # [0, 1] (Enochson & Goodman 1965): atanh(sqrt(MSC)) = atanh(|coherence|).
        # Applying Fisher's atanh to the *unsquared* magnitude is the established
        # transform; atanh(MSC) is not. The delta-method derivative back to the
        # MSC scale is d(MSC)/d(atanh(sqrt(MSC))) = 2 * sqrt(MSC) * (1 - MSC).
        _warn_fisher_boundary(estimate_array >= 1)
        epsilon = np.finfo(float).eps
        clipped_estimate = np.clip(estimate_array, 0, 1)
        transformed_estimate = np.arctanh(
            np.clip(np.sqrt(clipped_estimate), 0, 1 - epsilon)
        )
        transformed_replicates = np.arctanh(
            np.clip(np.sqrt(np.clip(replicates, 0, 1)), 0, 1 - epsilon)
        )
        inverse = _hyperbolic_tangent_squared
        derivative = 2 * np.sqrt(clipped_estimate) * (1 - clipped_estimate)
    else:
        # Unwrap every replicate onto the branch nearest the full estimate.
        transformed_estimate = estimate_array
        transformed_replicates = estimate_array + np.angle(
            np.exp(1j * (replicates - estimate_array))
        )
        inverse = _wrap_phase
        derivative = np.ones_like(estimate_array)

    replicate_mean = np.mean(transformed_replicates, axis=0)
    bias_corrected_transformed = (
        n_observations * transformed_estimate - (n_observations - 1) * replicate_mean
    )
    transformed_standard_error = np.sqrt(
        (n_observations - 1)
        / n_observations
        * np.sum((transformed_replicates - replicate_mean) ** 2, axis=0)
    )
    critical_value = scipy.stats.norm.ppf(0.5 + confidence_level / 2)
    lower = inverse(transformed_estimate - critical_value * transformed_standard_error)
    upper = inverse(transformed_estimate + critical_value * transformed_standard_error)
    return JackknifeResult(
        estimate=estimate_array,
        bias_corrected=inverse(bias_corrected_transformed),
        standard_error=np.abs(derivative) * transformed_standard_error,
        confidence_interval=(lower, upper),
        n_observations=n_observations,
        transformation=transformation,
    )


def _require_scipy_false_discovery_control() -> None:
    """Raise an actionable error if SciPy is too old for ``false_discovery_control``.

    ``scipy.stats.false_discovery_control`` was added in SciPy 1.11. The project
    declares ``scipy>=1.11``, but an environment can still resolve an older
    SciPy (a conda pin held back by another package, or an editable install that
    did not re-resolve its dependencies). Without this guard the call fails with
    a bare ``AttributeError`` that never names the cause; mirror the CuPy
    version gate in :mod:`spectral_connectivity.transforms` with a message that
    states the requirement, the installed version, and the fix.
    """
    if not hasattr(scipy.stats, "false_discovery_control"):
        raise RuntimeError(
            f"scipy.stats.false_discovery_control is unavailable: "
            f"spectral_connectivity requires scipy>=1.11 for the "
            f"Benjamini-Hochberg procedure, but scipy {scipy.__version__} is "
            f"installed. Upgrade with `pip install -U 'scipy>=1.11'` (or the "
            f"conda/mamba equivalent)."
        )


def _validate_alpha(alpha: float) -> None:
    """Validate a significance level shared by correction procedures."""
    if (
        isinstance(alpha, (bool, np.bool_))
        or not isinstance(alpha, (int, float, np.integer, np.floating))
        or not np.isfinite(alpha)
        or not 0 < alpha < 1
    ):
        raise ValueError(
            f"alpha must be a finite number strictly between 0 and 1, got {alpha!r}."
        )


def _warn_all_p_values_nonfinite(procedure_name: str) -> None:
    """Warn that a multiple-comparison family is entirely undefined.

    Shared by the BH and Bonferroni procedures: when every p-value is non-finite
    the all-False result is indistinguishable from a valid family with no true
    effects, so it must be surfaced rather than returned silently.
    """
    warnings.warn(
        f"{procedure_name}: every p-value is non-finite (NaN/inf), so no test "
        "is defined and nothing is flagged significant. This usually means "
        "every tested pair involves a dead/zero-power channel. Returning "
        "all-False; check your inputs rather than reading this as 'no "
        "significant effects'.",
        UserWarning,
        stacklevel=3,
    )


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

    Notes
    -----
    Non-finite p-values (``NaN``/``inf``) mark undefined tests — for example a
    coherence pair involving a dead/zero-power channel — and are excluded from
    the family: they neither count toward the number of tests nor tighten the
    threshold for the valid ones, and are returned as ``False``. If *every*
    p-value is non-finite (the whole family is undefined, e.g. every tested pair
    involves a dead channel) a ``UserWarning`` is emitted, because the all-False
    result would otherwise be indistinguishable from a valid family with no true
    effects. All input dimensions are pooled into a single family (the array is
    raveled); the result has the input shape. Delegates to
    :func:`scipy.stats.false_discovery_control` (SciPy >= 1.11; a clear
    ``RuntimeError`` is raised on older SciPy), which rejects finite p-values
    outside ``[0, 1]``.
    """
    _validate_alpha(alpha)
    p_values = np.asarray(p_values, dtype=float)
    is_significant = np.zeros(p_values.shape, dtype=bool)
    valid = np.isfinite(p_values)
    if valid.any():
        _require_scipy_false_discovery_control()
        try:
            adjusted = scipy.stats.false_discovery_control(p_values[valid], method="bh")
        except ValueError as exc:
            # SciPy raises about its own parameter name ("ps"); restate with this
            # function's parameter and a domain hint, since out-of-range values
            # usually mean a non-p-value (e.g. a coherence magnitude) was passed.
            # Name the offending values so the caller can spot the bad input.
            finite_values = p_values[valid]
            out_of_range = finite_values[(finite_values < 0) | (finite_values > 1)]
            if not out_of_range.size:
                # A ValueError not caused by out-of-range p-values: don't
                # misattribute it -- surface the original error unchanged.
                raise
            raise ValueError(
                "p_values must all be in [0, 1]; "
                f"got {out_of_range.size} value(s) outside that range "
                f"(min={out_of_range.min():.3g}, max={out_of_range.max():.3g}). "
                "If these came from a connectivity measure, pass p-values (e.g. "
                "from coherence_significance_pvalue), not coherence magnitudes or "
                "correlations."
            ) from exc
        is_significant[valid] = adjusted <= alpha
    elif p_values.size > 0:
        _warn_all_p_values_nonfinite("Benjamini_Hochberg_procedure")
    return is_significant


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
    _validate_alpha(alpha)
    p_values = np.asarray(p_values, dtype=float)
    is_significant = np.zeros(p_values.shape, dtype=bool)
    valid = np.isfinite(p_values)
    finite_values = p_values[valid]
    out_of_range = finite_values[(finite_values < 0) | (finite_values > 1)]
    if out_of_range.size:
        raise ValueError(
            "p_values must all be in [0, 1]; "
            f"got {out_of_range.size} value(s) outside that range "
            f"(min={out_of_range.min():.3g}, max={out_of_range.max():.3g})."
        )
    if finite_values.size:
        # Undefined (NaN/inf) tests are excluded from the family, matching the BH
        # implementation above, and remain False in the returned mask.
        is_significant[valid] = finite_values <= alpha / finite_values.size
    elif p_values.size > 0:
        _warn_all_p_values_nonfinite("Bonferroni_correction")
    return is_significant


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
    # Note: This function treats all p-values as a single family of tests by
    # flattening the input array. This is the standard approach for multiple
    # comparison correction. An axis parameter could be added in the future if
    # there's a need to correct along specific dimensions independently, but
    # current use cases don't require this functionality.
    try:
        correction = MULTIPLE_COMPARISONS[method]
    except KeyError as exc:
        choices = ", ".join(sorted(MULTIPLE_COMPARISONS))
        raise ValueError(
            f"Unknown multiple-comparisons method {method!r}; choose one of: {choices}."
        ) from exc
    return correction(p_values, alpha=alpha)


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
    >>> # Compare two coherences (the intended two-sample use).
    >>> coh1 = np.array([0.5 + 0.2j, 0.3 + 0.1j])
    >>> coh2 = np.array([0.3 + 0.15j, 0.4 + 0.05j])
    >>> diff_z = coherence_fisher_z_transform(coh1, 100, coh2, 120)
    >>>
    >>> # To test a single coherence against ZERO, use the exact null instead:
    >>> from spectral_connectivity.statistics import coherence_significance_pvalue
    >>> coherence = np.array([0.1 + 0.05j, 0.3 + 0.2j, 0.8 + 0.1j])
    >>> p_values = coherence_significance_pvalue(coherence, n_observations=100)

    Notes
    -----
    The transformation uses bias correction based on the number of observations
    to improve the normal approximation for small sample sizes.

    The Fisher approximation is derived around a non-zero operating point. To
    test a single coherence against **zero** coherence, prefer
    :func:`coherence_significance_pvalue`, which uses the exact null
    distribution; the Fisher one-sample form (the ``coherency2=0, n_obs2=0``
    default) is miscalibrated at that boundary and over-rejects the null.
    """
    # coherence_bias evaluates 1 / (2 * (n_obs - 1)); n_obs == 1 divides by zero,
    # and non-finite/fractional counts give NaN with a runtime warning.
    if not np.isfinite(n_obs1) or int(n_obs1) != n_obs1 or n_obs1 < 2:
        raise ValueError(f"n_obs1 must be a finite integer >= 2, got {n_obs1}.")
    if not np.isfinite(n_obs2) or int(n_obs2) != n_obs2 or (n_obs2 != 0 and n_obs2 < 2):
        raise ValueError(
            f"n_obs2 must be a finite integer equal to 0 (one-sample test) or "
            f">= 2, got {n_obs2}."
        )
    coherence_magnitude1 = np.abs(coherency1)
    coherence_magnitude1[coherence_magnitude1 >= 1] = 1 - np.finfo(float).eps

    coherence_magnitude2 = np.array(np.abs(coherency2))
    coherence_magnitude2[coherence_magnitude2 >= 1] = 1 - np.finfo(float).eps

    bias1 = coherence_bias(n_obs1)
    # When there is no second sample (n_obs2 == 0) the comparison is against a
    # fixed null value (coherency2, default 0), which carries no estimation
    # bias or sampling variance. Evaluating ``coherence_bias(0) = -0.5`` here
    # would instead make ``sqrt(bias1 + bias2)`` negative and return NaN.
    bias2 = coherence_bias(n_obs2) if n_obs2 else 0.0

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
    >>> np.round(p_vals, 3).tolist()
    [0.975, 0.5, 0.025, 0.005]

    Notes
    -----
    This function handles both NumPy and CuPy arrays automatically. Inputs cross
    the package's explicit host boundary before SciPy computes the CDF.
    """
    # Use the survival function (sf = 1 - cdf) rather than ``1 - cdf`` so that
    # far-tail p-values keep full precision: ``1 - norm.cdf(8.3)`` underflows to
    # exactly 0, while ``norm.sf(8.3)`` returns ~5.2e-17.
    return scipy.stats.norm.sf(to_numpy(data), loc=mean, scale=std_deviation)


def coherence_significance_pvalue(
    coherency: NDArray[np.complexfloating],
    n_observations: int,
) -> NDArray[np.floating]:
    """P-value for testing squared coherence magnitude against zero.

    Tests the null hypothesis that the true coherence is zero. Under this null,
    for ``n`` independent complex-Gaussian observations the magnitude-squared
    coherence estimate follows a Beta(1, n - 1) distribution, so the upper-tail
    probability is ``P(|C|^2 >= c) = (1 - c)^(n - 1)``.

    This exact boundary distribution should be used instead of the Fisher
    z-transform (:func:`coherence_fisher_z_transform`) when testing against zero
    coherence: the Fisher approximation is derived around a non-zero operating
    point and is badly miscalibrated at ``coherence == 0`` (it over-rejects the
    null by 3-4x, e.g. ~16-22% actual rejection at a nominal 5% level).

    Parameters
    ----------
    coherency : NDArray[complexfloating], shape (...,)
        Complex coherency values between signals.
    n_observations : int
        Number of independent observations used to estimate the coherency
        (n_tapers * n_trials).

    Returns
    -------
    p_values : NDArray[floating], shape (...,)
        Upper-tail p-values for the test of zero coherence.

    References
    ----------
    .. [1] Hannan, E. J. (1970). Multiple Time Series. Wiley. (Null
           distribution of magnitude-squared coherence.)
    .. [2] Thomson, D. J., & Chave, A. D. (1991). Jackknifed error estimates
           for spectra, coherences, and transfer functions. In Advances in
           Spectrum Analysis and Array Processing.
    """
    if (
        not np.isfinite(n_observations)
        or int(n_observations) != n_observations
        or n_observations < 2
    ):
        raise ValueError(
            f"n_observations must be a finite integer >= 2 for the "
            f"zero-coherence null distribution (Beta(1, n_observations - 1)), "
            f"got {n_observations}. With fewer observations the coherence "
            f"estimate is degenerate; a non-finite or non-integer count gives a "
            f"NaN or degenerate p-value."
        )
    magnitude_squared_coherence = np.clip(np.abs(coherency) ** 2, 0.0, 1.0)
    return (1.0 - magnitude_squared_coherence) ** (n_observations - 1)


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
    >>> print(f"Bias with 100 obs: {coherence_bias(100):.6f}")
    Bias with 100 obs: 0.005051
    >>> print(f"Bias with 1000 obs: {coherence_bias(1000):.6f}")
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
    >>> rate1, rate2 = 15.0, 10.0  # firing rate decreases in condition 2
    >>> adjustment = coherence_rate_adjustment(rate1, rate2, power_spec)
    >>> print(f"Adjustment range: {adjustment.min():.3f} to {adjustment.max():.3f}")
    Adjustment range: 0.004 to 0.250

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
    if not np.isfinite(firing_rate_condition1) or firing_rate_condition1 <= 0:
        raise ValueError(
            f"firing_rate_condition1 must be a finite positive number, got "
            f"{firing_rate_condition1}."
        )
    if not np.isfinite(firing_rate_condition2) or firing_rate_condition2 <= 0:
        raise ValueError(
            f"firing_rate_condition2 must be a finite positive number, got "
            f"{firing_rate_condition2}."
        )
    # alpha in [1]
    firing_rate_ratio = firing_rate_condition2 / firing_rate_condition1
    adjusted_firing_rate = (
        (1 / firing_rate_ratio - 1) * firing_rate_condition1
        + homogeneous_poisson_noise / firing_rate_ratio**2
    ) * dt**2
    # Spike power spectral density is non-negative by definition; a non-positive
    # value is invalid input (not just undefined), so it is masked below.
    spike_power = np.asarray(spike_power_spectrum, dtype=float)
    # Compute the argument and the adjustment under a scoped errstate: zero
    # spike power makes the division non-finite and a non-positive argument
    # makes the sqrt undefined. Both are handled explicitly below rather than
    # leaking a RuntimeWarning or a silent 0/inf.
    with np.errstate(invalid="ignore", divide="ignore"):
        argument = 1 + (adjusted_firing_rate / spike_power)
        adjustment = 1 / np.sqrt(argument)
    # The adjustment is undefined wherever the spike power is non-positive
    # (invalid input) or the argument is not strictly positive (a large rate
    # increase relative to the spike power). Return NaN for those entries.
    undefined = (spike_power <= 0) | ~(np.isfinite(argument) & (argument > 0))
    if np.any(undefined):
        warnings.warn(
            "coherence_rate_adjustment is undefined at some frequencies "
            "(non-positive spike power, or 1 + adjusted_rate / "
            "spike_power_spectrum <= 0); those entries are returned as NaN. "
            "This typically happens for a large firing-rate increase relative "
            "to the spike power, or invalid (non-positive) spike power.",
            UserWarning,
            stacklevel=2,
        )
    return np.where(undefined, np.nan, adjustment)


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
        Confidence level, must be in range [0.5, 1.0).

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
    95% CI: [0.488, 3.080]
    >>> # Multiple power estimates
    >>> power_vals = np.array([0.5, 1.0, 2.0, 5.0])
    >>> lower, upper = power_confidence_intervals(5, power_vals, 0.95)
    >>> np.round(lower, 3).tolist()
    [0.244, 0.488, 0.976, 2.441]

    References
    ----------
    .. [1] Kramer, M.A., and Eden, U.T. (2016). Case studies in neural
           data analysis: a guide for the practicing neuroscientist (MIT Press).
    """
    if not np.isfinite(n_tapers) or int(n_tapers) != n_tapers or n_tapers < 1:
        raise ValueError(
            f"n_tapers must be a finite positive integer, got {n_tapers}. It sets "
            f"the chi-squared degrees of freedom (2 * n_tapers); a non-positive, "
            f"non-finite, or fractional value gives NaN or meaningless intervals."
        )
    if not 0.5 <= ci < 1.0:
        raise ValueError(
            f"Confidence level `ci` must be in the range [0.5, 1.0), got {ci}. "
            "`ci` is the total probability mass inside the interval (e.g. 0.95 "
            "for a 95% confidence interval)."
        )
    # Power spectral density is non-negative; a negative value scales the bounds
    # negative and reversed (e.g. power=-1 -> ~(-0.488, -3.080)), and a
    # non-finite value gives NaN/Inf bounds. Reject both explicitly.
    power_values = np.asarray(power, dtype=float)
    if not np.all(np.isfinite(power_values)) or np.any(power_values < 0):
        raise ValueError(
            "power must be finite and non-negative (it is a power spectral "
            "density); got a non-finite or negative value."
        )
    # A two-sided (1 - alpha) interval splits the tail mass alpha = 1 - ci
    # evenly between the two tails, so the chi-squared quantiles are taken at
    # alpha / 2 and 1 - alpha / 2. Using the full alpha on each tail (the
    # previous behavior) produces an interval with coverage 2 * ci - 1 (a
    # requested 95% interval only covered ~90%).
    degrees_of_freedom = 2 * n_tapers
    alpha = 1 - ci
    lower_bound = (
        degrees_of_freedom
        / scipy.stats.chi2.ppf(1 - alpha / 2, degrees_of_freedom)
        * power
    )
    upper_bound = (
        degrees_of_freedom / scipy.stats.chi2.ppf(alpha / 2, degrees_of_freedom) * power
    )

    return lower_bound, upper_bound


def power_bias(n_observations: int) -> float:
    """Bias of the log power spectrum.

    A multitaper power estimate satisfies ``S_hat / S ~ chi2_nu / nu`` with
    ``nu = 2 * n_observations`` degrees of freedom. Writing ``chi2_nu`` as
    ``2 * Gamma(nu / 2)`` gives ``E[log(S_hat / S)] = psi(nu / 2) - log(nu / 2)``,
    i.e. the digamma/log are evaluated at the chi-squared shape parameter
    ``nu / 2 = n_observations`` (not at ``nu``).

    Parameters
    ----------
    n_observations : int
        n_observations is n_tapers * n_trials

    Returns
    -------
    bias : float

    Examples
    --------
    >>> print(f"Bias with 100 obs: {power_bias(100):.6f}")
    Bias with 100 obs: -0.005008
    >>> print(f"Bias with 1000 obs: {power_bias(1000):.6f}")
    Bias with 1000 obs: -0.000500
    """
    return scipy.special.psi(n_observations) - np.log(n_observations)


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
    Variance with 100 obs: 0.010050
    >>> print(f"Variance with 1000 obs: {var_1000:.6f}")
    Variance with 1000 obs: 0.001001

    Notes
    -----
    With ``S_hat / S ~ chi2_nu / nu`` and ``nu = 2 * n_observations``, the
    variance of ``log(S_hat)`` is the trigamma function evaluated at the
    chi-squared shape parameter ``nu / 2 = n_observations``.
    """
    return scipy.special.polygamma(1, n_observations)


def power_fisher_z_transform(
    spectrum1: NDArray[np.floating],
    n_obs1: int,
    spectrum2: NDArray[np.floating] | float = 1.0,
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
    spectrum2 : NDArray[floating] or float, default=1.0
        For a two-sample comparison (``n_obs2 > 0``), the power spectrum
        estimates from the second condition. For a one-sample test
        (``n_obs2 == 0``), a fixed positive baseline power against which
        ``spectrum1`` is compared; must be > 0 because the test operates on
        ``log(power)``.
    n_obs2 : int, default=0
        Number of observations for spectrum2 (n_tapers * n_trials). If 0,
        performs a one-sample test of ``spectrum1`` against the fixed baseline
        ``spectrum2``.

    Returns
    -------
    z_scores : NDArray[floating], shape (...,)
        Z-scores for statistical testing of power differences.

    Examples
    --------
    >>> import numpy as np
    >>> # One-sample test against a fixed baseline power of 1.0
    >>> power1 = np.array([0.5, 1.0, 2.0, 0.8])
    >>> z_one = power_fisher_z_transform(power1, n_obs1=100, spectrum2=1.0)
    >>>
    >>> # Two-sample comparison
    >>> power2 = np.array([0.3, 0.8, 1.5, 0.9])
    >>> z_two = power_fisher_z_transform(power1, 100, power2, 120)

    Notes
    -----
    Uses bias correction based on sample size to improve the normal
    approximation for statistical testing.
    """
    # Observation counts must be valid: power_bias/power_variance evaluate
    # digamma/trigamma at n_obs, which have poles at 0; non-finite or fractional
    # counts would silently produce NaN z-scores.
    if not np.isfinite(n_obs1) or int(n_obs1) != n_obs1 or n_obs1 < 1:
        raise ValueError(f"n_obs1 must be a finite integer >= 1, got {n_obs1}.")
    if not np.isfinite(n_obs2) or int(n_obs2) != n_obs2 or n_obs2 < 0:
        raise ValueError(
            f"n_obs2 must be a finite integer >= 0 (0 for a one-sample test), "
            f"got {n_obs2}."
        )
    # The test operates on log(power); non-finite or non-positive inputs would
    # produce silent -inf/nan z-scores. Fail loudly instead.
    spectrum1_values = np.asarray(spectrum1, dtype=float)
    if not np.all(np.isfinite(spectrum1_values)) or np.any(spectrum1_values <= 0):
        raise ValueError(
            "spectrum1 must be finite and strictly positive (the test uses "
            "log(power)); got a non-finite or <= 0 value."
        )
    spectrum2_values = np.asarray(spectrum2, dtype=float)
    if not np.all(np.isfinite(spectrum2_values)) or np.any(spectrum2_values <= 0):
        raise ValueError(
            "spectrum2 must be finite and strictly positive (the test uses "
            "log(power)); got a non-finite or <= 0 value. For a one-sample test, "
            "pass a positive baseline power as spectrum2 (default 1.0)."
        )

    bias1 = power_bias(n_obs1)
    variance1 = power_variance(n_obs1)
    # When there is no second sample (n_obs2 == 0) spectrum2 is a fixed
    # baseline that carries no estimation bias or sampling variance. Evaluating
    # power_bias(0)/power_variance(0) would instead hit the digamma/trigamma
    # poles and return NaN.
    if n_obs2:
        bias2 = power_bias(n_obs2)
        variance2 = power_variance(n_obs2)
    else:
        bias2 = 0.0
        variance2 = 0.0

    # Bias correction
    z1 = np.log(spectrum1) - bias1
    z2 = np.log(spectrum2) - bias2

    return (z1 - z2) / np.sqrt(variance1 + variance2)
