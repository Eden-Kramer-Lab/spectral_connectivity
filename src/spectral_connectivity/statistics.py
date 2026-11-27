"""Statistical procedures for connectivity analysis.

This module provides statistical functions for testing significance of
connectivity measures, including multiple comparison corrections and
transforms for coherence-based measures. Functions support both parametric
and non-parametric approaches for statistical inference in frequency domain
connectivity analysis.
"""

import math
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import scipy.special
import scipy.stats
from numpy.typing import NDArray

from spectral_connectivity.utils import (
    _validated_flag,
    both_spellings_message,
    stacklevel_outside_package,
    to_numpy,
    warn_deprecated,
)

# The public API: what the API reference documents (autosummary_ignore_module_all
# in docs/conf.py) and ``import *`` imports.
__all__ = [
    "MULTIPLE_COMPARISONS",
    "Benjamini_Hochberg_procedure",
    "Bonferroni_correction",
    "JackknifeResult",
    "SurrogateTestResult",
    "adjust_for_multiple_comparisons",
    "coherence_bias",
    "coherence_fisher_z_transform",
    "coherence_rate_adjustment",
    "coherence_significance_pvalue",
    "get_normal_distribution_p_values",
    "jackknife_confidence_interval",
    "power_bias",
    "power_confidence_intervals",
    "power_fisher_z_transform",
    "power_variance",
]


@dataclass(frozen=True)
class JackknifeResult:
    """Leave-one-observation-out estimate and Student-t jackknife interval.

    Every array attribute but ``replicates`` has the shape of the underlying
    measure, e.g. ``(n_time, n_nonnegative_frequencies, n_signals, n_signals)``
    for a pairwise connectivity measure, or ``(n_time, n_bands, n_signals,
    n_signals)`` when :meth:`Connectivity.jackknife` reduces it to
    ``frequency_bands``.

    Attributes
    ----------
    estimate : array
        The full-sample estimate on the original scale.
    bias_corrected : array
        Jackknife bias-corrected estimate on the original scale.
    standard_error : array
        Standard error on the original scale (delta method through the
        transformation).
    confidence_interval : tuple of (lower, upper) arrays
        Confidence bounds on the original scale, formed on the transformed
        scale as ``estimate -/+ t * standard_error`` with the Student t
        critical value on ``n_observations - 1`` degrees of freedom. For the
        ``"circular"`` transformation the bounds are wrapped to ``(-pi, pi]``:
        ``lower > upper`` means the interval crosses ``+/-pi`` and is
        ``[lower, pi] U (-pi, upper]``, and a bin whose half-width reaches
        ``pi`` (the whole circle, phase unresolved) is reported as
        ``(-pi, pi)``.
    n_observations : int or array of int
        Number of leave-one-out replicates; the interval's critical value has
        ``n_observations - 1`` degrees of freedom. From
        :meth:`Connectivity.jackknife` it is the number of averaged
        observations (``n_trials * n_tapers`` for ``trials_tapers``) with
        ``leave_out="observation"`` and the number of trials with
        ``leave_out="trial"``, so it is not always
        ``Connectivity.n_observations``. Where observation weights drop
        different observations in different bins (see
        ``Connectivity.n_valid_observations``), one count per bin, an array
        that broadcasts against the estimate.
    transformation : str
        Variance-stabilizing transformation used for the interval.
    replicates : array, shape (n_replicates, *estimate.shape), or None
        Use replicates from ``leave_out="trial"`` for any statistic that
        combines frequencies, unless its coverage with other units has been
        checked: different tapers' coefficients are correlated across nearby
        frequencies, so leave-one-taper-out replicates of a band mean give too
        narrow an interval (see :meth:`Connectivity.jackknife`).

        The leave-one-out replicates the interval summarizes, on the scale of
        ``estimate`` (untransformed), one per left-out unit on the first axis.
        ``None`` from :meth:`Connectivity.jackknife` unless it was called with
        ``keep_replicates=True``; then one per observation
        (``leave_out="observation"``; for ``trials_tapers``, observation ``k``
        is trial ``k // n_tapers``, taper ``k % n_tapers``) or per trial
        (``leave_out="trial"``), in order, without the units that observation
        weights drop from every bin, and band-reduced with
        ``frequency_bands``. A replicate is NaN in a bin that never had the
        unit it leaves out (see ``Connectivity.n_valid_observations``), since
        leaving that unit out removes nothing there;
        :func:`jackknife_confidence_interval` skips it, so passing
        ``estimate`` and ``replicates`` back reproduces the interval. The
        array is the one the interval was computed from, ``n_replicates``
        times the memory of ``estimate``. From
        :func:`jackknife_confidence_interval`, ``leave_one_out`` itself (not a
        copy).

        For an interval on another statistic of the measure (a difference
        between bands, a ratio of two pairs), compute it from ``estimate``
        and from each replicate and pass both to
        :func:`jackknife_confidence_interval`.
    frequency_bands : dict of str to (float, float), or None
        The bands of :meth:`Connectivity.jackknife`'s ``frequency_bands``, in
        the order of the estimate's band axis (``list(frequency_bands)`` names
        its entries); ``None`` when the frequencies were kept, and from
        :func:`jackknife_confidence_interval`.
    leave_out : {"observation", "trial"} or None
        The unit :meth:`Connectivity.jackknife` left out; ``None`` from
        :func:`jackknife_confidence_interval`, which is given the replicates.
    """

    estimate: NDArray[np.floating]
    bias_corrected: NDArray[np.floating]
    standard_error: NDArray[np.floating]
    confidence_interval: tuple[NDArray[np.floating], NDArray[np.floating]]
    n_observations: int | NDArray[np.integer]
    transformation: str
    replicates: NDArray[np.floating] | None = None
    frequency_bands: Mapping[str, tuple[float, float]] | None = None
    leave_out: str | None = None


@dataclass(frozen=True)
class SurrogateTestResult:
    """Monte Carlo surrogate test of a connectivity measure.

    Returned by :meth:`Connectivity.surrogate_test`. ``estimate``,
    ``p_value`` and ``fwer_p_value`` have the shape of the measure, e.g.
    ``(n_time, n_nonnegative_frequencies, n_signals, n_signals)`` for a
    pairwise connectivity measure. An element is tested unless it is on the
    diagonal (a signal with itself), its estimate is not finite, or its
    frequency is within ``half_bandwidth`` of 0 Hz or Nyquist (for
    ``kind="phase_rotation"``) or its coefficients are real, which
    conjugation leaves unchanged (for ``kind="conjugation"``: the
    frequencies of :attr:`Connectivity.real_valued_frequencies
    <spectral_connectivity.Connectivity.real_valued_frequencies>`, and 0 Hz
    and Nyquist of a real signal); both p-values are NaN where it is not, and
    it is in no family's maximum.

    Attributes
    ----------
    estimate : array
        The measure of the original coefficients.
    p_value : array
        Uncorrected Monte Carlo p-value of each element, ``(1 + b) /
        (n_surrogates + 1)``, where ``b`` counts the surrogates whose
        statistic at that element is at least the observed one (see
        ``alternative``). It is the test of one element chosen in advance (one
        pair at one frequency); to say where among many elements the measure
        is significant, report ``fwer_p_value``, or correct ``p_value`` with
        :func:`spectral_connectivity.statistics.adjust_for_multiple_comparisons`
        for a false discovery rate.
    fwer_p_value : array
        Family-wise adjusted p-value from the maximum statistic (Nichols and
        Holmes [1]_): ``(1 + m) / (n_surrogates + 1)``, where ``m`` counts the
        surrogates whose largest statistic over the element's family (the
        tested elements along ``fwer_axis``) is at least the observed
        statistic. Rejecting every element with ``fwer_p_value <= alpha``
        rejects a true null anywhere in the family with probability at most
        ``alpha``. This is the p-value to report for where an effect is.
    null_maxima : array, shape (n_surrogates, ...)
        The largest statistic of each surrogate in each family: the axes of
        ``estimate`` not in ``fwer_axis``, in order, follow the surrogate
        axis (``(n_surrogates,)`` when every element is one family). Its
        ``1 - alpha`` quantile along axis 0 is the family's critical value at
        level ``alpha``, on the scale of the statistic (the value for
        ``"greater"``, its negative for ``"less"``, its magnitude for
        ``"two-sided"``). A family with no tested element holds ``-inf``.
    n_surrogates : int
        Number of surrogate datasets.
    kind : str
        The surrogate construction, which sets the null hypothesis (see
        :meth:`Connectivity.surrogate_test`).
    alternative : str
        ``"greater"``, ``"less"``, or ``"two-sided"``: the alternative used,
        with ``"auto"`` resolved.
    fwer_axis : tuple of int
        The axes of ``estimate`` whose tested elements form one family for
        ``fwer_p_value``, normalized: non-negative, in the order given, and
        every axis when ``fwer_axis=None`` was passed.

    References
    ----------
    .. [1] Nichols, T. E., and Holmes, A. P. (2002). Nonparametric permutation
           tests for functional neuroimaging: a primer with examples. Human
           Brain Mapping 15, 1-25.
    """

    estimate: NDArray[np.floating]
    p_value: NDArray[np.floating]
    fwer_p_value: NDArray[np.floating]
    null_maxima: NDArray[np.floating]
    n_surrogates: int
    kind: str
    alternative: str
    fwer_axis: tuple[int, ...]


# A surrogate's statistic counts as at least the observed one when it is no
# more than this far below it. Surrogates equal to the data in exact arithmetic
# (the identity permutation, a conjugation that flips nothing) reproduce the
# observed value only to rounding, and the directed measures only to the
# minimum-phase factorization's tolerance (up to 6e-8 relative for pairwise
# spectral Granger computed alone or among surrogates); a tie must count.
_TIE_RELATIVE_TOLERANCE = 1e-6
_TIE_ABSOLUTE_TOLERANCE = 1e-12
# Elements compared at once when counting surrogate family maxima.
_FAMILY_COMPARISON_CHUNK_ELEMENTS = 2**22


def _surrogate_statistic(
    values: NDArray[np.floating], alternative: str
) -> NDArray[np.floating]:
    """The test statistic of ``values``: large values are evidence against the null."""
    if alternative == "greater":
        return values
    if alternative == "less":
        return -values
    return np.abs(values)


class _SurrogateTally:
    """Exceedance counts and family maxima of surrogate statistics.

    Keeps memory at the size of one result plus ``n_surrogates`` family
    maxima, rather than every surrogate's result.

    Parameters
    ----------
    statistic : array
        Observed statistic (see :func:`_surrogate_statistic`).
    tested : array of bool, shape like ``statistic``
        The elements tested; the others get NaN p-values and are left out of
        the family maxima.
    family_axes : tuple of int
        Non-negative axes of ``statistic`` whose elements form one family.
    n_surrogates : int
        Number of surrogate statistics :meth:`add` will receive in total.
    """

    def __init__(
        self,
        statistic: NDArray[np.floating],
        tested: NDArray[np.bool_],
        family_axes: tuple[int, ...],
        n_surrogates: int,
    ) -> None:
        self._tested = tested
        self._threshold = statistic - (
            _TIE_RELATIVE_TOLERANCE * np.abs(statistic) + _TIE_ABSOLUTE_TOLERANCE
        )
        other_axes = tuple(axis for axis in range(statistic.ndim) if axis not in family_axes)
        # Each family's elements last, flattened: (*other_shape, n_family_elements).
        self._order = (*other_axes, *family_axes)
        self._other_shape = tuple(statistic.shape[axis] for axis in other_axes)
        self._counts = np.zeros(statistic.shape, dtype=np.int64)
        self.null_maxima = np.empty((n_surrogates, *self._other_shape))
        self._n_added = 0
        # Surrogates with a NaN at a tested element, and how many such values.
        self.n_surrogates_with_nan = 0
        self.n_nan_values = 0

    def add(self, null_statistic: NDArray[np.floating]) -> None:
        """Count one surrogate's statistic, shaped like the observed one.

        A NaN surrogate value at a tested element counts as exceeding the
        observed statistic (and makes its family maximum ``inf``): a surrogate
        the measure could not evaluate is not evidence against the null.
        """
        is_nan = np.isnan(null_statistic)
        n_nan = int(np.count_nonzero(is_nan & self._tested))
        if n_nan:
            self.n_surrogates_with_nan += 1
            self.n_nan_values += n_nan
        self._counts += (null_statistic >= self._threshold) | is_nan
        family_values = np.where(
            self._tested, np.where(is_nan, np.inf, null_statistic), -np.inf
        )
        self.null_maxima[self._n_added] = (
            np.transpose(family_values, self._order)
            .reshape(*self._other_shape, -1)
            .max(axis=-1, initial=-np.inf)
        )
        self._n_added += 1

    def p_values(self) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Element-wise and family-wise adjusted p-values, NaN where untested."""
        n_surrogates = self.null_maxima.shape[0]
        p_value = (1 + self._counts) / (n_surrogates + 1)
        threshold = np.transpose(self._threshold, self._order)
        flat_threshold = threshold.reshape(*self._other_shape, -1)
        family_counts = np.zeros(flat_threshold.shape, dtype=np.int64)
        chunk = max(1, _FAMILY_COMPARISON_CHUNK_ELEMENTS // max(flat_threshold.size, 1))
        for start in range(0, n_surrogates, chunk):
            maxima = self.null_maxima[start : start + chunk, ..., np.newaxis]
            family_counts += np.count_nonzero(maxima >= flat_threshold, axis=0)
        fwer_p_value = np.transpose(
            ((1 + family_counts) / (n_surrogates + 1)).reshape(threshold.shape),
            np.argsort(self._order),
        )
        return (
            np.where(self._tested, p_value, np.nan),
            np.where(self._tested, fwer_p_value, np.nan),
        )


def _identity(value: NDArray[np.floating]) -> NDArray[np.floating]:
    return value


def _wrap_phase(value: NDArray[np.floating]) -> NDArray[np.floating]:
    return np.angle(np.exp(1j * value))


def _pin_whole_circle_interval(
    half_width: NDArray[np.floating],
    lower: NDArray[np.floating],
    upper: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Report a circular interval with half-width >= pi as the whole circle.

    The circular standard error is a linear standard error of the unwrapped
    replicates and is unbounded, but wrapping bounds that are more than 2*pi
    apart makes a whole-circle interval look narrow (a half-width of 11 rad
    wraps to about (-1.65, 1.65)). Pin such bins to ``(-pi, pi)`` and warn that
    the phase is not resolved there.
    """
    whole_circle = half_width >= np.pi
    if bool(np.any(whole_circle)):
        warnings.warn(
            f"Jackknife circular interval: {int(np.count_nonzero(whole_circle))} "
            "value(s) have a half-width of at least pi rad, so the interval covers "
            "the whole circle and the phase is not resolved there. Those bounds are "
            "reported as (-pi, pi).",
            UserWarning,
            stacklevel=3,
        )
        lower = np.where(whole_circle, -np.pi, lower)
        upper = np.where(whole_circle, np.pi, upper)
    return lower, upper


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
    """Warn that saturated estimates have a Fisher standard error of 0.

    At ``|coherence| == 1`` the delta-method derivative is exactly zero, so the
    back-transformed standard error is reported as ``0`` and the interval
    degenerates, implying perfect certainty rather than a boundary. Surface it
    so the zero is not mistaken for a genuinely tight estimate.
    """
    if bool(np.any(at_boundary)):
        warnings.warn(
            f"Fisher jackknife: {int(np.count_nonzero(at_boundary))} value(s) sit "
            "at saturated coherence (|coherence| == 1 to machine precision), where "
            "the transform's derivative vanishes, so the delta-method standard "
            "error is exactly 0 and the interval degenerates there. This reflects "
            "a boundary, not perfect certainty; interpret those values with care.",
            UserWarning,
            stacklevel=3,
        )


def _validate_confidence_level(confidence_level: float) -> None:
    """Reject a ``confidence_level`` that is not a probability strictly inside (0, 1).

    A real number is accepted as a Python or NumPy scalar or a 0-d array; a
    boolean is not a probability.
    """
    value = np.asarray(confidence_level)
    if (
        value.ndim != 0
        or value.dtype.kind not in "iuf"
        or not np.isfinite(value)
        or not 0 < value < 1
    ):
        msg = (
            "confidence_level must be finite and strictly between 0 and 1, got "
            f"{confidence_level!r}. It is the probability mass inside the "
            "interval, e.g. 0.95 for a 95% interval."
        )
        raise ValueError(msg)


def jackknife_confidence_interval(
    estimate: NDArray[np.floating],
    leave_one_out: NDArray[np.floating],
    *,
    confidence_level: float = 0.95,
    transformation: Literal[
        "identity", "log", "fisher", "fisher_squared", "circular"
    ] = "identity",
    _saturated_by_construction: NDArray[np.bool_] | bool = False,
    _replicate_is_valid: NDArray[np.bool_] | None = None,
) -> JackknifeResult:
    """Summarize leave-one-out replicates with a jackknife confidence interval.

    Parameters
    ----------
    estimate : array, shape (...)
        Full-sample estimate of a real-valued measure.
    leave_one_out : array, shape (n_observations, ...)
        Replicates with one observation omitted each, stacked on the first
        axis; the remaining axes must match ``estimate``.
    confidence_level : float, default=0.95
        Two-sided coverage of the interval, in (0, 1). The critical value is
        the Student t quantile with ``n_observations - 1`` degrees of freedom,
        the standard choice for a jackknife variance estimated from
        ``n_observations`` replicates [1]_ [2]_; a normal quantile under-covers
        when there are few replicates (about 0.88 instead of 0.95 at five).
    transformation : {"identity", "log", "fisher", "fisher_squared", "circular"}
        Scale on which the interval is formed. Log is appropriate for positive
        spectra, Fisher's ``atanh`` for magnitude coherence in ``[-1, 1]``,
        ``fisher_squared`` (``atanh(sqrt(.))``) for magnitude-squared coherence
        in ``[0, 1]``, and circular for angles in radians.

    Returns
    -------
    JackknifeResult
        Estimate, bias-corrected estimate, standard error, and confidence
        bounds, all on the original scale and with the shape of ``estimate``.
        The standard error is converted back with the local delta method.
        ``replicates`` is ``leave_one_out``.

    Notes
    -----
    For ``"circular"`` the replicates are unwrapped onto the branch nearest the
    estimate, the interval is formed on that linear scale, and the bounds are
    wrapped back to ``(-pi, pi]``. A bound that wraps past ``+/-pi`` leaves
    ``lower > upper``; the interval is then ``[lower, pi] U (-pi, upper]``.
    The circular standard error itself is not wrapped, so it can exceed
    ``pi``; when the half-width is at least ``pi`` the interval covers the
    whole circle, the phase is not resolved, and the bounds are reported as
    ``(-pi, pi)`` with a ``UserWarning``.

    The interval describes the size of a measure; it is not a test that the
    measure differs from 0. In Monte Carlo simulation with independent
    complex-Gaussian observations and 95% nominal intervals, ``fisher_squared``
    covers the true magnitude-squared coherence 94-95% of the time when the
    true ``|coherence|`` is 0.3-0.8, but when the true coherence is 0 it
    excludes 0 in 11-15% of datasets at 5 to 100 observations, and more
    observations do not help. (An estimate of exactly 0, which requires an
    exactly cancelling cross-spectrum, gets a standard error of 0, since the
    delta-method derivative vanishes there.) To test for nonzero coherence use
    the exact zero-coherence null, :func:`coherence_significance_pvalue`.
    ``"circular"`` intervals under-cover when the phase is poorly determined
    (77-88% coverage at a true ``|coherence|`` of 0.1). See
    :meth:`spectral_connectivity.connectivity.Connectivity.jackknife` for the
    per-measure figures.

    References
    ----------
    .. [1] Thomson, D. J., & Chave, A. D. (1991). Jackknifed error estimates
           for spectra, coherences, and transfer functions. In Advances in
           Spectrum Analysis and Array Processing.
    .. [2] Efron, B., & Tibshirani, R. J. (1993). An Introduction to the
           Bootstrap. Chapman & Hall.
    """
    _validate_confidence_level(confidence_level)
    valid_transformations = {"identity", "log", "fisher", "fisher_squared", "circular"}
    if transformation not in valid_transformations:
        msg = (
            "transformation must be 'identity', 'log', 'fisher', "
            f"'fisher_squared', or 'circular'; got {transformation!r}."
        )
        raise ValueError(msg)
    estimate_array = np.asarray(estimate)
    replicates = np.asarray(leave_one_out)
    if np.iscomplexobj(estimate_array) or np.iscomplexobj(replicates):
        msg = "Jackknife intervals require a real-valued measure."
        raise TypeError(msg)
    if (
        replicates.ndim != estimate_array.ndim + 1
        or replicates.shape[1:] != estimate_array.shape
    ):
        msg = "leave_one_out must have shape (n_observations, *estimate.shape)."
        raise ValueError(msg)
    n_observations = replicates.shape[0]
    if n_observations < 2:
        msg = "Jackknife inference requires at least 2 observations."
        raise ValueError(msg)

    # _saturated_by_construction is private to Connectivity.jackknife: it marks
    # entries that sit at |coherence| == 1 by definition (phase_locking_value's
    # diagonal), which the saturation warning must not count.
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
            transformed_estimate = np.where(estimate_array > 0, np.log(estimate_array), np.nan)
            transformed_replicates = np.where(replicates > 0, np.log(replicates), np.nan)
        inverse = _exponential
        derivative = estimate_array
    elif transformation == "fisher":
        epsilon = np.finfo(float).eps
        # Count every value the atanh clip below pins: a saturated estimate often
        # lands a few ulp below 1 rather than exactly on it.
        _warn_fisher_boundary(
            (np.abs(estimate_array) >= 1 - epsilon) & ~np.asarray(_saturated_by_construction)
        )
        transformed_estimate = np.arctanh(np.clip(estimate_array, -1 + epsilon, 1 - epsilon))
        transformed_replicates = np.arctanh(np.clip(replicates, -1 + epsilon, 1 - epsilon))
        inverse = _hyperbolic_tangent
        derivative = 1 - np.clip(estimate_array, -1, 1) ** 2
    elif transformation == "fisher_squared":
        # Variance-stabilizing transform for magnitude-squared coherence in
        # [0, 1] (Enochson & Goodman 1965): atanh(sqrt(MSC)) = atanh(|coherence|).
        # Applying Fisher's atanh to the *unsquared* magnitude is the established
        # transform; atanh(MSC) is not. The delta-method derivative back to the
        # MSC scale is d(MSC)/d(atanh(sqrt(MSC))) = 2 * sqrt(MSC) * (1 - MSC),
        # which vanishes at both ends of [0, 1].
        epsilon = np.finfo(float).eps
        clipped_estimate = np.clip(estimate_array, 0, 1)
        _warn_fisher_boundary(
            (np.sqrt(clipped_estimate) >= 1 - epsilon)
            & ~np.asarray(_saturated_by_construction)
        )
        transformed_estimate = np.arctanh(np.clip(np.sqrt(clipped_estimate), 0, 1 - epsilon))
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

    # _replicate_is_valid is private to Connectivity.jackknife: which replicates
    # count in each estimate (broadcast against leave_one_out), where zero
    # observation weights drop different observations in different bins. A
    # replicate leaving out an observation the bin never had is not one.
    counts: Any = n_observations
    if _replicate_is_valid is None:
        replicate_mean = np.mean(transformed_replicates, axis=0)
        squared_deviations = np.sum((transformed_replicates - replicate_mean) ** 2, axis=0)
    else:
        valid = np.broadcast_to(_replicate_is_valid, replicates.shape)
        counts = np.sum(_replicate_is_valid, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            replicate_mean = (
                np.sum(np.where(valid, transformed_replicates, 0.0), axis=0) / counts
            )
            squared_deviations = np.sum(
                np.where(valid, (transformed_replicates - replicate_mean) ** 2, 0.0), axis=0
            )
    with np.errstate(invalid="ignore", divide="ignore"):
        bias_corrected_transformed = (
            counts * transformed_estimate - (counts - 1) * replicate_mean
        )
        transformed_standard_error = np.sqrt((counts - 1) / counts * squared_deviations)
    # Student t on n - 1 degrees of freedom: the jackknife variance is estimated
    # from n replicates, and the normal quantile under-covers at small n.
    critical_value = scipy.stats.t.ppf(0.5 + confidence_level / 2, df=counts - 1)
    half_width = critical_value * transformed_standard_error
    lower = inverse(transformed_estimate - half_width)
    upper = inverse(transformed_estimate + half_width)
    if transformation == "circular":
        lower, upper = _pin_whole_circle_interval(half_width, lower, upper)
    return JackknifeResult(
        estimate=estimate_array,
        bias_corrected=inverse(bias_corrected_transformed),
        standard_error=np.abs(derivative) * transformed_standard_error,
        confidence_interval=(lower, upper),
        n_observations=counts,
        transformation=transformation,
        replicates=replicates,
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
        msg = (
            f"scipy.stats.false_discovery_control is unavailable: "
            f"spectral_connectivity requires scipy>=1.11 for the "
            f"Benjamini-Hochberg procedure, but scipy {scipy.__version__} is "
            f"installed. Upgrade with `pip install -U 'scipy>=1.11'` (or the "
            f"conda/mamba equivalent)."
        )
        raise RuntimeError(msg)


def _validate_alpha(alpha: float, name: str = "alpha") -> None:
    """Validate a significance level, named ``name`` in the caller's signature."""
    if (
        isinstance(alpha, (bool, np.bool_))
        or not isinstance(alpha, (int, float, np.integer, np.floating))  # type: ignore[redundant-expr]  # user input
        or not np.isfinite(alpha)
        or not 0 < alpha < 1
    ):
        msg = (
            f"{name} must be a finite number strictly between 0 and 1, got "
            f"{alpha!r}: it is a significance level, e.g. 0.05."
        )
        raise ValueError(msg)


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
        stacklevel=stacklevel_outside_package(),
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
    raveled); the result has the input shape. For a family per slice (per time
    window, say), use :func:`adjust_for_multiple_comparisons` with ``axis``. Delegates to
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
            msg = (
                "p_values must all be in [0, 1]; "
                f"got {out_of_range.size} value(s) outside that range "
                f"(min={out_of_range.min():.3g}, max={out_of_range.max():.3g}). "
                "If these came from a connectivity measure, pass p-values (e.g. "
                "from coherence_significance_pvalue), not coherence magnitudes or "
                "correlations."
            )
            raise ValueError(msg) from exc
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
        msg = (
            "p_values must all be in [0, 1]; "
            f"got {out_of_range.size} value(s) outside that range "
            f"(min={out_of_range.min():.3g}, max={out_of_range.max():.3g})."
        )
        raise ValueError(msg)
    if finite_values.size:
        # Undefined (NaN/inf) tests are excluded from the family, matching the BH
        # implementation above, and remain False in the returned mask.
        is_significant[valid] = finite_values <= alpha / finite_values.size
    elif p_values.size > 0:
        _warn_all_p_values_nonfinite("Bonferroni_correction")
    return is_significant


#: The corrections :func:`adjust_for_multiple_comparisons` applies, by its ``method`` name.
MULTIPLE_COMPARISONS: dict[str, Callable[..., NDArray[np.bool_]]] = {
    "Benjamini_Hochberg_procedure": Benjamini_Hochberg_procedure,
    "Bonferroni_correction": Bonferroni_correction,
}


def adjust_for_multiple_comparisons(
    p_values: NDArray[np.floating],
    alpha: float = 0.05,
    method: Literal[
        "Benjamini_Hochberg_procedure", "Bonferroni_correction"
    ] = "Benjamini_Hochberg_procedure",
    *,
    axis: int | tuple[int, ...] | None = None,
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
    axis : int or tuple of int or None, default=None
        The axes whose tests form one family, as in
        :func:`scipy.stats.false_discovery_control`: each slice along the
        remaining axes is corrected on its own. ``None`` (the default) pools
        every p-value into a single family. For p-values shaped
        ``(n_time_windows, n_frequencies)``, ``axis=-1`` corrects across
        frequencies separately in each time window.

    Returns
    -------
    is_significant : NDArray[bool], shape (...,)
        Boolean array indicating which tests remain significant after correction.

    Notes
    -----
    **A family per slice controls errors only within each slice.** With
    ``axis``, the false discovery rate (or family-wise error rate) is held at
    ``alpha`` in each family separately, not across them: for null p-values
    shaped ``(100, 126)`` (time windows, frequencies), ``axis=-1`` gives at
    least one false rejection somewhere in 99% of simulated datasets, against
    5% when pooled. Choose the families before looking at the results, to
    match the claims you will make: one per slice when each slice is a separate
    question, the pooled default when results across slices are reported
    together. Keeping only the slices with a discovery and then reporting
    results within them is a selection that needs its own adjustment
    (Benjamini and Bogomolov [1]_).

    **Test each pair once.** A matrix of p-values over ``(source, target)``
    holds a signal's coherence with itself (``p = 0``) on its diagonal and every
    undirected pair twice. The diagonal's sure rejections loosen the
    Benjamini-Hochberg threshold for the real pairs: with 16 signals and no
    coupling, some pair is declared significant in 35% of simulated datasets,
    against 6% for the upper triangle alone. Pass the pairs once, e.g.
    ``p_values[(..., *np.triu_indices(n_signals, 1))]``, with the pair axis in
    the family.

    **Dependence.** P-values at neighboring frequencies (within the taper
    bandwidth), in overlapping windows, and of pairs sharing a signal are
    positively correlated. Bonferroni holds under any dependence, at the cost
    of power; Benjamini-Hochberg is guaranteed for independent or positively
    regression-dependent tests (Benjamini and Yekutieli [2]_).

    References
    ----------
    .. [1] Benjamini, Y., and Bogomolov, M. (2014). Selective inference on
           multiple families of hypotheses. Journal of the Royal Statistical
           Society: Series B 76, 297-318.
    .. [2] Benjamini, Y., and Yekutieli, D. (2001). The control of the false
           discovery rate in multiple testing under dependency. The Annals of
           Statistics 29, 1165-1188.

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

    One family per row (say, per time window) instead of one for the whole
    array. The second row's 0.015 passes Bonferroni among its own three tests
    (0.05 / 3) but not among all six (0.05 / 6):

    >>> p_vals = np.array([[0.001, 0.2, 0.5], [0.015, 0.6, 0.9]])
    >>> adjust_for_multiple_comparisons(p_vals, method="Bonferroni_correction")
    array([[ True, False, False],
           [False, False, False]])
    >>> adjust_for_multiple_comparisons(p_vals, method="Bonferroni_correction", axis=-1)
    array([[ True, False, False],
           [ True, False, False]])
    """
    try:
        correction = MULTIPLE_COMPARISONS[method]
    except KeyError as exc:
        choices = ", ".join(sorted(MULTIPLE_COMPARISONS))
        msg = f"Unknown multiple-comparisons method {method!r}; choose one of: {choices}."
        raise ValueError(msg) from exc
    if axis is None:
        return correction(p_values, alpha=alpha)
    _validate_alpha(alpha)
    p_values = np.asarray(p_values, dtype=float)
    family_axes = _normalized_axes(axis, p_values.ndim)
    other_axes = tuple(dim for dim in range(p_values.ndim) if dim not in family_axes)
    order = (*other_axes, *family_axes)
    # (..., family): one row per family, each corrected by the one-family procedure.
    moved = np.transpose(p_values, order)
    n_families = math.prod(moved.shape[: len(other_axes)])
    families = moved.reshape(n_families, math.prod(moved.shape[len(other_axes) :]))
    is_significant = np.zeros(families.shape, dtype=bool)
    # A family with no finite p-value has no test to correct; it stays
    # all-False and is reported once below, not once per family.
    defined = np.isfinite(families).any(axis=-1)
    for index in np.flatnonzero(defined):
        is_significant[index] = correction(families[index], alpha=alpha)
    n_undefined = int(np.count_nonzero(~defined)) if families.shape[-1] else 0
    if n_undefined:
        warnings.warn(
            f"adjust_for_multiple_comparisons: {n_undefined} of {n_families} families "
            f"(the slices along axis={axis!r}) have no finite p-value (all NaN/inf), "
            "so they hold no defined test and are returned all-False. This usually "
            "means their tests involve a dead/zero-power channel; check your inputs "
            "rather than reading them as 'no significant effects'.",
            UserWarning,
            stacklevel=stacklevel_outside_package(),
        )
    return np.transpose(is_significant.reshape(moved.shape), np.argsort(order))


def _normalized_axes(
    axis: int | tuple[int, ...],
    ndim: int,
    name: str = "axis",
    *,
    array_description: str | None = None,
) -> tuple[int, ...]:
    """``axis`` as distinct non-negative axes of an ``ndim``-dimensional array.

    ``name`` is the argument's name in the error messages, and
    ``array_description`` names the array in the out-of-bounds message
    (``p_values`` when not given).
    """
    axes = (axis,) if not isinstance(axis, tuple) else axis
    normalized = []
    for value in axes:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            msg = f"{name} must be an int, a tuple of ints, or None, got {axis!r}."
            raise TypeError(msg)
        if not -ndim <= value < ndim:
            described = (
                f"p_values with {ndim} dimension(s)"
                if array_description is None
                else f"{array_description} ({ndim} dimension(s))"
            )
            msg = f"{name} {value} is out of bounds for {described}."
            raise ValueError(msg)
        normalized.append(int(value) % ndim)
    if len(set(normalized)) != len(normalized):
        msg = f"{name} {axis!r} names the same axis more than once."
        raise ValueError(msg)
    return tuple(normalized)


def coherence_fisher_z_transform(
    coherency1: NDArray[np.complexfloating] | complex,
    n_obs1: int | NDArray[np.integer],
    coherency2: NDArray[np.complexfloating] | complex = 0,
    n_obs2: int | NDArray[np.integer] = 0,
) -> NDArray[np.floating]:
    """Transform coherence magnitude to approximately normal distribution.

    Applies Fisher's z-transformation to coherence magnitudes, which
    approximately normalizes the distribution for statistical testing.
    Can compute single-sample test against zero or two-sample test.

    Parameters
    ----------
    coherency1 : NDArray[complexfloating] or complex, shape (...,)
        Complex coherency values between signals, ``Connectivity.coherency()``,
        or their magnitudes. A scalar is accepted and gives a scalar (0-d)
        result. Do not pass ``magnitude_squared_coherence()``: its values are
        already squared, and they are read as magnitudes, so the z-scores are
        wrong with no error (take the square root first).
    n_obs1 : int or NDArray[int]
        Number of observations for coherency1 (n_tapers * n_trials), at least
        2, or an array of counts that broadcasts against it, one per estimate,
        such as ``Connectivity.n_valid_observations``. An entry of the array
        below 2 (a bin that keeps fewer observations) makes the z-score NaN
        there; a single count below 2 raises ``ValueError``.
    coherency2 : NDArray[complexfloating] or float, default=0
        Second coherency for comparison, in the same form as ``coherency1``. If
        0, tests against null hypothesis.
    n_obs2 : int or NDArray[int], default=0
        Number of observations for coherency2 (n_tapers * n_trials), 0 for no
        second sample, or an array of counts as for ``n_obs1``, in which 0
        means no second sample at that entry and 1 gives NaN. Condition 2's
        ``n_valid_observations`` is 0 in a bin that keeps no observation, so
        such a bin would be read as a one-sample test: mask those bins
        yourself (its ``coherency2`` is NaN there, which gives NaN).

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

    The bias and variance terms assume complex Fourier coefficients. At the
    0 Hz bin and, for an even FFT length, the Nyquist bin of a real signal the
    coefficients are real, ``arctanh(|C|)`` has about twice the variance, and
    this test is not calibrated: comparing two equal coherences it rejected
    9-18% at a nominal 5% in simulation (8-40 observations). Leave those bins
    out, or test each against zero with
    ``coherence_significance_pvalue(..., real_valued=True)``.
    """
    # coherence_bias evaluates 1 / (2 * (n_obs - 1)); n_obs == 1 divides by zero,
    # and non-finite/fractional counts give NaN with a runtime warning. Checked
    # here, too, so the message names n_obs1 or n_obs2; 0 (no second sample) is
    # valid for n_obs2.
    why = "The coherence bias is undefined below 2 observations."
    counts1, too_few1 = _validated_n_observations(n_obs1, 2, why, name="n_obs1")
    counts2, too_few2 = _validated_n_observations(
        n_obs2, 2, why, name="n_obs2", zero_is_one_sample=True
    )
    # Clip saturated magnitudes below 1 so arctanh stays finite. np.minimum
    # accepts Python and 0-d scalars (whose np.abs is a NumPy scalar that does
    # not support item assignment) and preserves the input's shape.
    largest_magnitude = 1 - np.finfo(float).eps
    coherence_magnitude1 = np.minimum(np.abs(np.asarray(coherency1)), largest_magnitude)
    coherence_magnitude2 = np.minimum(np.abs(np.asarray(coherency2)), largest_magnitude)

    bias1 = coherence_bias(counts1)
    # Where there is no second sample (n_obs2 == 0) the comparison is against a
    # fixed null value (coherency2, default 0), which carries no estimation
    # bias or sampling variance. Evaluating ``coherence_bias(0) = -0.5`` there
    # would instead make ``sqrt(bias1 + bias2)`` negative and return NaN.
    no_second_sample = counts2 == 0
    bias2 = np.where(
        no_second_sample, 0.0, coherence_bias(np.where(no_second_sample, 2, counts2))
    )

    z1: NDArray[np.floating] = np.arctanh(coherence_magnitude1) - bias1
    z2: NDArray[np.floating] = np.arctanh(coherence_magnitude2) - bias2
    scale = np.sqrt(bias1 + bias2)
    return _nan_where((z1 - z2) / scale, too_few1 | too_few2)


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
    # scipy-stubs types distribution methods as returning Any.
    return cast(
        NDArray[np.floating],
        scipy.stats.norm.sf(to_numpy(data), loc=mean, scale=std_deviation),
    )


def _require_broadcastable(
    name: str,
    shape: tuple[int, ...],
    estimates_name: str,
    estimates_shape: tuple[int, ...],
    advice: str,
) -> None:
    """Raise ``ValueError`` unless ``name``, of ``shape``, broadcasts against
    the estimates; the message ends with ``advice`` on aligning it."""
    try:
        np.broadcast_shapes(shape, estimates_shape)
    except ValueError:
        msg = (
            f"{name} has shape {shape}, which does not broadcast against "
            f"{estimates_name} of shape {estimates_shape}. {advice}"
        )
        raise ValueError(msg) from None


def _validated_real_valued(
    real_valued: bool | NDArray[np.bool_],
    estimates_shape: tuple[int, ...] | None = None,
    estimates_name: str = "the estimates",
) -> NDArray[np.bool_]:
    """``real_valued`` as a boolean array that broadcasts against the estimates.

    A non-boolean value (an integer mask, a string) and a mask of the wrong
    shape raise rather than being read as truthy or failing inside NumPy.
    """
    mask = np.asarray(real_valued)
    if mask.dtype != np.bool_:
        msg = (
            f"real_valued must be a bool or a boolean array, got dtype {mask.dtype}. "
            "It marks the estimates from real-valued Fourier coefficients: use "
            "Connectivity.real_valued_frequencies, which is True at 0 Hz and, for "
            "an even FFT length, Nyquist for a real signal."
        )
        raise TypeError(msg)
    if estimates_shape is not None:
        _require_broadcastable(
            "real_valued",
            mask.shape,
            estimates_name,
            estimates_shape,
            "Align it with their frequency axis: pass "
            "Connectivity.real_valued_frequencies[:, np.newaxis] for power shaped "
            "(..., n_frequencies, n_signals), or [:, np.newaxis, np.newaxis] for "
            "coherency shaped (..., n_frequencies, n_signals, n_signals).",
        )
    return mask


def coherence_significance_pvalue(
    coherency: NDArray[np.complexfloating],
    n_observations: int | NDArray[np.integer],
    *,
    real_valued: bool | NDArray[np.bool_] = False,
) -> NDArray[np.floating]:
    """P-value for testing squared coherence magnitude against zero.

    Tests the null hypothesis that the true coherence is zero. Under this null,
    for ``n`` independent complex-Gaussian observations the magnitude-squared
    coherence estimate follows a Beta(1, n - 1) distribution, so the upper-tail
    probability is ``P(|C|^2 >= c) = (1 - c)^(n - 1)``. Where the Fourier
    coefficients are real -- the 0 Hz bin and, for an even FFT length, the
    Nyquist bin of a real signal -- each observation carries one real degree
    of freedom rather than two. The estimate is then the squared cosine of the
    angle between two independent isotropic vectors in R^n rather than C^n,
    which follows Beta(1/2, (n - 1) / 2); the complex null there rejects about
    twice as often as its level (10% at a nominal 5% with five observations).

    This exact boundary distribution should be used instead of the Fisher
    z-transform (:func:`coherence_fisher_z_transform`) when testing against zero
    coherence: the Fisher approximation is derived around a non-zero operating
    point and is badly miscalibrated at ``coherence == 0`` (it over-rejects the
    null by 3-4x, e.g. ~16-22% actual rejection at a nominal 5% level).

    Parameters
    ----------
    coherency : NDArray[complexfloating], shape (...,)
        Complex coherency values between signals, ``Connectivity.coherency()``,
        or their magnitudes; the function squares their absolute value. Do not
        pass ``magnitude_squared_coherence()``: its values are already squared,
        so they would be squared again and the p-values are too large with no
        error (for independent noise, none fell below 0.05 where
        ``coherency()`` gives about the nominal 5%). Take its square root
        first.
    n_observations : int or NDArray[int]
        Number of independent observations used to estimate the coherency
        (n_tapers * n_trials): one count for every estimate, at least 2, or an
        array of counts that broadcasts against ``coherency``, one per estimate.
        Pass ``Connectivity.n_observations``, or, where observation weights drop
        observations (``Multitaper(artifact_samples=...)``),
        ``Connectivity.n_valid_observations[..., np.newaxis, np.newaxis]`` for
        coherency shaped ``(..., n_frequencies, n_signals, n_signals)``. An
        estimate whose count in the array is 0 or 1 (a bin that keeps no
        observation, or one) has no test, and its p-value is NaN; a single
        count below 2 leaves no estimate testable and raises ``ValueError``.
    real_valued : bool or NDArray[bool], default=False
        Which estimates average real-valued Fourier coefficients, a boolean
        array broadcast against ``coherency``; see
        :attr:`~spectral_connectivity.Connectivity.real_valued_frequencies` for
        which bins and why. For coherency shaped ``(..., n_frequencies,
        n_signals, n_signals)`` pass
        ``real_valued_frequencies[:, np.newaxis, np.newaxis]``.

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

    Examples
    --------
    >>> import numpy as np
    >>> coherency = np.array([0.6, 0.6])
    >>> p_values = coherence_significance_pvalue(
    ...     coherency, n_observations=5, real_valued=np.array([False, True])
    ... )
    >>> np.round(p_values, 3).tolist()  # (1 - 0.36)**4, and the real null
    [0.168, 0.208]

    The whole path from a ``Connectivity``: test every time window and
    frequency of a pair, giving the 0 Hz and Nyquist bins their own null and
    the count of trials times tapers, then correct across the tests:

    >>> from spectral_connectivity import Connectivity, Multitaper
    >>> from spectral_connectivity.simulate import simulate_shared_oscillation
    >>> from spectral_connectivity.statistics import adjust_for_multiple_comparisons
    >>> time_series = simulate_shared_oscillation(  # a 20 Hz rhythm in noise
    ...     20.0, 500, 1000, n_trials=10, amplitudes=[1.0, 1.0],
    ...     noise_levels=[2.0, 2.0], random_state=0,
    ... )  # (n_time_samples, n_trials, n_signals) = (1000, 10, 2)
    >>> multitaper = Multitaper(time_series, sampling_frequency=500, time_window_duration=0.5)
    >>> connectivity = Connectivity.from_transform(multitaper)
    >>> coherency = connectivity.coherency()
    >>> coherency.shape  # (n_time_windows, n_frequencies, n_signals, n_signals)
    (4, 126, 2, 2)
    >>> connectivity.n_observations  # 10 trials x 5 tapers
    50
    >>> p_values = coherence_significance_pvalue(
    ...     coherency,
    ...     connectivity.n_observations,
    ...     # (n_frequencies,) -> (n_frequencies, 1, 1): broadcasts over time windows and pairs
    ...     real_valued=connectivity.real_valued_frequencies[:, np.newaxis, np.newaxis],
    ... )
    >>> p_values.shape
    (4, 126, 2, 2)
    >>> # Select the pair first: a signal's coherence with itself is 1 (p = 0)
    >>> # and does not belong in the family. One family: every window and bin
    >>> # (axis=-1 would instead correct each window's bins on their own,
    >>> # controlling errors only within each window).
    >>> significant = adjust_for_multiple_comparisons(p_values[..., 0, 1], alpha=0.05)
    >>> significant.shape  # (n_time_windows, n_frequencies)
    (4, 126)
    >>> # Significant in every window: 20 Hz and neighbors inside its 6 Hz
    >>> # half-bandwidth (NW = 3 over 0.5 s) that the tapers spread it into
    >>> connectivity.frequencies[significant.all(axis=0)].tolist()
    [16.0, 18.0, 20.0, 22.0, 24.0]
    """
    counts, too_few = _validated_n_observations(
        n_observations,
        2,
        "The zero-coherence null, Beta(1, n_observations - 1), needs at least 2.",
        how=(
            "Pass the number of averaged observations (trials x tapers), "
            "Connectivity.n_observations, or, where artifact_samples drops "
            "observations, each bin's count, "
            "Connectivity.n_valid_observations[..., np.newaxis, np.newaxis]."
        ),
    )
    magnitude_squared_coherence: NDArray[np.floating] = np.clip(
        np.abs(coherency) ** 2, 0.0, 1.0
    )
    _require_broadcastable(
        "n_observations",
        counts.shape,
        "coherency",
        magnitude_squared_coherence.shape,
        "Pass one count, or one per estimate: "
        "Connectivity.n_valid_observations[..., np.newaxis, np.newaxis] for "
        "coherency shaped (..., n_frequencies, n_signals, n_signals).",
    )
    # One count keeps its Python scalar, as before arrays were accepted.
    n_observations = counts.item() if counts.ndim == 0 else counts
    real_valued = _validated_real_valued(
        real_valued, magnitude_squared_coherence.shape, "coherency"
    )
    p_values: NDArray[np.floating] = (1.0 - magnitude_squared_coherence) ** (
        n_observations - 1
    )
    if np.any(real_valued):
        # Only the real-valued entries (typically two bins) need the beta tail.
        shape = np.broadcast_shapes(real_valued.shape, np.shape(p_values))
        is_real = np.broadcast_to(real_valued, shape)
        p_values = np.array(np.broadcast_to(p_values, shape), dtype=float, copy=True)
        p_values[is_real] = scipy.stats.beta.sf(
            np.broadcast_to(magnitude_squared_coherence, shape)[is_real],
            0.5,
            (np.broadcast_to(n_observations, shape)[is_real] - 1) / 2,
        )
    return _nan_where(p_values, too_few)


def _validated_n_observations(
    n_observations: Any,
    minimum: int,
    why_below_minimum: str,
    name: str = "n_observations",
    *,
    zero_is_one_sample: bool = False,
    how: str = (
        "Pass the number of averaged observations (trials x tapers), "
        "Connectivity.n_observations."
    ),
) -> tuple[NDArray[Any], NDArray[np.bool_]]:
    """Observation counts, a scalar or one per estimate, validated, and which are too few.

    Raises ``TypeError`` unless every count is a real number (a boolean, a
    string, or None is not a count), and ``ValueError`` unless every count is
    a finite whole number. A single count must be at least ``minimum``
    (``why_below_minimum`` is added to the message when it is not), since it
    would leave no estimate defined. An array of counts, one per estimate,
    may hold counts from 0 up: those below ``minimum`` (bins that keep too few
    observations) are flagged in the returned mask and replaced by
    ``minimum`` in the returned counts, so the caller's formula stays finite
    there, and the caller makes those results NaN with :func:`_nan_where`.
    ``how``, the counts to pass, ends every message. With
    ``zero_is_one_sample``, a count of 0 (no second sample, for a one-sample
    test) is valid too and is not flagged.

    Returns
    -------
    counts : NDArray
        The counts, with ``minimum`` in place of each flagged entry.
    too_few : NDArray[bool]
        True where an array count is below ``minimum``; all False for a
        single count.
    """
    counts = np.asarray(n_observations)
    if counts.dtype.kind not in "iuf":
        msg = (
            f"{name} must be a number (a count of observations), got "
            f"{type(n_observations).__name__} {n_observations!r}. {how}"
        )
        raise TypeError(msg)
    is_array = counts.ndim > 0
    floor = 0 if is_array else minimum
    whole = np.isfinite(counts) & (counts == np.round(counts))
    below_floor = counts < floor
    too_few = whole & (counts < minimum)
    if zero_is_one_sample:
        below_floor &= counts != 0
        too_few &= counts != 0
    invalid = ~whole | below_floor
    if np.any(invalid):
        offending = ", ".join(repr(value) for value in counts[invalid].ravel().tolist()[:5])
        below_minimum = not is_array and np.any(whole & below_floor)
        one_sample = zero_is_one_sample and not is_array
        accepted = f">= {floor}" + (", or 0 for a one-sample test" if one_sample else "")
        array_rules = []
        if minimum > int(zero_is_one_sample):
            array_rules.append(
                f"an entry below {minimum} (a bin that keeps too few observations) gives NaN"
            )
        if zero_is_one_sample:
            array_rules.append("0 means no second sample there")
        in_an_array = (
            "In an array of counts " + ", and ".join(array_rules) + ". " if is_array else ""
        )
        msg = (
            f"{name} must be a finite integer {accepted}, got {offending}. "
            f"{why_below_minimum + ' ' if below_minimum and why_below_minimum else ''}"
            f"{in_an_array}{how}"
        )
        raise ValueError(msg)
    return np.where(too_few, minimum, counts) if is_array else counts, too_few


def _nan_where(
    values: NDArray[np.floating], too_few: NDArray[np.bool_]
) -> NDArray[np.floating]:
    """``values`` with NaN where ``too_few`` (from :func:`_validated_n_observations`)."""
    return np.where(too_few, np.nan, values) if np.any(too_few) else values


def coherence_bias(
    n_observations: int | NDArray[np.integer],
) -> float | NDArray[np.floating]:
    """Estimate bias correction for coherence estimates.

    Coherence estimates are biased by finite sample size. This function
    computes the bias correction factor that can be subtracted from
    Fisher z-transformed coherence estimates. It assumes complex Fourier
    coefficients and does not hold at the 0 Hz and Nyquist bins of a real
    signal, whose coefficients are real (see
    :func:`coherence_fisher_z_transform`).

    Parameters
    ----------
    n_observations : int or NDArray[int]
        Number of observations used in coherence estimation (n_tapers *
        n_trials), at least 2: pass ``Connectivity.n_observations``. Or an
        array of counts, one per estimate, such as
        ``Connectivity.n_valid_observations``: an entry below 2 (a bin that
        keeps fewer observations) gives NaN there.

    Returns
    -------
    bias : float or NDArray[floating]
        Bias correction factor for Fisher z-transform of coherence: a float
        for a single count, otherwise an array of the counts' shape.

    Raises
    ------
    ValueError
        If a single ``n_observations`` is not a finite integer of at least 2,
        or an array of them holds a negative, non-finite, or fractional count.

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
           337-345. 10.1016/j.jneumeth.2006.07.011.
    """
    counts, too_few = _validated_n_observations(
        n_observations,
        2,
        "The bias, 1 / (2 * n_observations - 2), is undefined below 2 observations.",
    )
    degrees_of_freedom = 2 * counts
    return _float_if_scalar(_nan_where(1.0 / (degrees_of_freedom - 2), too_few))


def coherence_rate_adjustment(
    firing_rate_condition1: float,
    firing_rate_condition2: float,
    spike_power_spectrum: NDArray[np.floating],
    homogeneous_poisson_noise: float = 0,
    dt: float = 1,
    *,
    is_one_sided: bool | None = None,
) -> NDArray[np.floating]:
    """Return the factor by which a firing-rate change alone alters spike-field coherence.

    Spike-field coherence depends on the firing rate even when the coupling
    between spikes and field does not change. This function gives the factor
    by which the coherence falls when a fraction of a spike train's spikes is
    removed at random (thinning), optionally with independent Poisson spikes
    added [1]_. Multiply the first condition's coherence by it to get the
    coherence expected at the second condition's rate under unchanged
    coupling, and compare the second condition's coherence with that.

    Parameters
    ----------
    firing_rate_condition1 : float
        Average firing rate in the first condition (spikes/sec).
    firing_rate_condition2 : float
        Average firing rate in the second condition (spikes/sec), including
        any spikes counted in ``homogeneous_poisson_noise``.
    spike_power_spectrum : NDArray[floating], shape (n_frequencies,)
        Power spectral density of the first condition's spike train, where the
        train was given as spike counts per time bin (0 or 1 for small bins),
        not as a rate. Say with ``is_one_sided`` whether it is the one-sided
        density, as :meth:`Connectivity.power
        <spectral_connectivity.connectivity.Connectivity.power>` returns it,
        or a two-sided one.
    homogeneous_poisson_noise : float, default=0
        Rate (spikes/sec) of spikes added in the second condition at a
        constant rate, independently of the field (``beta`` in [1]_). The
        second condition is then the first with a fraction ``alpha`` of its
        spikes kept, plus these, so ``alpha`` is
        ``(firing_rate_condition2 - homogeneous_poisson_noise) /
        firing_rate_condition1``. This rate is not estimated here: give it
        only if the experiment tells you what it is, and leave 0 when spikes
        are only removed. Must be at least 0 and below
        ``firing_rate_condition2``.

        .. versionchanged:: 3.0
           ``firing_rate_condition2`` includes these spikes. 2.x took it as
           the rate of the kept spikes alone, against its documentation, so
           a call with a non-zero rate here warns about the change (until
           4.0).
    dt : float, default=1
        Width of the time bins the spikes are counted in, in seconds: for a
        spectrum from this package, ``1 / sampling_frequency`` of the spike
        train. The default of 1 suits only a spectrum computed with a
        sampling frequency of 1.
    is_one_sided : bool, optional
        Whether ``spike_power_spectrum`` is a one-sided density, in which each
        frequency bin includes the power of its mirror negative frequency, as
        ``Connectivity.power()`` returns it (``True``), or a two-sided one,
        such as ``Connectivity.power(one_sided=False)`` (``False``). The
        adjustment is defined for the two-sided density, which is half of the
        one-sided one at frequencies other than 0 Hz and Nyquist. If omitted,
        the spectrum is taken as one-sided and a warning says so, because the
        two cannot be told apart from their values.

        .. versionadded:: 3.0
           In 2.x the spectrum was read as two-sided, which is what
           ``power()`` returned then.

    Returns
    -------
    rate_adjustment_factor : NDArray[floating], shape (n_frequencies,)
        Factor that multiplies the magnitude of the first condition's
        coherence (not its square); at most 1 when the rate does not rise.
        ``NaN`` where it is undefined. Leave out the value at 0 Hz, and with
        the default mean removal (``detrend_type="constant"``) the values
        within a taper's bandwidth of it: the removal of each window's mean
        distorts the estimated spike spectrum there. Without mean removal the
        mean's power dominates 0 Hz, and with added Poisson spikes the
        distortion reaches further. The value at the Nyquist frequency is
        valid only with a two-sided spectrum, because a one-sided density is
        not doubled there.

    Examples
    --------
    >>> import numpy as np
    >>> rate1, rate2 = 15.0, 10.0  # firing rate decreases in condition 2
    >>> dt = 0.002  # spikes counted in 2 ms bins (sampling frequency 500 Hz)
    >>> freqs = np.linspace(1, 100, 50)
    >>> # A one-sided spike spectrum: the level of a Poisson train of that rate,
    >>> # 2 * rate * dt**2, with a peak at 40 Hz.
    >>> power_spec = 2 * rate1 * dt**2 * (1 + 3 * np.exp(-(((freqs - 40) / 5) ** 2)))
    >>> adjustment = coherence_rate_adjustment(
    ...     rate1, rate2, power_spec, dt=dt, is_one_sided=True
    ... )
    >>> print(f"Adjustment range: {adjustment.min():.3f} to {adjustment.max():.3f}")
    Adjustment range: 0.816 to 0.942

    Notes
    -----
    The spectrum a spike train of rate ``mu`` would have without any
    structure is ``mu * dt**2`` as a two-sided density and twice that as a
    one-sided one. The adjustment depends on the given spectrum relative to
    that level, which is why the kind of density matters.

    For spike-spike coherence comparisons, apply this adjustment twice,
    once for each spike train.

    References
    ----------
    .. [1] Aoi, M.C., Lepage, K.Q., Kramer, M.A., and Eden, U.T. (2015).
           Rate-adjusted spike-LFP coherence comparisons from spike-train
           statistics. Journal of Neuroscience Methods 240, 141-153.
    """
    if not np.isfinite(firing_rate_condition1) or firing_rate_condition1 <= 0:
        msg = (
            f"firing_rate_condition1 must be a finite positive number, got "
            f"{firing_rate_condition1}."
        )
        raise ValueError(msg)
    if not np.isfinite(firing_rate_condition2) or firing_rate_condition2 <= 0:
        msg = (
            f"firing_rate_condition2 must be a finite positive number, got "
            f"{firing_rate_condition2}."
        )
        raise ValueError(msg)
    if is_one_sided is not None:
        is_one_sided = _validated_flag("is_one_sided", is_one_sided)
    # NaN and inf fail the comparison too.
    if not 0 <= homogeneous_poisson_noise < firing_rate_condition2:
        msg = (
            "homogeneous_poisson_noise must be a finite rate of at least 0 and "
            f"below firing_rate_condition2 ({firing_rate_condition2}), which "
            f"includes the added spikes; got {homogeneous_poisson_noise}. If "
            "firing_rate_condition2 is the rate of the kept spikes alone, add "
            "the noise rate to it."
        )
        raise ValueError(msg)
    if homogeneous_poisson_noise > 0:
        # Remove in 4.0 (issue #101).
        warnings.warn(
            "coherence_rate_adjustment, changed in spectral_connectivity 3.0: "
            "firing_rate_condition2 includes the spikes counted in "
            "homogeneous_poisson_noise: it is the second condition's measured "
            "rate. 2.x took it as the rate of the kept spikes alone. If a call "
            "written for 2.x passed that rate, add the noise rate to it. This "
            "notice accompanies calls with added spikes until 4.0; "
            "filter it with warnings.filterwarnings('ignore', "
            "message='coherence_rate_adjustment, changed in').",
            UserWarning,
            stacklevel=stacklevel_outside_package(),
        )
    if is_one_sided is None:
        warnings.warn(
            "coherence_rate_adjustment was not told whether spike_power_spectrum "
            "is one-sided, and takes it as one-sided, which is what "
            "Connectivity.power() returns since 3.0. Pass is_one_sided=True to "
            "confirm that and silence this warning, or is_one_sided=False for a "
            "two-sided density (2.x's power(), Connectivity.power(one_sided=False), "
            "or a spectrum from elsewhere that runs over negative and positive "
            "frequencies).",
            UserWarning,
            stacklevel=stacklevel_outside_package(),
        )
        is_one_sided = True
    # alpha in [1]: the second condition's intensity is alpha * lambda(t) + beta,
    # so its mean rate is alpha * firing_rate_condition1 + beta.
    intensity_scale = (
        firing_rate_condition2 - homogeneous_poisson_noise
    ) / firing_rate_condition1
    adjusted_firing_rate = (
        (1 / intensity_scale - 1) * firing_rate_condition1
        + homogeneous_poisson_noise / intensity_scale**2
    ) * dt**2
    # Spike power spectral density is non-negative by definition; a non-positive
    # value is invalid input (not just undefined), so it is masked below.
    spike_power = np.asarray(spike_power_spectrum, dtype=float)
    if is_one_sided:
        # The reference's spectrum runs over both signs of the frequency.
        spike_power = spike_power / 2
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
            stacklevel=stacklevel_outside_package(),
        )
    return np.where(undefined, np.nan, adjustment)


class _DefaultConfidenceLevel(float):
    """0.95, the default level: shown as ``0.95`` in the signature, but a
    different object from any level a caller passes (0.95 included), so
    giving it together with the deprecated ``ci`` can be detected."""

    __slots__ = ()


_DEFAULT_CONFIDENCE_LEVEL = _DefaultConfidenceLevel(0.95)


def power_confidence_intervals(
    n_observations: int | NDArray[np.integer] | None = None,
    power: NDArray[np.floating] | float = 1,
    confidence_level: float = _DEFAULT_CONFIDENCE_LEVEL,
    *,
    n_tapers: int | None = None,
    ci: float | None = None,
    real_valued: bool | NDArray[np.bool_] = False,
) -> tuple[NDArray[np.floating] | float, NDArray[np.floating] | float]:
    """Compute confidence intervals for multitaper power spectrum estimates.

    Uses chi-squared distribution to compute confidence bounds for power
    spectral density estimates from multitaper analysis.

    .. versionchanged:: 3.0
       The count is named ``n_observations`` and the level
       ``confidence_level``, as in
       :meth:`~spectral_connectivity.Connectivity.jackknife`; both keep their
       positions, so positional calls are unchanged. ``n_tapers=`` and
       ``ci=`` still work through 3.x with a ``DeprecationWarning`` and are
       removed in 4.0.

    Parameters
    ----------
    n_observations : int
        Number of independent observations averaged into each power estimate:
        tapers times trials (``Connectivity.n_observations``), not the taper
        count alone unless a single trial was averaged. It sets the chi-squared
        degrees of freedom ``2 * n_observations``, as in :func:`power_bias` and
        :func:`power_variance`. Passing only the taper count for a multi-trial
        average makes the interval far too wide (about 100% coverage instead of
        95% for 5 tapers x 5 trials). Or an array of counts that broadcasts
        against ``power``, one per estimate, such as
        ``Connectivity.n_valid_observations[..., np.newaxis]``: where an entry
        is 0 (a bin that keeps no observation) the bounds are NaN; a single
        count of 0 raises ``ValueError``.
    power : NDArray[floating] or float, default=1
        Power spectrum estimates. Can be array of values or scalar. It must be
        finite and non-negative, except where an array count is 0, where it
        may be NaN, as ``Connectivity.power()`` is there.
    confidence_level : float, default=0.95
        The probability mass inside the two-sided interval (0.95 for a 95%
        interval), in the range [0.5, 1.0).
    n_tapers : int, optional
        Deprecated name of ``n_observations``, by keyword only; removed in 4.0.
        Giving both raises ``ValueError``.
    ci : float, optional
        Deprecated name of ``confidence_level``, by keyword only; removed in
        4.0. Giving both raises ``ValueError``.
    real_valued : bool or NDArray[bool], default=False
        Which estimates average real-valued Fourier coefficients, a boolean
        array broadcast against ``power``; see
        :attr:`~spectral_connectivity.Connectivity.real_valued_frequencies` for
        which bins and why. For ``Connectivity.power()``, shaped ``(...,
        n_frequencies, n_signals)``, pass ``real_valued_frequencies[:,
        np.newaxis]``. Those bins use ``n_observations`` degrees of freedom; with
        ``2 * n_observations`` there a nominal 95% interval from five
        observations covers 83%.

    Returns
    -------
    lower_bound : NDArray[floating]
        Lower confidence bounds for power estimates.
    upper_bound : NDArray[floating]
        Upper confidence bounds for power estimates.

    Examples
    --------
    >>> import numpy as np
    >>> # Single power estimate from 5 observations (5 tapers x 1 trial)
    >>> lower, upper = power_confidence_intervals(
    ...     n_observations=5, power=1.0, confidence_level=0.95
    ... )
    >>> print(f"95% CI: [{lower:.3f}, {upper:.3f}]")
    95% CI: [0.488, 3.080]
    >>> # Averaging 5 tapers x 5 trials gives 25 observations, a much tighter interval
    >>> lower, upper = power_confidence_intervals(
    ...     n_observations=25, power=1.0, confidence_level=0.95
    ... )
    >>> print(f"95% CI: [{lower:.3f}, {upper:.3f}]")
    95% CI: [0.700, 1.545]
    >>> # Multiple power estimates
    >>> power_vals = np.array([0.5, 1.0, 2.0, 5.0])
    >>> lower, upper = power_confidence_intervals(5, power_vals, 0.95)
    >>> np.round(lower, 3).tolist()
    [0.244, 0.488, 0.976, 2.441]
    >>> # The 0 Hz bin of a real signal: half the degrees of freedom, a wider interval
    >>> lower, upper = power_confidence_intervals(5, 1.0, 0.95, real_valued=True)
    >>> print(f"95% CI: [{lower:.3f}, {upper:.3f}]")
    95% CI: [0.390, 6.015]

    From a ``Connectivity``: pass its ``n_observations`` and mark its real
    0 Hz and Nyquist bins:

    >>> from spectral_connectivity import Connectivity, Multitaper
    >>> rng = np.random.default_rng(0)
    >>> time_series = rng.standard_normal((1000, 10, 2))  # (time, trials, signals)
    >>> multitaper = Multitaper(time_series, sampling_frequency=500, time_window_duration=0.5)
    >>> connectivity = Connectivity.from_transform(multitaper)
    >>> power = connectivity.power(one_sided=True)
    >>> power.shape  # (n_time_windows, n_frequencies, n_signals)
    (4, 126, 2)
    >>> lower, upper = power_confidence_intervals(
    ...     connectivity.n_observations,  # 10 trials x 5 tapers = 50
    ...     power,
    ...     confidence_level=0.95,
    ...     # (n_frequencies,) -> (n_frequencies, 1): broadcasts over time windows and signals
    ...     real_valued=connectivity.real_valued_frequencies[:, np.newaxis],
    ... )
    >>> lower.shape
    (4, 126, 2)
    >>> connectivity.frequencies[[0, 10]].tolist()
    [0.0, 20.0]
    >>> # Upper bound over the estimate: wider at the real 0 Hz bin
    >>> np.round(upper[0, [0, 10], 0] / power[0, [0, 10], 0], 2).tolist()
    [1.55, 1.35]

    References
    ----------
    .. [1] Kramer, M.A., and Eden, U.T. (2016). Case studies in neural
           data analysis: a guide for the practicing neuroscientist (MIT Press).
    """
    if n_tapers is not None:
        if n_observations is not None:
            msg = both_spellings_message(
                "power_confidence_intervals", "n_observations", "n_tapers"
            )
            raise ValueError(msg)
        n_observations = n_tapers
    if n_observations is None:
        msg = "power_confidence_intervals() missing required argument: 'n_observations'"
        raise TypeError(msg)
    level_name = "confidence_level"
    if ci is not None:
        if confidence_level is not _DEFAULT_CONFIDENCE_LEVEL:
            msg = both_spellings_message(
                "power_confidence_intervals", "confidence_level", "ci"
            )
            raise ValueError(msg)
        confidence_level, level_name = ci, "ci"
    counts, too_few = _validated_n_observations(
        n_observations,
        1,
        "It sets the chi-squared degrees of freedom, 2 * n_observations, which "
        "must be positive.",
    )
    if not 0.5 <= confidence_level < 1.0:
        msg = (
            f"{level_name} must be at least 0.5 and less than 1, got "
            f"{confidence_level!r}. It is the probability mass inside the interval, "
            "e.g. 0.95 for a 95% interval."
        )
        raise ValueError(msg)
    # Power spectral density is non-negative; a negative value scales the bounds
    # negative and reversed (e.g. power=-1 -> ~(-0.488, -3.080)), and a
    # non-finite value gives NaN/Inf bounds. Reject both explicitly, except
    # where an array count keeps no observation: the power there is NaN, and
    # its bounds are NaN whatever it is.
    power_values = np.asarray(power, dtype=float)
    checked_power = np.where(too_few, 0.0, power_values)
    if not np.all(np.isfinite(checked_power)) or np.any(checked_power < 0):
        msg = (
            "power must be finite and non-negative (it is a power spectral "
            "density); got a non-finite or negative value."
        )
        raise ValueError(msg)
    # Warned only once the call is valid, so a rejected call shows its error.
    if n_tapers is not None:
        warn_deprecated(
            "power_confidence_intervals(n_tapers=...)",
            "power_confidence_intervals(n_observations=...)",
            "gives the same interval; the count is every averaged observation, "
            "tapers x trials",
        )
    if ci is not None:
        warn_deprecated(
            "power_confidence_intervals(ci=...)",
            "power_confidence_intervals(confidence_level=...)",
            "gives the same interval, under the name jackknife uses",
        )
    # A two-sided (1 - alpha) interval splits the tail mass
    # alpha = 1 - confidence_level evenly between the two tails, so the
    # chi-squared quantiles are taken at alpha / 2 and 1 - alpha / 2. Using the
    # full alpha on each tail (the previous behavior) produces an interval with
    # coverage 2 * confidence_level - 1 (a requested 95% interval only covered
    # ~90%).
    real_valued = _validated_real_valued(real_valued, power_values.shape, "power")
    degrees_of_freedom = 2 * _chi_squared_shape(counts, real_valued)
    alpha = 1 - confidence_level
    lower_bound = (
        degrees_of_freedom / scipy.stats.chi2.ppf(1 - alpha / 2, degrees_of_freedom) * power
    )
    upper_bound = (
        degrees_of_freedom / scipy.stats.chi2.ppf(alpha / 2, degrees_of_freedom) * power
    )

    return _nan_where(lower_bound, too_few), _nan_where(upper_bound, too_few)


def _chi_squared_shape(
    n_observations: int | NDArray[Any], real_valued: bool | NDArray[np.bool_]
) -> NDArray[np.floating]:
    """Shape ``nu / 2`` of a power estimate's chi-squared distribution.

    Each complex coefficient carries two degrees of freedom and each real one
    (the 0 Hz and Nyquist bins of a real signal) one, so ``nu`` is
    ``2 * n_observations``, or ``n_observations`` where ``real_valued``.
    """
    mask = _validated_real_valued(real_valued)
    shape: NDArray[np.floating] = np.where(mask, n_observations / 2, n_observations)
    return shape


def _float_if_scalar(value: NDArray[np.floating]) -> float | NDArray[np.floating]:
    """A float for a 0-d result (scalar count and mask), else the array."""
    return float(value) if np.ndim(value) == 0 else value


def power_bias(
    n_observations: int | NDArray[np.integer],
    *,
    real_valued: bool | NDArray[np.bool_] = False,
) -> float | NDArray[np.floating]:
    """Bias of the log power spectrum.

    A multitaper power estimate satisfies ``S_hat / S ~ chi2_nu / nu`` with
    ``nu = 2 * n_observations`` degrees of freedom. Writing ``chi2_nu`` as
    ``2 * Gamma(nu / 2)`` gives ``E[log(S_hat / S)] = psi(nu / 2) - log(nu / 2)``,
    i.e. the digamma/log are evaluated at the chi-squared shape parameter
    ``nu / 2 = n_observations`` (not at ``nu``). Where the Fourier
    coefficients are real (the 0 Hz bin and, for an even FFT length, the
    Nyquist bin of a real signal) ``nu = n_observations``.

    Parameters
    ----------
    n_observations : int or NDArray[int]
        Number of observations used in power estimation (n_tapers * n_trials),
        at least 1: pass ``Connectivity.n_observations``. Or an array of
        counts, one per estimate, such as ``Connectivity.n_valid_observations``:
        an entry of 0 (a bin that keeps no observation) gives NaN there.
    real_valued : bool or NDArray[bool], default=False
        Which estimates average real-valued Fourier coefficients, a boolean
        mask such as
        :attr:`~spectral_connectivity.Connectivity.real_valued_frequencies`.

    Returns
    -------
    bias : float or NDArray[floating]
        A float for a scalar count and ``real_valued``; otherwise an array
        of their broadcast shape.

    Raises
    ------
    ValueError
        If a single ``n_observations`` is not a finite positive integer, or an
        array of them holds a negative, non-finite, or fractional count.

    Examples
    --------
    >>> print(f"Bias with 100 obs: {power_bias(100):.6f}")
    Bias with 100 obs: -0.005008
    >>> print(f"Bias with 1000 obs: {power_bias(1000):.6f}")
    Bias with 1000 obs: -0.000500
    >>> print(f"At a real-valued bin: {power_bias(100, real_valued=True):.6f}")
    At a real-valued bin: -0.010033
    """
    counts, too_few = _validated_n_observations(
        n_observations, 1, "The digamma function has poles at 0 and the negative integers."
    )
    shape = _chi_squared_shape(counts, real_valued)
    return _float_if_scalar(_nan_where(scipy.special.psi(shape) - np.log(shape), too_few))


def power_variance(
    n_observations: int | NDArray[np.integer],
    *,
    real_valued: bool | NDArray[np.bool_] = False,
) -> float | NDArray[np.floating]:
    """Compute variance of log-power spectrum estimates.

    Calculates the variance of log-transformed power spectrum estimates
    for use in statistical testing and confidence interval computation.

    Parameters
    ----------
    n_observations : int or NDArray[int]
        Number of observations used in power estimation (n_tapers * n_trials),
        at least 1, or an array of counts, as in :func:`power_bias`: an entry
        of 0 gives NaN there.
    real_valued : bool or NDArray[bool], default=False
        Which estimates average real-valued Fourier coefficients, as in
        :func:`power_bias`.

    Returns
    -------
    variance : float or NDArray[floating]
        Variance of log-power estimates: a float for a scalar count and
        ``real_valued``; otherwise an array of their broadcast shape.

    Raises
    ------
    ValueError
        If a single ``n_observations`` is not a finite positive integer, or an
        array of them holds a negative, non-finite, or fractional count.

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
    chi-squared shape parameter ``nu / 2 = n_observations``; at a real-valued
    bin ``nu = n_observations``, which about doubles it.
    """
    counts, too_few = _validated_n_observations(
        n_observations, 1, "The trigamma function has poles at 0 and the negative integers."
    )
    shape = _chi_squared_shape(counts, real_valued)
    return _float_if_scalar(_nan_where(scipy.special.polygamma(1, shape), too_few))


def power_fisher_z_transform(
    spectrum1: NDArray[np.floating],
    n_obs1: int | NDArray[np.integer],
    spectrum2: NDArray[np.floating] | float = 1.0,
    n_obs2: int | NDArray[np.integer] = 0,
    *,
    real_valued: bool | NDArray[np.bool_] = False,
) -> NDArray[np.floating]:
    """Transform power spectrum estimates for statistical testing.

    Applies log-transformation with bias correction to power spectrum estimates,
    enabling approximately normal distributions for hypothesis testing.
    Can perform one-sample test against baseline or two-sample comparison.

    Parameters
    ----------
    spectrum1 : NDArray[floating], shape (...,)
        Power spectrum estimates from first condition, finite and positive
        except where an array ``n_obs1`` is 0, where it may be NaN, as
        ``Connectivity.power()`` is there.
    n_obs1 : int or NDArray[int]
        Number of observations for spectrum1 (n_tapers * n_trials), at least
        1, or an array of counts that broadcasts against it, one per estimate,
        such as ``Connectivity.n_valid_observations[..., np.newaxis]``. An
        entry of 0 (a bin that keeps no observation) makes the z-score NaN
        there; a single count of 0 raises ``ValueError``.
    spectrum2 : NDArray[floating] or float, default=1.0
        For a two-sample comparison (``n_obs2 > 0``), the power spectrum
        estimates from the second condition. For a one-sample test
        (``n_obs2 == 0``), a fixed positive baseline power against which
        ``spectrum1`` is compared; must be > 0 because the test operates on
        ``log(power)``.
    n_obs2 : int or NDArray[int], default=0
        Number of observations for spectrum2 (n_tapers * n_trials). If 0,
        performs a one-sample test of ``spectrum1`` against the fixed baseline
        ``spectrum2``; in an array of counts, at that entry. Condition 2's
        ``n_valid_observations`` is 0 in a bin that keeps no observation, so
        such a bin is read as a one-sample test (and its NaN ``spectrum2``
        raises): mask those bins yourself.
    real_valued : bool or NDArray[bool], default=False
        Which estimates average real-valued Fourier coefficients, broadcast
        against the spectra (see :func:`power_bias`); there the log power has
        about twice the variance, so treating them as complex overstates z.

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
    # counts would silently produce NaN z-scores. Checked here so the message
    # names n_obs1 or n_obs2; 0 (no second sample) is valid for n_obs2.
    why = "The power bias and variance (digamma and trigamma) have poles at 0."
    counts1, too_few = _validated_n_observations(n_obs1, 1, why, name="n_obs1")
    # Every n_obs2 from 0 up is valid, so none is too few.
    counts2, _ = _validated_n_observations(
        n_obs2, 1, why, name="n_obs2", zero_is_one_sample=True
    )
    # The test operates on log(power); non-finite or non-positive inputs would
    # produce silent -inf/nan z-scores. Fail loudly instead, except where an
    # array n_obs1 keeps no observation: spectrum1 is NaN there, and so is z.
    spectrum1_values = np.asarray(spectrum1, dtype=float)
    checked_spectrum1 = np.where(too_few, 1.0, spectrum1_values)
    if not np.all(np.isfinite(checked_spectrum1)) or np.any(checked_spectrum1 <= 0):
        msg = (
            "spectrum1 must be finite and strictly positive (the test uses "
            "log(power)); got a non-finite or <= 0 value."
        )
        raise ValueError(msg)
    spectrum2_values = np.asarray(spectrum2, dtype=float)
    if not np.all(np.isfinite(spectrum2_values)) or np.any(spectrum2_values <= 0):
        msg = (
            "spectrum2 must be finite and strictly positive (the test uses "
            "log(power)); got a non-finite or <= 0 value. For a one-sample test, "
            "pass a positive baseline power as spectrum2 (default 1.0)."
        )
        raise ValueError(msg)

    real_valued = _validated_real_valued(
        real_valued,
        np.broadcast_shapes(spectrum1_values.shape, spectrum2_values.shape),
        "the spectra",
    )
    bias1 = power_bias(counts1, real_valued=real_valued)
    variance1 = power_variance(counts1, real_valued=real_valued)
    # Where there is no second sample (n_obs2 == 0) spectrum2 is a fixed
    # baseline that carries no estimation bias or sampling variance. Evaluating
    # power_bias(0)/power_variance(0) there would instead hit the
    # digamma/trigamma poles and return NaN.
    no_second_sample = counts2 == 0
    second_counts = np.where(no_second_sample, 1, counts2)
    bias2 = np.where(no_second_sample, 0.0, power_bias(second_counts, real_valued=real_valued))
    variance2 = np.where(
        no_second_sample, 0.0, power_variance(second_counts, real_valued=real_valued)
    )

    # Bias correction
    z1: NDArray[np.floating] = np.log(spectrum1) - bias1
    z2: NDArray[np.floating] = np.log(spectrum2) - bias2

    scale = np.sqrt(variance1 + variance2)
    return _nan_where((z1 - z2) / scale, too_few)
