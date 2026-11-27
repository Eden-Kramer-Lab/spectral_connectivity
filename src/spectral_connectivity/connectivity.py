"""Compute metrics for relating signals in the frequency domain."""

import contextlib
import inspect
import math
import re
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, replace
from functools import cached_property, wraps
from itertools import combinations
from logging import getLogger
from typing import TYPE_CHECKING, Any, Literal, ParamSpec, TypeVar, cast

import numpy as np
from numpy.typing import DTypeLike, NDArray

from spectral_connectivity._array_utils import (
    TIKHONOV_REGULARIZATION_FACTOR,
    _batched_inverse_square_root,
    _clip_rounding_above_one,
    _complex_inner_product,
    _conjugate_transpose,
    _divide_where,
    _NumberT,
    _regularized_inverse,
    _squared_magnitude,
    _weight_observations,
)
from spectral_connectivity._backend import xp
from spectral_connectivity._granger import (
    BLOCK_DEGENERATE_WARNING_START,
    CONDITIONAL_DEGENERATE_WARNING_START,
    NAN_PAIRS_WARNING_AFTER_MEASURE,
    _estimate_all_conditional_spectral_granger,
    _estimate_blockwise_spectral_granger,
    _estimate_spectral_granger_prediction,
    _estimate_subset_spectral_granger_prediction,
    _factorize_spectrum,
    _var_model_from_factor,
    _warn_granger_failures,
)
from spectral_connectivity._multivariate import (
    GLOBAL_COHERENCE_BATCH_CHUNK_ELEMENTS,
    _canonical_coherency_components,
    _estimate_canonical_coherence,
    _global_coherence,
    _mic_components,
    _normalize_fourier_coefficients,
)
from spectral_connectivity.minimum_phase_decomposition import (
    CHOLESKY_FALLBACK_WARNING_START,
    NONCONVERGENCE_WARNING_START,
)
from spectral_connectivity.statistics import (
    JackknifeResult,
    SurrogateTestResult,
    _normalized_axes,
    _surrogate_statistic,
    _SurrogateTally,
    _validate_alpha,
    _validate_confidence_level,
    adjust_for_multiple_comparisons,
    coherence_significance_pvalue,
    jackknife_confidence_interval,
)
from spectral_connectivity.transforms import VARSpectrum, _var_settings_by_keyword
from spectral_connectivity.utils import (
    BackendArray,
    _generator,
    _real_scalar,
    _validated_flag,
    did_you_mean,
    is_positive_integer,
    mark_readonly_chain_if_supported,
    mark_readonly_if_supported,
    stacklevel_outside_package,
    to_numpy,
    warn_deprecated,
    warn_migration,
)

if TYPE_CHECKING:
    from spectral_connectivity.transforms import SpectralTransform

logger = getLogger(__name__)

# The first sentence of the warning for NaN or Inf coefficients, which
# from_transform matches to drop it after the transform's own warning.
_NONFINITE_COEFFICIENTS_WARNING = "fourier_coefficients contains NaN or Inf values."

# Public helpers on Connectivity that are not connectivity measures. Keep this
# definition shared with the high-level wrapper so method discovery and
# the resampling drivers' validation cannot drift apart.
_NON_MEASURE_METHODS = frozenset(
    {"clear_cache", "jackknife", "partial_coherence_shrinkage", "surrogate_test"}
)
# Connectivity.surrogate_test: the surrogate constructions and alternatives it
# takes.
# The question each kind answers, for the message naming them.
_SURROGATE_QUESTIONS = {
    "trial_shuffle": "do the signals covary from trial to trial (several trials)?",
    "phase_rotation": "are the signals independent at all?",
    "conjugation": "is there more than zero-lag mixing such as volume conduction?",
}
_SURROGATE_KINDS = tuple(_SURROGATE_QUESTIONS)
_SURROGATE_ALTERNATIVES = ("auto", "greater", "less", "two-sided")
# Why surrogate_test refuses a registered measure, keyed by what the registry
# says of it (see _surrogate_refusal).
_SURROGATE_REFUSALS = {
    "single_signal": (
        "it describes one signal, and every surrogate keeps each signal's "
        "spectrum, so its null distribution is the estimate itself. Test a "
        "measure of the relation between signals."
    ),
    "candidate_delays": (
        "it returns candidate delays at the frequencies its own coherence test "
        "selects, not a statistic. Test magnitude_squared_coherence, or "
        "phase_slope_index for the direction of a lag."
    ),
    "phase": (
        "a phase is circular, so no ordering of its values is evidence against "
        "the null, and it is undefined where coherence is weak. Test "
        "imaginary_coherence (or imaginary_coherency, two-sided) for a lagged "
        "relation, or phase_locking_value for phase coupling."
    ),
}
# Measure names released in 2.x and replaced in 3.0, mapped to the canonical
# name. Each still works through 3.x with a DeprecationWarning and is removed
# in 4.0; the canonical measure is the one registered and listed.
_DEPRECATED_MEASURE_NAMES: dict[str, str] = {
    "coherence_magnitude": "magnitude_squared_coherence",
}


def _resolve_deprecated_measure_name(method: str, spelling: str) -> str:
    """The canonical name of ``method``; a deprecated name warns at the caller.

    ``spelling`` writes a name the way the caller did, e.g. ``"method={!r}"``,
    for the old and new spellings in the warning.
    """
    canonical = _DEPRECATED_MEASURE_NAMES.get(method)
    if canonical is None:
        return method
    warn_deprecated(spelling.format(method), spelling.format(canonical))
    return canonical


# In 2.x blocks was meant only to chunk the cross-spectral computation, but it
# also changed the phase-lag-index measures (see the migration guide).
_BLOCKS_REMOVED = (
    "blocks was removed in 3.0 (memory is bounded automatically), so drop it. "
    "Recompute phase-lag-index measures that 2.x computed with blocks >= 1: "
    "blocks=1 reversed the sign of phase_lag_index and weighted_phase_lag_index, "
    "and larger values did not return valid results."
)
# Where 2.x took blocks by position, which is dtype's position now.
_BLOCKS_POSITION = (
    "In 2.x, blocks took this position (third in from_multitaper, fifth in Connectivity)."
)


def _validated_complex_dtype(dtype: Any) -> Any:
    """``dtype`` if it is a complex dtype; otherwise a ``TypeError``.

    A real dtype (including ``None``, which NumPy reads as float64) would cast
    the Fourier coefficients to real values and silently drop every phase. A
    ``None`` or an integer is most likely 2.x's ``blocks``, passed in the
    position ``dtype`` now has, so the message says so.
    """
    try:
        kind = np.dtype(dtype).kind
    except TypeError:
        kind = None
    if kind != "c":
        msg = (
            f"dtype must be a complex dtype such as np.complex128 or np.complex64, "
            f"got {dtype!r}; a real dtype would discard the imaginary part of the "
            "Fourier coefficients."
        )
        if dtype is None or (
            isinstance(dtype, (int, np.integer)) and not isinstance(dtype, bool)
        ):
            msg += f" {_BLOCKS_POSITION} {_BLOCKS_REMOVED} Pass dtype by keyword."
        raise TypeError(msg)
    return dtype


# How a single observation fixes the maximized imaginary coherency and the
# multivariate interaction measure (1, and 1 or 2, for a group pair with more
# than two signals in total), for the single-observation warning.
_RANK_ONE_GROUP_CONSEQUENCE = (
    "the cross-spectral matrix has rank one and the value is fixed by the "
    "number of signals in each group (for two single-signal groups, by one "
    "phase difference)"
)
# What adds observations along each axis an expectation can average, for the
# same warning.
_MORE_OBSERVATIONS = {
    "time": "more time windows",
    "trials": "more trials",
    "tapers": (
        "more tapers (for Multitaper, a larger time_halfbandwidth_product or "
        "n_tapers; for MorletWavelet, smoothing_time, which collects neighboring "
        "coefficients on that axis)"
    ),
}
# And how it fixes the canonical coherency.
_RANK_ONE_CANONICAL_CONSEQUENCE = (
    "the leading canonical coherency of every group pair is mathematically "
    "forced to magnitude 1 (apparent perfect connectivity)"
)

# Measures whose values are magnitudes in [0, 1], so their Fisher (atanh)
# jackknife interval is clamped at 0 on the way back.
_NONNEGATIVE_MAGNITUDE_MEASURES = frozenset({"phase_locking_value", "imaginary_coherence"})

EXPECTATION_AXES = {
    "time": (0,),
    "trials": (1,),
    "tapers": (2,),
    "time_trials": (0, 1),
    "time_tapers": (0, 2),
    "trials_tapers": (1, 2),
    "time_trials_tapers": (0, 1, 2),
}

# Per-observation functions of Im(S_ij) averaged by the phase-lag-index family;
# see Connectivity._imaginary_cross_spectrum_moments. They look ``xp`` up when
# called, not at import, so a swapped backend (the device-emulation tests) applies.
_IMAGINARY_MOMENTS: dict[str, Callable[[BackendArray], BackendArray]] = {
    "sign": lambda imaginary: xp.sign(imaginary),
    "nonzero": lambda imaginary: xp.abs(xp.sign(imaginary)),
    "imaginary": lambda imaginary: imaginary,
    "absolute": lambda imaginary: xp.abs(imaginary),
    "squared": lambda imaginary: imaginary**2,
}
# Im(X_j conj(X_i)) = -Im(X_i conj(X_j)), exactly in floating point too, so each
# moment is exactly antisymmetric (-1) or symmetric (+1) across the signal pair.
_IMAGINARY_MOMENT_PAIR_SYMMETRY = {
    "sign": -1,
    "nonzero": 1,
    "imaginary": -1,
    "absolute": 1,
    "squared": 1,
}


@dataclass(frozen=True)
class MultivariateConnectivityResult:
    """Component-resolved multivariate connectivity and spatial projections.

    ``scores`` has shape ``(..., frequency, connection, component)``. Filters
    and patterns, when present, append ``(side, signal)`` where side 0 is the
    connection's source (first) group and side 1 its target (second) group;
    the xarray wrappers label them ``side="source"`` and ``side="target"``.
    Entries for signals outside a side's group are NaN. ``connections``
    contains the corresponding group-label pair for each connection and
    ``group_membership`` has shape ``(group, signal)``.

    Attributes
    ----------
    method : str
        Name of the measure that produced the result.
    scores : NDArray[number], shape (..., frequency, connection, component)
        Per-component connectivity. Complex for ``canonical_coherency``
        (magnitude times ``exp(-1j * phi)``), real for MIC. A component a
        connection cannot supply (its smaller group has fewer channels than the
        requested ``n_components``) is NaN.
    connections : NDArray, shape (connection, 2)
        The ``(source_group_label, target_group_label)`` pair for each
        connection, the wrappers' ``connection_source_group`` and
        ``connection_target_group`` coordinates.
    group_labels : NDArray, shape (group,)
        Sorted unique group labels.
    group_membership : NDArray[bool], shape (group, signal)
        ``True`` where a signal belongs to a group.
    filters : NDArray[floating] or None, shape (..., frequency, connection, component, side, signal)
        Spatial filters mapping channel data to each component; NaN outside a
        side's group.
    patterns : NDArray[floating] or None, same shape as ``filters``
        Haufe-style patterns (``within-group real CSD @ filter``) mapping each
        component back to channel space.
    """

    method: str
    scores: NDArray[np.number]
    connections: NDArray[Any]
    group_labels: NDArray[Any]
    group_membership: NDArray[np.bool_]
    filters: NDArray[np.floating] | None = None
    patterns: NDArray[np.floating] | None = None


def _is_real_number(value: Any) -> bool:
    """Whether ``value`` is a Python or NumPy real number other than a boolean."""
    return not isinstance(value, bool) and isinstance(
        value, (int, float, np.integer, np.floating)
    )


def _validated_regularization(value: Any) -> float:
    """Return a finite non-negative scalar regularization factor."""
    message = f"regularization must be a finite non-negative scalar, got {value!r}."
    if not _is_real_number(value):
        raise ValueError(message)
    regularization = float(value)
    if not np.isfinite(regularization) or regularization < 0:
        raise ValueError(message)
    return regularization


def _validated_shrinkage(value: Any) -> float | Literal["auto"] | None:
    """Return ``None``, ``"auto"``, or a shrinkage intensity in [0, 1]."""
    if value is None:
        return None
    message = (
        "shrinkage must be None (no shrinkage), 'auto' (estimate the intensity "
        "for each time window and frequency), or a number from 0 to 1 (a fixed "
        f"intensity), got {value!r}."
    )
    if isinstance(value, str):
        if value == "auto":
            return "auto"
        msg = f"{message} Strings are case-sensitive."
        raise ValueError(msg)
    if isinstance(value, bool):
        msg = f"{message} Booleans are not accepted."
        raise ValueError(msg)
    if not _is_real_number(value) or not 0.0 <= float(value) <= 1.0:  # rejects NaN
        raise ValueError(message)
    return float(value)


def _validated_rank(rank: int | None) -> int | None:
    """Return a positive-integer rank or None, rejecting other values."""
    if rank is not None and not is_positive_integer(rank):
        msg = f"rank must be a positive integer or None, got {rank!r}."
        raise ValueError(msg)
    return rank


# Peak workspace cap for the phase-lag-index family's observation-level signal
# tiles. The tile loop holds two float buffers of this many elements, plus, one
# at a time, the three boolean comparison temporaries of the same count that
# each tile's sign and NaN reductions make. The final reduced signal-by-signal
# result is unavoidable, but the large trial/taper/time-resolved outer product
# is never materialized in full.
PHASE_LAG_INDEX_MAX_WORKSPACE_ELEMENTS = 16_000_000
# Fewest averaged observations (``Connectivity.n_observations``) at which, without
# observation weights, the first phase-lag measure reduces all the moments from
# one tile pass instead of only the requested ones. A tile holds n_observations
# times as many values as each reduced moment, so with many observations the
# tile formation dominates and two extra reductions are nearly free, while with
# few (e.g. time-resolved single-trial spectra) writing and retaining two extra
# window-resolved moments costs more than re-forming the tile for a later
# measure. Measured on 32 signals, 100 FFT bins, CPU, with windows x
# observations held near 3600: a lone phase_lag_index computing all four was
# 1.49x (3 observations), 1.33x (8), 1.23x (16), 1.15x (32), 1.08x (64) and
# 1.03-1.07x (128-500) the time of computing only its two, while the four
# phase-lag measures together took 0.80x (3) to 0.42x (500) the time. 64 is
# the smallest count at which a lone measure costs at most 10% more and is
# also faster than before one-pass reduction (0.75-0.79 s vs 0.89 s).
PHASE_LAG_ALL_MOMENTS_MIN_OBSERVATIONS = 64
# Element cap for the complex coefficients gathered per chunk of channel pairs
# by ``Connectivity._subset_cross_spectral_matrix``. The gather and its
# observation-major copy are each at most this size (32 MB at complex128), so
# peak workspace stays bounded however many pairs are requested.
SUBSET_CROSS_SPECTRUM_MAX_WORKSPACE_ELEMENTS = 2_000_000

# Machine epsilons of E[|Im S_ij|] relative to sqrt(P_i P_j) below which a pair
# has no phase lag. In-phase signals leave only rounding noise, measured at up
# to about 1 eps; a genuine lag of this size (~4e-15 rad) is not resolvable.
_ZERO_PHASE_LAG_EPSILONS = 16

# Largest relative gap between the diagonal-noise-power denominator that
# ``directed_coherence`` uses and the true power spectral density, tolerated
# before it warns that its diagonal-noise-covariance assumption is violated (see
# the note in that method). A dimension-aware criterion: unlike a pairwise
# correlation threshold, it catches many weakly-but-jointly correlated sources
# whose cross-power still omits a large fraction of the true power. Non-parametric
# estimation of a truly diagonal covariance leaves a small discrepancy from
# finite-sample noise, so the threshold sits above that floor: only a material
# omission (>= 10% of the true power) triggers the warning.
DIRECTED_COHERENCE_DISCREPANCY_TOLERANCE = 0.1

# Signature and return type of a decorated measure, preserved by its decorators.
_P = ParamSpec("_P")
_R = TypeVar("_R")


def _rejects_removed_blocks(caller: str) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Reject 2.x's ``blocks=`` keyword with a message that says to drop it.

    Its released position is ``dtype``'s now, where
    :func:`_validated_complex_dtype` recognizes it. 2.x's ``blocks`` by
    position with ``dtype`` by keyword would bind ``dtype`` twice, so that
    form is recognized here, before Python reports it without naming
    ``blocks``.
    """

    def decorate(function: Callable[_P, _R]) -> Callable[_P, _R]:
        dtype_position = list(inspect.signature(function).parameters).index("dtype")

        @wraps(function)
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            if "blocks" in kwargs:
                msg = f"{caller} got blocks={kwargs['blocks']!r}. {_BLOCKS_REMOVED}"
                raise TypeError(msg)
            if "dtype" in kwargs and len(args) > dtype_position:
                msg = (
                    f"{caller} got {args[dtype_position]!r} by position where dtype "
                    f"is, and dtype by keyword. {_BLOCKS_POSITION} {_BLOCKS_REMOVED} "
                    "Pass dtype by keyword only."
                )
                raise TypeError(msg)
            return function(*args, **kwargs)

        return wrapper

    return decorate


def _asnumpy(connectivity_measure: Callable[_P, _R]) -> Callable[_P, _R]:
    """Transform cupy array to numpy array.

    If cupy is not installed, then return original.

    Parameters
    ----------
    connectivity_measure : callable
        Connectivity measure function to wrap.

    Returns
    -------
    callable
        Wrapped function that converts output to numpy.

    """

    @wraps(connectivity_measure)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        measure = connectivity_measure(*args, **kwargs)
        if measure is None:
            return measure
        # The type checker analyzes the NumPy path, where this is the identity.
        return cast(_R, to_numpy(measure))

    return wrapper


def _source_first(connectivity_measure: Callable[_P, _R]) -> Callable[_P, _R]:
    """Reorder a measure's ``[..., target, source]`` output to ``[..., source, target]``.

    The Wilson-factorized kernels work in the transfer function's native
    layout, where row ``i`` collects the inflow to signal ``i``. The public
    measures built on them (the spectral Granger family and the transfer
    function / MVAR measures) return the transpose so that ``[..., i, j]``
    reads ``i -> j`` like the labeled wrapper's ``sel(source=i, target=j)``.
    The lead/lag measures are computed source first and do not use it.

    Apply it above :func:`_asnumpy`: swapping the host array is a free view,
    whereas a swapped device array would force a contiguous device copy on
    transfer.
    """

    @wraps(connectivity_measure)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        native = connectivity_measure(*args, **kwargs)
        return cast(_R, np.swapaxes(cast(NDArray[Any], native), -1, -2))

    return wrapper


# Why a measure that needs samples is refused for an exact spectrum.
_ExactSpectrumRefusal = Literal[
    "per_observation", "significance_gate", "sampling_error", "surrogate_null", "shrinkage"
]
_SIMULATE_INSTEAD = (
    "simulate data from the model (spectral_connectivity.simulate.simulate_MVAR) "
    "and estimate it with Multitaper"
)


def _exact_spectrum_refusal_message(method: str, reason: _ExactSpectrumRefusal) -> str:
    """What to say when ``method`` is called on an exact spectrum, for ``reason``."""
    intro = (
        f"{method} is not defined for an exact spectrum (is_exact_spectrum is "
        "True, as for VARSpectrum), whose coefficients are one factor of the "
        "model's cross-spectral matrix, not samples."
    )
    if reason == "sampling_error":
        return (
            f"{intro} {method} estimates sampling error by leaving out one "
            "observation at a time, and an exact spectrum has no sampling error "
            f"to estimate. For the spread of an estimate, {_SIMULATE_INSTEAD}, and "
            f"apply {method} to that."
        )
    if reason == "surrogate_null":
        return (
            f"{intro} {method} compares an estimate with surrogates of the "
            "samples, and an exact spectrum has no sampling error for a null "
            f"distribution to describe. To test a VAR model's connectivity, "
            f"{_SIMULATE_INSTEAD}, and apply {method} to that."
        )
    if reason == "shrinkage":
        return (
            f"{intro} {method} estimates each coherency's sampling variance "
            "across the observations, and an exact spectrum has none to shrink "
            "against. Its partial coherence is exact without shrinkage; use "
            "partial_coherence() or a fixed shrinkage between 0 and 1, or "
            f"{_SIMULATE_INSTEAD}."
        )
    if reason == "significance_gate":
        return (
            f"{intro} {method} uses only the frequencies whose coherence passes "
            "a significance test whose degrees of freedom are n_observations, "
            "which here counts the columns of that factor, so which frequencies "
            "it keeps would be arbitrary. phase_slope_index and coherence_phase "
            f"are computed from the spectral matrix alone; or {_SIMULATE_INSTEAD}."
        )
    return (
        f"{intro} {method} averages a function of each trial/taper "
        "observation (the debiased measures and pairwise phase consistency "
        "also count them), so its value would depend on an arbitrary choice "
        "of factor. Use a measure of the cross-spectral matrix (the coherence "
        "family, partial coherence, the group and global measures, "
        "phase_slope_index, and the spectral Granger and transfer-function "
        f"families), or {_SIMULATE_INSTEAD}."
    )


# How jackknife learns that observations are segments in time, not tapers.
_SEGMENTS_ADVICE = (
    "If the observations are segments of each trial in time instead (Welch), "
    "build with Connectivity.from_transform(Welch(...)), which records it, to "
    "leave out one segment at a time."
)
# What jackknife says when only tapers could be left out for frequency_bands.
_NO_BAND_JACKKNIFE_ADVICE = (
    "A band statistic of tapers of the same samples has no valid jackknife with "
    "fewer than 3 trials: collect at least 3 trials, or jackknife each frequency "
    "without frequency_bands, and do not average those intervals or their "
    "replicates over a band."
)


def _refused_for_exact_spectrum(
    reason: _ExactSpectrumRefusal,
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Make a method raise, for ``reason``, on a ``Connectivity`` of an exact spectrum.

    For the measures that use each trial/taper observation or count them, and
    for ``jackknife`` and ``surrogate_test``, which resample observations.
    Apply it outermost, so
    nothing is computed before the refusal; the measure registry's
    ``defined_for_exact_spectrum`` records which measures carry it.
    """

    def decorate(method: Callable[_P, _R]) -> Callable[_P, _R]:
        @wraps(method)
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            if cast("Connectivity", args[0])._is_exact_spectrum:
                raise ValueError(_exact_spectrum_refusal_message(method.__name__, reason))
            return method(*args, **kwargs)

        return wrapper

    return decorate


class DirectedOrientationWarning(UserWarning):
    """A directed measure's orientation differs from spectral_connectivity 2.x.

    Emitted by the ``Connectivity`` methods that returned
    ``[..., target, source]`` in 2.x and now return ``[..., source, target]``,
    and by :func:`~spectral_connectivity.multitaper_connectivity` and
    :func:`~spectral_connectivity.connectivity_to_xarray` when they compute
    pairwise or subset spectral Granger prediction, whose
    ``sel(source=a, target=b)`` was ``b -> a`` in 2.x (the 2.x wrapper rejected
    the transfer-function measures).
    Code written for 2.x keeps running but reads the opposite direction. This
    warning is temporary. It appears once per Python session for each measure:
    once for calls through the wrappers (about ``sel(source, target)``) and
    once for calls to the ``Connectivity`` method (about the array's axes).
    ``warnings.simplefilter("always", DirectedOrientationWarning)`` shows it at
    every call; a call made while it is ignored does not count as shown.
    Silence it, without hiding other warnings, with
    ``warnings.filterwarnings("ignore", category=DirectedOrientationWarning)``.

    By design, a bare ``warnings.catch_warnings(record=True)`` with no filter
    records nothing once the notice has been shown; add
    ``warnings.simplefilter("always", DirectedOrientationWarning)`` inside it
    (``pytest.warns`` does the equivalent).
    """


class SidednessWarning(UserWarning):
    """A result is one-sided now, unlike in spectral_connectivity 2.x.

    Emitted by :meth:`Connectivity.power` when it is called without
    ``one_sided`` on two-sided coefficients. Since 3.0 it returns the one-sided
    density, with the interior frequency bins doubled, where 2.x returned the
    two-sided density, so code written for 2.x keeps running and gets values
    twice as large away from 0 Hz and Nyquist. Pass ``one_sided=True`` or
    ``one_sided=False`` to the call to silence it.

    Also emitted by :meth:`Connectivity.global_coherence` on two-sided
    coefficients: since 3.0 it returns only the non-negative frequencies,
    labelled by :attr:`Connectivity.frequencies`, where 2.x returned every bin
    of the two-sided transform, labelled by :attr:`Connectivity.all_frequencies`.

    This warning is temporary. Each of the two notices appears once per Python
    session; ``warnings.simplefilter("always", SidednessWarning)`` shows
    them at every call, and a call made while it is ignored does not count as
    shown. Silence them with
    ``warnings.filterwarnings("ignore", category=SidednessWarning)``.

    By design, a bare ``warnings.catch_warnings(record=True)`` with no filter
    records nothing once the notice has been shown; add
    ``warnings.simplefilter("always", SidednessWarning)`` inside it
    (``pytest.warns`` does the equivalent).
    """


_MIGRATION_GUIDE_URL = "https://spectral-connectivity.readthedocs.io/en/latest/migration.html"
# Every DirectedOrientationWarning and SidednessWarning message starts with
# this, so pyproject.toml can filter them by message: a category filter would
# import the package while pytest reads its configuration, before coverage starts.
_ORIENTATION_WARNING_PREFIX = "Since spectral_connectivity 3.0, "


def _silence_hint(category: type[UserWarning]) -> str:
    """How to silence a migration notice, and to see it at every call."""
    name = f"spectral_connectivity.{category.__name__}"
    return (
        f'Silence this warning with warnings.filterwarnings("ignore", category={name}). '
        "Shown once per Python session; to see it at every call: "
        f'warnings.simplefilter("always", {name}).'
    )


_SILENCE_ORIENTATION_WARNING = _silence_hint(DirectedOrientationWarning)

# The methods that returned [..., target, source] arrays in 2.x. The Granger
# measures added in 3.0 and the lead/lag measures were never target first.
_ORIENTATION_CHANGED_MEASURES = frozenset(
    {
        "pairwise_spectral_granger_prediction",
        "subset_pairwise_spectral_granger_prediction",
        "directed_transfer_function",
        "directed_coherence",
        "partial_directed_coherence",
        "generalized_partial_directed_coherence",
        "direct_directed_transfer_function",
    }
)

# Whether a measure emits its per-call 3.x migration notices
# (DirectedOrientationWarning, SidednessWarning). False while the labeled
# wrapper or a jackknife replicate runs a measure: the wrapper warns about its
# source/target labels instead (and its labeled results state their
# sidedness), and a jackknife warns once, for its full estimate.
_emit_migration_notices: ContextVar[bool] = ContextVar("_emit_migration_notices", default=True)

# True while a labeled wrapper runs a measure on the Connectivity it built,
# whose expectation_type the caller could not choose ("trials_tapers"), so
# advice to change expectation_type is replaced by the call that does.
_expectation_fixed_by_wrapper: ContextVar[bool] = ContextVar(
    "_expectation_fixed_by_wrapper", default=False
)


def _orientation_changed(connectivity_measure: Callable[_P, _R]) -> Callable[_P, _R]:
    """Warn that a measure's array is source first, unlike in 2.x.

    Applied to the methods in ``_ORIENTATION_CHANGED_MEASURES``; remove it with
    the warning in 3.2.
    """

    @wraps(connectivity_measure)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        result = connectivity_measure(*args, **kwargs)
        if _emit_migration_notices.get():
            warn_migration(
                f"{_ORIENTATION_WARNING_PREFIX}{connectivity_measure.__name__} "
                "returns [..., source, target]: [..., i, j] is i -> j. 2.x "
                "returned [..., target, source]; review the indexing and "
                "normalization axes of code written for 2.x. See the migration "
                f"guide: {_MIGRATION_GUIDE_URL}. {_SILENCE_ORIENTATION_WARNING}",
                DirectedOrientationWarning,
            )
        return result

    return wrapper


def _ignore_nan_propagation_warnings(
    connectivity_measure: Callable[_P, _R],
) -> Callable[_P, _R]:
    """Suppress NumPy invalid/divide warnings from expected NaN propagation.

    The directed measures (DTF, PDC and relatives) normalize the transfer
    function or MVAR coefficients by an inflow/outflow sum. When the Wilson
    minimum-phase decomposition fails to converge those inputs are NaN (already
    warned about at decomposition time), and the normalization then emits
    ``invalid value encountered in divide``. Scope the suppression to these
    measures rather than silencing NumPy globally; well-conditioned inputs
    produce no NaN and are unaffected.

    Parameters
    ----------
    connectivity_measure : callable
        Connectivity measure method to wrap.

    Returns
    -------
    callable
        Wrapped method whose numpy arithmetic runs under a scoped errstate.
    """

    @wraps(connectivity_measure)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        with np.errstate(invalid="ignore", divide="ignore"):
            return connectivity_measure(*args, **kwargs)

    return wrapper


# The starts of the warnings a spectral factorization gives when it fails.
# Computed from NaN or Inf coefficients it always fails, and these warnings
# then blame duplicated or collinear channels for what the NaN warning
# already explained.
# (warnings.filterwarnings patterns; the starts are defined where each
# warning is raised.)
_FACTORIZATION_FAILURE_WARNINGS = (
    re.escape(CHOLESKY_FALLBACK_WARNING_START),
    re.escape(NONCONVERGENCE_WARNING_START),
    r"\w+" + re.escape(NAN_PAIRS_WARNING_AFTER_MEASURE),
    re.escape(CONDITIONAL_DEGENERATE_WARNING_START),
    re.escape(BLOCK_DEGENERATE_WARNING_START),
)


def _nonfinite_decomposition_message(measure: str, error: Exception | None = None) -> str:
    """Why ``measure`` cannot be computed from NaN or Inf coefficients; the
    backend's ``error``, when it raised one, is quoted."""
    detail = f" ({error})" if error is not None else ""
    return (
        f"{measure} cannot be computed: the Fourier coefficients contain "
        "NaN or Inf (from NaN or Inf in the time series, if it was "
        "transformed here; see the earlier warning), on which its matrix "
        f"decomposition does not converge{detail}. Interpolate or remove "
        "the NaN or Inf samples, or drop the affected trials or signals, "
        "and compute it again."
    )


def _factorization_explained_by_nonfinite_input(
    directed_measure: Callable[_P, _R],
) -> Callable[_P, _R]:
    """Drop the factorization-failure warnings of a directed measure computed
    from NaN or Inf coefficients, which were already warned about (by the
    transform for its time series, or by ``Connectivity`` for coefficients
    passed directly). Finite coefficients keep every warning."""

    @wraps(directed_measure)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        connectivity = cast("Connectivity", args[0])
        with _user_warnings_ignored(
            _FACTORIZATION_FAILURE_WARNINGS, not connectivity._coefficients_are_finite
        ):
            return directed_measure(*args, **kwargs)

    return wrapper


@contextlib.contextmanager
def _user_warnings_ignored(messages: Sequence[str], ignore: bool) -> Iterator[None]:
    """Ignore the ``UserWarning``\\ s whose messages match ``messages`` (regular
    expressions) inside the block, if ``ignore``; otherwise change nothing.

    It uses :class:`warnings.catch_warnings`, which changes the process-wide
    warning filters while the block runs and restores them after: it is not
    thread-safe (another thread's matching warnings are dropped meanwhile, and
    one thread restoring the filters can undo another's), and changing the
    filters invalidates each module's record of warnings already shown, so a
    warning shown once under the "default" action can be shown again. Only
    NaN or Inf coefficients (``ignore``) take this path;
    finite input never touches the filters (nor do the jackknife replicates
    of a bin with too few kept observations, whose single-observation
    warnings it ignores, on any other input). Recording the warnings and
    re-emitting the others would need ``catch_warnings`` as well, so it would
    not be safer.
    """
    if not ignore:
        yield
        return
    with warnings.catch_warnings():
        for message in messages:
            warnings.filterwarnings("ignore", message=message, category=UserWarning)
        yield


_ABSENT = object()

# Where the warnings of bins that zero weights leave a single observation say
# so; the jackknife ignores them in its replicates (see Connectivity.jackknife).
_SINGLE_OBSERVATION_IN_SOME_BINS = (
    "in some bins, where observation_weights give only one observation a nonzero "
    "weight (see n_valid_observations)"
)
_SINGLE_KEPT_OBSERVATION = "observation_weights keep a single observation in"
_SINGLE_KEPT_OBSERVATION_WARNINGS = (
    ".*" + re.escape(f"computed from a single observation {_SINGLE_OBSERVATION_IN_SOME_BINS}"),
    ".*" + re.escape(_SINGLE_KEPT_OBSERVATION),
)


def _optional_transform_attribute(transform: Any, name: str, default: Any) -> Any:
    """Read an optional ``SpectralTransform`` attribute, or ``default`` if absent.

    Unlike ``getattr(transform, name, default)``, an ``AttributeError`` raised
    inside a property the transform does define propagates instead of being
    taken for the attribute's absence, which would silently apply the default.
    """
    try:
        return getattr(transform, name)
    except AttributeError:
        if inspect.getattr_static(transform, name, _ABSENT) is not _ABSENT:
            raise
        return default


def _transform_flag(transform: Any, name: str, default: bool) -> bool:
    """Read an optional boolean ``SpectralTransform`` attribute strictly."""
    return _validated_flag(
        f"transform.{name}", _optional_transform_attribute(transform, name, default)
    )


# A frequency grid's bins are trusted to one part in a million of their
# spacing (plus the rounding of their stored values; see _last_place): the
# uniform-grid test, the FFT-order check, and whether a bin lies on a
# frequency bound all allow this much.
_GRID_SPACING_RTOL = 1e-6


def _last_place(values: Any) -> NDArray[np.float64]:
    """Distance from each value to the next representable one, in its stored precision.

    The unit in the last place of a float32 value is that of float32, of a
    float64 value that of float64; a non-floating value (an integer bound) is
    taken as float64. :func:`_uniform_grid_spacing` and
    :func:`_frequency_interval_mask` allow two of these units per stored value,
    so a grid's uniformity and the bins a bound names follow the same rounding.

    Parameters
    ----------
    values : array-like or scalar
        Values on the host or the device.

    Returns
    -------
    NDArray[float64], same shape as ``values``
    """
    stored = np.asarray(to_numpy(values))
    if not np.issubdtype(stored.dtype, np.floating):
        stored = stored.astype(float)
    last_place: NDArray[np.float64] = np.spacing(np.abs(stored)).astype(float)
    return last_place


def _uniform_grid_spacing(frequencies: Any) -> float | None:
    """Step of a uniformly spaced frequency grid, or ``None`` if it is not one.

    The measures that combine adjacent bins, the FFT-order check of two-sided
    frequencies, and the ``frequency_bin_spacing`` recorded on results share
    this definition. Every step must lie within one
    part in a million of the mean step, plus the rounding of the two values it
    separates in the coordinate's own precision (two units in the last place
    each). A uniform grid stored as float32 is therefore still uniform, however
    fine its step, while a grid rounded to a few decimals is uniform only if
    that rounding is within one part in a million of the step. The step is
    negative for a decreasing grid, and a grid with fewer than two values or
    with a non-finite value has none.

    Parameters
    ----------
    frequencies : array, shape (n_frequencies,)
        The frequency coordinate, on the host or the device.

    Returns
    -------
    float or None
        The mean step, in the coordinate's units.
    """
    stored = to_numpy(frequencies)
    if stored.size < 2 or not np.all(np.isfinite(stored)):
        return None
    last_place = _last_place(stored)
    values = stored.astype(float)
    spacing = float((values[-1] - values[0]) / (values.size - 1))
    rounding = 2 * (last_place[:-1] + last_place[1:])
    if np.all(
        np.abs(np.diff(values) - spacing) <= _GRID_SPACING_RTOL * abs(spacing) + rounding
    ):
        return spacing
    return None


def _nearest_bin_gap(grid: NDArray[np.float64]) -> NDArray[np.float64]:
    """Distance from each bin to its nearest neighbouring bin, in any order.

    Zero for a single-bin grid, which has no neighbour.

    Parameters
    ----------
    grid : NDArray[float64], shape (n_frequencies,)

    Returns
    -------
    NDArray[float64], shape (n_frequencies,)
    """
    order = np.argsort(grid, kind="stable")
    steps = np.diff(grid[order])
    no_neighbour = np.array([np.inf])
    nearest = np.minimum(
        np.concatenate((no_neighbour, steps)), np.concatenate((steps, no_neighbour))
    )
    gap = np.empty_like(grid)
    gap[order] = np.where(np.isfinite(nearest), nearest, 0.0)
    return gap


def _bins_on_bounds(values: NDArray[Any], bounds: Sequence[Any]) -> list[NDArray[np.bool_]]:
    """Which frequency bins lie on each bound, by the rule of :func:`_frequency_interval_mask`.

    Parameters
    ----------
    values : array, shape (n_frequencies,)
        The frequency grid on the host, in its stored precision.
    bounds : sequence of float
        Each bound in its stored precision (a float32 scalar stays float32).

    Returns
    -------
    list of NDArray[bool], shape (n_frequencies,)
        One mask per bound.
    """
    grid = values.astype(np.float64)
    grid_allowance = _GRID_SPACING_RTOL * _nearest_bin_gap(grid) + 2 * _last_place(values)
    return [
        np.abs(grid - float(bound)) <= grid_allowance + 2 * _last_place(bound)
        for bound in bounds
    ]


def _frequency_interval_mask(
    label: str, bounds: Any, frequencies: Any
) -> tuple[NDArray[np.bool_], tuple[float, float]]:
    """Validate ``(low, high)`` bounds and return the frequency bins they select.

    The package's frequency-selection rule, used by ``frequency_range`` (the
    wrappers' crop and the measures that consume a frequency interval) and by
    each of ``frequency_bands``. It keeps every bin from ``low`` to ``high``,
    including a bin lying on either bound. Bin ``f`` lies on bound ``b`` when
    ``abs(f - b) <= 1e-6 * gap(f) + 2 * (ulp(f) + ulp(b))``: ``gap(f)`` is the
    distance to its nearest neighbouring bin (:func:`_nearest_bin_gap`, zero
    for a single bin) and ``ulp`` is :func:`_last_place` of each value in its
    own stored precision (the grid's dtype for ``f``; float64 for a Python
    ``float`` or integer bound, float32 for a float32 bound). These are the
    allowances of :func:`_uniform_grid_spacing`, with the local gap in place of
    the mean step so irregular grids are judged bin by bin. A bound 1e-3 of
    the spacing away from a bin does not select it.

    A decimal bound names a bin only when the bin's stored value has drifted
    from that decimal by at most one part in a million of its spacing (plus
    rounding). Float32 coordinates, ``0.1 * k``, ``np.linspace``,
    ``rfftfreq``, and ``np.arange(1.0, 100.0, 0.1)`` stay within that;
    ``np.arange(1000.0, 1100.0, 0.001)``, whose bins drift by up to 2.4e-6 of
    its spacing, and float32 cumulative sums do not.

    The deprecated ``frequencies_of_interest`` does not use this rule; see
    :func:`_frequencies_of_interest_mask`.

    Parameters
    ----------
    label : str
        The argument or band named in error messages.
    bounds : (float, float)
        Lower and upper bound, in the units of ``frequencies``.
    frequencies : array, shape (n_frequencies,)
        The frequency grid, on the host or the device.

    Returns
    -------
    mask : NDArray[bool], shape (n_frequencies,)
        Host mask of the selected bins.
    bounds : (float, float)
        The validated bounds.

    Raises
    ------
    ValueError
        If ``bounds`` is not two finite values with ``low <= high``, or
        selects no bin.
    """
    values = to_numpy(frequencies)
    invalid = (
        f"{label} must be two finite values (low, high) with low <= high, got {bounds!r}."
    )
    try:
        stored_lower, stored_upper = bounds
        lower, upper = float(stored_lower), float(stored_upper)
    except (TypeError, ValueError) as error:
        raise ValueError(invalid) from error
    if not (np.isfinite(lower) and np.isfinite(upper)) or lower > upper:
        raise ValueError(invalid)

    grid = values.astype(np.float64)
    on_lower, on_upper = _bins_on_bounds(values, (stored_lower, stored_upper))
    mask: NDArray[np.bool_] = ((lower < grid) | on_lower) & ((grid < upper) | on_upper)
    if not mask.any():
        msg = (
            f"{label} ({lower:g}, {upper:g}) contains no frequency bin (both endpoints "
            f"included); the {grid.size} bins span {grid.min():g} to {grid.max():g}. "
            "Widen it."
        )
        raise ValueError(msg)
    return mask, (lower, upper)


def _require_fft_order(frequencies: Any) -> None:
    """Raise unless two-sided ``frequencies`` are uniformly spaced in FFT order.

    Two-sided coefficients are folded onto their first ``n // 2 + 1`` bins as
    the non-negative frequencies, so any other layout (e.g. ``rfft`` output
    labelled two-sided, or an ``fftshift``-ed spectrum) would silently drop or
    mislabel frequencies.
    """
    values = to_numpy(frequencies)
    if values.size == 0:
        return
    if values.size == 1:
        in_order = bool(values[0] == 0.0)
    else:
        # FFT order puts the zero bin first, and fftshift then sorts the bins
        # into one increasing, uniformly spaced grid.
        spacing = _uniform_grid_spacing(np.fft.fftshift(values))
        in_order = (
            spacing is not None
            and spacing > 0
            and abs(float(values[0])) <= _GRID_SPACING_RTOL * spacing
        )
    if not in_order:
        values = values.astype(float)
        msg = (
            f"frequencies must be uniformly spaced in standard FFT order (zero and "
            f"positive bins followed by negative bins, as numpy.fft.fftfreq); got "
            f"{values[:3]} ... {values[-2:]}. The first n // 2 + 1 bins of a "
            f"two-sided spectrum are taken as its non-negative half, so any other "
            f"layout would drop or mislabel frequencies. If the coefficients hold "
            f"only non-negative frequencies (e.g. numpy.fft.rfft or a wavelet "
            f"transform), mark them one-sided (is_one_sided=True); otherwise order "
            f"them as numpy.fft.fft does."
        )
        raise ValueError(msg)


class Connectivity:
    """
    Compute functional and directed connectivity measures from spectral data.

    This class provides a comprehensive suite of connectivity analysis methods
    based on cross-spectral matrices derived from Fourier-transformed time series.
    Methods range from basic coherence to advanced Granger causality measures.

    Parameters
    ----------
    fourier_coefficients : NDArray[complexfloating], shape (n_time_windows, n_trials, n_tapers, n_frequencies, n_signals)
        Complex-valued Fourier coefficients from spectral analysis. Must be
        two-sided (positive and negative frequencies) for Granger methods.
        Usually obtained from multitaper or other spectral estimation methods.
        **Validation**: Must be 5-dimensional and complex. The measures relating
        signals need at least 2 signals (``power()`` and
        ``cross_spectral_density()`` take one). NaN or Inf values are accepted
        with a ``UserWarning`` and propagate to the estimates computed from
        them, except that ``canonical_coherence`` and ``global_coherence``,
        whose decompositions do not converge on them, raise ``ValueError``.
    expectation_type : {"trials_tapers", "trials", "tapers", "time",
        "time_trials", "time_tapers", "time_trials_tapers"},
        default="trials_tapers"
        Which axes of the coefficients to average over. The others are kept,
        ahead of the measure's own axes (see "Result shape by
        expectation_type" in Notes):

        - "trials_tapers": average over trials and tapers (most common); keep
          the time-window axis
        - "trials": average over trials; keep the time-window and taper axes
        - "tapers": average over tapers; keep the time-window and trial axes
        - "time": average over time windows; keep the trial and taper axes
        - "time_trials", "time_tapers", "time_trials_tapers": average over
          every named axis; keep the rest
    frequencies : NDArray[floating], shape (n_frequencies,), optional
        Frequency values in Hz corresponding to FFT bins. If None, uses
        normalized frequencies.
    time : NDArray[floating], shape (n_time_windows,), optional
        Time values in seconds for each time window. If None, uses indices.
    dtype : np.dtype, default=complex128
        Complex data type for internal computations. Should match input
        precision. A real dtype (or ``None``) raises ``TypeError``: it would
        discard the coefficients' imaginary parts.
    minimum_phase_tolerance : float, default=1e-8
        Relative convergence tolerance for the Wilson minimum-phase
        factorization used by the directed measures (spectral Granger, DTF,
        PDC, and relatives).
    minimum_phase_max_iterations : int, default=500
        Maximum Wilson iterations. Near-singular cross-spectral matrices (highly
        correlated channels) can need several hundred iterations; if the
        directed measures return NaN with a non-convergence warning, increase
        this value. The factorization returns early once every sub-spectrum has
        converged, so a large ceiling is cheap for well-conditioned data.
    is_one_sided : bool, default=False
        Whether coefficients contain only non-negative frequencies. One-sided
        transforms are returned without FFT half-spectrum slicing or power
        doubling and cannot be used by Wilson-factorized directed measures.
    observation_weights : ndarray, optional
        Finite non-negative weights with shape ``(time, trial, taper,
        frequency, 1)``. They are applied to every expectation and shared
        across signals. Transform constructors supply these automatically when
        smoothing uses a non-uniform kernel, masks invalid edge estimates, or
        (``Multitaper(artifact_samples=...)``) drops trial windows with artifact
        samples. Weights that only drop observations (0, or one common weight
        in each bin) leave each bin the plain mean of the observations it
        keeps, which the measures that count observations count
        (:attr:`n_valid_observations`).
    observations_are_independent : bool, default=True
        Whether the trial/taper observations are statistically independent.
        Transform constructors set this ``False`` when the observation axis
        holds correlated estimates (a ``MorletWavelet`` smoothing neighborhood,
        or ``Welch`` segments overlapping by more than half). Measures whose
        finite-sample corrections or null distributions count observations
        warn in that case, and ``jackknife`` refuses to leave out tapers
        (``leave_out="trial"``, which leaves out whole trials, still works).
    time_bins_are_independent : bool, default=True
        Whether the time bins may be counted as independent observations when
        the expectation averages over time. Transform constructors set this
        ``False`` when successive time bins are correlated (``Multitaper`` or
        ``ShortTimeFourierTransform`` windows overlapping by more than half, or
        ``MorletWavelet`` samples closer than four wavelet standard
        deviations). The measures that count observations then warn for
        expectations that include time; it has no effect otherwise.
    time_bins_are_disjoint : bool, default=False
        Whether successive time windows share no samples (a step of at least
        the window length), which :meth:`from_transform` reads from
        ``Multitaper`` and ``ShortTimeFourierTransform``. Only then does
        :meth:`surrogate_test` randomize each averaged time window as a
        separate unit: windows that overlap, even by half, have correlated
        coefficients. The default assumes that coefficients passed directly
        may come from overlapping windows; pass ``True`` for ones that do
        not. It changes no measure.
    half_bandwidth : float, optional
        Half-width of the transform's spectral smoothing, in the units of
        ``frequencies`` (``frequency_resolution / 2`` for ``Multitaper``,
        which :meth:`from_transform` reads from the transform). Within it of
        0 Hz and Nyquist the coefficients are not circular;
        :meth:`surrogate_test` with ``kind="phase_rotation"`` leaves those
        frequencies untested. ``None`` (the default) when unknown.
    is_exact_spectrum : bool, default=False
        Keyword-only. Whether the taper-mean outer product of the coefficients
        is an exact (population) cross-spectral matrix and the observations
        only one factorization of it, not samples, as for
        :class:`~spectral_connectivity.transforms.VARSpectrum`. The measures
        that use each observation or count them (the phase-locking and
        phase-lag families, pairwise phase consistency, ``delay``,
        ``group_delay``), ``jackknife``, ``surrogate_test`` and
        ``partial_coherence(shrinkage="auto")`` then raise ``ValueError``, and the
        expectation must average the tapers.

        .. versionadded:: 3.0

    Attributes
    ----------
    n_observations : int
        Number of trial/taper observations reduced by the expectation. This is
        the raw count, not a weighted effective sample size, and it is not
        reduced for correlated observations (see
        ``observations_are_independent``).
    n_valid_observations : ndarray of int
        The observations with nonzero weight in each bin of a result, the
        count the measures that count observations use.

    See Also
    --------
    spectral_connectivity.transforms.Multitaper : Produce the Fourier
        coefficients this class consumes.
    spectral_connectivity.wrapper.multitaper_connectivity : High-level interface
        returning labeled xarray results with explicit ``source``/``target``
        axes.

    Notes
    -----
    **Array orientation**: pairwise measures end in an ``(n_signals, n_signals)``
    pair (``(n_groups, n_groups)`` for group measures, ordered as the returned
    labels), indexed source first. For the directed measures (the spectral
    Granger family, directed transfer function, directed coherence,
    (generalized) partial directed coherence, and direct directed transfer
    function) ``result[..., i, j]`` is the influence of signal ``i`` on signal
    ``j`` (``i -> j``). For the lead/lag measures (``directed_phase_lag_index``,
    ``phase_slope_index``, ``group_delay``, ``delay``) and the antisymmetric
    phase measures (``coherence_phase``, ``imaginary_coherency``,
    ``phase_lag_index``, ``weighted_phase_lag_index``) positive
    ``result[..., i, j]`` (above 0.5 for ``directed_phase_lag_index``) means
    signal ``i`` leads signal ``j``.

    The labeled wrappers :func:`~spectral_connectivity.multitaper_connectivity`
    and :func:`~spectral_connectivity.fourier_connectivity` use the same order:
    ``result.sel(source=a, target=b)`` is ``a -> b`` (or "``a`` leads ``b``").
    Prefer them unless you need this lower-level API.

    **Result shape by expectation_type**: the averaged axes are dropped and
    the others kept, in the order (time windows, trials, tapers), ahead of the
    measure's own axes. For ``magnitude_squared_coherence`` of coefficients
    with 4 time windows, 3 trials, 5 tapers, 126 non-negative frequencies, and
    2 signals:

    ========================  ==============================  =================
    expectation_type          leading axes kept               result shape
    ========================  ==============================  =================
    ``"trials_tapers"``       (n_time_windows)                (4, 126, 2, 2)
    ``"trials"``              (n_time_windows, n_tapers)      (4, 5, 126, 2, 2)
    ``"tapers"``              (n_time_windows, n_trials)      (4, 3, 126, 2, 2)
    ``"time"``                (n_trials, n_tapers)            (3, 5, 126, 2, 2)
    ``"time_trials"``         (n_tapers)                      (5, 126, 2, 2)
    ``"time_tapers"``         (n_trials)                      (3, 126, 2, 2)
    ``"time_trials_tapers"``  ()                              (126, 2, 2)
    ========================  ==============================  =================

    **Return types**: most measures return one array. The group measures,
    which take ``group_labels``, return more:

    - ``canonical_coherence``, ``blockwise_spectral_granger_prediction``,
      ``maximized_imaginary_coherency``, and
      ``multivariate_interaction_measure`` return a tuple ``(values,
      labels)``: ``values`` ends in ``(n_groups, n_groups)``, ordered as
      ``labels``, the sorted unique group labels. Unpack it:
      ``values, labels = conn.canonical_coherence(group_labels)``.
    - ``canonical_coherency`` and ``maximized_imaginary_coherency_components``
      return a :class:`MultivariateConnectivityResult`; read its attributes
      (``result.scores``, ``result.connections``, ``result.filters``, ...)
      rather than unpacking it.
    - ``global_coherence`` (which takes no groups) returns two arrays,
      ``fractions, vectors = conn.global_coherence()``, whose frequency axis,
      like every other measure's, holds the non-negative frequencies (2.x
      returned every FFT bin). It needs an ``expectation_type`` that averages
      the tapers (see its Raises).

    ``group_delay`` returns ``(delay, slope, r_value)`` and :meth:`jackknife` a
    :class:`~spectral_connectivity.statistics.JackknifeResult`. The labeled
    wrappers return an ``xarray.Dataset`` for each of these instead.
    :meth:`surrogate_test` returns a
    :class:`~spectral_connectivity.statistics.SurrogateTestResult`.

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> conn = Connectivity(
    ...     rng.standard_normal((1, 6, 3, 64, 4)) + 1j * rng.standard_normal((1, 6, 3, 64, 4))
    ... )
    >>> group_labels = np.array(["CA1", "CA1", "PFC", "PFC"])
    >>> values, labels = conn.canonical_coherence(group_labels)
    >>> values.shape, labels.tolist()  # (n_time_windows, n_frequencies, n_groups, n_groups)
    ((1, 33, 2, 2), ['CA1', 'PFC'])
    >>> result = conn.canonical_coherency(group_labels)
    >>> result.scores.shape  # (n_time_windows, n_frequencies, n_connections, n_components)
    (1, 33, 1, 1)
    >>> fractions, vectors = conn.global_coherence()
    >>> fractions.shape, vectors.shape  # (n_time_windows, n_frequencies, ...)
    ((1, 33, 1), (1, 33, 4, 1))

    **Coefficients of a complex signal**: the transforms take only real time
    series, whose negative frequencies mirror the positive ones. Two-sided
    coefficients passed here directly may come from a complex signal, whose
    spectrum is not conjugate-symmetric. Then :meth:`power` and
    :meth:`cross_spectral_density` add each negative frequency to its positive
    one (``S(f) + conj(S(-f))``), while every other measure, coherency
    included, describes the non-negative frequencies ``f`` only; the directed
    measures factor the full two-sided spectrum.

    Intermediates shared across measures (the expected cross-spectral matrix,
    power, phase-lag moments, and the minimum-phase factor with the transfer
    function, noise covariance, and MVAR coefficients derived from it) are
    cached on first access. Reassigning ``fourier_coefficients``,
    ``expectation_type``, or ``observation_weights`` automatically invalidates
    these caches, so reusing an instance for new data is safe (constructing a
    new instance is still the clearer choice). Call :meth:`clear_cache` to
    release them while keeping the instance.

    The class supports both CPU (NumPy) and GPU (CuPy) computation depending
    on the SPECTRAL_CONNECTIVITY_ENABLE_GPU environment variable. For Granger
    causality measures, minimum phase decomposition [1]_ is used to estimate
    transfer functions and noise covariances non-parametrically.

    References
    ----------
    .. [1] Dhamala, M., Rangarajan, G., and Ding, M. (2008). Analyzing
           information flow in brain networks with nonparametric Granger
           causality. NeuroImage 41, 354-362.
    .. [2] Bastos, A. M., & Schoffelen, J. M. (2016). A tutorial review of
           functional connectivity analysis methods and their interpretational
           pitfalls. Frontiers in systems neuroscience, 9, 175.

    Examples
    --------
    >>> import numpy as np
    >>> from spectral_connectivity import Connectivity
    >>> rng = np.random.default_rng(0)
    >>> n_times, n_trials, n_tapers, n_freqs, n_signals = 50, 10, 5, 100, 2
    >>> # Create complex coefficients with coherence injected at frequency bin 10
    >>> phase_diff = np.pi / 4  # 45 degree phase difference
    >>> coeffs = (
    ...     rng.standard_normal((n_times, n_trials, n_tapers, n_freqs, n_signals))
    ...     + 1j
    ...     * rng.standard_normal((n_times, n_trials, n_tapers, n_freqs, n_signals))
    ... )
    >>> coeffs[:, :, :, 10, 1] = coeffs[:, :, :, 10, 0] * np.exp(1j * phase_diff)
    >>> conn = Connectivity(coeffs, expectation_type="trials_tapers")
    >>> coherence = conn.magnitude_squared_coherence()
    >>> coherence.shape  # (n_times, non-negative freqs, n_signals, n_signals)
    (50, 51, 2, 2)
    >>> print(f"Peak coherence: {np.max(coherence[:, 10, 0, 1]):.3f}")
    Peak coherence: 1.000
    """

    _observation_weights: BackendArray | None

    @_rejects_removed_blocks("Connectivity()")
    def __init__(
        self,
        fourier_coefficients: NDArray[np.complexfloating],
        expectation_type: str = "trials_tapers",
        frequencies: NDArray[np.floating] | None = None,
        time: NDArray[np.floating] | None = None,
        dtype: DTypeLike = xp.complex128,
        minimum_phase_tolerance: float = 1e-8,
        minimum_phase_max_iterations: int = 500,
        is_one_sided: bool = False,
        observation_weights: NDArray[np.floating] | None = None,
        observations_are_independent: bool = True,
        time_bins_are_independent: bool = True,
        *,
        time_bins_are_disjoint: bool = False,
        half_bandwidth: float | None = None,
        is_exact_spectrum: bool = False,
        _adopt_fourier_coefficients: bool = False,
    ) -> None:
        # fourier_coefficients and expectation_type are validated in their
        # property setters (below), which also clear the cached intermediates so
        # reassigning either on an existing instance cannot serve stale results.
        # _adopt_fourier_coefficients is a private fast path for from_transform:
        # the fft() output is unshared, so it is frozen in place instead of
        # copied, avoiding a transient 2x peak of the largest array. It must not
        # be set for a caller-owned array (see _adopt_fourier_coefficients).
        dtype = _validated_complex_dtype(dtype)
        # First, so the expectation_type and observation_weights setters can
        # check what an exact spectrum allows.
        self._is_exact_spectrum = _validated_flag("is_exact_spectrum", is_exact_spectrum)
        if _adopt_fourier_coefficients:
            self._adopt_fourier_coefficients(fourier_coefficients)
        else:
            self.fourier_coefficients = fourier_coefficients
        self.expectation_type = expectation_type
        self.observation_weights = observation_weights
        # Wilson minimum-phase factorization controls, used by the directed
        # measures. Near-singular cross-spectral matrices can need more than the
        # default iterations to converge; exposing these lets callers recover
        # (the non-convergence warning advises increasing max_iterations).
        self._minimum_phase_tolerance = minimum_phase_tolerance
        self._minimum_phase_max_iterations = minimum_phase_max_iterations
        self._is_one_sided = _validated_flag("is_one_sided", is_one_sided)
        self._observations_are_independent = _validated_flag(
            "observations_are_independent", observations_are_independent
        )
        self._time_bins_are_independent = _validated_flag(
            "time_bins_are_independent", time_bins_are_independent
        )
        self._time_bins_are_disjoint = _validated_flag(
            "time_bins_are_disjoint", time_bins_are_disjoint
        )
        self._half_bandwidth = (
            None if half_bandwidth is None else _validated_half_bandwidth(half_bandwidth)
        )
        # Only from_transform knows it (see observations_are_time_segments).
        self._observations_are_time_segments = False
        # Fill documented defaults when coordinates are omitted: normalized
        # (sampling-frequency-1) FFT frequencies and integer time-window indices.
        # Otherwise coordinate-dependent methods (delay, group_delay,
        # canonical_coherence) would dereference None.
        n_fft_samples = self._fourier_coefficients.shape[-2]
        n_time_windows = self._fourier_coefficients.shape[0]

        # Supplied coordinates must be 1-D, finite, and match the data geometry.
        # A wrong shape (e.g. (n, 1)) or length would silently misalign/drop
        # frequency bins (phase_slope_index) or crash with a broadcasting error
        # (delay, group_delay, canonical_coherence); non-finite coordinates
        # would propagate NaN into the frequency step and delays.
        def _validate_coordinate(name: str, coord: Any, expected_length: int) -> None:
            arr = xp.asarray(coord)
            if arr.ndim != 1:
                msg = f"{name} must be a 1-D array, got shape {tuple(arr.shape)}."
                raise ValueError(msg)
            if arr.shape[0] != expected_length:
                msg = f"{name} must have length {expected_length}, got {arr.shape[0]}."
                raise ValueError(msg)
            if not bool(xp.all(xp.isfinite(arr))):
                msg = f"{name} must contain only finite values."
                raise ValueError(msg)

        if frequencies is not None:
            _validate_coordinate("frequencies", frequencies, n_fft_samples)
            frequency_values = xp.asarray(frequencies)
            if self._is_one_sided and (
                bool(xp.any(frequency_values < 0))
                or (frequency_values.size > 1 and bool(xp.any(xp.diff(frequency_values) <= 0)))
            ):
                msg = "One-sided frequencies must be non-negative and strictly increasing."
                raise ValueError(msg)
            if not self._is_one_sided:
                _require_fft_order(frequency_values)
        if time is not None:
            _validate_coordinate("time", time, n_time_windows)
        if frequencies is None:
            frequencies = (
                xp.linspace(0.0, 0.5, n_fft_samples)
                if self._is_one_sided
                else xp.fft.fftfreq(n_fft_samples)
            )
        time_values = xp.arange(n_time_windows) if time is None else time
        # Frequencies live on the backend (``frequencies`` indexes them with a
        # device index array); time is a host coordinate (see ``time`` users).
        self._frequencies = xp.asarray(frequencies)
        self._dtype = dtype
        self.time = to_numpy(time_values)

    @property
    def observation_weights(self) -> BackendArray | None:
        """Non-negative weights used when averaging spectral observations.

        Weights have shape ``(time, trial, taper, frequency, 1)`` and are
        shared by every signal. A detached, read-only copy is returned so cached
        expectations cannot be invalidated by mutation.
        """
        if self._observation_weights is None:
            return None
        return mark_readonly_if_supported(self._observation_weights.copy())

    @observation_weights.setter
    def observation_weights(self, value: NDArray[np.floating] | None) -> None:
        if value is None:
            self._observation_weights = None
            self.clear_cache()
            return
        if self._is_exact_spectrum:
            msg = (
                "observation_weights cannot be set on an exact spectrum "
                "(is_exact_spectrum is True, as for VARSpectrum). Its tapers are "
                "the columns of one factor of the model's cross-spectral matrix, "
                "and weighting them would change that matrix into an arbitrary "
                "other one (power and coherence included). Leave "
                "observation_weights as None; to weight time windows, change the "
                "models instead."
            )
            raise ValueError(msg)
        weights = xp.asarray(value)
        expected_shape = (*self._fourier_coefficients.shape[:-1], 1)
        if tuple(weights.shape) != expected_shape:
            msg = (
                "observation_weights must have shape "
                f"{expected_shape}, got {tuple(weights.shape)}. Weights must be "
                "shared across signals."
            )
            raise ValueError(msg)
        if not bool(xp.all(xp.isfinite(weights))) or bool(xp.any(weights < 0)):
            msg = "observation_weights must contain only finite, non-negative values."
            raise ValueError(msg)
        real_dtype = self._fourier_coefficients.real.dtype
        self._observation_weights = mark_readonly_if_supported(
            weights.astype(real_dtype, copy=True)
        )
        self.clear_cache()

    def clear_cache(self) -> None:
        """Free the intermediates cached for reuse across measures.

        Measures computed on one instance share intermediates such as the
        expected cross-spectral matrix and the minimum-phase factorization,
        which can each take ``n_frequencies * n_signals**2`` values for every
        observation that ``expectation_type`` leaves unaveraged (each time
        window by default). Call this after the last measure that needs them to
        release the memory while keeping the instance; later measures recompute
        them and return the same results. Replacing ``fourier_coefficients``,
        ``expectation_type``, or ``observation_weights`` clears the cache
        automatically.

        Examples
        --------
        >>> import numpy as np
        >>> from spectral_connectivity import Connectivity
        >>> rng = np.random.default_rng(0)
        >>> fourier_coefficients = rng.standard_normal((1, 5, 3, 16, 4)) + 0j
        >>> connectivity = Connectivity(fourier_coefficients)
        >>> coherence = connectivity.magnitude_squared_coherence()
        >>> connectivity.clear_cache()
        >>> bool(np.array_equal(connectivity.magnitude_squared_coherence(), coherence, equal_nan=True))
        True
        """
        # Discovering descriptors avoids a second, hand-maintained registry that
        # could omit a newly added cache. Subclass caches are cleared as well.
        for klass in type(self).__mro__:
            for name, descriptor in vars(klass).items():
                if isinstance(descriptor, cached_property):
                    self.__dict__.pop(name, None)

    @property
    def fourier_coefficients(self) -> BackendArray:
        """Multitaper Fourier coefficients.

        Shape (n_time_windows, n_trials, n_tapers, n_fft_samples, n_signals).
        The instance owns an immutable snapshot so cached calculations cannot
        become stale through in-place mutation. This accessor returns a detached
        copy, marked read-only when the backend supports it. Assign a new array
        through the setter to replace the coefficients and clear the caches.
        """
        return mark_readonly_if_supported(self._fourier_coefficients.copy())

    @fourier_coefficients.setter
    def fourier_coefficients(self, value: NDArray[np.complexfloating]) -> None:
        # Public assignment always copies defensively: the caller may keep its
        # array and later mutate it in place, which would silently invalidate the
        # cached intermediates.
        self._set_fourier_coefficients(value, adopt=False)
        # Nothing says what the new observations are (see
        # observations_are_time_segments).
        self._observations_are_time_segments = False

    def _adopt_fourier_coefficients(self, value: NDArray[np.complexfloating]) -> None:
        """Take ownership of a freshly produced, unshared array without copying.

        Used only by ``from_transform``, where ``value`` is ``transform.fft()``,
        which the ``SpectralTransform`` contract requires to be fresh and
        referenced nowhere else. This avoids a full copy of the largest array in
        the pipeline -- a transient ~2x peak on every construction, which can
        push a memory-constrained GPU into OOM.

        This is deliberately private and has no public ``copy=False`` surface: it
        is safe only when the array *and its writable NumPy base buffer* are
        unshared, which ``from_transform`` relies on the transform for and
        cannot check.
        """
        self._set_fourier_coefficients(value, adopt=True)

    def _set_fourier_coefficients(
        self, value: NDArray[np.complexfloating], *, adopt: bool
    ) -> None:
        # Move a host array (or array-like) onto the active backend before the
        # first device operation below (CuPy rejects NumPy operands). This is a
        # no-op for an array already on the backend, so the adopt fast path
        # stays copy-free.
        value = xp.asarray(value)
        if value.ndim != 5:
            msg = (
                f"fourier_coefficients must be 5-dimensional, got {value.ndim}D array.\n"
                f"Expected shape: (n_time_windows, n_trials, n_tapers, n_fft_samples, n_signals)\n"
                f"Got shape: {value.shape}\n\n"
                f"If you have time series data, use the Multitaper class to transform it:\n"
                f"  from spectral_connectivity import Multitaper\n"
                f"  m = Multitaper(time_series, sampling_frequency=your_fs, ...)\n"
                f"  fourier_coefficients = m.fft()"
            )
            raise ValueError(msg)
        if not xp.iscomplexobj(value):
            msg = (
                f"fourier_coefficients must be complex, got dtype {value.dtype}. "
                f"Real-valued coefficients carry no phase, so the imaginary "
                f"coherence, the phase-lag indices and the coherence phase would "
                f"all be exactly 0. Pass the complex FFT output (e.g. "
                f"numpy.fft.fft), not its real part or magnitude."
            )
            raise TypeError(msg)
        # Power spectral density can be computed on single signals, but
        # connectivity metrics require >= 2 signals; that is validated per-method
        # in _validate_multiple_signals.
        # Read by the directed measures and the multivariate kernels, whose
        # failures on NaN or Inf are explained by the warning below (or the
        # transform's).
        self._coefficients_are_finite = bool(xp.all(xp.isfinite(value)))
        if not self._coefficients_are_finite:
            warnings.warn(
                f"{_NONFINITE_COEFFICIENTS_WARNING} This may indicate:\n"
                "  - NaN/Inf in your input time series data\n"
                "  - Issues with windowing parameters (e.g., window too short)\n"
                "  - Numerical instability in preprocessing\n\n"
                "Suggestions:\n"
                "  - Check your input data for NaN/Inf values\n"
                "  - Consider interpolating missing data points\n"
                "  - Review artifact removal procedures\n"
                "  - Verify time_window_duration and time_halfbandwidth_product parameters",
                UserWarning,
                stacklevel=stacklevel_outside_package(),
            )
        # Own the coefficients as an immutable snapshot: the cached intermediates
        # (_power, the reduced cross-spectrum, and the directed-measure factors)
        # assume the coefficients change only through the setter, which clears
        # them. Marking the snapshot read-only turns an in-place edit via the
        # getter into a clear error rather than silently stale results.
        if adopt:
            # `value` is unshared by the SpectralTransform contract but may be a
            # view (Multitaper's is a swapaxes view) whose base buffer is
            # writable; freeze the whole base chain, not just the outer view, or
            # the data stays reachable and mutable through `.base`. No copy --
            # this is the memory-saving path.
            mark_readonly_chain_if_supported(value)
            owned = value
        else:
            # copy(order="K") keeps the caller's array's memory layout, so
            # downstream matmuls see the same strides and results are unchanged to
            # the bit (a plain C-order copy would perturb the BLAS summation order
            # by ~1e-16). CuPy may not support the writeable flag; the copy alone
            # still decouples the instance from later mutation of the caller's
            # array there.
            owned = mark_readonly_if_supported(value.copy(order="K"))
        value = owned
        self._fourier_coefficients = value
        self.clear_cache()
        # On reassignment (not initial construction), a change in the number of
        # FFT bins or time windows invalidates the stored frequency/time
        # coordinates. Reset them to geometry-matching defaults so
        # coordinate-dependent methods stay consistent -- stale coordinates would
        # otherwise silently drop or misalign bins (e.g. phase_slope_index) or
        # raise (group_delay). Warn because any user-supplied coordinates (e.g.
        # Hz frequencies from from_transform) are discarded.
        n_fft_samples = value.shape[-2]
        n_time_windows = value.shape[0]
        frequencies_stale = (
            getattr(self, "_frequencies", None) is not None
            and len(self._frequencies) != n_fft_samples
        )
        time_stale = (
            getattr(self, "time", None) is not None and len(self.time) != n_time_windows
        )
        observation_weights = getattr(self, "_observation_weights", None)
        expected_weight_shape = (*value.shape[:-1], 1)
        weights_stale = observation_weights is not None and (
            tuple(observation_weights.shape) != expected_weight_shape
        )
        if frequencies_stale or time_stale or weights_stale:
            warnings.warn(
                "Reassigning fourier_coefficients changed the FFT/time geometry; "
                "incompatible frequency/time coordinates, observation weights, "
                "half_bandwidth and time_bins_are_disjoint were reset to defaults or "
                "cleared. Construct a new Connectivity "
                "if you need specific coordinates or weights for the new data.",
                UserWarning,
                stacklevel=stacklevel_outside_package(),
            )
            if frequencies_stale:
                # A half bandwidth is in the units of the discarded coordinate.
                self._half_bandwidth = None
                self._frequencies = (
                    xp.linspace(0.0, 0.5, n_fft_samples)
                    if getattr(self, "_is_one_sided", False)
                    else xp.fft.fftfreq(n_fft_samples)
                )
            if time_stale:
                self.time = np.arange(n_time_windows)  # host coordinate, as in __init__
                # Nothing says the new windows are disjoint; assume they may overlap.
                self._time_bins_are_disjoint = False
            if weights_stale:
                self._observation_weights = None

    @property
    def expectation_type(self) -> str:
        """Which dimensions the cross-spectral matrix is averaged over.

        Reassigning clears all cached intermediates (see :meth:`clear_cache`).
        """
        return self._expectation_type

    @expectation_type.setter
    def expectation_type(self, value: str) -> None:
        if value not in EXPECTATION_AXES:
            # Detect the common mistake of the right words in the wrong order.
            words = set(value.split("_"))
            valid_words = {"time", "trials", "tapers"}
            suggestion = None
            if words.issubset(valid_words):
                for valid_key in EXPECTATION_AXES:
                    if set(valid_key.split("_")) == words:
                        suggestion = valid_key
                        break

            error_msg = (
                f"Invalid expectation_type '{value}' is not supported.\n"
                f"This parameter controls which dimensions to average over when computing "
                f"the cross-spectral matrix.\n"
            )
            if suggestion:
                error_msg += (
                    f"\nDid you mean '{suggestion}'? (The words must be in a specific order)\n"
                )
            error_msg += "\nValid options are:\n"
            for key in sorted(EXPECTATION_AXES):
                error_msg += f"  - '{key}'\n"
            error_msg += "\nMost common: 'trials_tapers' (average over both trials and tapers)"
            raise ValueError(error_msg)
        self._require_tapers_averaged_for_exact_spectrum(value)

        self._expectation_type = value
        self.clear_cache()

    @classmethod
    @_rejects_removed_blocks("Connectivity.from_multitaper()")
    def from_multitaper(
        cls,
        multitaper_instance: "SpectralTransform",
        expectation_type: str = "trials_tapers",
        dtype: Any = xp.complex128,
        minimum_phase_tolerance: float = 1e-8,
        minimum_phase_max_iterations: int = 500,
    ) -> "Connectivity":
        """Construct from a spectral transform (deprecated name).

        .. deprecated:: 3.0
           Use :meth:`from_transform`, which takes the same arguments (the
           first is named ``transform``) and builds the same instance.
           ``from_multitaper`` is removed in 4.0. Neither name calls the
           other, so a subclass that overrides ``from_multitaper`` must move
           the override, and its ``super()`` call, to ``from_transform``.

        .. versionchanged:: 3.0
           The 2.x ``blocks`` argument (third, before ``dtype``) is removed;
           passing it raises ``TypeError``.

        Accepts any :class:`~spectral_connectivity.transforms.SpectralTransform`
        and builds the instance as :meth:`from_transform` does; a subclass's
        override of ``from_transform`` is not called.

        Parameters
        ----------
        multitaper_instance : SpectralTransform
            The transform; ``transform`` in :meth:`from_transform`.
        expectation_type, dtype, minimum_phase_tolerance, minimum_phase_max_iterations
            As in :meth:`from_transform`.

        Returns
        -------
        Connectivity
            New Connectivity instance.

        """
        instance = cls._from_transform(
            multitaper_instance,
            expectation_type=expectation_type,
            dtype=dtype,
            minimum_phase_tolerance=minimum_phase_tolerance,
            minimum_phase_max_iterations=minimum_phase_max_iterations,
        )
        # Warned once the instance is built, as the wrappers do, so a rejected
        # call (a 2.x blocks in dtype's position, a bad expectation type, ...)
        # shows only its error.
        warn_deprecated(
            f"{cls.__name__}.from_multitaper()",
            f"{cls.__name__}.from_transform()",
            "takes the same arguments (the first is named transform, not "
            "multitaper_instance) and builds the same instance",
        )
        return instance

    @classmethod
    @_rejects_removed_blocks("Connectivity.from_transform()")
    def from_transform(
        cls,
        transform: "SpectralTransform",
        expectation_type: str = "trials_tapers",
        dtype: Any = xp.complex128,
        minimum_phase_tolerance: float = 1e-8,
        minimum_phase_max_iterations: int = 500,
    ) -> "Connectivity":
        """Construct from any spectral transform.

        .. versionadded:: 3.0
           The canonical constructor; :meth:`from_multitaper` is its
           deprecated 2.x name.

        Parameters
        ----------
        transform : SpectralTransform
            Such as :class:`~spectral_connectivity.transforms.Multitaper`,
            :class:`~spectral_connectivity.transforms.MorletWavelet`, or a
            user-defined class; see :class:`~spectral_connectivity.transforms.SpectralTransform`
            for the required and optional members. ``fft()`` must return fresh,
            unshared storage on each call. Neither the transform nor its caller
            may subsequently mutate it through any alias: the result is used
            without copying and marked read-only where the backend supports this.
        expectation_type : str, default="trials_tapers"
            How to average the cross-spectral matrix.
        dtype : np.dtype, default=complex128
            Complex data type for computations; a real dtype (or ``None``)
            raises ``TypeError``.
        minimum_phase_tolerance : float, default=1e-8
            Relative convergence tolerance for the Wilson minimum-phase
            factorization used by the directed measures.
        minimum_phase_max_iterations : int, default=500
            Maximum Wilson iterations. Increase for near-singular cross-spectral
            matrices (highly correlated channels) that fail to converge.

        Returns
        -------
        Connectivity
            New Connectivity instance.
        """
        return cls._from_transform(
            transform,
            expectation_type=expectation_type,
            dtype=dtype,
            minimum_phase_tolerance=minimum_phase_tolerance,
            minimum_phase_max_iterations=minimum_phase_max_iterations,
        )

    @classmethod
    @_var_settings_by_keyword
    def from_var(
        cls,
        coefficients: NDArray[np.floating],
        noise_covariance: NDArray[np.floating] | None = None,
        *,
        sampling_frequency: float,
        n_fft_samples: int | None = None,
        time: NDArray[np.floating] | None = None,
        expectation_type: str = "trials_tapers",
        dtype: Any = xp.complex128,
        minimum_phase_tolerance: float = 1e-8,
        minimum_phase_max_iterations: int = 500,
    ) -> "Connectivity":
        """Construct from the exact spectrum of a vector autoregressive (VAR) model.

        The shortcut for ``Connectivity.from_transform(VARSpectrum(coefficients,
        noise_covariance, sampling_frequency=..., n_fft_samples=..., time=...))``:
        every measure that is a function of the cross-spectral matrix is then
        the model's population value, with no estimation error (the directed
        measures to the accuracy of the Wilson factorization), and the
        measures that use individual observations raise (see
        :class:`~spectral_connectivity.transforms.VARSpectrum`). For labeled
        results, call
        :func:`~spectral_connectivity.wrapper.connectivity_to_xarray` with the
        ``VARSpectrum``, once per measure.

        .. versionadded:: 3.0

        Parameters
        ----------
        coefficients : array, shape (n_lags, n_signals, n_signals) or (n_time_windows, n_lags, n_signals, n_signals)
            ``coefficients[k - 1]`` is ``A_k`` of ``x(t) = sum_k A_k x(t - k) +
            e(t)``, as for :func:`~spectral_connectivity.simulate.simulate_MVAR`
            and statsmodels' ``VARResults.coefs``; one model per window with
            the leading axis.
        noise_covariance : array, shape (n_signals, n_signals) or (n_time_windows, n_signals, n_signals), optional
            Covariance of the innovations ``e(t)`` (statsmodels' ``sigma_u``);
            the identity by default.
        sampling_frequency : float
            Samples per second; required.
        n_fft_samples : int, optional
            Number of frequency bins; by default the shortest fast length on
            which the directed measures are exact (see
            :attr:`VARSpectrum.minimum_n_fft_samples
            <spectral_connectivity.transforms.VARSpectrum.minimum_n_fft_samples>`).
        time : array, shape (n_time_windows,), optional
            Time of each window in seconds; window indices by default.
        expectation_type : str, default="trials_tapers"
            Must average the tapers: ``"trials_tapers"``, ``"tapers"``,
            ``"time_tapers"`` (the mean spectrum over windows), or
            ``"time_trials_tapers"``.
        dtype, minimum_phase_tolerance, minimum_phase_max_iterations
            As in :meth:`from_transform`.

        Returns
        -------
        Connectivity
            New instance with ``is_exact_spectrum`` set.

        Examples
        --------
        Signal 0 drives signal 1 one sample later, never the reverse:

        >>> import numpy as np
        >>> from spectral_connectivity import Connectivity
        >>> coefficients = np.array([[[0.5, 0.0], [0.4, 0.5]]])  # (n_lags, target, source)
        >>> connectivity = Connectivity.from_var(coefficients, sampling_frequency=100)
        >>> granger = connectivity.pairwise_spectral_granger_prediction()  # [..., source, target]
        >>> bool(granger[0, :, 0, 1].min() > 0.05), bool(granger[0, :, 1, 0].max() < 1e-10)
        (True, True)
        """
        return cls.from_transform(
            VARSpectrum(
                coefficients,
                noise_covariance,
                sampling_frequency=sampling_frequency,
                n_fft_samples=n_fft_samples,
                time=time,
            ),
            expectation_type=expectation_type,
            dtype=dtype,
            minimum_phase_tolerance=minimum_phase_tolerance,
            minimum_phase_max_iterations=minimum_phase_max_iterations,
        )

    @classmethod
    def _from_transform(
        cls,
        transform: "SpectralTransform",
        *,
        expectation_type: str,
        dtype: Any,
        minimum_phase_tolerance: float,
        minimum_phase_max_iterations: int,
    ) -> "Connectivity":
        """Build an instance from ``transform``; see :meth:`from_transform`.

        Both public constructors call it, and neither calls the other, so an
        override of either name never re-enters the other.
        """
        init_kwargs: dict[str, Any] = {
            "expectation_type": expectation_type,
            # Checked before the (possibly expensive) fft(), like the flags below.
            "dtype": _validated_complex_dtype(dtype),
            "minimum_phase_tolerance": minimum_phase_tolerance,
            "minimum_phase_max_iterations": minimum_phase_max_iterations,
        }
        # The optional SpectralTransform attributes must always reach the
        # instance: a subclass that cannot accept them fails loudly here rather
        # than silently treating a one-sided, weighted, or correlated spectrum as
        # two-sided, unweighted, and independent. They are passed only when
        # non-default so a subclass mirroring the older signature keeps working
        # with a plain two-sided transform. The flags are checked before the
        # (possibly expensive) fft() so a bad one fails fast; the coordinates
        # and weights are read after it, which may set them.
        if _transform_flag(transform, "is_one_sided", False):
            init_kwargs["is_one_sided"] = True
        if not _transform_flag(transform, "observations_are_independent", True):
            init_kwargs["observations_are_independent"] = False
        if not _transform_flag(transform, "time_bins_are_independent", True):
            init_kwargs["time_bins_are_independent"] = False
        if _transform_flag(transform, "is_exact_spectrum", False):
            init_kwargs["is_exact_spectrum"] = True
        # Unlike the flags above, the half bandwidth and whether the time bins
        # are disjoint do not change any measure, so they are set after
        # construction: a subclass mirroring the older signature keeps working
        # with a Multitaper, which defines both.
        half_bandwidth = _optional_transform_attribute(transform, "half_bandwidth", None)
        if half_bandwidth is not None:
            half_bandwidth = _validated_half_bandwidth(half_bandwidth)
        time_bins_are_disjoint = _transform_flag(transform, "time_bins_are_disjoint", False)
        observations_are_time_segments = _transform_flag(
            transform, "observations_are_time_segments", False
        )
        init_kwargs["fourier_coefficients"] = transform.fft()
        init_kwargs["time"] = transform.time
        init_kwargs["frequencies"] = transform.frequencies
        weights = _optional_transform_attribute(transform, "observation_weights", None)
        if weights is not None:
            init_kwargs["observation_weights"] = weights
        # The SpectralTransform contract requires fft() to return a freshly
        # built, unshared array, so adopt it in place instead of copying (see
        # Connectivity._adopt_fourier_coefficients). Only pass the private
        # keyword when the subclass has not overridden __init__:
        # an overriding subclass need not accept it, and passing it would raise
        # TypeError. Such a subclass falls back to the defensive-copy path.
        if cls.__init__ is Connectivity.__init__:
            init_kwargs["_adopt_fourier_coefficients"] = True
        # A transform of this package that warned about NaN or Inf in its time
        # series records it, and the coefficients' warning would repeat it in
        # other words. Coefficients from any other transform, or that are not
        # finite although the time series was, still warn.
        with _user_warnings_ignored(
            [re.escape(_NONFINITE_COEFFICIENTS_WARNING)],
            not _optional_transform_attribute(transform, "_time_series_is_finite", True),
        ):
            connectivity = cls(**init_kwargs)
        connectivity._half_bandwidth = half_bandwidth
        connectivity._time_bins_are_disjoint = time_bins_are_disjoint
        connectivity._observations_are_time_segments = observations_are_time_segments
        return connectivity

    @contextlib.contextmanager
    def _decomposition_of_nonfinite_coefficients(self, measure: str) -> Iterator[None]:
        """Report a matrix decomposition that fails on NaN or Inf coefficients
        as that, in the user's terms, rather than as LAPACK's ``LinAlgError``.

        A failure on finite coefficients is re-raised unchanged. CuPy's
        decompositions return NaN rather than raise, so the caller also checks
        the result with :meth:`_require_finite_decomposition`.
        """
        try:
            yield
        except np.linalg.LinAlgError as error:
            if self._coefficients_are_finite:
                raise
            raise ValueError(_nonfinite_decomposition_message(measure, error)) from error

    @cached_property
    def _nonfinite_coefficients_are_weighted(self) -> bool:
        """Whether some NaN or Inf coefficient has a nonzero observation weight.

        Coefficients that are not finite only where zero weights omit them
        never reach a weighted decomposition.
        """
        if self._coefficients_are_finite:
            return False
        if self._observation_weights is None:
            return True
        omitted = xp.broadcast_to(
            self._observation_weights == 0, self._fourier_coefficients.shape
        )
        return not bool(xp.all(xp.isfinite(self._fourier_coefficients) | omitted))

    def _require_finite_decomposition(self, measure: str, values: BackendArray) -> None:
        """Raise, as :meth:`_decomposition_of_nonfinite_coefficients` does,
        when NaN or Inf coefficients left ``values`` (a decomposition's output,
        before any deliberate NaN) not finite: the backend returned NaN rather
        than raising. Coefficients that are not finite only where zero weights
        omit them pass: a bin left with no weighted observation is NaN from
        zero power, as for finite coefficients."""
        if self._nonfinite_coefficients_are_weighted and not bool(xp.all(xp.isfinite(values))):
            raise ValueError(_nonfinite_decomposition_message(measure))

    def _validate_multiple_signals(self) -> None:
        """Raise if fewer than two signals are present.

        Connectivity measures quantify relationships between pairs of signals
        and are undefined for a single signal (they would otherwise return an
        all-NaN array with no error). ``power()`` is exempt because power
        spectral density is well-defined for one signal.
        """
        n_signals = self._fourier_coefficients.shape[-1]
        if n_signals < 2:
            msg = (
                f"Connectivity measures require at least 2 signals, but "
                f"fourier_coefficients has {n_signals} signal "
                f"(shape[-1] == {n_signals}).\n"
                f"Connectivity quantifies relationships between pairs of "
                f"signals; for a single signal use power() instead "
                f"(method='power' in multitaper_connectivity).\n"
                f"If you sliced to one channel, keep the signal axis, e.g. "
                f"fourier_coefficients[..., [channel_index]] rather than "
                f"fourier_coefficients[..., channel_index]."
            )
            raise ValueError(msg)

    def _require_uniform_frequency_grid(self, measure: str) -> None:
        """Raise if the frequency coordinate is not equally spaced.

        Measures that combine adjacent bins (phase slope, delay estimates)
        interpret one bin step as one frequency step; a wavelet transform can
        expose an arbitrary grid, on which those combinations are meaningless.
        """
        frequencies = to_numpy(self.frequencies)
        if frequencies.size >= 2 and _uniform_grid_spacing(frequencies) is None:
            steps = np.diff(frequencies)
            msg = (
                f"{measure} requires uniformly spaced frequencies because it "
                f"combines adjacent frequency bins, but the grid steps range "
                f"from {steps.min():g} to {steps.max():g} Hz. Use a linearly "
                "spaced frequency grid (e.g. MorletWavelet(frequencies="
                "np.arange(low, high, step)))."
            )
            raise ValueError(msg)

    def _require_uniform_observation_weights(self, measure: str, reason: str) -> None:
        """Raise if observation weights reweight observations for a measure
        that assumes equally weighted independent observations.

        Weights that only keep or drop observations (see
        :attr:`_observation_weights_only_drop`) pass: the measure counts each
        bin's kept observations instead.
        """
        if self._observation_weights_only_drop:
            return
        msg = (
            f"{measure} does not support non-uniform observation_weights. "
            f"{reason} Non-uniform weights come from a smoothing_kernel other "
            "than 'boxcar'; weights that only drop observations (0 or one common "
            "weight in each bin, as artifact_samples and edge_mode='nan' give) are "
            "supported. Use smoothing_kernel='boxcar'."
        )
        raise ValueError(msg)

    def _validate_counted_observations(
        self,
        measure: str,
        *,
        why: str = (
            "This bias correction divides by a factor that is zero when "
            "n_observations < 2, so it is undefined for a single observation."
        ),
        remedy: str = (
            "more trials/tapers, or the non-debiased measure "
            "(phase_lag_index / phase_locking_value)"
        ),
        weights_reason: str = (
            "Its finite-sample correction assumes equally weighted independent "
            "observations; use a non-debiased measure instead."
        ),
        assumption: str = (
            "Its finite-sample bias correction treats every observation as an "
            "independent sample"
        ),
    ) -> None:
        """Raise if a measure that counts its observations has too few.

        The debiased phase-lag index and pairwise phase consistency (the
        defaults describe them) divide by ``n_observations - 1``
        (respectively ``n_observations ** 2 - n_observations``), which is zero
        for a single observation and would otherwise return silent inf/NaN;
        the ``"auto"`` shrinkage intensity of partial coherence estimates a
        variance across the observations. ``why`` says why one observation is
        not enough and ``remedy`` what to use instead; ``weights_reason`` and
        ``assumption`` go to the refusal of reweighting observation weights
        and to the warning for correlated observations. A bin that zero
        weights leave with fewer than two is NaN instead (see
        :meth:`_counts_for_correction`).
        """
        n_observations = self.n_observations
        if n_observations < 2:
            msg = (
                f"{measure} requires at least 2 observations "
                f"(n_observations == n_trials * n_tapers), but got "
                f"{n_observations}. {why} Use {remedy}."
            )
            raise ValueError(msg)
        self._require_uniform_observation_weights(measure, weights_reason)
        self._warn_correlated_observations(measure, assumption)

    def _warn_correlated_observations(
        self,
        measure: str,
        assumption: str,
        time_bin_remedy: str = (
            "use time_window_step >= time_window_duration / 2, more MorletWavelet "
            "decimation, or an expectation_type that does not average over time"
        ),
    ) -> None:
        """Warn once that ``measure`` counts correlated observations as independent.

        ``n_observations`` is a raw count of the averaged observations. When
        the transform reports ``observations_are_independent=False`` (a
        MorletWavelet smoothing neighborhood or Welch segments overlapping by
        more than half on the observation axis), or the expectation averages
        over time bins it reports as correlated
        (``time_bins_are_independent=False``: overlapping Multitaper/STFT
        windows or closely spaced Morlet samples), that count overstates the
        effective sample size, biasing every consumer that uses it as a
        degrees-of-freedom or bias-correction factor. ``assumption`` states
        what ``measure`` uses the count for so the message is actionable, and
        ``time_bin_remedy`` what to do about correlated time bins.
        """
        averages_correlated_time_bins = (
            not self._time_bins_are_independent and 0 in self._expectation_axes
        )
        if self._observations_are_independent and not averages_correlated_time_bins:
            return
        reasons = []
        if not self._observations_are_independent:
            reasons.append(
                "this transform's trial/taper observations are correlated "
                "(observations_are_independent is False: a MorletWavelet smoothing "
                "neighborhood, or Welch segments overlapping by more than half; use "
                "a Multitaper transform, Welch with segment_overlap <= 0.5, or "
                "MorletWavelet without smoothing)"
            )
        if averages_correlated_time_bins:
            reasons.append(
                f"expectation_type={self.expectation_type!r} averages over time "
                "bins that are correlated (time_bins_are_independent is False: "
                "Multitaper or ShortTimeFourierTransform windows overlapping by "
                "more than half, or MorletWavelet samples closer than 4 wavelet "
                f"standard deviations; {time_bin_remedy})"
            )
        warnings.warn(
            f"{measure} assumes independent observations, but "
            f"{' and '.join(reasons)}. {assumption}, so n_observations == "
            f"{self.n_observations} overstates the effective sample size and the "
            f"result is biased.",
            UserWarning,
            stacklevel=stacklevel_outside_package(),
        )

    def _more_observations_remedies(self, *, other_expectations: bool = True) -> list[str]:
        """What raises :attr:`n_observations`: more of each averaged axis, and
        (if ``other_expectations``) an expectation_type that averages more."""
        averaged = self.expectation_type.split("_")
        remedies = [_MORE_OBSERVATIONS[axis] for axis in averaged]
        if other_expectations and len(averaged) < len(_MORE_OBSERVATIONS):
            remedies.append(
                f"an expectation_type that averages more axes than {self.expectation_type!r}"
            )
        return remedies

    def _warn_single_observation_degenerate(
        self,
        measure: str,
        consequence: str = (
            "every magnitude-normalized value is mathematically forced to 1 "
            "(apparent perfect connectivity)"
        ),
    ) -> None:
        """Warn that a normalized measure is degenerate for one observation.

        Coherency, the phase-locking value, and every measure derived from
        them normalize each cross-spectral entry by its magnitude; the
        phase-lag family averages the sign of one imaginary cross-spectrum per
        observation; partial coherence inverts, and the group measures and
        global coherence decompose, a rank-one cross-spectral matrix. With a
        single observation (one trial times one taper/window -- e.g. a
        single-trial :class:`~spectral_connectivity.transforms.MorletWavelet`
        transform without ``smoothing_time``) the result is fixed by that
        normalization rather than by the data (``consequence`` states how for
        ``measure``), so the measure reports perfect connectivity or a
        perfectly consistent lag between unrelated signals and carries no
        information. This is a silent-failure trap rather than an error, so it
        warns instead of raising.
        """
        if self.n_observations < 2:
            where = "(n_observations == 1)"
        elif self._fewest_valid_observations() == 1:
            where = _SINGLE_OBSERVATION_IN_SOME_BINS
        else:
            return
        remedies = self._more_observations_remedies()
        warnings.warn(
            f"{measure} is computed from a single observation {where}, so "
            f"{consequence} regardless of the data, and the result carries no "
            f"information. Use {', or '.join(remedies)}.",
            UserWarning,
            stacklevel=stacklevel_outside_package(),
        )

    def _warn_few_global_coherence_observations(self) -> None:
        """Warn when global coherence pools fewer observations than signals.

        With ``n`` pooled observations and ``p`` signals the value is at least
        ``1 / min(n, p)`` whatever the data, so for ``n < p`` it cannot reach
        the ``1 / p`` of independent signals. For ``n >= p`` it is still biased
        above ``1 / p`` but no longer bounded away from it, so ``n < p`` is
        where the warning starts. ``n`` counts the observations with nonzero
        weight in the bin that has fewest (see :attr:`n_valid_observations`);
        the floor and the bias are described in the Notes of
        :meth:`global_coherence`.
        """
        n_observations, n_signals = self._fewest_valid_observations(), self.n_signals
        if n_observations == 0 or n_observations >= n_signals:
            return
        averaged = self.expectation_type.split("_")
        # A labelled wrapper fixed expectation_type at "trials_tapers": name the
        # call that pools the windows instead, when there are any.
        fixed_expectation = _expectation_fixed_by_wrapper.get()
        remedies = self._more_observations_remedies(other_expectations=not fixed_expectation)
        if fixed_expectation and self._fourier_coefficients.shape[0] > 1:
            remedies.append(
                "Connectivity.from_transform(transform, "
                "expectation_type='time_trials_tapers').global_coherence() to "
                "pool the time windows (Connectivity(fourier_coefficients, "
                "expectation_type='time_trials_tapers') for coefficients)"
            )
        remedies.append("fewer signals")
        dropped = n_observations < self.n_observations
        if n_observations == 1:
            where = _SINGLE_OBSERVATION_IN_SOME_BINS if dropped else "(n_observations == 1)"
            floor = (
                f"is computed from a single observation {where}. The "
                "cross-spectral matrix then has rank one, so its first component "
                "holds all of the power. Global coherence is forced to 1 whatever "
                "the data, and the result carries no information"
            )
        else:
            pooled = (
                f"as few as {n_observations} observations with nonzero weight in "
                "some bins (see n_valid_observations; of the averaged "
                f"{' x '.join(averaged)})"
                if dropped
                else f"n_observations == {n_observations} observations (the averaged "
                f"{' x '.join(averaged)})"
            )
            floor = (
                f"pools {pooled} for {n_signals} signals. The "
                f"cross-spectral matrix then has rank at most {n_observations}, so "
                f"global coherence is at least 1/{n_observations} = "
                f"{1 / n_observations:.3g} whatever the data"
            )
        warnings.warn(
            f"global_coherence {floor}. Independent signals of equal power have "
            f"1/n_signals = {1 / n_signals:.3g}, which fewer observations than "
            "signals cannot give. Even with more observations, the estimate stays "
            "above that unless they far outnumber the signals. Compare values only "
            "at matched numbers of observations and signals. "
            f"Use {', or '.join(remedies)}.",
            UserWarning,
            stacklevel=stacklevel_outside_package(),
        )

    def _require_multiple_frequencies(self, measure: str) -> None:
        """Raise if fewer than two frequency bins are available.

        ``delay``, ``group_delay`` and ``phase_slope_index`` read
        ``frequencies[1] - frequencies[0]`` to get the frequency step; with a
        single frequency bin that indexing would raise a raw ``IndexError``
        instead of a clear message.
        """
        n_frequencies = len(self.frequencies)
        if n_frequencies < 2:
            msg = (
                f"{measure} requires at least 2 frequency bins, but the data has "
                f"{n_frequencies}. Use a longer FFT (larger n_fft_samples / "
                f"n_time_samples_per_window)."
            )
            raise ValueError(msg)

    def _require_two_sided_spectrum(self, measure: str) -> None:
        """Reject directed factorization for positive-frequency-only inputs."""
        if self._is_one_sided:
            msg = (
                f"{measure} requires a full two-sided spectrum in standard FFT "
                "order. One-sided transforms such as Morlet wavelets support "
                "functional connectivity measures, but not Wilson-factorized "
                "directed measures."
            )
            raise ValueError(msg)

    def _nonnegative_frequency_count(self, n_frequencies: int) -> int:
        """Number of non-negative-frequency bins among ``n_frequencies``.

        Every bin for one-sided input; ``n_frequencies // 2 + 1`` (DC through
        Nyquist) for a two-sided spectrum in standard FFT order. The single
        source of truth for trimming results to non-negative frequencies.
        """
        return n_frequencies if self._is_one_sided else n_frequencies // 2 + 1

    def _one_sided_density(self, spectrum: BackendArray, frequency_axis: int) -> BackendArray:
        """Fold a cached (cross-)spectral density onto non-negative frequencies.

        Two-sided input is trimmed to its non-negative bins and each interior
        bin ``S(f)`` gains its negative-frequency partner as ``conj(S(-f))``,
        so the one-sided density integrates to the same total power as the
        two-sided spectrum. For a real signal ``S(-f) = conj(S(f))``, so the
        bin is doubled (exactly, for an exactly conjugate-symmetric FFT); for
        the coefficients of a complex signal the power on the diagonal is
        ``P(f) + P(-f)`` rather than twice one side. The fold keeps the matrix
        Hermitian. DC (bin 0) and the Nyquist bin (present only for an even FFT
        length) have no partner and are kept as they are. One-sided input is
        returned as a copy, detached from the cache so a caller mutating the
        result cannot corrupt measures that reuse it.

        Parameters
        ----------
        spectrum : array, shape (..., n_fft_samples, ...)
            Cached spectrum with frequency at ``frequency_axis``.
        frequency_axis : int
            Negative index of the frequency axis.

        Returns
        -------
        array, shape (..., n_frequencies, ...)
        """
        if self._is_one_sided:
            return spectrum.copy()
        n_fft_samples = spectrum.shape[frequency_axis]
        trailing = (slice(None),) * (-frequency_axis - 1)
        index: tuple[Any, ...] = (
            ...,
            slice(self._nonnegative_frequency_count(n_fft_samples)),
            *trailing,
        )
        nonnegative = spectrum[index]
        positive, negative = _paired_frequency_bins(n_fft_samples)
        n_interior = positive.stop - 1
        interior: tuple[Any, ...] = (..., positive, *trailing)
        partners: tuple[Any, ...] = (..., negative, *trailing)
        # One output buffer: the partners' conjugate is written into it and the
        # positive bins added in place; 0 Hz and Nyquist are copied as they are.
        density: BackendArray = xp.empty_like(nonnegative)
        xp.conj(spectrum[partners], out=density[interior])
        density[interior] += nonnegative[interior]
        unpaired_bins: tuple[tuple[Any, ...], ...] = (
            (..., slice(0, 1), *trailing),
            (..., slice(n_interior + 1, None), *trailing),
        )
        for unpaired in unpaired_bins:
            density[unpaired] = nonnegative[unpaired]
        return density

    @property
    @_asnumpy
    def frequencies(self) -> NDArray[np.floating]:
        """Return non-negative frequencies of the transform.

        Returns
        -------
        NDArray[floating], shape (n_frequencies,)
            Non-negative frequency values.

        """
        n_nonnegative = self._nonnegative_frequency_count(len(self._frequencies))
        freqs = xp.take(self._frequencies, indices=xp.arange(n_nonnegative), axis=0)

        # fftfreq returns negative Nyquist for even N, fix the sign
        if len(freqs) > 0 and freqs[-1] < 0:
            freqs = freqs.copy()  # Don't modify the original
            freqs[-1] = abs(freqs[-1])
        return freqs

    @property
    @_asnumpy
    def all_frequencies(self) -> NDArray[np.floating]:
        """Return positive and negative frequencies of the transform.

        Returns
        -------
        NDArray[floating], shape (n_frequencies,)
            All frequency values including negative frequencies.

        """
        return self._frequencies

    @cached_property
    def _power(self) -> NDArray[np.floating]:
        # Reused by coherency and directed measures; the input setters discover
        # and invalidate cached_property values automatically.
        return self._expectation(
            self._fourier_coefficients * self._fourier_coefficients.conjugate()
        ).real

    def _nonnegative_pairwise_power_scale(self) -> NDArray[np.floating]:
        """``sqrt(P_i P_j)`` at the non-negative frequencies.

        Shape (..., n_nonnegative_frequencies, n_signals, n_signals). Built only
        for the bins the normalized measures report, and recomputed per call: it
        costs one outer product of ``sqrt(_power)``, which also cannot underflow
        or overflow.
        """
        n_nonnegative = self._nonnegative_frequency_count(self._power.shape[-2])
        root_power = xp.sqrt(self._power[..., :n_nonnegative, :])
        return root_power[..., :, xp.newaxis] * root_power[..., xp.newaxis, :]

    def _nonnegative_fourier_coefficients(
        self, dtype: DTypeLike | None = None
    ) -> NDArray[np.complexfloating]:
        """Fourier coefficients at the non-negative frequencies, at ``self._dtype``.

        ``dtype`` overrides the precision. A view when the dtype already matches.
        """
        n_nonnegative = self._nonnegative_frequency_count(self._fourier_coefficients.shape[-2])
        coefficients: NDArray[np.complexfloating] = self._fourier_coefficients[
            ..., :n_nonnegative, :
        ].astype(self._dtype if dtype is None else dtype, copy=False)
        return coefficients

    def _nonnegative_cross_spectral_matrix(self) -> NDArray[np.complexfloating]:
        """Expected cross-spectral matrix at the non-negative frequencies (a view)."""
        cross_spectral_matrix = self._expectation_cross_spectral_matrix()
        n_nonnegative = self._nonnegative_frequency_count(cross_spectral_matrix.shape[-3])
        return cross_spectral_matrix[..., :n_nonnegative, :, :]

    @property
    def _cross_spectral_matrix(self) -> NDArray[np.complexfloating]:
        """Return the complex-valued linear association between fourier coefficients.

        Returns
        -------
        cross_spectral_matrix : array
            Shape (n_time_windows, n_trials, n_tapers, n_fft_samples,
            n_signals, n_signals). Complex cross-spectral matrix.

        """
        fourier_coefficients = self._fourier_coefficients[..., xp.newaxis]
        return _complex_inner_product(
            fourier_coefficients, fourier_coefficients, dtype=self._dtype
        )

    def _expectation_cross_spectral_matrix(self) -> NDArray[np.complexfloating]:
        """Expected cross-spectral matrix, reduced over the averaged observations.

        Validates that at least two signals are present, then returns the cached
        reduced cross-spectral matrix -- a single batched matmul over the
        averaged time/trials/tapers axes (see ``_reduced_cross_spectral_matrix``)
        rather than a per-observation outer product.

        Returns
        -------
        array, shape (..., n_frequencies, n_signals, n_signals)
            Expected cross-spectral matrix.
        """
        self._validate_multiple_signals()
        return self._cached_reduced_cross_spectral_matrix

    def _reduced_cross_spectral_matrix(
        self,
        fourier_coefficients: NDArray[np.complexfloating] | None = None,
        *,
        dtype: DTypeLike | None = None,
    ) -> NDArray[np.complexfloating]:
        """Expected cross-spectral matrix via a single batched matmul.

        Numerically equivalent (to floating-point tolerance) to
        ``self._expectation(self._cross_spectral_matrix)``, but contracts the
        averaged observation axes (any subset of time/trials/tapers, taken from
        the active ``expectation_type``) directly instead of materializing the
        full ``(..., n_signals, n_signals)`` outer product for every
        observation. For the default ``trials_tapers`` expectation this replaces
        a large intermediate with a small result and is markedly faster.

        Parameters
        ----------
        fourier_coefficients : array, optional
            Coefficients of shape
            ``(n_time_windows, n_trials, n_tapers, n_fft_samples, *batch,
            n_signals)`` to reduce; defaults to this instance's coefficients.
            ``phase_locking_value`` passes unit-normalized coefficients so the
            same batched matmul yields its normalized cross-spectrum, and
            ``_subset_cross_spectral_matrix`` passes a pair axis as ``batch``.
            The frequency axis may hold only the leading bins (e.g. the
            non-negative frequencies); observation weights are sliced to match.
        dtype : complex dtype, optional
            Precision of the product; ``self._dtype`` by default.

        Returns
        -------
        array, shape (..., n_frequencies, *batch, n_signals, n_signals)
            Expected cross-spectral matrix. The leading axes are whichever of
            time/trials/tapers are *not* averaged, matching the shape produced
            by the equivalent expectation over the full outer product.

        """
        if fourier_coefficients is None:
            fourier_coefficients = self._fourier_coefficients
        average_axes = self._expectation_axes

        signal_axis = fourier_coefficients.ndim - 1
        frequency_axis = 3
        # Frequency and any extra batch axes (axes 3 to signal_axis - 1) stay.
        batch_axes = list(range(frequency_axis, signal_axis))
        kept_axes = self._kept_observation_axes
        # Reorder to (kept leading axes..., frequency, batch..., averaged axes...,
        # signals) so the averaged axes collapse into a single observation axis
        # adjacent to signals, ready for a batched matmul.
        order = [*kept_axes, *batch_axes, *average_axes, signal_axis]
        observations = xp.transpose(fourier_coefficients, order)

        n_observations = int(
            np.prod([fourier_coefficients.shape[axis] for axis in average_axes])
        )
        n_signals = fourier_coefficients.shape[signal_axis]
        observations = observations.reshape(
            (
                *observations.shape[: len(kept_axes) + len(batch_axes)],
                n_observations,
                n_signals,
            )
        )
        weights = None
        if self._observation_weights is not None:
            # Weights vary over observations and frequency, not the extra batch.
            weights = xp.transpose(
                self._observation_weights[
                    ..., : fourier_coefficients.shape[frequency_axis], 0
                ],
                [*kept_axes, frequency_axis, *average_axes],
            ).reshape(
                (
                    *observations.shape[: len(kept_axes) + 1],
                    *(1,) * (len(batch_axes) - 1),
                    n_observations,
                )
            )
            observations = _weight_observations(
                observations, xp.sqrt(weights)[..., xp.newaxis]
            )
        # cross_spectral_matrix[..., i, j] = mean_obs f_i * conj(f_j), matching
        # _complex_inner_product's convention, then averaged over observations.
        # Cast before multiplying (see _complex_inner_product).
        observations = observations.astype(self._dtype if dtype is None else dtype, copy=False)
        cross_spectral_matrix: NDArray[np.complexfloating] = xp.matmul(
            xp.swapaxes(observations, -1, -2), xp.conj(observations)
        )
        if weights is None:
            return cross_spectral_matrix / n_observations
        # At the product's precision, so the weights do not promote it.
        denominator = xp.sum(weights, axis=-1)[..., xp.newaxis, xp.newaxis].astype(
            cross_spectral_matrix.real.dtype, copy=False
        )
        return _divide_where(cross_spectral_matrix, denominator, denominator > 0, xp.nan)

    @cached_property
    def _cached_reduced_cross_spectral_matrix(self) -> NDArray[np.complexfloating]:
        """Cache the reduced expected cross-spectral matrix.

        This is the reduced result of ``_expectation_cross_spectral_matrix()``.
        It is reused within a measure (``coherency`` divides it by power) and
        across measures that share one ``Connectivity`` instance
        (``magnitude_squared_coherence``, ``coherence_phase``, ``imaginary_coherence``,
        pairwise spectral Granger). Only the reduced ``(..., n_signals,
        n_signals)`` form is cached, never the observation-resolved
        ``_cross_spectral_matrix``. Invalidated with the other cached
        intermediates when the inputs change; consumers treat it as read-only.
        """
        return self._reduced_cross_spectral_matrix()

    def _subset_cross_spectral_matrix(
        self, pairs: Sequence[Sequence[int]] | NDArray[np.integer]
    ) -> NDArray[np.complexfloating]:
        """Compute compact expected cross-spectra for channel pairs.

        Each pair's two signals are reduced over observations by the batched
        matmul of :meth:`_reduced_cross_spectral_matrix`, with the pairs as a
        batch axis processed in chunks of at most
        ``SUBSET_CROSS_SPECTRUM_MAX_WORKSPACE_ELEMENTS`` gathered coefficients.
        Neither the full ``n_signals x n_signals`` matrix nor an
        observation-resolved 2-by-2 matrix per pair is formed.

        Parameters
        ----------
        pairs : array_like
            Pairs of channel indices.

        Returns
        -------
        array, shape (..., n_pairs, n_frequencies, 2, 2)
            One compact 2-by-2 expected cross-spectral matrix per requested
            pair. The leading axes are the observation axes the configured
            expectation keeps.

        """
        self._validate_multiple_signals()
        pair_indices = xp.asarray(pairs, dtype=int)
        if pair_indices.ndim != 2 or pair_indices.shape[1] != 2:
            msg = "pairs must have shape (n_pairs, 2)."
            raise ValueError(msg)
        if pair_indices.size == 0:
            msg = "pairs must contain at least one signal pair."
            raise ValueError(msg)
        n_signals = self._fourier_coefficients.shape[-1]
        if bool(xp.any(pair_indices < 0)) or bool(xp.any(pair_indices >= n_signals)):
            msg = f"pair indices must be between 0 and {n_signals - 1}."
            raise IndexError(msg)
        coefficients = self._fourier_coefficients
        n_pairs = pair_indices.shape[0]
        elements_per_pair = 2 * int(np.prod(coefficients.shape[:-1]))
        pairs_per_chunk = max(
            1, SUBSET_CROSS_SPECTRUM_MAX_WORKSPACE_ELEMENTS // elements_per_pair
        )
        # Each chunk: (time, trials, tapers, frequency, pair, 2) gathered, reduced
        # to (..., frequency, pair, 2, 2).
        chunks = [
            self._reduced_cross_spectral_matrix(
                coefficients[..., pair_indices[start : start + pairs_per_chunk]]
            )
            for start in range(0, n_pairs, pairs_per_chunk)
        ]
        # Put pair before frequency so frequency remains axis -3 as required by
        # the Wilson factorization.
        return xp.moveaxis(xp.concatenate(chunks, axis=-3), -3, -4)

    # These quantities feed every directed-connectivity measure and are
    # expensive to compute (the minimum-phase decomposition in particular), so
    # they are cached per instance. The validated input setters clear these
    # cached properties before accepting a replacement.
    @cached_property
    def _minimum_phase_factor(self) -> NDArray[np.complexfloating]:
        self._require_two_sided_spectrum("Directed connectivity")
        return _factorize_spectrum(
            self._expectation_cross_spectral_matrix(),
            minimum_phase_tolerance=self._minimum_phase_tolerance,
            minimum_phase_max_iterations=self._minimum_phase_max_iterations,
        )

    @cached_property
    def _var_model(self) -> tuple[NDArray[np.complexfloating], NDArray[np.complexfloating]]:
        return _var_model_from_factor(self._minimum_phase_factor)

    @property
    def _transfer_function(self) -> NDArray[np.complexfloating]:
        return self._var_model[0]

    @property
    def _noise_covariance(self) -> NDArray[np.complexfloating]:
        return self._var_model[1]

    @cached_property
    def _MVAR_Fourier_coefficients(self) -> NDArray[np.complexfloating]:
        return _regularized_inverse(self._transfer_function, per_signal=True)

    def _expectation(self, values: BackendArray, *, frequency_axis: int = 3) -> BackendArray:
        """Average observation axes, applying optional spectral weights.

        ``values`` may hold only the leading (non-negative) frequency bins of
        the spectrum; the weights are restricted to the same bins.
        """
        if self._observation_weights is None:
            expected: BackendArray = xp.mean(values, axis=self._expectation_axes)
            return expected

        if frequency_axis < 0:
            frequency_axis += values.ndim
        if frequency_axis < 3 or frequency_axis >= values.ndim:
            msg = "frequency_axis must follow the three observation axes."
            raise ValueError(msg)
        n_frequencies = values.shape[frequency_axis]
        weight_shape = [1] * values.ndim
        weight_shape[0:3] = self._observation_weights.shape[0:3]
        weight_shape[frequency_axis] = n_frequencies
        weights = self._observation_weights[..., :n_frequencies, 0].reshape(weight_shape)
        numerator = xp.sum(_weight_observations(values, weights), axis=self._expectation_axes)
        denominator = xp.sum(weights, axis=self._expectation_axes)
        return _divide_where(numerator, denominator, denominator > 0, xp.nan)

    @cached_property
    def _kept_observations(self) -> BackendArray | None:
        """The observations 0/1 weights keep, when they are the same at every
        frequency: ``(n_time_windows, n_trials, n_tapers, 1, 1)`` booleans.

        ``Multitaper(artifact_samples=...)`` gives such weights. The phase-lag
        moments then zero each dropped observation of a tile in place and
        divide by the count kept, the values of the weighted path at a
        fraction of its cost (the other expectations multiply by the weights
        either way). ``None`` without weights or for any other weights.
        """
        weights = self._observation_weights
        if weights is None:
            return None
        first = weights[..., :1, :]
        if not bool(xp.all((first == 0) | (first == 1))) or not bool(xp.all(weights == first)):
            return None
        kept: BackendArray = first > 0
        return kept

    @cached_property
    def _observation_weights_only_drop(self) -> bool:
        """Whether the observation weights only keep or drop observations.

        True when, in every reduced bin, the observations with nonzero weight
        share one weight, so the weighted expectation there is the plain mean
        of the observations it keeps: equal weights, or the 0/1 weights of
        ``Multitaper(artifact_samples=...)`` and of a boxcar ``MorletWavelet``
        with ``edge_mode="nan"``. The measures that count observations then
        count each bin's kept ones (:attr:`n_valid_observations`). False when
        the weights also reweight the observations they keep (a smoothing
        kernel other than a boxcar), which those measures refuse.
        """
        if self._observation_weights is None or self._kept_observations is not None:
            return True
        weights = self._observation_weights[..., 0]
        kept_axes = self._kept_observation_axes
        order = [*kept_axes, 3, *self._expectation_axes]
        reordered = xp.transpose(weights, order)
        flattened = reordered.reshape((*reordered.shape[: len(kept_axes) + 1], -1))
        largest = xp.max(flattened, axis=-1, keepdims=True)
        return bool(xp.all((flattened == largest) | (flattened == 0)))

    def _valid_counts_over(self, axes: tuple[int, ...]) -> BackendArray:
        """Observations with nonzero weight over the observation ``axes``.

        Shape: the other observation axes, then ``n_fft_samples``. Only with
        observation weights.
        """
        assert self._observation_weights is not None
        counts: BackendArray = xp.count_nonzero(
            self._observation_weights[..., 0] > 0, axis=axes
        )
        return counts

    @cached_property
    def _valid_observation_counts(self) -> BackendArray:
        """Observations with nonzero weight in each reduced bin, ``(*kept
        observation axes, n_fft_samples)``; only with observation weights."""
        return self._valid_counts_over(self._expectation_axes)

    @staticmethod
    def _fewest_positive(counts: BackendArray) -> int:
        """The smallest positive count, ignoring the bins that keep none
        (every measure is NaN there); 0 if none keeps any."""
        kept = counts[counts > 0]
        return int(xp.min(kept)) if kept.size else 0

    def _fewest_valid_observations(self) -> int:
        """The fewest observations any reduced bin keeps (see
        :meth:`_fewest_positive`); :attr:`n_observations` without weights."""
        if self._observation_weights is None:
            return self.n_observations
        return self._fewest_positive(self._valid_observation_counts)

    @staticmethod
    def _uses_observations(fewest: int, n_observations: int, where: str) -> str:
        """ "uses {n} observations", or "uses as few as {n} ... in some bins" with
        ``where`` saying where the counts are when zero weights drop some."""
        noun = f"observation{'' if fewest == 1 else 's'}"
        if fewest == n_observations:
            return f"uses {fewest} {noun}"
        return f"uses as few as {fewest} {noun} in some bins ({where})"

    def _counts_for_correction(self, measure: str, dtype: DTypeLike) -> int | BackendArray:
        """The observation count of each bin of a measure that counts them.

        ``n_observations`` without observation weights; otherwise each bin's
        count of kept observations, shaped ``(..., n_nonnegative_frequencies,
        1, 1)`` to broadcast against a pairwise measure, in the measure's real
        ``dtype`` so they do not promote it. A bin with fewer than
        two leaves the finite-sample correction undefined: the caller makes it
        NaN. This warns once if a bin keeps a single observation; a bin that
        keeps none is NaN in every measure, which the transform reports.
        """
        if self._observation_weights is None:
            return self.n_observations
        counts = self._valid_observation_counts
        n_nonnegative = self._nonnegative_frequency_count(counts.shape[-1])
        counts = counts[..., :n_nonnegative, xp.newaxis, xp.newaxis]
        single = counts == 1
        if bool(xp.any(single)):
            n_single, n_bins = int(xp.count_nonzero(single)), int(single.size)
            warnings.warn(
                f"{measure}: {_SINGLE_KEPT_OBSERVATION} "
                f"{n_single} of {n_bins} bins (one per kept observation index and "
                "frequency), where its finite-sample correction, which needs two, "
                "is undefined, so it is NaN there. Connectivity.n_valid_observations "
                "counts each bin's observations; mark fewer samples as artifacts, or "
                "use shorter windows or more trials.",
                UserWarning,
                stacklevel=stacklevel_outside_package(),
            )
        return counts.astype(dtype)

    @property
    def _expectation_axes(self) -> tuple[int, ...]:
        """Observation axes reduced by the configured expectation."""
        return EXPECTATION_AXES[self.expectation_type]

    @property
    def _kept_observation_axes(self) -> tuple[int, ...]:
        """Observation axes the expectation keeps, in (time, trials, tapers) order.

        They lead every measure's result in this order.
        """
        return tuple(axis for axis in range(3) if axis not in self._expectation_axes)

    @property
    def n_observations(self) -> int:
        """Return the raw number of observations averaged by the expectation.

        Returns
        -------
        int
            Product of the lengths of the averaged observation axes (for the
            default ``"trials_tapers"`` expectation, ``n_trials * n_tapers``).
            This is a raw count, not an effective number of independent
            observations: it ignores ``observation_weights``, so it includes
            the observations zero weights drop, and is not reduced when the
            transform's observations are correlated (see
            ``observations_are_independent``). Use
            :attr:`n_valid_observations` when ``artifact_samples`` or edge
            masking drops observations. For an exact spectrum
            (:attr:`is_exact_spectrum`, e.g. a
            :class:`~spectral_connectivity.transforms.VARSpectrum`) it counts
            the columns of one factor of the cross-spectral matrix
            (``n_signals`` per window), which are not samples: it is no degrees
            of freedom for a significance test or confidence interval, and the
            labeled results do not record it.

        """
        return int(
            np.prod(
                [self._fourier_coefficients.shape[axis] for axis in self._expectation_axes]
            )
        )

    @property
    def n_valid_observations(self) -> NDArray[np.intp]:
        """Number of observations each bin of a result averages.

        The observations with nonzero ``observation_weights`` among those the
        expectation averages: :attr:`n_observations` in every bin without
        weights, and fewer where zero weights drop observations, such as the
        trial windows ``Multitaper(artifact_samples=...)`` drops or the
        edges a ``MorletWavelet(edge_mode="nan")`` masks. The measures that
        count observations (the debiased phase-lag measures, pairwise phase
        consistency, the significance test of ``delay`` and ``group_delay``,
        and ``jackknife``) use these counts; pass them to the statistics
        functions too (for coherency shaped ``(..., n_frequencies, n_signals,
        n_signals)``, ``n_valid_observations[..., np.newaxis, np.newaxis]``).
        Like :attr:`n_observations` it is a raw count, not reduced for
        correlated observations.

        Returns
        -------
        NDArray[int], shape (*kept_observation_axes, n_frequencies)
            One count per bin of a measure's leading axes: the observation axes
            ``expectation_type`` keeps (``n_time_windows`` for the default),
            then the non-negative frequencies of :attr:`frequencies`. A host
            array.

        Examples
        --------
        Drop the windows of a trial that an artifact touches:

        >>> import numpy as np
        >>> from spectral_connectivity import Connectivity, Multitaper
        >>> time_series = np.random.default_rng(0).standard_normal((1000, 4, 2))
        >>> artifact_samples = np.zeros((1000, 4), dtype=bool)
        >>> artifact_samples[600:620, 1] = True  # trial 1, 0.6-0.62 s
        >>> multitaper = Multitaper(
        ...     time_series, sampling_frequency=1000, time_window_duration=0.25,
        ...     artifact_samples=artifact_samples,
        ... )
        >>> connectivity = Connectivity.from_transform(multitaper)
        >>> connectivity.n_observations  # 4 trials x 5 tapers
        20
        >>> connectivity.n_valid_observations[:, 0].tolist()  # per window, at 0 Hz
        [20, 20, 15, 20]
        """
        n_nonnegative = self._nonnegative_frequency_count(self._fourier_coefficients.shape[3])
        if self._observation_weights is None:
            kept_shape = tuple(
                self._fourier_coefficients.shape[axis] for axis in self._kept_observation_axes
            )
            return np.full((*kept_shape, n_nonnegative), self.n_observations, dtype=np.intp)
        counts = self._valid_observation_counts[..., :n_nonnegative]
        return np.asarray(to_numpy(counts), dtype=np.intp)

    @property
    def n_signals(self) -> int:
        """Number of signals represented by the Fourier coefficients."""
        return int(self._fourier_coefficients.shape[-1])

    @property
    def is_one_sided(self) -> bool:
        """Whether the input contains only non-negative frequencies."""
        return self._is_one_sided

    @property
    def observations_are_independent(self) -> bool:
        """Whether the trial/taper observations are statistically independent.

        ``False`` when the transform collected correlated estimates on the
        observation axis (a smoothing neighborhood of a
        :class:`~spectral_connectivity.transforms.MorletWavelet`, or
        :class:`~spectral_connectivity.transforms.Welch` segments overlapping by
        more than half). ``n_observations`` then overstates the effective sample
        size, so the measures that rely on it warn and ``jackknife`` refuses to
        treat tapers as leave-one-out units.
        """
        return self._observations_are_independent

    @property
    def observations_are_time_segments(self) -> bool:
        """Whether each trial's observations are segments of it in time.

        :meth:`from_transform` reads it from the transform (``True`` for
        :class:`~spectral_connectivity.transforms.Welch`, whose observations
        are successive segments; ``False`` for tapers of the same samples, as
        :class:`~spectral_connectivity.transforms.Multitaper`'s are). It is
        ``False`` for coefficients passed directly, and after reassigning
        them, since nothing then says what the observations are.
        :meth:`jackknife` with ``frequency_bands`` leaves out one observation
        at a time only when it is ``True``.

        .. versionadded:: 3.0
        """
        return self._observations_are_time_segments

    @property
    def half_bandwidth(self) -> float | None:
        """Half-width of the transform's spectral smoothing, or ``None`` if unknown.

        In the units of :attr:`frequencies`. :meth:`from_transform` reads it
        from the transform (``frequency_resolution / 2`` for
        :class:`~spectral_connectivity.transforms.Multitaper`); it is the
        default ``half_bandwidth`` of :meth:`surrogate_test`.
        """
        return self._half_bandwidth

    @property
    def time_bins_are_independent(self) -> bool:
        """Whether the time bins may be counted as independent observations.

        ``False`` when the transform's successive time bins are correlated
        (:class:`~spectral_connectivity.transforms.Multitaper` or
        :class:`~spectral_connectivity.transforms.ShortTimeFourierTransform`
        windows overlapping by more than half, or
        :class:`~spectral_connectivity.transforms.MorletWavelet` samples closer
        than four wavelet standard deviations). It matters only when the
        expectation averages over time: ``n_observations`` then counts the
        correlated bins and overstates the effective sample size, so the
        measures that rely on it warn.
        """
        return self._time_bins_are_independent

    @property
    def time_bins_are_disjoint(self) -> bool:
        """Whether successive time windows share no samples.

        :meth:`from_transform` reads it from the transform
        (:attr:`Multitaper.time_bins_are_disjoint
        <spectral_connectivity.transforms.Multitaper.time_bins_are_disjoint>`:
        a step of at least the window length); otherwise it is ``False``
        unless passed, and reassigning coefficients with another number of
        windows resets it to ``False``. :meth:`surrogate_test` randomizes each
        averaged time window as a separate unit only when it and
        :attr:`time_bins_are_independent` are both ``True``.
        """
        return self._time_bins_are_disjoint

    def _resampled_measure(self, method: Any, caller: str) -> tuple[str, Callable[..., Any]]:
        """The canonical name and bound method of the measure ``caller`` recomputes.

        Shared by the drivers that recompute a measure on resampled
        coefficients, so they accept and refuse the same names. A deprecated
        name warns, spelled as ``caller`` takes it.

        Raises
        ------
        TypeError
            If ``method`` is not a string.
        ValueError
            If ``method`` is not a public measure of this class.
        """
        # The registry imports this module, so its helpers are imported here.
        from spectral_connectivity._measure_registry import (
            _method_type_message,
            _suggest_measure_names,
        )

        if not isinstance(method, str):
            raise TypeError(
                _method_type_message(
                    "a measure name (str)", f"{type(method).__name__} {method!r}"
                )
            )
        method = _resolve_deprecated_measure_name(method, caller + "({!r})")
        method_attribute = inspect.getattr_static(type(self), method, None)
        if (
            method.startswith("_")
            or method in _NON_MEASURE_METHODS
            or not inspect.isfunction(method_attribute)
        ):
            msg = (
                f"{method!r} is not a connectivity measure {caller} can "
                f"compute.{did_you_mean(_suggest_measure_names(method))} Call "
                "spectral_connectivity.list_measures() to see the measure names."
            )
            raise ValueError(msg)
        # The static check above guarantees ``method`` names a public function
        # on the class, so the bound attribute is always callable here.
        return method, getattr(self, method)

    def _with_coefficients(
        self,
        coefficients: BackendArray,
        observation_weights: BackendArray | None,
    ) -> "Connectivity":
        """A copy of this instance holding other coefficients, with empty caches.

        For the resampling drivers: ``coefficients`` are fresh and unshared
        (a subset or a surrogate of this instance's), with the same time
        windows and frequencies, so they are adopted without a copy, and the
        coordinates, settings and flags are kept without validating them
        again. ``observation_weights`` must match their shape. The copy has
        this instance's type, so a subclass's own measures and attributes
        are kept.
        """
        clone = object.__new__(type(self))
        clone.__dict__.update(self.__dict__)
        clone.clear_cache()
        clone._fourier_coefficients = mark_readonly_chain_if_supported(coefficients)
        # A subset or surrogate of finite coefficients is finite.
        clone._coefficients_are_finite = self._coefficients_are_finite or bool(
            xp.all(xp.isfinite(coefficients))
        )
        clone._observation_weights = (
            None
            if observation_weights is None
            else mark_readonly_if_supported(observation_weights)
        )
        return clone

    @property
    def is_exact_spectrum(self) -> bool:
        """Whether the coefficients give an exact cross-spectral matrix, not samples.

        ``True`` for a :class:`~spectral_connectivity.transforms.VARSpectrum`:
        the taper-mean outer product of its coefficients is the model's
        cross-spectral matrix, and the individual observations are only one
        factorization of it. Every measure that is a function of that matrix
        is then exact; those that use each observation or count them,
        :meth:`jackknife`, :meth:`surrogate_test` and ``shrinkage="auto"`` of
        :meth:`partial_coherence`, raise ``ValueError``.

        .. versionadded:: 3.0
        """
        return self._is_exact_spectrum

    def _require_tapers_averaged_for_exact_spectrum(self, expectation_type: str) -> None:
        """Raise unless ``expectation_type`` averages the tapers of an exact spectrum.

        Only the taper mean of an exact spectrum's outer products is its
        cross-spectral matrix; a single taper is one column of a factor of it.
        """
        if not self._is_exact_spectrum or 2 in EXPECTATION_AXES[expectation_type]:
            return
        supported = ", ".join(
            repr(name) for name, axes in EXPECTATION_AXES.items() if 2 in axes
        )
        msg = (
            f"expectation_type={expectation_type!r} keeps the tapers as an output "
            "axis, but these coefficients are an exact spectrum (is_exact_spectrum "
            "is True, as for VARSpectrum): only their mean over the tapers is the "
            "model's cross-spectral matrix, and each taper is one column of an "
            f"arbitrary factor of it. Use an expectation_type that averages the "
            f"tapers: {supported}."
        )
        raise ValueError(msg)

    def _units_are_tapers(self, leave_out: str) -> bool:
        """Whether ``jackknife(leave_out=...)`` leaves out tapers of the same samples.

        Leaving out an observation of ``"tapers"``, or of ``"trials_tapers"``
        with more than one taper per trial, leaves out one taper of a trial's
        window, unless the transform recorded that the observations are
        segments in time (:attr:`observations_are_time_segments`);
        ``"trials"`` and one taper per trial leave out whole trials.
        """
        if leave_out != "observation" or self._observations_are_time_segments:
            return False
        if self.expectation_type == "tapers":
            return True
        return (
            self.expectation_type == "trials_tapers"
            and self._fourier_coefficients.shape[2] > 1
        )

    def _band_taper_jackknife_message(self, n_trials: int) -> str:
        """Why ``jackknife`` refuses to leave out tapers for ``frequency_bands``."""
        msg = (
            "jackknife(frequency_bands=...) cannot leave out one taper at a time "
            "(leave_out='observation' with "
            f"expectation_type={self.expectation_type!r}): a taper's Fourier "
            "coefficients are correlated with the other tapers' across the "
            "frequencies of a band, so the replicates are not independent and the "
            "band interval is too narrow (a nominal 95% interval covered 43-81% in "
            "simulation). "
        )
        if n_trials < 3:
            return msg + (
                "Leaving out whole trials (leave_out='trial') needs at least 3, and "
                f"these data have {n_trials}. "
                + _NO_BAND_JACKKNIFE_ADVICE
                + " "
                + _SEGMENTS_ADVICE
            )
        if self.expectation_type == "trials_tapers":
            how = "Pass leave_out='trial'"
        else:
            how = (
                "Build the Connectivity with expectation_type='trials_tapers' "
                "(which averages the trials into one estimate; per-trial band "
                "intervals have no valid jackknife) and pass leave_out='trial'"
            )
        return msg + (
            f"{how} to leave out one whole trial at a time; it needs at least 3 "
            f"trials, and these data have {n_trials}. " + _SEGMENTS_ADVICE
        )

    def _band_reducer(
        self,
        method: str,
        frequency_bands: Mapping[str, tuple[float, float]],
        frequency_reduction: str,
        method_kwargs: Mapping[str, Any],
    ) -> Callable[[NDArray[np.floating]], NDArray[np.floating]]:
        """Validate ``jackknife``'s band request and return its band reduction.

        The request is checked as :func:`frequency_band_reduce` checks it
        (bands, reduction, and which measures may be integrated), before any
        measure is computed. The returned function replaces the frequency axis
        of a result of ``method``, ``(..., n_nonnegative_frequencies, ...)``,
        with a band axis, ``(..., n_bands, ...)``, in the mapping's order.
        """
        # These modules import this one, so their helpers are imported here.
        from spectral_connectivity._frequency_bands import (
            _band_bins,
            _band_masks,
            _band_reduction,
            _nyquist_frequency,
            _reduce_band,
            _require_integral_bin_widths,
        )
        from spectral_connectivity._measure_registry import (
            _MEASURE_SPECS,
            _consumes_frequency_range,
        )

        if _consumes_frequency_range(method):
            msg = (
                f"jackknife cannot reduce {method!r} to frequency_bands: bands act on "
                f"frequency-resolved measures, and {method} already combines the "
                "frequency bins of its frequency_range (phase_slope_index and "
                "group_delay reduce them to one value)"
            )
            if method == "group_delay":
                msg += (
                    ", and returns a tuple (delay, slope, r_value), which jackknife "
                    "does not resample."
                )
            else:
                msg += (
                    ". For one interval per band, call jackknife once per band with "
                    "frequency_range=(low, high) and without frequency_bands; "
                    "leave_out='trial' (at least 3 trials) is the safer choice."
                )
            raise ValueError(msg)
        frequencies = self.frequencies
        band_masks_and_bounds = _band_masks(
            frequencies, frequency_bands, frequency_reduction=frequency_reduction
        )
        _require_integral_bin_widths(frequency_reduction, frequencies.size)
        if frequency_reduction == "integral" and method_kwargs.get("one_sided") is False:
            msg = (
                "frequency_reduction='integral' integrates the one-sided power "
                "density, but one_sided=False makes power return the two-sided "
                "density at the non-negative frequencies, whose integral is half "
                "the band power. Pass one_sided=True (or omit it)."
            )
            raise ValueError(msg)
        spec = _MEASURE_SPECS.get(method)
        reduction = _band_reduction(
            method,
            frequency_reduction,
            circular=None,
            units=None if spec is None else spec.units,
            is_complex=False,
        )
        # A two-sided FFT grid has a Nyquist bin only for an even length; a
        # one-sided grid is read as frequency_band_reduce reads one without a
        # recorded Nyquist frequency.
        nyquist_frequency = (
            _nyquist_frequency({}, frequencies)
            if self._is_one_sided
            else (float(frequencies[-1]) if len(self._frequencies) % 2 == 0 else None)
        )
        bands = _band_bins(
            frequencies,
            band_masks_and_bounds,
            frequency_reduction=frequency_reduction,
            nyquist_frequency=nyquist_frequency,
        )
        # The kept observation axes lead every measure's result.
        frequency_axis = len(self._kept_observation_axes)

        def reduce_to_bands(values: NDArray[np.floating]) -> NDArray[np.floating]:
            if values.ndim <= frequency_axis or values.shape[frequency_axis] != len(
                frequencies
            ):
                msg = (
                    f"jackknife cannot reduce {method!r} to frequency_bands: its "
                    f"result, of shape {values.shape}, has no axis of the "
                    f"{len(frequencies)} non-negative frequencies at position "
                    f"{frequency_axis}."
                )
                raise ValueError(msg)
            return np.stack(
                [
                    _reduce_band(
                        np.take(values, bins, axis=frequency_axis),
                        frequency_axis,
                        reduction=reduction,
                        weights=weights,
                    )
                    for bins, weights in bands
                ],
                axis=frequency_axis,
            )

        return reduce_to_bands

    @_refused_for_exact_spectrum("sampling_error")
    def jackknife(
        self,
        method: str,
        *,
        confidence_level: float = 0.95,
        transformation: Literal[
            "auto", "identity", "log", "fisher", "fisher_squared", "circular"
        ] = "auto",
        leave_out: Literal["observation", "trial"] = "observation",
        frequency_bands: Mapping[str, tuple[float, float]] | None = None,
        frequency_reduction: Literal["mean", "integral"] = "mean",
        keep_replicates: bool = False,
        **method_kwargs: Any,
    ) -> JackknifeResult:
        """Estimate uncertainty by leaving out one observation or one trial.

        The configured expectation must average trials, tapers, or their
        combination. By default each averaged observation is left out in turn,
        or each whole trial with ``leave_out="trial"``. The method is
        recomputed for every leave-one-out sample, so it supports nonlinear
        measures without an analytic variance formula (at a computational cost
        proportional to the number of samples left out).

        Parameters
        ----------
        method : str
            Name of a public, real-valued connectivity measure to recompute,
            e.g. ``"magnitude_squared_coherence"``. Complex-valued and
            tuple-valued measures are not supported. The deprecated
            ``"coherence_magnitude"`` is accepted with a ``DeprecationWarning``
            through 3.x.
        confidence_level : float, default=0.95
            Two-sided coverage of the interval, in (0, 1). The critical value
            is Student t with ``n_observations - 1`` degrees of freedom.
        transformation : {"auto", "identity", "log", "fisher",
                          "fisher_squared", "circular"}
            Scale on which the interval is formed. ``"auto"`` resolves to:

            - ``"log"`` for ``power``;
            - ``"fisher_squared"`` (``atanh(sqrt(.))``) for the
              magnitude-squared measures in ``[0, 1]``: ``magnitude_squared_coherence``
              and ``partial_coherence``;
            - ``"fisher"`` (``atanh(.)``) for the magnitudes in ``[0, 1]``:
              ``phase_locking_value`` and ``imaginary_coherence``. For these
              two measures the Fisher lower bound and bias-corrected estimate
              are clamped at 0, since a magnitude cannot be negative;
            - ``"circular"`` for ``coherence_phase``;
            - ``"identity"`` for every other measure.
        leave_out : {"observation", "trial"}, default="observation"
            What one leave-one-out replicate omits. ``"observation"`` omits
            one averaged observation (a trial-taper eigencoefficient for
            ``trials_tapers``). ``"trial"`` omits one whole trial (the
            coefficients' trial axis), all of its tapers together, so the
            interval has ``n_trials - 1`` degrees of freedom and reflects how
            much the measure varies from trial to trial.

            Prefer ``"trial"`` for recordings of several trials unless they
            are known to be alike: when trials differ (in coupling strength,
            state, or noise level), leaving out one taper at a time treats the
            tapers of one trial as independent draws from one shared spectrum
            and gives too narrow an interval. Its cost is a wider interval
            when trials are alike and few (see Notes). Keep ``"observation"``
            for a single trial.

            ``"trial"`` needs an expectation that averages trials
            (``"trials_tapers"``, or ``"trials"``, for which the two settings
            are the same) and at least 3 trials. It assumes the trials are
            independent, but not the tapers within one, so it is allowed when
            ``observations_are_independent`` is ``False``.

            A statistic over several frequencies (``frequency_bands``, or one
            computed from ``JackknifeResult.replicates``) should not leave out
            tapers; see Notes.
        frequency_bands : mapping of str to (float, float), optional
            Band names and their ``(low, high)`` bounds in the units of
            :attr:`frequencies`, both endpoints included, as in
            :func:`~spectral_connectivity.frequency_band_reduce`. The measure
            of the full data and of every replicate is reduced to these bands
            before the interval is formed, and ``transformation`` applies to
            the band value (the log of band power, ``atanh(sqrt(.))`` of the
            band-mean magnitude-squared coherence). The units left out must
            not be tapers of the same samples: use ``leave_out="trial"``, an
            expectation whose observations are trials (``"trials"``, or
            ``"trials_tapers"`` with one taper per trial), or observations
            that are segments in time (:attr:`observations_are_time_segments`,
            recorded by ``Connectivity.from_transform(Welch(...))``). ``None``
            (the default) keeps every frequency.
        frequency_reduction : {"mean", "integral"}, default="mean"
            How ``frequency_bands`` combine a band's bins, as in
            :func:`~spectral_connectivity.frequency_band_reduce`: the mean
            of the measure over the bins in the band (circular for
            ``coherence_phase``), or for ``power`` only, its integral over the
            band (band power). Ignored without ``frequency_bands``.
        keep_replicates : bool, default=False
            Return the leave-one-out replicates as ``JackknifeResult.replicates``,
            for an interval on another statistic of the measure through
            :func:`~spectral_connectivity.statistics.jackknife_confidence_interval`.
            They take ``n_replicates`` times the memory of the estimate, so
            they are dropped unless asked for.
        **method_kwargs
            Keyword arguments forwarded to ``method`` on every replicate.

        Returns
        -------
        JackknifeResult
            Estimate, bias-corrected estimate, standard error, and confidence
            bounds, each with the measure's own shape (for a pairwise measure
            and the default expectation, ``(n_time,
            n_nonnegative_frequencies, n_signals, n_signals)``; with
            ``expectation_type="trials"`` the taper axis follows time). With
            ``frequency_bands`` the frequency axis is replaced by a band axis,
            in the mapping's order (``(n_time, n_bands, n_signals,
            n_signals)`` for the default expectation), and
            ``JackknifeResult.frequency_bands`` names it. ``leave_out`` records
            the unit left out, and with ``keep_replicates=True``
            ``replicates`` holds the leave-one-out replicates,
            ``(n_replicates, *estimate.shape)``.
            For directed measures ``[..., i, j]`` is the influence ``i -> j``,
            the same order as the xarray wrapper's ``sel(source=i, target=j)``.

        Raises
        ------
        ValueError
            If ``leave_out="observation"`` and the expectation averages tapers
            (``"tapers"`` or ``"trials_tapers"``) but
            ``observations_are_independent`` is ``False``: correlated
            observations (a MorletWavelet smoothing neighborhood, overlapping
            Welch segments) are not valid leave-one-out units. With
            ``"trials_tapers"`` and at least 3 trials, pass
            ``leave_out="trial"``; otherwise recompute the transform with
            independent observations: ``Welch`` with ``segment_overlap <=
            0.5``, a ``Multitaper`` transform, or ``MorletWavelet`` without
            ``smoothing_time`` on at least 3 trials.
            (``expectation_type="trials"`` is accepted but keeps the correlated
            axis as an output dimension instead of averaging it, a different
            measure.)

            Also raised if ``leave_out`` is not ``"observation"`` or
            ``"trial"``, or is ``"trial"`` with an expectation that does not
            average trials or with fewer than 3 trials.

            With ``frequency_bands``, in this order after ``method`` is
            checked: for the bands and reductions
            :func:`~spectral_connectivity.frequency_band_reduce` refuses (a
            band containing no bin, ``"integral"`` of a measure other than
            ``power``), and ``"integral"`` with ``one_sided=False``, which
            would give half the band power; for a measure that combines the
            bins of its own ``frequency_range`` (``delay``, ``group_delay``,
            ``phase_slope_index``: call it once per band with
            ``frequency_range=(low, high)`` instead); and if
            ``leave_out="observation"`` leaves out tapers
            (``expectation_type="tapers"``, or ``"trials_tapers"`` with more
            than one taper per trial, unless
            :attr:`observations_are_time_segments`), since the interval would
            be too narrow (see Notes). All are raised before any measure is
            computed.

        Notes
        -----
        Where zero ``observation_weights`` drop observations (the invalid
        windows of ``Multitaper(artifact_samples=...)``, say), each bin's
        interval is the jackknife of the observations it keeps: leaving out an
        observation the bin does not have gives no replicate there, and the
        degrees of freedom are its count of kept observations less one
        (:attr:`n_valid_observations`; ``JackknifeResult.n_observations`` is
        then one count per bin). With ``leave_out="trial"`` the units are
        trials, a trial counting in a bin if any of its tapers is kept there.
        A bin keeping fewer than 3 is NaN, with a warning.

        Circular confidence bounds are wrapped to ``(-pi, pi]``, so when the
        interval crosses ``+/-pi`` the lower bound exceeds the upper bound;
        the interval is then the arc from ``lower`` up through ``pi`` and on
        to ``upper``.

        If the input Fourier coefficients were produced with
        ``Multitaper(taper_weighting="adaptive")``, the leave-one-out replicates
        reuse the full-sample Thomson weights (the adaptive weights are not
        recomputed for each reduced taper set), so the interval is an
        approximation in that case.

        The interval describes the size of a measure; it is not a test that
        the measure differs from 0. Monte Carlo coverage of the default
        (``"auto"``) 95% intervals, with independent complex-Gaussian
        observations and 5 to 100 observations per dataset:

        - ``magnitude_squared_coherence`` (``fisher_squared``): 94-95% when the true
          ``|coherency|`` is 0.3-0.8, but at zero true coherence the interval
          excludes 0 in 11-15% of datasets, and more observations do not help
          (the estimated magnitude's spread shrinks with ``n`` at the same
          rate as its mean). "The interval excludes 0" is therefore not
          evidence of nonzero coherence. Test that with the exact
          zero-coherence null instead:
          :func:`spectral_connectivity.statistics.coherence_significance_pvalue`
          applied to ``coherency()`` and ``n_observations`` (valid for
          independent, equally weighted observations; when ``artifact_samples``
          or edge masking drops observations, pass
          ``n_valid_observations[..., np.newaxis, np.newaxis]`` instead),
          corrected across
          frequencies and pairs with
          :func:`spectral_connectivity.statistics.adjust_for_multiple_comparisons`.
          For measures without an analytic null, use a permutation or
          surrogate test.
        - ``phase_locking_value`` (``fisher``): at zero true PLV the interval
          excludes 0 in 5% of datasets at 5 observations but 14% at 100; at a
          high true PLV it under-covers (85-95% at a true PLV of 0.82, 76-90%
          at 0.93).
        - ``coherence_phase`` (``circular``): under-covers when coherence is
          weak (77-88% at a true ``|coherency|`` of 0.1, 84-93% at 0.2,
          87-94% at 0.3, against 92-95% at 0.6).

        Leaving out trials: with 5 to 20 trials of 3 or 7 tapers each and the
        same pooled magnitude-squared coherence (0.45 ** 2), the default 95%
        interval for ``magnitude_squared_coherence`` covers 94-96% with either
        setting when every trial has the same coherency. When each trial's
        coherency is drawn from U(0, 0.9), ``leave_out="observation"`` covers
        82-92% (less with 7 tapers per trial than with 3), because it treats
        one trial's tapers as independent draws, and ``leave_out="trial"``
        93.5-96.5%, from ``n_tapers`` times fewer replicates. When the trials
        are alike, the median ``"trial"`` interval is wider by 19-47% with 3
        or 4 trials, 13-20% with 5 and 5-8% with 10, at the same coverage
        (two Monte Carlo runs of 1000-2000 datasets each).

        Band statistics: different tapers' coefficients are uncorrelated at
        one frequency but correlated (0.3-0.6) at frequencies a few bins
        apart, within the bandwidth, so the tapers of one window are not
        independent observations of a band. In Monte Carlo runs with 1-20
        trials of a bivariate VAR (256-2048 samples, time-halfbandwidth product
        3-4), a 95% interval leaving out one taper at a time covered only
        43-81% for band means and band maxima of a measure, while leaving out
        whole trials covered 91-96% for band means and 83-93% for band maxima.
        ``frequency_bands`` therefore refuses to leave out tapers of the same
        samples. Segments of a trial in time (``Welch``) are not correlated
        that way and may be left out one at a time: with disjoint segments a
        band-mean coherence interval covered about 95%, and about 93% at half
        overlap.

        ``phase_slope_index``, which combines adjacent bins inside its own fit
        over ``frequency_range``, is less affected: 92.8-94.7% leaving out
        tapers against 95.0-95.3% leaving out trials (1000 datasets each, 5-10
        trials). ``delay`` has not been measured. For any other statistic over
        frequencies computed from ``JackknifeResult.replicates``, the safe
        choice is replicates from ``leave_out="trial"`` unless its coverage
        with other units has been checked.

        Examples
        --------
        >>> import numpy as np
        >>> from spectral_connectivity import Connectivity, Multitaper
        >>> rng = np.random.default_rng(0)
        >>> signals = rng.standard_normal((500, 10, 2))  # (time, trials, signals)
        >>> signals[..., 1] += 0.5 * signals[..., 0]
        >>> multitaper = Multitaper(signals, sampling_frequency=500)
        >>> connectivity = Connectivity.from_transform(multitaper)
        >>> result = connectivity.jackknife(
        ...     "magnitude_squared_coherence",
        ...     leave_out="trial",
        ...     frequency_bands={"theta": (4, 8), "beta": (13, 30)},
        ... )
        >>> result.estimate.shape  # (n_time, n_bands, n_signals, n_signals)
        (1, 2, 2, 2)
        >>> list(result.frequency_bands)  # the band axis, in order
        ['theta', 'beta']
        >>> lower, upper = result.confidence_interval
        >>> bool(lower[0, 0, 0, 1] < result.estimate[0, 0, 0, 1] < upper[0, 0, 0, 1])
        True
        """
        if self.expectation_type not in {"trials", "tapers", "trials_tapers"}:
            msg = (
                "jackknife leaves out one trial or taper at a time, so it needs a "
                "Connectivity built with expectation_type='trials_tapers' (the "
                "default), 'trials', or 'tapers'; this one has "
                f"expectation_type={self.expectation_type!r}. Build one for the "
                "interval, e.g. Connectivity.from_transform(transform) or "
                "Connectivity(fourier_coefficients, expectation_type='trials_tapers')."
            )
            raise ValueError(msg)
        _validate_confidence_level(confidence_level)
        keep_replicates = _validated_flag("keep_replicates", keep_replicates)
        if leave_out not in ("observation", "trial"):
            msg = f"leave_out must be 'observation' or 'trial'; got {leave_out!r}."
            raise ValueError(msg)
        if leave_out == "trial" and self.expectation_type == "tapers":
            msg = (
                "jackknife(leave_out='trial') leaves out one trial at a time, but "
                "expectation_type='tapers' does not average trials (each trial is "
                "its own output). Build the Connectivity with "
                "expectation_type='trials_tapers' to average trials and leave one "
                "out, or keep leave_out='observation' to leave out one taper."
            )
            raise ValueError(msg)
        method, measure = self._resampled_measure(method, "jackknife")
        reduce_to_bands = (
            None
            if frequency_bands is None
            else self._band_reducer(
                method, frequency_bands, frequency_reduction, method_kwargs
            )
        )
        coefficients = self._fourier_coefficients
        n_trials = coefficients.shape[1]
        # Each replicate leaves out one index of ``observation_axis``; for one
        # trial-taper observation of ``trials_tapers`` the two axes are
        # flattened into axis 1 first.
        flatten_trials_tapers = (
            self.expectation_type == "trials_tapers" and leave_out == "observation"
        )
        observation_axis = 2 if self.expectation_type == "tapers" else 1
        n_observations = (
            n_trials * coefficients.shape[2]
            if flatten_trials_tapers
            else coefficients.shape[observation_axis]
        )
        if (
            leave_out == "observation"
            and not self._observations_are_independent
            and self.expectation_type in {"tapers", "trials_tapers"}
        ):
            msg = (
                f"jackknife with expectation_type={self.expectation_type!r} leaves "
                "out one taper observation at a time, but this transform's "
                "observations are correlated (observations_are_independent is "
                "False: a MorletWavelet smoothing neighborhood, or Welch segments "
                "overlapping by more than half), so leave-one-out intervals over "
                "them are invalid. For an interval on the same averaged measure, "
            )
            if self.expectation_type == "trials_tapers" and n_trials >= 3:
                msg += (
                    "pass leave_out='trial', which leaves out whole trials and is "
                    "valid with correlated tapers, or "
                )
            msg += (
                "recompute the transform with independent observations: Welch "
                "with segment_overlap <= 0.5, a Multitaper transform, or "
                "MorletWavelet without smoothing_time on at least 3 trials (the "
                "trials are then the observations, and there is no time "
                "smoothing). expectation_type='trials' keeps the correlated axis "
                "as an output dimension instead of averaging it, so it measures "
                "something else."
            )
            raise ValueError(msg)
        if frequency_bands is not None and self._units_are_tapers(leave_out):
            raise ValueError(self._band_taper_jackknife_message(n_trials))

        if n_observations < 3:
            if leave_out == "trial":
                msg = (
                    "jackknife(leave_out='trial') requires at least 3 trials, got "
                    f"{n_observations}: an interval over n trials has n - 1 degrees "
                    "of freedom, so 2 trials give a critical value near 13 and 1 "
                    "trial gives none."
                )
                if frequency_bands is not None and self._units_are_tapers("observation"):
                    msg += " " + _NO_BAND_JACKKNIFE_ADVICE
                elif (
                    self.expectation_type == "trials_tapers"
                    and self._observations_are_independent
                ):
                    msg += (
                        " Keep leave_out='observation' to leave out one trial-taper "
                        "observation at a time."
                    )
                else:
                    msg += " Collect at least 3 trials for an interval."
            else:
                msg = (
                    "jackknife requires at least 3 observations, got "
                    f"{n_observations}. With two, each leave-one-out replicate has a "
                    "single observation, which forces magnitude-normalized measures "
                    "to 1 and makes the interval degenerate (a zero standard error, "
                    "or NaN under a variance-stabilizing transform such as the "
                    "coherence default)."
                )
            raise ValueError(msg)

        full_estimate = _real_array_result(
            measure(**method_kwargs),
            method,
            "jackknife",
            structured_hint=(
                " Jackknife a measure that returns one real array, such as a "
                "pairwise measure; group measures are not resampled yet."
            ),
        )
        if reduce_to_bands is not None:
            full_estimate = reduce_to_bands(full_estimate)

        observation_weights = self._observation_weights
        observation_coefficients = coefficients
        if flatten_trials_tapers:
            observation_coefficients = coefficients.reshape(
                coefficients.shape[0],
                n_observations,
                1,
                coefficients.shape[3],
                coefficients.shape[4],
            )
            if observation_weights is not None:
                observation_weights = observation_weights.reshape(
                    observation_weights.shape[0],
                    n_observations,
                    1,
                    observation_weights.shape[3],
                    1,
                )
        # Where zero weights drop observations, which units each kept bin has:
        # (n_units, *kept observation axes), a unit counting in a bin if it has
        # a nonzero weight at any frequency there. Leaving out a unit a bin
        # never had gives no replicate there.
        unit_is_valid = None
        if observation_weights is not None:
            positive = observation_weights[..., 0] > 0
            if leave_out == "trial" and self.expectation_type == "trials_tapers":
                # The unit is a whole trial, kept in a bin if any of its
                # tapers is.
                positive = xp.any(positive, axis=2, keepdims=True)
            present = xp.any(positive, axis=-1)
            # A frequency keeping only some of a bin's units would need its own
            # replicates and count; one keeping none is NaN there anyway.
            if not bool(
                xp.all(
                    (positive == present[..., xp.newaxis])
                    | ~xp.any(positive, axis=observation_axis, keepdims=True)
                )
            ):
                msg = (
                    "jackknife needs observation_weights that drop the same "
                    "observations at every frequency of a bin (or all of them), as "
                    "Multitaper(artifact_samples=...) and MorletWavelet(edge_mode='nan') "
                    "do; these drop an observation at some frequencies only, which "
                    "would give those frequencies a different count of replicates. "
                    "Compute the interval one frequency at a time, or zero an "
                    "observation's weight at every frequency."
                )
                raise ValueError(msg)
            if self.expectation_type == "trials_tapers":
                present = present[:, :, 0]
            present = to_numpy(xp.moveaxis(present, observation_axis, 0))
            if not present.all():
                unit_is_valid = present
        units = "trials" if leave_out == "trial" else "observations"
        # The raw count was checked above; these are the units kept anywhere.
        if unit_is_valid is not None and (n_valid := int(unit_is_valid.sum(0).max())) < 3:
            caller = "jackknife(leave_out='trial')" if leave_out == "trial" else "jackknife"
            msg = (
                f"{caller} requires at least 3 {units}, got {n_valid} with "
                "nonzero weight in any bin. With two, "
                "each leave-one-out replicate has a single one, which forces "
                "magnitude-normalized measures to 1 and makes the interval "
                "degenerate (a zero standard error, or NaN under a "
                "variance-stabilizing transform such as the coherence default)."
            )
            raise ValueError(msg)
        omitted_units: Sequence[int] = range(n_observations)
        if unit_is_valid is not None:
            omitted_units = np.flatnonzero(
                unit_is_valid.reshape(n_observations, -1).any(axis=1)
            ).tolist()

        replicates: list[NDArray[np.floating]] = []
        # The full estimate carried the migration notices; the replicates do not.
        token = _emit_migration_notices.set(False)
        try:
            for omitted in omitted_units:
                keep = xp.arange(n_observations) != omitted
                subset = xp.compress(keep, observation_coefficients, axis=observation_axis)
                subset_weights = (
                    None
                    if observation_weights is None
                    else xp.compress(keep, observation_weights, axis=observation_axis)
                )
                replicate_connectivity = self._with_coefficients(subset, subset_weights)
                # Only a bin with fewer than 3 kept observations leaves a
                # replicate a single one, and the interval is NaN there.
                with _user_warnings_ignored(
                    _SINGLE_KEPT_OBSERVATION_WARNINGS, unit_is_valid is not None
                ):
                    replicate = getattr(replicate_connectivity, method)(**method_kwargs)
                if isinstance(replicate, tuple) or np.iscomplexobj(replicate):
                    msg = f"jackknife requires a real array result from {method!r}."
                    raise TypeError(msg)
                replicate = np.asarray(replicate)
                replicates.append(
                    replicate if reduce_to_bands is None else reduce_to_bands(replicate)
                )
        finally:
            _emit_migration_notices.reset(token)

        resolved_transformation: Literal[
            "identity", "log", "fisher", "fisher_squared", "circular"
        ]
        if transformation == "auto":
            if method == "power":
                resolved_transformation = "log"
            elif method in {"magnitude_squared_coherence", "partial_coherence"}:
                # These return magnitude-*squared* coherence, whose
                # variance-stabilizing transform is atanh(sqrt(.)), not atanh(.).
                resolved_transformation = "fisher_squared"
            elif method in _NONNEGATIVE_MAGNITUDE_MEASURES:
                # Unsquared magnitudes in [0, 1]: Fisher's atanh applies directly.
                resolved_transformation = "fisher"
            elif method == "coherence_phase":
                resolved_transformation = "circular"
            else:
                resolved_transformation = "identity"
        else:
            resolved_transformation = transformation
        leave_one_out = np.stack(replicates, axis=0)
        replicate_is_valid = None
        if unit_is_valid is not None and not unit_is_valid[omitted_units].all():
            # Each bin's replicates lead the measure's own axes.
            kept_shape = unit_is_valid.shape[1:]
            replicate_is_valid = unit_is_valid[omitted_units].reshape(
                (
                    len(omitted_units),
                    *kept_shape,
                    *(1,) * (full_estimate.ndim - len(kept_shape)),
                )
            )
            # Leaving out a unit a bin never had gives no replicate there.
            leave_one_out[~np.broadcast_to(replicate_is_valid, leave_one_out.shape)] = np.nan
            too_few = unit_is_valid.sum(axis=0) < 3
            if too_few.any():
                warnings.warn(
                    f"jackknife: observation_weights leave fewer than 3 {units} "
                    f"in {int(too_few.sum())} of {too_few.size} bins (see "
                    "n_valid_observations), too few for a leave-one-out interval, so "
                    "the bias-corrected estimate, standard error and bounds are NaN "
                    "there.",
                    UserWarning,
                    stacklevel=stacklevel_outside_package(),
                )
                replicate_is_valid = replicate_is_valid & ~too_few.reshape(
                    replicate_is_valid.shape[1:]
                )
        result = jackknife_confidence_interval(
            full_estimate,
            leave_one_out,
            confidence_level=confidence_level,
            transformation=resolved_transformation,
            # PLV's diagonal is 1 by definition, not a saturated estimate.
            _saturated_by_construction=(
                np.eye(self.n_signals, dtype=bool)
                if method == "phase_locking_value"
                else False
            ),
            _replicate_is_valid=replicate_is_valid,
            # NaN replicates that are not dropped units are NaN values here.
            _skip_nan_replicates=False,
        )
        if resolved_transformation == "fisher" and method in _NONNEGATIVE_MAGNITUDE_MEASURES:
            # tanh maps the atanh-scale interval onto [-1, 1], but a magnitude
            # cannot be negative: clamp at 0, as fisher_squared's back-transform
            # does for magnitude-squared coherence.
            lower, upper = result.confidence_interval
            result = replace(
                result,
                bias_corrected=np.maximum(result.bias_corrected, 0.0),
                confidence_interval=(np.maximum(lower, 0.0), upper),
            )
        return replace(
            result,
            replicates=leave_one_out if keep_replicates else None,
            frequency_bands=None if frequency_bands is None else dict(frequency_bands),
            leave_out=leave_out,
        )

    @_refused_for_exact_spectrum("surrogate_null")
    def surrogate_test(
        self,
        method: str,
        *,
        kind: Literal["trial_shuffle", "phase_rotation", "conjugation"],
        n_surrogates: int = 999,
        alternative: Literal["auto", "greater", "less", "two-sided"] = "auto",
        fwer_axis: int | tuple[int, ...] | None = None,
        half_bandwidth: float | None = None,
        random_state: int | np.random.Generator | None = None,
        **method_kwargs: Any,
    ) -> SurrogateTestResult:
        """Test a measure against surrogate data that satisfy a null hypothesis.

        Each surrogate is the coefficients changed in a way that leaves their
        distribution unchanged when the null hypothesis of ``kind`` holds. The
        method is recomputed for ``n_surrogates`` surrogates, and each element
        of the estimate gets a Monte Carlo p-value (the rank of the estimate
        among the surrogates' values) and a p-value adjusted for testing every
        element of its family (the maximum statistic over the family).

        Choose ``kind`` for the question:

        ==========================================  ====================  ==================================  =================================
        Question                                    ``kind``              Needs                               A shared stimulus-locked response
        ==========================================  ====================  ==================================  =================================
        Do the signals covary from trial to trial?  ``"trial_shuffle"``   2+ trials                           stays in the null
        Are the signals independent at all?         ``"phase_rotation"``  2+ units (see Notes), the half      is rejected
                                                                          bandwidth
        Is there more than zero-lag mixing?         ``"conjugation"``     2+ units (see Notes)                is rejected if lagged
        ==========================================  ====================  ==================================  =================================

        A shared stimulus-locked response is one common to the signals and the
        same in every trial (see Notes).

        Parameters
        ----------
        method : str
            Name of a public, real-valued connectivity measure of pairs of
            signals, e.g. ``"magnitude_squared_coherence"``. Complex-valued,
            tuple-valued and structured measures are refused, as are
            ``power`` (every surrogate keeps each signal's spectrum),
            ``delay`` (candidate delays, not a statistic) and
            ``coherence_phase`` (circular).
        kind : {"trial_shuffle", "phase_rotation", "conjugation"}
            The surrogate construction, which sets the null hypothesis:

            - ``"trial_shuffle"``: permute each signal's trials independently
              (signal 0 keeps its order), keeping every trial's coefficients
              for that signal intact across time windows, tapers and
              frequencies. Null: the signals are independent from trial to
              trial; each signal keeps its own spectrum and whatever it does
              at the same time in every trial. Valid for every measure,
              including the directed ones. Needs an expectation that averages
              trials and at least 2 trials; with ``n_trials! <
              n_surrogates`` (fewer than 7 trials for the default) a warning
              says how coarse the p-values are.
            - ``"phase_rotation"``: multiply each signal's coefficients by an
              independent random phase per unit (each trial, or averaged time
              window; see Notes), the same for every taper and positive
              frequency of the unit (and conjugate at the negative
              frequencies, so a real signal stays real). Null: the signals
              are independent. The test is exact for circular Gaussian
              coefficients; the windowed coefficients of stationary Gaussian
              signals are approximately circular away from 0 Hz and Nyquist,
              where leakage across 0 Hz (or Nyquist) mixes each frequency with
              its mirror image. So the frequencies within ``half_bandwidth``
              of 0 Hz and Nyquist, where rotating does not give valid
              surrogates, are not tested. Needs at least 2 units: one trial
              is refused unless its non-overlapping time windows are
              averaged.
            - ``"conjugation"``: replace each unit (every taper, signal and
              frequency of it together) by its complex conjugate with
              probability 1/2, which for a real signal amounts to reversing
              that unit in time. Null: the coherency is real at every
              frequency, as for instantaneous (zero-lag) mixing such as volume
              conduction of independent sources; exact when the signals are
              stationary and Gaussian, whose distribution such mixing leaves
              unchanged by time reversal. Needs at least 2 units. Conjugation
              leaves real coefficients unchanged, so the frequencies of
              :attr:`real_valued_frequencies` are not tested, nor are 0 Hz and
              Nyquist where their coefficients are real to rounding, as they
              are for a real signal. For the coefficients of a complex signal,
              passed directly, 0 Hz and Nyquist are tested like any other
              frequency. Pair it
              with a measure of lagged interaction: ``imaginary_coherence``,
              the phase-lag indices, ``phase_slope_index``, or pairwise
              spectral Granger (for Granger causality beyond what mixing
              produces). With ``magnitude_squared_coherence`` it asks only
              whether the coupling exceeds what a real coherency gives.
        n_surrogates : int, default=999
            Number of surrogates. The smallest p-value is ``1 / (n_surrogates
            + 1)``; the computation takes ``n_surrogates`` times as long as
            the measure.
        alternative : {"auto", "greater", "less", "two-sided"}, default="auto"
            Which values are evidence against the null: large values of the
            measure (``"greater"``), small values (``"less"``), or large
            magnitudes (``"two-sided"``). ``"auto"`` resolves to
            ``"two-sided"`` for the measures whose sign says which signal
            leads (``imaginary_coherency``, ``phase_lag_index``,
            ``weighted_phase_lag_index``, ``phase_slope_index``) and to
            ``"greater"`` for every other measure, including the debiased
            estimators that can fall below 0 by chance
            (``pairwise_phase_consistency`` and the debiased squared phase-lag
            indices) and ``directed_phase_lag_index``, whose ``[..., i, j]``
            above 0.5 means ``i`` leads ``j``. The result records the
            resolved value.
        fwer_axis : int or tuple of int or None, default=None
            The axes of the measure's result whose elements form one family
            for ``fwer_p_value``, as in
            :func:`~spectral_connectivity.statistics.adjust_for_multiple_comparisons`:
            each slice along the remaining axes is its own family. ``None``
            (the default) makes every tested element one family. For a
            pairwise result ``(n_time, n_frequencies, n_signals,
            n_signals)``, ``fwer_axis=(1, 2, 3)`` corrects across
            frequencies and pairs separately in each time window. The result
            records the axes as non-negative integers (``None`` as every
            axis). ``()`` is refused: with each element its own family, use
            ``p_value``.
        half_bandwidth : float, optional
            For ``kind="phase_rotation"`` only (refused otherwise): the
            half-width of the transform's spectral smoothing, in the units of
            ``frequencies``. The frequencies below it and above Nyquist minus
            it are not tested. Defaults to :attr:`half_bandwidth`, which
            :meth:`from_transform` records for a
            :class:`~spectral_connectivity.Multitaper` transform
            (``frequency_resolution / 2``); pass it for coefficients from
            elsewhere.
        random_state : int, numpy.random.Generator, or None
            Seed or generator of the surrogates' random numbers. ``None``
            draws a fresh seed.
        **method_kwargs
            Keyword arguments forwarded to ``method`` on the data and every
            surrogate.

        Returns
        -------
        SurrogateTestResult
            The estimate, the uncorrected ``p_value`` and the family-wise
            ``fwer_p_value``, each with the measure's own shape (for a
            pairwise measure ``(n_time, n_nonnegative_frequencies, n_signals,
            n_signals)``), and the surrogate maxima of each family. The
            diagonal, non-finite estimates, the frequencies within
            ``half_bandwidth`` of 0 Hz and Nyquist (for ``"phase_rotation"``)
            and the frequencies whose coefficients are real (for
            ``"conjugation"``: :attr:`real_valued_frequencies`, and 0 Hz and
            Nyquist of a real signal) are not tested: their p-values are NaN, and
            they are in no family's maximum. For directed measures
            ``[..., i, j]`` is ``i -> j``.

        Raises
        ------
        ValueError
            If ``kind``, ``n_surrogates``, ``alternative``, ``fwer_axis`` or
            ``half_bandwidth`` is invalid, or ``kind="phase_rotation"`` has no
            half bandwidth; if the expectation does not average what ``kind``
            randomizes (trials for ``"trial_shuffle"``, units otherwise) or
            there are fewer than 2 of them (one trial and no averaged
            non-overlapping time windows, for ``"phase_rotation"`` and
            ``"conjugation"``); if
            ``"trial_shuffle"`` meets observation weights that differ between
            trials; if ``"phase_rotation"`` meets one-sided coefficients, or a
            measure that does not return one value per frequency, or a
            ``phase_slope_index`` band within ``half_bandwidth`` of 0 Hz or
            Nyquist; if this ``Connectivity`` holds an exact spectrum
            (``is_exact_spectrum``, e.g. from a VAR model), which has no
            sampling error to test against; or if ``method`` is not a measure
            this test can use.
        TypeError
            If ``method`` is not a string, or the measure's result is not a
            real array ending in a ``(n_signals, n_signals)`` pair.

        Warns
        -----
        UserWarning
            If the data allow fewer distinct surrogates than ``n_surrogates``
            (the p-values stay valid but coarse), or if surrogates return NaN
            at tested elements (each counts as at least the estimate and sets
            its family's maximum to infinity).

        Notes
        -----
        **What a rejection means.** A p-value at or below ``alpha`` rejects
        the null hypothesis of ``kind`` at that element, not "no
        connectivity" in general. Under the null, the probability of a
        p-value at or below ``alpha`` is at most ``alpha`` (exactly
        ``floor(alpha * (n_surrogates + 1)) / (n_surrogates + 1)`` when the
        surrogates are exchangeable with the data and there are no ties).
        Ties count against rejection, including values within rounding (one
        part in a million) of the estimate, and a surrogate the measure
        returns NaN for counts as exceeding it.

        **Which p-value to report.** ``fwer_p_value``, to say where the
        measure is significant; ``p_value`` is uncorrected (see
        :class:`~spectral_connectivity.statistics.SurrogateTestResult`).

        **Units.** ``"phase_rotation"`` and ``"conjugation"`` draw one phase
        or flip for each unit the expectation averages: each trial, and each
        time window too when the expectation averages time and the windows do
        not overlap (:attr:`time_bins_are_disjoint`, which
        :meth:`from_transform` reads from a ``Multitaper`` or
        ``ShortTimeFourierTransform``, and which is ``False`` for coefficients
        passed directly unless stated). Windows that overlap, even by half,
        share samples and so have correlated coefficients; all the windows of
        a trial then share its draw, and a single recording with overlapping
        windows is one unit. Every taper of a unit shares its draw too. A
        trial's tapers are uncorrelated only where the spectrum is flat across
        ``half_bandwidth``; near a spectral peak or on a steep spectrum they
        are correlated, and randomizing each taper separately would destroy
        that correlation and make the null too narrow (see Measured sizes).
        Randomizing whole units keeps it, so for independent units of
        stationary Gaussian signals the test holds its size whatever their
        spectrum (for ``"phase_rotation"``, away from 0 Hz and Nyquist). With
        a single unit (one trial, no averaged non-overlapping windows)
        whole-unit randomization cannot change the magnitude of a
        cross-spectrum (rotation) or does no more than conjugate the whole
        spectrum (conjugation), so it is refused: average non-overlapping time
        windows (an ``expectation_type`` that includes ``"time"``), record
        more trials, or use ``"trial_shuffle"`` where there are trials.
        Windows of one recording are independent units only if they are
        longer than the signals' memory: a narrow spectral peak has a long one
        (see Measured sizes). The axes the expectation keeps (each time window
        by default) share the draws, so the surrogates keep the dependence
        between them that the family maximum depends on.

        **Shared stimulus-locked responses.** ``"trial_shuffle"`` keeps each
        signal's response in every surrogate, so a response common to the
        signals is in its null: the test asks about trial-by-trial
        covariation beyond it. ``"phase_rotation"`` randomizes each trial's
        phase, which destroys stimulus-locked timing, so a shared response
        alone is significant. ``"conjugation"`` keeps a response
        shared at zero lag in its null (its coherency is real) and rejects
        one that reaches the signals at different times. To ask about
        interaction beyond a common drive, use ``"trial_shuffle"``.
        Subtracting each signal's mean across trials before the transform
        removes the response but leaves every trial's residual correlated
        with the others (by ``-1 / (n_trials - 1)``), which per-trial
        randomization ignores, so the other two kinds then give p-values that
        are too small unless there are many trials (see Measured sizes).

        **Multiple comparisons.** ``fwer_p_value`` controls the family-wise
        error rate within each family, not across families: choose the
        families before looking at the results, to match the claims you will
        make (see the Notes of
        :func:`~spectral_connectivity.statistics.adjust_for_multiple_comparisons`).
        The raw maximum is dominated by the elements whose null distribution
        is widest, so elements with a narrow null (frequencies with fewer
        effective observations, or a measure with a smaller range there) lose
        power to the others. The diagonal is never in a family. Each pair of
        a symmetric measure appears twice, which does not change the
        maximum. For a false discovery rate instead, pass ``p_value`` to
        :func:`~spectral_connectivity.statistics.adjust_for_multiple_comparisons`.

        **Measures that combine frequencies.** For ``"phase_rotation"`` the
        frequencies near 0 Hz and Nyquist are left untested in the output, but
        the measures that factor the whole spectrum (the spectral Granger
        family, the transfer-function measures) still use them, which spreads
        an excess past ``half_bandwidth``, and with it an excess family-wise
        rate (see Measured sizes). For these measures prefer
        ``"trial_shuffle"`` where the trials allow it, or exclude the
        frequencies within about two half bandwidths of an edge from a claim.
        ``phase_slope_index`` sums over its ``frequency_range``, so for
        ``"phase_rotation"`` that band must avoid the untested frequencies;
        for ``"conjugation"`` it is tested whole.

        **Measured sizes.** Monte Carlo rejection rates at nominal 5% with 99
        surrogates per dataset and multitaper transforms. These were measured
        once, during development; the test suite checks each kind's size and
        the family-wise error rate in smaller designs, within stated limits.

        - ``"trial_shuffle"``, three independent AR(1) signals (coefficient
          0.5) at 200 Hz, 10 trials, 1 s windows with NW=3 (5 tapers), 2100
          datasets: 4.9-5.1% for magnitude-squared and imaginary coherence and
          pairwise spectral Granger, including 0 Hz and Nyquist.
        - ``"phase_rotation"``, two independent AR(1) signals (coefficient
          0.5) at 200 Hz, 5 trials, 2 s windows with NW=4 (7 tapers, half
          bandwidth 2 Hz), 2000 datasets: 5.0% for magnitude-squared and
          imaginary coherence at frequencies more than 2.5 half bandwidths
          from an edge and 4.7-5.4% nearer, but past one half bandwidth.
          Within one half bandwidth of 0 Hz or Nyquist (computed apart, 1000
          datasets) rotation rejected up to 7.9% (coherence) and 7.4%
          (Granger) at single frequencies, which is why those are not tested.
          Pairwise spectral Granger rejected 6.5% from one to 1.5 half
          bandwidths from an edge, 5.8-5.9% up to 2.5, and 5.1% farther away.
        - ``"conjugation"``, instantaneous mixing of two independent AR(2)
          sources at 200 Hz, peaked at 20 and 40 Hz with pole radii 0.9 and
          0.8 (each signal one source plus 0.7 times the other), 10 trials, 2
          s windows with NW=4 (7 tapers), 400 datasets: 4.9-5.2% for
          ``imaginary_coherence``, 5.0-5.1% for pairwise spectral Granger and
          4.0-4.5% for a two-sided ``phase_slope_index`` over 10-30 Hz, where
          the trial shuffle finds Granger causality in 96% of datasets. With a
          3-sample lag instead (unmixed, the second source plus half the first
          delayed by 3 samples), conjugation rejects the imaginary coherence
          at its 20 Hz peak, and the phase slope index, in every dataset, and
          pairwise Granger in 93% (10-90 Hz).
        - Spectra that are not flat across the half bandwidth, two independent
          signals at 128 Hz, 1 s windows with NW=4 (7 tapers, half bandwidth 4
          Hz), 300-400 datasets. Around a narrow 20 Hz peak (AR(2), pole
          radius 0.99), at 17-23 Hz, with 10 trials: 4.6% (rotation) and 5.2%
          (conjugation), against 41% and 27% when each taper was randomized
          separately (57% and 28% with one trial). Just past the half
          bandwidth of a steep spectrum (AR(1), coefficient 0.99), at 4-7 Hz,
          with 10 trials: 5.1% and 4.7%, against 18% and 14% per taper (19%
          and 10% with one trial). For one 10 s recording, its ten 1 s windows
          averaged (``expectation_type="time_tapers"``): 4.3-4.8% for the
          steep spectrum and 5.3-5.4% for a broadband one (AR(1), coefficient
          0.5), but 13% (rotation) and 10% (conjugation) around the narrow
          peak, whose memory (about 0.8 s) spans the windows.
        - Overlapping windows, two independent AR(1) signals (coefficient 0.5)
          at 128 Hz, magnitude-squared coherence at every tested frequency,
          300 datasets. One 10 s recording in 1 s windows with NW=4 every 0.5
          s (19 windows, ``expectation_type="time_tapers"``): randomizing each
          window separately rejected 19.5% (rotation) and 15.0% (conjugation),
          against 4.5% and 5.0% for ten non-overlapping windows of the same
          data. With 5 trials of 4 s in the same windows
          (``expectation_type="time_trials_tapers"``, 7 windows each): 17.7%
          and 13.3% per window, against 5.3% and 1.7% with one draw per trial,
          the unit used for windows that overlap (5 trials allow only 16
          distinct conjugations, so those p-values are coarse and
          conservative).
        - Family-wise error under the global null, 2100 datasets of the
          trial-shuffle design with every tested frequency and pair one
          family: some element was rejected in 5.0-5.1% of datasets with
          ``"trial_shuffle"``, 4.9-5.1% with ``"phase_rotation"`` and 4.8-5.3%
          with ``"conjugation"`` for magnitude-squared and imaginary
          coherence, against every dataset with the uncorrected p-values.
          Pairwise spectral Granger gave 6.7% with ``"phase_rotation"`` (the
          excess near Nyquist above) and 5.8% with ``"conjugation"``. In the
          rotation design above (5 trials) the family-wise rate was 4.0%
          (magnitude-squared coherence), 5.4% (imaginary coherence) and 5.5%
          (Granger).
        - Correlated observations, ``Welch`` segments of 0.5 s overlapping by
          75% (13 per 2 s trial) of the AR(2) sources of the conjugation
          design at 200 Hz, 1000 datasets: one phase per trial held 5.0-5.2%
          for the independent sources (5 trials, magnitude-squared coherence
          and pairwise Granger), against 19.6% when each segment was rotated
          separately; under the mixing above with 10 trials, conjugation of
          whole trials held 4.9%, against 11.5-14.8% per segment.
        - A shared stimulus-locked response: the two AR(2) sources of the
          conjugation design, unmixed, plus the same 10 Hz burst in each of 10
          trials (a cosine of amplitude 3 under a Gaussian envelope of
          standard deviation 50 ms, centered 0.8 s into each 2 s trial; NW=4,
          600 datasets), rejection at 5-15 Hz. With the burst at zero lag:
          4.5-5.6% for the trial shuffle, 3.9-5.5% for conjugation, and 71-72%
          (magnitude-squared coherence) and 51-52% (pairwise Granger) for
          phase rotation (1.4-1.9% for the imaginary coherence, which a
          zero-lag response does not change); rotation's pairwise Granger also
          rejected 6.5-6.6% at 25-90 Hz, away from the burst. With the burst
          15 ms later in one signal: 4.2-6.0% for the trial shuffle and 58-81%
          for the other two. After subtracting each signal's mean across
          trials, phase rotation rejected 5.5-7.0% and conjugation 5.0-6.3% at
          every frequency with 10 trials (6.9% and 6.0% at 5-95 Hz for
          magnitude-squared coherence of independent AR(1) signals,
          coefficient 0.5, with no response, 400 datasets) and 5.3% and 5.2%
          with 40 trials, while the trial shuffle kept 4.9-6.1%.

        References
        ----------
        .. [1] Nichols, T. E., and Holmes, A. P. (2002). Nonparametric
               permutation tests for functional neuroimaging: a primer with
               examples. Human Brain Mapping 15, 1-25.

        Examples
        --------
        Signal 1 is signal 0 delayed by 3 samples (6 ms), plus noise; signal 2
        is independent. Conjugation tests for lagged interaction:

        >>> import numpy as np
        >>> from spectral_connectivity import Connectivity, Multitaper
        >>> rng = np.random.default_rng(1)
        >>> leader = rng.standard_normal((503, 10))
        >>> signals = np.stack(
        ...     [leader[3:], leader[:-3], rng.standard_normal((500, 10))], axis=-1
        ... )  # (time, trials, signals)
        >>> signals += 0.5 * rng.standard_normal(signals.shape)
        >>> multitaper = Multitaper(signals, sampling_frequency=500, time_halfbandwidth_product=2)
        >>> connectivity = Connectivity.from_transform(multitaper)
        >>> result = connectivity.surrogate_test(
        ...     "imaginary_coherence", kind="conjugation", n_surrogates=199, random_state=0
        ... )
        >>> result.p_value.shape  # (n_time_windows, n_frequencies, n_signals, n_signals)
        (1, 251, 3, 3)
        >>> significant = result.fwer_p_value <= 0.05  # one family: every frequency and pair
        >>> bool(significant[0, 40, 0, 1]), bool(significant[..., [0, 1], [2, 2]].any())
        (True, False)

        ``kind="phase_rotation"`` takes the half bandwidth that
        :meth:`from_transform` recorded (here 2 Hz); the cookbook recipe "Test
        against surrogate data" shows more.
        """
        if kind not in _SURROGATE_KINDS:
            msg = (
                f"kind must be one of {', '.join(map(repr, _SURROGATE_KINDS))}, got "
                f"{kind!r}. Each answers a different question: "
                + "; ".join(
                    f"{name!r}: {question}" for name, question in _SURROGATE_QUESTIONS.items()
                )
                + " See Connectivity.surrogate_test."
            )
            raise ValueError(msg)
        if not is_positive_integer(n_surrogates):
            msg = (
                f"n_surrogates must be a positive integer, got {n_surrogates!r}. "
                "The smallest p-value is 1 / (n_surrogates + 1), e.g. 0.001 for 999."
            )
            raise ValueError(msg)
        n_surrogates = int(n_surrogates)
        if alternative not in _SURROGATE_ALTERNATIVES:
            msg = (
                "alternative must be one of "
                f"{', '.join(map(repr, _SURROGATE_ALTERNATIVES))}, got {alternative!r}."
            )
            raise ValueError(msg)
        if isinstance(fwer_axis, tuple) and not fwer_axis:
            msg = (
                "fwer_axis=() makes every element its own family, so fwer_p_value "
                "would equal p_value while null_maxima kept every surrogate's whole "
                "result (n_surrogates times the measure's size). Use "
                "result.p_value for the uncorrected test of each element, or name "
                "the axes of one family, e.g. fwer_axis=(1, 2, 3) for every "
                "frequency and pair within each time window."
            )
            raise ValueError(msg)
        rotation_half_bandwidth = None
        if kind == "phase_rotation":
            rotation_half_bandwidth = self._rotation_half_bandwidth(half_bandwidth)
        elif half_bandwidth is not None:
            msg = (
                f"half_bandwidth applies only to kind='phase_rotation', whose "
                f"surrogates are invalid within it of 0 Hz and Nyquist; "
                f"kind={kind!r} does not use it. Drop half_bandwidth."
            )
            raise ValueError(msg)
        method, measure = self._resampled_measure(method, "surrogate_test")
        refusal = _surrogate_refusal(method)
        if refusal is not None:
            msg = f"surrogate_test cannot test {method!r}: {_SURROGATE_REFUSALS[refusal]}"
            raise ValueError(msg)
        unit_axes = self._surrogate_unit_axes(kind, n_surrogates)
        if alternative == "auto":
            alternative = _signed_measure_alternative(method)
        estimate = _real_array_result(
            measure(**method_kwargs),
            method,
            "surrogate_test",
            complex_hint=(
                " Test a real measure of it: magnitude_squared_coherence (coupling), "
                "imaginary_coherence or imaginary_coherency (lagged coupling), or "
                "phase_locking_value (phase coupling)."
            ),
        )
        if estimate.shape[-2:] != (self.n_signals, self.n_signals):
            msg = (
                "surrogate_test tests relations between pairs of signals, so the "
                f"measure must end in a (n_signals, n_signals) = ({self.n_signals}, "
                f"{self.n_signals}) pair; {method!r} returned shape {estimate.shape}."
            )
            raise TypeError(msg)
        family_axes = (
            tuple(range(estimate.ndim))
            if fwer_axis is None
            else _normalized_axes(
                fwer_axis,
                estimate.ndim,
                "fwer_axis",
                array_description=(
                    f"{method}'s result, "
                    f"{_axes_description(self._result_dims(method, estimate.ndim))}"
                    f"shape {estimate.shape}"
                ),
            )
        )
        tested = (
            np.isfinite(estimate)
            & ~np.eye(self.n_signals, dtype=bool)
            & self._valid_surrogate_frequencies(
                kind, method, estimate, rotation_half_bandwidth, method_kwargs
            )
        )
        tally = _SurrogateTally(
            _surrogate_statistic(estimate, alternative), tested, family_axes, n_surrogates
        )
        rng = _generator(random_state)
        # The estimate carried the migration notices; the surrogates do not.
        token = _emit_migration_notices.set(False)
        try:
            for _ in range(n_surrogates):
                surrogate_connectivity = self._with_coefficients(
                    self._surrogate_coefficients(kind, unit_axes, rng),
                    self._observation_weights,
                )
                values = getattr(surrogate_connectivity, method)(**method_kwargs)
                tally.add(
                    _surrogate_statistic(
                        _real_array_result(values, method, "surrogate_test"), alternative
                    )
                )
        finally:
            _emit_migration_notices.reset(token)
        if tally.n_surrogates_with_nan:
            warnings.warn(
                f"surrogate_test({method!r}): {tally.n_surrogates_with_nan} of "
                f"{n_surrogates} surrogates returned NaN at tested elements "
                f"({tally.n_nan_values} values in all). Each such value counts as at "
                "least the estimate, which raises p_value there, and sets its "
                "family's maximum to inf, which raises fwer_p_value for the whole "
                "family. Check why the measure is undefined on surrogates (for the "
                "directed measures, a non-convergence warning; raise "
                "minimum_phase_max_iterations), or test fewer elements.",
                UserWarning,
                stacklevel=stacklevel_outside_package(),
            )
        p_value, fwer_p_value = tally.p_values()
        return SurrogateTestResult(
            estimate=estimate,
            p_value=p_value,
            fwer_p_value=fwer_p_value,
            null_maxima=tally.null_maxima,
            n_surrogates=n_surrogates,
            kind=kind,
            alternative=alternative,
            fwer_axis=family_axes,
        )

    def _result_dims(self, method: str, ndim: int) -> tuple[str, ...] | None:
        """The axis names of ``method``'s ``ndim``-dimensional result, if known.

        A registered measure's dimensions (:attr:`_MeasureSpec.dims`, which
        lead with the default expectation's one time axis) with the axes this
        expectation keeps in place of that time axis; ``None`` for an
        unregistered measure or a result of another size.
        """
        # The registry imports this module, so it is imported here.
        from spectral_connectivity._measure_registry import _MEASURE_SPECS

        spec = _MEASURE_SPECS.get(method)
        if spec is None:
            return None
        kept = tuple(("time", "trial", "taper")[axis] for axis in self._kept_observation_axes)
        dims = kept + spec.dims[1:]
        return dims if len(dims) == ndim else None

    def _rotation_half_bandwidth(self, half_bandwidth: float | None) -> float:
        """The half bandwidth ``kind="phase_rotation"`` masks with, or the error it raises.

        An explicit ``half_bandwidth`` overrides the one this instance holds.
        """
        if self._is_one_sided:
            msg = (
                "surrogate_test(kind='phase_rotation') needs two-sided coefficients: "
                "its surrogates are invalid within the half-bandwidth of 0 Hz and of "
                "Nyquist, and one-sided coefficients (e.g. MorletWavelet) state "
                "neither the Nyquist frequency nor one bandwidth for every "
                "frequency. Use kind='trial_shuffle' (at least 2 trials) or "
                "kind='conjugation', or a two-sided transform such as Multitaper."
            )
            raise ValueError(msg)
        if half_bandwidth is None:
            half_bandwidth = self._half_bandwidth
        if half_bandwidth is None:
            msg = (
                "surrogate_test(kind='phase_rotation') needs the half-width of the "
                "transform's spectral smoothing, and this Connectivity does not "
                "know it (half_bandwidth is None). Rotated coefficients within it "
                "of 0 Hz or Nyquist are not valid surrogates, so those frequencies "
                "are not tested. Build the Connectivity with "
                "Connectivity.from_transform(multitaper), which records it, or pass "
                "half_bandwidth= in the units of frequencies (for a Multitaper, "
                "multitaper.frequency_resolution / 2)."
            )
            raise ValueError(msg)
        return _validated_half_bandwidth(half_bandwidth)

    def _surrogate_unit_axes(self, kind: str, n_surrogates: int) -> tuple[int, ...]:
        """Observation axes indexing the independent units a surrogate randomizes.

        For ``trial_shuffle`` the trials (axis 1). For ``phase_rotation`` and
        ``conjugation`` the averaged trials, and the averaged time windows
        when they do not overlap (``time_bins_are_disjoint``); never the
        tapers.
        Coefficients sharing a unit (every taper of a trial) get the same phase
        or flip.

        Raises
        ------
        ValueError
            If there are fewer than 2 units (one trial and no averaged
            non-overlapping time windows).
        """
        shape = self._fourier_coefficients.shape
        averaged = self._expectation_axes
        if kind == "trial_shuffle":
            if 1 not in averaged:
                msg = (
                    "surrogate_test(kind='trial_shuffle') pairs each signal's trials "
                    "with other signals' trials, so the expectation must average "
                    f"trials; this Connectivity has expectation_type="
                    f"{self.expectation_type!r}. Build it with "
                    "expectation_type='trials_tapers' (the default), or average "
                    "non-overlapping time windows (e.g. "
                    "expectation_type='time_tapers') and use kind='phase_rotation' "
                    "or 'conjugation', which randomize each averaged window."
                )
                raise ValueError(msg)
            n_trials = shape[1]
            if n_trials < 2:
                msg = (
                    "surrogate_test(kind='trial_shuffle') needs at least 2 trials to "
                    f"shuffle; got {n_trials}. With one trial, average "
                    "non-overlapping time windows (e.g. expectation_type='time_tapers' "
                    "with time_window_step >= time_window_duration) and use "
                    "kind='phase_rotation' (independence) or kind='conjugation' (no "
                    "lagged interaction); windows that overlap are not independent "
                    "units."
                )
                raise ValueError(msg)
            weights = self._observation_weights
            if weights is not None and not bool(xp.all(weights == weights[:, :1])):
                msg = (
                    "surrogate_test(kind='trial_shuffle') moves each signal's trials, "
                    "but observation_weights differ between trials and are shared by "
                    "the signals, so a shuffled trial would take another trial's "
                    "weight. Use weights equal across trials, or "
                    "kind='phase_rotation' or 'conjugation'."
                )
                raise ValueError(msg)
            # Two signals' trials can be aligned in n_trials! ways; a
            # Monte Carlo p-value cannot resolve finer than that.
            if math.lgamma(n_trials + 1) < math.log(n_surrogates):
                warnings.warn(
                    f"surrogate_test(kind='trial_shuffle'): only {n_trials} trials, "
                    f"so one signal's trials can be paired with another's in only "
                    f"{n_trials}! = {math.factorial(n_trials)} ways (fewer than "
                    f"n_surrogates={n_surrogates}): p-values cannot go below about "
                    f"1/{math.factorial(n_trials)}. They remain valid. Use more "
                    "trials for a finer test.",
                    UserWarning,
                    stacklevel=stacklevel_outside_package(),
                )
            return (1,)
        # Whole trials, and the time windows when the expectation averages
        # windows that do not overlap and are independent; never the tapers
        # of one trial (see Notes).
        windows_are_units = self._time_bins_are_disjoint and self._time_bins_are_independent
        unit_axes = tuple(
            axis for axis in averaged if axis == 1 or (axis == 0 and windows_are_units)
        )
        n_units = int(np.prod([shape[axis] for axis in unit_axes]))
        if n_units < 2:
            raise ValueError(self._single_unit_message(kind, n_units))
        if kind == "conjugation" and n_units - 1 < math.log2(n_surrogates):
            # Flipping every unit conjugates the whole spectrum, which leaves
            # the measures symmetric under conjugation unchanged.
            n_distinct = 2 ** (n_units - 1)
            warnings.warn(
                f"surrogate_test(kind='conjugation'): only {n_units} independent "
                f"units (trials and averaged non-overlapping time windows), so "
                f"conjugation can make "
                f"only 2**{n_units - 1} = {n_distinct} distinct surrogates (fewer "
                f"than n_surrogates={n_surrogates}): p-values cannot go below about "
                f"1/{n_distinct}. They remain valid. Use more trials, or average "
                "more time windows, for a finer test.",
                UserWarning,
                stacklevel=stacklevel_outside_package(),
            )
        return unit_axes

    def _single_unit_message(self, kind: str, n_units: int) -> str:
        """Why ``kind`` cannot randomize these coefficients, and what would let it.

        The remedies follow what the expectation already averages: windows
        that overlap are told to stop overlapping, not to be averaged, and
        the tapers are mentioned only where there are several.
        """
        n_windows, n_trials, n_tapers = self._fourier_coefficients.shape[:3]
        averaged = self._expectation_axes
        msg = (
            f"surrogate_test(kind={kind!r}) randomizes whole independent units "
            "(each trial, and each time window when the expectation averages "
            f"windows that do not overlap), and this Connectivity has {n_units} "
            f"({n_trials} trial(s), {n_windows} time window(s), expectation_type="
            f"{self.expectation_type!r}); it needs at least 2."
        )
        windows_overlap = 0 in averaged and n_windows > 1
        if windows_overlap:
            msg += (
                " Its time windows are averaged but may overlap "
                "(time_bins_are_disjoint is False), and windows that overlap are "
                "not independent units: their coefficients are correlated, and "
                "randomizing each 50%-overlapping window separately rejected "
                "independent signals in 19.5% (phase rotation) and 15.0% "
                "(conjugation) of tests at nominal 5% (one 10 s recording, 1 s "
                "windows, 99 surrogates)."
            )
        if n_tapers > 1:
            msg += (
                " Randomizing the tapers of one trial separately is not offered: "
                "their coefficients are correlated wherever the spectrum is not "
                "flat across the half bandwidth, and for independent signals with "
                "one trial (1 s, 7 tapers, 99 surrogates, nominal 5%) it rejected "
                "57% (phase rotation) and 28% (conjugation) of tests around a "
                "narrow spectral peak, and 19% and 10% just past the half "
                "bandwidth of a steep 1/f**2 spectrum."
            )
        if windows_overlap:
            windows_remedy = (
                "use non-overlapping windows (a Multitaper or "
                "ShortTimeFourierTransform with time_window_step >= "
                "time_window_duration; for coefficients passed directly, "
                "time_bins_are_disjoint=True says their windows do not overlap)"
            )
        elif 0 in averaged:
            windows_remedy = (
                "split the recording into more non-overlapping time windows, "
                "each longer than the signals' memory"
            )
        else:
            windows_remedy = (
                "average non-overlapping time windows (an expectation_type that "
                "includes 'time', e.g. 'time_tapers' for one recording, with "
                "windows longer than the signals' memory, which a narrow spectral "
                "peak makes long)"
            )
        trials_remedy = (
            "use an expectation that averages the trials (kind='trial_shuffle' then works too)"
            if n_trials > 1
            else "record more trials (kind='trial_shuffle' then works too)"
        )
        return f"{msg} To get 2 or more units, {windows_remedy}, or {trials_remedy}."

    def _surrogate_coefficients(
        self,
        kind: str,
        unit_axes: tuple[int, ...],
        rng: np.random.Generator,
    ) -> BackendArray:
        """One surrogate of the coefficients, a fresh array of their shape.

        Random numbers are drawn on the host from ``rng``, so a seed gives the
        same surrogates on every backend.
        """
        coefficients = self._fourier_coefficients
        n_trials, n_frequencies, n_signals = (
            coefficients.shape[1],
            coefficients.shape[3],
            coefficients.shape[4],
        )
        if kind == "trial_shuffle":
            # One permutation of the trials per signal, the same in every time
            # window, taper and frequency; signal 0 keeps its order.
            order = np.tile(np.arange(n_trials), (n_signals, 1))
            order[1:] = rng.permuted(order[1:], axis=-1)
            shuffled: BackendArray = xp.empty(coefficients.shape, dtype=coefficients.dtype)
            shuffled[..., 0] = coefficients[..., 0]
            for signal in range(1, n_signals):
                shuffled[..., signal] = xp.take(
                    coefficients[..., signal], xp.asarray(order[signal]), axis=1
                )
            return shuffled
        unit_shape = tuple(
            coefficients.shape[axis] if axis in unit_axes else 1 for axis in range(3)
        )
        if kind == "conjugation":
            # One flip per unit, shared by every signal and frequency.
            flip = xp.asarray(rng.random((*unit_shape, 1, 1)) < 0.5)
            conjugated: BackendArray = xp.where(flip, xp.conj(coefficients), coefficients)
            return conjugated
        # phase_rotation: one phase per unit and signal. A real signal's
        # coefficients at -f are the conjugates of those at +f, so -f turns the
        # other way and the surrogate stays the coefficients of a real signal
        # (that of the analytic signal rotated by the phase); the real 0 Hz and
        # Nyquist bins, which have no partner, get the sign of its cosine.
        phase = rng.uniform(0.0, 2 * np.pi, (*unit_shape, 1, n_signals))
        rotation = np.exp(1j * phase)
        # The factor is built on the device with only the units' shape (one
        # phase per unit and signal, repeated over frequency), then applied in
        # one broadcast product.
        rotation = xp.asarray(rotation.astype(coefficients.dtype))
        factor = xp.empty((*unit_shape, n_frequencies, n_signals), dtype=coefficients.dtype)
        positive, negative = _paired_frequency_bins(n_frequencies)
        factor[..., positive, :] = rotation
        factor[..., negative, :] = xp.conj(rotation)
        sign = xp.where(xp.real(rotation) < 0, -1.0, 1.0)
        factor[..., 0:1, :] = sign
        if n_frequencies % 2 == 0:
            factor[..., n_frequencies // 2 : n_frequencies // 2 + 1, :] = sign
        rotated: BackendArray = coefficients * factor
        return rotated

    def _valid_surrogate_frequencies(
        self,
        kind: str,
        method: str,
        estimate: NDArray[np.floating],
        half_bandwidth: float | None,
        method_kwargs: dict[str, Any],
    ) -> NDArray[np.bool_]:
        """Where ``kind``'s surrogates are valid, broadcastable to ``estimate``.

        ``"phase_rotation"``: rotated coefficients within ``half_bandwidth``
        of 0 Hz or Nyquist are not valid surrogates (the coefficients there
        are not circular), so the frequencies there are not tested, and a
        measure that sums over a band (``phase_slope_index``) is tested only
        if its band avoids them. ``"conjugation"`` leaves unchanged, and so
        does not test, every frequency of :attr:`real_valued_frequencies`
        (each coefficient's imaginary part exactly 0, as at Nyquist of
        one-sided input of an even FFT length), and 0 Hz and Nyquist of an
        even two-sided FFT where their coefficients are real to rounding (as
        for a real signal); a band-summed measure, or one whose frequency axis
        cannot be located, is tested whole. ``"trial_shuffle"`` tests every
        frequency.

        Raises
        ------
        ValueError
            For ``"phase_rotation"``: if no frequency is valid, the band of a
            band-summed measure includes an invalid one, or the measure does
            not return one value per frequency.
        """
        everywhere = np.ones((), dtype=bool)
        if kind == "trial_shuffle":
            return everywhere
        frequencies = self.frequencies
        grid = frequencies.astype(np.float64)
        n_fft_samples = len(self._frequencies)
        # A registered measure without a frequency axis sums over its
        # frequency_range. Otherwise the frequency axis follows the kept
        # observation axes; an unregistered measure is taken to have it there
        # if its length is right.
        dims = self._result_dims(method, estimate.ndim)
        band_summed = dims is not None and "frequency" not in dims
        frequency_axis = (
            dims.index("frequency")
            if dims is not None and not band_summed
            else len(self._kept_observation_axes)
        )
        per_frequency = (
            not band_summed
            and estimate.ndim >= frequency_axis + 3
            and estimate.shape[frequency_axis] == len(frequencies)
        )
        trailing = [1] * (estimate.ndim - frequency_axis - 1)
        if kind == "conjugation":
            if not per_frequency:
                return everywhere
            # Exactly real coefficients also mark Nyquist of one-sided input,
            # which does not record whether its FFT length was even. 0 Hz and
            # Nyquist of an even two-sided FFT are real for a real signal but
            # may carry rounding from an FFT computed elsewhere, so they are
            # left out when real to rounding; for a complex signal they are
            # not real, and conjugation changes them like any other bin.
            untouched = self._real_valued_frequency_mask.copy()
            edges = grid == 0
            if not self._is_one_sided and n_fft_samples % 2 == 0:
                edges[-1] = True
            if edges.any():
                untouched[edges] |= self._real_to_rounding(np.flatnonzero(edges))
            conjugated: NDArray[np.bool_] = ~untouched.reshape(-1, *trailing)
            return conjugated
        assert half_bandwidth is not None
        # Nyquist is the last non-negative bin for an even FFT length and half a
        # bin above it for an odd one.
        nyquist = float(frequencies[-1])
        if n_fft_samples % 2 and n_fft_samples > 1:
            nyquist += float(frequencies[1] - frequencies[0]) / 2
        # The bounds are inclusive, by the package's rule for a bin lying on a
        # bound (that of frequency_range), so a band starting at half_bandwidth
        # is accepted below whenever its bins are.
        lower, upper = half_bandwidth, nyquist - half_bandwidth
        on_lower, on_upper = _bins_on_bounds(frequencies, (lower, upper))
        valid = ((grid > lower) | on_lower) & ((grid < upper) | on_upper)
        if not valid.any():
            msg = (
                f"surrogate_test(kind='phase_rotation') tests no frequency: every "
                f"frequency lies within half_bandwidth={half_bandwidth!r} of 0 or "
                f"of Nyquist ({nyquist!r}). Check half_bandwidth is in the units "
                "of frequencies (half the transform's frequency resolution)."
            )
            raise ValueError(msg)
        if band_summed:
            band = _select_measure_frequencies(
                frequencies,
                method_kwargs.get("frequency_range"),
                method_kwargs.get("frequencies_of_interest"),
            ).mask
            if not valid[band].all():
                msg = (
                    f"surrogate_test(kind='phase_rotation') cannot test {method!r} over "
                    f"a band within half_bandwidth={half_bandwidth!r} of 0 Hz or "
                    f"Nyquist ({nyquist!r}), where rotated coefficients are not valid "
                    "surrogates. Pass frequency_range=(low, high) with "
                    f"low >= {lower!r} and high <= {upper!r}."
                )
                raise ValueError(msg)
            return everywhere
        if not per_frequency:
            msg = (
                "surrogate_test(kind='phase_rotation') leaves untested the frequencies "
                "within half_bandwidth of 0 Hz and Nyquist, so the measure must "
                f"return one value per frequency on axis {frequency_axis}; {method!r} "
                f"returned shape {estimate.shape}. Use kind='trial_shuffle' or "
                "kind='conjugation'."
            )
            raise ValueError(msg)
        valid_by_frequency: NDArray[np.bool_] = valid.reshape(-1, *trailing)
        return valid_by_frequency

    def _real_to_rounding(self, bins: NDArray[np.intp]) -> NDArray[np.bool_]:
        """Whether every finite coefficient at each of ``bins`` is real to rounding.

        Rounding is the square root of the coefficients' machine epsilon
        times their largest finite magnitude anywhere, so a bin whose own
        coefficients are small (0 Hz after detrending) is judged on the
        scale of the FFT that computed it. Coefficients computed in a lower
        precision and then upcast carry that precision's rounding, which can
        exceed this tolerance; their edge bins are then tested.

        Parameters
        ----------
        bins : NDArray[intp], shape (n_bins,)
            Indices into the frequency axis of the coefficients.

        Returns
        -------
        NDArray[bool], shape (n_bins,)
            Host array.
        """
        coefficients = self._fourier_coefficients
        magnitude = xp.abs(coefficients)
        scale = xp.max(xp.where(xp.isfinite(magnitude), magnitude, 0.0))
        tolerance = np.sqrt(np.finfo(coefficients.real.dtype).eps) * scale
        imaginary = xp.abs(coefficients[..., xp.asarray(bins), :].imag)
        real: NDArray[np.bool_] = to_numpy(
            xp.all((imaginary <= tolerance) | xp.isnan(imaginary), axis=(0, 1, 2, 4))
        )
        return real

    @_asnumpy
    def power(self, *, one_sided: bool | None = None) -> NDArray[np.floating]:
        """Return the power spectral density of the signal (one-sided by default).

        Only the non-negative frequencies are returned. Each interior
        positive-frequency bin holds its own power plus that of the matching
        negative frequency, ``P(f) + P(-f)``, so integrating the returned
        spectrum over frequency recovers the full signal power. For a real
        signal the two are equal, so the bin is doubled; for the coefficients
        of a complex signal (which the transforms refuse, but which can be
        passed to ``Connectivity`` directly) the sum keeps a tone at ``-f``,
        though not which side it was on. The DC bin, and the Nyquist bin for an
        even FFT length, have no partner and are returned as they are.

        For one-sided input (``is_one_sided=True``) the coefficients are
        returned as provided: a one-sided transform such as
        :class:`~spectral_connectivity.transforms.MorletWavelet` already
        scales its coefficients to the one-sided density, and externally
        supplied one-sided coefficients keep whatever scale they were given.

        Parameters
        ----------
        one_sided : bool, optional
            ``True`` returns the one-sided density described above, which is
            also what an omitted ``one_sided`` gives. Called without it on
            two-sided coefficients, ``power`` warns that 2.x returned the
            two-sided density
            (:class:`~spectral_connectivity.SidednessWarning`, once per
            session, until 3.2); pass ``True`` to confirm and silence it.
            ``False`` returns the two-sided density instead, still at the
            non-negative frequencies only: no bin is doubled, as for a
            spectrum laid out over negative and positive frequencies. For a
            real signal the negative frequencies mirror the interior positive
            ones, so integrating this density over the returned frequencies
            gives about half the signal power. It is the density that
            formulas written for a two-sided spectrum take, such as
            :func:`~spectral_connectivity.statistics.coherence_rate_adjustment`
            with ``is_one_sided=False``. Not available for one-sided input,
            whose two-sided scale is not known.

            .. versionchanged:: 3.0
               The default result is one-sided. ``one_sided=False`` returns
               what ``power()`` returned in 2.x.

        Returns
        -------
        NDArray[floating]
            Power spectral density for non-negative frequencies, shape
            ``(..., n_nonnegative_frequencies, n_signals)``.

        Raises
        ------
        ValueError
            If ``one_sided=False`` and the coefficients are one-sided.

        Notes
        -----
        **Range**: [0, ∞). Power spectral density is always non-negative
        with no finite upper bound.

        Examples
        --------
        >>> import numpy as np
        >>> from spectral_connectivity import Connectivity, Multitaper
        >>> rng = np.random.default_rng(0)
        >>> leader = rng.standard_normal((1003, 20))
        >>> # Signal 1 is signal 0 delayed by 3 samples (6 ms at 500 Hz), plus noise.
        >>> signals = np.stack([leader[3:], leader[:-3]], axis=-1)  # (time, trials, signals)
        >>> signals += 0.5 * rng.standard_normal(signals.shape)
        >>> multitaper = Multitaper(signals, sampling_frequency=500)
        >>> connectivity = Connectivity.from_transform(multitaper)
        >>> power = connectivity.power(one_sided=True)
        >>> power.shape  # (n_time_windows, n_frequencies, n_signals)
        (1, 501, 2)
        >>> connectivity.frequencies[[0, -1]]  # 0 Hz up to Nyquist in 0.5 Hz steps
        array([  0., 250.])
        >>> two_sided = connectivity.power(one_sided=False)
        >>> bool(np.allclose(power[0, 1:-1], 2 * two_sided[0, 1:-1]))  # interior bins
        True
        >>> bool(np.allclose(power[0, [0, -1]], two_sided[0, [0, -1]]))  # 0 Hz, Nyquist
        True
        """
        if one_sided is None:
            if not self._is_one_sided and _emit_migration_notices.get():
                # Remove in 3.2, with DirectedOrientationWarning (issue #98).
                warn_migration(
                    f"{_ORIENTATION_WARNING_PREFIX}Connectivity.power returns the "
                    "one-sided power spectral density: the bins between 0 Hz and "
                    "Nyquist are doubled, so integrating over the non-negative "
                    "frequencies gives the signal's power. 2.x returned the "
                    "two-sided density, half of that. Pass one_sided=True to "
                    "confirm and silence this warning, or one_sided=False for the "
                    f"2.x values. See the migration guide: {_MIGRATION_GUIDE_URL}. "
                    + _silence_hint(SidednessWarning),
                    SidednessWarning,
                )
            one_sided = True
        if _validated_flag("one_sided", one_sided):
            return self._one_sided_density(self._power, frequency_axis=-2)
        if self._is_one_sided:
            msg = (
                "power(one_sided=False) needs two-sided Fourier coefficients. "
                "These are one-sided coefficients (is_one_sided=True, or a "
                "one-sided transform such as MorletWavelet), which carry a "
                "one-sided scale of their own, so no two-sided density follows "
                "from them. Use power(), or build the Connectivity from "
                "two-sided coefficients."
            )
            raise ValueError(msg)
        n_nonnegative = self._nonnegative_frequency_count(self._power.shape[-2])
        two_sided: BackendArray = self._power[..., :n_nonnegative, :].copy()
        return two_sided

    @_asnumpy
    def cross_spectral_density(self) -> NDArray[np.complexfloating]:
        """Return the one-sided cross-spectral density matrix.

        The diagonal contains the one-sided power spectral densities returned
        by :meth:`power`; off-diagonal entries retain both the amplitude and
        relative-phase information between signal pairs.  Each interior
        positive-frequency bin ``S(f)`` adds ``conj(S(-f))``, so that the
        one-sided result has the same total power as the two-sided spectrum;
        for a real signal ``S(-f) = conj(S(f))`` and the bin is doubled.  DC
        and, for an even FFT length, Nyquist have no partner and are kept as
        they are.

        Returns
        -------
        cross_spectral_density : array
            Shape ``(..., n_nonnegative_frequencies, n_signals, n_signals)``.

        Notes
        -----
        The matrix is Hermitian at every time-frequency bin and has physical
        units of signal squared per Hz when the input signal has physical
        units.  Unlike connectivity measures normalized to ``[0, 1]``, its
        magnitude has no finite upper bound.

        Examples
        --------
        >>> import numpy as np
        >>> from spectral_connectivity import Connectivity, Multitaper
        >>> rng = np.random.default_rng(0)
        >>> leader = rng.standard_normal((1003, 20))
        >>> # Signal 1 is signal 0 delayed by 3 samples (6 ms at 500 Hz), plus noise.
        >>> signals = np.stack([leader[3:], leader[:-3]], axis=-1)  # (time, trials, signals)
        >>> signals += 0.5 * rng.standard_normal(signals.shape)
        >>> multitaper = Multitaper(signals, sampling_frequency=500)
        >>> connectivity = Connectivity.from_transform(multitaper)
        >>> csd = connectivity.cross_spectral_density()
        >>> csd.shape  # (n_time_windows, n_frequencies, n_signals, n_signals)
        (1, 501, 2, 2)
        >>> # The diagonal is the power spectrum; the matrix is Hermitian.
        >>> bool(np.allclose(csd[..., 0, 0].real, connectivity.power(one_sided=True)[..., 0]))
        True
        >>> bool(np.allclose(csd[..., 0, 1], np.conj(csd[..., 1, 0])))
        True
        """
        return self._one_sided_density(
            self._cached_reduced_cross_spectral_matrix, frequency_axis=-3
        )

    @_asnumpy
    def coherency(self) -> NDArray[np.complexfloating]:
        """Return the complex-valued linear association between time series.

        Computed in the frequency domain.

        Returns
        -------
        complex_coherency : array
            Shape ``(..., n_nonnegative_frequencies, n_signals, n_signals)``.
            Complex coherency between all signal pairs; Hermitian in the signal
            pair, with its angle given by :meth:`coherence_phase`. The diagonal
            is NaN: the package masks self-coherency, which is 1 for a signal
            with nonzero power.

        Notes
        -----
        **Range**: Magnitude :math:`|C_{xy}(f)|` is in [0, 1]; phase is in
        [-π, π].
        Values lie in the unit disk of the complex plane.

        **Phase convention**: ``[..., i, j]`` is ``S_ij / sqrt(S_ii S_jj)``
        with ``S_ij = E[X_i conj(X_j)]``, so a positive angle means signal
        ``i`` leads signal ``j``. :meth:`canonical_coherency` uses the
        conjugate convention (``magnitude * exp(-1j * phi)``): with
        single-channel groups its score is ``conj`` of this coherency.

        Examples
        --------
        >>> import numpy as np
        >>> from spectral_connectivity import Connectivity, Multitaper
        >>> rng = np.random.default_rng(0)
        >>> leader = rng.standard_normal((1003, 20))
        >>> # Signal 1 is signal 0 delayed by 3 samples (6 ms at 500 Hz), plus noise.
        >>> signals = np.stack([leader[3:], leader[:-3]], axis=-1)  # (time, trials, signals)
        >>> signals += 0.5 * rng.standard_normal(signals.shape)
        >>> multitaper = Multitaper(signals, sampling_frequency=500)
        >>> connectivity = Connectivity.from_transform(multitaper)
        >>> coherency = connectivity.coherency()
        >>> coherency.shape  # (n_time_windows, n_frequencies, n_signals, n_signals)
        (1, 501, 2, 2)
        >>> # At 10 Hz (bin 20) the magnitude is the coupling strength and a positive
        >>> # angle at [..., 0, 1] means signal 0 leads signal 1.
        >>> round(float(abs(coherency[0, 20, 0, 1])), 1)
        0.8
        >>> bool(np.angle(coherency[0, 20, 0, 1]) > 0)
        True
        """
        return self._coherency()

    def _coherency(self) -> NDArray[np.complexfloating]:
        """Device-native complex coherency (see the public ``coherency``).

        Kept on the active array namespace (``xp``) so internal consumers --
        ``magnitude_squared_coherence``/``coherence_phase``, ``group_delay``, ``delay``,
        ``phase_slope_index`` -- operate on device arrays without a premature
        host transfer; the public ``coherency`` converts the result to NumPy.
        """
        self._warn_single_observation_degenerate("coherency")
        complex_coherency = _divide_masking_zero_denominator(
            self._nonnegative_cross_spectral_matrix(),
            self._nonnegative_pairwise_power_scale(),
            "Some signals have (near-)zero power, so coherency is undefined "
            "for those pairs and is returned as NaN. This usually indicates "
            "a flat/dead channel or all-zero input.",
        )
        n_signals = self._fourier_coefficients.shape[-1]
        diagonal_ind = xp.arange(0, n_signals)
        complex_coherency[..., diagonal_ind, diagonal_ind] = xp.nan
        return complex_coherency

    @_asnumpy
    def coherence_phase(self) -> NDArray[np.floating]:
        """Return the phase angle of the complex coherency.

        Returns
        -------
        phase : array
            Shape ``(..., n_nonnegative_frequencies, n_signals, n_signals)``.
            Phase angles in radians. Positive ``[..., i, j]`` means signal ``i``
            leads signal ``j``; the result is antisymmetric in the signal pair.
            The diagonal is NaN, the angle of coherency's masked diagonal.

        Notes
        -----
        **Range**: [-π, π]. Phase angles in radians for complex coherency.
        A pure delay ``tau`` (seconds) gives a phase of ``2 * pi * f * tau``
        (wrapped into ``[-π, π]``). :meth:`canonical_coherency` reports its
        phase in the conjugate convention (``magnitude * exp(-1j * phi)``), so
        for single-channel groups its angle is the negative of this one.

        Examples
        --------
        >>> import numpy as np
        >>> from spectral_connectivity import Connectivity, Multitaper
        >>> rng = np.random.default_rng(0)
        >>> leader = rng.standard_normal((1003, 20))
        >>> # Signal 1 is signal 0 delayed by 3 samples (6 ms at 500 Hz), plus noise.
        >>> signals = np.stack([leader[3:], leader[:-3]], axis=-1)  # (time, trials, signals)
        >>> signals += 0.5 * rng.standard_normal(signals.shape)
        >>> multitaper = Multitaper(signals, sampling_frequency=500)
        >>> connectivity = Connectivity.from_transform(multitaper)
        >>> phase = connectivity.coherence_phase()
        >>> phase.shape  # (n_time_windows, n_frequencies, n_signals, n_signals)
        (1, 501, 2, 2)
        >>> # Signal 0 leads, so [..., 0, 1] is positive: about 2 * pi * 10 Hz * 6 ms at 10 Hz.
        >>> round(float(phase[0, 20, 0, 1]), 1)  # bin 20 is 10 Hz
        0.4
        >>> round(float(phase[0, 20, 1, 0]), 1)
        -0.4
        """
        phase: NDArray[np.floating] = xp.angle(self._coherency())
        return phase

    @_asnumpy
    def magnitude_squared_coherence(self) -> NDArray[np.floating]:
        """Return the magnitude squared of the complex coherency.

        ``|S_ij|^2 / (S_ii * S_jj)``, the squared modulus of :meth:`coherency`,
        with ``S_ij = E[X_i conj(X_j)]``. This is the quantity
        :func:`scipy.signal.coherence` calls coherence.

        .. versionadded:: 3.0
           The canonical name of the calculation 2.x called
           ``coherence_magnitude``, which is deprecated.

        Returns
        -------
        magnitude_squared_coherence : array
            Shape ``(..., n_nonnegative_frequencies, n_signals, n_signals)``.
            Magnitude-squared coherence values (symmetric in the signal pair).
            The diagonal is NaN: the package masks self-coherence, which is 1
            for a signal with nonzero power.

        Notes
        -----
        **Range**: [0, 1]; values are clipped to it, removing floating-point
        excursions beyond the bounds.

        References
        ----------
        .. [1] Hansson-Sandsten M (2011) Cross-spectrum and coherence function
               estimation using time-delayed Thomson multitapers. In: 2011 IEEE
               International Conference on Acoustics, Speech and Signal
               Processing (ICASSP), pp 4240-4243.

        Examples
        --------
        >>> import numpy as np
        >>> from spectral_connectivity import Connectivity, Multitaper
        >>> rng = np.random.default_rng(0)
        >>> leader = rng.standard_normal((1003, 20))
        >>> # Signal 1 is signal 0 delayed by 3 samples (6 ms at 500 Hz), plus noise.
        >>> signals = np.stack([leader[3:], leader[:-3]], axis=-1)  # (time, trials, signals)
        >>> signals += 0.5 * rng.standard_normal(signals.shape)
        >>> multitaper = Multitaper(signals, sampling_frequency=500)
        >>> connectivity = Connectivity.from_transform(multitaper)
        >>> coherence = connectivity.magnitude_squared_coherence()
        >>> coherence.shape  # (n_time_windows, n_frequencies, n_signals, n_signals)
        (1, 501, 2, 2)
        >>> round(float(coherence[0, 20, 0, 1]), 1)  # strong coupling at 10 Hz (bin 20)
        0.6
        >>> # The squared magnitude of coherency, not its magnitude.
        >>> coherency = connectivity.coherency()
        >>> bool(np.allclose(coherence[0, 20, 0, 1], abs(coherency[0, 20, 0, 1]) ** 2))
        True
        """
        magnitude = _squared_magnitude(self._coherency())
        clipped: NDArray[np.floating] = xp.clip(magnitude, 0, 1)
        return clipped

    def coherence_magnitude(self) -> NDArray[np.floating]:
        """Return magnitude-squared coherence (deprecated name).

        .. deprecated:: 3.0
           Use :meth:`magnitude_squared_coherence`, which returns the same
           values. ``coherence_magnitude`` is removed in 4.0.

        Returns
        -------
        magnitude_squared_coherence : array
            Shape ``(..., n_nonnegative_frequencies, n_signals, n_signals)``.
        """
        _resolve_deprecated_measure_name("coherence_magnitude", "Connectivity.{}()")
        return self.magnitude_squared_coherence()

    @_asnumpy
    def imaginary_coherence(self) -> NDArray[np.floating]:
        """Return the normalized imaginary component of the cross-spectrum.

        Projects the cross-spectrum onto the imaginary axis to mitigate the
        effect of volume-conducted dependencies. Assumes volume-conducted
        sources arrive at sensors at the same time, resulting in
        a cross-spectrum with phase angle of 0 (perfectly in-phase) or π
        (anti-phase) if the sensors are on opposite sides of a dipole
        source. With the imaginary coherence, in-phase and anti-phase
        associations are set to zero.

        Returns
        -------
        imaginary_coherence_magnitude : array
            Shape ``(..., n_nonnegative_frequencies, n_signals, n_signals)``.
            Imaginary coherence magnitudes (symmetric in the signal pair; see
            :meth:`imaginary_coherency` for the signed, lead/lag-aware version).
            The diagonal is 0 (to within rounding), the imaginary part of a
            signal's coherency with itself, except for a zero-power signal,
            whose entries are all NaN.

        Notes
        -----
        **Range**: [0, 1]. ``|Im(S_ij)| / sqrt(S_ii * S_jj)``, the magnitude of
        :meth:`imaginary_coherency`, which ranges in [-1, 1]. The two differ on
        the diagonal: 0 here, NaN there, where the package masks it like
        :meth:`coherency`.

        References
        ----------
        .. [1] Nolte, G., Bai, O., Wheaton, L., Mari, Z., Vorbach, S., and
               Hallett, M. (2004). Identifying true brain interaction from
               EEG data using the imaginary part of coherency. Clinical
               Neurophysiology 115, 2292-2307.

        Examples
        --------
        >>> import numpy as np
        >>> from spectral_connectivity import Connectivity, Multitaper
        >>> rng = np.random.default_rng(0)
        >>> leader = rng.standard_normal((1003, 20))
        >>> # Signal 1 is signal 0 delayed by 3 samples (6 ms at 500 Hz), plus noise.
        >>> signals = np.stack([leader[3:], leader[:-3]], axis=-1)  # (time, trials, signals)
        >>> signals += 0.5 * rng.standard_normal(signals.shape)
        >>> multitaper = Multitaper(signals, sampling_frequency=500)
        >>> connectivity = Connectivity.from_transform(multitaper)
        >>> imaginary_coherence = connectivity.imaginary_coherence()
        >>> imaginary_coherence.shape  # (n_time_windows, n_frequencies, n_signals, n_signals)
        (1, 501, 2, 2)
        >>> # The 6 ms lag is not zero-phase, so the imaginary part survives (10 Hz = bin 20).
        >>> bool(imaginary_coherence[0, 20, 0, 1] > 0.2)
        True
        """
        self._warn_single_observation_degenerate(
            "imaginary_coherence",
            "coherency has magnitude 1 for every pair and each value is the "
            "absolute sine of one phase difference",
        )
        imaginary_coh = xp.abs(
            _divide_masking_zero_denominator(
                self._nonnegative_cross_spectral_matrix().imag,
                self._nonnegative_pairwise_power_scale(),
                "Some signals have (near-)zero power, so imaginary coherence is "
                "undefined for those pairs and is returned as NaN. This usually "
                "indicates a flat/dead channel or all-zero input.",
            )
        )
        # abs()/clip() leave the NaN-masked zero-power entries as NaN.
        return xp.clip(imaginary_coh, 0, 1)

    @_asnumpy
    def imaginary_coherency(self) -> NDArray[np.floating]:
        """Return the signed imaginary component of coherency.

        This is the signed counterpart of :meth:`imaginary_coherence`, which
        returns its magnitude.  The sign is antisymmetric across a signal pair
        and preserves the pair's phase-lead/phase-lag orientation.

        Returns
        -------
        imaginary_coherency : array
            Shape ``(..., n_nonnegative_frequencies, n_signals, n_signals)``.
            Positive ``[..., i, j]`` means signal ``i`` leads signal ``j``.

        Notes
        -----
        **Range**: ``[-1, 1]``. ``Im(S_ij) / sqrt(S_ii * S_jj)``. Pairs
        involving a zero-power signal are 0, not NaN: :meth:`coherency` masks
        them with ``nan + 0j``, whose imaginary part is 0, so use
        :meth:`power` to find such signals. A pair is NaN where a signal's
        power is not finite: NaN, as in a bin that keeps no observation (see
        :attr:`n_valid_observations`), or infinite. The diagonal is NaN, a mask
        matching :meth:`coherency`: the imaginary part of a signal's coherency
        with itself is 0, which :meth:`imaginary_coherence` returns there.

        Examples
        --------
        >>> import numpy as np
        >>> from spectral_connectivity import Connectivity, Multitaper
        >>> rng = np.random.default_rng(0)
        >>> leader = rng.standard_normal((1003, 20))
        >>> # Signal 1 is signal 0 delayed by 3 samples (6 ms at 500 Hz), plus noise.
        >>> signals = np.stack([leader[3:], leader[:-3]], axis=-1)  # (time, trials, signals)
        >>> signals += 0.5 * rng.standard_normal(signals.shape)
        >>> multitaper = Multitaper(signals, sampling_frequency=500)
        >>> connectivity = Connectivity.from_transform(multitaper)
        >>> imaginary_coherency = connectivity.imaginary_coherency()
        >>> imaginary_coherency.shape  # (n_time_windows, n_frequencies, n_signals, n_signals)
        (1, 501, 2, 2)
        >>> # Signal 0 leads, so [..., 0, 1] is positive and [..., 1, 0] negative (10 Hz).
        >>> bool(imaginary_coherency[0, 20, 0, 1] > 0 > imaginary_coherency[0, 20, 1, 0])
        True
        """
        imaginary = self._coherency().imag
        # Coherency is nan + 0j both where a signal has zero power (0 here, as
        # documented) and where its power is not finite: NaN, such as a bin
        # that keeps no observation, or infinite, from overflow. Those stay NaN.
        n_nonnegative = self._nonnegative_frequency_count(self._power.shape[-2])
        bad_power = ~xp.isfinite(self._power[..., :n_nonnegative, :])
        imaginary[bad_power[..., :, xp.newaxis] | bad_power[..., xp.newaxis, :]] = xp.nan
        diagonal = xp.arange(self.n_signals)
        imaginary[..., diagonal, diagonal] = xp.nan
        return imaginary

    @_asnumpy
    def partial_coherence(
        self,
        *,
        regularization: float = TIKHONOV_REGULARIZATION_FACTOR,
        shrinkage: float | Literal["auto"] | None = None,
    ) -> NDArray[np.floating]:
        """Return magnitude-squared coherence conditional on all other signals.

        Partial coherence is computed by normalizing the off-diagonal elements
        of the inverse cross-spectral density (the spectral precision matrix).
        It measures the remaining linear association between each pair after
        conditioning on every other observed signal.

        Parameters
        ----------
        regularization : float, default=1e-12
            Non-negative relative diagonal loading applied independently to
            each time-frequency cross-spectral matrix before inversion: each
            signal's power ``S_ii`` is increased by ``regularization * S_ii``,
            so the result does not depend on the signals' scales; the matrix
            inverted is the coherency matrix, loaded by ``regularization``.
            Increase this value for statistically rank-deficient or
            ill-conditioned spectra.
        shrinkage : {None, "auto"} or float, default=None
            Shrink each time-frequency cross-spectral matrix toward its
            diagonal before inversion, ``(1 - lambda) * S + lambda * diag(S)``:
            every coherency is multiplied by ``1 - lambda`` and the powers are
            kept, so the result does not depend on the signals' scales.
            ``None`` does not shrink. A number in ``[0, 1]`` is ``lambda`` for
            every bin: 0 does not shrink, and 1 makes every partial coherence 0.
            ``"auto"`` estimates ``lambda`` for each time window and frequency
            from the averaged observations, and
            :meth:`partial_coherence_shrinkage` returns it. Unlike
            ``regularization``, a tiny loading for numerical stability,
            shrinkage is a statistical choice that changes the estimate; see
            Notes for when it helps and what to report.

        Returns
        -------
        partial_coherence : array
            Shape ``(..., n_nonnegative_frequencies, n_signals, n_signals)``.

        Raises
        ------
        ValueError
            If ``regularization`` is negative or not finite, if ``shrinkage``
            is not ``None``, ``"auto"``, or a number in ``[0, 1]``, or, for
            ``shrinkage="auto"``, if there is a single observation or the
            observation weights are not uniform.

        Notes
        -----
        **Range**: ``[0, 1]``. The diagonal is NaN: the package masks a
        signal's partial coherence with itself, which is 1.
        This undirected measure is distinct from partial directed coherence.
        Regularization stabilizes inversion but also changes the estimand, so
        analyses should report a non-default value.

        The averaged trials x tapers are the observations. With fewer
        observations than signals the cross-spectral matrix is rank-deficient
        and its (regularized) inverse is dominated by the null space; with
        exactly one null direction every partial coherence is forced to 1 for
        any data. This case warns, with shrinkage too: shrinkage makes the
        matrix invertible, but the estimate then rests on the intensity. A
        small one leaves it dominated by the null space; as the intensity
        approaches 1, partial coherence approaches ``(1 - lambda)**2`` times
        the coherence, which no longer conditions on the other signals.

        **Shrinkage intensity.** With ``n`` observations ``x_k`` (one trial x
        taper eigencoefficient vector each) of a time window and frequency,
        ``S = (1/n) sum_k x_k x_k^H`` and the coherency is
        ``r_ij = S_ij / sqrt(S_ii S_jj)``. ``"auto"`` takes the intensity that
        minimizes the expected squared error of the shrunk coherencies,
        ``sum_{i != j} E|(1 - lambda) r_ij - rho_ij|^2`` for true coherencies
        ``rho``, with ``E r = rho`` to first order. That is the analytic
        intensity of Ledoit & Wolf [1]_ [2]_ for the target of Schäfer &
        Strimmer [3]_ that keeps the powers and shrinks the correlations
        toward 0 (their "target D", Eq. 10), for complex data,

            lambda = sum_{i != j} var(r_ij) / sum_{i != j} |r_ij|^2,

        clipped to ``[0, 1]``. ``var(r_ij)`` estimates ``E|r_ij - rho_ij|^2``
        by the delta method,

            var(r_ij) = sum_k |w_kij - (r_ij / 2) (u_ki + u_kj)|^2 / (n (n - 1)),

        with ``w_kij = x_ki conj(x_kj) / sqrt(S_ii S_jj)`` and
        ``u_ki = |x_ki|^2 / S_ii``. The ``u`` terms account for the estimated
        powers that normalize ``r``. Schäfer & Strimmer leave them out, which
        for complex Gaussian data estimates ``1/n`` whatever the coherence,
        where the variance is ``(1 - |rho|^2) (1 - |rho|^2 / 2) / n``: it
        overstates the intensity where coherence is high. ``lambda``
        decreases as ``1/n``, so with many observations the result approaches
        the unshrunk one. Partial coherence is then read from the shrunk
        matrix, as Schäfer & Strimmer read partial correlations (their
        Sections 2.4 and 3). ``"auto"`` took 1.4-2.4 times as long as no
        shrinkage and allocated 1.3-1.6 times its peak memory, the input
        aside (16 to 64 signals, 100 to 1400 observations, 251 to 1001
        frequencies, one or 100 time windows, on the CPU);
        :meth:`partial_coherence_shrinkage` then reuses the intensity.

        The intensity is a first-order estimate. At 5 observations it
        averaged 0.77-0.98 of the Monte Carlo optimum (4 to 32 signals, lowest
        with the most). With 10 to 40 observations it was within about 10% of
        the optimum in the VARs simulated below; where the optimum is near 0
        (strongly coherent signals) it can be several times the optimum while
        staying small. For independent signals, whose optimum is 1, it
        averaged 0.74-0.96.

        In simulations of a chain and a dense VAR with 4, 8 and 16 signals
        (complex Gaussian observations with the model's exact spectrum, 500
        replicates), ``"auto"`` changed the mean squared error against the
        model's partial coherence as follows, by observations per signal:

        - 0.5 to 2.5: lowered it in every case, to 0.01-0.80 of the unshrunk
          error (to 0.01-0.21 at 1 to 1.25; with fewer observations than
          signals, for 8 and 16 signals, also below the error of estimating 0
          for every pair);
        - 5: lowered it in four of the six cases (to 0.54-0.86) and raised it
          in two (to 1.12 and 1.63 times, by at most 4.9e-3);
        - 10 to 100: lowered it for the chain with 4 and 8 signals (to
          0.73-0.97) and otherwise left it unchanged or raised it, by up to
          2.2 times (by at most 3.2e-3).

        The intensity minimizes the error of the coherencies, not of their
        inverse, which is why it can hurt with many observations. Use
        ``"auto"`` with at most about 2.5 observations per signal.

        **What to report.** Shrinkage changes the estimand, toward 0. Report
        the setting, for ``"auto"`` the median and range of the intensities,
        and the numbers of trials, tapers and signals. The wrappers record the
        setting (``arg_shrinkage``) but not the intensities; compute them with
        ``Connectivity.from_transform(transform).partial_coherence_shrinkage()``.
        Under ``"auto"`` the intensity falls with the number of observations,
        so conditions with different trial counts are shrunk by different
        amounts and biased differently: compare them with one fixed intensity
        or with matched counts. :meth:`jackknife` re-estimates the intensity
        in every leave-one-out replicate, so its intervals are for the shrunk
        estimand.

        See Also
        --------
        partial_coherence_shrinkage : The intensities ``shrinkage="auto"`` uses.

        References
        ----------
        .. [1] Ledoit O, Wolf M (2003) Improved estimation of the covariance
               matrix of stock returns with an application to portfolio
               selection. Journal of Empirical Finance 10(5):603-621.
               https://doi.org/10.1016/S0927-5398(03)00007-0
        .. [2] Ledoit O, Wolf M (2004) A well-conditioned estimator for
               large-dimensional covariance matrices. Journal of Multivariate
               Analysis 88(2):365-411.
               https://doi.org/10.1016/S0047-259X(03)00096-4
        .. [3] Schäfer J, Strimmer K (2005) A shrinkage approach to large-scale
               covariance matrix estimation and implications for functional
               genomics. Statistical Applications in Genetics and Molecular
               Biology 4(1):Article 32. https://doi.org/10.2202/1544-6115.1175

        Examples
        --------
        >>> import numpy as np
        >>> from spectral_connectivity import Connectivity, Multitaper
        >>> rng = np.random.default_rng(0)
        >>> leader = rng.standard_normal((1000, 20, 1))
        >>> # Signals 1 and 2 are noisy copies of signal 0 and share nothing else.
        >>> copies = leader + 0.5 * rng.standard_normal((1000, 20, 2))
        >>> signals = np.concatenate([leader, copies], axis=-1)  # (time, trials, signals)
        >>> multitaper = Multitaper(signals, sampling_frequency=500)
        >>> connectivity = Connectivity.from_transform(multitaper)
        >>> partial = connectivity.partial_coherence()
        >>> partial.shape  # (n_time_windows, n_frequencies, n_signals, n_signals)
        (1, 501, 3, 3)
        >>> # Signals 1 and 2 are coherent, but not once signal 0 is accounted for (10 Hz).
        >>> bool(connectivity.magnitude_squared_coherence()[0, 20, 1, 2] > 0.5)
        True
        >>> bool(partial[0, 20, 1, 2] < 0.05)
        True
        """
        self._validate_multiple_signals()
        regularization = _validated_regularization(regularization)
        # Before any warning: "auto" raises for a single observation.
        cross_spectral_density, intensity, shrinkage = (
            self._cross_spectral_matrix_and_intensity(shrinkage)
        )
        # With observations dropped by zero weights, the bin keeping fewest.
        n_observations, n_signals = self._fewest_valid_observations(), self.n_signals
        if shrinkage is None:
            # The inverse of a rank-deficient cross-spectral matrix is dominated
            # by its null space: with one null direction the normalized
            # off-diagonal precision has unit magnitude for every pair, and with
            # more it is still unrelated to the data.
            self._warn_single_observation_degenerate(
                "partial_coherence",
                "the cross-spectral matrix has rank one and its inverse is fixed "
                "by the null space: the estimate is 1 for every pair of two "
                "signals and otherwise unrelated to the data",
            )
            consequence = (
                "and its inverse is dominated by the null space: the estimate is "
                "forced to 1 for every pair (one missing observation) or is "
                "otherwise unrelated to the data."
            )
            few = 2 <= n_observations < n_signals
        else:
            # Shrinkage makes the matrix invertible, but the estimate then rests
            # on the intensity. ("auto" has raised for a single observation.)
            used = (
                "see Connectivity.partial_coherence_shrinkage()"
                if shrinkage == "auto"
                else f"{shrinkage:g}"
            )
            consequence = (
                "and only the shrinkage makes it invertible: the result is set by "
                f"the shrinkage intensity ({used}), not conditioned on the other "
                "signals. Near 1 it is about (1 - intensity)**2 times the "
                "coherence; near 0 the null space dominates it."
                + (
                    " With one observation it does not depend on the data."
                    if n_observations == 1
                    else ""
                )
            )
            few = 1 <= n_observations < n_signals
        if few:
            uses = self._uses_observations(
                n_observations, self.n_observations, "see n_valid_observations"
            )
            warnings.warn(
                f"partial_coherence {uses} (the averaged "
                f"trials x tapers), fewer than the {n_signals} signals, so the "
                f"cross-spectral matrix is rank-deficient {consequence} Provide "
                "more trials or tapers than signals, or analyze fewer signals.",
                UserWarning,
                stacklevel=stacklevel_outside_package(),
            )

        matrix_rms = xp.sqrt(
            xp.mean(
                xp.real(xp.conj(cross_spectral_density) * cross_spectral_density),
                axis=(-2, -1),
                keepdims=True,
            )
        )
        zero_power = matrix_rms <= xp.finfo(matrix_rms.dtype).tiny
        if bool(xp.any(zero_power)):
            warnings.warn(
                "Some time-frequency cross-spectral matrices have zero power, "
                "so partial coherence is undefined there and is returned as NaN.",
                UserWarning,
                stacklevel=stacklevel_outside_package(),
            )
        # Invert the coherency matrix D^-1/2 S D^-1/2 (D = diag(S)) loaded by
        # regularization * I: the inverse of S + regularization * D up to the
        # diagonal scaling the normalization below removes, but with the
        # signals' scales out of the solve, so that its rounding does not
        # depend on them. Shrinkage keeps the powers, so it multiplies the
        # coherencies by 1 - lambda in place. A zero-power bin is solved as
        # the identity (its result is NaN); a signal with no power keeps a
        # zero diagonal, loaded by regularization alone, which decouples it.
        power = xp.real(xp.diagonal(cross_spectral_density, axis1=-2, axis2=-1))
        inverse_root_power = 1 / xp.sqrt(xp.where(power > 0, power, 1))
        coherency_matrix = cross_spectral_density * inverse_root_power[..., :, xp.newaxis]
        coherency_matrix *= inverse_root_power[..., xp.newaxis, :]
        if intensity is not None:
            off_diagonal = 1 - xp.eye(n_signals, dtype=power.dtype)
            coherency_matrix *= 1 - intensity * off_diagonal
        coherency_matrix *= ~zero_power
        diagonal = xp.arange(n_signals)
        coherency_matrix[..., diagonal, diagonal] = (
            xp.where((power > 0) | zero_power[..., 0], 1.0, 0.0) + regularization
        )
        identity = xp.eye(n_signals, dtype=coherency_matrix.dtype)
        # Broadcast so that CuPy's batched solve accepts the right-hand side.
        precision: BackendArray = xp.linalg.solve(
            coherency_matrix, xp.broadcast_to(identity, coherency_matrix.shape)
        )
        precision_diagonal = xp.maximum(
            xp.real(xp.diagonal(precision, axis1=-2, axis2=-1)), 0.0
        )
        denominator = xp.sqrt(
            precision_diagonal[..., :, xp.newaxis] * precision_diagonal[..., xp.newaxis, :]
        )
        partial_coherency = _divide_masking_zero_denominator(
            -precision,
            denominator,
            "Some spectral-precision diagonal entries are (near-)zero, so "
            "partial coherence is undefined for those pairs and is returned as NaN.",
        )
        result = xp.clip(_squared_magnitude(partial_coherency), 0.0, 1.0)
        result = xp.where(zero_power, xp.nan, result)
        result[..., diagonal, diagonal] = xp.nan
        return result

    def _cross_spectral_matrix_and_intensity(
        self, shrinkage: Any
    ) -> tuple[BackendArray, BackendArray | float | None, float | Literal["auto"] | None]:
        """The cross-spectral matrix at the non-negative frequencies and its shrinkage.

        Validates ``shrinkage`` (0 means none), runs the checks of ``"auto"``,
        and returns ``(S, intensity, shrinkage)``: ``S``, ``(...,
        n_nonnegative_frequencies, n_signals, n_signals)``; the intensity
        ``lambda`` by which every coherency is multiplied by ``1 - lambda``
        (``None`` without shrinkage, an array ``(..., n_nonnegative_frequencies,
        1, 1)`` for ``"auto"``); and the validated ``shrinkage`` (``None``,
        ``"auto"``, or a float in ``(0, 1]``).
        """
        shrinkage = _validated_shrinkage(shrinkage)
        if shrinkage == 0:
            shrinkage = None
        # Drop negative frequencies before the per-bin inversion, not after.
        cross_spectral_density = self._nonnegative_cross_spectral_matrix()
        if shrinkage is None:
            return cross_spectral_density, None, None
        intensity = (
            self._auto_shrinkage_intensity(
                'partial_coherence(shrinkage="auto")',
                other_remedy="a fixed shrinkage between 0 and 1",
            )[..., xp.newaxis, xp.newaxis]
            if shrinkage == "auto"
            else shrinkage
        )
        return cross_spectral_density, intensity, shrinkage

    @_asnumpy
    def partial_coherence_shrinkage(self) -> NDArray[np.floating]:
        """Return the shrinkage intensity of ``partial_coherence(shrinkage="auto")``.

        Returns
        -------
        intensity : array
            Shape ``(..., n_nonnegative_frequencies)``: for each time window
            (the observation axes the expectation keeps) and frequency, the
            ``lambda`` in ``[0, 1]`` by which ``partial_coherence`` shrinks
            every coherency, ``r -> (1 - lambda) * r``. NaN where the
            cross-spectral matrix has zero power, or where zero observation
            weights leave fewer than two observations.

        Raises
        ------
        ValueError
            If there is a single observation, or the observation weights are
            not uniform.

        Notes
        -----
        Values near 0 mean the coherencies are estimated precisely enough to
        be kept; values near 1 mean they are indistinguishable from sampling
        noise. The estimator, its accuracy, and what to report are in the
        Notes of :meth:`partial_coherence`. It treats the observations as
        independent, and warns when the transform reports them correlated.

        See Also
        --------
        partial_coherence : The estimate these intensities shrink.

        Examples
        --------
        >>> import numpy as np
        >>> from spectral_connectivity import Connectivity
        >>> rng = np.random.default_rng(0)
        >>> def coefficients(n_trials):  # 3 tapers, 8 FFT bins, 6 signals
        ...     shape = (1, n_trials, 3, 8, 6)
        ...     values = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        ...     values[..., 1:] += values[..., :1]  # every signal shares signal 0
        ...     return values
        >>> few = Connectivity(coefficients(4)).partial_coherence_shrinkage()
        >>> few.shape  # (n_time_windows, n_nonnegative_frequencies)
        (1, 5)
        >>> # Ten times the trials: the intensity falls about tenfold.
        >>> many = Connectivity(coefficients(40)).partial_coherence_shrinkage()
        >>> bool(np.all(many < few / 3))
        True
        """
        self._validate_multiple_signals()
        return self._auto_shrinkage_intensity("partial_coherence_shrinkage()").copy()

    def _auto_shrinkage_intensity(
        self, caller: str, other_remedy: str | None = None
    ) -> BackendArray:
        """``lambda`` of ``partial_coherence(shrinkage="auto")`` for each bin,
        ``(..., n_nonnegative_frequencies)``, after the checks of a measure
        that counts its observations. ``caller`` names the call in their
        errors and warning; ``other_remedy`` is offered besides more trials or
        tapers."""
        if self._is_exact_spectrum:
            raise ValueError(_exact_spectrum_refusal_message(caller, "shrinkage"))
        alternative = "" if other_remedy is None else f", or {other_remedy}"
        self._validate_counted_observations(
            caller,
            why="The intensity estimates each coherency's variance across the "
            "observations, which is undefined for one.",
            remedy=f"more trials or tapers{alternative}",
            weights_reason="The intensity estimates each coherency's variance "
            "from equally weighted independent observations"
            + ("." if other_remedy is None else f"; use {other_remedy} instead."),
            assumption="The intensity estimates each coherency's sampling "
            "variance from the observations as independent samples",
        )
        return self._partial_coherence_shrinkage_intensity

    @cached_property
    def _partial_coherence_shrinkage_intensity(self) -> BackendArray:
        """``lambda`` of ``partial_coherence(shrinkage="auto")`` for each bin,
        ``(..., n_nonnegative_frequencies)``, cached: ``"auto"`` has no
        settings, so :meth:`partial_coherence_shrinkage` after
        ``partial_coherence(shrinkage="auto")`` reuses it. Read-only for its
        consumers. The estimator is in the Notes of :meth:`partial_coherence`.
        With zero observation weights each bin counts the observations it
        keeps.
        """
        cross_spectral_density = self._nonnegative_cross_spectral_matrix()
        power = xp.real(xp.diagonal(cross_spectral_density, axis1=-2, axis2=-1))
        root_power = xp.sqrt(power)
        has_power = power > 0  # False for NaN (a bin with no observation weights)
        # Count each bin's kept observations, not n_observations: where zero
        # weights drop observations (artifact_samples, say), the raw count
        # would give a wrong intensity without any error.
        counts = (
            self.n_observations
            if self._observation_weights is None
            else self._valid_observation_counts[..., : power.shape[-2]]
        )
        squared_coherency, mean_squared_influence = self._coherency_influence(
            cross_spectral_density, root_power, has_power, counts
        )
        off_diagonal = 1 - xp.eye(self.n_signals, dtype=power.dtype)
        variance_sum = xp.sum(mean_squared_influence * off_diagonal, axis=(-2, -1)) / (
            xp.maximum(counts, 2) - 1
        )
        coherence_sum = xp.sum(squared_coherency * off_diagonal, axis=(-2, -1))
        # Every coherency exactly 0: nothing to keep, and any intensity gives
        # the same partial coherence.
        intensity = xp.clip(
            _divide_where(variance_sum, coherence_sum, coherence_sum > 0, 1.0), 0.0, 1.0
        )
        # Also NaN where non-finite coefficients reach the powers.
        undefined = (
            (counts < 2) | ~xp.any(has_power, axis=-1) | ~xp.all(xp.isfinite(power), axis=-1)
        )
        return xp.where(undefined, xp.nan, intensity)

    def _coherency_influence(
        self,
        cross_spectral_density: BackendArray,
        root_power: BackendArray,
        has_power: BackendArray,
        counts: int | BackendArray,
    ) -> tuple[BackendArray, BackendArray]:
        """``(|r_ij|^2, mean_k |psi_kij|^2)`` for each bin of the coherency
        ``r`` of ``cross_spectral_density`` (non-negative frequencies), with
        ``psi_kij = w_kij - (r_ij / 2) (u_ki + u_kj)`` the delta-method
        influence of observation ``k`` (see the Notes of
        :meth:`partial_coherence`). ``root_power`` and ``has_power`` are
        ``sqrt(S_ii)`` and ``S_ii > 0``, ``(..., n_nonnegative_frequencies,
        n_signals)``, and ``counts`` the observations each bin keeps. Pairs
        with a zero-power signal are 0. Observation weights must only keep or
        drop observations, which the callers' guard ensures.
        """
        coefficients = self._nonnegative_fourier_coefficients()
        bins = root_power.shape[:-1]  # (*kept observation axes, frequency)
        n_observations, n_signals = self.n_observations, self.n_signals
        averaged_last = [*self._kept_observation_axes, 3, *self._expectation_axes]
        # One layout, (*bins, observation, signal), with each observation over
        # its signal's root power in its bin (0 for a signal without power).
        scale = xp.where(has_power, 1 / xp.where(has_power, root_power, 1), 0)
        observations = (
            xp.transpose(coefficients, [*averaged_last, 4]).reshape(
                (*bins, n_observations, n_signals)
            )
            * scale[..., xp.newaxis, :]
        )
        if self._observation_weights is not None:
            # Leave out the zero-weight observations, which may be NaN.
            kept = xp.transpose(
                self._observation_weights[..., : bins[-1], 0] > 0, averaged_last
            ).reshape((*bins, n_observations))
            observations = _weight_observations(
                observations, kept[..., xp.newaxis].astype(root_power.dtype)
            )
        per_bin = xp.maximum(xp.asarray(counts, dtype=root_power.dtype), 1)[
            ..., xp.newaxis, xp.newaxis
        ]
        relative_power = observations.real**2 + observations.imag**2  # u
        # Means over the observations of u_i u_j (a real product) and of
        # u_i w_ij = conj(sum_k conj(u_ki x_ki) x_kj) / n.
        power_products = (
            xp.matmul(xp.swapaxes(relative_power, -1, -2), relative_power) / per_bin
        )
        conjugated = relative_power * observations
        xp.conj(conjugated, out=conjugated)
        weighted_coherency = (
            xp.conj(xp.matmul(xp.swapaxes(conjugated, -1, -2), observations)) / per_bin
        )
        pair_has_power = has_power[..., :, xp.newaxis] & has_power[..., xp.newaxis, :]
        coherency = _divide_where(
            cross_spectral_density,
            root_power[..., :, xp.newaxis] * root_power[..., xp.newaxis, :],
            pair_has_power,
            0.0,
        )
        product_diagonal = xp.diagonal(power_products, axis1=-2, axis2=-1)
        squared_coherency = _squared_magnitude(coherency)
        # mean_k |w_kij - (r_ij / 2) (u_ki + u_kj)|^2, expanded.
        mean_squared_influence = (
            power_products
            - xp.real(
                xp.conj(coherency)
                * (weighted_coherency + _conjugate_transpose(weighted_coherency))
            )
            + squared_coherency
            / 4
            * (
                product_diagonal[..., :, xp.newaxis]
                + product_diagonal[..., xp.newaxis, :]
                + 2 * power_products
            )
        )
        return squared_coherency, mean_squared_influence

    def _validated_group_indices(
        self, group_labels: NDArray[Any]
    ) -> tuple[NDArray[Any], list[NDArray[np.intp]], NDArray[np.bool_]]:
        """Validate a one-label-per-signal grouping and return its geometry."""
        self._validate_multiple_signals()
        labels_array = np.asarray(group_labels)
        if labels_array.ndim != 1 or len(labels_array) != self.n_signals:
            msg = (
                f"group_labels must be one-dimensional with length "
                f"n_signals ({self.n_signals}), got shape {labels_array.shape}."
            )
            raise ValueError(msg)
        has_missing_label = False
        # Check the labels as given: np.asarray turns a NaN among strings into
        # the string "nan", which would otherwise become a group of its own.
        for group_label in np.asarray(group_labels, dtype=object):
            if group_label is None:
                has_missing_label = True
                break
            try:
                has_missing_label = not bool(group_label == group_label)
            except (TypeError, ValueError):
                # An indeterminate equality result (for example a nullable
                # scalar) cannot define stable group membership either.
                has_missing_label = True
            if has_missing_label:
                break
        if has_missing_label:
            msg = "group_labels must not contain missing values such as NaN or None."
            raise ValueError(msg)
        labels = np.unique(labels_array)
        if len(labels) < 2:
            msg = "group_labels must define at least two groups."
            raise ValueError(msg)
        indices = [np.flatnonzero(labels_array == label) for label in labels]
        membership = np.asarray(
            np.stack([labels_array == label for label in labels]), dtype=bool
        )
        return labels, indices, membership

    def canonical_coherence(
        self, group_labels: NDArray[np.integer]
    ) -> tuple[NDArray[np.floating], NDArray[np.integer]]:
        """Return the historical magnitude-squared canonical correlation.

        The canonical coherence finds two sets of complex weights such that the
        magnitude-squared coherence between the linear combination of group1
        and the linear combination of group2 is maximized:
        ``max_{a, b} |a^H S_12 b|^2 / ((a^H S_11 a) (b^H S_22 b))``, the square
        of the groups' first canonical correlation. For two single-signal
        groups it is :meth:`magnitude_squared_coherence`.

        This is a different estimator from :meth:`canonical_coherency`, which
        maximizes over real weights and one phase and reports the unsquared
        coherence; neither is the square or square root of the other.

        Parameters
        ----------
        group_labels : array-like, shape (n_signals,)
            Links each signal to a group.

        Returns
        -------
        canonical_coherence : array
            Shape ``(n_time_windows, n_nonnegative_frequencies, n_groups, n_groups)``.
            The maximal coherence for each group pair (symmetric). The diagonal
            is NaN: a group with itself is not computed.
        labels : array, shape (n_groups,)
            The sorted unique group labels that correspond to `n_groups`.

        Notes
        -----
        **Range**: [0, 1]. Maximal coherence values are bounded like
        magnitude-squared coherence.

        Trials x tapers are the observations. A group pair with more signals
        than observations has intersecting observation subspaces, which forces
        its value to 1 for any data; this case warns.

        Only a group's linearly independent signals contribute: a dead,
        duplicated, or numerically collinear channel (relative singular value
        below ``sqrt(n_signals * eps)`` after each channel is scaled to unit
        norm) is ignored, with a warning naming the group. A group with no power
        at a bin gives NaN for its pairs there, also with a warning.

        References
        ----------
        .. [1] Stephen, E.P. (2015). Characterizing dynamically evolving
               functional networks in humans with application to speech.
               Boston University.

        See Also
        --------
        canonical_coherency
            Exact complex, phase-optimised Vidaurre CaCoh with component
            filters and patterns.

        Examples
        --------
        >>> import numpy as np
        >>> from spectral_connectivity import Connectivity, Multitaper
        >>> rng = np.random.default_rng(0)
        >>> leader = rng.standard_normal((1003, 20))
        >>> # Group "a": two noisy copies of signal 0; "b": two of its 6 ms-delayed copy.
        >>> pair = np.stack([leader[3:], leader[:-3]], axis=-1)
        >>> signals = np.repeat(pair, 2, axis=-1)  # (time, trials, 4 signals)
        >>> signals += 0.5 * rng.standard_normal(signals.shape)
        >>> multitaper = Multitaper(signals, sampling_frequency=500)
        >>> connectivity = Connectivity.from_transform(multitaper)
        >>> coherence, labels = connectivity.canonical_coherence(["a", "a", "b", "b"])
        >>> coherence.shape  # (n_time_windows, n_frequencies, n_groups, n_groups)
        (1, 501, 2, 2)
        >>> labels
        array(['a', 'b'], dtype='<U1')
        >>> bool(coherence[0, 20, 0, 1] > 0.5)  # strong a-b coupling at 10 Hz (bin 20)
        True
        >>> # For single-signal groups it is the pairwise magnitude-squared coherence.
        >>> two = Connectivity.from_transform(
        ...     Multitaper(signals[..., [0, 2]], sampling_frequency=500)
        ... )
        >>> single, _ = two.canonical_coherence(["a", "b"])
        >>> pairwise = two.magnitude_squared_coherence()
        >>> bool(np.allclose(single[..., 0, 1], pairwise[..., 0, 1]))
        True
        """
        labels, group_indices, _ = self._validated_group_indices(group_labels)
        n_frequencies = self._fourier_coefficients.shape[-2]
        non_negative_frequencies = xp.arange(
            0, self._nonnegative_frequency_count(n_frequencies)
        )
        observation_weights = None
        # The estimate treats each window's trials x tapers as observations,
        # less those zero weights drop: (n_time_windows, n_frequencies) counts.
        n_observations = (
            self._fourier_coefficients.shape[1] * self._fourier_coefficients.shape[2]
        )
        valid_counts = None
        fewest = n_observations
        if self._observation_weights is not None:
            observation_weights = self._observation_weights[..., non_negative_frequencies, :]
            # Whatever expectation_type is: canonical coherence always pools
            # each window's trials and tapers.
            valid_counts = self._valid_counts_over((1, 2))[..., non_negative_frequencies]
            fewest = self._fewest_positive(valid_counts) or n_observations
        # When a group pair has more signals than observations, the two groups'
        # observation subspaces must intersect and the value is forced to 1 for
        # any data.
        degenerate_pairs = [
            (labels[first], labels[second])
            for first, second in combinations(range(len(labels)), 2)
            if len(group_indices[first]) + len(group_indices[second]) > fewest
        ]
        if degenerate_pairs:
            pairs = ", ".join(f"({first}, {second})" for first, second in degenerate_pairs)
            uses = self._uses_observations(
                fewest,
                n_observations,
                "time windows and frequencies whose trials x tapers observation_weights "
                "partly drop",
            )
            warnings.warn(
                f"canonical_coherence {uses} (trials x "
                f"tapers), fewer than the signals in group pair(s) {pairs}; "
                "their canonical coherence is forced to 1 regardless of the data. "
                "Provide more trials or tapers, or use smaller groups.",
                UserWarning,
                stacklevel=stacklevel_outside_package(),
            )
        fourier_coefficients = self._fourier_coefficients[..., non_negative_frequencies, :]
        if observation_weights is not None:
            # Canonical correlation is computed from observation covariance
            # matrices. Multiplying every observation by sqrt(weight) gives the
            # weighted covariance while retaining the existing whitening
            # implementation. A shared normalization by sum(weight) cancels
            # from the canonical correlation and is therefore unnecessary.
            fourier_coefficients = _weight_observations(
                fourier_coefficients, xp.sqrt(observation_weights)
            )
        # Group membership was resolved on the host by _validated_group_indices
        # (one ascending index array per sorted label); index the device
        # coefficients with device index arrays, as ``xp.isin`` over host labels
        # fails on CuPy.
        with self._decomposition_of_nonfinite_coefficients("canonical_coherence"):
            normalized_fourier_coefficients, ranks = zip(
                *(
                    _normalize_fourier_coefficients(
                        fourier_coefficients[..., xp.asarray(indices)]
                    )
                    for indices in group_indices
                ),
                strict=True,
            )
            canonical_coherency = xp.stack(
                [
                    _estimate_canonical_coherence(fourier_coefficients1, fourier_coefficients2)
                    for fourier_coefficients1, fourier_coefficients2 in combinations(
                        normalized_fourier_coefficients, 2
                    )
                ],
                axis=-1,
            )
        # Before the no-power and diagonal NaN are written in.
        self._require_finite_decomposition("canonical_coherence", canonical_coherency)

        n_groups = len(labels)
        new_shape = (self.time.size, self.frequencies.size, n_groups, n_groups)
        # A canonical correlation is at most 1; with fewer observations than
        # signals it reaches 1, where rounding left it up to 4e-15 above.
        magnitude = _clip_rounding_above_one(_squared_magnitude(canonical_coherency))
        # (n_time_windows, n_frequencies) bins already reported as undefined.
        no_valid_observations = xp.zeros(ranks[0].shape, dtype=bool)
        if valid_counts is not None:
            no_valid_observations = valid_counts == 0
        # A group with no power at a bin has no direction left after whitening;
        # its canonical coherence there is 0/0, so report NaN (as coherency does)
        # rather than the 0 the empty projection would give.
        rank = xp.stack(ranks, axis=-1)  # (n_time_windows, n_frequencies, n_groups)
        no_power = rank == 0
        pair_indices = np.array(list(combinations(range(n_groups), 2)))
        firsts, seconds = xp.asarray(pair_indices[:, 0]), xp.asarray(pair_indices[:, 1])
        magnitude = xp.where(
            no_power[..., firsts] | no_power[..., seconds] | no_valid_observations[..., None],
            xp.nan,
            magnitude,
        )
        # A group spans at most as many directions as the bin has observations.
        group_sizes = xp.asarray([len(indices) for indices in group_indices])
        full_rank = (
            xp.minimum(group_sizes, n_observations)
            if valid_counts is None
            else xp.minimum(group_sizes, valid_counts[..., xp.newaxis])
        )
        reported = ~no_valid_observations[..., xp.newaxis]
        # One host synchronization for both warnings: (2, n_groups) flags.
        flags = to_numpy(
            xp.stack(
                [
                    xp.any(no_power & reported, axis=(0, 1)),
                    xp.any((rank > 0) & (rank < full_rank) & reported, axis=(0, 1)),
                ]
            )
        )
        label_values = np.asarray(to_numpy(labels)).tolist()
        groups_without_power = [label_values[group] for group in np.flatnonzero(flags[0])]
        if groups_without_power:
            warnings.warn(
                f"canonical_coherence: group(s) {groups_without_power} have no power at "
                "some time-frequency bins (every signal in the group is zero there), so "
                "their canonical coherence is undefined and returned as NaN at those bins.",
                UserWarning,
                stacklevel=stacklevel_outside_package(),
            )
        groups_with_dependent_signals = [
            label_values[group] for group in np.flatnonzero(flags[1])
        ]
        if groups_with_dependent_signals:
            warnings.warn(
                f"canonical_coherence: group(s) {groups_with_dependent_signals} contain "
                "linearly dependent signals at some time-frequency bins (a dead, "
                "duplicated, or collinear channel); only the group's independent "
                "signals contribute there.",
                UserWarning,
                stacklevel=stacklevel_outside_package(),
            )

        canonical_coherence_magnitude = xp.full(new_shape, xp.nan)
        canonical_coherence_magnitude[..., firsts, seconds] = magnitude
        canonical_coherence_magnitude[..., seconds, firsts] = magnitude

        return to_numpy(canonical_coherence_magnitude), to_numpy(labels)

    def canonical_coherency(
        self,
        group_labels: NDArray[Any],
        *,
        rank: int | None = None,
        n_components: int = 1,
        regularization: float = TIKHONOV_REGULARIZATION_FACTOR,
    ) -> MultivariateConnectivityResult:
        """Return exact complex canonical coherency (CaCoh) components.

        This implements Vidaurre et al.'s phase-optimised CaCoh definition: for
        each candidate phase, the real projection of the between-group CSD is
        whitened by the real within-group CSDs, and the phase giving the largest
        singular value is selected. Scores are complex, with magnitude equal to
        the maximised coherence and phase encoded using MNE's
        ``magnitude * exp(-1j * phi)`` convention.

        The first component's score is
        ``max_{phi, a, b} a^T Re(exp(-1j phi) S_12) b / sqrt((a^T Re(S_11) a)
        (b^T Re(S_22) b)) * exp(-1j phi)`` over real weights ``a``, ``b``: its
        magnitude is the maximized coherence, not squared. For two
        single-signal groups the score is ``conj`` of :meth:`coherency`.

        Unlike the historical :meth:`canonical_coherence`, a different
        estimator (the squared canonical correlation over complex weights,
        neither the square nor the square root of this one), this method returns
        component-resolved scores, spatial filters, and Haufe-style patterns.
        Additional components are extracted by CCA-style deflation in the
        whitened space: each is sought in the orthogonal complement of the
        previous components' whitened directions, so successive component
        signals are uncorrelated within each group (``a_j^T Re(Caa) a_k = 0``
        for ``j != k``) and every component is invariant to invertible real
        mixing of a group's channels.

        Parameters
        ----------
        group_labels : array-like, shape (n_signals,)
            Label assigning each signal to a group; every unordered pair of
            groups is one connection.
        rank : int, optional
            Retain at most this many within-group whitening directions per group.
            ``None`` keeps every numerically non-zero direction. Applied
            identically to both groups of every connection. This is an
            estimation rank: it restricts the subspace every component is
            sought in, so it changes the scores themselves.
        n_components : int, default=1
            Number of coherency components to return, an output count: the
            earlier components are the same however many are returned. A
            connection whose smaller group has fewer channels returns NaN for
            the unavailable components.
        regularization : float, default=1e-12
            Relative diagonal loading used by the whitening decomposition.

        Returns
        -------
        MultivariateConnectivityResult
            Complex ``scores`` of shape ``(..., frequency, connection,
            component)`` plus real filters and patterns; see the class docstring.

        Notes
        -----
        **Range**: score magnitudes lie in ``[0, 1]``. The whitening,
        singular-value decomposition, and phase optimization (every local
        maximum of a 74-point coarse phase grid is refined by a batched Newton
        iteration and the best is kept, so near-equal lobes of the phase
        objective are resolved unless two lie within about ``2 * pi / 74`` of
        each other) are vectorized over the time/frequency axes on the active
        ``xp`` backend, so this runs on the GPU when GPU support is enabled.

        **Phase convention**: a spatial filter and its negative span the same
        direction, so the canonical phase is intrinsically defined only modulo
        pi. Each filter's sign is fixed so that the largest-magnitude
        coefficient of its spatial pattern is positive; with that convention
        the score is the conjugate of the canonical coherency between the
        positively oriented components, and for single-channel groups it
        reduces exactly to the conjugate pairwise coherency.

        **Relation to mne-connectivity**: the first component maximizes the
        same objective as mne-connectivity's ``cacoh`` and matches it
        (magnitude, and phase modulo pi) wherever both optimizers reach the
        global maximum. Components ``>= 2`` differ from mne-connectivity 0.9,
        which deflates the cross-spectrum with the previous components'
        channel-space filters; the whitened-space deflation used here keeps
        them uncorrelated within each group and invariant to invertible real
        within-group mixing.

        References
        ----------
        .. [1] Vidaurre C, et al. (2019) Canonical maximization of coherence: A
               novel tool for investigation of neuronal interactions between two
               datasets. NeuroImage 201:116009.
        .. [2] Haufe S, et al. (2014) On the interpretation of weight vectors of
               linear models in multivariate neuroimaging. NeuroImage 87:96-110.

        Examples
        --------
        >>> import numpy as np
        >>> from spectral_connectivity import Connectivity, Multitaper
        >>> rng = np.random.default_rng(0)
        >>> leader = rng.standard_normal((1003, 20))
        >>> # Group "a": two noisy copies of signal 0; "b": two of its 6 ms-delayed copy.
        >>> pair = np.stack([leader[3:], leader[:-3]], axis=-1)
        >>> signals = np.repeat(pair, 2, axis=-1)  # (time, trials, 4 signals)
        >>> signals += 0.5 * rng.standard_normal(signals.shape)
        >>> multitaper = Multitaper(signals, sampling_frequency=500)
        >>> connectivity = Connectivity.from_transform(multitaper)
        >>> result = connectivity.canonical_coherency(["a", "a", "b", "b"])
        >>> result.scores.shape  # (n_time_windows, n_frequencies, n_connections, n_components)
        (1, 501, 1, 1)
        >>> result.connections  # one row per group pair
        array([['a', 'b']], dtype='<U1')
        >>> result.filters.shape  # (..., n_connections, n_components, side, n_signals)
        (1, 501, 1, 1, 2, 4)
        >>> bool(abs(result.scores[0, 20, 0, 0]) > 0.5)  # |CaCoh| at 10 Hz (bin 20)
        True
        >>> # For single-signal groups the score is the conjugate coherency, unsquared.
        >>> two = Connectivity.from_transform(
        ...     Multitaper(signals[..., [0, 2]], sampling_frequency=500)
        ... )
        >>> score = two.canonical_coherency(["a", "b"]).scores[0, 20, 0, 0]
        >>> bool(np.isclose(score, np.conj(two.coherency()[0, 20, 0, 1])))
        True
        """
        return self._multivariate_component_result(
            "canonical_coherency",
            group_labels,
            rank=rank,
            n_components=n_components,
            regularization=regularization,
        )

    def maximized_imaginary_coherency_components(
        self,
        group_labels: NDArray[Any],
        *,
        rank: int | None = None,
        n_components: int = 1,
        regularization: float = TIKHONOV_REGULARIZATION_FACTOR,
    ) -> MultivariateConnectivityResult:
        """Return component-resolved MIC scores, filters, and patterns.

        The singular vectors of the whitened imaginary between-group CSD are
        returned in descending singular-value order. Filters map channel data
        to the components; patterns map the components back to channel space.
        This is the component-resolved counterpart of the scalar
        :meth:`maximized_imaginary_coherency`.

        Parameters
        ----------
        group_labels : array-like, shape (n_signals,)
            Label assigning each signal to a group; every unordered pair of
            groups is one connection.
        rank : int, optional
            Retain at most this many within-group whitening directions per group.
            ``None`` keeps every numerically non-zero direction. This is an
            estimation rank: it restricts the subspace the singular components
            are computed in, so it changes the scores themselves.
        n_components : int, default=1
            Number of singular components to return, an output count: the
            earlier components are the same however many are returned. A
            connection whose smaller group has fewer channels returns NaN for
            the unavailable components.
        regularization : float, default=1e-12
            Relative diagonal loading used by the whitening decomposition.

        Returns
        -------
        MultivariateConnectivityResult
            Real ``scores`` of shape ``(..., frequency, connection, component)``
            plus filters and patterns; see the class docstring.

        Notes
        -----
        **Range**: scores lie in ``[0, 1]``. The whitening and singular-value
        decomposition are vectorized over the time/frequency axes on the active
        ``xp`` backend, so this runs on the GPU when GPU support is enabled.

        References
        ----------
        .. [1] Ewald A, et al. (2012) Estimating true brain connectivity from EEG/
               MEG data invariant to linear and static transformations in sensor
               space. NeuroImage 60(1):476-488.

        Examples
        --------
        >>> import numpy as np
        >>> from spectral_connectivity import Connectivity, Multitaper
        >>> rng = np.random.default_rng(0)
        >>> leader = rng.standard_normal((1003, 20))
        >>> # Group "a": two noisy copies of signal 0; "b": two of its 6 ms-delayed copy.
        >>> pair = np.stack([leader[3:], leader[:-3]], axis=-1)
        >>> signals = np.repeat(pair, 2, axis=-1)  # (time, trials, 4 signals)
        >>> signals += 0.5 * rng.standard_normal(signals.shape)
        >>> multitaper = Multitaper(signals, sampling_frequency=500)
        >>> connectivity = Connectivity.from_transform(multitaper)
        >>> labels = ["a", "a", "b", "b"]
        >>> result = connectivity.maximized_imaginary_coherency_components(labels)
        >>> result.scores.shape  # (n_time_windows, n_frequencies, n_connections, n_components)
        (1, 501, 1, 1)
        >>> result.connections  # one row per group pair
        array([['a', 'b']], dtype='<U1')
        >>> result.patterns.shape  # (..., n_connections, n_components, side, n_signals)
        (1, 501, 1, 1, 2, 4)
        >>> bool(result.scores[0, 20, 0, 0] > 0.2)  # lagged a-b coupling at 10 Hz (bin 20)
        True
        """
        return self._multivariate_component_result(
            "maximized_imaginary_coherency_components",
            group_labels,
            rank=rank,
            n_components=n_components,
            regularization=regularization,
        )

    def _multivariate_component_result(
        self,
        method: Literal["canonical_coherency", "maximized_imaginary_coherency_components"],
        group_labels: NDArray[Any],
        *,
        rank: int | None,
        n_components: int,
        regularization: float,
    ) -> MultivariateConnectivityResult:
        """Compute rich CaCoh/MIC results from the expected CSD."""
        labels, group_indices, membership = self._validated_group_indices(group_labels)
        rank = _validated_rank(rank)
        if not is_positive_integer(n_components):
            msg = f"n_components must be a positive integer, got {n_components!r}."
            raise ValueError(msg)
        regularization = _validated_regularization(regularization)
        self._warn_single_observation_degenerate(
            method,
            _RANK_ONE_CANONICAL_CONSEQUENCE
            if method == "canonical_coherency"
            else _RANK_ONE_GROUP_CONSEQUENCE,
        )
        rank_cap = rank if rank is not None else self.n_signals
        # ``pairs`` is the single connection ordering shared by the capacities,
        # the ``connections`` labels, and the main loop below.
        pairs = list(combinations(range(len(labels)), 2))
        # The number of components a connection can support is bounded by the two
        # groups *in that connection* (and the requested rank), not by the
        # smallest group overall. Compute the per-connection capacity and reject
        # only when no connection can supply n_components; connections whose
        # groups are smaller return NaN for the unavailable components.
        connection_capacities = [
            min(len(group_indices[first]), len(group_indices[second]), rank_cap)
            for first, second in pairs
        ]
        max_components = max(connection_capacities)
        if n_components > max_components:
            msg = (
                f"n_components ({n_components}) must not exceed the largest "
                f"per-connection group rank/size ({max_components}); no group pair "
                f"is large enough to supply that many components."
            )
            raise ValueError(msg)

        spectrum = self._nonnegative_cross_spectral_matrix()
        leading_shape = spectrum.shape[:-3]
        n_frequencies = spectrum.shape[-3]
        connections = np.asarray([(labels[first], labels[second]) for first, second in pairs])
        n_connections = len(connections)
        score_dtype = xp.complex128 if method == "canonical_coherency" else float
        scores = xp.full(
            (*leading_shape, n_frequencies, n_connections, n_components),
            xp.nan,
            dtype=score_dtype,
        )
        projection_shape = (
            *leading_shape,
            n_frequencies,
            n_connections,
            n_components,
            2,
            self.n_signals,
        )
        filters = xp.full(projection_shape, xp.nan, dtype=float)
        patterns = xp.full(projection_shape, xp.nan, dtype=float)

        component_fn = (
            _canonical_coherency_components
            if method == "canonical_coherency"
            else _mic_components
        )
        any_phantom = False
        for connection_index, (first, second) in enumerate(pairs):
            first_indices = group_indices[first]
            second_indices = group_indices[second]
            n_first = len(first_indices)
            # This connection can only supply as many components as its smaller
            # group (and the rank cap); the rest stay NaN in the pre-filled array.
            component_count = min(n_components, connection_capacities[connection_index])
            combined = xp.asarray(np.concatenate((first_indices, second_indices)))
            # Sub-CSD over every leading/frequency bin at once: (..., freq, m, m).
            subsystem = spectrum[..., combined[:, xp.newaxis], combined[xp.newaxis, :]]
            # A non-finite bin (e.g. a dead channel) would make the batched
            # eigendecomposition fail; compute a placeholder there and mask it
            # back to NaN afterward, matching the old per-bin skip.
            finite_bin = xp.all(xp.isfinite(subsystem), axis=(-2, -1))
            identity = xp.eye(combined.shape[0], dtype=subsystem.dtype)
            safe = xp.where(finite_bin[..., xp.newaxis, xp.newaxis], subsystem, identity)
            local_scores, (filter_a, filter_b), (pattern_a, pattern_b), rank_here = (
                component_fn(
                    safe[..., :n_first, :n_first],
                    safe[..., :n_first, n_first:],
                    safe[..., n_first:, n_first:],
                    rank=rank,
                    n_components=component_count,
                    regularization=regularization,
                )
            )
            valid = finite_bin[..., xp.newaxis]
            scores[..., connection_index, :component_count] = xp.where(
                valid, local_scores, xp.nan
            )
            for side, (indices, side_filter, side_pattern) in enumerate(
                (
                    (first_indices, filter_a, pattern_a),
                    (second_indices, filter_b, pattern_b),
                )
            ):
                masked_filter = xp.where(valid[..., xp.newaxis], side_filter, xp.nan)
                masked_pattern = xp.where(valid[..., xp.newaxis], side_pattern, xp.nan)
                signal_index = xp.asarray(indices)
                # Assign one component at a time: a single trailing fancy index
                # (the group's channels) with only integer indices before it keeps
                # the scattered axis at the end, avoiding NumPy's mixed
                # slice/advanced-index dimension reordering.
                for component in range(component_count):
                    filters[..., connection_index, component, side, signal_index] = (
                        masked_filter[..., component]
                    )
                    patterns[..., connection_index, component, side, signal_index] = (
                        masked_pattern[..., component]
                    )
            # A rank-deficient within-group block (collinear/duplicated channels)
            # supplies fewer directions than requested; the extra "phantom"
            # components come back with a ~0 score and an all-zero filter. Flag it
            # via the eigenvalue-based rank (scale-invariant), not the filter norm.
            if bool(xp.any(finite_bin & (rank_here < component_count))):
                any_phantom = True

        if any_phantom:
            warnings.warn(
                f"{method}: some requested components fall in the null space of a "
                "rank-deficient within-group cross-spectrum (collinear or "
                "duplicated channels), so they are returned with a zero score and "
                "an all-zero spatial filter. Reduce n_components or pass an "
                "explicit rank to avoid these phantom components.",
                UserWarning,
                stacklevel=stacklevel_outside_package(),
            )

        return MultivariateConnectivityResult(
            method=method,
            scores=to_numpy(scores),
            connections=connections,
            group_labels=np.asarray(labels),
            group_membership=np.asarray(membership, dtype=bool),
            filters=to_numpy(filters),
            patterns=to_numpy(patterns),
        )

    def maximized_imaginary_coherency(
        self,
        group_labels: NDArray[np.integer],
        *,
        rank: int | None = None,
        regularization: float = TIKHONOV_REGULARIZATION_FACTOR,
    ) -> tuple[NDArray[np.floating], NDArray[np.integer]]:
        """Return maximized imaginary coherency (MIC) between signal groups.

        Each group's real within-group cross-spectrum is whitened before the
        largest singular value of the between-group imaginary cross-spectrum is
        taken. This makes the result invariant to invertible, static real-valued
        mixing within either group.

        Parameters
        ----------
        group_labels : array-like, shape (n_signals,)
            Label assigning each signal to a group.
        rank : int, optional
            Retain at most this many within-group whitening components. ``None``
            retains every numerically non-zero component independently per bin.
        regularization : float, default=1e-12
            Relative diagonal loading used by the whitening decomposition.

        Returns
        -------
        mic : array
            Shape ``(..., n_nonnegative_frequencies, n_groups, n_groups)``.
        labels : array, shape (n_groups,)
            Sorted unique group labels.

        Notes
        -----
        **Range**: ``[0, 1]``. The diagonal is returned as NaN.

        Examples
        --------
        >>> import numpy as np
        >>> from spectral_connectivity import Connectivity, Multitaper
        >>> rng = np.random.default_rng(0)
        >>> leader = rng.standard_normal((1003, 20))
        >>> # Group "a": two noisy copies of signal 0; "b": two of its 6 ms-delayed copy.
        >>> pair = np.stack([leader[3:], leader[:-3]], axis=-1)
        >>> signals = np.repeat(pair, 2, axis=-1)  # (time, trials, 4 signals)
        >>> signals += 0.5 * rng.standard_normal(signals.shape)
        >>> multitaper = Multitaper(signals, sampling_frequency=500)
        >>> connectivity = Connectivity.from_transform(multitaper)
        >>> mic, labels = connectivity.maximized_imaginary_coherency(["a", "a", "b", "b"])
        >>> mic.shape  # (n_time_windows, n_frequencies, n_groups, n_groups)
        (1, 501, 2, 2)
        >>> labels
        array(['a', 'b'], dtype='<U1')
        >>> bool(mic[0, 20, 0, 1] > 0.2)  # lagged a-b coupling at 10 Hz (bin 20)
        True
        """
        return self._group_imaginary_coherency(
            "maximized_imaginary_coherency",
            group_labels,
            lambda whitened: xp.clip(
                xp.linalg.svd(whitened, full_matrices=False, compute_uv=False)[..., 0],
                0.0,
                1.0,
            ),
            rank=rank,
            regularization=regularization,
        )

    def multivariate_interaction_measure(
        self,
        group_labels: NDArray[np.integer],
        *,
        rank: int | None = None,
        regularization: float = TIKHONOV_REGULARIZATION_FACTOR,
    ) -> tuple[NDArray[np.floating], NDArray[np.integer]]:
        """Return the multivariate interaction measure (MIM) between groups.

        MIM sums the squared singular values of the whitened imaginary
        cross-spectrum, incorporating every phase-lagged interaction component
        rather than only the strongest component returned by MIC.

        Parameters
        ----------
        group_labels : array-like, shape (n_signals,)
            Label assigning each signal to a group.
        rank : int, optional
            Retain at most this many within-group whitening components. ``None``
            retains every numerically non-zero component independently per bin.
        regularization : float, default=1e-12
            Relative diagonal loading used by the whitening decomposition.

        Returns
        -------
        mim : array
            Shape ``(..., n_nonnegative_frequencies, n_groups, n_groups)``.
        labels : array, shape (n_groups,)
            Sorted unique group labels.

        Notes
        -----
        **Range**: ``[0, min(rank_group_1, rank_group_2)]``; unlike MIC, MIM can
        exceed one. The diagonal is returned as NaN.

        References
        ----------
        .. [1] Ewald, A., Marzetti, L., Zappasodi, F., Meinecke, F.C., and
               Nolte, G. (2012). Estimating true brain connectivity from
               EEG/MEG data invariant to linear and static transformations in
               sensor space. NeuroImage 60, 476-488.

        Examples
        --------
        >>> import numpy as np
        >>> from spectral_connectivity import Connectivity, Multitaper
        >>> rng = np.random.default_rng(0)
        >>> leader = rng.standard_normal((1003, 20))
        >>> # Group "a": two noisy copies of signal 0; "b": two of its 6 ms-delayed copy.
        >>> pair = np.stack([leader[3:], leader[:-3]], axis=-1)
        >>> signals = np.repeat(pair, 2, axis=-1)  # (time, trials, 4 signals)
        >>> signals += 0.5 * rng.standard_normal(signals.shape)
        >>> multitaper = Multitaper(signals, sampling_frequency=500)
        >>> connectivity = Connectivity.from_transform(multitaper)
        >>> mim, labels = connectivity.multivariate_interaction_measure(["a", "a", "b", "b"])
        >>> mim.shape  # (n_time_windows, n_frequencies, n_groups, n_groups)
        (1, 501, 2, 2)
        >>> labels
        array(['a', 'b'], dtype='<U1')
        >>> bool(mim[0, 20, 0, 1] > 0.05)  # lagged a-b interaction at 10 Hz (bin 20)
        True
        """
        return self._group_imaginary_coherency(
            "multivariate_interaction_measure",
            group_labels,
            lambda whitened: xp.sum(whitened**2, axis=(-2, -1)),
            rank=rank,
            regularization=regularization,
        )

    def _group_imaginary_coherency(
        self,
        measure: str,
        group_labels: NDArray[np.integer],
        reduce: Callable[[BackendArray], BackendArray],
        *,
        rank: int | None,
        regularization: float,
    ) -> tuple[NDArray[np.floating], NDArray[np.integer]]:
        """Reduce whitened imaginary CSD blocks to a group-by-group measure.

        Shared by MIC and MIM. ``reduce`` maps each group pair's whitened
        imaginary cross-spectrum, shape ``(..., n_first, n_second)``, to its
        value, shape ``(...)``, which fills both ``[first, second]`` and
        ``[second, first]`` of the returned ``(..., n_groups, n_groups)`` array
        (diagonal NaN); the sorted labels are returned alongside. Invalid bins (for example the NaN edges of
        an ``edge_mode="nan"`` Morlet transform) are replaced with a safe value
        before the batched eigendecomposition/SVD so they cannot fail. Validity
        is tracked per group and combined per connection, so a NaN confined to
        one group only invalidates connections that include it -- an unrelated
        connection between two healthy groups is preserved.
        """
        labels, numpy_group_indices, _ = self._validated_group_indices(group_labels)
        rank = _validated_rank(rank)
        regularization = _validated_regularization(regularization)
        self._warn_single_observation_degenerate(measure, _RANK_ONE_GROUP_CONSEQUENCE)

        spectrum = self._nonnegative_cross_spectral_matrix()
        group_indices = [xp.asarray(indices) for indices in numpy_group_indices]
        # Whiten each group's within-block, substituting the identity at that
        # group's non-finite bins so the eigendecomposition converges there.
        inverse_square_roots = []
        group_finite = []
        for indices in group_indices:
            within = spectrum[..., indices[:, xp.newaxis], indices[xp.newaxis, :]].real
            finite = xp.all(xp.isfinite(within), axis=(-2, -1))
            identity = xp.eye(indices.shape[0], dtype=within.dtype)
            safe_within = xp.where(finite[..., xp.newaxis, xp.newaxis], within, identity)
            inverse_square_roots.append(
                _batched_inverse_square_root(
                    safe_within,
                    rank=rank,
                    regularization=regularization,
                )[0]
            )
            group_finite.append(finite)

        result = xp.full(
            (*spectrum.shape[:-2], len(labels), len(labels)), xp.nan, dtype=spectrum.real.dtype
        )
        for first, second in combinations(range(len(labels)), 2):
            first_indices = group_indices[first]
            second_indices = group_indices[second]
            # A connection is valid only where both of its groups are finite;
            # zero the between-block elsewhere so the SVD stays finite.
            connection_finite = group_finite[first] & group_finite[second]
            between = spectrum[
                ...,
                first_indices[:, xp.newaxis],
                second_indices[xp.newaxis, :],
            ].imag
            between = xp.where(connection_finite[..., xp.newaxis, xp.newaxis], between, 0.0)
            whitened = xp.matmul(
                xp.matmul(inverse_square_roots[first], between),
                inverse_square_roots[second],
            )
            value = xp.where(connection_finite, reduce(whitened), xp.nan)
            result[..., first, second] = value
            result[..., second, first] = value
        return to_numpy(result), to_numpy(labels)

    def global_coherence(
        self,
        max_rank: int = 1,
        *,
        max_workspace_elements: int = GLOBAL_COHERENCE_BATCH_CHUNK_ELEMENTS,
    ) -> tuple[NDArray[np.floating], NDArray[np.complexfloating]]:
        """Find linear combinations that capture the most coherent power.

        The linear combinations of signals that capture the most coherent
        power at each frequency, from the cross-spectral matrix averaged over
        the observations ``expectation_type`` averages.

        This is a frequency domain analog of PCA over signals at a given
        frequency (and time window, for the default expectation). Global
        coherence [1]_ is the leading component's fraction: the largest
        eigenvalue of the cross-spectral matrix divided by its trace.

        Parameters
        ----------
        max_rank : int, default=1
            The number of components returned, strongest first (like the
            number of PC dimensions kept); more than
            ``min(n_signals, n_observations)`` is clamped with a warning.
            It limits the output only: the decomposition uses the whole
            cross-spectral matrix, each value is a fraction of the total power
            of all components, and a returned component does not depend on
            how many are returned. It is not an estimation rank like the
            ``rank`` of the group measures.

            .. versionchanged:: 3.0
               Components are always strongest first. 2.x returned them
               weakest first when ``1 < max_rank < n_signals - 1``.
        max_workspace_elements : int, default=16_000_000
            Approximate working-set target, in array elements, for the batched
            decomposition: frequency bins are processed in chunks sized so the
            main intermediates stay near this many complex elements (the default
            ~16M ≈ 256 MB of complex128). It is a soft target, not a hard memory
            cap — it counts the dominant per-bin intermediates, not the outputs or
            LAPACK's internal workspace, and it never goes below one bin per chunk,
            so actual peak memory is somewhat higher. Lower it to reduce peak
            memory on a constrained CPU or GPU (at the cost of more, smaller
            chunks); the default favors speed and does not change the result.
            Ignored on the per-bin fallback path used for a large decomposition
            dimension.

        Returns
        -------
        global_coherence : ndarray
            Shape (..., n_frequencies, n_components), the leading axes those
            ``expectation_type`` keeps (``n_time_windows`` for the default).
            The fraction of total coherent power captured by each component
            (eigenvalue of the cross-spectral matrix divided by the sum of all
            eigenvalues), ordered strongest component first.
        vectors : ndarray
            Shape (..., n_frequencies, n_signals, n_components).
            Each component's unit-norm weights over the signals (the left
            singular vectors of the coefficient matrix), in the same order;
            the wrappers call them ``global_coherence_vectors``.

        Raises
        ------
        ValueError
            If ``expectation_type`` keeps the tapers as an output axis
            (``"time"``, ``"trials"``, ``"time_trials"``), if
            ``max_workspace_elements`` is not a positive integer, or if the
            Fourier coefficients contain NaN or Inf (where weights do not omit
            them).

        Warns
        -----
        UserWarning
            If ``max_rank`` exceeds ``min(n_signals, n_observations)``, the
            number of components there are; it is clamped to that.
        UserWarning
            If some bins have (near-)zero total power, such as a flat channel,
            all-zero input, or a bin whose observation weights are all zero:
            those bins are NaN.
        UserWarning
            If the pooled observations are fewer than the signals
            (``n_observations < n_signals``): the value is then at least
            ``1 / n_observations`` whatever the data (1 for a single
            observation), above the ``1 / n_signals`` of independent signals.
            Where zero ``observation_weights`` drop observations, the count is
            that of the bin keeping fewest (:attr:`n_valid_observations`).
            Also if the expectation
            pools time windows the transform reports as correlated
            (``time_bins_are_independent`` is ``False``, e.g. windows
            overlapping by more than half), which add fewer effective
            observations than their count.

        Notes
        -----
        **Frequency axis**: the non-negative frequencies, labelled by
        :attr:`frequencies`, as for every other measure.

        .. versionchanged:: 3.0
           2.x returned every bin of the two-sided transform, labelled by
           :attr:`all_frequencies`. For a real signal the negative bins repeat
           the non-negative ones (the same fractions, conjugated vectors).
           Until 3.2, a call on two-sided coefficients emits a
           :class:`~spectral_connectivity.SidednessWarning` once per
           Python session as a reminder.

        **Observations**: the matrix decomposed at each frequency is the
        cross-spectral matrix averaged over the axes ``expectation_type``
        averages, which must include the tapers: ``"trials_tapers"`` (the
        default) pools each time window's trials and tapers,
        ``"time_trials_tapers"`` every window, trial and taper (one value per
        frequency), ``"time_tapers"`` each trial's windows and tapers, and
        ``"tapers"`` the tapers of each window and trial. The kept axes lead
        the result in the order of the "Result shape by expectation_type"
        table of :class:`Connectivity`. Observation weights give the weighted
        average: each pooled coefficient is scaled by the square root of its
        weight.

        .. versionchanged:: 3.0
           It follows ``expectation_type``; 2.x pooled each window's trials
           and tapers whatever the setting, as ``"trials_tapers"`` does.

        **Range**: [0, 1]. Each value is the fraction of total coherent power
        in that component, so the components sum to at most 1 and rescaling
        every signal by the same factor leaves them unchanged. Rescaling
        signals separately does change them: the decomposition is of the
        signals as given, so put them on a common scale first if their units
        differ.

        **Interpreting the value**:

        - *Floor*: with ``n`` pooled observations (:attr:`n_observations`:
          trials x tapers by default, times the windows when time is pooled)
          and ``p`` signals, the leading value is at least ``1 / min(n, p)``
          whatever the data, and 1 for one observation; it warns when
          ``n < p``. Observations that zero ``observation_weights`` drop (the
          dropped windows of ``Multitaper(artifact_samples=...)`` or the masked
          edges of a Morlet transform, say) do not count: ``n`` is each bin's
          :attr:`n_valid_observations`, and the warning sees the bin keeping
          fewest.
        - *Bias without coupling*: independent signals of equal power have the
          value ``1 / p``, but the estimate is well above it unless
          ``n >> p``. As ``n`` and ``p`` grow with ``p / n`` fixed, the largest
          eigenvalue of their cross-spectral matrix tends to
          ``(1 + sqrt(p / n))**2`` times the common power [4]_, [5]_ and the
          trace to ``p`` times it, so the estimate tends to
          ``(1 + sqrt(p / n))**2 / p``: 4 times the true value at ``n = p``
          and about 1.5 times at ``n = 20 p``. At finite sizes it is
          somewhat lower (approximately, by a fixed-seed Monte Carlo of
          complex Gaussian coefficients: 2.95 times at ``p = n = 8``, 3.57
          times at ``p = n = 32``, and 1.37 to 1.45 times at ``n = 20 p``).
          Compare values only at matched ``n`` and ``p``
          (trial counts, rejected epochs, and removed channels all change
          them), or against a null simulated at the same ``n`` and ``p``.
        - *Effective observations*: correlated observations (overlapping
          windows; overlapping Welch segments or Morlet smoothing,
          ``observations_are_independent=False``) count fewer than ``n``, so
          the bias is larger than ``n`` suggests. Pool non-overlapping windows:
          with a full taper set, 50% overlap, which does not yet set
          ``time_bins_are_independent=False`` or warn, still leaves about half
          the nominal count (by Monte Carlo, 7 tapers: about 39 effective of
          70), and halving the step over the same span adds almost nothing.
        - *Pooling time*: pooling windows whose coherent pattern changes
          lowers the value, so a smaller value after longer pooling is not by
          itself evidence of weaker coherence.
        - *Raw cross-spectrum*: the matrix is not normalized to coherency, so
          high-power channels weigh more, and per-channel gain, a common
          reference, and volume conduction raise the value without coupling.
          Published analyses re-reference to a surface Laplacian first [2]_,
          [3]_.
        - *Mean, not median*: pooling over time is an equal-weight mean, as in
          Chronux ``CrossSpecMatc`` (whose ``Ctot`` is ``"time_trials_tapers"``
          on non-overlapping windows), not the median over windows of the
          taper-averaged matrices that Purdon et al. [2]_ and Weiner et al.
          [3]_ decompose, which is not provided.
        - *0 Hz and Nyquist*: the coefficients of a real signal are real
          there (:attr:`real_valued_frequencies`), so the matrix is real and
          its bias without coupling slightly larger than at the other
          frequencies.
        - No significance test or interval is returned.

        **Algorithm**: when the number of estimates (the pooled
        observations, :attr:`n_observations`) is at least ``n_signals`` and
        ``n_signals`` is small
        (``<= 64``), the components are obtained from an eigendecomposition of
        the ``(n_signals, n_signals)`` cross-spectral matrix ``A @ Aᴴ`` rather
        than a singular value decomposition of ``A``. This is substantially
        faster but squares the condition number, so for a nearly rank-deficient
        cross-spectral matrix (near-duplicate channels) the *weakest* returned
        components (large ``max_rank``) may lose relative precision. The dominant
        component(s) — the usual use of this measure — are unaffected. A thin
        matrix (fewer estimates than signals) diagonalizes the smaller
        ``(n_estimates, n_estimates)`` matrix ``Aᴴ @ A`` in the same way and maps
        each eigenvector ``v`` to the vector ``A v / |A v|``, with the same
        precision caveat; a component with (numerically) zero power, at most
        ``eps * n_estimates`` times the strongest, gets a zero vector.

        References
        ----------
        .. [1] Cimenser, A., Purdon, P.L., Pierce, E.T., Walsh, J.L.,
               Salazar-Gomez, A.F., Harrell, P.G., Tavares-Stoeckel, C.,
               Habeeb, K., and Brown, E.N. (2011). Tracking brain states under
               general anesthesia by using global coherence analysis.
               Proceedings of the National Academy of Sciences 108(21),
               8832-8837. https://doi.org/10.1073/pnas.1017041108
        .. [2] Purdon, P.L., Pierce, E.T., Mukamel, E.A., Prerau, M.J., Walsh,
               J.L., Wong, K.F.K., Salazar-Gomez, A.F., Harrell, P.G., Sampson,
               A.L., Cimenser, A., et al. (2013). Electroencephalogram
               signatures of loss and recovery of consciousness from propofol.
               Proceedings of the National Academy of Sciences 110(12),
               E1142. https://doi.org/10.1073/pnas.1221180110
        .. [3] Weiner, V.S., Zhou, D.W., Kahali, P., Stephen, E.P.,
               Peterfreund, R.A., Aglio, L.S., Szabo, M.D., Eskandar, E.N.,
               Salazar-Gomez, A.F., Sampson, A.L., Cash, S.S., Brown, E.N., and
               Purdon, P.L. (2023). Propofol disrupts alpha dynamics in
               functionally distinct thalamocortical networks during loss of
               consciousness. Proceedings of the National Academy of Sciences
               120(11), e2207831120. https://doi.org/10.1073/pnas.2207831120
        .. [4] Marčenko, V.A., and Pastur, L.A. (1967). Distribution of
               eigenvalues for some sets of random matrices. Mathematics of the
               USSR-Sbornik 1(4), 457-483.
               https://doi.org/10.1070/SM1967v001n04ABEH001994
        .. [5] Johansson, K. (2000). Shape fluctuations and random matrices.
               Communications in Mathematical Physics 209(2), 437-476.
               https://doi.org/10.1007/s002200050027

        Examples
        --------
        >>> import numpy as np
        >>> from spectral_connectivity import Connectivity, Multitaper
        >>> rng = np.random.default_rng(0)
        >>> leader = rng.standard_normal((1003, 20))
        >>> # Signal 1 is signal 0 delayed by 3 samples (6 ms at 500 Hz), plus noise.
        >>> signals = np.stack([leader[3:], leader[:-3]], axis=-1)  # (time, trials, signals)
        >>> signals += 0.5 * rng.standard_normal(signals.shape)
        >>> multitaper = Multitaper(signals, sampling_frequency=500)
        >>> connectivity = Connectivity.from_transform(multitaper)
        >>> global_coherence, vectors = connectivity.global_coherence(max_rank=1)
        >>> global_coherence.shape  # (n_time_windows, n_frequencies, n_components)
        (1, 501, 1)
        >>> vectors.shape  # (n_time_windows, n_frequencies, n_signals, n_components)
        (1, 501, 2, 1)
        >>> float(connectivity.frequencies[20])
        10.0
        >>> # One component captures most of the power of the coupled pair (10 Hz = bin 20).
        >>> bool(global_coherence[0, 20, 0] > 0.8)
        True

        Pool time windows for one value per frequency, as Chronux
        ``CrossSpecMatc`` does: eight channels of one minute-long recording
        share a 10 Hz oscillation, and every non-overlapping 2 s window
        (``time_window_step`` defaults to the duration) adds its 3 tapers.
        ``n_observations`` should exceed ``n_signals``, ideally by a wide
        margin. Unlike Purdon et al. [2]_, this is the mean over windows, not
        the median; it decomposes the raw cross-spectrum, with no Laplacian
        unless you apply one; and it returns no significance test.

        >>> t = np.arange(60 * 200) / 200  # 60 s at 200 Hz
        >>> shared = np.sin(2 * np.pi * 10 * t)[:, np.newaxis] * np.linspace(0.5, 1.5, 8)
        >>> recording = (shared + rng.standard_normal((t.size, 8)))[:, np.newaxis, :]
        >>> multitaper = Multitaper(
        ...     recording,
        ...     sampling_frequency=200,
        ...     time_window_duration=2,
        ...     time_halfbandwidth_product=2,
        ...     n_tapers=3,
        ... )
        >>> pooled = Connectivity.from_transform(multitaper, expectation_type="time_trials_tapers")
        >>> pooled.n_observations, pooled.n_signals  # 30 windows x 3 tapers, 8 signals
        (90, 8)
        >>> fractions, _ = pooled.global_coherence()
        >>> fractions.shape  # (n_frequencies, n_components): no time axis
        (201, 1)
        >>> bool(fractions[20, 0] > 0.9)  # 10 Hz
        True
        """
        self._validate_multiple_signals()
        pooled_axes = self._expectation_axes
        if 2 not in pooled_axes:
            others = [
                repr(name)
                for name, axes in EXPECTATION_AXES.items()
                if 2 in axes and name != "trials_tapers"
            ]
            supported = ", ".join(["'trials_tapers' (the default)", *others])
            msg = (
                f"global_coherence needs the tapers averaged, but "
                f"expectation_type={self.expectation_type!r} keeps them as an "
                "output axis. Global coherence decomposes one cross-spectral "
                "matrix pooled over the tapers (and whatever else is averaged); "
                "a value for each taper has no established meaning. Use one of "
                f"{supported}, e.g. Connectivity.from_transform(transform, "
                "expectation_type='trials_tapers'), or set "
                "conn.expectation_type = 'trials_tapers' on this instance."
            )
            raise ValueError(msg)
        self._warn_few_global_coherence_observations()
        if 0 in pooled_axes:
            # Overlapping windows share data, so pooling them adds fewer
            # effective observations than their count. Only when time is
            # pooled: correlated trial/taper observations alone (Morlet
            # smoothing, overlapping Welch segments) are left to the Notes, so
            # the default use of those transforms does not start warning.
            self._warn_correlated_observations(
                "global_coherence",
                "Its bias above the 1/n_signals of independent signals, and its "
                "floor of 1/min(n_observations, n_signals), are set by the number "
                "of independent observations pooled",
                time_bin_remedy="use non-overlapping windows (time_window_step >= "
                "time_window_duration; with a full taper set even 50% overlap leaves "
                "about half the nominal count of effective observations), more "
                "MorletWavelet decimation, or an expectation_type that does not "
                "average over time",
            )

        # A rank-r decomposition of the (n_signals, n_observations) coefficient
        # matrix has at most min(n_signals, n_observations) non-trivial
        # components. Requesting more than that would crash svds or (in the
        # dense branch) broadcast a single component into duplicates, so clamp
        # the requested rank to what is realizable.
        max_available_rank = min(self.n_signals, self.n_observations)
        if max_rank > max_available_rank:
            warnings.warn(
                f"max_rank={max_rank} exceeds the number of available "
                f"global-coherence components "
                f"(min(n_signals, n_observations) = {max_available_rank}); "
                f"clamping to {max_available_rank}.",
                UserWarning,
                stacklevel=stacklevel_outside_package(),
            )
            max_rank = max_available_rank

        # Must be a genuine positive integer: it is a floor-divided into a chunk
        # size, so a float (NaN/inf included) or bool would either pick a
        # nonsensical chunk or blow up later inside range(). bool is an int
        # subclass, so reject it explicitly.
        if not is_positive_integer(max_workspace_elements):
            msg = (
                f"max_workspace_elements must be a positive integer (e.g. "
                f"1_000_000), got {max_workspace_elements!r}. It bounds the memory "
                f"budget, in array elements, for global_coherence's batched "
                f"decomposition; lower it to reduce peak memory."
            )
            raise ValueError(msg)

        # The kernel decomposes, for each index of its first axis, the columns
        # spanned by its second and third axes. Put the kept observation axes
        # (in time, trials, tapers order, as the other measures keep them)
        # together on the first axis and the pooled ones on the next two. For
        # every supported expectation_type these are views; for the default,
        # "trials_tapers", the layout is the coefficients' own. Only the
        # non-negative frequencies are reported, so only those are decomposed:
        # for a real signal S(-f) = conj(S(f)), and the negative bins have the
        # same fractions and conjugated vectors.
        coefficient_shape = self._fourier_coefficients.shape
        kept_axes = self._kept_observation_axes
        kept_shape = tuple(coefficient_shape[axis] for axis in kept_axes)
        n_nonnegative = self._nonnegative_frequency_count(coefficient_shape[3])
        pooled_layout = (
            int(np.prod(kept_shape)),
            int(np.prod([coefficient_shape[axis] for axis in pooled_axes[:-1]])),
            coefficient_shape[pooled_axes[-1]],
        )

        def pooled(values: BackendArray) -> BackendArray:
            arranged = xp.transpose(
                values[..., :n_nonnegative, :], (*kept_axes, *pooled_axes, 3, 4)
            )
            return arranged.reshape((*pooled_layout, *arranged.shape[3:]))

        weights = self._observation_weights
        # NaN or Inf coefficients (unless zero weights omit them) make the
        # decomposition fail, which is reported in the user's terms; NumPy's
        # "invalid value" RuntimeWarning from the scaling before it would only
        # repeat that, so it is silenced for such coefficients.
        with (
            self._decomposition_of_nonfinite_coefficients("global_coherence"),
            contextlib.nullcontext()
            if self._coefficients_are_finite
            else np.errstate(invalid="ignore", divide="ignore"),
        ):
            global_coherence, unnormalized_global_coherence = _global_coherence(
                pooled(self._fourier_coefficients),
                None if weights is None else pooled(weights),
                max_rank,
                max_workspace_elements,
            )
        global_coherence = global_coherence.reshape((*kept_shape, *global_coherence.shape[1:]))
        unnormalized_global_coherence = unnormalized_global_coherence.reshape(
            (*kept_shape, *unnormalized_global_coherence.shape[1:])
        )
        self._require_finite_decomposition("global_coherence", global_coherence)

        if xp.any(xp.isnan(global_coherence)):
            warnings.warn(
                "Some time-frequency bins have (near-)zero total power, so "
                "global coherence is undefined there and is returned as NaN. "
                "This usually indicates a flat/dead channel or all-zero input.",
                UserWarning,
                stacklevel=stacklevel_outside_package(),
            )

        if not self._is_one_sided and _emit_migration_notices.get():
            # Remove in 3.2, with DirectedOrientationWarning (issue #98).
            warn_migration(
                f"{_ORIENTATION_WARNING_PREFIX}Connectivity.global_coherence returns "
                "only the non-negative frequencies, labelled by "
                "Connectivity.frequencies (2.x returned every bin of the two-sided "
                "transform, labelled by all_frequencies; for a real signal the "
                "negative bins only repeated the positive ones). Label the frequency axis with "
                "frequencies and drop indexing from the end of the axis. It also "
                "honours expectation_type; the default is unchanged. This notice "
                f"will be removed in 3.2; see the migration guide: {_MIGRATION_GUIDE_URL}. "
                + _silence_hint(SidednessWarning),
                SidednessWarning,
            )
        return to_numpy(global_coherence), to_numpy(unnormalized_global_coherence)

    def _phase_locking_value(
        self, dtype: DTypeLike | None = None, *, warn_single_observation: bool = True
    ) -> NDArray[np.complexfloating]:
        # Normalize each Fourier coefficient to unit magnitude, then reuse the
        # batched reduced cross-spectral matmul: because
        #   (z_i conj(z_j)) / |z_i conj(z_j)| = (z_i / |z_i|) conj(z_j / |z_j|),
        # the mean over observations of the normalized per-observation
        # cross-spectrum equals the reduced cross-spectral matrix of the
        # unit-normalized coefficients -- with no per-observation outer product,
        # so peak memory is O(observations * signals) rather than
        # O(observations * signals**2). Kept on the active namespace (xp); the
        # public ``phase_locking_value`` wrapper converts to NumPy.
        self._validate_multiple_signals()
        if warn_single_observation:
            self._warn_single_observation_degenerate("phase_locking_value")
        # Normalize at the computation dtype (``self._dtype``, complex128 by
        # default): the previous materialized path formed the outer product at
        # that dtype, so normalizing complex64 inputs at their own precision here
        # would let float32 rounding push the unit magnitudes -- and thus the
        # averaged PLV/PPC -- slightly past 1. copy=False avoids a copy when the
        # dtype already matches (the division below allocates a fresh array).
        # Only the non-negative frequencies are reported, so only those are
        # normalized and reduced.
        coefficients = self._nonnegative_fourier_coefficients(dtype)
        magnitude = xp.abs(coefficients)
        zero_magnitude = magnitude == 0
        if self._observation_weights is not None:
            # An observation zero weights drop (a zero-filled invalid window)
            # never reaches the average.
            zero_magnitude &= self._observation_weights[..., : coefficients.shape[-2], :] > 0
        if bool(xp.any(zero_magnitude)):
            warnings.warn(
                "Some cross-spectrum entries have zero magnitude (e.g. a "
                "flat/dead channel or all-zero input at a taper/trial), so "
                "the phase-locking normalization z / |z| is undefined there "
                "and is returned as NaN.",
                UserWarning,
                stacklevel=stacklevel_outside_package(),
            )
        # z / |z| is undefined where |z| == 0; those coefficients are NaN
        # (rather than leaking a RuntimeWarning). A NaN coefficient at any
        # observation makes every pair involving it reduce to NaN, matching the
        # previous per-observation path where a zero-magnitude cross-spectrum
        # entry became NaN before averaging.
        normalized: NDArray[np.complexfloating] = _divide_where(
            coefficients, magnitude, magnitude > 0, xp.nan
        )
        return self._reduced_cross_spectral_matrix(normalized, dtype=dtype)

    @_refused_for_exact_spectrum("per_observation")
    @_asnumpy
    def phase_locking_value(self) -> NDArray[np.floating]:
        """Return the cross-spectrum with power scaled to magnitude 1.

        The phase locking value attempts to mitigate power differences
        between realizations (tapers or trials) by treating all values of
        the cross-spectrum as the same power. This has the effect of
        downweighting high power realizations and upweighting low power
        realizations.

        Returns
        -------
        phase_locking_value : array
            Shape ``(..., n_nonnegative_frequencies, n_signals, n_signals)``.
            Phase locking values between all signal pairs.

        Notes
        -----
        **Range**: [0, 1]. 0 indicates random phases; 1 indicates
        constant phase difference.

        References
        ----------
        .. [1] Lachaux, J.-P., Rodriguez, E., Martinerie, J., Varela, F.J.,
               and others (1999). Measuring phase synchrony in brain
               signals. Human Brain Mapping 8, 194-208.

        Examples
        --------
        >>> import numpy as np
        >>> from spectral_connectivity import Connectivity, Multitaper
        >>> rng = np.random.default_rng(0)
        >>> leader = rng.standard_normal((1003, 20))
        >>> # Signal 1 is signal 0 delayed by 3 samples (6 ms at 500 Hz), plus noise.
        >>> signals = np.stack([leader[3:], leader[:-3]], axis=-1)  # (time, trials, signals)
        >>> signals += 0.5 * rng.standard_normal(signals.shape)
        >>> multitaper = Multitaper(signals, sampling_frequency=500)
        >>> connectivity = Connectivity.from_transform(multitaper)
        >>> plv = connectivity.phase_locking_value()
        >>> plv.shape  # (n_time_windows, n_frequencies, n_signals, n_signals)
        (1, 501, 2, 2)
        >>> bool(plv[0, 20, 0, 1] > 0.5)  # consistent phase difference at 10 Hz (bin 20)
        True
        """
        # Clip to the documented [0, 1] range: |mean of unit-magnitude entries|
        # is <= 1 mathematically, but floating-point rounding can leave it a few
        # ulp above 1 (matching the bounds clipping magnitude_squared_coherence applies).
        return xp.clip(xp.abs(self._phase_locking_value()), 0.0, 1.0)

    @_refused_for_exact_spectrum("per_observation")
    @_asnumpy
    def corrected_imaginary_phase_locking_value(self) -> NDArray[np.floating]:
        """Return corrected imaginary phase-locking value (ciPLV).

        ciPLV removes the contribution of zero- and pi-lag phase locking while
        correcting the imaginary PLV for the reduction in its attainable range.

        Returns
        -------
        corrected_imaginary_phase_locking_value : array
            Shape ``(..., n_nonnegative_frequencies, n_signals, n_signals)``.

        Notes
        -----
        **Range**: ``[0, 1]``.  Exact zero- or pi-lag locking has a zero
        numerator and denominator and is defined as zero. A signal with
        itself is such a case, but its PLV is computed: rounding can leave its
        real part just below 1 and its imaginary part of rounding size, and
        the square root in the denominator amplifies that by about
        ``1 / sqrt(eps)``. The diagonal is therefore rounding noise below about
        ``sqrt(eps)`` of the computation precision, and exactly 0 only where
        the rounding cancels. The PLV behind this measure is formed in at least
        complex128 whatever ``dtype`` is, so that noise is below about 1.5e-8,
        and a small consistent lag (1e-4 rad, say) of complex64 coefficients
        still gives 1; the result is returned at the precision of ``dtype``.
        A zero-power signal's PLV is NaN, and so is its diagonal entry here.

        References
        ----------
        .. [1] Bruña, R., Maestú, F., and Pereda, E. (2018). Phase locking
               value revisited: teaching new tricks to an old dog. Journal of
               Neural Engineering 15, 056011.

        Examples
        --------
        >>> import numpy as np
        >>> from spectral_connectivity import Connectivity, Multitaper
        >>> rng = np.random.default_rng(0)
        >>> leader = rng.standard_normal((1003, 20))
        >>> # Signal 1 is signal 0 delayed by 3 samples (6 ms at 500 Hz), plus noise.
        >>> signals = np.stack([leader[3:], leader[:-3]], axis=-1)  # (time, trials, signals)
        >>> signals += 0.5 * rng.standard_normal(signals.shape)
        >>> multitaper = Multitaper(signals, sampling_frequency=500)
        >>> connectivity = Connectivity.from_transform(multitaper)
        >>> ciplv = connectivity.corrected_imaginary_phase_locking_value()
        >>> ciplv.shape  # (n_time_windows, n_frequencies, n_signals, n_signals)
        (1, 501, 2, 2)
        >>> # The 6 ms lag is not zero-phase, so lagged locking remains (10 Hz = bin 20).
        >>> bool(ciplv[0, 20, 0, 1] > 0.2)
        True
        """
        # ``1 - Re(PLV)**2`` cancels for a small consistent lag, which a
        # complex64 PLV cannot represent: a lag of 1e-4 rad rounds Re(PLV) to
        # 1, and the ratio to 0 instead of 1. The coefficients carry the lag
        # to their own precision, so the PLV is formed in at least complex128.
        complex_plv = self._phase_locking_value(xp.promote_types(self._dtype, xp.complex128))
        numerator = xp.abs(complex_plv.imag)
        denominator_squared = xp.maximum(0.0, 1.0 - complex_plv.real**2)
        denominator = xp.sqrt(denominator_squared)
        # The only mathematically valid zero-denominator case also has a zero
        # numerator (perfect zero- or pi-lag locking). Define that limit as 0.
        # A NaN PLV (zero-power channel) must stay NaN rather than fall into
        # that zero limit, matching every other phase-locking measure.
        nonzero = denominator > xp.finfo(denominator.dtype).tiny
        result = _divide_where(numerator, denominator, nonzero, 0.0)
        result[xp.isnan(complex_plv)] = xp.nan
        # Reported at the precision of ``dtype``, like the other measures.
        return xp.clip(result, 0.0, 1.0).astype(xp.finfo(self._dtype).dtype, copy=False)

    @cached_property
    def _imaginary_moment_cache(self) -> dict[str, BackendArray]:
        """Lazily populated phase-lag moments tied to the current inputs."""
        return {}

    def _has_no_phase_lag(self, mean_absolute: NDArray[np.floating]) -> NDArray[np.bool_]:
        """Pairs whose imaginary cross-spectrum is zero up to rounding.

        For in-phase signals (a channel and a scaled copy of it) each
        observation's ``Im(X_i conj(X_j))`` is rounding noise of order
        ``eps * |X_i| |X_j|``, not exactly 0, so ``E[|Im S_ij|]`` is compared
        with the pair's power scale ``sqrt(P_i P_j)`` rather than with 0. The
        test is relative, so it does not depend on the signal's amplitude.

        Parameters
        ----------
        mean_absolute : array, shape (..., n_nonnegative_frequencies, n_signals, n_signals)
            ``E[|Im S_ij|]`` from :meth:`_imaginary_cross_spectrum_moments`, at
            the non-negative frequencies.

        Returns
        -------
        no_lag : array of bool, shape (..., n_nonnegative_frequencies, n_signals, n_signals)
        """
        tolerance = _ZERO_PHASE_LAG_EPSILONS * xp.finfo(mean_absolute.dtype).eps
        power_scale = self._nonnegative_pairwise_power_scale()
        no_lag: NDArray[np.bool_] = mean_absolute <= tolerance * power_scale
        return no_lag

    def _imaginary_cross_spectrum_moments(
        self, *keys: str
    ) -> tuple[NDArray[np.floating], ...]:
        """Reduced moments of the per-observation imaginary cross-spectrum.

        The phase-lag-index family (``phase_lag_index``,
        ``weighted_phase_lag_index``, ``debiased_squared_phase_lag_index``, and
        ``debiased_squared_weighted_phase_lag_index``) each average a function --
        ``sign``, its square, identity, ``abs`` or square -- of the imaginary part of the
        per-observation cross-spectral matrix, with the diagonal zeroed. This
        returns the requested reduced moments from a cache, computing any that
        are missing from signal-row tiles of the observation-level
        cross-spectrum. Only the non-negative bins are reduced, and each tile
        covers targets from its first row on; the strict lower triangle is then
        filled by pair symmetry (``_IMAGINARY_MOMENT_PAIR_SYMMETRY``).

        Which moments a call computes depends on the regime. Without
        observation weights and with at least
        ``PHASE_LAG_ALL_MOMENTS_MIN_OBSERVATIONS`` averaged observations,
        forming the tiles (memory traffic over every observation) dominates the
        cost and each reduction is a single further pass over a tile, so the
        first request reduces all five moments from one formation; that costs
        less than forming the tiles again for a later measure that needs
        different moments. The five reduced moments (each ``(n_time_windows,
        n_frequencies, n_signals, n_signals)`` for the default expectation) are
        then retained until :meth:`clear_cache`. In this regime a single
        measure is also faster and has a lower peak than before (741 -> 536 MB
        of traced allocation for one measure on a 100-trial, 5-taper,
        1001-bin, 32-signal spectrum, with the four moments other than
        ``nonzero``), because the tiles reuse two buffers instead of fresh
        temporaries. Adding ``nonzero`` added one reduced moment, 8 MB, to that
        peak and no measurable time.

        Otherwise only the missing requested moments are computed. With few
        observations (e.g. a time-resolved single-trial spectrum with 3
        tapers) each window-resolved moment is nearly as large as the tile, so
        writing and retaining two extra ones cost more than re-forming the
        tiles later (a lone ``phase_lag_index`` on 1199 windows x 3 tapers x
        100 bins x 32 signals took 1.49x as long and 1.5x the peak memory).
        With observation weights each moment is a full weighted
        :meth:`_expectation` with tile-sized temporaries; computing all four
        made a lone weighted measure 7-14% (``phase_lag_index``) and 33-54%
        (``weighted_phase_lag_index``) slower on a Hann-weighted Morlet
        transform (4000 samples, 40 trials, 16 signals, 30 frequencies).

        Each tile is reduced immediately (see :meth:`_reduce_phase_lag_tile`)
        into two workspace buffers allocated once per call, avoiding an
        observation-resolved ``n_signals**2`` intermediate. The reduced
        ``n_signals**2`` outputs are unavoidable. The cached moments are
        invalidated with the other cached intermediates and are treated as
        read-only (callers copy before any in-place edit).

        Parameters
        ----------
        *keys : str
            Any of ``"sign"``, ``"nonzero"``, ``"imaginary"``, ``"absolute"``,
            ``"squared"`` for ``E[sign(Im)]``, ``E[sign(Im)**2]`` (the fraction
            of observations whose ``Im`` is not 0), ``E[Im]``, ``E[|Im|]`` and
            ``E[Im**2]``.

        Returns
        -------
        tuple of arrays, each shape (..., n_nonnegative_frequencies, n_signals, n_signals)
            The requested moments, in the order of ``keys``, at the non-negative
            frequencies the phase-lag measures report.
        """
        self._validate_multiple_signals()
        cache = self._imaginary_moment_cache
        if any(key not in cache for key in keys):
            coefficients = self._nonnegative_fourier_coefficients()
            n_signals = coefficients.shape[-1]
            kept_observation_axes = [
                axis for axis in range(3) if axis not in self._expectation_axes
            ]
            result_shape = (
                *[coefficients.shape[axis] for axis in kept_observation_axes],
                coefficients.shape[-2],
                n_signals,
                n_signals,
            )
            real_dtype = coefficients.real.dtype
            # Without weights and with many observations all the moments are
            # cheap single-pass reductions of the tile, so they are computed
            # together (see PHASE_LAG_ALL_MOMENTS_MIN_OBSERVATIONS). Otherwise
            # only the missing requested ones are: with weights each is a full
            # ``_expectation`` with tile-sized temporaries, and with few
            # observations the reduced moments are nearly as large as the tile.
            # Filled here and cached only once complete, so an error partway
            # through leaves no uninitialized moments behind.
            if (
                self._observation_weights is None or self._kept_observations is not None
            ) and self.n_observations >= PHASE_LAG_ALL_MOMENTS_MIN_OBSERVATIONS:
                computed_keys = list(_IMAGINARY_MOMENTS)
            else:
                computed_keys = list(dict.fromkeys(key for key in keys if key not in cache))
            moments = {key: xp.empty(result_shape, dtype=real_dtype) for key in computed_keys}

            observation_frequency_elements = int(np.prod(coefficients.shape[:-1]))
            elements_per_source = max(1, observation_frequency_elements * n_signals)
            signals_per_block = max(
                1,
                min(
                    n_signals,
                    PHASE_LAG_INDEX_MAX_WORKSPACE_ELEMENTS // elements_per_source,
                ),
            )

            # Nonlinear sign/abs/square transforms prevent contracting the
            # observation axes before the outer product. Form only a source-row
            # tile at a time, reduce it immediately, and write the small result.
            # Im(X_i conj(X_j)) = Im(X_i) Re(X_j) - Re(X_i) Im(X_j), formed in
            # real arithmetic rather than as a complex product. Each tile pairs
            # its source rows only with targets from ``start`` on; the strict
            # lower triangle is filled by pair symmetry below.
            # Contiguous copies: transforms leave the signal axis slowest, which
            # makes the tiles' broadcast products stride badly through memory.
            real = xp.ascontiguousarray(coefficients.real)
            imag = xp.ascontiguousarray(coefficients.imag)
            # Each tile is a contiguous view of the leading part of these flat
            # buffers, so later (narrower) tiles reuse the first tile's memory.
            workspace_size = observation_frequency_elements * signals_per_block * n_signals
            tile_buffer = xp.empty(workspace_size, dtype=real_dtype)
            scratch_buffer = xp.empty(workspace_size, dtype=real_dtype)
            observation_frequency_shape = coefficients.shape[:-1]
            block_diagonal = xp.arange(signals_per_block)
            # Contracts a tile's observation axes in the squared moment, e.g.
            # "abcdef,abcdef->adef" (time, trial, taper, frequency, row, column).
            letters = "abcdef"
            kept = "".join(
                letter
                for axis, letter in enumerate(letters)
                if axis not in self._expectation_axes
            )
            squared_subscripts = f"{letters},{letters}->{kept}"
            for start in range(0, n_signals, signals_per_block):
                stop = min(n_signals, start + signals_per_block)
                tile_shape = (*observation_frequency_shape, stop - start, n_signals - start)
                tile_size = (
                    observation_frequency_elements * (stop - start) * (n_signals - start)
                )
                imaginary = tile_buffer[:tile_size].reshape(tile_shape)
                scratch = scratch_buffer[:tile_size].reshape(tile_shape)
                xp.multiply(
                    imag[..., start:stop, xp.newaxis],
                    real[..., xp.newaxis, start:],
                    out=imaginary,
                )
                xp.multiply(
                    real[..., start:stop, xp.newaxis],
                    imag[..., xp.newaxis, start:],
                    out=scratch,
                )
                xp.subtract(imaginary, scratch, out=imaginary)
                local_diagonal = block_diagonal[: stop - start]
                imaginary[..., local_diagonal, local_diagonal] = 0
                self._reduce_phase_lag_tile(
                    imaginary, scratch, moments, start, stop, squared_subscripts
                )

            # Overwrite the strict lower triangle from the upper one: the pairs
            # each tile skipped and the in-tile lower entries alike.
            rows, columns = xp.tril_indices(n_signals, k=-1)
            for key, reduced in moments.items():
                reduced[..., rows, columns] = (
                    _IMAGINARY_MOMENT_PAIR_SYMMETRY[key] * reduced[..., columns, rows]
                )
            cache.update(moments)
        return tuple(cache[key] for key in keys)

    def _reduce_phase_lag_tile(
        self,
        imaginary: BackendArray,
        scratch: BackendArray,
        moments: dict[str, BackendArray],
        start: int,
        stop: int,
        squared_subscripts: str,
    ) -> None:
        """Average the phase-lag moments of one tile into ``moments``.

        Parameters
        ----------
        imaginary : array, shape (n_time_windows, n_trials, n_tapers, n_nonnegative_frequencies, stop - start, n_signals - start)
            ``Im(X_i conj(X_j))`` per observation for source rows
            ``start:stop`` and targets ``start:``, with the diagonal zeroed.
        scratch : array, same shape as ``imaginary``
            Workspace; overwritten.
        moments : dict of str to array, each shape (..., n_nonnegative_frequencies, n_signals, n_signals)
            Reduced moments to fill, keyed as ``_IMAGINARY_MOMENTS``: all of
            them or the requested subset (see
            :meth:`_imaginary_cross_spectrum_moments`). The block
            ``[..., start:stop, start:]`` of each is written.
        start, stop : int
            The tile's source rows.
        squared_subscripts : str
            ``einsum`` subscripts contracting ``imaginary * imaginary`` over the
            averaged observation axes (used without observation weights).

        Notes
        -----
        Without observation weights every moment is a plain mean, reduced in a
        single pass over the tile. The sign moment is counted from boolean
        comparisons, which are False for NaN, so bins with any NaN observation
        are set to NaN explicitly, as the mean of ``sign`` would give. The
        ``imaginary`` moment is the mean of the same per-observation values as
        ``absolute`` rather than the imaginary part of the (differently
        rounded) cross-spectral matrix, so their ratio in
        ``weighted_phase_lag_index`` is exactly 1 in magnitude at a constant
        lag above the no-lag guard (:meth:`_has_no_phase_lag`, below which it
        is defined as 0). With observation weights each moment goes through
        :meth:`_expectation`, which needs the per-observation values.
        """
        block = (..., slice(start, stop), slice(start, None))
        keep = self._kept_observations
        if self._observation_weights is not None and keep is None:
            if "sign" in moments or "nonzero" in moments:
                # One sign pass into the scratch buffer serves both moments.
                xp.sign(imaginary, out=scratch)
                if "sign" in moments:
                    moments["sign"][block] = self._expectation(scratch)
                if "nonzero" in moments:
                    xp.abs(scratch, out=scratch)
                    moments["nonzero"][block] = self._expectation(scratch)
            for key, reduced in moments.items():
                if key not in ("sign", "nonzero"):
                    reduced[block] = self._expectation(_IMAGINARY_MOMENTS[key](imaginary))
            return

        observation_axes = self._expectation_axes
        if keep is not None:
            # Zero the dropped observations in place (they add nothing to any
            # sum) and divide each sum by the count kept.
            keep = keep[..., 0, 0].reshape((*keep.shape[:3], 1, 1, 1))
            xp.copyto(imaginary, 0, where=~keep)
            count = xp.sum(keep, axis=observation_axes)

            def mean(total: BackendArray) -> BackendArray:
                return _divide_where(total, count, count > 0, xp.nan)

        else:
            n_observations = self.n_observations

            def mean(total: BackendArray) -> BackendArray:
                return total / n_observations

        if "sign" in moments or "nonzero" in moments:
            positive = xp.count_nonzero(imaginary > 0, axis=observation_axes)
            negative = xp.count_nonzero(imaginary < 0, axis=observation_axes)
            invalid = xp.any(xp.isnan(imaginary), axis=observation_axes)
            if "sign" in moments:
                moments["sign"][block] = xp.where(invalid, xp.nan, mean(positive - negative))
            if "nonzero" in moments:
                moments["nonzero"][block] = xp.where(
                    invalid, xp.nan, mean(positive + negative)
                )
        if "imaginary" in moments:
            moments["imaginary"][block] = (
                xp.mean(imaginary, axis=observation_axes)
                if keep is None
                else mean(xp.sum(imaginary, axis=observation_axes))
            )
        if "squared" in moments:
            if keep is None:
                # Sum of squares without materializing imaginary * imaginary.
                squared_sum = xp.einsum(squared_subscripts, imaginary, imaginary)
            else:
                # As the weighted path sums them, to the bit.
                xp.multiply(imaginary, imaginary, out=scratch)
                squared_sum = xp.sum(scratch, axis=observation_axes)
            moments["squared"][block] = mean(squared_sum)
        if "absolute" in moments:
            xp.abs(imaginary, out=scratch)
            moments["absolute"][block] = (
                xp.mean(scratch, axis=observation_axes)
                if keep is None
                else mean(xp.sum(scratch, axis=observation_axes))
            )

    @_refused_for_exact_spectrum("per_observation")
    @_asnumpy
    def phase_lag_index(self) -> NDArray[np.floating]:
        """Return non-parametric synchrony measure mitigating power differences.

        A non-parametric synchrony measure designed to mitigate power
        differences between realizations (tapers, trials) and
        volume-conduction.

        The phase lag index is the average sign of the imaginary
        component of the cross-spectrum. The imaginary component sets
        in-phase or anti-phase signals to zero and the sign scales it to
        have the same magnitude regardless of phase.

        Note that this is the signed version of the phase lag index. In order
        to obtain the unsigned version, as in [1], take the absolute value
        of this quantity.

        Returns
        -------
        phase_lag_index : array
            Shape ``(..., n_nonnegative_frequencies, n_signals, n_signals)``.
            Signed phase lag index values for all signal pairs. Positive
            ``[..., i, j]`` means signal ``i`` leads signal ``j``.

        Notes
        -----
        **Range**: [-1, 1] (signed version). For unsigned version (as in [1]),
        take absolute value to get range [0, 1]. In-phase or anti-phase pairs,
        whose imaginary cross-spectrum is zero up to rounding, are 0.

        References
        ----------
        .. [1] Stam, C.J., Nolte, G., and Daffertshofer, A. (2007). Phase
               lag index: Assessment of functional connectivity from multi
               channel EEG and MEG with diminished bias from common
               sources. Human Brain Mapping 28, 1178-1193.

        Examples
        --------
        >>> import numpy as np
        >>> from spectral_connectivity import Connectivity, Multitaper
        >>> rng = np.random.default_rng(0)
        >>> leader = rng.standard_normal((1003, 20))
        >>> # Signal 1 is signal 0 delayed by 3 samples (6 ms at 500 Hz), plus noise.
        >>> signals = np.stack([leader[3:], leader[:-3]], axis=-1)  # (time, trials, signals)
        >>> signals += 0.5 * rng.standard_normal(signals.shape)
        >>> multitaper = Multitaper(signals, sampling_frequency=500)
        >>> connectivity = Connectivity.from_transform(multitaper)
        >>> pli = connectivity.phase_lag_index()
        >>> pli.shape  # (n_time_windows, n_frequencies, n_signals, n_signals)
        (1, 501, 2, 2)
        >>> # Signal 0 leads, so [..., 0, 1] is positive and [..., 1, 0] negative (10 Hz).
        >>> bool(pli[0, 20, 0, 1] > 0 > pli[0, 20, 1, 0])
        True
        """

        self._warn_single_observation_degenerate(
            "phase_lag_index",
            "every value is the sign of one imaginary cross-spectrum, forced to "
            "+/-1 (a perfectly consistent lag)",
        )
        # E[sign(Im)] of the cross-spectrum (real-valued). Pairs with no phase
        # lag (in-phase signals) have Im at rounding level, whose sign is noise,
        # so they are 0. xp.where returns a fresh array, not the cached moment.
        mean_sign, mean_absolute = self._imaginary_cross_spectrum_moments("sign", "absolute")
        pli: NDArray[np.floating] = xp.where(
            self._has_no_phase_lag(mean_absolute), 0.0, mean_sign.real
        )
        return pli

    @_refused_for_exact_spectrum("per_observation")
    @_asnumpy
    def directed_phase_lag_index(self) -> NDArray[np.floating]:
        """Return the directed phase-lag index (dPLI).

        Values above 0.5 indicate that one signal consistently phase-leads the
        other; values below 0.5 indicate that it phase-lags.  A value of 0.5
        represents no preferred phase-lag direction, including in-phase or
        anti-phase pairs, whose imaginary cross-spectrum is zero up to rounding.

        Returns
        -------
        directed_phase_lag_index : array
            Shape ``(..., n_nonnegative_frequencies, n_signals, n_signals)``.
            ``[..., i, j]`` above 0.5 means signal ``i`` leads signal ``j``;
            below 0.5 means signal ``i`` lags signal ``j``.

        Notes
        -----
        **Range**: ``[0, 1]``.  With the convention ``H(0) = 0.5``, dPLI is
        ``(1 + signed_PLI) / 2``. Consequently ``dPLI[i, j] = 1 - dPLI[j, i]``
        and the diagonal is 0.5.

        References
        ----------
        .. [1] Stam, C.J., and van Straaten, E.C.W. (2012). Go with the flow:
               use of a directed phase lag index (dPLI) to characterize
               patterns of phase relations in a large-scale model of brain
               dynamics. NeuroImage 62, 1415-1428.

        Examples
        --------
        >>> import numpy as np
        >>> from spectral_connectivity import Connectivity, Multitaper
        >>> rng = np.random.default_rng(0)
        >>> leader = rng.standard_normal((1003, 20))
        >>> # Signal 1 is signal 0 delayed by 3 samples (6 ms at 500 Hz), plus noise.
        >>> signals = np.stack([leader[3:], leader[:-3]], axis=-1)  # (time, trials, signals)
        >>> signals += 0.5 * rng.standard_normal(signals.shape)
        >>> multitaper = Multitaper(signals, sampling_frequency=500)
        >>> connectivity = Connectivity.from_transform(multitaper)
        >>> dpli = connectivity.directed_phase_lag_index()
        >>> dpli.shape  # (n_time_windows, n_frequencies, n_signals, n_signals)
        (1, 501, 2, 2)
        >>> # Signal 0 leads signal 1, so [..., 0, 1] > 0.5 > [..., 1, 0] (10 Hz = bin 20).
        >>> bool(dpli[0, 20, 0, 1] > 0.5 > dpli[0, 20, 1, 0])
        True
        """
        self._warn_single_observation_degenerate(
            "directed_phase_lag_index",
            "every value is set by the sign of one imaginary cross-spectrum, "
            "forced to 0 or 1 (a perfectly consistent lag)",
        )
        # Pairs with no phase lag (in-phase signals) have Im at rounding level,
        # whose sign is noise, so they are neutral (0.5).
        mean_sign, mean_absolute = self._imaginary_cross_spectrum_moments("sign", "absolute")
        directed_pli: NDArray[np.floating] = xp.where(
            self._has_no_phase_lag(mean_absolute),
            0.5,
            xp.clip((1.0 + mean_sign.real) / 2.0, 0.0, 1.0),
        )
        return directed_pli

    @_refused_for_exact_spectrum("per_observation")
    @_asnumpy
    def weighted_phase_lag_index(self) -> NDArray[np.floating]:
        """Return weighted average of phase lag index using imaginary coherency magnitudes.

        Weighted average of the phase lag index using the imaginary
        coherency magnitudes as weights.

        Note that this is the signed version of the weighted phase lag index
        (mirroring :meth:`phase_lag_index`). In order to obtain the unsigned
        version, as in [1], take the absolute value of this quantity.

        Returns
        -------
        weighted_phase_lag_index : array
            Shape ``(..., n_nonnegative_frequencies, n_signals, n_signals)``.
            Signed weighted phase lag index values. Positive ``[..., i, j]``
            means signal ``i`` leads signal ``j``.

        Notes
        -----
        **Range**: [-1, 1] (signed version). For the unsigned version (as in
        [1]), take the absolute value to get range [0, 1]. The result is
        antisymmetric in the signal pair (``wpli[..., i, j] = -wpli[..., j, i]``).

        References
        ----------
        .. [1] Vinck, M., Oostenveld, R., van Wingerden, M., Battaglia, F.,
               and Pennartz, C.M.A. (2011). An improved index of
               phase-synchronization for electrophysiological data in the
               presence of volume-conduction, noise and sample-size bias.
               NeuroImage 55, 1548-1565.

        Examples
        --------
        >>> import numpy as np
        >>> from spectral_connectivity import Connectivity, Multitaper
        >>> rng = np.random.default_rng(0)
        >>> leader = rng.standard_normal((1003, 20))
        >>> # Signal 1 is signal 0 delayed by 3 samples (6 ms at 500 Hz), plus noise.
        >>> signals = np.stack([leader[3:], leader[:-3]], axis=-1)  # (time, trials, signals)
        >>> signals += 0.5 * rng.standard_normal(signals.shape)
        >>> multitaper = Multitaper(signals, sampling_frequency=500)
        >>> connectivity = Connectivity.from_transform(multitaper)
        >>> wpli = connectivity.weighted_phase_lag_index()
        >>> wpli.shape  # (n_time_windows, n_frequencies, n_signals, n_signals)
        (1, 501, 2, 2)
        >>> # Signal 0 leads, so [..., 0, 1] is positive and [..., 1, 0] negative (10 Hz).
        >>> bool(wpli[0, 20, 0, 1] > 0 > wpli[0, 20, 1, 0])
        True
        """

        self._warn_single_observation_degenerate(
            "weighted_phase_lag_index",
            "every value is one imaginary cross-spectrum divided by its own "
            "magnitude, forced to +/-1 (a perfectly consistent lag)",
        )
        mean_imaginary, mean_absolute = self._imaginary_cross_spectrum_moments(
            "imaginary", "absolute"
        )
        # Pairs with no phase lag (in-phase signals, the zeroed diagonal) have
        # E[Im] and E[|Im|] both at rounding level, so their ratio is noise; the
        # 0/0 is defined as 0, matching phase_lag_index's sign(0) == 0.
        no_lag = self._has_no_phase_lag(mean_absolute)
        return _divide_where(mean_imaginary, mean_absolute, ~no_lag, 0.0)

    @_refused_for_exact_spectrum("per_observation")
    @_asnumpy
    def debiased_squared_phase_lag_index(self) -> NDArray[np.floating]:
        """Return square of phase lag index corrected for positive bias.

        The square of the phase lag index corrected for the positive
        bias induced by using the magnitude of the complex cross-spectrum.

        Returns
        -------
        phase_lag_index : array
            Shape ``(..., n_nonnegative_frequencies, n_signals, n_signals)``.
            Debiased squared phase lag index values (symmetric in the signal
            pair, so they carry no lead/lag direction).

        Notes
        -----
        **Range**: [-1 / (n_observations - 1), 1]. The unbiased finite-sample
        estimate can be negative when the observed phase consistency is below
        its null bias; negative values do not represent negative coupling.
        Pairs whose imaginary cross-spectrum is exactly zero for every
        observation (in-phase signals, and the diagonal) have no phase lag to
        estimate and are returned as 0 rather than the lower bound.

        References
        ----------
        .. [1] Vinck, M., Oostenveld, R., van Wingerden, M., Battaglia, F.,
               and Pennartz, C.M.A. (2011). An improved index of
               phase-synchronization for electrophysiological data in the
               presence of volume-conduction, noise and sample-size bias.
               NeuroImage 55, 1548-1565.

        Examples
        --------
        >>> import numpy as np
        >>> from spectral_connectivity import Connectivity, Multitaper
        >>> rng = np.random.default_rng(0)
        >>> leader = rng.standard_normal((1003, 20))
        >>> # Signal 1 is signal 0 delayed by 3 samples (6 ms at 500 Hz), plus noise.
        >>> signals = np.stack([leader[3:], leader[:-3]], axis=-1)  # (time, trials, signals)
        >>> signals += 0.5 * rng.standard_normal(signals.shape)
        >>> multitaper = Multitaper(signals, sampling_frequency=500)
        >>> connectivity = Connectivity.from_transform(multitaper)
        >>> debiased_pli = connectivity.debiased_squared_phase_lag_index()
        >>> debiased_pli.shape  # (n_time_windows, n_frequencies, n_signals, n_signals)
        (1, 501, 2, 2)
        >>> bool(debiased_pli[0, 20, 0, 1] > 0.05)  # lagged coupling at 10 Hz (bin 20)
        True
        """
        self._validate_counted_observations("debiased_squared_phase_lag_index")
        mean_sign, mean_nonzero, mean_absolute = self._imaginary_cross_spectrum_moments(
            "sign", "nonzero", "absolute"
        )
        n_observations = self._counts_for_correction(
            "debiased_squared_phase_lag_index", mean_sign.dtype
        )
        # The mean of s_k s_l over ordered pairs of distinct observations,
        # (n**2 m**2 - sum_k s_k**2) / (n (n - 1)), is unbiased for E[s]**2.
        # sum_k s_k**2 / n is the fraction of observations whose Im is not 0;
        # Vinck's closed form (n m**2 - 1) / (n - 1) takes it as 1. With
        # observations dropped by zero weights, n is each bin's kept count and
        # the moments are means over the kept observations.
        # A bin of one kept observation divides by 0; it is NaN below.
        with np.errstate(divide="ignore", invalid="ignore"):
            debiased = (n_observations * mean_sign**2 - mean_nonzero) / (n_observations - 1.0)
        # Where there is no phase lag (in-phase signals, the zeroed diagonal)
        # the signs are rounding noise; there is no lag to estimate, so 0.
        result = xp.where(self._has_no_phase_lag(mean_absolute), 0.0, debiased)
        return self._nan_where_too_few(result, n_observations)

    @_refused_for_exact_spectrum("per_observation")
    @_asnumpy
    def debiased_squared_weighted_phase_lag_index(self) -> NDArray[np.floating]:
        """Return square of weighted phase lag index corrected for bias.

        The square of the weighted phase lag index corrected for the
        positive bias induced by using the magnitude of the complex
        cross-spectrum.

        Returns
        -------
        weighted_phase_lag_index : array
            Shape ``(..., n_nonnegative_frequencies, n_signals, n_signals)``.
            Debiased squared weighted phase lag index values (symmetric in the
            signal pair, so they carry no lead/lag direction).

        Notes
        -----
        **Range**: [-1, 1]. The debiased finite-sample estimate can be negative
        when the signed cross-products are dominated by inconsistent phase lags;
        negative values do not represent negative coupling. Pairs whose
        imaginary cross-spectrum is zero up to rounding (in-phase signals, the
        diagonal, and the purely real DC and Nyquist bins) have no phase lag to
        estimate and are returned as 0.

        References
        ----------
        .. [1] Vinck, M., Oostenveld, R., van Wingerden, M., Battaglia, F.,
               and Pennartz, C.M.A. (2011). An improved index of
               phase-synchronization for electrophysiological data in the
               presence of volume-conduction, noise and sample-size bias.
               NeuroImage 55, 1548-1565.

        Examples
        --------
        >>> import numpy as np
        >>> from spectral_connectivity import Connectivity, Multitaper
        >>> rng = np.random.default_rng(0)
        >>> leader = rng.standard_normal((1003, 20))
        >>> # Signal 1 is signal 0 delayed by 3 samples (6 ms at 500 Hz), plus noise.
        >>> signals = np.stack([leader[3:], leader[:-3]], axis=-1)  # (time, trials, signals)
        >>> signals += 0.5 * rng.standard_normal(signals.shape)
        >>> multitaper = Multitaper(signals, sampling_frequency=500)
        >>> connectivity = Connectivity.from_transform(multitaper)
        >>> debiased_wpli = connectivity.debiased_squared_weighted_phase_lag_index()
        >>> debiased_wpli.shape  # (n_time_windows, n_frequencies, n_signals, n_signals)
        (1, 501, 2, 2)
        >>> bool(debiased_wpli[0, 20, 0, 1] > 0.3)  # lagged coupling at 10 Hz (bin 20)
        True
        """
        self._validate_counted_observations("debiased_squared_weighted_phase_lag_index")
        mean_imaginary, mean_squared, mean_absolute = self._imaginary_cross_spectrum_moments(
            "imaginary", "squared", "absolute"
        )
        n_observations = self._counts_for_correction(
            "debiased_squared_weighted_phase_lag_index", mean_imaginary.dtype
        )
        # Each product is a fresh array, so the cached moments are not mutated.
        # With observations dropped by zero weights the moments are means over
        # each bin's kept observations, and these their sums.
        imaginary_csm_sum = mean_imaginary * n_observations
        squared_imaginary_csm_sum = mean_squared * n_observations
        imaginary_csm_magnitude_sum = mean_absolute * n_observations
        weights = imaginary_csm_magnitude_sum**2 - squared_imaginary_csm_sum
        # Pairs with no phase lag (in-phase signals, the zeroed diagonal) make
        # the ratio one of rounding errors or 0/0; there is no lag to estimate,
        # so 0, matching debiased_squared_phase_lag_index.
        debiased: NDArray[np.floating] = xp.where(
            self._has_no_phase_lag(mean_absolute),
            0.0,
            _divide_where(
                imaginary_csm_sum**2 - squared_imaginary_csm_sum, weights, weights != 0, xp.nan
            ),
        )
        return self._nan_where_too_few(debiased, n_observations)

    @_refused_for_exact_spectrum("per_observation")
    @_asnumpy
    def pairwise_phase_consistency(self) -> NDArray[np.floating]:
        """Return square of phase locking value corrected for bias.

        The square of the phase locking value corrected for the
        positive bias induced by using the magnitude of the complex
        cross-spectrum.

        Returns
        -------
        phase_locking_value : array
            Shape ``(..., n_nonnegative_frequencies, n_signals, n_signals)``.
            Pairwise phase consistency values.

        Notes
        -----
        **Range**: [-1 / (n_observations - 1), 1]. The unbiased finite-sample
        estimate can be negative when phase consistency is below its null bias;
        negative values do not represent negative coupling.

        References
        ----------
        .. [1] Vinck, M., van Wingerden, M., Womelsdorf, T., Fries, P., and
               Pennartz, C.M.A. (2010). The pairwise phase consistency: A
               bias-free measure of rhythmic neuronal synchronization.
               NeuroImage 51, 112-122.

        Examples
        --------
        >>> import numpy as np
        >>> from spectral_connectivity import Connectivity, Multitaper
        >>> rng = np.random.default_rng(0)
        >>> leader = rng.standard_normal((1003, 20))
        >>> # Signal 1 is signal 0 delayed by 3 samples (6 ms at 500 Hz), plus noise.
        >>> signals = np.stack([leader[3:], leader[:-3]], axis=-1)  # (time, trials, signals)
        >>> signals += 0.5 * rng.standard_normal(signals.shape)
        >>> multitaper = Multitaper(signals, sampling_frequency=500)
        >>> connectivity = Connectivity.from_transform(multitaper)
        >>> ppc = connectivity.pairwise_phase_consistency()
        >>> ppc.shape  # (n_time_windows, n_frequencies, n_signals, n_signals)
        (1, 501, 2, 2)
        >>> bool(ppc[0, 20, 0, 1] > 0.3)  # consistent phase difference at 10 Hz (bin 20)
        True
        """
        self._validate_counted_observations("pairwise_phase_consistency")
        # A bin of a single observation is reported once, as PPC's.
        phase_locking_value = self._phase_locking_value(warn_single_observation=False)
        n_observations = self._counts_for_correction(
            "pairwise_phase_consistency", phase_locking_value.real.dtype
        )
        # The mean over ordered pairs of distinct observations of
        # Re(u_k conj(u_l)), unit phasors u: (|sum_k u_k|**2 - n) / (n (n - 1)).
        # With observations dropped by zero weights, n is each bin's kept count
        # and the PLV their mean.
        plv_sum = phase_locking_value * n_observations
        if isinstance(n_observations, int):
            ppc = (plv_sum * plv_sum.conjugate() - n_observations) / (
                n_observations**2 - n_observations
            )
            return ppc.real
        ppc_values: NDArray[np.floating] = _divide_where(
            (plv_sum * plv_sum.conjugate()).real - n_observations,
            n_observations**2 - n_observations,
            n_observations >= 2,
            xp.nan,
        )
        return ppc_values

    @staticmethod
    def _nan_where_too_few(
        values: BackendArray, n_observations: int | BackendArray
    ) -> BackendArray:
        """``values`` with NaN in the bins of fewer than two observations.

        ``n_observations`` is :meth:`_counts_for_correction`'s count: an
        ``int`` (every bin has it, so ``values`` is returned as it is) or one
        count per bin.
        """
        if isinstance(n_observations, int):
            return values
        return xp.where(n_observations >= 2, values, xp.nan)

    @_orientation_changed
    @_source_first
    @_asnumpy
    @_factorization_explained_by_nonfinite_input
    def pairwise_spectral_granger_prediction(self) -> NDArray[np.floating]:
        """Return amount of power at a node explained by other nodes.

        The amount of power at a node in a frequency explained by (is
        predictive of) the power at other nodes.

        Also known as spectral granger causality.

        .. versionchanged:: 3.0
           The output is source first (``[..., i, j]`` is ``i -> j``); 2.x
           returned the transpose (``j -> i``).

        Returns
        -------
        pairwise_granger : array
            Shape ``(..., n_nonnegative_frequencies, n_signals, n_signals)``.
            Spectral Granger prediction values. Output ``[..., i, j]`` is the
            influence of signal ``i`` on signal ``j`` (``i -> j``).

        Notes
        -----
        **Non-negativity**: spectral Granger is ``>= 0`` by definition.
        Negative estimates within roundoff of zero -- above ``-100 * eps`` of
        the result dtype on the log-ratio scale, about ``-2e-14`` for float64
        -- are clipped to ``0``; materially negative bins below that threshold
        (a degenerate factorization) are returned as ``NaN``. Other
        packages (FieldTrip, MVGC, mne-connectivity) return such values as-is.

        **Range**: [0, ∞). Non-negative values with no finite upper bound.

        **Window and FFT length**: this measure comes from the Wilson
        factorization of the estimated spectrum. It needs a window longer
        than the slowest correlation in the data, and its values depend
        slightly on the FFT length (``n_fft_samples``): see the Notes of
        :func:`~spectral_connectivity.minimum_phase_decomposition.minimum_phase_decomposition`.

        References
        ----------
        .. [1] Geweke, J. (1982). Measurement of Linear Dependence and
               Feedback Between Multiple Time Series. Journal of the
               American Statistical Association 77, 304.

        Examples
        --------
        >>> import numpy as np
        >>> from spectral_connectivity import Connectivity, Multitaper
        >>> rng = np.random.default_rng(0)
        >>> leader = rng.standard_normal((1003, 20))
        >>> # Signal 1 is signal 0 delayed by 3 samples (6 ms at 500 Hz), plus noise.
        >>> signals = np.stack([leader[3:], leader[:-3]], axis=-1)  # (time, trials, signals)
        >>> signals += 0.5 * rng.standard_normal(signals.shape)
        >>> multitaper = Multitaper(signals, sampling_frequency=500)
        >>> connectivity = Connectivity.from_transform(multitaper)
        >>> granger = connectivity.pairwise_spectral_granger_prediction()
        >>> granger.shape  # (n_time_windows, n_frequencies, n_signals, n_signals)
        (1, 501, 2, 2)
        >>> # [..., i, j] is i -> j, so 0 -> 1 is [..., 0, 1]. It dominates at 10 Hz (bin 20).
        >>> bool(granger[0, 20, 0, 1] > 10 * granger[0, 20, 1, 0])
        True
        """
        return self._pairwise_spectral_granger(
            "pairwise_spectral_granger_prediction", time_reversed=False
        )

    def _pairwise_spectral_granger(
        self, measure: str, *, time_reversed: bool
    ) -> NDArray[np.floating]:
        """Pairwise spectral Granger, optionally of the time-reversed process.

        Time reversal of a real stationary process transposes its
        cross-spectral matrix.
        """
        self._require_two_sided_spectrum(measure)
        csm = self._expectation_cross_spectral_matrix()
        if time_reversed:
            csm = xp.swapaxes(csm, -1, -2)
        result, not_minimum_phase = _estimate_spectral_granger_prediction(
            self._power,
            csm,
            combinations(range(self.n_signals), 2),
            minimum_phase_tolerance=self._minimum_phase_tolerance,
            minimum_phase_max_iterations=self._minimum_phase_max_iterations,
        )
        _warn_granger_failures(result, not_minimum_phase, measure)
        return result

    @_orientation_changed
    @_source_first
    @_asnumpy
    @_factorization_explained_by_nonfinite_input
    def subset_pairwise_spectral_granger_prediction(
        self, pairs: Sequence[Sequence[int]] | NDArray[np.integer]
    ) -> NDArray[np.floating]:
        """Return predictive power for a subset of signal pairs.

        .. versionchanged:: 3.0
           The output is source first (``[..., i, j]`` is ``i -> j``); 2.x
           returned the transpose (``j -> i``).

        Parameters
        ----------
        pairs : array_like, shape (n_pairs, 2)
            Pairs of signal indices. Each pair is estimated in both directions.

        Returns
        -------
        pairwise_granger : array
            Shape ``(..., n_nonnegative_frequencies, n_signals, n_signals)``.
            Spectral Granger prediction for the specified pairs; entries for
            pairs not requested (and the diagonal) are NaN. Output ``[..., i, j]``
            is the influence of signal ``i`` on signal ``j`` (``i -> j``).

        Notes
        -----
        **Window and FFT length**: this measure comes from the Wilson
        factorization of the estimated spectrum. It needs a window longer
        than the slowest correlation in the data, and its values depend
        slightly on the FFT length (``n_fft_samples``): see the Notes of
        :func:`~spectral_connectivity.minimum_phase_decomposition.minimum_phase_decomposition`.

        Examples
        --------
        >>> import numpy as np
        >>> from spectral_connectivity import Connectivity, Multitaper
        >>> rng = np.random.default_rng(0)
        >>> leader = rng.standard_normal((1003, 20))
        >>> # Signal 1 is signal 0 delayed by 3 samples (6 ms at 500 Hz); signal 2 is noise.
        >>> signals = np.stack([leader[3:], leader[:-3], np.zeros((1000, 20))], axis=-1)
        >>> signals += 0.5 * rng.standard_normal(signals.shape)
        >>> multitaper = Multitaper(signals, sampling_frequency=500)
        >>> connectivity = Connectivity.from_transform(multitaper)
        >>> granger = connectivity.subset_pairwise_spectral_granger_prediction(pairs=[(0, 1)])
        >>> granger.shape  # (n_time_windows, n_frequencies, n_signals, n_signals)
        (1, 501, 3, 3)
        >>> # [..., i, j] is i -> j, so 0 -> 1 is [..., 0, 1]. It dominates at 10 Hz (bin 20).
        >>> bool(granger[0, 20, 0, 1] > 10 * granger[0, 20, 1, 0])
        True
        >>> bool(np.isnan(granger[0, 20, 0, 2]))  # pair (0, 2) was not requested
        True
        """
        self._require_two_sided_spectrum("subset_pairwise_spectral_granger_prediction")
        pairs = np.array(pairs)
        result, not_minimum_phase = _estimate_subset_spectral_granger_prediction(
            self._power,
            self._subset_cross_spectral_matrix(pairs),
            pairs,
            n_signals=self._fourier_coefficients.shape[-1],
            minimum_phase_tolerance=self._minimum_phase_tolerance,
            minimum_phase_max_iterations=self._minimum_phase_max_iterations,
        )
        requested = np.zeros((self.n_signals, self.n_signals), dtype=bool)
        requested[pairs[:, 0], pairs[:, 1]] = requested[pairs[:, 1], pairs[:, 0]] = True
        _warn_granger_failures(
            result,
            not_minimum_phase,
            "subset_pairwise_spectral_granger_prediction",
            requested=requested,
        )
        return result

    @_source_first
    @_asnumpy
    @_factorization_explained_by_nonfinite_input
    def time_reversed_spectral_granger_prediction(self) -> NDArray[np.floating]:
        """Return pairwise spectral Granger prediction after time reversal.

        For a real stationary process, time reversal transposes the
        cross-spectral matrix at every frequency. Contrasting this result with
        :meth:`pairwise_spectral_granger_prediction` helps identify apparent
        directionality caused by instantaneous mixing or data asymmetries.

        Returns
        -------
        time_reversed_granger : array
            Shape ``(..., n_nonnegative_frequencies, n_signals, n_signals)``.
            Output ``[..., i, j]`` is the influence of signal ``i`` on signal
            ``j`` (``i -> j``) in the time-reversed data.

        Notes
        -----
        **Non-negativity**: spectral Granger is ``>= 0`` by definition.
        Negative estimates within roundoff of zero -- above ``-100 * eps`` of
        the result dtype on the log-ratio scale, about ``-2e-14`` for float64
        -- are clipped to ``0``; materially negative bins below that threshold
        (a degenerate factorization) are returned as ``NaN``. Other
        packages (FieldTrip, MVGC, mne-connectivity) return such values as-is.

        **Window and FFT length**: this measure comes from the Wilson
        factorization of the estimated spectrum. It needs a window longer
        than the slowest correlation in the data, and its values depend
        slightly on the FFT length (``n_fft_samples``): see the Notes of
        :func:`~spectral_connectivity.minimum_phase_decomposition.minimum_phase_decomposition`.

        References
        ----------
        .. [1] Winkler, I., Panknin, D., Bartz, D., Müller, K.-R., and Haufe,
               S. (2016). Validity of time reversal for testing Granger
               causality. IEEE Transactions on Signal Processing 64, 2746-2760.

        Examples
        --------
        >>> import numpy as np
        >>> from spectral_connectivity import Connectivity, Multitaper
        >>> rng = np.random.default_rng(0)
        >>> leader = rng.standard_normal((1003, 20))
        >>> # Signal 1 is signal 0 delayed by 3 samples (6 ms at 500 Hz), plus noise.
        >>> signals = np.stack([leader[3:], leader[:-3]], axis=-1)  # (time, trials, signals)
        >>> signals += 0.5 * rng.standard_normal(signals.shape)
        >>> multitaper = Multitaper(signals, sampling_frequency=500)
        >>> connectivity = Connectivity.from_transform(multitaper)
        >>> reversed_granger = connectivity.time_reversed_spectral_granger_prediction()
        >>> reversed_granger.shape  # (n_time_windows, n_frequencies, n_signals, n_signals)
        (1, 501, 2, 2)
        >>> # A genuine 0 -> 1 lead flips under time reversal: 1 -> 0 ([..., 1, 0]) dominates.
        >>> bool(reversed_granger[0, 20, 1, 0] > 10 * reversed_granger[0, 20, 0, 1])
        True
        """
        return self._pairwise_spectral_granger(
            "time_reversed_spectral_granger_prediction", time_reversed=True
        )

    @_source_first
    @_asnumpy
    @_factorization_explained_by_nonfinite_input
    def conditional_spectral_granger_prediction(self) -> NDArray[np.floating]:
        """Return pairwise spectral Granger prediction conditioned on all others.

        For each ordered source-target pair, the influence of the source on the
        target is measured after accounting for every remaining signal, using
        the frequency-domain conditional measure of Chen, Bressler and Ding
        (2006): the full model containing every signal and the reduced model
        omitting the source are each spectrally factorized, and the reduced
        model's innovation spectrum for the target is split into the part
        explained by the target's own full-model innovations and the remainder
        attributable to the source. With two signals this reduces to ordinary
        pairwise spectral Granger.

        Returns
        -------
        conditional_granger : array
            Shape ``(..., n_nonnegative_frequencies, n_signals, n_signals)``.
            Output ``[..., i, j]`` is the influence of signal ``i`` on signal
            ``j`` (``i -> j``), conditional on every signal other than ``i``
            and ``j``. The diagonal is NaN.

        Notes
        -----
        **Non-negativity**: spectral Granger is ``>= 0`` by definition.
        Negative estimates within roundoff of zero -- above ``-100 * eps`` of
        the result dtype on the log-ratio scale, about ``-2e-14`` for float64
        -- are clipped to ``0``; materially negative bins below that threshold
        (a degenerate factorization) are returned as ``NaN``. Other
        packages (FieldTrip, MVGC, mne-connectivity) return such values as-is.

        **Range**: ``[0, ∞)``. The measure is a log-ratio of a total to an
        intrinsic innovation spectrum, so it is non-negative up to roundoff;
        bins where either spectrum is not positive (a degenerate factorization)
        are returned as NaN with a warning.

        **Cost**: ``n_signals + 1`` minimum-phase factorizations (the full
        system once, plus one ``(n_signals - 1)``-channel system per source),
        each shared by every target. The full-system factorization is cached
        and shared with the other directed measures.

        **Window and FFT length**: this measure comes from the Wilson
        factorization of the estimated spectrum. It needs a window longer
        than the slowest correlation in the data, and its values depend
        slightly on the FFT length (``n_fft_samples``): see the Notes of
        :func:`~spectral_connectivity.minimum_phase_decomposition.minimum_phase_decomposition`.

        References
        ----------
        .. [1] Chen, Y., Bressler, S.L., and Ding, M. (2006). Frequency
               decomposition of conditional Granger causality and application
               to multivariate neural field potential data. Journal of
               Neuroscience Methods 150, 228-237.
        .. [2] Geweke, J.F. (1984). Measures of conditional linear dependence
               and feedback between time series. Journal of the American
               Statistical Association 79, 907-915.

        Examples
        --------
        >>> import numpy as np
        >>> from spectral_connectivity import Connectivity, Multitaper
        >>> rng = np.random.default_rng(0)
        >>> # Chain 0 -> 1 -> 2: each signal is the previous one delayed 3 samples, plus noise.
        >>> source = rng.standard_normal((1006, 20))
        >>> middle = source[:-3] + 0.5 * rng.standard_normal((1003, 20))
        >>> target = middle[:-3] + 0.5 * rng.standard_normal((1000, 20))
        >>> signals = np.stack([source[6:], middle[3:], target], axis=-1)  # (time, trials, 3)
        >>> multitaper = Multitaper(signals, sampling_frequency=500)
        >>> connectivity = Connectivity.from_transform(multitaper)
        >>> conditional = connectivity.conditional_spectral_granger_prediction()
        >>> conditional.shape  # (n_time_windows, n_frequencies, n_signals, n_signals)
        (1, 501, 3, 3)
        >>> # [..., i, j] is i -> j. Pairwise Granger sees the indirect 0 -> 2 ([..., 0, 2]),
        >>> # but conditioning on signal 1 removes it while keeping 1 -> 2 (10 Hz = bin 20).
        >>> pairwise = connectivity.pairwise_spectral_granger_prediction()
        >>> bool(pairwise[0, 20, 0, 2] > 0.5), bool(conditional[0, 20, 0, 2] < 0.05)
        (True, True)
        >>> bool(conditional[0, 20, 1, 2] > 0.5)
        True
        """
        return self._conditional_spectral_granger(
            "conditional_spectral_granger_prediction", time_reversed=False
        )

    def _conditional_spectral_granger(
        self, measure: str, *, time_reversed: bool
    ) -> NDArray[np.floating]:
        """Conditional spectral Granger, optionally of the time-reversed process.

        Time reversal of a real stationary process transposes its
        cross-spectral matrix.
        """
        self._require_two_sided_spectrum(measure)
        spectrum = self._expectation_cross_spectral_matrix()
        if time_reversed:
            spectrum = xp.swapaxes(spectrum, -1, -2)
        tolerance = self._minimum_phase_tolerance
        max_iterations = self._minimum_phase_max_iterations
        if self.n_signals == 2:
            # No conditioning set: the measure is pairwise Geweke Granger.
            result, not_minimum_phase = _estimate_blockwise_spectral_granger(
                spectrum,
                [np.array([0]), np.array([1])],
                minimum_phase_tolerance=tolerance,
                minimum_phase_max_iterations=max_iterations,
            )
        else:
            # The full model is factored with its own warnings: the forward one
            # is the instance's cached factorization (the same CSM, tolerance
            # and iteration cap), shared with the other directed measures; the
            # reversed one goes through the same path. The pair warnings below
            # name the failures of the reduced models.
            transfer_function, noise_covariance = (
                _var_model_from_factor(
                    _factorize_spectrum(
                        spectrum,
                        minimum_phase_tolerance=tolerance,
                        minimum_phase_max_iterations=max_iterations,
                    )
                )
                if time_reversed
                else (self._transfer_function, self._noise_covariance)
            )
            result, not_minimum_phase = _estimate_all_conditional_spectral_granger(
                spectrum,
                transfer_function,
                noise_covariance,
                minimum_phase_tolerance=tolerance,
                minimum_phase_max_iterations=max_iterations,
            )
        _warn_granger_failures(result, not_minimum_phase, measure)
        return result

    @_source_first
    @_asnumpy
    @_factorization_explained_by_nonfinite_input
    def time_reversed_conditional_spectral_granger_prediction(self) -> NDArray[np.floating]:
        """Return conditional spectral Granger after time reversal.

        :meth:`conditional_spectral_granger_prediction` of the time-reversed
        data. For a real stationary process, time reversal transposes the
        cross-spectral matrix at every frequency. Genuine directed influence
        reverses under time reversal; contrasting this result with
        :meth:`conditional_spectral_granger_prediction` helps identify
        apparent directionality caused by instantaneous mixing, as
        :meth:`time_reversed_spectral_granger_prediction` does for pairwise
        Granger.

        Returns
        -------
        time_reversed_conditional_granger : array
            Shape ``(..., n_nonnegative_frequencies, n_signals, n_signals)``.
            Output ``[..., i, j]`` is the influence of signal ``i`` on signal
            ``j`` (``i -> j``) in the time-reversed data, conditional on every
            signal other than ``i`` and ``j``. The diagonal is NaN.

        Notes
        -----
        **Non-negativity and range**: as for
        :meth:`conditional_spectral_granger_prediction`, ``[0, ∞)``; roundoff
        negatives are clipped to ``0`` and materially negative bins are NaN.

        **Cost**: ``n_signals + 1`` minimum-phase factorizations, as for the
        forward measure, except that the full-system one is not cached or
        shared with the other directed measures: every call factors the
        reversed full system again.

        **Window and FFT length**: this measure comes from the Wilson
        factorization of the estimated spectrum. It needs a window longer
        than the slowest correlation in the data, and its values depend
        slightly on the FFT length (``n_fft_samples``): see the Notes of
        :func:`~spectral_connectivity.minimum_phase_decomposition.minimum_phase_decomposition`.

        References
        ----------
        .. [1] Winkler, I., Panknin, D., Bartz, D., Müller, K.-R., and Haufe,
               S. (2016). Validity of time reversal for testing Granger
               causality. IEEE Transactions on Signal Processing 64, 2746-2760.
        .. [2] Chen, Y., Bressler, S.L., and Ding, M. (2006). Frequency
               decomposition of conditional Granger causality and application
               to multivariate neural field potential data. Journal of
               Neuroscience Methods 150, 228-237.

        Examples
        --------
        >>> import numpy as np
        >>> from spectral_connectivity import Connectivity, Multitaper
        >>> rng = np.random.default_rng(0)
        >>> # Chain 0 -> 1 -> 2: each signal is the previous one delayed 3 samples, plus noise.
        >>> source = rng.standard_normal((1006, 20))
        >>> middle = source[:-3] + 0.5 * rng.standard_normal((1003, 20))
        >>> target = middle[:-3] + 0.5 * rng.standard_normal((1000, 20))
        >>> signals = np.stack([source[6:], middle[3:], target], axis=-1)  # (time, trials, 3)
        >>> multitaper = Multitaper(signals, sampling_frequency=500)
        >>> connectivity = Connectivity.from_transform(multitaper)
        >>> reversed_granger = connectivity.time_reversed_conditional_spectral_granger_prediction()
        >>> reversed_granger.shape  # (n_time_windows, n_frequencies, n_signals, n_signals)
        (1, 501, 3, 3)
        >>> # The direct link 1 -> 2 flips under time reversal: 2 -> 1 ([..., 2, 1]) dominates.
        >>> bool(reversed_granger[0, 20, 2, 1] > 10 * reversed_granger[0, 20, 1, 2])
        True
        """
        return self._conditional_spectral_granger(
            "time_reversed_conditional_spectral_granger_prediction", time_reversed=True
        )

    @_factorization_explained_by_nonfinite_input
    def blockwise_spectral_granger_prediction(
        self, group_labels: NDArray[np.integer]
    ) -> tuple[NDArray[np.floating], NDArray[np.integer]]:
        """Return spectral Granger prediction between multichannel groups.

        Parameters
        ----------
        group_labels : array-like, shape (n_signals,)
            Label assigning each signal to one non-overlapping group.

        Returns
        -------
        blockwise_granger : array
            Shape ``(..., n_nonnegative_frequencies, n_groups, n_groups)``.
            Output ``[..., i, j]`` is the influence of group ``i`` on group ``j``
            (``i -> j``), with groups ordered as in ``labels``. The diagonal is
            NaN.
        labels : array, shape (n_groups,)
            Sorted unique group labels.

        Notes
        -----
        **Window and FFT length**: this measure comes from the Wilson
        factorization of the estimated spectrum. It needs a window longer
        than the slowest correlation in the data, and its values depend
        slightly on the FFT length (``n_fft_samples``): see the Notes of
        :func:`~spectral_connectivity.minimum_phase_decomposition.minimum_phase_decomposition`.

        Examples
        --------
        >>> import numpy as np
        >>> from spectral_connectivity import Connectivity, Multitaper
        >>> rng = np.random.default_rng(0)
        >>> leader = rng.standard_normal((1003, 20))
        >>> # Group "a": two noisy copies of signal 0; "b": two of its 6 ms-delayed copy.
        >>> pair = np.stack([leader[3:], leader[:-3]], axis=-1)
        >>> signals = np.repeat(pair, 2, axis=-1)  # (time, trials, 4 signals)
        >>> signals += 0.5 * rng.standard_normal(signals.shape)
        >>> multitaper = Multitaper(signals, sampling_frequency=500)
        >>> connectivity = Connectivity.from_transform(multitaper)
        >>> granger, labels = connectivity.blockwise_spectral_granger_prediction(
        ...     ["a", "a", "b", "b"]
        ... )
        >>> granger.shape  # (n_time_windows, n_frequencies, n_groups, n_groups)
        (1, 501, 2, 2)
        >>> labels
        array(['a', 'b'], dtype='<U1')
        >>> # [..., i, j] is group i -> group j, so a -> b is [..., 0, 1] (10 Hz = bin 20).
        >>> bool(granger[0, 20, 0, 1] > 10 * granger[0, 20, 1, 0])
        True
        """
        return self._blockwise_spectral_granger(
            "blockwise_spectral_granger_prediction", group_labels, time_reversed=False
        )

    def _blockwise_spectral_granger(
        self, measure: str, group_labels: NDArray[np.integer], *, time_reversed: bool
    ) -> tuple[NDArray[np.floating], NDArray[np.integer]]:
        """Blockwise spectral Granger, optionally of the time-reversed process.

        Time reversal of a real stationary process transposes its
        cross-spectral matrix.
        """
        self._require_two_sided_spectrum(measure)
        labels, indices, _ = self._validated_group_indices(group_labels)
        spectrum = self._expectation_cross_spectral_matrix()
        if time_reversed:
            spectrum = xp.swapaxes(spectrum, -1, -2)
        result, not_minimum_phase = _estimate_blockwise_spectral_granger(
            spectrum,
            indices,
            minimum_phase_tolerance=self._minimum_phase_tolerance,
            minimum_phase_max_iterations=self._minimum_phase_max_iterations,
        )
        _warn_granger_failures(result, not_minimum_phase, measure, names=to_numpy(labels))
        # Swap on the host, like _source_first: [..., target, source] -> source first.
        return np.swapaxes(to_numpy(result), -1, -2), to_numpy(labels)

    @_factorization_explained_by_nonfinite_input
    def time_reversed_blockwise_spectral_granger_prediction(
        self, group_labels: NDArray[np.integer]
    ) -> tuple[NDArray[np.floating], NDArray[np.integer]]:
        """Return blockwise spectral Granger after time reversal.

        :meth:`blockwise_spectral_granger_prediction` of the time-reversed
        data. For a real stationary process, time reversal transposes the
        cross-spectral matrix at every frequency. Genuine directed influence
        between groups reverses under time reversal; contrasting this result
        with :meth:`blockwise_spectral_granger_prediction` helps identify
        apparent directionality caused by instantaneous mixing, as
        :meth:`time_reversed_spectral_granger_prediction` does for single
        signals.

        Parameters
        ----------
        group_labels : array-like, shape (n_signals,)
            Label assigning each signal to one non-overlapping group.

        Returns
        -------
        time_reversed_blockwise_granger : array
            Shape ``(..., n_nonnegative_frequencies, n_groups, n_groups)``.
            Output ``[..., i, j]`` is the influence of group ``i`` on group ``j``
            (``i -> j``) in the time-reversed data, with groups ordered as in
            ``labels``. The diagonal is NaN.
        labels : array, shape (n_groups,)
            Sorted unique group labels.

        Notes
        -----
        **Window and FFT length**: this measure comes from the Wilson
        factorization of the estimated spectrum. It needs a window longer
        than the slowest correlation in the data, and its values depend
        slightly on the FFT length (``n_fft_samples``): see the Notes of
        :func:`~spectral_connectivity.minimum_phase_decomposition.minimum_phase_decomposition`.

        References
        ----------
        .. [1] Winkler, I., Panknin, D., Bartz, D., Müller, K.-R., and Haufe,
               S. (2016). Validity of time reversal for testing Granger
               causality. IEEE Transactions on Signal Processing 64, 2746-2760.

        Examples
        --------
        >>> import numpy as np
        >>> from spectral_connectivity import Connectivity, Multitaper
        >>> rng = np.random.default_rng(0)
        >>> leader = rng.standard_normal((1003, 20))
        >>> # Group "a": two noisy copies of signal 0; "b": two of its 6 ms-delayed copy.
        >>> pair = np.stack([leader[3:], leader[:-3]], axis=-1)
        >>> signals = np.repeat(pair, 2, axis=-1)  # (time, trials, 4 signals)
        >>> signals += 0.5 * rng.standard_normal(signals.shape)
        >>> multitaper = Multitaper(signals, sampling_frequency=500)
        >>> connectivity = Connectivity.from_transform(multitaper)
        >>> reversed_granger, labels = (
        ...     connectivity.time_reversed_blockwise_spectral_granger_prediction(["a", "a", "b", "b"])
        ... )
        >>> reversed_granger.shape  # (n_time_windows, n_frequencies, n_groups, n_groups)
        (1, 501, 2, 2)
        >>> # The lead a -> b flips under time reversal: b -> a ([..., 1, 0]) dominates.
        >>> bool(reversed_granger[0, 20, 1, 0] > 10 * reversed_granger[0, 20, 0, 1])
        True
        """
        return self._blockwise_spectral_granger(
            "time_reversed_blockwise_spectral_granger_prediction",
            group_labels,
            time_reversed=True,
        )

    @_orientation_changed
    @_ignore_nan_propagation_warnings
    @_source_first
    @_asnumpy
    @_factorization_explained_by_nonfinite_input
    def directed_transfer_function(self) -> NDArray[np.floating]:
        """Return transfer function coupling strength normalized by inflow.

        The transfer function coupling strength normalized by the total
        influence of other signals on that signal (inflow).

        Characterizes the direct and indirect coupling to a node.

        .. versionchanged:: 3.0
           The output is source first (``[..., i, j]`` is ``i -> j``); 2.x
           returned the transpose (``j -> i``).

        Returns
        -------
        directed_transfer_function : array
            Shape ``(..., n_nonnegative_frequencies, n_signals, n_signals)``.
            Directed transfer function values. Output ``[..., i, j]`` is the
            influence of signal ``i`` on signal ``j`` (``i -> j``).

        Notes
        -----
        **Range**: [0, 1] (normalized). Represents proportion of inflow
        via transfer function; each target's values sum to 1 over sources
        (``result.sum(axis=-2)`` is 1).

        **Window and FFT length**: this measure comes from the Wilson
        factorization of the estimated spectrum. It needs a window longer
        than the slowest correlation in the data, and its values depend
        slightly on the FFT length (``n_fft_samples``): see the Notes of
        :func:`~spectral_connectivity.minimum_phase_decomposition.minimum_phase_decomposition`.

        References
        ----------
        .. [1] Kaminski, M., and Blinowska, K.J. (1991). A new method of
               the description of the information flow in the brain
               structures. Biological Cybernetics 65, 203-210.

        Examples
        --------
        >>> import numpy as np
        >>> from spectral_connectivity import Connectivity, Multitaper
        >>> rng = np.random.default_rng(0)
        >>> leader = rng.standard_normal((1003, 20))
        >>> # Signal 1 is signal 0 delayed by 3 samples (6 ms at 500 Hz), plus noise.
        >>> signals = np.stack([leader[3:], leader[:-3]], axis=-1)  # (time, trials, signals)
        >>> signals += 0.5 * rng.standard_normal(signals.shape)
        >>> multitaper = Multitaper(signals, sampling_frequency=500)
        >>> connectivity = Connectivity.from_transform(multitaper)
        >>> dtf = connectivity.directed_transfer_function()
        >>> dtf.shape  # (n_time_windows, n_frequencies, n_signals, n_signals)
        (1, 501, 2, 2)
        >>> # [..., i, j] is i -> j, so 0 -> 1 is [..., 0, 1]. It dominates at 10 Hz (bin 20).
        >>> bool(dtf[0, 20, 0, 1] > 10 * dtf[0, 20, 1, 0])
        True
        >>> bool(np.allclose(dtf.sum(axis=-2), 1))  # each target's inflow sums to 1
        True
        """
        return _squared_magnitude(
            self._transfer_function / _total_inflow(self._transfer_function)
        )

    @_orientation_changed
    @_ignore_nan_propagation_warnings
    @_source_first
    @_asnumpy
    @_factorization_explained_by_nonfinite_input
    def directed_coherence(self) -> NDArray[np.floating]:
        """Return the squared directed coherence (noise-weighted DTF).

        Like the directed transfer function, but the noise variance weights
        **both** the numerator and the inflow normalization. The returned value
        is the squared directed coherence ``nv_j |H_ij|^2 / sum_k nv_k |H_ik|^2``,
        written in the transfer function's native ``[target, source]`` indexing
        (``H_ij`` is ``j -> i``), where ``nv`` is the per-signal innovation
        (noise) variance and ``H`` is the transfer function. Each target's values
        sum to 1 over sources (``result.sum(axis=-2)`` is 1).

        .. versionchanged:: 3.0
           The output is source first (``[..., i, j]`` is ``i -> j``); 2.x
           returned the transpose (``j -> i``).

        Returns
        -------
        directed_coherence : array
            Shape ``(..., n_nonnegative_frequencies, n_signals, n_signals)``.
            Squared directed coherence values. Output ``[..., i, j]`` is the
            influence of signal ``i`` on signal ``j`` (``i -> j``).

        Notes
        -----
        **Range**: [0, 1]. Normalized directional connectivity measure.

        **Assumption**: This measure follows Baccala et al. (1998), which
        assumes the MVAR innovations are uncorrelated (a diagonal noise
        covariance). The denominator then equals the signal's power spectral
        density ``S_ii = sum_k nv_k |H_ik|^2``. When the estimated innovation
        covariance has non-negligible off-diagonal terms (common for
        non-parametrically estimated MVARs), the true PSD
        ``S_ii = (H Cov H^H)_ii`` also contains cross-power between correlated
        sources that this diagonal formula omits, so the values are approximate.
        A ``UserWarning`` is emitted in that case.

        **Window and FFT length**: this measure comes from the Wilson
        factorization of the estimated spectrum. It needs a window longer
        than the slowest correlation in the data, and its values depend
        slightly on the FFT length (``n_fft_samples``): see the Notes of
        :func:`~spectral_connectivity.minimum_phase_decomposition.minimum_phase_decomposition`.

        References
        ----------
        .. [1] Baccala, L., Sameshima, K., Ballester, G., Do Valle, A., and
               Timo-Iaria, C. (1998). Studying the interaction between
               brain structures via directed coherence and Granger
               causality. Applied Signal Processing 5, 40.

        Examples
        --------
        >>> import numpy as np
        >>> from spectral_connectivity import Connectivity, Multitaper
        >>> rng = np.random.default_rng(0)
        >>> leader = rng.standard_normal((1003, 20))
        >>> # Signal 1 is signal 0 delayed by 3 samples (6 ms at 500 Hz), plus noise.
        >>> signals = np.stack([leader[3:], leader[:-3]], axis=-1)  # (time, trials, signals)
        >>> signals += 0.5 * rng.standard_normal(signals.shape)
        >>> multitaper = Multitaper(signals, sampling_frequency=500)
        >>> connectivity = Connectivity.from_transform(multitaper)
        >>> directed_coherence = connectivity.directed_coherence()
        >>> directed_coherence.shape  # (n_time_windows, n_frequencies, n_signals, n_signals)
        (1, 501, 2, 2)
        >>> # [..., i, j] is i -> j, so 0 -> 1 is [..., 0, 1]. It dominates at 10 Hz (bin 20).
        >>> bool(directed_coherence[0, 20, 0, 1] > 10 * directed_coherence[0, 20, 1, 0])
        True
        >>> bool(np.allclose(directed_coherence.sum(axis=-2), 1))  # each target's inflow
        True
        """
        # Directed coherence normalizes the noise-weighted inflow over sources
        # (native axis -1), so the per-source noise variance must vary along that axis.
        # The squared measure is nv_j |H_ij|^2 / sum_k nv_k |H_ik|^2, which sums
        # to 1 over sources like the directed transfer function. This uses only
        # the diagonal of the noise covariance, which equals the PSD denominator
        # only when the innovations are uncorrelated; warn when the omitted
        # cross-power is a material fraction of the true PSD.
        if (
            _max_psd_discrepancy(self._transfer_function, self._noise_covariance)
            > DIRECTED_COHERENCE_DISCREPANCY_TOLERANCE
        ):
            warnings.warn(
                "directed_coherence assumes uncorrelated MVAR innovations (a "
                "diagonal noise covariance), but the estimated innovation "
                "covariance has non-negligible off-diagonal terms: the diagonal "
                "normalization omits a material fraction of the true power "
                "spectral density (cross-power between correlated sources). "
                "Interpret the values as approximate, or use "
                "partial_directed_coherence instead.",
                UserWarning,
                stacklevel=stacklevel_outside_package(),
            )
        noise_variance = _get_noise_variance(self._noise_covariance, axis=-1)
        directed_coherence: NDArray[np.floating] = (
            noise_variance
            * _squared_magnitude(self._transfer_function)
            / _total_inflow(self._transfer_function, noise_variance) ** 2
        )
        return directed_coherence

    def _partial_directed_coherence(self) -> NDArray[np.floating]:
        """Return device-native PDC for reuse by other device-native measures."""
        return _squared_magnitude(
            self._MVAR_Fourier_coefficients / _total_outflow(self._MVAR_Fourier_coefficients)
        )

    @_orientation_changed
    @_ignore_nan_propagation_warnings
    @_source_first
    @_asnumpy
    @_factorization_explained_by_nonfinite_input
    def partial_directed_coherence(self) -> NDArray[np.floating]:
        """Return transfer function coupling strength normalized by outflow.

        The transfer function coupling strength normalized by its
        strength of coupling to other signals (outflow).

        The partial directed coherence tries to regress out the influence
        of other observed signals, leaving only the direct coupling between
        two signals.

        .. versionchanged:: 3.0
           The output is source first (``[..., i, j]`` is ``i -> j``); 2.x
           returned the transpose (``j -> i``).

        Returns
        -------
        partial_directed_coherence : array
            Shape ``(..., n_nonnegative_frequencies, n_signals, n_signals)``.
            Partial directed coherence values. Output ``[..., i, j]`` is the
            influence of signal ``i`` on signal ``j`` (``i -> j``).

        Notes
        -----
        **Range**: [0, 1]. Normalized direct coupling measure; each source's
        values sum to 1 over targets (``result.sum(axis=-1)`` is 1).

        **Window and FFT length**: this measure comes from the Wilson
        factorization of the estimated spectrum. It needs a window longer
        than the slowest correlation in the data, and its values depend
        slightly on the FFT length (``n_fft_samples``): see the Notes of
        :func:`~spectral_connectivity.minimum_phase_decomposition.minimum_phase_decomposition`.

        References
        ----------
        .. [1] Baccala, L.A., and Sameshima, K. (2001). Partial directed
               coherence: a new concept in neural structure determination.
               Biological Cybernetics 84, 463-474.

        Examples
        --------
        >>> import numpy as np
        >>> from spectral_connectivity import Connectivity, Multitaper
        >>> rng = np.random.default_rng(0)
        >>> leader = rng.standard_normal((1003, 20))
        >>> # Signal 1 is signal 0 delayed by 3 samples (6 ms at 500 Hz), plus noise.
        >>> signals = np.stack([leader[3:], leader[:-3]], axis=-1)  # (time, trials, signals)
        >>> signals += 0.5 * rng.standard_normal(signals.shape)
        >>> multitaper = Multitaper(signals, sampling_frequency=500)
        >>> connectivity = Connectivity.from_transform(multitaper)
        >>> pdc = connectivity.partial_directed_coherence()
        >>> pdc.shape  # (n_time_windows, n_frequencies, n_signals, n_signals)
        (1, 501, 2, 2)
        >>> # [..., i, j] is i -> j, so 0 -> 1 is [..., 0, 1]. It dominates at 10 Hz (bin 20).
        >>> bool(pdc[0, 20, 0, 1] > 10 * pdc[0, 20, 1, 0])
        True
        >>> bool(np.allclose(pdc.sum(axis=-1), 1))  # each source's outflow sums to 1
        True
        """
        return self._partial_directed_coherence()

    @_orientation_changed
    @_ignore_nan_propagation_warnings
    @_source_first
    @_asnumpy
    @_factorization_explained_by_nonfinite_input
    def generalized_partial_directed_coherence(self) -> NDArray[np.floating]:
        """Return generalized partial directed coherence.

        The transfer function coupling strength normalized by its
        strength of coupling to other signals (outflow).

        The partial directed coherence tries to regress out the influence
        of other observed signals, leaving only the direct coupling between
        two signals.

        The generalized partial directed coherence scales the relative
        strength of coupling by the noise variance.

        .. versionchanged:: 3.0
           The output is source first (``[..., i, j]`` is ``i -> j``); 2.x
           returned the transpose (``j -> i``).

        Returns
        -------
        generalized_partial_directed_coherence : array
            Shape ``(..., n_nonnegative_frequencies, n_signals, n_signals)``.
            Generalized partial directed coherence values. Output ``[..., i, j]``
            is the influence of signal ``i`` on signal ``j`` (``i -> j``).

        Notes
        -----
        **Range**: [0, 1]. Normalized, scaled by noise variance; each source's
        values sum to 1 over targets (``result.sum(axis=-1)`` is 1).

        **Window and FFT length**: this measure comes from the Wilson
        factorization of the estimated spectrum. It needs a window longer
        than the slowest correlation in the data, and its values depend
        slightly on the FFT length (``n_fft_samples``): see the Notes of
        :func:`~spectral_connectivity.minimum_phase_decomposition.minimum_phase_decomposition`.

        References
        ----------
        .. [1] Baccala, L.A., Sameshima, K., and Takahashi, D.Y. (2007).
               Generalized partial directed coherence. In Digital Signal
               Processing, 2007 15th International Conference on, (IEEE),
               pp. 163-166.

        Examples
        --------
        >>> import numpy as np
        >>> from spectral_connectivity import Connectivity, Multitaper
        >>> rng = np.random.default_rng(0)
        >>> leader = rng.standard_normal((1003, 20))
        >>> # Signal 1 is signal 0 delayed by 3 samples (6 ms at 500 Hz), plus noise.
        >>> signals = np.stack([leader[3:], leader[:-3]], axis=-1)  # (time, trials, signals)
        >>> signals += 0.5 * rng.standard_normal(signals.shape)
        >>> multitaper = Multitaper(signals, sampling_frequency=500)
        >>> connectivity = Connectivity.from_transform(multitaper)
        >>> gpdc = connectivity.generalized_partial_directed_coherence()
        >>> gpdc.shape  # (n_time_windows, n_frequencies, n_signals, n_signals)
        (1, 501, 2, 2)
        >>> # [..., i, j] is i -> j, so 0 -> 1 is [..., 0, 1]. It dominates at 10 Hz (bin 20).
        >>> bool(gpdc[0, 20, 0, 1] > 10 * gpdc[0, 20, 1, 0])
        True
        >>> bool(np.allclose(gpdc.sum(axis=-1), 1))  # each source's outflow sums to 1
        True
        """
        noise_variance = _get_noise_variance(self._noise_covariance)
        return _squared_magnitude(
            self._MVAR_Fourier_coefficients
            / xp.sqrt(noise_variance)
            / _total_outflow(self._MVAR_Fourier_coefficients, noise_variance)
        )

    @_orientation_changed
    @_ignore_nan_propagation_warnings
    @_source_first
    @_asnumpy
    @_factorization_explained_by_nonfinite_input
    def direct_directed_transfer_function(self) -> NDArray[np.floating]:
        """Return the direct directed transfer function (dDTF).

        The squared dDTF of Korzeniewska et al. (2003) multiplies the
        full-frequency DTF, which keeps direct and indirect (cascade) influence,
        by the squared partial coherence, which is zero between two signals
        whose relation is fully explained by the others. The product is
        therefore zero for an influence relayed through observed signals, but a
        positive value does not establish a direct one: partial coherence is
        also nonzero between two signals that drive a common observed signal.
        In the VAR ``0 -> 1 -> 2`` with ``0 -> 3 <- 2`` (unit lag-one couplings,
        identity innovations) there is no ``0 -> 2`` coefficient, and
        :meth:`partial_directed_coherence` and
        :meth:`conditional_spectral_granger_prediction` are zero for it, yet
        the dDTF ``0 -> 2`` is positive at every frequency and sums to 1/18. Use
        those measures to test for a direct link:

        ``chi^2_ij(f) = ffDTF^2_ij(f) * kappa^2_ij(f)``, with
        ``ffDTF^2_ij(f) = |H_ij(f)|^2 / sum_f' sum_k |H_ik(f')|^2`` (summed over
        the non-negative frequencies) and
        ``kappa^2_ij(f) = |G_ij(f)|^2 / (G_ii(f) G_jj(f))``, where
        ``G = A^H Sigma^-1 A`` is the inverse spectral matrix of the MVAR model
        (``A = H^-1``, ``Sigma`` the innovation covariance). These formulas are
        written in the transfer function's native ``[target, source]`` indexing
        (``chi^2_ij`` is ``j -> i``).

        .. versionchanged:: 3.0
           The output is source first (``[..., i, j]`` is ``i -> j``); 2.x
           returned the transpose (``j -> i``).

        Returns
        -------
        direct_directed_transfer_function : array
            Shape ``(..., n_nonnegative_frequencies, n_signals, n_signals)``.
            Output ``[..., i, j]`` is the dDTF ``i -> j``: zero for influence
            relayed through observed signals, and positive for direct
            influence or for two signals with a common observed child.

        Notes
        -----
        **Range**: [0, 1]. Like :meth:`directed_transfer_function`, this
        returns the squared quantity; its square root is the amplitude form
        ``|ffDTF| * |kappa|``. The diagonal is the full-frequency DTF of
        each signal with itself (``kappa_ii = 1``).

        **Window and FFT length**: this measure comes from the Wilson
        factorization of the estimated spectrum. It needs a window longer
        than the slowest correlation in the data, and its values depend
        slightly on the FFT length (``n_fft_samples``): see the Notes of
        :func:`~spectral_connectivity.minimum_phase_decomposition.minimum_phase_decomposition`.

        References
        ----------
        .. [1] Korzeniewska, A., Manczak, M., Kaminski,
               M., Blinowska, K.J., and Kasicki, S. (2003). Determination
               of information flow direction among brain structures by a
               modified directed transfer function (dDTF) method.
               Journal of Neuroscience Methods 125, 195-207.

        Examples
        --------
        >>> import numpy as np
        >>> from spectral_connectivity import Connectivity, Multitaper
        >>> rng = np.random.default_rng(0)
        >>> # Chain 0 -> 1 -> 2: each signal is the previous one delayed 3 samples, plus noise.
        >>> source = rng.standard_normal((1006, 20))
        >>> middle = source[:-3] + 0.5 * rng.standard_normal((1003, 20))
        >>> target = middle[:-3] + 0.5 * rng.standard_normal((1000, 20))
        >>> signals = np.stack([source[6:], middle[3:], target], axis=-1)  # (time, trials, 3)
        >>> multitaper = Multitaper(signals, sampling_frequency=500)
        >>> connectivity = Connectivity.from_transform(multitaper)
        >>> ddtf = connectivity.direct_directed_transfer_function()
        >>> ddtf.shape  # (n_time_windows, n_frequencies, n_signals, n_signals)
        (1, 501, 3, 3)
        >>> # [..., i, j] is i -> j. The direct 1 -> 2 ([..., 1, 2]) far exceeds both the
        >>> # reverse 2 -> 1 ([..., 2, 1]) and the indirect 0 -> 2 ([..., 0, 2]) that is
        >>> # relayed through signal 1 (10 Hz = bin 20).
        >>> direct = ddtf[0, 20, 1, 2]
        >>> bool(direct > 10 * ddtf[0, 20, 2, 1]), bool(direct > 10 * ddtf[0, 20, 0, 2])
        (True, True)
        """
        full_frequency_dtf = _squared_magnitude(
            self._transfer_function / _total_inflow(self._transfer_function, axis=(-1, -3))
        )
        mvar_coefficients = self._MVAR_Fourier_coefficients
        inverse_noise_covariance = _regularized_inverse(
            self._noise_covariance, per_signal=True
        )
        inverse_spectrum = xp.matmul(
            xp.matmul(
                _conjugate_transpose(mvar_coefficients),
                inverse_noise_covariance[..., xp.newaxis, :, :],
            ),
            mvar_coefficients,
        )
        inverse_spectrum_diagonal = xp.real(xp.diagonal(inverse_spectrum, axis1=-2, axis2=-1))
        squared_partial_coherence: NDArray[np.floating] = _squared_magnitude(
            inverse_spectrum
        ) / (
            inverse_spectrum_diagonal[..., :, xp.newaxis]
            * inverse_spectrum_diagonal[..., xp.newaxis, :]
        )
        return full_frequency_dtf * squared_partial_coherence

    def _selected_frequencies(
        self,
        measure: str,
        frequency_range: tuple[float, float] | None,
        frequencies_of_interest: Sequence[float] | NDArray[np.floating] | None,
    ) -> NDArray[np.bool_]:
        """Mask of the bins ``measure`` consumes; warns for the deprecated keyword.

        Returns
        -------
        mask : NDArray[bool], shape (n_frequencies,)
            Host mask over :attr:`frequencies`.
        """
        selection = _select_measure_frequencies(
            self.frequencies, frequency_range, frequencies_of_interest
        )
        if selection.deprecated:
            assert selection.requested is not None  # the deprecated band
            low, high = selection.requested
            warnings.warn(
                f"{measure}(frequencies_of_interest=...) is deprecated and will be "
                "removed in 4.0; use frequency_range=(low, high). "
                "frequencies_of_interest excludes a bin lying on either edge, while "
                "frequency_range includes both endpoints. To keep the same "
                f"{int(selection.mask.sum())} bins, replace frequencies_of_interest=("
                f"{low!r}, {high!r}) with frequency_range=({selection.lowest!r}, "
                f"{selection.highest!r}), the lowest and highest bins it selects. "
                "The wrappers (multitaper_connectivity, fourier_connectivity, and "
                "connectivity_to_xarray) take frequency_range as their own argument, "
                "not in connectivity_kwargs.",
                DeprecationWarning,
                stacklevel=stacklevel_outside_package(),
            )
        return selection.mask

    @property
    def real_valued_frequencies(self) -> NDArray[np.bool_]:
        """Which of :attr:`frequencies` hold only real Fourier coefficients.

        For a real signal these are 0 Hz and, for an even FFT length, Nyquist;
        an odd FFT length has no Nyquist bin. Each observation there carries
        one real degree of freedom rather than two, so the zero-coherence null
        and the chi-squared degrees of freedom of power differ: pass this mask
        as ``real_valued`` to
        :func:`~spectral_connectivity.statistics.coherence_significance_pvalue`,
        :func:`~spectral_connectivity.statistics.power_confidence_intervals`,
        and the log-power statistics. :meth:`group_delay` and :meth:`delay`
        apply it themselves.

        A bin counts as real when every coefficient's imaginary part is exactly
        0, ignoring NaN coefficients (a bin with no finite coefficient does not
        count). The package's transforms give exactly
        real coefficients there on the CPU and the GPU alike; coefficients
        computed elsewhere may carry rounding in those bins and then do not
        count.

        Returns
        -------
        NDArray[bool], shape (n_frequencies,)
            Host array aligned with :attr:`frequencies`.

        Examples
        --------
        >>> import numpy as np
        >>> from spectral_connectivity import Connectivity, Multitaper
        >>> rng = np.random.default_rng(0)
        >>> multitaper = Multitaper(rng.standard_normal((200, 3, 2)), sampling_frequency=100)
        >>> connectivity = Connectivity.from_transform(multitaper)
        >>> connectivity.frequencies[connectivity.real_valued_frequencies].tolist()
        [0.0, 50.0]
        """
        return self._real_valued_frequency_mask.copy()

    @cached_property
    def _real_valued_frequency_mask(self) -> NDArray[np.bool_]:
        """:attr:`real_valued_frequencies`, scanned once per set of coefficients."""
        coefficients = self._nonnegative_fourier_coefficients(self._fourier_coefficients.dtype)
        imaginary = coefficients.imag
        # NaN in either part: np.nan assigned to a complex array is nan + 0j.
        is_nan = xp.isnan(coefficients)
        axes = (0, 1, 2, 4)
        # A bin with no finite coefficient gives no evidence either way: not real.
        real: NDArray[np.bool_] = to_numpy(
            xp.all((imaginary == 0) | is_nan, axis=axes) & ~xp.all(is_nan, axis=axes)
        )
        return real

    def _significant_pair_phase(
        self,
        measure: str,
        frequency_mask: NDArray[np.bool_],
        frequency_resolution: float | None,
        significance_threshold: float,
    ) -> tuple[np.ma.MaskedArray, NDArray[np.floating], NDArray[np.intp], int]:
        """Unwrapped coherency phase per signal pair, masked where not significant.

        Shared setup of :meth:`group_delay` and :meth:`delay`: validates the
        frequency grid and observation weights, keeps the coherency at the bins
        of ``frequency_mask`` (see :meth:`_selected_frequencies`),
        gathers the upper-triangle signal pairs, and masks frequencies whose
        coherence is not significant.

        Returns
        -------
        coherence_phase : masked array, shape (..., n_band_frequencies, n_pairs)
            Phase unwrapped along frequency; masked where not significant.
        bandpassed_frequencies : array, shape (n_band_frequencies,)
        signal_combination_ind : array of int, shape (n_pairs, 2)
            ``(i, j)`` with ``i < j`` for each pair column.
        n_signals : int
        """
        frequencies = self.frequencies
        self._require_multiple_frequencies(measure)
        self._require_uniform_frequency_grid(measure)
        self._require_uniform_observation_weights(
            measure,
            "Its coherence significance test uses the observation count as the "
            "degrees of freedom of the zero-coherence null, which assumes equally "
            "weighted observations.",
        )
        self._warn_correlated_observations(
            measure,
            "Its coherence significance test uses the observation count as the "
            "degrees of freedom of the zero-coherence null",
        )
        frequency_difference = frequencies[1] - frequencies[0]
        independent_frequency_step = _get_independent_frequency_step(
            frequency_difference, frequency_resolution
        )
        bandpassed_coherency, bandpassed_frequencies = _bandpass(
            self._coherency(), frequencies, frequency_mask
        )
        # Significance testing and the masked phase below are NumPy operations.
        # Make the GPU-to-host boundary explicit before passing data into them;
        # NumPy deliberately refuses implicit conversion of CuPy arrays.
        bandpassed_coherency = to_numpy(bandpassed_coherency)
        bandpassed_frequencies = to_numpy(bandpassed_frequencies)
        n_signals = bandpassed_coherency.shape[-1]
        signal_combination_ind = np.asarray(list(combinations(np.arange(n_signals), 2)))
        bandpassed_coherency = bandpassed_coherency[
            ..., signal_combination_ind[:, 0], signal_combination_ind[:, 1]
        ]

        # One count per window and frequency where zero weights drop
        # observations, (..., n_band_frequencies, 1) against the pairs.
        n_observations: int | NDArray[np.intp] = self.n_observations
        if self._observation_weights is not None:
            n_observations = self.n_valid_observations[..., frequency_mask, np.newaxis]
        is_significant = _find_significant_frequencies(
            bandpassed_coherency,
            n_observations,
            independent_frequency_step,
            significance_threshold=significance_threshold,
            real_valued=self._real_valued_frequency_mask[frequency_mask][:, np.newaxis],
        )
        coherence_phase = np.ma.masked_array(
            _unwrap_across_undefined(np.angle(bandpassed_coherency), axis=-2),
            mask=~is_significant,
        )
        return coherence_phase, bandpassed_frequencies, signal_combination_ind, n_signals

    @_refused_for_exact_spectrum("significance_gate")
    def group_delay(
        self,
        frequencies_of_interest: Sequence[float] | NDArray[np.floating] | None = None,
        frequency_resolution: float | None = None,
        significance_threshold: float = 0.05,
        *,
        frequency_range: tuple[float, float] | None = None,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
        """Return the average time-delay of a broadband signal.

        Parameters
        ----------
        frequencies_of_interest : array-like, shape (2,), optional
            Deprecated; use ``frequency_range``. Will be removed in 4.0. The
            historical band ``(low, high)``, with both edges excluded: only bins
            strictly inside it are used. It keeps its 2.x position as the first
            positional argument, so a positional band keeps this meaning.
        frequency_resolution : float, optional
            Frequency resolution for independent samples.
        significance_threshold : float, default=0.05
            False discovery rate of the test that selects the frequencies used.
            Each bin's coherence is tested against zero with the exact null of
            :func:`~spectral_connectivity.statistics.coherence_significance_pvalue`,
            and the Benjamini-Hochberg procedure is applied at this level to
            every bin in ``frequency_range``, signal pair, and time window of the
            call as one family. For each pair and window only the largest
            contiguous run of significant bins is kept, taking every independent
            bin (one per ``frequency_resolution``), and only if it has at least
            3; a pair with no such run is NaN (see Returns).
        frequency_range : (float, float), optional
            Frequency interval ``(low, high)`` to fit over, in the units of
            ``frequencies``. Both endpoints are included: a bin lies on a bound
            when they differ by at most one part in a million of the bin's
            spacing, plus rounding (see
            :func:`~spectral_connectivity.frequency_band_reduce`). At least one
            bin must be selected. ``None`` uses every frequency.

        Returns
        -------
        delay : array, shape (..., n_signals, n_signals)
            Time delays between signal pairs, in the reciprocal units of
            ``frequencies``: seconds for Hz, samples for cycles/sample.
            Positive ``[..., i, j]`` means signal ``i`` leads signal ``j``. The
            diagonal is NaN, and so is a pair whose coherence is not
            significant at enough frequencies (see ``significance_threshold``),
            such as two independent signals: NaN means no delay was estimated,
            not a delay of zero.
        slope : array, shape (..., n_signals, n_signals)
            Slope of the coherence phase vs frequency, in radians per unit of
            ``frequencies`` (``delay = slope / (2 * pi)``); same sign
            convention as ``delay``, and NaN where it is.
        r_value : array, shape (..., n_signals, n_signals)
            Correlation coefficient of the linear phase-frequency fit, with the
            sign of ``slope`` (so ``[..., j, i]`` is ``-[..., i, j]``). The
            diagonal is 1, where ``delay`` and ``slope`` are NaN; a pair
            without significant coherence is NaN.

        Notes
        -----
        **Range**: (-∞, ∞). Time delays can be positive or negative.

        References
        ----------
        .. [1] Gotman, J. (1983). Measurement of small time differences
               between EEG channels: method and application to epileptic
               seizure propagation. Electroencephalography and Clinical
               Neurophysiology 56, 501-514.

        Examples
        --------
        >>> import numpy as np
        >>> from spectral_connectivity import Connectivity, Multitaper
        >>> rng = np.random.default_rng(0)
        >>> leader = rng.standard_normal((1003, 20))
        >>> # Signal 1 is signal 0 delayed by 3 samples (6 ms at 500 Hz), plus noise.
        >>> signals = np.stack([leader[3:], leader[:-3]], axis=-1)  # (time, trials, signals)
        >>> signals += 0.5 * rng.standard_normal(signals.shape)
        >>> multitaper = Multitaper(signals, sampling_frequency=500)
        >>> connectivity = Connectivity.from_transform(multitaper)
        >>> delay, slope, r_value = connectivity.group_delay(frequency_range=(5, 50))
        >>> delay.shape  # (n_time_windows, n_signals, n_signals)
        (1, 2, 2)
        >>> # Signal 0 leads signal 1 by 6 ms, so [..., 0, 1] is +0.006 s.
        >>> round(float(delay[0, 0, 1]), 3), round(float(delay[0, 1, 0]), 3)
        (0.006, -0.006)
        >>> bool(r_value[0, 0, 1] > 0.9)  # phase is linear in frequency
        True
        """
        _validate_alpha(significance_threshold, "significance_threshold")
        coherence_phase, bandpassed_frequencies, signal_combination_ind, n_signals = (
            self._significant_pair_phase(
                "group_delay",
                self._selected_frequencies(
                    "group_delay", frequency_range, frequencies_of_interest
                ),
                frequency_resolution,
                significance_threshold,
            )
        )

        # Vectorized masked linear regression of the unwrapped phase on
        # frequency, per (batch, signal pair), replacing a per-slice scipy
        # ``linregress`` call (``apply_along_axis`` invokes it once per slice).
        # Uses the closed-form ordinary-least-squares slope and Pearson r from
        # centered masked sums over the frequency axis (-2); where at least two
        # distinct significant frequencies remain these match ``linregress`` to
        # floating-point tolerance, and degenerate slices (fewer than two, or a
        # single distinct frequency) yield NaN as ``linregress`` does.
        is_valid = ~np.ma.getmaskarray(coherence_phase)
        # Replace masked entries with a finite value before the arithmetic: the
        # masked phase can be NaN (e.g. a zero-power bin's coherency angle), and
        # ``0 * NaN`` is NaN, which would poison the whole slice's sums even
        # though the valid mask already excludes those entries.
        phase = np.where(is_valid, np.ma.getdata(coherence_phase), 0.0)
        frequency = np.asarray(bandpassed_frequencies, dtype=float).reshape(-1, 1)
        axis = -2
        count = is_valid.sum(axis, keepdims=True)
        safe_count = np.where(count == 0, 1, count)
        # Center x and y on their per-slice means before summing squares, so the
        # variance does not catastrophically cancel when the absolute
        # frequencies are large relative to their spacing (raw-moment
        # ``count * sum_xx - sum_x**2`` loses all precision there).
        mean_x = (is_valid * frequency).sum(axis, keepdims=True) / safe_count
        mean_y = (is_valid * phase).sum(axis, keepdims=True) / safe_count
        centered_x = frequency - mean_x
        centered_y = phase - mean_y
        sum_xx = (is_valid * centered_x * centered_x).sum(axis, keepdims=True)
        sum_yy = (is_valid * centered_y * centered_y).sum(axis, keepdims=True)
        sum_xy = (is_valid * centered_x * centered_y).sum(axis, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            pair_slope = (sum_xy / sum_xx)[..., 0, :]
            pair_r_value = (sum_xy / np.sqrt(sum_xx * sum_yy))[..., 0, :]
        # Guard against |r| drifting just past 1 from rounding.
        pair_r_value = np.clip(pair_r_value, -1.0, 1.0)

        new_shape = (*coherence_phase.shape[:-2], n_signals, n_signals)
        slope = np.full(new_shape, np.nan)
        slope[..., signal_combination_ind[:, 0], signal_combination_ind[:, 1]] = pair_slope
        slope[..., signal_combination_ind[:, 1], signal_combination_ind[:, 0]] = -pair_slope

        delay = slope / (2 * np.pi)

        r_value = np.ones(new_shape)
        r_value[..., signal_combination_ind[:, 0], signal_combination_ind[:, 1]] = pair_r_value
        # The reverse pair's phase is negated, so its correlation is too.
        r_value[
            ..., signal_combination_ind[:, 1], signal_combination_ind[:, 0]
        ] = -pair_r_value

        return delay, slope, r_value

    @_refused_for_exact_spectrum("significance_gate")
    @_asnumpy
    def delay(
        self,
        frequencies_of_interest: Sequence[float] | NDArray[np.floating] | None = None,
        frequency_resolution: float | None = None,
        significance_threshold: float = 0.05,
        n_range: int = 3,
        *,
        frequency_range: tuple[float, float] | None = None,
    ) -> NDArray[np.floating]:
        """Find a range of possible delays from the coherence phase.

        The delay (and phase) at each frequency is indistinguishable from
        2π phase jumps, but we can look at a range of possible delays
        and see which one is most likely.

        Parameters
        ----------
        frequencies_of_interest : array-like, shape (2,), optional
            Deprecated; use ``frequency_range``. Will be removed in 4.0. The
            historical band ``(low, high)``, with both edges excluded: only bins
            strictly inside it are used. It keeps its 2.x position as the first
            positional argument, so a positional band keeps this meaning.
        frequency_resolution : float, optional
            Frequency resolution for independent samples.
        significance_threshold : float, default=0.05
            False discovery rate of the test that selects the frequencies used.
            Each bin's coherence is tested against zero with the exact null of
            :func:`~spectral_connectivity.statistics.coherence_significance_pvalue`,
            and the Benjamini-Hochberg procedure is applied at this level to
            every bin in ``frequency_range``, signal pair, and time window of the
            call as one family. For each pair and window only the largest
            contiguous run of significant bins is kept, taking every independent
            bin (one per ``frequency_resolution``), and only if it has at least
            3; a pair with no such run is NaN (see Returns).
        n_range : int, default=3
            Number of phases to consider.
        frequency_range : (float, float), optional
            Frequency interval ``(low, high)`` to evaluate, in the units of
            ``frequencies``. Both endpoints are included: a bin lies on a bound
            when they differ by at most one part in a million of the bin's
            spacing, plus rounding (see
            :func:`~spectral_connectivity.frequency_band_reduce`). At least one
            bin must be selected. ``None`` uses every frequency.

        Returns
        -------
        possible_delays : array
            Shape (..., n_frequencies, (n_range * 2) + 1, n_signals, n_signals),
            where ``n_frequencies`` counts only the frequencies in
            ``frequency_range``. Candidate ``k`` (index ``k + n_range``)
            adds ``k`` cycles of phase. Array of possible time delays in the
            reciprocal units of ``frequencies`` (seconds for Hz, samples for
            cycles/sample); positive ``[..., i, j]`` means signal ``i`` leads
            signal ``j``. The true delay is the candidate that is consistent
            (frequency-independent) across the band. Frequencies without
            significant coherence (see ``significance_threshold``), and the
            0 Hz (DC) bin, are undefined and returned as NaN: a pair with no
            significant frequency, such as two independent signals, is NaN
            throughout.

        Examples
        --------
        >>> import numpy as np
        >>> from spectral_connectivity import Connectivity, Multitaper
        >>> rng = np.random.default_rng(0)
        >>> leader = rng.standard_normal((1003, 20))
        >>> # Signal 1 is signal 0 delayed by 3 samples (6 ms at 500 Hz), plus noise.
        >>> signals = np.stack([leader[3:], leader[:-3]], axis=-1)  # (time, trials, signals)
        >>> signals += 0.5 * rng.standard_normal(signals.shape)
        >>> multitaper = Multitaper(signals, sampling_frequency=500)
        >>> connectivity = Connectivity.from_transform(multitaper)
        >>> delays = connectivity.delay(frequency_range=(5, 50), n_range=3)
        >>> # 5 to 50 Hz in 0.5 Hz steps, both endpoints included: 91 bins.
        >>> delays.shape  # (n_time_windows, n_band_freqs, n_candidates, n_signals, n_signals)
        (1, 91, 7, 2, 2)
        >>> # The zero-wrap candidate (index n_range) recovers the 6 ms lead of signal 0.
        >>> round(float(np.nanmedian(delays[0, :, 3, 0, 1])), 3)
        0.006
        """
        _validate_alpha(significance_threshold, "significance_threshold")
        coherence_phase, bandpassed_frequencies, signal_combination_ind, n_signals = (
            self._significant_pair_phase(
                "delay",
                self._selected_frequencies("delay", frequency_range, frequencies_of_interest),
                frequency_resolution,
                significance_threshold,
            )
        )
        possible_range = 2 * np.pi * np.arange(-n_range, n_range + 1)
        # Convert phase to a time delay: tau = (phase + 2*pi*k) / (2*pi*f). The
        # 2*pi*k terms resolve the phase-wrapping ambiguity. Dividing only by
        # 2*pi (omitting f) would return cycles, not seconds, making a constant
        # physical delay appear frequency-dependent.
        cycles = np.rollaxis(
            (possible_range + coherence_phase[..., np.newaxis]) / (2 * np.pi), -1, -2
        )
        # cycles has shape (..., n_frequencies, n_candidates, n_pairs); divide by
        # the frequency along the n_frequencies axis (-3). DC (f == 0) has no
        # defined delay and becomes NaN.
        frequency = bandpassed_frequencies[:, np.newaxis, np.newaxis]
        with np.errstate(divide="ignore", invalid="ignore"):
            delays = cycles / frequency
        delays[..., bandpassed_frequencies == 0, :, :] = np.nan
        # Fill non-significant frequencies (masked) with NaN rather than the
        # masked array's underlying 0.0, so a non-significant bin is not read as
        # a genuine zero-lag delay. This matches the DC handling above.
        delays = np.ma.filled(delays, np.nan)
        new_shape = (
            *coherence_phase.shape[:-1],
            len(possible_range),
            n_signals,
            n_signals,
        )
        possible_delays = np.full(new_shape, np.nan)
        possible_delays[..., signal_combination_ind[:, 0], signal_combination_ind[:, 1]] = (
            delays
        )
        # The reverse pair's phase is -phase, so its candidate k,
        # (-phase + 2*pi*k) / (2*pi*f), is minus the forward candidate -k.
        possible_delays[
            ..., signal_combination_ind[:, 1], signal_combination_ind[:, 0]
        ] = -delays[..., ::-1, :]

        return possible_delays

    @_asnumpy
    def phase_slope_index(
        self,
        frequencies_of_interest: Sequence[float] | NDArray[np.floating] | None = None,
        frequency_resolution: float | None = None,
        *,
        frequency_range: tuple[float, float] | None = None,
    ) -> NDArray[np.floating]:
        """Return weighted average of slopes projected onto imaginary axis.

        The phase slope index sums the product of the coherency at adjacent
        frequencies, ``conj(C(f)) * C(f + df)``, over the band and projects the
        result onto the imaginary axis to avoid volume-conduction effects
        (Nolte et al. 2008). The magnitude of the coherency at each frequency
        therefore weights the contribution of that frequency step.

        Parameters
        ----------
        frequencies_of_interest : array-like, shape (2,), optional
            Deprecated; use ``frequency_range``. Will be removed in 4.0. The
            historical band ``(low, high)``, with both edges excluded: only bins
            strictly inside it are used. It keeps its 2.x position as the first
            positional argument, so a positional band keeps this meaning.
        frequency_resolution : float, optional
            Frequency resolution for independent samples.
        frequency_range : (float, float), optional
            Frequency interval ``(low, high)`` to sum over, in the units of
            ``frequencies``. Both endpoints are included: a bin lies on a bound
            when they differ by at most one part in a million of the bin's
            spacing, plus rounding (see
            :func:`~spectral_connectivity.frequency_band_reduce`). At least one
            bin must be selected. ``None`` uses every frequency.

        Returns
        -------
        phase_slope_index : array, shape (..., n_signals, n_signals)
            Phase slope index values. Positive ``[..., i, j]`` means signal
            ``i`` leads signal ``j``; the result is antisymmetric.

        Notes
        -----
        **Range**: (-∞, ∞). Signed directional measure with no bounds.

        References
        ----------
        .. [1] Nolte, G., Ziehe, A., Nikulin, V.V., Schlogl, A., Kramer,
               N., Brismar, T., and Muller, K.-R. (2008). Robustly
               Estimating the Flow Direction of Information in Complex
               Physical Systems. Physical Review Letters 100.

        Examples
        --------
        >>> import numpy as np
        >>> from spectral_connectivity import Connectivity, Multitaper
        >>> rng = np.random.default_rng(0)
        >>> leader = rng.standard_normal((1003, 20))
        >>> # Signal 1 is signal 0 delayed by 3 samples (6 ms at 500 Hz), plus noise.
        >>> signals = np.stack([leader[3:], leader[:-3]], axis=-1)  # (time, trials, signals)
        >>> signals += 0.5 * rng.standard_normal(signals.shape)
        >>> multitaper = Multitaper(signals, sampling_frequency=500)
        >>> connectivity = Connectivity.from_transform(multitaper)
        >>> psi = connectivity.phase_slope_index(frequency_range=(5, 50))
        >>> psi.shape  # (n_time_windows, n_signals, n_signals)
        (1, 2, 2)
        >>> # Signal 0 leads signal 1, so [..., 0, 1] is positive and [..., 1, 0] negative.
        >>> bool(psi[0, 0, 1] > 0 > psi[0, 1, 0])
        True
        """
        frequencies = self.frequencies
        bandpassed_coherency, bandpassed_frequencies = _bandpass(
            self._coherency(),
            frequencies,
            self._selected_frequencies(
                "phase_slope_index", frequency_range, frequencies_of_interest
            ),
        )

        self._require_multiple_frequencies("phase_slope_index")
        self._require_uniform_frequency_grid("phase_slope_index")
        frequency_difference = frequencies[1] - frequencies[0]
        independent_frequency_step = _get_independent_frequency_step(
            frequency_difference, frequency_resolution
        )
        frequency_index = xp.arange(
            0, bandpassed_frequencies.shape[0], independent_frequency_step
        )
        bandpassed_coherency = bandpassed_coherency[..., frequency_index, :, :]

        # The phase slope index needs at least two frequency bins to form an
        # adjacent-frequency product. With fewer, the sum below would be an empty
        # sum that NumPy reports as 0, which is indistinguishable from a genuine
        # "no directionality" result.
        n_band_frequencies = bandpassed_coherency.shape[-3]
        if n_band_frequencies < 2:
            msg = (
                f"phase_slope_index needs at least 2 frequency bins in the band "
                f"after subsampling, but got {n_band_frequencies}. Widen "
                f"frequency_range, or decrease frequency_resolution so "
                f"more than one independent frequency remains."
            )
            raise ValueError(msg)

        # Nolte et al. (2008): sum conj(C(f)) * C(f + df) over adjacent
        # (independent) frequency bins, then take the imaginary part. The
        # frequency axis is -3 (the two trailing axes are the signal pair).
        adjacent_product: NDArray[np.complexfloating] = (
            xp.conj(bandpassed_coherency[..., :-1, :, :]) * bandpassed_coherency[..., 1:, :, :]
        ).sum(axis=-3)
        return adjacent_product.imag


def _paired_frequency_bins(n_fft_samples: int) -> tuple[slice, slice]:
    """The interior positive bins of a two-sided FFT and their negative partners.

    In standard FFT order bin ``k`` (``1 <= k <= (n_fft_samples - 1) // 2``)
    is frequency ``k`` and bin ``n_fft_samples - k`` is ``-k``; the second
    slice lists the partners in the order of the first. 0 Hz (bin 0) and,
    for an even length, Nyquist (bin ``n_fft_samples // 2``) have no partner.
    """
    n_interior = (n_fft_samples - 1) // 2
    return (
        slice(1, n_interior + 1),
        slice(n_fft_samples - 1, n_fft_samples - 1 - n_interior, -1),
    )


def _axes_description(dims: tuple[str, ...] | None) -> str:
    """``"axes (time, frequency, ...) with "`` for known ``dims``, else ``""``."""
    return "" if dims is None else f"axes ({', '.join(dims)}) with "


def _surrogate_refusal(method: str) -> str | None:
    """Why ``surrogate_test`` refuses a registered measure (a key of
    ``_SURROGATE_REFUSALS``), or ``None``.

    A measure of one signal (``output_kind="power"``), candidate delays
    (``"delay"``), and a phase (``quantity="phase"``) are refused.
    """
    # The registry imports this module, so it is imported here.
    from spectral_connectivity._measure_registry import _MEASURE_SPECS

    spec = _MEASURE_SPECS.get(method)
    if spec is None:
        return None
    if spec.output_kind == "power":
        return "single_signal"
    if spec.output_kind == "delay":
        return "candidate_delays"
    if spec.quantity == "phase":
        return "phase"
    return None


def _signed_measure_alternative(method: str) -> Literal["greater", "two-sided"]:
    """``surrogate_test``'s ``alternative="auto"`` for ``method``.

    ``"two-sided"`` for a registered measure whose range is signed and whose
    sign says which signal leads; ``"greater"`` otherwise, including the
    debiased estimators whose range is signed only because a bias correction
    can push them below 0, and unregistered extension measures.
    """
    # The registry imports this module, so it is imported here.
    from spectral_connectivity._measure_registry import _MEASURE_SPECS

    spec = _MEASURE_SPECS.get(method)
    if spec is not None and spec.value_range[0] < 0 and spec.sign_convention is not None:
        return "two-sided"
    return "greater"


def _validated_half_bandwidth(half_bandwidth: Any) -> float:
    """``half_bandwidth`` as a float, validated like a sampling frequency.

    Raises
    ------
    TypeError
        If it is not one real number (see :func:`_real_scalar`).
    ValueError
        If it is not finite and positive.
    """
    value = _real_scalar(half_bandwidth)
    advice = (
        "the half-width of the transform's spectral smoothing in the units of "
        "frequencies (for a Multitaper, frequency_resolution / 2)"
    )
    if value is None:
        msg = (
            f"half_bandwidth must be a number, {advice}; got "
            f"{type(half_bandwidth).__name__} {half_bandwidth!r}."
        )
        raise TypeError(msg)
    if not np.isfinite(value) or value <= 0:
        msg = f"half_bandwidth must be a positive number, {advice}; got {half_bandwidth!r}."
        raise ValueError(msg)
    return value


def _real_array_result(
    result: Any,
    method: str,
    caller: str,
    *,
    structured_hint: str = "",
    complex_hint: str = "",
) -> NDArray[np.floating]:
    """``result`` of ``method`` as a real array, or the error ``caller`` raises.

    Raises
    ------
    TypeError
        If the measure returned a tuple, a structured (object) result, or a
        complex array. ``structured_hint`` and ``complex_hint`` are appended
        to the structured-result and complex-result messages.
    """
    if isinstance(result, tuple):
        # The registry imports this module, so its helper is imported here.
        from spectral_connectivity._measure_registry import _is_group_measure

        if _is_group_measure(method):
            msg = (
                f"{caller} does not support the group measure {method!r}: it "
                "returns its values together with the group labels, and "
                f"{caller} does not resample group measures yet. Use a pairwise "
                "measure, or call the group measure directly."
            )
        else:
            msg = f"{caller} does not support tuple-valued measure {method!r}."
        raise TypeError(msg)
    values: NDArray[np.floating] = np.asarray(result)
    if values.dtype == object:
        msg = (
            f"{caller} requires a real array result from {method!r}; it "
            f"returned a structured result.{structured_hint}"
        )
        raise TypeError(msg)
    if np.iscomplexobj(values):
        msg = f"{caller} requires a real-valued measure; {method!r} is complex.{complex_hint}"
        raise TypeError(msg)
    return values


def _divide_masking_zero_denominator(
    numerator: NDArray[_NumberT],
    denominator: NDArray[np.floating],
    message: str,
) -> NDArray[_NumberT]:
    """Divide, returning NaN where the denominator is (near-)zero.

    The coherence measures are undefined where a signal has (near-)zero power.
    Rather than dividing by a floored epsilon (which yields spuriously large
    values), this emits a single ``UserWarning`` (``message``) when any
    denominator entry is at or below the dtype's smallest positive value,
    divides against a safe (1.0-substituted) denominator, and sets those entries
    to NaN.

    Parameters
    ----------
    numerator : NDArray
        Numerator array (real or complex).
    denominator : NDArray[floating]
        Non-negative denominator, broadcastable against ``numerator``.
    message : str
        Warning text emitted once if any denominator entry is (near-)zero.

    Returns
    -------
    NDArray
        ``numerator / denominator`` with (near-)zero-denominator entries NaN.
    """
    zero = denominator <= xp.finfo(denominator.dtype).tiny
    invalid = zero | ~xp.isfinite(denominator)
    if xp.any(zero):
        warnings.warn(message, UserWarning, stacklevel=stacklevel_outside_package())
    # Dividing by a real array keeps the numerator's real/complex kind.
    return cast(NDArray[_NumberT], _divide_where(numerator, denominator, ~invalid, xp.nan))


def _total_inflow(
    transfer_function: NDArray[np.complexfloating],
    noise_variance: float | NDArray[np.floating] = 1.0,
    axis: int | tuple[int, ...] = -1,
) -> NDArray[np.floating]:
    """Measure effect of incoming signals onto a node via sum of squares.

    Parameters
    ----------
    transfer_function : array_like
        Transfer function matrix.
    noise_variance : float or array_like, default=1.0
        Noise variance values.
    axis : int, default=-1
        Axis for summation.

    Returns
    -------
    array_like
        Total inflow values.

    """
    return xp.sqrt(
        xp.sum(
            noise_variance * _squared_magnitude(transfer_function),
            keepdims=True,
            axis=axis,
        )
    )


def _get_noise_variance(
    noise_covariance: NDArray[np.complexfloating],
    axis: int = -2,
) -> NDArray[np.floating]:
    """Extract noise variance and broadcast it along the requested signal axis.

    The transfer function / MVAR coefficients use the convention
    ``[..., n_fft, target, source]`` (axis -2 is the target/row, axis -1 is the
    source/column). Different directed measures weight by the noise variance
    along different axes, so the diagonal of the covariance must be broadcast to
    match: partial-directed-coherence-family measures sum over the target axis
    (-2), while directed coherence sums over the source axis (-1).

    Parameters
    ----------
    noise_covariance : array, shape (..., n_signals, n_signals)
        Hermitian noise covariance matrix; its diagonal is real.
    axis : int, default=-2
        Signal axis the noise variance should vary along, either -2 (target) or
        -1 (source). A leading ``newaxis`` is always inserted for the frequency
        axis.

    Returns
    -------
    noise_variance : array
        Diagonal elements (noise variances) reshaped to broadcast against the
        transfer function along `axis`.

    """
    noise_variance = xp.real(xp.diagonal(noise_covariance, axis1=-1, axis2=-2))
    if axis == -2:
        return noise_variance[..., xp.newaxis, :, xp.newaxis]
    if axis == -1:
        return noise_variance[..., xp.newaxis, xp.newaxis, :]
    msg = f"axis must be -2 (target) or -1 (source), got {axis}"
    raise ValueError(msg)


def _max_psd_discrepancy(
    transfer_function: NDArray[np.complexfloating],
    noise_covariance: NDArray[np.complexfloating],
) -> float:
    """Return the largest relative gap between the diagonal-noise PSD and the true PSD.

    ``directed_coherence`` normalizes each target ``i`` by the diagonal-only power
    ``D_i = sum_k Cov_kk |H_ik|^2``, which equals the true power spectral density
    ``S_ii = (H Cov H^H)_ii`` only when the innovations are uncorrelated. This
    returns the largest relative gap ``|S_ii - D_i| / S_ii`` over all targets,
    frequencies, and batch elements -- a dimension-aware measure of how much
    cross-power the diagonal formula omits. Unlike a pairwise-correlation
    criterion, it flags many weakly-but-jointly correlated sources whose cross
    terms still omit a large fraction of the true power.

    Parameters
    ----------
    transfer_function : array, shape (..., n_fft_samples, n_signals, n_signals)
        MVAR transfer function ``H`` (axis -2 target, axis -1 source).
    noise_covariance : array, shape (..., n_signals, n_signals)
        Estimated MVAR innovation covariance ``Cov``.

    Returns
    -------
    float
        Maximum relative PSD discrepancy, or 0.0 when it cannot be assessed
        (single signal, or all entries non-finite).

    """
    if transfer_function.shape[-1] < 2:
        return 0.0
    H = transfer_function
    # True PSD S_ii = (H Cov H^H)_ii = sum_l (H Cov)_il conj(H_il). Broadcast the
    # frequency-independent covariance over the frequency axis.
    covariance = noise_covariance[..., xp.newaxis, :, :]
    true_psd = xp.real(xp.sum(xp.matmul(H, covariance) * xp.conj(H), axis=-1))
    # Diagonal-only power D_i = sum_k Cov_kk |H_ik|^2.
    noise_variance = xp.real(xp.diagonal(noise_covariance, axis1=-1, axis2=-2))
    diagonal_psd = xp.sum(
        noise_variance[..., xp.newaxis, xp.newaxis, :] * _squared_magnitude(H),
        axis=-1,
    )
    # The true PSD is non-negative in exact arithmetic (diagonal of a PSD
    # matrix); clamp roundoff-negative values to 0 so a near-zero true power with
    # nonzero diagonal power reads as an infinite (not a large-negative) gap.
    true_psd = xp.maximum(true_psd, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        relative = xp.abs(true_psd - diagonal_psd) / true_psd
    # Keep +inf (true power 0 but diagonal power > 0 -- the diagonal formula is
    # infinitely wrong there, the strongest reason to warn); drop only 0/0 NaN
    # (no power at all, so nothing to normalize against).
    relative = relative[~xp.isnan(relative)]
    if relative.size == 0:
        return 0.0
    return float(xp.max(relative))


def _total_outflow(
    MVAR_Fourier_coefficients: NDArray[np.complexfloating],
    noise_variance: float | NDArray[np.floating] = 1.0,
) -> NDArray[np.floating]:
    """Measure effect of outgoing signals on the node via sum of squares.

    Parameters
    ----------
    MVAR_Fourier_coefficients : array_like
        MVAR Fourier coefficients.
    noise_variance : float or array_like, default=1.0
        Noise variance values.

    Returns
    -------
    array_like
        Total outflow values.

    """
    return xp.sqrt(
        xp.sum(
            _squared_magnitude(MVAR_Fourier_coefficients) / noise_variance,
            keepdims=True,
            axis=-2,
        )
    )


def _unwrap_across_undefined(
    phase: NDArray[np.floating], axis: int = -1
) -> NDArray[np.floating]:
    """``np.unwrap`` that skips NaN (undefined) phases instead of spreading them.

    ``np.unwrap`` accumulates corrections along ``axis``, so a single NaN makes
    every later value NaN. Each NaN is instead held at the last defined phase
    (or the first, before any is defined) while unwrapping, which unwraps the
    defined values as if the undefined ones were absent, and is restored to NaN
    afterwards. Without NaN the result equals ``np.unwrap``.
    """
    phase = np.moveaxis(phase, axis, -1)
    is_defined = ~np.isnan(phase)
    positions = np.arange(phase.shape[-1])
    last_defined = np.maximum.accumulate(np.where(is_defined, positions, 0), axis=-1)
    first_defined = np.argmax(is_defined, axis=-1)[..., np.newaxis]
    source = np.where(np.cumsum(is_defined, axis=-1) == 0, first_defined, last_defined)
    filled = np.take_along_axis(phase, source, axis=-1)
    unwrapped = np.where(is_defined, np.unwrap(filled, axis=-1), np.nan)
    return np.moveaxis(unwrapped, -1, axis)


def _bandpass(
    data: NDArray[np.complexfloating],
    frequencies: NDArray[np.floating],
    frequency_mask: NDArray[np.bool_],
    axis: int = -3,
) -> tuple[NDArray[np.complexfloating], NDArray[np.floating]]:
    """Keep the frequency bins of ``data`` selected by ``frequency_mask``.

    Parameters
    ----------
    data : array, shape (..., n_frequencies, ...)
        Input data array, on the host or the device.
    frequencies : array, shape (n_frequencies,)
        Frequency values.
    frequency_mask : NDArray[bool], shape (n_frequencies,)
        Host mask of the bins to keep (see :func:`_frequency_interval_mask`).
    axis : int, default=-3
        Frequency axis of ``data``.

    Returns
    -------
    filtered_data : array, shape (..., n_selected, ...)
        ``data`` at the selected bins; ``data`` itself when every bin is.
    filtered_frequencies : array, shape (n_selected,)
        The selected frequencies.
    """
    if frequency_mask.all():
        return data, frequencies
    index = np.flatnonzero(frequency_mask)
    return xp.take(data, xp.asarray(index), axis=axis), frequencies[index]


@dataclass(frozen=True)
class _FrequencySelection:
    """A frequency interval selected on a full grid, and how it was chosen.

    One call resolves its interval once, on the full grid, and every part of
    the result reads this record: the bins an interval measure consumes, its
    frequency coordinate and attributes, the wrappers' crop, and the Morlet
    validity masks. The interval is taken before any ``frequency_resolution``
    subsampling: ``phase_slope_index`` with ``frequency_resolution`` sums only
    every independent step of these bins, and ``group_delay``'s significance
    test can leave some out of its fit.

    Attributes
    ----------
    mask : NDArray[bool], shape (n_frequencies,)
        Host mask of the bins in the interval, over the full grid.
    requested : (float, float) or None
        The requested bounds; ``None`` when every bin is used.
    deprecated : bool
        Whether the interval was given as the deprecated
        ``frequencies_of_interest`` and selected by :func:`_frequencies_of_interest_mask`.
    lowest, highest : float
        The lowest and highest selected bin.
    """

    mask: NDArray[np.bool_]
    requested: tuple[float, float] | None
    deprecated: bool
    lowest: float
    highest: float

    @property
    def index(self) -> NDArray[np.intp]:
        """Positions of the selected bins on the full grid."""
        index: NDArray[np.intp] = np.flatnonzero(self.mask)
        return index

    def attrs(self) -> dict[str, Any]:
        """Result attributes recording the interval, in the grid's units.

        The requested bounds are omitted when every bin is used, and a side
        left open by an infinite bound of the deprecated keyword records no
        requested bound, so every attribute is a finite float; the selected
        bounds are the lowest and highest bins in the interval, before any
        ``frequency_resolution`` subsampling. ``frequency_interval_closed`` is
        ``"neither"`` for the deprecated keyword's exclusive edges, else
        ``"both"``.
        """
        attrs: dict[str, Any] = {}
        if self.requested is not None:
            lower, upper = self.requested
            if np.isfinite(lower):
                attrs["requested_frequency_lower"] = lower
            if np.isfinite(upper):
                attrs["requested_frequency_upper"] = upper
        attrs["selected_frequency_lower"] = self.lowest
        attrs["selected_frequency_upper"] = self.highest
        attrs["frequency_interval_closed"] = "neither" if self.deprecated else "both"
        return attrs


def _frequencies_of_interest_mask(
    frequencies_of_interest: Any, frequencies: NDArray[np.floating]
) -> tuple[NDArray[np.bool_], tuple[float, float]]:
    """The deprecated ``frequencies_of_interest`` selection, frozen at v2.0.1.

    v2.0.1's ``_bandpass`` evaluated ``(band[0] < frequencies) & (frequencies
    < band[1])`` on the band as the caller gave it and the grid as stored;
    this evaluates the same expression, so the installed NumPy promotes it
    exactly as it promotes v2.0.1's. On a float32 grid a Python-number bound
    is compared in float32 under every NumPy; a NumPy float64 scalar (from an
    ndarray band) is compared in float64 under NumPy 2 (NEP 50) but cast to
    float32 under NumPy 1.x's value-based casting, so the two forms can select
    different edge bins under NumPy 2 and the same bins under NumPy 1.x, as in
    v2.0.1. Every band of two values with a bin inside selects the bins
    v2.0.1 selects under the same NumPy, including an open-ended one such as
    ``(10, np.inf)``.

    The validation is this function's own and never changes with
    :func:`_frequency_interval_mask`. It rejects a band that is not exactly
    two values (v2.0.1 silently used the first two of a longer one), a NaN
    bound, ``low >= high``, and a band with no bin strictly inside, for which
    v2.0.1 returned an empty ``delay`` axis or failed with an unrelated error.

    Returns
    -------
    mask : NDArray[bool], shape (n_frequencies,)
        Host mask of the bins strictly inside the band.
    bounds : (float, float)
        The band as floats (possibly infinite), for messages and attributes.
    """
    invalid = (
        "frequencies_of_interest must be two values (low, high), not NaN, with "
        f"low < high, got {frequencies_of_interest!r}."
    )
    try:
        stored_lower, stored_upper = frequencies_of_interest
        lower, upper = float(stored_lower), float(stored_upper)
    except (TypeError, ValueError) as error:
        raise ValueError(invalid) from error
    if np.isnan(lower) or np.isnan(upper) or lower >= upper:
        raise ValueError(invalid)
    values = to_numpy(frequencies)
    mask: NDArray[np.bool_] = (stored_lower < values) & (values < stored_upper)
    if not mask.any():
        grid = values.astype(np.float64)
        msg = (
            f"frequencies_of_interest ({lower:g}, {upper:g}) contains no frequency bin "
            f"(both edges excluded); the {grid.size} bins span {grid.min():g} to "
            f"{grid.max():g}. Widen it."
        )
        raise ValueError(msg)
    return mask, (lower, upper)


def _select_measure_frequencies(
    frequencies: NDArray[np.floating],
    frequency_range: tuple[float, float] | None,
    frequencies_of_interest: Sequence[float] | NDArray[np.floating] | None,
) -> _FrequencySelection:
    """The bins ``delay``, ``group_delay``, or ``phase_slope_index`` consume.

    The one place their two keywords are resolved, shared by the measures and
    by the labeled results that describe them, so both name the same bins.

    Raises
    ------
    ValueError
        If both keywords are given, or the interval is invalid or selects no bin.
    """
    if frequency_range is not None and frequencies_of_interest is not None:
        msg = (
            "frequency_range and the deprecated frequencies_of_interest were both "
            "given; pass only frequency_range=(low, high), which includes both "
            "endpoints."
        )
        raise ValueError(msg)
    grid = to_numpy(frequencies)
    requested: tuple[float, float] | None = None
    if frequencies_of_interest is not None:
        mask, requested = _frequencies_of_interest_mask(frequencies_of_interest, grid)
    elif frequency_range is not None:
        mask, requested = _frequency_interval_mask("frequency_range", frequency_range, grid)
    else:
        mask = np.ones(grid.shape, dtype=bool)
    selected = grid[mask]
    return _FrequencySelection(
        mask,
        requested,
        deprecated=frequencies_of_interest is not None,
        lowest=float(selected[0]),
        highest=float(selected[-1]),
    )


def _get_independent_frequency_step(
    frequency_difference: float, frequency_resolution: float | None
) -> int:
    """Find number of points for statistically independent frequencies.

    Find the number of points of a frequency axis such that they
    are statistically independent given a frequency resolution.

    Parameters
    ----------
    frequency_difference : float
        The distance between two frequency points.
    frequency_resolution : float | None
        The ability to resolve frequency points. If None, returns 1.

    Returns
    -------
    frequency_step : int
        The number of points required so that two
        frequency points are statistically independent.

    """
    if frequency_resolution is None:
        return 1
    if not np.isfinite(frequency_resolution) or frequency_resolution <= 0:
        msg = (
            f"frequency_resolution must be a finite positive number when "
            f"provided, got {frequency_resolution}."
        )
        raise ValueError(msg)
    return int(xp.ceil(frequency_resolution / frequency_difference))


# Element cap (rows * n_frequencies) for one chunk of the significant-frequency
# selector. The per-slice int32 run-length temporaries dominate its memory, so
# processing the flattened signal-pair slices in chunks keeps peak usage bounded
# regardless of the number of slices.
_SIGNIFICANCE_SELECTION_CHUNK_ELEMENTS = 2_000_000


def _select_largest_independent_cluster(
    block: NDArray[np.bool_], frequency_step: int, min_group_size: int
) -> NDArray[np.bool_]:
    """Largest independent significant cluster per row (frequency on last axis).

    ``block`` has shape ``(n_rows, n_frequencies)``. See
    ``_largest_independent_group_along_frequency`` for the selection rule.
    """
    n_frequencies = block.shape[-1]
    # run_length[r, f] = length of the contiguous True-run ending at f (0 where
    # False): a cumulative count that resets at each False (running count minus
    # its value at the most recent False). int32 suffices (runs <= n_frequencies)
    # and halves the temporaries relative to the default int64.
    cumulative = np.cumsum(block, axis=-1, dtype=np.int32)
    run_length = cumulative - np.maximum.accumulate(
        np.where(block, np.int32(0), cumulative), axis=-1
    )

    # Largest run per row; run_length only reaches this value at a run's end, so
    # the first index attaining it is the end of the first largest cluster
    # (matching the "first cluster on ties" rule).
    max_size = run_length.max(axis=-1, keepdims=True)
    end_index = np.expand_dims(np.argmax(run_length == max_size, axis=-1), -1)
    start_index = end_index - max_size + 1

    frequency_index = np.arange(n_frequencies)
    # The largest cluster is contiguous [start_index, end_index]; max_size == 0
    # means no significant frequency, giving an all-False row.
    in_largest_cluster = (
        (frequency_index >= start_index) & (frequency_index <= end_index) & (max_size > 0)
    )
    # Independent points are start_index, start_index + frequency_step, ...
    independent: NDArray[np.bool_] = in_largest_cluster & (
        (frequency_index - start_index) % frequency_step == 0
    )
    count: NDArray[np.integer] = independent.sum(axis=-1, keepdims=True)
    return independent & (count >= min_group_size)


def _largest_independent_group_along_frequency(
    is_significant: NDArray[np.bool_], frequency_step: int, min_group_size: int
) -> NDArray[np.bool_]:
    """Largest independent significant-frequency group along axis -2.

    For every slice along axis -2, keep the largest contiguous cluster of
    significant frequencies (the first cluster on ties), subsample it every
    ``frequency_step`` points, and drop the slice to all-False if fewer than
    ``min_group_size`` independent points remain. Computed for all slices at
    once instead of one Python call per slice; the slices are processed in
    bounded chunks so peak memory stays independent of their number.

    Parameters
    ----------
    is_significant : bool array, shape (..., n_frequencies, n_signal_pairs)
    frequency_step : int
        Spacing (in points) between retained independent frequencies.
    min_group_size : int
        Minimum number of independent points for a cluster to be kept.

    Returns
    -------
    bool array, same shape as ``is_significant``.
    """
    axis = -2
    n_frequencies = is_significant.shape[axis]
    if is_significant.size == 0:
        # An empty frequency band (or any zero-length axis) has no cluster to
        # select; return an all-False array of the same (empty) shape.
        return np.zeros(is_significant.shape, dtype=bool)
    # Move frequency to the last axis and flatten the rest so the signal-pair
    # slices can be processed in chunks. asarray(copy=False) avoids a needless
    # copy of an already-boolean input; the reshape of the moved (non-contiguous)
    # view then makes a single bool copy.
    moved = np.moveaxis(np.asarray(is_significant, dtype=bool), axis, -1)
    flattened = moved.reshape(-1, n_frequencies)

    result = np.empty_like(flattened)
    chunk = max(1, _SIGNIFICANCE_SELECTION_CHUNK_ELEMENTS // max(1, n_frequencies))
    for start in range(0, flattened.shape[0], chunk):
        block = flattened[start : start + chunk]
        result[start : start + chunk] = _select_largest_independent_cluster(
            block, frequency_step, min_group_size
        )
    return np.moveaxis(result.reshape(moved.shape), -1, axis)


def _find_significant_frequencies(
    coherency: NDArray[np.complexfloating],
    n_obs: int | NDArray[np.integer],
    frequency_step: int = 1,
    significance_threshold: float = 0.05,
    min_group_size: int = 3,
    multiple_comparisons_method: Literal[
        "Benjamini_Hochberg_procedure", "Bonferroni_correction"
    ] = "Benjamini_Hochberg_procedure",
    real_valued: bool | NDArray[np.bool_] = False,
) -> NDArray[np.bool_]:
    """Determine the largest significant cluster along the frequency axis.

    This function uses the exact zero-coherence null distribution to determine
    the p-values and adjusts for multiple comparisons using the
    `multiple_comparisons_method`. Only independent frequencies are
    returned and there must be at least `min_group_size` frequency
    points for the cluster to be returned. If there are several significant
    groups, then only the largest group is returned.

    Parameters
    ----------
    coherency : array, shape (..., n_frequencies, n_signals, n_signals)
        The complex coherency between signals.
    n_obs : int or array of int
        The number of observations used to estimate the coherency: one count,
        or one per estimate, broadcastable against ``coherency``. An estimate
        of fewer than 2 observations has no test (its p-value is NaN, so it is
        never significant and does not count in the correction).
    frequency_step : int
        The number of points between each independent frequency step
    significance_threshold : float
        The threshold for a p-value to be considered significant.
    min_group_size : int
        The minimum number of independent frequency points for
    multiple_comparisons_method : 'Benjamini_Hochberg_procedure' |
        'Bonferroni_correction'
        Procedure used to correct for multiple comparisons.
    real_valued : bool or bool array, broadcastable to ``coherency``
        Estimates from real-valued coefficients, tested against the real null.

    Returns
    -------
    is_significant : bool array, shape (..., n_frequencies,
                                        n_signal_combintaions)

    """
    # Test each frequency's coherence against zero using the exact
    # magnitude-squared-coherence null distribution. The Fisher z-transform is
    # miscalibrated at the zero boundary and over-rejects the null ~3-4x.
    counts = np.asarray(n_obs)
    too_few = counts < 2
    p_values = coherence_significance_pvalue(
        coherency, np.where(too_few, 2, counts), real_valued=real_valued
    )
    if np.any(too_few):
        p_values = np.where(too_few, np.nan, p_values)
    is_significant = adjust_for_multiple_comparisons(
        p_values, alpha=significance_threshold, method=multiple_comparisons_method
    )
    return _largest_independent_group_along_frequency(
        is_significant, frequency_step, min_group_size
    )
