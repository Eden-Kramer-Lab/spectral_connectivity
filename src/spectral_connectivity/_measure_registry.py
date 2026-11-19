"""Registry of the measures the xarray wrapper can compute, and name checks."""

import difflib
import inspect
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import KW_ONLY, dataclass
from typing import Literal

import numpy as np

from spectral_connectivity.connectivity import (
    _DEPRECATED_MEASURE_NAMES,
    _NON_MEASURE_METHODS,
    Connectivity,
)
from spectral_connectivity.utils import stacklevel_outside_package


@dataclass(frozen=True)
class _MeasureSpec:
    """One measure's wrapper contract and what its values mean.

    ``output_kind`` and the capability flags describe the result's shape and
    requirements; the remaining fields label and interpret its values.
    ``units`` follows UDUNITS spelling, with ``"1"`` marking a dimensionless
    score and ``None`` a spectral density, whose units derive from the input's
    (see :meth:`units_for`). ``value_range`` bounds the returned
    values, or their magnitude when ``is_complex``.

    ``diagonal_policy`` says what a self-connection (a signal, or group, with
    itself) holds; ``diagonal_value`` is the value of a ``"constant"`` or
    ``"rounding_noise"`` one:

    - ``"masked_nan"``: NaN although the estimator defines a value there (for
      coherence, 1 at nonzero power): the package masks it.
    - ``"undefined_nan"``: NaN because the estimator has no self-connection.
    - ``"constant"``: the estimator's value, the same for any signal with
      nonzero power, to within the rounding of the averages it is computed
      from (about ``n_observations`` units in the last place).
    - ``"rounding_noise"``: nominally ``diagonal_value`` (an identity or a
      defined 0/0 limit), but computed from rounding residues that the
      measure's formula amplifies, so it can differ from that value by much
      more than rounding (see ``corrected_imaginary_phase_locking_value``).
    - ``"estimated"``: the estimator's value, which depends on the data.
    - ``"not_applicable"``: the result has no self-connection entries.

    ``diagonal_nan_for_zero_power`` marks a ``"constant"`` or
    ``"rounding_noise"`` measure that normalizes by each signal's amplitude,
    so a zero-power (flat) signal's self-connection is NaN instead of
    ``diagonal_value``; the others keep the value for a flat signal too.

    ``sign_convention`` says which signal leads when a measure's sign or angle
    shows it. ``quantity`` and ``formula`` define the coherency family's values
    (``None`` for other measures).
    """

    output_kind: Literal[
        "pairwise",
        "power",
        "group_pairwise",
        "delay",
        "global",
        "group_delay",
        "phase_slope",
        "multivariate_components",
    ]
    # The labels are keyword-only: three are strings, so a positional mix-up
    # would pass silently into results and the generated docs.
    _: KW_ONLY
    long_name: str
    units: str | None
    value_range: tuple[float, float]
    interpretation: str
    diagonal_policy: Literal[
        "masked_nan",
        "undefined_nan",
        "constant",
        "rounding_noise",
        "estimated",
        "not_applicable",
    ]
    diagonal_value: float | None = None
    diagonal_nan_for_zero_power: bool = False
    sign_convention: str | None = None
    quantity: (
        Literal["magnitude", "squared_magnitude", "signed", "phase", "complex"] | None
    ) = None
    formula: str | None = None
    is_complex: bool = False
    is_default: bool = False
    # Scientific directionality and spectrum requirements are independent
    # capabilities. For example dPLI and PSI are directional but do not need
    # Wilson factorization.
    is_directed: bool = False
    requires_two_sided: bool = False

    def __post_init__(self) -> None:
        nominal = self.diagonal_policy in {"constant", "rounding_noise"}
        if nominal != (self.diagonal_value is not None):
            msg = (
                "diagonal_value is the value of a constant or rounding_noise "
                "diagonal, and only that."
            )
            raise ValueError(msg)
        if self.diagonal_nan_for_zero_power and not nominal:
            msg = "diagonal_nan_for_zero_power qualifies a nominal diagonal value."
            raise ValueError(msg)

    @property
    def dims(self) -> tuple[str, ...]:
        """Dimensions of the main variable before band reduction or squeezing."""
        return _CATEGORY_DIMS[self.output_kind]

    @property
    def consumes_frequency_range(self) -> bool:
        """Whether the estimator selects the bins it uses with ``frequency_range``.

        The wrappers pass their ``frequency_range`` to such a measure instead of
        cropping its result.
        """
        return self.output_kind in _FREQUENCY_RANGE_KINDS

    def units_for(self, signal_units: str | None) -> str:
        """UDUNITS string of the values for input in ``signal_units``.

        A spectral density (``units is None``) is in ``(signal_units)^2/Hz``,
        and has no known units (``""``) when ``signal_units`` is unknown.
        """
        if self.units is not None:
            return self.units
        return f"({signal_units})^2/Hz" if signal_units else ""

    def result_attrs(self, signal_units: str | None) -> dict[str, str | float | int]:
        """The attrs of a result variable of this measure.

        NetCDF-safe scalars: an unbounded side of ``value_range`` and the
        facts the measure does not state are omitted, and unknown units too
        (see :meth:`units_for`).
        """
        units = self.units_for(signal_units)
        lower, upper = self.value_range
        optional: dict[str, str | float | int | None] = {
            "units": units or None,
            "value_range_lower": lower if np.isfinite(lower) else None,
            "value_range_upper": upper if np.isfinite(upper) else None,
            "value_range_applies_to": "magnitude" if self.is_complex else None,
            "sign_convention": self.sign_convention,
            "quantity": self.quantity,
            "formula": self.formula,
            # Which value a constant or rounding_noise diagonal holds, and
            # whether a flat signal's is NaN instead (0/1: NetCDF has no bool).
            "diagonal_value": self.diagonal_value,
            "diagonal_nan_for_zero_power": (
                None if self.diagonal_value is None else int(self.diagonal_nan_for_zero_power)
            ),
        }
        return {
            "long_name": self.long_name,
            "interpretation": self.interpretation,
            "diagonal_policy": self.diagonal_policy,
            **{key: value for key, value in optional.items() if value is not None},
        }


def _wilson_directed_spec(
    output_kind: Literal["pairwise", "group_pairwise"],
    *,
    long_name: str,
    units: str | None,
    value_range: tuple[float, float],
    interpretation: str,
    diagonal_policy: Literal["undefined_nan", "estimated"],
    is_default: bool = False,
) -> _MeasureSpec:
    """Spec of a directed measure computed from the Wilson-factorized spectrum.

    Such measures are directed and need a two-sided spectrum. Keeping the two
    flags together means a new measure of this kind cannot miss one.
    """
    return _MeasureSpec(
        output_kind,
        long_name=long_name,
        units=units,
        value_range=value_range,
        interpretation=interpretation,
        diagonal_policy=diagonal_policy,
        is_default=is_default,
        is_directed=True,
        requires_two_sided=True,
    )


_INFINITY = float("inf")
_PI = float(np.pi)
_LEADS = "Positive (source=a, target=b) means a leads b."
# coherency's cross-spectrum, reused by the coherency family's formulas.
_CROSS_SPECTRUM = "S_st = E[X_s * conj(X_t)], averaged over the expectation_type axes"
_DEBIASED = "Negative values are finite-sample noise around zero, not negative coupling."


# This is the single source of truth for wrapper capabilities, defaults, and
# value labels: adding a measure means adding one complete entry here.
# Insertion order preserves the historical Dataset variable order.
_MEASURE_SPECS: dict[str, _MeasureSpec] = {
    "magnitude_squared_coherence": _MeasureSpec(
        "pairwise",
        long_name="Magnitude-squared coherence",
        units="1",
        value_range=(0.0, 1.0),
        interpretation="Linear coupling at each frequency: 0 is none, 1 is a perfectly consistent "
        "amplitude and phase relationship. Biased upward when trials x tapers is small.",
        diagonal_policy="masked_nan",
        quantity="squared_magnitude",
        formula=f"|S_st|^2 / (S_ss * S_tt), where {_CROSS_SPECTRUM}",
        is_default=True,
    ),
    "coherence_phase": _MeasureSpec(
        "pairwise",
        long_name="Coherency phase",
        units="rad",
        value_range=(-_PI, _PI),
        interpretation="Mean phase difference in radians.",
        diagonal_policy="masked_nan",
        sign_convention=_LEADS,
        quantity="phase",
        formula=f"angle(S_st), the angle of coherency, where {_CROSS_SPECTRUM}",
        is_default=True,
    ),
    "debiased_squared_phase_lag_index": _MeasureSpec(
        "pairwise",
        long_name="Debiased squared phase lag index",
        units="1",
        value_range=(-1.0, 1.0),
        interpretation="Bias-corrected squared phase lag index. "
        f"{_DEBIASED} Lower bound is -1 / (n_observations - 1).",
        diagonal_policy="constant",
        diagonal_value=0.0,
        is_default=True,
    ),
    "debiased_squared_weighted_phase_lag_index": _MeasureSpec(
        "pairwise",
        long_name="Debiased squared weighted phase lag index",
        units="1",
        value_range=(-1.0, 1.0),
        interpretation=f"Bias-corrected squared weighted phase lag index. {_DEBIASED}",
        diagonal_policy="constant",
        diagonal_value=0.0,
        is_default=True,
    ),
    "imaginary_coherence": _MeasureSpec(
        "pairwise",
        long_name="Imaginary coherence (magnitude)",
        units="1",
        value_range=(0.0, 1.0),
        interpretation="Magnitude of the imaginary part of coherency; blind to zero-lag coupling "
        "such as volume conduction.",
        diagonal_policy="constant",
        diagonal_value=0.0,
        diagonal_nan_for_zero_power=True,
        quantity="magnitude",
        formula=f"|Im(S_st)| / sqrt(S_ss * S_tt), where {_CROSS_SPECTRUM}",
        is_default=True,
    ),
    "pairwise_phase_consistency": _MeasureSpec(
        "pairwise",
        long_name="Pairwise phase consistency",
        units="1",
        value_range=(-1.0, 1.0),
        interpretation="Bias-free estimate of the squared phase-locking value. "
        f"{_DEBIASED} Lower bound is -1 / (n_observations - 1).",
        diagonal_policy="constant",
        diagonal_value=1.0,
        diagonal_nan_for_zero_power=True,
        is_default=True,
    ),
    "pairwise_spectral_granger_prediction": _wilson_directed_spec(
        "pairwise",
        long_name="Spectral Granger prediction",
        units="1",
        value_range=(0.0, _INFINITY),
        interpretation="Nonparametric spectral Granger causality from source to target: 0 is no "
        "directed influence; larger values mean more of the target's power is "
        "predicted by the source's past. Not conditioned on other signals.",
        diagonal_policy="undefined_nan",
    ),
    "phase_lag_index": _MeasureSpec(
        "pairwise",
        long_name="Phase lag index",
        units="1",
        value_range=(-1.0, 1.0),
        interpretation="Signed asymmetry of the phase-difference distribution; blind to zero-lag "
        "coupling. Take the absolute value for the unsigned index.",
        diagonal_policy="constant",
        diagonal_value=0.0,
        sign_convention=_LEADS,
        is_default=True,
    ),
    "phase_locking_value": _MeasureSpec(
        "pairwise",
        long_name="Phase-locking value",
        units="1",
        value_range=(0.0, 1.0),
        interpretation="Consistency of the phase difference across trials and tapers, ignoring "
        "amplitude: 0 is random, 1 is constant. Biased upward with few observations.",
        diagonal_policy="constant",
        diagonal_value=1.0,
        diagonal_nan_for_zero_power=True,
        is_default=True,
    ),
    "power": _MeasureSpec(
        "power",
        long_name="Power spectral density",
        units=None,
        value_range=(0.0, _INFINITY),
        interpretation="One-sided power spectral density of each signal.",
        diagonal_policy="not_applicable",
        is_default=True,
    ),
    "weighted_phase_lag_index": _MeasureSpec(
        "pairwise",
        long_name="Weighted phase lag index",
        units="1",
        value_range=(-1.0, 1.0),
        interpretation="Phase lag index weighted by the magnitude of the imaginary cross-spectrum; "
        "less sensitive to noise than the unweighted index.",
        diagonal_policy="constant",
        diagonal_value=0.0,
        sign_convention=_LEADS,
        is_default=True,
    ),
    "coherency": _MeasureSpec(
        "pairwise",
        long_name="Coherency",
        units="1",
        value_range=(0.0, 1.0),
        interpretation="Complex coherency: its squared magnitude is "
        "magnitude_squared_coherence and its angle is coherence_phase.",
        diagonal_policy="masked_nan",
        sign_convention="A positive angle at (source=a, target=b) means a leads b.",
        quantity="complex",
        formula=f"S_st / sqrt(S_ss * S_tt), where {_CROSS_SPECTRUM}",
        is_complex=True,
    ),
    "cross_spectral_density": _MeasureSpec(
        "pairwise",
        long_name="Cross-spectral density",
        units=None,
        value_range=(0.0, _INFINITY),
        interpretation="Complex, Hermitian cross-spectrum; unnormalized, so it scales with signal "
        "power. Use coherency for a normalized version.",
        diagonal_policy="estimated",
        is_complex=True,
    ),
    "imaginary_coherency": _MeasureSpec(
        "pairwise",
        long_name="Imaginary part of coherency",
        units="1",
        value_range=(-1.0, 1.0),
        interpretation="Signed imaginary part of coherency; blind to zero-lag coupling.",
        diagonal_policy="masked_nan",
        sign_convention=_LEADS,
        quantity="signed",
        formula=f"Im(S_st) / sqrt(S_ss * S_tt), where {_CROSS_SPECTRUM}",
    ),
    "partial_coherence": _MeasureSpec(
        "pairwise",
        long_name="Partial coherence",
        units="1",
        value_range=(0.0, 1.0),
        interpretation="Magnitude-squared coherence after removing the linear influence of every "
        "other signal; near 0 for pairs coupled only through other signals.",
        diagonal_policy="masked_nan",
        quantity="squared_magnitude",
        formula="|P_st|^2 / (P_ss * P_tt), where P is the (regularized) inverse of the "
        "cross-spectral matrix",
    ),
    "corrected_imaginary_phase_locking_value": _MeasureSpec(
        "pairwise",
        long_name="Corrected imaginary phase-locking value",
        units="1",
        value_range=(0.0, 1.0),
        interpretation="Phase locking with zero- and pi-lag contributions removed; insensitive to "
        "volume conduction.",
        # A self-connection is the defined 0/0 limit, but the 0/0 is computed:
        # rounding in the self-PLV leaves |Im| of order eps over
        # sqrt(1 - Re^2) of order sqrt(eps).
        diagonal_policy="rounding_noise",
        diagonal_value=0.0,
        diagonal_nan_for_zero_power=True,
    ),
    "directed_phase_lag_index": _MeasureSpec(
        "pairwise",
        long_name="Directed phase lag index",
        units="1",
        value_range=(0.0, 1.0),
        interpretation="Above 0.5, the source phase-leads the target; below 0.5 it lags; 0.5 is "
        "no preferred direction.",
        diagonal_policy="constant",
        diagonal_value=0.5,
        is_directed=True,
    ),
    "subset_pairwise_spectral_granger_prediction": _wilson_directed_spec(
        "pairwise",
        long_name="Spectral Granger prediction",
        units="1",
        value_range=(0.0, _INFINITY),
        interpretation="pairwise_spectral_granger_prediction for only the requested pairs; other "
        "entries are NaN.",
        diagonal_policy="undefined_nan",
    ),
    "conditional_spectral_granger_prediction": _wilson_directed_spec(
        "pairwise",
        long_name="Conditional spectral Granger prediction",
        units="1",
        value_range=(0.0, _INFINITY),
        interpretation="Spectral Granger causality from source to target conditioned on every "
        "other signal, removing influence relayed through observed signals.",
        diagonal_policy="undefined_nan",
    ),
    "time_reversed_spectral_granger_prediction": _wilson_directed_spec(
        "pairwise",
        long_name="Time-reversed spectral Granger prediction",
        units="1",
        value_range=(0.0, _INFINITY),
        interpretation="Pairwise spectral Granger causality of the time-reversed data. Genuine "
        "directed influence reverses under time reversal; directionality that does "
        "not reverse suggests instantaneous mixing.",
        diagonal_policy="undefined_nan",
    ),
    # Directed-transfer-function family (opt-in).
    "directed_transfer_function": _wilson_directed_spec(
        "pairwise",
        long_name="Directed transfer function",
        units="1",
        value_range=(0.0, 1.0),
        interpretation="Fraction of the target's inflow at each frequency that comes from the "
        "source, including indirect paths; sums to 1 over sources.",
        diagonal_policy="estimated",
    ),
    "directed_coherence": _wilson_directed_spec(
        "pairwise",
        long_name="Directed coherence",
        units="1",
        value_range=(0.0, 1.0),
        interpretation="Noise-weighted directed transfer function: the fraction of the target's "
        "power attributable to the source; sums to 1 over sources. Assumes "
        "uncorrelated innovations.",
        diagonal_policy="estimated",
    ),
    "partial_directed_coherence": _wilson_directed_spec(
        "pairwise",
        long_name="Partial directed coherence",
        units="1",
        value_range=(0.0, 1.0),
        interpretation="Direct influence from source to target, normalized by the source's total "
        "outflow; sums to 1 over targets.",
        diagonal_policy="estimated",
    ),
    "generalized_partial_directed_coherence": _wilson_directed_spec(
        "pairwise",
        long_name="Generalized partial directed coherence",
        units="1",
        value_range=(0.0, 1.0),
        interpretation="Partial directed coherence with each signal scaled by its innovation "
        "variance, making it insensitive to differences in signal scale.",
        diagonal_policy="estimated",
    ),
    "direct_directed_transfer_function": _wilson_directed_spec(
        "pairwise",
        long_name="Direct directed transfer function",
        units="1",
        value_range=(0.0, 1.0),
        interpretation="Direct (not relayed) influence from source to target. Normalized over all "
        "frequencies, so values are small: compare pairs, not against 1.",
        diagonal_policy="estimated",
    ),
    "blockwise_spectral_granger_prediction": _wilson_directed_spec(
        "group_pairwise",
        long_name="Blockwise spectral Granger prediction",
        units="1",
        value_range=(0.0, _INFINITY),
        interpretation="Spectral Granger causality between groups of signals set by group_labels, "
        "from source_group to target_group.",
        diagonal_policy="undefined_nan",
    ),
    "canonical_coherence": _MeasureSpec(
        "group_pairwise",
        long_name="Canonical coherence",
        units="1",
        value_range=(0.0, 1.0),
        interpretation="Largest magnitude-squared coherence between linear combinations of two "
        "groups of signals, the squared first canonical correlation (the historical "
        "estimator). A different estimator from canonical_coherency, not its square.",
        diagonal_policy="masked_nan",
        quantity="squared_magnitude",
        formula="max over complex weights a, b of |a^H S_st b|^2 / ((a^H S_ss a) * "
        "(b^H S_tt b)), with S_st the cross-spectral matrix from the source group's signals "
        "to the target group's",
    ),
    "maximized_imaginary_coherency": _MeasureSpec(
        "group_pairwise",
        long_name="Maximized imaginary coherency",
        units="1",
        value_range=(0.0, 1.0),
        interpretation="Largest imaginary coherency between linear combinations of two groups; "
        "blind to zero-lag coupling.",
        diagonal_policy="masked_nan",
    ),
    "multivariate_interaction_measure": _MeasureSpec(
        "group_pairwise",
        long_name="Multivariate interaction measure",
        units="1",
        value_range=(0.0, _INFINITY),
        interpretation="Total phase-lagged interaction between two groups (the sum of squared "
        "imaginary-coherency components); at most the smaller group's rank.",
        diagonal_policy="masked_nan",
    ),
    "canonical_coherency": _MeasureSpec(
        "multivariate_components",
        long_name="Canonical coherency",
        units="1",
        value_range=(0.0, 1.0),
        interpretation="Phase-optimized complex canonical coherency (Vidaurre et al. 2019) per "
        "component between two groups, with the spatial filters and patterns that produce "
        "it; its magnitude is the maximized coherence, not squared. A different estimator "
        "from canonical_coherence, not its square root.",
        diagonal_policy="not_applicable",
        sign_convention="Conjugate convention: the score is magnitude * exp(-1j * phi), so with "
        "single-channel groups it is conj(coherency) and a negative angle means the source "
        "group leads. The phase is defined only up to the sign of the spatial filters.",
        quantity="complex",
        formula="max over a phase phi and real weights a, b of a^T Re(exp(-1j * phi) * S_st) b "
        "/ sqrt((a^T Re(S_ss) a) * (b^T Re(S_tt) b)), times exp(-1j * phi), with S_st the "
        "cross-spectral matrix from the source group's signals to the target group's",
        is_complex=True,
    ),
    "maximized_imaginary_coherency_components": _MeasureSpec(
        "multivariate_components",
        long_name="Maximized imaginary coherency",
        units="1",
        value_range=(0.0, 1.0),
        interpretation="maximized_imaginary_coherency resolved into components, with the spatial "
        "filters and patterns that produce them.",
        diagonal_policy="not_applicable",
    ),
    "delay": _MeasureSpec(
        "delay",
        long_name="Delay",
        units="s",
        value_range=(-_INFINITY, _INFINITY),
        interpretation="Candidate delays in seconds, one per 2*pi phase ambiguity; the true delay is "
        "the candidate that is constant across frequency. Frequencies without "
        "significant coherence are NaN.",
        diagonal_policy="masked_nan",
        sign_convention=_LEADS,
        is_directed=True,
    ),
    "global_coherence": _MeasureSpec(
        "global",
        long_name="Global coherence",
        units="1",
        value_range=(0.0, 1.0),
        interpretation="Fraction of the total cross-spectral power in each component; a large "
        "leading component indicates one dominant coherent network.",
        diagonal_policy="not_applicable",
    ),
    "group_delay": _MeasureSpec(
        "group_delay",
        long_name="Group delay",
        units="s",
        value_range=(-_INFINITY, _INFINITY),
        interpretation="Delay in seconds from the slope of phase against frequency over "
        "frequency_range (every bin by default); "
        "check group_delay_r_value for the quality of the fit.",
        diagonal_policy="masked_nan",
        sign_convention=_LEADS,
        is_directed=True,
    ),
    "phase_slope_index": _MeasureSpec(
        "phase_slope",
        long_name="Phase slope index",
        units="1",
        value_range=(-_INFINITY, _INFINITY),
        interpretation="Coherence-weighted slope of phase against frequency over "
        "frequency_range (every bin by default). "
        "Unnormalized, so judge it "
        "against a null distribution rather than a fixed threshold.",
        diagonal_policy="masked_nan",
        sign_convention=_LEADS,
        is_directed=True,
    ),
}


# The other value variables of a measure's Dataset, each with its own record
# (not listed by list_measures: they are parts of the measure named first).
_COMPANION_SPECS: dict[str, _MeasureSpec] = {
    "group_delay_slope": _MeasureSpec(
        "group_delay",
        long_name="phase slope",
        units="rad/Hz",
        value_range=(-_INFINITY, _INFINITY),
        interpretation="Slope of the unwrapped coherency phase against frequency over "
        "frequency_range, from the linear fit group_delay is computed from "
        "(group_delay = slope / (2*pi)).",
        diagonal_policy="masked_nan",
        sign_convention=_LEADS,
    ),
    "group_delay_r_value": _MeasureSpec(
        "group_delay",
        long_name="phase-frequency correlation",
        units="1",
        value_range=(-1.0, 1.0),
        interpretation="Correlation coefficient of the linear phase-frequency fit, with the "
        "sign of the slope: near +1 or -1 the phase is linear in frequency and "
        "group_delay is reliable.",
        diagonal_policy="constant",
        diagonal_value=1.0,
        sign_convention=_LEADS,
    ),
}

# Output kinds whose estimator consumes a frequency interval.
_FREQUENCY_RANGE_KINDS = frozenset({"delay", "group_delay", "phase_slope"})

#: Measures that consume ``frequency_range``, in registry order.
_FREQUENCY_RANGE_MEASURES: tuple[str, ...] = tuple(
    name for name, spec in _MEASURE_SPECS.items() if spec.consumes_frequency_range
)

# Dimensions of each category's main variable before any band reduction or
# squeezing.
_CATEGORY_DIMS: dict[str, tuple[str, ...]] = {
    "pairwise": ("time", "frequency", "source", "target"),
    "power": ("time", "frequency", "source"),
    "group_pairwise": ("time", "frequency", "source_group", "target_group"),
    "multivariate_components": ("time", "frequency", "connection", "component"),
    "delay": ("time", "frequency", "candidate", "source", "target"),
    "global": ("time", "frequency", "component"),
    "group_delay": ("time", "source", "target"),
    "phase_slope": ("time", "source", "target"),
}


def _canonical_measure_name(method: str) -> str:
    """The registered name ``method`` stands for: a deprecated name's replacement,
    otherwise ``method`` itself."""
    return _DEPRECATED_MEASURE_NAMES.get(method, method)


def _measure_spec(method: str) -> _MeasureSpec | None:
    """The registry record of ``method`` (or of the measure a deprecated name
    stands for); ``None`` for an unregistered extension measure."""
    return _MEASURE_SPECS.get(_canonical_measure_name(method))


def _requires_two_sided(method: str) -> bool:
    """Whether a registered measure needs a two-sided spectrum (Wilson factorization)."""
    spec = _measure_spec(method)
    return spec is not None and spec.requires_two_sided


def _is_group_measure(method: str) -> bool:
    """Whether a measure compares groups of signals, i.e. takes ``group_labels``.

    Decided from the ``Connectivity`` method's signature rather than the
    registry, so an unregistered extension measure that takes ``group_labels``
    counts too.
    """
    measure = getattr(Connectivity, method, None)
    return callable(measure) and "group_labels" in inspect.signature(measure).parameters


def _consumes_frequency_range(method: str) -> bool:
    """Whether a registered measure consumes ``frequency_range`` (see
    :attr:`_MeasureSpec.consumes_frequency_range`)."""
    spec = _measure_spec(method)
    return spec is not None and spec.consumes_frequency_range


def _listed(names: Sequence[str]) -> str:
    """``names`` as English prose: ``"a"``, ``"a and b"``, ``"a, b, and c"``."""
    if len(names) < 3:
        return " and ".join(names)
    return f"{', '.join(names[:-1])}, and {names[-1]}"


def _reduces_frequency(method: str) -> bool:
    """Whether a registered measure's result has no frequency axis."""
    spec = _measure_spec(method)
    return spec is not None and "frequency" not in spec.dims


def _requested_methods(
    method: str | Iterable[str] | None, defaults: Sequence[str]
) -> tuple[list[str], bool]:
    """The measures a wrapper call requests, and whether it named a single one.

    ``None`` requests ``defaults``; a string requests that one measure, whose
    result is returned as a DataArray rather than a Dataset. An empty request
    or an unknown name raises.
    """
    if method is None:
        methods = list(defaults)
    elif isinstance(method, str):
        methods = [method]
    else:
        methods = list(method)
    if not methods:
        msg = "method must name at least one connectivity measure; got an empty list."
        raise ValueError(msg)
    _validate_method_names(methods)
    _warn_deprecated_measure_names(methods)
    return methods, isinstance(method, str)


def _warn_deprecated_measure_names(methods: Sequence[str]) -> None:
    """Warn at the caller for each deprecated measure name requested.

    A request naming a measure under both its names is rejected: the two would
    be the same estimate under two variable names.
    """
    for method in methods:
        canonical = _DEPRECATED_MEASURE_NAMES.get(method)
        if canonical is None:
            continue
        if canonical in methods:
            msg = (
                f"method names {canonical!r} twice: {method!r} is the deprecated name "
                f"of {canonical!r}. Request it once, as {canonical!r}."
            )
            raise ValueError(msg)
        warnings.warn(
            f"method={method!r} is deprecated and will be removed in 4.0; use "
            f"method={canonical!r}, which computes the same values. Through 3.x a "
            f"result requested as {method!r} keeps that variable name and records "
            f"measure={method!r}.",
            DeprecationWarning,
            stacklevel=stacklevel_outside_package(),
        )


def _measure_description(name: str) -> str:
    """Return the one-line summary from a ``Connectivity`` method docstring."""
    docstring = getattr(Connectivity, name).__doc__ or ""
    for line in docstring.strip().splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _suggest_measure_names(name: str, limit: int = 5) -> list[str]:
    """Rank plausible measure names for a misspelled or abbreviated request.

    Substring matches (which handle abbreviations such as ``"granger"``) are
    preferred over ``difflib`` fuzzy matches (which handle single-character
    typos), since the former is what mistaken measure names usually look like.
    """
    lowered = name.lower()
    ranked: list[str] = [
        measure
        for measure in _MEASURE_SPECS
        if lowered in measure.lower() or measure.lower() in lowered
    ]
    lower_to_name = {measure.lower(): measure for measure in _MEASURE_SPECS}
    for hit in difflib.get_close_matches(lowered, lower_to_name, n=limit, cutoff=0.5):
        measure = lower_to_name[hit]
        if measure not in ranked:
            ranked.append(measure)
    return ranked[:limit]


def _is_extension_measure(name: str) -> bool:
    """Whether ``name`` is a public instance method usable as a measure."""
    if (
        name.startswith("_")
        or name in _NON_MEASURE_METHODS
        or name in _DEPRECATED_MEASURE_NAMES
    ):
        return False
    attribute = inspect.getattr_static(Connectivity, name, None)
    return inspect.isfunction(attribute)


def _validate_method_names(methods: Sequence[str]) -> None:
    """Reject unknown measure names with a helpful, actionable message.

    A name is accepted if it is either a registered measure or a public
    instance method of ``Connectivity``, so subclass/monkeypatched extension
    measures (which the wrapper supports) still pass. Properties, private
    helpers, classmethods, and the non-measure ``jackknife`` driver are rejected
    before they can produce an obscure ``TypeError``.
    """
    unknown = [
        method
        for method in methods
        if method not in _MEASURE_SPECS
        and method not in _DEPRECATED_MEASURE_NAMES
        and not _is_extension_measure(method)
    ]
    if not unknown:
        return
    parts = []
    for name in unknown:
        suggestions = _suggest_measure_names(name)
        if suggestions:
            hint = " Did you mean: " + ", ".join(repr(s) for s in suggestions) + "?"
        else:
            hint = ""
        parts.append(f"{name!r} is not a known connectivity measure.{hint}")
    parts.append(
        f"Call spectral_connectivity.list_measures() to see the "
        f"{len(_MEASURE_SPECS)} available measures."
    )
    raise ValueError(" ".join(parts))


def _measure_attrs(method: str, signal_units: str | None) -> dict[str, str | float]:
    """The attrs of ``method``'s result variable (see :meth:`_MeasureSpec.result_attrs`);
    an unregistered extension measure gets only its name as ``long_name``."""
    spec = _measure_spec(method)
    if spec is None:
        return {"long_name": method}
    return spec.result_attrs(signal_units)
