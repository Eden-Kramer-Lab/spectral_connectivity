"""Registry of the measures the xarray wrapper can compute, and name checks."""

import difflib
import inspect
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from spectral_connectivity.connectivity import _NON_MEASURE_METHODS, Connectivity


@dataclass(frozen=True)
class _MeasureSpec:
    """One measure's wrapper contract and what its values mean.

    ``output_kind`` and the capability flags describe the result's shape and
    orientation; the remaining fields label and interpret its values.
    ``units`` follows UDUNITS spelling, with ``"1"`` marking a dimensionless
    score and ``None`` a spectral density, whose units derive from the input's
    (see :func:`_measure_label_attrs`). ``value_range`` bounds the returned
    values, or their magnitude when ``is_complex``.
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
    long_name: str
    units: str | None
    value_range: tuple[float, float]
    interpretation: str
    is_complex: bool = False
    is_default: bool = False
    # Scientific directionality, native matrix orientation, and spectrum
    # requirements are independent capabilities. For example dPLI and PSI are
    # directional but already use source -> target orientation and do not need
    # Wilson factorization.
    is_directed: bool = False
    transpose_output: bool = False
    requires_two_sided: bool = False

    def __post_init__(self) -> None:
        # Make the field couplings unrepresentable rather than merely unused, so
        # a future registry entry cannot silently violate them.
        if self.transpose_output and self.output_kind not in {
            "pairwise",
            "group_pairwise",
        }:
            msg = "transpose_output requires pairwise or group_pairwise output."
            raise ValueError(msg)
        if self.transpose_output and not self.is_directed:
            msg = "transpose_output requires a directional measure."
            raise ValueError(msg)


_INFINITY = float("inf")
_PI = float(np.pi)
_LEADS = "Positive (source=a, target=b) means a leads b."
_DEBIASED = "Negative values are finite-sample noise around zero, not negative coupling."


# This is the single source of truth for wrapper capabilities, defaults, and
# value labels: adding a measure means adding one complete entry here.
# Insertion order preserves the historical Dataset variable order.
_MEASURE_SPECS: dict[str, _MeasureSpec] = {
    "coherence_magnitude": _MeasureSpec(
        "pairwise",
        "Magnitude-squared coherence",
        "1",
        (0.0, 1.0),
        "Linear coupling at each frequency: 0 is none, 1 is a perfectly consistent "
        "amplitude and phase relationship. Biased upward when trials x tapers is small.",
        is_default=True,
    ),
    "coherence_phase": _MeasureSpec(
        "pairwise",
        "Coherency phase",
        "rad",
        (-_PI, _PI),
        f"Mean phase difference in radians. {_LEADS}",
        is_default=True,
    ),
    "debiased_squared_phase_lag_index": _MeasureSpec(
        "pairwise",
        "Debiased squared phase lag index",
        "1",
        (-1.0, 1.0),
        "Bias-corrected squared phase lag index. "
        f"{_DEBIASED} Lower bound is -1 / (n_observations - 1).",
        is_default=True,
    ),
    "debiased_squared_weighted_phase_lag_index": _MeasureSpec(
        "pairwise",
        "Debiased squared weighted phase lag index",
        "1",
        (-1.0, 1.0),
        f"Bias-corrected squared weighted phase lag index. {_DEBIASED}",
        is_default=True,
    ),
    "imaginary_coherence": _MeasureSpec(
        "pairwise",
        "Imaginary coherence (magnitude)",
        "1",
        (0.0, 1.0),
        "Magnitude of the imaginary part of coherency; blind to zero-lag coupling "
        "such as volume conduction.",
        is_default=True,
    ),
    "pairwise_phase_consistency": _MeasureSpec(
        "pairwise",
        "Pairwise phase consistency",
        "1",
        (-1.0, 1.0),
        "Bias-free estimate of the squared phase-locking value. "
        f"{_DEBIASED} Lower bound is -1 / (n_observations - 1).",
        is_default=True,
    ),
    "pairwise_spectral_granger_prediction": _MeasureSpec(
        "pairwise",
        "Spectral Granger prediction",
        "1",
        (0.0, _INFINITY),
        "Nonparametric spectral Granger causality from source to target: 0 is no "
        "directed influence; larger values mean more of the target's power is "
        "predicted by the source's past. Not conditioned on other signals.",
        is_default=True,
        is_directed=True,
        transpose_output=True,
        requires_two_sided=True,
    ),
    "phase_lag_index": _MeasureSpec(
        "pairwise",
        "Phase lag index",
        "1",
        (-1.0, 1.0),
        "Signed asymmetry of the phase-difference distribution; blind to zero-lag "
        f"coupling. Take the absolute value for the unsigned index. {_LEADS}",
        is_default=True,
    ),
    "phase_locking_value": _MeasureSpec(
        "pairwise",
        "Phase-locking value",
        "1",
        (0.0, 1.0),
        "Consistency of the phase difference across trials and tapers, ignoring "
        "amplitude: 0 is random, 1 is constant. Biased upward with few observations.",
        is_default=True,
    ),
    "power": _MeasureSpec(
        "power",
        "Power spectral density",
        None,
        (0.0, _INFINITY),
        "One-sided power spectral density of each signal.",
        is_default=True,
    ),
    "weighted_phase_lag_index": _MeasureSpec(
        "pairwise",
        "Weighted phase lag index",
        "1",
        (-1.0, 1.0),
        "Phase lag index weighted by the magnitude of the imaginary cross-spectrum; "
        f"less sensitive to noise than the unweighted index. {_LEADS}",
        is_default=True,
    ),
    "coherency": _MeasureSpec(
        "pairwise",
        "Coherency",
        "1",
        (0.0, 1.0),
        "Complex coherency: its squared magnitude is coherence_magnitude and its "
        "angle is coherence_phase.",
        is_complex=True,
    ),
    "cross_spectral_density": _MeasureSpec(
        "pairwise",
        "Cross-spectral density",
        None,
        (0.0, _INFINITY),
        "Complex, Hermitian cross-spectrum; unnormalized, so it scales with signal "
        "power. Use coherency for a normalized version.",
        is_complex=True,
    ),
    "imaginary_coherency": _MeasureSpec(
        "pairwise",
        "Imaginary part of coherency",
        "1",
        (-1.0, 1.0),
        f"Signed imaginary part of coherency; blind to zero-lag coupling. {_LEADS}",
    ),
    "partial_coherence": _MeasureSpec(
        "pairwise",
        "Partial coherence",
        "1",
        (0.0, 1.0),
        "Magnitude-squared coherence after removing the linear influence of every "
        "other signal; near 0 for pairs coupled only through other signals.",
    ),
    "corrected_imaginary_phase_locking_value": _MeasureSpec(
        "pairwise",
        "Corrected imaginary phase-locking value",
        "1",
        (0.0, 1.0),
        "Phase locking with zero- and pi-lag contributions removed; insensitive to "
        "volume conduction.",
    ),
    # dPLI's native row/column layout is already phase-leader -> phase-lagger,
    # so it must not receive the transpose used by Granger/DTF-family outputs.
    "directed_phase_lag_index": _MeasureSpec(
        "pairwise",
        "Directed phase lag index",
        "1",
        (0.0, 1.0),
        "Above 0.5, the source phase-leads the target; below 0.5 it lags; 0.5 is "
        "no preferred direction.",
        is_directed=True,
    ),
    "subset_pairwise_spectral_granger_prediction": _MeasureSpec(
        "pairwise",
        "Spectral Granger prediction",
        "1",
        (0.0, _INFINITY),
        "pairwise_spectral_granger_prediction for only the requested pairs; other "
        "entries are NaN.",
        is_directed=True,
        transpose_output=True,
        requires_two_sided=True,
    ),
    "conditional_spectral_granger_prediction": _MeasureSpec(
        "pairwise",
        "Conditional spectral Granger prediction",
        "1",
        (0.0, _INFINITY),
        "Spectral Granger causality from source to target conditioned on every "
        "other signal, removing influence relayed through observed signals.",
        is_directed=True,
        transpose_output=True,
        requires_two_sided=True,
    ),
    "time_reversed_spectral_granger_prediction": _MeasureSpec(
        "pairwise",
        "Time-reversed spectral Granger prediction",
        "1",
        (0.0, _INFINITY),
        "Pairwise spectral Granger causality of the time-reversed data. Genuine "
        "directed influence reverses under time reversal; directionality that does "
        "not reverse suggests instantaneous mixing.",
        is_directed=True,
        transpose_output=True,
        requires_two_sided=True,
    ),
    # Directed-transfer-function family: opt-in (not in the default set),
    # directed (output[i, j] = influence j -> i, transposed to source -> target),
    # and returning the full (time, frequency, source, target) layout.
    "directed_transfer_function": _MeasureSpec(
        "pairwise",
        "Directed transfer function",
        "1",
        (0.0, 1.0),
        "Fraction of the target's inflow at each frequency that comes from the "
        "source, including indirect paths; sums to 1 over sources.",
        is_directed=True,
        transpose_output=True,
        requires_two_sided=True,
    ),
    "directed_coherence": _MeasureSpec(
        "pairwise",
        "Directed coherence",
        "1",
        (0.0, 1.0),
        "Noise-weighted directed transfer function: the fraction of the target's "
        "power attributable to the source; sums to 1 over sources. Assumes "
        "uncorrelated innovations.",
        is_directed=True,
        transpose_output=True,
        requires_two_sided=True,
    ),
    "partial_directed_coherence": _MeasureSpec(
        "pairwise",
        "Partial directed coherence",
        "1",
        (0.0, 1.0),
        "Direct influence from source to target, normalized by the source's total "
        "outflow; sums to 1 over targets.",
        is_directed=True,
        transpose_output=True,
        requires_two_sided=True,
    ),
    "generalized_partial_directed_coherence": _MeasureSpec(
        "pairwise",
        "Generalized partial directed coherence",
        "1",
        (0.0, 1.0),
        "Partial directed coherence with each signal scaled by its innovation "
        "variance, making it insensitive to differences in signal scale.",
        is_directed=True,
        transpose_output=True,
        requires_two_sided=True,
    ),
    "direct_directed_transfer_function": _MeasureSpec(
        "pairwise",
        "Direct directed transfer function",
        "1",
        (0.0, 1.0),
        "Direct (not relayed) influence from source to target. Normalized over all "
        "frequencies, so values are small: compare pairs, not against 1.",
        is_directed=True,
        transpose_output=True,
        requires_two_sided=True,
    ),
    "blockwise_spectral_granger_prediction": _MeasureSpec(
        "group_pairwise",
        "Blockwise spectral Granger prediction",
        "1",
        (0.0, _INFINITY),
        "Spectral Granger causality between groups of signals set by group_labels, "
        "from source_group to target_group.",
        is_directed=True,
        transpose_output=True,
        requires_two_sided=True,
    ),
    "canonical_coherence": _MeasureSpec(
        "group_pairwise",
        "Canonical coherence",
        "1",
        (0.0, 1.0),
        "Largest coherence between linear combinations of two groups of signals "
        "(historical estimator; see canonical_coherency).",
    ),
    "maximized_imaginary_coherency": _MeasureSpec(
        "group_pairwise",
        "Maximized imaginary coherency",
        "1",
        (0.0, 1.0),
        "Largest imaginary coherency between linear combinations of two groups; "
        "blind to zero-lag coupling.",
    ),
    "multivariate_interaction_measure": _MeasureSpec(
        "group_pairwise",
        "Multivariate interaction measure",
        "1",
        (0.0, _INFINITY),
        "Total phase-lagged interaction between two groups (the sum of squared "
        "imaginary-coherency components); at most the smaller group's rank.",
    ),
    "canonical_coherency": _MeasureSpec(
        "multivariate_components",
        "Canonical coherency",
        "1",
        (0.0, 1.0),
        "Complex canonical coherency per component between two groups, with the "
        "spatial filters and patterns that produce it.",
        is_complex=True,
    ),
    "maximized_imaginary_coherency_components": _MeasureSpec(
        "multivariate_components",
        "Maximized imaginary coherency",
        "1",
        (0.0, 1.0),
        "maximized_imaginary_coherency resolved into components, with the spatial "
        "filters and patterns that produce them.",
    ),
    "delay": _MeasureSpec(
        "delay",
        "Delay",
        "s",
        (-_INFINITY, _INFINITY),
        "Candidate delays in seconds, one per 2*pi phase ambiguity; the true delay is "
        "the candidate that is constant across frequency. Frequencies without "
        f"significant coherence are NaN. {_LEADS}",
        is_directed=True,
    ),
    "global_coherence": _MeasureSpec(
        "global",
        "Global coherence",
        "1",
        (0.0, 1.0),
        "Fraction of the total cross-spectral power in each component; a large "
        "leading component indicates one dominant coherent network.",
    ),
    "group_delay": _MeasureSpec(
        "group_delay",
        "Group delay",
        "s",
        (-_INFINITY, _INFINITY),
        "Delay in seconds from the slope of phase against frequency over the band; "
        f"check group_delay_r_value for the quality of the fit. {_LEADS}",
        is_directed=True,
    ),
    "phase_slope_index": _MeasureSpec(
        "phase_slope",
        "Phase slope index",
        "1",
        (-_INFINITY, _INFINITY),
        "Coherence-weighted slope of phase against frequency over the band. "
        "Unnormalized, so judge it "
        f"against a null distribution rather than a fixed threshold. {_LEADS}",
        is_directed=True,
    ),
}


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
    if name.startswith("_") or name in _NON_MEASURE_METHODS:
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
        if method not in _MEASURE_SPECS and not _is_extension_measure(method)
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


def _get_measure_spec(method: str) -> _MeasureSpec | None:
    """Return wrapper metadata for a registered measure, or None."""
    return _MEASURE_SPECS.get(method)


def _measure_label_attrs(method: str, signal_units: str | None) -> dict[str, str]:
    """``long_name``/``units`` attrs for a measure's main variable.

    Spectral densities are in (input units)^2/Hz when the input's units are
    known; otherwise they get no ``units`` rather than an invented one.
    """
    spec = _MEASURE_SPECS.get(method)
    long_name, units = (spec.long_name, spec.units) if spec else (method, "")
    if units is None:
        units = f"({signal_units})^2/Hz" if signal_units else ""
    return {"long_name": long_name, **({"units": units} if units else {})}
