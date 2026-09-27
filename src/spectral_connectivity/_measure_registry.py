"""Registry of the measures the xarray wrapper can compute, and name checks."""

import difflib
import inspect
from collections.abc import Iterable, Sequence
from dataclasses import KW_ONLY, dataclass
from typing import Literal

import numpy as np

from spectral_connectivity.connectivity import _NON_MEASURE_METHODS, Connectivity


@dataclass(frozen=True)
class _MeasureSpec:
    """One measure's wrapper contract and what its values mean.

    ``output_kind`` and the capability flags describe the result's shape and
    requirements; the remaining fields label and interpret its values.
    ``units`` follows UDUNITS spelling, with ``"1"`` marking a dimensionless
    score and ``None`` a spectral density, whose units derive from the input's
    (see :meth:`units_for`). ``value_range`` bounds the returned
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
    # The labels are keyword-only: three are strings, so a positional mix-up
    # would pass silently into results and the generated docs.
    _: KW_ONLY
    long_name: str
    units: str | None
    value_range: tuple[float, float]
    interpretation: str
    is_complex: bool = False
    is_default: bool = False
    # Scientific directionality and spectrum requirements are independent
    # capabilities. For example dPLI and PSI are directional but do not need
    # Wilson factorization.
    is_directed: bool = False
    requires_two_sided: bool = False

    @property
    def dims(self) -> tuple[str, ...]:
        """Dimensions of the main variable before band reduction or squeezing."""
        return _CATEGORY_DIMS[self.output_kind]

    def units_for(self, signal_units: str | None) -> str:
        """UDUNITS string of the values for input in ``signal_units``.

        A spectral density (``units is None``) is in ``(signal_units)^2/Hz``,
        and has no known units (``""``) when ``signal_units`` is unknown.
        """
        if self.units is not None:
            return self.units
        return f"({signal_units})^2/Hz" if signal_units else ""


def _wilson_directed_spec(
    output_kind: Literal["pairwise", "group_pairwise"],
    *,
    long_name: str,
    units: str | None,
    value_range: tuple[float, float],
    interpretation: str,
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
        is_default=is_default,
        is_directed=True,
        requires_two_sided=True,
    )


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
        long_name="Magnitude-squared coherence",
        units="1",
        value_range=(0.0, 1.0),
        interpretation="Linear coupling at each frequency: 0 is none, 1 is a perfectly consistent "
        "amplitude and phase relationship. Biased upward when trials x tapers is small.",
        is_default=True,
    ),
    "coherence_phase": _MeasureSpec(
        "pairwise",
        long_name="Coherency phase",
        units="rad",
        value_range=(-_PI, _PI),
        interpretation=f"Mean phase difference in radians. {_LEADS}",
        is_default=True,
    ),
    "debiased_squared_phase_lag_index": _MeasureSpec(
        "pairwise",
        long_name="Debiased squared phase lag index",
        units="1",
        value_range=(-1.0, 1.0),
        interpretation="Bias-corrected squared phase lag index. "
        f"{_DEBIASED} Lower bound is -1 / (n_observations - 1).",
        is_default=True,
    ),
    "debiased_squared_weighted_phase_lag_index": _MeasureSpec(
        "pairwise",
        long_name="Debiased squared weighted phase lag index",
        units="1",
        value_range=(-1.0, 1.0),
        interpretation=f"Bias-corrected squared weighted phase lag index. {_DEBIASED}",
        is_default=True,
    ),
    "imaginary_coherence": _MeasureSpec(
        "pairwise",
        long_name="Imaginary coherence (magnitude)",
        units="1",
        value_range=(0.0, 1.0),
        interpretation="Magnitude of the imaginary part of coherency; blind to zero-lag coupling "
        "such as volume conduction.",
        is_default=True,
    ),
    "pairwise_phase_consistency": _MeasureSpec(
        "pairwise",
        long_name="Pairwise phase consistency",
        units="1",
        value_range=(-1.0, 1.0),
        interpretation="Bias-free estimate of the squared phase-locking value. "
        f"{_DEBIASED} Lower bound is -1 / (n_observations - 1).",
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
    ),
    "phase_lag_index": _MeasureSpec(
        "pairwise",
        long_name="Phase lag index",
        units="1",
        value_range=(-1.0, 1.0),
        interpretation="Signed asymmetry of the phase-difference distribution; blind to zero-lag "
        f"coupling. Take the absolute value for the unsigned index. {_LEADS}",
        is_default=True,
    ),
    "phase_locking_value": _MeasureSpec(
        "pairwise",
        long_name="Phase-locking value",
        units="1",
        value_range=(0.0, 1.0),
        interpretation="Consistency of the phase difference across trials and tapers, ignoring "
        "amplitude: 0 is random, 1 is constant. Biased upward with few observations.",
        is_default=True,
    ),
    "power": _MeasureSpec(
        "power",
        long_name="Power spectral density",
        units=None,
        value_range=(0.0, _INFINITY),
        interpretation="One-sided power spectral density of each signal.",
        is_default=True,
    ),
    "weighted_phase_lag_index": _MeasureSpec(
        "pairwise",
        long_name="Weighted phase lag index",
        units="1",
        value_range=(-1.0, 1.0),
        interpretation="Phase lag index weighted by the magnitude of the imaginary cross-spectrum; "
        f"less sensitive to noise than the unweighted index. {_LEADS}",
        is_default=True,
    ),
    "coherency": _MeasureSpec(
        "pairwise",
        long_name="Coherency",
        units="1",
        value_range=(0.0, 1.0),
        interpretation="Complex coherency: its squared magnitude is coherence_magnitude and its "
        "angle is coherence_phase.",
        is_complex=True,
    ),
    "cross_spectral_density": _MeasureSpec(
        "pairwise",
        long_name="Cross-spectral density",
        units=None,
        value_range=(0.0, _INFINITY),
        interpretation="Complex, Hermitian cross-spectrum; unnormalized, so it scales with signal "
        "power. Use coherency for a normalized version.",
        is_complex=True,
    ),
    "imaginary_coherency": _MeasureSpec(
        "pairwise",
        long_name="Imaginary part of coherency",
        units="1",
        value_range=(-1.0, 1.0),
        interpretation=f"Signed imaginary part of coherency; blind to zero-lag coupling. {_LEADS}",
    ),
    "partial_coherence": _MeasureSpec(
        "pairwise",
        long_name="Partial coherence",
        units="1",
        value_range=(0.0, 1.0),
        interpretation="Magnitude-squared coherence after removing the linear influence of every "
        "other signal; near 0 for pairs coupled only through other signals.",
    ),
    "corrected_imaginary_phase_locking_value": _MeasureSpec(
        "pairwise",
        long_name="Corrected imaginary phase-locking value",
        units="1",
        value_range=(0.0, 1.0),
        interpretation="Phase locking with zero- and pi-lag contributions removed; insensitive to "
        "volume conduction.",
    ),
    "directed_phase_lag_index": _MeasureSpec(
        "pairwise",
        long_name="Directed phase lag index",
        units="1",
        value_range=(0.0, 1.0),
        interpretation="Above 0.5, the source phase-leads the target; below 0.5 it lags; 0.5 is "
        "no preferred direction.",
        is_directed=True,
    ),
    "subset_pairwise_spectral_granger_prediction": _wilson_directed_spec(
        "pairwise",
        long_name="Spectral Granger prediction",
        units="1",
        value_range=(0.0, _INFINITY),
        interpretation="pairwise_spectral_granger_prediction for only the requested pairs; other "
        "entries are NaN.",
    ),
    "conditional_spectral_granger_prediction": _wilson_directed_spec(
        "pairwise",
        long_name="Conditional spectral Granger prediction",
        units="1",
        value_range=(0.0, _INFINITY),
        interpretation="Spectral Granger causality from source to target conditioned on every "
        "other signal, removing influence relayed through observed signals.",
    ),
    "time_reversed_spectral_granger_prediction": _wilson_directed_spec(
        "pairwise",
        long_name="Time-reversed spectral Granger prediction",
        units="1",
        value_range=(0.0, _INFINITY),
        interpretation="Pairwise spectral Granger causality of the time-reversed data. Genuine "
        "directed influence reverses under time reversal; directionality that does "
        "not reverse suggests instantaneous mixing.",
    ),
    # Directed-transfer-function family (opt-in).
    "directed_transfer_function": _wilson_directed_spec(
        "pairwise",
        long_name="Directed transfer function",
        units="1",
        value_range=(0.0, 1.0),
        interpretation="Fraction of the target's inflow at each frequency that comes from the "
        "source, including indirect paths; sums to 1 over sources.",
    ),
    "directed_coherence": _wilson_directed_spec(
        "pairwise",
        long_name="Directed coherence",
        units="1",
        value_range=(0.0, 1.0),
        interpretation="Noise-weighted directed transfer function: the fraction of the target's "
        "power attributable to the source; sums to 1 over sources. Assumes "
        "uncorrelated innovations.",
    ),
    "partial_directed_coherence": _wilson_directed_spec(
        "pairwise",
        long_name="Partial directed coherence",
        units="1",
        value_range=(0.0, 1.0),
        interpretation="Direct influence from source to target, normalized by the source's total "
        "outflow; sums to 1 over targets.",
    ),
    "generalized_partial_directed_coherence": _wilson_directed_spec(
        "pairwise",
        long_name="Generalized partial directed coherence",
        units="1",
        value_range=(0.0, 1.0),
        interpretation="Partial directed coherence with each signal scaled by its innovation "
        "variance, making it insensitive to differences in signal scale.",
    ),
    "direct_directed_transfer_function": _wilson_directed_spec(
        "pairwise",
        long_name="Direct directed transfer function",
        units="1",
        value_range=(0.0, 1.0),
        interpretation="Direct (not relayed) influence from source to target. Normalized over all "
        "frequencies, so values are small: compare pairs, not against 1.",
    ),
    "blockwise_spectral_granger_prediction": _wilson_directed_spec(
        "group_pairwise",
        long_name="Blockwise spectral Granger prediction",
        units="1",
        value_range=(0.0, _INFINITY),
        interpretation="Spectral Granger causality between groups of signals set by group_labels, "
        "from source_group to target_group.",
    ),
    "canonical_coherence": _MeasureSpec(
        "group_pairwise",
        long_name="Canonical coherence",
        units="1",
        value_range=(0.0, 1.0),
        interpretation="Largest coherence between linear combinations of two groups of signals "
        "(historical estimator; see canonical_coherency).",
    ),
    "maximized_imaginary_coherency": _MeasureSpec(
        "group_pairwise",
        long_name="Maximized imaginary coherency",
        units="1",
        value_range=(0.0, 1.0),
        interpretation="Largest imaginary coherency between linear combinations of two groups; "
        "blind to zero-lag coupling.",
    ),
    "multivariate_interaction_measure": _MeasureSpec(
        "group_pairwise",
        long_name="Multivariate interaction measure",
        units="1",
        value_range=(0.0, _INFINITY),
        interpretation="Total phase-lagged interaction between two groups (the sum of squared "
        "imaginary-coherency components); at most the smaller group's rank.",
    ),
    "canonical_coherency": _MeasureSpec(
        "multivariate_components",
        long_name="Canonical coherency",
        units="1",
        value_range=(0.0, 1.0),
        interpretation="Complex canonical coherency per component between two groups, with the "
        "spatial filters and patterns that produce it.",
        is_complex=True,
    ),
    "maximized_imaginary_coherency_components": _MeasureSpec(
        "multivariate_components",
        long_name="Maximized imaginary coherency",
        units="1",
        value_range=(0.0, 1.0),
        interpretation="maximized_imaginary_coherency resolved into components, with the spatial "
        "filters and patterns that produce them.",
    ),
    "delay": _MeasureSpec(
        "delay",
        long_name="Delay",
        units="s",
        value_range=(-_INFINITY, _INFINITY),
        interpretation="Candidate delays in seconds, one per 2*pi phase ambiguity; the true delay is "
        "the candidate that is constant across frequency. Frequencies without "
        f"significant coherence are NaN. {_LEADS}",
        is_directed=True,
    ),
    "global_coherence": _MeasureSpec(
        "global",
        long_name="Global coherence",
        units="1",
        value_range=(0.0, 1.0),
        interpretation="Fraction of the total cross-spectral power in each component; a large "
        "leading component indicates one dominant coherent network.",
    ),
    "group_delay": _MeasureSpec(
        "group_delay",
        long_name="Group delay",
        units="s",
        value_range=(-_INFINITY, _INFINITY),
        interpretation="Delay in seconds from the slope of phase against frequency over the band; "
        f"check group_delay_r_value for the quality of the fit. {_LEADS}",
        is_directed=True,
    ),
    "phase_slope_index": _MeasureSpec(
        "phase_slope",
        long_name="Phase slope index",
        units="1",
        value_range=(-_INFINITY, _INFINITY),
        interpretation="Coherence-weighted slope of phase against frequency over the band. "
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


def _requires_two_sided(method: str) -> bool:
    """Whether a registered measure needs a two-sided spectrum (Wilson factorization)."""
    spec = _MEASURE_SPECS.get(method)
    return spec is not None and spec.requires_two_sided


def _is_group_measure(method: str) -> bool:
    """Whether a measure compares groups of signals, i.e. takes ``group_labels``.

    Decided from the ``Connectivity`` method's signature rather than the
    registry, so an unregistered extension measure that takes ``group_labels``
    counts too.
    """
    measure = getattr(Connectivity, method, None)
    return callable(measure) and "group_labels" in inspect.signature(measure).parameters


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
    return methods, isinstance(method, str)


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


def _measure_label_attrs(method: str, signal_units: str | None) -> dict[str, str]:
    """``long_name``/``units`` attrs; no ``units`` when they are unknown."""
    spec = _MEASURE_SPECS.get(method)
    if spec is None:
        return {"long_name": method}
    units = spec.units_for(signal_units)
    return {"long_name": spec.long_name, **({"units": units} if units else {})}
