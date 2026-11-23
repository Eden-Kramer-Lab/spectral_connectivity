"""Functions for getting connectivity measures in a labeled array format."""

import contextlib
import inspect
import warnings
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from logging import getLogger
from typing import Any, Literal

import numpy as np
import xarray as xr
from numpy.typing import DTypeLike, NDArray

from spectral_connectivity._frequency_bands import (
    _BAND_INTEGRAL_LONG_NAMES,
    _check_frequency_operations,
    _crop_frequency_range,
    _decimate_and_reduce_frequencies,
    _nyquist_frequency,
    _reduce_frequency_bands,
    _with_frequency_attrs,
)
from spectral_connectivity._input_handling import (
    _UNSET,
    _is_real_numeric_dtype,
    _SignalLabel,
    _SignalMetadata,
    _unwrap_fourier_input,
    _unwrap_xarray_input,
    _validated_signal_labels,
)
from spectral_connectivity._measure_registry import (
    _MEASURE_SPECS,
    _SINGLE_SIGNAL_MEASURES,
    _consumes_frequency_range,
    _format_value_range,
    _is_group_measure,
    _measure_description,
    _reduces_frequency,
    _requested_methods,
    _requires_two_sided,
    _warn_deprecated_measure_names,
)
from spectral_connectivity._provenance import (
    _shared_provenance_attrs,
    _with_frequency_grid_metadata,
)
from spectral_connectivity._result_formatting import (
    UnsupportedMeasureError,
    _check_method_accepts_kwargs,
    _connectivity_result_to_xarray,
)
from spectral_connectivity.connectivity import (
    _MIGRATION_GUIDE_URL,
    _ORIENTATION_WARNING_PREFIX,
    _SILENCE_ORIENTATION_WARNING,
    Connectivity,
    DirectedOrientationWarning,
    _select_measure_frequencies,
)
from spectral_connectivity.transforms import (
    Multitaper,
    _is_recording_too_short,
    _transposed_layout_error,
)
from spectral_connectivity.utils import (
    SAMPLING_FREQUENCY_ALIASES,
    _validated_flag,
    close_names,
    did_you_mean,
    sampling_frequency_hint,
    stacklevel_outside_package,
    to_numpy,
    warn_deprecated,
    warn_migration,
)

# The public API. UnsupportedMeasureError is defined in _result_formatting
# (which raises it) and re-exported here; listing it makes the API reference
# document it with the wrapper (see autosummary_ignore_module_all in docs/conf.py).
__all__ = [
    "DEFAULT_METHODS",
    "MeasureInfo",
    "UnsupportedMeasureError",
    "connectivity_to_xarray",
    "fourier_connectivity",
    "frequency_band_reduce",
    "list_measures",
    "multitaper_connectivity",
]

logger = getLogger(__name__)


#: Measures computed when ``method`` is omitted, in result-variable order.
DEFAULT_METHODS: tuple[str, ...] = tuple(
    name for name, spec in _MEASURE_SPECS.items() if spec.is_default
)


@dataclass(frozen=True)
class MeasureInfo:
    """A single connectivity measure the high-level wrapper can compute.

    Its repr shows only ``name``, ``category``, ``value_range``,
    ``is_directed`` (when true), and ``description``, so ``list_measures()``
    prints one line per measure. ``description`` is the one-line summary of
    the measure; ``interpretation`` says how to read its values. Read any
    field below as an attribute, or all of them with ``dataclasses.asdict``.

    Attributes
    ----------
    name : str
        Value to pass as ``method`` to :func:`multitaper_connectivity` or
        :func:`fourier_connectivity`, and the name of the corresponding
        :class:`~spectral_connectivity.Connectivity` method.
    category : str
        The output-shape contract, one of ``"pairwise"``, ``"power"``,
        ``"group_pairwise"``, ``"multivariate_components"``, ``"delay"``,
        ``"global"``, ``"group_delay"``, or ``"phase_slope"``.
    description : str
        One-line summary taken from the ``Connectivity`` method's docstring.
    is_default : bool
        Whether the measure is in the default set computed when ``method`` is
        omitted (see ``DEFAULT_METHODS``).
    is_directed : bool
        Whether the measure is directional (``source -> target`` asymmetric).
    requires_two_sided : bool
        Whether the measure requires a full two-sided spectrum, including
        negative-frequency bins.
    long_name : str
        Human-readable name, also the ``long_name`` attribute of the result.
    units : str
        Units of the values: ``"1"`` for a dimensionless score, ``"rad"``,
        ``"s"``, or ``"(input units)^2/Hz"`` for a spectral density.
    value_range : tuple of float
        ``(lower, upper)`` theoretical bounds of the values, or of their
        magnitude when ``is_complex``; ``inf`` marks an unbounded side. When a
        computed value can lie outside them is stated in
        ``docs/CONNECTIVITY_METRIC_RANGES.md``.
    is_complex : bool
        Whether the values are complex.
    dims : tuple of str
        Dimensions of the measure's variable in the wrapper's result, before
        band reduction or squeezing. Rich results (``canonical_coherency``,
        ``global_coherence``, ...) are Datasets whose variable named ``name``
        has these dimensions.
    interpretation : str
        How to read the values.
    diagonal_policy : str
        What a self-connection (a signal, or group, with itself) holds:
        ``"masked_nan"``, ``"undefined_nan"``, ``"constant"``,
        ``"rounding_noise"``, ``"estimated"``, or ``"not_applicable"``, as
        defined in the diagonal-policy table of
        ``docs/CONNECTIVITY_METRIC_RANGES.md``.
    diagonal_value : float or None
        The value of a ``"constant"`` diagonal (to within rounding), or the
        nominal value of a ``"rounding_noise"`` one; ``None`` otherwise.
    nan_for_zero_power_signal : bool
        Whether every entry involving a zero-power (flat) signal, the diagonal
        included, is NaN.
    sign_convention : str or None
        Which signal leads when the sign or angle shows it; ``None`` for
        measures whose sign carries no lead/lag.
    quantity : str or None
        For the coherency family, the form of the value: ``"magnitude"``,
        ``"squared_magnitude"``, ``"signed"`` (a real-valued component with
        its sign, e.g. the imaginary part of coherency), ``"phase"`` (an angle
        in radians), or ``"complex"``; ``None`` for other measures.
    formula : str or None
        For the coherency family, the definition of the value; ``None`` for
        other measures.
    """

    name: str
    category: str
    description: str
    is_default: bool
    is_directed: bool
    requires_two_sided: bool
    long_name: str
    units: str
    value_range: tuple[float, float]
    is_complex: bool
    dims: tuple[str, ...]
    interpretation: str
    diagonal_policy: str
    diagonal_value: float | None
    nan_for_zero_power_signal: bool
    sign_convention: str | None
    quantity: str | None
    formula: str | None

    def __repr__(self) -> str:
        """One line, as the class docstring describes; ``...`` marks the other fields."""
        directed = ", is_directed=True" if self.is_directed else ""
        return (
            f"MeasureInfo(name={self.name!r}, category={self.category!r}, "
            f"value_range={_format_value_range(*self.value_range)}{directed}, "
            f"description={self.description!r}, ...)"
        )


def list_measures(
    *,
    category: str | None = None,
    default_only: bool = False,
    directed: bool | None = None,
) -> list[MeasureInfo]:
    """List the connectivity measures the high-level wrapper can compute.

    This is the discovery entry point: it enumerates every valid ``method``
    name for :func:`multitaper_connectivity` and :func:`fourier_connectivity`,
    together with each measure's output category, a one-line description, and
    whether it is in the default set and/or directional, and what its values
    mean: units, range, result dimensions, and how to read the direction.
    A deprecated name (``"coherence_magnitude"``, now
    ``"magnitude_squared_coherence"``) is still accepted through 3.x but is
    not listed: it is the same measure.

    Parameters
    ----------
    category : str, optional
        Return only measures with this output category (e.g. ``"pairwise"``,
        ``"power"``, ``"group_pairwise"``). Raises ``ValueError`` for an
        unknown category.
    default_only : bool, default False
        Return only the measures computed when ``method`` is omitted.
    directed : bool, optional
        If ``True``, return only directional measures; if ``False``, only
        non-directional ones; if ``None`` (default), return both. Non-directional
        does not necessarily mean symmetric: for example, phase-valued measures
        may be antisymmetric and complex coherency is Hermitian.

    Returns
    -------
    measures : list of MeasureInfo
        One record per measure, in the wrapper's canonical order. Each
        prints as one line; ``m.description`` is the one-line summary and
        ``m.interpretation`` says how to read the values.

    Examples
    --------
    >>> from spectral_connectivity import list_measures
    >>> [m.name for m in list_measures(default_only=True)][:3]
    ['magnitude_squared_coherence', 'coherence_phase', 'debiased_squared_phase_lag_index']
    >>> next(m for m in list_measures() if m.name == "power").description
    'Return the power spectral density of the signal (one-sided by default).'
    """
    valid_categories = {spec.output_kind for spec in _MEASURE_SPECS.values()}
    if category is not None and category not in valid_categories:
        msg = (
            f"Unknown category {category!r}. Valid categories are: "
            f"{', '.join(sorted(valid_categories))}."
        )
        raise ValueError(msg)

    measures = []
    for name, spec in _MEASURE_SPECS.items():
        if default_only and not spec.is_default:
            continue
        if category is not None and spec.output_kind != category:
            continue
        if directed is not None and spec.is_directed != directed:
            continue
        measures.append(
            MeasureInfo(
                name=name,
                category=spec.output_kind,
                description=_measure_description(name),
                is_default=spec.is_default,
                is_directed=spec.is_directed,
                requires_two_sided=spec.requires_two_sided,
                long_name=spec.long_name,
                units=spec.units_for("input units"),
                value_range=spec.value_range,
                is_complex=spec.is_complex,
                dims=spec.dims,
                interpretation=spec.interpretation,
                diagonal_policy=spec.diagonal_policy,
                diagonal_value=spec.diagonal_value,
                nan_for_zero_power_signal=spec.nan_for_zero_power_signal,
                sign_convention=spec.sign_convention,
                quantity=spec.quantity,
                formula=spec.formula,
            )
        )
    return measures


def frequency_band_reduce(
    result: xr.DataArray | xr.Dataset,
    frequency_bands: Mapping[str, tuple[float, float]],
    *,
    frequency_reduction: Literal["mean", "integral"] = "mean",
    circular: bool | None = None,
) -> xr.DataArray | xr.Dataset:
    """Reduce a frequency-resolved result into labeled frequency bands.

    The ``frequency_bands`` and ``frequency_reduction`` arguments mean what they
    mean in :func:`multitaper_connectivity` and :func:`fourier_connectivity`,
    applied to an already-computed result.

    ``frequency_reduction="mean"`` averages the already-computed connectivity
    score over the bins in each band, both endpoints included, so adjacent
    bands such as ``(4, 8)`` and ``(8, 12)`` both average a bin lying at
    exactly 8 Hz. Phase is treated specially: a
    ``coherence_phase`` result uses a circular mean, while complex-valued
    measures use their ordinary complex (vector) mean.
    ``frequency_reduction="integral"``
    integrates a spectral density over ``[low, high]`` and is intentionally
    restricted to ``power`` and ``cross_spectral_density``, where it represents
    band power/covariance rather than a frequency-averaged score. Each bin
    stands for the frequency cell between the midpoints to its neighbours, and
    contributes its density times the part of that cell inside the band, so
    band edges need not fall on bins, a one-bin band is not zero, and adjacent
    bands add up to their union: a bin on their shared endpoint contributes
    half its cell to each, so no power is counted twice. On a one-sided grid
    starting at 0 Hz the DC
    bin and the Nyquist bin (present for an even FFT length) own only the
    half-cell toward their neighbour but count with a full spacing, matching
    the one-sided convention in which those two bins are not doubled;
    integrating ``power`` over a band that covers every bin therefore
    reproduces the total signal power (Parseval).

    Parameters
    ----------
    result : xarray.DataArray or xarray.Dataset
        Result with a one-dimensional ``frequency`` coordinate.
    frequency_bands : mapping of str to (float, float)
        Band names and their ``(low, high)`` bounds in the coordinate's units,
        both endpoints included. Bin ``f`` lies on bound ``b`` when
        ``abs(f - b) <= 1e-6 * gap(f) + 2 * (ulp(f) + ulp(b))``, where
        ``gap(f)`` is the distance from ``f`` to its nearest neighbouring bin
        (zero for a single bin) and ``ulp`` is the unit in the last place of
        each value in its own stored precision (float32 for a float32
        coordinate or bound, float64 for a Python number); a bound farther
        from every bin is an ordinary edge between bins. A decimal bound names
        a bin only when the bin's stored value has drifted from that decimal by
        at most one part in a million of its spacing (plus rounding). Float32
        coordinates, ``0.1 * k``, ``np.linspace``, ``rfftfreq``, and
        ``np.arange(1.0, 100.0, 0.1)`` stay within that;
        ``np.arange(1000.0, 1100.0, 0.001)``, whose bins drift by up to 2.4e-6
        of its spacing, and float32 cumulative sums do not, so for them take
        bounds from the coordinate itself.
    frequency_reduction : {"mean", "integral"}, default="mean"
        Scientifically defined reduction to apply within each band.
    circular : bool, optional
        Use a circular mean (for phase angles in radians). By default it is
        inferred per variable: ``coherence_phase`` results and variables with
        ``units="rad"`` are averaged circularly. Pass ``True``/``False`` to
        override, e.g. for a phase array whose name and attrs were removed.

    Returns
    -------
    xarray.DataArray or xarray.Dataset
        Same type as ``result`` with the ``frequency`` dimension replaced by a
        ``band`` dimension holding the band names, and the band definitions
        recorded in ``attrs["frequency_bands_json"]``. An integral is in the
        density's ``units`` without the ``/Hz`` (e.g. ``(uV)^2/Hz`` becomes
        ``(uV)^2``) and is labeled ``"Band power"`` or ``"Band cross-power"``.
        A measure variable's ``interpretation`` and ``formula``, which describe
        one frequency bin, name the reduction that made each band value (e.g.
        ``"Each value is the mean over the band's frequency bins of the
        per-frequency values. ..."``); its value range, sign convention,
        ``quantity``, and ``diagonal_policy`` hold of the band values and are
        kept.

    Notes
    -----
    Spatial filters, spatial patterns, and global-coherence vectors have an
    arbitrary sign or complex phase independently at each frequency. A Dataset
    containing those variables is therefore rejected instead of averaging them
    into a scientifically undefined band projection. Select the scalar score
    variable from the Dataset and reduce that DataArray when only band scores
    are needed.

    A band is undefined wherever any of its bins is ``NaN`` (for example an
    edge-invalid ``MorletWavelet`` bin under ``edge_mode="nan"``): both
    reductions return ``NaN`` there rather than silently reducing the valid
    bins only. When the input carries a ``valid_time_frequency`` coordinate the
    result gains a ``valid_time_band`` coordinate that is ``True`` only where
    every bin of the band had full support.

    An integral covers the cells of the bins ``result`` has. If ``result`` was
    cropped first, by ``frequency_range`` or ``sel``, the part of a band that
    lay in the cell of a dropped bin is gone: a band whose edge falls between
    the last kept bin and its dropped neighbour comes out low by that part
    (up to half a bin's cell at each end: 2.5% of a 20 Hz band per end on
    1 Hz bins). Integrate before cropping, since a band limits itself, or
    crop at least one bin wider than the bands.
    :func:`multitaper_connectivity` and :func:`fourier_connectivity`
    integrate the uncropped grid themselves when given ``frequency_range``
    together with ``frequency_bands``. Results of
    :func:`connectivity_to_xarray` (the route for
    ``ShortTimeFourierTransform``, ``Welch``, and ``MorletWavelet``) are
    integrated here, so leave its ``frequency_range`` unset for them.

    The Nyquist frequency is each variable's ``nyquist_frequency`` provenance
    attr (0.5 cycles/sample on normalized Fourier grids, half the sampling rate
    wherever that is known) or the released ``mt_nyquist_frequency``
    compatibility attr, and otherwise ``result``'s own. If none is recorded,
    the last bin of a grid starting at 0 Hz is taken to be the Nyquist bin, so
    reduce such a result before cropping it. Existing ``band``,
    ``band_lower``, ``band_upper``, or ``valid_time_band`` coordinates are
    discarded, with a warning: band reduction defines its own.
    """
    return _with_frequency_grid_metadata(
        _reduce_frequency_bands(
            result,
            frequency_bands,
            frequency_reduction=frequency_reduction,
            circular=circular,
            nyquist_frequency=_nyquist_frequency(
                result.attrs,
                np.asarray(result.coords["frequency"].values)
                if "frequency" in result.coords
                else None,
            ),
        )
    )


# The directed measures whose 2.x wrapper labels were transposed. The 2.x
# wrapper rejected every method named "directed", so the transfer-function
# measures had no 2.x labels to flip.
_LABEL_ORIENTATION_CHANGED_MEASURES = frozenset(
    {"pairwise_spectral_granger_prediction", "subset_pairwise_spectral_granger_prediction"}
)


def _warn_label_orientation_changed(methods: Sequence[str]) -> None:
    """Warn that ``sel(source, target)`` of the 2.x Granger measures flipped.

    Remove with :class:`DirectedOrientationWarning` in 3.2.
    """
    # One message per measure, so each measure's is shown once per session.
    for changed in sorted(_LABEL_ORIENTATION_CHANGED_MEASURES.intersection(methods)):
        warn_migration(
            f"{_ORIENTATION_WARNING_PREFIX}sel(source=a, target=b) of "
            f"{changed} is a -> b; 2.x returned b -> a. Review code written "
            "for 2.x that selects its results. See the migration guide: "
            f"{_MIGRATION_GUIDE_URL}. {_SILENCE_ORIENTATION_WARNING}",
            DirectedOrientationWarning,
        )


def _setting_mapping(name: str, value: Mapping[str, Any] | None) -> dict[str, Any]:
    """A copy of the ``name`` mapping argument; ``None`` is empty.

    Anything that is not a mapping (a string, or a list of pairs that ``dict``
    would accept) raises ``TypeError`` instead of being read as settings.
    """
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        example = (  # type: ignore[unreachable]  # user input
            "{'n_fft_samples': 1024}" if name == "transform_kwargs" else "{'pairs': [(0, 1)]}"
        )
        msg = (
            f"{name} must be a mapping of setting names to values, such as {example}, "
            f"got {type(value).__name__}."
        )
        raise TypeError(msg)
    return dict(value)


def _unknown_setting_message(
    problem: str, key: Any, *, sampling_frequency_given: bool = False
) -> str:
    """``"<problem> '<key>'."`` and the ``Multitaper`` or ``Connectivity``
    settings ``key`` most likely misspells.

    A suggestion that is one of ``multitaper_connectivity``'s named settings
    says so, since it goes outside ``transform_kwargs``.
    """
    message = f"{problem} {key!r}."
    if key in SAMPLING_FREQUENCY_ALIASES:
        return message + sampling_frequency_hint(sampling_frequency_given)
    close = close_names(str(key), (*_MULTITAPER_SETTINGS, *_CONNECTIVITY_SETTINGS))
    message += did_you_mean(close)
    named = [
        name for name in close if name in (*_NAMED_TRANSFORM_SETTINGS, *_CONNECTIVITY_SETTINGS)
    ]
    if named:
        role = "is a named argument" if len(named) == 1 else "are named arguments"
        message += (
            f" {', '.join(named)} {role} of multitaper_connectivity "
            f"(pass {named[0]}=... directly)."
        )
    return message


def _merge_legacy_settings(
    function: str,
    mapping_name: str,
    mapping: dict[str, Any],
    legacy: Mapping[str, Any],
) -> dict[str, Any]:
    """Settings from ``mapping_name`` and from ``function``'s legacy ``**kwargs``.

    A key given by both routes raises ``ValueError`` naming both.
    """
    for key in legacy:
        if key in mapping:
            msg = (
                f"{key} was given both inside {mapping_name} and as a keyword argument "
                f"of {function}; pass it once, in {mapping_name}."
            )
            raise ValueError(msg)
    return {**mapping, **legacy}


def _warn_legacy_settings(function: str, mapping_name: str, legacy: Mapping[str, Any]) -> None:
    """Warn once, at the caller, that ``legacy`` settings belong in ``mapping_name``.

    Called just before the wrapper returns, so a call that is rejected for any
    reason does not warn first. A deprecated ``frequencies_of_interest`` is
    left out: its own warning gives the ``frequency_range`` that replaces it,
    and moving it into the mapping would only keep a deprecated keyword.
    """
    legacy = {key: value for key, value in legacy.items() if key != "frequencies_of_interest"}
    if not legacy:
        return
    old = ", ".join(f"{key}=..." for key in legacy)
    new = ", ".join(f"{key!r}: ..." for key in legacy)
    warn_deprecated(
        f"{function}({old})",
        f"{function}({mapping_name}={{{new}}})",
        "gives the same result",
    )


def _transform_settings(
    transform_kwargs: Mapping[str, Any] | None,
    legacy: Mapping[str, Any],
    *,
    sampling_frequency_given: bool = False,
) -> dict[str, Any]:
    """The advanced ``Multitaper`` settings of a ``multitaper_connectivity`` call.

    Every key must be a ``Multitaper`` setting that is not one of the wrapper's
    named arguments, so a setting reaches the transform by exactly one route.
    """
    mapping = _setting_mapping("transform_kwargs", transform_kwargs)
    # The mapping's keys, then the plain keywords (which Python has already kept
    # from naming one of the wrapper's named arguments).
    for key, in_mapping in [
        *((key, True) for key in mapping),
        *((key, False) for key in legacy),
    ]:
        if key in _TRANSFORM_KWARGS_SETTINGS:
            continue
        if in_mapping and key in (*_NAMED_TRANSFORM_SETTINGS, *_CONNECTIVITY_SETTINGS):
            msg = (
                f"{key} was given inside transform_kwargs; it is a named argument of "
                f"multitaper_connectivity, so pass it once, as {key}=..., not in the "
                "mapping."
            )
            raise ValueError(msg)
        if in_mapping:
            msg = _unknown_setting_message(
                "transform_kwargs got an unexpected key",
                key,
                sampling_frequency_given=sampling_frequency_given,
            )
            msg += (
                f" It accepts the Multitaper settings {', '.join(_TRANSFORM_KWARGS_SETTINGS)}."
            )
        else:
            msg = _unknown_setting_message(
                "multitaper_connectivity() got an unexpected keyword argument",
                key,
                sampling_frequency_given=sampling_frequency_given,
            )
            msg += (
                " Measure settings go in connectivity_kwargs and advanced Multitaper "
                "settings in transform_kwargs."
            )
        raise TypeError(msg)
    return _merge_legacy_settings(
        "multitaper_connectivity", "transform_kwargs", mapping, legacy
    )


def _check_measure_settings(
    methods: Sequence[str],
    connectivity_kwargs: Mapping[str, Any] | None,
    *,
    function: str,
    transform_settings_hint: str | None,
    frequency_decimation: int = 1,
    frequency_bands: Mapping[str, tuple[float, float]] | None = None,
) -> None:
    """Reject a measure setting before any spectral work.

    The one check of every wrapper's measure settings (``connectivity_kwargs``,
    merged with ``connectivity_to_xarray``'s plain keywords): every requested
    measure must take every key (``group_labels`` is routed to the group
    measures separately), the frequency options must suit the measures
    (see :func:`_check_frequency_operations`), and ``power`` stays one-sided,
    as a labelled result describes it. A ``Connectivity`` setting (``dtype``
    and the Wilson factorization's) is a named argument of ``function``, not a
    measure setting. ``transform_settings_hint`` says where a transform setting
    goes instead (see :func:`_check_method_accepts_kwargs`).
    """
    settings = _setting_mapping("connectivity_kwargs", connectivity_kwargs)
    for key in settings:
        if key in _CONNECTIVITY_SETTINGS:
            msg = (
                f"{key} was given inside connectivity_kwargs; it sets up the "
                f"Connectivity object, not a measure, and is a named argument of "
                f"{function}, so pass it as {key}=..., not in the mapping."
            )
            raise ValueError(msg)
    _check_frequency_operations(
        methods,
        settings,
        frequency_decimation=frequency_decimation,
        frequency_bands=frequency_bands,
    )
    measure_settings = {key: value for key, value in settings.items() if key != "group_labels"}
    if (
        "power" in methods
        and "one_sided" in measure_settings
        and not _validated_flag("one_sided", measure_settings["one_sided"])
    ):
        # Checked for the whole request, so the answer does not depend on
        # where power stands among the methods.
        msg = (
            f"one_sided={measure_settings['one_sided']!r} is not "
            "supported for 'power' in a labelled result, which is always the "
            "one-sided density: its attributes say so and its band integrals rely "
            "on it. For the two-sided density, build the object yourself: "
            "Connectivity.from_transform(Multitaper(...)).power(one_sided=False)."
        )
        raise ValueError(msg)
    for method in methods:
        _check_method_accepts_kwargs(
            method,
            getattr(Connectivity, method),
            measure_settings,
            transform_settings_hint,
            several_measures=len(methods) > 1,
        )


def connectivity_to_xarray(
    m: Any,
    method: str = "magnitude_squared_coherence",
    signal_names: Sequence[_SignalLabel] | None = None,
    squeeze: bool = False,
    *,
    group_labels: Sequence[Hashable] | NDArray[Any] | None = None,
    frequency_range: tuple[float, float] | None = None,
    connectivity_kwargs: Mapping[str, Any] | None = None,
    dtype: DTypeLike = np.complex128,
    minimum_phase_tolerance: float = 1e-8,
    minimum_phase_max_iterations: int = 500,
    **kwargs: Any,
) -> xr.DataArray | xr.Dataset:
    """Calculate one connectivity measure and return a labeled array.

    Ordinary pairwise measures return a DataArray; component-resolved or
    multi-quantity measures return a Dataset with explicit semantic axes.

    .. versionchanged:: 3.0
       For ``pairwise_spectral_granger_prediction`` and
       ``subset_pairwise_spectral_granger_prediction``,
       ``sel(source=a, target=b)`` is ``a -> b``; 2.x returned ``b -> a``. The
       2.x wrapper rejected the transfer-function measures and could not return
       ``phase_slope_index``, ``group_delay``, or ``delay``, whose
       ``Connectivity`` arrays were already source first.

    Parameters
    ----------
    m : transform
        A spectral transform (e.g. ``Multitaper``, ``MorletWavelet``, ``Welch``)
        whose coefficients the measure is computed from. Any object satisfying
        :class:`~spectral_connectivity.transforms.SpectralTransform` works; the
        result of one that is not built in records its class name as
        ``transform`` and none of the built-in transforms' settings.
    method : str, default="magnitude_squared_coherence"
        Measure name from :func:`list_measures`.

        .. versionchanged:: 3.0
           The default is ``"magnitude_squared_coherence"``, so the default
           result is named that; 2.x defaulted to ``"coherence_magnitude"``,
           which still works when requested, with a ``DeprecationWarning``.
    signal_names : sequence, optional
        Labels for the ``source``/``target`` coordinates; defaults to
        ``"0"``, ``"1"``, ....
    squeeze : bool, default=False
        With exactly 2 signals, reduce a pairwise measure to the ordered pair
        (first source, last target), keeping ``source`` and ``target`` as
        scalar coordinates.
    group_labels : sequence, optional
        One label per signal naming its group, required by the group measures
        (see :func:`multitaper_connectivity`). It may instead be given in
        ``connectivity_kwargs``, but not both. A coordinate name is not
        accepted here: ``m`` keeps no coordinates.
    frequency_range : (float, float), optional
        Frequency interval ``(low, high)``, both endpoints included, as in
        :func:`multitaper_connectivity`: ``delay``, ``group_delay``, and
        ``phase_slope_index`` consume only those bins, and any other measure
        is computed from the full spectrum and cropped to them. Pass it here,
        not in ``connectivity_kwargs``. Leave it unset for a result whose bands
        :func:`frequency_band_reduce` will integrate: a band edge between the
        last kept bin and its dropped neighbour loses the part of the band in
        that neighbour's cell.
    connectivity_kwargs : mapping, optional
        Settings for the measure, e.g. ``{"pairs": [(0, 1)]}`` for
        ``subset_pairwise_spectral_granger_prediction`` or
        ``{"frequency_resolution": 10.0}`` for ``phase_slope_index``. A key the
        measure does not take raises ``TypeError``. There is no
        ``transform_kwargs``: ``m`` is already built.
    dtype : numpy.dtype, default=complex128
        Working precision for the connectivity computations.
    minimum_phase_tolerance : float, default=1e-8
        Relative convergence tolerance of the Wilson factorization used by the
        directed measures.
    minimum_phase_max_iterations : int, default=500
        Maximum Wilson iterations for the directed measures.
    **kwargs
        Measure settings given directly as keyword arguments.

        .. deprecated:: 3.0
           Pass them in ``connectivity_kwargs``, which gives the same result;
           this route warns and is removed in 4.0. A setting given both ways
           raises ``ValueError``.

    Returns
    -------
    xarray.DataArray or xarray.Dataset
        The labeled result with provenance in ``attrs``; see
        :func:`multitaper_connectivity` for the dimensions and orientation
        (``sel(source=a, target=b)`` is the influence ``a -> b``).

    Examples
    --------
    >>> import numpy as np
    >>> from spectral_connectivity.transforms import Multitaper
    >>> data = np.random.default_rng(0).standard_normal((100, 5, 3))
    >>> mt = Multitaper(data, sampling_frequency=1000)
    >>> connectivity_to_xarray(mt).dims
    ('time', 'frequency', 'source', 'target')
    """
    if not isinstance(method, str):
        msg = (  # type: ignore[unreachable]  # user input
            "method must be a single measure name (a string) for "
            f"connectivity_to_xarray, got {method!r}; to compute several measures "
            "call it once per measure."
        )
        raise TypeError(msg)
    (method,), output_names, _ = _requested_methods(method, ())
    measure_settings = _merge_legacy_settings(
        "connectivity_to_xarray",
        "connectivity_kwargs",
        _setting_mapping("connectivity_kwargs", connectivity_kwargs),
        kwargs,
    )
    # A misspelled or misplaced setting fails before any work, whichever route
    # it took; frequency_range is a named argument here as in every wrapper.
    _check_measure_settings(
        [method],
        measure_settings,
        function="connectivity_to_xarray",
        transform_settings_hint=None,
    )
    consumes_range = _consumes_frequency_range(method)
    if consumes_range and frequency_range is not None:
        measure_settings["frequency_range"] = frequency_range
    group_labels, measure_settings, group_methods = _resolve_group_labels(
        [method], measure_settings, group_labels
    )
    group_labels = _group_labels_from_coordinate(
        group_labels,
        None,
        no_coordinates=(
            "connectivity_to_xarray takes a transform, which keeps no coordinates: "
            f"pass the coordinate's values, da.coords[{group_labels!r}].values, or "
            "call multitaper_connectivity with the DataArray"
        ),
    )
    connectivity = Connectivity.from_transform(
        m,
        dtype=dtype,
        minimum_phase_tolerance=minimum_phase_tolerance,
        minimum_phase_max_iterations=minimum_phase_max_iterations,
    )
    # The signal-count and missing-labels messages need the signal count,
    # which a custom transform may expose only through its coefficients, so
    # they follow the FFT.
    _require_two_signals(
        [method],
        connectivity.n_signals,
        subject="The transform",
        keep_signal_axis=_KEEP_TRANSFORM_SIGNAL_AXIS,
    )
    _require_group_labels([method], group_labels, connectivity.n_signals)
    if group_methods:
        measure_settings["group_labels"] = group_labels
    provenance_metadata = getattr(m, "_provenance_metadata", None)
    if provenance_metadata is not None:
        metadata = provenance_metadata()
    else:
        # The public SpectralTransform interface has no settings record. It
        # lets a transform leave either coordinate to Connectivity, which then
        # uses normalized frequencies and window indices, not Hz and seconds.
        metadata = {"transform": type(m).__name__}
        if m.frequencies is None:
            metadata.update(
                frequency_coordinate="normalized",
                frequency_units="cycles/sample",
                nyquist_frequency=0.5,
            )
        if m.time is None:
            metadata["time_coordinate"] = "index"
    signal_labels = _validated_signal_labels(signal_names, connectivity.n_signals)
    shared_attrs = _shared_provenance_attrs(connectivity, metadata)
    # The call's interval, resolved once; the result, its crop, and its
    # validity read it.
    selection = (
        _select_measure_frequencies(
            to_numpy(connectivity.frequencies),
            frequency_range,
            measure_settings.get("frequencies_of_interest") if consumes_range else None,
        )
        if consumes_range or frequency_range is not None
        else None
    )
    result = _connectivity_result_to_xarray(
        connectivity,
        method,
        signal_labels,
        squeeze,
        shared_attrs,
        frequency_selection=selection if consumes_range else None,
        result_name=output_names[method],
        **measure_settings,
    )
    valid_time_frequency = getattr(m, "valid_time_frequency", None)
    if valid_time_frequency is not None:
        validity = to_numpy(valid_time_frequency).astype(bool)
        expected_shape = (len(connectivity.time), len(connectivity.frequencies))
        if validity.shape != expected_shape:
            msg = (
                "transform.valid_time_frequency must have shape "
                f"{expected_shape}, got {validity.shape}."
            )
            raise ValueError(msg)
        validity_attrs = {"long_name": "Full wavelet and smoothing support is in-record"}
        if "frequency" in result.dims:
            # delay keeps only the bins of its interval; every other
            # frequency-resolved result keeps the full grid until it is cropped.
            bins = selection.mask if consumes_range and selection is not None else slice(None)
            result = result.assign_coords(
                valid_time_frequency=(("time", "frequency"), validity[:, bins], validity_attrs)
            )
        elif "time" in result.dims:
            # PSI and group delay aggregate a frequency interval. They have no
            # frequency dimension on which a 2-D coordinate can live, so expose
            # whether every bin of the interval has full wavelet/smoothing
            # support at each time point. With frequency_resolution PSI sums
            # only some of those bins, so this is conservative: it can be False
            # where every bin PSI summed was valid, never True where one was not.
            assert selection is not None  # PSI and group delay consume an interval
            valid_time = validity[:, selection.mask].all(axis=1)
            result = result.assign_coords(valid_time=(("time",), valid_time, validity_attrs))
    if frequency_range is not None and not consumes_range:
        assert selection is not None  # resolved above whenever a range is given
        result = _crop_frequency_range(result, selection)
    _warn_deprecated_measure_names(output_names)
    _warn_label_orientation_changed([method])
    _warn_legacy_settings("connectivity_to_xarray", "connectivity_kwargs", kwargs)
    return _with_frequency_grid_metadata(result)


def _combine_formatted_results(
    results: Sequence[xr.DataArray | xr.Dataset],
    shared_attrs: Mapping[str, Any],
) -> xr.Dataset:
    """Merge heterogeneous formatted measures without losing sub-variables."""
    datasets = [
        result.to_dataset(name=result.name) if isinstance(result, xr.DataArray) else result
        for result in results
    ]
    try:
        combined = xr.merge(datasets, compat="no_conflicts", join="exact")
    except ValueError as error:
        msg = (
            "Requested measures produced conflicting xarray variables or "
            "coordinates; request them separately or use compatible group labels."
        )
        raise ValueError(msg) from error
    combined.attrs = dict(shared_attrs)
    return combined


def _resolve_group_labels(
    methods: Sequence[str],
    connectivity_kwargs: Mapping[str, Any] | None,
    group_labels: Sequence[Hashable] | NDArray[Any] | None,
) -> tuple[Sequence[Hashable] | NDArray[Any] | None, dict[str, Any], frozenset[str]]:
    """Resolve ``group_labels`` from the argument or ``connectivity_kwargs``.

    Needs no signal count, so every wrapper runs it before any spectral work
    and a misplaced label fails fast; :func:`_require_group_labels` checks
    for missing labels once the signal count is known.

    Parameters
    ----------
    methods : sequence of str
        The requested measure names.
    connectivity_kwargs : mapping or None
        The caller's measure keyword arguments; copied, never mutated.
    group_labels : sequence, array or None
        The ``group_labels=`` argument.

    Returns
    -------
    group_labels : sequence, array or None
        The labels from whichever form supplied them.
    label_free_kwargs : dict
        A copy of ``connectivity_kwargs`` without ``group_labels``.
    group_methods : frozenset of str
        The requested measures that take ``group_labels``.

    Raises
    ------
    ValueError
        If the labels are given both ways, or given with no group measure
        requested.
    """
    label_free_kwargs = _setting_mapping("connectivity_kwargs", connectivity_kwargs)
    legacy_labels = label_free_kwargs.pop("group_labels", None)
    if group_labels is not None and legacy_labels is not None:
        msg = (
            "group_labels was given both as an argument and inside "
            "connectivity_kwargs; pass it once, as group_labels=..."
        )
        raise ValueError(msg)
    group_labels = group_labels if group_labels is not None else legacy_labels
    group_methods = [method for method in methods if _is_group_measure(method)]
    if group_labels is not None and not group_methods:
        msg = (
            "group_labels was given, but none of the requested measures compares "
            f"groups of signals: {list(methods)!r}. Request a group measure (see "
            "list_measures(category='group_pairwise') or "
            "list_measures(category='multivariate_components')) or drop group_labels."
        )
        raise ValueError(msg)
    return group_labels, label_free_kwargs, frozenset(group_methods)


def _group_labels_from_coordinate(
    group_labels: Sequence[Hashable] | NDArray[Any] | str | None,
    signal_metadata: _SignalMetadata | None,
    *,
    no_coordinates: str,
) -> Sequence[Hashable] | NDArray[Any] | None:
    """The labels ``group_labels`` gives, reading a string as a coordinate name.

    A string names a 1-D coordinate along the input DataArray's signal
    dimension (e.g. ``"region"``), whose values label the signals. Other
    values are returned unchanged. ``no_coordinates`` says why the input has
    no such coordinate when ``signal_metadata`` is ``None``.
    """
    if not isinstance(group_labels, str):
        return group_labels
    if signal_metadata is None:
        msg = f"group_labels={group_labels!r} names a coordinate, but {no_coordinates}."
        raise TypeError(msg)
    coordinates = signal_metadata.coordinates
    if group_labels not in coordinates:
        dimension = signal_metadata.signal_dimension
        available = (
            f"those it has are {', '.join(repr(name) for name in coordinates)}"
            if coordinates
            else "it has none"
        )
        msg = (
            f"group_labels={group_labels!r} names no 1-D coordinate along the signal "
            f"dimension {dimension!r} other than its index; {available}. Name one of "
            "them, or pass one label per signal, e.g. group_labels=['CA1', 'CA1', "
            "'PFC', 'PFC']."
        )
        if group_labels == dimension:
            msg += (
                f" {dimension!r} itself labels each signal, which would make one group each."
            )
        raise ValueError(msg)
    return coordinates[group_labels][0]


# How to keep the signal axis when selecting one channel, for the layouts
# multitaper_connectivity takes, and for a transform's 3-D one.
_KEEP_SIGNAL_AXIS = (
    "When selecting one channel, keep the signal axis: x[:, [i]] rather "
    "than x[:, i] for (n_time_samples, n_signals) input, and x[:, :, [i]] "
    "rather than x[:, :, i] for (n_time_samples, n_trials, n_signals) input."
)
_KEEP_TRANSFORM_SIGNAL_AXIS = (
    "When selecting one channel of the time series for the transform, keep "
    "the signal axis: x[:, :, [i]] rather than x[:, :, i]."
)


def _require_two_signals(
    methods: Sequence[str],
    n_signals: int,
    *,
    subject: str = "The time series",
    keep_signal_axis: str = _KEEP_SIGNAL_AXIS,
) -> None:
    """Raise, in the wrapper's terms, if a measure relating signals gets one.

    ``Connectivity`` raises too, but names its ``fourier_coefficients``, which a
    caller of a wrapper never sees. ``subject`` is what has the one signal,
    and ``keep_signal_axis`` how to select a channel without dropping its
    axis, for the input the wrapper was given. Unregistered extension
    measures are left to their own checks.
    """
    pairwise = [
        name
        for name in methods
        if name in _MEASURE_SPECS and name not in _SINGLE_SIGNAL_MEASURES
    ]
    if n_signals >= 2 or not pairwise:
        return
    shown = 3
    named = ", ".join(pairwise[:shown])
    if len(pairwise) > shown:
        named += f", and {len(pairwise) - shown} more measures"
    verbs = (
        "relates pairs of signals and needs"
        if len(pairwise) == 1
        else "relate pairs of signals and need"
    )
    msg = (
        f"{subject} has {n_signals} signal, but {named} {verbs} at least "
        f"2. For the spectrum of one signal use method='power'. {keep_signal_axis}"
    )
    raise ValueError(msg)


def _require_group_labels(
    methods: Sequence[str],
    group_labels: Sequence[Hashable] | NDArray[Any] | None,
    n_signals: int,
) -> None:
    """Raise if a group measure in ``methods`` was requested without ``group_labels``.

    The message quotes ``n_signals``, the one fact
    :func:`_resolve_group_labels` does not need.
    """
    group_methods = [method for method in methods if _is_group_measure(method)]
    if group_methods and group_labels is None:
        msg = (
            f"group_labels is required for {', '.join(group_methods)}: group measures "
            "compare groups of signals and need one label per signal "
            f"({n_signals} here) naming the group each signal belongs to, e.g. "
            "group_labels=['CA1', 'CA1', 'PFC', 'PFC'].\n"
            "Pass it as group_labels=... (the same labels apply to every group "
            "measure in this call)."
        )
        raise ValueError(msg)


def _format_and_reduce_measures(
    connectivity: Connectivity,
    methods: list[str],
    *,
    return_dataarray: bool,
    signal_labels: NDArray[Any],
    squeeze: bool,
    shared_attrs: Mapping[str, Any],
    connectivity_kwargs: Mapping[str, Any],
    group_labels: Sequence[Hashable] | NDArray[Any] | None,
    group_methods: frozenset[str],
    frequency_range: tuple[float, float] | None,
    frequency_decimation: int,
    frequency_bands: Mapping[str, tuple[float, float]] | None,
    frequency_reduction: Literal["mean", "integral"],
    output_names: Mapping[str, str],
    signal_metadata: _SignalMetadata | None = None,
) -> xr.DataArray | xr.Dataset:
    """Format the requested measures to xarray and apply frequency reduction.

    ``methods`` are canonical names; ``output_names`` maps each to the name the
    caller requested, which names its result variable (see
    :func:`_requested_methods`).

    Shared tail of :func:`multitaper_connectivity` and :func:`fourier_connectivity`:
    passes the already-resolved ``group_labels`` (see :func:`_resolve_group_labels`)
    to ``group_methods`` only, honors ``squeeze`` only for a single-measure
    DataArray, formats each measure (skipping structurally-unsupported ones in a
    multi-measure batch), merges the survivors, and applies any frequency
    crop/decimation/band reduction. ``frequency_range`` is passed to the
    measures that consume an interval and crops each other result before they
    are merged, so every variable keeps the same bins. ``connectivity_kwargs``
    must be label-free, and the callers check it first (see
    :func:`_check_measure_settings`).
    """
    frequencies = to_numpy(connectivity.frequencies)
    consuming_methods = frozenset(m for m in methods if _consumes_frequency_range(m))
    # One selection per call, on the full grid: the interval measures' results
    # and attributes and the crop of every other result read it.
    selection = _select_measure_frequencies(
        frequencies,
        frequency_range,
        connectivity_kwargs.get("frequencies_of_interest") if consuming_methods else None,
    )

    # An integral leaves its densities uncropped, so a band inside the range
    # also integrates the part of it lying in the cell of a bin just outside
    # the range. A request the reduction refuses is cropped as any other.
    kept = (
        (selection.mask, selection.requested)
        if frequency_range is not None
        and selection.requested is not None
        and frequency_bands is not None
        and frequency_reduction == "integral"
        and all(m in _BAND_INTEGRAL_LONG_NAMES or _reduces_frequency(m) for m in methods)
        else None
    )

    def measure_kwargs(method: str) -> Mapping[str, Any]:
        """Keyword arguments for one measure: ``group_labels`` reaches group
        measures only, and ``frequency_range`` the measures that consume it."""
        kwargs = dict(connectivity_kwargs)
        if method in group_methods:
            kwargs["group_labels"] = group_labels
        if frequency_range is not None and method in consuming_methods:
            kwargs["frequency_range"] = frequency_range
        return kwargs

    if squeeze and not return_dataarray:
        # squeeze reduces a pairwise measure to a (time, frequency) array whose
        # source/target become scalar coordinates; in a Dataset those scalars are
        # shared across variables and collide with a sibling's axes, so squeeze is
        # honored only for a single-method DataArray.
        warnings.warn(
            "squeeze=True is ignored for multi-measure results (a Dataset); "
            "request a single method (a string) to get a squeezed DataArray.",
            UserWarning,
            stacklevel=stacklevel_outside_package(),
        )
        squeeze = False

    def formatted(method: str, squeeze: bool) -> xr.DataArray | xr.Dataset:
        """One measure's labeled result, cut to the call's ``frequency_range``."""
        consumes_range = method in consuming_methods
        result = _connectivity_result_to_xarray(
            connectivity,
            method,
            signal_labels,
            squeeze,
            shared_attrs,
            signal_metadata=signal_metadata,
            frequency_selection=selection if consumes_range else None,
            result_name=output_names[method],
            **measure_kwargs(method),
        )
        if frequency_range is None or consumes_range:
            return result
        if kept is not None:
            return _with_frequency_attrs(result, **selection.attrs())
        return _crop_frequency_range(result, selection)

    if return_dataarray:
        result: xr.DataArray | xr.Dataset = formatted(methods[0], squeeze)
    else:
        formatted_results: list[xr.DataArray | xr.Dataset] = []
        for this_method in methods:
            try:
                formatted_results.append(formatted(this_method, False))
            except UnsupportedMeasureError as error:  # noqa: PERF203 -- per-measure skip
                # A measure whose result shape does not fit the xarray layout can
                # be skipped in a batch. _connectivity_result_to_xarray raises
                # UnsupportedMeasureError from its shape check after the measure
                # has run (an unregistered extension returning a non-pairwise
                # shape); a genuine NotImplementedError is not caught, so a broken
                # measure fails loudly instead of silently vanishing from the
                # Dataset.
                if len(methods) == 1:
                    raise
                logger.warning("Skipping %s: %s", this_method, error)
        if not formatted_results:
            msg = (
                "None of the requested methods produced a compatible result "
                f"for the xarray interface: {methods!r}."
            )
            raise UnsupportedMeasureError(msg)
        result = _combine_formatted_results(formatted_results, shared_attrs)

    if signal_metadata is not None and signal_metadata.recording_coordinates:
        # Band reduction warns itself before replacing a band coordinate.
        reserved = set(result.coords) | set(result.dims)
        if isinstance(result, xr.Dataset):
            reserved.update(result.data_vars)
        elif result.name is not None:
            reserved.add(result.name)
        dropped = sorted(set(signal_metadata.recording_coordinates) & reserved)
        if dropped:
            warnings.warn(
                f"Input scalar coordinates {dropped} share names with coordinates "
                "or variables of the result and were not carried over. Rename them "
                "before the call to keep them.",
                UserWarning,
                stacklevel=stacklevel_outside_package(),
            )
        result = result.assign_coords(
            {
                name: ((), values, attrs)
                for name, (values, attrs) in signal_metadata.recording_coordinates.items()
                if name not in reserved
            }
        )
    return _decimate_and_reduce_frequencies(
        result,
        frequency_decimation=frequency_decimation,
        frequency_bands=frequency_bands,
        frequency_reduction=frequency_reduction,
        # The computed grid's, before the crop can make an interior bin the last.
        nyquist_frequency=_nyquist_frequency(shared_attrs, frequencies),
        kept=kept,
    )


def multitaper_connectivity(
    time_series: NDArray[np.floating] | xr.DataArray,
    sampling_frequency: float | None = None,
    time_window_duration: float | None = None,
    method: str | list[str] | None = None,
    signal_names: Sequence[_SignalLabel] | None = None,
    squeeze: bool = False,
    connectivity_kwargs: dict[str, Any] | None = None,
    *,
    time_window_step: float | None = None,
    time_halfbandwidth_product: float = 3,
    n_tapers: int | None = None,
    detrend_type: str | None = "constant",
    transform_kwargs: Mapping[str, Any] | None = None,
    group_labels: Sequence[Hashable] | NDArray[Any] | str | None = None,
    frequency_range: tuple[float, float] | None = None,
    frequency_decimation: int = 1,
    frequency_bands: Mapping[str, tuple[float, float]] | None = None,
    frequency_reduction: Literal["mean", "integral"] = "mean",
    time_dim: Hashable | None = None,
    trial_dim: Hashable | None = None,
    signal_dim: Hashable | None = None,
    dtype: DTypeLike = np.complex128,
    minimum_phase_tolerance: float = 1e-8,
    minimum_phase_max_iterations: int = 500,
    **kwargs: Any,
) -> xr.DataArray | xr.Dataset:
    """
    Compute connectivity measures with multitaper spectral estimation.

    This is the main high-level function for connectivity analysis. It performs
    multitaper spectral analysis on the input time series and computes the
    requested connectivity measures, returning results as labeled xarray objects.

    .. versionchanged:: 3.0
       For ``pairwise_spectral_granger_prediction`` and
       ``subset_pairwise_spectral_granger_prediction``,
       ``sel(source=a, target=b)`` is ``a -> b``; 2.x returned ``b -> a``. The
       2.x wrapper rejected the transfer-function measures and could not return
       ``phase_slope_index``, ``group_delay``, or ``delay``, whose
       ``Connectivity`` arrays were already source first.

    Parameters
    ----------
    time_series : NDArray[floating] or xarray.DataArray,
        shape (n_time_samples, n_trials, n_signals) or (n_time_samples, n_signals)
        Time series data, time first. In a 3-D array **trials are the middle
        axis and signals the last**: ``x[:, k, :]`` is trial ``k``. An array of
        shape ``(500, 3, 40)`` is read as 500 time samples of 3 trials of 40
        signals; if it holds 3 signals and 40 trials, pass
        ``np.swapaxes(x, 1, 2)``. Array input cannot be checked for this (a
        DataArray's dimension names can). A 2-D array is one trial. For
        multiple trials, trials are averaged in spectral
        domain. For a DataArray, common time/trial/signal dimension names are
        inferred and transposed automatically; use ``time_dim``, ``trial_dim``,
        and ``signal_dim`` for domain-specific names. Ambiguous names raise rather
        than falling back to dimension position, though a single unrecognized
        dimension left for the one remaining role is assigned by elimination with
        a warning (a spectral name such as ``frequency`` or ``band`` is rejected
        instead, since it marks an already-transformed input). A dask-backed
        DataArray is rejected (materialize it first with
        ``DataArray.compute()``). A numeric time index is
        interpreted as elapsed seconds (a ``sample`` index as sample numbers)
        and used to label output window centers. When ``sampling_frequency`` is
        given it is validated against the index; when it is omitted, an
        elapsed-seconds ``time`` coordinate infers it (a ``sample`` index cannot,
        having no time scale). Datetime, timedelta, and object-valued time
        coordinates are not yet supported and must first be converted to numeric
        elapsed seconds.
        When ``signal_names`` is omitted,
        labels from a 1-D index coordinate on the signal dimension are carried to
        the output's ``source`` and ``target`` coordinates without changing their
        type; if such labels are present but unusable a warning is issued and
        default string labels are used.
    sampling_frequency : float, optional
        Sampling rate in Hz of the time series data. Required for array input;
        for a DataArray it may be omitted and inferred from a sufficiently precise
        numeric elapsed-seconds ``time`` coordinate. Pass it explicitly when a
        low-precision or large-offset coordinate cannot resolve the rate reliably.
        An inferred rate of at most 1 Hz, or a coordinate whose ``units``
        attribute says milliseconds, warns that the coordinate may be in
        milliseconds (read as seconds, it gives a rate 1000 times too low);
        passing the rate, or ``units="s"`` on the coordinate for a slow
        recording, silences it. A millisecond coordinate of data sampled faster
        than 1 kHz infers a rate above 1 Hz and is not detected unless its
        ``units`` attribute says milliseconds: set ``units`` or pass the rate.
    time_window_duration : float, optional
        Duration of sliding window in seconds for time-resolved analysis.
        If None, analyzes entire time series (no time resolution).
    method : str or list of str, optional
        Connectivity method(s) to compute. If None, computes the default set of
        real-valued measures that fit the xarray/NetCDF interface (see
        ``DEFAULT_METHODS``) — not every measure. ``coherency`` is left out of the
        default because complex arrays are not portably serializable across all
        supported xarray versions and NetCDF engines, but it can be requested by
        name. Every directed measure is opt-in by name, including
        ``pairwise_spectral_granger_prediction`` and the
        directed-transfer-function family (``directed_transfer_function``,
        ``directed_coherence``, ``partial_directed_coherence``,
        ``generalized_partial_directed_coherence``,
        ``direct_directed_transfer_function``); the spectral Granger and
        transfer-function measures factorize the spectrum and dominate the cost
        (see the Notes on directed orientation). Measures with nonstandard layouts,
        including ``global_coherence``, ``phase_slope_index``, ``group_delay``,
        ``delay``, ``canonical_coherence``, and blockwise spectral Granger, are
        available by name and return labeled DataArrays or Datasets with their
        component, group, candidate-delay, or frequency-reduced dimensions.
        ``"global_coherence"`` here pools each time window's trials and tapers;
        to pool the windows too, use
        ``Connectivity.from_transform(transform,
        expectation_type="time_trials_tapers").global_coherence()`` (see
        :meth:`Connectivity.global_coherence`).
        Examples:
        "magnitude_squared_coherence", "imaginary_coherence", "phase_locking_value".
        The 2.x name ``"coherence_magnitude"`` still works through 3.x with a
        ``DeprecationWarning`` and keeps that variable name; it is removed in 4.0.
    signal_names : sequence of scalar, optional
        Scalar, non-missing, unique xarray-compatible coordinate labels for signal
        channels. Integer labels must fit the signed 32-bit range for portable
        NetCDF3 serialization. Nested or structured labels are not supported. If
        None, uses the DataArray signal index when available, otherwise stringified
        indices.
    squeeze : bool, default=False
        Only honored when a single ``method`` (a string) is requested, so the
        result is a DataArray. If there are exactly 2 channels, reduce a pairwise
        measure to the single ordered pair (first source, last target), returning
        a ``(time, frequency)`` array whose selected ``source`` and ``target`` are
        retained as scalar coordinates -- so the pair (and, for directed measures,
        the direction) is still recorded. With more than 2 channels a warning is
        issued and the full matrix is returned; for ``power`` (no target axis)
        squeeze is a no-op. For multi-measure requests (which return a Dataset,
        whose variables can have incompatible axes such as ``power``'s), squeeze
        is ignored with a warning.
    connectivity_kwargs : dict, optional
        Settings for the *measure* (a ``Connectivity`` method), e.g. ``pairs``
        for ``subset_pairwise_spectral_granger_prediction`` or ``n_components``
        for ``canonical_coherency``; passed to every requested measure, so a
        key a requested measure does not take raises ``TypeError``. Transform
        settings do not go here (see ``transform_kwargs``).
    time_window_step : float, optional
        Step between successive windows in seconds, rounded to the nearest
        sample; defaults to the window duration (no overlap).
    time_halfbandwidth_product : float, default=3
        DPSS time-half-bandwidth product ``NW``; the concentration bandwidth is
        ``2 * NW / time_window_duration`` Hz.
    n_tapers : int, optional
        Number of DPSS tapers to request; defaults to ``floor(2 * NW) - 1``.
        Only about ``2 * NW - 1`` are well concentrated: a taper whose spectral
        concentration is at or below 0.9 is dropped, with a warning, unless
        ``transform_kwargs={"is_low_bias": False}``. The result's
        ``mt_n_tapers`` and ``n_observations`` count the tapers used.
    detrend_type : {"constant", "linear"} or None, default="constant"
        Detrending applied to each window before the FFT; ``None`` for none.
    transform_kwargs : mapping, optional
        Advanced ``Multitaper`` settings by name, e.g. ``{"n_fft_samples": 1024,
        "start_time": 0.5, "taper_weighting": "adaptive", "fft_workers": -1}``.
        Every key must be a ``Multitaper`` setting (a misspelled one raises
        ``TypeError``). The common settings above are named arguments of this
        function and cannot also be given here (``ValueError``), so each
        setting has one value.
    group_labels : sequence or str, optional
        One label per signal naming the group it belongs to; labels are scalars
        such as integers or area names (``"CA1"``), and missing values
        (``None``, NaN) are rejected. Required by
        the group measures (``canonical_coherence``, ``canonical_coherency``,
        ``maximized_imaginary_coherency``, ``multivariate_interaction_measure``,
        ``blockwise_spectral_granger_prediction`` and
        ``maximized_imaginary_coherency_components``); an error is raised if no
        requested measure takes it (labels are passed only to the measures that
        do). It may instead be given in ``connectivity_kwargs``, but not both.
        A single string, rather than a sequence, names a coordinate along the
        input DataArray's signal dimension whose values are the labels, e.g.
        ``group_labels="region"`` for a ``region`` coordinate on ``channel``.
    frequency_range : (float, float), optional
        Frequency interval ``(low, high)``, both endpoints included; a bin lies
        on a bound when they differ by at most one part in a million of the
        bin's spacing, plus rounding (see :func:`frequency_band_reduce`).
        ``delay``, ``group_delay``, and
        ``phase_slope_index`` receive it as their own ``frequency_range`` and
        consume only those bins; every other measure is computed from the full
        spectrum (Wilson-factorized measures factorize all of it) and then
        cropped to the same bins. Pass it here, not in ``connectivity_kwargs``.
    frequency_decimation : int, default=1
        Keep every Nth frequency bin after applying ``frequency_range``.
        ``phase_slope_index`` and ``group_delay`` have no frequency axis and are
        left as computed over ``frequency_range``; to make them combine fewer
        bins, pass ``connectivity_kwargs={"frequency_resolution": ...}``.
    frequency_bands : mapping of str to (float, float), optional
        Reduce the selected bins into named bands, both endpoints included, so
        adjacent bands sharing an endpoint both average a bin lying on it. With
        ``frequency_reduction="mean"``, scores are averaged, complex measures
        use a complex vector mean, and ``coherence_phase`` uses a circular mean.
        ``delay`` is frequency-resolved and is averaged like any score;
        ``phase_slope_index`` and ``group_delay`` are left as computed over
        ``frequency_range`` (call once per band, with ``frequency_range``, for
        band values). Requesting ``frequency_bands`` or ``frequency_decimation``
        with only those two measures raises ``ValueError``.
    frequency_reduction : {"mean", "integral"}, default="mean"
        Band reduction. Integration is restricted to ``power`` and
        ``cross_spectral_density``, where it yields band power/covariance;
        adjacent band integrals add up to their union without counting a
        shared edge bin twice (see :func:`frequency_band_reduce`). A band
        inside ``frequency_range`` integrates the same with and without the
        range, even when its edge falls between bins (given
        ``frequency_decimation=1``; otherwise the kept bins start at the first
        bin in the range). A band reaching past the range is cut at the
        range's bounds, or at the outer edge of the first or last selected
        bin's cell where that lies further out.
    time_dim : hashable, optional
        DataArray dimension containing time samples. Common names such as
        ``"time"`` and ``"sample"`` are inferred automatically.
    trial_dim : hashable, optional
        DataArray dimension containing trials or epochs. Required for a 3-D
        DataArray when its role cannot be inferred unambiguously.
    signal_dim : hashable, optional
        DataArray dimension containing signals or channels. Common names such as
        ``"signal"`` and ``"channel"`` are inferred automatically.
    dtype : numpy.dtype, default=complex128
        Working precision for the connectivity computations.
    minimum_phase_tolerance : float, default=1e-8
        Relative convergence tolerance of the Wilson factorization used by the
        directed measures.
    minimum_phase_max_iterations : int, default=500
        Maximum Wilson iterations for the directed measures.
    **kwargs
        Advanced ``Multitaper`` settings given directly as keyword arguments.

        .. deprecated:: 3.0
           Pass them in ``transform_kwargs``, which gives the same result;
           this route warns and is removed in 4.0. A setting given both ways
           raises ``ValueError``, and a name that is not a ``Multitaper``
           setting raises ``TypeError``. Measure settings do not go here (see
           ``connectivity_kwargs``).

    Returns
    -------
    result : xarray.DataArray or xarray.Dataset
        A plain single-quantity method returns a DataArray. Component-resolved
        and multi-quantity methods return a Dataset even when requested alone;
        multiple methods are merged into one Dataset without flattening their
        semantic dimensions.

        Tip: ``method="magnitude_squared_coherence"`` returns a single
        DataArray, while the default ``method=None`` returns a Dataset of
        every measure in ``DEFAULT_METHODS``: select one with
        ``result["magnitude_squared_coherence"]``. Without
        ``time_window_duration`` the whole recording is one window, so
        ``time`` is a dimension of length 1: drop it for plotting with
        ``result.isel(time=0)`` or ``result.squeeze("time")``.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> # Generate coupled oscillator data
    >>> t = np.arange(0, 1, 1/500)  # 500 Hz, 1 second
    >>> sig1 = np.sin(2*np.pi*10*t) + 0.1*rng.standard_normal(len(t))
    >>> sig2 = np.sin(2*np.pi*10*t + np.pi/4) + 0.1*rng.standard_normal(len(t))
    >>> # Shape (n_time, n_channels); a single trial of 2 signals. The 2-D form
    >>> # is promoted to a single-trial 3-D array internally.
    >>> data = np.stack([sig1, sig2], axis=-1)  # (500, 2)
    >>>
    >>> # Compute coherence
    >>> coherence = multitaper_connectivity(
    ...     data, sampling_frequency=500,
    ...     method="magnitude_squared_coherence",
    ...     signal_names=["Signal_1", "Signal_2"]
    ... )
    >>> coherence.dims
    ('time', 'frequency', 'source', 'target')
    >>> coherence.sizes["time"]  # one window: the whole recording
    1
    >>> coherence.squeeze("time").dims  # e.g. for plotting against frequency
    ('frequency', 'source', 'target')

    >>> # Compute multiple measures
    >>> measures = multitaper_connectivity(
    ...     data, sampling_frequency=500,
    ...     method=["magnitude_squared_coherence", "imaginary_coherence"]
    ... )
    >>> list(measures.data_vars)
    ['magnitude_squared_coherence', 'imaginary_coherence']

    >>> # An xarray.DataArray labels axes by dimension name and can supply the
    >>> # sampling rate and channel labels itself (no sampling_frequency needed).
    >>> import xarray as xr
    >>> da = xr.DataArray(
    ...     data,
    ...     dims=("time", "channel"),
    ...     coords={"time": t, "channel": ["Signal_1", "Signal_2"]},
    ... )
    >>> coherence = multitaper_connectivity(da, method="magnitude_squared_coherence")
    >>> coherence.coords["source"].values.tolist()
    ['Signal_1', 'Signal_2']

    Notes
    -----
    Uses multitaper spectral estimation for robust power spectral density
    estimation before computing connectivity measures. This provides better
    spectral estimates than single-taper methods, especially for short time series.

    For directed measures (e.g. ``pairwise_spectral_granger_prediction``) the
    ``source`` and ``target`` axes are oriented so that
    ``result.sel(source=a, target=b)`` is the influence *from* ``a`` *to* ``b``,
    the same order as the underlying ``Connectivity`` arrays, where
    ``result[..., i, j]`` is ``i -> j``. Signed undirected phase measures
    (``coherence_phase``, ``imaginary_coherency``, ``phase_lag_index``,
    ``weighted_phase_lag_index``) are positive at ``sel(source=a, target=b)``
    when ``a`` leads ``b``.

    Every measure variable has a ``long_name``. Known units are recorded as
    ``units`` (``"1"`` for
    dimensionless scores, ``"rad"`` for phase, ``"s"`` for delay; spectral
    densities are ``"(<units>)^2/Hz"`` when an input DataArray states its
    ``units``, and ``"(<units>)^2"`` once integrated over a band). Power and
    cross-spectral density omit ``units`` when input units are unknown. Non-index
    coordinates on an input DataArray's signal dimension (e.g. ``region``) are
    carried as ``source_<name>``/``target_<name>`` with their attrs unchanged,
    so those attrs must be NetCDF-serializable to save the result.

    Each measure variable also records how to read it, from the same record
    :func:`list_measures` returns: ``interpretation``, ``sign_convention``
    (measures whose sign or angle says which signal leads),
    ``value_range_lower``/``value_range_upper`` (theoretical bounds, see
    ``docs/CONNECTIVITY_METRIC_RANGES.md``; an unbounded side omitted;
    ``value_range_applies_to="magnitude"`` for complex measures), ``quantity``
    and ``formula`` (the coherency family), and ``diagonal_policy``, what a
    self-connection holds (defined in ``docs/CONNECTIVITY_METRIC_RANGES.md``);
    a policy with a nominal value adds ``diagonal_value``. ``nan_for_zero_power_signal`` (0 or 1) says whether
    every entry involving a zero-power signal is NaN.
    ``group_delay``'s other value variables,
    ``group_delay_slope`` and ``group_delay_r_value``, record their own.

    ``power`` has a ``source`` axis, the one it shares with the pairwise
    measures in a Dataset; component filters and patterns index channels
    along ``signal`` because they cover both sides of a connection. Use
    ``power.rename(source="signal")`` for a standalone channel axis.

    Real-valued results write with any NetCDF engine (booleans are stored as
    0/1). Complex results (``coherency``, ``cross_spectral_density``,
    ``canonical_coherency``, and the global-coherence vectors) need an engine
    that stores complex data, e.g.
    ``result.to_netcdf("result.h5", engine="h5netcdf", invalid_netcdf=True)``,
    or netCDF4 >= 1.7 with xarray >= 2024.10 and
    ``engine="netcdf4", auto_complex=True`` (open with the same option).

    The result records provenance as NetCDF-safe attributes so a saved file is
    self-describing:

    - ``output_schema_version=1``, ``transform``, ``sampling_frequency``,
      ``n_trials``, ``n_signals``, ``n_observations``, and independence flags --
      common known facts. ``n_observations`` is the actual raw expectation count,
      not an effective independent sample size.
    - ``transform_parameters_json`` -- remaining settings in ``estimator`` and
      ``execution`` records, with optional values stored as JSON ``null``.
    - ``spectral_bandwidth`` with its definition and units -- full DPSS
      concentration bandwidth when using generated DPSS tapers. It is distinct
      from ``frequency_bin_spacing`` on a uniform returned grid; cropping and
      decimation update spacing, and band reduction removes
      ``frequency_bin_spacing``. Selecting bins
      afterward with xarray keeps the attribute unchanged, so recompute spacing
      from the ``frequency`` coordinate after such a selection.
    - The 15 scalar ``mt_*`` attributes released in 2.0.1 -- deprecated
      compatibility copies through 3.x, removed in 4.0. The supported
      ``Multitaper.frequency_resolution`` property and parameter helpers retain
      their names and values. New transform/execution settings get no copies.
    - ``measure`` and ``measure_kwargs_json`` -- the measure name as requested
      and a canonical, JSON-normalized representation of its keyword
      arguments. A result requested by the deprecated ``"coherence_magnitude"``
      records that name, and its variable has that name too.
    - ``arg_<key>`` / ``arg_<key>_json`` -- each measure keyword argument
      individually for quick inspection; a scalar is stored as-is under
      ``arg_<key>``, while a structured or non-finite value is stored as a JSON
      string under ``arg_<key>_json`` (parse with ``json.loads``;
      ``measure_kwargs_json`` is the canonical record).
    - ``package``, ``package_version``, ``backend``, ``expectation_type`` --
      software provenance; ``backend`` is ``"cpu"`` or ``"gpu"``.
    - ``input_attrs_json`` -- a canonical, JSON-normalized record of attributes
      carried over from an input ``xarray.DataArray`` (e.g. subject or session
      metadata). Keeping the complete mapping in one record preserves arbitrary
      keys without collisions or invalid NetCDF attribute names.

    Unambiguous scalar recording coordinates (e.g. ``subject``/``session``)
    retain their values and attrs; one named like a coordinate or variable of
    the result (e.g. ``frequency``, or ``band`` when ``frequency_bands`` is
    given) is not carried over, with a warning. Labels along averaged trial
    axes are omitted.

    JSON records are canonical for scalar, numpy, mapping, and sequence values.
    A value outside those kinds is recorded best-effort via its ``repr``, which
    may embed a memory address and is therefore not guaranteed reproducible
    across runs.

    References
    ----------
    .. [1] Thomson, D. J. (1982). Spectrum estimation and harmonic analysis.
           Proceedings of the IEEE, 70(9), 1055-1096.
    .. [2] Percival, D. B., & Walden, A. T. (1993). Spectral Analysis for Physical
           Applications: Multitaper and Conventional Univariate Techniques.
    """
    transform_settings = _transform_settings(
        transform_kwargs, kwargs, sampling_frequency_given=sampling_frequency is not None
    )
    explicit_start_time = transform_settings.get("start_time", _UNSET)
    (
        time_series_data,
        signal_names,
        inferred_sampling_frequency,
        inferred_start_time,
        input_attrs,
        signal_metadata,
    ) = _unwrap_xarray_input(
        time_series,
        signal_names,
        sampling_frequency,
        time_dim=time_dim,
        trial_dim=trial_dim,
        signal_dim=signal_dim,
        explicit_start_time=explicit_start_time,
    )
    if inferred_sampling_frequency is not None:
        sampling_frequency = inferred_sampling_frequency
    if sampling_frequency is None:
        msg = (
            "sampling_frequency is required unless the input is an "
            "xarray.DataArray with a numeric 'time' coordinate (in elapsed "
            "seconds) to infer it from."
        )
        raise ValueError(msg)
    if inferred_start_time is not None and explicit_start_time is _UNSET:
        transform_settings["start_time"] = inferred_start_time
    # The default set (DEFAULT_METHODS) is portably serializable; complex,
    # component/group, frequency-reduced, and directed-transfer-function results
    # remain opt-in by name.
    method, output_names, return_dataarray = _requested_methods(method, DEFAULT_METHODS)
    # Every measure setting is checked before the transform runs its FFT.
    _check_measure_settings(
        method,
        connectivity_kwargs,
        function="multitaper_connectivity",
        transform_settings_hint=(
            "pass it to multitaper_connectivity: by name for the common settings "
            f"({', '.join(_NAMED_TRANSFORM_SETTINGS)}), otherwise in transform_kwargs"
        ),
        frequency_decimation=frequency_decimation,
        frequency_bands=frequency_bands,
    )
    group_labels, connectivity_kwargs, group_methods = _resolve_group_labels(
        method, connectivity_kwargs, group_labels
    )
    group_labels = _group_labels_from_coordinate(
        group_labels,
        signal_metadata,
        no_coordinates=(
            "time_series is not an xarray.DataArray, so it has none: pass one label "
            "per signal, e.g. group_labels=['CA1', 'CA1', 'PFC', 'PFC']"
        ),
    )
    if time_series_data.ndim not in (2, 3):
        # Said here, in this function's layouts: Multitaper's message speaks
        # of its own 3-D input and prepare_time_series.
        n_dimensions = time_series_data.ndim
        msg = (
            f"time_series has {n_dimensions} dimension{'' if n_dimensions == 1 else 's'}, "
            f"shape {tuple(time_series_data.shape)}. multitaper_connectivity needs "
            "(n_time_samples, n_signals) or (n_time_samples, n_trials, n_signals)."
        )
        if n_dimensions == 1:
            msg += " For one signal use x[:, np.newaxis] with method='power'."
        raise ValueError(msg)
    # Accept the documented (n_times, n_channels) 2-D form by inserting a
    # singleton trial axis; Multitaper requires 3-D (n_times, n_trials,
    # n_signals).
    if time_series_data.ndim == 2:
        time_series_data = time_series_data[:, np.newaxis, :]
    try:
        m = Multitaper(
            time_series=time_series_data,
            sampling_frequency=sampling_frequency,
            time_window_duration=time_window_duration,
            time_window_step=time_window_step,
            time_halfbandwidth_product=time_halfbandwidth_product,
            n_tapers=n_tapers,
            detrend_type=detrend_type,
            **transform_settings,
        )
    except ValueError as error:
        # Multitaper's layout error is the one caused by a recording too short
        # for its window; it names only its own 3-D layout, and this function
        # takes 2-D too.
        if not _is_recording_too_short(error.__cause__):
            raise
        n_time, _, n_signals = time_series_data.shape
        raise _transposed_layout_error(
            n_time,
            n_signals,
            "(n_time_samples, n_trials, n_signals) or (n_time_samples, n_signals)",
        ) from error.__cause__
    # Check the signal count and missing labels before Connectivity.from_transform
    # runs the FFT. The constructor has validated the input shape, so m.n_signals
    # is the true signal count.
    _require_two_signals(method, m.n_signals)
    _require_group_labels(method, group_labels, m.n_signals)
    # Capture metadata and build the shared calculation object from the same
    # immutable transform. The private formatter below never accepts a separate
    # Multitaper, so data and labels cannot be paired accidentally.
    metadata = m._provenance_metadata()
    shared_connectivity = Connectivity.from_transform(
        m,
        dtype=dtype,
        minimum_phase_tolerance=minimum_phase_tolerance,
        minimum_phase_max_iterations=minimum_phase_max_iterations,
    )
    # Validate labels and build shared provenance once; both are invariant across
    # the requested measures.
    signal_labels = _validated_signal_labels(signal_names, shared_connectivity.n_signals)
    shared_attrs = _shared_provenance_attrs(
        shared_connectivity, metadata, input_attrs=input_attrs
    )
    # NaN or Inf in the time series was reported once, by Multitaper; the
    # measures' arithmetic on the values it propagates would add NumPy
    # RuntimeWarnings ("invalid value encountered in divide") about the same
    # values.
    with (
        contextlib.nullcontext()
        if m._time_series_is_finite
        else np.errstate(invalid="ignore", divide="ignore")
    ):
        result = _format_and_reduce_measures(
            shared_connectivity,
            method,
            return_dataarray=return_dataarray,
            signal_labels=signal_labels,
            squeeze=squeeze,
            shared_attrs=shared_attrs,
            connectivity_kwargs=connectivity_kwargs,
            group_labels=group_labels,
            group_methods=group_methods,
            frequency_range=frequency_range,
            frequency_decimation=frequency_decimation,
            frequency_bands=frequency_bands,
            frequency_reduction=frequency_reduction,
            output_names=output_names,
            signal_metadata=signal_metadata,
        )
    _warn_deprecated_measure_names(output_names)
    _warn_label_orientation_changed(method)
    _warn_legacy_settings("multitaper_connectivity", "transform_kwargs", kwargs)
    return result


# Every Multitaper setting, in its signature's order; the ones that
# multitaper_connectivity names (in its signature's order) are given by name,
# and the others go in transform_kwargs. Read when a call is checked, so they
# can follow the function they are derived from.
_MULTITAPER_SETTINGS = tuple(
    name for name in inspect.signature(Multitaper).parameters if name != "time_series"
)
_NAMED_TRANSFORM_SETTINGS = tuple(
    name
    for name in inspect.signature(multitaper_connectivity).parameters
    if name in _MULTITAPER_SETTINGS
)
_TRANSFORM_KWARGS_SETTINGS = tuple(
    name for name in _MULTITAPER_SETTINGS if name not in _NAMED_TRANSFORM_SETTINGS
)
# The Connectivity settings every wrapper takes by name.
_CONNECTIVITY_SETTINGS = ("dtype", "minimum_phase_tolerance", "minimum_phase_max_iterations")

# Spellings of the frequency units fourier_connectivity accepts without
# conversion, compared ignoring case and whitespace.
_SECOND_SPELLINGS = ("s", "sec", "second")
_HERTZ_SPELLINGS = frozenset(
    {"hz", "hertz", "cps"}
    | {
        f"{numerator}/{second}"
        for numerator in ("1", "cycle", "cycles")
        for second in _SECOND_SPELLINGS
    }
    | {f"{second}{power}" for second in _SECOND_SPELLINGS for power in ("-1", "^-1", "**-1")}
)
_CYCLES_PER_SAMPLE_SPELLINGS = frozenset({"cycle/sample", "cycles/sample"})


def _canonical_frequency_units(units: Any) -> str | None:
    """Canonical ``frequency_units`` for a frequency coordinate's ``units``.

    ``None`` for units that need converting first (e.g. ``"kHz"`` or
    ``"rad/s"``) and for a value that is not a string.
    """
    if not isinstance(units, str):
        return None
    spelling = "".join(units.split()).lower()
    if spelling in _HERTZ_SPELLINGS:
        return "Hz"
    if spelling in _CYCLES_PER_SAMPLE_SPELLINGS:
        return "cycles/sample"
    return None


def fourier_connectivity(
    fourier_coefficients: NDArray[np.complexfloating] | xr.DataArray,
    frequencies: NDArray[np.floating] | None = None,
    time: NDArray[np.floating] | None = None,
    method: str | list[str] | None = None,
    signal_names: Sequence[_SignalLabel] | None = None,
    *,
    squeeze: bool = False,
    connectivity_kwargs: dict[str, Any] | None = None,
    is_one_sided: bool | None = None,
    group_labels: Sequence[Hashable] | NDArray[Any] | str | None = None,
    frequency_range: tuple[float, float] | None = None,
    frequency_decimation: int = 1,
    frequency_bands: Mapping[str, tuple[float, float]] | None = None,
    frequency_reduction: Literal["mean", "integral"] = "mean",
    time_dim: Hashable | None = None,
    trial_dim: Hashable | None = None,
    taper_dim: Hashable | None = None,
    frequency_dim: Hashable | None = None,
    signal_dim: Hashable | None = None,
    dtype: DTypeLike = np.complex128,
    minimum_phase_tolerance: float = 1e-8,
    minimum_phase_max_iterations: int = 500,
) -> xr.DataArray | xr.Dataset:
    """Compute labeled connectivity from externally estimated FFT coefficients.

    The labeled output has one time axis, so the expectation is always
    ``"trials_tapers"``; use :class:`Connectivity` directly for expectations
    that retain trial/taper axes or average over time.

    Parameters
    ----------
    fourier_coefficients : array or xarray.DataArray
        Complex coefficients in ``(n_observations, n_frequencies, n_signals)``,
        ``(n_trials, n_tapers, n_frequencies, n_signals)``, or the core's full
        ``(n_time, n_trials, n_tapers, n_frequencies, n_signals)`` layout. A
        DataArray is transposed by semantic dimension names (or the explicit
        ``*_dim`` arguments) and its frequency, time, and signal coordinates
        and attributes are preserved.
    frequencies : array, shape (n_frequencies,), optional
        Frequency of each bin in Hz. A two-sided coordinate must be in standard
        FFT order; a non-negative, strictly increasing coordinate is treated as
        one-sided when ``is_one_sided`` is omitted. Taken from the DataArray
        coordinate when not given; that coordinate may instead declare
        ``units="cycles/sample"``. Its ``units`` must be a spelling of Hz or
        cycles/sample (e.g. ``"Hertz"``, ``"1/s"``, ``"s-1"``, ``"cycles/s"``,
        ignoring case and whitespace) or empty; convert other units first.
        Without either, bins are labeled in normalized cycles/sample.
    time : array, shape (n_time,), optional
        Center time of each window in seconds; defaults to window indices.
        Taken from the DataArray time coordinate when not given, including a
        numeric scalar coordinate named ``time`` or ``times`` on coefficients
        without a time dimension (as left by ``isel(time=0)``), which is the
        time of the single window.
    method : str or list of str, optional
        Measure name(s) from :func:`list_measures`. A single name returns a
        DataArray; a list (or ``None`` for :data:`DEFAULT_METHODS`) returns a
        Dataset with one variable per measure.
    signal_names : sequence, optional
        Labels for the ``source``/``target`` coordinates; defaults to the
        DataArray signal coordinate or ``"0"``, ``"1"``, ....
    squeeze : bool, default=False
        Only honored when a single ``method`` (a string) is requested. If there
        are exactly 2 signals, reduce a pairwise measure to the single ordered
        pair (first source, last target), returning a ``(time, frequency)`` array
        that keeps the selected ``source`` and ``target`` as scalar coordinates.
        With more than 2 signals a warning is issued and the full matrix is
        returned; for ``power`` squeeze is a no-op. Length-one ``time`` is kept.
    connectivity_kwargs : dict, optional
        Settings for the *measure* (a ``Connectivity`` method), e.g. ``pairs``
        for ``subset_pairwise_spectral_granger_prediction`` or ``n_components``
        for ``canonical_coherency``; passed to every requested measure, so a
        key a requested measure does not take raises ``TypeError``. Transform
        settings do not go here: they belong to the transform that produced
        ``fourier_coefficients``.
    is_one_sided : bool, optional
        Declare whether the coefficients cover only non-negative frequencies.
        When no frequency coordinate is available the sidedness cannot be
        inferred: pass ``True`` for one-sided input (e.g. ``rfft`` output) or
        ``False`` for a full FFT-order spectrum. ``False`` is honored as a
        declaration, so measures that require a two-sided spectrum run on the
        coefficients as given; they warn if the coefficients are not
        conjugate-symmetric, as the FFT of real-valued signals is, because
        mislabeled one-sided input makes those measures wrong. Leaving it unset in that case assumes two-sided,
        warns, and refuses those measures because the assumption cannot be
        checked. With a frequency coordinate it is inferred from
        ``frequencies``.
    group_labels : sequence or str, optional
        One label per signal naming the group it belongs to; labels are scalars
        such as integers or area names (``"CA1"``), and missing values
        (``None``, NaN) are rejected. Required by
        the group measures (``canonical_coherence``, ``canonical_coherency``,
        ``maximized_imaginary_coherency``, ``multivariate_interaction_measure``,
        ``blockwise_spectral_granger_prediction`` and
        ``maximized_imaginary_coherency_components``); an error is raised if no
        requested measure takes it (labels are passed only to the measures that
        do).
        A single string, rather than a sequence, names a coordinate along the
        input DataArray's signal dimension whose values are the labels, e.g.
        ``group_labels="region"`` for a ``region`` coordinate on ``channel``.
    frequency_range : (float, float), optional
        ``(low, high)`` bounds, in the units of the frequency coordinate, to
        keep before any decimation or band reduction; both endpoints are
        included, to within one part in a million of the bin spacing plus
        rounding (see :func:`frequency_band_reduce`). ``delay``, ``group_delay``, and
        ``phase_slope_index`` consume only these bins; other measures are
        computed from every bin and cropped, as in
        :func:`multitaper_connectivity`.
    frequency_decimation : int, default=1
        Keep every ``frequency_decimation``-th frequency bin of the
        frequency-resolved measures.
    frequency_bands : mapping of str to (float, float), optional
        Named bands, both endpoints included, to reduce the frequency axis of
        the frequency-resolved measures into; see :func:`frequency_band_reduce`
        and :func:`multitaper_connectivity` for ``phase_slope_index`` and
        ``group_delay``.
    frequency_reduction : {"mean", "integral"}, default="mean"
        Within-band reduction used with ``frequency_bands``.
    time_dim, trial_dim, taper_dim, frequency_dim, signal_dim : hashable, optional
        DataArray dimension names for each axis role, when they cannot be
        inferred from common names.
    dtype : numpy.dtype, default=complex128
        Working precision for the connectivity computations.
    minimum_phase_tolerance : float, default=1e-8
        Relative convergence tolerance of the Wilson factorization used by the
        directed measures.
    minimum_phase_max_iterations : int, default=500
        Maximum Wilson iterations for the directed measures.

    Returns
    -------
    xarray.DataArray or xarray.Dataset
        Labeled result with ``time``, ``frequency`` (or ``band``),
        and measure-specific dimensions such as ``source``/``target``; a
        DataArray for a single ``method`` name, otherwise a Dataset. Directed
        measures are oriented so ``sel(source=a, target=b)`` is the influence
        from ``a`` to ``b``. One-sided coefficients support functional
        measures, but measures that need a full two-sided spectrum raise.
    """
    (
        coefficient_data,
        frequencies,
        time,
        signal_names,
        input_attrs,
        signal_metadata,
    ) = _unwrap_fourier_input(
        fourier_coefficients,
        frequencies=frequencies,
        time=time,
        signal_names=signal_names,
        time_dim=time_dim,
        trial_dim=trial_dim,
        taper_dim=taper_dim,
        frequency_dim=frequency_dim,
        signal_dim=signal_dim,
    )
    # Coordinates are validated on the host; on the GPU, Multitaper's
    # frequencies and time are device arrays that reject np.asarray.
    if frequencies is not None:
        frequencies = to_numpy(frequencies)
    if time is not None:
        time = to_numpy(time)
    if time is not None and not _is_real_numeric_dtype(np.asarray(time).dtype):
        msg = (
            "time must contain numeric elapsed seconds (window centers); "
            f"got dtype {np.asarray(time).dtype!r}. Convert a datetime axis to "
            "elapsed seconds, e.g. (t - t[0]) / np.timedelta64(1, 's')."
        )
        raise TypeError(msg)
    inferred_one_sided = False
    if is_one_sided is not None:
        is_one_sided = _validated_flag("is_one_sided", is_one_sided)
    if frequencies is not None:
        frequency_values = np.asarray(frequencies, dtype=float)
        if frequency_values.ndim != 1:
            msg = "frequencies must be a one-dimensional coordinate."
            raise ValueError(msg)
        inferred_one_sided = bool(
            frequency_values.size > 0 and not np.any(frequency_values < 0)
        )
        one_sided = inferred_one_sided if is_one_sided is None else bool(is_one_sided)
    else:
        if is_one_sided is None:
            warnings.warn(
                "fourier_connectivity received no frequency coordinate and no "
                "is_one_sided flag; assuming a two-sided spectrum in standard "
                "FFT order. For rfft or wavelet coefficients (non-negative "
                "frequencies only) pass is_one_sided=True, otherwise "
                "is_one_sided=False to silence this warning.",
                UserWarning,
                stacklevel=stacklevel_outside_package(),
            )
        one_sided = bool(is_one_sided) if is_one_sided is not None else False

    # No default measure needs a two-sided spectrum, so the defaults suit any input.
    # The request and its settings are checked before the coefficients are copied.
    methods, output_names, return_dataarray = _requested_methods(method, DEFAULT_METHODS)
    _check_measure_settings(
        methods,
        connectivity_kwargs,
        function="fourier_connectivity",
        transform_settings_hint="set it on the transform that produced fourier_coefficients",
        frequency_decimation=frequency_decimation,
        frequency_bands=frequency_bands,
    )
    group_labels, connectivity_kwargs, group_methods = _resolve_group_labels(
        methods, connectivity_kwargs, group_labels
    )
    group_labels = _group_labels_from_coordinate(
        group_labels,
        signal_metadata if isinstance(fourier_coefficients, xr.DataArray) else None,
        no_coordinates=(
            "fourier_coefficients is not an xarray.DataArray, so it has none: pass "
            "one label per signal, e.g. group_labels=['CA1', 'CA1', 'PFC', 'PFC']"
        ),
    )
    connectivity = Connectivity(
        coefficient_data,
        expectation_type="trials_tapers",
        frequencies=frequencies,
        time=time,
        dtype=dtype,
        minimum_phase_tolerance=minimum_phase_tolerance,
        minimum_phase_max_iterations=minimum_phase_max_iterations,
        is_one_sided=one_sided,
    )
    _require_group_labels(methods, group_labels, connectivity.n_signals)
    if frequencies is None:
        # Without a frequency coordinate two-sidedness cannot be verified, so an
        # *assumed* two-sided spectrum (is_one_sided=None) must not let one-sided
        # input (e.g. rfft/wavelet coefficients) reach Wilson factorization and
        # produce a silently wrong result. Reject methods that declare the
        # full-spectrum requirement unless the caller declared the spectrum
        # two-sided with is_one_sided=False; other directional measures such as
        # dPLI and PSI remain valid on one-sided coefficients.
        two_sided_methods = [name for name in methods if _requires_two_sided(name)]
        if two_sided_methods and one_sided:
            # The caller already declared one-sided input, so no frequency vector
            # would enable Wilson factorization -- give the accurate reason.
            msg = (
                f"Measures {sorted(set(two_sided_methods))} require a full "
                "two-sided spectrum in standard FFT order. One-sided transforms "
                "(is_one_sided=True) support functional connectivity measures but "
                "not Wilson-factorized measures. Request only one-sided-compatible "
                "measures, or supply full two-sided coefficients."
            )
            raise ValueError(msg)
        if two_sided_methods and is_one_sided is None:
            msg = (
                f"Measures {sorted(set(two_sided_methods))} require a full "
                "two-sided spectrum in standard FFT order, which cannot be verified "
                "without a frequency coordinate. Pass `frequencies` (the FFT "
                "frequency vector, including negative bins) so two-sidedness can be "
                "checked, pass is_one_sided=False to declare a full FFT-order "
                "spectrum, or request only one-sided-compatible measures."
            )
            raise ValueError(msg)
        if two_sided_methods and is_one_sided is not None:
            # The declaration is trusted here, so check what it implies for real
            # signals: bin k is the conjugate of bin -k (FFT order). Measured
            # relative residuals: <= 4e-16 for FFTs of real noise in complex128
            # and <= 2e-7 in complex64 (up to 65536 bins), versus 1.3-1.45 for
            # rfft output declared two-sided and for complex-valued signals. A
            # threshold of 1e-3 leaves over three orders of magnitude on each side.
            positive_bins = coefficient_data[..., 1:, :]
            mirrored_bins = coefficient_data[..., :0:-1, :].conj()
            asymmetry = float((abs(positive_bins - mirrored_bins) ** 2).sum()) ** 0.5
            scale = float((abs(positive_bins) ** 2).sum()) ** 0.5
            if asymmetry > 1e-3 * scale:
                warnings.warn(
                    "The Fourier coefficients declared two-sided (is_one_sided="
                    "False) are not conjugate-symmetric along the frequency axis "
                    f"(relative residual {asymmetry / scale:.2g}), as the FFT of "
                    "real-valued signals is. This happens when one-sided "
                    "coefficients (e.g. rfft or wavelet output) are declared "
                    f"two-sided, and then {sorted(set(two_sided_methods))}, which "
                    "require a two-sided spectrum, are wrong; or when the signals "
                    "are complex-valued. Pass is_one_sided=True for one-sided "
                    "coefficients and request only one-sided-compatible measures, "
                    "or pass the full two-sided FFT of real-valued signals.",
                    UserWarning,
                    stacklevel=stacklevel_outside_package(),
                )
    signal_labels = _validated_signal_labels(signal_names, connectivity.n_signals)
    frequency_units = "cycles/sample" if frequencies is None else "Hz"
    coordinate_units = (signal_metadata.frequency_coordinate_attrs or {}).get("units")
    if coordinate_units is not None:
        canonical_units = _canonical_frequency_units(coordinate_units)
        if canonical_units is None:
            msg = (
                "External Fourier frequencies must be in Hz or cycles/sample, got "
                f"units={coordinate_units!r}. Convert the coordinate before calling "
                "fourier_connectivity, or, if it is already in one of those units "
                "under another spelling, set its units to 'Hz' or 'cycles/sample'."
            )
            raise ValueError(msg)
        frequency_units = canonical_units
    metadata: dict[str, Any] = {
        "transform": "external_fourier",
        "coefficient_shape": tuple(coefficient_data.shape),
        "frequency_coordinate": "provided" if frequencies is not None else "normalized",
        "time_coordinate": "provided" if time is not None else "index",
        "frequency_units": frequency_units,
        "is_one_sided": one_sided,
        "one_sided_inferred": is_one_sided is None and inferred_one_sided,
    }
    all_frequencies = connectivity.all_frequencies
    if signal_metadata.n_trials is not None:
        metadata["n_trials"] = signal_metadata.n_trials
    if frequency_units == "cycles/sample":
        # The normalized grid establishes a Nyquist coordinate, not a physical
        # sampling rate in Hz. Retain it for integration after cropping.
        metadata["nyquist_frequency"] = 0.5
    elif not one_sided and all_frequencies.size > 1:
        # Bin n_fft // 2 of an FFT-order grid is the farthest from 0 Hz, so its
        # rounding moves the rate least. For an even n_fft it is the Nyquist
        # bin itself: the recorded Nyquist frequency is that bin's stored value
        # whatever the grid's rounding. An odd n_fft has no Nyquist bin: its top
        # bin is n_fft // 2 bins of sampling_frequency / n_fft above zero. The
        # arithmetic stays in the grid's precision (float32 at least), which
        # rounds the rate back to within a unit in its last place: a 3-bin
        # float32 grid of a 100 Hz rate records 100.0.
        n_fft = all_frequencies.size
        precision = np.result_type(all_frequencies.dtype, np.float32).type
        # Converted before abs: an integer grid's Nyquist bin can be its
        # dtype's minimum, which has no absolute value in that dtype.
        top = abs(precision(all_frequencies[n_fft // 2]))
        sampling_frequency = (
            precision(2) * top
            if n_fft % 2 == 0
            else top * precision(n_fft) / precision(n_fft // 2)
        )
        metadata["sampling_frequency"] = float(sampling_frequency)
        metadata["nyquist_frequency"] = float(sampling_frequency) / 2
    shared_attrs = _shared_provenance_attrs(
        connectivity,
        metadata,
        input_attrs=input_attrs,
    )
    result = _format_and_reduce_measures(
        connectivity,
        methods,
        return_dataarray=return_dataarray,
        signal_labels=signal_labels,
        squeeze=squeeze,
        shared_attrs=shared_attrs,
        connectivity_kwargs=connectivity_kwargs,
        group_labels=group_labels,
        group_methods=group_methods,
        frequency_range=frequency_range,
        frequency_decimation=frequency_decimation,
        frequency_bands=frequency_bands,
        frequency_reduction=frequency_reduction,
        output_names=output_names,
        signal_metadata=signal_metadata,
    )
    _warn_deprecated_measure_names(output_names)
    return result
