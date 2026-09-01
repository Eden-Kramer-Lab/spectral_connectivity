"""Functions for getting connectivity measures in a labeled array format."""

import difflib
import json
import warnings
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from logging import getLogger
from typing import Any, Literal, NamedTuple, TypeAlias

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from spectral_connectivity.connectivity import (
    Connectivity,
    MultivariateConnectivityResult,
)
from spectral_connectivity.transforms import Multitaper
from spectral_connectivity.utils import BackendArray, get_compute_backend, to_numpy

logger = getLogger(__name__)

_UNSET = object()

# Per-element type of a signal label. The array-level invariants -- homogeneous
# dtype, uniqueness, no missing values, int32 range -- cannot be expressed in an
# element union and are enforced in ``_validated_signal_labels``.
_SignalLabel: TypeAlias = (
    str
    | bytes
    | bool
    | int
    | float
    | np.integer
    | np.floating
    | np.bool_
    | np.str_
    | np.bytes_
    | np.datetime64
    | np.timedelta64
)


class _TimeAxis(NamedTuple):
    """Resolved time axis: an inferred rate (None if given) and start time."""

    inferred_sampling_frequency: float | None
    start_time: float | None


class _UnwrappedInput(NamedTuple):
    """Array data plus what a DataArray contributed, named to avoid swaps."""

    data: BackendArray
    signal_names: Sequence[_SignalLabel] | None
    inferred_sampling_frequency: float | None
    inferred_start_time: float | None
    input_attrs: Mapping[Any, Any] | None


def _json_compatible(value: Any) -> Any:
    """Convert provenance values to deterministic, JSON-compatible objects.

    JSON-native values convert deterministically. NumPy arrays/scalars and
    Python sequences are normalized to their JSON value representation, so
    container and dtype distinctions are not retained. Mappings with non-string
    keys use a tagged item-list representation so unlike keys such as ``1`` and
    ``"1"`` cannot collide. Any other object is recorded on a best-effort basis
    as ``{"python_type", "repr"}``; that ``repr`` is not guaranteed stable
    across runs because it may embed a memory address.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if np.isfinite(value):
            return value
        return {"nonfinite_float": repr(value)}
    if isinstance(value, np.generic):
        return _json_compatible(value.item())
    if isinstance(value, np.ndarray):
        # Host-convert via the util so a device-backed array is never moved by
        # an implicit ``np.asarray``/``tolist`` transfer.
        return _json_compatible(to_numpy(value).tolist())
    if isinstance(value, Mapping):
        if all(isinstance(key, str) for key in value):
            return {key: _json_compatible(item) for key, item in sorted(value.items())}
        converted_items = [
            [_json_compatible(key), _json_compatible(item)]
            for key, item in value.items()
        ]
        # Keys are already JSON-compatible; sort by their canonical form so the
        # serialization contract lives in one place (``_canonical_json``).
        converted_items.sort(key=lambda pair: _canonical_json(pair[0]))
        return {"python_type": "mapping", "items": converted_items}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_json_compatible(item) for item in value]
    return {
        "python_type": f"{type(value).__module__}.{type(value).__qualname__}",
        "repr": repr(value),
    }


def _canonical_json(value: Any) -> str:
    """Serialize provenance as stable JSON suitable for a NetCDF attribute."""
    return json.dumps(
        _json_compatible(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _netcdf_provenance_value(value: Any) -> Any:
    """Return a NetCDF-safe scalar, using JSON for structured values.

    Non-finite floats are encoded as JSON too, so the ``arg_<key>`` view matches
    the ``measure_kwargs_json`` record rather than storing a bare ``NaN``/``inf``
    that not every NetCDF engine round-trips cleanly.
    """
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return _canonical_json(value)
    if isinstance(value, (str, int, float, np.integer, np.floating, np.bool_)):
        return value
    return _canonical_json(value)


def _store_provenance_item(
    attrs: dict[str, Any], prefix: str, key: Any, value: Any
) -> None:
    """Record ``value`` under ``<prefix><key>`` as a NetCDF-safe attribute.

    A scalar is stored as-is; a structured or non-finite value is stored as a
    canonical JSON string under ``<prefix><key>_json`` so a consumer knows to
    ``json.loads`` it (mirrors the ``arg_<key>`` / ``arg_<key>_json`` split).
    Raises ``ValueError`` on a name collision (e.g. a structured ``x`` and a
    scalar ``x_json`` both mapping to ``<prefix>x_json``) rather than silently
    overwriting the earlier value; ``measure_kwargs_json`` remains the canonical
    full record.
    """
    netcdf_value = _netcdf_provenance_value(value)
    if isinstance(netcdf_value, str) and not isinstance(value, str):
        attr_name = f"{prefix}{key}_json"
    else:
        attr_name = f"{prefix}{key}"
    if attr_name in attrs:
        raise ValueError(
            f"Provenance attribute {attr_name!r} is assigned twice; two keys "
            f"(one of them {key!r}) collide under the {prefix!r} namespace. "
            "Rename the offending argument."
        )
    attrs[attr_name] = netcdf_value


class UnsupportedMeasureError(ValueError):
    """A method has no registered semantic xarray output contract.

    Built-in nonstandard results (components, groups, delays, and multi-variable
    outputs) have explicit schemas. This exception remains for unregistered
    extensions whose returned shape cannot be inferred safely. It subclasses
    ``ValueError`` for backward compatibility and lets multi-measure wrappers
    distinguish structural incompatibility from genuine numerical errors.
    """


def _package_version() -> str:
    """Return the installed spectral_connectivity version (or ``"unknown"``)."""
    try:
        from importlib.metadata import PackageNotFoundError, version

        return version("spectral_connectivity")
    except (PackageNotFoundError, ImportError):
        # Package not installed / metadata unavailable; a genuinely unexpected
        # error is left to surface rather than being masked as "unknown".
        return "unknown"


@dataclass(frozen=True)
class _MeasureSpec:
    """Shape/capability metadata used by the xarray wrapper."""

    output_kind: Literal[
        "pairwise",
        "power",
        "group_pairwise",
        "delay",
        "global",
        "group_delay",
        "phase_slope",
        "multivariate_components",
        "unsupported",
    ]
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
            raise ValueError(
                "transpose_output requires pairwise or group_pairwise output."
            )
        if self.transpose_output and not self.is_directed:
            raise ValueError("transpose_output requires a directional measure.")
        if self.is_default and self.output_kind == "unsupported":
            raise ValueError("an unsupported measure cannot be a default.")


_PAIRWISE_SPEC = _MeasureSpec("pairwise")
_DIRECTED_PAIRWISE_SPEC = _MeasureSpec(
    "pairwise",
    is_directed=True,
    transpose_output=True,
    requires_two_sided=True,
)
_UNSUPPORTED_SPEC = _MeasureSpec("unsupported")

# This is the single source of truth for wrapper capabilities and defaults.
# Insertion order preserves the historical Dataset variable order.
_MEASURE_SPECS: dict[str, _MeasureSpec] = {
    "coherence_magnitude": _MeasureSpec("pairwise", is_default=True),
    "coherence_phase": _MeasureSpec("pairwise", is_default=True),
    "debiased_squared_phase_lag_index": _MeasureSpec("pairwise", is_default=True),
    "debiased_squared_weighted_phase_lag_index": _MeasureSpec(
        "pairwise", is_default=True
    ),
    "imaginary_coherence": _MeasureSpec("pairwise", is_default=True),
    "pairwise_phase_consistency": _MeasureSpec("pairwise", is_default=True),
    "pairwise_spectral_granger_prediction": _MeasureSpec(
        "pairwise",
        is_default=True,
        is_directed=True,
        transpose_output=True,
        requires_two_sided=True,
    ),
    "phase_lag_index": _MeasureSpec("pairwise", is_default=True),
    "phase_locking_value": _MeasureSpec("pairwise", is_default=True),
    "power": _MeasureSpec("power", is_default=True),
    "weighted_phase_lag_index": _MeasureSpec("pairwise", is_default=True),
    "coherency": _PAIRWISE_SPEC,
    "cross_spectral_density": _PAIRWISE_SPEC,
    "imaginary_coherency": _PAIRWISE_SPEC,
    "partial_coherence": _PAIRWISE_SPEC,
    "corrected_imaginary_phase_locking_value": _PAIRWISE_SPEC,
    # dPLI's native row/column layout is already phase-leader -> phase-lagger,
    # so it must not receive the transpose used by Granger/DTF-family outputs.
    "directed_phase_lag_index": _MeasureSpec("pairwise", is_directed=True),
    "subset_pairwise_spectral_granger_prediction": _DIRECTED_PAIRWISE_SPEC,
    "conditional_spectral_granger_prediction": _DIRECTED_PAIRWISE_SPEC,
    "time_reversed_spectral_granger_prediction": _DIRECTED_PAIRWISE_SPEC,
    # Directed-transfer-function family: opt-in (not in the default set),
    # directed (output[i, j] = influence j -> i, transposed to source -> target),
    # and returning the full (time, frequency, source, target) layout.
    "directed_transfer_function": _DIRECTED_PAIRWISE_SPEC,
    "directed_coherence": _DIRECTED_PAIRWISE_SPEC,
    "partial_directed_coherence": _DIRECTED_PAIRWISE_SPEC,
    "generalized_partial_directed_coherence": _DIRECTED_PAIRWISE_SPEC,
    "direct_directed_transfer_function": _DIRECTED_PAIRWISE_SPEC,
    "blockwise_spectral_granger_prediction": _MeasureSpec(
        "group_pairwise",
        is_directed=True,
        transpose_output=True,
        requires_two_sided=True,
    ),
    "canonical_coherence": _MeasureSpec("group_pairwise"),
    "maximized_imaginary_coherency": _MeasureSpec("group_pairwise"),
    "multivariate_interaction_measure": _MeasureSpec("group_pairwise"),
    "canonical_coherency": _MeasureSpec("multivariate_components"),
    "maximized_imaginary_coherency_components": _MeasureSpec("multivariate_components"),
    "delay": _MeasureSpec("delay", is_directed=True),
    "global_coherence": _MeasureSpec("global"),
    "group_delay": _MeasureSpec("group_delay", is_directed=True),
    "phase_slope_index": _MeasureSpec("phase_slope", is_directed=True),
}

DEFAULT_METHODS: tuple[str, ...] = tuple(
    name for name, spec in _MEASURE_SPECS.items() if spec.is_default
)


@dataclass(frozen=True)
class MeasureInfo:
    """A single connectivity measure the high-level wrapper can compute.

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
    """

    name: str
    category: str
    description: str
    is_default: bool
    is_directed: bool
    requires_two_sided: bool


def _measure_description(name: str) -> str:
    """Return the one-line summary from a ``Connectivity`` method docstring."""
    docstring = getattr(Connectivity, name).__doc__ or ""
    for line in docstring.strip().splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


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
    whether it is in the default set and/or directional.

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
        One record per measure, in the wrapper's canonical order.

    Examples
    --------
    >>> from spectral_connectivity import list_measures
    >>> [m.name for m in list_measures(default_only=True)][:3]
    ['coherence_magnitude', 'coherence_phase', 'debiased_squared_phase_lag_index']
    >>> next(m for m in list_measures() if m.name == "power").description
    'Return the one-sided power spectral density of the signal.'
    """
    valid_categories = {spec.output_kind for spec in _MEASURE_SPECS.values()}
    if category is not None and category not in valid_categories:
        raise ValueError(
            f"Unknown category {category!r}. Valid categories are: "
            f"{', '.join(sorted(valid_categories))}."
        )

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
            )
        )
    return measures


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


def _validate_method_names(methods: Sequence[str]) -> None:
    """Reject unknown measure names with a helpful, actionable message.

    A name is accepted if it is either a registered measure or any callable
    ``Connectivity`` attribute, so subclass/monkeypatched extension measures
    (which the wrapper supports) still pass. Properties and other non-callable
    attributes are rejected before they can produce an obscure ``TypeError``.
    """
    unknown = [
        method
        for method in methods
        if method not in _MEASURE_SPECS
        and not callable(getattr(Connectivity, method, None))
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
    """Return wrapper metadata, rejecting known incompatible measures."""
    measure_spec = _MEASURE_SPECS.get(method)
    if measure_spec is None or measure_spec.output_kind != "unsupported":
        return measure_spec
    raise UnsupportedMeasureError(
        f"The method '{method}' is not supported by the xarray interface "
        f"(it does not return a plain (time, frequency, source, target) "
        f"array). Please use the Connectivity class directly instead:\n\n"
        f"from spectral_connectivity import Connectivity\n"
        f"conn = Connectivity.from_multitaper(m)\n"
        f"result = conn.{method}()\n"
    )


def _validated_signal_labels(
    signal_names: Sequence[_SignalLabel] | None,
    n_signals: int,
) -> BackendArray:
    """Return a unique, portable, one-dimensional xarray signal coordinate."""
    if signal_names is None:
        names: list[_SignalLabel] = [str(index) for index in range(n_signals)]
    else:
        names = list(signal_names)
    if len(names) != n_signals:
        raise ValueError(
            f"signal_names must contain {n_signals} names, got {len(names)}."
        )
    try:
        signal_coordinate = xr.IndexVariable("signal", names)
        signal_index = signal_coordinate.to_index()
    except (TypeError, ValueError) as error:
        raise ValueError(
            "signal_names must form a one-dimensional xarray coordinate of "
            "scalar labels; nested or structured labels are not supported."
        ) from error
    if signal_coordinate.dtype.kind not in "biufSUMm":
        raise ValueError(
            "signal_names must contain NetCDF-compatible string, real numeric, "
            "datetime, or timedelta scalar labels; object and complex labels "
            "are not supported."
        )
    if signal_coordinate.dtype.kind in "iu" and signal_coordinate.size:
        # SciPy is a required dependency and therefore xarray's only guaranteed
        # NetCDF writer in a minimum installation. Its NetCDF3 backend cannot
        # represent integer coordinate values outside the signed 32-bit range.
        integer_values = np.asarray(signal_coordinate.data)
        minimum = int(integer_values.min())
        maximum = int(integer_values.max())
        int32 = np.iinfo(np.int32)
        if minimum < int32.min or maximum > int32.max:
            raise ValueError(
                "Integer signal_names must fit the signed 32-bit range for "
                "portable NetCDF3 serialization; got range "
                f"[{minimum}, {maximum}]. Use string labels for larger identifiers."
            )
    if bool(getattr(signal_index, "hasnans", False)):
        raise ValueError(
            "signal_names must not contain missing labels (NaN, NaT, or None)."
        )
    if not signal_index.is_unique:
        duplicates = sorted(
            signal_index[signal_index.duplicated(keep=False)].unique().tolist(),
            key=repr,
        )
        raise ValueError(
            "signal_names must be unique to label the source/target axes; "
            f"duplicates: {duplicates}."
        )
    return signal_coordinate.data


def _connectivity_result_to_xarray(
    connectivity: Connectivity,
    method: str,
    signal_labels: BackendArray,
    squeeze: bool,
    shared_attrs: Mapping[str, Any],
    **kwargs: Any,
) -> xr.DataArray | xr.Dataset:
    """Format one result from an already-built ``Connectivity`` instance.

    ``signal_labels`` and ``shared_attrs`` are invariant across the measures of
    one transform, so the caller validates/builds them once and passes them in.
    """
    measure_spec = _get_measure_spec(method)
    numerical_result = getattr(connectivity, method)(**kwargs)

    pairwise_shape = (
        len(connectivity.time),
        len(connectivity.frequencies),
        connectivity.n_signals,
        connectivity.n_signals,
    )
    power_shape = pairwise_shape[:-1]
    if measure_spec is None:
        actual_shape = tuple(numerical_result.shape)
        if actual_shape != pairwise_shape:
            raise UnsupportedMeasureError(
                f"The method '{method}' returned shape {actual_shape}, but an "
                f"unregistered wrapper extension must return {pairwise_shape}. "
                "Register its output contract or use Connectivity directly."
            )
        measure_spec = _PAIRWISE_SPEC

    # Copy the shared provenance so per-measure keys never leak across measures.
    attrs = dict(shared_attrs)
    attrs["measure"] = method
    attrs["measure_kwargs_json"] = _canonical_json(kwargs)
    for key, value in kwargs.items():
        _store_provenance_item(attrs, "arg_", key, value)

    base_coordinates: dict[str, Any] = {
        "time": (
            "time",
            connectivity.time,
            {"long_name": "Window center time", "units": "s"},
        ),
        "frequency": (
            "frequency",
            connectivity.frequencies,
            {"long_name": "Frequency", "units": "Hz"},
        ),
    }
    signal_coordinates = {
        "source": ("source", signal_labels, {"long_name": "Source signal"}),
        "target": ("target", signal_labels, {"long_name": "Target signal"}),
    }

    if measure_spec.output_kind in {"pairwise", "power"}:
        connectivity_mat = np.asarray(numerical_result)
        expected_shape = (
            power_shape if measure_spec.output_kind == "power" else pairwise_shape
        )
        if tuple(connectivity_mat.shape) != expected_shape:
            raise ValueError(
                f"The method '{method}' returned shape {connectivity_mat.shape}; "
                f"its wrapper contract requires {expected_shape}."
            )
        if measure_spec.transpose_output:
            connectivity_mat = np.swapaxes(connectivity_mat, -1, -2)
        coordinates = {**base_coordinates, "source": signal_coordinates["source"]}
    else:
        coordinates = dict(base_coordinates)

    if measure_spec.output_kind == "power":
        # squeeze has no meaning for power (no target axis); it is a no-op here.
        xar = xr.DataArray(
            connectivity_mat,
            coords=coordinates,
            dims=("time", "frequency", "source"),
            name=method,
            attrs=attrs,
        )
        return xar

    if measure_spec.output_kind == "pairwise":
        coordinates["target"] = signal_coordinates["target"]
        xar = xr.DataArray(
            connectivity_mat,
            coords=coordinates,
            dims=("time", "frequency", "source", "target"),
            name=method,
            attrs=attrs,
        )
        if squeeze and connectivity.n_signals == 2:
            # Reduce to the single ordered pair (first source, last target).
            # drop=False keeps ``source`` and ``target`` as scalar coordinates so
            # the returned (time, frequency) array still records which pair -- and
            # for directed measures, which direction -- it represents. The caller
            # only passes squeeze=True when returning a standalone DataArray;
            # these scalar coordinates would otherwise collide, Dataset-wide, with
            # a sibling ``power`` variable's ``source`` dimension.
            xar = xar.isel(source=0, target=-1, drop=False)
        elif squeeze and connectivity.n_signals > 2:
            warnings.warn(
                f"squeeze=True but there are {connectivity.n_signals} signals; "
                "returning the full (source, target) matrix.",
                UserWarning,
                stacklevel=2,
            )
        return xar

    if measure_spec.output_kind == "group_pairwise":
        connectivity_mat, group_labels = numerical_result
        connectivity_mat = np.asarray(connectivity_mat)
        group_labels = np.asarray(group_labels)
        expected_shape = (
            len(connectivity.time),
            len(connectivity.frequencies),
            len(group_labels),
            len(group_labels),
        )
        if connectivity_mat.shape != expected_shape:
            raise ValueError(
                f"The method '{method}' returned shape {connectivity_mat.shape}; "
                f"its group-pairwise contract requires {expected_shape}."
            )
        if measure_spec.transpose_output:
            connectivity_mat = np.swapaxes(connectivity_mat, -1, -2)
        coordinates.update(
            {
                "source_group": ("source_group", group_labels),
                "target_group": ("target_group", group_labels),
            }
        )
        return xr.DataArray(
            connectivity_mat,
            coords=coordinates,
            dims=("time", "frequency", "source_group", "target_group"),
            name=method,
            attrs=attrs,
        )

    if measure_spec.output_kind == "delay":
        connectivity_mat = np.asarray(numerical_result)
        frequencies = np.asarray(connectivity.frequencies)
        frequency_band = kwargs.get("frequencies_of_interest")
        if frequency_band is not None:
            frequencies = frequencies[
                (frequency_band[0] < frequencies) & (frequencies < frequency_band[1])
            ]
        delay_expected_shape = (
            len(connectivity.time),
            len(frequencies),
            connectivity_mat.shape[-3],
            connectivity.n_signals,
            connectivity.n_signals,
        )
        if connectivity_mat.shape != delay_expected_shape:
            raise ValueError(
                f"The method '{method}' returned shape {connectivity_mat.shape}; "
                f"its delay contract requires {delay_expected_shape}."
            )
        coordinates = {
            "time": base_coordinates["time"],
            "frequency": ("frequency", frequencies, {"units": "Hz"}),
            "candidate": np.arange(
                -int(kwargs.get("n_range", 3)), int(kwargs.get("n_range", 3)) + 1
            ),
            **signal_coordinates,
        }
        return xr.DataArray(
            connectivity_mat,
            coords=coordinates,
            dims=("time", "frequency", "candidate", "source", "target"),
            name=method,
            attrs=attrs,
        )

    if measure_spec.output_kind == "phase_slope":
        connectivity_mat = np.asarray(numerical_result)
        expected_shape = (
            len(connectivity.time),
            connectivity.n_signals,
            connectivity.n_signals,
        )
        if connectivity_mat.shape != expected_shape:
            raise ValueError(
                f"The method '{method}' returned shape {connectivity_mat.shape}; "
                f"its phase-slope contract requires {expected_shape}."
            )
        band = kwargs.get("frequencies_of_interest")
        if band is None:
            band = (connectivity.frequencies[0], connectivity.frequencies[-1])
        coordinates = {
            "time": base_coordinates["time"],
            **signal_coordinates,
            "frequency_band_lower": float(band[0]),
            "frequency_band_upper": float(band[1]),
        }
        return xr.DataArray(
            connectivity_mat,
            coords=coordinates,
            dims=("time", "source", "target"),
            name=method,
            attrs=attrs,
        )

    if measure_spec.output_kind == "group_delay":
        delay, slope, r_value = numerical_result
        dataset_coordinates = {
            "time": base_coordinates["time"],
            **signal_coordinates,
        }
        variables = {
            "group_delay": ("Group delay", np.asarray(delay), "s"),
            "group_delay_slope": ("phase slope", np.asarray(slope), "rad/Hz"),
            "group_delay_r_value": (
                "phase-frequency correlation",
                np.asarray(r_value),
                "1",
            ),
        }
        data_vars: dict[str, xr.DataArray] = {}
        for name, (long_name, values, units) in variables.items():
            if values.shape != (
                len(connectivity.time),
                connectivity.n_signals,
                connectivity.n_signals,
            ):
                raise ValueError(f"The method '{method}' returned an invalid shape.")
            variable_attrs = {**attrs, "long_name": long_name, "units": units}
            data_vars[name] = xr.DataArray(
                values,
                coords=dataset_coordinates,
                dims=("time", "source", "target"),
                attrs=variable_attrs,
            )
        return xr.Dataset(data_vars, attrs=attrs)

    if measure_spec.output_kind == "global":
        scores, vectors = numerical_result
        scores = np.asarray(scores)[..., : len(connectivity.frequencies), :]
        vectors = np.asarray(vectors)[..., : len(connectivity.frequencies), :, :]
        n_components = scores.shape[-1]
        dataset_coordinates = {
            **base_coordinates,
            "component": np.arange(n_components),
            "source": signal_coordinates["source"],
        }
        return xr.Dataset(
            {
                "global_coherence": xr.DataArray(
                    scores,
                    coords={
                        key: dataset_coordinates[key]
                        for key in ("time", "frequency", "component")
                    },
                    dims=("time", "frequency", "component"),
                    attrs=attrs,
                ),
                "global_coherence_vectors": xr.DataArray(
                    vectors,
                    coords=dataset_coordinates,
                    dims=("time", "frequency", "source", "component"),
                    attrs={**attrs, "long_name": "Global coherence spatial vectors"},
                ),
            },
            attrs=attrs,
        )

    if measure_spec.output_kind == "multivariate_components":
        if not isinstance(numerical_result, MultivariateConnectivityResult):
            raise TypeError(
                f"The method '{method}' did not return MultivariateConnectivityResult."
            )
        n_connections = numerical_result.scores.shape[-2]
        n_components = numerical_result.scores.shape[-1]
        expected_scores = (
            len(connectivity.time),
            len(connectivity.frequencies),
            n_connections,
            n_components,
        )
        if numerical_result.scores.shape != expected_scores:
            raise ValueError(
                f"The method '{method}' returned score shape "
                f"{numerical_result.scores.shape}; expected {expected_scores}."
            )
        component_coordinates = {
            **base_coordinates,
            "connection": np.arange(n_connections),
            "component": np.arange(n_components),
            # Per-connection group labels on the ``connection`` dimension. Named
            # distinctly from the ``source_group``/``target_group`` *dimension*
            # coordinates used by group-pairwise results so the two contracts
            # never alias (and are silently overwritten) when merged in one
            # Dataset.
            "connection_seed_group": (
                "connection",
                numerical_result.connections[:, 0],
            ),
            "connection_target_group": (
                "connection",
                numerical_result.connections[:, 1],
            ),
            "side": ("side", ["seed", "target"]),
            "signal": ("signal", signal_labels),
            "group": ("group", numerical_result.group_labels),
        }
        data_vars = {
            method: xr.DataArray(
                numerical_result.scores,
                coords={
                    key: component_coordinates[key]
                    for key in (
                        "time",
                        "frequency",
                        "connection",
                        "component",
                        "connection_seed_group",
                        "connection_target_group",
                    )
                },
                dims=("time", "frequency", "connection", "component"),
                attrs=attrs,
            ),
            "group_membership": xr.DataArray(
                numerical_result.group_membership,
                coords={
                    "group": component_coordinates["group"],
                    "signal": component_coordinates["signal"],
                },
                dims=("group", "signal"),
            ),
        }
        projection_dims = (
            "time",
            "frequency",
            "connection",
            "component",
            "side",
            "signal",
        )
        projection_coordinates = {
            key: component_coordinates[key]
            for key in (
                "time",
                "frequency",
                "connection",
                "component",
                "connection_seed_group",
                "connection_target_group",
                "side",
                "signal",
            )
        }
        if numerical_result.filters is not None:
            data_vars[f"{method}_filters"] = xr.DataArray(
                numerical_result.filters,
                coords=projection_coordinates,
                dims=projection_dims,
                attrs={**attrs, "long_name": "Spatial filters"},
            )
        if numerical_result.patterns is not None:
            data_vars[f"{method}_patterns"] = xr.DataArray(
                numerical_result.patterns,
                coords=projection_coordinates,
                dims=projection_dims,
                attrs={**attrs, "long_name": "Spatial patterns"},
            )
        return xr.Dataset(data_vars, attrs=attrs)

    raise UnsupportedMeasureError(
        f"No xarray formatter is registered for method {method!r}."
    )


def _shared_provenance_attrs(
    connectivity: Connectivity,
    transform_metadata: Mapping[str, Any],
    input_attrs: Mapping[Any, Any] | None = None,
    *,
    transform_prefix: str = "mt_",
) -> dict[str, Any]:
    """Provenance shared by every measure computed from one transform.

    Covers the package/version, the imported backend, the expectation type, and
    the multitaper parameters (``mt_*``) -- everything that does not depend on
    the specific measure. The per-measure attributes (``measure``,
    ``measure_kwargs_json``, and the convenient ``arg_*`` views) are added by
    the caller. Attributes on an input ``xarray.DataArray`` are carried through
    as one canonical JSON record so arbitrary keys cannot collide or produce
    invalid NetCDF attribute names.
    """
    # Namespace transform settings so they cannot collide with measure-level or
    # package-level provenance attributes.
    attrs: dict[str, Any] = {
        transform_prefix + attr: value for attr, value in transform_metadata.items()
    }
    attrs["package"] = "spectral_connectivity"
    attrs["package_version"] = _package_version()
    # get_compute_backend() reports the backend actually imported (numpy vs
    # cupy), not the current env var; is_gpu_enabled() would mislabel a result if
    # SPECTRAL_CONNECTIVITY_ENABLE_GPU changed after import.
    attrs["backend"] = get_compute_backend()["backend"].upper()
    attrs["expectation_type"] = connectivity.expectation_type
    # A single fixed key is both collision-proof and a valid NetCDF attribute
    # name. Flattening arbitrary user keys would make unlike keys such as 1 and
    # "1" collide, make a structured ``x`` collide with a literal ``x_json``,
    # and let characters such as "/" create an invalid NetCDF attribute name.
    if input_attrs:
        attrs["input_attrs_json"] = _canonical_json(input_attrs)
    return attrs


def frequency_band_reduce(
    result: xr.DataArray | xr.Dataset,
    bands: Mapping[str, tuple[float, float]],
    *,
    reduction: Literal["mean", "integral"] = "mean",
) -> xr.DataArray | xr.Dataset:
    """Reduce a frequency-resolved result into labeled frequency bands.

    ``reduction="mean"`` averages the already-computed connectivity score over
    the bins in each inclusive band. Phase is treated specially: a
    ``coherence_phase`` result uses a circular mean, while complex-valued
    measures use their ordinary complex (vector) mean. ``reduction="integral"``
    computes a trapezoidal integral and is intentionally restricted to spectral
    densities (``power`` and ``cross_spectral_density``), where it represents
    band power/covariance rather than a frequency-averaged score.

    Parameters
    ----------
    result : xarray.DataArray or xarray.Dataset
        Result with a one-dimensional ``frequency`` coordinate.
    bands : mapping of str to (float, float)
        Inclusive lower and upper frequency bounds in the coordinate's units.
    reduction : {"mean", "integral"}, default="mean"
        Scientifically defined reduction to apply within each band.
    """
    if "frequency" not in result.dims:
        raise ValueError("result must have a 'frequency' dimension.")
    if reduction not in {"mean", "integral"}:
        raise ValueError("reduction must be either 'mean' or 'integral'.")
    if not isinstance(bands, Mapping) or len(bands) == 0:
        raise ValueError("bands must be a non-empty mapping of names to bounds.")

    frequencies = np.asarray(result.coords["frequency"].values)
    if frequencies.ndim != 1 or frequencies.size == 0:
        raise ValueError("frequency must be a non-empty one-dimensional coordinate.")
    if not np.all(np.isfinite(frequencies)):
        raise ValueError("frequency must contain only finite values.")
    if frequencies.size > 1 and not np.all(np.diff(frequencies) > 0):
        raise ValueError("frequency must be strictly increasing for band reduction.")

    band_names = list(bands)
    if len(set(band_names)) != len(band_names) or not all(
        isinstance(name, str) and name for name in band_names
    ):
        raise ValueError("band names must be unique, non-empty strings.")

    band_masks: list[NDArray[np.bool_]] = []
    for name, bounds in bands.items():
        try:
            lower, upper = bounds
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Band {name!r} must contain exactly two bounds (low, high)."
            ) from error
        lower = float(lower)
        upper = float(upper)
        if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
            raise ValueError(
                f"Band {name!r} must have finite bounds with low <= high; "
                f"got ({lower!r}, {upper!r})."
            )
        mask = (frequencies >= lower) & (frequencies <= upper)
        if not np.any(mask):
            raise ValueError(
                f"Band {name!r} ({lower:g}, {upper:g}) contains no frequency bins."
            )
        band_masks.append(mask)

    def _reduce_dataarray(data: xr.DataArray) -> xr.DataArray:
        measure = str(data.attrs.get("measure", data.name or ""))
        if reduction == "integral" and measure not in {
            "power",
            "cross_spectral_density",
        }:
            raise ValueError(
                "reduction='integral' is defined only for power and "
                "cross_spectral_density; use reduction='mean' for "
                f"{measure or 'this result'!r}."
            )

        reduced_bands: list[xr.DataArray] = []
        for mask in band_masks:
            selected = data.isel(frequency=np.flatnonzero(mask))
            if reduction == "integral":
                reduced = selected.integrate("frequency")
            elif measure == "coherence_phase":
                # Circular mean prevents phases near -pi and +pi from
                # spuriously cancelling toward zero.
                phase_vectors = xr.apply_ufunc(np.exp, 1j * selected)
                reduced = xr.apply_ufunc(
                    np.angle,
                    phase_vectors.mean("frequency", keep_attrs=True),
                    keep_attrs=True,
                )
            else:
                reduced = selected.mean("frequency", keep_attrs=True)
            reduced_bands.append(reduced)

        band_coordinate = xr.IndexVariable("band", band_names)
        reduced = xr.concat(reduced_bands, dim=band_coordinate)
        desired_dims = tuple(
            "band" if dimension == "frequency" else dimension for dimension in data.dims
        )
        reduced = reduced.transpose(*desired_dims)
        reduced.attrs = dict(data.attrs)
        reduced.attrs["frequency_bands_json"] = _canonical_json(bands)
        reduced.attrs["frequency_reduction"] = reduction
        return reduced

    if isinstance(result, xr.DataArray):
        return _reduce_dataarray(result)

    data_vars = {
        name: _reduce_dataarray(data) if "frequency" in data.dims else data
        for name, data in result.data_vars.items()
    }
    reduced_dataset = xr.Dataset(data_vars, attrs=dict(result.attrs))
    reduced_dataset.attrs["frequency_bands_json"] = _canonical_json(bands)
    reduced_dataset.attrs["frequency_reduction"] = reduction
    return reduced_dataset


def _select_and_reduce_frequencies(
    result: xr.DataArray | xr.Dataset,
    *,
    frequency_range: tuple[float, float] | None,
    frequency_decimation: int,
    frequency_bands: Mapping[str, tuple[float, float]] | None,
    frequency_reduction: Literal["mean", "integral"],
) -> xr.DataArray | xr.Dataset:
    """Apply the wrapper's shared, coordinate-aware frequency operations."""
    if isinstance(frequency_decimation, bool) or not isinstance(
        frequency_decimation, (int, np.integer)
    ):
        raise TypeError("frequency_decimation must be a positive integer.")
    if frequency_decimation < 1:
        raise ValueError("frequency_decimation must be a positive integer.")

    selected = result
    requests_frequency_operation = (
        frequency_range is not None
        or frequency_decimation != 1
        or frequency_bands is not None
    )
    if requests_frequency_operation and "frequency" not in selected.dims:
        raise ValueError(
            "This result has no frequency dimension: the requested method "
            "already reduces frequency (for example phase_slope_index or "
            "group_delay), so frequency_range, frequency_decimation, and "
            "frequency_bands cannot be applied afterward. Pass the method's "
            "frequencies_of_interest argument through connectivity_kwargs instead."
        )
    if frequency_range is not None:
        try:
            lower, upper = frequency_range
        except (TypeError, ValueError) as error:
            raise ValueError(
                "frequency_range must contain exactly two bounds (low, high)."
            ) from error
        lower = float(lower)
        upper = float(upper)
        if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
            raise ValueError(
                "frequency_range must have finite bounds with low <= high."
            )
        frequencies = np.asarray(selected.coords["frequency"].values)
        indices = np.flatnonzero((frequencies >= lower) & (frequencies <= upper))
        if indices.size == 0:
            raise ValueError(
                f"frequency_range ({lower:g}, {upper:g}) contains no bins."
            )
        selected = selected.isel(frequency=indices)
        selected.attrs = dict(selected.attrs)
        selected.attrs["frequency_range_json"] = _canonical_json((lower, upper))

    if frequency_decimation != 1:
        selected = selected.isel(frequency=slice(None, None, frequency_decimation))
        selected.attrs = dict(selected.attrs)
        selected.attrs["frequency_decimation"] = int(frequency_decimation)

    if frequency_bands is not None:
        selected = frequency_band_reduce(
            selected, frequency_bands, reduction=frequency_reduction
        )
    return selected


def connectivity_to_xarray(
    m: Any,
    method: str = "coherence_magnitude",
    signal_names: Sequence[_SignalLabel] | None = None,
    squeeze: bool = False,
    **kwargs: Any,
) -> xr.DataArray | xr.Dataset:
    """Calculate one connectivity measure and return a labeled array.

    Ordinary pairwise measures return a DataArray; component-resolved or
    multi-quantity measures return a Dataset with explicit semantic axes.

    Examples
    --------
    >>> import numpy as np
    >>> from spectral_connectivity.transforms import Multitaper
    >>> data = np.random.default_rng(0).standard_normal((100, 5, 3))
    >>> mt = Multitaper(data, sampling_frequency=1000)
    >>> connectivity_to_xarray(mt).dims
    ('time', 'frequency', 'source', 'target')
    """
    _get_measure_spec(method)
    metadata = m._provenance_metadata()
    connectivity = Connectivity.from_transform(m)
    signal_labels = _validated_signal_labels(signal_names, connectivity.n_signals)
    shared_attrs = _shared_provenance_attrs(
        connectivity,
        metadata,
        transform_prefix=getattr(m, "_provenance_prefix", "mt_"),
    )
    result = _connectivity_result_to_xarray(
        connectivity, method, signal_labels, squeeze, shared_attrs, **kwargs
    )
    valid_time_frequency = getattr(m, "valid_time_frequency", None)
    if valid_time_frequency is not None:
        validity = to_numpy(valid_time_frequency).astype(bool)
        expected_shape = (len(connectivity.time), len(connectivity.frequencies))
        if validity.shape != expected_shape:
            raise ValueError(
                "transform.valid_time_frequency must have shape "
                f"{expected_shape}, got {validity.shape}."
            )
        validity_attrs = {
            "long_name": "Full wavelet and smoothing support is in-record"
        }
        if "frequency" in result.dims:
            full_validity = xr.DataArray(
                validity,
                coords={
                    "time": np.asarray(connectivity.time),
                    "frequency": np.asarray(connectivity.frequencies),
                },
                dims=("time", "frequency"),
                attrs=validity_attrs,
            )
            # Delay and other nonstandard schemas may retain only a requested
            # frequency band. Select the matching validity bins rather than
            # attaching the transform's full frequency axis to the result.
            result_frequencies = np.asarray(result.coords["frequency"])
            aligned_validity = full_validity.sel(frequency=result_frequencies)
            result = result.assign_coords(valid_time_frequency=aligned_validity)
        elif "time" in result.dims:
            # PSI and group delay aggregate a frequency band. They have no
            # frequency dimension on which a 2-D coordinate can live, so expose
            # whether every frequency contributing to each time point has full
            # wavelet/smoothing support.
            frequencies = np.asarray(connectivity.frequencies)
            frequency_band = kwargs.get("frequencies_of_interest")
            if frequency_band is None:
                frequency_index = np.ones(frequencies.shape, dtype=bool)
            else:
                frequency_index = (frequency_band[0] < frequencies) & (
                    frequencies < frequency_band[1]
                )
            valid_time = validity[:, frequency_index].all(axis=1)
            result = result.assign_coords(
                valid_time=(("time",), valid_time, validity_attrs)
            )
    return result


def _combine_formatted_results(
    results: Sequence[xr.DataArray | xr.Dataset],
    shared_attrs: Mapping[str, Any],
) -> xr.Dataset:
    """Merge heterogeneous formatted measures without losing sub-variables."""
    datasets = [
        result.to_dataset(name=result.name)
        if isinstance(result, xr.DataArray)
        else result
        for result in results
    ]
    try:
        combined = xr.merge(datasets, compat="no_conflicts", join="exact")
    except ValueError as error:
        raise ValueError(
            "Requested measures produced conflicting xarray variables or "
            "coordinates; request them separately or use compatible group labels."
        ) from error
    combined.attrs = dict(shared_attrs)
    return combined


def _format_and_reduce_measures(
    connectivity: Connectivity,
    methods: list[str],
    *,
    return_dataarray: bool,
    signal_labels: Sequence[_SignalLabel],
    squeeze: bool,
    shared_attrs: Mapping[str, Any],
    connectivity_kwargs: Mapping[str, Any],
    frequency_range: tuple[float, float] | None,
    frequency_decimation: int,
    frequency_bands: Mapping[str, tuple[float, float]] | None,
    frequency_reduction: Literal["mean", "integral"],
) -> xr.DataArray | xr.Dataset:
    """Format the requested measures to xarray and apply frequency reduction.

    Shared tail of :func:`multitaper_connectivity` and :func:`fourier_connectivity`:
    honors ``squeeze`` only for a single-measure DataArray, formats each measure
    (skipping structurally-unsupported ones in a multi-measure batch), merges the
    survivors, and applies any frequency crop/decimation/band reduction.
    """
    if squeeze and not return_dataarray:
        # squeeze reduces a pairwise measure to a (time, frequency) array whose
        # source/target become scalar coordinates; in a Dataset those scalars are
        # shared across variables and collide with a sibling's axes, so squeeze is
        # honored only for a single-method DataArray.
        warnings.warn(
            "squeeze=True is ignored for multi-measure results (a Dataset); "
            "request a single method (a string) to get a squeezed DataArray.",
            UserWarning,
            stacklevel=3,
        )
        squeeze = False

    if return_dataarray:
        result: xr.DataArray | xr.Dataset = _connectivity_result_to_xarray(
            connectivity,
            methods[0],
            signal_labels,
            squeeze,
            shared_attrs,
            **connectivity_kwargs,
        )
    else:
        formatted_results: list[xr.DataArray | xr.Dataset] = []
        for this_method in methods:
            try:
                formatted_results.append(
                    _connectivity_result_to_xarray(
                        connectivity,
                        this_method,
                        signal_labels,
                        False,
                        shared_attrs,
                        **connectivity_kwargs,
                    )
                )
            except UnsupportedMeasureError as error:
                # A measure whose result shape does not fit the xarray layout can
                # be skipped in a batch. In-package structural incompatibility is
                # surfaced as UnsupportedMeasureError before the measure runs; a
                # genuine NotImplementedError is not caught, so a broken measure
                # fails loudly instead of silently vanishing from the Dataset.
                if len(methods) == 1:
                    raise
                logger.warning("Skipping %s: %s", this_method, error)
        if not formatted_results:
            raise UnsupportedMeasureError(
                "None of the requested methods produced a compatible result "
                f"for the xarray interface: {methods!r}."
            )
        result = _combine_formatted_results(formatted_results, shared_attrs)

    return _select_and_reduce_frequencies(
        result,
        frequency_range=frequency_range,
        frequency_decimation=frequency_decimation,
        frequency_bands=frequency_bands,
        frequency_reduction=frequency_reduction,
    )


# Common dimension names let the wrapper infer semantic roles. DataArrays are
# transposed into the numerical core's (time[, trial], signal) order; callers
# provide explicit ``*_dim`` arguments when their names are domain-specific.
# Integer sample-number names are a time sub-kind, tracked separately for the
# unit distinction in ``_time_axis_from_dataarray`` (they carry no time scale).
_SAMPLE_DIM_NAMES = frozenset({"sample", "samples"})
# Single source of truth mapping each semantic role to its recognized dimension
# names; ``_dimension_role`` is the derived inverse lookup.
_ROLE_SYNONYMS: dict[str, frozenset[str]] = {
    "time": frozenset({"time", "times", "timestamp", "timestamps", *_SAMPLE_DIM_NAMES}),
    "trial": frozenset({"trial", "trials", "epoch", "epochs"}),
    "signal": frozenset(
        {
            "signal",
            "signals",
            "channel",
            "channels",
            "electrode",
            "electrodes",
            "sensor",
            "sensors",
            "node",
            "nodes",
        }
    ),
}
_SYNONYM_TO_ROLE: dict[str, str] = {
    name: role for role, names in _ROLE_SYNONYMS.items() for name in names
}

# Inference is a convenience, so prefer requiring an explicit rate over silently
# deriving a scientifically meaningful frequency scale from a precision-starved
# coordinate. This bounds the endpoint-quantization contribution to the inferred
# rate at 100 parts per million of the observed time span.
_MAX_INFERRED_RATE_RELATIVE_RESOLUTION = 1e-4


def _dimension_role(dimension: Hashable) -> str | None:
    """Return the recognized semantic role of an xarray dimension name."""
    return _SYNONYM_TO_ROLE.get(str(dimension).lower())


def _resolve_dataarray_dimensions(
    time_series: xr.DataArray,
    *,
    time_dim: Hashable | None,
    trial_dim: Hashable | None,
    signal_dim: Hashable | None,
) -> tuple[Hashable, ...]:
    """Resolve semantic dimensions without falling back to unsafe positions."""
    expected_roles = {
        2: ("time", "signal"),
        3: ("time", "trial", "signal"),
    }.get(time_series.ndim)
    if expected_roles is None:
        raise ValueError(
            "A DataArray input must have dimensions (time, signal) or "
            "(time, trial, signal); "
            f"got {time_series.ndim} dimensions {time_series.dims!r}."
        )

    requested = {
        "time": time_dim,
        "trial": trial_dim,
        "signal": signal_dim,
    }
    if trial_dim is not None and "trial" not in expected_roles:
        raise ValueError(
            "trial_dim cannot be used with a 2-D DataArray; a 2-D input has no "
            "trial axis. Use dimensions (time, signal), or provide a 3-D array."
        )

    resolved: dict[str, Hashable] = {}
    used_dimensions: dict[Hashable, str] = {}
    for role in expected_roles:
        dimension = requested[role]
        if dimension is None:
            continue
        if dimension not in time_series.dims:
            raise ValueError(
                f"{role}_dim={dimension!r} is not an input dimension; "
                f"available dimensions are {time_series.dims!r}."
            )
        previous_role = used_dimensions.get(dimension)
        if previous_role is not None:
            raise ValueError(
                f"Dimension {dimension!r} was assigned to both {previous_role}_dim "
                f"and {role}_dim; each semantic role needs a distinct dimension."
            )
        inferred_role = _dimension_role(dimension)
        if inferred_role is not None and inferred_role != role:
            raise ValueError(
                f"{role}_dim={dimension!r} conflicts with its recognized "
                f"{inferred_role} meaning. Rename the dimension or pass the "
                "correct mapping."
            )
        resolved[role] = dimension
        used_dimensions[dimension] = role

    for dimension in time_series.dims:
        if dimension in used_dimensions:
            continue
        inferred_role = _dimension_role(dimension)
        if inferred_role is None:
            continue
        if inferred_role not in expected_roles:
            raise ValueError(
                f"Dimension {dimension!r} denotes a {inferred_role} axis, but a "
                f"{time_series.ndim}-D input has no {inferred_role} axis. Drop "
                "or reshape that dimension."
            )
        if inferred_role in resolved:
            raise ValueError(
                f"Dimensions {resolved[inferred_role]!r} and {dimension!r} both "
                f"denote the {inferred_role} axis. Rename them or pass an "
                "unambiguous dimension mapping."
            )
        resolved[inferred_role] = dimension
        used_dimensions[dimension] = inferred_role

    unresolved_roles = [role for role in expected_roles if role not in resolved]
    unused_dimensions = [
        dimension for dimension in time_series.dims if dimension not in used_dimensions
    ]
    if len(unresolved_roles) == 1 and len(unused_dimensions) == 1:
        # One role and one unrecognized dimension remain, so the mapping is
        # determined by elimination. Warn: whether an axis is the trial axis
        # decides whether it is averaged away, so a wrong guess here silently
        # changes the science. The caller can silence this by naming the role.
        assumed_role = unresolved_roles[0]
        assumed_dimension = unused_dimensions[0]
        warnings.warn(
            f"Assuming DataArray dimension {assumed_dimension!r} is the "
            f"{assumed_role} axis because it is the only unassigned dimension. "
            f"Pass {assumed_role}_dim explicitly to silence this warning, or if "
            "the mapping is wrong.",
            UserWarning,
            stacklevel=4,
        )
        resolved[assumed_role] = assumed_dimension
        unresolved_roles.clear()
    if unresolved_roles:
        arguments = ", ".join(f"{role}_dim" for role in unresolved_roles)
        raise ValueError(
            "Could not infer the semantic roles of DataArray dimensions "
            f"{time_series.dims!r}. Pass {arguments} explicitly; dimension "
            "positions are not used for labeled input."
        )

    return tuple(resolved[role] for role in expected_roles)


def _time_axis_from_dataarray(
    time_series: xr.DataArray,
    time_dimension: Hashable,
    sampling_frequency: float | None,
    explicit_start_time: Any = _UNSET,
) -> _TimeAxis:
    """Resolve the sampling rate and start time from a numeric time index.

    Returns ``(inferred_sampling_frequency, start_time)``. When
    ``sampling_frequency`` is given, the coordinate spacing is validated against
    it and ``inferred_sampling_frequency`` is ``None``. When it is ``None``, the
    rate is inferred from a numeric ``time`` (elapsed-seconds) coordinate; an
    integer ``sample`` coordinate carries no time scale and cannot supply one.
    Returns ``(None, None)`` when the dimension carries no recognized time
    coordinate at all (a non-numeric time coordinate raises instead).
    """
    candidates = [
        (name, coordinate)
        for name, coordinate in time_series.coords.items()
        if coordinate.dims == (time_dimension,)
        and (name == time_dimension or _dimension_role(name) == "time")
    ]
    if not candidates:
        return _TimeAxis(None, None)
    if sampling_frequency is not None and (
        not np.isfinite(sampling_frequency) or sampling_frequency <= 0
    ):
        # This path takes the reciprocal of the rate below; validate up front so
        # a bad value gives a clear message instead of a raw ZeroDivisionError or
        # a misleading coordinate-spacing error.
        raise ValueError(
            "sampling_frequency must be a positive, finite number for a "
            f"DataArray with a numeric time coordinate; got {sampling_frequency!r}."
        )
    exact_time = [item for item in candidates if str(item[0]).lower() == "time"]
    semantic_auxiliary = [
        item
        for item in candidates
        if item[0] != time_dimension and _dimension_role(item[0]) == "time"
    ]
    if len(exact_time) == 1:
        coordinate_name, coordinate = exact_time[0]
    elif not exact_time and len(semantic_auxiliary) == 1:
        coordinate_name, coordinate = semantic_auxiliary[0]
    elif not exact_time and len(candidates) == 1:
        coordinate_name, coordinate = candidates[0]
    else:
        # Falls through here when several coordinates are case-insensitively
        # "time" (len(exact_time) > 1): that is genuinely ambiguous, not a cue to
        # silently prefer an auxiliary coordinate.
        coordinate_names = [name for name, _ in candidates]
        raise ValueError(
            f"Multiple coordinates {coordinate_names!r} could label time "
            f"dimension {time_dimension!r}. Keep one time-like coordinate or "
            "rename the others so the intended elapsed-seconds coordinate is "
            "unambiguous."
        )

    values = np.asarray(coordinate.to_numpy())
    if (
        not np.issubdtype(values.dtype, np.number)
        or np.issubdtype(values.dtype, np.complexfloating)
        or np.issubdtype(values.dtype, np.bool_)
        or np.issubdtype(values.dtype, np.datetime64)
        or np.issubdtype(values.dtype, np.timedelta64)
    ):
        raise TypeError(
            f"The DataArray time coordinate {coordinate_name!r} must contain "
            "numeric elapsed seconds, or integer-like sample numbers for a "
            "'sample' coordinate. Datetime, timedelta, complex, boolean, and "
            f"object time coordinates are not yet supported (got dtype "
            f"{values.dtype!r}). Convert a datetime axis to elapsed seconds, "
            "e.g. (da.time - da.time[0]) / np.timedelta64(1, 's')."
        )

    times = values.astype(np.float64, copy=False)
    if times.size == 0:
        raise ValueError(
            f"The DataArray time coordinate {coordinate_name!r} must not be empty."
        )
    if not np.all(np.isfinite(times)):
        raise ValueError(
            f"The DataArray time coordinate {coordinate_name!r} must contain "
            "only finite values."
        )
    differences = np.diff(times)
    if np.any(differences <= 0):
        raise ValueError(
            f"The DataArray time coordinate {coordinate_name!r} must be strictly "
            "increasing."
        )

    coordinate_is_sample_index = str(coordinate_name).lower() in _SAMPLE_DIM_NAMES
    if coordinate_is_sample_index and not np.all(times == np.rint(times)):
        raise ValueError(
            f"The DataArray sample coordinate {coordinate_name!r} must contain "
            "integer-like sample numbers. Use a 'time' coordinate for elapsed "
            "fractional seconds."
        )
    inferred_sampling_frequency: float | None = None
    if sampling_frequency is None:
        # Infer the rate from an elapsed-seconds coordinate. Integer sample
        # numbers have no time scale, so they cannot supply one.
        if coordinate_is_sample_index:
            raise ValueError(
                f"Cannot infer sampling_frequency from the integer sample "
                f"coordinate {coordinate_name!r}, which has no time scale. Pass "
                "sampling_frequency, or use a numeric 'time' coordinate in "
                "elapsed seconds."
            )
        if times.size < 2:
            raise ValueError(
                "Cannot infer sampling_frequency from a single-sample time "
                f"coordinate {coordinate_name!r}; pass sampling_frequency."
            )
        # Span-based estimate averages float noise over the whole (uniform) grid.
        coordinate_span = float(times[-1]) - float(times[0])
        if not np.isfinite(coordinate_span) or coordinate_span <= 0:
            raise ValueError(
                "Cannot infer sampling_frequency: the DataArray time coordinate "
                f"{coordinate_name!r} does not have a finite positive span. Pass "
                "sampling_frequency explicitly."
            )
        if np.issubdtype(values.dtype, np.floating):
            # Estimate the resolution of the stored endpoints in their original
            # dtype. If one representable step is material relative to the whole
            # span, the reciprocal interval would report false precision (e.g. a
            # float32 1-kHz axis at a large offset can appear to be 1024 Hz).
            storage_scale = float(np.max(np.abs(times), initial=0.0))
            storage_resolution = abs(
                float(np.spacing(np.asarray(storage_scale, dtype=values.dtype)))
            )
            relative_resolution = storage_resolution / coordinate_span
            if (
                not np.isfinite(relative_resolution)
                or relative_resolution > _MAX_INFERRED_RATE_RELATIVE_RESOLUTION
            ):
                raise ValueError(
                    "Cannot reliably infer sampling_frequency from DataArray time "
                    f"coordinate {coordinate_name!r}: its {values.dtype} resolution "
                    f"({storage_resolution!r} s) is too large relative to the "
                    f"observed span ({coordinate_span!r} s). Pass "
                    "sampling_frequency explicitly, or use a higher-precision or "
                    "zero-based elapsed-seconds coordinate."
                )
        expected_interval = coordinate_span / (times.size - 1)
        if (
            not np.isfinite(expected_interval)
            or expected_interval <= 0
            or expected_interval < 1.0 / np.finfo(np.float64).max
        ):
            raise ValueError(
                "Cannot infer sampling_frequency: the DataArray time coordinate "
                f"{coordinate_name!r} implies a non-finite sampling rate. Pass "
                "sampling_frequency explicitly."
            )
        inferred_sampling_frequency = 1.0 / expected_interval
    else:
        expected_interval = (
            1.0 if coordinate_is_sample_index else 1.0 / float(sampling_frequency)
        )
    coordinate_scale = max(float(np.max(np.abs(times), initial=0.0)), 1.0)
    coordinate_resolution = (
        abs(float(np.spacing(np.asarray(coordinate_scale, dtype=values.dtype))))
        if np.issubdtype(values.dtype, np.floating)
        else 0.0
    )
    coordinate_tolerance = max(
        expected_interval * 1e-9,
        coordinate_resolution,
        np.spacing(coordinate_scale) * 8,
    )
    expected_coordinates = times[0] + np.arange(times.size) * expected_interval
    if not np.allclose(
        times,
        expected_coordinates,
        rtol=0,
        atol=coordinate_tolerance,
    ):
        observed_median = float(np.median(differences))
        if sampling_frequency is None:
            # Inference requires a regular grid; an irregular one has no single
            # rate to derive.
            raise ValueError(
                f"Cannot infer sampling_frequency: the DataArray time coordinate "
                f"{coordinate_name!r} is not uniformly spaced (observed median "
                f"step {observed_median!r} s). Pass sampling_frequency "
                "explicitly, or provide a regularly sampled time coordinate."
            )
        expected_description = (
            "1 sample per coordinate step"
            if coordinate_is_sample_index
            else f"{expected_interval!r} seconds per sample"
        )
        raise ValueError(
            f"The DataArray time coordinate spacing does not match "
            f"sampling_frequency={sampling_frequency!r} Hz (expected "
            f"{expected_description}, observed median {observed_median!r})."
        )

    if coordinate_is_sample_index:
        # A sample coordinate only reaches here with an explicit rate; inference
        # already rejected it above.
        assert sampling_frequency is not None
        time_scale = float(sampling_frequency)
    else:
        time_scale = 1.0
    inferred_start_time = float(times[0]) / time_scale
    start_time_tolerance = coordinate_tolerance / time_scale
    if explicit_start_time is not _UNSET:
        explicit = to_numpy(explicit_start_time)
        if explicit.size != 1:
            raise ValueError(
                "A DataArray with one time coordinate requires scalar start_time; "
                f"got shape {explicit.shape}."
            )
        explicit_value = float(explicit.reshape(-1)[0])
        if not np.isclose(
            explicit_value,
            inferred_start_time,
            rtol=0,
            atol=start_time_tolerance,
        ):
            raise ValueError(
                f"start_time={explicit_value!r} conflicts with the first "
                f"DataArray time coordinate {inferred_start_time!r}. Remove "
                "start_time or make the values agree."
            )
    return _TimeAxis(inferred_sampling_frequency, inferred_start_time)


def _signal_labels_from_dataarray(
    time_series: xr.DataArray, signal_dimension: Hashable
) -> Sequence[_SignalLabel] | None:
    """Signal labels from a 1-D index coordinate on the signal dimension.

    Returns ``None`` (default string, i.e. stringified-index, labels used
    downstream) when the signal dimension has no usable 1-D index coordinate. If
    the DataArray *does* carry coordinates along that dimension but none is a
    usable 1-D index coordinate, warn rather than silently dropping the user's
    labels.
    """
    # Membership, not ``coords.get``: ``.get`` fabricates a default integer
    # index for a bare dimension, which would mask the no-coordinate case.
    if signal_dimension in time_series.coords:
        index_coordinate = time_series.coords[signal_dimension]
        if index_coordinate.dims == (signal_dimension,):
            # ``list(ndarray)`` retains NumPy datetime/timedelta scalars, whereas
            # ``ndarray.tolist()`` can coerce nanosecond values to bare integers.
            return list(index_coordinate.to_numpy())

    has_unusable_labels = any(
        signal_dimension in coordinate.dims
        for coordinate in time_series.coords.values()
    )
    if has_unusable_labels:
        warnings.warn(
            f"The input DataArray has coordinates along its signal dimension "
            f"{signal_dimension!r} that are not a 1-D index coordinate, so "
            f"signal labels could not be inferred; default string labels will "
            f"be used. Pass ``signal_names`` explicitly, or attach a 1-D "
            f"coordinate named {signal_dimension!r} to label the output "
            f"source/target axes.",
            stacklevel=4,
        )
    return None


def _reject_unmaterialized_backing(data: Any) -> None:
    """Reject a lazy backing array the positional spectral math cannot consume.

    xarray materializes a masked array to a NaN-filled ndarray on construction,
    so a mask surfaces loudly as NaN downstream and needs no guard here. A dask
    array, by contrast, is handed through ``.data`` unmaterialized.
    """
    if callable(getattr(data, "__dask_graph__", None)):
        raise TypeError(
            "multitaper_connectivity received a dask-backed DataArray, which is "
            "not supported. Materialize it first with DataArray.compute() (or "
            "DataArray.load()) and pass the result."
        )


def _unwrap_xarray_input(
    time_series: NDArray[np.floating] | xr.DataArray,
    signal_names: Sequence[_SignalLabel] | None,
    sampling_frequency: float | None,
    *,
    time_dim: Hashable | None,
    trial_dim: Hashable | None,
    signal_dim: Hashable | None,
    explicit_start_time: Any = _UNSET,
) -> _UnwrappedInput:
    """Extract array data and, when available, labels from a DataArray input.

    Semantic dimensions are inferred from common names or supplied explicitly,
    then transposed into the numerical core's positional order. A numeric time
    index supplies ``start_time`` and, when ``sampling_frequency`` is omitted,
    the sampling rate; when the rate is given it is validated against the index.
    Returns ``(data, signal_names, inferred_sampling_frequency,
    inferred_start_time)``.
    """
    if not isinstance(time_series, xr.DataArray):
        if any(
            dimension is not None for dimension in (time_dim, trial_dim, signal_dim)
        ):
            raise TypeError(
                "time_dim, trial_dim, and signal_dim apply only to an "
                "xarray.DataArray input."
            )
        return _UnwrappedInput(time_series, signal_names, None, None, None)

    dimension_order = _resolve_dataarray_dimensions(
        time_series,
        time_dim=time_dim,
        trial_dim=trial_dim,
        signal_dim=signal_dim,
    )
    time_dimension = dimension_order[0]
    signal_dimension = dimension_order[-1]
    if signal_names is None:
        signal_names = _signal_labels_from_dataarray(time_series, signal_dimension)
    inferred_sampling_frequency, inferred_start_time = _time_axis_from_dataarray(
        time_series,
        time_dimension,
        sampling_frequency,
        explicit_start_time,
    )

    data = time_series.transpose(*dimension_order).data
    _reject_unmaterialized_backing(data)
    return _UnwrappedInput(
        data,
        signal_names,
        inferred_sampling_frequency,
        inferred_start_time,
        dict(time_series.attrs),
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
    frequency_range: tuple[float, float] | None = None,
    frequency_decimation: int = 1,
    frequency_bands: Mapping[str, tuple[float, float]] | None = None,
    frequency_reduction: Literal["mean", "integral"] = "mean",
    time_dim: Hashable | None = None,
    trial_dim: Hashable | None = None,
    signal_dim: Hashable | None = None,
    **kwargs: Any,
) -> xr.DataArray | xr.Dataset:
    """
    Compute connectivity measures with multitaper spectral estimation.

    This is the main high-level function for connectivity analysis. It performs
    multitaper spectral analysis on the input time series and computes the
    requested connectivity measures, returning results as labeled xarray objects.

    Parameters
    ----------
    time_series : NDArray[floating] or xarray.DataArray,
        shape (n_times, n_trials, n_channels) or (n_times, n_channels)
        Time series data. For multiple trials, trials are averaged in spectral
        domain. For a DataArray, common time/trial/signal dimension names are
        inferred and transposed automatically; use ``time_dim``, ``trial_dim``,
        and ``signal_dim`` for domain-specific names. Ambiguous names raise rather
        than falling back to dimension position, though a single unrecognized
        dimension left for the one remaining role is assigned by elimination with
        a warning. A dask-backed DataArray is rejected (materialize it first with
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
    time_window_duration : float, optional
        Duration of sliding window in seconds for time-resolved analysis.
        If None, analyzes entire time series (no time resolution).
    method : str or list of str, optional
        Connectivity method(s) to compute. If None, computes the default set of
        real-valued measures that fit the xarray/NetCDF interface (see
        ``DEFAULT_METHODS``) — not every measure. ``coherency`` is left out of the
        default because complex arrays are not portably serializable across all
        supported xarray versions and NetCDF engines, but it can be requested by
        name. The directed-transfer-function family
        (``directed_transfer_function``, ``directed_coherence``,
        ``partial_directed_coherence``, ``generalized_partial_directed_coherence``,
        ``direct_directed_transfer_function``) is also opt-in by name (see the
        Notes on directed orientation). Measures with nonstandard layouts,
        including ``global_coherence``, ``phase_slope_index``, ``group_delay``,
        ``delay``, ``canonical_coherence``, and blockwise spectral Granger, are
        available by name and return labeled DataArrays or Datasets with their
        component, group, candidate-delay, or frequency-reduced dimensions.
        Examples:
        "coherence_magnitude", "imaginary_coherence", "phase_locking_value".
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
        Additional keyword arguments passed to connectivity methods.
    frequency_range : (float, float), optional
        Inclusive frequency interval retained in the labeled result.
    frequency_decimation : int, default=1
        Keep every Nth frequency bin after applying ``frequency_range``.
    frequency_bands : mapping of str to (float, float), optional
        Reduce the selected bins into named, inclusive bands. With
        ``frequency_reduction="mean"``, scores are averaged, complex measures
        use a complex vector mean, and ``coherence_phase`` uses a circular mean.
    frequency_reduction : {"mean", "integral"}, default="mean"
        Band reduction. Integration is restricted to ``power`` and
        ``cross_spectral_density``, where it yields band power/covariance.
    time_dim : hashable, optional
        DataArray dimension containing time samples. Common names such as
        ``"time"`` and ``"sample"`` are inferred automatically.
    trial_dim : hashable, optional
        DataArray dimension containing trials or epochs. Required for a 3-D
        DataArray when its role cannot be inferred unambiguously.
    signal_dim : hashable, optional
        DataArray dimension containing signals or channels. Common names such as
        ``"signal"`` and ``"channel"`` are inferred automatically.
    **kwargs : dict
        Additional arguments passed to the Multitaper constructor
        (e.g., time_halfbandwidth_product, n_tapers, n_fft_samples,
        fft_workers=-1 to parallelize the CPU FFT across all cores).

    Returns
    -------
    result : xarray.DataArray or xarray.Dataset
        A plain single-quantity method returns a DataArray. Component-resolved
        and multi-quantity methods return a Dataset even when requested alone;
        multiple methods are merged into one Dataset without flattening their
        semantic dimensions.

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
    ...     method="coherence_magnitude",
    ...     signal_names=["Signal_1", "Signal_2"]
    ... )
    >>> coherence.dims
    ('time', 'frequency', 'source', 'target')

    >>> # Compute multiple measures
    >>> measures = multitaper_connectivity(
    ...     data, sampling_frequency=500,
    ...     method=["coherence_magnitude", "imaginary_coherence"]
    ... )
    >>> list(measures.data_vars)
    ['coherence_magnitude', 'imaginary_coherence']

    >>> # An xarray.DataArray labels axes by dimension name and can supply the
    >>> # sampling rate and channel labels itself (no sampling_frequency needed).
    >>> import xarray as xr
    >>> da = xr.DataArray(
    ...     data,
    ...     dims=("time", "channel"),
    ...     coords={"time": t, "channel": ["Signal_1", "Signal_2"]},
    ... )
    >>> coherence = multitaper_connectivity(da, method="coherence_magnitude")
    >>> coherence.coords["source"].values.tolist()
    ['Signal_1', 'Signal_2']

    Notes
    -----
    Uses multitaper spectral estimation for robust power spectral density
    estimation before computing connectivity measures. This provides better
    spectral estimates than single-taper methods, especially for short time series.

    For directed measures (e.g. ``pairwise_spectral_granger_prediction``) the
    ``source`` and ``target`` axes are oriented so that
    ``result.sel(source=a, target=b)`` is the influence *from* ``a`` *to* ``b``.
    (The underlying ``Connectivity`` methods use the transposed convention
    ``output[i, j] = influence j -> i``; the wrapper transposes to the intuitive
    source -> target layout.)

    The result records provenance as NetCDF-safe attributes so a saved file is
    self-describing:

    - ``mt_*`` -- the Multitaper transform parameters.
    - ``measure`` and ``measure_kwargs_json`` -- the measure name and a canonical,
      JSON-normalized representation of its keyword arguments.
    - ``arg_<key>`` / ``arg_<key>_json`` -- each measure keyword argument
      individually for quick inspection; a scalar is stored as-is under
      ``arg_<key>``, while a structured or non-finite value is stored as a JSON
      string under ``arg_<key>_json`` (parse with ``json.loads``;
      ``measure_kwargs_json`` is the canonical record).
    - ``package``, ``package_version``, ``backend``, ``expectation_type`` --
      software provenance.
    - ``input_attrs_json`` -- a canonical, JSON-normalized record of attributes
      carried over from an input ``xarray.DataArray`` (e.g. subject or session
      metadata). Keeping the complete mapping in one record preserves arbitrary
      keys without collisions or invalid NetCDF attribute names.

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
    explicit_start_time = kwargs.get("start_time", _UNSET)
    (
        time_series_data,
        signal_names,
        inferred_sampling_frequency,
        inferred_start_time,
        input_attrs,
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
        raise ValueError(
            "sampling_frequency is required unless the input is an "
            "xarray.DataArray with a numeric 'time' coordinate (in elapsed "
            "seconds) to infer it from."
        )
    if inferred_start_time is not None and explicit_start_time is _UNSET:
        kwargs["start_time"] = inferred_start_time
    if connectivity_kwargs is None:
        connectivity_kwargs = {}
    return_dataarray = False  # Default: return dataset
    if method is None:
        # The explicit, portably serializable / xarray-compatible default set
        # (see DEFAULT_METHODS). Complex, component/group, frequency-reduced,
        # and directed-transfer-function results remain opt-in by name.
        method = list(DEFAULT_METHODS)
    elif isinstance(method, str):
        method = [method]  # Convert to list
        return_dataarray = True  # Return dataarray if methods was not an iterable
    else:
        method = list(method)
    if len(method) == 0:
        raise ValueError(
            "method must name at least one connectivity measure; got an empty list."
        )
    _validate_method_names(method)
    # Accept the documented (n_times, n_channels) 2-D form by inserting a
    # singleton trial axis; Multitaper requires 3-D (n_times, n_trials,
    # n_signals).
    if getattr(time_series_data, "ndim", None) == 2:
        time_series_data = time_series_data[:, np.newaxis, :]
    m = Multitaper(
        time_series=time_series_data,
        sampling_frequency=sampling_frequency,
        time_window_duration=time_window_duration,
        **kwargs,
    )
    # Capture metadata and build the shared calculation object from the same
    # immutable transform. The private formatter below never accepts a separate
    # Multitaper, so data and labels cannot be paired accidentally.
    metadata = m._provenance_metadata()
    shared_connectivity = Connectivity.from_multitaper(m)
    # Validate labels and build shared provenance once; both are invariant across
    # the requested measures.
    signal_labels = _validated_signal_labels(
        signal_names, shared_connectivity.n_signals
    )
    shared_attrs = _shared_provenance_attrs(
        shared_connectivity, metadata, input_attrs=input_attrs
    )
    return _format_and_reduce_measures(
        shared_connectivity,
        method,
        return_dataarray=return_dataarray,
        signal_labels=signal_labels,
        squeeze=squeeze,
        shared_attrs=shared_attrs,
        connectivity_kwargs=connectivity_kwargs,
        frequency_range=frequency_range,
        frequency_decimation=frequency_decimation,
        frequency_bands=frequency_bands,
        frequency_reduction=frequency_reduction,
    )


_FOURIER_ROLE_SYNONYMS: dict[str, frozenset[str]] = {
    "time": frozenset(
        {"time", "times", "window", "windows", "time_window", "time_windows"}
    ),
    "trial": _ROLE_SYNONYMS["trial"] | frozenset({"observation", "observations"}),
    "taper": frozenset({"taper", "tapers"}),
    "frequency": frozenset({"frequency", "frequencies", "freq", "freqs"}),
    "signal": _ROLE_SYNONYMS["signal"],
}


def _coordinates_agree(explicit: Any, labeled: NDArray[Any]) -> bool:
    """Whether an explicit numeric coordinate matches its DataArray coordinate."""
    explicit_values = np.asarray(explicit)
    labeled_values = np.asarray(labeled)
    if explicit_values.shape != labeled_values.shape:
        return False
    try:
        return bool(np.allclose(explicit_values, labeled_values, rtol=1e-12, atol=0))
    except TypeError:
        return bool(np.array_equal(explicit_values, labeled_values))


def _unwrap_fourier_input(
    fourier_coefficients: NDArray[np.complexfloating] | xr.DataArray,
    *,
    frequencies: NDArray[np.floating] | None,
    time: NDArray[np.floating] | None,
    signal_names: Sequence[_SignalLabel] | None,
    time_dim: Hashable | None,
    trial_dim: Hashable | None,
    taper_dim: Hashable | None,
    frequency_dim: Hashable | None,
    signal_dim: Hashable | None,
) -> tuple[
    BackendArray,
    NDArray[np.floating] | None,
    NDArray[np.floating] | None,
    Sequence[_SignalLabel] | None,
    Mapping[Any, Any] | None,
]:
    """Normalize external coefficients to the core's five-dimensional layout."""
    dimension_arguments = {
        "time": time_dim,
        "trial": trial_dim,
        "taper": taper_dim,
        "frequency": frequency_dim,
        "signal": signal_dim,
    }
    if not isinstance(fourier_coefficients, xr.DataArray):
        if any(dimension is not None for dimension in dimension_arguments.values()):
            raise TypeError(
                "The *_dim arguments apply only to an xarray.DataArray input."
            )
        data = fourier_coefficients
        ndim = getattr(data, "ndim", None)
        if ndim == 3:
            # (observation, frequency, signal)
            data = data[np.newaxis, :, np.newaxis, :, :]
        elif ndim == 4:
            # (trial, taper, frequency, signal)
            data = data[np.newaxis, :, :, :, :]
        elif ndim != 5:
            raise ValueError(
                "fourier_coefficients must have 3, 4, or 5 dimensions: "
                "(observation, frequency, signal), (trial, taper, frequency, "
                "signal), or (time, trial, taper, frequency, signal)."
            )
        return data, frequencies, time, signal_names, None

    coefficient_array = fourier_coefficients
    if coefficient_array.ndim < 3 or coefficient_array.ndim > 5:
        raise ValueError("A Fourier coefficient DataArray must have 3 to 5 dimensions.")
    _reject_unmaterialized_backing(coefficient_array.data)

    role_to_dimension: dict[str, Hashable] = {}
    claimed_dimensions: set[Hashable] = set()
    for role, dimension in dimension_arguments.items():
        if dimension is None:
            continue
        if dimension not in coefficient_array.dims:
            raise ValueError(
                f"{role}_dim={dimension!r} is not one of the DataArray "
                f"dimensions {coefficient_array.dims!r}."
            )
        if dimension in claimed_dimensions:
            raise ValueError(
                f"DataArray dimension {dimension!r} was assigned to more than one role."
            )
        role_to_dimension[role] = dimension
        claimed_dimensions.add(dimension)

    for role, synonyms in _FOURIER_ROLE_SYNONYMS.items():
        if role in role_to_dimension:
            continue
        candidates = [
            dimension
            for dimension in coefficient_array.dims
            if dimension not in claimed_dimensions
            and str(dimension).lower() in synonyms
        ]
        if len(candidates) > 1:
            raise ValueError(
                f"Multiple dimensions look like the Fourier {role} axis: "
                f"{candidates!r}. Pass {role}_dim explicitly."
            )
        if candidates:
            role_to_dimension[role] = candidates[0]
            claimed_dimensions.add(candidates[0])

    for required_role in ("frequency", "signal"):
        if required_role not in role_to_dimension:
            raise ValueError(
                f"Could not identify the Fourier {required_role} dimension. "
                f"Use {required_role}_dim=... explicitly."
            )

    unclaimed = [
        dimension
        for dimension in coefficient_array.dims
        if dimension not in claimed_dimensions
    ]
    # A lone coefficient-observation dimension is unambiguously a trial axis.
    # More than one unnamed observation axis could be time/trial/taper in several
    # scientifically different ways, so require the caller to label it.
    if len(unclaimed) == 1 and "trial" not in role_to_dimension:
        role_to_dimension["trial"] = unclaimed.pop()
    if unclaimed:
        raise ValueError(
            f"Could not infer the roles of Fourier dimensions {unclaimed!r}. "
            "Name them time/trial/taper, or pass the corresponding *_dim arguments."
        )

    ordered_roles = ("time", "trial", "taper", "frequency", "signal")
    present_dimensions = [
        role_to_dimension[role] for role in ordered_roles if role in role_to_dimension
    ]
    data = coefficient_array.transpose(*present_dimensions).data
    if "time" not in role_to_dimension:
        data = data[np.newaxis, ...]
    if "trial" not in role_to_dimension:
        data = data[:, np.newaxis, ...]
    if "taper" not in role_to_dimension:
        data = data[:, :, np.newaxis, ...]

    frequency_dimension = role_to_dimension["frequency"]
    has_frequency_coordinate = frequency_dimension in coefficient_array.coords
    frequency_coordinate_is_1d = has_frequency_coordinate and coefficient_array.coords[
        frequency_dimension
    ].dims == (frequency_dimension,)
    coordinate_frequencies = (
        coefficient_array.coords[frequency_dimension].to_numpy()
        if frequency_coordinate_is_1d
        else None
    )
    if (
        has_frequency_coordinate
        and not frequency_coordinate_is_1d
        and frequencies is None
    ):
        warnings.warn(
            f"The DataArray frequency coordinate {frequency_dimension!r} is not "
            "one-dimensional and was ignored; the result falls back to normalized "
            "FFT-bin labels. Pass a 1-D `frequencies` array to keep meaningful "
            "frequency labels.",
            UserWarning,
            stacklevel=3,
        )
    if frequencies is None:
        frequencies = coordinate_frequencies
    elif coordinate_frequencies is not None and not _coordinates_agree(
        frequencies, coordinate_frequencies
    ):
        raise ValueError(
            "frequencies conflicts with the DataArray frequency coordinate."
        )

    if "time" in role_to_dimension:
        time_dimension = role_to_dimension["time"]
        coordinate_time = (
            coefficient_array.coords[time_dimension].to_numpy()
            if time_dimension in coefficient_array.coords
            and coefficient_array.coords[time_dimension].dims == (time_dimension,)
            else None
        )
        if time is None:
            time = coordinate_time
        elif coordinate_time is not None and not _coordinates_agree(
            time, coordinate_time
        ):
            raise ValueError("time conflicts with the DataArray time coordinate.")

    if signal_names is None:
        signal_names = _signal_labels_from_dataarray(
            coefficient_array, role_to_dimension["signal"]
        )
    return data, frequencies, time, signal_names, dict(coefficient_array.attrs)


def fourier_connectivity(
    fourier_coefficients: NDArray[np.complexfloating] | xr.DataArray,
    frequencies: NDArray[np.floating] | None = None,
    time: NDArray[np.floating] | None = None,
    method: str | list[str] | None = None,
    signal_names: Sequence[_SignalLabel] | None = None,
    squeeze: bool = False,
    connectivity_kwargs: dict[str, Any] | None = None,
    expectation_type: Literal["trials_tapers"] = "trials_tapers",
    is_one_sided: bool | None = None,
    *,
    frequency_range: tuple[float, float] | None = None,
    frequency_decimation: int = 1,
    frequency_bands: Mapping[str, tuple[float, float]] | None = None,
    frequency_reduction: Literal["mean", "integral"] = "mean",
    time_dim: Hashable | None = None,
    trial_dim: Hashable | None = None,
    taper_dim: Hashable | None = None,
    frequency_dim: Hashable | None = None,
    signal_dim: Hashable | None = None,
    dtype: np.dtype = np.dtype(np.complex128),
    minimum_phase_tolerance: float = 1e-8,
    minimum_phase_max_iterations: int = 500,
) -> xr.DataArray | xr.Dataset:
    """Compute labeled connectivity from externally estimated FFT coefficients.

    NumPy-like inputs may use ``(observation, frequency, signal)``, ``(trial,
    taper, frequency, signal)``, or the core's full ``(time, trial, taper,
    frequency, signal)`` layout. A DataArray is transposed by semantic dimension
    names (or explicit ``*_dim`` arguments), and its frequency, time, signal
    coordinates and attributes are preserved. When ``is_one_sided`` is omitted,
    a supplied non-negative, increasing frequency coordinate is recognized as a
    one-sided transform. One-sided coefficients support functional connectivity,
    but the core rejects directed measures that require a full two-sided spectrum.
    Set ``is_one_sided`` explicitly when no frequency coordinate is available.

    The xarray layout currently requires ``expectation_type="trials_tapers"``;
    use :class:`Connectivity` directly for expectations that retain trial/taper
    axes or average over time.
    """
    if expectation_type != "trials_tapers":
        raise ValueError(
            "fourier_connectivity supports expectation_type='trials_tapers' "
            "because the labeled output has one time axis; use Connectivity "
            "directly for other expectation layouts."
        )
    (
        coefficient_data,
        frequencies,
        time,
        signal_names,
        input_attrs,
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
    if getattr(getattr(coefficient_data, "dtype", None), "kind", None) != "c":
        raise TypeError("fourier_coefficients must be complex-valued.")
    inferred_one_sided = False
    if is_one_sided is not None and not isinstance(is_one_sided, (bool, np.bool_)):
        raise TypeError("is_one_sided must be a boolean or None.")
    if frequencies is not None:
        frequency_values = np.asarray(frequencies, dtype=float)
        if frequency_values.ndim != 1:
            raise ValueError("frequencies must be a one-dimensional coordinate.")
        inferred_one_sided = bool(
            frequency_values.size > 1 and not np.any(frequency_values < 0)
        )
        one_sided = inferred_one_sided if is_one_sided is None else bool(is_one_sided)
        if one_sided:
            if np.any(frequency_values < 0) or (
                frequency_values.size > 1 and not np.all(np.diff(frequency_values) > 0)
            ):
                raise ValueError(
                    "One-sided frequencies must be non-negative and strictly "
                    "increasing."
                )
        elif frequency_values.size > 1:
            frequency_step = (
                frequency_values[1] - frequency_values[0]
                if frequency_values.size > 2
                else abs(frequency_values[1])
            )
            expected_frequencies = np.fft.fftfreq(
                frequency_values.size,
                d=1.0 / (frequency_step * frequency_values.size),
            )
            tolerance = max(abs(frequency_step) * 1e-9, np.finfo(float).eps)
            if frequency_step <= 0 or not np.allclose(
                frequency_values,
                expected_frequencies,
                rtol=1e-9,
                atol=tolerance,
            ):
                raise ValueError(
                    "frequencies must be uniformly spaced in standard FFT "
                    "order (zero and positive bins followed by negative bins)."
                )
    else:
        one_sided = bool(is_one_sided) if is_one_sided is not None else False

    connectivity = Connectivity(
        coefficient_data,
        expectation_type=expectation_type,
        frequencies=frequencies,
        time=time,
        dtype=dtype,
        minimum_phase_tolerance=minimum_phase_tolerance,
        minimum_phase_max_iterations=minimum_phase_max_iterations,
        is_one_sided=one_sided,
    )
    if connectivity_kwargs is None:
        connectivity_kwargs = {}

    return_dataarray = isinstance(method, str)
    if method is None:
        methods = [
            name
            for name in DEFAULT_METHODS
            if not (
                one_sided
                and _MEASURE_SPECS.get(name, _UNSUPPORTED_SPEC).requires_two_sided
            )
        ]
    elif isinstance(method, str):
        methods = [method]
    else:
        methods = list(method)
    if not methods:
        raise ValueError(
            "method must name at least one connectivity measure; got an empty list."
        )
    _validate_method_names(methods)
    if frequencies is None:
        # Without a frequency coordinate, orientation and two-sidedness cannot be
        # verified, so the default ``is_one_sided=False`` lets a one-sided input
        # (e.g. rfft/wavelet coefficients) reach Wilson factorization and produce
        # a silently wrong Wilson-factorized result. Reject methods that declare
        # the full-spectrum requirement; other directional measures such as dPLI
        # and PSI remain valid on one-sided coefficients.
        two_sided_methods = [
            name
            for name in methods
            if _MEASURE_SPECS.get(name, _UNSUPPORTED_SPEC).requires_two_sided
        ]
        if two_sided_methods and one_sided:
            # The caller already declared one-sided input, so no frequency vector
            # would enable Wilson factorization -- give the accurate reason.
            raise ValueError(
                f"Measures {sorted(set(two_sided_methods))} require a full "
                "two-sided spectrum in standard FFT order. One-sided transforms "
                "(is_one_sided=True) support functional connectivity measures but "
                "not Wilson-factorized measures. Request only one-sided-compatible "
                "measures, or supply full two-sided coefficients."
            )
        if two_sided_methods:
            raise ValueError(
                f"Measures {sorted(set(two_sided_methods))} require a full "
                "two-sided spectrum in standard FFT order, which cannot be verified "
                "without a frequency coordinate. Pass `frequencies` (the FFT "
                "frequency vector, including negative bins) so two-sidedness can be "
                "checked, or request only one-sided-compatible measures."
            )
    signal_labels = _validated_signal_labels(signal_names, connectivity.n_signals)
    metadata = {
        "source": "external_fourier_coefficients",
        "coefficient_shape_json": _canonical_json(tuple(coefficient_data.shape)),
        "frequency_coordinate": "provided" if frequencies is not None else "normalized",
        "time_coordinate": "provided" if time is not None else "index",
        "is_one_sided": one_sided,
        "one_sided_inferred": is_one_sided is None and inferred_one_sided,
    }
    shared_attrs = _shared_provenance_attrs(
        connectivity,
        metadata,
        input_attrs=input_attrs,
        transform_prefix="fourier_",
    )
    return _format_and_reduce_measures(
        connectivity,
        methods,
        return_dataarray=return_dataarray,
        signal_labels=signal_labels,
        squeeze=squeeze,
        shared_attrs=shared_attrs,
        connectivity_kwargs=connectivity_kwargs,
        frequency_range=frequency_range,
        frequency_decimation=frequency_decimation,
        frequency_bands=frequency_bands,
        frequency_reduction=frequency_reduction,
    )
