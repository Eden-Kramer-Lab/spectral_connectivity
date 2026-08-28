"""Functions for getting connectivity measures in a labeled array format."""

import json
import warnings
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from logging import getLogger
from typing import Any, Literal, TypeAlias

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from spectral_connectivity.connectivity import Connectivity
from spectral_connectivity.transforms import Multitaper
from spectral_connectivity.utils import BackendArray, get_compute_backend, to_numpy

logger = getLogger(__name__)

_UNSET = object()

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
        return _json_compatible(value.tolist())
    if isinstance(value, Mapping):
        if all(isinstance(key, str) for key in value):
            return {key: _json_compatible(item) for key, item in sorted(value.items())}
        converted_items = [
            [_json_compatible(key), _json_compatible(item)]
            for key, item in value.items()
        ]
        converted_items.sort(
            key=lambda pair: json.dumps(
                pair[0],
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
                allow_nan=False,
            )
        )
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
    """
    netcdf_value = _netcdf_provenance_value(value)
    if isinstance(netcdf_value, str) and not isinstance(value, str):
        attrs[f"{prefix}{key}_json"] = netcdf_value
    else:
        attrs[f"{prefix}{key}"] = netcdf_value


class UnsupportedMeasureError(ValueError):
    """A measure does not fit the ``(time, frequency, source, target)`` layout.

    Raised by :func:`connectivity_to_xarray` for measures that cannot be
    represented as a plain per-signal-pair xarray (``global_coherence``,
    ``phase_slope_index``, ``group_delay``, ``delay``, ``canonical_coherence``,
    and the conditional/blockwise spectral Granger measures). It subclasses
    ``ValueError`` for backward compatibility, but is a
    distinct type so :func:`multitaper_connectivity` can skip an unsupported
    measure in a multi-measure batch *without* also swallowing a genuine
    computation error (e.g. a measure that raises ``ValueError`` because the data
    has too few observations).
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

    output_kind: Literal["pairwise", "power", "unsupported"]
    is_default: bool = False
    # Directed measures use the convention output[i, j] = influence j -> i (row
    # = receiver, col = driver). The wrapper transposes them so the labeled
    # (source, target) axes read source -> target; symmetric measures are
    # unaffected by the transpose.
    is_directed: bool = False


_PAIRWISE_SPEC = _MeasureSpec("pairwise")
_DIRECTED_PAIRWISE_SPEC = _MeasureSpec("pairwise", is_directed=True)
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
        "pairwise", is_default=True, is_directed=True
    ),
    "phase_lag_index": _MeasureSpec("pairwise", is_default=True),
    "phase_locking_value": _MeasureSpec("pairwise", is_default=True),
    "power": _MeasureSpec("power", is_default=True),
    "weighted_phase_lag_index": _MeasureSpec("pairwise", is_default=True),
    "coherency": _PAIRWISE_SPEC,
    "subset_pairwise_spectral_granger_prediction": _DIRECTED_PAIRWISE_SPEC,
    # Directed-transfer-function family: opt-in (not in the default set),
    # directed (output[i, j] = influence j -> i, transposed to source -> target),
    # and returning the full (time, frequency, source, target) layout.
    "directed_transfer_function": _DIRECTED_PAIRWISE_SPEC,
    "directed_coherence": _DIRECTED_PAIRWISE_SPEC,
    "partial_directed_coherence": _DIRECTED_PAIRWISE_SPEC,
    "generalized_partial_directed_coherence": _DIRECTED_PAIRWISE_SPEC,
    "direct_directed_transfer_function": _DIRECTED_PAIRWISE_SPEC,
    **dict.fromkeys(
        (
            "blockwise_spectral_granger_prediction",
            "canonical_coherence",
            "conditional_spectral_granger_prediction",
            "delay",
            "global_coherence",
            "group_delay",
            "phase_slope_index",
        ),
        _UNSUPPORTED_SPEC,
    ),
}

DEFAULT_METHODS: tuple[str, ...] = tuple(
    name for name, spec in _MEASURE_SPECS.items() if spec.is_default
)


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
    multitaper_metadata: Mapping[str, Any],
    method: str,
    signal_names: Sequence[_SignalLabel] | None,
    squeeze: bool,
    *,
    input_attrs: Mapping[Any, Any] | None = None,
    **kwargs: Any,
) -> xr.DataArray:
    """Format one result from an already-built ``Connectivity`` instance."""
    measure_spec = _get_measure_spec(method)
    signal_labels = _validated_signal_labels(signal_names, connectivity.n_signals)
    connectivity_mat = getattr(connectivity, method)(**kwargs)

    pairwise_shape = (
        len(connectivity.time),
        len(connectivity.frequencies),
        connectivity.n_signals,
        connectivity.n_signals,
    )
    power_shape = pairwise_shape[:-1]
    actual_shape = tuple(connectivity_mat.shape)
    if measure_spec is None:
        if actual_shape != pairwise_shape:
            raise UnsupportedMeasureError(
                f"The method '{method}' returned shape {actual_shape}, but an "
                f"unregistered wrapper extension must return {pairwise_shape}. "
                "Register its output contract or use Connectivity directly."
            )
        measure_spec = _PAIRWISE_SPEC
    expected_shape = (
        power_shape if measure_spec.output_kind == "power" else pairwise_shape
    )
    if actual_shape != expected_shape:
        raise ValueError(
            f"The method '{method}' returned shape {actual_shape}; its wrapper "
            f"contract requires {expected_shape}."
        )

    if measure_spec.is_directed:
        # Directed measures return output[i, j] = influence j -> i (row =
        # receiver, col = driver). Transpose the trailing signal axes so the
        # stored value at [i, j] is i -> j, matching the (source, target) labels
        # applied below: sel(source=i, target=j) then reads "i drives j".
        connectivity_mat = np.swapaxes(connectivity_mat, -1, -2)

    attrs = _shared_provenance_attrs(
        connectivity, multitaper_metadata, input_attrs=input_attrs
    )
    attrs["measure"] = method
    attrs["measure_kwargs_json"] = _canonical_json(kwargs)
    for key, value in kwargs.items():
        _store_provenance_item(attrs, "arg_", key, value)

    coordinates: dict[str, Any] = {
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
        "source": (
            "source",
            signal_labels,
            {"long_name": "Source signal"},
        ),
    }
    if measure_spec.output_kind == "power":
        # squeeze has no meaning for power (no target axis); it is a no-op here.
        xar = xr.DataArray(
            connectivity_mat,
            coords=coordinates,
            dims=("time", "frequency", "source"),
            name=method,
            attrs=attrs,
        )
    else:
        coordinates["target"] = (
            "target",
            signal_labels,
            {"long_name": "Target signal"},
        )
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


def _shared_provenance_attrs(
    connectivity: Connectivity,
    multitaper_metadata: Mapping[str, Any],
    input_attrs: Mapping[Any, Any] | None = None,
) -> dict[str, Any]:
    """Provenance shared by every measure computed from one transform.

    Covers the package/version, the imported backend, the expectation type, and
    the multitaper parameters (``mt_*``) -- everything that does not depend on
    the specific measure. The per-measure attributes (``measure``,
    ``measure_kwargs_json``, and the convenient ``arg_*`` views) are added by
    the caller. Any attributes on an input ``xarray.DataArray`` are carried
    through under an ``input_*`` namespace so the caller's own metadata survives.
    """
    # Namespace transform settings so they cannot collide with measure-level or
    # package-level provenance attributes.
    attrs: dict[str, Any] = {
        "mt_" + attr: value for attr, value in multitaper_metadata.items()
    }
    attrs["package"] = "spectral_connectivity"
    attrs["package_version"] = _package_version()
    # get_compute_backend() reports the backend actually imported (numpy vs
    # cupy), not the current env var; is_gpu_enabled() would mislabel a result if
    # SPECTRAL_CONNECTIVITY_ENABLE_GPU changed after import.
    attrs["backend"] = get_compute_backend()["backend"].upper()
    attrs["expectation_type"] = connectivity.expectation_type
    # Preserve the caller's DataArray attributes. The ``input_`` prefix keeps
    # them in a distinct namespace from our own provenance keys; each value is
    # made NetCDF-safe (structured values land under ``input_<key>_json``).
    for key, value in (input_attrs or {}).items():
        _store_provenance_item(attrs, "input_", key, value)
    return attrs


def connectivity_to_xarray(
    m: Multitaper,
    method: str = "coherence_magnitude",
    signal_names: Sequence[_SignalLabel] | None = None,
    squeeze: bool = False,
    **kwargs: Any,
) -> xr.DataArray:
    """Calculate one connectivity measure and return a labeled array.

    Pairwise measures use ``(time, frequency, source, target)`` dimensions;
    power uses ``(time, frequency, source)``. Measures with different output
    contracts should be called on :class:`Connectivity` directly.

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
    connectivity = Connectivity.from_multitaper(m)
    return _connectivity_result_to_xarray(
        connectivity, metadata, method, signal_names, squeeze, **kwargs
    )


# Common dimension names let the wrapper infer semantic roles. DataArrays are
# transposed into the numerical core's (time[, trial], signal) order; callers
# provide explicit ``*_dim`` arguments when their names are domain-specific.
_SAMPLE_DIM_NAMES = frozenset({"sample", "samples"})
_TIME_DIM_NAMES = frozenset(
    {"time", "times", "timestamp", "timestamps", *_SAMPLE_DIM_NAMES}
)
_TRIAL_DIM_NAMES = frozenset({"trial", "trials", "epoch", "epochs"})
_SIGNAL_DIM_NAMES = frozenset(
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
)


def _dimension_role(dimension: Hashable) -> str | None:
    """Return the recognized semantic role of an xarray dimension name."""
    name = str(dimension).lower()
    if name in _TIME_DIM_NAMES:
        return "time"
    if name in _TRIAL_DIM_NAMES:
        return "trial"
    if name in _SIGNAL_DIM_NAMES:
        return "signal"
    return None


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
        resolved[unresolved_roles[0]] = unused_dimensions[0]
        unresolved_roles.clear()
    if unresolved_roles:
        arguments = ", ".join(f"{role}_dim" for role in unresolved_roles)
        raise ValueError(
            "Could not infer the semantic roles of DataArray dimensions "
            f"{time_series.dims!r}. Pass {arguments} explicitly; dimension "
            "positions are not used for labeled input."
        )

    return tuple(resolved[role] for role in expected_roles)


def _start_time_from_dataarray(
    time_series: xr.DataArray,
    time_dimension: Hashable,
    sampling_frequency: float,
    explicit_start_time: Any = _UNSET,
) -> float | None:
    """Validate a numeric time index and return its first sample in seconds."""
    candidates = [
        (name, coordinate)
        for name, coordinate in time_series.coords.items()
        if coordinate.dims == (time_dimension,)
        and (name == time_dimension or _dimension_role(name) == "time")
    ]
    if not candidates:
        return None
    exact_time = [item for item in candidates if str(item[0]).lower() == "time"]
    semantic_auxiliary = [
        item
        for item in candidates
        if item[0] != time_dimension and _dimension_role(item[0]) == "time"
    ]
    if len(exact_time) == 1:
        coordinate_name, coordinate = exact_time[0]
    elif len(semantic_auxiliary) == 1:
        coordinate_name, coordinate = semantic_auxiliary[0]
    elif len(candidates) == 1:
        coordinate_name, coordinate = candidates[0]
    else:
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
            "'sample' coordinate. Datetime, timedelta, and object time "
            "coordinates are not yet supported."
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
        expected_description = (
            "1 sample per coordinate step"
            if coordinate_is_sample_index
            else f"{expected_interval!r} seconds per sample"
        )
        raise ValueError(
            f"The DataArray time coordinate spacing does not match "
            f"sampling_frequency={sampling_frequency!r} Hz (expected "
            f"{expected_description}, observed median "
            f"{float(np.median(differences))!r})."
        )

    inferred_start_time = float(times[0]) / (
        float(sampling_frequency) if coordinate_is_sample_index else 1.0
    )
    start_time_tolerance = coordinate_tolerance / (
        float(sampling_frequency) if coordinate_is_sample_index else 1.0
    )
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
    return inferred_start_time


def _signal_labels_from_dataarray(
    time_series: xr.DataArray, signal_dimension: Hashable
) -> Sequence[_SignalLabel] | None:
    """Signal labels from a 1-D index coordinate on the signal dimension.

    Returns ``None`` (default integer labels used downstream) when the signal
    dimension has no usable 1-D index coordinate. If the DataArray *does* carry
    coordinates along that dimension but none is a usable 1-D index coordinate,
    warn rather than silently dropping the user's labels.
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
            f"signal labels could not be inferred; default integer labels will "
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
    sampling_frequency: float,
    *,
    time_dim: Hashable | None,
    trial_dim: Hashable | None,
    signal_dim: Hashable | None,
    explicit_start_time: Any = _UNSET,
) -> tuple[BackendArray, Sequence[_SignalLabel] | None, float | None]:
    """Extract array data and, when available, labels from a DataArray input.

    Semantic dimensions are inferred from common names or supplied explicitly,
    then transposed into the numerical core's positional order. A numeric time
    index is validated against the sampling rate and supplies ``start_time``.
    """
    if not isinstance(time_series, xr.DataArray):
        if any(
            dimension is not None for dimension in (time_dim, trial_dim, signal_dim)
        ):
            raise TypeError(
                "time_dim, trial_dim, and signal_dim apply only to an "
                "xarray.DataArray input."
            )
        return time_series, signal_names, None

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
    inferred_start_time = _start_time_from_dataarray(
        time_series,
        time_dimension,
        sampling_frequency,
        explicit_start_time,
    )

    data = time_series.transpose(*dimension_order).data
    _reject_unmaterialized_backing(data)
    return data, signal_names, inferred_start_time


def multitaper_connectivity(
    time_series: NDArray[np.floating] | xr.DataArray,
    sampling_frequency: float,
    time_window_duration: float | None = None,
    method: str | list[str] | None = None,
    signal_names: Sequence[_SignalLabel] | None = None,
    squeeze: bool = False,
    connectivity_kwargs: dict[str, Any] | None = None,
    *,
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
        than falling back to dimension position. A numeric time index is
        interpreted as elapsed seconds (a ``sample`` index as sample numbers),
        validated against ``sampling_frequency``, and used to label output window
        centers. Datetime, timedelta, and object-valued time coordinates are not
        yet supported and must first be converted to numeric elapsed seconds.
        When ``signal_names`` is omitted,
        labels from a 1-D index coordinate on the signal dimension are carried to
        the output's ``source`` and ``target`` coordinates without changing their
        type; if such labels are present but unusable a warning is issued and
        default string labels are used.
    sampling_frequency : float
        Sampling rate in Hz of the time series data.
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
        Notes on directed orientation). Other measures that do not fit the
        ``(time, frequency, source, target)`` layout — ``global_coherence``,
        ``phase_slope_index``, ``group_delay``, ``delay``, ``canonical_coherence``,
        and the conditional/blockwise spectral Granger measures — are *not*
        available through this wrapper at all (requesting one raises with a
        pointer to use ``Connectivity`` directly). Examples:
        "coherence_magnitude", "imaginary_coherence", "phase_locking_value".
    signal_names : sequence of scalar, optional
        Scalar, non-missing, unique xarray-compatible coordinate labels for signal
        channels. Nested or structured labels are not supported. If None, uses the
        DataArray signal index when available, otherwise stringified indices.
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
        - DataArray if single method requested: connectivity values with dimensions
          ['time', 'frequency', 'source', 'target'] or ['time', 'frequency'] if squeezed
        - Dataset if multiple methods: collection of DataArrays, one per method

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
    - ``input_<key>`` / ``input_<key>_json`` -- attributes carried over from an
      input ``xarray.DataArray`` (e.g. subject or session metadata), preserved
      under the ``input_`` namespace so the caller's own metadata survives.

    References
    ----------
    .. [1] Thomson, D. J. (1982). Spectrum estimation and harmonic analysis.
           Proceedings of the IEEE, 70(9), 1055-1096.
    .. [2] Percival, D. B., & Walden, A. T. (1993). Spectral Analysis for Physical
           Applications: Multitaper and Conventional Univariate Techniques.
    """
    input_attrs = (
        dict(time_series.attrs) if isinstance(time_series, xr.DataArray) else None
    )
    explicit_start_time = kwargs.get("start_time", _UNSET)
    time_series_data, signal_names, inferred_start_time = _unwrap_xarray_input(
        time_series,
        signal_names,
        sampling_frequency,
        time_dim=time_dim,
        trial_dim=trial_dim,
        signal_dim=signal_dim,
        explicit_start_time=explicit_start_time,
    )
    if inferred_start_time is not None and explicit_start_time is _UNSET:
        kwargs["start_time"] = inferred_start_time
    if connectivity_kwargs is None:
        connectivity_kwargs = {}
    return_dataarray = False  # Default: return dataset
    if method is None:
        # The explicit, portably serializable / xarray-compatible default set
        # (see DEFAULT_METHODS). Not every Connectivity method — coherency
        # (complex), global_coherence / phase_slope_index, and the directed-
        # transfer-function family are excluded from the default (the last is
        # still opt-in by name).
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
    if squeeze and not return_dataarray:
        # squeeze reduces a pairwise measure to a (time, frequency) array whose
        # source/target become scalar coordinates. In a Dataset those scalars are
        # shared across all variables and would collide with a sibling variable's
        # axes (e.g. power's ``source`` dimension), so squeeze is honored only
        # when a single method is requested and the result is a DataArray.
        warnings.warn(
            "squeeze=True is ignored for multi-measure results (a Dataset); "
            "request a single method (a string) to get a squeezed DataArray.",
            UserWarning,
            stacklevel=2,
        )
        squeeze = False
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
    if return_dataarray:
        return _connectivity_result_to_xarray(
            shared_connectivity,
            metadata,
            method[0],
            signal_names,
            squeeze,
            input_attrs=input_attrs,
            **connectivity_kwargs,
        )

    data_vars: dict[str, xr.DataArray] = {}
    for this_method in method:
        try:
            data_vars[this_method] = _connectivity_result_to_xarray(
                shared_connectivity,
                metadata,
                this_method,
                signal_names,
                squeeze,
                input_attrs=input_attrs,
                **connectivity_kwargs,
            )
        except (NotImplementedError, UnsupportedMeasureError) as e:
            # Structural incompatibility can be skipped in a batch. Other
            # computation errors are intentionally not caught.
            if len(method) == 1:
                raise
            logger.warning("Skipping %s: %s", this_method, e)
    if not data_vars:
        raise UnsupportedMeasureError(
            "None of the requested methods produced a compatible result "
            f"for the xarray interface: {method!r}."
        )
    # Shared coordinates are aligned once during construction. Dataset-level
    # provenance makes a multi-measure result self-describing without requiring
    # callers to inspect an individual variable.
    return xr.Dataset(
        data_vars=data_vars,
        attrs=_shared_provenance_attrs(
            shared_connectivity, metadata, input_attrs=input_attrs
        ),
    )
