"""Validate the wrapper's inputs and unwrap labeled (xarray) ones.

Resolves the semantic roles of DataArray dimensions, infers the sampling rate
and start time from a time coordinate, and extracts signal labels and
metadata, so the numerical core always receives positional arrays.
"""

import warnings
from collections.abc import Hashable, Mapping, Sequence
from typing import Any, NamedTuple, TypeAlias

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from spectral_connectivity.utils import BackendArray, to_numpy

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


class _SignalMetadata(NamedTuple):
    """What an input DataArray says about its signals, carried into results."""

    # 1-D non-index coordinates on the signal dimension (e.g. brain region);
    # each becomes ``<axis>_<name>`` on the result's source/target/signal axis.
    coordinates: Mapping[str, NDArray[Any]]
    # The input's ``units`` attribute; spectral densities report (units)^2/Hz.
    units: str | None


class _UnwrappedInput(NamedTuple):
    """Array data plus what a DataArray contributed, named to avoid swaps."""

    data: BackendArray
    signal_names: Sequence[_SignalLabel] | None
    inferred_sampling_frequency: float | None
    inferred_start_time: float | None
    input_attrs: Mapping[Any, Any] | None
    signal_metadata: _SignalMetadata | None


def _validated_signal_labels(
    signal_names: Sequence[_SignalLabel] | None,
    n_signals: int,
) -> NDArray[Any]:
    """Return a unique, portable, one-dimensional xarray signal coordinate."""
    if signal_names is None:
        names: list[_SignalLabel] = [str(index) for index in range(n_signals)]
    else:
        names = list(signal_names)
    if len(names) != n_signals:
        msg = f"signal_names must contain {n_signals} names, got {len(names)}."
        raise ValueError(msg)
    try:
        signal_coordinate = xr.IndexVariable("signal", names)
        signal_index = signal_coordinate.to_index()
    except (TypeError, ValueError) as error:
        msg = (
            "signal_names must form a one-dimensional xarray coordinate of "
            "scalar labels; nested or structured labels are not supported."
        )
        raise ValueError(msg) from error
    if signal_coordinate.dtype.kind not in "biufSUMm":
        msg = (
            "signal_names must contain NetCDF-compatible string, real numeric, "
            "datetime, or timedelta scalar labels; object and complex labels "
            "are not supported."
        )
        raise ValueError(msg)
    if signal_coordinate.dtype.kind in "iu" and signal_coordinate.size:
        # SciPy is a required dependency and therefore xarray's only guaranteed
        # NetCDF writer in a minimum installation. Its NetCDF3 backend cannot
        # represent integer coordinate values outside the signed 32-bit range.
        integer_values = np.asarray(signal_coordinate.data)
        minimum = int(integer_values.min())
        maximum = int(integer_values.max())
        int32 = np.iinfo(np.int32)
        if minimum < int32.min or maximum > int32.max:
            msg = (
                "Integer signal_names must fit the signed 32-bit range for "
                "portable NetCDF3 serialization; got range "
                f"[{minimum}, {maximum}]. Use string labels for larger identifiers."
            )
            raise ValueError(msg)
    if bool(getattr(signal_index, "hasnans", False)):
        msg = "signal_names must not contain missing labels (NaN, NaT, or None)."
        raise ValueError(msg)
    if not signal_index.is_unique:
        duplicates = sorted(
            signal_index[signal_index.duplicated(keep=False)].unique().tolist(),
            key=repr,
        )
        msg = (
            "signal_names must be unique to label the source/target axes; "
            f"duplicates: {duplicates}."
        )
        raise ValueError(msg)
    return np.asarray(signal_coordinate.data)


def _is_real_numeric_dtype(dtype: np.dtype[Any]) -> bool:
    """Whether ``dtype`` holds real numbers (not complex, boolean, or time types)."""
    return bool(
        np.issubdtype(dtype, np.number)
        and not np.issubdtype(dtype, np.complexfloating)
        and not np.issubdtype(dtype, np.bool_)
        and not np.issubdtype(dtype, np.datetime64)
        and not np.issubdtype(dtype, np.timedelta64)
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
# Axes a spectral result has but a time-domain input never does. One left over
# in a DataArray marks an already-transformed input (a spectrogram or band
# powers), so it must not be promoted to a time-series role by elimination.
_FREQUENCY_DIM_NAMES = frozenset({"frequency", "frequencies", "freq", "freqs"})
_SPECTRAL_DIM_NAMES = _FREQUENCY_DIM_NAMES | frozenset({"band", "bands"})

# Inference is a convenience, so prefer requiring an explicit rate over silently
# deriving a scientifically meaningful frequency scale from a precision-starved
# coordinate. This bounds the endpoint-quantization contribution to the inferred
# rate at 100 parts per million of the observed time span.
_MAX_INFERRED_RATE_RELATIVE_RESOLUTION = 1e-4

# A time coordinate counts as uniformly spaced when every step is within
# _TIME_STEP_TOLERANCE sampling intervals of the expected interval and every
# sample lies within that plus _TIME_COORDINATE_RELATIVE_TOLERANCE of its
# elapsed time from the regular grid. Measured on float64 axes built with
# ``np.cumsum``, each step is exact to 2e-8 intervals even after 1e8 samples,
# while the accumulated deviation from the grid grows roughly quadratically with
# length (1e-6 intervals at 30 kHz x 10 s, 2e-4 at 1 kHz x 1 h, 0.2 at
# 30 kHz x 1 h) yet stays below 2e-9 per elapsed interval. A dropped sample is a
# step off by a whole interval (a duplicated timestamp, a zero step, already
# fails the strictly-increasing check), and timestamp jitter of 0.05 intervals
# moves a step by at most 0.1, so a quarter interval separates them. The drift allowance admits accumulated round-off at any
# practical length yet rejects an explicit rate that is off by more than a part
# per million over a long axis (a unit mismatch fails every step).
_TIME_STEP_TOLERANCE = 0.25
_TIME_COORDINATE_RELATIVE_TOLERANCE = 1e-6


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
        msg = (
            "A DataArray input must have dimensions (time, signal) or "
            "(time, trial, signal); "
            f"got {time_series.ndim} dimensions {time_series.dims!r}."
        )
        raise ValueError(msg)

    requested = {
        "time": time_dim,
        "trial": trial_dim,
        "signal": signal_dim,
    }
    if trial_dim is not None and "trial" not in expected_roles:
        msg = (
            "trial_dim cannot be used with a 2-D DataArray; a 2-D input has no "
            "trial axis. Use dimensions (time, signal), or provide a 3-D array."
        )
        raise ValueError(msg)

    resolved: dict[str, Hashable] = {}
    used_dimensions: dict[Hashable, str] = {}
    for role in expected_roles:
        dimension = requested[role]
        if dimension is None:
            continue
        if dimension not in time_series.dims:
            msg = (
                f"{role}_dim={dimension!r} is not an input dimension; "
                f"available dimensions are {time_series.dims!r}."
            )
            raise ValueError(msg)
        previous_role = used_dimensions.get(dimension)
        if previous_role is not None:
            msg = (
                f"Dimension {dimension!r} was assigned to both {previous_role}_dim "
                f"and {role}_dim; each semantic role needs a distinct dimension."
            )
            raise ValueError(msg)
        inferred_role = _dimension_role(dimension)
        if inferred_role is not None and inferred_role != role:
            msg = (
                f"{role}_dim={dimension!r} conflicts with its recognized "
                f"{inferred_role} meaning. Rename the dimension or pass the "
                "correct mapping."
            )
            raise ValueError(msg)
        resolved[role] = dimension
        used_dimensions[dimension] = role

    for dimension in time_series.dims:
        if dimension in used_dimensions:
            continue
        if str(dimension).lower() in _SPECTRAL_DIM_NAMES:
            msg = (
                f"Dimension {dimension!r} denotes a spectral (frequency or band) "
                "axis, but multitaper_connectivity expects time-domain signals "
                "with dimensions (time, signal) or (time, trial, signal). Drop or "
                "reshape that dimension, or pass already-computed Fourier "
                "coefficients to fourier_connectivity. If it really indexes "
                f"signals, pass signal_dim={dimension!r}."
            )
            raise ValueError(msg)
        inferred_role = _dimension_role(dimension)
        if inferred_role is None:
            continue
        if inferred_role not in expected_roles:
            msg = (
                f"Dimension {dimension!r} denotes a {inferred_role} axis, but a "
                f"{time_series.ndim}-D input has no {inferred_role} axis. Drop "
                "or reshape that dimension."
            )
            raise ValueError(msg)
        if inferred_role in resolved:
            msg = (
                f"Dimensions {resolved[inferred_role]!r} and {dimension!r} both "
                f"denote the {inferred_role} axis. Rename them or pass an "
                "unambiguous dimension mapping."
            )
            raise ValueError(msg)
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
        msg = (
            "Could not infer the semantic roles of DataArray dimensions "
            f"{time_series.dims!r}. Pass {arguments} explicitly; dimension "
            "positions are not used for labeled input."
        )
        raise ValueError(msg)

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
        msg = (
            "sampling_frequency must be a positive, finite number for a "
            f"DataArray with a numeric time coordinate; got {sampling_frequency!r}."
        )
        raise ValueError(msg)
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
        msg = (
            f"Multiple coordinates {coordinate_names!r} could label time "
            f"dimension {time_dimension!r}. Keep one time-like coordinate or "
            "rename the others so the intended elapsed-seconds coordinate is "
            "unambiguous."
        )
        raise ValueError(msg)

    values = np.asarray(coordinate.to_numpy())
    if not _is_real_numeric_dtype(values.dtype):
        msg = (
            f"The DataArray time coordinate {coordinate_name!r} must contain "
            "numeric elapsed seconds, or integer-like sample numbers for a "
            "'sample' coordinate. Datetime, timedelta, complex, boolean, and "
            f"object time coordinates are not yet supported (got dtype "
            f"{values.dtype!r}). Convert a datetime axis to elapsed seconds, "
            "e.g. (da.time - da.time[0]) / np.timedelta64(1, 's')."
        )
        raise TypeError(msg)

    times = values.astype(np.float64, copy=False)
    if times.size == 0:
        msg = f"The DataArray time coordinate {coordinate_name!r} must not be empty."
        raise ValueError(msg)
    if not np.all(np.isfinite(times)):
        msg = (
            f"The DataArray time coordinate {coordinate_name!r} must contain "
            "only finite values."
        )
        raise ValueError(msg)
    differences = np.diff(times)
    if np.any(differences <= 0):
        msg = f"The DataArray time coordinate {coordinate_name!r} must be strictly increasing."
        raise ValueError(msg)

    coordinate_is_sample_index = str(coordinate_name).lower() in _SAMPLE_DIM_NAMES
    if coordinate_is_sample_index and not np.all(times == np.rint(times)):
        msg = (
            f"The DataArray sample coordinate {coordinate_name!r} must contain "
            "integer-like sample numbers. Use a 'time' coordinate for elapsed "
            "fractional seconds."
        )
        raise ValueError(msg)
    inferred_sampling_frequency: float | None = None
    if sampling_frequency is None:
        # Infer the rate from an elapsed-seconds coordinate. Integer sample
        # numbers have no time scale, so they cannot supply one.
        if coordinate_is_sample_index:
            msg = (
                f"Cannot infer sampling_frequency from the integer sample "
                f"coordinate {coordinate_name!r}, which has no time scale. Pass "
                "sampling_frequency, or use a numeric 'time' coordinate in "
                "elapsed seconds."
            )
            raise ValueError(msg)
        if times.size < 2:
            msg = (
                "Cannot infer sampling_frequency from a single-sample time "
                f"coordinate {coordinate_name!r}; pass sampling_frequency."
            )
            raise ValueError(msg)
        # Span-based estimate averages float noise over the whole (uniform) grid.
        coordinate_span = float(times[-1]) - float(times[0])
        if not np.isfinite(coordinate_span) or coordinate_span <= 0:
            msg = (
                "Cannot infer sampling_frequency: the DataArray time coordinate "
                f"{coordinate_name!r} does not have a finite positive span. Pass "
                "sampling_frequency explicitly."
            )
            raise ValueError(msg)
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
                msg = (
                    "Cannot reliably infer sampling_frequency from DataArray time "
                    f"coordinate {coordinate_name!r}: its {values.dtype} resolution "
                    f"({storage_resolution!r} s) is too large relative to the "
                    f"observed span ({coordinate_span!r} s). Pass "
                    "sampling_frequency explicitly, or use a higher-precision or "
                    "zero-based elapsed-seconds coordinate."
                )
                raise ValueError(msg)
        expected_interval = coordinate_span / (times.size - 1)
        if (
            not np.isfinite(expected_interval)
            or expected_interval <= 0
            or expected_interval < 1.0 / np.finfo(np.float64).max
        ):
            msg = (
                "Cannot infer sampling_frequency: the DataArray time coordinate "
                f"{coordinate_name!r} implies a non-finite sampling rate. Pass "
                "sampling_frequency explicitly."
            )
            raise ValueError(msg)
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
        expected_interval * _TIME_COORDINATE_RELATIVE_TOLERANCE,
        coordinate_resolution,
        np.spacing(coordinate_scale) * 8,
    )
    # Judge each step separately so accumulated round-off (which grows with the
    # axis length) cannot mask or mimic a dropped or duplicated sample, and the
    # whole axis against the regular grid so a rate that is slightly off, or
    # changes partway, cannot pass as uniform one step at a time.
    step_tolerance = max(expected_interval * _TIME_STEP_TOLERANCE, coordinate_tolerance)
    elapsed = np.arange(times.size) * expected_interval
    grid_tolerance = step_tolerance + elapsed * _TIME_COORDINATE_RELATIVE_TOLERANCE
    if np.any(np.abs(differences - expected_interval) > step_tolerance) or np.any(
        np.abs(times - (times[0] + elapsed)) > grid_tolerance
    ):
        observed_median = float(np.median(differences))
        if sampling_frequency is None:
            # Inference requires a regular grid; an irregular one has no single
            # rate to derive.
            msg = (
                f"Cannot infer sampling_frequency: the DataArray time coordinate "
                f"{coordinate_name!r} is not uniformly spaced (observed median "
                f"step {observed_median!r} s). Pass sampling_frequency "
                "explicitly, or provide a regularly sampled time coordinate."
            )
            raise ValueError(msg)
        expected_description = (
            "1 sample per coordinate step"
            if coordinate_is_sample_index
            else f"{expected_interval!r} seconds per sample"
        )
        msg = (
            f"The DataArray time coordinate spacing does not match "
            f"sampling_frequency={sampling_frequency!r} Hz (expected "
            f"{expected_description}, observed median {observed_median!r})."
        )
        raise ValueError(msg)

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
            msg = (
                "A DataArray with one time coordinate requires scalar start_time; "
                f"got shape {explicit.shape}."
            )
            raise ValueError(msg)
        explicit_value = float(explicit.reshape(-1)[0])
        if not np.isclose(
            explicit_value,
            inferred_start_time,
            rtol=0,
            atol=start_time_tolerance,
        ):
            msg = (
                f"start_time={explicit_value!r} conflicts with the first "
                f"DataArray time coordinate {inferred_start_time!r}. Remove "
                "start_time or make the values agree."
            )
            raise ValueError(msg)
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
        signal_dimension in coordinate.dims for coordinate in time_series.coords.values()
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


def _signal_coordinates_from_dataarray(
    data_array: xr.DataArray, signal_dimension: Hashable
) -> dict[str, NDArray[Any]]:
    """1-D non-index coordinates along the signal dimension (e.g. brain region)."""
    return {
        str(name): np.asarray(coordinate.to_numpy())
        for name, coordinate in data_array.coords.items()
        if name != signal_dimension and coordinate.dims == (signal_dimension,)
    }


def _reject_unmaterialized_backing(data: Any) -> None:
    """Reject a lazy backing array the positional spectral math cannot consume.

    xarray materializes a masked array to a NaN-filled ndarray on construction,
    so a mask surfaces loudly as NaN downstream and needs no guard here. A dask
    array, by contrast, is handed through ``.data`` unmaterialized.
    """
    if callable(getattr(data, "__dask_graph__", None)):
        msg = (
            "multitaper_connectivity received a dask-backed DataArray, which is "
            "not supported. Materialize it first with DataArray.compute() (or "
            "DataArray.load()) and pass the result."
        )
        raise TypeError(msg)


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
        if any(dimension is not None for dimension in (time_dim, trial_dim, signal_dim)):
            msg = (
                "time_dim, trial_dim, and signal_dim apply only to an xarray.DataArray input."
            )
            raise TypeError(msg)
        return _UnwrappedInput(time_series, signal_names, None, None, None, None)

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
    units = time_series.attrs.get("units")
    return _UnwrappedInput(
        data,
        signal_names,
        inferred_sampling_frequency,
        inferred_start_time,
        dict(time_series.attrs),
        _SignalMetadata(
            _signal_coordinates_from_dataarray(time_series, signal_dimension),
            units if isinstance(units, str) and units else None,
        ),
    )


_FOURIER_ROLE_SYNONYMS: dict[str, frozenset[str]] = {
    "time": frozenset({"time", "times", "window", "windows", "time_window", "time_windows"}),
    "trial": _ROLE_SYNONYMS["trial"] | frozenset({"observation", "observations"}),
    "taper": frozenset({"taper", "tapers"}),
    "frequency": _FREQUENCY_DIM_NAMES,
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
    _SignalMetadata | None,
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
            msg = "The *_dim arguments apply only to an xarray.DataArray input."
            raise TypeError(msg)
        data = fourier_coefficients
        ndim = getattr(data, "ndim", None)
        if ndim == 3:
            # (observation, frequency, signal)
            data = data[np.newaxis, :, np.newaxis, :, :]
        elif ndim == 4:
            # (trial, taper, frequency, signal)
            data = data[np.newaxis, :, :, :, :]
        elif ndim != 5:
            msg = (
                "fourier_coefficients must have 3, 4, or 5 dimensions: "
                "(observation, frequency, signal), (trial, taper, frequency, "
                "signal), or (time, trial, taper, frequency, signal)."
            )
            raise ValueError(msg)
        return data, frequencies, time, signal_names, None, None

    coefficient_array = fourier_coefficients
    if coefficient_array.ndim < 3 or coefficient_array.ndim > 5:
        msg = "A Fourier coefficient DataArray must have 3 to 5 dimensions."
        raise ValueError(msg)
    _reject_unmaterialized_backing(coefficient_array.data)

    role_to_dimension: dict[str, Hashable] = {}
    claimed_dimensions: set[Hashable] = set()
    for role, dimension in dimension_arguments.items():
        if dimension is None:
            continue
        if dimension not in coefficient_array.dims:
            msg = (
                f"{role}_dim={dimension!r} is not one of the DataArray "
                f"dimensions {coefficient_array.dims!r}."
            )
            raise ValueError(msg)
        if dimension in claimed_dimensions:
            msg = f"DataArray dimension {dimension!r} was assigned to more than one role."
            raise ValueError(msg)
        role_to_dimension[role] = dimension
        claimed_dimensions.add(dimension)

    for role, synonyms in _FOURIER_ROLE_SYNONYMS.items():
        if role in role_to_dimension:
            continue
        candidates = [
            dimension
            for dimension in coefficient_array.dims
            if dimension not in claimed_dimensions and str(dimension).lower() in synonyms
        ]
        if len(candidates) > 1:
            msg = (
                f"Multiple dimensions look like the Fourier {role} axis: "
                f"{candidates!r}. Pass {role}_dim explicitly."
            )
            raise ValueError(msg)
        if candidates:
            role_to_dimension[role] = candidates[0]
            claimed_dimensions.add(candidates[0])

    for required_role in ("frequency", "signal"):
        if required_role not in role_to_dimension:
            msg = (
                f"Could not identify the Fourier {required_role} dimension. "
                f"Use {required_role}_dim=... explicitly."
            )
            raise ValueError(msg)

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
        msg = (
            f"Could not infer the roles of Fourier dimensions {unclaimed!r}. "
            "Name them time/trial/taper, or pass the corresponding *_dim arguments."
        )
        raise ValueError(msg)

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
    if has_frequency_coordinate and not frequency_coordinate_is_1d and frequencies is None:
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
        msg = "frequencies conflicts with the DataArray frequency coordinate."
        raise ValueError(msg)

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
        elif coordinate_time is not None and not _coordinates_agree(time, coordinate_time):
            msg = "time conflicts with the DataArray time coordinate."
            raise ValueError(msg)

    if signal_names is None:
        signal_names = _signal_labels_from_dataarray(
            coefficient_array, role_to_dimension["signal"]
        )
    # Coefficient units are not time-series units, so no density units follow.
    signal_metadata = _SignalMetadata(
        _signal_coordinates_from_dataarray(coefficient_array, role_to_dimension["signal"]),
        None,
    )
    return (
        data,
        frequencies,
        time,
        signal_names,
        dict(coefficient_array.attrs),
        signal_metadata,
    )
