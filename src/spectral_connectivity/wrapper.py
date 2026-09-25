"""Functions for getting connectivity measures in a labeled array format."""

import inspect
import warnings
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from logging import getLogger
from typing import Any, Literal

import numpy as np
import xarray as xr
from numpy.typing import DTypeLike, NDArray

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
    _CATEGORY_DIMS,
    _MEASURE_SPECS,
    _get_measure_spec,
    _measure_description,
    _measure_label_attrs,
    _validate_method_names,
)
from spectral_connectivity._provenance import (
    _canonical_json,
    _shared_provenance_attrs,
    _store_provenance_item,
)
from spectral_connectivity.connectivity import (
    Connectivity,
    MultivariateConnectivityResult,
    _frequencies_in_band,
)
from spectral_connectivity.transforms import Multitaper
from spectral_connectivity.utils import is_positive_integer, to_numpy

logger = getLogger(__name__)


class UnsupportedMeasureError(ValueError):
    """A method has no registered semantic xarray output contract.

    Built-in nonstandard results (components, groups, delays, and multi-variable
    outputs) have explicit schemas. This exception remains for unregistered
    extensions whose returned shape cannot be inferred safely. It subclasses
    ``ValueError`` for backward compatibility and lets multi-measure wrappers
    distinguish structural incompatibility from genuine numerical errors.
    """


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
    long_name : str
        Human-readable name, also the ``long_name`` attribute of the result.
    units : str
        Units of the values: ``"1"`` for a dimensionless score, ``"rad"``,
        ``"s"``, or ``"(input units)^2/Hz"`` for a spectral density.
    value_range : tuple of float
        ``(lower, upper)`` bounds of the values, or of their magnitude when
        ``is_complex``; ``inf`` marks an unbounded side.
    is_complex : bool
        Whether the values are complex.
    dims : tuple of str
        Dimensions of the measure's variable in the wrapper's result, before
        band reduction or squeezing. Rich results (``canonical_coherency``,
        ``global_coherence``, ...) are Datasets whose variable named ``name``
        has these dimensions.
    array_orientation : {"target_source", "source_target"} or None
        Index order of a directed measure's signal axes in the arrays returned
        by the lower-level ``Connectivity`` method. ``"target_source"`` means
        ``result[..., i, j]`` is the influence ``j -> i``; ``"source_target"``
        means it describes ``i`` relative to ``j`` (e.g. ``i`` leads ``j``).
        ``None`` for non-directed measures. The wrapper's results are always
        labeled: ``result.sel(source=a, target=b)`` is ``a -> b``.
    interpretation : str
        How to read the values, including the sign convention where one exists.
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
    array_orientation: Literal["target_source", "source_target"] | None
    interpretation: str


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
        orientation: Literal["target_source", "source_target"] | None = None
        if spec.is_directed:
            orientation = "target_source" if spec.transpose_output else "source_target"
        measures.append(
            MeasureInfo(
                name=name,
                category=spec.output_kind,
                description=_measure_description(name),
                is_default=spec.is_default,
                is_directed=spec.is_directed,
                requires_two_sided=spec.requires_two_sided,
                long_name=spec.long_name,
                units=spec.units or "(input units)^2/Hz",
                value_range=spec.value_range,
                is_complex=spec.is_complex,
                dims=_CATEGORY_DIMS[spec.output_kind],
                array_orientation=orientation,
                interpretation=spec.interpretation,
            )
        )
    return measures


def _check_method_accepts_kwargs(
    method: str, measure: Callable[..., Any], kwargs: Mapping[str, Any]
) -> None:
    """Raise an actionable error when ``kwargs`` names a parameter ``measure``
    does not accept.

    ``connectivity_kwargs`` is broadcast to every requested method, so a
    keyword needed by one measure (e.g. ``group_labels``) reaches the others.
    """
    parameters = inspect.signature(measure).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return
    rejected = sorted(set(kwargs) - set(parameters))
    if rejected:
        msg = (
            f"{method} does not accept keyword argument(s) "
            f"{', '.join(map(repr, rejected))}. connectivity_kwargs is passed to "
            "every requested method, so request measures that need different "
            "arguments in separate calls."
        )
        raise TypeError(msg)


def _frequency_band_attrs(
    connectivity: Connectivity, kwargs: Mapping[str, Any]
) -> dict[str, float]:
    """The band a frequency-reducing measure summarized, as variable attrs.

    Stored on the measure's own variables rather than as scalar coordinates,
    which a Dataset would broadcast onto every other variable.
    """
    band = kwargs.get("frequencies_of_interest")
    if band is None:
        band = (connectivity.frequencies[0], connectivity.frequencies[-1])
    return {"frequency_band_lower": float(band[0]), "frequency_band_upper": float(band[1])}


def _coordinate_attrs(
    shared_attrs: Mapping[str, Any],
) -> tuple[dict[str, str], dict[str, str]]:
    """``(time_attrs, frequency_attrs)`` metadata for a result's coordinates.

    ``fourier_connectivity`` records when it filled in a coordinate: the default
    frequency grid is normalized (cycles/sample) and the default time is the
    window index, so those must not be labeled Hz and seconds.
    """
    if shared_attrs.get("fourier_frequency_coordinate") == "normalized":
        frequency_attrs = {"long_name": "Normalized frequency", "units": "cycles/sample"}
    else:
        frequency_attrs = {"long_name": "Frequency", "units": "Hz"}
    if shared_attrs.get("fourier_time_coordinate") == "index":
        time_attrs = {"long_name": "Window index"}
    else:
        time_attrs = {"long_name": "Window center time", "units": "s"}
    return time_attrs, frequency_attrs


def _connectivity_result_to_xarray(
    connectivity: Connectivity,
    method: str,
    signal_labels: NDArray[Any],
    squeeze: bool,
    shared_attrs: Mapping[str, Any],
    *,
    signal_metadata: _SignalMetadata | None = None,
    **kwargs: Any,
) -> xr.DataArray | xr.Dataset:
    """Format one result from an already-built ``Connectivity`` instance.

    ``signal_labels`` and ``shared_attrs`` are invariant across the measures of
    one transform, so the caller validates/builds them once and passes them in.
    """
    measure_spec = _get_measure_spec(method)
    measure = getattr(connectivity, method)
    _check_method_accepts_kwargs(method, measure, kwargs)
    numerical_result = measure(**kwargs)

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
            msg = (
                f"The method '{method}' returned shape {actual_shape}, but an "
                f"unregistered wrapper extension must return {pairwise_shape}. "
                "Register its output contract or use Connectivity directly."
            )
            raise UnsupportedMeasureError(msg)
        # A proven-pairwise extension keeps its native, untransposed orientation.
        output_kind, transpose_output = "pairwise", False
    else:
        output_kind = measure_spec.output_kind
        transpose_output = measure_spec.transpose_output

    # Copy the shared provenance so per-measure keys never leak across measures.
    attrs = dict(shared_attrs)
    attrs["measure"] = method
    attrs["measure_kwargs_json"] = _canonical_json(kwargs)
    for key, value in kwargs.items():
        _store_provenance_item(attrs, "arg_", key, value)

    time_attrs, frequency_attrs = _coordinate_attrs(shared_attrs)
    base_coordinates: dict[str, Any] = {
        "time": ("time", connectivity.time, time_attrs),
        "frequency": (
            "frequency",
            connectivity.frequencies,
            frequency_attrs,
        ),
    }
    signal_coordinates: dict[str, Any] = {
        "source": ("source", signal_labels, {"long_name": "Source signal"}),
        "target": ("target", signal_labels, {"long_name": "Target signal"}),
    }
    extra_signal_coordinates = (
        {} if signal_metadata is None else dict(signal_metadata.coordinates)
    )
    source_extras = {
        f"source_{name}": ("source", values)
        for name, values in extra_signal_coordinates.items()
    }
    target_extras = {
        f"target_{name}": ("target", values)
        for name, values in extra_signal_coordinates.items()
    }
    signal_coordinates.update(source_extras)
    signal_coordinates.update(target_extras)
    measure_attrs = {
        **attrs,
        **_measure_label_attrs(
            method, None if signal_metadata is None else signal_metadata.units
        ),
    }

    if output_kind in {"pairwise", "power"}:
        connectivity_mat = np.asarray(numerical_result)
        expected_shape = power_shape if output_kind == "power" else pairwise_shape
        if tuple(connectivity_mat.shape) != expected_shape:
            msg = (
                f"The method '{method}' returned shape {connectivity_mat.shape}; "
                f"its wrapper contract requires {expected_shape}."
            )
            raise ValueError(msg)
        if transpose_output:
            connectivity_mat = np.swapaxes(connectivity_mat, -1, -2)
        coordinates = {
            **base_coordinates,
            "source": signal_coordinates["source"],
            **source_extras,
        }
    else:
        coordinates = dict(base_coordinates)

    if output_kind == "power":
        # squeeze has no meaning for power (no target axis); it is a no-op here.
        return xr.DataArray(
            connectivity_mat,
            coords=coordinates,
            dims=("time", "frequency", "source"),
            name=method,
            attrs=measure_attrs,
        )

    if output_kind == "pairwise":
        coordinates["target"] = signal_coordinates["target"]
        coordinates.update(target_extras)
        xar = xr.DataArray(
            connectivity_mat,
            coords=coordinates,
            dims=("time", "frequency", "source", "target"),
            name=method,
            attrs=measure_attrs,
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

    if output_kind == "group_pairwise":
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
            msg = (
                f"The method '{method}' returned shape {connectivity_mat.shape}; "
                f"its group-pairwise contract requires {expected_shape}."
            )
            raise ValueError(msg)
        if transpose_output:
            connectivity_mat = np.swapaxes(connectivity_mat, -1, -2)
        coordinates.update(
            {
                "source_group": ("source_group", group_labels, {"long_name": "Source group"}),
                "target_group": ("target_group", group_labels, {"long_name": "Target group"}),
            }
        )
        return xr.DataArray(
            connectivity_mat,
            coords=coordinates,
            dims=("time", "frequency", "source_group", "target_group"),
            name=method,
            attrs=measure_attrs,
        )

    if output_kind == "delay":
        connectivity_mat = np.asarray(numerical_result)
        frequencies = np.asarray(connectivity.frequencies)
        frequency_band = kwargs.get("frequencies_of_interest")
        if frequency_band is not None:
            frequencies = frequencies[_frequencies_in_band(frequencies, frequency_band)]
        delay_expected_shape = (
            len(connectivity.time),
            len(frequencies),
            connectivity_mat.shape[-3],
            connectivity.n_signals,
            connectivity.n_signals,
        )
        if connectivity_mat.shape != delay_expected_shape:
            msg = (
                f"The method '{method}' returned shape {connectivity_mat.shape}; "
                f"its delay contract requires {delay_expected_shape}."
            )
            raise ValueError(msg)
        coordinates = {
            "time": base_coordinates["time"],
            "frequency": ("frequency", frequencies, frequency_attrs),
            "candidate": (
                "candidate",
                np.arange(-int(kwargs.get("n_range", 3)), int(kwargs.get("n_range", 3)) + 1),
                {
                    "long_name": "Phase-wrap candidate",
                    "description": "k in delay = (phase + 2 pi k) / (2 pi f)",
                },
            ),
            **signal_coordinates,
        }
        return xr.DataArray(
            connectivity_mat,
            coords=coordinates,
            dims=("time", "frequency", "candidate", "source", "target"),
            name=method,
            attrs=measure_attrs,
        )

    if output_kind == "phase_slope":
        connectivity_mat = np.asarray(numerical_result)
        expected_shape = (
            len(connectivity.time),
            connectivity.n_signals,
            connectivity.n_signals,
        )
        if connectivity_mat.shape != expected_shape:
            msg = (
                f"The method '{method}' returned shape {connectivity_mat.shape}; "
                f"its phase-slope contract requires {expected_shape}."
            )
            raise ValueError(msg)
        return xr.DataArray(
            connectivity_mat,
            coords={"time": base_coordinates["time"], **signal_coordinates},
            dims=("time", "source", "target"),
            name=method,
            attrs={**measure_attrs, **_frequency_band_attrs(connectivity, kwargs)},
        )

    if output_kind == "group_delay":
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
                msg = f"The method '{method}' returned an invalid shape."
                raise ValueError(msg)
            variable_attrs = {
                **attrs,
                **_frequency_band_attrs(connectivity, kwargs),
                "long_name": long_name,
                "units": units,
            }
            data_vars[name] = xr.DataArray(
                values,
                coords=dataset_coordinates,
                dims=("time", "source", "target"),
                attrs=variable_attrs,
            )
        return xr.Dataset(data_vars, attrs=attrs)

    if output_kind == "global":
        scores, vectors = numerical_result
        scores = np.asarray(scores)[..., : len(connectivity.frequencies), :]
        vectors = np.asarray(vectors)[..., : len(connectivity.frequencies), :, :]
        n_components = scores.shape[-1]
        dataset_coordinates = {
            **base_coordinates,
            "component": ("component", np.arange(n_components), {"long_name": "Component"}),
            "source": signal_coordinates["source"],
            **source_extras,
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
                    attrs=measure_attrs,
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

    if output_kind == "multivariate_components":
        if not isinstance(numerical_result, MultivariateConnectivityResult):
            msg = f"The method '{method}' did not return MultivariateConnectivityResult."
            raise TypeError(msg)
        n_connections = numerical_result.scores.shape[-2]
        n_components = numerical_result.scores.shape[-1]
        expected_scores = (
            len(connectivity.time),
            len(connectivity.frequencies),
            n_connections,
            n_components,
        )
        if numerical_result.scores.shape != expected_scores:
            msg = (
                f"The method '{method}' returned score shape "
                f"{numerical_result.scores.shape}; expected {expected_scores}."
            )
            raise ValueError(msg)
        component_coordinates = {
            **base_coordinates,
            "connection": (
                "connection",
                np.arange(n_connections),
                {"long_name": "Group-pair connection"},
            ),
            "component": ("component", np.arange(n_components), {"long_name": "Component"}),
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
            "side": ("side", ["seed", "target"], {"long_name": "Side of the connection"}),
            "signal": ("signal", signal_labels, {"long_name": "Signal"}),
            "group": ("group", numerical_result.group_labels, {"long_name": "Signal group"}),
        }
        signal_extras = {
            f"signal_{name}": ("signal", values)
            for name, values in extra_signal_coordinates.items()
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
                attrs=measure_attrs,
            ),
            "group_membership": xr.DataArray(
                numerical_result.group_membership,
                coords={
                    "group": component_coordinates["group"],
                    "signal": component_coordinates["signal"],
                    **signal_extras,
                },
                dims=("group", "signal"),
                attrs={"long_name": "Signal belongs to group"},
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
        projection_coordinates.update(signal_extras)
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

    # A lone raise is exempt from mypy's unreachable check; a `msg` line is not.
    raise AssertionError(f"unreachable: unknown output kind for {method!r}")  # noqa: EM102


def _inclusive_frequency_mask(
    label: str,
    bounds: Any,
    frequencies: NDArray[np.floating],
) -> tuple[NDArray[np.bool_], tuple[float, float]]:
    """Validate ``(low, high)`` bounds and return the inclusive bin mask.

    Shared by ``frequency_range`` and ``frequency_bands`` so both arguments keep
    the same semantics and error messages; ``label`` names the offending
    argument or band in those messages.
    """
    try:
        lower, upper = bounds
    except (TypeError, ValueError) as error:
        msg = f"{label} must contain exactly two bounds (low, high)."
        raise ValueError(msg) from error
    lower = float(lower)
    upper = float(upper)
    if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
        msg = f"{label} must have finite bounds with low <= high; got ({lower!r}, {upper!r})."
        raise ValueError(msg)
    mask = (frequencies >= lower) & (frequencies <= upper)
    if not np.any(mask):
        msg = f"{label} ({lower:g}, {upper:g}) contains no frequency bins."
        raise ValueError(msg)
    return mask, (lower, upper)


def _band_integration_weights(
    frequencies: NDArray[np.floating],
    low: float,
    high: float,
    nyquist_frequency: float | None,
) -> NDArray[np.floating]:
    """Width of each bin's frequency cell that lies inside ``[low, high]``.

    Bin ``k`` owns the cell between the midpoints to its neighbours (the outer
    cells extend half a spacing, and never below 0 Hz on a non-negative grid),
    so ``sum(weights * density)`` integrates a piecewise-constant density
    exactly: any band edges, one-bin bands, and bands that tile additively.

    A grid anchored at 0 Hz is the one-sided spectrum of a real signal. Its DC
    and Nyquist bins are not doubled by the folding convention because each is
    its own mirror image (0 Hz) or alias (Nyquist), so their cells are folded
    onto the grid -- ``[0, spacing / 2]`` and ``[f_last - spacing / 2, f_last]``
    -- and weighted twice. Each edge bin then still contributes a full spacing
    of power and a band covering every bin integrates to
    ``sum(density) * spacing`` (Parseval). Only a bin at ``nyquist_frequency``
    is folded at the top: an odd-length FFT has no Nyquist bin, and after
    cropping the last bin is an ordinary interior bin whose density is already
    doubled.
    """
    midpoints = (frequencies[1:] + frequencies[:-1]) / 2
    lower = np.concatenate(
        ([frequencies[0] - (frequencies[1] - frequencies[0]) / 2], midpoints)
    )
    upper = np.concatenate(
        (midpoints, [frequencies[-1] + (frequencies[-1] - frequencies[-2]) / 2])
    )
    scale = np.ones(frequencies.shape)
    if frequencies[0] >= 0:
        lower = np.maximum(lower, 0.0)
    if frequencies[0] == 0.0:
        scale[0] = 2.0
    if nyquist_frequency is not None and np.isclose(
        frequencies[-1], nyquist_frequency, rtol=1e-12, atol=0.0
    ):
        upper[-1] = frequencies[-1]
        scale[-1] = 2.0
    weights: NDArray[np.floating] = scale * np.clip(
        np.minimum(upper, high) - np.maximum(lower, low), 0.0, None
    )
    return weights


# Provenance prefixes of the transforms whose results record ``sampling_frequency``.
_TRANSFORM_PROVENANCE_PREFIXES = ("mt_", "stft_", "welch_", "morlet_", "fourier_")


def _nyquist_frequency(result: xr.DataArray | xr.Dataset) -> float | None:
    """The Nyquist frequency of ``result``'s one-sided spectrum, if known.

    A result records its transform's sampling rate, so its Nyquist frequency is
    half of it; the grid has a bin there only for an even FFT length. Without a
    recorded rate, a grid anchored at 0 Hz is taken to be a complete one-sided
    spectrum whose last bin is Nyquist. Only the grid as computed can say so:
    cropping or decimating it makes an interior bin the last one.
    """
    for prefix in _TRANSFORM_PROVENANCE_PREFIXES:
        sampling_frequency = result.attrs.get(prefix + "sampling_frequency")
        if sampling_frequency is not None:
            return float(sampling_frequency) / 2
    if "frequency" not in result.coords:
        return None
    frequencies = np.asarray(result.coords["frequency"].values)
    if frequencies.ndim != 1 or frequencies.size < 2 or frequencies[0] != 0.0:
        return None
    return float(frequencies[-1])


def frequency_band_reduce(
    result: xr.DataArray | xr.Dataset,
    bands: Mapping[str, tuple[float, float]],
    *,
    reduction: Literal["mean", "integral"] = "mean",
    circular: bool | None = None,
) -> xr.DataArray | xr.Dataset:
    """Reduce a frequency-resolved result into labeled frequency bands.

    ``reduction="mean"`` averages the already-computed connectivity score over
    the bins in each inclusive band. Phase is treated specially: a
    ``coherence_phase`` result uses a circular mean, while complex-valued
    measures use their ordinary complex (vector) mean. ``reduction="integral"``
    integrates a spectral density over ``[low, high]`` and is intentionally
    restricted to ``power`` and ``cross_spectral_density``, where it represents
    band power/covariance rather than a frequency-averaged score. Each bin
    stands for the frequency cell between the midpoints to its neighbours, and
    contributes its density times the part of that cell inside the band, so
    band edges need not fall on bins, a one-bin band is not zero, and adjacent
    bands add up to their union. On a one-sided grid starting at 0 Hz the DC
    bin and the Nyquist bin (present for an even FFT length) own only the
    half-cell toward their neighbour but count with a full spacing, matching
    the one-sided convention in which those two bins are not doubled;
    integrating ``power`` over a band that covers every bin therefore
    reproduces the total signal power (Parseval).

    Parameters
    ----------
    result : xarray.DataArray or xarray.Dataset
        Result with a one-dimensional ``frequency`` coordinate.
    bands : mapping of str to (float, float)
        Inclusive lower and upper frequency bounds in the coordinate's units.
    reduction : {"mean", "integral"}, default="mean"
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

    The Nyquist frequency is half the sampling rate recorded in ``result``'s
    provenance attrs (e.g. ``mt_sampling_frequency``). If none is recorded,
    the last bin of a grid starting at 0 Hz is taken to be the Nyquist bin, so
    reduce such a result before cropping it.
    """
    return _reduce_frequency_bands(
        result,
        bands,
        reduction=reduction,
        circular=circular,
        nyquist_frequency=_nyquist_frequency(result),
    )


# ``long_name`` of a spectral density integrated over a band.
_BAND_INTEGRAL_LONG_NAMES = {
    "power": "Band power",
    "cross_spectral_density": "Band cross-power",
}


def _reduce_frequency_bands(
    result: xr.DataArray | xr.Dataset,
    bands: Mapping[str, tuple[float, float]],
    *,
    reduction: Literal["mean", "integral"],
    circular: bool | None,
    nyquist_frequency: float | None,
) -> xr.DataArray | xr.Dataset:
    """:func:`frequency_band_reduce` with the grid's Nyquist bin given.

    ``nyquist_frequency`` is the frequency of the one-sided spectrum's Nyquist
    bin, or ``None`` if the grid does not reach it or is not one-sided.
    """
    if "frequency" not in result.dims:
        msg = "result must have a 'frequency' dimension."
        raise ValueError(msg)
    if reduction not in {"mean", "integral"}:
        msg = "reduction must be either 'mean' or 'integral'."
        raise ValueError(msg)
    if not isinstance(bands, Mapping) or len(bands) == 0:  # type: ignore[redundant-expr]  # user input
        msg = "bands must be a non-empty mapping of names to bounds."
        raise ValueError(msg)

    frequencies = np.asarray(result.coords["frequency"].values)
    if frequencies.ndim != 1 or frequencies.size == 0:
        msg = "frequency must be a non-empty one-dimensional coordinate."
        raise ValueError(msg)
    if not np.all(np.isfinite(frequencies)):
        msg = "frequency must contain only finite values."
        raise ValueError(msg)
    if frequencies.size > 1 and not np.all(np.diff(frequencies) > 0):
        msg = "frequency must be strictly increasing for band reduction."
        raise ValueError(msg)

    band_names = list(bands)
    if len(set(band_names)) != len(band_names) or not all(
        isinstance(name, str) and name  # type: ignore[redundant-expr]  # user input
        for name in band_names
    ):
        msg = "band names must be unique, non-empty strings."
        raise ValueError(msg)

    band_masks_and_bounds = [
        _inclusive_frequency_mask(f"Band {name!r}", bounds, frequencies)
        for name, bounds in bands.items()
    ]
    if reduction == "integral" and frequencies.size < 2:
        msg = "reduction='integral' needs at least two frequency bins to know their widths."
        raise ValueError(msg)

    def _reduce_dataarray(data: xr.DataArray) -> xr.DataArray:
        measure = str(data.attrs.get("measure", "" if data.name is None else data.name))
        if reduction == "integral" and measure not in {
            "power",
            "cross_spectral_density",
        }:
            msg = (
                "reduction='integral' is defined only for power and "
                "cross_spectral_density; use reduction='mean' for "
                f"{measure or 'this result'!r}."
            )
            raise ValueError(msg)

        reduced_bands: list[xr.DataArray] = []
        band_validity: list[xr.DataArray] = []
        for mask, (low, high) in band_masks_and_bounds:
            if reduction == "integral":
                weights = _band_integration_weights(frequencies, low, high, nyquist_frequency)
                # Bins inside the band count for validity even with zero weight
                # (a zero-width band), so an invalid bin is never read as zero.
                used = np.flatnonzero((weights > 0) | mask)
                selected = data.isel(frequency=used)
            else:
                selected = data.isel(frequency=np.flatnonzero(mask))
            # A NaN bin (an edge-invalid or undefined estimate) makes the band
            # value undefined for every reduction; skipping it silently would
            # average a different set of bins per time point.
            if reduction == "integral":
                reduced = xr.dot(selected, xr.DataArray(weights[used], dims="frequency"))
            elif (
                circular
                if circular is not None
                else measure == "coherence_phase" or data.attrs.get("units") == "rad"
            ):
                # Circular mean prevents phases near -pi and +pi from
                # spuriously cancelling toward zero.
                phase_vectors = xr.apply_ufunc(np.exp, 1j * selected)
                reduced = xr.apply_ufunc(
                    np.angle,
                    phase_vectors.mean("frequency", skipna=False, keep_attrs=True),
                    keep_attrs=True,
                )
            else:
                reduced = selected.mean("frequency", skipna=False, keep_attrs=True)
            # Apply the shared validity rule after every reduction so an invalid
            # bin can never be hidden inside a band value.
            reduced = reduced.where(selected.notnull().all("frequency"))
            reduced_bands.append(reduced)
            if "valid_time_frequency" in selected.coords:
                band_validity.append(selected.coords["valid_time_frequency"].all("frequency"))

        reduced = xr.concat(reduced_bands, dim="band").assign_coords(band=band_names)
        edge_attrs = {
            key: value
            for key, value in data.coords["frequency"].attrs.items()
            if key == "units"
        }
        reduced = reduced.assign_coords(
            band_lower=("band", [low for _, (low, _) in band_masks_and_bounds], edge_attrs),
            band_upper=("band", [high for _, (_, high) in band_masks_and_bounds], edge_attrs),
        )
        if band_validity:
            reduced = reduced.assign_coords(
                valid_time_band=xr.concat(band_validity, dim="band")
                .assign_coords(band=band_names)
                # Any surviving non-frequency axes (typically "time", but none
                # if the caller already selected a single time point) come
                # first; "band" is placed last without naming "time" explicitly.
                .transpose(..., "band")
                .assign_attrs(
                    long_name="Every bin of the band has full wavelet and smoothing support"
                )
            )
        desired_dims = tuple(
            "band" if dimension == "frequency" else dimension for dimension in data.dims
        )
        reduced = reduced.transpose(*desired_dims)
        reduced.attrs = dict(data.attrs)
        if reduction == "integral":
            # Integrating a density over frequency removes its per-Hz unit; a
            # unit that does not end in /Hz cannot be converted, so it is dropped.
            reduced.attrs["long_name"] = _BAND_INTEGRAL_LONG_NAMES[measure]
            units = reduced.attrs.pop("units", None)
            if isinstance(units, str) and units.endswith("/Hz"):
                reduced.attrs["units"] = units.removesuffix("/Hz")
        reduced.attrs["frequency_bands_json"] = _canonical_json(bands)
        reduced.attrs["frequency_reduction"] = reduction
        return reduced

    if isinstance(result, xr.DataArray):
        return _reduce_dataarray(result)

    non_reducible_variables = sorted(
        str(name)
        for name, data in result.data_vars.items()
        if "frequency" in data.dims
        and (
            name == "global_coherence_vectors" or str(name).endswith(("_filters", "_patterns"))
        )
    )
    if non_reducible_variables:
        names = ", ".join(repr(name) for name in non_reducible_variables)
        msg = (
            "Frequency-band reduction is not defined for spatial filters, "
            "patterns, or component vectors because their sign/phase is "
            f"arbitrary at each frequency; offending variables: {names}. "
            "Select the scalar score variable from the Dataset and pass that "
            "DataArray to frequency_band_reduce, or keep the full "
            "frequency-resolved Dataset."
        )
        raise ValueError(msg)

    # The band record lives on each reduced variable (see _reduce_dataarray).
    # Start from the attrs and coordinates on no frequency axis, then re-add every
    # variable in its original order.
    return (
        result.drop_vars(list(result.data_vars))
        .drop_dims("frequency")
        .assign(
            {
                name: _reduce_dataarray(data) if "frequency" in data.dims else data
                for name, data in result.data_vars.items()
            }
        )
    )


def _with_frequency_attrs(
    result: xr.DataArray | xr.Dataset, **new_attrs: Any
) -> xr.DataArray | xr.Dataset:
    """Record frequency-operation provenance on each variable with a frequency axis.

    A Dataset may also hold frequency-reduced variables (phase_slope_index,
    group_delay) that the operation did not touch, so the record goes on the
    variables it describes, never on the Dataset as a whole.
    """
    if isinstance(result, xr.DataArray):
        return result.assign_attrs(new_attrs)
    return result.assign(
        {
            name: variable.assign_attrs(new_attrs)
            for name, variable in result.data_vars.items()
            if "frequency" in variable.dims
        }
    )


def _select_and_reduce_frequencies(
    result: xr.DataArray | xr.Dataset,
    *,
    frequency_range: tuple[float, float] | None,
    frequency_decimation: int,
    frequency_bands: Mapping[str, tuple[float, float]] | None,
    frequency_reduction: Literal["mean", "integral"],
) -> xr.DataArray | xr.Dataset:
    """Apply the wrapper's shared, coordinate-aware frequency operations."""
    if not is_positive_integer(frequency_decimation):
        msg = "frequency_decimation must be a positive integer."
        raise ValueError(msg)

    selected = result
    # Before cropping or decimation, which can make an interior bin the last.
    nyquist_frequency = _nyquist_frequency(result)
    requests_frequency_operation = (
        frequency_range is not None or frequency_decimation != 1 or frequency_bands is not None
    )
    if requests_frequency_operation and "frequency" not in selected.dims:
        msg = (
            "This result has no frequency dimension: the requested method "
            "already reduces frequency (for example phase_slope_index or "
            "group_delay), so frequency_range, frequency_decimation, and "
            "frequency_bands cannot be applied afterward. Pass the method's "
            "frequencies_of_interest argument through connectivity_kwargs instead."
        )
        raise ValueError(msg)
    if frequency_range is not None:
        mask, (lower, upper) = _inclusive_frequency_mask(
            "frequency_range",
            frequency_range,
            np.asarray(selected.coords["frequency"].values),
        )
        selected = _with_frequency_attrs(
            selected.isel(frequency=np.flatnonzero(mask)),
            frequency_range_json=_canonical_json((lower, upper)),
        )

    if frequency_decimation != 1:
        selected = _with_frequency_attrs(
            selected.isel(frequency=slice(None, None, frequency_decimation)),
            frequency_decimation=int(frequency_decimation),
        )

    if frequency_bands is not None:
        selected = _reduce_frequency_bands(
            selected,
            frequency_bands,
            reduction=frequency_reduction,
            circular=None,
            nyquist_frequency=nyquist_frequency,
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

    Parameters
    ----------
    m : transform
        A spectral transform (e.g. ``Multitaper``, ``MorletWavelet``, ``Welch``)
        whose coefficients the measure is computed from.
    method : str, default="coherence_magnitude"
        Measure name from :func:`list_measures`.
    signal_names : sequence, optional
        Labels for the ``source``/``target`` coordinates; defaults to
        ``"0"``, ``"1"``, ....
    squeeze : bool, default=False
        With exactly 2 signals, reduce a pairwise measure to the ordered pair
        (first source, last target), keeping ``source`` and ``target`` as
        scalar coordinates.
    **kwargs
        Keyword arguments for the measure (e.g. ``group_labels``).

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
    _validate_method_names([method])
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
            msg = (
                "transform.valid_time_frequency must have shape "
                f"{expected_shape}, got {validity.shape}."
            )
            raise ValueError(msg)
        validity_attrs = {"long_name": "Full wavelet and smoothing support is in-record"}
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
                frequency_index = _frequencies_in_band(frequencies, frequency_band)
            valid_time = validity[:, frequency_index].all(axis=1)
            result = result.assign_coords(valid_time=(("time",), valid_time, validity_attrs))
    return result


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


def _format_and_reduce_measures(
    connectivity: Connectivity,
    methods: list[str],
    *,
    return_dataarray: bool,
    signal_labels: NDArray[Any],
    squeeze: bool,
    shared_attrs: Mapping[str, Any],
    connectivity_kwargs: Mapping[str, Any],
    frequency_range: tuple[float, float] | None,
    frequency_decimation: int,
    frequency_bands: Mapping[str, tuple[float, float]] | None,
    frequency_reduction: Literal["mean", "integral"],
    signal_metadata: _SignalMetadata | None = None,
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
            signal_metadata=signal_metadata,
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
                        signal_metadata=signal_metadata,
                        **connectivity_kwargs,
                    )
                )
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

    return _select_and_reduce_frequencies(
        result,
        frequency_range=frequency_range,
        frequency_decimation=frequency_decimation,
        frequency_bands=frequency_bands,
        frequency_reduction=frequency_reduction,
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
    source -> target layout.) Signed undirected phase measures
    (``coherence_phase``, ``imaginary_coherency``, ``phase_lag_index``,
    ``weighted_phase_lag_index``) are positive at ``sel(source=a, target=b)``
    when ``a`` leads ``b``.

    Every variable has ``long_name`` and ``units`` attrs (``"1"`` for
    dimensionless scores, ``"rad"`` for phase, ``"s"`` for delay; spectral
    densities are ``"(<units>)^2/Hz"`` when an input DataArray states its
    ``units``, and ``"(<units>)^2"`` once integrated over a band). Non-index coordinates on an input DataArray's signal dimension
    (e.g. ``region``) are carried as ``source_<name>``/``target_<name>``.

    Real-valued results write with any NetCDF engine (booleans are stored as
    0/1). Complex results (``coherency``, ``cross_spectral_density``,
    ``canonical_coherency``, and the global-coherence vectors) need an engine
    that stores complex data, e.g.
    ``result.to_netcdf("result.h5", engine="h5netcdf", invalid_netcdf=True)``,
    or netCDF4 >= 1.7 with ``engine="netcdf4", auto_complex=True`` (open with the
    same option).

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
        msg = "method must name at least one connectivity measure; got an empty list."
        raise ValueError(msg)
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
    signal_labels = _validated_signal_labels(signal_names, shared_connectivity.n_signals)
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
        signal_metadata=signal_metadata,
    )


def fourier_connectivity(
    fourier_coefficients: NDArray[np.complexfloating] | xr.DataArray,
    frequencies: NDArray[np.floating] | None = None,
    time: NDArray[np.floating] | None = None,
    method: str | list[str] | None = None,
    signal_names: Sequence[_SignalLabel] | None = None,
    squeeze: bool = False,
    connectivity_kwargs: dict[str, Any] | None = None,
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
        coordinate when not given.
    time : array, shape (n_time,), optional
        Center time of each window in seconds; defaults to window indices.
    method : str or list of str, optional
        Measure name(s) from :func:`list_measures`. A single name returns a
        DataArray; a list (or ``None`` for :data:`DEFAULT_METHODS`) returns a
        Dataset with one variable per measure. With ``None``, measures that
        require a two-sided spectrum are omitted when the input is one-sided,
        or when it has no frequency coordinate and ``is_one_sided`` was not
        passed to declare its sidedness.
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
        Keyword arguments passed to every requested measure (for example
        ``group_labels`` for group measures). Measures that need different
        arguments must be requested in separate calls.
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
    frequency_range : (float, float), optional
        Inclusive ``(low, high)`` bounds in Hz to keep before any decimation
        or band reduction.
    frequency_decimation : int, default=1
        Keep every ``frequency_decimation``-th frequency bin.
    frequency_bands : mapping of str to (float, float), optional
        Named inclusive bands to reduce the frequency axis into; see
        :func:`frequency_band_reduce`.
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
    if getattr(getattr(coefficient_data, "dtype", None), "kind", None) != "c":
        msg = "fourier_coefficients must be complex-valued."
        raise TypeError(msg)
    if time is not None and not _is_real_numeric_dtype(np.asarray(time).dtype):
        msg = (
            "time must contain numeric elapsed seconds (window centers); "
            f"got dtype {np.asarray(time).dtype!r}. Convert a datetime axis to "
            "elapsed seconds, e.g. (t - t[0]) / np.timedelta64(1, 's')."
        )
        raise TypeError(msg)
    inferred_one_sided = False
    if is_one_sided is not None and not isinstance(is_one_sided, (bool, np.bool_)):
        # Runtime check of user input the annotation already excludes; a lone
        # raise is exempt from mypy's unreachable check.
        raise TypeError("is_one_sided must be a boolean or None.")  # noqa: EM101
    if frequencies is not None:
        frequency_values = np.asarray(frequencies, dtype=float)
        if frequency_values.ndim != 1:
            msg = "frequencies must be a one-dimensional coordinate."
            raise ValueError(msg)
        inferred_one_sided = bool(
            frequency_values.size > 0 and not np.any(frequency_values < 0)
        )
        one_sided = inferred_one_sided if is_one_sided is None else bool(is_one_sided)
        # A one-sided coordinate (non-negative, strictly increasing) is validated
        # by Connectivity itself; only the two-sided FFT-order check lives here.
        if not one_sided and frequency_values.size == 1 and frequency_values[0] != 0.0:
            msg = (
                "frequencies must be uniformly spaced in standard FFT "
                "order (a one-bin two-sided spectrum can contain only zero Hz)."
            )
            raise ValueError(msg)
        if not one_sided and frequency_values.size > 1:
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
                msg = (
                    "frequencies must be uniformly spaced in standard FFT "
                    "order (zero and positive bins followed by negative bins)."
                )
                raise ValueError(msg)
    else:
        if is_one_sided is None:
            warnings.warn(
                "fourier_connectivity received no frequency coordinate and no "
                "is_one_sided flag; assuming a two-sided spectrum in standard "
                "FFT order. For rfft or wavelet coefficients (non-negative "
                "frequencies only) pass is_one_sided=True, otherwise "
                "is_one_sided=False to silence this warning.",
                UserWarning,
                stacklevel=2,
            )
        one_sided = bool(is_one_sided) if is_one_sided is not None else False

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
    if connectivity_kwargs is None:
        connectivity_kwargs = {}

    return_dataarray = isinstance(method, str)
    if method is None:
        # Two-sided-only measures are rejected below when sidedness is neither
        # verifiable nor declared, so leave them out of the default set then too.
        methods = [
            name
            for name in DEFAULT_METHODS
            if not (
                (one_sided or (frequencies is None and is_one_sided is None))
                and name in _MEASURE_SPECS
                and _MEASURE_SPECS[name].requires_two_sided
            )
        ]
    elif isinstance(method, str):
        methods = [method]
    else:
        methods = list(method)
    if not methods:
        msg = "method must name at least one connectivity measure; got an empty list."
        raise ValueError(msg)
    _validate_method_names(methods)
    if frequencies is None:
        # Without a frequency coordinate two-sidedness cannot be verified, so an
        # *assumed* two-sided spectrum (is_one_sided=None) must not let one-sided
        # input (e.g. rfft/wavelet coefficients) reach Wilson factorization and
        # produce a silently wrong result. Reject methods that declare the
        # full-spectrum requirement unless the caller declared the spectrum
        # two-sided with is_one_sided=False; other directional measures such as
        # dPLI and PSI remain valid on one-sided coefficients.
        two_sided_methods = [
            name
            for name in methods
            if name in _MEASURE_SPECS and _MEASURE_SPECS[name].requires_two_sided
        ]
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
                    stacklevel=2,
                )
    signal_labels = _validated_signal_labels(signal_names, connectivity.n_signals)
    metadata: dict[str, Any] = {
        "source": "external_fourier_coefficients",
        "coefficient_shape_json": _canonical_json(tuple(coefficient_data.shape)),
        "frequency_coordinate": "provided" if frequencies is not None else "normalized",
        "time_coordinate": "provided" if time is not None else "index",
        "is_one_sided": one_sided,
        "one_sided_inferred": is_one_sided is None and inferred_one_sided,
    }
    all_frequencies = connectivity.all_frequencies
    if not one_sided and all_frequencies.size > 1:
        # FFT-order bins are sampling_frequency / n_fft apart (bin 1 is -spacing
        # when n_fft == 2), so the rate follows and places the Nyquist bin,
        # which only an even n_fft has.
        metadata["sampling_frequency"] = float(all_frequencies.size * abs(all_frequencies[1]))
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
        signal_metadata=signal_metadata,
    )
