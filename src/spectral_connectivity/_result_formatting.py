"""Format one connectivity measure's result as a labeled xarray object."""

import inspect
import warnings
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from spectral_connectivity._input_handling import _SignalMetadata
from spectral_connectivity._measure_registry import (
    _get_measure_spec,
    _measure_label_attrs,
)
from spectral_connectivity._provenance import _canonical_json, _store_provenance_item
from spectral_connectivity.connectivity import (
    Connectivity,
    MultivariateConnectivityResult,
    _frequencies_in_band,
)


class UnsupportedMeasureError(ValueError):
    """A method has no registered semantic xarray output contract.

    Built-in nonstandard results (components, groups, delays, and multi-variable
    outputs) have explicit schemas. This exception remains for unregistered
    extensions whose returned shape cannot be inferred safely. It subclasses
    ``ValueError`` for backward compatibility and lets multi-measure wrappers
    distinguish structural incompatibility from genuine numerical errors.
    """


# Raised by and documented with the public wrapper API; keep that its home for
# tracebacks, pickling, and the API reference.
UnsupportedMeasureError.__module__ = "spectral_connectivity.wrapper"


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
