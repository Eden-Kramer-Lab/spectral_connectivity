"""Frequency cropping, decimation, and band reduction of labeled results."""

import warnings
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from spectral_connectivity._measure_registry import (
    _FREQUENCY_RANGE_MEASURES,
    _listed,
    _reduces_frequency,
)
from spectral_connectivity._provenance import _canonical_json, _with_frequency_grid_metadata
from spectral_connectivity.connectivity import (
    _bins_on_bound,
    _frequency_interval_mask,
    _FrequencySelection,
)
from spectral_connectivity.utils import is_positive_integer, stacklevel_outside_package


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
    doubled. The last bin is at ``nyquist_frequency`` by the rule that places a
    bin on a band's bound (:func:`_frequency_interval_mask`), so a coordinate
    stored as float32 still has its Nyquist bin.
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
    if nyquist_frequency is not None and _bins_on_bound(frequencies, nyquist_frequency)[-1]:
        upper[-1] = frequencies[-1]
        scale[-1] = 2.0
    weights: NDArray[np.floating] = scale * np.clip(
        np.minimum(upper, high) - np.maximum(lower, low), 0.0, None
    )
    return weights


def _recorded_nyquist_frequency(attrs: Mapping[Any, Any]) -> float | None:
    """The Nyquist frequency recorded in a result's or variable's attributes."""
    nyquist_frequency = attrs.get("nyquist_frequency", attrs.get("mt_nyquist_frequency"))
    return None if nyquist_frequency is None else float(nyquist_frequency)


def _nyquist_frequency(
    attrs: Mapping[Any, Any], frequencies: NDArray[np.floating] | None
) -> float | None:
    """The Nyquist frequency of a one-sided spectrum, if known.

    ``attrs`` are a result's attributes and ``frequencies`` its grid as
    computed. A result records ``nyquist_frequency`` (or, from 2.x,
    ``mt_nyquist_frequency``) when its sampling rate or normalized grid is
    known; the grid has a bin there only for an even FFT length. Without it, a
    grid anchored at 0 is taken to be a complete one-sided spectrum whose last
    bin is Nyquist. Only the grid as computed can say so: cropping or
    decimating it makes an interior bin the last one, so pass the grid before
    either.
    """
    nyquist_frequency = _recorded_nyquist_frequency(attrs)
    if nyquist_frequency is not None:
        return nyquist_frequency
    if frequencies is None:
        return None
    if frequencies.ndim != 1 or frequencies.size < 2 or frequencies[0] != 0.0:
        return None
    return float(frequencies[-1])


# ``long_name`` of a spectral density integrated over a band.
_BAND_INTEGRAL_LONG_NAMES = {
    "power": "Band power",
    "cross_spectral_density": "Band cross-power",
}

# What a band value is, by reduction. A reduced variable's ``interpretation``
# and ``formula`` describe one frequency bin, so the reduction is named in
# them; its range, sign convention, quantity, and diagonal policy hold of the
# band values as they are (a mean or integral of values within a range stays
# within it, and a NaN or constant diagonal stays one).
_BAND_VALUE_DESCRIPTIONS = {
    "mean": "the mean over the band's frequency bins",
    "circular_mean": (
        "the circular mean (the angle of the mean unit phasor) over the band's frequency bins"
    ),
    "complex_mean": "the complex (vector) mean over the band's frequency bins",
    "integral": "the integral over the band",
}


def _reduce_frequency_bands(
    result: xr.DataArray | xr.Dataset,
    frequency_bands: Mapping[str, tuple[float, float]],
    *,
    frequency_reduction: Literal["mean", "integral"],
    circular: bool | None,
    nyquist_frequency: float | None,
) -> xr.DataArray | xr.Dataset:
    """:func:`frequency_band_reduce` with the grid's Nyquist bin given.

    ``nyquist_frequency`` is the frequency of the one-sided spectrum's Nyquist
    bin, or ``None`` if the grid does not reach it or is not one-sided. A
    variable that records its own (a Dataset assembled from result variables
    has no attributes of its own) is integrated with that one.
    """
    if "frequency" not in result.dims:
        msg = "result must have a 'frequency' dimension."
        raise ValueError(msg)
    if frequency_reduction not in {"mean", "integral"}:
        msg = "frequency_reduction must be either 'mean' or 'integral'."
        raise ValueError(msg)
    if not isinstance(frequency_bands, Mapping) or len(frequency_bands) == 0:  # type: ignore[redundant-expr]  # user input
        msg = "frequency_bands must be a non-empty mapping of names to bounds."
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

    band_names = list(frequency_bands)
    if len(set(band_names)) != len(band_names) or not all(
        isinstance(name, str) and name  # type: ignore[redundant-expr]  # user input
        for name in band_names
    ):
        msg = "band names must be unique, non-empty strings."
        raise ValueError(msg)

    band_masks_and_bounds = [
        _frequency_interval_mask(f"Band {name!r}", bounds, frequencies)
        for name, bounds in frequency_bands.items()
    ]
    discarded = [
        name
        for name in ("band", "band_lower", "band_upper", "valid_time_band")
        if name in result.coords
    ]
    if discarded:
        warnings.warn(
            f"Band reduction defines its own coordinates named {discarded}, so the "
            "result's existing ones (e.g. scalar recording coordinates carried "
            "from the input) are discarded. Rename them before reducing to keep "
            "them.",
            UserWarning,
            stacklevel=stacklevel_outside_package(),
        )
        # Discard them first: one left on a Dataset would take precedence over
        # the per-band coordinate of each reduced variable.
        result = result.drop_vars(discarded)
    if frequency_reduction == "integral" and frequencies.size < 2:
        msg = (
            "frequency_reduction='integral' needs at least two frequency bins to "
            "know their widths."
        )
        raise ValueError(msg)

    def _reduce_dataarray(data: xr.DataArray) -> xr.DataArray:
        measure = str(data.attrs.get("measure", "" if data.name is None else data.name))
        if frequency_reduction == "integral" and measure not in {
            "power",
            "cross_spectral_density",
        }:
            msg = (
                "frequency_reduction='integral' is defined only for power and "
                "cross_spectral_density; use frequency_reduction='mean' for "
                f"{measure or 'this result'!r}."
            )
            raise ValueError(msg)

        if frequency_reduction == "integral":
            reduction = "integral"
        elif (
            circular
            if circular is not None
            else measure == "coherence_phase" or data.attrs.get("units") == "rad"
        ):
            reduction = "circular_mean"
        else:
            reduction = "complex_mean" if np.iscomplexobj(data) else "mean"

        recorded_nyquist = _recorded_nyquist_frequency(data.attrs)
        variable_nyquist = nyquist_frequency if recorded_nyquist is None else recorded_nyquist
        reduced_bands: list[xr.DataArray] = []
        band_validity: list[xr.DataArray] = []
        for mask, (low, high) in band_masks_and_bounds:
            if frequency_reduction == "integral":
                weights = _band_integration_weights(frequencies, low, high, variable_nyquist)
                # Bins inside the band count for validity even with zero weight
                # (a zero-width band), so an invalid bin is never read as zero.
                used = np.flatnonzero((weights > 0) | mask)
                selected = data.isel(frequency=used)
            else:
                selected = data.isel(frequency=np.flatnonzero(mask))
            # A NaN bin (an edge-invalid or undefined estimate) makes the band
            # value undefined for every reduction; skipping it silently would
            # average a different set of bins per time point.
            if frequency_reduction == "integral":
                reduced = xr.dot(selected, xr.DataArray(weights[used], dims="frequency"))
                # Some xarray versions (e.g. 2025.6) drop coordinate attrs in
                # xr.dot; restore them from the input's coordinates.
                for name, coordinate in reduced.coords.items():
                    if name in selected.coords:
                        coordinate.attrs.update(selected.coords[name].attrs)
            elif reduction == "circular_mean":
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

        band = ("band", band_names, {"long_name": "Frequency band"})
        reduced = xr.concat(reduced_bands, dim="band").assign_coords(band=band)
        units = {
            key: value
            for key, value in data.coords["frequency"].attrs.items()
            if key == "units"
        }
        reduced = reduced.assign_coords(
            band_lower=(
                "band",
                [low for _, (low, _) in band_masks_and_bounds],
                {"long_name": "Lower edge of the frequency band", **units},
            ),
            band_upper=(
                "band",
                [high for _, (_, high) in band_masks_and_bounds],
                {"long_name": "Upper edge of the frequency band", **units},
            ),
        )
        if band_validity:
            reduced = reduced.assign_coords(
                valid_time_band=xr.concat(band_validity, dim="band")
                # The same labeled band coordinate, so the merge keeps its attrs.
                .assign_coords(band=band)
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
        description = _BAND_VALUE_DESCRIPTIONS[reduction]
        if "interpretation" in reduced.attrs:
            reduced.attrs["interpretation"] = (
                f"Each value is {description} of the per-frequency values. "
                f"{reduced.attrs['interpretation']}"
            )
        if "formula" in reduced.attrs:
            reduced.attrs["formula"] = f"{description} of {reduced.attrs['formula']}"
        if frequency_reduction == "integral":
            # Integrating a density over frequency removes its per-Hz unit; a
            # unit that does not end in /Hz cannot be converted, so it is dropped.
            reduced.attrs["long_name"] = _BAND_INTEGRAL_LONG_NAMES[measure]
            units = reduced.attrs.pop("units", None)
            if isinstance(units, str) and units.endswith("/Hz"):
                reduced.attrs["units"] = units.removesuffix("/Hz")
        reduced.attrs["frequency_bands_json"] = _canonical_json(frequency_bands)
        reduced.attrs["frequency_reduction"] = frequency_reduction
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


def _check_frequency_operations(
    methods: Sequence[str],
    connectivity_kwargs: Mapping[str, Any],
    *,
    frequency_decimation: int,
    frequency_bands: Mapping[str, tuple[float, float]] | None,
) -> None:
    """Reject the wrappers' frequency operations a request cannot honor.

    Runs before any measure is computed. ``frequency_range`` reaches the
    measures that consume an interval (``_FREQUENCY_RANGE_MEASURES``) as their
    own keyword, so it is a wrapper argument only. ``frequency_bands`` and ``frequency_decimation`` act on the
    frequency axis of frequency-resolved variables; ``phase_slope_index`` and
    ``group_delay`` have already reduced theirs over ``frequency_range``, so
    those variables are left as computed, never reduced a second time. A
    request in which every measure is frequency-reduced would be left entirely
    unchanged, so it is refused.
    """
    if "frequency_range" in connectivity_kwargs:
        msg = (
            "frequency_range was given inside connectivity_kwargs; pass "
            "frequency_range=(low, high) to the wrapper itself. It selects the bins "
            f"{_listed(_FREQUENCY_RANGE_MEASURES)} consume and crops every other "
            "measure to the same bins."
        )
        raise ValueError(msg)
    if not is_positive_integer(frequency_decimation):
        msg = "frequency_decimation must be a positive integer."
        raise ValueError(msg)
    if (frequency_bands is not None or frequency_decimation != 1) and all(
        _reduces_frequency(method) for method in methods
    ):
        operation = (
            "frequency_bands and frequency_decimation"
            if frequency_bands is not None and frequency_decimation != 1
            else "frequency_bands"
            if frequency_bands is not None
            else "frequency_decimation"
        )
        msg = (
            f"{operation} cannot be applied to {_listed(methods)}: these measures "
            "already reduce their frequency bins (every bin, or those of "
            "frequency_range) to one value, while frequency_bands and "
            "frequency_decimation act on the frequency axis of frequency-resolved "
            "measures. To summarize several bands, call once per band with "
            "frequency_range=(low, high); to combine fewer, independent bins, pass "
            "connectivity_kwargs={'frequency_resolution': ...}."
        )
        raise ValueError(msg)


def _crop_frequency_range(
    result: xr.DataArray | xr.Dataset, selection: _FrequencySelection
) -> xr.DataArray | xr.Dataset:
    """Cut a frequency-resolved result to the call's ``frequency_range``.

    ``selection`` is that range, resolved once on the full grid, which is the
    frequency axis of ``result``. Each frequency-resolved variable records
    ``selection.attrs()``, the interval record a measure that consumed the
    range writes for itself, so every variable the range applied to carries
    the same attributes.
    """
    return _with_frequency_attrs(result.isel(frequency=selection.index), **selection.attrs())


def _decimate_and_reduce_frequencies(
    result: xr.DataArray | xr.Dataset,
    *,
    frequency_decimation: int,
    frequency_bands: Mapping[str, tuple[float, float]] | None,
    frequency_reduction: Literal["mean", "integral"],
    nyquist_frequency: float | None,
) -> xr.DataArray | xr.Dataset:
    """Apply the wrappers' decimation and band reduction after the crop.

    ``nyquist_frequency`` is that of the grid as computed, before cropping
    (see :func:`_nyquist_frequency`).
    """
    selected = result
    if frequency_decimation != 1:
        selected = _with_frequency_attrs(
            selected.isel(frequency=slice(None, None, frequency_decimation)),
            frequency_decimation=int(frequency_decimation),
        )

    if frequency_bands is not None:
        selected = _reduce_frequency_bands(
            selected,
            frequency_bands,
            frequency_reduction=frequency_reduction,
            circular=None,
            nyquist_frequency=nyquist_frequency,
        )
    return _with_frequency_grid_metadata(selected)
