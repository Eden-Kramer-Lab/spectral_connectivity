"""Frequency cropping, decimation, and band reduction of labeled results."""

from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from spectral_connectivity._provenance import _canonical_json
from spectral_connectivity.utils import is_positive_integer


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
