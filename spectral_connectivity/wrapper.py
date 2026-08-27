"""Functions for getting connectivity measures in a labeled array format."""

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from logging import getLogger
from typing import Any, Literal

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from spectral_connectivity.connectivity import Connectivity
from spectral_connectivity.transforms import Multitaper
from spectral_connectivity.utils import get_compute_backend

logger = getLogger(__name__)


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


def _connectivity_result_to_xarray(
    connectivity: Connectivity,
    multitaper_metadata: Mapping[str, Any],
    method: str,
    signal_names: Sequence[str] | None,
    squeeze: bool,
    **kwargs: Any,
) -> xr.DataArray:
    """Format one result from an already-built ``Connectivity`` instance."""
    measure_spec = _get_measure_spec(method)
    if signal_names is None:
        signal_names = [str(index) for index in range(connectivity.n_signals)]
    elif len(signal_names) != connectivity.n_signals:
        raise ValueError(
            f"signal_names must contain {connectivity.n_signals} names, "
            f"got {len(signal_names)}."
        )
    elif len(set(signal_names)) != len(signal_names):
        duplicates = sorted(
            {n for n in signal_names if list(signal_names).count(n) > 1}
        )
        raise ValueError(
            "signal_names must be unique to label the source/target axes; "
            f"duplicates: {duplicates}."
        )
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

    if connectivity.n_signals > 2 and squeeze:
        warnings.warn(
            f"squeeze=True but there are {connectivity.n_signals} signals; "
            "returning the full (source, target) matrix.",
            UserWarning,
            stacklevel=2,
        )

    if measure_spec.output_kind == "power":
        xar = xr.DataArray(
            connectivity_mat,
            coords=[connectivity.time, connectivity.frequencies, signal_names],
            dims=["time", "frequency", "source"],
        )
    else:
        xar = xr.DataArray(
            connectivity_mat,
            coords=[
                connectivity.time,
                connectivity.frequencies,
                signal_names,
                signal_names,
            ],
            dims=["time", "frequency", "source", "target"],
        )
        if connectivity.n_signals == 2 and squeeze:
            # Reduce to the single ordered pair (first source, last target).
            # drop=False keeps ``source`` and ``target`` as scalar coordinates so
            # the returned (time, frequency) array still records which pair -- and
            # for directed measures, which direction -- it represents.
            xar = xar.isel(source=0, target=-1, drop=False)

    xar.name = method

    # The caller captures metadata from the same transform used to build the
    # Connectivity. The prefix avoids xarray's ``.dt`` accessor treating a bare
    # attribute name as a datetime coordinate.
    for attr, value in multitaper_metadata.items():
        xar.attrs["mt_" + attr] = value

    # CF-style coordinate metadata so the result is self-describing (plotting
    # libraries and NetCDF readers use units / long_name for axis labels).
    if "time" in xar.coords:
        xar.coords["time"].attrs.setdefault("long_name", "Time")
        xar.coords["time"].attrs.setdefault("units", "s")
    if "frequency" in xar.coords:
        xar.coords["frequency"].attrs.setdefault("long_name", "Frequency")
        xar.coords["frequency"].attrs.setdefault("units", "Hz")
    for signal_axis in ("source", "target"):
        if signal_axis in xar.coords:
            xar.coords[signal_axis].attrs.setdefault("long_name", "Signal")

    # Provenance: enough to trace how the result was produced. All values are
    # NetCDF-serializable strings/numbers (the mt_* attributes above already
    # record the multitaper parameters).
    xar.attrs["measure"] = method
    xar.attrs["package"] = "spectral_connectivity"
    xar.attrs["package_version"] = _package_version()
    # get_compute_backend() reports the backend actually imported (numpy vs
    # cupy), not the current env var; is_gpu_enabled() would mislabel a result if
    # SPECTRAL_CONNECTIVITY_ENABLE_GPU changed after import.
    xar.attrs["backend"] = get_compute_backend()["backend"].upper()
    xar.attrs["expectation_type"] = connectivity.expectation_type
    # Record the measure's keyword arguments; stringify anything that is not a
    # plain NetCDF-serializable scalar so the record cannot break to_netcdf.
    for key, value in kwargs.items():
        xar.attrs["arg_" + key] = (
            value
            if isinstance(value, (str, int, float, np.integer, np.floating, np.bool_))
            else str(value)
        )

    return xar


def connectivity_to_xarray(
    m: Multitaper,
    method: str = "coherence_magnitude",
    signal_names: Sequence[str] | None = None,
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


def multitaper_connectivity(
    time_series: NDArray[np.floating],
    sampling_frequency: float,
    time_window_duration: float | None = None,
    method: str | list[str] | None = None,
    signal_names: Sequence[str] | None = None,
    squeeze: bool = False,
    connectivity_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> xr.DataArray | xr.Dataset:
    """
    Compute connectivity measures with multitaper spectral estimation.

    This is the main high-level function for connectivity analysis. It performs
    multitaper spectral analysis on the input time series and computes the
    requested connectivity measures, returning results as labeled xarray objects.

    Parameters
    ----------
    time_series : NDArray[floating],
        shape (n_times, n_trials, n_channels) or (n_times, n_channels)
        Time series data. For multiple trials, trials are averaged in spectral domain.
    sampling_frequency : float
        Sampling rate in Hz of the time series data.
    time_window_duration : float, optional
        Duration of sliding window in seconds for time-resolved analysis.
        If None, analyzes entire time series (no time resolution).
    method : str or list of str, optional
        Connectivity method(s) to compute. If None, computes the default set of
        real-valued measures that fit the xarray/NetCDF interface (see
        ``DEFAULT_METHODS``) — not every measure. ``coherency`` is left out of the
        default only because it is complex (NetCDF cannot store it), but it can be
        requested by name. The directed-transfer-function family
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
    signal_names : sequence of str, optional
        Names for signal channels used to label dimensions. If None, uses indices.
    squeeze : bool, default=False
        If True and there are exactly 2 channels, reduce to the single ordered
        pair (first source, last target), returning a ``(time, frequency)``
        array. The selected ``source`` and ``target`` are retained as scalar
        coordinates, so the pair -- and, for directed measures, the direction --
        is still recorded. With more than 2 channels a warning is issued and the
        full matrix is returned.
    connectivity_kwargs : dict, optional
        Additional keyword arguments passed to connectivity methods.
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

    References
    ----------
    .. [1] Thomson, D. J. (1982). Spectrum estimation and harmonic analysis.
           Proceedings of the IEEE, 70(9), 1055-1096.
    .. [2] Percival, D. B., & Walden, A. T. (1993). Spectral Analysis for Physical
           Applications: Multitaper and Conventional Univariate Techniques.
    """
    if connectivity_kwargs is None:
        connectivity_kwargs = {}
    return_dataarray = False  # Default: return dataset
    if method is None:
        # The explicit NetCDF-safe / xarray-compatible default set (see
        # DEFAULT_METHODS). Not every Connectivity method — coherency (complex),
        # global_coherence / phase_slope_index, and the directed-transfer-function
        # family are excluded from the default (the last is still opt-in by name).
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
    # Accept the documented (n_times, n_channels) 2-D form by inserting a
    # singleton trial axis; Multitaper requires 3-D (n_times, n_trials,
    # n_signals).
    if getattr(time_series, "ndim", None) == 2:
        time_series = time_series[:, np.newaxis, :]
    m = Multitaper(
        time_series=time_series,
        sampling_frequency=sampling_frequency,
        time_window_duration=time_window_duration,
        **kwargs,
    )
    # Capture metadata and build the shared calculation object from the same
    # immutable transform. The private formatter below never accepts a separate
    # Multitaper, so data and labels cannot be paired accidentally.
    metadata = m._provenance_metadata()
    shared_connectivity = Connectivity.from_multitaper(m)
    cons = xr.Dataset()
    for this_method in method:
        try:
            con = _connectivity_result_to_xarray(
                shared_connectivity,
                metadata,
                this_method,
                signal_names,
                squeeze,
                **connectivity_kwargs,
            )
            cons[this_method] = con
        except (NotImplementedError, UnsupportedMeasureError) as e:
            # Structural incompatibility can be skipped in a batch. Other
            # computation errors are intentionally not caught.
            if len(method) == 1:
                raise
            logger.warning("Skipping %s: %s", this_method, e)
    if len(cons.data_vars) == 0:
        raise UnsupportedMeasureError(
            "None of the requested methods produced a compatible result "
            f"for the xarray interface: {method!r}."
        )
    if return_dataarray and method[0] in cons:
        return cons[method[0]]
    return cons
