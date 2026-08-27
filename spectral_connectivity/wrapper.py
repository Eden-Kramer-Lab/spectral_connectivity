"""Functions for getting connectivity measures in a labeled array format."""

import warnings
from collections.abc import Sequence
from logging import getLogger
from typing import Any

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
    ``phase_slope_index``, ``group_delay``, ``canonical_coherence``, the directed
    family). It subclasses ``ValueError`` for backward compatibility, but is a
    distinct type so :func:`multitaper_connectivity` can skip an unsupported
    measure in a multi-measure batch *without* also swallowing a genuine
    computation error (e.g. a measure that raises ``ValueError`` because the data
    has too few observations).
    """


def _package_version() -> str:
    """Return the installed spectral_connectivity version (or ``"unknown"``)."""
    try:
        from importlib.metadata import version

        return version("spectral_connectivity")
    except Exception:
        return "unknown"


# Default measures for ``multitaper_connectivity(method=None)``: the real-valued
# connectivity measures that fit the xarray/NetCDF interface. Defined explicitly
# (rather than discovered by excluding a denylist) so the default set is stable
# and documentable, and so a newly added Connectivity method cannot silently join
# the default and break NetCDF serialization. Excluded on purpose: ``coherency``
# (complex; NetCDF cannot store it — its content is covered by
# coherence_magnitude/phase + imaginary_coherence), ``global_coherence`` and
# ``phase_slope_index`` (do not fit the per-signal-pair xarray layout), the
# directed transfer-function family and canonical/group-delay/conditional
# measures (different output shapes), and utility properties.
# Order is significant: xarray preserves insertion order, so this fixes the
# variable/iteration/serialization order of the default Dataset. It is kept
# alphabetical to match the order the previous ``inspect.getmembers`` discovery
# produced, so the default result is unchanged for existing users.
DEFAULT_METHODS: tuple[str, ...] = (
    "coherence_magnitude",
    "coherence_phase",
    "debiased_squared_phase_lag_index",
    "debiased_squared_weighted_phase_lag_index",
    "imaginary_coherence",
    "pairwise_phase_consistency",
    "pairwise_spectral_granger_prediction",
    "phase_lag_index",
    "phase_locking_value",
    "power",
    "weighted_phase_lag_index",
)


def _to_host_array(x: Any) -> NDArray:
    """Return ``x`` as a host NumPy array, moving it off the GPU if needed.

    Under GPU mode ``Multitaper`` coordinates are CuPy arrays, which NumPy will
    not implicitly convert. CuPy arrays expose ``.get()`` to copy to host (so the
    GPU case returns a copy, not a view); NumPy arrays have no such method and
    pass through unchanged.
    """
    to_host = getattr(x, "get", None)
    return np.asarray(to_host() if callable(to_host) else x)


def _validate_connectivity_matches_multitaper(
    connectivity: Connectivity, m: Multitaper
) -> None:
    """Raise if an injected ``Connectivity`` was not built from ``m``.

    ``connectivity_to_xarray`` takes the data and coordinates from
    ``connectivity`` but the metadata attributes from ``m``. If the two describe
    different transforms the result would be silently mislabeled — real values
    from one recording tagged with another's parameters.

    Matching geometry (channel count, frequency grid, time bins) is *necessary
    but not sufficient*: two different recordings with the same sampling
    frequency, window, and channel count share it, so geometry alone cannot
    establish provenance. Both are therefore required — geometry is validated
    first (it also catches a from_multitaper instance whose public ``time`` /
    ``frequencies`` were mutated after construction), and then provenance is
    verified by identity: ``Connectivity.from_multitaper`` records a weakref to
    its source transform, and only an instance whose recorded source is ``m`` is
    accepted.

    Also require the default ``expectation_type`` ("trials_tapers"): the xarray
    layout below assumes the result keeps the time and frequency axes, which
    other expectation types do not (they average time or keep the trial/taper
    axes), so they would not fit the fixed (time, frequency, source, target)
    dimensions.
    """
    # Always validate geometry first (a necessary condition): this catches a
    # mismatched instance and also post-construction mutation of the public
    # ``time`` / ``frequencies`` coordinates on an otherwise-from_multitaper
    # instance, which the identity check below would not.
    # Read the private snapshot for the shape: the public getter returns a
    # defensive copy on backends without a writeable flag (CuPy), wasteful here
    # and unnecessary for shape.
    n_signals = connectivity._fourier_coefficients.shape[-1]
    mismatches = []
    if n_signals != m.n_signals:
        mismatches.append(f"n_signals ({n_signals} != {m.n_signals})")
    if not np.array_equal(
        _to_host_array(connectivity.all_frequencies), _to_host_array(m.frequencies)
    ):
        mismatches.append("frequencies")
    if not np.array_equal(_to_host_array(connectivity.time), _to_host_array(m.time)):
        mismatches.append("time")
    if mismatches:
        raise ValueError(
            "The provided `connectivity` was not built from this "
            f"`Multitaper`; they disagree on: {', '.join(mismatches)}. "
            "`connectivity_to_xarray` labels results with `m`'s coordinates "
            "and metadata, so a mismatched instance would produce silently "
            "mislabeled output. Pass a `Connectivity` built from `m` (e.g. "
            "`Connectivity.from_multitaper(m)`) or leave `connectivity=None` "
            "to build one automatically."
        )
    # Geometry alone cannot prove provenance (two recordings can share it), so
    # additionally require an identity link to m.
    source = connectivity._source_multitaper
    if source is None or source() is not m:
        raise ValueError(
            "The provided `connectivity` cannot be verified to come from this "
            "`Multitaper`: it was not built by `Connectivity.from_multitaper(m)` "
            "(or its coefficients were reassigned afterwards). A matching channel "
            "count, frequency grid, and time bins do NOT prove it holds the same "
            "data — two different recordings can share them — and "
            "`connectivity_to_xarray` labels the result with `m`'s coordinates "
            "and metadata, which would silently mislabel it. Build it with "
            "`Connectivity.from_multitaper(m)`, or leave `connectivity=None` to "
            "build one automatically."
        )
    if connectivity.expectation_type != "trials_tapers":
        raise ValueError(
            "connectivity_to_xarray supports only expectation_type="
            "'trials_tapers' (the default), which yields a result over "
            "(time, frequency, source, target); got expectation_type="
            f"'{connectivity.expectation_type}', which averages or keeps "
            "different axes and does not fit that xarray layout. Use the "
            "`Connectivity` class directly for other expectation types."
        )


def connectivity_to_xarray(
    m: Multitaper,
    method: str = "coherence_magnitude",
    signal_names: Sequence[str] | None = None,
    squeeze: bool = False,
    connectivity: Connectivity | None = None,
    **kwargs: Any,
) -> xr.DataArray:
    """
    Calculate connectivity measures and return as labeled xarray.

    Computes the specified connectivity measure from multitaper spectral analysis
    and returns results in an xarray.DataArray with properly labeled dimensions.

    Parameters
    ----------
    m : Multitaper
        Multitaper object containing spectral transform results.
    method : str, default="coherence_magnitude"
        Name of connectivity method to compute (e.g., "coherence_magnitude",
        "imaginary_coherence", "phase_locking_value").
    signal_names : sequence of str, optional
        Names for signal channels used to label 'source' and 'target' dimensions.
        If None, uses integer indices.
    squeeze : bool, default=False
        If True and only 2 signals, return connectivity between first and last
        signal only. Only meaningful for symmetric measures.
    connectivity : Connectivity, optional
        A ``Connectivity`` built from ``m`` via ``Connectivity.from_multitaper(m)``.
        When computing several measures from the same transform, pass a shared
        instance to avoid recomputing the (uncached) FFT for each measure and to
        reuse the cached power / cross-spectrum across the coherence-family
        measures. When ``None`` (the default) one is constructed from ``m``
        automatically. It **must** be built from ``m`` — ``m`` supplies the
        output coordinates/metadata, so a mismatched ``connectivity`` would
        mislabel the result. This is enforced by identity: ``from_multitaper``
        records its source transform, and only an instance whose recorded source
        is ``m`` (with the default ``expectation_type``) is accepted. A directly
        constructed ``Connectivity``, or one whose coefficients were reassigned,
        cannot be verified and is rejected (matching geometry alone does not
        prove the data is the same).
    **kwargs : dict
        Additional keyword arguments passed to connectivity method.

    Returns
    -------
    connectivity : xarray.DataArray
        Connectivity results with dimensions:
        - ['time', 'frequency', 'source', 'target'] for pairwise measures
        - ['time', 'frequency', 'source'] for power spectral density
        - ['time', 'frequency'] if squeeze=True and n_signals=2

    Raises
    ------
    ValueError
        In either of two cases: (1) the requested method does not fit the
        ``(time, frequency, source, target)`` xarray layout
        (``global_coherence``, ``phase_slope_index``, ``group_delay``,
        ``canonical_coherence``, or a directed measure); the message points to
        using ``Connectivity`` directly. (2) a ``connectivity`` instance is
        passed whose channel count, frequency grid, or time bins disagree with
        ``m`` (it must have been built from ``m``); the message names the
        disagreeing fields.

    Examples
    --------
    >>> import numpy as np
    >>> from spectral_connectivity.transforms import Multitaper
    >>> # Simulate data: (100 time points, 5 trials, 3 channels)
    >>> data = np.random.randn(100, 5, 3)
    >>> mt = Multitaper(data, sampling_frequency=1000)
    >>> coherence = connectivity_to_xarray(mt, method="coherence_magnitude")
    >>> coherence.dims
    ('time', 'frequency', 'source', 'target')
    """
    if (
        method
        in [
            "group_delay",
            "canonical_coherence",
            "global_coherence",
            "phase_slope_index",
        ]
    ) or ("directed" in method):
        raise UnsupportedMeasureError(
            f"The method '{method}' is not supported by the xarray interface "
            f"(it does not return a plain (time, frequency, source, target) "
            f"array). Please use the Connectivity class directly instead:\n\n"
            f"from spectral_connectivity import Connectivity\n"
            f"conn = Connectivity.from_multitaper(m)\n"
            f"result = conn.{method}()\n"
        )
    # Name the source and target axes
    signal_names_list: Sequence[str]
    if signal_names is None:
        signal_names_list = list(np.arange(m.time_series.shape[-1]).astype(str))
    else:
        signal_names_list = signal_names

    if connectivity is None:
        connectivity = Connectivity.from_multitaper(m)
    else:
        _validate_connectivity_matches_multitaper(connectivity, m)
    connectivity_mat = getattr(connectivity, method)(**kwargs)
    # Only one couple (only makes sense for symmetrical metrics)
    if (m.time_series.shape[-1] > 2) and squeeze:
        warnings.warn(
            f"squeeze=True but there are {m.time_series.shape[-1]} signals "
            f"(more than 2 pairs); ignoring squeeze and returning the full "
            f"(source, target) matrix.",
            UserWarning,
            stacklevel=2,
        )

    if method == "power":
        xar = xr.DataArray(
            connectivity_mat,
            coords=[connectivity.time, connectivity.frequencies, signal_names_list],
            dims=["time", "frequency", "source"],
        )

    elif (m.time_series.shape[-1] == 2) and squeeze:
        connectivity_mat = connectivity_mat[..., 0, -1]
        xar = xr.DataArray(
            connectivity_mat,
            coords=[connectivity.time, connectivity.frequencies],
            dims=["time", "frequency"],
        )

    else:
        xar = xr.DataArray(
            connectivity_mat,
            coords=[
                connectivity.time,
                connectivity.frequencies,
                signal_names_list,
                signal_names_list,
            ],
            dims=["time", "frequency", "source", "target"],
        )

    xar.name = method

    for attr in dir(m):
        if (attr[0] == "_") or (
            attr in ["time_series", "fft", "tapers", "frequencies", "time"]
        ):
            continue
        value = getattr(m, attr)
        # NetCDF attributes must be strings, numbers, or (non-complex) numeric
        # arrays. Skip callables (e.g. the bound ``summarize_parameters``
        # method); encode None (e.g. ``detrend_type=None``) as a string so the
        # parameter is still recorded; and skip any other unsupported type.
        # Storing an unsupported value would make ``to_netcdf`` raise.
        if callable(value):
            continue
        if value is None:
            value = "None"
        elif isinstance(value, np.ndarray):
            if value.dtype.kind not in "biufSU":  # exclude complex/object arrays
                continue
        elif not isinstance(
            value, (str, bytes, bool, int, float, np.integer, np.floating, np.bool_)
        ):
            continue
        # If we don't add 'mt_', get:
        # TypeError: '.dt' accessor only available for DataArray with
        # datetime64 timedelta64 dtype
        # or for arrays containing cftime datetime objects.
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
        requested by name. Other measures that do not fit the ``(time, frequency,
        source, target)`` layout — ``global_coherence``, ``phase_slope_index``,
        ``group_delay``, ``canonical_coherence``, and the directed-transfer-
        function family — are *not* available through this wrapper at all
        (requesting one raises with a pointer to use ``Connectivity`` directly).
        Examples: "coherence_magnitude", "imaginary_coherence",
        "phase_locking_value".
    signal_names : sequence of str, optional
        Names for signal channels used to label dimensions. If None, uses indices.
    squeeze : bool, default=False
        If True and n_channels=2, return connectivity between first and last
        channel only for symmetric measures.
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
        # family are excluded.
        method = list(DEFAULT_METHODS)
    elif isinstance(method, str):
        method = [method]  # Convert to list
        return_dataarray = True  # Return dataarray if methods was not an iterable
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
    # Build the Connectivity once and share it across every requested measure:
    # from_multitaper recomputes the (uncached) FFT on each call, and a fresh
    # instance would also discard cached intermediates. connectivity_kwargs are
    # passed to the measure methods, not the constructor, so a single
    # default-constructed instance matches the previous per-method construction.
    shared_connectivity = Connectivity.from_multitaper(m)
    cons = xr.Dataset()  # Initialize
    for this_method in method:
        try:
            con = connectivity_to_xarray(
                m,
                this_method,
                signal_names,
                squeeze,
                connectivity=shared_connectivity,
                **connectivity_kwargs,
            )
            cons[this_method] = con  # Add data variable
        except (NotImplementedError, UnsupportedMeasureError) as e:
            # Skip ONLY measures that structurally do not fit the xarray layout
            # (UnsupportedMeasureError) or are not implemented. A genuine
            # computation error — e.g. a debiased measure raising ValueError
            # because the data has too few observations — is deliberately NOT
            # caught here, so it surfaces instead of silently dropping a measure
            # the user asked for. When only one measure was requested, re-raise
            # even the structural case so nothing is swallowed.
            if len(method) == 1:
                raise e  # If that was the only method requested
            else:
                logger.warning(f"Skipping {this_method}: {e}")
    if return_dataarray and method[0] in cons:
        return cons[method[0]]
    else:
        return cons
