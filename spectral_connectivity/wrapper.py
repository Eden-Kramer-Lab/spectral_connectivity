"""Functions for getting connectivity measures in a labeled array format."""

from logging import getLogger
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from spectral_connectivity.connectivity import Connectivity
from spectral_connectivity.transforms import Multitaper

logger = getLogger(__name__)


def connectivity_to_xarray(
    m: Multitaper,
    method: str = "coherence_magnitude",
    signal_names: Optional[Sequence[str]] = None,
    squeeze: bool = False,
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
    NotImplementedError
        If the requested method is not supported by xarray interface.

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
    if (method in ["group_delay", "canonical_coherence"]) or ("directed" in method):
        raise ValueError(
            f"The method '{method}' is not supported by the xarray interface. "
            f"Please use the Connectivity class directly instead:\n\n"
            f"from spectral_connectivity import Connectivity\n"
            f"conn = Connectivity.from_multitaper(m)\n"
            f"result = conn.{method}()\n"
        )
    # Name the source and target axes
    if signal_names is None:
        signal_names = np.arange(m.time_series.shape[-1])
    connectivity = Connectivity.from_multitaper(m)
    if method == "canonical_coherence":
        connectivity_mat, labels = getattr(connectivity, method)(**kwargs)
    else:
        connectivity_mat = getattr(connectivity, method)(**kwargs)
    # Only one couple (only makes sense for symmetrical metrics)
    if (m.time_series.shape[-1] > 2) and squeeze:
        logger.warning(f"Squeeze is on, but there are {m.time_series.shape[-1]} pairs!")

    if method == "power":
        xar = xr.DataArray(
            connectivity_mat,
            coords=[connectivity.time, connectivity.frequencies, signal_names],
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
                signal_names,
                signal_names,
            ],
            dims=["time", "frequency", "source", "target"],
        )

    xar.name = method

    for attr in dir(m):
        if (attr[0] == "_") or (
            attr in ["time_series", "fft", "tapers", "frequencies", "time"]
        ):
            continue
        # If we don't add 'mt_', get:
        # TypeError: '.dt' accessor only available for DataArray with
        # datetime64 timedelta64 dtype
        # or for arrays containing cftime datetime objects.
        xar.attrs["mt_" + attr] = getattr(m, attr)

    return xar


def multitaper_connectivity(
    time_series: NDArray[np.floating],
    sampling_frequency: float,
    time_window_duration: Optional[float] = None,
    method: Optional[Union[str, List[str]]] = None,
    signal_names: Optional[Sequence[str]] = None,
    squeeze: bool = False,
    connectivity_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Union[xr.DataArray, xr.Dataset]:
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
        Connectivity method(s) to compute. If None, computes all available methods.
        Examples: "coherence_magnitude", "imaginary_coherence", "phase_locking_value".
    signal_names : sequence of str, optional
        Names for signal channels used to label dimensions. If None, uses indices.
    squeeze : bool, default=False
        If True and n_channels=2, return connectivity between first and last
        channel only for symmetric measures.
    connectivity_kwargs : dict, optional
        Additional keyword arguments passed to connectivity methods.
    **kwargs : dict
        Additional arguments passed to Multitaper constructor
        (e.g., time_bandwidth_product,
        n_tapers, n_fft_samples).

    Returns
    -------
    result : xarray.DataArray or xarray.Dataset
        - DataArray if single method requested: connectivity values with dimensions
          ['time', 'frequency', 'source', 'target'] or ['time', 'frequency'] if squeezed
        - Dataset if multiple methods: collection of DataArrays, one per method

    Examples
    --------
    >>> import numpy as np
    >>> # Generate coupled oscillator data
    >>> t = np.arange(0, 1, 1/500)  # 500 Hz, 1 second
    >>> sig1 = np.sin(2*np.pi*10*t) + 0.1*np.random.randn(len(t))
    >>> sig2 = np.sin(2*np.pi*10*t + np.pi/4) + 0.1*np.random.randn(len(t))
    >>> data = np.column_stack([sig1, sig2])  # Shape: (500, 2)
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
        # All implemented methods except internal
        # TODO is there a better way to get all Connectivity methods?
        bad_methods = [
            "delay",
            "n_observations",
            "frequencies",
            "all_frequencies",
            "global_coherence",
            "from_multitaper",
            "phase_slope_index",
            "subset_pairwise_spectral_granger_prediction",
            # Methods not supported by xarray interface
            "group_delay",
            "canonical_coherence",
            "directed_transfer_function",
            "directed_coherence",
            "partial_directed_coherence",
            "generalized_partial_directed_coherence",
            "direct_directed_transfer_function",
            "blockwise_spectral_granger_prediction",
        ]
        method = [
            x
            for x in dir(Connectivity)
            if not x.startswith("_") and x not in bad_methods
        ]
    elif isinstance(method, str):
        method = [method]  # Convert to list
        return_dataarray = True  # Return dataarray if methods was not an iterable
    m = Multitaper(
        time_series=time_series,
        sampling_frequency=sampling_frequency,
        time_window_duration=time_window_duration,
        **kwargs,
    )
    cons = xr.Dataset()  # Initialize
    for this_method in method:
        try:
            con = connectivity_to_xarray(
                m, this_method, signal_names, squeeze, **connectivity_kwargs
            )
            cons[this_method] = con  # Add data variable
        except NotImplementedError as e:
            if len(method) == 1:
                raise e  # If that was the only method requested
            else:
                # If one measure among many, just warn
                logger.warning(f"{this_method} is not implemented in xarray")
    if return_dataarray and method[0] in cons:
        return cons[method[0]]
    else:
        return cons
