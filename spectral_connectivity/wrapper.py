from logging import getLogger

import numpy as np
import xarray as xr
from spectral_connectivity.connectivity import Connectivity
from spectral_connectivity.transforms import Multitaper

logger = getLogger(__name__)


def connectivity_to_xarray(m, method='coherence_magnitude', signal_names=None,
                           squeeze=False, **kwargs):
    """
    calculate connectivity using `method`. Returns an xarray
    with dimensions of ['Time', 'Frequency', 'Source', 'Target']
    or ['Time', 'Frequency'] if squeeze=True

    Parameters
    -----------
    signal_names : iterable of strings
        Sames of time series used to name the 'Source' and 'Target' axes of
        xarray.
    squeeze : bool
        Whether to only take the first and last source and target time series.
        Only makes sense for one pair of signals and symmetrical measures

    """
    assert method not in ['power', 'group_delay', 'canonical_coherence'], \
        f'{method} is not supported by xarray interface'
    connectivity = Connectivity.from_multitaper(m)
    if method == 'canonical_coherence':
        connectivity_mat, labels = getattr(connectivity, method)(**kwargs)
    else:
        connectivity_mat = getattr(connectivity, method)(**kwargs)
    # Only one couple (only makes sense for symmetrical metrics)
    if (m.time_series.shape[-1] == 2) and squeeze:
        logger.warning(
            f'Squeeze is on, but there are {m.time_series.shape[-1]} pairs!')
        connectivity_mat = connectivity_mat[:, :, 0, -1]
        xar = xr.DataArray(connectivity_mat,
                           coords=[connectivity.time,
                                   connectivity.frequencies],
                           dims=['Time', 'Frequency'])

    else:  # Name the source and target axes
        if signal_names is None:
            signal_names = np.arange(m.time_series.shape[-1])

        xar = xr.DataArray(connectivity_mat,
                           coords=[connectivity.time, connectivity.frequencies,
                                   signal_names, signal_names],
                           dims=['Time', 'Frequency', 'Source', 'Target'])

    xar.name = method

    for attr in dir(m):
        if (attr[0] == '_') or (attr == 'time_series'):
            continue
        # If we don't add 'mt_', get:
        # TypeError: '.dt' accessor only available for DataArray with
        # datetime64 timedelta64 dtype
        # or for arrays containing cftime datetime objects.
        xar.attrs['mt_' + attr] = getattr(m, attr)

    return xar


def multitaper_connectivity(time_series, sampling_frequency,
                            time_window_duration=None,
                            method='coherence_magnitude', signal_names=None,
                            squeeze=False, connectivity_kwargs=None, **kwargs):
    """
    Transform time series to multitaper and
    calculate connectivity using `method`. Returns an xarray.DataArray
    with dimensions of ['Time', 'Frequency', 'Source', 'Target']
    or ['Time', 'Frequency'] if squeeze=True

    Parameters
    -----------
    signal_names : iterable of strings
        Sames of time series used to name the 'Source' and 'Target' axes of
        xarray.
    squeeze : bool
        Whether to only take the first and last source and target time series.
        Only makes sense for one pair of signals and symmetrical measures.

    Attributes
    ----------
    time_series : array, shape (n_time_samples, n_trials, n_signals) or
                               (n_time_samples, n_signals)
    sampling_frequency : float
        Number of samples per time unit the signal(s) are recorded at.
    method : str
        Method used for connectivity calculation
    time_window_duration : float, optional
        Duration of sliding window in which to compute the fft. Defaults to
        the entire time if not set.
    signal_names : iterable of strings
        Sames of time series used to name the 'Source' and 'Target' axes of
        xarray.
    squeeze : bool
        Whether to only take the first and last source and target time series.
        Only makes sense for one pair of signals and symmetrical measures.
    connectivity_kwargs : dict
        Arguments to pass to connectivity calculation


    """
    if connectivity_kwargs is None:
        connectivity_kwargs = {}
    m = Multitaper(time_series=time_series,
                   sampling_frequency=sampling_frequency,
                   time_window_duration=time_window_duration,
                   **kwargs)
    return connectivity_to_xarray(m, method, signal_names, squeeze,
                                  **connectivity_kwargs)
