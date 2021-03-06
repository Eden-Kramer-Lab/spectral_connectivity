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
    if ((method in ['group_delay', 'canonical_coherence']) or
            ('directed' in method)):
        raise NotImplementedError(
            f'{method} is not supported by xarray interface')
    # Name the source and target axes
    if signal_names is None:
        signal_names = np.arange(m.time_series.shape[-1])
    connectivity = Connectivity.from_multitaper(m)
    if method == 'canonical_coherence':
        connectivity_mat, labels = getattr(connectivity, method)(**kwargs)
    else:
        connectivity_mat = getattr(connectivity, method)(**kwargs)
    # Only one couple (only makes sense for symmetrical metrics)
    if (m.time_series.shape[-1] > 2) and squeeze:
        logger.warning(
            f'Squeeze is on, but there are {m.time_series.shape[-1]} pairs!')

    if method == 'power':
        xar = xr.DataArray(connectivity_mat,
                           coords=[connectivity.time,
                                   connectivity.frequencies,
                                   signal_names],
                           dims=['Time', 'Frequency', 'Source'])

    elif (m.time_series.shape[-1] == 2) and squeeze:
        connectivity_mat = connectivity_mat[..., 0, -1]
        xar = xr.DataArray(connectivity_mat,
                           coords=[connectivity.time,
                                   connectivity.frequencies],
                           dims=['Time', 'Frequency'])

    else:

        xar = xr.DataArray(connectivity_mat,
                           coords=[connectivity.time, connectivity.frequencies,
                                   signal_names, signal_names],
                           dims=['Time', 'Frequency', 'Source', 'Target'])

    xar.name = method

    for attr in dir(m):
        if (attr[0] == '_') or (attr in ['time_series', 'fft', 'tapers', 'frequencies', 'time']):
            continue
        # If we don't add 'mt_', get:
        # TypeError: '.dt' accessor only available for DataArray with
        # datetime64 timedelta64 dtype
        # or for arrays containing cftime datetime objects.
        xar.attrs['mt_' + attr] = getattr(m, attr)

    return xar


def multitaper_connectivity(time_series, sampling_frequency,
                            time_window_duration=None,
                            method=None, signal_names=None,
                            squeeze=False, connectivity_kwargs=None, **kwargs):
    """
    Transform time series to multitaper and
    calculate connectivity using `method`. Returns an xarray.DataSet
    with dimensions of ['Time', 'Frequency', 'Source', 'Target']
    or ['Time', 'Frequency'] if squeeze=True.
    Its Data variables are measures

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
    method : iterable of strings, optional
        Method used for connectivity calculation. If None, all available
        measures are calculated
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

    Returns
    --------
    connectivities : Xarray.Dataset with connectivity measure(s) as data variables



    """
    if connectivity_kwargs is None:
        connectivity_kwargs = {}
    return_dataarray = False  # Default: return dataset
    if method is None:
        # All implemented methods except internal
        # TODO is there a better way to get all Connectivity methods?
        bad_methods = ['delay', 'n_observations', 'frequencies',
                       'from_multitaper', 'phase_slope_index']
        method = [x for x in dir(Connectivity) if not x.startswith(
            '_') and x not in bad_methods]
    elif type(method) == str:
        method = [method]  # Convert to list
        return_dataarray = True  # Return dataarray if methods was not an iterable
    m = Multitaper(time_series=time_series,
                   sampling_frequency=sampling_frequency,
                   time_window_duration=time_window_duration,
                   **kwargs)
    cons = xr.Dataset()  # Initialize
    for this_method in method:
        try:
            con = connectivity_to_xarray(m, this_method, signal_names, squeeze,
                                         **connectivity_kwargs)
            cons[this_method] = con  # Add data variable
        except NotImplementedError as e:
            if len(method) == 1:
                raise e  # If that was the only method requested
            else:
                # If one measure among many, just warn
                logger.warning(f'{this_method} is not implemented in xarray')
    if return_dataarray and method[0] in cons:
        return cons[method[0]]
    else:
        return cons
