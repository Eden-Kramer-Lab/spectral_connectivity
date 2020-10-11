from spectral_connectivity.connectivity import Connectivity
from spectral_connectivity.transforms import Multitaper
import xarray as xr
import numpy as np
from logging import getLogger

logger = getLogger(__name__)


def multitaper_connectivity(time_series, sampling_frequency, method='coherence_magnitude', time_window_duration=None,
                            signal_names=None, squeeze=False, **kwargs):
    """
    Transform time series to multitaper and calculate connectivity using `method`. Returns an xarray
    with dimensions of ['Time', 'Frequency', 'Source', 'Target']
    or ['Time', 'Frequency'] if squeeze=True

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
    signal_names : tuple
        Sames of time series used to name the 'Source' and 'Target' axes of xarray
    squeeze : bool
        Whether to only take the first and last source and target time series. Only makes sense for
        one pair of signals and symmetrical measures

    """

    m = Multitaper(time_series=time_series,
                   sampling_frequency=sampling_frequency,
                   time_window_duration=time_window_duration,
                   **kwargs)
    cc = Connectivity.from_multitaper(m)
    spectrogram = getattr(cc, method)()

    if (time_series.shape[-1] == 2) and squeeze:  # Only one couple (only makes sense for symmetrical metrics
        logger.warning(f'Squeeze is on, but there are {time_series.shape[-1]} pairs!')
        spectrogram = spectrogram[:, :, 0, -1]
        xar = xr.DataArray(spectrogram,
                           coords=[cc.time, cc.frequencies],
                           dims=['Time', 'Frequency'])

    else:  # Name the source and target axes
        if signal_names is None:
            signal_names = np.arange(time_series.shape[-1])
        xar = xr.DataArray(spectrogram,
                           coords=[cc.time, cc.frequencies, signal_names, signal_names],
                           dims=['Time', 'Frequency', 'Source', 'Target'])

    xar.name = method
    return xar
