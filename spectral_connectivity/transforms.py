from logging import getLogger

import numpy as np
from numpy.fft import fftfreq
from scipy import interpolate
from scipy.fftpack import fft, ifft, next_fast_len
from scipy.linalg import eigvals_banded
from scipy.signal import detrend

logger = getLogger(__name__)


class Multitaper(object):
    '''Transform time-domain signal(s) to the frequency domain by using
    multiple tapering windows.

    Attributes
    ----------
    time_series : array, shape (n_time_samples, n_trials, n_signals) or
                               (n_time_samples, n_signals)
    sampling_frequency : float, optional
        Number of samples per time unit the signal(s) are recorded at.
    time_halfbandwidth_product : float, optional
        Specifies the time-frequency tradeoff of the tapers and also the number
        of tapers if `n_tapers` is not set.
    detrend_type : string or None, optional
        Subtracting a constant or a linear trend from each time window. If None
        then no detrending is done.
    start_time : float, optional
        Start time of time series.
    time_window_duration : float, optional
        Duration of sliding window in which to compute the fft. Defaults to
        the entire time if not set.
    time_window_step : float, optional
        Duration of time to skip when moving the window forward. By default,
        this equals the duration of the time window.
    tapers : array, optional, shape (n_time_samples_per_window, n_tapers)
        Pass in a pre-computed set of tapers. If `None`, then the tapers are
        automically calulated based on the `time_halfbandwidth_product`,
        `n_tapers`, and `n_time_samples_per_window`.
    n_tapers : int, optional
        Set the number of tapers. If `None`, the number of tapers is computed
        by 2 * `time_halfbandwidth_product` - 1.
    n_time_samples_per_window : int, optional
        Number of samples in each sliding window. If `time_window_duration` is
        set, then this is calculated automically.
    n_time_samples_per_step : int, optional
        Number of samples to skip when moving the window forward. If
        `time_window_step` is set, then this is calculated automically.
    is_low_bias : bool, optional
        If `True`, excludes tapers with eigenvalues < 0.9

    '''

    def __init__(self, time_series, sampling_frequency=1000,
                 time_halfbandwidth_product=3,
                 detrend_type='constant', time_window_duration=None,
                 time_window_step=None, n_tapers=None,  tapers=None,
                 start_time=0, n_fft_samples=None,
                 n_time_samples_per_window=None,
                 n_time_samples_per_step=None, is_low_bias=True):

        self.time_series = time_series
        self.sampling_frequency = sampling_frequency
        self.time_halfbandwidth_product = time_halfbandwidth_product
        self.detrend_type = detrend_type
        self._time_window_duration = time_window_duration
        self._time_window_step = time_window_step
        self.is_low_bias = is_low_bias
        self.start_time = start_time
        self._n_fft_samples = n_fft_samples
        self._tapers = tapers
        self._n_tapers = n_tapers
        self._n_time_samples_per_window = n_time_samples_per_window
        self._n_samples_per_time_step = n_time_samples_per_step

    def __repr__(self):
        return (
            'Multitaper('
            'sampling_frequency={0.sampling_frequency!r}, '
            'time_halfbandwidth_product={0.time_halfbandwidth_product!r},\n'
            '           time_window_duration={0.time_window_duration!r}, '
            'time_window_step={0.time_window_step!r},\n'
            '           detrend_type={0.detrend_type!r}, '
            'start_time={0.start_time}, '
            'n_tapers={0.n_tapers}'
            ')'.format(self))

    @property
    def tapers(self):
        '''

        Returns
        -------
        tapers : array_like, shape (n_time_samples_per_window, n_tapers)

        '''
        if self._tapers is None:
            self._tapers = _make_tapers(
                self.n_time_samples_per_window, self.sampling_frequency,
                self.time_halfbandwidth_product, self.n_tapers,
                is_low_bias=self.is_low_bias)
        return self._tapers

    @property
    def time_window_duration(self):
        if self._time_window_duration is None:
            self._time_window_duration = (self.n_time_samples_per_window /
                                          self.sampling_frequency)
        return self._time_window_duration

    @property
    def time_window_step(self):
        if self._time_window_step is None:
            self._time_window_step = (self.n_time_samples_per_step /
                                      self.sampling_frequency)
        return self._time_window_step

    @property
    def n_tapers(self):
        '''Number of desired tapers.

        Note that the number of tapers may be less than this number if
        the bias of the tapers is too high (eigenvalues > 0.9)

        '''
        if self._n_tapers is None:
            return int(np.floor(
                2 * self.time_halfbandwidth_product - 1))
        return self._n_tapers

    @property
    def n_time_samples_per_window(self):
        if (self._n_time_samples_per_window is None and
                self._time_window_duration is None):
            self._n_time_samples_per_window = self.time_series.shape[0]
        elif self._time_window_duration is not None:
            self._n_time_samples_per_window = int(np.round(
                self.time_window_duration * self.sampling_frequency))
        return self._n_time_samples_per_window

    @property
    def n_fft_samples(self):
        if self._n_fft_samples is None:
            self._n_fft_samples = next_fast_len(
                self.n_time_samples_per_window)
        return self._n_fft_samples

    @property
    def frequencies(self):
        return fftfreq(self.n_fft_samples, 1.0 / self.sampling_frequency)

    @property
    def n_time_samples_per_step(self):
        '''If `time_window_step` is set, then calculate the
        `n_time_samples_per_step` based on the time window duration. If
        `time_window_step` and `n_time_samples_per_step` are both not set,
        default the window step size to the same size as the window.
        '''
        if (self._n_samples_per_time_step is None and
                self._time_window_step is None):
            self._n_samples_per_time_step = self.n_time_samples_per_window
        elif self._time_window_step is not None:
            self._n_samples_per_time_step = int(
                self.time_window_step * self.sampling_frequency)
        return self._n_samples_per_time_step

    @property
    def time(self):
        original_time = (np.arange(0, self.time_series.shape[0]) /
                         self.sampling_frequency)
        window_start_time = _sliding_window(
            original_time, self.n_time_samples_per_window,
            self.n_time_samples_per_step)[:, 0]
        return self.start_time + window_start_time

    @property
    def n_signals(self):
        return (1 if len(self.time_series.shape) < 2 else
                self.time_series.shape[-1])

    @property
    def n_trials(self):
        return (1 if len(self.time_series.shape) < 3 else
                self.time_series.shape[1])

    @property
    def frequency_resolution(self):
        return (2.0 * self.time_halfbandwidth_product /
                self.time_window_duration)

    @property
    def nyquist_frequency(self):
        return self.sampling_frequency / 2

    def fft(self):
        '''Compute the fast Fourier transform using the multitaper method.

        Returns
        -------
        fourier_coefficients : array, shape (n_time_windows, n_trials,
                                             n_tapers, n_fft_samples,
                                             n_signals)

        '''
        time_series = _add_axes(self.time_series)
        time_series = _sliding_window(
            time_series, window_size=self.n_time_samples_per_window,
            step_size=self.n_time_samples_per_step, axis=0)
        if self.detrend_type is not None:
            time_series = detrend(time_series, type=self.detrend_type)

        logger.info(self)

        return _multitaper_fft(
            self.tapers, time_series, self.n_fft_samples,
            self.sampling_frequency).swapaxes(2, -1)


def _add_axes(time_series):
    '''If no trial or signal axes included, add one in.
    '''
    n_axes = len(time_series.shape)
    if n_axes == 1:  # add trials and signals axes
        return time_series[:, np.newaxis, np.newaxis]
    elif n_axes == 2:  # add trials axis
        return time_series[:, np.newaxis, ...]
    else:
        return time_series


def _sliding_window(data, window_size, step_size=1,
                    axis=-1, is_copy=True):
    '''
    Calculate a sliding window over a signal

    Parameters
    ----------
    data : numpy array
        The array to be slided over.
    window_size : int
        Number of samples per window
    step_size : int
        Number of samples to step the window forward. Defaults to 1.
    axis : int
        The axis to slide over. Defaults to the last axis.
    is_copy : bool
        Return strided array as copy to avoid sideffects when manipulating
        the output array.

    Returns
    -------
    data : array-like
        A matrix where row in last dimension consists of one instance
        of the sliding window.

    Notes
    -----
    - Be wary of setting `copy` to `False` as undesired sideffects with the
      output values may occurr.

    Examples
    --------
    >>> a = numpy.array([1, 2, 3, 4, 5])
    >>> _sliding_window(a, window_size=3)
    array([[1, 2, 3],
           [2, 3, 4],
           [3, 4, 5]])
    >>> _sliding_window(a, window_size=3, step_size=2)
    array([[1, 2, 3],
           [3, 4, 5]])

    References
    ----------
    .. [1] https://gist.github.com/nils-werner/9d321441006b112a4b116a8387c2
    280c

    '''
    shape = list(data.shape)
    shape[axis] = np.floor(
        (data.shape[axis] / step_size) - (window_size / step_size) + 1
    ).astype(int)
    shape.append(window_size)

    strides = list(data.strides)
    strides[axis] *= step_size
    strides.append(data.strides[axis])

    strided = np.lib.stride_tricks.as_strided(
        data, shape=shape, strides=strides)

    return strided.copy() if is_copy else strided


def _multitaper_fft(tapers, time_series, n_fft_samples,
                    sampling_frequency, axis=-2):
    '''Projects the data on the tapers and returns the discrete Fourier
    transform

    Parameters
    ----------
    tapers : array_like, shape (n_time_samples_per_window, n_tapers)
    time_series : array_like, shape (n_windows, n_trials,
                                     n_time_samples_per_window)
    n_fft_samples : int
    sampling_frequency : int

    Returns
    -------
    fourier_coefficients : array_like, shape (n_windows, n_trials, n_tapers
                                              n_fft_samples, n_signals)

    '''
    projected_time_series = (time_series[..., np.newaxis] *
                             tapers[np.newaxis, np.newaxis, ...])
    return (fft(projected_time_series, n=n_fft_samples, axis=axis) /
            sampling_frequency)


def _make_tapers(n_time_samples_per_window, sampling_frequency,
                 time_halfbandwidth_product, n_tapers, is_low_bias=True):
    '''Returns the Discrete prolate spheroidal sequences (tapers) for
    multi-taper spectral analysis.

    Parameters
    ----------
    n_time_samples_per_window : int
    sampling_frequency : int
    time_halfbandwidth_product : float
    n_tapers : int
    is_low_bias : bool
        Keep only tapers with eigenvalues > 0.9

    Returns
    -------
    tapers : array_like, shape (n_time_samples_per_window, n_tapers)

    '''
    tapers, _ = dpss_windows(
        n_time_samples_per_window, time_halfbandwidth_product, n_tapers,
        is_low_bias=is_low_bias)
    return tapers.T * np.sqrt(sampling_frequency)


def tridisolve(d, e, b, overwrite_b=True):
    '''Symmetric tridiagonal system solver, from Golub and Van Loan p157.
    .. note:: Copied from NiTime.
    Parameters
    ----------
    d : ndarray
      main diagonal stored in d[:]
    e : ndarray
      superdiagonal stored in e[:-1]
    b : ndarray
      RHS vector
    Returns
    -------
    x : ndarray
      Solution to Ax = b (if overwrite_b is False). Otherwise solution is
      stored in previous RHS vector b
    '''
    N = len(b)
    # work vectors
    dw = d.copy()
    ew = e.copy()
    if overwrite_b:
        x = b
    else:
        x = b.copy()
    for k in range(1, N):
        # e^(k-1) = e(k-1) / d(k-1)
        # d(k) = d(k) - e^(k-1)e(k-1) / d(k-1)
        t = ew[k - 1]
        ew[k - 1] = t / dw[k - 1]
        dw[k] = dw[k] - t * ew[k - 1]
    for k in range(1, N):
        x[k] = x[k] - ew[k - 1] * x[k - 1]
    x[N - 1] = x[N - 1] / dw[N - 1]
    for k in range(N - 2, -1, -1):
        x[k] = x[k] / dw[k] - ew[k] * x[k + 1]

    if not overwrite_b:
        return x


def tridi_inverse_iteration(d, e, w, x0=None, rtol=1e-8):
    '''Perform an inverse iteration.
    This will find the eigenvector corresponding to the given eigenvalue
    in a symmetric tridiagonal system.
    ..note:: Copied from NiTime.
    Parameters
    ----------
    d : array
      main diagonal of the tridiagonal system
    e : array
      offdiagonal stored in e[:-1]
    w : float
      eigenvalue of the eigenvector
    x0 : array
      initial point to start the iteration
    rtol : float
      tolerance for the norm of the difference of iterates
    Returns
    -------
    e: array
      The converged eigenvector
    '''
    eig_diag = d - w
    if x0 is None:
        x0 = np.random.randn(len(d))
    x_prev = np.zeros_like(x0)
    norm_x = np.linalg.norm(x0)
    # the eigenvector is unique up to sign change, so iterate
    # until || |x^(n)| - |x^(n-1)| ||^2 < rtol
    x0 /= norm_x
    while np.linalg.norm(np.abs(x0) - np.abs(x_prev)) > rtol:
        x_prev = x0.copy()
        tridisolve(eig_diag, e, x0)
        norm_x = np.linalg.norm(x0)
        x0 /= norm_x
    return x0


def dpss_windows(n_time_samples_per_window, time_halfbandwidth_product,
                 n_tapers, is_low_bias=True, interp_from=None,
                 interp_kind='linear'):
    '''Compute Discrete Prolate Spheroidal Sequences.

    Will give of orders [0, n_tapers-1] for a given frequency-spacing
    multiple NW and sequence length `n_time_samples_per_window`.

    Copied from NiTime and MNE-Python

    Parameters
    ----------
    n_time_samples_per_window : int
        Sequence length
    time_halfbandwidth_product : float, unitless
        Standardized half bandwidth corresponding to 2 * half_bw = BW * f0
        = BW * `n_time_samples_per_window` / dt but with dt taken as 1
    n_tapers : int
        Number of DPSS windows to return
    is_low_bias : bool
        Keep only tapers with eigenvalues > 0.9
    interp_from : int (optional)
        The tapers can be calculated using interpolation from a set of
        tapers with the same NW and n_tapers, but shorter
        n_time_samples_per_window.
        This is the length of the shorter set of tapers.
    interp_kind : str (optional)
        This input variable is passed to scipy.interpolate.interp1d and
        specifies the kind of interpolation as a string ('linear',
        'nearest', 'zero', 'slinear', 'quadratic, 'cubic') or as an integer
        specifying the order of the spline interpolator to use.

    Returns
    -------
    tapers, eigenvalues : tuple
        tapers is an array, shape (n_tapers, n_time_samples_per_window)

    Notes
    -----
    Tridiagonal form of DPSS calculation from:
    Slepian, D. Prolate spheroidal wave functions, Fourier analysis, and
    uncertainty V: The discrete case. Bell System Technical Journal,
    Volume 57 (1978), 1371430

    '''
    n_tapers = int(n_tapers)
    half_bandwidth = (float(time_halfbandwidth_product) /
                      n_time_samples_per_window)
    time_index = np.arange(n_time_samples_per_window, dtype='d')

    if interp_from is not None:
        tapers = _find_tapers_from_interpolation(
            interp_from, time_halfbandwidth_product, n_tapers,
            n_time_samples_per_window, interp_kind)
    else:
        tapers = _find_tapers_from_optimization(
            n_time_samples_per_window, time_index, half_bandwidth,
            n_tapers)

    tapers = _fix_taper_sign(tapers, n_time_samples_per_window)
    eigenvalues = _get_taper_eigenvalues(
        tapers, half_bandwidth, time_index)

    return (_get_low_bias_tapers(tapers, eigenvalues)
            if is_low_bias else (tapers, eigenvalues))


def _find_tapers_from_interpolation(
    interp_from, time_halfbandwidth_product, n_tapers,
        n_time_samples_per_window, interp_kind):
    '''Create the tapers of the smaller size `interp_from` and then
    interpolate to the larger size `n_time_samples_per_window`.'''
    smaller_tapers, _ = dpss_windows(
        interp_from, time_halfbandwidth_product, n_tapers,
        is_low_bias=False)

    return [_interpolate_taper(
        taper, interp_kind, n_time_samples_per_window)
        for taper in smaller_tapers]


def _interpolate_taper(taper, interp_kind, n_time_samples_per_window):
    interpolation_function = interpolate.interp1d(
        np.arange(taper.shape[-1]), taper, kind=interp_kind)
    interpolated_taper = interpolation_function(
        np.linspace(0, taper.shape[-1] - 1, n_time_samples_per_window,
                    endpoint=False))
    return interpolated_taper / np.sqrt(np.sum(interpolated_taper ** 2))


def _find_tapers_from_optimization(n_time_samples_per_window, time_index,
                                   half_bandwidth, n_tapers):
    '''here we want to set up an optimization problem to find a sequence
    whose energy is maximally concentrated within band
    [-half_bandwidth, half_bandwidth]. Thus,
    the measure lambda(T, half_bandwidth) is the ratio between the
    energy within that band, and the total energy. This leads to the
    eigen-system (A - (l1)I)v = 0, where the eigenvector corresponding
    to the largest eigenvalue is the sequence with maximally
    concentrated energy. The collection of eigenvectors of this system
    are called Slepian sequences, or discrete prolate spheroidal
    sequences (DPSS). Only the first K, K = 2NW/dt orders of DPSS will
    exhibit good spectral concentration
    [see http://en.wikipedia.org/wiki/Spectral_concentration_problem]

    Here I set up an alternative symmetric tri-diagonal eigenvalue
    problem such that
    (B - (l2)I)v = 0, and v are our DPSS (but eigenvalues l2 != l1)
    the main diagonal = ([n_time_samples_per_window-1-2*t]/2)**2 cos(2PIW),
    t=[0,1,2,...,n_time_samples_per_window-1] and the first off-diagonal =
    t(n_time_samples_per_window-t)/2, t=[1,2,...,
    n_time_samples_per_window-1] [see Percival and Walden, 1993]'''
    diagonal = (
        ((n_time_samples_per_window - 1 - 2 * time_index) / 2.) ** 2
        * np.cos(2 * np.pi * half_bandwidth))
    off_diag = np.zeros_like(time_index)
    off_diag[:-1] = (
        time_index[1:] * (n_time_samples_per_window - time_index[1:]) / 2.)
    # put the diagonals in LAPACK 'packed' storage
    ab = np.zeros((2, n_time_samples_per_window), dtype='d')
    ab[1] = diagonal
    ab[0, 1:] = off_diag[:-1]
    # only calculate the highest n_tapers eigenvalues
    w = eigvals_banded(
        ab, select='i',
        select_range=(n_time_samples_per_window - n_tapers,
                      n_time_samples_per_window - 1))
    w = w[::-1]

    # find the corresponding eigenvectors via inverse iteration
    t = np.linspace(0, np.pi, n_time_samples_per_window)
    tapers = np.zeros((n_tapers, n_time_samples_per_window), dtype='d')
    for taper_ind in range(n_tapers):
        tapers[taper_ind, :] = tridi_inverse_iteration(
            diagonal, off_diag, w[taper_ind],
            x0=np.sin((taper_ind + 1) * t))
    return tapers


def _fix_taper_sign(tapers, n_time_samples_per_window):
    '''By convention (Percival and Walden, 1993 pg 379)
    symmetric tapers (k=0,2,4,...) should have a positive average and
    antisymmetric tapers should begin with a positive lobe.

    Parameters
    ----------
    tapers : array, shape (n_tapers, n_time_samples_per_window)
    '''

    # Fix sign of symmetric tapers
    is_not_symmetric = tapers[::2, :].sum(axis=1) < 0
    fix_sign = is_not_symmetric * -1
    fix_sign[fix_sign == 0] = 1
    tapers[::2, :] *= fix_sign[:, np.newaxis]

    # Fix sign of antisymmetric tapers.
    # rather than test the sign of one point, test the sign of the
    # linear slope up to the first (largest) peak
    largest_peak_ind = np.argmax(
        np.abs(tapers[1::2, :n_time_samples_per_window // 2]), axis=1)
    for taper_ind, peak_ind in enumerate(largest_peak_ind):
        if np.sum(tapers[2 * taper_ind + 1, :peak_ind]) < 0:
            tapers[2 * taper_ind + 1, :] *= -1
    return tapers


def _auto_correlation(data, axis=-1):
    n_time_samples_per_window = data.shape[axis]
    n_fft_samples = next_fast_len(2 * n_time_samples_per_window - 1)
    dpss_fft = fft(data, n_fft_samples, axis=axis)
    power = dpss_fft * dpss_fft.conj()
    return np.real(ifft(power, axis=axis))


def _get_low_bias_tapers(tapers, eigenvalues):
    is_low_bias = eigenvalues > 0.9
    if not np.any(is_low_bias):
        logger.warning('Could not properly use low_bias, '
                       'keeping lowest-bias taper')
        is_low_bias = [np.argmax(eigenvalues)]
    return tapers[is_low_bias, :], eigenvalues[is_low_bias]


def _get_taper_eigenvalues(tapers, half_bandwidth, time_index):
    '''Finds the eigenvalues of the original spectral concentration
    problem using the autocorr sequence technique from Percival and Walden,
    1993 pg 390

    Parameters
    ----------
    tapers : array, shape (n_tapers, n_time_samples_per_window)
    half_bandwidth : float
    time_index : array, (n_time_samples_per_window,)

    Returns
    -------
    eigenvalues : array, shape (n_tapers,)

    '''

    ideal_filter = 4 * half_bandwidth * np.sinc(
        2 * half_bandwidth * time_index)
    ideal_filter[0] = 2 * half_bandwidth
    n_time_samples_per_window = len(time_index)
    return np.dot(
        _auto_correlation(tapers)[:, :n_time_samples_per_window],
        ideal_filter)
