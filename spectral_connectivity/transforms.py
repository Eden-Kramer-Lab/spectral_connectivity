"""Transforms time domain signals to the frequency domain."""

import os
from logging import getLogger
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy import interpolate
from scipy.linalg import eigvals_banded

logger = getLogger(__name__)

if os.environ.get("SPECTRAL_CONNECTIVITY_ENABLE_GPU") == "true":
    try:
        logger.info("Using GPU for spectral_connectivity...")
        import cupy as xp
        from cupy.linalg import lstsq
        from cupyx.scipy.fft import fft, fftfreq, ifft, next_fast_len
    except ImportError:
        print(
            "Cupy not installed. Cupy is needed to use GPU for "
            "spectral_connectivity."
        )
        import numpy as xp
        from scipy.fft import fft, fftfreq, ifft, next_fast_len
        from scipy.linalg import lstsq
else:
    logger.info("Using CPU for spectral_connectivity...")
    import numpy as xp
    from scipy.fft import fft, fftfreq, ifft, next_fast_len
    from scipy.linalg import lstsq


class Multitaper(object):
    """
    Multitaper spectral analysis for robust power spectral density estimation.

    Transforms time-domain signals to frequency domain using multiple orthogonal
    tapering windows (Slepian sequences). This approach reduces spectral leakage
    and provides better spectral estimates than single-taper methods.

    Parameters
    ----------
    time_series : NDArray[floating],
        shape (n_time_samples, n_trials, n_signals) or (n_time_samples, n_signals)
        Input time series data. Multiple trials are supported and will be
        averaged in the spectral domain.
    sampling_frequency : float, default=1000
        Sampling rate in Hz of the time series data.
    time_halfbandwidth_product : float, default=3
        Time-bandwidth product controlling frequency resolution and number of
        tapers. Larger values give better frequency resolution but more spectral
        smoothing. Typical values are 2-4.
    detrend_type : {"constant", "linear", None}, default="constant"
        Type of detrending applied to each time window:
        - "constant": remove DC component
        - "linear": remove linear trend
        - None: no detrending
    time_window_duration : float, optional
        Duration in seconds of sliding time windows. If None, analyzes entire
        time series (no time resolution).
    time_window_step : float, optional
        Step size in seconds between consecutive time windows. If None, uses
        non-overlapping windows (step = window duration).
    n_tapers : int, optional
        Number of DPSS tapers to use. If None, computed as
        floor(2 * time_halfbandwidth_product) - 1.
    tapers : NDArray[floating], shape (n_time_samples_per_window, n_tapers), optional
        Pre-computed tapering windows. If None, DPSS tapers are computed
        automatically.
    start_time : float or NDArray[floating], default=0
        Start time in seconds of the time series data.
    n_fft_samples : int, optional
        Length of FFT. If None, uses next power of 2 >= n_time_samples_per_window.
    n_time_samples_per_window : int, optional
        Number of samples per time window. Computed from time_window_duration
        if not provided.
    n_time_samples_per_step : int, optional
        Number of samples to advance between windows. Computed from
        time_window_step if not provided.
    is_low_bias : bool, default=True
        If True, exclude tapers with eigenvalues < 0.9 to reduce bias.

    Attributes
    ----------
    fft : NDArray[complex128]
        Complex-valued FFT coefficients with shape
        (n_time_windows, n_trials, n_tapers, n_frequencies, n_signals).
    frequencies : NDArray[float64], shape (n_frequencies,)
        Frequency values in Hz corresponding to FFT bins.
    time : NDArray[float64], shape (n_time_windows,)
        Time values in seconds for center of each time window.

    Examples
    --------
    >>> import numpy as np
    >>> # Generate test signal: 50Hz + noise
    >>> fs = 1000  # 1 kHz sampling
    >>> t = np.arange(0, 1, 1/fs)
    >>> signal = np.sin(2*np.pi*50*t) + 0.1*np.random.randn(len(t))
    >>> data = signal[:, np.newaxis]  # Shape: (1000, 1)
    >>>
    >>> # Multitaper analysis
    >>> mt = Multitaper(data, sampling_frequency=fs,
    ...                 time_halfbandwidth_product=4)
    >>> print(f"FFT shape: {mt.fft.shape}")
    >>> print(
    ...     f"Frequencies: {len(mt.frequencies)} bins, "
    ...     f"max = {mt.frequencies[-1]:.1f} Hz"
    ... )

    Notes
    -----
    The multitaper method uses discrete prolate spheroidal sequences (DPSS)
    as tapers, which are optimal for spectral analysis in the sense of minimizing
    spectral leakage while maximizing energy concentration in the frequency band
    of interest.

    References
    ----------
    .. [1] Thomson, D. J. (1982). Spectrum estimation and harmonic analysis.
           Proceedings of the IEEE, 70(9), 1055-1096.
    .. [2] Percival, D. B., & Walden, A. T. (1993). Spectral Analysis for
           Physical Applications. Cambridge University Press.
    """

    def __init__(
        self,
        time_series: NDArray[np.floating],
        sampling_frequency: float = 1000,
        time_halfbandwidth_product: float = 3,
        detrend_type: Optional[str] = "constant",
        time_window_duration: Optional[float] = None,
        time_window_step: Optional[float] = None,
        n_tapers: Optional[int] = None,
        tapers: Optional[NDArray[np.floating]] = None,
        start_time: Union[float, NDArray[np.floating]] = 0,
        n_fft_samples: Optional[int] = None,
        n_time_samples_per_window: Optional[int] = None,
        n_time_samples_per_step: Optional[int] = None,
        is_low_bias: bool = True,
    ) -> None:
        self.time_series = xp.asarray(time_series)
        self.sampling_frequency = sampling_frequency
        self.time_halfbandwidth_product = time_halfbandwidth_product
        self.detrend_type = detrend_type
        self._time_window_duration = time_window_duration
        self._time_window_step = time_window_step
        self.is_low_bias = is_low_bias
        self.start_time = xp.asarray(start_time)
        self._n_fft_samples = n_fft_samples
        self._tapers = tapers
        self._n_tapers = n_tapers
        self._n_time_samples_per_window = n_time_samples_per_window
        self._n_samples_per_time_step = n_time_samples_per_step

    def __repr__(self) -> str:
        """Return string representation of Multitaper object.

        Returns
        -------
        str
            String representation of the Multitaper object.

        """
        return (
            "Multitaper("
            "sampling_frequency={0.sampling_frequency!r}, "
            "time_halfbandwidth_product={0.time_halfbandwidth_product!r},\n"
            "           time_window_duration={0.time_window_duration!r}, "
            "time_window_step={0.time_window_step!r},\n"
            "           detrend_type={0.detrend_type!r}, "
            "start_time={0.start_time}, "
            "n_tapers={0.n_tapers}"
            ")".format(self)
        )

    @property
    def tapers(self) -> NDArray[np.floating]:
        """Return the tapers used for the multitaper function.

        Tapers are the windowing function.

        Returns
        -------
        tapers : array_like, shape (n_time_samples_per_window, n_tapers)
            The tapers used for windowing.

        """
        if self._tapers is None:
            self._tapers = _make_tapers(
                self.n_time_samples_per_window,
                self.sampling_frequency,
                self.time_halfbandwidth_product,
                self.n_tapers,
                is_low_bias=self.is_low_bias,
            )
        return self._tapers

    @property
    def time_window_duration(self) -> float:
        """Return duration of each time bin.

        Returns
        -------
        float
            Duration in seconds of each time window.

        """
        if self._time_window_duration is None:
            self._time_window_duration = (
                self.n_time_samples_per_window / self.sampling_frequency
            )
        return self._time_window_duration

    @property
    def time_window_step(self) -> float:
        """Return how much each time window slides.

        Returns
        -------
        float
            Step size in seconds between consecutive time windows.

        """
        if self._time_window_step is None:
            self._time_window_step = (
                self.n_time_samples_per_step / self.sampling_frequency
            )
        return self._time_window_step

    @property
    def n_tapers(self) -> int:
        """Return number of desired tapers.

        Note that the number of tapers may be less than this number if
        the bias of the tapers is too high (eigenvalues > 0.9).

        Returns
        -------
        int
            Number of tapers to use.

        """
        if self._n_tapers is None:
            return int(xp.floor(2 * self.time_halfbandwidth_product - 1))
        return self._n_tapers

    @property
    def n_time_samples_per_window(self) -> int:
        """Return number of samples per time bin.

        Returns
        -------
        int
            Number of time samples in each window.

        """
        if (
            self._n_time_samples_per_window is None
            and self._time_window_duration is None
        ):
            self._n_time_samples_per_window = self.time_series.shape[0]
        elif self._time_window_duration is not None:
            self._n_time_samples_per_window = int(
                xp.around(self.time_window_duration * self.sampling_frequency)
            )
        return self._n_time_samples_per_window

    @property
    def n_fft_samples(self) -> int:
        """Return number of frequency bins.

        Returns
        -------
        int
            Number of FFT samples.

        """
        if self._n_fft_samples is None:
            self._n_fft_samples = next_fast_len(self.n_time_samples_per_window)
        return self._n_fft_samples

    @property
    def frequencies(self) -> NDArray[np.floating]:
        """Return frequency of each frequency bin.

        Returns
        -------
        NDArray[float64], shape (n_frequencies,)
            Frequency values in Hz corresponding to FFT bins.

        """
        return fftfreq(self.n_fft_samples, 1.0 / self.sampling_frequency)

    @property
    def n_time_samples_per_step(self) -> int:
        """Return number of samples to step between windows.

        If `time_window_step` is set, then calculate the
        `n_time_samples_per_step` based on the time window duration. If
        `time_window_step` and `n_time_samples_per_step` are both not set,
        default the window step size to the same size as the window.

        Returns
        -------
        int
            Number of samples to advance between windows.

        """
        if self._n_samples_per_time_step is None and self._time_window_step is None:
            self._n_samples_per_time_step = self.n_time_samples_per_window
        elif self._time_window_step is not None:
            self._n_samples_per_time_step = int(
                self.time_window_step * self.sampling_frequency
            )
        return self._n_samples_per_time_step

    @property
    def time(self) -> NDArray[np.floating]:
        """Return time of each time bin.

        Returns
        -------
        NDArray[float64], shape (n_time_windows,)
            Time values in seconds for center of each time window.

        """
        original_time = (
            xp.arange(0, self.time_series.shape[0]) / self.sampling_frequency
        )
        window_start_time = _sliding_window(
            original_time, self.n_time_samples_per_window, self.n_time_samples_per_step
        )[:, 0]
        return self.start_time + window_start_time

    @property
    def n_signals(self) -> int:
        """Return number of signals computed.

        Returns
        -------
        int
            Number of signals in the time series.

        """
        return 1 if len(self.time_series.shape) < 2 else self.time_series.shape[-1]

    @property
    def n_trials(self) -> int:
        """Return number of trials computed.

        Returns
        -------
        int
            Number of trials in the time series.

        """
        return 1 if len(self.time_series.shape) < 3 else self.time_series.shape[1]

    @property
    def frequency_resolution(self) -> float:
        """Return range of frequencies the transform is able to resolve.

        Given the time-frequency tradeoff.

        Returns
        -------
        float
            Frequency resolution in Hz.

        """
        return 2.0 * self.time_halfbandwidth_product / self.time_window_duration

    @property
    def nyquist_frequency(self) -> float:
        """Return maximum resolvable frequency.

        Returns
        -------
        float
            Nyquist frequency in Hz.

        """
        return self.sampling_frequency / 2

    def fft(self) -> NDArray[np.complexfloating]:
        """Compute the fast Fourier transform using the multitaper method.

        Returns
        -------
        fourier_coefficients : array
            Shape (n_time_windows, n_trials, n_tapers, n_fft_samples, n_signals).
            Complex-valued Fourier coefficients.

        """
        time_series = _add_axes(self.time_series)
        time_series = _sliding_window(
            time_series,
            window_size=self.n_time_samples_per_window,
            step_size=self.n_time_samples_per_step,
            axis=0,
        )
        if self.detrend_type is not None:
            time_series = detrend(time_series, type=self.detrend_type)

        logger.info(self)

        return _multitaper_fft(
            self.tapers, time_series, self.n_fft_samples, self.sampling_frequency
        ).swapaxes(2, -1)


def _add_axes(time_series: NDArray[np.floating]) -> NDArray[np.floating]:
    """If no trial or signal axes included, add one in."""
    n_axes = len(time_series.shape)
    if n_axes == 1:  # add trials and signals axes
        return time_series[:, xp.newaxis, xp.newaxis]
    elif n_axes == 2:  # add trials axis
        return time_series[:, xp.newaxis, ...]
    else:
        return time_series


def _sliding_window(
    data: NDArray[np.floating],
    window_size: int,
    step_size: int = 1,
    axis: int = -1,
    is_copy: bool = True,
) -> NDArray[np.floating]:
    """Calculate a sliding window over a signal.

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
      output values may occur.

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

    """
    shape = list(data.shape)
    shape[axis] = np.floor(
        (data.shape[axis] / step_size) - (window_size / step_size) + 1
    ).astype(int)
    shape.append(window_size)

    strides = list(data.strides)
    strides[axis] *= step_size
    strides.append(data.strides[axis])

    strided = xp.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)

    return strided.copy() if is_copy else strided


def _multitaper_fft(
    tapers: NDArray[np.floating],
    time_series: NDArray[np.floating],
    n_fft_samples: int,
    sampling_frequency: float,
    axis: int = -2,
) -> NDArray[np.complexfloating]:
    """Project data onto tapers and compute discrete Fourier transform.

    Projects the data on the tapers and returns the discrete Fourier
    transform

    Parameters
    ----------
    tapers : array_like, shape (n_time_samples_per_window, n_tapers)
    time_series : array_like, shape (n_windows, n_trials, n_time_samples_per_window)
    n_fft_samples : int
    sampling_frequency : int

    Returns
    -------
    fourier_coefficients : array_like,
        shape (n_windows, n_trials, n_tapers n_fft_samples, n_signals)

    """
    projected_time_series = (
        time_series[..., xp.newaxis] * tapers[xp.newaxis, xp.newaxis, ...]
    )
    return fft(projected_time_series, n=n_fft_samples, axis=axis) / sampling_frequency


def _make_tapers(
    n_time_samples_per_window,
    sampling_frequency,
    time_halfbandwidth_product,
    n_tapers,
    is_low_bias=True,
):
    """Return discrete prolate spheroidal sequences (tapers) for multitaper analysis.

    Returns the Discrete prolate spheroidal sequences (tapers) for
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

    """
    tapers, _ = dpss_windows(
        n_time_samples_per_window,
        time_halfbandwidth_product,
        n_tapers,
        is_low_bias=is_low_bias,
    )
    return tapers.T * xp.sqrt(sampling_frequency)


def tridisolve(
    d: NDArray[np.floating],
    e: NDArray[np.floating],
    b: NDArray[np.floating],
    overwrite_b: bool = True,
) -> NDArray[np.floating]:
    """Symmetric tridiagonal system solver, from Golub and Van Loan p157.

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
    """
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


def tridi_inverse_iteration(
    d: NDArray[np.floating],
    e: NDArray[np.floating],
    w: float,
    x0: Optional[NDArray[np.floating]] = None,
    rtol: float = 1e-8,
) -> NDArray[np.floating]:
    """Perform an inverse iteration.

    This will find the eigenvector corresponding to the given eigenvalue
    in a symmetric tridiagonal system.

    .. note:: Copied from NiTime.

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
    """
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


def dpss_windows(
    n_time_samples_per_window: int,
    time_halfbandwidth_product: float,
    n_tapers: int,
    is_low_bias: bool = True,
    interp_from: Optional[int] = None,
    interp_kind: str = "linear",
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Compute Discrete Prolate Spheroidal Sequences.

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
        This ixput variable is passed to scipy.interpolate.interp1d and
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

    """
    n_tapers = int(n_tapers)
    half_bandwidth = float(time_halfbandwidth_product) / n_time_samples_per_window
    time_index = xp.arange(n_time_samples_per_window, dtype="d")

    if interp_from is not None:
        tapers = _find_tapers_from_interpolation(
            interp_from,
            time_halfbandwidth_product,
            n_tapers,
            n_time_samples_per_window,
            interp_kind,
        )
    else:
        tapers = _find_tapers_from_optimization(
            n_time_samples_per_window, time_index, half_bandwidth, n_tapers
        )

    _fix_taper_sign(tapers, n_time_samples_per_window)
    eigenvalues = _get_taper_eigenvalues(tapers, half_bandwidth, time_index)

    return (
        _get_low_bias_tapers(tapers, eigenvalues)
        if is_low_bias
        else (tapers, eigenvalues)
    )


def _find_tapers_from_interpolation(
    interp_from: int,
    time_halfbandwidth_product: float,
    n_tapers: int,
    n_time_samples_per_window: int,
    interp_kind: str,
) -> list:
    """Create tapers of smaller size and interpolate to larger size.

    Create the tapers of the smaller size `interp_from` and then
    interpolate to the larger size `n_time_samples_per_window`.
    """
    smaller_tapers, _ = dpss_windows(
        interp_from, time_halfbandwidth_product, n_tapers, is_low_bias=False
    )

    return [
        _interpolate_taper(taper, interp_kind, n_time_samples_per_window)
        for taper in smaller_tapers
    ]


def _interpolate_taper(
    taper: NDArray[np.floating],
    interp_kind: str,
    n_time_samples_per_window: int,
) -> NDArray[np.floating]:
    interpolation_function = interpolate.interp1d(
        xp.arange(taper.shape[-1]), taper, kind=interp_kind
    )
    interpolated_taper = interpolation_function(
        xp.linspace(0, taper.shape[-1] - 1, n_time_samples_per_window, endpoint=False)
    )
    return interpolated_taper / xp.sqrt(xp.sum(interpolated_taper**2))


def _find_tapers_from_optimization(
    n_time_samples_per_window: int,
    time_index: NDArray[np.floating],
    half_bandwidth: float,
    n_tapers: int,
) -> NDArray[np.floating]:
    """Set up optimization problem to find sequence with concentrated energy.

    Set up an optimization problem to find a sequence
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
    n_time_samples_per_window-1] [see Percival and Walden, 1993]
    """
    try:
        time_index = xp.asnumpy(time_index)
    except AttributeError:
        pass
    diagonal = ((n_time_samples_per_window - 1 - 2 * time_index) / 2.0) ** 2 * np.cos(
        2 * np.pi * half_bandwidth
    )
    off_diag = np.zeros_like(time_index)
    off_diag[:-1] = time_index[1:] * (n_time_samples_per_window - time_index[1:]) / 2.0
    # put the diagonals in LAPACK 'packed' storage
    ab = np.zeros((2, n_time_samples_per_window), dtype=float)
    ab[1] = diagonal
    ab[0, 1:] = off_diag[:-1]
    # only calculate the highest n_tapers eigenvalues
    w = eigvals_banded(
        ab,
        select="i",
        select_range=(
            n_time_samples_per_window - n_tapers,
            n_time_samples_per_window - 1,
        ),
    )
    w = w[::-1]

    # find the corresponding eigenvectors via inverse iteration
    t = np.linspace(0, np.pi, n_time_samples_per_window)
    tapers = np.zeros((n_tapers, n_time_samples_per_window), dtype=float)
    for taper_ind in range(n_tapers):
        tapers[taper_ind, :] = tridi_inverse_iteration(
            diagonal, off_diag, w[taper_ind], x0=np.sin((taper_ind + 1) * t)
        )
    return xp.asarray(tapers)


def _fix_taper_sign(
    tapers: NDArray[np.floating], n_time_samples_per_window: int
) -> NDArray[np.floating]:
    """Fix taper signs according to convention.

    By convention (Percival and Walden, 1993 pg 379)
    symmetric tapers (k=0,2,4,...) should have a positive average and
    antisymmetric tapers should begin with a positive lobe.

    Parameters
    ----------
    tapers : array, shape (n_tapers, n_time_samples_per_window)
    """
    # Fix sign of symmetric tapers
    is_not_symmetric = tapers[::2, :].sum(axis=1) < 0
    fix_sign = is_not_symmetric * -1
    fix_sign[fix_sign == 0] = 1
    tapers[::2, :] *= fix_sign[:, xp.newaxis]

    # Fix sign of antisymmetric tapers.
    # rather than test the sign of one point, test the sign of the
    # linear slope up to the first (largest) peak
    largest_peak_ind = xp.argmax(
        xp.abs(tapers[1::2, : n_time_samples_per_window // 2]), axis=1
    )
    for taper_ind, peak_ind in enumerate(largest_peak_ind):
        if xp.sum(tapers[2 * taper_ind + 1, :peak_ind]) < 0:
            tapers[2 * taper_ind + 1, :] *= -1
    return tapers


def _auto_correlation(
    data: NDArray[np.floating], axis: int = -1
) -> NDArray[np.floating]:
    n_time_samples_per_window = data.shape[axis]
    n_fft_samples = next_fast_len(2 * n_time_samples_per_window - 1)
    dpss_fft = fft(data, n_fft_samples, axis=axis)
    power = dpss_fft * dpss_fft.conj()
    return xp.real(ifft(power, axis=axis))


def _get_low_bias_tapers(
    tapers: NDArray[np.floating], eigenvalues: NDArray[np.floating]
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    is_low_bias = eigenvalues > 0.9
    if not xp.any(is_low_bias):
        logger.warning("Could not properly use low_bias, " "keeping lowest-bias taper")
        is_low_bias = [xp.argmax(eigenvalues)]
    return tapers[is_low_bias, :], eigenvalues[is_low_bias]


def _get_taper_eigenvalues(
    tapers: NDArray[np.floating],
    half_bandwidth: float,
    time_index: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Find eigenvalues of spectral concentration problem.

    Find the eigenvalues of the original spectral concentration
    problem using the autocorr sequence technique from Percival and Walden,
    1993 pg 390.

    Parameters
    ----------
    tapers : array, shape (n_tapers, n_time_samples_per_window)
    half_bandwidth : float
    time_index : array, (n_time_samples_per_window,)

    Returns
    -------
    eigenvalues : array, shape (n_tapers,)

    """
    ideal_filter = 4 * half_bandwidth * xp.sinc(2 * half_bandwidth * time_index)
    ideal_filter[0] = 2 * half_bandwidth
    n_time_samples_per_window = len(time_index)
    return xp.dot(
        _auto_correlation(tapers)[:, :n_time_samples_per_window], ideal_filter
    )


def detrend(
    data: NDArray[np.floating],
    axis: int = -1,
    type: str = "linear",
    bp: Union[int, list, NDArray[np.integer]] = 0,
    overwrite_data: bool = False,
) -> NDArray[np.floating]:
    """
    Remove linear trend along axis from data.

    Copied from scipy and now uses cupy or numpy functions.

    Parameters
    ----------
    data : array_like
        The input data.
    axis : int, optional
        The axis along which to detrend the data. By default this is the
        last axis (-1).
    type : {'linear', 'constant'}, optional
        The type of detrending. If ``type == 'linear'`` (default),
        the result of a linear least-squares fit to `data` is subtracted
        from `data`.
        If ``type == 'constant'``, only the mean of `data` is subtracted.
    bp : array_like of ints, optional
        A sequence of break points. If given, an individual linear fit is
        performed for each part of `data` between two break points.
        Break points are specified as indices into `data`. This parameter
        only has an effect when ``type == 'linear'``.
    overwrite_data : bool, optional
        If True, perform in place detrending and avoid a copy. Default is False

    Returns
    -------
    ret : ndarray
        The detrended input data.

    Examples
    --------
    >>> from scipy import signal
    >>> from numpy.random import default_rng
    >>> rng = default_rng()
    >>> npoints = 1000
    >>> noise = rng.standard_normal(npoints)
    >>> x = 3 + 2*np.linspace(0, 1, npoints) + noise
    >>> (signal.detrend(x) - noise).max()
    0.06  # random
    """
    if type not in ["linear", "l", "constant", "c"]:
        raise ValueError("Trend type must be 'linear' or 'constant'.")
    data = xp.asarray(data)
    dtype = data.dtype.char
    if dtype not in "dfDF":
        dtype = "d"
    if type in ["constant", "c"]:
        return data - xp.mean(data, axis, keepdims=True)
    else:
        dshape = data.shape
        N = dshape[axis]
        bp = xp.sort(xp.unique(xp.r_[0, bp, N]))
        if xp.any(bp > N):
            raise ValueError(
                "Breakpoints must be less than length " "of data along given axis."
            )
        Nreg = len(bp) - 1
        # Restructure data so that axis is along first dimension and
        #  all other dimensions are collapsed into second dimension
        rnk = len(dshape)
        if axis < 0:
            axis = axis + rnk
        newdims = xp.r_[axis, 0:axis, axis + 1 : rnk]
        newdata = xp.reshape(
            xp.transpose(data, tuple(newdims)), (N, np.prod(dshape) // N)
        )
        if not overwrite_data:
            newdata = newdata.copy()  # make sure we have a copy
        if newdata.dtype.char not in "dfDF":
            newdata = newdata.astype(dtype)
        # Find leastsq fit and remove it for each piece
        for m in range(Nreg):
            Npts = bp[m + 1] - bp[m]
            A = xp.ones((Npts, 2), dtype)
            A[:, 0] = xp.cast[dtype](np.arange(1, Npts + 1) * 1.0 / Npts)
            sl = slice(bp[m], bp[m + 1])
            coef, resids, rank, s = lstsq(A, newdata[sl])
            newdata[sl] = newdata[sl] - A @ coef
        # Put data back in original shape.
        tdshape = xp.take(dshape, newdims, 0)
        ret = xp.reshape(newdata, tuple(tdshape))
        vals = list(range(1, rnk))
        olddims = vals[:axis] + [0] + vals[axis:]
        return xp.transpose(ret, tuple(olddims))
