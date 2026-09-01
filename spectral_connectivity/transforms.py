"""Transforms time domain signals to the frequency domain."""

from logging import getLogger
from typing import Any, Literal, TypedDict

import numpy as np
from numpy.typing import NDArray
from scipy.signal.windows import dpss as scipy_dpss
from scipy.signal.windows import hann as scipy_hann

from spectral_connectivity.utils import (
    BackendArray,
    is_gpu_enabled,
    is_positive_integer,
    mark_readonly_if_supported,
    to_numpy,
)

logger = getLogger(__name__)

# Physical Constants with Scientific Rationale
#
# MIN_EIGENVALUE_THRESHOLD: Minimum eigenvalue for low-bias tapers
# - Eigenvalues represent the concentration of taper energy within the desired bandwidth
# - Value of 0.9 means 90% of the taper's energy is contained in the main lobe
# - Tapers with lower eigenvalues have more spectral leakage from side lobes
# - This threshold provides a good balance between bias reduction and preserving enough tapers
# - Reference: Thomson (1982), "Spectrum estimation and harmonic analysis"
MIN_EIGENVALUE_THRESHOLD = 0.9

# TAPER_MULTIPLIER: Multiplier for calculating number of tapers from time_halfbandwidth_product
# - The number of tapers is floor(2 * NW) - 1, where NW is time_halfbandwidth_product
# - Factor of 2 comes from the spectral concentration theory: approximately 2*NW orthogonal
#   functions (Slepian sequences) exist that are well-concentrated in both time and frequency
# - The -1 ensures we stay within the well-concentrated region
# - Reference: Slepian (1978), "Prolate spheroidal wave functions"
TAPER_MULTIPLIER = 2.0


def _validate_sampling_frequency(sampling_frequency: float) -> None:
    """Raise an actionable error for a non-finite or non-positive sampling rate.

    Shared by every transform so the newer STFT/Welch/Morlet classes give the
    same guidance as :class:`Multitaper` rather than a bare one-line message.
    """
    if not np.isfinite(sampling_frequency) or sampling_frequency <= 0:
        raise ValueError(
            f"sampling_frequency must be finite and positive, got "
            f"{sampling_frequency!r}.\n"
            "\n"
            "The sampling frequency is the rate at which your data was collected.\n"
            "Common values:\n"
            "  - EEG: 250-1000 Hz\n"
            "  - LFP/ephys: 1000-30000 Hz\n"
            "  - fMRI: 0.5-2 Hz (1/TR)\n"
            "\n"
            "Check your data acquisition settings or metadata."
        )


def _resolve_sample_count(
    explicit: int | None,
    duration: float | None,
    sampling_frequency: float,
    mismatch_message: str,
) -> int | None:
    """Resolve a window/segment length from an explicit count or a duration.

    Returns the sample count when either ``explicit`` or ``duration`` is given
    (cross-checking that they agree when both are), or ``None`` when neither is
    provided so the caller can apply its own default. Shared by the STFT window,
    STFT step, and Welch segment resolution.
    """
    if explicit is not None:
        samples = int(explicit)
        if duration is not None and int(np.around(duration * sampling_frequency)) != (
            samples
        ):
            raise ValueError(mismatch_message)
        return samples
    if duration is not None:
        return int(np.around(duration * sampling_frequency))
    return None


class MultitaperParameters(TypedDict):
    """Parameter suggestions for multitaper analysis.

    Attributes
    ----------
    sampling_frequency : float
        Sampling rate in Hz.
    time_halfbandwidth_product : float
        Suggested time-bandwidth product (NW).
    time_window_duration : float
        Suggested window duration in seconds.
    n_tapers : int
        Number of tapers.
    frequency_resolution : float
        Resulting frequency resolution in Hz.
    n_time_windows : int
        Approximate number of time windows.
    nyquist_frequency : float
        Maximum frequency (sampling_frequency / 2) in Hz.
    """

    sampling_frequency: float
    time_halfbandwidth_product: float
    time_window_duration: float
    n_tapers: int
    frequency_resolution: float
    n_time_windows: int
    nyquist_frequency: float


def estimate_frequency_resolution(
    sampling_frequency: float,
    time_window_duration: float,
    time_halfbandwidth_product: float,
) -> float:
    """
    Estimate the frequency resolution for given multitaper parameters.

    The frequency resolution (Δf) represents the bandwidth over which spectral
    energy is averaged. It is determined by the time-frequency trade-off inherent
    in spectral analysis.

    Parameters
    ----------
    sampling_frequency : float
        Sampling rate in Hz of the time series data.
        Note: This doesn't affect frequency resolution, only the maximum
        frequency (Nyquist = sampling_frequency / 2).
    time_window_duration : float
        Duration in seconds of each analysis window.
        Longer windows provide better (finer) frequency resolution.
    time_halfbandwidth_product : float
        Time-bandwidth product controlling the spectral concentration.
        Higher values provide more spectral smoothing (coarser resolution).

    Returns
    -------
    frequency_resolution : float
        Frequency resolution in Hz. This represents the bandwidth over which
        spectral estimates are averaged.

    Notes
    -----
    The frequency resolution formula is:

    .. math::
        \\Delta f = \\frac{2 \\cdot NW}{T}

    where:
    - NW is the time-halfbandwidth product
    - T is the time window duration in seconds

    **Key relationships:**
    - Longer time windows (↑T) → Better resolution (↓Δf)
    - Higher time-halfbandwidth product (↑NW) → More smoothing (↑Δf)

    **Typical values:**
    - For 1 Hz resolution: Use T=6s with NW=3
    - For 5 Hz resolution: Use T=1.2s with NW=3
    - For 0.5 Hz resolution: Use T=12s with NW=3

    Examples
    --------
    EEG application with 1 Hz resolution:

    >>> freq_res = estimate_frequency_resolution(
    ...     sampling_frequency=250,
    ...     time_window_duration=6.0,
    ...     time_halfbandwidth_product=3,
    ... )
    >>> print(f"Frequency resolution: {freq_res} Hz")
    Frequency resolution: 1.0 Hz

    LFP application with 5 Hz resolution:

    >>> freq_res = estimate_frequency_resolution(
    ...     sampling_frequency=1000,
    ...     time_window_duration=1.2,
    ...     time_halfbandwidth_product=3,
    ... )
    >>> print(f"Frequency resolution: {freq_res} Hz")
    Frequency resolution: 5.0 Hz

    See Also
    --------
    estimate_n_tapers : Estimate number of tapers
    suggest_parameters : Automatically suggest parameters for target resolution
    """
    return TAPER_MULTIPLIER * time_halfbandwidth_product / time_window_duration


def estimate_n_tapers(time_halfbandwidth_product: float) -> int:
    """
    Estimate the number of tapers for a given time-halfbandwidth product.

    The number of tapers determines how many independent spectral estimates
    are averaged together. More tapers provide better variance reduction
    but also more spectral smoothing.

    Parameters
    ----------
    time_halfbandwidth_product : float
        Time-bandwidth product. Typical values are 2-5.

    Returns
    -------
    n_tapers : int
        Number of discrete prolate spheroidal sequence (DPSS) tapers that
        will be used.

    Notes
    -----
    The number of tapers is calculated as:

    .. math::
        n_{\\text{tapers}} = \\lfloor 2 \\cdot NW \\rfloor - 1

    where NW is the time-halfbandwidth product.

    **Note:** The actual number of tapers used by Multitaper may be lower
    if `is_low_bias=True` (default), which excludes tapers with eigenvalues < MIN_EIGENVALUE_THRESHOLD (0.9).

    **Typical values:**
    - NW=2 → 3 tapers (minimal averaging)
    - NW=3 → 5 tapers (balanced, recommended)
    - NW=4 → 7 tapers (strong averaging)
    - NW=5 → 9 tapers (very strong averaging)

    Examples
    --------
    >>> n_tapers = estimate_n_tapers(time_halfbandwidth_product=3)
    >>> print(f"Number of tapers: {n_tapers}")
    Number of tapers: 5

    >>> n_tapers = estimate_n_tapers(time_halfbandwidth_product=4)
    >>> print(f"Number of tapers: {n_tapers}")
    Number of tapers: 7

    See Also
    --------
    estimate_frequency_resolution : Estimate frequency resolution
    suggest_parameters : Automatically suggest parameters
    """
    return int(np.floor(TAPER_MULTIPLIER * time_halfbandwidth_product)) - 1


def suggest_parameters(
    sampling_frequency: float,
    signal_duration: float,
    desired_freq_resolution: float | None = None,
    desired_n_tapers: int | None = None,
) -> MultitaperParameters:
    """
    Suggest appropriate multitaper parameters for your analysis.

    This helper function recommends parameters based on your data characteristics
    and analysis goals. It helps answer the common question: "What parameters
    should I use for my data?"

    Parameters
    ----------
    sampling_frequency : float
        Sampling rate in Hz of your data.
    signal_duration : float
        Total duration in seconds of your signal.
    desired_freq_resolution : float, optional
        Target frequency resolution in Hz. If specified, parameters will be
        chosen to achieve approximately this resolution.
        Cannot be specified together with desired_n_tapers.
    desired_n_tapers : int, optional
        Target number of tapers for variance reduction. If specified,
        time_halfbandwidth_product will be chosen to achieve this.
        Cannot be specified together with desired_freq_resolution.

    Returns
    -------
    params : MultitaperParameters
        TypedDict containing suggested parameters with the following keys:
        - 'sampling_frequency': float - Input sampling frequency
        - 'time_halfbandwidth_product': float - Suggested NW
        - 'time_window_duration': float - Suggested window duration (seconds)
        - 'n_tapers': int - Number of tapers
        - 'frequency_resolution': float - Resulting frequency resolution (Hz)
        - 'n_time_windows': int - Approximate number of time windows
        - 'nyquist_frequency': float - Maximum frequency (Hz)

    Raises
    ------
    ValueError
        If desired frequency resolution is impossible to achieve with given
        signal duration.

    Warns
    -----
    UserWarning
        If both desired_freq_resolution and desired_n_tapers are specified.
        In this case, desired_freq_resolution takes precedence and
        desired_n_tapers is ignored.

    Notes
    -----
    **Default behavior** (no targets specified):
    - Uses NW=3 (balanced trade-off)
    - Sets window duration to capture ~5 time windows
    - Aims for reasonable frequency and time resolution

    **With desired_freq_resolution:**
    - Calculates required window duration: T = 2*NW / Δf_target
    - Uses NW=3 by default unless this gives too few time windows
    - Ensures at least 3 time windows for temporal dynamics

    **With desired_n_tapers:**
    - Calculates NW from: NW = (n_tapers + 1) / 2
    - Uses reasonable window duration based on signal length

    Examples
    --------
    Get reasonable defaults for EEG data:

    >>> params = suggest_parameters(
    ...     sampling_frequency=250,
    ...     signal_duration=60.0,
    ... )
    >>> print(f"Suggested NW: {params['time_halfbandwidth_product']}")
    Suggested NW: 3.0
    >>> print(f"Window duration: {params['time_window_duration']:.2f}s")
    Window duration: 12.00s
    >>> print(f"Frequency resolution: {params['frequency_resolution']:.2f} Hz")
    Frequency resolution: 0.50 Hz
    >>> print(f"Number of tapers: {params['n_tapers']}")
    Number of tapers: 5

    Target specific frequency resolution:

    >>> params = suggest_parameters(
    ...     sampling_frequency=1000,
    ...     signal_duration=10.0,
    ...     desired_freq_resolution=2.0,  # Want 2 Hz resolution
    ... )
    >>> print(f"Achieved resolution: {params['frequency_resolution']:.2f} Hz")
    Achieved resolution: 2.00 Hz

    Target specific number of tapers:

    >>> params = suggest_parameters(
    ...     sampling_frequency=1000,
    ...     signal_duration=5.0,
    ...     desired_n_tapers=9,  # Want strong averaging
    ... )
    >>> print(f"Number of tapers: {params['n_tapers']}")
    Number of tapers: 9

    See Also
    --------
    estimate_frequency_resolution : Calculate frequency resolution
    estimate_n_tapers : Calculate number of tapers
    Multitaper.summarize_parameters : Display parameters for existing analysis
    """
    import warnings

    # Validate inputs
    if desired_freq_resolution is not None and desired_n_tapers is not None:
        warnings.warn(
            "Both 'desired_freq_resolution' and 'desired_n_tapers' were specified. "
            "This is typically not recommended as they have competing effects on the analysis. "
            "Using 'desired_freq_resolution' and ignoring 'desired_n_tapers'.",
            UserWarning,
            stacklevel=2,
        )
        desired_n_tapers = None

    # Default: balanced parameters (NW=3)
    if desired_freq_resolution is None and desired_n_tapers is None:
        time_halfbandwidth_product = 3.0
        # Use ~20% of signal duration as window, aim for ~5 windows
        time_window_duration = min(signal_duration / 5.0, signal_duration * 0.2)
        # But ensure at least 0.5s window for reasonable freq resolution
        time_window_duration = max(time_window_duration, 0.5)
        # And don't exceed signal duration
        time_window_duration = min(time_window_duration, signal_duration)

    # User wants specific frequency resolution
    elif desired_freq_resolution is not None:
        # Start with NW=3 (typical balanced value)
        time_halfbandwidth_product = 3.0

        # Calculate required window duration: T = 2*NW / Δf
        time_window_duration = (
            TAPER_MULTIPLIER * time_halfbandwidth_product / desired_freq_resolution
        )

        # Check if this is achievable with the signal duration
        if time_window_duration > signal_duration:
            raise ValueError(
                f"Cannot achieve desired frequency resolution of {desired_freq_resolution} Hz "
                f"with signal duration of {signal_duration}s.\n"
                "\n"
                f"Required window duration: {time_window_duration:.2f}s\n"
                f"Available signal duration: {signal_duration:.2f}s\n"
                "\n"
                "To achieve this resolution, you need either:\n"
                f"  - Longer signal (at least {time_window_duration:.2f}s)\n"
                f"  - Coarser frequency resolution (at least "
                f"{TAPER_MULTIPLIER * time_halfbandwidth_product / signal_duration:.2f} Hz)"
            )

        # If window would give us fewer than 3 time windows, increase NW slightly
        # to reduce window duration (at the cost of coarser freq resolution)
        min_n_windows = 3
        max_window_for_min_windows = signal_duration / min_n_windows
        if time_window_duration > max_window_for_min_windows:
            # Adjust NW to give us at least min_n_windows
            time_window_duration = max_window_for_min_windows
            # Recalculate NW to achieve target resolution with this window
            time_halfbandwidth_product = (
                desired_freq_resolution * time_window_duration / 2.0
            )
            # But keep NW >= 1
            time_halfbandwidth_product = max(time_halfbandwidth_product, 1.0)

    # User wants specific number of tapers
    elif desired_n_tapers is not None:
        # Calculate NW from n_tapers: n_tapers = floor(2*NW) - 1
        # So: NW = (n_tapers + 1) / 2
        time_halfbandwidth_product = (desired_n_tapers + 1) / 2.0

        # Use reasonable window duration (~20% of signal, but at least 0.5s)
        time_window_duration = min(signal_duration / 5.0, signal_duration * 0.2)
        time_window_duration = max(time_window_duration, 0.5)
        time_window_duration = min(time_window_duration, signal_duration)

    else:
        # This should never happen given the logic above, but for type safety
        raise ValueError("Internal error: unexpected parameter combination")

    # Calculate derived parameters
    n_tapers = estimate_n_tapers(time_halfbandwidth_product)
    frequency_resolution = estimate_frequency_resolution(
        sampling_frequency, time_window_duration, time_halfbandwidth_product
    )
    n_time_windows = int(
        np.floor(signal_duration / time_window_duration)
    )  # Non-overlapping estimate
    nyquist_frequency = sampling_frequency / 2.0

    return {
        "sampling_frequency": sampling_frequency,
        "time_halfbandwidth_product": time_halfbandwidth_product,
        "time_window_duration": time_window_duration,
        "n_tapers": n_tapers,
        "frequency_resolution": frequency_resolution,
        "n_time_windows": n_time_windows,
        "nyquist_frequency": nyquist_frequency,
    }


if is_gpu_enabled():
    try:
        import cupy as xp
        from cupyx.scipy.fft import fft, fftfreq, ifft, next_fast_len
    except ImportError as exc:
        raise RuntimeError(
            "GPU support was explicitly requested via SPECTRAL_CONNECTIVITY_ENABLE_GPU='true', "
            "but CuPy is not installed. Please install CuPy with: "
            "'pip install cupy' or 'conda install cupy'"
        ) from exc
    try:
        # cupyx.scipy.signal.detrend was added in CuPy 13; a CuPy-12 install
        # imports cupy fine but fails here, which must not be reported as
        # "CuPy is not installed".
        from cupyx.scipy.signal import detrend as _backend_detrend
    except ImportError as exc:
        raise RuntimeError(
            f"GPU support requires cupy-cuda12x>=13.0, but CuPy {xp.__version__} "
            f"is installed: cupyx.scipy.signal.detrend (used by transforms.detrend) "
            f"was added in CuPy 13. Upgrade with 'pip install -U cupy-cuda12x'."
        ) from exc

    # Log GPU device information
    try:
        device = xp.cuda.Device()
        # Try to get the actual GPU model name first
        try:
            device_name = xp.cuda.runtime.getDeviceProperties(device.id)[
                "name"
            ].decode()
            device_name = device_name.strip("\x00")
        except Exception:
            # Fallback to compute capability
            compute_cap = device.compute_capability
            device_name = f"GPU (Compute Capability {compute_cap[0]}.{compute_cap[1]})"
        logger.info(f"Using GPU for spectral_connectivity on {device_name}")
    except Exception:
        logger.info("Using GPU for spectral_connectivity...")
else:
    logger.info("Using CPU for spectral_connectivity...")
    import numpy as xp
    from scipy.fft import fft, fftfreq, ifft, next_fast_len
    from scipy.signal import detrend as _backend_detrend


def _immutable_array_snapshot(value: Any) -> BackendArray:
    """Copy an input array and make the owned snapshot read-only when supported."""
    return mark_readonly_if_supported(xp.array(value, copy=True))


def _readonly_array_copy(array: BackendArray) -> BackendArray:
    """Return a detached read-only copy of an internal array snapshot."""
    return mark_readonly_if_supported(array.copy())


class Multitaper:
    """
    Multitaper spectral analysis for robust power spectral density estimation.

    Transforms time-domain signals to frequency domain using multiple orthogonal
    tapering windows (Slepian sequences). This approach reduces spectral leakage
    and provides better spectral estimates than single-taper methods.

    Parameters
    ----------
    time_series : NDArray[floating], shape (n_time_samples, n_trials, n_signals)
        Input time series data. **Must be 3D array.**
        - n_time_samples: number of time points
        - n_trials: number of trials (use 1 for single trial)
        - n_signals: number of signals/channels
        Multiple trials are averaged in the spectral domain.

        **Important:** If your data is 1D or 2D, use `prepare_time_series()`
        helper function to convert it to the required 3D format.
    sampling_frequency : float, default=1000
        Sampling rate in Hz of the time series data.
    time_halfbandwidth_product : float, default=3
        Time-bandwidth product (often denoted as NW) controlling the trade-off
        between frequency resolution and variance reduction.

        **Effect on analysis:**
        - Determines frequency resolution: Δf = 2·NW / T (where T is window duration)
        - Determines number of tapers: n_tapers = floor(2·NW) - 1
        - Higher values → More spectral smoothing, better variance reduction
        - Lower values → Better frequency resolution, less averaging

        **Typical values:**
        - NW = 2: Minimal smoothing, 3 tapers, best frequency resolution
        - NW = 3: Balanced trade-off (recommended default), 5 tapers
        - NW = 4: More smoothing, 7 tapers, strong variance reduction
        - NW = 5+: Heavy smoothing, 9+ tapers, very strong variance reduction

        **Examples:**
        For 1 Hz frequency resolution with 1 second windows: use NW ≤ 0.5
        For 5 Hz frequency resolution with 1 second windows: use NW ≤ 2.5
        For 10 Hz frequency resolution with 1 second windows: use NW ≤ 5

        Use `estimate_frequency_resolution()` to calculate the resolution
        for different parameter combinations, or use `suggest_parameters()`
        to get recommendations for your specific data and analysis goals.
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
        Length of FFT. If None, uses a value >= n_time_samples_per_window chosen
        to be fast for the FFT algorithm. The value is determined by
        scipy.fft.next_fast_len (or cupy.fft.next_fast_len when GPU is enabled).
    n_time_samples_per_window : int, optional
        Number of samples per time window. Computed from time_window_duration
        if not provided.
    n_time_samples_per_step : int, optional
        Number of samples to advance between windows. Computed from
        time_window_step if not provided.
    is_low_bias : bool, default=True
        If True, exclude tapers with eigenvalues < MIN_EIGENVALUE_THRESHOLD (0.9) to reduce bias.
    fft_workers : int, optional
        Number of parallel worker threads for the CPU FFT (forwarded to
        ``scipy.fft.fft``; ``-1`` uses all cores). ``None`` (the default) keeps
        SciPy's single-threaded default, which avoids oversubscribing CPUs when
        the analysis is already parallelized at a higher level (e.g. across
        trials or subjects). This is a CPU-only option; it is ignored on the GPU
        backend, whose FFT is already parallel.
    taper_weighting : {"uniform", "eigen", "adaptive"}, default="uniform"
        How DPSS eigencoefficients contribute to spectra. ``"uniform"`` keeps
        historical equal weighting (a good default). ``"eigen"`` down-weights
        the less concentrated tapers by their eigenvalues. ``"adaptive"``
        iteratively estimates frequency- and signal-specific Thomson weights;
        reach for it when high spectral dynamic range or line noise makes some
        tapers systematically leakier, at the cost of extra iterations for a
        typically modest bias reduction. Non-uniform modes require internally
        generated DPSS tapers.
    adaptive_max_iterations : int, default=50
        Iteration ceiling for Thomson adaptive weighting.
    adaptive_tolerance : float, default=1e-8
        Relative convergence tolerance for the adaptive spectrum estimate.

    Attributes
    ----------
    fft : NDArray[complex128]
        Complex-valued FFT coefficients with shape
        (n_time_windows, n_trials, n_tapers, n_frequencies, n_signals).
    frequencies : NDArray[float64], shape (n_frequencies,)
        Frequency values in Hz corresponding to FFT bins.
    time : NDArray[float64], shape (n_time_windows,)
        Time values in seconds for center of each time window.

    See Also
    --------
    spectral_connectivity.connectivity.Connectivity : Compute connectivity
        measures from a Multitaper transform.
    spectral_connectivity.transforms.prepare_time_series : Reshape 1-D/2-D input
        to the required 3-D format.

    Notes
    -----
    A ``Multitaper`` represents one immutable transform configuration. Constructor
    arrays are copied, and array properties return detached read-only copies. To
    change the data or a parameter, create a new instance.

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

    Examples
    --------
    Using the helper function (recommended for 2D data):

    >>> import numpy as np
    >>> from spectral_connectivity.transforms import Multitaper, prepare_time_series
    >>> # EEG recording: 5 seconds at 1000 Hz, 64 channels
    >>> eeg_data = np.random.randn(5000, 64)  # Shape: (n_time, n_channels)
    >>> eeg_3d = prepare_time_series(eeg_data, axis='signals')
    >>> mt = Multitaper(eeg_3d, sampling_frequency=1000, time_halfbandwidth_product=4)
    >>> print(f"FFT shape: {mt.fft().shape}")
    FFT shape: (1, 1, 7, 5000, 64)
    >>> print(f"Frequencies: {len(mt.frequencies)} bins, max = {mt.frequencies.max():.1f} Hz")
    Frequencies: 5000 bins, max = 499.8 Hz

    Manual reshaping with np.newaxis (advanced):

    >>> # Generate test signal: 50Hz + noise
    >>> fs = 1000  # 1 kHz sampling
    >>> t = np.arange(0, 1, 1/fs)
    >>> signal = np.sin(2*np.pi*50*t) + 0.1*np.random.randn(len(t))
    >>> # Manually reshape to 3D: (n_time, n_trials, n_signals)
    >>> data = signal[:, np.newaxis, np.newaxis]  # Shape: (1000, 1, 1)
    >>> mt = Multitaper(data, sampling_frequency=fs, time_halfbandwidth_product=4)

    Multiple trials (already 3D):

    >>> # Epoched data: 100 trials, 5 channels, 1 second each at 1000 Hz
    >>> epoched_data = np.random.randn(1000, 100, 5)  # (n_time, n_trials, n_signals)
    >>> mt = Multitaper(epoched_data, sampling_frequency=1000)
    >>> print(f"Trials: {mt.n_trials}, Signals: {mt.n_signals}")
    Trials: 100, Signals: 5
    """

    #: Multitaper returns the full two-sided FFT spectrum (both positive and
    #: negative frequencies), so consumers must not assume a one-sided layout.
    is_one_sided = False

    _IMMUTABLE_PUBLIC_PARAMETERS = frozenset(
        {
            "sampling_frequency",
            "time_halfbandwidth_product",
            "detrend_type",
            "is_low_bias",
            "fft_workers",
            "taper_weighting",
            "adaptive_max_iterations",
            "adaptive_tolerance",
        }
    )
    _PROVENANCE_FIELDS = (
        "detrend_type",
        "fft_workers",
        "frequency_resolution",
        "is_low_bias",
        "n_fft_samples",
        "n_signals",
        "n_tapers",
        "n_time_samples_per_step",
        "n_time_samples_per_window",
        "n_trials",
        "nyquist_frequency",
        "sampling_frequency",
        "start_time",
        "taper_weighting",
        "adaptive_max_iterations",
        "adaptive_tolerance",
        "time_halfbandwidth_product",
        "time_window_duration",
        "time_window_step",
    )

    def __setattr__(self, name: str, value: Any) -> None:
        """Keep transform parameters immutable once construction has completed."""
        if (
            getattr(self, "_initialized", False)
            and name in self._IMMUTABLE_PUBLIC_PARAMETERS
        ):
            raise AttributeError(
                f"Multitaper.{name} is immutable after construction; create a new "
                "Multitaper instance with the desired parameters."
            )
        object.__setattr__(self, name, value)

    def __init__(
        self,
        time_series: NDArray[np.floating],
        sampling_frequency: float = 1000,
        time_halfbandwidth_product: float = 3,
        detrend_type: str | None = "constant",
        time_window_duration: float | None = None,
        time_window_step: float | None = None,
        n_tapers: int | None = None,
        tapers: NDArray[np.floating] | None = None,
        start_time: float | NDArray[np.floating] = 0,
        n_fft_samples: int | None = None,
        n_time_samples_per_window: int | None = None,
        n_time_samples_per_step: int | None = None,
        is_low_bias: bool = True,
        fft_workers: int | None = None,
        taper_weighting: Literal["uniform", "eigen", "adaptive"] = "uniform",
        adaptive_max_iterations: int = 50,
        adaptive_tolerance: float = 1e-8,
    ) -> None:
        object.__setattr__(self, "_initialized", False)
        self._time_series = _immutable_array_snapshot(time_series)
        # fft_workers is forwarded verbatim to scipy.fft.fft(workers=...): it must
        # be a nonzero integer (a negative value counts back from os.cpu_count(),
        # so -1 uses all cores) or None (SciPy's single-threaded default).
        # Validate here so a bad value names this parameter, instead of surfacing
        # a bare "workers must not be zero" ValueError or an opaque TypeError from
        # deep inside fft(). bool is an int subclass, so reject it explicitly.
        if fft_workers is not None and (
            isinstance(fft_workers, bool)
            or not isinstance(fft_workers, (int, np.integer))
            or fft_workers == 0
        ):
            raise ValueError(
                f"fft_workers must be None or a nonzero integer (the number of "
                f"parallel FFT threads; -1 uses all cores), got {fft_workers!r}. "
                f"It is forwarded to scipy.fft.fft(workers=...). Use None (the "
                f"default) for SciPy's single-threaded FFT."
            )
        self.fft_workers = fft_workers
        if taper_weighting not in {"uniform", "eigen", "adaptive"}:
            raise ValueError(
                "taper_weighting must be 'uniform', 'eigen', or 'adaptive', got "
                f"{taper_weighting!r}. Use 'uniform' for the historical equal "
                "weighting, 'eigen' to weight tapers by their spectral "
                "concentration, or 'adaptive' for Thomson's iterative weighting."
            )
        if not is_positive_integer(adaptive_max_iterations):
            raise ValueError(
                "adaptive_max_iterations must be a positive integer, got "
                f"{adaptive_max_iterations!r}."
            )
        if not np.isfinite(adaptive_tolerance) or adaptive_tolerance <= 0:
            raise ValueError(
                "adaptive_tolerance must be finite and positive, got "
                f"{adaptive_tolerance!r}."
            )
        if tapers is not None and taper_weighting != "uniform":
            raise ValueError(
                "Eigenvalue/adaptive weighting requires internally generated "
                "DPSS tapers; custom tapers do not provide concentration ratios. "
                "Use taper_weighting='uniform' with custom tapers, or omit the "
                "tapers argument to let Multitaper generate DPSS tapers."
            )
        self.taper_weighting = taper_weighting
        self.adaptive_max_iterations = int(adaptive_max_iterations)
        self.adaptive_tolerance = float(adaptive_tolerance)

        # Validate that time_series is 3D
        if self._time_series.ndim != 3:
            error_msg = (
                f"Expected 3D array with shape (n_time_samples, n_trials, n_signals), "
                f"but got {self._time_series.ndim}D array with shape {self._time_series.shape}.\n"
                "\n"
            )

            if self._time_series.ndim == 1:
                error_msg += (
                    "For a single time series, use:\n"
                    "  >>> from spectral_connectivity.transforms import prepare_time_series\n"
                    "  >>> time_series_3d = prepare_time_series(time_series)\n"
                    "Or manually:\n"
                    "  >>> time_series_3d = time_series[:, np.newaxis, np.newaxis]"
                )
            elif self._time_series.ndim == 2:
                error_msg += (
                    "For 2D data, you must clarify the meaning of the second dimension.\n"
                    "\n"
                    "Use prepare_time_series() helper:\n"
                    "  >>> from spectral_connectivity.transforms import prepare_time_series\n"
                    "  >>> # If shape is (n_time, n_signals) with 1 trial:\n"
                    "  >>> time_series_3d = prepare_time_series(time_series, axis='signals')\n"
                    "  >>> # If shape is (n_time, n_trials) with 1 signal:\n"
                    "  >>> time_series_3d = prepare_time_series(time_series, axis='trials')\n"
                    "\n"
                    "Or manually with np.newaxis:\n"
                    "  >>> # For (n_time, n_signals) → (n_time, 1, n_signals):\n"
                    "  >>> time_series_3d = time_series[:, np.newaxis, :]\n"
                    "  >>> # For (n_time, n_trials) → (n_time, n_trials, 1):\n"
                    "  >>> time_series_3d = time_series[:, :, np.newaxis]"
                )
            else:
                error_msg += (
                    f"Arrays with {self._time_series.ndim} dimensions are not supported.\n"
                    "Expected shape: (n_time_samples, n_trials, n_signals)"
                )

            raise ValueError(error_msg)

        _validate_sampling_frequency(sampling_frequency)

        # Validate time_halfbandwidth_product
        if time_halfbandwidth_product < 1:
            raise ValueError(
                f"time_halfbandwidth_product must be at least 1, got {time_halfbandwidth_product}.\n"
                "\n"
                "The time-halfbandwidth product controls the spectral concentration and\n"
                "number of tapers used. It represents the trade-off between:\n"
                "  - Frequency resolution (lower values = better resolution)\n"
                "  - Variance reduction (higher values = more averaging, less variance)\n"
                "\n"
                "Typical values:\n"
                "  - 1-2: Good frequency resolution, minimal smoothing\n"
                "  - 3-4: Balanced (recommended for most applications)\n"
                "  - 5+: Heavy smoothing, strong variance reduction\n"
                "\n"
                "A value less than 1 is not physically meaningful."
            )

        # Warn if time_halfbandwidth_product is unusually large
        if time_halfbandwidth_product > 10:
            import warnings

            warnings.warn(
                f"time_halfbandwidth_product = {time_halfbandwidth_product} is unusually large.\n"
                "\n"
                "Values above 10 apply very heavy spectral smoothing and are rarely used.\n"
                "This will create many tapers and significantly slow computation.\n"
                "\n"
                "Common values are 1-5. If you're unsure, try 3 (a typical default).\n"
                "If you specifically need heavy smoothing, you can ignore this warning.",
                UserWarning,
                stacklevel=2,
            )

        # Validate time_window_duration
        if time_window_duration is not None and time_window_duration <= 0:
            raise ValueError(
                f"time_window_duration must be positive, got {time_window_duration}.\n"
                "\n"
                "The time window duration is the length of each analysis window in seconds.\n"
                "It determines the frequency resolution: Δf ≈ 2 * time_halfbandwidth_product / time_window_duration.\n"
                "\n"
                "Examples:\n"
                "  - 1.0 second window with NW=3 → ~6 Hz resolution\n"
                "  - 0.5 second window with NW=3 → ~12 Hz resolution\n"
                "\n"
                "Use None (default) to analyze the entire time series without windowing."
            )

        # Validate time_window_step
        if time_window_step is not None and time_window_step <= 0:
            raise ValueError(
                f"time_window_step must be positive, got {time_window_step}.\n"
                "\n"
                "The time window step is how far to advance the window for each computation (in seconds).\n"
                "  - Small step: More time resolution, more overlapping windows, slower computation\n"
                "  - Large step: Less overlap, faster computation, coarser time resolution\n"
                "\n"
                "Common choices:\n"
                "  - step = duration: No overlap (consecutive windows)\n"
                "  - step = duration/2: 50% overlap (recommended for most applications)\n"
                "  - step < duration: More overlap, smoother temporal evolution\n"
                "\n"
                "Use None (default) to match time_window_duration (no overlap)."
            )

        # Warn if time_window_step creates gaps between windows
        if (
            time_window_step is not None
            and time_window_duration is not None
            and time_window_step > time_window_duration
        ):
            import warnings

            warnings.warn(
                f"time_window_step ({time_window_step}s) is larger than "
                f"time_window_duration ({time_window_duration}s).\n"
                "\n"
                "This creates gaps between analysis windows - some data will not be analyzed.\n"
                "If you want contiguous coverage, set step ≤ duration.\n"
                "\n"
                "If gaps are intended (e.g., sampling every Nth window), ignore this warning.",
                UserWarning,
                stacklevel=2,
            )

        # Warn if data appears to be transposed (very few time points, many signals)
        n_time, _, n_signals = self._time_series.shape
        if n_time < n_signals:
            import warnings

            warnings.warn(
                f"Your time series has only {n_time} time points but {n_signals} signals. "
                "This seems unusual and your data may be transposed.\n"
                "\n"
                "Expected shape: (n_time_samples, n_trials, n_signals)\n"
                f"Your shape: {self._time_series.shape}\n"
                "\n"
                "If your data is transposed, use:\n"
                "  >>> time_series_correct = time_series.T  # or appropriate transpose\n"
                "\n"
                "If your data is intentionally short (e.g., very brief epochs), you can ignore this warning.",
                UserWarning,
                stacklevel=2,
            )

        # Warn if data contains NaN or Inf
        if not xp.all(xp.isfinite(self._time_series)):
            import warnings

            warnings.warn(
                "Input time_series contains NaN or infinite values.\n"
                "\n"
                "This will produce invalid spectral estimates. Common causes:\n"
                "  - Missing data or recording artifacts\n"
                "  - Incorrect preprocessing (division by zero, log of negative)\n"
                "  - Corrupted data files\n"
                "\n"
                "Suggestions:\n"
                "  - Interpolate missing values: scipy.interpolate.interp1d()\n"
                "  - Remove bad segments or trials\n"
                "  - Check your preprocessing pipeline\n"
                "  - Verify data integrity\n"
                "\n"
                "For artifact removal, consider using MNE-Python or similar tools.",
                UserWarning,
                stacklevel=2,
            )

        self.sampling_frequency = sampling_frequency
        self.time_halfbandwidth_product = time_halfbandwidth_product
        self.detrend_type = detrend_type
        self._time_window_duration = time_window_duration
        self._time_window_step = time_window_step
        self.is_low_bias = is_low_bias
        self._start_time = _immutable_array_snapshot(start_time)
        self._n_fft_samples = n_fft_samples
        self._tapers = None if tapers is None else _immutable_array_snapshot(tapers)
        self._taper_eigenvalues: BackendArray | None = None
        # Reject a fractional n_tapers at construction so the reported
        # n_tapers metadata cannot disagree with the (integer) taper count used.
        if n_tapers is not None and (
            not np.isfinite(n_tapers) or int(n_tapers) != n_tapers
        ):
            raise ValueError(f"n_tapers must be an integer, got {n_tapers}.")
        self._n_tapers = n_tapers
        self._n_time_samples_per_window = n_time_samples_per_window
        self._n_samples_per_time_step = n_time_samples_per_step
        object.__setattr__(self, "_initialized", True)

    @property
    def time_series(self) -> BackendArray:
        """Independent read-only copy of the input time-series snapshot."""
        return _readonly_array_copy(self._time_series)

    @property
    def start_time(self) -> BackendArray:
        """Independent read-only copy of the transform's start-time coordinate."""
        return _readonly_array_copy(self._start_time)

    def __repr__(self) -> str:
        """Return string representation of Multitaper object.

        Returns
        -------
        str
            String representation of the Multitaper object.

        """
        return (
            "Multitaper("
            f"sampling_frequency={self.sampling_frequency!r}, "
            f"time_halfbandwidth_product={self.time_halfbandwidth_product!r},\n"
            f"           time_window_duration={self.time_window_duration!r}, "
            f"time_window_step={self.time_window_step!r},\n"
            f"           detrend_type={self.detrend_type!r}, "
            f"start_time={self.start_time}, "
            f"n_tapers={self.n_tapers}"
            ")"
        )

    def _provenance_metadata(self) -> dict[str, Any]:
        """Public scalar parameters of this transform, for provenance metadata.

        Collects the public, non-callable attributes whose values NetCDF can
        store (strings, numbers, bools, and real numeric/string arrays),
        encoding ``None`` as ``"None"`` and skipping the large coordinate/data
        arrays (``time_series``, ``fft``, ``tapers``, ``frequencies``,
        ``time``). Used to label results with ``mt_*`` attributes in
        :func:`spectral_connectivity.wrapper.connectivity_to_xarray`.
        """
        metadata: dict[str, Any] = {}
        for attr in self._PROVENANCE_FIELDS:
            value = getattr(self, attr)
            if value is None:
                value = "None"
            else:
                value = to_numpy(value) if hasattr(value, "shape") else value
            if isinstance(value, np.ndarray):
                if value.dtype.kind not in "biufSU":  # exclude complex/object
                    continue
            elif not isinstance(
                value,
                (str, bytes, bool, int, float, np.integer, np.floating, np.bool_),
            ):
                continue
            metadata[attr] = value
        return metadata

    def summarize_parameters(self) -> str:
        """
        Generate a human-readable summary of the multitaper analysis parameters.

        This method displays key parameters and their implications for your analysis,
        making it easier to understand and communicate your spectral analysis settings.

        Returns
        -------
        summary : str
            A formatted string containing:
            - Input parameters (sampling frequency, time-halfbandwidth product)
            - Derived parameters (n_tapers, frequency resolution)
            - Data dimensions (n_signals, n_trials, n_time_samples)
            - Frequency range (0 to Nyquist)

        Examples
        --------
        >>> import numpy as np
        >>> from spectral_connectivity.transforms import Multitaper
        >>> data = np.random.randn(5000, 1, 64)  # 5s, 64 EEG channels
        >>> mt = Multitaper(
        ...     data,
        ...     sampling_frequency=1000,
        ...     time_window_duration=1.0,
        ...     time_halfbandwidth_product=3,
        ... )
        >>> print(mt.summarize_parameters())
        Multitaper Spectral Analysis Configuration
        ===========================================
        <BLANKLINE>
        Data Shape
        ----------
        Time samples:    5000 (5.00 seconds)
        Signals:         64
        Trials:          1
        <BLANKLINE>
        Spectral Parameters
        -------------------
        Sampling frequency:            1000 Hz
        Time-halfbandwidth product:    3
        Number of tapers:              5
        <BLANKLINE>
        Time Windowing
        --------------
        Window duration:  1.000 s (1000 samples)
        Window step:      1.000 s (non-overlapping)
        Number of windows: 5
        <BLANKLINE>
        Frequency Analysis
        ------------------
        Frequency resolution: 6.0 Hz
        Nyquist frequency:    500.0 Hz
        Frequency range:      0.0 - 500.0 Hz
        FFT samples:          1000
        <BLANKLINE>

        See Also
        --------
        suggest_parameters : Get parameter suggestions before creating Multitaper
        estimate_frequency_resolution : Estimate frequency resolution
        estimate_n_tapers : Estimate number of tapers
        """
        # Calculate time windows info
        n_time_samples = self._time_series.shape[0]
        signal_duration = n_time_samples / self.sampling_frequency
        n_windows = int(
            xp.floor(
                (n_time_samples - self.n_time_samples_per_window)
                / self.n_time_samples_per_step
            )
            + 1
        )

        # Determine overlap description
        if self.time_window_step == self.time_window_duration:
            overlap_desc = "(non-overlapping)"
        else:
            overlap_percent = (
                100
                * (self.time_window_duration - self.time_window_step)
                / self.time_window_duration
            )
            overlap_desc = f"({overlap_percent:.0f}% overlap)"

        summary = f"""Multitaper Spectral Analysis Configuration
===========================================

Data Shape
----------
Time samples:    {n_time_samples} ({signal_duration:.2f} seconds)
Signals:         {self.n_signals}
Trials:          {self.n_trials}

Spectral Parameters
-------------------
Sampling frequency:            {self.sampling_frequency} Hz
Time-halfbandwidth product:    {self.time_halfbandwidth_product}
Number of tapers:              {self.n_tapers}

Time Windowing
--------------
Window duration:  {self.time_window_duration:.3f} s ({self.n_time_samples_per_window} samples)
Window step:      {self.time_window_step:.3f} s {overlap_desc}
Number of windows: {n_windows}

Frequency Analysis
------------------
Frequency resolution: {self.frequency_resolution:.1f} Hz
Nyquist frequency:    {self.nyquist_frequency:.1f} Hz
Frequency range:      0.0 - {self.nyquist_frequency:.1f} Hz
FFT samples:          {self.n_fft_samples}
"""

        return summary

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
            tapers, eigenvalues = dpss_windows(
                self.n_time_samples_per_window,
                self.time_halfbandwidth_product,
                self.n_tapers,
                is_low_bias=self.is_low_bias,
            )
            self._tapers = _immutable_array_snapshot(
                tapers.T * xp.sqrt(self.sampling_frequency)
            )
            self._taper_eigenvalues = _immutable_array_snapshot(eigenvalues)
        return _readonly_array_copy(self._tapers)

    @property
    def taper_eigenvalues(self) -> NDArray[np.floating] | None:
        """DPSS spectral-concentration ratios used for taper weighting.

        Returns ``None`` for custom tapers, whose concentration ratios are not
        known. The returned array is a detached read-only copy.
        """
        if self._tapers is None:
            _ = self.tapers
        if self._taper_eigenvalues is None:
            return None
        return _readonly_array_copy(self._taper_eigenvalues)

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
        the bias of the tapers is too high (eigenvalues > MIN_EIGENVALUE_THRESHOLD = 0.9).

        Returns
        -------
        int
            Number of tapers to use.

        """
        if self._n_tapers is None:
            return int(xp.floor(TAPER_MULTIPLIER * self.time_halfbandwidth_product - 1))
        return self._n_tapers

    @property
    def n_time_samples_per_window(self) -> int:
        """Return number of samples per time bin.

        Returns
        -------
        int
            Number of time samples in each window.

        Raises
        ------
        ValueError
            If neither n_time_samples_per_window nor time_window_duration is set.

        """
        if (
            self._n_time_samples_per_window is None
            and self._time_window_duration is None
        ):
            self._n_time_samples_per_window = self._time_series.shape[0]
        elif self._time_window_duration is not None:
            self._n_time_samples_per_window = int(
                xp.around(self.time_window_duration * self.sampling_frequency)
            )
        # Otherwise n_time_samples_per_window was set explicitly.
        assert self._n_time_samples_per_window is not None
        # Validate the resolved window length regardless of which input path set
        # it: an explicit n_time_samples_per_window=0 (or a duration rounding to
        # 0) would divide by zero downstream, and an oversized window yields an
        # empty transform.
        n_time_samples = self._time_series.shape[0]
        if self._n_time_samples_per_window < 1:
            raise ValueError(
                f"n_time_samples_per_window resolved to "
                f"{self._n_time_samples_per_window}, but each window needs at "
                f"least 1 sample. If you set time_window_duration "
                f"({self._time_window_duration}), it is too short for "
                f"sampling_frequency ({self.sampling_frequency}); use a duration "
                f">= {1.0 / self.sampling_frequency} s. Otherwise pass a positive "
                f"n_time_samples_per_window."
            )
        if self._n_time_samples_per_window > n_time_samples:
            raise ValueError(
                f"n_time_samples_per_window ({self._n_time_samples_per_window}) is "
                f"larger than the signal length ({n_time_samples}); no window fits, "
                f"which would yield an empty transform. Use a smaller window "
                f"(time_window_duration <= "
                f"{n_time_samples / self.sampling_frequency} s)."
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
        elif self._n_fft_samples < self.n_time_samples_per_window:
            # scipy/cupy fft crops (does not zero-pad) when n < len(signal), so
            # a too-small n_fft_samples silently discards most of each window.
            raise ValueError(
                f"n_fft_samples ({self._n_fft_samples}) must be >= the number "
                f"of time samples per window ({self.n_time_samples_per_window}).\n"
                f"n_fft_samples is the FFT length (used for zero-padding), not "
                f"the number of output frequency bins. A value smaller than the "
                f"window length would silently truncate the signal before the "
                f"FFT.\n"
                f"Either omit n_fft_samples (it defaults to a fast length >= the "
                f"window length) or set it >= {self.n_time_samples_per_window}."
            )
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
        # Otherwise n_time_samples_per_step was set explicitly.
        assert self._n_samples_per_time_step is not None
        # Validate the resolved step regardless of which input path set it: an
        # explicit n_time_samples_per_step=0 (or a step truncating to 0) would
        # divide by zero when building the sliding windows.
        if self._n_samples_per_time_step < 1:
            raise ValueError(
                f"n_time_samples_per_step resolved to "
                f"{self._n_samples_per_time_step}, but each step must advance at "
                f"least 1 sample. If you set time_window_step "
                f"({self._time_window_step}), it is too short for sampling_frequency "
                f"({self.sampling_frequency}); use a step "
                f">= {1.0 / self.sampling_frequency} s. Otherwise pass a positive "
                f"n_time_samples_per_step."
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
            xp.arange(0, self._time_series.shape[0]) / self.sampling_frequency
        )
        # Label each window by its center time, as documented (the mean of the
        # window's sample times equals the center for uniformly spaced samples).
        window_center_time = _sliding_window(
            original_time, self.n_time_samples_per_window, self.n_time_samples_per_step
        ).mean(axis=-1)
        return self._start_time + window_center_time

    @property
    def n_signals(self) -> int:
        """Return number of signals computed.

        Returns
        -------
        int
            Number of signals in the time series.

        """
        return 1 if len(self._time_series.shape) < 2 else self._time_series.shape[-1]

    @property
    def n_trials(self) -> int:
        """Return number of trials computed.

        Returns
        -------
        int
            Number of trials in the time series.

        """
        return 1 if len(self._time_series.shape) < 3 else self._time_series.shape[1]

    @property
    def frequency_resolution(self) -> float:
        """Return range of frequencies the transform is able to resolve.

        Given the time-frequency tradeoff.

        Returns
        -------
        float
            Frequency resolution in Hz.

        """
        return (
            TAPER_MULTIPLIER
            * self.time_halfbandwidth_product
            / self.time_window_duration
        )

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
        time_series = _add_axes(self._time_series)
        time_series = _sliding_window(
            time_series,
            window_size=self.n_time_samples_per_window,
            step_size=self.n_time_samples_per_step,
            axis=0,
        )
        if self.detrend_type is not None:
            time_series = detrend(time_series, type=self.detrend_type)

        logger.info(self)

        coefficients = _multitaper_fft(
            self._tapers_for_fft(),
            time_series,
            self.n_fft_samples,
            self.sampling_frequency,
            workers=self.fft_workers,
        ).swapaxes(2, -1)
        if self.taper_weighting == "uniform":
            return coefficients

        assert self._taper_eigenvalues is not None
        eigenvalues = self._taper_eigenvalues
        if self.taper_weighting == "eigen":
            weights = xp.sqrt(eigenvalues)
            weights = weights / xp.sqrt(xp.mean(weights**2))
            return (
                coefficients
                * weights[xp.newaxis, xp.newaxis, :, xp.newaxis, xp.newaxis]
            )

        # Thomson's adaptive weights balance the taper periodogram against the
        # process-noise level, which must be on the *same* power-spectral-density
        # scale. ``fft()`` returns ``fft(taper * x) / sampling_frequency`` with
        # tapers scaled by ``sqrt(sampling_frequency)``, so ``|coefficients| ** 2``
        # is a PSD (signal ** 2 / Hz) while ``var(time_series)`` is a raw
        # time-domain variance (signal ** 2). Dividing by ``sampling_frequency``
        # puts the noise term on the PSD scale; without it the noise is inflated
        # by a factor of ``sampling_frequency`` and the weighting is miscalibrated.
        noise_power_spectral_density = (
            xp.var(time_series, axis=-1) / self.sampling_frequency
        )
        return _apply_adaptive_taper_weights(
            coefficients,
            eigenvalues,
            noise_power_spectral_density,
            max_iterations=self.adaptive_max_iterations,
            tolerance=self.adaptive_tolerance,
        )

    def _tapers_for_fft(self) -> NDArray[np.floating]:
        """Return the internal taper snapshot without an unnecessary public view."""
        if self._tapers is None:
            # Populate through the public property so generation and freezing have
            # one implementation, then use the owned internal snapshot.
            _ = self.tapers
        assert self._tapers is not None
        return self._tapers


class ShortTimeFourierTransform(Multitaper):
    """Short-time Fourier transform using an L2-normalized Hann window.

    The returned coefficients follow the same five-dimensional contract as
    :class:`Multitaper`, with a singleton taper axis. This makes the transform
    directly usable with :meth:`Connectivity.from_transform` and
    :func:`spectral_connectivity.wrapper.connectivity_to_xarray`.

    Parameters
    ----------
    time_series : ndarray, shape (n_time_samples, n_trials, n_signals)
        Input signals. Use :func:`prepare_time_series` for 1-D/2-D input.
    sampling_frequency : float, default=1000
        Samples per second (Hz).
    detrend_type : {"constant", "linear"} or None, default="constant"
        Detrending applied to each window before the FFT.
    time_window_duration : float, optional
        Window length in seconds. Give this or ``n_time_samples_per_window``.
    time_window_step : float, optional
        Step between successive windows in seconds (defaults to the window
        duration, i.e. no overlap).
    start_time : float or ndarray, default=0
        Time of the first sample, in seconds.
    n_fft_samples : int, optional
        FFT length; defaults to the window length.
    n_time_samples_per_window : int, optional
        Window length in samples (alternative to ``time_window_duration``).
    n_time_samples_per_step : int, optional
        Step between windows in samples (alternative to ``time_window_step``).
    fft_workers : int, optional
        Worker threads for SciPy's CPU FFT (``-1`` uses all cores).
    """

    _provenance_prefix = "stft_"
    is_one_sided = False

    def __init__(
        self,
        time_series: NDArray[np.floating],
        sampling_frequency: float = 1000,
        detrend_type: str | None = "constant",
        time_window_duration: float | None = None,
        time_window_step: float | None = None,
        start_time: float | NDArray[np.floating] = 0,
        n_fft_samples: int | None = None,
        n_time_samples_per_window: int | None = None,
        n_time_samples_per_step: int | None = None,
        fft_workers: int | None = None,
    ) -> None:
        _validate_sampling_frequency(sampling_frequency)
        for name, value in (
            ("time_window_duration", time_window_duration),
            ("time_window_step", time_window_step),
        ):
            if value is not None and (not np.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive.")
        for name, value in (
            ("n_time_samples_per_window", n_time_samples_per_window),
            ("n_time_samples_per_step", n_time_samples_per_step),
        ):
            if value is not None and not is_positive_integer(value):
                raise ValueError(f"{name} must be a positive integer.")
        data_shape: tuple[int, ...] = tuple(getattr(time_series, "shape", ()))
        if len(data_shape) != 3:
            raise ValueError(
                "time_series must have shape (n_time_samples, n_trials, n_signals)."
            )
        resolved_window = _resolve_sample_count(
            n_time_samples_per_window,
            time_window_duration,
            sampling_frequency,
            "time_window_duration and n_time_samples_per_window resolve to "
            "different Hann window lengths.",
        )
        window_samples = (
            int(data_shape[0]) if resolved_window is None else resolved_window
        )
        if window_samples < 2:
            raise ValueError("A Hann transform window requires at least 2 samples.")

        resolved_step = _resolve_sample_count(
            n_time_samples_per_step,
            time_window_step,
            sampling_frequency,
            "time_window_step and n_time_samples_per_step resolve to different "
            "step lengths.",
        )
        step_samples = window_samples if resolved_step is None else resolved_step

        window = scipy_hann(window_samples, sym=False)
        norm = np.linalg.norm(window)
        if norm == 0:
            raise ValueError("The requested Hann window has zero energy.")
        window = (window / norm * np.sqrt(sampling_frequency))[:, np.newaxis]
        super().__init__(
            time_series,
            sampling_frequency=sampling_frequency,
            time_halfbandwidth_product=1,
            detrend_type=detrend_type,
            time_window_duration=window_samples / sampling_frequency,
            time_window_step=step_samples / sampling_frequency,
            n_tapers=1,
            tapers=window,
            start_time=start_time,
            n_fft_samples=n_fft_samples,
            n_time_samples_per_window=window_samples,
            n_time_samples_per_step=step_samples,
            is_low_bias=False,
            fft_workers=fft_workers,
        )

    @property
    def frequency_resolution(self) -> float:
        """Equivalent-noise bandwidth of the periodic Hann window in Hz."""
        return 1.5 / self.time_window_duration

    def _provenance_metadata(self) -> dict[str, Any]:
        metadata = super()._provenance_metadata()
        for key in (
            "time_halfbandwidth_product",
            "is_low_bias",
            "taper_weighting",
            "adaptive_max_iterations",
            "adaptive_tolerance",
        ):
            metadata.pop(key, None)
        metadata["window"] = "hann_periodic"
        return metadata


class Welch:
    """Welch spectral transform using overlapping Hann-windowed segments.

    Segment coefficients are represented on the taper/observation axis, so the
    default ``trials_tapers`` expectation averages both trials and Welch
    segments and returns a single spectrum centered on the analyzed record.

    Parameters
    ----------
    time_series : ndarray, shape (n_time_samples, n_trials, n_signals)
        Input signals. Use :func:`prepare_time_series` for 1-D/2-D input.
    sampling_frequency : float, default=1000
        Samples per second (Hz).
    segment_duration : float, optional
        Segment length in seconds; sets the frequency resolution
        (``1 / segment_duration`` Hz). Strongly recommended -- the fallback of
        256 samples does not scale with the sampling rate. Give this or
        ``n_time_samples_per_segment``.
    segment_overlap : float, default=0.5
        Fractional overlap between successive segments, in ``[0, 1)``.
    n_time_samples_per_segment : int, optional
        Segment length in samples (alternative to ``segment_duration``).
    detrend_type : {"constant", "linear"} or None, default="constant"
        Detrending applied to each segment before the FFT.
    start_time : float or ndarray, default=0
        Time of the first sample, in seconds.
    n_fft_samples : int, optional
        FFT length; defaults to the segment length.
    fft_workers : int, optional
        Worker threads for SciPy's CPU FFT (``-1`` uses all cores).
    """

    _provenance_prefix = "welch_"
    is_one_sided = False

    def __init__(
        self,
        time_series: NDArray[np.floating],
        sampling_frequency: float = 1000,
        segment_duration: float | None = None,
        segment_overlap: float = 0.5,
        n_time_samples_per_segment: int | None = None,
        detrend_type: str | None = "constant",
        start_time: float | NDArray[np.floating] = 0,
        n_fft_samples: int | None = None,
        fft_workers: int | None = None,
    ) -> None:
        _validate_sampling_frequency(sampling_frequency)
        if segment_duration is not None and (
            not np.isfinite(segment_duration) or segment_duration <= 0
        ):
            raise ValueError("segment_duration must be finite and positive.")
        if n_time_samples_per_segment is not None and not is_positive_integer(
            n_time_samples_per_segment, minimum=2
        ):
            raise ValueError(
                "n_time_samples_per_segment must be an integer of at least 2."
            )
        n_time_samples = int(getattr(time_series, "shape", (0,))[0])
        segment_samples = _resolve_sample_count(
            n_time_samples_per_segment,
            segment_duration,
            sampling_frequency,
            "segment_duration and n_time_samples_per_segment resolve to different "
            "lengths.",
        )
        if segment_samples is None:
            import warnings

            segment_samples = min(256, n_time_samples)
            # SciPy's historical 256-sample default does not scale with the
            # sampling rate, so at typical electrophysiology rates it yields a
            # very short window and coarse frequency resolution (e.g. ~117 Hz at
            # 30 kHz), collapsing neuroscience bands. Warn so the choice is
            # explicit rather than silent.
            frequency_resolution = sampling_frequency / segment_samples
            if frequency_resolution > 4:
                warnings.warn(
                    f"Welch is using the default segment length of "
                    f"{segment_samples} samples, giving a "
                    f"{segment_samples / sampling_frequency * 1e3:.1f} ms window "
                    f"and {frequency_resolution:.1f} Hz frequency resolution at "
                    f"sampling_frequency={sampling_frequency:g} Hz. This is too "
                    "coarse for typical neuroscience bands. Pass segment_duration "
                    "(seconds) or n_time_samples_per_segment to set the resolution "
                    "explicitly.",
                    UserWarning,
                    stacklevel=2,
                )
        if not np.isfinite(segment_overlap) or not 0 <= segment_overlap < 1:
            raise ValueError("segment_overlap must satisfy 0 <= overlap < 1.")
        step_samples = max(1, int(np.around(segment_samples * (1 - segment_overlap))))
        self._stft = ShortTimeFourierTransform(
            time_series,
            sampling_frequency=sampling_frequency,
            detrend_type=detrend_type,
            start_time=start_time,
            n_fft_samples=n_fft_samples,
            n_time_samples_per_window=segment_samples,
            n_time_samples_per_step=step_samples,
            fft_workers=fft_workers,
        )
        self.sampling_frequency = sampling_frequency
        self.segment_overlap = float(segment_overlap)
        self.n_time_samples_per_segment = segment_samples
        self.n_time_samples_per_step = step_samples

    @property
    def frequencies(self) -> NDArray[np.floating]:
        return self._stft.frequencies

    @property
    def time(self) -> NDArray[np.floating]:
        return xp.asarray(self._stft.time).mean(keepdims=True)

    @property
    def n_trials(self) -> int:
        return self._stft.n_trials

    @property
    def n_signals(self) -> int:
        return self._stft.n_signals

    @property
    def n_segments(self) -> int:
        return len(self._stft.time)

    def fft(self) -> NDArray[np.complexfloating]:
        coefficients = self._stft.fft()[:, :, 0, :, :]
        return xp.transpose(coefficients, (1, 0, 2, 3))[xp.newaxis, ...]

    def _provenance_metadata(self) -> dict[str, Any]:
        return {
            "detrend_type": self._stft.detrend_type or "None",
            "fft_workers": self._stft.fft_workers
            if self._stft.fft_workers is not None
            else "None",
            "n_fft_samples": self._stft.n_fft_samples,
            "n_segments": self.n_segments,
            "n_signals": self.n_signals,
            "n_time_samples_per_segment": self.n_time_samples_per_segment,
            "n_time_samples_per_step": self.n_time_samples_per_step,
            "n_trials": self.n_trials,
            "sampling_frequency": self.sampling_frequency,
            "segment_overlap": self.segment_overlap,
            "window": "hann_periodic",
        }


class MorletWavelet:
    """Complex Morlet transform with explicit edge and 2-D smoothing controls.

    Coefficients contain only the requested positive frequencies and therefore
    support functional, but not Wilson-factorized directed, connectivity. With
    multiple trials, the default expectation averages trials at each time point.
    For a single continuous trial, set ``smoothing_time`` to collect neighboring
    wavelet coefficients on the observation axis; otherwise normalized pairwise
    measures are degenerate at unit magnitude.

    Parameters
    ----------
    time_series : ndarray, shape (n_time_samples, n_trials, n_signals)
        Input signals. Use :func:`prepare_time_series` for 1-D/2-D input.
    sampling_frequency : float
        Samples per second (Hz).
    frequencies : ndarray, shape (n_frequencies,)
        Strictly increasing positive frequencies (Hz) below Nyquist.
    n_cycles : float or ndarray, default=7.0
        Number of oscillation cycles per wavelet, scalar or one per frequency.
        Controls the time/frequency-resolution trade-off: more cycles give
        sharper frequency but coarser time resolution.
    decimation : int, default=1
        Keep every ``decimation``-th output sample to reduce memory/compute.
    smoothing_time : float, optional
        Duration (seconds) of a sliding window whose coefficients are collected
        onto the observation axis. Set this for single-trial data so normalized
        measures (coherence, PLV) are not degenerate.
    smoothing_step : float, optional
        Step (seconds) between smoothing windows; defaults to ``smoothing_time``
        and requires it.
    smoothing_frequency : int, default=1
        Odd number of adjacent requested frequency bins collected into each
        local estimate. Frequency boundaries are reflected, matching the
        boundary convention used by MNE's time-resolved spectral smoothing.
    smoothing_kernel : {"boxcar", "hann"}, default="boxcar"
        Separable time/frequency weights for the local estimate. ``"boxcar"``
        preserves the historical equal-weight smoothing; ``"hann"`` reduces
        discontinuities at the neighborhood boundary. The Hann weights are the
        interior of a symmetric Hann window two samples wider than the
        neighborhood, so every sample carries a non-zero weight (e.g.
        ``smoothing_frequency=3`` weights the bins ``[0.5, 1, 0.5]``).
    padding_mode : {"constant", "reflect", "edge"}, default="constant"
        How the time series is extended before convolution. ``"constant"``
        (zero padding) preserves the historical transform.
    edge_mode : {"keep", "nan", "trim"}, default="keep"
        Treatment of coefficients whose five-standard-deviation wavelet support
        extends beyond the original record. ``"keep"`` retains padded values,
        ``"nan"`` makes derived estimates NaN, and ``"trim"`` removes times
        that are not valid for every requested frequency.
    zero_mean : bool, default=True
        Subtract the wavelet's mean so it has no DC response.
    start_time : float, default=0.0
        Time of the first sample, in seconds.
    """

    _provenance_prefix = "morlet_"
    # Coefficients are already on the one-sided PSD scale (see fft), so
    # Connectivity must not double them again.
    is_one_sided = True

    def __init__(
        self,
        time_series: NDArray[np.floating],
        sampling_frequency: float,
        frequencies: NDArray[np.floating],
        n_cycles: float | NDArray[np.floating] = 7.0,
        *,
        decimation: int = 1,
        smoothing_time: float | None = None,
        smoothing_step: float | None = None,
        smoothing_frequency: int = 1,
        smoothing_kernel: Literal["boxcar", "hann"] = "boxcar",
        padding_mode: Literal["constant", "reflect", "edge"] = "constant",
        edge_mode: Literal["keep", "nan", "trim"] = "keep",
        zero_mean: bool = True,
        start_time: float = 0.0,
    ) -> None:
        data = xp.asarray(time_series)
        if data.ndim != 3:
            raise ValueError(
                "time_series must have shape (n_time_samples, n_trials, n_signals)."
            )
        _validate_sampling_frequency(sampling_frequency)
        frequency_values = np.asarray(frequencies, dtype=float)
        if (
            frequency_values.ndim != 1
            or frequency_values.size == 0
            or not np.all(np.isfinite(frequency_values))
            or not np.all(frequency_values > 0)
            or not np.all(np.diff(frequency_values) > 0)
            or np.any(frequency_values >= sampling_frequency / 2)
        ):
            raise ValueError(
                "frequencies must be a non-empty, finite, strictly increasing "
                "positive array below Nyquist."
            )
        cycle_values = np.asarray(n_cycles, dtype=float)
        if cycle_values.ndim == 0:
            cycle_values = np.full(frequency_values.shape, float(cycle_values))
        if cycle_values.shape != frequency_values.shape or not np.all(
            np.isfinite(cycle_values) & (cycle_values > 0)
        ):
            raise ValueError(
                "n_cycles must be a positive scalar or have one value per frequency."
            )
        if not is_positive_integer(decimation):
            raise ValueError("decimation must be a positive integer.")
        if smoothing_time is not None and (
            not np.isfinite(smoothing_time) or smoothing_time <= 0
        ):
            raise ValueError("smoothing_time must be finite and positive.")
        if smoothing_step is not None and (
            smoothing_time is None
            or not np.isfinite(smoothing_step)
            or smoothing_step <= 0
        ):
            raise ValueError(
                "smoothing_step requires smoothing_time and must be finite and positive."
            )
        if not is_positive_integer(smoothing_frequency) or smoothing_frequency % 2 == 0:
            raise ValueError("smoothing_frequency must be a positive odd integer.")
        if smoothing_kernel not in {"boxcar", "hann"}:
            raise ValueError("smoothing_kernel must be 'boxcar' or 'hann'.")
        if padding_mode not in {"constant", "reflect", "edge"}:
            raise ValueError("padding_mode must be 'constant', 'reflect', or 'edge'.")
        if padding_mode == "reflect" and data.shape[0] < 2:
            raise ValueError("padding_mode='reflect' requires at least 2 time samples.")
        if edge_mode not in {"keep", "nan", "trim"}:
            raise ValueError("edge_mode must be 'keep', 'nan', or 'trim'.")

        self._time_series = _immutable_array_snapshot(data)
        self.sampling_frequency = float(sampling_frequency)
        self._frequencies = _immutable_array_snapshot(frequency_values)
        self._n_cycles = _immutable_array_snapshot(cycle_values)
        self.decimation = int(decimation)
        self.smoothing_time = smoothing_time
        self.smoothing_step = smoothing_step
        self.smoothing_frequency = int(smoothing_frequency)
        self.smoothing_kernel = smoothing_kernel
        self.padding_mode = padding_mode
        self.edge_mode = edge_mode
        self.zero_mean = bool(zero_mean)
        self.start_time = float(start_time)

        half_widths = np.maximum(
            1,
            np.ceil(
                5
                * cycle_values
                / (2 * np.pi * frequency_values)
                * self.sampling_frequency
            ).astype(int),
        )
        self._edge_half_width_samples = _immutable_array_snapshot(half_widths)
        sample_indices = np.arange(0, data.shape[0], self.decimation)
        validity = (sample_indices[:, np.newaxis] >= half_widths) & (
            sample_indices[:, np.newaxis] < data.shape[0] - half_widths
        )
        if edge_mode == "trim":
            keep = np.all(validity, axis=1)
            if not np.any(keep):
                raise ValueError(
                    "edge_mode='trim' leaves no samples valid for every requested "
                    "frequency; use fewer cycles, a longer record, or edge_mode='nan'."
                )
            sample_indices = sample_indices[keep]
            validity = validity[keep]
        self._sample_indices = _immutable_array_snapshot(sample_indices)
        self._base_validity = _immutable_array_snapshot(validity)

        n_decimated = len(sample_indices)
        if smoothing_time is None:
            self._smoothing_samples = 1
            self._smoothing_step_samples = 1
        else:
            output_rate = self.sampling_frequency / self.decimation
            self._smoothing_samples = int(np.around(smoothing_time * output_rate))
            step_time = smoothing_time if smoothing_step is None else smoothing_step
            self._smoothing_step_samples = int(np.around(step_time * output_rate))
            if self._smoothing_samples < 1 or self._smoothing_step_samples < 1:
                raise ValueError(
                    "smoothing_time/smoothing_step resolve to less than one "
                    "decimated sample."
                )
            if self._smoothing_samples > n_decimated:
                raise ValueError(
                    "smoothing_time is longer than the decimated wavelet record."
                )

    @property
    def frequencies(self) -> NDArray[np.floating]:
        return _readonly_array_copy(self._frequencies)

    @property
    def n_cycles(self) -> NDArray[np.floating]:
        return _readonly_array_copy(self._n_cycles)

    @property
    def n_trials(self) -> int:
        return int(self._time_series.shape[1])

    @property
    def n_signals(self) -> int:
        return int(self._time_series.shape[2])

    @property
    def time(self) -> NDArray[np.floating]:
        sample_times = self.start_time + self._sample_indices / self.sampling_frequency
        if self._smoothing_samples == 1 and self._smoothing_step_samples == 1:
            return sample_times
        return _sliding_window(
            sample_times,
            self._smoothing_samples,
            self._smoothing_step_samples,
            axis=0,
        ).mean(axis=-1)

    @property
    def edge_half_width(self) -> NDArray[np.floating]:
        """Wavelet half-support at each frequency, in seconds."""
        return _readonly_array_copy(
            self._edge_half_width_samples / self.sampling_frequency
        )

    def _smooth_frequency_axis(
        self, array: BackendArray, frequency_axis: int
    ) -> BackendArray:
        """Reflect-pad and window ``frequency_axis`` for adjacent-bin smoothing.

        Appends the smoothing-frequency window as a new trailing axis (or a
        singleton axis when no frequency smoothing is requested). Shared by
        :meth:`_windowed_validity` and :meth:`fft` so the padding convention
        (reflect, or edge for a single frequency) cannot drift between the
        validity mask and the coefficients it describes.
        """
        frequency_half_width = self.smoothing_frequency // 2
        if not frequency_half_width:
            return array[..., xp.newaxis]
        frequency_mode = "reflect" if array.shape[frequency_axis] > 1 else "edge"
        pad_width = [(0, 0)] * array.ndim
        pad_width[frequency_axis] = (frequency_half_width, frequency_half_width)
        padded = xp.pad(array, tuple(pad_width), mode=frequency_mode)
        return _sliding_window(padded, self.smoothing_frequency, axis=frequency_axis)

    def _windowed_validity(self) -> BackendArray:
        """Strict validity of every time/frequency output neighborhood.

        Recomputed on each call rather than cached: ``MorletWavelet`` does not
        enforce parameter immutability, so a cache keyed on object identity could
        silently go stale if a smoothing parameter were mutated after first
        access. The computation is a cheap boolean pad-and-window.
        """
        validity = xp.asarray(self._base_validity)
        windows = _sliding_window(
            validity,
            self._smoothing_samples,
            self._smoothing_step_samples,
            axis=0,
        )
        windows = self._smooth_frequency_axis(windows, frequency_axis=1)
        return xp.all(windows, axis=(-2, -1))

    @property
    def valid_time_frequency(self) -> NDArray[np.bool_]:
        """Mask where the full wavelet and smoothing support is in-record."""
        return _readonly_array_copy(self._windowed_validity())

    @staticmethod
    def _kernel_values(size: int, kernel: str) -> BackendArray:
        if kernel == "boxcar" or size == 1:
            return xp.ones(size, dtype=float)
        # Interior of a symmetric Hann window two samples wider, so every
        # sample in the window carries a non-zero weight (a symmetric Hann of
        # ``size`` itself has zero endpoints, and a size-3 kernel would then
        # weight only the centre sample).
        return xp.asarray(scipy_hann(size + 2, sym=True)[1:-1])

    def _smoothing_kernel_values(self) -> BackendArray:
        """Time-by-frequency smoothing weights, shape (n_time_window, n_freq_window)."""
        time_weights = self._kernel_values(
            self._smoothing_samples, self.smoothing_kernel
        )
        frequency_weights = self._kernel_values(
            self.smoothing_frequency, self.smoothing_kernel
        )
        return time_weights[:, xp.newaxis] * frequency_weights[xp.newaxis, :]

    @property
    def observation_weights(self) -> NDArray[np.floating]:
        """Weights consumed by :class:`Connectivity` for local expectations."""
        kernel = self._smoothing_kernel_values().reshape(1, 1, -1, 1, 1)
        shape = (
            len(self.time),
            self.n_trials,
            self._smoothing_samples * self.smoothing_frequency,
            len(self._frequencies),
            1,
        )
        weights = xp.broadcast_to(kernel, shape).copy()
        if self.edge_mode == "nan":
            weights *= self._windowed_validity()[
                :, xp.newaxis, xp.newaxis, :, xp.newaxis
            ]
        return _readonly_array_copy(weights)

    def fft(self) -> NDArray[np.complexfloating]:
        # Pad once to the widest wavelet and transform the data once; each
        # frequency then costs one kernel FFT, a multiply, and an inverse FFT.
        # Padding wider than a given wavelet needs does not change its 'valid'
        # output: the extra samples never enter that wavelet's support.
        n_time_samples = self._time_series.shape[0]
        max_half_width = int(xp.max(self._edge_half_width_samples))
        padded = xp.pad(
            self._time_series,
            ((max_half_width, max_half_width), (0, 0), (0, 0)),
            mode=self.padding_mode,
        )
        n_fft = next_fast_len(padded.shape[0] + 2 * max_half_width)
        # The wavelets are complex128; transform the data at that precision so
        # a float32 record is not convolved in single precision.
        data_spectrum = fft(padded.astype(xp.float64, copy=False), n=n_fft, axis=0)
        # Unit-energy wavelet response scaled to a one-sided power spectral
        # density: |coefficient|^2 is in signal^2 / Hz and, like Multitaper's
        # power, counts the negative-frequency half of a real signal (the
        # factor 2). This is FieldTrip's convention; MNE omits both factors.
        scale = xp.sqrt(2.0 / self.sampling_frequency)

        coefficients: list[BackendArray] = []
        for frequency, cycles, half_width in zip(
            self._frequencies,
            self._n_cycles,
            self._edge_half_width_samples,
            strict=True,
        ):
            sigma = cycles / (2 * xp.pi * frequency)
            half_width = int(half_width)
            wavelet_time = (
                xp.arange(-half_width, half_width + 1) / self.sampling_frequency
            )
            oscillation = xp.exp(2j * xp.pi * frequency * wavelet_time)
            gaussian = xp.exp(-(wavelet_time**2) / (2 * sigma**2))
            if self.zero_mean:
                oscillation = oscillation - xp.exp(
                    -0.5 * (2 * xp.pi * frequency * sigma) ** 2
                )
            wavelet = oscillation * gaussian
            wavelet = wavelet / xp.sqrt(xp.sum(xp.abs(wavelet) ** 2))
            kernel = xp.conjugate(wavelet[::-1])
            kernel_spectrum = fft(kernel, n=n_fft)[:, xp.newaxis, xp.newaxis]
            convolved = ifft(data_spectrum * kernel_spectrum, axis=0)
            # The 'valid' output centred on original sample i sits at
            # full-convolution index i + max_half_width + half_width.
            start = max_half_width + half_width
            coefficient = convolved[start : start + n_time_samples] * scale
            coefficients.append(coefficient[self._sample_indices])

        transformed = xp.stack(coefficients, axis=2)
        windows = _sliding_window(
            transformed,
            self._smoothing_samples,
            self._smoothing_step_samples,
            axis=0,
        )
        windows = self._smooth_frequency_axis(windows, frequency_axis=2)
        windows = xp.transpose(windows, (0, 1, 4, 5, 2, 3))
        return windows.reshape(
            windows.shape[0],
            windows.shape[1],
            self._smoothing_samples * self.smoothing_frequency,
            windows.shape[4],
            windows.shape[5],
        )

    def _provenance_metadata(self) -> dict[str, Any]:
        return {
            "decimation": self.decimation,
            "frequencies_json": str(to_numpy(self._frequencies).tolist()),
            "n_cycles_json": str(to_numpy(self._n_cycles).tolist()),
            "n_signals": self.n_signals,
            "n_trials": self.n_trials,
            "sampling_frequency": self.sampling_frequency,
            "edge_mode": self.edge_mode,
            "padding_mode": self.padding_mode,
            "smoothing_frequency": self.smoothing_frequency,
            "smoothing_kernel": self.smoothing_kernel,
            "smoothing_step": self.smoothing_step
            if self.smoothing_step is not None
            else "None",
            "smoothing_time": self.smoothing_time
            if self.smoothing_time is not None
            else "None",
            "zero_mean": self.zero_mean,
        }


def prepare_time_series(
    time_series: NDArray[np.floating], axis: str | None = None
) -> NDArray[np.floating]:
    """
    Convert time series data to the 3D format required by Multitaper.

    This helper function ensures your data has the correct shape
    (n_time_samples, n_trials, n_signals) for spectral analysis.

    Parameters
    ----------
    time_series : NDArray[floating]
        Input time series data with 1D, 2D, or 3D shape.
    axis : {"signals", "trials"}, optional
        For 2D input, specify which axis represents the second dimension:
        - "signals": shape is (n_time_samples, n_signals), adds trials axis
        - "trials": shape is (n_time_samples, n_trials), adds signals axis
        Required for 2D input, ignored for 1D and 3D input.

    Returns
    -------
    time_series_3d : NDArray[floating], shape (n_time_samples, n_trials, n_signals)
        Time series data reshaped to 3D format.

    Raises
    ------
    ValueError
        If input is 2D and axis parameter is not provided.
        If axis is not "signals" or "trials".
        If input has more than 3 dimensions.

    Examples
    --------
    Single-trial EEG/LFP recording with multiple channels:

    >>> import numpy as np
    >>> # Load continuous EEG: 5 seconds at 1000 Hz, 64 channels
    >>> eeg_data = np.random.randn(5000, 64)  # Shape: (n_time, n_channels)
    >>> eeg_3d = prepare_time_series(eeg_data, axis="signals")
    >>> eeg_3d.shape
    (5000, 1, 64)

    Multiple trials of a single electrode:

    >>> # 20 trials of one LFP channel, 2 seconds each at 1000 Hz
    >>> lfp_trials = np.random.randn(2000, 20)  # Shape: (n_time, n_trials)
    >>> lfp_3d = prepare_time_series(lfp_trials, axis="trials")
    >>> lfp_3d.shape
    (2000, 20, 1)

    Single time series (e.g., spike times converted to continuous):

    >>> # One neuron's firing rate over time
    >>> firing_rate = np.random.randn(1000)
    >>> firing_rate_3d = prepare_time_series(firing_rate)
    >>> firing_rate_3d.shape
    (1000, 1, 1)

    Already properly formatted (pass-through):

    >>> # Epoched data from MNE or similar: 10 trials, 5 channels, 100 timepoints
    >>> epoched_data = np.random.randn(100, 10, 5)
    >>> result = prepare_time_series(epoched_data)
    >>> result.shape
    (100, 10, 5)

    Notes
    -----
    **Common mistake:** Using 2D data without specifying the axis parameter.
    A 2D array (100, 5) could mean either:
    - 100 time points × 5 signals (1 trial) → use axis="signals"
    - 100 time points × 5 trials (1 signal) → use axis="trials"

    You must explicitly specify which interpretation is correct.

    See Also
    --------
    Multitaper : Multitaper spectral analysis class
    """
    time_series_array = xp.asarray(time_series)
    ndim = time_series_array.ndim

    if ndim == 1:
        # Single time series: (n_time,) → (n_time, 1, 1)
        return time_series_array[:, xp.newaxis, xp.newaxis]

    elif ndim == 2:
        # Ambiguous case - require explicit axis specification
        if axis is None:
            raise ValueError(
                "For 2D input, you must specify the 'axis' parameter.\n"
                f"Input shape: {time_series_array.shape}\n"
                "\n"
                "Specify:\n"
                "  - axis='signals' if shape is (n_time_samples, n_signals)\n"
                "  - axis='trials' if shape is (n_time_samples, n_trials)\n"
                "\n"
                "Example:\n"
                "  >>> # 5 EEG channels, 1 trial:\n"
                "  >>> data_3d = prepare_time_series(data, axis='signals')\n"
                "  >>> # 5 trials, 1 channel:\n"
                "  >>> data_3d = prepare_time_series(data, axis='trials')"
            )

        if axis == "signals":
            # (n_time, n_signals) → (n_time, 1, n_signals)
            return time_series_array[:, xp.newaxis, :]
        elif axis == "trials":
            # (n_time, n_trials) → (n_time, n_trials, 1)
            return time_series_array[:, :, xp.newaxis]
        else:
            raise ValueError(
                f"axis must be either 'signals' or 'trials', got: {axis!r}"
            )

    elif ndim == 3:
        # Already in correct format
        return time_series_array

    else:
        raise ValueError(
            f"Expected 1D, 2D, or 3D array, got {ndim}D array with shape "
            f"{time_series_array.shape}"
        )


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
    >>> import numpy as np
    >>> a = np.array([1, 2, 3, 4, 5])
    >>> _sliding_window(a, window_size=3)
    array([[1, 2, 3],
           [2, 3, 4],
           [3, 4, 5]])
    >>> _sliding_window(a, window_size=3, step_size=2)
    array([[1, 2, 3],
           [3, 4, 5]])

    """
    # Validate before normalizing: ``axis % data.ndim`` would otherwise wrap an
    # out-of-range axis onto a real one (e.g. axis=2 -> 0 on a 2-D array),
    # windowing the wrong dimension silently instead of raising.
    if not -data.ndim <= axis < data.ndim:
        raise ValueError(
            f"axis {axis} is out of bounds for an array of rank {data.ndim}."
        )
    # A non-positive step would pass a negative slice step below (reversing the
    # windows) or an empty step, neither of which is a forward slide.
    if step_size < 1:
        raise ValueError(f"step_size must be a positive integer, got {step_size}.")
    # ``sliding_window_view`` appends the length-``window_size`` window axis at
    # the end and validates bounds/shape, which NumPy recommends over the
    # lower-level ``as_strided`` used previously. It only produces unit-step
    # windows, so subsample the windowed axis to apply ``step_size``. Normalize
    # a negative ``axis`` against the input rank first, because the view adds a
    # trailing axis and would otherwise shift a negative index onto it.
    axis = axis % data.ndim
    strided = xp.lib.stride_tricks.sliding_window_view(data, window_size, axis=axis)
    if step_size != 1:
        subsample = [slice(None)] * strided.ndim
        subsample[axis] = slice(None, None, step_size)
        strided = strided[tuple(subsample)]

    return strided.copy() if is_copy else strided


def _multitaper_fft(
    tapers: NDArray[np.floating],
    time_series: NDArray[np.floating],
    n_fft_samples: int,
    sampling_frequency: float,
    axis: int = -2,
    workers: int | None = None,
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
    workers : int, optional
        Number of parallel worker threads for SciPy's CPU FFT (see
        ``scipy.fft.fft``; ``-1`` uses all cores). ``None`` keeps SciPy's default
        (single-threaded). Ignored on the GPU backend, whose FFT has no such
        parameter and is already parallel.

    Returns
    -------
    fourier_coefficients : array_like,
        shape (n_windows, n_trials, n_tapers n_fft_samples, n_signals)

    """
    projected_time_series = (
        time_series[..., xp.newaxis] * tapers[xp.newaxis, xp.newaxis, ...]
    )
    # Only SciPy's CPU FFT accepts ``workers``; cupyx's FFT does not, so pass it
    # only when a worker count is requested and we are on the CPU backend.
    fft_kwargs = {}
    if workers is not None and not is_gpu_enabled():
        fft_kwargs["workers"] = workers
    return (
        fft(projected_time_series, n=n_fft_samples, axis=axis, **fft_kwargs)
        / sampling_frequency
    )


def _apply_adaptive_taper_weights(
    coefficients: NDArray[np.complexfloating],
    eigenvalues: NDArray[np.floating],
    noise_power_spectral_density: NDArray[np.floating],
    *,
    max_iterations: int,
    tolerance: float,
) -> NDArray[np.complexfloating]:
    """Apply Thomson adaptive DPSS weights to Fourier coefficients.

    The iterative spectrum estimate follows Thomson's frequency- and
    signal-specific weighting. Returned weights are RMS-normalized across the
    taper axis, allowing :class:`Connectivity`'s ordinary taper mean to compute
    the corresponding weighted auto- and cross-spectra without changing their
    scale.

    ``noise_power_spectral_density`` must already be on the same
    power-spectral-density scale as ``|coefficients| ** 2`` (see the caller in
    :meth:`Multitaper.fft`); a raw time-domain variance would over-weight the
    noise term by a factor of the sampling frequency.
    """
    import warnings

    taper_power = xp.abs(coefficients) ** 2
    n_initial = min(2, coefficients.shape[2])
    spectrum = xp.mean(taper_power[:, :, :n_initial, :, :], axis=2)

    concentration = eigenvalues[xp.newaxis, xp.newaxis, :, xp.newaxis, xp.newaxis]
    root_concentration = xp.sqrt(concentration)
    noise = noise_power_spectral_density[:, :, xp.newaxis, xp.newaxis, :]
    eps = xp.finfo(taper_power.dtype).eps

    weights = xp.ones_like(taper_power, dtype=taper_power.dtype)
    for _ in range(max_iterations):
        expanded_spectrum = spectrum[:, :, xp.newaxis, :, :]
        denominator = concentration * expanded_spectrum + (1.0 - concentration) * noise
        weights = xp.divide(
            root_concentration * expanded_spectrum,
            denominator,
            out=xp.zeros_like(taper_power, dtype=taper_power.dtype),
            where=xp.abs(denominator) > eps,
        )
        weight_power = weights**2
        weight_sum = xp.sum(weight_power, axis=2)
        updated = xp.divide(
            xp.sum(weight_power * taper_power, axis=2),
            weight_sum,
            out=xp.zeros_like(spectrum),
            where=weight_sum > eps,
        )
        scale = xp.maximum(xp.abs(spectrum), eps)
        if bool(xp.all(xp.abs(updated - spectrum) <= tolerance * scale)):
            spectrum = updated
            break
        spectrum = updated
    else:
        warnings.warn(
            f"Adaptive taper weighting did not converge within {max_iterations} "
            f"iterations (tolerance={tolerance:g}). The returned spectrum uses the "
            "last iterate, which may bias power and connectivity estimates. "
            "Increase adaptive_max_iterations or relax adaptive_tolerance.",
            UserWarning,
            stacklevel=2,
        )

    rms = xp.sqrt(xp.mean(weights**2, axis=2, keepdims=True))
    normalized_weights = xp.divide(
        weights,
        rms,
        out=xp.ones_like(weights),
        where=rms > eps,
    )
    return coefficients * normalized_weights


def dpss_windows(
    n_time_samples_per_window: int,
    time_halfbandwidth_product: float,
    n_tapers: int,
    is_low_bias: bool = True,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Compute Discrete Prolate Spheroidal Sequences.

    Returns the DPSS (Slepian) tapers of orders [0, n_tapers-1] for a given
    time-halfbandwidth product NW and window length
    ``n_time_samples_per_window``, together with their spectral-concentration
    ratios (eigenvalues).

    Delegates to :func:`scipy.signal.windows.dpss`, which solves the same
    symmetric tridiagonal eigenproblem (Percival & Walden 1993) via LAPACK. The
    ``sym=True`` / ``norm=2`` options reproduce the symmetric, unit-L2-norm
    convention used here, matching the previous vendored NiTime/MNE
    implementation to floating-point tolerance. It is CPU-only (banded linear
    algebra); the result is moved to the active array namespace afterward, as
    before.

    Parameters
    ----------
    n_time_samples_per_window : int
        Sequence length.
    time_halfbandwidth_product : float, unitless
        Standardized half bandwidth NW.
    n_tapers : int
        Number of DPSS windows to return.
    is_low_bias : bool
        Keep only tapers with eigenvalues > MIN_EIGENVALUE_THRESHOLD (0.9).

    Returns
    -------
    tapers, eigenvalues : tuple
        ``tapers`` has shape (n_tapers, n_time_samples_per_window);
        ``eigenvalues`` has shape (n_tapers,).

    Notes
    -----
    Tridiagonal form of DPSS calculation from:
    Slepian, D. Prolate spheroidal wave functions, Fourier analysis, and
    uncertainty V: The discrete case. Bell System Technical Journal,
    Volume 57 (1978), 1371430
    """
    # Reject a fractional n_tapers before coercion: silently truncating (e.g.
    # 2.9 -> 2) would disagree with the reported Multitaper.n_tapers metadata.
    if not np.isfinite(n_tapers) or int(n_tapers) != n_tapers:
        raise ValueError(f"n_tapers must be an integer, got {n_tapers}.")
    n_tapers = int(n_tapers)
    if n_time_samples_per_window < 2:
        raise ValueError(
            f"n_time_samples_per_window must be >= 2 for a multitaper "
            f"decomposition, got {n_time_samples_per_window}. A single-sample "
            f"window carries no spectral information."
        )
    if not 0 < time_halfbandwidth_product < n_time_samples_per_window / 2:
        raise ValueError(
            f"time_halfbandwidth_product (NW={time_halfbandwidth_product}) must "
            f"satisfy 0 < NW < n_time_samples_per_window / 2 "
            f"(= {n_time_samples_per_window / 2}). Otherwise the concentration "
            f"bandwidth reaches or exceeds Nyquist and the result is not a valid "
            f"set of DPSS tapers (their concentration ratios can exceed 1). Use a "
            f"longer window or a smaller time_halfbandwidth_product."
        )
    if not 1 <= n_tapers <= n_time_samples_per_window:
        raise ValueError(
            f"n_tapers must satisfy 1 <= n_tapers <= n_time_samples_per_window "
            f"(= {n_time_samples_per_window}), got {n_tapers}."
        )
    if n_time_samples_per_window == 2 and n_tapers == 2:
        # scipy.signal.windows.dpss cannot disambiguate the sign of the second
        # (antisymmetric) taper for a two-sample window: its heuristic keeps the
        # first sample with |value|^2 above max(1e-7, 1 / n_time_samples) = 0.5,
        # but the length-two antisymmetric taper is [1, -1] / sqrt(2), whose
        # samples are exactly 0.5, so none clear the threshold and SciPy raises a
        # bare IndexError (both 1.10.x and current releases). A two-sample window
        # cannot support a second taper usefully anyway, so reject the pair with
        # an actionable message rather than surface SciPy's internal error.
        raise ValueError(
            "A two-sample window (n_time_samples_per_window=2) supports only a "
            "single DPSS taper; request n_tapers=1 or use a longer window."
        )

    tapers, eigenvalues = scipy_dpss(
        n_time_samples_per_window,
        time_halfbandwidth_product,
        n_tapers,
        sym=True,
        norm=2,
        return_ratios=True,
    )
    tapers = xp.asarray(tapers)
    eigenvalues = xp.asarray(eigenvalues)

    return (
        _get_low_bias_tapers(tapers, eigenvalues)
        if is_low_bias
        else (tapers, eigenvalues)
    )


def _get_low_bias_tapers(
    tapers: NDArray[np.floating], eigenvalues: NDArray[np.floating]
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    is_low_bias = eigenvalues > MIN_EIGENVALUE_THRESHOLD
    if not xp.any(is_low_bias):
        logger.warning("Could not properly use low_bias, keeping lowest-bias taper")
        is_low_bias = xp.array([xp.argmax(eigenvalues)])
    return tapers[is_low_bias, :], eigenvalues[is_low_bias]


def detrend(
    data: NDArray[np.floating],
    axis: int = -1,
    type: str = "linear",
    bp: int | list[int] | NDArray[np.integer] = 0,
    overwrite_data: bool = False,
) -> NDArray[np.floating]:
    """
    Remove linear trend along axis from data.

    Thin wrapper that validates ``type``/``bp`` (raising actionable errors) and
    delegates the computation to ``scipy.signal.detrend`` on CPU or
    ``cupyx.scipy.signal.detrend`` on GPU.

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
    >>> import numpy as np
    >>> from scipy import signal
    >>> rng = np.random.default_rng(0)
    >>> npoints = 1000
    >>> noise = rng.standard_normal(npoints)
    >>> x = 3 + 2 * np.linspace(0, 1, npoints) + noise
    >>> # Removing the linear trend leaves (approximately) the original noise.
    >>> bool(np.abs(signal.detrend(x) - noise).max() < 0.2)
    True
    """
    if type not in ["linear", "l", "constant", "c"]:
        raise ValueError(
            f"Invalid trend type '{type}' is not supported.\n"
            f"The detrend function only supports linear and constant detrending.\n"
            f"Valid options are:\n"
            f"  - 'linear' or 'l': Remove linear trend (best-fit line)\n"
            f"  - 'constant' or 'c': Remove mean (DC offset)\n"
            f"Example: detrend(data, type='linear')"
        )
    # Normalize the short aliases so the backend (which documents only the long
    # forms) is never handed 'l'/'c'.
    type = "linear" if type in ["linear", "l"] else "constant"
    data = xp.asarray(data)
    # Validate breakpoints up front (linear only) so the error names the
    # offending value and the data length; the backend raises a terser message.
    if type == "linear":
        N = data.shape[axis]
        # Breakpoints are indices into the detrended axis; the documented valid
        # range is [0, N). Normalize the user's input (int, list, tuple, or
        # numpy/cupy array) to a single array up front, then validate BOTH bounds
        # against it directly -- rather than the 0/N-padded array previously used
        # to build segments, whose padding always contained N (so a breakpoint at
        # exactly N slipped past a ``> N`` check) and which never rejected
        # negative indices (they reached the backend as a cryptic error).
        bp_values = xp.atleast_1d(xp.asarray(bp))

        def _to_list(a: NDArray) -> list:
            # Works for both numpy and cupy arrays.
            return to_numpy(a).tolist()

        out_of_range = bp_values[(bp_values < 0) | (bp_values >= N)]
        if out_of_range.size:
            raise ValueError(
                f"Breakpoint value(s) {_to_list(out_of_range)} are outside the "
                f"valid range [0, {N}).\n"
                f"Data has {N} samples along axis {axis}; breakpoints are indices "
                f"into that axis and must satisfy 0 <= breakpoint < {N}.\n"
                f"Check your breakpoints: {_to_list(bp_values)}"
            )
    # Delegate the least-squares/mean removal to SciPy (CPU) or CuPy (GPU),
    # which implement the same computation. Their detrend signatures match.
    return _backend_detrend(
        data, axis=axis, type=type, bp=bp, overwrite_data=overwrite_data
    )
