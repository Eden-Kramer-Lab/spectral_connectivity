"""Transforms time domain signals to the frequency domain."""

import contextlib
import warnings
from collections.abc import Callable, Hashable, Iterator, Sequence
from logging import getLogger
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    NamedTuple,
    Protocol,
    TypedDict,
    TypeVar,
    runtime_checkable,
)

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh as scipy_eigh
from scipy.signal.windows import dpss as scipy_dpss
from scipy.signal.windows import hann as scipy_hann
from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator
from scipy.sparse.linalg import eigsh as scipy_eigsh

from spectral_connectivity._array_utils import _divide_where
from spectral_connectivity._backend import (
    ON_GPU,
    default_fft_length,
    fft,
    fftfreq,
    ifft,
    next_fast_len,
    rfft,
    xp,
)
from spectral_connectivity._backend import detrend as _backend_detrend
from spectral_connectivity.utils import (
    TIME_SERIES_RAGGED_FIX,
    BackendArray,
    _validate_sampling_frequency,
    both_spellings_message,
    close_names,
    did_you_mean,
    is_positive_integer,
    mark_readonly_if_supported,
    numeric_array,
    reject_sampling_frequency_aliases,
    stacklevel_outside_package,
    to_numpy,
    warn_deprecated,
)

logger = getLogger(__name__)

# Element type of a dtype-preserving array helper (real or complex input).
_ScalarT = TypeVar("_ScalarT", bound=np.generic)

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
# Smallest NW that gives two tapers: floor(TAPER_MULTIPLIER * 1.5) - 1 = 2.
MIN_TIME_HALFBANDWIDTH_FOR_TWO_TAPERS = 1.5


def _duration_to_samples(duration: float, sampling_frequency: float) -> int:
    """Nearest whole number of samples in ``duration`` seconds.

    Rounds rather than truncates: ``step / fs * fs`` can land just below
    ``step`` in floating point (e.g. 0.57 s at 100 Hz gives 56.99999...), and
    ``int()`` would truncate to 56.
    """
    return int(np.around(duration * sampling_frequency))


def _resolve_sample_count(
    explicit: int | None,
    duration: float | None,
    sampling_frequency: float,
    mismatch_message: str,
) -> int | None:
    """Resolve Welch's segment length from an explicit count or a duration.

    Returns the sample count when either ``explicit`` or ``duration`` is given
    (cross-checking that they agree when both are), or ``None`` when neither is
    provided so the caller can apply its own default.
    """
    if explicit is not None:
        samples = int(explicit)
        if (
            duration is not None
            and _duration_to_samples(duration, sampling_frequency) != samples
        ):
            raise ValueError(mismatch_message)
        return samples
    if duration is not None:
        return _duration_to_samples(duration, sampling_frequency)
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


@reject_sampling_frequency_aliases
def estimate_frequency_resolution(
    sampling_frequency: float,  # noqa: ARG001 -- public API; resolution does not use it
    time_window_duration: float,
    time_halfbandwidth_product: float,
) -> float:
    """
    Estimate the frequency resolution for given multitaper parameters.

    Frequency resolution here is the full DPSS concentration bandwidth,
    ``2 * time_halfbandwidth_product / time_window_duration``, in Hz. It
    describes spectral smoothing and is distinct from FFT bin spacing;
    zero-padding changes the grid without narrowing this bandwidth.

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
    if `is_low_bias=True` (default), which excludes tapers with eigenvalues at or
    below MIN_EIGENVALUE_THRESHOLD (0.9), with a warning.

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


def _ceil_hundredths(value: float) -> float:
    """Round ``value`` up to two decimals, ignoring floating-point error."""
    return float(np.ceil(round(value * 100, 6)) / 100)


@reject_sampling_frequency_aliases
def suggest_parameters(
    sampling_frequency: float,
    signal_duration: float,
    frequency_resolution: float | None = None,
    desired_n_tapers: int | None = None,
    *,
    desired_freq_resolution: float | None = None,
) -> MultitaperParameters:
    """
    Suggest appropriate multitaper parameters for your analysis.

    This helper function recommends parameters based on your data characteristics
    and analysis goals. It helps answer the common question: "What parameters
    should I use for my data?"

    ``frequency_resolution`` (the target you pass) and the returned
    ``frequency_resolution`` (the resolution achieved) are both full DPSS
    concentration bandwidths in Hz, ``2 * NW / time_window_duration``, not FFT
    bin spacing.

    .. versionchanged:: 3.0
       The target resolution is named ``frequency_resolution``, as in
       :class:`Multitaper` and :func:`estimate_frequency_resolution`; it keeps
       its third position, so positional calls are unchanged.
       ``desired_freq_resolution=`` still works through 3.x with a
       ``DeprecationWarning`` and is removed in 4.0.

    Parameters
    ----------
    sampling_frequency : float
        Sampling rate in Hz of your data.
    signal_duration : float
        Total duration in seconds of your signal.
    frequency_resolution : float, optional
        Target frequency resolution in Hz. If specified, parameters will be
        chosen to achieve approximately this resolution. Given with
        ``desired_n_tapers``, it takes precedence (see Warns).
    desired_n_tapers : int, optional
        Target number of tapers for variance reduction. If specified,
        time_halfbandwidth_product will be chosen to achieve this.
        ``desired_n_tapers`` is a target; the returned ``n_tapers`` is what
        you get. Ignored, with a warning, when ``frequency_resolution`` is
        also given.
    desired_freq_resolution : float, optional
        Deprecated name of ``frequency_resolution``, by keyword only; removed
        in 4.0. Giving both raises ``ValueError``.

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
        If both frequency_resolution and desired_n_tapers are specified.
        In this case, frequency_resolution takes precedence and
        desired_n_tapers is ignored.
    UserWarning
        If keeping frequency_resolution with at least 3 time windows leaves
        fewer than 2 tapers; use a longer signal or a coarser resolution.

    Notes
    -----
    **Default behavior** (no targets specified):
    - Uses NW=3 (balanced trade-off)
    - Sets window duration to capture ~5 time windows
    - Aims for reasonable frequency and time resolution

    **With frequency_resolution:**
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
    ...     frequency_resolution=2.0,  # Want 2 Hz resolution
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
    if desired_freq_resolution is not None:
        if frequency_resolution is not None:
            msg = both_spellings_message(
                "suggest_parameters", "frequency_resolution", "desired_freq_resolution"
            )
            raise ValueError(msg)
        frequency_resolution = desired_freq_resolution
    # Validate inputs
    if frequency_resolution is not None and desired_n_tapers is not None:
        warnings.warn(
            "Both 'frequency_resolution' and 'desired_n_tapers' were specified. "
            "This is typically not recommended as they have competing effects on the analysis. "
            "Using 'frequency_resolution' and ignoring 'desired_n_tapers'.",
            UserWarning,
            stacklevel=stacklevel_outside_package(),
        )
        desired_n_tapers = None

    # Default: balanced parameters (NW=3)
    if frequency_resolution is None and desired_n_tapers is None:
        time_halfbandwidth_product = 3.0
        # Use ~20% of signal duration as window, aim for ~5 windows
        time_window_duration = min(signal_duration / 5.0, signal_duration * 0.2)
        # But ensure at least 0.5s window for reasonable freq resolution
        time_window_duration = max(time_window_duration, 0.5)
        # And don't exceed signal duration
        time_window_duration = min(time_window_duration, signal_duration)

    # User wants specific frequency resolution
    elif frequency_resolution is not None:
        # Start with NW=3 (typical balanced value)
        time_halfbandwidth_product = 3.0

        # Calculate required window duration: T = 2*NW / Δf
        time_window_duration = (
            TAPER_MULTIPLIER * time_halfbandwidth_product / frequency_resolution
        )

        # Check if this is achievable with the signal duration
        if time_window_duration > signal_duration:
            msg = (
                f"Cannot achieve desired frequency resolution of {frequency_resolution} Hz "
                f"with signal duration of {signal_duration}s.\n"
                "\n"
                f"Required window duration: {time_window_duration:.2f}s\n"
                f"Available signal duration: {signal_duration:.2f}s\n"
                "\n"
                "To achieve this resolution, you need either:\n"
                f"  - Longer signal (at least {_ceil_hundredths(time_window_duration):.2f}s)\n"
                f"  - Coarser frequency resolution (at least "
                f"{_ceil_hundredths(TAPER_MULTIPLIER * time_halfbandwidth_product / signal_duration):.2f}"
                " Hz)"
            )
            raise ValueError(msg)

        # If the window would give fewer than 3 time windows, shorten it and lower
        # NW so the target resolution (2 * NW / T) is kept, at the cost of fewer
        # tapers (noisier estimates). Only the NW >= 1 floor below coarsens it.
        min_n_windows = 3
        max_window_for_min_windows = signal_duration / min_n_windows
        if time_window_duration > max_window_for_min_windows:
            # Adjust NW to give us at least min_n_windows
            time_window_duration = max_window_for_min_windows
            # Recalculate NW to achieve target resolution with this window.
            # The taper count is floor(2 * NW) - 1, so floating-point error just
            # below a whole 2 * NW, such as 30 Hz * 0.1 s = 2.9999999999999996,
            # would cost a taper: only such a value is raised to that whole.
            twice_nw = frequency_resolution * time_window_duration
            whole = np.ceil(twice_nw)
            if twice_nw < whole and np.isclose(twice_nw, whole, rtol=1e-12, atol=0.0):
                twice_nw = type(twice_nw)(whole)
            time_halfbandwidth_product = twice_nw / 2.0
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
        msg = "Internal error: unexpected parameter combination"
        raise ValueError(msg)

    # Calculate derived parameters
    n_tapers = estimate_n_tapers(time_halfbandwidth_product)
    if frequency_resolution is not None and n_tapers < 2:
        # Only the shortened-window path above can leave one taper. Two tapers
        # need resolution * window >= this bandwidth product (delta_f = 2 NW / T).
        two_taper_bandwidth = TAPER_MULTIPLIER * MIN_TIME_HALFBANDWIDTH_FOR_TWO_TAPERS
        # Round the "at least" values up so that passing them back as printed
        # meets the bound.
        min_signal_duration = _ceil_hundredths(
            min_n_windows * two_taper_bandwidth / frequency_resolution
        )
        min_freq_resolution = _ceil_hundredths(
            two_taper_bandwidth * min_n_windows / signal_duration
        )
        warnings.warn(
            f"A {frequency_resolution} Hz resolution with at least "
            f"{min_n_windows} time windows of a {signal_duration} s signal "
            f"leaves {n_tapers} taper (time_halfbandwidth_product="
            f"{time_halfbandwidth_product:.2f}), so the estimate is not "
            "averaged across tapers and will be noisy.\n"
            "For at least 2 tapers, use either:\n"
            f"  - a longer signal (at least {min_signal_duration:.2f} s), or\n"
            f"  - a coarser frequency resolution (at least {min_freq_resolution:.2f} Hz).",
            UserWarning,
            stacklevel=stacklevel_outside_package(),
        )
    achieved_resolution = estimate_frequency_resolution(
        sampling_frequency, time_window_duration, time_halfbandwidth_product
    )
    n_time_windows = int(
        np.floor(signal_duration / time_window_duration)
    )  # Non-overlapping estimate
    nyquist_frequency = sampling_frequency / 2.0
    # Warned only once the call is valid, so a rejected call shows its error.
    if desired_freq_resolution is not None:
        warn_deprecated(
            "suggest_parameters(desired_freq_resolution=...)",
            "suggest_parameters(frequency_resolution=...)",
            "gives the same suggestion, under the name Multitaper uses",
        )

    return {
        "sampling_frequency": sampling_frequency,
        "time_halfbandwidth_product": time_halfbandwidth_product,
        "time_window_duration": time_window_duration,
        "n_tapers": n_tapers,
        "frequency_resolution": achieved_resolution,
        "n_time_windows": n_time_windows,
        "nyquist_frequency": nyquist_frequency,
    }


def _finite_scalar_start_time(start_time: Any) -> float:
    """``start_time`` as a finite float, squeezing a single-element array.

    The time axis is one-dimensional, so a per-trial start time cannot be
    represented. A single-element array (``time[0]`` of a column time axis) is
    one start time and is accepted.
    """
    value = to_numpy(start_time)
    if value.size == 1:
        try:
            scalar = float(value.reshape(()))
        except (TypeError, ValueError):
            scalar = float("nan")
        if np.isfinite(scalar):
            return scalar
    msg = (
        f"start_time must be a finite scalar (the time in seconds of the first "
        f"sample), got {start_time!r}. The time axis is one-dimensional, so a "
        "per-trial start time is not supported; align trials before stacking "
        "them or analyze them separately."
    )
    raise ValueError(msg)


def _immutable_array_snapshot(value: Any) -> BackendArray:
    """Copy an input array and make the owned snapshot read-only when supported."""
    return mark_readonly_if_supported(xp.array(value, copy=True))


def _readonly_array_copy(array: BackendArray) -> BackendArray:
    """Return a detached read-only copy of an internal array snapshot."""
    return mark_readonly_if_supported(array.copy())


@runtime_checkable
class SpectralTransform(Protocol):
    """Interface a transform needs for :meth:`Connectivity.from_transform`.

    Any object with an ``fft()`` method and ``frequencies`` and ``time``
    attributes (plain attributes or properties) satisfies it; no subclassing or
    registration is needed. :class:`Multitaper`, :class:`ShortTimeFourierTransform`,
    :class:`Welch`, and :class:`MorletWavelet` all satisfy it.

    Attributes
    ----------
    frequencies : ndarray, shape (n_fft_samples,), or None
        Frequency of each bin of the ``fft()`` output, in Hz. A two-sided
        transform lists them in standard FFT order (``numpy.fft.fftfreq``); a
        one-sided transform lists non-negative, strictly increasing values.
        ``None`` makes :class:`Connectivity` use normalized frequencies in cycles per
        sample: ``numpy.fft.fftfreq(n_fft_samples)`` for a two-sided transform,
        ``numpy.linspace(0, 0.5, n_fft_samples)`` for a one-sided one.
    time : ndarray, shape (n_time_windows,), or None
        Time of each time window, in seconds. ``None`` makes
        :class:`Connectivity` use the window indices.

    Notes
    -----
    ``fft()`` must return the Fourier coefficients with shape
    ``(n_time_windows, n_trials, n_tapers, n_fft_samples, n_signals)``.
    ``fft()`` must return fresh, unshared storage on each call. Neither the
    transform nor its caller may subsequently mutate that storage through any
    alias. :class:`Connectivity` keeps it without copying and marks it read-only
    where the backend supports this. The flag is only a safeguard: existing
    writable NumPy views remain writable, and CuPy has no read-only flag.
    Mutating the storage would silently change the coefficients while cached
    results keep the old values.

    The coefficients' scale sets the units of :meth:`Connectivity.power`;
    normalized measures such as coherence and the phase-lag indices do not
    depend on it, so an unscaled FFT is enough for those. For ``power()`` to be
    a power spectral density (signal² / Hz), ``|coefficient|²`` must be a
    density:

    - a two-sided transform returns ``fft(window * x) / sqrt(sampling_frequency)``
      for a unit-energy window (``sum(window**2) == 1``), as :class:`Multitaper`
      does; ``power()`` then folds it onto the non-negative frequencies,
      doubling every bin but DC and, for an even FFT length, Nyquist;
    - a one-sided transform (``is_one_sided = True``) must fold in the negative
      frequencies itself, multiplying every bin but 0 Hz and Nyquist by
      ``sqrt(2)``, because ``power()`` uses one-sided coefficients as given.
      :class:`MorletWavelet`, whose frequencies all lie strictly between 0 Hz
      and Nyquist, scales every coefficient of a unit-energy wavelet by
      ``sqrt(2 / sampling_frequency)``.

    :meth:`Connectivity.from_transform` also reads the following optional
    attributes. A transform that lacks one gets the default, which describes a
    plain two-sided FFT of independent observations, so only a transform whose
    output differs needs to define it:

    ``is_one_sided`` : bool, default False
        Whether the coefficients hold only non-negative frequencies (as for a
        wavelet transform). One-sided coefficients are used as given, without
        taking the half spectrum or doubling power, and cannot be used by the
        Wilson-factorized directed measures.
    ``observation_weights`` : ndarray, shape (n_time_windows, n_trials, n_tapers, n_fft_samples, 1), default None
        Finite, non-negative weights applied to every expectation over the
        coefficients and shared across signals, e.g. a smoothing kernel or a
        mask for invalid edge estimates or artifact samples (0 drops an
        observation). ``None`` weights all observations equally.
    ``observations_are_independent`` : bool, default True
        Whether the trial and taper observations are statistically
        independent. Set it ``False`` when that axis holds correlated
        estimates; measures whose corrections count observations then warn,
        and the jackknife refuses to leave out tapers.
    ``time_bins_are_independent`` : bool, default True
        Whether successive time windows are independent (e.g. ``False`` for
        windows overlapping by more than half). It affects only expectations that
        average over time.

    ``isinstance`` checks only that ``fft``, ``frequencies``, and ``time``
    exist (on Python 3.11 and earlier it also evaluates the two properties).
    :meth:`Connectivity.from_transform` checks the coefficients' shape and
    complex dtype, the coordinate lengths, the frequency order, and that the
    flags are bools (NumPy bools included), not methods.

    Examples
    --------
    Wrap Fourier coefficients computed elsewhere:

    >>> import numpy as np
    >>> from spectral_connectivity import Connectivity, SpectralTransform
    >>> class PrecomputedTransform:
    ...     def __init__(self, coefficients, frequencies, time):
    ...         self._coefficients = coefficients
    ...         self.frequencies = frequencies
    ...         self.time = time
    ...
    ...     def fft(self):
    ...         return self._coefficients.copy()
    >>> rng = np.random.default_rng(0)
    >>> shape = (1, 10, 1, 16, 2)  # (time windows, trials, tapers, FFT bins, signals)
    >>> coefficients = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    >>> transform = PrecomputedTransform(
    ...     coefficients, np.fft.fftfreq(16, d=1 / 500), time=np.array([0.0])
    ... )
    >>> isinstance(transform, SpectralTransform)
    True
    >>> connectivity = Connectivity.from_transform(transform)
    >>> connectivity.magnitude_squared_coherence().shape  # (time, frequency, signal, signal)
    (1, 9, 2, 2)
    """

    @property
    def frequencies(self) -> NDArray[np.floating] | None:
        """Frequency of each FFT bin, in Hz."""

    @property
    def time(self) -> NDArray[np.floating] | None:
        """Time of each time window, in seconds."""

    def fft(self) -> NDArray[np.complexfloating]:
        """Return the coefficients, ``(n_time_windows, n_trials, n_tapers, n_fft_samples, n_signals)``."""


# The axis each role is on in a transform's input.
_INPUT_AXIS_ROLES = ("time", "trial", "signal")
# The shape every transform takes its time series in.
_TRANSFORM_LAYOUT = "(n_time_samples, n_trials, n_signals)"


def _misplaced_dimensions(dimensions: Sequence[Hashable]) -> list[str]:
    """Each of ``dimensions`` whose name gives it a role on another axis of
    the ``(time, trial, signal)`` layout, as "'name' is on the role axis"."""
    # The naming rules belong to the wrapper's layer, above this module, so
    # the dependency on them stays local.
    from spectral_connectivity._input_handling import _dimension_role

    return [
        f"{dimension!r} is on the {expected} axis"
        for dimension, expected in zip(dimensions, _INPUT_AXIS_ROLES, strict=False)
        if _dimension_role(dimension) not in (None, expected)
    ]


def _check_labeled_axis_order(time_series: Any, transform: str) -> None:
    """Reject a labeled input whose dimension names put a role on another axis.

    A transform reads ``time_series`` by position. An ``xarray.DataArray``
    named, say, ``(trial, time, signal)`` would be transformed along its trial
    axis and give a plausible result of the wrong quantity, so it is refused.
    Names that say nothing of an axis's role (``dim_0``, ``session``) are not
    evidence either way and are read by position.
    """
    dimensions = getattr(time_series, "dims", None)
    # A labeled 3-D array; a Dataset also has ``dims`` but no axis order.
    if dimensions is None or getattr(time_series, "ndim", None) != len(_INPUT_AXIS_ROLES):
        return
    from spectral_connectivity._input_handling import _resolve_dataarray_dimensions

    misplaced = _misplaced_dimensions(dimensions)
    if not misplaced:
        return
    # Spell out the transpose only when the names determine it by the wrapper's
    # own rules: all three roles named once, or two named and one dimension
    # that is not a frequency axis left for the third.
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # its notice of the elimination
            order = _resolve_dataarray_dimensions(
                time_series, time_dim=None, trial_dim=None, signal_dim=None
            )
    except ValueError:
        advice = (
            "Transpose it to (time, trial, signal) order first "
            "(time_series.transpose with the dimension names in that order)"
        )
    else:
        advice = f"Transpose it first, time_series.transpose({', '.join(map(repr, order))})"
    msg = (
        f"{transform} reads time_series by position, as (time, trial, signal), "
        f"but its dimensions are named {tuple(dimensions)!r}: "
        f"{'; '.join(misplaced)}. {advice}, or pass time_series.values if the "
        "names are wrong and the axes are already in that order. "
        "multitaper_connectivity assigns axes from dimension names itself."
    )
    raise ValueError(msg)


def _reject_complex_time_series(time_series: Any, transform: str) -> None:
    """Refuse a complex-valued time series, whose spectrum is not symmetric.

    Every transform reports the non-negative frequencies, and the one-sided
    quantities built on them assume the negative frequencies mirror the
    positive ones, as they do for a real signal. ``np.iscomplexobj`` reads an
    array's dtype without converting it (CuPy and xarray included).
    """
    if np.iscomplexobj(time_series):
        msg = (
            f"{transform} got a complex-valued time series (dtype {time_series.dtype}). "
            "Its outputs cover only the "
            "non-negative frequencies and assume the negative ones mirror them, "
            "which holds only for a real signal: the negative-frequency content "
            "of a complex signal would be lost, and its positive-frequency "
            "content double-counted. If the imaginary part is zero or negligible "
            "(as after an inverse FFT), pass np.real(time_series). Otherwise pass "
            "the real and imaginary parts as separate real signals, or compute "
            "the Fourier coefficients yourself and pass them to Connectivity."
        )
        raise TypeError(msg)


def _validated_time_series(time_series: Any, transform: str) -> BackendArray:
    """A transform's input checked as every transform checks it: the axis order
    of a labeled input, then numbers in an array (a nested list converted),
    then real values."""
    _check_labeled_axis_order(time_series, transform)
    data = numeric_array(time_series, "time_series", _TRANSFORM_LAYOUT, TIME_SERIES_RAGGED_FIX)
    _reject_complex_time_series(data, transform)
    return data


def _check_time_series_is_3d(time_series: BackendArray) -> None:
    """Reject a time series that is not ``(n_time_samples, n_trials, n_signals)``.

    The message shows how to reshape 1-D and 2-D input.
    """
    if time_series.ndim != 3:
        error_msg = (
            f"Expected 3D array with shape (n_time_samples, n_trials, n_signals), "
            f"but got {time_series.ndim}D array with shape {time_series.shape}.\n"
            "\n"
        )

        if time_series.ndim == 1:
            error_msg += (
                "For a single time series, use:\n"
                "  >>> from spectral_connectivity.transforms import prepare_time_series\n"
                "  >>> time_series_3d = prepare_time_series(time_series)\n"
                "Or manually:\n"
                "  >>> time_series_3d = time_series[:, np.newaxis, np.newaxis]"
            )
        elif time_series.ndim == 2:
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
                f"Arrays with {time_series.ndim} dimensions are not supported.\n"
                "Expected shape: (n_time_samples, n_trials, n_signals)"
            )

        raise ValueError(error_msg)


def _validate_fft_workers(fft_workers: int | None) -> None:
    """Reject an ``fft_workers`` value that ``scipy.fft.fft(workers=...)`` cannot use."""
    # fft_workers is forwarded verbatim to scipy.fft.fft(workers=...): it must
    # be a nonzero integer (a negative value counts back from os.cpu_count(),
    # so -1 uses all cores) or None (SciPy's single-threaded default).
    # Validate here so a bad value names this parameter, instead of surfacing
    # a bare "workers must not be zero" ValueError or an opaque TypeError from
    # deep inside fft(). bool is an int subclass, so reject it explicitly.
    if fft_workers is not None and (
        isinstance(fft_workers, bool)
        or not isinstance(fft_workers, (int, np.integer))  # type: ignore[redundant-expr]  # user input
        or fft_workers == 0
    ):
        msg = (
            f"fft_workers must be None or a nonzero integer (the number of "
            f"parallel FFT threads; -1 uses all cores), got {fft_workers!r}. "
            f"It is forwarded to scipy.fft.fft(workers=...). Use None (the "
            f"default) for SciPy's single-threaded FFT."
        )
        raise ValueError(msg)


# The trend types :func:`detrend` accepts, with SciPy's one-letter spellings.
_DETREND_TYPES = ("constant", "c", "linear", "l")


def _validate_detrend_type(detrend_type: str | None) -> None:
    """Reject a ``detrend_type`` that :func:`detrend` would reject at ``fft()``."""
    if detrend_type is not None and detrend_type not in _DETREND_TYPES:
        msg = (
            f"detrend_type must be 'constant', 'linear', or None, got "
            f"{detrend_type!r}. 'constant' (the default) removes each window's "
            "mean, 'linear' its best-fit line, and None leaves the windows as "
            "they are."
        )
        raise ValueError(msg)


def _validated_artifact_samples(
    artifact_samples: Any, time_series: BackendArray, transform: str
) -> tuple[str | None, BackendArray | None]:
    """``Multitaper``'s ``artifact_samples`` as its kind and a read-only
    ``(n_time_samples, n_trials)`` boolean mask, ``True`` at an artifact,
    shared by the signals.

    The kind is ``"nonfinite"``, ``"mask"``, or ``None`` (not given, and no
    mask).
    """
    if artifact_samples is None:
        return None, None
    n_time, n_trials, n_signals = time_series.shape
    if isinstance(artifact_samples, str):
        if artifact_samples != "nonfinite":
            msg = (
                f"artifact_samples must be a boolean array, 'nonfinite', or None, got "
                f"{artifact_samples!r}. 'nonfinite' marks every NaN or infinite sample "
                "as an artifact."
            )
            raise ValueError(msg)
        return "nonfinite", _immutable_array_snapshot(
            ~xp.all(xp.isfinite(time_series), axis=-1)
        )
    dimensions = getattr(artifact_samples, "dims", None)
    if dimensions is not None:
        # Read by position, like time_series: a name saying otherwise is refused.
        misplaced = _misplaced_dimensions(dimensions)
        if misplaced:
            msg = (
                f"{transform} reads artifact_samples by position, as (time, trial[, "
                f"signal]), but its dimensions are named {tuple(dimensions)!r}: "
                f"{'; '.join(misplaced)}. Transpose it to that order first."
            )
            raise ValueError(msg)
        artifact_samples = artifact_samples.data
    mask = xp.asarray(artifact_samples)
    if mask.dtype != bool:
        msg = (
            f"artifact_samples must be a boolean array (True marks an artifact, a "
            f"sample to leave out), got dtype {mask.dtype}. For 0/1 values pass "
            "artifact_samples.astype(bool); to leave out NaN or infinite samples "
            "pass artifact_samples='nonfinite'."
        )
        raise TypeError(msg)
    shapes = ((n_time,), (n_time, n_trials), (n_time, n_trials, n_signals))
    if tuple(mask.shape) not in shapes:
        msg = (
            f"artifact_samples has shape {tuple(mask.shape)}; it must be "
            f"(n_time_samples,) = {shapes[0]}, shared by every trial and signal, "
            f"(n_time_samples, n_trials) = {shapes[1]}, shared by every signal, or "
            f"the time series' own shape (n_time_samples, n_trials, n_signals) = "
            f"{shapes[2]}."
        )
        raise ValueError(msg)
    if mask.ndim == 1:
        mask = xp.broadcast_to(mask[:, xp.newaxis], (n_time, n_trials))
    if mask.ndim == 3:
        # The weights are shared across signals, so an artifact in one signal
        # leaves the sample out of all.
        mask = xp.any(mask, axis=-1)
    return "mask", _immutable_array_snapshot(mask)


_ARTIFACT_HANDLINGS = ("drop", "missing_data_tapers")


def _validate_artifact_handling(
    artifact_handling: Any, supported: Sequence[str], transform: str
) -> None:
    """Raise unless ``artifact_handling`` is one of the options in
    ``supported``, those of the transform named ``transform``."""
    if isinstance(artifact_handling, str) and artifact_handling in supported:
        return
    if artifact_handling in _ARTIFACT_HANDLINGS:
        msg = (
            f"{transform} supports only artifact_handling='drop' (it has no DPSS "
            "taper family to adapt). Use Multitaper(..., "
            "artifact_handling='missing_data_tapers') or keep 'drop'."
        )
        raise ValueError(msg)
    hint = (
        did_you_mean(close_names(artifact_handling, supported))
        if isinstance(artifact_handling, str)
        else ""
    )
    options = " or ".join(repr(option) for option in supported)
    msg = (
        f"artifact_handling must be {options}, got {artifact_handling!r}.{hint} See "
        f"the artifact_handling parameter of {transform}."
    )
    raise ValueError(msg)


def _validate_window_step(
    time_window_duration: float | None,
    time_window_step: float | None,
) -> None:
    """Reject a non-positive window step and warn when steps leave gaps.

    ``time_window_duration`` must already be positive or ``None``.
    """
    if time_window_step is not None and time_window_step <= 0:
        msg = (
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
        raise ValueError(msg)

    # Warn if time_window_step creates gaps between windows
    if (
        time_window_step is not None
        and time_window_duration is not None
        and time_window_step > time_window_duration
    ):
        warnings.warn(
            f"time_window_step ({time_window_step}s) is larger than "
            f"time_window_duration ({time_window_duration}s).\n"
            "\n"
            "This creates gaps between analysis windows - some data will not be analyzed.\n"
            "If you want contiguous coverage, set step ≤ duration.\n"
            "\n"
            "If gaps are intended (e.g., sampling every Nth window), ignore this warning.",
            UserWarning,
            stacklevel=stacklevel_outside_package(),
        )


def _recording_too_short(msg: str) -> ValueError:
    """The ``ValueError`` for a recording too short for its window: the window
    is longer than it, or, spanning it, too short for the DPSS tapers.

    Marked so :func:`_transposed_layout_hint` (see
    :func:`_is_recording_too_short`) can tell it from every other
    ``ValueError``. A plain ``ValueError`` with an attribute, rather than a
    subclass, prints and pickles as the ``ValueError`` it was; pickling keeps
    the attribute.
    """
    error = ValueError(msg)
    error._recording_too_short = True  # type: ignore[attr-defined]
    return error


def _is_recording_too_short(error: BaseException | None) -> bool:
    """Whether ``error`` was made by :func:`_recording_too_short`."""
    return getattr(error, "_recording_too_short", False)


def _transposed_layout_error(n_time: int, n_signals: int, expected: str) -> ValueError:
    """The error for a recording too short for its window when it has more
    signals than time samples, so is most likely transposed.

    It says so instead of advising a longer window; raised from the window
    error (see :func:`_recording_too_short`), by which a caller accepting
    other layouts (``expected``) recognizes it and rebuilds the message.
    """
    return ValueError(
        f"Your array has {n_time} time sample{'' if n_time == 1 else 's'} and "
        f"{n_signals} signals; expected {expected}. It looks transposed: put "
        "time on the first axis, e.g. x.T. Read as given, its recording is too "
        "short to analyze."
    )


@contextlib.contextmanager
def _transposed_layout_hint(time_series: BackendArray) -> Iterator[None]:
    """Re-raise a recording too short for its window as a layout error when
    the ``(n_time_samples, n_trials, n_signals)`` series looks transposed;
    every other ``ValueError`` passes unchanged."""
    try:
        yield
    except ValueError as error:
        n_time, _, n_signals = time_series.shape
        if _is_recording_too_short(error) and n_time < n_signals:
            raise _transposed_layout_error(n_time, n_signals, _TRANSFORM_LAYOUT) from error
        raise


def _listed_indices(flags: BackendArray, limit: int = 10) -> str:
    """The indices where the 1-D boolean ``flags`` is true, as ``[0, 3, ...]``."""
    indices = np.flatnonzero(to_numpy(flags)).tolist()
    shown = ", ".join(map(str, indices[:limit]))
    return f"[{shown}, ...]" if len(indices) > limit else f"[{shown}]"


def _warn_about_unusual_time_series(
    time_series: BackendArray,
    artifact_samples: BackendArray | None = None,
    *,
    unused_samples: Callable[[], BackendArray] | None = None,
    suggest_artifact_samples: bool = False,
    finite_outside_artifacts: bool = False,
) -> bool:
    """Warn when a ``(n_time_samples, n_trials, n_signals)`` series looks transposed or not finite.

    Returns whether every value is finite, so a transform can record that it
    already warned about NaN or Inf in its input (see
    :meth:`Connectivity.from_transform`). With ``artifact_samples``, a
    ``(n_time_samples, n_trials)`` mask, the samples it marks as artifacts
    do not count: they never reach an estimate (and are zeroed before the
    FFT). ``unused_samples`` returns a mask of the same shape that also marks
    the other samples no kept trial window covers (called only for NaN or
    Inf input): NaN or Inf there is not reported either, but may reach
    coefficients (of weight 0), so the result is ``False``, with no warning
    when there is no other. ``suggest_artifact_samples``
    names the transform's ``artifact_samples`` among the remedies, and
    ``finite_outside_artifacts`` (``artifact_samples="nonfinite"``) skips the
    finiteness check its caller has already made.
    """
    # Warn if data appears to be transposed (very few time points, many signals)
    n_time, _, n_signals = time_series.shape
    if n_time < n_signals:
        warnings.warn(
            f"Your time series has only {n_time} time points but {n_signals} signals. "
            "This seems unusual and your data may be transposed.\n"
            "\n"
            "Expected shape: (n_time_samples, n_trials, n_signals)\n"
            f"Your shape: {time_series.shape}\n"
            "\n"
            "If your data is transposed, use:\n"
            "  >>> time_series_correct = time_series.T  # or appropriate transpose\n"
            "\n"
            "If your data is intentionally short (e.g., very brief epochs), you can ignore this warning.",
            UserWarning,
            stacklevel=stacklevel_outside_package(),
        )

    if finite_outside_artifacts:
        return True
    is_finite = xp.isfinite(time_series)
    if artifact_samples is not None:
        is_finite |= artifact_samples[..., xp.newaxis]
    # One pass and one device synchronization for the common, finite input;
    # the counts and locations are computed only for a warning.
    if bool(xp.all(is_finite)):
        return True
    is_nonfinite = ~is_finite
    if unused_samples is not None:
        is_nonfinite &= ~unused_samples()[..., xp.newaxis]
    n_nonfinite = int(xp.sum(is_nonfinite))
    if n_nonfinite == 0:
        # Each lies where no kept trial window reaches (in a dropped one's
        # coefficients, of weight 0, or in no window): no estimate sees it.
        return False
    where = f"signals {_listed_indices(xp.any(is_nonfinite, axis=(0, 1)))}"
    if time_series.shape[1] > 1:
        where = f"trials {_listed_indices(xp.any(is_nonfinite, axis=(0, 2)))}, {where}"
    at_valid = "" if unused_samples is None else " in the trial windows artifact_samples keeps"
    leave_out = (
        "pass artifact_samples='nonfinite' to leave them out (it drops every trial "
        "window that contains one), or "
        if suggest_artifact_samples
        else ""
    )
    remedy = (
        f"{leave_out}interpolate the affected samples (e.g. with numpy.interp) or drop "
        "the affected trials or signals before the analysis"
    )
    remedy = remedy[0].upper() + remedy[1:]
    warnings.warn(
        f"time_series contains {n_nonfinite} NaN or infinite "
        f"value{'' if n_nonfinite == 1 else 's'}{at_valid} (0-based indices: {where}). They "
        "propagate to the outputs: the power of an affected signal is NaN in every "
        "time window containing one, as is every connectivity value involving that "
        "signal there (every value, for measures that model all signals "
        "jointly, such as partial coherence and Granger prediction; "
        "canonical_coherence and global_coherence raise a ValueError instead). "
        f"{remedy}; for your array x, np.argwhere(~np.isfinite(x)) lists their indices.",
        UserWarning,
        stacklevel=stacklevel_outside_package(),
    )
    return False


class _MissingDataTapers(NamedTuple):
    """A Multitaper's missing-data tapers, computed at construction.

    The tables' row 0 holds the DPSS tapers and row ``k`` the tapers of the
    ``k``-th artifact pattern that is kept; each trial window's row is in
    ``table_index``. The tapers and concentrations are zero past the ones a
    pattern keeps (:func:`_missing_data_taper_sets`).
    """

    #: (n_time_windows, n_trials): each trial window's row; 0 without
    #: artifacts or when dropped.
    table_index: BackendArray
    #: The window and trial indices of the kept trial windows with artifacts.
    windows: tuple[BackendArray, BackendArray]
    #: (1 + n_kept_patterns, n_time_samples_per_window, n_tapers), scaled by
    #: sqrt(sampling_frequency) as the DPSS tapers are.
    tapers: BackendArray
    #: (1 + n_kept_patterns, n_tapers); row 0 the DPSS eigenvalues.
    concentrations: BackendArray
    #: (1 + n_kept_patterns, n_time_samples_per_window): the valid samples.
    valid: BackendArray
    #: The trial windows with artifacts, kept or dropped.
    n_gapped: int
    #: What the dropped windows with artifacts look like, for the warnings.
    dropped_summary: str

    @property
    def n_rescued(self) -> int:
        """The trial windows with artifacts that are kept."""
        return int(self.windows[0].size)

    def valid_of_rescued(self) -> BackendArray:
        """The valid samples of each kept window with artifacts, ``(n_rescued,
        1, n_time_samples_per_window)`` (broadcasting over signals)."""
        valid: BackendArray = self.valid[self.table_index[self.windows]][:, xp.newaxis, :]
        return valid


class _WindowedFourierTransform:
    """Tapered sliding-window FFT shared by Multitaper and the Hann STFT.

    Owns what the two transforms have in common: the window and step sample
    counts and their durations, the FFT length, the time and frequency
    coordinates, the windowed FFT itself, and the summary and provenance
    scaffolding.

    A subclass constructor validates its own arguments in its own order, using
    :func:`_validate_fft_workers`, :func:`_validate_window_step`, and
    :func:`_warn_about_unusual_time_series`. It sets ``_time_series``, the
    read-only snapshot of the input (``_immutable_array_snapshot``), and
    ``_time_series_is_finite``, what :func:`_warn_about_unusual_time_series`
    returns for it (:meth:`fft` and ``Connectivity.from_transform`` read it so
    NaN or Inf input is reported once), and calls
    :meth:`_set_window_state`, which validates the start time and sets the
    other private fields annotated below. The subclass must also supply:

    - ``sampling_frequency``, ``detrend_type``, and ``fft_workers``, fixed
      after construction (immutable attributes on :class:`Multitaper`,
      properties without setters on :class:`ShortTimeFourierTransform`);
    - ``_transform_name``, and any attributes of its own that
      :meth:`_provenance_metadata` should record, appended to the base's
      ``_PROVENANCE_FIELDS``;
    - :meth:`_tapers_for_fft`, the tapers applied to each window, of shape
      ``(n_time_samples_per_window, n_tapers)``. Generated DPSS tapers and the
      Hann window are unit-energy windows scaled by ``sqrt(sampling_frequency)``,
      so ``power()`` is a density; custom :class:`Multitaper` tapers are used
      as given;
    - :meth:`_spectral_bandwidth_metadata`, the taper identity and its
      qualified bandwidth;
    - the public ``observations_are_independent``, ``summarize_parameters()``
      (built with :meth:`_summary`), and ``__repr__`` (which :meth:`fft` logs);
    - optionally :meth:`_weight_taper_coefficients`, which reweights the
      coefficients of each taper;
    - optionally, to estimate trial windows that contain artifact samples
      rather than drop them, :meth:`_kept_trial_windows` (which may set
      ``_n_kept_tapers``), :meth:`_detrend_gapped_windows`,
      :meth:`_estimate_gapped_windows`, :meth:`_window_drop_rule`,
      :meth:`_dropped_window_notes`, and :meth:`_artifact_samples_details`.
    """

    #: The transform returns the full two-sided FFT spectrum (both positive and
    #: negative frequencies), so consumers must not assume a one-sided layout.
    is_one_sided = False
    _transform_name: str
    # Shortest window the subclass's tapers can use, in samples.
    _MIN_WINDOW_SAMPLES = 1
    # The shared window settings; a subclass extends the tuple with its own.
    _PROVENANCE_FIELDS: tuple[str, ...] = (
        "detrend_type",
        "fft_workers",
        "n_fft_samples",
        "n_signals",
        "n_time_samples_per_step",
        "n_time_samples_per_window",
        "n_trials",
        "nyquist_frequency",
        "sampling_frequency",
        "start_time",
        "time_window_duration",
        "time_window_step",
    )

    if TYPE_CHECKING:
        # Read here, provided by each subclass (see the class docstring).
        @property
        def sampling_frequency(self) -> float: ...

        @property
        def detrend_type(self) -> str | None: ...

        @property
        def fft_workers(self) -> int | None: ...

        @property
        def artifact_handling(self) -> str: ...

    _time_series: BackendArray
    # A subclass from outside the package may not record it; then assume finite.
    _time_series_is_finite: bool = True
    # Set by _set_artifact_samples; a subclass that does not call it has none.
    _artifact_samples_mode: str | None = None
    _artifact_samples: BackendArray | None = None
    _valid_windows: BackendArray | None = None
    # The number of leading tapers each trial window keeps, (n_time_windows,
    # n_trials), 0 where it is dropped; None when every kept trial window
    # keeps every taper.
    _n_kept_tapers: BackendArray | None = None
    _n_kept_trial_windows: int
    _start_time: BackendArray
    _time_window_duration: float | None
    _time_window_step: float | None
    _n_fft_samples: int | None
    _n_time_samples_per_window: int | None
    _n_samples_per_time_step: int | None

    def _set_window_state(
        self,
        start_time: Any,
        *,
        time_window_duration: float | None,
        time_window_step: float | None,
        n_fft_samples: int | None,
        n_time_samples_per_window: int | None,
        n_time_samples_per_step: int | None,
    ) -> None:
        """Validate and store the start time, and store the window request.

        The durations, sample counts, and FFT length are stored as given
        (``None`` when not requested); the properties resolve and check them.

        Raises
        ------
        ValueError
            If ``start_time`` is not a finite scalar.
        """
        # ``time`` adds start_time to a 1-D array of window centers, so a
        # per-trial array would either broadcast-fail or silently produce a 2-D
        # time axis.
        self._start_time = _immutable_array_snapshot(_finite_scalar_start_time(start_time))
        self._time_window_duration = time_window_duration
        self._time_window_step = time_window_step
        self._n_fft_samples = n_fft_samples
        self._n_time_samples_per_window = n_time_samples_per_window
        self._n_samples_per_time_step = n_time_samples_per_step

    @property
    def time_series(self) -> BackendArray:
        """Independent read-only copy of the input time-series snapshot."""
        return _readonly_array_copy(self._time_series)

    @property
    def start_time(self) -> BackendArray:
        """Time of the first sample, in seconds.

        Returns an independent read-only copy of the start-time coordinate.
        """
        return _readonly_array_copy(self._start_time)

    def _provenance_metadata(self) -> dict[str, Any]:
        """Native transform settings, serialized by the shared provenance builder."""
        metadata: dict[str, Any] = {"transform": self._transform_name}
        for attr in self._PROVENANCE_FIELDS:
            value = getattr(self, attr)
            value = to_numpy(value) if hasattr(value, "shape") else value
            if isinstance(value, np.ndarray):
                if value.dtype.kind not in "biufSU":  # exclude complex/object
                    continue
            elif value is not None and not isinstance(
                value,
                (str, bytes, bool, int, float, np.integer, np.floating, np.bool_),
            ):
                continue
            metadata[attr] = value
        metadata.update(self._spectral_bandwidth_metadata())
        if self._artifact_samples_mode is not None:
            # The mask itself can be as large as the recording, so its kind and
            # what it removed are recorded rather than the mask.
            assert self._artifact_samples is not None
            metadata["artifact_samples"] = self._artifact_samples_mode
            metadata["artifact_handling"] = self.artifact_handling
            metadata["n_artifact_samples"] = int(xp.sum(self._artifact_samples))
            metadata["n_dropped_trial_windows"] = (
                len(self.time) * self.n_trials - self._n_kept_trial_windows
            )
        return metadata

    def _spectral_bandwidth_metadata(self) -> dict[str, Any]:
        """Taper identity and its qualified spectral bandwidth."""
        raise NotImplementedError

    def _summary(
        self,
        title: str,
        spectral_rows: list[tuple[str, Any]],
        bandwidth_row: tuple[str, str],
    ) -> str:
        """Human-readable configuration summary shared by the windowed transforms.

        ``spectral_rows`` follow the sampling frequency in "Spectral Parameters";
        ``bandwidth_row`` heads "Frequency Analysis".
        """
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

        if self._artifact_samples_mode is not None:
            spectral_rows = [
                *spectral_rows,
                ("Artifact samples", self._artifact_samples_description()),
            ]
            if self.artifact_handling != "drop":
                spectral_rows.append(("Artifact handling", self.artifact_handling))
        spectral = "\n".join(
            f"{label + ':':<31}{value}"
            for label, value in [
                ("Sampling frequency", f"{self.sampling_frequency:g} Hz"),
                *spectral_rows,
            ]
        )
        frequency_rows = [
            bandwidth_row,
            ("Nyquist frequency", f"{self.nyquist_frequency:.1f} Hz"),
            ("Frequency range", f"0.0 - {self.nyquist_frequency:.1f} Hz"),
            ("FFT samples", f"{self.n_fft_samples}"),
        ]
        width = max(22, max(len(label) + 2 for label, _ in frequency_rows))
        frequency = "\n".join(
            f"{label + ':':<{width}}{value}" for label, value in frequency_rows
        )
        return f"""{title}
{"=" * (len(title) + 1)}

Data Shape
----------
Time samples:    {n_time_samples} ({signal_duration:.2f} seconds)
Signals:         {self.n_signals}
Trials:          {self.n_trials}

Spectral Parameters
-------------------
{spectral}

Time Windowing
--------------
Window duration:  {self.time_window_duration:.3f} s ({self.n_time_samples_per_window} samples)
Window step:      {self.time_window_step:.3f} s {overlap_desc}
Number of windows: {n_windows}

Frequency Analysis
------------------
{frequency}
"""

    @property
    def time_window_duration(self) -> float:
        """Window length in seconds, after rounding to whole samples.

        Returns
        -------
        float
            Duration in seconds of each time window actually analyzed,
            ``n_time_samples_per_window / sampling_frequency`` (a requested
            duration is rounded to the nearest sample).

        """
        return self.n_time_samples_per_window / self.sampling_frequency

    @property
    def time_window_step(self) -> float:
        """Step between window starts in seconds, after rounding to whole samples.

        Returns
        -------
        float
            Step size in seconds between consecutive time windows actually
            taken, ``n_time_samples_per_step / sampling_frequency`` (a
            requested step is rounded to the nearest sample).

        """
        return self.n_time_samples_per_step / self.sampling_frequency

    @property
    def n_time_samples_per_window(self) -> int:
        """Window length in samples.

        An explicit ``n_time_samples_per_window`` is used as given. Otherwise,
        if ``time_window_duration`` is set, the window is the nearest whole
        number of samples in that duration. If neither is set, the window is
        the whole time series.

        Returns
        -------
        int
            Number of time samples in each window.

        Raises
        ------
        ValueError
            If the resolved window is shorter than the transform's tapers
            allow or longer than the time series.

        """
        # An explicit n_time_samples_per_window is used as given (a duration
        # given with it was checked to agree at construction).
        if self._n_time_samples_per_window is None:
            if self._time_window_duration is None:
                self._n_time_samples_per_window = self._time_series.shape[0]
            else:
                self._n_time_samples_per_window = _duration_to_samples(
                    self._time_window_duration, self.sampling_frequency
                )
        assert self._n_time_samples_per_window is not None
        # Validate the resolved window length regardless of which input path set
        # it: an explicit n_time_samples_per_window=0 (or a duration rounding to
        # 0) would divide by zero downstream, and an oversized window yields an
        # empty transform.
        n_time_samples = self._time_series.shape[0]
        minimum = self._MIN_WINDOW_SAMPLES
        if self._n_time_samples_per_window < minimum:
            samples, count = (
                ("1 sample", "a positive n_time_samples_per_window")
                if minimum == 1
                else (f"{minimum} samples", f"n_time_samples_per_window >= {minimum}")
            )
            msg = (
                f"n_time_samples_per_window resolved to "
                f"{self._n_time_samples_per_window}, but each window needs at "
                f"least {samples}. If you set time_window_duration "
                f"({self._time_window_duration}), it is too short for "
                f"sampling_frequency ({self.sampling_frequency}); use a duration "
                f">= {minimum / self.sampling_frequency} s. Otherwise pass {count}."
            )
            raise ValueError(msg)
        if self._n_time_samples_per_window > n_time_samples:
            msg = (
                f"n_time_samples_per_window ({self._n_time_samples_per_window}) is "
                f"larger than the signal length ({n_time_samples}); no window fits, "
                f"which would yield an empty transform. Use a smaller window "
                f"(time_window_duration <= "
                f"{n_time_samples / self.sampling_frequency} s)."
            )
            raise _recording_too_short(msg)
        return self._n_time_samples_per_window

    @property
    def n_fft_samples(self) -> int:
        """FFT length of each window, which is also the number of frequency bins.

        An explicit ``n_fft_samples`` is used as given. Otherwise it is a
        length at least ``n_time_samples_per_window`` that is fast for the FFT
        (``scipy.fft.next_fast_len``), so windows may be zero-padded.

        Returns
        -------
        int
            Number of FFT samples.

        Raises
        ------
        ValueError
            If the requested ``n_fft_samples`` is smaller than
            ``n_time_samples_per_window``.

        """
        if self._n_fft_samples is None:
            self._n_fft_samples = default_fft_length(self.n_time_samples_per_window)
        elif self._n_fft_samples < self.n_time_samples_per_window:
            # scipy/cupy fft crops (does not zero-pad) when n < len(signal), so
            # a too-small n_fft_samples silently discards most of each window.
            msg = (
                f"n_fft_samples ({self._n_fft_samples}) must be >= the number "
                f"of time samples per window ({self.n_time_samples_per_window}).\n"
                f"n_fft_samples is the FFT length (used for zero-padding), not "
                f"the number of output frequency bins. A value smaller than the "
                f"window length would silently truncate the signal before the "
                f"FFT.\n"
                f"Either omit n_fft_samples (it defaults to a fast length >= the "
                f"window length) or set it >= {self.n_time_samples_per_window}."
            )
            raise ValueError(msg)
        return self._n_fft_samples

    @property
    def frequencies(self) -> NDArray[np.floating]:
        """Frequency of each FFT bin in Hz, in standard FFT order.

        These label the coefficients of :meth:`fft`, negative frequencies
        included, not the results computed from them. To label the frequency
        axis of a power or connectivity result, use
        :attr:`Connectivity.frequencies
        <spectral_connectivity.Connectivity.frequencies>`: the
        ``n_fft_samples // 2 + 1`` non-negative bins, 0 Hz up to Nyquist.

        Returns
        -------
        NDArray[float64], shape (n_fft_samples,)
            Frequency values in Hz corresponding to FFT bins, as
            ``numpy.fft.fftfreq`` orders them (non-negative frequencies first,
            then negative; for an even ``n_fft_samples`` the Nyquist bin is
            listed as negative).

        Examples
        --------
        >>> import numpy as np
        >>> from spectral_connectivity import Connectivity, Multitaper
        >>> rng = np.random.default_rng(0)
        >>> multitaper = Multitaper(rng.standard_normal((100, 2, 2)), sampling_frequency=100)
        >>> multitaper.frequencies.shape, multitaper.frequencies[[0, 1, 50, -1]].tolist()
        ((100,), [0.0, 1.0, -50.0, -1.0])
        >>> connectivity = Connectivity.from_transform(multitaper)
        >>> connectivity.frequencies.shape, connectivity.frequencies[[0, -1]].tolist()
        ((51,), [0.0, 50.0])
        >>> connectivity.power(one_sided=True).shape  # (time, frequency, signal)
        (1, 51, 2)
        """
        return fftfreq(self.n_fft_samples, 1.0 / self.sampling_frequency)

    @property
    def n_time_samples_per_step(self) -> int:
        """Step between window starts, in samples.

        An explicit `n_time_samples_per_step` is used as given. Otherwise, if
        `time_window_step` is set, the step is the nearest whole number of
        samples in that duration. If neither is set, the step defaults to the
        window length (non-overlapping windows).

        Returns
        -------
        int
            Number of samples to advance between windows.

        Raises
        ------
        ValueError
            If the resolved step is shorter than one sample.

        """
        if self._n_samples_per_time_step is None:
            if self._time_window_step is None:
                self._n_samples_per_time_step = self.n_time_samples_per_window
            else:
                self._n_samples_per_time_step = _duration_to_samples(
                    self._time_window_step, self.sampling_frequency
                )
        assert self._n_samples_per_time_step is not None
        # Validate the resolved step regardless of which input path set it: an
        # explicit n_time_samples_per_step=0 (or a step truncating to 0) would
        # divide by zero when building the sliding windows.
        if self._n_samples_per_time_step < 1:
            msg = (
                f"n_time_samples_per_step resolved to "
                f"{self._n_samples_per_time_step}, but each step must advance at "
                f"least 1 sample. If you set time_window_step "
                f"({self._time_window_step}), it is too short for sampling_frequency "
                f"({self.sampling_frequency}); use a step "
                f">= {1.0 / self.sampling_frequency} s. Otherwise pass a positive "
                f"n_time_samples_per_step."
            )
            raise ValueError(msg)
        return self._n_samples_per_time_step

    @property
    def time(self) -> NDArray[np.floating]:
        """Center time of each window, in seconds.

        Returns
        -------
        NDArray[float64], shape (n_time_windows,)
            Time values in seconds for center of each time window, offset by
            ``start_time``.

        """
        original_time = xp.arange(0, self._time_series.shape[0]) / self.sampling_frequency
        # Label each window by its center time, as documented (the mean of the
        # window's sample times equals the center for uniformly spaced samples).
        window_center_time: NDArray[np.floating] = _sliding_window(
            original_time, self.n_time_samples_per_window, self.n_time_samples_per_step
        ).mean(axis=-1)
        center_time: NDArray[np.floating] = self._start_time + window_center_time
        return center_time

    @property
    def n_signals(self) -> int:
        """Number of signals in the time series.

        Returns
        -------
        int
            Number of signals in the time series.

        """
        return int(self._time_series.shape[-1])

    @property
    def n_trials(self) -> int:
        """Number of trials in the time series.

        Returns
        -------
        int
            Number of trials in the time series.

        """
        return int(self._time_series.shape[1])

    @property
    def nyquist_frequency(self) -> float:
        """Maximum resolvable frequency, in Hz.

        Returns
        -------
        float
            Nyquist frequency in Hz, ``sampling_frequency / 2``.

        """
        return self.sampling_frequency / 2

    @property
    def time_bins_are_independent(self) -> bool:
        """Whether the time windows may be counted as independent observations.

        An expectation that averages over time (``"time"``, ``"time_tapers"``,
        ...) counts every window in ``n_observations``. Windows that share
        samples are correlated, so beyond some overlap that count overstates
        the effective sample size and biases the measures that rely on it
        (pairwise phase consistency of independent signals, for example, is
        no longer centered on 0). Like :class:`Welch`,
        which treats segments overlapping by at most 50% as approximately
        independent (Welch 1967; Percival & Walden 1993, sec. 6.17), windows
        count as independent when ``n_time_samples_per_step`` is at least half
        of ``n_time_samples_per_window``. :class:`Connectivity` warns when an
        expectation averages over correlated time bins.

        Returns
        -------
        bool
            ``True`` when successive windows overlap by at most half.
        """
        return 2 * self.n_time_samples_per_step >= self.n_time_samples_per_window

    def fft(self) -> NDArray[np.complexfloating]:
        """Compute the tapered Fourier coefficients of every time window.

        Each window is detrended as ``detrend_type`` sets, multiplied by each
        taper, and transformed with an FFT of length ``n_fft_samples``. A
        transform with a ``taper_weighting`` other than ``"uniform"`` then
        rescales each taper's coefficients by its weight.

        Returns
        -------
        fourier_coefficients : array
            Shape (n_time_windows, n_trials, n_tapers, n_fft_samples, n_signals).
            Complex-valued Fourier coefficients, with one entry per taper on
            the taper axis; a transform with a single window function, such as
            a Hann window, has a singleton taper axis (``n_tapers`` is 1).

        """
        logger.info(self)
        time_series = _sliding_window(
            self._time_series,
            window_size=self.n_time_samples_per_window,
            step_size=self.n_time_samples_per_step,
            axis=0,
        )
        if self._artifact_samples is not None:
            # Zeroing the artifacts in the windows' copy keeps the coefficients
            # of the windows containing them finite: those of a dropped window
            # are unused, and the others are detrended again over their valid
            # samples (:meth:`_detrend_gapped_windows`). The other windows are
            # unchanged.
            xp.copyto(time_series, 0, where=self._artifact_windows()[:, :, xp.newaxis, :])
        # NaN or Inf input was reported at construction; detrending and
        # tapering it (Inf times a taper's zero end) would add NumPy "invalid value"
        # RuntimeWarnings about the same values.
        with (
            contextlib.nullcontext()
            if self._time_series_is_finite
            else np.errstate(invalid="ignore")
        ):
            if self.detrend_type is not None:
                time_series = detrend(time_series, type=self.detrend_type)
            self._detrend_gapped_windows(time_series)

            coefficients = _multitaper_fft(
                self._tapers_for_fft(),
                time_series,
                self.n_fft_samples,
                self.sampling_frequency,
                workers=self.fft_workers,
            )
            self._estimate_gapped_windows(coefficients, time_series)
            return self._weight_taper_coefficients(coefficients.swapaxes(2, -1), time_series)

    def _detrend_gapped_windows(self, windows: NDArray[np.floating]) -> None:
        """Detrend again, in place, the kept trial windows that contain
        artifact samples; none by default (they are all dropped).

        ``windows`` are the detrended zero-filled windows, ``(n_time_windows,
        n_trials, n_signals, n_time_samples_per_window)``.
        """

    def _estimate_gapped_windows(
        self,
        coefficients: NDArray[np.complexfloating],
        windows: NDArray[np.floating],
    ) -> None:
        """Replace, in place, the coefficients of the kept trial windows that
        contain artifact samples with those of their own tapers; none by
        default (they are all dropped).

        ``coefficients`` are every window's with :meth:`_tapers_for_fft`,
        ``(n_time_windows, n_trials, n_signals, n_fft_samples, n_tapers)``, and
        ``windows`` the detrended windows they were computed from.
        """

    def _tapers_for_fft(self) -> NDArray[np.floating]:
        """Return the internal taper snapshot, ``(n_time_samples_per_window, n_tapers)``."""
        raise NotImplementedError

    def _artifact_windows(self) -> BackendArray:
        """The artifact mask of each window, a view ``(n_time_windows, n_trials, n_time_samples_per_window)``."""
        assert self._artifact_samples is not None
        return _sliding_window(
            self._artifact_samples,
            self.n_time_samples_per_window,
            self.n_time_samples_per_step,
            axis=0,
            is_copy=False,
        )

    def _set_artifact_samples(self, artifact_samples: Any) -> None:
        """Validate ``artifact_samples``, find the trial windows it keeps, and
        warn about NaN or Inf outside it; the last step of construction.

        Sets ``_artifact_samples_mode``, ``_artifact_samples``,
        ``_valid_windows`` (``None`` when every trial window is kept, so the
        transform is the one without a mask), ``_n_kept_trial_windows`` and
        ``_time_series_is_finite``. Warns when more than half of the trial
        windows are dropped or a window keeps no trial, and raises if no trial
        window is kept.
        """
        name = type(self).__name__
        self._artifact_samples_mode, self._artifact_samples = _validated_artifact_samples(
            artifact_samples, self._time_series, name
        )
        self._valid_windows = None
        if self._artifact_samples is not None:
            valid_windows = self._kept_trial_windows(self._artifact_windows())
            self._n_kept_trial_windows = int(xp.sum(valid_windows))
            if self._n_kept_trial_windows < valid_windows.size:
                self._check_kept_trial_windows(valid_windows)
                self._valid_windows = _immutable_array_snapshot(valid_windows)
        # With "nonfinite" every NaN or Inf is an artifact, so its validation
        # already looked at every value.
        self._time_series_is_finite = _warn_about_unusual_time_series(
            self._time_series,
            self._artifact_samples,
            unused_samples=None if self._artifact_samples is None else self._unused_samples,
            suggest_artifact_samples=True,
            finite_outside_artifacts=self._artifact_samples_mode == "nonfinite",
        )

    def _kept_trial_windows(self, artifact_windows: BackendArray) -> BackendArray:
        """Which trial windows are kept, ``(n_time_windows, n_trials)``
        booleans: by default those without an artifact sample.

        ``artifact_windows`` is :meth:`_artifact_windows`.
        """
        clean: BackendArray = ~xp.any(artifact_windows, axis=-1)
        return clean

    @property
    def _window_length(self) -> str:
        """The window's length in samples and seconds, for messages."""
        return f"{self.n_time_samples_per_window} samples ({self.time_window_duration:g} s)"

    def _window_drop_rule(self) -> str:
        """The sentence of the warnings saying when a trial's window is dropped."""
        return (
            f"A trial's window of {self._window_length} is dropped if any of its samples is an "
            "artifact; a shorter time_window_duration loses less data around "
            "artifacts. Alternatively, artifact_handling='missing_data_tapers' "
            "(Multitaper only) estimates such a window from its valid samples."
        )

    def _dropped_window_notes(self) -> list[tuple[str, str]]:
        """Warnings of a subclass about the windows it drops, as (text, closing
        sentence) pairs, given before the base ones; none by default."""
        return []

    def _unused_samples(self) -> BackendArray:
        """``(n_time_samples, n_trials)``, ``True`` at the samples no kept trial
        window covers: the artifacts, the rest of each dropped trial window
        that no kept window overlaps, and any sample past the last window."""
        assert self._artifact_samples is not None
        n_time, n_trials = self._artifact_samples.shape
        kept = (
            xp.ones((len(self.time), n_trials), dtype=xp.int64)
            if self._valid_windows is None
            else self._valid_windows.astype(xp.int64)
        )
        # Each kept window adds 1 from its first sample and removes it after
        # its last; the running sum counts the kept windows over each sample.
        # The starts are distinct, as are the ends, so each += sees each index once.
        starts = xp.arange(kept.shape[0]) * self.n_time_samples_per_step
        steps = xp.zeros((n_time + 1, n_trials), dtype=xp.int64)
        steps[starts] += kept
        steps[starts + self.n_time_samples_per_window] -= kept
        unused: BackendArray = xp.cumsum(steps, axis=0)[:n_time] == 0
        return unused

    def _check_kept_trial_windows(self, valid_windows: BackendArray) -> None:
        """Raise if ``valid_windows`` keeps no trial window; warn if it keeps
        fewer than half, or if a window keeps no trial, after any of
        :meth:`_dropped_window_notes`. The drop rule is stated in the first
        warning only."""
        n_windows = int(valid_windows.shape[0])
        empty = ~xp.any(valid_windows, axis=1)
        n_empty = int(xp.sum(empty))
        name = type(self).__name__
        rule = self._window_drop_rule()
        inverted = "True marks an artifact; for a mask of valid samples pass ~mask."
        if n_empty == n_windows:
            msg = (
                "artifact_samples leaves no trial window: every one of the "
                f"{n_windows} time window{'' if n_windows == 1 else 's'} is dropped "
                f"in every trial, so nothing can be estimated. {rule} {inverted} "
                "Otherwise mark fewer samples as artifacts."
            )
            raise ValueError(msg)
        notes = self._dropped_window_notes()
        n_kept, n_trial_windows = self._n_kept_trial_windows, int(valid_windows.size)
        if 2 * n_kept < n_trial_windows:
            notes.append(
                (
                    (
                        f"artifact_samples drops {1 - n_kept / n_trial_windows:.0%} of "
                        f"trial windows ({n_kept} of {n_trial_windows} kept)."
                    ),
                    f"{inverted} {name}.valid_windows shows which are kept.",
                )
            )
        if n_empty:
            notes.append(
                (
                    (
                        f"artifact_samples leaves {n_empty} of {n_windows} time windows "
                        f"(0-based indices {_listed_indices(empty)}) with no trial, so "
                        "every measure is NaN there."
                    ),
                    f"{name}.valid_windows shows which are kept.",
                )
            )
        for number, (text, closing) in enumerate(notes):
            warnings.warn(
                f"{text} {rule} {closing}" if number == 0 else f"{text} {closing}",
                UserWarning,
                stacklevel=stacklevel_outside_package(),
            )

    @property
    def artifact_samples(self) -> NDArray[np.bool_] | None:
        """Which samples the transform leaves out, shared across signals.

        Returns
        -------
        NDArray[bool], shape (n_time_samples, n_trials), or None
            ``True`` at an artifact: ``artifact_samples`` as given, a
            ``(n_time_samples,)`` mask broadcast over trials, a per-signal mask
            reduced over signals (an artifact in any signal leaves the sample
            out of all), and ``"nonfinite"`` resolved to the NaN or infinite
            samples; ``None`` when not given. A detached copy, read-only on
            the NumPy backend (a CuPy array on the GPU, which has no
            read-only flag).
        """
        if self._artifact_samples is None:
            return None
        return _readonly_array_copy(self._artifact_samples)

    @property
    def valid_windows(self) -> NDArray[np.bool_]:
        """Which trial windows the transform keeps.

        ``True`` at a kept trial window, the opposite sense of
        :attr:`artifact_samples`; the coefficients of every taper of a dropped
        one get observation weight 0. With
        ``artifact_handling="missing_data_tapers"`` a window containing
        artifact samples is kept if it keeps at least 2 tapers.

        Returns
        -------
        NDArray[bool], shape (n_time_windows, n_trials)
            A detached copy, all ``True`` without ``artifact_samples``;
            read-only on the NumPy backend (CuPy arrays have no read-only flag).
        """
        if self._valid_windows is None:
            return mark_readonly_if_supported(
                xp.ones((len(self.time), self.n_trials), dtype=bool)
            )
        return _readonly_array_copy(self._valid_windows)

    @property
    def n_kept_tapers(self) -> NDArray[np.intp]:
        """How many tapers each trial window keeps.

        All of them (``n_tapers``) for a window without artifacts, 0 for a
        dropped one, and, with ``Multitaper(artifact_handling=
        "missing_data_tapers")``, the number of missing-data tapers a window
        with artifacts keeps. The kept tapers are the leading ones; the
        others get observation weight 0.

        Returns
        -------
        NDArray[int], shape (n_time_windows, n_trials)
            A detached read-only copy.
        """
        if self._n_kept_tapers is not None:
            return _readonly_array_copy(self._n_kept_tapers)
        n_tapers = self._tapers_for_fft().shape[1]
        return mark_readonly_if_supported(self.valid_windows.astype(xp.intp) * n_tapers)

    @property
    def observation_weights(self) -> NDArray[np.floating] | None:
        """Weights consumed by :class:`Connectivity`: 0 for a dropped trial window.

        Returns
        -------
        observation_weights : array or None
            Shape ``(n_time_windows, n_trials, n_tapers, n_fft_samples, 1)``: 1
            for every taper of a trial window :attr:`valid_windows` keeps and 0
            for one it drops. With ``artifact_handling="missing_data_tapers"``
            a kept window containing artifact samples has 1 for each of its
            missing-data tapers that is kept and 0 for the others. ``None``
            when every window keeps every taper, so :class:`Connectivity` uses
            its plain, unweighted expectation.
        """
        if self._valid_windows is None and self._n_kept_tapers is None:
            return None
        tapers = self._tapers_for_fft()
        n_tapers = tapers.shape[1]
        kept = xp.arange(n_tapers) < self.n_kept_tapers[:, :, xp.newaxis]
        # The precision of the coefficients' real part.
        weights = kept.astype(
            xp.result_type(self._time_series.dtype, tapers.dtype, xp.float32)
        )
        shape = (*weights.shape[:2], n_tapers, self.n_fft_samples, 1)
        return mark_readonly_if_supported(
            xp.broadcast_to(weights[:, :, :, xp.newaxis, xp.newaxis], shape)
        )

    def _artifact_samples_description(self) -> str:
        """``artifact_samples`` as given ("nonfinite" or a mask) and the windows kept."""
        given = "'nonfinite'" if self._artifact_samples_mode == "nonfinite" else "<mask>"
        n_trial_windows = len(self.time) * self.n_trials
        return (
            f"{given} ({self._n_kept_trial_windows} of {n_trial_windows} trial windows "
            f"kept{self._artifact_samples_details()})"
        )

    def _artifact_samples_details(self) -> str:
        """What a subclass adds to the description of the windows kept."""
        return ""

    def _artifact_samples_repr(self) -> str:
        """The repr's ``artifact_samples`` entry, empty when it was not given."""
        if self._artifact_samples_mode is None:
            return ""
        return f",\n           artifact_samples={self._artifact_samples_description()}"

    def _weight_taper_coefficients(
        self,
        coefficients: NDArray[np.complexfloating],
        windows: NDArray[np.floating],  # noqa: ARG002 -- used by overrides
    ) -> NDArray[np.complexfloating]:
        """Reweight the coefficients of each taper; equal weights by default.

        Parameters
        ----------
        coefficients : array, shape (n_time_windows, n_trials, n_tapers, n_fft_samples, n_signals)
            Tapered Fourier coefficients.
        windows : array, shape (n_time_windows, n_trials, n_signals, n_time_samples_per_window)
            The detrended windows the coefficients were computed from.
        """
        return coefficients


class Multitaper(_WindowedFourierTransform):
    """
    Multitaper spectral analysis for robust power spectral density estimation.

    Transforms time-domain signals to frequency domain using multiple orthogonal
    tapering windows (Slepian sequences). This approach reduces spectral leakage
    and provides better spectral estimates than single-taper methods.

    Parameters
    ----------
    time_series : NDArray[floating], shape (n_time_samples, n_trials, n_signals)
        Input time series data. **Must be a real-valued 3D array**; a complex
        one raises ``TypeError``. **Trials are the middle axis and signals the
        last**: ``x[:, k, :]`` is trial ``k``.

        - n_time_samples: number of time points
        - n_trials: number of trials (use 1 for single trial)
        - n_signals: number of signals/channels

        An array of shape ``(500, 3, 40)`` is read as 500 time samples of 3
        trials of 40 signals; if it holds 3 signals and 40 trials, pass
        ``np.swapaxes(x, 1, 2)``. The order cannot be checked from the values.
        Multiple trials are averaged in the spectral domain.

        **Important:** If your data is 1D or 2D, use `prepare_time_series()`
        helper function to convert it to the required 3D format.

        An ``xarray.DataArray`` is read by position, like any array; one whose
        dimension names put time, trials, or signals on another axis raises
        ``ValueError`` (transpose it first, or use
        :func:`~spectral_connectivity.multitaper_connectivity`, which assigns
        axes from dimension names).
    sampling_frequency : float
        Samples per second; required because it labels the frequency axis and scales power.
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
        For 2 Hz frequency resolution with 1 second windows: use NW = 1, the
        smallest allowed (one taper); 1 Hz resolution needs a window of at
        least 2 seconds
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
        Duration in seconds of sliding time windows, rounded to the nearest
        sample (the ``time_window_duration`` attribute reports the rounded
        duration). If None, analyzes entire time series (no time resolution).
    time_window_step : float, optional
        Step size in seconds between consecutive time windows, rounded to the
        nearest sample (the ``time_window_step`` attribute reports the rounded
        step). If None, uses non-overlapping windows (step = window duration).
    n_tapers : int, optional
        Number of DPSS tapers to request. If None, computed as
        floor(2 * time_halfbandwidth_product) - 1. With ``is_low_bias=True``
        (the default) a requested taper whose spectral concentration is at or
        below 0.9 is dropped, with a warning: only about
        ``2 * time_halfbandwidth_product - 1`` are well concentrated, so a
        larger request needs a larger ``time_halfbandwidth_product`` or
        ``is_low_bias=False``. ``n_tapers`` reports the count used.
    tapers : NDArray[floating], shape (n_time_samples_per_window, n_tapers), optional
        Pre-computed tapering windows. If None, DPSS tapers are computed
        automatically.
    start_time : float, default=0
        Time in seconds of the first sample. Must be a scalar: ``time`` is
        one-dimensional, so per-trial start times are not supported.
    n_fft_samples : int, optional
        Length of FFT. If None, uses a value >= n_time_samples_per_window chosen
        to be fast for the FFT algorithm: scipy.fft.next_fast_len, on the GPU
        backend too, so the default frequency grid is the same on CPU and GPU.
        The directed measures (spectral Granger, DTF, PDC) depend slightly
        on this length: see the Notes of
        :func:`~spectral_connectivity.minimum_phase_decomposition.minimum_phase_decomposition`.
    n_time_samples_per_window : int, optional
        Number of samples per time window. Computed (to the nearest sample)
        from time_window_duration if not provided; when both are given they
        must resolve to the same count, otherwise a ``ValueError`` is raised.
    n_time_samples_per_step : int, optional
        Number of samples to advance between windows. Computed (to the nearest
        sample) from time_window_step if not provided; when both are given they
        must resolve to the same count, otherwise a ``ValueError`` is raised.
    is_low_bias : bool, default=True
        If True, exclude tapers with eigenvalues (spectral concentrations) at or
        below MIN_EIGENVALUE_THRESHOLD (0.9) to reduce bias, warning when that
        leaves fewer tapers than requested.
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

        The adaptive weights are applied to the coefficients themselves: each
        signal's weights ``d_k(f)`` are RMS-normalized over tapers and
        multiplied into its eigencoefficients, so the downstream taper mean of
        ``|coefficient|**2`` is exactly Thomson's adaptive auto-spectrum
        ``sum_k d_k^2 |Y_k|^2 / sum_k d_k^2``. A cross-spectrum between
        signals x and y, however, becomes
        ``sum_k d_xk d_yk Y_xk conj(Y_yk) / sqrt(sum_k d_xk^2 sum_k d_yk^2)``
        rather than Thomson's joint ``... / sum_k d_xk d_yk``. By the
        Cauchy-Schwarz inequality the two differ by the cosine similarity of
        the two weight vectors, so adaptive coherence (and every normalized
        cross-signal measure) is shrunk by that factor wherever the signals'
        spectra, and hence their weights, differ: a perfectly coherent pair
        with different spectra does not reach ``|coherence| = 1``. The
        shrinkage vanishes when the weights agree (white spectra, or the
        ``"eigen"`` mode, whose weights are signal-independent).
    adaptive_max_iterations : int, default=50
        Iteration ceiling for Thomson adaptive weighting.
    adaptive_tolerance : float, default=1e-8
        Relative convergence tolerance for the adaptive spectrum estimate.
    artifact_samples : NDArray[bool] or "nonfinite", optional
        Samples to leave out: artifacts, gaps, or missing values (NaN).
        ``True`` marks an **artifact**, a sample to leave out, as MNE's
        ``BAD_`` annotations, FieldTrip's artifact definitions, EEGLAB's
        rejection ranges, and NumPy masked arrays do; if your mask marks valid
        samples as ``True``, pass ``~mask``. Artifact intervals convert
        directly: set the mask ``True`` inside each one (for an MNE ``BAD_``
        annotation, the samples from its onset for its duration; for a row
        ``[begsample, endsample]`` of a FieldTrip artifact matrix, 1-based
        sample numbers that include both ends, ``mask[begsample - 1:endsample]``).
        With the default ``artifact_handling="drop"``, a
        trial's window touching any artifact sample is dropped whole: all its
        tapers' coefficients get observation weight 0 in every expectation,
        and the window's other trials are averaged on their own, so long
        windows lose more data; choose the window to fit the gaps between
        artifacts. The outputs mark what is kept instead:
        :attr:`valid_windows` is ``True`` at the trial windows kept, and
        ``Connectivity.n_valid_observations`` counts what each bin keeps. Every
        per-bin statistic, including those that count observations (listed
        under ``Connectivity.n_valid_observations``), then equals its value for
        the data with the dropped trial windows removed. (``delay`` and ``group_delay``
        correct those p-values for multiple comparisons jointly over every
        window of the call, so their selected frequencies match a per-window
        computation only window by window.)

        - A boolean array of shape ``(n_time_samples,)``, shared by every
          trial and signal; ``(n_time_samples, n_trials)``, shared by every
          signal; or ``(n_time_samples, n_trials, n_signals)``, where an
          artifact in any signal leaves the sample out of all: the
          observations, and so their weights, are shared across signals, as
          every cross-spectrum needs both signals of a pair at the same
          samples.
        - ``"nonfinite"``: every NaN or infinite sample is an artifact.
        - ``None`` (the default): no sample is left out.

        Construction warns when more than half of the
        trial windows are dropped, and when a window keeps no trial (it is
        NaN in every measure); a mask that leaves no trial window raises
        ``ValueError``. The artifact samples are set to zero before
        detrending and tapering, so the coefficients of a dropped window are
        finite unless it holds NaN or Inf the mask does not mark (they are
        unused either way), and the NaN warning reports only NaN or Inf in
        the trial windows kept. The mask is read by position, like
        ``time_series``. Provenance records the kind of mask (``"mask"`` or
        ``"nonfinite"``) with ``artifact_handling`` and the numbers of
        artifact samples and dropped trial windows, not the mask itself.
    artifact_handling : {"drop", "missing_data_tapers"}, default="drop"
        What to do with a trial's window that contains artifact samples; it
        needs ``artifact_samples``.

        - ``"drop"``: leave it out, as described under ``artifact_samples``.
        - ``"missing_data_tapers"``: estimate it from its valid samples with
          missing-data Slepian tapers [3]_ [4]_, the ``n_tapers`` sequences
          zero at its artifact samples that are most concentrated in the band
          ``[-W, W]``, ``W = time_halfbandwidth_product /
          n_time_samples_per_window`` cycles per sample (the leading
          eigenvectors of the sinc kernel restricted to the valid samples),
          with unit energy; the window is detrended over its valid samples.
          The tapers concentrated above 0.9, the ``is_low_bias`` rule, are
          kept and the others get observation weight 0
          (:attr:`n_kept_tapers`); a window keeping fewer than 2 is dropped,
          with a warning. Windows without artifacts are unchanged, and
          ``taper_weighting`` uses each window's own concentrations. Needs
          generated DPSS tapers, ``is_low_bias=True`` and at least 2 tapers.
          Construction finds the tapers of each distinct artifact pattern,
          about 2 ms per pattern for a 1000-sample window (with
          multithreaded OpenBLAS, as in older NumPy and SciPy wheels, it can be
          hundreds of times slower unless threadpoolctl is installed, which
          limits the solve to one thread, or ``OPENBLAS_NUM_THREADS=1`` is
          set). When to use it, its leakage and its
          gap limit:
          https://spectral-connectivity.readthedocs.io/en/latest/cookbook.html#leave-out-artifacts-and-missing-samples

    Attributes
    ----------
    fft : NDArray[complex128]
        Complex-valued FFT coefficients with shape
        (n_time_windows, n_trials, n_tapers, n_fft_samples, n_signals).
    frequencies : NDArray[float64], shape (n_fft_samples,)
        Frequency of each FFT bin in Hz, negative frequencies included, in FFT
        order. To label the frequency axis of a power or connectivity result,
        use ``Connectivity.frequencies`` (the non-negative bins) instead.
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

    With ``artifact_handling="missing_data_tapers"`` the windows with artifacts
    have tapers of their own, so line removal (``line_frequencies``), which
    fits the shared DPSS tapers to every sample of a window, is refused at
    construction.

    References
    ----------
    .. [1] Thomson, D. J. (1982). Spectrum estimation and harmonic analysis.
           Proceedings of the IEEE, 70(9), 1055-1096.
    .. [2] Percival, D. B., & Walden, A. T. (1993). Spectral Analysis for
           Physical Applications. Cambridge University Press.
    .. [3] Bronez, T. P. (1988). Spectral estimation of irregularly sampled
           multidimensional processes by generalized prolate spheroidal
           sequences. IEEE Transactions on Acoustics, Speech, and Signal
           Processing, 36(12), 1862-1873.
    .. [4] Chave, A. D. (2019). A multitaper spectral estimator for
           time-series with missing data. Geophysical Journal International,
           218(3), 2165-2178.

    Examples
    --------
    Using the helper function (recommended for 2D data):

    >>> import numpy as np
    >>> from spectral_connectivity.transforms import Multitaper, prepare_time_series
    >>> # EEG recording: 5 seconds at 1000 Hz, 64 channels
    >>> eeg_data = np.random.default_rng(0).standard_normal((5000, 64))  # Shape: (n_time, n_channels)
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
    >>> rng = np.random.default_rng(0)
    >>> signal = np.sin(2*np.pi*50*t) + 0.1*rng.standard_normal(len(t))
    >>> # Manually reshape to 3D: (n_time, n_trials, n_signals)
    >>> data = signal[:, np.newaxis, np.newaxis]  # Shape: (1000, 1, 1)
    >>> mt = Multitaper(data, sampling_frequency=fs, time_halfbandwidth_product=4)

    Multiple trials (already 3D):

    >>> # Epoched data: 100 trials, 5 channels, 1 second each at 1000 Hz
    >>> epoched_data = np.random.default_rng(0).standard_normal((1000, 100, 5))  # (n_time, n_trials, n_signals)
    >>> mt = Multitaper(epoched_data, sampling_frequency=1000)
    >>> print(f"Trials: {mt.n_trials}, Signals: {mt.n_signals}")
    Trials: 100, Signals: 5
    """

    _transform_name = "multitaper"
    if TYPE_CHECKING:
        # Plain attributes here. Declared for type checking only, so the API
        # docs do not list them undocumented.
        sampling_frequency: float
        detrend_type: str | None
        fft_workers: int | None
        artifact_handling: str

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
            "artifact_samples",
            "artifact_handling",
        }
    )
    _PROVENANCE_FIELDS: tuple[str, ...] = (
        *_WindowedFourierTransform._PROVENANCE_FIELDS,
        "frequency_resolution",
        "is_low_bias",
        "n_tapers",
        "taper_weighting",
        "adaptive_max_iterations",
        "adaptive_tolerance",
        "time_halfbandwidth_product",
    )
    # Set by _kept_trial_windows with artifact_handling="missing_data_tapers"
    # when some trial window has artifacts.
    _missing_data: _MissingDataTapers | None = None

    def __setattr__(self, name: str, value: Any) -> None:
        """Keep transform parameters immutable once construction has completed,
        and refuse line removal with missing-data tapers whichever is set
        first."""
        if getattr(self, "_initialized", False) and name in self._IMMUTABLE_PUBLIC_PARAMETERS:
            msg = (
                f"Multitaper.{name} is immutable after construction; create a new "
                "Multitaper instance with the desired parameters."
            )
            raise AttributeError(msg)
        if name in ("artifact_handling", "line_frequencies"):
            self._refuse_line_removal_with_missing_data_tapers(**{name: value})
        object.__setattr__(self, name, value)

    @reject_sampling_frequency_aliases
    def __init__(
        self,
        time_series: NDArray[np.floating],
        sampling_frequency: float,
        time_halfbandwidth_product: float = 3,
        detrend_type: str | None = "constant",
        time_window_duration: float | None = None,
        time_window_step: float | None = None,
        n_tapers: int | None = None,
        tapers: NDArray[np.floating] | None = None,
        start_time: float = 0,
        n_fft_samples: int | None = None,
        n_time_samples_per_window: int | None = None,
        n_time_samples_per_step: int | None = None,
        is_low_bias: bool = True,
        fft_workers: int | None = None,
        taper_weighting: Literal["uniform", "eigen", "adaptive"] = "uniform",
        adaptive_max_iterations: int = 50,
        adaptive_tolerance: float = 1e-8,
        artifact_samples: NDArray[np.bool_] | Literal["nonfinite"] | None = None,
        artifact_handling: Literal["drop", "missing_data_tapers"] = "drop",
    ) -> None:
        object.__setattr__(self, "_initialized", False)
        _validate_artifact_handling(
            artifact_handling, _ARTIFACT_HANDLINGS, type(self).__name__
        )
        time_series = _validated_time_series(time_series, type(self).__name__)
        self._time_series = _immutable_array_snapshot(time_series)
        _validate_fft_workers(fft_workers)
        _validate_detrend_type(detrend_type)
        self.fft_workers = fft_workers
        if taper_weighting not in {"uniform", "eigen", "adaptive"}:
            msg = (
                "taper_weighting must be 'uniform', 'eigen', or 'adaptive', got "
                f"{taper_weighting!r}. Use 'uniform' for the historical equal "
                "weighting, 'eigen' to weight tapers by their spectral "
                "concentration, or 'adaptive' for Thomson's iterative weighting."
            )
            raise ValueError(msg)
        if not is_positive_integer(adaptive_max_iterations):
            msg = (
                "adaptive_max_iterations must be a positive integer, got "
                f"{adaptive_max_iterations!r}."
            )
            raise ValueError(msg)
        if not np.isfinite(adaptive_tolerance) or adaptive_tolerance <= 0:
            msg = (
                f"adaptive_tolerance must be finite and positive, got {adaptive_tolerance!r}."
            )
            raise ValueError(msg)
        if tapers is not None and taper_weighting != "uniform":
            msg = (
                "Eigenvalue/adaptive weighting requires internally generated "
                "DPSS tapers; custom tapers do not provide concentration ratios. "
                "Use taper_weighting='uniform' with custom tapers, or omit the "
                "tapers argument to let Multitaper generate DPSS tapers."
            )
            raise ValueError(msg)
        self.taper_weighting = taper_weighting
        self.adaptive_max_iterations = int(adaptive_max_iterations)
        self.adaptive_tolerance = float(adaptive_tolerance)

        _check_time_series_is_3d(self._time_series)

        sampling_frequency = _validate_sampling_frequency(sampling_frequency)

        # Validate time_halfbandwidth_product
        if time_halfbandwidth_product < 1:
            msg = (
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
            raise ValueError(msg)

        # Warn if time_halfbandwidth_product is unusually large
        if time_halfbandwidth_product > 10:
            warnings.warn(
                f"time_halfbandwidth_product = {time_halfbandwidth_product} is unusually large.\n"
                "\n"
                "Values above 10 apply very heavy spectral smoothing and are rarely used.\n"
                "This will create many tapers and significantly slow computation.\n"
                "\n"
                "Common values are 1-5. If you're unsure, try 3 (a typical default).\n"
                "If you specifically need heavy smoothing, you can ignore this warning.",
                UserWarning,
                stacklevel=stacklevel_outside_package(),
            )

        # Validate time_window_duration
        if time_window_duration is not None and time_window_duration <= 0:
            msg = (
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
            raise ValueError(msg)
        _validate_window_step(time_window_duration, time_window_step)

        self.sampling_frequency = sampling_frequency
        self.time_halfbandwidth_product = time_halfbandwidth_product
        self.detrend_type = detrend_type
        self.is_low_bias = is_low_bias
        with _transposed_layout_hint(self._time_series):
            self._set_window_state(
                start_time,
                time_window_duration=time_window_duration,
                time_window_step=time_window_step,
                n_fft_samples=n_fft_samples,
                n_time_samples_per_window=n_time_samples_per_window,
                n_time_samples_per_step=n_time_samples_per_step,
            )
            # Resolve the window now, so one longer than the recording or of no
            # samples fails here.
            window = self.n_time_samples_per_window
        custom_tapers = None if tapers is None else _immutable_array_snapshot(tapers)
        # Reject a fractional (or boolean) n_tapers: a taper count is a whole
        # number, and dpss_windows would otherwise see a float.
        # Any real scalar (a 0-d array too) goes on to the checks below; a
        # boolean is a number to NumPy and gets their ValueError.
        if n_tapers is not None and (
            np.ndim(n_tapers) != 0 or np.asarray(n_tapers).dtype.kind not in "biuf"
        ):
            msg = (
                f"n_tapers must be an integer (a count of tapers), got "
                f"{type(n_tapers).__name__} {n_tapers!r}. Pass e.g. n_tapers=5, or "
                "None for the default floor(2 * time_halfbandwidth_product) - 1."
            )
            raise TypeError(msg)
        if n_tapers is not None and (
            isinstance(n_tapers, (bool, np.bool_))
            or not np.isfinite(n_tapers)
            or int(n_tapers) != n_tapers
        ):
            msg = f"n_tapers must be an integer, got {n_tapers!r}."
            raise ValueError(msg)
        self._n_tapers = n_tapers
        # A duration and a sample count for the same window or step must name
        # the same number of samples, as in ShortTimeFourierTransform: silently
        # preferring either one would analyze windows the caller did not ask for.
        for duration_name, duration, count_name, count in (
            (
                "time_window_duration",
                time_window_duration,
                "n_time_samples_per_window",
                n_time_samples_per_window,
            ),
            (
                "time_window_step",
                time_window_step,
                "n_time_samples_per_step",
                n_time_samples_per_step,
            ),
        ):
            if duration is not None and count is not None:
                duration_samples = _duration_to_samples(duration, sampling_frequency)
                if duration_samples != count:
                    msg = (
                        f"{duration_name}={duration} s and {count_name}={count} "
                        f"disagree: at sampling_frequency={sampling_frequency} Hz "
                        f"the duration is {duration_samples} samples. Give only one "
                        f"of them, or values that resolve to the same sample count."
                    )
                    raise ValueError(msg)
        # Validate custom tapers now; a mismatch would otherwise surface as an
        # opaque broadcasting error inside fft().
        if custom_tapers is not None and (
            custom_tapers.ndim != 2 or custom_tapers.shape[0] != window
        ):
            msg = (
                f"tapers must have shape ({window}, n_tapers) -- one column per "
                f"taper, n_time_samples_per_window rows -- got {custom_tapers.shape}."
            )
            raise ValueError(msg)
        if n_tapers is not None:
            # Check the count against the window, for custom tapers as for the
            # DPSS tapers generated below.
            if not 1 <= n_tapers <= window:
                msg = (
                    f"n_tapers must satisfy 1 <= n_tapers <= n_time_samples_per_window "
                    f"(= {window}), got {n_tapers}."
                )
                raise ValueError(msg)
            if custom_tapers is not None and custom_tapers.shape[1] != n_tapers:
                msg = (
                    f"n_tapers ({n_tapers}) disagrees with the {custom_tapers.shape[1]} "
                    "columns of tapers; pass one or make them agree."
                )
                raise ValueError(msg)
        self._tapers: BackendArray
        self._taper_eigenvalues: BackendArray | None
        if custom_tapers is not None:
            self._tapers, self._taper_eigenvalues = custom_tapers, None
        else:
            # Generate the DPSS tapers now, so every reported count is the
            # number used and a request low-bias filtering cuts short is
            # announced where it was made.
            with _transposed_layout_hint(self._time_series):
                self._tapers, self._taper_eigenvalues = self._dpss_tapers()
        if artifact_handling == "missing_data_tapers":
            self._check_missing_data_taper_settings(artifact_samples)
        self.artifact_handling = artifact_handling
        # Warned only once every check has passed, so an invalid Multitaper
        # (a transposed array among them) only raises.
        self._set_artifact_samples(artifact_samples)
        object.__setattr__(self, "_initialized", True)

    def _refuse_line_removal_with_missing_data_tapers(self, **new: Any) -> None:
        """Raise if line removal (``line_frequencies``, when Multitaper has
        it) would be combined with ``artifact_handling="missing_data_tapers"``;
        called by :meth:`__setattr__` with the value about to be set, before
        it is stored."""
        line_frequencies = new.get("line_frequencies", getattr(self, "line_frequencies", None))
        artifact_handling = new.get(
            "artifact_handling", getattr(self, "artifact_handling", "drop")
        )
        if (
            artifact_handling != "missing_data_tapers"
            or line_frequencies is None
            or np.size(line_frequencies) == 0
        ):
            return
        msg = (
            "line_frequencies cannot be combined with "
            "artifact_handling='missing_data_tapers': line removal fits each line "
            "with the shared DPSS tapers over every sample of the window, so it "
            "would treat the zero-filled artifact samples as data. Use "
            "artifact_handling='drop' with line_frequencies, or remove the lines "
            "before marking the artifacts."
        )
        raise ValueError(msg)

    def _check_missing_data_taper_settings(self, artifact_samples: Any) -> None:
        """Raise unless ``artifact_samples`` is given and the tapers are
        generated DPSS tapers, low-bias filtered, at least two of them: the
        missing-data tapers generalize them and keep those the same rule
        keeps."""
        if artifact_samples is None:
            reason = (
                "it sets how the trial windows containing artifact samples are "
                "estimated, and artifact_samples marks none. Pass artifact_samples "
                "(a boolean mask, True at an artifact, or 'nonfinite'), or leave "
                "artifact_handling at 'drop'."
            )
        elif self._taper_eigenvalues is None:
            reason = (
                "custom tapers are not DPSS tapers, so they have no missing-data "
                "counterpart. Omit tapers, or use artifact_handling='drop'."
            )
        elif not self.is_low_bias:
            reason = (
                "it keeps a window's missing-data tapers whose concentration is "
                f"above {MIN_EIGENVALUE_THRESHOLD}, the is_low_bias rule, and "
                "is_low_bias=False sets no such threshold. Use is_low_bias=True, or "
                "artifact_handling='drop'."
            )
        elif self.n_tapers < 2:
            reason = (
                "it keeps a window with an artifact only if at least 2 of its tapers "
                f"are kept, and this Multitaper uses {self.n_tapers}, so it would drop "
                "every such window. Use more tapers (n_tapers >= 2, with "
                "time_halfbandwidth_product of at least 1.5), or "
                "artifact_handling='drop'."
            )
        else:
            return
        msg = f"artifact_handling='missing_data_tapers' cannot be used here: {reason}"
        raise ValueError(msg)

    @property
    def _n_tapers_to_generate(self) -> int:
        """DPSS tapers to generate: ``n_tapers``, else the count NW implies."""
        if self._n_tapers is not None:
            return int(self._n_tapers)
        return estimate_n_tapers(self.time_halfbandwidth_product)

    def _dpss_tapers(self) -> tuple[BackendArray, BackendArray]:
        """The DPSS tapers and their eigenvalues, warning if low-bias filtering drops any."""
        requested = self._n_tapers_to_generate
        window = self.n_time_samples_per_window
        dpss, eigenvalues = dpss_windows(
            window,
            self.time_halfbandwidth_product,
            requested,
            is_low_bias=self.is_low_bias,
        )
        tapers = _immutable_array_snapshot(dpss.T * xp.sqrt(self.sampling_frequency))
        n_used = int(tapers.shape[1])
        if n_used < requested:
            if self._n_tapers is not None:
                using = (
                    f"{n_used} of the {requested} DPSS tapers you requested "
                    f"(n_tapers={requested})"
                )
                raise_nw = (
                    "; for more well-concentrated tapers raise "
                    f"time_halfbandwidth_product to about {(requested + 1) / 2:g}"
                )
            else:
                using = (
                    f"{n_used} of the default floor(2 * time_halfbandwidth_product) - 1 "
                    f"= {requested} DPSS tapers"
                )
                raise_nw = ""
            warnings.warn(
                f"Multitaper is using {using}. With time_halfbandwidth_product="
                f"{self.time_halfbandwidth_product} and a {window}-sample window only "
                f"{n_used} are well concentrated (concentration > "
                f"{MIN_EIGENVALUE_THRESHOLD}); is_low_bias=True drops the rest because "
                "their leakage would bias the estimate. To keep them anyway pass "
                "is_low_bias=False (transform_kwargs={'is_low_bias': False} in "
                f"multitaper_connectivity){raise_nw}; to silence this and keep the "
                f"result, pass n_tapers={n_used}.",
                UserWarning,
                stacklevel=stacklevel_outside_package(),
            )
        return tapers, _immutable_array_snapshot(eigenvalues)

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
            f"n_tapers={self.n_tapers}{self._artifact_samples_repr()})"
        )

    def _spectral_bandwidth_metadata(self) -> dict[str, Any]:
        """Taper kind and, for generated DPSS tapers, their qualified bandwidth."""
        # Generated DPSS tapers carry concentration ratios; custom tapers never do.
        if self._taper_eigenvalues is None:
            return {"taper_kind": "custom"}
        return {
            "taper_kind": "dpss",
            "spectral_bandwidth": self.frequency_resolution,
            "spectral_bandwidth_definition": "full_dpss_concentration_bandwidth",
            "spectral_bandwidth_units": "Hz",
        }

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
        >>> data = np.random.default_rng(0).standard_normal((5000, 1, 64))  # 5s, 64 EEG channels
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
        n_tapers: int | str = self.n_tapers
        if self._taper_eigenvalues is not None and self.n_tapers < self._n_tapers_to_generate:
            n_tapers = (
                f"{self.n_tapers} (of {self._n_tapers_to_generate} requested; low-bias "
                "filtering dropped the rest)"
            )
        spectral_rows: list[tuple[str, Any]] = [
            ("Time-halfbandwidth product", self.time_halfbandwidth_product),
            ("Number of tapers", n_tapers),
        ]
        return self._summary(
            "Multitaper Spectral Analysis Configuration",
            spectral_rows,
            ("Frequency resolution", f"{self.frequency_resolution:.1f} Hz"),
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
        return _readonly_array_copy(self._tapers)

    @property
    def taper_eigenvalues(self) -> NDArray[np.floating] | None:
        """DPSS spectral-concentration ratios used for taper weighting.

        Returns ``None`` for custom tapers, whose concentration ratios are not
        known. The returned array is a detached read-only copy.
        """
        if self._taper_eigenvalues is None:
            return None
        return _readonly_array_copy(self._taper_eigenvalues)

    @property
    def n_tapers(self) -> int:
        """Return the number of tapers used.

        The number of columns of custom ``tapers``, or of the generated DPSS
        tapers. That is the requested ``n_tapers`` (by default
        ``floor(2 * time_halfbandwidth_product) - 1``) less any that low-bias
        filtering dropped, which construction announces with a warning.

        Returns
        -------
        int
            Number of tapers used.

        """
        return int(self._tapers.shape[1])

    def _provenance_metadata(self) -> dict[str, Any]:
        """Native settings, with the ``n_tapers`` argument as given.

        ``n_tapers`` is the count used; ``requested_n_tapers`` is the argument
        (``None`` for the default), which with ``is_low_bias`` reproduces it.
        """
        metadata = super()._provenance_metadata()
        metadata["requested_n_tapers"] = self._n_tapers
        if self._artifact_samples_mode is not None and self.artifact_handling != "drop":
            metadata["n_missing_data_taper_trial_windows"] = self._n_missing_data_trial_windows
        return metadata

    @property
    def frequency_resolution(self) -> float:
        """Return frequency resolution (full DPSS concentration bandwidth).

        This is ``2 * time_halfbandwidth_product / time_window_duration``,
        using the resolved window duration. It describes spectral smoothing,
        independently of the FFT bin spacing and any zero-padding. For custom
        tapers the property retains the nominal DPSS value specified by NW;
        their actual bandwidth requires a definition for the supplied tapers.

        Returns
        -------
        float
            Frequency resolution in Hz.

        """
        return estimate_frequency_resolution(
            self.sampling_frequency, self.time_window_duration, self.time_halfbandwidth_product
        )

    @property
    def observations_are_independent(self) -> bool:
        """Whether the trial/taper observations may be counted as independent.

        DPSS tapers are orthogonal, so for a process that is smooth across the
        taper bandwidth the eigencoefficients of one window are approximately
        uncorrelated (Thomson 1982; Percival & Walden 1993, ch. 7), and trials
        are independent realizations. :class:`Connectivity` reads this flag to
        decide whether ``n_observations`` (trials x tapers) counts effectively
        independent samples: the jackknife, the debiased measures (pairwise
        phase consistency, debiased squared PLI/WPLI) and the zero-coherence
        significance test (``Beta(1, n_observations - 1)`` null) all assume it
        does. ``True`` for DPSS tapers and for custom tapers that are mutually
        orthogonal; ``False`` for custom tapers that are not, whose
        eigencoefficients are correlated even for white noise (copies of one
        taper give identical ones). The flag describes the observations within
        one window; correlation between overlapping windows, which matters
        only for expectations that also average over time, is reported by
        :attr:`time_bins_are_independent`.

        Returns
        -------
        bool
        """
        if self._taper_eigenvalues is not None:
            return True  # DPSS tapers, generated here
        # For white noise the eigencoefficients of tapers k and l correlate by
        # <w_k, w_l> / (|w_k| |w_l|). Orthogonal tapers computed in floating
        # point leave that at rounding level; sqrt(eps) admits orthogonal
        # tapers computed elsewhere to a lower precision.
        # Custom tapers may be integers (a boxcar of ones, say).
        tapers = self._tapers.astype(
            xp.result_type(self._tapers.dtype, xp.float32), copy=False
        )
        gram = xp.matmul(tapers.T, tapers)
        norms = xp.sqrt(xp.diagonal(gram))
        correlation = xp.abs(gram) / (norms[:, xp.newaxis] * norms[xp.newaxis, :])
        off_diagonal = ~xp.eye(gram.shape[0], dtype=bool)
        tolerance = float(np.sqrt(xp.finfo(gram.dtype).eps))
        return not bool(xp.any(correlation[off_diagonal] > tolerance))

    def _weight_taper_coefficients(
        self,
        coefficients: NDArray[np.complexfloating],
        windows: NDArray[np.floating],
    ) -> NDArray[np.complexfloating]:
        """Apply ``taper_weighting`` to the DPSS eigencoefficients."""
        if self.taper_weighting == "uniform":
            return coefficients

        assert self._taper_eigenvalues is not None
        eigenvalues = self._taper_eigenvalues
        missing_data = self._missing_data if self._rescues_windows else None
        if missing_data is not None:
            # Each trial window's own concentrations, zero past its kept tapers.
            eigenvalues = missing_data.concentrations[missing_data.table_index]
        if self.taper_weighting == "eigen":
            weights: NDArray[np.floating] = xp.sqrt(eigenvalues)
            if missing_data is None:
                weights = weights / xp.sqrt(xp.mean(weights**2))
                return (
                    coefficients * weights[xp.newaxis, xp.newaxis, :, xp.newaxis, xp.newaxis]
                )
            n_kept = xp.count_nonzero(eigenvalues, axis=-1)[..., xp.newaxis]
            weights = weights / xp.sqrt(_mean_square_over_kept(weights, n_kept, axis=-1))
            return coefficients * weights[..., xp.newaxis, xp.newaxis]

        # Thomson's adaptive weights balance the taper periodogram against the
        # process-noise level, which must be on the *same* power-spectral-density
        # scale. ``fft()`` returns ``fft(taper * x) / sampling_frequency`` with
        # tapers scaled by ``sqrt(sampling_frequency)``, so ``|coefficients| ** 2``
        # is a PSD (signal ** 2 / Hz) while ``var(windows)`` is a raw
        # time-domain variance (signal ** 2). Dividing by ``sampling_frequency``
        # puts the noise term on the PSD scale; without it the noise is inflated
        # by a factor of ``sampling_frequency`` and the weighting is miscalibrated.
        noise_power_spectral_density = xp.var(windows, axis=-1) / self.sampling_frequency
        if missing_data is not None:
            # A window with missing-data tapers: the variance of its valid samples.
            valid = missing_data.valid_of_rescued()
            centered = _detrend_valid_samples(windows[missing_data.windows], valid, "constant")
            noise_power_spectral_density[missing_data.windows] = (
                xp.sum(centered**2, axis=-1) / xp.sum(valid, axis=-1) / self.sampling_frequency
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
        return self._tapers

    def _kept_trial_windows(self, artifact_windows: BackendArray) -> BackendArray:
        """The trial windows without artifacts and, with missing-data tapers,
        those with artifacts that keep at least 2 tapers.

        With ``artifact_handling="missing_data_tapers"`` this computes the
        missing-data tapers of every distinct artifact pattern among the trial
        windows (:func:`_missing_data_taper_sets`) and records the
        ``_gapped_*`` attributes that :meth:`_estimate_gapped_windows` and
        :meth:`_weight_taper_coefficients` read, and ``_n_kept_tapers``.
        """
        clean = super()._kept_trial_windows(artifact_windows)
        if self.artifact_handling == "drop" or bool(xp.all(clean)):
            return clean
        window_index, trial_index = np.nonzero(~to_numpy(clean))
        patterns, pattern_index = np.unique(
            to_numpy(artifact_windows[xp.asarray(window_index), xp.asarray(trial_index)]),
            axis=0,
            return_inverse=True,
        )
        pattern_index = pattern_index.reshape(-1)
        n_kept, tapers, concentrations = _missing_data_taper_sets(
            ~patterns,
            self.time_halfbandwidth_product,
            self.n_tapers,
            to_numpy(self._tapers),
        )
        assert self._taper_eigenvalues is not None  # checked at construction
        is_kept = n_kept[pattern_index] > 0
        # Each pattern's row in the tables: 1, 2, ... for the kept ones.
        pattern_row = np.cumsum(n_kept > 0) * (n_kept > 0)
        table_index = np.zeros(clean.shape, dtype=np.intp)
        table_index[window_index, trial_index] = pattern_row[pattern_index]
        n_time_samples = self.n_time_samples_per_window
        self._missing_data = _MissingDataTapers(
            table_index=xp.asarray(table_index),
            windows=(xp.asarray(window_index[is_kept]), xp.asarray(trial_index[is_kept])),
            tapers=xp.concatenate(
                [
                    self._tapers[xp.newaxis],
                    xp.asarray(tapers * np.sqrt(self.sampling_frequency)),
                ]
            ),
            concentrations=xp.concatenate(
                [self._taper_eigenvalues[xp.newaxis], xp.asarray(concentrations)]
            ),
            valid=xp.asarray(
                np.concatenate(
                    [np.ones((1, n_time_samples), dtype=bool), ~patterns[n_kept > 0]]
                )
            ),
            n_gapped=int(window_index.size),
            dropped_summary=(
                "" if is_kept.all() else _dropped_gap_summary(patterns[n_kept == 0])
            ),
        )
        counts = np.full(clean.shape, self.n_tapers, dtype=np.intp)
        counts[window_index, trial_index] = n_kept[pattern_index]
        if np.any(n_kept[pattern_index[is_kept]] < self.n_tapers):
            self._n_kept_tapers = _immutable_array_snapshot(xp.asarray(counts))
        kept_windows: BackendArray = xp.asarray(counts > 0)
        return kept_windows

    @property
    def _rescues_windows(self) -> bool:
        """Whether missing-data tapers estimate some trial window."""
        return self._missing_data is not None and self._missing_data.n_rescued > 0

    @property
    def _n_gapped_trial_windows(self) -> int:
        """The trial windows with artifacts, with missing-data tapers."""
        return 0 if self._missing_data is None else self._missing_data.n_gapped

    @property
    def _n_missing_data_trial_windows(self) -> int:
        """The trial windows with artifacts that missing-data tapers keep."""
        return 0 if self._missing_data is None else self._missing_data.n_rescued

    def _dropped_window_notes(self) -> list[tuple[str, str]]:
        """With missing-data tapers, the windows with artifacts dropped anyway."""
        n_dropped = self._n_gapped_trial_windows - self._n_missing_data_trial_windows
        if self.artifact_handling == "drop" or n_dropped == 0:
            return []
        return [
            (
                (
                    f"artifact_handling='missing_data_tapers': {n_dropped} of "
                    f"{self._n_gapped_trial_windows} trial windows with artifacts kept "
                    "fewer than 2 tapers and were dropped."
                ),
                f"{type(self).__name__}.n_kept_tapers shows what each window keeps.",
            )
        ]

    def _window_drop_rule(self) -> str:
        """When a trial's window is dropped, for the warnings."""
        if self.artifact_handling == "drop":
            return super()._window_drop_rule()
        recurrence = self.n_time_samples_per_window / (2 * self.time_halfbandwidth_product)
        return (
            "With artifact_handling='missing_data_tapers' a trial's window of "
            f"{self._window_length} "
            "is dropped if fewer than 2 of its missing-data tapers have enough of "
            f"their energy ({MIN_EIGENVALUE_THRESHOLD:.0%}) inside the frequency band: "
            "too few of its samples are valid, or its artifacts recur more often than "
            f"every 1/(2W) = {recurrence:.3g} samples "
            f"({recurrence / self.sampling_frequency:.3g} s, one over the frequency "
            "resolution), which caps that energy at the fraction of samples valid. "
            f"{'' if self._missing_data is None else self._missing_data.dropped_summary}"
            "To keep such windows use a larger "
            "time_halfbandwidth_product (a wider band, so that 1/(2W) falls below "
            "the artifacts' spacing) or a longer window (more valid samples around "
            "a long artifact), or artifact_handling='drop', which drops every "
            "window with an artifact."
        )

    def _detrend_gapped_windows(self, windows: NDArray[np.floating]) -> None:
        """Detrend each kept window with artifacts over its valid samples.

        That equals detrending the samples before zero-filling, as the fit of
        the zero-filled window is a constant or line too.
        """
        if not self._rescues_windows:
            return
        assert self._missing_data is not None
        rescued = self._missing_data.windows
        windows[rescued] = _detrend_valid_samples(
            windows[rescued], self._missing_data.valid_of_rescued(), self.detrend_type
        )

    def _estimate_gapped_windows(
        self,
        coefficients: NDArray[np.complexfloating],
        windows: NDArray[np.floating],
    ) -> None:
        """Recompute the kept windows with artifacts with their own
        missing-data tapers.

        Only those windows' tapers are gathered, ``(n_rescued, 1,
        n_time_samples_per_window, n_tapers)``, rather than a set for every
        trial window: the DPSS coefficients of the others are already right.
        """
        if not self._rescues_windows:
            return
        assert self._missing_data is not None
        rescued = self._missing_data.windows
        tapers = self._missing_data.tapers[self._missing_data.table_index[rescued]]
        coefficients[rescued] = _multitaper_fft(
            tapers[:, xp.newaxis],
            windows[rescued],
            self.n_fft_samples,
            self.sampling_frequency,
            workers=self.fft_workers,
        )

    def _artifact_samples_details(self) -> str:
        """With missing-data tapers, the trial windows that had artifacts and
        those of them rescued."""
        if self.artifact_handling == "drop":
            return ""
        return (
            f"; {self._n_gapped_trial_windows} had artifacts, "
            f"{self._n_missing_data_trial_windows} rescued with missing-data tapers"
        )

    def _artifact_samples_repr(self) -> str:
        """The base entry, followed by a non-default ``artifact_handling``."""
        entry = super()._artifact_samples_repr()
        if self.artifact_handling == "drop":
            return entry
        return f"{entry},\n           artifact_handling={self.artifact_handling!r}"


class ShortTimeFourierTransform(_WindowedFourierTransform):
    """Short-time Fourier transform using an L2-normalized Hann window.

    The returned coefficients follow the same five-dimensional contract as
    :class:`Multitaper`, with a singleton taper axis. This makes the transform
    directly usable with :meth:`Connectivity.from_transform` and
    :func:`spectral_connectivity.wrapper.connectivity_to_xarray`.

    Parameters
    ----------
    time_series : ndarray, shape (n_time_samples, n_trials, n_signals)
        Real-valued input signals (a complex series raises ``TypeError``).
        Use :func:`prepare_time_series` for 1-D/2-D input. An
        ``xarray.DataArray`` is read by position; one whose dimension names
        put time, trials, or signals on another axis raises ``ValueError``.
    sampling_frequency : float
        Samples per second; required because it labels the frequency axis and scales power.
    detrend_type : {"constant", "linear"} or None, default="constant"
        Detrending applied to each window before the FFT.
    time_window_duration : float, optional
        Window length in seconds. Give this or ``n_time_samples_per_window``.
    time_window_step : float, optional
        Step between successive windows in seconds (defaults to the window
        duration, i.e. no overlap).
    start_time : float, default=0
        Time of the first sample, in seconds (scalar only).
    n_fft_samples : int, optional
        FFT length. Defaults to ``scipy.fft.next_fast_len`` of the window
        length, which may zero-pad (e.g. a 257-sample window gives 264 bins);
        pass the window length explicitly for an unpadded frequency grid.
        The directed measures (spectral Granger, DTF, PDC) depend slightly
        on this length: see the Notes of
        :func:`~spectral_connectivity.minimum_phase_decomposition.minimum_phase_decomposition`.
    n_time_samples_per_window : int, optional
        Window length in samples (alternative to ``time_window_duration``).
    n_time_samples_per_step : int, optional
        Step between windows in samples (alternative to ``time_window_step``).
    fft_workers : int, optional
        Worker threads for SciPy's CPU FFT (``-1`` uses all cores).
    artifact_samples : NDArray[bool] or "nonfinite", optional
        Samples to leave out, ``True`` at an artifact, as in :class:`Multitaper`
        (see its parameter): a trial's window containing an artifact sample is
        dropped from every expectation.
    artifact_handling : {"drop"}, default="drop"
        Only ``"drop"``; ``"missing_data_tapers"``, which estimates such a
        window from its valid samples, is a :class:`Multitaper` option, as
        those tapers generalize its DPSS tapers.
    """

    _transform_name = "stft"
    # A 1-sample "Hann" window is the rectangular window [1.0], not a taper.
    _MIN_WINDOW_SAMPLES = 2

    @reject_sampling_frequency_aliases
    def __init__(
        self,
        time_series: NDArray[np.floating],
        sampling_frequency: float,
        *,
        detrend_type: str | None = "constant",
        time_window_duration: float | None = None,
        time_window_step: float | None = None,
        start_time: float = 0,
        n_fft_samples: int | None = None,
        n_time_samples_per_window: int | None = None,
        n_time_samples_per_step: int | None = None,
        fft_workers: int | None = None,
        artifact_samples: NDArray[np.bool_] | Literal["nonfinite"] | None = None,
        artifact_handling: Literal["drop"] = "drop",
    ) -> None:
        _validate_artifact_handling(artifact_handling, ("drop",), type(self).__name__)
        sampling_frequency = _validate_sampling_frequency(sampling_frequency)
        for name, value in (
            ("time_window_duration", time_window_duration),
            ("time_window_step", time_window_step),
        ):
            if value is not None and (not np.isfinite(value) or value <= 0):
                msg = f"{name} must be finite and positive."
                raise ValueError(msg)
        if n_time_samples_per_window is not None and not is_positive_integer(
            n_time_samples_per_window, minimum=self._MIN_WINDOW_SAMPLES
        ):
            msg = (
                "n_time_samples_per_window must be an integer of at least "
                f"{self._MIN_WINDOW_SAMPLES}."
            )
            raise ValueError(msg)
        if n_time_samples_per_step is not None and not is_positive_integer(
            n_time_samples_per_step
        ):
            msg = "n_time_samples_per_step must be a positive integer."
            raise ValueError(msg)
        time_series = _validated_time_series(time_series, type(self).__name__)
        self._time_series = _immutable_array_snapshot(time_series)
        _check_time_series_is_3d(self._time_series)
        for duration, count, mismatch_message in (
            (
                time_window_duration,
                n_time_samples_per_window,
                (
                    "time_window_duration and n_time_samples_per_window resolve to "
                    "different Hann window lengths."
                ),
            ),
            (
                time_window_step,
                n_time_samples_per_step,
                "time_window_step and n_time_samples_per_step resolve to different step lengths.",
            ),
        ):
            if (
                duration is not None
                and count is not None
                and _duration_to_samples(duration, sampling_frequency) != count
            ):
                raise ValueError(mismatch_message)
        _validate_fft_workers(fft_workers)
        _validate_detrend_type(detrend_type)
        self._sampling_frequency = sampling_frequency
        self._detrend_type = detrend_type
        self._fft_workers = fft_workers
        self._set_window_state(
            start_time,
            time_window_duration=time_window_duration,
            time_window_step=time_window_step,
            n_fft_samples=n_fft_samples,
            # NumPy integer counts are stored as Python ints.
            n_time_samples_per_window=(
                None if n_time_samples_per_window is None else int(n_time_samples_per_window)
            ),
            n_time_samples_per_step=(
                None if n_time_samples_per_step is None else int(n_time_samples_per_step)
            ),
        )
        # Resolve the window and step now, so a window that is too short or does
        # not fit the signal, or a step shorter than one sample, fails here
        # rather than in fft().
        with _transposed_layout_hint(self._time_series):
            window_samples = self.n_time_samples_per_window
        window = scipy_hann(window_samples, sym=False)
        self._window = _immutable_array_snapshot(
            (window / np.linalg.norm(window) * np.sqrt(sampling_frequency))[:, np.newaxis]
        )
        _validate_window_step(self.time_window_duration, self.time_window_step)
        self._set_artifact_samples(artifact_samples)

    @property
    def sampling_frequency(self) -> float:
        """Samples per second, in Hz."""
        return self._sampling_frequency

    @property
    def detrend_type(self) -> str | None:
        """Detrending applied to each window before the FFT."""
        return self._detrend_type

    @property
    def fft_workers(self) -> int | None:
        """Worker threads for SciPy's CPU FFT (``None`` uses SciPy's default)."""
        return self._fft_workers

    @property
    def artifact_handling(self) -> str:
        """How trial windows containing artifact samples are handled: ``"drop"``."""
        return "drop"

    @property
    def observations_are_independent(self) -> bool:
        """Whether the trial observations may be counted as independent.

        Each window has one Hann-windowed estimate per trial, and trials are
        independent realizations, so this is always ``True``. Correlation
        between overlapping windows, which matters only for expectations that
        also average over time, is reported by :attr:`time_bins_are_independent`.

        Returns
        -------
        bool
        """
        return True

    def _tapers_for_fft(self) -> NDArray[np.floating]:
        """Return the scaled Hann window, ``(n_time_samples_per_window, 1)``."""
        return self._window

    @property
    def equivalent_noise_bandwidth(self) -> float:
        """
        Return periodic Hann equivalent noise bandwidth in Hz.

        The resolved window determines the bandwidth independently of FFT
        zero-padding and frequency-grid spacing.

        Returns
        -------
        float
            Equivalent noise bandwidth of the resolved window, in Hz.

        See Also
        --------
        Welch.equivalent_noise_bandwidth : Equivalent noise bandwidth of Welch segments.

        Notes
        -----
        ``sampling_frequency * sum(window**2) / sum(window)**2`` equals
        ``1.5 / time_window_duration`` for windows of at least three samples.
        The two-sample Hann window has bandwidth ``sampling_frequency``.

        Examples
        --------
        >>> stft = ShortTimeFourierTransform(
        ...     np.zeros((100, 1, 1)), 100, time_window_duration=0.5
        ... )
        >>> round(stft.equivalent_noise_bandwidth, 6)
        3.0
        """
        window = self._window[:, 0]
        return float(
            to_numpy(self.sampling_frequency * xp.sum(window**2) / xp.sum(window) ** 2)
        )

    def __repr__(self) -> str:
        """Return the STFT's settings; its single Hann window has no NW or tapers."""
        return (
            "ShortTimeFourierTransform("
            f"sampling_frequency={self.sampling_frequency!r}, window='hann_periodic',\n"
            f"                          time_window_duration={self.time_window_duration!r}, "
            f"time_window_step={self.time_window_step!r},\n"
            f"                          detrend_type={self.detrend_type!r}, "
            f"start_time={self.start_time}{self._artifact_samples_repr()})"
        )

    def summarize_parameters(self) -> str:
        """Human-readable summary of the STFT's settings.

        Returns
        -------
        summary : str
            Data shape, the Hann window, time windowing, and the window's
            equivalent noise bandwidth, Nyquist frequency, and FFT length.

        Examples
        --------
        >>> import numpy as np
        >>> from spectral_connectivity.transforms import ShortTimeFourierTransform
        >>> stft = ShortTimeFourierTransform(
        ...     np.zeros((1000, 1, 2)), sampling_frequency=100, time_window_duration=0.5
        ... )
        >>> print(stft.summarize_parameters().splitlines()[0])
        Short-Time Fourier Transform Configuration
        """
        return self._summary(
            "Short-Time Fourier Transform Configuration",
            [("Window", "Hann (periodic, L2-normalized)")],
            ("Equivalent noise bandwidth", f"{self.equivalent_noise_bandwidth:.1f} Hz"),
        )

    def _spectral_bandwidth_metadata(self) -> dict[str, Any]:
        """The Hann window and its equivalent noise bandwidth."""
        return {
            "window": "hann_periodic",
            "spectral_bandwidth": self.equivalent_noise_bandwidth,
            "spectral_bandwidth_definition": "equivalent_noise_bandwidth",
            "spectral_bandwidth_units": "Hz",
        }


class Welch:
    """Welch spectral transform using overlapping Hann-windowed segments.

    Segment coefficients are represented on the taper/observation axis, so the
    default ``trials_tapers`` expectation averages both trials and Welch
    segments and returns a single spectrum centered on the analyzed record.

    Parameters
    ----------
    time_series : ndarray, shape (n_time_samples, n_trials, n_signals)
        Real-valued input signals (a complex series raises ``TypeError``).
        Use :func:`prepare_time_series` for 1-D/2-D input. An
        ``xarray.DataArray`` is read by position; one whose dimension names
        put time, trials, or signals on another axis raises ``ValueError``.
    sampling_frequency : float
        Samples per second; required because it labels the frequency axis and scales power.
    segment_duration : float, optional
        Segment length in seconds; sets the spectral bandwidth
        (:attr:`equivalent_noise_bandwidth`, ``1.5 / segment_duration`` Hz for
        the Hann window). Strongly recommended -- the fallback of 256 samples
        does not scale with the sampling rate. Give this or
        ``n_time_samples_per_segment``.
    segment_overlap : float, default=0.5
        Fractional overlap between successive segments, in ``[0, 1)``.
    n_time_samples_per_segment : int, optional
        Segment length in samples (alternative to ``segment_duration``).
    detrend_type : {"constant", "linear"} or None, default="constant"
        Detrending applied to each segment before the FFT.
    start_time : float, default=0
        Time of the first sample, in seconds (scalar only).
    n_fft_samples : int, optional
        FFT length. Defaults to ``scipy.fft.next_fast_len`` of the segment
        length, which may zero-pad (e.g. a 257-sample segment gives 264 bins);
        pass the segment length explicitly for an unpadded frequency grid.
        The directed measures (spectral Granger, DTF, PDC) depend slightly
        on this length: see the Notes of
        :func:`~spectral_connectivity.minimum_phase_decomposition.minimum_phase_decomposition`.
    fft_workers : int, optional
        Worker threads for SciPy's CPU FFT (``-1`` uses all cores).
    """

    _transform_name = "welch"
    is_one_sided = False

    @reject_sampling_frequency_aliases
    def __init__(
        self,
        time_series: NDArray[np.floating],
        sampling_frequency: float,
        *,
        segment_duration: float | None = None,
        segment_overlap: float = 0.5,
        n_time_samples_per_segment: int | None = None,
        detrend_type: str | None = "constant",
        start_time: float = 0,
        n_fft_samples: int | None = None,
        fft_workers: int | None = None,
    ) -> None:
        sampling_frequency = _validate_sampling_frequency(sampling_frequency)
        if segment_duration is not None and (
            not np.isfinite(segment_duration) or segment_duration <= 0
        ):
            msg = "segment_duration must be finite and positive."
            raise ValueError(msg)
        if n_time_samples_per_segment is not None and not is_positive_integer(
            n_time_samples_per_segment, minimum=2
        ):
            msg = "n_time_samples_per_segment must be an integer of at least 2."
            raise ValueError(msg)
        time_series = _validated_time_series(time_series, type(self).__name__)
        # np.shape reads the array's own shape, without a copy.
        data_shape = np.shape(time_series)
        n_time_samples = int(data_shape[0]) if data_shape else 0
        segment_samples = _resolve_sample_count(
            n_time_samples_per_segment,
            segment_duration,
            sampling_frequency,
            "segment_duration and n_time_samples_per_segment resolve to different lengths.",
        )
        uses_default_segment = segment_samples is None
        if segment_samples is None:
            segment_samples = min(256, n_time_samples)
        if not np.isfinite(segment_overlap) or not 0 <= segment_overlap < 1:
            msg = "segment_overlap must satisfy 0 <= overlap < 1."
            raise ValueError(msg)
        # An explicit n_time_samples_per_segment was checked above, so a short
        # segment comes from the duration or from a short time series.
        if segment_samples < 2:
            if uses_default_segment:
                msg = (
                    "Welch needs a segment of at least 2 samples, but the time series "
                    f"has shape {data_shape}; pass a time series of shape "
                    "(n_time_samples, n_trials, n_signals) with at least 2 time samples."
                )
            else:
                msg = (
                    f"segment_duration={segment_duration} s is {segment_samples} "
                    f"sample{'' if segment_samples == 1 else 's'} "
                    f"at sampling_frequency={sampling_frequency:g} Hz; a Welch segment "
                    f"needs at least 2 samples (segment_duration >= "
                    f"{2 / sampling_frequency} s)."
                )
            raise ValueError(msg)
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
        self._segment_overlap = float(segment_overlap)
        # SciPy's historical 256-sample default does not scale with the
        # sampling rate, so at typical electrophysiology rates it yields a very
        # short segment and a wide spectral bandwidth (e.g. ~176 Hz at 30 kHz),
        # collapsing neuroscience bands. Warn for a default segment shorter than
        # 0.25 s so the choice is explicit rather than silent.
        if uses_default_segment and sampling_frequency / segment_samples > 4:
            warnings.warn(
                f"Welch is using the default segment length of "
                f"{segment_samples} samples, giving a "
                f"{self.segment_duration * 1e3:.1f} ms segment with a "
                f"{self.equivalent_noise_bandwidth:.1f} Hz equivalent noise "
                f"bandwidth at sampling_frequency={sampling_frequency:g} Hz. This "
                "is too coarse for typical neuroscience bands. Pass "
                "segment_duration (seconds) or n_time_samples_per_segment to set "
                "the segment length explicitly.",
                UserWarning,
                stacklevel=stacklevel_outside_package(),
            )

    # The settings below are read from the internal STFT that does the
    # computation, so they cannot be reassigned out of step with it.
    @property
    def _time_series_is_finite(self) -> bool:
        return self._stft._time_series_is_finite

    @property
    def sampling_frequency(self) -> float:
        """Samples per second, in Hz."""
        return self._stft.sampling_frequency

    @property
    def n_time_samples_per_segment(self) -> int:
        """Segment length in samples."""
        return int(self._stft.n_time_samples_per_window)

    @property
    def segment_duration(self) -> float:
        """Segment length in seconds, after rounding to whole samples."""
        return self.n_time_samples_per_segment / self.sampling_frequency

    @property
    def segment_overlap(self) -> float:
        """Requested overlap between segments, as a fraction of a segment.

        The realized step, ``n_time_samples_per_step``, is this fraction's
        complement of a segment rounded to whole samples (at least one).
        """
        return self._segment_overlap

    @property
    def n_time_samples_per_step(self) -> int:
        """Realized step between segment starts, in samples."""
        return int(self._stft.n_time_samples_per_step)

    @property
    def detrend_type(self) -> str | None:
        """Detrending applied to each segment before the FFT."""
        return self._stft.detrend_type

    @property
    def start_time(self) -> Any:
        """Time of the first sample, in seconds."""
        return self._stft.start_time

    @property
    def n_fft_samples(self) -> int:
        """FFT length of each segment."""
        return int(self._stft.n_fft_samples)

    @property
    def fft_workers(self) -> int | None:
        """Worker threads for SciPy's CPU FFT (``None`` uses SciPy's default)."""
        return self._stft.fft_workers

    @property
    def equivalent_noise_bandwidth(self) -> float:
        """
        Return the segment's periodic Hann equivalent noise bandwidth.

        Uses the actual rounded segment length and the same Hann-window
        definition as :class:`ShortTimeFourierTransform`.

        Returns
        -------
        float
            Equivalent noise bandwidth of the resolved segment, in Hz.

        See Also
        --------
        ShortTimeFourierTransform.equivalent_noise_bandwidth : Hann-window bandwidth.

        Examples
        --------
        >>> welch = Welch(np.zeros((100, 1, 1)), 100, segment_duration=0.5)
        >>> round(welch.equivalent_noise_bandwidth, 6)
        3.0
        """
        return self._stft.equivalent_noise_bandwidth

    @property
    def frequencies(self) -> NDArray[np.floating]:
        """Frequency of each FFT bin in Hz, in standard FFT order.

        These label the coefficients of :meth:`fft`, negative frequencies
        included. To label the frequency axis of a power or connectivity
        result, use :attr:`Connectivity.frequencies
        <spectral_connectivity.Connectivity.frequencies>`, the non-negative
        bins.

        Returns
        -------
        frequencies : array, shape (n_fft_samples,)
        """
        return self._stft.frequencies

    @property
    def time(self) -> NDArray[np.floating]:
        """Mean of the segment center times, in seconds.

        Returns
        -------
        time : array, shape (1,)
            Welch averages all segments into a single time bin.
        """
        return xp.asarray(self._stft.time).mean(keepdims=True)

    @property
    def n_trials(self) -> int:
        """Number of trials in the time series."""
        return self._stft.n_trials

    @property
    def n_signals(self) -> int:
        """Number of signals in the time series."""
        return self._stft.n_signals

    @property
    def n_segments(self) -> int:
        """Number of (overlapping) segments averaged by the estimate."""
        return len(self._stft.time)

    @property
    def observations_are_independent(self) -> bool:
        """Whether the segments may be counted as independent observations.

        Welch's estimate averages the periodograms of overlapping Hann-windowed
        segments. For a Hann window the correlation between the periodograms
        of segments overlapping by 50% is small, so up to that overlap the
        average has close to the degrees of freedom of independent segments
        (Welch 1967; Percival & Walden 1993, sec. 6.17). Beyond 50% the
        additional segments are strongly correlated with their neighbors and
        add little independent information, so ``n_observations`` (trials x
        segments) overstates the effective sample size. :class:`Connectivity`
        reads this flag before the jackknife, the debiased measures (pairwise
        phase consistency, debiased squared PLI/WPLI) and the zero-coherence
        significance test, whose finite-sample corrections assume independent
        observations.

        Returns
        -------
        bool
            ``True`` when ``segment_overlap <= 0.5``.
        """
        return self.segment_overlap <= 0.5

    def fft(self) -> NDArray[np.complexfloating]:
        """Compute the Hann-windowed Fourier coefficients of every segment.

        Returns
        -------
        fourier_coefficients : array
            Shape ``(1, n_trials, n_segments, n_fft_samples, n_signals)``.
            Segments occupy the taper axis, so :class:`Connectivity`'s
            trial/taper expectation averages over segments (and trials) into a
            single time bin.
        """
        coefficients = self._stft.fft()[:, :, 0, :, :]
        return xp.transpose(coefficients, (1, 0, 2, 3))[xp.newaxis, ...]

    def _provenance_metadata(self) -> dict[str, Any]:
        return {
            "transform": self._transform_name,
            "detrend_type": self.detrend_type,
            "fft_workers": self.fft_workers,
            "start_time": to_numpy(self.start_time),
            "n_fft_samples": self._stft.n_fft_samples,
            "n_segments": self.n_segments,
            "n_signals": self.n_signals,
            "n_time_samples_per_segment": self.n_time_samples_per_segment,
            "n_time_samples_per_step": self.n_time_samples_per_step,
            "n_trials": self.n_trials,
            "sampling_frequency": self.sampling_frequency,
            "nyquist_frequency": self._stft.nyquist_frequency,
            "segment_duration": self.segment_duration,
            "segment_overlap": self.segment_overlap,
            **self._stft._spectral_bandwidth_metadata(),
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
        Real-valued input signals (a complex series raises ``TypeError``).
        Use :func:`prepare_time_series` for 1-D/2-D input. An
        ``xarray.DataArray`` is read by position; one whose dimension names
        put time, trials, or signals on another axis raises ``ValueError``.
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
        measures (coherence, PLV) are not degenerate. Coefficients closer than
        a few wavelet standard deviations ``sigma_t = n_cycles / (2 pi f)`` are
        strongly autocorrelated, so the window must span several sigma_t to
        hold more than one effectively independent sample: use at least
        ``4 * n_cycles / (2 * pi * f_min)`` seconds (0.45 s for 7 cycles at
        10 Hz), and expect the effective sample count to stay well below the
        number of coefficients in the window. For a single trial a shorter
        window emits a ``UserWarning``, because it biases normalized measures of
        independent signals toward 1; with multiple trials the trial/taper
        expectation also averages independent realizations, so no warning is
        emitted.
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

    _transform_name = "morlet"
    # Coefficients are already on the one-sided PSD scale (see fft), so
    # Connectivity must not double them again.
    is_one_sided = True

    @reject_sampling_frequency_aliases
    def __init__(
        self,
        time_series: NDArray[np.floating],
        sampling_frequency: float,
        *,
        frequencies: NDArray[np.floating],
        n_cycles: float | NDArray[np.floating] = 7.0,
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
        time_series = _validated_time_series(time_series, type(self).__name__)
        data = xp.asarray(time_series)
        if data.ndim != 3:
            msg = "time_series must have shape (n_time_samples, n_trials, n_signals)."
            raise ValueError(msg)
        sampling_frequency = _validate_sampling_frequency(sampling_frequency)
        # Parameter arrays are validated on the host; ``to_numpy`` brings a CuPy
        # array over explicitly (CuPy rejects implicit ``np.asarray``) and is a
        # no-op for NumPy input.
        frequency_values = np.asarray(to_numpy(frequencies), dtype=float)
        if (
            frequency_values.ndim != 1
            or frequency_values.size == 0
            or not np.all(np.isfinite(frequency_values))
            or not np.all(frequency_values > 0)
            or not np.all(np.diff(frequency_values) > 0)
            or np.any(frequency_values >= sampling_frequency / 2)
        ):
            msg = (
                "frequencies must be a non-empty, finite, strictly increasing "
                "positive array below Nyquist."
            )
            raise ValueError(msg)
        cycle_values = np.asarray(to_numpy(n_cycles), dtype=float)
        if cycle_values.ndim == 0:
            cycle_values = np.full(frequency_values.shape, float(cycle_values))
        if cycle_values.shape != frequency_values.shape or not np.all(
            np.isfinite(cycle_values) & (cycle_values > 0)
        ):
            msg = "n_cycles must be a positive scalar or have one value per frequency."
            raise ValueError(msg)
        if not is_positive_integer(decimation):
            msg = "decimation must be a positive integer."
            raise ValueError(msg)
        if smoothing_time is not None and (
            not np.isfinite(smoothing_time) or smoothing_time <= 0
        ):
            msg = "smoothing_time must be finite and positive."
            raise ValueError(msg)
        if smoothing_step is not None and (
            smoothing_time is None or not np.isfinite(smoothing_step) or smoothing_step <= 0
        ):
            msg = "smoothing_step requires smoothing_time and must be finite and positive."
            raise ValueError(msg)
        if not is_positive_integer(smoothing_frequency) or smoothing_frequency % 2 == 0:
            msg = "smoothing_frequency must be a positive odd integer."
            raise ValueError(msg)
        if smoothing_kernel not in {"boxcar", "hann"}:
            msg = "smoothing_kernel must be 'boxcar' or 'hann'."
            raise ValueError(msg)
        if padding_mode not in {"constant", "reflect", "edge"}:
            msg = "padding_mode must be 'constant', 'reflect', or 'edge'."
            raise ValueError(msg)
        if padding_mode == "reflect" and data.shape[0] < 2:
            msg = "padding_mode='reflect' requires at least 2 time samples."
            raise ValueError(msg)
        if edge_mode not in {"keep", "nan", "trim"}:
            msg = "edge_mode must be 'keep', 'nan', or 'trim'."
            raise ValueError(msg)

        self._time_series = _immutable_array_snapshot(data)
        self.sampling_frequency = sampling_frequency
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
        self.start_time = _finite_scalar_start_time(start_time)

        half_widths = np.maximum(
            1,
            np.ceil(
                5 * cycle_values / (2 * np.pi * frequency_values) * self.sampling_frequency
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
                msg = (
                    "edge_mode='trim' leaves no samples valid for every requested "
                    "frequency; use fewer cycles, a longer record, or edge_mode='nan'."
                )
                raise ValueError(msg)
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
                msg = (
                    "smoothing_time/smoothing_step resolve to less than one decimated sample."
                )
                raise ValueError(msg)
            if self._smoothing_samples > n_decimated:
                msg = "smoothing_time is longer than the decimated wavelet record."
                raise ValueError(msg)
            # Coefficients within a few envelope standard deviations
            # sigma_t = n_cycles / (2 pi f) of each other are strongly
            # autocorrelated (coefficient autocorrelation ~ exp(-dt^2 / (4 sigma_t^2))),
            # so a window shorter than ~4 sigma_t holds fewer than ~2 effectively
            # independent samples. For a single trial the window is the only
            # source of observations, so normalized measures are forced toward
            # 1; independent trials already give the trial/taper expectation at
            # least two independent observations, so only a single trial warns.
            # The widest wavelet (largest sigma_t) sets the requirement.
            sigma_t = cycle_values / (2 * np.pi * frequency_values)
            widest = int(np.argmax(sigma_t))
            minimum_smoothing_time = 4 * sigma_t[widest]
            if data.shape[1] == 1 and smoothing_time < minimum_smoothing_time:
                warnings.warn(
                    f"smoothing_time={smoothing_time:g} s is shorter than 4 wavelet "
                    f"standard deviations at {frequency_values[widest]:g} Hz "
                    f"(sigma_t = n_cycles / (2 pi f) = {sigma_t[widest]:.3g} s, so "
                    f"4 sigma_t = {minimum_smoothing_time:.3g} s). Wavelet "
                    "coefficients closer than a few sigma_t are strongly "
                    "autocorrelated, so with a single trial this window holds fewer "
                    "than ~2 effectively independent samples and normalized "
                    "measures (coherence, PLV, PPC) are biased toward 1. Either "
                    "lengthen smoothing_time to at least 4 * n_cycles / "
                    f"(2 * pi * f_min) = {minimum_smoothing_time:.3g} s, or supply "
                    "multiple trials so the expectation also averages independent "
                    "realizations.",
                    UserWarning,
                    stacklevel=stacklevel_outside_package(),
                )

    @property
    def frequencies(self) -> NDArray[np.floating]:
        """Wavelet center frequencies in Hz.

        Returns
        -------
        frequencies : array, shape (n_frequencies,)
        """
        return _readonly_array_copy(self._frequencies)

    @property
    def n_cycles(self) -> NDArray[np.floating]:
        """Number of wavelet cycles at each frequency, shape (n_frequencies,)."""
        return _readonly_array_copy(self._n_cycles)

    @property
    def n_trials(self) -> int:
        """Number of trials in the time series."""
        return int(self._time_series.shape[1])

    @property
    def n_signals(self) -> int:
        """Number of signals in the time series."""
        return int(self._time_series.shape[2])

    @property
    def time(self) -> NDArray[np.floating]:
        """Center time of each output time bin, in seconds.

        Returns
        -------
        time : array, shape (n_time,)
            Decimated sample times, averaged over each smoothing window when
            time smoothing is enabled.
        """
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
        return _readonly_array_copy(self._edge_half_width_samples / self.sampling_frequency)

    @property
    def observations_are_independent(self) -> bool:
        """Whether the observation axis holds independent samples.

        Without smoothing the observation axis holds one coefficient per trial,
        and trials are independent realizations. With ``smoothing_time`` or
        ``smoothing_frequency`` the axis also holds neighboring coefficients of
        the same trial, which are strongly autocorrelated: the wavelet's
        Gaussian envelope (temporal standard deviation ``n_cycles / (2 pi f)``)
        correlates coefficients closer than a few standard deviations, and
        adjacent frequency bins overlap in bandwidth. ``n_observations`` then
        overstates the effective sample size, and normalized measures of
        independent signals are biased toward 1. :class:`Connectivity` reads
        this flag before the jackknife, the debiased measures (pairwise phase
        consistency, debiased squared PLI/WPLI) and the zero-coherence
        significance test, whose finite-sample corrections assume independent
        observations. Correlation between successive time bins, which matters
        only for expectations that also average over time, is reported by
        :attr:`time_bins_are_independent`.

        Returns
        -------
        bool
            ``True`` only when the smoothing neighborhood is a single sample
            (no smoothing, or a one-sample window that merely decimates).
        """
        return self._smoothing_samples * self.smoothing_frequency == 1

    @property
    def time_bins_are_independent(self) -> bool:
        """Whether the output time bins may be counted as independent observations.

        An expectation that averages over time (``"time_trials"``, ...) counts
        every time bin in ``n_observations``. Wavelet coefficients closer than
        a few envelope standard deviations ``sigma_t = n_cycles / (2 pi f)``
        are strongly autocorrelated (coefficient autocorrelation
        ~ ``exp(-dt^2 / (4 sigma_t^2))``), so without decimation adjacent
        samples are nearly duplicates and that count overstates the effective
        sample size. Bins count as independent when the closest coefficients
        of successive bins are at least ``4 * sigma_t`` apart at the widest
        wavelet, the spacing the ``smoothing_time`` guidance also uses.
        :class:`Connectivity` warns when an expectation averages over
        correlated time bins.

        Returns
        -------
        bool
            ``True`` when decimation and the smoothing step leave at least
            ``4 * n_cycles / (2 pi f_min)`` seconds between successive bins.
        """
        sigma_t = to_numpy(self._n_cycles) / (2 * np.pi * to_numpy(self._frequencies))
        # The last coefficient of one smoothing window and the first of the next
        # are (step - window + 1) decimated samples apart; without smoothing
        # both are 1, leaving one decimated sample between bins.
        gap_samples = self._smoothing_step_samples - self._smoothing_samples + 1
        gap = gap_samples * self.decimation / self.sampling_frequency
        return bool(gap >= 4 * np.max(sigma_t))

    def _smooth_frequency_axis(self, array: BackendArray, frequency_axis: int) -> BackendArray:
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
        frequency_mode: Literal["reflect", "edge"] = (
            "reflect" if array.shape[frequency_axis] > 1 else "edge"
        )
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
        time_weights = self._kernel_values(self._smoothing_samples, self.smoothing_kernel)
        frequency_weights = self._kernel_values(
            self.smoothing_frequency, self.smoothing_kernel
        )
        return time_weights[:, xp.newaxis] * frequency_weights[xp.newaxis, :]

    @property
    def observation_weights(self) -> NDArray[np.floating] | None:
        """Weights consumed by :class:`Connectivity` for local expectations.

        Returns
        -------
        observation_weights : array or None
            Shape ``(n_time, n_trials, n_smoothing_time * n_smoothing_frequency,
            n_frequencies, 1)``. ``None`` when every observation has equal weight
            (a uniform smoothing kernel without ``edge_mode="nan"`` masking), so
            :class:`Connectivity` uses its plain, unweighted expectation.
        """
        kernel_values = self._smoothing_kernel_values()
        if self.edge_mode != "nan" and bool(xp.all(kernel_values == kernel_values[0, 0])):
            return None
        kernel = kernel_values.reshape(1, 1, -1, 1, 1)
        shape = (
            len(self.time),
            self.n_trials,
            self._smoothing_samples * self.smoothing_frequency,
            len(self._frequencies),
            1,
        )
        weights = xp.broadcast_to(kernel, shape).copy()
        if self.edge_mode == "nan":
            weights *= self._windowed_validity()[:, xp.newaxis, xp.newaxis, :, xp.newaxis]
        return _readonly_array_copy(weights)

    def fft(self) -> NDArray[np.complexfloating]:
        """Compute one-sided Morlet coefficients grouped by smoothing neighborhood.

        Returns
        -------
        fourier_coefficients : array
            Shape ``(n_time, n_trials, n_smoothing_time * n_smoothing_frequency,
            n_frequencies, n_signals)``. The neighboring time samples and
            frequencies of each smoothing window occupy the taper axis, so
            :class:`Connectivity`'s trial/taper expectation (weighted by
            :attr:`observation_weights`) forms the local smoothed estimate.
            ``|coefficient|**2`` is a one-sided power spectral density.
        """
        # Pad once to the widest wavelet and transform the data once; each
        # frequency then costs one kernel FFT, a multiply, and an inverse FFT.
        # Padding wider than a given wavelet needs does not change its 'valid'
        # output: the extra samples never enter that wavelet's support.
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

        # Fill one preallocated array rather than stacking per-frequency results,
        # which would hold the per-frequency list and the stacked array at once.
        n_trials, n_signals = data_spectrum.shape[1:]
        transformed = xp.empty(
            (len(self._sample_indices), n_trials, len(self._frequencies), n_signals),
            dtype=data_spectrum.dtype,
        )
        for frequency_index, (frequency, cycles, half_width) in enumerate(
            zip(
                self._frequencies,
                self._n_cycles,
                self._edge_half_width_samples,
                strict=True,
            )
        ):
            sigma = cycles / (2 * xp.pi * frequency)
            half_width = int(half_width)
            wavelet_time = xp.arange(-half_width, half_width + 1) / self.sampling_frequency
            oscillation = xp.exp(2j * xp.pi * frequency * wavelet_time)
            gaussian = xp.exp(-(wavelet_time**2) / (2 * sigma**2))
            if self.zero_mean:
                oscillation = oscillation - xp.exp(-0.5 * (2 * xp.pi * frequency * sigma) ** 2)
            wavelet = oscillation * gaussian
            wavelet = wavelet / xp.sqrt(xp.sum(xp.abs(wavelet) ** 2))
            kernel = xp.conjugate(wavelet[::-1])
            kernel_spectrum = fft(kernel, n=n_fft)[:, xp.newaxis, xp.newaxis]
            convolved = ifft(data_spectrum * kernel_spectrum, axis=0)
            # The 'valid' output centred on original sample i sits at
            # full-convolution index i + max_half_width + half_width.
            start = max_half_width + half_width
            transformed[:, :, frequency_index] = (
                convolved[start + self._sample_indices] * scale
            )

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
        # The smoothing window and step as used: whole decimated samples.
        smoothed = self.smoothing_time is not None
        rate = self.sampling_frequency
        return {
            "transform": self._transform_name,
            "decimation": self.decimation,
            "frequencies": to_numpy(self._frequencies),
            "n_cycles": to_numpy(self._n_cycles),
            "n_signals": self.n_signals,
            "n_trials": self.n_trials,
            "sampling_frequency": self.sampling_frequency,
            "start_time": self.start_time,
            "nyquist_frequency": self.sampling_frequency / 2,
            "edge_mode": self.edge_mode,
            "padding_mode": self.padding_mode,
            "smoothing_frequency": self.smoothing_frequency,
            "smoothing_kernel": self.smoothing_kernel,
            "smoothing_step": (
                self._smoothing_step_samples * self.decimation / rate if smoothed else None
            ),
            "smoothing_time": (
                self._smoothing_samples * self.decimation / rate if smoothed else None
            ),
            "zero_mean": self.zero_mean,
        }


if TYPE_CHECKING:
    # mypy verifies that every built-in transform satisfies the public protocol.
    _BUILTIN_TRANSFORMS: tuple[type[SpectralTransform], ...] = (
        Multitaper,
        ShortTimeFourierTransform,
        Welch,
        MorletWavelet,
    )


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
    >>> eeg_data = np.random.default_rng(0).standard_normal((5000, 64))  # Shape: (n_time, n_channels)
    >>> eeg_3d = prepare_time_series(eeg_data, axis="signals")
    >>> eeg_3d.shape
    (5000, 1, 64)

    Multiple trials of a single electrode:

    >>> # 20 trials of one LFP channel, 2 seconds each at 1000 Hz
    >>> lfp_trials = np.random.default_rng(0).standard_normal((2000, 20))  # Shape: (n_time, n_trials)
    >>> lfp_3d = prepare_time_series(lfp_trials, axis="trials")
    >>> lfp_3d.shape
    (2000, 20, 1)

    Single time series (e.g., spike times converted to continuous):

    >>> # One neuron's firing rate over time
    >>> firing_rate = np.random.default_rng(0).standard_normal(1000)
    >>> firing_rate_3d = prepare_time_series(firing_rate)
    >>> firing_rate_3d.shape
    (1000, 1, 1)

    Already properly formatted (pass-through):

    >>> # Epoched data from MNE or similar: 10 trials, 5 channels, 100 timepoints
    >>> epoched_data = np.random.default_rng(0).standard_normal((100, 10, 5))
    >>> result = prepare_time_series(epoched_data)
    >>> result.shape
    (100, 10, 5)

    Notes
    -----
    **Common mistake:** Using 2D data without specifying the axis parameter.
    A 2D array (100, 5) could mean either:
    - 100 time points x 5 signals (1 trial) → use axis="signals"
    - 100 time points x 5 trials (1 signal) → use axis="trials"

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

    if ndim == 2:
        # Ambiguous case - require explicit axis specification
        if axis is None:
            msg = (
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
            raise ValueError(msg)

        if axis == "signals":
            # (n_time, n_signals) → (n_time, 1, n_signals)
            return time_series_array[:, xp.newaxis, :]
        if axis == "trials":
            # (n_time, n_trials) → (n_time, n_trials, 1)
            return time_series_array[:, :, xp.newaxis]
        msg = f"axis must be either 'signals' or 'trials', got: {axis!r}"
        raise ValueError(msg)

    if ndim == 3:
        # Already in correct format
        return time_series_array

    msg = (
        f"Expected 1D, 2D, or 3D array, got {ndim}D array with shape {time_series_array.shape}"
    )
    raise ValueError(msg)


def _sliding_window(
    data: NDArray[_ScalarT],
    window_size: int,
    step_size: int = 1,
    axis: int = -1,
    is_copy: bool = True,
) -> NDArray[_ScalarT]:
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
        msg = f"axis {axis} is out of bounds for an array of rank {data.ndim}."
        raise ValueError(msg)
    # A non-positive step would pass a negative slice step below (reversing the
    # windows) or an empty step, neither of which is a forward slide.
    if step_size < 1:
        msg = f"step_size must be a positive integer, got {step_size}."
        raise ValueError(msg)
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
    tapers : array_like, shape (..., n_time_samples_per_window, n_tapers)
        One set of tapers for every window, or a set per window broadcasting
        against the leading axes of ``time_series``.
    time_series : array_like, shape (..., n_time_samples_per_window)
        The windows, such as ``(n_windows, n_trials, n_signals,
        n_time_samples_per_window)``.
    n_fft_samples : int
    sampling_frequency : int
    workers : int, optional
        Number of parallel worker threads for SciPy's CPU FFT (see
        ``scipy.fft.fft``; ``-1`` uses all cores). ``None`` keeps SciPy's default
        (single-threaded). Ignored on the GPU backend, whose FFT has no such
        parameter and is already parallel.

    Returns
    -------
    fourier_coefficients : array_like, shape (..., n_fft_samples, n_tapers)

    """
    projected_time_series: NDArray[np.floating] = time_series[..., xp.newaxis] * tapers
    # Only SciPy's CPU FFT accepts ``workers``; cupyx's FFT does not, so pass it
    # only when a worker count is requested and we are on the CPU backend.
    fft_kwargs: dict[str, Any] = {}
    if workers is not None and not ON_GPU:
        fft_kwargs["workers"] = workers
    if ON_GPU and not xp.iscomplexobj(projected_time_series):
        # cuFFT's complex FFT of real input is conjugate-symmetric only to
        # rounding, which fails the Wilson factorization's exact symmetry check.
        # rfft plus the exact mirror X[n - k] = conj(X[k]) is exactly symmetric.
        # (SciPy's complex FFT already is, and is faster on the CPU.)
        half_spectrum: NDArray[np.complexfloating] = rfft(
            projected_time_series, n=n_fft_samples, axis=axis
        )
        n_non_negative = half_spectrum.shape[axis]
        shape = list(half_spectrum.shape)
        shape[axis] = n_fft_samples
        coefficients: NDArray[np.complexfloating] = xp.empty(shape, dtype=half_spectrum.dtype)
        coefficients_last = xp.moveaxis(coefficients, axis, -1)
        xp.divide(
            xp.moveaxis(half_spectrum, axis, -1),
            sampling_frequency,
            out=coefficients_last[..., :n_non_negative],
        )
        # The zero and (even n) Nyquist bins are their own mirrors, but cuFFT
        # can leave ~1e-15 in their imaginary part. Zero it through the ``imag``
        # view: CuPy 14 drops ``x[..., k] = x[..., k].real`` for some strided
        # layouts (the source aliases the destination).
        self_mirrored_bins = [0] if n_fft_samples % 2 else [0, n_fft_samples // 2]
        for bin_index in self_mirrored_bins:
            coefficients_last[..., bin_index].imag[...] = 0
        xp.conjugate(
            coefficients_last[..., n_fft_samples - n_non_negative : 0 : -1],
            out=coefficients_last[..., n_non_negative:],
        )
        return coefficients
    coefficients = fft(projected_time_series, n=n_fft_samples, axis=axis, **fft_kwargs)
    return coefficients / sampling_frequency


def _mean_square_over_kept(
    weights: NDArray[np.floating], n_kept: NDArray[np.integer], axis: int
) -> NDArray[np.floating]:
    """The mean of ``weights ** 2`` over the ``n_kept`` (broadcasting, with
    ``axis`` kept) nonzero entries on ``axis``, the others being zero: the
    plain mean when every entry is kept, as the scale is then exactly 1."""
    mean_square: NDArray[np.floating] = xp.mean(weights**2, axis=axis, keepdims=True) * (
        weights.shape[axis] / n_kept
    )
    return mean_square


def _apply_adaptive_taper_weights(
    coefficients: NDArray[np.complexfloating],
    eigenvalues: NDArray[np.floating],
    noise_power_spectral_density: NDArray[np.floating],
    *,
    max_iterations: int,
    tolerance: float,
) -> NDArray[np.complexfloating]:
    """Apply Thomson adaptive DPSS weights to Fourier coefficients.

    ``eigenvalues`` are the tapers' concentrations, ``(n_tapers,)``, or each
    trial window's, ``(n_time_windows, n_trials, n_tapers)``, where a taper
    of concentration 0 gets weight 0 and the weights are RMS-normalized over
    the others.

    The iterative spectrum estimate follows Thomson's frequency- and
    signal-specific weighting. The weights are RMS-normalized across the taper
    axis and multiplied into the coefficients, so :class:`Connectivity`'s
    ordinary taper mean of ``|coefficient|**2`` is exactly Thomson's adaptive
    auto-spectrum ``sum_k d_k^2 |Y_k|^2 / sum_k d_k^2``. Cross-spectra do not
    follow Thomson's joint normalization, however: the taper mean gives
    ``sum_k d_xk d_yk Y_xk conj(Y_yk) / sqrt(sum_k d_xk^2 sum_k d_yk^2)``
    instead of ``/ sum_k d_xk d_yk``, which by Cauchy-Schwarz shrinks coherence
    by the cosine similarity of the two signals' weight vectors. A joint
    normalization would need pairwise weights inside the expectation.

    ``noise_power_spectral_density`` must already be on the same
    power-spectral-density scale as ``|coefficients| ** 2`` (see the caller in
    :meth:`Multitaper.fft`); a raw time-domain variance would over-weight the
    noise term by a factor of the sampling frequency.
    """
    taper_power = xp.abs(coefficients) ** 2
    n_initial = min(2, coefficients.shape[2])
    spectrum = xp.mean(taper_power[:, :, :n_initial, :, :], axis=2)

    concentration = eigenvalues[..., xp.newaxis, xp.newaxis]
    root_concentration = xp.sqrt(concentration)
    noise = noise_power_spectral_density[:, :, xp.newaxis, xp.newaxis, :]
    eps = xp.finfo(taper_power.dtype).eps
    # The denominator and spectrum are power-spectral densities, so an absolute
    # epsilon floor would discard valid low-power estimates once the signal is
    # scaled down (its PSD can fall below machine epsilon). Floor relative to a
    # PSD scale instead, keeping the weighting invariant to input scale; a
    # genuinely dead channel has a zero scale and is still masked out. Average
    # only within each window/trial (over tapers and frequency) so a NaN in one
    # window cannot poison the threshold, and thus the weighting, of the others.
    psd_scale = eps * xp.mean(taper_power, axis=(2, 3))  # (window, trial, signal)
    denominator_floor = psd_scale[:, :, xp.newaxis, xp.newaxis, :]
    spectrum_floor = psd_scale[:, :, xp.newaxis, :]

    weights = xp.ones_like(taper_power, dtype=taper_power.dtype)
    for _ in range(max_iterations):
        expanded_spectrum = spectrum[:, :, xp.newaxis, :, :]
        denominator = concentration * expanded_spectrum + (1.0 - concentration) * noise
        weights = _divide_where(
            root_concentration * expanded_spectrum,
            denominator,
            xp.abs(denominator) > denominator_floor,
            0.0,
        )
        weight_power = weights**2
        weight_sum = xp.sum(weight_power, axis=2)
        updated = _divide_where(
            xp.sum(weight_power * taper_power, axis=2),
            weight_sum,
            weight_sum > eps,
            0.0,
        )
        scale = xp.maximum(xp.abs(spectrum), spectrum_floor)
        # A window/trial containing NaN has a NaN spectrum or PSD floor, so its
        # scale is NaN and it can never meet the tolerance; don't let it block
        # convergence of the finite bins.
        settled = (xp.abs(updated - spectrum) <= tolerance * scale) | ~xp.isfinite(scale)
        if bool(xp.all(settled)):
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
            stacklevel=stacklevel_outside_package(),
        )

    if eigenvalues.ndim == 1:
        mean_square = xp.mean(weights**2, axis=2, keepdims=True)
    else:
        n_used = xp.count_nonzero(eigenvalues, axis=-1)[
            ..., xp.newaxis, xp.newaxis, xp.newaxis
        ]
        mean_square = _mean_square_over_kept(weights, n_used, axis=2)
    rms = xp.sqrt(mean_square)
    normalized_weights = _divide_where(weights, rms, rms > eps, 1.0)
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
        msg = f"n_tapers must be an integer, got {n_tapers}."
        raise ValueError(msg)
    n_tapers = int(n_tapers)
    if n_time_samples_per_window < 2:
        msg = (
            f"n_time_samples_per_window must be >= 2 for a multitaper "
            f"decomposition, got {n_time_samples_per_window}. A single-sample "
            f"window carries no spectral information."
        )
        raise _recording_too_short(msg)
    if not 0 < time_halfbandwidth_product < n_time_samples_per_window / 2:
        # NW <= 0 is a bad NW; NW >= n / 2, a window too short for it.
        too_short = time_halfbandwidth_product > 0
        msg = (
            f"time_halfbandwidth_product (NW={time_halfbandwidth_product}) must "
            f"satisfy 0 < NW < n_time_samples_per_window / 2 "
            f"(= {n_time_samples_per_window / 2}). Otherwise the concentration "
            f"bandwidth reaches or exceeds Nyquist and the result is not a valid "
            f"set of DPSS tapers (their concentration ratios can exceed 1). Use a "
            f"longer window or a smaller time_halfbandwidth_product."
        )
        raise _recording_too_short(msg) if too_short else ValueError(msg)
    if not 1 <= n_tapers <= n_time_samples_per_window:
        msg = (
            f"n_tapers must satisfy 1 <= n_tapers <= n_time_samples_per_window "
            f"(= {n_time_samples_per_window}), got {n_tapers}."
        )
        raise ValueError(msg)
    if n_time_samples_per_window == 2 and n_tapers == 2:
        # scipy.signal.windows.dpss cannot disambiguate the sign of the second
        # (antisymmetric) taper for a two-sample window: its heuristic keeps the
        # first sample with |value|^2 above max(1e-7, 1 / n_time_samples) = 0.5,
        # but the length-two antisymmetric taper is [1, -1] / sqrt(2), whose
        # samples are exactly 0.5, so none clear the threshold and SciPy raises a
        # bare IndexError (both 1.10.x and current releases). A two-sample window
        # cannot support a second taper usefully anyway, so reject the pair with
        # an actionable message rather than surface SciPy's internal error.
        msg = (
            "A two-sample window (n_time_samples_per_window=2) supports only a "
            "single DPSS taper; request n_tapers=1 or use a longer window."
        )
        raise ValueError(msg)

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

    return _get_low_bias_tapers(tapers, eigenvalues) if is_low_bias else (tapers, eigenvalues)


def _get_low_bias_tapers(
    tapers: NDArray[np.floating], eigenvalues: NDArray[np.floating]
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    is_low_bias = eigenvalues > MIN_EIGENVALUE_THRESHOLD
    if not xp.any(is_low_bias):
        warnings.warn(
            f"No taper has a spectral concentration above {MIN_EIGENVALUE_THRESHOLD} "
            "(time_halfbandwidth_product is too small for low-bias tapers); keeping "
            "only the lowest-bias taper, whose spectral leakage is high.",
            UserWarning,
            stacklevel=stacklevel_outside_package(),
        )
        is_low_bias = xp.array([xp.argmax(eigenvalues)])
    return tapers[is_low_bias, :], eigenvalues[is_low_bias]


def _restricted_toeplitz_operator(
    first_row: NDArray[np.floating], indices: NDArray[np.intp]
) -> LinearOperator:
    """The symmetric Toeplitz matrix ``first_row[|i - j|]`` restricted to the
    rows and columns ``indices``, as an operator.

    Each product embeds the vector at ``indices`` in a vector of the full
    length, multiplies by the matrix as a circular convolution by FFT, and
    reads ``indices`` back: ``O(n log n)`` time and ``O(n)`` memory for a
    matrix of order ``n = first_row.size``. On the host whatever the backend,
    as the tapers are computed there, like the DPSS.
    """
    n = first_row.size
    n_fft = int(default_fft_length(2 * n - 1))
    column = np.zeros(n_fft)
    column[:n] = first_row
    column[n_fft - n + 1 :] = first_row[:0:-1]
    transfer = np.fft.rfft(column)

    def multiply(vector: NDArray[np.floating]) -> NDArray[np.floating]:
        embedded = np.zeros(n_fft)
        embedded[indices] = np.ravel(vector)
        product: NDArray[np.floating] = np.fft.irfft(np.fft.rfft(embedded) * transfer, n_fft)
        return product[indices]

    return LinearOperator((indices.size, indices.size), matvec=multiply, dtype=np.float64)


def _missing_data_slepian_tapers(
    valid: NDArray[np.bool_],
    time_halfbandwidth_product: float,
    n_tapers: int,
    orientation: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """The Slepian sequences of a window with missing samples (Bronez 1988).

    Of the sequences that are zero at the missing samples, the ones most
    concentrated in the band ``[-W, W]``, ``W = time_halfbandwidth_product /
    n_time_samples_per_window`` cycles per sample, as the DPSS are for a
    window without gaps: the leading eigenvectors of the sinc kernel
    ``sin(2 pi W (t - s)) / (pi (t - s))`` (``2 W`` at ``t = s``) restricted
    to the valid sample times. The eigenvalues are their concentrations, the
    fraction of each one's energy inside the band. Without missing samples
    these are the DPSS. Computed on the host.

    Parameters
    ----------
    valid : NDArray[bool], shape (n_time_samples_per_window,)
        ``True`` at the window's valid samples.
    time_halfbandwidth_product : float
        NW, the half bandwidth times the window length.
    n_tapers : int
        Number of sequences, in decreasing order of concentration.
    orientation : NDArray[float], shape (n_time_samples_per_window, n_tapers)
        The window's DPSS tapers, which fix each sequence's sign and the
        basis of sequences of equal concentration (:func:`_aligned_with_dpss`).

    Returns
    -------
    tapers : NDArray[float], shape (n_time_samples_per_window, n_tapers)
        Unit energy, zero at the missing samples; columns past the number of
        valid samples are zero.
    concentrations : NDArray[float], shape (n_tapers,)
        Decreasing; zero for those zero columns. They are the eigenvalues:
        within a group of nearly equal ones, whose tapers are rotated toward
        the DPSS, a rotated taper's own concentration can differ from its
        entry by up to the group's spread (7e-7 at NW = 8).
    """
    n_time_samples = valid.shape[0]
    (valid_times,) = np.nonzero(valid)
    half_bandwidth = time_halfbandwidth_product / n_time_samples
    # The kernel depends only on the lag between two samples.
    sinc = 2 * half_bandwidth * np.sinc(2 * half_bandwidth * np.arange(n_time_samples))
    n_computed = min(n_tapers, valid_times.size)
    eigenvalues, eigenvectors = _leading_eigenpairs(sinc, valid_times, n_computed)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    eigenvectors = _aligned_with_dpss(
        eigenvectors, eigenvalues, orientation[valid_times, :n_computed]
    )
    tapers = np.zeros((n_time_samples, n_tapers))
    tapers[valid_times, :n_computed] = eigenvectors
    concentrations = np.zeros(n_tapers)
    concentrations[:n_computed] = eigenvalues
    return tapers, concentrations


def _leading_eigenpairs(
    first_row: NDArray[np.floating], indices: NDArray[np.intp], k: int
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """The ``k`` largest eigenpairs, in no particular order, of the Toeplitz
    matrix ``first_row[|i - j|]`` restricted to ``indices``.

    Implicitly restarted Lanczos (ARPACK) on the FFT operator of
    :func:`_restricted_toeplitz_operator`, from a fixed start vector so the
    result is reproducible. Lanczos from one vector can miss copies of a
    repeated eigenvalue (SciPy 1.11's ARPACK finds 9 of the 10 concentrations
    within 1e-5 of 1 of a 1000-sample window at NW = 8), so the largest
    eigenvalue of the matrix with the pairs found deflated, which any vector
    of its eigenspace gives, is added while it is not below the smallest
    found, and the pairs are recomputed on their span. ARPACK needs more
    Lanczos vectors than eigenpairs and fewer than the matrix order; at or
    below its default count, ``max(2 k + 1, 20)``, the matrix is solved
    densely, as it is if ARPACK does not converge.
    """
    n = indices.size
    if n > max(2 * k + 1, 20):
        try:
            return _lanczos_eigenpairs(first_row, indices, k)
        except ArpackNoConvergence:
            pass
    kernel = first_row[np.abs(indices[:, np.newaxis] - indices[np.newaxis, :])]
    return scipy_eigh(kernel, subset_by_index=[n - k, n - 1])


def _lanczos_eigenpairs(
    first_row: NDArray[np.floating], indices: NDArray[np.intp], k: int
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """The Lanczos solve and completeness check of :func:`_leading_eigenpairs`."""
    n = indices.size
    operator = _restricted_toeplitz_operator(first_row, indices)
    # Generic, so no eigenvector is orthogonal to it by symmetry.
    start = np.random.default_rng(0).standard_normal(n)
    eigenvalues, eigenvectors = scipy_eigsh(operator, k=k, which="LA", tol=0, v0=start)
    for _ in range(k):

        def deflated(
            vector: NDArray[np.floating],
            values: NDArray[np.floating] = eigenvalues,
            vectors: NDArray[np.floating] = eigenvectors,
        ) -> NDArray[np.floating]:
            vector = np.ravel(vector)
            product: NDArray[np.floating] = operator.matvec(vector) - vectors @ (
                values * (vectors.T @ vector)
            )
            return product

        (largest,), missed = scipy_eigsh(
            LinearOperator((n, n), matvec=deflated, dtype=np.float64),
            k=1,
            which="LA",
            tol=0,
            v0=start,
        )
        if largest <= eigenvalues.min():
            break
        basis, _ = np.linalg.qr(np.column_stack([eigenvectors, missed]))
        projected = basis.T @ np.column_stack([operator.matvec(column) for column in basis.T])
        ritz_values, ritz_vectors = np.linalg.eigh((projected + projected.T) / 2)
        eigenvalues, eigenvectors = ritz_values[1:], basis @ ritz_vectors[:, 1:]
    return eigenvalues, eigenvectors


# Concentrations closer than this are equal for the purpose of choosing their
# eigenvectors. The eigenvectors of eigenvalues dlambda apart are determined
# only to about eps / dlambda, so solvers agree on them to about 2e-10 above
# this gap; below it any basis of their span is as good (with a large NW the
# leading concentrations are all within 1e-10 of 1), and the concentrations
# of such a basis stay within the group's spread.
_DEGENERATE_CONCENTRATION_GAP = 1e-6


def _aligned_with_dpss(
    eigenvectors: NDArray[np.floating],
    eigenvalues: NDArray[np.floating],
    reference: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Choose the basis and signs of ``eigenvectors`` closest to the DPSS.

    Eigenvectors whose ``eigenvalues`` (decreasing) are within
    ``_DEGENERATE_CONCENTRATION_GAP`` of the next form a group whose span
    alone is defined, the individual vectors depending on the solver. Each
    group is rotated (orthogonal Procrustes) to be closest to the DPSS of the
    same orders, ``reference``, restricted to the valid samples; a lone
    eigenvector gets the sign that agrees with its DPSS. So the tapers do not
    depend on the eigensolver, and without missing samples they are the DPSS.

    Parameters
    ----------
    eigenvectors : NDArray[float], shape (n_valid, k)
    eigenvalues : NDArray[float], shape (k,)
    reference : NDArray[float], shape (n_valid, k)

    Returns
    -------
    NDArray[float], shape (n_valid, k)
    """
    aligned = eigenvectors.copy()
    starts = np.flatnonzero(
        np.concatenate([[True], -np.diff(eigenvalues) > _DEGENERATE_CONCENTRATION_GAP])
    )
    for start, stop in zip(starts, [*starts[1:], eigenvalues.size], strict=True):
        group = slice(start, stop)
        left, _, right = np.linalg.svd(eigenvectors[:, group].T @ reference[:, group])
        aligned[:, group] = eigenvectors[:, group] @ (left @ right)
    return aligned


def _one_blas_thread() -> contextlib.AbstractContextManager[Any]:
    """Limit BLAS to one thread, if threadpoolctl is installed.

    The per-pattern eigensolves are small and sequential, and multithreaded
    OpenBLAS can run them hundreds of times slower (600 ms against 3 ms per
    1000-sample pattern with the NumPy 1.24 and SciPy 1.11 wheels on 18
    threads). threadpoolctl is optional: without it this does nothing, and
    ``OPENBLAS_NUM_THREADS=1`` does the same.
    """
    try:
        from threadpoolctl import threadpool_limits  # type: ignore[import-not-found]
    except ImportError:
        return contextlib.nullcontext()
    limits: contextlib.AbstractContextManager[Any] = threadpool_limits(
        limits=1, user_api="blas"
    )
    return limits


def _missing_data_taper_sets(
    valid: NDArray[np.bool_],
    time_halfbandwidth_product: float,
    n_tapers: int,
    orientation: NDArray[np.floating],
) -> tuple[NDArray[np.intp], NDArray[np.floating], NDArray[np.floating]]:
    """The missing-data tapers each gap pattern keeps, on the host.

    A pattern keeps its tapers concentrated above ``MIN_EIGENVALUE_THRESHOLD``
    if there are at least 2 of them, and none otherwise.

    Parameters
    ----------
    valid : NDArray[bool], shape (n_patterns, n_time_samples_per_window)
        ``True`` at each pattern's valid samples.
    time_halfbandwidth_product : float
    n_tapers : int
    orientation : NDArray[float], shape (n_time_samples_per_window, n_tapers)
        The DPSS tapers, for :func:`_missing_data_slepian_tapers`.

    Returns
    -------
    n_kept : NDArray[int], shape (n_patterns,)
        The tapers each pattern keeps, 0 if it is dropped.
    tapers : NDArray[float], shape (n_kept_patterns, n_time_samples_per_window, n_tapers)
        Unit energy, zero past the ones kept, for the patterns that keep any.
    concentrations : NDArray[float], shape (n_kept_patterns, n_tapers)
        Zero past the ones kept.
    """
    n_patterns, n_time_samples = valid.shape
    half_bandwidth = time_halfbandwidth_product / n_time_samples
    n_valid = np.count_nonzero(valid, axis=1)
    # The concentrations are at most the restricted kernel's trace, 2 W n_valid,
    # in sum, so below twice the threshold two cannot pass: no eigenproblem.
    is_solved = 2 * half_bandwidth * n_valid > 2 * MIN_EIGENVALUE_THRESHOLD
    n_kept = np.zeros(n_patterns, dtype=np.intp)
    tapers, concentrations = [], []
    with _one_blas_thread():
        for pattern in np.flatnonzero(is_solved):
            taper, concentration = _missing_data_slepian_tapers(
                valid[pattern], time_halfbandwidth_product, n_tapers, orientation
            )
            kept = int(np.count_nonzero(concentration > MIN_EIGENVALUE_THRESHOLD))
            if kept < 2:
                continue
            taper[:, kept:] = 0
            concentration[kept:] = 0
            n_kept[pattern] = kept
            tapers.append(taper)
            concentrations.append(concentration)
    if not tapers:
        return n_kept, np.zeros((0, n_time_samples, n_tapers)), np.zeros((0, n_tapers))
    return n_kept, np.stack(tapers), np.stack(concentrations)


def _dropped_gap_summary(artifacts: NDArray[np.bool_]) -> str:
    """A sentence on the valid fraction and artifact spacing of the dropped
    windows' artifact patterns ``(n_patterns, n_time_samples_per_window)``."""
    valid_fraction = 1 - artifacts.mean(axis=1)
    # Separate artifacts: runs of artifact samples.
    n_runs = np.count_nonzero(
        np.diff(artifacts.astype(np.int8), prepend=0, axis=1) == 1, axis=1
    )
    most = int(n_runs.max())
    spacing = (
        f", and up to {most} separate artifacts in one window "
        f"({artifacts.shape[1] / most:.3g} samples apart on average)"
        if most > 1
        else ""
    )
    lowest, highest = valid_fraction.min(), valid_fraction.max()
    fraction = f"{lowest:.0%}" if lowest == highest else f"{lowest:.0%} to {highest:.0%}"
    return f"The dropped windows have {fraction} of their samples valid{spacing}. "


def _detrend_valid_samples(
    windows: NDArray[np.floating],
    valid: NDArray[np.bool_],
    detrend_type: str | None,
) -> NDArray[np.floating]:
    """Detrend each window over its valid samples only; zero the others.

    The least-squares constant (``"constant"``) or line (``"linear"``) is
    fitted to the valid samples, as :func:`detrend` fits it to a whole
    window, and subtracted from them; ``None`` leaves them as they are.

    Parameters
    ----------
    windows : array, shape (..., n_time_samples_per_window)
        The windows.
    valid : array of bool, broadcastable to ``windows``
        ``True`` at the samples to keep; at least two in each window.
    detrend_type : {"constant", "c", "linear", "l"} or None

    Returns
    -------
    array, shape (..., n_time_samples_per_window)
        The detrended windows, zero at the samples that are not valid.
    """
    weights = valid.astype(windows.dtype)
    if detrend_type is None:
        return windows * weights
    n_valid = xp.sum(weights, axis=-1, keepdims=True)
    residuals = windows - xp.sum(windows * weights, axis=-1, keepdims=True) / n_valid
    if detrend_type in ("linear", "l"):
        times = xp.arange(windows.shape[-1], dtype=windows.dtype)
        mean_time = xp.sum(times * weights, axis=-1, keepdims=True) / n_valid
        centered_times = (times - mean_time) * weights
        slopes = xp.sum(residuals * centered_times, axis=-1, keepdims=True) / xp.sum(
            centered_times**2, axis=-1, keepdims=True
        )
        residuals = residuals - slopes * centered_times
    return residuals * weights


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
    if type not in _DETREND_TYPES:
        msg = (
            f"Invalid trend type '{type}' is not supported.\n"
            f"The detrend function only supports linear and constant detrending.\n"
            f"Valid options are:\n"
            f"  - 'linear' or 'l': Remove linear trend (best-fit line)\n"
            f"  - 'constant' or 'c': Remove mean (DC offset)\n"
            f"Example: detrend(data, type='linear')"
        )
        raise ValueError(msg)
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

        def _to_list(a: NDArray[Any]) -> list[Any]:
            # Works for both numpy and cupy arrays.
            return list(to_numpy(a).tolist())

        out_of_range = bp_values[(bp_values < 0) | (bp_values >= N)]
        if out_of_range.size:
            msg = (
                f"Breakpoint value(s) {_to_list(out_of_range)} are outside the "
                f"valid range [0, {N}).\n"
                f"Data has {N} samples along axis {axis}; breakpoints are indices "
                f"into that axis and must satisfy 0 <= breakpoint < {N}.\n"
                f"Check your breakpoints: {_to_list(bp_values)}"
            )
            raise ValueError(msg)
    # Delegate the least-squares/mean removal to SciPy (CPU) or CuPy (GPU),
    # which implement the same computation. Their detrend signatures match.
    # ``type`` was validated above; expand the one-letter aliases.
    trend_type: Literal["linear", "constant"] = (
        "linear" if type in ("linear", "l") else "constant"
    )
    return _backend_detrend(
        data, axis=axis, type=trend_type, bp=bp, overwrite_data=overwrite_data
    )
