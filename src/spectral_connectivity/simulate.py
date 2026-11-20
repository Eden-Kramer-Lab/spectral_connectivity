"""Functions to simulate time series with known connectivity structure.

``simulate_MVAR`` generates multivariate autoregressive (MVAR) processes with
known directed coupling. ``simulate_lagged_broadband`` generates delayed noisy
copies of one broadband source, with a known lead/lag between every pair of
signals. ``simulate_shared_oscillation`` generates one sinusoid seen by several
signals with known amplitudes and phase offsets. All return ``float64`` NumPy
arrays of shape ``(n_time_samples, n_trials, n_signals)``, one trial by
default; ``simulate_lagged_broadband(..., n_trials=None)`` omits the trial axis.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from spectral_connectivity.utils import is_positive_integer


def _generator(random_state: int | np.random.Generator | None) -> np.random.Generator:
    """Return ``random_state`` if it is a Generator, else a Generator seeded by it."""
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


def _per_signal(
    values: ArrayLike, n_signals: int, name: str, *, nonnegative: bool = False
) -> NDArray[np.floating]:
    """Broadcast a scalar or per-signal parameter to shape ``(n_signals,)``."""
    values_array = np.asarray(values, dtype=float)
    if values_array.ndim > 1 or values_array.size not in (1, n_signals):
        msg = (
            f"{name} must be a scalar or have one entry per signal ({n_signals}); "
            f"got shape {values_array.shape}"
        )
        raise ValueError(msg)
    if nonnegative and np.any(values_array < 0):
        msg = f"{name} are standard deviations and must be non-negative; got {values_array}"
        raise ValueError(msg)
    return np.broadcast_to(values_array.reshape(-1), (n_signals,))


def _check_count(value: object, name: str, hint: str = "", *, hint_if_none: str = "") -> None:
    """Raise ``ValueError`` unless ``value`` is a positive integer (not a bool).

    ``hint`` follows the requirement in the message; ``hint_if_none``
    replaces it when ``value`` is None.
    """
    if not is_positive_integer(value):
        suffix = hint_if_none if value is None else hint
        msg = f"{name} must be a positive integer{suffix}; got {value!r}"
        raise ValueError(msg)


def _add_noise(
    time_series: NDArray[np.floating],
    noise_levels: NDArray[np.floating],
    rng: np.random.Generator,
) -> NDArray[np.floating]:
    """Add independent Gaussian noise of per-signal standard deviation ``noise_levels``.

    ``noise_levels`` has shape ``(n_signals,)`` and scales the last axis. The
    noise is drawn in one call of shape ``time_series.shape``.
    """
    return time_series + noise_levels * rng.standard_normal(time_series.shape)


def simulate_MVAR(
    coefficients: NDArray[np.floating],
    noise_covariance: NDArray[np.floating] | None = None,
    n_time_samples: int = 100,
    n_trials: int = 1,
    n_burnin_samples: int = 100,
    random_state: int | np.random.Generator | None = None,
) -> NDArray[np.floating]:
    """
    Simulate multivariate autoregressive (MVAR) process.

    Generates time series data following the MVAR model:
    X(t) = sum(A_k * X(t-k)) + E(t), where A_k are coefficient matrices
    and E(t) is multivariate Gaussian noise. Time ``t`` and the lags ``k``
    count samples.

    Parameters
    ----------
    coefficients : NDArray[floating], shape (n_lags, n_signals, n_signals)
        MVAR coefficient matrices for each lag. ``coefficients[k - 1]`` is
        A_k, the linear influence of the signals ``k`` samples earlier: lags
        are sample steps, so lag ``k`` is ``k / sampling_frequency`` seconds at
        whatever rate the series is analyzed with.
    noise_covariance : NDArray[floating], shape (n_signals, n_signals), optional
        Covariance matrix of the noise process. If None, uses identity matrix
        (independent unit-variance noise).
    n_time_samples : int, default=100
        Number of time samples to generate (after burn-in).
    n_trials : int, default=1
        Number of independent trials to simulate.
    n_burnin_samples : int, default=100
        Number of initial samples to discard for equilibrium.
    random_state : int, np.random.Generator, or None, optional
        Random number generator seed or instance for reproducible results.

    Returns
    -------
    time_series : NDArray[floating], shape (n_time_samples, n_trials, n_signals)
        Simulated time series data with specified MVAR dynamics.

    Examples
    --------
    >>> import numpy as np
    >>> # Simple 2-signal VAR(1) with coupling
    >>> coefficients = np.array([[[0.5, 0.3], [0.2, 0.6]]])
    >>> data = simulate_MVAR(coefficients, n_time_samples=1000, n_trials=5)
    >>> data.shape
    (1000, 5, 2)

    Notes
    -----
    The simulation uses a burn-in period to reach statistical equilibrium
    before collecting the requested samples.

    """
    n_lags, n_signals, _ = coefficients.shape
    if noise_covariance is None:
        noise_covariance = np.eye(n_signals)

    rng = _generator(random_state)

    time_series = rng.multivariate_normal(
        np.zeros((n_signals,)),
        noise_covariance,
        size=(n_time_samples + n_burnin_samples, n_trials),
    )

    for time_ind in np.arange(n_lags, n_time_samples + n_burnin_samples):
        for lag_ind in np.arange(n_lags):
            # For each trial, add A_k @ X(t - k). With X_prev of shape
            # (n_trials, n_signals), ``X_prev @ A_k.T`` computes this for all
            # trials at once and preserves the (n_trials, n_signals) shape.
            # (The previous ``matmul(...).squeeze()`` collapsed the signal axis
            # when n_signals == 1, crashing univariate multi-trial simulations.)
            time_series[time_ind] += (
                time_series[time_ind - (lag_ind + 1)] @ coefficients[lag_ind].T
            )
    return time_series[n_burnin_samples:, ...]


def simulate_lagged_broadband(
    lags_samples: ArrayLike,
    noise_levels: ArrayLike,
    n_time_samples: int,
    n_trials: int | None = 1,
    random_state: int | np.random.Generator | None = None,
) -> NDArray[np.floating]:
    """Noisy copies of one white-noise source, each delayed by whole samples.

    Signal ``k`` is ``source[t - lags_samples[k]] + noise_levels[k] * N(0, 1)``,
    so a signal with a smaller lag leads one with a larger lag by their
    difference in samples. The source is broadband on purpose: a sinusoid
    delayed by a whole number of cycles is indistinguishable from the original
    and carries no lag information for group delay or the phase slope index.

    Parameters
    ----------
    lags_samples : array_like of int, shape (n_signals,)
        Delay of each signal behind the source, in whole samples (not
        seconds). Must be non-negative integers; express a lead as a smaller
        lag on the leading signal, e.g. ``lags_samples=(0, 3)`` for signal 0
        leading signal 1 by 3 samples. Convert a delay in seconds with
        ``round(delay * sampling_frequency)``, as in the second example.
        Python's ``round`` sends an exact half to the even neighbour (10.5
        samples become 10, 11.5 become 12), and a delay of half a sample or
        less becomes 0, so the signals are simultaneous: check the simulated
        delay, ``lags_samples[k] / sampling_frequency`` seconds, or simulate
        at a higher rate.
    noise_levels : float or array_like of float, shape (n_signals,)
        Standard deviation of the independent Gaussian noise added to each
        signal. A scalar applies to every signal; 0 gives the pure delayed
        source.
    n_time_samples : int
        Number of time samples per signal.
    n_trials : int or None, default=1
        Number of independent trials. None draws the default single trial and
        returns it without its trial axis, as a 2-D array.
    random_state : int, np.random.Generator, or None, optional
        Seed or generator for reproducible results.

    Returns
    -------
    time_series : NDArray[floating], shape (n_time_samples, n_trials, n_signals) or (n_time_samples, n_signals)
        The delayed noisy copies, ``(time, trial, signal)`` like
        ``simulate_MVAR``; ``(time, signal)`` when ``n_trials`` is None.

    Raises
    ------
    ValueError
        If any lag is negative or not an integer, if ``noise_levels`` is
        neither a scalar nor one value per signal, or if ``n_time_samples``
        or ``n_trials`` (unless None) is not a positive integer.

    Notes
    -----
    The source has ``n_time_samples + max(lags_samples)`` samples and signal
    ``k`` is its slice starting at ``max(lags_samples) - lags_samples[k]``, so
    no sample wraps around.
    The source is drawn first, then all of the noise in one draw of shape
    ``time_series.shape``.

    Examples
    --------
    >>> import numpy as np
    >>> time_series = simulate_lagged_broadband(
    ...     lags_samples=(0, 3), noise_levels=0.0, n_time_samples=100, random_state=0
    ... )
    >>> time_series.shape
    (100, 1, 2)
    >>> # Signal 1 repeats signal 0 three samples later: signal 0 leads.
    >>> bool(np.array_equal(time_series[3:, :, 1], time_series[:-3, :, 0]))
    True

    For a delay given in seconds, round it to whole samples at the sampling
    rate and pass that count:

    >>> sampling_frequency = 1500.0  # Hz
    >>> delay = 0.0102  # seconds: signal 0 should lead signal 1 by 10.2 ms
    >>> lag_samples = round(delay * sampling_frequency)  # 15.3 samples, rounded
    >>> lag_samples
    15
    >>> time_series = simulate_lagged_broadband(
    ...     lags_samples=(0, lag_samples),
    ...     noise_levels=0.0,
    ...     n_time_samples=1000,
    ...     random_state=0,
    ... )
    >>> lag_samples / sampling_frequency  # the delay actually simulated, in seconds
    0.01

    """
    lags_array = np.asarray(lags_samples)
    if lags_array.size == 0:
        msg = "lags_samples must have at least one entry, one per signal"
        raise ValueError(msg)
    if (
        lags_array.ndim != 1
        or not np.issubdtype(lags_array.dtype, np.integer)
        or np.any(lags_array < 0)
    ):
        msg = (
            "lags_samples must be non-negative integers, in samples (an integer "
            "dtype, so 3 rather than 3.0); convert a delay in seconds with "
            "round(delay * sampling_frequency), and express a lead as a smaller "
            "lag on the leading signal, e.g. lags_samples=(0, 3)"
        )
        raise ValueError(msg)
    _check_count(n_time_samples, "n_time_samples")
    if n_trials is not None:
        _check_count(n_trials, "n_trials", " (or None to omit the trial axis)")
    n_signals = lags_array.size
    noise_array = _per_signal(noise_levels, n_signals, "noise_levels", nonnegative=True)
    rng = _generator(random_state)

    max_lag = int(lags_array.max())
    # n_trials=None is the single trial without its trial axis.
    source = rng.standard_normal(
        (n_time_samples + max_lag, 1 if n_trials is None else n_trials)
    )
    time_series = np.stack(
        # Python ints, so a narrow dtype such as int8 cannot overflow the bounds.
        [
            source[max_lag - lag : max_lag - lag + n_time_samples]
            for lag in lags_array.tolist()
        ],
        axis=-1,
    )  # (n_time_samples, n_trials, n_signals)
    noisy = _add_noise(time_series, noise_array, rng)
    return noisy[:, 0, :] if n_trials is None else noisy


def simulate_shared_oscillation(
    frequency: float,
    sampling_frequency: float,
    n_time_samples: int,
    n_trials: int = 1,
    *,
    amplitudes: ArrayLike,
    phase_offsets: ArrayLike = 0.0,
    noise_levels: ArrayLike = 0.0,
    random_phase_per_trial: bool = True,
    random_state: int | np.random.Generator | None = None,
) -> NDArray[np.floating]:
    """One sinusoid seen by every signal, with per-signal amplitude and phase.

    ``signal[t, r, k] = amplitudes[k] * sin(2 pi frequency t / sampling_frequency
    + phi_r + phase_offsets[k]) + noise_levels[k] * N(0, 1)``. Signal ``k`` leads
    signal ``m`` by ``phase_offsets[k] - phase_offsets[m]`` radians (modulo
    2 pi; differences beyond +-pi read as lags). Each trial
    ``r`` draws an independent uniform phase ``phi_r`` unless
    ``random_phase_per_trial=False`` (then ``phi_r = 0``). Set an amplitude to
    0 to leave a signal out of the oscillation; add two calls (same
    ``random_state`` generator) to give one group a private rhythm.

    Parameters
    ----------
    frequency : float
        Frequency of the shared sinusoid, in Hz.
    sampling_frequency : float
        Sampling rate, in Hz.
    n_time_samples : int
        Number of time samples per trial; time ``t`` runs from 0.
    n_trials : int, default=1
        Number of trials.
    amplitudes : array_like of float, shape (n_signals,)
        Amplitude of the sinusoid in each signal; its length sets
        ``n_signals``. Keyword-only and required.
    phase_offsets : float or array_like of float, shape (n_signals,), default=0.0
        Phase added to the sinusoid in each signal, in radians.
    noise_levels : float or array_like of float, shape (n_signals,), default=0.0
        Standard deviation of the independent Gaussian noise added to each
        signal.
    random_phase_per_trial : bool, default=True
        If True, each trial draws a phase uniformly from ``[0, 2 pi)``, shared
        by all signals; if False, every trial starts at phase 0.
    random_state : int, np.random.Generator, or None, optional
        Seed or generator for reproducible results.

    Returns
    -------
    time_series : NDArray[floating], shape (n_time_samples, n_trials, n_signals)
        The oscillation plus noise.

    Raises
    ------
    ValueError
        If ``amplitudes`` is not 1-D, if ``phase_offsets`` or
        ``noise_levels`` is neither a scalar nor one value per signal, or if
        ``n_time_samples`` or ``n_trials`` is not a positive integer (there is
        no 2-D form: ``n_trials=None`` raises).

    Notes
    -----
    The trial phases are drawn before the noise, so two calls with the same
    integer ``random_state`` share their trial phases; the one with
    ``noise_levels=0`` is the noise-free version of the other.

    Examples
    --------
    >>> import numpy as np
    >>> time_series = simulate_shared_oscillation(
    ...     frequency=10,
    ...     sampling_frequency=1000,
    ...     n_time_samples=500,
    ...     n_trials=20,
    ...     amplitudes=[1.0, 2.0, 0.0],
    ...     phase_offsets=[0.0, np.pi / 2, 0.0],
    ...     noise_levels=0.5,
    ...     random_state=0,
    ... )
    >>> time_series.shape
    (500, 20, 3)
    >>> # 10 Hz is FFT bin 5 of 500 samples at 1000 Hz. Signal 1 leads signal 0
    >>> # by pi / 2, so the phase of X_1 conj(X_0) there is about +pi / 2.
    >>> fourier = np.fft.rfft(time_series, axis=0)[5]  # (n_trials, n_signals)
    >>> phase = np.angle(np.mean(fourier[:, 1] * np.conj(fourier[:, 0])))
    >>> bool(abs(phase - np.pi / 2) < 0.1)
    True

    """
    amplitudes_array = np.asarray(amplitudes, dtype=float)
    if amplitudes_array.ndim != 1 or amplitudes_array.size == 0:
        msg = (
            "amplitudes must be 1-D with at least one entry, one per signal; "
            f"got shape {amplitudes_array.shape}"
        )
        raise ValueError(msg)
    _check_count(n_time_samples, "n_time_samples")
    _check_count(
        n_trials,
        "n_trials",
        hint_if_none="; simulate_shared_oscillation has no 2-D form, so pass "
        "n_trials=1 for a single trial",
    )
    n_signals = amplitudes_array.size
    phase_offsets_array = _per_signal(phase_offsets, n_signals, "phase_offsets")
    noise_array = _per_signal(noise_levels, n_signals, "noise_levels", nonnegative=True)
    rng = _generator(random_state)

    trial_phases = (
        rng.uniform(0, 2 * np.pi, size=n_trials)
        if random_phase_per_trial
        else np.zeros(n_trials)
    )
    time = np.arange(n_time_samples) / sampling_frequency
    phase = (
        2 * np.pi * frequency * time[:, np.newaxis, np.newaxis]
        + trial_phases[np.newaxis, :, np.newaxis]
        + phase_offsets_array
    )  # (n_time_samples, n_trials, n_signals)
    time_series: NDArray[np.floating] = amplitudes_array * np.sin(phase)
    return _add_noise(time_series, noise_array, rng)
