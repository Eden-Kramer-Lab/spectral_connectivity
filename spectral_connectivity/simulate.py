"""Functions to simulate time series processes for connectivity analysis.

This module provides functions for generating synthetic time series data
following various dynamical models, primarily multivariate autoregressive
(MVAR) processes commonly used for testing connectivity methods.
"""

import numpy as np
from numpy.typing import NDArray


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
    and E(t) is multivariate Gaussian noise.

    Parameters
    ----------
    coefficients : NDArray[floating], shape (n_lags, n_signals, n_signals)
        MVAR coefficient matrices for each lag. Each A_k matrix defines
        the linear influence of signals at lag k.
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

    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )

    time_series = rng.multivariate_normal(
        np.zeros((n_signals,)),
        noise_covariance,
        size=(n_time_samples + n_burnin_samples, n_trials),
    )

    for time_ind in np.arange(n_lags, n_time_samples + n_burnin_samples):
        for lag_ind in np.arange(n_lags):
            time_series[time_ind, ...] += np.matmul(
                coefficients[np.newaxis, np.newaxis, lag_ind, ...],
                time_series[time_ind - (lag_ind + 1), ..., np.newaxis],
            ).squeeze()
    return time_series[n_burnin_samples:, ...]
