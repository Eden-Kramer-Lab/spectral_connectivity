"""Functions to simulate processes."""

import numpy as np


def simulate_MVAR(
    coefficients,
    noise_covariance=None,
    n_time_samples=100,
    n_trials=1,
    n_burnin_samples=100,
):
    """
    Simulate multivariate autoregressive (MVAR) process.

    Parameters
    ----------
    coefficients : array, shape (n_time_samples, n_lags, n_signals, n_signals)
    noise_covariance : array, shape (n_signals, n_signals)

    Returns
    -------
    time_series : array, shape (n_time_samples - n_burnin_samples,
                                n_trials, n_signals)

    """
    n_lags, n_signals, _ = coefficients.shape
    if noise_covariance is None:
        noise_covariance = np.eye(n_signals)
    time_series = np.random.multivariate_normal(
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
