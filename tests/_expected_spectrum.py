"""What a multitaper estimate of a VAR's cross-spectrum averages to.

The cross-spectrum that a detrended, tapered, zero-padded window of a VAR
averages to follows exactly from the model's autocovariance
(``_var_autocovariance`` in ``tests/_var_oracle.py``). It differs from the
model's spectrum by the bias of the window and tapers, so an estimate can be
compared with it knowing that only sampling error is left
(``_assert_multitaper_estimates``).
"""

import numpy as np
from scipy.signal import fftconvolve
from scipy.signal.windows import dpss

from tests._var_oracle import _var_autocovariance

_DETREND_BASIS = {
    None: lambda n_time: np.ones((n_time, 0)),
    "constant": lambda n_time: np.ones((n_time, 1)),
    "linear": lambda n_time: np.stack([np.ones(n_time), np.arange(n_time)], axis=1),
}


def _expected_tapered_cross_spectrum(
    coefficients, noise_covariance, tapers, n_fft, detrend_type
):
    """Exact ``E[J(f) J(f)^H]`` of a VAR's tapered Fourier coefficients, averaged
    over tapers, shape ``(n_fft, n_signals, n_signals)`` in FFT order.

    ``J(f) = sum_t taper(t) y(t) exp(-2 pi i f t / n_fft)``, where ``y`` is a
    window of ``n_time`` consecutive samples of the VAR, detrended as
    ``scipy.signal.detrend`` does (``detrend_type`` is ``None``,
    ``"constant"`` or ``"linear"``). ``tapers`` has shape
    ``(n_time, n_tapers)``, and the FFT is at least as long as the window.

    Detrending subtracts a least-squares fit, ``y = x - basis @ beta`` with
    ``beta = fit @ x``. With ``Gamma`` the autocovariance of ``x``, the
    expectation is the sum over ``s`` and ``t`` of
    ``taper(s) taper(t) exp(-i w (s - t))`` times
    ``Gamma(s - t) - E[x(s) beta^T] basis(t)^T - basis(s) E[beta x(t)^T] +
    basis(s) E[beta beta^T] basis(t)^T``. The first term depends on ``s - t``
    alone and is the transform of ``Gamma`` times the taper's
    autocorrelation; the others are products of transforms of the taper times
    the basis and times ``E[x(s) beta^T]``.
    """
    n_time, n_tapers = tapers.shape
    assert n_fft >= n_time, "the FFT must be at least as long as the window"
    n_signals = coefficients.shape[-1]
    gamma = _var_autocovariance(coefficients, noise_covariance, n_time - 1)
    # Gamma at lags -(n_time - 1) ... n_time - 1; Gamma(-h) is Gamma(h) transposed.
    lags = np.arange(-(n_time - 1), n_time)
    two_sided = np.concatenate([gamma[:0:-1].swapaxes(-1, -2), gamma])
    basis = _DETREND_BASIS[detrend_type](n_time)  # (n_time, n_basis)
    fit = np.linalg.pinv(basis)  # (n_basis, n_time)
    # window_fit[j, s] = E[x(s) beta_j^T] = sum_u fit[j, u] Gamma(s - u)
    window_fit = np.array(
        [fftconvolve(two_sided, row[:, None, None], mode="valid", axes=0) for row in fit]
    ).reshape(-1, n_time, n_signals, n_signals)
    fit_fit = np.einsum("js,lsab->jlab", fit, window_fit)  # E[beta_j beta_l^T]

    expected = np.zeros((n_fft, n_signals, n_signals), dtype=complex)
    for taper in tapers.T:
        autocorrelation = np.correlate(taper, taper, mode="full")  # at ``lags``
        stationary = np.zeros((n_fft, n_signals, n_signals))
        np.add.at(stationary, lags % n_fft, two_sided * autocorrelation[:, None, None])
        basis_transform = np.fft.fft(taper[:, None] * basis, n=n_fft, axis=0)
        window_fit_transform = np.fft.fft(
            np.moveaxis(taper[:, None, None] * window_fit, 0, 1), n=n_fft, axis=0
        )  # (n_fft, n_basis, n_signals, n_signals)
        cross = np.einsum("fjab,fj->fab", window_fit_transform, basis_transform.conj())
        expected += (
            np.fft.fft(stationary, axis=0)
            - cross
            - cross.conj().swapaxes(-1, -2)
            + np.einsum("fj,fl,jlab->fab", basis_transform, basis_transform.conj(), fit_fit)
        )
    return expected / n_tapers


def _expected_multitaper_cross_spectrum(multitaper, coefficients, noise_covariance):
    """What a ``Multitaper``'s cross-spectral estimate of a VAR averages to,
    ``(n_fft_samples, n_signals, n_signals)`` in FFT order.

    Only the transform's settings are read (window length, FFT length,
    time-halfbandwidth product, number of tapers, detrending, sampling
    frequency); the tapers are SciPy's. Those settings are taken as given: a
    transform that chose a different window length or number of tapers than
    it should would be compared with the expectation for its own choice.
    ``Multitaper`` scales unit-energy tapers by ``sqrt(sampling_frequency)``
    and the FFT by ``1 / sampling_frequency``, so its cross-spectrum is that
    of unit-energy tapers over the sampling frequency. Uniform taper weighting
    is assumed, and that the transform kept every taper it reports.
    """
    assert multitaper.taper_weighting == "uniform", "only uniform taper weighting"
    assert multitaper.tapers.shape[1] == multitaper.n_tapers, "the transform dropped a taper"
    n_time = multitaper.n_time_samples_per_window
    tapers = dpss(n_time, multitaper.time_halfbandwidth_product, multitaper.n_tapers)
    return (
        _expected_tapered_cross_spectrum(
            coefficients,
            noise_covariance,
            tapers.reshape(-1, n_time).T,
            multitaper.n_fft_samples,
            multitaper.detrend_type,
        )
        / multitaper.sampling_frequency
    )


def _whiten(estimate, expected):
    """``L^-1 estimate L^-H`` with ``expected = L L^H``.

    When ``expected`` is the expectation of the Hermitian ``estimate``, the
    result averages to the identity whatever the spectrum's shape, so its
    entries can be compared across frequencies and averaged over them.
    """
    cholesky = np.linalg.cholesky(expected)
    half = np.linalg.solve(cholesky, estimate)  # L^-1 estimate
    return np.linalg.solve(cholesky, half.conj().swapaxes(-1, -2))


def _assert_multitaper_estimates(multitaper, estimate, expected, label):
    """``estimate``, the cross-spectrum of one window from ``multitaper`` on its
    full FFT grid, is a draw around ``expected``, bin by bin and on average
    over frequency. ``label`` names the estimate in a failure message.

    The estimate averages ``n_observations`` (trials times tapers) outer
    products of Fourier coefficients. Whitened by ``expected``, each of its
    entries averages products of unit-variance Gaussians. Were the
    observations independent, its standard error would be
    ``1 / sqrt(n_observations)``, and ``sqrt(2 / n_observations)`` at 0 Hz and
    Nyquist, where the coefficients are real. The larger is taken as the
    standard error of every bin, and each bin is allowed six.

    The mean of the whitened estimate over the non-negative frequencies is far
    more precise. Bins within a taper's bandwidth are correlated; counting
    each band as one bin gives
    ``sqrt(n_bins_per_band / (n_observations n_frequencies))`` as the standard
    error of the mean, which is allowed five.

    Both standard errors are working estimates, not bounds. The tapers of one
    trial are uncorrelated, with equal expected power, only where the spectrum
    is flat across their bandwidth. For the transforms tested with this
    helper, the exact standard deviation of a whitened entry is up to 1.23
    times the first (at 0 Hz, with several tapers and detrending) and that of
    the mean 0.80 to 0.97 times the second, so the limits are 4.9 to 6.2
    standard deviations. They are not calibrated beyond that: for several
    tapers on a spectral peak much narrower than their bandwidth (poles at a
    radius of 0.98 in a 64-sample window) the ratios reach 2, and a correct
    estimate fails.
    """
    n_frequencies = multitaper.n_fft_samples // 2 + 1
    n_observations = multitaper.n_trials * multitaper.n_tapers
    n_bins_per_band = (
        2
        * multitaper.time_halfbandwidth_product
        * multitaper.n_fft_samples
        / multitaper.n_time_samples_per_window
    )
    deviation = _whiten(estimate[:n_frequencies], expected[:n_frequencies]) - np.eye(
        expected.shape[-1]
    )

    bin_limit = 6 * np.sqrt(2 / n_observations)
    frequency, row, column = np.unravel_index(np.argmax(np.abs(deviation)), deviation.shape)
    assert np.abs(deviation).max() < bin_limit, (
        f"{label}: whitened entry [{row}, {column}] at frequency bin {frequency} is "
        f"{np.abs(deviation).max():.3g} from the identity (limit {bin_limit:.3g})"
    )

    mean_limit = 5 * np.sqrt(n_bins_per_band / (n_observations * n_frequencies))
    mean_deviation = np.abs(deviation.mean(axis=0))
    row, column = np.unravel_index(np.argmax(mean_deviation), mean_deviation.shape)
    assert mean_deviation.max() < mean_limit, (
        f"{label}: whitened entry [{row}, {column}], averaged over frequency, is "
        f"{mean_deviation.max():.3g} from the identity (limit {mean_limit:.3g})"
    )
