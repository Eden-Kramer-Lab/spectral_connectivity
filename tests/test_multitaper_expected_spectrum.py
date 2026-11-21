"""The multitaper cross-spectral estimate against its exact expectation for a
VAR (``tests/_expected_spectrum.py``), and that expectation against the
covariance of the window written out in full."""

import numpy as np
import pytest
from scipy.signal import detrend
from scipy.signal.windows import dpss

from spectral_connectivity import Connectivity, Multitaper
from spectral_connectivity.simulate import simulate_MVAR
from tests._backend_helpers import to_host
from tests._expected_spectrum import (
    _DETREND_BASIS,
    _assert_multitaper_estimates,
    _expected_multitaper_cross_spectrum,
    _expected_tapered_cross_spectrum,
)
from tests._var_oracle import _analytic_var, _dense_stable_var, _var_autocovariance

_DETREND_TYPES = list(_DETREND_BASIS)
# Every signal drives every other, with a spectral peak that a 128-sample
# window does not resolve.
_COEFFICIENTS, _NOISE_COVARIANCE = _dense_stable_var(
    n_signals=3, n_lags=2, spectral_radius=0.9, seed=3
)


@pytest.mark.parametrize("detrend_type", _DETREND_TYPES)
@pytest.mark.parametrize(("n_time", "n_fft"), [(64, 64), (63, 128)], ids=["same", "padded"])
def test_expected_cross_spectrum_is_the_covariance_of_the_tapered_window(
    detrend_type, n_time, n_fft
):
    """The expectation equals ``L C L^H`` written out in full: ``C`` the
    covariance of every pair of samples of the window and ``L`` the matrix that
    detrends, tapers and Fourier transforms it."""
    tapers = dpss(n_time, 2.5, 4).T
    gamma = _var_autocovariance(_COEFFICIENTS, _NOISE_COVARIANCE, n_time - 1)
    lag = np.arange(n_time)[:, np.newaxis] - np.arange(n_time)
    # covariance[s, t] = E[x(s) x(t)^T] = Gamma(s - t), and Gamma(-h) = Gamma(h)^T
    covariance = np.where(
        (lag >= 0)[..., np.newaxis, np.newaxis],
        gamma[np.abs(lag)],
        gamma[np.abs(lag)].swapaxes(-1, -2),
    )
    detrending = (
        np.eye(n_time)
        if detrend_type is None
        else detrend(np.eye(n_time), axis=0, type=detrend_type)
    )
    fourier = np.exp(-2j * np.pi * np.outer(np.arange(n_fft), np.arange(n_time)) / n_fft)
    in_full = np.zeros((n_fft, 3, 3), dtype=complex)
    for taper in tapers.T:
        transform = (fourier * taper) @ detrending  # (n_fft, n_time)
        in_full += np.einsum("fs,stab,ft->fab", transform, covariance, transform.conj()) / 4

    expected = _expected_tapered_cross_spectrum(
        _COEFFICIENTS, _NOISE_COVARIANCE, tapers, n_fft, detrend_type
    )
    np.testing.assert_allclose(expected, in_full, rtol=0, atol=1e-11 * np.abs(in_full).max())


@pytest.mark.parametrize("detrend_type", _DETREND_TYPES)
@pytest.mark.parametrize("n_fft_samples", [None, 200], ids=["window_length", "zero_padded"])
def test_multitaper_estimate_is_a_draw_around_its_expected_cross_spectrum(
    detrend_type, n_fft_samples
):
    """In each of two sliding windows, the estimated cross-spectrum differs from
    its exact expectation by no more than sampling error, bin by bin and (far
    more tightly) on average over frequency, with any detrending and with
    zero-padding. The window is short for this model, so the expectation is
    not the model's spectrum: that one the same estimate does not match."""
    sampling_frequency = 250
    time_series = simulate_MVAR(
        _COEFFICIENTS,
        noise_covariance=_NOISE_COVARIANCE,
        n_time_samples=256,
        n_trials=1000,
        n_burnin_samples=300,
        random_state=0,
    )
    multitaper = Multitaper(
        time_series,
        sampling_frequency=sampling_frequency,
        time_halfbandwidth_product=3,
        time_window_duration=128 / sampling_frequency,
        detrend_type=detrend_type,
        n_fft_samples=n_fft_samples,
    )
    estimate = to_host(
        Connectivity.from_transform(multitaper)._expectation_cross_spectral_matrix()
    )
    assert estimate.shape[0] == 2

    expected = _expected_multitaper_cross_spectrum(
        multitaper, _COEFFICIENTS, _NOISE_COVARIANCE
    )
    for window, window_estimate in enumerate(estimate):
        _assert_multitaper_estimates(multitaper, window_estimate, expected, f"window {window}")

    model_spectrum = (
        _analytic_var(_COEFFICIENTS, _NOISE_COVARIANCE, multitaper.n_fft_samples)[2]
        / sampling_frequency
    )
    # No bin-by-bin draw around the model's own spectrum.
    with pytest.raises(AssertionError, match="at frequency bin"):
        _assert_multitaper_estimates(multitaper, estimate[0], model_spectrum, "model")
