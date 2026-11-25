"""Independent oracles for partial coherence and its shrinkage.

Every expected value here is computed in the test from its definition, not
from package internals: the population partial coherence from the inverse of
a VAR's exact spectrum; the exact mean of the sample partial coherence of
complex Gaussian observations; the closed-form first-order variance of a
sample coherency; the documented ``"auto"`` intensity written out one
observation at a time; and exact algebraic limits. Monte Carlo tests use a
fixed seed with replicate datasets on the time-window axis, and a limit of
five standard errors of the replicate mean; an allowance that is not
sampling error is stated beside its measured value.
"""

import numpy as np
import pytest
from scipy.special import hyp2f1

from spectral_connectivity import Connectivity, Multitaper
from spectral_connectivity.simulate import simulate_MVAR
from tests._expected_spectrum import _expected_multitaper_cross_spectrum
from tests._var_oracle import (
    _analytic_var,
    _coherency,
    _companion_spectral_radius,
    _dense_stable_var,
    _gaussian_coefficients,
    _mean_and_limit,
    _normalized_inverse,
)

_BINS = [2, 5, 9]  # of a 32-point FFT grid


def _chain(n_signals):
    """VAR(2) in which signal i drives signal i + 1 only."""
    coefficients = np.zeros((2, n_signals, n_signals))
    coefficients[0] = 0.5 * np.eye(n_signals) + 0.4 * np.eye(n_signals, k=-1)
    coefficients[1] = -0.25 * np.eye(n_signals)
    return coefficients, np.eye(n_signals)


def _dense(n_signals):
    """VAR(2) in which every signal drives every other, correlated innovations."""
    return _dense_stable_var(n_signals, n_lags=2, spectral_radius=0.8, seed=n_signals)


_MODELS = {"chain": _chain, "dense": _dense}


def _model_spectrum(model, n_signals):
    coefficients, noise_covariance = _MODELS[model](n_signals)
    assert _companion_spectral_radius(coefficients) < 1
    return _analytic_var(coefficients, noise_covariance, 32)[2][_BINS]


def _sample_spectrum(coefficients):
    """``mean_k x_k x_k^H`` over trials and tapers, ``(R, F, n, n)``."""
    return np.einsum("rktfi,rktfj->rfij", coefficients, coefficients.conj()) / (
        coefficients.shape[1] * coefficients.shape[2]
    )


def _expected_sample_coherence(true_coherence, m):
    """Exact ``E|r|^2`` of the sample coherence of ``m`` independent zero-mean
    complex Gaussian observations whose coherence is ``true_coherence``:
    ``1 - (m - 1) / m (1 - c) 2F1(1, 1; m + 1; c)``, the mean of Goodman's
    (1963) density (checked against numerical integration of that density
    for m = 6 to 54)."""
    values = 1 - (m - 1) / m * (1 - true_coherence) * hyp2f1(1, 1, m + 1, true_coherence)
    assert np.all(np.isfinite(values))
    return values


def _off(n_signals):
    return ~np.eye(n_signals, dtype=bool)


# ---------------------------------------------------------------------------
# 1. The unshrunk estimate against exact population values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model", "n_signals", "n_trials"),
    [("chain", 8, 40), ("dense", 8, 20), ("chain", 16, 40), ("dense", 16, 60)],
)
def test_unshrunk_mean_is_the_exact_mean_for_gaussian_observations(model, n_signals, n_trials):
    """For n complex Gaussian observations of p signals, the sample partial
    coherence of a pair is distributed as a sample coherence of m = n - p + 2
    observations with the population partial coherence (the 2 x 2 Schur
    complement is complex Wishart with m degrees of freedom), so its replicate
    mean is the exact mean of that coherence, for every pair (zero and
    non-zero) and frequency."""
    spectrum = _model_spectrum(model, n_signals)
    truth = _normalized_inverse(spectrum)
    coefficients = _gaussian_coefficients(spectrum, 300, n_trials, seed=1)
    estimate = Connectivity(coefficients, is_one_sided=True).partial_coherence()
    off = _off(n_signals)
    expected = _expected_sample_coherence(truth[..., off], n_trials - n_signals + 2)
    mean, limit = _mean_and_limit(estimate[..., off])
    assert np.all(np.abs(mean - expected) <= limit)
    if model == "chain":  # banded precision: pairs two or more apart are 0
        distance = np.abs(np.arange(n_signals)[:, None] - np.arange(n_signals))
        assert np.max(truth[..., distance >= 2]) < 1e-20


def test_multitaper_estimate_is_a_draw_around_the_expected_spectrum_version():
    """From VAR time series, the multitaper partial coherence averages to the
    exact mean (as above) around the partial coherence of the spectrum that
    the window and tapers average to, not around the model's: the remaining
    error is the window's bias, not sampling.

    The eigencoefficients of different tapers are not exactly independent or
    identically distributed, so the exact Gaussian mean is only approximate;
    no allowance was needed (largest difference 4.2 standard errors over 165
    pair-frequencies, limits 0.011-0.047). The mean differs from the exact
    mean around the model's own partial coherence by up to 0.13.
    """
    n_signals, n_trials_per_replicate, n_replicates = 6, 12, 120
    coefficients_var, noise_covariance = _dense_stable_var(
        n_signals, n_lags=2, spectral_radius=0.9, seed=3
    )
    time_series = simulate_MVAR(
        coefficients_var,
        noise_covariance=noise_covariance,
        n_time_samples=32,
        n_trials=n_trials_per_replicate * n_replicates,
        n_burnin_samples=200,
        random_state=4,
    )
    multitaper = Multitaper(time_series, sampling_frequency=1, time_halfbandwidth_product=2)
    fourier = multitaper.fft()  # (1, trials, tapers, n_fft, signals)
    n_tapers = fourier.shape[2]
    replicates = fourier.reshape(n_replicates, n_trials_per_replicate, *fourier.shape[2:])
    estimate = Connectivity(replicates).partial_coherence()

    expected_spectrum = _expected_multitaper_cross_spectrum(
        multitaper, coefficients_var, noise_covariance
    )
    model_spectrum = _analytic_var(coefficients_var, noise_covariance, 32)[2]
    bins = np.arange(3, 14)  # away from 0, the Nyquist frequency, and real coefficients
    off = _off(n_signals)
    m = n_trials_per_replicate * n_tapers - n_signals + 2
    windowed = _expected_sample_coherence(
        _normalized_inverse(expected_spectrum[bins])[..., off], m
    )
    model = _expected_sample_coherence(_normalized_inverse(model_spectrum[bins])[..., off], m)
    mean, limit = _mean_and_limit(estimate[:, bins][..., off])
    assert np.all(np.abs(mean - windowed) <= limit)
    assert np.max(np.abs(mean - model)) > 0.05  # the window's bias is resolved


# ---------------------------------------------------------------------------
# 2. Exact algebraic limits and definitions
# ---------------------------------------------------------------------------


def _test_coefficients(shape, seed, mixing=0.7):
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    n_signals = shape[-1]
    return white @ (np.eye(n_signals) + mixing * rng.standard_normal((n_signals, n_signals)))


def test_unshrunk_equals_the_per_signal_loaded_inverse_of_the_sample_spectrum():
    """No shrinkage: ``|P_ij|^2 / (P_ii P_jj)`` with ``P = inv(S + eps
    diag(S))`` and ``S`` formed in the test, with a large loading and signals
    scaled 1e-3 to 1e3 so the loading's form matters: loading every signal by
    the same multiple of the RMS entry differs by up to 0.95."""
    coefficients = _test_coefficients((2, 30, 2, 5, 4), seed=10) * [1e-3, 1.0, 10.0, 1e3]
    spectrum = _sample_spectrum(coefficients)
    power = np.real(np.diagonal(spectrum, axis1=-2, axis2=-1))
    off = _off(4)
    for regularization in (1e-12, 1e-2):
        actual = Connectivity(coefficients, is_one_sided=True).partial_coherence(
            regularization=regularization
        )
        expected = _normalized_inverse(spectrum, regularization * power)
        np.testing.assert_allclose(actual[..., off], expected[..., off], rtol=1e-9, atol=1e-14)
        assert np.isnan(actual[..., ~off]).all()
    # actual is at the large loading, 1e-2.
    rms = np.sqrt(np.mean(np.abs(spectrum) ** 2, axis=(-2, -1)))[..., None]
    uniform = _normalized_inverse(spectrum, 1e-2 * rms * np.ones(4))
    assert np.max(np.abs(actual - uniform)[..., off]) > 0.1


def test_shrinkage_limits_and_symmetry():
    """0 is no shrinkage bitwise, 1 gives exactly 0 off the diagonal, a fixed
    intensity is the inverse of ``(1 - lambda) S + lambda diag(S)`` formed
    in the test, and the result is symmetric."""
    coefficients = _test_coefficients((2, 12, 2, 5, 5), seed=13)
    connectivity = Connectivity(coefficients, is_one_sided=True)
    off = _off(5)
    unshrunk = connectivity.partial_coherence()
    for zero in (0, 0.0):
        np.testing.assert_array_equal(connectivity.partial_coherence(shrinkage=zero), unshrunk)
    assert np.all(connectivity.partial_coherence(shrinkage=1.0)[..., off] == 0.0)

    spectrum = _sample_spectrum(coefficients)
    power = np.real(np.diagonal(spectrum, axis1=-2, axis2=-1))
    for intensity in (0.05, 0.4, 0.9):
        actual = connectivity.partial_coherence(shrinkage=intensity)
        shrunk = spectrum * (1 - intensity) + intensity * power[..., None, :] * np.eye(5)
        expected = _normalized_inverse(shrunk, 1e-12 * power)
        np.testing.assert_allclose(actual[..., off], expected[..., off], rtol=1e-9)
        np.testing.assert_allclose(
            actual, np.swapaxes(actual, -1, -2), rtol=1e-12, equal_nan=True
        )


@pytest.mark.parametrize("shrinkage", [0.25, "auto"])
def test_two_signals_shrunk_partial_coherence_is_scaled_coherence(shrinkage):
    """With two signals there is nothing to condition on, so the shrunk
    partial coherence is exactly ``(1 - lambda)^2`` times the coherence."""
    coefficients = _test_coefficients((3, 8, 3, 6, 2), seed=14)
    connectivity = Connectivity(coefficients, is_one_sided=True)
    coherence = np.abs(_coherency(_sample_spectrum(coefficients))[..., 0, 1]) ** 2
    intensity = (
        connectivity.partial_coherence_shrinkage() if shrinkage == "auto" else shrinkage
    )
    actual = connectivity.partial_coherence(shrinkage=shrinkage)[..., 0, 1]
    np.testing.assert_allclose(actual, (1 - intensity) ** 2 * coherence, rtol=1e-9)


def test_auto_equals_each_bins_own_fixed_intensity():
    """``"auto"`` shrinks every bin by the intensity
    ``partial_coherence_shrinkage`` returns, applied in the test."""
    coefficients = _test_coefficients((3, 6, 2, 5, 6), seed=15)
    connectivity = Connectivity(coefficients, is_one_sided=True)
    intensity = connectivity.partial_coherence_shrinkage()
    spectrum = _sample_spectrum(coefficients)
    power = np.real(np.diagonal(spectrum, axis1=-2, axis2=-1))
    off = _off(6)
    expected = _normalized_inverse(
        spectrum * (1 - intensity[..., None, None] * off), 1e-12 * power
    )
    actual = connectivity.partial_coherence(shrinkage="auto")
    np.testing.assert_allclose(actual[..., off], expected[..., off], rtol=1e-9)


def test_repeated_calls_with_other_arguments_match_fresh_instances():
    """No cached value leaks between shrinkage settings, regularizations, or
    expectation types on one instance."""
    coefficients = _test_coefficients((3, 5, 5, 6, 4), seed=16)
    connectivity = Connectivity(coefficients)
    calls = [
        ("trials_tapers", {"shrinkage": "auto"}),
        ("trials_tapers", {"shrinkage": 0.3}),
        ("trials_tapers", {}),
        ("trials_tapers", {"shrinkage": "auto", "regularization": 1e-3}),
        ("tapers", {"shrinkage": "auto"}),
        ("trials_tapers", {"shrinkage": "auto"}),
    ]
    for expectation_type, kwargs in calls:
        connectivity.expectation_type = expectation_type
        fresh = Connectivity(coefficients, expectation_type=expectation_type)
        np.testing.assert_array_equal(
            connectivity.partial_coherence(**kwargs), fresh.partial_coherence(**kwargs)
        )
        np.testing.assert_array_equal(
            connectivity.partial_coherence_shrinkage(), fresh.partial_coherence_shrinkage()
        )


# ---------------------------------------------------------------------------
# 3. The "auto" intensity: its documented formula and its variance
# ---------------------------------------------------------------------------


def _documented_intensity(observations):
    """The documented intensity for one bin, from ``(n, p)`` observations:
    ``sum_{i != j} var(r_ij) / sum_{i != j} |r_ij|^2``, clipped to [0, 1], with
    ``var(r_ij) = sum_k |w_kij - (r_ij / 2)(u_ki + u_kj)|^2 / (n (n - 1))``,
    ``w_kij = x_ki conj(x_kj) / sqrt(S_ii S_jj)``, ``u_ki = |x_ki|^2 / S_ii``.
    Signals without power are left out."""
    keep = np.sum(np.abs(observations) ** 2, axis=0) > 0
    x = observations[:, keep]
    n = x.shape[0]
    total_variance = total_squared = 0.0
    power = np.mean(np.abs(x) ** 2, axis=0)
    for i in range(x.shape[1]):
        for j in range(x.shape[1]):
            if i == j:
                continue
            scale = np.sqrt(power[i] * power[j])
            w = x[:, i] * np.conj(x[:, j]) / scale
            r = w.mean()
            u_i, u_j = np.abs(x[:, i]) ** 2 / power[i], np.abs(x[:, j]) ** 2 / power[j]
            total_variance += np.sum(np.abs(w - r / 2 * (u_i + u_j)) ** 2) / (n * (n - 1))
            total_squared += np.abs(r) ** 2
    return float(np.clip(total_variance / total_squared, 0, 1))


@pytest.mark.parametrize(
    ("shape", "expectation_type", "mixing"),
    [
        ((2, 7, 3, 4, 5), "trials_tapers", 0.7),  # 21 observations, unclipped
        ((2, 3, 2, 4, 8), "trials_tapers", 0.7),  # fewer observations than signals
        ((2, 4, 3, 4, 2), "trials_tapers", 0.7),  # two signals
        ((2, 3, 2, 4, 6), "trials_tapers", 0.0),  # independent signals: clipped to 1
        ((2, 4, 5, 4, 4), "tapers", 0.5),  # trials kept, tapers averaged
        ((5, 2, 3, 4, 6), "time_tapers", 0.6),  # trials kept, time and tapers averaged
    ],
)
def test_auto_intensity_is_the_documented_formula(shape, expectation_type, mixing):
    coefficients = _test_coefficients(shape, seed=17, mixing=mixing)
    if shape[-1] > 2:
        coefficients[1, ..., -1] = 0.0  # a signal without power in window 1
    connectivity = Connectivity(
        coefficients, expectation_type=expectation_type, is_one_sided=True
    )
    with np.errstate(all="ignore"):
        actual = connectivity.partial_coherence_shrinkage()
    if expectation_type == "tapers":
        observations = np.moveaxis(coefficients, 2, -2)  # (time, trials, freq, tapers, signal)
    elif expectation_type == "time_tapers":  # (trials, freq, time x tapers, signal)
        observations = coefficients.transpose(1, 3, 0, 2, 4).reshape(
            shape[1], shape[3], -1, shape[-1]
        )
    else:
        observations = coefficients.transpose(0, 3, 1, 2, 4).reshape(
            shape[0], shape[3], -1, shape[-1]
        )
    expected = np.vectorize(
        lambda *index: _documented_intensity(observations[index]), otypes=[float]
    )(*np.indices(observations.shape[:-2], sparse=False))
    np.testing.assert_allclose(actual, expected, rtol=1e-10)
    if mixing == 0.0:
        assert np.any(actual == 1.0)  # where the unclipped value exceeds 1


def test_estimated_variance_matches_the_monte_carlo_variance_of_the_coherency():
    """The intensity times ``sum |r_ij|^2`` is the estimated ``sum var(r_ij)``.
    Its replicate mean, and the Monte Carlo ``E sum |r_ij - rho_ij|^2`` across
    replicates, both equal the first-order variance of a complex Gaussian
    coherency, ``sum (1 - |rho|^2)(1 - |rho|^2 / 2) / n``, with n = trials x
    tapers.

    Both are first-order in 1/n; the allowance of 1.5/n of the value covers
    the second-order terms (measured: the estimate 0.39% below, the Monte Carlo
    variance 0.1% above, at n = 200; five standard errors are 0.54% and 4.4%
    of the value). Without the terms for the estimated powers the estimate
    would be 1/n per pair, 4.0 times as large here."""
    rng = np.random.default_rng(18)
    n_signals, n_trials, n_tapers, n_replicates = 4, 100, 2, 8000
    mixing = rng.standard_normal((n_signals, n_signals)) + 1j * rng.standard_normal(
        (n_signals, n_signals)
    )
    mixing[:, 0] *= 3  # a strong common component: |rho| 0.59 to 0.91, complex
    spectrum = (mixing @ mixing.conj().T)[np.newaxis]
    off = _off(n_signals)
    rho = _coherency(spectrum)[0][off]
    coefficients = _gaussian_coefficients(spectrum, n_replicates, n_trials, n_tapers, seed=19)
    connectivity = Connectivity(coefficients, is_one_sided=True)
    intensity = connectivity.partial_coherence_shrinkage()[:, 0]
    coherency = _coherency(_sample_spectrum(coefficients))[:, 0][:, off]
    assert intensity.max() < 1  # unclipped, so the variance can be recovered
    estimated = intensity * np.sum(np.abs(coherency) ** 2, axis=-1)
    squared_error = np.sum(np.abs(coherency - rho) ** 2, axis=-1)

    n = n_trials * n_tapers
    first_order = np.sum((1 - np.abs(rho) ** 2) * (1 - np.abs(rho) ** 2 / 2)) / n
    allowance = 1.5 / n * first_order
    for values in (estimated, squared_error):
        mean, limit = _mean_and_limit(values)
        assert abs(mean - first_order) <= limit + allowance
    assert rho.size / n > 1.5 * first_order


# ---------------------------------------------------------------------------
# 4. Monte Carlo: the error that "auto" targets, and the partial-coherence error
# ---------------------------------------------------------------------------

_GRID = np.array([0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9])


@pytest.mark.parametrize("model", ["chain", "dense"])
@pytest.mark.parametrize(("n_signals", "n_trials"), [(8, 10), (8, 40), (16, 20), (16, 40)])
def test_auto_is_near_the_optimum_of_the_coherency_error(model, n_signals, n_trials):
    """``"auto"`` targets ``sum E|(1 - lambda) r - rho|^2`` over pairs (the
    coherency-scale loss in the Notes), whose minimizing fixed intensity over
    the replicates is ``1 - sum Re(conj(rho) r) / sum |r|^2``.

    The mean intensity is within 15% of it, or 0.01 where it is below 0.067
    (first-order estimate; measured 0.90-1.26 times, the largest ratio at an
    optimum of 0.027). The coherency error of ``"auto"`` is not above the
    unshrunk one by more than five standard errors at any frequency and is
    below it over all frequencies, and is within 20% of that of the optimal
    fixed intensity (measured 1.01-1.14 times: the noise of a per-bin
    estimate costs more than its bias; at one frequency it was 1.008 times the
    unshrunk error, within sampling)."""
    spectrum = _model_spectrum(model, n_signals)
    off = _off(n_signals)
    rho = _coherency(spectrum)[:, off]
    coefficients = _gaussian_coefficients(spectrum, 400, n_trials, seed=20)
    connectivity = Connectivity(coefficients, is_one_sided=True)
    coherency = _coherency(_sample_spectrum(coefficients))[..., off]
    intensity = connectivity.partial_coherence_shrinkage()

    aligned = np.sum(np.real(np.conj(rho) * coherency), axis=(0, -1))
    squared = np.sum(np.abs(coherency) ** 2, axis=(0, -1))
    optimum = 1 - aligned / squared  # per frequency
    mean_intensity, limit = _mean_and_limit(intensity)
    assert np.all(np.abs(mean_intensity - optimum) <= np.maximum(0.15 * optimum, 0.01) + limit)

    def error(shrink):
        return np.sum(np.abs((1 - shrink)[..., None] * coherency - rho) ** 2, axis=-1)

    auto = error(intensity)
    difference = auto - error(np.zeros_like(intensity))
    mean, limit = _mean_and_limit(difference)
    assert np.all(mean - limit <= 0)
    mean, limit = _mean_and_limit(difference.sum(axis=-1))
    assert mean + limit < 0
    best = error(np.broadcast_to(optimum, intensity.shape))
    assert np.all(auto.mean(axis=0) <= 1.2 * best.mean(axis=0))


@pytest.mark.parametrize("model", ["chain", "dense"])
@pytest.mark.parametrize(("n_signals", "n_trials"), [(8, 10), (8, 20), (16, 20), (16, 40)])
def test_auto_lowers_the_partial_coherence_error_with_few_observations(
    model, n_signals, n_trials
):
    """Against the VAR's partial coherence, with 1.25 to 2.5 observations per
    signal (where the Notes recommend ``"auto"``), ``"auto"`` lowers the mean
    squared error by more than five standard errors, and is within 35% of the
    best of a grid of fixed intensities (measured 0.045-0.36 times the
    unshrunk error, and 1.09-1.31 times the grid's best, which was at 0.05-0.3)."""
    spectrum = _model_spectrum(model, n_signals)
    off = _off(n_signals)
    truth = _normalized_inverse(spectrum)[:, off]
    coefficients = _gaussian_coefficients(spectrum, 300, n_trials, seed=21)
    connectivity = Connectivity(coefficients, is_one_sided=True)

    def error(estimate):
        return np.mean((estimate[..., off] - truth) ** 2, axis=(-2, -1))

    grid = np.array([error(connectivity.partial_coherence(shrinkage=float(s))) for s in _GRID])
    auto = error(connectivity.partial_coherence(shrinkage="auto"))
    mean, limit = _mean_and_limit(auto - grid[0])
    assert mean + limit < 0
    assert auto.mean() <= 1.35 * grid.mean(axis=1).min()
