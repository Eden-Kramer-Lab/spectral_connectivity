"""The Wilson factorization and the directed measures against independent factorizations.

Two oracles that involve no Wilson iteration:

- The exact spectrum of a stable VAR has the minimum-phase factor
  ``G = H U``, its transfer function ``H`` times the upper-triangular ``U``
  with ``U Uᴴ = Σ``. That is the factor the iteration's upper-triangular
  zero-lag constraint selects, so the factor itself is compared, not only
  ``G Gᴴ``.
- A multitaper estimate of one window of ``N`` samples, zero padded to
  ``n_fft >= 2N - 1`` bins, is a trigonometric polynomial of degree ``N - 1``:
  its autocovariance vanishes beyond lag ``N - 1``. Its minimum-phase factor is
  then the moving-average factor ``W(z) = I + C (zI - A)⁻¹ K`` of the
  innovations form of that autocovariance, found from the discrete algebraic
  Riccati equation of the positive-real lemma (Faurre's equation; see
  Lindquist & Picci, *Linear Stochastic Systems*, 2015, ch. 6). The factor is
  minimum phase exactly when ``A - K C`` is stable, which the oracle asserts.
  On a grid padded well beyond ``2N`` the Wilson factor is that factor up to
  the iteration's tolerance and the grid's aliasing.

The estimates use signals whose powers differ by up to twelve orders of
magnitude and data that are complex, so both the half-spectrum and the
two-sided iteration are checked. Replicate windows are on the time-window axis,
so one factorization covers all of them.
"""

import functools
import itertools
import warnings

import numpy as np
import pytest

from spectral_connectivity import Connectivity, DirectedOrientationWarning, Multitaper
from spectral_connectivity import minimum_phase_decomposition as mpd_module
from spectral_connectivity.minimum_phase_decomposition import minimum_phase_decomposition
from spectral_connectivity.simulate import simulate_MVAR
from tests._backend_helpers import to_device, to_host, xp
from tests._var_oracle import (
    _analytic_directed_measures,
    _analytic_var,
    _dense_stable_var,
    _prediction_error_covariance,
)


def _reverse_cholesky(matrix):
    """Upper-triangular ``U`` with a positive diagonal and ``U Uᴴ = matrix``."""
    return np.linalg.cholesky(matrix[..., ::-1, ::-1])[..., ::-1, ::-1]


def _mirrored(spectrum):
    """``spectrum`` made exactly conjugate-symmetric from its non-negative bins."""
    n_fft = spectrum.shape[0]
    mirrored = spectrum.copy()
    mirrored[0] = mirrored[0].real
    if n_fft % 2 == 0:
        mirrored[n_fft // 2] = mirrored[n_fft // 2].real
    mirrored[n_fft // 2 + 1 :] = mirrored[1 : (n_fft + 1) // 2][::-1].conj()
    return mirrored


def _riccati_minimum_phase_factor(spectrum, max_lag):
    """Minimum-phase factor of a grid spectrum whose autocovariance vanishes
    beyond ``max_lag``, shape ``(n_fft, n_signals, n_signals)``.

    The autocovariance ``R_k`` (``S(ω) = Σ_k R_k e^{-iωk}``) is realized as
    ``R_k = C A^{k-1} Ḡ`` with ``A`` the block shift, ``Ḡ = [I, 0, ...]`` and
    ``C = [R_1, ..., R_q]``. The minimal solution ``P`` of
    ``P = A P Aᴴ + (Ḡ - A P Cᴴ)(R_0 - C P Cᴴ)⁻¹(Ḡ - A P Cᴴ)ᴴ`` gives the
    innovation covariance ``Σ = R_0 - C P Cᴴ`` and gain ``K``, and the factor
    is ``W(e^{iω}) U`` with ``U Uᴴ = Σ`` upper triangular. For ``-P`` the
    equation is the Kalman filter's with state noise 0, observation noise
    ``R_0`` and cross-covariance ``Ḡ``, solved by
    ``_prediction_error_covariance``.

    The minimum-phase factor of ``D S D`` is ``D G`` for a positive diagonal
    ``D`` (``D U`` is again upper triangular), so the equation is solved for
    signals scaled to unit power, where the iteration's tolerance, relative to
    the largest entry, holds for every signal.
    """
    n_fft, n_signals, _ = spectrum.shape
    scale = np.sqrt(np.mean(np.diagonal(spectrum, axis1=-2, axis2=-1).real, axis=0))
    spectrum = spectrum / np.multiply.outer(scale, scale)
    autocovariance = np.fft.ifft(spectrum, axis=0)
    beyond = autocovariance[max_lag + 1 : n_fft - max_lag]
    assert np.abs(beyond).max() < 1e-10 * np.abs(autocovariance[0]).max()
    n_states = max_lag * n_signals
    shift = np.zeros((n_states, n_states), dtype=complex)
    shift[n_signals:, :-n_signals] = np.eye(n_states - n_signals)
    observation = np.concatenate(list(autocovariance[1 : max_lag + 1]), axis=1)
    input_matrix = np.zeros((n_states, n_signals), dtype=complex)
    input_matrix[:n_signals] = np.eye(n_signals)
    zero_lag = (autocovariance[0] + autocovariance[0].conj().T) / 2
    state = -_prediction_error_covariance(
        shift,
        observation,
        np.zeros((n_states, n_states), dtype=complex),
        zero_lag,
        input_matrix,
    )
    innovation = zero_lag - observation @ state @ observation.conj().T
    innovation = (innovation + innovation.conj().T) / 2
    gain = (input_matrix - shift @ state @ observation.conj().T) @ np.linalg.inv(innovation)
    # The inverse filter is stable, so W is minimum phase.
    assert np.abs(np.linalg.eigvals(shift - gain @ observation)).max() < 1
    lags = np.zeros((n_fft, n_signals, n_signals), dtype=complex)
    lags[0] = np.eye(n_signals)
    power = np.eye(n_states)
    for lag in range(1, max_lag + 1):
        lags[lag] = observation @ power @ gain
        power = power @ shift
    return scale[:, np.newaxis] * (np.fft.fft(lags, axis=0) @ _reverse_cholesky(innovation))


def _relative_row_error(factor, expected):
    """Largest error of each signal's row of the factor, relative to that row's
    largest entry, so a weak signal's row is judged on its own scale."""
    row_scale = np.abs(expected).max(axis=(-3, -1), keepdims=True)
    return np.abs(factor - expected).max(axis=(-3, -1), keepdims=True) / row_scale


# Dense VARs at spectral radius 0.99 (from ``_dense_stable_var``) and per-signal
# scales. The factorization from ``chol(R0)ᵀ`` without scaling the signals
# (before 3.0) misses each of these factors by 4e-6 to 8e-2 of a row's size.
_DENSE_VARS = pytest.mark.parametrize(
    ("model", "scales"),
    [
        pytest.param((4, 1, 1572889066, False), (1.0, 1.0, 1.0, 1.0), id="var1_4_signals"),
        pytest.param((5, 3, 201839291, True), (1.0,) * 5, id="var3_complex"),
        pytest.param((3, 2, 7, False), (1e-4, 1.0, 1e4), id="var2_rescaled"),
        pytest.param((4, 2, 11, True), (1e3, 1e-3, 1.0, 1e-6), id="var2_complex_rescaled"),
        pytest.param((2, 3, 3, False), (1e-6, 1e6), id="var3_2_signals_rescaled"),
    ],
)


@_DENSE_VARS
def test_factor_of_an_exact_var_spectrum_is_its_minimum_phase_factor(model, scales):
    """The factor of a VAR's exact spectrum is ``H U``, row by row to 1e-6 of
    each row's size (observed at most 9e-9 at the default tolerance), on the
    two-sided path and, for real models, the half-spectrum path, with no
    warning. The 8192-bin grid keeps the aliasing of these slowly decaying
    models below 1e-8 (at 2048 bins it reaches 1e-3)."""
    n_signals, n_lags, seed, complex_valued = model
    coefficients, noise_covariance = _dense_stable_var(
        n_signals, n_lags, 0.99, seed, complex_valued=complex_valued
    )
    scale = np.diag(scales)
    coefficients = scale @ coefficients @ np.linalg.inv(scale)
    noise_covariance = scale @ noise_covariance @ scale
    _, transfer_function, spectrum = _analytic_var(coefficients, noise_covariance, 8192)
    expected = transfer_function @ _reverse_cholesky(noise_covariance)
    spectra = [spectrum] if complex_valued else [spectrum, _mirrored(spectrum)]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        factors = [
            to_host(minimum_phase_decomposition(to_device(each[np.newaxis])))[0]
            for each in spectra
        ]
    for factor in factors:
        assert _relative_row_error(factor, expected).max() < 1e-6


def _multitaper_estimate(n_signals, scales, complex_valued, seed, n_windows=6, n_time=16):
    """Fourier coefficients and the cross-spectrum of ``n_windows`` windows of a
    dense VAR, zero padded to ``16 n_time`` bins.

    Five trials of three tapers make 15 observations per window, so every
    estimate has full rank with margin. Complex data are the sum of two
    independent real simulations, the second times ``i``.
    """
    coefficients, noise_covariance = _dense_stable_var(n_signals, 2, 0.9, seed)

    def simulate(random_state):
        return simulate_MVAR(
            coefficients,
            noise_covariance=noise_covariance,
            n_time_samples=n_time * n_windows,
            n_trials=5,
            random_state=random_state,
        )

    def transform(random_state):
        return Multitaper(
            simulate(random_state) * np.asarray(scales),
            sampling_frequency=1.0,
            time_halfbandwidth_product=2,
            time_window_duration=n_time,
            n_fft_samples=16 * n_time,
            detrend_type=None,
        ).fft()

    # Multitaper takes real signals only; its transform is linear, so the
    # coefficients of x + i y are those of x plus i times those of y.
    fourier_coefficients = transform(seed)
    if complex_valued:
        fourier_coefficients = fourier_coefficients + 1j * transform(seed + 1)
    coefficients_host = to_host(fourier_coefficients)
    # Mean over trials and tapers of X Xᴴ: (window, frequency, signal, signal).
    spectrum = np.mean(
        coefficients_host[..., :, np.newaxis] * coefficients_host[..., np.newaxis, :].conj(),
        axis=(1, 2),
    )
    return fourier_coefficients, spectrum


_ESTIMATES = pytest.mark.parametrize(
    ("n_signals", "scales", "complex_valued"),
    [
        pytest.param(2, (1.0, 1.0), False, id="2_signals"),
        pytest.param(3, (1e-6, 1.0, 1e6), False, id="3_signals_rescaled"),
        pytest.param(3, (1e-6, 1.0, 1e6), True, id="3_complex_signals_rescaled"),
        pytest.param(5, (1.0, 1e3, 1e-3, 1e5, 1e-5), False, id="5_signals_rescaled"),
    ],
)


@functools.cache
def _estimate_with_riccati_factors(n_signals, scales, complex_valued):
    """``_multitaper_estimate`` (seeded by ``n_signals``) and the Riccati
    factor of each of its windows, computed once for the tests sharing them."""
    fourier_coefficients, spectrum = _multitaper_estimate(
        n_signals, scales, complex_valued, seed=n_signals
    )
    riccati = np.stack([_riccati_minimum_phase_factor(each, max_lag=15) for each in spectrum])
    return fourier_coefficients, spectrum, riccati


# Wilson's factor of these estimates differs from the Riccati factor by at most
# 7e-8 of a row's size (the 1e-8 tolerance, rounding and the grid's aliasing);
# the allowance is 1e-5.
_ESTIMATE_FACTOR_ALLOWANCE = 1e-5


@_ESTIMATES
def test_factor_of_a_multitaper_estimate_is_its_riccati_factor(
    n_signals, scales, complex_valued
):
    """Every window's factor matches the minimum-phase factor that the
    Riccati equation gives for the same estimate, row by row, with no
    warning."""
    _, spectrum, riccati = _estimate_with_riccati_factors(n_signals, scales, complex_valued)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        factor = to_host(minimum_phase_decomposition(to_device(spectrum)))
    for window, expected in enumerate(riccati):
        error = _relative_row_error(factor[window], expected).max()
        assert error < _ESTIMATE_FACTOR_ALLOWANCE, (window, error)


def _directed_measures_of_factor(factor):
    """DTF, PDC, gPDC and dDTF read from a two-sided factor in closed form,
    ``[window][measure]`` in the native ``[target, source]`` layout."""
    zero_lag = np.fft.ifft(factor, axis=-3)[..., 0, :, :]
    transfer_function = factor @ np.linalg.inv(zero_lag)[..., np.newaxis, :, :]
    noise_covariance = zero_lag @ zero_lag.conj().swapaxes(-1, -2)
    return [
        _analytic_directed_measures(np.linalg.inv(each), each, covariance)
        for each, covariance in zip(transfer_function, noise_covariance, strict=True)
    ]


_TRANSFER_FUNCTION_MEASURES = (
    "directed_transfer_function",
    "partial_directed_coherence",
    "generalized_partial_directed_coherence",
    "direct_directed_transfer_function",
)


@_ESTIMATES
def test_transfer_function_measures_of_an_estimate_match_its_riccati_factor(
    n_signals, scales, complex_valued
):
    """DTF, PDC, gPDC and dDTF of each window match those of the Riccati
    factor of the same estimate to 1e-5 (observed at most 2e-7; they are
    ratios of magnitudes of order one)."""
    fourier_coefficients, _, riccati = _estimate_with_riccati_factors(
        n_signals, scales, complex_valued
    )
    connectivity = Connectivity(fourier_coefficients=fourier_coefficients)
    expected = _directed_measures_of_factor(riccati)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warnings.filterwarnings("ignore", category=DirectedOrientationWarning)
        for measure in _TRANSFER_FUNCTION_MEASURES:
            result = to_host(getattr(connectivity, measure)())
            for window, oracle in enumerate(expected):
                # The measures are source-first: [i, j] is i -> j.
                np.testing.assert_allclose(
                    result[window],
                    np.swapaxes(oracle[measure], -1, -2),
                    atol=1e-5,
                    rtol=0,
                    err_msg=f"{measure}, window {window}",
                )


def _pairwise_granger_of_riccati_factors(spectrum):
    """Geweke's pairwise spectral Granger from the Riccati factor of every 2x2
    sub-spectrum, ``(window, n_nonnegative, source, target)``."""
    n_windows, n_fft, n_signals, _ = spectrum.shape
    n_nonnegative = n_fft // 2 + 1
    granger = np.full((n_windows, n_nonnegative, n_signals, n_signals), np.nan)
    for window, (first, second) in itertools.product(
        range(n_windows), itertools.combinations(range(n_signals), 2)
    ):
        pair = [first, second]
        sub_spectrum = spectrum[window][:, pair][:, :, pair]
        factor = _riccati_minimum_phase_factor(sub_spectrum, max_lag=15)
        zero_lag = np.fft.ifft(factor, axis=0)[0]
        transfer_function = (factor @ np.linalg.inv(zero_lag))[:n_nonnegative]
        noise = zero_lag @ zero_lag.conj().T
        power = sub_spectrum[:n_nonnegative, [0, 1], [0, 1]].real
        for target, source in ((0, 1), (1, 0)):
            intrinsic = (
                noise[source, source] - abs(noise[target, source]) ** 2 / noise[target, target]
            ).real
            explained = intrinsic * np.abs(transfer_function[:, target, source]) ** 2
            granger[window, :, pair[source], pair[target]] = np.log(
                power[:, target] / (power[:, target] - explained)
            )
    return granger


@pytest.mark.parametrize(
    ("n_signals", "scales"),
    [
        pytest.param(3, (1.0, 1.0, 1.0), id="3_signals"),
        pytest.param(3, (1e-6, 1.0, 1e6), id="3_signals_rescaled"),
        pytest.param(4, (1e5, 1e-5, 1.0, 1e-3), id="4_signals_rescaled"),
    ],
)
def test_pairwise_granger_of_an_estimate_matches_the_riccati_factors_of_its_pairs(
    n_signals, scales
):
    """Pairwise spectral Granger of each window matches Geweke's formula read
    from the Riccati factor of each pair's 2x2 sub-spectrum, to 1e-5
    (observed at most 2e-8, for values up to 2.7)."""
    fourier_coefficients, spectrum = _multitaper_estimate(
        n_signals, scales, complex_valued=False, seed=n_signals
    )
    connectivity = Connectivity(fourier_coefficients=fourier_coefficients)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warnings.filterwarnings("ignore", category=DirectedOrientationWarning)
        result = to_host(connectivity.pairwise_spectral_granger_prediction())
    expected = _pairwise_granger_of_riccati_factors(spectrum)
    off_diagonal = ~np.eye(n_signals, dtype=bool)
    np.testing.assert_allclose(
        result[..., off_diagonal], expected[..., off_diagonal], atol=1e-5, rtol=0
    )


# A chain 0 -> 1 -> 2 of nearly integrating signals; their powers differ by
# five orders of magnitude before any rescaling.
_INTEGRATING_CHAIN = 0.99 * np.eye(3)[np.newaxis] + np.array(
    [[[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.2, 0.0]]]
)


@pytest.mark.parametrize(
    "measure",
    [
        "pairwise_spectral_granger_prediction",
        "conditional_spectral_granger_prediction",
        "directed_coherence",
        "generalized_partial_directed_coherence",
        "blockwise",
    ],
)
def test_scale_invariant_measures_of_estimates_do_not_change_when_the_signals_are_standardized(
    measure,
):
    """Spectral Granger, directed coherence and gPDC are unchanged by
    rescaling the signals. On 20 windows of a multitaper estimate on the
    default grid, the values for the raw signals and for the signals divided
    by their standard deviations (7, 1e2 and 2e3) agree to 1e-9 (observed at
    most 7e-14; from ``chol(R0)ᵀ`` with the RMS-loaded inverses, before
    the covariant start, they differed by 0.5 to 1.1)."""
    data = simulate_MVAR(
        _INTEGRATING_CHAIN,
        noise_covariance=np.eye(3),
        n_time_samples=20 * 256,
        n_trials=4,
        random_state=0,
    )
    standardized = data / data.std(axis=(0, 1))

    def compute(signals):
        connectivity = Connectivity.from_transform(
            Multitaper(
                signals,
                sampling_frequency=1.0,
                time_halfbandwidth_product=3,
                time_window_duration=256,
            )
        )
        if measure == "blockwise":
            return connectivity.blockwise_spectral_granger_prediction([0, 1, 1])[0]
        return getattr(connectivity, measure)()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warnings.filterwarnings("ignore", category=DirectedOrientationWarning)
        warnings.filterwarnings("ignore", message="directed_coherence assumes uncorrelated")
        raw, rescaled = to_host(compute(data)), to_host(compute(standardized))
    np.testing.assert_allclose(raw, rescaled, atol=1e-9, rtol=0)


@pytest.mark.parametrize("real_signals", [False, True], ids=["two_sided", "half_spectrum"])
def test_a_factor_that_is_not_minimum_phase_warns_on_both_paths(real_signals, monkeypatch):
    """From the start ``chol(R0)ᵀ``, the exact spectrum of this dense VAR
    of signals of comparable power converges to a factor whose determinant
    winds once around zero, on the two-sided path and on the half-spectrum
    one that real signals take, where the determinant on the negative
    frequencies is mirrored from the others before its winding is counted.
    From the covariant start (``U Uᵀ = R0``) it factors without a
    warning."""
    coefficients, noise_covariance = _dense_stable_var(3, 1, 0.99, seed=857449258)
    spectrum = to_device(_analytic_var(coefficients, noise_covariance, 4096)[2])
    if real_signals:
        spectrum = to_device(_mirrored(to_host(spectrum)))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        minimum_phase_decomposition(spectrum[np.newaxis])
    monkeypatch.setattr(
        mpd_module,
        "_reverse_cholesky",
        lambda matrices: xp.linalg.cholesky(matrices).swapaxes(-1, -2),
    )
    with pytest.warns(UserWarning, match="not minimum phase for 1 of 1 sub-spectrum"):
        factor = minimum_phase_decomposition(spectrum[np.newaxis])
    assert to_host(mpd_module._determinant_winding_number(factor))[0] == pytest.approx(-1)
