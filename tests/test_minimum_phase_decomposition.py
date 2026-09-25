import numpy as np
import pytest
from scipy.fft import fft, ifft
from scipy.signal import freqz_zpk

from spectral_connectivity import minimum_phase_decomposition as mpd_module
from spectral_connectivity.minimum_phase_decomposition import (
    _check_convergence,
    _conjugate_transpose,
    _get_causal_signal,
    _get_initial_conditions,
    _hermitian_square_root,
    _is_conjugate_symmetric,
    _singular_matrix_mask,
    _solve_isolating_singular,
    minimum_phase_decomposition,
    minimum_phase_reconstruction_error,
)


def _analytic_var_spectrum(coefficients, noise_covariance, n_fft):
    """Analytic cross-spectrum S(f) of a VAR on the full FFT grid."""
    n_lags, n_signals, _ = coefficients.shape
    omega = 2 * np.pi * np.arange(n_fft) / n_fft
    A = np.tile(np.eye(n_signals, dtype=complex), (n_fft, 1, 1))
    for lag in range(n_lags):
        A -= coefficients[lag][None] * np.exp(-1j * omega * (lag + 1))[:, None, None]
    H = np.linalg.inv(A)
    return (H @ noise_covariance.astype(complex) @ H.conj().swapaxes(-1, -2))[None]


def test_minimum_phase_reconstruction_error_flags_underresolved_spectrum():
    """The diagnostic is tiny for a resolved spectrum and large when aliased."""
    coefficients = np.array([[[0.9, 0.0], [0.8, 0.9]]])  # (1 lag, 2, 2)
    noise = np.eye(2)

    resolved = _analytic_var_spectrum(coefficients, noise, n_fft=1024)
    coarse = _analytic_var_spectrum(coefficients, noise, n_fft=64)

    resolved_error = float(minimum_phase_reconstruction_error(resolved)[0])
    coarse_error = float(minimum_phase_reconstruction_error(coarse)[0])

    assert resolved_error < 1e-4
    assert coarse_error > 0.2
    assert coarse_error > resolved_error


def test_minimum_phase_reconstruction_error_accepts_precomputed_factor():
    coefficients = np.array([[[0.5, 0.0], [0.4, 0.5]]])
    spectrum = _analytic_var_spectrum(coefficients, np.eye(2), n_fft=512)
    factor = minimum_phase_decomposition(spectrum)
    from_factor = minimum_phase_reconstruction_error(spectrum, factor)
    recomputed = minimum_phase_reconstruction_error(spectrum)
    np.testing.assert_allclose(from_factor, recomputed, rtol=1e-6, atol=1e-10)


def test_minimum_phase_decomposition_non_convergence_warns_and_nans():
    """Only the unconverged sub-spectra are NaN (entirely), with a counted warning.

    Window 0 is a white, uncorrelated spectrum ``diag([2, 0.5])``: its Cholesky
    start is already the exact factor ``diag(sqrt(2), sqrt(0.5))``, so it
    converges in one iteration. Windows 1 and 2 are generic spectra that need
    many iterations. With ``max_iterations=1`` the documented contract is that
    the converged window is returned intact, every entry of each unconverged
    window is NaN, and the warning reports the failed count.
    """
    rng = np.random.default_rng(0)
    n_times, n_freqs, n_signals = 3, 16, 2
    coeffs = rng.standard_normal((n_times, n_freqs, n_signals, n_signals))
    cross_spectral_matrix = np.matmul(coeffs, coeffs.conj().swapaxes(-1, -2))
    cross_spectral_matrix[0] = np.diag([2.0, 0.5])

    with pytest.warns(UserWarning, match="did not converge for 2 of 3"):
        factor = minimum_phase_decomposition(cross_spectral_matrix, max_iterations=1)
    np.testing.assert_allclose(
        factor[0], np.broadcast_to(np.diag([np.sqrt(2.0), np.sqrt(0.5)]), factor[0].shape)
    )
    assert np.isnan(factor[1:]).all()

    # With enough iterations the same input converges cleanly (no warning, no
    # NaN), and window 0's factor is unchanged by the longer run.
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        converged = minimum_phase_decomposition(cross_spectral_matrix, max_iterations=500)
    assert not np.isnan(converged).any()
    np.testing.assert_array_equal(converged[0], factor[0])


def test_solve_isolating_singular_isolates_bad_units():
    """A singular matrix in the batch must not abort the solve for the rest.

    Regression: the batched ``xp.linalg.solve`` inside the Wilson iteration
    raises ``LinAlgError`` if *any* sub-matrix is exactly singular, which
    previously NaN-poisoned the entire batch (and diverged from the GPU path,
    where CuPy returns NaN instead of raising). ``_solve_isolating_singular``
    resolves only the singular unit to NaN and solves the others normally.
    """
    identity = np.eye(2)
    good = np.array([[2.0, 0.0], [0.0, 3.0]])
    singular = np.array([[1.0, 2.0], [2.0, 4.0]])  # rank 1
    rhs = np.eye(2)
    coefficient = np.stack([good, singular, good])
    right_hand_side = np.stack([rhs, rhs, rhs])

    # The plain batched solve raises on the singular unit.
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.solve(coefficient, right_hand_side)

    solved = _solve_isolating_singular(coefficient, right_hand_side, identity)
    assert np.allclose(solved[0], np.linalg.inv(good))
    assert np.isnan(solved[1]).all()
    assert np.allclose(solved[2], np.linalg.inv(good))


def test_hermitian_square_root_factors_psd_and_rejects_invalid_matrices():
    """``L Lᴴ = S`` for positive semidefinite S, including singular S.

    Singular S (a duplicated channel) has no Cholesky factor but does have this
    square root. Non-finite and indefinite matrices resolve to NaN without
    affecting the rest of the batch.
    """
    identity = np.eye(2, dtype=complex)
    positive_definite = np.array([[2.0, 0.5 - 0.3j], [0.5 + 0.3j, 1.0]])
    singular = np.array([[1.0, 1j], [-1j, 1.0]])  # rank 1
    non_finite = np.array([[np.nan, 0.0], [0.0, 1.0]], dtype=complex)
    indefinite = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    matrices = np.stack([positive_definite, singular, non_finite, indefinite])

    square_root = _hermitian_square_root(matrices, identity)

    np.testing.assert_allclose(
        square_root[:2] @ _conjugate_transpose(square_root[:2]), matrices[:2], atol=1e-14
    )
    assert np.isnan(square_root[2:]).all()


def test_near_collinear_channels_converge():
    """Near-duplicate channels converge instead of stalling above the tolerance.

    Regression: forming the update ``G⁻¹ S G⁻ᴴ`` from S directly (two solves or
    an explicit inverse) leaves rounding asymmetry that, for a cross-spectrum
    with condition number around 1e10, keeps the relative change between
    iterates just above the 1e-8 tolerance. Every window then came back NaN with
    a non-convergence warning. Building the update from a square root of S
    converges, and the factor reconstructs S as well as it does for
    well-conditioned channels.
    """
    signals = _lagged_signals(64, np.random.default_rng(0))
    near_duplicate = signals.copy()
    near_duplicate[..., 2] = signals[..., 0] + 3e-5 * signals[..., 2]
    spectrum = _cross_spectrum_of(near_duplicate)

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        factor = minimum_phase_decomposition(spectrum)

    reference_error = minimum_phase_reconstruction_error(_cross_spectrum_of(signals))
    np.testing.assert_array_less(
        minimum_phase_reconstruction_error(spectrum, factor), 2 * reference_error
    )


def test_singular_matrix_mask_flags_singular_and_nonfinite():
    """The mask flags rank-deficient and non-finite matrices, not healthy ones."""
    identity = np.eye(2)
    good = np.array([[2.0, 0.0], [0.0, 3.0]])
    singular = np.array([[1.0, 2.0], [2.0, 4.0]])
    non_finite = np.array([[np.nan, 0.0], [0.0, 1.0]])

    mask = _singular_matrix_mask(np.stack([good, singular, good]), identity)
    assert mask.tolist() == [False, True, False]

    # Non-finite matrices are flagged without the SVD choking on NaN/Inf input.
    mask_nan = _singular_matrix_mask(np.stack([good, non_finite]), identity)
    assert mask_nan.tolist() == [False, True]


def test_minimum_phase_decomposition_isolates_one_singular_subspectrum():
    """One rank-deficient sub-spectrum must not NaN-poison the whole batch.

    Regression: a singular factor in a single time window used to abort the
    Wilson iteration for the entire batch (all sub-spectra returned NaN, with a
    warning implying the whole dataset was rank-deficient). The healthy windows
    must now converge to finite factors; only the bad window is NaN, and the
    warning reports the correct count.
    """

    rng = np.random.default_rng(0)
    n_times, n_freqs, n_signals = 3, 16, 2
    coeffs = rng.standard_normal(
        (n_times, n_freqs, n_signals, n_signals)
    ) + 1j * rng.standard_normal((n_times, n_freqs, n_signals, n_signals))
    cross_spectral_matrix = np.matmul(coeffs, coeffs.conj().swapaxes(-1, -2))
    # Window 1: duplicate a channel so its sub-spectrum is rank-deficient.
    bad = coeffs[1].copy()
    bad[:, 1, :] = bad[:, 0, :]
    cross_spectral_matrix[1] = np.matmul(bad, bad.conj().swapaxes(-1, -2))

    with pytest.warns(UserWarning, match="did not converge for 1 of 3"):
        factor = minimum_phase_decomposition(cross_spectral_matrix, max_iterations=500)
    assert np.isfinite(factor[0]).all()  # healthy window converged
    assert np.isnan(factor[1]).all()  # rank-deficient window isolated
    assert np.isfinite(factor[2]).all()  # healthy window converged


def test_minimum_phase_decomposition_runs_with_debug_logging(caplog):
    """The per-iteration debug log (guarded to avoid a device sync) still works.

    The convergence-count log line is only evaluated when debug logging is
    enabled; exercise that branch so a formatting error in it cannot hide behind
    the default (disabled) log level, and confirm the result is unaffected.
    """
    import logging

    rng = np.random.default_rng(0)
    coeffs = rng.standard_normal((1, 8, 2, 2)) + 1j * rng.standard_normal((1, 8, 2, 2))
    cross_spectral_matrix = np.matmul(coeffs, coeffs.conj().swapaxes(-1, -2)) + 2 * np.eye(2)

    import warnings

    # A clean, well-conditioned input converges, so the loop must take the
    # early-return path -- no "did not converge" warning. This also pins that
    # the merged early-exit returns (rather than breaking into the post-loop
    # NaN-marking/warning path) when every sub-spectrum converges.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        factor_default = minimum_phase_decomposition(cross_spectral_matrix)

    with caplog.at_level(
        logging.DEBUG, logger="spectral_connectivity.minimum_phase_decomposition"
    ):
        factor_debug = minimum_phase_decomposition(cross_spectral_matrix)
    np.testing.assert_array_equal(factor_debug, factor_default)
    assert any("converged" in message for message in caplog.messages)


def test_get_initial_conditions_isolates_non_positive_definite_units():
    """A non-PD sub-spectrum must not change the healthy units' initialization.

    Regression: _get_initial_conditions previously used an all-or-nothing
    fallback, so one rank-deficient window replaced every unit's Cholesky start.
    Only the bad unit should use the deterministic identity fallback; the healthy
    unit keeps its exact Cholesky initialization.
    """
    rng = np.random.default_rng(5)
    n_freq, n_signals = 16, 2
    coeffs = rng.standard_normal((n_freq, n_signals, n_signals)) + 1j * rng.standard_normal(
        (n_freq, n_signals, n_signals)
    )
    healthy = np.matmul(coeffs, coeffs.conj().swapaxes(-1, -2)) + 2 * np.eye(n_signals)
    # Real rank-one spectrum, constant across frequency: its zero-lag matrix is
    # exactly singular, so the batched Cholesky raises.
    v = rng.standard_normal((n_signals, 1))
    rank_one = np.broadcast_to(
        (v @ v.T).astype(complex), (n_freq, n_signals, n_signals)
    ).copy()

    solo = _get_initial_conditions(healthy[np.newaxis])
    with pytest.warns(UserWarning, match="Cholesky failed"):
        batched = _get_initial_conditions(np.stack([healthy, rank_one]))
    # The healthy unit's deterministic Cholesky start is identical whether or not
    # the singular unit shares the batch.
    np.testing.assert_allclose(batched[0], solo[0])
    assert np.isfinite(batched[0]).all()


@pytest.mark.parametrize(
    ("dtype", "small"),
    [
        (np.complex128, 1e-12),  # below eps(float64)-scaled floors
        (np.complex64, 1e-7),  # below eps(float32)-scaled floors, but PD
    ],
)
def test_get_initial_conditions_keeps_valid_ill_conditioned_units(dtype, small):
    """A valid but ill-conditioned unit must keep its Cholesky start in a mixed batch.

    Regression: the non-PD flag used an eigenvalue-ratio / numerical-rank floor
    stricter than Cholesky, so a unit Cholesky factors fine (``diag([1, 1e-12])``
    in float64, ``diag([1, 1e-7])`` in float32) was randomized when batched with a
    truly singular ``diag([1, 0])``. Detection is now per-unit Cholesky, so every
    successfully-factorable unit -- at any dtype -- keeps its exact start.
    """
    ill_conditioned = np.broadcast_to(np.diag([1.0, small]).astype(dtype), (4, 2, 2)).copy()
    # Sanity: this unit really is Cholesky-factorable standalone.
    np.linalg.cholesky(ill_conditioned[0])
    singular = np.broadcast_to(np.diag([1.0, 0.0]).astype(dtype), (4, 2, 2)).copy()

    solo = _get_initial_conditions(ill_conditioned[np.newaxis])
    with pytest.warns(UserWarning, match="Cholesky failed"):
        batched = _get_initial_conditions(np.stack([ill_conditioned, singular]))
    np.testing.assert_allclose(batched[0], solo[0])
    assert np.isfinite(batched[0]).all()
    # The singular-unit fallback must not promote the initialization dtype.
    assert batched.dtype == np.empty(0, dtype=dtype).real.dtype


def test_initial_conditions_fallback_is_deterministic():
    """The non-PD fallback must not depend on the global NumPy random state.

    It previously seeded the failed unit from ``np.random.standard_normal``, so a
    pathological spectrum's initialization depended on unrelated random calls
    (the test suite had to reset the global state). The fallback is now a fixed
    positive-definite start, so the result is identical regardless of global
    state, and the failed unit's start is the Cholesky of ``n_signals * I``.
    """
    n_freq, n_signals = 16, 2
    # A truly singular (rank-one, frequency-constant) unit forces the fallback.
    v = np.array([[1.0], [0.5]])
    singular = np.broadcast_to(
        (v @ v.T).astype(complex), (n_freq, n_signals, n_signals)
    ).copy()

    # The fallback start draws no random numbers, so repeated calls agree.
    with pytest.warns(UserWarning, match="Cholesky failed"):
        first = _get_initial_conditions(singular[np.newaxis])
    with pytest.warns(UserWarning, match="Cholesky failed"):
        second = _get_initial_conditions(singular[np.newaxis])

    np.testing.assert_array_equal(first, second)
    # The fixed start n_signals * I has Cholesky sqrt(n_signals) * I.
    np.testing.assert_allclose(first[0, 0], np.sqrt(n_signals) * np.eye(n_signals))


def test__check_convergence():
    # Realistic shape (n_time, n_fft, n_signals, n_signals); one flag per time.
    # Convergence is relative to the factor magnitude, so use a unit-magnitude
    # baseline and perturb it by a known *relative* amount per sub-spectrum.
    tolerance = 1e-8
    n_time_points, n_fft, n_signals = 5, 4, 3
    current = np.ones((n_time_points, n_fft, n_signals, n_signals))
    old = current.copy()
    old[0] += 1e-9  # relative change 1e-9 -> converged
    old[1] += 1e-7  # relative change 1e-7 -> not converged
    old[3] += 1.0  # large relative change -> not converged
    old[4, :, 1, 1] += 1e-7  # one element exceeds tolerance -> not converged
    # index 2 is unchanged -> converged

    expected_is_converged = np.array([True, False, True, False, False])

    is_converged = _check_convergence(current, old, tolerance)

    assert is_converged.shape == (n_time_points,)
    assert np.all(is_converged == expected_is_converged)


def test__check_convergence_is_scale_invariant():
    """Convergence must not depend on the overall magnitude of the spectrum.

    An absolute-tolerance criterion (falsely) declares convergence for a
    spectrum rescaled to a tiny gain; a relative criterion gives the same
    verdict at every scale.
    """
    rng = np.random.default_rng(0)
    shape = (3, 4, 2, 2)
    current = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    # A ~1e-6 relative perturbation per element.
    old = current * (1 + 1e-6 * rng.standard_normal(shape))

    baseline = _check_convergence(current, old, tolerance=1e-4)
    for scale in (1e-8, 1e-3, 1e3, 1e8):
        scaled = _check_convergence(scale * current, scale * old, tolerance=1e-4)
        assert np.all(scaled == baseline)


def test__check_convergence_tracks_extra_batch_dims():
    """Convergence is per sub-spectrum, not collapsed onto the time axis.

    Regression test: with a retained trial/taper dimension the mask must be
    shape (n_time, n_trials) so one failing sub-spectrum does not mark the
    others at that time point as unconverged.
    """
    tolerance = 1e-8
    n_time, n_trials, n_fft, n_signals = 2, 3, 4, 2
    current = np.zeros((n_time, n_trials, n_fft, n_signals, n_signals))
    old = np.zeros((n_time, n_trials, n_fft, n_signals, n_signals))
    current[0, 1] = 1.0  # only (time=0, trial=1) fails to converge

    is_converged = _check_convergence(current, old, tolerance)

    assert is_converged.shape == (n_time, n_trials)
    expected = np.ones((n_time, n_trials), dtype=bool)
    expected[0, 1] = False
    assert np.all(is_converged == expected)


def test__conjugate_transpose():
    test_array = np.zeros((2, 2, 4), dtype=complex)
    test_array[1, ...] = [
        [1 + 2j, 3 + 4j, 5 + 6j, 7 + 8j],
        [1 - 2j, 3 - 4j, 5 - 6j, 7 - 8j],
    ]
    expected_array = np.zeros((2, 4, 2), dtype=complex)
    expected_array[1, ...] = test_array[1, ...].conj().transpose()
    assert np.allclose(_conjugate_transpose(test_array), expected_array)


def test__get_initial_conditions():
    """The start is the upper-triangular Cholesky factor of the zero-lag matrix.

    Each window's spectrum is ``S(f) = S0 + B e^{-iw} + B^T e^{iw}`` (Hermitian),
    whose inverse-FFT zero lag is exactly the positive-definite ``S0`` with
    non-zero off-diagonals. The returned factor is ``R = cholesky(S0).T``: upper
    triangular with a positive diagonal and ``R^T R = S0``.
    """
    n_time_samples, n_fft_samples, n_signals = 3, 11, 2
    omega = 2 * np.pi * np.arange(n_fft_samples) / n_fft_samples
    lagged = np.array([[0.3, -0.7], [0.2, 0.4]])
    zero_lag = np.stack(
        [np.array([[4.0, 1.2], [1.2, 3.0]]) + time * np.eye(n_signals) for time in range(3)]
    )  # (n_time_samples, n_signals, n_signals), symmetric positive definite
    cross_spectral_matrix = (
        zero_lag[:, np.newaxis]
        + lagged * np.exp(-1j * omega)[:, np.newaxis, np.newaxis]
        + lagged.T * np.exp(1j * omega)[:, np.newaxis, np.newaxis]
    )
    np.testing.assert_allclose(
        cross_spectral_matrix, _conjugate_transpose(cross_spectral_matrix)
    )

    minimum_phase_factor = _get_initial_conditions(cross_spectral_matrix)

    assert minimum_phase_factor.shape == (n_time_samples, 1, n_signals, n_signals)
    expected = np.linalg.cholesky(zero_lag).swapaxes(-1, -2)[:, np.newaxis]
    np.testing.assert_allclose(minimum_phase_factor, expected, rtol=1e-12)
    # Upper triangular with a positive diagonal, and R^T R reproduces S0.
    np.testing.assert_array_equal(minimum_phase_factor[..., 1, 0], 0.0)
    assert np.all(np.diagonal(minimum_phase_factor, axis1=-2, axis2=-1) > 0)
    np.testing.assert_allclose(
        minimum_phase_factor.swapaxes(-1, -2) @ minimum_phase_factor,
        zero_lag[:, np.newaxis],
        rtol=1e-12,
    )


def test__get_causal_signal_removes_roots_outside_unit_circle():
    n_signals = 1
    _, transfer_function = freqz_zpk(4, 2, 1.00, whole=True)
    n_fft_samples = transfer_function.shape[0]
    linear_predictor = np.zeros((1, n_fft_samples, n_signals, n_signals), dtype=complex)
    linear_predictor[0, :, 0, 0] = transfer_function

    expected_causal_signal = np.ones((1, n_fft_samples, n_signals, n_signals), dtype=complex)

    causal_signal = _get_causal_signal(linear_predictor)

    assert np.allclose(causal_signal, expected_causal_signal)


def test__get_causal_signal_preserves_roots_inside_unit_circle():
    n_signals = 1
    _, transfer_function = freqz_zpk(0.25, 0.5, 1.00, whole=True)
    n_fft_samples = transfer_function.shape[0]
    linear_predictor = np.zeros((1, n_fft_samples, n_signals, n_signals), dtype=complex)
    linear_predictor[0, :, 0, 0] = transfer_function

    _, expected_transfer_function = freqz_zpk(0.25, 0.5, 1.00, whole=True)
    linear_coef = ifft(expected_transfer_function)
    linear_coef[0] *= 0.5

    expected_causal_signal = np.zeros((1, n_fft_samples, n_signals, n_signals), dtype=complex)
    expected_causal_signal[0, :, 0, 0] = fft(linear_coef)

    causal_signal = _get_causal_signal(linear_predictor)

    assert np.allclose(causal_signal, expected_causal_signal)


def test_minimum_phase_decomposition():
    n_signals = 1
    # minimum phase is all poles and zeros inside the unit circle
    _, transfer_function = freqz_zpk(0.25, 0.50, 1.00, whole=True)
    n_fft_samples = transfer_function.shape[0]
    expected_minimum_phase_factor = np.zeros(
        (2, n_fft_samples, n_signals, n_signals), dtype=complex
    )
    expected_minimum_phase_factor[0, :, 0, 0] = transfer_function

    _, transfer_function2 = freqz_zpk(0.125, 0.25, 1.00, whole=True)
    expected_minimum_phase_factor[1, :, 0, 0] = transfer_function2

    expected_cross_spectral_matrix = np.matmul(
        expected_minimum_phase_factor,
        _conjugate_transpose(expected_minimum_phase_factor),
    )
    minimum_phase_factor = minimum_phase_decomposition(expected_cross_spectral_matrix)
    cross_spectral_matrix = np.matmul(
        minimum_phase_factor, _conjugate_transpose(minimum_phase_factor)
    )

    assert np.allclose(minimum_phase_factor, expected_minimum_phase_factor)
    assert np.allclose(cross_spectral_matrix, expected_cross_spectral_matrix)


def test_minimum_phase_decomposition_recovers_matrix_factor():
    """A 2x2 minimum-phase MA(1) factor is recovered, not just its product.

    ``G(z) = G0 + G1 z^-1`` with ``G0`` upper triangular with a positive diagonal
    (the normalization Wilson's algorithm converges to) and ``det G(z)`` zero-free
    outside the unit circle, so ``G`` is the unique minimum-phase factor of
    ``S = G G^H``. A transposed or elementwise (non-matmul) factor would fail.
    """
    n_fft = 64
    g0 = np.array([[1.5, 0.4], [0.0, 0.8]])
    g1 = np.array([[0.3, -0.2], [0.5, 0.1]])
    # det(G0 + G1 w) has its roots at |w| ~ 3.04 > 1 (w = z^-1): minimum phase.
    determinant_coefficients = [
        np.linalg.det(g1),
        g0[0, 0] * g1[1, 1] + g1[0, 0] * g0[1, 1] - g0[0, 1] * g1[1, 0] - g1[0, 1] * g0[1, 0],
        np.linalg.det(g0),
    ]
    assert np.all(np.abs(np.roots(determinant_coefficients)) > 1)

    omega = 2 * np.pi * np.arange(n_fft) / n_fft
    expected_factor = (g0 + g1 * np.exp(-1j * omega)[:, np.newaxis, np.newaxis])[np.newaxis]
    cross_spectral_matrix = np.matmul(expected_factor, _conjugate_transpose(expected_factor))

    factor = minimum_phase_decomposition(cross_spectral_matrix)

    np.testing.assert_allclose(factor, expected_factor, atol=1e-6)
    np.testing.assert_allclose(
        np.matmul(factor, _conjugate_transpose(factor)), cross_spectral_matrix, atol=1e-6
    )


@pytest.mark.parametrize("bad_tolerance", [0.0, -1e-8, np.inf, np.nan])
def test_minimum_phase_decomposition_rejects_invalid_tolerance(bad_tolerance):
    """A non-finite or non-positive tolerance must raise, not silently NaN."""
    csm = np.tile(np.eye(2), (4, 1, 1))  # (n_fft, n_signals, n_signals)
    with pytest.raises(ValueError, match="tolerance must be a finite positive"):
        minimum_phase_decomposition(csm, tolerance=bad_tolerance)


@pytest.mark.parametrize("bad_max_iterations", [0, -5, 2.5])
def test_minimum_phase_decomposition_rejects_invalid_max_iterations(bad_max_iterations):
    """A non-positive or non-integer iteration limit must raise."""
    csm = np.tile(np.eye(2), (4, 1, 1))
    with pytest.raises(ValueError, match="max_iterations must be a positive integer"):
        minimum_phase_decomposition(csm, max_iterations=bad_max_iterations)


def test_minimum_phase_decomposition_promotes_complex64_working_precision():
    """The 1e-8 convergence target requires precision above complex64."""
    factor = np.ones((8, 1, 1), dtype=np.complex64)
    csm = factor @ factor.swapaxes(-1, -2).conj()
    result = minimum_phase_decomposition(csm)
    assert result.dtype == np.complex128


def _cross_spectrum_of(signals):
    """Trial-averaged cross-spectrum of ``signals`` shaped (window, trial, time, signal)."""
    coefficients = fft(signals, axis=-2)
    n_trials = signals.shape[1]
    return np.einsum("wtfi,wtfj->wfij", coefficients, coefficients.conj()) / n_trials


def _lagged_signals(n_fft, rng, dtype=float):
    """Three signals where signal 1 follows signal 0 by one sample."""
    shape = (2, 40, n_fft, 3)
    signals = rng.standard_normal(shape)
    if dtype is complex:
        signals = signals + 1j * rng.standard_normal(shape)
    signals[..., 1:, 1] += 0.8 * signals[..., :-1, 0]
    return signals


@pytest.mark.parametrize("n_fft", [16, 17])
def test_is_conjugate_symmetric_detects_real_valued_signals(n_fft):
    """Real signals give S(-f) == conj(S(f)) exactly; complex signals do not."""
    rng = np.random.default_rng(0)
    real_spectrum = _cross_spectrum_of(_lagged_signals(n_fft, rng))
    complex_spectrum = _cross_spectrum_of(_lagged_signals(n_fft, rng, dtype=complex))

    assert _is_conjugate_symmetric(real_spectrum)
    assert not _is_conjugate_symmetric(complex_spectrum)


@pytest.mark.parametrize("n_fft", [16, 17])
def test_real_signal_factorization_matches_full_spectrum_iteration(monkeypatch, n_fft):
    """Iterating on the non-negative frequencies reproduces the full iteration.

    For real-valued signals every Wilson iterate satisfies G(-f) == conj(G(f)),
    so the factorization may run on the non-negative half of the spectrum and
    mirror the rest. Forcing the full two-sided iteration must give the same
    factor up to rounding.
    """
    spectrum = _cross_spectrum_of(_lagged_signals(n_fft, np.random.default_rng(1)))

    half_spectrum_factor = minimum_phase_decomposition(spectrum)
    monkeypatch.setattr(mpd_module, "_is_conjugate_symmetric", lambda _: False)
    full_spectrum_factor = minimum_phase_decomposition(spectrum)

    assert half_spectrum_factor.shape == full_spectrum_factor.shape
    np.testing.assert_allclose(
        half_spectrum_factor,
        full_spectrum_factor,
        rtol=0,
        atol=1e-10 * np.abs(full_spectrum_factor).max(),
    )
    mirrored = half_spectrum_factor[:, (-np.arange(n_fft)) % n_fft]
    np.testing.assert_array_equal(half_spectrum_factor, mirrored.conj())


def _real_signal_spectrum(n_fft=16):
    return _cross_spectrum_of(_lagged_signals(n_fft, np.random.default_rng(0)))


def test_is_conjugate_symmetric_accepts_mirrored_nan():
    """NaN mirrored at the conjugate frequency keeps the fast path; unpaired NaN does not."""
    spectrum = _real_signal_spectrum()
    spectrum[1] = np.nan  # a dead window
    assert _is_conjugate_symmetric(spectrum)

    spectrum = _real_signal_spectrum()
    spectrum[0, 1, 0, 0] = np.nan
    assert not _is_conjugate_symmetric(spectrum)


def test_nan_window_leaves_the_other_windows_unchanged():
    """A NaN window is NaN, and the healthy window matches its own factorization."""
    spectrum = _real_signal_spectrum()
    spectrum[1] = np.nan

    # The NaN window also fails the Cholesky start, which warns separately.
    with (
        pytest.warns(UserWarning, match="did not converge for 1 of 2"),
        pytest.warns(UserWarning, match="Cholesky failed"),
    ):
        factor = minimum_phase_decomposition(spectrum)

    assert factor.shape == spectrum.shape
    assert np.isnan(factor[1]).all()
    np.testing.assert_allclose(
        factor[:1], minimum_phase_decomposition(spectrum[:1]), rtol=1e-12
    )
