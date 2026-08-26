import numpy as np
import pytest
from scipy.fft import fft, ifft
from scipy.signal import freqz_zpk

from spectral_connectivity.minimum_phase_decomposition import (
    _check_convergence,
    _conjugate_transpose,
    _get_causal_signal,
    _get_initial_conditions,
    _singular_matrix_mask,
    _solve_isolating_singular,
    minimum_phase_decomposition,
)


def test_minimum_phase_decomposition_non_convergence_warns_and_nans():
    """Unconverged time points return NaN with a warning, not a silent factor."""
    rng = np.random.default_rng(0)
    n_times, n_freqs, n_signals = 3, 16, 2
    coeffs = rng.standard_normal((n_times, n_freqs, n_signals, n_signals))
    cross_spectral_matrix = np.matmul(coeffs, coeffs.conj().swapaxes(-1, -2))

    # One iteration is not enough to converge, forcing the non-convergence path.
    with pytest.warns(UserWarning, match="did not converge"):
        factor = minimum_phase_decomposition(cross_spectral_matrix, max_iterations=1)
    assert np.isnan(factor).any()

    # With enough iterations the same input converges cleanly (no warning, no NaN).
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        converged = minimum_phase_decomposition(
            cross_spectral_matrix, max_iterations=500
        )
    assert not np.isnan(converged).any()


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
    cross_spectral_matrix = np.matmul(
        coeffs, coeffs.conj().swapaxes(-1, -2)
    ) + 2 * np.eye(2)

    factor_default = minimum_phase_decomposition(cross_spectral_matrix)
    with caplog.at_level(
        logging.DEBUG, logger="spectral_connectivity.minimum_phase_decomposition"
    ):
        factor_debug = minimum_phase_decomposition(cross_spectral_matrix)
    np.testing.assert_array_equal(factor_debug, factor_default)
    assert any("converged" in message for message in caplog.messages)


def test_get_initial_conditions_isolates_non_positive_definite_units():
    """A non-PD sub-spectrum must not change the healthy units' initialization.

    Regression: _get_initial_conditions ran one batched Cholesky with an
    all-or-nothing random fallback, so a single rank-deficient window replaced
    EVERY unit's deterministic Cholesky start with a random one -- which can stop
    otherwise-convergent windows from converging. Only the bad unit should fall
    back to random; the healthy unit keeps its exact Cholesky initialization.
    """
    rng = np.random.default_rng(5)
    n_freq, n_signals = 16, 2
    coeffs = rng.standard_normal(
        (n_freq, n_signals, n_signals)
    ) + 1j * rng.standard_normal((n_freq, n_signals, n_signals))
    healthy = np.matmul(coeffs, coeffs.conj().swapaxes(-1, -2)) + 2 * np.eye(n_signals)
    # Real rank-one spectrum, constant across frequency: its zero-lag matrix is
    # exactly singular, so the batched Cholesky raises.
    v = rng.standard_normal((n_signals, 1))
    rank_one = np.broadcast_to(
        (v @ v.T).astype(complex), (n_freq, n_signals, n_signals)
    ).copy()

    solo = _get_initial_conditions(healthy[np.newaxis])
    batched = _get_initial_conditions(np.stack([healthy, rank_one]))
    # The healthy unit's deterministic Cholesky start is identical whether or not
    # the singular unit shares the batch.
    np.testing.assert_allclose(batched[0], solo[0])
    assert np.isfinite(batched[0]).all()


@pytest.mark.parametrize(
    "dtype, small",
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
    import warnings

    ill_conditioned = np.broadcast_to(
        np.diag([1.0, small]).astype(dtype), (4, 2, 2)
    ).copy()
    # Sanity: this unit really is Cholesky-factorable standalone.
    np.linalg.cholesky(ill_conditioned[0])
    singular = np.broadcast_to(np.diag([1.0, 0.0]).astype(dtype), (4, 2, 2)).copy()

    solo = _get_initial_conditions(ill_conditioned[np.newaxis])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the logger.warning is not a UserWarning
        batched = _get_initial_conditions(np.stack([ill_conditioned, singular]))
    np.testing.assert_allclose(batched[0], solo[0])
    assert np.isfinite(batched[0]).all()
    # The singular-unit fallback must not promote the initialization dtype.
    assert batched.dtype == np.empty(0, dtype=dtype).real.dtype


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
    n_time_samples, n_fft_samples, n_signals = 3, 11, 2
    cross_spectral_matrix = (
        np.ones((n_time_samples, n_fft_samples, n_signals, n_signals), dtype=complex)
        * 4
    )
    cross_spectral_matrix[..., 1, 0] = 0
    minimum_phase_factor = _get_initial_conditions(cross_spectral_matrix)
    expected_cross_spectral_matrix = np.zeros(
        (n_time_samples, 1, n_signals, n_signals), dtype=complex
    )
    expected_cross_spectral_matrix[..., :, :] = np.eye(n_signals) * 2
    assert np.allclose(minimum_phase_factor, expected_cross_spectral_matrix)


def test__get_causal_signal_removes_roots_outside_unit_circle():
    n_signals = 1
    _, transfer_function = freqz_zpk(4, 2, 1.00, whole=True)
    n_fft_samples = transfer_function.shape[0]
    linear_predictor = np.zeros((1, n_fft_samples, n_signals, n_signals), dtype=complex)
    linear_predictor[0, :, 0, 0] = transfer_function

    expected_causal_signal = np.ones(
        (1, n_fft_samples, n_signals, n_signals), dtype=complex
    )

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

    expected_causal_signal = np.zeros(
        (1, n_fft_samples, n_signals, n_signals), dtype=complex
    )
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
    cross_spectral_matrix = minimum_phase_factor * _conjugate_transpose(
        minimum_phase_factor
    )

    assert np.allclose(minimum_phase_factor, expected_minimum_phase_factor)
    assert np.allclose(cross_spectral_matrix, expected_cross_spectral_matrix)


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
