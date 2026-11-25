import contextlib
import warnings

import numpy as np
import pytest
from scipy.fft import fft, ifft
from scipy.signal import freqz_zpk

from spectral_connectivity import _array_utils
from spectral_connectivity import minimum_phase_decomposition as mpd_module
from spectral_connectivity._array_utils import _conjugate_transpose
from spectral_connectivity.minimum_phase_decomposition import (
    _check_convergence,
    _get_causal_signal,
    _get_initial_conditions,
    _hermitian_square_root,
    _is_conjugate_symmetric,
    _singular_matrix_mask,
    _solve_isolating_singular,
    _to_two_sided,
    minimum_phase_decomposition,
)
from tests._backend_helpers import ON_GPU, to_device, to_host, xp
from tests._var_oracle import _analytic_var, _dense_stable_var, _mirrored


def _analytic_var_spectrum(coefficients, noise_covariance, n_fft):
    """Analytic cross-spectrum S(f) of a VAR on the full FFT grid, as one window."""
    return _analytic_var(coefficients, noise_covariance, n_fft)[2][None]


def _reconstruction_error(spectrum, factor):
    """``max |G Gᴴ - S| / max |S|`` of each sub-spectrum, on the host."""
    spectrum, factor = to_host(spectrum), to_host(factor)
    residual = factor @ factor.conj().swapaxes(-1, -2) - spectrum
    return np.abs(residual).max(axis=(-3, -2, -1)) / np.abs(spectrum).max(axis=(-3, -2, -1))


def _cross_spectrum_of(signals):
    """Trial-averaged cross-spectrum of ``signals`` shaped (window, trial, time, signal)."""
    coefficients = fft(signals, axis=-2)
    n_trials = signals.shape[1]
    return np.einsum("wtfi,wtfj->wfij", coefficients, coefficients.conj()) / n_trials


def _lagged_signals(n_fft, rng, dtype=float, n_windows=2, n_signals=3):
    """Signals shaped (window, trial, time, signal); signal 1 follows signal 0 by one sample."""
    shape = (n_windows, 40, n_fft, n_signals)
    signals = rng.standard_normal(shape)
    if dtype is complex:
        signals = signals + 1j * rng.standard_normal(shape)
    signals[..., 1:, 1] += 0.8 * signals[..., :-1, 0]
    return signals


def _real_signal_spectrum(n_fft=16, seed=0, n_signals=3):
    """Exactly conjugate-symmetric cross-spectrum (2 windows) of real signals."""
    return _cross_spectrum_of(
        _lagged_signals(n_fft, np.random.default_rng(seed), n_signals=n_signals)
    )


# Spectra of real-valued signals are exactly conjugate-symmetric and take the
# half-spectrum iteration; those of complex-valued signals the two-sided one.
_BOTH_FACTORIZATION_PATHS = pytest.mark.parametrize(
    "real_signals", [False, True], ids=["two_sided", "half_spectrum"]
)


# A VAR(1) whose autocovariance has decayed to 3% by lag 32 and to 1e-12 by
# lag 256. With real coefficients its spectrum is conjugate-symmetric and is
# factored on the non-negative frequencies; the complex rotation makes it not
# so, which takes the two-sided iteration.
_SLOWLY_DECAYING_VAR = np.array([[[0.9, 0.0], [0.8, 0.9]]])  # (1 lag, 2, 2)


@pytest.mark.parametrize("n_fft", [64, 65, 512, 513])
@pytest.mark.parametrize(
    "rotation", [1.0, np.exp(0.7j)], ids=["conjugate-symmetric", "two-sided"]
)
def test_factor_reproduces_the_spectrum_on_even_and_odd_grids(n_fft, rotation):
    """``G Gᴴ`` equals ``S`` at a fixed point of the iteration, whether or not
    the spectrum's autocovariance has decayed within the grid. On an even grid
    the lag ``n_fft / 2`` is its own negative, so half of it is causal, like
    lag 0; dropping it left that lag of the residual unconstrained, and the
    64-point factor here missed its spectrum by a third."""
    spectrum = to_device(
        _analytic_var_spectrum(_SLOWLY_DECAYING_VAR * rotation, np.eye(2), n_fft=n_fft)
    )
    factor = minimum_phase_decomposition(spectrum)
    assert _reconstruction_error(spectrum, factor)[0] < 1e-7


@_BOTH_FACTORIZATION_PATHS
def test_minimum_phase_decomposition_non_convergence_warns_and_nans(real_signals):
    """Only the unconverged sub-spectra are NaN (entirely), with a counted warning.

    Window 0 is a white, uncorrelated spectrum ``diag([2, 0.5])``: its Cholesky
    start is already the exact factor ``diag(sqrt(2), sqrt(0.5))``, so it
    converges in one iteration. Windows 1 and 2 are generic spectra that need
    many iterations. With ``max_iterations=1`` the documented contract is that
    the converged window is returned intact, every entry of each unconverged
    window is NaN, and the warning reports the failed count. Spectra of real
    signals are exactly conjugate-symmetric, so that case runs the
    half-spectrum iteration and must return the full two-sided shape.
    """
    rng = np.random.default_rng(0)
    n_times, n_freqs, n_signals = 3, 16, 2
    signals = _lagged_signals(
        n_freqs,
        rng,
        dtype=float if real_signals else complex,
        n_windows=n_times,
        n_signals=n_signals,
    )
    cross_spectral_matrix = _cross_spectrum_of(signals)
    cross_spectral_matrix[0] = np.diag([2.0, 0.5])
    cross_spectral_matrix = to_device(cross_spectral_matrix)
    assert _is_conjugate_symmetric(cross_spectral_matrix) == real_signals

    with pytest.warns(UserWarning, match="did not converge for 2 of 3"):
        factor = to_host(minimum_phase_decomposition(cross_spectral_matrix, max_iterations=1))
    assert factor.shape == cross_spectral_matrix.shape
    np.testing.assert_allclose(
        factor[0], np.broadcast_to(np.diag([np.sqrt(2.0), np.sqrt(0.5)]), factor[0].shape)
    )
    assert np.isnan(factor[1:]).all()

    # With enough iterations the same input converges cleanly (no warning, no
    # NaN), and window 0's factor is unchanged by the longer run.
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        converged = to_host(
            minimum_phase_decomposition(cross_spectral_matrix, max_iterations=500)
        )
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
    identity = np.eye(3)
    good = np.diag([2.0, 3.0, 4.0])
    # Identical rows produce an exact zero pivot across LAPACK builds. With
    # proportional rows, NumPy 1.24 on macOS can round elimination into tiny
    # nonzero pivots and miss the singularity, bypassing the recovery path.
    singular = np.ones((3, 3))  # rank 1
    rhs = np.eye(3)
    coefficient = np.stack([good, singular, good])
    right_hand_side = np.stack([rhs, rhs, rhs])

    # The plain batched solve raises on the singular unit.
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.solve(coefficient, right_hand_side)

    solved = to_host(
        _solve_isolating_singular(
            to_device(coefficient), to_device(right_hand_side), to_device(identity)
        )
    )
    assert np.allclose(solved[0], np.linalg.inv(good))
    if ON_GPU:  # CuPy's solve does not raise; it returns NaN/Inf for the singular unit.
        assert not np.isfinite(solved[1]).any()
    else:
        assert np.isnan(solved[1]).all()
    assert np.allclose(solved[2], np.linalg.inv(good))


def test_solve_2x2_marks_exactly_singular_and_non_finite_systems_nan():
    """The closed-form 2x2 solve is NaN exactly where ``det(A) == 0`` or ``A``
    is non-finite, without NumPy warnings; a near-singular system is solved."""
    good = np.array([[2.0, 1j], [-1j, 3.0]])
    near_singular = np.array([[1.0, 1.0], [1.0, 1.0 + 1e-12]], dtype=complex)
    matrices = np.stack(
        [
            good,
            np.array([[1.0, 2.0], [2.0, 4.0]], dtype=complex),  # det == 0
            np.array([[np.nan, 0.0], [0.0, 1.0]], dtype=complex),
            np.array([[np.inf, 0.0], [0.0, 1.0]], dtype=complex),
            near_singular,
        ]
    )
    right_hand_side = np.broadcast_to(np.eye(2, dtype=complex), matrices.shape)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        solution = to_host(
            _solve_isolating_singular(
                to_device(matrices),
                to_device(right_hand_side),
                to_device(np.eye(2, dtype=complex)),
            )
        )

    np.testing.assert_allclose(solution[0], np.linalg.inv(good))
    assert np.isnan(solution[1:4]).all()
    np.testing.assert_allclose(solution[4], np.linalg.inv(near_singular), rtol=1e-4)


def test_hermitian_square_root_factors_psd_and_rejects_invalid_matrices():
    """``L Lᴴ = S`` for positive semidefinite S, including singular S.

    Singular S (a duplicated channel) has no Cholesky factor but does have this
    square root. Non-finite and indefinite matrices resolve to NaN without
    affecting the rest of the batch.
    """
    positive_definite = np.array([[2.0, 0.5 - 0.3j], [0.5 + 0.3j, 1.0]])
    singular = np.array([[1.0, 1j], [-1j, 1.0]])  # rank 1
    non_finite = np.array([[np.nan, 0.0], [0.0, 1.0]], dtype=complex)
    indefinite = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    matrices = np.stack([positive_definite, singular, non_finite, indefinite])

    square_root = to_host(_hermitian_square_root(to_device(matrices)))

    np.testing.assert_allclose(
        square_root[:2] @ _conjugate_transpose(square_root[:2]), matrices[:2], atol=1e-14
    )
    assert np.isnan(square_root[2:]).all()


def _tril(matrices):
    """Strictly-lower-triangle indices of ``matrices``' trailing square axes."""
    return xp.tril_indices(matrices.shape[-1], k=-1)


def _chunks_of_three(monkeypatch):
    """Split batched eigh calls into chunks of 3 matrices."""
    monkeypatch.setattr(_array_utils, "_eigh_chunk_size", lambda n_signals: 3)


def test_hermitian_square_root_in_chunks_matches_a_single_batch(monkeypatch):
    """Splitting the batched eigh into chunks gives the single-call result exactly.

    Regression: CuPy's batched eigh fails (CUSOLVER_STATUS_INVALID_VALUE) above
    about 2**21 2x2 matrices (about 1M at 32 signals), which a long recording or
    subset pairwise Granger reaches. The batch is split into chunks; here the budget is shrunk so 20
    matrices (including a non-finite and an indefinite one) take 7 chunks, the
    last one partial.
    """
    rng = np.random.default_rng(0)
    a = rng.standard_normal((4, 5, 2, 2)) + 1j * rng.standard_normal((4, 5, 2, 2))
    matrices = a @ _conjugate_transpose(a)
    matrices[1, 2] = [[np.nan, 0.0], [0.0, 1.0]]
    matrices[3, 4] = [[1.0, 0.0], [0.0, -1.0]]
    single = to_host(_hermitian_square_root(to_device(matrices)))

    _chunks_of_three(monkeypatch)
    chunked = to_host(_hermitian_square_root(to_device(matrices)))

    np.testing.assert_array_equal(chunked, single)
    assert np.isnan(chunked[1, 2]).all()
    assert np.isnan(chunked[3, 4]).all()


def test_minimum_phase_decomposition_with_chunked_eigh_matches(monkeypatch):
    """The Wilson factor is unchanged when the square root of S is chunked."""
    signals = _lagged_signals(64, np.random.default_rng(0))
    spectrum = to_device(_cross_spectrum_of(signals))
    single = to_host(minimum_phase_decomposition(spectrum))

    _chunks_of_three(monkeypatch)
    chunked = to_host(minimum_phase_decomposition(spectrum))

    np.testing.assert_array_equal(chunked, single)


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
    spectrum = to_device(_cross_spectrum_of(near_duplicate))

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        factor = minimum_phase_decomposition(spectrum)

    reference = to_device(_cross_spectrum_of(signals))
    reference_error = _reconstruction_error(reference, minimum_phase_decomposition(reference))
    np.testing.assert_array_less(_reconstruction_error(spectrum, factor), 2 * reference_error)


@pytest.mark.parametrize("n_signals", [2, 3])
def test_solve_isolating_singular_matches_lapack(n_signals):
    """2x2 systems take a closed-form path; both paths match numpy.linalg.solve.

    Includes a singular matrix, which must come back as NaN, alongside
    well-conditioned and ill-conditioned complex ones, which must not.
    """
    rng = np.random.default_rng(3)
    shape = (5, 7, n_signals, n_signals)
    matrices = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    right_hand_side = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    matrices[1, 2] = np.outer(np.arange(1, n_signals + 1), np.ones(n_signals))  # rank 1
    matrices[3, 4] = np.eye(n_signals) + 1e-9 * matrices[3, 4]  # ill-conditioned scaling
    matrices[3, 4, 0] *= 1e-6
    identity = np.eye(n_signals, dtype=complex)

    solution = to_host(
        _solve_isolating_singular(
            to_device(matrices), to_device(right_hand_side), to_device(identity)
        )
    )

    singular = np.zeros(shape[:2], dtype=bool)
    singular[1, 2] = True
    assert np.isnan(solution[singular]).all()
    np.testing.assert_allclose(
        solution[~singular],
        np.linalg.solve(matrices[~singular], right_hand_side[~singular]),
        rtol=1e-8,
        atol=0,
    )


def test_singular_matrix_mask_flags_singular_and_nonfinite():
    """The mask flags rank-deficient and non-finite matrices, not healthy ones."""
    identity = np.eye(2)
    good = np.array([[2.0, 0.0], [0.0, 3.0]])
    singular = np.array([[1.0, 2.0], [2.0, 4.0]])
    non_finite = np.array([[np.nan, 0.0], [0.0, 1.0]])

    mask = _singular_matrix_mask(
        to_device(np.stack([good, singular, good])), to_device(identity)
    )
    assert mask.tolist() == [False, True, False]

    # Non-finite matrices are flagged without the SVD choking on NaN/Inf input.
    mask_nan = _singular_matrix_mask(
        to_device(np.stack([good, non_finite])), to_device(identity)
    )
    assert mask_nan.tolist() == [False, True]


def _spectra_with_one_duplicated_channel(real_signals):
    """Three windows of 2-signal spectra; window 1 duplicates a channel (rank 1)."""
    rng = np.random.default_rng(0)
    n_times, n_freqs, n_signals = 3, 16, 2
    if real_signals:
        signals = _lagged_signals(n_freqs, rng, n_windows=n_times, n_signals=n_signals)
        signals[1, ..., 1] = signals[1, ..., 0]
        return _cross_spectrum_of(signals)
    coeffs = rng.standard_normal(
        (n_times, n_freqs, n_signals, n_signals)
    ) + 1j * rng.standard_normal((n_times, n_freqs, n_signals, n_signals))
    cross_spectral_matrix = np.matmul(coeffs, coeffs.conj().swapaxes(-1, -2))
    bad = coeffs[1].copy()
    bad[:, 1, :] = bad[:, 0, :]
    cross_spectral_matrix[1] = np.matmul(bad, bad.conj().swapaxes(-1, -2))
    return cross_spectral_matrix


@contextlib.contextmanager
def _ignoring_cholesky_start_warning():
    """Ignore the Cholesky-start warning, which depends on the platform's LAPACK.

    Whether the Cholesky start of a rank-deficient or NaN window raises (and so
    warns) differs between LAPACK builds; the tests using this are about the
    iteration. A context manager rather than a ``filterwarnings`` mark, which
    CI's ``-p no:warnings`` job does not register.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Computing the initial conditions using the Cholesky failed",
            category=UserWarning,
        )
        yield


@_BOTH_FACTORIZATION_PATHS
def test_minimum_phase_decomposition_isolates_one_singular_subspectrum(real_signals):
    """One rank-deficient sub-spectrum must not NaN-poison the whole batch.

    Regression: a singular factor in a single time window used to abort the
    Wilson iteration for the entire batch (all sub-spectra returned NaN, with a
    warning implying the whole dataset was rank-deficient). The healthy windows
    must now converge to the factors they get on their own; only the bad window
    is NaN, and the warning reports the correct count. The core the spectral
    Granger measures use returns the same factor without the warning, and
    marks the window unconverged and singular.
    """
    cross_spectral_matrix = to_device(_spectra_with_one_duplicated_channel(real_signals))
    assert _is_conjugate_symmetric(cross_spectral_matrix) == real_signals

    with _ignoring_cholesky_start_warning():
        with pytest.warns(UserWarning, match="did not converge for 1 of 3"):
            factor = to_host(minimum_phase_decomposition(cross_spectral_matrix))
        warnings.simplefilter("error", UserWarning)
        warnings.filterwarnings("ignore", message=mpd_module.CHOLESKY_FALLBACK_WARNING_START)
        factorization = mpd_module._minimum_phase_factor_and_flags(
            cross_spectral_matrix, 1e-8, 500
        )
    silent = to_host(factorization.factor)
    np.testing.assert_array_equal(to_host(factorization.unconverged), [False, True, False])
    np.testing.assert_array_equal(to_host(factorization.singular), [False, True, False])
    assert factor.shape == cross_spectral_matrix.shape
    assert np.isnan(factor[1]).all()  # rank-deficient window isolated
    healthy = to_host(minimum_phase_decomposition(cross_spectral_matrix[[0, 2]]))
    np.testing.assert_allclose(factor[[0, 2]], healthy, rtol=1e-12)
    np.testing.assert_array_equal(silent, factor)


@_BOTH_FACTORIZATION_PATHS
def test_a_window_with_no_power_does_not_hold_the_batch_to_max_iterations(
    real_signals, caplog
):
    """A window where every signal is zero is returned as NaN and reported as
    singular, and the other windows stop when they converge: the batch takes
    the iterations they take on their own.

    Regression: the window's finite all-zero start never converged, so the
    whole batch ran to ``max_iterations``, and the warning blamed the
    iteration budget ("within 500 iterations") rather than the window."""
    import logging

    rng = np.random.default_rng(0)
    live = _cross_spectrum_of(
        _lagged_signals(16, rng, dtype=float if real_signals else complex, n_windows=2)
    )
    spectrum = np.concatenate([live[:1], np.zeros_like(live[:1]), live[1:]])
    assert _is_conjugate_symmetric(to_device(spectrum)) == real_signals

    def factor_and_iterations(cross_spectral_matrix):
        caplog.clear()
        with caplog.at_level(
            logging.DEBUG, logger="spectral_connectivity.minimum_phase_decomposition"
        ):
            factor = minimum_phase_decomposition(to_device(cross_spectral_matrix))
        n_iterations = sum(message.startswith("iteration: ") for message in caplog.messages)
        return to_host(factor), n_iterations

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        live_factor, live_iterations = factor_and_iterations(live)
    with (
        _ignoring_cholesky_start_warning(),
        pytest.warns(UserWarning, match=r"for 1 of 3 .*\(a sub-spectrum became singular"),
    ):
        factor, iterations = factor_and_iterations(spectrum)

    assert live_iterations < 500
    assert iterations == live_iterations
    assert np.isnan(factor[1]).all()
    np.testing.assert_allclose(factor[[0, 2]], live_factor, rtol=1e-12)


def test_minimum_phase_decomposition_runs_with_debug_logging(caplog):
    """The per-iteration debug log (guarded to avoid a device sync) still works.

    The convergence-count log line is only evaluated when debug logging is
    enabled; exercise that branch so a formatting error in it cannot hide behind
    the default (disabled) log level, and confirm the result is unaffected.
    """
    import logging

    rng = np.random.default_rng(0)
    coeffs = rng.standard_normal((1, 8, 2, 2)) + 1j * rng.standard_normal((1, 8, 2, 2))
    cross_spectral_matrix = to_device(
        np.matmul(coeffs, coeffs.conj().swapaxes(-1, -2)) + 2 * np.eye(2)
    )

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
    np.testing.assert_array_equal(to_host(factor_debug), to_host(factor_default))
    assert any("converged" in message for message in caplog.messages)


@contextlib.contextmanager
def _expect_cholesky_fallback_warning():
    """Expect the Cholesky-fallback warning on NumPy, and no such warning on CuPy.

    NumPy's batched Cholesky raises for a non-positive-definite unit, which takes
    the per-unit fallback and warns. CuPy's returns without raising, so there is
    no fallback and no warning (see ``_get_initial_conditions``).
    """
    if not ON_GPU:
        with pytest.warns(UserWarning, match="Cholesky failed"):
            yield
        return
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield
    assert not [w for w in caught if "Cholesky failed" in str(w.message)]


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

    solo = to_host(_get_initial_conditions(to_device(healthy[np.newaxis])))
    with _expect_cholesky_fallback_warning():
        batched = to_host(_get_initial_conditions(to_device(np.stack([healthy, rank_one]))))
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

    solo = to_host(_get_initial_conditions(to_device(ill_conditioned[np.newaxis])))
    with _expect_cholesky_fallback_warning():
        batched = to_host(
            _get_initial_conditions(to_device(np.stack([ill_conditioned, singular])))
        )
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
    with _expect_cholesky_fallback_warning():
        first = to_host(_get_initial_conditions(to_device(singular[np.newaxis])))
    with _expect_cholesky_fallback_warning():
        second = to_host(_get_initial_conditions(to_device(singular[np.newaxis])))

    np.testing.assert_array_equal(first, second)
    if not ON_GPU:  # CuPy's Cholesky does not raise, so it takes no fallback start.
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

    is_converged = to_host(_check_convergence(to_device(current), to_device(old), tolerance))

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

    current, old = to_device(current), to_device(old)
    baseline = to_host(_check_convergence(current, old, tolerance=1e-4))
    for scale in (1e-8, 1e-3, 1e3, 1e8):
        scaled = to_host(_check_convergence(scale * current, scale * old, tolerance=1e-4))
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

    is_converged = to_host(_check_convergence(to_device(current), to_device(old), tolerance))

    assert is_converged.shape == (n_time, n_trials)
    expected = np.ones((n_time, n_trials), dtype=bool)
    expected[0, 1] = False
    assert np.all(is_converged == expected)


def test__get_initial_conditions():
    """The start is the upper-triangular square root ``U`` of the zero-lag
    matrix with ``U U^T = S0``, and the start for ``D S D`` is ``D U``.

    Each window's spectrum is ``S(f) = S0 + B e^{-iw} + B^T e^{iw}`` (Hermitian),
    whose inverse-FFT zero lag is exactly the positive-definite ``S0`` with
    non-zero off-diagonals. ``cholesky(S0).T`` is also upper triangular but has
    ``U^T U = S0``, and its start for ``D S D`` is ``U D``, which is not the
    start of the rescaled factor ``D G``.
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

    minimum_phase_factor = to_host(_get_initial_conditions(to_device(cross_spectral_matrix)))

    assert minimum_phase_factor.shape == (n_time_samples, 1, n_signals, n_signals)
    # Upper triangular with a positive diagonal, and U U^T reproduces S0.
    np.testing.assert_array_equal(minimum_phase_factor[..., 1, 0], 0.0)
    assert np.all(np.diagonal(minimum_phase_factor, axis1=-2, axis2=-1) > 0)
    np.testing.assert_allclose(
        minimum_phase_factor @ minimum_phase_factor.swapaxes(-1, -2),
        zero_lag[:, np.newaxis],
        rtol=1e-12,
    )
    scale = np.array([1e-3, 1e3])
    rescaled = scale[:, np.newaxis] * cross_spectral_matrix * scale[np.newaxis, :]
    np.testing.assert_allclose(
        to_host(_get_initial_conditions(to_device(rescaled))),
        scale[:, np.newaxis] * minimum_phase_factor,
        rtol=1e-12,
    )


def test__get_causal_signal_removes_roots_outside_unit_circle():
    n_signals = 1
    _, transfer_function = freqz_zpk(4, 2, 1.00, whole=True)
    n_fft_samples = transfer_function.shape[0]
    linear_predictor = np.zeros((1, n_fft_samples, n_signals, n_signals), dtype=complex)
    linear_predictor[0, :, 0, 0] = transfer_function

    expected_causal_signal = np.ones((1, n_fft_samples, n_signals, n_signals), dtype=complex)

    causal_signal = to_host(
        _get_causal_signal(
            to_device(linear_predictor), lower_triangular_ind=_tril(linear_predictor)
        )
    )

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

    causal_signal = to_host(
        _get_causal_signal(
            to_device(linear_predictor), lower_triangular_ind=_tril(linear_predictor)
        )
    )

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
    minimum_phase_factor = to_host(
        minimum_phase_decomposition(to_device(expected_cross_spectral_matrix))
    )
    cross_spectral_matrix = np.matmul(
        minimum_phase_factor, _conjugate_transpose(minimum_phase_factor)
    )

    assert np.allclose(minimum_phase_factor, expected_minimum_phase_factor)
    assert np.allclose(cross_spectral_matrix, expected_cross_spectral_matrix)


# A 2x2 minimum-phase MA(1) factor G(z) = G0 + G1 z^-1.
_MA1_COEFFICIENTS = (np.array([[1.5, 0.4], [0.0, 0.8]]), np.array([[0.3, -0.2], [0.5, 0.1]]))


def _minimum_phase_ma1_factor(n_fft):
    """The minimum-phase MA(1) factor on an ``n_fft`` grid, one window."""
    g0, g1 = _MA1_COEFFICIENTS
    omega = 2 * np.pi * np.arange(n_fft) / n_fft
    return (g0 + g1 * np.exp(-1j * omega)[:, np.newaxis, np.newaxis])[np.newaxis]


def test_minimum_phase_decomposition_recovers_matrix_factor():
    """A 2x2 minimum-phase MA(1) factor is recovered, not just its product.

    ``G(z) = G0 + G1 z^-1`` with ``G0`` upper triangular with a positive diagonal
    (the normalization Wilson's algorithm converges to) and ``det G(z)`` zero-free
    outside the unit circle, so ``G`` is the unique minimum-phase factor of
    ``S = G G^H``. A transposed or elementwise (non-matmul) factor would fail.
    """
    g0, g1 = _MA1_COEFFICIENTS
    # det(G0 + G1 w) has its roots at |w| ~ 3.04 > 1 (w = z^-1): minimum phase.
    determinant_coefficients = [
        np.linalg.det(g1),
        g0[0, 0] * g1[1, 1] + g1[0, 0] * g0[1, 1] - g0[0, 1] * g1[1, 0] - g1[0, 1] * g0[1, 0],
        np.linalg.det(g0),
    ]
    assert np.all(np.abs(np.roots(determinant_coefficients)) > 1)

    expected_factor = _minimum_phase_ma1_factor(n_fft=64)
    cross_spectral_matrix = np.matmul(expected_factor, _conjugate_transpose(expected_factor))

    factor = to_host(minimum_phase_decomposition(to_device(cross_spectral_matrix)))

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
    result = minimum_phase_decomposition(to_device(csm))
    assert result.dtype == np.complex128


@pytest.mark.parametrize("n_fft", [16, 17])
def test_is_conjugate_symmetric_detects_real_valued_signals(n_fft):
    """Real signals give S(-f) == conj(S(f)) exactly; complex signals do not."""
    rng = np.random.default_rng(0)
    real_spectrum = _cross_spectrum_of(_lagged_signals(n_fft, rng))
    complex_spectrum = _cross_spectrum_of(_lagged_signals(n_fft, rng, dtype=complex))

    assert _is_conjugate_symmetric(to_device(real_spectrum))
    assert not _is_conjugate_symmetric(to_device(complex_spectrum))


def test_is_conjugate_symmetric_requires_a_real_zero_frequency():
    """A Hermitian but complex zero-frequency matrix is not symmetric."""
    spectrum = _real_signal_spectrum()
    spectrum[..., 0, 0, 1] += 1e-3j
    spectrum[..., 0, 1, 0] -= 1e-3j
    assert not _is_conjugate_symmetric(to_device(spectrum))


def test_is_conjugate_symmetric_requires_conjugated_imaginary_parts():
    """Equal real parts are not enough: the imaginary parts must flip sign.

    Copying S(f) to -f (instead of conj(S(f))) keeps every matrix Hermitian and
    the real parts symmetric, so only the imaginary-part check can reject it.
    """
    spectrum = _real_signal_spectrum()
    spectrum[..., -1, :, :] = spectrum[..., 1, :, :]
    assert np.any(spectrum[..., 1, :, :].imag != 0)
    assert not _is_conjugate_symmetric(to_device(spectrum))


def test_is_conjugate_symmetric_is_exact():
    """A one-ulp change to one negative-frequency real part breaks symmetry."""
    spectrum = _real_signal_spectrum()
    spectrum[..., -1, 0, 0] = np.nextafter(spectrum[..., -1, 0, 0].real, np.inf)
    assert not _is_conjugate_symmetric(to_device(spectrum))


def test_is_conjugate_symmetric_accepts_mirrored_nan():
    """NaN mirrored at the conjugate frequency keeps the fast path; unpaired NaN does not."""
    spectrum = _real_signal_spectrum()
    spectrum[1] = np.nan  # a dead window
    assert _is_conjugate_symmetric(to_device(spectrum))

    spectrum = _real_signal_spectrum()
    spectrum[0, 1, 0, 0] = np.nan
    assert not _is_conjugate_symmetric(to_device(spectrum))


@pytest.mark.parametrize("n_signals", [2, 3])
def test_nan_window_leaves_the_other_windows_unchanged(n_signals):
    """A NaN window is NaN, and the healthy window matches its own factorization.

    Two signals take the closed-form 2x2 solve, which must not emit NumPy
    ``RuntimeWarning`` for the NaN window.
    """
    spectrum = _real_signal_spectrum(n_signals=n_signals)
    spectrum[1] = np.nan
    spectrum = to_device(spectrum)

    with _ignoring_cholesky_start_warning():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.warns(UserWarning, match="did not converge for 1 of 2"):
            factor = to_host(minimum_phase_decomposition(spectrum))

    assert factor.shape == spectrum.shape
    assert np.isnan(factor[1]).all()
    np.testing.assert_allclose(
        factor[:1], to_host(minimum_phase_decomposition(spectrum[:1])), rtol=1e-12
    )


@pytest.mark.parametrize("n_fft", [16, 17])
def test_to_two_sided_restores_the_negative_frequencies(n_fft):
    spectrum = _real_signal_spectrum(n_fft)
    np.testing.assert_array_equal(
        to_host(_to_two_sided(to_device(spectrum[..., : n_fft // 2 + 1, :, :]), n_fft)),
        spectrum,
    )


@pytest.mark.parametrize("n_fft", [16, 17])
def test_get_causal_signal_on_the_half_spectrum_matches_the_two_sided_projection(n_fft):
    """Real FFTs on the non-negative half give the two-sided projection's half."""
    linear_predictor = to_device(_real_signal_spectrum(n_fft))
    n_nonnegative = n_fft // 2 + 1

    tril = _tril(linear_predictor)
    half = to_host(
        _get_causal_signal(
            linear_predictor[..., :n_nonnegative, :, :], n_fft, lower_triangular_ind=tril
        )
    )
    full = to_host(_get_causal_signal(linear_predictor, lower_triangular_ind=tril))

    assert half.shape == (2, n_nonnegative, 3, 3)
    np.testing.assert_allclose(half, full[..., :n_nonnegative, :, :], rtol=0, atol=1e-13)


@pytest.mark.parametrize("n_fft", [16, 17])
def test_real_signal_factorization_matches_full_spectrum_iteration(monkeypatch, n_fft):
    """Iterating on the non-negative frequencies reproduces the full iteration.

    For real-valued signals every Wilson iterate satisfies G(-f) == conj(G(f)),
    so the factorization may run on the non-negative half of the spectrum and
    mirror the rest. Forcing the full two-sided iteration must give the same
    factor up to rounding. Two leading batch axes check that the mirroring and
    per-unit convergence handle extra dimensions.
    """
    spectrum = to_device(
        np.stack([_real_signal_spectrum(n_fft, seed) for seed in (1, 2)], axis=1)
    )  # (window, batch, frequency, signal, signal)

    half_spectrum_factor = to_host(minimum_phase_decomposition(spectrum))
    monkeypatch.setattr(mpd_module, "_is_conjugate_symmetric", lambda _: False)
    full_spectrum_factor = to_host(minimum_phase_decomposition(spectrum))

    assert half_spectrum_factor.shape == full_spectrum_factor.shape
    np.testing.assert_allclose(
        half_spectrum_factor,
        full_spectrum_factor,
        rtol=0,
        atol=1e-13 * np.abs(full_spectrum_factor).max(),
    )
    mirrored = half_spectrum_factor[..., (-np.arange(n_fft)) % n_fft, :, :]
    np.testing.assert_array_equal(half_spectrum_factor, mirrored.conj())


@_BOTH_FACTORIZATION_PATHS
def test_rescaling_the_signals_rescales_the_factor(real_signals):
    """Each signal is scaled to unit mean power before the iteration, so the
    factor of ``D S D`` is ``D G`` to rounding for any positive diagonal ``D``,
    in every window: the iterates do not see the signals' units."""
    signals = _lagged_signals(
        16, np.random.default_rng(0), dtype=float if real_signals else complex
    )
    spectrum = _cross_spectrum_of(signals)
    scale = np.array([[1e-4, 1.0, 1e4], [3.0, 1e-6, 1.0]])  # (window, signal)
    rescaled = (
        scale[:, np.newaxis, :, np.newaxis] * spectrum * scale[:, np.newaxis, np.newaxis, :]
    )
    assert _is_conjugate_symmetric(to_device(rescaled)) == real_signals

    factor = to_host(minimum_phase_decomposition(to_device(spectrum)))
    rescaled_factor = to_host(minimum_phase_decomposition(to_device(rescaled)))

    np.testing.assert_allclose(
        rescaled_factor / scale[:, np.newaxis, :, np.newaxis],
        factor,
        rtol=0,
        atol=1e-12 * np.abs(factor).max(),
    )


def _cholesky_returning_nan(matrices):
    """``_reverse_cholesky`` as CuPy computes it: a matrix that is not
    positive definite gives NaN instead of raising."""
    flipped = to_host(matrices)[..., ::-1, ::-1]
    factor = np.full_like(flipped, np.nan)
    for index in np.ndindex(flipped.shape[:-2]):
        with contextlib.suppress(np.linalg.LinAlgError):
            factor[index] = np.linalg.cholesky(flipped[index])
    return to_device(factor[..., ::-1, ::-1])


@_BOTH_FACTORIZATION_PATHS
@pytest.mark.parametrize("cholesky", ["raises", "returns_nan"])
def test_a_zero_power_signal_is_factored_as_a_rounding_level_white_signal(
    real_signals, cholesky, monkeypatch
):
    """A signal with no power is factored as an independent white signal and
    unscaled by ``eps`` times the weakest other signal's scale. Its row of
    the factor is then ``eps`` of that scale, its column rounding, it
    reconstructs a power ``eps²`` of the weakest signal's, the other signals
    get the factor of their own spectrum whatever their power (here 1e-20),
    and no Cholesky fallback is needed, also where the Cholesky returns NaN
    instead of raising (CuPy).

    Regression: the dead signal kept a scale of 1 while the others were
    scaled to unit power, so its row, of the iteration's absolute residual
    (about 1e-8), reconstructed a power of 3e-17 against 1e-20 for the live
    signals; with a NaN-returning Cholesky its window started from NaN and
    was returned as NaN."""
    signals = _lagged_signals(
        32, np.random.default_rng(2), dtype=float if real_signals else complex, n_signals=3
    )
    spectrum = 1e-20 * _cross_spectrum_of(signals)
    spectrum[0, :, 1, :] = 0
    spectrum[0, :, :, 1] = 0
    assert _is_conjugate_symmetric(to_device(spectrum)) == real_signals
    if cholesky == "returns_nan":
        monkeypatch.setattr(mpd_module, "_reverse_cholesky", _cholesky_returning_nan)
    live = [0, 2]

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        factor = to_host(minimum_phase_decomposition(to_device(spectrum)))
        live_factor = to_host(
            minimum_phase_decomposition(to_device(spectrum[:1][..., live, :][..., live]))
        )[0]

    assert np.isfinite(factor).all()
    weakest_power = spectrum[0, :, live, live].real.mean(axis=-1).min()
    eps = np.finfo(float).eps
    assert np.abs(factor[0, :, 1, :]).max() <= 1.01 * eps * np.sqrt(weakest_power)
    # The other signals' entries in its column are rounding of their own size.
    row_size = np.abs(factor[0][..., live, :]).max(axis=(0, 2))
    assert (np.abs(factor[0][..., live, 1]) < 1e-14 * row_size).all()
    reconstructed = factor[0] @ factor[0].conj().swapaxes(-1, -2)
    assert np.abs(reconstructed[:, 1, 1]).max() <= 1.1 * eps**2 * weakest_power
    np.testing.assert_allclose(
        factor[0][..., live, :][..., live],
        live_factor,
        rtol=0,
        atol=1e-9 * np.abs(live_factor).max(),
    )
    assert _reconstruction_error(spectrum, factor).max() < 1e-7


@_BOTH_FACTORIZATION_PATHS
@pytest.mark.parametrize("cholesky", ["raises", "returns_nan"])
def test_a_window_where_no_signal_has_power_is_nan(real_signals, cholesky, monkeypatch):
    """A window whose spectrum is zero has nothing to factor: it is NaN with
    the non-convergence warning, also where the Cholesky returns NaN instead
    of raising (CuPy), and the other window is factored as on its own.

    Regression: every signal of that window was factored as a white signal,
    so its factor was a rounding-level identity without a warning."""
    signals = _lagged_signals(
        32, np.random.default_rng(2), dtype=float if real_signals else complex, n_signals=3
    )
    spectrum = _cross_spectrum_of(signals)
    spectrum[1] = 0
    assert _is_conjugate_symmetric(to_device(spectrum)) == real_signals
    if cholesky == "returns_nan":
        monkeypatch.setattr(mpd_module, "_reverse_cholesky", _cholesky_returning_nan)
    live_factor = to_host(minimum_phase_decomposition(to_device(spectrum[:1])))

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        factor = to_host(minimum_phase_decomposition(to_device(spectrum)))

    assert any("did not converge for 1 of 2" in str(w.message) for w in record)
    assert np.isnan(factor[1]).all()
    np.testing.assert_allclose(
        factor[:1], live_factor, rtol=0, atol=1e-12 * np.abs(live_factor).max()
    )


def test_a_single_precision_signal_near_its_largest_value_is_scaled_by_its_power():
    """The mean power of a complex64 spectrum is accumulated in float64, so a
    signal whose power sums past float32's largest value (here 1.6e37 at each
    of 64 bins) gets the square root of its power as its scale, not the
    fallback of 1 after an overflow (with a NumPy RuntimeWarning), and its
    factor is that of the same spectrum in complex128."""
    spectrum = _cross_spectrum_of(_lagged_signals(64, np.random.default_rng(0), n_windows=1))
    scale = np.array([5e17, 1.0, 1e-3])
    rescaled = scale[:, np.newaxis] * spectrum * scale[np.newaxis, :]
    single = rescaled.astype(np.complex64)
    power = np.diagonal(single, axis1=-2, axis2=-1).real.mean(axis=-2, dtype=float)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        signal_scale, _, _ = mpd_module._signal_scale(to_device(single))
        factor = to_host(minimum_phase_decomposition(to_device(single)))
    double_factor = to_host(minimum_phase_decomposition(to_device(single.astype(complex))))

    np.testing.assert_allclose(to_host(signal_scale), np.sqrt(power), rtol=1e-12)
    row_size = np.abs(double_factor).max(axis=(-3, -1), keepdims=True)
    np.testing.assert_allclose(factor / row_size, double_factor / row_size, rtol=0, atol=1e-12)


@pytest.mark.parametrize("n_fft", [64, 65])
def test_an_all_pass_rotation_of_the_factor_winds_around_zero(n_fft):
    """``G U`` reproduces ``G Gᴴ`` for any unitary ``U(f)``. The all-pass
    ``U = diag(b, 1)`` with ``b = (z⁻¹ - a) / (1 - a z⁻¹)`` is causal with a
    zero inside the unit circle (in ``z⁻¹``), so the product is causal but
    not minimum phase and its determinant winds once around zero, while the
    determinants of ``G`` and of ``G`` times a constant unitary matrix do not
    wind.

    The anticausal all-pass ``(z - a) / (1 - a z)`` also winds once on the
    grid, but the recount reads a factor as a causal filter wrapped round the
    grid (as the Wilson iteration builds it), which for that one is a
    polynomial of degree ``n - 1`` whose count changes with every finer grid,
    so it is not counted (NaN), as documented."""
    factor = _minimum_phase_ma1_factor(n_fft)
    spectrum = factor @ _conjugate_transpose(factor)
    delay = np.exp(-2j * np.pi * np.arange(n_fft) / n_fft)
    all_pass = np.zeros((2, n_fft, 2, 2), dtype=complex)
    all_pass[0, :, 0, 0] = (delay - 0.5) / (1 - 0.5 * delay)
    all_pass[1, :, 0, 0] = (1 / delay - 0.5) / (1 - 0.5 / delay)
    all_pass[..., 1, 1] = 1
    rotation = np.array([[np.cos(0.3), -np.sin(0.3)], [np.sin(0.3), np.cos(0.3)]])
    factors = np.concatenate([factor, factor @ rotation, factor @ all_pass])
    spectra = np.concatenate([spectrum] * 4)
    np.testing.assert_allclose(factors @ _conjugate_transpose(factors), spectra, atol=1e-12)
    phase = mpd_module._determinant_phase(to_device(factors))

    winding = to_host(mpd_module._determinant_winding_number(to_device(factors)))

    # det(G U) = b det G, and the causal b winds once round 0, clockwise as
    # ω increases; the anticausal one winds once anticlockwise on the grid.
    np.testing.assert_allclose(
        to_host(mpd_module._grid_winding_number(phase)), [0, 0, -1, 1], atol=1e-12
    )
    np.testing.assert_allclose(winding[:3], [0, 0, -1], atol=1e-12)
    # Not counted (NaN), or counted as winding; never a false 0.
    assert np.isnan(winding[3]) or abs(winding[3]) > 0.5


@_BOTH_FACTORIZATION_PATHS
def test_a_winding_aliased_by_a_coarse_grid_is_not_reported(real_signals):
    """Eight independent MA(1) signals with factor ``h = 1 - c z⁻¹``,
    ``|c| = 0.7``, on 16 bins: the factor is diagonal and minimum phase,
    within 0.7^16 (the wrap-around of ``1 / h`` on the grid) of ``h``, but
    ``det G ≈ h⁸`` turns by up to 7.3 radians between bins, each step
    aliased to under ``π / 2``, so the grid counts a winding of -2. A nonzero
    count is recounted on finer grids, interpolating the factor's
    coefficients, where it is 0, so no warning is given. An imaginary ``c``
    makes the spectrum asymmetric, for the two-sided iteration, and keeps
    the count of -2 (``c = 0.7 e^{0.3i}`` turns by 2.07 radians in a step, so
    the grid does not count it and the recount is never reached)."""
    n_fft, n_signals = 16, 8
    coefficient = 0.7 if real_signals else 0.7j
    filter_response = 1 - coefficient * np.exp(-2j * np.pi * np.arange(n_fft) / n_fft)
    power = np.abs(filter_response) ** 2
    if real_signals:  # exactly symmetric, not only to rounding
        power = _mirrored(power)
    diagonal = (slice(None), np.arange(n_signals), np.arange(n_signals))
    spectrum = np.zeros((n_fft, n_signals, n_signals), dtype=complex)
    spectrum[diagonal] = power[:, np.newaxis]
    assert _is_conjugate_symmetric(to_device(spectrum)) == real_signals

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        factor = to_host(minimum_phase_decomposition(to_device(spectrum[np.newaxis])))[0]

    phase = mpd_module._determinant_phase(to_device(factor))
    assert float(mpd_module._grid_winding_number(phase)) == pytest.approx(-2)
    expected = np.zeros_like(spectrum)
    expected[diagonal] = filter_response[:, np.newaxis]
    np.testing.assert_allclose(factor, expected, rtol=0, atol=2 * 0.7**n_fft)
    assert _reconstruction_error(spectrum[np.newaxis], factor[np.newaxis])[0] < 1e-12


def test_the_recount_keeps_the_winding_of_a_causal_factor_with_a_slow_tail():
    """``g = (1 - z₀ z⁻¹) / ((1 - p₁ z⁻¹)(1 - p₂ z⁻¹))`` on 32 bins, with its
    zero ``z₀`` outside the unit circle (``|z₀| = 1.12``): a causal factor
    that is not minimum phase, winding once round zero clockwise. Its pole
    at radius 0.91 leaves a tail of a tenth of its largest lag coefficient at
    lag 16. The recount reads the lags as causal, 0 to ``n - 1``, as the
    Wilson iteration builds them, and keeps the count of -1.

    Regression: the recount interpolated on lags -16 to 15, read that tail
    as negative lags and counted 0, so the factor was not flagged."""
    n_fft = 32
    delay = np.exp(-2j * np.pi * np.arange(n_fft) / n_fft)
    zero, pole_1, pole_2 = 1.112616 + 0.144738j, 0.8534 - 0.3145j, 0.2607 + 0.7289j
    factor = (1 - zero * delay) / ((1 - pole_1 * delay) * (1 - pole_2 * delay))
    factor = factor[np.newaxis, :, np.newaxis, np.newaxis]
    phase = mpd_module._determinant_phase(to_device(factor[0]))

    winding = to_host(mpd_module._determinant_winding_number(to_device(factor)))

    assert float(mpd_module._grid_winding_number(phase)) == pytest.approx(-1)
    assert winding[0] == pytest.approx(-1)


def test_winding_number_of_a_determinant_too_small_to_multiply_is_counted():
    """The phase increments are taken from the unit phase of the determinant,
    so a factor whose determinant is about 8e-189 (64 signals correlated at
    0.999999) still counts the winding of a one-sample delay on one signal,
    where the product of neighbouring determinants underflows to zero."""
    n_fft, n_signals, correlation = 64, 64, 0.999999
    covariance = np.full((n_signals, n_signals), correlation) + (1 - correlation) * np.eye(
        n_signals
    )
    delay = np.exp(-2j * np.pi * np.arange(n_fft) / n_fft)
    factor = np.tile(np.linalg.cholesky(covariance), (n_fft, 1, 1)).astype(complex)
    factor[..., -1] *= delay[:, np.newaxis]
    with np.errstate(all="ignore"):  # NumPy's det and slogdet warn here
        assert abs(np.linalg.det(factor[0])) < 1e-180
        winding = to_host(
            mpd_module._determinant_winding_number(to_device(factor[np.newaxis]))
        )

    assert winding[0] == pytest.approx(-1)


def test_winding_number_is_not_counted_where_the_phase_jumps_between_bins():
    """A determinant whose phase turns by pi / 2 or more between bins cannot
    be counted reliably (an aliased count of 2 came with turns of 1.9-3.1), so
    it is NaN, not tested, rather than a spurious winding, while one that
    turns by pi / 4 per bin is counted and flagged."""
    n_fft = 8
    omega = 2 * np.pi * np.arange(n_fft) / n_fft
    slow = np.exp(1j * omega)[np.newaxis, :, np.newaxis, np.newaxis]
    fast = np.exp(3j * omega)[np.newaxis, :, np.newaxis, np.newaxis]
    winding = to_host(
        mpd_module._determinant_winding_number(to_device(np.concatenate([slow, fast])))
    )
    # ``slow`` is the anticausal advance z (lag -1) and winds once on the
    # grid. The recount reads a factor as a causal filter wrapped round the
    # grid, as the Wilson iteration builds it, so it reads that lag as
    # n - 1 and counts -7; either count flags it, which is what the check is
    # for, so only that is asserted.
    assert np.isfinite(winding[0])
    assert abs(winding[0]) > 0.5
    assert np.isnan(winding[1])


@_BOTH_FACTORIZATION_PATHS
def test_a_factor_that_is_not_minimum_phase_warns_on_both_paths(real_signals, monkeypatch):
    """From the start ``chol(R0)ᵀ``, the exact spectrum of this dense VAR
    of signals of comparable power converges to a factor whose determinant
    winds once around zero, on the two-sided path and on the half-spectrum
    one that real signals take, where the determinant on the negative
    frequencies is mirrored from the others before its winding is counted.
    From the covariant start (``U Uᵀ = R0``) it factors without a
    warning."""
    coefficients, noise_covariance = _dense_stable_var(3, 1, 0.99, seed=857449258)
    spectrum = _analytic_var(coefficients, noise_covariance, 4096)[2]
    spectrum = to_device(_mirrored(spectrum) if real_signals else spectrum)
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


def test_an_infinite_power_window_is_unconverged_without_a_runtime_warning():
    """A window with an infinite power is non-finite: it is NaN with the
    non-convergence warning, and scaling its signals emits no NumPy
    ``RuntimeWarning`` on the way."""
    spectrum = _cross_spectrum_of(
        _lagged_signals(32, np.random.default_rng(2), dtype=complex, n_windows=2)
    )
    spectrum[1, :, 2, :] = 0
    spectrum[1, :, :, 2] = 0
    spectrum[1, :, 2, 2] = np.inf

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.warns(UserWarning, match="did not converge for 1 of 2"):
            factor = to_host(minimum_phase_decomposition(to_device(spectrum)))
    assert np.isnan(factor[1]).all()
    assert np.isfinite(factor[0]).all()
