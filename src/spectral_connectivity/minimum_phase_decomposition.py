"""Minimum phase decomposition for spectral density matrices.

A spectral density matrix can be decomposed into minimum phase functions
using the Wilson algorithm. This decomposition is used in computing
pairwise spectral Granger prediction and other directed connectivity measures.
"""

import warnings
from logging import DEBUG, getLogger
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from spectral_connectivity.utils import gpu_request_error_message, is_gpu_enabled

# Type-check against the NumPy API, which CuPy mirrors: mypy sees only the CPU
# branch (CuPy is untyped, so importing it would make ``xp`` ``Any``).
if not TYPE_CHECKING and is_gpu_enabled():
    try:
        import cupy as xp
        from cupyx.scipy.fft import fft, ifft, irfft, rfft
    except ImportError as exc:
        raise RuntimeError(gpu_request_error_message()) from exc
else:
    import numpy as xp  # noqa: ICN001 -- the backend-neutral array namespace
    from scipy.fft import fft, ifft, irfft, rfft


logger = getLogger(__name__)


def _conjugate_transpose(x: NDArray[np.complexfloating]) -> NDArray[np.complexfloating]:
    """Compute conjugate transpose of the last two dimensions.

    Parameters
    ----------
    x : NDArray[complexfloating], shape (..., M, N)
        Input array.

    Returns
    -------
    x_H : NDArray[complexfloating], shape (..., N, M)
        Conjugate transpose of last two dimensions.
    """
    return x.swapaxes(-1, -2).conjugate()


def _is_conjugate_symmetric(cross_spectral_matrix: NDArray[np.complexfloating]) -> bool:
    """Whether ``S(-f) == conj(S(f))`` exactly along the frequency axis (-3).

    The comparison is exact (see :func:`minimum_phase_decomposition` Notes). NaN
    counts as symmetric when it is mirrored at the conjugate frequency, so a
    batch with a NaN window (e.g. a dead channel) keeps the fast path; that
    window's factor is NaN on either path.

    Parameters
    ----------
    cross_spectral_matrix : NDArray[complexfloating],
        shape (..., n_fft_samples, n_signals, n_signals)
        Two-sided cross-spectral matrix in standard FFT order.

    Returns
    -------
    bool
    """
    zero_frequency = cross_spectral_matrix[..., 0, :, :]
    positive = cross_spectral_matrix[..., 1:, :, :]
    negative = cross_spectral_matrix[..., :0:-1, :, :]
    mirrored_nan = xp.isnan(positive) & xp.isnan(negative)
    return bool(
        xp.all((zero_frequency.imag == 0) | xp.isnan(zero_frequency))
        and xp.all((positive.real == negative.real) | mirrored_nan)
        and xp.all((positive.imag == -negative.imag) | mirrored_nan)
    )


def _to_two_sided(
    nonnegative: NDArray[np.complexfloating], n_fft_samples: int
) -> NDArray[np.complexfloating]:
    """Restore negative frequencies from a conjugate-symmetric half spectrum.

    Parameters
    ----------
    nonnegative : NDArray[complexfloating],
        shape (..., n_fft_samples // 2 + 1, n_signals, n_signals)
        Frequencies ``0`` through ``n_fft_samples // 2``.
    n_fft_samples : int
        Length of the two-sided frequency axis.

    Returns
    -------
    NDArray[complexfloating], shape (..., n_fft_samples, n_signals, n_signals)
        The spectrum in standard FFT order, with ``X(-f) = conj(X(f))``.
    """
    n_nonnegative = nonnegative.shape[-3]
    negative = nonnegative[..., n_fft_samples - n_nonnegative : 0 : -1, :, :].conjugate()
    return xp.concatenate((nonnegative, negative), axis=-3)


def _get_initial_conditions(
    cross_spectral_matrix: NDArray[np.complexfloating],
    n_fft_samples: int | None = None,
) -> NDArray[np.floating]:
    """Generate initial guess for minimum phase factor using Cholesky decomposition.

    Provides an initial estimate for the Wilson algorithm by taking the Cholesky
    decomposition of the zero-lag cross-spectral matrix (real part of inverse FFT).
    Falls back to a deterministic positive-definite start if Cholesky fails.

    Parameters
    ----------
    cross_spectral_matrix : NDArray[complexfloating],
        shape (n_time_samples, ..., n_frequencies, n_signals, n_signals)
        Cross-spectral density matrix to be decomposed. ``n_frequencies`` is
        ``n_fft_samples`` for a two-sided spectrum, or ``n_fft_samples // 2 + 1``
        when ``n_fft_samples`` is given.
    n_fft_samples : int, optional
        If given, ``cross_spectral_matrix`` holds only the non-negative
        frequencies of a conjugate-symmetric spectrum of this two-sided length.

    Returns
    -------
    minimum_phase_factor : NDArray[floating],
        shape (n_time_samples, ..., 1, n_signals, n_signals)
        Initial guess for minimum phase square root matrix. Real-valued (the
        Cholesky factor of the real zero-lag matrix); the caller promotes it to
        the complex working dtype.

    Notes
    -----
    If the zero-lag matrix of a sub-spectrum is not positive-definite (Cholesky
    fails), only that sub-spectrum falls back to a fixed positive-definite start
    (``n_signals * I``); the healthy sub-spectra keep their Cholesky start. The
    fallback start is now deterministic instead of a random draw, so the result
    for such a pathological sub-spectrum no longer depends on the global NumPy
    random state. This is a NumPy-backend detail: on CuPy the batched Cholesky
    above returns NaN rather than raising, so this branch is never taken.
    """
    if n_fft_samples is None:
        zero_lag = ifft(cross_spectral_matrix, axis=-3)[..., 0:1, :, :].real
    else:
        zero_lag = irfft(cross_spectral_matrix, n=n_fft_samples, axis=-3)[..., 0:1, :, :]
    try:
        return xp.linalg.cholesky(zero_lag).swapaxes(-1, -2)
    except xp.linalg.LinAlgError:
        # One or more sub-spectra are not positive-definite (rank-deficient /
        # duplicated channels). Replace ONLY those with a deterministic PD start
        # so the healthy sub-spectra keep their exact Cholesky initialization,
        # rather than the whole batch falling back (which can stop
        # otherwise-convergent units from converging). This matches the GPU path,
        # where cholesky returns NaN for the bad unit instead of raising.
        # Use warnings.warn (not just logger.warning) so this surfaces under
        # pytest.warns / -W error and reaches the same audience as the GPU path:
        # there, a rank-deficient unit's Cholesky returns NaN and the unit ends
        # up NaN with a loud UserWarning. On CPU the deterministic fallback may
        # instead let the iteration "converge" from a substitute start to a
        # finite value that is not guaranteed correct, so it must be equally
        # visible rather than only logged.
        warnings.warn(
            "Computing the initial conditions using the Cholesky failed for "
            "some sub-spectra (rank-deficient / duplicated channels); using a "
            "deterministic positive-definite fallback start for those units. "
            "Their directed-connectivity values are not guaranteed correct — "
            "check for duplicated or near-collinear channels.",
            UserWarning,
            stacklevel=2,
        )
        logger.warning(
            "Computing the initial conditions using the Cholesky failed for "
            "some sub-spectra; using a deterministic fallback for those."
        )
        # Determine exactly which sub-spectra Cholesky cannot factor by
        # attempting it per unit. Any numerical-rank threshold is only an
        # approximation of potrf's actual pivoting (it depends on dtype and
        # scaling -- e.g. a valid float32 ``diag([1, 1e-7])`` sits below
        # ``n * eps(float32)``), so a per-unit attempt is the only reliable way
        # to preserve every successfully-factorable unit. This runs only on the
        # NumPy backend: CuPy's cholesky returns NaN for the bad unit instead of
        # raising, so the batched call above already isolates it there.
        flat_zero_lag = zero_lag.reshape((-1, *zero_lag.shape[-2:]))
        not_positive_definite = xp.zeros(flat_zero_lag.shape[0], dtype=bool)
        for index in range(flat_zero_lag.shape[0]):
            try:
                xp.linalg.cholesky(flat_zero_lag[index])
            except xp.linalg.LinAlgError:  # noqa: PERF203 -- per-unit on purpose (see above)
                not_positive_definite[index] = True
        # Deterministic well-conditioned PD start for the failed units. The
        # previous code averaged N_RAND=1000 random Wishart draws
        # (mean of R @ Rᴴ), whose expectation is exactly n_signals * I; use that
        # expectation directly as a fixed starting point. This only sets the
        # iteration's initial guess for the pathological units, so their result
        # no longer depends on unrelated global-RNG calls and is reproducible
        # without reseeding (it does not otherwise guarantee a particular
        # converged value for a non-positive-definite input). Built in the
        # zero-lag's own dtype so a float32 spectrum is not promoted to float64;
        # only the failed units are replaced, so the healthy ones keep their
        # exact Cholesky start.
        failed_indices = xp.nonzero(not_positive_definite)[0]
        n_signals = zero_lag.shape[-1]
        deterministic_start = n_signals * xp.eye(n_signals, dtype=zero_lag.dtype)
        safe_flat = flat_zero_lag.copy()
        safe_flat[failed_indices] = deterministic_start
        return xp.linalg.cholesky(safe_flat.reshape(zero_lag.shape)).swapaxes(-1, -2)


def _get_causal_signal(
    linear_predictor: NDArray[np.complexfloating],
    n_fft_samples: int | None = None,
) -> NDArray[np.complexfloating]:
    """Extract causal part of linear predictor (plus operator).

    Implements the "plus" operator from the Wilson algorithm by:
    1. Taking half the roots on the unit circle (zero lag)
    2. Taking all roots inside the unit circle (positive lags)
    3. Making zero-lag term upper triangular

    This gives A_(t+1)(Z) / A_(t)(Z) in the Wilson algorithm notation.

    Parameters
    ----------
    linear_predictor : NDArray[complexfloating],
        shape (..., n_frequencies, n_signals, n_signals)
        Linear predictor matrix in frequency domain. ``n_frequencies`` is
        ``n_fft_samples`` for a two-sided spectrum, or ``n_fft_samples // 2 + 1``
        when ``n_fft_samples`` is given.
    n_fft_samples : int, optional
        If given, ``linear_predictor`` holds only the non-negative frequencies of
        a conjugate-symmetric spectrum of this two-sided length. Its lag
        coefficients are then real, so real FFTs replace the complex ones and
        the result is again the non-negative half.

    Returns
    -------
    causal_part_of_linear_predictor : NDArray[complexfloating],
        shape (..., n_frequencies, n_signals, n_signals)
        Causal part of the linear predictor after plus operator, on the same
        frequencies as ``linear_predictor``.

    Notes
    -----
    The plus operator is a key component of the Wilson algorithm for
    minimum phase decomposition. It ensures causality by zeroing out
    negative lag components and enforcing upper triangular structure
    at zero lag.
    """
    n_signals = linear_predictor.shape[-1]
    if n_fft_samples is None:
        n_lags = linear_predictor.shape[-3]
        linear_predictor_coefficients = ifft(linear_predictor, axis=-3)
    else:
        n_lags = n_fft_samples
        linear_predictor_coefficients = irfft(linear_predictor, n=n_fft_samples, axis=-3)

    # Take half of the roots on the unit circle
    linear_predictor_coefficients[..., 0, :, :] *= 0.5

    # Make the unit circle roots upper triangular. Use xp (not np) so the
    # index arrays match the array backend (mixing a NumPy index array with a
    # CuPy array is a GPU-only footgun).
    lower_triangular_ind = xp.tril_indices(n_signals, k=-1)
    linear_predictor_coefficients[..., 0, lower_triangular_ind[0], lower_triangular_ind[1]] = 0

    # Take only the roots inside the unit circle (positive lags)
    linear_predictor_coefficients[..., (n_lags + 1) // 2 :, :, :] = 0
    if n_fft_samples is not None:
        return rfft(linear_predictor_coefficients, axis=-3)
    return fft(linear_predictor_coefficients, axis=-3)


def _check_convergence(
    current: NDArray[np.complexfloating],
    old: NDArray[np.complexfloating],
    tolerance: float = 1e-8,
) -> NDArray[np.bool_]:
    """Check Wilson-algorithm convergence for each independent sub-spectrum.

    Each Wilson factorization couples all frequencies of one sub-spectrum (the
    causal projection uses an ifft/fft over the frequency axis), so the unit of
    convergence is a single sub-spectrum: everything but the frequency axis (-3)
    and the two signal axes (-2, -1). The maximum absolute difference (infinity
    norm) is taken over those last three axes, leaving one convergence flag per
    leading batch element. This matters for expectation modes that retain a
    trial or taper dimension in addition to time.

    Parameters
    ----------
    current : NDArray[complexfloating],
        shape (..., n_fft_samples, n_signals, n_signals)
        Current iteration's minimum phase factor estimates.
    old : NDArray[complexfloating], same shape
        Previous iteration's minimum phase factor estimates.
    tolerance : float, default=1e-8
        Relative convergence tolerance. Sub-spectra whose maximum successive
        change, normalized by the factor magnitude, is below this value are
        considered converged. Normalizing makes the criterion scale-invariant:
        an absolute threshold would falsely declare convergence for a spectrum
        rescaled to a tiny gain.

    Returns
    -------
    is_converged : NDArray[bool], shape (...,)
        Convergence status per leading batch element (``current.shape[:-3]``).

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> shape = (10, 8, 5, 5)
    >>> current = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    >>> old = current + 1e-10 * rng.standard_normal(shape)
    >>> converged = _check_convergence(current, old, tolerance=1e-8)
    >>> converged.shape
    (10,)
    """
    batch_shape = current.shape[:-3]
    error: NDArray[np.floating] = xp.max(
        xp.abs(current - old).reshape(*batch_shape, -1), axis=-1
    )
    # Normalize by the factor magnitude so the criterion is scale-invariant.
    # Floor the scale to avoid dividing by zero for an all-zero sub-spectrum
    # (there error is also zero, so the block is trivially converged).
    scale: NDArray[np.floating] = xp.max(xp.abs(current).reshape(*batch_shape, -1), axis=-1)
    scale = xp.maximum(scale, xp.finfo(scale.dtype).tiny)
    return error / scale < tolerance


def _all_finite_units(
    factor: NDArray[np.complexfloating],
    batch_shape: tuple[int, ...],
) -> NDArray[np.bool_]:
    """Return a per-sub-spectrum mask of which units are entirely finite.

    Parameters
    ----------
    factor : NDArray[complexfloating], shape (..., n_fft, n_signals, n_signals)
        Minimum phase factor estimate.
    batch_shape : tuple of int
        Leading batch dimensions (``factor.shape[:-3]``); one flag per unit.

    Returns
    -------
    NDArray[bool], shape ``batch_shape``
        True where every element of that sub-spectrum is finite.
    """
    return xp.isfinite(factor).reshape(*batch_shape, -1).all(axis=-1)


def minimum_phase_reconstruction_error(
    cross_spectral_matrix: NDArray[np.complexfloating],
    minimum_phase_factor: NDArray[np.complexfloating] | None = None,
    *,
    tolerance: float = 1e-8,
    max_iterations: int = 500,
) -> NDArray[np.floating]:
    """Relative reconstruction error of a Wilson factorization, per sub-spectrum.

    The Wilson iteration's convergence test only measures the change between
    successive iterates; a stable iterate does not guarantee ``G Gᴴ ≈ S``. When
    the cross-spectrum is under-resolved in frequency (its autocovariance has not
    decayed within the window, so the periodic factorization aliases), the
    iteration can "converge" to a factor that reconstructs ``S`` poorly, silently
    biasing every directed-connectivity measure built on it (spectral Granger,
    DTF, PDC). This is an opt-in diagnostic: it returns
    ``max |G Gᴴ - S| / max |S|`` for each sub-spectrum -- the entrywise
    maximum absolute residual over every frequency and matrix entry, relative
    to the entrywise maximum magnitude of ``S`` -- so callers can check
    factorization quality explicitly.

    ``cross_spectral_matrix`` must be a full two-sided spectrum in standard
    FFT order, as required by the factorization itself. This module-level
    function cannot verify that from the array alone (a one-sided spectrum has
    the same shape); the :meth:`Connectivity.minimum_phase_reconstruction_error
    <spectral_connectivity.Connectivity.minimum_phase_reconstruction_error>`
    method enforces it from the transform's declared sidedness.

    A relative error near machine precision indicates a faithful factorization;
    values of a few percent are typical for finite-resolution estimated spectra;
    tens of percent or more indicate the spectrum is too coarsely resolved to
    trust the directed measures -- use a longer FFT (larger ``n_fft_samples`` /
    ``n_time_samples_per_window``). The error is deliberately *not* raised as a
    warning during factorization: it does not cleanly separate an under-resolved
    spectrum from a merely short or noisy one, so an always-on threshold would
    either cry wolf on ordinary short-window analyses or miss real problems.

    Parameters
    ----------
    cross_spectral_matrix : NDArray[complexfloating],
        shape (..., n_fft_samples, n_signals, n_signals)
        The two-sided cross-spectral matrix (standard FFT order, positive and
        negative frequencies) that was (or will be) factored.
    minimum_phase_factor : NDArray[complexfloating], optional
        A precomputed factor from :func:`minimum_phase_decomposition`. If omitted,
        the factorization is computed here with ``tolerance`` / ``max_iterations``.
    tolerance, max_iterations
        Passed to :func:`minimum_phase_decomposition` when it must be computed.

    Returns
    -------
    relative_error : NDArray[floating], shape (...,)
        Entrywise max-abs relative reconstruction error per sub-spectrum (the
        leading batch dimensions ``cross_spectral_matrix.shape[:-3]``). ``NaN``
        where the factor is non-finite (the factorization did not converge).
    """
    if minimum_phase_factor is None:
        minimum_phase_factor = minimum_phase_decomposition(
            cross_spectral_matrix, tolerance=tolerance, max_iterations=max_iterations
        )
    batch_shape = cross_spectral_matrix.shape[:-3]
    reconstructed = xp.matmul(minimum_phase_factor, _conjugate_transpose(minimum_phase_factor))
    reference = cross_spectral_matrix.astype(reconstructed.dtype, copy=False)
    residual: NDArray[np.floating] = xp.max(
        xp.abs(reconstructed - reference).reshape(*batch_shape, -1), axis=-1
    )
    scale: NDArray[np.floating] = xp.max(xp.abs(reference).reshape(*batch_shape, -1), axis=-1)
    scale = xp.maximum(scale, xp.finfo(scale.dtype).tiny)
    return residual / scale


def _singular_matrix_mask(
    matrices: NDArray[np.complexfloating],
    identity_matrix: NDArray[np.complexfloating],
) -> NDArray[np.bool_]:
    """Flag which matrices in a batched stack are singular or non-finite.

    A sub-matrix is flagged when it is not finite, or when its smallest singular
    value is at/below the numerical rank tolerance ``max_sv * n_signals * eps``.
    Non-finite matrices are temporarily replaced by the identity before the SVD
    (which does not converge on NaN/Inf input) and flagged directly instead.

    Parameters
    ----------
    matrices : NDArray[complexfloating], shape (..., n_signals, n_signals)
        Batched matrix stack to test.
    identity_matrix : NDArray[complexfloating], shape (n_signals, n_signals)
        Identity used as a stand-in for non-finite matrices.

    Returns
    -------
    NDArray[bool], shape (...,)
        True for each matrix that is singular or non-finite.
    """
    n_signals = matrices.shape[-1]
    is_finite: NDArray[np.bool_] = xp.isfinite(matrices).all(axis=(-2, -1))
    cleaned = xp.where(is_finite[..., xp.newaxis, xp.newaxis], matrices, identity_matrix)
    singular_values: NDArray[np.floating] = xp.linalg.svd(cleaned, compute_uv=False)
    largest = singular_values[..., 0]
    smallest = singular_values[..., -1]
    tolerance: NDArray[np.floating] = largest * n_signals * xp.finfo(singular_values.dtype).eps
    return (~is_finite) | (smallest <= tolerance)


def _solve_isolating_singular(
    coefficient_matrix: NDArray[np.complexfloating],
    right_hand_side: NDArray[np.complexfloating],
    identity_matrix: NDArray[np.complexfloating],
) -> NDArray[np.complexfloating]:
    """Batched solve that isolates singular sub-matrices as NaN.

    NumPy's ``linalg.solve`` raises ``LinAlgError`` if *any* matrix in the
    batched stack is exactly singular, which would otherwise abort the whole
    Wilson iteration for every sub-spectrum sharing the batch. CuPy instead
    returns NaN/Inf for the offending matrices and solves the rest. This helper
    gives the NumPy path the same behavior: singular (or already non-finite)
    sub-matrices resolve to NaN while the remaining ones are solved normally, so
    a single rank-deficient window (e.g. duplicated channels) does not poison
    the entire batch, and exactly singular sub-matrices resolve to NaN on both
    backends.

    Parameters
    ----------
    coefficient_matrix : NDArray[complexfloating], shape (..., n_signals, n_signals)
        Batched left-hand-side matrices ``A`` in ``A x = B``.
    right_hand_side : NDArray[complexfloating], shape (..., n_signals, n_signals)
        Batched right-hand sides ``B``.
    identity_matrix : NDArray[complexfloating], shape (n_signals, n_signals)
        Identity used to stand in for singular matrices during the solve.

    Returns
    -------
    NDArray[complexfloating], same shape as ``right_hand_side``
        Solution ``x``, with NaN for singular/non-finite ``A``.
    """
    try:
        return xp.linalg.solve(coefficient_matrix, right_hand_side)
    except xp.linalg.LinAlgError:
        singular = _singular_matrix_mask(coefficient_matrix, identity_matrix)
        broadcast = singular[..., xp.newaxis, xp.newaxis]
        safe_matrix = xp.where(broadcast, identity_matrix, coefficient_matrix)
        solved = xp.linalg.solve(safe_matrix, right_hand_side)
        return xp.where(broadcast, xp.nan, solved)


def _hermitian_square_root(
    matrices: NDArray[np.complexfloating],
) -> NDArray[np.complexfloating]:
    """Batched square root ``L`` with ``L Lᴴ = S`` of Hermitian PSD matrices.

    Built from the eigendecomposition ``S = V Λ Vᴴ`` as ``L = V Λ^(1/2)``, which
    exists for every positive semidefinite ``S``, including singular ones
    (duplicated channels), where a Cholesky factorization fails. Eigenvalues
    that are negative only by rounding (within ``n_signals * eps`` of the
    largest magnitude) are clipped to zero. A matrix with a larger negative
    eigenvalue is not positive semidefinite, and a non-finite matrix has no
    eigendecomposition; both resolve to NaN so the Wilson iteration marks that
    sub-spectrum as failed instead of silently factoring a different matrix.

    Parameters
    ----------
    matrices : NDArray[complexfloating], shape (..., n_signals, n_signals)
        Batched Hermitian matrices ``S``. Only the lower triangle is read.

    Returns
    -------
    NDArray[complexfloating], same shape as ``matrices``
        Square roots ``L``, with NaN for non-finite or indefinite ``S``.
    """
    n_signals = matrices.shape[-1]
    is_finite = xp.isfinite(matrices).all(axis=(-2, -1), keepdims=True)
    eigenvalues, eigenvectors = xp.linalg.eigh(xp.where(is_finite, matrices, 0))
    rounding = (
        n_signals
        * xp.finfo(eigenvalues.dtype).eps
        * xp.max(xp.abs(eigenvalues), axis=-1, keepdims=True)
    )
    is_valid = is_finite & (eigenvalues >= -rounding).all(axis=-1)[..., xp.newaxis, xp.newaxis]
    square_root = eigenvectors * xp.sqrt(xp.maximum(eigenvalues, 0))[..., xp.newaxis, :]
    return xp.where(is_valid, square_root, xp.nan)


def _get_linear_predictor(
    minimum_phase_factor: NDArray[np.complexfloating],
    cross_spectral_square_root: NDArray[np.complexfloating],
    identity_matrix: NDArray[np.complexfloating],
) -> NDArray[np.complexfloating]:
    """Compute linear predictor for Wilson algorithm update step.

    Calculates how much to adjust the current minimum phase factor guess
    as G^{-1} S G^{-H} + I, where G is the current guess, S is the
    cross-spectral matrix, and H denotes conjugate transpose.

    Parameters
    ----------
    minimum_phase_factor : NDArray[complexfloating],
        shape (n_time_samples, ..., n_frequencies, n_signals, n_signals)
        Current minimum phase square root estimate G.
    cross_spectral_square_root : NDArray[complexfloating],
        shape (n_time_samples, ..., n_frequencies, n_signals, n_signals)
        Square root ``L`` of the target cross-spectral matrix, ``S = L Lᴴ``
        (see :func:`_hermitian_square_root`).
    identity_matrix : NDArray[complexfloating], shape (n_signals, n_signals)
        Identity matrix.

    Returns
    -------
    linear_predictor : NDArray[complexfloating],
        shape (n_time_samples, ..., n_frequencies, n_signals, n_signals)
        Adjustment matrix for updating minimum phase factor estimate.

    Notes
    -----
    This implements the core update step of the Wilson algorithm:
    computing the "covariance sandwich estimator" that measures the
    discrepancy between the current factorization and target matrix.
    It is formed as ``X Xᴴ`` with ``X = G⁻¹ L``: one solve per matrix, and the
    result is Hermitian positive semidefinite by construction. Applying ``G⁻¹``
    to ``S`` on both sides (two solves, or an explicit inverse) leaves rounding
    asymmetry in the update, which for near-collinear channels keeps the
    iteration from reaching the convergence tolerance.
    """
    whitened_square_root = _solve_isolating_singular(
        minimum_phase_factor, cross_spectral_square_root, identity_matrix
    )
    covariance_sandwich_estimator: NDArray[np.complexfloating] = xp.matmul(
        whitened_square_root, _conjugate_transpose(whitened_square_root)
    )
    covariance_sandwich_estimator += identity_matrix
    return covariance_sandwich_estimator


def minimum_phase_decomposition(
    cross_spectral_matrix: NDArray[np.complexfloating],
    tolerance: float = 1e-8,
    max_iterations: int = 500,
    *,
    _warn_on_failure: bool = True,
) -> NDArray[np.complexfloating]:
    """Compute minimum phase decomposition using Wilson algorithm.

    Finds a minimum phase matrix square root G of the cross-spectral density
    matrix S such that S = G G^H, where all poles of G are inside the unit
    circle. This decomposition is essential for computing directed connectivity
    measures like spectral Granger causality.

    Parameters
    ----------
    cross_spectral_matrix : NDArray[complexfloating],
        shape (n_time_samples, ..., n_fft_samples, n_signals, n_signals)
        Cross-spectral density matrix to be decomposed. Must be Hermitian
        positive semidefinite for each frequency.
    tolerance : float, default=1e-8
        Relative convergence tolerance for Wilson algorithm iterations.
    max_iterations : int, default=500
        Maximum number of iterations before stopping algorithm. Near-singular
        cross-spectral matrices (highly correlated channels) can need several
        hundred iterations to reach the relative tolerance; the loop returns
        early as soon as every sub-spectrum has converged.

    Returns
    -------
    minimum_phase_factor : NDArray[complexfloating],
        shape (n_time_samples, ..., n_fft_samples, n_signals, n_signals)
        Minimum phase square root of cross_spectral_matrix. All eigenvalues
        have negative real parts (minimum phase property).

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> n_times, n_freqs, n_signals = 1, 32, 2
    >>> # A valid cross-spectral matrix (Hermitian positive definite). Here it is
    >>> # constant across frequency (a white process), which factors exactly.
    >>> a = rng.standard_normal((n_signals, n_signals))
    >>> spd = a @ a.T + n_signals * np.eye(n_signals)
    >>> cross_spec = np.tile(spd, (n_times, n_freqs, 1, 1))
    >>> min_phase = minimum_phase_decomposition(cross_spec)
    >>> # The factor reconstructs the input: G @ G^H == S.
    >>> reconstructed = np.matmul(min_phase, min_phase.conj().swapaxes(-1, -2))
    >>> error = np.abs(reconstructed - cross_spec).max()
    >>> bool(error < 1e-6)
    True

    Notes
    -----
    The Wilson algorithm iteratively refines an initial guess using the
    "plus" operator (causal projection) until convergence. The algorithm
    may not converge for all time points; warnings are issued when the
    maximum iteration count is reached.

    When the spectrum is exactly conjugate-symmetric, ``S(-f) == conj(S(f))``
    bit for bit (as SciPy's FFT gives for real-valued signals), every iterate
    keeps that symmetry, so the iteration runs on the non-negative frequencies
    with real FFTs and mirrors the rest. Spectra symmetric only up to rounding
    (possible with other FFT backends such as CuPy's), or not at all
    (complex-valued signals), take the two-sided iteration, which does about
    twice the arithmetic per iteration.

    Each update is built from a square root ``L`` of ``S`` (``S = L Lᴴ``),
    computed once, which lets sub-spectra with near-collinear channels converge.
    An indefinite or non-finite ``S`` has no such square root and is returned as
    NaN with the non-convergence warning.

    Convergence of the iterate does not by itself guarantee ``G Gᴴ ≈ S``. The
    factorization assumes the cross-spectrum is resolved finely enough in
    frequency that the corresponding autocovariance decays within the analysis
    window; an under-resolved (aliased) spectrum can satisfy the successive-
    iterate convergence test yet reconstruct ``S`` poorly, silently biasing the
    directed-connectivity measures built on it. Use
    :func:`minimum_phase_reconstruction_error` to check factorization quality
    explicitly, and a longer FFT (larger ``n_fft_samples`` /
    ``n_time_samples_per_window``) if the error is large.

    References
    ----------
    .. [1] Wilson, G. T. (1972). The factorization of matricial spectral
           densities. SIAM Journal on Applied Mathematics, 23(4), 420-426.
    .. [2] Dhamala, M., Rangarajan, G., & Ding, M. (2008). Analyzing
           information flow in brain networks with nonparametric Granger
           causality. NeuroImage, 41(2), 354-362.
    """
    # _warn_on_failure=False is private to the spectral Granger measures, which
    # report the NaN signal pairs themselves instead of this pair-less warning.
    if not np.isfinite(tolerance) or tolerance <= 0:
        msg = f"tolerance must be a finite positive number, got {tolerance}."
        raise ValueError(msg)
    if (
        not isinstance(max_iterations, (int, np.integer))  # type: ignore[redundant-expr]  # user input
        or max_iterations < 1
    ):
        msg = f"max_iterations must be a positive integer, got {max_iterations}."
        raise ValueError(msg)
    n_signals = cross_spectral_matrix.shape[-1]
    # Wilson's default relative tolerance (1e-8) is below float32 epsilon. A
    # complex64 iteration therefore stalls at its rounding floor and otherwise
    # marks valid sub-spectra as unconverged/NaN. Directed-connectivity accuracy
    # takes precedence over retaining the input storage dtype: perform the
    # factorization at complex128 or better, matching the historical behavior.
    working_dtype = xp.result_type(cross_spectral_matrix.dtype, xp.complex128)
    # The iterates of an exactly conjugate-symmetric spectrum stay symmetric (the
    # Cholesky start is real, and so are the linear predictor's lag
    # coefficients), so iterate on the non-negative frequencies and mirror the
    # rest at the end. Slice before the upcast so the unused negative
    # frequencies are never copied.
    n_fft_samples = cross_spectral_matrix.shape[-3]
    half_spectrum_n_fft: int | None = None
    working_cross_spectral_matrix = cross_spectral_matrix
    if _is_conjugate_symmetric(cross_spectral_matrix):
        half_spectrum_n_fft = n_fft_samples
        working_cross_spectral_matrix = cross_spectral_matrix[
            ..., : n_fft_samples // 2 + 1, :, :
        ]
    working_cross_spectral_matrix = working_cross_spectral_matrix.astype(
        working_dtype, copy=False
    )
    identity_matrix = xp.eye(n_signals, dtype=working_dtype)
    # S is fixed across iterations, so factor it once as S = L Lᴴ.
    cross_spectral_square_root = _hermitian_square_root(working_cross_spectral_matrix)
    # One convergence flag per independent sub-spectrum (all leading batch dims
    # except the frequency and signal axes), so that a sub-spectrum failing to
    # converge does not mask the others sharing its time point.
    batch_shape = cross_spectral_matrix.shape[:-3]
    n_units = int(np.prod(batch_shape))  # np.prod(()) == 1 for a single unit
    is_converged = xp.zeros(batch_shape, dtype=bool)
    initial = _get_initial_conditions(
        working_cross_spectral_matrix, half_spectrum_n_fft
    ).astype(working_dtype, copy=False)
    minimum_phase_factor = xp.broadcast_to(initial, working_cross_spectral_matrix.shape).copy()

    for iteration in range(max_iterations):
        # ``int(is_converged.sum())`` would sync the device (and reduce on CPU)
        # every iteration; guard it so it only runs when debug logging is on.
        if logger.isEnabledFor(DEBUG):
            logger.debug(
                "iteration: %d, %d of %d converged",
                iteration,
                int(is_converged.sum()),
                n_units,
            )
        # No copy: ``old_minimum_phase_factor`` aliases the previous iterate,
        # which is safe only while every update below builds a new array. Never
        # update ``minimum_phase_factor`` in place.
        old_minimum_phase_factor = minimum_phase_factor
        # A rank-deficient sub-spectrum makes the batched solve inside
        # _get_linear_predictor singular; _solve_isolating_singular resolves only
        # that unit to NaN (matching the GPU path) instead of aborting the batch.
        linear_predictor = _get_linear_predictor(
            minimum_phase_factor,
            cross_spectral_square_root,
            identity_matrix,
        )
        minimum_phase_factor = xp.matmul(
            minimum_phase_factor, _get_causal_signal(linear_predictor, half_spectrum_n_fft)
        )

        # Freeze sub-spectra that already converged (broadcast the per-unit mask
        # over the frequency and signal axes).
        frozen = is_converged[..., xp.newaxis, xp.newaxis, xp.newaxis]
        minimum_phase_factor = xp.where(frozen, old_minimum_phase_factor, minimum_phase_factor)
        is_converged = _check_convergence(
            minimum_phase_factor, old_minimum_phase_factor, tolerance
        )
        # A sub-spectrum that became singular is now NaN and can never converge;
        # treat such units as finished so a single rank-deficient window does not
        # force the whole batch to exhaust the iteration budget. Combine the
        # "all finished" and "all converged" tests so the loop reduces the device
        # to a Python bool once per iteration (a GPU synchronization; a no-op
        # difference on CPU).
        singular_units = ~_all_finite_units(minimum_phase_factor, batch_shape)
        if xp.all(is_converged | singular_units):
            break

    # If not every sub-spectrum converged (iteration budget exhausted, or a
    # factor became singular), returning the partially-converged factor silently
    # would feed numerically wrong values into every downstream
    # directed-connectivity measure with no way for the caller to tell. Mark only
    # the unconverged sub-spectra as NaN (leaving the converged ones intact) and
    # warn loudly.
    n_failed = int((~is_converged).sum())
    if n_failed:
        # A singular sub-spectrum is the unconverged one whose factor is non-finite.
        singular_factor = bool(
            (~is_converged & ~_all_finite_units(minimum_phase_factor, batch_shape)).any()
        )
        unconverged = ~is_converged[..., xp.newaxis, xp.newaxis, xp.newaxis]
        minimum_phase_factor = xp.where(
            unconverged,
            xp.asarray(xp.nan, dtype=minimum_phase_factor.dtype),
            minimum_phase_factor,
        )
        if _warn_on_failure:
            reason = (
                "a sub-spectrum became singular (rank-deficient / duplicated channels)"
                if singular_factor
                else f"within {max_iterations} iterations (tolerance={tolerance})"
            )
            warnings.warn(
                f"Wilson minimum-phase decomposition did not converge for "
                f"{n_failed} of {n_units} sub-spectrum/spectra ({reason}). Those "
                f"sub-spectra are returned as NaN and will produce NaN in any "
                f"directed connectivity measure (spectral Granger, DTF, etc.). "
                f"Consider increasing max_iterations, using more tapers/trials, or "
                f"checking for near-singular cross-spectral matrices (highly "
                f"correlated channels).",
                UserWarning,
                stacklevel=2,
            )
    if half_spectrum_n_fft is not None:
        minimum_phase_factor = _to_two_sided(minimum_phase_factor, half_spectrum_n_fft)
    return minimum_phase_factor
