"""Minimum phase decomposition for spectral density matrices.

A spectral density matrix can be decomposed into minimum phase functions
using the Wilson algorithm. This decomposition is used in computing
pairwise spectral Granger prediction and other directed connectivity measures.
"""

import warnings
from logging import DEBUG, getLogger
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from spectral_connectivity._array_utils import _batched_eigh, _conjugate_transpose
from spectral_connectivity._backend import fft, ifft, irfft, rfft, xp
from spectral_connectivity.utils import stacklevel_outside_package

logger = getLogger(__name__)

# The opening words of the two warnings a failed factorization gives, which
# Connectivity matches to drop them for NaN or Inf coefficients.
CHOLESKY_FALLBACK_WARNING_START = "Computing the initial conditions using the Cholesky failed"
NONCONVERGENCE_WARNING_START = "Wilson minimum-phase decomposition did not converge"
NOT_MINIMUM_PHASE_WARNING_START = (
    "Wilson minimum-phase decomposition converged to a factor that is not minimum phase"
)

# The winding number of det G is counted only where its phase changes by less
# than this between neighbouring bins. Factors that were not minimum phase
# changed by at most 0.31 per bin; aliased counts on coarse grids (a winding
# of 2 for factors whose zero-padded winding is 0) came with changes of 1.9 to
# 3.1.
WINDING_MAX_PHASE_INCREMENT = np.pi / 2


def _not_minimum_phase_message(where: str) -> str:
    """The warning for factors that converged but are not minimum phase.

    ``where`` names what was factored, e.g. "3 of 10 sub-spectra" or a
    measure's signal pairs.
    """
    return (
        f"{NOT_MINIMUM_PHASE_WARNING_START} for {where}: its determinant winds "
        "around zero on the frequency grid. The factor "
        "reproduces the spectrum, but its transfer function and noise covariance, "
        "and every directed measure read from them (spectral Granger, DTF, PDC, "
        "etc.), are wrong there. This is a numerical failure of the factorization "
        "that a tighter tolerance or more iterations do not fix; treat those "
        "values as unreliable and report the case, with the spectrum that "
        "produced it, at https://github.com/Eden-Kramer-Lab/spectral_connectivity/"
        "issues. The check is best effort: a factor can fail to be minimum phase "
        "without being caught."
    )


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


def _signal_scale(
    cross_spectral_matrix: NDArray[np.complexfloating],
) -> NDArray[np.floating]:
    """Root-mean-square amplitude of each signal of each sub-spectrum.

    The square root of the signal's power averaged over the frequency axis
    (-3), its zero-lag autocovariance on the grid. A signal with zero or
    non-finite power gets a scale of 1, so that sub-spectrum is factored as it
    would be without scaling.

    Parameters
    ----------
    cross_spectral_matrix : NDArray[complexfloating],
        shape (..., n_fft_samples, n_signals, n_signals)
        Two-sided cross-spectral matrix.

    Returns
    -------
    NDArray[floating], shape (..., n_signals)
        Positive scale of each signal, in float64 or wider.
    """
    power = xp.mean(
        xp.diagonal(cross_spectral_matrix, axis1=-2, axis2=-1).real, axis=-2
    ).astype(xp.result_type(cross_spectral_matrix.real.dtype, xp.float64), copy=False)
    has_power = xp.isfinite(power) & (power > 0)
    return xp.sqrt(xp.where(has_power, power, 1))


def _reverse_cholesky(matrices: NDArray[np.floating]) -> NDArray[np.floating]:
    """Upper-triangular ``U`` with a positive diagonal and ``U Uᵀ = M``.

    ``J chol(J M J) J`` for the exchange matrix ``J``, which reverses the
    order of the signals; batched over the leading axes.
    """
    flipped: NDArray[np.floating] = xp.linalg.cholesky(matrices[..., ::-1, ::-1])
    return flipped[..., ::-1, ::-1]


def _get_initial_conditions(
    cross_spectral_matrix: NDArray[np.complexfloating],
    n_fft_samples: int | None = None,
) -> NDArray[np.floating]:
    """Generate initial guess for minimum phase factor using Cholesky decomposition.

    Provides an initial estimate for the Wilson algorithm: the upper-triangular
    ``U`` with a positive diagonal and ``U Uᵀ = R0``, where ``R0`` is the
    zero-lag cross-spectral matrix (real part of the inverse FFT). It is the
    "reverse" Cholesky factor ``J chol(J R0 J) J``, with ``J`` the exchange
    matrix. For a positive diagonal ``D`` the start for ``D R0 D`` is then
    ``D U``, as the factor of ``D S D`` is ``D G``, so the iteration is
    covariant under rescaling the signals. The transposed lower Cholesky
    factor ``chol(R0)ᵀ`` satisfies ``Uᵀ U = R0`` instead: its start for
    ``D R0 D`` is ``U D``, and for some spectra of signals of comparable power
    the iteration then converged to a factor that is not minimum phase.
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
        Initial guess for minimum phase square root matrix. Real-valued (a
        Cholesky factor of the real zero-lag matrix); the caller promotes it to
        the complex working dtype.

    Notes
    -----
    If the zero-lag matrix of a sub-spectrum is not positive-definite (Cholesky
    fails), only that sub-spectrum falls back to a fixed positive-definite start
    (``n_signals * I``); the healthy sub-spectra keep their Cholesky start. The
    fallback start is now deterministic instead of a random draw, so the result
    for such a pathological sub-spectrum no longer depends on the global NumPy
    random state. This is a NumPy-backend detail: CuPy's batched Cholesky never
    raises, so this branch is never taken there. For a bad unit CuPy returns NaN
    (indefinite matrix) or a finite factor with a zero diagonal entry
    (rank-deficient positive-semidefinite matrix) instead.
    """
    if n_fft_samples is None:
        zero_lag = ifft(cross_spectral_matrix, axis=-3)[..., 0:1, :, :].real
    else:
        zero_lag = irfft(cross_spectral_matrix, n=n_fft_samples, axis=-3)[..., 0:1, :, :]
    try:
        return _reverse_cholesky(zero_lag)
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
            f"{CHOLESKY_FALLBACK_WARNING_START} for "
            "some sub-spectra (rank-deficient / duplicated channels); using a "
            "deterministic positive-definite fallback start for those units. "
            "Their directed-connectivity values are not guaranteed correct — "
            "check for duplicated or near-collinear channels.",
            UserWarning,
            stacklevel=stacklevel_outside_package(),
        )
        logger.debug(
            "Computing the initial conditions using the Cholesky failed for "
            "some sub-spectra; using a deterministic fallback for those."
        )
        # Determine exactly which sub-spectra Cholesky cannot factor by
        # attempting it per unit. Any numerical-rank threshold is only an
        # approximation of potrf's actual pivoting (it depends on dtype and
        # scaling -- e.g. a valid float32 ``diag([1, 1e-7])`` sits below
        # ``n * eps(float32)``), so a per-unit attempt is the only reliable way
        # to preserve every successfully-factorable unit. This runs only on the
        # NumPy backend: CuPy's cholesky never raises (a bad unit gets NaN or a
        # zero-diagonal factor), so the batched call above isolates it there.
        flat_zero_lag = zero_lag.reshape((-1, *zero_lag.shape[-2:]))
        not_positive_definite = xp.zeros(flat_zero_lag.shape[0], dtype=bool)
        for index in range(flat_zero_lag.shape[0]):
            try:
                _reverse_cholesky(flat_zero_lag[index])
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
        return _reverse_cholesky(safe_flat.reshape(zero_lag.shape))


def _get_causal_signal(
    linear_predictor: NDArray[np.complexfloating],
    n_fft_samples: int | None = None,
    *,
    lower_triangular_ind: tuple[NDArray[np.intp], NDArray[np.intp]],
) -> NDArray[np.complexfloating]:
    """Extract causal part of linear predictor (plus operator).

    Implements the "plus" operator from the Wilson algorithm by:
    1. Taking half the roots on the unit circle (zero lag, and on an even grid
       the lag ``n / 2``, which is its own negative there)
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
    lower_triangular_ind : tuple of two index arrays
        ``xp.tril_indices(n_signals, k=-1)``. CuPy's ``tril_indices``
        synchronizes the device (it calls ``nonzero``), so the Wilson iteration
        computes it once and passes it to every call.

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
    linear_predictor_coefficients[..., 0, lower_triangular_ind[0], lower_triangular_ind[1]] = 0

    # Take only the roots inside the unit circle (positive lags). On an even
    # grid the lag n_lags / 2 is its own negative, exp(i * pi * k) at bin k, so
    # like lag 0 it is half causal: the projection of a Hermitian function
    # plus its conjugate transpose must give that function back. Dropping the
    # lag instead left it unconstrained, and the iteration settled on a factor
    # with G^-1 S G^-H = I + (-1)^k C in place of I.
    if n_lags % 2 == 0:
        linear_predictor_coefficients[..., n_lags // 2, :, :] *= 0.5
    linear_predictor_coefficients[..., n_lags // 2 + 1 :, :, :] = 0
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
    >>> from spectral_connectivity._backend import xp
    >>> rng = np.random.default_rng(0)
    >>> shape = (10, 8, 5, 5)
    >>> current = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    >>> old = current + 1e-10 * rng.standard_normal(shape)
    >>> converged = _check_convergence(xp.asarray(current), xp.asarray(old), tolerance=1e-8)
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


def _determinant_winding_number(
    factor: NDArray[np.complexfloating],
    n_fft_samples: int | None = None,
) -> NDArray[np.floating]:
    """Times ``det G(e^{iω})`` winds around zero as ``ω`` goes once round the circle.

    The phase increments between neighbouring bins, each wrapped to
    ``(-π, π]``, summed and divided by ``2π``. A minimum-phase factor's
    determinant has no zeros or poles outside the unit disk (in ``z⁻¹``), so
    its winding number is 0; rescaling the signals multiplies ``det G`` by a
    positive constant and leaves it unchanged. The count is reliable only if
    the phase changes by much less than ``π`` between neighbouring bins, so a
    factor whose largest increment reaches ``WINDING_MAX_PHASE_INCREMENT`` is
    reported as NaN (not tested).

    As a test of minimum phase it is best effort: a factor that is not
    minimum phase but whose determinant does not wind countably (a causal
    factor with zeros outside the unit circle that the count misses, or an
    anticausal one) passes it.

    Parameters
    ----------
    factor : NDArray[complexfloating],
        shape (..., n_frequencies, n_signals, n_signals)
        Factors on every bin of the grid, or on the non-negative half of a
        conjugate-symmetric one of ``n_fft_samples`` bins.
    n_fft_samples : int, optional
        Two-sided length of a half spectrum.

    Returns
    -------
    NDArray[floating], shape (...,)
        The winding number, or NaN for a non-finite factor or one whose phase
        changes too fast between bins to count.
    """
    if factor.shape[-1] == 2:
        # Closed form: LAPACK's per-matrix overhead dominates for 2x2 (the
        # pairwise and subset Granger factors), as in _solve_2x2.
        determinant = (
            factor[..., 0, 0] * factor[..., 1, 1] - factor[..., 0, 1] * factor[..., 1, 0]
        )
    else:
        determinant = xp.linalg.det(factor)
    if n_fft_samples is not None:
        n_nonnegative = determinant.shape[-1]
        determinant = xp.concatenate(
            (
                determinant,
                xp.conj(determinant[..., n_fft_samples - n_nonnegative : 0 : -1]),
            ),
            axis=-1,
        )
    increments = xp.angle(xp.roll(determinant, -1, axis=-1) * xp.conj(determinant))
    resolved = xp.max(xp.abs(increments), axis=-1) < WINDING_MAX_PHASE_INCREMENT
    winding: NDArray[np.floating] = xp.where(
        resolved, xp.sum(increments, axis=-1) / (2 * np.pi), xp.nan
    )
    return winding


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


def _solve_2x2(
    coefficient_matrix: NDArray[np.complexfloating],
    right_hand_side: NDArray[np.complexfloating],
) -> NDArray[np.complexfloating]:
    """Closed-form solve of 2x2 systems; singular or non-finite ones are NaN.

    Every 2-signal Wilson factorization (e.g. pairwise and subset spectral
    Granger) solves 2x2 systems, where LAPACK's per-matrix overhead dominates.
    Cramer's rule avoids it and, being backward stable for 2x2 systems, agrees
    with ``linalg.solve`` to within rounding amplified by the condition number.

    Only an exactly zero determinant counts as singular: a near-singular system
    is solved like any other, whatever else is in the batch.

    Parameters
    ----------
    coefficient_matrix : NDArray[complexfloating], shape (..., 2, 2)
        Batched left-hand-side matrices ``A`` in ``A x = B``.
    right_hand_side : NDArray[complexfloating], shape (..., 2, n_columns)
        Batched right-hand sides ``B``.

    Returns
    -------
    NDArray[complexfloating], same shape as ``right_hand_side``
        Solution ``x``, with NaN where ``det(A) == 0`` or ``A`` is non-finite.
    """
    a, b = coefficient_matrix[..., 0, 0, None], coefficient_matrix[..., 0, 1, None]
    c, d = coefficient_matrix[..., 1, 0, None], coefficient_matrix[..., 1, 1, None]
    first_row, second_row = right_hand_side[..., 0, :], right_hand_side[..., 1, :]
    # Any non-finite entry of A makes the determinant non-finite, so checking
    # the determinant alone catches non-finite A. Such systems (and exactly
    # singular ones, divided by 1 instead) are replaced by NaN at the end, so
    # NumPy need not warn about the invalid values they produce on the way.
    with np.errstate(invalid="ignore", over="ignore"):
        determinant = (a * d - b * c)[..., xp.newaxis]
        singular = ~xp.isfinite(determinant) | (determinant == 0)
        solution = xp.stack(
            (d * first_row - b * second_row, a * second_row - c * first_row), axis=-2
        )
        solution /= xp.where(singular, 1, determinant)
    return xp.where(singular, xp.nan, solution)


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
    gives the NumPy path the same isolation: singular (or already non-finite)
    sub-matrices resolve to NaN while the remaining ones are solved normally, so
    a single rank-deficient window (e.g. duplicated channels) does not poison
    the entire batch. Exactly singular sub-matrices are non-finite on both
    backends: all NaN on NumPy, a mix of NaN and +/-Inf on CuPy. 2x2 systems bypass LAPACK via :func:`_solve_2x2`, which treats
    only an exactly zero determinant as singular.

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
    if coefficient_matrix.shape[-1] == 2:
        return _solve_2x2(coefficient_matrix, right_hand_side)
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
    eigenvalues, eigenvectors = _batched_eigh(xp.where(is_finite, matrices, 0))
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
        Relative convergence tolerance for Wilson algorithm iterations,
        applied to the factor of the scaled spectrum (see Notes).
    max_iterations : int, default=500
        Maximum number of iterations before stopping algorithm. Near-singular
        cross-spectral matrices (highly correlated channels) can need several
        hundred iterations to reach the relative tolerance; the loop returns
        early as soon as every sub-spectrum has converged. 2.x defaulted to
        60; pass ``max_iterations=60`` for that cap.

        ``tolerance`` and ``max_iterations`` may be passed by position or by
        keyword; keywords make a call easier to read.

    Returns
    -------
    minimum_phase_factor : NDArray[complexfloating],
        shape (n_time_samples, ..., n_fft_samples, n_signals, n_signals)
        Minimum phase square root of cross_spectral_matrix. All eigenvalues
        have negative real parts (minimum phase property). An array of the
        active backend: CuPy on the GPU, even for NumPy input.

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
    >>> min_phase = minimum_phase_decomposition(
    ...     cross_spec, tolerance=1e-8, max_iterations=500
    ... )
    >>> # The factor reconstructs the input: G @ G^H == S.
    >>> from spectral_connectivity.utils import to_numpy
    >>> factor = to_numpy(min_phase)  # a CuPy array on the GPU backend
    >>> bool(np.allclose(factor @ factor.conj().swapaxes(-1, -2), cross_spec))
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

    Every factor ``G`` with ``G Gᴴ = S`` is a fixed point of the iteration;
    which one it reaches depends on where it starts. It starts from the
    upper-triangular ``U`` with ``U Uᵀ = R0`` for the zero-lag matrix ``R0``
    (the reverse Cholesky factor), so the factorization is covariant under
    rescaling the signals: for a positive diagonal ``D`` the factor of
    ``D S D`` is ``D G``, up to rounding and the convergence tolerance. The
    transposed Cholesky factor ``chol(R0)ᵀ``, which satisfies ``Uᵀ U = R0``
    instead, is the start of FieldTrip's ``sfactorization_wilson.m`` (MATLAB's
    ``chol``) and of this function before 3.0; it is not covariant. As
    preconditioning, each sub-spectrum is also factored with its signals
    scaled to unit mean power: the iteration factors ``D S D``, with ``D_ii``
    one over the square root of the mean of ``S_ii`` over frequency, and
    returns ``D⁻¹ G'`` for its factor ``G'``. A signal with zero or
    non-finite power keeps a scale of 1.

    Each converged factor is then checked, on a best-effort basis, for being
    minimum phase: a sub-spectrum whose ``det G(e^{iω})`` winds around zero,
    counted where its phase turns by less than ``π / 2`` between bins, gives
    a ``UserWarning`` (a spectral Granger measure warns once per call, naming
    the signal pairs), and its factor is returned as computed. A factor that
    fails to be minimum phase without its determinant winding countably
    passes the check. It costs less than one iteration.

    Each update is built from a square root ``L`` of ``S`` (``S = L Lᴴ``),
    computed once, which lets sub-spectra with near-collinear channels converge.
    An indefinite or non-finite ``S`` has no such square root and is returned as
    NaN with the non-convergence warning.

    A fixed point of the iteration satisfies ``G Gᴴ = S`` at every frequency
    bin. That does not make the directed measures accurate. Two limits remain,
    and nothing here detects either:

    - The grid. The factor is that of the spectrum's periodic extension on the
      grid, so lags of ``S`` beyond ``n_fft_samples / 2`` wrap around. A
      spectrum estimated from windows of ``N`` samples has lags out to
      ``N - 1``, which wrap on a grid of ``N`` bins, so the directed measures
      depend on the FFT length. In simulations, zero-padding the transform to
      ``n_fft_samples = 2 * N`` moved spectral Granger prediction by about
      2e-2 (RMS over frequencies) for a 1000-sample recording, and padding
      further moved it by less than 1e-3. Padding is not a gain in accuracy in
      general: against the generating model's true values the error of
      spectral Granger, DTF, and PDC was the same with and without it (within
      1%) for well-resolved recordings, and 4-9% lower for DTF and PDC only
      where the window was too short for the signal. It costs about twice the
      time and frequency bins.
    - The window. An estimate from a window shorter than the slowest
      correlation in the data is smoothed in frequency, which no FFT length
      repairs. Use a longer window.

    .. versionchanged:: 3.0
       On an even-length grid the lag ``n_fft_samples / 2`` is its own
       negative, so half of it belongs to the causal part, as with lag 0.
       Before 3.0 the causal projection dropped it. The iteration then
       converged to a factor with ``G⁻¹ S G⁻ᴴ = I + (-1)ᵏ C`` at bin ``k``
       instead of ``I``: it missed the spectrum by the size of that lag
       (0.2-10% for spectra estimated from simulated data, more for short
       windows and highly correlated channels) and added a bin-to-bin
       alternation to every measure derived from it. Odd-length grids were
       not affected.

    .. versionchanged:: 3.0
       The iteration starts from ``U`` with ``U Uᵀ = R0`` instead of
       ``chol(R0)ᵀ``, and the signals are scaled to unit mean power. From the
       old start the iteration could converge, with no warning and ``G Gᴴ``
       within 1e-9 of ``S``, to a factor that is not minimum phase, and every
       directed measure was then wrong: for 9 of 400 random dense VARs (2-6
       signals of comparable power, companion-matrix radius 0.8-0.99, exact
       spectra; all 9 at 0.99), DTF and PDC missed their true values by
       0.3-0.8, and for a chain of three nearly integrating signals (mean
       powers 50, 1e4 and 3e6) pairwise spectral Granger missed by up to 6.5
       and differed between the raw and the standardized signals. Without the
       scaling, the covariant start still missed conditional Granger by 1e-4
       for that chain with its signals rescaled by 0.1, 1 and 10, and
       conditional Granger by 4e-3 and PDC by 0.4 with them rescaled by 1e-3,
       1 and 1e3.

    References
    ----------
    .. [1] Wilson, G. T. (1972). The factorization of matricial spectral
           densities. SIAM Journal on Applied Mathematics, 23(4), 420-426.
    .. [2] Dhamala, M., Rangarajan, G., & Ding, M. (2008). Analyzing
           information flow in brain networks with nonparametric Granger
           causality. NeuroImage, 41(2), 354-362.
    """
    factorization = _minimum_phase_factor_and_flags(
        cross_spectral_matrix, tolerance, max_iterations
    )
    n_units = factorization.unconverged.size
    n_failed = int(factorization.unconverged.sum())
    if n_failed:
        reason = (
            "a sub-spectrum became singular (rank-deficient / duplicated channels)"
            if bool(factorization.singular.any())
            else f"within {max_iterations} iterations (tolerance={tolerance})"
        )
        warnings.warn(
            f"{NONCONVERGENCE_WARNING_START} for "
            f"{n_failed} of {n_units} sub-spectrum/spectra ({reason}). Those "
            f"sub-spectra are returned as NaN and will produce NaN in any "
            f"directed connectivity measure (spectral Granger, DTF, etc.). "
            f"Consider increasing max_iterations, using more tapers/trials, or "
            f"checking for near-singular cross-spectral matrices (highly "
            f"correlated channels).",
            UserWarning,
            stacklevel=stacklevel_outside_package(),
        )
    n_not_minimum_phase = int(factorization.not_minimum_phase.sum())
    if n_not_minimum_phase:
        warnings.warn(
            _not_minimum_phase_message(
                f"{n_not_minimum_phase} of {n_units} sub-spectrum/spectra"
            ),
            UserWarning,
            stacklevel=stacklevel_outside_package(),
        )
    return factorization.factor


class _Factorization(NamedTuple):
    """A Wilson factorization and what went wrong in it, per sub-spectrum.

    The masks have shape ``cross_spectral_matrix.shape[:-3]``.
    """

    factor: NDArray[np.complexfloating]
    """Two-sided factor; NaN for the unconverged sub-spectra."""
    not_minimum_phase: NDArray[np.bool_]
    """Converged factors whose determinant winds around zero."""
    unconverged: NDArray[np.bool_]
    """Sub-spectra that did not converge."""
    singular: NDArray[np.bool_]
    """Unconverged sub-spectra whose factor became non-finite (singular)."""


def _minimum_phase_factor_and_flags(
    cross_spectral_matrix: NDArray[np.complexfloating],
    tolerance: float,
    max_iterations: int,
) -> _Factorization:
    """:func:`minimum_phase_decomposition` without its warnings.

    Returns the factor with the masks of its failures; the callers report
    them (the spectral Granger measures name the signal pairs).
    """
    if not np.isfinite(tolerance) or tolerance <= 0:
        msg = f"tolerance must be a finite positive number, got {tolerance}."
        raise ValueError(msg)
    if (
        not isinstance(max_iterations, (int, np.integer))  # type: ignore[redundant-expr]  # user input
        or max_iterations < 1
    ):
        msg = f"max_iterations must be a positive integer, got {max_iterations}."
        raise ValueError(msg)
    # Accept host (NumPy) arrays on the GPU backend; a no-op for device arrays.
    cross_spectral_matrix = xp.asarray(cross_spectral_matrix)
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
    # Factor D S D, with D scaling every signal to unit mean power, and return
    # D^-1 G for its factor G. The covariant start makes that the factor of S
    # up to rounding; the scaling preconditions the iteration for signals
    # whose powers differ by orders of magnitude (see Notes).
    signal_scale = _signal_scale(cross_spectral_matrix)
    # NumPy divides a complex Inf by the real scale as by a complex number,
    # which gives NaN and warns; such a sub-spectrum is non-finite either way
    # and is reported as unconverged below.
    with np.errstate(invalid="ignore"):
        working_cross_spectral_matrix = (
            working_cross_spectral_matrix
            / (
                signal_scale[..., xp.newaxis, :, xp.newaxis]
                * signal_scale[..., xp.newaxis, xp.newaxis, :]
            )
        ).astype(working_dtype, copy=False)
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
    # Computed once: CuPy's tril_indices synchronizes the device.
    lower_triangular_ind = xp.tril_indices(n_signals, k=-1)

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
            minimum_phase_factor,
            _get_causal_signal(
                linear_predictor,
                half_spectrum_n_fft,
                lower_triangular_ind=lower_triangular_ind,
            ),
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
        # to a Python bool once per iteration: on CuPy this is the iteration's
        # only device synchronization, needed for the early exit.
        singular_units = ~_all_finite_units(minimum_phase_factor, batch_shape)
        if xp.all(is_converged | singular_units):
            break

    # If not every sub-spectrum converged (iteration budget exhausted, or a
    # factor became singular), returning the partially-converged factor silently
    # would feed numerically wrong values into every downstream
    # directed-connectivity measure with no way for the caller to tell. Mark only
    # the unconverged sub-spectra as NaN (leaving the converged ones intact);
    # the callers report them.
    unconverged = ~is_converged
    # A singular sub-spectrum is the unconverged one whose factor is non-finite.
    singular = unconverged & ~_all_finite_units(minimum_phase_factor, batch_shape)
    if bool(unconverged.any()):
        minimum_phase_factor = xp.where(
            unconverged[..., xp.newaxis, xp.newaxis, xp.newaxis],
            xp.asarray(xp.nan, dtype=minimum_phase_factor.dtype),
            minimum_phase_factor,
        )
    # A converged factor whose determinant winds around zero is not minimum
    # phase. The winding number is unchanged by the scaling; the unconverged
    # sub-spectra are NaN by now and are not flagged.
    winding = _determinant_winding_number(minimum_phase_factor, half_spectrum_n_fft)
    not_minimum_phase = xp.abs(winding) > 0.5
    minimum_phase_factor = signal_scale[..., xp.newaxis, :, xp.newaxis] * minimum_phase_factor
    if half_spectrum_n_fft is not None:
        minimum_phase_factor = _to_two_sided(minimum_phase_factor, half_spectrum_n_fft)
    return _Factorization(minimum_phase_factor, not_minimum_phase, unconverged, singular)
