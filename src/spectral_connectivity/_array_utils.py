"""Backend-neutral array helpers shared by the transform and connectivity code."""

from typing import TypeVar

import numpy as np
from numpy.typing import DTypeLike, NDArray

from spectral_connectivity._backend import xp
from spectral_connectivity.utils import BackendArray

# Tikhonov regularization factor for stabilizing matrix inversions
# Used to prevent numerical instability with near-singular matrices
TIKHONOV_REGULARIZATION_FACTOR = 1e-12

# Preserves a helper's input dtype (real vs complex) in its return annotation.
_NumberT = TypeVar("_NumberT", bound=np.number)


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


def _divide_where(
    numerator: BackendArray,
    denominator: BackendArray,
    condition: BackendArray,
    fill: float,
) -> BackendArray:
    """Elementwise ``numerator / denominator`` where ``condition``, else ``fill``.

    Backend-neutral replacement for ``xp.divide(..., where=...)``: CuPy ufuncs
    do not accept the public ``where`` keyword, and the substituted unit
    denominator also avoids NumPy divide warnings.
    """
    quotient = numerator / xp.where(condition, denominator, 1)
    return xp.where(condition, quotient, xp.asarray(fill, dtype=quotient.dtype))


def _regularized_inverse(
    matrix: NDArray[_NumberT],
    regularization: float = TIKHONOV_REGULARIZATION_FACTOR,
) -> NDArray[_NumberT]:
    """Return the Tikhonov-regularized inverse of a batched matrix.

    Solves ``(M + λI) X = I`` instead of inverting ``M`` directly. The diagonal
    loading ``λ = TIKHONOV_REGULARIZATION_FACTOR * sqrt(mean(|M|^2))`` is scaled
    per batched matrix (over the last two axes), so windows/frequencies with very
    different power are each conditioned appropriately. Using the RMS magnitude
    (amplitude units) rather than the mean square keeps ``λ`` in the same units
    as ``M``: adding ``λI`` is dimensionally consistent and the regularized
    inverse is scale-covariant (rescaling ``M`` by ``c`` rescales ``λ`` by ``c``,
    leaving downstream connectivity measures invariant).

    Parameters
    ----------
    matrix : NDArray, shape (..., n_signals, n_signals)
        Batched (real or complex) matrices to invert.
    regularization : float, default=1e-12
        Relative diagonal-loading factor.

    Returns
    -------
    NDArray, shape (..., n_signals, n_signals)
        Regularized inverse of each batched matrix.
    """
    lam = regularization * xp.sqrt(
        xp.mean(xp.real(xp.conj(matrix) * matrix), axis=(-2, -1), keepdims=True)
    )
    identity = xp.eye(matrix.shape[-1], dtype=matrix.dtype)
    # Broadcast identity to the batch dimensions so CuPy's batched solve accepts
    # the RHS shape (NumPy tolerates the mismatch; CuPy does not).
    identity_batched = xp.broadcast_to(identity, matrix.shape)
    return xp.linalg.solve(matrix + lam * identity_batched, identity_batched)


def _batched_inverse_square_root(
    matrices: NDArray[np.floating], *, rank: int | None, regularization: float
) -> tuple[NDArray[np.floating], NDArray[np.integer]]:
    """Inverse square root of batched real symmetric matrices, with kept rank.

    ``matrices`` has shape ``(..., n, n)``; the eigendecomposition, rank mask and
    regularization are applied independently per leading (time/frequency) bin on
    the active ``xp`` backend. Returns ``(T, kept_rank)`` where ``kept_rank`` is
    the number of retained (numerically non-zero, rank-capped) directions per
    bin -- used to detect null-space "phantom" components scale-invariantly.
    """
    symmetric = (matrices + matrices.swapaxes(-1, -2)) / 2
    eigenvalues, eigenvectors = xp.linalg.eigh(symmetric)
    largest = xp.maximum(eigenvalues[..., -1:], 0.0)
    tolerance = xp.finfo(eigenvalues.dtype).eps * matrices.shape[-1] * largest
    keep = eigenvalues > tolerance
    n_channels = matrices.shape[-1]
    if rank is not None and rank < n_channels:
        keep = keep & (xp.arange(n_channels) >= (n_channels - rank))
    matrix_rms = xp.sqrt(xp.mean(symmetric**2, axis=(-2, -1)))[..., xp.newaxis]
    safe_values = xp.where(keep, eigenvalues + regularization * matrix_rms, 1.0)
    inverse_values = xp.where(keep, 1.0 / xp.sqrt(safe_values), 0.0)
    transform = (eigenvectors * inverse_values[..., xp.newaxis, :]) @ eigenvectors.swapaxes(
        -1, -2
    )
    return transform, keep.sum(-1)


def _squared_magnitude(x: NDArray[np.complexfloating]) -> NDArray[np.floating]:
    """Return squared magnitude of complex array.

    Parameters
    ----------
    x : array_like
        Complex input array.

    Returns
    -------
    array_like
        Squared magnitude values.

    """
    return xp.abs(x) ** 2


def _complex_inner_product(
    a: NDArray[np.complexfloating],
    b: NDArray[np.complexfloating],
    dtype: DTypeLike = xp.complex128,
) -> NDArray[np.complexfloating]:
    """Measure orthogonality (similarity) of complex arrays.

    Measures the orthogonality (similarity) of complex arrays in
    the last two dimensions.

    Parameters
    ----------
    a, b : array_like
        Complex input arrays.
    dtype : np.dtype, default=complex128
        Data type for computation.

    Returns
    -------
    array_like
        Complex inner product.

    """
    product: NDArray[np.complexfloating] = xp.matmul(a, _conjugate_transpose(b), dtype=dtype)
    return product
