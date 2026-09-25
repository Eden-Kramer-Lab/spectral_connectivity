"""Multivariate coupling kernels: canonical coherence and coherency, MIC, and
global coherence.

Array-level computations behind the group and multivariate measures of
:class:`Connectivity`.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from spectral_connectivity._array_utils import (
    _batched_inverse_square_root,
    _complex_inner_product,
    _conjugate_transpose,
    _divide_where,
)
from spectral_connectivity._backend import svds, xp

# Bin-chunk element cap for the batched path: peak memory scales with
# chunk * (n_signals * n_estimates + min(n_signals, n_estimates)**2), so cap the
# element count to keep it bounded regardless of the number of bins.
GLOBAL_COHERENCE_BATCH_CHUNK_ELEMENTS = 16_000_000


def _dominant_sign(vectors: NDArray[np.floating]) -> NDArray[np.floating]:
    """Sign (+1/-1) of the largest-magnitude entry along the last axis, per bin."""
    dominant = xp.take_along_axis(
        vectors, xp.argmax(xp.abs(vectors), axis=-1)[..., xp.newaxis], axis=-1
    )[..., 0]
    return xp.where(dominant < 0, -1.0, 1.0)


def _optimize_canonical_coherency_phase(
    whitened: NDArray[np.complexfloating],
    *,
    n_grid: int = 74,
    n_refine: int = 12,
) -> tuple[
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
]:
    """Vidaurre's phase objective, optimized per bin over batched leading axes.

    ``whitened`` is the whitened between-group cross-spectrum ``W = Taa Cab
    Tbb`` with shape ``(..., n_a, n_b)``. The objective
    ``sigma_max(Re(exp(-i phi) W))`` is pi-periodic and can have several lobes
    of nearly equal height, so refining only the best coarse-grid point (or a
    fixed number of the highest) can settle on the wrong lobe. Instead, every
    local maximum of a coarse ``n_grid``-point grid over ``[0, pi)`` is refined
    by a batched Newton iteration on finite-difference derivatives, and the best
    refined candidate is kept -- never below the coarse-grid optimum, hence at
    least ``cos(pi / (2 n_grid))`` times the global maximum. Each lobe with its
    own coarse-grid maximum is therefore found; two lobes within about two grid
    steps (``2 pi / n_grid``) of each other can merge into one coarse-grid
    maximum, whose refinement then settles on only one of them. Fully
    vectorized over the leading (time/frequency) axes and backend-agnostic (no
    per-bin ``scipy.optimize`` loop). Returns the maximized magnitude, the
    optimizing phase, and the full orthogonal left/right singular-vector
    matrices of ``Re(exp(-i phi) W)`` at that phase, shapes ``(..., n_a, n_a)``
    and ``(..., n_b, n_b)``: column 0 holds the top pair, and the remaining
    columns span its orthogonal complement.
    """
    leading_shape = whitened.shape[:-2]
    n_a, n_b = whitened.shape[-2:]
    # One flat bin axis, so the local maxima of every bin form one Newton batch.
    flat = whitened.reshape(-1, n_a, n_b)
    n_bins = flat.shape[0]

    def objective(
        phase: NDArray[np.floating], matrices: NDArray[np.complexfloating]
    ) -> NDArray[np.floating]:
        projected = xp.real(xp.exp(-1j * phase)[..., xp.newaxis, xp.newaxis] * matrices)
        return xp.linalg.svd(projected, compute_uv=False)[..., 0]

    grid_values = [k * float(np.pi) / n_grid for k in range(n_grid)]
    # One SVD batch per grid point keeps the grid's transient at a single
    # bin-batch; the Newton batch below holds one matrix per local maximum.
    grid_scores = xp.stack([objective(xp.full(n_bins, phase), flat) for phase in grid_values])
    grid = xp.asarray(grid_values)
    # Local maxima of the pi-periodic grid (its ends are neighbours), shape
    # (n_grid, n_bins). The strict comparison on one side counts a flat top once,
    # so no two maxima are adjacent: at most n_grid // 2 per bin, and typically
    # one or two. A constant objective has none; its grid optimum stands.
    is_local_maximum = (grid_scores > xp.roll(grid_scores, 1, axis=0)) & (
        grid_scores >= xp.roll(grid_scores, -1, axis=0)
    )
    grid_index, bin_index = xp.nonzero(is_local_maximum)
    matrices = flat[bin_index]  # (n_maxima, n_a, n_b)
    phase = grid[grid_index]
    step = 1e-5
    for _ in range(n_refine):
        forward = objective(phase + step, matrices)
        centre = objective(phase, matrices)
        backward = objective(phase - step, matrices)
        first_derivative = (forward - backward) / (2 * step)
        second_derivative = (forward - 2 * centre + backward) / step**2
        newton_step = _divide_where(
            first_derivative, second_derivative, xp.abs(second_derivative) > 1e-12, 0.0
        )
        phase = phase - xp.clip(newton_step, -0.1, 0.1)
    # Each maximum keeps its refined phase only where that improves on its grid
    # point, so the best per bin is never below the coarse-grid optimum.
    refined = objective(phase, matrices)
    improved = refined > grid_scores[grid_index, bin_index]
    scores = grid_scores.copy()
    phases = xp.broadcast_to(grid[:, xp.newaxis], grid_scores.shape).copy()
    scores[grid_index, bin_index] = xp.where(improved, refined, scores[grid_index, bin_index])
    phases[grid_index, bin_index] = xp.where(improved, phase, phases[grid_index, bin_index])
    best = xp.argmax(scores, axis=0)
    phase = xp.take_along_axis(phases, best[xp.newaxis], axis=0)[0]
    projected = xp.real(xp.exp(-1j * phase)[..., xp.newaxis, xp.newaxis] * flat)
    left, singular_values, right_h = xp.linalg.svd(projected)
    return (
        singular_values[:, 0].reshape(leading_shape),
        phase.reshape(leading_shape),
        left.reshape(*leading_shape, n_a, n_a),
        right_h.swapaxes(-1, -2).reshape(*leading_shape, n_b, n_b),
    )


def _canonical_coherency_components(
    Caa: NDArray[np.complexfloating],
    Cab: NDArray[np.complexfloating],
    Cbb: NDArray[np.complexfloating],
    *,
    rank: int | None,
    n_components: int,
    regularization: float,
) -> tuple[
    NDArray[np.complexfloating],
    tuple[NDArray[np.floating], NDArray[np.floating]],
    tuple[NDArray[np.floating], NDArray[np.floating]],
    NDArray[np.integer],
]:
    """Exact phase-optimised CaCoh components, batched over leading axes.

    Each of ``Caa``/``Cab``/``Cbb`` has shape ``(..., n_a/n_b, n_a/n_b)`` with
    arbitrary leading (time/frequency) axes. Returns per-component scores,
    filters, patterns (all with a trailing ``component`` axis) plus the per-bin
    effective within-group rank.

    Components are extracted by CCA-style deflation in whitened space. With
    ``W = Taa Cab Tbb`` (``Taa``/``Tbb`` the real within-group inverse square
    roots), component ``k`` maximizes the phase objective over
    ``Q_a^T W Q_b``, where the columns of ``Q`` are an orthonormal basis of the
    orthogonal complement of the previous components' whitened directions, and
    its filters are ``a_k = Taa u_k`` with whitened direction ``u_k = Q_a x_k``
    (``x_k`` the top singular vector). Hence ``a_j^T Re(Caa) a_k = u_j^T u_k =
    0`` for ``j != k`` (uncorrelated component signals) and every component is
    invariant to invertible real mixing within a group. Deflating the
    channel-space filters instead would tie later components to the channel
    basis. The next ``Q`` is ``Q`` times the remaining singular vectors, so it
    stays orthonormal even where ``W`` vanishes and the singular vectors are
    arbitrary (subtracting ``u u^T`` from a projector would not).
    """
    real_aa = xp.real(Caa)
    real_bb = xp.real(Cbb)
    leading_shape = Caa.shape[:-2]
    n_a = Caa.shape[-1]
    n_b = Cbb.shape[-1]
    transform_aa, rank_a = _batched_inverse_square_root(
        real_aa, rank=rank, regularization=regularization
    )
    transform_bb, rank_b = _batched_inverse_square_root(
        real_bb, rank=rank, regularization=regularization
    )
    whitened = transform_aa @ Cab @ transform_bb
    # Orthonormal bases of the not-yet-extracted whitened directions, shapes
    # (..., n_a, n_a - component) and (..., n_b, n_b - component).
    basis_a: NDArray[np.floating] = xp.broadcast_to(xp.eye(n_a), (*leading_shape, n_a, n_a))
    basis_b: NDArray[np.floating] = xp.broadcast_to(xp.eye(n_b), (*leading_shape, n_b, n_b))
    scores = xp.full((*leading_shape, n_components), xp.nan, dtype=Cab.dtype)
    filters_a = xp.full((*leading_shape, n_a, n_components), xp.nan)
    filters_b = xp.full((*leading_shape, n_b, n_components), xp.nan)
    patterns_a = xp.full_like(filters_a, xp.nan)
    patterns_b = xp.full_like(filters_b, xp.nan)

    for component in range(n_components):
        restricted = basis_a.swapaxes(-1, -2) @ whitened @ basis_b
        magnitude, phase, left, right = _optimize_canonical_coherency_phase(restricted)
        filter_a = (transform_aa @ basis_a @ left[..., :, :1])[..., 0]
        filter_b = (transform_bb @ basis_b @ right[..., :, :1])[..., 0]
        pattern_a = (real_aa @ filter_a[..., xp.newaxis])[..., 0]
        pattern_b = (real_bb @ filter_b[..., xp.newaxis])[..., 0]
        # A spatial filter and its negative span the same direction, so the
        # optimizer's phase is only defined modulo pi. Fix each filter's sign by
        # its pattern's dominant coefficient; flipping one filter negates the
        # projected coherency, i.e. shifts the phase by pi.
        sign_a = _dominant_sign(pattern_a)
        sign_b = _dominant_sign(pattern_b)
        phase = xp.where(sign_a * sign_b < 0, phase + xp.pi, phase)
        scores[..., component] = magnitude * xp.exp(-1j * phase)
        filters_a[..., component] = filter_a * sign_a[..., xp.newaxis]
        filters_b[..., component] = filter_b * sign_b[..., xp.newaxis]
        patterns_a[..., component] = pattern_a * sign_a[..., xp.newaxis]
        patterns_b[..., component] = pattern_b * sign_b[..., xp.newaxis]
        if component + 1 < n_components:
            # The remaining singular vectors span the complement of the extracted
            # direction within the current basis.
            basis_a = basis_a @ left[..., :, 1:]
            basis_b = basis_b @ right[..., :, 1:]

    # Deflation only removes the extracted directions, not the group's null
    # space, so a component beyond the joint within-group rank still optimizes a
    # spurious (zero) direction.
    return _zero_unsupported_components(
        scores, (filters_a, filters_b), (patterns_a, patterns_b), xp.minimum(rank_a, rank_b)
    )


def _zero_unsupported_components(
    scores: NDArray[Any],
    filters: tuple[NDArray[np.floating], NDArray[np.floating]],
    patterns: tuple[NDArray[np.floating], NDArray[np.floating]],
    effective_rank: NDArray[np.integer],
) -> tuple[
    NDArray[Any],
    tuple[NDArray[np.floating], NDArray[np.floating]],
    tuple[NDArray[np.floating], NDArray[np.floating]],
    NDArray[np.integer],
]:
    """Zero the phantom components beyond the joint within-group rank.

    ``scores`` has shape ``(..., n_components)``; each filter/pattern has shape
    ``(..., n_group_signals, n_components)``; ``effective_rank`` has the leading
    shape ``(...)``. Components at index ``>= effective_rank`` get a zero score
    and an all-zero filter/pattern, matching the caller's warning.
    """
    supported = xp.arange(scores.shape[-1]) < effective_rank[..., xp.newaxis]
    supported_sides = supported[..., xp.newaxis, :]
    filter_a, filter_b = filters
    pattern_a, pattern_b = patterns
    return (
        xp.where(supported, scores, 0.0),
        (
            xp.where(supported_sides, filter_a, 0.0),
            xp.where(supported_sides, filter_b, 0.0),
        ),
        (
            xp.where(supported_sides, pattern_a, 0.0),
            xp.where(supported_sides, pattern_b, 0.0),
        ),
        effective_rank,
    )


def _mic_components(
    Caa: NDArray[np.complexfloating],
    Cab: NDArray[np.complexfloating],
    Cbb: NDArray[np.complexfloating],
    *,
    rank: int | None,
    n_components: int,
    regularization: float,
) -> tuple[
    NDArray[np.floating],
    tuple[NDArray[np.floating], NDArray[np.floating]],
    tuple[NDArray[np.floating], NDArray[np.floating]],
    NDArray[np.integer],
]:
    """MIC singular components and channel-space projections, batched.

    Same batched shape contract as :func:`_canonical_coherency_components`.
    """
    real_aa = xp.real(Caa)
    real_bb = xp.real(Cbb)
    transform_aa, rank_a = _batched_inverse_square_root(
        real_aa, rank=rank, regularization=regularization
    )
    transform_bb, rank_b = _batched_inverse_square_root(
        real_bb, rank=rank, regularization=regularization
    )
    transformed = transform_aa @ xp.imag(Cab) @ transform_bb
    left, singular_values, right_h = xp.linalg.svd(transformed, full_matrices=False)
    left = left[..., :, :n_components]
    right = right_h.swapaxes(-1, -2)[..., :, :n_components]
    # MIC is a coherence in [0, 1]; clip roundoff excursions like the scalar
    # ``maximized_imaginary_coherency`` does.
    scores = xp.clip(singular_values[..., :n_components], 0.0, 1.0)
    filters_a = transform_aa @ left
    filters_b = transform_bb @ right
    patterns_a = real_aa @ filters_a
    patterns_b = real_bb @ filters_b
    # Singular vectors past the joint within-group rank span the whitening
    # transform's null space: their score is ~0 but the vectors are arbitrary.
    return _zero_unsupported_components(
        scores,
        (filters_a, filters_b),
        (patterns_a, patterns_b),
        xp.minimum(rank_a, rank_b),
    )


def _reshape(
    fourier_coefficients: NDArray[np.complexfloating],
) -> NDArray[np.complexfloating]:
    """Combine trials and tapers dimensions and move to last axis.

    Combine trials and tapers dimensions and move the combined dimension
    to the last axis position.

    Parameters
    ----------
    fourier_coefficients : array
        Shape (n_time_windows, n_trials, n_tapers, n_fft_samples, n_signals).
        Input Fourier coefficients.

    Returns
    -------
    fourier_coefficients : array
        Shape (n_time_windows, n_fft_samples, n_signals, n_trials * n_tapers).
        Reshaped Fourier coefficients.

    """
    (n_time_windows, _, _, n_fft_samples, n_signals) = fourier_coefficients.shape
    new_shape = (n_time_windows, -1, n_fft_samples, n_signals)
    return xp.moveaxis(fourier_coefficients.reshape(new_shape), 1, -1)


def _normalize_fourier_coefficients(
    fourier_coefficients: NDArray[np.complexfloating],
) -> NDArray[np.complexfloating]:
    """Normalize fourier coefficients by power within group.

    Parameters
    ----------
    fourier_coefficients : array
        Shape (n_time_windows, n_trials, n_tapers, n_fft_samples, n_signals).
        Input Fourier coefficients.

    Returns
    -------
    normalized_fourier_coefficients : array
        Shape (n_time_windows, n_fft_samples, n_signals, n_trials * n_tapers).
        Normalized Fourier coefficients.

    """
    U, _, V_transpose = xp.linalg.svd(_reshape(fourier_coefficients), full_matrices=False)
    phase_factor: NDArray[np.complexfloating] = xp.matmul(U, V_transpose)
    return phase_factor


def _estimate_canonical_coherence(
    normalized_fourier_coefficients1: NDArray[np.complexfloating],
    normalized_fourier_coefficients2: NDArray[np.complexfloating],
) -> NDArray[np.floating]:
    """Find maximum complex correlation between groups of signals.

    Find the maximum complex correlation between groups of signals
    at each time and frequency.

    Parameters
    ----------
    normalized_fourier_coefficients1 : array
        Shape (n_time_windows, n_fft_samples, n_signals, n_trials * n_tapers).
        First group of normalized coefficients.
    normalized_fourier_coefficients2 : array
        Shape (n_time_windows, n_fft_samples, n_signals, n_trials * n_tapers).
        Second group of normalized coefficients.

    Returns
    -------
    canonical_coherence : array, shape (n_time_windows, n_fft_samples)
        Canonical coherence values.

    """
    group_cross_spectrum = _complex_inner_product(
        normalized_fourier_coefficients1, normalized_fourier_coefficients2
    )
    return xp.linalg.svd(group_cross_spectrum, full_matrices=False, compute_uv=False)[..., 0]


def _global_coherence_components(
    block: NDArray[np.complexfloating], max_rank: int, use_eigh: bool
) -> tuple[NDArray[np.floating], NDArray[np.complexfloating]]:
    """Global-coherence fractions and vectors for one chunk of bins.

    Parameters
    ----------
    block : ndarray, shape (n_bins, n_signals, n_estimates)
        Per-bin coefficient matrices for this chunk.
    max_rank : int
        Number of strongest components to return.
    use_eigh : bool
        If True (``n_estimates >= n_signals``), diagonalize the
        ``(n_signals, n_signals)`` cross-spectral matrix with ``eigh``. If False
        (a *thin* matrix), use the economy SVD, which computes only the
        ``n_estimates`` non-trivial components rather than a large rank-deficient
        cross-spectral matrix.

    Returns
    -------
    fractions : ndarray, shape (n_bins, max_rank)
        Fraction of total coherent power per component, strongest first, in
        [0, 1]; NaN for a (near-)zero-power bin.
    vectors : ndarray, shape (n_bins, n_signals, max_rank)
        Global-coherence vectors (left singular vectors).
    """
    # Rescale each bin by its max magnitude first: the coherence fraction is
    # invariant to this, but summing squares of extreme-magnitude coefficients
    # would under/overflow to a false zero/inf (see the per-bin path). A
    # genuinely zero-power bin is flagged and returned as NaN.
    max_magnitude = xp.max(xp.abs(block), axis=(-2, -1), keepdims=True)
    is_zero_power = max_magnitude == 0
    scaled = block / xp.where(is_zero_power, 1, max_magnitude)
    # total_power is the squared Frobenius norm (== sum of squared singular
    # values); using it as the denominator keeps each fraction in [0, 1] exactly.
    total_power = xp.sum(xp.abs(scaled) ** 2, axis=(-2, -1))

    if use_eigh:
        # Eigenvalues of the Hermitian PSD cross-spectral matrix are the squared
        # singular values; eigenvectors are the left singular vectors.
        cross_spectral_matrix = xp.matmul(scaled, _conjugate_transpose(scaled))
        eigenvalues, eigenvectors = xp.linalg.eigh(cross_spectral_matrix)
        # eigh returns ascending order; take the strongest components first.
        component_power = xp.flip(eigenvalues, axis=-1)[..., :max_rank]
        vectors = xp.flip(eigenvectors, axis=-1)[..., :max_rank]
    else:
        # Thin matrix: the economy SVD returns descending singular values and the
        # left singular vectors directly, computing only n_estimates components.
        left_vectors, singular_values, _ = xp.linalg.svd(scaled, full_matrices=False)
        component_power = singular_values[..., :max_rank] ** 2
        vectors = left_vectors[..., :max_rank]

    safe_total = xp.where(total_power == 0, 1, total_power)
    # The cross-spectral matrix is PSD by construction (``scaled @ scaledᴴ``), so
    # any negative eigenvalue is round-off in eigh and is clipped to 0. (The SVD
    # path returns squared singular values, which are non-negative already, so
    # the clip is a no-op there.)
    fractions = xp.clip(component_power, 0.0, None) / safe_total[..., xp.newaxis]

    undefined = is_zero_power[..., 0, 0]
    fractions = xp.where(undefined[:, xp.newaxis], xp.nan, fractions)
    vectors = xp.where(undefined[:, xp.newaxis, xp.newaxis], xp.nan, vectors)
    return fractions, vectors


def _batched_global_coherence(
    fourier_coefficients: NDArray[np.complexfloating],
    max_rank: int,
    max_workspace_elements: int = GLOBAL_COHERENCE_BATCH_CHUNK_ELEMENTS,
) -> tuple[NDArray[np.floating], NDArray[np.complexfloating]]:
    """Global coherence for all time-frequency bins, batched over bins.

    Global coherence is defined as the eigenvalues of the per-bin cross-spectral
    matrix, normalized by their total (Cimenser et al. 2011). Diagonalizing all
    bins at once replaces the Python loop over bins and its per-bin device syncs.
    The eigenvalues equal the squared singular values used by the per-bin path,
    so the returned fractions match it to floating-point tolerance. The vectors
    are only defined up to a per-component phase where the components are
    distinct, and up to an arbitrary unitary rotation/permutation within any set
    of repeated (degenerate) components, so they need not match the per-bin path.

    Bins are processed in chunks taken from the original tensor (only each chunk
    is rearranged, never the whole array), and the chunk size is derived from the
    actual per-bin working set so peak memory stays bounded on both CPU and GPU.

    Parameters
    ----------
    fourier_coefficients : ndarray,
        shape (n_time_windows, n_trials, n_tapers, n_fft_samples, n_signals)
    max_rank : int
        Number of strongest components to return.

    Returns
    -------
    global_coherence : ndarray, shape (n_time_windows, n_fft_samples, max_rank)
    vectors : ndarray,
        shape (n_time_windows, n_fft_samples, n_signals, max_rank)
    """
    n_time, n_trials, n_tapers, n_fft, n_signals = fourier_coefficients.shape
    n_estimates = n_trials * n_tapers

    global_coherence = xp.empty((n_time, n_fft, max_rank))
    vectors = xp.empty((n_time, n_fft, n_signals, max_rank), dtype=xp.complex128)

    # Size the frequency chunk from the per-bin peak working set so memory stays
    # bounded regardless of the number of bins. Several coefficient-sized arrays
    # are live at once (the rearranged block, its rescaled copy, the conjugate
    # transpose fed to matmul, and the SVD's U/Vh factors on the thin path), plus
    # the decomposition of a min(n_signals, n_estimates)-square matrix and its
    # vectors; the 4x / 2x factors approximate that simultaneous footprint.
    decomposition_dim = min(n_signals, n_estimates)
    per_bin_elements = 4 * n_signals * n_estimates + 2 * decomposition_dim**2
    chunk = max(1, max_workspace_elements // per_bin_elements)
    use_eigh = n_estimates >= n_signals

    for time_ind in range(n_time):
        # View of this time slice as (n_fft, n_signals, n_trials, n_tapers); only
        # a chunk of frequencies is materialized (reshaped) at a time, so the
        # full-size transpose/copy of the tensor is never formed.
        time_slice = fourier_coefficients[time_ind].transpose(2, 3, 0, 1)
        for freq_start in range(0, n_fft, chunk):
            freq_stop = min(freq_start + chunk, n_fft)
            block = time_slice[freq_start:freq_stop].reshape(
                freq_stop - freq_start, n_signals, n_estimates
            )
            fractions, block_vectors = _global_coherence_components(block, max_rank, use_eigh)
            global_coherence[time_ind, freq_start:freq_stop] = fractions
            vectors[time_ind, freq_start:freq_stop] = block_vectors
    return global_coherence, vectors


def _estimate_global_coherence(
    fourier_coefficients: NDArray[np.complexfloating], max_rank: int = 1
) -> tuple[NDArray[np.floating], NDArray[np.complexfloating]]:
    """Estimate global coherence.

    Parameters
    ----------
    fourier_coefficients : ndarray, shape (n_signals, n_trials * n_tapers)
        The fourier coefficients for a given frequency across all channels
    max_rank : float, optional
        The maximum number of singular values to keep

    Returns
    -------
    global_coherence : ndarray, shape (max_rank,)
        The fraction of total coherent power per component (squared singular
        value divided by the sum of all squared singular values), in [0, 1],
        strongest component first. NaN for a (near-)zero-power bin.
    unnormalized_global_coherence : ndarray, shape (n_signals, max_rank)
        The global coherence vectors (left singular vectors)

    """
    n_signals, n_estimates = fourier_coefficients.shape
    # The coefficient matrix has at most min(n_signals, n_estimates) non-trivial
    # singular values, and svds requires a rank strictly below that minimum.
    n_components = min(n_signals, n_estimates)

    # Global coherence is the fraction of total coherent power in each component:
    # the eigenvalue of the cross-spectral matrix divided by the sum of all
    # eigenvalues (Cimenser et al. 2011). The eigenvalues are the squared
    # singular values of the coefficient matrix (up to the shared 1/n_estimates
    # factor), and their sum equals the squared Frobenius norm, so normalizing
    # by it makes the measure scale-invariant and bounded in [0, 1].
    #
    # Rescale by the maximum coefficient magnitude first. The fraction is
    # invariant to this rescaling, but computing the sum of squares directly on
    # extreme-magnitude coefficients would underflow (e.g. ~1e-200 -> 0, a false
    # zero-power bin) or overflow (~1e200 -> inf) and return NaN.
    max_magnitude = float(xp.max(xp.abs(fourier_coefficients)))
    if max_magnitude == 0:
        # Genuinely zero-power bin (e.g. a dead/flat channel): global coherence
        # is 0/0 and undefined. Return NaN rather than silently substituting 0,
        # mirroring coherency() / imaginary_coherence(); the caller warns once.
        return (
            xp.full(max_rank, xp.nan),
            xp.full((n_signals, max_rank), xp.nan, dtype=xp.complex128),
        )
    scaled_coefficients = fourier_coefficients / max_magnitude
    total_power = float(xp.sum(xp.abs(scaled_coefficients) ** 2))

    if max_rank >= n_components - 1:
        unnormalized_global_coherence, singular_values, _ = xp.linalg.svd(
            scaled_coefficients, full_matrices=False
        )
        global_coherence = singular_values[:max_rank] ** 2 / total_power
        unnormalized_global_coherence = unnormalized_global_coherence[:, :max_rank]
    else:
        unnormalized_global_coherence, singular_values, _ = svds(scaled_coefficients, max_rank)
        # svds does not guarantee the order of the returned singular values, so
        # sort strongest-first explicitly (rather than assuming ascending) and
        # apply the same ordering to the vectors, matching the dense (svd)
        # branch.
        order = xp.argsort(singular_values)[::-1]
        singular_values = singular_values[order]
        unnormalized_global_coherence = unnormalized_global_coherence[:, order]
        global_coherence = singular_values**2 / total_power

    return global_coherence, unnormalized_global_coherence
