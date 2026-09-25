"""Spectral Granger prediction from nonparametric VAR models of the cross-spectrum.

Kernels behind the spectral Granger measures of :class:`Connectivity`: each
factors (sub-)spectra with the Wilson minimum-phase decomposition, reads off the
transfer function and noise covariance, and decomposes predictive power by
frequency (Geweke 1982; Dhamala, Rangarajan & Ding 2008).
"""

import warnings
from collections.abc import Iterable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from spectral_connectivity._array_utils import (
    _complex_inner_product,
    _conjugate_transpose,
    _regularized_inverse,
    _squared_magnitude,
)
from spectral_connectivity._backend import ifft, xp
from spectral_connectivity.minimum_phase_decomposition import minimum_phase_decomposition
from spectral_connectivity.utils import to_numpy


def _estimate_noise_covariance(
    minimum_phase: NDArray[np.complexfloating],
) -> NDArray[np.floating]:
    """Estimate noise covariance non-parametrically from minimum phase factor.

    Given a matrix square root of the cross spectral matrix (
    minimum phase factor), non-parametrically estimate the noise covariance
    of a multivariate autoregressive model (MVAR).

    Parameters
    ----------
    minimum_phase : array, shape (n_time_windows, n_fft_samples, n_signals, n_signals)
        The matrix square root of a cross spectral matrix.

    Returns
    -------
    noise_covariance : array, shape (n_time_windows, n_signals, n_signals)
        The noise covariance of a MVAR model.

    References
    ----------
    .. [1] Dhamala, M., Rangarajan, G., and Ding, M. (2008). Analyzing
           information flow in brain networks with nonparametric Granger
           causality. NeuroImage 41, 354-362.

    """
    inverse_fourier_coefficients = ifft(minimum_phase, axis=-3).real
    return _complex_inner_product(
        inverse_fourier_coefficients[..., 0, :, :],
        inverse_fourier_coefficients[..., 0, :, :],
    ).real


def _estimate_transfer_function(
    minimum_phase: NDArray[np.complexfloating],
) -> NDArray[np.complexfloating]:
    """Estimate transfer function non-parametrically from minimum phase factor.

    Given a matrix square root of the cross spectral matrix (
    minimum phase factor), non-parametrically estimate the transfer
    function of a multivariate autoregressive model (MVAR).

    Parameters
    ----------
    minimum_phase : array, shape (n_time_windows, n_fft_samples, n_signals, n_signals)
        The matrix square root of a cross spectral matrix.

    Returns
    -------
    transfer_function : array
        Shape (n_time_windows, n_fft_samples, n_signals, n_signals).
        The transfer function of a MVAR model.

    References
    ----------
    .. [1] Dhamala, M., Rangarajan, G., and Ding, M. (2008). Analyzing
           information flow in brain networks with nonparametric Granger
           causality. NeuroImage 41, 354-362.

    """
    inverse_fourier_coefficients = ifft(minimum_phase, axis=-3).real
    H_0 = inverse_fourier_coefficients[..., 0:1, :, :]
    transfer_function: NDArray[np.complexfloating] = xp.matmul(
        minimum_phase, _regularized_inverse(H_0)
    )
    return transfer_function


def _granger_result_dtype(spectrum: NDArray[np.complexfloating]) -> np.dtype[Any]:
    """Real dtype at which the spectral Granger variants report their results.

    The Wilson factorization behind every variant runs at ``complex128`` or
    better whatever the spectrum's precision (see
    :func:`~spectral_connectivity.minimum_phase_decomposition.minimum_phase_decomposition`),
    and the pairwise variant and DTF report that working precision. The
    conditional and blockwise variants pre-allocate their output, so they use
    the same rule rather than downcasting to a ``complex64`` spectrum's
    ``float32``.
    """
    return xp.result_type(spectrum.real.dtype, xp.float64)


def _sanitized_nonnegative_granger(
    value: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Enforce the non-negativity invariant shared by every spectral Granger variant.

    Spectral Granger prediction is ``>= 0`` by definition -- it is the log-ratio
    of a total to an intrinsic spectral density, which is at least one. Two
    numerically-computed log-determinants can still differ by a tiny negative
    amount around a true zero (no causality) or, when the underlying
    factorization is degenerate, by a materially negative amount. Clip the
    roundoff band to exactly zero and mark materially-negative (invalid) values
    as NaN. NaN inputs pass through unchanged.
    """
    tolerance = 100 * xp.finfo(value.dtype).eps
    value = xp.where((value < 0) & (value > -tolerance), 0.0, value)
    return xp.where(value < 0, xp.nan, value)


def _estimate_predictive_power(
    total_power: NDArray[np.floating],
    rotated_covariance: NDArray[np.floating],
    transfer_function: NDArray[np.complexfloating],
) -> NDArray[np.floating]:
    """Estimate predictive power from total power and transfer function.

    Parameters
    ----------
    total_power : array_like
        Total power of signals.
    rotated_covariance : array_like
        Rotated noise covariance matrix.
    transfer_function : array_like
        Transfer function matrix.

    Returns
    -------
    array_like
        Predictive power values.

    """
    intrinsic_power = total_power[..., xp.newaxis] - rotated_covariance[
        ..., xp.newaxis, :, :
    ] * _squared_magnitude(transfer_function)
    intrinsic_power[intrinsic_power == 0] = xp.finfo(float).eps
    # A near-singular rotation can drive intrinsic_power negative; log() then
    # yields NaN, which is deliberately masked out below. Scope the warning
    # suppression to this operation rather than silencing it process-wide.
    with np.errstate(invalid="ignore", divide="ignore"):
        predictive_power = xp.log(total_power[..., xp.newaxis]) - xp.log(intrinsic_power)
    # A near-singular rotation can drive intrinsic_power above total_power,
    # giving a negative log-ratio; clip roundoff to zero and NaN the rest.
    return _sanitized_nonnegative_granger(predictive_power)


def _remove_instantaneous_causality(
    noise_covariance: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Remove instantaneous causality effects from noise covariance.

    Rotates the noise covariance so that the effect of instantaneous
    signals (like those caused by volume conduction) are removed.

    x -> y: var(x) - (cov(x,y) ** 2 / var(y))
    y -> x: var(y) - (cov(x,y) ** 2 / var(x))

    Parameters
    ----------
    noise_covariance : array, shape (..., n_signals, n_signals)
        Input noise covariance matrix.

    Returns
    -------
    rotated_noise_covariance : array, shape (..., n_signals, n_signals)
        The noise covariance without the instantaneous causality effects.

    """
    variance = xp.diagonal(noise_covariance, axis1=-1, axis2=-2)[..., xp.newaxis]
    return variance.swapaxes(-1, -2) - noise_covariance**2 / variance


def _estimate_spectral_granger_prediction(
    total_power: NDArray[np.floating],
    csm: NDArray[np.complexfloating],
    pairs: Iterable[tuple[int, int]] | NDArray[np.integer],
    minimum_phase_tolerance: float = 1e-8,
    minimum_phase_max_iterations: int = 500,
) -> NDArray[np.floating]:
    """
    Estimate spectral granger causality.

    Parameters
    ----------
    total_power : ndarray, shape (..., n_frequencies, n_signals)
        The total power of the signals.
    csm : ndarray, shape (..., n_frequencies, n_signals, n_signals)
        The cross spectral matrix of the signals.
    pairs : list of tuples
        The pairs of signals to estimate the spectral granger
        causality for.

    Returns
    -------
    predictive_power : ndarray, shape (..., n_frequencies, n_signals, n_signals)
        The spectral granger causality of the signals.
    """
    n_nonnegative = total_power.shape[-2] // 2 + 1
    total_power = total_power[..., :n_nonnegative, :]

    new_shape = list(csm.shape)
    new_shape[-3] = n_nonnegative
    predictive_power = xp.full(new_shape, xp.nan)

    for pair in pairs:
        pair_indices = xp.array(pair)[:, xp.newaxis]
        try:
            transfer_function, noise_covariance = _var_model_from_spectrum(
                csm[..., pair_indices, pair_indices.T],
                minimum_phase_tolerance=minimum_phase_tolerance,
                minimum_phase_max_iterations=minimum_phase_max_iterations,
            )
            predictive_power[..., pair_indices, pair_indices.T] = _estimate_predictive_power(
                total_power[..., pair_indices[:, 0]],
                _remove_instantaneous_causality(noise_covariance),
                transfer_function,
            )
        except np.linalg.LinAlgError:
            # Left NaN; the calling measure names the NaN pairs
            # (_warn_nan_granger_pairs).
            continue

    n_signals = csm.shape[-1]
    diagonal_ind = xp.diag_indices(n_signals)
    predictive_power[..., diagonal_ind[0], diagonal_ind[1]] = xp.nan

    return predictive_power


def _warn_nan_granger_pairs(
    result: NDArray[np.floating],
    measure: str,
    *,
    requested: NDArray[np.bool_] | None = None,
    names: NDArray[Any] | None = None,
    stacklevel: int = 4,
) -> None:
    """Warn once, naming the source -> target pairs whose Granger values failed.

    A pair whose factorization is singular or does not converge, or whose
    target has no power, is NaN at every frequency of the affected time
    window; the factorization marks it NaN without raising. Name those pairs
    so the user can find the offending channels. Isolated NaN bins (a
    degenerate bin of an otherwise valid factorization) are not reported here.

    Parameters
    ----------
    result : array, shape (..., n_frequencies, n_units, n_units)
        Granger result, ``[..., target, source]``; the diagonal is ignored.
    measure : str
        Name of the public measure, used in the message.
    requested : bool array, shape (n_units, n_units), optional
        Entries the caller computed; unrequested entries are NaN by design.
    names : array, shape (n_units,), optional
        Group labels; signals are named by their 0-based index otherwise.
    stacklevel : int, default=4
        Points the warning at the user's call through the measure and its
        ``_asnumpy`` wrapper; 3 for an undecorated measure.
    """
    n_units = result.shape[-1]
    # NaN at every frequency in at least one time window.
    failed = xp.all(xp.isnan(result), axis=-3).reshape(-1, n_units, n_units)
    is_failed = to_numpy(xp.any(failed, axis=0)) & ~np.eye(n_units, dtype=bool)
    if requested is not None:
        is_failed &= requested
    if not is_failed.any():
        return
    targets, sources = np.nonzero(is_failed)
    if names is None:
        listing = ", ".join(
            f"{source} -> {target}"
            for source, target in sorted(zip(sources, targets, strict=True))
        )
        unit = "signal"
        indexing = "0-based signal indices, "
    else:
        listing = ", ".join(
            f"{names[source].item()!r} -> {names[target].item()!r}"
            for source, target in sorted(zip(sources, targets, strict=True))
        )
        unit = "group"
        indexing = ""
    warnings.warn(
        f"{measure}: NaN at every frequency for {len(targets)} source -> target "
        f"{unit} pair(s): {listing} ({indexing}in at least one time window). The "
        "spectral factorization for those pairs was singular or did not "
        "converge, or the target has no power; this usually means duplicated, "
        "linearly dependent, or dead (zero-power) channels, so check those "
        "channels. If they are valid but highly correlated, a larger "
        "minimum_phase_max_iterations (a Connectivity argument) may let the "
        "factorization converge.",
        UserWarning,
        stacklevel=stacklevel,
    )


def _var_model_from_spectrum(
    csm: NDArray[np.complexfloating],
    *,
    minimum_phase_tolerance: float,
    minimum_phase_max_iterations: int,
) -> tuple[NDArray[np.complexfloating], NDArray[np.floating]]:
    """Wilson-factorize a cross-spectrum into a transfer function and noise covariance.

    Parameters
    ----------
    csm : array, shape (..., n_fft_samples, n_signals, n_signals)
        Two-sided cross-spectral matrix in standard FFT order.

    Returns
    -------
    transfer_function : array
        Shape ``(..., n_nonnegative_frequencies, n_signals, n_signals)``.
    noise_covariance : array, shape (..., n_signals, n_signals)
    """
    minimum_phase = minimum_phase_decomposition(
        csm,
        tolerance=minimum_phase_tolerance,
        max_iterations=minimum_phase_max_iterations,
        _warn_on_failure=False,
    )
    n_nonnegative = csm.shape[-3] // 2 + 1
    transfer = _estimate_transfer_function(minimum_phase)[..., :n_nonnegative, :, :]
    return transfer, _estimate_noise_covariance(minimum_phase)


def _estimate_conditional_spectral_granger_prediction(
    full_transfer: NDArray[np.complexfloating],
    full_covariance: NDArray[np.floating],
    reduced_inverse_transfer: NDArray[np.complexfloating],
    reduced_indices: NDArray[np.integer],
    target: int,
) -> NDArray[np.floating]:
    """Conditional spectral Granger ``source -> target | rest`` (Chen et al. 2006).

    ``full_transfer``/``full_covariance`` describe the full model of every
    signal and ``reduced_inverse_transfer`` is the inverse transfer function of
    the reduced model over ``reduced_indices`` (every signal but the source).

    The full-model innovations are first transformed so that the target's
    innovation is uncorrelated with all others (Geweke's normalization). The
    reduced model's innovation for the target then decomposes, through
    ``Q = G_ext^{-1} H``, into a term driven by the target's own innovation
    (the intrinsic spectrum) plus a positive semidefinite remainder driven by
    the other innovations. The measure is the log-ratio of the total to the
    intrinsic spectrum, so it is non-negative up to roundoff.

    Parameters
    ----------
    full_transfer : array, shape (..., n_frequencies, n_signals, n_signals)
    full_covariance : array, shape (..., n_signals, n_signals)
    reduced_inverse_transfer : array
        Shape ``(..., n_frequencies, n_signals - 1, n_signals - 1)``.
    reduced_indices : array, shape (n_signals - 1,)
        Full-model indices of the reduced model's signals, in reduced order.
    target : int
        Full-model index of the target signal; must be in ``reduced_indices``.

    Returns
    -------
    conditional_granger : array, shape (..., n_frequencies)
    """
    n_signals = full_covariance.shape[-1]
    reduced_indices = np.asarray(reduced_indices, dtype=int)
    (target_reduced,) = np.flatnonzero(reduced_indices == target)

    # Geweke normalization: P = I - coupling e_t^T removes the correlation of
    # every other innovation with the target's, leaving Sigma' = P Sigma P^T
    # with a zero target row/column off the diagonal and H' = H P^{-1}.
    identity = xp.eye(n_signals, dtype=full_covariance.dtype)
    unit_target = identity[:, target]
    coupling = (
        full_covariance[..., :, target] / full_covariance[..., target : target + 1, target]
    )
    coupling = coupling - unit_target  # zero at the target itself
    normalizer = identity - coupling[..., :, xp.newaxis] * unit_target
    inverse_normalizer = identity + coupling[..., :, xp.newaxis] * unit_target
    normalized_covariance = xp.matmul(
        xp.matmul(normalizer, full_covariance), normalizer.swapaxes(-1, -2)
    )
    normalized_covariance = (
        normalized_covariance + normalized_covariance.swapaxes(-1, -2)
    ) / 2.0
    normalized_transfer = xp.matmul(full_transfer, inverse_normalizer[..., xp.newaxis, :, :])

    # Target row of Q = G_ext^{-1} H', where G_ext embeds the reduced model's
    # inverse transfer function with an identity block for the omitted source.
    reduced_rows = xp.asarray(reduced_indices)
    q_target = xp.matmul(
        reduced_inverse_transfer[..., target_reduced : target_reduced + 1, :],
        normalized_transfer[..., reduced_rows, :],
    )  # (..., n_frequencies, 1, n_signals)
    total = xp.real(
        xp.matmul(
            xp.matmul(q_target, normalized_covariance.astype(q_target.dtype)[..., None, :, :]),
            _conjugate_transpose(q_target),
        )
    )[..., 0, 0]
    intrinsic = (
        _squared_magnitude(q_target[..., 0, target])
        * normalized_covariance[..., target : target + 1, target]
    )

    positive = (total > 0) & (intrinsic > 0)
    # NaN spectra come from a failed factorization, which the calling measure
    # reports by pair; warn here only about finite, non-positive spectra.
    if bool(xp.any(~positive & xp.isfinite(total) & xp.isfinite(intrinsic))):
        warnings.warn(
            "Conditional spectral Granger: the total or intrinsic innovation "
            "spectrum of the target was not positive at some time-frequency "
            "bins (a degenerate factorization, typically from near-singular "
            "conditioning). Those bins are returned as NaN. Consider increasing "
            "minimum_phase_max_iterations or checking for collinear channels.",
            UserWarning,
            stacklevel=3,
        )
    safe_total = xp.where(positive, total, 1.0)
    safe_intrinsic = xp.where(positive, intrinsic, 1.0)
    value = _sanitized_nonnegative_granger(xp.log(safe_total) - xp.log(safe_intrinsic))
    return xp.where(positive, value, xp.nan)


def _estimate_block_spectral_granger_prediction(
    csm: NDArray[np.complexfloating],
    first_indices: NDArray[np.integer],
    second_indices: NDArray[np.integer],
    *,
    minimum_phase_tolerance: float,
    minimum_phase_max_iterations: int,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Estimate block spectral Granger in both directions between two blocks.

    The subsystem ``[first, second]`` is factorized once. Because the
    spectral factorization with ``H(0) = I`` is unique, permuting its signals
    permutes the transfer function and noise covariance the same way, so the
    reverse direction reuses the same model with the blocks swapped.

    Parameters
    ----------
    csm : array, shape (..., n_fft_samples, n_signals, n_signals)
        Two-sided cross-spectral matrix in standard FFT order.
    first_indices, second_indices : array, shape (n_first,), (n_second,)
        Non-overlapping signal indices of the two blocks.

    Returns
    -------
    first_from_second : array, shape (..., n_nonnegative_frequencies)
        Influence ``second -> first``.
    second_from_first : array, shape (..., n_nonnegative_frequencies)
        Influence ``first -> second``.
    """
    combined = xp.asarray(np.concatenate((first_indices, second_indices)))
    subsystem = csm[..., combined[:, xp.newaxis], combined[xp.newaxis, :]]
    transfer, covariance = _var_model_from_spectrum(
        subsystem,
        minimum_phase_tolerance=minimum_phase_tolerance,
        minimum_phase_max_iterations=minimum_phase_max_iterations,
    )
    n_first = first_indices.size
    swapped = xp.asarray(
        np.concatenate((np.arange(n_first, combined.shape[0]), np.arange(n_first)))
    )

    def swap_blocks(matrix: NDArray[Any]) -> NDArray[Any]:
        return matrix[..., swapped[:, xp.newaxis], swapped[xp.newaxis, :]]

    return (
        _block_spectral_granger_from_model(subsystem, transfer, covariance, n_first),
        _block_spectral_granger_from_model(
            swap_blocks(subsystem),
            swap_blocks(transfer),
            swap_blocks(covariance),
            second_indices.size,
        ),
    )


def _block_spectral_granger_from_model(
    subsystem: NDArray[np.complexfloating],
    transfer: NDArray[np.complexfloating],
    covariance: NDArray[np.floating],
    n_target: int,
) -> NDArray[np.floating]:
    """Block spectral Granger from the trailing (source) to the leading (target) block.

    The subsystem is ordered ``[target, source]``. Its innovations are
    block-orthogonalized while preserving the target innovations, and the
    source contribution is removed from the target spectral block. The log
    determinant ratio of total to intrinsic target spectra is Geweke's
    multivariate spectral Granger measure.

    Parameters
    ----------
    subsystem : array, shape (..., n_fft_samples, n_sub, n_sub)
        Two-sided cross-spectral matrix of the ``[target, source]`` subsystem.
    transfer : array, shape (..., n_nonnegative_frequencies, n_sub, n_sub)
        Transfer function of the subsystem's VAR model.
    covariance : array, shape (..., n_sub, n_sub)
        Innovation covariance of the subsystem's VAR model.
    n_target : int
        Number of leading target signals.

    Returns
    -------
    block_granger : array, shape (..., n_nonnegative_frequencies)
    """
    n_nonnegative = transfer.shape[-3]
    covariance_xx = covariance[..., :n_target, :n_target]
    covariance_xy = covariance[..., :n_target, n_target:]
    covariance_yx = covariance[..., n_target:, :n_target]
    covariance_yy = covariance[..., n_target:, n_target:]
    conditional_source_covariance = covariance_yy - xp.matmul(
        xp.matmul(covariance_yx, _regularized_inverse(covariance_xx)),
        covariance_xy,
    )

    total_target_spectrum = subsystem[..., :n_nonnegative, :n_target, :n_target]
    source_transfer = transfer[..., :n_target, n_target:]
    source_contribution = xp.matmul(
        xp.matmul(source_transfer, conditional_source_covariance[..., xp.newaxis, :, :]),
        _conjugate_transpose(source_transfer),
    )
    intrinsic = total_target_spectrum - source_contribution
    # Remove tiny anti-Hermitian roundoff before determinant evaluation.
    intrinsic = (intrinsic + _conjugate_transpose(intrinsic)) / 2.0
    hermitian_total_target_spectrum = (
        total_target_spectrum + _conjugate_transpose(total_target_spectrum)
    ) / 2.0
    # A failed factorization leaves NaN in ``intrinsic``, and a target block
    # with no power is singular (both log-determinants -inf, their difference
    # NaN). The calling measure reports those NaN pairs, so the floating-point
    # flags some LAPACK builds (OpenBLAS) raise for them are silenced here.
    with np.errstate(divide="ignore", invalid="ignore"):
        _, total_logdet = xp.linalg.slogdet(hermitian_total_target_spectrum)
        _, intrinsic_logdet = xp.linalg.slogdet(intrinsic)
        value = _sanitized_nonnegative_granger(xp.real(total_logdet - intrinsic_logdet))
    # ``intrinsic`` is a difference of spectral blocks and is only guaranteed
    # positive-definite in exact arithmetic; near-degenerate conditioning can
    # make it (or the total spectrum) indefinite/singular, in which case the
    # log-determinant ratio is physically meaningless. Test positive-definiteness
    # directly via the smallest eigenvalue of these Hermitian matrices -- a
    # determinant-sign test would miss an even number of negative eigenvalues.
    # Return NaN (and warn) rather than a plausible but wrong finite influence.
    # A value that is already NaN (a failed factorization, or a target with no
    # power) is reported by pair by the calling measure, so it is not counted.
    smallest_total_eigenvalue = xp.linalg.eigvalsh(hermitian_total_target_spectrum)[..., 0]
    smallest_intrinsic_eigenvalue = xp.linalg.eigvalsh(intrinsic)[..., 0]
    positive_definite = (smallest_total_eigenvalue > 0) & (smallest_intrinsic_eigenvalue > 0)
    if bool(xp.any(~positive_definite & ~xp.isnan(value))):
        warnings.warn(
            "Block spectral Granger: the intrinsic or total target spectrum was "
            "not positive-definite at some time-frequency bins (typically from "
            "near-singular conditioning after removing the source block). Those "
            "bins are returned as NaN. Consider increasing "
            "minimum_phase_max_iterations or checking for collinear channels.",
            UserWarning,
            stacklevel=3,
        )
    return xp.where(positive_definite, value, xp.nan)


def _estimate_subset_spectral_granger_prediction(
    total_power: NDArray[np.floating],
    pair_csm: NDArray[np.complexfloating],
    pairs: NDArray[np.integer],
    n_signals: int,
    minimum_phase_tolerance: float = 1e-8,
    minimum_phase_max_iterations: int = 500,
) -> NDArray[np.floating]:
    """Estimate selected pairwise Granger values from compact 2-by-2 spectra.

    ``pair_csm`` has shape ``(..., n_pairs, n_frequencies, 2, 2)``. Keeping the
    pair axis as a batch dimension avoids allocating a full signal-by-signal CSM
    with uninitialized entries merely to consume its requested 2-by-2 slices.
    """
    pair_indices = xp.asarray(pairs, dtype=int)
    one_sided_power = total_power[..., : total_power.shape[-2] // 2 + 1, :]

    # Gather the two powers for every pair, then move pair before frequency to
    # match pair_csm's (..., pair, frequency, 2) batch layout.
    pair_power = one_sided_power[..., pair_indices]
    pair_power = xp.moveaxis(pair_power, -2, -3)

    transfer_function, noise_covariance = _var_model_from_spectrum(
        pair_csm,
        minimum_phase_tolerance=minimum_phase_tolerance,
        minimum_phase_max_iterations=minimum_phase_max_iterations,
    )
    pair_predictive_power = _estimate_predictive_power(
        pair_power,
        _remove_instantaneous_causality(noise_covariance),
        transfer_function,
    )

    output_shape = (*one_sided_power.shape, n_signals)
    predictive_power = xp.full(output_shape, xp.nan)
    for pair_number, pair in enumerate(pair_indices):
        matrix_indices = pair[:, xp.newaxis]
        predictive_power[..., matrix_indices, matrix_indices.T] = xp.take(
            pair_predictive_power, pair_number, axis=-4
        )
    diagonal_indices = xp.diag_indices(n_signals)
    predictive_power[..., diagonal_indices[0], diagonal_indices[1]] = xp.nan
    return predictive_power
