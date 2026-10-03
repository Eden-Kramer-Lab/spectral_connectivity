"""Spectral Granger prediction from nonparametric VAR models of the cross-spectrum.

Kernels behind the spectral Granger measures of :class:`Connectivity`: each
factors (sub-)spectra with the Wilson minimum-phase decomposition, reads off the
transfer function and noise covariance, and decomposes predictive power by
frequency (Geweke 1982; Dhamala, Rangarajan & Ding 2008).

Signal-by-signal arrays here (the transfer function and the pairwise, subset,
conditional, and blockwise Granger matrices) are indexed
``[..., target, source]``: ``[..., i, j]`` is ``j -> i``. Per-pair kernels
return ``(..., n_frequencies)`` values whose direction their docstrings name.
The public :class:`Connectivity` methods return ``[..., source, target]``.
"""

import warnings
from collections.abc import Iterable, Sequence
from itertools import combinations
from typing import Any

import numpy as np
from numpy.typing import NDArray

from spectral_connectivity._array_utils import (
    _batched_eigvalsh,
    _conjugate_transpose,
    _regularized_inverse,
    _squared_magnitude,
)
from spectral_connectivity._backend import ON_GPU, xp
from spectral_connectivity.minimum_phase_decomposition import minimum_phase_decomposition
from spectral_connectivity.utils import stacklevel_outside_package, to_numpy


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
    zero_lag = _zero_lag_coefficient(minimum_phase)
    noise_covariance: NDArray[np.floating] = xp.matmul(zero_lag, zero_lag.swapaxes(-1, -2))
    return noise_covariance


def _zero_lag_coefficient(
    minimum_phase: NDArray[np.complexfloating],
) -> NDArray[np.floating]:
    """Lag-0 coefficient of the minimum-phase factor, shape (..., n_signals, n_signals).

    The inverse DFT at lag 0 is the mean over frequencies, so no full inverse
    transform is needed; ``minimum_phase`` must therefore hold all
    ``n_fft_samples`` bins, not a frequency slice. ``.real`` discards an
    imaginary part that is rounding-level for the factor of a real process.
    """
    zero_lag: NDArray[np.floating] = xp.mean(minimum_phase, axis=-3).real
    return zero_lag


def _estimate_transfer_function(
    minimum_phase: NDArray[np.complexfloating],
    n_frequencies: int,
) -> NDArray[np.complexfloating]:
    """Estimate transfer function non-parametrically from minimum phase factor.

    Given a matrix square root of the cross spectral matrix (
    minimum phase factor), non-parametrically estimate the transfer
    function of a multivariate autoregressive model (MVAR).

    Parameters
    ----------
    minimum_phase : array, shape (n_time_windows, n_fft_samples, n_signals, n_signals)
        The matrix square root of a cross spectral matrix.
    n_frequencies : int
        Return only the first ``n_frequencies`` bins (e.g. the non-negative
        frequencies).

    Returns
    -------
    transfer_function : array
        Shape (n_time_windows, n_frequencies, n_signals, n_signals). The
        transfer function of a MVAR model; its lag-0 normalization always uses
        all bins.

    References
    ----------
    .. [1] Dhamala, M., Rangarajan, G., and Ding, M. (2008). Analyzing
           information flow in brain networks with nonparametric Granger
           causality. NeuroImage 41, 354-362.

    """
    H_0 = _zero_lag_coefficient(minimum_phase)[..., xp.newaxis, :, :]
    transfer_function: NDArray[np.complexfloating] = xp.matmul(
        minimum_phase[..., :n_frequencies, :, :], _regularized_inverse(H_0)
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
    intrinsic_power = xp.where(intrinsic_power == 0, xp.finfo(float).eps, intrinsic_power)
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


# Most spectral elements (sub-spectra x FFT bins x signals**2) one GPU Wilson
# factorization stacks across signal (or group) pairs, bounding working memory
# (64 MiB per complex128 working array). One pair alone is launch-bound there.
GRANGER_GPU_BATCH_MAX_WORKSPACE_ELEMENTS = 2**22


def _pairs_per_wilson_batch(elements_per_pair: int, *, cpu_pairs: int = 1) -> int:
    """Number of signal or group pairs to Wilson-factorize per call.

    Parameters
    ----------
    elements_per_pair : int
        Spectral elements one pair's sub-spectrum holds (leading batch axes x
        FFT bins x sub-system signals squared).
    cpu_pairs : int, default 1
        Pairs per call on the CPU, where factoring one pair at a time measured
        fastest for the pairwise and blockwise measures.

    Returns
    -------
    int
        ``cpu_pairs`` on the CPU; on the GPU as many pairs as fit in
        ``GRANGER_GPU_BATCH_MAX_WORKSPACE_ELEMENTS``, and at least 1.
    """
    if not ON_GPU:
        return cpu_pairs
    return max(1, GRANGER_GPU_BATCH_MAX_WORKSPACE_ELEMENTS // elements_per_pair)


def _pair_spectral_granger(
    pair_power: NDArray[np.floating],
    pair_csm: NDArray[np.complexfloating],
    *,
    minimum_phase_tolerance: float,
    minimum_phase_max_iterations: int,
) -> NDArray[np.floating]:
    """Pairwise spectral Granger of a stack of 2-by-2 sub-spectra.

    Each sub-spectrum is a separate unit of the Wilson factorization, which
    tracks convergence per unit, so stacking pairs does not change any pair's
    iterates.

    Parameters
    ----------
    pair_power : array, shape (..., n_pairs, n_nonnegative_frequencies, 2)
        Total power of each pair's two signals.
    pair_csm : array, shape (..., n_pairs, n_fft_samples, 2, 2)
        Two-sided cross-spectral matrix of each pair in standard FFT order.

    Returns
    -------
    predictive_power : array, shape (..., n_pairs, n_nonnegative_frequencies, 2, 2)
        ``[..., i, j]`` is ``j -> i`` within each pair; the diagonal is not
        meaningful.
    """
    transfer_function, noise_covariance = _var_model_from_spectrum(
        pair_csm,
        minimum_phase_tolerance=minimum_phase_tolerance,
        minimum_phase_max_iterations=minimum_phase_max_iterations,
    )
    return _estimate_predictive_power(
        pair_power,
        _remove_instantaneous_causality(noise_covariance),
        transfer_function,
    )


def _scatter_pairwise_granger(
    predictive_power: NDArray[np.floating],
    total_power: NDArray[np.floating],
    csm: NDArray[np.complexfloating],
    pairs: NDArray[np.integer],
    *,
    minimum_phase_tolerance: float,
    minimum_phase_max_iterations: int,
) -> None:
    """Factor ``pairs`` together and write their Granger values in place.

    Parameters
    ----------
    predictive_power : array, shape (..., n_nonnegative_frequencies, n_signals, n_signals)
        Output, ``[..., target, source]``; updated in place.
    total_power : array, shape (..., n_nonnegative_frequencies, n_signals)
    csm : array, shape (..., n_fft_samples, n_signals, n_signals)
    pairs : host int array, shape (n_pairs, 2)
        Distinct signal pairs.
    """
    device_pairs = xp.asarray(pairs)
    rows = device_pairs[:, :, xp.newaxis]
    columns = device_pairs[:, xp.newaxis, :]
    try:
        value = _pair_spectral_granger(
            xp.moveaxis(total_power[..., device_pairs], -2, -3),
            xp.moveaxis(csm[..., rows, columns], -3, -4),
            minimum_phase_tolerance=minimum_phase_tolerance,
            minimum_phase_max_iterations=minimum_phase_max_iterations,
        )
    except np.linalg.LinAlgError:
        # A lone pair is left NaN; the calling measure names the NaN pairs
        # (_warn_nan_granger_pairs). Retry a stack pair by pair so one failing
        # pair does not take the others with it (warnings the failed stack
        # already issued can repeat). NumPy raises LinAlgError, as does CuPy
        # under ``cupyx.errstate(linalg="raise")``.
        if pairs.shape[0] > 1:
            for pair in pairs:
                _scatter_pairwise_granger(
                    predictive_power,
                    total_power,
                    csm,
                    pair[np.newaxis],
                    minimum_phase_tolerance=minimum_phase_tolerance,
                    minimum_phase_max_iterations=minimum_phase_max_iterations,
                )
        return
    predictive_power[..., rows, columns] = xp.moveaxis(value, -4, -3)


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

    pair_array = np.asarray(list(pairs), dtype=int).reshape(-1, 2)
    pairs_per_chunk = _pairs_per_wilson_batch(4 * int(np.prod(csm.shape[:-2])))
    for start in range(0, pair_array.shape[0], pairs_per_chunk):
        _scatter_pairwise_granger(
            predictive_power,
            total_power,
            csm,
            pair_array[start : start + pairs_per_chunk],
            minimum_phase_tolerance=minimum_phase_tolerance,
            minimum_phase_max_iterations=minimum_phase_max_iterations,
        )

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
        stacklevel=stacklevel_outside_package(),
    )


def _factorize_spectrum(
    csm: NDArray[np.complexfloating],
    *,
    minimum_phase_tolerance: float,
    minimum_phase_max_iterations: int,
    warn_on_failure: bool = True,
) -> NDArray[np.complexfloating]:
    """Wilson minimum-phase factor of a two-sided cross-spectrum.

    Every factorization behind the directed measures, the full model cached on
    :class:`Connectivity` as well as the reduced and pairwise models, goes
    through here and :func:`_var_model_from_factor`, so they share one binding
    of the factorization and of the transfer-function and noise-covariance
    estimators.
    """
    return minimum_phase_decomposition(
        csm,
        tolerance=minimum_phase_tolerance,
        max_iterations=minimum_phase_max_iterations,
        _warn_on_failure=warn_on_failure,
    )


def _var_model_from_factor(
    minimum_phase: NDArray[np.complexfloating],
) -> tuple[NDArray[np.complexfloating], NDArray[np.floating]]:
    """Transfer function and noise covariance of a two-sided minimum-phase factor.

    Parameters
    ----------
    minimum_phase : array, shape (..., n_fft_samples, n_signals, n_signals)
        Minimum-phase factor of a two-sided cross-spectrum in standard FFT order.

    Returns
    -------
    transfer_function : array
        Shape ``(..., n_nonnegative_frequencies, n_signals, n_signals)``.
    noise_covariance : array, shape (..., n_signals, n_signals)
    """
    n_nonnegative = minimum_phase.shape[-3] // 2 + 1
    transfer = _estimate_transfer_function(minimum_phase, n_nonnegative)
    return transfer, _estimate_noise_covariance(minimum_phase)


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
    return _var_model_from_factor(
        _factorize_spectrum(
            csm,
            minimum_phase_tolerance=minimum_phase_tolerance,
            minimum_phase_max_iterations=minimum_phase_max_iterations,
            warn_on_failure=False,
        )
    )


def _estimate_conditional_spectral_granger_prediction(
    full_transfer: NDArray[np.complexfloating],
    full_covariance: NDArray[np.floating],
    reduced_inverse_transfer: NDArray[np.complexfloating],
    reduced_indices: NDArray[np.integer],
    target: int,
) -> tuple[NDArray[np.floating], NDArray[np.bool_]]:
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
    degenerate : bool array, shape ()
        Whether the total or intrinsic spectrum was finite but not positive at
        some bin (returned as NaN there). It stays on the device so that the
        caller can test it once for all pairs rather than synchronizing per pair.
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
    # reports by pair; flag here only finite, non-positive spectra.
    degenerate: NDArray[np.bool_] = xp.asarray(
        xp.any(~positive & xp.isfinite(total) & xp.isfinite(intrinsic))
    )
    safe_total = xp.where(positive, total, 1.0)
    safe_intrinsic = xp.where(positive, intrinsic, 1.0)
    value = _sanitized_nonnegative_granger(xp.log(safe_total) - xp.log(safe_intrinsic))
    return xp.where(positive, value, xp.nan), degenerate


def _estimate_all_conditional_spectral_granger(
    spectrum: NDArray[np.complexfloating],
    full_transfer: NDArray[np.complexfloating],
    full_covariance: NDArray[np.floating],
    *,
    minimum_phase_tolerance: float,
    minimum_phase_max_iterations: int,
) -> NDArray[np.floating]:
    """Conditional spectral Granger for every ordered pair of three or more signals.

    Parameters
    ----------
    spectrum : array, shape (..., n_fft_samples, n_signals, n_signals)
        Two-sided cross-spectral matrix in standard FFT order.
    full_transfer : array, shape (..., n_nonnegative_frequencies, n_signals, n_signals)
        Transfer function of the full model of every signal.
    full_covariance : array, shape (..., n_signals, n_signals)
        Noise covariance of the full model.

    Returns
    -------
    conditional_granger : array, shape (..., n_nonnegative_frequencies, n_signals, n_signals)
        ``[..., target, source]`` is ``source -> target`` conditioned on the
        other signals; the diagonal is NaN.
    """
    n_signals = spectrum.shape[-1]
    n_nonnegative = spectrum.shape[-3] // 2 + 1
    result = xp.full(
        (*spectrum.shape[:-3], n_nonnegative, n_signals, n_signals),
        xp.nan,
        dtype=_granger_result_dtype(spectrum),
    )
    all_indices = np.arange(n_signals)
    degenerate = xp.asarray(False)
    for source in range(n_signals):
        reduced_indices = all_indices[all_indices != source]
        # Index the device spectrum with a device index array; the host copy is
        # what the conditional estimator's bookkeeping consumes.
        device_indices = xp.asarray(reduced_indices)
        reduced_transfer, _ = _var_model_from_spectrum(
            spectrum[..., device_indices[:, xp.newaxis], device_indices[xp.newaxis, :]],
            minimum_phase_tolerance=minimum_phase_tolerance,
            minimum_phase_max_iterations=minimum_phase_max_iterations,
        )
        reduced_inverse_transfer = _regularized_inverse(reduced_transfer)
        for target in range(n_signals):
            if target == source:
                continue
            result[..., target, source], pair_degenerate = (
                _estimate_conditional_spectral_granger_prediction(
                    full_transfer,
                    full_covariance,
                    reduced_inverse_transfer,
                    reduced_indices,
                    target,
                )
            )
            degenerate = degenerate | pair_degenerate
    # One synchronization for all pairs rather than one per pair.
    if bool(degenerate):
        warnings.warn(
            "Conditional spectral Granger: the total or intrinsic innovation "
            "spectrum of the target was not positive at some time-frequency "
            "bins (a degenerate factorization, typically from near-singular "
            "conditioning). Those bins are returned as NaN. Consider increasing "
            "minimum_phase_max_iterations or checking for collinear channels.",
            UserWarning,
            stacklevel=stacklevel_outside_package(),
        )
    return result


def _estimate_blockwise_spectral_granger(
    spectrum: NDArray[np.complexfloating],
    group_indices: Sequence[NDArray[np.integer]],
    *,
    minimum_phase_tolerance: float,
    minimum_phase_max_iterations: int,
) -> NDArray[np.floating]:
    """Block spectral Granger between every pair of signal groups.

    Parameters
    ----------
    spectrum : array, shape (..., n_fft_samples, n_signals, n_signals)
        Two-sided cross-spectral matrix in standard FFT order.
    group_indices : sequence of int arrays
        Signal indices of each group.

    Returns
    -------
    block_granger : array, shape (..., n_nonnegative_frequencies, n_groups, n_groups)
        ``[..., target, source]`` is ``source -> target``; the diagonal is NaN.
    """
    n_groups = len(group_indices)
    n_nonnegative = spectrum.shape[-3] // 2 + 1
    result = xp.full(
        (*spectrum.shape[:-3], n_nonnegative, n_groups, n_groups),
        xp.nan,
        dtype=_granger_result_dtype(spectrum),
    )
    # One factorization per unordered group pair supplies both directions.
    # Group pairs with the same block sizes are factored together (one per call
    # on the CPU, bounded chunks on the GPU; see _pairs_per_wilson_batch).
    by_sizes: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for first, second in combinations(range(n_groups), 2):
        sizes = (group_indices[first].size, group_indices[second].size)
        by_sizes.setdefault(sizes, []).append((first, second))
    for (n_first, n_second), group_pairs in by_sizes.items():
        n_sub = n_first + n_second
        per_chunk = _pairs_per_wilson_batch(int(np.prod(spectrum.shape[:-2])) * n_sub * n_sub)
        for start in range(0, len(group_pairs), per_chunk):
            firsts, seconds = np.array(group_pairs[start : start + per_chunk]).T
            first_from_second, second_from_first = _estimate_block_spectral_granger_prediction(
                spectrum,
                np.stack([group_indices[first] for first in firsts]),
                np.stack([group_indices[second] for second in seconds]),
                minimum_phase_tolerance=minimum_phase_tolerance,
                minimum_phase_max_iterations=minimum_phase_max_iterations,
            )
            firsts_device, seconds_device = xp.asarray(firsts), xp.asarray(seconds)
            result[..., firsts_device, seconds_device] = xp.moveaxis(first_from_second, -2, -1)
            result[..., seconds_device, firsts_device] = xp.moveaxis(second_from_first, -2, -1)
    return result


def _estimate_block_spectral_granger_prediction(
    csm: NDArray[np.complexfloating],
    first_indices: NDArray[np.integer],
    second_indices: NDArray[np.integer],
    *,
    minimum_phase_tolerance: float,
    minimum_phase_max_iterations: int,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Estimate block spectral Granger in both directions for a stack of block pairs.

    Each subsystem ``[first, second]`` is factorized once. Because the
    spectral factorization with ``H(0) = I`` is unique, permuting its signals
    permutes the transfer function and noise covariance the same way, so the
    reverse direction reuses the same model with the blocks swapped.

    Parameters
    ----------
    csm : array, shape (..., n_fft_samples, n_signals, n_signals)
        Two-sided cross-spectral matrix in standard FFT order.
    first_indices, second_indices : host int arrays
        Shapes ``(n_block_pairs, n_first)`` and ``(n_block_pairs, n_second)``:
        non-overlapping signal indices of the two blocks of each block pair.

    Returns
    -------
    first_from_second : array, shape (..., n_block_pairs, n_nonnegative_frequencies)
        Influence ``second -> first``.
    second_from_first : array, shape (..., n_block_pairs, n_nonnegative_frequencies)
        Influence ``first -> second``.
    """
    combined = xp.asarray(np.concatenate((first_indices, second_indices), axis=1))
    # (..., n_fft_samples, n_block_pairs, n_sub, n_sub) -> block pair before frequency.
    subsystem = xp.moveaxis(
        csm[..., combined[:, :, xp.newaxis], combined[:, xp.newaxis, :]], -3, -4
    )
    transfer, covariance = _var_model_from_spectrum(
        subsystem,
        minimum_phase_tolerance=minimum_phase_tolerance,
        minimum_phase_max_iterations=minimum_phase_max_iterations,
    )
    n_first = first_indices.shape[1]
    swapped = xp.asarray(
        np.concatenate((np.arange(n_first, combined.shape[1]), np.arange(n_first)))
    )

    def swap_blocks(matrix: NDArray[Any]) -> NDArray[Any]:
        return matrix[..., swapped[:, xp.newaxis], swapped[xp.newaxis, :]]

    return (
        _block_spectral_granger_from_model(subsystem, transfer, covariance, n_first),
        _block_spectral_granger_from_model(
            swap_blocks(subsystem),
            swap_blocks(transfer),
            swap_blocks(covariance),
            second_indices.shape[1],
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
    smallest_total_eigenvalue = _batched_eigvalsh(hermitian_total_target_spectrum)[..., 0]
    smallest_intrinsic_eigenvalue = _batched_eigvalsh(intrinsic)[..., 0]
    positive_definite = (smallest_total_eigenvalue > 0) & (smallest_intrinsic_eigenvalue > 0)
    if bool(xp.any(~positive_definite & ~xp.isnan(value))):
        warnings.warn(
            "Block spectral Granger: the intrinsic or total target spectrum was "
            "not positive-definite at some time-frequency bins (typically from "
            "near-singular conditioning after removing the source block). Those "
            "bins are returned as NaN. Consider increasing "
            "minimum_phase_max_iterations or checking for collinear channels.",
            UserWarning,
            stacklevel=stacklevel_outside_package(),
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

    # All pairs in one factorization on the CPU; bounded chunks on the GPU.
    n_pairs = pair_csm.shape[-4]
    pairs_per_chunk = _pairs_per_wilson_batch(pair_csm.size // n_pairs, cpu_pairs=n_pairs)
    pair_predictive_power = xp.concatenate(
        [
            _pair_spectral_granger(
                pair_power[..., start : start + pairs_per_chunk, :, :],
                pair_csm[..., start : start + pairs_per_chunk, :, :, :],
                minimum_phase_tolerance=minimum_phase_tolerance,
                minimum_phase_max_iterations=minimum_phase_max_iterations,
            )
            for start in range(0, n_pairs, pairs_per_chunk)
        ],
        axis=-4,
    )

    # A pair requested more than once, in either order, writes the same
    # entries; keep its last occurrence, which a pair-by-pair write would leave.
    host_pairs = to_numpy(pairs)
    _, last_in_reversed = np.unique(
        np.sort(host_pairs, axis=1)[::-1], axis=0, return_index=True
    )
    kept = np.sort(host_pairs.shape[0] - 1 - last_in_reversed)
    kept_pairs = xp.asarray(host_pairs[kept])

    output_shape = (*one_sided_power.shape, n_signals)
    predictive_power = xp.full(output_shape, xp.nan)
    if kept.size < n_pairs:
        pair_predictive_power = xp.take(pair_predictive_power, xp.asarray(kept), axis=-4)
    predictive_power[..., kept_pairs[:, :, xp.newaxis], kept_pairs[:, xp.newaxis, :]] = (
        xp.moveaxis(pair_predictive_power, -4, -3)
    )
    diagonal_indices = xp.diag_indices(n_signals)
    predictive_power[..., diagonal_indices[0], diagonal_indices[1]] = xp.nan
    return predictive_power
