"""Analytic VAR oracle helpers shared by the test modules.

For a stable VAR ``x(t) = sum_k A_k x(t - k) + e(t)`` with innovation
covariance ``Sigma``, the transfer function and cross-spectrum are known
exactly; these helpers build them and Fourier coefficients that reproduce
them, so a ``Connectivity`` can be fed an exact spectrum (see
``tests/test_directed_measures_oracle.py``).
"""

import numpy as np
from scipy.linalg import solve_discrete_are


def _analytic_var(coefficients, noise_covariance, n_fft):
    """Return (A(f), H(f), S(f)) on the full FFT grid for a VAR.

    coefficients : (n_lags, n_signals, n_signals) with the convention
    ``x(t) = sum_k coefficients[k] x(t - (k + 1)) + e(t)`` (matching
    ``simulate.simulate_MVAR``).
    """
    n_lags, n_signals, _ = coefficients.shape
    omega = 2 * np.pi * np.arange(n_fft) / n_fft
    A = np.tile(np.eye(n_signals, dtype=complex), (n_fft, 1, 1))
    for lag in range(n_lags):
        A -= coefficients[lag][None] * np.exp(-1j * omega * (lag + 1))[:, None, None]
    H = np.linalg.inv(A)
    S = H @ noise_covariance.astype(complex) @ H.conj().swapaxes(-1, -2)
    return A, H, S


def _fourier_coefficients_with_cross_spectrum(S, conjugate_symmetric=False):
    """Fourier coefficients whose expected cross-spectrum is exactly ``S``.

    With ``S = L L^H`` (Cholesky) and ``n_tapers = n_signals``, taper ``k`` set to
    ``sqrt(n_signals) * L[:, k]`` makes the taper-mean of the outer products equal
    ``L L^H = S`` exactly. Shape: (1, 1, n_signals, n_fft, n_signals).

    The analytic ``S`` is conjugate-symmetric only up to rounding, so the Wilson
    factorization takes its two-sided path. With ``conjugate_symmetric=True`` the
    coefficients are instead mirrored from the non-negative frequencies, with a
    real zero and Nyquist frequency, as for real-valued signals; the
    cross-spectrum is then exactly conjugate-symmetric and the factorization
    takes its half-spectrum path.
    """
    n_fft, n_signals, _ = S.shape
    L = np.linalg.cholesky(S)  # (n_fft, n_signals, n_signals), lower-triangular
    # taper axis <- columns of L; scale so the taper-mean reproduces S.
    fc = np.sqrt(n_signals) * np.moveaxis(L, -1, -2)  # (n_fft, taper, signal)
    fc = np.moveaxis(fc, 0, 1)  # (taper, n_fft, signal)
    if conjugate_symmetric:
        fc[:, 0] = fc[:, 0].real
        if n_fft % 2 == 0:
            fc[:, n_fft // 2] = fc[:, n_fft // 2].real
        fc[:, n_fft // 2 + 1 :] = fc[:, 1 : (n_fft + 1) // 2][:, ::-1].conj()
    return fc[None, None]  # (1, 1, n_tapers, n_fft, n_signals)


def _analytic_directed_measures(A, H, noise_covariance):
    """Closed-form directed measures of a VAR on the non-negative FFT grid.

    Results are in the native ``[target, source]`` layout (see the module
    docstring of ``tests/test_directed_measures_oracle.py``). ``A`` and ``H``
    are on the full FFT grid. Returns squared DTF, PDC, DC, gPDC, and dDTF. DC
    and gPDC weight by the innovation variances ``sigma``, the diagonal of
    ``noise_covariance``; DC is the directed coherence of the model only when
    that matrix is diagonal.
    """
    n_non_negative = A.shape[0] // 2 + 1
    A, H = A[:n_non_negative], H[:n_non_negative]
    sigma = np.diag(noise_covariance)
    H2, A2 = np.abs(H) ** 2, np.abs(A) ** 2
    # DTF / DC normalize each target row by its total inflow over sources.
    dtf = H2 / H2.sum(axis=-1, keepdims=True)
    dc = sigma * H2 / (sigma * H2).sum(axis=-1, keepdims=True)
    # PDC / gPDC normalize each source column by its total outflow over targets.
    pdc = A2 / A2.sum(axis=-2, keepdims=True)
    weighted = A2 / sigma[:, np.newaxis]
    gpdc = weighted / weighted.sum(axis=-2, keepdims=True)
    # dDTF (Korzeniewska et al. 2003): chi^2_ij = ffDTF^2_ij * kappa^2_ij, the
    # full-frequency DTF (inflow summed over sources and frequencies) times the
    # squared partial coherence from the inverse spectral matrix
    # G = A^H Sigma^-1 A, kappa^2_ij = |G_ij|^2 / (G_ii G_jj).
    full_frequency_dtf = H2 / H2.sum(axis=(-1, -3), keepdims=True)
    inverse_spectrum = np.conj(np.swapaxes(A, -1, -2)) @ np.linalg.inv(noise_covariance) @ A
    g_diagonal = np.real(np.diagonal(inverse_spectrum, axis1=-2, axis2=-1))
    partial_coherence = np.abs(inverse_spectrum) ** 2 / (
        g_diagonal[..., :, np.newaxis] * g_diagonal[..., np.newaxis, :]
    )
    ddtf = full_frequency_dtf * partial_coherence
    return {
        "directed_transfer_function": dtf,
        "partial_directed_coherence": pdc,
        "directed_coherence": dc,
        "generalized_partial_directed_coherence": gpdc,
        "direct_directed_transfer_function": ddtf,
    }


def _pairwise_granger_from_state_space(coefficients, noise_covariance, n_fft):
    """Pairwise spectral Granger of a VAR, ``[frequency, source, target]``.

    An oracle that involves no spectral factorization. Each pair of signals is
    a sub-process of the VAR; its innovations form (transfer function ``H``
    and innovation covariance ``V``) comes from the discrete algebraic Riccati
    equation of the VAR's state-space form (Barnett & Seth, 2015), and Geweke's
    formula gives the influence of each member of the pair on the other,
    ``log(S_tt / (S_tt - (V_ss - V_ts^2 / V_tt) |H_ts|^2))``. Values are on the
    non-negative FFT grid; the diagonal is NaN.
    """
    n_lags, n_signals, _ = coefficients.shape
    C = np.concatenate(list(coefficients), axis=1)
    A = np.zeros((n_signals * n_lags, n_signals * n_lags))
    A[:n_signals] = C
    if n_lags > 1:
        A[n_signals:, :-n_signals] = np.eye(n_signals * (n_lags - 1))
    K = np.zeros((n_signals * n_lags, n_signals))
    K[:n_signals] = np.eye(n_signals)
    z = np.exp(2j * np.pi * np.arange(n_fft // 2 + 1) / n_fft)
    resolvent = np.linalg.inv(z[:, None, None] * np.eye(A.shape[0]) - A)
    granger = np.full((z.size, n_signals, n_signals), np.nan)
    for first in range(n_signals):
        for second in range(first + 1, n_signals):
            pair = [first, second]
            C_pair = C[pair]
            Q = K @ noise_covariance @ K.T
            R = noise_covariance[np.ix_(pair, pair)]
            S = K @ noise_covariance[:, pair]
            P = solve_discrete_are(A.T, C_pair.T, Q, R, s=S)
            V = C_pair @ P @ C_pair.T + R
            gain = (A @ P @ C_pair.T + S) @ np.linalg.inv(V)
            H = np.eye(2) + C_pair @ resolvent @ gain  # (frequency, 2, 2)
            spectrum = H @ V @ H.conj().swapaxes(-1, -2)
            for target, source in ((0, 1), (1, 0)):
                intrinsic = V[source, source] - V[target, source] ** 2 / V[target, target]
                total = spectrum[:, target, target].real
                granger[:, pair[source], pair[target]] = np.log(
                    total / (total - intrinsic * np.abs(H[:, target, source]) ** 2)
                )
    return granger


def _companion_spectral_radius(coefficients):
    """Largest eigenvalue modulus of a VAR's companion matrix (< 1 is stable)."""
    n_lags, n_signals, _ = coefficients.shape
    companion = np.zeros((n_lags * n_signals, n_lags * n_signals))
    companion[:n_signals] = np.concatenate(list(coefficients), axis=1)
    companion[n_signals:, :-n_signals] = np.eye((n_lags - 1) * n_signals)
    return np.max(np.abs(np.linalg.eigvals(companion)))


# A 3-node chain 0 -> 1 -> 2 (no direct 0 -> 2 link) with 0.8 couplings, strong
# enough for simulated-data thresholds: analytic partial coherence peaks at
# 0.250 for (0, 1) and 0.817 for (1, 2) and is exactly 0 for the mediated
# (0, 2), whose pairwise coherence peaks at 0.785. Companion-matrix spectral
# radius 0.775.
_STRONG_CHAIN_COEFFICIENTS = np.stack(
    [np.array([[0.5, 0.0, 0.0], [0.8, 0.5, 0.0], [0.0, 0.8, 0.5]]), -0.6 * np.eye(3)]
)
