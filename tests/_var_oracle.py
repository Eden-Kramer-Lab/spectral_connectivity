"""Analytic VAR oracle helpers shared by the test modules.

For a stable VAR ``x(t) = sum_k A_k x(t - k) + e(t)`` with innovation
covariance ``Sigma``, the transfer function and cross-spectrum are known
exactly; these helpers build them and Fourier coefficients that reproduce
them, so a ``Connectivity`` can be fed an exact spectrum (see
``tests/test_directed_measures_oracle.py``).
"""

import numpy as np


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
