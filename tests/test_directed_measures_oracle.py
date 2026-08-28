"""Analytic VAR oracles for the directed connectivity measures.

The directed measures are otherwise validated against bounds, invariants, and
regression snapshots, which a consistent-but-wrong implementation could satisfy
(and re-baseline). These tests instead compare against a closed-form oracle
derived directly from a known vector-autoregressive (VAR) model.

For a stable VAR ``x(t) = sum_k A_k x(t - k) + e(t)`` with innovation covariance
``Sigma``, the transfer function and cross-spectrum are known exactly:

    A(f) = I - sum_k A_k exp(-i w k),   H(f) = A(f)^-1,   S(f) = H(f) Sigma H(f)^H

We inject that exact ``S(f)`` into ``Connectivity`` (by choosing Fourier
coefficients whose expected cross-spectrum equals ``S(f)``) and check the
measures against the analytic transfer function. Because a VAR is causal and
stable, the Wilson minimum-phase factorization inside the package recovers
``H(f)`` (up to the innovation Cholesky), so the recovered directed measures
match the analytic ones.

Direction convention (matching the package): a measure's ``[i, j]`` entry is the
influence ``j -> i``. A **unidirectional, lower-triangular** VAR (signal 0 drives
signal 1, never the reverse) has an exactly lower-triangular ``A(f)`` and
``H(f)``, so the non-causal ``[0, 1]`` entry is analytically zero for every
directed measure -- a strong oracle for direction (a flipped implementation
would put the energy in the wrong triangle).
"""

import warnings

import numpy as np
import pytest
from pytest import mark

from spectral_connectivity import Connectivity
from spectral_connectivity.wrapper import _connectivity_result_to_xarray


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


def _fourier_coefficients_with_cross_spectrum(S):
    """Fourier coefficients whose expected cross-spectrum is exactly ``S``.

    With ``S = L L^H`` (Cholesky) and ``n_tapers = n_signals``, taper ``k`` set to
    ``sqrt(n_signals) * L[:, k]`` makes the taper-mean of the outer products equal
    ``L L^H = S`` exactly. Shape: (1, 1, n_signals, n_fft, n_signals).
    """
    _, n_signals, _ = S.shape
    L = np.linalg.cholesky(S)  # (n_fft, n_signals, n_signals), lower-triangular
    # taper axis <- columns of L; scale so the taper-mean reproduces S.
    fc = np.sqrt(n_signals) * np.moveaxis(L, -1, -2)  # (n_fft, taper, signal)
    fc = np.moveaxis(fc, 0, 1)  # (taper, n_fft, signal)
    return fc[None, None]  # (1, 1, n_tapers, n_fft, n_signals)


# A unidirectional VAR: signal 0 drives signal 1 (lower-triangular coefficients),
# each an AR(2) with complex-conjugate poles for a genuine spectral peak.
_A1 = np.array([[0.5, 0.0], [0.4, 0.5]])
_A2 = np.array([[-0.6, 0.0], [0.0, -0.6]])
_COEFFICIENTS = np.stack([_A1, _A2])
_NOISE = np.eye(2)
_N_FFT = 128


@pytest.fixture(scope="module")
def var_oracle():
    """Analytic A/H/S and a Connectivity fed the exact analytic cross-spectrum."""
    A, H, S = _analytic_var(_COEFFICIENTS, _NOISE, _N_FFT)
    connectivity = Connectivity(
        fourier_coefficients=_fourier_coefficients_with_cross_spectrum(S)
    )
    return {"A": A, "H": H, "S": S, "connectivity": connectivity, "n_fft": _N_FFT}


def test_injected_cross_spectrum_matches_analytic(var_oracle):
    """Sanity: the constructed Fourier coefficients reproduce S(f) exactly."""
    c = var_oracle["connectivity"]
    csm = np.asarray(c._expectation_cross_spectral_matrix())[0]  # (n_fft, n, n)
    np.testing.assert_allclose(csm, var_oracle["S"], atol=1e-8)


@mark.parametrize(
    "measure",
    [
        "directed_transfer_function",
        "partial_directed_coherence",
        "directed_coherence",
        "generalized_partial_directed_coherence",
        "direct_directed_transfer_function",
        "pairwise_spectral_granger_prediction",
    ],
)
def test_non_causal_direction_is_zero(var_oracle, measure):
    """The non-causal [0, 1] entry must be ~0 for a unidirectional VAR.

    Signal 1 does not influence signal 0, so A(f) and H(f) are exactly
    lower-triangular and every directed measure's [0, 1] entry is analytically
    zero. The causal [1, 0] entry must be clearly positive.
    """
    c = var_oracle["connectivity"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = np.asarray(getattr(c, measure)())[0]  # (n_fft, n, n)

    non_causal = np.abs(result[..., 0, 1])
    causal = result[..., 1, 0]
    assert np.nanmax(non_causal) < 1e-8, (measure, np.nanmax(non_causal))
    assert np.nanmax(causal) > 0.05, (measure, np.nanmax(causal))


def test_pdc_matches_analytic_closed_form(var_oracle):
    """Package PDC equals the closed-form PDC of the known VAR.

    PDC_{ij}(f) = |A_{ij}(f)| / sqrt(sum_k |A_{kj}(f)|^2); the package returns the
    squared value over non-negative frequencies.
    """
    c = var_oracle["connectivity"]
    A, n_fft = var_oracle["A"], var_oracle["n_fft"]
    n_non_negative = n_fft // 2 + 1

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pdc = np.asarray(c.partial_directed_coherence())[0]

    analytic_pdc_10 = (
        np.abs(A[:, 1, 0]) / np.sqrt(np.abs(A[:, 0, 0]) ** 2 + np.abs(A[:, 1, 0]) ** 2)
    )[:n_non_negative]
    np.testing.assert_allclose(np.sqrt(pdc[:, 1, 0]), analytic_pdc_10, atol=1e-5)


def test_dtf_matches_analytic_closed_form(var_oracle):
    """Package DTF equals the closed-form DTF of the known VAR.

    DTF_{ij}(f) = |H_{ij}(f)| / sqrt(sum_k |H_{ik}(f)|^2); the package returns the
    squared value over non-negative frequencies. The causal-direction peak must
    also fall at the analytic transfer-function peak.
    """
    c = var_oracle["connectivity"]
    H, n_fft = var_oracle["H"], var_oracle["n_fft"]
    n_non_negative = n_fft // 2 + 1

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dtf = np.asarray(c.directed_transfer_function())[0]

    analytic_dtf_10 = (
        np.abs(H[:, 1, 0]) / np.sqrt(np.abs(H[:, 1, 0]) ** 2 + np.abs(H[:, 1, 1]) ** 2)
    )[:n_non_negative]
    np.testing.assert_allclose(np.sqrt(dtf[:, 1, 0]), analytic_dtf_10, atol=1e-5)

    analytic_peak = np.argmax(np.abs(H[:n_non_negative, 1, 0]) ** 2)
    assert np.argmax(dtf[:, 1, 0]) == analytic_peak


def test_wrapper_source_target_labels_follow_causal_direction(var_oracle):
    """The xarray wrapper must label directed measures source -> target.

    The ``Connectivity`` layer returns ``[i, j] = j -> i``; the wrapper
    transposes directed measures so that ``sel(source=driver, target=receiver)``
    reads out the causal entry. For this unidirectional VAR (signal 0 drives
    signal 1), the causal entry is ``sel(source="0", target="1")`` and the
    anti-causal ``sel(source="1", target="0")`` is analytically zero. A wrapper
    that forgot the transpose would swap these two.
    """
    connectivity = var_oracle["connectivity"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = _connectivity_result_to_xarray(
            connectivity,
            "pairwise_spectral_granger_prediction",
            ["0", "1"],
            False,
            {},
        )

    causal = result.sel(source="0", target="1").values  # 0 -> 1
    anti_causal = result.sel(source="1", target="0").values  # 1 -> 0
    assert np.nanmax(causal) > 0.05, np.nanmax(causal)
    assert np.nanmax(np.abs(anti_causal)) < 1e-8, np.nanmax(np.abs(anti_causal))
