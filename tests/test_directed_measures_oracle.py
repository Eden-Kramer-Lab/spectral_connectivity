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
would put the energy in the wrong triangle). A 3-node chain ``0 -> 1 -> 2`` with
unequal innovation variances additionally separates the normalizations: its
indirect path is visible to DTF but not PDC, and the unequal variances make the
noise-weighted DC and gPDC differ from DTF and PDC.
"""

import warnings

import numpy as np
import pytest
from scipy.linalg import solve_discrete_are

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


@pytest.mark.parametrize(
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

    non_causal = result[..., 0, 1]
    causal = result[..., 1, 0]
    # Both directions are defined at every frequency: NaN is not "zero".
    assert np.isfinite(non_causal).all(), measure
    assert np.isfinite(causal).all(), measure
    assert np.max(np.abs(non_causal)) < 1e-8, (measure, np.max(np.abs(non_causal)))
    assert np.max(causal) > 0.05, (measure, np.max(causal))


# A 3-node chain 0 -> 1 -> 2 (no direct 0 -> 2 link) with unequal, uncorrelated
# innovation variances. The indirect path makes DTF[2, 0] > 0 while PDC[2, 0] == 0,
# and the unequal variances make DC differ from DTF and gPDC differ from PDC, so
# each measure's normalization (row vs column, noise weighting) is identifiable.
_CHAIN_NOISE = np.diag([1.0, 2.0, 0.5])
_CHAIN_N_FFT = 256


def _analytic_directed_measures(A, H, noise_covariance):
    """Closed-form directed measures of a VAR on the non-negative FFT grid.

    ``[..., i, j]`` is the influence ``j -> i``. ``A`` and ``H`` are on the full
    FFT grid; ``noise_covariance`` must be diagonal (variances ``sigma``).
    Returns squared DTF, PDC, DC, and gPDC, and the (unsquared) dDTF.
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
    # dDTF: full-frequency DTF (inflow summed over sources and frequencies)
    # times the PDC magnitude.
    full_frequency_dtf = np.abs(H) / np.sqrt(H2.sum(axis=(-1, -3), keepdims=True))
    ddtf = full_frequency_dtf * np.sqrt(pdc)
    return {
        "directed_transfer_function": dtf,
        "partial_directed_coherence": pdc,
        "directed_coherence": dc,
        "generalized_partial_directed_coherence": gpdc,
        "direct_directed_transfer_function": ddtf,
    }


@pytest.fixture(scope="module")
def chain_oracle():
    """Analytic chain-VAR measures and a Connectivity fed its exact spectrum."""
    A, H, S = _analytic_var(_CHAIN_COEFFICIENTS, _CHAIN_NOISE, _CHAIN_N_FFT)
    connectivity = Connectivity(
        fourier_coefficients=_fourier_coefficients_with_cross_spectrum(S),
        minimum_phase_tolerance=1e-12,
        minimum_phase_max_iterations=5000,
    )
    return {
        "H": H,
        "measures": _analytic_directed_measures(A, H, _CHAIN_NOISE),
        "connectivity": connectivity,
    }


def test_chain_oracle_distinguishes_the_measures(chain_oracle):
    """Sanity: the chain system separates every pair of normalizations."""
    measures = chain_oracle["measures"]
    dtf, pdc = measures["directed_transfer_function"], measures["partial_directed_coherence"]
    dc = measures["directed_coherence"]
    gpdc = measures["generalized_partial_directed_coherence"]
    # Indirect path 0 -> 1 -> 2: DTF sees it, PDC (direct only) does not.
    assert dtf[:, 2, 0].max() > 0.3
    np.testing.assert_array_equal(pdc[:, 2, 0], 0.0)
    # Unequal noise variances make the noise-weighted measures differ.
    assert np.abs(dc - dtf).max() > 0.1
    assert np.abs(gpdc - pdc).max() > 0.1


@pytest.mark.parametrize(
    "measure",
    [
        "directed_transfer_function",
        "partial_directed_coherence",
        "directed_coherence",
        "generalized_partial_directed_coherence",
        "direct_directed_transfer_function",
    ],
)
def test_directed_measure_matches_analytic_closed_form(chain_oracle, measure):
    """Every entry (diagonal included) equals the closed form of the known VAR."""
    connectivity = chain_oracle["connectivity"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = np.asarray(getattr(connectivity, measure)())[0]
    np.testing.assert_allclose(result, chain_oracle["measures"][measure], rtol=0, atol=1e-9)


def test_dtf_peak_matches_analytic_transfer_function_peak(chain_oracle):
    """The indirect 0 -> 2 DTF peaks where the analytic |H[2, 0]| peaks."""
    connectivity = chain_oracle["connectivity"]
    H = chain_oracle["H"]
    n_non_negative = H.shape[0] // 2 + 1
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        dtf = np.asarray(connectivity.directed_transfer_function())[0]
    assert np.argmax(dtf[:, 2, 0]) == np.argmax(np.abs(H[:n_non_negative, 2, 0]))


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


def test_scalar_blockwise_and_conditional_granger_match_pairwise(var_oracle):
    """One-channel blocks and a two-node conditional system reduce to pairwise GC."""
    connectivity = var_oracle["connectivity"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pairwise = connectivity.pairwise_spectral_granger_prediction()
        blockwise, labels = connectivity.blockwise_spectral_granger_prediction([0, 1])
        conditional = connectivity.conditional_spectral_granger_prediction()

    np.testing.assert_array_equal(labels, [0, 1])
    np.testing.assert_allclose(blockwise, pairwise, atol=1e-6, equal_nan=True)
    np.testing.assert_allclose(conditional, pairwise, atol=1e-6, equal_nan=True)


def test_pairwise_granger_zero_influence_is_zero_not_nan(var_oracle):
    """A truly absent causal direction returns 0 (like the block path), not NaN.

    For the unidirectional oracle (0 -> 1) the [0, 1] direction (1 -> 0) has no
    causal influence. Roundoff can drive the log-ratio slightly negative there;
    it must be clipped to 0 rather than discarded as NaN, matching the
    conditional/block Granger convention.
    """
    connectivity = var_oracle["connectivity"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        granger = connectivity.pairwise_spectral_granger_prediction()[0]

    non_causal = granger[..., 0, 1]
    # No NaN masquerading as "no result"; the absent direction is a finite ~0
    # at every frequency.
    assert np.isfinite(non_causal).all()
    assert np.max(np.abs(non_causal)) < 1e-6


def test_pairwise_granger_matches_geweke_closed_form(var_oracle):
    """Pairwise Granger equals Geweke's formula for uncorrelated innovations.

    With a diagonal innovation covariance, Geweke's causality ``0 -> 1`` is
    ``log(S_11(f) / (Sigma_11 |H_11(f)|^2))`` and ``1 -> 0`` is
    ``log(S_00(f) / (Sigma_00 |H_00(f)|^2))`` (analytically zero here, since
    ``H_01 == 0``).
    """
    connectivity = var_oracle["connectivity"]
    H, S = var_oracle["H"], var_oracle["S"]
    n_non_negative = var_oracle["n_fft"] // 2 + 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        granger = connectivity.pairwise_spectral_granger_prediction()[0]

    for target, source in [(1, 0), (0, 1)]:
        geweke = np.log(
            S[:n_non_negative, target, target].real
            / (_NOISE[target, target] * np.abs(H[:n_non_negative, target, target]) ** 2)
        )
        np.testing.assert_allclose(granger[:, target, source], geweke, rtol=0, atol=1e-5)


def _state_space_conditional_granger(coefficients, noise_covariance, n_fft, target, source):
    """Conditional spectral Granger ``source -> target | others`` via state space.

    Independent oracle following Barnett & Seth (2015): the VAR is written in
    innovations form, the reduced model without ``source`` is obtained from the
    discrete algebraic Riccati equation (no reduced regression and no spectral
    factorization), and Geweke's conditional formula is evaluated with the
    reduced model's inverse transfer function. Returns values on the
    non-negative FFT grid.
    """
    n_lags, n_signals, _ = coefficients.shape
    C = np.concatenate(list(coefficients), axis=1)
    A = np.zeros((n_signals * n_lags, n_signals * n_lags))
    A[:n_signals] = C
    if n_lags > 1:
        A[n_signals:, :-n_signals] = np.eye(n_signals * (n_lags - 1))
    K = np.zeros((n_signals * n_lags, n_signals))
    K[:n_signals] = np.eye(n_signals)
    V = noise_covariance
    reduced = [i for i in range(n_signals) if i != source]
    C_r = C[reduced]
    Q, R, S = K @ V @ K.T, V[np.ix_(reduced, reduced)], K @ V[:, reduced]
    P = solve_discrete_are(A.T, C_r.T, Q, R, s=S)
    V_r = C_r @ P @ C_r.T + R
    K_r = (A @ P @ C_r.T + S) @ np.linalg.inv(V_r)
    target_r = reduced.index(target)
    others = [i for i in range(n_signals) if i != target]
    partial = (
        V[np.ix_(others, others)]
        - np.outer(V[others, target], V[target, others]) / V[target, target]
    )
    L = np.linalg.cholesky(partial)
    identity = np.eye(n_signals * n_lags)
    omegas = 2 * np.pi * np.arange(n_fft // 2 + 1) / n_fft
    values = np.empty(omegas.size)
    for k, omega in enumerate(omegas):
        z = np.exp(1j * omega)
        H = np.eye(n_signals) + C @ np.linalg.solve(z * identity - A, K)
        B_r = np.eye(len(reduced)) - C_r @ np.linalg.solve(z * identity - (A - K_r @ C_r), K_r)
        H_r = B_r[target_r] @ H[np.ix_(reduced, others)] @ L
        values[k] = np.log(V_r[target_r, target_r]) - np.log(
            V_r[target_r, target_r] - np.real(H_r @ H_r.conj())
        )
    return values


_DENSE_COEFFICIENTS = np.stack(
    [
        np.array([[0.5, 0.2, -0.1], [0.4, 0.5, 0.15], [0.1, 0.4, 0.5]]),
        np.array([[-0.6, 0.1, 0.0], [0.0, -0.6, -0.1], [0.05, 0.0, -0.6]]),
    ]
)
_CORRELATED_NOISE = np.array([[1.0, 0.3, 0.0], [0.3, 1.0, 0.2], [0.0, 0.2, 1.0]])
_CHAIN_COEFFICIENTS = np.stack(
    [
        np.array([[0.5, 0.0, 0.0], [0.4, 0.5, 0.0], [0.0, 0.4, 0.5]]),
        -0.6 * np.eye(3),
    ]
)


@pytest.mark.parametrize(
    ("coefficients", "noise_covariance"),
    [(_CHAIN_COEFFICIENTS, np.eye(3)), (_DENSE_COEFFICIENTS, _CORRELATED_NOISE)],
    ids=["chain", "dense_correlated"],
)
@pytest.mark.parametrize(
    ("n_fft", "atol"), [(128, 2e-5), (1024, 1e-10)], ids=["nfft128", "nfft1024"]
)
def test_conditional_granger_matches_state_space_oracle(
    coefficients, noise_covariance, n_fft, atol
):
    """The reduced-model factorization agrees pointwise with the state-space
    (Riccati) route used by MVGC, so the result does not depend on how the
    reduced model is obtained. The residual shrinks with the Wilson
    factorization's own frequency-discretization error."""
    _, _, spectrum = _analytic_var(coefficients, noise_covariance, n_fft)
    connectivity = Connectivity(
        fourier_coefficients=_fourier_coefficients_with_cross_spectrum(spectrum),
        minimum_phase_tolerance=1e-12,
        minimum_phase_max_iterations=5000,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        conditional = connectivity.conditional_spectral_granger_prediction()[0]
    assert np.isfinite(conditional[..., ~np.eye(3, dtype=bool)]).all()
    for target in range(3):
        for source in range(3):
            if target == source:
                continue
            oracle = _state_space_conditional_granger(
                coefficients, noise_covariance, n_fft, target, source
            )
            np.testing.assert_allclose(
                conditional[:, target, source], oracle, atol=atol, rtol=0
            )


def test_conditional_granger_removes_mediated_influence():
    """A 3-node chain 0 -> 1 -> 2 has zero conditional influence 0 -> 2 given 1.

    Signal 0 reaches signal 2 only through the mediator 1, so unconditional
    pairwise Granger 0 -> 2 is positive, but the conditional Granger 0 -> 2 | 1
    is analytically zero once 1 is accounted for (Chen, Bressler & Ding 2006).
    This exercises the non-empty conditioning path that the scalar/2-node oracle
    cannot reach.
    """
    # Lower-triangular VAR with no *direct* 0 -> 2 link (A[2, 0] == 0 at all lags).
    a1 = np.array([[0.5, 0.0, 0.0], [0.4, 0.5, 0.0], [0.0, 0.4, 0.5]])
    a2 = np.array([[-0.6, 0.0, 0.0], [0.0, -0.6, 0.0], [0.0, 0.0, -0.6]])
    coefficients = np.stack([a1, a2])
    _, _, spectrum = _analytic_var(coefficients, np.eye(3), _N_FFT)
    connectivity = Connectivity(
        fourier_coefficients=_fourier_coefficients_with_cross_spectrum(spectrum)
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pairwise = connectivity.pairwise_spectral_granger_prediction()[0]
        conditional = connectivity.conditional_spectral_granger_prediction()[0]

    # Unconditional 0 -> 2 (row 2, col 0) is clearly non-zero via the mediator.
    assert np.nanmax(pairwise[..., 2, 0]) > 0.05
    # The analytic spectrum is well conditioned, so every off-diagonal entry is
    # a finite, non-negative value; a true-null direction must not degrade into
    # NaN through roundoff-negative estimates.
    off_diagonal = ~np.eye(3, dtype=bool)
    assert np.isfinite(conditional[..., off_diagonal]).all()
    assert (conditional[..., off_diagonal] >= 0).all()
    # Conditioning on signal 1 removes it: 0 -> 2 | 1 collapses toward zero.
    assert conditional[..., 2, 0].max() < 1e-8
    # The genuine direct link 1 -> 2 | 0 survives conditioning.
    assert conditional[..., 2, 1].max() > 0.05


def test_time_reversed_granger_flips_unidirectional_oracle(var_oracle):
    """Time reversal makes the originally causal direction predominantly reverse."""
    connectivity = var_oracle["connectivity"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reversed_gc = connectivity.time_reversed_spectral_granger_prediction()[0]

    # Original system is 0 -> 1 ([1, 0]); after reversal the [0, 1] direction
    # must dominate strongly, even though correlated reversed innovations can
    # leave a small residual in the original direction.
    assert np.nanmax(reversed_gc[..., 0, 1]) > 0.5
    assert np.nanmax(reversed_gc[..., 0, 1]) > 10 * np.nanmax(reversed_gc[..., 1, 0])
