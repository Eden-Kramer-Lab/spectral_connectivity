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
influence ``i -> j``. ``A(f)``, ``H(f)`` and the analytic closed-form measures
below keep the native ``[target, source]`` layout (``[i, j]`` is ``j -> i``), so
they are transposed before comparison. A **unidirectional, lower-triangular**
VAR (signal 0 drives signal 1, never the reverse) has an exactly
lower-triangular ``A(f)`` and ``H(f)``, so the non-causal ``1 -> 0`` entry
``[1, 0]`` is analytically zero for every directed measure -- a strong oracle
for direction (a flipped implementation would put the energy in the wrong
triangle). A 3-node chain ``0 -> 1 -> 2`` with unequal innovation variances
additionally separates the normalizations: its indirect path is visible to DTF
but not PDC, and the unequal variances make the noise-weighted DC and gPDC
differ from DTF and PDC.
"""

import itertools
import warnings

import numpy as np
import pytest
from scipy.linalg import solve_discrete_are

from spectral_connectivity import Connectivity, DirectedOrientationWarning, Multitaper
from spectral_connectivity._result_formatting import _connectivity_result_to_xarray
from spectral_connectivity.minimum_phase_decomposition import _is_conjugate_symmetric
from spectral_connectivity.simulate import simulate_MVAR
from tests._backend_helpers import to_host
from tests._var_oracle import (
    _analytic_directed_measures,
    _analytic_var,
    _block_granger_from_state_space,
    _companion_matrix,
    _companion_spectral_radius,
    _dense_stable_var,
    _fourier_coefficients_with_cross_spectrum,
    _pairwise_granger_from_state_space,
    _prediction_error_covariance,
    _time_reversed_var,
    _var_autocovariance,
)

# A unidirectional VAR: signal 0 drives signal 1 (lower-triangular coefficients),
# each an AR(2) with complex-conjugate poles for a genuine spectral peak.
_A1 = np.array([[0.5, 0.0], [0.4, 0.5]])
_A2 = np.array([[-0.6, 0.0], [0.0, -0.6]])
_COEFFICIENTS = np.stack([_A1, _A2])
_NOISE = np.eye(2)
_N_FFT = 128


# Both Wilson factorization paths (see ``_fourier_coefficients_with_cross_spectrum``).
_BOTH_FACTORIZATION_PATHS = pytest.mark.parametrize(
    "conjugate_symmetric", [False, True], ids=["two_sided", "half_spectrum"]
)


def _tightly_factored_connectivity(spectrum, conjugate_symmetric=False):
    """A ``Connectivity`` fed the exact ``spectrum`` and factored to a tolerance
    far below the ones the oracle comparisons allow."""
    return Connectivity(
        fourier_coefficients=_fourier_coefficients_with_cross_spectrum(
            spectrum, conjugate_symmetric=conjugate_symmetric
        ),
        minimum_phase_tolerance=1e-12,
        minimum_phase_max_iterations=5000,
    )


@pytest.fixture(scope="module", params=[False, True], ids=["two_sided", "half_spectrum"])
def var_oracle(request):
    """Analytic A/H/S and a Connectivity fed the exact analytic cross-spectrum.

    Parametrized over both Wilson factorization paths (see
    ``_fourier_coefficients_with_cross_spectrum``).
    """
    A, H, S = _analytic_var(_COEFFICIENTS, _NOISE, _N_FFT)
    connectivity = Connectivity(
        fourier_coefficients=_fourier_coefficients_with_cross_spectrum(
            S, conjugate_symmetric=request.param
        )
    )
    return {
        "A": A,
        "H": H,
        "S": S,
        "connectivity": connectivity,
        "n_fft": _N_FFT,
        "conjugate_symmetric": request.param,
    }


def test_injected_cross_spectrum_matches_analytic(var_oracle):
    """Sanity: the constructed Fourier coefficients reproduce S(f) exactly.

    Also checks which Wilson path the spectrum takes, so each fixture variant
    covers the path it is meant to.
    """
    c = var_oracle["connectivity"]
    csm = c._expectation_cross_spectral_matrix()[0]  # (n_fft, n, n)
    np.testing.assert_allclose(to_host(csm), var_oracle["S"], atol=1e-8)
    assert _is_conjugate_symmetric(csm) == var_oracle["conjugate_symmetric"]


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
    """The non-causal [1, 0] entry (1 -> 0) must be ~0 for a unidirectional VAR.

    Signal 1 does not influence signal 0, so A(f) and H(f) are exactly
    lower-triangular and every directed measure's [1, 0] entry is analytically
    zero. The causal [0, 1] entry (0 -> 1) must be clearly positive.
    """
    c = var_oracle["connectivity"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = np.asarray(getattr(c, measure)())[0]  # (n_fft, n, n)

    non_causal = result[..., 1, 0]
    causal = result[..., 0, 1]
    # Both directions are defined at every frequency: NaN is not "zero".
    assert np.isfinite(non_causal).all(), measure
    assert np.isfinite(causal).all(), measure
    assert np.max(np.abs(non_causal)) < 1e-8, (measure, np.max(np.abs(non_causal)))
    # dDTF is normalized over all frequencies at once, so compare its
    # frequency-summed share of the target's inflow; the others are per bin.
    if measure == "direct_directed_transfer_function":
        causal_strength = np.sum(causal)
    else:
        causal_strength = np.max(causal)
    assert causal_strength > 0.05, (measure, causal_strength)


# A 3-node chain 0 -> 1 -> 2 (no direct 0 -> 2 link) with unequal, uncorrelated
# innovation variances. The indirect path makes the analytic (native
# [target, source]) DTF[2, 0] > 0 while PDC[2, 0] == 0, and the unequal
# variances make DC differ from DTF and gPDC differ from PDC, so each
# measure's normalization (row vs column, noise weighting) is identifiable.
_CHAIN_NOISE = np.diag([1.0, 2.0, 0.5])
_CHAIN_N_FFT = 256


@pytest.fixture(scope="module")
def chain_oracle():
    """Analytic chain-VAR measures and a Connectivity fed its exact spectrum."""
    A, H, S = _analytic_var(_CHAIN_COEFFICIENTS, _CHAIN_NOISE, _CHAIN_N_FFT)
    connectivity = _tightly_factored_connectivity(S)
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
    # dDTF keeps only direct links: zero on the indirect 0 -> 2 path, positive on
    # the direct 0 -> 1 and 1 -> 2 links.
    ddtf = measures["direct_directed_transfer_function"]
    np.testing.assert_allclose(ddtf[:, 2, 0], 0.0, atol=1e-15)
    assert ddtf[:, 1, 0].max() > 0
    assert ddtf[:, 2, 1].max() > 0


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
    """Every entry (diagonal included) equals the closed form of the known VAR,
    transposed from its native layout (see the module docstring)."""
    connectivity = chain_oracle["connectivity"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warnings.filterwarnings("ignore", category=DirectedOrientationWarning)
        result = np.asarray(getattr(connectivity, measure)())[0]
    expected = np.swapaxes(chain_oracle["measures"][measure], -1, -2)
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-9)


def test_dtf_peak_matches_analytic_transfer_function_peak(chain_oracle):
    """The indirect 0 -> 2 DTF peaks where the analytic |H[2, 0]| peaks."""
    connectivity = chain_oracle["connectivity"]
    H = chain_oracle["H"]
    n_non_negative = H.shape[0] // 2 + 1
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warnings.filterwarnings("ignore", category=DirectedOrientationWarning)
        dtf = np.asarray(connectivity.directed_transfer_function())[0]
    # dtf is [source, target]; H is [target, source].
    assert np.argmax(dtf[:, 0, 2]) == np.argmax(np.abs(H[:n_non_negative, 2, 0]))


def test_wrapper_source_target_labels_follow_causal_direction(var_oracle):
    """The xarray wrapper must label directed measures source -> target.

    The ``Connectivity`` layer returns ``[i, j] = i -> j`` and the wrapper labels
    those axes ``source`` and ``target`` without reordering them, so
    ``sel(source=driver, target=receiver)`` reads out the causal entry. For this
    unidirectional VAR (signal 0 drives signal 1), the causal entry is
    ``sel(source="0", target="1")`` and the anti-causal
    ``sel(source="1", target="0")`` is analytically zero. A wrapper that
    transposed the array would swap these two.
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
        granger = connectivity.pairwise_spectral_granger_prediction()

    causal = result.sel(source="0", target="1").values  # 0 -> 1
    anti_causal = result.sel(source="1", target="0").values  # 1 -> 0
    assert np.nanmax(causal) > 0.05, np.nanmax(causal)
    assert np.nanmax(np.abs(anti_causal)) < 1e-8, np.nanmax(np.abs(anti_causal))
    np.testing.assert_array_equal(causal, granger[..., 0, 1])
    np.testing.assert_array_equal(anti_causal, granger[..., 1, 0])


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


# Two-channel blocks {0, 1} -> {2, 3}: block {2, 3} never feeds back into
# {0, 1}, so H_XY == 0 and the reverse block direction is analytically zero.
_BLOCK_A1 = np.array(
    [
        [0.5, 0.1, 0.0, 0.0],
        [0.1, 0.5, 0.0, 0.0],
        [0.4, 0.2, 0.5, 0.1],
        [0.3, 0.3, 0.1, 0.5],
    ]
)
_BLOCK_COEFFICIENTS = np.stack([_BLOCK_A1, -0.6 * np.eye(4)])
# Correlated innovations, within and across the blocks (positive definite).
_CORRELATED_BLOCK_NOISE = np.array(
    [
        [1.0, 0.3, 0.5, 0.2],
        [0.3, 1.0, 0.1, 0.4],
        [0.5, 0.1, 1.0, 0.3],
        [0.2, 0.4, 0.3, 1.0],
    ]
)
_BLOCK_LABELS = [0, 0, 1, 1]
_BLOCK_N_FFT = 256


@pytest.mark.parametrize(
    "noise_covariance",
    [np.eye(4), _CORRELATED_BLOCK_NOISE],
    ids=["uncorrelated", "correlated"],
)
@_BOTH_FACTORIZATION_PATHS
def test_blockwise_granger_matches_geweke_block_closed_form(
    conjugate_symmetric, noise_covariance
):
    """Blockwise Granger equals Geweke's block measure.

    With X = {0, 1} and Y = {2, 3}, ``F_{X->Y}(f) = log det S_YY(f) -
    log det(S_YY(f) - H_YX(f) Sigma_{X|Y} H_YX(f)^H)``, where
    ``Sigma_{X|Y} = Sigma_XX - Sigma_XY Sigma_YY^-1 Sigma_YX`` removes the
    instantaneous (innovation) correlation (equal to ``Sigma_XX`` when the
    blocks' innovations are uncorrelated). ``F_{Y->X}`` is 0 because
    ``H_XY == 0``. Group 0 is X and group 1 is Y, so ``[..., 0, 1]`` is
    ``X -> Y``.
    """
    assert _companion_spectral_radius(_BLOCK_COEFFICIENTS) < 1
    assert np.all(np.linalg.eigvalsh(noise_covariance) > 0)
    _, H, S = _analytic_var(_BLOCK_COEFFICIENTS, noise_covariance, _BLOCK_N_FFT)
    connectivity = Connectivity(
        fourier_coefficients=_fourier_coefficients_with_cross_spectrum(
            S, conjugate_symmetric=conjugate_symmetric
        )
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        blockwise, labels = connectivity.blockwise_spectral_granger_prediction(_BLOCK_LABELS)

    n_non_negative = _BLOCK_N_FFT // 2 + 1
    source, target = [0, 1], [2, 3]
    S_yy = S[:n_non_negative][:, target][:, :, target]
    H_yx = H[:n_non_negative][:, target][:, :, source]
    sigma = noise_covariance
    sigma_x_given_y = sigma[np.ix_(source, source)] - sigma[
        np.ix_(source, target)
    ] @ np.linalg.solve(sigma[np.ix_(target, target)], sigma[np.ix_(target, source)])
    intrinsic = S_yy - H_yx @ sigma_x_given_y @ H_yx.conj().swapaxes(-1, -2)
    # NumPy 1.24's complex slogdet warns of a division by zero on
    # well-conditioned input on some platforms.
    with np.errstate(divide="ignore", invalid="ignore"):
        geweke = np.linalg.slogdet(S_yy)[1] - np.linalg.slogdet(intrinsic)[1]

    np.testing.assert_array_equal(labels, [0, 1])
    assert geweke.max() > 1.0  # premise: a strong planted block influence
    np.testing.assert_allclose(blockwise[0, :, 0, 1], geweke, rtol=0, atol=1e-5)
    assert np.max(np.abs(blockwise[0, :, 1, 0])) < 1e-5


def test_blockwise_granger_direction_on_simulated_var():
    """On data simulated from the block VAR, {0, 1} -> {2, 3} dominates the
    reverse direction (analytically zero) by more than 5x on average."""
    time_series = simulate_MVAR(
        _BLOCK_COEFFICIENTS, n_time_samples=2000, n_trials=30, random_state=0
    )
    connectivity = Connectivity.from_transform(Multitaper(time_series, sampling_frequency=200))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        blockwise, _ = connectivity.blockwise_spectral_granger_prediction(_BLOCK_LABELS)

    planted, reverse = blockwise[0, :, 0, 1], blockwise[0, :, 1, 0]
    assert np.isfinite(planted).all()
    assert np.isfinite(reverse).all()
    assert planted.mean() > 5 * reverse.mean()


def test_pairwise_granger_zero_influence_is_zero_not_nan(var_oracle):
    """A truly absent causal direction returns 0 (like the block path), not NaN.

    For the unidirectional oracle (0 -> 1) the [1, 0] direction (1 -> 0) has no
    causal influence. Roundoff can drive the log-ratio slightly negative there;
    it must be clipped to 0 rather than discarded as NaN, matching the
    conditional/block Granger convention.
    """
    connectivity = var_oracle["connectivity"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        granger = connectivity.pairwise_spectral_granger_prediction()[0]

    non_causal = granger[..., 1, 0]
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
        np.testing.assert_allclose(granger[:, source, target], geweke, rtol=0, atol=1e-5)


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
    A = _companion_matrix(coefficients)
    K = np.zeros((n_signals * n_lags, n_signals))
    K[:n_signals] = np.eye(n_signals)
    V = noise_covariance
    reduced = [i for i in range(n_signals) if i != source]
    C_r = C[reduced]
    Q, R, S = K @ V @ K.T, V[np.ix_(reduced, reduced)], K @ V[:, reduced]
    P = _prediction_error_covariance(A, C_r, Q, R, S)
    V_r = C_r @ P @ C_r.conj().T + R
    K_r = (A @ P @ C_r.conj().T + S) @ np.linalg.inv(V_r)
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
        reduced_variance = np.real(V_r[target_r, target_r])
        values[k] = np.log(reduced_variance) - np.log(
            reduced_variance - np.real(H_r @ H_r.conj())
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
_CHAIN_AND_DENSE_MODELS = pytest.mark.parametrize(
    ("coefficients", "noise_covariance"),
    [(_CHAIN_COEFFICIENTS, np.eye(3)), (_DENSE_COEFFICIENTS, _CORRELATED_NOISE)],
    ids=["chain", "dense_correlated"],
)


@pytest.mark.parametrize(
    ("coefficients", "noise_covariance"),
    [
        (_CHAIN_COEFFICIENTS, np.eye(3)),
        (_DENSE_COEFFICIENTS, _CORRELATED_NOISE),
        _dense_stable_var(n_signals=5, n_lags=2, spectral_radius=0.8, seed=0),
    ],
    ids=["chain", "dense_correlated", "dense_5_signals"],
)
def test_riccati_iteration_matches_scipy_on_real_models(coefficients, noise_covariance):
    """The state-space oracles solve their Riccati equations by iteration,
    which also covers complex models; for real ones it must give SciPy's
    ``solve_discrete_are`` solution, for every observed sub-system the oracles
    use (each signal pair, and every signal but one)."""
    n_signals = coefficients.shape[-1]
    C = np.concatenate(list(coefficients), axis=1)
    A = _companion_matrix(coefficients)
    K = np.zeros((A.shape[0], n_signals))
    K[:n_signals] = np.eye(n_signals)
    Q = K @ noise_covariance @ K.T
    subsets = [
        *itertools.combinations(range(n_signals), 2),
        *itertools.combinations(range(n_signals), n_signals - 1),
    ]
    for members in map(list, subsets):
        C_sub = C[members]
        R = noise_covariance[np.ix_(members, members)]
        S = K @ noise_covariance[:, members]
        expected = solve_discrete_are(A.T, C_sub.T, Q, R, s=S)
        np.testing.assert_allclose(
            _prediction_error_covariance(A, C_sub, Q, R, S),
            expected,
            rtol=0,
            atol=1e-12 * (np.abs(Q).max() + np.abs(expected).max()),
            err_msg=str(members),
        )


@_CHAIN_AND_DENSE_MODELS
@pytest.mark.parametrize(
    ("n_fft", "atol"), [(128, 2e-5), (1024, 1e-10)], ids=["nfft128", "nfft1024"]
)
@_BOTH_FACTORIZATION_PATHS
def test_conditional_granger_matches_state_space_oracle(
    coefficients, noise_covariance, n_fft, atol, conjugate_symmetric
):
    """The reduced-model factorization agrees pointwise with the state-space
    (Riccati) route used by MVGC, so the result does not depend on how the
    reduced model is obtained. The residual shrinks with the Wilson
    factorization's own frequency-discretization error."""
    _, _, spectrum = _analytic_var(coefficients, noise_covariance, n_fft)
    connectivity = _tightly_factored_connectivity(spectrum, conjugate_symmetric)
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
                conditional[:, source, target], oracle, atol=atol, rtol=0
            )


@_CHAIN_AND_DENSE_MODELS
@_BOTH_FACTORIZATION_PATHS
def test_pairwise_granger_matches_state_space_oracle(
    coefficients, noise_covariance, conjugate_symmetric
):
    """Pairwise Granger from the Wilson factor of each pair's spectrum agrees
    with the innovations form of that pair obtained from a Riccati equation,
    which factors nothing: influences in both directions, through other
    signals, and with correlated innovations."""
    n_fft = 1024
    _, _, spectrum = _analytic_var(coefficients, noise_covariance, n_fft)
    connectivity = _tightly_factored_connectivity(spectrum, conjugate_symmetric)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DirectedOrientationWarning)
        pairwise = connectivity.pairwise_spectral_granger_prediction()[0]
    oracle = _pairwise_granger_from_state_space(coefficients, noise_covariance, n_fft)
    off_diagonal = ~np.eye(3, dtype=bool)
    assert oracle[:, off_diagonal].max() > 0.1
    np.testing.assert_allclose(
        pairwise[:, off_diagonal], oracle[:, off_diagonal], atol=1e-9, rtol=0
    )


@pytest.mark.parametrize(
    "group_labels",
    [["a", "b", "b", "c", "c"], ["b", "a", "b", "c", "a"], [0, 0, 0, 1, 2]],
    ids=["sizes_1_2_2", "interleaved", "sizes_3_1_1"],
)
@_BOTH_FACTORIZATION_PATHS
def test_blockwise_granger_among_three_groups_matches_state_space_oracle(
    group_labels, conjugate_symmetric
):
    """With more than two groups, the measure between two of them is that of
    the sub-process they form, for which the remaining signals are unobserved.
    The full model's transfer function does not give it; the innovations form
    of each sub-process, from a Riccati equation, does. Every signal drives
    every other here and the innovations are correlated, so no direction is
    zero, and the groups differ in size and need not be adjacent signals."""
    n_fft = 1024
    coefficients, noise_covariance = _dense_stable_var(
        n_signals=5, n_lags=2, spectral_radius=0.8, seed=0
    )
    _, _, spectrum = _analytic_var(coefficients, noise_covariance, n_fft)
    connectivity = _tightly_factored_connectivity(spectrum, conjugate_symmetric)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warnings.filterwarnings("ignore", category=DirectedOrientationWarning)
        blockwise, labels = connectivity.blockwise_spectral_granger_prediction(group_labels)

    blocks = [
        [signal for signal, label in enumerate(group_labels) if label == group]
        for group in labels
    ]
    oracle = _block_granger_from_state_space(coefficients, noise_covariance, n_fft, blocks)
    off_diagonal = ~np.eye(3, dtype=bool)
    # Premise: an influence in every direction, at its peak far above the tolerance.
    assert oracle[:, off_diagonal].max(axis=0).min() > 0.01
    np.testing.assert_allclose(
        blockwise[0][:, off_diagonal], oracle[:, off_diagonal], atol=1e-9, rtol=0
    )


_DIRECTED_MEASURES = [
    "pairwise_spectral_granger_prediction",
    "conditional_spectral_granger_prediction",
    "directed_transfer_function",
    "partial_directed_coherence",
    "directed_coherence",
    "generalized_partial_directed_coherence",
    "direct_directed_transfer_function",
]


@pytest.mark.parametrize(
    "measure", [*_DIRECTED_MEASURES, "blockwise_spectral_granger_prediction"]
)
def test_complex_white_noise_has_no_direction(measure):
    """Complex white noise with correlated innovations has no lagged dependence.

    Its spectrum is the same Hermitian matrix at every frequency, with
    imaginary off-diagonal entries, so the minimum-phase factor is that
    matrix's constant square root: the transfer function is the identity and
    every directed measure is exactly zero off the diagonal. Discarding the
    imaginary part of the factor's lag-0 coefficient left ``H(0) != I`` and
    reported a direction (pairwise Granger 0.45, DTF 0.26 for a coupling of
    0.6j).
    """
    covariance = np.array([[1.0, 0.6j, 0.2], [-0.6j, 1.0, -0.3j], [0.2, 0.3j, 1.0]])
    spectrum = np.broadcast_to(covariance, (_N_FFT, 3, 3))
    connectivity = Connectivity(_fourier_coefficients_with_cross_spectrum(spectrum))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if measure == "blockwise_spectral_granger_prediction":
            result, _ = connectivity.blockwise_spectral_granger_prediction([0, 1, 1])
        else:
            result = getattr(connectivity, measure)()
    off_diagonal = ~np.eye(result.shape[-1], dtype=bool)
    values = np.asarray(result)[..., off_diagonal]
    assert np.isfinite(values).all(), measure
    np.testing.assert_allclose(values, 0.0, atol=1e-12, rtol=0, err_msg=measure)


@pytest.fixture(scope="module")
def complex_var():
    """A dense complex VAR: every signal drives every other, the innovations
    are correlated with imaginary covariances, and the spectrum is not
    conjugate-symmetric. Its exact spectrum is fed to a ``Connectivity``."""
    n_fft = 1024
    coefficients, noise_covariance = _dense_stable_var(
        n_signals=3, n_lags=2, spectral_radius=0.8, seed=1, complex_valued=True
    )
    _, _, spectrum = _analytic_var(coefficients, noise_covariance, n_fft)
    connectivity = _tightly_factored_connectivity(spectrum)
    assert not _is_conjugate_symmetric(connectivity._expectation_cross_spectral_matrix())
    return coefficients, noise_covariance, n_fft, connectivity


def test_pairwise_and_blockwise_granger_of_a_complex_var_match_state_space_oracle(
    complex_var,
):
    """The Riccati oracle needs no spectral factorization and holds for a
    complex VAR, so it checks that the Wilson-based measures keep the
    imaginary parts of the lag-0 coefficient and the innovation covariance."""
    coefficients, noise_covariance, n_fft, connectivity = complex_var
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DirectedOrientationWarning)
        pairwise = connectivity.pairwise_spectral_granger_prediction()[0]
        blockwise, labels = connectivity.blockwise_spectral_granger_prediction([0, 1, 1])
    off_diagonal = ~np.eye(3, dtype=bool)
    oracle = _pairwise_granger_from_state_space(coefficients, noise_covariance, n_fft)
    assert oracle[:, off_diagonal].max(axis=0).min() > 0.01
    np.testing.assert_allclose(
        pairwise[:, off_diagonal], oracle[:, off_diagonal], atol=1e-9, rtol=0
    )
    blocks = [
        [signal for signal, label in enumerate([0, 1, 1]) if label == group]
        for group in labels
    ]
    block_oracle = _block_granger_from_state_space(
        coefficients, noise_covariance, n_fft, blocks
    )
    block_off_diagonal = ~np.eye(2, dtype=bool)
    np.testing.assert_allclose(
        blockwise[0][:, block_off_diagonal],
        block_oracle[:, block_off_diagonal],
        atol=1e-9,
        rtol=0,
    )


def test_conditional_granger_of_a_complex_var_matches_state_space_oracle(complex_var):
    """Geweke's normalization of the full model's innovations takes the
    conjugate transpose, ``P Sigma P^H``, for a Hermitian covariance."""
    coefficients, noise_covariance, n_fft, connectivity = complex_var
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        conditional = connectivity.conditional_spectral_granger_prediction()[0]
    for target in range(3):
        for source in range(3):
            if target == source:
                continue
            oracle = _state_space_conditional_granger(
                coefficients, noise_covariance, n_fft, target, source
            )
            assert oracle.max() > 0.01
            np.testing.assert_allclose(
                conditional[:, source, target], oracle, atol=1e-9, rtol=0
            )


def test_ddtf_is_positive_between_parents_of_a_common_child():
    """dDTF is zero for a relayed influence, not only for an absent one: in
    ``0 -> 1 -> 2`` with ``0 -> 3 <- 2`` there is no ``0 -> 2`` coefficient,
    so PDC and conditional Granger are zero, but 0 and 2 drive the common child
    3 and their partial coherence is not, so dDTF ``0 -> 2`` is positive at
    every frequency. Its closed form sums to exactly 1/18 for unit couplings
    and identity innovations."""
    coefficients = np.zeros((1, 4, 4))  # native [target, source]
    for source, target in [(0, 1), (1, 2), (0, 3), (2, 3)]:
        coefficients[0, target, source] = 1.0
    A, H, spectrum = _analytic_var(coefficients, np.eye(4), 256)
    connectivity = _tightly_factored_connectivity(spectrum)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ddtf = connectivity.direct_directed_transfer_function()[0]
        pdc = connectivity.partial_directed_coherence()[0]
        conditional = connectivity.conditional_spectral_granger_prediction()[0]

    analytic = _analytic_directed_measures(A, H, np.eye(4))[
        "direct_directed_transfer_function"
    ]
    np.testing.assert_allclose(ddtf[:, 0, 2], analytic[:, 2, 0], rtol=1e-9)
    assert ddtf[:, 0, 2].min() > 1e-4
    assert ddtf[:, 0, 2].sum() == pytest.approx(1 / 18, rel=1e-9)
    np.testing.assert_allclose(pdc[:, 0, 2], 0.0, atol=1e-12)
    np.testing.assert_allclose(conditional[:, 0, 2], 0.0, atol=1e-10)


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

    # Unconditional 0 -> 2 ([0, 2]) is clearly non-zero via the mediator.
    assert np.nanmax(pairwise[..., 0, 2]) > 0.05
    # The analytic spectrum is well conditioned, so every off-diagonal entry is
    # a finite, non-negative value; a true-null direction must not degrade into
    # NaN through roundoff-negative estimates.
    off_diagonal = ~np.eye(3, dtype=bool)
    assert np.isfinite(conditional[..., off_diagonal]).all()
    assert (conditional[..., off_diagonal] >= 0).all()
    # Conditioning on signal 1 removes it: 0 -> 2 | 1 collapses toward zero.
    assert conditional[..., 0, 2].max() < 1e-8
    # The genuine direct link 1 -> 2 | 0 survives conditioning.
    assert conditional[..., 1, 2].max() > 0.05


def test_time_reversed_granger_flips_unidirectional_oracle(var_oracle):
    """Time reversal makes the originally causal direction predominantly reverse."""
    connectivity = var_oracle["connectivity"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reversed_gc = connectivity.time_reversed_spectral_granger_prediction()[0]

    # Original system is 0 -> 1 ([0, 1]); after reversal the 1 -> 0 direction
    # ([1, 0]) must dominate strongly, even though correlated reversed
    # innovations can leave a small residual in the original direction.
    assert np.nanmax(reversed_gc[..., 1, 0]) > 0.5
    assert np.nanmax(reversed_gc[..., 1, 0]) > 10 * np.nanmax(reversed_gc[..., 0, 1])


@pytest.mark.parametrize("n_lags", [1, 2, 3])
def test_var_autocovariance_is_the_inverse_transform_of_the_spectrum(n_lags):
    """``Gamma(h) = E[x(t + h) x(t)^T]`` from the state covariance and the
    Yule-Walker recursion equals the inverse Fourier transform of the model's
    spectrum, at lags below, at and beyond the model's order. The lag-one
    covariance is not symmetric, so a transposed convention would fail."""
    n_fft, max_lag = 4096, 6
    coefficients, noise_covariance = _dense_stable_var(
        n_signals=3, n_lags=n_lags, spectral_radius=0.8, seed=n_lags
    )
    _, _, spectrum = _analytic_var(coefficients, noise_covariance, n_fft)
    from_spectrum = np.fft.ifft(spectrum, axis=0)[: max_lag + 1].real

    gamma = _var_autocovariance(coefficients, noise_covariance, max_lag)

    assert np.abs(gamma[1] - gamma[1].T).max() > 0.1
    np.testing.assert_allclose(gamma, from_spectrum, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(
        _var_autocovariance(coefficients, noise_covariance, 0), gamma[:1]
    )


@_CHAIN_AND_DENSE_MODELS
@_BOTH_FACTORIZATION_PATHS
def test_time_reversed_granger_matches_the_reversed_var(
    coefficients, noise_covariance, conjugate_symmetric
):
    """Read backwards in time, a VAR is again a VAR, with coefficients and
    innovations that follow from its autocovariance. Time-reversed Granger of
    the forward spectrum equals the state-space pairwise Granger of that
    reversed VAR. It is not the forward result with the two directions
    exchanged, which a transposed output would give."""
    n_fft = 1024
    _, _, spectrum = _analytic_var(coefficients, noise_covariance, n_fft)
    reversed_coefficients, reversed_noise = _time_reversed_var(coefficients, noise_covariance)
    # Premise: the reversed VAR has the transposed spectrum.
    np.testing.assert_allclose(
        _analytic_var(reversed_coefficients, reversed_noise, n_fft)[2],
        spectrum.swapaxes(-1, -2),
        atol=1e-10,
        rtol=0,
    )
    connectivity = _tightly_factored_connectivity(spectrum, conjugate_symmetric)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DirectedOrientationWarning)
        time_reversed = connectivity.time_reversed_spectral_granger_prediction()[0]

    oracle = _pairwise_granger_from_state_space(reversed_coefficients, reversed_noise, n_fft)
    forward = _pairwise_granger_from_state_space(coefficients, noise_covariance, n_fft)
    off_diagonal = ~np.eye(3, dtype=bool)
    exchanged = forward.swapaxes(-1, -2)
    # Premise: reversal does more than exchange the directions.
    assert np.abs(oracle - exchanged)[:, off_diagonal].max() > 0.05
    np.testing.assert_allclose(
        time_reversed[:, off_diagonal], oracle[:, off_diagonal], atol=1e-9, rtol=0
    )


# Signal 0 drives 1 and 1 drives 2, with poles at 0.9, 0.8 and 0.7: the
# autocovariance is still 3% of its peak at lag 32, so it wraps around a
# 64-point grid.
_SLOW_CHAIN_COEFFICIENTS = np.array([[[0.9, 0.0, 0.0], [0.6, 0.8, 0.0], [0.0, 0.5, 0.7]]])


@pytest.mark.parametrize("n_fft", [64, 65])
def test_pairwise_granger_is_defined_on_a_grid_the_spectrum_wraps_around(n_fft):
    """Spectral Granger is a log-ratio of a signal's power to the part its own
    past explains. The Wilson factor reproduces the spectrum on the grid it
    was given, so that part never exceeds the power and every bin is finite
    and non-negative, however coarse the grid. On an even grid the factor
    once missed the spectrum (the lag ``n_fft / 2`` was dropped from the
    causal projection), the ratio turned negative, and bins came back NaN."""
    _, _, S = _analytic_var(_SLOW_CHAIN_COEFFICIENTS, _CHAIN_NOISE, n_fft)
    connectivity = Connectivity(
        fourier_coefficients=_fourier_coefficients_with_cross_spectrum(S)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DirectedOrientationWarning)
        granger = connectivity.pairwise_spectral_granger_prediction()[0]

    off_diagonal = ~np.eye(3, dtype=bool)
    assert np.isfinite(granger[:, off_diagonal]).all()
    assert (granger[:, off_diagonal] >= 0).all()
