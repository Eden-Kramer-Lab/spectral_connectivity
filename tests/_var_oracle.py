"""Analytic VAR oracle helpers shared by the test modules.

For a stable VAR ``x(t) = sum_k A_k x(t - k) + e(t)`` with innovation
covariance ``Sigma``, the transfer function and cross-spectrum are known
exactly; these helpers build them and Fourier coefficients that reproduce
them, so a ``Connectivity`` can be fed an exact spectrum (see
``tests/test_directed_measures_oracle.py``). The VAR's state-space form gives
what needs no spectrum: its autocovariance, the spectral Granger measure
between sub-processes, and the VAR its time reversal follows. The spectra and
the state-space Granger oracle use conjugate transposes throughout, so they
hold for a complex-valued VAR (complex coefficients and a Hermitian innovation
covariance) as well as a real one.
"""

import itertools

import numpy as np
from scipy.linalg import solve_discrete_lyapunov


def _state_space_form(coefficients, noise_covariance):
    """A VAR's state space ``(A, C, K, Q)``: ``s(t + 1) = A s(t) + K e(t)`` for
    the state ``s(t) = [x(t - 1), ..., x(t - n_lags)]``, observed as ``x(t) =
    C s(t) + e(t)``, with state noise covariance ``Q = K Sigma K^T``."""
    n_signals = coefficients.shape[-1]
    C = np.concatenate(list(coefficients), axis=1)
    A = _companion_matrix(coefficients)
    K = np.zeros((A.shape[0], n_signals))
    K[:n_signals] = np.eye(n_signals)
    return A, C, K, K @ noise_covariance @ K.T


def _observed_subsystem(C, K, noise_covariance, members):
    """``(C_sub, R, S)`` for observing only the signals ``members``: their
    observation rows, innovation covariance, and state-observation noise
    covariance."""
    return (
        C[members],
        noise_covariance[np.ix_(members, members)],
        K @ noise_covariance[:, members],
    )


def _prediction_error_covariance(A, C, Q, R, S, max_iterations=100_000):
    """Steady-state state prediction-error covariance ``P`` of a Kalman filter.

    The state ``s(t + 1) = A s(t) + w(t)`` is observed as ``y(t) = C s(t) +
    v(t)`` with ``Cov(w) = Q``, ``Cov(v) = R`` and ``Cov(w, v) = S``. ``P`` is
    the stabilizing solution of the discrete algebraic Riccati equation, found
    by iterating the filter's covariance recursion from zero, which converges
    for a stable ``A`` (Anderson & Moore 1979, ch. 4). The iteration takes
    conjugate transposes, so it holds for complex systems, for which SciPy's
    ``solve_discrete_are`` can fail its QZ step.
    """
    P = np.zeros_like(Q)
    for _ in range(max_iterations):
        cross = A @ P @ C.conj().T + S
        gain = cross @ np.linalg.inv(C @ P @ C.conj().T + R)
        updated = A @ P @ A.conj().T + Q - gain @ cross.conj().T
        updated = (updated + updated.conj().T) / 2
        change = np.abs(updated - P).max()
        # Relative to Q as well as P, which is 0 when C observes the whole state;
        # the update's rounding floor grows with both.
        tolerance = 1e-14 * (np.abs(Q).max() + np.abs(updated).max())
        P = updated
        if change <= tolerance:
            return P
    msg = (
        f"Riccati iteration did not converge: last change {change:.2e}, "
        f"tolerance {tolerance:.2e}"
    )
    raise RuntimeError(msg)


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


def _companion_matrix(coefficients):
    """Transition matrix of a VAR's state ``[x(t - 1), ..., x(t - n_lags)]``."""
    n_lags, n_signals, _ = coefficients.shape
    companion = np.zeros((n_lags * n_signals, n_lags * n_signals), dtype=coefficients.dtype)
    companion[:n_signals] = np.concatenate(list(coefficients), axis=1)
    companion[n_signals:, :-n_signals] = np.eye((n_lags - 1) * n_signals)
    return companion


def _companion_spectral_radius(coefficients):
    """Largest eigenvalue modulus of a VAR's companion matrix (< 1 is stable)."""
    return np.max(np.abs(np.linalg.eigvals(_companion_matrix(coefficients))))


def _dense_stable_var(n_signals, n_lags, spectral_radius, seed, complex_valued=False):
    """A VAR in which every signal drives every other and the innovations are
    correlated: ``(coefficients, noise_covariance)``.

    Coefficients are drawn at random and lag ``k`` is scaled by ``c**k``,
    which scales every eigenvalue of the companion matrix by ``c``, so the
    companion matrix has exactly the requested spectral radius. With
    ``complex_valued`` the coefficients and the innovation mixing are complex
    Gaussian, so the innovation covariance is Hermitian with genuinely
    imaginary off-diagonal entries and the spectrum is not
    conjugate-symmetric.
    """
    rng = np.random.default_rng(seed)

    def draw(shape):
        if complex_valued:
            return (rng.normal(size=shape) + 1j * rng.normal(size=shape)) / np.sqrt(2)
        return rng.normal(size=shape)

    coefficients = draw((n_lags, n_signals, n_signals))
    scale = spectral_radius / _companion_spectral_radius(coefficients)
    coefficients *= (scale ** np.arange(1, n_lags + 1))[:, np.newaxis, np.newaxis]
    mixing = draw((n_signals, n_signals))
    noise_covariance = mixing @ mixing.conj().T / n_signals + np.eye(n_signals)
    return coefficients, noise_covariance


def _block_granger_from_state_space(coefficients, noise_covariance, n_fft, blocks):
    """Spectral Granger between blocks of a VAR's signals, ``[frequency, source, target]``.

    An oracle that involves no spectral factorization. The signals of two
    blocks are a sub-process of the VAR; its innovations form (transfer
    function ``H`` and innovation covariance ``V``) comes from the discrete
    algebraic Riccati equation of the VAR's state-space form (Barnett & Seth,
    2015), and Geweke's formula gives the influence of each block on the
    other: for a source block ``s`` and a target block ``t``,
    ``log det S_tt - log det(S_tt - H_ts (V_ss - V_st V_tt^-1 V_ts) H_ts^H)``.
    ``blocks`` lists the signal indices of each block. Values are on the
    non-negative FFT grid; the diagonal is NaN.
    """
    A, C, K, Q = _state_space_form(coefficients, noise_covariance)
    z = np.exp(2j * np.pi * np.arange(n_fft // 2 + 1) / n_fft)
    resolvent = np.linalg.inv(z[:, None, None] * np.eye(A.shape[0]) - A)
    granger = np.full((z.size, len(blocks), len(blocks)), np.nan)
    for first, second in itertools.combinations(range(len(blocks)), 2):
        members = [*blocks[first], *blocks[second]]
        C_sub, R, S = _observed_subsystem(C, K, noise_covariance, members)
        P = _prediction_error_covariance(A, C_sub, Q, R, S)
        V = C_sub @ P @ C_sub.conj().T + R
        gain = (A @ P @ C_sub.conj().T + S) @ np.linalg.inv(V)
        H = np.eye(len(members)) + C_sub @ resolvent @ gain  # (frequency, member, member)
        spectrum = H @ V @ H.conj().swapaxes(-1, -2)
        in_first = np.arange(len(members)) < len(blocks[first])
        for source, target, source_block, target_block in (
            (in_first, ~in_first, first, second),
            (~in_first, in_first, second, first),
        ):
            intrinsic = V[np.ix_(source, source)] - V[
                np.ix_(source, target)
            ] @ np.linalg.solve(V[np.ix_(target, target)], V[np.ix_(target, source)])
            H_ts = H[:, target][:, :, source]
            total = spectrum[:, target][:, :, target]
            explained = H_ts @ intrinsic @ H_ts.conj().swapaxes(-1, -2)
            # NumPy 1.24's complex slogdet warns of a division by zero on
            # well-conditioned input on some platforms.
            with np.errstate(divide="ignore", invalid="ignore"):
                granger[:, source_block, target_block] = (
                    np.linalg.slogdet(total)[1] - np.linalg.slogdet(total - explained)[1]
                )
    return granger


def _pairwise_granger_from_state_space(coefficients, noise_covariance, n_fft):
    """Pairwise spectral Granger of a VAR, ``[frequency, source, target]``: the
    block measure with every signal a block of its own."""
    blocks = [[signal] for signal in range(coefficients.shape[-1])]
    return _block_granger_from_state_space(coefficients, noise_covariance, n_fft, blocks)


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
    A, C, K, Q = _state_space_form(coefficients, noise_covariance)
    V = noise_covariance
    reduced = [i for i in range(n_signals) if i != source]
    C_r, R, S = _observed_subsystem(C, K, V, reduced)
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


def _var_autocovariance(coefficients, noise_covariance, max_lag):
    """``Gamma(h) = E[x(t + h) x(t)^T]`` of a VAR for ``h = 0 ... max_lag``,
    shape ``(max_lag + 1, n_signals, n_signals)``.

    The covariance of the state ``[x(t - 1), ..., x(t - n_lags)]`` solves a
    discrete Lyapunov equation and holds the lags below ``n_lags``; the
    Yule-Walker recursion ``Gamma(h) = sum_k A_k Gamma(h - k)`` gives the rest.
    """
    n_lags, n_signals, _ = coefficients.shape
    companion = _companion_matrix(coefficients)
    state_noise = np.zeros_like(companion)
    state_noise[:n_signals, :n_signals] = noise_covariance
    state_covariance = solve_discrete_lyapunov(companion, state_noise)
    gamma = list(
        state_covariance[:n_signals].reshape(n_signals, n_lags, n_signals).swapaxes(0, 1)
    )
    for lag in range(n_lags, max_lag + 1):
        gamma.append(sum(coefficients[k] @ gamma[lag - k - 1] for k in range(n_lags)))
    return np.stack(gamma[: max_lag + 1])


def _time_reversed_var(coefficients, noise_covariance):
    """The VAR that a VAR follows when read backwards in time:
    ``(coefficients, noise_covariance)``.

    The reversed process ``y(s) = x(-s)`` has the autocovariance of ``x``
    transposed at every lag, and is again a VAR of the same order. Its
    coefficients ``B_k`` solve the Yule-Walker equations of that
    autocovariance, ``Gamma(j)^T = sum_k B_k Gamma(j - k)^T`` for
    ``j = 1 ... n_lags``, and its innovation covariance is
    ``Gamma(0) - sum_k B_k Gamma(k)``.
    """
    n_lags = coefficients.shape[0]
    gamma = _var_autocovariance(coefficients, noise_covariance, n_lags)

    def reversed_gamma(lag):
        return gamma[lag].T if lag >= 0 else gamma[-lag]

    lags = range(1, n_lags + 1)
    # [B_1 ... B_n] @ system = right_hand_side, block (k, j) of system being
    # the reversed autocovariance at lag j - k.
    system = np.block([[reversed_gamma(j - k) for j in lags] for k in lags])
    right_hand_side = np.concatenate([reversed_gamma(j) for j in lags], axis=1)
    reversed_coefficients = np.stack(
        np.split(np.linalg.solve(system.T, right_hand_side.T).T, n_lags, axis=1)
    )
    reversed_noise = gamma[0] - sum(
        reversed_coefficients[k] @ gamma[k + 1] for k in range(n_lags)
    )
    return reversed_coefficients, reversed_noise


# A 3-node chain 0 -> 1 -> 2 (no direct 0 -> 2 link) with 0.8 couplings, strong
# enough for simulated-data thresholds: analytic partial coherence peaks at
# 0.250 for (0, 1) and 0.817 for (1, 2) and is exactly 0 for the mediated
# (0, 2), whose pairwise coherence peaks at 0.785. Companion-matrix spectral
# radius 0.775.
_STRONG_CHAIN_COEFFICIENTS = np.stack(
    [np.array([[0.5, 0.0, 0.0], [0.8, 0.5, 0.0], [0.0, 0.8, 0.5]]), -0.6 * np.eye(3)]
)
