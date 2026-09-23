import warnings
from contextlib import nullcontext
from unittest.mock import PropertyMock, patch

import numpy as np
import pytest
import scipy.stats

from spectral_connectivity.connectivity import (
    Connectivity,
    _bandpass,
    _complex_inner_product,
    _conjugate_transpose,
    _find_largest_independent_group,
    _find_largest_significant_group,
    _get_independent_frequencies,
    _get_independent_frequency_step,
    _max_psd_discrepancy,
    _optimize_canonical_coherency_phase,
    _remove_instantaneous_causality,
    _reshape,
    _sanitized_nonnegative_granger,
    _set_diagonal_to_zero,
    _squared_magnitude,
    _total_inflow,
    _total_outflow,
)


@pytest.mark.parametrize("axis", [(0), (1), (2), (3)])
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_cross_spectrum(axis, dtype):
    """Test that the cross spectrum is correct for each dimension."""
    n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals = (2, 2, 2, 2, 2)
    fourier_coefficients = np.zeros(
        (n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals), dtype=dtype
    )

    signal_fourier_coefficient = [
        2 * np.exp(1j * np.pi / 2),
        3 * np.exp(1j * -np.pi / 2),
    ]
    fourier_ind = [slice(0, 4)] * 5
    fourier_ind[-1] = slice(None)
    fourier_ind[axis] = slice(1, 2)
    fourier_coefficients[tuple(fourier_ind)] = signal_fourier_coefficient

    expected_cross_spectral_matrix = np.zeros(
        (n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals, n_signals),
        dtype=dtype,
    )

    expected_slice = np.array([[4, -6], [-6, 9]], dtype=dtype)
    expected_ind = [slice(0, 5)] * 6
    expected_ind[-1] = slice(None)
    expected_ind[-2] = slice(None)
    expected_ind[axis] = slice(1, 2)
    expected_cross_spectral_matrix[tuple(expected_ind)] = expected_slice

    this_Conn = Connectivity(fourier_coefficients=fourier_coefficients)
    assert np.allclose(expected_cross_spectral_matrix, this_Conn._cross_spectral_matrix)


def test_minimum_phase_reconstruction_error_is_exposed_on_connectivity():
    """The public diagnostic evaluates the factor used by directed measures."""
    target_cross_spectrum = np.array([[2.0, 0.4], [0.4, 1.0]])
    cholesky_factor = np.linalg.cholesky(target_cross_spectrum)
    taper_coefficients = (np.sqrt(2.0) * cholesky_factor.T).astype(np.complex128)
    coefficients = np.broadcast_to(taper_coefficients[:, np.newaxis, :], (2, 16, 2))[
        np.newaxis, np.newaxis
    ].copy()
    connectivity = Connectivity(coefficients)

    error = connectivity.minimum_phase_reconstruction_error()

    assert error.shape == (1,)
    assert error[0] < 1e-7


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_power(dtype):
    n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals = (1, 1, 1, 1, 2)
    fourier_coefficients = np.zeros(
        (n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals), dtype=dtype
    )

    fourier_coefficients[..., :] = [
        2 * np.exp(1j * np.pi / 2),
        3 * np.exp(1j * -np.pi / 2),
    ]

    expected_power = np.zeros((n_time_samples, n_fft_samples, n_signals))

    expected_power[..., :] = [4, 9]

    this_Conn = Connectivity(fourier_coefficients=fourier_coefficients)
    assert np.allclose(expected_power, this_Conn.power())


def test_one_sided_power_and_csd_return_detached_arrays():
    """A returned array must not alias an internal cache: mutating it must not
    change later results. The one-sided path returned the cache directly."""
    rng = np.random.default_rng(6)
    coefficients = rng.standard_normal((1, 20, 3, 9, 2)) + 1j * rng.standard_normal(
        (1, 20, 3, 9, 2)
    )
    connectivity = Connectivity(coefficients, is_one_sided=True)

    power = connectivity.power()
    power *= 2.0
    np.testing.assert_allclose(connectivity.power(), power / 2.0, equal_nan=True)

    coherence_before = connectivity.coherence_magnitude()
    csd = connectivity.cross_spectral_density()
    csd *= 3.0
    np.testing.assert_allclose(
        connectivity.coherence_magnitude(), coherence_before, equal_nan=True
    )


@pytest.mark.parametrize("n_fft_samples", [5, 6])
def test_cross_spectral_density_is_one_sided_and_matches_power_diagonal(
    n_fft_samples,
):
    """CSD uses the same one-sided scaling as public power."""
    rng = np.random.default_rng(101)
    coefficients = rng.standard_normal((2, 3, 2, n_fft_samples, 3)) + 1j * (
        rng.standard_normal((2, 3, 2, n_fft_samples, 3))
    )
    conn = Connectivity(coefficients)
    csd = conn.cross_spectral_density()

    assert csd.shape == (2, n_fft_samples // 2 + 1, 3, 3)
    np.testing.assert_allclose(csd, np.conj(np.swapaxes(csd, -1, -2)))
    diagonal = np.diagonal(csd, axis1=-2, axis2=-1).real
    np.testing.assert_allclose(diagonal, conn.power())


@pytest.mark.parametrize(
    ("expectation_type", "expected_shape"),
    [("trials_tapers", (1, 4, 5)), ("trials", (1, 3, 4, 5)), ("tapers", (1, 2, 4, 5))],
)
def test_expectation(expectation_type, expected_shape):
    n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals = (1, 2, 3, 4, 5)
    fourier_coefficients = np.zeros(
        (n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals), dtype=complex
    )

    this_Conn = Connectivity(
        fourier_coefficients=fourier_coefficients,
        expectation_type=expectation_type,
    )
    expectation_function = this_Conn._expectation
    assert expectation_function(fourier_coefficients).shape == expected_shape


@pytest.mark.parametrize(
    ("expectation_type", "expected_n_observations"),
    [("trials_tapers", 6), ("trials", 2), ("tapers", 3)],
)
def test_n_observations(expectation_type, expected_n_observations):
    n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals = (1, 2, 3, 4, 5)
    fourier_coefficients = np.zeros(
        (n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals), dtype=complex
    )

    this_Conn = Connectivity(
        fourier_coefficients=fourier_coefficients,
        expectation_type=expectation_type,
    )
    assert this_Conn.n_observations == expected_n_observations


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_coherency(dtype):
    """A constant phase difference with proportional magnitudes has coherency
    of magnitude 1 and angle equal to that phase difference, even when the
    per-trial magnitudes and absolute phases vary."""
    rng = np.random.default_rng(193)
    n_time_samples, n_trials, n_tapers, n_fft_samples = (1, 30, 1, 1)
    shape = (n_time_samples, n_trials, n_tapers, n_fft_samples)
    magnitude = rng.uniform(0.5, 2.0, shape)
    common_phase = rng.uniform(0, 2 * np.pi, shape)
    # Signal 0 leads signal 1 by 2*pi/3, well away from the +/-pi branch cut.
    phase_difference = 2 * np.pi / 3
    fourier_coefficients = np.stack(
        [
            magnitude * np.exp(1j * (common_phase + phase_difference)),
            1.5 * magnitude * np.exp(1j * common_phase),
        ],
        axis=-1,
    ).astype(dtype)
    this_Conn = Connectivity(fourier_coefficients=fourier_coefficients)
    coherency = this_Conn.coherency().squeeze()

    # The diagonal is NaN by design (self-coherency is not reported).
    expected_coherence_magnitude = np.array([[np.nan, 1.0], [1.0, np.nan]])
    expected_phase = np.array([[np.nan, phase_difference], [-phase_difference, np.nan]])
    np.testing.assert_allclose(np.abs(coherency), expected_coherence_magnitude, rtol=1e-6)
    np.testing.assert_allclose(np.angle(coherency), expected_phase, atol=1e-6)


def test_imaginary_coherence_is_zero_for_in_phase_signals():
    """Test that imaginary coherence sets signals with the same phase
    to zero."""
    n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals = (1, 30, 1, 1, 2)
    fourier_coefficients = np.zeros(
        (n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals), dtype=complex
    )

    fourier_coefficients[..., :] = [2 * np.exp(1j * 0), 3 * np.exp(1j * 0)]
    expected_imaginary_coherence = np.zeros((2, 2))

    this_Conn = Connectivity(fourier_coefficients=fourier_coefficients)
    assert np.allclose(this_Conn.imaginary_coherence().squeeze(), expected_imaginary_coherence)


def test_imaginary_coherence_matches_hand_computed_definition():
    """|Im(S_01)| / sqrt(S_00 S_11) from a hand-computed trial/taper average,
    and exactly 1 for consistent quadrature coupling."""
    rng = np.random.default_rng(219)
    shape = (1, 50, 2, 1, 2)
    coefficients = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    coefficients[..., 1] += 0.7 * np.exp(1j * 1.1) * coefficients[..., 0]
    observations = coefficients[0, :, :, 0, :].reshape(-1, 2)  # (n_obs, n_signals)
    cross_01 = np.mean(observations[:, 0] * np.conj(observations[:, 1]))
    power = np.mean(np.abs(observations) ** 2, axis=0)
    expected = np.abs(cross_01.imag) / np.sqrt(power[0] * power[1])
    assert expected > 0.1  # a genuinely nonzero value

    actual = Connectivity(coefficients).imaginary_coherence().squeeze()
    assert actual[0, 1] == pytest.approx(expected, rel=1e-12)
    assert actual[1, 0] == pytest.approx(expected, rel=1e-12)
    np.testing.assert_allclose(np.diag(actual), 0.0, atol=1e-15)  # roundoff on some NumPy

    quadrature = np.ones((1, 10, 1, 1, 2), dtype=complex)
    quadrature[..., 0] = 2j * rng.uniform(0.5, 2.0, (1, 10, 1, 1))
    quadrature[..., 1] *= 2 * np.abs(quadrature[..., 0])
    quadrature_result = Connectivity(quadrature).imaginary_coherence().squeeze()
    assert quadrature_result[0, 1] == pytest.approx(1.0)


def test_imaginary_coherency_preserves_pair_orientation():
    coefficients = np.empty((1, 8, 1, 1, 2), dtype=complex)
    coefficients[..., 0] = np.exp(1j * np.pi / 2)
    coefficients[..., 1] = 1.0
    signed = Connectivity(coefficients).imaginary_coherency().squeeze()

    assert signed[0, 1] == pytest.approx(1.0)
    assert signed[1, 0] == pytest.approx(-1.0)
    assert np.isnan(signed[0, 0])


def test_partial_coherence_matches_inverse_spectral_matrix_definition():
    rng = np.random.default_rng(102)
    coefficients = rng.standard_normal((1, 200, 3, 4, 3)) + 1j * (
        rng.standard_normal((1, 200, 3, 4, 3))
    )
    conn = Connectivity(coefficients)
    actual = conn.partial_coherence(regularization=1e-10)

    spectrum = conn._expectation_cross_spectral_matrix()
    scale = np.sqrt(np.mean(np.abs(spectrum) ** 2, axis=(-2, -1), keepdims=True))
    identity = np.eye(3)
    precision = np.linalg.solve(spectrum + 1e-10 * scale * identity, identity)
    diagonal = np.diagonal(precision, axis1=-2, axis2=-1).real
    denominator = np.sqrt(diagonal[..., :, None] * diagonal[..., None, :])
    expected = np.abs(-precision / denominator) ** 2
    index = np.arange(3)
    expected[..., index, index] = np.nan

    np.testing.assert_allclose(actual, expected[..., :3, :, :], equal_nan=True)


@pytest.mark.parametrize("regularization", [-1, np.inf, np.nan, True, [0.1]])
def test_partial_coherence_rejects_invalid_regularization(regularization):
    coefficients = np.ones((1, 2, 2, 2, 2), dtype=complex)
    with pytest.raises(ValueError, match="regularization"):
        Connectivity(coefficients).partial_coherence(regularization=regularization)


def test_phase_locking_value():
    """Make sure phase locking value ignores magnitudes."""
    rng = np.random.default_rng(42)
    n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals = (1, 30, 1, 1, 2)
    fourier_coefficients = rng.uniform(
        0, 2, (n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals)
    ) * np.exp(1j * np.pi / 2)
    # (n_time_samples, n_fft_samples, n_signals, n_signals)
    expected_phase_locking_value = np.ones(
        (n_time_samples, n_fft_samples, n_signals, n_signals)
    )
    this_Conn = Connectivity(fourier_coefficients=fourier_coefficients)
    phase_locking_value = this_Conn.phase_locking_value()

    assert phase_locking_value.shape == expected_phase_locking_value.shape
    np.testing.assert_allclose(phase_locking_value, expected_phase_locking_value)


def test_corrected_imaginary_phase_locking_value():
    """ciPLV rejects zero lag and retains consistent quadrature locking."""
    zero_lag = np.ones((1, 10, 1, 1, 2), dtype=complex)
    zero_result = Connectivity(zero_lag).corrected_imaginary_phase_locking_value().squeeze()
    assert zero_result[0, 1] == 0.0

    quadrature = zero_lag.copy()
    quadrature[..., 0] = 1j
    quadrature_result = (
        Connectivity(quadrature).corrected_imaginary_phase_locking_value().squeeze()
    )
    assert quadrature_result[0, 1] == pytest.approx(1.0)
    assert quadrature_result[1, 0] == pytest.approx(1.0)


def test_phase_lag_index_sets_zero_phase_signals_to_zero():
    n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals = (1, 30, 1, 1, 2)
    fourier_coefficients = np.zeros(
        (n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals), dtype=complex
    )

    fourier_coefficients[..., :] = [2 * np.exp(1j * 0), 3 * np.exp(1j * 0)]
    expected_phase_lag_index = np.zeros((2, 2))

    this_Conn = Connectivity(fourier_coefficients=fourier_coefficients)
    assert np.allclose(this_Conn.phase_lag_index().squeeze(), expected_phase_lag_index)


@pytest.mark.parametrize(
    "phase_difference", [np.pi / 8, np.pi / 4, np.pi / 2, 3 * np.pi / 4, 7 * np.pi / 8]
)
def test_phase_lag_index_sets_angles_up_to_pi_to_same_value(phase_difference):
    """Any consistent phase lead in (0, pi) gives the same PLI of +1 (and -1
    for the reversed pair), regardless of the lag size or magnitudes."""
    rng = np.random.default_rng(42)
    n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals = (1, 30, 1, 1, 2)
    fourier_coefficients = np.zeros(
        (n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals), dtype=complex
    )
    fourier_coefficients[..., 0] = rng.uniform(
        0.1, 2, (n_time_samples, n_trials, n_tapers, n_fft_samples)
    ) * np.exp(1j * np.pi / 2)
    fourier_coefficients[..., 1] = rng.uniform(
        0.1, 2, (n_time_samples, n_trials, n_tapers, n_fft_samples)
    ) * np.exp(1j * (np.pi / 2 - phase_difference))

    expected_phase_lag_index = np.zeros((2, 2))
    expected_phase_lag_index[0, 1] = 1
    expected_phase_lag_index[1, 0] = -1

    this_Conn = Connectivity(fourier_coefficients=fourier_coefficients)

    assert np.allclose(this_Conn.phase_lag_index().squeeze(), expected_phase_lag_index)


def test_directed_phase_lag_index_orientation_and_complement():
    coefficients = np.empty((1, 12, 1, 1, 2), dtype=complex)
    coefficients[..., 0] = np.exp(1j * np.pi / 2)
    coefficients[..., 1] = 1.0
    dpli = Connectivity(coefficients).directed_phase_lag_index().squeeze()

    assert dpli[0, 1] == 1.0
    assert dpli[1, 0] == 0.0
    assert dpli[0, 0] == 0.5
    np.testing.assert_allclose(dpli + dpli.T, 1.0)


def test_mic_and_mim_reduce_to_imaginary_coherency_for_scalar_groups():
    coefficients = np.empty((1, 20, 2, 1, 2), dtype=complex)
    coefficients[..., 0] = 1j
    coefficients[..., 1] = 1.0
    conn = Connectivity(coefficients)

    mic, mic_labels = conn.maximized_imaginary_coherency([0, 1])
    mim, mim_labels = conn.multivariate_interaction_measure([0, 1])

    np.testing.assert_array_equal(mic_labels, [0, 1])
    np.testing.assert_array_equal(mim_labels, [0, 1])
    assert mic.squeeze()[0, 1] == pytest.approx(1.0)
    assert mim.squeeze()[0, 1] == pytest.approx(1.0)
    assert np.isnan(mic.squeeze()[0, 0])


def test_scalar_mic_mim_handle_edge_masked_morlet_data():
    """Edge-masked Morlet data supplies NaN bins; scalar MIC/MIM must mask them
    (as the component-resolved path does) rather than crash the batched SVD."""
    from spectral_connectivity import MorletWavelet

    rng = np.random.default_rng(7)
    wavelet = MorletWavelet(
        rng.standard_normal((1500, 4, 4)),
        sampling_frequency=250.0,
        frequencies=np.array([10.0, 20.0, 40.0]),
        smoothing_time=0.3,
        edge_mode="nan",
    )
    connectivity = Connectivity.from_transform(wavelet)
    labels = np.array([0, 0, 1, 1])
    mic, _ = connectivity.maximized_imaginary_coherency(labels)
    mim, _ = connectivity.multivariate_interaction_measure(labels)

    validity = np.asarray(wavelet.valid_time_frequency)  # (time, frequency)
    for value in (mic, mim):
        off_diagonal = value[..., 0, 1]
        assert np.all(np.isnan(off_diagonal[~validity]))  # invalid bins are NaN
        assert np.all(np.isfinite(off_diagonal[validity]))  # valid bins computed


def test_scalar_mic_mim_mask_only_participating_groups():
    """A NaN confined to one group must not invalidate a connection between two
    other, healthy groups; each connection is masked using only its own groups."""
    rng = np.random.default_rng(8)
    coefficients = rng.standard_normal((1, 40, 2, 6, 3)) + 1j * rng.standard_normal(
        (1, 40, 2, 6, 3)
    )
    coefficients[..., 2] = np.nan  # group 2 is entirely invalid
    with pytest.warns(UserWarning, match="NaN or Inf"):
        connectivity = Connectivity(coefficients)
    labels = np.array([0, 1, 2])

    for measure in (
        "maximized_imaginary_coherency",
        "multivariate_interaction_measure",
    ):
        value, _ = getattr(connectivity, measure)(labels)
        # The 0<->1 connection uses only healthy groups and must stay finite.
        assert np.all(np.isfinite(value[..., 0, 1]))
        assert np.all(np.isfinite(value[..., 1, 0]))
        # Connections that include the invalid group 2 are NaN.
        assert np.all(np.isnan(value[..., 0, 2]))
        assert np.all(np.isnan(value[..., 1, 2]))


def test_exact_cacoh_reduces_to_scalar_complex_coherency():
    rng = np.random.default_rng(923)
    first = rng.standard_normal(200) + 1j * rng.standard_normal(200)
    second = 0.6 * np.exp(-0.8j) * first + 0.8 * (
        rng.standard_normal(200) + 1j * rng.standard_normal(200)
    )
    coefficients = np.stack((first, second), axis=-1)[np.newaxis, :, np.newaxis, np.newaxis]
    connectivity = Connectivity(coefficients)

    result = connectivity.canonical_coherency([0, 1])
    score = result.scores.squeeze()
    csd = connectivity._expectation_cross_spectral_matrix().squeeze()
    coherency = csd[0, 1] / np.sqrt(csd[0, 0].real * csd[1, 1].real)

    assert abs(score) == pytest.approx(abs(coherency), rel=1e-9)
    # Filter signs are fixed by the pattern convention, so the score is exactly
    # the conjugate coherency, not merely equal up to a sign (a pi phase flip).
    ratio = score / np.conjugate(coherency)
    assert ratio.imag == pytest.approx(0.0, abs=1e-7)
    assert ratio.real == pytest.approx(1.0, rel=1e-9)


def test_cacoh_phase_distinguishes_lead_from_lag():
    """A lagging seed must not be reported with its phase shifted by pi."""
    from spectral_connectivity import Multitaper

    fs = 500.0
    t = np.arange(0, 4, 1 / fs)
    rng = np.random.default_rng(3)
    trials = []
    for _ in range(20):
        offset = rng.uniform(0, 2 * np.pi)
        x = np.sin(2 * np.pi * 20 * t + offset)
        y = np.sin(2 * np.pi * 20 * t + offset - 2.0)  # y lags x by 2 rad
        trials.append(np.stack([y, x], 1) + 0.1 * rng.standard_normal((t.size, 2)))
    data = np.stack(trials, 1)
    connectivity = Connectivity.from_multitaper(
        Multitaper(data, fs, time_window_duration=4.0, time_halfbandwidth_product=2)
    )
    index = np.argmin(np.abs(connectivity.frequencies - 20))
    pairwise_phase = connectivity.coherence_phase()[0, index, 0, 1]
    score = connectivity.canonical_coherency(np.array([0, 1])).scores[0, index, 0, 0]
    # coherence_phase[0, 1] is angle(S_yx) = phase(y) - phase(x) = -2 rad.
    assert pairwise_phase == pytest.approx(-2.0, abs=0.05)
    # Scores use the conjugate convention: angle(score) == -coherence_phase.
    assert np.angle(score) == pytest.approx(2.0, abs=0.05)
    assert np.angle(score) == pytest.approx(-pairwise_phase, abs=0.05)


def test_exact_cacoh_matches_dense_phase_grid_oracle():
    rng = np.random.default_rng(924)
    coefficients = rng.standard_normal((1, 300, 1, 1, 4)) + 1j * rng.standard_normal(
        (1, 300, 1, 1, 4)
    )
    connectivity = Connectivity(coefficients)
    result = connectivity.canonical_coherency([0, 0, 1, 1], regularization=0.0)
    csd = connectivity._expectation_cross_spectral_matrix().squeeze()

    def inverse_sqrt(matrix):
        values, vectors = np.linalg.eigh((matrix + matrix.T) / 2)
        return (vectors / np.sqrt(values)[np.newaxis, :]) @ vectors.T

    Taa = inverse_sqrt(csd[:2, :2].real)
    Tbb = inverse_sqrt(csd[2:, 2:].real)
    phases = np.linspace(0, np.pi, 20001)
    grid_maximum = max(
        np.linalg.svd(
            Taa @ np.real(np.exp(-1j * phase) * csd[:2, 2:]) @ Tbb,
            compute_uv=False,
        )[0]
        for phase in phases
    )
    assert abs(result.scores.squeeze()) == pytest.approx(grid_maximum, abs=2e-8)


def test_rich_mic_scores_filters_and_patterns_match_svd_oracle():
    rng = np.random.default_rng(925)
    coefficients = rng.standard_normal((1, 250, 1, 1, 4)) + 1j * rng.standard_normal(
        (1, 250, 1, 1, 4)
    )
    connectivity = Connectivity(coefficients)
    result = connectivity.maximized_imaginary_coherency_components(
        [0, 0, 1, 1], n_components=2, regularization=0.0
    )
    # Cross-spectral matrix C[i, j] = mean_k x_i conj(x_j), computed by hand.
    observations = coefficients[0, :, 0, 0, :]  # (n_trials, n_signals)
    csd = np.mean(observations[:, :, np.newaxis] * np.conj(observations[:, np.newaxis, :]), 0)
    filters = result.filters.squeeze()
    patterns = result.patterns.squeeze()
    first_filters = filters[:, 0, :2].T  # (n_group_signals, n_components)
    second_filters = filters[:, 1, 2:].T

    # Oracle (Ewald 2012): singular values of Re(Caa)^-1/2 Im(Cab) Re(Cbb)^-1/2;
    # filters are the whitened singular vectors.
    def inverse_sqrt(matrix):
        values, vectors = np.linalg.eigh(matrix)
        return (vectors / np.sqrt(values)) @ vectors.T

    whiten_a = inverse_sqrt(csd[:2, :2].real)
    whiten_b = inverse_sqrt(csd[2:, 2:].real)
    left, singular_values, right_h = np.linalg.svd(whiten_a @ csd[:2, 2:].imag @ whiten_b)
    np.testing.assert_allclose(result.scores.squeeze(), singular_values, rtol=1e-10)
    for component in range(2):
        # Each singular-vector pair is defined up to a joint sign flip, which
        # the outer product of the two filters removes.
        expected_outer = np.outer(whiten_a @ left[:, component], whiten_b @ right_h[component])
        np.testing.assert_allclose(
            np.outer(first_filters[:, component], second_filters[:, component]),
            expected_outer,
            atol=1e-10,
        )
    # Filters are normalized to unit variance and uncorrelated: f.T Re(C) f = I.
    np.testing.assert_allclose(
        first_filters.T @ csd[:2, :2].real @ first_filters, np.eye(2), atol=1e-12
    )
    np.testing.assert_allclose(
        second_filters.T @ csd[2:, 2:].real @ second_filters, np.eye(2), atol=1e-12
    )

    reconstructed = np.asarray(
        [
            first_filters[:, component].T @ csd[:2, 2:].imag @ second_filters[:, component]
            for component in range(2)
        ]
    )
    np.testing.assert_allclose(result.scores.squeeze(), reconstructed, atol=1e-12)
    assert np.all(np.diff(result.scores.squeeze()) <= 0)
    np.testing.assert_allclose(patterns[:, 0, :2].T, csd[:2, :2].real @ first_filters)
    np.testing.assert_allclose(patterns[:, 1, 2:].T, csd[2:, 2:].real @ second_filters)
    assert np.isnan(filters[:, 0, 2:]).all()
    assert np.isnan(filters[:, 1, :2]).all()


def test_rich_cacoh_filters_and_patterns_match_definition():
    rng = np.random.default_rng(931)
    coefficients = rng.standard_normal((1, 300, 1, 1, 4)) + 1j * rng.standard_normal(
        (1, 300, 1, 1, 4)
    )
    connectivity = Connectivity(coefficients)
    result = connectivity.canonical_coherency([0, 0, 1, 1], regularization=0.0)
    csd = connectivity._expectation_cross_spectral_matrix().squeeze()
    score = result.scores.squeeze()
    filters = result.filters.squeeze()  # (side, signal)
    patterns = result.patterns.squeeze()
    filter_a = filters[0, :2]
    filter_b = filters[1, 2:]

    # The score magnitude is the filter-projected coherence at the fitted phase.
    phi = -np.angle(score)
    projected = np.real(np.exp(-1j * phi) * csd[:2, 2:])
    assert abs(score) == pytest.approx(filter_a @ projected @ filter_b, abs=1e-8)
    # Haufe patterns = within-group real CSD @ filter.
    np.testing.assert_allclose(patterns[0, :2], csd[:2, :2].real @ filter_a, atol=1e-10)
    np.testing.assert_allclose(patterns[1, 2:], csd[2:, 2:].real @ filter_b, atol=1e-10)
    assert np.isnan(filters[0, 2:]).all()
    assert np.isnan(filters[1, :2]).all()


def test_multivariate_components_allow_more_components_for_larger_groups():
    # Groups of unequal size: the (size-5, size-5) connection supports 3
    # components while the connections touching the size-2 group return NaN for
    # the unavailable third component (a per-connection, not global, bound).
    rng = np.random.default_rng(932)
    coefficients = rng.standard_normal((1, 8, 1, 1, 12)) + 1j * rng.standard_normal(
        (1, 8, 1, 1, 12)
    )
    labels = [0, 0] + [1] * 5 + [2] * 5
    result = Connectivity(coefficients).canonical_coherency(labels, n_components=3)
    connection_labels = result.connections.tolist()
    big = connection_labels.index([1, 2])
    small = connection_labels.index([0, 1])
    assert np.isfinite(result.scores[0, :, big, 2]).all()
    assert np.isnan(result.scores[0, :, small, 2]).all()


def test_multivariate_components_reject_when_no_connection_is_large_enough():
    connectivity = Connectivity(np.ones((1, 4, 2, 1, 4), dtype=complex))
    with pytest.raises(ValueError, match="largest per-connection"):
        connectivity.canonical_coherency([0, 0, 1, 1], n_components=3)


def test_multivariate_components_rank_truncates_whitening():
    # rank equal to the group size is a no-op; a smaller rank keeps only the
    # top whitening directions and changes the result (exercises the rank branch
    # of the inverse-square-root whitening).
    rng = np.random.default_rng(934)
    coefficients = rng.standard_normal((1, 200, 1, 1, 6)) + 1j * rng.standard_normal(
        (1, 200, 1, 1, 6)
    )
    labels = [0, 0, 0, 1, 1, 1]
    connectivity = Connectivity(coefficients)
    full = connectivity.maximized_imaginary_coherency_components(labels, rank=3)
    same = connectivity.maximized_imaginary_coherency_components(labels, rank=None)
    truncated = connectivity.maximized_imaginary_coherency_components(labels, rank=2)
    np.testing.assert_allclose(full.scores, same.scores, atol=1e-10)
    assert not np.allclose(full.scores, truncated.scores)


def test_multivariate_components_warn_on_rank_deficient_group():
    rng = np.random.default_rng(933)
    coefficients = rng.standard_normal((1, 6, 2, 16, 6)) + 1j * rng.standard_normal(
        (1, 6, 2, 16, 6)
    )
    coefficients[..., 2] = coefficients[..., 0]  # duplicate channel -> rank 2 of 3
    connectivity = Connectivity(coefficients)
    with pytest.warns(UserWarning, match="null space"):
        connectivity.maximized_imaginary_coherency_components(
            [0, 0, 0, 1, 1, 1], n_components=3
        )


def test_multicomponent_cacoh_components_are_uncorrelated_within_each_group():
    """Successive components are deflated in whitened space (CCA-style), so
    the component signals ``a_k^T x`` are mutually uncorrelated with unit
    variance within each group: ``A Re(Caa) A^T = I`` for the stacked filters.
    Channel-space orthogonality (``A A^T`` diagonal) is *not* the invariant --
    it depends on the channel basis."""
    rng = np.random.default_rng(926)
    coefficients = rng.standard_normal((1, 300, 1, 1, 6)) + 1j * rng.standard_normal(
        (1, 300, 1, 1, 6)
    )
    connectivity = Connectivity(coefficients)
    result = connectivity.canonical_coherency([0, 0, 0, 1, 1, 1], n_components=3)
    csd = connectivity._expectation_cross_spectral_matrix().squeeze()
    filters = result.filters.squeeze()  # (component, side, signal)

    for side, indices in enumerate((slice(0, 3), slice(3, 6))):
        local = filters[:, side, indices]  # (component, group signal)
        component_covariance = local @ csd[indices, indices].real @ local.T
        np.testing.assert_allclose(component_covariance, np.eye(3), atol=1e-8)
    assert result.group_membership.tolist() == [
        [True, True, True, False, False, False],
        [False, False, False, True, True, True],
    ]


def test_multicomponent_cacoh_is_invariant_to_within_group_mixing():
    """CaCoh whitens each group, so an invertible real mixing of one group's
    channels must not change any component's coherence: the deflation has to
    happen in whitened space, not on the channel-space filters (which made
    component 2 depend on the channel basis)."""
    rng = np.random.default_rng(5)
    n_observations = 400
    observations = rng.standard_normal((n_observations, 6)) + 1j * rng.standard_normal(
        (n_observations, 6)
    )
    observations = observations @ (
        rng.standard_normal((6, 6)) + 1j * rng.standard_normal((6, 6))
    )
    mixing = np.eye(6)
    mixing[0, 0] = 10.0  # rescale one channel of group 0 ...
    mixing[1, 2] = 0.7  # ... and mix two others
    labels = [0, 0, 0, 1, 1, 1]

    def cacoh(data):
        coefficients = data[np.newaxis, :, np.newaxis, np.newaxis, :]
        return Connectivity(coefficients).canonical_coherency(
            labels, n_components=3, regularization=0.0
        )

    base = cacoh(observations)
    mixed = cacoh(observations @ mixing.T)
    magnitudes = np.abs(base.scores[0, 0, 0])
    assert np.all(np.diff(magnitudes) <= 1e-10)  # deflated maxima are nested
    assert magnitudes[-1] > 0.05  # three genuine components, no phantom
    np.testing.assert_allclose(np.abs(mixed.scores[0, 0, 0]), magnitudes, atol=1e-8)
    # The sign convention (dominant pattern coefficient) is not mixing-invariant,
    # so the phase is invariant modulo pi.
    np.testing.assert_allclose(
        np.exp(2j * np.angle(mixed.scores)), np.exp(2j * np.angle(base.scores)), atol=1e-7
    )


def test_single_component_cacoh_matches_reference_values():
    """``n_components=1`` is untouched by the multi-component deflation
    rewrite: these values were recorded from the previous implementation."""
    rng = np.random.default_rng(926)
    shape = (1, 300, 1, 4, 6)
    coefficients = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    mixing = rng.standard_normal((6, 6)) + 1j * rng.standard_normal((6, 6))
    connectivity = Connectivity(coefficients @ mixing.T)
    labels = [0, 0, 0, 1, 1, 1]

    result = connectivity.canonical_coherency(labels, n_components=1, regularization=0.0)
    expected = np.array(
        [
            -0.3155533259549125 + 0.53533243028994j,
            -0.45746410150711864 + 0.48380284370675963j,
            -0.4677036349374889 + 0.4739229712029209j,
        ]
    )
    np.testing.assert_allclose(result.scores[0, :, 0, 0], expected, rtol=1e-7, atol=1e-9)

    truncated = connectivity.canonical_coherency(labels, n_components=1, rank=2)
    expected_rank2 = np.array(
        [
            -0.2513435224216316 + 0.5014039550925226j,
            -0.24636457719656235 + 0.4589689712533661j,
            -0.23909146778158655 + 0.48089648128444523j,
        ]
    )
    np.testing.assert_allclose(
        truncated.scores[0, :, 0, 0], expected_rank2, rtol=1e-7, atol=1e-9
    )


def _cacoh_phase_objective(whitened, phase):
    """sigma_max(Re(exp(-i phase) W)) for one whitened cross-spectrum."""
    return np.linalg.svd(np.real(np.exp(-1j * phase) * whitened), compute_uv=False)[0]


def test_cacoh_phase_optimizer_resolves_near_equal_lobes():
    """The phase objective can have two lobes of nearly equal height. Refining
    Newton from the single best coarse-grid point picks whichever lobe happens
    to sit closer to a grid point; the optimizer must refine every candidate
    lobe and keep the true maximum."""
    n_grid = 37
    grid = np.arange(n_grid) * np.pi / n_grid
    # Lobe 1 (the true maximum, 0.80) mid-way between two coarse grid points;
    # lobe 2 (0.7995) exactly on a grid point, so the coarse grid ranks it first.
    theta_1 = grid[10] + np.pi / (2 * n_grid)
    theta_2 = grid[27]
    directions = np.eye(3)
    whitened = 0.80 * np.exp(1j * theta_1) * np.outer(directions[0], directions[0]) + (
        0.7995 * np.exp(1j * theta_2) * np.outer(directions[1], directions[1])
    )
    coarse_scores = [_cacoh_phase_objective(whitened, phase) for phase in grid]
    assert np.argmax(coarse_scores) == 27  # premise: the coarse grid prefers lobe 2

    magnitude, phase, left, right = _optimize_canonical_coherency_phase(whitened[np.newaxis])

    dense_best = max(
        _cacoh_phase_objective(whitened, phase) for phase in np.linspace(0, np.pi, 20001)
    )
    assert magnitude[0] >= max(coarse_scores) - 1e-12
    assert magnitude[0] == pytest.approx(dense_best, abs=1e-6)
    assert magnitude[0] == pytest.approx(0.80, abs=1e-6)
    # The phase is defined modulo pi; it must be lobe 1, not lobe 2.
    assert np.exp(2j * phase[0]) == pytest.approx(np.exp(2j * theta_1), abs=1e-6)
    np.testing.assert_allclose(np.abs(left[0]), directions[0], atol=1e-6)
    np.testing.assert_allclose(np.abs(right[0]), directions[0], atol=1e-6)


def test_cacoh_phase_optimizer_never_returns_below_the_coarse_grid():
    rng = np.random.default_rng(21)
    whitened = rng.standard_normal((40, 3, 4)) + 1j * rng.standard_normal((40, 3, 4))
    n_grid = 37
    grid = np.arange(n_grid) * np.pi / n_grid
    coarse_best = np.array(
        [max(_cacoh_phase_objective(matrix, phase) for phase in grid) for matrix in whitened]
    )
    magnitude, _, _, _ = _optimize_canonical_coherency_phase(whitened)
    assert np.all(magnitude >= coarse_best - 1e-12)


@pytest.mark.parametrize(
    "method", ["canonical_coherency", "maximized_imaginary_coherency_components"]
)
def test_multivariate_components_zero_components_beyond_group_rank(method):
    """A rank-deficient group cannot support more components than its rank; the
    extra "phantom" components must come back with a zero score and an all-zero
    filter/pattern, as the warning promises, not a spurious optimized value or
    an arbitrary null-space singular vector."""
    rng = np.random.default_rng(4)
    coefficients = rng.standard_normal((1, 400, 1, 1, 5)) + 1j * rng.standard_normal(
        (1, 400, 1, 1, 5)
    )
    # Append a copy of channel 0 as channel 5, making group A = {0, 2, 5} rank 2.
    coefficients = np.concatenate(
        [coefficients, coefficients[..., :1]], axis=-1
    )  # channels 0..5, channel 5 == channel 0
    connectivity = Connectivity(coefficients)
    with pytest.warns(UserWarning, match="phantom"):
        result = getattr(connectivity, method)([0, 1, 0, 1, 1, 0], n_components=3)
    # Group A ({0, 2, 5}) has a duplicated channel -> rank 2; the third component
    # is unsupported.
    phantom_score = np.abs(result.scores[..., 0, 2])
    phantom_filters = result.filters[..., 0, 2, :, :]
    phantom_patterns = result.patterns[..., 0, 2, :, :]
    assert np.nanmax(phantom_score) < 1e-10
    assert np.nanmax(np.abs(phantom_filters)) < 1e-10
    assert np.nanmax(np.abs(phantom_patterns)) < 1e-10
    # The two supported components are still non-degenerate.
    assert np.nanmax(np.abs(result.scores[..., 0, 0])) > 1e-3


@pytest.mark.parametrize(
    "method", ["canonical_coherency", "maximized_imaginary_coherency_components"]
)
def test_multivariate_components_validate_group_geometry(method):
    connectivity = Connectivity(np.ones((1, 4, 2, 1, 4), dtype=complex))
    with pytest.raises(ValueError, match="length n_signals"):
        getattr(connectivity, method)([0, 1])
    with pytest.raises(ValueError, match="n_components"):
        getattr(connectivity, method)([0, 0, 1, 1], n_components=3)


@pytest.mark.parametrize(
    "method",
    [
        "canonical_coherence",
        "canonical_coherency",
        "maximized_imaginary_coherency",
        "multivariate_interaction_measure",
        "maximized_imaginary_coherency_components",
    ],
)
@pytest.mark.parametrize("missing_label", [np.nan, None])
def test_group_measures_reject_missing_labels(method, missing_label):
    """A missing label must not create an empty, all-false signal group."""
    rng = np.random.default_rng(924)
    coefficients = rng.standard_normal((1, 6, 2, 8, 4)) + 1j * rng.standard_normal(
        (1, 6, 2, 8, 4)
    )
    connectivity = Connectivity(coefficients)

    with pytest.raises(ValueError, match="missing values"):
        getattr(connectivity, method)([0, 0, 1, missing_label])


def test_mic_rejects_single_group_and_non_positive_rank():
    coefficients = np.empty((1, 20, 2, 1, 3), dtype=complex)
    coefficients[..., 0] = 1j
    coefficients[..., 1] = 1.0
    coefficients[..., 2] = 0.5j
    conn = Connectivity(coefficients)
    with pytest.raises(ValueError, match="at least two groups"):
        conn.maximized_imaginary_coherency([0, 0, 0])
    with pytest.raises(ValueError, match="rank must be a positive integer"):
        conn.multivariate_interaction_measure([0, 1, 1], rank=0)


def test_partial_coherence_warns_and_nans_on_zero_power():
    rng = np.random.default_rng(919)
    coefficients = rng.standard_normal((1, 6, 2, 4, 3)) + 1j * rng.standard_normal(
        (1, 6, 2, 4, 3)
    )
    coefficients[:, :, :, 0, :] = 0.0  # a fully dead frequency bin
    conn = Connectivity(coefficients)
    with pytest.warns(UserWarning, match="zero power"):
        result = conn.partial_coherence(regularization=1e-10)
    assert np.isnan(result[:, 0]).all()


def test_mic_and_mim_are_invariant_to_within_group_real_mixing():
    rng = np.random.default_rng(103)
    coefficients = rng.standard_normal((1, 300, 2, 3, 4)) + 1j * (
        rng.standard_normal((1, 300, 2, 3, 4))
    )
    labels = np.array([0, 0, 1, 1])
    original = Connectivity(coefficients)

    first_mix = np.array([[2.0, 0.4], [-0.3, 1.2]])
    second_mix = np.array([[0.8, -0.2], [0.5, 1.7]])
    mixed = coefficients.copy()
    mixed[..., :2] = np.einsum("...i,ji->...j", coefficients[..., :2], first_mix)
    mixed[..., 2:] = np.einsum("...i,ji->...j", coefficients[..., 2:], second_mix)
    transformed = Connectivity(mixed)

    original_mic, _ = original.maximized_imaginary_coherency(labels)
    mixed_mic, _ = transformed.maximized_imaginary_coherency(labels)
    original_mim, _ = original.multivariate_interaction_measure(labels)
    mixed_mim, _ = transformed.multivariate_interaction_measure(labels)

    np.testing.assert_allclose(mixed_mic, original_mic, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(mixed_mim, original_mim, rtol=1e-9, atol=1e-10)


def test_cacoh_magnitude_is_at_least_mic():
    # CaCoh maximizes |Re(exp(-i*phi) * whitened CSD)| over all phases; MIC is
    # the imaginary-axis (phi = pi/2) special case, so |CaCoh| >= MIC.
    rng = np.random.default_rng(935)
    coefficients = rng.standard_normal((1, 300, 2, 3, 4)) + 1j * (
        rng.standard_normal((1, 300, 2, 3, 4))
    )
    connectivity = Connectivity(coefficients)
    labels = [0, 0, 1, 1]
    cacoh = connectivity.canonical_coherency(labels, n_components=1)
    mic = connectivity.maximized_imaginary_coherency_components(labels, n_components=1)
    assert np.all(np.abs(cacoh.scores) >= mic.scores - 1e-9)


def test_cacoh_zero_cross_spectrum_does_not_warn():
    # Independent one-hot observations give identity within-group spectra and an
    # exactly zero between-group spectrum. The phase objective is therefore flat:
    # both finite-difference derivatives are zero throughout Newton refinement.
    coefficients = np.zeros((1, 4, 1, 1, 4), dtype=complex)
    coefficients[0, :, 0, 0, :] = np.eye(4)
    connectivity = Connectivity(coefficients, is_one_sided=True, frequencies=np.array([1.0]))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = connectivity.canonical_coherency([0, 0, 1, 1], n_components=2)

    np.testing.assert_array_equal(result.scores, 0.0)


def test_component_methods_invariant_to_within_group_real_mixing():
    # Within-group whitening makes the scores invariant to invertible real
    # within-group mixing. MIC components all come from one whitened SVD, so all
    # are invariant; CaCoh's higher components deflate in (Euclidean) channel
    # space, which mixing does not preserve, so only its first component is
    # invariant (this matches mne-connectivity's deflation).
    rng = np.random.default_rng(936)
    coefficients = rng.standard_normal((1, 300, 2, 3, 4)) + 1j * (
        rng.standard_normal((1, 300, 2, 3, 4))
    )
    labels = np.array([0, 0, 1, 1])
    original = Connectivity(coefficients)
    first_mix = np.array([[2.0, 0.4], [-0.3, 1.2]])
    second_mix = np.array([[0.8, -0.2], [0.5, 1.7]])
    mixed = coefficients.copy()
    mixed[..., :2] = np.einsum("...i,ji->...j", coefficients[..., :2], first_mix)
    mixed[..., 2:] = np.einsum("...i,ji->...j", coefficients[..., 2:], second_mix)
    transformed = Connectivity(mixed)

    original_mic = original.maximized_imaginary_coherency_components(
        labels, n_components=2
    ).scores
    mixed_mic = transformed.maximized_imaginary_coherency_components(
        labels, n_components=2
    ).scores
    np.testing.assert_allclose(mixed_mic, original_mic, rtol=1e-7, atol=1e-9)

    original_cacoh = original.canonical_coherency(labels, n_components=1).scores
    mixed_cacoh = transformed.canonical_coherency(labels, n_components=1).scores
    np.testing.assert_allclose(
        np.abs(mixed_cacoh), np.abs(original_cacoh), rtol=1e-7, atol=1e-9
    )


def test_weighted_phase_lag_index_sets_zero_phase_signals_to_zero():
    n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals = (1, 30, 1, 1, 2)
    fourier_coefficients = np.zeros(
        (n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals), dtype=complex
    )

    fourier_coefficients[..., :] = [2 * np.exp(1j * 0), 3 * np.exp(1j * 0)]
    expected_phase_lag_index = np.zeros((2, 2))

    this_Conn = Connectivity(fourier_coefficients=fourier_coefficients)
    assert np.allclose(
        this_Conn.weighted_phase_lag_index().squeeze(), expected_phase_lag_index
    )


def test_weighted_phase_lag_index_equals_phase_lag_index_for_identical_observations():
    n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals = (1, 30, 1, 1, 2)
    fourier_coefficients = np.zeros(
        (n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals), dtype=complex
    )

    fourier_coefficients[..., :] = [
        1 * np.exp(1j * 3 * np.pi / 4),
        1 * np.exp(1j * 5 * np.pi / 4),
    ]

    this_Conn = Connectivity(fourier_coefficients=fourier_coefficients)
    assert np.allclose(this_Conn.phase_lag_index(), this_Conn.weighted_phase_lag_index())


def _two_signal_coefficients(first, second):
    """Stack per-observation coefficients, shape (1, n_trials, n_tapers, 1), into
    Fourier coefficients of shape (1, n_trials, n_tapers, 1, 2)."""
    return np.stack([first, second], axis=-1)


def _imaginary_cross_spectrum(coefficients):
    """Per-observation Im(z_0 conj(z_1)), flattened to shape (n_observations,)."""
    return np.imag(coefficients[..., 0] * np.conj(coefficients[..., 1])).ravel()


def _consistent_lead_coefficients(rng, lead, shape=(1, 200, 5, 1)):
    """Random magnitudes and absolute phases; signal 0 leads signal 1 by ``lead``
    (which may vary per observation)."""
    common_phase = rng.uniform(0, 2 * np.pi, shape)
    first = rng.uniform(0.5, 3, shape) * np.exp(1j * (common_phase + lead))
    second = rng.uniform(0.5, 3, shape) * np.exp(1j * common_phase)
    return _two_signal_coefficients(first, second)


def _three_quarters_lead(rng, shape=(1, 200, 5, 1)):
    """Per-observation lead of +pi/2 for 3/4 of observations and -pi/2 for 1/4."""
    n_observations = int(np.prod(shape))
    signs = np.ones(n_observations)
    signs[: n_observations // 4] = -1
    return rng.permutation(signs).reshape(shape) * np.pi / 2


def _independent_phase_coefficients(rng, shape=(1, 200, 5, 1)):
    """Two signals with independent uniform phases and random magnitudes."""
    first = rng.uniform(0.5, 3, shape) * np.exp(1j * rng.uniform(0, 2 * np.pi, shape))
    second = rng.uniform(0.5, 3, shape) * np.exp(1j * rng.uniform(0, 2 * np.pi, shape))
    return _two_signal_coefficients(first, second)


def test_debiased_squared_phase_lag_index():
    """Equals (n * PLI**2 - 1) / (n - 1) with PLI the mean sign of Im(S_01).

    The plain PLI**2 differs by ~1/n, so the closed form (not a loose
    near-zero bound) is what distinguishes the debiased estimator.
    """
    rng = np.random.default_rng(0)
    n_observations = 200 * 5

    independent = _independent_phase_coefficients(rng)
    pli = np.mean(np.sign(_imaginary_cross_spectrum(independent)))
    expected = (n_observations * pli**2 - 1) / (n_observations - 1)
    result = Connectivity(independent).debiased_squared_phase_lag_index().squeeze()
    assert result[0, 1] == pytest.approx(expected, rel=1e-12, abs=1e-15)
    assert result[1, 0] == pytest.approx(expected, rel=1e-12, abs=1e-15)
    # Unbiased under the null: within a few 1 / n of zero (not <= 0).
    assert abs(result[0, 1]) < 30 / n_observations

    # 3/4 of observations lead, 1/4 lag -> PLI = 0.5 exactly.
    coupled = _consistent_lead_coefficients(rng, _three_quarters_lead(rng))
    coupled_result = Connectivity(coupled).debiased_squared_phase_lag_index().squeeze()
    expected_coupled = (n_observations * 0.25 - 1) / (n_observations - 1)
    assert coupled_result[0, 1] == pytest.approx(expected_coupled, rel=1e-12)


def test_debiased_squared_weighted_phase_lag_index():
    """Matches ((sum Im)**2 - sum Im**2) / ((sum |Im|)**2 - sum Im**2) (Vinck
    2011, eq. 8), is unbiased (not non-positive) for independent phases, and is
    exactly 1 for a consistent phase lead."""
    rng = np.random.default_rng(0)
    n_observations = 200 * 5

    def hand_computed(coefficients):
        imaginary = _imaginary_cross_spectrum(coefficients)
        squared_sum = np.sum(imaginary**2)
        return (np.sum(imaginary) ** 2 - squared_sum) / (
            np.sum(np.abs(imaginary)) ** 2 - squared_sum
        )

    independent = _independent_phase_coefficients(rng)
    result = Connectivity(independent).debiased_squared_weighted_phase_lag_index().squeeze()
    assert result[0, 1] == pytest.approx(hand_computed(independent), rel=1e-10, abs=1e-15)
    assert result[1, 0] == pytest.approx(result[0, 1], rel=1e-12)
    assert abs(result[0, 1]) < 30 / n_observations

    coupled = _consistent_lead_coefficients(rng, _three_quarters_lead(rng))
    coupled_result = Connectivity(coupled).debiased_squared_weighted_phase_lag_index()
    expected_coupled = hand_computed(coupled)
    assert expected_coupled > 0.1
    assert coupled_result.squeeze()[0, 1] == pytest.approx(expected_coupled, rel=1e-10)

    locked = _consistent_lead_coefficients(rng, np.pi / 3)
    locked_result = Connectivity(locked).debiased_squared_weighted_phase_lag_index()
    assert locked_result.squeeze()[0, 1] == pytest.approx(1.0, rel=1e-12)


def test_pairwise_phase_consistency():
    """Matches (|sum exp(i dphi)|**2 - n) / (n**2 - n) (Vinck 2010), ignores
    magnitudes, and is unbiased (not non-positive) for independent phases."""
    rng = np.random.default_rng(0)
    n_observations = 200 * 5

    def hand_computed(coefficients):
        unit = coefficients / np.abs(coefficients)
        resultant = np.sum(unit[..., 0] * np.conj(unit[..., 1]))
        return (np.abs(resultant) ** 2 - n_observations) / (n_observations**2 - n_observations)

    independent = _independent_phase_coefficients(rng)
    ppc = Connectivity(independent).pairwise_phase_consistency().squeeze()
    assert ppc[0, 1] == pytest.approx(hand_computed(independent), rel=1e-10, abs=1e-15)
    assert ppc[1, 0] == pytest.approx(ppc[0, 1], rel=1e-12)
    np.testing.assert_allclose(np.diag(ppc), 1.0)
    # Unbiased under the null: within a few 1 / n of zero (not <= 0).
    assert abs(ppc[0, 1]) < 20 / n_observations

    # Power is ignored: unit-magnitude coefficients with the same phases give
    # the same result.
    unit_magnitude = independent / np.abs(independent)
    unit_ppc = Connectivity(unit_magnitude).pairwise_phase_consistency().squeeze()
    np.testing.assert_allclose(ppc, unit_ppc, rtol=1e-10, atol=1e-15)

    # Phase differences of pi for 3/4 and 0 for 1/4 of observations give a
    # resultant of n / 2, so PPC = (n / 4 - 1) / (n - 1).
    coupled = _consistent_lead_coefficients(rng, _three_quarters_lead(rng) + np.pi / 2)
    coupled_ppc = Connectivity(coupled).pairwise_phase_consistency().squeeze()
    expected_coupled = (n_observations / 4 - 1) / (n_observations - 1)
    assert coupled_ppc[0, 1] == pytest.approx(expected_coupled, rel=1e-10)


def test__reshape():
    n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals = (20, 100, 3, 10, 2)
    fourier_coefficients = np.zeros(
        (n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals), dtype=complex
    )
    expected_shape = (n_time_samples, n_fft_samples, n_signals, n_trials * n_tapers)
    assert np.allclose(_reshape(fourier_coefficients).shape, expected_shape)


def test__squared_magnitude():
    test_array = np.array([[1, 2], [3, 4]])
    expected_array = np.array([[1, 4], [9, 16]])
    assert np.allclose(_squared_magnitude(test_array), expected_array)


def test__conjugate_transpose():
    test_array = np.zeros((2, 2, 4), dtype=complex)
    test_array[1, ...] = [
        [1 + 2j, 3 + 4j, 5 + 6j, 7 + 8j],
        [1 - 2j, 3 - 4j, 5 - 6j, 7 - 8j],
    ]
    expected_array = np.zeros((2, 4, 2), dtype=complex)
    expected_array[1, ...] = test_array[1, ...].conj().transpose()
    assert np.allclose(_conjugate_transpose(test_array), expected_array)


def test__complex_inner_product():
    """Test that the complex inner product is taken over the last two
    dimensions."""
    test_array1 = np.zeros((3, 2, 4), dtype=complex)
    test_array2 = np.zeros((3, 2, 4), dtype=complex)

    x1 = np.ones((2, 4)) * np.exp(1j * np.pi / 2)
    x2 = np.ones((2, 4)) * np.exp(1j * 0)

    test_array1[1, :, :] = x1
    test_array2[1, :, :] = x2

    test_array1[2, :, :] = x1
    test_array2[2, :, :] = x1

    expected_inner_product = np.zeros((3, 2, 2), dtype=complex)
    expected_inner_product[1, ...] = x1.dot(x2.T.conj())
    expected_inner_product[2, ...] = x1.dot(x1.T.conj())

    assert np.allclose(
        _complex_inner_product(test_array1, test_array2), expected_inner_product
    )


def test__set_diagonal_to_zero():
    test_array = np.ones((2, 2, 2))
    expected_array = np.ones((2, 2, 2))
    expected_array[0, 0, 0] = 0
    expected_array[0, 1, 1] = 0
    expected_array[1, 0, 0] = 0
    expected_array[1, 1, 1] = 0
    assert np.allclose(_set_diagonal_to_zero(test_array), expected_array)


def test__bandpass():
    test_data = np.arange(0, 10).reshape((2, 5))
    labels = np.arange(0, 5) * 2
    labels_of_interest = [1, 5]

    expected_labels = np.array([2, 4])
    expected_data = np.array([[1, 2], [6, 7]])

    filtered_data, filtered_labels = _bandpass(test_data, labels, labels_of_interest, axis=-1)

    assert np.allclose(expected_data, filtered_data) & np.allclose(
        expected_labels, filtered_labels
    )


def test__bandpass_band_edges_are_exclusive():
    """Only frequencies strictly inside the band are kept: a bin exactly on
    either edge is dropped. This pins the documented (exclusive) semantics."""
    data = np.arange(0, 10).reshape((2, 5))
    frequencies = np.arange(0, 5) * 2.0  # 0, 2, 4, 6, 8
    filtered_data, filtered_frequencies = _bandpass(data, frequencies, [2.0, 6.0], axis=-1)
    np.testing.assert_array_equal(filtered_frequencies, [4.0])
    np.testing.assert_array_equal(filtered_data, [[2], [7]])


@pytest.mark.parametrize("measure", ["group_delay", "delay", "phase_slope_index"])
@pytest.mark.parametrize(
    "band",
    [[10.0], [10.0, 20.0, 30.0], [20.0, 10.0], [10.0, 10.0], [np.nan, 10.0], [0.0, np.inf]],
)
def test_frequencies_of_interest_must_be_two_finite_increasing_values(measure, band):
    rng = np.random.default_rng(19)
    shape = (1, 6, 2, 32, 2)
    connectivity = Connectivity(
        rng.standard_normal(shape) + 1j * rng.standard_normal(shape),
        frequencies=np.fft.fftfreq(32, d=1 / 100.0),
    )
    with pytest.raises(ValueError, match="frequencies_of_interest must be two finite"):
        getattr(connectivity, measure)(frequencies_of_interest=band)


@pytest.mark.parametrize("measure", ["group_delay", "delay", "phase_slope_index"])
@pytest.mark.parametrize(
    "band",
    [
        pytest.param([60.0, 70.0], id="beyond-nyquist"),
        pytest.param([12.6, 15.5], id="between-adjacent-bins"),
    ],
)
def test_frequencies_of_interest_must_contain_a_frequency_bin(measure, band):
    """A well-formed band holding no bin must raise, not return a silent result.

    Regression: group_delay returned all-NaN delays and delay an empty
    frequency axis without a warning; only phase_slope_index raised.
    """
    rng = np.random.default_rng(19)
    shape = (1, 6, 2, 32, 2)
    connectivity = Connectivity(
        rng.standard_normal(shape) + 1j * rng.standard_normal(shape),
        frequencies=np.fft.fftfreq(32, d=1 / 100.0),  # bins every 3.125 Hz to 50 Hz
    )
    with pytest.raises(ValueError, match="contains no frequency bin"):
        getattr(connectivity, measure)(frequencies_of_interest=band)


@pytest.mark.parametrize(
    ("frequency_difference", "frequency_resolution", "expected_step"),
    [(2.0, 5.0, 3), (5.0, 2.0, 1), (2.0, 2.0, 1)],
)
def test__get_independent_frequency_step(
    frequency_difference, frequency_resolution, expected_step
):
    step = _get_independent_frequency_step(frequency_difference, frequency_resolution)
    assert step == expected_step


@pytest.mark.parametrize(
    ("is_significant", "expected_is_significant"),
    [
        (
            np.array([False, True, True, False, True, True, True, False]),
            np.array([False, False, False, False, True, True, True, False]),
        ),
        (np.ones((10,), dtype=bool), np.ones((10,), dtype=bool)),
    ],
)
def test__find_largest_significant_group(is_significant, expected_is_significant):
    assert np.allclose(
        _find_largest_significant_group(is_significant), expected_is_significant
    )


def test__find_largest_significant_group_with_no_significant():
    is_significant = np.zeros((10,), dtype=bool)
    expected_is_significant = np.zeros((10,), dtype=bool)

    assert np.allclose(
        _find_largest_significant_group(is_significant), expected_is_significant
    )


def test__get_independent_frequencies():
    is_significant = np.zeros((10,), dtype=bool)
    is_significant[3:7] = True
    frequency_step = 2

    expected_is_significant = np.zeros((10,), dtype=bool)
    expected_is_significant[3:7:frequency_step] = True

    assert np.allclose(
        _get_independent_frequencies(is_significant, frequency_step),
        expected_is_significant,
    )


@pytest.mark.parametrize(
    ("min_group_size", "expected_is_significant"),
    [
        (3, np.zeros((10,), dtype=bool)),
        (
            1,
            np.array(
                [False, False, False, False, True, False, True, False, False, False],
                dtype=bool,
            ),
        ),
    ],
)
def test__find_largest_independent_group(min_group_size, expected_is_significant):
    is_significant = np.zeros((10,), dtype=bool)
    is_significant[1:3] = True
    is_significant[4:7] = True
    is_significant[8] = True
    frequency_step = 2

    assert np.allclose(
        _find_largest_independent_group(
            is_significant, frequency_step, min_group_size=min_group_size
        ),
        expected_is_significant,
    )


def test_largest_independent_group_vectorized_matches_reference():
    """The vectorized selection equals the per-slice reference exactly.

    ``_find_significant_frequencies`` selects the largest independent
    significant cluster per (batch, pair) slice with a single vectorized pass
    instead of ``np.apply_along_axis(_find_largest_independent_group, ...)``.
    The two must agree bit-for-bit (it is boolean logic) over random inputs and
    the edge cases (all/none significant, a single frequency, tied clusters).
    The chunked path (bounded memory) must match the single-pass path.
    """
    from spectral_connectivity import connectivity as conn_mod
    from spectral_connectivity.connectivity import (
        _largest_independent_group_along_frequency,
    )

    rng = np.random.default_rng(0)
    for _ in range(200):
        n_batch = int(rng.integers(1, 4))
        n_frequencies = int(rng.integers(1, 25))
        n_pairs = int(rng.integers(1, 6))
        is_significant = rng.random((n_batch, n_frequencies, n_pairs)) < rng.uniform(0.1, 0.9)
        frequency_step = int(rng.integers(1, 4))
        min_group_size = int(rng.integers(1, 5))
        reference = np.apply_along_axis(
            _find_largest_independent_group,
            -2,
            is_significant,
            frequency_step,
            min_group_size,
        )
        vectorized = _largest_independent_group_along_frequency(
            is_significant, frequency_step, min_group_size
        )
        np.testing.assert_array_equal(vectorized, reference)
        # A tiny chunk cap forces the slices through several bounded chunks and
        # must give the identical result.
        with patch.object(conn_mod, "_SIGNIFICANCE_SELECTION_CHUNK_ELEMENTS", 7):
            chunked = _largest_independent_group_along_frequency(
                is_significant, frequency_step, min_group_size
            )
        np.testing.assert_array_equal(chunked, reference)

    # An empty frequency band (n_frequencies == 0) must return an empty result,
    # not raise (the reshape cannot infer a -1 dimension at size 0).
    empty = np.zeros((2, 0, 3), dtype=bool)
    empty_result = _largest_independent_group_along_frequency(empty, 2, 3)
    assert empty_result.shape == (2, 0, 3)
    assert empty_result.dtype == bool
    assert empty_result.size == 0

    # Explicit edge cases, including two equal-size clusters (first is kept).
    tie = np.array([[True, True, False, True, True, False]]).reshape(1, 6, 1)
    for shape_case in (
        np.zeros((1, 8, 2), bool),
        np.ones((1, 8, 2), bool),
        np.ones((1, 1, 3), bool),
        tie,
    ):
        for frequency_step in (1, 2, 3):
            for min_group_size in (1, 3):
                np.testing.assert_array_equal(
                    _largest_independent_group_along_frequency(
                        shape_case, frequency_step, min_group_size
                    ),
                    np.apply_along_axis(
                        _find_largest_independent_group,
                        -2,
                        shape_case,
                        frequency_step,
                        min_group_size,
                    ),
                )


def test__total_inflow():
    transfer_function = np.ones((2, 3, 3))
    noise_variance = [4, 2, 3]
    expected_total_inflow = 3 * np.ones((2, 3, 1))

    assert np.allclose(_total_inflow(transfer_function, noise_variance), expected_total_inflow)


def test__total_outflow():
    MVAR_Fourier_coefficients = np.ones((2, 3, 3))
    noise_variance = np.array([0.25, 0.5, 1 / 3])
    expected_total_outflow = np.ones((2, 1, 3)) * np.sqrt(1.0 / noise_variance * 3)

    assert np.allclose(
        _total_outflow(MVAR_Fourier_coefficients, noise_variance),
        expected_total_outflow,
    )


def test__remove_instantaneous_causality():
    noise_covariance = np.zeros((2, 2, 2))
    x1 = np.array([[1, 2], [2, 4]], dtype=float)
    x2 = np.array([[8, 4], [4, 16]], dtype=float)
    noise_covariance[0, ...] = x1
    noise_covariance[1, ...] = x2

    # x -> y: var(x) - (cov(x,y) ** 2 / var(y))
    # y -> x: var(y) - (cov(x,y) ** 2 / var(x))
    expected_rotated_noise_covariance = np.zeros((2, 2, 2))

    expected_rotated_noise_covariance[0, 0, 1] = x1[1, 1] - (x1[0, 1] ** 2 / x1[0, 0])
    expected_rotated_noise_covariance[0, 1, 0] = x1[0, 0] - (x1[1, 0] ** 2 / x1[1, 1])

    expected_rotated_noise_covariance[1, 0, 1] = x2[1, 1] - (x2[0, 1] ** 2 / x2[0, 0])
    expected_rotated_noise_covariance[1, 1, 0] = x2[0, 0] - (x2[1, 0] ** 2 / x2[1, 1])

    assert np.allclose(
        _remove_instantaneous_causality(noise_covariance),
        expected_rotated_noise_covariance,
    )


def test_directed_transfer_function():
    """DTF_ij = |H_ij|**2 / sum_k |H_ik|**2 (normalized over sources)."""
    # The coefficients are unused: the transfer function is patched below.
    c = Connectivity(fourier_coefficients=np.ones((1, 1, 1, 1, 2), dtype=complex))
    # |H|**2 = [[1, 4], [9, 25]] ([target, source]); the complex entries check
    # that the squared magnitude, not the real part, is used.
    transfer_function = np.array([[1.0, 2.0j], [3.0, 4.0 - 3.0j]])
    expected = np.array([[1 / 5, 4 / 5], [9 / 34, 25 / 34]])
    with patch.object(
        Connectivity, "_transfer_function", new_callable=PropertyMock
    ) as mock_prop:
        mock_prop.return_value = transfer_function
        dtf = c.directed_transfer_function()
    np.testing.assert_allclose(dtf, expected, rtol=1e-12)


def test_partial_directed_coherence():
    """PDC_ij = |A_ij|**2 / sum_k |A_kj|**2 (normalized over targets)."""
    # The coefficients are unused: the MVAR coefficients are patched below.
    c = Connectivity(fourier_coefficients=np.ones((1, 1, 1, 1, 2), dtype=complex))
    # |A|**2 = [[1, 4], [9, 25]] ([target, source]).
    mvar_coefficients = np.array([[1.0, 2.0j], [3.0, 4.0 - 3.0j]])
    expected = np.array([[1 / 10, 4 / 29], [9 / 10, 25 / 29]])
    with patch.object(
        Connectivity, "_MVAR_Fourier_coefficients", new_callable=PropertyMock
    ) as mock_prop:
        mock_prop.return_value = mvar_coefficients
        pdc = c.partial_directed_coherence()
    np.testing.assert_allclose(pdc, expected, rtol=1e-12)


def test_directed_coherence_is_bounded_and_normalized():
    """Directed coherence must stay in [0, 1] and normalize over sources.

    Regression test for a noise-variance broadcasting bug: the source noise
    variance was applied on the target axis (-2) instead of the source axis
    (-1), producing values > 1 whenever channels had unequal noise variances.
    The squared directed coherence sums to 1 over sources (like DTF).
    """
    c = Connectivity(fourier_coefficients=np.ones((1, 1, 1, 1, 2), dtype=complex))
    transfer_function = np.arange(1, 5).reshape((2, 2)).astype(float)  # [target, src]
    noise_covariance = np.diag([10.0, 1.0])  # unequal per-source noise variances
    with (
        patch.object(
            Connectivity, "_transfer_function", new_callable=PropertyMock
        ) as mock_transfer,
        patch.object(
            Connectivity, "_noise_covariance", new_callable=PropertyMock
        ) as mock_noise,
    ):
        mock_transfer.return_value = transfer_function
        mock_noise.return_value = noise_covariance
        dc = c.directed_coherence()
        assert np.all((dc >= 0.0) & (dc <= 1.0))
        assert np.allclose(dc.sum(axis=-1), 1.0)
        expected = np.array([[10 / 14, 4 / 14], [90 / 106, 16 / 106]])
        assert np.allclose(np.squeeze(dc), expected)


def test_max_psd_discrepancy():
    """The helper reports the relative gap between the diagonal and true PSD."""
    # Diagonal covariance: the diagonal denominator equals the true PSD.
    dense_H = np.arange(1, 5).reshape((1, 1, 2, 2)).astype(float)
    assert _max_psd_discrepancy(dense_H, np.diag([1.0, 2.0])[np.newaxis]) == 0.0
    # Single signal: not assessable.
    assert _max_psd_discrepancy(np.ones((1, 1, 1, 1)), np.array([[[3.0]]])) == 0.0
    # Aggregate cross-power a pairwise-correlation threshold would miss: 20
    # equicorrelated sources at rho=0.09 (every pair < 0.1) with an all-ones
    # transfer row omit ~63% of the true PSD.
    n, rho = 20, 0.09
    equicorrelated = ((1 - rho) * np.eye(n) + rho * np.ones((n, n)))[np.newaxis]
    all_ones = np.ones((1, 1, n, n))
    assert np.isclose(_max_psd_discrepancy(all_ones, equicorrelated), 0.631, atol=0.01)
    # True power exactly 0 (a rank-deficient but valid PSD covariance) with
    # nonzero diagonal power: the diagonal formula is infinitely wrong, so the
    # discrepancy is +inf and must not be dropped as unassessable.
    singular_cov = np.array([[1.0, -1.0], [-1.0, 1.0]])[np.newaxis]
    invertible_H = np.array([[1.0, 1.0], [1.0, 0.0]])[np.newaxis, np.newaxis]
    assert _max_psd_discrepancy(invertible_H, singular_cov) == np.inf
    # No power at all (0/0) is genuinely unassessable and stays 0.
    assert _max_psd_discrepancy(np.zeros((1, 1, 2, 2)), np.eye(2)[np.newaxis]) == 0.0


def test_directed_coherence_warns_on_material_cross_power():
    """directed_coherence warns when the diagonal denominator omits material power.

    The denominator ``sum_k nv_k|H_ik|^2`` equals the true PSD ``(H Cov H^H)_ii``
    only for uncorrelated innovations. It must warn when the omitted cross-power
    is a material fraction of the true PSD -- including the dimension-aware case
    where every pairwise correlation is below any fixed threshold but the sources
    jointly omit most of the power -- and stay silent for a diagonal covariance.
    """
    dense_H = np.arange(1, 5).reshape((1, 1, 2, 2)).astype(float)
    n, rho = 20, 0.09
    equicorrelated = ((1 - rho) * np.eye(n) + rho * np.ones((n, n)))[np.newaxis]
    # (transfer_function, noise_covariance, should_warn)
    cases = [
        (dense_H, np.array([[1.0, 0.9], [0.9, 1.0]])[np.newaxis], True),
        (dense_H, np.diag([1.0, 2.0])[np.newaxis], False),
        (np.ones((1, 1, n, n)), equicorrelated, True),  # weak pairwise, large sum
        # True power 0 vs diagonal power 2 -> infinite discrepancy -> must warn.
        (
            np.array([[1.0, 1.0], [1.0, 0.0]])[np.newaxis, np.newaxis],
            np.array([[1.0, -1.0], [-1.0, 1.0]])[np.newaxis],
            True,
        ),
    ]
    for transfer_function, noise_covariance, should_warn in cases:
        c = Connectivity(
            fourier_coefficients=np.ones(
                (1, 1, 1, 1, transfer_function.shape[-1]), dtype=complex
            )
        )
        with (
            patch.object(
                Connectivity, "_transfer_function", new_callable=PropertyMock
            ) as mock_transfer,
            patch.object(
                Connectivity, "_noise_covariance", new_callable=PropertyMock
            ) as mock_noise,
        ):
            mock_transfer.return_value = transfer_function
            mock_noise.return_value = noise_covariance
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                c.directed_coherence()
            warned = any("uncorrelated MVAR innovations" in str(w.message) for w in caught)
            assert warned is should_warn


def test_single_signal_connectivity_raises_but_power_works():
    """Connectivity on a single signal raises; power() still works."""
    c = Connectivity(fourier_coefficients=np.ones((1, 2, 1, 4, 1), dtype=complex))
    for method in (c.coherence_magnitude, c.phase_locking_value, c.global_coherence):
        with pytest.raises(ValueError, match="at least 2 signals"):
            method()
    with pytest.raises(ValueError, match="at least 2 signals"):
        c.canonical_coherence(group_labels=np.array([0]))
    # Power is well-defined for a single signal.
    power = c.power()
    assert power.shape[-1] == 1
    assert np.all(np.isfinite(power))


def test_debiased_measures_require_multiple_observations():
    """Debiased PLI / PPC divide by (n_obs - 1); reject n_observations == 1."""
    # n_trials == n_tapers == 1 => n_observations == 1 with default expectation.
    c = Connectivity(fourier_coefficients=np.ones((1, 1, 1, 4, 2), dtype=complex))
    assert c.n_observations == 1
    with pytest.raises(ValueError, match="at least 2 observations"):
        c.debiased_squared_phase_lag_index()
    with pytest.raises(ValueError, match="at least 2 observations"):
        c.pairwise_phase_consistency()


def test_coherency_zero_power_returns_nan():
    """A dead (all-zero) channel yields NaN coherency, not huge values."""
    rng = np.random.default_rng(0)
    fourier = rng.standard_normal((1, 1, 2, 4, 2)) + 1j * rng.standard_normal((1, 1, 2, 4, 2))
    fourier[..., 1] = 0.0  # signal 1 is a flat/dead channel -> zero power
    c = Connectivity(fourier_coefficients=fourier)
    with pytest.warns(UserWarning, match="zero power"):
        coherency = c.coherency()
    # Pairs involving the dead channel are undefined -> NaN (not > 1).
    assert np.all(np.isnan(coherency[..., 0, 1]))
    assert np.all(np.isnan(coherency[..., 1, 0]))

    # imaginary_coherence has the same guard and must also return NaN (not a
    # clipped value) for the dead-channel pairs.
    c_imag = Connectivity(fourier_coefficients=fourier)
    with pytest.warns(UserWarning, match="zero power"):
        imag_coh = c_imag.imaginary_coherence()
    assert np.all(np.isnan(imag_coh[..., 0, 1]))
    assert np.all(np.isnan(imag_coh[..., 1, 0]))


@pytest.mark.parametrize("expectation_type", ["trials_tapers", "tapers"])
def test_phase_lag_index_family_matches_per_fcn_reference(expectation_type):
    """The tiled phase-lag-index family matches the per-fcn reference path.

    phase_lag_index, weighted_phase_lag_index and
    debiased_squared_weighted_phase_lag_index now share tiled, reduced imaginary
    cross-spectrum moments instead of re-forming the full outer product per
    ``fcn``. Each must equal the original per-fcn computation. Parametrized over
    ``expectation_type`` because ``debiased_squared_weighted_phase_lag_index``
    scales by ``n_observations``, which changes with it. Also checks that
    computing one measure does not corrupt a cached moment another relies on.
    """
    rng = np.random.default_rng(0)
    shape = (2, 8, 5, 32, 5)
    fc = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex128)

    def zero_diagonal_imag(x):
        imag = x.imag
        n_signals = imag.shape[-1]
        di = np.diag_indices(n_signals)
        imag[..., di[0], di[1]] = 0
        return imag

    def non_negative(a):  # mirror the @_non_negative_frequencies(-3) decorator
        return a[..., : a.shape[-3] // 2 + 1, :, :]

    conn = Connectivity(fc, expectation_type=expectation_type)
    n_observations = conn.n_observations
    # Reference moments computed independently by averaging a transform of the
    # per-observation cross-spectral matrix (a fresh matrix per call, since
    # zero_diagonal_imag mutates the imaginary view in place).
    mean_sign = conn._expectation(np.sign(zero_diagonal_imag(conn._cross_spectral_matrix)))
    mean_imag = conn._expectation(zero_diagonal_imag(conn._cross_spectral_matrix))
    mean_abs = conn._expectation(np.abs(zero_diagonal_imag(conn._cross_spectral_matrix)))
    mean_sq = conn._expectation(zero_diagonal_imag(conn._cross_spectral_matrix) ** 2)

    expected_pli = non_negative(mean_sign.real)
    weights = mean_abs.copy()
    weights[weights < np.finfo(float).eps] = 1
    expected_wpli = non_negative(mean_imag / weights)
    imag_sum = mean_imag * n_observations
    sq_sum = mean_sq * n_observations
    abs_sum = mean_abs * n_observations
    dwpli_weights = abs_sum**2 - sq_sum
    dwpli_weights[dwpli_weights == 0] = np.nan
    expected_dwpli = non_negative((imag_sum**2 - sq_sum) / dwpli_weights)

    np.testing.assert_array_equal(conn.phase_lag_index(), expected_pli)
    np.testing.assert_array_equal(conn.weighted_phase_lag_index(), expected_wpli)
    np.testing.assert_array_equal(
        conn.debiased_squared_weighted_phase_lag_index(), expected_dwpli
    )

    # Computing wpli (which guards its weights in place on a copy) must not
    # change a later debiased_squared_weighted_phase_lag_index result.
    warm = Connectivity(fc, expectation_type=expectation_type)
    warm.weighted_phase_lag_index()
    np.testing.assert_array_equal(
        warm.debiased_squared_weighted_phase_lag_index(), expected_dwpli
    )


def test_phase_lag_index_moments_are_computed_lazily():
    """A single-measure call computes only the moments that measure needs.

    The reduced imaginary-cross-spectrum moments are computed per key on demand,
    so a lone ``phase_lag_index`` does not also compute (or retain) the other
    measures' moments, while the family shares the ones already computed.
    """
    rng = np.random.default_rng(1)
    shape = (2, 6, 4, 16, 4)
    fc = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex128)

    pli_only = Connectivity(fc)
    pli_only.phase_lag_index()
    assert set(pli_only.__dict__["_imaginary_moment_cache"]) == {"sign"}

    wpli_only = Connectivity(fc)
    wpli_only.weighted_phase_lag_index()
    assert set(wpli_only.__dict__["_imaginary_moment_cache"]) == {
        "imaginary",
        "absolute",
    }

    # The family accumulates all four moments across measures.
    family = Connectivity(fc)
    family.phase_lag_index()
    family.weighted_phase_lag_index()
    family.debiased_squared_weighted_phase_lag_index()
    assert set(family.__dict__["_imaginary_moment_cache"]) == {
        "sign",
        "imaginary",
        "absolute",
        "squared",
    }

    # A repeated measure reuses the cached moment object (it is not recomputed).
    reuse = Connectivity(fc)
    reuse.phase_lag_index()
    cached_sign = reuse.__dict__["_imaginary_moment_cache"]["sign"]
    reuse.phase_lag_index()
    assert reuse.__dict__["_imaginary_moment_cache"]["sign"] is cached_sign


def test_phase_lag_family_uses_tiled_workspace_not_full_outer_product(monkeypatch):
    """Every PLI variant works when the full observation CSM is unavailable."""
    import spectral_connectivity.connectivity as connectivity_module

    rng = np.random.default_rng(19)
    shape = (2, 5, 3, 12, 4)
    coefficients = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    expected_conn = Connectivity(coefficients)
    expected = (
        expected_conn.phase_lag_index(),
        expected_conn.weighted_phase_lag_index(),
        expected_conn.debiased_squared_phase_lag_index(),
        expected_conn.debiased_squared_weighted_phase_lag_index(),
    )

    # Force one source signal per tile, then make any accidental access to the
    # full observation-level outer product fail loudly.
    monkeypatch.setattr(connectivity_module, "PHASE_LAG_INDEX_MAX_WORKSPACE_ELEMENTS", 1)
    tiled = Connectivity(coefficients)
    with patch.object(
        Connectivity,
        "_cross_spectral_matrix",
        new_callable=PropertyMock,
        side_effect=AssertionError("full outer product was materialized"),
    ):
        actual = (
            tiled.phase_lag_index(),
            tiled.weighted_phase_lag_index(),
            tiled.debiased_squared_phase_lag_index(),
            tiled.debiased_squared_weighted_phase_lag_index(),
        )

    for actual_measure, expected_measure in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(actual_measure, expected_measure)


def test_phase_lag_index_family_fully_cached_path_matches_cold():
    """The fully-cached (no missing key) return path yields the cold result.

    When every requested moment is already cached, the method returns purely
    from the cache. Exercise that path with the reverse family order (dwpli
    caches imaginary/absolute/squared, so wpli then needs nothing new) and a
    repeated call, and require the values to match a fresh instance. This would
    catch a cache-return bug (e.g. iterating the cache instead of the requested
    keys) that the forward-order tests, which always add a key, cannot.
    """
    rng = np.random.default_rng(2)
    shape = (2, 6, 4, 16, 4)
    fc = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex128)

    warm = Connectivity(fc)
    warm.debiased_squared_weighted_phase_lag_index()  # caches imag/abs/squared
    wpli_from_cache = warm.weighted_phase_lag_index()  # all keys already cached
    pli_first = warm.phase_lag_index()
    pli_again = warm.phase_lag_index()  # sign already cached

    cold = Connectivity(fc)
    np.testing.assert_array_equal(wpli_from_cache, cold.weighted_phase_lag_index())
    np.testing.assert_array_equal(pli_first, cold.phase_lag_index())
    np.testing.assert_array_equal(pli_first, pli_again)


def test_debiased_weighted_pli_requires_multiple_observations():
    """debiased_squared_weighted_phase_lag_index guards n_observations < 2."""
    c = Connectivity(fourier_coefficients=np.ones((1, 1, 1, 4, 2), dtype=complex))
    assert c.n_observations == 1
    with pytest.raises(ValueError, match="at least 2 observations"):
        c.debiased_squared_weighted_phase_lag_index()


def test_subset_pairwise_granger_prediction():
    rng = np.random.default_rng(0)
    n_trials, n_time = 20, 64

    # Causal signals x -> y, with independent noise on y so the spectrum is
    # full rank.
    x = rng.standard_normal((n_trials, n_time))
    y = 0.1 * rng.standard_normal((n_trials, n_time))
    y[:, 1:] += 0.8 * x[:, :-1]

    # (n_trials, n_time, n_signals) -> two-sided FFT over time, then
    # (n_time_windows, n_trials, n_tapers, n_fft_samples, n_signals).
    fft_data = np.fft.fft(np.stack([x, y], axis=-1), axis=1)
    fourier_coefficients = fft_data[np.newaxis, :, np.newaxis, :, :]
    c = Connectivity(fourier_coefficients=fourier_coefficients)
    pairs = np.array([[0, 1]])
    gp_subset = c.subset_pairwise_spectral_granger_prediction(pairs)
    gp_all = c.pairwise_spectral_granger_prediction()
    # Output [i, j] is j -> i: the x -> y influence dominates y -> x.
    assert np.nanmean(gp_all[..., 1, 0]) > 10 * np.nanmean(gp_all[..., 0, 1])
    assert gp_subset.shape == gp_all.shape
    for i, j in pairs:
        assert np.allclose(gp_subset[..., i, j], gp_all[..., i, j], equal_nan=True)
        assert np.allclose(gp_subset[..., j, i], gp_all[..., j, i], equal_nan=True)


def test_subset_pairwise_granger_prediction_masks_global_diagonal():
    """Scattering compact pair blocks must not expose numerical self-Granger."""
    rng = np.random.default_rng(4)
    shape = (1, 8, 3, 16, 4)
    coefficients = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    conn = Connectivity(coefficients)

    subset = conn.subset_pairwise_spectral_granger_prediction(np.array([[0, 1], [2, 3]]))

    assert np.isnan(np.diagonal(subset, axis1=-2, axis2=-1)).all()


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_sanitized_nonnegative_granger_enforces_invariant(dtype):
    """The shared sanitizer clips roundoff, NaNs real negatives, keeps the rest."""
    eps = np.finfo(dtype).eps
    tiny_negative = -10 * eps  # inside the 100 * eps roundoff band
    material_negative = -1e-3  # far outside the roundoff band
    value = np.array([0.0, 0.5, tiny_negative, material_negative, np.nan], dtype=dtype)

    sanitized = _sanitized_nonnegative_granger(value)

    # Exact zero (no causality) is preserved, not discarded as NaN.
    assert sanitized[0] == 0.0
    # A genuine positive influence passes through untouched.
    assert sanitized[1] == dtype(0.5)
    # Roundoff around a true zero is clipped up to exactly zero.
    assert sanitized[2] == 0.0
    # A materially-negative (invalid) value becomes NaN.
    assert np.isnan(sanitized[3])
    # NaN propagates unchanged.
    assert np.isnan(sanitized[4])


def test_spectral_granger_variants_use_sanitizer_and_return_nonnegative():
    """Every Granger path sanitizes a nonempty set of finite results.

    A well-conditioned input avoids the all-NaN degeneracy that would make a
    non-negativity assertion pass vacuously. The raw (pre-sanitization) values
    for this input are all non-negative, so the non-negativity assertion alone
    cannot detect a missing sanitizer; the call-count assertion on the wrapped
    helper is what verifies that every variant routes through it.
    """
    rng = np.random.default_rng(0)
    n_time, n_trials, n_tapers, n_fft, n_signals = 1, 20, 3, 32, 4
    shape = (n_time, n_trials, n_tapers, n_fft, n_signals)
    coefficients = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    conn = Connectivity(coefficients)

    computations = {
        "pairwise": conn.pairwise_spectral_granger_prediction,
        "conditional": conn.conditional_spectral_granger_prediction,
        "blockwise": lambda: conn.blockwise_spectral_granger_prediction(
            np.array([0, 0, 1, 1])
        )[0],
    }
    with patch(
        "spectral_connectivity.connectivity._sanitized_nonnegative_granger",
        wraps=_sanitized_nonnegative_granger,
    ) as sanitizer:
        for name, compute in computations.items():
            sanitizer.reset_mock()
            result = compute()
            assert sanitizer.call_count > 0, f"{name} bypassed the shared sanitizer"
            finite = np.isfinite(result)
            assert finite.any(), f"{name} returned no finite values"
            assert np.all(result[finite] >= 0.0), name


def test_jackknife_requires_three_observations():
    """With two observations each replicate has one, which forces magnitude-
    normalized measures to 1 and yields a zero-width interval."""
    rng = np.random.default_rng(8)
    shape = (1, 2, 1, 8, 2)
    connectivity = Connectivity(rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
    with pytest.raises(ValueError, match="at least 3 observations"):
        connectivity.jackknife("coherence_magnitude")


def test_jackknife_rejects_structured_result_measures():
    """Component-result measures are not arrays; fail with a clear message
    rather than deep inside the interval computation."""
    rng = np.random.default_rng(9)
    shape = (1, 6, 2, 16, 3)
    connectivity = Connectivity(rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
    with pytest.raises(TypeError, match="real array result"):
        connectivity.jackknife("canonical_coherency", group_labels=np.array([0, 0, 1]))


@pytest.mark.parametrize(
    ("measure", "expected"),
    [
        ("coherence_magnitude", "fisher_squared"),
        ("partial_coherence", "fisher_squared"),
        ("phase_locking_value", "fisher"),
        ("imaginary_coherence", "fisher"),
        ("coherence_phase", "circular"),
        ("power", "log"),
        ("phase_lag_index", "identity"),
    ],
)
def test_jackknife_auto_transformation_matches_the_measure_range(measure, expected):
    """``"auto"`` picks the variance-stabilizing scale for each measure family:
    atanh(sqrt(.)) for magnitude-squared measures in [0, 1], atanh(.) for
    magnitudes in [0, 1], circular for phases, log for power, identity otherwise.
    """
    rng = np.random.default_rng(11)
    shape = (1, 6, 2, 8, 3)
    connectivity = Connectivity(rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
    if measure == "coherence_phase":
        # White noise has no resolvable phase, so the circular interval spans
        # the whole circle and is reported as such.
        with pytest.warns(UserWarning, match="covers the whole circle"):
            result = connectivity.jackknife(measure)
    else:
        result = connectivity.jackknife(measure)
    assert result.transformation == expected


@pytest.mark.parametrize(
    ("method", "transformation"),
    [("phase_locking_value", "fisher"), ("coherence_magnitude", "fisher_squared")],
)
def test_jackknife_warns_only_for_off_diagonal_saturation(method, transformation):
    """A duplicated channel saturates the off-diagonal pair, which the Fisher
    warning must count; PLV's diagonal is 1 by definition (the coherency-derived
    measures NaN theirs) and must not be counted.

    Regression: every PLV jackknife warned, counting the diagonal (142 entries
    here instead of 66), and off-diagonal values a few ulp below 1 -- pinned by
    the atanh clip all the same -- were not counted (48 instead of 66 for
    coherence_magnitude).
    """
    from spectral_connectivity import Multitaper

    rng = np.random.default_rng(12)
    signal = rng.standard_normal((64, 5))
    time_series = np.stack([signal, rng.standard_normal((64, 5)), signal], axis=-1)
    connectivity = Connectivity.from_multitaper(
        Multitaper(time_series, sampling_frequency=64, time_halfbandwidth_product=1)
    )
    estimate = getattr(connectivity, method)()
    n_frequencies = estimate.shape[-3]
    # The warning's criterion: pinned by the atanh clip (on the magnitude scale
    # for fisher_squared). Whether a duplicated pair lands exactly there at a
    # given bin depends on the NumPy version's rounding, so count it here.
    magnitude = np.sqrt(estimate) if transformation == "fisher_squared" else np.abs(estimate)
    pinned = magnitude >= 1 - np.finfo(float).eps
    off_diagonal = ~np.eye(3, dtype=bool)
    expected = int(np.count_nonzero(pinned[..., off_diagonal]))
    # Only the duplicated pair saturates, at nearly every frequency.
    assert np.count_nonzero(pinned[..., [0, 1], [1, 2]]) == 0
    assert expected > n_frequencies

    with pytest.warns(UserWarning, match="saturated coherence") as record:
        result = connectivity.jackknife(method)

    assert result.transformation == transformation
    assert len(record) == 1
    assert f"Fisher jackknife: {expected} value(s)" in str(record[0].message)


@pytest.mark.parametrize(
    "method",
    ["minimum_phase_reconstruction_error", "from_transform", "from_multitaper"],
)
def test_jackknife_rejects_public_non_measure_methods(method):
    """Diagnostics and alternate constructors are not connectivity measures."""
    rng = np.random.default_rng(10)
    shape = (1, 3, 2, 16, 2)
    connectivity = Connectivity(rng.standard_normal(shape) + 1j * rng.standard_normal(shape))

    with pytest.raises(ValueError, match="public connectivity measure"):
        connectivity.jackknife(method)


class _UfuncWhereRejectingNamespace:
    """NumPy stand-in for the ``xp`` backend namespace whose ufuncs reject the
    ``where=`` keyword, emulating CuPy's ufunc signature on the CPU."""

    def __getattr__(self, name):
        attribute = getattr(np, name)
        if not isinstance(attribute, np.ufunc):
            return attribute

        def strict_ufunc(*args, **kwargs):
            if "where" in kwargs:
                msg = f"{name}() got an unexpected keyword argument 'where'"
                raise TypeError(msg)
            return attribute(*args, **kwargs)

        return strict_ufunc


def test_weighted_paths_avoid_ufunc_where_keyword(monkeypatch):
    """CuPy ufuncs reject the public ``where=`` keyword, so no backend call may
    use it. Emulate that restriction by swapping every module's ``xp`` namespace
    for one whose ufuncs raise on ``where=``, then exercise every path that
    divides under a mask: weighted expectations, adaptive tapers, CaCoh."""
    from spectral_connectivity import (
        MorletWavelet,
        Multitaper,
        minimum_phase_decomposition,
        transforms,
    )
    from spectral_connectivity import connectivity as connectivity_module

    strict_namespace = _UfuncWhereRejectingNamespace()
    for module in (connectivity_module, transforms, minimum_phase_decomposition):
        assert module.xp is np  # the CPU backend this emulation replaces
        monkeypatch.setattr(module, "xp", strict_namespace)
    with pytest.raises(TypeError, match="where"):
        strict_namespace.divide(np.ones(2), np.ones(2), where=np.ones(2, dtype=bool))

    rng = np.random.default_rng(7)
    data = rng.standard_normal((600, 3, 3))
    wavelet = MorletWavelet(
        data, 200.0, [10.0, 20.0], smoothing_time=0.1, smoothing_kernel="hann"
    )
    weighted = Connectivity.from_transform(wavelet)
    assert np.isfinite(weighted.power()).any()
    assert np.isfinite(weighted.coherence_magnitude()).any()
    adaptive = Multitaper(data, 200.0, taper_weighting="adaptive")
    assert np.isfinite(adaptive.fft()).all()
    conn = Connectivity.from_multitaper(Multitaper(data, 200.0))
    assert np.isfinite(conn.canonical_coherency(np.array([0, 0, 1])).scores).any()


@pytest.mark.parametrize(
    "measure",
    ["pairwise_spectral_granger_prediction", "time_reversed_spectral_granger_prediction"],
)
def test_pairwise_granger_warns_when_a_pair_factorization_fails(measure, monkeypatch):
    """A LinAlgError inside one pair's factorization used to be swallowed
    silently, leaving NaN with no explanation; it must warn once, naming the
    measure and the affected pairs."""
    from spectral_connectivity import connectivity as connectivity_module

    rng = np.random.default_rng(18)
    shape = (1, 6, 2, 16, 3)
    connectivity = Connectivity(rng.standard_normal(shape) + 1j * rng.standard_normal(shape))

    def failing_transfer_function(*args, **kwargs):
        error_message = "Singular matrix"
        raise np.linalg.LinAlgError(error_message)

    monkeypatch.setattr(
        connectivity_module, "_estimate_transfer_function", failing_transfer_function
    )
    with pytest.warns(UserWarning, match="source -> target") as record:
        result = getattr(connectivity, measure)()
    assert np.isnan(result).all()
    assert len(record) == 1
    assert measure in str(record[0].message)
    assert "0 -> 1, 0 -> 2, 1 -> 0, 1 -> 2, 2 -> 0, 2 -> 1" in str(record[0].message)


def _granger_connectivity(defect=None):
    """Three noise signals; optionally signal 2 duplicates signal 0 or is dead."""
    from spectral_connectivity import Multitaper

    signals = np.random.default_rng(0).standard_normal((512, 10, 3))
    if defect == "duplicated":
        signals[..., 2] = signals[..., 0]
    elif defect == "dead":
        signals[..., 2] = 0.0
    return Connectivity.from_multitaper(
        Multitaper(signals, sampling_frequency=256, time_halfbandwidth_product=2)
    )


_GRANGER_MEASURES = {
    "pairwise_spectral_granger_prediction": lambda c: c.pairwise_spectral_granger_prediction(),
    "time_reversed_spectral_granger_prediction": (
        lambda c: c.time_reversed_spectral_granger_prediction()
    ),
    "subset_pairwise_spectral_granger_prediction": (
        lambda c: c.subset_pairwise_spectral_granger_prediction([(0, 2), (0, 1)])
    ),
    "conditional_spectral_granger_prediction": (
        lambda c: c.conditional_spectral_granger_prediction()
    ),
    "blockwise_spectral_granger_prediction": (
        lambda c: c.blockwise_spectral_granger_prediction(["a", "b", "c"])[0]
    ),
}


@pytest.mark.parametrize(
    ("measure", "defect", "named_pairs"),
    [
        # A duplicated channel makes the (0, 2) factorization singular.
        ("pairwise_spectral_granger_prediction", "duplicated", "0 -> 2, 2 -> 0"),
        ("time_reversed_spectral_granger_prediction", "duplicated", "0 -> 2, 2 -> 0"),
        ("subset_pairwise_spectral_granger_prediction", "duplicated", "0 -> 2, 2 -> 0"),
        # Conditioning on every signal makes every pair's full model singular.
        (
            "conditional_spectral_granger_prediction",
            "duplicated",
            "0 -> 1, 0 -> 2, 1 -> 0, 1 -> 2, 2 -> 0, 2 -> 1",
        ),
        ("blockwise_spectral_granger_prediction", "duplicated", "'a' -> 'c', 'c' -> 'a'"),
        # A dead channel has no power to explain, so influences on it are NaN.
        ("pairwise_spectral_granger_prediction", "dead", "0 -> 2, 1 -> 2"),
        ("time_reversed_spectral_granger_prediction", "dead", "0 -> 2, 1 -> 2"),
        ("subset_pairwise_spectral_granger_prediction", "dead", "0 -> 2"),
        ("blockwise_spectral_granger_prediction", "dead", "'a' -> 'c', 'b' -> 'c'"),
    ],
)
def test_granger_warns_once_naming_the_nan_pairs(measure, defect, named_pairs):
    """A real duplicated or dead channel must produce one warning that names
    the NaN pairs as source -> target, with advice that exists.

    Regression: the pair-naming warning fired only when a test monkeypatched a
    LinAlgError; for these inputs the Wilson factorization swallowed the
    failure and warned "did not converge for 1 of 1 sub-spectrum" (or nothing,
    for a dead channel) without naming the pair, and the advice mentioned a
    regularization parameter no Granger method has.
    """
    connectivity = _granger_connectivity(defect)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        result = _GRANGER_MEASURES[measure](connectivity)

    messages = [str(warning.message) for warning in record]
    named = [message for message in messages if "source -> target" in message]
    assert len(named) == 1, messages
    assert measure in named[0]
    assert f": {named_pairs} (" in named[0]
    assert "minimum_phase_max_iterations" in named[0]
    assert "regularization" not in named[0]
    # The generic, pair-less non-convergence warning is not repeated for the
    # measure's own factorizations, and no other warning reports the same NaN
    # bins. Only conditional Granger's full model is the factorization cached
    # and shared with DTF, PDC and the other directed measures; its warning is
    # kept because it is the only one those measures get.
    n_generic = sum("Wilson minimum-phase" in message for message in messages)
    shares_full_model = measure == "conditional_spectral_granger_prediction"
    assert n_generic == int(shares_full_model), messages
    assert not any("not positive" in message for message in messages), messages
    assert not any("not positive-definite" in message for message in messages), messages
    assert not any(issubclass(warning.category, RuntimeWarning) for warning in record)
    assert np.isnan(result).any()


@pytest.mark.parametrize("measure", list(_GRANGER_MEASURES))
def test_granger_is_silent_on_well_conditioned_signals(measure):
    connectivity = _granger_connectivity()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = _GRANGER_MEASURES[measure](connectivity)
    off_diagonal = ~np.eye(result.shape[-1], dtype=bool)
    if measure == "subset_pairwise_spectral_granger_prediction":
        off_diagonal[1, 2] = off_diagonal[2, 1] = False  # pair (1, 2) not requested
    assert np.isfinite(result[..., off_diagonal]).all()


def test_conditional_granger_factorizes_each_channel_set_once():
    """The full system and each leave-one-source-out system are factorized once.

    Every (target, source) pair reuses those factorizations, so the cost is
    ``n_signals + 1`` Wilson factorizations rather than ``n_signals ** 2``.
    """
    from spectral_connectivity import connectivity as connectivity_module

    rng = np.random.default_rng(1)
    shape = (1, 20, 3, 32, 4)
    coefficients = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    conn = Connectivity(coefficients)
    with patch.object(
        connectivity_module,
        "minimum_phase_decomposition",
        wraps=connectivity_module.minimum_phase_decomposition,
    ) as factorize:
        conn.conditional_spectral_granger_prediction()
    assert factorize.call_count == shape[-1] + 1


@pytest.mark.parametrize(
    ("measure", "kwargs"),
    [
        ("conditional_spectral_granger_prediction", {}),
        ("blockwise_spectral_granger_prediction", {"group_labels": [0, 0, 1]}),
    ],
)
def test_complex64_granger_variants_report_the_working_precision(measure, kwargs):
    """Every spectral Granger variant is computed from the same Wilson
    factorization, which runs at complex128 (or better) regardless of the
    requested dtype, so all of them must report the same float64 result dtype
    as pairwise_spectral_granger_prediction rather than downcasting to the
    complex64 spectrum's float32 and misrepresenting the working precision."""
    rng = np.random.default_rng(0)
    shape = (1, 10, 3, 32, 3)
    shared = rng.standard_normal((*shape[:-1], 1)) + 1j * rng.standard_normal((*shape[:-1], 1))
    noise = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    connectivity = Connectivity(
        (shared + 0.35 * noise).astype(np.complex64), dtype=np.complex64
    )

    result = getattr(connectivity, measure)(**kwargs)
    result = result[0] if isinstance(result, tuple) else result
    pairwise = connectivity.pairwise_spectral_granger_prediction()
    assert pairwise.dtype == np.float64
    assert result.dtype == pairwise.dtype


def test_complex64_directed_measure_uses_viable_wilson_precision():
    """Correlated complex64 spectra must converge at the default 1e-8 tolerance."""
    rng = np.random.default_rng(0)
    shape = (1, 10, 3, 32, 3)
    shared = rng.standard_normal((*shape[:-1], 1)) + 1j * rng.standard_normal((*shape[:-1], 1))
    noise = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    coefficients64 = (shared + 0.35 * noise).astype(np.complex64)

    result64 = Connectivity(
        coefficients64, dtype=np.complex64
    ).pairwise_spectral_granger_prediction()
    result128 = Connectivity(
        coefficients64.astype(np.complex128), dtype=np.complex128
    ).pairwise_spectral_granger_prediction()

    off_diagonal = ~np.eye(shape[-1], dtype=bool)
    assert np.isfinite(result64[..., off_diagonal]).all()
    np.testing.assert_allclose(
        result64[..., off_diagonal],
        result128[..., off_diagonal],
        rtol=1e-5,
        atol=1e-7,
    )


def test_subset_cross_spectral_matrix_is_compact_and_fully_initialized():
    """Subset computation carries one 2-by-2 matrix per pair, not a full CSM."""
    rng = np.random.default_rng(12)
    coefficients = rng.standard_normal((2, 3, 2, 8, 5)) + 1j * rng.standard_normal(
        (2, 3, 2, 8, 5)
    )
    conn = Connectivity(coefficients)
    pairs = np.array([[0, 3], [2, 4]])

    compact = conn._subset_cross_spectral_matrix(pairs)
    full = conn._cross_spectral_matrix
    assert compact.shape == (2, 3, 2, 2, 8, 2, 2)
    for pair_number, pair in enumerate(pairs):
        expected = full[..., pair[:, None], pair[None, :]]
        actual = np.take(compact, pair_number, axis=-4)
        np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    "expectation_type",
    [
        "time",
        "trials",
        "tapers",
        "time_trials",
        "time_tapers",
        "trials_tapers",
        "time_trials_tapers",
    ],
)
def test_compact_subset_cross_spectrum_preserves_every_expectation(expectation_type):
    rng = np.random.default_rng(17)
    coefficients = rng.standard_normal((2, 3, 2, 8, 5)) + 1j * rng.standard_normal(
        (2, 3, 2, 8, 5)
    )
    conn = Connectivity(coefficients, expectation_type=expectation_type)
    pairs = np.array([[0, 3], [2, 4]])

    compact = conn._expectation(conn._subset_cross_spectral_matrix(pairs))
    full = conn._expectation(conn._cross_spectral_matrix)
    for pair_number, pair in enumerate(pairs):
        expected = full[..., pair[:, None], pair[None, :]]
        actual = np.take(compact, pair_number, axis=-4)
        np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    ("n_fft_samples", "expected_n_frequencies"),
    [
        (1024, 513),  # even N: N // 2 + 1 frequencies, including Nyquist
        (1023, 512),  # odd N: (N + 1) // 2 frequencies, no Nyquist bin
    ],
)
def test_nyquist_bin_count(n_fft_samples, expected_n_frequencies):
    """Non-negative frequency count for even and odd FFT lengths."""
    rng = np.random.default_rng(42)
    n_time_samples, n_trials, n_tapers, n_signals = 1, 2, 1, 2

    # Create random fourier coefficients with full frequency spectrum
    fourier_coefficients = rng.random(
        (n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals)
    ).astype(complex)

    c = Connectivity(fourier_coefficients=fourier_coefficients)

    # Test coherence which uses @_non_negative_frequencies decorator
    coherence = c.coherence_magnitude()

    assert coherence.shape[-3] == expected_n_frequencies, (
        f"Expected {expected_n_frequencies} frequencies, got {coherence.shape[-3]}"
    )


@pytest.mark.parametrize("n_samples", [1000, 1023])
def test_nyquist_frequency_sign(n_samples):
    """Frequencies are the non-negative grid k * fs / N for even and odd N.

    Regression test for issue where fftfreq() returns negative Nyquist
    for even N, causing frequency axis misalignment in spectrograms. For odd N
    there is no Nyquist bin and the last frequency is below Nyquist.
    """
    from spectral_connectivity.transforms import Multitaper, prepare_time_series

    sampling_frequency = 1500
    rng = np.random.default_rng(0)
    signal = rng.standard_normal(n_samples)

    signal_3d = prepare_time_series(signal)
    multitaper = Multitaper(
        signal_3d, sampling_frequency=sampling_frequency, n_fft_samples=n_samples
    )
    connectivity = Connectivity.from_multitaper(multitaper)
    n_fft = multitaper.n_fft_samples
    assert n_fft == n_samples

    freqs = connectivity.frequencies
    expected = np.arange(n_fft // 2 + 1) * sampling_frequency / n_fft
    np.testing.assert_allclose(freqs, expected)

    nyquist = sampling_frequency / 2
    if n_fft % 2 == 0:
        assert np.isclose(freqs[-1], nyquist), (
            f"Last frequency should be Nyquist ({nyquist} Hz), got {freqs[-1]} Hz"
        )
    else:
        assert freqs[-1] < nyquist, (
            f"For odd N, last frequency should be < Nyquist ({nyquist} Hz), got {freqs[-1]} Hz"
        )


def test_spectrogram_frequency_alignment():
    """Test that spectrogram power peaks align with correct frequencies.

    Regression test for frequency axis misalignment where negative Nyquist
    caused spectrograms to show power at wrong frequencies.
    """
    from spectral_connectivity.transforms import Multitaper, prepare_time_series

    # Create signal with known frequency content
    sampling_frequency = 500
    duration = 10
    time = np.arange(0, duration, 1 / sampling_frequency)
    n_time_samples = len(time)

    # Signal with 50 Hz that turns on at t=5s, plus constant 100 Hz
    signal_50 = np.sin(2 * np.pi * time * 50)
    signal_50[: n_time_samples // 2] = 0  # Turn on at t=5s
    signal_100 = np.sin(2 * np.pi * time * 100)
    signal = signal_50 + signal_100

    # Compute spectrogram
    signal_3d = prepare_time_series(signal)
    multitaper = Multitaper(
        signal_3d,
        sampling_frequency=sampling_frequency,
        time_halfbandwidth_product=1,
        time_window_duration=1.0,
        time_window_step=0.5,
    )
    connectivity = Connectivity.from_multitaper(multitaper)
    power = connectivity.power()

    # Find frequency bins
    freqs = connectivity.frequencies
    freq_50_idx = np.argmin(np.abs(freqs - 50))
    freq_100_idx = np.argmin(np.abs(freqs - 100))

    # Verify frequencies are correct
    assert np.abs(freqs[freq_50_idx] - 50) < 2, (
        f"50 Hz bin at {freqs[freq_50_idx]} Hz, should be ~50 Hz"
    )
    assert np.abs(freqs[freq_100_idx] - 100) < 2, (
        f"100 Hz bin at {freqs[freq_100_idx]} Hz, should be ~100 Hz"
    )

    # Verify power dynamics
    power_50 = power[:, freq_50_idx, 0]
    power_100 = power[:, freq_100_idx, 0]

    # 50 Hz should increase dramatically after t=5s
    power_50_before = power_50[: len(power_50) // 2].mean()
    power_50_after = power_50[len(power_50) // 2 :].mean()
    assert power_50_after > 100 * power_50_before, (
        "50 Hz power should increase >100x after t=5s"
    )

    # 100 Hz should remain constant
    power_100_before = power_100[: len(power_100) // 2].mean()
    power_100_after = power_100[len(power_100) // 2 :].mean()
    ratio = power_100_after / power_100_before
    assert 0.5 < ratio < 2.0, f"100 Hz power should be constant (ratio ~1.0), got {ratio:.2f}"


def _near_singular_fourier(perturbation, seed=999):
    """Build highly correlated (near-singular) Fourier coefficients."""
    rng = np.random.default_rng(seed)
    n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals = (1, 10, 1, 5, 3)
    shape = (n_time_samples, n_trials, n_tapers, n_fft_samples)
    fourier_coefficients = np.zeros((*shape, n_signals), dtype=complex)
    base_signal = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    fourier_coefficients[..., 0] = base_signal
    for signal in (1, 2):
        fourier_coefficients[..., signal] = base_signal + perturbation * (
            rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        )
    return fourier_coefficients


def test_mvar_regularized_inverse_ill_conditioned_is_finite():
    """Tikhonov regularization keeps an ill-conditioned (full-rank) inverse finite."""
    conn = Connectivity(fourier_coefficients=_near_singular_fourier(1e-2))
    mvar_coeffs = conn._MVAR_Fourier_coefficients
    assert mvar_coeffs is not None
    assert np.all(np.isfinite(mvar_coeffs))
    assert np.all(np.isfinite(conn._transfer_function))


def test_mvar_rank_deficient_fails_gracefully_without_linalg_error():
    """A rank-deficient input must not raise LinAlgError.

    With near-identical channels the cross-spectral matrix is singular and the
    Wilson decomposition cannot converge; the result is NaN (with a convergence
    warning), but the regularized solve must not crash.
    """
    conn = Connectivity(fourier_coefficients=_near_singular_fourier(1e-10))
    with (
        pytest.warns(UserWarning, match="Cholesky failed"),
        pytest.warns(UserWarning, match="did not converge"),
    ):
        mvar_coeffs = conn._MVAR_Fourier_coefficients  # must not raise LinAlgError
    # Downstream directed measures must also not crash; for a rank-deficient
    # input that fails to converge they propagate NaN rather than raising (the
    # failure was already reported by the warnings above, so none repeat here).
    dtf = conn.directed_transfer_function()
    assert np.isnan(mvar_coeffs).all()
    assert np.isnan(dtf).all()


def test_regularized_solve_rhs_matches_batched_lhs():
    """RHS identity passed to xp.linalg.solve must match batched LHS shape.

    NumPy accepts an unbatched (M, M) identity against a batched LHS, but CuPy
    rejects the mismatch and crashes. We assert the contract on CPU so this
    class of bug is caught without GPU CI.
    """
    from spectral_connectivity import connectivity as conn_mod

    real_solve = conn_mod.xp.linalg.solve
    captured = []

    def recording_solve(a, b):
        captured.append((a.shape, b.shape))
        return real_solve(a, b)

    rng = np.random.default_rng(0)
    n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals = (2, 2, 2, 4, 3)
    fourier_coefficients = rng.standard_normal(
        (n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals)
    ) + 1j * rng.standard_normal(
        (n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals)
    )
    conn = Connectivity(fourier_coefficients=fourier_coefficients.astype(complex))

    with patch.object(conn_mod.xp.linalg, "solve", side_effect=recording_solve):
        # Touch both fixed code paths.
        _ = conn._transfer_function  # _estimate_transfer_function
        _ = conn._MVAR_Fourier_coefficients  # _MVAR_Fourier_coefficients

    assert captured, "expected xp.linalg.solve to be called"
    for a_shape, b_shape in captured:
        assert a_shape == b_shape, (
            f"solve received mismatched shapes a={a_shape}, b={b_shape}; "
            "RHS must be broadcast to LHS batched shape for CuPy compatibility"
        )


def test_connectivity_rejects_wrong_ndim():
    """Test that Connectivity rejects inputs with wrong number of dimensions."""
    # Test 1D array
    fourier_1d = np.ones(10, dtype=np.complex128)
    with pytest.raises(ValueError, match="must be 5-dimensional, got 1D"):
        Connectivity(fourier_coefficients=fourier_1d)

    # Test 2D array
    fourier_2d = np.ones((10, 5), dtype=np.complex128)
    with pytest.raises(ValueError, match="must be 5-dimensional, got 2D"):
        Connectivity(fourier_coefficients=fourier_2d)

    # Test 3D array
    fourier_3d = np.ones((10, 5, 2), dtype=np.complex128)
    with pytest.raises(ValueError, match="must be 5-dimensional, got 3D"):
        Connectivity(fourier_coefficients=fourier_3d)

    # Test 4D array
    fourier_4d = np.ones((10, 5, 2, 100), dtype=np.complex128)
    with pytest.raises(ValueError, match="must be 5-dimensional, got 4D"):
        Connectivity(fourier_coefficients=fourier_4d)

    # Test 6D array
    fourier_6d = np.ones((10, 5, 2, 100, 3, 4), dtype=np.complex128)
    with pytest.raises(ValueError, match="must be 5-dimensional, got 6D"):
        Connectivity(fourier_coefficients=fourier_6d)

    # Verify error message contains helpful information
    fourier_3d = np.ones((10, 5, 2), dtype=np.complex128)
    with pytest.raises(
        ValueError, match=r"Expected shape.*n_time_windows.*n_trials.*n_tapers"
    ):
        Connectivity(fourier_coefficients=fourier_3d)

    # Verify error message suggests using Multitaper
    fourier_2d = np.ones((10, 5), dtype=np.complex128)
    with pytest.raises(ValueError, match="use the Multitaper class"):
        Connectivity(fourier_coefficients=fourier_2d)


def test_connectivity_warns_on_nan():
    """Test that Connectivity warns when fourier_coefficients contains NaN or Inf."""
    # Test NaN values
    fourier_with_nan = np.ones((2, 2, 2, 100, 2), dtype=np.complex128)
    fourier_with_nan[0, 0, 0, 0, 0] = np.nan

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        Connectivity(fourier_coefficients=fourier_with_nan)
        assert len(w) == 1
        assert issubclass(w[0].category, UserWarning)
        assert "NaN or Inf values" in str(w[0].message)
        assert "Check your input data" in str(w[0].message)

    # Test Inf values
    fourier_with_inf = np.ones((2, 2, 2, 100, 2), dtype=np.complex128)
    fourier_with_inf[0, 0, 0, 0, 0] = np.inf

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        Connectivity(fourier_coefficients=fourier_with_inf)
        assert len(w) == 1
        assert issubclass(w[0].category, UserWarning)
        assert "NaN or Inf values" in str(w[0].message)

    # Test complex Inf values
    fourier_with_complex_inf = np.ones((2, 2, 2, 100, 2), dtype=np.complex128)
    fourier_with_complex_inf[0, 0, 0, 0, 0] = complex(np.inf, 1.0)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        Connectivity(fourier_coefficients=fourier_with_complex_inf)
        assert len(w) == 1
        assert "NaN or Inf values" in str(w[0].message)
        # Check for actionable suggestions
        assert "interpolating" in str(w[0].message) or "artifact removal" in str(w[0].message)

    # Test valid data (no warning)
    fourier_valid = np.ones((2, 2, 2, 100, 2), dtype=np.complex128)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        Connectivity(fourier_coefficients=fourier_valid)
        # Filter out any warnings that are not from Connectivity
        connectivity_warnings = [
            warning for warning in w if "fourier_coefficients" in str(warning.message)
        ]
        assert len(connectivity_warnings) == 0


def test_reduced_cross_spectral_matrix_matches_outer_product():
    """The reduced (batched-matmul) CSM matches the full outer-product mean.

    ``_expectation_cross_spectral_matrix`` contracts the averaged observation
    axes directly instead of materializing the per-observation outer product. It
    must agree, to floating-point tolerance, with the explicit
    ``self._expectation(self._cross_spectral_matrix)`` for every expectation
    type, and must propagate NaNs the same way.
    """
    n_time_windows, n_trials, n_tapers, n_frequencies, n_signals = 4, 6, 5, 32, 4
    rng = np.random.default_rng(7)
    shape = (n_time_windows, n_trials, n_tapers, n_frequencies, n_signals)
    fourier_coefficients = (
        rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    ).astype(np.complex128)

    for expectation_type in [
        "time",
        "trials",
        "tapers",
        "time_trials",
        "time_tapers",
        "trials_tapers",
        "time_trials_tapers",
    ]:
        conn = Connectivity(
            fourier_coefficients=fourier_coefficients,
            expectation_type=expectation_type,
        )
        reduced = conn._expectation_cross_spectral_matrix()
        reference = conn._expectation(conn._cross_spectral_matrix)
        assert reduced.shape == reference.shape, expectation_type
        np.testing.assert_allclose(
            reduced, reference, rtol=1e-10, atol=1e-12, err_msg=expectation_type
        )

    # NaN in one observation/signal must poison the same rows/columns as the
    # explicit outer-product mean would.
    nan_coefficients = fourier_coefficients.copy()
    nan_coefficients[0, 0, 0, 0, 1] = np.nan
    with pytest.warns(UserWarning, match="NaN or Inf"):
        conn = Connectivity(
            fourier_coefficients=nan_coefficients, expectation_type="trials_tapers"
        )
    reduced = conn._expectation_cross_spectral_matrix()
    reference = conn._expectation(conn._cross_spectral_matrix)
    np.testing.assert_array_equal(np.isnan(reduced), np.isnan(reference))


def test_weighted_expectation_matches_manual_cross_spectrum():
    rng = np.random.default_rng(922)
    coefficients = rng.standard_normal((2, 3, 4, 5, 2)) + 1j * rng.standard_normal(
        (2, 3, 4, 5, 2)
    )
    weights = rng.uniform(0.1, 1.0, size=(2, 3, 4, 5, 1))
    connectivity = Connectivity(coefficients, observation_weights=weights)

    outer = coefficients[..., :, np.newaxis] * np.conjugate(coefficients[..., np.newaxis, :])
    expected = (
        np.sum(outer * weights[..., np.newaxis], axis=(1, 2))
        / np.sum(weights, axis=(1, 2))[..., np.newaxis]
    )
    np.testing.assert_allclose(connectivity._expectation_cross_spectral_matrix(), expected)


@pytest.mark.parametrize(
    "weights",
    [
        np.ones((2, 3, 4, 5)),
        np.full((2, 3, 4, 5, 1), -1.0),
        np.full((2, 3, 4, 5, 1), np.nan),
    ],
)
def test_observation_weights_validation(weights):
    coefficients = np.ones((2, 3, 4, 5, 2), dtype=np.complex128)
    with pytest.raises(ValueError, match="observation_weights"):
        Connectivity(coefficients, observation_weights=weights)


def _reference_normalized_cross_spectrum(conn):
    """Honest per-observation phase-locking cross-spectrum (materialized).

    Normalizes each per-observation cross-spectrum entry ``z_i conj(z_j)`` by its
    magnitude and averages, then restricts to non-negative frequencies -- the
    original implementation the factorized ``_phase_locking_value`` replaced.
    """
    csm = np.asarray(conn._cross_spectral_matrix)
    magnitude = np.abs(csm)
    with np.errstate(invalid="ignore", divide="ignore"):
        normalized = csm / magnitude
    normalized[magnitude == 0] = np.nan
    reduced = np.asarray(conn._expectation(normalized))
    return reduced[..., : reduced.shape[-3] // 2 + 1, :, :]


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("dead", [False, True])
@pytest.mark.parametrize(
    "expectation_type",
    [
        "time",
        "trials",
        "tapers",
        "time_trials",
        "time_tapers",
        "trials_tapers",
        "time_trials_tapers",
    ],
)
def test_phase_locking_value_matches_per_observation_reference(dtype, dead, expectation_type):
    """Factorized PLV/PPC equal the materialized per-observation reference.

    ``phase_locking_value`` now unit-normalizes each Fourier coefficient and
    reuses the batched reduced cross-spectral matmul, using
    ``(z_i conj(z_j)) / |z_i conj(z_j)| = (z_i/|z_i|) conj(z_j/|z_j|)``. It must
    match the honest per-observation normalization-then-average across every
    expectation mode, both dtypes, and a dead (all-zero) channel that makes the
    normalization undefined (NaN) -- the case ``blocks`` used to serve.
    """
    rng = np.random.default_rng(0)
    shape = (3, 4, 5, 8, 4)
    fc = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(dtype)
    if dead:
        fc[:, 1, :, :, 0] = 0.0  # dead channel 0 on trial 1

    conn = Connectivity(fourier_coefficients=fc, expectation_type=expectation_type)
    # Both the factorized measure and the reference normalize at the computation
    # dtype (complex128 default), so even a complex64 input agrees to ~1e-15.
    tol = {"rtol": 1e-9, "atol": 1e-11}

    ref_complex = _reference_normalized_cross_spectrum(conn)
    ref_plv = np.abs(ref_complex)

    def expect_dead_channel_warning():
        # The dead channel makes the z / |z| normalization undefined.
        if dead:
            return pytest.warns(UserWarning, match="zero magnitude")
        return nullcontext()

    with expect_dead_channel_warning():
        plv = conn.phase_locking_value()
        raw = np.asarray(conn._phase_locking_value())

    # NaN placement identical, values equal off the NaNs.
    np.testing.assert_array_equal(np.isnan(plv), np.isnan(ref_plv))
    np.testing.assert_array_equal(np.isnan(raw), np.isnan(ref_complex))
    finite = ~np.isnan(ref_plv)
    np.testing.assert_allclose(plv[finite], ref_plv[finite], **tol)
    np.testing.assert_allclose(raw[finite], ref_complex[finite], **tol)
    # Diagonal (self-consistency) is 1 where defined.
    diag = np.diagonal(plv, axis1=-2, axis2=-1)
    np.testing.assert_allclose(diag[~np.isnan(diag)], 1.0, **tol)
    # Documented bounds hold exactly for PLV (public method clips to [0, 1]),
    # including at the input precision where the raw normalization can overshoot.
    assert np.nanmin(plv) >= 0.0
    assert np.nanmax(plv) <= 1.0

    # Pairwise phase consistency built from the same complex reference. Every
    # expectation here averages >= 3 observations; the n_observations < 2 guard
    # is covered by test_debiased_measures_require_multiple_observations.
    n = conn.n_observations
    assert n >= 2
    plv_sum = ref_complex * n
    ref_ppc = ((plv_sum * plv_sum.conjugate() - n) / (n**2 - n)).real
    with expect_dead_channel_warning():
        ppc = conn.pairwise_phase_consistency()
    np.testing.assert_array_equal(np.isnan(ppc), np.isnan(ref_ppc))
    fppc = ~np.isnan(ref_ppc)
    np.testing.assert_allclose(ppc[fppc], ref_ppc[fppc], **tol)
    # PPC is not clipped (it can be slightly negative for random phases),
    # but must not exceed 1 beyond floating-point rounding.
    assert np.nanmax(ppc) <= 1.0 + 1e-9


def test_default_coordinates_created_when_omitted():
    """The constructor must populate frequencies/time defaults, as documented.

    Directly constructing Connectivity without frequencies/time should yield
    normalized frequencies and integer time indices rather than None, so that
    coordinate-dependent methods (e.g. group_delay) do not crash.
    """
    n_time_windows, n_trials, n_tapers, n_fft_samples, n_signals = 3, 1, 2, 8, 2
    rng = np.random.default_rng(0)
    fourier_coefficients = rng.standard_normal(
        (n_time_windows, n_trials, n_tapers, n_fft_samples, n_signals)
    ) + 1j * rng.standard_normal(
        (n_time_windows, n_trials, n_tapers, n_fft_samples, n_signals)
    )

    conn = Connectivity(fourier_coefficients=fourier_coefficients)

    assert conn.frequencies is not None
    assert conn.time is not None
    # Normalized frequencies: non-negative half of fftfreq(n_fft_samples).
    np.testing.assert_allclose(
        conn.frequencies,
        np.abs(np.fft.fftfreq(n_fft_samples))[: n_fft_samples // 2 + 1],
    )
    np.testing.assert_array_equal(conn.time, np.arange(n_time_windows))
    # A coordinate-dependent method must not raise.
    conn.group_delay()


@pytest.mark.parametrize(
    "measure",
    [
        "directed_transfer_function",
        "directed_coherence",
        "partial_directed_coherence",
        "generalized_partial_directed_coherence",
        "direct_directed_transfer_function",
    ],
)
def test_directed_measures_no_runtime_warning_on_nan_transfer_function(measure):
    """Directed measures must propagate NaN silently, not emit divide warnings.

    Force Wilson non-convergence with ``minimum_phase_max_iterations=1`` so the
    transfer function is deterministically NaN; normalizing by the
    inflow/outflow sum would emit ``invalid value encountered in divide`` unless
    scoped. The NaN is already reported via the convergence UserWarning.
    """
    from spectral_connectivity.transforms import Multitaper

    rng = np.random.default_rng(0)
    time_series = rng.standard_normal((256, 4, 3))
    conn = Connectivity.from_multitaper(
        Multitaper(time_series, sampling_frequency=256, time_halfbandwidth_product=3),
        minimum_phase_max_iterations=1,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("error", RuntimeWarning)
        result = getattr(conn, measure)()
    # The forced non-convergence yields NaN; the point is it did so without a
    # RuntimeWarning (which would have been raised as an error above).
    assert np.isnan(result).any()


def test_power_one_sided_preserves_total_power():
    """The one-sided PSD must double interior bins to conserve total power.

    Summing the returned one-sided power (over frequency) must equal summing the
    full two-sided spectrum; otherwise integrating power() recovers only half
    the variance.
    """
    from spectral_connectivity.transforms import Multitaper

    rng = np.random.default_rng(0)
    for n_time in (512, 511):  # even and odd FFT lengths
        time_series = rng.standard_normal((n_time, 4, 2))
        conn = Connectivity.from_multitaper(Multitaper(time_series, sampling_frequency=256))
        one_sided = conn.power()  # non-negative frequencies, interior doubled
        two_sided = conn._power  # full spectrum
        np.testing.assert_allclose(one_sided.sum(axis=-2), two_sided.sum(axis=-2), rtol=1e-10)


def test_power_one_sided_integrates_to_signal_power():
    """Parseval oracle for the one-sided PSD convention.

    Integrating the returned one-sided density over the non-negative grid,
    ``sum(power) * df``, must recover the tapered signal's mean square,
    ``mean_k sum_t (u_k(t) x(t))**2`` for unit-energy tapers ``u_k``; for a
    constant signal that is exactly its squared value. ``detrend_type=None``
    keeps a DC offset so the undoubled DC bin -- and, for an even FFT length,
    the undoubled Nyquist bin -- carry power and are exercised.
    """
    from spectral_connectivity.transforms import Multitaper

    rng = np.random.default_rng(3)
    sampling_frequency = 200.0
    for n_time in (512, 511):  # even and odd FFT lengths
        time_series = rng.standard_normal((n_time, 3, 2)) + 2.5  # DC offset
        multitaper = Multitaper(
            time_series, sampling_frequency=sampling_frequency, detrend_type=None
        )
        conn = Connectivity.from_transform(multitaper)
        frequency_step = conn.frequencies[1] - conn.frequencies[0]
        recovered = conn.power()[0].sum(axis=0) * frequency_step  # (n_signals,)

        # Multitaper scales its tapers to energy fs; undo that for the identity.
        unit_tapers = np.asarray(multitaper.tapers) / np.sqrt(sampling_frequency)
        assert unit_tapers.shape[0] == n_time
        tapered = unit_tapers[:, :, np.newaxis, np.newaxis] * time_series[:, np.newaxis]
        expected = (tapered**2).sum(axis=0).mean(axis=(0, 1))  # over tapers, trials
        np.testing.assert_allclose(recovered, expected, rtol=1e-10)

        constant = Connectivity.from_transform(
            Multitaper(
                np.full((n_time, 2, 2), 3.0),
                sampling_frequency=sampling_frequency,
                detrend_type=None,
            )
        )
        np.testing.assert_allclose(
            constant.power()[0].sum(axis=0) * frequency_step, 9.0, rtol=1e-10
        )


def test_power_preserves_float32_dtype():
    """power() must not upcast a float32 (complex64) spectrum to float64.

    Regression: the one-sided doubling multiplied by a float64 scale array,
    silently widening a complex64 spectrum back to float64 and defeating the
    memory/precision choice. The scale now matches the spectrum dtype.
    """
    rng = np.random.default_rng(0)
    fourier_coefficients = (
        rng.standard_normal((2, 3, 2, 8, 2)) + 1j * rng.standard_normal((2, 3, 2, 8, 2))
    ).astype(np.complex64)
    conn = Connectivity(fourier_coefficients=fourier_coefficients)
    assert conn._power.dtype == np.float32  # spectrum is genuinely float32
    assert conn.power().dtype == np.float32  # and power() must not widen it


def test_phase_slope_index_uses_adjacent_frequency_bins():
    """PSI must sum conj(C(f)) * C(f + df) over adjacent bins (Nolte 2008).

    The previous implementation summed over all i<j frequency-pair combinations,
    which is a different statistic. This checks the public result equals the
    adjacent-bin reference and differs from the all-pairs sum.
    """
    from spectral_connectivity.transforms import Multitaper

    rng = np.random.default_rng(0)
    time_series = rng.standard_normal((256, 12, 2))
    conn = Connectivity.from_multitaper(
        Multitaper(time_series, sampling_frequency=200, time_halfbandwidth_product=3)
    )

    # Reproduce the internal subsampling to build a faithful reference.
    frequencies = conn.frequencies
    bandpassed, band_freqs = _bandpass(conn.coherency(), frequencies, None)
    step = _get_independent_frequency_step(frequencies[1] - frequencies[0], None)
    idx = np.arange(0, band_freqs.shape[0], step)
    bandpassed = bandpassed[..., idx, :, :]

    adjacent_ref = (
        (np.conj(bandpassed[..., :-1, :, :]) * bandpassed[..., 1:, :, :]).sum(axis=-3).imag
    )
    # All-pairs sum (the previous, incorrect statistic) for contrast.
    from itertools import combinations

    pair_index = np.array(list(combinations(range(bandpassed.shape[-3]), 2)))
    all_pairs_ref = (
        (
            np.conj(bandpassed[..., pair_index[:, 0], :, :])
            * bandpassed[..., pair_index[:, 1], :, :]
        )
        .sum(axis=-3)
        .imag
    )

    psi = conn.phase_slope_index()
    np.testing.assert_allclose(psi, adjacent_ref, atol=1e-12)
    # The two references genuinely differ, so this discriminates the fix.
    assert not np.allclose(adjacent_ref, all_pairs_ref)
    # PSI is antisymmetric in the signal pair.
    np.testing.assert_allclose(psi[..., 0, 1], -psi[..., 1, 0], atol=1e-12)


def _correlated_fixture():
    """Ordinary near-singular LFP-like fixture (highly correlated channels)."""
    from spectral_connectivity.transforms import Multitaper

    rng = np.random.default_rng(42)
    n_time, n_trials, n_signals, sf = 100, 5, 3, 500
    t = np.arange(n_time) / sf
    base = np.sin(2 * np.pi * 10 * t)
    sig = np.zeros((n_time, n_trials, n_signals))
    for k in range(n_trials):
        sig[:, k, 0] = base + 0.1 * rng.standard_normal(n_time)
        sig[:, k, 1] = np.sin(2 * np.pi * 10 * t + np.pi / 4) + 0.1 * rng.standard_normal(
            n_time
        )
        sig[:, k, 2] = 0.1 * base + 0.9 * rng.standard_normal(n_time)
    return Multitaper(sig, sampling_frequency=sf, time_halfbandwidth_product=2, n_tapers=3)


@pytest.mark.parametrize(
    "measure",
    [
        "directed_transfer_function",
        "partial_directed_coherence",
        "pairwise_spectral_granger_prediction",
    ],
)
def test_directed_measures_finite_on_ordinary_correlated_data(measure):
    """Directed measures must return finite values for realistic correlated data.

    Regression: the Wilson relative-tolerance change with the old
    max_iterations=60 returned an all-NaN minimum-phase factor for near-singular
    cross-spectral matrices (highly correlated channels), which the range tests
    accepted vacuously. This is a non-vacuous finiteness check.
    """
    conn = Connectivity.from_multitaper(_correlated_fixture())
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)  # no non-convergence warning
        result = getattr(conn, measure)()
    assert np.isfinite(result).mean() > 0.5


def test_minimum_phase_max_iterations_is_configurable():
    """Users can raise max_iterations to recover from Wilson non-convergence."""
    m = _correlated_fixture()
    conn_low = Connectivity.from_multitaper(m, minimum_phase_max_iterations=1)
    with pytest.warns(UserWarning, match="did not converge"):
        dtf_low = conn_low.directed_transfer_function()
    assert np.isnan(dtf_low).all()

    conn_high = Connectivity.from_multitaper(m, minimum_phase_max_iterations=500)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        dtf_high = conn_high.directed_transfer_function()
    assert np.isfinite(dtf_high).mean() > 0.5


@pytest.mark.parametrize(
    "measure",
    ["directed_transfer_function", "pairwise_spectral_granger_prediction"],
)
def test_directed_measures_are_scale_invariant(measure):
    """Rescaling the signal must not change these scale-invariant measures.

    Regression: the Tikhonov diagonal loading used lambda proportional to
    mean(|H|**2) (amplitude-squared units) added to H (amplitude units), so the
    regularization strength was not scale-covariant. Rescaling the signal by a
    large factor then changed the transfer-function inverse and shifted DTF /
    spectral Granger by orders of magnitude. lambda now scales with the RMS
    magnitude, restoring invariance.
    """
    from spectral_connectivity.simulate import simulate_MVAR
    from spectral_connectivity.transforms import Multitaper

    coeffs = np.array([[[0.5, 0.3], [0.0, 0.4]], [[-0.2, 0.0], [0.1, -0.3]]])
    ts = simulate_MVAR(
        coeffs,
        noise_covariance=np.eye(2),
        n_time_samples=600,
        n_trials=8,
        random_state=np.random.default_rng(0),
    )

    def compute(scale):
        m = Multitaper(
            ts * scale,
            sampling_frequency=1.0,
            time_halfbandwidth_product=3,
            start_time=0,
        )
        return getattr(Connectivity.from_multitaper(m), measure)()

    base = compute(1.0)
    scaled = compute(1e12)
    finite = np.isfinite(base) & np.isfinite(scaled)
    assert finite.any()
    np.testing.assert_allclose(scaled[finite], base[finite], rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize(
    "measure",
    [
        "phase_lag_index",
        "weighted_phase_lag_index",
        "directed_phase_lag_index",
        "debiased_squared_phase_lag_index",
        "debiased_squared_weighted_phase_lag_index",
    ],
)
@pytest.mark.parametrize("scale", [1e-9, 1e9])
def test_phase_lag_family_is_scale_invariant(measure, scale):
    """The phase-lag measures are ratios of imaginary cross-spectrum moments,
    so rescaling the signal (e.g. volts vs microvolts) must not change them.

    Regression: weighted_phase_lag_index guarded its denominator with an
    absolute ``finfo(float).eps`` threshold on a quantity in signal**2 units,
    so inputs of order 1e-9 saw every weight replaced by 1 and wPLI collapsed
    toward 0.
    """
    rng = np.random.default_rng(16)
    shape = (1, 3, 4, 8, 3)
    coefficients = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    mixing = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
    coefficients = coefficients @ mixing.T  # correlated, phase-lagged channels
    reference = getattr(Connectivity(coefficients), measure)()
    scaled = getattr(Connectivity(coefficients * scale), measure)()
    np.testing.assert_allclose(scaled, reference, rtol=1e-9, atol=1e-12)


def test_debiased_squared_phase_lag_index_is_zero_for_exactly_zero_imaginary_part():
    """Vinck's closed form ``(n * PLI**2 - 1) / (n - 1)`` assumes every sign is
    +/-1. When Im S_xy is exactly 0 for every observation (in-phase real
    signals, and the diagonal) the sign is 0, and the estimate must be 0 rather
    than the spurious lower bound ``-1 / (n - 1)``."""
    rng = np.random.default_rng(17)
    shape = (1, 3, 4, 8, 3)
    real_coefficients = rng.standard_normal(shape).astype(complex)
    connectivity = Connectivity(real_coefficients)
    np.testing.assert_array_equal(connectivity.debiased_squared_phase_lag_index(), 0.0)


@pytest.mark.parametrize("scale", [1e-9, 1.0, 3.0, -0.7, 1e9])
def test_phase_lag_family_is_zero_for_in_phase_signals(scale):
    """A channel paired with a scaled copy of itself (zero-lag coupling, e.g.
    volume conduction) has no phase lag. Its per-observation imaginary
    cross-spectrum is rounding noise, not exactly 0, so an exact-zero guard
    reports wPLI as a ratio of rounding errors (|wPLI| up to 0.6).

    Regression: the zero-lag test compared E[|Im S_xy|] with exactly 0 instead
    of with the signals' own power.
    """
    from spectral_connectivity import Multitaper

    source = np.random.default_rng(18).standard_normal((1000, 20))
    time_series = np.stack([source, scale * source], axis=-1)
    connectivity = Connectivity.from_transform(
        Multitaper(time_series, sampling_frequency=500, time_halfbandwidth_product=3)
    )
    np.testing.assert_array_equal(connectivity.weighted_phase_lag_index()[..., 0, 1], 0.0)
    np.testing.assert_array_equal(
        connectivity.debiased_squared_phase_lag_index()[..., 0, 1], 0.0
    )


@pytest.fixture(scope="module")
def weakly_coupled_connectivity():
    """Two weakly coupled signals, 10 trials: most bins have a small magnitude,
    where an unclamped Fisher interval's lower bound falls below 0."""
    from spectral_connectivity import Multitaper

    rng = np.random.default_rng(19)
    source = rng.standard_normal((1000, 10))
    time_series = np.stack([source, 0.3 * source + rng.standard_normal((1000, 10))], -1)
    return Connectivity.from_transform(
        Multitaper(time_series, sampling_frequency=500, time_halfbandwidth_product=1)
    )


@pytest.mark.parametrize("method", ["phase_locking_value", "imaginary_coherence"])
def test_jackknife_of_a_nonnegative_magnitude_stays_in_its_range(
    weakly_coupled_connectivity, method
):
    """PLV and imaginary coherence are magnitudes in [0, 1]. Their Fisher
    interval is formed on atanh, whose back-transform tanh reaches below 0, so
    it must be clamped like fisher_squared's.

    Regression: 94% of imaginary-coherence lower bounds and 31% of its
    bias-corrected estimates were negative (down to -0.7) on this data.
    """
    result = weakly_coupled_connectivity.jackknife(method)
    lower, upper = result.confidence_interval
    assert result.transformation == "fisher"
    assert np.nanmin(lower) >= 0.0
    assert np.nanmin(result.bias_corrected) >= 0.0
    assert np.nanmax(upper) <= 1.0


def test_jackknife_fisher_stays_signed_for_a_signed_measure(weakly_coupled_connectivity):
    """An explicit Fisher interval on a signed measure in [-1, 1] must keep its
    negative bounds; only the non-negative magnitudes are clamped at 0."""
    result = weakly_coupled_connectivity.jackknife(
        "imaginary_coherency", transformation="fisher"
    )
    lower, _ = result.confidence_interval
    assert np.nanmin(lower[..., 0, 1]) < 0.0


def test_global_coherence_sparse_branch_orders_strongest_first():
    """global_coherence must order components strongest-first regardless of the
    order svds returns (which SciPy does not guarantee)."""
    from spectral_connectivity import connectivity as conn_mod

    real_svds = conn_mod.svds

    def ascending_svds(matrix, k):
        u, s, vh = real_svds(matrix, k)
        order = np.argsort(s)  # force ascending
        return u[:, order], s[order], vh[order]

    def descending_svds(matrix, k):
        u, s, vh = real_svds(matrix, k)
        order = np.argsort(s)[::-1]  # force descending
        return u[:, order], s[order], vh[order]

    rng = np.random.default_rng(0)
    fc = rng.standard_normal((1, 8, 1, 4, 6)) + 1j * rng.standard_normal((1, 8, 1, 4, 6))
    # Force the per-bin svds fallback (the moderate-n_signals default is the
    # batched eigendecomposition, which never calls svds) so the mock takes
    # effect and this exercises the svds ordering logic it is written for.
    with patch.object(conn_mod, "GLOBAL_COHERENCE_MAX_DENSE_COMPONENTS", 1):
        with patch.object(conn_mod, "svds", ascending_svds):
            gc_asc, _ = Connectivity(fourier_coefficients=fc).global_coherence(max_rank=3)
        with patch.object(conn_mod, "svds", descending_svds):
            gc_desc, _ = Connectivity(fourier_coefficients=fc).global_coherence(max_rank=3)
    # Same result regardless of the order svds returned, and strongest-first.
    np.testing.assert_allclose(gc_asc, gc_desc)
    assert np.all(gc_asc[..., 0] >= gc_asc[..., 1] - 1e-9)
    assert np.all(gc_asc[..., 1] >= gc_asc[..., 2] - 1e-9)


def test_global_coherence_batched_matches_per_bin_fallback():
    """The batched path matches the per-bin svds/svd path, thin and wide.

    global_coherence batches over bins with an ``eigh`` of the cross-spectral
    matrix (``n_estimates >= n_signals``) or the economy SVD of the thin matrix
    (``n_estimates < n_signals``), falling back to a per-bin loop for large
    square matrices. All must agree on the coherence fractions (the vectors need
    not agree: they are defined only up to a per-component phase, or an arbitrary
    unitary rotation within a degenerate subspace), including NaN placement for
    zero-power bins.
    """
    from spectral_connectivity import connectivity as conn_mod

    rng = np.random.default_rng(4)
    # wide: n_estimates (30) >= n_signals (8) -> eigh path
    # thin: n_estimates (2) <  n_signals (8) -> economy SVD path
    for shape in [(2, 10, 3, 20, 8), (2, 1, 2, 20, 8)]:
        fc = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(
            np.complex128
        )
        fc = fc.copy()
        fc[0, :, :, 5, :] = 0.0  # a zero-power bin -> NaN on both paths
        max_available = min(shape[4], shape[1] * shape[2])

        for max_rank in (1, min(3, max_available)):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gc_batched, _ = Connectivity(fc).global_coherence(max_rank=max_rank)
                # Force the per-bin fallback by lowering the batching threshold.
                with patch.object(conn_mod, "GLOBAL_COHERENCE_MAX_DENSE_COMPONENTS", 0):
                    gc_loop, _ = Connectivity(fc).global_coherence(max_rank=max_rank)
            np.testing.assert_array_equal(np.isnan(gc_batched), np.isnan(gc_loop))
            np.testing.assert_allclose(
                gc_batched, gc_loop, rtol=1e-9, atol=1e-11, equal_nan=True
            )
            assert np.all(gc_batched[~np.isnan(gc_batched)] >= 0)
            assert np.all(gc_batched[~np.isnan(gc_batched)] <= 1)


def test_global_coherence_batched_matches_per_bin_ill_conditioned():
    """Batched eigh and per-bin SVD agree even for near-duplicate channels.

    The batched path diagonalizes ``A @ Aᴴ`` with ``eigh``, which squares the
    condition number relative to the per-bin ``svd(A)``. For a nearly
    rank-deficient cross-spectral matrix (near-duplicate channels) the weakest
    components can lose relative precision, but the coherence fractions must
    still agree to a tight absolute tolerance, and the dominant component -- the
    usual use of this measure -- must agree closely. This guards the documented
    eigh/SVD tradeoff against a regression that widens the gap; the existing
    equivalence test uses only well-conditioned Gaussian data.
    """
    from spectral_connectivity import connectivity as conn_mod

    rng = np.random.default_rng(20240827)
    n_time, n_trials, n_tapers, n_fft, n_signals = 2, 30, 2, 10, 4
    # Near-duplicate channels: a shared complex component broadcast across all
    # signals plus a tiny (1e-6) per-channel perturbation. The resulting per-bin
    # cross-spectral matrix is nearly rank-one (condition number ~1e6+), the
    # regime where eigh(A @ Aᴴ) and svd(A) diverge most.
    shared = rng.standard_normal(
        (n_time, n_trials, n_tapers, n_fft, 1)
    ) + 1j * rng.standard_normal((n_time, n_trials, n_tapers, n_fft, 1))
    perturbation = 1e-6 * (
        rng.standard_normal((n_time, n_trials, n_tapers, n_fft, n_signals))
        + 1j * rng.standard_normal((n_time, n_trials, n_tapers, n_fft, n_signals))
    )
    fc = (shared + perturbation).astype(np.complex128)

    for max_rank in (1, n_signals):
        gc_batched, _ = Connectivity(fc).global_coherence(max_rank=max_rank)
        # Force the per-bin svd/svds fallback (the well-conditioned reference).
        with patch.object(conn_mod, "GLOBAL_COHERENCE_MAX_DENSE_COMPONENTS", 0):
            gc_loop, _ = Connectivity(fc).global_coherence(max_rank=max_rank)

        np.testing.assert_array_equal(np.isnan(gc_batched), np.isnan(gc_loop))
        # Weak components may differ in relative terms but are ~machine-epsilon
        # in absolute terms, so compare all components on an absolute tolerance.
        np.testing.assert_allclose(gc_batched, gc_loop, atol=1e-8, equal_nan=True)
        # The dominant component holds essentially all the coherent power for
        # near-duplicate channels and must agree closely in relative terms.
        np.testing.assert_allclose(gc_batched[..., 0], gc_loop[..., 0], rtol=1e-6)
        assert np.all(gc_batched[..., 0] > 0.99)


def test_global_coherence_batched_chunking_matches_single_chunk():
    """The multi-chunk path matches processing all bins in one chunk.

    The default element cap keeps every test's bins in a single chunk, so the
    partial-last-chunk reshape/strided-write logic is otherwise unexercised.
    Force a tiny budget so bins are processed in several chunks (with a zero-power
    bin near a boundary) and require an identical result.
    """
    rng = np.random.default_rng(7)
    shape = (2, 10, 3, 20, 6)
    fc = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex128)
    fc = fc.copy()
    fc[0, :, :, 7, :] = 0.0  # zero-power bin (-> NaN), placed to straddle chunks

    with pytest.warns(UserWarning, match="zero total power"):
        gc_single, vec_single = Connectivity(fc).global_coherence(max_rank=2)
    # Budget chosen so `chunk` is only a few bins, forcing multiple iterations and
    # a partial final chunk (20 frequency bins do not divide evenly).
    with pytest.warns(UserWarning, match="zero total power"):
        gc_multi, vec_multi = Connectivity(fc).global_coherence(
            max_rank=2, max_workspace_elements=6 * 6 * 3
        )

    np.testing.assert_array_equal(np.isnan(gc_single), np.isnan(gc_multi))
    np.testing.assert_allclose(gc_single, gc_multi, equal_nan=True)
    np.testing.assert_allclose(vec_single, vec_multi, equal_nan=True)


def test_global_coherence_workspace_budget_is_configurable_and_result_invariant():
    """max_workspace_elements bounds peak memory without changing the result.

    Chunking is a memory-only concern: a tiny budget (many small chunks) and a
    huge budget (a single chunk) must yield the same fractions and vectors, and a
    non-positive budget must be rejected.
    """
    rng = np.random.default_rng(31)
    shape = (2, 10, 3, 20, 6)
    fc = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex128)
    c = Connectivity(fc)

    gc_default, vec_default = c.global_coherence(max_rank=2)
    gc_tiny, vec_tiny = c.global_coherence(max_rank=2, max_workspace_elements=5_000)
    gc_huge, vec_huge = c.global_coherence(max_rank=2, max_workspace_elements=10**9)

    np.testing.assert_array_equal(gc_default, gc_tiny)
    np.testing.assert_array_equal(gc_default, gc_huge)
    np.testing.assert_array_equal(vec_default, vec_tiny)
    np.testing.assert_array_equal(vec_default, vec_huge)

    # Must be a genuine positive integer: reject non-positive, fractional,
    # non-finite, and boolean budgets (a float would corrupt the chunk size and
    # crash later inside range(); bool is an int subclass).
    for bad in [0, -1, 2.5, np.nan, np.inf, True]:
        with pytest.raises(ValueError, match="max_workspace_elements must be a positive"):
            c.global_coherence(max_workspace_elements=bad)


def test_global_coherence_vectors_are_orthonormal_eigenvectors():
    """The returned vectors are unit-norm eigenvectors, even ill-conditioned.

    The coherence vectors are a documented output but are only checked for shape
    elsewhere. Verify (including near-duplicate, ill-conditioned channels) that
    the returned vectors are orthonormal and that every component is an
    eigenvector of the per-bin cross-spectral matrix with the eigenvalue implied
    by its global coherence.
    """
    rng = np.random.default_rng(8)
    n_time, n_trials, n_tapers, n_fft, n_signals = 1, 12, 2, 8, 4
    base = rng.standard_normal((n_time, n_trials, n_tapers, n_fft)) + 1j * rng.standard_normal(
        (n_time, n_trials, n_tapers, n_fft)
    )
    fc = np.zeros((n_time, n_trials, n_tapers, n_fft, n_signals), dtype=complex)
    fc[..., 0] = base
    fc[..., 1] = base * (1 + 1e-10)  # near-duplicate -> ill-conditioned
    fc[..., 2] = 0.5 * base + 0.01 * (
        rng.standard_normal(base.shape) + 1j * rng.standard_normal(base.shape)
    )
    fc[..., 3] = rng.standard_normal(base.shape) + 1j * rng.standard_normal(base.shape)

    global_coherence, vectors = Connectivity(fc).global_coherence(max_rank=2)

    # Per bin the vectors are orthonormal: V^H V = I.
    gram = np.conj(vectors).swapaxes(-1, -2) @ vectors
    np.testing.assert_allclose(gram, np.broadcast_to(np.eye(2), gram.shape), atol=1e-8)

    # Every component is an eigenvector of the per-bin cross-spectral matrix
    # A A^H with eigenvalue global_coherence * ||A||_F**2 (the fraction of total
    # power), computed here directly from the coefficients.
    observations = fc.transpose(0, 3, 4, 1, 2).reshape(
        n_time, n_fft, n_signals, n_trials * n_tapers
    )
    cross_spectral = observations @ np.conj(observations).swapaxes(-1, -2)
    total_power = np.sum(np.abs(observations) ** 2, axis=(-2, -1))
    eigenvalues = global_coherence * total_power[..., np.newaxis]
    residual = cross_spectral @ vectors - vectors * eigenvalues[..., np.newaxis, :]
    np.testing.assert_allclose(
        np.linalg.norm(residual, axis=-2) / total_power[..., np.newaxis], 0.0, atol=1e-8
    )


def test_phase_slope_index_raises_with_fewer_than_two_bins():
    """Fewer than 2 frequency bins in the band must raise, not return a false 0."""
    from spectral_connectivity.transforms import Multitaper

    rng = np.random.default_rng(0)
    conn = Connectivity.from_multitaper(
        Multitaper(
            rng.standard_normal((256, 6, 2)),
            sampling_frequency=200,
            time_halfbandwidth_product=3,
        )
    )
    # A huge frequency_resolution subsamples the band to a single bin.
    with pytest.raises(ValueError, match="at least 2 frequency bins"):
        conn.phase_slope_index(frequency_resolution=1e6)


@pytest.mark.parametrize("bad_resolution", [0.0, -1.0, np.nan, np.inf])
def test_frequency_resolution_must_be_finite_positive(bad_resolution):
    """delay/phase_slope_index reject an invalid frequency_resolution."""
    from spectral_connectivity.transforms import Multitaper

    conn = Connectivity.from_multitaper(
        Multitaper(
            np.random.default_rng(0).standard_normal((256, 6, 2)),
            sampling_frequency=200,
            time_halfbandwidth_product=3,
        )
    )
    with pytest.raises(ValueError, match="frequency_resolution must be a finite"):
        conn.phase_slope_index(frequency_resolution=bad_resolution)


@pytest.mark.parametrize("measure", ["delay", "group_delay", "phase_slope_index"])
def test_single_frequency_bin_raises_clear_error(measure):
    """One frequency bin must raise a clear ValueError, not a raw IndexError."""
    # n_fft_samples = 1 -> a single non-negative frequency bin.
    fc = np.ones((2, 3, 2, 1, 2), dtype=complex)
    conn = Connectivity(fourier_coefficients=fc)
    assert conn.frequencies.size == 1
    with pytest.raises(ValueError, match="at least 2 frequency bins"):
        getattr(conn, measure)()


@pytest.mark.parametrize(
    "measure",
    [
        "phase_locking_value",
        "pairwise_phase_consistency",
        "corrected_imaginary_phase_locking_value",
    ],
)
def test_phase_locking_zero_power_is_nan_without_runtime_warning(measure):
    """A dead (zero) channel yields NaN with a UserWarning, not a RuntimeWarning."""
    fc = np.ones((1, 5, 2, 4, 2), dtype=complex)
    fc[..., 1] = 0.0  # dead second channel
    conn = Connectivity(fourier_coefficients=fc)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)  # no leaked divide warning
        with pytest.warns(UserWarning, match="zero magnitude"):
            result = getattr(conn, measure)()
    # Every pair involving the dead channel is undefined, never a finite value.
    assert np.isnan(result[..., 1, :]).all()
    assert np.isnan(result[..., :, 1]).all()
    assert np.isfinite(result[..., 0, 0]).all()


def _leave_one_out_standard_error(statistic, observations):
    """Jackknife standard error of ``statistic`` over the observation axis.

    Recomputes ``statistic`` on ``observations`` with each observation (axis 0)
    deleted in turn and returns ``sqrt((n - 1) / n * sum((theta_k -
    mean(theta)) ** 2))`` (Efron & Stein 1981).
    """
    n_observations = observations.shape[0]
    replicates = np.stack(
        [statistic(np.delete(observations, k, axis=0)) for k in range(n_observations)]
    )
    return np.sqrt(
        (n_observations - 1)
        / n_observations
        * np.sum((replicates - replicates.mean(axis=0)) ** 2, axis=0)
    )


def _magnitude_squared_coherence(observations):
    """|S_01|**2 / (S_00 S_11) from observations of shape (n_obs, n_fft, 2),
    restricted to the non-negative frequencies."""
    n_nonnegative = observations.shape[1] // 2 + 1
    cross = np.mean(observations[..., 0] * np.conj(observations[..., 1]), axis=0)
    power = np.mean(np.abs(observations) ** 2, axis=0)
    return (np.abs(cross) ** 2 / (power[:, 0] * power[:, 1]))[:n_nonnegative]


def _fisher_squared_coherence(observations):
    return np.arctanh(np.sqrt(_magnitude_squared_coherence(observations)))


def _trial_taper_observations(coefficients):
    """(1, n_trials, n_tapers, n_fft, n_signals) -> (n_trials * n_tapers, n_fft,
    n_signals): each trial-taper eigencoefficient is one observation."""
    return coefficients[0].reshape(-1, *coefficients.shape[-2:])


def test_connectivity_jackknife_recomputes_leave_one_out_measure():
    """The standard error equals a hand-rolled leave-one-trial-taper-out
    jackknife of magnitude-squared coherence on the atanh(sqrt(.)) scale,
    mapped back with the delta method."""
    rng = np.random.default_rng(510)
    coefficients = rng.standard_normal((1, 4, 3, 16, 2)) + 1j * rng.standard_normal(
        (1, 4, 3, 16, 2)
    )
    connectivity = Connectivity(coefficients)

    result = connectivity.jackknife("coherence_magnitude")

    assert result.n_observations == 12
    # coherence_magnitude is magnitude-*squared* coherence, so auto resolves to
    # the atanh(sqrt(.)) variance-stabilizing transform, not plain atanh.
    assert result.transformation == "fisher_squared"
    assert result.estimate.shape == (1, 9, 2, 2)
    assert result.standard_error.shape == result.estimate.shape

    observations = _trial_taper_observations(coefficients)
    coherence = _magnitude_squared_coherence(observations)
    transformed = np.arctanh(np.sqrt(coherence))
    transformed_standard_error = _leave_one_out_standard_error(
        _fisher_squared_coherence, observations
    )
    standard_error = 2 * np.sqrt(coherence) * (1 - coherence) * transformed_standard_error
    # Student t on n_observations - 1 degrees of freedom, not the normal quantile.
    critical_value = scipy.stats.t.ppf(0.975, df=result.n_observations - 1)

    np.testing.assert_allclose(result.estimate[0, :, 0, 1], coherence, rtol=1e-10)
    assert np.all(result.standard_error[0, :, 0, 1] > 0)
    np.testing.assert_allclose(result.standard_error[0, :, 0, 1], standard_error, rtol=1e-8)
    np.testing.assert_allclose(result.standard_error[0, :, 1, 0], standard_error, rtol=1e-8)
    np.testing.assert_allclose(
        result.confidence_interval[0][0, :, 0, 1],
        np.clip(np.tanh(transformed - critical_value * transformed_standard_error), 0, 1) ** 2,
        rtol=1e-8,
    )
    np.testing.assert_allclose(
        result.confidence_interval[1][0, :, 0, 1],
        np.tanh(transformed + critical_value * transformed_standard_error) ** 2,
        rtol=1e-8,
    )


def test_connectivity_jackknife_documents_and_accepts_fisher_squared():
    """The public annotation and docstring must expose the supported transform."""
    from typing import get_args

    rng = np.random.default_rng(514)
    coefficients = rng.standard_normal((1, 4, 2, 16, 2)) + 1j * rng.standard_normal(
        (1, 4, 2, 16, 2)
    )

    annotation = Connectivity.jackknife.__annotations__["transformation"]
    assert "fisher_squared" in get_args(annotation)
    assert "fisher_squared" in (Connectivity.jackknife.__doc__ or "")
    result = Connectivity(coefficients).jackknife(
        "coherence_magnitude", transformation="fisher_squared"
    )
    assert result.transformation == "fisher_squared"

    observations = _trial_taper_observations(coefficients)
    coherence = _magnitude_squared_coherence(observations)
    expected_standard_error = (
        2
        * np.sqrt(coherence)
        * (1 - coherence)
        * _leave_one_out_standard_error(_fisher_squared_coherence, observations)
    )
    np.testing.assert_allclose(
        result.standard_error[0, :, 0, 1], expected_standard_error, rtol=1e-8
    )


def test_connectivity_jackknife_auto_uses_log_power_and_rejects_complex_result():
    rng = np.random.default_rng(511)
    coefficients = rng.standard_normal((1, 3, 2, 12, 2)) + 1j * rng.standard_normal(
        (1, 3, 2, 12, 2)
    )
    connectivity = Connectivity(coefficients)

    power = connectivity.jackknife("power")
    assert power.transformation == "log"
    assert np.all(power.confidence_interval[0] > 0)

    # On the log scale the (constant) one-sided power scaling cancels, so the
    # relative standard error is the jackknife SE of log(mean |x|**2).
    def log_power(observations):
        return np.log(np.mean(np.abs(observations) ** 2, axis=0))[:7]

    expected_relative_error = _leave_one_out_standard_error(
        log_power, _trial_taper_observations(coefficients)
    )
    assert np.all(expected_relative_error > 0)
    np.testing.assert_allclose(
        power.standard_error[0] / power.estimate[0], expected_relative_error, rtol=1e-8
    )
    with pytest.raises(TypeError, match="real-valued measure"):
        connectivity.jackknife("coherency")


# Measures whose result is fixed by the normalization rather than the data from
# a single observation, with the set of values they are forced to. Partial
# coherence inverts a rank-one cross-spectrum: forced to 1 only for two signals
# (see test_partial_coherence_warns_when_signals_outnumber_observations), and
# null-space garbage for the three used here, so no saturation is asserted.
SINGLE_OBSERVATION_DEGENERATE_MEASURES = {
    "coherence_magnitude": [1.0],
    "phase_locking_value": [1.0],
    "partial_coherence": None,
    "phase_lag_index": [-1.0, 1.0],
    "weighted_phase_lag_index": [-1.0, 1.0],
    "directed_phase_lag_index": [0.0, 1.0],
}


@pytest.mark.parametrize("measure", sorted(SINGLE_OBSERVATION_DEGENERATE_MEASURES))
def test_single_observation_normalized_measure_warns(measure):
    # A single observation (1 trial x 1 taper) forces every normalized value to
    # an extreme (apparent perfect connectivity or a perfectly consistent lag);
    # this must warn rather than silently return a misleading result.
    rng = np.random.default_rng(707)
    coefficients = rng.standard_normal((1, 1, 1, 16, 3)) + 1j * rng.standard_normal(
        (1, 1, 1, 16, 3)
    )
    connectivity = Connectivity(coefficients)
    # coherence_magnitude reports through the shared coherency, hence no name match.
    with pytest.warns(UserWarning, match="is computed from a single observation"):
        result = getattr(connectivity, measure)()
    # And the degenerate result is indeed saturated at its extreme value(s).
    forced_values = SINGLE_OBSERVATION_DEGENERATE_MEASURES[measure]
    if forced_values is not None:
        off_diagonal = result[..., 0, 1]
        distance_to_extreme = np.min(
            np.abs(off_diagonal[..., np.newaxis] - np.asarray(forced_values)), axis=-1
        )
        np.testing.assert_allclose(distance_to_extreme, 0.0, atol=1e-6)


@pytest.mark.parametrize("measure", sorted(SINGLE_OBSERVATION_DEGENERATE_MEASURES))
def test_multiple_observations_normalized_measure_does_not_warn(measure):
    rng = np.random.default_rng(708)
    coefficients = rng.standard_normal((1, 4, 3, 16, 3)) + 1j * rng.standard_normal(
        (1, 4, 3, 16, 3)
    )
    connectivity = Connectivity(coefficients)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        getattr(connectivity, measure)()


def test_canonical_coherence_warns_when_groups_outnumber_observations():
    """With fewer trial x taper observations than the two groups' signals, the
    groups' observation subspaces must intersect, forcing canonical coherence
    to 1 for any data; that must be announced, not returned silently."""
    rng = np.random.default_rng(12)
    shape = (1, 3, 3, 8, 10)  # 9 observations for 5 + 5 signals
    coefficients = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    labels = np.array([0] * 5 + [1] * 5)
    with pytest.warns(UserWarning, match="9 observations"):
        values, _ = Connectivity(coefficients).canonical_coherence(labels)
    # Independent noise, yet the value is forced to 1: the reason for the warning.
    np.testing.assert_allclose(values[..., 0, 1], 1.0)


def test_canonical_coherence_does_not_warn_with_enough_observations():
    rng = np.random.default_rng(13)
    shape = (1, 4, 3, 8, 10)  # 12 observations for 5 + 5 signals
    coefficients = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    values, _ = Connectivity(coefficients).canonical_coherence(np.array([0] * 5 + [1] * 5))
    assert np.all(values[..., 0, 1] < 1.0 - 1e-6)


def test_partial_coherence_warns_when_signals_outnumber_observations():
    """With fewer trial x taper observations than signals the cross-spectral
    matrix is rank-deficient; its regularized inverse is then dominated by the
    null space, and with exactly one null direction every partial coherence is
    forced to 1 for any data. That must be announced, not returned silently."""
    rng = np.random.default_rng(14)
    shape = (1, 1, 5, 8, 6)  # 5 observations for 6 white-noise channels
    coefficients = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    with pytest.warns(UserWarning, match="partial_coherence uses 5 observations"):
        values = Connectivity(coefficients).partial_coherence()
    # Independent noise, yet every pair is forced to 1: the reason for the warning.
    off_diagonal = values[..., ~np.eye(6, dtype=bool)]
    np.testing.assert_allclose(off_diagonal, 1.0)


def test_partial_coherence_does_not_warn_with_enough_observations():
    rng = np.random.default_rng(15)
    shape = (1, 2, 3, 8, 6)  # 6 observations for 6 channels: full rank
    coefficients = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        values = Connectivity(coefficients).partial_coherence()
    assert np.all(values[..., 0, 1] < 1.0 - 1e-6)
