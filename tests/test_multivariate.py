"""Tests for the multivariate coupling kernels in ``spectral_connectivity._multivariate``."""

import numpy as np
import pytest

from spectral_connectivity import Connectivity, Multitaper
from spectral_connectivity._multivariate import _optimize_canonical_coherency_phase, _reshape
from spectral_connectivity.simulate import simulate_shared_oscillation


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

    magnitude, phase, left, right = _optimize_canonical_coherency_phase(
        whitened[np.newaxis], n_grid=n_grid
    )

    dense_best = max(
        _cacoh_phase_objective(whitened, phase) for phase in np.linspace(0, np.pi, 20001)
    )
    assert magnitude[0] >= max(coarse_scores) - 1e-12
    assert magnitude[0] == pytest.approx(dense_best, abs=1e-6)
    assert magnitude[0] == pytest.approx(0.80, abs=1e-6)
    # The phase is defined modulo pi; it must be lobe 1, not lobe 2.
    assert np.exp(2j * phase[0]) == pytest.approx(np.exp(2j * theta_1), abs=1e-6)
    np.testing.assert_allclose(np.abs(left[0, :, 0]), directions[0], atol=1e-6)
    np.testing.assert_allclose(np.abs(right[0, :, 0]), directions[0], atol=1e-6)


def test_cacoh_phase_optimizer_refines_every_coarse_grid_lobe():
    """With four near-equal lobes the true maximum, mid-way between two coarse
    grid points, is only the fourth-highest coarse-grid local maximum, so
    refining a fixed number of the highest ones misses it. Every local maximum
    must be refined, independently in each bin of a batched leading shape."""
    n_grid = 37
    grid = np.arange(n_grid) * np.pi / n_grid
    half_cell = np.pi / (2 * n_grid)
    lobe_grid_indices = [3, 11, 19, 27]
    # On the grid the true lobe (0.80, mid-cell) peaks at 0.80 * cos(half_cell);
    # the other three sit exactly on grid points, just above that.
    lower_height = 0.80 * np.cos(half_cell) + 1e-5
    whitened = np.zeros((2, 2, 4, 4), dtype=complex)
    true_phase = np.zeros((2, 2))
    for bin_index, true_lobe in zip(np.ndindex(2, 2), range(4), strict=True):
        heights = np.full(4, lower_height)
        heights[true_lobe] = 0.80
        phases = grid[lobe_grid_indices]
        phases[true_lobe] += half_cell
        whitened[bin_index] = np.diag(heights * np.exp(1j * phases))
        true_phase[bin_index] = phases[true_lobe]
        # Premise: the true lobe is the lowest of the four on the coarse grid.
        coarse = np.array([_cacoh_phase_objective(whitened[bin_index], p) for p in grid])
        lobe_peaks = [max(coarse[k], coarse[k + 1]) for k in lobe_grid_indices]
        assert np.argmin(lobe_peaks) == true_lobe

    magnitude, phase, _, _ = _optimize_canonical_coherency_phase(whitened, n_grid=n_grid)

    np.testing.assert_allclose(magnitude, 0.80, rtol=0, atol=1e-9)
    # The phase is defined modulo pi; it must be the true lobe's.
    np.testing.assert_allclose(np.exp(2j * phase), np.exp(2j * true_phase), atol=1e-6)


def test_cacoh_phase_optimizer_never_returns_below_the_coarse_grid():
    rng = np.random.default_rng(21)
    whitened = rng.standard_normal((40, 3, 4)) + 1j * rng.standard_normal((40, 3, 4))
    n_grid = 37
    grid = np.arange(n_grid) * np.pi / n_grid
    coarse_best = np.array(
        [max(_cacoh_phase_objective(matrix, phase) for phase in grid) for matrix in whitened]
    )
    magnitude, _, _, _ = _optimize_canonical_coherency_phase(whitened, n_grid=n_grid)
    assert np.all(magnitude >= coarse_best - 1e-12)


def test__reshape():
    n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals = (20, 100, 3, 10, 2)
    fourier_coefficients = np.zeros(
        (n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals), dtype=complex
    )
    expected_shape = (n_time_samples, n_fft_samples, n_signals, n_trials * n_tapers)
    assert np.allclose(_reshape(fourier_coefficients).shape, expected_shape)


def test_global_coherence_weighted_per_bin_path_matches_batched(monkeypatch):
    """Observation weights give the same components on the per-bin path, which
    weights one bin at a time, as on the batched path."""
    from spectral_connectivity import Connectivity, _multivariate

    rng = np.random.default_rng(13)
    shape = (2, 5, 3, 8, 4)
    coefficients = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    weights = rng.uniform(0.2, 1.0, (*shape[:-1], 1))

    batched, _ = Connectivity(coefficients, observation_weights=weights).global_coherence(
        max_rank=2
    )
    monkeypatch.setattr(_multivariate, "GLOBAL_COHERENCE_MAX_DENSE_COMPONENTS", 0)
    per_bin, _ = Connectivity(coefficients, observation_weights=weights).global_coherence(
        max_rank=2
    )
    unweighted, _ = Connectivity(coefficients).global_coherence(max_rank=2)

    np.testing.assert_allclose(per_bin, batched, rtol=1e-10)
    assert not np.allclose(per_bin, unweighted)  # the weights took effect


def test_global_coherence_per_bin_path_is_reproducible(monkeypatch):
    """The per-bin path's svds branch gives identical results run to run."""
    from spectral_connectivity import Connectivity, _multivariate

    monkeypatch.setattr(_multivariate, "GLOBAL_COHERENCE_MAX_DENSE_COMPONENTS", 0)
    rng = np.random.default_rng(14)
    shape = (1, 6, 3, 8, 8)  # max_rank=2 < min(n_signals, n_estimates) - 1: svds
    coefficients = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)

    first = Connectivity(coefficients).global_coherence(max_rank=2)
    second = Connectivity(coefficients).global_coherence(max_rank=2)

    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])


_GROUP_LABELS = [0, 0, 0, 1, 1, 1]
_SHARED_FREQUENCY = 20.0


def _two_group_oscillation(group_1_phase):
    """Six signals sharing a 20 Hz rhythm, group 1 offset by ``group_1_phase``.

    Signals 0-2 (group 0) and 3-5 (group 1) see one sinusoid with unequal
    amplitudes plus independent noise; 100 trials x 5 tapers (NW 3) give 500
    observations, enough to hold MIC's positive finite-sample bias near 0.08.
    Returns the ``Connectivity`` and the indices of the 20 Hz bin and of the
    bins more than 10 Hz from it.
    """
    time_series = simulate_shared_oscillation(
        _SHARED_FREQUENCY,
        500,
        1000,
        n_trials=100,
        amplitudes=[1, 0.8, 1.2, 1, 1.1, 0.9],
        phase_offsets=[0, 0, 0, group_1_phase, group_1_phase, group_1_phase],
        noise_levels=0.5,
        random_state=1,
    )
    connectivity = Connectivity.from_transform(
        Multitaper(time_series, sampling_frequency=500, time_halfbandwidth_product=3)
    )
    frequencies = connectivity.frequencies
    peak = np.flatnonzero(frequencies == _SHARED_FREQUENCY).item()
    off_peak = np.abs(frequencies - _SHARED_FREQUENCY) > 10
    return connectivity, peak, off_peak


def test_mic_recovers_lagged_between_group_source():
    """A pi / 2 lag between the groups is an imaginary-axis interaction: MIC
    reaches it at 20 Hz and stays at its bias level elsewhere, and MIM (the sum
    of squared singular values) is at least MIC squared (the largest)."""
    connectivity, peak, off_peak = _two_group_oscillation(np.pi / 2)
    mic, labels = connectivity.maximized_imaginary_coherency(_GROUP_LABELS)
    mim, _ = connectivity.multivariate_interaction_measure(_GROUP_LABELS)
    np.testing.assert_array_equal(labels, [0, 1])
    mic, mim = mic[0, :, 0, 1], mim[0, :, 0, 1]

    assert mic[peak] > 0.9
    assert np.median(mic[off_peak]) < 0.15
    assert np.all(mim >= mic**2 - 1e-9)


def test_mic_rejects_zero_lag_shared_source():
    """A zero-lag shared source (volume conduction) has no imaginary part: MIC
    at 20 Hz stays at its own off-peak bias level, while canonical coherence,
    which is not blind to zero lag, finds the shared rhythm."""
    connectivity, peak, off_peak = _two_group_oscillation(0.0)
    mic, _ = connectivity.maximized_imaginary_coherency(_GROUP_LABELS)
    canonical, _ = connectivity.canonical_coherence(_GROUP_LABELS)
    mic = mic[0, :, 0, 1]

    assert mic[peak] < 0.15
    assert mic[peak] <= 2 * np.median(mic[off_peak])
    assert canonical[0, peak, 0, 1] > 0.9
