import numpy as np
import pytest

from spectral_connectivity.simulate import (
    simulate_lagged_broadband,
    simulate_MVAR,
    simulate_shared_oscillation,
)


def test_simulate_MVAR_deterministic_with_seed():
    """Test that seeded simulations produce identical results."""
    coefficients = np.array([[[0.5, 0.1], [0.2, 0.3]]])
    noise_covariance = np.eye(2)

    # Run simulation twice with same seed
    result1 = simulate_MVAR(
        coefficients=coefficients,
        noise_covariance=noise_covariance,
        n_time_samples=50,
        n_trials=2,
        random_state=42,
    )

    result2 = simulate_MVAR(
        coefficients=coefficients,
        noise_covariance=noise_covariance,
        n_time_samples=50,
        n_trials=2,
        random_state=42,
    )

    # Should be identical
    np.testing.assert_array_equal(result1, result2)


def test_simulate_MVAR_different_seeds():
    """Test that different seeds produce different results."""
    coefficients = np.array([[[0.5, 0.1], [0.2, 0.3]]])

    result1 = simulate_MVAR(coefficients=coefficients, n_time_samples=50, random_state=42)

    result2 = simulate_MVAR(coefficients=coefficients, n_time_samples=50, random_state=123)

    # Should be different
    assert not np.allclose(result1, result2)


def test_simulate_MVAR_generator_instance():
    """Test using numpy Generator instance."""
    coefficients = np.array([[[0.4, 0.0], [0.0, 0.4]]])

    rng = np.random.default_rng(42)
    result = simulate_MVAR(coefficients=coefficients, n_time_samples=10, random_state=rng)

    # Should run without error and produce expected shape
    assert result.shape == (10, 1, 2)


def test_simulate_MVAR_univariate_multi_trial():
    """A single-signal, multi-trial simulation must not crash (regression)."""
    coefficients = np.array([[[0.5]]])  # n_lags=1, n_signals=1
    result = simulate_MVAR(
        coefficients=coefficients, n_time_samples=20, n_trials=3, random_state=0
    )
    assert result.shape == (20, 3, 1)
    assert np.all(np.isfinite(result))


def test_simulate_MVAR_recursion_matches_explicit_per_trial():
    """The vectorized ``X_prev @ A_k.T`` must equal the explicit ``A_k @ x``.

    Pins the multivariate recursion against a hand-written per-trial reference
    with asymmetric coefficients, so a transpose error in the vectorized form
    (which is invisible to the determinism-only tests) would be caught.
    """
    coefficients = np.array([[[0.5, 0.1], [0.2, 0.3]]])  # VAR(1), asymmetric
    noise_covariance = np.eye(2)
    n_time, n_trials = 20, 4
    library = simulate_MVAR(
        coefficients,
        noise_covariance=noise_covariance,
        n_time_samples=n_time,
        n_trials=n_trials,
        n_burnin_samples=0,
        random_state=0,
    )
    # Same noise draw, explicit per-trial A @ x_prev recursion.
    rng = np.random.default_rng(0)
    reference = rng.multivariate_normal(np.zeros(2), noise_covariance, size=(n_time, n_trials))
    for t in range(1, n_time):
        for trial in range(n_trials):
            reference[t, trial] += coefficients[0] @ reference[t - 1, trial]
    np.testing.assert_allclose(library, reference)


@pytest.mark.parametrize("lags", [(-3, 0), (0, 1.5)], ids=["negative", "fractional"])
def test_lagged_broadband_rejects_negative_or_fractional_lags(lags):
    """Lags must be non-negative integers; a lead is a smaller lag, not a negative one."""
    with pytest.raises(ValueError, match="non-negative integers"):
        simulate_lagged_broadband(lags, noise_levels=0.1, n_time_samples=50, random_state=0)


@pytest.mark.parametrize(
    ("n_trials", "expected_shape"),
    [(None, (50, 3)), (4, (50, 4, 3))],
    ids=["no_trials", "trials"],
)
def test_lagged_broadband_shape(n_trials, expected_shape):
    """The trial axis is present only when ``n_trials`` is given."""
    time_series = simulate_lagged_broadband(
        (0, 2, 5), noise_levels=0.1, n_time_samples=50, n_trials=n_trials, random_state=0
    )
    assert time_series.shape == expected_shape
    assert time_series.dtype == np.float64


def test_lagged_broadband_cross_correlation_peaks_at_lag():
    """Signal 1 repeats signal 0 ``lags[1] - lags[0]`` samples later."""
    lags = (2, 9)
    time_series = simulate_lagged_broadband(
        lags, noise_levels=0.1, n_time_samples=5000, random_state=0
    )
    signal0, signal1 = time_series[:, 0], time_series[:, 1]
    # np.correlate(a, v, "full")[i] = sum_n a[n + k] * v[n] with k = i - (len(v) - 1).
    cross_correlation = np.correlate(signal1, signal0, mode="full")
    shift = np.argmax(cross_correlation) - (signal0.size - 1)
    assert shift == lags[1] - lags[0]


@pytest.mark.parametrize("leader", [0, 1])
@pytest.mark.parametrize("n_trials", [None, 3])
def test_lagged_broadband_reproduces_pair_helper_semantics(leader, n_trials):
    """``lags=(0, L)`` slices one seeded source and adds one noise draw.

    Replays by hand the leader/follower construction (source first, then
    ``rng.normal(0, noise_sd, shape)``); the simulator must reproduce it bit for
    bit, so that results built on the earlier construction do not change.
    """
    n_time_samples, lag, noise_sd, seed = 200, 15, 0.25, 42
    lags = (0, lag) if leader == 0 else (lag, 0)

    rng = np.random.default_rng(seed)
    extra = () if n_trials is None else (n_trials,)
    source = rng.standard_normal((n_time_samples + lag, *extra))
    ahead, behind = source[lag:], source[:n_time_samples]
    expected = np.stack([ahead, behind] if leader == 0 else [behind, ahead], axis=-1)
    expected = expected + rng.normal(0, noise_sd, expected.shape)

    time_series = simulate_lagged_broadband(
        lags, noise_sd, n_time_samples, n_trials, random_state=np.random.default_rng(seed)
    )
    np.testing.assert_array_equal(time_series, expected)


def test_lagged_broadband_noise_free_is_shifted_source():
    """With ``noise_levels=0`` every signal is the source delayed by its lag."""
    lags, n_time_samples, n_trials = np.array([0, 3, 8]), 100, 4
    time_series = simulate_lagged_broadband(
        lags, noise_levels=0, n_time_samples=n_time_samples, n_trials=n_trials, random_state=7
    )
    source = np.random.default_rng(7).standard_normal((n_time_samples + lags.max(), n_trials))
    # source[max_lag + t] is "now", so signal k at time t is source[max_lag + t - lags[k]].
    for k, lag in enumerate(lags):
        np.testing.assert_array_equal(
            time_series[..., k], source[lags.max() - lag : lags.max() - lag + n_time_samples]
        )
    # Signal 0 leads: signal 2 at time t + 8 equals signal 0 at time t.
    np.testing.assert_array_equal(time_series[8:, :, 2], time_series[:-8, :, 0])


def test_lagged_broadband_seed_determinism():
    """The same seed reproduces the output; a different seed changes it."""
    kwargs = {"lags": (0, 4), "noise_levels": [0.2, 0.5], "n_time_samples": 64, "n_trials": 2}
    first = simulate_lagged_broadband(**kwargs, random_state=3)
    np.testing.assert_array_equal(first, simulate_lagged_broadband(**kwargs, random_state=3))
    assert not np.allclose(first, simulate_lagged_broadband(**kwargs, random_state=4))


_SAMPLING_FREQUENCY = 1000
_N_TIME_SAMPLES = 1000  # 1 s, so ``frequency`` below falls exactly on an FFT bin
_FREQUENCY = 40


def _fourier_at_frequency(time_series):
    """FFT coefficient at ``_FREQUENCY``, shape (n_trials, n_signals)."""
    return np.fft.rfft(time_series, axis=0)[
        round(_FREQUENCY * _N_TIME_SAMPLES / _SAMPLING_FREQUENCY)
    ]


def test_shared_oscillation_shape():
    """Shape is (n_time_samples, n_trials, n_signals) with n_signals = len(amplitudes)."""
    time_series = simulate_shared_oscillation(
        _FREQUENCY, _SAMPLING_FREQUENCY, 300, 5, amplitudes=[1.0, 0.5, 2.0], random_state=0
    )
    assert time_series.shape == (300, 5, 3)
    assert time_series.dtype == np.float64


def test_shared_oscillation_frequency_dominates_spectrum():
    """The planted frequency is the largest bin of every signal's spectrum."""
    time_series = simulate_shared_oscillation(
        _FREQUENCY,
        _SAMPLING_FREQUENCY,
        _N_TIME_SAMPLES,
        10,
        amplitudes=[1.0, 2.0, 0.5],
        noise_levels=1.0,
        random_state=0,
    )
    power = np.abs(np.fft.rfft(time_series, axis=0)) ** 2  # (n_freqs, n_trials, n_signals)
    frequencies = np.fft.rfftfreq(_N_TIME_SAMPLES, d=1 / _SAMPLING_FREQUENCY)
    np.testing.assert_array_equal(frequencies[np.argmax(power, axis=0)], _FREQUENCY)


def test_shared_oscillation_phase_offsets():
    """Cross-spectrum phase at the planted bin is the difference of the offsets."""
    phase_offsets = np.array([0.0, np.pi / 3, -np.pi / 4])
    time_series = simulate_shared_oscillation(
        _FREQUENCY,
        _SAMPLING_FREQUENCY,
        _N_TIME_SAMPLES,
        2,
        amplitudes=[1.0, 2.0, 0.5],
        phase_offsets=phase_offsets,
        noise_levels=0,
        random_phase_per_trial=False,
    )
    fourier = _fourier_at_frequency(time_series)  # (n_trials, n_signals)
    cross_spectrum = fourier[:, :, np.newaxis] * np.conj(fourier[:, np.newaxis, :])
    expected = phase_offsets[:, np.newaxis] - phase_offsets[np.newaxis, :]
    # Compare on the circle, so a difference near +-pi does not wrap spuriously.
    phase_error = np.angle(cross_spectrum * np.exp(-1j * expected))
    np.testing.assert_allclose(phase_error, 0, atol=1e-6)


@pytest.mark.parametrize("random_phase_per_trial", [True, False])
def test_shared_oscillation_random_trial_phase(random_phase_per_trial):
    """Trials start at different phases only when ``random_phase_per_trial`` is set.

    Either way, all signals share each trial's phase: the between-signal phase
    difference is the same on every trial.
    """
    time_series = simulate_shared_oscillation(
        _FREQUENCY,
        _SAMPLING_FREQUENCY,
        _N_TIME_SAMPLES,
        20,
        amplitudes=[1.0, 1.0],
        phase_offsets=[0.0, np.pi / 2],
        random_phase_per_trial=random_phase_per_trial,
        random_state=0,
    )
    fourier = _fourier_at_frequency(time_series)  # (n_trials, n_signals)
    # Each trial's phase relative to trial 0, compared on the circle.
    trial_phase_change = np.angle(fourier[:, 0] * np.conj(fourier[0, 0]))
    if random_phase_per_trial:
        assert np.all(np.abs(trial_phase_change[1:]) > 1e-3)
    else:
        np.testing.assert_allclose(trial_phase_change, 0, atol=1e-6)
    np.testing.assert_allclose(
        np.angle(fourier[:, 0] * np.conj(fourier[:, 1])), -np.pi / 2, atol=1e-6
    )


def test_shared_oscillation_zero_amplitude_is_flat():
    """An amplitude of 0 (without noise) leaves that signal identically zero."""
    time_series = simulate_shared_oscillation(
        _FREQUENCY, _SAMPLING_FREQUENCY, 200, 3, amplitudes=[1.0, 0.0], random_state=0
    )
    assert np.all(time_series[..., 1] == 0)
    assert np.any(time_series[..., 0] != 0)


def test_shared_oscillation_seed_determinism():
    """The same seed reproduces the output; a different seed changes it."""
    kwargs = {
        "frequency": _FREQUENCY,
        "sampling_frequency": _SAMPLING_FREQUENCY,
        "n_time_samples": 100,
        "n_trials": 3,
        "amplitudes": [1.0, 0.5],
        "noise_levels": 0.3,
    }
    first = simulate_shared_oscillation(**kwargs, random_state=5)
    np.testing.assert_array_equal(first, simulate_shared_oscillation(**kwargs, random_state=5))
    assert not np.allclose(first, simulate_shared_oscillation(**kwargs, random_state=6))


def test_shared_oscillation_parameters_apply_per_signal():
    """``amplitudes[k]`` and ``noise_levels[k]`` set signal k's amplitude and noise."""
    amplitudes = np.array([0.5, 1.0, 3.0])
    noise_levels = np.array([0.2, 1.0, 0.5])
    kwargs = {
        "frequency": _FREQUENCY,
        "sampling_frequency": _SAMPLING_FREQUENCY,
        "n_time_samples": _N_TIME_SAMPLES,
        "n_trials": 20,
        "amplitudes": amplitudes,
        "random_state": 0,
    }
    noise_free = simulate_shared_oscillation(**kwargs)
    noisy = simulate_shared_oscillation(**kwargs, noise_levels=noise_levels)
    # Same seed: same trial phases, so the difference is exactly the added noise.
    residual_sd = (noisy - noise_free).std(axis=(0, 1))
    np.testing.assert_allclose(residual_sd, noise_levels, rtol=0.02)
    # 25 samples per 40 Hz cycle put a sample within 7.2 degrees of each crest.
    peak_amplitude = np.abs(noise_free).max(axis=(0, 1))
    np.testing.assert_allclose(peak_amplitude, amplitudes, rtol=0.01)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"amplitudes": 1.0}, "amplitudes must be 1-D"),
        ({"amplitudes": [1.0, 1.0], "noise_levels": [0.1, 0.2, 0.3]}, "noise_levels"),
        ({"amplitudes": [1.0, 1.0], "phase_offsets": [0.0, 1.0, 2.0]}, "phase_offsets"),
    ],
    ids=["scalar_amplitudes", "noise_levels_length", "phase_offsets_length"],
)
def test_shared_oscillation_rejects_mismatched_parameters(kwargs, match):
    """Per-signal parameters must be scalars or have one entry per signal."""
    with pytest.raises(ValueError, match=match):
        simulate_shared_oscillation(_FREQUENCY, _SAMPLING_FREQUENCY, 100, 2, **kwargs)


def test_lagged_broadband_rejects_mismatched_noise_levels():
    """``noise_levels`` must be a scalar or have one entry per lag."""
    with pytest.raises(ValueError, match="noise_levels"):
        simulate_lagged_broadband((0, 3), [0.1, 0.2, 0.3], n_time_samples=50)
