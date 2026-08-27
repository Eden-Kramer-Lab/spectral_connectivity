import numpy as np

from spectral_connectivity.simulate import simulate_MVAR


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

    result1 = simulate_MVAR(
        coefficients=coefficients, n_time_samples=50, random_state=42
    )

    result2 = simulate_MVAR(
        coefficients=coefficients, n_time_samples=50, random_state=123
    )

    # Should be different
    assert not np.allclose(result1, result2)


def test_simulate_MVAR_generator_instance():
    """Test using numpy Generator instance."""
    coefficients = np.array([[[0.4, 0.0], [0.0, 0.4]]])

    rng = np.random.default_rng(42)
    result = simulate_MVAR(
        coefficients=coefficients, n_time_samples=10, random_state=rng
    )

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
    reference = rng.multivariate_normal(
        np.zeros(2), noise_covariance, size=(n_time, n_trials)
    )
    for t in range(1, n_time):
        for trial in range(n_trials):
            reference[t, trial] += coefficients[0] @ reference[t - 1, trial]
    np.testing.assert_allclose(library, reference)
