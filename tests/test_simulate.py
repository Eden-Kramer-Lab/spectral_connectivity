import numpy as np
import pytest

from spectral_connectivity.simulate import simulate_MVAR


def test_simulate_MVAR_deterministic_with_seed():
    """Test that seeded simulations produce identical results."""
    coefficients = np.array([
        [[0.5, 0.1],
         [0.2, 0.3]]
    ])
    noise_covariance = np.eye(2)

    # Run simulation twice with same seed
    result1 = simulate_MVAR(
        coefficients=coefficients,
        noise_covariance=noise_covariance,
        n_time_samples=50,
        n_trials=2,
        random_state=42
    )

    result2 = simulate_MVAR(
        coefficients=coefficients,
        noise_covariance=noise_covariance,
        n_time_samples=50,
        n_trials=2,
        random_state=42
    )

    # Should be identical
    np.testing.assert_array_equal(result1, result2)


def test_simulate_MVAR_different_seeds():
    """Test that different seeds produce different results."""
    coefficients = np.array([
        [[0.5, 0.1],
         [0.2, 0.3]]
    ])

    result1 = simulate_MVAR(
        coefficients=coefficients,
        n_time_samples=50,
        random_state=42
    )

    result2 = simulate_MVAR(
        coefficients=coefficients,
        n_time_samples=50,
        random_state=123
    )

    # Should be different
    assert not np.allclose(result1, result2)


def test_simulate_MVAR_generator_instance():
    """Test using numpy Generator instance."""
    coefficients = np.array([
        [[0.4, 0.0],
         [0.0, 0.4]]
    ])

    rng = np.random.default_rng(42)
    result = simulate_MVAR(
        coefficients=coefficients,
        n_time_samples=10,
        random_state=rng
    )

    # Should run without error and produce expected shape
    assert result.shape == (10, 1, 2)