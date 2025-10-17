"""Pytest configuration and fixtures for spectral_connectivity tests.

This module has been modernized to use np.random.default_rng() throughout the
test suite, which provides better test isolation than the legacy np.random.seed()
approach. Each test now creates its own independent RNG instance.

However, we still reset the global NumPy random state before each test because:
1. The production code (minimum_phase_decomposition.py) uses xp.random.standard_normal()
   which pulls from global state when using the numpy backend
2. This ensures reproducible test results across different test execution orders

For more information on the modern numpy random API, see:
https://numpy.org/doc/stable/reference/random/generator.html
"""

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def reset_global_random_state():
    """Reset global numpy random state before each test for reproducibility.

    Even though tests use isolated RNG instances (np.random.default_rng(seed)),
    the production code may still use global random state, so we reset it here.
    """
    np.random.seed(42)
    yield
    # Cleanup optional
    np.random.seed(None)
