"""Pytest configuration and fixtures for spectral_connectivity tests.

Newer tests create their own ``np.random.default_rng(seed)`` instances, but many
existing tests still draw from the global NumPy generator (``np.random.random``,
``np.random.randn``, ...). To keep those reproducible regardless of test
execution order, the autouse fixture below reseeds the global generator before
each test.

Note this is now purely a *test* concern: the production code no longer touches
the global random state (the Wilson minimum-phase fallback in
``minimum_phase_decomposition`` became a deterministic positive-definite start).
The fixture can be removed once the remaining tests are migrated to local
generators.

For the modern NumPy random API, see:
https://numpy.org/doc/stable/reference/random/generator.html
"""

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def reset_global_random_state():
    """Reseed the global NumPy generator before each test for reproducibility.

    Tests that use their own ``np.random.default_rng`` are unaffected; tests that
    still draw from the global generator become order-independent.
    """
    np.random.seed(42)
    yield
    np.random.seed(None)
