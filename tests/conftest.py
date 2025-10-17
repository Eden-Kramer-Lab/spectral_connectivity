"""Pytest configuration and fixtures for spectral_connectivity tests."""

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def reset_random_state():
    """Reset numpy random state before each test to ensure reproducibility.

    This fixture runs automatically before each test to prevent test ordering
    issues caused by shared random state. Each test will have a consistent
    starting random state.
    """
    # Reset to a fixed seed for reproducibility
    np.random.seed(42)
    yield
    # Cleanup after test (optional, but good practice)
    np.random.seed(None)  # Reset to unpredictable state
