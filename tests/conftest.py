"""Pytest configuration and fixtures for spectral_connectivity tests.

The test suite uses ``np.random.default_rng()`` throughout: each test creates
its own independent RNG instance, so tests do not share global random state.

Historically this module also reset the global NumPy random state before each
test, because the production Wilson minimum-phase fallback
(``minimum_phase_decomposition._get_initial_conditions``) drew from the global
generator via ``xp.random.standard_normal``. That fallback is now deterministic
(a fixed positive-definite start), so no production code touches the global
random state and no global reset is needed for reproducibility.

For the modern NumPy random API, see:
https://numpy.org/doc/stable/reference/random/generator.html
"""
