"""Lightweight tests to verify connectivity metrics are within expected ranges."""

import numpy as np
import pytest

from spectral_connectivity import Connectivity
from spectral_connectivity.transforms import Multitaper


def _in_range(x, min_val, max_val, tolerance=1e-12):
    """Check if values are within expected range with small numerical tolerance."""
    x = np.asarray(x)
    # Remove NaN values for range checking
    x_clean = x[~np.isnan(x)]
    if len(x_clean) == 0:
        return True  # All NaN values are acceptable
    return np.all((x_clean >= min_val - tolerance) & (x_clean <= max_val + tolerance))


@pytest.fixture
def simple_synthetic_data():
    """Create simple synthetic data for fast testing."""
    rng = np.random.default_rng(42)
    n_time_samples, n_trials, n_signals = 100, 5, 3
    sampling_frequency = 500

    # Create correlated signals with some noise
    t = np.arange(n_time_samples) / sampling_frequency

    # Base signal: 10 Hz oscillation
    base_signal = np.sin(2 * np.pi * 10 * t)

    # Create signals with different phase relationships
    signals = np.zeros((n_time_samples, n_trials, n_signals))
    for trial in range(n_trials):
        # Signal 0: base + noise
        signals[:, trial, 0] = base_signal + 0.1 * rng.standard_normal((n_time_samples))

        # Signal 1: base with phase lag + noise
        signals[:, trial, 1] = np.sin(
            2 * np.pi * 10 * t + np.pi / 4
        ) + 0.1 * rng.standard_normal((n_time_samples))

        # Signal 2: mostly independent + noise
        signals[:, trial, 2] = 0.1 * base_signal + 0.9 * rng.standard_normal((n_time_samples))

    return signals, sampling_frequency


@pytest.fixture
def connectivity_obj(simple_synthetic_data):
    """Create Connectivity object from synthetic data."""
    time_series, sampling_frequency = simple_synthetic_data

    # Use minimal multitaper parameters for speed
    mt = Multitaper(
        time_series,
        sampling_frequency=sampling_frequency,
        time_halfbandwidth_product=2,
        n_tapers=3,
    )

    return Connectivity.from_multitaper(mt)


def test_coherence_magnitude_range(connectivity_obj):
    """Test that coherence magnitude is in [0, 1]."""
    coherence = connectivity_obj.coherence_magnitude()
    assert _in_range(coherence, 0, 1), "Coherence magnitude should be in [0, 1]"


def test_coherency_range(connectivity_obj):
    """Test that coherency magnitude is in [0, 1] and phase in [-π, π]."""
    coherency = connectivity_obj.coherency()

    # Check magnitude
    magnitude = np.abs(coherency)
    assert _in_range(magnitude, 0, 1), "Coherency magnitude should be in [0, 1]"

    # Check phase
    phase = np.angle(coherency)
    assert _in_range(phase, -np.pi, np.pi), "Coherency phase should be in [-π, π]"


def test_imaginary_coherence_range(connectivity_obj):
    """Test that imaginary coherence is in [0, 1] (magnitude version)."""
    imag_coh = connectivity_obj.imaginary_coherence()
    assert _in_range(imag_coh, 0, 1), "Imaginary coherence should be in [0, 1]"


def test_phase_locking_value_range(connectivity_obj):
    """Test that PLV is in [0, 1]."""
    plv = connectivity_obj.phase_locking_value()
    assert _in_range(plv, 0, 1), "PLV should be in [0, 1]"


def test_phase_lag_index_range(connectivity_obj):
    """Test that signed PLI is in [-1, 1]."""
    pli = connectivity_obj.phase_lag_index()
    assert _in_range(pli, -1, 1), "Phase lag index should be in [-1, 1]"


def test_granger_causality_range(connectivity_obj):
    """Test that Granger causality is non-negative."""
    gc = connectivity_obj.pairwise_spectral_granger_prediction()

    # Remove NaN values (diagonal elements)
    gc_clean = gc[~np.isnan(gc)]

    assert np.all(gc_clean >= -1e-12), "Granger causality should be non-negative"


def test_directed_transfer_function_range(connectivity_obj):
    """Test that DTF is in [0, 1]."""
    dtf = connectivity_obj.directed_transfer_function()

    # Remove NaN values if any
    dtf_clean = dtf[~np.isnan(dtf)]

    assert _in_range(dtf_clean, 0, 1), "DTF should be in [0, 1]"


def test_partial_directed_coherence_range(connectivity_obj):
    """Test that PDC is in [0, 1]."""
    pdc = connectivity_obj.partial_directed_coherence()

    # Remove NaN values if any
    pdc_clean = pdc[~np.isnan(pdc)]

    assert _in_range(pdc_clean, 0, 1), "PDC should be in [0, 1]"


def test_pairwise_phase_consistency_range(connectivity_obj):
    """Test that PPC is bounded (can be negative due to bias correction)."""
    ppc = connectivity_obj.pairwise_phase_consistency()

    # PPC can be negative due to bias correction, but should be reasonable
    # Upper bound should be 1, but allow some flexibility for bias correction
    ppc_clean = ppc[~np.isnan(ppc)]
    assert np.all(ppc_clean <= 1.01), "PPC should not exceed 1 by much"
    assert np.all(ppc_clean >= -0.5), "PPC should not be extremely negative"


def test_coherence_phase_range(connectivity_obj):
    """Test that coherence phase is in [-π, π]."""
    phase = connectivity_obj.coherence_phase()
    assert _in_range(phase, -np.pi, np.pi), "Coherence phase should be in [-π, π]"


if __name__ == "__main__":
    pytest.main([__file__])
