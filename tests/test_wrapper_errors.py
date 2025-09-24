import numpy as np
import pytest

from spectral_connectivity.wrapper import connectivity_to_xarray
from spectral_connectivity.transforms import Multitaper


def test_unsupported_method_error_message():
    """Test that unsupported methods provide actionable error messages."""
    # Create test data
    n_time_samples, n_trials, n_signals = 100, 5, 2
    time_series = np.random.random((n_time_samples, n_trials, n_signals))

    m = Multitaper(
        time_series=time_series,
        sampling_frequency=1000,
        time_window_duration=0.1,
    )

    # Test directed method
    with pytest.raises(ValueError) as exc_info:
        connectivity_to_xarray(m, method="directed_coherence")

    error_msg = str(exc_info.value)
    assert "is not supported by the xarray interface" in error_msg
    assert "Connectivity class directly" in error_msg
    assert "from spectral_connectivity import Connectivity" in error_msg
    assert "conn = Connectivity.from_multitaper(m)" in error_msg
    assert "result = conn.directed_coherence()" in error_msg

    # Test canonical_coherence method
    with pytest.raises(ValueError) as exc_info:
        connectivity_to_xarray(m, method="canonical_coherence")

    error_msg = str(exc_info.value)
    assert "canonical_coherence" in error_msg
    assert "result = conn.canonical_coherence()" in error_msg

    # Test group_delay method
    with pytest.raises(ValueError) as exc_info:
        connectivity_to_xarray(m, method="group_delay")

    error_msg = str(exc_info.value)
    assert "group_delay" in error_msg
    assert "result = conn.group_delay()" in error_msg