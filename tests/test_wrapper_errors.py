import numpy as np
import pytest

from spectral_connectivity.transforms import Multitaper
from spectral_connectivity.wrapper import (
    connectivity_to_xarray,
    multitaper_connectivity,
)


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

    # global_coherence returns a tuple; the wrapper must give the friendly
    # message rather than a cryptic xarray "coords is not dict-like" error.
    with pytest.raises(ValueError) as exc_info:
        connectivity_to_xarray(m, method="global_coherence")
    error_msg = str(exc_info.value)
    assert "is not supported by the xarray interface" in error_msg
    assert "result = conn.global_coherence()" in error_msg


def test_time_halfbandwidth_product_kwarg_passthrough():
    """The documented Multitaper kwarg name works through the wrapper."""
    rng = np.random.default_rng(0)
    time_series = rng.random((200, 5, 2))
    result = multitaper_connectivity(
        time_series,
        sampling_frequency=1000,
        method="coherence_magnitude",
        time_halfbandwidth_product=3,
    )
    assert result.name == "coherence_magnitude"
    # The (misspelled) name from the old docstring must not silently work.
    with pytest.raises(TypeError):
        multitaper_connectivity(
            time_series,
            sampling_frequency=1000,
            method="coherence_magnitude",
            time_bandwidth_product=3,
        )


def test_injected_connectivity_mismatch_raises():
    """A Connectivity not built from ``m`` is rejected, not silently mislabeled.

    ``connectivity_to_xarray`` takes results/coordinates from the injected
    ``connectivity`` but metadata from ``m``; a mismatched pair (e.g. a
    different sampling frequency) would otherwise produce output whose frequency
    axis and ``mt_sampling_frequency`` attribute disagree.
    """
    from spectral_connectivity.connectivity import Connectivity

    rng = np.random.default_rng(0)
    time_series = rng.standard_normal((512, 3, 4))
    m_500 = Multitaper(time_series, sampling_frequency=500)
    m_1000 = Multitaper(time_series, sampling_frequency=1000)
    connectivity_500 = Connectivity.from_multitaper(m_500)

    # Consistent instance is accepted.
    connectivity_to_xarray(m_500, "coherence_magnitude", connectivity=connectivity_500)

    # Different sampling frequency -> different frequency grid -> rejected.
    with pytest.raises(ValueError, match="not built from this `Multitaper`"):
        connectivity_to_xarray(
            m_1000, "coherence_magnitude", connectivity=connectivity_500
        )

    # Different channel count -> rejected.
    m_two_signals = Multitaper(time_series[..., :2], sampling_frequency=500)
    with pytest.raises(ValueError, match="n_signals"):
        connectivity_to_xarray(
            m_two_signals, "coherence_magnitude", connectivity=connectivity_500
        )
