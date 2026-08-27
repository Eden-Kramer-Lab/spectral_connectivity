"""Error contracts for the high-level xarray wrapper."""

import numpy as np
import pytest

from spectral_connectivity.connectivity import Connectivity
from spectral_connectivity.transforms import Multitaper
from spectral_connectivity.wrapper import (
    UnsupportedMeasureError,
    connectivity_to_xarray,
    multitaper_connectivity,
)


def test_unsupported_method_error_message():
    """Incompatible result shapes point users to the lower-level API."""
    m = Multitaper(
        np.random.default_rng(0).random((100, 5, 2)),
        sampling_frequency=1000,
        time_window_duration=0.1,
    )

    for method in ("phase_slope_index", "canonical_coherence", "group_delay"):
        with pytest.raises(UnsupportedMeasureError) as exc_info:
            connectivity_to_xarray(m, method=method)
        message = str(exc_info.value)
        assert "is not supported by the xarray interface" in message
        assert "Connectivity class directly" in message
        assert "from spectral_connectivity import Connectivity" in message
        assert f"result = conn.{method}()" in message

    with pytest.raises(UnsupportedMeasureError, match="global_coherence"):
        connectivity_to_xarray(m, method="global_coherence")


def test_time_halfbandwidth_product_kwarg_passthrough():
    """The documented Multitaper keyword works through the wrapper."""
    time_series = np.random.default_rng(0).random((200, 5, 2))
    result = multitaper_connectivity(
        time_series,
        sampling_frequency=1000,
        method="coherence_magnitude",
        time_halfbandwidth_product=3,
    )
    assert result.name == "coherence_magnitude"

    with pytest.raises(TypeError):
        multitaper_connectivity(
            time_series,
            sampling_frequency=1000,
            method="coherence_magnitude",
            time_bandwidth_product=3,
        )


def test_connectivity_injection_is_rejected():
    """The formatter does not accept a separately constructed result object."""
    rng = np.random.default_rng(5)
    m = Multitaper(rng.standard_normal((256, 3, 3)), sampling_frequency=500)
    connectivity = Connectivity.from_multitaper(m)

    with pytest.raises(TypeError):
        connectivity_to_xarray(m, "coherence_magnitude", connectivity=connectivity)
    with pytest.raises(TypeError):
        connectivity_to_xarray(m, "coherence_magnitude", _connectivity=connectivity)


def test_unregistered_nonpairwise_extension_is_rejected(monkeypatch):
    """Unknown methods must prove the pairwise shape instead of being assumed."""
    monkeypatch.setattr(
        Connectivity,
        "custom_summary",
        lambda self: np.zeros((len(self.time), len(self.frequencies))),
        raising=False,
    )
    m = Multitaper(np.zeros((128, 3, 2)), sampling_frequency=128)
    with pytest.raises(UnsupportedMeasureError, match="unregistered wrapper extension"):
        connectivity_to_xarray(m, "custom_summary")


def test_signal_names_length_is_validated():
    """Coordinate mismatches fail before xarray emits a lower-level error."""
    m = Multitaper(np.zeros((128, 3, 2)), sampling_frequency=128)
    with pytest.raises(ValueError, match="signal_names must contain 2 names"):
        connectivity_to_xarray(m, signal_names=["only-one"])
