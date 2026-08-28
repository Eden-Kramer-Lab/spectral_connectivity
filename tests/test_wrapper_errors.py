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
