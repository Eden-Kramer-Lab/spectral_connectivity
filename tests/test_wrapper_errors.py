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
    """The documented Multitaper keyword reaches the transform through the wrapper.

    NW = 4 is not the default (3), so a dropped keyword changes the recorded
    parameters: K = 2 * NW - 1 = 7 tapers and a 2 * NW / T = 8 / 0.2 s = 40 Hz
    resolution for the 200-sample (0.2 s at 1000 Hz) window.
    """
    time_series = np.random.default_rng(0).random((200, 5, 2))
    result = multitaper_connectivity(
        time_series,
        sampling_frequency=1000,
        method="coherence_magnitude",
        time_halfbandwidth_product=4,
    )
    assert result.name == "coherence_magnitude"
    assert result.attrs["mt_time_halfbandwidth_product"] == 4
    assert result.attrs["mt_n_tapers"] == 7
    assert np.isclose(result.attrs["mt_frequency_resolution"], 40.0)

    with pytest.raises(
        TypeError,
        match=(
            r"unexpected keyword argument 'time_bandwidth_product'\. "
            r"Did you mean 'time_halfbandwidth_product'\?"
        ),
    ):
        multitaper_connectivity(
            time_series,
            sampling_frequency=1000,
            method="coherence_magnitude",
            time_bandwidth_product=3,
        )


def test_connectivity_injection_is_rejected():
    """The formatter does not accept a separately constructed result object.

    ``connectivity`` collides with the formatter's own internal argument rather
    than replacing the object it builds, and ``_connectivity`` is forwarded to
    the measure, which rejects it as an unknown measure keyword.
    """
    rng = np.random.default_rng(5)
    m = Multitaper(rng.standard_normal((256, 3, 3)), sampling_frequency=500)
    connectivity = Connectivity.from_multitaper(m)

    with pytest.raises(TypeError, match="multiple values for argument 'connectivity'"):
        connectivity_to_xarray(m, "coherence_magnitude", connectivity=connectivity)
    with pytest.raises(
        TypeError,
        match=r"coherence_magnitude does not accept keyword argument\(s\) '_connectivity'",
    ):
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


def test_batch_skips_unsupported_measure_with_warning(monkeypatch, caplog):
    """In a multi-measure batch an unsupported measure is skipped, not fatal."""
    import logging

    monkeypatch.setattr(
        Connectivity,
        "custom_summary",
        lambda self: np.zeros((len(self.time), len(self.frequencies))),
        raising=False,
    )
    time_series = np.random.default_rng(1).standard_normal((256, 3, 2))
    with caplog.at_level(logging.WARNING):
        result = multitaper_connectivity(
            time_series,
            sampling_frequency=128,
            method=["coherence_magnitude", "custom_summary"],
        )
    assert "coherence_magnitude" in result.data_vars
    assert "custom_summary" not in result.data_vars
    assert "Skipping custom_summary" in caplog.text


def test_combine_formatted_results_reraises_on_coordinate_conflict():
    """Merging measures with conflicting coordinates raises an actionable error."""
    import xarray as xr

    from spectral_connectivity.wrapper import _combine_formatted_results

    first = xr.DataArray(
        [1.0, 2.0], dims=("frequency",), coords={"frequency": [1.0, 2.0]}, name="a"
    )
    second = xr.DataArray(
        [3.0, 4.0], dims=("frequency",), coords={"frequency": [1.0, 3.0]}, name="b"
    )
    with pytest.raises(ValueError, match="conflicting xarray variables or coordinates"):
        _combine_formatted_results([first, second], {})


def test_batch_of_only_unsupported_measures_raises(monkeypatch):
    """A batch in which every measure is unsupported fails loudly, not empty."""
    for name in ("custom_summary", "other_summary"):
        monkeypatch.setattr(
            Connectivity,
            name,
            lambda self: np.zeros((len(self.time), len(self.frequencies))),
            raising=False,
        )
    time_series = np.random.default_rng(2).standard_normal((256, 3, 2))
    with pytest.raises(UnsupportedMeasureError, match="None of the requested methods"):
        multitaper_connectivity(
            time_series,
            sampling_frequency=128,
            method=["custom_summary", "other_summary"],
        )


def test_signal_names_length_is_validated():
    """Coordinate mismatches fail before xarray emits a lower-level error."""
    m = Multitaper(np.zeros((128, 3, 2)), sampling_frequency=128)
    with pytest.raises(ValueError, match="signal_names must contain 2 names"):
        connectivity_to_xarray(m, signal_names=["only-one"])
