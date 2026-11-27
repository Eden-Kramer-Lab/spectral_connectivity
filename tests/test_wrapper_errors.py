"""Error contracts for the high-level xarray wrapper."""

import warnings

import numpy as np
import pytest

from spectral_connectivity.connectivity import Connectivity
from spectral_connectivity.transforms import Multitaper, VARSpectrum
from spectral_connectivity.wrapper import (
    UnsupportedMeasureError,
    connectivity_to_xarray,
    fourier_connectivity,
    multitaper_connectivity,
)


@pytest.fixture(scope="module")
def two_signal_multitaper():
    return Multitaper(
        np.random.default_rng(0).standard_normal((256, 3, 2)), sampling_frequency=100
    )


@pytest.mark.parametrize(
    ("method", "received"),
    [(3, "int 3"), (["power", 3], "a list holding int 3")],
    ids=["not-a-name", "list-holding-not-a-name"],
)
def test_connectivity_to_xarray_rejects_a_method_that_is_not_names(
    two_signal_multitaper, method, received
):
    """connectivity_to_xarray takes a name or a list of names."""
    with pytest.raises(
        TypeError,
        match=rf"^method must be a measure name \(str\) or a list of names, got {received}\.",
    ):
        connectivity_to_xarray(two_signal_multitaper, method)


def test_connectivity_to_xarray_rejects_none_and_names_the_default_set(
    two_signal_multitaper,
):
    """It has no default set for None to request, so the error says how to ask
    for multitaper_connectivity's."""
    with pytest.raises(TypeError, match=r"^method=None is not accepted") as excinfo:
        connectivity_to_xarray(two_signal_multitaper, None)
    assert "method=spectral_connectivity.DEFAULT_METHODS" in str(excinfo.value)
    assert "list_measures()" in str(excinfo.value)


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
        method="magnitude_squared_coherence",
        time_halfbandwidth_product=4,
    )
    assert result.name == "magnitude_squared_coherence"
    assert result.attrs["mt_time_halfbandwidth_product"] == 4
    assert result.attrs["mt_n_tapers"] == 7
    assert np.isclose(result.attrs["mt_frequency_resolution"], 40.0)

    # The package's own message, which suggests the named setting.
    with pytest.raises(
        TypeError,
        match=r"unexpected keyword argument 'time_bandwidth_product'\. "
        r"Did you mean 'time_halfbandwidth_product'",
    ):
        multitaper_connectivity(
            time_series,
            sampling_frequency=1000,
            method="magnitude_squared_coherence",
            time_bandwidth_product=3,
        )


def test_connectivity_injection_is_rejected():
    """The formatter does not accept a separately constructed result object.

    Keyword arguments are measure settings, checked against the measure before
    anything is built, so ``connectivity`` and ``_connectivity`` are rejected as
    unknown measure keywords rather than replacing the object the formatter
    builds.
    """
    rng = np.random.default_rng(5)
    m = Multitaper(rng.standard_normal((256, 3, 3)), sampling_frequency=500)
    connectivity = Connectivity.from_transform(m)

    for name in ("connectivity", "_connectivity"):
        with pytest.raises(
            TypeError,
            match=rf"magnitude_squared_coherence does not accept keyword argument\(s\) '{name}'",
        ):
            connectivity_to_xarray(m, "magnitude_squared_coherence", **{name: connectivity})


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
            method=["magnitude_squared_coherence", "custom_summary"],
        )
    assert "magnitude_squared_coherence" in result.data_vars
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


class _MaskedMultitaper(Multitaper):
    """A transform whose validity mask has the wrong shape."""

    valid_time_frequency = np.ones((2, 2), dtype=bool)


@pytest.mark.parametrize(
    ("transform", "kwargs", "match"),
    [
        pytest.param(_MaskedMultitaper, {}, "valid_time_frequency", id="mask-shape"),
        pytest.param(
            Multitaper, {"frequency_decimation": 0}, "frequency_decimation", id="decimation"
        ),
    ],
)
@pytest.mark.parametrize(
    "method",
    ["coherence_magnitude", ["coherence_magnitude", "power"]],
    ids=["name", "list"],
)
def test_connectivity_to_xarray_rejected_call_does_not_warn(transform, kwargs, match, method):
    """A call that raises does not also warn about a deprecated name it was
    given: the warning comes only from a call that succeeds."""
    time_series = np.random.default_rng(0).standard_normal((256, 2, 3))
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match=match):
            connectivity_to_xarray(
                transform(time_series, sampling_frequency=100), method, **kwargs
            )
    assert record == []


@pytest.mark.parametrize(
    "wrapper",
    [
        lambda transform: multitaper_connectivity(transform, sampling_frequency=100),
        fourier_connectivity,
    ],
    ids=["multitaper_connectivity", "fourier_connectivity"],
)
@pytest.mark.parametrize(
    "make",
    [
        lambda: VARSpectrum(np.array([[[0.5, 0.0], [0.4, 0.5]]]), sampling_frequency=100),
        lambda: Multitaper(np.zeros((100, 1, 2)), sampling_frequency=100),
    ],
    ids=["VARSpectrum", "Multitaper"],
)
def test_a_transform_given_as_data_points_to_connectivity_to_xarray(wrapper, make):
    """A transform passed where data goes names the route for transforms,
    not a conversion with np.asarray."""
    transform = make()
    with pytest.raises(
        TypeError, match=rf"is a spectral transform \({type(transform).__name__}\)"
    ) as info:
        wrapper(transform)
    message = str(info.value)
    assert "connectivity_to_xarray(transform, method)" in message
    assert "np.asarray" not in message
