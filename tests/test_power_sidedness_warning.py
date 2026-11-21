"""The temporary warning that ``Connectivity.power`` is one-sided since 3.0.

``pyproject.toml`` ignores warnings that start with "Since spectral_connectivity
3.0," for the rest of the suite, so each test here records warnings with the
``"always"`` filter.
"""

import warnings

import numpy as np
import pytest

import spectral_connectivity
from spectral_connectivity import (
    Connectivity,
    MorletWavelet,
    Multitaper,
    PowerSidednessWarning,
    fourier_connectivity,
    multitaper_connectivity,
)
from spectral_connectivity.wrapper import connectivity_to_xarray


@pytest.fixture(scope="module")
def time_series():
    return np.random.default_rng(0).standard_normal((256, 4, 2))


@pytest.fixture(scope="module")
def connectivity(time_series):
    return Connectivity.from_transform(Multitaper(time_series, sampling_frequency=100))


def _sidedness_warnings(record):
    return [w for w in record if issubclass(w.category, PowerSidednessWarning)]


def test_warning_is_a_public_user_warning():
    assert issubclass(PowerSidednessWarning, UserWarning)
    assert "PowerSidednessWarning" in spectral_connectivity.__all__


def test_power_without_one_sided_warns_once_at_the_call(connectivity):
    """A 2.x call, ``power()``, gets values twice as large at interior bins
    as in 2.x, so it says so, at the caller's line, and how to choose."""
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        power = connectivity.power()
    sidedness = _sidedness_warnings(record)
    assert len(sidedness) == 1
    assert sidedness[0].filename == __file__
    message = str(sidedness[0].message)
    assert message.startswith("Since spectral_connectivity 3.0,")
    assert "one_sided=True" in message
    assert "one_sided=False" in message
    np.testing.assert_array_equal(power, connectivity.power(one_sided=True))


@pytest.mark.parametrize("one_sided", [True, False])
def test_power_with_one_sided_is_silent(connectivity, one_sided):
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        connectivity.power(one_sided=one_sided)
    assert _sidedness_warnings(record) == []


def test_power_of_one_sided_coefficients_is_silent(time_series):
    """One-sided coefficients (a Morlet transform, which 2.x did not have, or
    coefficients declared one-sided) are returned as given, as before."""
    morlet = Connectivity.from_transform(
        MorletWavelet(time_series, sampling_frequency=100, frequencies=np.array([10.0, 20.0]))
    )
    # (frequency, trial, signal) -> (time window, trial, taper, frequency, signal)
    half_spectrum = np.moveaxis(np.fft.rfft(time_series, axis=0), 0, 1)
    declared = Connectivity(half_spectrum[np.newaxis, :, np.newaxis], is_one_sided=True)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        morlet.power()
        declared.power()
    assert _sidedness_warnings(record) == []


def test_labelled_power_is_silent(time_series):
    """The wrappers' power results are one-sided too and do not warn, a
    documented exception; their attributes state the sidedness."""
    multitaper = Multitaper(time_series, sampling_frequency=100)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        results = [
            multitaper_connectivity(time_series, sampling_frequency=100, method="power"),
            multitaper_connectivity(time_series, sampling_frequency=100),
            connectivity_to_xarray(multitaper, "power"),
            fourier_connectivity(
                multitaper.fft(), multitaper.frequencies, method="power", is_one_sided=False
            ),
        ]
    assert _sidedness_warnings(record) == []
    assert all(
        "One-sided" in result["power"].attrs["interpretation"]
        if "power" in getattr(result, "data_vars", {})
        else "One-sided" in result.attrs["interpretation"]
        for result in results
    )


def test_jackknife_of_power_warns_once_for_its_full_estimate(connectivity):
    """The jackknife's estimate is power(), so it carries the same notice,
    once: its leave-one-out replicates do not repeat it."""
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        connectivity.jackknife("power")
    assert len(_sidedness_warnings(record)) == 1
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        connectivity.jackknife("power", one_sided=True)
    assert _sidedness_warnings(record) == []


def test_warning_is_silenced_by_its_category(connectivity):
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", category=PowerSidednessWarning)
        connectivity.power()
    assert _sidedness_warnings(record) == []
