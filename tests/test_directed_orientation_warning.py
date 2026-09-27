"""The temporary warning that directed measures are source first since 3.0.

``pyproject.toml`` ignores ``DirectedOrientationWarning`` for the rest of the
suite, so each test here records warnings with the ``"always"`` filter.
"""

import warnings

import numpy as np
import pytest

import spectral_connectivity
from spectral_connectivity import (
    Connectivity,
    DirectedOrientationWarning,
    Multitaper,
    fourier_connectivity,
    list_measures,
    multitaper_connectivity,
)
from spectral_connectivity.connectivity import _ORIENTATION_CHANGED_MEASURES
from spectral_connectivity.wrapper import connectivity_to_xarray

_KWARGS = {
    "blockwise_spectral_granger_prediction": {"group_labels": np.array([0, 1])},
    "subset_pairwise_spectral_granger_prediction": {"pairs": [(0, 1)]},
    "delay": {"frequencies_of_interest": [5.0, 20.0]},
    "group_delay": {"frequencies_of_interest": [5.0, 20.0]},
    "phase_slope_index": {"frequencies_of_interest": [5.0, 20.0]},
}
_UNCHANGED = [
    *sorted(
        {measure.name for measure in list_measures(directed=True)}
        - _ORIENTATION_CHANGED_MEASURES
    ),
    "coherence_magnitude",
]


@pytest.fixture(scope="module")
def time_series():
    return np.random.default_rng(0).standard_normal((256, 4, 2))


@pytest.fixture(scope="module")
def connectivity(time_series):
    return Connectivity.from_multitaper(Multitaper(time_series, sampling_frequency=100))


def _orientation_warnings(record):
    return [w for w in record if issubclass(w.category, DirectedOrientationWarning)]


def test_warning_is_a_public_user_warning():
    assert issubclass(DirectedOrientationWarning, UserWarning)
    assert "DirectedOrientationWarning" in spectral_connectivity.__all__


def test_changed_measures_are_the_2x_target_first_directed_measures():
    directed = {measure.name for measure in list_measures(directed=True)}
    assert directed >= _ORIENTATION_CHANGED_MEASURES
    assert len(_ORIENTATION_CHANGED_MEASURES) == 7


@pytest.mark.parametrize("method", sorted(_ORIENTATION_CHANGED_MEASURES))
def test_changed_measure_warns_at_the_callers_line(connectivity, method):
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        getattr(connectivity, method)(**_KWARGS.get(method, {}))
    (warning,) = _orientation_warnings(record)
    message = str(warning.message)
    assert method in message
    assert "[..., source, target]" in message
    assert "migration-guide" in message
    assert warning.filename == __file__


@pytest.mark.parametrize("method", _UNCHANGED)
def test_measure_whose_orientation_did_not_change_is_silent(connectivity, method):
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        getattr(connectivity, method)(**_KWARGS.get(method, {}))
    assert _orientation_warnings(record) == []


def test_repeated_calls_warn_once_under_the_default_filter(connectivity):
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("default")
        for _ in range(3):
            connectivity.directed_transfer_function()
    assert len(_orientation_warnings(record)) == 1


def test_filtering_the_category_silences_it(connectivity):
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", category=DirectedOrientationWarning)
        connectivity.directed_transfer_function()
    assert _orientation_warnings(record) == []


def test_jackknife_warns_once_for_its_full_estimate(connectivity):
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        connectivity.jackknife("directed_transfer_function")
    (warning,) = _orientation_warnings(record)
    assert warning.filename == __file__


def test_wrapper_warns_once_about_its_labels(time_series):
    methods = [
        "pairwise_spectral_granger_prediction",
        "directed_transfer_function",
        "coherence_magnitude",
    ]
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        multitaper_connectivity(time_series, sampling_frequency=100, method=methods)
    (warning,) = _orientation_warnings(record)
    message = str(warning.message)
    assert "sel(source=a, target=b) is a -> b" in message
    # The 2.x wrapper rejected the transfer-function measures, so only Granger's
    # labels changed.
    assert "For pairwise_spectral_granger_prediction, sel" in message
    assert "[..., source, target]" not in message
    assert warning.filename == __file__


def test_wrapper_is_silent_for_unchanged_measures(time_series):
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        multitaper_connectivity(
            time_series,
            sampling_frequency=100,
            method=[
                "directed_phase_lag_index",
                "time_reversed_spectral_granger_prediction",
                "coherence_magnitude",
            ],
        )
    assert _orientation_warnings(record) == []


@pytest.mark.parametrize(
    "method",
    [
        "directed_transfer_function",
        "directed_coherence",
        "partial_directed_coherence",
        "generalized_partial_directed_coherence",
        "direct_directed_transfer_function",
    ],
)
def test_wrapper_is_silent_for_measures_the_2x_wrapper_rejected(time_series, method):
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        multitaper_connectivity(time_series, sampling_frequency=100, method=method)
    assert _orientation_warnings(record) == []


def test_connectivity_to_xarray_warns_once_about_its_labels(time_series):
    transform = Multitaper(time_series, sampling_frequency=100)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        connectivity_to_xarray(transform, method="pairwise_spectral_granger_prediction")
    (warning,) = _orientation_warnings(record)
    assert "sel(source=a, target=b)" in str(warning.message)
    assert warning.filename == __file__


def test_fourier_connectivity_is_silent(time_series):
    # fourier_connectivity is new in 3.0, so no 2.x code reads its labels.
    transform = Multitaper(time_series, sampling_frequency=100)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        fourier_connectivity(
            transform.fft(),
            frequencies=transform.frequencies,
            method="pairwise_spectral_granger_prediction",
        )
    assert _orientation_warnings(record) == []
