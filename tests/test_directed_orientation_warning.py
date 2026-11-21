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
from tests._fresh_session import run_fresh_session

_KWARGS = {
    "blockwise_spectral_granger_prediction": {"group_labels": np.array([0, 1])},
    "subset_pairwise_spectral_granger_prediction": {"pairs": [(0, 1)]},
    "delay": {"frequency_range": (5.0, 20.0)},
    "group_delay": {"frequency_range": (5.0, 20.0)},
    "phase_slope_index": {"frequency_range": (5.0, 20.0)},
}
_UNCHANGED = [
    *sorted(
        {measure.name for measure in list_measures(directed=True)}
        - _ORIENTATION_CHANGED_MEASURES
    ),
    "magnitude_squared_coherence",
]

# Each message says it will not repeat, and how to see it at every call.
_ONCE_PER_SESSION = (
    "Shown once per Python session; to see it at every call: "
    'warnings.simplefilter("always", spectral_connectivity.DirectedOrientationWarning).'
)


@pytest.fixture(scope="module")
def time_series():
    return np.random.default_rng(0).standard_normal((256, 4, 2))


@pytest.fixture(scope="module")
def connectivity(time_series):
    return Connectivity.from_transform(Multitaper(time_series, sampling_frequency=100))


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
    # pyproject.toml filters the warning for the rest of the suite by this prefix.
    assert message.startswith("Since spectral_connectivity 3.0, ")
    assert method in message
    assert "[..., source, target]" in message
    assert "migration-guide" in message
    assert _ONCE_PER_SESSION in message
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


_SESSION_SCRIPT = """
granger = "pairwise_spectral_granger_prediction"
connectivity.directed_transfer_function()
connectivity.directed_transfer_function()  # another line
connectivity.jackknife("directed_transfer_function")
connectivity.partial_directed_coherence()
multitaper_connectivity(time_series, sampling_frequency=100, method=granger)
multitaper_connectivity(time_series, sampling_frequency=100, method=granger)
multitaper_connectivity(
    time_series,
    sampling_frequency=100,
    method="subset_pairwise_spectral_granger_prediction",
    connectivity_kwargs={"pairs": [(0, 1)]},
)
first = list(shown)
warnings.simplefilter("always", DirectedOrientationWarning)
connectivity.directed_transfer_function()
print(json.dumps([first, shown[len(first):]]))
"""


def test_warning_appears_once_per_session_for_each_measure():
    """Under Python's default filters each measure's notice appears once per
    session, whichever line calls it; the wrappers' label notice is separate,
    and also once per measure. A filter set after import shows it again."""
    first, after_always = run_fresh_session(_SESSION_SCRIPT)
    # Each message's fourth word: the measure, or "sel(source=a," for the labels.
    assert [message.split()[3] for message in first] == [
        "directed_transfer_function",
        "partial_directed_coherence",
        "sel(source=a,",
        "sel(source=a,",
    ]
    assert [message.split()[3] for message in after_always] == ["directed_transfer_function"]


_NONFINITE_SCRIPT = """
nan_series = time_series.copy()
nan_series[10, 0, 0] = np.nan
nan_connectivity = Connectivity.from_transform(
    Multitaper(nan_series, sampling_frequency=100)
)
granger = "pairwise_spectral_granger_prediction"
nan_connectivity.directed_transfer_function()
nan_connectivity.directed_transfer_function()  # another line
nan_connectivity.partial_directed_coherence()
multitaper_connectivity(nan_series, sampling_frequency=100, method=granger)
multitaper_connectivity(nan_series, sampling_frequency=100, method=granger)
print(json.dumps(shown))
"""


def test_nonfinite_input_still_shows_each_notice_once_per_session():
    """A directed measure of NaN coefficients drops its factorization
    warnings inside ``catch_warnings``; the orientation notice, issued
    outside that block, still appears once per measure, and so does the
    wrappers' label notice."""
    shown = run_fresh_session(_NONFINITE_SCRIPT)
    notices = [message.split()[3] for message in shown if message.startswith("Since ")]
    assert notices == [
        "directed_transfer_function",
        "partial_directed_coherence",
        "sel(source=a,",
    ]
    assert any("NaN or infinite value" in message for message in shown)


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
        "magnitude_squared_coherence",
    ]
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        multitaper_connectivity(time_series, sampling_frequency=100, method=methods)
    (warning,) = _orientation_warnings(record)
    message = str(warning.message)
    # The 2.x wrapper rejected the transfer-function measures, so only Granger's
    # labels changed.
    assert (
        "sel(source=a, target=b) of pairwise_spectral_granger_prediction is a -> b" in message
    )
    assert "[..., source, target]" not in message
    assert _ONCE_PER_SESSION in message
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
                "magnitude_squared_coherence",
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
