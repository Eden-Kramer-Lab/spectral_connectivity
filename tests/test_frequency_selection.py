"""Frequency selection: ``frequency_range`` includes both endpoints everywhere.

One helper decides which bins an interval selects, for the wrappers' crop,
named bands, and the estimators that consume a frequency interval (``delay``,
``group_delay``, ``phase_slope_index``). A bin lies on a bound when they differ by
at most two units in the last place of each, in its own stored precision, so
an edge written as a decimal still selects the bin a float32 or
decimal-rounded coordinate stores for it.
"""

import json
import warnings

import numpy as np
import pytest
import xarray as xr

from spectral_connectivity import Connectivity, MorletWavelet, Multitaper
from spectral_connectivity.connectivity import _frequency_interval_mask
from spectral_connectivity.wrapper import (
    connectivity_to_xarray,
    fourier_connectivity,
    multitaper_connectivity,
)
from tests._backend_helpers import to_host

# Bins 0, 2, ..., 50 Hz: every bound below is either exactly a bin or between bins.
_EXACT_GRID = np.arange(0.0, 51.0, 2.0)


@pytest.mark.parametrize(
    ("bounds", "closed", "expected"),
    [
        pytest.param((10.0, 20.0), "both", [10, 12, 14, 16, 18, 20], id="both-on-bins"),
        pytest.param((10.0, 20.0), "neither", [12, 14, 16, 18], id="neither-on-bins"),
        pytest.param((9.0, 21.0), "both", [10, 12, 14, 16, 18, 20], id="both-off-bins"),
        pytest.param((9.0, 21.0), "neither", [10, 12, 14, 16, 18, 20], id="neither-off"),
        pytest.param((10.0, 10.0), "both", [10], id="both-single-bin"),
        pytest.param((0.0, 50.0), "both", list(range(0, 51, 2)), id="both-whole-grid"),
    ],
)
def test_bounds_on_bins_are_included_only_when_closed_on_both_sides(bounds, closed, expected):
    mask, selected_bounds = _frequency_interval_mask(
        "frequency_range", bounds, _EXACT_GRID, closed=closed
    )
    np.testing.assert_array_equal(_EXACT_GRID[mask], expected)
    assert selected_bounds == bounds


def test_a_bound_just_off_a_bin_does_not_select_it():
    """The tolerance is rounding-sized, not a fraction of the bin spacing."""
    mask, _ = _frequency_interval_mask(
        "frequency_range", (10.0 + 1e-6, 20.0 - 1e-6), _EXACT_GRID, closed="both"
    )
    np.testing.assert_array_equal(_EXACT_GRID[mask], [12, 14, 16, 18])


@pytest.mark.parametrize(("n_last_places", "selects_edge_bins"), [(4, True), (5, False)])
def test_a_bin_is_on_a_bound_within_two_last_places_of_each(n_last_places, selects_edge_bins):
    """|f - b| <= 2 * (ulp(f) + ulp(b)): at 10 and 20 Hz in float64 that is
    four units in the last place (both values share one ulp), and no more."""

    def nudge(value, direction):
        for _ in range(n_last_places):
            value = np.nextafter(value, direction)
        return float(value)

    bounds = (nudge(10.0, np.inf), nudge(20.0, -np.inf))
    mask, _ = _frequency_interval_mask("frequency_range", bounds, _EXACT_GRID, closed="both")
    expected = [10, 12, 14, 16, 18, 20] if selects_edge_bins else [12, 14, 16, 18]
    np.testing.assert_array_equal(_EXACT_GRID[mask], expected)


@pytest.mark.parametrize(
    "grid",
    [
        # 0.1 * k in float64: bin 3 is 0.30000000000000004 and bin 7 is
        # 0.7000000000000001, so exact comparison would drop the 0.7 bin.
        pytest.param(np.arange(64) * 0.1, id="decimal-rounded-float64"),
        # float32(0.7) is below 0.7 and float32(1.2) above 1.2, so exact
        # comparison would drop both edge bins.
        pytest.param(np.linspace(0, 6.3, 64).astype(np.float32), id="float32"),
    ],
)
@pytest.mark.parametrize("bounds", [(0.3, 0.7), (0.7, 1.2)])
def test_both_includes_a_bin_stored_with_rounding_of_its_decimal_bound(grid, bounds):
    """Bin ``k`` stores ``0.1 * k``, so the bounds name bins 3-7 or 7-12."""
    mask, _ = _frequency_interval_mask("frequency_range", bounds, grid, closed="both")
    np.testing.assert_array_equal(
        np.flatnonzero(mask), np.arange(round(bounds[0] * 10), round(bounds[1] * 10) + 1)
    )


def test_the_rounded_grids_have_edge_bins_that_exact_comparison_drops():
    """Guard for the test above: without the tolerance its edge bins are lost
    (bins are compared in float64)."""
    decimal = np.arange(64) * 0.1
    assert not decimal[7] <= 0.7
    # ... and the exclusive comparison keeps bin 3 inside a band starting at 0.3.
    assert decimal[3] > 0.3
    float32 = np.linspace(0, 6.3, 64).astype(np.float32).astype(np.float64)
    assert not float32[7] >= 0.7
    assert not float32[12] <= 1.2


@pytest.mark.parametrize(
    "grid",
    [
        pytest.param(np.arange(64) * 0.1, id="decimal-rounded-float64"),
        pytest.param(np.linspace(0, 6.3, 64).astype(np.float32), id="float32"),
    ],
)
@pytest.mark.parametrize("bounds", [(0.3, 0.7), (0.7, 1.2), (1.0, 5.0)])
def test_neither_is_the_historical_exact_exclusive_comparison(grid, bounds):
    """The deprecated exclusive selection is kept bit for bit, including where
    a rounded grid puts a bin just inside an edge. The reference is the
    comparison the exclusive selection has always made."""
    band = np.asarray(bounds, dtype=float)
    historical = (band[0] < grid) & (grid < band[1])
    mask, _ = _frequency_interval_mask(
        "frequencies_of_interest", bounds, grid, closed="neither"
    )
    np.testing.assert_array_equal(mask, historical)


def test_float32_bounds_on_a_float64_grid_use_float32_rounding():
    """Bounds read off a float32 coordinate still select the bins they name."""
    bounds = tuple(np.linspace(0, 6.3, 64).astype(np.float32)[[7, 12]])
    grid = np.arange(64) * 0.1
    mask, _ = _frequency_interval_mask("frequency_range", bounds, grid, closed="both")
    np.testing.assert_array_equal(np.flatnonzero(mask), np.arange(7, 13))


@pytest.mark.parametrize(
    ("bounds", "closed", "message"),
    [
        ([10.0], "both", "must be two finite values"),
        ([10.0, 20.0, 30.0], "both", "must be two finite values"),
        ([20.0, 10.0], "both", r"low <= high"),
        ([np.nan, 10.0], "both", "must be two finite values"),
        ([0.0, np.inf], "both", "must be two finite values"),
        ([10.0, 10.0], "neither", r"low < high"),
        ([12.6, 13.5], "both", "contains no frequency bin"),
        ([60.0, 70.0], "neither", "contains no frequency bin"),
        ([10.0, 12.0], "neither", "contains no frequency bin"),
    ],
)
def test_invalid_or_empty_intervals_raise_naming_the_argument(bounds, closed, message):
    with pytest.raises(ValueError, match=message) as error:
        _frequency_interval_mask("frequency_range", bounds, _EXACT_GRID, closed=closed)
    assert str(error.value).startswith("frequency_range")


def test_wrapper_crop_includes_rounded_edge_bins_of_a_float32_coordinate():
    """The wrapper's crop uses the same rule, so a float32 frequency coordinate
    keeps the bins its decimal bounds name."""
    rng = np.random.default_rng(3)
    grid = np.linspace(0, 6.3, 64).astype(np.float32)
    coefficients = xr.DataArray(
        rng.standard_normal((8, 64, 2)) + 1j * rng.standard_normal((8, 64, 2)),
        dims=("trial", "frequency", "signal"),
        coords={"frequency": grid},
    )
    cropped = fourier_connectivity(
        coefficients, method="coherence_magnitude", frequency_range=(0.7, 1.2)
    )
    np.testing.assert_array_equal(cropped.frequency.values, grid[7:13])


def test_wrapper_crop_and_bands_include_bins_on_their_bounds():
    time_series = np.random.default_rng(4).standard_normal((512, 3, 2))
    kwargs = {"sampling_frequency": 128, "method": "coherence_magnitude"}
    full = multitaper_connectivity(time_series, **kwargs)
    assert np.isin([8.0, 12.0], full.frequency.values).all()

    cropped = multitaper_connectivity(time_series, frequency_range=(8.0, 12.0), **kwargs)
    np.testing.assert_array_equal(
        cropped.frequency.values,
        full.frequency.values[(full.frequency.values >= 8) & (full.frequency.values <= 12)],
    )
    assert cropped.frequency.values[[0, -1]].tolist() == [8.0, 12.0]

    band = multitaper_connectivity(
        time_series, frequency_bands={"alpha": (8.0, 12.0)}, **kwargs
    )
    np.testing.assert_allclose(
        band.sel(band="alpha").values, cropped.mean("frequency").values, rtol=1e-12
    )


# --- Measures that consume a frequency interval --------------------------------

_SELECTING_MEASURES = ("delay", "group_delay", "phase_slope_index")
_LAGGED_TRANSFORM = {"sampling_frequency": 500, "time_window_duration": 0.5}


@pytest.fixture(scope="module")
def lagged_time_series():
    """Signal 0 leads signal 1 by 3 samples (6 ms at 500 Hz), plus noise.

    Returns
    -------
    time_series : np.ndarray, shape (1000, 20, 2)
    """
    rng = np.random.default_rng(0)
    leader = rng.standard_normal((1003, 20))
    time_series = np.stack([leader[3:], leader[:-3]], axis=-1)
    return time_series + 0.5 * rng.standard_normal(time_series.shape)


@pytest.fixture(scope="module")
def lagged_connectivity(lagged_time_series):
    """``Connectivity`` on a 2 Hz grid (0.5 s windows at 500 Hz)."""
    connectivity = Connectivity.from_transform(
        Multitaper(lagged_time_series, **_LAGGED_TRANSFORM)
    )
    assert np.isin([10.0, 50.0], connectivity.frequencies).all()
    return connectivity


def _call(connectivity, measure, *args, **kwargs):
    """The measure's values as one array (group delay's three outputs stacked)."""
    result = getattr(connectivity, measure)(*args, **kwargs)
    return np.stack(result) if measure == "group_delay" else result


def _deprecated(connectivity, measure, *args, **kwargs):
    with pytest.warns(DeprecationWarning, match="frequencies_of_interest"):
        return _call(connectivity, measure, *args, **kwargs)


@pytest.mark.parametrize("measure", _SELECTING_MEASURES)
def test_frequency_range_consumes_the_bins_on_its_bounds(lagged_connectivity, measure):
    """frequency_range=(10, 50) uses the 10 and 50 Hz bins: it equals the
    exclusive selection of (9, 51), whose edges lie between bins, and differs
    from the exclusive selection of (10, 50)."""
    inclusive = _call(lagged_connectivity, measure, frequency_range=(10.0, 50.0))
    np.testing.assert_array_equal(
        inclusive,
        _deprecated(lagged_connectivity, measure, frequencies_of_interest=(9.0, 51.0)),
    )
    exclusive = _deprecated(lagged_connectivity, measure, frequencies_of_interest=(10.0, 50.0))
    assert inclusive.shape != exclusive.shape or not np.allclose(
        inclusive, exclusive, equal_nan=True
    )


def test_delay_returns_the_bins_of_frequency_range(lagged_connectivity):
    delays = lagged_connectivity.delay(frequency_range=(10.0, 50.0), n_range=1)
    # 10, 12, ..., 50 Hz.
    assert delays.shape[-4] == 21


@pytest.mark.parametrize("measure", _SELECTING_MEASURES)
def test_frequencies_of_interest_warns_with_an_exact_migration(lagged_connectivity, measure):
    """The deprecated exclusive keyword names its replacement, removal in 4.0,
    and a frequency_range that reproduces its result exactly."""
    with pytest.warns(DeprecationWarning, match="frequencies_of_interest") as record:
        deprecated = _call(lagged_connectivity, measure, frequencies_of_interest=(10.0, 50.0))
    (warning,) = [w for w in record if issubclass(w.category, DeprecationWarning)]
    message = str(warning.message)
    assert warning.filename == __file__
    assert "removed in 4.0" in message
    assert (
        "replace frequencies_of_interest=(10.0, 50.0) with frequency_range=(12.0, 48.0)"
        in message
    )
    np.testing.assert_array_equal(
        deprecated, _call(lagged_connectivity, measure, frequency_range=(12.0, 48.0))
    )


@pytest.mark.parametrize("measure", _SELECTING_MEASURES)
def test_positional_band_is_the_deprecated_exclusive_selection(lagged_connectivity, measure):
    """A 2.x positional band keeps its exclusive meaning, with the warning,
    instead of silently gaining the edge bins."""
    np.testing.assert_array_equal(
        _deprecated(lagged_connectivity, measure, (10.0, 50.0)),
        _call(lagged_connectivity, measure, frequency_range=(12.0, 48.0)),
    )


@pytest.mark.parametrize("measure", _SELECTING_MEASURES)
def test_both_spellings_are_rejected(lagged_connectivity, measure):
    with pytest.raises(ValueError, match="frequency_range and the deprecated"):
        getattr(lagged_connectivity, measure)(
            frequency_range=(10.0, 50.0), frequencies_of_interest=(10.0, 50.0)
        )


def _measure_variables(result):
    """The result's measure variables (all three of group delay's)."""
    return list(result.data_vars.values()) if isinstance(result, xr.Dataset) else [result]


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        pytest.param(
            {"frequency_range": (10.0, 50.0)},
            {
                "requested_frequency_lower": 10.0,
                "requested_frequency_upper": 50.0,
                "selected_frequency_lower": 10.0,
                "selected_frequency_upper": 50.0,
                "frequency_interval_closed": "both",
            },
            id="on-bins",
        ),
        pytest.param(
            {"frequency_range": (9.0, 51.0)},
            {
                "requested_frequency_lower": 9.0,
                "requested_frequency_upper": 51.0,
                "selected_frequency_lower": 10.0,
                "selected_frequency_upper": 50.0,
                "frequency_interval_closed": "both",
            },
            id="off-bins",
        ),
        pytest.param(
            {},
            {
                "selected_frequency_lower": 0.0,
                "selected_frequency_upper": 250.0,
                "frequency_interval_closed": "both",
            },
            id="every-bin",
        ),
    ],
)
@pytest.mark.parametrize("measure", _SELECTING_MEASURES)
def test_results_record_the_requested_and_selected_interval(
    lagged_time_series, measure, kwargs, expected
):
    result = connectivity_to_xarray(
        Multitaper(lagged_time_series, **_LAGGED_TRANSFORM), method=measure, **kwargs
    )
    for variable in _measure_variables(result):
        assert {key: variable.attrs[key] for key in expected} == expected
        assert ("requested_frequency_lower" in variable.attrs) == bool(kwargs)
        assert json.loads(variable.attrs["measure_kwargs_json"]) == {
            key: list(value) for key, value in kwargs.items()
        }
    if measure == "delay":
        assert result.frequency.values[[0, -1]].tolist() == [
            expected["selected_frequency_lower"],
            expected["selected_frequency_upper"],
        ]


@pytest.mark.parametrize("measure", _SELECTING_MEASURES)
def test_deprecated_results_record_the_exclusive_interval_and_keyword(
    lagged_time_series, measure
):
    with pytest.warns(DeprecationWarning, match="frequencies_of_interest"):
        result = connectivity_to_xarray(
            Multitaper(lagged_time_series, **_LAGGED_TRANSFORM),
            method=measure,
            frequencies_of_interest=(10.0, 50.0),
        )
    for variable in _measure_variables(result):
        assert variable.attrs["requested_frequency_lower"] == 10.0
        assert variable.attrs["requested_frequency_upper"] == 50.0
        assert variable.attrs["selected_frequency_lower"] == 12.0
        assert variable.attrs["selected_frequency_upper"] == 48.0
        assert variable.attrs["frequency_interval_closed"] == "neither"
        assert json.loads(variable.attrs["measure_kwargs_json"]) == {
            "frequencies_of_interest": [10.0, 50.0]
        }
        assert "arg_frequencies_of_interest_json" in variable.attrs
        assert "arg_frequency_range_json" not in variable.attrs
    if measure == "delay":
        assert result.frequency.values[[0, -1]].tolist() == [12.0, 48.0]


@pytest.mark.parametrize(
    ("kwargs", "consumed_bins"),
    [
        pytest.param({"frequency_range": (8.0, 16.0)}, [1, 2, 3], id="frequency_range"),
        pytest.param({"frequencies_of_interest": (8.0, 20.0)}, [2, 3], id="deprecated"),
    ],
)
@pytest.mark.parametrize("measure", _SELECTING_MEASURES)
def test_morlet_validity_follows_the_consumed_bins(measure, kwargs, consumed_bins):
    """valid_time of a frequency-reduced measure, and valid_time_frequency of
    delay, describe exactly the bins the estimator used."""
    transform = MorletWavelet(
        np.random.default_rng(319).standard_normal((192, 5, 2)),
        64,
        np.array([4.0, 8.0, 12.0, 16.0, 20.0]),
        edge_mode="nan",
    )
    validity = to_host(transform.valid_time_frequency)[:, consumed_bins]
    # The bins differ in validity, so using the wrong ones would show.
    assert not np.array_equal(
        validity.all(axis=1), to_host(transform.valid_time_frequency).all(axis=1)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        result = connectivity_to_xarray(transform, method=measure, **kwargs)
    if measure == "delay":
        np.testing.assert_array_equal(result.valid_time_frequency, validity)
    else:
        np.testing.assert_array_equal(result.valid_time, validity.all(axis=1))
