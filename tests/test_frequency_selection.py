"""Frequency selection: ``frequency_range`` includes both endpoints everywhere.

One helper decides which bins an interval selects, for the wrappers' crop,
named bands, and the estimators that consume a frequency interval (``delay``,
``group_delay``, ``phase_slope_index``). A bin lies on a bound when they differ by
at most two units in the last place of each, in its own stored precision, so
an edge written as a decimal still selects the bin a float32 or
decimal-rounded coordinate stores for it.
"""

import numpy as np
import pytest
import xarray as xr

from spectral_connectivity.connectivity import _frequency_interval_mask
from spectral_connectivity.wrapper import fourier_connectivity, multitaper_connectivity

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
