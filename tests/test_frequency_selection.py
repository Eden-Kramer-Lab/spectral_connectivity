"""Frequency selection: ``frequency_range`` includes both endpoints everywhere.

One helper decides which bins an interval selects, for the wrappers' crop,
named bands, and the estimators that consume a frequency interval (``delay``,
``group_delay``, ``phase_slope_index``). Bin ``f`` lies on bound ``b`` when
``|f - b| <= 1e-6 * gap(f) + 2 * (ulp(f) + ulp(b))``, with ``gap(f)`` the
distance to its nearest neighbouring bin and ``ulp`` each value's unit in the
last place in its own stored precision -- the allowances of the uniform-grid
check. So an edge written as a decimal still selects the bin a coordinate
stores for it when that bin drifted by at most a millionth of its spacing.
"""

import inspect
import json
import warnings

import numpy as np
import pytest
import xarray as xr

from spectral_connectivity import Connectivity, MorletWavelet, Multitaper
from spectral_connectivity._measure_registry import _FREQUENCY_RANGE_MEASURES, _MEASURE_SPECS
from spectral_connectivity.connectivity import (
    _frequencies_of_interest_mask,
    _frequency_interval_mask,
    _uniform_grid_spacing,
)
from spectral_connectivity.wrapper import (
    connectivity_to_xarray,
    fourier_connectivity,
    frequency_band_reduce,
    multitaper_connectivity,
)
from tests._backend_helpers import to_host

# Bins 0, 2, ..., 50 Hz: every bound below is either exactly a bin or between bins.
_EXACT_GRID = np.arange(0.0, 51.0, 2.0)
# Bin k stores 0.1 * k in float64: bin 3 is 0.30000000000000004, bin 7
# 0.7000000000000001.
_DECIMAL_TENTHS = np.arange(64) * 0.1
# The same tenths stored as float32: float32(0.7) is below 0.7, float32(1.2) above.
_FLOAT32_TENTHS = np.linspace(0, 6.3, 64).astype(np.float32)
# NumPy takes this grid's step as (1.0 + 0.1) - 1.0, so bin 142 is
# 15.200000000000014, four units in the last place from 15.2.
_ARANGE_TENTHS = np.arange(1.0, 100.0, 0.1)
_LOG_GRID = np.geomspace(1.0, 100.0, 50)


def _fourier_input(grid, seed, n_trials=6):
    """Labeled one-sided coefficients on ``grid``, and the same as ``Connectivity``.

    Returns
    -------
    coefficients : xarray.DataArray, shape (n_trials, n_frequencies, 2)
    connectivity : Connectivity
    """
    rng = np.random.default_rng(seed)
    shape = (n_trials, grid.size, 2)
    coefficients = xr.DataArray(
        rng.standard_normal(shape) + 1j * rng.standard_normal(shape),
        dims=("trial", "frequency", "signal"),
        coords={"frequency": grid},
    )
    connectivity = Connectivity(
        coefficients.values[np.newaxis, :, np.newaxis], frequencies=grid, is_one_sided=True
    )
    return coefficients, connectivity


@pytest.fixture(scope="module")
def morlet_transform():
    """A Morlet transform on 4, 8, ..., 20 Hz whose bins differ in validity."""
    return MorletWavelet(
        np.random.default_rng(319).standard_normal((192, 5, 2)),
        64,
        np.array([4.0, 8.0, 12.0, 16.0, 20.0]),
        edge_mode="nan",
    )


@pytest.mark.parametrize(
    ("grid", "bounds", "expected"),
    [
        pytest.param(_EXACT_GRID, (10.0, 20.0), [10, 12, 14, 16, 18, 20], id="on-bins"),
        pytest.param(_EXACT_GRID, (9.0, 21.0), [10, 12, 14, 16, 18, 20], id="off-bins"),
        pytest.param(_EXACT_GRID, (10.0, 10.0), [10], id="single-bin"),
        pytest.param(_EXACT_GRID, (0.0, 50.0), list(range(0, 51, 2)), id="whole-grid"),
        pytest.param(np.array([10.0]), (10.0, 10.0), [10], id="single-bin-grid"),
    ],
)
def test_bounds_on_bins_are_included(grid, bounds, expected):
    mask, selected_bounds = _frequency_interval_mask("frequency_range", bounds, grid)
    np.testing.assert_array_equal(grid[mask], expected)
    assert selected_bounds == bounds


def test_a_bound_a_clear_fraction_of_a_bin_away_does_not_select_it():
    """The allowance is one part in a million of the bin spacing, not a fraction
    of it: 1e-3 of the 2 Hz spacing away, the edge bins are not selected."""
    mask, _ = _frequency_interval_mask(
        "frequency_range", (10.0 + 2e-3, 20.0 - 2e-3), _EXACT_GRID
    )
    np.testing.assert_array_equal(_EXACT_GRID[mask], [12, 14, 16, 18])


@pytest.mark.parametrize(
    ("fraction_of_allowance", "selects_edge_bins"), [(1 - 1e-3, True), (1 + 1e-3, False)]
)
def test_the_allowance_on_a_uniform_grid_is_a_millionth_of_the_spacing(
    fraction_of_allowance, selects_edge_bins
):
    """|f - b| <= 1e-6 * gap(f) + 2 * (ulp(f) + ulp(b)): on the 2 Hz grid the
    10 and 20 Hz bins are 2 Hz from their neighbours, so 2e-6 Hz (plus a few
    units in the last place, negligible here) is on the bound and no more."""
    offset = fraction_of_allowance * 1e-6 * 2.0
    mask, _ = _frequency_interval_mask(
        "frequency_range", (10.0 + offset, 20.0 - offset), _EXACT_GRID
    )
    expected = [10, 12, 14, 16, 18, 20] if selects_edge_bins else [12, 14, 16, 18]
    np.testing.assert_array_equal(_EXACT_GRID[mask], expected)


@pytest.mark.parametrize(("n_last_places", "selected"), [(4, True), (5, False)])
def test_a_single_bin_has_only_the_last_place_allowance(n_last_places, selected):
    """With no neighbour the gap term is zero, leaving 2 * (ulp(f) + ulp(b)):
    four units in the last place at 10 Hz in float64, and no more."""
    bound = 10.0
    for _ in range(n_last_places):
        bound = float(np.nextafter(bound, np.inf))
    grid = np.array([10.0])
    if selected:
        mask, _ = _frequency_interval_mask("frequency_range", (bound, 20.0), grid)
        assert mask.tolist() == [True]
    else:
        with pytest.raises(ValueError, match="contains no frequency bin"):
            _frequency_interval_mask("frequency_range", (bound, 20.0), grid)


@pytest.mark.parametrize(
    ("fraction_of_gap", "expected"), [(0.5e-6, [10.0, 12.0]), (1e-3, [12.0])]
)
def test_two_bins_use_their_distance_as_the_gap(fraction_of_gap, expected):
    grid = np.array([10.0, 12.0])
    offset = fraction_of_gap * 2.0
    mask, _ = _frequency_interval_mask("frequency_range", (10.0 + offset, 12.0 + offset), grid)
    np.testing.assert_array_equal(grid[mask], expected)


_LOG_MEAN_STEP = (_LOG_GRID[-1] - _LOG_GRID[0]) / (_LOG_GRID.size - 1)


@pytest.mark.parametrize(
    ("k", "fraction_of_gap", "selected"),
    [
        # Near the top the local gap is about 3 times the mean step: 0.9e-6 of
        # it is on the bound, though it exceeds 1e-6 of the mean step.
        pytest.param(45, 0.9e-6, True, id="wide-gap-inside"),
        # Near the bottom it is a fourteenth of the mean step: 1.5e-6 of it is
        # off the bound, though it is within 1e-6 of the mean step.
        pytest.param(5, 1.5e-6, False, id="narrow-gap-outside"),
        pytest.param(20, 0.5e-6, True, id="inside"),
        pytest.param(20, 1e-3, False, id="clear-fraction"),
    ],
)
def test_the_gap_is_local_on_a_log_spaced_grid(k, fraction_of_gap, selected):
    """On an irregular grid each bin's allowance follows its own nearest
    neighbour, not the mean step."""
    grid = _LOG_GRID
    gap = min(grid[k + 1] - grid[k], grid[k] - grid[k - 1])
    offset = fraction_of_gap * gap
    if k in (45, 5):  # the mean step would decide these the other way
        assert (offset <= 1e-6 * _LOG_MEAN_STEP) == (not selected)
    upper = float(grid[k] - offset)
    mask, _ = _frequency_interval_mask("frequency_range", (1.0, upper), grid)
    assert np.flatnonzero(mask)[-1] == (k if selected else k - 1)


def _drifting_grids():
    """Grids whose bins drift from their decimal values by more than a few
    units in the last place, as NumPy builds them."""
    return [
        pytest.param(_ARANGE_TENTHS, (10.0, 15.2), id="arange-start-1"),
        pytest.param(np.arange(1000.0, 1101.0, 0.1), (1010.0, 1099.1), id="arange-start-1000"),
        pytest.param(
            np.concatenate([[0.0], np.cumsum(np.full(1000, 0.1))]), (10.0, 99.9), id="cumsum"
        ),
    ]


def test_a_bin_that_drifted_past_a_millionth_of_its_spacing_is_not_named_by_its_decimal():
    """The documented reach of the rule: np.arange(1000.0, 1100.0, 0.001) takes
    its step as 0.0009999999999763531, so bin 42351 is 1042.3509999989985,
    1.0016e-6 of the spacing below 1042.351, and a lower bound of 1042.351
    starts at the next bin."""
    grid = np.arange(1000.0, 1100.0, 0.001)
    drift = (grid[42351] - 1042.351) / 0.001
    assert -1.01e-6 < drift < -1e-6
    mask, _ = _frequency_interval_mask("frequency_range", (1042.351, 1043.0), grid)
    assert np.flatnonzero(mask)[0] == 42352


def _decimal_bins(grid, bounds):
    """Indices of the bins whose decimal value (to 9 places) lies in ``bounds``."""
    decimal = np.round(grid, 9)
    return np.flatnonzero((decimal >= bounds[0]) & (decimal <= bounds[1]))


@pytest.mark.parametrize(("grid", "bounds"), _drifting_grids())
def test_decimal_bounds_name_the_printed_bins_of_drifting_grids(grid, bounds):
    """np.arange(1.0, 100.0, 0.1)'s bin 142 is 15.200000000000014, four units
    in the last place from 15.2; it is still the 15.2 bin."""
    expected = _decimal_bins(grid, bounds)
    exact = np.flatnonzero((grid >= bounds[0]) & (grid <= bounds[1]))
    assert not np.array_equal(exact, expected)  # exact comparison misses an edge bin
    mask, _ = _frequency_interval_mask("frequency_range", bounds, grid)
    np.testing.assert_array_equal(np.flatnonzero(mask), expected)


@pytest.mark.parametrize(
    ("grid", "bounds"),
    [
        *_drifting_grids(),
        pytest.param(_LOG_GRID, (2.0, 50.0), id="log"),
        pytest.param(_FLOAT32_TENTHS, (0.7, 1.2), id="float32"),
    ],
)
def test_wrapper_and_direct_paths_select_the_same_bins_on_irregular_and_drifting_grids(
    grid, bounds
):
    """The wrapper's crop uses the same rule as the direct call, so a float32
    or drifting coordinate keeps the bins its decimal bounds name."""
    coefficients, connectivity = _fourier_input(grid, seed=8)
    cropped = fourier_connectivity(
        coefficients, method="coherence_magnitude", frequency_range=bounds
    )
    mask, _ = _frequency_interval_mask("frequency_range", bounds, grid)
    np.testing.assert_array_equal(cropped.frequency.values, grid[mask])
    if _uniform_grid_spacing(grid) is not None:  # PSI needs a uniform grid
        psi = fourier_connectivity(
            coefficients, method="phase_slope_index", frequency_range=bounds
        )
        np.testing.assert_array_equal(
            psi.values, connectivity.phase_slope_index(frequency_range=bounds)
        )
        assert psi.attrs["selected_frequency_upper"] == float(grid[mask][-1])


_ONE_BIN_RANGES = [
    pytest.param((15.15, 15.2), id="upper-on-drifted-bin"),
    pytest.param((15.2, 15.2), id="low-equals-high"),
]


@pytest.mark.parametrize("bounds", _ONE_BIN_RANGES)
@pytest.mark.parametrize(
    "methods",
    [
        pytest.param("delay", id="delay"),
        pytest.param(["delay", "coherence_magnitude"], id="delay-and-coherence"),
        pytest.param(
            ["coherence_magnitude", "delay", "group_delay"], id="mixed-with-group-delay"
        ),
    ],
)
def test_a_one_bin_range_on_a_drifting_grid_selects_the_same_bin_everywhere(methods, bounds):
    """The selection is made once, on the full grid; a result already cut to
    it is not selected again, where a lone bin would have lost the gap term
    that put it on the bound (bin 142 is 15.200000000000014)."""
    grid = _ARANGE_TENTHS
    coefficients, connectivity = _fourier_input(grid, seed=9, n_trials=8)
    direct = connectivity.delay(frequency_range=bounds)
    assert direct.shape[1] == 1

    result = fourier_connectivity(coefficients, method=methods, frequency_range=bounds)
    delay = result["delay"] if isinstance(result, xr.Dataset) else result
    np.testing.assert_array_equal(delay.frequency.values, [grid[142]])
    np.testing.assert_array_equal(delay.values, direct)
    if isinstance(result, xr.Dataset):
        assert result.coherence_magnitude.sizes["frequency"] == 1


def test_a_one_bin_range_through_multitaper_connectivity_matches_the_direct_call():
    time_series = np.random.default_rng(10).standard_normal((1000, 6, 2))
    settings = {"sampling_frequency": 500, "time_window_duration": 0.5}
    connectivity = Connectivity.from_transform(Multitaper(time_series, **settings))
    for bounds in [(10.0, 10.0), (9.5, 10.0)]:
        result = multitaper_connectivity(
            time_series,
            method=["delay", "coherence_magnitude"],
            frequency_range=bounds,
            **settings,
        )
        assert result.frequency.values.tolist() == [10.0]
        np.testing.assert_array_equal(
            result.delay.values, connectivity.delay(frequency_range=bounds)
        )


def test_band_means_use_the_inclusive_rule_and_integrals_the_exact_bounds():
    """On np.arange(1.0, 100.0, 0.1) the (10, 15.2) band mean includes the
    15.2 bin, while its integral of a flat density is the exact width 5.2."""
    grid = _ARANGE_TENTHS
    bins = _decimal_bins(grid, (10.0, 15.2))
    density = xr.DataArray(
        grid.copy(),
        dims="frequency",
        coords={"frequency": grid},
        name="power",
        attrs={"measure": "power"},
    )
    band = {"b": (10.0, 15.2)}
    mean = frequency_band_reduce(density, band)
    np.testing.assert_allclose(float(mean.squeeze()), grid[bins].mean(), rtol=1e-12)
    flat = density.copy(data=np.ones(grid.size))
    integral = frequency_band_reduce(flat, band, frequency_reduction="integral")
    np.testing.assert_allclose(float(integral.squeeze()), 15.2 - 10.0, rtol=1e-12)


@pytest.mark.parametrize(
    ("grid", "bounds", "exact_drops_an_edge_bin"),
    [
        pytest.param(_DECIMAL_TENTHS, (0.3, 0.7), True, id="decimal-0.3-0.7"),
        pytest.param(_DECIMAL_TENTHS, (0.7, 1.2), True, id="decimal-0.7-1.2"),
        pytest.param(_FLOAT32_TENTHS, (0.3, 0.7), False, id="float32-0.3-0.7"),
        pytest.param(_FLOAT32_TENTHS, (0.7, 1.2), True, id="float32-0.7-1.2"),
    ],
)
def test_a_bin_stored_with_rounding_of_its_decimal_bound_is_included(
    grid, bounds, exact_drops_an_edge_bin
):
    """Bin ``k`` stores ``0.1 * k``, so the bounds name bins 3-7 or 7-12; an
    exact float64 ``>=``/``<=`` comparison would drop an edge bin where noted."""
    expected = np.arange(round(bounds[0] * 10), round(bounds[1] * 10) + 1)
    exact = np.flatnonzero(
        (grid.astype(float) >= bounds[0]) & (grid.astype(float) <= bounds[1])
    )
    assert (not np.array_equal(exact, expected)) == exact_drops_an_edge_bin
    mask, _ = _frequency_interval_mask("frequency_range", bounds, grid)
    np.testing.assert_array_equal(np.flatnonzero(mask), expected)


def _released_band_mask(frequencies_of_interest, frequencies):
    """The bin mask of v2.0.1's ``_bandpass``, verbatim: the band as the caller
    gave it compared against the grid in its stored dtype, so a Python float
    bound meets a float32 grid in float32 and an ndarray bound in float64."""
    return (frequencies_of_interest[0] < frequencies) & (
        frequencies < frequencies_of_interest[1]
    )


_BAND_FORMS = {
    "tuple": tuple,
    "list": list,
    "ndarray": lambda band: np.asarray(band, dtype=float),
}


@pytest.mark.parametrize(
    ("bounds", "requested"),
    [
        pytest.param((10.0, np.inf), {"requested_frequency_lower": 10.0}, id="above"),
        pytest.param((-np.inf, 50.0), {"requested_frequency_upper": 50.0}, id="below"),
        pytest.param((-np.inf, np.inf), {}, id="unbounded"),
    ],
)
def test_an_open_deprecated_band_records_only_its_finite_requested_bounds(
    lagged_time_series, tmp_path, bounds, requested
):
    """An infinite bound requests no bound on that side, so it records no
    requested_frequency_* there; every attribute stays a finite float and
    survives a NetCDF round trip."""
    with pytest.warns(DeprecationWarning, match="frequencies_of_interest"):
        result = connectivity_to_xarray(
            Multitaper(lagged_time_series, **_LAGGED_TRANSFORM),
            method="phase_slope_index",
            frequencies_of_interest=bounds,
        )
    names = [
        "requested_frequency_lower",
        "requested_frequency_upper",
        "selected_frequency_lower",
        "selected_frequency_upper",
        "frequency_interval_closed",
    ]
    recorded = {name: result.attrs[name] for name in names if name in result.attrs}
    lowest = 12.0 if np.isfinite(bounds[0]) else 0.0
    highest = 48.0 if np.isfinite(bounds[1]) else 250.0
    assert recorded == {
        **requested,
        "selected_frequency_lower": lowest,
        "selected_frequency_upper": highest,
        "frequency_interval_closed": "neither",
    }
    result.to_netcdf(tmp_path / "psi.nc", engine="scipy")
    with xr.open_dataarray(tmp_path / "psi.nc", engine="scipy") as reloaded:
        assert {name: reloaded.attrs[name] for name in recorded} == recorded


@pytest.mark.parametrize(
    "grid",
    [
        pytest.param(_DECIMAL_TENTHS, id="decimal-rounded-float64"),
        pytest.param(_FLOAT32_TENTHS, id="float32"),
    ],
)
@pytest.mark.parametrize("form", list(_BAND_FORMS))
@pytest.mark.parametrize(
    "bounds",
    [
        (0.3, 0.7),
        (0.7, 1.2),
        (1.0, 5.0),
        (1, 5),
        (0.3, np.inf),
        (-np.inf, 0.7),
        (-np.inf, np.inf),
    ],
)
def test_neither_is_the_released_exclusive_comparison(grid, form, bounds):
    """The deprecated exclusive selection is v2.0.1's, bit for bit: on a
    float32 grid a Python-number band is compared in float32, so float32(0.3)
    excludes the 0.3 bin, while an ndarray band is compared in float64. An
    open-ended band, which v2.0.1 accepted, selects what v2.0.1 selected."""
    band = _BAND_FORMS[form](bounds)
    mask, _ = _frequencies_of_interest_mask(band, grid)
    np.testing.assert_array_equal(mask, _released_band_mask(band, grid))


@pytest.mark.parametrize(
    ("bounds", "expected"),
    [
        pytest.param((10.0, 20.0), [12, 14, 16, 18], id="on-bins"),
        pytest.param((9.0, 21.0), [10, 12, 14, 16, 18, 20], id="off-bins"),
    ],
)
def test_neither_excludes_bins_on_the_bounds(bounds, expected):
    mask, _ = _frequencies_of_interest_mask(bounds, _EXACT_GRID)
    np.testing.assert_array_equal(_EXACT_GRID[mask], expected)


@pytest.mark.parametrize(
    ("bounds", "expected"),
    [
        pytest.param((10.0, np.inf), list(range(12, 51, 2)), id="above"),
        pytest.param((-np.inf, 10.0), [0, 2, 4, 6, 8], id="below"),
        pytest.param((-np.inf, np.inf), list(range(0, 51, 2)), id="unbounded"),
    ],
)
def test_the_frozen_validation_accepts_open_bands_the_canonical_rule_rejects(bounds, expected):
    """Infinite bounds are valid for the deprecated keyword, as in v2.0.1,
    while frequency_range keeps requiring two finite values."""
    mask, recorded = _frequencies_of_interest_mask(bounds, _EXACT_GRID)
    np.testing.assert_array_equal(_EXACT_GRID[mask], expected)
    assert recorded == bounds
    with pytest.raises(ValueError, match="frequency_range must be two finite values"):
        _frequency_interval_mask("frequency_range", bounds, _EXACT_GRID)


def test_the_released_comparison_depends_on_the_band_form_on_float32_grids():
    """Guard for the test above: its float32 cases exercise both promotions."""
    as_tuple = _released_band_mask((0.3, 0.7), _FLOAT32_TENTHS)
    as_array = _released_band_mask(np.array([0.3, 0.7]), _FLOAT32_TENTHS)
    assert np.flatnonzero(as_tuple).tolist() == [4, 5, 6]
    assert np.flatnonzero(as_array).tolist() == [3, 4, 5, 6, 7]


@pytest.mark.parametrize(
    "bounds",
    [
        pytest.param((10.1, 20.3), id="closed"),
        pytest.param((10.1, np.inf), id="above"),
        pytest.param((-np.inf, 20.3), id="below"),
    ],
)
@pytest.mark.parametrize("form", list(_BAND_FORMS))
@pytest.mark.parametrize("positional", [False, True], ids=["keyword", "positional"])
@pytest.mark.parametrize("measure", ["delay", "phase_slope_index"])
def test_deprecated_calls_on_a_float32_grid_consume_the_released_bins(
    form, positional, measure, bounds
):
    """End to end on a float32 0.1 Hz grid: every v2.0.1 call form, open-ended
    bands included, consumes the bins v2.0.1 consumed, and the warning's
    frequency_range, finite even for an open band, keeps them."""
    rng = np.random.default_rng(1)
    shape = (1, 4, 3, 2000, 2)
    grid = np.fft.fftfreq(2000, d=1 / 200).astype(np.float32)
    connectivity = Connectivity(
        rng.standard_normal(shape) + 1j * rng.standard_normal(shape),
        expectation_type="trials_tapers",
        frequencies=grid,
    )
    band = _BAND_FORMS[form](bounds)
    released = _released_band_mask(band, connectivity.frequencies)
    args, kwargs = ((band,), {}) if positional else ((), {"frequencies_of_interest": band})
    with pytest.warns(DeprecationWarning, match="frequencies_of_interest") as record:
        deprecated = getattr(connectivity, measure)(*args, **kwargs)
    selected = connectivity.frequencies[released]
    assert np.isfinite(selected[[0, -1]]).all()
    suggestion = f"frequency_range=({float(selected[0])!r}, {float(selected[-1])!r})"
    assert suggestion in str(record[0].message)
    migrated = getattr(connectivity, measure)(
        frequency_range=(float(selected[0]), float(selected[-1]))
    )
    np.testing.assert_array_equal(deprecated, migrated)
    if measure == "delay":
        assert deprecated.shape[1] == released.sum()


def test_float32_bounds_on_a_float64_grid_use_float32_rounding():
    """Bounds read off a float32 coordinate still select the bins they name."""
    bounds = tuple(_FLOAT32_TENTHS[[7, 12]])
    mask, _ = _frequency_interval_mask("frequency_range", bounds, _DECIMAL_TENTHS)
    np.testing.assert_array_equal(np.flatnonzero(mask), np.arange(7, 13))


@pytest.mark.parametrize(
    ("bounds", "message"),
    [
        ([10.0], "must be two finite values"),
        ([10.0, 20.0, 30.0], "must be two finite values"),
        ([20.0, 10.0], r"low <= high"),
        ([np.nan, 10.0], "must be two finite values"),
        ([0.0, np.inf], "must be two finite values"),
        ([12.6, 13.5], "contains no frequency bin"),
        ([60.0, 70.0], "contains no frequency bin"),
    ],
)
def test_invalid_or_empty_intervals_raise_naming_the_argument(bounds, message):
    with pytest.raises(ValueError, match=message) as error:
        _frequency_interval_mask("frequency_range", bounds, _EXACT_GRID)
    assert str(error.value).startswith("frequency_range")


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


def test_adjacent_bands_share_their_endpoint_bin_without_double_counting_power():
    """(10, 20) and (20, 30) both average the 20 Hz bin, while their integrals
    split its cell, so they add up to the integral over (10, 30)."""
    time_series = np.random.default_rng(6).standard_normal((1000, 4, 2))
    settings = {"sampling_frequency": 500, "time_window_duration": 0.5, "method": "power"}
    power = multitaper_connectivity(time_series, **settings)
    bands = {"low": (10.0, 20.0), "high": (20.0, 30.0), "union": (10.0, 30.0)}

    means = multitaper_connectivity(time_series, frequency_bands=bands, **settings)
    for name, (low, high) in bands.items():
        np.testing.assert_allclose(
            means.sel(band=name).values,
            power.sel(frequency=slice(low, high)).mean("frequency").values,
            rtol=1e-12,
        )
    assert power.sel(frequency=slice(10.0, 20.0)).frequency.values[-1] == 20.0
    assert power.sel(frequency=slice(20.0, 30.0)).frequency.values[0] == 20.0

    integrals = multitaper_connectivity(
        time_series, frequency_bands=bands, frequency_reduction="integral", **settings
    )
    np.testing.assert_allclose(
        (integrals.sel(band="low") + integrals.sel(band="high")).values,
        integrals.sel(band="union").values,
        rtol=1e-12,
    )
    # Bins are 2 Hz apart, so the union is the 11 bins' density times 2 Hz,
    # less the outer halves of the 10 and 30 Hz cells.
    union = power.sel(frequency=slice(10.0, 30.0))
    expected = 2.0 * union.sum("frequency") - (
        union.isel(frequency=0) + union.isel(frequency=-1)
    )
    np.testing.assert_allclose(integrals.sel(band="union").values, expected.values, rtol=1e-12)


# --- Measures that consume a frequency interval --------------------------------

_SELECTING_MEASURES = ("delay", "group_delay", "phase_slope_index")


def test_the_registry_names_the_measures_that_consume_frequency_range():
    """The registry property agrees with the measures' signatures, so the
    wrappers route frequency_range to exactly the measures that accept it."""
    assert _FREQUENCY_RANGE_MEASURES == _SELECTING_MEASURES
    for name, spec in _MEASURE_SPECS.items():
        parameters = inspect.signature(getattr(Connectivity, name)).parameters
        assert spec.consumes_frequency_range == ("frequency_range" in parameters), name


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


@pytest.mark.parametrize("measure", _SELECTING_MEASURES)
def test_frequency_range_consumes_the_bins_on_its_bounds(lagged_connectivity, measure):
    """frequency_range=(10, 50) uses the 10 and 50 Hz bins: it equals
    (9, 51), whose bounds lie between bins and enclose them, and differs from
    (11, 49), which leaves them out."""
    inclusive = _call(lagged_connectivity, measure, frequency_range=(10.0, 50.0))
    np.testing.assert_array_equal(
        inclusive, _call(lagged_connectivity, measure, frequency_range=(9.0, 51.0))
    )
    without_edges = _call(lagged_connectivity, measure, frequency_range=(11.0, 49.0))
    assert inclusive.shape != without_edges.shape or not np.allclose(
        inclusive, without_edges, equal_nan=True
    )
    if measure == "delay":  # 10, 12, ..., 50 Hz
        assert inclusive.shape[-4] == 21


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
    with pytest.warns(DeprecationWarning, match="frequencies_of_interest"):
        positional = _call(lagged_connectivity, measure, (10.0, 50.0))
    np.testing.assert_array_equal(
        positional, _call(lagged_connectivity, measure, frequency_range=(12.0, 48.0))
    )


@pytest.mark.parametrize("measure", _SELECTING_MEASURES)
def test_both_spellings_are_rejected(lagged_connectivity, measure):
    with pytest.raises(ValueError, match="frequency_range and the deprecated"):
        getattr(lagged_connectivity, measure)(
            frequency_range=(10.0, 50.0), frequencies_of_interest=(10.0, 50.0)
        )


@pytest.mark.parametrize(
    ("band", "message"),
    [
        pytest.param((5.0, 5.0), "low < high", id="equal-bounds"),
        pytest.param((20.0, 10.0), "low < high", id="reversed"),
        pytest.param((np.inf, np.inf), "low < high", id="both-infinite"),
        pytest.param((10.0, 12.0), "contains no frequency bin", id="between-adjacent-bins"),
        pytest.param((300.0, 400.0), "contains no frequency bin", id="beyond-nyquist"),
        pytest.param((300.0, np.inf), "contains no frequency bin", id="open-beyond-nyquist"),
        pytest.param((10.0,), "must be two values", id="one-value"),
        pytest.param((10.0, 50.0, 99.0), "must be two values", id="three-values"),
        pytest.param(np.array([10.0, 50.0, 99.0]), "must be two values", id="three-array"),
        pytest.param(10.0, "must be two values", id="scalar"),
        pytest.param((np.nan, 50.0), "not NaN", id="nan-lower"),
        pytest.param((10.0, np.nan), "not NaN", id="nan-upper"),
    ],
)
@pytest.mark.parametrize("measure", _SELECTING_MEASURES)
def test_deprecated_bands_without_a_bin_inside_raise_where_2x_returned_empty_results(
    lagged_connectivity, measure, band, message
):
    """As the CHANGELOG states: 2.x returned all-NaN or empty results for
    these bands; the deprecated keyword now validates them."""
    with pytest.raises(ValueError, match=message):
        getattr(lagged_connectivity, measure)(frequencies_of_interest=band)


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
def test_morlet_validity_follows_the_consumed_bins(
    morlet_transform, measure, kwargs, consumed_bins
):
    """valid_time of a frequency-reduced measure, and valid_time_frequency of
    delay, describe exactly the bins the estimator used."""
    transform = morlet_transform
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


# --- The wrappers' frequency_range ---------------------------------------------


def _main_variable(result, measure):
    """The measure's own variable (group delay's ``group_delay``)."""
    return result[measure] if isinstance(result, xr.Dataset) else result


@pytest.mark.parametrize("wrapper", ["multitaper_connectivity", "fourier_connectivity"])
@pytest.mark.parametrize("measure", _SELECTING_MEASURES)
def test_wrapper_frequency_range_selects_the_estimators_bins(
    lagged_time_series, lagged_connectivity, measure, wrapper
):
    """Both wrappers pass frequency_range to the estimator, so it consumes the
    same bins as the direct call instead of the wrapper refusing the range."""
    if wrapper == "multitaper_connectivity":
        result = multitaper_connectivity(
            lagged_time_series,
            method=measure,
            frequency_range=(10.0, 50.0),
            **_LAGGED_TRANSFORM,
        )
    else:
        transform = Multitaper(lagged_time_series, **_LAGGED_TRANSFORM)
        result = fourier_connectivity(
            transform.fft(),
            frequencies=transform.frequencies,
            method=measure,
            frequency_range=(10.0, 50.0),
        )
    direct = getattr(lagged_connectivity, measure)(frequency_range=(10.0, 50.0))
    direct = direct[0] if measure == "group_delay" else direct
    np.testing.assert_array_equal(_main_variable(result, measure).values, direct)
    if measure == "delay":
        frequencies = lagged_connectivity.frequencies
        np.testing.assert_array_equal(
            result.frequency.values, frequencies[(frequencies >= 10) & (frequencies <= 50)]
        )
    variable = _main_variable(result, measure)
    assert variable.attrs["frequency_interval_closed"] == "both"
    assert json.loads(variable.attrs["measure_kwargs_json"]) == {
        "frequency_range": [10.0, 50.0]
    }


def test_the_wrapper_records_its_frequency_range_on_every_variable_it_applied_to(
    lagged_time_series,
):
    """Cropped and consuming variables alike carry one interval record, so
    filtering variables by it finds the interval measures too. It stays on
    the variables, not the Dataset, and survives decimation and band
    reduction, which record themselves."""
    interval = {
        "requested_frequency_lower": 10.0,
        "requested_frequency_upper": 50.0,
        "selected_frequency_lower": 10.0,
        "selected_frequency_upper": 50.0,
        "frequency_interval_closed": "both",
    }
    methods = ["phase_slope_index", "coherence_magnitude", "delay", "group_delay"]
    result = multitaper_connectivity(
        lagged_time_series, method=methods, frequency_range=(10.0, 50.0), **_LAGGED_TRANSFORM
    )
    assert set(result.data_vars) == {
        "phase_slope_index",
        "coherence_magnitude",
        "delay",
        "group_delay",
        "group_delay_slope",
        "group_delay_r_value",
    }
    for name, variable in result.data_vars.items():
        assert {key: variable.attrs.get(key) for key in interval} == interval, name
        assert "frequency_range_json" not in variable.attrs
    assert not set(interval) & set(result.attrs)

    decimated = multitaper_connectivity(
        lagged_time_series,
        method=methods,
        frequency_range=(10.0, 50.0),
        frequency_decimation=3,
        **_LAGGED_TRANSFORM,
    )
    # The coordinate ends at 46 Hz; the record still names the selected interval.
    assert decimated.frequency.values[-1] == 46.0
    for name, variable in decimated.data_vars.items():
        assert {key: variable.attrs.get(key) for key in interval} == interval, name
    assert decimated.coherence_magnitude.attrs["frequency_decimation"] == 3
    assert "frequency_decimation" not in decimated.phase_slope_index.attrs

    banded = multitaper_connectivity(
        lagged_time_series,
        method="coherence_magnitude",
        frequency_range=(10.0, 50.0),
        frequency_bands={"low": (10.0, 20.0)},
        **_LAGGED_TRANSFORM,
    )
    assert {key: banded.attrs.get(key) for key in interval} == interval
    assert json.loads(banded.attrs["frequency_bands_json"]) == {"low": [10.0, 20.0]}


def test_one_frequency_range_crops_and_selects_the_same_bins_in_a_dataset(
    lagged_time_series, lagged_connectivity
):
    """In one Dataset the crop of a frequency-resolved measure, the bins delay
    returns, and the bins phase_slope_index sums are the same."""
    result = multitaper_connectivity(
        lagged_time_series,
        method=["coherence_magnitude", "delay", "phase_slope_index"],
        frequency_range=(10.0, 50.0),
        **_LAGGED_TRANSFORM,
    )
    frequencies = lagged_connectivity.frequencies
    expected = frequencies[(frequencies >= 10) & (frequencies <= 50)]
    np.testing.assert_array_equal(result.frequency.values, expected)
    assert result.coherence_magnitude.sizes["frequency"] == expected.size
    assert result.delay.sizes["frequency"] == expected.size
    np.testing.assert_array_equal(
        result.phase_slope_index.values,
        lagged_connectivity.phase_slope_index(frequency_range=(10.0, 50.0)),
    )
    assert result.phase_slope_index.attrs["selected_frequency_lower"] == 10.0
    assert result.coherence_magnitude.attrs["selected_frequency_lower"] == 10.0
    # The crop is recorded as an interval; only the estimators got the keyword.
    assert json.loads(result.coherence_magnitude.attrs["measure_kwargs_json"]) == {}


@pytest.mark.parametrize(
    "method", ["phase_slope_index", "group_delay", ["phase_slope_index", "group_delay"]]
)
@pytest.mark.parametrize(
    ("operation", "named"),
    [
        ({"frequency_bands": {"theta": (4.0, 8.0)}}, "frequency_bands cannot"),
        ({"frequency_decimation": 2}, "frequency_decimation cannot"),
        (
            {"frequency_bands": {"theta": (4.0, 8.0)}, "frequency_decimation": 2},
            "frequency_bands and frequency_decimation cannot",
        ),
    ],
)
def test_frequency_operations_on_only_frequency_reduced_measures_raise(
    lagged_time_series, method, operation, named
):
    """frequency_bands and frequency_decimation act on a frequency axis, which
    phase_slope_index and group_delay have already reduced: a request with no
    other measure would return them unchanged, so it raises instead, naming
    what was asked and both remedies. No frequency_range is given, so the
    message must not claim the measures reduced one."""
    with pytest.raises(ValueError, match="already reduce") as error:
        multitaper_connectivity(
            lagged_time_series, method=method, **operation, **_LAGGED_TRANSFORM
        )
    message = str(error.value)
    assert message.startswith(named)
    assert "once per band" in message
    assert "frequency_resolution" in message
    assert "over frequency_range" not in message


@pytest.mark.parametrize("measure", ["phase_slope_index", "group_delay"])
@pytest.mark.parametrize(
    "operation",
    [{"frequency_bands": {"theta": (10.0, 20.0)}}, {"frequency_decimation": 2}],
)
def test_frequency_reduced_variables_of_a_dataset_are_not_reduced_again(
    lagged_time_series, measure, operation
):
    """Bands and decimation reduce the frequency-resolved variables of a
    Dataset; the frequency-reduced measure keeps its value over frequency_range
    and records none of them."""
    kwargs = {"frequency_range": (10.0, 50.0), **_LAGGED_TRANSFORM}
    result = multitaper_connectivity(
        lagged_time_series, method=["coherence_magnitude", measure], **operation, **kwargs
    )
    alone = multitaper_connectivity(lagged_time_series, method=[measure], **kwargs)
    if "frequency_bands" in operation:
        assert result.coherence_magnitude.dims == ("time", "band", "source", "target")
    else:  # every other bin of 10, 12, ..., 50 Hz
        np.testing.assert_array_equal(result.frequency.values, np.arange(10.0, 51.0, 4.0))
    xr.testing.assert_identical(result[measure], alone[measure])
    for key in ("frequency_bands_json", "frequency_decimation"):
        assert key not in result[measure].attrs


def test_delay_is_frequency_resolved_so_bands_aggregate_it(lagged_time_series):
    result = multitaper_connectivity(
        lagged_time_series,
        method="delay",
        frequency_range=(10.0, 50.0),
        frequency_bands={"low": (10.0, 20.0)},
        **_LAGGED_TRANSFORM,
    )
    unreduced = multitaper_connectivity(
        lagged_time_series, method="delay", frequency_range=(10.0, 50.0), **_LAGGED_TRANSFORM
    )
    np.testing.assert_array_equal(
        result.sel(band="low").values,
        unreduced.sel(frequency=slice(10.0, 20.0)).mean("frequency", skipna=False).values,
    )


def test_frequency_range_in_connectivity_kwargs_points_to_the_wrapper_argument(
    lagged_time_series,
):
    with pytest.raises(ValueError, match="pass frequency_range=") as error:
        multitaper_connectivity(
            lagged_time_series,
            method="phase_slope_index",
            connectivity_kwargs={"frequency_range": (10.0, 50.0)},
            **_LAGGED_TRANSFORM,
        )
    assert "connectivity_kwargs" in str(error.value)


def test_deprecated_keyword_through_the_wrapper_warns_at_the_caller(
    lagged_time_series, lagged_connectivity
):
    with pytest.warns(DeprecationWarning, match="frequencies_of_interest") as record:
        result = multitaper_connectivity(
            lagged_time_series,
            method="phase_slope_index",
            connectivity_kwargs={"frequencies_of_interest": (10.0, 50.0)},
            **_LAGGED_TRANSFORM,
        )
    assert {w.filename for w in record if issubclass(w.category, DeprecationWarning)} == {
        __file__
    }
    assert "multitaper_connectivity and fourier_connectivity take frequency_range" in str(
        record[0].message
    )
    np.testing.assert_array_equal(
        result.values, lagged_connectivity.phase_slope_index(frequency_range=(12.0, 48.0))
    )
    assert result.attrs["frequency_interval_closed"] == "neither"
    with pytest.raises(ValueError, match="frequency_range and the deprecated"):
        multitaper_connectivity(
            lagged_time_series,
            method="phase_slope_index",
            connectivity_kwargs={"frequencies_of_interest": (10.0, 50.0)},
            frequency_range=(10.0, 50.0),
            **_LAGGED_TRANSFORM,
        )


def test_wilson_factorization_sees_the_full_two_sided_spectrum(monkeypatch):
    """frequency_range crops a Wilson-factorized measure's result: the
    factorization's input, and so every retained value, is unchanged."""
    import spectral_connectivity._granger as granger

    factorized = []
    original = granger.minimum_phase_decomposition

    def recording_decomposition(cross_spectral_matrix, *args, **kwargs):
        factorized.append(to_host(cross_spectral_matrix).copy())
        return original(cross_spectral_matrix, *args, **kwargs)

    monkeypatch.setattr(granger, "minimum_phase_decomposition", recording_decomposition)
    time_series = np.random.default_rng(5).standard_normal((512, 10, 2))
    kwargs = {
        "sampling_frequency": 128,
        "time_halfbandwidth_product": 3,
        "method": "pairwise_spectral_granger_prediction",
    }
    full = multitaper_connectivity(time_series, **kwargs)
    cropped = multitaper_connectivity(time_series, frequency_range=(10.0, 30.0), **kwargs)

    assert len(factorized) == 2
    np.testing.assert_array_equal(factorized[1], factorized[0])
    # The full two-sided FFT grid, not the cropped bins.
    assert factorized[0].shape[-3] == 2 * (full.sizes["frequency"] - 1)
    np.testing.assert_array_equal(cropped.values, full.sel(frequency=slice(10.0, 30.0)).values)
    assert cropped.frequency.values[[0, -1]].tolist() == [10.0, 30.0]


def test_selected_bounds_describe_the_interval_before_subsampling(lagged_time_series):
    """With frequency_resolution=2, PSI on the 0.5 Hz grid sums every fourth
    bin of 5-50 Hz, the last at 49 Hz: its value is that of (5, 49), yet the
    record names the requested interval, up to 50 Hz."""
    settings = {"sampling_frequency": 500, "method": "phase_slope_index"}
    resolution = {"frequency_resolution": 2.0}
    psi = multitaper_connectivity(
        lagged_time_series,
        frequency_range=(5.0, 50.0),
        connectivity_kwargs=resolution,
        **settings,
    )
    up_to_last_summed = multitaper_connectivity(
        lagged_time_series,
        frequency_range=(5.0, 49.0),
        connectivity_kwargs=resolution,
        **settings,
    )
    np.testing.assert_array_equal(psi.values, up_to_last_summed.values)
    assert (psi.attrs["selected_frequency_lower"], psi.attrs["selected_frequency_upper"]) == (
        5.0,
        50.0,
    )
    assert up_to_last_summed.attrs["selected_frequency_upper"] == 49.0


def test_morlet_validity_covers_bins_that_frequency_resolution_skips():
    """valid_time covers every bin of the interval, so it can be stricter than
    the bins PSI sums: here the 12 Hz wavelet (20 cycles) is far longer than
    the others, and PSI with every second bin (10, 14, 18 Hz) skips it."""
    transform = MorletWavelet(
        np.random.default_rng(320).standard_normal((256, 5, 2)),
        64,
        np.array([10.0, 12.0, 14.0, 16.0, 18.0, 20.0]),
        n_cycles=np.array([3.0, 20.0, 3.0, 3.0, 3.0, 3.0]),
        edge_mode="nan",
    )
    validity = to_host(transform.valid_time_frequency)
    summed = validity[:, [0, 2, 4]].all(axis=1)
    every_bin = validity.all(axis=1)
    # Some times are valid for every summed bin but not for the skipped 12 Hz bin.
    assert (summed & ~every_bin).any()
    psi = connectivity_to_xarray(
        transform,
        method="phase_slope_index",
        frequency_range=(10.0, 20.0),
        frequency_resolution=4.0,  # a step of two 2 Hz bins
    )
    np.testing.assert_array_equal(psi.valid_time, every_bin)
