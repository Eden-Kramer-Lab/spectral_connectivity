"""Discovery API for the connectivity measures the wrapper can compute."""

import warnings

import numpy as np
import pytest
import xarray as xr

from spectral_connectivity import (
    Connectivity,
    Multitaper,
    list_measures,
    multitaper_connectivity,
)
from spectral_connectivity.simulate import simulate_MVAR
from spectral_connectivity.wrapper import (
    _MEASURE_SPECS,
    DEFAULT_METHODS,
    MeasureInfo,
)


def test_lists_every_registered_measure():
    """Every wrapper-supported measure is returned exactly once."""
    names = [measure.name for measure in list_measures()]
    assert names == list(_MEASURE_SPECS)


def test_every_public_connectivity_measure_is_registered():
    """A new Connectivity measure must declare its wrapper contract.

    An unregistered method would be invisible to ``list_measures()`` and, when
    requested by name, would fall back to the undirected pairwise contract
    (no source/target transpose, no two-sided-spectrum requirement).
    """
    import inspect

    non_measure_methods = {"jackknife", "minimum_phase_reconstruction_error"}
    public_methods = {
        name
        for name in dir(Connectivity)
        if not name.startswith("_")
        and inspect.isfunction(inspect.getattr_static(Connectivity, name))
    }
    assert public_methods - non_measure_methods == set(_MEASURE_SPECS)


def test_returns_measureinfo_records_with_populated_fields():
    """Records expose name, category, description, and capability flags."""
    coherence = next(
        measure for measure in list_measures() if measure.name == "coherence_magnitude"
    )
    assert isinstance(coherence, MeasureInfo)
    assert coherence.category == "pairwise"
    assert coherence.is_default is True
    assert coherence.is_directed is False
    assert coherence.requires_two_sided is False
    assert coherence.description == ("Return the magnitude squared of the complex coherency.")


def test_description_is_the_method_docstring_summary():
    """Descriptions are sourced from the method docstring, not duplicated."""
    for measure in list_measures():
        docstring = getattr(Connectivity, measure.name).__doc__ or ""
        first_line = docstring.strip().splitlines()[0].strip()
        assert measure.description == first_line
        assert measure.description  # never empty


def test_default_only_matches_default_methods():
    """The default-only view equals the exported DEFAULT_METHODS set."""
    names = [measure.name for measure in list_measures(default_only=True)]
    assert names == list(DEFAULT_METHODS)


def test_directed_filter_selects_directed_measures():
    """directed=True yields only directed measures; False only undirected."""
    directed = list_measures(directed=True)
    assert directed  # non-empty
    assert all(measure.is_directed for measure in directed)
    assert "pairwise_spectral_granger_prediction" in {measure.name for measure in directed}

    undirected = list_measures(directed=False)
    assert all(not measure.is_directed for measure in undirected)
    assert "coherence_magnitude" in {measure.name for measure in undirected}


def test_directionality_is_independent_of_spectrum_requirement():
    measures = {measure.name: measure for measure in list_measures()}
    for name in (
        "directed_phase_lag_index",
        "phase_slope_index",
        "delay",
        "group_delay",
    ):
        assert measures[name].is_directed
        assert not measures[name].requires_two_sided

    granger = measures["pairwise_spectral_granger_prediction"]
    assert granger.is_directed
    assert granger.requires_two_sided


def test_category_filter_selects_matching_output_kind():
    """category filters to measures with that output kind."""
    power = list_measures(category="power")
    assert [measure.name for measure in power] == ["power"]


def test_invalid_category_lists_valid_categories():
    """An unknown category raises and names the valid categories."""
    with pytest.raises(ValueError, match="Unknown category 'not_a_category'") as excinfo:
        list_measures(category="not_a_category")
    message = str(excinfo.value)
    assert "not_a_category" in message
    assert "pairwise" in message


_GROUPS = ["a", "a", "b", "b"]
_MEASURE_KWARGS = {
    "canonical_coherence": {"group_labels": _GROUPS},
    "canonical_coherency": {"group_labels": _GROUPS},
    "maximized_imaginary_coherency": {"group_labels": _GROUPS},
    "maximized_imaginary_coherency_components": {"group_labels": _GROUPS},
    "multivariate_interaction_measure": {"group_labels": _GROUPS},
    "blockwise_spectral_granger_prediction": {"group_labels": _GROUPS},
    "subset_pairwise_spectral_granger_prediction": {"pairs": [(0, 1)]},
}


@pytest.fixture(scope="module")
def coupled_time_series():
    """A coupled VAR(1) chain 0 -> 1 -> 2 -> 3, shape (1024, 4 trials, 4).

    Coupled data (rather than white noise) gives every measure finite values,
    including the delay measures, which are NaN where coherence is not
    significant.
    """
    coefficients = np.zeros((1, 4, 4))
    coefficients[0] = [
        [0.8, 0.0, 0.0, 0.0],
        [0.5, 0.3, 0.0, 0.0],
        [0.0, 0.4, 0.5, 0.0],
        [0.0, 0.0, 0.3, 0.2],
    ]
    return simulate_MVAR(coefficients, n_time_samples=1024, n_trials=4, random_state=1)


def _every_measure(time_series):
    """Every measure computed on ``time_series`` by the wrapper, keyed by name."""
    results = {}
    with warnings.catch_warnings():
        # Some measures warn about degenerate bins; irrelevant to the metadata.
        warnings.simplefilter("ignore", UserWarning)
        for measure in list_measures():
            results[measure.name] = multitaper_connectivity(
                time_series,
                sampling_frequency=500,
                time_window_duration=0.5,
                method=measure.name,
                connectivity_kwargs=_MEASURE_KWARGS.get(measure.name, {}),
            )
    return results


@pytest.fixture(scope="module")
def coupled_results(coupled_time_series):
    return _every_measure(coupled_time_series)


@pytest.fixture(scope="module")
def coupled_results_doubled(coupled_time_series):
    """The same measures with the input amplitude doubled, to check ``units``."""
    return _every_measure(2.0 * coupled_time_series)


def _main_variable(result, name):
    """The measure's own variable; rich results carry filters etc. alongside."""
    return result[name] if isinstance(result, xr.Dataset) else result


# The output contract each category promises: whether the wrapper returns a
# bare DataArray or a Dataset with companion variables, and the dimensions of
# the measure's own variable.
_CATEGORY_STRUCTURE = {
    "pairwise": (xr.DataArray, ("time", "frequency", "source", "target")),
    "power": (xr.DataArray, ("time", "frequency", "source")),
    "group_pairwise": (xr.DataArray, ("time", "frequency", "source_group", "target_group")),
    "multivariate_components": (xr.Dataset, ("time", "frequency", "connection", "component")),
    "delay": (xr.DataArray, ("time", "frequency", "candidate", "source", "target")),
    "global": (xr.Dataset, ("time", "frequency", "component")),
    "group_delay": (xr.Dataset, ("time", "source", "target")),
    "phase_slope": (xr.DataArray, ("time", "source", "target")),
}


@pytest.mark.parametrize("measure", list_measures(), ids=lambda measure: measure.name)
def test_metadata_describes_the_computed_output(
    coupled_results, coupled_results_doubled, measure
):
    """category, dims, dtype, value range, units, and labels match what the
    wrapper returns."""
    result = coupled_results[measure.name]
    variable = _main_variable(result, measure.name)

    result_type, category_dims = _CATEGORY_STRUCTURE[measure.category]
    assert isinstance(result, result_type)
    assert variable.dims == category_dims
    assert variable.dims == measure.dims
    assert np.iscomplexobj(variable) == measure.is_complex
    values = np.abs(variable.values) if measure.is_complex else variable.values
    finite = values[np.isfinite(values)]
    assert finite.size > 0
    lower, upper = measure.value_range
    assert finite.min() >= lower - 1e-12
    assert finite.max() <= upper + 1e-12
    # A declared negative bound must be reachable, so a range that is too loose
    # for a non-negative measure (e.g. coherence as [-1, 1]) is caught too.
    if lower < 0:
        assert finite.min() < 0
    assert variable.attrs["long_name"] == measure.long_name
    assert measure.interpretation

    # Units name the input dependence: a density scales with the squared
    # amplitude, while a dimensionless score, an angle, or a time does not.
    assert measure.units in {"(input units)^2/Hz", "1", "rad", "s"}
    doubled = _main_variable(coupled_results_doubled[measure.name], measure.name)
    factor = 4.0 if measure.units == "(input units)^2/Hz" else 1.0
    np.testing.assert_allclose(doubled.values, factor * variable.values, rtol=1e-6, atol=0)


@pytest.fixture(scope="module")
def one_sided_connectivity(coupled_time_series):
    """The coupled system's non-negative-frequency coefficients, declared one-sided."""
    multitaper = Multitaper(
        coupled_time_series, sampling_frequency=500, time_window_duration=0.5
    )
    two_sided = Connectivity.from_multitaper(multitaper)
    n_nonnegative = two_sided.frequencies.size
    return Connectivity(
        multitaper.fft()[..., :n_nonnegative, :],
        frequencies=two_sided.frequencies,
        expectation_type="trials_tapers",
        is_one_sided=True,
    )


@pytest.mark.parametrize("measure", list_measures(), ids=lambda measure: measure.name)
def test_requires_two_sided_matches_the_one_sided_guard(one_sided_connectivity, measure):
    """``requires_two_sided`` flags exactly the measures the core refuses on a
    one-sided spectrum; every other measure runs on it."""
    compute = getattr(one_sided_connectivity, measure.name)
    kwargs = _MEASURE_KWARGS.get(measure.name, {})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        if measure.requires_two_sided:
            with pytest.raises(ValueError, match="two-sided"):
                compute(**kwargs)
        else:
            compute(**kwargs)


# What users get when ``method`` is omitted. Pinned explicitly: every other
# check derives the default set from the same registry flag it would test.
_DEFAULT_MEASURES = {
    "coherence_magnitude",
    "coherence_phase",
    "debiased_squared_phase_lag_index",
    "debiased_squared_weighted_phase_lag_index",
    "imaginary_coherence",
    "pairwise_phase_consistency",
    "pairwise_spectral_granger_prediction",
    "phase_lag_index",
    "phase_locking_value",
    "power",
    "weighted_phase_lag_index",
}


def test_is_default_matches_the_measures_computed_without_a_method(coupled_time_series):
    """``is_default`` names exactly the measures the wrapper computes when
    ``method`` is omitted, and that set is the documented default."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = multitaper_connectivity(
            coupled_time_series, sampling_frequency=500, time_window_duration=0.5
        )
    assert set(result.data_vars) == _DEFAULT_MEASURES
    assert {m.name for m in list_measures() if m.is_default} == _DEFAULT_MEASURES


def test_time_and_angle_units_match_a_known_lag():
    """Signal 1 is signal 0 delayed by 3 samples (6 ms at 500 Hz): the measures
    in seconds report that delay, and the angle in radians is 2 pi f * 6 ms."""
    units = {measure.name: measure.units for measure in list_measures()}
    assert {name for name, unit in units.items() if unit == "s"} == {"delay", "group_delay"}
    assert {name for name, unit in units.items() if unit == "rad"} == {"coherence_phase"}

    rng = np.random.default_rng(53)
    leader = rng.standard_normal((1003, 20))
    time_series = np.stack([leader[3:], leader[:-3]], axis=-1)
    time_series += 0.3 * rng.standard_normal(time_series.shape)
    kwargs = {"sampling_frequency": 500, "time_halfbandwidth_product": 3}
    band = {"frequencies_of_interest": [5.0, 50.0]}
    lag = 3 / 500

    group_delay = multitaper_connectivity(
        time_series, method="group_delay", connectivity_kwargs=band, **kwargs
    )["group_delay"].sel(source="0", target="1")
    assert float(group_delay.squeeze()) == pytest.approx(lag, rel=0.05)

    delay = multitaper_connectivity(
        time_series, method="delay", connectivity_kwargs=band, **kwargs
    ).sel(source="0", target="1")
    zero_wrap = delay.isel(candidate=delay.sizes["candidate"] // 2)
    assert float(np.nanmedian(zero_wrap.values)) == pytest.approx(lag, rel=0.05)

    phase = (
        multitaper_connectivity(time_series, method="coherence_phase", **kwargs)
        .sel(source="0", target="1")
        .sel(frequency=20.0, method="nearest")
    )
    frequency = float(phase.frequency)
    assert float(phase.squeeze()) == pytest.approx(2 * np.pi * frequency * lag, rel=0.05)


def test_units_name_the_input_dependence_of_spectral_densities():
    """Densities scale with the input, so their units are given relative to it."""
    units = {measure.name: measure.units for measure in list_measures()}
    assert units["power"] == "(input units)^2/Hz"
    assert units["cross_spectral_density"] == "(input units)^2/Hz"
    assert units["coherence_magnitude"] == "1"
    assert units["group_delay"] == "s"


@pytest.fixture(scope="module")
def zero_drives_one():
    """Three signals: 0 drives 1 at a lag; 2 is independent."""
    coefficients = np.zeros((2, 3, 3))
    coefficients[0] = [[0.9, 0.0, 0.0], [0.6, 0.2, 0.0], [0.0, 0.0, 0.3]]
    coefficients[1] = [[-0.6, 0.0, 0.0], [0.3, 0.0, 0.0], [0.0, 0.0, 0.0]]
    return simulate_MVAR(coefficients, n_time_samples=1000, n_trials=40, random_state=0)


_BAND = [5.0, 60.0]
_ORIENTATION_KWARGS = {
    "blockwise_spectral_granger_prediction": {"group_labels": np.array([0, 1, 2])},
    "subset_pairwise_spectral_granger_prediction": {"pairs": [(0, 1)]},
    "delay": {"frequencies_of_interest": _BAND},
    "group_delay": {"frequencies_of_interest": _BAND},
    "phase_slope_index": {"frequencies_of_interest": _BAND},
}


def _entries(values):
    """``(values[..., 0, 1], values[..., 1, 0])`` summarized by their medians."""
    return np.nanmedian(values[..., 0, 1]), np.nanmedian(values[..., 1, 0])


def _low_level_entries(connectivity, name):
    """Band summaries of a directed measure's native ``[0, 1]`` and ``[1, 0]``."""
    result = getattr(connectivity, name)(**_ORIENTATION_KWARGS.get(name, {}))
    if name == "blockwise_spectral_granger_prediction":
        result = result[0]
    elif name == "group_delay":
        return _entries(result[0])
    elif name == "phase_slope_index":
        return _entries(result)
    elif name == "delay":
        # (..., frequency, candidate, n, n): the zero-wrap candidate over the band.
        return _entries(result[..., result.shape[-3] // 2, :, :])
    frequencies = connectivity.frequencies
    in_band = (frequencies >= _BAND[0]) & (frequencies <= _BAND[1])
    return _entries(np.asarray(result)[..., in_band, :, :])


@pytest.mark.parametrize(
    "measure", list_measures(directed=True), ids=lambda measure: measure.name
)
def test_array_orientation_matches_the_computed_direction(zero_drives_one, measure):
    """Signal 0 drives signal 1, so the 0 -> 1 entry must dominate the 1 -> 0
    entry at the position ``array_orientation`` names. Time reversal flips the
    apparent direction, so the time-reversed measure is given reversed data."""
    time_series = zero_drives_one
    if measure.name == "time_reversed_spectral_granger_prediction":
        time_series = time_series[::-1]
    connectivity = Connectivity.from_transform(
        Multitaper(time_series, sampling_frequency=200, time_halfbandwidth_product=3)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        entry_01, entry_10 = _low_level_entries(connectivity, measure.name)

    if measure.array_orientation == "target_source":
        zero_to_one, one_to_zero = entry_10, entry_01
    else:
        assert measure.array_orientation == "source_target"
        zero_to_one, one_to_zero = entry_01, entry_10
    assert zero_to_one > one_to_zero


def test_only_directed_measures_have_an_array_orientation():
    for measure in list_measures():
        assert (measure.array_orientation is not None) == measure.is_directed
