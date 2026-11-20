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
from spectral_connectivity._measure_registry import _MEASURE_SPECS
from spectral_connectivity.simulate import simulate_MVAR
from spectral_connectivity.wrapper import (
    DEFAULT_METHODS,
    MeasureInfo,
    frequency_band_reduce,
)


def test_lists_every_registered_measure():
    """Every wrapper-supported measure is returned exactly once."""
    names = [measure.name for measure in list_measures()]
    assert names == list(_MEASURE_SPECS)


def test_every_public_connectivity_measure_is_registered():
    """A new Connectivity measure must declare its wrapper contract.

    An unregistered method would be invisible to ``list_measures()`` and, when
    requested by name, would fall back to the undirected pairwise contract
    (not directed, no two-sided-spectrum requirement).
    """
    import inspect

    from spectral_connectivity.connectivity import (
        _DEPRECATED_MEASURE_NAMES,
        _NON_MEASURE_METHODS,
    )

    public_methods = {
        name
        for name in dir(Connectivity)
        if not name.startswith("_")
        and inspect.isfunction(inspect.getattr_static(Connectivity, name))
    }
    # A deprecated name is a method standing for a registered measure.
    assert set(_DEPRECATED_MEASURE_NAMES) <= public_methods
    assert set(_DEPRECATED_MEASURE_NAMES.values()) <= set(_MEASURE_SPECS)
    assert public_methods - _NON_MEASURE_METHODS - set(_DEPRECATED_MEASURE_NAMES) == set(
        _MEASURE_SPECS
    )


def test_returns_measureinfo_records_with_populated_fields():
    """Records expose name, category, description, and capability flags."""
    coherence = next(
        measure for measure in list_measures() if measure.name == "magnitude_squared_coherence"
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
    assert "magnitude_squared_coherence" in {measure.name for measure in undirected}


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


_MEASURES = {measure.name: measure for measure in list_measures()}
# The other value variables of a measure's Dataset, by variable name.
_COMPANIONS = {
    name: companion for spec in _MEASURE_SPECS.values() for name, companion in spec.companions
}
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
    "magnitude_squared_coherence",
    "coherence_phase",
    "debiased_squared_phase_lag_index",
    "debiased_squared_weighted_phase_lag_index",
    "imaginary_coherence",
    "pairwise_phase_consistency",
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
    band = {"frequency_range": (5.0, 50.0)}
    lag = 3 / 500

    group_delay = multitaper_connectivity(time_series, method="group_delay", **band, **kwargs)[
        "group_delay"
    ].sel(source="0", target="1")
    assert float(group_delay.squeeze()) == pytest.approx(lag, rel=0.05)

    delay = multitaper_connectivity(time_series, method="delay", **band, **kwargs).sel(
        source="0", target="1"
    )
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
    assert units["magnitude_squared_coherence"] == "1"
    assert units["group_delay"] == "s"


@pytest.fixture(scope="module")
def zero_drives_one():
    """Three signals: 0 drives 1 at a lag; 2 is independent."""
    coefficients = np.zeros((2, 3, 3))
    coefficients[0] = [[0.9, 0.0, 0.0], [0.6, 0.2, 0.0], [0.0, 0.0, 0.3]]
    coefficients[1] = [[-0.6, 0.0, 0.0], [0.3, 0.0, 0.0], [0.0, 0.0, 0.0]]
    return simulate_MVAR(coefficients, n_time_samples=1000, n_trials=40, random_state=0)


@pytest.fixture(scope="module")
def zero_drives_one_connectivity(zero_drives_one):
    """``Connectivity`` of ``zero_drives_one`` with the settings used below."""
    return Connectivity.from_transform(
        Multitaper(zero_drives_one, sampling_frequency=200, time_halfbandwidth_product=3)
    )


_BAND = [5.0, 60.0]
_ORIENTATION_KWARGS = {
    "blockwise_spectral_granger_prediction": {"group_labels": np.array([0, 1, 2])},
    "subset_pairwise_spectral_granger_prediction": {"pairs": [(0, 1)]},
    "delay": {"frequency_range": _BAND},
    "group_delay": {"frequency_range": _BAND},
    "phase_slope_index": {"frequency_range": _BAND},
}


# Lead/lag measures: a value above this at [..., i, j] means signal i leads j.
_LEADS_ABOVE = {
    "directed_phase_lag_index": 0.5,
    "phase_slope_index": 0.0,
    "delay": 0.0,
    "group_delay": 0.0,
}


def _in_band(connectivity, result):
    """A frequency-resolved ``(..., frequency, n, n)`` result within the band."""
    frequencies = connectivity.frequencies
    in_band = (frequencies >= _BAND[0]) & (frequencies <= _BAND[1])
    return np.asarray(result)[..., in_band, :, :]


# How each measure's output becomes (..., n, n) values over the band; the
# default is the frequency slice.
_BAND_MATRICES = {
    # (values, labels)
    "blockwise_spectral_granger_prediction": lambda c, result: _in_band(c, result[0]),
    # (delay, slope, r_value), already fit over the band
    "group_delay": lambda c, result: result[0],
    # (..., frequency, candidate, n, n): the zero-wrap candidate
    "delay": lambda c, result: result[..., result.shape[-3] // 2, :, :],
    # already summed over the band
    "phase_slope_index": lambda c, result: result,
}


def _zero_one_and_one_zero(connectivity, name):
    """A directed measure's ``[..., 0, 1]`` and ``[..., 1, 0]`` over the band."""
    result = getattr(connectivity, name)(**_ORIENTATION_KWARGS.get(name, {}))
    result = _BAND_MATRICES.get(name, _in_band)(connectivity, result)
    return result[..., 0, 1], result[..., 1, 0]


@pytest.mark.parametrize(
    "measure", list_measures(directed=True), ids=lambda measure: measure.name
)
def test_directed_measures_place_source_first(zero_drives_one, measure):
    """Signal 0 drives and leads signal 1, so every directed ``Connectivity``
    array holds ``0 -> 1`` at ``[..., 0, 1]``. Time reversal flips the apparent
    direction, so the time-reversed measure is given reversed data, on which the
    planted ``0 -> 1`` again lands at ``[..., 0, 1]``."""
    time_series = zero_drives_one
    if measure.name == "time_reversed_spectral_granger_prediction":
        time_series = time_series[::-1]
    connectivity = Connectivity.from_transform(
        Multitaper(time_series, sampling_frequency=200, time_halfbandwidth_product=3)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        zero_to_one, one_to_zero = _zero_one_and_one_zero(connectivity, measure.name)

    if measure.name in _LEADS_ABOVE:
        assert np.nanmedian(zero_to_one) > _LEADS_ABOVE[measure.name]
    else:
        assert np.nanmax(zero_to_one) > 10 * np.nanmax(one_to_zero)


@pytest.mark.parametrize(
    "measure",
    [measure for measure in list_measures(directed=True) if measure.category == "pairwise"],
    ids=lambda measure: measure.name,
)
def test_wrapper_returns_directed_connectivity_arrays_unchanged(
    zero_drives_one, zero_drives_one_connectivity, measure
):
    """The wrapper labels a directed ``Connectivity`` array's last two axes
    ``source`` and ``target`` without reordering them, so its values equal the
    ``Connectivity`` method's output exactly."""
    kwargs = _ORIENTATION_KWARGS.get(measure.name, {})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = multitaper_connectivity(
            zero_drives_one,
            sampling_frequency=200,
            time_halfbandwidth_product=3,
            method=measure.name,
            connectivity_kwargs=kwargs,
        )
        expected = getattr(zero_drives_one_connectivity, measure.name)(**kwargs)

    assert result.dims[-2:] == ("source", "target")
    assert result.shape == expected.shape
    # The two computations agree only to roundoff (it varies with the BLAS);
    # swapped axes would differ by orders of magnitude, since 0 -> 1 dominates.
    np.testing.assert_allclose(result.values, expected, rtol=1e-9, atol=1e-12)


# What the documentation states about each value variable (the measures and
# group delay's companions), pinned so that a change to a measure's diagonal,
# or to what the registry says about it, fails:
# (diagonal_policy, diagonal_value, nan_for_zero_power_signal).
_DOCUMENTED_DIAGONALS = {
    "magnitude_squared_coherence": ("masked_nan", None, True),
    "coherence_phase": ("masked_nan", None, True),
    "debiased_squared_phase_lag_index": ("constant", 0.0, False),
    "debiased_squared_weighted_phase_lag_index": ("constant", 0.0, False),
    "imaginary_coherence": ("constant", 0.0, True),
    "pairwise_phase_consistency": ("constant", 1.0, True),
    "pairwise_spectral_granger_prediction": ("undefined_nan", None, False),
    "phase_lag_index": ("constant", 0.0, False),
    "phase_locking_value": ("constant", 1.0, True),
    "power": ("not_applicable", None, False),
    "weighted_phase_lag_index": ("constant", 0.0, False),
    "coherency": ("masked_nan", None, True),
    "cross_spectral_density": ("estimated", None, False),
    "imaginary_coherency": ("masked_nan", None, False),
    "partial_coherence": ("masked_nan", None, False),
    "corrected_imaginary_phase_locking_value": ("rounding_noise", 0.0, True),
    "directed_phase_lag_index": ("constant", 0.5, False),
    "subset_pairwise_spectral_granger_prediction": ("undefined_nan", None, False),
    "conditional_spectral_granger_prediction": ("undefined_nan", None, False),
    "time_reversed_spectral_granger_prediction": ("undefined_nan", None, False),
    "directed_transfer_function": ("estimated", None, False),
    "directed_coherence": ("estimated", None, False),
    "partial_directed_coherence": ("estimated", None, False),
    "generalized_partial_directed_coherence": ("estimated", None, False),
    "direct_directed_transfer_function": ("estimated", None, False),
    "blockwise_spectral_granger_prediction": ("undefined_nan", None, False),
    "canonical_coherence": ("masked_nan", None, False),
    "maximized_imaginary_coherency": ("masked_nan", None, False),
    "multivariate_interaction_measure": ("masked_nan", None, False),
    "canonical_coherency": ("not_applicable", None, False),
    "maximized_imaginary_coherency_components": ("not_applicable", None, False),
    "delay": ("masked_nan", None, True),
    "global_coherence": ("not_applicable", None, False),
    "group_delay": ("masked_nan", None, True),
    "group_delay_slope": ("masked_nan", None, True),
    "group_delay_r_value": ("constant", 1.0, False),
    "phase_slope_index": ("masked_nan", None, True),
}
# Every value variable and its record: the measures, then their companions.
_VALUE_VARIABLES = [
    *((measure.name, measure) for measure in list_measures()),
    *_COMPANIONS.items(),
]

# Pairs of axes whose shared labels form a variable's diagonal.
_PAIR_AXES = (("source", "target"), ("source_group", "target_group"))


def _diagonal(variable):
    """The self-connection entries of a variable with a pair of axes, or None.

    Shape ``(n_signals, ...)``: one slice per signal (or group).
    """
    for first, second in _PAIR_AXES:
        if first in variable.dims and second in variable.dims:
            return np.stack(
                [
                    variable.sel({first: label, second: label}).values
                    for label in variable[first].values
                ]
            )
    return None


@pytest.mark.parametrize("measure", list_measures(), ids=lambda measure: measure.name)
def test_diagonal_matches_its_policy(coupled_results, measure):
    """The recorded ``diagonal_policy`` describes the values on the diagonal."""
    _assert_diagonal_matches(
        measure, _main_variable(coupled_results[measure.name], measure.name)
    )


def _rounding(variable, policy="constant"):
    """How far ``variable``'s values may lie from an exact value by rounding:
    the bounds that ``_DIAGONAL_POLICIES`` derives, ``4 * (n_observations + 4)
    * eps`` for an average and ``sqrt(eps)`` for a ``rounding_noise`` diagonal."""
    eps = np.finfo(np.real(variable.values).dtype).eps
    if policy == "rounding_noise":
        return np.sqrt(eps)
    return 4 * (variable.attrs["n_observations"] + 4) * eps


def _assert_diagonal_matches(measure, variable):
    """``variable``'s diagonal holds what ``measure.diagonal_policy`` says."""
    diagonal = _diagonal(variable)
    if measure.diagonal_policy == "not_applicable":
        assert diagonal is None
    elif measure.diagonal_policy in {"masked_nan", "undefined_nan"}:
        assert np.isnan(diagonal).all()
    elif measure.diagonal_value is not None:
        _assert_nominal_diagonal(measure, variable, diagonal)
    else:
        assert measure.diagonal_policy == "estimated"
        assert np.isfinite(diagonal).all()
        assert np.ptp(np.abs(diagonal)) > 0  # data-dependent, not a constant


def _assert_nominal_diagonal(measure, variable, diagonal):
    """``diagonal`` (entries of ``variable``) holds ``measure.diagonal_value``
    to within the rounding its policy allows."""
    np.testing.assert_allclose(
        diagonal,
        measure.diagonal_value,
        rtol=0,
        atol=_rounding(variable, measure.diagonal_policy),
    )


# Measures that need at least two observations (a debiasing factor n - 1).
_NEED_TWO_OBSERVATIONS = {
    "debiased_squared_phase_lag_index",
    "debiased_squared_weighted_phase_lag_index",
    "pairwise_phase_consistency",
}


@pytest.mark.parametrize(
    ("measure", "n_trials"),
    [
        pytest.param(measure, n_trials, id=f"{measure.name}-{n_trials}")
        for measure in list_measures()
        if measure.diagonal_value is not None
        for n_trials in (1, 2)
        if n_trials > 1 or measure.name not in _NEED_TWO_OBSERVATIONS
    ],
)
def test_nominal_diagonal_holds_with_one_or_two_observations(
    coupled_time_series, measure, n_trials
):
    """With one taper per trial, the fewest observations a measure accepts,
    where rounding is least averaged out: the tolerance still holds."""
    with warnings.catch_warnings():
        # One observation makes the normalized measures warn that they are
        # degenerate; that is expected here.
        warnings.simplefilter("ignore")
        variable = multitaper_connectivity(
            coupled_time_series[:, :n_trials],
            sampling_frequency=500,
            time_window_duration=0.5,
            time_halfbandwidth_product=1,
            method=measure.name,
        )
    assert variable.attrs["n_observations"] == n_trials  # one taper per trial
    _assert_nominal_diagonal(measure, variable, _diagonal(variable))


_NOMINAL_DIAGONALS = [m for m in list_measures() if m.diagonal_value is not None]
_FLAT_SIGNAL, _FLAT_SIGNAL_GROUP = "3", "b"


@pytest.fixture(scope="module")
def flat_signal_results(coupled_time_series):
    """Every measure, and its other variables, with signal "3" flat (all zeros).

    One call computes them all except subset Granger, whose ``pairs`` no other
    measure accepts; its pairs include the flat signal.
    """
    flat = coupled_time_series.copy()
    flat[..., -1] = 0.0
    settings = {"sampling_frequency": 500, "time_window_duration": 0.5}
    subset = "subset_pairwise_spectral_granger_prediction"
    with warnings.catch_warnings():
        # A flat signal makes the normalized measures warn; that is expected.
        warnings.simplefilter("ignore")
        merged = multitaper_connectivity(
            flat,
            method=[measure.name for measure in list_measures() if measure.name != subset],
            group_labels=_GROUPS,
            **settings,
        )
        subset_result = multitaper_connectivity(
            flat, method=subset, connectivity_kwargs={"pairs": [(0, 3), (1, 0)]}, **settings
        )
    return merged.assign({subset: subset_result})


def _entries_involving_the_flat_signal(variable):
    """The entries of ``variable`` that depend on the flat signal: its rows and
    columns of a signal-pair axis, its group's of a group axis, and otherwise
    (a group-pair connection or a whole-system component) every entry."""
    for axes, label in (
        (("source", "target"), _FLAT_SIGNAL),
        (("source_group", "target_group"), _FLAT_SIGNAL_GROUP),
    ):
        present = [axis for axis in axes if axis in variable.dims]
        if present:
            return np.concatenate(
                [variable.sel({axis: label}).values.ravel() for axis in present]
            )
    return variable.values.ravel()


@pytest.mark.parametrize(
    ("name", "record"), _VALUE_VARIABLES, ids=[n for n, _ in _VALUE_VARIABLES]
)
def test_nan_for_zero_power_signal_matches_what_a_flat_signal_gives(
    flat_signal_results, name, record
):
    """The flag is set exactly where every entry involving a zero-power signal,
    the diagonal included, is NaN."""
    entries = _entries_involving_the_flat_signal(flat_signal_results[name])
    assert record.nan_for_zero_power_signal == bool(np.isnan(entries).all())


@pytest.mark.parametrize(
    "measure",
    [m for m in _NOMINAL_DIAGONALS if not m.nan_for_zero_power_signal],
    ids=lambda measure: measure.name,
)
def test_a_flat_signal_keeps_the_nominal_diagonal_unless_it_is_nan(
    flat_signal_results, measure
):
    """Where the flag is not set, a flat signal's self-connection keeps the
    nominal value."""
    variable = flat_signal_results[measure.name]
    flat_self = variable.sel(source=_FLAT_SIGNAL, target=_FLAT_SIGNAL).values
    _assert_nominal_diagonal(measure, variable, flat_self)


def test_imaginary_coherency_is_zero_not_nan_for_a_flat_signal(flat_signal_results):
    """As its docstring says: coherency masks a flat signal's pairs with
    nan + 0j, and the imaginary part of that is 0."""
    variable = flat_signal_results["imaginary_coherency"]
    flat_pairs = variable.sel(source=_FLAT_SIGNAL).drop_sel(target=_FLAT_SIGNAL)
    np.testing.assert_array_equal(flat_pairs, 0.0)


def test_documented_diagonals_are_the_registered_ones():
    registered = {
        name: (record.diagonal_policy, record.diagonal_value, record.nan_for_zero_power_signal)
        for name, record in _VALUE_VARIABLES
    }
    assert registered == _DOCUMENTED_DIAGONALS


def test_cross_spectral_density_diagonal_is_the_power_spectrum(coupled_results):
    """The diagonal is the power, real to within the rounding of its average."""
    csd = coupled_results["cross_spectral_density"]
    diagonal = np.diagonal(csd.values, axis1=-2, axis2=-1)
    np.testing.assert_allclose(
        diagonal.real, coupled_results["power"].values, rtol=1e-12, atol=0
    )
    assert np.all(np.abs(diagonal.imag) <= _rounding(csd) * diagonal.real)


_OPTIONAL_MEASURE_ATTRS = (
    "value_range_lower",
    "value_range_upper",
    "value_range_applies_to",
    "sign_convention",
    "quantity",
    "formula",
    "diagonal_value",
)


def _expected_measure_attrs(measure):
    """The attrs a measure variable takes from its registry record."""
    lower, upper = measure.value_range
    expected = {
        "interpretation": measure.interpretation,
        "diagonal_policy": measure.diagonal_policy,
    }
    if np.isfinite(lower):
        expected["value_range_lower"] = lower
    if np.isfinite(upper):
        expected["value_range_upper"] = upper
    if getattr(measure, "is_complex", False):
        expected["value_range_applies_to"] = "magnitude"
    for key in ("sign_convention", "quantity", "formula"):
        if getattr(measure, key, None) is not None:
            expected[key] = getattr(measure, key)
    if measure.diagonal_value is not None:
        expected["diagonal_value"] = measure.diagonal_value
    # A flag is stored as 0/1: NetCDF has no boolean.
    expected["nan_for_zero_power_signal"] = int(measure.nan_for_zero_power_signal)
    return expected


@pytest.mark.parametrize("measure", list_measures(), ids=lambda measure: measure.name)
def test_measure_variable_records_how_to_read_it(coupled_results, measure):
    """Interpretation, sign convention, theoretical range, and diagonal policy
    are on the measure's own variable, so an extracted variable keeps them; an
    unbounded side of the range and facts a measure does not state are absent."""
    _assert_records_how_to_read(
        measure, _main_variable(coupled_results[measure.name], measure.name)
    )


def _assert_records_how_to_read(record, variable, **rewritten):
    """``variable`` carries ``record``'s attrs, with any ``rewritten`` ones, and
    no fact the record omits."""
    expected = {**_expected_measure_attrs(record), **rewritten}
    assert {key: variable.attrs.get(key) for key in expected} == expected
    absent = set(_OPTIONAL_MEASURE_ATTRS) - set(expected)
    assert not absent & set(variable.attrs)


# Dataset variables that are not measure values: group membership and the
# spatial vectors, filters, and patterns. They carry provenance and labels,
# not a measure's interpretation, range, or diagonal.
_AUXILIARY_VARIABLES = {
    "group_membership",
    "global_coherence_vectors",
    *(
        f"{measure.name}_{kind}"
        for measure in list_measures(category="multivariate_components")
        for kind in ("filters", "patterns")
    ),
}


@pytest.mark.parametrize(
    "measure",
    [m for m in list_measures() if _CATEGORY_STRUCTURE[m.category][0] is xr.Dataset],
    ids=lambda measure: measure.name,
)
def test_every_variable_of_a_dataset_result_records_how_to_read_it(coupled_results, measure):
    """A measure's companion variables (group delay's slope and fit
    correlation) carry their own record, whose range and diagonal hold of their
    values; the auxiliary variables carry none. (The measure's own variable is
    checked by ``test_measure_variable_records_how_to_read_it``.)"""
    companions = dict(_MEASURE_SPECS[measure.name].companions)
    for name, variable in coupled_results[measure.name].data_vars.items():
        if name in _AUXILIARY_VARIABLES:
            assert not {"interpretation", "diagonal_policy"} & set(variable.attrs), name
        elif name != measure.name:
            record = companions[name]
            _assert_records_how_to_read(record, variable)
            finite = variable.values[np.isfinite(variable.values)]
            assert (
                record.value_range[0] <= finite.min() <= finite.max() <= record.value_range[1]
            )
            _assert_diagonal_matches(record, variable)


def test_coherence_family_states_its_formula_and_quantity():
    measures = _MEASURES
    quantities = {
        "magnitude_squared_coherence": "squared_magnitude",
        "partial_coherence": "squared_magnitude",
        "canonical_coherence": "squared_magnitude",
        "coherence_phase": "phase",
        "coherency": "complex",
        "canonical_coherency": "complex",
        "imaginary_coherence": "magnitude",
        "imaginary_coherency": "signed",
    }
    assert {name: m.quantity for name, m in measures.items() if m.quantity} == quantities
    assert {name for name, m in measures.items() if m.formula} == set(quantities)


def test_signed_and_phase_measures_state_their_sign_convention():
    """Every measure whose sign or angle says which signal leads names how."""
    signed = {
        "coherence_phase",
        "coherency",
        "imaginary_coherency",
        "phase_lag_index",
        "weighted_phase_lag_index",
        "canonical_coherency",
        "directed_phase_lag_index",
        "delay",
        "group_delay",
        "phase_slope_index",
    }
    assert {m.name for m in list_measures() if m.sign_convention} == signed


_BANDS = {"low": (4.0, 40.0), "high": (40.0, 120.0)}


@pytest.mark.parametrize(
    ("name", "frequency_reduction", "description"),
    [
        ("power", "integral", "the integral over the band"),
        ("cross_spectral_density", "integral", "the integral over the band"),
        (
            "magnitude_squared_coherence",
            "mean",
            "the mean over the band's frequency bins",
        ),
        (
            "coherence_phase",
            "mean",
            (
                "the circular mean (the angle of the mean unit phasor) over the band's "
                "frequency bins"
            ),
        ),
        (
            "coherency",
            "mean",
            "the complex (vector) mean over the band's frequency bins",
        ),
    ],
)
def test_band_reduction_keeps_each_measure_attribute_true(
    coupled_results, name, frequency_reduction, description
):
    """A band value is a reduction of the per-frequency measure, so the attrs
    that describe single bins (interpretation, formula) say which reduction it
    is; the range, sign convention, quantity, and diagonal policy still hold of
    the band values unchanged."""
    measure = _MEASURES[name]
    reduced = frequency_band_reduce(
        _main_variable(coupled_results[name], name),
        _BANDS,
        frequency_reduction=frequency_reduction,
    )
    rewritten = {
        "interpretation": f"Each value is {description} of the per-frequency values. "
        f"{measure.interpretation}"
    }
    if measure.formula is not None:
        rewritten["formula"] = f"{description} of {measure.formula}"
    _assert_records_how_to_read(measure, reduced, **rewritten)

    values = np.abs(reduced.values) if measure.is_complex else reduced.values
    finite = values[np.isfinite(values)]
    assert finite.size > 0
    lower, upper = measure.value_range
    assert lower - _rounding(reduced) <= finite.min()
    assert finite.max() <= upper + _rounding(reduced)
    _assert_diagonal_matches(measure, reduced)


def test_cropping_and_decimation_leave_measure_attributes_unchanged(coupled_time_series):
    """Selecting bins does not change what each bin's value means."""
    names = ["power", "magnitude_squared_coherence", "coherence_phase", "coherency"]
    result = multitaper_connectivity(
        coupled_time_series,
        sampling_frequency=500,
        time_window_duration=0.5,
        method=names,
        frequency_range=(10.0, 100.0),
        frequency_decimation=3,
    )
    for name in names:
        _assert_records_how_to_read(_MEASURES[name], result[name])
