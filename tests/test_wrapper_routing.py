"""How the wrappers route settings to the transform and to the measures.

Common settings are named arguments; advanced transform settings go in
``transform_kwargs`` and measure settings in ``connectivity_kwargs``. Each
setting reaches its target by one route, and the result, including its
provenance, does not depend on which supported route supplied it.
"""

import inspect
import warnings

import numpy as np
import pytest

from spectral_connectivity import Connectivity, Multitaper, multitaper_connectivity
from spectral_connectivity.wrapper import connectivity_to_xarray

_COMMON_TRANSFORM_SETTINGS = (
    "time_window_step",
    "time_halfbandwidth_product",
    "n_tapers",
    "detrend_type",
)


@pytest.fixture(scope="module")
def time_series():
    """Signal 1 follows signal 0 by 3 samples; shape (1000, 6, 3)."""
    rng = np.random.default_rng(0)
    leader = rng.standard_normal((1003, 6))
    signals = np.stack([leader[3:], leader[:-3], rng.standard_normal((1000, 6))], axis=-1)
    return signals + 0.5 * rng.standard_normal(signals.shape)


def _assert_identical(actual, expected):
    """Same values, coordinates, and attributes (provenance included)."""
    assert actual.identical(expected)
    assert actual.attrs == expected.attrs


def test_multitaper_connectivity_keeps_its_released_positional_order():
    parameters = inspect.signature(multitaper_connectivity).parameters
    positional = [
        name
        for name, parameter in parameters.items()
        if parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    ]
    assert positional == [
        "time_series",
        "sampling_frequency",
        "time_window_duration",
        "method",
        "signal_names",
        "squeeze",
        "connectivity_kwargs",
    ]
    for name in (*_COMMON_TRANSFORM_SETTINGS, "transform_kwargs"):
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY


def test_connectivity_to_xarray_keeps_its_released_positional_order():
    parameters = inspect.signature(connectivity_to_xarray).parameters
    positional = [
        name
        for name, parameter in parameters.items()
        if parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    ]
    assert positional == ["m", "method", "signal_names", "squeeze"]
    for name in ("group_labels", "connectivity_kwargs"):
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
    assert "transform_kwargs" not in parameters


def test_common_settings_default_to_the_multitaper_defaults():
    wrapper = inspect.signature(multitaper_connectivity).parameters
    transform = inspect.signature(Multitaper).parameters
    for name in _COMMON_TRANSFORM_SETTINGS:
        assert wrapper[name].default == transform[name].default, name


def test_common_settings_by_name_and_by_position_are_silent_and_reach_the_transform(
    time_series,
):
    settings = {
        "time_window_step": 0.25,
        "time_halfbandwidth_product": 2,
        "n_tapers": 2,
        "detrend_type": "linear",
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        by_name = multitaper_connectivity(
            time_series,
            sampling_frequency=500,
            time_window_duration=0.5,
            method="magnitude_squared_coherence",
            **settings,
        )
        by_position = multitaper_connectivity(
            time_series, 500, 0.5, "magnitude_squared_coherence", **settings
        )
    expected = connectivity_to_xarray(
        Multitaper(time_series, 500, time_window_duration=0.5, **settings),
        "magnitude_squared_coherence",
    )
    _assert_identical(by_position, by_name)
    np.testing.assert_array_equal(by_name.values, expected.values)
    assert (
        by_name.attrs["transform_parameters_json"]
        == (expected.attrs["transform_parameters_json"])
    )
    assert by_name.attrs["mt_n_tapers"] == 2
    assert by_name.attrs["mt_detrend_type"] == "linear"


def test_transform_kwargs_reach_the_transform_silently(time_series):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = multitaper_connectivity(
            time_series,
            sampling_frequency=500,
            method="power",
            transform_kwargs={"n_fft_samples": 2048, "start_time": 1.5},
        )
    assert result.attrs["mt_n_fft_samples"] == 2048
    assert result.attrs["mt_start_time"] == 1.5


def test_legacy_transform_keywords_warn_once_at_the_caller_and_match_the_mapping(
    time_series,
):
    settings = {"n_fft_samples": 2048, "start_time": 1.5, "fft_workers": 1}
    canonical = multitaper_connectivity(
        time_series, sampling_frequency=500, method="power", transform_kwargs=settings
    )
    call = lambda: multitaper_connectivity(  # noqa: E731
        time_series, sampling_frequency=500, method="power", **settings
    )
    with pytest.warns(DeprecationWarning, match="transform_kwargs") as record:
        legacy = call()
    deprecations = [w for w in record if issubclass(w.category, DeprecationWarning)]
    assert [(w.filename, w.lineno) for w in deprecations] == [
        (__file__, call.__code__.co_firstlineno)
    ]
    message = str(deprecations[0].message)
    assert "removed in 4.0" in message
    for key in settings:
        assert key in message
    _assert_identical(legacy, canonical)


def test_legacy_and_mapping_transform_keywords_combine(time_series):
    canonical = multitaper_connectivity(
        time_series,
        sampling_frequency=500,
        method="power",
        transform_kwargs={"n_fft_samples": 2048, "start_time": 1.5},
    )
    with pytest.warns(DeprecationWarning, match="start_time"):
        mixed = multitaper_connectivity(
            time_series,
            sampling_frequency=500,
            method="power",
            transform_kwargs={"n_fft_samples": 2048},
            start_time=1.5,
        )
    _assert_identical(mixed, canonical)


def test_a_transform_setting_given_twice_is_rejected(time_series):
    with pytest.raises(ValueError, match=r"n_fft_samples.*transform_kwargs.*keyword"):
        multitaper_connectivity(
            time_series,
            sampling_frequency=500,
            method="power",
            transform_kwargs={"n_fft_samples": 2048},
            n_fft_samples=1024,
        )


@pytest.mark.parametrize("name", ["sampling_frequency", *_COMMON_TRANSFORM_SETTINGS])
def test_a_named_setting_inside_transform_kwargs_is_rejected(time_series, name):
    """A common setting has one route, its named argument, so naming it in the
    mapping too can never silently pick one of two values."""
    with pytest.raises(ValueError, match=rf"{name}.*transform_kwargs.*{name}=") as excinfo:
        multitaper_connectivity(
            time_series,
            sampling_frequency=500,
            method="power",
            transform_kwargs={name: 2},
        )
    assert "named argument" in str(excinfo.value)


def test_a_misspelled_transform_kwargs_key_is_rejected(time_series):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(TypeError, match=r"'n_fft_sample'.*n_fft_samples"):
            multitaper_connectivity(
                time_series,
                sampling_frequency=500,
                method="power",
                transform_kwargs={"n_fft_sample": 2048},
            )


def test_an_unknown_legacy_keyword_is_rejected_before_any_warning(time_series):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(TypeError, match=r"'pairs'.*connectivity_kwargs"):
            multitaper_connectivity(
                time_series, sampling_frequency=500, method="power", pairs=[(0, 1)]
            )


def test_transform_setting_in_connectivity_kwargs_names_both_transform_routes(time_series):
    with pytest.raises(TypeError, match="connectivity_kwargs configures the measure only") as (
        excinfo
    ):
        multitaper_connectivity(
            time_series,
            sampling_frequency=500,
            method="magnitude_squared_coherence",
            connectivity_kwargs={"n_fft_samples": 2048},
        )
    message = str(excinfo.value)
    assert "transform_kwargs" in message
    assert "by name" in message


@pytest.fixture(scope="module")
def multitaper(time_series):
    return Multitaper(time_series, sampling_frequency=500, time_window_duration=0.5)


def test_connectivity_kwargs_reach_the_measure_of_connectivity_to_xarray(multitaper):
    pairs = [(0, 1), (1, 0)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)  # orientation reminder
        warnings.simplefilter("error", category=DeprecationWarning)
        result = connectivity_to_xarray(
            multitaper,
            "subset_pairwise_spectral_granger_prediction",
            connectivity_kwargs={"pairs": pairs},
        )
        expected = Connectivity.from_transform(
            multitaper
        ).subset_pairwise_spectral_granger_prediction(pairs)
    np.testing.assert_array_equal(result.values, expected)
    assert result.attrs["arg_pairs_json"] == "[[0,1],[1,0]]"


def test_connectivity_to_xarray_legacy_keywords_warn_once_and_match_the_mapping(multitaper):
    settings = {"frequency_range": (20.0, 80.0), "frequency_resolution": 10.0}
    canonical = connectivity_to_xarray(
        multitaper, "phase_slope_index", connectivity_kwargs=settings
    )
    call = lambda: connectivity_to_xarray(multitaper, "phase_slope_index", **settings)  # noqa: E731
    with pytest.warns(DeprecationWarning, match="connectivity_kwargs") as record:
        legacy = call()
    deprecations = [w for w in record if issubclass(w.category, DeprecationWarning)]
    assert [(w.filename, w.lineno) for w in deprecations] == [
        (__file__, call.__code__.co_firstlineno)
    ]
    assert "removed in 4.0" in str(deprecations[0].message)
    _assert_identical(legacy, canonical)


def test_connectivity_to_xarray_rejects_a_measure_setting_given_twice(multitaper):
    with pytest.raises(ValueError, match=r"frequency_range.*connectivity_kwargs.*keyword"):
        connectivity_to_xarray(
            multitaper,
            "phase_slope_index",
            connectivity_kwargs={"frequency_range": (20.0, 80.0)},
            frequency_range=(10.0, 90.0),
        )


@pytest.mark.parametrize("route", ["mapping", "legacy"])
def test_connectivity_to_xarray_rejects_a_misspelled_measure_setting(multitaper, route):
    kwargs = (
        {"connectivity_kwargs": {"frequency_rang": (20.0, 80.0)}}
        if route == "mapping"
        else {"frequency_rang": (20.0, 80.0)}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(TypeError, match="frequency_rang"):
            connectivity_to_xarray(multitaper, "phase_slope_index", **kwargs)


def test_group_labels_reach_connectivity_to_xarray_by_name_or_mapping(multitaper):
    labels = [0, 0, 1]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        named = connectivity_to_xarray(multitaper, "canonical_coherence", group_labels=labels)
        mapped = connectivity_to_xarray(
            multitaper, "canonical_coherence", connectivity_kwargs={"group_labels": labels}
        )
    expected, _ = Connectivity.from_transform(multitaper).canonical_coherence(labels)
    np.testing.assert_array_equal(named.values, expected)
    assert named.attrs["arg_group_labels_json"] == "[0,0,1]"
    _assert_identical(mapped, named)


def test_group_labels_by_name_and_in_the_mapping_is_rejected(multitaper):
    with pytest.raises(ValueError, match="group_labels was given both"):
        connectivity_to_xarray(
            multitaper,
            "canonical_coherence",
            group_labels=[0, 0, 1],
            connectivity_kwargs={"group_labels": [0, 1, 1]},
        )


def test_group_labels_reach_multitaper_connectivity_by_name_or_mapping(time_series):
    labels = [0, 0, 1]
    named = multitaper_connectivity(
        time_series, sampling_frequency=500, method="canonical_coherence", group_labels=labels
    )
    mapped = multitaper_connectivity(
        time_series,
        sampling_frequency=500,
        method="canonical_coherence",
        connectivity_kwargs={"group_labels": labels},
    )
    _assert_identical(mapped, named)


@pytest.mark.parametrize("value", ["n_fft_samples", [("n_fft_samples", 512)]], ids=repr)
@pytest.mark.parametrize(
    "call",
    [
        pytest.param(
            lambda series, value: multitaper_connectivity(
                series, sampling_frequency=500, method="power", transform_kwargs=value
            ),
            id="multitaper_connectivity-transform_kwargs",
        ),
        pytest.param(
            lambda series, value: multitaper_connectivity(
                series, sampling_frequency=500, method="power", connectivity_kwargs=value
            ),
            id="multitaper_connectivity-connectivity_kwargs",
        ),
        pytest.param(
            lambda series, value: connectivity_to_xarray(
                Multitaper(series, sampling_frequency=500), "power", connectivity_kwargs=value
            ),
            id="connectivity_to_xarray-connectivity_kwargs",
        ),
    ],
)
def test_a_settings_mapping_that_is_not_a_mapping_is_rejected(time_series, call, value):
    """A string or a list of pairs is not silently read as settings."""
    with pytest.raises(TypeError, match=r"_kwargs must be a mapping.*got (str|list)"):
        call(time_series, value)


@pytest.fixture
def no_fft(monkeypatch):
    """Fail the test if any transform's FFT runs."""

    def fail(self):
        msg = "the FFT ran before the call was rejected"
        raise AssertionError(msg)

    monkeypatch.setattr(Multitaper, "fft", fail)


@pytest.mark.parametrize(
    ("connectivity_kwargs", "error", "match"),
    [
        pytest.param({"max_rnk": 2}, TypeError, "max_rnk", id="misspelled"),
        pytest.param({"n_fft_samples": 256}, TypeError, "transform_kwargs", id="transform"),
        pytest.param(
            {"frequency_range": (1.0, 2.0)}, ValueError, "frequency_range", id="named"
        ),
    ],
)
def test_a_bad_connectivity_kwargs_key_fails_before_the_fft(
    time_series, no_fft, connectivity_kwargs, error, match
):
    with pytest.raises(error, match=match):
        multitaper_connectivity(
            time_series,
            sampling_frequency=500,
            method=["power", "global_coherence"],
            connectivity_kwargs=connectivity_kwargs,
        )


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(
            lambda series: multitaper_connectivity(
                series,
                sampling_frequency=500,
                method="power",
                n_fft_samples=1024,
                connectivity_kwargs={"bogus": 1},
            ),
            id="misspelled-measure-setting",
        ),
        pytest.param(
            lambda series: multitaper_connectivity(
                series,
                sampling_frequency=500,
                method="canonical_coherence",
                n_fft_samples=1024,
            ),
            id="missing-group-labels",
        ),
        pytest.param(
            lambda series: multitaper_connectivity(
                series[:, 0, 0], sampling_frequency=500, method="power", n_fft_samples=1024
            ),
            id="one-signal",
        ),
        pytest.param(
            lambda series: connectivity_to_xarray(
                Multitaper(series, sampling_frequency=500),
                "canonical_coherency",
                n_components=1,
            ),
            id="connectivity_to_xarray-missing-group-labels",
        ),
    ],
)
def test_a_rejected_call_does_not_also_warn_about_plain_keywords(time_series, call):
    """The plain-keyword deprecation is only raised for a call that succeeds,
    so it can never hide or precede the error a caller has to fix."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        with pytest.raises((TypeError, ValueError)):
            call(time_series)


@pytest.mark.parametrize(
    ("transform_kwargs", "suggestion"),
    [
        pytest.param(
            {"time_halfbandwith_product": 3}, "time_halfbandwidth_product", id="named"
        ),
        pytest.param({"n_fft": 512}, "n_fft_samples", id="prefix"),
        pytest.param({"taper_weightings": "eigen"}, "taper_weighting", id="advanced"),
    ],
)
def test_a_misspelled_transform_setting_suggests_every_kind_of_setting(
    time_series, transform_kwargs, suggestion
):
    with pytest.raises(TypeError, match=rf"Did you mean.*'{suggestion}'") as excinfo:
        multitaper_connectivity(
            time_series,
            sampling_frequency=500,
            method="power",
            transform_kwargs=transform_kwargs,
        )
    if suggestion == "time_halfbandwidth_product":
        assert "named argument" in str(excinfo.value)


def test_a_misspelled_plain_keyword_suggests_by_prefix(time_series):
    with pytest.raises(TypeError, match=r"Did you mean.*'n_fft_samples'"):
        multitaper_connectivity(time_series, sampling_frequency=500, method="power", n_fft=512)


def test_a_misspelled_measure_setting_suggests_the_measures_own(time_series):
    with pytest.raises(TypeError, match=r"Did you mean.*'n_components'"):
        multitaper_connectivity(
            time_series,
            sampling_frequency=500,
            method="canonical_coherency",
            group_labels=[0, 0, 1],
            connectivity_kwargs={"n_component": 2},
        )


def test_one_measure_is_not_told_its_settings_reach_every_requested_measure(time_series):
    for call in (
        lambda: connectivity_to_xarray(
            Multitaper(time_series, sampling_frequency=500),
            "phase_slope_index",
            connectivity_kwargs={"frequency_rang": (10.0, 20.0)},
        ),
        lambda: multitaper_connectivity(
            time_series,
            sampling_frequency=500,
            method="phase_slope_index",
            connectivity_kwargs={"frequency_rang": (10.0, 20.0)},
        ),
    ):
        with pytest.raises(TypeError, match=r"Did you mean.*'frequency_range'") as excinfo:
            call()
        assert "every requested method" not in str(excinfo.value)
