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

from spectral_connectivity import (
    Connectivity,
    Multitaper,
    fourier_connectivity,
    multitaper_connectivity,
)
from spectral_connectivity.wrapper import connectivity_to_xarray
from tests._deprecation_helpers import warns_once_at_the_call

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
    legacy, message = warns_once_at_the_call(
        lambda: multitaper_connectivity(
            time_series, sampling_frequency=500, method="power", **settings
        ),
        "transform_kwargs",
    )
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


def test_an_unknown_legacy_keyword_is_rejected_before_any_warning(time_series):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(TypeError, match=r"'pairs'.*connectivity_kwargs"):
            multitaper_connectivity(
                time_series, sampling_frequency=500, method="power", pairs=[(0, 1)]
            )


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
    settings = {"frequency_resolution": 10.0}
    canonical = connectivity_to_xarray(
        multitaper, "phase_slope_index", connectivity_kwargs=settings
    )
    legacy, _ = warns_once_at_the_call(
        lambda: connectivity_to_xarray(multitaper, "phase_slope_index", **settings),
        "connectivity_kwargs",
    )
    _assert_identical(legacy, canonical)


def test_connectivity_to_xarray_rejects_a_measure_setting_given_twice(multitaper):
    with pytest.raises(
        ValueError, match=r"frequency_resolution.*connectivity_kwargs.*keyword"
    ):
        connectivity_to_xarray(
            multitaper,
            "phase_slope_index",
            connectivity_kwargs={"frequency_resolution": 10.0},
            frequency_resolution=20.0,
        )


def test_connectivity_to_xarray_takes_frequency_range_by_name_like_the_wrappers(
    multitaper,
):
    """frequency_range is a named argument of every wrapper: the interval
    measures consume it, any other measure is cropped to it, and it does not
    go in connectivity_kwargs."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        psi = connectivity_to_xarray(
            multitaper, "phase_slope_index", frequency_range=(20.0, 80.0)
        )
        cropped = connectivity_to_xarray(
            multitaper, "magnitude_squared_coherence", frequency_range=(20.0, 80.0)
        )
    full = connectivity_to_xarray(multitaper, "magnitude_squared_coherence")
    expected = Connectivity.from_transform(multitaper).phase_slope_index(
        frequency_range=(20.0, 80.0)
    )
    np.testing.assert_array_equal(psi.values, expected)
    in_range = (full.frequency >= 20.0) & (full.frequency <= 80.0)
    np.testing.assert_array_equal(cropped.values, full.values[:, in_range.values])
    np.testing.assert_array_equal(cropped.frequency, full.frequency[in_range])
    assert cropped.attrs["frequency_interval_closed"] == "both"
    assert cropped.attrs["requested_frequency_lower"] == 20.0
    with pytest.raises(ValueError, match="pass frequency_range="):
        connectivity_to_xarray(
            multitaper,
            "phase_slope_index",
            connectivity_kwargs={"frequency_range": (20.0, 80.0)},
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
    """A string or a list of pairs is not silently read as settings, and the
    example in the message is a setting of that mapping's kind."""
    with pytest.raises(
        TypeError, match=r"_kwargs must be a mapping.*got (str|list)"
    ) as excinfo:
        call(time_series, value)
    message = str(excinfo.value)
    if message.startswith("transform_kwargs"):
        assert "{'n_fft_samples': 1024}" in message
    else:
        assert message.startswith("connectivity_kwargs")
        assert "{'pairs': [(0, 1)]}" in message


@pytest.fixture
def no_fft(monkeypatch):
    """Fail the test if any transform's FFT runs."""

    def fail(self):
        msg = "the FFT ran before the call was rejected"
        raise AssertionError(msg)

    monkeypatch.setattr(Multitaper, "fft", fail)


def test_two_sided_power_is_refused_for_a_labelled_result(time_series, no_fft):
    """A labelled power result is one-sided: its attributes say so and its
    band integrals rely on it. The two-sided density is ``Connectivity.power``'s
    to give, and the call is refused before the FFT."""
    with pytest.raises(ValueError, match=r"Connectivity\.power\(one_sided=False\)"):
        multitaper_connectivity(
            time_series,
            sampling_frequency=500,
            method="power",
            connectivity_kwargs={"one_sided": False},
        )


def test_one_sided_power_can_be_asked_for_by_name(time_series):
    """``one_sided=True`` is the labelled result as it always is."""
    explicit = multitaper_connectivity(
        time_series,
        sampling_frequency=500,
        method="power",
        connectivity_kwargs={"one_sided": True},
    )
    default = multitaper_connectivity(time_series, sampling_frequency=500, method="power")
    np.testing.assert_array_equal(explicit.values, default.values)


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
        pytest.param({"n_fft_sample": 2048}, "n_fft_samples", id="close"),
    ],
)
def test_a_misspelled_transform_setting_suggests_every_kind_of_setting(
    time_series, transform_kwargs, suggestion
):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(TypeError, match=rf"Did you mean.*'{suggestion}'") as excinfo:
            multitaper_connectivity(
                time_series,
                sampling_frequency=500,
                method="power",
                transform_kwargs=transform_kwargs,
            )
    if suggestion == "time_halfbandwidth_product":
        assert "named argument" in str(excinfo.value)


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


def test_a_self_key_gets_the_packages_own_message(time_series):
    """``self`` is not a measure setting, so it is rejected like any unknown
    key instead of reaching Python's "multiple values for argument" error."""
    with pytest.raises(TypeError, match=r"power does not accept keyword argument\(s\) 'self'"):
        multitaper_connectivity(
            time_series,
            sampling_frequency=500,
            method="power",
            connectivity_kwargs={"self": 1},
        )


def test_fourier_connectivity_checks_settings_before_the_coefficients():
    """Real-valued coefficients would be rejected when Connectivity is built;
    the misspelled key is reported first, before they are copied."""
    with pytest.raises(TypeError, match="does not accept keyword argument"):
        fourier_connectivity(
            np.ones((4, 8, 2)),
            is_one_sided=False,
            method="power",
            connectivity_kwargs={"bogus": 1},
        )


_LABEL_ROUTING_ERRORS = {
    "both-routes": (
        {"group_labels": [0, 0, 1], "connectivity_kwargs": {"group_labels": [0, 0, 1]}},
        "canonical_coherence",
        "group_labels was given both",
    ),
    "no-group-measure": (
        {"group_labels": [0, 0, 1]},
        "magnitude_squared_coherence",
        "none of the requested measures compares groups",
    ),
}


@pytest.mark.parametrize("case", sorted(_LABEL_ROUTING_ERRORS))
def test_a_misrouted_group_labels_fails_before_the_fft(time_series, multitaper, no_fft, case):
    """Labels given two ways, or given with no group measure, need no signal
    count, so every wrapper rejects them before any spectral work: no
    transform FFT, and for fourier_connectivity before the coefficients are
    checked (these are real-valued, which Connectivity would reject)."""
    kwargs, method, match = _LABEL_ROUTING_ERRORS[case]
    with pytest.raises(ValueError, match=match):
        multitaper_connectivity(time_series, sampling_frequency=500, method=method, **kwargs)
    with pytest.raises(ValueError, match=match):
        connectivity_to_xarray(multitaper, method, **kwargs)
    with pytest.raises(ValueError, match=match):
        fourier_connectivity(np.ones((4, 8, 3)), is_one_sided=False, method=method, **kwargs)
