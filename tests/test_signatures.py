"""Published call signatures: released positional orders and keyword-only settings.

The released positional orders below are copied from the v2.0.1 sources; a
change to any of them would silently rebind existing positional calls.
"""

import inspect

import numpy as np
import pytest

from spectral_connectivity import (
    MorletWavelet,
    Multitaper,
    ShortTimeFourierTransform,
    Welch,
)

_POSITIONAL = (
    inspect.Parameter.POSITIONAL_ONLY,
    inspect.Parameter.POSITIONAL_OR_KEYWORD,
)


def _positional_names(function):
    return [
        name
        for name, parameter in inspect.signature(function).parameters.items()
        if parameter.kind in _POSITIONAL and name != "self"
    ]


def _keyword_only_names(function):
    return [
        name
        for name, parameter in inspect.signature(function).parameters.items()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY
    ]


def test_multitaper_keeps_its_released_positional_order():
    assert _positional_names(Multitaper)[:13] == [
        "time_series",
        "sampling_frequency",
        "time_halfbandwidth_product",
        "detrend_type",
        "time_window_duration",
        "time_window_step",
        "n_tapers",
        "tapers",
        "start_time",
        "n_fft_samples",
        "n_time_samples_per_window",
        "n_time_samples_per_step",
        "is_low_bias",
    ]


def test_multitaper_positional_call_binds_the_released_parameters():
    time_series = np.random.default_rng(0).standard_normal((400, 2, 2))
    positional = Multitaper(time_series, 200, 2, "linear", 1.0, 0.5, 2)
    keyword = Multitaper(
        time_series,
        sampling_frequency=200,
        time_halfbandwidth_product=2,
        detrend_type="linear",
        time_window_duration=1.0,
        time_window_step=0.5,
        n_tapers=2,
    )
    assert positional.time_halfbandwidth_product == 2
    assert positional.detrend_type == "linear"
    assert positional.n_tapers == 2
    np.testing.assert_array_equal(positional.fft(), keyword.fft())


@pytest.mark.parametrize(
    "transform", [ShortTimeFourierTransform, Welch, MorletWavelet], ids=lambda t: t.__name__
)
def test_new_transforms_take_only_data_and_rate_positionally(transform):
    assert _positional_names(transform) == ["time_series", "sampling_frequency"]
    assert _keyword_only_names(transform)


@pytest.mark.parametrize(
    ("make", "third"),
    [
        pytest.param(ShortTimeFourierTransform, "constant", id="ShortTimeFourierTransform"),
        pytest.param(Welch, 0.5, id="Welch"),
        pytest.param(MorletWavelet, [10.0, 20.0], id="MorletWavelet"),
    ],
)
def test_new_transforms_reject_a_positional_setting(make, third):
    time_series = np.zeros((200, 1, 2))
    with pytest.raises(TypeError, match="positional argument"):
        make(time_series, 100, third)
