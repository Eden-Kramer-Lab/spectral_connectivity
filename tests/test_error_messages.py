"""Tests for error message quality and helpfulness.

This module tests that error messages follow the WHAT/WHY/HOW pattern:
- WHAT: Clear statement of the problem
- WHY: Brief explanation of the cause
- HOW: Specific, actionable recovery steps
"""

import functools
import re
import warnings

import numpy as np
import pytest

from spectral_connectivity import (
    Connectivity,
    DirectedOrientationWarning,
    Multitaper,
    multitaper_connectivity,
)
from spectral_connectivity.transforms import (
    MorletWavelet,
    ShortTimeFourierTransform,
    Welch,
    detrend,
)
from spectral_connectivity.wrapper import connectivity_to_xarray, fourier_connectivity
from tests._backend_helpers import to_host


class TestDetrendErrorMessages:
    """Test detrend function error messages follow WHAT/WHY/HOW pattern."""

    def test_invalid_trend_type_error_message(self):
        """Test that invalid trend type error is helpful."""
        rng = np.random.default_rng(0)
        data = rng.standard_normal(100)

        with pytest.raises(ValueError, match="Invalid trend type") as excinfo:
            detrend(data, type="invalid")

        error_msg = str(excinfo.value)

        # WHAT: Should state the problem clearly
        assert (
            "trend type 'invalid' is not supported" in error_msg.lower()
            or "invalid trend type" in error_msg.lower()
        )

        # HOW: Should provide valid options
        assert "linear" in error_msg
        assert "constant" in error_msg

        # Message should be specific about what went wrong
        assert "invalid" in error_msg


class TestExpectationTypeErrorMessages:
    """Test expectation_type parameter error messages."""

    def test_invalid_expectation_type_error_message(self):
        """Test that invalid expectation_type error is helpful."""
        # Create valid 5D fourier coefficients
        rng = np.random.default_rng(0)
        fourier_coefficients = rng.standard_normal(
            (10, 5, 3, 50, 2)
        ) + 1j * rng.standard_normal((10, 5, 3, 50, 2))

        with pytest.raises(
            ValueError, match="Invalid expectation_type 'invalid_type'"
        ) as excinfo:
            Connectivity(fourier_coefficients, expectation_type="invalid_type")

        error_msg = str(excinfo.value)

        # WHAT: Should state what's wrong
        assert "invalid" in error_msg.lower() or "not supported" in error_msg.lower()
        assert "invalid_type" in error_msg

        # HOW: Should list every valid option, one per "  - 'name'" line
        listed = set(re.findall(r"^  - '(\w+)'$", error_msg, flags=re.MULTILINE))
        assert listed == {
            "time",
            "trials",
            "tapers",
            "time_trials",
            "time_tapers",
            "trials_tapers",
            "time_trials_tapers",
        }

    def test_expectation_type_suggests_correct_order(self):
        """Test that wrong order in expectation_type gets helpful suggestion."""
        # Create valid 5D fourier coefficients
        rng = np.random.default_rng(0)
        fourier_coefficients = rng.standard_normal(
            (10, 5, 3, 50, 2)
        ) + 1j * rng.standard_normal((10, 5, 3, 50, 2))

        # Try common mistake: wrong order (e.g., "tapers_trials" instead of "trials_tapers")
        with pytest.raises(
            ValueError, match="Invalid expectation_type 'tapers_trials'"
        ) as excinfo:
            Connectivity(fourier_coefficients, expectation_type="tapers_trials")

        error_msg = str(excinfo.value)

        # Should detect the wrong order and suggest the correct one
        assert "tapers_trials" in error_msg
        assert "Did you mean 'trials_tapers'?" in error_msg


class TestMultitaperParameterErrorMessages:
    """Window/step sizes that resolve to an unusable sample count are rejected."""

    def test_window_duration_rounding_to_zero_is_rejected(self):
        """A positive duration that rounds to 0 samples must raise, not divide by 0."""
        rng = np.random.default_rng(0)
        time_series = rng.standard_normal((1000, 1, 2))
        # 0.0004 s * 1000 Hz = 0.4 samples -> rounds to 0.
        with pytest.raises(
            ValueError, match="n_time_samples_per_window resolved to 0"
        ) as excinfo:
            Multitaper(time_series, sampling_frequency=1000.0, time_window_duration=0.0004)
        error_msg = str(excinfo.value)
        assert "time_window_duration" in error_msg
        assert "1 sample" in error_msg

    def test_window_step_truncating_to_zero_is_rejected(self):
        """A positive step that truncates to 0 samples must raise, not divide by 0."""
        rng = np.random.default_rng(0)
        time_series = rng.standard_normal((1000, 1, 2))
        mt = Multitaper(
            time_series,
            sampling_frequency=1000.0,
            time_window_duration=0.05,
            time_window_step=0.0004,  # -> 0 samples
        )
        with pytest.raises(
            ValueError, match="n_time_samples_per_step resolved to 0"
        ) as excinfo:
            mt.fft()
        error_msg = str(excinfo.value)
        assert "time_window_step" in error_msg

    def test_oversized_window_is_rejected(self):
        """A window longer than the signal must raise instead of returning empty."""
        rng = np.random.default_rng(0)
        time_series = rng.standard_normal((1000, 1, 2))  # 1 s at 1000 Hz
        with pytest.raises(ValueError, match="larger than the signal length") as excinfo:
            Multitaper(time_series, sampling_frequency=1000.0, time_window_duration=5.0)
        error_msg = str(excinfo.value)
        assert "time_window_duration" in error_msg
        assert "larger than the signal" in error_msg


class TestErrorMessagePatterns:
    """Test that error messages follow consistent patterns across the codebase."""

    def test_error_messages_provide_solutions(self):
        """Verify error messages suggest how to fix the problem."""
        # Example: Invalid shape should suggest using Multitaper
        rng = np.random.default_rng(0)
        fourier_coefficients = rng.standard_normal((100, 2))  # Wrong shape

        with pytest.raises(ValueError, match="must be 5-dimensional") as excinfo:
            Connectivity(fourier_coefficients)

        error_msg = str(excinfo.value)

        # Should suggest the correct approach
        assert "Multitaper" in error_msg or "transform" in error_msg.lower()


class TestExplicitSampleCountGuards:
    """Explicit n_time_samples_per_window/step must be validated like durations."""

    def test_explicit_zero_window_is_rejected(self):
        rng = np.random.default_rng(0)
        ts = rng.standard_normal((1000, 1, 2))
        with pytest.raises(ValueError, match="at least 1 sample"):
            Multitaper(ts, sampling_frequency=1000.0, n_time_samples_per_window=0)

    def test_explicit_oversized_window_is_rejected(self):
        rng = np.random.default_rng(0)
        ts = rng.standard_normal((100, 1, 2))
        with pytest.raises(ValueError, match="larger than the signal"):
            Multitaper(ts, sampling_frequency=1000.0, n_time_samples_per_window=500)

    def test_explicit_zero_step_is_rejected(self):
        rng = np.random.default_rng(0)
        ts = rng.standard_normal((1000, 1, 2))
        mt = Multitaper(
            ts,
            sampling_frequency=1000.0,
            n_time_samples_per_window=50,
            n_time_samples_per_step=0,
        )
        with pytest.raises(ValueError, match="at least 1 sample"):
            mt.fft()


def test_sampling_frequency_string_names_the_argument():
    """A rate passed as a string names the argument instead of a raw ufunc error."""
    ts = np.random.default_rng(0).standard_normal((256, 1, 2))
    with pytest.raises(TypeError, match="sampling_frequency must be a number") as excinfo:
        multitaper_connectivity(ts, sampling_frequency="1000", method="power")
    # A string gets its own hint on top of the generic one.
    assert "rather than '1000'" in str(excinfo.value)


# ``True`` is an ``int`` subclass and a complex rate would silently lose its
# imaginary part, so neither is a sampling rate.
@pytest.mark.parametrize(
    "bad_rate",
    [
        "1000",
        np.array("1000"),
        True,
        np.bool_(True),
        1 + 0j,
        np.complex128(500),
        None,
        np.array([500.0]),
    ],
    ids=[
        "str",
        "0d_str_array",
        "bool",
        "numpy_bool",
        "complex",
        "numpy_complex",
        "none",
        "1d_array",
    ],
)
def test_non_scalar_or_non_numeric_sampling_frequency_is_rejected(bad_rate):
    ts = np.random.default_rng(0).standard_normal((256, 1, 2))
    with pytest.raises(TypeError, match="sampling_frequency must be a number") as excinfo:
        Multitaper(ts, sampling_frequency=bad_rate)
    assert "e.g. sampling_frequency=1000" in str(excinfo.value)


@pytest.mark.parametrize(
    "rate",
    [np.array(500.0), np.float32(500), np.int64(500)],
    ids=["0d_array", "float32", "int64"],
)
def test_numpy_scalar_sampling_frequency_is_accepted(rate):
    """A NumPy scalar or 0-d array (e.g. ``dataset["fs"].values``) is a valid rate."""
    ts = np.random.default_rng(0).standard_normal((500, 2, 2))
    transform = Multitaper(ts, sampling_frequency=rate)
    assert type(transform.sampling_frequency) is float
    expected = Multitaper(ts, sampling_frequency=500.0)
    np.testing.assert_allclose(to_host(transform.frequencies), to_host(expected.frequencies))

    result = multitaper_connectivity(
        ts, sampling_frequency=rate, method="magnitude_squared_coherence"
    )
    baseline = multitaper_connectivity(
        ts, sampling_frequency=500.0, method="magnitude_squared_coherence"
    )
    np.testing.assert_allclose(result.frequency, baseline.frequency)
    np.testing.assert_allclose(result.values, baseline.values, equal_nan=True)
    assert result.attrs["mt_sampling_frequency"] == 500.0
    assert type(result.attrs["mt_sampling_frequency"]) is float


@pytest.mark.parametrize(
    ("transform_class", "extra", "prefix"),
    [
        (Multitaper, {}, "mt_"),
        (ShortTimeFourierTransform, {}, "stft_"),
        (Welch, {}, "welch_"),
        (MorletWavelet, {"frequencies": [10.0, 20.0]}, "morlet_"),
    ],
    ids=["multitaper", "stft", "welch", "morlet"],
)
def test_every_transform_records_a_numpy_rate_as_a_float(transform_class, extra, prefix):
    """The rate is normalized to ``float`` on the transform and in provenance,
    so a 0-d array is never serialized as a string or a NumPy scalar."""
    ts = np.random.default_rng(0).standard_normal((500, 2, 2))
    transform = transform_class(ts, sampling_frequency=np.array(500.0), **extra)
    assert type(transform.sampling_frequency) is float
    attrs = connectivity_to_xarray(transform, method="power").attrs
    rate = attrs["sampling_frequency"]
    assert type(rate) is float
    if prefix == "mt_":
        assert attrs["mt_sampling_frequency"] == rate
    else:
        assert not any(key.startswith(prefix) for key in attrs)
    assert rate == 500.0


@pytest.mark.parametrize("measure", ["group_delay", "delay"])
@pytest.mark.parametrize("threshold", [1.0, 0, -0.1, 5, np.nan, True])
def test_delay_measures_name_an_invalid_significance_threshold(
    measure, threshold, monkeypatch
):
    """group_delay and delay reject a significance_threshold outside (0, 1)
    under that name, before any coherency is computed."""

    def no_coherency(self):
        msg = "computed coherency before validating the threshold"
        raise AssertionError(msg)

    monkeypatch.setattr(Connectivity, "_coherency", no_coherency)
    ts = np.random.default_rng(0).standard_normal((500, 5, 2))
    connectivity = Connectivity.from_transform(Multitaper(ts, sampling_frequency=500))
    with pytest.raises(
        ValueError,
        match=r"^significance_threshold must be a finite number strictly between "
        r"0 and 1, got .*e\.g\. 0\.05\.$",
    ):
        getattr(connectivity, measure)(significance_threshold=threshold)


_DETREND_CONSTRUCTORS = {
    "Multitaper": lambda ts, **kw: Multitaper(ts, 500, **kw),
    "STFT": lambda ts, **kw: ShortTimeFourierTransform(ts, 500, **kw),
    "Welch": lambda ts, **kw: Welch(ts, 500, n_time_samples_per_segment=100, **kw),
    "wrapper": lambda ts, **kw: multitaper_connectivity(ts, 500, method="power", **kw),
}


@pytest.mark.parametrize(
    "make", _DETREND_CONSTRUCTORS.values(), ids=list(_DETREND_CONSTRUCTORS)
)
@pytest.mark.parametrize("detrend_type", ["quadratic", "foo", 1])
def test_invalid_detrend_type_raises_at_construction(make, detrend_type):
    """An unknown detrend_type is rejected by name when the transform is
    built, listing the valid choices, not at its first fft()."""
    ts = np.random.default_rng(0).standard_normal((500, 2, 2))
    expected = f"detrend_type must be 'constant', 'linear', or None, got {detrend_type!r}."
    with pytest.raises(ValueError, match="^" + re.escape(expected)):
        make(ts, detrend_type=detrend_type)


@pytest.mark.parametrize(
    "make", _DETREND_CONSTRUCTORS.values(), ids=list(_DETREND_CONSTRUCTORS)
)
@pytest.mark.parametrize("detrend_type", ["constant", "c", "linear", "l", None])
def test_valid_detrend_types_are_accepted(make, detrend_type):
    """The spellings detrend() accepts, and None, construct and transform."""
    ts = np.random.default_rng(0).standard_normal((500, 2, 2))
    result = make(ts, detrend_type=detrend_type)
    if hasattr(result, "fft"):
        assert np.all(np.isfinite(to_host(result.fft())))


_SIGNALS_BY_TIME = np.random.default_rng(1).standard_normal((3, 1000))
_WRAPPER_LAYOUTS = "(n_time_samples, n_trials, n_signals) or (n_time_samples, n_signals)"
_TRANSFORM_LAYOUT = "(n_time_samples, n_trials, n_signals)"


@pytest.mark.parametrize(
    ("call", "expected"),
    [
        pytest.param(
            lambda: multitaper_connectivity(_SIGNALS_BY_TIME, sampling_frequency=500),
            _WRAPPER_LAYOUTS,
            id="wrapper-2d",
        ),
        pytest.param(
            lambda: multitaper_connectivity(
                _SIGNALS_BY_TIME[:, None, :], sampling_frequency=500
            ),
            _WRAPPER_LAYOUTS,
            id="wrapper-3d",
        ),
        pytest.param(
            lambda: multitaper_connectivity(
                _SIGNALS_BY_TIME, sampling_frequency=500, time_window_duration=0.5
            ),
            _WRAPPER_LAYOUTS,
            id="wrapper-window_longer_than_recording",
        ),
        pytest.param(
            lambda: Multitaper(_SIGNALS_BY_TIME[:, None, :], sampling_frequency=500),
            _TRANSFORM_LAYOUT,
            id="Multitaper",
        ),
        pytest.param(
            lambda: ShortTimeFourierTransform(
                _SIGNALS_BY_TIME[:, None, :], 500, time_window_duration=0.5
            ),
            _TRANSFORM_LAYOUT,
            id="STFT",
        ),
    ],
)
def test_transposed_input_names_its_layout(call, expected):
    """A window too short for an array with more signals than time samples is
    reported as a layout problem, with the layout expected and x.T, instead
    of advice to lengthen the window; it raises without warning first."""
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        with pytest.raises(
            ValueError,
            match="^"
            + re.escape(
                f"Your array has 3 time samples and 1000 signals; expected {expected}. "
                "It looks transposed: put time on the first axis, e.g. x.T. Read as "
                "given, its recording is too short to analyze."
            )
            + "$",
        ) as excinfo:
            call()
    assert record == []
    # The window error, with its advice for an array laid out as intended,
    # stays the cause.
    assert isinstance(excinfo.value.__cause__, ValueError)
    assert "window" in str(excinfo.value.__cause__)


def test_short_window_without_transposed_layout_keeps_its_message():
    """An array with more time samples than signals keeps the window advice."""
    with pytest.raises(ValueError, match=r"^time_halfbandwidth_product \(NW=3\)"):
        Multitaper(np.ones((4, 1, 2)), sampling_frequency=500)


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        (
            (200,),
            (
                "time_series has 1 dimension, shape (200,). multitaper_connectivity "
                "needs (n_time_samples, n_signals) or (n_time_samples, n_trials, "
                "n_signals). For one signal use x[:, np.newaxis] with method='power'."
            ),
        ),
        (
            (200, 2, 2, 2),
            (
                "time_series has 4 dimensions, shape (200, 2, 2, 2). "
                "multitaper_connectivity needs (n_time_samples, n_signals) or "
                "(n_time_samples, n_trials, n_signals)."
            ),
        ),
    ],
)
def test_wrapper_names_its_own_layouts_for_another_dimension_count(shape, expected):
    """The wrapper's error speaks of its own 2-D and 3-D layouts, not of
    Multitaper's 3-D input or prepare_time_series."""
    with pytest.raises(ValueError, match="^" + re.escape(expected) + "$"):
        multitaper_connectivity(np.ones(shape), sampling_frequency=500)


_WIDE = np.random.default_rng(5).standard_normal((200, 1, 300))


@pytest.mark.parametrize(
    ("call", "match"),
    [
        pytest.param(
            lambda: Multitaper(_WIDE, 500, start_time="x"),
            "start_time must be a finite scalar",
            id="Multitaper-start_time",
        ),
        pytest.param(
            lambda: Multitaper(_WIDE, 500, start_time=np.nan),
            "start_time must be a finite scalar",
            id="Multitaper-nan_start_time",
        ),
        pytest.param(
            lambda: Multitaper(_WIDE, 500, n_tapers=2.5),
            "n_tapers must be an integer",
            id="Multitaper-n_tapers",
        ),
        pytest.param(
            lambda: ShortTimeFourierTransform(_WIDE, 500, start_time=np.nan),
            "start_time must be a finite scalar",
            id="STFT-start_time",
        ),
        pytest.param(
            lambda: multitaper_connectivity(
                _WIDE[:, 0, :], 500, transform_kwargs={"start_time": "x"}
            ),
            "start_time must be a finite scalar",
            id="wrapper-start_time",
        ),
    ],
)
def test_other_errors_on_a_wide_array_are_not_called_transposed(call, match):
    """Only a recording too short for its window is reported as a layout
    problem; any other error on an array with more signals than time samples
    keeps its own message."""
    with pytest.raises(ValueError, match="^" + match) as excinfo:
        call()
    assert "transposed" not in str(excinfo.value)


_TWO_SIGNALS = np.random.default_rng(2).standard_normal((500, 2))
_METHOD_CALLS = {
    "multitaper_connectivity": lambda method: multitaper_connectivity(
        _TWO_SIGNALS, sampling_frequency=500, method=method
    ),
    "fourier_connectivity": lambda method: fourier_connectivity(
        np.ones((4, 8, 2), dtype=complex), frequencies=np.arange(8.0), method=method
    ),
}


@pytest.mark.parametrize("call", _METHOD_CALLS.values(), ids=list(_METHOD_CALLS))
@pytest.mark.parametrize(
    ("method", "received"),
    [
        (3, "int 3"),
        (b"power", "a bytes holding int 112"),
        (["power", 3], "a list holding int 3"),
        (("power", None), "a tuple holding NoneType None"),
    ],
)
def test_method_of_the_wrong_type_names_the_accepted_types(call, method, received):
    """A method that is not a name or a list of names raises a TypeError saying
    what method may be, giving what was received, and pointing to
    list_measures()."""
    with pytest.raises(TypeError) as excinfo:
        call(method)
    assert str(excinfo.value) == (
        "method must be a measure name (str), a list of names, or None (the "
        f"default measures), got {received}. Call "
        "spectral_connectivity.list_measures() to see the measure names, e.g. "
        "method='magnitude_squared_coherence'."
    )


def _jackknife_connectivity(expectation_type="trials_tapers"):
    ts = np.random.default_rng(3).standard_normal((200, 4, 2))
    return Connectivity.from_transform(
        Multitaper(ts, sampling_frequency=100), expectation_type=expectation_type
    )


@pytest.mark.parametrize(("method", "received"), [(3, "int 3"), (None, "NoneType None")])
def test_jackknife_says_what_method_may_be(method, received):
    """A method that is not a name gets the wrappers' TypeError wording, not
    an AttributeError from str.startswith."""
    with pytest.raises(TypeError) as excinfo:
        _jackknife_connectivity().jackknife(method)
    assert str(excinfo.value) == (
        f"method must be a measure name (str), got {received}. Call "
        "spectral_connectivity.list_measures() to see the measure names, e.g. "
        "method='magnitude_squared_coherence'."
    )


def test_jackknife_suggests_measure_names_for_an_unknown_one():
    """An unknown measure name gets the wrapper's suggestions and a pointer to
    list_measures()."""
    with pytest.raises(ValueError, match="jackknife can compute") as excinfo:
        _jackknife_connectivity().jackknife("coherence")
    message = str(excinfo.value)
    assert message.startswith(
        "'coherence' is not a connectivity measure jackknife can compute. "
        "Did you mean 'magnitude_squared_coherence', "
    )
    assert message.endswith(
        "Call spectral_connectivity.list_measures() to see the measure names."
    )


@pytest.mark.parametrize("confidence_level", [95, 0, 1.0, -0.5, np.nan, True])
def test_jackknife_echoes_an_invalid_confidence_level(confidence_level, monkeypatch):
    """A confidence_level outside (0, 1) is echoed with an example, before any
    measure is computed."""

    def no_measure(self):
        msg = "computed the measure before validating confidence_level"
        raise AssertionError(msg)

    monkeypatch.setattr(Connectivity, "magnitude_squared_coherence", no_measure)
    expected = (
        "confidence_level must be finite and strictly between 0 and 1, got "
        f"{confidence_level!r}. It is the probability mass inside the interval, "
        "e.g. 0.95 for a 95% interval."
    )
    with pytest.raises(ValueError, match="^" + re.escape(expected) + "$"):
        _jackknife_connectivity().jackknife(
            "magnitude_squared_coherence", confidence_level=confidence_level
        )


@pytest.mark.parametrize("expectation_type", ["time_trials_tapers", "time", "time_trials"])
def test_jackknife_names_the_expectation_type_to_build_with(expectation_type):
    """An expectation_type jackknife cannot use names the ones it can."""
    with pytest.raises(ValueError, match="leaves out one trial or taper") as excinfo:
        _jackknife_connectivity(expectation_type).jackknife("magnitude_squared_coherence")
    message = str(excinfo.value)
    assert "expectation_type='trials_tapers' (the default), 'trials', or 'tapers'" in message
    assert f"this one has expectation_type={expectation_type!r}" in message


_RATE_ALIASES = ["fs", "Fs", "sfreq", "srate", "sampling_rate", "sample_rate"]
_SERIES = np.random.default_rng(4).standard_normal((500, 2, 2))
_RATE_CALLS = {
    "multitaper_connectivity": (
        lambda **kw: multitaper_connectivity(_SERIES, **kw),
        "multitaper_connectivity",
    ),
    "Multitaper": (lambda **kw: Multitaper(_SERIES, **kw), "Multitaper"),
    "STFT": (
        lambda **kw: ShortTimeFourierTransform(_SERIES, **kw),
        "ShortTimeFourierTransform",
    ),
    "Welch": (lambda **kw: Welch(_SERIES, **kw), "Welch"),
    "MorletWavelet": (
        lambda **kw: MorletWavelet(_SERIES, frequencies=[10.0], **kw),
        "MorletWavelet",
    ),
}


@pytest.mark.parametrize(("call", "name"), _RATE_CALLS.values(), ids=list(_RATE_CALLS))
@pytest.mark.parametrize("alias", _RATE_ALIASES)
def test_sampling_rate_aliases_point_to_sampling_frequency(call, name, alias):
    """Other packages' names for the sampling rate, which no string metric
    relates to sampling_frequency, are answered with its name."""
    with pytest.raises(
        TypeError,
        match="^"
        + re.escape(
            f"{name}() got an unexpected keyword argument {alias!r}. "
            "Did you mean sampling_frequency (in Hz)?"
        ),
    ):
        call(**{alias: 500})


def test_constructors_keep_their_signatures():
    """The alias check leaves each transform's signature intact for inspect."""
    import inspect

    for transform in (Multitaper, ShortTimeFourierTransform, Welch, MorletWavelet):
        parameters = inspect.signature(transform).parameters
        assert list(parameters)[:2] == ["time_series", "sampling_frequency"]


_TRANSFORMS = {
    "Multitaper": lambda ts: Multitaper(ts, 500),
    "STFT": lambda ts: ShortTimeFourierTransform(ts, 500),
    "Welch": lambda ts: Welch(ts, 500),
    "MorletWavelet": lambda ts: MorletWavelet(ts, 500, frequencies=[10.0]),
}
_LAYOUT = "(n_time_samples, n_trials, n_signals)"


@pytest.mark.parametrize("make", _TRANSFORMS.values(), ids=list(_TRANSFORMS))
def test_transforms_name_a_ragged_nested_list(make):
    """A nested list with rows of different lengths raises naming the array
    each transform takes, not NumPy's inhomogeneous-shape message."""
    ragged = [[[1.0, 2.0]], [[3.0]]]
    with pytest.raises(
        ValueError,
        match="^"
        + re.escape(
            f"time_series must be a numeric array of shape {_LAYOUT}, but NumPy "
            "cannot make it one array: its nested lists have rows of different "
            "lengths. Give every signal (and trial) the same number of samples"
        ),
    ):
        make(ragged)


@pytest.mark.parametrize("make", _TRANSFORMS.values(), ids=list(_TRANSFORMS))
@pytest.mark.parametrize(
    ("values", "received"),
    [
        ([[["a", "b"]]] * 8, "values of dtype <U1"),
        (np.full((8, 1, 2), "a"), "values of dtype <U1"),
        (None, "NoneType None"),
        (np.full((8, 1, 2), None), "values of dtype object"),
    ],
    ids=["list", "array", "none", "object_array_of_none"],
)
def test_transforms_name_values_that_are_not_numbers(make, values, received):
    """Strings or None raise naming the array and the conversion, not a ufunc
    error (or, for Welch given None, a message about its segment length)."""
    with pytest.raises(
        TypeError,
        match="^"
        + re.escape(
            f"time_series must be a numeric array of shape {_LAYOUT}, got {received}, "
            "not numbers. Convert them first, e.g. np.asarray(time_series, dtype=float)."
        )
        + "$",
    ):
        make(values)


@pytest.mark.parametrize("make", _TRANSFORMS.values(), ids=list(_TRANSFORMS))
def test_an_object_array_of_numbers_is_read_as_its_numbers(make):
    """Every transform reads an object array holding floats as those floats:
    MorletWavelet accepted one, and Multitaper, STFT, and Welch failed with
    NumPy's isfinite error."""
    values = np.random.default_rng(9).standard_normal((300, 2, 2))
    np.testing.assert_array_equal(
        to_host(make(values.astype(object)).fft()), to_host(make(values).fft())
    )


def test_the_wrapper_reads_an_object_array_of_numbers():
    values = np.random.default_rng(9).standard_normal((300, 2))
    expected = multitaper_connectivity(values, 500, method="power")
    result = multitaper_connectivity(values.astype(object), 500, method="power")
    np.testing.assert_array_equal(result.values, expected.values)
    coefficients = np.random.default_rng(10).standard_normal((4, 8, 2)) * (1 + 1j)
    frequencies = np.arange(8.0)
    np.testing.assert_array_equal(
        fourier_connectivity(coefficients.astype(object), frequencies, method="power").values,
        fourier_connectivity(coefficients, frequencies, method="power").values,
    )


def test_fourier_connectivity_names_a_ragged_nested_list():
    """The coefficients' list path names its layouts for a ragged list."""
    ragged = [[[1 + 1j, 2j], [3j]]]
    with pytest.raises(
        ValueError,
        match="^"
        + re.escape(
            "fourier_coefficients must be a numeric array of shape (observation, "
            "frequency, signal), (trial, taper, frequency, signal), or (time, trial, "
            "taper, frequency, signal), but NumPy cannot make it one array"
        ),
    ):
        fourier_connectivity(ragged, frequencies=np.arange(2.0))


def _series_with_nan():
    series = np.random.default_rng(6).standard_normal((1000, 3, 3))
    series[10, 1, 0] = np.nan
    series[500, 2, 2] = np.inf
    return series


def _coefficients_with_nan():
    coefficients = Multitaper(np.random.default_rng(7).standard_normal((400, 3, 3)), 500).fft()
    coefficients[0, 1, 0, 5, 0] = np.nan
    return coefficients


@pytest.mark.parametrize(
    ("call", "expected"),
    [
        pytest.param(
            lambda: multitaper_connectivity(
                _series_with_nan(), 500, method="pairwise_spectral_granger_prediction"
            ),
            "time_series contains 2 NaN or infinite values",
            id="wrapper-pairwise_granger",
        ),
        pytest.param(
            lambda: multitaper_connectivity(
                _series_with_nan(), 500, method="directed_transfer_function"
            ),
            "time_series contains 2 NaN or infinite values",
            id="wrapper-DTF",
        ),
        pytest.param(
            lambda: Connectivity.from_transform(
                Multitaper(_series_with_nan(), 500)
            ).partial_directed_coherence(),
            "time_series contains 2 NaN or infinite values",
            id="from_transform-PDC",
        ),
        pytest.param(
            lambda: Connectivity(
                _coefficients_with_nan()
            ).conditional_spectral_granger_prediction(),
            "fourier_coefficients contains NaN or Inf values.",
            id="Connectivity-conditional_granger",
        ),
    ],
)
def test_directed_measures_on_nonfinite_input_give_only_the_nan_warning(call, expected, capfd):
    """The factorization's failure warnings (Cholesky fallback, Wilson
    non-convergence, NaN at every frequency) blamed duplicated channels for
    the NaN the user was already warned about; only that warning (and the
    orientation reminder) remains, and nothing is logged to stderr."""
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        call()
    messages = [
        str(w.message)
        for w in record
        if not issubclass(w.category, DirectedOrientationWarning)
    ]
    assert len(messages) == 1
    assert messages[0].startswith(expected)
    assert capfd.readouterr().err == ""


def test_directed_measures_on_finite_collinear_input_still_warn():
    """With finite coefficients the factorization warnings are the only sign
    of a duplicated channel, so they stay."""
    series = np.random.default_rng(8).standard_normal((400, 3, 3))
    series[..., 2] = series[..., 0]
    connectivity = Connectivity.from_transform(Multitaper(series, 500))
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        connectivity.pairwise_spectral_granger_prediction()
    assert any(
        "duplicated" in str(w.message)
        for w in record
        if not issubclass(w.category, DirectedOrientationWarning)
    )


@pytest.mark.parametrize(
    ("measure", "kwargs"),
    [
        ("canonical_coherence", {"group_labels": np.array([0, 0, 1])}),
        ("global_coherence", {}),
    ],
)
@pytest.mark.parametrize("route", ["Connectivity", "wrapper"])
def test_decompositions_name_nonfinite_coefficients(measure, kwargs, route):
    """canonical_coherence and global_coherence raised LAPACK's LinAlgError
    on NaN or Inf; they say the coefficients hold NaN or Inf and what to do.
    (The other multivariate kernels already return NaN for those bins.)"""
    with warnings.catch_warnings():
        # The NaN warnings, and NumPy's on the class path (see the CHANGELOG).
        warnings.simplefilter("ignore")
        if route == "Connectivity":
            compute = getattr(Connectivity(_coefficients_with_nan()), measure)
        else:
            compute = functools.partial(
                multitaper_connectivity,
                _series_with_nan(),
                500,
                method=measure,
                group_labels=kwargs.pop("group_labels", None),
            )
        with pytest.raises(
            ValueError,
            match=f"^{measure} cannot be computed: the Fourier coefficients contain "
            "NaN or Inf.*Interpolate or remove the NaN or Inf samples",
        ) as excinfo:
            compute(**kwargs)
    assert isinstance(excinfo.value.__cause__, np.linalg.LinAlgError)
