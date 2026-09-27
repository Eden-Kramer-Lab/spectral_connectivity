"""Tests for error message quality and helpfulness.

This module tests that error messages follow the WHAT/WHY/HOW pattern:
- WHAT: Clear statement of the problem
- WHY: Brief explanation of the cause
- HOW: Specific, actionable recovery steps
"""

import re

import numpy as np
import pytest

from spectral_connectivity import Connectivity, Multitaper, multitaper_connectivity
from spectral_connectivity.transforms import (
    MorletWavelet,
    ShortTimeFourierTransform,
    Welch,
    detrend,
)
from spectral_connectivity.wrapper import connectivity_to_xarray


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
        mt = Multitaper(time_series, sampling_frequency=1000.0, time_window_duration=0.0004)
        with pytest.raises(
            ValueError, match="n_time_samples_per_window resolved to 0"
        ) as excinfo:
            mt.fft()
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
        mt = Multitaper(time_series, sampling_frequency=1000.0, time_window_duration=5.0)
        with pytest.raises(ValueError, match="larger than the signal length") as excinfo:
            mt.fft()
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
        mt = Multitaper(ts, sampling_frequency=1000.0, n_time_samples_per_window=0)
        with pytest.raises(ValueError, match="at least 1 sample"):
            mt.fft()

    def test_explicit_oversized_window_is_rejected(self):
        rng = np.random.default_rng(0)
        ts = rng.standard_normal((100, 1, 2))
        mt = Multitaper(ts, sampling_frequency=1000.0, n_time_samples_per_window=500)
        with pytest.raises(ValueError, match="larger than the signal"):
            mt.fft()

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
    np.testing.assert_allclose(transform.frequencies, expected.frequencies)

    result = multitaper_connectivity(ts, sampling_frequency=rate, method="coherence_magnitude")
    baseline = multitaper_connectivity(
        ts, sampling_frequency=500.0, method="coherence_magnitude"
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
    rate = attrs[f"{prefix}sampling_frequency"]
    assert type(rate) is float
    assert rate == 500.0
