import warnings

import numpy as np
import pytest
from nitime.algorithms.spectral import dpss_windows as nitime_dpss_windows
from pytest import mark

from spectral_connectivity.connectivity import Connectivity
from spectral_connectivity.transforms import (
    MorletWavelet,
    Multitaper,
    ShortTimeFourierTransform,
    Welch,
    _add_axes,
    _get_low_bias_tapers,
    _multitaper_fft,
    _sliding_window,
    dpss_windows,
)


def test__add_axes():
    # Add dimension if no trials
    n_time_samples, n_signals = (2, 3)
    test_data = np.ones((n_time_samples, n_signals))
    expected_shape = (n_time_samples, 1, n_signals)
    assert np.allclose(_add_axes(test_data).shape, expected_shape)

    # Add two dimensions if no trials and signals
    test_data = np.ones((n_time_samples,))
    expected_shape = (n_time_samples, 1, 1)
    assert np.allclose(_add_axes(test_data).shape, expected_shape)

    # if there is a trial dimension, do nothing
    n_trials = 10
    test_data = np.ones((n_time_samples, n_trials, n_signals))
    expected_shape = (n_time_samples, n_trials, n_signals)
    assert np.allclose(_add_axes(test_data).shape, expected_shape)


@mark.parametrize(
    "test_array, window_size, step_size, axis, expected_array",
    [
        (np.arange(1, 6), 3, 1, -1, np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])),
        (np.arange(1, 6), 3, 2, -1, np.array([[1, 2, 3], [3, 4, 5]])),
        (
            np.arange(0, 6).reshape((2, 3)),
            2,
            1,
            0,
            np.array([[[0, 3], [1, 4], [2, 5]]]),
        ),
        # Negative axis on a 2-D array: windows run along the last axis and the
        # window dimension is appended, so a negative index must be normalized
        # before the step is applied.
        (
            np.arange(0, 6).reshape((2, 3)),
            2,
            1,
            -1,
            np.array([[[0, 1], [1, 2]], [[3, 4], [4, 5]]]),
        ),
        # Step larger than one along a non-default axis.
        (
            np.arange(0, 10),
            3,
            4,
            0,
            np.array([[0, 1, 2], [4, 5, 6]]),
        ),
    ],
)
def test__sliding_window(test_array, window_size, step_size, axis, expected_array):
    assert np.allclose(
        _sliding_window(
            test_array, window_size=window_size, step_size=step_size, axis=axis
        ),
        expected_array,
    )


@mark.parametrize("axis", [2, 3, -3, -4])
def test__sliding_window_rejects_out_of_range_axis(axis):
    """An out-of-range axis must raise, not wrap onto a real dimension.

    ``axis % data.ndim`` would silently window the wrong dimension (e.g.
    ``axis=2`` -> 0 on a 2-D array) instead of raising.
    """
    data = np.arange(6).reshape((2, 3))
    with pytest.raises(ValueError, match="out of bounds"):
        _sliding_window(data, window_size=2, axis=axis)


@mark.parametrize("step_size", [0, -1, -2])
def test__sliding_window_rejects_non_positive_step(step_size):
    """A non-positive step is not a forward slide and must raise.

    A negative step would otherwise reverse the window order via the slice.
    """
    with pytest.raises(ValueError, match="step_size must be a positive integer"):
        _sliding_window(np.arange(6), window_size=2, step_size=step_size)


@mark.parametrize(
    "time_halfbandwidth_product, expected_n_tapers", [(3, 5), (1, 1), (1.75, 2)]
)
def test_n_tapers(time_halfbandwidth_product, expected_n_tapers):
    n_time_samples, n_trials, n_signals = 100, 10, 2
    time_series = np.zeros((n_time_samples, n_trials, n_signals))
    m = Multitaper(
        time_series=time_series, time_halfbandwidth_product=time_halfbandwidth_product
    )
    assert m.n_tapers == expected_n_tapers


@mark.parametrize(
    "sampling_frequency, time_window_duration, expected_duration",
    [(1000, None, 0.1), (2000, None, 0.05), (1000, 0.1, 0.1)],
)
def test_time_window_duration(
    sampling_frequency, time_window_duration, expected_duration
):
    n_time_samples, n_trials, n_signals = 100, 10, 2
    time_series = np.zeros((n_time_samples, n_trials, n_signals))
    m = Multitaper(
        time_series=time_series,
        sampling_frequency=sampling_frequency,
        time_window_duration=time_window_duration,
    )
    assert m.time_window_duration == expected_duration


@mark.parametrize(
    "sampling_frequency, time_window_step, expected_step",
    [(1000, None, 0.1), (2000, None, 0.05), (1000, 0.1, 0.1)],
)
def test_time_window_step(sampling_frequency, time_window_step, expected_step):
    n_time_samples, n_trials, n_signals = 100, 10, 2
    time_series = np.zeros((n_time_samples, n_trials, n_signals))
    m = Multitaper(
        time_series=time_series,
        sampling_frequency=sampling_frequency,
        time_window_step=time_window_step,
    )
    assert m.time_window_step == expected_step


@mark.parametrize(
    ("sampling_frequency, time_window_duration,expected_n_time_samples_per_window"),
    [(1000, None, 100), (1000, 0.1, 100), (2000, 0.025, 50)],
)
def test_n_time_samples(
    sampling_frequency, time_window_duration, expected_n_time_samples_per_window
):
    n_time_samples, n_trials, n_signals = 100, 10, 2
    time_series = np.zeros((n_time_samples, n_trials, n_signals))
    m = Multitaper(
        time_series=time_series,
        sampling_frequency=sampling_frequency,
        time_window_duration=time_window_duration,
    )
    assert m.n_time_samples_per_window == expected_n_time_samples_per_window


@mark.parametrize(
    ("sampling_frequency, time_window_duration, n_fft_samples,expected_n_fft_samples"),
    [(1000, None, 128, 128), (1000, 0.1, None, 100)],
)
def test_n_fft_samples(
    sampling_frequency, time_window_duration, n_fft_samples, expected_n_fft_samples
):
    n_time_samples, n_trials, n_signals = 100, 10, 2
    time_series = np.zeros((n_time_samples, n_trials, n_signals))
    m = Multitaper(
        time_series=time_series,
        sampling_frequency=sampling_frequency,
        time_window_duration=time_window_duration,
        n_fft_samples=n_fft_samples,
    )
    assert m.n_fft_samples == expected_n_fft_samples


def test_n_fft_samples_smaller_than_window_raises():
    """n_fft_samples < window length would silently truncate the signal."""
    n_time_samples, n_trials, n_signals = 100, 10, 2
    time_series = np.zeros((n_time_samples, n_trials, n_signals))
    m = Multitaper(
        time_series=time_series,
        sampling_frequency=1000,
        n_fft_samples=5,  # window is 100 samples
    )
    with pytest.raises(ValueError, match="n_fft_samples"):
        _ = m.n_fft_samples


def test_frequencies():
    # Window length must not exceed n_fft_samples, so use a 4-sample window.
    n_time_samples, n_trials, n_signals = 4, 10, 2
    time_series = np.zeros((n_time_samples, n_trials, n_signals))
    n_fft_samples = 4
    sampling_frequency = 1000
    m = Multitaper(
        time_series=time_series,
        sampling_frequency=sampling_frequency,
        n_fft_samples=n_fft_samples,
    )
    expected_frequencies = np.array([0, 250, -500, -250])
    assert np.allclose(m.frequencies, expected_frequencies)


def test_n_signals():
    n_time_samples, n_trials, n_signals = 100, 10, 2
    time_series = np.zeros((n_time_samples, n_trials, n_signals))
    m = Multitaper(time_series=time_series)
    assert m.n_signals == n_signals


def test_n_trials():
    n_time_samples, n_trials, n_signals = 100, 10, 2
    time_series = np.zeros((n_time_samples, n_trials, n_signals))
    m = Multitaper(time_series=time_series)
    assert m.n_trials == n_trials

    # Test with 2D input converted using prepare_time_series
    from spectral_connectivity.transforms import prepare_time_series

    time_series_2d = np.zeros((n_time_samples, n_signals))
    time_series_3d = prepare_time_series(time_series_2d, axis="signals")
    m = Multitaper(time_series=time_series_3d)
    assert m.n_trials == 1


@mark.parametrize(
    ("time_halfbandwidth_product, time_window_duration, expected_frequency_resolution"),
    [(3, 0.10, 60), (1, 0.02, 100), (5, 1, 10)],
)
def test_frequency_resolution(
    time_halfbandwidth_product, time_window_duration, expected_frequency_resolution
):
    n_time_samples, n_trials, n_signals = 100, 10, 2
    time_series = np.zeros((n_time_samples, n_trials, n_signals))
    m = Multitaper(
        time_series=time_series,
        time_halfbandwidth_product=time_halfbandwidth_product,
        time_window_duration=time_window_duration,
    )
    assert m.frequency_resolution == expected_frequency_resolution


@mark.parametrize(
    ("time_window_step, n_time_samples_per_step, expected_n_samples_per_time_step"),
    [(None, None, 100), (0.001, None, 1), (0.002, None, 2), (None, 10, 10)],
)
def test_n_samples_per_time_step(
    time_window_step, n_time_samples_per_step, expected_n_samples_per_time_step
):
    n_time_samples, n_trials, n_signals = 100, 10, 2
    time_series = np.zeros((n_time_samples, n_trials, n_signals))

    m = Multitaper(
        time_window_duration=0.10,
        n_time_samples_per_step=n_time_samples_per_step,
        time_series=time_series,
        time_window_step=time_window_step,
    )
    assert m.n_time_samples_per_step == expected_n_samples_per_time_step


@mark.parametrize("time_window_duration", [0.1, 0.2, 2.4, 0.16])
def test_time(time_window_duration):
    sampling_frequency = 1500
    start_time, end_time = -2.4, 2.4
    n_trials, n_signals = 10, 2
    n_time_samples = int((end_time - start_time) * sampling_frequency) + 1
    time_series = np.zeros((n_time_samples, n_trials, n_signals))
    expected_time = np.arange(start_time, end_time, time_window_duration)
    if not np.allclose(expected_time[-1] + time_window_duration, end_time):
        expected_time = expected_time[:-1]
    # Windows are labeled by their center time, not their start.
    expected_time = expected_time + (
        round(time_window_duration * sampling_frequency) - 1
    ) / (2 * sampling_frequency)
    m = Multitaper(
        sampling_frequency=sampling_frequency,
        time_series=time_series,
        start_time=start_time,
        time_window_duration=time_window_duration,
    )
    assert np.allclose(m.time, expected_time)


def test_tapers():
    n_time_samples, n_trials, n_signals = 100, 10, 2
    time_series = np.zeros((n_time_samples, n_trials, n_signals))
    m = Multitaper(time_series, is_low_bias=False)
    assert np.allclose(m.tapers.shape, (n_time_samples, m.n_tapers))

    m = Multitaper(time_series, tapers=np.zeros((10, 3)))
    assert np.allclose(m.tapers.shape, (10, 3))


@mark.parametrize(
    "eigenvalues, expected_n_tapers",
    [
        (np.array([0.95, 0.95, 0.95]), 3),
        (np.array([0.95, 0.8, 0.95]), 2),
        (np.array([0.8, 0.8, 0.8]), 1),
    ],
)
def test__get_low_bias_tapers(eigenvalues, expected_n_tapers):
    tapers = np.zeros((3, 100))
    filtered_tapers, filtered_eigenvalues = _get_low_bias_tapers(tapers, eigenvalues)
    assert (
        filtered_tapers.shape[0] == filtered_eigenvalues.shape[0] == expected_n_tapers
    )


@mark.parametrize(
    "n_time_samples, time_halfbandwidth_product, n_tapers",
    [(1000, 3, 5), (31, 6, 4), (31, 7, 4)],
)
def test_dpss_windows(n_time_samples, time_halfbandwidth_product, n_tapers):
    tapers, eigenvalues = dpss_windows(
        n_time_samples, time_halfbandwidth_product, n_tapers, is_low_bias=False
    )
    nitime_tapers, nitime_eigenvalues = nitime_dpss_windows(
        n_time_samples, time_halfbandwidth_product, n_tapers
    )
    assert np.allclose(np.sum(tapers**2, axis=1), 1.0)
    assert np.allclose(tapers, nitime_tapers)
    assert np.allclose(eigenvalues, nitime_eigenvalues)


def test__multitaper_fft():
    n_windows, n_trials, n_time_samples, n_tapers, n_fft_samples = (2, 10, 100, 3, 100)
    sampling_frequency = 1000
    time_series = np.ones((n_windows, n_trials, n_time_samples))
    tapers = np.ones((n_time_samples, n_tapers))

    fourier_coefficients = _multitaper_fft(
        tapers, time_series, n_fft_samples, sampling_frequency
    )
    assert np.allclose(
        fourier_coefficients.shape, (n_windows, n_trials, n_fft_samples, n_tapers)
    )


def test_fft():
    n_time_samples, n_trials, n_signals, n_windows = 100, 10, 2, 1
    time_series = np.zeros((n_time_samples, n_trials, n_signals))
    m = Multitaper(time_series=time_series)
    assert np.allclose(
        m.fft().shape,
        (n_windows, n_trials, m.tapers.shape[1], m.n_fft_samples, n_signals),
    )


def test_multitaper_requires_3d_input():
    """Test that Multitaper requires 3D input array."""
    rng = np.random.default_rng(42)
    # 1D input should raise ValueError
    time_series_1d = rng.standard_normal(100)
    with pytest.raises(ValueError, match=r"Expected 3D array.*got 1D"):
        Multitaper(time_series=time_series_1d)

    # 2D input should raise ValueError with helpful message
    time_series_2d = rng.standard_normal((100, 5))
    with pytest.raises(
        ValueError,
        match=r"Expected 3D array.*got 2D.*Use prepare_time_series|np.newaxis",
    ):
        Multitaper(time_series=time_series_2d)

    # 4D input should raise ValueError
    time_series_4d = rng.standard_normal((100, 10, 5, 3))
    with pytest.raises(ValueError, match=r"Expected 3D array.*got 4D"):
        Multitaper(time_series=time_series_4d)

    # 3D input should work
    time_series_3d = rng.standard_normal((100, 10, 5))
    m = Multitaper(time_series=time_series_3d)
    assert m.time_series.ndim == 3


def test_prepare_time_series_single_trial():
    """Test helper function for converting 2D (time, signals) to 3D."""
    rng = np.random.default_rng(42)
    from spectral_connectivity.transforms import prepare_time_series

    # Case 1: Single trial with multiple signals
    time_series_2d = rng.standard_normal((100, 5))  # 100 time points, 5 signals
    result = prepare_time_series(time_series_2d, axis="signals")
    assert result.shape == (100, 1, 5)  # (n_time, 1 trial, n_signals)

    # Verify data is preserved
    assert np.allclose(result[:, 0, :], time_series_2d)


def test_prepare_time_series_single_signal():
    """Test helper function for converting 2D (time, trials) to 3D."""
    rng = np.random.default_rng(42)
    from spectral_connectivity.transforms import prepare_time_series

    # Case 2: Multiple trials with single signal
    time_series_2d = rng.standard_normal((100, 10))  # 100 time points, 10 trials
    result = prepare_time_series(time_series_2d, axis="trials")
    assert result.shape == (100, 10, 1)  # (n_time, n_trials, 1 signal)

    # Verify data is preserved
    assert np.allclose(result[:, :, 0], time_series_2d)


def test_prepare_time_series_1d():
    """Test helper function for converting 1D (time,) to 3D."""
    rng = np.random.default_rng(42)
    from spectral_connectivity.transforms import prepare_time_series

    # 1D input: single trial, single signal
    time_series_1d = rng.standard_normal(100)
    result = prepare_time_series(time_series_1d)
    assert result.shape == (100, 1, 1)  # (n_time, 1 trial, 1 signal)

    # Verify data is preserved
    assert np.allclose(result[:, 0, 0], time_series_1d)


def test_prepare_time_series_3d_passthrough():
    """Test that prepare_time_series passes through 3D arrays unchanged."""
    rng = np.random.default_rng(42)
    from spectral_connectivity.transforms import prepare_time_series

    # 3D input should be returned unchanged
    time_series_3d = rng.standard_normal((100, 10, 5))
    result = prepare_time_series(time_series_3d)
    assert result.shape == (100, 10, 5)
    assert np.allclose(result, time_series_3d)


def test_prepare_time_series_invalid_axis():
    """Test that prepare_time_series raises error for invalid axis."""
    rng = np.random.default_rng(42)
    from spectral_connectivity.transforms import prepare_time_series

    time_series_2d = rng.standard_normal((100, 5))
    with pytest.raises(ValueError, match=r"axis must be.*'signals'.*'trials'"):
        prepare_time_series(time_series_2d, axis="invalid")


def test_prepare_time_series_requires_axis_for_2d():
    """Test that prepare_time_series requires axis parameter for 2D input."""
    rng = np.random.default_rng(42)
    from spectral_connectivity.transforms import prepare_time_series

    time_series_2d = rng.standard_normal((100, 5))
    with pytest.raises(
        ValueError, match=r"For 2D input.*must specify.*axis.*parameter"
    ):
        prepare_time_series(time_series_2d)


def test_multitaper_dimension_consistency():
    """Test that Multitaper produces consistent output for properly shaped 3D input."""
    rng = np.random.default_rng(42)
    n_time_samples, n_trials, n_signals = 100, 10, 5
    time_series = rng.standard_normal((n_time_samples, n_trials, n_signals))

    m = Multitaper(time_series=time_series, sampling_frequency=1000)

    # Check that properties report correct dimensions
    assert m.n_signals == n_signals
    assert m.n_trials == n_trials

    # Check that FFT output has correct shape
    fft_result = m.fft()
    assert fft_result.shape[1] == n_trials  # trials dimension
    assert fft_result.shape[4] == n_signals  # signals dimension


# Task 1.3: Parameter Validation Tests


def test_multitaper_rejects_negative_sampling_freq():
    """Test that Multitaper rejects negative sampling frequencies."""
    rng = np.random.default_rng(42)
    time_series = rng.standard_normal((100, 1, 1))

    with pytest.raises(ValueError, match=r"sampling_frequency.*must be positive"):
        Multitaper(time_series=time_series, sampling_frequency=-1000)

    with pytest.raises(ValueError, match=r"sampling_frequency.*must be positive"):
        Multitaper(time_series=time_series, sampling_frequency=0)


def test_multitaper_rejects_invalid_time_halfbandwidth():
    """Test that Multitaper rejects invalid time_halfbandwidth_product values."""
    rng = np.random.default_rng(42)
    time_series = rng.standard_normal((100, 1, 1))

    # Test negative value
    with pytest.raises(
        ValueError, match=r"time_halfbandwidth_product.*must be at least 1"
    ):
        Multitaper(time_series=time_series, time_halfbandwidth_product=-1)

    # Test zero
    with pytest.raises(
        ValueError, match=r"time_halfbandwidth_product.*must be at least 1"
    ):
        Multitaper(time_series=time_series, time_halfbandwidth_product=0)

    # Test value less than 1
    with pytest.raises(
        ValueError, match=r"time_halfbandwidth_product.*must be at least 1"
    ):
        Multitaper(time_series=time_series, time_halfbandwidth_product=0.5)


def test_multitaper_rejects_negative_time_window_duration():
    """Test that Multitaper rejects negative time_window_duration."""
    rng = np.random.default_rng(42)
    time_series = rng.standard_normal((100, 1, 1))

    with pytest.raises(ValueError, match=r"time_window_duration.*must be positive"):
        Multitaper(
            time_series=time_series,
            sampling_frequency=1000,
            time_window_duration=-0.5,
        )

    with pytest.raises(ValueError, match=r"time_window_duration.*must be positive"):
        Multitaper(
            time_series=time_series, sampling_frequency=1000, time_window_duration=0
        )


def test_multitaper_rejects_negative_time_window_step():
    """Test that Multitaper rejects negative time_window_step."""
    rng = np.random.default_rng(42)
    time_series = rng.standard_normal((100, 1, 1))

    with pytest.raises(ValueError, match=r"time_window_step.*must be positive"):
        Multitaper(
            time_series=time_series, sampling_frequency=1000, time_window_step=-0.1
        )

    with pytest.raises(ValueError, match=r"time_window_step.*must be positive"):
        Multitaper(time_series=time_series, sampling_frequency=1000, time_window_step=0)


def test_multitaper_warns_likely_transposed():
    """Test that Multitaper warns when data appears to be transposed."""
    rng = np.random.default_rng(42)
    # Create time series where n_time < n_signals (likely transposed)
    # Shape: (10 time points, 1 trial, 100 signals) - suspiciously few time points
    time_series = rng.standard_normal((10, 1, 100))

    with pytest.warns(UserWarning, match=r"data may be transposed"):
        Multitaper(time_series=time_series, sampling_frequency=1000)


def test_multitaper_warns_on_nan_input():
    """Test that Multitaper warns when input contains NaN or Inf values."""
    rng = np.random.default_rng(42)
    # Test NaN
    time_series_nan = rng.standard_normal((100, 1, 1))
    time_series_nan[50, 0, 0] = np.nan

    with pytest.warns(UserWarning, match=r"contains NaN.*infinite values"):
        Multitaper(time_series=time_series_nan, sampling_frequency=1000)

    # Test Inf
    time_series_inf = rng.standard_normal((100, 1, 1))
    time_series_inf[50, 0, 0] = np.inf

    with pytest.warns(UserWarning, match=r"contains NaN.*infinite values"):
        Multitaper(time_series=time_series_inf, sampling_frequency=1000)

    # Test -Inf
    time_series_neginf = rng.standard_normal((100, 1, 1))
    time_series_neginf[50, 0, 0] = -np.inf

    with pytest.warns(UserWarning, match=r"contains NaN.*infinite values"):
        Multitaper(time_series=time_series_neginf, sampling_frequency=1000)


def test_multitaper_warns_on_large_time_halfbandwidth():
    """Test that Multitaper warns when time_halfbandwidth_product is unusually large."""
    rng = np.random.default_rng(42)
    time_series = rng.standard_normal((100, 1, 1))

    with pytest.warns(UserWarning, match=r"unusually large"):
        Multitaper(time_series=time_series, time_halfbandwidth_product=15)


def test_multitaper_warns_on_step_larger_than_duration():
    """Test that Multitaper warns when time_window_step > time_window_duration."""
    rng = np.random.default_rng(42)
    time_series = rng.standard_normal((1000, 1, 1))

    with pytest.warns(UserWarning, match=r"creates gaps"):
        Multitaper(
            time_series=time_series,
            sampling_frequency=1000,
            time_window_duration=0.5,
            time_window_step=1.0,
        )


def test_multitaper_configuration_and_array_snapshots_are_immutable():
    """Derived state cannot become stale through public mutation."""
    source = np.arange(200.0).reshape(100, 1, 2)
    custom_tapers = np.ones((100, 3))
    m = Multitaper(
        source,
        sampling_frequency=100,
        time_halfbandwidth_product=2,
        n_tapers=3,
        tapers=custom_tapers,
    )

    source[0, 0, 0] = -1
    custom_tapers[0, 0] = -1
    assert m.time_series[0, 0, 0] == 0
    assert m.tapers[0, 0] == 1

    with pytest.raises(ValueError, match="read-only"):
        m.time_series[0, 0, 0] = 5
    with pytest.raises(ValueError, match="read-only"):
        m.tapers[0, 0] = 5
    exposed_time_series = m.time_series
    exposed_time_series.flags.writeable = True
    exposed_time_series[0, 0, 0] = 99
    exposed_tapers = m.tapers
    exposed_tapers.flags.writeable = True
    exposed_tapers[0, 0] = 99
    assert m.time_series[0, 0, 0] == 0
    assert m.tapers[0, 0] == 1
    with pytest.raises(AttributeError, match="immutable after construction"):
        m.time_halfbandwidth_product = 4
    with pytest.raises(AttributeError, match="immutable after construction"):
        m.sampling_frequency = 200


def test_multitaper_provenance_uses_explicit_backend_neutral_fields():
    """Unrelated public attributes do not silently alter serialized metadata."""
    m = Multitaper(np.zeros((100, 1, 2)), sampling_frequency=100)
    m.unrelated_extension_attribute = 42
    metadata = m._provenance_metadata()

    assert tuple(metadata) == m._PROVENANCE_FIELDS
    assert "unrelated_extension_attribute" not in metadata
    assert isinstance(metadata["start_time"], np.ndarray)


def test_short_time_fourier_transform_hann_shape_and_peak():
    sampling_frequency = 128
    time = np.arange(256) / sampling_frequency
    signal = np.sin(2 * np.pi * 16 * time)
    data = np.stack((signal, signal), axis=-1)[:, np.newaxis, :]
    transform = ShortTimeFourierTransform(
        data,
        sampling_frequency=sampling_frequency,
        time_window_duration=1,
        time_window_step=0.5,
    )

    coefficients = transform.fft()
    assert coefficients.shape == (3, 1, 1, 128, 2)
    positive_power = np.abs(coefficients[0, 0, 0, :65, 0]) ** 2
    assert transform.frequencies[np.argmax(positive_power)] == pytest.approx(16)
    assert transform.frequency_resolution == pytest.approx(1.5)


def test_welch_packs_segments_on_observation_axis():
    data = np.random.default_rng(401).standard_normal((256, 3, 2))
    transform = Welch(
        data,
        sampling_frequency=128,
        n_time_samples_per_segment=64,
        segment_overlap=0.5,
    )

    assert transform.n_segments == 7
    assert transform.fft().shape == (1, 3, 7, 64, 2)
    assert transform.time.shape == (1,)


def test_morlet_wavelet_tracks_requested_frequency_and_smoothing():
    sampling_frequency = 128
    time = np.arange(256) / sampling_frequency
    signal = np.sin(2 * np.pi * 16 * time)
    data = np.stack((signal, signal), axis=-1)[:, np.newaxis, :]
    transform = MorletWavelet(
        data,
        sampling_frequency,
        np.array([8.0, 16.0, 32.0]),
        n_cycles=5,
        smoothing_time=0.25,
    )

    coefficients = transform.fft()
    assert coefficients.shape == (8, 1, 32, 3, 2)
    mean_power = np.mean(np.abs(coefficients[..., 0]) ** 2, axis=(0, 1, 2))
    assert transform.frequencies[np.argmax(mean_power)] == pytest.approx(16)

    connectivity = Connectivity.from_transform(transform)
    assert connectivity.power().shape[-2] == 3
    np.testing.assert_array_equal(connectivity.frequencies, transform.frequencies)
    canonical, _ = connectivity.canonical_coherence([0, 1])
    mic, _ = connectivity.maximized_imaginary_coherency([0, 1])
    assert canonical.shape[-3] == 3
    assert mic.shape[-3] == 3
    with pytest.raises(ValueError, match="full two-sided spectrum"):
        connectivity.pairwise_spectral_granger_prediction()


def test_morlet_default_zero_padding_matches_same_convolution():
    from scipy.signal import fftconvolve

    rng = np.random.default_rng(918)
    data = rng.standard_normal((96, 2, 2))
    transform = MorletWavelet(data, 64, np.array([8.0]), n_cycles=4)

    sigma = 4 / (2 * np.pi * 8)
    half_width = int(np.ceil(5 * sigma * 64))
    wavelet_time = np.arange(-half_width, half_width + 1) / 64
    oscillation = np.exp(2j * np.pi * 8 * wavelet_time)
    oscillation -= np.exp(-0.5 * (2 * np.pi * 8 * sigma) ** 2)
    wavelet = oscillation * np.exp(-(wavelet_time**2) / (2 * sigma**2))
    wavelet /= np.sqrt(np.sum(np.abs(wavelet) ** 2))
    expected = fftconvolve(
        data,
        np.conjugate(wavelet[::-1])[:, np.newaxis, np.newaxis],
        mode="same",
        axes=0,
    ) / np.sqrt(64)

    np.testing.assert_allclose(transform.fft()[:, :, 0, 0], expected)


@mark.parametrize("padding_mode", ["reflect", "edge"])
def test_morlet_padding_modes_match_padded_convolution(padding_mode):
    from scipy.signal import fftconvolve

    rng = np.random.default_rng(920)
    data = rng.standard_normal((96, 2, 2))
    transform = MorletWavelet(
        data, 64, np.array([8.0]), n_cycles=4, padding_mode=padding_mode
    )

    sigma = 4 / (2 * np.pi * 8)
    half_width = int(np.ceil(5 * sigma * 64))
    wavelet_time = np.arange(-half_width, half_width + 1) / 64
    oscillation = np.exp(2j * np.pi * 8 * wavelet_time)
    oscillation -= np.exp(-0.5 * (2 * np.pi * 8 * sigma) ** 2)
    wavelet = oscillation * np.exp(-(wavelet_time**2) / (2 * sigma**2))
    wavelet /= np.sqrt(np.sum(np.abs(wavelet) ** 2))
    padded = np.pad(data, ((half_width, half_width), (0, 0), (0, 0)), mode=padding_mode)
    expected = fftconvolve(
        padded,
        np.conjugate(wavelet[::-1])[:, np.newaxis, np.newaxis],
        mode="valid",
        axes=0,
    ) / np.sqrt(64)

    np.testing.assert_allclose(transform.fft()[:, :, 0, 0], expected, atol=1e-12)


def test_morlet_edge_mask_nan_and_trim_contracts():
    rng = np.random.default_rng(919)
    data = rng.standard_normal((256, 2, 2))
    frequencies = np.array([8.0, 16.0, 32.0])
    kept = MorletWavelet(
        data,
        128,
        frequencies,
        n_cycles=5,
        smoothing_time=0.25,
        smoothing_frequency=3,
        edge_mode="keep",
    )
    masked = MorletWavelet(
        data,
        128,
        frequencies,
        n_cycles=5,
        smoothing_time=0.25,
        smoothing_frequency=3,
        edge_mode="nan",
    )
    trimmed = MorletWavelet(
        data,
        128,
        frequencies,
        n_cycles=5,
        edge_mode="trim",
    )

    np.testing.assert_array_equal(
        kept.valid_time_frequency, masked.valid_time_frequency
    )
    assert not np.all(masked.valid_time_frequency)
    masked_power = Connectivity.from_transform(masked).power()
    np.testing.assert_array_equal(
        np.isnan(masked_power[..., 0]), ~masked.valid_time_frequency
    )
    assert np.all(np.isfinite(Connectivity.from_transform(kept).power()))
    assert np.all(trimmed.valid_time_frequency)
    assert trimmed.time[0] >= trimmed.edge_half_width.max()
    assert trimmed.time[-1] <= (len(data) - 1) / 128 - trimmed.edge_half_width.max()


def test_morlet_frequency_smoothing_is_local_cross_spectral_average():
    rng = np.random.default_rng(920)
    data = rng.standard_normal((128, 3, 2))
    frequencies = np.array([8.0, 12.0, 20.0])
    raw = MorletWavelet(data, 64, frequencies, n_cycles=3)
    smoothed = MorletWavelet(
        data,
        64,
        frequencies,
        n_cycles=3,
        smoothing_frequency=3,
        smoothing_kernel="boxcar",
    )
    raw_coefficients = raw.fft()[:, :, 0]
    # Reflection maps the first frequency neighborhood to [12, 8, 12] Hz.
    expected = np.mean(
        raw_coefficients[:, :, [1, 0, 1], :][..., :, :, np.newaxis]
        * np.conjugate(raw_coefficients[:, :, [1, 0, 1], :][..., :, np.newaxis, :]),
        axis=(1, 2),
    )
    actual = Connectivity.from_transform(smoothed).cross_spectral_density()[:, 0]
    np.testing.assert_allclose(actual, expected)


def test_morlet_hann_frequency_smoothing_weights_the_cross_spectral_average():
    rng = np.random.default_rng(922)
    data = rng.standard_normal((128, 3, 2))
    frequencies = np.array([6.0, 8.0, 12.0, 16.0, 20.0])
    raw = MorletWavelet(data, 64, frequencies, n_cycles=3)
    smoothed = MorletWavelet(
        data,
        64,
        frequencies,
        n_cycles=3,
        smoothing_frequency=5,
        smoothing_kernel="hann",
    )
    raw_coefficients = raw.fft()[:, :, 0]  # (time, trial, frequency, signal)
    weights = np.array([0.0, 0.5, 1.0, 0.5, 0.0])  # scipy hann(5, sym=True)

    # Centre frequency (index 2): neighborhood [0, 1, 2, 3, 4], no reflection.
    outer = raw_coefficients[..., :, np.newaxis] * np.conjugate(
        raw_coefficients[..., np.newaxis, :]
    )
    weighted = (
        np.sum(weights[None, None, :, None, None] * outer, axis=2) / weights.sum()
    )
    expected = weighted.mean(axis=1)  # unweighted mean over trials

    actual = Connectivity.from_transform(smoothed).cross_spectral_density()[:, 2]
    np.testing.assert_allclose(actual, expected)


def test_morlet_hann_weights_are_used_and_reject_debiased_measure():
    data = np.random.default_rng(921).standard_normal((128, 1, 2))
    transform = MorletWavelet(
        data,
        64,
        np.array([8.0, 12.0, 20.0]),
        n_cycles=3,
        smoothing_time=0.25,
        smoothing_kernel="hann",
    )
    connectivity = Connectivity.from_transform(transform)

    assert np.unique(transform.observation_weights).size > 1
    with pytest.raises(ValueError, match="non-uniform observation_weights"):
        connectivity.pairwise_phase_consistency()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"smoothing_frequency": 2}, "positive odd integer"),
        ({"smoothing_kernel": "triangle"}, "boxcar.*hann"),
        ({"padding_mode": "wrap"}, "constant.*reflect.*edge"),
        ({"edge_mode": "drop"}, "keep.*nan.*trim"),
    ],
)
def test_morlet_rejects_invalid_edge_and_smoothing_controls(kwargs, message):
    with pytest.raises(ValueError, match=message):
        MorletWavelet(
            np.ones((64, 1, 2)),
            sampling_frequency=64,
            frequencies=[8, 16],
            **kwargs,
        )


def test_multitaper_weighting_modes_are_finite_and_default_is_stable():
    data = np.random.default_rng(402).standard_normal((256, 4, 2))
    default = Multitaper(data, sampling_frequency=128, time_halfbandwidth_product=3)
    uniform = Multitaper(
        data,
        sampling_frequency=128,
        time_halfbandwidth_product=3,
        taper_weighting="uniform",
    )
    eigen = Multitaper(
        data,
        sampling_frequency=128,
        time_halfbandwidth_product=3,
        taper_weighting="eigen",
    )
    adaptive = Multitaper(
        data,
        sampling_frequency=128,
        time_halfbandwidth_product=3,
        taper_weighting="adaptive",
    )

    np.testing.assert_array_equal(default.fft(), uniform.fft())
    assert eigen.taper_eigenvalues is not None
    assert len(eigen.taper_eigenvalues) == eigen.fft().shape[2]
    assert np.all(np.isfinite(eigen.fft()))
    assert np.all(np.isfinite(adaptive.fft()))
    assert not np.allclose(adaptive.fft(), uniform.fft())


def test_adaptive_weighting_matches_eigen_for_white_noise():
    # For a flat (white) spectrum the process noise level equals the mean taper
    # power, so Thomson's denominator collapses to the spectrum and the adaptive
    # weights approach sqrt(eigenvalue) -- i.e. the eigenvalue weighting. This
    # only holds when the noise term is on the same power-spectral-density scale
    # as the periodogram; the previous code left it a factor of the sampling
    # frequency too large, which this oracle would catch.
    data = np.random.default_rng(913).standard_normal((3000, 1, 1))
    eigen = Multitaper(
        data,
        sampling_frequency=500,
        time_halfbandwidth_product=4,
        taper_weighting="eigen",
    ).fft()
    adaptive = Multitaper(
        data,
        sampling_frequency=500,
        time_halfbandwidth_product=4,
        taper_weighting="adaptive",
        adaptive_max_iterations=200,
    ).fft()
    relative_difference = np.abs(adaptive - eigen) / (np.abs(eigen) + 1e-12)
    assert float(np.median(relative_difference)) < 0.05


def test_adaptive_weighting_is_invariant_to_input_scale():
    # The weights are ratios, so scaling the input must not change them once the
    # noise term is on the periodogram's scale.
    data = np.random.default_rng(914).standard_normal((1024, 1, 2))

    def transform(values, weighting):
        return Multitaper(
            values,
            sampling_frequency=256,
            time_halfbandwidth_product=3,
            taper_weighting=weighting,
        ).fft()

    uniform = transform(data, "uniform")
    uniform_scaled = transform(data * 1000, "uniform")
    adaptive = transform(data, "adaptive")
    adaptive_scaled = transform(data * 1000, "adaptive")
    weights = adaptive / uniform
    weights_scaled = adaptive_scaled / uniform_scaled
    np.testing.assert_allclose(weights, weights_scaled, rtol=1e-9, atol=1e-9)


def test_adaptive_weighting_warns_on_non_convergence():
    data = np.random.default_rng(915).standard_normal((512, 1, 2))
    multitaper = Multitaper(
        data,
        sampling_frequency=256,
        time_halfbandwidth_product=4,
        taper_weighting="adaptive",
        adaptive_max_iterations=1,
        adaptive_tolerance=1e-15,
    )
    with pytest.warns(UserWarning, match="did not converge"):
        multitaper.fft()


def test_welch_default_segment_warns_on_coarse_resolution():
    data = np.random.default_rng(916).standard_normal((30000, 1, 1))
    with pytest.warns(UserWarning, match="default segment length"):
        Welch(data, sampling_frequency=30000)


def test_welch_explicit_segment_does_not_warn():
    data = np.random.default_rng(917).standard_normal((30000, 1, 1))
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        Welch(data, sampling_frequency=30000, segment_duration=1.0)


@pytest.mark.parametrize("transform_cls", [ShortTimeFourierTransform, Welch])
def test_transform_rejects_non_positive_sampling_frequency(transform_cls):
    with pytest.raises(ValueError, match="sampling_frequency must be finite"):
        transform_cls(np.ones((64, 1, 2)), sampling_frequency=0)


def test_morlet_rejects_non_positive_sampling_frequency():
    with pytest.raises(ValueError, match="sampling_frequency must be finite"):
        MorletWavelet(np.ones((64, 1, 2)), sampling_frequency=-1, frequencies=[4, 8])


def test_morlet_rejects_frequencies_at_or_above_nyquist():
    with pytest.raises(ValueError, match="below Nyquist"):
        MorletWavelet(np.ones((64, 1, 2)), sampling_frequency=100, frequencies=[10, 60])


def test_welch_rejects_out_of_range_overlap():
    with pytest.raises(ValueError, match="segment_overlap"):
        Welch(np.ones((256, 1, 2)), sampling_frequency=128, segment_overlap=1.0)


def test_nonuniform_weighting_rejects_custom_tapers():
    with pytest.raises(ValueError, match="custom tapers"):
        Multitaper(
            np.ones((64, 2, 2)),
            sampling_frequency=64,
            time_halfbandwidth_product=2,
            n_tapers=3,
            tapers=np.ones((64, 3)),
            taper_weighting="adaptive",
        )
