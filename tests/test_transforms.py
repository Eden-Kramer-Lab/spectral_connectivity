import numpy as np
import pytest
from nitime.algorithms.spectral import dpss_windows as nitime_dpss_windows
from pytest import mark
from scipy.signal import correlate

from spectral_connectivity.transforms import (
    Multitaper,
    _add_axes,
    _auto_correlation,
    _fix_taper_sign,
    _get_low_bias_tapers,
    _get_taper_eigenvalues,
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
    ],
)
def test__sliding_window(test_array, window_size, step_size, axis, expected_array):
    assert np.allclose(
        _sliding_window(
            test_array, window_size=window_size, step_size=step_size, axis=axis
        ),
        expected_array,
    )


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
    ("sampling_frequency, time_window_duration," "expected_n_time_samples_per_window"),
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
    (
        "sampling_frequency, time_window_duration, n_fft_samples,"
        "expected_n_fft_samples"
    ),
    [(1000, None, 5, 5), (1000, 0.1, None, 100)],
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


def test_frequencies():
    n_time_samples, n_trials, n_signals = 100, 10, 2
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
    (
        "time_halfbandwidth_product, time_window_duration, "
        "expected_frequency_resolution"
    ),
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
    ("time_window_step, n_time_samples_per_step, " "expected_n_samples_per_time_step"),
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


def test__fix_taper_sign():
    n_time_samples, n_tapers = 100, 4
    tapers = -3 * np.ones((n_tapers, n_time_samples))
    tapers[1, :3] = -1 * np.arange(0, 3)  # Begin with negative lobe
    tapers[2, :] = 2
    tapers[3, :3] = np.arange(0, 3)  # Begin with positive lobe
    fixed_tapers = _fix_taper_sign(tapers, n_time_samples)
    assert np.all(fixed_tapers[::2, :].sum(axis=1) >= 0)
    assert np.all(fixed_tapers[2, :] == 2)
    assert np.all(fixed_tapers[1, :].sum() >= 0)
    assert ~np.all(fixed_tapers[3, :].sum() >= 0)


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


@mark.parametrize(
    "n_time_samples, time_halfbandwidth_product, n_tapers",
    [(31, 6, 4), (31, 7, 4), (31, 8, 4), (31, 8, 4.2)],
)
def test__get_taper_eigenvalues(n_time_samples, time_halfbandwidth_product, n_tapers):
    time_index = np.arange(n_time_samples, dtype="d")
    half_bandwidth = float(time_halfbandwidth_product) / n_time_samples
    nitime_tapers, _ = nitime_dpss_windows(
        n_time_samples, time_halfbandwidth_product, n_tapers
    )
    eigenvalues = _get_taper_eigenvalues(nitime_tapers, half_bandwidth, time_index)
    assert np.allclose(eigenvalues, 1.0)


def test__auto_correlation():
    np.random.seed(42)
    n_time_samples, n_tapers = 100, 3
    test_data = np.random.rand(n_tapers, n_time_samples)
    rxx = _auto_correlation(test_data)[:, :n_time_samples]

    for taper_ind in np.arange(n_tapers):
        expected_correlation = correlate(
            test_data[taper_ind, :], test_data[taper_ind, :]
        )[n_time_samples - 1 :]
        assert np.allclose(rxx[taper_ind], expected_correlation)


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
    # 1D input should raise ValueError
    time_series_1d = np.random.randn(100)
    with pytest.raises(ValueError, match=r"Expected 3D array.*got 1D"):
        Multitaper(time_series=time_series_1d)

    # 2D input should raise ValueError with helpful message
    time_series_2d = np.random.randn(100, 5)
    with pytest.raises(
        ValueError,
        match=r"Expected 3D array.*got 2D.*Use prepare_time_series|np.newaxis",
    ):
        Multitaper(time_series=time_series_2d)

    # 4D input should raise ValueError
    time_series_4d = np.random.randn(100, 10, 5, 3)
    with pytest.raises(ValueError, match=r"Expected 3D array.*got 4D"):
        Multitaper(time_series=time_series_4d)

    # 3D input should work
    time_series_3d = np.random.randn(100, 10, 5)
    m = Multitaper(time_series=time_series_3d)
    assert m.time_series.ndim == 3


def test_prepare_time_series_single_trial():
    """Test helper function for converting 2D (time, signals) to 3D."""
    from spectral_connectivity.transforms import prepare_time_series

    # Case 1: Single trial with multiple signals
    time_series_2d = np.random.randn(100, 5)  # 100 time points, 5 signals
    result = prepare_time_series(time_series_2d, axis="signals")
    assert result.shape == (100, 1, 5)  # (n_time, 1 trial, n_signals)

    # Verify data is preserved
    assert np.allclose(result[:, 0, :], time_series_2d)


def test_prepare_time_series_single_signal():
    """Test helper function for converting 2D (time, trials) to 3D."""
    from spectral_connectivity.transforms import prepare_time_series

    # Case 2: Multiple trials with single signal
    time_series_2d = np.random.randn(100, 10)  # 100 time points, 10 trials
    result = prepare_time_series(time_series_2d, axis="trials")
    assert result.shape == (100, 10, 1)  # (n_time, n_trials, 1 signal)

    # Verify data is preserved
    assert np.allclose(result[:, :, 0], time_series_2d)


def test_prepare_time_series_1d():
    """Test helper function for converting 1D (time,) to 3D."""
    from spectral_connectivity.transforms import prepare_time_series

    # 1D input: single trial, single signal
    time_series_1d = np.random.randn(100)
    result = prepare_time_series(time_series_1d)
    assert result.shape == (100, 1, 1)  # (n_time, 1 trial, 1 signal)

    # Verify data is preserved
    assert np.allclose(result[:, 0, 0], time_series_1d)


def test_prepare_time_series_3d_passthrough():
    """Test that prepare_time_series passes through 3D arrays unchanged."""
    from spectral_connectivity.transforms import prepare_time_series

    # 3D input should be returned unchanged
    time_series_3d = np.random.randn(100, 10, 5)
    result = prepare_time_series(time_series_3d)
    assert result.shape == (100, 10, 5)
    assert np.allclose(result, time_series_3d)


def test_prepare_time_series_invalid_axis():
    """Test that prepare_time_series raises error for invalid axis."""
    from spectral_connectivity.transforms import prepare_time_series

    time_series_2d = np.random.randn(100, 5)
    with pytest.raises(ValueError, match=r"axis must be.*'signals'.*'trials'"):
        prepare_time_series(time_series_2d, axis="invalid")


def test_prepare_time_series_requires_axis_for_2d():
    """Test that prepare_time_series requires axis parameter for 2D input."""
    from spectral_connectivity.transforms import prepare_time_series

    time_series_2d = np.random.randn(100, 5)
    with pytest.raises(
        ValueError, match=r"For 2D input.*must specify.*axis.*parameter"
    ):
        prepare_time_series(time_series_2d)


def test_multitaper_dimension_consistency():
    """Test that Multitaper produces consistent output for properly shaped 3D input."""
    n_time_samples, n_trials, n_signals = 100, 10, 5
    time_series = np.random.randn(n_time_samples, n_trials, n_signals)

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
    time_series = np.random.randn(100, 1, 1)

    with pytest.raises(ValueError, match=r"sampling_frequency.*must be positive"):
        Multitaper(time_series=time_series, sampling_frequency=-1000)

    with pytest.raises(ValueError, match=r"sampling_frequency.*must be positive"):
        Multitaper(time_series=time_series, sampling_frequency=0)


def test_multitaper_rejects_invalid_time_halfbandwidth():
    """Test that Multitaper rejects invalid time_halfbandwidth_product values."""
    time_series = np.random.randn(100, 1, 1)

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
    time_series = np.random.randn(100, 1, 1)

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
    time_series = np.random.randn(100, 1, 1)

    with pytest.raises(ValueError, match=r"time_window_step.*must be positive"):
        Multitaper(
            time_series=time_series, sampling_frequency=1000, time_window_step=-0.1
        )

    with pytest.raises(ValueError, match=r"time_window_step.*must be positive"):
        Multitaper(time_series=time_series, sampling_frequency=1000, time_window_step=0)


def test_multitaper_warns_likely_transposed():
    """Test that Multitaper warns when data appears to be transposed."""
    # Create time series where n_time < n_signals (likely transposed)
    # Shape: (10 time points, 1 trial, 100 signals) - suspiciously few time points
    time_series = np.random.randn(10, 1, 100)

    with pytest.warns(UserWarning, match=r"data may be transposed"):
        Multitaper(time_series=time_series, sampling_frequency=1000)


def test_multitaper_warns_on_nan_input():
    """Test that Multitaper warns when input contains NaN or Inf values."""
    # Test NaN
    time_series_nan = np.random.randn(100, 1, 1)
    time_series_nan[50, 0, 0] = np.nan

    with pytest.warns(UserWarning, match=r"contains NaN.*infinite values"):
        Multitaper(time_series=time_series_nan, sampling_frequency=1000)

    # Test Inf
    time_series_inf = np.random.randn(100, 1, 1)
    time_series_inf[50, 0, 0] = np.inf

    with pytest.warns(UserWarning, match=r"contains NaN.*infinite values"):
        Multitaper(time_series=time_series_inf, sampling_frequency=1000)

    # Test -Inf
    time_series_neginf = np.random.randn(100, 1, 1)
    time_series_neginf[50, 0, 0] = -np.inf

    with pytest.warns(UserWarning, match=r"contains NaN.*infinite values"):
        Multitaper(time_series=time_series_neginf, sampling_frequency=1000)


def test_multitaper_warns_on_large_time_halfbandwidth():
    """Test that Multitaper warns when time_halfbandwidth_product is unusually large."""
    time_series = np.random.randn(100, 1, 1)

    with pytest.warns(UserWarning, match=r"unusually large"):
        Multitaper(time_series=time_series, time_halfbandwidth_product=15)


def test_multitaper_warns_on_step_larger_than_duration():
    """Test that Multitaper warns when time_window_step > time_window_duration."""
    time_series = np.random.randn(1000, 1, 1)

    with pytest.warns(UserWarning, match=r"creates gaps"):
        Multitaper(
            time_series=time_series,
            sampling_frequency=1000,
            time_window_duration=0.5,
            time_window_step=1.0,
        )
