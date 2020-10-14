import numpy as np
from nitime.algorithms.spectral import dpss_windows as nitime_dpss_windows
from pytest import mark
from scipy.signal import correlate
from spectral_connectivity.transforms import (Multitaper, _add_axes,
                                              _auto_correlation,
                                              _fix_taper_sign,
                                              _get_low_bias_tapers,
                                              _get_taper_eigenvalues,
                                              _multitaper_fft, _sliding_window,
                                              dpss_windows)


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
    'test_array, window_size, step_size, axis, expected_array',
    [(np.arange(1, 6), 3, 1, -1, np.array([[1, 2, 3],
                                           [2, 3, 4],
                                           [3, 4, 5]])),
     (np.arange(1, 6), 3, 2, -1, np.array([[1, 2, 3],
                                           [3, 4, 5]])),
     (np.arange(0, 6).reshape((2, 3)), 2, 1, 0, np.array([[[0, 3],
                                                           [1, 4],
                                                           [2, 5]]]))
     ])
def test__sliding_window(
        test_array, window_size, step_size, axis, expected_array):
    assert np.allclose(
        _sliding_window(
            test_array, window_size=window_size, step_size=step_size,
            axis=axis),
        expected_array)


@mark.parametrize(
    'time_halfbandwidth_product, expected_n_tapers',
    [(3, 5), (1, 1), (1.75, 2)])
def test_n_tapers(time_halfbandwidth_product, expected_n_tapers):
    n_time_samples, n_trials, n_signals = 100, 10, 2
    time_series = np.zeros((n_time_samples, n_trials, n_signals))
    m = Multitaper(
        time_series=time_series,
        time_halfbandwidth_product=time_halfbandwidth_product)
    assert m.n_tapers == expected_n_tapers


@mark.parametrize(
    'sampling_frequency, time_window_duration, expected_duration',
    [(1000, None, 0.1), (2000, None, 0.05), (1000, 0.1, 0.1)])
def test_time_window_duration(sampling_frequency, time_window_duration,
                              expected_duration):
    n_time_samples, n_trials, n_signals = 100, 10, 2
    time_series = np.zeros((n_time_samples, n_trials, n_signals))
    m = Multitaper(
        time_series=time_series,
        sampling_frequency=sampling_frequency,
        time_window_duration=time_window_duration)
    assert m.time_window_duration == expected_duration


@mark.parametrize(
    'sampling_frequency, time_window_step, expected_step',
    [(1000, None, 0.1), (2000, None, 0.05), (1000, 0.1, 0.1)])
def test_time_window_step(
        sampling_frequency, time_window_step, expected_step):
    n_time_samples, n_trials, n_signals = 100, 10, 2
    time_series = np.zeros((n_time_samples, n_trials, n_signals))
    m = Multitaper(
        time_series=time_series,
        sampling_frequency=sampling_frequency,
        time_window_step=time_window_step)
    assert m.time_window_step == expected_step


@mark.parametrize(
    ('sampling_frequency, time_window_duration,'
     'expected_n_time_samples_per_window'),
    [(1000, None, 100), (1000, 0.1, 100), (2000, 0.025, 50)])
def test_n_time_samples(
        sampling_frequency, time_window_duration,
        expected_n_time_samples_per_window):
    n_time_samples, n_trials, n_signals = 100, 10, 2
    time_series = np.zeros((n_time_samples, n_trials, n_signals))
    m = Multitaper(
        time_series=time_series,
        sampling_frequency=sampling_frequency,
        time_window_duration=time_window_duration)
    assert (m.n_time_samples_per_window ==
            expected_n_time_samples_per_window)


@mark.parametrize(
    ('sampling_frequency, time_window_duration, n_fft_samples,'
     'expected_n_fft_samples'),
    [(1000, None, 5, 5), (1000, 0.1, None, 100)])
def test_n_fft_samples(
    sampling_frequency, time_window_duration, n_fft_samples,
        expected_n_fft_samples):
    n_time_samples, n_trials, n_signals = 100, 10, 2
    time_series = np.zeros((n_time_samples, n_trials, n_signals))
    m = Multitaper(
        time_series=time_series,
        sampling_frequency=sampling_frequency,
        time_window_duration=time_window_duration,
        n_fft_samples=n_fft_samples)
    assert m.n_fft_samples == expected_n_fft_samples


def test_frequencies():
    n_time_samples, n_trials, n_signals = 100, 10, 2
    time_series = np.zeros((n_time_samples, n_trials, n_signals))
    n_fft_samples = 4
    sampling_frequency = 1000
    m = Multitaper(
        time_series=time_series,
        sampling_frequency=sampling_frequency,
        n_fft_samples=n_fft_samples)
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

    time_series = np.zeros((n_time_samples, n_signals))
    m = Multitaper(time_series=time_series)
    assert m.n_trials == 1


@mark.parametrize(
    ('time_halfbandwidth_product, time_window_duration, '
     'expected_frequency_resolution'),
    [(3, .10, 30), (1, 0.02, 50), (5, 1, 5)])
def test_frequency_resolution(
        time_halfbandwidth_product, time_window_duration,
        expected_frequency_resolution):
    n_time_samples, n_trials, n_signals = 100, 10, 2
    time_series = np.zeros((n_time_samples, n_trials, n_signals))
    m = Multitaper(
        time_series=time_series,
        time_halfbandwidth_product=time_halfbandwidth_product,
        time_window_duration=time_window_duration)
    assert m.frequency_resolution == expected_frequency_resolution


@mark.parametrize(
    ('time_window_step, n_time_samples_per_step, '
     'expected_n_samples_per_time_step'),
    [(None, None, 100), (0.001, None, 1), (0.002, None, 2),
     (None, 10, 10)])
def test_n_samples_per_time_step(
        time_window_step, n_time_samples_per_step,
        expected_n_samples_per_time_step):
    n_time_samples, n_trials, n_signals = 100, 10, 2
    time_series = np.zeros((n_time_samples, n_trials, n_signals))

    m = Multitaper(
        time_window_duration=0.10,
        n_time_samples_per_step=n_time_samples_per_step,
        time_series=time_series,
        time_window_step=time_window_step)
    assert m.n_time_samples_per_step == expected_n_samples_per_time_step


@mark.parametrize('time_window_duration', [0.1, 0.2, 2.4, 0.16])
def test_time(time_window_duration):
    sampling_frequency = 1500
    start_time, end_time = -2.4, 2.4
    n_trials, n_signals = 10, 2
    n_time_samples = int(
        (end_time - start_time) * sampling_frequency) + 1
    time_series = np.zeros((n_time_samples, n_trials, n_signals))
    expected_time = np.arange(start_time, end_time, time_window_duration)
    if not np.allclose(expected_time[-1] + time_window_duration, end_time):
        expected_time = expected_time[:-1]
    m = Multitaper(
        sampling_frequency=sampling_frequency,
        time_series=time_series,
        start_time=start_time,
        time_window_duration=time_window_duration)
    assert np.allclose(m.time, expected_time)


def test_tapers():
    n_time_samples, n_trials, n_signals = 100, 10, 2
    time_series = np.zeros((n_time_samples, n_trials, n_signals))
    m = Multitaper(time_series, is_low_bias=False)
    assert np.allclose(m.tapers.shape, (n_time_samples, m.n_tapers))

    m = Multitaper(time_series, tapers=np.zeros((10, 3)))
    assert np.allclose(m.tapers.shape, (10, 3))


@mark.parametrize(
    'eigenvalues, expected_n_tapers',
    [(np.array([0.95, 0.95, 0.95]), 3),
     (np.array([0.95, 0.8, 0.95]), 2),
     (np.array([0.8, 0.8, 0.8]), 1)])
def test__get_low_bias_tapers(eigenvalues, expected_n_tapers):
    tapers = np.zeros((3, 100))
    filtered_tapers, filtered_eigenvalues = _get_low_bias_tapers(
        tapers, eigenvalues)
    assert (filtered_tapers.shape[0] == filtered_eigenvalues.shape[0] ==
            expected_n_tapers)


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
    'n_time_samples, time_halfbandwidth_product, n_tapers',
    [(1000, 3, 5), (31, 6, 4), (31, 7, 4)])
def test_dpss_windows(
        n_time_samples, time_halfbandwidth_product, n_tapers):
    tapers, eigenvalues = dpss_windows(
        n_time_samples, time_halfbandwidth_product, n_tapers,
        is_low_bias=False)
    nitime_tapers, nitime_eigenvalues = nitime_dpss_windows(
        n_time_samples, time_halfbandwidth_product, n_tapers)
    assert np.allclose(np.sum(tapers ** 2, axis=1), 1.0)
    assert np.allclose(tapers, nitime_tapers)
    assert np.allclose(eigenvalues, nitime_eigenvalues)


@mark.parametrize(
    'n_time_samples, time_halfbandwidth_product, n_tapers',
    [(31, 6, 4), (31, 7, 4), (31, 8, 4), (31, 8, 4.2)])
def test__get_taper_eigenvalues(
        n_time_samples, time_halfbandwidth_product, n_tapers):
    time_index = np.arange(n_time_samples, dtype='d')
    half_bandwidth = float(time_halfbandwidth_product) / n_time_samples
    nitime_tapers, _ = nitime_dpss_windows(
        n_time_samples, time_halfbandwidth_product, n_tapers)
    eigenvalues = _get_taper_eigenvalues(
        nitime_tapers, half_bandwidth, time_index)
    assert np.allclose(eigenvalues, 1.0)


def test__auto_correlation():
    n_time_samples, n_tapers = 100, 3
    test_data = np.random.rand(n_tapers, n_time_samples)
    rxx = _auto_correlation(test_data)[:, :n_time_samples]

    for taper_ind in np.arange(n_tapers):
        expected_correlation = correlate(
            test_data[taper_ind, :], test_data[taper_ind, :])[
                n_time_samples - 1:]
        assert np.allclose(rxx[taper_ind], expected_correlation)


def test__multitaper_fft():
    n_windows, n_trials, n_time_samples, n_tapers, n_fft_samples = (
        2, 10, 100, 3, 100)
    sampling_frequency = 1000
    time_series = np.ones((n_windows, n_trials, n_time_samples))
    tapers = np.ones((n_time_samples, n_tapers))

    fourier_coefficients = _multitaper_fft(
        tapers, time_series, n_fft_samples, sampling_frequency)
    assert np.allclose(
        fourier_coefficients.shape,
        (n_windows, n_trials, n_fft_samples, n_tapers))


def test_fft():
    n_time_samples, n_trials, n_signals, n_windows = 100, 10, 2, 1
    time_series = np.zeros((n_time_samples, n_trials, n_signals))
    m = Multitaper(time_series=time_series)
    assert np.allclose(
        m.fft().shape,
        (n_windows, n_trials, m.tapers.shape[1], m.n_fft_samples,
         n_signals))
