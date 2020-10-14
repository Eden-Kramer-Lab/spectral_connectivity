import numpy as np
from pytest import mark
from spectral_connectivity.wrapper import multitaper_connectivity


@mark.parametrize('time_window_duration', [0.1, 0.2, 2.4, 0.16])
def test_multitaper_coherence_magnitude(time_window_duration):
    sampling_frequency = 1500
    start_time, end_time = 0, 4.8
    n_trials, n_signals = 10, 2
    n_time_samples = int((end_time - start_time) * sampling_frequency) + 1
    # time_series = np.zeros((n_time_samples, n_trials, n_signals))
    time_series = np.random.random(size=(n_time_samples, n_trials, n_signals))
    expected_time = np.arange(start_time, end_time, time_window_duration)

    if not np.allclose(expected_time[-1] + time_window_duration, end_time):
        expected_time = expected_time[:-1]

    m = multitaper_connectivity(time_series,
                                method='coherence_magnitude',
                                sampling_frequency=sampling_frequency,
                                time_window_duration=time_window_duration
                                )

    assert np.allclose(m.Time.values, expected_time)
    assert not (m.values == 0).all()
    assert not (np.isnan(m.values)).all()


def test_multitaper_connectivity():
    time_window_duration = .1
    sampling_frequency = 1500
    start_time, end_time = 0, 4.8
    n_trials, n_signals = 10, 2
    n_time_samples = int((end_time - start_time) * sampling_frequency) + 1
    time_series = np.random.random(size=(n_time_samples, n_trials, n_signals))

    for method in ['coherence_magnitude',
                   'coherency',
                   'coherence_magnitude',
                   'coherence_phase',
                   'imaginary_coherence',
                   'phase_locking_value',
                   'phase_lag_index',
                   'weighted_phase_lag_index',
                   'debiased_squared_phase_lag_index',
                   'debiased_squared_weighted_phase_lag_index',
                   'pairwise_phase_consistency',
                   'directed_transfer_function',
                   'directed_coherence',
                   'partial_directed_coherence',
                   'generalized_partial_directed_coherence',
                   'direct_directed_transfer_function',
                   'phase_lag_index',
                   'pairwise_spectral_granger_prediction',
                   # 'canonical_coherence',
                   # 'group_delay',
                   # 'power'
                   ]:
        m = multitaper_connectivity(time_series,
                                    method=method,
                                    sampling_frequency=sampling_frequency,
                                    time_window_duration=time_window_duration,
                                    )

        assert not (m.values == 0).all()
        assert not (np.isnan(m.values)).all()


@mark.parametrize('n_signals', range(2, 5))
def test_multitaper_n_signals(n_signals):
    time_window_duration = .1
    sampling_frequency = 1500
    start_time, end_time = 0, 4.8
    n_trials, n_signals = 10, n_signals
    n_time_samples = int((end_time - start_time) * sampling_frequency) + 1
    # time_series = np.zeros((n_time_samples, n_trials, n_signals))
    time_series = np.random.random(size=(n_time_samples, n_trials, n_signals))
    expected_time = np.arange(start_time, end_time, time_window_duration)

    if not np.allclose(expected_time[-1] + time_window_duration, end_time):
        expected_time = expected_time[:-1]

    m = multitaper_connectivity(time_series,
                                sampling_frequency=sampling_frequency,
                                time_window_duration=time_window_duration
                                )

    assert np.allclose(m.Time.values, expected_time)
    assert not (m.values == 0).all()
    assert not (np.isnan(m.values)).all()


def test_frequencies():
    n_time_samples, n_trials, n_signals = 100, 10, 2
    time_series = np.random.random((n_time_samples, n_trials, n_signals))
    # time_series = np.zeros((n_time_samples, n_trials, n_signals))
    n_fft_samples = 4
    sampling_frequency = 1000

    m = multitaper_connectivity(time_series,
                                sampling_frequency=sampling_frequency,
                                time_window_duration=None,
                                n_fft_samples=n_fft_samples
                                )

    assert not (m.values == 0).all()
    assert not (np.isnan(m.values)).all()

    expected_frequencies = np.array([0, 250])
    print(expected_frequencies)
    assert np.allclose(m.Frequency, expected_frequencies)
