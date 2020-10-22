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
                   'phase_lag_index',
                   'pairwise_spectral_granger_prediction',
                   # Below measures are not implemented, will throw NotImplementedError
                   'directed_transfer_function',
                   'directed_coherence',
                   'partial_directed_coherence',
                   'generalized_partial_directed_coherence',
                   'direct_directed_transfer_function',
                   'canonical_coherence',
                   'group_delay',
                   'power'
                   ]:
        try:
            m = multitaper_connectivity(time_series,
                                    method=method,
                                    sampling_frequency=sampling_frequency,
                                    time_window_duration=time_window_duration,
                                    )
        except NotImplementedError:
            pass
        assert not (m.values == 0).all()
        assert not (np.isnan(m.values)).all()


# def test_multitaper_canonical_coherence():
#     time_window_duration = .2
#     time_halfbandwidth_product = 2
#     frequency_of_interest = 200
#     sampling_frequency = 1500
#     time_extent = (0, 2.400)
#     n_trials = 100
#     n_signals = 6
#     n_time_samples = int(((time_extent[1] - time_extent[0]) * sampling_frequency) + 1)
#     time = np.linspace(time_extent[0], time_extent[1], num=n_time_samples, endpoint=True)
#
#     signal = np.zeros((n_time_samples, n_trials, n_signals))
#     signal[:, :, 0:2] = (
#             np.sin(2 * np.pi * time * frequency_of_interest)[:, np.newaxis, np.newaxis] *
#             np.ones((1, n_trials, 2)))
#
#     expected_time = np.arange(time_extent[0], time_extent[-1], time_window_duration)
#
#     if not np.allclose(expected_time[-1] + time_window_duration, time_extent[-1]):
#         expected_time = expected_time[:-1]
#
#     other_signals = (n_signals + 1) // 2
#     n_other_signals = n_signals - other_signals
#     phase_offset = np.random.uniform(-np.pi, np.pi, size=(n_time_samples, n_trials, n_other_signals))
#     phase_offset[np.where(time > 1.5), :] = np.pi / 2
#     signal[:, :, other_signals:] = np.sin(
#         (2 * np.pi * time[:, np.newaxis, np.newaxis] * frequency_of_interest) + phase_offset)
#     noise = np.random.normal(10, 7, signal.shape)
#     group_labels = (['a'] * (n_signals - n_other_signals)) + (['b'] * n_other_signals)
#
#     m = multitaper_connectivity(signal + noise,
#                                 sampling_frequency=sampling_frequency,
#                                 time_halfbandwidth_product=time_halfbandwidth_product,
#                                 time_window_duration=time_window_duration,
#                                 time_window_step=0.080,
#                                 method='canonical_coherence',
#                                 connectivity_kwargs={"group_labels": group_labels}
#                                 )
#
#     assert np.allclose(m.Time.values, expected_time)
#     assert not (m.values == 0).all()
#     assert not (np.isnan(m.values)).all()


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
