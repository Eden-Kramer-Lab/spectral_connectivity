import inspect

import numpy as np
from pytest import mark

from spectral_connectivity.connectivity import Connectivity
from spectral_connectivity.wrapper import multitaper_connectivity


@mark.parametrize("time_window_duration", [0.1, 0.2, 2.4, 0.16])
@mark.parametrize("dtype", [np.complex64, np.complex128])
def test_multitaper_coherence_magnitude(time_window_duration, dtype):
    np.random.default_rng(42)
    sampling_frequency = 1500
    start_time, end_time = 0, 4.8
    n_trials, n_signals = 10, 2
    n_time_samples = int((end_time - start_time) * sampling_frequency) + 1
    # time_series = np.zeros((n_time_samples, n_trials, n_signals))
    time_series = np.random.random(size=(n_time_samples, n_trials, n_signals))
    expected_time = np.arange(start_time, end_time, time_window_duration)

    if not np.allclose(expected_time[-1] + time_window_duration, end_time):
        expected_time = expected_time[:-1]

    m = multitaper_connectivity(
        time_series,
        method="coherence_magnitude",
        sampling_frequency=sampling_frequency,
        time_window_duration=time_window_duration,
    )

    assert np.allclose(m.time.values, expected_time)
    assert not (m.values == 0).all()
    assert not (np.isnan(m.values)).all()


def test_multitaper_connectivity():
    np.random.default_rng(42)
    time_window_duration = 0.1
    sampling_frequency = 1500
    start_time, end_time = 0, 4.8
    n_trials, n_signals = 10, 2
    n_time_samples = int((end_time - start_time) * sampling_frequency) + 1
    time_series = np.random.random(size=(n_time_samples, n_trials, n_signals))

    for method in [
        "coherence_magnitude",
        "coherency",
        "coherence_magnitude",
        "coherence_phase",
        "imaginary_coherence",
        "phase_locking_value",
        "phase_lag_index",
        "weighted_phase_lag_index",
        "debiased_squared_phase_lag_index",
        "debiased_squared_weighted_phase_lag_index",
        "pairwise_phase_consistency",
        "phase_lag_index",
        "pairwise_spectral_granger_prediction",
        # Below measures are not implemented, will throw NotImplementedError
        "directed_transfer_function",
        "directed_coherence",
        "partial_directed_coherence",
        "generalized_partial_directed_coherence",
        "direct_directed_transfer_function",
        "canonical_coherence",
        "group_delay",
        "power",
    ]:
        try:
            m = multitaper_connectivity(
                time_series,
                method=method,
                sampling_frequency=sampling_frequency,
                time_window_duration=time_window_duration,
            )
        except (NotImplementedError, ValueError):
            pass

        assert not (m.values == 0).all()
        assert not (np.isnan(m.values)).all()


@mark.parametrize("n_signals", range(2, 5))
def test_multitaper_n_signals(n_signals):
    """
    Test dataarray interface
    """
    np.random.default_rng(42)
    time_window_duration = 0.1
    sampling_frequency = 1500
    start_time, end_time = 0, 4.8
    n_trials, n_signals = 10, n_signals
    n_time_samples = int((end_time - start_time) * sampling_frequency) + 1
    # time_series = np.zeros((n_time_samples, n_trials, n_signals))
    time_series = np.random.random(size=(n_time_samples, n_trials, n_signals))
    expected_time = np.arange(start_time, end_time, time_window_duration)

    if not np.allclose(expected_time[-1] + time_window_duration, end_time):
        expected_time = expected_time[:-1]

    bad_methods = [
        "delay",
        "n_observations",
        "frequencies",
        "all_frequencies",
        "global_coherence",
        "from_multitaper",
        "phase_slope_index",
        "subset_pairwise_spectral_granger_prediction",
    ]
    methods = [
        x for x in dir(Connectivity) if not x.startswith("_") and x not in bad_methods
    ]

    for method in methods:
        try:
            m = multitaper_connectivity(
                time_series,
                method=method,
                sampling_frequency=sampling_frequency,
                time_window_duration=time_window_duration,
            )
            assert np.allclose(m.time.values, expected_time)
            assert not (m.values == 0).all()
            assert not (np.isnan(m.values)).all()

        except (NotImplementedError, ValueError):
            pass


@mark.parametrize("n_signals", range(2, 5))
def test_multitaper_connectivities_n_signals(n_signals):
    np.random.default_rng(42)
    time_window_duration = 0.1
    sampling_frequency = 1500
    start_time, end_time = 0, 4.8
    n_trials, n_signals = 10, n_signals
    n_time_samples = int((end_time - start_time) * sampling_frequency) + 1
    time_series = np.random.random(size=(n_time_samples, n_trials, n_signals))
    expected_time = np.arange(start_time, end_time, time_window_duration)

    if not np.allclose(expected_time[-1] + time_window_duration, end_time):
        expected_time = expected_time[:-1]

    cons = multitaper_connectivity(
        time_series,
        sampling_frequency=sampling_frequency,
        time_window_duration=time_window_duration,
    )
    for mea in cons.data_vars:
        assert np.allclose(cons[mea].time.values, expected_time)
        assert not (cons[mea].values == 0).all()
        assert not (np.isnan(cons[mea].values)).all()

    cons = multitaper_connectivity(
        time_series,
        method=["coherence_magnitude"],
        sampling_frequency=sampling_frequency,
        time_window_duration=time_window_duration,
    )
    mea = "coherence_magnitude"
    assert np.allclose(cons[mea].time.values, expected_time)
    assert not (cons[mea].values == 0).all()
    assert not (np.isnan(cons[mea].values)).all()


def test_frequencies():
    np.random.default_rng(42)
    n_time_samples, n_trials, n_signals = 100, 10, 2
    time_series = np.random.random((n_time_samples, n_trials, n_signals))
    # time_series = np.zeros((n_time_samples, n_trials, n_signals))
    n_fft_samples = 4
    sampling_frequency = 1000

    cons = multitaper_connectivity(
        time_series,
        sampling_frequency=sampling_frequency,
        time_window_duration=None,
        n_fft_samples=n_fft_samples,
    )

    for mea in cons.data_vars:
        assert not (cons[mea].values == 0).all()
        assert not (np.isnan(cons[mea].values)).all()

        expected_frequencies = np.array([0, 250, -500])  # Now includes Nyquist bin
        assert np.allclose(cons[mea].frequency, expected_frequencies)


def test_method_discovery_with_inspect():
    """Test that inspect.getmembers() correctly identifies Connectivity methods.

    This test verifies that the refactored method discovery in wrapper.py
    using inspect.getmembers() finds all expected connectivity methods.
    """
    # Methods that should be excluded (not connectivity measures or not xarray-compatible)
    excluded_methods = {
        # Properties and utility methods
        "delay",
        "n_observations",
        "frequencies",
        "all_frequencies",
        "global_coherence",
        "from_multitaper",
        "phase_slope_index",
        "subset_pairwise_spectral_granger_prediction",
        # Methods not supported by xarray interface
        "group_delay",
        "canonical_coherence",
        "directed_transfer_function",
        "directed_coherence",
        "partial_directed_coherence",
        "generalized_partial_directed_coherence",
        "direct_directed_transfer_function",
        "blockwise_spectral_granger_prediction",
    }

    # Get methods using inspect (same as wrapper.py implementation)
    methods_via_inspect = [
        name
        for name, member in inspect.getmembers(
            Connectivity, predicate=inspect.isfunction
        )
        if not name.startswith("_") and name not in excluded_methods
    ]

    # Get methods using dir() (old implementation)
    methods_via_dir = [
        x
        for x in dir(Connectivity)
        if not x.startswith("_") and x not in excluded_methods
    ]

    # Both methods should find the same set of methods
    assert set(methods_via_inspect) == set(methods_via_dir)

    # Verify we find expected connectivity methods
    expected_methods = {
        "coherence_magnitude",
        "coherency",
        "imaginary_coherence",
        "phase_locking_value",
        "power",
    }
    assert expected_methods.issubset(set(methods_via_inspect))

    # Verify excluded methods are not included
    found_methods_set = set(methods_via_inspect)
    assert not excluded_methods.intersection(found_methods_set)
