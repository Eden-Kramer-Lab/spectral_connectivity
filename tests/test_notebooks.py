"""Snapshot tests for tutorial notebooks.

These tests verify that key numerical outputs from the tutorial
notebooks remain stable across code changes. Tests are inspired
by notebook examples but hand-written for clarity and focus.

Uses syrupy with custom NumPy extension for approximate equality.
"""

import numpy as np
import pytest
from syrupy.extensions.amber import AmberSnapshotExtension

from spectral_connectivity import Connectivity, Multitaper
from spectral_connectivity.simulate import simulate_MVAR
from spectral_connectivity.transforms import prepare_time_series


class NumPySnapshotExtension(AmberSnapshotExtension):
    """Custom syrupy extension that uses np.allclose for array comparison."""

    @classmethod
    def matches(cls, *, serialized_data, snapshot_data):
        """Check if data matches using np.allclose for arrays."""
        if isinstance(serialized_data, dict) and isinstance(snapshot_data, dict):
            if set(serialized_data.keys()) != set(snapshot_data.keys()):
                return False

            for key in serialized_data:
                s_val = serialized_data[key]
                snap_val = snapshot_data[key]

                if isinstance(s_val, np.ndarray) and isinstance(snap_val, np.ndarray):
                    if not np.allclose(s_val, snap_val, rtol=1e-7, atol=1e-10):
                        return False
                elif s_val != snap_val:
                    return False
            return True
        return serialized_data == snapshot_data


@pytest.fixture
def snapshot(snapshot):
    """Override snapshot fixture to use NumPy extension."""
    return snapshot.use_extension(NumPySnapshotExtension)


def test_power_spectrum_200hz(snapshot):
    """Power spectrum of 200 Hz signal."""
    np.random.seed(42)
    sampling_frequency = 1500
    time = np.linspace(0, 50, 75001, endpoint=True)
    signal = np.sin(2 * np.pi * time * 200)
    noise = np.random.normal(0, 4, signal.shape)

    multitaper = Multitaper(
        prepare_time_series(signal + noise),
        sampling_frequency=sampling_frequency,
        time_halfbandwidth_product=3,
        start_time=time[0],
    )
    connectivity = Connectivity.from_multitaper(multitaper)

    # Snapshot outputs as a dict
    outputs = {
        "power": connectivity.power(),
        "frequencies": connectivity.frequencies,
    }
    assert outputs == snapshot


def test_coherence_magnitude_phase_offset(snapshot):
    """Coherence with fixed phase offset between signals."""
    np.random.seed(42)
    sampling_frequency = 1500
    time_extent = (0, 50)
    n_time_samples = int(((time_extent[1] - time_extent[0]) * sampling_frequency) + 1)
    time = np.linspace(
        time_extent[0], time_extent[1], num=n_time_samples, endpoint=True
    )

    # Create 2 signals with pi/2 phase offset
    frequency_of_interest = 200
    n_signals = 2
    signal = np.zeros((n_time_samples, n_signals))
    signal[:, 0] = np.sin(2 * np.pi * time * frequency_of_interest)
    phase_offset = np.pi / 2
    signal[:, 1] = np.sin((2 * np.pi * time * frequency_of_interest) + phase_offset)
    noise = np.random.normal(0, 4, signal.shape)

    multitaper = Multitaper(
        prepare_time_series(signal + noise, axis="signals"),
        sampling_frequency=sampling_frequency,
        time_halfbandwidth_product=5,
        start_time=time[0],
    )
    connectivity = Connectivity.from_multitaper(multitaper)

    outputs = {
        "coherence_magnitude": connectivity.coherence_magnitude(),
        "frequencies": connectivity.frequencies,
    }
    assert outputs == snapshot


def test_spectrogram_temporal_dynamics(snapshot):
    """Spectrogram showing 50 Hz turning on at t=25s."""
    np.random.seed(42)
    sampling_frequency = 1500
    time_extent = (0, 50)
    frequency_of_interest = [200, 50]
    n_time_samples = int(((time_extent[1] - time_extent[0]) * sampling_frequency) + 1)
    time = np.linspace(
        time_extent[0], time_extent[1], num=n_time_samples, endpoint=True
    )

    # Create signal with 200 Hz constant, 50 Hz turns on at t=25s
    signal = np.sin(2 * np.pi * time[:, np.newaxis] * frequency_of_interest)
    signal[: n_time_samples // 2, 1] = 0  # 50 Hz only in second half
    signal = signal.sum(axis=1)
    noise = np.random.normal(0, 4, signal.shape)

    multitaper = Multitaper(
        prepare_time_series(signal + noise),
        sampling_frequency=sampling_frequency,
        time_halfbandwidth_product=3,
        time_window_duration=0.600,
        time_window_step=0.300,
        start_time=time[0],
    )
    connectivity = Connectivity.from_multitaper(multitaper)

    outputs = {
        "power": connectivity.power(),
        "frequencies": connectivity.frequencies,
        "time": connectivity.time,
    }
    assert outputs == snapshot


def test_coherogram_phase_change(snapshot):
    """Coherogram showing phase offset changing at t=1.5s."""
    np.random.seed(42)
    sampling_frequency = 1500
    time_extent = (0, 2.400)
    n_trials = 100
    n_signals = 2
    frequency_of_interest = 200
    n_time_samples = int(((time_extent[1] - time_extent[0]) * sampling_frequency) + 1)
    time = np.linspace(
        time_extent[0], time_extent[1], num=n_time_samples, endpoint=True
    )

    # Random phase before t=1.5s, fixed phase after
    signal = np.zeros((n_time_samples, n_trials, n_signals))
    signal[:, :, 0] = np.sin(2 * np.pi * time[:, np.newaxis] * frequency_of_interest)
    phase_offset = np.random.uniform(-np.pi, np.pi, size=(n_time_samples, n_trials))
    phase_offset[np.where(time > 1.5), :] = np.pi / 2
    signal[:, :, 1] = np.sin(
        (2 * np.pi * time[:, np.newaxis] * frequency_of_interest) + phase_offset
    )
    noise = np.random.normal(0, 2, signal.shape)

    multitaper = Multitaper(
        prepare_time_series(signal + noise),
        sampling_frequency=sampling_frequency,
        time_halfbandwidth_product=1,
        time_window_duration=0.080,
        time_window_step=0.080,
        start_time=time[0],
    )
    connectivity = Connectivity.from_multitaper(multitaper)

    outputs = {
        "coherence_magnitude": connectivity.coherence_magnitude(),
        "time": connectivity.time,
    }
    assert outputs == snapshot


def test_power_spectrum_30hz(snapshot):
    """Power spectrum of 30 Hz signal."""
    np.random.seed(42)
    frequency_of_interest = 30
    sampling_frequency = 1500
    time_extent = (0, 50)
    n_time_samples = int(((time_extent[1] - time_extent[0]) * sampling_frequency) + 1)
    time = np.linspace(
        time_extent[0], time_extent[1], num=n_time_samples, endpoint=True
    )
    signal = np.sin(2 * np.pi * time * frequency_of_interest)
    noise = np.random.normal(0, 4, len(signal))

    multitaper = Multitaper(
        prepare_time_series(signal + noise),
        sampling_frequency=sampling_frequency,
        time_halfbandwidth_product=3,
        start_time=time[0],
    )
    connectivity = Connectivity.from_multitaper(multitaper)

    outputs = {
        "power": connectivity.power(),
        "frequencies": connectivity.frequencies,
    }
    assert outputs == snapshot


def test_spectrogram_with_trials(snapshot):
    """Spectrogram with trial structure (time x trials)."""
    np.random.seed(42)
    sampling_frequency = 1500
    time_extent = (0, 50)
    frequency_of_interest = [200, 50]
    n_time_samples = int(((time_extent[1] - time_extent[0]) * sampling_frequency) + 1)
    n_trials = 10
    time = np.linspace(
        time_extent[0], time_extent[1], num=n_time_samples, endpoint=True
    )

    # Create signal with 200 Hz constant, 50 Hz turns on at t=25s
    signal = np.sin(2 * np.pi * time[:, np.newaxis] * frequency_of_interest)
    signal[: n_time_samples // 2, 1] = 0  # 50 Hz only in second half
    signal = signal.sum(axis=1)

    # Replicate across trials with noise
    signal = np.tile(signal[:, np.newaxis], (1, n_trials))
    noise = np.random.normal(0, 4, signal.shape)

    multitaper = Multitaper(
        prepare_time_series(signal + noise, axis="trials"),
        sampling_frequency=sampling_frequency,
        time_halfbandwidth_product=3,
        time_window_duration=0.600,
        time_window_step=0.300,
        start_time=time[0],
    )
    connectivity = Connectivity.from_multitaper(multitaper)

    outputs = {
        "power": connectivity.power(),
        "frequencies": connectivity.frequencies,
        "time": connectivity.time,
    }
    assert outputs == snapshot


def test_spectrogram_decreased_frequency_resolution(snapshot):
    """Spectrogram with decreased frequency resolution."""
    np.random.seed(42)
    sampling_frequency = 1500
    time_extent = (0, 50)
    frequency_of_interest = [200, 50]
    n_time_samples = int(((time_extent[1] - time_extent[0]) * sampling_frequency) + 1)
    time = np.linspace(
        time_extent[0], time_extent[1], num=n_time_samples, endpoint=True
    )

    signal = np.sin(2 * np.pi * time[:, np.newaxis] * frequency_of_interest)
    signal[: n_time_samples // 2, 1] = 0
    signal = signal.sum(axis=1)
    noise = np.random.normal(0, 4, signal.shape)

    multitaper = Multitaper(
        prepare_time_series(signal + noise),
        sampling_frequency=sampling_frequency,
        time_halfbandwidth_product=1,  # Decreased from 3
        time_window_duration=0.600,
        time_window_step=0.300,
        start_time=time[0],
    )
    connectivity = Connectivity.from_multitaper(multitaper)

    outputs = {
        "power": connectivity.power(),
        "frequencies": connectivity.frequencies,
        "time": connectivity.time,
    }
    assert outputs == snapshot


def test_coherence_no_trials(snapshot):
    """Coherence without trial structure."""
    np.random.seed(42)
    sampling_frequency = 1500
    time_extent = (0, 50)
    n_time_samples = int(((time_extent[1] - time_extent[0]) * sampling_frequency) + 1)
    time = np.linspace(
        time_extent[0], time_extent[1], num=n_time_samples, endpoint=True
    )

    frequency_of_interest = 200
    n_signals = 2
    signal = np.zeros((n_time_samples, n_signals))
    signal[:, 0] = np.sin(2 * np.pi * time * frequency_of_interest)
    phase_offset = np.pi / 2
    signal[:, 1] = np.sin((2 * np.pi * time * frequency_of_interest) + phase_offset)
    noise = np.random.normal(0, 4, signal.shape)

    multitaper = Multitaper(
        prepare_time_series(signal + noise, axis="signals"),
        sampling_frequency=sampling_frequency,
        time_halfbandwidth_product=5,
        start_time=time[0],
    )
    connectivity = Connectivity.from_multitaper(multitaper)

    outputs = {
        "coherence_magnitude": connectivity.coherence_magnitude(),
        "frequencies": connectivity.frequencies,
    }
    assert outputs == snapshot


def test_coherence_with_trials(snapshot):
    """Coherence with trial structure, 200 Hz, pi/2 phase offset."""
    np.random.seed(42)
    sampling_frequency = 1500
    time_extent = (0, 2.400)
    n_trials = 100
    n_signals = 2
    frequency_of_interest = 200
    n_time_samples = int(((time_extent[1] - time_extent[0]) * sampling_frequency) + 1)
    time = np.linspace(
        time_extent[0], time_extent[1], num=n_time_samples, endpoint=True
    )

    signal = np.zeros((n_time_samples, n_trials, n_signals))
    signal[:, :, 0] = np.sin(2 * np.pi * time[:, np.newaxis] * frequency_of_interest)
    phase_offset = np.pi / 2
    signal[:, :, 1] = np.sin(
        (2 * np.pi * time[:, np.newaxis] * frequency_of_interest) + phase_offset
    )
    noise = np.random.normal(0, 2, signal.shape)

    multitaper = Multitaper(
        prepare_time_series(signal + noise),
        sampling_frequency=sampling_frequency,
        time_halfbandwidth_product=1,
        start_time=time[0],
    )
    connectivity = Connectivity.from_multitaper(multitaper)

    outputs = {
        "coherence_magnitude": connectivity.coherence_magnitude(),
        "frequencies": connectivity.frequencies,
    }
    assert outputs == snapshot


def test_imaginary_coherence(snapshot):
    """Imaginary coherence with phase offset."""
    np.random.seed(42)
    sampling_frequency = 1500
    time_extent = (0, 2.400)
    n_trials = 100
    n_signals = 2
    frequency_of_interest = 200
    n_time_samples = int(((time_extent[1] - time_extent[0]) * sampling_frequency) + 1)
    time = np.linspace(
        time_extent[0], time_extent[1], num=n_time_samples, endpoint=True
    )

    signal = np.zeros((n_time_samples, n_trials, n_signals))
    signal[:, :, 0] = np.sin(2 * np.pi * time[:, np.newaxis] * frequency_of_interest)
    phase_offset = np.pi / 2
    signal[:, :, 1] = np.sin(
        (2 * np.pi * time[:, np.newaxis] * frequency_of_interest) + phase_offset
    )
    noise = np.random.normal(0, 2, signal.shape)

    multitaper = Multitaper(
        prepare_time_series(signal + noise),
        sampling_frequency=sampling_frequency,
        time_halfbandwidth_product=1,
        start_time=time[0],
    )
    connectivity = Connectivity.from_multitaper(multitaper)

    outputs = {
        "imaginary_coherence": connectivity.imaginary_coherence(),
        "frequencies": connectivity.frequencies,
    }
    assert outputs == snapshot


def test_phase_locking_value(snapshot):
    """Phase locking value with phase offset."""
    np.random.seed(42)
    sampling_frequency = 1500
    time_extent = (0, 2.400)
    n_trials = 100
    n_signals = 2
    frequency_of_interest = 200
    n_time_samples = int(((time_extent[1] - time_extent[0]) * sampling_frequency) + 1)
    time = np.linspace(
        time_extent[0], time_extent[1], num=n_time_samples, endpoint=True
    )

    signal = np.zeros((n_time_samples, n_trials, n_signals))
    signal[:, :, 0] = np.sin(2 * np.pi * time[:, np.newaxis] * frequency_of_interest)
    phase_offset = np.pi / 2
    signal[:, :, 1] = np.sin(
        (2 * np.pi * time[:, np.newaxis] * frequency_of_interest) + phase_offset
    )
    noise = np.random.normal(0, 2, signal.shape)

    multitaper = Multitaper(
        prepare_time_series(signal + noise),
        sampling_frequency=sampling_frequency,
        time_halfbandwidth_product=1,
        start_time=time[0],
    )
    connectivity = Connectivity.from_multitaper(multitaper)

    outputs = {
        "phase_locking_value": connectivity.phase_locking_value(),
        "frequencies": connectivity.frequencies,
    }
    assert outputs == snapshot


def test_phase_lag_index(snapshot):
    """Phase lag index with phase offset."""
    np.random.seed(42)
    sampling_frequency = 1500
    time_extent = (0, 2.400)
    n_trials = 100
    n_signals = 2
    frequency_of_interest = 200
    n_time_samples = int(((time_extent[1] - time_extent[0]) * sampling_frequency) + 1)
    time = np.linspace(
        time_extent[0], time_extent[1], num=n_time_samples, endpoint=True
    )

    signal = np.zeros((n_time_samples, n_trials, n_signals))
    signal[:, :, 0] = np.sin(2 * np.pi * time[:, np.newaxis] * frequency_of_interest)
    phase_offset = np.pi / 2
    signal[:, :, 1] = np.sin(
        (2 * np.pi * time[:, np.newaxis] * frequency_of_interest) + phase_offset
    )
    noise = np.random.normal(0, 2, signal.shape)

    multitaper = Multitaper(
        prepare_time_series(signal + noise),
        sampling_frequency=sampling_frequency,
        time_halfbandwidth_product=1,
        start_time=time[0],
    )
    connectivity = Connectivity.from_multitaper(multitaper)

    outputs = {
        "phase_lag_index": connectivity.phase_lag_index(),
        "frequencies": connectivity.frequencies,
    }
    assert outputs == snapshot


def test_weighted_phase_lag_index(snapshot):
    """Weighted phase lag index with phase offset."""
    np.random.seed(42)
    sampling_frequency = 1500
    time_extent = (0, 2.400)
    n_trials = 100
    n_signals = 2
    frequency_of_interest = 200
    n_time_samples = int(((time_extent[1] - time_extent[0]) * sampling_frequency) + 1)
    time = np.linspace(
        time_extent[0], time_extent[1], num=n_time_samples, endpoint=True
    )

    signal = np.zeros((n_time_samples, n_trials, n_signals))
    signal[:, :, 0] = np.sin(2 * np.pi * time[:, np.newaxis] * frequency_of_interest)
    phase_offset = np.pi / 2
    signal[:, :, 1] = np.sin(
        (2 * np.pi * time[:, np.newaxis] * frequency_of_interest) + phase_offset
    )
    noise = np.random.normal(0, 2, signal.shape)

    multitaper = Multitaper(
        prepare_time_series(signal + noise),
        sampling_frequency=sampling_frequency,
        time_halfbandwidth_product=1,
        start_time=time[0],
    )
    connectivity = Connectivity.from_multitaper(multitaper)

    outputs = {
        "weighted_phase_lag_index": connectivity.weighted_phase_lag_index(),
        "frequencies": connectivity.frequencies,
    }
    assert outputs == snapshot


def test_debiased_squared_weighted_phase_lag_index(snapshot):
    """Debiased squared weighted phase lag index."""
    np.random.seed(42)
    sampling_frequency = 1500
    time_extent = (0, 2.400)
    n_trials = 100
    n_signals = 2
    frequency_of_interest = 200
    n_time_samples = int(((time_extent[1] - time_extent[0]) * sampling_frequency) + 1)
    time = np.linspace(
        time_extent[0], time_extent[1], num=n_time_samples, endpoint=True
    )

    signal = np.zeros((n_time_samples, n_trials, n_signals))
    signal[:, :, 0] = np.sin(2 * np.pi * time[:, np.newaxis] * frequency_of_interest)
    phase_offset = np.pi / 2
    signal[:, :, 1] = np.sin(
        (2 * np.pi * time[:, np.newaxis] * frequency_of_interest) + phase_offset
    )
    noise = np.random.normal(0, 2, signal.shape)

    multitaper = Multitaper(
        prepare_time_series(signal + noise),
        sampling_frequency=sampling_frequency,
        time_halfbandwidth_product=1,
        start_time=time[0],
    )
    connectivity = Connectivity.from_multitaper(multitaper)

    outputs = {
        "debiased_squared_wpli": connectivity.debiased_squared_weighted_phase_lag_index(),
        "frequencies": connectivity.frequencies,
    }
    assert outputs == snapshot


def test_pairwise_phase_consistency(snapshot):
    """Pairwise phase consistency with phase offset."""
    np.random.seed(42)
    sampling_frequency = 1500
    time_extent = (0, 2.400)
    n_trials = 100
    n_signals = 2
    frequency_of_interest = 200
    n_time_samples = int(((time_extent[1] - time_extent[0]) * sampling_frequency) + 1)
    time = np.linspace(
        time_extent[0], time_extent[1], num=n_time_samples, endpoint=True
    )

    signal = np.zeros((n_time_samples, n_trials, n_signals))
    signal[:, :, 0] = np.sin(2 * np.pi * time[:, np.newaxis] * frequency_of_interest)
    phase_offset = np.pi / 2
    signal[:, :, 1] = np.sin(
        (2 * np.pi * time[:, np.newaxis] * frequency_of_interest) + phase_offset
    )
    noise = np.random.normal(0, 2, signal.shape)

    multitaper = Multitaper(
        prepare_time_series(signal + noise),
        sampling_frequency=sampling_frequency,
        time_halfbandwidth_product=1,
        start_time=time[0],
    )
    connectivity = Connectivity.from_multitaper(multitaper)

    outputs = {
        "pairwise_phase_consistency": connectivity.pairwise_phase_consistency(),
        "frequencies": connectivity.frequencies,
    }
    assert outputs == snapshot


def test_group_delay_signal1_leads(snapshot):
    """Group delay: Signal #1 leads Signal #2."""
    np.random.seed(42)
    sampling_frequency = 1500
    time_extent = (0, 50)
    n_time_samples = int(((time_extent[1] - time_extent[0]) * sampling_frequency) + 1)
    time = np.linspace(
        time_extent[0], time_extent[1], num=n_time_samples, endpoint=True
    )

    frequency_of_interest = 200
    n_signals = 2
    time_lag = 0.010  # 10 ms lag
    signal = np.zeros((n_time_samples, n_signals))
    signal[:, 0] = np.sin(2 * np.pi * time * frequency_of_interest)

    # Create time-shifted version
    time_shifted = time - time_lag
    signal[:, 1] = np.sin(2 * np.pi * time_shifted * frequency_of_interest)
    noise = np.random.normal(0, 4, signal.shape)

    multitaper = Multitaper(
        prepare_time_series(signal + noise, axis="signals"),
        sampling_frequency=sampling_frequency,
        time_halfbandwidth_product=5,
        start_time=time[0],
    )
    connectivity = Connectivity.from_multitaper(multitaper)

    outputs = {
        "group_delay": connectivity.group_delay(),
        "frequencies": connectivity.frequencies,
    }
    assert outputs == snapshot


def test_group_delay_signal2_leads(snapshot):
    """Group delay: Signal #2 leads Signal #1."""
    np.random.seed(42)
    sampling_frequency = 1500
    time_extent = (0, 50)
    n_time_samples = int(((time_extent[1] - time_extent[0]) * sampling_frequency) + 1)
    time = np.linspace(
        time_extent[0], time_extent[1], num=n_time_samples, endpoint=True
    )

    frequency_of_interest = 200
    n_signals = 2
    time_lag = 0.010
    signal = np.zeros((n_time_samples, n_signals))

    # Signal 2 leads (appears first in time)
    time_shifted = time + time_lag
    signal[:, 0] = np.sin(2 * np.pi * time_shifted * frequency_of_interest)
    signal[:, 1] = np.sin(2 * np.pi * time * frequency_of_interest)
    noise = np.random.normal(0, 4, signal.shape)

    multitaper = Multitaper(
        prepare_time_series(signal + noise, axis="signals"),
        sampling_frequency=sampling_frequency,
        time_halfbandwidth_product=5,
        start_time=time[0],
    )
    connectivity = Connectivity.from_multitaper(multitaper)

    outputs = {
        "group_delay": connectivity.group_delay(),
        "frequencies": connectivity.frequencies,
    }
    assert outputs == snapshot


def test_group_delay_signal2_leads_over_time(snapshot):
    """Group delay: Signal #2 leads Signal #1 over time (with trials)."""
    np.random.seed(42)
    sampling_frequency = 1500
    time_extent = (0, 2.400)
    n_trials = 100  # Need trials for sufficient observations with windowing
    n_time_samples = int(((time_extent[1] - time_extent[0]) * sampling_frequency) + 1)
    time = np.linspace(
        time_extent[0], time_extent[1], num=n_time_samples, endpoint=True
    )

    frequency_of_interest = 200
    n_signals = 2
    time_lag = 0.010
    signal = np.zeros((n_time_samples, n_trials, n_signals))

    # Signal 2 leads (appears first in time)
    time_shifted = time + time_lag
    signal[:, :, 0] = np.sin(
        2 * np.pi * time_shifted[:, np.newaxis] * frequency_of_interest
    )
    signal[:, :, 1] = np.sin(2 * np.pi * time[:, np.newaxis] * frequency_of_interest)
    noise = np.random.normal(0, 4, signal.shape)

    multitaper = Multitaper(
        prepare_time_series(signal + noise),
        sampling_frequency=sampling_frequency,
        time_halfbandwidth_product=1,
        time_window_duration=0.080,
        time_window_step=0.080,
        start_time=time[0],
    )
    connectivity = Connectivity.from_multitaper(multitaper)

    outputs = {
        "group_delay": connectivity.group_delay(),
        "frequencies": connectivity.frequencies,
        "time": connectivity.time,
    }
    assert outputs == snapshot


def test_phase_slope_index_signal1_leads(snapshot):
    """Phase slope index: Signal #1 leads Signal #2."""
    np.random.seed(42)
    sampling_frequency = 1500
    time_extent = (0, 50)
    n_time_samples = int(((time_extent[1] - time_extent[0]) * sampling_frequency) + 1)
    time = np.linspace(
        time_extent[0], time_extent[1], num=n_time_samples, endpoint=True
    )

    frequency_of_interest = 200
    n_signals = 2
    time_lag = 0.010
    signal = np.zeros((n_time_samples, n_signals))
    signal[:, 0] = np.sin(2 * np.pi * time * frequency_of_interest)

    time_shifted = time - time_lag
    signal[:, 1] = np.sin(2 * np.pi * time_shifted * frequency_of_interest)
    noise = np.random.normal(0, 4, signal.shape)

    multitaper = Multitaper(
        prepare_time_series(signal + noise, axis="signals"),
        sampling_frequency=sampling_frequency,
        time_halfbandwidth_product=5,
        start_time=time[0],
    )
    connectivity = Connectivity.from_multitaper(multitaper)

    outputs = {
        "phase_slope_index": connectivity.phase_slope_index(),
        "frequencies": connectivity.frequencies,
    }
    assert outputs == snapshot


def test_phase_slope_index_signal2_leads(snapshot):
    """Phase slope index: Signal #2 leads Signal #1."""
    np.random.seed(42)
    sampling_frequency = 1500
    time_extent = (0, 50)
    n_time_samples = int(((time_extent[1] - time_extent[0]) * sampling_frequency) + 1)
    time = np.linspace(
        time_extent[0], time_extent[1], num=n_time_samples, endpoint=True
    )

    frequency_of_interest = 200
    n_signals = 2
    time_lag = 0.010
    signal = np.zeros((n_time_samples, n_signals))

    time_shifted = time + time_lag
    signal[:, 0] = np.sin(2 * np.pi * time_shifted * frequency_of_interest)
    signal[:, 1] = np.sin(2 * np.pi * time * frequency_of_interest)
    noise = np.random.normal(0, 4, signal.shape)

    multitaper = Multitaper(
        prepare_time_series(signal + noise, axis="signals"),
        sampling_frequency=sampling_frequency,
        time_halfbandwidth_product=5,
        start_time=time[0],
    )
    connectivity = Connectivity.from_multitaper(multitaper)

    outputs = {
        "phase_slope_index": connectivity.phase_slope_index(),
        "frequencies": connectivity.frequencies,
    }
    assert outputs == snapshot


def test_canonical_coherence(snapshot):
    """Canonical coherence with multiple signal groups."""
    np.random.seed(42)
    sampling_frequency = 1500
    time_extent = (0, 50)
    n_time_samples = int(((time_extent[1] - time_extent[0]) * sampling_frequency) + 1)
    time = np.linspace(
        time_extent[0], time_extent[1], num=n_time_samples, endpoint=True
    )

    frequency_of_interest = 200
    n_group1_signals = 3
    n_group2_signals = 3
    n_signals = n_group1_signals + n_group2_signals

    signal = np.zeros((n_time_samples, n_signals))
    base_signal = np.sin(2 * np.pi * time * frequency_of_interest)

    # Group 1: same base signal
    for i in range(n_group1_signals):
        signal[:, i] = base_signal

    # Group 2: phase-shifted version
    phase_offset = np.pi / 2
    for i in range(n_group2_signals):
        signal[:, n_group1_signals + i] = np.sin(
            (2 * np.pi * time * frequency_of_interest) + phase_offset
        )

    noise = np.random.normal(0, 4, signal.shape)

    multitaper = Multitaper(
        prepare_time_series(signal + noise, axis="signals"),
        sampling_frequency=sampling_frequency,
        time_halfbandwidth_product=5,
        start_time=time[0],
    )
    connectivity = Connectivity.from_multitaper(multitaper)

    outputs = {
        "canonical_coherence": connectivity.canonical_coherence(
            (np.arange(n_group1_signals), np.arange(n_group1_signals, n_signals))
        ),
        "frequencies": connectivity.frequencies,
    }
    assert outputs == snapshot


def test_canonical_coherence_high_noise(snapshot):
    """Canonical coherence with more signals and higher noise."""
    np.random.seed(42)
    sampling_frequency = 1500
    time_extent = (0, 50)
    n_time_samples = int(((time_extent[1] - time_extent[0]) * sampling_frequency) + 1)
    time = np.linspace(
        time_extent[0], time_extent[1], num=n_time_samples, endpoint=True
    )

    frequency_of_interest = 200
    n_group1_signals = 5
    n_group2_signals = 5
    n_signals = n_group1_signals + n_group2_signals

    signal = np.zeros((n_time_samples, n_signals))
    base_signal = np.sin(2 * np.pi * time * frequency_of_interest)

    for i in range(n_group1_signals):
        signal[:, i] = base_signal

    phase_offset = np.pi / 2
    for i in range(n_group2_signals):
        signal[:, n_group1_signals + i] = np.sin(
            (2 * np.pi * time * frequency_of_interest) + phase_offset
        )

    noise = np.random.normal(0, 8, signal.shape)  # Higher noise

    multitaper = Multitaper(
        prepare_time_series(signal + noise, axis="signals"),
        sampling_frequency=sampling_frequency,
        time_halfbandwidth_product=5,
        start_time=time[0],
    )
    connectivity = Connectivity.from_multitaper(multitaper)

    outputs = {
        "canonical_coherence": connectivity.canonical_coherence(
            (np.arange(n_group1_signals), np.arange(n_group1_signals, n_signals))
        ),
        "frequencies": connectivity.frequencies,
    }
    assert outputs == snapshot


def test_global_coherence(snapshot):
    """Global coherence across multiple signals."""
    np.random.seed(42)
    sampling_frequency = 1500
    time_extent = (0, 50)
    n_time_samples = int(((time_extent[1] - time_extent[0]) * sampling_frequency) + 1)
    time = np.linspace(
        time_extent[0], time_extent[1], num=n_time_samples, endpoint=True
    )

    frequency_of_interest = 200
    n_signals = 5
    signal = np.zeros((n_time_samples, n_signals))
    base_signal = np.sin(2 * np.pi * time * frequency_of_interest)

    # All signals are the same base with different noise
    for i in range(n_signals):
        signal[:, i] = base_signal

    noise = np.random.normal(0, 4, signal.shape)

    multitaper = Multitaper(
        prepare_time_series(signal + noise, axis="signals"),
        sampling_frequency=sampling_frequency,
        time_halfbandwidth_product=5,
        start_time=time[0],
    )
    connectivity = Connectivity.from_multitaper(multitaper)

    outputs = {
        "global_coherence": connectivity.global_coherence(),
        "frequencies": connectivity.frequencies,
    }
    assert outputs == snapshot


# ============ Tutorial_Using_Paper_Examples tests ============
# Only keeping 3 representative MVAR examples


def test_baccala_example2(snapshot):
    """Baccala Example 2: Partial directed coherence (representative PDC example)."""
    np.random.seed(42)
    sampling_frequency = 200
    n_time_samples, n_signals = 1000, 3

    coefficients = np.array([[[0.5, 0.3, 0.4], [-0.5, 0.3, 1.0], [0.0, -0.3, -0.2]]])
    noise_covariance = np.eye(n_signals)

    time_series = simulate_MVAR(
        coefficients,
        noise_covariance=noise_covariance,
        n_time_samples=n_time_samples,
        n_trials=50,  # Reduced for faster test runtime
        n_burnin_samples=500,
    )

    multitaper = Multitaper(
        prepare_time_series(time_series),
        sampling_frequency=sampling_frequency,
        time_halfbandwidth_product=1,
        start_time=0,
    )
    connectivity = Connectivity.from_multitaper(multitaper)

    outputs = {
        "pairwise_spectral_granger": connectivity.pairwise_spectral_granger_prediction(),
        "directed_transfer_function": connectivity.directed_transfer_function(),
        "partial_directed_coherence": connectivity.partial_directed_coherence(),
        "frequencies": connectivity.frequencies,
    }
    assert outputs == snapshot


def test_ding_example1(snapshot):
    """Ding Example 1: Direct DTF (representative dDTF example)."""
    np.random.seed(42)
    sampling_frequency = 200
    n_time_samples, n_signals = 1000, 2

    coefficients = np.array([[[0.8, 0.0], [0.4, 0.5]]])
    noise_covariance = np.eye(n_signals)

    time_series = simulate_MVAR(
        coefficients,
        noise_covariance=noise_covariance,
        n_time_samples=n_time_samples,
        n_trials=50,  # Reduced for faster test runtime
        n_burnin_samples=500,
    )

    multitaper = Multitaper(
        prepare_time_series(time_series),
        sampling_frequency=sampling_frequency,
        time_halfbandwidth_product=2,
        start_time=0,
    )
    connectivity = Connectivity.from_multitaper(multitaper)

    outputs = {
        "directed_transfer_function": connectivity.directed_transfer_function(),
        "direct_directed_transfer_function": connectivity.direct_directed_transfer_function(),
        "frequencies": connectivity.frequencies,
    }
    assert outputs == snapshot


def test_nedungadi_example2(snapshot):
    """Nedungadi Example 2: Conditional Granger (representative example showing confounds)."""
    np.random.seed(42)
    sampling_frequency = 200
    n_time_samples, n_signals = 1000, 3

    coefficients = np.array([[[0.5, 0.3, 0.0], [0.4, 0.5, 0.0], [0.5, 0.3, 0.5]]])
    noise_covariance = np.eye(n_signals)

    time_series = simulate_MVAR(
        coefficients,
        noise_covariance=noise_covariance,
        n_time_samples=n_time_samples,
        n_trials=50,  # Reduced for faster test runtime
        n_burnin_samples=500,
    )

    multitaper = Multitaper(
        prepare_time_series(time_series),
        sampling_frequency=sampling_frequency,
        time_halfbandwidth_product=2,
        start_time=0,
    )
    connectivity = Connectivity.from_multitaper(multitaper)

    outputs = {
        "pairwise_spectral_granger": connectivity.pairwise_spectral_granger_prediction(),
        "conditional_spectral_granger": connectivity.conditional_spectral_granger_prediction(),
        "frequencies": connectivity.frequencies,
    }
    assert outputs == snapshot


@pytest.mark.slow
def test_tutorial_simulated_examples_executes():
    """Verify Tutorial_On_Simulated_Examples notebook executes without errors."""
    import subprocess

    result = subprocess.run(
        [
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--ExecutePreprocessor.timeout=600",
            "examples/Tutorial_On_Simulated_Examples.ipynb",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, (
        f"Notebook execution failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    )


@pytest.mark.slow
def test_tutorial_paper_examples_executes():
    """Verify Tutorial_Using_Paper_Examples notebook executes without errors."""
    import subprocess

    result = subprocess.run(
        [
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--ExecutePreprocessor.timeout=600",
            "examples/Tutorial_Using_Paper_Examples.ipynb",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, (
        f"Notebook execution failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    )
