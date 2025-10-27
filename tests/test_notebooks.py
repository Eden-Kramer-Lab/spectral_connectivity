"""Snapshot tests for tutorial notebooks.

These tests verify that key numerical outputs from the tutorial
notebooks remain stable across code changes. Tests are inspired
by notebook examples but hand-written for clarity and focus.

Uses syrupy with custom NumPy extension for approximate equality.
"""
import numpy as np
import pytest
from syrupy.extensions.single_file import SingleFileSnapshotExtension, WriteMode
from syrupy.extensions.amber import AmberSnapshotExtension

from spectral_connectivity import Connectivity, Multitaper
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
    time = np.linspace(time_extent[0], time_extent[1], num=n_time_samples, endpoint=True)

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
    time = np.linspace(time_extent[0], time_extent[1], num=n_time_samples, endpoint=True)

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
    time = np.linspace(time_extent[0], time_extent[1], num=n_time_samples, endpoint=True)

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
        f"Notebook execution failed:\n"
        f"STDOUT: {result.stdout}\n"
        f"STDERR: {result.stderr}"
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
        f"Notebook execution failed:\n"
        f"STDOUT: {result.stdout}\n"
        f"STDERR: {result.stderr}"
    )
