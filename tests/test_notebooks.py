"""Snapshot tests for tutorial notebooks.

These tests verify that key numerical outputs from the tutorial
notebooks remain stable across code changes. Tests are inspired
by notebook examples but hand-written for clarity and focus.

Uses syrupy with a custom NumPy extension for approximate (allclose) equality.
"""

import base64
import gzip
import json

import numpy as np
import pytest
from syrupy.extensions.amber import AmberSnapshotExtension

from spectral_connectivity import Connectivity, Multitaper
from spectral_connectivity.simulate import simulate_MVAR
from spectral_connectivity.transforms import prepare_time_series

# Arrays are stored as float32 (its ~1e-7 relative precision matches the
# comparison tolerance) to keep the compressed baseline small; rtol=1e-6 is far
# tighter than any real regression yet robust to cross-library float noise.
_RTOL = 1e-6
_ATOL = 1e-9
_STORE_DTYPE = np.float32


def _encode(data):
    """Encode numeric data as a JSON structure with full arrays stored compressed.

    syrupy compares the *serialized* form (a string), not the original objects,
    so a true numerical tolerance requires (a) a serialization that can be parsed
    back to the FULL array and (b) an element-wise ``allclose`` comparison in
    ``matches``. Every array is stored as gzip-compressed float32 bytes (base64
    text), so the comparison covers every element -- a compact summary
    (statistics + a few samples) would miss changes at unsampled positions or
    permutations of equal-magnitude values. Complex arrays are stored as separate
    real/imag parts.
    """
    if isinstance(data, dict):
        return {"dict": {k: _encode(v) for k, v in sorted(data.items())}}
    if isinstance(data, (list, tuple)):
        return {"seq": [_encode(v) for v in data]}
    array = np.asarray(data)
    if array.dtype.kind == "c":
        return {
            "dict": {
                "__real__": _encode(np.real(array)),
                "__imag__": _encode(np.imag(array)),
            }
        }
    contiguous = np.ascontiguousarray(array, dtype=_STORE_DTYPE)
    blob = base64.b64encode(gzip.compress(contiguous.tobytes(), 9)).decode("ascii")
    return {"array": {"shape": list(array.shape), "gzip_b64": blob}}


def _decode(obj):
    """Inverse of :func:`_encode`, producing arrays for numeric comparison."""
    if "dict" in obj:
        return {k: _decode(v) for k, v in obj["dict"].items()}
    if "seq" in obj:
        return [_decode(v) for v in obj["seq"]]
    spec = obj["array"]
    raw = gzip.decompress(base64.b64decode(spec["gzip_b64"]))
    return np.frombuffer(raw, dtype=_STORE_DTYPE).reshape(spec["shape"])


def _numeric_allclose(a, b):
    """Recursively compare decoded structures element-wise with ``np.allclose``."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_numeric_allclose(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_numeric_allclose(x, y) for x, y in zip(a, b))
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and bool(
        np.allclose(a, b, rtol=_RTOL, atol=_ATOL, equal_nan=True)
    )


class NumPySnapshotExtension(AmberSnapshotExtension):
    """Snapshot extension that stores each full array (gzip-compressed float32)
    and compares every element with ``np.allclose`` (rtol=1e-6, atol=1e-9), so
    snapshots tolerate tiny floating-point differences (e.g. across BLAS /
    library versions) with a true, array-wide numerical tolerance rather than a
    bit-exact string match or a lossy summary."""

    def serialize(self, data, **kwargs):
        return json.dumps(_encode(data), indent=2)

    def matches(self, *, serialized_data, snapshot_data):
        try:
            return _numeric_allclose(
                _decode(json.loads(serialized_data)),
                _decode(json.loads(snapshot_data)),
            )
        except (ValueError, TypeError, KeyError, json.JSONDecodeError):
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
            np.array([0] * n_group1_signals + [1] * n_group2_signals)
        )[0],
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
            np.array([0] * n_group1_signals + [1] * n_group2_signals)
        )[0],
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
        # Snapshot only the coherence fractions (phase-invariant). The singular
        # vectors (second return value) have arbitrary sign/complex phase that
        # can change across SciPy/BLAS versions and are not snapshot-stable.
        "global_coherence": connectivity.global_coherence()[0],
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
        random_state=42,
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
        random_state=42,
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


@pytest.mark.skip(
    reason="conditional_spectral_granger_prediction is not implemented "
    "(raises NotImplementedError); re-enable when it lands."
)
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
        random_state=42,
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


def _snapshot_matches(a, b):
    """Emulate the extension's serialize -> matches path for a pair of values."""
    ext = NumPySnapshotExtension.__new__(NumPySnapshotExtension)
    return ext.matches(serialized_data=ext.serialize(a), snapshot_data=ext.serialize(b))


def test_snapshot_tolerance_is_a_true_allclose():
    """Element-wise allclose: within-tolerance matches, any real change is caught.

    Guards against the earlier bugs: the tolerance was dead code (syrupy compares
    serialized strings); a significant-figure-rounding approximation failed for
    values straddling a quantization boundary; and a compact statistics+samples
    fingerprint missed changes at unsampled positions and permutations of
    equal-magnitude values. The full-array comparison handles all of these.
    """
    values = {"x": np.array([1.0, 0.5, 750.123456, 0.0, np.nan, np.inf, -np.inf])}
    # ~5e-8 relative perturbation is within rtol=1e-6.
    assert _snapshot_matches(values, {"x": values["x"] * (1 + 5e-8)})

    # Boundary case (9.9999994 vs 9.9999996): within tolerance, but a
    # round-then-compare scheme would straddle the 7th-sig-fig boundary and fail.
    assert _snapshot_matches({"x": np.array([9.9999994])}, {"x": np.array([9.9999996])})

    # A clearly larger difference must NOT match.
    coarse = {"x": values["x"].copy()}
    coarse["x"][0] = 1.001  # 1e-3 relative change
    assert not _snapshot_matches(values, coarse)

    # Array-wide coverage (what a statistics+samples fingerprint missed):
    rng = np.random.default_rng(0)
    base = {"a": rng.standard_normal(4096)}
    # Swapping two elements leaves order-insensitive statistics unchanged but is
    # a real change; it must be caught.
    swapped = {"a": base["a"].copy()}
    swapped["a"][[10, 3000]] = swapped["a"][[3000, 10]]
    assert not _snapshot_matches(base, swapped)
    # A single localized 7e-5 relative change anywhere must be caught.
    localized = {"a": base["a"].copy()}
    localized["a"][2000] *= 1 + 7e-5
    assert not _snapshot_matches(base, localized)

    # Complex data is compared too.
    z = {"z": np.array([1 + 2j, 3 - 4j])}
    assert _snapshot_matches(z, {"z": np.array([1 + 2j, 3 - 4j]) * (1 + 3e-8)})
    assert not _snapshot_matches(z, {"z": np.array([1 + 2j, 3 - 5j])})
