"""Snapshot tests for tutorial notebooks.

These tests verify that key numerical outputs from the tutorial
notebooks remain stable across code changes. Tests are inspired
by notebook examples but hand-written for clarity and focus.

Uses syrupy with a custom NumPy extension for approximate (allclose) equality.
"""

import base64
import gzip
import json
import os
import subprocess
import sys
from pathlib import Path

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
    # mtime=0 keeps the gzip header constant, so identical data serializes to
    # identical bytes and a snapshot diff shows only arrays that changed.
    blob = base64.b64encode(gzip.compress(contiguous.tobytes(), 9, mtime=0)).decode("ascii")
    return {"array": {"shape": list(array.shape), "gzip_b64": blob}}


def test_encode_is_independent_of_wall_clock_time():
    """Identical data must serialize to identical bytes on different days, so a
    snapshot diff shows only arrays whose values changed."""
    from unittest.mock import patch

    array = np.arange(12.0).reshape(3, 4)
    with patch("time.time", return_value=1_000_000.0):
        first = _encode(array)
    with patch("time.time", return_value=2_000_000.0):
        second = _encode(array)
    assert first == second


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
        return len(a) == len(b) and all(
            _numeric_allclose(x, y) for x, y in zip(a, b, strict=True)
        )
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


def _white_noise_psd(noise_sd, sampling_frequency):
    """One-sided power spectral density of white noise, ``2 * sd**2 / fs``."""
    return 2 * noise_sd**2 / sampling_frequency


def _on_and_off_peak(values, frequencies, peak_frequency=200, exclusion_half_width=10):
    """Split ``values`` (frequency on axis 0) into the bin nearest ``peak_frequency``
    and the bins farther than ``exclusion_half_width`` Hz from it."""
    frequencies = np.abs(frequencies)
    on_peak = values[np.argmin(np.abs(frequencies - peak_frequency))]
    off_peak = values[np.abs(frequencies - peak_frequency) > exclusion_half_width]
    return on_peak, off_peak


def _assert_50hz_turns_on_at_25s(outputs, noise_sd, sampling_frequency, half_window):
    """200 Hz is present throughout; 50 Hz is at the noise floor before 25 s and
    well above it after (windows straddling 25 s are excluded)."""
    power = outputs["power"][..., 0]  # (n_windows, n_frequencies)
    frequencies, time = outputs["frequencies"], outputs["time"]
    at_50hz = power[:, np.argmin(np.abs(frequencies - 50))]
    at_200hz = power[:, np.argmin(np.abs(frequencies - 200))]
    before, after = time + half_window < 25, time - half_window > 25
    noise_floor = _white_noise_psd(noise_sd, sampling_frequency)
    np.testing.assert_allclose(at_50hz[before].mean(), noise_floor, rtol=0.25)
    assert at_50hz[after].mean() > 3 * noise_floor
    assert at_200hz[before].mean() > 3 * noise_floor
    assert at_200hz[after].mean() > 3 * noise_floor


def test_power_spectrum_200hz(snapshot):
    """Power spectrum of 200 Hz signal."""
    rng = np.random.default_rng(42)
    sampling_frequency = 1500
    time = np.linspace(0, 50, 75001, endpoint=True)
    signal = np.sin(2 * np.pi * time * 200)
    noise = rng.normal(0, 4, signal.shape)

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
    power, frequencies = outputs["power"][0, :, 0], outputs["frequencies"]
    assert frequencies[np.argmax(power)] == pytest.approx(200, abs=0.1)
    _, off_peak = _on_and_off_peak(power, frequencies, exclusion_half_width=5)
    np.testing.assert_allclose(
        off_peak.mean(), _white_noise_psd(4, sampling_frequency), rtol=0.05
    )
    assert outputs == snapshot


def test_coherence_magnitude_phase_offset(snapshot):
    """Coherence with fixed phase offset between signals."""
    rng = np.random.default_rng(42)
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
    noise = rng.normal(0, 4, signal.shape)

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
    on_peak, off_peak = _on_and_off_peak(
        outputs["coherence_magnitude"][0, :, 0, 1], outputs["frequencies"]
    )
    assert on_peak > 0.95
    assert np.median(off_peak) < 0.2
    # cross-spectrum[i, j] = E[X_i conj(X_j)], so its phase is phase_i - phase_j:
    # signal 1 leads signal 0 by pi/2, giving -pi/2 for [0, 1] and +pi/2 for [1, 0].
    phase, _ = _on_and_off_peak(connectivity.coherence_phase()[0], connectivity.frequencies)
    np.testing.assert_allclose(phase[0, 1], -np.pi / 2, atol=0.1)
    np.testing.assert_allclose(phase[1, 0], np.pi / 2, atol=0.1)
    assert outputs == snapshot


def test_spectrogram_temporal_dynamics(snapshot):
    """Spectrogram showing 50 Hz turning on at t=25s."""
    rng = np.random.default_rng(42)
    sampling_frequency = 1500
    time_extent = (0, 50)
    frequency_of_interest = [200, 50]
    n_time_samples = int(((time_extent[1] - time_extent[0]) * sampling_frequency) + 1)
    time = np.linspace(time_extent[0], time_extent[1], num=n_time_samples, endpoint=True)

    # Create signal with 200 Hz constant, 50 Hz turns on at t=25s
    signal = np.sin(2 * np.pi * time[:, np.newaxis] * frequency_of_interest)
    signal[: n_time_samples // 2, 1] = 0  # 50 Hz only in second half
    signal = signal.sum(axis=1)
    noise = rng.normal(0, 4, signal.shape)

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
    _assert_50hz_turns_on_at_25s(outputs, 4, sampling_frequency, half_window=0.3)
    assert outputs == snapshot


def test_coherogram_phase_change(snapshot):
    """Coherogram showing phase offset changing at t=1.5s."""
    rng = np.random.default_rng(42)
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
    phase_offset = rng.uniform(-np.pi, np.pi, size=(n_time_samples, n_trials))
    phase_offset[np.where(time > 1.5), :] = np.pi / 2
    signal[:, :, 1] = np.sin(
        (2 * np.pi * time[:, np.newaxis] * frequency_of_interest) + phase_offset
    )
    noise = rng.normal(0, 2, signal.shape)

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
    frequencies = connectivity.frequencies
    at_200hz = outputs["coherence_magnitude"][:, np.argmin(np.abs(frequencies - 200)), 0, 1]
    half_window = 0.040
    # Random phase (incoherent) in windows ending before 1.5 s; fixed phase after.
    assert np.all(at_200hz[outputs["time"] + half_window < 1.5] < 0.1)
    assert np.all(at_200hz[outputs["time"] - half_window > 1.5] > 0.5)
    assert outputs == snapshot


def test_power_spectrum_30hz(snapshot):
    """Power spectrum of 30 Hz signal."""
    rng = np.random.default_rng(42)
    frequency_of_interest = 30
    sampling_frequency = 1500
    time_extent = (0, 50)
    n_time_samples = int(((time_extent[1] - time_extent[0]) * sampling_frequency) + 1)
    time = np.linspace(time_extent[0], time_extent[1], num=n_time_samples, endpoint=True)
    signal = np.sin(2 * np.pi * time * frequency_of_interest)
    noise = rng.normal(0, 4, len(signal))

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
    power, frequencies = outputs["power"][0, :, 0], outputs["frequencies"]
    assert frequencies[np.argmax(power)] == pytest.approx(frequency_of_interest, abs=0.1)
    _, off_peak = _on_and_off_peak(
        power, frequencies, peak_frequency=frequency_of_interest, exclusion_half_width=5
    )
    np.testing.assert_allclose(
        off_peak.mean(), _white_noise_psd(4, sampling_frequency), rtol=0.05
    )
    assert outputs == snapshot


def test_spectrogram_with_trials(snapshot):
    """Spectrogram with trial structure (time x trials)."""
    rng = np.random.default_rng(42)
    sampling_frequency = 1500
    time_extent = (0, 50)
    frequency_of_interest = [200, 50]
    n_time_samples = int(((time_extent[1] - time_extent[0]) * sampling_frequency) + 1)
    n_trials = 10
    time = np.linspace(time_extent[0], time_extent[1], num=n_time_samples, endpoint=True)

    # Create signal with 200 Hz constant, 50 Hz turns on at t=25s
    signal = np.sin(2 * np.pi * time[:, np.newaxis] * frequency_of_interest)
    signal[: n_time_samples // 2, 1] = 0  # 50 Hz only in second half
    signal = signal.sum(axis=1)

    # Replicate across trials with noise
    signal = np.tile(signal[:, np.newaxis], (1, n_trials))
    noise = rng.normal(0, 4, signal.shape)

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
    _assert_50hz_turns_on_at_25s(outputs, 4, sampling_frequency, half_window=0.3)
    assert outputs == snapshot


def test_spectrogram_finer_frequency_resolution(snapshot):
    """Spectrogram with a smaller time-halfbandwidth product (finer resolution).

    Lowering TW from 3 to 1 narrows the multitaper bandwidth 2W = 2 * TW / T from
    10 Hz to 3.3 Hz for 0.6 s windows, so the spectral peaks become narrower.
    """
    rng = np.random.default_rng(42)
    sampling_frequency = 1500
    time_extent = (0, 50)
    frequency_of_interest = [200, 50]
    n_time_samples = int(((time_extent[1] - time_extent[0]) * sampling_frequency) + 1)
    time = np.linspace(time_extent[0], time_extent[1], num=n_time_samples, endpoint=True)

    signal = np.sin(2 * np.pi * time[:, np.newaxis] * frequency_of_interest)
    signal[: n_time_samples // 2, 1] = 0
    signal = signal.sum(axis=1)
    noise = rng.normal(0, 4, signal.shape)

    multitaper = Multitaper(
        prepare_time_series(signal + noise),
        sampling_frequency=sampling_frequency,
        time_halfbandwidth_product=1,  # 2W = 3.3 Hz (vs 10 Hz for TW = 3)
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
    _assert_50hz_turns_on_at_25s(outputs, 4, sampling_frequency, half_window=0.3)
    # The contiguous half-power width of each (time-averaged, post-onset) peak
    # must fit within the bandwidth 2W = 2 * TW / T.
    frequencies = outputs["frequencies"]
    mean_power = outputs["power"][outputs["time"] - 0.3 > 25, :, 0].mean(axis=0)
    bandwidth = 2 * 1 / 0.600
    for peak_frequency in (50, 200):
        peak = np.argmin(np.abs(frequencies - peak_frequency))
        above_half = mean_power >= mean_power[peak] / 2
        low = peak
        while above_half[low - 1]:
            low -= 1
        high = peak
        while above_half[high + 1]:
            high += 1
        assert frequencies[high] - frequencies[low] <= bandwidth
    assert outputs == snapshot


@pytest.fixture(scope="module")
def phase_offset_trials():
    """Connectivity for 100 trials of a 200 Hz pair, signal 1 leading by pi/2.

    Shared by the trial-based phase-synchrony tests; every measure is computed
    from the same (cached) cross-spectral quantities.
    """
    rng = np.random.default_rng(42)
    sampling_frequency = 1500
    time_extent = (0, 2.400)
    n_trials = 100
    n_signals = 2
    frequency_of_interest = 200
    n_time_samples = int(((time_extent[1] - time_extent[0]) * sampling_frequency) + 1)
    time = np.linspace(time_extent[0], time_extent[1], num=n_time_samples, endpoint=True)

    signal = np.zeros((n_time_samples, n_trials, n_signals))
    signal[:, :, 0] = np.sin(2 * np.pi * time[:, np.newaxis] * frequency_of_interest)
    phase_offset = np.pi / 2
    signal[:, :, 1] = np.sin(
        (2 * np.pi * time[:, np.newaxis] * frequency_of_interest) + phase_offset
    )
    noise = rng.normal(0, 2, signal.shape)

    multitaper = Multitaper(
        prepare_time_series(signal + noise),
        sampling_frequency=sampling_frequency,
        time_halfbandwidth_product=1,
        start_time=time[0],
    )
    return Connectivity.from_multitaper(multitaper)


def _phase_measure_outputs(connectivity, name, method):
    """Snapshot dict for one measure plus its [0, 1] / [1, 0] on- and off-peak values."""
    outputs = {name: getattr(connectivity, method)(), "frequencies": connectivity.frequencies}
    values = outputs[name][0]  # (n_frequencies, n_signals, n_signals)
    on_peak, off_peak = _on_and_off_peak(values, outputs["frequencies"])
    return outputs, on_peak, off_peak[:, 0, 1]


def test_coherence_with_trials(phase_offset_trials, snapshot):
    """Coherence with trial structure, 200 Hz, pi/2 phase offset."""
    outputs, on_peak, off_peak = _phase_measure_outputs(
        phase_offset_trials, "coherence_magnitude", "coherence_magnitude"
    )
    assert on_peak[0, 1] > 0.95
    assert np.median(off_peak) < 0.05
    assert outputs == snapshot


def test_imaginary_coherence(phase_offset_trials, snapshot):
    """Imaginary coherence with phase offset.

    A pi/2 lag is purely imaginary, so the (magnitude) imaginary coherence is ~1.
    """
    outputs, on_peak, off_peak = _phase_measure_outputs(
        phase_offset_trials, "imaginary_coherence", "imaginary_coherence"
    )
    assert on_peak[0, 1] > 0.95
    assert np.median(off_peak) < 0.1
    assert outputs == snapshot


def test_phase_locking_value(phase_offset_trials, snapshot):
    """Phase locking value with phase offset."""
    outputs, on_peak, off_peak = _phase_measure_outputs(
        phase_offset_trials, "phase_locking_value", "phase_locking_value"
    )
    assert on_peak[0, 1] > 0.95
    assert np.median(off_peak) < 0.15
    assert outputs == snapshot


def test_phase_lag_index(phase_offset_trials, snapshot):
    """Phase lag index with phase offset.

    Im(cross-spectrum[0, 1]) = Im(E[X_0 conj(X_1)]) < 0 because signal 1 leads by
    pi/2, so the signed PLI is -1 for [0, 1] and +1 for [1, 0].
    """
    outputs, on_peak, off_peak = _phase_measure_outputs(
        phase_offset_trials, "phase_lag_index", "phase_lag_index"
    )
    assert on_peak[0, 1] < -0.95
    assert on_peak[1, 0] == pytest.approx(-on_peak[0, 1])
    assert np.abs(np.median(off_peak)) < 0.05
    assert outputs == snapshot


def test_weighted_phase_lag_index(phase_offset_trials, snapshot):
    """Weighted phase lag index with phase offset (same sign convention as PLI)."""
    outputs, on_peak, off_peak = _phase_measure_outputs(
        phase_offset_trials, "weighted_phase_lag_index", "weighted_phase_lag_index"
    )
    assert on_peak[0, 1] < -0.95
    assert on_peak[1, 0] == pytest.approx(-on_peak[0, 1])
    assert np.abs(np.median(off_peak)) < 0.05
    assert outputs == snapshot


def test_debiased_squared_weighted_phase_lag_index(phase_offset_trials, snapshot):
    """Debiased squared weighted phase lag index."""
    outputs, on_peak, off_peak = _phase_measure_outputs(
        phase_offset_trials,
        "debiased_squared_wpli",
        "debiased_squared_weighted_phase_lag_index",
    )
    assert on_peak[0, 1] > 0.95
    # NaN where the cross-spectrum is purely real (DC and Nyquist): 0 / 0.
    assert np.abs(np.nanmedian(off_peak)) < 0.05
    assert outputs == snapshot


def test_pairwise_phase_consistency(phase_offset_trials, snapshot):
    """Pairwise phase consistency with phase offset."""
    outputs, on_peak, off_peak = _phase_measure_outputs(
        phase_offset_trials, "pairwise_phase_consistency", "pairwise_phase_consistency"
    )
    assert on_peak[0, 1] > 0.95
    assert np.abs(np.median(off_peak)) < 0.05
    assert outputs == snapshot


def _lagged_broadband_pair(rng, n_time_samples, lag_samples, leader, noise_sd, n_trials=None):
    """Two noisy copies of one white-noise source, the leader ``lag_samples`` ahead.

    A broadband source is needed: a pure sinusoid delayed by a whole number of
    cycles (e.g. 200 Hz by 10 ms) is indistinguishable from the original, so it
    carries no lag information for group delay or the phase slope index.
    Returns shape ``(n_time_samples, n_signals)`` or, with ``n_trials``,
    ``(n_time_samples, n_trials, n_signals)``.
    """
    extra = () if n_trials is None else (n_trials,)
    source = rng.standard_normal((n_time_samples + lag_samples, *extra))
    ahead, behind = (
        source[lag_samples:],
        source[:n_time_samples],
    )  # behind[t] == ahead[t - lag]
    pair = [ahead, behind] if leader == 0 else [behind, ahead]
    signal = np.stack(pair, axis=-1)
    return signal + rng.normal(0, noise_sd, signal.shape)


def test_group_delay_signal1_leads(snapshot):
    """Group delay: Signal #1 leads Signal #2."""
    rng = np.random.default_rng(42)
    sampling_frequency = 1500
    time_extent = (0, 50)
    n_time_samples = int(((time_extent[1] - time_extent[0]) * sampling_frequency) + 1)
    time = np.linspace(time_extent[0], time_extent[1], num=n_time_samples, endpoint=True)

    time_lag = 0.010  # 10 ms = 15 samples
    signal = _lagged_broadband_pair(
        rng,
        n_time_samples,
        round(time_lag * sampling_frequency),
        leader=0,
        noise_sd=0.25,
    )

    multitaper = Multitaper(
        prepare_time_series(signal, axis="signals"),
        sampling_frequency=sampling_frequency,
        time_halfbandwidth_product=5,
        start_time=time[0],
    )
    connectivity = Connectivity.from_multitaper(multitaper)

    outputs = {
        "group_delay": connectivity.group_delay(),
        "frequencies": connectivity.frequencies,
    }
    delay = outputs["group_delay"][0]
    np.testing.assert_allclose(delay[..., 0, 1], time_lag, atol=5e-4)
    np.testing.assert_allclose(delay[..., 1, 0], -time_lag, atol=5e-4)
    assert outputs == snapshot


def test_group_delay_signal2_leads(snapshot):
    """Group delay: Signal #2 leads Signal #1."""
    rng = np.random.default_rng(42)
    sampling_frequency = 1500
    time_extent = (0, 50)
    n_time_samples = int(((time_extent[1] - time_extent[0]) * sampling_frequency) + 1)
    time = np.linspace(time_extent[0], time_extent[1], num=n_time_samples, endpoint=True)

    time_lag = 0.010  # 10 ms = 15 samples
    signal = _lagged_broadband_pair(
        rng,
        n_time_samples,
        round(time_lag * sampling_frequency),
        leader=1,
        noise_sd=0.25,
    )

    multitaper = Multitaper(
        prepare_time_series(signal, axis="signals"),
        sampling_frequency=sampling_frequency,
        time_halfbandwidth_product=5,
        start_time=time[0],
    )
    connectivity = Connectivity.from_multitaper(multitaper)

    outputs = {
        "group_delay": connectivity.group_delay(),
        "frequencies": connectivity.frequencies,
    }
    delay = outputs["group_delay"][0]
    np.testing.assert_allclose(delay[..., 0, 1], -time_lag, atol=5e-4)
    assert outputs == snapshot


def test_group_delay_signal2_leads_over_time(snapshot):
    """Group delay: Signal #2 leads Signal #1 over time (with trials)."""
    rng = np.random.default_rng(42)
    sampling_frequency = 1500
    time_extent = (0, 2.400)
    n_trials = 100  # Need trials for sufficient observations with windowing
    n_time_samples = int(((time_extent[1] - time_extent[0]) * sampling_frequency) + 1)
    time = np.linspace(time_extent[0], time_extent[1], num=n_time_samples, endpoint=True)

    time_lag = 0.010  # 10 ms = 15 samples
    # Signal 2 leads (appears first in time)
    signal = _lagged_broadband_pair(
        rng,
        n_time_samples,
        round(time_lag * sampling_frequency),
        leader=1,
        noise_sd=1.0,
        n_trials=n_trials,
    )

    multitaper = Multitaper(
        prepare_time_series(signal),
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
    delay = outputs["group_delay"][0]
    np.testing.assert_allclose(delay[..., 0, 1], -time_lag, atol=5e-4)  # every window
    assert outputs == snapshot


def test_phase_slope_index_signal1_leads(snapshot):
    """Phase slope index: Signal #1 leads Signal #2."""
    rng = np.random.default_rng(42)
    sampling_frequency = 1500
    time_extent = (0, 50)
    n_time_samples = int(((time_extent[1] - time_extent[0]) * sampling_frequency) + 1)
    time = np.linspace(time_extent[0], time_extent[1], num=n_time_samples, endpoint=True)

    time_lag = 0.010  # 10 ms = 15 samples
    signal = _lagged_broadband_pair(
        rng,
        n_time_samples,
        round(time_lag * sampling_frequency),
        leader=0,
        noise_sd=0.25,
    )

    multitaper = Multitaper(
        prepare_time_series(signal, axis="signals"),
        sampling_frequency=sampling_frequency,
        time_halfbandwidth_product=5,
        start_time=time[0],
    )
    connectivity = Connectivity.from_multitaper(multitaper)

    outputs = {
        "phase_slope_index": connectivity.phase_slope_index(),
        "frequencies": connectivity.frequencies,
    }
    psi = outputs["phase_slope_index"]
    assert np.all(psi[..., 0, 1] > 0)  # positive [0, 1]: signal 1 leads signal 2
    np.testing.assert_allclose(psi[..., 1, 0], -psi[..., 0, 1])
    assert outputs == snapshot


def test_phase_slope_index_signal2_leads(snapshot):
    """Phase slope index: Signal #2 leads Signal #1."""
    rng = np.random.default_rng(42)
    sampling_frequency = 1500
    time_extent = (0, 50)
    n_time_samples = int(((time_extent[1] - time_extent[0]) * sampling_frequency) + 1)
    time = np.linspace(time_extent[0], time_extent[1], num=n_time_samples, endpoint=True)

    time_lag = 0.010  # 10 ms = 15 samples
    signal = _lagged_broadband_pair(
        rng,
        n_time_samples,
        round(time_lag * sampling_frequency),
        leader=1,
        noise_sd=0.25,
    )

    multitaper = Multitaper(
        prepare_time_series(signal, axis="signals"),
        sampling_frequency=sampling_frequency,
        time_halfbandwidth_product=5,
        start_time=time[0],
    )
    connectivity = Connectivity.from_multitaper(multitaper)

    outputs = {
        "phase_slope_index": connectivity.phase_slope_index(),
        "frequencies": connectivity.frequencies,
    }
    psi = outputs["phase_slope_index"]
    assert np.all(psi[..., 0, 1] < 0)  # negative [0, 1]: signal 2 leads signal 1
    np.testing.assert_allclose(psi[..., 1, 0], -psi[..., 0, 1])
    assert outputs == snapshot


def _canonical_coherence_two_groups(n_per_group, noise_sd):
    """Canonical coherence between two groups of noisy 200 Hz signals.

    Group 1 carries the same sinusoid; group 2 carries it shifted by pi/2. The
    sinusoid's phase is random across trials but shared by both groups. 20 trials
    x 9 tapers = 180 observations, far more than the 2 * ``n_per_group``
    signals, so the off-peak canonical coherence is a genuine noise floor (with
    fewer observations than signals it is identically 1).
    """
    rng = np.random.default_rng(42)
    sampling_frequency = 1500
    n_trials = 20
    time = np.arange(int(2.0 * sampling_frequency) + 1) / sampling_frequency
    frequency_of_interest = 200

    trial_phase = rng.uniform(-np.pi, np.pi, size=n_trials)
    group1 = np.sin(2 * np.pi * frequency_of_interest * time[:, np.newaxis] + trial_phase)
    group2 = np.sin(
        2 * np.pi * frequency_of_interest * time[:, np.newaxis] + trial_phase + np.pi / 2
    )
    signal = np.concatenate(
        [
            np.repeat(group1[..., np.newaxis], n_per_group, axis=-1),
            np.repeat(group2[..., np.newaxis], n_per_group, axis=-1),
        ],
        axis=-1,
    )  # (n_time_samples, n_trials, 2 * n_per_group)
    noise = rng.normal(0, noise_sd, signal.shape)

    multitaper = Multitaper(
        signal + noise,
        sampling_frequency=sampling_frequency,
        time_halfbandwidth_product=5,
        start_time=time[0],
    )
    connectivity = Connectivity.from_multitaper(multitaper)
    assert connectivity.n_observations == 180

    outputs = {
        "canonical_coherence": connectivity.canonical_coherence(
            np.array([0] * n_per_group + [1] * n_per_group)
        )[0],
        "frequencies": connectivity.frequencies,
    }
    on_peak, off_peak = _on_and_off_peak(
        outputs["canonical_coherence"][0, :, 0, 1], outputs["frequencies"]
    )
    return outputs, on_peak, off_peak


def test_canonical_coherence(snapshot):
    """Canonical coherence between two groups of three signals."""
    outputs, on_peak, off_peak = _canonical_coherence_two_groups(n_per_group=3, noise_sd=4)
    assert on_peak > 0.5
    assert on_peak > 10 * np.median(off_peak)
    assert outputs == snapshot


def test_canonical_coherence_high_noise(snapshot):
    """Canonical coherence with more signals per group and higher noise."""
    outputs, on_peak, off_peak = _canonical_coherence_two_groups(n_per_group=5, noise_sd=8)
    assert on_peak > 0.5
    assert on_peak > 5 * np.median(off_peak)
    assert outputs == snapshot


def test_global_coherence(snapshot):
    """Global coherence across multiple signals."""
    rng = np.random.default_rng(42)
    sampling_frequency = 1500
    time_extent = (0, 50)
    n_time_samples = int(((time_extent[1] - time_extent[0]) * sampling_frequency) + 1)
    time = np.linspace(time_extent[0], time_extent[1], num=n_time_samples, endpoint=True)

    frequency_of_interest = 200
    n_signals = 5
    signal = np.zeros((n_time_samples, n_signals))
    base_signal = np.sin(2 * np.pi * time * frequency_of_interest)

    # All signals are the same base with different noise
    for i in range(n_signals):
        signal[:, i] = base_signal

    noise = rng.normal(0, 4, signal.shape)

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
    # global_coherence spans all FFT bins, so pair it with all_frequencies.
    on_peak, off_peak = _on_and_off_peak(
        outputs["global_coherence"][0, :, 0], connectivity.all_frequencies
    )
    assert on_peak > 0.95
    assert on_peak > 2 * np.median(off_peak)
    assert outputs == snapshot


# ============ Tutorial_Using_Paper_Examples tests ============
# Only keeping 3 representative MVAR examples. Directed measures follow the
# [i, j] = j -> i convention, matching simulate_MVAR's coefficients[lag, i, j]
# (x_i(t) += coefficients[lag, i, j] * x_j(t - lag - 1)).


def _mean_over_frequencies(measure):
    """Average a (1, n_frequencies, n_signals, n_signals) measure over frequency.

    Only off-diagonal entries are meaningful (the Granger diagonal is NaN).
    """
    mean = np.full(measure.shape[-2:], np.nan)
    off_diagonal = ~np.eye(measure.shape[-1], dtype=bool)
    mean[off_diagonal] = measure[0][:, off_diagonal].mean(axis=0)
    return mean


def test_baccala_example2(snapshot):
    """Baccala Example 2: Partial directed coherence (representative PDC example)."""
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
    # PDC reflects direct coupling only: coefficients[0, 2, 0] == 0 (no 0 -> 2),
    # while every other off-diagonal coefficient is non-zero.
    pdc = _mean_over_frequencies(outputs["partial_directed_coherence"])
    assert pdc[2, 0] < 0.03
    for i, j in [(0, 1), (0, 2), (1, 0), (1, 2), (2, 1)]:
        assert pdc[i, j] > 0.05
    assert outputs == snapshot


def test_dtf_and_ddtf_one_way_coupled_var(snapshot):
    """DTF and direct DTF for a two-signal VAR(1) where only signal 0 drives signal 1.

    (A simplified one-lag model, not the tutorial's two-lag Ding Example 1.)
    """
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
    # coefficients[0, 1, 0] = 0.4 (0 -> 1); coefficients[0, 0, 1] = 0 (no 1 -> 0).
    for measure in ("directed_transfer_function", "direct_directed_transfer_function"):
        mean = _mean_over_frequencies(outputs[measure])
        assert mean[0, 1] < 0.01
        assert mean[1, 0] > 10 * mean[0, 1]
    assert outputs == snapshot


def test_conditional_granger_three_signal_regression(snapshot):
    """Regression coverage for conditional Granger on a three-signal VAR."""
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
    # Signal 2 drives neither signal 0 nor signal 1 (coefficients[0, :2, 2] == 0);
    # 0 <-> 1, 0 -> 2 and 1 -> 2 are all coupled.
    for measure in ("pairwise_spectral_granger", "conditional_spectral_granger"):
        granger = _mean_over_frequencies(outputs[measure])
        assert granger[0, 2] < 0.02
        assert granger[1, 2] < 0.02
        for i, j in [(0, 1), (1, 0), (2, 0), (2, 1)]:
            assert granger[i, j] > 0.05
    assert outputs == snapshot


_EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


@pytest.mark.slow
@pytest.mark.parametrize(
    "notebook", ["Tutorial_On_Simulated_Examples.ipynb", "Tutorial_Using_Paper_Examples.ipynb"]
)
def test_tutorial_notebook_executes(notebook, tmp_path):
    """The tutorial notebook executes without errors.

    Runs nbconvert with this interpreter (via a kernelspec pinned to
    ``sys.executable``), from any working directory, and writes the executed
    copy to ``tmp_path`` so the repository is never modified.
    """
    pytest.importorskip("nbconvert")
    pytest.importorskip("ipykernel")
    kernel_name = "spectral-connectivity-tests"
    kernel_dir = tmp_path / "jupyter" / "kernels" / kernel_name
    kernel_dir.mkdir(parents=True)
    (kernel_dir / "kernel.json").write_text(
        json.dumps(
            {
                "argv": [
                    sys.executable,
                    "-m",
                    "ipykernel_launcher",
                    "-f",
                    "{connection_file}",
                ],
                "display_name": kernel_name,
                "language": "python",
            }
        )
    )
    env = {**os.environ, "JUPYTER_PATH": str(tmp_path / "jupyter")}

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--ExecutePreprocessor.timeout=600",
            f"--ExecutePreprocessor.kernel_name={kernel_name}",
            f"--output-dir={tmp_path}",
            str(_EXAMPLES_DIR / notebook),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode == 0, (
        f"Notebook execution failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    )
    assert (tmp_path / notebook).exists()


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
