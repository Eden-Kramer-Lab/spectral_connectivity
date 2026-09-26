"""Tests for the public ``SpectralTransform`` contract of ``Connectivity.from_transform``.

A transform needs only ``fft()``, ``frequencies``, and ``time``; the optional
capability attributes (``is_one_sided``, ``observation_weights``,
``observations_are_independent``, ``time_bins_are_independent``) default to a
two-sided, unweighted, independent spectrum when absent. The forwarding of the
two independence flags is covered in ``test_observation_independence.py``.
"""

import numpy as np
import pytest

import spectral_connectivity
from spectral_connectivity import (
    Connectivity,
    MorletWavelet,
    Multitaper,
    ShortTimeFourierTransform,
    SpectralTransform,
    Welch,
)

N_TIME_WINDOWS, N_TRIALS, N_TAPERS, N_FFT_SAMPLES, N_SIGNALS = 2, 4, 3, 16, 2


class _MinimalTransform:
    """Exposes only the required members of the contract."""

    def __init__(self, coefficients, frequencies):
        self._coefficients = coefficients
        self.frequencies = frequencies
        self.time = np.arange(coefficients.shape[0], dtype=float)

    def fft(self):
        return self._coefficients.copy()


class _OneSidedWeightedTransform(_MinimalTransform):
    """Also exposes every optional capability attribute, all non-default."""

    is_one_sided = True
    observations_are_independent = False
    time_bins_are_independent = False

    def __init__(self, coefficients, frequencies, observation_weights):
        super().__init__(coefficients, frequencies)
        self.observation_weights = observation_weights


@pytest.fixture
def coefficients():
    rng = np.random.default_rng(0)
    shape = (N_TIME_WINDOWS, N_TRIALS, N_TAPERS, N_FFT_SAMPLES, N_SIGNALS)
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


@pytest.fixture
def builtin_transforms():
    time_series = np.random.default_rng(1).standard_normal((1000, 2, 2))
    return [
        Multitaper(time_series, sampling_frequency=500, time_window_duration=0.5),
        ShortTimeFourierTransform(
            time_series, sampling_frequency=500, time_window_duration=0.5
        ),
        Welch(time_series, sampling_frequency=500, segment_duration=0.5),
        MorletWavelet(time_series, sampling_frequency=500, frequencies=[10.0, 20.0]),
    ]


def test_protocol_is_exported_from_the_package():
    assert "SpectralTransform" in spectral_connectivity.__all__
    assert spectral_connectivity.SpectralTransform is SpectralTransform


def test_builtin_transforms_satisfy_the_protocol(builtin_transforms):
    for transform in builtin_transforms:
        assert isinstance(transform, SpectralTransform), type(transform).__name__


def test_objects_missing_a_required_member_do_not_satisfy_the_protocol(coefficients):
    frequencies = np.fft.fftfreq(N_FFT_SAMPLES)
    assert isinstance(_MinimalTransform(coefficients, frequencies), SpectralTransform)

    class NoTime:
        frequencies = np.fft.fftfreq(N_FFT_SAMPLES)

        def fft(self):
            return coefficients

    class NoFFT:
        frequencies = np.fft.fftfreq(N_FFT_SAMPLES)
        time = np.arange(N_TIME_WINDOWS, dtype=float)

    assert not isinstance(NoTime(), SpectralTransform)
    assert not isinstance(NoFFT(), SpectralTransform)
    assert not isinstance(coefficients, SpectralTransform)


def test_minimal_transform_gets_the_documented_defaults(coefficients):
    frequencies = np.fft.fftfreq(N_FFT_SAMPLES, d=1 / 500)
    transform = _MinimalTransform(coefficients, frequencies)

    connectivity = Connectivity.from_transform(transform)

    assert connectivity.is_one_sided is False
    assert connectivity.observation_weights is None
    assert connectivity.observations_are_independent is True
    assert connectivity.time_bins_are_independent is True
    np.testing.assert_array_equal(connectivity.time, transform.time)
    np.testing.assert_array_equal(connectivity.all_frequencies, frequencies)
    expected = Connectivity(
        coefficients, frequencies=frequencies, time=transform.time
    ).coherence_magnitude()
    np.testing.assert_allclose(connectivity.coherence_magnitude(), expected)


def test_optional_capability_attributes_are_honored(coefficients):
    frequencies = np.linspace(0.0, 250.0, N_FFT_SAMPLES)
    weights = np.random.default_rng(2).uniform(
        0.5, 1.5, (N_TIME_WINDOWS, N_TRIALS, N_TAPERS, N_FFT_SAMPLES, 1)
    )
    transform = _OneSidedWeightedTransform(coefficients, frequencies, weights)

    connectivity = Connectivity.from_transform(transform)

    assert connectivity.is_one_sided is True
    np.testing.assert_array_equal(connectivity.frequencies, frequencies)
    np.testing.assert_array_equal(connectivity.observation_weights, weights)
    assert connectivity.observations_are_independent is False
    assert connectivity.time_bins_are_independent is False


class _DensityScaledTransform:
    """One unit-energy Hann-windowed FFT per trial, scaled as SpectralTransform
    documents for ``power()`` to be a power spectral density."""

    def __init__(self, time_series, sampling_frequency, one_sided):
        n_samples = time_series.shape[0]
        window = np.hanning(n_samples)
        window /= np.linalg.norm(window)  # unit energy
        windowed = window[:, np.newaxis, np.newaxis] * time_series
        if one_sided:
            coefficients = np.fft.rfft(windowed, axis=0) / np.sqrt(sampling_frequency)
            # Fold the negative frequencies in: every bin except DC and an
            # even-length Nyquist carries twice the density.
            n_interior_end = (n_samples + 1) // 2
            coefficients[1:n_interior_end] *= np.sqrt(2.0)
            self.frequencies = np.fft.rfftfreq(n_samples, 1 / sampling_frequency)
            self.is_one_sided = True
        else:
            coefficients = np.fft.fft(windowed, axis=0) / np.sqrt(sampling_frequency)
            self.frequencies = np.fft.fftfreq(n_samples, 1 / sampling_frequency)
        # (time, trials, signals) -> (1 window, trials, 1 taper, frequency, signals)
        self._coefficients = np.moveaxis(coefficients, 0, 1)[np.newaxis, :, np.newaxis]
        self.time = np.array([0.0])
        self.energy = np.mean(np.sum(windowed**2, axis=0), axis=0)  # (n_signals,)

    def fft(self):
        return self._coefficients.copy()


@pytest.mark.parametrize("one_sided", [False, True])
@pytest.mark.parametrize("n_samples", [16, 15])
def test_documented_scaling_makes_power_a_density(one_sided, n_samples):
    """With the scaling the SpectralTransform docstring gives, ``power()``
    integrates over frequency to the windowed signal's energy (Parseval)."""
    sampling_frequency = 250.0
    time_series = np.random.default_rng(4).standard_normal((n_samples, 6, N_SIGNALS))
    transform = _DensityScaledTransform(time_series, sampling_frequency, one_sided)

    power = Connectivity.from_transform(transform).power()

    frequency_step = sampling_frequency / n_samples
    np.testing.assert_allclose(
        np.sum(power[0], axis=0) * frequency_step, transform.energy, rtol=1e-12
    )
