"""Tests for the public ``SpectralTransform`` contract of ``Connectivity.from_transform``.

A transform needs only ``fft()``, ``frequencies``, and ``time``; the optional
capability attributes (``is_one_sided``, ``observation_weights``,
``observations_are_independent``, ``time_bins_are_independent``) default to a
two-sided, unweighted, independent spectrum when absent. How the two
independence flags change the measures is covered in
``test_observation_independence.py``.
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
FFT_FREQUENCIES = np.fft.fftfreq(N_FFT_SAMPLES, d=1 / 500)


class _MinimalTransform:
    """Exposes only the required members of the contract."""

    def __init__(self, coefficients, frequencies):
        self._coefficients = coefficients
        self.frequencies = frequencies
        self.time = np.arange(coefficients.shape[0], dtype=float)

    def fft(self):
        return self._coefficients.copy()


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


def test_builtin_transforms_satisfy_the_protocol(builtin_transforms):
    for transform in builtin_transforms:
        assert isinstance(transform, SpectralTransform), type(transform).__name__


@pytest.mark.parametrize("missing", ["fft", "frequencies", "time"])
def test_objects_missing_a_required_member_do_not_satisfy_the_protocol(coefficients, missing):
    members = {
        "fft": lambda self: coefficients,
        "frequencies": FFT_FREQUENCIES,
        "time": np.arange(N_TIME_WINDOWS, dtype=float),
    }
    assert isinstance(type("Complete", (), members)(), SpectralTransform)
    del members[missing]
    assert not isinstance(type("Incomplete", (), members)(), SpectralTransform)
    assert not isinstance(coefficients, SpectralTransform)


def test_minimal_transform_gets_the_documented_defaults(coefficients):
    frequencies = FFT_FREQUENCIES
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
    transform = _MinimalTransform(coefficients, frequencies)
    transform.is_one_sided = True
    transform.observation_weights = weights
    transform.observations_are_independent = False
    transform.time_bins_are_independent = False

    connectivity = Connectivity.from_transform(transform)

    assert connectivity.is_one_sided is True
    np.testing.assert_array_equal(connectivity.frequencies, frequencies)
    np.testing.assert_array_equal(connectivity.observation_weights, weights)
    assert connectivity.observations_are_independent is False
    assert connectivity.time_bins_are_independent is False


@pytest.mark.parametrize("one_sided", [False, True])
def test_missing_frequencies_and_time_get_normalized_defaults(coefficients, one_sided):
    """``frequencies = None`` gives normalized frequencies (cycles per sample)
    and ``time = None`` the window indices, as the protocol documents."""
    transform = _MinimalTransform(coefficients, None)
    transform.time = None
    transform.is_one_sided = one_sided

    connectivity = Connectivity.from_transform(transform)

    expected = (
        np.linspace(0.0, 0.5, N_FFT_SAMPLES) if one_sided else np.fft.fftfreq(N_FFT_SAMPLES)
    )
    np.testing.assert_array_equal(connectivity.all_frequencies, expected)
    np.testing.assert_array_equal(connectivity.time, np.arange(N_TIME_WINDOWS))
    off_diagonal = ~np.eye(N_SIGNALS, dtype=bool)
    assert np.isfinite(connectivity.coherence_magnitude()[..., off_diagonal]).all()


def test_from_transform_forwards_dtype_and_wilson_options():
    series = np.random.default_rng(5).standard_normal((256, 4, 3))
    series[1:, :, 1] += 0.5 * series[:-1, :, 0]
    transform = Multitaper(series, sampling_frequency=256, time_halfbandwidth_product=3)

    # dtype sets the precision of the phase-locking computation.
    single = Connectivity.from_transform(transform, dtype=np.complex64)
    assert single.phase_locking_value().dtype == np.float32

    # One Wilson iteration cannot converge at the default tolerance ...
    capped = Connectivity.from_transform(transform, minimum_phase_max_iterations=1)
    with pytest.warns(UserWarning, match="did not converge"):
        assert np.isnan(capped.directed_transfer_function()).all()
    # ... but does at a tolerance loose enough to accept the first iterate.
    loose = Connectivity.from_transform(
        transform, minimum_phase_max_iterations=1, minimum_phase_tolerance=1e6
    )
    assert np.isfinite(loose.directed_transfer_function()).all()


_CAPABILITY_FLAGS = (
    "is_one_sided",
    "observations_are_independent",
    "time_bins_are_independent",
)


@pytest.mark.parametrize("flag", _CAPABILITY_FLAGS)
@pytest.mark.parametrize("value", [np.True_, np.False_, np.array(True), np.array(False)])
def test_capability_flags_accept_numpy_bools(coefficients, flag, value):
    """Including 0-d boolean arrays, e.g. a flag computed as ``xp.all(...)``."""
    transform = _MinimalTransform(coefficients, FFT_FREQUENCIES)
    if flag == "is_one_sided" and value:
        transform.frequencies = np.linspace(0.0, 250.0, N_FFT_SAMPLES)
    setattr(transform, flag, value)

    assert getattr(Connectivity.from_transform(transform), flag) is bool(value)


@pytest.mark.parametrize("flag", _CAPABILITY_FLAGS)
@pytest.mark.parametrize("kind", ["method", "string", "none", "int"])
def test_capability_flags_must_be_bools(coefficients, flag, kind):
    """``bool()`` would read a method or the string "False" as True, and None
    as False, silently flipping how the spectrum is treated."""

    def method(self):
        return False

    value = {"method": method, "string": "False", "none": None, "int": 0}[kind]
    transform_class = type("Transform", (_MinimalTransform,), {flag: value})
    transform = transform_class(coefficients, FFT_FREQUENCIES)

    with pytest.raises(TypeError, match=rf"transform\.{flag} must be a bool"):
        Connectivity.from_transform(transform)


def test_an_attribute_error_inside_a_capability_property_propagates(coefficients):
    """A bug inside an optional property must not be mistaken for the
    attribute being absent, which would silently apply the default."""

    class Buggy(_MinimalTransform):
        @property
        def is_one_sided(self):
            return self._one_sidded  # typo: raises AttributeError

    transform = Buggy(coefficients, FFT_FREQUENCIES)
    with pytest.raises(AttributeError, match="_one_sidded"):
        Connectivity.from_transform(transform)


@pytest.mark.parametrize(
    "frequencies",
    [
        np.linspace(0.0, 250.0, N_FFT_SAMPLES),  # one-sided, is_one_sided not set
        np.fft.fftshift(FFT_FREQUENCIES),
    ],
    ids=["one_sided_unflagged", "fftshifted"],
)
def test_two_sided_frequencies_must_be_in_fft_order(coefficients, frequencies):
    """Two-sided coefficients are folded assuming numpy.fft order, so any other
    layout would silently drop or mislabel frequencies."""
    transform = _MinimalTransform(coefficients, frequencies)

    with pytest.raises(ValueError, match="standard FFT order"):
        Connectivity.from_transform(transform)
    with pytest.raises(ValueError, match="standard FFT order"):
        Connectivity(coefficients, frequencies=frequencies)


def test_real_valued_coefficients_are_rejected(coefficients):
    """Real coefficients carry no phase, so the imaginary coherence and the
    phase-lag indices would be exactly 0 rather than an error."""
    transform = _MinimalTransform(coefficients.real, FFT_FREQUENCIES)
    with pytest.raises(TypeError, match="must be complex"):
        Connectivity.from_transform(transform)
    with pytest.raises(TypeError, match="must be complex"):
        Connectivity(coefficients.real)


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
        # (frequency, trials, signals) -> (1 window, trials, 1 taper, frequency, signals)
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
