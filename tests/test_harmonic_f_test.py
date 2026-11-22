"""Thomson's harmonic F-test, Multitaper.harmonic_f_test."""

import numpy as np
import pytest

from spectral_connectivity import HarmonicFTestResult, Multitaper
from tests._backend_helpers import to_host

_SAMPLING_FREQUENCY = 1000.0


def _line(time, amplitude, frequency, phase):
    return amplitude * np.cos(2 * np.pi * frequency * time + phase)


def test_complex_amplitude_is_the_line_at_each_window_center():
    """The fitted complex amplitude is linear in the data, so over noise
    replicates (trials) its mean is the line's amplitude with its phase at
    each window's center time, the time Multitaper labels the window by. Its
    phase at the window's first sample would differ by 2 pi f times half the
    window, far outside the limit."""
    n_time, n_trials = 2000, 200
    # Half a window (0.2495 s) of 62 Hz is 15.47 cycles: the phase at the
    # window's start and at its center differ by nearly pi.
    amplitude, frequency, phase = 0.8, 62.0, 1.0
    time = np.arange(n_time) / _SAMPLING_FREQUENCY
    rng = np.random.default_rng(300)
    signals = rng.standard_normal((n_time, n_trials, 1))
    signals += _line(time, amplitude, frequency, phase)[:, np.newaxis, np.newaxis]
    multitaper = Multitaper(
        signals,
        _SAMPLING_FREQUENCY,
        time_window_duration=0.5,
        time_window_step=0.25,
        time_halfbandwidth_product=4,
    )
    result = multitaper.harmonic_f_test()
    at_line = result.frequencies == frequency
    # (window, trial): the fitted line as amplitude * exp(i phase)
    estimate = (result.amplitude * np.exp(1j * result.phase))[:, :, at_line, 0][..., 0]
    expected = amplitude * np.exp(1j * (2 * np.pi * frequency * result.time + phase))
    mean = estimate.mean(axis=1)
    for part in (np.real, np.imag):
        standard_error = part(estimate).std(axis=1, ddof=1) / np.sqrt(n_trials)
        np.testing.assert_array_less(np.abs(part(mean) - part(expected)), 5 * standard_error)
    # The window-start phase is many limits away.
    start = result.time - (multitaper.n_time_samples_per_window - 1) / (
        2 * _SAMPLING_FREQUENCY
    )
    at_start = amplitude * np.exp(1j * (2 * np.pi * frequency * start + phase))
    noise = estimate.std(axis=1) / np.sqrt(n_trials)  # the complex standard error
    assert (np.abs(mean - at_start) > 50 * noise).all()


def test_p_values_are_uniform_without_a_line():
    """With no line the F statistic follows F(2, 2K - 2), so the p-values are
    uniform: the fraction below each level is that level, within five binomial
    standard errors. Bins at least 2W apart are nearly independent."""
    n_time, n_trials = 1000, 400
    rng = np.random.default_rng(301)
    multitaper = Multitaper(
        rng.standard_normal((n_time, n_trials, 2)),
        _SAMPLING_FREQUENCY,
        time_halfbandwidth_product=4,
    )
    result = multitaper.harmonic_f_test()
    spacing = int(np.ceil(2 * result.half_bandwidth / result.frequencies[1]))
    defined = np.flatnonzero(np.isfinite(result.p_value[0, 0, :, 0]))
    p_values = result.p_value[:, :, defined[::spacing], :].ravel()
    assert p_values.size > 10_000
    for level in (0.01, 0.05, 0.5):
        fraction = np.mean(p_values < level)
        standard_error = np.sqrt(level * (1 - level) / p_values.size)
        assert abs(fraction - level) < 5 * standard_error, (level, fraction)


def test_matches_a_least_squares_fit_of_the_taper_sums():
    """Each F statistic and amplitude is the complex least-squares fit of the
    tapers' coefficients on their sums, computed here one coefficient vector
    at a time with numpy.linalg.lstsq."""
    rng = np.random.default_rng(302)
    multitaper = Multitaper(
        rng.standard_normal((600, 3, 2)),
        _SAMPLING_FREQUENCY,
        time_window_duration=0.3,
        time_halfbandwidth_product=3,
    )
    result = multitaper.harmonic_f_test()
    coefficients = to_host(multitaper.fft())
    taper_sums = to_host(multitaper.tapers).sum(axis=0) / _SAMPLING_FREQUENCY
    n_tapers = taper_sums.size
    center = (multitaper.n_time_samples_per_window - 1) / (2 * _SAMPLING_FREQUENCY)
    for window, trial, frequency, signal in [(0, 0, 40, 0), (1, 2, 75, 1), (0, 1, 120, 1)]:
        observed = coefficients[window, trial, :, frequency, signal]
        (fit,), residual, *_ = np.linalg.lstsq(taper_sums[:, np.newaxis], observed, rcond=None)
        f_statistic = (n_tapers - 1) * abs(fit) ** 2 * np.sum(taper_sums**2) / residual[0]
        index = (window, trial, frequency, signal)
        np.testing.assert_allclose(result.f_statistic[index], f_statistic, rtol=1e-10)
        np.testing.assert_allclose(result.amplitude[index], 2 * abs(fit), rtol=1e-10)
        at_center = fit * np.exp(2j * np.pi * result.frequencies[frequency] * center)
        np.testing.assert_allclose(result.phase[index], np.angle(at_center), atol=1e-10)


def test_frequencies_within_the_half_bandwidth_of_the_edges_are_nan():
    """A line within W of 0 Hz or Nyquist overlaps its mirror image, so those
    frequencies are NaN in every field and all others are finite."""
    rng = np.random.default_rng(303)
    multitaper = Multitaper(
        rng.standard_normal((500, 2, 1)),
        _SAMPLING_FREQUENCY,
        time_halfbandwidth_product=5,
    )
    result = multitaper.harmonic_f_test()
    assert result.half_bandwidth == pytest.approx(10.0)  # NW = 5 over 0.5 s
    nyquist = _SAMPLING_FREQUENCY / 2
    edge = (result.frequencies < 10.0) | (result.frequencies > nyquist - 10.0)
    assert edge[0]
    assert edge[-1]
    assert not edge.all()
    for values in (result.f_statistic, result.p_value, result.amplitude, result.phase):
        assert np.isnan(values[:, :, edge]).all()
        assert np.isfinite(values[:, :, ~edge]).all()


def test_result_describes_its_axes_and_null_distribution():
    rng = np.random.default_rng(304)
    multitaper = Multitaper(
        rng.standard_normal((1000, 2, 3)),
        _SAMPLING_FREQUENCY,
        time_window_duration=0.5,
        time_halfbandwidth_product=3,
    )
    result = multitaper.harmonic_f_test()
    assert isinstance(result, HarmonicFTestResult)
    assert result.f_statistic.shape == (2, 2, multitaper.n_fft_samples // 2 + 1, 3)
    assert result.degrees_of_freedom == (2, 2 * multitaper.n_tapers - 2)
    np.testing.assert_array_equal(result.time, to_host(multitaper.time))
    np.testing.assert_array_equal(
        result.frequencies,
        np.fft.rfftfreq(multitaper.n_fft_samples, 1 / _SAMPLING_FREQUENCY),
    )


@pytest.mark.parametrize("taper_weighting", ["eigen", "adaptive"])
def test_uses_equally_weighted_tapers_whatever_the_weighting(taper_weighting):
    """The test is defined for the unweighted eigencoefficients, which the
    weighted fft() would change."""
    signals = np.random.default_rng(305).standard_normal((1000, 2, 2))
    uniform = Multitaper(signals, _SAMPLING_FREQUENCY, time_halfbandwidth_product=4)
    weighted = Multitaper(
        signals,
        _SAMPLING_FREQUENCY,
        time_halfbandwidth_product=4,
        taper_weighting=taper_weighting,
    )
    # Premise: the weighting changes the coefficients.
    assert not np.allclose(to_host(uniform.fft()), to_host(weighted.fft()))
    np.testing.assert_array_equal(
        weighted.harmonic_f_test().f_statistic, uniform.harmonic_f_test().f_statistic
    )


def test_rejects_custom_tapers():
    tapers = np.ones((100, 2)) / 10
    multitaper = Multitaper(np.zeros((100, 1, 1)), _SAMPLING_FREQUENCY, tapers=tapers)
    with pytest.raises(ValueError, match="needs DPSS tapers"):
        multitaper.harmonic_f_test()


def test_rejects_a_single_taper():
    multitaper = Multitaper(np.zeros((100, 1, 1)), _SAMPLING_FREQUENCY, n_tapers=1)
    with pytest.raises(ValueError, match="needs at least 2 tapers, got 1"):
        multitaper.harmonic_f_test()
