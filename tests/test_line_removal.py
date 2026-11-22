"""Line removal in Multitaper (line_frequencies, line_significance_threshold)."""

import json
import warnings

import numpy as np
import pytest

from spectral_connectivity import Connectivity, Multitaper, multitaper_connectivity
from spectral_connectivity.statistics import coherence_significance_pvalue
from tests._backend_helpers import to_host

_SAMPLING_FREQUENCY = 1000.0
_SETTINGS = {"time_window_duration": 1.0, "time_halfbandwidth_product": 4}


def _power(multitaper):
    return float(np.sum(np.abs(to_host(multitaper.fft())) ** 2))


def _lines(time, *lines):
    return sum(
        amplitude * np.cos(2 * np.pi * frequency * time + phase)
        for amplitude, frequency, phase in lines
    )


@pytest.mark.parametrize(
    ("lines", "window_settings"),
    [
        ([(1.0, 60.0, 0.3)], {}),
        ([(1.0, 60.0, 0.3), (0.5, 120.3, 1.1)], {"time_window_step": 0.25}),
        # Within the bandwidth of each other (W = 4 Hz): fitted jointly
        ([(1.0, 60.0, 0.3), (0.7, 61.0, 2.0)], {"start_time": 3.2}),
    ],
    ids=["one_line", "off_grid_overlapping_windows", "lines_1_hz_apart"],
)
@pytest.mark.parametrize("detrend_type", ["constant", "linear", None])
def test_removes_noise_free_lines(lines, window_settings, detrend_type):
    """Lines at their named frequencies leave essentially nothing, on or off
    the FFT grid, in overlapping windows, for lines closer than the bandwidth
    (fitted jointly), and whatever the detrending (the fit is of the line as
    detrending leaves it)."""
    time = np.arange(3000) / _SAMPLING_FREQUENCY
    signals = _lines(time, *lines)[:, np.newaxis, np.newaxis]
    settings = {**_SETTINGS, **window_settings, "detrend_type": detrend_type}
    raw = Multitaper(signals, _SAMPLING_FREQUENCY, **settings)
    cleaned = Multitaper(
        signals,
        _SAMPLING_FREQUENCY,
        line_frequencies=[frequency for _, frequency, _ in lines],
        **settings,
    )
    assert _power(cleaned) < 1e-12 * _power(raw)


def test_removing_where_there_is_no_line_takes_one_taper_share_of_the_noise():
    """Where there is no line the fit takes the noise's projection on the
    tapers' sums: for white noise the share of a trial's power at the named
    frequency (summed over tapers) that is removed is Beta(1, K - 1), of mean
    exactly 1 / K. Far from it the power is untouched. Trials are replicates."""
    n_trials = 400
    rng = np.random.default_rng(320)
    noise = rng.standard_normal((1000, n_trials, 1))
    raw = Multitaper(noise, _SAMPLING_FREQUENCY, **_SETTINGS)
    cleaned = Multitaper(noise, _SAMPLING_FREQUENCY, line_frequencies=[60.0], **_SETTINGS)

    def power_at(multitaper, frequency):
        index = np.argmin(np.abs(multitaper.frequencies - frequency))
        # (trial,): power summed over the tapers at that bin
        return np.sum(np.abs(to_host(multitaper.fft())[0, :, :, index, 0]) ** 2, axis=1)

    removed = 1 - power_at(cleaned, 60.0) / power_at(raw, 60.0)
    standard_error = removed.std(ddof=1) / np.sqrt(n_trials)
    assert abs(removed.mean() - 1 / raw.n_tapers) < 5 * standard_error
    # Far from it only the taper sidelobes of the subtracted sinusoid reach:
    # per trial a change below 0.1% at 150 Hz, and none on average.
    change = power_at(cleaned, 150.0) / power_at(raw, 150.0) - 1
    assert np.max(np.abs(change)) < 1e-2
    assert np.median(np.abs(change)) < 1e-3
    assert abs(change.mean()) < 5 * change.std(ddof=1) / np.sqrt(n_trials)


def test_a_threshold_removes_a_line_only_where_it_is_significant():
    """With line_significance_threshold, a window whose F-test at the named frequency is not
    significant keeps its coefficients exactly; one with the line is cleaned."""
    time = np.arange(2000) / _SAMPLING_FREQUENCY
    rng = np.random.default_rng(321)
    signals = rng.standard_normal((2000, 1, 1))
    signals[1000:, 0, 0] += _lines(time[1000:], (3.0, 60.0, 0.5))  # the second window only
    raw = Multitaper(signals, _SAMPLING_FREQUENCY, **_SETTINGS)
    cleaned = Multitaper(
        signals,
        _SAMPLING_FREQUENCY,
        line_frequencies=[60.0],
        line_significance_threshold=1e-3,
        **_SETTINGS,
    )
    raw_coefficients, cleaned_coefficients = to_host(raw.fft()), to_host(cleaned.fft())
    np.testing.assert_array_equal(cleaned_coefficients[0], raw_coefficients[0])
    at_60 = np.argmin(np.abs(raw.frequencies - 60.0))
    # What is left at 60 Hz is the window's noise, a few tenths of a percent
    # of the line's power there.
    assert np.sum(np.abs(cleaned_coefficients[1, ..., at_60, :]) ** 2) < 0.05 * np.sum(
        np.abs(raw_coefficients[1, ..., at_60, :]) ** 2
    )


def test_a_line_never_significant_warns():
    signals = np.random.default_rng(322).standard_normal((2000, 1, 1))
    multitaper = Multitaper(
        signals,
        _SAMPLING_FREQUENCY,
        line_frequencies=[60.0],
        line_significance_threshold=1e-6,
        **_SETTINGS,
    )
    with pytest.warns(UserWarning, match=r"line_frequencies \[60.0\] Hz: the line was not"):
        multitaper.fft()
    # Shown once per transform, however often the coefficients are recomputed.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        multitaper.fft()
        multitaper.harmonic_f_test()


def test_harmonic_f_test_of_a_cleaned_transform_sees_the_cleaned_data():
    time = np.arange(2000) / _SAMPLING_FREQUENCY
    signals = np.random.default_rng(323).standard_normal((2000, 2, 1))
    signals += _lines(time, (1.0, 60.25, 0.5))[:, np.newaxis, np.newaxis]
    raw = Multitaper(signals, _SAMPLING_FREQUENCY, **_SETTINGS).harmonic_f_test()
    cleaned = Multitaper(
        signals, _SAMPLING_FREQUENCY, line_frequencies=[60.25], **_SETTINGS
    ).harmonic_f_test()
    at_line = raw.frequencies == 60.25  # a bin of the padded grid, off the window's
    assert (raw.p_value[:, :, at_line, 0] < 1e-6).all()
    # Removed at exactly 60.25 Hz: nothing of a line is left there.
    assert (cleaned.amplitude[:, :, at_line, 0] < 1e-6).all()


def test_the_settings_are_recorded_and_reach_the_wrappers():
    time = np.arange(2000) / _SAMPLING_FREQUENCY
    signals = np.random.default_rng(324).standard_normal((2000, 2, 2))
    signals += _lines(time, (1.0, 60.0, 0.5))[:, np.newaxis, np.newaxis]
    multitaper = Multitaper(
        signals,
        _SAMPLING_FREQUENCY,
        line_frequencies=(60, 120),
        line_significance_threshold=0.01,
        **_SETTINGS,
    )
    assert multitaper.line_frequencies.tolist() == [60.0, 120.0]
    assert multitaper.line_significance_threshold == 0.01
    with pytest.raises(AttributeError, match="immutable"):
        multitaper.line_significance_threshold = 0.5
    assert "line_frequencies=[60.0, 120.0], line_significance_threshold=0.01" in repr(
        multitaper
    )
    assert "Lines removed:                 60, 120 Hz (where F-test p < 0.01)" in (
        multitaper.summarize_parameters()
    )

    missed_120 = r"\[120.0\] Hz: the line was not"
    with pytest.warns(UserWarning, match=missed_120):
        result = multitaper_connectivity(
            signals,
            sampling_frequency=_SAMPLING_FREQUENCY,
            method="power",
            transform_kwargs={
                "line_frequencies": [60, 120],
                "line_significance_threshold": 0.01,
            },
            **_SETTINGS,
        )
    with pytest.warns(UserWarning, match=missed_120):  # this Multitaper's first fit
        expected = Connectivity.from_transform(multitaper).power(one_sided=True)
    estimator = json.loads(result.attrs["transform_parameters_json"])["estimator"]
    assert estimator["line_frequencies"] == [60.0, 120.0]
    assert estimator["line_significance_threshold"] == 0.01
    np.testing.assert_allclose(result.values, to_host(expected), rtol=1e-12)


def test_without_lines_the_transform_is_unchanged():
    signals = np.random.default_rng(325).standard_normal((1000, 2, 2))
    plain = Multitaper(signals, _SAMPLING_FREQUENCY, **_SETTINGS)
    empty = Multitaper(signals, _SAMPLING_FREQUENCY, line_frequencies=[], **_SETTINGS)
    np.testing.assert_array_equal(to_host(empty.fft()), to_host(plain.fft()))
    assert plain.line_frequencies.size == 0
    assert plain.line_significance_threshold is None
    assert "line_frequencies" not in repr(plain)


@pytest.mark.parametrize(
    ("settings", "match"),
    [
        ({"line_frequencies": [2.0]}, r"half-bandwidth W = 4 Hz of 0 Hz.*lower the first"),
        ({"line_frequencies": [498.0]}, "within the tapers' half-bandwidth"),
        ({"line_frequencies": [np.nan]}, "1-D sequence of finite numbers"),
        ({"line_frequencies": [True]}, "1-D sequence of finite numbers"),
        ({"line_frequencies": ["60"]}, "1-D sequence of finite numbers"),
        ({"line_frequencies": [[60.0, 120.0]]}, "1-D sequence of finite numbers"),
        ({"line_significance_threshold": 0.01}, "without line_frequencies"),
        (
            {"line_frequencies": [60.0, 62.0], "line_significance_threshold": 0.01},
            r"\[\(60.0, 62.0\)\] are closer than the tapers' half-bandwidth W = 4 Hz",
        ),
        ({"line_frequencies": [60.0, 60.0]}, "more than once"),
        (
            {"line_frequencies": [60.0], "line_significance_threshold": 0.0},
            "strictly between 0 and 1",
        ),
        (
            {"line_frequencies": [60.0], "line_significance_threshold": True},
            "strictly between 0 and 1",
        ),
        ({"line_frequencies": [60.0], "tapers": np.ones((1000, 2))}, "needs DPSS tapers"),
        ({"line_frequencies": [60.0], "n_tapers": 1}, "needs at least 2 tapers"),
        ({"line_frequencies": [60.0, 60.0 + 1e-9]}, "too close to fit separately"),
    ],
    ids=str,
)
def test_rejects_invalid_line_settings(settings, match):
    settings = {**_SETTINGS, **settings}
    with pytest.raises(ValueError, match=match):
        Multitaper(np.zeros((1000, 1, 1)), _SAMPLING_FREQUENCY, **settings)


@pytest.mark.parametrize("detrend_type", ["constant", "linear", "c", "l"])
def test_a_low_frequency_line_is_removed_whatever_the_detrending(detrend_type):
    """A line near the bandwidth from 0 Hz has a large window mean and trend,
    which detrending removes from the data; fitting the plain sinusoid would
    leave about 0.1% (constant) to 0.8% (linear) of a 4.3 Hz line. The
    short spellings of the detrend types fit the same way."""
    time = np.arange(1000) / _SAMPLING_FREQUENCY
    signals = _lines(time, (1.0, 4.3, 1.1))[:, np.newaxis, np.newaxis]
    settings = {"time_halfbandwidth_product": 4, "detrend_type": detrend_type}
    raw = Multitaper(signals, _SAMPLING_FREQUENCY, **settings)
    cleaned = Multitaper(signals, _SAMPLING_FREQUENCY, line_frequencies=[4.3], **settings)
    assert _power(cleaned) < 1e-9 * _power(raw)


def test_two_strong_lines_w_apart_are_both_removed_with_a_threshold():
    """Each line's F-test is of its own regression with the other's fitted
    contribution removed, so neither hides the other."""
    time = np.arange(1000) / _SAMPLING_FREQUENCY
    rng = np.random.default_rng(326)
    signals = rng.standard_normal((1000, 50, 1))
    signals += _lines(time, (3.0, 60.0, 0.4), (3.0, 64.0, 2.0))[:, np.newaxis, np.newaxis]
    noise_only = Multitaper(
        signals - _lines(time, (3.0, 60.0, 0.4), (3.0, 64.0, 2.0))[:, np.newaxis, np.newaxis],
        _SAMPLING_FREQUENCY,
        time_halfbandwidth_product=4,
        line_frequencies=[60.0, 64.0],
    )
    cleaned = Multitaper(
        signals,
        _SAMPLING_FREQUENCY,
        time_halfbandwidth_product=4,
        line_frequencies=[60.0, 64.0],
        line_significance_threshold=1e-3,
    )
    # Removed in every trial: what is left is the noise, cleaned the same way.
    np.testing.assert_allclose(
        to_host(cleaned.fft()), to_host(noise_only.fft()), rtol=0, atol=1e-10
    )


def test_an_absent_line_w_from_a_strong_one_is_removed_at_the_nominal_rate():
    """Without a line at 64 Hz, its F-test (after the 60 Hz line's fit is taken
    out) is significant in a fraction ``line_significance_threshold`` of the
    trials, within five binomial standard errors. A trial where it is not
    keeps exactly the coefficients of removing the 60 Hz line alone."""
    n_trials, threshold = 3000, 0.05
    time = np.arange(1000) / _SAMPLING_FREQUENCY
    rng = np.random.default_rng(327)
    signals = rng.standard_normal((1000, n_trials, 1))
    signals += _lines(time, (3.0, 60.0, 0.4))[:, np.newaxis, np.newaxis]
    settings = {"time_halfbandwidth_product": 4, "line_significance_threshold": threshold}
    both = to_host(
        Multitaper(
            signals, _SAMPLING_FREQUENCY, line_frequencies=[60.0, 64.0], **settings
        ).fft()
    )
    alone = to_host(
        Multitaper(signals, _SAMPLING_FREQUENCY, line_frequencies=[60.0], **settings).fft()
    )
    removed_64 = ~np.all(np.isclose(both, alone, rtol=0, atol=1e-12), axis=(0, 2, 3, 4))
    standard_error = np.sqrt(threshold * (1 - threshold) / n_trials)
    assert abs(removed_64.mean() - threshold) < 5 * standard_error, removed_64.mean()


def test_a_threshold_works_per_trial_and_signal():
    """In overlapping windows with an offset start, a line in one trial and
    signal only is removed there, and the others keep their coefficients."""
    time = 2.5 + np.arange(2000) / _SAMPLING_FREQUENCY
    rng = np.random.default_rng(328)
    signals = rng.standard_normal((2000, 3, 2))
    signals[:, 1, 0] += _lines(time, (3.0, 120.0, 1.0))
    settings = {
        **_SETTINGS,
        "time_window_step": 0.5,
        "start_time": 2.5,
        "line_frequencies": [120.0],
        "line_significance_threshold": 1e-4,
    }
    plain = {key: value for key, value in settings.items() if not key.startswith("line_")}
    raw = to_host(Multitaper(signals, _SAMPLING_FREQUENCY, **plain).fft())
    cleaned = to_host(Multitaper(signals, _SAMPLING_FREQUENCY, **settings).fft())
    without_line = np.ones((3, 2), dtype=bool)
    without_line[1, 0] = False
    np.testing.assert_array_equal(
        cleaned[:, without_line.nonzero()[0], ..., without_line.nonzero()[1]],
        raw[:, without_line.nonzero()[0], ..., without_line.nonzero()[1]],
    )
    at_120 = np.argmin(
        np.abs(Multitaper(signals, _SAMPLING_FREQUENCY, **plain).frequencies - 120)
    )
    assert np.sum(np.abs(cleaned[:, 1, :, at_120, 0]) ** 2) < 0.01 * np.sum(
        np.abs(raw[:, 1, :, at_120, 0]) ** 2
    )


def test_a_single_frequency_and_float32_input():
    time = np.arange(1000) / _SAMPLING_FREQUENCY
    signals = np.random.default_rng(329).standard_normal((1000, 2, 1))
    signals += _lines(time, (1.0, 60.0, 0.3))[:, np.newaxis, np.newaxis]
    scalar = Multitaper(signals, _SAMPLING_FREQUENCY, line_frequencies=60, **_SETTINGS)
    listed = Multitaper(signals, _SAMPLING_FREQUENCY, line_frequencies=[60.0], **_SETTINGS)
    np.testing.assert_array_equal(to_host(scalar.fft()), to_host(listed.fft()))
    single = Multitaper(
        signals.astype(np.float32), _SAMPLING_FREQUENCY, line_frequencies=60, **_SETTINGS
    )
    np.testing.assert_allclose(to_host(single.fft()), to_host(listed.fft()), rtol=0, atol=1e-6)
    with pytest.raises(ValueError, match="read-only"):
        listed.line_frequencies[0] = 50.0


def test_coherence_tests_count_one_taper_fewer_per_trial_near_a_removed_line():
    """Removal takes one of the K tapers' degrees of freedom per trial within W
    of the line, so the zero-coherence test is liberal there with
    n_observations (one 5-trial x 7-taper dataset in 13 was significant at
    0.05 in simulation) and calibrated with n_observations - n_trials, as the
    docs advise. Replicate datasets are time windows."""
    n_windows, n_trials, level = 3000, 5, 0.05
    rng = np.random.default_rng(330)
    signals = rng.standard_normal((1000 * n_windows, n_trials, 2))
    multitaper = Multitaper(signals, _SAMPLING_FREQUENCY, line_frequencies=[60.0], **_SETTINGS)
    connectivity = Connectivity.from_transform(multitaper)
    near = np.abs(connectivity.frequencies - 60.0) < 3.5  # within W = 4 Hz
    coherency = connectivity.coherency()[:, near, 0, 1]
    standard_error = np.sqrt(level * (1 - level) / coherency.size)
    counted = coherence_significance_pvalue(coherency, connectivity.n_observations)
    corrected = coherence_significance_pvalue(
        coherency, connectivity.n_observations - n_trials
    )
    assert np.mean(counted < level) - level > 5 * standard_error
    assert abs(np.mean(corrected < level) - level) < 5 * standard_error
