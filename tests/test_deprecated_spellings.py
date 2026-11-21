"""A 2.x-style sequence of calls that uses every deprecated spelling.

Each deprecation also has its own tests; this one checks that, used together,
every deprecated spelling warns exactly once, at the line that used it, and
that no deprecated path triggers another's warning.
"""

import warnings

import numpy as np

from spectral_connectivity import (
    Connectivity,
    Multitaper,
    multitaper_connectivity,
    suggest_parameters,
)
from spectral_connectivity.statistics import power_confidence_intervals
from spectral_connectivity.wrapper import connectivity_to_xarray, fourier_connectivity

_BAND = (10.0, 50.0)


def test_every_deprecated_spelling_warns_once_at_its_own_line():
    x = np.random.default_rng(0).standard_normal((1000, 4, 3))
    m = Multitaper(x, 500, time_window_duration=0.5)
    c = Connectivity.from_transform(m)
    # Each step is (call, the message fragments its warnings contain, in order).
    steps = [
        (lambda: Connectivity.from_multitaper(m), ["from_multitaper()"]),
        (lambda: c.coherence_magnitude(), ["Connectivity.coherence_magnitude()"]),
        (lambda: c.jackknife("coherence_magnitude"), ["jackknife('coherence_magnitude')"]),
        (lambda: c.delay(frequencies_of_interest=_BAND), ["delay(frequencies_of_interest"]),
        (lambda: c.group_delay(frequencies_of_interest=_BAND), ["group_delay(frequencies_of"]),
        (lambda: c.phase_slope_index(_BAND), ["phase_slope_index(frequencies_of_interest"]),
        (
            lambda: multitaper_connectivity(
                x, 500, 0.5, method="coherence_magnitude", n_fft_samples=512
            ),
            ["method='coherence_magnitude'", "multitaper_connectivity(n_fft_samples=...)"],
        ),
        (
            lambda: connectivity_to_xarray(m, "coherence_magnitude"),
            ["method='coherence_magnitude'"],
        ),
        (
            lambda: connectivity_to_xarray(
                m, "subset_pairwise_spectral_granger_prediction", pairs=[(0, 1)]
            ),
            ["connectivity_to_xarray(pairs=...)"],
        ),
        (
            lambda: connectivity_to_xarray(
                m, "delay", connectivity_kwargs={"frequencies_of_interest": _BAND}
            ),
            ["delay(frequencies_of_interest"],
        ),
        (
            lambda: connectivity_to_xarray(
                m, "phase_slope_index", frequencies_of_interest=_BAND
            ),
            ["phase_slope_index(frequencies_of_interest"],
        ),
        (
            lambda: fourier_connectivity(m.fft(), m.frequencies, method="coherence_magnitude"),
            ["method='coherence_magnitude'"],
        ),
        (
            lambda: power_confidence_intervals(n_tapers=25, power=2.0),
            ["power_confidence_intervals(n_tapers=...)"],
        ),
        (
            lambda: power_confidence_intervals(25, 2.0, ci=0.9),
            ["power_confidence_intervals(ci=...)"],
        ),
        (
            lambda: suggest_parameters(500, 10.0, desired_freq_resolution=2.0),
            ["suggest_parameters(desired_freq_resolution=...)"],
        ),
    ]
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        # The directed-orientation reminder is a UserWarning, not a deprecation.
        warnings.simplefilter("ignore", UserWarning)
        for call, _ in steps:
            call()
    received = [
        (w.filename, w.lineno, str(w.message))
        for w in record
        if issubclass(w.category, DeprecationWarning)
    ]
    expected = [
        (call.__code__.co_filename, call.__code__.co_firstlineno, fragment)
        for call, fragments in steps
        for fragment in fragments
    ]
    assert len(received) == len(expected), [message for *_, message in received]
    for (filename, line, message), (want_file, want_line, fragment) in zip(
        received, expected, strict=True
    ):
        assert (filename, line) == (want_file, want_line), message
        assert fragment in message
        assert "removed in 4.0" in message
