"""The temporary warning that ``Connectivity.power`` is one-sided since 3.0.

``pyproject.toml`` ignores warnings that start with "Since spectral_connectivity
3.0," for the rest of the suite, so each test here records warnings with the
``"always"`` filter.
"""

import json
import os
import subprocess
import sys
import warnings

import numpy as np
import pytest

import spectral_connectivity
from spectral_connectivity import (
    Connectivity,
    MorletWavelet,
    Multitaper,
    PowerSidednessWarning,
    fourier_connectivity,
    multitaper_connectivity,
)
from spectral_connectivity.wrapper import connectivity_to_xarray


@pytest.fixture(scope="module")
def time_series():
    return np.random.default_rng(0).standard_normal((256, 4, 2))


@pytest.fixture(scope="module")
def connectivity(time_series):
    return Connectivity.from_transform(Multitaper(time_series, sampling_frequency=100))


def _sidedness_warnings(record):
    return [w for w in record if issubclass(w.category, PowerSidednessWarning)]


def test_warning_is_a_public_user_warning():
    assert issubclass(PowerSidednessWarning, UserWarning)
    assert "PowerSidednessWarning" in spectral_connectivity.__all__


def test_power_without_one_sided_warns_once_at_the_call(connectivity):
    """A 2.x call, ``power()``, gets values twice as large at interior bins
    as in 2.x, so it says so, at the caller's line, and how to choose."""
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        power = connectivity.power()
    sidedness = _sidedness_warnings(record)
    assert len(sidedness) == 1
    assert sidedness[0].filename == __file__
    message = str(sidedness[0].message)
    assert message.startswith("Since spectral_connectivity 3.0,")
    assert "one_sided=True" in message
    assert "one_sided=False" in message
    # It says it will not repeat, and how to see it again or silence it.
    assert (
        "Shown once per Python session; to see it at every call: "
        'warnings.simplefilter("always", spectral_connectivity.PowerSidednessWarning).'
    ) in message
    assert "Silence this warning with" in message
    np.testing.assert_array_equal(power, connectivity.power(one_sided=True))


@pytest.mark.parametrize("one_sided", [True, False])
def test_power_with_one_sided_is_silent(connectivity, one_sided):
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        connectivity.power(one_sided=one_sided)
    assert _sidedness_warnings(record) == []


def test_power_of_one_sided_coefficients_is_silent(time_series):
    """One-sided coefficients (a Morlet transform, which 2.x did not have, or
    coefficients declared one-sided) are returned as given, as before."""
    morlet = Connectivity.from_transform(
        MorletWavelet(time_series, sampling_frequency=100, frequencies=np.array([10.0, 20.0]))
    )
    # (frequency, trial, signal) -> (time window, trial, taper, frequency, signal)
    half_spectrum = np.moveaxis(np.fft.rfft(time_series, axis=0), 0, 1)
    declared = Connectivity(half_spectrum[np.newaxis, :, np.newaxis], is_one_sided=True)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        morlet.power()
        declared.power()
    assert _sidedness_warnings(record) == []


def test_labelled_power_is_silent(time_series):
    """The wrappers' power results are one-sided too and do not warn, a
    documented exception; their attributes state the sidedness."""
    multitaper = Multitaper(time_series, sampling_frequency=100)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        results = [
            multitaper_connectivity(time_series, sampling_frequency=100, method="power"),
            multitaper_connectivity(time_series, sampling_frequency=100),
            connectivity_to_xarray(multitaper, "power"),
            fourier_connectivity(
                multitaper.fft(), multitaper.frequencies, method="power", is_one_sided=False
            ),
        ]
    assert _sidedness_warnings(record) == []
    assert all(
        "One-sided" in result["power"].attrs["interpretation"]
        if "power" in getattr(result, "data_vars", {})
        else "One-sided" in result.attrs["interpretation"]
        for result in results
    )


def test_jackknife_of_power_warns_once_for_its_full_estimate(connectivity):
    """The jackknife's estimate is power(), so it carries the same notice,
    once, at the jackknife call: its leave-one-out replicates do not repeat
    it. A ``one_sided`` passed through to ``power`` is honored."""
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        connectivity.jackknife("power")
    (warning,) = _sidedness_warnings(record)
    assert warning.filename == __file__
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        connectivity.jackknife("power", one_sided=True)
    assert _sidedness_warnings(record) == []


def test_warning_is_silenced_by_its_category(connectivity):
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", category=PowerSidednessWarning)
        connectivity.power()
    assert _sidedness_warnings(record) == []


_SESSION_SCRIPT = """
import json
import warnings

import numpy as np

shown = []
# Record what Python's filters let through, without changing the filters.
warnings.showwarning = lambda message, category, *args, **kwargs: shown.append(
    category.__name__
)
from spectral_connectivity import Connectivity, Multitaper, PowerSidednessWarning

time_series = np.random.default_rng(0).standard_normal((64, 3, 2))
connectivity = Connectivity.from_transform(Multitaper(time_series, sampling_frequency=100))
counts = []
connectivity.power()
counts.append(len(shown))
with warnings.catch_warnings():  # resets what Python's "once" action showed
    pass
connectivity.power()  # another line
connectivity.jackknife("power")
counts.append(len(shown))
warnings.simplefilter("always", PowerSidednessWarning)
connectivity.power()
connectivity.power()
counts.append(len(shown))
warnings.simplefilter("error", PowerSidednessWarning)
try:
    connectivity.power()
except PowerSidednessWarning:
    counts.append("raised")
print(json.dumps(counts))
"""


def test_warning_appears_once_per_session_unless_a_filter_says_otherwise():
    """Under Python's default filters the notice appears once per session,
    not at every line that calls ``power()``; a filter the user sets after
    import still shows it every time, or turns it into an error."""
    environment = {k: v for k, v in os.environ.items() if k != "PYTHONWARNINGS"}
    output = subprocess.check_output(
        [sys.executable, "-c", _SESSION_SCRIPT], text=True, env=environment
    )
    assert json.loads(output) == [1, 1, 3, "raised"]


_SILENCED_FIRST_SCRIPT = """
import json
import warnings

import numpy as np

shown = []
warnings.showwarning = lambda message, category, *args, **kwargs: shown.append(
    category.__name__
)
from spectral_connectivity import Connectivity, Multitaper, PowerSidednessWarning

time_series = np.random.default_rng(0).standard_normal((64, 3, 2))
connectivity = Connectivity.from_transform(Multitaper(time_series, sampling_frequency=100))
counts = []
# Silenced, then un-silenced: the notice was never seen, so it is still due.
warnings.filterwarnings("ignore", category=PowerSidednessWarning)
connectivity.power()
warnings.resetwarnings()
connectivity.power()
connectivity.power()  # another line
counts.append(len(shown))
print(json.dumps(counts))
"""

_ERROR_FIRST_SCRIPT = """
import json
import warnings

import numpy as np

shown = []
warnings.showwarning = lambda message, category, *args, **kwargs: shown.append(
    category.__name__
)
from spectral_connectivity import Connectivity, Multitaper, PowerSidednessWarning

time_series = np.random.default_rng(0).standard_normal((64, 3, 2))
connectivity = Connectivity.from_transform(Multitaper(time_series, sampling_frequency=100))
counts = []
# Raised as an error: the user read the message in the exception, so it counts.
warnings.filterwarnings("error", category=PowerSidednessWarning)
try:
    connectivity.power()
except PowerSidednessWarning:
    counts.append("raised")
warnings.resetwarnings()
connectivity.power()
counts.append(len(shown))
print(json.dumps(counts))
"""


@pytest.mark.parametrize(
    ("script", "expected"),
    [(_SILENCED_FIRST_SCRIPT, [1]), (_ERROR_FIRST_SCRIPT, ["raised", 0])],
    ids=["ignored-first-call-does-not-count", "raised-first-call-counts"],
)
def test_only_a_notice_the_user_saw_uses_up_the_session(script, expected):
    """An ignored call does not count as shown, so the notice appears once
    the filter is removed (a notebook that silenced it, then un-silenced it).
    A call raised as an error does: its message was in the exception."""
    environment = {k: v for k, v in os.environ.items() if k != "PYTHONWARNINGS"}
    output = subprocess.check_output(
        [sys.executable, "-c", script], text=True, env=environment
    )
    assert json.loads(output) == expected


_CONTEXT_SCRIPT = """
import warnings

import numpy as np

from spectral_connectivity import Connectivity, Multitaper, PowerSidednessWarning

time_series = np.random.default_rng(0).standard_normal((64, 3, 2))
connectivity = Connectivity.from_transform(Multitaper(time_series, sampling_frequency=100))
counts = []
for _ in range(2):
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        connectivity.power()
    counts.append(sum(issubclass(w.category, PowerSidednessWarning) for w in record))
print(counts)
"""


@pytest.mark.skipif(
    sys.version_info < (3, 14), reason="context-aware warnings are new in Python 3.14"
)
def test_a_repeat_obeys_a_filter_set_in_catch_warnings_under_context_aware_warnings():
    """Python 3.14's context-aware warnings (the default on free-threaded
    builds) keep the filters a ``catch_warnings`` block sets, as
    ``pytest.warns`` does, out of ``warnings.filters``."""
    environment = {k: v for k, v in os.environ.items() if k != "PYTHONWARNINGS"}
    output = subprocess.check_output(
        [sys.executable, "-X", "context_aware_warnings=1", "-c", _CONTEXT_SCRIPT],
        text=True,
        env=environment,
    )
    assert output.strip() == "[1, 1]"
