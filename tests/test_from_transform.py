"""``Connectivity.from_transform`` and its 2.x name ``from_multitaper``."""

import inspect
import warnings

import numpy as np
import pytest

from spectral_connectivity import Connectivity, Multitaper, multitaper_connectivity
from spectral_connectivity.wrapper import connectivity_to_xarray


@pytest.fixture(scope="module")
def multitaper():
    time_series = np.random.default_rng(0).standard_normal((512, 6, 3))
    return Multitaper(time_series, sampling_frequency=256, time_window_duration=0.5)


def test_package_internals_construct_through_from_transform(monkeypatch, multitaper):
    """Normal use never reaches the 2.x constructor name."""

    def fail(*args, **kwargs):
        msg = "package code called Connectivity.from_multitaper"
        raise AssertionError(msg)

    monkeypatch.setattr(Connectivity, "from_multitaper", classmethod(fail))
    multitaper_connectivity(
        multitaper.time_series, sampling_frequency=256, method="magnitude_squared_coherence"
    )
    connectivity_to_xarray(multitaper, "magnitude_squared_coherence")
    Connectivity.from_transform(multitaper).magnitude_squared_coherence()


def _deprecations(record):
    return [
        (w.filename, w.lineno, str(w.message))
        for w in record
        if issubclass(w.category, DeprecationWarning)
    ]


def _assert_same_connectivity(actual, expected):
    assert type(actual) is type(expected)
    np.testing.assert_array_equal(actual.fourier_coefficients, expected.fourier_coefficients)
    np.testing.assert_array_equal(actual.frequencies, expected.frequencies)
    np.testing.assert_array_equal(actual.time, expected.time)
    assert actual.expectation_type == expected.expectation_type
    np.testing.assert_array_equal(
        actual.magnitude_squared_coherence(), expected.magnitude_squared_coherence()
    )


def test_from_multitaper_warns_once_at_the_caller_and_agrees_exactly(multitaper):
    call = lambda: Connectivity.from_multitaper(multitaper)  # noqa: E731
    with pytest.warns(DeprecationWarning, match="from_multitaper") as record:
        deprecated = call()
    ((filename, lineno, message),) = _deprecations(record)
    assert (filename, lineno) == (__file__, call.__code__.co_firstlineno)
    assert "Connectivity.from_transform" in message
    assert "removed in 4.0" in message
    _assert_same_connectivity(deprecated, Connectivity.from_transform(multitaper))


def test_from_multitaper_keeps_its_arguments(multitaper):
    """The released keyword name and the positional settings still bind."""
    with pytest.warns(DeprecationWarning, match="from_multitaper"):
        by_keyword = Connectivity.from_multitaper(
            multitaper_instance=multitaper, expectation_type="tapers"
        )
    with pytest.warns(DeprecationWarning, match="from_multitaper"):
        by_position = Connectivity.from_multitaper(multitaper, "tapers", np.complex64)
    _assert_same_connectivity(
        by_keyword, Connectivity.from_transform(multitaper, expectation_type="tapers")
    )
    _assert_same_connectivity(
        by_position, Connectivity.from_transform(multitaper, "tapers", np.complex64)
    )


class _PlainSubclass(Connectivity):
    pass


class _TransformOverride(Connectivity):
    """A 3.x subclass that customizes the canonical constructor."""

    @classmethod
    def from_transform(cls, transform, **kwargs):
        instance = super().from_transform(transform, **kwargs)
        instance.marker = "from_transform override"
        return instance


class _LegacyOverride(Connectivity):
    """A 2.x subclass that customizes the old constructor through super()."""

    @classmethod
    def from_multitaper(cls, multitaper_instance, **kwargs):
        instance = super().from_multitaper(multitaper_instance, **kwargs)  # 2.x call
        instance.marker = "from_multitaper override"
        return instance


def test_from_multitaper_on_a_subclass_builds_the_subclass(multitaper):
    call = lambda: _PlainSubclass.from_multitaper(multitaper)  # noqa: E731
    with pytest.warns(DeprecationWarning, match="from_multitaper") as record:
        deprecated = call()
    ((filename, lineno, message),) = _deprecations(record)
    assert (filename, lineno) == (__file__, call.__code__.co_firstlineno)
    assert "_PlainSubclass.from_multitaper()" in message
    _assert_same_connectivity(deprecated, _PlainSubclass.from_transform(multitaper))


def test_from_multitaper_reaches_a_subclass_from_transform_override(multitaper):
    call = lambda: _TransformOverride.from_multitaper(multitaper)  # noqa: E731
    with pytest.warns(DeprecationWarning, match="from_multitaper") as record:
        deprecated = call()
    ((filename, lineno, _),) = _deprecations(record)
    assert (filename, lineno) == (__file__, call.__code__.co_firstlineno)
    assert deprecated.marker == "from_transform override"


def test_a_subclass_overriding_from_multitaper_warns_once_at_its_super_call(multitaper):
    """The override is user code: its own call to the alias is the line warned
    about. from_transform does not go through the override and does not warn."""
    with pytest.warns(DeprecationWarning, match="from_multitaper") as record:
        deprecated = _LegacyOverride.from_multitaper(multitaper)
    ((filename, lineno, _),) = _deprecations(record)
    source, first = inspect.getsourcelines(_LegacyOverride.from_multitaper)
    super_line = first + next(i for i, line in enumerate(source) if "# 2.x call" in line)
    assert (filename, lineno) == (__file__, super_line)
    assert deprecated.marker == "from_multitaper override"

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        canonical = _LegacyOverride.from_transform(multitaper)
    assert not hasattr(canonical, "marker")
    _assert_same_connectivity(deprecated, canonical)
