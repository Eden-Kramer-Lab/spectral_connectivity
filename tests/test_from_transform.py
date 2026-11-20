"""``Connectivity.from_transform`` and its 2.x name ``from_multitaper``."""

import inspect
import warnings

import numpy as np
import pytest

from spectral_connectivity import Connectivity, Multitaper, multitaper_connectivity
from spectral_connectivity.wrapper import connectivity_to_xarray
from tests._deprecation_helpers import warns_once_at_the_call


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
    deprecated, _ = warns_once_at_the_call(
        lambda: Connectivity.from_multitaper(multitaper), "Connectivity.from_transform"
    )
    _assert_same_connectivity(deprecated, Connectivity.from_transform(multitaper))


def test_from_multitaper_keeps_its_arguments(multitaper):
    """The released keyword name and the 3.0 positional order (expectation
    type, then dtype) bind as in from_transform. The released order, with
    blocks third, is rejected; see the blocks tests below."""
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
    deprecated, _ = warns_once_at_the_call(
        lambda: _PlainSubclass.from_multitaper(multitaper), "_PlainSubclass.from_multitaper()"
    )
    _assert_same_connectivity(deprecated, _PlainSubclass.from_transform(multitaper))


def test_from_multitaper_does_not_call_a_subclass_from_transform_override(multitaper):
    """The deprecated name builds the instance itself; only code that calls
    from_transform reaches an override of it."""
    deprecated, _ = warns_once_at_the_call(
        lambda: _TransformOverride.from_multitaper(multitaper), "from_multitaper"
    )
    assert not hasattr(deprecated, "marker")
    assert _TransformOverride.from_transform(multitaper).marker == "from_transform override"


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


_BLOCKS_MESSAGE = r"blocks.*removed in 3\.0.*no effect on results.*drop it"


@pytest.mark.parametrize("blocks", [None, 1, 5], ids=repr)
def test_blocks_by_keyword_is_rejected_with_its_own_message(multitaper, blocks):
    """2.x's blocks (removed in 3.0) never changed a value; every keyword use
    of it fails with a message that says so, before anything is computed."""
    with pytest.raises(TypeError, match=_BLOCKS_MESSAGE):
        Connectivity(multitaper.fft(), blocks=blocks)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(TypeError, match=_BLOCKS_MESSAGE):
            Connectivity.from_multitaper(multitaper, blocks=blocks)
    with pytest.raises(TypeError, match=_BLOCKS_MESSAGE):
        Connectivity.from_transform(multitaper, blocks=blocks)


@pytest.mark.parametrize("blocks", [None, 1, 5], ids=repr)
def test_blocks_in_its_released_position_is_rejected_with_its_own_message(multitaper, blocks):
    """In 2.x blocks was from_multitaper's third and Connectivity's fifth
    positional argument, where dtype now is. None used to become float64 and
    silently drop every coefficient's imaginary part, and an int failed only
    when a measure first used it."""
    coefficients, frequencies, time = (
        multitaper.fft(),
        multitaper.frequencies,
        multitaper.time,
    )
    released_position = rf"complex dtype.*fifth in Connectivity.*{_BLOCKS_MESSAGE}"
    with pytest.raises(TypeError, match=released_position):
        Connectivity(coefficients, "trials_tapers", frequencies, time, blocks)
    with pytest.raises(TypeError, match=released_position):
        # 2.x's full positional form: blocks, then dtype.
        Connectivity(coefficients, "trials_tapers", frequencies, time, blocks, np.complex128)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(
            TypeError, match=rf"complex dtype.*third in from_multitaper.*{_BLOCKS_MESSAGE}"
        ):
            Connectivity.from_multitaper(multitaper, "trials_tapers", blocks)


@pytest.mark.parametrize(
    ("value", "looks_like_blocks"),
    [(None, True), (5, True), (np.float64, False), ("float32", False)],
    ids=repr,
)
def test_a_dtype_that_is_not_complex_is_rejected(multitaper, value, looks_like_blocks):
    """A real dtype would discard every coefficient's imaginary part; a None
    or an integer, 2.x's blocks values, also gets the note about blocks."""
    for call in (
        lambda: Connectivity.from_transform(multitaper, "trials_tapers", value),
        lambda: Connectivity(multitaper.fft(), dtype=value),
    ):
        with pytest.raises(TypeError, match="complex dtype") as excinfo:
            call()
        assert ("blocks" in str(excinfo.value)) is looks_like_blocks


@pytest.mark.parametrize(
    "value", [np.complex64, "complex128", np.dtype("complex64")], ids=repr
)
def test_a_complex_dtype_is_accepted(multitaper, value):
    connectivity = Connectivity.from_transform(multitaper, dtype=value)
    assert connectivity.magnitude_squared_coherence().shape[-2:] == (3, 3)


class _HalfMigrated(Connectivity):
    """A 2.x override renamed to from_transform, its super() call not yet."""

    @classmethod
    def from_transform(cls, transform, **kwargs):
        instance = super().from_multitaper(transform, **kwargs)
        instance.marker = "half migrated"
        return instance


def test_a_half_migrated_subclass_works_and_warns_once(multitaper):
    """An override renamed to from_transform that still calls
    super().from_multitaper builds its instance, applies its customization
    once, and warns once about the old name: neither name calls the other."""
    with pytest.warns(DeprecationWarning, match="from_multitaper") as record:
        instance = _HalfMigrated.from_transform(multitaper)
    assert len(record) == 1
    assert isinstance(instance, _HalfMigrated)
    assert instance.marker == "half migrated"
    np.testing.assert_array_equal(
        instance.magnitude_squared_coherence(),
        Connectivity.from_transform(multitaper).magnitude_squared_coherence(),
    )


def test_the_from_multitaper_warning_names_the_renamed_first_argument(multitaper):
    _, message = warns_once_at_the_call(
        lambda: Connectivity.from_multitaper(multitaper_instance=multitaper),
        "the first is named transform, not multitaper_instance",
    )
    assert message.index("multitaper_instance") > message.index("from_transform()")


def test_a_rejected_from_multitaper_call_does_not_warn(multitaper):
    """The deprecation is raised once the instance is built, so a call that
    is rejected shows only its error, as the wrappers do."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        with pytest.raises(ValueError, match="expectation_type"):
            Connectivity.from_multitaper(multitaper, expectation_type="bogus")
