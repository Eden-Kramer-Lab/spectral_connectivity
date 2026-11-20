"""``magnitude_squared_coherence`` and its deprecated 2.x name ``coherence_magnitude``.

The suite turns every warning into an error, so each call here that uses the
canonical name, or the defaults, also checks that it does not warn.
"""

import inspect
import json
import re

import numpy as np
import pytest
import xarray as xr

from spectral_connectivity import (
    DEFAULT_METHODS,
    Connectivity,
    Multitaper,
    fourier_connectivity,
    list_measures,
    multitaper_connectivity,
)
from spectral_connectivity.wrapper import connectivity_to_xarray

_OLD = "coherence_magnitude"
_NEW = "magnitude_squared_coherence"


@pytest.fixture(scope="module")
def time_series():
    """Signal 1 is signal 0 delayed by 3 samples plus noise, shape (1003, 20, 3)."""
    rng = np.random.default_rng(0)
    leader = rng.standard_normal((1003, 20))
    signals = np.stack([leader[3:], leader[:-3], rng.standard_normal((1000, 20))], axis=-1)
    return signals + 0.5 * rng.standard_normal(signals.shape)


@pytest.fixture(scope="module")
def multitaper(time_series):
    return Multitaper(time_series, sampling_frequency=500, time_window_duration=0.5)


@pytest.fixture
def connectivity(multitaper):
    return Connectivity.from_transform(multitaper)


# How each route writes a measure name in its warning.
_METHOD, _STRING, _JACKKNIFE = "Connectivity.{}()", "method={!r}", "jackknife({!r})"


def _warns(spelling):
    """Expect the deprecation of the old name as ``spelling`` writes it."""
    return pytest.warns(DeprecationWarning, match=re.escape(spelling.format(_OLD)))


def _assert_one_caller_warning(record, spelling):
    """Exactly one DeprecationWarning, attributed to this file, naming the old
    and new spellings and the removal in 4.0."""
    (warning,) = [w for w in record if issubclass(w.category, DeprecationWarning)]
    assert warning.filename == __file__
    message = str(warning.message)
    assert spelling.format(_OLD) in message
    assert spelling.format(_NEW) in message
    assert "removed in 4.0" in message


def test_canonical_method_is_the_squared_magnitude_of_coherency(connectivity):
    coherency = connectivity.coherency()
    expected = np.clip(np.abs(coherency) ** 2, 0, 1)
    np.testing.assert_allclose(
        connectivity.magnitude_squared_coherence(), expected, rtol=1e-12, atol=0
    )


def test_deprecated_method_warns_at_the_caller_and_agrees_exactly(connectivity):
    canonical = connectivity.magnitude_squared_coherence()
    with _warns(_METHOD) as record:
        deprecated = connectivity.coherence_magnitude()
    _assert_one_caller_warning(record, _METHOD)
    np.testing.assert_array_equal(deprecated, canonical)


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(
            lambda series, method: multitaper_connectivity(
                series, sampling_frequency=500, time_window_duration=0.5, method=method
            ),
            id="multitaper_connectivity",
        ),
        pytest.param(
            lambda series, method: connectivity_to_xarray(
                Multitaper(series, sampling_frequency=500, time_window_duration=0.5), method
            ),
            id="connectivity_to_xarray",
        ),
    ],
)
def test_deprecated_method_string_keeps_its_name_values_and_warns_once(time_series, call):
    """An explicitly requested old name keeps its variable name and the values."""
    canonical = call(time_series, _NEW)
    with _warns(_STRING) as record:
        deprecated = call(time_series, _OLD)
    _assert_one_caller_warning(record, _STRING)

    assert canonical.name == _NEW
    assert deprecated.name == _OLD
    np.testing.assert_array_equal(deprecated.values, canonical.values)
    # Provenance records the name the caller asked for; the labels are the
    # canonical measure's.
    assert deprecated.attrs["measure"] == _OLD
    assert canonical.attrs["measure"] == _NEW
    assert json.loads(deprecated.attrs["measure_kwargs_json"]) == {}
    unchanged = {key: value for key, value in canonical.attrs.items() if key != "measure"}
    assert {key: value for key, value in deprecated.attrs.items() if key != "measure"} == (
        unchanged
    )


def test_deprecated_method_string_in_a_list_keeps_its_variable(time_series):
    methods = ["power", _OLD, "phase_locking_value"]
    with _warns(_STRING) as record:
        result = multitaper_connectivity(time_series, sampling_frequency=500, method=methods)
    _assert_one_caller_warning(record, _STRING)
    assert list(result.data_vars) == methods
    assert result[_OLD].attrs["measure"] == _OLD


def test_deprecated_name_requested_twice_warns_once(time_series):
    with _warns(_STRING) as record:
        result = multitaper_connectivity(
            time_series, sampling_frequency=500, method=[_OLD, _OLD]
        )
    _assert_one_caller_warning(record, _STRING)
    assert list(result.data_vars) == [_OLD]


def test_deprecated_method_string_through_fourier_connectivity(multitaper):
    coefficients = multitaper.fft()
    frequencies = multitaper.frequencies
    canonical = fourier_connectivity(coefficients, frequencies=frequencies, method=_NEW)
    with _warns(_STRING) as record:
        deprecated = fourier_connectivity(coefficients, frequencies=frequencies, method=_OLD)
    _assert_one_caller_warning(record, _STRING)
    assert deprecated.name == _OLD
    np.testing.assert_array_equal(deprecated.values, canonical.values)


def test_default_result_uses_the_canonical_variable_and_old_indexing_fails(time_series):
    """The migration example: 2.x ``result["coherence_magnitude"]`` on a default
    result raises instead of silently selecting something else."""
    result = multitaper_connectivity(time_series, sampling_frequency=500)
    assert _NEW in result.data_vars
    assert _OLD not in result.data_vars
    with pytest.raises(KeyError):
        result[_OLD]
    with pytest.raises(AttributeError):
        result.coherence_magnitude  # noqa: B018
    assert next(iter(result.data_vars)) == _NEW


def test_alias_is_neither_a_default_nor_a_separate_measure():
    names = [measure.name for measure in list_measures()]
    assert _NEW in DEFAULT_METHODS
    assert _OLD not in DEFAULT_METHODS
    assert _NEW in names
    assert _OLD not in names
    assert names.count(_NEW) == 1


def test_requesting_both_spellings_is_rejected(time_series):
    with pytest.raises(ValueError, match="deprecated name of 'magnitude_squared_coherence'"):
        multitaper_connectivity(time_series, sampling_frequency=500, method=[_NEW, _OLD])


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(
            lambda series, m: multitaper_connectivity(
                series, sampling_frequency=500, method=_OLD, transform_kwargs={"n_fft": 256}
            ),
            id="multitaper_connectivity",
        ),
        pytest.param(
            lambda series, m: fourier_connectivity(
                m.fft(), m.frequencies, method=_OLD, connectivity_kwargs={"n_range": 2}
            ),
            id="fourier_connectivity",
        ),
        pytest.param(
            lambda series, m: connectivity_to_xarray(
                m, _OLD, connectivity_kwargs={"n_range": 2}
            ),
            id="connectivity_to_xarray",
        ),
    ],
)
def test_a_rejected_call_does_not_warn_about_the_deprecated_name(
    time_series, multitaper, call
):
    """A call that is then rejected raises only its error, not the deprecation
    first (which ``-W error`` would raise in its place)."""
    with pytest.raises(TypeError):
        call(time_series, multitaper)


def test_connectivity_to_xarray_defaults_to_the_canonical_name(multitaper):
    assert inspect.signature(connectivity_to_xarray).parameters["method"].default == _NEW
    assert connectivity_to_xarray(multitaper).name == _NEW


def test_jackknife_accepts_the_deprecated_name_with_one_warning(connectivity):
    """The old name resolves to the same estimate and the same default interval
    scale (atanh(sqrt(.)) for a squared magnitude), not the identity scale an
    unknown name would get."""
    canonical = connectivity.jackknife(_NEW)
    with _warns(_JACKKNIFE) as record:
        deprecated = connectivity.jackknife(_OLD)
    _assert_one_caller_warning(record, _JACKKNIFE)
    assert deprecated.transformation == canonical.transformation == "fisher_squared"
    for field in ("estimate", "bias_corrected", "standard_error"):
        np.testing.assert_array_equal(getattr(deprecated, field), getattr(canonical, field))
    for deprecated_bound, canonical_bound in zip(
        deprecated.confidence_interval, canonical.confidence_interval, strict=True
    ):
        np.testing.assert_array_equal(deprecated_bound, canonical_bound)


def test_saved_result_requested_by_the_old_name_reads_back_by_that_name(time_series, tmp_path):
    with _warns(_STRING):
        result = multitaper_connectivity(time_series, sampling_frequency=500, method=[_OLD])
    path = tmp_path / "result.nc"
    result.to_netcdf(path, engine="scipy")
    with xr.open_dataset(path, engine="scipy") as loaded:
        assert loaded[_OLD].attrs["measure"] == _OLD
