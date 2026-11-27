import inspect
import json
import warnings
from unittest import mock

import numpy as np
import pytest
import scipy.fft
import xarray as xr

from spectral_connectivity import MorletWavelet, Multitaper, Welch
from spectral_connectivity._input_handling import (
    _reject_unmaterialized_backing,
    _time_axis_from_dataarray,
)
from spectral_connectivity._measure_registry import (
    _MEASURE_SPECS,
    _is_group_measure,
    _MeasureSpec,
    _requires_two_sided,
)
from spectral_connectivity._provenance import (
    _canonical_json,
    _json_compatible,
    _netcdf_provenance_value,
)
from spectral_connectivity.connectivity import _NON_MEASURE_METHODS, Connectivity
from spectral_connectivity.wrapper import (
    DEFAULT_METHODS,
    connectivity_to_xarray,
    fourier_connectivity,
    frequency_band_reduce,
    list_measures,
    multitaper_connectivity,
)
from tests._backend_helpers import to_host
from tests._spectral_transforms import PrecomputedTransform

# Shared window-grid case: 2.4 s at 500 Hz plus one sample, so most window
# durations leave a trailing partial window that must be dropped.
_GRID_SAMPLING_FREQUENCY = 500
_GRID_N_SAMPLES = 1201
_GRID_N_TRIALS = 3


def _expected_window_centers(
    n_samples: int, sampling_frequency: float, window_duration: float
) -> np.ndarray:
    """Center times of the non-overlapping full windows, from first principles.

    Returns
    -------
    centers : np.ndarray, shape (n_windows,)
        ``(start + (n_window - 1) / 2) / fs`` for each window that fits entirely
        in the record; a trailing partial window is not labeled.
    """
    n_window = round(window_duration * sampling_frequency)
    n_windows = (n_samples - n_window) // n_window + 1
    starts = np.arange(n_windows) * n_window
    return (starts + (n_window - 1) / 2) / sampling_frequency


def _grid_noise(n_signals: int, seed: int) -> np.ndarray:
    """White noise shaped ``(_GRID_N_SAMPLES, _GRID_N_TRIALS, n_signals)``."""
    return np.random.default_rng(seed).standard_normal(
        (_GRID_N_SAMPLES, _GRID_N_TRIALS, n_signals)
    )


@pytest.mark.parametrize("time_window_duration", [0.1, 0.14, 0.16, 2.4])
def test_coherence_magnitude_is_bounded_on_window_center_time_grid(time_window_duration):
    """Windows are labeled by center time and coherence stays in [0, 1].

    Signal 1 is signal 0 plus 5% noise (coherence near 1) and signal 2 is
    independent, so a scaled or mis-normalized estimate leaves the unit range.
    """
    noise = _grid_noise(3, seed=42)
    time_series = noise.copy()
    time_series[..., 1] = noise[..., 0] + 0.05 * noise[..., 1]

    m = multitaper_connectivity(
        time_series,
        method="magnitude_squared_coherence",
        sampling_frequency=_GRID_SAMPLING_FREQUENCY,
        time_window_duration=time_window_duration,
    )

    np.testing.assert_allclose(
        m.time.values,
        _expected_window_centers(
            _GRID_N_SAMPLES, _GRID_SAMPLING_FREQUENCY, time_window_duration
        ),
    )
    off_diagonal = m.values[..., ~np.eye(3, dtype=bool)]
    assert np.all((off_diagonal >= 0) & (off_diagonal <= 1 + 1e-12))
    # Squared coherence of x with x + 0.05 * noise is ~400/401 at every bin.
    assert np.all(m.sel(source="0", target="1").values > 0.9)


@pytest.mark.parametrize(
    "method",
    [
        "canonical_coherence",
        "maximized_imaginary_coherency",
        "multivariate_interaction_measure",
    ],
)
def test_connectivity_to_xarray_exposes_group_pairwise_results(method):
    rng = np.random.default_rng(42)
    transform = Multitaper(rng.standard_normal((256, 6, 4)), sampling_frequency=128)
    result = connectivity_to_xarray(
        transform,
        method=method,
        signal_names=["a", "b", "c", "d"],
        group_labels=[10, 10, 20, 20],
    )

    assert result.dims == ("time", "frequency", "source_group", "target_group")
    assert result.source_group.values.tolist() == [10, 20]
    assert result.target_group.values.tolist() == [10, 20]


def test_group_pairwise_directed_orientation_is_source_to_target(monkeypatch):
    transform = Multitaper(
        np.random.default_rng(427).standard_normal((128, 3, 4)),
        sampling_frequency=64,
    )

    def blockwise(self, group_labels):
        values = np.zeros((len(self.time), len(self.frequencies), 2, 2))
        values[..., 0, 1] = 7.0  # Connectivity's [..., source, target]: group 0 -> group 1
        return values, np.array([10, 20])

    monkeypatch.setattr(Connectivity, "blockwise_spectral_granger_prediction", blockwise)
    result = connectivity_to_xarray(
        transform,
        method="blockwise_spectral_granger_prediction",
        group_labels=[10, 10, 20, 20],
    )

    assert np.all(result.sel(source_group=10, target_group=20) == 7)
    assert np.all(result.sel(source_group=20, target_group=10) == 0)


def test_connectivity_to_xarray_exposes_rich_multivariate_components():
    rng = np.random.default_rng(43)
    transform = Multitaper(rng.standard_normal((256, 8, 4)), sampling_frequency=128)
    result = connectivity_to_xarray(
        transform,
        method="canonical_coherency",
        signal_names=["a", "b", "c", "d"],
        group_labels=[10, 10, 20, 20],
        connectivity_kwargs={"n_components": 2},
    )

    assert set(result.data_vars) == {
        "canonical_coherency",
        "canonical_coherency_filters",
        "canonical_coherency_patterns",
        "group_membership",
    }
    assert result.canonical_coherency.dims == (
        "time",
        "frequency",
        "connection",
        "component",
    )
    assert result.canonical_coherency_filters.dims == (
        "time",
        "frequency",
        "connection",
        "component",
        "side",
        "signal",
    )
    assert result.connection_source_group.values.tolist() == [10]
    assert result.connection_target_group.values.tolist() == [20]
    assert result.side.values.tolist() == ["source", "target"]
    # The source side's filter is NaN outside the source group (signals c, d).
    source_filters = result.canonical_coherency_filters.sel(side="source")
    assert np.isnan(source_filters.sel(signal=["c", "d"])).all()
    assert np.isfinite(source_filters.sel(signal=["a", "b"])).all()
    assert result.group_membership.sel(group=10, signal="a").item()
    assert result.group_membership.attrs["output_schema_version"] == 1
    assert result.group_membership.attrs["transform"] == "multitaper"


def test_connectivity_to_xarray_exposes_global_components():
    transform = Multitaper(
        np.random.default_rng(44).standard_normal((256, 6, 3)),
        sampling_frequency=128,
    )
    result = connectivity_to_xarray(
        transform,
        method="global_coherence",
        signal_names=["a", "b", "c"],
        connectivity_kwargs={"max_rank": 2},
    )

    assert set(result.data_vars) == {
        "global_coherence",
        "global_coherence_vectors",
    }
    assert result.global_coherence.dims == ("time", "frequency", "component")
    assert result.global_coherence_vectors.dims == (
        "time",
        "frequency",
        "source",
        "component",
    )
    assert result.sizes["component"] == 2


def test_connectivity_to_xarray_exposes_delay_and_frequency_reduced_results():
    transform = Multitaper(
        np.random.default_rng(45).standard_normal((256, 12, 2)),
        sampling_frequency=128,
    )
    delay = connectivity_to_xarray(
        transform,
        method="delay",
        signal_names=["a", "b"],
        frequency_range=(8, 40),
        connectivity_kwargs={"n_range": 1},
    )
    psi = connectivity_to_xarray(
        transform,
        method="phase_slope_index",
        signal_names=["a", "b"],
        frequency_range=(8, 40),
    )
    group_delay = connectivity_to_xarray(
        transform,
        method="group_delay",
        signal_names=["a", "b"],
        frequency_range=(8, 40),
    )

    assert delay.dims == ("time", "frequency", "candidate", "source", "target")
    assert delay.candidate.values.tolist() == [-1, 0, 1]
    assert delay.frequency.values[[0, -1]].tolist() == [8, 40]
    assert psi.dims == ("time", "source", "target")
    assert psi.attrs["requested_frequency_lower"] == 8
    assert psi.attrs["requested_frequency_upper"] == 40
    assert group_delay.group_delay.attrs["requested_frequency_lower"] == 8
    assert group_delay.group_delay.attrs["requested_frequency_upper"] == 40
    assert set(group_delay.data_vars) == {
        "group_delay",
        "group_delay_slope",
        "group_delay_r_value",
    }
    assert group_delay.group_delay.attrs["units"] == "s"


def test_fourier_connectivity_exposes_global_dataset():
    rng = np.random.default_rng(429)
    coefficients = rng.standard_normal((2, 4, 2, 16, 3)) + 1j * rng.standard_normal(
        (2, 4, 2, 16, 3)
    )
    result = fourier_connectivity(
        coefficients,
        frequencies=np.fft.fftfreq(16, d=1 / 128),
        method="global_coherence",
        connectivity_kwargs={"max_rank": 2},
    )

    assert set(result.data_vars) == {
        "global_coherence",
        "global_coherence_vectors",
    }
    assert result.sizes["frequency"] == 9


@pytest.mark.parametrize("method", [*DEFAULT_METHODS, "coherency"])
def test_single_method_result_is_on_window_center_time_grid(method):
    """Every default measure (plus complex coherency) labels the same time grid."""
    n_signals, time_window_duration = 3, 0.1
    m = multitaper_connectivity(
        _grid_noise(n_signals, seed=0),
        method=method,
        sampling_frequency=_GRID_SAMPLING_FREQUENCY,
        time_window_duration=time_window_duration,
    )

    np.testing.assert_allclose(
        m.time.values,
        _expected_window_centers(
            _GRID_N_SAMPLES, _GRID_SAMPLING_FREQUENCY, time_window_duration
        ),
    )
    assert m.sizes["source"] == n_signals
    assert m.sizes.get("target", n_signals) == n_signals
    assert not (m.values == 0).all()
    assert not np.isnan(m.values).all()


@pytest.mark.parametrize("n_signals", [2, 3, 4])
def test_default_dataset_variables_share_window_center_time_grid(n_signals):
    """The merged default Dataset (and a one-item method list) keeps the grid."""
    time_series = _grid_noise(n_signals, seed=0)
    time_window_duration = 0.1
    expected_time = _expected_window_centers(
        _GRID_N_SAMPLES, _GRID_SAMPLING_FREQUENCY, time_window_duration
    )

    cons = multitaper_connectivity(
        time_series,
        sampling_frequency=_GRID_SAMPLING_FREQUENCY,
        time_window_duration=time_window_duration,
    )
    assert tuple(cons.data_vars) == DEFAULT_METHODS
    for mea in cons.data_vars:
        np.testing.assert_allclose(cons[mea].time.values, expected_time)
        assert cons[mea].sizes["source"] == n_signals
        assert not (cons[mea].values == 0).all()
        assert not np.isnan(cons[mea].values).all()

    # A one-item method list still returns a Dataset.
    cons = multitaper_connectivity(
        time_series,
        method=["magnitude_squared_coherence"],
        sampling_frequency=_GRID_SAMPLING_FREQUENCY,
        time_window_duration=time_window_duration,
    )
    assert isinstance(cons, xr.Dataset)
    assert tuple(cons.data_vars) == ("magnitude_squared_coherence",)
    np.testing.assert_allclose(cons["magnitude_squared_coherence"].time.values, expected_time)


def test_default_dataset_frequency_grid_is_nonnegative_fft_bins():
    rng = np.random.default_rng(42)
    n_time_samples, n_trials, n_signals = 64, 3, 2
    time_series = rng.random((n_time_samples, n_trials, n_signals))
    # n_fft_samples must be >= the window length (here the full 64 samples),
    # otherwise the FFT would silently truncate the signal.
    n_fft_samples = 64
    sampling_frequency = 1000

    cons = multitaper_connectivity(
        time_series,
        sampling_frequency=sampling_frequency,
        time_window_duration=None,
        transform_kwargs={"n_fft_samples": n_fft_samples},
    )

    # Non-negative frequency grid, 0 .. Nyquist (n_fft // 2 + 1 bins).
    expected_frequencies = (
        sampling_frequency * np.arange(0, n_fft_samples // 2 + 1) / n_fft_samples
    )
    for mea in cons.data_vars:
        assert not (cons[mea].values == 0).all()
        assert not (np.isnan(cons[mea].values)).all()
        assert np.allclose(cons[mea].frequency, expected_frequencies)


def test_default_methods_are_real_connectivity_methods():
    """Every DEFAULT_METHODS entry must be a real public Connectivity method.

    The default set is an explicit allowlist (no longer discovered by
    inspecting Connectivity), so a typo or a rename of a measure would silently
    make the default request a nonexistent method. Guard the allowlist against
    that by checking each name resolves to a public callable on Connectivity.
    """
    from spectral_connectivity.wrapper import DEFAULT_METHODS

    public_callables = {
        name
        for name, _ in inspect.getmembers(Connectivity, predicate=inspect.isfunction)
        if not name.startswith("_")
    }
    for name in DEFAULT_METHODS:
        assert name in public_callables, f"{name} is not a public Connectivity method"


def test_result_is_netcdf_serializable(tmp_path):
    """The xarray result must round-trip through NetCDF.

    Copying callable Multitaper members (e.g. the bound ``summarize_parameters``
    method) into ``attrs`` makes ``to_netcdf`` raise.
    """
    rng = np.random.default_rng(0)
    time_series = rng.standard_normal((512, 3, 2))
    result = multitaper_connectivity(
        time_series,
        sampling_frequency=500,
        method="magnitude_squared_coherence",
    )
    # No attribute value may be a callable.
    assert not any(callable(v) for v in result.attrs.values())
    path = tmp_path / "conn.nc"
    result.to_netcdf(path)
    assert path.exists()


def test_accepts_documented_2d_input():
    """The documented (n_times, n_channels) 2-D form must work.

    It is promoted to a single-trial 3-D array internally and must match the
    explicit 3-D form.
    """
    rng = np.random.default_rng(0)
    data_2d = rng.standard_normal((512, 2))
    data_3d = data_2d[:, np.newaxis, :]
    result_2d = multitaper_connectivity(
        data_2d, sampling_frequency=500, method="magnitude_squared_coherence"
    )
    result_3d = multitaper_connectivity(
        data_3d, sampling_frequency=500, method="magnitude_squared_coherence"
    )
    np.testing.assert_allclose(result_2d.values, result_3d.values, equal_nan=True)


def test_nested_list_input_matches_the_array_input():
    """A nested list is read like the array it describes: 2-D time series in
    multitaper_connectivity, and 3-D and 4-D coefficients in
    fourier_connectivity, whose shapes the wrappers check before converting."""
    data = np.random.default_rng(0).standard_normal((512, 2))
    from_array = multitaper_connectivity(data, sampling_frequency=500, method="power")
    from_list = multitaper_connectivity(data.tolist(), sampling_frequency=500, method="power")
    xr.testing.assert_identical(from_list, from_array)

    # A contiguous copy, which the list's array is: a strided view can change
    # the summation order, and so the last bit.
    coefficients = np.ascontiguousarray(
        to_host(Multitaper(data[:, np.newaxis, :], sampling_frequency=500).fft()[0])
    )
    frequencies = np.fft.fftfreq(coefficients.shape[-2], d=1 / 500)
    for layout in (coefficients[0], coefficients):  # 3-D and 4-D
        from_array = fourier_connectivity(layout, frequencies=frequencies, method="coherency")
        from_list = fourier_connectivity(
            layout.tolist(), frequencies=frequencies, method="coherency"
        )
        xr.testing.assert_identical(from_list, from_array)


@pytest.mark.parametrize(
    ("time_series", "error", "match"),
    [
        ([[1.0, 2.0], [3.0]], ValueError, "rows of different lengths"),
        ([["1", "2"], ["3", "4"]], TypeError, "dtype <U1"),
    ],
    ids=["ragged", "strings"],
)
def test_time_series_that_is_not_a_numeric_array_is_rejected_clearly(
    time_series, error, match
):
    """A ragged nested list or one of strings names the expected array instead
    of NumPy's inhomogeneous-shape or ufunc message."""
    with pytest.raises(error, match=match) as raised:
        multitaper_connectivity(time_series, sampling_frequency=500)
    assert "(n_time, n_signals) or (n_time, n_trials, n_signals)" in str(raised.value)


@pytest.mark.parametrize("dtype", [np.float32, np.int64, np.float64])
def test_array_input_passes_through_and_matches_its_list(dtype):
    """An ndarray reaches the transform as it is, and its nested list gives the
    same result. A float32 array keeps its precision, so its list (Python
    floats, read as float64) agrees only to float32 rounding."""
    data = (100 * np.random.default_rng(0).standard_normal((512, 2))).astype(dtype)
    from_array = multitaper_connectivity(data, sampling_frequency=500, method="power")
    from_list = multitaper_connectivity(data.tolist(), sampling_frequency=500, method="power")
    if dtype is np.float32:
        np.testing.assert_allclose(from_array.values, from_list.values, rtol=1e-5)
    else:
        xr.testing.assert_identical(from_array, from_list)


def test_fourier_array_input_matches_the_multitaper_wrapper():
    """Coefficients passed as an ndarray give the coherency multitaper_connectivity
    computes from the same transform."""
    data = np.random.default_rng(0).standard_normal((512, 2))
    multitaper = Multitaper(data[:, np.newaxis, :], sampling_frequency=500)
    from_coefficients = fourier_connectivity(
        multitaper.fft()[0], frequencies=multitaper.frequencies, method="coherency"
    )
    from_time_series = multitaper_connectivity(
        data, sampling_frequency=500, method="coherency"
    )
    np.testing.assert_array_equal(
        from_coefficients.frequency.values, from_time_series.frequency.values
    )
    np.testing.assert_allclose(
        from_coefficients.values, from_time_series.values, rtol=1e-12, equal_nan=True
    )


class _ForeignArray:
    """An array-like whose ``dtype`` is not a NumPy dtype (as a torch tensor's)."""

    dtype = object()

    def __init__(self, values):
        self._values = values

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self._values, dtype=dtype)


def test_array_like_with_a_foreign_dtype_is_converted():
    """An array-like whose dtype NumPy cannot read is converted like a list
    rather than failing on ``dtype.kind``."""
    data = np.random.default_rng(0).standard_normal((512, 2))
    xr.testing.assert_identical(
        multitaper_connectivity(_ForeignArray(data), sampling_frequency=500, method="power"),
        multitaper_connectivity(data, sampling_frequency=500, method="power"),
    )


# Channel 0 ("right") has 25x the variance of channel 1 ("left") and the labels
# are deliberately unsorted, so per-signal power exposes a data/label
# mispairing, a channel reversal, or a label sort (a symmetric two-signal
# measure such as magnitude_squared_coherence cannot).
_UNSORTED_LABELS = ["right", "left"]
_CHANNEL_SCALE = np.array([5.0, 1.0])


def _assert_power_follows_labels(result: xr.DataArray) -> None:
    """The loud channel's power is reported under its own label, in input order."""
    assert result.source.values.tolist() == _UNSORTED_LABELS
    loud = float(result.sel(source="right").mean())
    quiet = float(result.sel(source="left").mean())
    assert loud > 10 * quiet


@pytest.mark.parametrize("with_trial_dimension", [False, True])
def test_dataarray_input_preserves_signal_labels(with_trial_dimension):
    """A DataArray's final dimension labels carry through to the result."""
    rng = np.random.default_rng(4)
    if with_trial_dimension:
        data = rng.standard_normal((256, 3, 2)) * _CHANNEL_SCALE
        dims = ("sample", "trial", "channel")
    else:
        data = rng.standard_normal((256, 2)) * _CHANNEL_SCALE
        dims = ("sample", "channel")
    labeled = xr.DataArray(
        data,
        dims=dims,
        coords={"channel": _UNSORTED_LABELS},
    )

    actual = multitaper_connectivity(
        labeled,
        sampling_frequency=256,
        method="power",
    )
    expected = multitaper_connectivity(
        data,
        sampling_frequency=256,
        method="power",
        signal_names=_UNSORTED_LABELS,
    )

    xr.testing.assert_identical(actual, expected)
    _assert_power_follows_labels(actual)


def test_explicit_signal_names_override_dataarray_coordinate():
    data = xr.DataArray(
        np.random.default_rng(5).standard_normal((256, 2)),
        dims=("sample", "channel"),
        coords={"channel": ["left", "right"]},
    )

    result = multitaper_connectivity(
        data,
        sampling_frequency=256,
        method="magnitude_squared_coherence",
        signal_names=["first", "second"],
    )

    assert result.coords["source"].values.tolist() == ["first", "second"]
    assert result.coords["target"].values.tolist() == ["first", "second"]


def test_dataarray_without_final_coordinate_uses_default_labels():
    """A DataArray whose signal dim has no coordinate falls back to indices."""
    data = xr.DataArray(
        np.random.default_rng(6).standard_normal((256, 2)),
        dims=("sample", "channel"),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # bare unlabeled input must not warn
        result = multitaper_connectivity(
            data, sampling_frequency=256, method="magnitude_squared_coherence"
        )
    assert result.coords["source"].values.tolist() == ["0", "1"]


def test_dataarray_non_1d_final_coordinate_warns_and_uses_default_labels():
    """A non-1-D coordinate on the signal dim cannot label it; warn and fall back."""
    # A non-index name (not equal to a dimension) so older xarray accepts the
    # 2-D coordinate; it still spans the signal dim, so it is unusable as labels.
    data = xr.DataArray(
        np.random.default_rng(7).standard_normal((256, 2)),
        dims=("sample", "channel"),
        coords={"channel_grid": (("sample", "channel"), np.zeros((256, 2)))},
    )
    with pytest.warns(UserWarning, match="not a 1-D index coordinate"):
        result = multitaper_connectivity(
            data, sampling_frequency=256, method="magnitude_squared_coherence"
        )
    assert result.coords["source"].values.tolist() == ["0", "1"]


def test_dataarray_labels_on_non_index_coordinate_warn():
    """Labels on a differently-named coordinate are not silently dropped."""
    data = xr.DataArray(
        np.random.default_rng(8).standard_normal((256, 2)),
        dims=("sample", "channel"),
        coords={"channel_name": ("channel", ["left", "right"])},
    )
    with pytest.warns(UserWarning, match="not a 1-D index coordinate"):
        result = multitaper_connectivity(
            data, sampling_frequency=256, method="magnitude_squared_coherence"
        )
    assert result.coords["source"].values.tolist() == ["0", "1"]


def test_dataarray_integer_final_coordinate_is_preserved():
    """Coordinate label types survive xarray input/output round-tripping."""
    data = xr.DataArray(
        np.random.default_rng(9).standard_normal((256, 2)),
        dims=("sample", "channel"),
        coords={"channel": [10, 20]},
    )
    result = multitaper_connectivity(
        data, sampling_frequency=256, method="magnitude_squared_coherence"
    )
    assert result.coords["source"].values.tolist() == [10, 20]
    assert result.sel(source=10).coords["source"].item() == 10


def test_dataarray_datetime_signal_coordinate_is_preserved():
    """Nanosecond datetime labels must not be coerced to integer timestamps."""
    labels = np.array(["2025-01-01", "2025-01-02"], dtype="datetime64[ns]")
    data = xr.DataArray(
        np.random.default_rng(18).standard_normal((256, 2)),
        dims=("sample", "channel"),
        coords={"channel": labels},
    )

    result = multitaper_connectivity(
        data, sampling_frequency=256, method="magnitude_squared_coherence"
    )

    assert result.source.dtype == np.dtype("datetime64[ns]")
    np.testing.assert_array_equal(result.source.values, labels)


def test_dataarray_named_dimensions_are_transposed_automatically():
    """Dimension names, rather than input position, determine semantic axes."""
    raw = np.random.default_rng(10).standard_normal((256, 2)) * _CHANNEL_SCALE
    data = xr.DataArray(
        raw.T,
        dims=("channel", "time"),
        coords={"channel": _UNSORTED_LABELS},
    )

    actual = multitaper_connectivity(data, sampling_frequency=256, method="power")
    expected = multitaper_connectivity(
        raw,
        sampling_frequency=256,
        method="power",
        signal_names=_UNSORTED_LABELS,
    )

    xr.testing.assert_identical(actual, expected)
    _assert_power_follows_labels(actual)


def test_dataarray_swapped_time_and_trial_dims_are_transposed_automatically():
    """Named trial/time axes are normalized before entering numerical code."""
    raw = np.random.default_rng(11).standard_normal((256, 4, 2)) * _CHANNEL_SCALE
    data = xr.DataArray(
        raw.transpose(1, 0, 2),
        dims=("trial", "time", "channel"),
        coords={"channel": _UNSORTED_LABELS},
    )

    actual = multitaper_connectivity(data, sampling_frequency=256, method="power")
    expected = multitaper_connectivity(
        raw,
        sampling_frequency=256,
        method="power",
        signal_names=_UNSORTED_LABELS,
    )

    xr.testing.assert_identical(actual, expected)
    _assert_power_follows_labels(actual)


def test_dataarray_unrecognized_dimensions_require_explicit_roles():
    """Domain-specific dimension names never fall back to unsafe positions."""
    raw = np.random.default_rng(14).standard_normal((8, 256, 2))
    data = xr.DataArray(
        raw,
        dims=("replicate_id", "clock_tick", "unit_id"),
        coords={"unit_id": ["left", "right"]},
    )

    with pytest.raises(ValueError, match="time_dim, trial_dim, signal_dim"):
        multitaper_connectivity(data, sampling_frequency=256, method="power")

    actual = multitaper_connectivity(
        data,
        sampling_frequency=256,
        method="power",
        time_dim="clock_tick",
        trial_dim="replicate_id",
        signal_dim="unit_id",
    )
    expected = multitaper_connectivity(
        raw.transpose(1, 0, 2),
        sampling_frequency=256,
        method="power",
        signal_names=["left", "right"],
    )

    xr.testing.assert_identical(actual, expected)


def test_dataarray_single_unrecognized_dimension_warns_before_assuming_role():
    """The by-elimination role assignment is not silent (it could average data)."""
    raw = np.random.default_rng(40).standard_normal((256, 4, 2)) * _CHANNEL_SCALE
    data = xr.DataArray(
        raw,
        dims=("time", "drug_dose", "channel"),
        coords={"channel": _UNSORTED_LABELS},
    )
    with pytest.warns(UserWarning, match=r"Assuming.*'drug_dose'.*trial axis"):
        result = multitaper_connectivity(data, sampling_frequency=256, method="power")
    # Naming the role silences the warning and gives the same result.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        explicit = multitaper_connectivity(
            data,
            sampling_frequency=256,
            method="power",
            trial_dim="drug_dose",
        )
    xr.testing.assert_identical(result, explicit)
    _assert_power_follows_labels(result)


@pytest.mark.parametrize(
    ("dims", "shape"),
    [
        (("time", "frequency"), (256, 8)),
        (("time", "freq"), (256, 8)),
        (("time", "trial", "band"), (256, 3, 4)),
    ],
    ids=["2d-frequency", "2d-freq", "3d-band"],
)
def test_dataarray_spectral_dimension_is_not_promoted_to_a_time_series_role(dims, shape):
    """A ``frequency``/``band`` dimension marks an already-transformed input such
    as a spectrogram. Filling the last role with it by elimination would run
    frequency bins as channels, so it is rejected instead of merely warned."""
    data = xr.DataArray(np.random.default_rng(42).standard_normal(shape), dims=dims)
    with pytest.raises(ValueError, match="spectral") as excinfo:
        multitaper_connectivity(data, sampling_frequency=256, method="power")
    message = str(excinfo.value)
    assert repr(dims[-1]) in message
    assert "fourier_connectivity" in message
    # Naming the role explicitly remains the escape hatch for a signal axis
    # that really is called that.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        multitaper_connectivity(
            data,
            sampling_frequency=256,
            method="power",
            signal_dim=dims[-1],
            **({"trial_dim": "trial"} if len(dims) == 3 else {}),
        )


def test_dataarray_explicit_role_conflicting_with_recognized_name_is_rejected():
    data = xr.DataArray(
        np.random.default_rng(41).standard_normal((256, 2)),
        dims=("time", "channel"),
    )
    with pytest.raises(ValueError, match="conflicts with its recognized"):
        multitaper_connectivity(
            data, sampling_frequency=256, method="power", signal_dim="time"
        )


def test_dataarray_two_dimensions_inferring_same_role_are_rejected():
    data = xr.DataArray(
        np.random.default_rng(42).standard_normal((256, 3, 2)),
        dims=("channel", "electrode", "time"),
    )
    with pytest.raises(ValueError, match="both denote the signal axis"):
        multitaper_connectivity(data, sampling_frequency=256, method="power")


def test_dataarray_trial_dim_rejected_for_2d_input():
    data = xr.DataArray(
        np.random.default_rng(43).standard_normal((256, 2)),
        dims=("time", "channel"),
    )
    with pytest.raises(ValueError, match="trial_dim cannot be used with a 2-D"):
        multitaper_connectivity(
            data, sampling_frequency=256, method="power", trial_dim="channel"
        )


def test_dataarray_explicit_dim_naming_nonexistent_dimension_is_rejected():
    data = xr.DataArray(
        np.random.default_rng(44).standard_normal((256, 2)),
        dims=("time", "channel"),
    )
    with pytest.raises(ValueError, match="is not an input dimension"):
        multitaper_connectivity(
            data, sampling_frequency=256, method="power", signal_dim="nope"
        )


def test_dataarray_same_dimension_assigned_to_two_roles_is_rejected():
    data = xr.DataArray(
        np.random.default_rng(45).standard_normal((256, 2)),
        dims=("a", "b"),
    )
    with pytest.raises(ValueError, match="was assigned to both"):
        multitaper_connectivity(
            data,
            sampling_frequency=256,
            method="power",
            time_dim="a",
            signal_dim="a",
        )


def test_dataarray_ambiguous_time_coordinates_are_rejected():
    raw = np.random.default_rng(46).standard_normal((128, 2))
    seconds = np.arange(raw.shape[0]) / 64.0
    data = xr.DataArray(
        raw,
        dims=("time", "channel"),
        coords={
            "timestamp": ("time", seconds),
            "times": ("time", seconds),
            "channel": ["left", "right"],
        },
    )
    with pytest.raises(ValueError, match="Multiple coordinates"):
        multitaper_connectivity(data, sampling_frequency=64, method="power")


def test_dataarray_case_insensitive_duplicate_time_coordinates_are_rejected():
    """Two coordinates that are both case-insensitively 'time' are ambiguous."""
    raw = np.random.default_rng(48).standard_normal((128, 2))
    seconds = np.arange(raw.shape[0]) / 64.0
    data = xr.DataArray(
        raw,
        dims=("time", "channel"),
        coords={
            "time": seconds,
            "TIME": ("time", 100.0 + seconds),
            "channel": ["left", "right"],
        },
    )
    with pytest.raises(ValueError, match="Multiple coordinates"):
        multitaper_connectivity(data, sampling_frequency=64, method="power")


@pytest.mark.parametrize("bad_rate", [0, -64, float("nan"), float("inf")])
def test_dataarray_nonpositive_sampling_frequency_is_rejected(bad_rate):
    """A non-positive/non-finite rate raises a clear error, not ZeroDivisionError."""
    raw = np.random.default_rng(49).standard_normal((128, 2))
    data = xr.DataArray(
        raw,
        dims=("time", "channel"),
        coords={"time": np.arange(raw.shape[0]) / 64.0},
    )
    with pytest.raises(ValueError, match="sampling_frequency must be finite and positive"):
        multitaper_connectivity(data, sampling_frequency=bad_rate, method="power")


def test_dataarray_path_rejects_string_sampling_frequency():
    """The DataArray time-coordinate check gives the same typed error as the transforms."""
    raw = np.random.default_rng(49).standard_normal((128, 2))
    data = xr.DataArray(
        raw,
        dims=("time", "channel"),
        coords={"time": np.arange(raw.shape[0]) / 500.0},
    )
    with pytest.raises(TypeError, match="sampling_frequency must be a number") as excinfo:
        _time_axis_from_dataarray(data, "time", "500")
    assert "'500'" in str(excinfo.value)
    with pytest.raises(TypeError, match="sampling_frequency must be a number"):
        multitaper_connectivity(data, sampling_frequency="500", method="power")


def test_dataarray_non_scalar_start_time_is_rejected():
    raw = np.random.default_rng(47).standard_normal((128, 2))
    data = xr.DataArray(
        raw,
        dims=("time", "channel"),
        coords={"time": np.arange(raw.shape[0]) / 64.0},
    )
    with pytest.raises(ValueError, match="requires scalar start_time"):
        multitaper_connectivity(
            data,
            sampling_frequency=64,
            method="power",
            transform_kwargs={"start_time": [0, 1]},
        )


def test_measure_spec_labels_are_keyword_only():
    """The labels cannot be passed positionally, so a mix-up cannot pass silently."""
    labels = {
        "long_name": "Label",
        "units": "1",
        "value_range": (0.0, 1.0),
        "interpretation": "Interpretation.",
    }
    with pytest.raises(TypeError, match="positional"):
        _MeasureSpec("pairwise", *labels.values())  # labels must be named


def test_dataarray_numeric_time_coordinate_sets_output_time():
    """A numeric time index supplies the transform's start time in seconds."""
    sampling_frequency = 64
    raw = np.random.default_rng(15).standard_normal((128, 2))
    time = 10.0 + np.arange(raw.shape[0]) / sampling_frequency
    data = xr.DataArray(
        raw,
        dims=("time", "channel"),
        coords={"time": time, "channel": ["left", "right"]},
    )

    actual = multitaper_connectivity(
        data, sampling_frequency=sampling_frequency, method="power"
    )
    expected = multitaper_connectivity(
        raw,
        sampling_frequency=sampling_frequency,
        method="power",
        signal_names=["left", "right"],
        transform_kwargs={"start_time": time[0]},
    )

    xr.testing.assert_identical(actual, expected)
    assert actual.time.item() == pytest.approx(10.9921875)


def test_dataarray_auxiliary_time_coordinate_sets_output_time():
    """A named time coordinate may label a separate sample dimension."""
    sampling_frequency = 64
    raw = np.random.default_rng(19).standard_normal((128, 2))
    time = 10.0 + np.arange(raw.shape[0]) / sampling_frequency
    data = xr.DataArray(
        raw,
        dims=("sample", "channel"),
        coords={
            "sample": np.arange(raw.shape[0]),
            "time": ("sample", time),
            "channel": ["left", "right"],
        },
    )

    actual = multitaper_connectivity(
        data, sampling_frequency=sampling_frequency, method="power"
    )

    assert actual.time.item() == pytest.approx(10.9921875)


def test_dataarray_sample_coordinate_sets_output_time():
    """A sample-number index is converted to elapsed seconds."""
    sampling_frequency = 64
    raw = np.random.default_rng(20).standard_normal((128, 2))
    data = xr.DataArray(
        raw,
        dims=("sample", "channel"),
        coords={
            "sample": 640 + np.arange(raw.shape[0]),
            "channel": ["left", "right"],
        },
    )

    actual = multitaper_connectivity(
        data, sampling_frequency=sampling_frequency, method="power"
    )

    assert actual.time.item() == pytest.approx(10.9921875)


def test_dataarray_large_float32_time_coordinate_uses_actual_resolution():
    """Representable float32 axes are valid even at a large absolute offset."""
    sampling_frequency = 4
    start_time = 1_000_000.0
    raw = np.random.default_rng(21).standard_normal((128, 2))
    time = (start_time + np.arange(raw.shape[0]) / sampling_frequency).astype(np.float32)
    data = xr.DataArray(
        raw,
        dims=("time", "channel"),
        coords={"time": time, "channel": ["left", "right"]},
    )

    actual = multitaper_connectivity(
        data, sampling_frequency=sampling_frequency, method="power"
    )

    assert actual.time.item() == pytest.approx(start_time + 15.875)


def test_dataarray_datetime_time_coordinate_has_conversion_hint():
    """Datetime time axes fail explicitly until absolute-time output is supported."""
    raw = np.random.default_rng(22).standard_normal((128, 2))
    time = np.datetime64("2025-01-01") + np.arange(raw.shape[0]) * np.timedelta64(1, "s")
    data = xr.DataArray(
        raw,
        dims=("time", "channel"),
        coords={"time": time, "channel": ["left", "right"]},
    )

    with pytest.raises(TypeError, match="not yet supported") as excinfo:
        multitaper_connectivity(data, sampling_frequency=1, method="power")
    # The message names the offending dtype and gives a copy-paste conversion.
    assert "M8" in str(excinfo.value) or "datetime64" in str(excinfo.value)
    assert "np.timedelta64(1, 's')" in str(excinfo.value)


def test_dataarray_time_spacing_must_match_sampling_frequency():
    raw = np.random.default_rng(16).standard_normal((128, 2))
    data = xr.DataArray(
        raw,
        dims=("time", "channel"),
        coords={"time": np.arange(raw.shape[0]) / 32},
    )

    with pytest.raises(ValueError, match="spacing does not match"):
        multitaper_connectivity(data, sampling_frequency=64, method="power")


def test_dataarray_time_coordinate_must_agree_with_explicit_start_time():
    raw = np.random.default_rng(17).standard_normal((128, 2))
    data = xr.DataArray(
        raw,
        dims=("time", "channel"),
        coords={"time": 10 + np.arange(raw.shape[0]) / 64},
    )

    with pytest.raises(ValueError, match=r"start_time=.*conflicts"):
        multitaper_connectivity(
            data,
            sampling_frequency=64,
            method="power",
            transform_kwargs={"start_time": 0},
        )


def test_dataarray_infers_sampling_frequency_from_time_coordinate():
    """An elapsed-seconds time coordinate supplies the rate when it is omitted."""
    sampling_frequency = 64  # power of two: 1/fs is exact, so inference round-trips
    raw = np.random.default_rng(30).standard_normal((256, 3))
    time = 5.0 + np.arange(raw.shape[0]) / sampling_frequency
    data = xr.DataArray(
        raw,
        dims=("time", "channel"),
        coords={"time": time, "channel": ["a", "b", "c"]},
    )

    inferred = multitaper_connectivity(data, method="magnitude_squared_coherence")
    explicit = multitaper_connectivity(
        data, sampling_frequency=sampling_frequency, method="magnitude_squared_coherence"
    )

    xr.testing.assert_identical(inferred, explicit)
    # The frequency grid reflects the inferred rate (Nyquist = fs / 2).
    assert float(inferred.frequency.max()) == pytest.approx(sampling_frequency / 2)


def _recording(time, **coordinate_attrs):
    """Two channels of noise on a ``time`` coordinate with the given attrs."""
    raw = np.random.default_rng(38).standard_normal((time.size, 2))
    data = xr.DataArray(raw, dims=("time", "channel"), coords={"time": time})
    data.time.attrs.update(coordinate_attrs)
    return data


def test_inferred_rate_warns_for_a_millisecond_time_coordinate():
    """A coordinate in milliseconds is read as seconds, giving a rate 1000 times
    too low: inference warns, naming the rate and the conversion, when the rate
    is at most 1 Hz or the coordinate's units say milliseconds."""
    at_500_hz = _recording(np.arange(1000) / 500 * 1000)  # ms, no units
    with pytest.warns(UserWarning, match=r"inferred as 0\.5 Hz") as record:
        result = multitaper_connectivity(at_500_hz, method="power")
    message = str(record[0].message)
    assert "elapsed seconds" in message
    assert "fMRI" in message  # no units attribute, so the rate may be genuine
    conversion = "da = da.assign_coords({'time': (da['time'] / 1000).assign_attrs(units='s')})"
    assert conversion in message
    assert "sampling_frequency=0.5 " in message
    assert result.attrs["sampling_frequency"] == pytest.approx(0.5)

    # 1 kHz on a millisecond axis, the commonest case, steps by exactly 1.
    with pytest.warns(UserWarning, match=r"inferred as 1 Hz"):
        multitaper_connectivity(_recording(np.arange(1000.0)), method="power")

    # Above 1 Hz, the units attribute is what gives a millisecond axis away,
    # and the message leaves out the slow-recording aside.
    at_2_khz = _recording(np.arange(1000) / 2000 * 1000, units="ms")
    with pytest.warns(
        UserWarning, match=r"inferred as 2 Hz.*units attribute is 'ms'"
    ) as record:
        multitaper_connectivity(at_2_khz, method="power")
    assert "fMRI" not in str(record[0].message)
    namespace = {"da": at_2_khz}
    exec(conversion, namespace)  # the message's own fix
    assert multitaper_connectivity(namespace["da"], method="power").attrs[
        "sampling_frequency"
    ] == pytest.approx(2000)


def test_spacing_mismatch_asks_whether_the_coordinate_is_in_milliseconds():
    """With the rate given, a millisecond coordinate fails the spacing check by a
    factor of 1000; the error says so."""
    data = _recording(np.arange(1000) / 500 * 1000)
    with pytest.raises(ValueError, match=r"spacing does not match.*in milliseconds"):
        multitaper_connectivity(data, sampling_frequency=500, method="power")


def test_inferred_slow_rate_is_silent_when_confirmed():
    """Slow data (fMRI at TR = 2 s is 0.5 Hz) is legitimate: passing the rate,
    or stating seconds as the coordinate's units, silences the warning."""
    fmri = _recording(np.arange(200) * 2.0)
    with pytest.warns(UserWarning, match=r"inferred as 0\.5 Hz"):
        multitaper_connectivity(fmri, method="power")
    explicit = multitaper_connectivity(fmri, sampling_frequency=0.5, method="power")
    stated = multitaper_connectivity(
        _recording(np.arange(200) * 2.0, units="s"), method="power"
    )
    assert explicit.attrs["sampling_frequency"] == 0.5
    assert stated.attrs["sampling_frequency"] == 0.5


def test_dataarray_infers_rate_from_precise_float32_time_coordinate():
    """Float32 can infer a rate when a zero-based axis retains enough precision."""
    sampling_frequency = 1000
    raw = np.random.default_rng(35).standard_normal((256, 2))
    time = (np.arange(raw.shape[0]) / sampling_frequency).astype(np.float32)
    data = xr.DataArray(raw, dims=("time", "channel"), coords={"time": time})

    result = multitaper_connectivity(data, method="power")

    assert result.attrs["mt_sampling_frequency"] == pytest.approx(sampling_frequency, rel=1e-6)


def test_dataarray_refuses_precision_limited_rate_inference():
    """A quantized large-offset axis must not silently turn 1 kHz into 1024 Hz."""
    sampling_frequency = 1000
    raw = np.random.default_rng(36).standard_normal((16, 2))
    time = (10_000.0 + np.arange(raw.shape[0]) / sampling_frequency).astype(np.float32)
    data = xr.DataArray(raw, dims=("time", "channel"), coords={"time": time})

    with pytest.raises(ValueError, match="Cannot reliably infer sampling_frequency"):
        multitaper_connectivity(data, method="power", time_halfbandwidth_product=2)

    # The same quantized coordinate remains usable when the caller supplies the
    # rate; its dtype-aware validation already permits the representational error.
    explicit = multitaper_connectivity(
        data,
        sampling_frequency=sampling_frequency,
        method="power",
        time_halfbandwidth_product=2,
    )
    assert explicit.attrs["mt_sampling_frequency"] == sampling_frequency
    assert float(explicit.frequency.max()) == pytest.approx(sampling_frequency / 2)


def test_dataarray_rejects_nonfinite_inferred_sampling_frequency():
    """Subnormal time steps fail clearly instead of reaching transform division."""
    raw = np.random.default_rng(37).standard_normal((16, 2))
    time = np.arange(raw.shape[0]) * 1e-310
    data = xr.DataArray(raw, dims=("time", "channel"), coords={"time": time})

    with pytest.raises(ValueError, match="non-finite sampling rate"):
        multitaper_connectivity(data, method="power")


def test_array_without_sampling_frequency_is_rejected():
    """A NumPy input cannot infer a rate and must be given one."""
    with pytest.raises(ValueError, match="sampling_frequency is required"):
        multitaper_connectivity(
            np.random.default_rng(31).standard_normal((256, 2)),
            method="magnitude_squared_coherence",
        )


def test_dataarray_without_time_coordinate_requires_sampling_frequency():
    """A DataArray with no numeric time coordinate cannot infer a rate."""
    data = xr.DataArray(
        np.random.default_rng(32).standard_normal((256, 2)),
        dims=("time", "channel"),
        coords={"channel": ["left", "right"]},
    )
    with pytest.raises(ValueError, match="sampling_frequency is required"):
        multitaper_connectivity(data, method="magnitude_squared_coherence")


def test_dataarray_sample_coordinate_cannot_infer_sampling_frequency():
    """Integer sample numbers carry no time scale, so inference is refused."""
    data = xr.DataArray(
        np.random.default_rng(33).standard_normal((256, 2)),
        dims=("sample", "channel"),
        coords={"sample": np.arange(256)},
    )
    with pytest.raises(ValueError, match="Cannot infer sampling_frequency"):
        multitaper_connectivity(data, method="magnitude_squared_coherence")


def test_dataarray_nonuniform_time_coordinate_cannot_infer_sampling_frequency():
    """A non-uniform time axis has no single rate to infer."""
    raw = np.random.default_rng(34).standard_normal((128, 2))
    times = np.arange(raw.shape[0]) / 64.0
    times[50:] += 0.5  # break regular spacing
    data = xr.DataArray(raw, dims=("time", "channel"), coords={"time": times})
    with pytest.raises(ValueError, match="not uniformly spaced"):
        multitaper_connectivity(data, method="magnitude_squared_coherence")


def test_dataarray_accumulated_time_coordinate_is_accepted_as_uniform():
    """A time axis built by accumulating the sampling interval (a common
    acquisition pattern) carries float round-off of order 1e-10 s over 200 s,
    far below any real irregularity. It must infer the rate and agree with the
    same rate given explicitly."""
    n_samples = 200_000
    times = np.cumsum(np.full(n_samples, 0.001))
    data = xr.DataArray(
        np.random.default_rng(35).standard_normal((n_samples, 2)),
        dims=("time", "channel"),
        coords={"time": times},
    )
    inferred = multitaper_connectivity(data, method="power", time_window_duration=1.0)
    explicit = multitaper_connectivity(
        data, sampling_frequency=1000, method="power", time_window_duration=1.0
    )
    assert inferred.attrs["mt_sampling_frequency"] == pytest.approx(1000.0, rel=1e-6)
    xr.testing.assert_allclose(inferred, explicit)


def test_dataarray_time_coordinate_with_a_dropped_sample_is_rejected():
    """One missing sample shifts every later time by a full interval, which the
    round-off tolerance must still catch, with and without an explicit rate."""
    raw = np.random.default_rng(36).standard_normal((1000, 2))
    times = np.delete(np.arange(1001) / 1000.0, 500)
    data = xr.DataArray(raw, dims=("time", "channel"), coords={"time": times})
    with pytest.raises(ValueError, match="not uniformly spaced"):
        multitaper_connectivity(data, method="power")
    with pytest.raises(ValueError, match="spacing does not match"):
        multitaper_connectivity(data, sampling_frequency=1000, method="power")


def _time_only_dataarray(times: np.ndarray) -> xr.DataArray:
    """A 1-D DataArray carrying only a ``time`` coordinate, for axis validation."""
    return xr.DataArray(np.zeros(times.size), dims=("time",), coords={"time": times})


@pytest.mark.parametrize(
    ("sampling_frequency", "n_samples"),
    [(1000.0, 200_000), (30_000.0, 300_000), (2000.0, 600_000), (1000.0, 3_600_000)],
    ids=["1kHz-200s", "30kHz-10s", "2kHz-5min", "1kHz-1h"],
)
@pytest.mark.parametrize("start_time", [0.0, 1000.0])
@pytest.mark.parametrize("construction", ["cumsum", "linspace"])
def test_dataarray_float64_time_axes_are_uniform_at_recording_lengths(
    sampling_frequency, n_samples, start_time, construction
):
    """Accumulated round-off grows along a ``cumsum`` axis (to ~1e-4 intervals
    after an hour at 1 kHz) while every individual step stays exact to ~1e-10
    intervals, so realistic float64 axes must be accepted with and without an
    explicit rate, and the inferred rate must stay accurate."""
    interval = 1.0 / sampling_frequency
    if construction == "cumsum":
        times = np.cumsum(np.r_[start_time, np.full(n_samples - 1, interval)])
    else:
        times = np.linspace(start_time, start_time + (n_samples - 1) * interval, n_samples)
    data = _time_only_dataarray(times)

    inferred = _time_axis_from_dataarray(data, "time", None)
    explicit = _time_axis_from_dataarray(data, "time", sampling_frequency)

    assert inferred.inferred_sampling_frequency == pytest.approx(sampling_frequency, rel=1e-8)
    assert inferred.start_time == explicit.start_time == times[0]
    assert explicit.inferred_sampling_frequency is None


@pytest.mark.parametrize("sampling_frequency", [30_000.0, 1000.0])
def test_dataarray_time_axis_accepts_timestamp_jitter(sampling_frequency):
    """Hardware timestamps jitter by a small fraction of an interval (1e-6 s is
    0.03 intervals at 30 kHz); up to 0.05 intervals per sample is still one
    sample per step and must be accepted with and without an explicit rate."""
    n_samples = 300_000
    rng = np.random.default_rng(38)
    jitter = rng.uniform(-0.05, 0.05, n_samples) / sampling_frequency
    times = np.arange(n_samples) / sampling_frequency + jitter
    data = _time_only_dataarray(times)

    inferred = _time_axis_from_dataarray(data, "time", None)
    _time_axis_from_dataarray(data, "time", sampling_frequency)

    assert inferred.inferred_sampling_frequency == pytest.approx(sampling_frequency, rel=1e-6)


@pytest.mark.parametrize("n_samples", [1000, 3_600_000], ids=["1s", "1h"])
@pytest.mark.parametrize("position", ["start", "middle", "end"])
def test_dataarray_time_axis_rejects_one_dropped_sample_anywhere(n_samples, position):
    """A single missing sample is a two-interval step wherever it falls, even on
    a long ``cumsum`` axis whose accumulated round-off is otherwise tolerated."""
    times = np.cumsum(np.full(n_samples + 1, 0.001))
    dropped_index = {"start": 1, "middle": n_samples // 2, "end": n_samples - 1}[position]
    data = _time_only_dataarray(np.delete(times, dropped_index))

    with pytest.raises(ValueError, match="not uniformly spaced"):
        _time_axis_from_dataarray(data, "time", None)
    with pytest.raises(ValueError, match="spacing does not match"):
        _time_axis_from_dataarray(data, "time", 1000.0)


@pytest.mark.parametrize("position", ["start", "middle", "end"])
def test_dataarray_time_axis_rejects_one_duplicated_timestamp_anywhere(position):
    times = np.arange(1000) / 1000.0
    duplicated_index = {"start": 0, "middle": 500, "end": 999}[position]
    data = _time_only_dataarray(np.insert(times, duplicated_index, times[duplicated_index]))

    with pytest.raises(ValueError, match="strictly increasing"):
        _time_axis_from_dataarray(data, "time", None)
    with pytest.raises(ValueError, match="strictly increasing"):
        _time_axis_from_dataarray(data, "time", 1000.0)


def test_dataarray_time_axis_rejects_rate_errors_below_one_step():
    """Every step of a 1024 Hz axis is within 2.4% of a 1 kHz interval, but the
    axis ends 23 samples away from the explicit 1 kHz grid; likewise an axis
    whose rate changes halfway is far from any single regular grid."""
    rate_mismatch = _time_only_dataarray(np.arange(1000) / 1024.0)
    with pytest.raises(ValueError, match="spacing does not match"):
        _time_axis_from_dataarray(rate_mismatch, "time", 1000.0)

    first_half = np.arange(500) / 1000.0
    rate_change = _time_only_dataarray(
        np.r_[first_half, first_half[-1] + np.arange(1, 501) / 1100.0]
    )
    with pytest.raises(ValueError, match="not uniformly spaced"):
        _time_axis_from_dataarray(rate_change, "time", None)


def test_dataarray_millisecond_time_coordinate_mismatches_explicit_rate():
    """A coordinate in milliseconds is 1000x off a 1 kHz rate in seconds."""
    raw = np.random.default_rng(37).standard_normal((1000, 2))
    data = xr.DataArray(raw, dims=("time", "channel"), coords={"time": np.arange(1000.0)})
    with pytest.raises(ValueError, match="spacing does not match"):
        multitaper_connectivity(data, sampling_frequency=1000, method="power")


def test_dataarray_role_absent_at_dimensionality_is_rejected_with_reshape_hint():
    """A role with no slot at this ndim (trial in 2-D) points at the shape, not transpose."""
    data = xr.DataArray(
        np.random.default_rng(13).standard_normal((256, 4)),
        dims=("time", "trial"),
    )
    with pytest.raises(ValueError, match="has no trial axis"):
        multitaper_connectivity(
            data, sampling_frequency=256, method="magnitude_squared_coherence"
        )


class _DaskProtocolArray:
    """Minimal NEP-18 duck array exposing the dask collection protocol.

    xarray keeps it as the DataArray's backing array (as it does a real dask
    array), and like dask it is indexable and converts implicitly through
    ``__array__``, so a missing guard would compute silently instead of
    raising. dask itself is not a test dependency.
    """

    def __init__(self, values: np.ndarray) -> None:
        self._values = values
        self.shape = values.shape
        self.dtype = values.dtype
        self.ndim = values.ndim

    def __dask_graph__(self) -> dict:
        return {}

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        return self._values

    def __getitem__(self, key) -> np.ndarray:
        return self._values[key]

    def __array_function__(self, func, types, args, kwargs):
        return NotImplemented

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        return NotImplemented


def test_dataarray_dask_backing_is_rejected():
    """A dask-backed DataArray raises with a materialization hint."""
    data = xr.DataArray(
        _DaskProtocolArray(np.random.default_rng(12).standard_normal((256, 2))),
        dims=("sample", "channel"),
    )
    assert isinstance(data.data, _DaskProtocolArray)  # premise: backing kept lazy
    with pytest.raises(TypeError, match="multitaper_connectivity received a dask-backed"):
        multitaper_connectivity(
            data, sampling_frequency=256, method="magnitude_squared_coherence"
        )


def test_fourier_dataarray_dask_backing_is_rejected_by_name():
    """The error names the function the user called."""
    rng = np.random.default_rng(13)
    coefficients = rng.standard_normal((3, 8, 2)) + 1j * rng.standard_normal((3, 8, 2))
    data = xr.DataArray(
        _DaskProtocolArray(coefficients), dims=("trial", "frequency", "channel")
    )
    with pytest.raises(TypeError, match="fourier_connectivity received a dask-backed"):
        fourier_connectivity(data, method="magnitude_squared_coherence")


def test_dask_protocol_backing_is_rejected_without_optional_dependency():
    """Dask detection uses its collection protocol, not a module-name heuristic."""

    class LazyArray:
        def __dask_graph__(self):
            return {}

    with pytest.raises(TypeError, match="dask-backed"):
        _reject_unmaterialized_backing(LazyArray(), "multitaper_connectivity")


def test_dataarray_input_attrs_are_carried_into_provenance(tmp_path):
    """A DataArray's own attrs survive in one canonical provenance record."""
    input_attrs = {"subject": "m1", "session": 7, "montage": [1, 2, 3]}
    data = xr.DataArray(
        np.random.default_rng(20).standard_normal((256, 2)),
        dims=("time", "channel"),
        coords={"channel": ["left", "right"]},
        attrs=input_attrs,
    )

    # Single-method DataArray result.
    result = multitaper_connectivity(
        data, sampling_frequency=256, method="magnitude_squared_coherence"
    )
    assert json.loads(result.attrs["input_attrs_json"]) == input_attrs
    # Namespacing keeps caller metadata from clobbering our own provenance.
    assert result.attrs["package"] == "spectral_connectivity"

    # Multi-method Dataset result: carried on the Dataset and each variable.
    ds = multitaper_connectivity(
        data,
        sampling_frequency=256,
        method=["magnitude_squared_coherence", "imaginary_coherence"],
    )
    assert json.loads(ds.attrs["input_attrs_json"]) == input_attrs
    assert (
        json.loads(ds["magnitude_squared_coherence"].attrs["input_attrs_json"]) == input_attrs
    )

    # Provenance must remain NetCDF-serializable.
    ds.to_netcdf(tmp_path / "input_attrs.nc")


def test_dataarray_input_attrs_cannot_collide_or_break_netcdf(tmp_path):
    """Arbitrary attr keys remain distinct inside the fixed JSON record."""
    input_attrs = {
        1: "integer key",
        "1": "string key",
        "x": [1, 2],
        "x_json": "literal suffix",
        "subject/id": "m1",
    }
    data = xr.DataArray(
        np.random.default_rng(22).standard_normal((256, 2)),
        dims=("time", "channel"),
        attrs=input_attrs,
    )

    result = multitaper_connectivity(data, sampling_frequency=256, method="power")
    assert result.attrs["input_attrs_json"] == _canonical_json(input_attrs)
    assert {key for key in result.attrs if key.startswith("input_")} == {"input_attrs_json"}

    path = tmp_path / "arbitrary_input_attrs.nc"
    result.to_netcdf(path)
    reloaded = xr.open_dataarray(path)
    try:
        assert reloaded.attrs["input_attrs_json"] == _canonical_json(input_attrs)
    finally:
        reloaded.close()


def test_plain_ndarray_input_has_no_input_namespace():
    """A NumPy input contributes no ``input_*`` attributes."""
    result = multitaper_connectivity(
        np.random.default_rng(21).standard_normal((256, 2)),
        sampling_frequency=256,
        method="magnitude_squared_coherence",
    )
    assert not any(key.startswith("input_") for key in result.attrs)


class TestProvenanceSerialization:
    """Direct coverage of the provenance-serialization helpers.

    Every branch exists to keep ``to_netcdf`` from breaking on unusual measure
    kwargs; each is exercised here across the value taxonomy rather than only
    incidentally through a measure that happens to pass such a value.
    """

    def test_scalar_and_none_passthrough(self):
        assert _json_compatible(None) is None
        assert _json_compatible(True) is True
        assert _json_compatible(3) == 3
        assert _json_compatible(2.5) == 2.5
        assert _json_compatible("x") == "x"
        assert _canonical_json(None) == "null"

    def test_nonfinite_float_becomes_marker(self):
        assert _json_compatible(float("nan")) == {"nonfinite_float": "nan"}
        assert _json_compatible(float("inf")) == {"nonfinite_float": "inf"}
        # measure_kwargs_json must not emit a bare NaN token.
        assert "NaN" not in _canonical_json({"x": float("nan")})

    def test_numpy_scalars_and_arrays_are_plain_python(self):
        assert _json_compatible(np.float64(0.5)) == 0.5
        assert _json_compatible(np.int64(7)) == 7
        assert _json_compatible(np.array([1, 2, 3])) == [1, 2, 3]
        # No numpy reprs leak into the JSON string.
        encoded = _canonical_json({"w": np.array([1.0, 2.0])})
        assert "float64" not in encoded
        assert "array" not in encoded

    def test_nested_mapping_is_sorted_and_deterministic(self):
        value = {"b": 1, "a": {"d": 2, "c": 3}}
        assert _canonical_json(value) == '{"a":{"c":3,"d":2},"b":1}'

    def test_non_string_mapping_keys_cannot_collide(self):
        encoded = _canonical_json({1: "integer", "1": "string"})
        assert json.loads(encoded) == {
            "python_type": "mapping",
            "items": [["1", "string"], [1, "integer"]],
        }

    def test_arbitrary_object_falls_back_to_type_and_repr(self):
        compatible = _json_compatible({1, 2})  # sets are not JSON-native
        assert set(compatible) == {"python_type", "repr"}
        assert compatible["python_type"] == "builtins.set"
        # The fallback must still produce valid JSON.
        json.loads(_canonical_json({1, 2}))

    def test_netcdf_provenance_value_routes_structured_and_nonfinite_to_json(self):
        assert _netcdf_provenance_value(0.5) == 0.5
        assert _netcdf_provenance_value("s") == "s"
        assert json.loads(_netcdf_provenance_value([1, 2])) == [1, 2]
        assert json.loads(_netcdf_provenance_value(float("nan"))) == {"nonfinite_float": "nan"}


@pytest.fixture
def stub_measure(monkeypatch):
    """Register a zero-valued pairwise extension measure that accepts any kwargs.

    None of the registered measures accept arbitrary keyword arguments, so this
    stub (named ``stub_measure``) exercises kwarg provenance recording.
    """

    def stub_measure(connectivity, **kwargs):
        return np.zeros(
            (
                len(connectivity.time),
                len(connectivity.frequencies),
                connectivity.n_signals,
                connectivity.n_signals,
            )
        )

    monkeypatch.setattr(Connectivity, "stub_measure", stub_measure, raising=False)
    return "stub_measure"


def test_structured_and_nonfinite_kwargs_survive_netcdf(tmp_path, stub_measure):
    """Unusual measure kwargs serialize and round-trip through NetCDF."""
    rng = np.random.default_rng(13)
    m = Multitaper(rng.standard_normal((256, 4, 3)), sampling_frequency=500)

    da = connectivity_to_xarray(
        m,
        method=stub_measure,
        connectivity_kwargs={
            "nested": {"b": 1, "a": 2},
            "weights": np.array([1.0, 2.0]),
            "cutoff": float("inf"),
        },
    )
    assert json.loads(da.attrs["arg_nested_json"]) == {"a": 2, "b": 1}
    assert json.loads(da.attrs["arg_weights_json"]) == [1.0, 2.0]
    assert json.loads(da.attrs["arg_cutoff_json"]) == {"nonfinite_float": "inf"}

    path = tmp_path / "structured.nc"
    da.to_netcdf(path)
    reloaded = xr.open_dataarray(path)
    try:
        assert json.loads(reloaded.attrs["measure_kwargs_json"]) == {
            "nested": {"a": 2, "b": 1},
            "weights": [1.0, 2.0],
            "cutoff": {"nonfinite_float": "inf"},
        }
    finally:
        reloaded.close()


def test_result_netcdf_serializable_with_detrend_none(tmp_path):
    """to_netcdf must work even when a Multitaper option is None (detrend_type)."""
    rng = np.random.default_rng(0)
    time_series = rng.standard_normal((512, 3, 2))
    result = multitaper_connectivity(
        time_series,
        sampling_frequency=500,
        method="magnitude_squared_coherence",
        detrend_type=None,
    )
    # None is encoded as a string so the attribute is still recorded.
    assert result.attrs["mt_detrend_type"] == "None"
    path = tmp_path / "conn.nc"
    result.to_netcdf(path)
    assert path.exists()


def test_fft_workers_does_not_change_results():
    """The `fft_workers` FFT-parallelism option must not change the output.

    `fft_workers` only sets SciPy's CPU FFT thread count. A threaded FFT is not
    guaranteed bit-for-bit identical to the single-threaded one (summation order
    can differ), so the results are compared with a tight tolerance rather than
    exact equality. The wrapper forwards the argument to `Multitaper` via
    **kwargs.
    """
    from spectral_connectivity.transforms import Multitaper

    rng = np.random.default_rng(0)
    time_series = rng.standard_normal((1024, 6, 3))

    reference = Multitaper(time_series, sampling_frequency=500).fft()
    for workers in (1, 2, -1):
        result = Multitaper(time_series, sampling_frequency=500, fft_workers=workers).fft()
        np.testing.assert_allclose(to_host(result), to_host(reference), rtol=1e-10, atol=1e-12)

    # The wrapper forwards fft_workers via **kwargs; results are equivalent.
    baseline = multitaper_connectivity(
        time_series, sampling_frequency=500, method="magnitude_squared_coherence"
    )
    parallel = multitaper_connectivity(
        time_series,
        sampling_frequency=500,
        method="magnitude_squared_coherence",
        transform_kwargs={"fft_workers": -1},
    )
    np.testing.assert_allclose(baseline.values, parallel.values, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("bad", [0, 1.5, "4", True, np.float64(2.0)])
def test_fft_workers_invalid_values_raise_named_error(bad):
    """Invalid `fft_workers` must fail with a message naming the parameter.

    Forwarding a bad value straight to ``scipy.fft.fft(workers=...)`` surfaces an
    opaque error (``0`` -> "workers must not be zero"; ``"4"`` -> a bare
    ``TypeError``) that never mentions ``fft_workers``. Validate at construction
    so the user gets an actionable message, mirroring ``max_workspace_elements``.
    ``True`` is rejected because ``bool`` is an ``int`` subclass but not a
    meaningful thread count.
    """
    from spectral_connectivity.transforms import Multitaper

    rng = np.random.default_rng(0)
    time_series = rng.standard_normal((64, 2, 2))
    with pytest.raises(ValueError, match="fft_workers"):
        Multitaper(time_series, sampling_frequency=500, fft_workers=bad)


@pytest.mark.parametrize("good", [None, 1, 2, -1, np.int64(3)])
def test_fft_workers_valid_values_accepted(good):
    """None and any nonzero integer thread count are accepted."""
    from spectral_connectivity.transforms import Multitaper

    rng = np.random.default_rng(0)
    time_series = rng.standard_normal((64, 2, 2))
    mt = Multitaper(time_series, sampling_frequency=500, fft_workers=good)
    assert mt.fft_workers == good


@pytest.mark.cpu_only(
    reason="spies on scipy.fft workers forwarding; the CuPy FFT takes no workers argument"
)
def test_fft_workers_is_actually_forwarded_to_scipy():
    """`fft_workers` must reach SciPy's FFT (and only on the CPU backend).

    Output invariance alone cannot detect a dropped passthrough. Spy on the
    module-level ``fft`` to confirm ``workers`` is forwarded when set, omitted
    when ``None`` (SciPy's default), forwarded through the wrapper's **kwargs,
    and NOT forwarded when the GPU backend is (simulated as) active.
    """
    from unittest.mock import patch

    from spectral_connectivity import transforms
    from spectral_connectivity.transforms import Multitaper

    rng = np.random.default_rng(0)
    time_series = rng.standard_normal((256, 3, 2))
    real_fft = transforms.fft

    def spying_fft(recorded, real_fft=real_fft):
        def _fft(*args, **kwargs):
            recorded.append(kwargs.get("workers", "MISSING"))
            return real_fft(*args, **kwargs)

        return _fft

    def transform_workers(multitaper):
        # Prime the tapers first (their DPSS eigenvalue FFT also uses this
        # module's `fft`), so the spy records only the taper-projection FFT.
        _ = multitaper.tapers  # prime the DPSS fft
        recorded = []
        with patch.object(transforms, "fft", spying_fft(recorded)):
            multitaper.fft()
        assert len(recorded) == 1
        return recorded[0]

    # Default: no `workers` key is passed (SciPy's single-threaded default).
    assert transform_workers(Multitaper(time_series, sampling_frequency=500)) == ("MISSING")

    # Explicit value is forwarded verbatim.
    assert (
        transform_workers(Multitaper(time_series, sampling_frequency=500, fft_workers=3)) == 3
    )

    # Forwarded through the wrapper's **kwargs (which reach Multitaper).
    recorded = []
    with patch.object(transforms, "fft", spying_fft(recorded)):
        multitaper_connectivity(
            time_series,
            sampling_frequency=500,
            method="magnitude_squared_coherence",
            transform_kwargs={"fft_workers": 2},
        )
    assert 2 in recorded  # the taper-projection FFT received workers=2

    # On the GPU backend `workers` is not forwarded (cupyx's FFT has no such
    # parameter). Simulate GPU on the CPU by patching the imported backend. The
    # GPU transforms real input with rfft (see _multitaper_fft), so spy on both.
    gpu_multitaper = Multitaper(time_series, sampling_frequency=500, fft_workers=-1)
    _ = gpu_multitaper.tapers
    recorded = []
    with (
        patch.object(transforms, "ON_GPU", True),
        patch.object(transforms, "fft", spying_fft(recorded)),
        patch.object(transforms, "rfft", spying_fft(recorded, transforms.rfft)),
    ):
        gpu_multitaper.fft()
    assert recorded == ["MISSING"]

    # The backend imported at startup decides, not the environment variable,
    # which may change afterwards (e.g. %env in a notebook).
    recorded = []
    with (
        patch.dict("os.environ", {"SPECTRAL_CONNECTIVITY_ENABLE_GPU": "true"}),
        patch.object(transforms, "ON_GPU", False),
        patch.object(transforms, "fft", spying_fft(recorded)),
    ):
        gpu_multitaper.fft()
    assert recorded == [-1]


def test_to_numpy_handles_device_arrays():
    """The shared backend boundary handles explicit device-to-host transfer."""
    from spectral_connectivity.utils import to_numpy

    host = np.arange(5.0)
    np.testing.assert_array_equal(to_numpy(host), host)

    class _DeviceLike:
        """Mimics cupy.ndarray: no implicit conversion, but ``.get()`` works."""

        def __init__(self, host_array):
            self._host = host_array

        def get(self):
            return self._host

        def __array__(self, dtype=None):
            msg = "Implicit conversion to a NumPy array is not allowed."
            raise TypeError(msg)

        @property
        def __cuda_array_interface__(self):
            # Like cupy.ndarray: marks this as a device array.
            return {"version": 3}

    device = _DeviceLike(np.arange(5.0))
    with pytest.raises(TypeError):
        np.asarray(device)  # guards the premise: implicit conversion fails
    np.testing.assert_array_equal(to_numpy(device), np.arange(5.0))


def test_connectivity_to_xarray_accepts_device_backed_validity_mask():
    """``valid_time_frequency`` may live on the device (CuPy); the wrapper must
    transfer it explicitly rather than rely on implicit ``np.asarray``."""

    class _DeviceLike:
        def __init__(self, host_array):
            self._host = host_array

        def get(self):
            return self._host

        def __array__(self, dtype=None, copy=None):
            msg = "Implicit conversion to a NumPy array is not allowed."
            raise TypeError(msg)

        @property
        def __cuda_array_interface__(self):
            # Like cupy.ndarray: marks this as a device array.
            return {"version": 3}

    rng = np.random.default_rng(11)
    transform = MorletWavelet(
        rng.standard_normal((512, 2, 2)),
        sampling_frequency=128,
        frequencies=np.array([4.0, 8.0, 16.0]),
        edge_mode="nan",
    )
    host_mask = to_host(transform.valid_time_frequency)

    class DeviceMaskTransform:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        @property
        def valid_time_frequency(self):
            return _DeviceLike(host_mask)

    result = connectivity_to_xarray(
        DeviceMaskTransform(transform), method="magnitude_squared_coherence"
    )
    np.testing.assert_array_equal(result["valid_time_frequency"].values, host_mask)


@pytest.mark.parametrize(
    "name",
    [*sorted(_NON_MEASURE_METHODS), "from_transform", "_expectation_cross_spectral_matrix"],
)
def test_non_measure_callables_are_rejected_as_unknown_measures(name):
    rng = np.random.default_rng(12)
    with pytest.raises(ValueError, match="not a known connectivity measure"):
        multitaper_connectivity(
            rng.standard_normal((256, 2, 3)), sampling_frequency=250, method=name
        )


def test_batch_kwargs_not_accepted_by_a_method_raise_actionable_error():
    rng = np.random.default_rng(13)
    with pytest.raises(TypeError, match="passed to every requested method"):
        multitaper_connectivity(
            rng.standard_normal((256, 2, 3)),
            sampling_frequency=250,
            method=["magnitude_squared_coherence", "phase_slope_index"],
            connectivity_kwargs={"frequency_resolution": 5.0},
        )


def test_transform_setting_in_connectivity_kwargs_points_to_multitaper_connectivity():
    rng = np.random.default_rng(13)
    with pytest.raises(
        TypeError, match="connectivity_kwargs configures the measure only"
    ) as excinfo:
        multitaper_connectivity(
            rng.standard_normal((256, 2, 3)),
            sampling_frequency=250,
            method="magnitude_squared_coherence",
            connectivity_kwargs={"time_halfbandwidth_product": 4},
        )
    message = str(excinfo.value)
    assert "pass it to multitaper_connectivity: by name" in message
    assert "time_halfbandwidth_product" in message
    assert "otherwise in transform_kwargs" in message
    assert "fourier_coefficients" not in message


def test_transform_setting_in_connectivity_kwargs_points_to_the_fourier_transform(
    four_signal_coefficients,
):
    coefficients, frequencies, time = four_signal_coefficients
    with pytest.raises(
        TypeError, match="connectivity_kwargs configures the measure only"
    ) as excinfo:
        fourier_connectivity(
            coefficients,
            frequencies=frequencies,
            time=time,
            method="magnitude_squared_coherence",
            connectivity_kwargs={"time_halfbandwidth_product": 4},
        )
    message = str(excinfo.value)
    assert "set it on the transform that produced fourier_coefficients" in message
    assert "multitaper_connectivity" not in message


def test_connectivity_to_xarray_points_transform_settings_to_the_transform():
    """Its caller built the transform, so a transform setting goes there."""
    transform = Multitaper(np.random.default_rng(13).standard_normal((256, 2, 3)), 250)
    with pytest.raises(TypeError, match="does not accept keyword argument") as excinfo:
        connectivity_to_xarray(
            transform,
            "magnitude_squared_coherence",
            connectivity_kwargs={"time_halfbandwidth_product": 4},
        )
    message = str(excinfo.value)
    assert (
        "If 'time_halfbandwidth_product' is a transform setting, set it on the "
        "transform you pass as m (for example Multitaper(..., "
        "time_halfbandwidth_product=3))." in message
    )
    assert "multitaper_connectivity" not in message


@pytest.fixture(scope="module")
def four_signal_noise():
    """Seeded white noise, shape (1000, 4, 4): four signals, so two groups of two."""
    return np.random.default_rng(0).standard_normal((1000, 4, 4))


@pytest.fixture(scope="module")
def four_signal_coefficients(four_signal_noise):
    """``(fft, frequencies, time)`` of ``four_signal_noise`` for fourier_connectivity."""
    transform = Multitaper(four_signal_noise, sampling_frequency=1000)
    return transform.fft(), transform.frequencies, transform.time


def test_is_group_measure_matches_the_measures_that_take_group_labels():
    """The signature rule agrees with the registry's group output kinds."""
    registered_group = {
        name
        for name, spec in _MEASURE_SPECS.items()
        if spec.output_kind in {"group_pairwise", "multivariate_components"}
    }
    assert registered_group
    assert {name for name in _MEASURE_SPECS if _is_group_measure(name)} == registered_group


@pytest.fixture
def extension_group_measure(monkeypatch):
    """An unregistered ``Connectivity`` extension that takes ``group_labels``.

    Returns the list of labels each call received.
    """
    received = []

    def custom_group_power(self, group_labels=None):
        received.append(group_labels)
        return np.zeros(
            (len(self.time), len(self.frequencies), self.n_signals, self.n_signals)
        )

    monkeypatch.setattr(Connectivity, "custom_group_power", custom_group_power, raising=False)
    return received


@pytest.mark.parametrize(
    ("method", "label_kwargs"),
    [
        ("custom_group_power", {"group_labels": [0, 0, 1, 1]}),
        ("custom_group_power", {"connectivity_kwargs": {"group_labels": [0, 0, 1, 1]}}),
        (["canonical_coherence", "custom_group_power"], {"group_labels": [0, 0, 1, 1]}),
    ],
    ids=["named_argument", "dict_form", "mixed_batch"],
)
def test_extension_measure_taking_group_labels_receives_them(
    four_signal_noise, extension_group_measure, method, label_kwargs
):
    multitaper_connectivity(
        four_signal_noise, sampling_frequency=1000, method=method, **label_kwargs
    )
    assert extension_group_measure == [[0, 0, 1, 1]]


def test_extension_measure_taking_group_labels_is_named_when_labels_are_missing(
    four_signal_noise, extension_group_measure
):
    with pytest.raises(
        ValueError,
        match="group_labels is required for canonical_coherence, custom_group_power",
    ):
        multitaper_connectivity(
            four_signal_noise,
            sampling_frequency=1000,
            method=["canonical_coherence", "custom_group_power"],
        )
    assert extension_group_measure == []


# One case per group_labels error: (measure/label arguments, message pattern).
_GROUP_LABEL_ERRORS = {
    # The message names the argument to pass, not only the failure.
    "missing": (
        {"method": "canonical_coherence"},
        r"group_labels is required for[\s\S]*group_labels=",
    ),
    "no_group_measure": (
        {"method": "magnitude_squared_coherence", "group_labels": [0, 1]},
        "none of the requested measures compares groups",
    ),
    "given_twice": (
        {
            "method": "canonical_coherence",
            "group_labels": [0, 0, 1, 1],
            "connectivity_kwargs": {"group_labels": [0, 1, 0, 1]},
        },
        "pass it once",
    ),
}


@pytest.mark.parametrize("case", list(_GROUP_LABEL_ERRORS))
def test_group_label_errors_fail_before_the_fft(four_signal_noise, case):
    arguments, pattern = _GROUP_LABEL_ERRORS[case]
    with (
        mock.patch.object(Multitaper, "fft", autospec=True) as fft,
        pytest.raises(ValueError, match=pattern),
    ):
        multitaper_connectivity(four_signal_noise, sampling_frequency=1000, **arguments)
    fft.assert_not_called()


@pytest.mark.parametrize("case", list(_GROUP_LABEL_ERRORS))
def test_fourier_connectivity_group_label_errors(four_signal_coefficients, case):
    coefficients, frequencies, time = four_signal_coefficients
    arguments, pattern = _GROUP_LABEL_ERRORS[case]
    with pytest.raises(ValueError, match=pattern):
        fourier_connectivity(coefficients, frequencies=frequencies, time=time, **arguments)


@pytest.mark.parametrize(
    "label_arguments",
    [
        {"group_labels": [0, 0, 1, 1]},
        {"connectivity_kwargs": {"group_labels": [0, 0, 1, 1]}},
    ],
    ids=["argument", "connectivity_kwargs"],
)
def test_group_labels_reach_every_group_measure_in_a_batch(four_signal_noise, label_arguments):
    result = multitaper_connectivity(
        four_signal_noise,
        sampling_frequency=1000,
        method=["canonical_coherence", "magnitude_squared_coherence"],
        **label_arguments,
    )
    assert isinstance(result, xr.Dataset)
    assert result["canonical_coherence"].sizes["source_group"] == 2
    assert result["magnitude_squared_coherence"].sizes["source"] == 4
    # The labels reached the group measure only.
    assert "arg_group_labels_json" in result["canonical_coherence"].attrs
    assert "arg_group_labels_json" not in result["magnitude_squared_coherence"].attrs


def test_group_labels_recorded_in_provenance(four_signal_noise):
    labels = ["CA1", "CA1", "PFC", "PFC"]
    result = multitaper_connectivity(
        four_signal_noise,
        sampling_frequency=1000,
        method=["canonical_coherence", "magnitude_squared_coherence"],
        group_labels=labels,
    )
    assert json.loads(result["canonical_coherence"].attrs["arg_group_labels_json"]) == labels


def test_connectivity_kwargs_group_labels_still_works(four_signal_noise):
    connectivity_kwargs = {"group_labels": [0, 0, 1, 1]}
    from_dict = multitaper_connectivity(
        four_signal_noise,
        sampling_frequency=1000,
        method="canonical_coherence",
        connectivity_kwargs=connectivity_kwargs,
    )
    from_argument = multitaper_connectivity(
        four_signal_noise,
        sampling_frequency=1000,
        method="canonical_coherence",
        group_labels=[0, 0, 1, 1],
    )
    # Identical including measure_kwargs_json: both forms record the same kwargs.
    xr.testing.assert_identical(from_dict, from_argument)
    assert connectivity_kwargs == {"group_labels": [0, 0, 1, 1]}


def test_group_labels_none_in_connectivity_kwargs_means_no_labels(four_signal_noise):
    """``{"group_labels": None}`` matches the argument's ``None`` default."""
    with_none = multitaper_connectivity(
        four_signal_noise,
        sampling_frequency=1000,
        method="magnitude_squared_coherence",
        connectivity_kwargs={"group_labels": None},
    )
    without = multitaper_connectivity(
        four_signal_noise, sampling_frequency=1000, method="magnitude_squared_coherence"
    )
    xr.testing.assert_identical(with_none, without)
    with pytest.raises(ValueError, match="group_labels is required for"):
        multitaper_connectivity(
            four_signal_noise,
            sampling_frequency=1000,
            method="canonical_coherence",
            connectivity_kwargs={"group_labels": None},
        )


def test_fourier_connectivity_accepts_group_labels(four_signal_coefficients):
    coefficients, frequencies, time = four_signal_coefficients
    result = fourier_connectivity(
        coefficients,
        frequencies=frequencies,
        time=time,
        method="canonical_coherence",
        group_labels=["CA1", "CA1", "PFC", "PFC"],
    )
    assert result.source_group.values.tolist() == ["CA1", "PFC"]


def test_multi_method_shares_single_fft(monkeypatch):
    """A multi-method call computes the FFT once, not once per measure.

    ``multitaper_connectivity`` builds one shared ``Connectivity`` and reuses it
    across every requested measure. Since ``Connectivity.from_transform`` calls
    the (uncached) ``Multitaper.fft``, the FFT must run exactly once regardless
    of how many measures are requested.
    """
    from spectral_connectivity.transforms import Multitaper

    rng = np.random.default_rng(0)
    time_series = rng.standard_normal((512, 4, 3))
    methods = ["magnitude_squared_coherence", "coherence_phase", "imaginary_coherence"]

    original_fft = Multitaper.fft
    calls = {"n": 0}

    def counting_fft(self):
        calls["n"] += 1
        return original_fft(self)

    monkeypatch.setattr(Multitaper, "fft", counting_fft)
    multitaper_connectivity(time_series, sampling_frequency=500, method=methods)

    assert calls["n"] == 1, f"FFT computed {calls['n']} times for {len(methods)} methods"


def test_shared_connectivity_matches_per_method_construction():
    """Sharing one Connectivity yields identical results to building per method.

    Reusing a single instance only avoids recomputation; it must not change any
    numbers. Results must match a fresh ``Connectivity.from_transform`` per
    measure bit-for-bit.
    """
    import xarray as xr

    from spectral_connectivity.transforms import Multitaper
    from spectral_connectivity.wrapper import connectivity_to_xarray

    rng = np.random.default_rng(1)
    time_series = rng.standard_normal((512, 4, 3))
    methods = ["magnitude_squared_coherence", "coherence_phase", "imaginary_coherence"]

    shared = multitaper_connectivity(time_series, sampling_frequency=500, method=methods)

    m = Multitaper(time_series, sampling_frequency=500)
    per_method = xr.Dataset()
    for meth in methods:
        # connectivity=None forces a fresh Connectivity (and FFT) each call.
        per_method[meth] = connectivity_to_xarray(m, meth)

    for meth in methods:
        np.testing.assert_array_equal(
            shared[meth].values, per_method[meth].values, err_msg=meth
        )


def test_default_result_is_netcdf_serializable(tmp_path):
    """The documented default (method=None) result must save to NetCDF.

    The explicit DEFAULT_METHODS allowlist omits complex-valued coherency so
    the default remains portable across all supported xarray versions and
    NetCDF engines.
    """
    rng = np.random.default_rng(0)
    ds = multitaper_connectivity(rng.standard_normal((512, 5, 2)), sampling_frequency=500)
    assert "coherency" not in ds.data_vars
    assert not any(np.iscomplexobj(da.values) for da in ds.data_vars.values())
    path = tmp_path / "default.nc"
    ds.to_netcdf(path)
    assert path.exists()


def test_default_method_set_is_explicit_and_ordered():
    """method=None uses the explicit, ordered DEFAULT_METHODS allowlist.

    The exact tuple (including order) is locked: xarray preserves insertion
    order, so the default Dataset's variable/iteration/serialization order is
    part of the public contract. It keeps the alphabetical order of the 2.x
    allowlist, with magnitude_squared_coherence in the place of its 2.x name.
    """
    from spectral_connectivity.wrapper import DEFAULT_METHODS

    expected = (
        "magnitude_squared_coherence",
        "coherence_phase",
        "debiased_squared_phase_lag_index",
        "debiased_squared_weighted_phase_lag_index",
        "imaginary_coherence",
        "pairwise_phase_consistency",
        "phase_lag_index",
        "phase_locking_value",
        "power",
        "weighted_phase_lag_index",
    )
    assert expected == DEFAULT_METHODS
    # The deliberately excluded measures must not be in the default.
    for excluded in ("coherency", "global_coherence", "phase_slope_index"):
        assert excluded not in DEFAULT_METHODS
    # Every directed measure is opt-in, so no default needs a two-sided
    # spectrum, and fourier_connectivity's default suits one-sided input.
    assert not {measure.name for measure in list_measures(directed=True)} & set(
        DEFAULT_METHODS
    )
    assert not any(_requires_two_sided(name) for name in DEFAULT_METHODS)

    rng = np.random.default_rng(1)
    ds = multitaper_connectivity(rng.standard_normal((256, 3)), sampling_frequency=250)
    # Same measures AND same variable order as the allowlist.
    assert tuple(ds.data_vars) == expected


def test_from_transform_supports_subclass_overriding_init():
    """from_transform must work for a subclass with the previous constructor.

    The private adoption fast-path passes a keyword the base __init__ accepts; a
    subclass that overrides __init__ (mirroring the old signature) need not, so
    from_transform must fall back to the plain (defensive-copy) path for it
    rather than raising TypeError.
    """
    rng = np.random.default_rng(3)
    m = Multitaper(
        rng.standard_normal((300, 6, 3)),
        sampling_frequency=300,
        time_halfbandwidth_product=3,
    )

    class LegacyConnectivity(Connectivity):
        def __init__(
            self,
            fourier_coefficients,
            expectation_type="trials_tapers",
            frequencies=None,
            time=None,
            dtype=np.complex128,
            minimum_phase_tolerance=1e-8,
            minimum_phase_max_iterations=500,
        ):
            super().__init__(
                fourier_coefficients,
                expectation_type,
                frequencies,
                time,
                dtype,
                minimum_phase_tolerance,
                minimum_phase_max_iterations,
            )
            self.marker = "subclass"

    sub = LegacyConnectivity.from_transform(m)
    assert isinstance(sub, LegacyConnectivity)
    assert sub.marker == "subclass"
    assert sub.half_bandwidth == m.half_bandwidth  # recorded without the new keyword
    # Fell back to the defensive-copy path (owns its data), and works normally.
    assert sub._fourier_coefficients.base is None
    base = Connectivity.from_transform(m)
    np.testing.assert_array_equal(sub.power(), base.power())
    # The base class still uses the no-copy adoption path (stores a view).
    assert base._fourier_coefficients.base is not None


def test_from_transform_subclass_overriding_init_keeps_transform_contract():
    """A subclass with a pass-through __init__ must receive the transform's
    sidedness and observation weights, not silently fall back to two-sided,
    unweighted defaults."""
    rng = np.random.default_rng(4)
    mw = MorletWavelet(
        rng.standard_normal((1000, 2, 2)),
        sampling_frequency=500,
        frequencies=[10.0, 20.0, 30.0, 40.0, 50.0],
        smoothing_time=0.1,
        smoothing_kernel="hann",
    )

    class PassThrough(Connectivity):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    sub = PassThrough.from_transform(mw)
    base = Connectivity.from_transform(mw)
    assert sub.is_one_sided is True
    assert sub.observation_weights is not None
    np.testing.assert_array_equal(
        to_host(sub.observation_weights), to_host(base.observation_weights)
    )
    np.testing.assert_array_equal(to_host(sub.frequencies), to_host(base.frequencies))
    np.testing.assert_allclose(sub.power(), base.power())
    np.testing.assert_allclose(
        sub.magnitude_squared_coherence(), base.magnitude_squared_coherence(), equal_nan=True
    )


def test_result_carries_descriptive_coordinate_metadata():
    """Coordinates carry unambiguous axis labels and physical units."""
    rng = np.random.default_rng(0)
    ds = multitaper_connectivity(rng.standard_normal((512, 5, 3)), sampling_frequency=500)
    assert ds.coords["time"].attrs["units"] == "s"
    assert ds.coords["time"].attrs["long_name"] == "Window center time"
    assert ds.coords["frequency"].attrs["units"] == "Hz"
    assert ds.coords["frequency"].attrs["long_name"] == "Frequency"
    assert ds.coords["source"].attrs["long_name"] == "Source signal"
    assert ds.coords["target"].attrs["long_name"] == "Target signal"


def test_result_carries_provenance_metadata():
    """Each measure records package/version/backend/expectation_type/measure."""
    from importlib.metadata import version

    rng = np.random.default_rng(1)
    da = connectivity_to_xarray(
        Multitaper(
            rng.standard_normal((512, 5, 3)),
            sampling_frequency=500,
            time_halfbandwidth_product=2,
        ),
        method="magnitude_squared_coherence",
    )
    assert da.attrs["measure"] == "magnitude_squared_coherence"
    assert da.attrs["measure_kwargs_json"] == "{}"
    assert da.attrs["package"] == "spectral_connectivity"
    assert da.attrs["package_version"] == version("spectral_connectivity")
    assert da.attrs["backend"] in ("cpu", "gpu")
    assert da.attrs["expectation_type"] == "trials_tapers"
    # The multitaper parameters are recorded under the mt_ prefix. Expected
    # values follow from the inputs: the default window spans all 512 samples
    # (T = 1.024 s), 2NW - 1 = 3 tapers, and resolution 2NW / T = 3.90625 Hz.
    assert da.attrs["mt_sampling_frequency"] == 500
    assert da.attrs["mt_nyquist_frequency"] == 250
    assert da.attrs["mt_n_trials"] == 5
    assert da.attrs["mt_n_signals"] == 3
    assert da.attrs["mt_n_time_samples_per_window"] == 512
    assert da.attrs["mt_time_halfbandwidth_product"] == 2
    assert da.attrs["mt_n_tapers"] == 3
    assert da.attrs["mt_frequency_resolution"] == pytest.approx(3.90625)


@pytest.mark.parametrize("n_tapers", [None, 31])
def test_low_bias_taper_filtering_reports_the_used_count_and_keeps_provenance(n_tapers):
    """When low-bias filtering drops DPSS tapers, the reported taper count is the
    number used, the provenance records the request, and repeated computations
    with the same transform record the same provenance."""
    data = np.random.default_rng(123).standard_normal((1000, 2, 2))
    requested = (
        r"of the 31 DPSS tapers you requested \(n_tapers=31\)"
        if n_tapers
        else r"of the default floor\(2 \* time_halfbandwidth_product\) - 1 = 31 DPSS"
    )
    with (
        pytest.warns(UserWarning, match="unusually large"),
        pytest.warns(UserWarning, match=requested) as record,
    ):
        mt = Multitaper(
            data,
            sampling_frequency=100,
            time_halfbandwidth_product=16,
            n_tapers=n_tapers,
        )
    if n_tapers is None:
        # NW already sets the default count: raising it is not the remedy.
        message = next(str(w.message) for w in record if "DPSS" in str(w.message))
        assert "raise time_halfbandwidth_product" not in message
        assert "is_low_bias=False" in message
    n_used = mt.tapers.shape[1]
    assert n_used < 31  # Low-bias filtering actually removed tapers.
    assert mt.n_tapers == n_used

    first = connectivity_to_xarray(mt, method="power")
    second = connectivity_to_xarray(mt, method="power")
    assert first.attrs["mt_n_tapers"] == n_used
    assert first.attrs["n_observations"] == 2 * n_used
    estimator = json.loads(first.attrs["transform_parameters_json"])["estimator"]
    assert estimator["n_tapers"] == n_used
    assert estimator["requested_n_tapers"] == n_tapers
    xr.testing.assert_identical(first, second)


def test_multitaper_connectivity_reports_the_tapers_low_bias_filtering_kept():
    """The wrapper's result counts the 5 tapers used, not the 50 requested."""
    data = np.random.default_rng(0).standard_normal((1000, 3))
    with pytest.warns(UserWarning, match="using 5 of the 50 DPSS tapers you requested"):
        result = multitaper_connectivity(
            data, sampling_frequency=500, n_tapers=50, method="power"
        )
    assert result.attrs["mt_n_tapers"] == 5
    assert result.attrs["n_observations"] == 5
    estimator = json.loads(result.attrs["transform_parameters_json"])["estimator"]
    assert estimator["n_tapers"] == 5
    assert estimator["requested_n_tapers"] == 50


def test_provenance_records_measure_kwargs(tmp_path, stub_measure):
    """Measure keyword arguments are recorded as ``arg_<key>``.

    Scalar kwargs remain convenient individual attributes, while structured
    values and the complete kwargs mapping use canonical JSON. Exercised through
    a stub measure that fits the (time, frequency, source, target) layout and
    accepts kwargs, since none of the default xarray-compatible measures take
    keyword arguments.
    """
    rng = np.random.default_rng(3)
    m = Multitaper(rng.standard_normal((256, 4, 3)), sampling_frequency=500)

    da = connectivity_to_xarray(
        m,
        method=stub_measure,
        connectivity_kwargs={"threshold": 0.5, "window": [1, 2, 3]},
    )
    # Scalar kwarg stored as-is under ``arg_<key>``; structured kwarg stored as
    # parseable JSON under the ``arg_<key>_json`` name so it self-identifies.
    assert da.attrs["arg_threshold"] == 0.5
    assert "arg_window" not in da.attrs
    assert json.loads(da.attrs["arg_window_json"]) == [1, 2, 3]
    assert json.loads(da.attrs["measure_kwargs_json"]) == {
        "threshold": 0.5,
        "window": [1, 2, 3],
    }
    # Structured provenance must not break NetCDF serialization.
    da.to_netcdf(tmp_path / "args.nc")


def test_provenance_arg_key_collision_raises_instead_of_overwriting(stub_measure):
    """A structured ``x`` and a scalar ``x_json`` cannot silently share a key."""
    rng = np.random.default_rng(3)
    m = Multitaper(rng.standard_normal((256, 4, 3)), sampling_frequency=500)

    with pytest.raises(ValueError, match="assigned twice"):
        connectivity_to_xarray(
            m,
            method=stub_measure,
            connectivity_kwargs={"x": [1, 2, 3], "x_json": 5},
        )


def test_broken_measure_in_batch_propagates_not_implemented(monkeypatch):
    """A genuine NotImplementedError is not swallowed into a missing variable."""
    rng = np.random.default_rng(9)

    def broken_measure(connectivity):
        msg = "backend cannot compute this"
        raise NotImplementedError(msg)

    monkeypatch.setattr(Connectivity, "broken_measure", broken_measure, raising=False)

    with pytest.raises(NotImplementedError, match="backend cannot compute"):
        multitaper_connectivity(
            rng.standard_normal((256, 4, 2)),
            sampling_frequency=256,
            method=["magnitude_squared_coherence", "broken_measure"],
        )


def test_wrapper_capabilities_do_not_use_method_name_substrings(monkeypatch):
    """A pairwise extension containing 'directed' is not given a directed
    measure's two-sided-spectrum requirement because of its name, and is
    labeled exactly as returned."""
    rng = np.random.default_rng(8)
    m = Multitaper(rng.standard_normal((128, 3, 2)), sampling_frequency=128)
    # Non-symmetric native matrix: entry [i, j] = 10 * i + j.
    native = np.array([[0.0, 1.0], [10.0, 11.0]])

    def undirected_similarity(connectivity):
        return np.broadcast_to(
            native, (len(connectivity.time), len(connectivity.frequencies), 2, 2)
        ).copy()

    monkeypatch.setattr(
        Connectivity, "undirected_similarity", undirected_similarity, raising=False
    )

    data_array = connectivity_to_xarray(m, method="undirected_similarity")
    assert data_array.dims == ("time", "frequency", "source", "target")
    # An unregistered extension is labeled as returned: [i, j] is (source i, target j).
    np.testing.assert_array_equal(data_array.sel(source="0", target="1"), 1.0)
    np.testing.assert_array_equal(data_array.sel(source="1", target="0"), 10.0)


def test_multitaper_connectivity_merges_nonstandard_dataset_in_batch():
    rng = np.random.default_rng(0)
    result = multitaper_connectivity(
        rng.standard_normal((256, 4, 3)),
        sampling_frequency=500,
        method=["magnitude_squared_coherence", "global_coherence"],
    )
    assert "magnitude_squared_coherence" in result
    assert "global_coherence" in result
    assert "global_coherence_vectors" in result


def test_frequency_bands_on_mixed_dataset_leaves_frequency_free_variables():
    # A batch mixing a frequency-carrying measure with one that has no frequency
    # axis (group_delay): band reduction applies to the former and passes the
    # latter through unchanged, rather than erroring on the frequency-free var.
    time_series = np.random.default_rng(452).standard_normal((512, 4, 3))
    methods = ["magnitude_squared_coherence", "group_delay"]
    bands = {"alpha": (8, 12), "beta": (13, 30)}
    result = multitaper_connectivity(
        time_series,
        sampling_frequency=200,
        method=methods,
        frequency_bands=bands,
    )
    unreduced = multitaper_connectivity(time_series, sampling_frequency=200, method=methods)

    assert result["magnitude_squared_coherence"].dims == ("time", "band", "source", "target")
    assert result["group_delay"].dims == ("time", "source", "target")
    assert result.band.values.tolist() == ["alpha", "beta"]
    for name, (lower, upper) in bands.items():
        expected = (
            unreduced["magnitude_squared_coherence"]
            .sel(frequency=slice(lower, upper))
            .mean("frequency")
        )
        xr.testing.assert_allclose(
            result["magnitude_squared_coherence"].sel(band=name, drop=True), expected
        )
    xr.testing.assert_identical(result["group_delay"], unreduced["group_delay"])


def test_multitaper_connectivity_merges_rich_multivariate_datasets():
    result = multitaper_connectivity(
        np.random.default_rng(430).standard_normal((128, 5, 4)),
        sampling_frequency=64,
        method=[
            "canonical_coherency",
            "maximized_imaginary_coherency_components",
        ],
        connectivity_kwargs={
            "group_labels": [0, 0, 1, 1],
            "n_components": 1,
        },
    )

    assert "canonical_coherency_filters" in result
    assert "maximized_imaginary_coherency_components_patterns" in result
    assert list(result.data_vars).count("group_membership") == 1


def test_multitaper_connectivity_group_pairwise_and_components_coordinates_do_not_collide():
    # A group-pairwise measure uses source_group/target_group as *dimensions*;
    # components use connection_source_group/connection_target_group as per-
    # connection coordinates. Merging them must keep the per-connection labels
    # intact rather than overwrite them with the group dimension index.
    result = multitaper_connectivity(
        np.random.default_rng(451).standard_normal((128, 5, 4)),
        sampling_frequency=64,
        method=["blockwise_spectral_granger_prediction", "canonical_coherency"],
        connectivity_kwargs={"group_labels": [0, 0, 1, 1]},
    )
    assert result["target_group"].dims == ("target_group",)
    assert result["connection_target_group"].dims == ("connection",)
    assert result.connection_source_group.values.tolist() == [0]
    assert result.connection_target_group.values.tolist() == [1]


def test_multitaper_connectivity_genuine_error_not_swallowed():
    """A real computation error in a batch surfaces; it is not silently dropped.

    A debiased measure requires >= 2 observations. With one trial and one taper
    it raises ValueError and must propagate, rather than leave the user with a
    Dataset that silently omits the requested measure alongside the ones that
    happened to succeed.
    """
    rng = np.random.default_rng(0)
    ts = rng.standard_normal((256, 1, 3))  # one trial
    with pytest.raises(ValueError, match="at least 2 observations"):
        multitaper_connectivity(
            ts,
            sampling_frequency=500,
            time_halfbandwidth_product=1,  # -> one taper -> n_observations = 1
            method=["power", "debiased_squared_phase_lag_index"],
        )


def test_multitaper_connectivity_single_dataset_measure_is_dataset():
    rng = np.random.default_rng(0)
    result = multitaper_connectivity(
        rng.standard_normal((256, 4, 3)),
        sampling_frequency=500,
        method=["global_coherence"],
    )
    assert set(result.data_vars) == {
        "global_coherence",
        "global_coherence_vectors",
    }


def test_metadata_survives_netcdf_round_trip(tmp_path):
    """Provenance attrs and coordinate units survive a NetCDF round-trip."""
    import xarray as xr

    rng = np.random.default_rng(2)
    ds = multitaper_connectivity(rng.standard_normal((512, 5, 3)), sampling_frequency=500)
    path = tmp_path / "provenance.nc"
    ds.to_netcdf(path)
    reloaded = xr.open_dataset(path)
    try:
        assert reloaded.coords["time"].attrs["units"] == "s"
        assert reloaded.coords["frequency"].attrs["units"] == "Hz"
        var = reloaded["magnitude_squared_coherence"]
        assert var.attrs["package"] == "spectral_connectivity"
        assert var.attrs["backend"] in ("cpu", "gpu")
        assert var.attrs["expectation_type"] == "trials_tapers"
        # The shared provenance attached at the Dataset level also round-trips.
        assert reloaded.attrs["package"] == "spectral_connectivity"
        assert reloaded.attrs["backend"] in ("cpu", "gpu")
        assert reloaded.attrs["expectation_type"] == "trials_tapers"
        assert reloaded.attrs["mt_sampling_frequency"] == 500
    finally:
        reloaded.close()


def test_backend_provenance_reflects_imported_backend_not_env(monkeypatch):
    """The backend attr must reflect the imported backend, not the live env var.

    The backend is fixed when the package is imported; toggling
    SPECTRAL_CONNECTIVITY_ENABLE_GPU afterwards must not mislabel a result.
    """
    from spectral_connectivity import _backend

    monkeypatch.setenv("SPECTRAL_CONNECTIVITY_ENABLE_GPU", "true")
    rng = np.random.default_rng(0)
    da = connectivity_to_xarray(
        Multitaper(rng.standard_normal((256, 5, 3)), sampling_frequency=500),
        method="magnitude_squared_coherence",
    )
    # Matches the actually-imported backend, not the toggled env var. (Left
    # backend-agnostic so the suite can also run under the GPU backend.)
    assert da.attrs["backend"] == ("gpu" if _backend.ON_GPU else "cpu")


def test_multitaper_connectivity_rejects_empty_method_list():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="at least one connectivity measure"):
        multitaper_connectivity(
            rng.standard_normal((256, 3, 2)), sampling_frequency=256, method=[]
        )


def test_multitaper_connectivity_merges_frequency_and_band_only_outputs():
    rng = np.random.default_rng(0)
    result = multitaper_connectivity(
        rng.standard_normal((256, 3, 2)),
        sampling_frequency=256,
        method=["global_coherence", "phase_slope_index"],
    )
    assert "global_coherence" in result
    assert "phase_slope_index" in result
    assert "frequency" not in result.phase_slope_index.dims


def test_multitaper_connectivity_rejects_duplicate_signal_names():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="signal_names must be unique"):
        multitaper_connectivity(
            rng.standard_normal((256, 3, 2)),
            sampling_frequency=256,
            method="magnitude_squared_coherence",
            signal_names=["a", "a"],
        )


def test_multitaper_connectivity_rejects_duplicate_nan_signal_names():
    """Index semantics treat distinct NaN objects as duplicate labels."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="must not contain missing labels"):
        multitaper_connectivity(
            rng.standard_normal((256, 3, 2)),
            sampling_frequency=256,
            method="magnitude_squared_coherence",
            signal_names=[float("nan"), float("nan")],
        )


def test_multitaper_connectivity_rejects_structured_signal_names():
    """Hashable tuples are not silently accepted as non-portable coordinates."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="one-dimensional xarray coordinate"):
        multitaper_connectivity(
            rng.standard_normal((256, 3, 2)),
            sampling_frequency=256,
            method="magnitude_squared_coherence",
            signal_names=[("region", 1), ("region", 2)],
        )


def test_multitaper_connectivity_rejects_nonportable_integer_signal_names():
    """Accepted integer coordinates must remain writable by the SciPy backend."""
    rng = np.random.default_rng(0)
    labels = np.array([2**63, 2**63 + 1], dtype=np.uint64)
    with pytest.raises(ValueError, match="signed 32-bit range"):
        multitaper_connectivity(
            rng.standard_normal((256, 3, 2)),
            sampling_frequency=256,
            method="power",
            signal_names=labels,
        )


def test_portable_integer_signal_name_boundaries_survive_netcdf(tmp_path):
    """The documented signed 32-bit boundary values serialize successfully."""
    labels = np.array([np.iinfo(np.int32).min, np.iinfo(np.int32).max], dtype=np.int64)
    result = multitaper_connectivity(
        np.random.default_rng(0).standard_normal((256, 3, 2)),
        sampling_frequency=256,
        method="power",
        signal_names=labels,
    )

    path = tmp_path / "integer_signal_boundaries.nc"
    result.to_netcdf(path)
    reloaded = xr.open_dataarray(path)
    try:
        np.testing.assert_array_equal(reloaded.source.values, labels)
    finally:
        reloaded.close()


def test_multitaper_connectivity_squeeze_retains_pair_labels():
    """squeeze=True reduces to (time, frequency) but keeps the pair as coords.

    The old behavior dropped the source/target labels entirely, so a squeezed
    result no longer recorded which pair (or, for directed measures, which
    direction) it represented. isel(drop=False) keeps them as scalar coords.
    """
    rng = np.random.default_rng(0)
    da = multitaper_connectivity(
        rng.standard_normal((256, 5, 2)),
        sampling_frequency=256,
        method="magnitude_squared_coherence",
        signal_names=["x", "y"],
        squeeze=True,
    )
    assert da.dims == ("time", "frequency")
    assert da.coords["source"].item() == "x"
    assert da.coords["target"].item() == "y"


def test_multitaper_connectivity_squeeze_warns_and_keeps_matrix_for_many_signals():
    rng = np.random.default_rng(0)
    with pytest.warns(UserWarning, match="squeeze=True"):
        da = multitaper_connectivity(
            rng.standard_normal((256, 5, 3)),
            sampling_frequency=256,
            method="magnitude_squared_coherence",
            squeeze=True,
        )
    assert da.dims == ("time", "frequency", "source", "target")


def test_multitaper_connectivity_squeeze_ignored_for_multi_measure_dataset():
    """squeeze=True must not corrupt a mixed Dataset (pairwise + power).

    Scalar source/target coordinates are Dataset-wide, so a squeezed pairwise
    variable would collide with ``power``'s ``source`` dimension: coherence would
    lose its source label and power would inherit a bogus scalar target. squeeze
    is therefore ignored (with a warning) whenever the result is a Dataset, and
    every variable keeps its full, correct axes.
    """
    rng = np.random.default_rng(0)
    with pytest.warns(UserWarning, match="ignored for multi-measure"):
        ds = multitaper_connectivity(
            rng.standard_normal((256, 5, 2)),
            sampling_frequency=256,
            method=None,  # default set: pairwise measures plus power
            signal_names=["x", "y"],
            squeeze=True,
        )
    assert "power" in ds.data_vars
    assert ds["magnitude_squared_coherence"].dims == ("time", "frequency", "source", "target")
    assert ds["power"].dims == ("time", "frequency", "source")
    # No Dataset-wide scalar coordinate leaked onto the wrong variable.
    assert "target" not in ds["power"].coords
    assert ds["magnitude_squared_coherence"].sizes["source"] == 2


@pytest.fixture(scope="module")
def unidirectional_var():
    """Multi-trial VAR(1) in which signal 0 (x) drives signal 1 (y) only.

    Returns
    -------
    time_series : np.ndarray, shape (2000, 8, 2)
    """
    rng = np.random.default_rng(0)
    n_time, n_trials = 2000, 8
    time_series = np.zeros((n_time, n_trials, 2))
    for trial in range(n_trials):
        x = np.zeros(n_time)
        y = np.zeros(n_time)
        e_x = rng.standard_normal(n_time)
        e_y = rng.standard_normal(n_time)
        for t in range(1, n_time):
            x[t] = 0.5 * x[t - 1] + e_x[t]
            y[t] = 0.5 * y[t - 1] + 0.6 * x[t - 1] + e_y[t]  # 0 -> 1
        time_series[:, trial, 0] = x
        time_series[:, trial, 1] = y
    return time_series


_DIRECTED_PAIRWISE_METHODS = [
    "pairwise_spectral_granger_prediction",
    "directed_transfer_function",
    "directed_coherence",
    "partial_directed_coherence",
    "generalized_partial_directed_coherence",
    "direct_directed_transfer_function",
]
# Minimum causal peak for measures whose scale makes a floor meaningful on this
# VAR (the dDTF peak is only ~0.05, so it is held to the ordering check alone).
_CAUSAL_PEAK_FLOOR = {"pairwise_spectral_granger_prediction": 0.05}


@pytest.mark.parametrize("method", _DIRECTED_PAIRWISE_METHODS)
def test_directed_measures_are_oriented_source_to_target(method, unidirectional_var):
    """Directed measures label sel(source=driver, target=receiver) correctly.

    For a unidirectional VAR (signal 0 drives 1), the causal entry is
    sel(source=0, target=1); it must dominate the anti-causal sel(source=1,
    target=0). A Connectivity array left target first would swap them.
    """
    da = multitaper_connectivity(
        unidirectional_var,
        sampling_frequency=200,
        time_halfbandwidth_product=3,
        method=method,
        signal_names=["x", "y"],
    )
    assert da.dims == ("time", "frequency", "source", "target")
    causal = da.sel(source="x", target="y").values  # x drives y
    anti_causal = da.sel(source="y", target="x").values
    assert np.nanmax(causal) > np.nanmax(anti_causal)
    assert np.nanmax(causal) > _CAUSAL_PEAK_FLOOR.get(method, 0.0)


@pytest.fixture(scope="module")
def lagged_triplet():
    """x drives y at a 3-sample lag (6 ms at 500 Hz); z is independent noise.

    Returns
    -------
    time_series : np.ndarray, shape (400, 6, 3)
    """
    rng = np.random.default_rng(0)
    n_time, n_trials, lag = 400, 6, 3
    x = rng.standard_normal((n_time + lag, n_trials))
    y = 0.8 * x[:-lag] + 0.5 * rng.standard_normal((n_time, n_trials))
    z = rng.standard_normal((n_time, n_trials))
    return np.stack([x[lag:], y, z], axis=-1)


_LAGGED_SAMPLING_FREQUENCY = 500
_LAGGED_LAG_SAMPLES = 3
_LAGGED_BAND = (5.0, 100.0)
_LAGGED_KWARGS = {
    "subset_pairwise_spectral_granger_prediction": {"pairs": [(0, 1)]},
    "blockwise_spectral_granger_prediction": {"group_labels": ["A", "B", "C"]},
}
# Measures that consume a frequency interval get the band as frequency_range.
_LAGGED_INTERVAL_MEASURES = frozenset({"phase_slope_index", "delay", "group_delay"})


def _directed_entry(values: xr.DataArray) -> float:
    """Median of one directed entry: over the band when frequency-resolved, at
    the zero-wrap candidate for ``delay``."""
    if "frequency" in values.dims:
        values = values.sel(frequency=slice(*_LAGGED_BAND))
    if "candidate" in values.dims:
        values = values.sel(candidate=0)
    return float(np.nanmedian(values.values))


@pytest.mark.parametrize(
    "method",
    [
        "conditional_spectral_granger_prediction",
        "subset_pairwise_spectral_granger_prediction",
        "time_reversed_spectral_granger_prediction",
        "blockwise_spectral_granger_prediction",
        "directed_phase_lag_index",
        "phase_slope_index",
        "delay",
        "group_delay",
    ],
)
def test_remaining_directed_measures_are_oriented_source_to_target(method, lagged_triplet):
    """``sel(source="x", target="y")`` is the large / positive entry for the
    directed measures the DTF-family test above omits. x drives y with a +6 ms
    lag, so Granger-type scores, dPLI (> 0.5), PSI, and the delays (+0.006 s)
    must all point x -> y. Time reversal flips the apparent direction, so the
    time-reversed measure is given reversed data."""
    time_series = lagged_triplet
    if method == "time_reversed_spectral_granger_prediction":
        time_series = time_series[::-1]
    result = multitaper_connectivity(
        time_series,
        sampling_frequency=_LAGGED_SAMPLING_FREQUENCY,
        time_halfbandwidth_product=3,
        method=method,
        signal_names=["x", "y", "z"],
        connectivity_kwargs=_LAGGED_KWARGS.get(method, {}),
        frequency_range=_LAGGED_BAND if method in _LAGGED_INTERVAL_MEASURES else None,
    )
    values = result[method] if isinstance(result, xr.Dataset) else result
    if method == "blockwise_spectral_granger_prediction":
        forward = _directed_entry(values.sel(source_group="A", target_group="B"))
        backward = _directed_entry(values.sel(source_group="B", target_group="A"))
    else:
        forward = _directed_entry(values.sel(source="x", target="y"))
        backward = _directed_entry(values.sel(source="y", target="x"))

    assert forward > backward
    expected_delay = _LAGGED_LAG_SAMPLES / _LAGGED_SAMPLING_FREQUENCY
    if method == "directed_phase_lag_index":
        assert forward > 0.5 > backward
    elif method == "phase_slope_index":
        assert forward > 0 > backward
    elif method in {"delay", "group_delay"}:
        assert forward == pytest.approx(expected_delay, abs=1e-3)
        assert backward == pytest.approx(-expected_delay, abs=1e-3)
    else:
        # Granger-type: x -> y is about 1.1 on this system, y -> x about 0.01.
        assert forward > 0.5
        assert forward > 10 * backward


def test_multitaper_connectivity_dataset_carries_shared_provenance():
    """A multi-measure Dataset exposes shared provenance at the top level.

    Previously provenance lived only on each DataArray; a returned Dataset had
    no top-level attrs, so tracing how a batch was produced meant inspecting an
    arbitrary variable. The shared attrs (package, backend, expectation type,
    multitaper parameters) are now on the Dataset too. (Their NetCDF round-trip
    is covered by ``test_metadata_survives_netcdf_round_trip``.)
    """
    rng = np.random.default_rng(0)
    ds = multitaper_connectivity(rng.standard_normal((512, 5, 3)), sampling_frequency=500)
    assert ds.attrs["package"] == "spectral_connectivity"
    assert ds.attrs["backend"] in ("cpu", "gpu")
    assert ds.attrs["expectation_type"] == "trials_tapers"
    assert ds.attrs["mt_sampling_frequency"] == 500
    # The shared attrs must not include per-measure fields.
    assert "measure" not in ds.attrs


def test_fourier_connectivity_matches_multitaper_adapter():
    """Externally supplied FFT coefficients reuse the same numerical contract."""
    rng = np.random.default_rng(301)
    transform = Multitaper(
        rng.standard_normal((256, 5, 3)),
        sampling_frequency=128,
        time_halfbandwidth_product=2,
    )
    expected = connectivity_to_xarray(
        transform,
        method="magnitude_squared_coherence",
        signal_names=["a", "b", "c"],
    )
    actual = fourier_connectivity(
        transform.fft(),
        frequencies=transform.frequencies,
        time=transform.time,
        method="magnitude_squared_coherence",
        signal_names=["a", "b", "c"],
    )

    xr.testing.assert_allclose(actual, expected)
    assert actual.attrs["transform"] == "external_fourier"


def test_fourier_connectivity_infers_and_transposes_labeled_dimensions():
    """A coefficient DataArray carries frequency, time, and signal coordinates."""
    rng = np.random.default_rng(302)
    transform = Multitaper(
        rng.standard_normal((256, 4, 2)),
        sampling_frequency=128,
        time_window_duration=1,
        time_halfbandwidth_product=2,
    )
    coefficients = transform.fft()
    labeled = xr.DataArray(
        coefficients.transpose(4, 3, 1, 0, 2),
        dims=("channel", "frequency", "epoch", "window", "taper"),
        coords={
            "channel": ["left", "right"],
            "frequency": transform.frequencies,
            "window": transform.time,
        },
        attrs={"subject": "rat-1"},
    )
    actual = fourier_connectivity(labeled, method="power")
    expected = fourier_connectivity(
        coefficients,
        frequencies=transform.frequencies,
        time=transform.time,
        signal_names=["left", "right"],
        method="power",
    )

    xr.testing.assert_allclose(actual, expected)
    assert json.loads(actual.attrs["input_attrs_json"]) == {"subject": "rat-1"}


def test_multitaper_frequency_crop_decimation_and_band_mean():
    """Frequency operations are coordinate-based and preserve axis order.

    Each band value is the plain mean of the unreduced result over the band's
    inclusive bins. After cropping to [8, 32] Hz and decimating the 0.5 Hz grid
    by 2 the bins are the integers 8..32, so every band endpoint below lands
    exactly on a bin and pins inclusivity at both ends.
    """
    time_series = np.random.default_rng(303).standard_normal((256, 3))
    frequency_kwargs = {
        "sampling_frequency": 128,
        "method": "magnitude_squared_coherence",
        "frequency_range": (8, 32),
        "frequency_decimation": 2,
    }
    bands = {"alpha": (8, 12), "beta": (13, 30)}
    result = multitaper_connectivity(time_series, frequency_bands=bands, **frequency_kwargs)
    unreduced = multitaper_connectivity(time_series, **frequency_kwargs)

    assert result.dims == ("time", "band", "source", "target")
    assert result.band.values.tolist() == ["alpha", "beta"]
    assert result.attrs["frequency_reduction"] == "mean"
    # Premise: both alpha endpoints are bins, so an exclusive bound drops one.
    np.testing.assert_array_equal(
        unreduced.frequency.sel(frequency=slice(8, 12)), [8.0, 9.0, 10.0, 11.0, 12.0]
    )
    for name, (lower, upper) in bands.items():
        expected = unreduced.sel(frequency=slice(lower, upper)).mean("frequency")
        xr.testing.assert_allclose(result.sel(band=name, drop=True), expected)


def test_frequency_band_reduce_takes_the_wrappers_band_keywords():
    """The helper spells its band options as the wrappers do, with the mapping
    still positional, so a wrapper call and a helper call read the same and
    give the same result."""
    parameters = inspect.signature(frequency_band_reduce).parameters
    assert list(parameters) == ["result", "frequency_bands", "frequency_reduction", "circular"]
    assert parameters["frequency_bands"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    for name in ("frequency_reduction", "circular"):
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["frequency_reduction"].default == "mean"
    assert parameters["circular"].default is None
    for wrapper in (multitaper_connectivity, fourier_connectivity):
        wrapper_parameters = inspect.signature(wrapper).parameters
        assert "frequency_bands" in wrapper_parameters
        assert wrapper_parameters["frequency_reduction"].default == "mean"

    time_series = np.random.default_rng(57).standard_normal((512, 3, 2))
    kwargs = {"sampling_frequency": 128, "method": "power"}
    bands = {"theta": (4.0, 8.0), "alpha": (8.0, 12.0)}
    power = multitaper_connectivity(time_series, **kwargs)
    for reduction in ("mean", "integral"):
        xr.testing.assert_identical(
            frequency_band_reduce(power, bands, frequency_reduction=reduction),
            multitaper_connectivity(
                time_series, frequency_bands=bands, frequency_reduction=reduction, **kwargs
            ),
        )
    xr.testing.assert_identical(
        frequency_band_reduce(power, frequency_bands=bands),
        frequency_band_reduce(power, bands),
    )
    # The unreleased spellings are gone, not aliased.
    with pytest.raises(TypeError, match="bands"):
        frequency_band_reduce(power, bands=bands)
    with pytest.raises(TypeError, match="reduction"):
        frequency_band_reduce(power, bands, reduction="integral")


def test_frequency_band_reduce_uses_circular_phase_mean():
    """Phases straddling the branch cut average near pi, not zero."""
    phase = xr.DataArray(
        np.array([np.pi - 0.1, -np.pi + 0.1]),
        dims=("frequency",),
        coords={"frequency": [10.0, 11.0]},
        name="coherence_phase",
        attrs={"measure": "coherence_phase"},
    )
    reduced = frequency_band_reduce(phase, {"alpha": (8, 12)})

    assert abs(float(reduced.sel(band="alpha"))) == pytest.approx(np.pi)


@pytest.mark.parametrize(
    ("method", "connectivity_kwargs", "score_name"),
    [
        ("global_coherence", {}, "global_coherence"),
        (
            "canonical_coherency",
            {"group_labels": [0, 0, 1, 1]},
            "canonical_coherency",
        ),
    ],
)
def test_frequency_band_reduce_rejects_unidentifiable_projection_averages(
    method, connectivity_kwargs, score_name
):
    """Frequency-specific vectors cannot be averaged without phase alignment."""
    result = multitaper_connectivity(
        np.random.default_rng(321).standard_normal((128, 5, 4)),
        sampling_frequency=64,
        method=method,
        connectivity_kwargs=connectivity_kwargs,
    )

    with pytest.raises(ValueError, match="sign/phase is arbitrary"):
        frequency_band_reduce(result, {"alpha": (8, 12)})

    score = frequency_band_reduce(result[score_name], {"alpha": (8, 12)})
    assert "band" in score.dims
    assert "frequency" not in score.dims


def test_frequency_band_mean_propagates_nan_and_keeps_band_validity():
    """A band containing an edge-invalid bin is undefined for both reductions,
    and the per-band validity is carried as a coordinate."""
    with pytest.warns(UserWarning, match="shorter than 4 wavelet standard deviations"):
        transform = MorletWavelet(
            np.random.default_rng(324).standard_normal((2400, 1, 2)),
            200,
            frequencies=np.array([4.0, 8.0, 16.0, 32.0]),
            smoothing_time=0.5,
            edge_mode="nan",
        )
    coherence = connectivity_to_xarray(transform, method="magnitude_squared_coherence")
    power = connectivity_to_xarray(transform, method="power")
    bands = {"low": (4, 8), "all": (4, 32)}
    mean = frequency_band_reduce(coherence, bands)
    integral = frequency_band_reduce(power, bands, frequency_reduction="integral")

    validity = to_host(transform.valid_time_frequency)
    expected_valid = np.stack([validity[:, :2].all(axis=1), validity.all(axis=1)], 1)
    assert not expected_valid.all()
    assert expected_valid.any()
    np.testing.assert_array_equal(mean.valid_time_band.values, expected_valid)
    np.testing.assert_array_equal(integral.valid_time_band.values, expected_valid)
    # NaN exactly where the band is not fully valid, for both reductions.
    np.testing.assert_array_equal(
        np.isnan(mean.sel(source="0", target="1").values), ~expected_valid
    )
    np.testing.assert_array_equal(np.isnan(integral.sel(source="0").values), ~expected_valid)


def test_frequency_band_reduce_after_time_selection():
    """Band reduction must work on a single selected time point; the band
    validity coordinate cannot assume a surviving 'time' dimension."""
    transform = MorletWavelet(
        np.random.default_rng(325).standard_normal((1500, 2, 3)),
        250,
        frequencies=np.arange(5.0, 45.0, 5.0),
        smoothing_time=0.3,
        edge_mode="nan",
    )
    result = connectivity_to_xarray(transform, method="magnitude_squared_coherence")
    reduced = frequency_band_reduce(result.isel(time=10), {"a": (5, 15), "b": (20, 40)})
    assert "time" not in reduced.dims
    assert "band" in reduced.dims
    assert reduced.valid_time_band.dims == ("band",)


def test_single_bin_band_integral_preserves_nan():
    """A zero-width band must not turn an invalid spectral bin into zero power."""
    power = xr.DataArray(
        [[np.nan, 2.0]],
        dims=("time", "frequency"),
        coords={
            "time": [0.0],
            "frequency": [10.0, 20.0],
            "valid_time_frequency": (
                ("time", "frequency"),
                [[False, True]],
            ),
        },
        name="power",
        attrs={"measure": "power"},
    )

    reduced = frequency_band_reduce(
        power, {"invalid": (10.0, 10.0)}, frequency_reduction="integral"
    )

    assert np.isnan(reduced.sel(band="invalid")).all()
    assert not reduced.valid_time_band.sel(band="invalid").any()


def test_frequency_band_integral_is_restricted_to_spectral_densities():
    score = xr.DataArray(
        [0.25, 0.5],
        dims=("frequency",),
        coords={"frequency": [1.0, 2.0]},
        name="magnitude_squared_coherence",
        attrs={"measure": "magnitude_squared_coherence"},
    )
    with pytest.raises(ValueError, match="defined only for power"):
        frequency_band_reduce(score, {"low": (1, 2)}, frequency_reduction="integral")


def test_multitaper_frequency_crop_and_decimation_select_correct_bins():
    """Cropping happens before decimation and selects exact coordinate values.

    The full grid is multiples of 128/256 = 0.5 Hz. Cropping to [8.5, 32] and
    then keeping every other bin starts at 8.5 Hz: 8.5, 9.5, ..., 31.5.
    Decimating first would instead keep the integer bins 9..32.
    """
    result = multitaper_connectivity(
        np.random.default_rng(313).standard_normal((256, 2)),
        sampling_frequency=128,
        method="magnitude_squared_coherence",
        frequency_range=(8.5, 32),
        frequency_decimation=2,
    )

    np.testing.assert_array_equal(result.frequency.values, np.arange(8.5, 32.0, 1.0))


def test_frequency_band_integral_equals_analytic_area():
    """Band integral of a flat spectral density equals value times bandwidth."""
    frequencies = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    flat_power = xr.DataArray(
        np.full_like(frequencies, 2.0),
        dims=("frequency",),
        coords={"frequency": frequencies},
        name="power",
        attrs={"measure": "power"},
    )
    reduced = frequency_band_reduce(
        flat_power, {"band": (1.0, 5.0)}, frequency_reduction="integral"
    )
    # The integral of the constant 2.0 over [1, 5] Hz is 2 * (5 - 1) = 8.
    assert float(reduced.sel(band="band")) == pytest.approx(8.0)


@pytest.fixture
def flat_density_on_2hz_grid():
    """A constant density 2.0 on the bins 0, 2, ..., 50 Hz."""
    frequencies = np.arange(0.0, 51.0, 2.0)
    return xr.DataArray(
        np.full_like(frequencies, 2.0),
        dims=("frequency",),
        coords={"frequency": frequencies},
        name="power",
        attrs={"measure": "power"},
    )


@pytest.mark.parametrize(
    "bounds",
    [(1.0, 4.0), (8.0, 13.0), (5.5, 6.5), (10.0, 30.0)],
    ids=["delta-off-grid", "alpha-off-grid", "one-bin", "on-grid"],
)
def test_frequency_band_integral_is_exact_for_any_band_edges(flat_density_on_2hz_grid, bounds):
    """Each bin owns the cell between the midpoints to its neighbours, so a flat
    density integrates to density * bandwidth whatever the band edges (the
    trapezoid over the bins inside the band gave 45% less for 1-4 Hz and 0 for a
    one-bin band)."""
    low, high = bounds
    reduced = frequency_band_reduce(
        flat_density_on_2hz_grid, {"band": bounds}, frequency_reduction="integral"
    )
    assert float(reduced.sel(band="band")) == pytest.approx(2.0 * (high - low))


def test_frequency_band_integrals_of_adjacent_bands_add_up(flat_density_on_2hz_grid):
    rng = np.random.default_rng(44)
    density = flat_density_on_2hz_grid.copy(data=rng.uniform(0.5, 3.0, 26))
    reduced = frequency_band_reduce(
        density,
        {"low": (10.0, 17.3), "high": (17.3, 30.0), "both": (10.0, 30.0)},
        frequency_reduction="integral",
    )
    assert float(reduced.sel(band="low") + reduced.sel(band="high")) == pytest.approx(
        float(reduced.sel(band="both"))
    )


def test_frequency_band_integral_of_power_recovers_the_variance():
    """Parseval: integrating one-sided power over [0, Nyquist] gives the variance."""
    time_series = 3.0 * np.random.default_rng(45).standard_normal((20000, 1, 1))
    power = multitaper_connectivity(
        time_series, sampling_frequency=500, method="power", time_window_duration=1.0
    ).mean("time")
    reduced = frequency_band_reduce(
        power, {"all": (0.0, 250.0)}, frequency_reduction="integral"
    )
    assert float(reduced.sel(band="all").squeeze()) == pytest.approx(9.0, rel=0.03)


@pytest.mark.parametrize("n_fft_samples", [4000, 4001], ids=["even-nfft", "odd-nfft"])
def test_frequency_band_integral_over_every_bin_is_parseval_exact(n_fft_samples):
    """A DC offset puts most of the power in the DC bin, which the one-sided
    spectrum does not double. Integrating over a band that covers every bin
    must weight that bin (and a Nyquist bin) with a full spacing, so the result
    equals ``sum(power) * spacing`` exactly and ``mean(x**2)`` physically."""
    rng = np.random.default_rng(46)
    time_series = 10.0 + 0.1 * rng.standard_normal((4000, 1, 1))
    power = multitaper_connectivity(
        time_series,
        sampling_frequency=100,
        method="power",
        detrend_type=None,
        time_halfbandwidth_product=1,
        transform_kwargs={"n_fft_samples": n_fft_samples},
    )
    assert power.frequency.size == n_fft_samples // 2 + 1
    spacing = float(power.frequency[1] - power.frequency[0])
    integral = float(
        frequency_band_reduce(
            power, {"all": (0.0, 50.0)}, frequency_reduction="integral"
        ).squeeze()
    )
    assert integral == pytest.approx(
        float(power.sum("frequency").squeeze()) * spacing, rel=1e-12
    )
    assert integral == pytest.approx(float(np.mean(time_series**2)), rel=1e-3)


def test_frequency_band_integral_counts_the_nyquist_bin_fully():
    """A tone at the Nyquist frequency lands in the last one-sided bin, which is
    not doubled either; integrating over [0, Nyquist] must recover its power."""
    rng = np.random.default_rng(47)
    n_time = 4000
    tone = 10.0 * (-1.0) ** np.arange(n_time) + 0.1 * rng.standard_normal(n_time)
    power = multitaper_connectivity(
        tone[:, None, None],
        sampling_frequency=100,
        method="power",
        detrend_type=None,
        time_halfbandwidth_product=1,
    )
    assert float(power.frequency[-1]) == 50.0
    spacing = float(power.frequency[1] - power.frequency[0])
    integral = float(
        frequency_band_reduce(
            power, {"all": (0.0, 50.0)}, frequency_reduction="integral"
        ).squeeze()
    )
    # Most of the power sits in the Nyquist bin, so this exercises its weight.
    assert float(power.isel(frequency=-1).squeeze()) * spacing > 0.8 * integral
    assert integral == pytest.approx(float(np.mean(tone**2)), rel=1e-3)


def _last_bin_halves(integrate, frequencies):
    """Integrals over the lower and upper half-cells of the last bin."""
    spacing = frequencies[1] - frequencies[0]
    last = frequencies[-1]
    return integrate(
        {"lower": (last - spacing / 2, last), "upper": (last, last + spacing / 2)}
    )


@pytest.mark.parametrize("path", ["wrapper", "frequency_band_reduce", "fourier"])
def test_frequency_band_integral_of_odd_fft_last_bin_is_not_nyquist(path):
    """An odd-length FFT has no Nyquist bin: its last one-sided bin is an
    ordinary doubled bin whose cell extends half a spacing above it. Folding it
    as Nyquist put its whole cell in the lower half and nothing in the upper.

    Regression: the last bin of any grid starting at 0 Hz was taken as Nyquist.
    """
    n_time, sampling_frequency = 401, 100.0
    time_series = np.random.default_rng(52).standard_normal((n_time, 1, 1))
    power_kwargs = {
        "sampling_frequency": sampling_frequency,
        "method": "power",
        "time_halfbandwidth_product": 1,
        "transform_kwargs": {"n_fft_samples": n_time},
    }
    power = multitaper_connectivity(time_series, **power_kwargs).squeeze()
    frequencies = power.frequency.values
    assert frequencies[-1] < sampling_frequency / 2

    if path == "wrapper":

        def integrate(bands):
            return multitaper_connectivity(
                time_series,
                frequency_bands=bands,
                frequency_reduction="integral",
                **power_kwargs,
            ).squeeze()

    elif path == "frequency_band_reduce":

        def integrate(bands):
            return frequency_band_reduce(power, bands, frequency_reduction="integral")

    else:
        coefficients = Multitaper(
            time_series,
            sampling_frequency=sampling_frequency,
            time_halfbandwidth_product=1,
            n_fft_samples=n_time,
        ).fft()

        def integrate(bands):
            return fourier_connectivity(
                coefficients,
                frequencies=np.fft.fftfreq(n_time, 1 / sampling_frequency),
                method="power",
                frequency_bands=bands,
                frequency_reduction="integral",
            ).squeeze()

    halves = _last_bin_halves(integrate, frequencies)
    half_cell = float(power.isel(frequency=-1)) * (frequencies[1] - frequencies[0]) / 2
    np.testing.assert_allclose(halves.sel(band="lower"), half_cell, rtol=1e-9)
    np.testing.assert_allclose(halves.sel(band="upper"), half_cell, rtol=1e-9)


@pytest.mark.parametrize(
    ("band", "frequency_range"),
    [
        ((0.0, 4.0), (0.0, 4.0)),
        ((4.0, 8.0), (0.0, 8.0)),
        ((10.0, 30.0), (0.0, 30.0)),
        # Edges between the 1 Hz bins: the 10 and 31 Hz bins are outside the
        # range, but part of each one's cell is inside the band.
        ((10.3, 30.6), (10.3, 30.6)),
        ((10.3, 30.6), (10.3, 40.0)),
        ((10.3, 30.6), (5.5, 30.6)),
    ],
)
def test_frequency_band_integral_does_not_depend_on_cropping(band, frequency_range):
    """Only the true Nyquist bin is undoubled. Cropping the grid with
    frequency_range must not turn its new last bin into a Nyquist bin, or a
    band ending at the crop edge integrates that bin twice (+3% to +15%). Nor
    may it drop the part of a band that lies in the cell of a bin just outside
    the range (up to about -3% for a band edge between these 1 Hz bins)."""
    time_series = np.random.default_rng(51).standard_normal((20000, 1, 1))
    kwargs = {
        "sampling_frequency": 1000,
        "time_window_duration": 1.0,
        "method": "power",
        "frequency_bands": {"band": band},
        "frequency_reduction": "integral",
    }
    uncropped = multitaper_connectivity(time_series, **kwargs)
    cropped = multitaper_connectivity(time_series, frequency_range=frequency_range, **kwargs)
    np.testing.assert_allclose(cropped.values, uncropped.values, rtol=1e-12)


def _one_hz_power(**kwargs):
    """Power on 1 Hz bins (0 to 500 Hz) of fixed white noise."""
    time_series = np.random.default_rng(51).standard_normal((20000, 1, 1))
    return multitaper_connectivity(
        time_series,
        sampling_frequency=1000,
        time_window_duration=1.0,
        method="power",
        **kwargs,
    )


@pytest.mark.parametrize(
    ("frequency_range", "integrated"),
    [
        pytest.param((10.0, 30.0), (9.5, 30.5), id="bounds-on-bins"),
        pytest.param((10.3, 30.6), (10.3, 30.6), id="bounds-between-bins"),
        pytest.param((10.3, 30.0), (10.3, 30.5), id="one-of-each"),
    ],
)
def test_band_reaching_past_frequency_range_is_cut_at_the_range_or_its_bins_cells(
    frequency_range, integrated
):
    """A band wider than frequency_range is integrated over the range, and out
    to the outer edge of the first or last selected bin's cell where that lies
    beyond the range's bound. A bound on a 1 Hz bin keeps that bin's whole cell
    (half a bin further); a bound between bins is where the band stops."""
    np.testing.assert_allclose(
        _one_hz_power(
            frequency_range=frequency_range,
            frequency_bands={"wide": (0.0, 100.0)},
            frequency_reduction="integral",
        ).values,
        frequency_band_reduce(
            _one_hz_power(), {"wide": integrated}, frequency_reduction="integral"
        ).values,
        rtol=1e-12,
    )


def test_band_integral_of_a_range_on_bins_equals_reducing_the_cropped_result():
    """With the range's bounds on bins, the wrapper's band integral is what
    frequency_band_reduce gives for the cropped result."""
    bands = {"wide": (0.0, 100.0)}
    np.testing.assert_allclose(
        _one_hz_power(
            frequency_range=(10.0, 30.0), frequency_bands=bands, frequency_reduction="integral"
        ).values,
        frequency_band_reduce(
            _one_hz_power(frequency_range=(10.0, 30.0)), bands, frequency_reduction="integral"
        ).values,
        rtol=1e-12,
    )


def test_band_integral_records_the_frequency_range_like_a_cropped_result():
    """The densities of an integral are left uncropped, but the result records
    the range it was limited to as a cropped result does."""
    kwargs = {"frequency_range": (10.3, 30.6), "frequency_bands": {"band": (12.0, 20.0)}}
    integral = _one_hz_power(frequency_reduction="integral", **kwargs)
    mean = _one_hz_power(**kwargs)
    names = (
        "requested_frequency_lower",
        "requested_frequency_upper",
        "selected_frequency_lower",
        "selected_frequency_upper",
        "frequency_interval_closed",
    )
    assert {name: integral.attrs[name] for name in names} == {
        "requested_frequency_lower": 10.3,
        "requested_frequency_upper": 30.6,
        "selected_frequency_lower": 11.0,
        "selected_frequency_upper": 30.0,
        "frequency_interval_closed": "both",
    }
    assert {name: mean.attrs[name] for name in names} == {
        name: integral.attrs[name] for name in names
    }


def test_band_integral_needs_two_bins_inside_frequency_range():
    """The range (10.6, 11.4) selects the 11 Hz bin only, as a crop to it would."""
    with pytest.raises(ValueError, match="at least two frequency bins"):
        _one_hz_power(
            frequency_range=(10.6, 11.4),
            frequency_bands={"band": (10.6, 11.4)},
            frequency_reduction="integral",
        )


def test_band_integral_refuses_a_band_with_no_bin_inside_frequency_range():
    """Bands are bands of the bins frequency_range selects, though the grid is
    left uncropped, and the message counts those bins."""
    with pytest.raises(
        ValueError,
        match=r"Band 'above' \(40, 50\) contains no frequency bin.*the 21 bins span 10 to 30",
    ):
        _one_hz_power(
            frequency_range=(10.0, 30.0),
            frequency_bands={"above": (40.0, 50.0)},
            frequency_reduction="integral",
        )


def test_band_integral_beside_phase_slope_index_equals_the_integral_alone():
    """phase_slope_index consumes frequency_range itself and has no frequency
    axis to integrate; power beside it is integrated as it is on its own."""
    time_series = np.random.default_rng(53).standard_normal((2000, 4, 2))
    kwargs = {
        "sampling_frequency": 1000,
        "time_window_duration": 1.0,  # 1 Hz bins, so both band edges lie between bins
        "frequency_range": (10.3, 30.6),
        "frequency_bands": {"band": (10.3, 30.6)},
        "frequency_reduction": "integral",
    }
    both = multitaper_connectivity(
        time_series, method=["power", "phase_slope_index"], **kwargs
    )
    alone = multitaper_connectivity(time_series, method="power", **kwargs)
    np.testing.assert_array_equal(both.power.values, alone.values)
    assert "band" not in both.phase_slope_index.dims


def test_band_integral_on_a_nonuniform_grid_uses_the_uncropped_grids_cells():
    """The cells of the bins frequency_range selects are those of the whole
    grid: on a logarithmic grid the first and last reach to the midpoints of
    their real neighbours, rather than where a cropped grid would mirror them."""
    rng = np.random.default_rng(54)
    frequencies = np.geomspace(1.0, 50.0, 33)
    coefficients = rng.standard_normal((12, 33, 2)) + 1j * rng.standard_normal((12, 33, 2))
    kwargs = {"frequencies": frequencies, "method": "power", "is_one_sided": True}
    low_cell = (frequencies[9] + frequencies[10]) / 2
    high_cell = (frequencies[20] + frequencies[21]) / 2
    np.testing.assert_allclose(
        fourier_connectivity(
            coefficients,
            frequency_range=(frequencies[10], frequencies[20]),
            frequency_bands={"wide": (0.0, 100.0)},
            frequency_reduction="integral",
            **kwargs,
        ).values,
        frequency_band_reduce(
            fourier_connectivity(coefficients, **kwargs),
            {"wide": (low_cell, high_cell)},
            frequency_reduction="integral",
        ).values,
        rtol=1e-12,
    )


def test_decimated_band_integral_keeps_the_part_of_the_range_between_kept_bins():
    """With frequency_decimation the kept bins start at the first bin in
    frequency_range (11, 13, ..., 29 Hz here, each a 2 Hz cell). The band's
    30 to 30.6 Hz lies in the cell of the next bin on that lattice, 31 Hz."""
    band = {"band": (10.3, 30.6)}
    lattice = _one_hz_power().isel(frequency=slice(1, None, 2))
    assert lattice.frequency.values[5] == 11.0
    np.testing.assert_allclose(
        _one_hz_power(
            frequency_range=(10.3, 30.6),
            frequency_decimation=2,
            frequency_bands=band,
            frequency_reduction="integral",
        ).values,
        frequency_band_reduce(lattice, band, frequency_reduction="integral").values,
        rtol=1e-12,
    )


def test_integral_with_frequency_range_still_refuses_other_measures():
    """Leaving power uncropped for the integral must not change which requests
    are refused, or how."""
    time_series = np.random.default_rng(53).standard_normal((2000, 4, 2))
    with pytest.raises(ValueError, match="defined only for power and cross_spectral_density"):
        multitaper_connectivity(
            time_series,
            sampling_frequency=1000,
            method=["power", "delay"],
            frequency_range=(10.3, 30.6),
            frequency_bands={"band": (10.3, 30.6)},
            frequency_reduction="integral",
        )


@pytest.mark.parametrize("frequency_reduction", ["mean", "integral"])
def test_band_reduced_dataarray_keeps_its_name(frequency_reduction):
    """The weighted sum of an integral comes back unnamed from older xarray
    (2023.1); a band result is named like its input under either reduction."""
    power = _one_hz_power()
    reduced = frequency_band_reduce(
        power, {"band": (10.0, 30.0)}, frequency_reduction=frequency_reduction
    )
    assert reduced.name == power.name == "power"


def test_frequency_band_integral_edge_bins_stay_additive_and_mean_is_unchanged():
    """Folding the DC and Nyquist cells must keep adjacent bands additive and
    must not touch ``frequency_reduction="mean"``."""
    frequencies = np.arange(0.0, 51.0, 2.0)
    density = xr.DataArray(
        np.random.default_rng(48).uniform(0.5, 3.0, frequencies.size),
        dims=("frequency",),
        coords={"frequency": frequencies},
        name="power",
        attrs={"measure": "power"},
    )
    reduced = frequency_band_reduce(
        density,
        {"low": (0.0, 0.7), "mid": (0.7, 49.2), "high": (49.2, 50.0), "all": (0.0, 50.0)},
        frequency_reduction="integral",
    )
    assert float(
        reduced.sel(band="low") + reduced.sel(band="mid") + reduced.sel(band="high")
    ) == (pytest.approx(float(reduced.sel(band="all"))))
    assert float(reduced.sel(band="all")) == pytest.approx(float(density.sum()) * 2.0)
    mean = frequency_band_reduce(density, {"all": (0.0, 50.0)})
    assert float(mean.sel(band="all")) == pytest.approx(float(density.mean()))


def test_fourier_connectivity_rejects_undeclared_unlabeled_directed_measure():
    """Without a frequency coordinate or an explicit ``is_one_sided``, two-sidedness
    cannot be verified, so Wilson-factorized measures are refused and the assumed
    sidedness is announced; a declared one-sided spectrum is refused outright."""
    coefficients = np.ones((3, 8, 2), dtype=np.complex128)
    with (
        pytest.warns(UserWarning, match="assuming a two-sided"),
        pytest.raises(ValueError, match="cannot be verified"),
    ):
        fourier_connectivity(coefficients, method="pairwise_spectral_granger_prediction")
    with pytest.raises(ValueError, match="is_one_sided=True"):
        fourier_connectivity(
            coefficients,
            method="pairwise_spectral_granger_prediction",
            is_one_sided=True,
        )


def test_fourier_connectivity_honors_explicit_two_sided_declaration():
    """``is_one_sided=False`` on unlabeled coefficients declares a full FFT-order
    spectrum, so Wilson-factorized measures run without the assumed-sidedness
    warning and match ``Connectivity.from_transform`` on the same transform."""
    rng = np.random.default_rng(325)
    time_series = rng.standard_normal((256, 4, 2))
    time_series[1:, :, 1] += 0.8 * time_series[:-1, :, 0]
    multitaper = Multitaper(time_series, sampling_frequency=200, time_halfbandwidth_product=2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = fourier_connectivity(
            multitaper.fft(),
            method="pairwise_spectral_granger_prediction",
            is_one_sided=False,
        )
    expected = Connectivity.from_transform(multitaper).pairwise_spectral_granger_prediction()

    assert result.dims == ("time", "frequency", "source", "target")
    estimator = json.loads(result.attrs["transform_parameters_json"])["estimator"]
    assert estimator["is_one_sided"] is False
    assert not estimator["one_sided_inferred"]
    # The core's [..., i, j] is the influence i -> j, the wrapper's source -> target.
    np.testing.assert_allclose(result.values, np.asarray(expected), equal_nan=True)
    driven = result.sel(source="0", target="1").values
    driver = result.sel(source="1", target="0").values
    assert np.nanmax(driven) > np.nanmax(driver)


def test_fourier_connectivity_warns_on_one_sided_input_declared_two_sided():
    """rfft-like coefficients declared ``is_one_sided=False`` would silently give
    wrong Granger values, so the missing conjugate symmetry is reported for a
    two-sided-only method; the default set and functional measures do not rely
    on the declaration's symmetry and stay silent."""
    rng = np.random.default_rng(326)
    time_series = rng.standard_normal((5, 256, 2))
    time_series[:, 1:, 1] += 0.8 * time_series[:, :-1, 0]
    one_sided = np.fft.rfft(time_series, axis=1)

    with pytest.warns(UserWarning, match="not conjugate-symmetric"):
        fourier_connectivity(
            one_sided,
            method="pairwise_spectral_granger_prediction",
            is_one_sided=False,
        )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fourier_connectivity(
            one_sided, method="magnitude_squared_coherence", is_one_sided=False
        )
        fourier_connectivity(one_sided, is_one_sided=False)


def test_fourier_connectivity_two_sided_check_tolerates_single_precision():
    """A full FFT of real signals computed in complex64 is conjugate-symmetric
    only to single-precision round-off, which must not trigger the warning."""
    rng = np.random.default_rng(327)
    time_series = rng.standard_normal((5, 256, 2)).astype(np.complex64)
    coefficients = scipy.fft.fft(time_series, axis=1)
    assert coefficients.dtype == np.complex64
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fourier_connectivity(
            coefficients,
            method="pairwise_spectral_granger_prediction",
            is_one_sided=False,
        )


_CONNECTIVITY_SETTINGS = [
    ("magnitude_squared_coherence", {"dtype": np.complex64}),
    ("pairwise_spectral_granger_prediction", {"minimum_phase_tolerance": 0.5}),
    ("pairwise_spectral_granger_prediction", {"minimum_phase_max_iterations": 1}),
]


@pytest.mark.parametrize("wrapper", ["multitaper_connectivity", "connectivity_to_xarray"])
@pytest.mark.parametrize(("method", "setting"), _CONNECTIVITY_SETTINGS, ids=str)
def test_wrappers_forward_connectivity_settings(wrapper, method, setting):
    """multitaper_connectivity and connectivity_to_xarray build the Connectivity
    with the given precision and Wilson settings, as fourier_connectivity does:
    the result is the one from_transform computes with them, which differs from
    the default's."""
    rng = np.random.default_rng(330)
    time_series = rng.standard_normal((400, 4, 2))
    multitaper = Multitaper(time_series, sampling_frequency=200, time_halfbandwidth_product=3)

    def run(**settings):
        with warnings.catch_warnings():
            # Stopping the factorization early is the point; it warns.
            warnings.simplefilter("ignore")
            if wrapper == "multitaper_connectivity":
                result = multitaper_connectivity(
                    time_series,
                    sampling_frequency=200,
                    time_halfbandwidth_product=3,
                    method=method,
                    **settings,
                )
            else:
                result = connectivity_to_xarray(multitaper, method, **settings)
            expected = getattr(Connectivity.from_transform(multitaper, **settings), method)()
        return result.values, to_host(expected)

    values, expected = run(**setting)
    np.testing.assert_array_equal(values, expected)
    default_values, _ = run()
    assert not np.array_equal(values, default_values, equal_nan=True)


@pytest.mark.parametrize("setting", ["dtype", "minimum_phase_max_iterations"])
@pytest.mark.parametrize("wrapper", ["multitaper_connectivity", "connectivity_to_xarray"])
def test_connectivity_setting_in_connectivity_kwargs_names_its_argument(wrapper, setting):
    """A Connectivity setting put in connectivity_kwargs fails before any work,
    saying it is a named argument of the wrapper, instead of reporting that the
    measure does not take it."""
    time_series = np.random.default_rng(331).standard_normal((200, 2, 2))
    value = {"dtype": np.complex64, "minimum_phase_max_iterations": 5}[setting]
    message = rf"{setting} was given inside connectivity_kwargs.*named argument of {wrapper}"
    if wrapper == "multitaper_connectivity":

        def call():
            return multitaper_connectivity(
                time_series, sampling_frequency=100, connectivity_kwargs={setting: value}
            )
    else:

        def call():
            return connectivity_to_xarray(
                Multitaper(time_series, sampling_frequency=100),
                connectivity_kwargs={setting: value},
            )

    with pytest.raises(ValueError, match=message):
        call()


def test_connectivity_setting_in_transform_kwargs_names_its_argument():
    """dtype is not a Multitaper setting; in transform_kwargs it is rejected as
    a named argument of multitaper_connectivity."""
    time_series = np.random.default_rng(332).standard_normal((200, 2, 2))
    with pytest.raises(ValueError, match="dtype was given inside transform_kwargs"):
        multitaper_connectivity(
            time_series, sampling_frequency=100, transform_kwargs={"dtype": np.complex64}
        )


def test_misspelled_wilson_setting_suggests_the_named_argument():
    """A misspelled Wilson setting is matched against the Connectivity settings
    too, not only against Multitaper's adaptive_max_iterations."""
    time_series = np.random.default_rng(333).standard_normal((200, 2, 2))
    with pytest.raises(
        TypeError,
        match=r"Did you mean 'minimum_phase_max_iterations'.*"
        r"are named arguments of multitaper_connectivity",
    ):
        multitaper_connectivity(
            time_series, sampling_frequency=100, minimum_phase_max_iteration=5
        )


def test_fourier_connectivity_allows_unlabeled_undirected_measure():
    coefficients = np.random.default_rng(314).standard_normal(
        (4, 8, 2)
    ) + 1j * np.random.default_rng(315).standard_normal((4, 8, 2))
    result = fourier_connectivity(
        coefficients, method="magnitude_squared_coherence", is_one_sided=False
    )

    # Eight two-sided bins keep DC through Nyquist on a normalized grid.
    assert result.sizes == {"time": 1, "frequency": 5, "source": 2, "target": 2}
    np.testing.assert_allclose(result.frequency, np.abs(np.fft.fftfreq(8)[:5]))
    # Magnitude-squared coherence over the 4 observations, from first principles.
    x, y = coefficients[:, :5, 0], coefficients[:, :5, 1]
    expected = np.abs(np.mean(x * y.conj(), axis=0)) ** 2 / (
        np.mean(np.abs(x) ** 2, axis=0) * np.mean(np.abs(y) ** 2, axis=0)
    )
    np.testing.assert_allclose(result.sel(source="0", target="1").values[0], expected)
    np.testing.assert_allclose(result.sel(source="1", target="0").values[0], expected)


def test_fourier_connectivity_accepts_one_sided_functional_input():
    rng = np.random.default_rng(316)
    coefficients = rng.standard_normal((4, 9, 2)) + 1j * rng.standard_normal((4, 9, 2))
    result = fourier_connectivity(
        coefficients,
        frequencies=np.linspace(0, 40, 9),
        method="magnitude_squared_coherence",
    )

    assert result.dims == ("time", "frequency", "source", "target")
    np.testing.assert_array_equal(result.frequency, np.linspace(0, 40, 9))
    estimator = json.loads(result.attrs["transform_parameters_json"])["estimator"]
    assert estimator["is_one_sided"]
    assert estimator["one_sided_inferred"]


def test_fourier_connectivity_one_sided_default_skips_two_sided_methods():
    rng = np.random.default_rng(320)
    coefficients = rng.standard_normal((5, 9, 2)) + 1j * rng.standard_normal((5, 9, 2))
    result = fourier_connectivity(
        coefficients,
        frequencies=np.linspace(0, 40, 9),
    )

    expected = tuple(
        name for name in DEFAULT_METHODS if name != "pairwise_spectral_granger_prediction"
    )
    assert tuple(result.data_vars) == expected


def test_fourier_connectivity_unlabeled_default_skips_two_sided_methods():
    """Without a frequency coordinate or a sidedness declaration two-sidedness
    cannot be verified, so the default method set must leave out the measures
    that require it instead of rejecting the caller's implicit request. An
    explicit ``is_one_sided=False`` declaration restores them."""
    rng = np.random.default_rng(324)
    # Full FFT of real signals, so the two-sided declaration below is truthful.
    coefficients = np.fft.fft(rng.standard_normal((5, 8, 2)), axis=1)
    with pytest.warns(UserWarning, match="assuming a two-sided"):
        result = fourier_connectivity(coefficients)

    expected = tuple(
        name for name in DEFAULT_METHODS if name != "pairwise_spectral_granger_prediction"
    )
    assert tuple(result.data_vars) == expected

    declared = fourier_connectivity(coefficients, is_one_sided=False)
    assert tuple(declared.data_vars) == DEFAULT_METHODS


def test_fourier_connectivity_warns_when_sidedness_is_assumed():
    """Without a frequency coordinate or an explicit flag, the two-sided
    assumption silently truncates one-sided input, so it must be announced."""
    rng = np.random.default_rng(323)
    coefficients = rng.standard_normal((6, 16, 2)) + 1j * rng.standard_normal((6, 16, 2))
    with pytest.warns(UserWarning, match="assuming a two-sided"):
        fourier_connectivity(coefficients, method="magnitude_squared_coherence")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fourier_connectivity(
            coefficients, method="magnitude_squared_coherence", is_one_sided=False
        )
        fourier_connectivity(
            coefficients, method="magnitude_squared_coherence", is_one_sided=True
        )


def test_fourier_connectivity_infers_one_bin_positive_input_as_one_sided():
    """A positive singleton coordinate must not enable directed measures."""
    rng = np.random.default_rng(322)
    coefficients = rng.standard_normal((5, 1, 2)) + 1j * rng.standard_normal((5, 1, 2))
    result = fourier_connectivity(coefficients, frequencies=np.array([10.0]))

    estimator = json.loads(result.attrs["transform_parameters_json"])["estimator"]
    assert estimator["is_one_sided"]
    assert estimator["one_sided_inferred"]
    assert "pairwise_spectral_granger_prediction" not in result

    with pytest.raises(ValueError, match="requires a full two-sided spectrum"):
        fourier_connectivity(
            coefficients,
            frequencies=np.array([10.0]),
            method="pairwise_spectral_granger_prediction",
        )


@pytest.mark.parametrize("frequency", [-10.0, -1e-12])
def test_fourier_connectivity_rejects_negative_singleton_frequency(frequency):
    """A lone negative bin cannot be a complete two-sided FFT spectrum."""
    coefficients = np.ones((5, 1, 2), dtype=np.complex128)

    with pytest.raises(ValueError, match="standard FFT order"):
        fourier_connectivity(
            coefficients,
            frequencies=np.array([frequency]),
            method="pairwise_spectral_granger_prediction",
        )


def test_fourier_connectivity_rejects_directed_one_sided_input():
    coefficients = np.ones((3, 9, 2), dtype=np.complex128)
    with pytest.raises(ValueError, match="requires a full two-sided spectrum"):
        fourier_connectivity(
            coefficients,
            frequencies=np.linspace(0, 40, 9),
            method="pairwise_spectral_granger_prediction",
        )


def test_fourier_connectivity_explicit_one_sided_without_frequencies():
    rng = np.random.default_rng(317)
    coefficients = rng.standard_normal((4, 9, 2)) + 1j * rng.standard_normal((4, 9, 2))
    result = fourier_connectivity(
        coefficients,
        method="magnitude_squared_coherence",
        is_one_sided=True,
    )

    # All 9 one-sided bins are kept (an rfft of 16 samples), labeled in
    # normalized cycles per sample from 0 to Nyquist (0.5).
    assert result.sizes["frequency"] == 9
    np.testing.assert_allclose(result.frequency, np.arange(9) / 16)
    assert result.attrs["frequency_coordinate"] == "normalized"
    assert json.loads(result.attrs["transform_parameters_json"])["estimator"]["is_one_sided"]


def test_fourier_connectivity_rejects_fftshifted_coordinate():
    coefficients = np.ones((3, 8, 2), dtype=np.complex128)
    with pytest.raises(ValueError, match="standard FFT order"):
        fourier_connectivity(
            coefficients,
            frequencies=np.fft.fftshift(np.fft.fftfreq(8, d=0.01)),
            method="magnitude_squared_coherence",
        )


@pytest.mark.parametrize("value", ["False", 0])
def test_fourier_connectivity_rejects_a_non_bool_sidedness(value):
    with pytest.raises(TypeError, match="is_one_sided must be a bool"):
        fourier_connectivity(
            np.ones((3, 8, 2), dtype=np.complex128),
            is_one_sided=value,
            method="magnitude_squared_coherence",
        )


def test_connectivity_to_xarray_shares_alternative_transform_provenance():
    data = np.random.default_rng(304).standard_normal((128, 3, 2))
    welch = connectivity_to_xarray(
        Welch(data, sampling_frequency=64, n_time_samples_per_segment=32),
        method="magnitude_squared_coherence",
    )
    morlet = connectivity_to_xarray(
        MorletWavelet(data, 64, frequencies=np.array([4.0, 8.0, 16.0])),
        method="magnitude_squared_coherence",
    )

    assert welch.attrs["transform"] == "welch"
    assert morlet.attrs["transform"] == "morlet"
    welch_parameters = json.loads(welch.attrs["transform_parameters_json"])["estimator"]
    morlet_parameters = json.loads(morlet.attrs["transform_parameters_json"])["estimator"]
    assert welch_parameters["window"] == "hann_periodic"
    assert morlet_parameters["decimation"] == 1
    assert morlet.frequency.values.tolist() == [4.0, 8.0, 16.0]
    assert morlet.valid_time_frequency.dims == ("time", "frequency")
    assert morlet_parameters["edge_mode"] == "keep"
    assert morlet_parameters["smoothing_kernel"] == "boxcar"


def test_connectivity_to_xarray_exposes_morlet_invalid_edges():
    data = np.random.default_rng(318).standard_normal((128, 2, 2))
    transform = MorletWavelet(
        data,
        64,
        frequencies=np.array([4.0, 8.0, 16.0]),
        smoothing_time=0.25,
        edge_mode="nan",
    )
    result = connectivity_to_xarray(transform, method="power")

    np.testing.assert_array_equal(result.isnull().all("source"), ~result.valid_time_frequency)


@pytest.mark.parametrize("method", ["magnitude_squared_coherence", "power", "delay"])
def test_morlet_results_keep_their_coordinate_labels(method, tmp_path):
    """Attaching the validity mask must not strip the time and frequency
    coordinates' long_name and units, and they survive a NetCDF round trip."""
    data = np.random.default_rng(320).standard_normal((192, 5, 2))
    morlet = connectivity_to_xarray(
        MorletWavelet(data, 64, frequencies=np.array([4.0, 8.0, 12.0, 16.0])),
        method=method,
    )
    reference = connectivity_to_xarray(Multitaper(data, sampling_frequency=64), method="power")
    assert "valid_time_frequency" in morlet.coords
    assert (
        morlet.time.attrs
        == reference.time.attrs
        == {
            "long_name": "Window center time",
            "units": "s",
        }
    )
    assert morlet.frequency.attrs == reference.frequency.attrs

    path = tmp_path / "morlet.nc"
    morlet.to_netcdf(path)
    with xr.open_dataarray(path) as reloaded:
        assert reloaded.time.attrs == morlet.time.attrs
        assert reloaded.frequency.attrs == morlet.frequency.attrs


def test_band_label_survives_the_validity_mask():
    """Reducing a Morlet result into bands also reduces its validity mask; that
    must not replace the labeled band coordinate with an unlabeled one."""
    data = np.random.default_rng(321).standard_normal((192, 5, 2))
    power = connectivity_to_xarray(
        MorletWavelet(data, 64, frequencies=np.array([4.0, 8.0, 12.0, 16.0])), method="power"
    )
    reduced = frequency_band_reduce(power, {"theta": (4.0, 12.0)})
    assert "valid_time_band" in reduced.coords
    assert reduced.band.attrs == {"long_name": "Frequency band"}


def test_morlet_validity_aligns_with_nonstandard_xarray_shapes():
    data = np.random.default_rng(319).standard_normal((192, 5, 2))
    # Adjacent-bin measures require a uniform grid; the band below still
    # selects the two interior bins.
    frequencies = np.array([4.0, 8.0, 12.0, 16.0])
    transform = MorletWavelet(
        data,
        64,
        frequencies=frequencies,
        smoothing_time=0.25,
        edge_mode="nan",
    )

    phase_slope = connectivity_to_xarray(transform, method="phase_slope_index")
    assert phase_slope.dims == ("time", "source", "target")
    assert phase_slope.valid_time.dims == ("time",)
    np.testing.assert_array_equal(
        phase_slope.valid_time, to_host(transform.valid_time_frequency).all(axis=1)
    )

    band = (5.0, 15.0)
    # The smoothed Morlet observations are correlated, which the significance
    # test inside delay/group_delay reports.
    with pytest.warns(UserWarning, match="assumes independent observations"):
        delay = connectivity_to_xarray(
            transform,
            method="delay",
            frequency_range=band,
        )
    np.testing.assert_array_equal(delay.frequency, [8.0, 12.0])
    np.testing.assert_array_equal(
        delay.valid_time_frequency,
        to_host(transform.valid_time_frequency)[:, [1, 2]],
    )

    with pytest.warns(UserWarning, match="assumes independent observations"):
        group_delay = connectivity_to_xarray(
            transform,
            method="group_delay",
            frequency_range=band,
        )
    assert "frequency" not in group_delay.dims
    assert group_delay.valid_time.dims == ("time",)
    np.testing.assert_array_equal(
        group_delay.valid_time,
        to_host(transform.valid_time_frequency)[:, [1, 2]].all(axis=1),
    )


def test_fourier_connectivity_default_coordinates_are_labeled_as_such():
    """Without coordinates, frequency is normalized (cycles/sample) and time is a
    window index; the coordinate metadata must say so, not claim Hz and s."""
    rng = np.random.default_rng(40)
    coefficients = rng.standard_normal((3, 8, 2)) + 1j * rng.standard_normal((3, 8, 2))
    result = fourier_connectivity(
        coefficients, method="magnitude_squared_coherence", is_one_sided=False
    )
    assert result.frequency.attrs == {
        "long_name": "Normalized frequency",
        "units": "cycles/sample",
    }
    assert result.time.attrs == {"long_name": "Window index"}


def test_fourier_connectivity_provided_coordinates_keep_physical_units():
    rng = np.random.default_rng(41)
    coefficients = rng.standard_normal((3, 8, 2)) + 1j * rng.standard_normal((3, 8, 2))
    result = fourier_connectivity(
        coefficients,
        frequencies=np.fft.fftfreq(8, 1 / 100),
        time=np.array([1.5]),  # 3-D input: (observations, frequencies, signals)
        method="magnitude_squared_coherence",
    )
    assert result.frequency.attrs == {"long_name": "Frequency", "units": "Hz"}
    assert result.time.attrs == {"long_name": "Window center time", "units": "s"}


@pytest.mark.parametrize(
    ("sampling_frequency", "delay_units", "slope_units"),
    [(None, "samples", "rad/(cycles/sample)"), (1000.0, "s", "rad/Hz")],
    ids=["normalized_frequency", "hz"],
)
def test_fourier_connectivity_delay_units_follow_the_frequency_coordinate(
    sampling_frequency, delay_units, slope_units
):
    """A delay is in the reciprocal units of frequency: samples on the default
    normalized grid (cycles/sample), seconds on a grid in Hz."""
    from spectral_connectivity.simulate import simulate_lagged_broadband

    time_series = simulate_lagged_broadband(
        [0, 1], 0.1, n_time_samples=512, n_trials=20, random_state=1
    )
    coefficients = to_host(Multitaper(time_series, sampling_frequency=1000).fft())
    frequencies = (
        None
        if sampling_frequency is None
        else np.fft.fftfreq(coefficients.shape[-2], 1 / sampling_frequency)
    )
    group = fourier_connectivity(
        coefficients, frequencies=frequencies, method="group_delay", is_one_sided=False
    )
    delay = fourier_connectivity(
        coefficients, frequencies=frequencies, method="delay", is_one_sided=False
    )

    one_sample = 1.0 if sampling_frequency is None else 1 / sampling_frequency
    value = group["group_delay"].sel(source="0", target="1").values.item()
    assert value == pytest.approx(one_sample, rel=0.01)
    assert group["group_delay"].attrs["units"] == delay_units
    assert group["group_delay_slope"].attrs["units"] == slope_units
    assert delay.attrs["units"] == delay_units


@pytest.mark.parametrize(
    "time",
    [
        np.array(["2020-01-01T00:00:00"], "M8[ns]"),
        np.array([1], "m8[s]"),
    ],
    ids=["datetime", "timedelta"],
)
def test_fourier_connectivity_rejects_non_numeric_time(time):
    """Time must be elapsed seconds, as for multitaper_connectivity; a datetime
    coordinate labeled ``units="s"`` could not even be written to NetCDF."""
    rng = np.random.default_rng(42)
    coefficients = rng.standard_normal((3, 8, 2)) + 1j * rng.standard_normal((3, 8, 2))
    with pytest.raises(TypeError, match="numeric elapsed seconds"):
        fourier_connectivity(
            coefficients,
            frequencies=np.fft.fftfreq(8, 1 / 100),
            time=time,
            method="magnitude_squared_coherence",
        )


def test_frequency_coordinate_attrs_do_not_depend_on_measure_order():
    """delay built its frequency coordinate with only ``units``; merged first,
    it dropped the long_name every other measure sets."""
    time_series = np.random.default_rng(43).standard_normal((1000, 3, 2))
    for methods in (
        ["delay", "magnitude_squared_coherence"],
        ["magnitude_squared_coherence", "delay"],
    ):
        result = multitaper_connectivity(time_series, sampling_frequency=500, method=methods)
        assert result.frequency.attrs == {"long_name": "Frequency", "units": "Hz"}, methods


def test_frequency_provenance_stays_on_the_variables_it_describes():
    """A frequency-reduced measure's band must not leak onto other variables,
    and a crop/decimation/band reduction is recorded only on the variables it
    was applied to, so an extracted variable keeps its own record."""
    time_series = np.random.default_rng(46).standard_normal((1000, 3, 2))
    dataset = multitaper_connectivity(
        time_series,
        sampling_frequency=500,
        method=["magnitude_squared_coherence", "phase_slope_index"],
        frequency_range=(4, 50),
        frequency_decimation=2,
    )
    coherence, psi = dataset.magnitude_squared_coherence, dataset.phase_slope_index
    # Both variables record the interval frequency_range selected: coherence
    # was cropped to it and phase_slope_index consumed it, uncropped and
    # undecimated. The Dataset itself carries no frequency record.
    for variable in (coherence, psi):
        assert variable.attrs["selected_frequency_lower"] == 4.0
        assert variable.attrs["selected_frequency_upper"] == 50.0
        assert variable.attrs["frequency_interval_closed"] == "both"
        assert "selected_frequency_lower" not in variable.coords
    assert coherence.attrs["frequency_decimation"] == 2
    assert "frequency_decimation" not in psi.attrs
    for key in (
        "selected_frequency_lower",
        "frequency_interval_closed",
        "frequency_decimation",
    ):
        assert key not in dataset.attrs

    banded = multitaper_connectivity(
        time_series,
        sampling_frequency=500,
        method=["magnitude_squared_coherence", "phase_slope_index"],
        frequency_bands={"beta": (13, 30)},
    )
    assert banded.magnitude_squared_coherence.attrs["frequency_reduction"] == "mean"
    for key in ("frequency_bands_json", "frequency_reduction"):
        assert key not in banded.phase_slope_index.attrs
        assert key not in banded.attrs


def test_band_reduction_keeps_dataset_coordinates_not_on_any_variable():
    power = xr.DataArray(
        np.ones((2, 5)),
        dims=("time", "frequency"),
        coords={"time": [0.0, 1.0], "frequency": np.arange(5.0)},
        name="power",
        attrs={"measure": "power"},
    )
    dataset = power.to_dataset().assign_coords(run=("run", ["a", "b", "c"]))
    reduced = frequency_band_reduce(dataset, {"low": (0.0, 2.0)})
    assert reduced.run.values.tolist() == ["a", "b", "c"]


@pytest.mark.parametrize("method", ["coherence", "jackknife"])
def test_connectivity_to_xarray_validates_the_method_name(method):
    """A typo or a non-measure helper must get the actionable error, not a raw
    AttributeError/TypeError from deep inside the formatter."""
    transform = Multitaper(
        np.random.default_rng(47).standard_normal((512, 2, 2)), sampling_frequency=256
    )
    with pytest.raises(ValueError, match="is not a known connectivity measure"):
        connectivity_to_xarray(transform, method=method)


def test_band_reduction_records_band_edges_as_coordinates():
    """Band edges are coordinates on the band axis (selectable and plottable),
    carrying the frequency coordinate's units, not only a JSON attribute."""
    power = xr.DataArray(
        np.ones((2, 11)),
        dims=("time", "frequency"),
        coords={
            "time": [0.0, 1.0],
            "frequency": (
                "frequency",
                np.arange(11.0),
                {"long_name": "Frequency", "units": "Hz"},
            ),
        },
        name="power",
        attrs={"measure": "power"},
    )
    reduced = frequency_band_reduce(power, {"theta": (4.0, 8.0), "alpha": (8.0, 10.0)})
    assert reduced.band.values.tolist() == ["theta", "alpha"]
    np.testing.assert_array_equal(reduced.band_lower, [4.0, 8.0])
    np.testing.assert_array_equal(reduced.band_upper, [8.0, 10.0])
    assert reduced.band_lower.dims == ("band",)
    assert reduced.band_lower.attrs["units"] == "Hz"
    assert reduced.where(reduced.band_lower >= 8.0, drop=True).band.values.tolist() == [
        "alpha"
    ]


@pytest.fixture
def labeled_channels():
    """Two channels with a label index plus extra per-channel coordinates."""
    return xr.DataArray(
        np.random.default_rng(48).standard_normal((1000, 2)),
        dims=("time", "channel"),
        coords={
            "channel": ["left", "right"],
            "region": ("channel", ["CA1", "PFC"]),
            "depth_um": ("channel", [120.0, 900.0]),
        },
        attrs={"units": "uV"},
    )


def test_per_signal_input_coordinates_follow_source_and_target(labeled_channels):
    """Non-index coordinates on the signal dimension (e.g. brain region) are
    carried onto the source/target axes instead of being dropped."""
    coherence = multitaper_connectivity(
        labeled_channels, sampling_frequency=500, method="magnitude_squared_coherence"
    )
    assert coherence.source_region.values.tolist() == ["CA1", "PFC"]
    assert coherence.target_region.values.tolist() == ["CA1", "PFC"]
    np.testing.assert_array_equal(coherence.source_depth_um, [120.0, 900.0])
    assert coherence.source_region.dims == ("source",)

    power = multitaper_connectivity(labeled_channels, sampling_frequency=500, method="power")
    assert power.source_region.values.tolist() == ["CA1", "PFC"]

    squeezed = multitaper_connectivity(
        labeled_channels,
        sampling_frequency=500,
        method="magnitude_squared_coherence",
        squeeze=True,
    )
    assert squeezed.source_region.item() == "CA1"
    assert squeezed.target_region.item() == "PFC"


def test_every_result_coordinate_is_described():
    """Band names and edges and per-connection group labels carry a long_name
    (and units where they have them); a channel coordinate copied from the
    input keeps the input's own attrs, which the package does not invent."""
    rng = np.random.default_rng(51)
    data = xr.DataArray(
        rng.standard_normal((1000, 4, 4)),
        dims=("time", "trial", "channel"),
        coords={
            "time": np.arange(1000) / 500,
            "channel": ["a", "b", "c", "d"],
            "region": ("channel", ["CA1", "CA1", "PFC", "PFC"], {"long_name": "Brain region"}),
            "depth": ("channel", [1.0, 2.0, 3.0, 4.0], {"units": "mm"}),
        },
    )
    power = multitaper_connectivity(
        data,
        method="power",
        frequency_bands={"theta": (4.0, 8.0)},
        frequency_reduction="integral",
    )
    assert power.band.attrs["long_name"] == "Frequency band"
    for edge in ("band_lower", "band_upper"):
        assert power[edge].attrs["units"] == "Hz"
        assert power[edge].attrs["long_name"]
    assert power.source_region.attrs == {"long_name": "Brain region"}
    assert power.source_depth.attrs == {"units": "mm"}

    coherence = multitaper_connectivity(data, method="magnitude_squared_coherence")
    assert coherence.target_region.attrs == {"long_name": "Brain region"}

    components = multitaper_connectivity(
        data, method="canonical_coherency", group_labels=["x", "x", "y", "y"]
    )
    copied = {"signal_region", "signal_depth"}  # the input's own attrs, as given
    assert components.signal_region.attrs == {"long_name": "Brain region"}
    assert components.signal_depth.attrs == {"units": "mm"}
    for name, coordinate in components.coords.items():
        if name not in copied:
            assert coordinate.attrs.get("long_name"), name


@pytest.fixture
def regions():
    """Four channels with a ``region`` coordinate grouping them in two."""
    rng = np.random.default_rng(52)
    return xr.DataArray(
        rng.standard_normal((1000, 4, 4)),
        dims=("time", "trial", "channel"),
        coords={
            "time": np.arange(1000) / 500,
            "channel": ["a", "b", "c", "d"],
            "region": ("channel", ["CA1", "CA1", "PFC", "PFC"]),
            "depth": ("channel", [1.0, 2.0, 3.0, 4.0]),
        },
    )


def test_group_labels_can_name_a_signal_coordinate(regions):
    """group_labels="region" reads the labels from the input's region
    coordinate: the result is the one the labels themselves give, whether the
    name is passed directly or in connectivity_kwargs."""
    labels = ["CA1", "CA1", "PFC", "PFC"]
    by_labels = multitaper_connectivity(
        regions, method="canonical_coherence", group_labels=labels
    )
    by_name = multitaper_connectivity(
        regions, method="canonical_coherence", group_labels="region"
    )
    in_kwargs = multitaper_connectivity(
        regions, method="canonical_coherence", connectivity_kwargs={"group_labels": "region"}
    )
    xr.testing.assert_identical(by_name, by_labels)
    xr.testing.assert_identical(in_kwargs, by_labels)
    assert by_name.source_group.values.tolist() == ["CA1", "PFC"]


def test_fourier_connectivity_group_labels_can_name_a_signal_coordinate(regions):
    multitaper = Multitaper(regions.values, sampling_frequency=500)
    coefficients = xr.DataArray(
        multitaper.fft(),
        dims=("time", "trial", "taper", "frequency", "channel"),
        coords={
            "frequency": multitaper.frequencies,
            "channel": regions.channel.values,
            "region": ("channel", regions.region.values),
        },
    )
    by_name = fourier_connectivity(
        coefficients, method="canonical_coherence", group_labels="region"
    )
    by_labels = fourier_connectivity(
        coefficients, method="canonical_coherence", group_labels=["CA1", "CA1", "PFC", "PFC"]
    )
    xr.testing.assert_identical(by_name, by_labels)


def test_group_labels_naming_a_missing_coordinate_lists_the_ones_there(regions):
    with pytest.raises(
        ValueError,
        match=r"names no 1-D coordinate along the signal dimension 'channel'.*'region', 'depth'",
    ):
        multitaper_connectivity(regions, method="canonical_coherence", group_labels="area")
    # The index labels each signal: one group per signal is not a grouping.
    with pytest.raises(ValueError, match="'channel' itself labels each signal"):
        multitaper_connectivity(regions, method="canonical_coherence", group_labels="channel")


@pytest.mark.parametrize(
    ("dims", "signal_dim"),
    [(("channel", "trial", "time"), None), (("electrode", "time", "trial"), "electrode")],
    ids=["transposed", "named_signal_dim"],
)
def test_group_labels_coordinate_follows_the_signal_axis(regions, dims, signal_dim):
    """The coordinate is read from the signal dimension, wherever it lies and
    whatever it is called, and stays aligned with the signals."""
    regions = regions.assign_coords(region=("channel", ["CA1", "PFC", "CA1", "PFC"]))
    expected = multitaper_connectivity(
        regions, method="canonical_coherence", group_labels=["CA1", "PFC", "CA1", "PFC"]
    )
    renamed = regions.rename(channel=dims[0]) if dims[0] != "channel" else regions
    result = multitaper_connectivity(
        renamed.transpose(*dims),
        method="canonical_coherence",
        group_labels="region",
        signal_dim=signal_dim,
    )
    np.testing.assert_array_equal(result.values, expected.values)
    assert result.source_group.values.tolist() == ["CA1", "PFC"]


@pytest.mark.parametrize(
    "wrapper", ["multitaper_connectivity", "fourier_connectivity", "connectivity_to_xarray"]
)
def test_group_labels_naming_a_coordinate_needs_a_dataarray(regions, wrapper):
    """A plain array, or a transform, has no coordinates to read the labels
    from; the error says to pass the labels themselves."""
    time_series = regions.values
    if wrapper == "multitaper_connectivity":

        def call():
            return multitaper_connectivity(
                time_series,
                sampling_frequency=500,
                method="canonical_coherence",
                group_labels="region",
            )
    elif wrapper == "fourier_connectivity":
        multitaper = Multitaper(time_series, sampling_frequency=500)

        def call():
            return fourier_connectivity(
                multitaper.fft(),
                frequencies=multitaper.frequencies,
                method="canonical_coherence",
                group_labels="region",
            )
    else:

        def call():
            return connectivity_to_xarray(
                Multitaper(time_series, sampling_frequency=500),
                "canonical_coherence",
                group_labels="region",
            )

    with pytest.raises(TypeError, match=r"group_labels='region' names a coordinate, but"):
        call()


def test_spectral_densities_carry_units_derived_from_the_input(labeled_channels):
    result = multitaper_connectivity(
        labeled_channels, sampling_frequency=500, method=["power", "cross_spectral_density"]
    )
    assert result.power.attrs["units"] == "(uV)^2/Hz"
    assert result.cross_spectral_density.attrs["units"] == "(uV)^2/Hz"
    unitless_input = multitaper_connectivity(
        labeled_channels.values, sampling_frequency=500, method="power"
    )
    assert "units" not in unitless_input.attrs  # unknown input units are not invented
    assert unitless_input.attrs["long_name"] == "Power spectral density"


def test_band_integral_of_a_density_drops_the_per_hz_unit(labeled_channels):
    """Integrating a density over frequency gives band power in (units)^2, not a
    density in (units)^2/Hz; a band mean is still a density."""
    kwargs = {
        "sampling_frequency": 500,
        "method": ["power", "cross_spectral_density"],
        "frequency_bands": {"theta": (4.0, 8.0)},
    }
    integral = multitaper_connectivity(
        labeled_channels, frequency_reduction="integral", **kwargs
    )
    assert integral.power.attrs["units"] == "(uV)^2"
    assert integral.power.attrs["long_name"] == "Band power"
    assert integral.cross_spectral_density.attrs["units"] == "(uV)^2"
    assert integral.cross_spectral_density.attrs["long_name"] == "Band cross-power"

    mean = multitaper_connectivity(labeled_channels, frequency_reduction="mean", **kwargs)
    assert mean.power.attrs["units"] == "(uV)^2/Hz"
    assert mean.power.attrs["long_name"] == "Power spectral density"

    unlabeled_units = frequency_band_reduce(
        multitaper_connectivity(
            labeled_channels, sampling_frequency=500, method="power"
        ).assign_attrs(units="uV**2 Hz**-1"),
        {"theta": (4.0, 8.0)},
        frequency_reduction="integral",
    )
    assert "units" not in unlabeled_units.attrs  # an unparsed density unit is not kept


@pytest.mark.parametrize(
    ("method", "units"),
    [
        ("magnitude_squared_coherence", "1"),
        ("coherence_phase", "rad"),
        ("phase_locking_value", "1"),
        ("pairwise_spectral_granger_prediction", "1"),
        ("phase_slope_index", "1"),
        ("delay", "s"),
    ],
)
def test_every_measure_has_a_long_name_and_units(method, units):
    result = multitaper_connectivity(
        np.random.default_rng(49).standard_normal((1000, 3, 2)),
        sampling_frequency=500,
        method=method,
    )
    assert result.attrs["units"] == units
    assert result.attrs["long_name"]


def test_large_array_input_attrs_are_summarized():
    """A large array attribute is recorded by shape and dtype, not copied into
    every variable's JSON (a 50,000-sample attr made a 12 MB file)."""
    data = xr.DataArray(
        np.random.default_rng(50).standard_normal((512, 2)),
        dims=("time", "channel"),
        attrs={"subject": "m1", "raw_trace": np.arange(50_000.0), "montage": [1, 2, 3]},
    )
    result = multitaper_connectivity(
        data, sampling_frequency=256, method="magnitude_squared_coherence"
    )
    record = json.loads(result.attrs["input_attrs_json"])
    assert record["subject"] == "m1"
    assert record["montage"] == [1, 2, 3]
    assert record["raw_trace"] == {"summarized_array": {"shape": [50000], "dtype": "float64"}}
    assert len(result.attrs["input_attrs_json"]) < 500


def _phases_near_pi():
    return xr.DataArray(
        np.array([3.10, -3.10, 3.12, -3.12]),
        dims=("frequency",),
        coords={"frequency": [1.0, 2.0, 3.0, 4.0]},
    )


def test_band_mean_of_phase_is_circular_when_units_are_radians():
    """Phases near +/-pi average to ~pi circularly but ~0 arithmetically; a
    renamed phase variable is recognized by its radian units."""
    phase = _phases_near_pi().rename("my_phase").assign_attrs(units="rad")
    reduced = frequency_band_reduce(phase, {"all": (1.0, 4.0)})
    assert abs(float(reduced.sel(band="all"))) == pytest.approx(np.pi, abs=0.05)


def test_band_mean_circular_can_be_requested_or_disabled_explicitly():
    phase = _phases_near_pi()  # no name, no attrs: nothing to infer from
    circular = frequency_band_reduce(phase, {"all": (1.0, 4.0)}, circular=True)
    assert abs(float(circular.sel(band="all"))) == pytest.approx(np.pi, abs=0.05)
    linear = frequency_band_reduce(
        phase.assign_attrs(units="rad"), {"all": (1.0, 4.0)}, circular=False
    )
    assert float(linear.sel(band="all")) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(
    "method",
    ["coherence_phase", "imaginary_coherency", "phase_lag_index", "weighted_phase_lag_index"],
)
def test_signed_phase_measures_are_positive_when_source_leads(method):
    """The documented convention: sel(source=a, target=b) > 0 when a leads b."""
    rng = np.random.default_rng(51)
    lag = 5  # a leads b by 10 ms at 500 Hz
    source = rng.standard_normal((10_000 + lag, 10))
    data = np.stack([source[lag:], source[:-lag]], axis=-1)
    data = data + 0.5 * rng.standard_normal(data.shape)
    result = multitaper_connectivity(
        data,
        sampling_frequency=500,
        method=method,
        signal_names=["a", "b"],
        time_window_duration=1.0,
    )
    band = result.sel(frequency=slice(5, 20)).mean(["time", "frequency"])
    assert float(band.sel(source="a", target="b")) > 0.3
    assert float(band.sel(source="b", target="a")) < -0.3


_WARNING_RNG = np.random.default_rng(41)
_THREE_SIGNALS = _WARNING_RNG.standard_normal((256, 4, 3))
_UNNAMED_TRIAL_AXIS = xr.DataArray(
    _WARNING_RNG.standard_normal((256, 4, 2)), dims=("time", "drug_dose", "channel")
)
_TWO_SIGNAL_COEFFICIENTS = _WARNING_RNG.standard_normal(
    (3, 8, 2)
) + 1j * _WARNING_RNG.standard_normal((3, 8, 2))


@pytest.mark.parametrize(
    ("match", "call"),
    [
        pytest.param(
            "squeeze=True but",
            lambda: multitaper_connectivity(
                _THREE_SIGNALS, 256, method="magnitude_squared_coherence", squeeze=True
            ),
            id="squeeze_with_many_signals",
        ),
        pytest.param(
            "squeeze=True is ignored",
            lambda: multitaper_connectivity(
                _THREE_SIGNALS,
                256,
                method=["magnitude_squared_coherence", "phase_locking_value"],
                squeeze=True,
            ),
            id="squeeze_with_many_measures",
        ),
        pytest.param(
            "no frequency coordinate",
            lambda: fourier_connectivity(
                _TWO_SIGNAL_COEFFICIENTS, method="magnitude_squared_coherence"
            ),
            id="fourier_without_frequencies",
        ),
        pytest.param(
            "Assuming DataArray dimension",
            lambda: multitaper_connectivity(_UNNAMED_TRIAL_AXIS, 256, method="power"),
            id="dataarray_role_by_elimination",
        ),
    ],
)
def test_wrapper_warnings_point_at_the_caller(match, call):
    """Warnings raised in the wrapper's helper modules name the user's call."""
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        call()
    matching = [w for w in record if match in str(w.message)]
    assert matching, [str(w.message)[:80] for w in record]
    assert {w.filename for w in matching} == {__file__}


# connectivity_to_xarray with several measures and frequency reductions. The
# multitaper settings below are multitaper_connectivity's defaults, so the
# Multitaper the tests build is the one the wrapper builds.
_MULTI_SAMPLING_FREQUENCY = 128
_MULTI_GROUPS = ["a", "a", "b", "b"]
_MULTI_BANDS = {"theta": (4.0, 8.0), "beta": (13.0, 30.0)}


@pytest.fixture(scope="module")
def four_signals():
    """Four signals sharing a 20 Hz rhythm in noise, shape (256, 3, 4)."""
    rng = np.random.default_rng(530)
    time = np.arange(256) / _MULTI_SAMPLING_FREQUENCY
    shifted = np.sin(2 * np.pi * 20 * time[:, np.newaxis, np.newaxis] - np.linspace(0, 1, 4))
    return shifted + rng.standard_normal((256, 3, 4))


def _recorded(call):
    """``call()`` and the messages of the warnings it raised."""
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        result = call()
    return result, [str(w.message) for w in record]


@pytest.mark.parametrize(
    "request_settings",
    [
        pytest.param({"method": list(DEFAULT_METHODS)}, id="default-set"),
        pytest.param({"method": ["power", "coherency", "imaginary_coherence"]}, id="several"),
        pytest.param(
            {
                "method": ["canonical_coherence", "power", "canonical_coherency"],
                "group_labels": _MULTI_GROUPS,
            },
            id="group-measures",
        ),
        pytest.param(
            {
                "method": ["phase_slope_index", "delay", "coherence_phase"],
                "frequency_range": (10.0, 40.0),
            },
            id="range-consumers",
        ),
        pytest.param(
            {
                "method": ["power", "magnitude_squared_coherence"],
                "frequency_range": (10.0, 40.0),
                "frequency_decimation": 2,
            },
            id="range-decimation",
        ),
        pytest.param(
            {
                "method": ["coherence_phase", "coherency", "phase_slope_index"],
                "frequency_bands": _MULTI_BANDS,
            },
            id="bands-mean",
        ),
        pytest.param(
            {
                "method": ["power", "cross_spectral_density"],
                "frequency_bands": {"beta": (12.6, 29.3)},
                "frequency_reduction": "integral",
                "frequency_range": (10.0, 35.0),
            },
            id="bands-integral-range",
        ),
        pytest.param(
            {
                "method": ["coherence_magnitude", "pairwise_spectral_granger_prediction"],
                "dtype": np.complex64,
                "minimum_phase_max_iterations": 40,
            },
            id="deprecated-name-and-wilson-settings",
        ),
        pytest.param(
            {"method": ["power", "magnitude_squared_coherence"], "squeeze": True},
            id="squeeze-ignored-for-a-list",
        ),
    ],
)
@pytest.mark.parametrize("windowed", [False, True], ids=["one-window", "windowed"])
def test_connectivity_to_xarray_of_a_list_equals_multitaper_connectivity(
    four_signals, request_settings, windowed
):
    """A list of measures gives multitaper_connectivity's Dataset for the same
    Multitaper: the same variables, coordinates, attributes, and warnings."""
    windows = {"time_window_duration": 0.5, "time_window_step": 0.25} if windowed else {}
    expected, expected_warnings = _recorded(
        lambda: multitaper_connectivity(
            four_signals, _MULTI_SAMPLING_FREQUENCY, **windows, **request_settings
        )
    )
    actual, actual_warnings = _recorded(
        lambda: connectivity_to_xarray(
            Multitaper(four_signals, _MULTI_SAMPLING_FREQUENCY, **windows),
            **request_settings,
        )
    )
    assert isinstance(actual, xr.Dataset)
    assert actual.identical(expected)
    assert actual_warnings == expected_warnings


def test_connectivity_to_xarray_of_one_name_in_a_list_is_a_dataset(four_signals):
    """A one-name list returns a Dataset whose variable is the string's result."""
    transform = Multitaper(four_signals, _MULTI_SAMPLING_FREQUENCY)
    alone = connectivity_to_xarray(transform, "magnitude_squared_coherence")
    listed = connectivity_to_xarray(transform, ["magnitude_squared_coherence"])
    assert isinstance(alone, xr.DataArray)
    assert isinstance(listed, xr.Dataset)
    assert listed["magnitude_squared_coherence"].identical(alone)


@pytest.fixture(scope="module")
def morlet(four_signals):
    """A Morlet transform on a uniform 4 Hz grid, with invalid edges as NaN."""
    return MorletWavelet(
        four_signals,
        _MULTI_SAMPLING_FREQUENCY,
        frequencies=np.arange(4.0, 33.0, 4.0),
        n_cycles=3.0,
        edge_mode="nan",
    )


def test_connectivity_to_xarray_bands_equal_frequency_band_reduce(morlet):
    """Bands through connectivity_to_xarray reduce its unreduced result, the
    validity mask included."""
    methods = ["power", "magnitude_squared_coherence", "coherence_phase"]
    reduced = connectivity_to_xarray(morlet, methods, frequency_bands=_MULTI_BANDS)
    expected = frequency_band_reduce(connectivity_to_xarray(morlet, methods), _MULTI_BANDS)
    assert reduced.identical(expected)
    assert "valid_time_band" in reduced.coords
    assert reduced.sizes["band"] == 2


def test_connectivity_to_xarray_integrates_a_band_on_the_uncropped_grid(morlet):
    """With frequency_range, a band inside the range integrates as without it,
    though its edges fall between bins; integrating the cropped result does not.

    The range (13, 27) keeps the bins 16, 20, and 24 Hz of the 4 Hz grid; the
    band's ends lie in the cells of the 12 and 28 Hz bins, which it drops.
    """
    band = {"beta": (13.0, 27.0)}
    settings = {"frequency_bands": band, "frequency_reduction": "integral"}
    with_range = connectivity_to_xarray(
        morlet, "power", frequency_range=(13.0, 27.0), **settings
    )
    without_range = connectivity_to_xarray(morlet, "power", **settings)
    np.testing.assert_allclose(with_range, without_range, rtol=1e-12)
    np.testing.assert_array_equal(with_range.valid_time_band, without_range.valid_time_band)
    cropped_first = frequency_band_reduce(
        connectivity_to_xarray(morlet, "power", frequency_range=(13.0, 27.0)),
        band,
        frequency_reduction="integral",
    )
    finite = np.isfinite(without_range.values)
    assert finite.any()
    assert np.all(cropped_first.values[finite] < without_range.values[finite])


def test_connectivity_to_xarray_decimates_after_the_range(morlet):
    decimated = connectivity_to_xarray(
        morlet, "coherency", frequency_range=(8.0, 28.0), frequency_decimation=2
    )
    cropped = connectivity_to_xarray(morlet, "coherency", frequency_range=(8.0, 28.0))
    np.testing.assert_array_equal(decimated.frequency, [8.0, 16.0, 24.0])
    # Equal values and coordinates; the attributes record the decimation.
    xr.testing.assert_equal(decimated, cropped.isel(frequency=slice(None, None, 2)))
    assert decimated.attrs["frequency_decimation"] == 2
    # The validity mask is decimated with the bins it describes.
    np.testing.assert_array_equal(
        decimated.valid_time_frequency,
        to_host(morlet.valid_time_frequency)[:, [1, 3, 5]],
    )


@pytest.mark.parametrize(
    "settings",
    [
        pytest.param({"method": "power", "frequency_decimation": 0}, id="decimation-zero"),
        pytest.param(
            {"method": "phase_slope_index", "frequency_bands": _MULTI_BANDS},
            id="bands-of-psi",
        ),
        pytest.param(
            {"method": ["phase_slope_index", "group_delay"], "frequency_decimation": 2},
            id="decimation-of-psi-and-group-delay",
        ),
        pytest.param(
            {
                "method": "magnitude_squared_coherence",
                "frequency_bands": _MULTI_BANDS,
                "frequency_reduction": "integral",
            },
            id="integral-of-coherence",
        ),
        pytest.param(
            {"method": "delay", "connectivity_kwargs": {"frequency_range": (1.0, 2.0)}},
            id="range-in-kwargs",
        ),
        pytest.param(
            {"method": ["power", "coherency"], "connectivity_kwargs": {"pairs": [(0, 1)]}},
            id="setting-of-one-measure",
        ),
        pytest.param({"method": ["power", "canonical_coherence"]}, id="missing-group-labels"),
        pytest.param(
            {"method": ["power", "coherency"], "group_labels": _MULTI_GROUPS},
            id="labels-without-group-measure",
        ),
        pytest.param({"method": ["power", "phase_lock_value"]}, id="unknown-name"),
        pytest.param(
            {"method": ["coherence_magnitude", "magnitude_squared_coherence"]},
            id="deprecated-and-canonical-name",
        ),
    ],
)
def test_connectivity_to_xarray_rejects_what_multitaper_connectivity_rejects(
    four_signals, settings
):
    """The same request fails with the same error through both wrappers, except
    for each wrapper's own hint on where transform settings go."""
    with pytest.raises((ValueError, TypeError)) as expected:
        multitaper_connectivity(four_signals, _MULTI_SAMPLING_FREQUENCY, **settings)
    with pytest.raises(expected.type) as actual:
        connectivity_to_xarray(Multitaper(four_signals, _MULTI_SAMPLING_FREQUENCY), **settings)
    transform_hint = " is a transform setting, "
    assert (
        str(actual.value).split(transform_hint)[0]
        == str(expected.value).split(transform_hint)[0]
    )


def test_connectivity_to_xarray_lists_measures_of_a_custom_transform(four_signals):
    """A list from a transform without coordinates or a settings record gives
    each measure's own result, labeled in normalized frequency and window index,
    with the provenance that says so."""
    coefficients = Multitaper(four_signals, _MULTI_SAMPLING_FREQUENCY).fft()
    methods = ["power", "magnitude_squared_coherence", "delay"]
    transform = PrecomputedTransform(coefficients, None, None)
    combined = connectivity_to_xarray(transform, methods)
    assert list(combined.data_vars) == methods
    for name in methods:
        assert combined[name].identical(connectivity_to_xarray(transform, name)), name
    assert combined.attrs["transform"] == "PrecomputedTransform"
    assert combined.attrs["frequency_units"] == "cycles/sample"
    assert combined.attrs["nyquist_frequency"] == 0.5
    assert combined.attrs["time_coordinate"] == "index"


def test_connectivity_to_xarray_lists_morlet_measures_with_their_validity(morlet):
    """Each variable of a Morlet list keeps the validity of its own bins: the
    frequency-resolved ones of the range, phase_slope_index every bin it sums."""
    band = (11.0, 25.0)
    validity = to_host(morlet.valid_time_frequency)
    frequencies = to_host(morlet.frequencies)
    in_band = (frequencies >= band[0]) & (frequencies <= band[1])
    combined = connectivity_to_xarray(
        morlet,
        ["magnitude_squared_coherence", "delay", "phase_slope_index"],
        frequency_range=band,
    )
    np.testing.assert_array_equal(combined.frequency, [12.0, 16.0, 20.0, 24.0])
    np.testing.assert_array_equal(combined.valid_time_frequency, validity[:, in_band])
    np.testing.assert_array_equal(combined.valid_time, validity[:, in_band].all(axis=1))
    assert not combined.valid_time_frequency.all()  # the NaN edges are marked
    assert "valid_time" in combined["phase_slope_index"].coords
    assert "frequency" not in combined["phase_slope_index"].dims
