import inspect
import json
import warnings

import numpy as np
import pytest
import xarray as xr
from pytest import mark

from spectral_connectivity import MorletWavelet, Multitaper, Welch
from spectral_connectivity.connectivity import Connectivity
from spectral_connectivity.wrapper import (
    DEFAULT_METHODS,
    _canonical_json,
    _json_compatible,
    _MeasureSpec,
    _netcdf_provenance_value,
    _reject_unmaterialized_backing,
    connectivity_to_xarray,
    fourier_connectivity,
    frequency_band_reduce,
    multitaper_connectivity,
)


@mark.parametrize("time_window_duration", [0.1, 0.2, 2.4, 0.16])
def test_multitaper_coherence_magnitude(time_window_duration):
    rng = np.random.default_rng(42)
    sampling_frequency = 1500
    start_time, end_time = 0, 4.8
    n_trials, n_signals = 10, 2
    n_time_samples = int((end_time - start_time) * sampling_frequency) + 1
    time_series = rng.random(size=(n_time_samples, n_trials, n_signals))
    expected_time = np.arange(start_time, end_time, time_window_duration)

    if not np.allclose(expected_time[-1] + time_window_duration, end_time):
        expected_time = expected_time[:-1]
    # Windows are labeled by their center time, not their start.
    expected_time = expected_time + (
        round(time_window_duration * sampling_frequency) - 1
    ) / (2 * sampling_frequency)

    m = multitaper_connectivity(
        time_series,
        method="coherence_magnitude",
        sampling_frequency=sampling_frequency,
        time_window_duration=time_window_duration,
    )

    assert np.allclose(m.time.values, expected_time)
    assert not (m.values == 0).all()
    assert not (np.isnan(m.values)).all()


@mark.parametrize(
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
        values[..., 1, 0] = 7.0  # native convention: group 0 -> group 1
        return values, np.array([10, 20])

    monkeypatch.setattr(
        Connectivity, "blockwise_spectral_granger_prediction", blockwise
    )
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
        n_components=2,
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
    assert result.connection_seed_group.values.tolist() == [10]
    assert result.connection_target_group.values.tolist() == [20]
    assert result.group_membership.sel(group=10, signal="a").item()


def test_connectivity_to_xarray_exposes_global_components():
    transform = Multitaper(
        np.random.default_rng(44).standard_normal((256, 6, 3)),
        sampling_frequency=128,
    )
    result = connectivity_to_xarray(
        transform,
        method="global_coherence",
        signal_names=["a", "b", "c"],
        max_rank=2,
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
        frequencies_of_interest=(8, 40),
        n_range=1,
    )
    psi = connectivity_to_xarray(
        transform,
        method="phase_slope_index",
        signal_names=["a", "b"],
        frequencies_of_interest=(8, 40),
    )
    group_delay = connectivity_to_xarray(
        transform,
        method="group_delay",
        signal_names=["a", "b"],
        frequencies_of_interest=(8, 40),
    )

    assert delay.dims == ("time", "frequency", "candidate", "source", "target")
    assert delay.candidate.values.tolist() == [-1, 0, 1]
    assert np.all((delay.frequency > 8) & (delay.frequency < 40))
    assert psi.dims == ("time", "source", "target")
    assert psi.frequency_band_lower.item() == 8
    assert psi.frequency_band_upper.item() == 40
    assert set(group_delay.data_vars) == {
        "group_delay",
        "group_delay_slope",
        "group_delay_r_value",
    }
    assert group_delay.group_delay.attrs["units"] == "s"


def test_frequency_operations_reject_already_reduced_output():
    data = np.random.default_rng(428).standard_normal((256, 4, 2))
    with pytest.raises(ValueError, match="no frequency dimension"):
        multitaper_connectivity(
            data,
            sampling_frequency=128,
            method="phase_slope_index",
            connectivity_kwargs={"frequencies_of_interest": (8, 40)},
            frequency_range=(10, 30),
        )


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


@mark.parametrize("n_signals", range(2, 5))
def test_multitaper_n_signals(n_signals):
    """
    Test dataarray interface
    """
    np.random.default_rng(42)
    time_window_duration = 0.1
    sampling_frequency = 1500
    start_time, end_time = 0, 4.8
    n_trials, n_signals = 10, n_signals
    n_time_samples = int((end_time - start_time) * sampling_frequency) + 1
    # time_series = np.zeros((n_time_samples, n_trials, n_signals))
    time_series = np.random.random(size=(n_time_samples, n_trials, n_signals))
    expected_time = np.arange(start_time, end_time, time_window_duration)

    if not np.allclose(expected_time[-1] + time_window_duration, end_time):
        expected_time = expected_time[:-1]
    # Windows are labeled by their center time, not their start.
    expected_time = expected_time + (
        round(time_window_duration * sampling_frequency) - 1
    ) / (2 * sampling_frequency)

    for method in (*DEFAULT_METHODS, "coherency"):
        m = multitaper_connectivity(
            time_series,
            method=method,
            sampling_frequency=sampling_frequency,
            time_window_duration=time_window_duration,
        )
        assert np.allclose(m.time.values, expected_time)
        assert not (m.values == 0).all()
        assert not (np.isnan(m.values)).all()


@mark.parametrize("n_signals", range(2, 5))
def test_multitaper_connectivities_n_signals(n_signals):
    np.random.default_rng(42)
    time_window_duration = 0.1
    sampling_frequency = 1500
    start_time, end_time = 0, 4.8
    n_trials, n_signals = 10, n_signals
    n_time_samples = int((end_time - start_time) * sampling_frequency) + 1
    time_series = np.random.random(size=(n_time_samples, n_trials, n_signals))
    expected_time = np.arange(start_time, end_time, time_window_duration)

    if not np.allclose(expected_time[-1] + time_window_duration, end_time):
        expected_time = expected_time[:-1]
    # Windows are labeled by their center time, not their start.
    expected_time = expected_time + (
        round(time_window_duration * sampling_frequency) - 1
    ) / (2 * sampling_frequency)

    cons = multitaper_connectivity(
        time_series,
        sampling_frequency=sampling_frequency,
        time_window_duration=time_window_duration,
    )
    for mea in cons.data_vars:
        assert np.allclose(cons[mea].time.values, expected_time)
        assert not (cons[mea].values == 0).all()
        assert not (np.isnan(cons[mea].values)).all()

    cons = multitaper_connectivity(
        time_series,
        method=["coherence_magnitude"],
        sampling_frequency=sampling_frequency,
        time_window_duration=time_window_duration,
    )
    mea = "coherence_magnitude"
    assert np.allclose(cons[mea].time.values, expected_time)
    assert not (cons[mea].values == 0).all()
    assert not (np.isnan(cons[mea].values)).all()


def test_frequencies():
    rng = np.random.default_rng(42)
    n_time_samples, n_trials, n_signals = 64, 10, 2
    time_series = rng.random((n_time_samples, n_trials, n_signals))
    # n_fft_samples must be >= the window length (here the full 64 samples),
    # otherwise the FFT would silently truncate the signal.
    n_fft_samples = 64
    sampling_frequency = 1000

    cons = multitaper_connectivity(
        time_series,
        sampling_frequency=sampling_frequency,
        time_window_duration=None,
        n_fft_samples=n_fft_samples,
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
        method="coherence_magnitude",
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
        data_2d, sampling_frequency=500, method="coherence_magnitude"
    )
    result_3d = multitaper_connectivity(
        data_3d, sampling_frequency=500, method="coherence_magnitude"
    )
    np.testing.assert_allclose(result_2d.values, result_3d.values, equal_nan=True)


@pytest.mark.parametrize("with_trial_dimension", [False, True])
def test_dataarray_input_preserves_signal_labels(with_trial_dimension):
    """A DataArray's final dimension labels carry through to the result."""
    rng = np.random.default_rng(4)
    if with_trial_dimension:
        data = rng.standard_normal((256, 3, 2))
        dims = ("sample", "trial", "channel")
    else:
        data = rng.standard_normal((256, 2))
        dims = ("sample", "channel")
    labeled = xr.DataArray(
        data,
        dims=dims,
        coords={"channel": ["left", "right"]},
    )

    actual = multitaper_connectivity(
        labeled,
        sampling_frequency=256,
        method="coherence_magnitude",
    )
    expected = multitaper_connectivity(
        data,
        sampling_frequency=256,
        method="coherence_magnitude",
        signal_names=["left", "right"],
    )

    xr.testing.assert_identical(actual, expected)


def test_explicit_signal_names_override_dataarray_coordinate():
    data = xr.DataArray(
        np.random.default_rng(5).standard_normal((256, 2)),
        dims=("sample", "channel"),
        coords={"channel": ["left", "right"]},
    )

    result = multitaper_connectivity(
        data,
        sampling_frequency=256,
        method="coherence_magnitude",
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
            data, sampling_frequency=256, method="coherence_magnitude"
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
            data, sampling_frequency=256, method="coherence_magnitude"
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
            data, sampling_frequency=256, method="coherence_magnitude"
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
        data, sampling_frequency=256, method="coherence_magnitude"
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
        data, sampling_frequency=256, method="coherence_magnitude"
    )

    assert result.source.dtype == np.dtype("datetime64[ns]")
    np.testing.assert_array_equal(result.source.values, labels)


def test_dataarray_named_dimensions_are_transposed_automatically():
    """Dimension names, rather than input position, determine semantic axes."""
    raw = np.random.default_rng(10).standard_normal((256, 2))
    data = xr.DataArray(
        raw.T,
        dims=("channel", "time"),
        coords={"channel": ["left", "right"]},
    )

    actual = multitaper_connectivity(
        data, sampling_frequency=256, method="coherence_magnitude"
    )
    expected = multitaper_connectivity(
        raw,
        sampling_frequency=256,
        method="coherence_magnitude",
        signal_names=["left", "right"],
    )

    xr.testing.assert_identical(actual, expected)


def test_dataarray_swapped_time_and_trial_dims_are_transposed_automatically():
    """Named trial/time axes are normalized before entering numerical code."""
    raw = np.random.default_rng(11).standard_normal((256, 4, 2))
    data = xr.DataArray(
        raw.transpose(1, 0, 2),
        dims=("trial", "time", "channel"),
        coords={"channel": ["left", "right"]},
    )

    actual = multitaper_connectivity(
        data, sampling_frequency=256, method="coherence_magnitude"
    )
    expected = multitaper_connectivity(
        raw,
        sampling_frequency=256,
        method="coherence_magnitude",
        signal_names=["left", "right"],
    )

    xr.testing.assert_identical(actual, expected)


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
    raw = np.random.default_rng(40).standard_normal((256, 4, 2))
    data = xr.DataArray(
        raw,
        dims=("time", "drug_dose", "channel"),
        coords={"channel": ["left", "right"]},
    )
    with pytest.warns(UserWarning, match=r"Assuming.*'drug_dose'.*trial axis"):
        result = multitaper_connectivity(
            data, sampling_frequency=256, method="coherence_magnitude"
        )
    # Naming the role silences the warning and gives the same result.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        explicit = multitaper_connectivity(
            data,
            sampling_frequency=256,
            method="coherence_magnitude",
            trial_dim="drug_dose",
        )
    xr.testing.assert_identical(result, explicit)


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
    with pytest.raises(ValueError, match="sampling_frequency must be a positive"):
        multitaper_connectivity(data, sampling_frequency=bad_rate, method="power")


def test_dataarray_non_scalar_start_time_is_rejected():
    raw = np.random.default_rng(47).standard_normal((128, 2))
    data = xr.DataArray(
        raw,
        dims=("time", "channel"),
        coords={"time": np.arange(raw.shape[0]) / 64.0},
    )
    with pytest.raises(ValueError, match="requires scalar start_time"):
        multitaper_connectivity(
            data, sampling_frequency=64, method="power", start_time=[0, 1]
        )


def test_measure_spec_rejects_inconsistent_field_combinations():
    """Illegal capability combinations are unrepresentable, not merely unused."""
    with pytest.raises(ValueError, match="is_directed requires"):
        _MeasureSpec("power", is_directed=True)
    with pytest.raises(ValueError, match="unsupported measure cannot be a default"):
        _MeasureSpec("unsupported", is_default=True)


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
        start_time=time[0],
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
    time = (start_time + np.arange(raw.shape[0]) / sampling_frequency).astype(
        np.float32
    )
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
    time = np.datetime64("2025-01-01") + np.arange(raw.shape[0]) * np.timedelta64(
        1, "s"
    )
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
            start_time=0,
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

    inferred = multitaper_connectivity(data, method="coherence_magnitude")
    explicit = multitaper_connectivity(
        data, sampling_frequency=sampling_frequency, method="coherence_magnitude"
    )

    xr.testing.assert_identical(inferred, explicit)
    # The frequency grid reflects the inferred rate (Nyquist = fs / 2).
    assert float(inferred.frequency.max()) == pytest.approx(sampling_frequency / 2)


def test_dataarray_infers_rate_from_precise_float32_time_coordinate():
    """Float32 can infer a rate when a zero-based axis retains enough precision."""
    sampling_frequency = 1000
    raw = np.random.default_rng(35).standard_normal((256, 2))
    time = (np.arange(raw.shape[0]) / sampling_frequency).astype(np.float32)
    data = xr.DataArray(raw, dims=("time", "channel"), coords={"time": time})

    result = multitaper_connectivity(data, method="power")

    assert result.attrs["mt_sampling_frequency"] == pytest.approx(
        sampling_frequency, rel=1e-6
    )


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
            method="coherence_magnitude",
        )


def test_dataarray_without_time_coordinate_requires_sampling_frequency():
    """A DataArray with no numeric time coordinate cannot infer a rate."""
    data = xr.DataArray(
        np.random.default_rng(32).standard_normal((256, 2)),
        dims=("time", "channel"),
        coords={"channel": ["left", "right"]},
    )
    with pytest.raises(ValueError, match="sampling_frequency is required"):
        multitaper_connectivity(data, method="coherence_magnitude")


def test_dataarray_sample_coordinate_cannot_infer_sampling_frequency():
    """Integer sample numbers carry no time scale, so inference is refused."""
    data = xr.DataArray(
        np.random.default_rng(33).standard_normal((256, 2)),
        dims=("sample", "channel"),
        coords={"sample": np.arange(256)},
    )
    with pytest.raises(ValueError, match="Cannot infer sampling_frequency"):
        multitaper_connectivity(data, method="coherence_magnitude")


def test_dataarray_nonuniform_time_coordinate_cannot_infer_sampling_frequency():
    """A non-uniform time axis has no single rate to infer."""
    raw = np.random.default_rng(34).standard_normal((128, 2))
    times = np.arange(raw.shape[0]) / 64.0
    times[50:] += 0.5  # break regular spacing
    data = xr.DataArray(raw, dims=("time", "channel"), coords={"time": times})
    with pytest.raises(ValueError, match="not uniformly spaced"):
        multitaper_connectivity(data, method="coherence_magnitude")


def test_dataarray_role_absent_at_dimensionality_is_rejected_with_reshape_hint():
    """A role with no slot at this ndim (trial in 2-D) points at the shape, not transpose."""
    data = xr.DataArray(
        np.random.default_rng(13).standard_normal((256, 4)),
        dims=("time", "trial"),
    )
    with pytest.raises(ValueError, match="has no trial axis"):
        multitaper_connectivity(
            data, sampling_frequency=256, method="coherence_magnitude"
        )


def test_dataarray_dask_backing_is_rejected():
    """A dask-backed DataArray raises with a materialization hint."""
    dask_array = pytest.importorskip("dask.array")
    data = xr.DataArray(
        dask_array.from_array(
            np.random.default_rng(12).standard_normal((256, 2)), chunks=(128, 2)
        ),
        dims=("sample", "channel"),
    )
    with pytest.raises(TypeError, match="dask-backed"):
        multitaper_connectivity(
            data, sampling_frequency=256, method="coherence_magnitude"
        )


def test_dask_protocol_backing_is_rejected_without_optional_dependency():
    """Dask detection uses its collection protocol, not a module-name heuristic."""

    class LazyArray:
        def __dask_graph__(self):
            return {}

    with pytest.raises(TypeError, match="dask-backed"):
        _reject_unmaterialized_backing(LazyArray())


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
        data, sampling_frequency=256, method="coherence_magnitude"
    )
    assert json.loads(result.attrs["input_attrs_json"]) == input_attrs
    # Namespacing keeps caller metadata from clobbering our own provenance.
    assert result.attrs["package"] == "spectral_connectivity"

    # Multi-method Dataset result: carried on the Dataset and each variable.
    ds = multitaper_connectivity(
        data,
        sampling_frequency=256,
        method=["coherence_magnitude", "imaginary_coherence"],
    )
    assert json.loads(ds.attrs["input_attrs_json"]) == input_attrs
    assert (
        json.loads(ds["coherence_magnitude"].attrs["input_attrs_json"]) == input_attrs
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
    assert {key for key in result.attrs if key.startswith("input_")} == {
        "input_attrs_json"
    }

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
        method="coherence_magnitude",
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
        assert "float64" not in encoded and "array" not in encoded

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
        assert json.loads(_netcdf_provenance_value(float("nan"))) == {
            "nonfinite_float": "nan"
        }


def test_structured_and_nonfinite_kwargs_survive_netcdf(tmp_path, monkeypatch):
    """Unusual measure kwargs serialize and round-trip through NetCDF."""
    rng = np.random.default_rng(13)
    m = Multitaper(rng.standard_normal((256, 4, 3)), sampling_frequency=500)

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

    da = connectivity_to_xarray(
        m,
        method="stub_measure",
        nested={"b": 1, "a": 2},
        weights=np.array([1.0, 2.0]),
        cutoff=float("inf"),
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
        method="coherence_magnitude",
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
        result = Multitaper(
            time_series, sampling_frequency=500, fft_workers=workers
        ).fft()
        np.testing.assert_allclose(result, reference, rtol=1e-10, atol=1e-12)

    # The wrapper forwards fft_workers via **kwargs; results are equivalent.
    baseline = multitaper_connectivity(
        time_series, sampling_frequency=500, method="coherence_magnitude"
    )
    parallel = multitaper_connectivity(
        time_series,
        sampling_frequency=500,
        method="coherence_magnitude",
        fft_workers=-1,
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

    def spying_fft(recorded):
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
    assert transform_workers(Multitaper(time_series, sampling_frequency=500)) == (
        "MISSING"
    )

    # Explicit value is forwarded verbatim.
    assert (
        transform_workers(
            Multitaper(time_series, sampling_frequency=500, fft_workers=3)
        )
        == 3
    )

    # Forwarded through the wrapper's **kwargs (which reach Multitaper).
    recorded = []
    primer = Multitaper(time_series, sampling_frequency=500)
    _ = primer.tapers  # warm the DPSS fft path unrelated to the transform
    with patch.object(transforms, "fft", spying_fft(recorded)):
        multitaper_connectivity(
            time_series,
            sampling_frequency=500,
            method="coherence_magnitude",
            fft_workers=2,
        )
    assert 2 in recorded  # the taper-projection FFT received workers=2

    # On the GPU backend `workers` is not forwarded (cupyx's FFT has no such
    # parameter). Simulate GPU on the CPU by patching the backend check.
    gpu_multitaper = Multitaper(time_series, sampling_frequency=500, fft_workers=-1)
    _ = gpu_multitaper.tapers
    recorded = []
    with patch.object(transforms, "is_gpu_enabled", lambda: True):
        with patch.object(transforms, "fft", spying_fft(recorded)):
            gpu_multitaper.fft()
    assert recorded == ["MISSING"]


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
            raise TypeError("Implicit conversion to a NumPy array is not allowed.")

    device = _DeviceLike(np.arange(5.0))
    with pytest.raises(TypeError):
        np.asarray(device)  # guards the premise: implicit conversion fails
    np.testing.assert_array_equal(to_numpy(device), np.arange(5.0))


def test_multi_method_shares_single_fft():
    """A multi-method call computes the FFT once, not once per measure.

    ``multitaper_connectivity`` builds one shared ``Connectivity`` and reuses it
    across every requested measure. Since ``Connectivity.from_multitaper`` calls
    the (uncached) ``Multitaper.fft``, the FFT must run exactly once regardless
    of how many measures are requested.
    """
    from spectral_connectivity.transforms import Multitaper

    rng = np.random.default_rng(0)
    time_series = rng.standard_normal((512, 4, 3))
    methods = ["coherence_magnitude", "coherence_phase", "imaginary_coherence"]

    original_fft = Multitaper.fft
    calls = {"n": 0}

    def counting_fft(self):
        calls["n"] += 1
        return original_fft(self)

    Multitaper.fft = counting_fft
    try:
        multitaper_connectivity(time_series, sampling_frequency=500, method=methods)
    finally:
        Multitaper.fft = original_fft

    assert calls["n"] == 1, (
        f"FFT computed {calls['n']} times for {len(methods)} methods"
    )


def test_shared_connectivity_matches_per_method_construction():
    """Sharing one Connectivity yields identical results to building per method.

    Reusing a single instance only avoids recomputation; it must not change any
    numbers. Results must match a fresh ``Connectivity.from_multitaper`` per
    measure bit-for-bit.
    """
    import xarray as xr

    from spectral_connectivity.transforms import Multitaper
    from spectral_connectivity.wrapper import connectivity_to_xarray

    rng = np.random.default_rng(1)
    time_series = rng.standard_normal((512, 4, 3))
    methods = ["coherence_magnitude", "coherence_phase", "imaginary_coherence"]

    shared = multitaper_connectivity(
        time_series, sampling_frequency=500, method=methods
    )

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

    Method discovery excludes complex-valued coherency so the default remains
    portable across all supported xarray versions and NetCDF engines.
    """
    rng = np.random.default_rng(0)
    ds = multitaper_connectivity(
        rng.standard_normal((512, 5, 2)), sampling_frequency=500
    )
    assert "coherency" not in ds.data_vars
    assert not any(np.iscomplexobj(da.values) for da in ds.data_vars.values())
    path = tmp_path / "default.nc"
    ds.to_netcdf(path)
    assert path.exists()


def test_default_method_set_is_explicit_and_ordered():
    """method=None uses the explicit, ordered DEFAULT_METHODS allowlist.

    The exact tuple (including order) is locked: xarray preserves insertion
    order, so the default Dataset's variable/iteration/serialization order is
    part of the public contract. The order is the alphabetical order the
    previous inspect-based discovery produced, so existing users see no change.
    """
    from spectral_connectivity.wrapper import DEFAULT_METHODS

    expected = (
        "coherence_magnitude",
        "coherence_phase",
        "debiased_squared_phase_lag_index",
        "debiased_squared_weighted_phase_lag_index",
        "imaginary_coherence",
        "pairwise_phase_consistency",
        "pairwise_spectral_granger_prediction",
        "phase_lag_index",
        "phase_locking_value",
        "power",
        "weighted_phase_lag_index",
    )
    assert DEFAULT_METHODS == expected
    # The deliberately excluded measures must not be in the default.
    for excluded in ("coherency", "global_coherence", "phase_slope_index"):
        assert excluded not in DEFAULT_METHODS

    rng = np.random.default_rng(1)
    ds = multitaper_connectivity(rng.standard_normal((256, 3)), sampling_frequency=250)
    # Same measures AND same variable order as the allowlist.
    assert tuple(ds.data_vars) == expected


def test_from_multitaper_supports_subclass_overriding_init():
    """from_multitaper must work for a subclass with the previous constructor.

    The private adoption fast-path passes a keyword the base __init__ accepts; a
    subclass that overrides __init__ (mirroring the old signature) need not, so
    from_multitaper must fall back to the plain (defensive-copy) path for it
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

    sub = LegacyConnectivity.from_multitaper(m)
    assert isinstance(sub, LegacyConnectivity)
    assert sub.marker == "subclass"
    # Fell back to the defensive-copy path (owns its data), and works normally.
    assert sub._fourier_coefficients.base is None
    base = Connectivity.from_multitaper(m)
    np.testing.assert_array_equal(sub.power(), base.power())
    # The base class still uses the no-copy adoption path (stores a view).
    assert base._fourier_coefficients.base is not None


def test_result_carries_descriptive_coordinate_metadata():
    """Coordinates carry unambiguous axis labels and physical units."""
    rng = np.random.default_rng(0)
    ds = multitaper_connectivity(
        rng.standard_normal((512, 5, 3)), sampling_frequency=500
    )
    assert ds.coords["time"].attrs["units"] == "s"
    assert ds.coords["time"].attrs["long_name"] == "Window center time"
    assert ds.coords["frequency"].attrs["units"] == "Hz"
    assert ds.coords["frequency"].attrs["long_name"] == "Frequency"
    assert ds.coords["source"].attrs["long_name"] == "Source signal"
    assert ds.coords["target"].attrs["long_name"] == "Target signal"


def test_result_carries_provenance_metadata():
    """Each measure records package/version/backend/expectation_type/measure."""
    from spectral_connectivity.wrapper import _package_version

    rng = np.random.default_rng(1)
    da = connectivity_to_xarray(
        Multitaper(rng.standard_normal((512, 5, 3)), sampling_frequency=500),
        method="coherence_magnitude",
    )
    assert da.attrs["measure"] == "coherence_magnitude"
    assert da.attrs["package"] == "spectral_connectivity"
    assert da.attrs["package_version"] == _package_version()
    assert da.attrs["backend"] in ("CPU", "GPU")
    assert da.attrs["expectation_type"] == "trials_tapers"
    # The multitaper parameters are still recorded under the mt_ prefix.
    assert any(key.startswith("mt_") for key in da.attrs)


def test_provenance_records_measure_kwargs(tmp_path, monkeypatch):
    """Measure keyword arguments are recorded as ``arg_<key>``.

    Scalar kwargs remain convenient individual attributes, while structured
    values and the complete kwargs mapping use canonical JSON. Exercised through
    a stub measure that fits the (time, frequency, source, target) layout and
    accepts kwargs, since none of the default xarray-compatible measures take
    keyword arguments.
    """
    rng = np.random.default_rng(3)
    m = Multitaper(rng.standard_normal((256, 4, 3)), sampling_frequency=500)

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

    da = connectivity_to_xarray(
        m,
        method="stub_measure",
        threshold=0.5,
        window=[1, 2, 3],
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


def test_provenance_arg_key_collision_raises_instead_of_overwriting(monkeypatch):
    """A structured ``x`` and a scalar ``x_json`` cannot silently share a key."""
    rng = np.random.default_rng(3)
    m = Multitaper(rng.standard_normal((256, 4, 3)), sampling_frequency=500)

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

    with pytest.raises(ValueError, match="assigned twice"):
        connectivity_to_xarray(m, method="stub_measure", x=[1, 2, 3], x_json=5)


def test_broken_measure_in_batch_propagates_not_implemented(monkeypatch):
    """A genuine NotImplementedError is not swallowed into a missing variable."""
    rng = np.random.default_rng(9)

    def broken_measure(connectivity):
        raise NotImplementedError("backend cannot compute this")

    monkeypatch.setattr(Connectivity, "broken_measure", broken_measure, raising=False)

    with pytest.raises(NotImplementedError, match="backend cannot compute"):
        multitaper_connectivity(
            rng.standard_normal((256, 4, 2)),
            sampling_frequency=256,
            method=["coherence_magnitude", "broken_measure"],
        )


def test_wrapper_capabilities_do_not_use_method_name_substrings(monkeypatch):
    """A pairwise extension containing 'directed' is not rejected by its name."""
    rng = np.random.default_rng(8)
    m = Multitaper(rng.standard_normal((128, 3, 2)), sampling_frequency=128)

    def undirected_similarity(connectivity):
        return np.zeros(
            (
                len(connectivity.time),
                len(connectivity.frequencies),
                connectivity.n_signals,
                connectivity.n_signals,
            )
        )

    monkeypatch.setattr(
        Connectivity, "undirected_similarity", undirected_similarity, raising=False
    )

    data_array = connectivity_to_xarray(m, method="undirected_similarity")
    assert data_array.dims == ("time", "frequency", "source", "target")


def test_multitaper_connectivity_merges_nonstandard_dataset_in_batch():
    rng = np.random.default_rng(0)
    result = multitaper_connectivity(
        rng.standard_normal((256, 4, 3)),
        sampling_frequency=500,
        method=["coherence_magnitude", "global_coherence"],
    )
    assert "coherence_magnitude" in result
    assert "global_coherence" in result
    assert "global_coherence_vectors" in result


def test_frequency_bands_on_mixed_dataset_leaves_frequency_free_variables():
    # A batch mixing a frequency-carrying measure with one that has no frequency
    # axis (group_delay): band reduction applies to the former and passes the
    # latter through unchanged, rather than erroring on the frequency-free var.
    result = multitaper_connectivity(
        np.random.default_rng(452).standard_normal((512, 4, 3)),
        sampling_frequency=200,
        method=["coherence_magnitude", "group_delay"],
        frequency_bands={"alpha": (8, 12), "beta": (13, 30)},
    )
    assert result["coherence_magnitude"].dims == ("time", "band", "source", "target")
    assert result["group_delay"].dims == ("time", "source", "target")
    assert result.band.values.tolist() == ["alpha", "beta"]


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
    # components use connection_seed_group/connection_target_group as per-
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
    assert result.connection_seed_group.values.tolist() == [0]
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
    ds = multitaper_connectivity(
        rng.standard_normal((512, 5, 3)), sampling_frequency=500
    )
    path = tmp_path / "provenance.nc"
    ds.to_netcdf(path)
    reloaded = xr.open_dataset(path)
    try:
        assert reloaded.coords["time"].attrs["units"] == "s"
        assert reloaded.coords["frequency"].attrs["units"] == "Hz"
        var = reloaded["coherence_magnitude"]
        assert var.attrs["package"] == "spectral_connectivity"
        assert var.attrs["backend"] in ("CPU", "GPU")
        assert var.attrs["expectation_type"] == "trials_tapers"
        # The shared provenance attached at the Dataset level also round-trips.
        assert reloaded.attrs["package"] == "spectral_connectivity"
        assert reloaded.attrs["backend"] in ("CPU", "GPU")
        assert reloaded.attrs["expectation_type"] == "trials_tapers"
        assert reloaded.attrs["mt_sampling_frequency"] == 500
    finally:
        reloaded.close()


def test_backend_provenance_reflects_imported_backend_not_env(monkeypatch):
    """The backend attr must reflect the imported backend, not the live env var.

    The backend is fixed when the package is imported; toggling
    SPECTRAL_CONNECTIVITY_ENABLE_GPU afterwards must not mislabel a result.
    """
    from spectral_connectivity.utils import get_compute_backend

    monkeypatch.setenv("SPECTRAL_CONNECTIVITY_ENABLE_GPU", "true")
    rng = np.random.default_rng(0)
    da = connectivity_to_xarray(
        Multitaper(rng.standard_normal((256, 5, 3)), sampling_frequency=500),
        method="coherence_magnitude",
    )
    # Matches the actually-imported backend, not the toggled env var. (Left
    # backend-agnostic so the suite can also run under the GPU backend.)
    assert da.attrs["backend"] == get_compute_backend()["backend"].upper()


def test_multitaper_connectivity_directed_source_target_orientation():
    """Directed measures must label sel(source=driver, target=receiver) correctly.

    Build a unidirectional VAR (signal 0 drives signal 1) as multi-trial time
    series and confirm the Granger DataArray reads the strong influence at
    source=0 -> target=1 and near-zero at source=1 -> target=0. Without the
    directed transpose in the wrapper these two are swapped.
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

    da = multitaper_connectivity(
        time_series,
        sampling_frequency=200,
        time_halfbandwidth_product=3,
        method="pairwise_spectral_granger_prediction",
        signal_names=["x", "y"],
    )
    causal = da.sel(source="x", target="y").values  # x drives y
    anti_causal = da.sel(source="y", target="x").values
    assert np.nanmax(causal) > np.nanmax(anti_causal)
    assert np.nanmax(causal) > 0.05


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
            method="coherence_magnitude",
            signal_names=["a", "a"],
        )


def test_multitaper_connectivity_rejects_duplicate_nan_signal_names():
    """Index semantics treat distinct NaN objects as duplicate labels."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="must not contain missing labels"):
        multitaper_connectivity(
            rng.standard_normal((256, 3, 2)),
            sampling_frequency=256,
            method="coherence_magnitude",
            signal_names=[float("nan"), float("nan")],
        )


def test_multitaper_connectivity_rejects_structured_signal_names():
    """Hashable tuples are not silently accepted as non-portable coordinates."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="one-dimensional xarray coordinate"):
        multitaper_connectivity(
            rng.standard_normal((256, 3, 2)),
            sampling_frequency=256,
            method="coherence_magnitude",
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
        method="coherence_magnitude",
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
            method="coherence_magnitude",
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
    assert ds["coherence_magnitude"].dims == ("time", "frequency", "source", "target")
    assert ds["power"].dims == ("time", "frequency", "source")
    # No Dataset-wide scalar coordinate leaked onto the wrong variable.
    assert "target" not in ds["power"].coords
    assert ds["coherence_magnitude"].sizes["source"] == 2


_DTF_FAMILY = [
    "directed_transfer_function",
    "directed_coherence",
    "partial_directed_coherence",
    "generalized_partial_directed_coherence",
    "direct_directed_transfer_function",
]


@mark.parametrize("method", _DTF_FAMILY)
def test_multitaper_connectivity_exposes_dtf_family_with_source_target(method):
    """The DTF family is opt-in through the wrapper and oriented source -> target.

    For a unidirectional VAR (signal 0 drives 1), the causal entry is
    sel(source=0, target=1); it must dominate the anti-causal sel(source=1,
    target=0). This also confirms the wrapper's directed transpose is applied to
    these newly exposed measures.
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

    da = multitaper_connectivity(
        time_series,
        sampling_frequency=200,
        time_halfbandwidth_product=3,
        method=method,
        signal_names=["x", "y"],
    )
    assert da.dims == ("time", "frequency", "source", "target")
    causal = da.sel(source="x", target="y").values  # x drives y
    anti_causal = da.sel(source="y", target="x").values
    assert np.nanmax(causal) > np.nanmax(anti_causal)


def test_multitaper_connectivity_dataset_carries_shared_provenance():
    """A multi-measure Dataset exposes shared provenance at the top level.

    Previously provenance lived only on each DataArray; a returned Dataset had
    no top-level attrs, so tracing how a batch was produced meant inspecting an
    arbitrary variable. The shared attrs (package, backend, expectation type,
    multitaper parameters) are now on the Dataset too. (Their NetCDF round-trip
    is covered by ``test_metadata_survives_netcdf_round_trip``.)
    """
    rng = np.random.default_rng(0)
    ds = multitaper_connectivity(
        rng.standard_normal((512, 5, 3)), sampling_frequency=500
    )
    assert ds.attrs["package"] == "spectral_connectivity"
    assert ds.attrs["backend"] in ("CPU", "GPU")
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
        method="coherence_magnitude",
        signal_names=["a", "b", "c"],
    )
    actual = fourier_connectivity(
        transform.fft(),
        frequencies=transform.frequencies,
        time=transform.time,
        method="coherence_magnitude",
        signal_names=["a", "b", "c"],
    )

    xr.testing.assert_allclose(actual, expected)
    assert actual.attrs["fourier_source"] == "external_fourier_coefficients"


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
    """Frequency operations are coordinate-based and preserve axis order."""
    result = multitaper_connectivity(
        np.random.default_rng(303).standard_normal((256, 3)),
        sampling_frequency=128,
        method="coherence_magnitude",
        frequency_range=(8, 32),
        frequency_decimation=2,
        frequency_bands={"alpha": (8, 12), "beta": (13, 30)},
    )

    assert result.dims == ("time", "band", "source", "target")
    assert result.band.values.tolist() == ["alpha", "beta"]
    assert result.attrs["frequency_reduction"] == "mean"


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


def test_frequency_band_integral_is_restricted_to_spectral_densities():
    score = xr.DataArray(
        [0.25, 0.5],
        dims=("frequency",),
        coords={"frequency": [1.0, 2.0]},
        name="coherence_magnitude",
        attrs={"measure": "coherence_magnitude"},
    )
    with pytest.raises(ValueError, match="defined only for power"):
        frequency_band_reduce(score, {"low": (1, 2)}, reduction="integral")


def test_multitaper_frequency_crop_and_decimation_select_correct_bins():
    """Cropping and decimation select the expected frequency-coordinate values."""
    result = multitaper_connectivity(
        np.random.default_rng(313).standard_normal((256, 2)),
        sampling_frequency=128,
        method="coherence_magnitude",
        frequency_range=(8, 32),
        frequency_decimation=2,
    )

    # Full grid bins are multiples of 128/256 = 0.5 Hz; cropping keeps [8, 32]
    # and decimation by 2 keeps every other surviving bin.
    full_frequencies = np.fft.rfftfreq(256, d=1 / 128)
    in_band = full_frequencies[(full_frequencies >= 8) & (full_frequencies <= 32)]
    expected = in_band[::2]
    np.testing.assert_allclose(result.frequency.values, expected)


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
        flat_power, {"band": (1.0, 5.0)}, reduction="integral"
    )
    # Trapezoidal integral of the constant 2.0 over [1, 5] Hz is 2 * (5 - 1) = 8.
    assert float(reduced.sel(band="band")) == pytest.approx(8.0)


def test_fourier_connectivity_rejects_unlabeled_directed_measure():
    """Directed measures need a frequency coordinate to verify two-sidedness."""
    coefficients = np.ones((3, 8, 2), dtype=np.complex128)
    with pytest.raises(ValueError, match="two-sided spectrum"):
        fourier_connectivity(
            coefficients,
            method="pairwise_spectral_granger_prediction",
        )


def test_fourier_connectivity_allows_unlabeled_undirected_measure():
    coefficients = np.random.default_rng(314).standard_normal(
        (4, 8, 2)
    ) + 1j * np.random.default_rng(315).standard_normal((4, 8, 2))
    result = fourier_connectivity(coefficients, method="coherence_magnitude")
    assert "frequency" in result.dims


def test_fourier_connectivity_accepts_one_sided_functional_input():
    rng = np.random.default_rng(316)
    coefficients = rng.standard_normal((4, 9, 2)) + 1j * rng.standard_normal((4, 9, 2))
    result = fourier_connectivity(
        coefficients,
        frequencies=np.linspace(0, 40, 9),
        method="coherence_magnitude",
    )

    assert result.dims == ("time", "frequency", "source", "target")
    np.testing.assert_array_equal(result.frequency, np.linspace(0, 40, 9))
    assert result.attrs["fourier_is_one_sided"]
    assert result.attrs["fourier_one_sided_inferred"]


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
        method="coherence_magnitude",
        is_one_sided=True,
    )

    assert np.all(result.frequency >= 0)
    assert result.attrs["fourier_is_one_sided"]


def test_fourier_connectivity_rejects_fftshifted_coordinate():
    coefficients = np.ones((3, 8, 2), dtype=np.complex128)
    with pytest.raises(ValueError, match="standard FFT order"):
        fourier_connectivity(
            coefficients,
            frequencies=np.fft.fftshift(np.fft.fftfreq(8, d=0.01)),
            method="coherence_magnitude",
        )


def test_connectivity_to_xarray_namespaces_alternative_transform_provenance():
    data = np.random.default_rng(304).standard_normal((128, 3, 2))
    welch = connectivity_to_xarray(
        Welch(data, sampling_frequency=64, n_time_samples_per_segment=32),
        method="coherence_magnitude",
    )
    morlet = connectivity_to_xarray(
        MorletWavelet(data, 64, np.array([4.0, 8.0, 16.0])),
        method="coherence_magnitude",
    )

    assert welch.attrs["welch_window"] == "hann_periodic"
    assert morlet.attrs["morlet_decimation"] == 1
    assert morlet.frequency.values.tolist() == [4.0, 8.0, 16.0]
    assert morlet.valid_time_frequency.dims == ("time", "frequency")
    assert morlet.attrs["morlet_edge_mode"] == "keep"
    assert morlet.attrs["morlet_smoothing_kernel"] == "boxcar"


def test_connectivity_to_xarray_exposes_morlet_invalid_edges():
    data = np.random.default_rng(318).standard_normal((128, 2, 2))
    transform = MorletWavelet(
        data,
        64,
        np.array([4.0, 8.0, 16.0]),
        smoothing_time=0.25,
        edge_mode="nan",
    )
    result = connectivity_to_xarray(transform, method="power")

    np.testing.assert_array_equal(
        result.isnull().all("source"), ~result.valid_time_frequency
    )
