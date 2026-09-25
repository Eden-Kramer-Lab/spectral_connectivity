import inspect
import json
import warnings

import numpy as np
import pytest
import scipy.fft
import xarray as xr

from spectral_connectivity import MorletWavelet, Multitaper, Welch
from spectral_connectivity.connectivity import Connectivity
from spectral_connectivity.wrapper import (
    DEFAULT_METHODS,
    _canonical_json,
    _json_compatible,
    _MeasureSpec,
    _netcdf_provenance_value,
    _reject_unmaterialized_backing,
    _time_axis_from_dataarray,
    connectivity_to_xarray,
    fourier_connectivity,
    frequency_band_reduce,
    multitaper_connectivity,
)

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
        method="coherence_magnitude",
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
        values[..., 1, 0] = 7.0  # native convention: group 0 -> group 1
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
    assert psi.attrs["frequency_band_lower"] == 8
    assert psi.attrs["frequency_band_upper"] == 40
    assert group_delay.group_delay.attrs["frequency_band_lower"] == 8
    assert group_delay.group_delay.attrs["frequency_band_upper"] == 40
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
        method=["coherence_magnitude"],
        sampling_frequency=_GRID_SAMPLING_FREQUENCY,
        time_window_duration=time_window_duration,
    )
    assert isinstance(cons, xr.Dataset)
    assert tuple(cons.data_vars) == ("coherence_magnitude",)
    np.testing.assert_allclose(cons["coherence_magnitude"].time.values, expected_time)


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


# Channel 0 ("right") has 25x the variance of channel 1 ("left") and the labels
# are deliberately unsorted, so per-signal power exposes a data/label
# mispairing, a channel reversal, or a label sort (a symmetric two-signal
# measure such as coherence_magnitude cannot).
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
        multitaper_connectivity(data, sampling_frequency=64, method="power", start_time=[0, 1])


def test_measure_spec_rejects_inconsistent_field_combinations():
    """Illegal capability combinations are unrepresentable, not merely unused."""
    with pytest.raises(ValueError, match="transpose_output requires pairwise"):
        _MeasureSpec("power", is_directed=True, transpose_output=True)
    with pytest.raises(ValueError, match="requires a directional measure"):
        _MeasureSpec("pairwise", transpose_output=True)


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
        multitaper_connectivity(data, sampling_frequency=256, method="coherence_magnitude")


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
    with pytest.raises(TypeError, match="dask-backed"):
        multitaper_connectivity(data, sampling_frequency=256, method="coherence_magnitude")


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
    assert json.loads(ds["coherence_magnitude"].attrs["input_attrs_json"]) == input_attrs

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
        result = Multitaper(time_series, sampling_frequency=500, fft_workers=workers).fft()
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
            method="coherence_magnitude",
            fft_workers=2,
        )
    assert 2 in recorded  # the taper-projection FFT received workers=2

    # On the GPU backend `workers` is not forwarded (cupyx's FFT has no such
    # parameter). Simulate GPU on the CPU by patching the backend check.
    gpu_multitaper = Multitaper(time_series, sampling_frequency=500, fft_workers=-1)
    _ = gpu_multitaper.tapers
    recorded = []
    with (
        patch.object(transforms, "is_gpu_enabled", lambda: True),
        patch.object(transforms, "fft", spying_fft(recorded)),
    ):
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
    host_mask = np.asarray(transform.valid_time_frequency)

    class DeviceMaskTransform:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        @property
        def valid_time_frequency(self):
            return _DeviceLike(host_mask)

    result = connectivity_to_xarray(
        DeviceMaskTransform(transform), method="coherence_magnitude"
    )
    np.testing.assert_array_equal(result["valid_time_frequency"].values, host_mask)


@pytest.mark.parametrize(
    "name",
    [
        "jackknife",
        "minimum_phase_reconstruction_error",
        "from_transform",
        "_clear_cached_intermediates",
    ],
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
            method=["coherence_magnitude", "canonical_coherence"],
            connectivity_kwargs={"group_labels": [0, 0, 1]},
        )


def test_multi_method_shares_single_fft(monkeypatch):
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

    monkeypatch.setattr(Multitaper, "fft", counting_fft)
    multitaper_connectivity(time_series, sampling_frequency=500, method=methods)

    assert calls["n"] == 1, f"FFT computed {calls['n']} times for {len(methods)} methods"


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
    part of the public contract. The allowlist is kept in alphabetical order.
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
    assert expected == DEFAULT_METHODS
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
    np.testing.assert_array_equal(sub.observation_weights, base.observation_weights)
    np.testing.assert_array_equal(sub.frequencies, base.frequencies)
    np.testing.assert_allclose(sub.power(), base.power())
    np.testing.assert_allclose(
        sub.coherence_magnitude(), base.coherence_magnitude(), equal_nan=True
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
        method="coherence_magnitude",
    )
    assert da.attrs["measure"] == "coherence_magnitude"
    assert da.attrs["measure_kwargs_json"] == "{}"
    assert da.attrs["package"] == "spectral_connectivity"
    assert da.attrs["package_version"] == version("spectral_connectivity")
    assert da.attrs["backend"] in ("CPU", "GPU")
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


def test_provenance_arg_key_collision_raises_instead_of_overwriting(stub_measure):
    """A structured ``x`` and a scalar ``x_json`` cannot silently share a key."""
    rng = np.random.default_rng(3)
    m = Multitaper(rng.standard_normal((256, 4, 3)), sampling_frequency=500)

    with pytest.raises(ValueError, match="assigned twice"):
        connectivity_to_xarray(m, method=stub_measure, x=[1, 2, 3], x_json=5)


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
            method=["coherence_magnitude", "broken_measure"],
        )


def test_wrapper_capabilities_do_not_use_method_name_substrings(monkeypatch):
    """A pairwise extension containing 'directed' is neither rejected nor
    transposed (treated as a directed measure) because of its name."""
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
    # An unregistered extension keeps its native [source, target] orientation.
    np.testing.assert_array_equal(data_array.sel(source="0", target="1"), 1.0)
    np.testing.assert_array_equal(data_array.sel(source="1", target="0"), 10.0)


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
    time_series = np.random.default_rng(452).standard_normal((512, 4, 3))
    methods = ["coherence_magnitude", "group_delay"]
    bands = {"alpha": (8, 12), "beta": (13, 30)}
    result = multitaper_connectivity(
        time_series,
        sampling_frequency=200,
        method=methods,
        frequency_bands=bands,
    )
    unreduced = multitaper_connectivity(time_series, sampling_frequency=200, method=methods)

    assert result["coherence_magnitude"].dims == ("time", "band", "source", "target")
    assert result["group_delay"].dims == ("time", "source", "target")
    assert result.band.values.tolist() == ["alpha", "beta"]
    for name, (lower, upper) in bands.items():
        expected = (
            unreduced["coherence_magnitude"]
            .sel(frequency=slice(lower, upper))
            .mean("frequency")
        )
        xr.testing.assert_allclose(
            result["coherence_magnitude"].sel(band=name, drop=True), expected
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
    ds = multitaper_connectivity(rng.standard_normal((512, 5, 3)), sampling_frequency=500)
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
    target=0). Without the wrapper's directed transpose these two are swapped.
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
    "phase_slope_index": {"frequencies_of_interest": list(_LAGGED_BAND)},
    "delay": {"frequencies_of_interest": list(_LAGGED_BAND)},
    "group_delay": {"frequencies_of_interest": list(_LAGGED_BAND)},
}


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
    """Frequency operations are coordinate-based and preserve axis order.

    Each band value is the plain mean of the unreduced result over the band's
    inclusive bins. After cropping to [8, 32] Hz and decimating the 0.5 Hz grid
    by 2 the bins are the integers 8..32, so every band endpoint below lands
    exactly on a bin and pins inclusivity at both ends.
    """
    time_series = np.random.default_rng(303).standard_normal((256, 3))
    frequency_kwargs = {
        "sampling_frequency": 128,
        "method": "coherence_magnitude",
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
            np.array([4.0, 8.0, 16.0, 32.0]),
            smoothing_time=0.5,
            edge_mode="nan",
        )
    coherence = connectivity_to_xarray(transform, method="coherence_magnitude")
    power = connectivity_to_xarray(transform, method="power")
    bands = {"low": (4, 8), "all": (4, 32)}
    mean = frequency_band_reduce(coherence, bands)
    integral = frequency_band_reduce(power, bands, reduction="integral")

    validity = np.asarray(transform.valid_time_frequency)
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
        np.arange(5.0, 45.0, 5.0),
        smoothing_time=0.3,
        edge_mode="nan",
    )
    result = connectivity_to_xarray(transform, method="coherence_magnitude")
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

    reduced = frequency_band_reduce(power, {"invalid": (10.0, 10.0)}, reduction="integral")

    assert np.isnan(reduced.sel(band="invalid")).all()
    assert not reduced.valid_time_band.sel(band="invalid").any()


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
    """Cropping happens before decimation and selects exact coordinate values.

    The full grid is multiples of 128/256 = 0.5 Hz. Cropping to [8.5, 32] and
    then keeping every other bin starts at 8.5 Hz: 8.5, 9.5, ..., 31.5.
    Decimating first would instead keep the integer bins 9..32.
    """
    result = multitaper_connectivity(
        np.random.default_rng(313).standard_normal((256, 2)),
        sampling_frequency=128,
        method="coherence_magnitude",
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
    reduced = frequency_band_reduce(flat_power, {"band": (1.0, 5.0)}, reduction="integral")
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
        flat_density_on_2hz_grid, {"band": bounds}, reduction="integral"
    )
    assert float(reduced.sel(band="band")) == pytest.approx(2.0 * (high - low))


def test_frequency_band_integrals_of_adjacent_bands_add_up(flat_density_on_2hz_grid):
    rng = np.random.default_rng(44)
    density = flat_density_on_2hz_grid.copy(data=rng.uniform(0.5, 3.0, 26))
    reduced = frequency_band_reduce(
        density,
        {"low": (10.0, 17.3), "high": (17.3, 30.0), "both": (10.0, 30.0)},
        reduction="integral",
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
    reduced = frequency_band_reduce(power, {"all": (0.0, 250.0)}, reduction="integral")
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
        n_fft_samples=n_fft_samples,
        time_halfbandwidth_product=1,
    )
    assert power.frequency.size == n_fft_samples // 2 + 1
    spacing = float(power.frequency[1] - power.frequency[0])
    integral = float(
        frequency_band_reduce(power, {"all": (0.0, 50.0)}, reduction="integral").squeeze()
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
        frequency_band_reduce(power, {"all": (0.0, 50.0)}, reduction="integral").squeeze()
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
        "n_fft_samples": n_time,
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
            return frequency_band_reduce(power, bands, reduction="integral")

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
    [((0.0, 4.0), (0.0, 4.0)), ((4.0, 8.0), (0.0, 8.0)), ((10.0, 30.0), (0.0, 30.0))],
)
def test_frequency_band_integral_does_not_depend_on_cropping(band, frequency_range):
    """Only the true Nyquist bin is undoubled. Cropping the grid with
    frequency_range must not turn its new last bin into a Nyquist bin, or a
    band ending at the crop edge integrates that bin twice (+3% to +15%)."""
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


def test_frequency_band_integral_edge_bins_stay_additive_and_mean_is_unchanged():
    """Folding the DC and Nyquist cells must keep adjacent bands additive and
    must not touch ``reduction="mean"``."""
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
        reduction="integral",
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
    warning and match ``Connectivity.from_multitaper`` on the same transform."""
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
    expected = Connectivity.from_multitaper(multitaper).pairwise_spectral_granger_prediction()

    assert result.dims == ("time", "frequency", "source", "target")
    assert not result.attrs["fourier_is_one_sided"]
    assert not result.attrs["fourier_one_sided_inferred"]
    # The core's [..., i, j] is the influence j -> i; the wrapper labels source -> target.
    np.testing.assert_allclose(
        result.values, np.swapaxes(np.asarray(expected), -1, -2), equal_nan=True
    )
    driven = result.sel(source="0", target="1").values
    driver = result.sel(source="1", target="0").values
    assert np.nanmax(driven) > np.nanmax(driver)


def test_fourier_connectivity_warns_on_one_sided_input_declared_two_sided():
    """rfft-like coefficients declared ``is_one_sided=False`` would silently give
    wrong Granger values, so the missing conjugate symmetry is reported, both
    for an explicit two-sided-only method and for the default method set."""
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
    with pytest.warns(UserWarning, match="not conjugate-symmetric"):
        fourier_connectivity(one_sided, is_one_sided=False)
    # Functional measures do not rely on the two-sided declaration's symmetry.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fourier_connectivity(one_sided, method="coherence_magnitude", is_one_sided=False)


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


def test_fourier_connectivity_allows_unlabeled_undirected_measure():
    coefficients = np.random.default_rng(314).standard_normal(
        (4, 8, 2)
    ) + 1j * np.random.default_rng(315).standard_normal((4, 8, 2))
    result = fourier_connectivity(
        coefficients, method="coherence_magnitude", is_one_sided=False
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
        method="coherence_magnitude",
    )

    assert result.dims == ("time", "frequency", "source", "target")
    np.testing.assert_array_equal(result.frequency, np.linspace(0, 40, 9))
    assert result.attrs["fourier_is_one_sided"]
    assert result.attrs["fourier_one_sided_inferred"]


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
        fourier_connectivity(coefficients, method="coherence_magnitude")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fourier_connectivity(coefficients, method="coherence_magnitude", is_one_sided=False)
        fourier_connectivity(coefficients, method="coherence_magnitude", is_one_sided=True)


def test_fourier_connectivity_infers_one_bin_positive_input_as_one_sided():
    """A positive singleton coordinate must not enable directed measures."""
    rng = np.random.default_rng(322)
    coefficients = rng.standard_normal((5, 1, 2)) + 1j * rng.standard_normal((5, 1, 2))
    result = fourier_connectivity(coefficients, frequencies=np.array([10.0]))

    assert result.attrs["fourier_is_one_sided"]
    assert result.attrs["fourier_one_sided_inferred"]
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
        method="coherence_magnitude",
        is_one_sided=True,
    )

    # All 9 one-sided bins are kept (an rfft of 16 samples), labeled in
    # normalized cycles per sample from 0 to Nyquist (0.5).
    assert result.sizes["frequency"] == 9
    np.testing.assert_allclose(result.frequency, np.arange(9) / 16)
    assert result.attrs["fourier_frequency_coordinate"] == "normalized"
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

    np.testing.assert_array_equal(result.isnull().all("source"), ~result.valid_time_frequency)


def test_morlet_validity_aligns_with_nonstandard_xarray_shapes():
    data = np.random.default_rng(319).standard_normal((192, 5, 2))
    # Adjacent-bin measures require a uniform grid; the band below still
    # selects the two interior bins.
    frequencies = np.array([4.0, 8.0, 12.0, 16.0])
    transform = MorletWavelet(
        data,
        64,
        frequencies,
        smoothing_time=0.25,
        edge_mode="nan",
    )

    phase_slope = connectivity_to_xarray(transform, method="phase_slope_index")
    assert phase_slope.dims == ("time", "source", "target")
    assert phase_slope.valid_time.dims == ("time",)
    np.testing.assert_array_equal(
        phase_slope.valid_time, transform.valid_time_frequency.all(axis=1)
    )

    band = (5.0, 15.0)
    # The smoothed Morlet observations are correlated, which the significance
    # test inside delay/group_delay reports.
    with pytest.warns(UserWarning, match="assumes independent observations"):
        delay = connectivity_to_xarray(
            transform,
            method="delay",
            frequencies_of_interest=band,
        )
    np.testing.assert_array_equal(delay.frequency, [8.0, 12.0])
    np.testing.assert_array_equal(
        delay.valid_time_frequency,
        transform.valid_time_frequency[:, [1, 2]],
    )

    with pytest.warns(UserWarning, match="assumes independent observations"):
        group_delay = connectivity_to_xarray(
            transform,
            method="group_delay",
            frequencies_of_interest=band,
        )
    assert "frequency" not in group_delay.dims
    assert group_delay.valid_time.dims == ("time",)
    np.testing.assert_array_equal(
        group_delay.valid_time,
        transform.valid_time_frequency[:, [1, 2]].all(axis=1),
    )


def test_fourier_connectivity_default_coordinates_are_labeled_as_such():
    """Without coordinates, frequency is normalized (cycles/sample) and time is a
    window index; the coordinate metadata must say so, not claim Hz and s."""
    rng = np.random.default_rng(40)
    coefficients = rng.standard_normal((3, 8, 2)) + 1j * rng.standard_normal((3, 8, 2))
    result = fourier_connectivity(
        coefficients, method="coherence_magnitude", is_one_sided=False
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
        method="coherence_magnitude",
    )
    assert result.frequency.attrs == {"long_name": "Frequency", "units": "Hz"}
    assert result.time.attrs == {"long_name": "Window center time", "units": "s"}


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
            method="coherence_magnitude",
        )


def test_frequency_coordinate_attrs_do_not_depend_on_measure_order():
    """delay built its frequency coordinate with only ``units``; merged first,
    it dropped the long_name every other measure sets."""
    time_series = np.random.default_rng(43).standard_normal((1000, 3, 2))
    for methods in (["delay", "coherence_magnitude"], ["coherence_magnitude", "delay"]):
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
        method=["coherence_magnitude", "phase_slope_index"],
        frequency_range=(4, 50),
        frequency_decimation=2,
    )
    coherence, psi = dataset.coherence_magnitude, dataset.phase_slope_index
    assert "frequency_band_lower" not in coherence.coords
    assert "frequency_band_lower" not in coherence.attrs
    # phase_slope_index used its own (default, full) band, not frequency_range.
    assert psi.attrs["frequency_band_lower"] == 0.0
    assert psi.attrs["frequency_band_upper"] == 250.0
    assert json.loads(coherence.attrs["frequency_range_json"]) == [4.0, 50.0]
    assert coherence.attrs["frequency_decimation"] == 2
    for key in ("frequency_range_json", "frequency_decimation"):
        assert key not in psi.attrs
        assert key not in dataset.attrs

    banded = multitaper_connectivity(
        time_series,
        sampling_frequency=500,
        method=["coherence_magnitude", "phase_slope_index"],
        frequency_bands={"beta": (13, 30)},
    )
    assert banded.coherence_magnitude.attrs["frequency_reduction"] == "mean"
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
        labeled_channels, sampling_frequency=500, method="coherence_magnitude"
    )
    assert coherence.source_region.values.tolist() == ["CA1", "PFC"]
    assert coherence.target_region.values.tolist() == ["CA1", "PFC"]
    np.testing.assert_array_equal(coherence.source_depth_um, [120.0, 900.0])
    assert coherence.source_region.dims == ("source",)

    power = multitaper_connectivity(labeled_channels, sampling_frequency=500, method="power")
    assert power.source_region.values.tolist() == ["CA1", "PFC"]

    squeezed = multitaper_connectivity(
        labeled_channels, sampling_frequency=500, method="coherence_magnitude", squeeze=True
    )
    assert squeezed.source_region.item() == "CA1"
    assert squeezed.target_region.item() == "PFC"


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
        reduction="integral",
    )
    assert "units" not in unlabeled_units.attrs  # an unparsed density unit is not kept


@pytest.mark.parametrize(
    ("method", "units"),
    [
        ("coherence_magnitude", "1"),
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


def test_measure_labels_cover_every_measure():
    from spectral_connectivity.wrapper import _MEASURE_DESCRIPTIONS, _MEASURE_SPECS

    assert set(_MEASURE_DESCRIPTIONS) == set(_MEASURE_SPECS)


def test_large_array_input_attrs_are_summarized():
    """A large array attribute is recorded by shape and dtype, not copied into
    every variable's JSON (a 50,000-sample attr made a 12 MB file)."""
    data = xr.DataArray(
        np.random.default_rng(50).standard_normal((512, 2)),
        dims=("time", "channel"),
        attrs={"subject": "m1", "raw_trace": np.arange(50_000.0), "montage": [1, 2, 3]},
    )
    result = multitaper_connectivity(
        data, sampling_frequency=256, method="coherence_magnitude"
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
