"""Public result-schema contracts across transforms and frequency operations."""

import json

import numpy as np
import pytest
import xarray as xr
from scipy.signal.windows import hann

from spectral_connectivity import (
    MorletWavelet,
    Multitaper,
    ShortTimeFourierTransform,
    Welch,
    estimate_frequency_resolution,
    suggest_parameters,
)
from spectral_connectivity.wrapper import (
    connectivity_to_xarray,
    fourier_connectivity,
    frequency_band_reduce,
    multitaper_connectivity,
)


@pytest.fixture
def data():
    return np.random.default_rng(910).standard_normal((128, 3, 2))


def _transform(name, data):
    if name == "multitaper":
        return Multitaper(
            data,
            64,
            time_halfbandwidth_product=2,
            time_window_duration=0.5,
            time_window_step=0.25,
        )
    if name == "stft":
        return ShortTimeFourierTransform(
            data, 64, time_window_duration=0.5, time_window_step=0.25
        )
    if name == "welch":
        return Welch(data, 64, segment_duration=0.5)
    return MorletWavelet(data, 64, [4, 8, 16])


@pytest.mark.parametrize(
    ("name", "n_observations"), [("multitaper", 9), ("stft", 3), ("welch", 21), ("morlet", 3)]
)
def test_transforms_share_known_facts_and_native_settings(data, name, n_observations):
    result = connectivity_to_xarray(_transform(name, data), method="power")
    assert result.attrs["output_schema_version"] == 1
    assert result.attrs["transform"] == name
    assert result.attrs["sampling_frequency"] == 64
    assert result.attrs["n_trials"] == 3
    assert result.attrs["n_signals"] == 2
    assert result.attrs["n_observations"] == n_observations
    assert result.attrs["expectation_type"] == "trials_tapers"
    assert result.attrs["backend"] in {"cpu", "gpu"}
    assert result.attrs["observations_are_independent"] == 1
    assert result.attrs["time_bins_are_independent"] == (name != "morlet")
    assert not any(
        key.startswith(("stft_", "welch_", "morlet_", "fourier_")) for key in result.attrs
    )
    if name != "multitaper":
        assert not any(key.startswith("mt_") for key in result.attrs)
    parameters = json.loads(result.attrs["transform_parameters_json"])
    assert set(parameters) == {"estimator", "execution"}
    common = {
        "transform",
        "sampling_frequency",
        "n_trials",
        "n_signals",
        "n_observations",
        "backend",
        "expectation_type",
    }
    assert common.isdisjoint(parameters["estimator"])
    assert common.isdisjoint(parameters["execution"])
    if name != "morlet":
        assert parameters["execution"] == {"fft_workers": None}
    else:
        assert parameters["estimator"]["smoothing_time"] is None
        assert parameters["estimator"]["smoothing_step"] is None
        assert parameters["estimator"]["frequencies"] == [4, 8, 16]


def test_only_released_multitaper_attributes_are_compatibility_copies(data):
    result = connectivity_to_xarray(
        Multitaper(
            data,
            64,
            time_halfbandwidth_product=2,
            time_window_duration=0.5,
            detrend_type=None,
            fft_workers=1,
            taper_weighting="eigen",
        ),
        method="power",
    )
    released = {
        "mt_detrend_type",
        "mt_is_low_bias",
        "mt_sampling_frequency",
        "mt_start_time",
        "mt_time_halfbandwidth_product",
        "mt_n_fft_samples",
        "mt_n_signals",
        "mt_n_tapers",
        "mt_n_time_samples_per_step",
        "mt_n_time_samples_per_window",
        "mt_n_trials",
        "mt_time_window_duration",
        "mt_time_window_step",
        "mt_frequency_resolution",
        "mt_nyquist_frequency",
    }
    assert {key for key in result.attrs if key.startswith("mt_")} == released
    assert result.attrs["mt_sampling_frequency"] == result.attrs["sampling_frequency"]
    assert result.attrs["mt_n_trials"] == result.attrs["n_trials"]
    assert result.attrs["mt_n_signals"] == result.attrs["n_signals"]
    assert result.attrs["mt_frequency_resolution"] == result.attrs["spectral_bandwidth"]
    assert result.attrs["mt_detrend_type"] == "None"
    assert json.loads(result.attrs["transform_parameters_json"]) == {
        "estimator": {
            "adaptive_max_iterations": 50,
            "adaptive_tolerance": 1e-8,
            "detrend_type": None,
            "is_low_bias": True,
            "n_fft_samples": 32,
            "n_tapers": 3,
            "n_time_samples_per_step": 32,
            "n_time_samples_per_window": 32,
            "start_time": 0.0,
            "taper_kind": "dpss",
            "taper_weighting": "eigen",
            "time_halfbandwidth_product": 2,
            "time_window_duration": 0.5,
            "time_window_step": 0.5,
        },
        "execution": {"fft_workers": 1},
    }


def test_equal_fft_grids_have_qualified_estimator_bandwidths(data):
    multitaper = connectivity_to_xarray(_transform("multitaper", data), "power")
    stft = connectivity_to_xarray(_transform("stft", data), "power")
    welch = connectivity_to_xarray(_transform("welch", data), "power")
    np.testing.assert_array_equal(multitaper.frequency, stft.frequency)
    np.testing.assert_array_equal(stft.frequency, welch.frequency)
    assert multitaper.attrs["spectral_bandwidth"] == 8
    assert (
        multitaper.attrs["spectral_bandwidth_definition"]
        == "full_dpss_concentration_bandwidth"
    )
    for result in (stft, welch):
        assert result.attrs["spectral_bandwidth"] == pytest.approx(3)
        assert result.attrs["spectral_bandwidth_definition"] == "equivalent_noise_bandwidth"
    for result in (multitaper, stft, welch):
        assert result.attrs["frequency_bin_spacing"] == 2
        assert result.attrs["frequency_bin_spacing_units"] == "Hz"
        assert result.attrs["spectral_bandwidth_units"] == "Hz"


@pytest.mark.parametrize("n_samples", [2, 3, 32])
def test_hann_bandwidth_matches_the_actual_window(data, n_samples):
    stft = ShortTimeFourierTransform(data, 64, n_time_samples_per_window=n_samples)
    welch = Welch(data, 64, n_time_samples_per_segment=n_samples)
    window = hann(n_samples, sym=False)
    expected = 64 * np.sum(window**2) / np.sum(window) ** 2
    assert stft.equivalent_noise_bandwidth == pytest.approx(expected)
    assert welch.equivalent_noise_bandwidth == pytest.approx(expected)
    assert not hasattr(stft, "frequency_resolution")
    with pytest.raises(AttributeError, match="equivalent_noise_bandwidth"):
        _ = stft.frequency_resolution


def test_familiar_multitaper_names_remain_the_single_supported_spelling(data):
    mt = _transform("multitaper", data)
    assert mt.frequency_resolution == 8
    assert estimate_frequency_resolution(64, 0.5, 2) == 8
    parameters = suggest_parameters(64, 10, desired_freq_resolution=2)
    assert parameters["frequency_resolution"] == 2
    assert "concentration_bandwidth" not in parameters
    assert not hasattr(mt, "concentration_bandwidth")


@pytest.mark.parametrize("method", ["power", ["power", "coherence_magnitude"]])
def test_spacing_describes_returned_grid_after_crop_and_decimation(data, method):
    result = multitaper_connectivity(
        data,
        64,
        time_window_duration=0.5,
        method=method,
        time_halfbandwidth_product=2,
        frequency_range=(4, 24),
        frequency_decimation=3,
    )
    np.testing.assert_array_equal(result.frequency, [4, 10, 16, 22])
    assert result.attrs["frequency_bin_spacing"] == 6
    assert result.attrs["spectral_bandwidth"] == 8
    if isinstance(result, xr.Dataset):
        for variable in result.data_vars.values():
            assert variable.attrs["transform"] == "multitaper"
            assert variable.attrs["frequency_bin_spacing"] == 6


@pytest.mark.parametrize("method", ["power", ["power", "coherence_magnitude"]])
def test_band_reduction_drops_obsolete_spacing_but_keeps_provenance(data, method):
    full = multitaper_connectivity(data, 64, time_window_duration=0.5, method=method)
    reduced = frequency_band_reduce(full, {"low": (4, 12)})
    assert "frequency_bin_spacing" not in reduced.attrs
    assert "frequency_bin_spacing_units" not in reduced.attrs
    assert reduced.attrs["transform"] == "multitaper"
    variables = reduced.data_vars.values() if isinstance(reduced, xr.Dataset) else [reduced]
    for variable in variables:
        assert "frequency_bin_spacing" not in variable.attrs
        assert variable.attrs["spectral_bandwidth"] == full.attrs["spectral_bandwidth"]


def test_irregular_singleton_and_reduced_frequency_outputs_have_no_spacing(data):
    morlet = connectivity_to_xarray(MorletWavelet(data, 64, [4, 7, 16]), "power")
    singleton = fourier_connectivity(
        np.ones((3, 1, 2), dtype=complex), frequencies=np.array([4.0]), method="power"
    )
    reduced = connectivity_to_xarray(_transform("multitaper", data), "group_delay")
    for result in (morlet, singleton, reduced):
        assert "frequency_bin_spacing" not in result.attrs
        if isinstance(result, xr.Dataset):
            assert all(
                "frequency_bin_spacing" not in variable.attrs
                for variable in result.data_vars.values()
            )


def test_custom_tapers_do_not_claim_a_dpss_bandwidth(data):
    mt = Multitaper(data, 64, n_time_samples_per_window=32, tapers=np.ones((32, 2)))
    result = connectivity_to_xarray(mt, "power")
    assert mt.frequency_resolution == 12  # Released nominal NW-based property.
    assert result.attrs["n_observations"] == 6
    assert result.attrs["mt_n_tapers"] == 2
    assert "spectral_bandwidth" not in result.attrs
    assert "spectral_bandwidth_definition" not in result.attrs
    assert (
        json.loads(result.attrs["transform_parameters_json"])["estimator"]["taper_kind"]
        == "custom"
    )


@pytest.mark.parametrize("physical", [False, True])
def test_external_fourier_metadata_only_infers_known_physical_facts(data, physical):
    mt = _transform("multitaper", data)
    result = fourier_connectivity(
        mt.fft(),
        frequencies=mt.frequencies if physical else None,
        method="power",
        is_one_sided=False,
    )
    assert result.attrs["transform"] == "external_fourier"
    assert result.attrs["n_observations"] == 9
    assert "n_trials" not in result.attrs
    assert "spectral_bandwidth" not in result.attrs
    if physical:
        assert result.attrs["sampling_frequency"] == 64
        assert result.attrs["frequency_bin_spacing"] == 2
        assert result.frequency.attrs["units"] == "Hz"
    else:
        assert "sampling_frequency" not in result.attrs
        assert result.attrs["frequency_bin_spacing"] == 1 / 32
        assert result.frequency.attrs["units"] == "cycles/sample"
        assert result.attrs["nyquist_frequency"] == 0.5


@pytest.mark.parametrize("external", [False, True])
def test_scalar_recording_identifiers_survive_without_trial_conditions(data, external):
    values = np.fft.fft(data, axis=0) if external else data
    frequency = np.fft.fftfreq(128, 1 / 64) if external else np.arange(128) / 64
    dimension = "frequency" if external else "time"
    labeled = xr.DataArray(
        values,
        dims=(dimension, "trial", "signal"),
        coords={
            dimension: frequency,
            "subject": ((), "rat-1", {"long_name": "Recording subject"}),
            "session": 3,
            "condition": ("trial", ["a", "b", "a"]),
        },
    )
    result = (
        fourier_connectivity(labeled, method="power")
        if external
        else multitaper_connectivity(labeled, method="power")
    )
    assert result.subject.item() == "rat-1"
    assert result.subject.attrs == {"long_name": "Recording subject"}
    assert result.session.item() == 3
    assert "condition" not in result.coords
    assert result.attrs["n_trials"] == 3


@pytest.mark.parametrize("trial_dimension", ["trial", "observations"])
def test_labeled_normalized_fourier_coordinates_keep_attrs_and_known_trials(
    data, trial_dimension
):
    coefficients = xr.DataArray(
        np.fft.fft(data, axis=0),
        dims=("frequency", trial_dimension, "signal"),
        coords={
            "frequency": (
                "frequency",
                np.fft.fftfreq(128),
                {"units": "cycles/sample", "long_name": "Recorded frequency"},
            ),
        },
    )
    result = fourier_connectivity(coefficients, method="power")
    assert result.frequency.attrs == coefficients.frequency.attrs
    assert result.attrs["frequency_units"] == "cycles/sample"
    assert result.attrs["frequency_bin_spacing_units"] == "cycles/sample"
    assert result.attrs["frequency_bin_spacing"] == 1 / 128
    assert "sampling_frequency" not in result.attrs
    if trial_dimension == "trial":
        assert result.attrs["n_trials"] == 3
    else:
        assert "n_trials" not in result.attrs


def test_provided_fourier_time_coordinates_keep_their_descriptions(data):
    values = np.fft.fft(data, axis=0).transpose(1, 0, 2)[np.newaxis, ...]
    coefficients = xr.DataArray(
        values,
        dims=("time", "trial", "frequency", "signal"),
        coords={
            "time": ("time", [2.0], {"units": "s", "long_name": "Epoch center"}),
            "frequency": np.fft.fftfreq(128, 1 / 64),
        },
    )
    result = fourier_connectivity(coefficients, method="power")
    assert result.time.attrs == coefficients.time.attrs


def test_coordinate_units_express_normalized_delay_without_claiming_seconds(data):
    coefficients = xr.DataArray(
        np.fft.fft(data, axis=0),
        dims=("frequency", "trial", "signal"),
        coords={"frequency": ("frequency", np.fft.fftfreq(128), {"units": "cycles/sample"})},
    )
    result = fourier_connectivity(coefficients, method="group_delay")
    assert result.group_delay.attrs["units"] == "samples"


def test_external_fourier_rejects_coordinate_units_that_require_conversion(data):
    coefficients = xr.DataArray(
        np.fft.fft(data, axis=0),
        dims=("frequency", "trial", "signal"),
        coords={"frequency": ("frequency", np.fft.fftfreq(128), {"units": "kHz"})},
    )
    with pytest.raises(ValueError, match="Hz or cycles/sample"):
        fourier_connectivity(coefficients, method="power")


@pytest.mark.parametrize("units", [None, "uV"])
def test_power_units_are_known_only_when_input_states_them(data, units):
    labeled = xr.DataArray(
        data, dims=("time", "trial", "signal"), attrs={} if units is None else {"units": units}
    )
    result = multitaper_connectivity(labeled, 64, method="power")
    assert result.attrs["long_name"]
    if units is None:
        assert "units" not in result.attrs
    else:
        assert result.attrs["units"] == "(uV)^2/Hz"


@pytest.mark.parametrize("engine", ["scipy", "netcdf4", "h5netcdf"])
@pytest.mark.parametrize("name", ["multitaper", "stft", "welch", "morlet"])
def test_transform_schema_roundtrips_with_each_engine(data, name, engine, tmp_path):
    if engine == "netcdf4":
        pytest.importorskip("netCDF4")
    elif engine == "h5netcdf":
        pytest.importorskip("h5netcdf")
    result = connectivity_to_xarray(_transform(name, data), "power")
    path = tmp_path / "schema.nc"
    result.to_netcdf(path, engine=engine)
    with xr.open_dataarray(path, engine=engine) as reloaded:
        xr.testing.assert_identical(reloaded.load(), result)
