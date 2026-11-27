"""Public result-schema contracts across transforms and frequency operations."""

import json
import re
import warnings
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from scipy.signal.windows import hann

from spectral_connectivity import (
    MorletWavelet,
    Multitaper,
    ShortTimeFourierTransform,
    VARSpectrum,
    Welch,
    estimate_frequency_resolution,
    list_measures,
    suggest_parameters,
)
from spectral_connectivity._measure_registry import _is_group_measure
from spectral_connectivity._provenance import _with_frequency_grid_metadata
from spectral_connectivity.wrapper import (
    connectivity_to_xarray,
    fourier_connectivity,
    frequency_band_reduce,
    multitaper_connectivity,
)

# Exact estimator settings of the Hann-window transforms: none are multitaper-only.
_HANN_ESTIMATOR_KEYS = {
    "stft": {
        "detrend_type",
        "n_fft_samples",
        "n_time_samples_per_step",
        "n_time_samples_per_window",
        "start_time",
        "time_window_duration",
        "time_window_step",
        "window",
    },
    "welch": {
        "detrend_type",
        "n_fft_samples",
        "n_segments",
        "n_time_samples_per_segment",
        "n_time_samples_per_step",
        "segment_duration",
        "segment_overlap",
        "start_time",
        "window",
    },
}


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
    return MorletWavelet(data, 64, frequencies=[4, 8, 16])


def _multitaper_without(field):
    """A Multitaper subclass whose settings record omits ``field``."""

    class Trimmed(Multitaper):
        _PROVENANCE_FIELDS = tuple(
            name for name in Multitaper._PROVENANCE_FIELDS if name != field
        )

    return Trimmed


class PrecomputedTransform:
    """Only the public SpectralTransform interface; a coordinate may be ``None``."""

    def __init__(self, coefficients, frequencies, time):
        self._coefficients = coefficients
        self.frequencies = frequencies
        self.time = time

    def fft(self):
        return self._coefficients.copy()


def _fourier_dataarray(data, frequency, units=None, trial_dimension="trial"):
    """Two-sided FFT coefficients of ``data`` labeled with ``frequency``."""
    attrs = {} if units is None else {"units": units}
    return xr.DataArray(
        np.fft.fft(data, axis=0),
        dims=("frequency", trial_dimension, "signal"),
        coords={"frequency": ("frequency", frequency, attrs)},
    )


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
    assert result.attrs["is_exact_spectrum"] == 0
    assert result.attrs["time_bins_are_independent"] == (name != "morlet")
    assert result.attrs["nyquist_frequency"] == 32
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
    if name in _HANN_ESTIMATOR_KEYS:
        assert set(parameters["estimator"]) == _HANN_ESTIMATOR_KEYS[name]
    if name != "morlet":
        assert parameters["execution"] == {"fft_workers": None}
    else:
        assert parameters["estimator"]["smoothing_time"] is None
        assert parameters["estimator"]["smoothing_step"] is None
        assert parameters["estimator"]["frequencies"] == [4, 8, 16]


@pytest.mark.parametrize(
    ("smoothing_step", "realized_step"),
    [(None, 0.12), (0.057, 0.06)],
    ids=["default", "given"],
)
def test_morlet_records_the_smoothing_window_and_step_it_used(
    data, smoothing_step, realized_step
):
    """Like the windowed transforms' durations, the recorded smoothing is what
    was used: whole decimated samples (12 and 6 of them at 100 Hz here), not
    the 0.123 s and 0.057 s requested."""
    transform = MorletWavelet(
        data,
        sampling_frequency=200,
        frequencies=np.array([20.0, 40.0]),
        decimation=2,
        smoothing_time=0.123,
        smoothing_step=smoothing_step,
    )
    result = connectivity_to_xarray(transform, "power")
    estimator = json.loads(result.attrs["transform_parameters_json"])["estimator"]
    assert estimator["smoothing_time"] == pytest.approx(0.12, rel=1e-12)
    assert estimator["smoothing_step"] == pytest.approx(realized_step, rel=1e-12)
    np.testing.assert_allclose(np.diff(result.time.values), realized_step, rtol=1e-12)


def test_morlet_records_a_whole_sample_smoothing_window_exactly():
    """100 samples of 3 ms are 0.3 s, not the 0.30000000000000004 that
    dividing by the decimated rate (1000 / 3) gives."""
    transform = MorletWavelet(
        np.random.default_rng(911).standard_normal((900, 3, 2)),
        sampling_frequency=1000,
        frequencies=np.array([50.0, 100.0]),
        decimation=3,
        smoothing_time=0.3,
    )
    estimator = json.loads(
        connectivity_to_xarray(transform, "power").attrs["transform_parameters_json"]
    )["estimator"]
    assert estimator["smoothing_time"] == 0.3
    assert estimator["smoothing_step"] == 0.3


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
    # Scalar types a 2.x reader compares against, not their string forms.
    assert type(result.attrs["mt_start_time"]) is float
    assert type(result.attrs["mt_sampling_frequency"]) is float
    assert all(type(result.attrs[key]) is int for key in released if key.startswith("mt_n_"))
    assert json.loads(result.attrs["transform_parameters_json"]) == {
        "estimator": {
            "adaptive_max_iterations": 50,
            "adaptive_tolerance": 1e-8,
            "detrend_type": None,
            "is_low_bias": True,
            "line_frequencies": [],
            "line_significance_threshold": None,
            "n_fft_samples": 32,
            "n_tapers": 3,
            "n_time_samples_per_step": 32,
            "requested_n_tapers": None,
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


def test_stft_has_no_multitaper_only_settings(data):
    """A Hann STFT is not a multitaper estimate with one taper."""
    stft = _transform("stft", data)
    assert not isinstance(stft, Multitaper)
    multitaper_only = {
        "frequency_resolution",
        "time_halfbandwidth_product",
        "n_tapers",
        "tapers",
        "taper_eigenvalues",
        "taper_weighting",
        "is_low_bias",
        "line_frequencies",
        "line_significance_threshold",
        "adaptive_max_iterations",
        "adaptive_tolerance",
    }
    assert multitaper_only <= set(dir(_transform("multitaper", data)))
    assert not multitaper_only & set(dir(stft))
    with pytest.raises(AttributeError):
        stft.sampling_frequency = 32


def test_familiar_multitaper_names_remain_the_single_supported_spelling(data):
    mt = _transform("multitaper", data)
    assert mt.frequency_resolution == 8
    assert estimate_frequency_resolution(64, 0.5, 2) == 8
    parameters = suggest_parameters(64, 10, frequency_resolution=2)
    assert parameters["frequency_resolution"] == 2
    assert "concentration_bandwidth" not in parameters
    assert not hasattr(mt, "concentration_bandwidth")


@pytest.mark.parametrize("method", ["power", ["power", "magnitude_squared_coherence"]])
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


@pytest.mark.parametrize("method", ["power", ["power", "magnitude_squared_coherence"]])
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


def test_spacing_is_per_variable_in_a_mixed_dataset(data):
    result = multitaper_connectivity(
        data, 64, time_window_duration=0.5, method=["power", "group_delay"]
    )
    assert result.power.attrs["frequency_bin_spacing"] == 2
    for name, variable in result.data_vars.items():
        if "frequency" not in variable.dims:
            assert "frequency_bin_spacing" not in variable.attrs, name


def test_irregular_singleton_and_reduced_frequency_outputs_have_no_spacing(data):
    morlet = connectivity_to_xarray(MorletWavelet(data, 64, frequencies=[4, 7, 16]), "power")
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
    # The 5-D layout names its trial axis.
    assert result.attrs["n_trials"] == 3
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


@pytest.mark.parametrize(
    ("coefficients", "n_trials"),
    [
        (np.ones((3, 4, 2), dtype=complex), None),  # (observation, frequency, signal)
        (np.ones((3, 1, 4, 2), dtype=complex), 3),  # (trial, taper, frequency, signal)
    ],
)
def test_numpy_fourier_layouts_record_trials_only_when_named(coefficients, n_trials):
    result = fourier_connectivity(coefficients, method="power", is_one_sided=False)
    assert result.attrs.get("n_trials") == n_trials
    assert result.attrs["n_observations"] == 3


def test_sidedness_of_input_coefficients_is_not_a_top_level_fact(data):
    mt = _transform("multitaper", data)
    morlet = connectivity_to_xarray(_transform("morlet", data), "power")
    external = fourier_connectivity(
        mt.fft(), frequencies=mt.frequencies, method="power", is_one_sided=False
    )
    assert "is_one_sided" not in connectivity_to_xarray(mt, "power").attrs
    assert "is_one_sided" not in morlet.attrs
    assert "is_one_sided" not in external.attrs
    estimator = json.loads(external.attrs["transform_parameters_json"])["estimator"]
    assert estimator["is_one_sided"] is False


def test_legacy_copies_omit_fields_a_subclass_does_not_record(data):
    result = connectivity_to_xarray(_multitaper_without("start_time")(data, 64), "power")
    assert "mt_start_time" not in result.attrs
    assert result.attrs["mt_sampling_frequency"] == 64


@pytest.mark.parametrize(
    ("frequency", "spacing"),
    [
        (np.fft.rfftfreq(3_600_000, 1 / 30_000), 1 / 120),  # 120 s at 30 kHz
        (np.fft.rfftfreq(3000, 1 / 1000).astype(np.float32), 1 / 3),  # e.g. from netCDF
    ],
)
def test_spacing_allows_for_the_rounding_of_uniform_grids(frequency, spacing):
    grid = xr.DataArray(
        np.zeros(frequency.size), dims="frequency", coords={"frequency": frequency}
    )
    recorded = _with_frequency_grid_metadata(grid).attrs["frequency_bin_spacing"]
    assert recorded == pytest.approx(spacing, rel=1e-6)


@pytest.mark.parametrize(
    "frequency",
    [
        np.array([0.0, 2.0, 4.0, 6.0, 8.001]),
        np.array([0.0, np.inf, 2.0]),
        # One bin moved by 12% of a step, 15 float32 units in the last place.
        np.where(np.arange(64) == 30, 0.03, 0.0).astype(np.float32)
        + (20_000 + 0.25 * np.arange(64)).astype(np.float32),
        np.geomspace(100, 101, 50).astype(np.float32),
        np.round(np.arange(4, 20, 1 / 3), 4),
    ],
    ids=[
        "nearly_uniform",
        "non_finite",
        "displaced_float32_bin",
        "log_spaced_float32",
        "rounded_to_4_decimals",
    ],
)
def test_spacing_is_omitted_for_a_grid_that_is_not_uniform(frequency):
    """Irregularity beyond the coordinate's own precision is not rounding."""
    grid = xr.DataArray(
        np.zeros(frequency.size), dims="frequency", coords={"frequency": frequency}
    )
    assert "frequency_bin_spacing" not in _with_frequency_grid_metadata(grid).attrs


def test_normalized_grid_keeps_its_nyquist_bin_after_cropping(data):
    mt = _transform("multitaper", data)
    full = fourier_connectivity(mt.fft(), method="power", is_one_sided=False)
    cropped = full.sel(frequency=slice(0, 0.4375))
    band = {"upper": (0.375, 0.4375)}
    xr.testing.assert_allclose(
        frequency_band_reduce(cropped, band, frequency_reduction="integral"),
        frequency_band_reduce(full, band, frequency_reduction="integral"),
        rtol=1e-12,
    )


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


@pytest.mark.parametrize("method", ["power", ["power", "magnitude_squared_coherence"]])
def test_band_reduction_discards_input_coordinates_named_like_its_own(data, method):
    """An input scalar named like a band coordinate never describes the bands."""
    labeled = xr.DataArray(
        data,
        dims=("time", "trial", "signal"),
        coords={
            "band": "theta",
            "band_lower": -1.0,
            "band_upper": 99.0,
            "valid_time_band": True,
            "subject": "rat-1",
        },
    )
    bands = {"alpha": (8, 12), "beta": (13, 30)}
    unreduced = multitaper_connectivity(labeled, 64, method=method)
    assert unreduced.band.item() == "theta"
    discarded = r"\['band', 'band_lower', 'band_upper', 'valid_time_band'\]"
    with pytest.warns(UserWarning, match=discarded):
        reduced_afterward = frequency_band_reduce(unreduced, bands)
    with pytest.warns(UserWarning, match=discarded):
        reduced_by_wrapper = multitaper_connectivity(
            labeled, 64, method=method, frequency_bands=bands
        )
    for reduced in (reduced_afterward, reduced_by_wrapper):
        assert reduced.band.values.tolist() == ["alpha", "beta"]
        assert reduced.band_lower.values.tolist() == [8.0, 13.0]
        assert reduced.band_upper.values.tolist() == [12.0, 30.0]
        # Multitaper estimates have no validity mask, so nothing replaces it.
        assert "valid_time_band" not in reduced.coords
        assert reduced.subject.item() == "rat-1"


def test_nyquist_follows_from_the_sampling_rate_when_the_transform_omits_it(data):
    """Cropping must not turn an interior bin into the Nyquist bin."""
    full = connectivity_to_xarray(_multitaper_without("nyquist_frequency")(data, 64), "power")
    assert full.attrs["nyquist_frequency"] == 32
    assert "mt_nyquist_frequency" not in full.attrs
    band = {"upper": (4, 8)}
    xr.testing.assert_allclose(
        frequency_band_reduce(
            full.sel(frequency=slice(0, 8)), band, frequency_reduction="integral"
        ),
        frequency_band_reduce(full, band, frequency_reduction="integral"),
        rtol=1e-12,
    )


@pytest.mark.parametrize("recorded", ["nyquist_frequency", "mt_nyquist_frequency"])
def test_dataset_built_from_result_variables_uses_their_nyquist_frequency(data, recorded):
    """A Dataset assembled from result variables has no attributes of its own;
    each variable's record, under the 3.0 name or the 2.x one, still says a
    cropped grid's last bin is interior."""
    full = multitaper_connectivity(data, sampling_frequency=64, method="power")
    band = {"upper": (4, 8)}
    cropped = full.sel(frequency=slice(0, 8))
    for name in {"nyquist_frequency", "mt_nyquist_frequency"} - {recorded}:
        del cropped.attrs[name]
    assert cropped.attrs[recorded] == 32
    assembled = xr.Dataset({"power": cropped})
    assert "nyquist_frequency" not in assembled.attrs
    xr.testing.assert_allclose(
        frequency_band_reduce(assembled, band, frequency_reduction="integral")["power"],
        frequency_band_reduce(full, band, frequency_reduction="integral"),
        rtol=1e-12,
    )


def test_group_membership_carries_only_shared_provenance():
    """The membership table is shared by every group measure of a result."""
    signals = np.random.default_rng(911).standard_normal((128, 6, 4))
    result = multitaper_connectivity(
        signals,
        64,
        method=["canonical_coherency", "maximized_imaginary_coherency_components"],
        group_labels=["a", "a", "b", "b"],
    )
    membership_attrs = result.group_membership.attrs
    assert membership_attrs["output_schema_version"] == 1
    assert membership_attrs["transform"] == "multitaper"
    assert not {key for key in membership_attrs if key.startswith(("measure", "arg_"))}
    assert result.canonical_coherency.attrs["measure"] == "canonical_coherency"


@pytest.mark.parametrize("has_frequencies", [False, True])
@pytest.mark.parametrize("has_time", [False, True])
def test_transform_with_only_the_public_interface_is_labeled_truthfully(
    data, has_frequencies, has_time
):
    """``frequencies=None`` and ``time=None`` mean cycles/sample and window indices."""
    builtin = _transform("multitaper", data)
    transform = PrecomputedTransform(
        builtin.fft(),
        builtin.frequencies if has_frequencies else None,
        builtin.time if has_time else None,
    )
    result = connectivity_to_xarray(transform, "power")
    delay = connectivity_to_xarray(transform, "group_delay")
    assert result.attrs["transform"] == "PrecomputedTransform"
    assert result.attrs["output_schema_version"] == 1
    assert "sampling_frequency" not in result.attrs
    assert not any(key.startswith("mt_") for key in result.attrs)
    # The copied coefficients are contiguous, so sums round differently.
    np.testing.assert_allclose(
        result.values, connectivity_to_xarray(builtin, "power").values, rtol=1e-12
    )
    if has_frequencies:
        assert result.frequency.attrs["units"] == "Hz"
        assert result.attrs["frequency_bin_spacing_units"] == "Hz"
        assert "frequency_units" not in result.attrs
        assert delay.group_delay.attrs["units"] == "s"
    else:
        assert result.frequency.values[-1] == 0.5
        assert result.frequency.attrs["units"] == "cycles/sample"
        assert result.attrs["frequency_units"] == "cycles/sample"
        assert result.attrs["frequency_bin_spacing_units"] == "cycles/sample"
        assert result.attrs["nyquist_frequency"] == 0.5
        assert delay.group_delay.attrs["units"] == "samples"
    if has_time:
        assert result.time.attrs["units"] == "s"
    else:
        assert result.time.values.tolist() == list(range(builtin.time.size))
        assert "units" not in result.time.attrs
        assert result.attrs["time_coordinate"] == "index"


def test_rounded_uniform_grid_is_uniform_for_both_spacing_and_phase_slope(data):
    """One definition of a uniform grid serves the metadata and the measures."""
    frequencies = np.round(np.arange(4, 20, 1 / 3), 7)
    wavelet = MorletWavelet(data, 64, frequencies=frequencies)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = connectivity_to_xarray(wavelet, "phase_slope_index")
        power = connectivity_to_xarray(wavelet, "power")
    assert np.all(np.isfinite(result.isel(source=0, target=1)))
    assert power.attrs["frequency_bin_spacing"] == pytest.approx(1 / 3, rel=1e-6)


def test_rounded_two_sided_grid_is_in_fft_order(data):
    """A two-sided grid is held to the same uniformity as a one-sided one."""
    coefficients = _fourier_dataarray(data, np.round(np.fft.fftfreq(128, 3 / 128), 7))
    result = fourier_connectivity(coefficients, method="power")
    assert result.attrs["frequency_bin_spacing"] == pytest.approx(1 / 3, rel=1e-6)
    assert result.attrs["sampling_frequency"] == pytest.approx(128 / 3, rel=1e-6)


def _power_up_to(result, top):
    """Power integrated from 0 Hz to ``top``, as a band ending on that bin."""
    return frequency_band_reduce(
        result, {"all": (0.0, top)}, frequency_reduction="integral"
    ).values


@pytest.mark.parametrize("n_fft", [90, 89], ids=["even", "odd"])
@pytest.mark.parametrize(
    "stored",
    [lambda grid: grid.astype(np.float32), lambda grid: np.round(grid, 7)],
    ids=["float32", "rounded"],
)
def test_two_sided_grid_rounding_does_not_lose_the_nyquist_bin(data, n_fft, stored):
    """The rate is read from the bin farthest from 0 Hz, the Nyquist bin of an
    even-length grid, so one bin's rounding is not multiplied by the FFT
    length and a band ending on that bin still integrates its whole cell."""
    data = data[:n_fft]
    exact_grid = np.fft.fftfreq(n_fft, 1 / 250)
    exact = fourier_connectivity(_fourier_dataarray(data, exact_grid), method="power")
    result = fourier_connectivity(_fourier_dataarray(data, stored(exact_grid)), method="power")

    assert result.attrs["sampling_frequency"] == pytest.approx(250, rel=1e-6)
    if n_fft % 2 == 0:
        assert result.attrs["sampling_frequency"] == 250
        assert result.attrs["nyquist_frequency"] == float(result.frequency[-1])
    np.testing.assert_allclose(
        _power_up_to(result, float(result.frequency[-1])),
        _power_up_to(exact, float(exact.frequency[-1])),
        rtol=1e-6,
    )


@pytest.mark.parametrize(
    "stored",
    [lambda frequency: frequency.astype(np.float32), lambda frequency: frequency.round(7)],
    ids=["float32", "rounded"],
)
def test_stored_frequency_coordinate_keeps_its_nyquist_bin(data, stored):
    """A result's coordinate stored as float32 (e.g. by netCDF) or rounded to a
    few decimals is still on the recorded Nyquist frequency, 64/3 Hz here,
    which neither holds exactly. NumPy 2 compares a float32 bin with the
    recorded value in float32, so the float32 case fails without the rule only
    on NumPy 1.x; the rounded case fails without it on any."""
    result = multitaper_connectivity(data, sampling_frequency=128 / 3, method="power")
    nyquist = result.attrs["nyquist_frequency"]
    rounded = result.assign_coords(frequency=stored(result.frequency))
    assert float(rounded.frequency[-1]) != nyquist
    np.testing.assert_allclose(
        _power_up_to(rounded, nyquist), _power_up_to(result, nyquist), rtol=1e-6
    )


def test_float32_grid_of_an_odd_length_records_its_rate_in_its_own_precision(data):
    """An odd-length grid has no Nyquist bin to read the rate from. Its top
    bin, 100/3 Hz, is not a float32 number, but the rate it gives is 100 in the
    grid's own precision."""
    grid = np.fft.fftfreq(3, 1 / 100).astype(np.float32)
    result = fourier_connectivity(_fourier_dataarray(data[:3], grid), method="power")
    assert result.attrs["sampling_frequency"] == 100.0
    assert result.attrs["nyquist_frequency"] == 50.0


def test_integer_grid_reaching_its_dtype_minimum_records_a_positive_rate():
    """The Nyquist bin of this int8 grid is -128, whose absolute value int8
    cannot hold; the rate is taken after converting it to floating point."""
    grid = np.fft.fftfreq(256, 1 / 256).astype(np.int8)
    assert grid[128] == -128
    coefficients = _fourier_dataarray(
        np.random.default_rng(912).standard_normal((256, 3, 2)), grid
    )
    with warnings.catch_warnings():
        # Other steps still overflow on such a grid; only the rate is checked.
        warnings.simplefilter("ignore")
        result = fourier_connectivity(coefficients, method="power")
    assert result.attrs["sampling_frequency"] == 256.0
    assert result.attrs["nyquist_frequency"] == 128.0


def test_float32_uniform_grid_is_uniform_for_phase_slope(data):
    """A coordinate's own rounding (e.g. float32 from netCDF) is not irregularity."""
    coefficients = xr.DataArray(
        np.fft.rfft(data, axis=0),
        dims=("frequency", "trial", "signal"),
        coords={"frequency": (400 + np.arange(65) / 3).astype(np.float32)},
    )
    result = fourier_connectivity(coefficients, method=["power", "phase_slope_index"])
    assert result.power.attrs["frequency_bin_spacing"] == pytest.approx(1 / 3, rel=1e-6)
    assert "phase_slope_index" in result


def test_scalar_coordinate_named_like_a_measure_does_not_replace_it(data):
    labeled = xr.DataArray(
        data, dims=("time", "trial", "signal"), coords={"power": 5, "subject": "rat-1"}
    )
    with pytest.warns(UserWarning, match=r"\['power'\]"):
        result = multitaper_connectivity(
            labeled, 64, method=["power", "magnitude_squared_coherence"]
        )
    assert "power" in result.data_vars
    assert "frequency" in result["power"].dims
    assert result.subject.item() == "rat-1"


@pytest.mark.parametrize(
    ("trial_dimension", "trial_dim", "n_trials"),
    [
        ("trial", None, 3),
        ("observations", None, None),  # A generic observation name.
        ("repeat", None, None),  # Inferred by elimination.
        ("observations", "observations", 3),  # Declared explicitly.
    ],
)
def test_labeled_normalized_fourier_coordinates_keep_attrs_and_known_trials(
    data, trial_dimension, trial_dim, n_trials
):
    coefficients = _fourier_dataarray(
        data, np.fft.fftfreq(128), "cycles/sample", trial_dimension
    )
    coefficients.frequency.attrs["long_name"] = "Recorded frequency"
    result = fourier_connectivity(coefficients, method="power", trial_dim=trial_dim)
    assert result.frequency.attrs == coefficients.frequency.attrs
    assert result.attrs["frequency_units"] == "cycles/sample"
    assert result.attrs["frequency_bin_spacing_units"] == "cycles/sample"
    assert result.attrs["frequency_bin_spacing"] == 1 / 128
    assert "sampling_frequency" not in result.attrs
    assert result.attrs.get("n_trials") == n_trials


@pytest.mark.parametrize("select_one_window", [False, True])
@pytest.mark.parametrize("units", ["s", ""])
def test_fourier_time_coordinate_keeps_its_description_and_stated_units(
    data, select_one_window, units
):
    """Blank units state nothing, so the result's default of seconds applies."""
    values = np.fft.fft(data, axis=0).transpose(1, 0, 2)[np.newaxis, ...]
    coefficients = xr.DataArray(
        values,
        dims=("time", "trial", "frequency", "signal"),
        coords={
            "time": ("time", [2.0], {"units": units, "long_name": "Epoch center"}),
            "frequency": np.fft.fftfreq(128, 1 / 64),
        },
    )
    if select_one_window:
        coefficients = coefficients.isel(time=0)
    result = fourier_connectivity(coefficients, method="power")
    assert result.time.attrs == {"long_name": "Epoch center", "units": "s"}


def test_scalar_fourier_time_coordinate_is_the_time_of_the_single_window(data):
    """Selecting one window keeps its time instead of a window index."""
    values = np.stack([np.fft.fft(data, axis=0).transpose(1, 0, 2)] * 2)
    coefficients = xr.DataArray(
        values,
        dims=("time", "trial", "frequency", "signal"),
        coords={
            "time": ("time", [2.0, 4.0], {"units": "s", "long_name": "Epoch center"}),
            "frequency": np.fft.fftfreq(128, 1 / 64),
            "subject": "rat-1",
        },
    )
    one_window = coefficients.isel(time=1)
    assert one_window.time.dims == ()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = fourier_connectivity(one_window, method="power")
    xr.testing.assert_identical(
        result, fourier_connectivity(coefficients.isel(time=[1]), method="power")
    )
    assert result.time.values.tolist() == [4.0]
    assert result.time.attrs == coefficients.time.attrs
    assert result.attrs["time_coordinate"] == "provided"
    assert result.subject.item() == "rat-1"
    with pytest.raises(ValueError, match="time conflicts"):
        fourier_connectivity(one_window, time=[3.0], method="power")


@pytest.mark.parametrize(
    ("name", "value"), [("window", 256), ("windows", 8), ("time_window", 0.5)]
)
def test_scalar_coordinate_named_like_a_window_is_not_taken_for_its_time(data, name, value):
    """Only a scalar coordinate named time or times is the single window's
    time. A window's length or count is as likely under these names as its
    time, so they stay recording identifiers and the window keeps its index."""
    coefficients = _fourier_dataarray(data, np.fft.fftfreq(128, 1 / 64)).assign_coords(
        {name: value}
    )
    result = fourier_connectivity(coefficients, method="power")
    assert result.time.values.tolist() == [0]
    assert result.attrs["time_coordinate"] == "index"
    assert result.coords[name].item() == value


def test_coordinate_units_express_normalized_delay_without_claiming_seconds(data):
    coefficients = _fourier_dataarray(data, np.fft.fftfreq(128), "cycles/sample")
    result = fourier_connectivity(coefficients, method=["group_delay", "delay"])
    for delay in (result.group_delay, result.delay):
        assert delay.attrs["units"] == "samples"
        assert "in seconds" not in delay.attrs["interpretation"]


@pytest.mark.parametrize(
    ("units", "canonical"),
    [
        ("hz", "Hz"),
        ("Hertz", "Hz"),
        ("1/s", "Hz"),
        ("s-1", "Hz"),  # CF/UDUNITS
        ("s**-1", "Hz"),
        ("sec^-1", "Hz"),
        ("1 / second", "Hz"),
        ("cycles/s", "Hz"),
        (" Cycles/Sample ", "cycles/sample"),
    ],
)
def test_external_fourier_accepts_spellings_of_supported_units(data, units, canonical):
    normalized = canonical == "cycles/sample"
    coefficients = _fourier_dataarray(
        data, np.fft.fftfreq(128, 1 if normalized else 1 / 64), units
    )
    result = fourier_connectivity(coefficients, method="power")
    assert result.attrs["frequency_units"] == canonical
    assert result.frequency.attrs["units"] == units
    if normalized:
        assert "sampling_frequency" not in result.attrs
    else:
        assert result.attrs["sampling_frequency"] == 64


@pytest.mark.parametrize("units", ["", "  "])
def test_external_fourier_treats_empty_coordinate_units_as_unstated(data, units):
    coefficients = _fourier_dataarray(data, np.fft.fftfreq(128, 1 / 64), units)
    result = fourier_connectivity(coefficients, method="power")
    assert result.attrs["frequency_units"] == "Hz"
    assert result.attrs["sampling_frequency"] == 64
    assert result.frequency.attrs["units"] == "Hz"


def test_external_fourier_rejects_coordinate_units_that_require_conversion(data):
    coefficients = _fourier_dataarray(data, np.fft.fftfreq(128), "kHz")
    with pytest.raises(ValueError, match="Hz or cycles/sample"):
        fourier_connectivity(coefficients, method="power")


@pytest.mark.parametrize("units", [None, "  ", "uV"])
def test_power_units_are_known_only_when_input_states_them(data, units):
    labeled = xr.DataArray(
        data, dims=("time", "trial", "signal"), attrs={} if units is None else {"units": units}
    )
    result = multitaper_connectivity(labeled, 64, method="power")
    assert result.attrs["long_name"]
    if units == "uV":
        assert result.attrs["units"] == "(uV)^2/Hz"
    else:
        assert "units" not in result.attrs


_COOKBOOK = Path(__file__).parents[1] / "docs" / "cookbook.md"
_SCHEMA_SECTION = "## Result schema and transform bandwidth"
_COMPATIBILITY_LIST_START = "remain compatibility copies"
# Measures that need a uniform frequency grid, which a Morlet transform lacks.
_UNIFORM_GRID_CATEGORIES = {"delay", "group_delay", "phase_slope"}


def _documented_attribute_names():
    """The names the cookbook's schema reference documents, read from its text:
    the first column of its attribute table and the list of ``mt_*``
    compatibility keys (only that list: the prose after it mentions some keys
    again, and must not document one the list dropped)."""
    text = _COOKBOOK.read_text(encoding="utf-8")
    section = text.split(_SCHEMA_SECTION, 1)[1].split("\n## ", 1)[0]
    table_names = [
        name
        for row in section.splitlines()
        if row.startswith("| `")
        for name in re.findall(r"`([^`]+)`", row.split("|")[1])
    ]
    compatibility_list = section.split(_COMPATIBILITY_LIST_START, 1)[1].split(".\n", 1)[0]
    return sorted(set(table_names + re.findall(r"`(mt_\w+)`", compatibility_list)))


def _name_pattern(name):
    """``arg_<name>`` stands for any keyword name; other names are literal."""
    return re.compile(re.escape(name).replace("<name>", r"\w+") + "$")


def _attribute_names(result):
    names = set(result.attrs)
    if isinstance(result, xr.Dataset):
        for variable in result.data_vars.values():
            names |= set(variable.attrs)
    return names


def _probe_results(data):
    """Results of every transform with every measure it supports, and of each
    wrapper option that writes attributes of its own."""
    labels = ["a", "a", "b", "b"]
    for transform_name in ("multitaper", "stft", "welch", "morlet"):
        transform = _transform(transform_name, data)
        for info in list_measures():
            if transform_name == "morlet" and (
                info.requires_two_sided or info.category in _UNIFORM_GRID_CATEGORIES
            ):
                continue
            settings = {}
            if _is_group_measure(info.name):
                settings["group_labels"] = labels
            if info.name == "subset_pairwise_spectral_granger_prediction":
                settings["connectivity_kwargs"] = {"pairs": [(0, 1)]}
            yield connectivity_to_xarray(transform, info.name, **settings)
    var = VARSpectrum(np.array([[[0.5, 0.0], [0.4, 0.5]]]), sampling_frequency=64)
    yield connectivity_to_xarray(var, "magnitude_squared_coherence")

    common = {"sampling_frequency": 64, "time_window_duration": 0.5}
    yield multitaper_connectivity(
        data,
        **common,
        method=["power", "magnitude_squared_coherence", "delay"],
        frequency_range=(4, 20),
        frequency_decimation=2,
    )
    for reduction in ("mean", "integral"):
        yield multitaper_connectivity(
            data,
            **common,
            method="power",
            frequency_bands={"theta": (4, 8)},
            frequency_reduction=reduction,
        )
    with pytest.warns(DeprecationWarning, match="coherence_magnitude"):
        deprecated_name = multitaper_connectivity(data, **common, method="coherence_magnitude")
    yield deprecated_name
    multitaper = _transform("multitaper", data)
    with pytest.warns(DeprecationWarning, match="frequencies_of_interest"):
        open_band = connectivity_to_xarray(
            multitaper, "delay", connectivity_kwargs={"frequencies_of_interest": (4, np.inf)}
        )
    yield open_band
    labeled = xr.DataArray(
        data,
        dims=("time", "trial", "signal"),
        coords={"time": np.arange(data.shape[0]) / 64, "subject": "s1"},
        attrs={"units": "uV", "note": "recorded"},
    )
    yield multitaper_connectivity(labeled, time_window_duration=0.5, method="power")
    yield multitaper_connectivity(
        data, **common, method="power", transform_kwargs={"tapers": np.ones((32, 1))}
    )
    coefficients = multitaper.fft()
    yield fourier_connectivity(coefficients, method="power")
    yield fourier_connectivity(coefficients, multitaper.frequencies, multitaper.time, "power")


def test_schema_reference_documents_exactly_the_attributes_results_carry():
    """Every attribute a result carries is named by the cookbook's schema
    reference, and every name the reference documents is carried by some
    result. Update the table in ``docs/cookbook.md`` when either fails."""
    data = np.random.default_rng(910).standard_normal((128, 3, 4))
    documented = _documented_attribute_names()
    with warnings.catch_warnings():
        # Warnings about the random data (orientation reminder, correlated
        # innovations, few observations) are not what this test is about.
        warnings.simplefilter("ignore", UserWarning)
        carried = set().union(*(_attribute_names(result) for result in _probe_results(data)))
    patterns = {name: _name_pattern(name) for name in documented}
    undocumented = sorted(
        name for name in carried if not any(p.match(name) for p in patterns.values())
    )
    never_carried = sorted(
        name
        for name, pattern in patterns.items()
        if not any(pattern.match(attribute) for attribute in carried)
    )
    assert undocumented == []
    assert never_carried == []
