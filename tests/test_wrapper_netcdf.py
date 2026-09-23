"""Every labeled result must round-trip through each available NetCDF engine.

xarray writes with netCDF4 when it is installed, h5netcdf otherwise, and falls
back to SciPy (NetCDF3). The engines accept different attribute and data types
(netCDF4 and h5netcdf reject booleans; NetCDF3 rejects complex data), so each
real-valued output is written and read back with every engine and must come
back identical, attrs included.
"""

import warnings

import numpy as np
import pytest
import xarray as xr

from spectral_connectivity import MorletWavelet, fourier_connectivity, multitaper_connectivity
from spectral_connectivity.wrapper import (
    _MEASURE_SPECS,
    connectivity_to_xarray,
    frequency_band_reduce,
)

_GROUPS = ["a", "a", "b", "b"]
_MEASURE_KWARGS = {
    "canonical_coherence": {"group_labels": _GROUPS},
    "maximized_imaginary_coherency": {"group_labels": _GROUPS},
    "multivariate_interaction_measure": {"group_labels": _GROUPS},
    "blockwise_spectral_granger_prediction": {"group_labels": _GROUPS},
    "canonical_coherency": {"group_labels": _GROUPS},
    "maximized_imaginary_coherency_components": {"group_labels": _GROUPS},
    "subset_pairwise_spectral_granger_prediction": {"pairs": [(0, 1)]},
}


def _engine(name):
    """Skip unless the engine's backend library is importable."""
    if name == "netcdf4":
        pytest.importorskip("netCDF4")
    elif name == "h5netcdf":
        pytest.importorskip("h5netcdf")
    return name


ENGINES = ["scipy", "netcdf4", "h5netcdf"]


def _is_complex(obj):
    variables = obj.data_vars.values() if isinstance(obj, xr.Dataset) else [obj]
    return any(np.iscomplexobj(variable) for variable in variables)


def _roundtrip(obj, path, engine, **write_kwargs):
    obj.to_netcdf(path, engine=engine, **write_kwargs)
    opener = xr.open_dataset if isinstance(obj, xr.Dataset) else xr.open_dataarray
    with opener(path, engine=engine) as loaded:
        return loaded.load()


@pytest.fixture(scope="module")
def time_series():
    return np.random.default_rng(0).standard_normal((1024, 4, 4))


@pytest.fixture(scope="module")
def measure_results(time_series):
    results = {}
    with warnings.catch_warnings():
        # Some measures warn about degenerate bins of random data; irrelevant here.
        warnings.simplefilter("ignore", UserWarning)
        for method in _MEASURE_SPECS:
            results[method] = multitaper_connectivity(
                time_series,
                sampling_frequency=500,
                time_window_duration=0.5,
                method=method,
                connectivity_kwargs=_MEASURE_KWARGS.get(method, {}),
            )
    return results


@pytest.fixture(scope="module")
def composite_results(time_series):
    rng = np.random.default_rng(1)
    two_sided = np.fft.fft(rng.standard_normal((20, 64, 3)), axis=1)
    one_sided = np.fft.rfft(rng.standard_normal((20, 64, 3)), axis=1)
    morlet = connectivity_to_xarray(
        MorletWavelet(
            rng.standard_normal((500, 3, 3)),
            250,
            frequencies=[4, 8, 16],
            smoothing_time=0.2,
            edge_mode="nan",
        ),
        "coherence_magnitude",
    )
    return {
        "default_dataset": multitaper_connectivity(time_series, sampling_frequency=500),
        "band_reduced": multitaper_connectivity(
            time_series,
            sampling_frequency=500,
            method="coherence_magnitude",
            frequency_bands={"theta": (4, 8), "gamma": (30, 50)},
        ),
        "squeezed": multitaper_connectivity(
            time_series[..., :2],
            sampling_frequency=500,
            method="coherence_magnitude",
            squeeze=True,
        ),
        "fourier_two_sided": fourier_connectivity(
            two_sided,
            frequencies=np.fft.fftfreq(64, 1 / 100),
            method="coherence_magnitude",
        ),
        "fourier_one_sided": fourier_connectivity(
            one_sided,
            frequencies=np.fft.rfftfreq(64, 1 / 100),
            method="coherence_magnitude",
        ),
        "fourier_unlabeled": fourier_connectivity(
            two_sided, method="coherence_magnitude", is_one_sided=False
        ),
        "morlet_validity": morlet,
        "labeled_input": multitaper_connectivity(
            xr.DataArray(
                time_series[:, 0, :],
                dims=("time", "channel"),
                coords={
                    "channel": ["a", "b", "c", "d"],
                    "region": ("channel", ["CA1", "CA1", "PFC", "PFC"]),
                },
                attrs={"units": "uV", "subject": "rat-1"},
            ),
            sampling_frequency=500,
            method=["power", "coherence_magnitude"],
        ),
        "morlet_band": frequency_band_reduce(morlet, {"lo": (4, 8)}),
    }


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("method", sorted(_MEASURE_SPECS))
def test_every_measure_roundtrips(measure_results, method, engine, tmp_path):
    result = measure_results[method]
    if _is_complex(result):
        pytest.skip("complex data needs an engine option; see test_complex_results_*")
    loaded = _roundtrip(result, tmp_path / "result.nc", _engine(engine))
    xr.testing.assert_identical(loaded, result)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize(
    "name",
    [
        "default_dataset",
        "band_reduced",
        "squeezed",
        "fourier_two_sided",
        "fourier_one_sided",
        "fourier_unlabeled",
        "morlet_validity",
        "morlet_band",
        "labeled_input",
    ],
)
def test_composite_results_roundtrip(composite_results, name, engine, tmp_path):
    result = composite_results[name]
    loaded = _roundtrip(result, tmp_path / "result.nc", _engine(engine))
    xr.testing.assert_identical(loaded, result)


def _complex_measures(measure_results):
    return [name for name, result in measure_results.items() if _is_complex(result)]


def test_complex_results_roundtrip_with_h5netcdf_invalid_netcdf(measure_results, tmp_path):
    """The documented recipe for complex results: h5netcdf with invalid_netcdf.

    The file is not standard NetCDF-4, so it gets an ``.h5`` extension (h5netcdf
    warns for ``.nc``)."""
    _engine("h5netcdf")
    complex_measures = _complex_measures(measure_results)
    assert {"coherency", "cross_spectral_density"} <= set(complex_measures)
    for method in complex_measures:
        result = measure_results[method]
        loaded = _roundtrip(result, tmp_path / f"{method}.h5", "h5netcdf", invalid_netcdf=True)
        xr.testing.assert_identical(loaded, result)


def test_complex_results_roundtrip_with_netcdf4_auto_complex(measure_results, tmp_path):
    """The documented recipe for complex results: netCDF4 with auto_complex."""
    netcdf4 = pytest.importorskip("netCDF4")
    if not hasattr(netcdf4.Dataset, "__init__") or netcdf4.__version__ < "1.7":
        pytest.skip("auto_complex needs netCDF4 >= 1.7")
    for method in _complex_measures(measure_results):
        result = measure_results[method]
        path = tmp_path / f"{method}.nc"
        result.to_netcdf(path, engine="netcdf4", auto_complex=True)
        opener = xr.open_dataset if isinstance(result, xr.Dataset) else xr.open_dataarray
        with opener(path, engine="netcdf4", auto_complex=True) as loaded:
            xr.testing.assert_identical(loaded.load(), result)
