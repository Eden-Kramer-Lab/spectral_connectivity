import inspect

import numpy as np
import pytest
from pytest import mark

from spectral_connectivity.connectivity import Connectivity
from spectral_connectivity.wrapper import multitaper_connectivity


@mark.parametrize("time_window_duration", [0.1, 0.2, 2.4, 0.16])
@mark.parametrize("dtype", [np.complex64, np.complex128])
def test_multitaper_coherence_magnitude(time_window_duration, dtype):
    np.random.default_rng(42)
    sampling_frequency = 1500
    start_time, end_time = 0, 4.8
    n_trials, n_signals = 10, 2
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

    m = multitaper_connectivity(
        time_series,
        method="coherence_magnitude",
        sampling_frequency=sampling_frequency,
        time_window_duration=time_window_duration,
    )

    assert np.allclose(m.time.values, expected_time)
    assert not (m.values == 0).all()
    assert not (np.isnan(m.values)).all()


def test_multitaper_connectivity():
    np.random.default_rng(42)
    time_window_duration = 0.1
    sampling_frequency = 1500
    start_time, end_time = 0, 4.8
    n_trials, n_signals = 10, 2
    n_time_samples = int((end_time - start_time) * sampling_frequency) + 1
    time_series = np.random.random(size=(n_time_samples, n_trials, n_signals))

    for method in [
        "coherence_magnitude",
        "coherency",
        "coherence_magnitude",
        "coherence_phase",
        "imaginary_coherence",
        "phase_locking_value",
        "phase_lag_index",
        "weighted_phase_lag_index",
        "debiased_squared_phase_lag_index",
        "debiased_squared_weighted_phase_lag_index",
        "pairwise_phase_consistency",
        "phase_lag_index",
        "pairwise_spectral_granger_prediction",
        # Below measures are not implemented, will throw NotImplementedError
        "directed_transfer_function",
        "directed_coherence",
        "partial_directed_coherence",
        "generalized_partial_directed_coherence",
        "direct_directed_transfer_function",
        "canonical_coherence",
        "group_delay",
        "power",
    ]:
        try:
            m = multitaper_connectivity(
                time_series,
                method=method,
                sampling_frequency=sampling_frequency,
                time_window_duration=time_window_duration,
            )
        except (NotImplementedError, ValueError):
            pass

        assert not (m.values == 0).all()
        assert not (np.isnan(m.values)).all()


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

    bad_methods = [
        "delay",
        "n_observations",
        "frequencies",
        "all_frequencies",
        "fourier_coefficients",
        "expectation_type",
        "global_coherence",
        "from_multitaper",
        "phase_slope_index",
        "subset_pairwise_spectral_granger_prediction",
    ]
    methods = [
        x for x in dir(Connectivity) if not x.startswith("_") and x not in bad_methods
    ]

    for method in methods:
        try:
            m = multitaper_connectivity(
                time_series,
                method=method,
                sampling_frequency=sampling_frequency,
                time_window_duration=time_window_duration,
            )
            assert np.allclose(m.time.values, expected_time)
            assert not (m.values == 0).all()
            assert not (np.isnan(m.values)).all()

        except (NotImplementedError, ValueError):
            pass


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


def test_method_discovery_with_inspect():
    """Test that inspect.getmembers() correctly identifies Connectivity methods.

    This test verifies that the refactored method discovery in wrapper.py
    using inspect.getmembers() finds all expected connectivity methods.
    """
    # Methods that should be excluded (not connectivity measures or not xarray-compatible)
    excluded_methods = {
        # Properties and utility methods
        "delay",
        "n_observations",
        "frequencies",
        "all_frequencies",
        "fourier_coefficients",
        "expectation_type",
        "global_coherence",
        "from_multitaper",
        "phase_slope_index",
        "subset_pairwise_spectral_granger_prediction",
        # Methods not supported by xarray interface
        "group_delay",
        "canonical_coherence",
        "directed_transfer_function",
        "directed_coherence",
        "partial_directed_coherence",
        "generalized_partial_directed_coherence",
        "direct_directed_transfer_function",
        "blockwise_spectral_granger_prediction",
    }

    # Get methods using inspect (same as wrapper.py implementation)
    methods_via_inspect = [
        name
        for name, member in inspect.getmembers(
            Connectivity, predicate=inspect.isfunction
        )
        if not name.startswith("_") and name not in excluded_methods
    ]

    # Get methods using dir() (old implementation)
    methods_via_dir = [
        x
        for x in dir(Connectivity)
        if not x.startswith("_") and x not in excluded_methods
    ]

    # Both methods should find the same set of methods
    assert set(methods_via_inspect) == set(methods_via_dir)

    # Verify we find expected connectivity methods
    expected_methods = {
        "coherence_magnitude",
        "coherency",
        "imaginary_coherence",
        "phase_locking_value",
        "power",
    }
    assert expected_methods.issubset(set(methods_via_inspect))

    # Verify excluded methods are not included
    found_methods_set = set(methods_via_inspect)
    assert not excluded_methods.intersection(found_methods_set)


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

    `fft_workers` only sets SciPy's CPU FFT thread count; the transform is
    identical regardless of the worker count, and the wrapper forwards the
    argument to `Multitaper` via **kwargs.
    """
    from spectral_connectivity.transforms import Multitaper

    rng = np.random.default_rng(0)
    time_series = rng.standard_normal((1024, 6, 3))

    reference = Multitaper(time_series, sampling_frequency=500).fft()
    for workers in (1, 2, -1):
        result = Multitaper(
            time_series, sampling_frequency=500, fft_workers=workers
        ).fft()
        np.testing.assert_array_equal(result, reference)

    # The wrapper forwards fft_workers via **kwargs; results are unchanged.
    baseline = multitaper_connectivity(
        time_series, sampling_frequency=500, method="coherence_magnitude"
    )
    parallel = multitaper_connectivity(
        time_series,
        sampling_frequency=500,
        method="coherence_magnitude",
        fft_workers=-1,
    )
    np.testing.assert_array_equal(baseline.values, parallel.values)


def test_to_host_array_handles_device_arrays():
    """Coordinate validation must not implicitly convert GPU arrays.

    Under GPU mode ``Multitaper`` coordinates are CuPy arrays, which raise on
    implicit ``np.asarray`` conversion. ``_to_host_array`` must route through
    ``.get()`` for such arrays while leaving NumPy arrays untouched, so the
    injected-``Connectivity`` validation works on both backends.
    """
    from spectral_connectivity.wrapper import _to_host_array

    host = np.arange(5.0)
    np.testing.assert_array_equal(_to_host_array(host), host)

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
    np.testing.assert_array_equal(_to_host_array(device), np.arange(5.0))


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

    method discovery must not include complex-valued coherency, which NetCDF
    cannot store.
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
