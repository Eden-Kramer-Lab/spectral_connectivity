import inspect

import numpy as np
import pytest
from pytest import mark

from spectral_connectivity import Multitaper
from spectral_connectivity.connectivity import Connectivity
from spectral_connectivity.wrapper import (
    DEFAULT_METHODS,
    UnsupportedMeasureError,
    connectivity_to_xarray,
    multitaper_connectivity,
)


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


# Measures that do not fit the wrapper's (time, frequency, source, target) /
# (time, frequency, source) xarray layouts. Requesting one raises
# UnsupportedMeasureError with a pointer to use Connectivity directly.
_WRAPPER_UNSUPPORTED_METHODS = [
    "canonical_coherence",
    "group_delay",
    "delay",
    "global_coherence",
    "phase_slope_index",
    "conditional_spectral_granger_prediction",
    "blockwise_spectral_granger_prediction",
]


@mark.parametrize("method", _WRAPPER_UNSUPPORTED_METHODS)
def test_multitaper_connectivity_rejects_unsupported_methods(method):
    """A measure without a plain xarray layout is rejected, not silently skipped.

    Replaces an earlier loop that caught the errors and then asserted against the
    *previous* iteration's result, so a raising regression in a supported measure
    could pass unnoticed. Supported measures are exercised non-degenerately by
    ``test_multitaper_n_signals`` / ``test_multitaper_connectivities_n_signals``.
    """
    rng = np.random.default_rng(42)
    time_series = rng.random((7201, 10, 2))
    with pytest.raises(UnsupportedMeasureError):
        multitaper_connectivity(
            time_series,
            sampling_frequency=1500,
            method=method,
            time_window_duration=0.1,
        )


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


def test_result_carries_cf_coordinate_metadata():
    """time/frequency coordinates get CF-style units and long_name."""
    rng = np.random.default_rng(0)
    ds = multitaper_connectivity(
        rng.standard_normal((512, 5, 3)), sampling_frequency=500
    )
    assert ds.coords["time"].attrs["units"] == "s"
    assert ds.coords["time"].attrs["long_name"] == "Time"
    assert ds.coords["frequency"].attrs["units"] == "Hz"
    assert ds.coords["frequency"].attrs["long_name"] == "Frequency"
    assert ds.coords["source"].attrs["long_name"] == "Signal"


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

    A scalar kwarg is stored as-is; a non-scalar kwarg is stringified so it
    cannot break ``to_netcdf``. Exercised through a stub measure that fits the
    (time, frequency, source, target) layout and accepts kwargs, since none of
    the default xarray-compatible measures take keyword arguments.
    """
    import xarray as xr  # noqa: F401

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
    # Scalar kwarg stored as-is; non-scalar kwarg stringified.
    assert da.attrs["arg_threshold"] == 0.5
    assert da.attrs["arg_window"] == str([1, 2, 3])
    # The stringified non-scalar must not break NetCDF serialization.
    da.to_netcdf(tmp_path / "args.nc")


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


def test_multitaper_connectivity_skips_unsupported_measure_in_batch():
    """A batch mixing a supported and an xarray-incompatible measure drops the
    latter with a warning rather than aborting.

    ``connectivity_to_xarray`` raises ``ValueError`` (not ``NotImplementedError``)
    for ``global_coherence``; the batch loop must catch it so the supported
    measure is still returned.
    """
    rng = np.random.default_rng(0)
    result = multitaper_connectivity(
        rng.standard_normal((256, 4, 3)),
        sampling_frequency=500,
        method=["coherence_magnitude", "global_coherence"],
    )
    assert "coherence_magnitude" in result
    assert "global_coherence" not in result


def test_multitaper_connectivity_genuine_error_not_swallowed():
    """A real computation error in a batch surfaces; it is not silently dropped.

    A debiased measure requires >= 2 observations. With one trial and one taper
    it raises ValueError — a genuine data problem, distinct from a measure that
    structurally does not fit the xarray layout (UnsupportedMeasureError). It
    must propagate, not leave the user with a Dataset that silently omits the
    requested measure alongside the ones that happened to succeed.
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


def test_multitaper_connectivity_single_unsupported_measure_raises():
    """Requesting only an xarray-incompatible measure re-raises, not swallowed."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        multitaper_connectivity(
            rng.standard_normal((256, 4, 3)),
            sampling_frequency=500,
            method=["global_coherence"],
        )


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


def test_multitaper_connectivity_raises_when_no_method_is_compatible():
    rng = np.random.default_rng(0)
    with pytest.raises(UnsupportedMeasureError, match="None of the requested methods"):
        multitaper_connectivity(
            rng.standard_normal((256, 3, 2)),
            sampling_frequency=256,
            method=["global_coherence", "phase_slope_index"],
        )


def test_multitaper_connectivity_rejects_duplicate_signal_names():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="signal_names must be unique"):
        multitaper_connectivity(
            rng.standard_normal((256, 3, 2)),
            sampling_frequency=256,
            method="coherence_magnitude",
            signal_names=["a", "a"],
        )


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
