"""End-to-end smoke coverage for a real CuPy/CUDA backend.

``test_real_gpu_backend_smoke`` is the GPU release gate
(``SPECTRAL_CONNECTIVITY_ENABLE_GPU=true pytest -m gpu``; see
``docs/contributing.md``). When that variable requests the GPU, a missing CuPy
or CUDA device fails the test instead of skipping it, so the gate cannot pass
without having run. The backend is chosen at import time, so the GPU and the CPU
reference are each computed in a fresh interpreter and compared numerically.
"""

import os
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import numpy as np
import pytest

from spectral_connectivity.utils import GPU_ENV_VAR, is_gpu_enabled

# Computes a fixed set of measures under whichever backend the interpreter
# imports, checks that the backend is the expected one, and saves host (NumPy)
# arrays to the .npz path given as argv[1]. argv[2] is the expected backend.
_MEASURES_SCRIPT = textwrap.dedent(
    """
    import sys
    import warnings

    import numpy as np

    from spectral_connectivity import Connectivity, Multitaper, get_compute_backend
    from spectral_connectivity.transforms import xp
    from spectral_connectivity.wrapper import multitaper_connectivity

    output_path, expected_backend = sys.argv[1], sys.argv[2]
    assert get_compute_backend()["backend"] == expected_backend, get_compute_backend()

    rng = np.random.default_rng(0)
    data = rng.standard_normal((256, 5, 3))
    m = Multitaper(data, sampling_frequency=256, time_halfbandwidth_product=2)
    coefficients = m.fft()
    assert isinstance(coefficients, xp.ndarray)

    connectivity = Connectivity.from_multitaper(m)
    assert isinstance(connectivity._fourier_coefficients, xp.ndarray)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        outputs = {
            "power": connectivity.power(),
            "coherency": connectivity.coherency(),
            "coherence_magnitude": connectivity.coherence_magnitude(),
            "phase_lag_index": connectivity.phase_lag_index(),
            "global_coherence": connectivity.global_coherence()[0],
            "pairwise_spectral_granger_prediction": (
                connectivity.pairwise_spectral_granger_prediction()
            ),
        }
    assert all(isinstance(output, np.ndarray) for output in outputs.values())

    wrapped = multitaper_connectivity(
        data,
        sampling_frequency=256,
        method=["coherence_magnitude", "phase_lag_index"],
        time_halfbandwidth_product=2,
    )
    assert set(wrapped.data_vars) == {"coherence_magnitude", "phase_lag_index"}
    outputs["wrapped_coherence_magnitude"] = wrapped["coherence_magnitude"].values
    np.savez(output_path, **outputs)
    """
)

# Float64 FFTs and closed-form spectral ratios differ between cuFFT/cuBLAS and
# NumPy/SciPy only by rounding. Granger goes through the iterative Wilson
# factorization, whose convergence-tolerance-level differences are looser.
_TOLERANCES = {
    "power": {"rtol": 1e-9, "atol": 1e-12},
    "coherency": {"rtol": 1e-9, "atol": 1e-12},
    "coherence_magnitude": {"rtol": 1e-9, "atol": 1e-12},
    "phase_lag_index": {"rtol": 1e-9, "atol": 1e-12},
    "global_coherence": {"rtol": 1e-9, "atol": 1e-12},
    "pairwise_spectral_granger_prediction": {"rtol": 1e-5, "atol": 1e-8},
    "wrapped_coherence_magnitude": {"rtol": 1e-9, "atol": 1e-12},
}


def _skip_or_fail(reason: str) -> None:
    """Skip when the GPU was not requested; fail when it was.

    Under ``SPECTRAL_CONNECTIVITY_ENABLE_GPU=true`` a skip would let the GPU
    release gate exit 0 without exercising the GPU.
    """
    if is_gpu_enabled():
        pytest.fail(
            f"{reason}, but {GPU_ENV_VAR} requests the GPU: the GPU smoke test could not run."
        )
    pytest.skip(reason)


def _require_cuda_device() -> None:
    """Return normally only if CuPy imports and sees at least one CUDA device."""
    try:
        import cupy
    except ImportError as error:
        _skip_or_fail(f"CuPy is not importable ({error})")
    try:
        n_devices = cupy.cuda.runtime.getDeviceCount()
    except cupy.cuda.runtime.CUDARuntimeError as error:
        _skip_or_fail(f"CUDA runtime is unavailable: {error}")
    if n_devices < 1:
        _skip_or_fail("No CUDA device is available")


def _run_measures(output_path: Path, *, gpu: bool) -> dict[str, np.ndarray]:
    """Run ``_MEASURES_SCRIPT`` in a fresh interpreter on the requested backend."""
    environment = {**os.environ, GPU_ENV_VAR: "true" if gpu else "false"}
    result = subprocess.run(
        [sys.executable, "-c", _MEASURES_SCRIPT, str(output_path), "gpu" if gpu else "cpu"],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        # Surface the child's traceback; a bare CalledProcessError hides it.
        pytest.fail(
            f"{'GPU' if gpu else 'CPU'} measures subprocess failed "
            f"(exit {result.returncode}):\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )
    with np.load(output_path) as saved:
        return {name: saved[name] for name in saved.files}


@pytest.mark.gpu
def test_real_gpu_backend_smoke(tmp_path):
    """GPU results match a CPU reference for transforms, linalg, and the wrapper."""
    _require_cuda_device()

    gpu_results = _run_measures(tmp_path / "gpu.npz", gpu=True)
    cpu_results = _run_measures(tmp_path / "cpu.npz", gpu=False)

    assert set(gpu_results) == set(cpu_results) == set(_TOLERANCES)
    for name, tolerance in _TOLERANCES.items():
        np.testing.assert_allclose(
            gpu_results[name], cpu_results[name], equal_nan=True, err_msg=name, **tolerance
        )


def test_measures_script_runs_on_cpu(tmp_path):
    """The shared measures script runs and saves every compared measure on CPU.

    The GPU gate cannot run in ordinary CI, so this keeps its script and result
    plumbing from rotting unnoticed.
    """
    results = _run_measures(tmp_path / "cpu.npz", gpu=False)

    assert set(results) == set(_TOLERANCES)
    n_frequencies = 256 // 2 + 1
    assert results["coherence_magnitude"].shape == (1, n_frequencies, 3, 3)
    assert np.all(np.isfinite(results["power"]))
    np.testing.assert_array_equal(
        results["wrapped_coherence_magnitude"], results["coherence_magnitude"]
    )


def _require_cuda_device_outcome() -> tuple[str, str]:
    """Run ``_require_cuda_device`` and report ``("fail" | "skip" | "ran", message)``.

    Catching both outcome exceptions keeps a wrong skip from being reported as a
    skipped (i.e. passing) test.
    """
    try:
        _require_cuda_device()
    except pytest.fail.Exception as outcome:
        return "fail", str(outcome)
    except pytest.skip.Exception as outcome:
        return "skip", str(outcome)
    return "ran", ""


def _block_cupy(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", None)


def _fake_cupy(monkeypatch, *, n_devices=0, error=None):
    """Install a fake ``cupy`` whose CUDA runtime reports ``n_devices`` or raises."""

    class CUDARuntimeError(RuntimeError):
        pass

    def get_device_count():
        if error is not None:
            raise CUDARuntimeError(error)
        return n_devices

    runtime = types.SimpleNamespace(
        getDeviceCount=get_device_count, CUDARuntimeError=CUDARuntimeError
    )
    fake = types.ModuleType("cupy")
    fake.cuda = types.SimpleNamespace(runtime=runtime)
    monkeypatch.setitem(sys.modules, "cupy", fake)


_UNAVAILABLE_GPU_CASES = {
    "cupy_missing": (_block_cupy, "CuPy is not importable"),
    "no_device": (lambda mp: _fake_cupy(mp, n_devices=0), "No CUDA device"),
    "runtime_error": (lambda mp: _fake_cupy(mp, error="driver too old"), "driver too old"),
}


@pytest.mark.parametrize(
    ("requested", "expected_outcome"), [("true", "fail"), (None, "skip"), ("false", "skip")]
)
@pytest.mark.parametrize("case", list(_UNAVAILABLE_GPU_CASES))
def test_unavailable_gpu_fails_only_when_requested(
    monkeypatch, case, requested, expected_outcome
):
    """A missing GPU fails when the env var requests it and skips otherwise."""
    if requested is None:
        monkeypatch.delenv(GPU_ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(GPU_ENV_VAR, requested)
    make_unavailable, reason = _UNAVAILABLE_GPU_CASES[case]
    make_unavailable(monkeypatch)

    outcome, message = _require_cuda_device_outcome()

    assert outcome == expected_outcome
    assert reason in message


def test_available_device_runs(monkeypatch):
    """With a CUDA device the gate proceeds instead of skipping or failing."""
    monkeypatch.setenv(GPU_ENV_VAR, "true")
    _fake_cupy(monkeypatch, n_devices=1)
    assert _require_cuda_device_outcome() == ("ran", "")
