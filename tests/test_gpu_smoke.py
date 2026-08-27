"""End-to-end smoke coverage for a real CuPy/CUDA backend."""

import os
import subprocess
import sys

import pytest


@pytest.mark.gpu
def test_real_gpu_backend_smoke():
    """Exercise transforms, linalg, host boundaries, and the xarray wrapper."""
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("No CUDA device is available")
    except cupy.cuda.runtime.CUDARuntimeError as error:
        pytest.skip(f"CUDA runtime is unavailable: {error}")

    code = """
import warnings
import numpy as np
import cupy as cp

from spectral_connectivity import Connectivity, Multitaper, get_compute_backend
from spectral_connectivity.wrapper import multitaper_connectivity

rng = np.random.default_rng(0)
data = rng.standard_normal((256, 5, 3))
m = Multitaper(data, sampling_frequency=256, time_halfbandwidth_product=2)
coefficients = m.fft()
assert isinstance(coefficients, cp.ndarray)

connectivity = Connectivity.from_multitaper(m)
assert isinstance(connectivity._fourier_coefficients, cp.ndarray)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    outputs = (
        connectivity.coherence_magnitude(),
        connectivity.phase_lag_index(),
        connectivity.global_coherence()[0],
        connectivity.pairwise_spectral_granger_prediction(),
    )
assert all(isinstance(output, np.ndarray) for output in outputs)

wrapped = multitaper_connectivity(
    data,
    sampling_frequency=256,
    method=["coherence_magnitude", "phase_lag_index"],
    time_halfbandwidth_product=2,
)
assert set(wrapped.data_vars) == {"coherence_magnitude", "phase_lag_index"}
assert get_compute_backend()["backend"] == "gpu"
"""
    environment = os.environ.copy()
    environment["SPECTRAL_CONNECTIVITY_ENABLE_GPU"] = "true"
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        env=environment,
        text=True,
        capture_output=True,
    )
