"""Utility functions for spectral_connectivity package."""

import os
import sys
from typing import Any


def get_compute_backend() -> dict[str, Any]:
    """Get information about the current compute backend (CPU or GPU).

    Returns a dictionary with detailed information about whether GPU
    acceleration is enabled and available, which backend is being used,
    and device-specific information.

    Returns
    -------
    dict[str, Any]
        Dictionary with the following keys:
        - backend : str
            Either "cpu" or "gpu" indicating which backend is currently active
        - gpu_enabled : bool
            True if GPU was requested via SPECTRAL_CONNECTIVITY_ENABLE_GPU='true'
        - gpu_available : bool
            True if CuPy is installed and can be imported
        - device_name : str
            Name of the compute device (e.g., "CPU", "NVIDIA Tesla V100-SXM2-16GB")
        - message : str
            Human-readable message explaining the current configuration

        Example return value when GPU is active:
            {
                'backend': 'gpu',
                'gpu_enabled': True,
                'gpu_available': True,
                'device_name': 'NVIDIA Tesla V100-SXM2-16GB',
                'message': 'Using GPU backend with CuPy on NVIDIA Tesla V100-SXM2-16GB.'
            }

    Examples
    --------
    Check if GPU acceleration is available and enabled:

    >>> import spectral_connectivity
    >>> backend = spectral_connectivity.get_compute_backend()
    >>> print(backend['message'])
    Using CPU backend with NumPy. To enable GPU, install CuPy and set...

    In a script, check GPU status before heavy computation:

    >>> backend = spectral_connectivity.get_compute_backend()
    >>> if backend['backend'] == 'gpu':
    ...     print(f"GPU acceleration enabled on {backend['device_name']}")
    ... else:
    ...     print("Running on CPU - consider enabling GPU for large datasets")

    In a Jupyter notebook, display compute configuration:

    >>> backend = spectral_connectivity.get_compute_backend()
    >>> for key, value in backend.items():
    ...     print(f"{key}: {value}")

    Notes
    -----
    This function does not change the backend configuration; it only reports
    the current state. The backend is determined when the package modules are
    first imported, based on the SPECTRAL_CONNECTIVITY_ENABLE_GPU environment
    variable.

    To enable GPU acceleration, set the environment variable before importing:

    - In shell: ``export SPECTRAL_CONNECTIVITY_ENABLE_GPU=true``
    - In Python: ``os.environ['SPECTRAL_CONNECTIVITY_ENABLE_GPU'] = 'true'``
    - In notebook: ``%env SPECTRAL_CONNECTIVITY_ENABLE_GPU=true``

    See Also
    --------
    Multitaper : Uses the configured compute backend for FFT operations
    Connectivity : Uses the configured compute backend for connectivity calculations
    """
    # Check if GPU was requested via environment variable
    gpu_enabled = os.environ.get("SPECTRAL_CONNECTIVITY_ENABLE_GPU") == "true"

    # Check if CuPy is available (either already imported or can be imported)
    cupy_available = False
    device_name = "CPU"

    # First check if cupy is already imported
    if "cupy" in sys.modules:
        cupy_available = True
        try:
            import cupy as cp

            # Try to get device info - prefer actual GPU name over compute capability
            try:
                device = cp.cuda.Device()
                # Try to get the actual GPU model name first
                try:
                    device_name = cp.cuda.runtime.getDeviceProperties(device.id)[
                        "name"
                    ].decode()
                    # Clean up the name if it has null bytes
                    device_name = device_name.strip("\x00")
                except Exception:
                    # Fallback to compute capability if name not available
                    compute_cap = device.compute_capability
                    device_name = (
                        f"GPU (Compute Capability {compute_cap[0]}.{compute_cap[1]})"
                    )
            except Exception:
                device_name = "GPU"
        except Exception:
            cupy_available = False

    # If cupy not already imported, check if it can be imported
    # (but don't actually import it to avoid side effects)
    if not cupy_available:
        try:
            # Use __import__ to check availability without fully importing
            import importlib.util

            spec = importlib.util.find_spec("cupy")
            cupy_available = spec is not None
        except (ImportError, ValueError, AttributeError):
            cupy_available = False

    # Determine actual backend being used
    # Check what was actually imported in transforms/connectivity modules
    backend = "cpu"
    if "spectral_connectivity.transforms" in sys.modules:
        transforms_module = sys.modules["spectral_connectivity.transforms"]
        # Check if xp is cupy
        if hasattr(transforms_module, "xp"):
            if "cupy" in str(type(transforms_module.xp)):
                backend = "gpu"

    # Generate helpful message
    if backend == "gpu":
        message = f"Using GPU backend with CuPy on {device_name}."
    elif gpu_enabled and not cupy_available:
        message = (
            "GPU acceleration was requested (SPECTRAL_CONNECTIVITY_ENABLE_GPU='true') "
            "but CuPy is not installed. Install CuPy with: "
            "'pip install cupy' or 'conda install cupy'. "
            "Currently using CPU backend with NumPy."
        )
    elif not gpu_enabled and cupy_available:
        message = (
            "Using CPU backend with NumPy. CuPy is installed and GPU acceleration "
            "is available. To enable GPU, set environment variable: "
            "SPECTRAL_CONNECTIVITY_ENABLE_GPU='true' before importing spectral_connectivity. "
            "See documentation for details."
        )
    else:  # CPU mode, cupy not available
        message = (
            "Using CPU backend with NumPy. To enable GPU acceleration:\n"
            "  1. Install CuPy: 'conda install -c conda-forge cupy' or 'pip install cupy'\n"
            "  2. Set environment variable SPECTRAL_CONNECTIVITY_ENABLE_GPU='true' before importing\n"
            "See documentation for detailed setup instructions."
        )

    return {
        "backend": backend,
        "gpu_enabled": gpu_enabled,
        "gpu_available": cupy_available,
        "device_name": device_name,
        "message": message,
    }
