"""Array backend shared by every module: NumPy/SciPy on CPU or CuPy on GPU.

The backend is chosen once, when this module is first imported, from the
``SPECTRAL_CONNECTIVITY_ENABLE_GPU`` environment variable (see
:func:`spectral_connectivity.utils.is_gpu_enabled`). Modules take ``xp`` and the
FFT, signal, and sparse linear-algebra routines from here, and test ``ON_GPU``
rather than the environment variable (which may change after import), so they
cannot disagree about the backend.
"""

from logging import getLogger
from typing import TYPE_CHECKING

from spectral_connectivity.utils import (
    cupy_device_name,
    gpu_request_error_message,
    is_gpu_enabled,
)

logger = getLogger(__name__)

# Type-check against the NumPy API, which CuPy mirrors: mypy sees only the CPU
# branch (CuPy is untyped, so importing it would make ``xp`` ``Any``).
if not TYPE_CHECKING and is_gpu_enabled():
    try:
        import cupy as xp
        from cupyx.scipy.fft import fft, fftfreq, ifft, irfft, next_fast_len, rfft
        from cupyx.scipy.sparse.linalg import svds
    except ImportError as exc:
        raise RuntimeError(gpu_request_error_message()) from exc
    try:
        # cupyx.scipy.signal.detrend was added in CuPy 13; a CuPy-12 install
        # imports cupy fine but fails here, which must not be reported as
        # "CuPy is not installed".
        from cupyx.scipy.signal import detrend
    except ImportError as exc:
        msg = (
            f"GPU support requires cupy-cuda12x>=13.0, but CuPy {xp.__version__} "
            f"is installed: cupyx.scipy.signal.detrend (used by transforms.detrend) "
            f"was added in CuPy 13. Upgrade with 'pip install -U cupy-cuda12x'."
        )
        raise RuntimeError(msg) from exc

    ON_GPU = True
    try:
        logger.info("Using GPU for spectral_connectivity on %s", cupy_device_name(xp))
    except Exception:
        logger.info("Using GPU for spectral_connectivity...")
else:
    logger.info("Using CPU for spectral_connectivity...")
    ON_GPU = False
    import numpy as xp  # noqa: ICN001 -- the backend-neutral array namespace
    from scipy.fft import fft, fftfreq, ifft, irfft, next_fast_len, rfft
    from scipy.signal import detrend
    from scipy.sparse.linalg import svds

__all__ = [
    "ON_GPU",
    "detrend",
    "fft",
    "fftfreq",
    "ifft",
    "irfft",
    "next_fast_len",
    "rfft",
    "svds",
    "xp",
]
