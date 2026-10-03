"""Backend-neutral helpers so the suite runs under NumPy and under CuPy.

The backend is fixed when ``spectral_connectivity`` is first imported (see
``_backend``); run the suite on the GPU with
``SPECTRAL_CONNECTIVITY_ENABLE_GPU=true uv run --extra gpu pytest``.

Public API results are already host (NumPy) arrays. Tests that call private
kernels directly must pass them device arrays (``to_device``) and bring the
results back (``to_host``) before comparing with ``np.testing``. Tests of
NumPy-only behavior take ``@pytest.mark.cpu_only(reason=...)``, which
``conftest.py`` skips on the GPU.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from spectral_connectivity._backend import ON_GPU, xp
from spectral_connectivity.utils import to_numpy

__all__ = ["ON_GPU", "to_device", "to_host", "xp"]


def to_device(array: Any) -> Any:
    """Return ``array`` in the active backend's namespace (a no-op on NumPy)."""
    return xp.asarray(array)


def to_host(array: Any) -> NDArray[Any]:
    """Return ``array`` as a NumPy array, copying it off the device if needed."""
    return np.asarray(to_numpy(array))
