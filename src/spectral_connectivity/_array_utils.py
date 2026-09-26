"""Backend-neutral array helpers shared by the transform and connectivity code."""

import numpy as np
from numpy.typing import NDArray

from spectral_connectivity._backend import xp
from spectral_connectivity.utils import BackendArray


def _conjugate_transpose(x: NDArray[np.complexfloating]) -> NDArray[np.complexfloating]:
    """Compute conjugate transpose of the last two dimensions.

    Parameters
    ----------
    x : NDArray[complexfloating], shape (..., M, N)
        Input array.

    Returns
    -------
    x_H : NDArray[complexfloating], shape (..., N, M)
        Conjugate transpose of last two dimensions.
    """
    return x.swapaxes(-1, -2).conjugate()


def _divide_where(
    numerator: BackendArray,
    denominator: BackendArray,
    condition: BackendArray,
    fill: float,
) -> BackendArray:
    """Elementwise ``numerator / denominator`` where ``condition``, else ``fill``.

    Backend-neutral replacement for ``xp.divide(..., where=...)``: CuPy ufuncs
    do not accept the public ``where`` keyword, and the substituted unit
    denominator also avoids NumPy divide warnings.
    """
    quotient = numerator / xp.where(condition, denominator, 1)
    return xp.where(condition, quotient, xp.asarray(fill, dtype=quotient.dtype))
