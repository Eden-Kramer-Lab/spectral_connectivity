"""Helpers shared by the tests of dropped observations and artifact samples."""

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np

from spectral_connectivity import list_measures

# The measures whose finite-sample corrections count observations.
COUNTING_MEASURES = (
    "debiased_squared_phase_lag_index",
    "debiased_squared_weighted_phase_lag_index",
    "pairwise_phase_consistency",
)


def quietly(function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """``function(*args, **kwargs)`` with its ``UserWarning``s ignored: the
    tests' few observations for several signals warn of rank deficiency."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return function(*args, **kwargs)


# corrected_imaginary_phase_locking_value's diagonal (a signal with itself)
# is rounding noise amplified by the square root of its denominator, below
# about sqrt(eps) of float64, as its docstring says: two computations agree
# there only to that bound, not to rounding (9e-10 apart on CuPy, measured).
# The registry records it as the diagonal policy "rounding_noise".
NOISY_DIAGONAL_MEASURES = frozenset(
    info.name for info in list_measures() if info.diagonal_policy == "rounding_noise"
)


def without_noisy_diagonal(measure: str, *arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    """``arrays`` (``(..., n_signals, n_signals)``) with the diagonal of a
    measure in ``NOISY_DIAGONAL_MEASURES`` checked against its noise bound and
    set to 0, so the rest is compared to rounding; others unchanged."""
    if measure not in NOISY_DIAGONAL_MEASURES:
        return arrays
    bound = np.sqrt(np.finfo(np.float64).eps)
    cleaned = []
    for array in arrays:
        diagonal = np.eye(array.shape[-1], dtype=bool)
        assert np.all(np.abs(array[..., diagonal]) <= bound), measure
        cleaned.append(np.where(diagonal, 0.0, array))
    return tuple(cleaned)
