"""Helpers shared by the tests of dropped observations and artifact samples."""

import warnings
from collections.abc import Callable
from typing import Any

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
