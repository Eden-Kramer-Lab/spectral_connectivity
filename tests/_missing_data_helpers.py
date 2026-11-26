"""Oracles shared by the tests of missing-data tapers."""

import numpy as np
from scipy.linalg import eigh


def dense_slepians(
    valid: np.ndarray, time_halfbandwidth_product: float
) -> tuple[np.ndarray, np.ndarray]:
    """Every eigenpair, in decreasing order, of ``sin(2 pi W (t - s)) / (pi (t -
    s))`` (``2 W`` at ``t = s``) on the valid sample times ``t, s``: the
    kernel written out entry by entry and solved densely.

    Returns eigenvalues ``(n_valid,)`` and eigenvectors ``(n_valid, n_valid)``.
    """
    half_bandwidth = time_halfbandwidth_product / valid.size
    times = np.flatnonzero(valid).astype(float)
    lag = times[:, None] - times[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        kernel = np.sin(2 * np.pi * half_bandwidth * lag) / (np.pi * lag)
    kernel[lag == 0] = 2 * half_bandwidth
    eigenvalues, eigenvectors = eigh(kernel)
    return eigenvalues[::-1], eigenvectors[:, ::-1]


def valid_outside_gaps(gaps: list[tuple[int, int]], n_time: int) -> np.ndarray:
    """``(n_time,)``, ``True`` but in the ``(start, length)`` gaps."""
    valid = np.ones(n_time, dtype=bool)
    for start, length in gaps:
        valid[start : start + length] = False
    return valid
