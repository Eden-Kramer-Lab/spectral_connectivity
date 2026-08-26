"""Regression tests for DPSS taper computation.

These live outside ``test_transforms.py`` so they do not depend on the optional
``nitime`` reference package: they validate against SciPy's ``dpss`` instead.
"""

import numpy as np
import pytest
from pytest import mark
from scipy.signal.windows import dpss as scipy_dpss

from spectral_connectivity.transforms import dpss_windows


@mark.parametrize(
    "n_time_samples, time_halfbandwidth_product, n_tapers",
    [(8, 2, 3), (16, 3, 5), (7, 2, 3)],
)
def test_dpss_windows_no_nan_matches_scipy(
    n_time_samples, time_halfbandwidth_product, n_tapers
):
    """A singular pivot during inverse iteration must not produce NaN tapers.

    dpss_windows(8, 2, 3) previously returned a NaN third taper because the
    unpivoted tridiagonal solve divided by a zero pivot. The concentration
    ratios must match SciPy and every taper must be finite and normalized.
    """
    tapers, eigenvalues = dpss_windows(
        n_time_samples, time_halfbandwidth_product, n_tapers, is_low_bias=False
    )
    scipy_tapers, scipy_ratios = scipy_dpss(
        n_time_samples, time_halfbandwidth_product, n_tapers, return_ratios=True
    )

    assert np.isfinite(tapers).all()
    assert np.allclose(np.sum(tapers**2, axis=1), 1.0)
    assert np.allclose(eigenvalues, scipy_ratios, atol=1e-6)
    # Tapers are unique up to sign; compare magnitudes.
    assert np.allclose(np.abs(tapers), np.abs(scipy_tapers), atol=1e-6)


def test_dpss_low_bias_keeps_valid_tapers():
    """Default low-bias mode must not silently drop a well-concentrated taper.

    For (8, 2, 3) all three tapers have concentration > 0.9, so all three must
    be retained rather than dropped as NaN.
    """
    tapers, eigenvalues = dpss_windows(8, 2, 3)
    assert tapers.shape[0] == 3
    assert np.all(eigenvalues > 0.9)


@mark.parametrize(
    "n, nw, k",
    [
        (3, 2, 3),  # NW >= n/2 -> invalid (eigenvalues could exceed 1)
        (8, 5, 3),  # NW >= n/2
        (8, 2, 20),  # n_tapers > window length
        (8, 2, 0),  # n_tapers < 1
        (1, 0.25, 1),  # window length < 2 (crashed _fix_taper_sign)
    ],
)
def test_dpss_windows_rejects_invalid_bandwidth_or_taper_count(n, nw, k):
    """Invalid NW/window/n_tapers combinations must raise, not return bad tapers."""
    with pytest.raises(ValueError):
        dpss_windows(n, nw, k, is_low_bias=False)


@mark.parametrize("bad_n_tapers", [2.9, 1.5, np.nan])
def test_dpss_windows_rejects_fractional_n_tapers(bad_n_tapers):
    """A fractional n_tapers must raise, not be silently truncated."""
    with pytest.raises(ValueError, match="n_tapers must be an integer"):
        dpss_windows(16, 3, bad_n_tapers, is_low_bias=False)


def test_multitaper_rejects_fractional_n_tapers():
    """Multitaper must reject a fractional n_tapers at construction.

    Otherwise Multitaper.n_tapers would report e.g. 2.9 while the taper array
    and FFT contain a truncated integer number of tapers.
    """
    from spectral_connectivity.transforms import Multitaper

    with pytest.raises(ValueError, match="n_tapers must be an integer"):
        Multitaper(np.random.randn(100, 1, 2), sampling_frequency=100.0, n_tapers=2.9)
