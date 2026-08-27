"""Value tests for :func:`spectral_connectivity.transforms.detrend`.

``detrend`` is on the hot path (``Multitaper.fft`` detrends by default) and now
delegates the least-squares/mean removal to ``scipy.signal.detrend`` (CPU) or
``cupyx.scipy.signal.detrend`` (GPU) after validating ``type``/``bp`` and
normalizing the short ``'l'``/``'c'`` aliases. These tests pin the actual
numerical behavior on CPU (equivalence tests in ``test_transforms.py`` are
skipped without nitime, and previously only error messages were covered).
"""

import numpy as np
import pytest

from spectral_connectivity.transforms import detrend


def test_linear_detrend_removes_linear_trend():
    """A pure linear trend is removed to ~0."""
    t = np.linspace(0.0, 1.0, 200)
    x = 3.0 + 2.0 * t
    np.testing.assert_allclose(detrend(x, type="linear"), 0.0, atol=1e-10)


def test_constant_detrend_removes_mean():
    """Constant detrend subtracts the mean and nothing else."""
    rng = np.random.default_rng(0)
    x = 5.0 + rng.standard_normal(200)
    result = detrend(x, type="constant")
    assert abs(result.mean()) < 1e-10
    np.testing.assert_allclose(result, x - x.mean(), atol=1e-12)


@pytest.mark.parametrize("alias,full", [("l", "linear"), ("c", "constant")])
def test_short_type_aliases_match_full_names(alias, full):
    """The ``'l'``/``'c'`` aliases map to linear/constant, not silently to one.

    The alias normalization ``type = "linear" if type in ["linear", "l"] else
    "constant"`` would silently route anything unexpected to ``"constant"``; this
    confirms each short alias matches its long form exactly.
    """
    rng = np.random.default_rng(1)
    x = 2.0 + 0.5 * np.linspace(0.0, 1.0, 150) + rng.standard_normal(150)
    np.testing.assert_array_equal(detrend(x, type=alias), detrend(x, type=full))


def test_linear_detrend_with_breakpoints():
    """A per-segment linear fit removes a piecewise-linear trend that a single
    fit cannot."""
    n = 200
    t = np.arange(n, dtype=float)
    x = np.empty(n)
    x[:100] = 1.0 + 0.3 * t[:100]
    x[100:] = -5.0 + 0.1 * (t[100:] - 100)

    # A single global linear fit leaves large residuals at the kink.
    single = detrend(x, type="linear")
    assert np.abs(single).max() > 1.0

    # A breakpoint at the join detrends each segment separately -> ~0.
    segmented = detrend(x, type="linear", bp=[100])
    np.testing.assert_allclose(segmented, 0.0, atol=1e-8)


def test_detrend_along_explicit_axis():
    """Detrending along a chosen axis removes each row's trend independently."""
    rng = np.random.default_rng(2)
    trend = 2.0 * np.arange(50)
    data = trend[None, :] + rng.standard_normal((3, 50))
    result = detrend(data, axis=-1, type="linear")
    assert result.shape == data.shape
    # No residual linear trend in the averaged detrended rows.
    slope = np.polyfit(np.arange(50), result.mean(axis=0), 1)[0]
    assert abs(slope) < 1e-9


@pytest.mark.parametrize("bad_bp", [[100], [150], [50, 100]])
def test_breakpoint_at_or_beyond_length_rejected(bad_bp):
    """The documented valid breakpoint range is ``[0, N)``.

    A breakpoint at exactly ``N`` used to slip past the ``> N`` guard (the
    validation array was padded with ``N``) and silently collapse into an empty
    trailing segment; it is now rejected with a clear message, matching the
    documented range.
    """
    x = np.random.default_rng(3).standard_normal(100)
    with pytest.raises(ValueError, match="Breakpoint"):
        detrend(x, type="linear", bp=bad_bp)


def test_valid_interior_breakpoint_accepted():
    """A breakpoint strictly inside ``[0, N)`` is accepted."""
    x = np.random.default_rng(4).standard_normal(100)
    result = detrend(x, type="linear", bp=[50])
    assert result.shape == x.shape


def test_breakpoint_tuple_input_supported():
    """``bp`` accepts a tuple (documented array-like) for valid and invalid values.

    A tuple has no ``.tolist()``; the validation must normalize it (via
    ``asarray``) rather than raising ``AttributeError`` on an out-of-range value.
    """
    x = np.random.default_rng(0).standard_normal(100)
    # Valid interior tuple breakpoint works.
    assert detrend(x, type="linear", bp=(50,)).shape == x.shape
    # A tuple breakpoint at N is rejected with the range message (no AttributeError).
    with pytest.raises(ValueError, match=r"\[0, 100\)"):
        detrend(x, type="linear", bp=(100,))


@pytest.mark.parametrize("bad_bp", [[-1], [-5, 10], (-3,)])
def test_negative_breakpoint_rejected(bad_bp):
    """Negative breakpoints are outside ``[0, N)`` and rejected up front.

    Previously only the upper bound was checked, so a negative index reached the
    backend as a cryptic error despite the documented range.
    """
    x = np.random.default_rng(1).standard_normal(100)
    with pytest.raises(ValueError, match=r"\[0, 100\)"):
        detrend(x, type="linear", bp=bad_bp)
