"""Tests for the spectral Granger kernels in ``spectral_connectivity._granger``."""

import warnings

import numpy as np
import pytest

from spectral_connectivity import (
    Connectivity,
    DirectedOrientationWarning,
    Multitaper,
    _granger,
)
from spectral_connectivity._granger import (
    _remove_instantaneous_causality,
    _sanitized_nonnegative_granger,
)
from tests._backend_helpers import to_device, to_host, xp


def test__remove_instantaneous_causality():
    noise_covariance = np.zeros((2, 2, 2))
    x1 = np.array([[1, 2], [2, 4]], dtype=float)
    x2 = np.array([[8, 4], [4, 16]], dtype=float)
    noise_covariance[0, ...] = x1
    noise_covariance[1, ...] = x2

    # x -> y: var(x) - (cov(x,y) ** 2 / var(y))
    # y -> x: var(y) - (cov(x,y) ** 2 / var(x))
    expected_rotated_noise_covariance = np.zeros((2, 2, 2))

    expected_rotated_noise_covariance[0, 0, 1] = x1[1, 1] - (x1[0, 1] ** 2 / x1[0, 0])
    expected_rotated_noise_covariance[0, 1, 0] = x1[0, 0] - (x1[1, 0] ** 2 / x1[1, 1])

    expected_rotated_noise_covariance[1, 0, 1] = x2[1, 1] - (x2[0, 1] ** 2 / x2[0, 0])
    expected_rotated_noise_covariance[1, 1, 0] = x2[0, 0] - (x2[1, 0] ** 2 / x2[1, 1])

    assert np.allclose(
        to_host(_remove_instantaneous_causality(to_device(noise_covariance))),
        expected_rotated_noise_covariance,
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_sanitized_nonnegative_granger_enforces_invariant(dtype):
    """The shared sanitizer clips roundoff, NaNs real negatives, keeps the rest."""
    eps = np.finfo(dtype).eps
    tiny_negative = -10 * eps  # inside the 100 * eps roundoff band
    material_negative = -1e-3  # far outside the roundoff band
    value = np.array([0.0, 0.5, tiny_negative, material_negative, np.nan], dtype=dtype)

    sanitized = to_host(_sanitized_nonnegative_granger(to_device(value)))

    # Exact zero (no causality) is preserved, not discarded as NaN.
    assert sanitized[0] == 0.0
    # A genuine positive influence passes through untouched.
    assert sanitized[1] == dtype(0.5)
    # Roundoff around a true zero is clipped up to exactly zero.
    assert sanitized[2] == 0.0
    # A materially-negative (invalid) value becomes NaN.
    assert np.isnan(sanitized[3])
    # NaN propagates unchanged.
    assert np.isnan(sanitized[4])


@pytest.mark.parametrize(
    ("measure", "kwargs", "expected_pairs"),
    [
        ("pairwise_spectral_granger_prediction", {}, "0 -> 1, 0 -> 2, 1 -> 0"),
        (
            "subset_pairwise_spectral_granger_prediction",
            {"pairs": [(0, 2)]},
            "2 source -> target signal pair(s): 0 -> 2, 2 -> 0 (0-based signal indices",
        ),
        ("conditional_spectral_granger_prediction", {}, "0 -> 1, 0 -> 2, 1 -> 0"),
        (
            "blockwise_spectral_granger_prediction",
            {"group_labels": ["a", "a", "b"]},
            "'a' -> 'b', 'b' -> 'a'",
        ),
        # Python str elements, which have no .item().
        (
            "blockwise_spectral_granger_prediction",
            {"group_labels": np.array(["a", "a", "b"], dtype=object)},
            "'a' -> 'b', 'b' -> 'a'",
        ),
    ],
)
def test_granger_measures_name_the_pairs_whose_factor_is_not_minimum_phase(
    measure, kwargs, expected_pairs, monkeypatch
):
    """Each Granger measure warns once per call, naming the source -> target
    pairs whose (pairwise, reduced, or block) factor the check flagged,
    however many factorizations the measure makes."""
    signals = np.random.default_rng(0).standard_normal((512, 10, 3))
    connectivity = Connectivity.from_transform(
        Multitaper(signals, sampling_frequency=256, time_halfbandwidth_product=2)
    )
    # The forward conditional measure uses the cached full model, factored
    # (and checked) once per instance; factor it before flagging.
    connectivity._minimum_phase_factor  # noqa: B018
    original = _granger._minimum_phase_factor_and_flags

    def flag_every_sub_spectrum(*args, **kwargs):
        factorization = original(*args, **kwargs)
        return factorization._replace(
            not_minimum_phase=xp.ones_like(factorization.not_minimum_phase)
        )

    monkeypatch.setattr(_granger, "_minimum_phase_factor_and_flags", flag_every_sub_spectrum)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.simplefilter("ignore", DirectedOrientationWarning)
        getattr(connectivity, measure)(**kwargs)
    messages = [str(w.message) for w in caught if "not minimum phase" in str(w.message)]
    assert len(messages) == 1, messages
    assert messages[0].startswith(f"{measure}: ")
    assert expected_pairs in messages[0]


_RESCALED_GRANGER_MEASURES = {
    "blockwise_three_signal_target": lambda connectivity: (
        connectivity.blockwise_spectral_granger_prediction([0, 0, 0, 1])[0]
    ),
    "subset_pairwise": lambda connectivity: (
        connectivity.subset_pairwise_spectral_granger_prediction([(0, 1), (1, 2), (2, 3)])
    ),
    "conditional": lambda connectivity: connectivity.conditional_spectral_granger_prediction(),
}


@pytest.mark.parametrize("measure", list(_RESCALED_GRANGER_MEASURES))
def test_granger_of_correlated_signals_is_unchanged_by_rescaling_them(measure):
    """Four white signals correlated at 0.6, rescaled by 1e-4 and 1e4: the
    blockwise measure with a three-signal target block, subset pairwise and
    conditional Granger are unchanged, NaN nowhere new and without a
    warning.

    Regression: the blockwise measure tested whether the target spectra were
    positive definite from eigenvalues in the signals' units, where rounding
    erased the smallest one, and returned NaN with a warning at 8 of its 34
    values."""
    rng = np.random.default_rng(0)
    mixing = np.linalg.cholesky(np.full((4, 4), 0.6) + 0.4 * np.eye(4))
    coefficients = np.fft.fft(rng.standard_normal((1, 400, 1, 32, 4)) @ mixing.T, axis=-2)
    scales = np.array([1.0, 1e-4, 1e4, 1.0])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warnings.filterwarnings("ignore", category=DirectedOrientationWarning)
        compute = _RESCALED_GRANGER_MEASURES[measure]
        as_drawn = to_host(compute(Connectivity(to_device(coefficients))))
        rescaled = to_host(compute(Connectivity(to_device(coefficients * scales))))

    assert np.isfinite(as_drawn).any()
    # The factorization stops at a relative change of minimum_phase_tolerance
    # (1e-8), so the two runs may stop at iterates that differ by about that
    # much; CI's Python 3.10 macOS and Windows wheels differed by 1.05e-9 at a
    # value of 2e-5. The defect guarded against moved values by 0.47 to 1.1
    # and added NaN, which equal_nan catches at any atol.
    np.testing.assert_allclose(rescaled, as_drawn, rtol=0, atol=1e-7, equal_nan=True)
