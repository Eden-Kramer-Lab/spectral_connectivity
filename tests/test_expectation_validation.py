"""expectation_type selects which observation axes are averaged.

The reference averages are computed directly with NumPy from the documented
meaning of each name: every word (``time``, ``trials``, ``tapers``) names one
leading axis of the ``(n_time, n_trials, n_tapers, n_fft, n_signals)`` Fourier
coefficients that is averaged away.
"""

import math

import numpy as np
import pytest

from spectral_connectivity import Connectivity

AXIS_SIZES = {"time": 4, "trials": 2, "tapers": 3}
AXIS_INDEX = {"time": 0, "trials": 1, "tapers": 2}
N_FFT, N_SIGNALS = 5, 2
# fftfreq(5) = [0, 0.2, 0.4, -0.4, -0.2]: three non-negative frequencies.
N_NONNEGATIVE_FREQUENCIES = 3

EXPECTATION_TYPES = [
    "time",
    "trials",
    "tapers",
    "time_trials",
    "time_tapers",
    "trials_tapers",
    "time_trials_tapers",
]


@pytest.fixture(scope="module")
def fourier_coefficients():
    """Complex coefficients, shape ``(n_time, n_trials, n_tapers, n_fft, n_signals)``."""
    shape = (*AXIS_SIZES.values(), N_FFT, N_SIGNALS)
    rng = np.random.default_rng(0)
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


@pytest.mark.parametrize("expectation_type", EXPECTATION_TYPES)
def test_expectation_type_reduces_the_named_axes(fourier_coefficients, expectation_type):
    """Only the named axes are averaged, with the matching observation count."""
    words = expectation_type.split("_")
    averaged_axes = tuple(AXIS_INDEX[word] for word in words)
    kept_sizes = tuple(size for name, size in AXIS_SIZES.items() if name not in words)

    conn = Connectivity(
        fourier_coefficients=fourier_coefficients, expectation_type=expectation_type
    )

    assert conn.expectation_type == expectation_type
    assert conn.n_observations == math.prod(AXIS_SIZES[word] for word in words)

    coherence = conn.coherence_magnitude()
    assert coherence.shape == (
        *kept_sizes,
        N_NONNEGATIVE_FREQUENCIES,
        N_SIGNALS,
        N_SIGNALS,
    )

    # Magnitude-squared coherence |<X0 X1*>|^2 / (<|X0|^2> <|X1|^2>), averaging
    # over exactly the named axes.
    x0, x1 = fourier_coefficients[..., 0], fourier_coefficients[..., 1]
    cross = np.mean(x0 * x1.conj(), axis=averaged_axes)
    power0 = np.mean(np.abs(x0) ** 2, axis=averaged_axes)
    power1 = np.mean(np.abs(x1) ** 2, axis=averaged_axes)
    expected = np.abs(cross) ** 2 / (power0 * power1)
    np.testing.assert_allclose(
        coherence[..., 0, 1], expected[..., :N_NONNEGATIVE_FREQUENCIES], rtol=1e-12
    )


def test_case_sensitive_expectation_type(fourier_coefficients):
    """Test that expectation_type validation is case sensitive."""
    with pytest.raises(ValueError, match="Invalid expectation_type 'TRIALS'"):
        Connectivity(
            fourier_coefficients=fourier_coefficients,
            expectation_type="TRIALS",  # uppercase should fail
        )
