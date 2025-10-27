import numpy as np

from spectral_connectivity import Connectivity


def test_coherence_magnitude_bounds():
    """Test that coherence magnitude is bounded by [0, 1]."""
    # Create test data with potential for numerical instability
    n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals = (1, 10, 1, 5, 3)
    fourier_coefficients = (
        np.random.randn(n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals)
        + 1j
        * np.random.randn(n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals)
    ).astype(np.complex128)

    # Add very small values that could cause numerical issues
    fourier_coefficients[0, 0, 0, 0, :] = 1e-15

    conn = Connectivity(fourier_coefficients=fourier_coefficients)
    coherence_mag = conn.coherence_magnitude()

    # Check bounds with small epsilon margin (excluding NaN values on diagonal)
    eps_margin = 1e-10
    valid_values = coherence_mag[~np.isnan(coherence_mag)]
    assert np.all(valid_values >= -eps_margin), (
        f"Coherence magnitude below 0: {valid_values.min()}"
    )
    assert np.all(valid_values <= 1 + eps_margin), (
        f"Coherence magnitude above 1: {valid_values.max()}"
    )


def test_imaginary_coherence_bounds():
    """Test that imaginary coherence is bounded by [0, 1]."""
    # Create test data
    n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals = (1, 5, 1, 3, 2)
    fourier_coefficients = (
        np.random.randn(n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals)
        + 1j
        * np.random.randn(n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals)
    ).astype(np.complex128)

    # Add edge case with very small power
    fourier_coefficients[0, 0, 0, 0, :] = 1e-16

    conn = Connectivity(fourier_coefficients=fourier_coefficients)
    imag_coherence = conn.imaginary_coherence()

    # Check bounds with small epsilon margin (excluding NaN values)
    eps_margin = 1e-10
    valid_values = imag_coherence[~np.isnan(imag_coherence)]
    assert np.all(valid_values >= -eps_margin), (
        f"Imaginary coherence below 0: {valid_values.min()}"
    )
    assert np.all(valid_values <= 1 + eps_margin), (
        f"Imaginary coherence above 1: {valid_values.max()}"
    )


def test_coherence_with_zero_power():
    """Test coherence behavior with zero power signals."""
    n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals = (1, 1, 1, 1, 2)
    fourier_coefficients = np.zeros(
        (n_time_samples, n_trials, n_tapers, n_fft_samples, n_signals), dtype=complex
    )

    # One signal has zero power, the other has non-zero power
    fourier_coefficients[0, 0, 0, 0, 1] = 1.0

    conn = Connectivity(fourier_coefficients=fourier_coefficients)

    # Should not raise errors and should handle division by near-zero gracefully
    coherence_mag = conn.coherence_magnitude()
    imag_coherence = conn.imaginary_coherence()

    # Values should be finite (not inf or -inf)
    assert np.all(np.isfinite(coherence_mag[~np.isnan(coherence_mag)]))
    assert np.all(np.isfinite(imag_coherence[~np.isnan(imag_coherence)]))
