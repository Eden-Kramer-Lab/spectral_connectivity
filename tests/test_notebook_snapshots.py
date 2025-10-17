"""Snapshot tests based on tutorial notebooks.

These tests ensure that the computational outputs from the tutorial notebooks
remain consistent across changes to the codebase. They test the full pipeline
from data simulation through connectivity analysis.

The tests use the Syrupy snapshot testing library. To update snapshots after
intentional changes, run: pytest --snapshot-update
"""

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from spectral_connectivity import (
    Connectivity,
    Multitaper,
    multitaper_connectivity,
    prepare_time_series,
)
from spectral_connectivity.simulate import simulate_MVAR

# =============================================================================
# Tutorial Intro Examples
# =============================================================================


@pytest.fixture
def intro_tutorial_signals() -> dict[str, Any]:
    """Generate the basic two-signal test case from Intro_tutorial.ipynb.

    Creates two 200 Hz oscillations offset by π/2 in phase, with added noise.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:
        - 'signal_plus_noise': NDArray with shape (60001, 2)
        - 'sampling_frequency': int, 1000 Hz
        - 'frequency_of_interest': int, 200 Hz
    """
    frequency_of_interest = 200
    sampling_frequency = 1000
    time_extent = (0, 60)
    n_signals = 2

    n_time_samples = ((time_extent[1] - time_extent[0]) * sampling_frequency) + 1

    time = np.linspace(
        time_extent[0], time_extent[1], num=n_time_samples, endpoint=True
    )
    signal = np.zeros((n_time_samples, n_signals))
    signal[:, 0] = np.sin(2 * np.pi * time * frequency_of_interest)

    phase_offset = np.pi / 2
    signal[:, 1] = np.sin((2 * np.pi * time * frequency_of_interest) + phase_offset)

    # Use fixed random seed for reproducibility
    rng = np.random.RandomState(42)
    noise = rng.normal(0, 4, signal.shape)

    return {
        "signal_plus_noise": signal + noise,
        "sampling_frequency": sampling_frequency,
        "frequency_of_interest": frequency_of_interest,
    }


def test_intro_basic_power_spectrum(snapshot, intro_tutorial_signals):
    """Test power spectrum computation from Intro_tutorial.ipynb."""
    time_series = prepare_time_series(
        intro_tutorial_signals["signal_plus_noise"], axis="signals"
    )
    multitaper = Multitaper(
        time_series,
        sampling_frequency=intro_tutorial_signals["sampling_frequency"],
        time_halfbandwidth_product=5,
    )
    connectivity = Connectivity.from_multitaper(
        multitaper, expectation_type="time_trials_tapers"
    )
    power = connectivity.power()

    # Test shape
    assert power.shape == snapshot(name="power_shape")

    # Test power values at key frequencies
    freq_200hz_idx = np.argmin(
        np.abs(
            connectivity.frequencies - intro_tutorial_signals["frequency_of_interest"]
        )
    )
    assert power[freq_200hz_idx, :].tolist() == snapshot(name="power_at_200hz")


def test_intro_basic_coherence(snapshot, intro_tutorial_signals):
    """Test coherence magnitude from Intro_tutorial.ipynb."""
    time_series = prepare_time_series(
        intro_tutorial_signals["signal_plus_noise"], axis="signals"
    )
    multitaper = Multitaper(
        time_series,
        sampling_frequency=intro_tutorial_signals["sampling_frequency"],
        time_halfbandwidth_product=5,
    )
    connectivity = Connectivity.from_multitaper(
        multitaper, expectation_type="time_trials_tapers"
    )
    coherence = connectivity.coherence_magnitude()

    # Test shape
    assert coherence.shape == snapshot(name="coherence_shape")

    # Test coherence at 200 Hz (should be high)
    freq_200hz_idx = np.argmin(
        np.abs(
            connectivity.frequencies - intro_tutorial_signals["frequency_of_interest"]
        )
    )
    assert coherence[freq_200hz_idx, 0, 1] == snapshot(name="coherence_at_200hz")


def test_intro_coherence_with_time_windows(snapshot, intro_tutorial_signals):
    """Test coherence with time windowing from Intro_tutorial.ipynb."""
    time_series = prepare_time_series(
        intro_tutorial_signals["signal_plus_noise"], axis="signals"
    )
    multitaper = Multitaper(
        time_series,
        sampling_frequency=intro_tutorial_signals["sampling_frequency"],
        time_halfbandwidth_product=5,
        time_window_duration=2.0,
    )
    connectivity = Connectivity.from_multitaper(
        multitaper, expectation_type="trials_tapers"
    )
    coherence = connectivity.coherence_magnitude()

    # Test shape includes time dimension
    assert coherence.shape == snapshot(name="coherence_time_shape")

    # Test frequency resolution changed
    assert multitaper.frequency_resolution == snapshot(name="frequency_resolution")


def test_intro_multitaper_connectivity_wrapper(snapshot, intro_tutorial_signals):
    """Test the multitaper_connectivity wrapper function."""
    time_series = prepare_time_series(
        intro_tutorial_signals["signal_plus_noise"], axis="signals"
    )
    coherence = multitaper_connectivity(
        time_series,
        sampling_frequency=intro_tutorial_signals["sampling_frequency"],
        time_halfbandwidth_product=5,
        time_window_duration=2.0,
        method="coherence_magnitude",
    )

    # Test that output is xarray DataArray
    assert type(coherence).__name__ == snapshot(name="output_type")

    # Test dimensions
    assert list(coherence.dims) == snapshot(name="output_dimensions")

    # Test coordinate names
    assert list(coherence.coords.keys()) == snapshot(name="coordinate_names")


# =============================================================================
# Paper Examples - Baccala et al. (2001)
# =============================================================================


def baccala_example2() -> tuple[npt.NDArray[np.floating], int]:
    """Baccalá, L.A., and Sameshima, K. (2001).

    Partial directed coherence: a new concept in neural structure
    determination. Biological Cybernetics 84, 463–474.

    Returns
    -------
    time_series : NDArray[floating], shape (1000, 500, 3)
        Simulated MVAR time series with 3 signals, 500 trials.
    sampling_frequency : int
        Sampling frequency of 200 Hz.
    """
    sampling_frequency = 200
    n_time_samples, _n_lags, n_signals = 1000, 1, 3

    coefficients = np.array([[[0.5, 0.3, 0.4], [-0.5, 0.3, 1.0], [0.0, -0.3, -0.2]]])
    noise_covariance = np.eye(n_signals)

    time_series = simulate_MVAR(
        coefficients,
        noise_covariance=noise_covariance,
        n_time_samples=n_time_samples,
        n_trials=500,
        n_burnin_samples=500,
        random_state=42,
    )
    return time_series, sampling_frequency


def test_baccala_example2_partial_directed_coherence(snapshot):
    """Test PDC computation for Baccala Example 2."""
    time_series, sampling_frequency = baccala_example2()

    m = Multitaper(
        time_series,
        sampling_frequency=sampling_frequency,
        time_halfbandwidth_product=1,
    )
    c = Connectivity.from_multitaper(m)
    pdc = c.partial_directed_coherence()

    # Test shape
    assert pdc.shape == snapshot(name="pdc_shape")

    # Test representative values at specific frequency bins
    freq_25hz_idx = np.argmin(np.abs(c.frequencies - 25))
    assert pdc[0, freq_25hz_idx, :, :].tolist() == snapshot(name="pdc_at_25hz")


def test_baccala_example2_directed_transfer_function(snapshot):
    """Test DTF computation for Baccala Example 2."""
    time_series, sampling_frequency = baccala_example2()

    m = Multitaper(
        time_series,
        sampling_frequency=sampling_frequency,
        time_halfbandwidth_product=1,
    )
    c = Connectivity.from_multitaper(m)
    dtf = c.directed_transfer_function()

    # Test shape
    assert dtf.shape == snapshot(name="dtf_shape")

    # Test representative values
    freq_25hz_idx = np.argmin(np.abs(c.frequencies - 25))
    assert dtf[0, freq_25hz_idx, :, :].tolist() == snapshot(name="dtf_at_25hz")


# =============================================================================
# Paper Examples - Ding et al. (2006)
# =============================================================================


def ding_example1() -> tuple[npt.NDArray[np.floating], int]:
    """Ding, M., Chen, Y., and Bressler, S.L. (2006).

    17 Granger causality: basic theory and application to neuroscience.
    Handbook of Time Series Analysis: Recent Theoretical Developments
    and Applications 437.

    Returns
    -------
    time_series : NDArray[floating], shape (1000, 500, 2)
        Simulated MVAR time series with 2 signals, 500 trials.
    sampling_frequency : int
        Sampling frequency of 200 Hz.
    """
    sampling_frequency = 200
    n_time_samples, n_lags, n_signals = 1000, 2, 2
    coefficients = np.zeros((n_lags, n_signals, n_signals))

    coefficients[0, ...] = np.array([[0.90, 0.00], [0.16, 0.80]])
    coefficients[1, ...] = np.array([[-0.50, 0.00], [-0.20, -0.50]])

    noise_covariance = np.array([[1.0, 0.4], [0.4, 0.7]])

    time_series = simulate_MVAR(
        coefficients,
        noise_covariance=noise_covariance,
        n_time_samples=n_time_samples,
        n_trials=500,
        n_burnin_samples=500,
        random_state=42,
    )
    return time_series, sampling_frequency


def test_ding_example1_granger_causality(snapshot):
    """Test Granger causality for Ding Example 1."""
    time_series, sampling_frequency = ding_example1()

    m = Multitaper(
        time_series,
        sampling_frequency=sampling_frequency,
        time_halfbandwidth_product=3,
    )
    c = Connectivity.from_multitaper(m)
    granger = c.pairwise_spectral_granger_prediction()

    # Test shape
    assert granger.shape == snapshot(name="granger_shape")

    # Test that x1 -> x2 causality is near zero (no causal influence)
    freq_50hz_idx = np.argmin(np.abs(c.frequencies - 50))
    assert granger[0, freq_50hz_idx, 1, 0] == snapshot(name="granger_x1_to_x2_at_50hz")

    # Test that x2 -> x1 causality is non-zero (x2 influences x1)
    assert granger[0, freq_50hz_idx, 0, 1] == snapshot(name="granger_x2_to_x1_at_50hz")


def test_ding_example1_generalized_pdc(snapshot):
    """Test generalized PDC for Ding Example 1."""
    time_series, sampling_frequency = ding_example1()

    m = Multitaper(
        time_series,
        sampling_frequency=sampling_frequency,
        time_halfbandwidth_product=3,
    )
    c = Connectivity.from_multitaper(m)
    gpdc = c.generalized_partial_directed_coherence()

    # Test shape
    assert gpdc.shape == snapshot(name="gpdc_shape")

    # Test representative values
    freq_50hz_idx = np.argmin(np.abs(c.frequencies - 50))
    assert gpdc[0, freq_50hz_idx, :, :].tolist() == snapshot(name="gpdc_at_50hz")


# =============================================================================
# Paper Examples - Dhamala et al. (2008)
# =============================================================================


def dhamala_example1() -> tuple[npt.NDArray[np.floating], int]:
    """Dhamala, M., Rangarajan, G., and Ding, M. (2008).

    Analyzing information flow in brain networks with nonparametric
    Granger causality. NeuroImage 41, 354–362.

    Returns
    -------
    time_series : NDArray[floating], shape (4000, 500, 3)
        Simulated MVAR time series with 3 signals, 500 trials.
    sampling_frequency : int
        Sampling frequency of 200 Hz.
    """
    sampling_frequency = 200
    n_time_samples, n_lags, n_signals = 4000, 2, 3
    coefficients = np.zeros((n_lags, n_signals, n_signals))

    coefficients[0, 0, 0] = 0.80
    coefficients[1, 0, 0] = -0.50
    coefficients[0, 0, 2] = 0.40
    coefficients[0, 1, 1] = 0.53
    coefficients[1, 1, 1] = -0.80
    coefficients[0, 2, 2] = 0.50
    coefficients[1, 2, 2] = -0.20
    coefficients[0, 2, 1] = 0.50

    noise_covariance = np.eye(n_signals) * [0.25, 1.00, 0.25]

    time_series = simulate_MVAR(
        coefficients,
        noise_covariance=noise_covariance,
        n_time_samples=n_time_samples,
        n_trials=500,
        n_burnin_samples=1000,
        random_state=42,
    )
    return time_series, sampling_frequency


def test_dhamala_example1_direct_dtf(snapshot):
    """Test direct DTF for Dhamala Example 1."""
    time_series, sampling_frequency = dhamala_example1()

    m = Multitaper(
        time_series,
        sampling_frequency=sampling_frequency,
        time_halfbandwidth_product=1,
    )
    c = Connectivity.from_multitaper(m)
    ddtf = c.direct_directed_transfer_function()

    # Test shape
    assert ddtf.shape == snapshot(name="ddtf_shape")

    # Test representative values at 30 Hz
    freq_30hz_idx = np.argmin(np.abs(c.frequencies - 30))
    assert ddtf[0, freq_30hz_idx, :, :].tolist() == snapshot(name="ddtf_at_30hz")


# =============================================================================
# Additional connectivity measures
# =============================================================================


def test_intro_coherence_phase(snapshot, intro_tutorial_signals):
    """Test coherence phase computation."""
    time_series = prepare_time_series(
        intro_tutorial_signals["signal_plus_noise"], axis="signals"
    )
    multitaper = Multitaper(
        time_series,
        sampling_frequency=intro_tutorial_signals["sampling_frequency"],
        time_halfbandwidth_product=5,
    )
    connectivity = Connectivity.from_multitaper(
        multitaper, expectation_type="time_trials_tapers"
    )
    phase = connectivity.coherence_phase()

    # Test shape
    assert phase.shape == snapshot(name="phase_shape")

    # Test phase at 200 Hz (should be close to π/2)
    freq_200hz_idx = np.argmin(
        np.abs(
            connectivity.frequencies - intro_tutorial_signals["frequency_of_interest"]
        )
    )
    # Phase should be approximately π/2 due to the phase offset
    assert np.abs(phase[freq_200hz_idx, 0, 1]) == snapshot(name="phase_at_200hz")


def test_intro_imaginary_coherence(snapshot, intro_tutorial_signals):
    """Test imaginary coherence computation."""
    time_series = prepare_time_series(
        intro_tutorial_signals["signal_plus_noise"], axis="signals"
    )
    multitaper = Multitaper(
        time_series,
        sampling_frequency=intro_tutorial_signals["sampling_frequency"],
        time_halfbandwidth_product=5,
    )
    connectivity = Connectivity.from_multitaper(
        multitaper, expectation_type="time_trials_tapers"
    )
    imag_coh = connectivity.imaginary_coherence()

    # Test shape
    assert imag_coh.shape == snapshot(name="imag_coherence_shape")

    # Test imaginary coherence at 200 Hz
    freq_200hz_idx = np.argmin(
        np.abs(
            connectivity.frequencies - intro_tutorial_signals["frequency_of_interest"]
        )
    )
    assert imag_coh[freq_200hz_idx, 0, 1] == snapshot(name="imag_coherence_at_200hz")


def test_intro_canonical_coherence(snapshot):
    """Test canonical coherence with multiple brain areas."""
    # Create signals with 3 channels per 2 brain areas
    sampling_frequency = 1000
    n_time_samples = 1000
    n_signals = 6

    rng = np.random.RandomState(42)
    time_series_2d = rng.randn(n_time_samples, n_signals)

    # Add some shared oscillation between areas
    time = np.linspace(0, 1, n_time_samples)
    shared_signal = np.sin(2 * np.pi * 50 * time)
    time_series_2d[:, :3] += shared_signal[:, np.newaxis] * 0.5
    time_series_2d[:, 3:] += shared_signal[:, np.newaxis] * 0.3

    time_series = prepare_time_series(time_series_2d, axis="signals")
    multitaper = Multitaper(
        time_series, sampling_frequency=sampling_frequency, time_halfbandwidth_product=3
    )
    connectivity = Connectivity.from_multitaper(
        multitaper, expectation_type="time_trials_tapers"
    )

    brain_area_labels = np.array([0, 0, 0, 1, 1, 1])
    can_coh, labels = connectivity.canonical_coherence(brain_area_labels)

    # Test shape (should be n_time x n_frequencies x n_areas x n_areas)
    assert can_coh.shape == snapshot(name="canonical_coherence_shape")

    # Test labels
    assert labels.tolist() == snapshot(name="canonical_coherence_labels")

    # Test value at 50 Hz between areas
    freq_50hz_idx = np.argmin(np.abs(connectivity.frequencies - 50))
    assert can_coh[0, freq_50hz_idx, 0, 1] == snapshot(
        name="canonical_coherence_at_50hz"
    )


def test_intro_debiased_squared_phase_lag_index(snapshot, intro_tutorial_signals):
    """Test debiased squared phase lag index computation."""
    time_series = prepare_time_series(
        intro_tutorial_signals["signal_plus_noise"], axis="signals"
    )
    multitaper = Multitaper(
        time_series,
        sampling_frequency=intro_tutorial_signals["sampling_frequency"],
        time_halfbandwidth_product=5,
    )
    connectivity = Connectivity.from_multitaper(
        multitaper, expectation_type="trials_tapers"
    )
    debiased_pli2 = connectivity.debiased_squared_phase_lag_index()

    # Test shape
    assert debiased_pli2.shape == snapshot(name="debiased_pli2_shape")

    # Test value at 200 Hz
    freq_200hz_idx = np.argmin(
        np.abs(
            connectivity.frequencies - intro_tutorial_signals["frequency_of_interest"]
        )
    )
    assert debiased_pli2[0, freq_200hz_idx, 0, 1] == snapshot(
        name="debiased_pli2_at_200hz"
    )


# =============================================================================
# Additional connectivity measures - Remaining methods
# =============================================================================


def test_intro_coherency(snapshot, intro_tutorial_signals):
    """Test coherency (complex-valued coherence) computation."""
    time_series = prepare_time_series(
        intro_tutorial_signals["signal_plus_noise"], axis="signals"
    )
    multitaper = Multitaper(
        time_series,
        sampling_frequency=intro_tutorial_signals["sampling_frequency"],
        time_halfbandwidth_product=5,
    )
    connectivity = Connectivity.from_multitaper(
        multitaper, expectation_type="time_trials_tapers"
    )
    coherency = connectivity.coherency()

    # Test shape
    assert coherency.shape == snapshot(name="coherency_shape")

    # Test value at 200 Hz
    freq_200hz_idx = np.argmin(
        np.abs(
            connectivity.frequencies - intro_tutorial_signals["frequency_of_interest"]
        )
    )
    # Test magnitude and phase separately for complex number
    assert np.abs(coherency[freq_200hz_idx, 0, 1]) == snapshot(
        name="coherency_magnitude_at_200hz"
    )
    assert np.angle(coherency[freq_200hz_idx, 0, 1]) == snapshot(
        name="coherency_phase_at_200hz"
    )


def test_intro_global_coherence(snapshot):
    """Test global coherence computation."""
    # Create 4 signals with shared oscillation
    sampling_frequency = 1000
    n_time_samples = 1000
    n_signals = 4

    rng = np.random.RandomState(42)
    time_series_2d = rng.randn(n_time_samples, n_signals)

    # Add some shared oscillation across all signals
    time = np.linspace(0, 1, n_time_samples)
    shared_signal = np.sin(2 * np.pi * 50 * time)
    for i in range(n_signals):
        time_series_2d[:, i] += shared_signal * (0.5 - 0.1 * i)

    time_series = prepare_time_series(time_series_2d, axis="signals")
    multitaper = Multitaper(
        time_series, sampling_frequency=sampling_frequency, time_halfbandwidth_product=3
    )
    connectivity = Connectivity.from_multitaper(
        multitaper, expectation_type="time_trials_tapers"
    )

    # global_coherence returns (coherence, components)
    global_coh, components = connectivity.global_coherence()

    # Test shape
    assert global_coh.shape == snapshot(name="global_coherence_shape")

    # Test components shape
    assert components.shape == snapshot(name="global_coherence_components_shape")

    # Test a representative value (mean of coherence)
    assert np.mean(global_coh) == snapshot(name="global_coherence_mean")


def test_intro_group_delay(snapshot, intro_tutorial_signals):
    """Test group delay computation."""
    time_series = prepare_time_series(
        intro_tutorial_signals["signal_plus_noise"], axis="signals"
    )
    multitaper = Multitaper(
        time_series,
        sampling_frequency=intro_tutorial_signals["sampling_frequency"],
        time_halfbandwidth_product=5,
    )
    connectivity = Connectivity.from_multitaper(
        multitaper, expectation_type="time_trials_tapers"
    )
    # Specify frequency range around 200 Hz
    delay, slope, r_value = connectivity.group_delay(
        frequencies_of_interest=[100, 300], frequency_resolution=5.0
    )

    # Test shapes
    assert delay.shape == snapshot(name="group_delay_shape")
    assert slope.shape == snapshot(name="group_delay_slope_shape")
    assert r_value.shape == snapshot(name="group_delay_r_value_shape")

    # Test a representative value (use nanmean since some values may be NaN)
    assert np.nanmean(delay) == snapshot(name="group_delay_mean")


def test_intro_delay(snapshot, intro_tutorial_signals):
    """Test delay computation."""
    time_series = prepare_time_series(
        intro_tutorial_signals["signal_plus_noise"], axis="signals"
    )
    multitaper = Multitaper(
        time_series,
        sampling_frequency=intro_tutorial_signals["sampling_frequency"],
        time_halfbandwidth_product=5,
    )
    connectivity = Connectivity.from_multitaper(
        multitaper, expectation_type="time_trials_tapers"
    )
    # Specify frequency range around 200 Hz
    delay_result = connectivity.delay(
        frequencies_of_interest=[100, 300], frequency_resolution=5.0
    )

    # Test shape
    assert delay_result.shape == snapshot(name="delay_shape")

    # Test a representative value
    assert np.mean(delay_result) == snapshot(name="delay_mean")


def test_intro_phase_locking_value(snapshot, intro_tutorial_signals):
    """Test phase locking value computation."""
    time_series = prepare_time_series(
        intro_tutorial_signals["signal_plus_noise"], axis="signals"
    )
    multitaper = Multitaper(
        time_series,
        sampling_frequency=intro_tutorial_signals["sampling_frequency"],
        time_halfbandwidth_product=5,
    )
    connectivity = Connectivity.from_multitaper(
        multitaper, expectation_type="trials_tapers"
    )
    plv = connectivity.phase_locking_value()

    # Test shape
    assert plv.shape == snapshot(name="plv_shape")

    # Test value at 200 Hz (should be high due to fixed phase relationship)
    freq_200hz_idx = np.argmin(
        np.abs(
            connectivity.frequencies - intro_tutorial_signals["frequency_of_interest"]
        )
    )
    assert plv[0, freq_200hz_idx, 0, 1] == snapshot(name="plv_at_200hz")


def test_intro_phase_slope_index(snapshot, intro_tutorial_signals):
    """Test phase slope index computation."""
    time_series = prepare_time_series(
        intro_tutorial_signals["signal_plus_noise"], axis="signals"
    )
    multitaper = Multitaper(
        time_series,
        sampling_frequency=intro_tutorial_signals["sampling_frequency"],
        time_halfbandwidth_product=5,
    )
    connectivity = Connectivity.from_multitaper(
        multitaper, expectation_type="time_trials_tapers"
    )
    # Specify frequency range around 200 Hz
    psi = connectivity.phase_slope_index(
        frequencies_of_interest=[100, 300], frequency_resolution=5.0
    )

    # Test shape
    assert psi.shape == snapshot(name="psi_shape")

    # Test a representative value
    assert np.mean(psi) == snapshot(name="psi_mean")


def test_intro_pairwise_phase_consistency(snapshot, intro_tutorial_signals):
    """Test pairwise phase consistency computation."""
    time_series = prepare_time_series(
        intro_tutorial_signals["signal_plus_noise"], axis="signals"
    )
    multitaper = Multitaper(
        time_series,
        sampling_frequency=intro_tutorial_signals["sampling_frequency"],
        time_halfbandwidth_product=5,
    )
    connectivity = Connectivity.from_multitaper(
        multitaper, expectation_type="trials_tapers"
    )
    ppc = connectivity.pairwise_phase_consistency()

    # Test shape
    assert ppc.shape == snapshot(name="ppc_shape")

    # Test value at 200 Hz
    freq_200hz_idx = np.argmin(
        np.abs(
            connectivity.frequencies - intro_tutorial_signals["frequency_of_interest"]
        )
    )
    assert ppc[0, freq_200hz_idx, 0, 1] == snapshot(name="ppc_at_200hz")


def test_intro_weighted_phase_lag_index(snapshot, intro_tutorial_signals):
    """Test weighted phase lag index computation."""
    time_series = prepare_time_series(
        intro_tutorial_signals["signal_plus_noise"], axis="signals"
    )
    multitaper = Multitaper(
        time_series,
        sampling_frequency=intro_tutorial_signals["sampling_frequency"],
        time_halfbandwidth_product=5,
    )
    connectivity = Connectivity.from_multitaper(
        multitaper, expectation_type="time_trials_tapers"
    )
    wpli = connectivity.weighted_phase_lag_index()

    # Test shape
    assert wpli.shape == snapshot(name="wpli_shape")

    # Test value at 200 Hz
    freq_200hz_idx = np.argmin(
        np.abs(
            connectivity.frequencies - intro_tutorial_signals["frequency_of_interest"]
        )
    )
    assert wpli[freq_200hz_idx, 0, 1] == snapshot(name="wpli_at_200hz")


def test_intro_debiased_squared_weighted_phase_lag_index(
    snapshot, intro_tutorial_signals
):
    """Test debiased squared weighted phase lag index computation."""
    time_series = prepare_time_series(
        intro_tutorial_signals["signal_plus_noise"], axis="signals"
    )
    multitaper = Multitaper(
        time_series,
        sampling_frequency=intro_tutorial_signals["sampling_frequency"],
        time_halfbandwidth_product=5,
    )
    connectivity = Connectivity.from_multitaper(
        multitaper, expectation_type="trials_tapers"
    )
    dwpli2 = connectivity.debiased_squared_weighted_phase_lag_index()

    # Test shape
    assert dwpli2.shape == snapshot(name="dwpli2_shape")

    # Test value at 200 Hz
    freq_200hz_idx = np.argmin(
        np.abs(
            connectivity.frequencies - intro_tutorial_signals["frequency_of_interest"]
        )
    )
    assert dwpli2[0, freq_200hz_idx, 0, 1] == snapshot(name="dwpli2_at_200hz")


def test_dhamala_example1_subset_granger(snapshot):
    """Test subset pairwise spectral Granger causality for Dhamala Example 1."""
    time_series, sampling_frequency = dhamala_example1()

    m = Multitaper(
        time_series,
        sampling_frequency=sampling_frequency,
        time_halfbandwidth_product=1,
    )
    c = Connectivity.from_multitaper(m)

    # Test subset GC: compute for signal pairs (0,1) and (1,2)
    # pairs parameter expects list of [from_node, to_node] pairs
    pairs = [[0, 1], [1, 2]]
    subset_gc = c.subset_pairwise_spectral_granger_prediction(pairs)

    # Test shape (should be n_time_windows, n_frequencies, n_signals, n_signals)
    assert subset_gc.shape == snapshot(name="subset_gc_shape")

    # Test value at 30 Hz for first pair (0->1)
    freq_30hz_idx = np.argmin(np.abs(c.frequencies - 30))
    assert subset_gc[0, freq_30hz_idx, 0, 1] == snapshot(
        name="subset_gc_pair0to1_at_30hz"
    )
