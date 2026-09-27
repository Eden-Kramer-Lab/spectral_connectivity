"""Tests for parameter helper functions."""

import re

import numpy as np
import pytest

from spectral_connectivity.transforms import (
    Multitaper,
    estimate_frequency_resolution,
    estimate_n_tapers,
    suggest_parameters,
)


class TestEstimateFrequencyResolution:
    """Test the estimate_frequency_resolution function."""

    def test_basic_calculation(self):
        """Test basic frequency resolution calculation."""
        # Δf = 2 * NW / T
        # For NW=3, T=1.0s: Δf = 6 Hz
        freq_res = estimate_frequency_resolution(
            sampling_frequency=1000,
            time_window_duration=1.0,
            time_halfbandwidth_product=3,
        )
        assert np.isclose(freq_res, 6.0)

    def test_different_window_durations(self):
        """Test that shorter windows give coarser resolution."""
        # Shorter window = coarser resolution (higher Δf)
        freq_res_long = estimate_frequency_resolution(
            sampling_frequency=1000,
            time_window_duration=2.0,
            time_halfbandwidth_product=3,
        )
        freq_res_short = estimate_frequency_resolution(
            sampling_frequency=1000,
            time_window_duration=0.5,
            time_halfbandwidth_product=3,
        )
        assert freq_res_short > freq_res_long
        assert np.isclose(freq_res_long, 3.0)
        assert np.isclose(freq_res_short, 12.0)

    def test_different_time_halfbandwidth_products(self):
        """Test that higher NW gives coarser resolution (more smoothing)."""
        freq_res_low = estimate_frequency_resolution(
            sampling_frequency=1000,
            time_window_duration=1.0,
            time_halfbandwidth_product=2,
        )
        freq_res_high = estimate_frequency_resolution(
            sampling_frequency=1000,
            time_window_duration=1.0,
            time_halfbandwidth_product=5,
        )
        assert freq_res_high > freq_res_low
        assert np.isclose(freq_res_low, 4.0)
        assert np.isclose(freq_res_high, 10.0)

    def test_consistency_with_multitaper(self):
        """Test that estimates match actual Multitaper.frequency_resolution."""
        rng = np.random.default_rng(0)
        time_series = rng.standard_normal((1000, 1, 1))
        mt = Multitaper(
            time_series,
            sampling_frequency=1000,
            time_window_duration=0.5,
            time_halfbandwidth_product=4,
        )
        estimated_res = estimate_frequency_resolution(
            sampling_frequency=1000,
            time_window_duration=0.5,
            time_halfbandwidth_product=4,
        )
        assert np.isclose(estimated_res, mt.frequency_resolution)

    def test_sampling_frequency_doesnt_affect_resolution(self):
        """Test that sampling frequency doesn't affect frequency resolution.

        Frequency resolution depends on window duration in seconds, not samples.
        """
        freq_res_1khz = estimate_frequency_resolution(
            sampling_frequency=1000,
            time_window_duration=1.0,
            time_halfbandwidth_product=3,
        )
        freq_res_2khz = estimate_frequency_resolution(
            sampling_frequency=2000,
            time_window_duration=1.0,
            time_halfbandwidth_product=3,
        )
        assert np.isclose(freq_res_1khz, freq_res_2khz)


class TestEstimateNTapers:
    """Test the estimate_n_tapers function."""

    def test_basic_calculation(self):
        """Test basic n_tapers calculation."""
        # n_tapers = floor(2 * NW) - 1
        # For NW=3: n_tapers = floor(6) - 1 = 5
        n_tapers = estimate_n_tapers(time_halfbandwidth_product=3)
        assert n_tapers == 5

    def test_different_nw_values(self):
        """Test n_tapers for different time_halfbandwidth_product values."""
        # NW=2: n_tapers = floor(4) - 1 = 3
        assert estimate_n_tapers(2) == 3
        # NW=4: n_tapers = floor(8) - 1 = 7
        assert estimate_n_tapers(4) == 7
        # NW=5: n_tapers = floor(10) - 1 = 9
        assert estimate_n_tapers(5) == 9

    def test_fractional_nw(self):
        """Test n_tapers with fractional time_halfbandwidth_product."""
        # NW=3.5: n_tapers = floor(7) - 1 = 6
        assert estimate_n_tapers(3.5) == 6
        # NW=2.2: n_tapers = floor(4.4) - 1 = 3
        assert estimate_n_tapers(2.2) == 3

    def test_consistency_with_multitaper(self):
        """Test that estimates match actual Multitaper.n_tapers."""
        rng = np.random.default_rng(0)
        time_series = rng.standard_normal((1000, 1, 1))
        mt = Multitaper(
            time_series,
            sampling_frequency=1000,
            time_halfbandwidth_product=4,
        )
        estimated_n_tapers = estimate_n_tapers(time_halfbandwidth_product=4)
        assert estimated_n_tapers == mt.n_tapers

    def test_minimum_value(self):
        """Test that n_tapers is at least 1 for NW >= 1."""
        # NW=1: n_tapers = floor(2) - 1 = 1
        assert estimate_n_tapers(1) == 1
        # NW=1.5: n_tapers = floor(3) - 1 = 2
        assert estimate_n_tapers(1.5) == 2


class TestSuggestParameters:
    """Test the suggest_parameters function."""

    def test_basic_usage(self):
        """Test basic parameter suggestion."""
        params = suggest_parameters(
            sampling_frequency=1000,
            signal_duration=5.0,
        )
        # Should return a dict with suggested parameters
        assert isinstance(params, dict)
        assert "time_halfbandwidth_product" in params
        assert "time_window_duration" in params
        assert "n_tapers" in params
        assert "frequency_resolution" in params

    def test_target_frequency_resolution(self):
        """Test suggesting parameters for target frequency resolution."""
        target_res = 5.0  # 5 Hz resolution
        params = suggest_parameters(
            sampling_frequency=1000,
            signal_duration=2.0,
            desired_freq_resolution=target_res,
        )
        # The reported resolution is the target, and it is the resolution the
        # suggested NW and window actually give: delta_f = 2 * NW / T.
        assert np.isclose(params["frequency_resolution"], target_res)
        assert np.isclose(
            2 * params["time_halfbandwidth_product"] / params["time_window_duration"],
            target_res,
        )

    def test_target_n_tapers(self):
        """Test suggesting parameters for target number of tapers."""
        target_tapers = 7
        params = suggest_parameters(
            sampling_frequency=1000,
            signal_duration=2.0,
            desired_n_tapers=target_tapers,
        )
        # Should suggest NW that gives approximately this many tapers
        assert "n_tapers" in params
        assert params["n_tapers"] == target_tapers

    def test_conflicting_targets_raises_warning(self):
        """Specifying both targets warns, and the resolution target wins."""
        with pytest.warns(UserWarning, match="Both.*were specified"):
            params = suggest_parameters(
                sampling_frequency=1000,
                signal_duration=2.0,
                desired_freq_resolution=5.0,
                desired_n_tapers=7,
            )
        resolution_only = suggest_parameters(
            sampling_frequency=1000,
            signal_duration=2.0,
            desired_freq_resolution=5.0,
        )
        assert params == resolution_only
        assert np.isclose(params["frequency_resolution"], 5.0)
        assert params["n_tapers"] != 7

    def test_invalid_target_resolution_raises_error(self):
        """Test that impossible frequency resolution raises error."""
        with pytest.raises(ValueError, match=r"Cannot achieve.*frequency resolution"):
            # Try to get 0.1 Hz resolution with only 0.5s of data
            suggest_parameters(
                sampling_frequency=1000,
                signal_duration=0.5,
                desired_freq_resolution=0.1,
            )

    def test_parameters_are_reasonable(self):
        """Test that suggested parameters fall in reasonable ranges."""
        params = suggest_parameters(
            sampling_frequency=1000,
            signal_duration=5.0,
        )
        # NW should typically be between 2-5
        assert 2 <= params["time_halfbandwidth_product"] <= 5
        # Window duration shouldn't exceed signal duration
        assert params["time_window_duration"] <= 5.0
        # Should have at least a few tapers
        assert params["n_tapers"] >= 3

    def test_eeg_typical_params(self):
        """Test suggestions for typical EEG parameters."""
        # Typical EEG: 250 Hz, 60 seconds, want ~1 Hz resolution
        params = suggest_parameters(
            sampling_frequency=250,
            signal_duration=60.0,
            desired_freq_resolution=1.0,
        )
        assert np.isclose(params["frequency_resolution"], 1.0)
        assert np.isclose(
            2 * params["time_halfbandwidth_product"] / params["time_window_duration"], 1.0
        )
        assert params["n_tapers"] >= 3  # Should have reasonable averaging

    def test_lfp_typical_params(self):
        """Test suggestions for typical LFP parameters."""
        # Typical LFP: 1000 Hz, 10 seconds, want ~2 Hz resolution
        params = suggest_parameters(
            sampling_frequency=1000,
            signal_duration=10.0,
            desired_freq_resolution=2.0,
        )
        assert np.isclose(params["frequency_resolution"], 2.0)
        assert np.isclose(
            2 * params["time_halfbandwidth_product"] / params["time_window_duration"], 2.0
        )
        assert params["n_tapers"] >= 5  # LFP often uses more tapers


class TestSummarizeParameters:
    """Test the summarize_parameters method of Multitaper."""

    @staticmethod
    def _summary(time_halfbandwidth_product, time_window_duration):
        rng = np.random.default_rng(0)
        time_series = rng.standard_normal((1000, 1, 1))  # 1 s at 1000 Hz
        return Multitaper(
            time_series,
            sampling_frequency=1000,
            time_halfbandwidth_product=time_halfbandwidth_product,
            time_window_duration=time_window_duration,
        ).summarize_parameters()

    def test_returns_string(self):
        """Test that method returns a string summary."""
        summary = self._summary(3, 0.5)
        assert isinstance(summary, str)
        assert len(summary) > 0

    @pytest.mark.parametrize(
        ("nw", "duration", "expected_lines"),
        [
            # K = 2 * NW - 1; delta_f = 2 * NW / T; windows = 1 s / T
            (
                4,
                0.5,
                [
                    r"Sampling frequency:\s+1000 Hz",
                    r"Time-halfbandwidth product:\s+4",
                    r"Number of tapers:\s+7",
                    r"Window duration:\s+0\.500 s \(500 samples\)",
                    r"Number of windows:\s+2",
                    r"Frequency resolution:\s+16\.0 Hz",
                    r"Nyquist frequency:\s+500\.0 Hz",
                ],
            ),
            (
                3,
                1.0,
                [
                    r"Sampling frequency:\s+1000 Hz",
                    r"Time-halfbandwidth product:\s+3",
                    r"Number of tapers:\s+5",
                    r"Window duration:\s+1\.000 s \(1000 samples\)",
                    r"Number of windows:\s+1",
                    r"Frequency resolution:\s+6\.0 Hz",
                    r"Nyquist frequency:\s+500\.0 Hz",
                ],
            ),
        ],
    )
    def test_reports_parameter_values(self, nw, duration, expected_lines):
        """Each key parameter is rendered on its own line with the right value."""
        summary_lines = self._summary(nw, duration).splitlines()
        for pattern in expected_lines:
            assert any(re.fullmatch(pattern, line.strip()) for line in summary_lines), (
                f"no summary line matches {pattern!r}"
            )

    def test_readable_format(self):
        """Test that summary is human-readable (not repr)."""
        summary = self._summary(3, 1.0)
        assert summary.startswith("Multitaper Spectral Analysis Configuration\n")
        assert "\n" in summary
