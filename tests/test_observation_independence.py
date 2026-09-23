"""Tests for the ``observations_are_independent`` contract on ``Connectivity``.

Transforms whose observations are correlated (a Morlet transform whose
smoothing neighborhood is collected on the observation axis, or Welch segments
overlapping by more than half) expose ``observations_are_independent=False``.
``Connectivity`` records the flag so consumers that assume independent
trial-taper observations (finite-sample bias corrections, the zero-coherence
null, leave-one-out intervals) can warn or refuse. The companion
``time_bins_are_independent`` flag marks correlated time bins (overlapping
windows, closely spaced Morlet samples), which matter only for expectations
that average over time.
"""

import warnings

import numpy as np
import pytest

from spectral_connectivity import Connectivity, MorletWavelet, Multitaper, Welch


class _StubTransform:
    """Duck-typed transform exposing the ``from_transform`` contract."""

    is_one_sided = False

    def __init__(self, coefficients, *, observations_are_independent=None):
        self._coefficients = coefficients
        if observations_are_independent is not None:
            self.observations_are_independent = observations_are_independent
        n_frequencies = coefficients.shape[-2]
        self.frequencies = np.fft.fftfreq(n_frequencies)
        self.time = np.arange(coefficients.shape[0], dtype=float)

    def fft(self):
        return self._coefficients.copy()


def _coefficients(rng, shape=(1, 4, 3, 16, 3)):
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def test_connectivity_defaults_to_independent_observations():
    conn = Connectivity(_coefficients(np.random.default_rng(0)))
    assert conn.observations_are_independent is True


def test_connectivity_records_correlated_observations():
    conn = Connectivity(
        _coefficients(np.random.default_rng(0)), observations_are_independent=False
    )
    assert conn.observations_are_independent is False
    with pytest.raises(AttributeError):
        conn.observations_are_independent = True  # read-only


def test_from_transform_forwards_the_flag_from_a_stub_transform():
    rng = np.random.default_rng(1)
    correlated = _StubTransform(_coefficients(rng), observations_are_independent=False)
    assert Connectivity.from_transform(correlated).observations_are_independent is False
    independent = _StubTransform(_coefficients(rng), observations_are_independent=True)
    assert Connectivity.from_transform(independent).observations_are_independent is True


def test_from_transform_treats_a_missing_attribute_as_independent():
    stub = _StubTransform(_coefficients(np.random.default_rng(2)))
    assert not hasattr(stub, "observations_are_independent")
    assert Connectivity.from_transform(stub).observations_are_independent is True


def test_from_transform_does_not_pass_the_keyword_to_an_older_subclass():
    """A subclass mirroring the pre-flag signature must keep working with an
    independent-observation transform; the keyword is only sent when False."""

    class Legacy(Connectivity):
        def __init__(self, fourier_coefficients, expectation_type="trials_tapers", **kwargs):
            kwargs.pop("is_one_sided", None)
            assert "observations_are_independent" not in kwargs
            assert "time_bins_are_independent" not in kwargs
            super().__init__(fourier_coefficients, expectation_type, **kwargs)

    stub = _StubTransform(_coefficients(np.random.default_rng(3)))
    assert Legacy.from_transform(stub).observations_are_independent is True


def test_flag_survives_coefficient_reassignment():
    rng = np.random.default_rng(4)
    conn = Connectivity(_coefficients(rng), observations_are_independent=False)
    conn.fourier_coefficients = _coefficients(rng)
    assert conn.observations_are_independent is False


# Every measure that reads ``n_observations`` as a count of independent samples:
# the finite-sample bias corrections and the zero-coherence significance null.
OBSERVATION_COUNTING_MEASURES = [
    "debiased_squared_phase_lag_index",
    "debiased_squared_weighted_phase_lag_index",
    "pairwise_phase_consistency",
    "group_delay",
    "delay",
]


@pytest.mark.parametrize("measure", OBSERVATION_COUNTING_MEASURES)
def test_observation_counting_measures_warn_once_for_correlated_observations(measure):
    conn = Connectivity(
        _coefficients(np.random.default_rng(5)), observations_are_independent=False
    )
    with pytest.warns(UserWarning, match=f"{measure} .*correlated") as record:
        getattr(conn, measure)()
    correlated = [w for w in record if "correlated" in str(w.message)]
    assert len(correlated) == 1  # one warning per call, not one per bin/pair


@pytest.mark.parametrize("measure", OBSERVATION_COUNTING_MEASURES)
def test_observation_counting_measures_are_silent_for_independent_observations(measure):
    conn = Connectivity(_coefficients(np.random.default_rng(5)))
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        getattr(conn, measure)()


@pytest.mark.parametrize("expectation_type", ["tapers", "trials_tapers"])
def test_jackknife_refuses_to_leave_out_correlated_taper_observations(expectation_type):
    conn = Connectivity(
        _coefficients(np.random.default_rng(6)),
        expectation_type=expectation_type,
        observations_are_independent=False,
    )
    with pytest.raises(ValueError, match="observations_are_independent"):
        conn.jackknife("phase_lag_index")


def test_jackknife_over_trials_is_allowed_with_correlated_tapers():
    conn = Connectivity(
        _coefficients(np.random.default_rng(6)),
        expectation_type="trials",
        observations_are_independent=False,
    )
    result = conn.jackknife("phase_lag_index")
    assert np.isfinite(result.standard_error).any()


def test_connectivity_defaults_to_independent_time_bins():
    conn = Connectivity(_coefficients(np.random.default_rng(0)))
    assert conn.time_bins_are_independent is True
    correlated = Connectivity(
        _coefficients(np.random.default_rng(0)), time_bins_are_independent=False
    )
    assert correlated.time_bins_are_independent is False
    with pytest.raises(AttributeError):
        correlated.time_bins_are_independent = True  # read-only


def test_from_transform_forwards_correlated_time_bins():
    stub = _StubTransform(_coefficients(np.random.default_rng(1), shape=(4, 2, 3, 16, 3)))
    stub.time_bins_are_independent = False
    assert Connectivity.from_transform(stub).time_bins_are_independent is False


@pytest.mark.parametrize("measure", OBSERVATION_COUNTING_MEASURES)
@pytest.mark.parametrize("expectation_type", ["time", "time_trials", "time_trials_tapers"])
def test_measures_warn_when_averaging_correlated_time_bins(measure, expectation_type):
    conn = Connectivity(
        _coefficients(np.random.default_rng(5), shape=(4, 4, 3, 16, 3)),
        expectation_type=expectation_type,
        time_bins_are_independent=False,
    )
    with pytest.warns(UserWarning, match=f"{measure} .*correlated") as record:
        getattr(conn, measure)()
    correlated = [w for w in record if "correlated" in str(w.message)]
    assert len(correlated) == 1


@pytest.mark.parametrize("measure", OBSERVATION_COUNTING_MEASURES)
def test_correlated_time_bins_are_silent_when_time_is_not_averaged(measure):
    """Each time bin is its own estimate under a trial/taper expectation, so
    correlation between bins does not enter the observation count."""
    conn = Connectivity(
        _coefficients(np.random.default_rng(5), shape=(4, 4, 3, 16, 3)),
        time_bins_are_independent=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        getattr(conn, measure)()


def test_overlapping_multitaper_windows_warn_only_when_time_is_averaged():
    series = np.random.default_rng(9).standard_normal((2000, 1, 2))
    overlapping = Multitaper(
        series,
        sampling_frequency=200,
        time_halfbandwidth_product=2,
        time_window_duration=0.5,
        time_window_step=0.05,
    )
    with pytest.warns(UserWarning, match="pairwise_phase_consistency .*correlated"):
        Connectivity.from_transform(
            overlapping, expectation_type="time_tapers"
        ).pairwise_phase_consistency()
    half_overlap = Multitaper(
        series,
        sampling_frequency=200,
        time_halfbandwidth_product=2,
        time_window_duration=0.5,
        time_window_step=0.25,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        Connectivity.from_transform(overlapping).pairwise_phase_consistency()
        Connectivity.from_transform(
            half_overlap, expectation_type="time_tapers"
        ).pairwise_phase_consistency()


def test_unsmoothed_morlet_warns_when_averaging_over_time():
    series = np.random.default_rng(10).standard_normal((800, 3, 2))
    morlet = MorletWavelet(
        series, sampling_frequency=200, frequencies=[20.0, 40.0], n_cycles=5
    )
    with pytest.warns(UserWarning, match="pairwise_phase_consistency .*correlated"):
        Connectivity.from_transform(
            morlet, expectation_type="time_trials"
        ).pairwise_phase_consistency()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        Connectivity.from_transform(
            morlet, expectation_type="trials"
        ).pairwise_phase_consistency()


def _two_channel_series(rng, n_time=800, n_trials=3):
    return rng.standard_normal((n_time, n_trials, 2))


def test_morlet_smoothing_neighborhood_blocks_taper_jackknife_end_to_end():
    # 20 Hz at 5 cycles has sigma_t ~ 0.04 s, so 0.5 s of smoothing is well above
    # the four-sigma guidance and does not trigger the MorletWavelet warning.
    morlet = MorletWavelet(
        _two_channel_series(np.random.default_rng(7)),
        sampling_frequency=200,
        frequencies=[20.0, 40.0],
        n_cycles=5,
        smoothing_time=0.5,
    )
    assert morlet.observations_are_independent is False
    conn = Connectivity.from_transform(morlet)
    assert conn.observations_are_independent is False
    with pytest.raises(ValueError, match="observations_are_independent"):
        conn.jackknife("phase_lag_index")


def test_welch_overlap_above_half_marks_observations_correlated_end_to_end():
    series = _two_channel_series(np.random.default_rng(8), n_time=2000, n_trials=1)
    independent = Welch(
        series, sampling_frequency=200, n_time_samples_per_segment=200, segment_overlap=0.5
    )
    correlated = Welch(
        series, sampling_frequency=200, n_time_samples_per_segment=200, segment_overlap=0.9
    )
    assert Connectivity.from_transform(independent).observations_are_independent is True
    conn = Connectivity.from_transform(correlated)
    assert conn.observations_are_independent is False
    with pytest.warns(UserWarning, match="pairwise_phase_consistency .*correlated"):
        conn.pairwise_phase_consistency()
