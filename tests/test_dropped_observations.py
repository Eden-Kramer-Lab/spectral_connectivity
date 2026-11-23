"""Observation weights that only keep or drop observations.

Zero weights drop observations (the invalid windows of
``Multitaper(artifact_samples=...)``, the masked edges of a Morlet transform);
every other observation keeps one common weight. Each reduced bin is then the
plain average of the observations it keeps, so every measure, including those
that count observations, must equal the measure of the coefficients with the
dropped observations physically removed.
"""

import numpy as np
import pytest

from spectral_connectivity import Connectivity
from spectral_connectivity.connectivity import _find_significant_frequencies
from tests._artifact_helpers import COUNTING_MEASURES, quietly

# Measures of every kind: plain averages, the normalized and phase-lag
# families, those that count observations, and those that factor or invert the
# cross-spectral matrix.
MEASURES = (
    "power",
    "cross_spectral_density",
    "coherency",
    "magnitude_squared_coherence",
    "imaginary_coherence",
    "phase_locking_value",
    "corrected_imaginary_phase_locking_value",
    "phase_lag_index",
    "weighted_phase_lag_index",
    "directed_phase_lag_index",
    "debiased_squared_phase_lag_index",
    "debiased_squared_weighted_phase_lag_index",
    "pairwise_phase_consistency",
    "partial_coherence",
    "pairwise_spectral_granger_prediction",
    "directed_transfer_function",
)
# The Wilson factorization iterates to a tolerance, so its inputs' rounding
# (a weighted rather than a plain mean) moves it by more than an ulp.
FACTORIZED = frozenset({"pairwise_spectral_granger_prediction", "directed_transfer_function"})


def _coefficients(rng, shape):
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def _window_trial_weights(valid, shape):
    """0/1 weights of every taper and frequency from ``valid`` (window, trial)."""
    return np.broadcast_to(valid[:, :, None, None, None], (*shape[:-1], 1))


@pytest.fixture(scope="module")
def dropped_trials():
    """Three windows of six trials, each window dropping different trials."""
    rng = np.random.default_rng(0)
    coefficients = _coefficients(rng, (3, 6, 3, 16, 3))
    valid = np.ones((3, 6), dtype=bool)
    valid[0, [1, 4]] = False
    valid[2, [0, 2, 3]] = False
    return coefficients, valid, _window_trial_weights(valid, coefficients.shape)


@pytest.mark.parametrize("measure", MEASURES)
def test_dropped_trials_match_their_removal(dropped_trials, measure):
    """Each window's value is the value of that window's kept trials alone."""
    coefficients, valid, weights = dropped_trials
    masked = quietly(getattr(Connectivity(coefficients, observation_weights=weights), measure))
    rtol = 1e-7 if measure in FACTORIZED else 1e-11
    for window in range(coefficients.shape[0]):
        kept = Connectivity(coefficients[[window]][:, valid[window]])
        np.testing.assert_allclose(
            masked[window], quietly(getattr(kept, measure))[0], rtol=rtol, atol=1e-13
        )


def test_group_measures_match_their_removal(dropped_trials):
    coefficients, valid, weights = dropped_trials
    groups = ["a", "a", "b"]
    connectivity = Connectivity(coefficients, observation_weights=weights)
    canonical, _ = connectivity.canonical_coherence(groups)
    fractions, _ = connectivity.global_coherence()
    for window in range(coefficients.shape[0]):
        kept = Connectivity(coefficients[[window]][:, valid[window]])
        np.testing.assert_allclose(
            canonical[window], kept.canonical_coherence(groups)[0][0], rtol=1e-10
        )
        np.testing.assert_allclose(
            fractions[window], kept.global_coherence()[0][0], rtol=1e-10
        )


def test_n_valid_observations_counts_each_window(dropped_trials):
    coefficients, valid, weights = dropped_trials
    connectivity = Connectivity(coefficients, observation_weights=weights)
    n_tapers = coefficients.shape[2]
    assert connectivity.n_observations == 6 * n_tapers  # the raw count
    counts = connectivity.n_valid_observations
    assert counts.shape == (3, 9)  # (n_time_windows, n_nonnegative_frequencies)
    expected = np.broadcast_to((valid.sum(axis=1) * n_tapers)[:, None], counts.shape)
    np.testing.assert_array_equal(counts, expected)


@pytest.mark.parametrize(
    ("expectation_type", "shape"),
    [
        ("trials_tapers", (3, 9)),
        ("tapers", (3, 6, 9)),
        ("trials", (3, 3, 9)),
        ("time_trials_tapers", (9,)),
    ],
)
def test_n_valid_observations_without_weights_is_n_observations(
    dropped_trials, expectation_type, shape
):
    coefficients, _, _ = dropped_trials
    connectivity = Connectivity(coefficients, expectation_type=expectation_type)
    counts = connectivity.n_valid_observations
    assert counts.shape == shape
    assert (counts == connectivity.n_observations).all()


@pytest.mark.parametrize("measure", COUNTING_MEASURES)
def test_counting_measures_use_each_frequency_count(measure):
    """Weights that drop different observations at different frequencies: each
    (window, frequency) bin counts its own."""
    rng = np.random.default_rng(1)
    coefficients = _coefficients(rng, (2, 8, 2, 8, 2))
    valid = rng.random((*coefficients.shape[:-1], 1)) > 0.3
    valid[..., 0] |= np.arange(8)[None, :, None, None] < 2  # at least 2 trials each
    connectivity = Connectivity(coefficients, observation_weights=valid.astype(float))
    value = getattr(connectivity, measure)()
    for window in range(2):
        for frequency in range(5):
            observations = valid[window, :, :, frequency, 0]
            kept = Connectivity(
                coefficients[window][observations][None, :, None, [frequency], :],
                is_one_sided=True,
            )
            np.testing.assert_allclose(
                value[window, frequency],
                getattr(kept, measure)()[0, 0],
                rtol=1e-11,
                atol=1e-14,
            )


def test_a_scaled_drop_weight_is_still_only_a_drop():
    """The kept observations' common weight need not be 1."""
    rng = np.random.default_rng(2)
    coefficients = _coefficients(rng, (1, 6, 2, 8, 2))
    weights = np.full((*coefficients.shape[:-1], 1), 2.5)
    weights[0, :2] = 0
    scaled = Connectivity(coefficients, observation_weights=weights)
    kept = Connectivity(coefficients[:, 2:])
    np.testing.assert_allclose(
        scaled.pairwise_phase_consistency(), kept.pairwise_phase_consistency(), rtol=1e-12
    )


def test_reweighting_observations_is_still_refused():
    rng = np.random.default_rng(3)
    coefficients = _coefficients(rng, (1, 6, 2, 8, 2))
    weights = np.ones((*coefficients.shape[:-1], 1))
    weights[0, 0] = 0.5
    connectivity = Connectivity(coefficients, observation_weights=weights)
    with pytest.raises(ValueError, match="non-uniform observation_weights"):
        connectivity.pairwise_phase_consistency()


@pytest.fixture
def single_observation_window():
    """Two windows of four trials of one taper; window 1 keeps one trial."""
    coefficients = _coefficients(np.random.default_rng(4), (2, 4, 1, 8, 2))
    valid = np.ones((2, 4), dtype=bool)
    valid[1, 1:] = False
    weights = _window_trial_weights(valid, coefficients.shape)
    return Connectivity(coefficients, observation_weights=weights)


@pytest.mark.parametrize("measure", COUNTING_MEASURES)
def test_a_bin_of_one_observation_is_nan_with_one_warning(single_observation_window, measure):
    connectivity = single_observation_window
    with pytest.warns(UserWarning, match="single observation") as record:
        value = getattr(connectivity, measure)()
    assert len(record) == 1  # one warning, naming this measure
    assert str(record[0].message).startswith(measure)
    assert np.isnan(value[1, :, 0, 1]).all()
    assert np.isfinite(value[0, 1:-1, 0, 1]).all()


def test_a_bin_of_one_observation_warns_for_a_normalized_measure(single_observation_window):
    with pytest.warns(UserWarning, match="single observation in some bins"):
        single_observation_window.magnitude_squared_coherence()


def test_zero_filled_dropped_observations_do_not_warn_of_zero_magnitude():
    """A dropped observation may be all zero (a zero-filled invalid window);
    the phase-locking normalization never averages it, so it does not warn."""
    rng = np.random.default_rng(6)
    coefficients = _coefficients(rng, (1, 5, 2, 8, 2))
    coefficients[0, 0] = 0
    weights = np.ones((*coefficients.shape[:-1], 1))
    weights[0, 0] = 0
    plv = Connectivity(coefficients, observation_weights=weights).phase_locking_value()
    np.testing.assert_allclose(
        plv, Connectivity(coefficients[:, 1:]).phase_locking_value(), rtol=1e-12
    )


@pytest.mark.parametrize("measure", ["delay", "group_delay"])
def test_delay_significance_counts_the_kept_observations(measure):
    """The zero-coherence test of each bin uses its kept count: the result is
    that of the kept observations alone, and differs from what the raw count
    would give."""
    rng = np.random.default_rng(7)
    n_samples, n_trials = 64, 12
    source = rng.standard_normal((n_trials, n_samples))
    lagged = np.roll(source, 2, axis=-1) + 1.5 * rng.standard_normal(source.shape)
    coefficients = np.fft.fft(np.stack([source, lagged], axis=-1), axis=1)
    coefficients = coefficients[None, :, None, :, :]  # (1, trials, 1, fft, 2)
    weights = np.ones((*coefficients.shape[:-1], 1))
    weights[0, 7:] = 0  # keep 7 of 12 trials
    frequencies = np.fft.fftfreq(n_samples)
    masked = Connectivity(coefficients, frequencies=frequencies, observation_weights=weights)
    kept = Connectivity(coefficients[:, :7], frequencies=frequencies)
    np.testing.assert_allclose(
        np.asarray(getattr(masked, measure)(), dtype=float),
        np.asarray(getattr(kept, measure)(), dtype=float),
        rtol=1e-10,
    )
    # The raw count, 12, would select different frequencies.
    coherency = kept.coherency()[..., [0], [1]]
    assert not np.array_equal(
        _find_significant_frequencies(coherency, 7),
        _find_significant_frequencies(coherency, masked.n_observations),
    )


@pytest.mark.parametrize(
    "method", ["magnitude_squared_coherence", "pairwise_phase_consistency", "power"]
)
def test_jackknife_leaves_out_only_kept_observations(dropped_trials, method):
    """Each window's interval is the jackknife of its kept trials alone: a
    dropped trial is no replicate there, and the degrees of freedom count the
    kept trials and tapers."""
    coefficients, valid, weights = dropped_trials
    masked = Connectivity(coefficients, observation_weights=weights).jackknife(method)
    n_tapers = coefficients.shape[2]
    expected_counts = valid.sum(axis=1) * n_tapers
    np.testing.assert_array_equal(np.ravel(masked.n_observations), expected_counts)
    for window in range(coefficients.shape[0]):
        kept = Connectivity(coefficients[[window]][:, valid[window]]).jackknife(method)
        assert kept.n_observations == expected_counts[window]
        for field in ("estimate", "bias_corrected", "standard_error"):
            np.testing.assert_allclose(
                getattr(masked, field)[window],
                getattr(kept, field)[0],
                rtol=1e-9,
                atol=1e-13,
                err_msg=field,
            )
        for bound, kept_bound in zip(
            masked.confidence_interval, kept.confidence_interval, strict=True
        ):
            np.testing.assert_allclose(bound[window], kept_bound[0], rtol=1e-9, atol=1e-13)


def test_jackknife_skips_a_unit_dropped_everywhere():
    rng = np.random.default_rng(8)
    coefficients = _coefficients(rng, (2, 5, 1, 8, 2))
    weights = np.ones((*coefficients.shape[:-1], 1))
    weights[:, 0] = 0  # trial 0 is dropped in every window
    masked = Connectivity(coefficients, observation_weights=weights)
    kept = Connectivity(coefficients[:, 1:])
    result = masked.jackknife("magnitude_squared_coherence")
    assert result.n_observations == 4
    np.testing.assert_allclose(
        result.standard_error,
        kept.jackknife("magnitude_squared_coherence").standard_error,
        rtol=1e-10,
    )


def test_jackknife_is_nan_in_a_bin_of_too_few_observations():
    rng = np.random.default_rng(9)
    coefficients = _coefficients(rng, (2, 5, 1, 8, 2))
    weights = np.ones((*coefficients.shape[:-1], 1))
    weights[1, 2:] = 0  # window 1 keeps two trials
    connectivity = Connectivity(coefficients, observation_weights=weights)
    with pytest.warns(UserWarning, match="fewer than 3 observations in 1 of 2 bins"):
        result = connectivity.jackknife("magnitude_squared_coherence")
    assert np.isnan(result.standard_error[1]).all()
    assert np.isfinite(result.standard_error[0, :, 0, 1]).all()


def test_jackknife_refuses_weights_that_drop_a_unit_at_some_frequencies():
    rng = np.random.default_rng(10)
    coefficients = _coefficients(rng, (1, 5, 1, 8, 2))
    weights = np.ones((*coefficients.shape[:-1], 1))
    weights[0, 0, 0, :3] = 0  # trial 0 is dropped at three frequencies only
    connectivity = Connectivity(coefficients, observation_weights=weights)
    with pytest.raises(ValueError, match="at some frequencies only"):
        connectivity.jackknife("magnitude_squared_coherence")


@pytest.mark.parametrize(
    "measure",
    [
        "power",
        "magnitude_squared_coherence",
        "phase_locking_value",
        "debiased_squared_phase_lag_index",
        "debiased_squared_weighted_phase_lag_index",
        "pairwise_phase_consistency",
    ],
)
def test_drop_weights_keep_the_working_precision(measure):
    """complex64 coefficients give the dtype with drop weights as without."""
    rng = np.random.default_rng(11)
    coefficients = _coefficients(rng, (1, 6, 2, 8, 2)).astype(np.complex64)
    weights = np.ones((*coefficients.shape[:-1], 1), dtype=np.float32)
    weights[0, 0] = 0
    masked = getattr(
        Connectivity(coefficients, dtype=np.complex64, observation_weights=weights), measure
    )()
    plain = getattr(Connectivity(coefficients, dtype=np.complex64), measure)()
    assert masked.dtype == plain.dtype


@pytest.mark.parametrize(
    "measure",
    [
        "phase_lag_index",
        "weighted_phase_lag_index",
        "debiased_squared_phase_lag_index",
        "debiased_squared_weighted_phase_lag_index",
    ],
)
def test_phase_lag_moments_of_many_observations_match_their_removal(measure):
    """With enough observations every moment is reduced from one tile pass,
    the dropped observations zeroed in place."""
    rng = np.random.default_rng(12)
    coefficients = _coefficients(rng, (2, 30, 3, 8, 3))
    valid = np.ones((2, 30), dtype=bool)
    valid[0, :4] = False
    valid[1, 10:19] = False
    weights = _window_trial_weights(valid, coefficients.shape)
    connectivity = Connectivity(coefficients, observation_weights=weights)
    assert connectivity._kept_observations is not None
    value = getattr(connectivity, measure)()
    for window in range(2):
        kept = Connectivity(coefficients[[window]][:, valid[window]])
        np.testing.assert_allclose(
            value[window], getattr(kept, measure)()[0], rtol=1e-12, atol=1e-15
        )
