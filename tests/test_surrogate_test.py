"""Tests of Connectivity.surrogate_test and the surrogate p-value arithmetic.

The Monte Carlo tests put replicate datasets on the time-window axis, so one
call tests every replicate (the expectation keeps that axis). A rejection
rate's limit is five standard errors of the replicate mean, where the standard
error is that of the per-replicate rejection fraction (frequencies within one
replicate are correlated, so they are not counted as independent).
"""

import re
import warnings

import numpy as np
import pytest
from scipy.signal import lfilter

from spectral_connectivity import (
    Connectivity,
    DirectedOrientationWarning,
    Multitaper,
    SurrogateTestResult,
    Welch,
)
from spectral_connectivity.connectivity import _NON_MEASURE_METHODS
from spectral_connectivity.statistics import _SurrogateTally
from tests._backend_helpers import to_host

_ALPHA = 0.05
_N_SURROGATES = 99  # P(p <= 0.05) is exactly 5 / 100 for an exact test without ties


def _coefficients(rng, shape=(2, 4, 3, 16, 3)):
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def _ar1(rng, n_time, n_columns, n_signals, coefficient=0.5):
    """Independent AR(1) series, shape (n_time, n_columns, n_signals)."""
    noise = rng.standard_normal((n_time + 100, n_columns, n_signals))
    return lfilter([1.0], [1.0, -coefficient], noise, axis=0)[100:]


def _ar2(rng, n_time, n_columns, peak, radius, sampling_frequency):
    """AR(2) series peaked at ``peak`` Hz, shape (n_time, n_columns)."""
    a1 = 2 * radius * np.cos(2 * np.pi * peak / sampling_frequency)
    noise = rng.standard_normal((n_time + 200, n_columns))
    return lfilter([1.0], [1.0, -a1, radius**2], noise, axis=0)[200:]


def _replicates(time_series, n_trials, transform=Multitaper, **transform_kwargs):
    """Coefficients of independent datasets stacked on the time-window axis.

    ``time_series`` has shape ``(n_time, n_replicates * n_trials, n_signals)``;
    its single window is split into ``n_replicates`` windows of ``n_trials``.
    Returns the coefficients and the transform.
    """
    spectral = transform(time_series, **transform_kwargs)
    coefficients = to_host(spectral.fft())
    n_replicates = coefficients.shape[1] // n_trials
    return coefficients.reshape(n_replicates, n_trials, *coefficients.shape[2:]), spectral


def _first_replicates(connectivity, n_replicates):
    """``connectivity`` restricted to its first ``n_replicates`` time windows
    (all of them for ``None``), to bound the cost of the directed measures."""
    if n_replicates is None:
        return connectivity
    return Connectivity(
        connectivity.fourier_coefficients[:n_replicates],
        frequencies=connectivity.all_frequencies,
    )


def _rejection(p_value, select=np.s_[...]):
    """Per-replicate fraction of tests rejected at 5% (NaN p-values excluded)."""
    values = p_value[select]
    rejected = np.where(np.isnan(values), np.nan, values <= _ALPHA)
    return np.nanmean(rejected.reshape(values.shape[0], -1), axis=1)


def _assert_rate(per_replicate, expected, *, allowance=0.0, label=""):
    """The replicate mean is within 5 standard errors (plus ``allowance``)."""
    mean = per_replicate.mean()
    limit = 5 * per_replicate.std(ddof=1) / np.sqrt(per_replicate.size) + allowance
    assert abs(mean - expected) <= limit, f"{label}: {mean:.4f} vs {expected} +/- {limit:.4f}"
    return mean, limit


# --------------------------------------------------------------------------
# Construction of each surrogate
# --------------------------------------------------------------------------


def test_trial_shuffle_moves_whole_trials_of_each_signal():
    """Each signal's trials are permuted, the same across windows, tapers and
    frequencies; signal 0 keeps its order; different signals get different
    permutations."""
    rng = np.random.default_rng(0)
    coefficients = _coefficients(rng, (3, 6, 2, 8, 4))
    connectivity = Connectivity(coefficients)
    surrogate = to_host(
        connectivity._surrogate_coefficients("trial_shuffle", (1,), np.random.default_rng(1))
    )
    np.testing.assert_array_equal(surrogate[..., 0], coefficients[..., 0])
    orders = []
    for signal in range(1, 4):
        # Match each surrogate trial to the original trial it came from.
        order = [
            next(
                trial
                for trial in range(6)
                if np.array_equal(
                    surrogate[:, position, ..., signal], coefficients[:, trial, ..., signal]
                )
            )
            for position in range(6)
        ]
        assert sorted(order) == list(range(6))
        orders.append(order)
    assert len({tuple(order) for order in orders}) > 1


@pytest.mark.parametrize(
    ("independent", "unit_axes"),
    [(True, (1, 2)), (False, (1,))],
    ids=["per taper", "per trial"],
)
def test_phase_rotation_turns_each_unit_by_one_phase(independent, unit_axes):
    """Ratio to the data is exp(i phi) at every positive frequency of a unit and
    signal, exp(-i phi) at the negative ones, and the sign of cos(phi) at 0 Hz
    and Nyquist; units with correlated tapers share phi; the kept time-window
    axis shares it too. A real signal's coefficients stay conjugate-symmetric."""
    rng = np.random.default_rng(2)
    n_fft = 16
    series = rng.standard_normal((2, 5, 3, n_fft, 2))
    coefficients = np.fft.fft(series, axis=3)
    connectivity = Connectivity(coefficients, observations_are_independent=independent)
    assert connectivity._surrogate_unit_axes("phase_rotation", 99) == unit_axes
    surrogate = to_host(
        connectivity._surrogate_coefficients(
            "phase_rotation", unit_axes, np.random.default_rng(3)
        )
    )
    ratio = surrogate / coefficients
    phase = np.angle(ratio[..., 1:2, :])
    np.testing.assert_allclose(np.abs(ratio), 1.0, rtol=1e-12)
    # One phase across the positive frequencies and its conjugate at the negative ones.
    positive = ratio[..., 1 : n_fft // 2, :]
    negative = ratio[..., n_fft // 2 + 1 :, :]
    np.testing.assert_allclose(
        positive, np.broadcast_to(np.exp(1j * phase), positive.shape), atol=1e-12
    )
    np.testing.assert_allclose(
        negative, np.broadcast_to(np.exp(-1j * phase), negative.shape), atol=1e-12
    )
    expected_sign = np.where(np.cos(phase) < 0, -1.0, 1.0)
    np.testing.assert_allclose(ratio[..., [0], :], expected_sign, atol=1e-12)
    np.testing.assert_allclose(ratio[..., [n_fft // 2], :], expected_sign, atol=1e-12)
    # The surrogate is still the FFT of a real series.
    np.testing.assert_allclose(np.fft.ifft(surrogate, axis=3).imag, 0.0, atol=1e-12)
    # Shared across time windows (kept), and across tapers when they are correlated.
    np.testing.assert_allclose(phase[0], phase[1], atol=1e-12)
    taper_phases_differ = not np.allclose(phase[:, :, 0], phase[:, :, 1])
    assert taper_phases_differ == independent
    # Independent across trials and signals.
    assert not np.allclose(phase[:, 0], phase[:, 1])
    assert not np.allclose(phase[..., 0], phase[..., 1])


def test_conjugation_flips_whole_observations():
    """Each trial-taper observation is either kept or conjugated, every signal
    and frequency together, with both outcomes present."""
    rng = np.random.default_rng(4)
    coefficients = _coefficients(rng, (2, 8, 3, 8, 3))
    connectivity = Connectivity(coefficients)
    surrogate = to_host(
        connectivity._surrogate_coefficients("conjugation", (1, 2), np.random.default_rng(5))
    )
    kept = np.all(surrogate == coefficients, axis=(0, 3, 4))
    conjugated = np.all(surrogate == np.conj(coefficients), axis=(0, 3, 4))
    assert np.all(kept ^ conjugated)
    assert kept.any()
    assert conjugated.any()


def test_correlated_time_bins_share_the_draws_when_time_is_averaged():
    """With expectation over time and trials, independent time bins are units;
    correlated ones are not."""
    rng = np.random.default_rng(6)
    coefficients = _coefficients(rng, (4, 3, 2, 8, 2))
    independent = Connectivity(coefficients, expectation_type="time_trials_tapers")
    correlated = Connectivity(
        coefficients, expectation_type="time_trials_tapers", time_bins_are_independent=False
    )
    assert independent._surrogate_unit_axes("conjugation", 9) == (0, 1, 2)
    assert correlated._surrogate_unit_axes("conjugation", 9) == (1, 2)


# --------------------------------------------------------------------------
# p-value arithmetic
# --------------------------------------------------------------------------


def test_tally_matches_brute_force_p_values():
    """Element-wise and max-statistic p-values against stored surrogates, with
    untested elements, a NaN surrogate value, and families along axis 1."""
    rng = np.random.default_rng(7)
    statistic = rng.standard_normal((3, 5, 2))
    tested = rng.random(statistic.shape) > 0.2
    nulls = rng.standard_normal((40, *statistic.shape))
    nulls[3, 0, 0, 0] = np.nan
    tally = _SurrogateTally(statistic, tested, (1,), 40)
    for null in nulls:
        tally.add(null)
    p_value, fwer_p_value = tally.p_values()

    exceeds = (nulls >= statistic) | np.isnan(nulls)
    expected_p = (1 + exceeds.sum(0)) / 41
    family_values = np.where(tested, np.where(np.isnan(nulls), np.inf, nulls), -np.inf)
    maxima = family_values.max(axis=2)  # (40, 3, 2): over the family axis 1
    np.testing.assert_array_equal(tally.null_maxima, maxima)
    expected_fwer = (1 + (maxima[:, :, np.newaxis, :] >= statistic).sum(0)) / 41
    np.testing.assert_allclose(p_value, np.where(tested, expected_p, np.nan))
    np.testing.assert_allclose(fwer_p_value, np.where(tested, expected_fwer, np.nan))
    assert np.all(fwer_p_value[tested] >= p_value[tested])


def test_tally_counts_a_rounding_tie_as_at_least_the_estimate():
    """A surrogate a few ulp below the estimate is a tie, not a smaller value."""
    statistic = np.array([0.3, 2.0])
    tally = _SurrogateTally(statistic, np.ones(2, dtype=bool), (0,), 1)
    tally.add(statistic * (1 - 1e-9))
    p_value, fwer_p_value = tally.p_values()
    np.testing.assert_array_equal(p_value, [1.0, 1.0])
    np.testing.assert_array_equal(fwer_p_value, [1.0, 1.0])


# --------------------------------------------------------------------------
# The method
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def lagged_connectivity():
    """Signal 1 is signal 0 delayed by 3 samples, each plus half as much noise;
    signal 2 is independent. 10 trials, NW=2 (3 tapers), 1 Hz bins at 128 Hz."""
    rng = np.random.default_rng(8)
    leader = rng.standard_normal((131, 10))
    signals = np.stack([leader[3:], leader[:-3], rng.standard_normal((128, 10))], axis=-1)
    signals += 0.5 * rng.standard_normal(signals.shape)
    multitaper = Multitaper(signals, sampling_frequency=128, time_halfbandwidth_product=2)
    return Connectivity.from_transform(multitaper)


def test_surrogate_test_returns_the_estimate_and_p_values_on_its_grid(lagged_connectivity):
    """The estimate is the measure; p-values are multiples of 1/(n + 1) in
    (0, 1], NaN on the diagonal; fwer p-values are no smaller; a seed
    reproduces the result and a Generator is accepted."""
    result = lagged_connectivity.surrogate_test(
        "magnitude_squared_coherence", kind="trial_shuffle", n_surrogates=19, random_state=0
    )
    assert isinstance(result, SurrogateTestResult)
    np.testing.assert_array_equal(
        result.estimate, lagged_connectivity.magnitude_squared_coherence()
    )
    diagonal = np.eye(3, dtype=bool)
    assert np.isnan(result.p_value[..., diagonal]).all()
    tested = result.p_value[..., ~diagonal]
    np.testing.assert_allclose(tested * 20, np.round(tested * 20), atol=1e-12)
    assert tested.min() >= 1 / 20
    assert tested.max() <= 1
    assert np.all(result.fwer_p_value[..., ~diagonal] >= tested)
    assert result.null_maxima.shape == (19,)
    assert (result.n_surrogates, result.kind, result.alternative) == (
        19,
        "trial_shuffle",
        "greater",
    )
    assert result.fwer_axis == (0, 1, 2, 3)
    again = lagged_connectivity.surrogate_test(
        "magnitude_squared_coherence",
        kind="trial_shuffle",
        n_surrogates=19,
        random_state=np.random.default_rng(0),
    )
    np.testing.assert_array_equal(again.p_value, result.p_value)
    np.testing.assert_array_equal(again.null_maxima, result.null_maxima)


def test_coupled_pair_is_detected_and_the_independent_one_is_not(lagged_connectivity):
    """Power: the lagged pair is significant family-wise at nearly every tested
    frequency, and no pair with the independent signal is."""
    result = lagged_connectivity.surrogate_test(
        "magnitude_squared_coherence",
        kind="phase_rotation",
        half_bandwidth=2.0,
        n_surrogates=99,
        random_state=1,
    )
    in_band = slice(2, 63)
    assert np.mean(result.fwer_p_value[0, in_band, 0, 1] <= _ALPHA) > 0.9
    assert not (result.fwer_p_value[..., [0, 1], [2, 2]] <= _ALPHA).any()


def test_fwer_axis_sets_the_families(lagged_connectivity):
    """fwer_axis=(1, 2, 3) gives one family per time window: null_maxima has the
    kept axis; fwer_axis=() makes each element its own family, so the adjusted
    p-values are the element-wise ones."""
    per_window = lagged_connectivity.surrogate_test(
        "imaginary_coherence",
        kind="conjugation",
        n_surrogates=9,
        random_state=2,
        fwer_axis=(1, 2, -1),
    )
    assert per_window.fwer_axis == (1, 2, 3)
    assert per_window.null_maxima.shape == (9, 1)
    each = lagged_connectivity.surrogate_test(
        "imaginary_coherence", kind="conjugation", n_surrogates=9, random_state=2, fwer_axis=()
    )
    np.testing.assert_array_equal(each.fwer_p_value, each.p_value)
    np.testing.assert_array_equal(each.p_value, per_window.p_value)


@pytest.mark.parametrize("alternative", ["less", "two-sided"])
def test_alternative_sets_the_statistic(lagged_connectivity, alternative):
    """'less' ranks the negated values and 'two-sided' the magnitudes: equal to
    'greater' applied to -x or |x| through the tally."""
    kwargs = {"kind": "conjugation", "n_surrogates": 19, "random_state": 3}
    signed = lagged_connectivity.surrogate_test(
        "imaginary_coherency", alternative=alternative, **kwargs
    )
    estimate = signed.estimate
    statistic = -estimate if alternative == "less" else np.abs(estimate)
    # Rebuild the surrogates with the same random numbers.
    connectivity = lagged_connectivity
    rng = np.random.default_rng(3)
    tested = np.isfinite(estimate) & ~np.eye(3, dtype=bool)
    tally = _SurrogateTally(statistic, tested, (0, 1, 2, 3), 19)
    for _ in range(19):
        values = Connectivity(
            connectivity._surrogate_coefficients("conjugation", (1, 2), rng),
            frequencies=connectivity._frequencies,
        ).imaginary_coherency()
        tally.add(-values if alternative == "less" else np.abs(values))
    p_value, fwer_p_value = tally.p_values()
    np.testing.assert_array_equal(signed.p_value, p_value)
    np.testing.assert_array_equal(signed.fwer_p_value, fwer_p_value)


def test_phase_rotation_leaves_the_edge_frequencies_untested(lagged_connectivity):
    """Frequencies below half_bandwidth and above Nyquist minus it are NaN; all
    others are tested."""
    result = lagged_connectivity.surrogate_test(
        "pairwise_spectral_granger_prediction",
        kind="phase_rotation",
        half_bandwidth=2.0,
        n_surrogates=4,
        random_state=4,
    )
    frequencies = lagged_connectivity.frequencies
    edge = (frequencies < 2.0) | (frequencies > 62.0)
    assert edge.sum() == 4  # 0 and 1 Hz; 63 and 64 Hz
    off_diagonal = ~np.eye(3, dtype=bool)
    assert np.isnan(result.p_value[:, edge][..., off_diagonal]).all()
    assert np.isfinite(result.p_value[:, ~edge][..., off_diagonal]).all()
    assert np.isnan(result.fwer_p_value[:, edge][..., off_diagonal]).all()


def test_phase_slope_index_band_must_avoid_the_edges(lagged_connectivity):
    """A band clear of the edges is tested whole; one reaching into them is refused."""
    result = lagged_connectivity.surrogate_test(
        "phase_slope_index",
        kind="phase_rotation",
        half_bandwidth=2.0,
        n_surrogates=4,
        random_state=5,
        frequency_range=(10, 30),
        alternative="two-sided",
    )
    assert result.p_value.shape == (1, 3, 3)
    assert np.isfinite(result.p_value[0, ~np.eye(3, dtype=bool)]).all()
    with pytest.raises(
        ValueError, match=r"Pass frequency_range=\(low, high\) with low >= 2.0"
    ):
        lagged_connectivity.surrogate_test(
            "phase_slope_index", kind="phase_rotation", half_bandwidth=2.0, n_surrogates=4
        )


@pytest.mark.parametrize(
    ("measure", "kwargs"),
    [
        ("coherence_phase", {}),
        ("partial_coherence", {}),
        ("phase_locking_value", {}),
        ("pairwise_phase_consistency", {}),
        ("weighted_phase_lag_index", {}),
        ("directed_phase_lag_index", {}),
        ("debiased_squared_weighted_phase_lag_index", {}),
        ("corrected_imaginary_phase_locking_value", {}),
        ("subset_pairwise_spectral_granger_prediction", {"pairs": [(0, 1)]}),
        ("conditional_spectral_granger_prediction", {}),
        ("time_reversed_spectral_granger_prediction", {}),
        ("directed_transfer_function", {}),
        ("partial_directed_coherence", {}),
        ("generalized_partial_directed_coherence", {}),
        ("direct_directed_transfer_function", {}),
        ("directed_coherence", {}),
    ],
)
def test_every_pairwise_measure_runs_with_every_kind(lagged_connectivity, measure, kwargs):
    """Each real pairwise measure has its frequency axis where phase_rotation
    looks for it, and every kind accepts it."""
    for kind in ("trial_shuffle", "phase_rotation", "conjugation"):
        extra = {"half_bandwidth": 2.0} if kind == "phase_rotation" else {}
        result = lagged_connectivity.surrogate_test(
            measure, kind=kind, n_surrogates=2, random_state=6, **extra, **kwargs
        )
        assert result.p_value.shape == result.estimate.shape
        assert np.nanmin(result.p_value) >= 1 / 3


def test_surrogate_test_is_not_a_measure():
    """The registry and the wrapper must not offer surrogate_test as a measure."""
    assert "surrogate_test" in _NON_MEASURE_METHODS


def test_migration_notice_is_emitted_once_per_call(lagged_connectivity):
    """The directed-orientation notice comes from the estimate only, not from
    each surrogate."""
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always", DirectedOrientationWarning)
        lagged_connectivity.surrogate_test(
            "directed_transfer_function", kind="trial_shuffle", n_surrogates=3, random_state=0
        )
    assert sum(issubclass(item.category, DirectedOrientationWarning) for item in record) == 1


# --------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_connectivity():
    return Connectivity(_coefficients(np.random.default_rng(9), (1, 4, 3, 16, 2)))


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"kind": "shuffle"}, ValueError, "kind must be one of 'trial_shuffle'"),
        (
            {"kind": "trial_shuffle", "n_surrogates": 0},
            ValueError,
            "n_surrogates must be a positive integer",
        ),
        ({"kind": "trial_shuffle", "n_surrogates": 9.5}, ValueError, "n_surrogates must be"),
        (
            {"kind": "trial_shuffle", "alternative": "larger"},
            ValueError,
            "alternative must be one of",
        ),
        ({"kind": "phase_rotation"}, ValueError, "needs half_bandwidth"),
        (
            {"kind": "phase_rotation", "half_bandwidth": -1.0},
            ValueError,
            "needs half_bandwidth",
        ),
        ({"kind": "phase_rotation", "half_bandwidth": 0.6}, ValueError, "tests no frequency"),
        (
            {"kind": "conjugation", "half_bandwidth": 2.0},
            ValueError,
            "applies only to kind='phase_rotation'",
        ),
        (
            {"kind": "conjugation", "fwer_axis": 4},
            ValueError,
            "fwer_axis 4 is out of bounds",
        ),
        ({"kind": "conjugation", "fwer_axis": (1, 1)}, ValueError, "more than once"),
        ({"kind": "conjugation", "fwer_axis": 1.0}, TypeError, "fwer_axis must be an int"),
    ],
)
def test_invalid_arguments_are_refused(small_connectivity, kwargs, error, match):
    with pytest.raises(error, match=match):
        small_connectivity.surrogate_test("magnitude_squared_coherence", **kwargs)


@pytest.mark.parametrize(
    ("method", "kwargs", "error", "match"),
    [
        ("power", {}, ValueError, "cannot test 'power': it describes one signal"),
        ("delay", {}, ValueError, "cannot test 'delay'"),
        ("coherency", {}, TypeError, "requires a real-valued measure; 'coherency' is complex"),
        ("canonical_coherence", {"group_labels": [0, 1]}, TypeError, "tuple-valued"),
        (
            "coherence",
            {},
            ValueError,
            "is not a connectivity measure surrogate_test can compute",
        ),
        ("surrogate_test", {}, ValueError, "is not a connectivity measure surrogate_test"),
        (3, {}, TypeError, "method must be a measure name"),
    ],
)
def test_measures_it_cannot_test_are_refused(small_connectivity, method, kwargs, error, match):
    with pytest.raises(error, match=re.escape(match) if "(" not in match else match):
        small_connectivity.surrogate_test(method, kind="conjugation", **kwargs)


def test_trial_shuffle_needs_an_expectation_over_several_trials():
    rng = np.random.default_rng(10)
    with pytest.raises(ValueError, match="expectation must average trials"):
        Connectivity(_coefficients(rng), expectation_type="tapers").surrogate_test(
            "magnitude_squared_coherence", kind="trial_shuffle"
        )
    with pytest.raises(ValueError, match="needs at least 2 trials to shuffle; got 1"):
        Connectivity(_coefficients(rng, (1, 1, 3, 16, 2))).surrogate_test(
            "magnitude_squared_coherence", kind="trial_shuffle"
        )


def test_trial_shuffle_refuses_weights_that_differ_between_trials():
    rng = np.random.default_rng(11)
    coefficients = _coefficients(rng, (1, 4, 2, 8, 2))
    weights = np.ones((1, 4, 2, 8, 1))
    weights[:, 0] = 2.0
    with pytest.raises(ValueError, match="observation_weights differ between trials"):
        Connectivity(coefficients, observation_weights=weights).surrogate_test(
            "magnitude_squared_coherence", kind="trial_shuffle"
        )
    # Weights equal across trials are kept with every trial.
    weights = np.broadcast_to(rng.uniform(0.5, 1.5, (1, 1, 2, 8, 1)), (1, 4, 2, 8, 1))
    Connectivity(coefficients, observation_weights=weights).surrogate_test(
        "magnitude_squared_coherence", kind="trial_shuffle", n_surrogates=2
    )


def test_randomizing_kinds_need_two_independent_observations():
    """One trial with correlated tapers leaves one unit; the message says why."""
    rng = np.random.default_rng(12)
    connectivity = Connectivity(
        _coefficients(rng, (1, 1, 5, 16, 2)), observations_are_independent=False
    )
    with pytest.raises(ValueError, match="its tapers are correlated") as excinfo:
        connectivity.surrogate_test("imaginary_coherence", kind="conjugation")
    assert "It needs at least 2" in str(excinfo.value)
    with pytest.raises(ValueError, match="has none"):
        Connectivity(
            _coefficients(rng), expectation_type="time", time_bins_are_independent=False
        ).surrogate_test("imaginary_coherence", kind="conjugation")


def test_phase_rotation_refuses_one_sided_coefficients():
    rng = np.random.default_rng(13)
    connectivity = Connectivity(_coefficients(rng, (1, 4, 3, 9, 2)), is_one_sided=True)
    with pytest.raises(ValueError, match="needs two-sided coefficients"):
        connectivity.surrogate_test(
            "magnitude_squared_coherence", kind="phase_rotation", half_bandwidth=0.05
        )


def test_few_distinct_surrogates_warn():
    """3 trials align in 3! = 6 ways; 4 independent observations give 2**3 = 8
    distinct conjugations of a conjugation-symmetric measure."""
    rng = np.random.default_rng(14)
    with pytest.warns(UserWarning, match="only 6 ways, fewer than n_surrogates=9"):
        Connectivity(_coefficients(rng, (1, 3, 1, 16, 2))).surrogate_test(
            "magnitude_squared_coherence", kind="trial_shuffle", n_surrogates=9
        )
    with pytest.warns(UserWarning, match="give only 8 distinct surrogates"):
        Connectivity(_coefficients(rng, (1, 4, 1, 16, 2))).surrogate_test(
            "imaginary_coherence", kind="conjugation", n_surrogates=9
        )


# --------------------------------------------------------------------------
# Monte Carlo: size, power, and family-wise error
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def independent_replicates():
    """300 replicate datasets of 3 independent broadband AR(1) signals, 10
    trials, 1 s at 64 Hz, NW=2 (3 tapers, half bandwidth 2 Hz)."""
    rng = np.random.default_rng(15)
    n_replicates, n_trials = 300, 10
    series = _ar1(rng, 64, n_replicates * n_trials, 3)
    coefficients, _ = _replicates(
        series, n_trials, sampling_frequency=64, time_halfbandwidth_product=2
    )
    return Connectivity(
        coefficients, frequencies=np.fft.fftfreq(coefficients.shape[3], 1 / 64)
    )


@pytest.mark.parametrize(
    "measure", ["magnitude_squared_coherence", "pairwise_spectral_granger_prediction"]
)
def test_trial_shuffle_holds_its_size_at_every_frequency(independent_replicates, measure):
    """Independent signals: the trial-shuffle test is exact (iid trials), at
    every frequency including 0 Hz and Nyquist."""
    connectivity = _first_replicates(
        independent_replicates, 100 if "granger" in measure else None
    )
    result = connectivity.surrogate_test(
        measure, kind="trial_shuffle", n_surrogates=_N_SURROGATES, random_state=16
    )
    off_diagonal = ~np.eye(3, dtype=bool)
    _assert_rate(
        _rejection(result.p_value[..., off_diagonal]), _ALPHA, label="all frequencies"
    )
    _assert_rate(
        _rejection(result.p_value[:, [0, -1]][..., off_diagonal]),
        _ALPHA,
        label="0 Hz and Nyquist",
    )


@pytest.mark.parametrize("kind", ["trial_shuffle", "phase_rotation", "conjugation"])
def test_family_wise_error_is_held_under_the_global_null(independent_replicates, kind):
    """One family per replicate (every frequency and pair): some element is
    rejected in at most 5% of replicates, while uncorrected p-values reject
    somewhere in most of them."""
    extra = {"half_bandwidth": 2.0} if kind == "phase_rotation" else {}
    result = independent_replicates.surrogate_test(
        "magnitude_squared_coherence",
        kind=kind,
        n_surrogates=_N_SURROGATES,
        random_state=17,
        fwer_axis=(1, 2, 3),
        **extra,
    )
    any_family = np.nanmin(
        result.fwer_p_value.reshape(result.fwer_p_value.shape[0], -1), axis=1
    )
    any_raw = np.nanmin(result.p_value.reshape(result.p_value.shape[0], -1), axis=1)
    # Exact for trial_shuffle; the other kinds' surrogates are exact only for
    # circular coefficients, so allow 2 points for that. The smallest adjusted
    # p-value of a family is itself a valid p-value, so check several levels.
    allowance = 0.0 if kind == "trial_shuffle" else 0.02
    for level in (_ALPHA, 0.2, 0.5):
        family_rejected = (any_family <= level).astype(float)
        _assert_rate(family_rejected, level, allowance=allowance, label=f"{kind} {level}")
    assert np.mean(any_raw <= _ALPHA) > 0.8  # the correction is needed
    assert result.null_maxima.shape == (_N_SURROGATES, result.estimate.shape[0])


@pytest.mark.parametrize("n_trials", [1, 5])
@pytest.mark.parametrize(
    "measure", ["magnitude_squared_coherence", "pairwise_spectral_granger_prediction"]
)
def test_phase_rotation_holds_its_size_away_from_the_edges(n_trials, measure):
    """Independent signals, 7 tapers per trial: about 5% at every tested frequency."""
    rng = np.random.default_rng(18 + n_trials)
    n_replicates = (200 if "granger" in measure else 400) // n_trials
    series = _ar1(rng, 128, n_replicates * n_trials, 2)
    coefficients, multitaper = _replicates(
        series, n_trials, sampling_frequency=128, time_halfbandwidth_product=4
    )
    connectivity = Connectivity(coefficients, frequencies=np.fft.fftfreq(128, 1 / 128))
    result = connectivity.surrogate_test(
        measure,
        kind="phase_rotation",
        half_bandwidth=multitaper.frequency_resolution / 2,
        n_surrogates=_N_SURROGATES,
        random_state=20,
    )
    # Directed measures carry a small excess just past the edges (see Notes).
    allowance = 0.01 if measure == "pairwise_spectral_granger_prediction" else 0.0
    _assert_rate(_rejection(result.p_value[..., 0, 1]), _ALPHA, allowance=allowance)


@pytest.fixture(scope="module")
def mixed_replicates():
    """Instantaneous mixing of two independent AR(2) sources (cross-weights
    0.7): real coherency, nonzero pairwise Granger in both directions.
    200 replicates of 10 trials, 1 s at 128 Hz, NW=2."""
    rng = np.random.default_rng(21)
    n_replicates, n_trials = 200, 10
    first = _ar2(rng, 128, n_replicates * n_trials, 20, 0.9, 128)
    second = _ar2(rng, 128, n_replicates * n_trials, 40, 0.8, 128)
    series = np.stack([first + 0.7 * second, second + 0.7 * first], axis=-1)
    coefficients, _ = _replicates(
        series, n_trials, sampling_frequency=128, time_halfbandwidth_product=2
    )
    return Connectivity(coefficients, frequencies=np.fft.fftfreq(128, 1 / 128))


@pytest.mark.parametrize(
    ("measure", "allowance"),
    [("imaginary_coherence", 0.0), ("pairwise_spectral_granger_prediction", 0.02)],
)
def test_conjugation_holds_its_size_under_instantaneous_mixing(
    mixed_replicates, measure, allowance
):
    """The null of real coherency holds, so conjugation rejects about 5%; the
    trial shuffle, whose null is independence, finds Granger 'causality' in
    most of the same data."""
    connectivity = _first_replicates(mixed_replicates, 100 if "granger" in measure else None)
    result = connectivity.surrogate_test(
        measure, kind="conjugation", n_surrogates=_N_SURROGATES, random_state=22
    )
    # Granger's null is approximate (Granger is a nonlinear function of the
    # whole spectrum); allow 2 points.
    rate, limit = _assert_rate(
        _rejection(result.p_value[..., 0, 1]), _ALPHA, allowance=allowance
    )
    if measure == "pairwise_spectral_granger_prediction":
        shuffled = connectivity.surrogate_test(
            measure, kind="trial_shuffle", n_surrogates=_N_SURROGATES, random_state=23
        )
        assert _rejection(shuffled.p_value[..., 0, 1]).mean() > rate + 10 * limit


def test_conjugation_detects_a_lagged_interaction():
    """Signal 1 receives signal 0 three samples late: conjugation rejects the
    imaginary coherence at its spectral peak in most replicates."""
    rng = np.random.default_rng(24)
    n_replicates, n_trials = 100, 10
    first = _ar2(rng, 131, n_replicates * n_trials, 20, 0.9, 128)
    second = _ar2(rng, 128, n_replicates * n_trials, 40, 0.8, 128) + 0.5 * first[:-3]
    series = np.stack([first[3:], second], axis=-1)
    coefficients, _ = _replicates(
        series, n_trials, sampling_frequency=128, time_halfbandwidth_product=2
    )
    result = Connectivity(
        coefficients, frequencies=np.fft.fftfreq(128, 1 / 128)
    ).surrogate_test(
        "imaginary_coherence", kind="conjugation", n_surrogates=_N_SURROGATES, random_state=25
    )
    # Measured 0.86.
    assert _rejection(result.p_value[:, 15:26, 0, 1]).mean() > 0.75


def test_correlated_tapers_are_randomized_as_whole_trials():
    """Welch segments overlapping by 75% are correlated: one phase per trial
    holds the size, one per segment (forcing observations_are_independent)
    rejects independent signals far more often."""
    rng = np.random.default_rng(26)
    n_replicates, n_trials = 300, 5
    series = _ar1(rng, 128, n_replicates * n_trials, 2)
    coefficients, welch = _replicates(
        series,
        n_trials,
        transform=Welch,
        sampling_frequency=128,
        segment_duration=0.25,
        segment_overlap=0.75,
    )
    assert not welch.observations_are_independent
    frequencies = np.fft.fftfreq(coefficients.shape[3], 1 / 128)
    rates = {}
    for independent in (False, True):
        result = Connectivity(
            coefficients, frequencies=frequencies, observations_are_independent=independent
        ).surrogate_test(
            "magnitude_squared_coherence",
            kind="phase_rotation",
            half_bandwidth=8.0,  # the Hann main lobe's half-width, 2 / 0.25 s
            n_surrogates=_N_SURROGATES,
            random_state=27,
        )
        rates[independent] = _rejection(result.p_value[..., 0, 1])
    _, limit = _assert_rate(rates[False], _ALPHA, label="one phase per trial")
    assert rates[True].mean() > _ALPHA + 5 * limit


def test_a_shared_evoked_response_is_in_the_shuffle_null_but_not_the_rotation_null():
    """The same 10 Hz evoked burst, at zero lag, in every trial of two otherwise
    independent signals: the trial shuffle keeps it in every surrogate and holds
    its size, conjugation keeps it too (zero lag, real coherency), and phase
    rotation, which destroys stimulus-locked timing, rejects it."""
    rng = np.random.default_rng(28)
    n_replicates, n_trials, sampling_frequency = 200, 10, 128
    time = np.arange(128) / sampling_frequency - 0.5
    evoked = 3 * np.exp(-(time**2) / (2 * 0.05**2)) * np.cos(2 * np.pi * 10 * time)
    series = np.stack(
        [
            _ar2(rng, 128, n_replicates * n_trials, 20, 0.9, sampling_frequency),
            _ar2(rng, 128, n_replicates * n_trials, 40, 0.8, sampling_frequency),
        ],
        axis=-1,
    )
    series += evoked[:, np.newaxis, np.newaxis]
    coefficients, _ = _replicates(
        series, n_trials, sampling_frequency=sampling_frequency, time_halfbandwidth_product=2
    )
    connectivity = Connectivity(coefficients, frequencies=np.fft.fftfreq(128, 1 / 128))
    evoked_band = np.s_[:, 5:16, 0, 1]
    rates = {}
    for kind in ("trial_shuffle", "conjugation", "phase_rotation"):
        extra = {"half_bandwidth": 2.0} if kind == "phase_rotation" else {}
        result = connectivity.surrogate_test(
            "magnitude_squared_coherence",
            kind=kind,
            n_surrogates=_N_SURROGATES,
            random_state=29,
            **extra,
        )
        rates[kind] = _rejection(result.p_value[evoked_band])
    _, limit = _assert_rate(rates["trial_shuffle"], _ALPHA, label="trial_shuffle")
    _assert_rate(rates["conjugation"], _ALPHA, allowance=0.02, label="conjugation")
    assert rates["phase_rotation"].mean() > _ALPHA + 5 * limit
