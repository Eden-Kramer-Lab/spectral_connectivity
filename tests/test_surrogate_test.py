"""Tests of Connectivity.surrogate_test: the surrogates, the p-value
arithmetic, the method's results and its refusals.

The Monte Carlo tests (size, power and family-wise error against known
nulls) are in ``tests/test_surrogate_oracles.py``.
"""

import re
import warnings

import numpy as np
import pytest

from spectral_connectivity import (
    Connectivity,
    DirectedOrientationWarning,
    Multitaper,
    SurrogateTestResult,
)
from spectral_connectivity.statistics import _SurrogateTally
from tests._backend_helpers import to_host

_ALPHA = 0.05


def _coefficients(rng, shape=(2, 4, 3, 16, 3)):
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


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


@pytest.mark.parametrize("independent", [True, False])
def test_phase_rotation_turns_each_trial_by_one_phase(independent):
    """Ratio to the data is exp(i phi) at every positive frequency of a trial and
    signal, exp(-i phi) at the negative ones, and the sign of cos(phi) at 0 Hz
    and Nyquist; every taper of a trial shares phi, whether or not the tapers
    are independent; the kept time-window axis shares it too. A real signal's
    coefficients stay conjugate-symmetric."""
    rng = np.random.default_rng(2)
    n_fft = 16
    series = rng.standard_normal((2, 5, 3, n_fft, 2))
    coefficients = np.fft.fft(series, axis=3)
    connectivity = Connectivity(coefficients, observations_are_independent=independent)
    assert connectivity._surrogate_unit_axes("phase_rotation", 99) == (1,)
    surrogate = to_host(
        connectivity._surrogate_coefficients("phase_rotation", (1,), np.random.default_rng(3))
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
    # Shared across time windows (kept) and the tapers of a trial.
    np.testing.assert_allclose(phase[0], phase[1], atol=1e-12)
    np.testing.assert_allclose(phase[:, :, 0], phase[:, :, 2], atol=1e-12)
    # Independent across trials and signals.
    assert not np.allclose(phase[:, 0], phase[:, 1])
    assert not np.allclose(phase[..., 0], phase[..., 1])


def test_conjugation_flips_whole_trials():
    """Each trial is either kept or conjugated, every taper, signal and
    frequency together, with both outcomes present."""
    rng = np.random.default_rng(4)
    coefficients = _coefficients(rng, (2, 8, 3, 8, 3))
    connectivity = Connectivity(coefficients)
    assert connectivity._surrogate_unit_axes("conjugation", 99) == (1,)
    surrogate = to_host(
        connectivity._surrogate_coefficients("conjugation", (1,), np.random.default_rng(5))
    )
    kept = np.all(surrogate == coefficients, axis=(0, 2, 3, 4))
    conjugated = np.all(surrogate == np.conj(coefficients), axis=(0, 2, 3, 4))
    assert np.all(kept ^ conjugated)
    assert kept.any()
    assert conjugated.any()


def test_correlated_time_bins_share_the_draws_when_time_is_averaged():
    """With expectation over time and trials, independent time bins are units
    besides the trials; correlated ones are not."""
    rng = np.random.default_rng(6)
    coefficients = _coefficients(rng, (4, 3, 2, 8, 2))
    independent = Connectivity(coefficients, expectation_type="time_trials_tapers")
    correlated = Connectivity(
        coefficients, expectation_type="time_trials_tapers", time_bins_are_independent=False
    )
    assert independent._surrogate_unit_axes("conjugation", 4) == (0, 1)
    assert correlated._surrogate_unit_axes("conjugation", 4) == (1,)


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
    reproduces the result, a Generator seeded alike gives the same surrogates,
    and another seed different ones."""
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
    other = lagged_connectivity.surrogate_test(
        "magnitude_squared_coherence", kind="trial_shuffle", n_surrogates=19, random_state=1
    )
    assert not np.array_equal(other.null_maxima, result.null_maxima)


def test_coupled_pair_is_detected_and_the_independent_one_is_not(lagged_connectivity):
    """Power: the lagged pair is significant family-wise at nearly every tested
    frequency, and no pair with the independent signal is."""
    result = lagged_connectivity.surrogate_test(
        "magnitude_squared_coherence", kind="phase_rotation", n_surrogates=99, random_state=1
    )
    in_band = slice(2, 63)
    assert np.mean(result.fwer_p_value[0, in_band, 0, 1] <= _ALPHA) > 0.9
    assert not (result.fwer_p_value[..., [0, 1], [2, 2]] <= _ALPHA).any()


def test_fwer_axis_sets_the_families(lagged_connectivity):
    """fwer_axis=(1, 2, 3) gives one family per time window, so null_maxima
    has the kept axis, and the stored axes are normalized; the element-wise
    p-values do not depend on the families."""
    per_window = lagged_connectivity.surrogate_test(
        "imaginary_coherence",
        kind="conjugation",
        n_surrogates=9,
        random_state=2,
        fwer_axis=(1, 2, -1),
    )
    assert per_window.fwer_axis == (1, 2, 3)
    assert per_window.null_maxima.shape == (9, 1)
    pooled = lagged_connectivity.surrogate_test(
        "imaginary_coherence", kind="conjugation", n_surrogates=9, random_state=2
    )
    assert pooled.fwer_axis == (0, 1, 2, 3)
    np.testing.assert_array_equal(pooled.p_value, per_window.p_value)


def test_empty_fwer_axis_is_refused_for_p_value(lagged_connectivity):
    """fwer_axis=() would store every surrogate's whole result to reproduce
    p_value; the message points there."""
    with pytest.raises(ValueError, match=r"fwer_axis=\(\) makes every element its own family"):
        lagged_connectivity.surrogate_test(
            "imaginary_coherence", kind="conjugation", n_surrogates=9, fwer_axis=()
        )


class _TransformedCoherency(Connectivity):
    """Measures of the imaginary coherency negated and in magnitude, to test
    ``alternative`` against ``"greater"`` applied to them."""

    def negated_imaginary_coherency(self):
        return -self.imaginary_coherency()

    def absolute_imaginary_coherency(self):
        return np.abs(self.imaginary_coherency())


@pytest.mark.parametrize(
    ("alternative", "transformed"),
    [("less", "negated_imaginary_coherency"), ("two-sided", "absolute_imaginary_coherency")],
)
def test_alternative_sets_the_statistic(lagged_connectivity, alternative, transformed):
    """'less' ranks the negated values and 'two-sided' the magnitudes: the
    same p-values as 'greater' for a measure returning -x or |x|, with the
    same surrogates (same random_state)."""
    connectivity = _TransformedCoherency(
        lagged_connectivity.fourier_coefficients,
        frequencies=lagged_connectivity.all_frequencies,
    )
    kwargs = {"kind": "conjugation", "n_surrogates": 19, "random_state": 3}
    signed = connectivity.surrogate_test(
        "imaginary_coherency", alternative=alternative, **kwargs
    )
    reference = connectivity.surrogate_test(transformed, alternative="greater", **kwargs)
    np.testing.assert_array_equal(signed.p_value, reference.p_value)
    np.testing.assert_array_equal(signed.fwer_p_value, reference.fwer_p_value)
    np.testing.assert_array_equal(signed.null_maxima, reference.null_maxima)


def test_phase_rotation_leaves_the_edge_frequencies_untested(lagged_connectivity):
    """Frequencies below half_bandwidth and above Nyquist minus it are NaN; all
    others are tested."""
    result = lagged_connectivity.surrogate_test(
        "pairwise_spectral_granger_prediction",
        kind="phase_rotation",
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


@pytest.mark.parametrize("is_one_sided", [False, True])
@pytest.mark.parametrize("n_fft", [16, 15])
def test_conjugation_leaves_0_hz_and_nyquist_untested(n_fft, is_one_sided):
    """The coefficients at 0 Hz and (for an even FFT length) Nyquist are real,
    so conjugation leaves them unchanged: those frequencies are NaN and in no
    family's maximum (with one family per frequency, theirs is -inf); every
    other frequency, the last of an odd FFT length among them, is tested. A
    band-summed measure is tested whole. One-sided input does not record the
    FFT length, so there the real coefficients themselves mark Nyquist."""
    rng = np.random.default_rng(30)
    time_series = rng.standard_normal((n_fft, 12, 2))
    if is_one_sided:
        # (n_fft, trial, signal) -> (time window, trial, taper, frequency, signal)
        coefficients = np.fft.rfft(time_series, axis=0).transpose(1, 0, 2)[None, :, None]
        connectivity = Connectivity(coefficients, is_one_sided=True)
    else:
        multitaper = Multitaper(
            time_series, sampling_frequency=n_fft, time_halfbandwidth_product=1
        )
        connectivity = Connectivity.from_transform(multitaper)
    result = connectivity.surrogate_test(
        "magnitude_squared_coherence",
        kind="conjugation",
        n_surrogates=9,
        random_state=0,
        fwer_axis=(2, 3),
    )
    untested = np.zeros(len(connectivity.frequencies), dtype=bool)
    untested[0] = True
    untested[-1] = n_fft % 2 == 0
    for p_value in (result.p_value, result.fwer_p_value):
        assert np.isnan(p_value[0, untested, 0, 1]).all()
        assert np.isfinite(p_value[0, ~untested, 0, 1]).all()
    assert np.all(result.null_maxima[:, 0, untested] == -np.inf)
    assert np.isfinite(result.null_maxima[:, 0, ~untested]).all()
    psi = connectivity.surrogate_test(
        "phase_slope_index", kind="conjugation", n_surrogates=9, random_state=0
    )
    assert np.isfinite(psi.p_value[0, 0, 1])


def test_phase_slope_index_band_must_avoid_the_edges(lagged_connectivity):
    """A band clear of the edges is tested whole; one reaching into them is refused."""
    result = lagged_connectivity.surrogate_test(
        "phase_slope_index",
        kind="phase_rotation",
        n_surrogates=4,
        random_state=5,
        frequency_range=(10, 30),
    )
    assert result.p_value.shape == (1, 3, 3)
    assert np.isfinite(result.p_value[0, ~np.eye(3, dtype=bool)]).all()
    with pytest.raises(
        ValueError, match=r"Pass frequency_range=\(low, high\) with low >= 2.0"
    ):
        lagged_connectivity.surrogate_test(
            "phase_slope_index", kind="phase_rotation", n_surrogates=4
        )


@pytest.mark.parametrize(
    ("measure", "kwargs"),
    [
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
@pytest.mark.parametrize("kind", ["trial_shuffle", "phase_rotation", "conjugation"])
def test_every_pairwise_measure_runs_with_every_kind(
    lagged_connectivity, measure, kwargs, kind
):
    """Each real pairwise measure has its frequency axis where phase_rotation
    looks for it, and every kind accepts it. The diagonal is never tested,
    including where the measure is finite there (1 for the phase-locking
    value, near 1 for the transfer-function measures, 0 for the lag indices),
    so it cannot enter a family maximum."""
    result = lagged_connectivity.surrogate_test(
        measure, kind=kind, n_surrogates=2, random_state=6, **kwargs
    )
    assert result.p_value.shape == result.estimate.shape
    assert np.nanmin(result.p_value) >= 1 / 3
    diagonal = np.eye(result.estimate.shape[-1], dtype=bool)
    assert np.isnan(result.p_value[..., diagonal]).all()
    assert np.isnan(result.fwer_p_value[..., diagonal]).all()


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
    """12 trials: enough for 999 distinct conjugation surrogates."""
    return Connectivity(_coefficients(np.random.default_rng(9), (1, 12, 3, 16, 2)))


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        (
            {"kind": "shuffle"},
            ValueError,
            "kind must be one of 'trial_shuffle'.*'conjugation': is there more than zero-lag",
        ),
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
        ({"kind": "phase_rotation"}, ValueError, "does not know it"),
        (
            {"kind": "phase_rotation", "half_bandwidth": -1.0},
            ValueError,
            "half_bandwidth must be a positive number",
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
            (
                r"fwer_axis 4 is out of bounds for magnitude_squared_coherence's result, "
                r"axes \(time, frequency, source, target\)"
            ),
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
        (
            "coherence_phase",
            {},
            ValueError,
            "cannot test 'coherence_phase': a phase is circular",
        ),
        (
            "coherency",
            {},
            TypeError,
            "'coherency' is complex. Test a real measure of it: magnitude_squared_coherence",
        ),
        ("canonical_coherence", {"group_labels": [0, 1]}, TypeError, "the group measure"),
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


@pytest.mark.parametrize("kind", ["phase_rotation", "conjugation"])
def test_randomizing_kinds_refuse_a_single_unit(kind):
    """One trial and no averaged time windows is one unit, refused whether or
    not its tapers are independent; the message gives the measured failure of
    per-taper randomization and the remedies. Averaging independent windows of
    the one trial makes each window a unit; correlated windows are not."""
    rng = np.random.default_rng(12)
    for independent in (True, False):
        connectivity = Connectivity(
            _coefficients(rng, (1, 1, 5, 16, 2)),
            observations_are_independent=independent,
            half_bandwidth=2.0,
        )
        with pytest.raises(ValueError, match="this Connectivity has 1 ") as excinfo:
            connectivity.surrogate_test("imaginary_coherence", kind=kind)
        message = str(excinfo.value)
        assert "57% (phase rotation) and 28% (conjugation)" in message
        assert "an expectation_type that includes 'time'" in message
    windows = _coefficients(rng, (12, 1, 5, 16, 2))
    averaged = Connectivity(windows, expectation_type="time_tapers", half_bandwidth=2.0)
    assert averaged._surrogate_unit_axes(kind, 99) == (0,)
    correlated = Connectivity(
        windows,
        expectation_type="time_tapers",
        half_bandwidth=2.0,
        time_bins_are_independent=False,
    )
    with pytest.raises(ValueError, match="time_bins_are_independent is False"):
        correlated.surrogate_test("imaginary_coherence", kind=kind)


def test_phase_rotation_refuses_one_sided_coefficients():
    rng = np.random.default_rng(13)
    connectivity = Connectivity(_coefficients(rng, (1, 4, 3, 9, 2)), is_one_sided=True)
    with pytest.raises(ValueError, match="needs two-sided coefficients"):
        connectivity.surrogate_test(
            "magnitude_squared_coherence", kind="phase_rotation", half_bandwidth=0.05
        )


def test_few_distinct_surrogates_warn():
    """3 trials align in 3! = 6 ways; 4 trials give 2**3 = 8 distinct
    conjugations of a conjugation-symmetric measure, however many tapers."""
    rng = np.random.default_rng(14)
    with pytest.warns(UserWarning, match=re.escape("3! = 6 ways (fewer than n_surrogates=9)")):
        Connectivity(_coefficients(rng, (1, 3, 1, 16, 2))).surrogate_test(
            "magnitude_squared_coherence", kind="trial_shuffle", n_surrogates=9
        )
    with pytest.warns(UserWarning, match=re.escape("only 2**3 = 8 distinct surrogates")):
        Connectivity(_coefficients(rng, (1, 4, 5, 16, 2))).surrogate_test(
            "imaginary_coherence", kind="conjugation", n_surrogates=9
        )


def test_phase_rotation_uses_the_half_bandwidth_from_the_transform(lagged_connectivity):
    """from_transform records the Multitaper's half bandwidth (NW / T = 2 Hz
    here), which phase_rotation uses unless one is passed; reassigning
    coefficients of another FFT length forgets it."""
    assert lagged_connectivity.half_bandwidth == 2.0
    recorded = lagged_connectivity.surrogate_test(
        "magnitude_squared_coherence", kind="phase_rotation", n_surrogates=4, random_state=0
    )
    explicit = lagged_connectivity.surrogate_test(
        "magnitude_squared_coherence",
        kind="phase_rotation",
        half_bandwidth=2.0,
        n_surrogates=4,
        random_state=0,
    )
    np.testing.assert_array_equal(recorded.p_value, explicit.p_value)
    wider = lagged_connectivity.surrogate_test(
        "magnitude_squared_coherence",
        kind="phase_rotation",
        half_bandwidth=5.0,
        n_surrogates=4,
        random_state=0,
    )
    assert np.isnan(wider.p_value[0, :5, 0, 1]).all()
    assert np.isfinite(wider.p_value[0, 5, 0, 1])

    connectivity = Connectivity(
        lagged_connectivity.fourier_coefficients,
        frequencies=lagged_connectivity.all_frequencies,
        half_bandwidth=2.0,
    )
    with pytest.warns(UserWarning, match="half_bandwidth were reset"):
        connectivity.fourier_coefficients = _coefficients(
            np.random.default_rng(0), (1, 4, 3, 8, 3)
        )
    assert connectivity.half_bandwidth is None


@pytest.mark.parametrize(
    ("value", "error", "match"),
    [
        (0.0, ValueError, "half_bandwidth must be a positive number"),
        (-1.0, ValueError, "half_bandwidth must be a positive number"),
        (np.inf, ValueError, "half_bandwidth must be a positive number"),
        (True, TypeError, "half_bandwidth must be a number"),
        ("2", TypeError, "half_bandwidth must be a number"),
        (np.array([2.0]), TypeError, "half_bandwidth must be a number"),
    ],
)
def test_connectivity_refuses_an_invalid_half_bandwidth(value, error, match):
    """Validated like a sampling frequency: a real scalar (a 0-d array too)."""
    with pytest.raises(error, match=match):
        Connectivity(_coefficients(np.random.default_rng(0)), half_bandwidth=value)
    accepted = Connectivity(
        _coefficients(np.random.default_rng(0)), half_bandwidth=np.array(2.0)
    )
    assert accepted.half_bandwidth == 2.0
    assert type(accepted.half_bandwidth) is float


def test_resampled_replicates_keep_the_subclass_and_settings(lagged_connectivity):
    """jackknife and surrogate_test recompute a subclass's own measure on
    replicates of the subclass, which keep its attributes and the instance's
    settings (the half bandwidth, the observation flags)."""

    class Recording(Connectivity):
        def __init__(self, fourier_coefficients, label, **kwargs):
            super().__init__(fourier_coefficients, **kwargs)
            self.label = label
            self.seen = []

        def labelled_coherence(self):
            self.seen.append((self.label, self.half_bandwidth, self.time_bins_are_independent))
            return self.magnitude_squared_coherence()

    connectivity = Recording(
        lagged_connectivity.fourier_coefficients,
        "recorded",
        frequencies=lagged_connectivity.all_frequencies,
        half_bandwidth=2.0,
        time_bins_are_independent=False,
    )
    jackknifed = connectivity.jackknife(
        "labelled_coherence", leave_out="trial", transformation="fisher_squared"
    )
    expected = connectivity.jackknife("magnitude_squared_coherence", leave_out="trial")
    np.testing.assert_array_equal(jackknifed.standard_error, expected.standard_error)
    tested = connectivity.surrogate_test(
        "labelled_coherence", kind="trial_shuffle", n_surrogates=3, random_state=0
    )
    assert tested.p_value.shape == tested.estimate.shape
    # The replicates' calls were recorded on the replicates, which share the
    # list with the original, and saw its attributes and settings.
    assert len(connectivity.seen) == 2 + 10 + 3
    assert set(connectivity.seen) == {("recorded", 2.0, False)}


@pytest.mark.parametrize(
    ("measure", "expected"),
    [
        ("imaginary_coherency", "two-sided"),
        ("phase_lag_index", "two-sided"),
        ("weighted_phase_lag_index", "two-sided"),
        ("magnitude_squared_coherence", "greater"),
        ("imaginary_coherence", "greater"),
        ("pairwise_phase_consistency", "greater"),
        ("debiased_squared_phase_lag_index", "greater"),
        ("directed_phase_lag_index", "greater"),
        ("pairwise_spectral_granger_prediction", "greater"),
        ("phase_slope_index", "two-sided"),
    ],
)
def test_auto_alternative_is_two_sided_for_lead_lag_signs(
    lagged_connectivity, measure, expected
):
    """'auto' is two-sided where the sign says which signal leads, greater
    otherwise (a debiased estimate below 0 is chance, not a direction); the
    result records the resolved alternative, which matches passing it."""
    kwargs = {"frequency_range": (10, 30)} if measure == "phase_slope_index" else {}
    automatic = lagged_connectivity.surrogate_test(
        measure, kind="conjugation", n_surrogates=4, random_state=0, **kwargs
    )
    assert automatic.alternative == expected
    explicit = lagged_connectivity.surrogate_test(
        measure,
        kind="conjugation",
        n_surrogates=4,
        random_state=0,
        alternative=expected,
        **kwargs,
    )
    np.testing.assert_array_equal(automatic.p_value, explicit.p_value)


def test_nan_surrogates_warn_once_with_their_count(lagged_connectivity, monkeypatch):
    """A measure undefined on some surrogates warns once per call, with the
    counts, and those surrogates raise the p-values (never lower them)."""
    calls = {"n": 0}
    original = Connectivity.magnitude_squared_coherence

    def sometimes_nan(self):
        values = original(self)
        calls["n"] += 1
        if calls["n"] in (3, 5):  # the 2nd and 4th surrogates (call 1 is the estimate)
            values = values.copy()
            values[0, 10, 0, 1] = np.nan
        return values

    monkeypatch.setattr(Connectivity, "magnitude_squared_coherence", sometimes_nan)
    with pytest.warns(
        UserWarning, match=r"2 of 9 surrogates returned NaN at tested elements \(2 values"
    ) as record:
        result = lagged_connectivity.surrogate_test(
            "magnitude_squared_coherence", kind="conjugation", n_surrogates=9, random_state=0
        )
    assert sum("returned NaN" in str(item.message) for item in record) == 1
    assert result.p_value[0, 10, 0, 1] >= 3 / 10
    assert np.all(result.fwer_p_value[np.isfinite(result.fwer_p_value)] >= 3 / 10)


def test_rotation_edges_use_the_frequency_range_rule_for_boundary_bins():
    """A half bandwidth that differs from a bin by rounding (NW / T = 2 / 1.152 s
    is 1.7361111111111112 Hz, the bin 1.736111111111111 Hz) keeps that bin
    tested, and a phase_slope_index band starting at it is accepted, by the
    rule frequency_range uses for a bin on a bound.

    Regression: the edge mask compared with >=, dropping the bin, and then
    refused frequency_range=(half_bandwidth, 40) asking for low >= half_bandwidth.
    """
    rng = np.random.default_rng(0)
    multitaper = Multitaper(
        rng.standard_normal((288, 4, 2)), sampling_frequency=250, time_halfbandwidth_product=2
    )
    connectivity = Connectivity.from_transform(multitaper)
    half_bandwidth = connectivity.half_bandwidth
    boundary = connectivity.frequencies[2]
    assert boundary < half_bandwidth
    assert np.isclose(boundary, half_bandwidth, rtol=1e-14, atol=0)
    result = connectivity.surrogate_test(
        "magnitude_squared_coherence", kind="phase_rotation", n_surrogates=3, random_state=0
    )
    assert np.isnan(result.p_value[0, :2, 0, 1]).all()
    assert np.isfinite(result.p_value[0, 2, 0, 1])
    # The top edge: Nyquist (125 Hz) minus the half bandwidth lies on a bin too.
    top = connectivity.frequencies[-3]
    assert np.isclose(top, 125 - half_bandwidth, rtol=1e-14, atol=0)
    assert np.isfinite(result.p_value[0, -3, 0, 1])
    assert np.isnan(result.p_value[0, -2:, 0, 1]).all()
    psi = connectivity.surrogate_test(
        "phase_slope_index",
        kind="phase_rotation",
        n_surrogates=3,
        random_state=0,
        frequency_range=(half_bandwidth, 40.0),
    )
    assert np.isfinite(psi.p_value[0, 0, 1])


def test_an_exact_spectrum_is_refused(small_connectivity, monkeypatch):
    """An exact spectrum (pseudo-observations from a VAR model) has no sampling
    error to test against. ``_is_exact_spectrum`` is set by Connectivity where
    exact spectra exist; set it here so the refusal is tested on every branch."""
    monkeypatch.setattr(small_connectivity, "_is_exact_spectrum", True, raising=False)
    with pytest.raises(ValueError, match="exact spectrum"):
        small_connectivity.surrogate_test(
            "magnitude_squared_coherence", kind="trial_shuffle", n_surrogates=3
        )
