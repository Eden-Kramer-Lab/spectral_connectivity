"""Monte Carlo and oracle tests of Connectivity.surrogate_test.

Monte Carlo checks run one ``surrogate_test`` call per replicate dataset, each
with its own seed, so replicates are independent in both their data and their
surrogates and the five-standard-error rule of ``tests/_monte_carlo.py``
applies. (Stacking replicates on the kept time-window axis would share the
surrogate draws between them, which adds variance the replicate standard error
does not see.) A rejection rate's standard error is that of the per-replicate
rejection fraction: frequencies within one replicate are correlated, so they
are not counted as independent.

The equivalence tests reproduce every surrogate with an independent
time-domain construction (trials re-paired, trials reversed in time) and match
the surrogates the package draws, exposed one per row of ``null_maxima`` when
each family is the one pair at one frequency.

The planted-defect tests patch a plausible bug into the implementation and
check that one of the oracles here fails on it.
"""

import itertools
import warnings

import numpy as np
import pytest
from scipy.signal import lfilter
from scipy.stats import ks_2samp

from spectral_connectivity import Connectivity, Multitaper
from spectral_connectivity.simulate import simulate_MVAR
from spectral_connectivity.statistics import _SurrogateTally
from tests._backend_helpers import to_device, to_host
from tests._monte_carlo import replicate_mean_and_limit
from tests._var_oracle import _analytic_var

# 19 surrogates: p-values are multiples of 1/20, so P(p <= alpha) is exactly
# floor(20 alpha) / 20 for an exact test without ties. 0.09 is off the grid:
# a p-value computed as b / n instead of (1 + b) / (n + 1) rejects 10% there.
_N_SURROGATES = 19
_LEVELS = (0.05, 0.09, 0.25, 0.5)
_GRANGER = "pairwise_spectral_granger_prediction"


def _exact_size(level, n_surrogates=_N_SURROGATES):
    return np.floor(level * (n_surrogates + 1) + 1e-9) / (n_surrogates + 1)


def _ar1(rng, n_time, n_columns, n_signals, coefficient=0.5):
    """Independent AR(1) series, shape (n_time, n_columns, n_signals)."""
    noise = rng.standard_normal((n_time + 200, n_columns, n_signals))
    return lfilter([1.0], [1.0, -coefficient], noise, axis=0)[200:]


def _ar2(rng, n_time, n_columns, n_signals, peak, radius, sampling_frequency):
    """Independent AR(2) series peaked at ``peak`` Hz, (n_time, n_columns, n_signals)."""
    a1 = 2 * radius * np.cos(2 * np.pi * peak / sampling_frequency)
    noise = rng.standard_normal((n_time + 3000, n_columns, n_signals))
    return lfilter([1.0], [1.0, -a1, radius**2], noise, axis=0)[3000:]


def _connectivities(series, n_trials, sampling_frequency, nw, **connectivity_kwargs):
    """One single-window Connectivity per replicate.

    ``series`` is ``(n_time, n_replicates * n_trials, n_signals)``.
    """
    multitaper = Multitaper(
        series, sampling_frequency=sampling_frequency, time_halfbandwidth_product=nw
    )
    coefficients = to_host(multitaper.fft())
    n_replicates = coefficients.shape[1] // n_trials
    coefficients = coefficients.reshape(n_replicates, 1, n_trials, *coefficients.shape[2:])
    frequencies = to_host(multitaper.frequencies)
    connectivity_kwargs.setdefault("half_bandwidth", multitaper.half_bandwidth)
    return [
        Connectivity(replicate, frequencies=frequencies, **connectivity_kwargs)
        for replicate in coefficients
    ]


def _results(connectivities, measure, kind, seed):
    """One surrogate_test per replicate, each with its own seed."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return [
            connectivity.surrogate_test(
                measure, kind=kind, n_surrogates=_N_SURROGATES, random_state=seed + index
            )
            for index, connectivity in enumerate(connectivities)
        ]


def _p_values(connectivities, measure, kind, seed):
    """(n_replicates, n_frequencies) p-values of pair (0, 1), one call each."""
    results = _results(connectivities, measure, kind, seed)
    return np.stack([result.p_value[0, :, 0, 1] for result in results])


def _per_replicate_rate(p_values, level, band=np.s_[:]):
    values = p_values[:, band]
    rejected = np.where(np.isnan(values), np.nan, values <= level)
    return np.nanmean(rejected, axis=1)


def _assert_size(p_values, label, band=np.s_[:], levels=_LEVELS, allowance=0.0):
    """Each level's rejection rate is within 5 standard errors (plus
    ``allowance``) of the exact size."""
    for level in levels:
        mean, limit = replicate_mean_and_limit(_per_replicate_rate(p_values, level, band))
        expected = _exact_size(level)
        assert abs(mean - expected) <= limit + allowance, (
            f"{label} at alpha={level}: {mean:.4f} vs {expected} +/- {limit + allowance:.4f}"
        )


# --------------------------------------------------------------------------
# 1. Size of each kind under its own exact null
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def independent_ten_trials():
    """400 replicates: 2 independent AR(1) signals, 10 trials, 1 s at 64 Hz,
    NW=2 (3 tapers, half bandwidth 2 Hz)."""
    rng = np.random.default_rng(100)
    return _connectivities(_ar1(rng, 64, 400 * 10, 2), 10, 64, 2)


@pytest.fixture(scope="module")
def zero_lag_mixture():
    """400 replicates: instantaneous mixing of 2 independent AR(1) sources
    (cross-weight 0.7), so the coherency is real at every frequency; 10
    trials, 1 s at 64 Hz, NW=2."""
    rng = np.random.default_rng(102)
    sources = _ar1(rng, 64, 400 * 10, 2)
    mixed = sources @ np.array([[1.0, 0.7], [0.7, 1.0]])
    return _connectivities(mixed, 10, 64, 2)


def _check_trial_shuffle_size(connectivities, measure="magnitude_squared_coherence"):
    p_values = _p_values(connectivities, measure, "trial_shuffle", 0)
    _assert_size(p_values, f"trial_shuffle {measure}")
    # Every frequency is tested, 0 Hz and Nyquist too.
    _assert_size(p_values, f"trial_shuffle {measure} at 0 Hz and Nyquist", np.s_[[0, -1]])


def _check_conjugation_size(connectivities, measure):
    p_values = _p_values(connectivities, measure, "conjugation", 20_000)
    # 0 Hz and Nyquist, which conjugation leaves unchanged, are NaN.
    assert np.isnan(p_values[:, [0, -1]]).all()
    _assert_size(p_values, f"conjugation {measure}")


@pytest.mark.parametrize(
    ("measure", "n_replicates"),
    [("magnitude_squared_coherence", None), (_GRANGER, 100)],
)
def test_trial_shuffle_size_at_several_levels(independent_ten_trials, measure, n_replicates):
    """Independent signals: exact size floor(20 alpha) / 20 at every frequency."""
    _check_trial_shuffle_size(independent_ten_trials[:n_replicates], measure)


def test_phase_rotation_size_at_several_levels(independent_ten_trials):
    """Every tested frequency, from W = 2 Hz to Nyquist - W; the untested ones
    are exactly those within W of 0 and 32 Hz."""
    p_values = _p_values(
        independent_ten_trials, "magnitude_squared_coherence", "phase_rotation", 10_000
    )
    _assert_size(p_values, "phase_rotation")
    untested = np.isnan(p_values).all(axis=0)
    frequencies = np.arange(33)
    np.testing.assert_array_equal(untested, (frequencies < 2) | (frequencies > 30))


def test_phase_rotation_size_of_pairwise_granger(independent_ten_trials):
    """Pairwise spectral Granger factors the whole spectrum, the untested
    frequencies near 0 Hz and Nyquist included, which spreads a small excess
    past W. Measured with 300 replicates (19 surrogates) at alpha=0.05: 5.4%
    over every tested frequency, 7.1% within 1.5 W of an edge (2-3 and 29-30
    Hz)."""
    p_values = _p_values(independent_ten_trials[:100], _GRANGER, "phase_rotation", 11_000)
    _assert_size(p_values, "phase_rotation Granger", levels=(0.05, 0.25), allowance=0.01)


@pytest.mark.parametrize(
    ("measure", "n_replicates"),
    [
        ("imaginary_coherence", None),
        ("magnitude_squared_coherence", None),
        (_GRANGER, 100),
    ],
)
def test_conjugation_size_under_zero_lag_mixing(zero_lag_mixture, measure, n_replicates):
    """Real coherency is conjugation's null, whatever the measure: conjugating
    a whole trial is reversing it in time, which leaves the distribution of
    zero-lag mixed Gaussian signals unchanged, so pairwise Granger (nonzero in
    both directions here) holds its size too, and a magnitude-squared
    coherence test asks only whether the coupling exceeds what a real
    coherency gives."""
    _check_conjugation_size(zero_lag_mixture[:n_replicates], measure)


# --------------------------------------------------------------------------
# 2. Family-wise error under a global null
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def four_independent_signals():
    """600 replicates: 4 independent AR(1) signals, 10 trials, 1 s at 32 Hz,
    NW=2 (17 frequencies x 6 pairs = 102 tests per family)."""
    rng = np.random.default_rng(103)
    return _connectivities(_ar1(rng, 32, 600 * 10, 4), 10, 32, 2)


def _family_rejections(connectivities, kind, seed):
    results = _results(connectivities, "magnitude_squared_coherence", kind, seed)
    smallest_fwer = np.array([np.nanmin(result.fwer_p_value) for result in results])
    smallest_raw = np.array([np.nanmin(result.p_value) for result in results])
    return smallest_fwer, smallest_raw


def _check_fwer(connectivities, kind):
    smallest_fwer, smallest_raw = _family_rejections(connectivities, kind, 30_000)
    for level in (0.05, 0.25, 0.5):
        rejected = (smallest_fwer <= level).astype(float)
        mean, limit = replicate_mean_and_limit(rejected)
        # The maximum statistic is exact for an exact surrogate null whose
        # every tested element the surrogates change.
        assert abs(mean - _exact_size(level)) <= limit, f"{kind} {level}: {mean:.4f}"
    # Uncorrected p-values reject somewhere in most families.
    assert np.mean(smallest_raw <= 0.05) > 0.9


@pytest.mark.parametrize("kind", ["trial_shuffle", "phase_rotation", "conjugation"])
def test_familywise_error_under_the_global_null(four_independent_signals, kind):
    """Exact for every kind, at several levels. For conjugation this needs 0 Hz
    and Nyquist out of the family: their values are the same in every
    surrogate, so they would floor every surrogate maximum (measured 0.18 at
    alpha = 0.25 and 0.30 at 0.5 with them in)."""
    _check_fwer(four_independent_signals, kind)


def test_a_family_of_one_element_gives_the_uncorrected_p_value(independent_ten_trials):
    """fwer_axis=(3,) with 2 signals: each family is one tested element (the
    other is the diagonal), so the max statistic is that element's."""
    result = independent_ten_trials[0].surrogate_test(
        "magnitude_squared_coherence",
        kind="trial_shuffle",
        n_surrogates=99,
        random_state=1,
        fwer_axis=(3,),
    )
    np.testing.assert_array_equal(result.fwer_p_value, result.p_value)


# --------------------------------------------------------------------------
# 3. Power rises with the true coherence of a coupled VAR
# --------------------------------------------------------------------------


def _lag_one_var(strength):
    """x1 white; x2(t) = strength * x1(t - 1) + e2(t)."""
    coefficients = np.zeros((1, 2, 2))
    coefficients[0, 1, 0] = strength
    return coefficients


@pytest.mark.parametrize(
    ("kind", "measure", "band", "power_at_half"),
    [
        ("trial_shuffle", "magnitude_squared_coherence", np.s_[:], 0.75),
        ("phase_rotation", "magnitude_squared_coherence", np.s_[:], 0.75),
        # Im(coherency) of a one-sample lag is proportional to sin(2 pi f / fs).
        # Conjugating whole trials draws 10 flips, not 30 as per taper would,
        # which costs power; measured 0.69 at c = 0.5.
        ("conjugation", "imaginary_coherence", np.s_[8:25], 0.6),
    ],
)
def test_power_rises_with_the_true_coherence(kind, measure, band, power_at_half):
    """Rejection at 5% grows with the VAR's true coherence (c^2 / (1 + c^2) at
    every frequency, from the analytic spectrum) from about 5% at c = 0."""
    n_fft, n_trials, n_replicates = 64, 10, 150
    rates = []
    for index, strength in enumerate((0.0, 0.25, 0.5)):
        _, _, spectrum = _analytic_var(_lag_one_var(strength), np.eye(2), n_fft)
        coherence = np.abs(spectrum[:, 0, 1]) ** 2 / (
            spectrum[:, 0, 0].real * spectrum[:, 1, 1].real
        )
        np.testing.assert_allclose(coherence, strength**2 / (1 + strength**2), atol=1e-12)
        series = simulate_MVAR(
            _lag_one_var(strength),
            n_time_samples=n_fft,
            n_trials=n_replicates * n_trials,
            random_state=200 + index,
        )
        connectivities = _connectivities(series, n_trials, 64, 2)
        p_values = _p_values(connectivities, measure, kind, 40_000 + 1000 * index)
        rates.append(_per_replicate_rate(p_values, 0.05, band))
    null_mean, null_limit = replicate_mean_and_limit(rates[0])
    assert abs(null_mean - 0.05) <= null_limit
    means = [rate.mean() for rate in rates]
    assert means[0] < means[1] < means[2], means
    assert means[2] > power_at_half, means


# --------------------------------------------------------------------------
# 4. Composite nulls: zero-lag versus lagged coupling, shared responses
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def lagged_coupling():
    """200 replicates of the lag-one VAR with c = 0.7, 10 trials, 64 samples."""
    series = simulate_MVAR(
        _lag_one_var(0.7), n_time_samples=64, n_trials=200 * 10, random_state=300
    )
    return _connectivities(series, 10, 64, 2)


@pytest.mark.parametrize(
    ("kind", "measure", "rejects_zero_lag"),
    [
        ("conjugation", "imaginary_coherence", False),
        ("conjugation", "magnitude_squared_coherence", False),
        ("phase_rotation", "magnitude_squared_coherence", True),
        ("trial_shuffle", "magnitude_squared_coherence", True),
    ],
)
def test_zero_lag_and_lagged_coupling(
    zero_lag_mixture, lagged_coupling, kind, measure, rejects_zero_lag
):
    """Docs: conjugation keeps zero-lag mixing in its null and rejects lagged
    coupling; rotation and the shuffle (null: independence) reject both."""
    zero_lag = _p_values(zero_lag_mixture[:100], measure, kind, 50_000)
    zero_lag_rate = _per_replicate_rate(zero_lag, 0.05).mean()
    if rejects_zero_lag:
        assert zero_lag_rate > 0.95
    else:
        assert zero_lag_rate < 0.1
    # Lagged: conjugation's imaginary coherence is tested where Im is large.
    band = np.s_[8:25] if kind == "conjugation" else np.s_[:]
    lagged = _p_values(lagged_coupling, measure, kind, 60_000)
    assert _per_replicate_rate(lagged, 0.05, band).mean() > 0.85


def test_a_shared_evoked_response_is_in_the_shuffle_null_but_not_the_rotation_null():
    """The same 10 Hz evoked burst, at zero lag, in every trial of two otherwise
    independent signals: the trial shuffle keeps it in every surrogate and holds
    its size, conjugation keeps it too (zero lag, real coherency), and phase
    rotation, which destroys stimulus-locked timing, rejects it."""
    rng = np.random.default_rng(28)
    n_replicates, n_trials, sampling_frequency = 200, 10, 128
    time = np.arange(128) / sampling_frequency - 0.5
    evoked = 3 * np.exp(-(time**2) / (2 * 0.05**2)) * np.cos(2 * np.pi * 10 * time)
    series = np.concatenate(
        [
            _ar2(rng, 128, n_replicates * n_trials, 1, 20, 0.9, sampling_frequency),
            _ar2(rng, 128, n_replicates * n_trials, 1, 40, 0.8, sampling_frequency),
        ],
        axis=-1,
    )
    series += evoked[:, np.newaxis, np.newaxis]
    connectivities = _connectivities(series, n_trials, sampling_frequency, 2)
    evoked_band = np.s_[5:16]
    for kind in ("trial_shuffle", "conjugation"):
        p_values = _p_values(connectivities, "magnitude_squared_coherence", kind, 70_000)
        _assert_size(p_values, kind, band=evoked_band, levels=(0.05,))
    # Rotation rejects in most tests; 30 replicates show it.
    p_values = _p_values(
        connectivities[:30], "magnitude_squared_coherence", "phase_rotation", 70_000
    )
    assert _per_replicate_rate(p_values, 0.05, evoked_band).mean() > 0.5


# --------------------------------------------------------------------------
# 5. Equivalence with independent constructions, and exact limits
# --------------------------------------------------------------------------


def _measure_of(series, measure, sampling_frequency, nw):
    """``measure`` at pair (0, 1), shape (n_frequencies,), of one dataset."""
    multitaper = Multitaper(
        series, sampling_frequency=sampling_frequency, time_halfbandwidth_product=nw
    )
    connectivity = Connectivity.from_transform(multitaper)
    return getattr(connectivity, measure)()[0, :, 0, 1]


def _match_rows(null_rows, references):
    """Index of the reference each null row equals (to rounding), or -1."""
    matches = np.full(len(null_rows), -1)
    for row_index, row in enumerate(null_rows):
        for reference_index, reference in enumerate(references):
            if np.allclose(row, reference, rtol=1e-9, atol=1e-13):
                matches[row_index] = reference_index
                break
    return matches


def _per_frequency_null(result):
    """Each surrogate's value at pair (0, 1) and every frequency, from
    ``null_maxima`` with one family per (time, frequency): the family is the
    pair and its transpose, which are equal for a symmetric measure."""
    assert result.fwer_axis == (2, 3)
    return result.null_maxima[:, 0, :]


@pytest.fixture(scope="module")
def three_trials():
    """One dataset of 2 independent AR(1) signals, 3 trials, 64 samples."""
    return _ar1(np.random.default_rng(400), 64, 3, 2)


def _check_trial_shuffle_equivalence(series):
    n_surrogates = 600
    connectivity = Connectivity.from_transform(
        Multitaper(series, sampling_frequency=64, time_halfbandwidth_product=2)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = connectivity.surrogate_test(
            "magnitude_squared_coherence",
            kind="trial_shuffle",
            n_surrogates=n_surrogates,
            fwer_axis=(2, 3),
            random_state=7,
        )
    # Independent construction: re-pair signal 1's trials in the time domain.
    permutations = list(itertools.permutations(range(3)))
    references = []
    for permutation in permutations:
        shuffled = series.copy()
        shuffled[:, :, 1] = series[:, list(permutation), 1]
        references.append(_measure_of(shuffled, "magnitude_squared_coherence", 64, 2))
    null_rows = _per_frequency_null(result)
    matches = _match_rows(null_rows, references)
    assert (matches >= 0).all(), "a surrogate is not a re-pairing of trials"
    counts = np.bincount(matches, minlength=len(permutations))
    expected = n_surrogates / len(permutations)
    standard_error = np.sqrt(n_surrogates * (1 / 6) * (5 / 6))
    assert np.all(np.abs(counts - expected) <= 5 * standard_error), counts
    # Exact limit: the identity permutation reproduces the estimate (index 0)
    # and must count as a tie in the p-value, which is (1 + b) / (n + 1).
    np.testing.assert_allclose(references[0], result.estimate[0, :, 0, 1], rtol=1e-12)
    exceed = (null_rows >= result.estimate[0, :, 0, 1] * (1 - 1e-6) - 1e-12).sum(axis=0)
    np.testing.assert_allclose(result.p_value[0, :, 0, 1], (1 + exceed) / (n_surrogates + 1))
    assert np.all(exceed >= counts[0])


def test_trial_shuffle_matches_time_domain_repairing(three_trials):
    """Every surrogate is one of the 3! re-pairings of signal 1's trials
    computed independently in the time domain, each drawn about equally
    often; the p-value is the brute-force count from those surrogates, with
    the identity counted as a tie."""
    _check_trial_shuffle_equivalence(three_trials)


def _check_conjugation_equivalence(series):
    n_surrogates = 800
    connectivity = Connectivity.from_transform(
        Multitaper(series, sampling_frequency=64, time_halfbandwidth_product=2)
    )
    assert connectivity.fourier_coefficients.shape[2] == 3
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = connectivity.surrogate_test(
            "magnitude_squared_coherence",
            kind="conjugation",
            n_surrogates=n_surrogates,
            fwer_axis=(2, 3),
            random_state=8,
        )
    # Independent construction: reverse a subset of trials (both signals) in
    # time. The tapers are even or odd about the window's center, so reversing
    # a trial conjugates every taper's coefficients (up to a sign per taper and
    # a phase per frequency that every signal shares).
    subsets = [
        subset for size in range(4) for subset in itertools.combinations(range(3), size)
    ]
    references = []
    for subset in subsets:
        reversed_series = series.copy()
        for trial in subset:
            reversed_series[:, trial] = series[::-1, trial]
        references.append(
            _measure_of(reversed_series, "magnitude_squared_coherence", 64, 2)[1:-1]
        )
    # 0 Hz and Nyquist are untested (-inf in each surrogate's family maximum).
    null_rows = _per_frequency_null(result)
    assert np.all(null_rows[:, [0, -1]] == -np.inf)
    matches = _match_rows(null_rows[:, 1:-1], references)
    assert (matches >= 0).all(), "a surrogate is not a time reversal of whole trials"
    # Reversing a subset or its complement gives the same coherence (conjugate
    # coherency); the 4 classes are equally likely.
    complement = {
        index: subsets.index(tuple(sorted(set(range(3)) - set(subset))))
        for index, subset in enumerate(subsets)
    }
    classes = np.array([min(index, complement[index]) for index in matches])
    counts = np.bincount(classes)[np.unique(classes)]
    assert counts.size == 4, counts
    standard_error = np.sqrt(n_surrogates * 0.25 * 0.75)
    assert np.all(np.abs(counts - n_surrogates / 4) <= 5 * standard_error), counts


def test_conjugation_matches_time_reversal_of_trials(three_trials):
    """With 3 tapers, every conjugation surrogate is the data with some trials
    reversed in time (computed independently), each class of reversals drawn
    about equally often. Magnitude-squared coherence is used because it
    changes under a partial reversal of a coupled or uncoupled pair alike."""
    _check_conjugation_equivalence(three_trials)


def _rotation_reference_null(coefficients, frequency_index, n_surrogates, rng, *, per_taper):
    """Magnitude-squared coherence of phase-rotated coefficients, independently.

    ``coefficients`` is ``(n_trials, n_tapers, n_fft, 2)``; one phase per
    (trial, signal), or per (trial, taper, signal) when ``per_taper``.
    """
    n_trials, n_tapers = coefficients.shape[:2]
    values = coefficients[:, :, frequency_index]  # (n_trials, n_tapers, 2)
    null = np.empty(n_surrogates)
    for index in range(n_surrogates):
        phase_shape = (n_trials, n_tapers if per_taper else 1, 2)
        rotated = values * np.exp(1j * rng.uniform(0, 2 * np.pi, phase_shape))
        cross = np.mean(rotated[..., 0] * rotated[..., 1].conj())
        power = np.mean(np.abs(rotated) ** 2, axis=(0, 1))
        null[index] = np.abs(cross) ** 2 / (power[0] * power[1])
    return null


def _check_rotation_distribution(series, seed):
    n_surrogates = 1500
    multitaper = Multitaper(series, sampling_frequency=64, time_halfbandwidth_product=2)
    connectivity = Connectivity.from_transform(multitaper)
    result = connectivity.surrogate_test(
        "magnitude_squared_coherence",
        kind="phase_rotation",
        n_surrogates=n_surrogates,
        fwer_axis=(2, 3),
        random_state=seed,
    )
    coefficients = to_host(multitaper.fft())[0]
    package_null = _per_frequency_null(result)
    p_values = {}
    for frequency_index in (5, 16, 27):
        reference = _rotation_reference_null(
            coefficients,
            frequency_index,
            n_surrogates,
            np.random.default_rng(seed + frequency_index),
            per_taper=False,
        )
        p_values[frequency_index] = ks_2samp(
            package_null[:, frequency_index], reference
        ).pvalue
    assert min(p_values.values()) > 1e-3, p_values
    return package_null, coefficients


def test_phase_rotation_null_matches_an_independent_rotation(three_trials):
    """The package's null distribution of magnitude-squared coherence at
    several frequencies matches (two-sample KS) one built by rotating each
    trial (every taper alike) by an independent phase here; a reference that
    rotates each taper separately is clearly different, so the comparison can
    tell the two apart."""
    package_null, coefficients = _check_rotation_distribution(three_trials, 9)
    per_taper = _rotation_reference_null(
        coefficients, 16, 1500, np.random.default_rng(1), per_taper=True
    )
    assert ks_2samp(package_null[:, 16], per_taper).pvalue < 1e-6


# --------------------------------------------------------------------------
# 6. Spectra that are not flat across the half bandwidth
# --------------------------------------------------------------------------


_PEAKS = (12, 24, 36, 48)
# Within 3 Hz of each peak, where the spectrum changes most across +/- W.
_PEAK_BANDS = np.concatenate([np.arange(peak - 3, peak + 4) for peak in _PEAKS])


@pytest.fixture(scope="module")
def narrowband_independent():
    """300 replicates of 2 independent signals, each the sum of 4 independent
    AR(2) series with sharp peaks at 12, 24, 36 and 48 Hz (pole radius 0.99,
    about 0.4 Hz wide against 2W = 8 Hz), 10 trials, 1 s at 128 Hz, NW=4 (7
    tapers). Near each peak a trial's 7 eigencoefficients are strongly
    correlated. Each peak gives an almost independent test, whereas the
    frequencies around one peak share its coefficients: with one peak the
    replicate limit at alpha=0.05 was about 0.06, so a size of 10% would
    have passed, and with four it is about 0.03."""
    rng = np.random.default_rng(500)
    series = sum(_ar2(rng, 128, 300 * 10, 2, peak, 0.99, 128) for peak in _PEAKS)
    return _connectivities(series, 10, 128, 4)


def _check_narrowband_size(connectivities, kind):
    p_values = _p_values(connectivities, "magnitude_squared_coherence", kind, 70_000)
    _assert_size(p_values, kind, band=_PEAK_BANDS, levels=(0.05, 0.25))


@pytest.mark.parametrize("kind", ["trial_shuffle", "phase_rotation", "conjugation"])
def test_size_for_independent_narrowband_signals(narrowband_independent, kind):
    """Independent signals are in every kind's null. Each kind moves whole
    trials, so the correlation between a trial's tapers is kept in every
    surrogate (randomizing each taper separately rejected 36% with rotation
    and 24% with conjugation here, 150 replicates, 19 surrogates)."""
    _check_narrowband_size(narrowband_independent, kind)


@pytest.fixture(scope="module")
def steep_one_recording():
    """400 replicates of one 10 s recording of 2 independent AR(1) signals
    with coefficient 0.99 (a steep, 1/f^2-like spectrum, as in field
    potentials) at 128 Hz, in 10 non-overlapping 1 s windows with NW=4 (W = 4
    Hz), the windows averaged (expectation_type="time_tapers")."""
    rng = np.random.default_rng(7)
    # 1000 more samples of burn-in: 0.99**1200 is 6e-6.
    series = _ar1(rng, 2280, 400, 2, coefficient=0.99)[1000:]
    multitaper = Multitaper(
        series, sampling_frequency=128, time_halfbandwidth_product=4, time_window_duration=1.0
    )
    coefficients = to_host(multitaper.fft())  # (10, 400, 7, 128, 2)
    frequencies = to_host(multitaper.frequencies)
    return [
        Connectivity(
            coefficients[:, [replicate]],
            frequencies=frequencies,
            expectation_type="time_tapers",
            half_bandwidth=multitaper.half_bandwidth,
            time_bins_are_disjoint=multitaper.time_bins_are_disjoint,
        )
        for replicate in range(coefficients.shape[1])
    ]


@pytest.mark.parametrize("kind", ["phase_rotation", "conjugation"])
def test_one_recording_is_tested_by_its_averaged_windows(steep_one_recording, kind):
    """A single trial is refused, but averaging its windows makes each window a
    unit: the size holds at the first tested frequencies, 4-7 Hz, where the
    spectrum changes most across +/- W (per-taper rotation of one 1 s trial
    rejected 19% there), and at every tested frequency."""
    single_window = Connectivity(
        steep_one_recording[0].fourier_coefficients[:1],
        frequencies=steep_one_recording[0].all_frequencies,
        half_bandwidth=4.0,
    )
    with pytest.raises(ValueError, match="this Connectivity has 1 "):
        single_window.surrogate_test("magnitude_squared_coherence", kind=kind)
    p_values = _p_values(steep_one_recording, "magnitude_squared_coherence", kind, 80_000)
    _assert_size(p_values, f"{kind} 4-7 Hz", band=np.s_[4:8], levels=(0.05, 0.25))
    _assert_size(p_values, f"{kind} every tested frequency", levels=(0.05, 0.25))


@pytest.fixture(scope="module")
def overlapping_windows():
    """400 replicates of 2 independent AR(1) signals, 10 trials of 4 s at 64
    Hz, in 1 s windows every 0.5 s (7 per trial, overlapping by half) with
    NW=2 (3 tapers), the windows averaged with the trials
    (expectation_type="time_trials_tapers")."""
    multitaper = Multitaper(
        _ar1(np.random.default_rng(900), 256, 400 * 10, 2),
        sampling_frequency=64,
        time_halfbandwidth_product=2,
        time_window_duration=1.0,
        time_window_step=0.5,
    )
    assert multitaper.time_bins_are_independent
    assert not multitaper.time_bins_are_disjoint
    coefficients = to_host(multitaper.fft())  # (7, 4000, 3, 64, 2)
    coefficients = coefficients.reshape(7, 400, 10, *coefficients.shape[2:])
    return coefficients, to_host(multitaper.frequencies), multitaper.half_bandwidth


@pytest.mark.parametrize("kind", ["phase_rotation", "conjugation"])
def test_overlapping_windows_share_their_trial_draw(overlapping_windows, kind):
    """Windows that overlap by half have correlated coefficients, so they are
    not units: one recording in such windows is refused, and with trials every
    window of a trial shares its draw and the size holds. Randomizing each
    window separately (claiming time_bins_are_disjoint=True for these
    windows) rejects far more often: 14.2% for rotation and 11.4% for
    conjugation at alpha=0.05, against 4.8% and 5.3% with the trial's draw
    shared, which is 7.5 and 4.5 limits away."""
    one_recording = Connectivity.from_transform(
        Multitaper(
            _ar1(np.random.default_rng(901), 640, 1, 2),
            sampling_frequency=64,
            time_halfbandwidth_product=2,
            time_window_duration=1.0,
            time_window_step=0.5,
        ),
        expectation_type="time_tapers",
    )
    with pytest.raises(ValueError, match="windows that overlap are not independent units"):
        one_recording.surrogate_test("magnitude_squared_coherence", kind=kind)
    coefficients, frequencies, half_bandwidth = overlapping_windows

    def p_values(disjoint):
        connectivities = [
            Connectivity(
                coefficients[:, replicate],
                frequencies=frequencies,
                expectation_type="time_trials_tapers",
                half_bandwidth=half_bandwidth,
                time_bins_are_disjoint=disjoint,
            )
            for replicate in range(coefficients.shape[1])
        ]
        results = _results(connectivities, "magnitude_squared_coherence", kind, 90_000)
        return np.stack([result.p_value[:, 0, 1] for result in results])

    shared = p_values(disjoint=False)
    _assert_size(shared, f"{kind}, overlapping windows", levels=(0.05, 0.25))
    _, limit = replicate_mean_and_limit(_per_replicate_rate(shared, 0.05))
    per_window, _ = replicate_mean_and_limit(
        _per_replicate_rate(p_values(disjoint=True), 0.05)
    )
    assert per_window - _exact_size(0.05) > 4 * limit, (per_window, limit)


# --------------------------------------------------------------------------
# 7. Planted defects: each is caught by an oracle above
# --------------------------------------------------------------------------


def _p_without_plus_one(original):
    def p_values(self):
        p_value, fwer_p_value = original(self)
        n = self.null_maxima.shape[0]
        return (p_value * (n + 1) - 1) / n, (fwer_p_value * (n + 1) - 1) / n

    return p_values


def _fwer_without_max(original):
    def p_values(self):
        p_value, _ = original(self)
        return p_value, p_value

    return p_values


def _randomize_each_taper(original):
    def surrogate_coefficients(self, kind, unit_axes, rng):
        if kind in ("phase_rotation", "conjugation"):
            unit_axes = (*unit_axes, 2)
        return original(self, kind, unit_axes, rng)

    return surrogate_coefficients


def _shuffle_signals_jointly(original):
    def surrogate_coefficients(self, kind, unit_axes, rng):
        if kind != "trial_shuffle":
            return original(self, kind, unit_axes, rng)
        coefficients = to_host(self._fourier_coefficients)
        return to_device(coefficients[:, rng.permutation(coefficients.shape[1])])

    return surrogate_coefficients


def _conjugate_each_signal_separately(original):
    def surrogate_coefficients(self, kind, unit_axes, rng):
        if kind != "conjugation":
            return original(self, kind, unit_axes, rng)
        coefficients = to_host(self._fourier_coefficients)
        unit_shape = tuple(
            coefficients.shape[axis] if axis in unit_axes else 1 for axis in range(3)
        )
        flip = rng.random((*unit_shape, 1, coefficients.shape[-1])) < 0.5
        return to_device(np.where(flip, coefficients.conj(), coefficients))

    return surrogate_coefficients


def _conjugation_tests_every_frequency(original):
    def valid_frequencies(self, kind, *args):
        if kind == "conjugation":
            return np.ones((), dtype=bool)
        return original(self, kind, *args)

    return valid_frequencies


def test_planted_p_value_without_plus_one_is_caught(monkeypatch, independent_ten_trials):
    monkeypatch.setattr(
        _SurrogateTally, "p_values", _p_without_plus_one(_SurrogateTally.p_values)
    )
    with pytest.raises(AssertionError, match=r"alpha=0\.09"):
        _check_trial_shuffle_size(independent_ten_trials)


def test_planted_fwer_without_max_is_caught(monkeypatch, four_independent_signals):
    monkeypatch.setattr(
        _SurrogateTally, "p_values", _fwer_without_max(_SurrogateTally.p_values)
    )
    with pytest.raises(AssertionError):
        _check_fwer(four_independent_signals[:200], "trial_shuffle")


@pytest.mark.parametrize("kind", ["phase_rotation", "conjugation"])
def test_planted_per_taper_randomization_is_caught(
    monkeypatch, narrowband_independent, three_trials, kind
):
    monkeypatch.setattr(
        Connectivity,
        "_surrogate_coefficients",
        _randomize_each_taper(Connectivity._surrogate_coefficients),
    )
    with pytest.raises(AssertionError):
        _check_narrowband_size(narrowband_independent[:150], kind)
    if kind == "phase_rotation":
        with pytest.raises(AssertionError):
            _check_rotation_distribution(three_trials, 9)
    else:
        with pytest.raises(AssertionError):
            _check_conjugation_equivalence(three_trials)


def test_planted_joint_shuffle_is_caught(monkeypatch, independent_ten_trials, three_trials):
    monkeypatch.setattr(
        Connectivity,
        "_surrogate_coefficients",
        _shuffle_signals_jointly(Connectivity._surrogate_coefficients),
    )
    with pytest.raises(AssertionError):
        _check_trial_shuffle_equivalence(three_trials)
    with pytest.raises(AssertionError):
        _check_trial_shuffle_size(independent_ten_trials[:200])


def test_planted_per_signal_conjugation_is_caught(monkeypatch, zero_lag_mixture, three_trials):
    monkeypatch.setattr(
        Connectivity,
        "_surrogate_coefficients",
        _conjugate_each_signal_separately(Connectivity._surrogate_coefficients),
    )
    with pytest.raises(AssertionError):
        _check_conjugation_size(zero_lag_mixture[:200], "magnitude_squared_coherence")
    with pytest.raises(AssertionError):
        _check_conjugation_equivalence(three_trials)


def test_planted_conjugation_at_0_hz_and_nyquist_is_caught(
    monkeypatch, four_independent_signals
):
    monkeypatch.setattr(
        Connectivity,
        "_valid_surrogate_frequencies",
        _conjugation_tests_every_frequency(Connectivity._valid_surrogate_frequencies),
    )
    with pytest.raises(AssertionError):
        _check_fwer(four_independent_signals, "conjugation")
