"""Oracles for ``Multitaper(artifact_samples=...)``, each independent of the
code path it checks.

- Physical removal: every listed measure, for each expectation that keeps the
  time axis, equals the measure of each window's kept trials alone, with
  overlapping windows, a ragged end and a padded FFT; pooled expectations
  equal the measure of the kept observations stacked as one window.
- Published definitions: pairwise phase consistency (Vinck et al. 2010) and
  the debiased squared PLI and wPLI (Vinck et al. 2011), written as explicit
  sums over ordered pairs of distinct kept observations.
- A hand-computed leave-one-out jackknife over the kept units.
- The exact zero-coherence null, Beta(1, n - 1), at each window's kept count.
- Unbiasedness at a nonzero population value: dropping trials independently
  of the data leaves the expectation of an unbiased measure unchanged, so the
  paired difference from the all-trial estimate has mean zero.
- mne-connectivity 0.9 (recorded): masking extra, corrupted trials gives
  MNE's outputs for the recorded epochs.
- The statistics recipe the docs give for a window that keeps no trial.
"""

import dataclasses
import inspect
import warnings
from pathlib import Path

import numpy as np
import pytest
import scipy.stats
from scipy.signal.windows import dpss

from spectral_connectivity import (
    Connectivity,
    Multitaper,
    list_measures,
)
from spectral_connectivity.simulate import simulate_lagged_broadband
from spectral_connectivity.statistics import coherence_significance_pvalue
from tests._artifact_helpers import quietly, without_noisy_diagonal
from tests._backend_helpers import ON_GPU, to_host
from tests.reference._mne_measures import MEASURES as MNE_MEASURES

SAMPLING_FREQUENCY = 200.0
WINDOW = 40
STEP = 30  # overlapping windows
N_FFT = 64  # padded
N_TRIALS = 7  # differs from the signal, taper and frequency counts
N_SIGNALS = 3
GROUPS = ["a", "a", "b"]
# The Wilson factorization iterates to a tolerance, and canonical coherency's
# component phase comes from an eigen-solver: both move by more than rounding.
LOOSE = frozenset(
    {
        "pairwise_spectral_granger_prediction",
        "subset_pairwise_spectral_granger_prediction",
        "conditional_spectral_granger_prediction",
        "time_reversed_spectral_granger_prediction",
        "time_reversed_conditional_spectral_granger_prediction",
        "blockwise_spectral_granger_prediction",
        "time_reversed_blockwise_spectral_granger_prediction",
        "directed_transfer_function",
        "directed_coherence",
        "partial_directed_coherence",
        "generalized_partial_directed_coherence",
        "direct_directed_transfer_function",
        "canonical_coherency",
    }
)
# global_coherence's components come from svds by default, iterative on CuPy.
GPU_ITERATIVE = frozenset({"global_coherence"})
# delay and group_delay correct their p-values jointly over the windows of a
# call, so they agree with one window's kept trials only window by window
# (tested in test_artifact_samples.py).
JOINT_OVER_WINDOWS = frozenset({"delay", "group_delay"})
PER_WINDOW_MEASURES = [
    info.name for info in list_measures() if info.name not in JOINT_OVER_WINDOWS
]


def _multitaper(time_series, artifact_samples=None, **kwargs):
    kwargs = {
        "n_time_samples_per_window": WINDOW,
        "n_time_samples_per_step": STEP,
        "n_fft_samples": N_FFT,
        "time_halfbandwidth_product": 2,
        **kwargs,
    }
    if artifact_samples is not None:
        kwargs["artifact_samples"] = artifact_samples
    return Multitaper(time_series, SAMPLING_FREQUENCY, **kwargs)


def _call(connectivity, name):
    """``name`` on ``connectivity`` with the arguments it needs."""
    method = getattr(connectivity, name)
    parameters = inspect.signature(method).parameters
    kwargs = {}
    if "group_labels" in parameters:
        kwargs["group_labels"] = GROUPS
    if "pairs" in parameters:
        kwargs["pairs"] = [[0, 1], [1, 2]]
    return quietly(method, **kwargs)


def _numeric_parts(result):
    """The numeric arrays of a measure's result, in a fixed order."""
    if dataclasses.is_dataclass(result):
        parts = [getattr(result, field.name) for field in dataclasses.fields(result)]
    elif isinstance(result, tuple):
        parts = list(result)
    else:
        parts = [result]
    arrays = [np.asarray(part) for part in parts]
    return [array for array in arrays if array.dtype.kind in "fcb" and array.ndim]


@pytest.fixture(scope="module")
def recording():
    """Seven trials of three coupled signals; windows of 40 samples every 30
    (the last 10 samples are in no window), artifacts NaN.

    Every window drops a different set of trials, and one artifact lies in the
    overlap of two windows, so both drop that trial.
    """
    rng = np.random.default_rng(144)
    n_time = 4 * STEP + WINDOW + 10
    time_series = rng.standard_normal((n_time, N_TRIALS, N_SIGNALS))
    time_series[:, :, 1] += 0.7 * np.roll(time_series[:, :, 0], 2, axis=0)
    time_series[:, :, 2] += 0.5 * np.roll(time_series[:, :, 1], 1, axis=0)
    artifacts = np.zeros((n_time, N_TRIALS), dtype=bool)
    artifacts[5:8, [0, 4]] = True  # window 0
    artifacts[35, 2] = True  # windows 0 and 1 (overlap 30-39)
    artifacts[65:70, [1, 5, 6]] = True  # windows 1 and 2
    artifacts[125, 3] = True  # windows 3 and 4
    artifacts[n_time - 3, 0] = True  # in no window: drops nothing
    time_series[artifacts] = np.nan
    return time_series, artifacts


def _kept(time_series, artifacts, start):
    """The kept trials of the window at ``start`` and their one-window data."""
    samples = slice(start, start + WINDOW)
    trials = ~artifacts[samples].any(axis=0)
    return trials, time_series[samples][:, trials]


@pytest.mark.parametrize("expectation_type", ["trials_tapers", "tapers", "trials"])
@pytest.mark.parametrize("measure", PER_WINDOW_MEASURES)
def test_every_measure_equals_that_of_each_windows_kept_trials(
    recording, measure, expectation_type
):
    time_series, artifacts = recording
    multitaper = quietly(_multitaper, time_series, artifacts)
    starts = np.arange(len(multitaper.time)) * STEP
    assert to_host(multitaper.valid_windows).sum(axis=1).tolist() == [4, 3, 4, 6, 6]
    connectivity = Connectivity.from_transform(multitaper, expectation_type=expectation_type)
    try:
        masked = _numeric_parts(_call(connectivity, measure))
    except ValueError as error:  # e.g. global_coherence needs the tapers averaged
        _, kept_data = _kept(time_series, artifacts, 0)
        kept = Connectivity.from_transform(
            _multitaper(kept_data), expectation_type=expectation_type
        )
        with pytest.raises(ValueError, match=str(error).split(",")[0][:40]):
            _call(kept, measure)
        return
    rtol = 1e-7 if measure in LOOSE else 1e-10
    for window, start in enumerate(starts):
        trials, kept_data = _kept(time_series, artifacts, start)
        kept = Connectivity.from_transform(
            _multitaper(kept_data), expectation_type=expectation_type
        )
        expected = _numeric_parts(_call(kept, measure))
        assert len(masked) == len(expected)
        for value, reference in zip(masked, expected, strict=True):
            if value.shape[0] != len(starts):  # labels, not windows
                np.testing.assert_array_equal(value, reference)
                continue
            value = value[window]
            if expectation_type == "tapers" and value.shape[0] == N_TRIALS:
                # Trials stay an output axis; the dropped trial windows are
                # tested in test_a_bin_keeping_no_observation_is_nan.
                value = value[trials]
            value, reference = without_noisy_diagonal(measure, value, reference[0])
            part_rtol = rtol
            if ON_GPU and measure in GPU_ITERATIVE and value.dtype.kind == "c":
                # CuPy's svds takes no start vector: its vectors agree to the
                # solver's accuracy (1.6e-6 measured) and only up to phase.
                value, reference = np.abs(value), np.abs(reference)
                part_rtol = 1e-5
            np.testing.assert_allclose(
                value,
                reference,
                rtol=part_rtol,
                atol=1e-12,
                err_msg=f"{measure}, window {window}",
            )


POOLED_MEASURES = (
    "power",
    "magnitude_squared_coherence",
    "imaginary_coherence",
    "phase_locking_value",
    "phase_lag_index",
    "weighted_phase_lag_index",
    "debiased_squared_phase_lag_index",
    "debiased_squared_weighted_phase_lag_index",
    "pairwise_phase_consistency",
    "partial_coherence",
)


@pytest.mark.parametrize("measure", POOLED_MEASURES)
@pytest.mark.parametrize(
    "expectation_type", ["time_trials_tapers", "time_tapers", "time_trials", "time"]
)
def test_pooled_expectations_equal_the_kept_observations_stacked(
    recording, measure, expectation_type
):
    """Pooling time, each bin averages the kept (window, trial, taper)
    observations it covers: the measure of those coefficients stacked as one
    window of one taper. With over 64 observations the phase-lag moments take
    their one-pass path."""
    time_series, artifacts = recording
    multitaper = quietly(_multitaper, time_series, artifacts)
    coefficients = to_host(multitaper.fft())  # (window, trial, taper, fft, signal)
    valid = to_host(multitaper.valid_windows)
    connectivity = Connectivity.from_transform(multitaper, expectation_type=expectation_type)
    value = quietly(getattr(connectivity, measure))
    _, n_trials, n_tapers = coefficients.shape[:3]
    # The output axes each expectation keeps, and the observations of one bin.
    if expectation_type == "time_trials_tapers":
        bins = {(): coefficients[valid]}
    elif expectation_type == "time_tapers":
        bins = {(trial,): coefficients[valid[:, trial], trial] for trial in range(n_trials)}
    elif expectation_type == "time_trials":
        bins = {(taper,): coefficients[:, :, taper][valid] for taper in range(n_tapers)}
    else:
        bins = {
            (trial, taper): coefficients[valid[:, trial], trial, taper]
            for trial in range(n_trials)
            for taper in range(n_tapers)
        }
    if expectation_type == "time_trials_tapers":
        assert connectivity.n_valid_observations[0] == valid.sum() * n_tapers > 64
    for index, observations in bins.items():
        stacked = observations.reshape(1, -1, 1, *coefficients.shape[3:])
        expected = quietly(getattr(Connectivity(stacked), measure))[0]
        np.testing.assert_allclose(
            value[index], expected, rtol=1e-10, atol=1e-13, err_msg=str(index)
        )


# Published definitions over ordered pairs (j, k), j != k, of the kept
# observations X_j = x_j conj(y_j) of a bin.
def _pair_sum(values):
    """sum over j != k of values[j] * values[k], along axis 0."""
    return values.sum(axis=0) ** 2 - (values**2).sum(axis=0)


def _vinck_ppc(cross):
    """Vinck et al. (2010), eq. 14: the mean over pairs of cos(theta_j - theta_k)."""
    n = cross.shape[0]
    phases = np.angle(cross)
    total = sum(np.cos(phases[j] - phases[k]) for j in range(n) for k in range(n) if j != k)
    return total / (n * (n - 1))


def _vinck_pli_squared(cross):
    """Vinck et al. (2011), the unbiased PLI^2: the mean over pairs of
    sign(Im X_j) sign(Im X_k)."""
    n = cross.shape[0]
    return _pair_sum(np.sign(cross.imag)) / (n * (n - 1))


def _vinck_wpli_squared(cross):
    """Vinck et al. (2011), eq. 33, the debiased wPLI^2:
    sum_{j != k} Im X_j Im X_k / sum_{j != k} |Im X_j| |Im X_k|."""
    return _pair_sum(cross.imag) / _pair_sum(np.abs(cross.imag))


DEFINITIONS = {
    "pairwise_phase_consistency": _vinck_ppc,
    "debiased_squared_phase_lag_index": _vinck_pli_squared,
    "debiased_squared_weighted_phase_lag_index": _vinck_wpli_squared,
}


@pytest.mark.parametrize("n_trials", [5, 30], ids=["weighted_path", "one_pass_path"])
@pytest.mark.parametrize("expectation_type", ["trials_tapers", "time_trials_tapers"])
@pytest.mark.parametrize("measure", list(DEFINITIONS))
def test_counting_measures_equal_their_published_definitions(
    measure, expectation_type, n_trials
):
    """Over the kept observations only, at the interior frequencies (0 Hz and
    Nyquist have no imaginary part). 30 trials x 3 tapers reach the one-pass
    phase-lag path (64 observations); 5 trials stay on the weighted one."""
    rng = np.random.default_rng(10)
    time_series = rng.standard_normal((3 * WINDOW, n_trials, 2))
    time_series[:, :, 1] += 0.8 * np.roll(time_series[:, :, 0], 1, axis=0)
    artifacts = np.zeros(time_series.shape[:2], dtype=bool)
    artifacts[10, :2] = True
    artifacts[50, [1, 3]] = True
    artifacts[100, 4] = True
    multitaper = _multitaper(
        time_series, artifacts, n_time_samples_per_step=WINDOW, n_fft_samples=WINDOW
    )
    coefficients = to_host(multitaper.fft())
    valid = to_host(multitaper.valid_windows)
    connectivity = Connectivity.from_transform(multitaper, expectation_type=expectation_type)
    value = getattr(connectivity, measure)()[..., 0, 1]
    interior = slice(1, WINDOW // 2)
    if expectation_type == "trials_tapers":
        bins = {(window,): coefficients[window, valid[window]] for window in range(3)}
    else:
        bins = {(): coefficients[valid]}
    for index, kept in bins.items():
        # (n_kept_trials, n_tapers, fft, signal) -> (n_observations, frequency)
        cross = (kept[..., 0] * kept[..., 1].conj()).reshape(-1, kept.shape[2])
        np.testing.assert_allclose(
            value[index][interior],
            DEFINITIONS[measure](cross[:, interior]),
            rtol=1e-9,
            atol=1e-13,
        )


def _msc(coefficients):
    """Magnitude-squared coherence of signals 0 and 1 from
    (n_observations, frequency, signal) coefficients."""
    cross = np.mean(coefficients[..., 0] * coefficients[..., 1].conj(), axis=0)
    power_0 = np.mean(np.abs(coefficients[..., 0]) ** 2, axis=0)
    power_1 = np.mean(np.abs(coefficients[..., 1]) ** 2, axis=0)
    return np.abs(cross) ** 2 / (power_0 * power_1)


def _hand_jackknife(units, confidence_level=0.95):
    """Leave-one-out jackknife of MSC over ``units`` (n, frequency, signal)."""
    n = units.shape[0]
    estimate = _msc(units)
    replicates = np.stack([_msc(np.delete(units, i, axis=0)) for i in range(n)])
    mean = replicates.mean(axis=0)
    bias_corrected = n * estimate - (n - 1) * mean
    standard_error = np.sqrt((n - 1) / n * ((replicates - mean) ** 2).sum(axis=0))
    half_width = scipy.stats.t.ppf(0.5 + confidence_level / 2, n - 1) * standard_error
    return bias_corrected, standard_error, estimate - half_width, estimate + half_width


@pytest.mark.parametrize("expectation_type", ["trials_tapers", "trials"])
def test_jackknife_equals_a_hand_computed_one_over_the_kept_units(recording, expectation_type):
    """trials_tapers leaves out one kept trial x taper; trials, one kept trial
    of each taper's bin. Identity transformation, Student t bounds."""
    time_series, artifacts = recording
    multitaper = quietly(_multitaper, time_series, artifacts)
    coefficients = to_host(multitaper.fft())
    valid = to_host(multitaper.valid_windows)
    connectivity = Connectivity.from_transform(multitaper, expectation_type=expectation_type)
    result = quietly(
        connectivity.jackknife, "magnitude_squared_coherence", transformation="identity"
    )
    n_nonnegative = N_FFT // 2 + 1
    lower, upper = result.confidence_interval
    for window in range(coefficients.shape[0]):
        kept = coefficients[window, valid[window], :, :n_nonnegative]  # (trial, taper, f, s)
        if expectation_type == "trials_tapers":
            bins = {(window,): kept.reshape(-1, *kept.shape[2:])}
        else:
            bins = {(window, taper): kept[:, taper] for taper in range(kept.shape[1])}
        for index, units in bins.items():
            expected = _hand_jackknife(units)
            observed = (
                result.bias_corrected[index][..., 0, 1],
                result.standard_error[index][..., 0, 1],
                lower[index][..., 0, 1],
                upper[index][..., 0, 1],
            )
            for name, ours, theirs in zip(
                ("bias_corrected", "standard_error", "lower", "upper"),
                observed,
                expected,
                strict=True,
            ):
                np.testing.assert_allclose(
                    ours, theirs, rtol=1e-9, atol=1e-13, err_msg=f"{name} {index}"
                )
            assert np.all(np.asarray(result.n_observations)[index] == units.shape[0])


@pytest.mark.parametrize("measure", PER_WINDOW_MEASURES)
def test_a_bin_keeping_no_observation_is_nan(recording, measure):
    """The transform warns that a window keeping no trial "is NaN in every
    measure" (and the docs say so); with expectation_type='tapers' every
    dropped trial window is such a bin. Integer and boolean parts (labels,
    ranks) are not values and are skipped."""
    time_series, artifacts = recording
    multitaper = quietly(_multitaper, time_series, artifacts)
    valid = to_host(multitaper.valid_windows)
    connectivity = Connectivity.from_transform(multitaper, expectation_type="tapers")
    try:
        parts = _numeric_parts(_call(connectivity, measure))
    except ValueError:
        pytest.skip(f"{measure} does not keep the trial axis")
    for part in parts:
        if part.dtype.kind not in "fc" or part.shape[:2] != valid.shape:
            continue
        values = part[~valid]  # (n_dropped, ...)
        off_diagonal = (
            ~np.eye(N_SIGNALS, dtype=bool)
            if part.ndim >= 4 and part.shape[-2:] == (N_SIGNALS, N_SIGNALS)
            else np.ones(part.shape[-1:], dtype=bool)
        )
        assert np.isnan(values[..., off_diagonal]).all(), (
            f"{measure}: {np.count_nonzero(~np.isnan(values[..., off_diagonal]))} "
            f"finite values in dropped bins, e.g. "
            f"{np.unique(values[..., off_diagonal][~np.isnan(values[..., off_diagonal])])[:3]}"
        )


# Monte Carlo: replicate datasets on the window axis, one window each, 1 Hz
# bins away from 0 Hz and Nyquist (32 Hz).
MC_WINDOW = 64
MC_TRIALS = 8


def _dropped_trials(rng, n_windows, keep_at_least):
    """``(n_windows * MC_WINDOW, MC_TRIALS)`` artifacts: window ``w`` drops
    ``w % (MC_TRIALS - keep_at_least + 1)`` trials chosen at random, one
    artifact sample each."""
    artifacts = np.zeros((n_windows * MC_WINDOW, MC_TRIALS), dtype=bool)
    for window in range(n_windows):
        n_dropped = window % (MC_TRIALS - keep_at_least + 1)
        dropped = rng.choice(MC_TRIALS, size=n_dropped, replace=False)
        artifacts[window * MC_WINDOW + rng.integers(MC_WINDOW), dropped] = True
    return artifacts


N_COUPLED_REPLICATES = 2400
# Every other 1 Hz bin: one taper at NW = 1 has a 1 Hz half-bandwidth.
COUPLED_FREQUENCIES = list(range(4, 29, 2))


@pytest.fixture(scope="module")
def coupled_replicates():
    """Two signals with a lag (nonzero PPC and PLI^2) and one taper, so the
    observations are independent trials; the masked and the all-trial
    Connectivity of the same data."""
    rng = np.random.default_rng(20)
    source = rng.standard_normal((N_COUPLED_REPLICATES * MC_WINDOW, MC_TRIALS))
    lagged = np.roll(source, 1, axis=0) + 1.5 * rng.standard_normal(source.shape)
    time_series = np.stack([source, lagged], axis=-1)
    artifacts = _dropped_trials(rng, N_COUPLED_REPLICATES, keep_at_least=2)
    settings = {
        "n_time_samples_per_window": MC_WINDOW,
        "time_halfbandwidth_product": 1,
        "n_tapers": 1,
    }
    masked = Multitaper(time_series, MC_WINDOW, artifact_samples=artifacts, **settings)
    full = Multitaper(time_series, MC_WINDOW, **settings)
    return masked, Connectivity.from_transform(full)


@pytest.mark.parametrize(
    "measure", ["pairwise_phase_consistency", "debiased_squared_phase_lag_index"]
)
def test_dropping_trials_leaves_unbiased_measures_unbiased_at_a_nonzero_value(
    coupled_replicates, measure, monkeypatch
):
    """PPC and the debiased PLI^2 are unbiased for i.i.d. observations, so the
    paired difference between the masked and the all-trial estimate of each
    replicate has mean zero; with the raw count it would not."""
    masked_transform, full = coupled_replicates
    reference = getattr(full, measure)()[:, COUPLED_FREQUENCIES, 0, 1]
    assert reference.mean() > 0.15  # a nonzero population value

    def paired_mean_and_limit(connectivity):
        values = getattr(connectivity, measure)()[:, COUPLED_FREQUENCIES, 0, 1]
        per_replicate = (values - reference).mean(axis=1)
        return per_replicate.mean(), 5 * per_replicate.std(ddof=1) / np.sqrt(
            N_COUPLED_REPLICATES
        )

    masked = Connectivity.from_transform(masked_transform)
    assert set(masked.n_valid_observations[:, 0].tolist()) == set(range(2, 9))
    mean, limit = paired_mean_and_limit(masked)
    assert abs(mean) < limit, (mean, limit)

    def raw_count(self, measure, dtype):
        return self.n_observations

    monkeypatch.setattr(Connectivity, "_counts_for_correction", raw_count)
    biased, _ = paired_mean_and_limit(Connectivity.from_transform(masked_transform))
    assert abs(biased) > 5 * limit, (biased, limit)


N_NULL_REPLICATES = 1200
# A bandwidth (NW = 2: 2 Hz half-bandwidth) apart.
NULL_FREQUENCIES = [6, 12, 18, 24]


@pytest.fixture(scope="module")
def independent_replicates():
    """Two independent white-noise signals, NW = 2 with three tapers (exactly
    independent for white noise), each window keeping 1 to 8 trials."""
    rng = np.random.default_rng(30)
    time_series = rng.standard_normal((N_NULL_REPLICATES * MC_WINDOW, MC_TRIALS, 2))
    artifacts = _dropped_trials(rng, N_NULL_REPLICATES, keep_at_least=1)
    return Connectivity.from_transform(
        Multitaper(
            time_series,
            MC_WINDOW,
            n_time_samples_per_window=MC_WINDOW,
            time_halfbandwidth_product=2,
            n_tapers=3,
            artifact_samples=artifacts,
        )
    )


@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.2, 0.5])
def test_coherence_pvalues_follow_the_exact_null_at_each_kept_count(
    independent_replicates, alpha
):
    """Under independence MSC of n circular complex Gaussian observations is
    Beta(1, n - 1), so P(p < alpha) = alpha at every count, the smallest (3)
    included. Allowance: at these interior bins the tapers' leakage to 2f makes
    the coefficients non-circular by less than 1e-3 of their variance."""
    connectivity = independent_replicates
    coherency = connectivity.coherency()[:, NULL_FREQUENCIES, 0, 1]
    counts = connectivity.n_valid_observations[:, NULL_FREQUENCIES]
    p_values = coherence_significance_pvalue(coherency, counts)
    rejected = p_values < alpha
    limit = 5 * np.sqrt(alpha * (1 - alpha) / rejected.size)
    assert abs(rejected.mean() - alpha) < limit, (rejected.mean(), limit)
    # The fewest kept, on their own, against their own (wider) limit.
    fewest = counts == 3
    fewest_limit = 5 * np.sqrt(alpha * (1 - alpha) / fewest.sum())
    assert abs(rejected[fewest].mean() - alpha) < fewest_limit
    # With the raw count the test over-rejects by several limits (measured
    # 5.6 at alpha = 0.5, more at smaller alpha).
    raw = coherence_significance_pvalue(coherency, connectivity.n_observations) < alpha
    assert raw.mean() - alpha > 4 * limit


_MNE_REFERENCE = Path(__file__).parent / "reference" / "mne_connectivity_reference.npz"
_MNE_TRANSFORMS = {"identity": np.asarray, "sqrt": np.sqrt, "abs": np.abs}
# Pairwise measures only; psi and cacoh take band or group arguments.
_MNE_PAIRWISE = [name for name in MNE_MEASURES if name not in ("psi", "cacoh")]


def _mne_tapers(n_time_samples, time_halfbandwidth_product):
    """mne-connectivity's periodic, low-bias DPSS weighted by sqrt(eigenvalue)
    (as in tests/test_cross_package.py)."""
    tapers, eigenvalues = dpss(
        n_time_samples,
        time_halfbandwidth_product,
        int(2 * time_halfbandwidth_product),
        sym=False,
        return_ratios=True,
    )
    keep = eigenvalues > 0.9
    return (tapers[keep] * np.sqrt(eigenvalues[keep])[:, np.newaxis]).T


@pytest.fixture(scope="module")
def mne_reference():
    with np.load(_MNE_REFERENCE) as reference:
        return {key: reference[key] for key in reference.files}


@pytest.fixture(scope="module")
def mne_epochs_with_corrupted_trials(mne_reference):
    """Two windows: the first holds the recorded epochs interleaved with five
    corrupted trials (huge values, a NaN run), each marked by one artifact
    sample; the second holds clean data in every trial, so the corrupted
    trials are dropped in the first window only."""
    n_time = int(mne_reference["n_time_samples"])
    epochs = simulate_lagged_broadband(
        mne_reference["lags"],
        float(mne_reference["noise_level"]),
        n_time,
        n_trials=int(mne_reference["n_trials"]),
        random_state=int(mne_reference["seed"]),
    )  # (n_time, n_trials, n_signals)
    rng = np.random.default_rng(40)
    n_recorded = epochs.shape[1]
    corrupted = [0, 7, 8, 20, n_recorded + 4]  # positions in the combined trials
    n_combined = n_recorded + len(corrupted)
    recorded = np.setdiff1d(np.arange(n_combined), corrupted)
    first = np.empty((n_time, n_combined, epochs.shape[2]))
    first[:, recorded] = epochs
    first[:, corrupted] = 1e6 * rng.standard_normal((n_time, len(corrupted), 3))
    first[100:140, corrupted[1]] = np.nan
    second = rng.standard_normal(first.shape)
    time_series = np.concatenate([first, second])
    artifacts = np.zeros(time_series.shape[:2], dtype=bool)
    artifacts[rng.integers(n_time, size=len(corrupted)), corrupted] = True
    artifacts[100:140, corrupted[1]] = True
    return time_series, artifacts, n_combined


@pytest.mark.parametrize("measure", _MNE_PAIRWISE)
def test_masking_corrupted_trials_gives_mne_connectivitys_outputs(
    measure, mne_reference, mne_epochs_with_corrupted_trials
):
    """MNE computed these from the recorded epochs alone; the masked transform
    of those epochs plus marked, corrupted trials must give the same values in
    the first window (every trial kept in the second)."""
    time_series, artifacts, n_combined = mne_epochs_with_corrupted_trials
    method, transform_name, time_halfbandwidth_product, atol = MNE_MEASURES[measure]
    n_time = int(mne_reference["n_time_samples"])
    multitaper = Multitaper(
        time_series,
        sampling_frequency=float(mne_reference["sampling_frequency"]),
        time_halfbandwidth_product=time_halfbandwidth_product,
        tapers=_mne_tapers(n_time, time_halfbandwidth_product),
        n_time_samples_per_window=n_time,
        artifact_samples=artifacts,
    )
    assert to_host(multitaper.valid_windows).sum(axis=1).tolist() == [
        n_combined - 5,
        n_combined,
    ]
    connectivity = Connectivity.from_transform(multitaper)
    frequency_index = np.searchsorted(connectivity.frequencies, mne_reference["freqs"])
    seeds, targets = mne_reference["seeds"], mne_reference["targets"]
    ours = quietly(getattr(connectivity, method))[0][frequency_index][:, seeds, targets].T
    np.testing.assert_allclose(
        _MNE_TRANSFORMS[transform_name](ours), mne_reference[measure], rtol=0, atol=atol
    )


def test_the_documented_pvalue_recipe_handles_a_window_that_keeps_no_trial():
    """The transform allows a window that keeps no trial (with a warning, NaN
    in every measure); the recipe the docs give for the zero-coherence test,
    ``n_valid_observations[..., np.newaxis, np.newaxis]``, should then give
    NaN there, as the measures do, rather than refuse the whole call."""
    rng = np.random.default_rng(60)
    time_series = rng.standard_normal((1000, 4, 2))
    artifacts = np.zeros((1000, 4), dtype=bool)
    artifacts[600, :] = True  # window 2 of 0.5 s at 500 Hz, every trial
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        connectivity = Connectivity.from_transform(
            Multitaper(time_series, 500, time_window_duration=0.5, artifact_samples=artifacts)
        )
        coherency = connectivity.coherency()
    counts = connectivity.n_valid_observations
    assert counts[:, 0].tolist() == [20, 20, 0, 20]
    p_values = coherence_significance_pvalue(coherency, counts[..., np.newaxis, np.newaxis])
    assert np.isnan(p_values[2]).all()
    assert np.isfinite(p_values[[0, 1, 3]][..., 0, 1]).all()
