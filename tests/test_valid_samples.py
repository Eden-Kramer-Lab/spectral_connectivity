"""``Multitaper(valid_samples=...)``: dropping the trial windows that contain
invalid samples.

A trial's window is used only if all its samples are valid. Every measure,
those that count observations and the jackknife included, must equal its value
for the data with the dropped trial windows physically removed; the counting
measures stay unbiased and the coherence test keeps its size.
"""

import warnings

import numpy as np
import pytest

from spectral_connectivity import Connectivity, Multitaper, multitaper_connectivity
from spectral_connectivity.statistics import coherence_significance_pvalue
from tests._backend_helpers import to_host

SAMPLING_FREQUENCY = 200.0
WINDOW = 50  # samples
# Plain averages, the normalized and phase-lag families, the measures that
# count observations, and those that invert or factor the cross-spectrum.
MEASURES = (
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
    "pairwise_spectral_granger_prediction",
)
FACTORIZED = frozenset({"pairwise_spectral_granger_prediction"})
JACKKNIFED = ("magnitude_squared_coherence", "pairwise_phase_consistency")


def _quietly(function, *args, **kwargs):
    with warnings.catch_warnings():
        # Few trials for three signals: the rank warnings are expected.
        warnings.simplefilter("ignore", UserWarning)
        return function(*args, **kwargs)


@pytest.fixture(scope="module")
def recording():
    """Five trials of three signals with artifacts (NaN) in some trial windows."""
    rng = np.random.default_rng(0)
    time_series = rng.standard_normal((4 * WINDOW, 5, 3))
    time_series[:, :, 1] += 0.6 * np.roll(time_series[:, :, 0], 2, axis=0)
    valid = np.ones(time_series.shape[:2], dtype=bool)
    valid[10:14, [0, 3]] = False  # window 0 loses trials 0 and 3
    valid[60, 1] = False  # window 1 loses trial 1
    valid[170:200, [2]] = False  # window 3 loses trial 2
    time_series[~valid] = np.nan
    return time_series, valid


@pytest.mark.parametrize("step", [WINDOW, WINDOW // 2], ids=["adjacent", "overlapping"])
def test_every_measure_equals_that_of_the_kept_trial_windows(recording, step):
    time_series, valid = recording
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # overlap; few trials
        multitaper = Multitaper(
            time_series,
            SAMPLING_FREQUENCY,
            time_halfbandwidth_product=2,
            n_time_samples_per_window=WINDOW,
            n_time_samples_per_step=step,
            valid_samples=valid,
        )
        connectivity = Connectivity.from_transform(multitaper)
    starts = np.arange(0, time_series.shape[0] - WINDOW + 1, step)
    assert len(starts) == len(multitaper.time)
    for measure in MEASURES:
        masked = _quietly(getattr(connectivity, measure))
        rtol = 1e-7 if measure in FACTORIZED else 1e-10
        for window, start in enumerate(starts):
            samples = slice(start, start + WINDOW)
            trials = valid[samples].all(axis=0)
            kept = Connectivity.from_transform(
                Multitaper(
                    time_series[samples, trials],
                    SAMPLING_FREQUENCY,
                    time_halfbandwidth_product=2,
                )
            )
            np.testing.assert_allclose(
                masked[window],
                _quietly(getattr(kept, measure))[0],
                rtol=rtol,
                atol=1e-13,
                err_msg=f"{measure}, window {window}",
            )
            assert (connectivity.n_valid_observations[window] == kept.n_observations).all()


def test_jackknife_equals_that_of_the_kept_trial_windows(recording):
    time_series, valid = recording
    multitaper = Multitaper(
        time_series,
        SAMPLING_FREQUENCY,
        time_halfbandwidth_product=2,
        n_time_samples_per_window=WINDOW,
        valid_samples=valid,
    )
    connectivity = Connectivity.from_transform(multitaper)
    for method in JACKKNIFED:
        masked = _quietly(connectivity.jackknife, method)
        for window in range(4):
            samples = slice(window * WINDOW, (window + 1) * WINDOW)
            trials = valid[samples].all(axis=0)
            kept = _quietly(
                Connectivity.from_transform(
                    Multitaper(
                        time_series[samples, trials],
                        SAMPLING_FREQUENCY,
                        time_halfbandwidth_product=2,
                    )
                ).jackknife,
                method,
            )
            for field in ("bias_corrected", "standard_error"):
                np.testing.assert_allclose(
                    getattr(masked, field)[window],
                    getattr(kept, field)[0],
                    rtol=1e-9,
                    atol=1e-13,
                    err_msg=f"{method} {field}, window {window}",
                )
            lower, upper = masked.confidence_interval
            kept_lower, kept_upper = kept.confidence_interval
            np.testing.assert_allclose(lower[window], kept_lower[0], rtol=1e-9, atol=1e-13)
            np.testing.assert_allclose(upper[window], kept_upper[0], rtol=1e-9, atol=1e-13)


def test_without_time_resolution_the_kept_trials_are_used(recording):
    time_series, valid = recording
    trials = valid.all(axis=0)
    masked = Connectivity.from_transform(
        Multitaper(time_series, SAMPLING_FREQUENCY, valid_samples=valid)
    )
    kept = Connectivity.from_transform(Multitaper(time_series[:, trials], SAMPLING_FREQUENCY))
    assert trials.sum() == 1  # only trial 4 is clean throughout
    for measure in ("power", "magnitude_squared_coherence"):
        np.testing.assert_allclose(
            _quietly(getattr(masked, measure)),
            _quietly(getattr(kept, measure)),
            rtol=1e-10,
        )


def test_finite_marks_the_nonfinite_samples_invalid(recording):
    time_series, valid = recording
    finite = Multitaper(
        time_series,
        SAMPLING_FREQUENCY,
        n_time_samples_per_window=WINDOW,
        valid_samples="finite",
    )
    mask = Multitaper(
        time_series, SAMPLING_FREQUENCY, n_time_samples_per_window=WINDOW, valid_samples=valid
    )
    np.testing.assert_array_equal(to_host(finite.valid_samples), valid)
    np.testing.assert_array_equal(to_host(finite.fft()), to_host(mask.fft()))
    assert np.isfinite(to_host(finite.fft())).all()
    np.testing.assert_array_equal(
        to_host(finite.valid_windows),
        [[0, 1, 1, 0, 1], [1, 0, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 0, 1, 1]],
    )


def test_a_per_signal_mask_invalidates_the_sample_for_every_signal(recording):
    time_series, valid = recording
    per_signal = np.ones(time_series.shape, dtype=bool)
    per_signal[..., 2] = valid  # only signal 2 marks the artifacts
    shared = Multitaper(
        time_series, SAMPLING_FREQUENCY, n_time_samples_per_window=WINDOW, valid_samples=valid
    )
    reduced = Multitaper(
        time_series,
        SAMPLING_FREQUENCY,
        n_time_samples_per_window=WINDOW,
        valid_samples=per_signal,
    )
    np.testing.assert_array_equal(to_host(reduced.valid_samples), valid)
    np.testing.assert_array_equal(to_host(reduced.fft()), to_host(shared.fft()))


def test_an_all_valid_mask_is_the_transform_without_one():
    rng = np.random.default_rng(1)
    time_series = rng.standard_normal((200, 3, 2))
    plain = Multitaper(time_series, SAMPLING_FREQUENCY, n_time_samples_per_window=WINDOW)
    masked = Multitaper(
        time_series,
        SAMPLING_FREQUENCY,
        n_time_samples_per_window=WINDOW,
        valid_samples=np.ones((200, 3), dtype=bool),
    )
    assert masked.observation_weights is None
    np.testing.assert_array_equal(to_host(masked.fft()), to_host(plain.fft()))
    assert (
        Multitaper(time_series, SAMPLING_FREQUENCY, valid_samples="finite").observation_weights
        is None
    )


def test_observation_weights_drop_every_taper_of_a_dropped_window(recording):
    time_series, valid = recording
    multitaper = Multitaper(
        time_series, SAMPLING_FREQUENCY, n_time_samples_per_window=WINDOW, valid_samples=valid
    )
    weights = to_host(multitaper.observation_weights)
    assert weights.shape == (4, 5, multitaper.n_tapers, multitaper.n_fft_samples, 1)
    expected = to_host(multitaper.valid_windows)[:, :, None, None, None]
    np.testing.assert_array_equal(weights, np.broadcast_to(expected, weights.shape))


def test_nan_at_valid_samples_still_warns(recording):
    time_series, valid = recording
    time_series = time_series.copy()
    time_series[120, 0, 1] = np.nan  # in window 2, which every trial keeps
    with pytest.warns(UserWarning, match="1 NaN or infinite value at samples valid_samples"):
        Multitaper(time_series, SAMPLING_FREQUENCY, valid_samples=valid)


def test_nan_covered_by_the_mask_does_not_warn(recording):
    time_series, valid = recording
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        multitaper = Multitaper(
            time_series,
            SAMPLING_FREQUENCY,
            n_time_samples_per_window=WINDOW,
            valid_samples=valid,
        )
        Connectivity.from_transform(multitaper).magnitude_squared_coherence()


def test_unmasked_nan_warning_suggests_valid_samples():
    time_series = np.random.default_rng(2).standard_normal((200, 2, 2))
    time_series[5, 0, 0] = np.nan
    with pytest.warns(UserWarning, match="valid_samples='finite' drops every"):
        Multitaper(time_series, SAMPLING_FREQUENCY)


def test_a_window_with_no_valid_trial_warns_and_is_nan():
    time_series = np.random.default_rng(3).standard_normal((200, 2, 2))
    valid = np.ones((200, 2), dtype=bool)
    valid[60, :] = False  # window 1, both trials
    with pytest.warns(
        UserWarning, match=r"leaves 1 of 4 time windows \(0-based indices \[1\]\)"
    ):
        multitaper = Multitaper(
            time_series,
            SAMPLING_FREQUENCY,
            n_time_samples_per_window=WINDOW,
            valid_samples=valid,
        )
    power = Connectivity.from_transform(multitaper).power()
    assert np.isnan(power[1]).all()
    assert np.isfinite(power[[0, 2, 3]]).all()


def test_a_mask_leaving_no_valid_trial_window_raises():
    time_series = np.random.default_rng(4).standard_normal((100, 2, 2))
    valid = np.ones((100, 2), dtype=bool)
    valid[[10, 60], :] = False
    with pytest.raises(ValueError, match="leaves no valid trial window"):
        Multitaper(
            time_series,
            SAMPLING_FREQUENCY,
            n_time_samples_per_window=WINDOW,
            valid_samples=valid,
        )


@pytest.mark.parametrize(
    ("valid_samples", "error", "match"),
    [
        ("nan", ValueError, "must be a boolean array, 'finite', or None"),
        (np.ones((100, 2)), TypeError, "must be a boolean array .* got dtype float64"),
        (np.ones((100,), dtype=bool), ValueError, r"has shape \(100,\)"),
        (np.ones((100, 3), dtype=bool), ValueError, r"must be \(n_time_samples, n_trials\)"),
        (np.ones((100, 2, 3), dtype=bool), ValueError, r"has shape \(100, 2, 3\)"),
    ],
)
def test_invalid_valid_samples_are_rejected(valid_samples, error, match):
    time_series = np.random.default_rng(5).standard_normal((100, 2, 2))
    with pytest.raises(error, match=match):
        Multitaper(time_series, SAMPLING_FREQUENCY, valid_samples=valid_samples)


def test_a_labeled_mask_is_read_by_position():
    xr = pytest.importorskip("xarray")
    time_series = np.random.default_rng(6).standard_normal((100, 2, 2))
    valid = np.ones((100, 2), dtype=bool)
    labeled = Multitaper(
        time_series,
        SAMPLING_FREQUENCY,
        valid_samples=xr.DataArray(valid, dims=("time", "trial")),
    )
    np.testing.assert_array_equal(to_host(labeled.valid_samples), valid)
    with pytest.raises(ValueError, match="'trial' is on the time axis"):
        Multitaper(
            time_series,
            SAMPLING_FREQUENCY,
            valid_samples=xr.DataArray(valid.T.copy(), dims=("trial", "time")),
        )


def test_valid_samples_is_immutable_and_reported(recording):
    time_series, valid = recording
    multitaper = Multitaper(
        time_series, SAMPLING_FREQUENCY, n_time_samples_per_window=WINDOW, valid_samples=valid
    )
    with pytest.raises(AttributeError, match="immutable"):
        multitaper.valid_samples = None
    returned = multitaper.valid_samples
    assert not returned.flags.writeable
    assert "valid_samples=<mask> (16 of 20 trial windows kept)" in repr(multitaper)
    assert "Valid samples:                 <mask> (16 of 20 trial windows kept)" in (
        multitaper.summarize_parameters()
    )
    finite = Multitaper(time_series, SAMPLING_FREQUENCY, valid_samples="finite")
    assert "valid_samples='finite' (1 of 5 trial windows kept)" in repr(finite)
    assert "valid_samples" not in repr(Multitaper(time_series[:, [4]], SAMPLING_FREQUENCY))


def test_provenance_records_the_kind_and_what_it_dropped(recording):
    time_series, valid = recording
    metadata = Multitaper(
        time_series, SAMPLING_FREQUENCY, n_time_samples_per_window=WINDOW, valid_samples=valid
    )._provenance_metadata()
    assert metadata["valid_samples"] == "mask"
    assert metadata["n_invalid_samples"] == int((~valid).sum())
    assert metadata["n_dropped_trial_windows"] == 4
    assert (
        "valid_samples"
        not in Multitaper(time_series[:, [4]], SAMPLING_FREQUENCY)._provenance_metadata()
    )


def test_the_wrapper_takes_valid_samples_in_transform_kwargs(recording):
    time_series, valid = recording
    result = _quietly(
        multitaper_connectivity,
        time_series,
        SAMPLING_FREQUENCY,
        method="pairwise_phase_consistency",
        transform_kwargs={"n_time_samples_per_window": WINDOW, "valid_samples": valid},
    )
    direct = _quietly(
        Connectivity.from_transform(
            Multitaper(
                time_series,
                SAMPLING_FREQUENCY,
                n_time_samples_per_window=WINDOW,
                valid_samples=valid,
            )
        ).pairwise_phase_consistency
    )
    np.testing.assert_allclose(result.values, direct, rtol=1e-12, equal_nan=True)
    assert result.attrs["n_observations"] == 25  # the raw count
    assert '"valid_samples":"mask"' in result.attrs["transform_parameters_json"]


@pytest.mark.parametrize("shared", [True, False], ids=["shared", "per_signal"])
def test_the_wrapper_reads_a_mask_of_2d_input_in_its_layout(shared):
    """(n_time, n_signals) input: the mask is (n_time,) or (n_time, n_signals)."""
    time_series = np.random.default_rng(7).standard_normal((200, 2))
    valid = np.ones(200 if shared else (200, 2), dtype=bool)
    valid[60] = False
    with pytest.warns(UserWarning, match="leaves 1 of 4 time windows"):
        result = multitaper_connectivity(
            time_series,
            SAMPLING_FREQUENCY,
            method="power",
            transform_kwargs={"n_time_samples_per_window": WINDOW, "valid_samples": valid},
        )
    power = result.values
    assert np.isnan(power[1]).all()
    assert np.isfinite(power[[0, 2, 3]]).all()


def test_the_wrapper_transposes_a_labeled_mask_like_its_labeled_input(recording):
    xr = pytest.importorskip("xarray")
    time_series, valid = recording
    labeled = xr.DataArray(
        np.transpose(time_series, (2, 0, 1)), dims=("channel", "time", "trial")
    )
    labeled_mask = xr.DataArray(valid.T.copy(), dims=("trial", "time"))
    transform_kwargs = {"n_time_samples_per_window": WINDOW}
    from_labels = _quietly(
        multitaper_connectivity,
        labeled,
        SAMPLING_FREQUENCY,
        method="power",
        transform_kwargs={**transform_kwargs, "valid_samples": labeled_mask},
    )
    from_arrays = _quietly(
        multitaper_connectivity,
        time_series,
        SAMPLING_FREQUENCY,
        method="power",
        transform_kwargs={**transform_kwargs, "valid_samples": valid},
    )
    np.testing.assert_array_equal(from_labels.values, from_arrays.values)


# Monte Carlo: replicate datasets on the window axis, one window each; every
# window drops a different number of its 8 trials, up to 7.
N_REPLICATES = 800
N_TRIALS = 8
MC_WINDOW = 64
# Interior bins at least a bandwidth (4 Hz at NW = 2) apart, away from 0 Hz
# and Nyquist, where the eigencoefficients of white noise are independent.
MC_FREQUENCIES = [4, 8, 12, 16, 20, 24, 28]


@pytest.fixture(scope="module")
def independent_signals():
    """Two independent white-noise signals; window ``w`` drops ``w % 8`` trials."""
    rng = np.random.default_rng(10)
    time_series = rng.standard_normal((N_REPLICATES * MC_WINDOW, N_TRIALS, 2))
    valid = np.ones(time_series.shape[:2], dtype=bool)
    for window in range(N_REPLICATES):
        dropped = rng.choice(N_TRIALS, size=window % 8, replace=False)
        valid[window * MC_WINDOW + MC_WINDOW // 2, dropped] = False
    multitaper = Multitaper(
        time_series,
        sampling_frequency=MC_WINDOW,  # 1 Hz bins
        time_halfbandwidth_product=2,
        n_tapers=3,
        n_time_samples_per_window=MC_WINDOW,
        valid_samples=valid,
    )
    return Connectivity.from_transform(multitaper)


def _replicate_mean_and_limit(values):
    """The mean over replicates (windows) of each window's mean over the
    tested frequencies, and five standard errors of it."""
    per_replicate = values[:, MC_FREQUENCIES, 0, 1].mean(axis=1)
    return per_replicate.mean(), 5 * per_replicate.std(ddof=1) / np.sqrt(N_REPLICATES)


@pytest.mark.parametrize(
    "measure",
    [
        "pairwise_phase_consistency",
        "debiased_squared_phase_lag_index",
        "debiased_squared_weighted_phase_lag_index",
    ],
)
def test_counting_measures_are_unbiased_with_dropped_trials(
    independent_signals, measure, monkeypatch
):
    """Their mean under independence is 0 with each window's kept count; with
    the raw count it would lie many limits away."""
    counts = independent_signals.n_valid_observations[:, 0]
    assert sorted(set(counts.tolist())) == [3, 6, 9, 12, 15, 18, 21, 24]
    mean, limit = _replicate_mean_and_limit(getattr(independent_signals, measure)())
    assert abs(mean) < limit, (mean, limit)

    def raw_count(self, measure):
        return self.n_observations

    monkeypatch.setattr(Connectivity, "_counts_for_correction", raw_count)
    independent_signals.clear_cache()
    biased, _ = _replicate_mean_and_limit(getattr(independent_signals, measure)())
    # Four limits: twenty standard errors.
    assert abs(biased) > 4 * limit, (biased, limit)


def test_coherence_test_keeps_its_size_with_dropped_trials(independent_signals):
    coherency = independent_signals.coherency()[:, MC_FREQUENCIES, 0, 1]
    counts = independent_signals.n_valid_observations[:, MC_FREQUENCIES]
    rejected = coherence_significance_pvalue(coherency, counts) < 0.05
    size = rejected.mean()
    limit = 5 * np.sqrt(0.05 * 0.95 / rejected.size)
    assert abs(size - 0.05) < limit, (size, limit)
    raw = coherence_significance_pvalue(coherency, independent_signals.n_observations) < 0.05
    assert raw.mean() - 0.05 > 10 * limit, raw.mean()
