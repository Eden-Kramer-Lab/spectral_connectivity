"""``Multitaper(artifact_samples=...)``: dropping the trial windows that
contain artifact samples (``True`` in the mask).

A trial's window is used only if none of its samples is an artifact. The
oracles (test_artifact_samples_oracles.py) test every measure against the data
with the dropped trial windows physically removed, and the coherence test's
null; these test the mask's forms and messages, the wrapper's layouts, the
counting measures' null mean, the STFT, and delay's per-window selection.
"""

import warnings

import numpy as np
import pytest

from spectral_connectivity import (
    Connectivity,
    Multitaper,
    ShortTimeFourierTransform,
    multitaper_connectivity,
)
from spectral_connectivity.connectivity import _find_significant_frequencies
from tests._artifact_helpers import COUNTING_MEASURES, quietly
from tests._backend_helpers import ON_GPU, to_host

SAMPLING_FREQUENCY = 200.0
WINDOW = 50  # samples
# The oracles test the jackknife of coherence against one computed by hand;
# this tests a counting measure with the default transformation.
JACKKNIFED = ("pairwise_phase_consistency",)


def _multitaper(time_series, artifacts, **kwargs):
    """A Multitaper of ``WINDOW``-sample windows leaving out ``artifacts``."""
    kwargs.setdefault("n_time_samples_per_window", WINDOW)
    return Multitaper(time_series, SAMPLING_FREQUENCY, artifact_samples=artifacts, **kwargs)


def _kept_window(time_series, artifacts, start, transform=Multitaper, **kwargs):
    """The Connectivity of the window at ``start`` with its artifact trials
    physically removed, one window of the whole of it."""
    samples = slice(start, start + WINDOW)
    trials = ~artifacts[samples].any(axis=0)
    return Connectivity.from_transform(
        transform(time_series[samples, trials], SAMPLING_FREQUENCY, **kwargs)
    )


@pytest.fixture(scope="module")
def recording():
    """Five trials of three signals with artifacts (NaN) in some trial windows."""
    rng = np.random.default_rng(0)
    time_series = rng.standard_normal((4 * WINDOW, 5, 3))
    time_series[:, :, 1] += 0.6 * np.roll(time_series[:, :, 0], 2, axis=0)
    artifacts = np.zeros(time_series.shape[:2], dtype=bool)
    artifacts[10:14, [0, 3]] = True  # window 0 loses trials 0 and 3
    artifacts[60, 1] = True  # window 1 loses trial 1
    artifacts[170:200, [2]] = True  # window 3 loses trial 2
    time_series[artifacts] = np.nan
    return time_series, artifacts


def test_jackknife_equals_that_of_the_kept_trial_windows(recording):
    time_series, artifacts = recording
    connectivity = Connectivity.from_transform(
        _multitaper(time_series, artifacts, time_halfbandwidth_product=2)
    )
    for method in JACKKNIFED:
        masked = quietly(connectivity.jackknife, method)
        for window in range(4):
            kept = quietly(
                _kept_window(
                    time_series, artifacts, window * WINDOW, time_halfbandwidth_product=2
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


def test_nonfinite_marks_the_nonfinite_samples_as_artifacts(recording):
    time_series, artifacts = recording
    nonfinite = _multitaper(time_series, "nonfinite")
    mask = _multitaper(time_series, artifacts)
    np.testing.assert_array_equal(to_host(nonfinite.artifact_samples), artifacts)
    np.testing.assert_array_equal(to_host(nonfinite.fft()), to_host(mask.fft()))
    assert np.isfinite(to_host(nonfinite.fft())).all()
    # The output marks what is kept.
    np.testing.assert_array_equal(
        to_host(nonfinite.valid_windows),
        [[0, 1, 1, 0, 1], [1, 0, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 0, 1, 1]],
    )


def test_a_per_signal_artifact_leaves_the_sample_out_of_every_signal(recording):
    time_series, artifacts = recording
    per_signal = np.zeros(time_series.shape, dtype=bool)
    per_signal[..., 2] = artifacts  # only signal 2 marks the artifacts
    shared = _multitaper(time_series, artifacts)
    reduced = _multitaper(time_series, per_signal)
    np.testing.assert_array_equal(to_host(reduced.artifact_samples), artifacts)
    np.testing.assert_array_equal(to_host(reduced.fft()), to_host(shared.fft()))


def test_a_mask_without_artifacts_is_the_transform_without_one():
    rng = np.random.default_rng(1)
    time_series = rng.standard_normal((200, 3, 2))
    plain = Multitaper(time_series, SAMPLING_FREQUENCY, n_time_samples_per_window=WINDOW)
    masked = Multitaper(
        time_series,
        SAMPLING_FREQUENCY,
        n_time_samples_per_window=WINDOW,
        artifact_samples=np.zeros((200, 3), dtype=bool),
    )
    assert masked.observation_weights is None
    np.testing.assert_array_equal(to_host(masked.fft()), to_host(plain.fft()))
    assert (
        Multitaper(
            time_series, SAMPLING_FREQUENCY, artifact_samples="nonfinite"
        ).observation_weights
        is None
    )


def test_observation_weights_drop_every_taper_of_a_dropped_window(recording):
    time_series, artifacts = recording
    multitaper = _multitaper(time_series, artifacts)
    weights = to_host(multitaper.observation_weights)
    assert weights.shape == (4, 5, multitaper.n_tapers, multitaper.n_fft_samples, 1)
    expected = to_host(multitaper.valid_windows)[:, :, None, None, None]
    np.testing.assert_array_equal(weights, np.broadcast_to(expected, weights.shape))


def test_nan_outside_the_artifacts_still_warns(recording):
    time_series, artifacts = recording
    time_series = time_series.copy()
    time_series[120, 0, 1] = np.nan  # in window 2, which every trial keeps
    with pytest.warns(
        UserWarning,
        match="1 NaN or infinite value in the trial windows artifact_samples keeps",
    ):
        _multitaper(time_series, artifacts)


def test_nan_covered_by_the_mask_does_not_warn(recording):
    time_series, artifacts = recording
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        multitaper = _multitaper(time_series, artifacts)
        Connectivity.from_transform(multitaper).magnitude_squared_coherence()


def test_nan_only_in_dropped_trial_windows_does_not_warn(recording):
    """An unmarked NaN in a trial window the mask drops reaches no estimate."""
    time_series, artifacts = recording
    with_nan = time_series.copy()
    with_nan[20, 0, 1] = np.nan  # window 0 drops trial 0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        coherence = Connectivity.from_transform(
            _multitaper(with_nan, artifacts)
        ).magnitude_squared_coherence()
    clean = Connectivity.from_transform(_multitaper(time_series, artifacts))
    np.testing.assert_array_equal(coherence, clean.magnitude_squared_coherence())


def test_unmasked_nan_warning_suggests_artifact_samples():
    time_series = np.random.default_rng(2).standard_normal((200, 2, 2))
    time_series[5, 0, 0] = np.nan
    with pytest.warns(UserWarning, match="NaN or infinite") as record:
        Multitaper(time_series, SAMPLING_FREQUENCY)
    message = str(record[0].message)
    # The first remedy named.
    assert "raise a ValueError instead). Pass artifact_samples='nonfinite'" in message


def test_a_window_keeping_no_trial_warns_and_is_nan():
    time_series = np.random.default_rng(3).standard_normal((200, 2, 2))
    artifacts = np.zeros((200, 2), dtype=bool)
    artifacts[60, :] = True  # window 1, both trials
    with pytest.warns(
        UserWarning, match=r"leaves 1 of 4 time windows \(0-based indices \[1\]\).*50 samples"
    ):
        multitaper = _multitaper(time_series, artifacts)
    power = Connectivity.from_transform(multitaper).power()
    assert np.isnan(power[1]).all()
    assert np.isfinite(power[[0, 2, 3]]).all()


def test_a_time_only_mask_is_shared_by_every_trial_and_signal():
    time_series = np.random.default_rng(12).standard_normal((200, 3, 2))
    shared = np.zeros(200, dtype=bool)
    shared[60] = True  # window 1, in every trial
    with pytest.warns(UserWarning, match="leaves 1 of 4 time windows"):
        from_time = _multitaper(time_series, shared)
    expected = np.broadcast_to(shared[:, None], (200, 3))
    np.testing.assert_array_equal(to_host(from_time.artifact_samples), expected)


def test_dropping_most_trial_windows_warns_and_suggests_an_inverted_mask():
    """A mask true at the valid samples, passed as is, keeps little: here the
    first window, clean but for the mask."""
    time_series = np.random.default_rng(8).standard_normal((200, 2, 2))
    artifacts = np.zeros((200, 2), dtype=bool)
    artifacts[60:65, 0] = True
    inverted = ~(artifacts | (np.arange(200)[:, None] < 50))
    with pytest.warns(UserWarning, match="artifact_samples") as record:
        _multitaper(time_series, inverted)
    messages = [str(warning.message) for warning in record]
    assert any(
        "drops 75% of trial windows (2 of 8 kept)" in message and "pass ~mask" in message
        for message in messages
    )
    # Passed the right way round, it drops one trial window and is quiet.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _multitaper(time_series, artifacts)


def test_a_mask_leaving_no_trial_window_raises():
    time_series = np.random.default_rng(4).standard_normal((100, 2, 2))
    artifacts = np.zeros((100, 2), dtype=bool)
    artifacts[[10, 60], :] = True
    with pytest.raises(ValueError, match=r"leaves no trial window.*pass ~mask"):
        _multitaper(time_series, artifacts)


@pytest.mark.parametrize(
    ("artifact_samples", "error", "match"),
    [
        ("finite", ValueError, "must be a boolean array, 'nonfinite', or None"),
        (np.ones((100, 2)), TypeError, "must be a boolean array .* got dtype float64"),
        (np.ones((99,), dtype=bool), ValueError, r"has shape \(99,\)"),
        (np.ones((100, 3), dtype=bool), ValueError, r"has shape \(100, 3\); it must be"),
        (np.ones((100, 2, 3), dtype=bool), ValueError, r"has shape \(100, 2, 3\)"),
    ],
)
def test_invalid_artifact_samples_are_rejected(artifact_samples, error, match):
    time_series = np.random.default_rng(5).standard_normal((100, 2, 2))
    with pytest.raises(error, match=match):
        Multitaper(time_series, SAMPLING_FREQUENCY, artifact_samples=artifact_samples)


def test_a_labeled_mask_is_read_by_position():
    xr = pytest.importorskip("xarray")
    time_series = np.random.default_rng(6).standard_normal((100, 2, 2))
    artifacts = np.zeros((100, 2), dtype=bool)
    artifacts[3, 1] = True
    labeled = Multitaper(
        time_series,
        SAMPLING_FREQUENCY,
        artifact_samples=xr.DataArray(artifacts, dims=("time", "trial")),
    )
    np.testing.assert_array_equal(to_host(labeled.artifact_samples), artifacts)
    with pytest.raises(ValueError, match="'trial' is on the time axis"):
        Multitaper(
            time_series,
            SAMPLING_FREQUENCY,
            artifact_samples=xr.DataArray(artifacts.T.copy(), dims=("trial", "time")),
        )


def test_artifact_samples_is_immutable_and_reported(recording):
    time_series, artifacts = recording
    multitaper = _multitaper(time_series, artifacts)
    with pytest.raises(AttributeError, match="immutable"):
        multitaper.artifact_samples = None
    returned = multitaper.artifact_samples
    if ON_GPU:  # CuPy arrays have no writeable flag; the copy alone isolates them
        returned[...] = True
        assert not to_host(multitaper.artifact_samples).all()
    else:
        assert not returned.flags.writeable
    assert "artifact_samples=<mask> (16 of 20 trial windows kept)" in repr(multitaper)
    assert "Artifact samples:              <mask> (16 of 20 trial windows kept)" in (
        multitaper.summarize_parameters()
    )
    with pytest.warns(UserWarning, match="drops 80% of trial windows"):
        nonfinite = Multitaper(time_series, SAMPLING_FREQUENCY, artifact_samples="nonfinite")
    assert "artifact_samples='nonfinite' (1 of 5 trial windows kept)" in repr(nonfinite)
    assert "artifact_samples" not in repr(Multitaper(time_series[:, [4]], SAMPLING_FREQUENCY))


def test_provenance_records_the_kind_and_what_it_dropped(recording):
    time_series, artifacts = recording
    metadata = _multitaper(time_series, artifacts)._provenance_metadata()
    assert metadata["artifact_samples"] == "mask"
    assert metadata["n_artifact_samples"] == int(artifacts.sum())
    assert metadata["n_dropped_trial_windows"] == 4
    assert (
        "artifact_samples"
        not in Multitaper(time_series[:, [4]], SAMPLING_FREQUENCY)._provenance_metadata()
    )


def test_the_wrapper_takes_artifact_samples_in_transform_kwargs(recording):
    time_series, artifacts = recording
    result = quietly(
        multitaper_connectivity,
        time_series,
        SAMPLING_FREQUENCY,
        method="pairwise_phase_consistency",
        transform_kwargs={"n_time_samples_per_window": WINDOW, "artifact_samples": artifacts},
    )
    direct = quietly(
        Connectivity.from_transform(
            _multitaper(time_series, artifacts)
        ).pairwise_phase_consistency
    )
    np.testing.assert_allclose(result.values, direct, rtol=1e-12, equal_nan=True)
    assert result.attrs["n_observations"] == 25  # the raw count
    assert '"artifact_samples":"mask"' in result.attrs["transform_parameters_json"]


@pytest.mark.parametrize("shared", [True, False], ids=["shared", "per_signal"])
def test_the_wrapper_reads_a_mask_of_2d_input_in_its_layout(shared):
    """(n_time, n_signals) input: the mask is (n_time,) or (n_time, n_signals)."""
    time_series = np.random.default_rng(7).standard_normal((200, 2))
    artifacts = np.zeros(200 if shared else (200, 2), dtype=bool)
    artifacts[60] = True
    with pytest.warns(UserWarning, match="leaves 1 of 4 time windows"):
        result = multitaper_connectivity(
            time_series,
            SAMPLING_FREQUENCY,
            method="power",
            transform_kwargs={
                "n_time_samples_per_window": WINDOW,
                "artifact_samples": artifacts,
            },
        )
    power = result.values
    assert np.isnan(power[1]).all()
    assert np.isfinite(power[[0, 2, 3]]).all()


@pytest.mark.parametrize(
    ("shape", "match"),
    [
        ((200, 4), r"has shape \(200, 4\); for the 2-D time_series \(200, 2\) .* \(200,\)"),
        ((2, 200), r"Did you transpose it\? Pass artifact_samples.T"),
    ],
)
def test_the_wrapper_states_the_mask_shapes_of_2d_input(shape, match):
    time_series = np.random.default_rng(9).standard_normal((200, 2))
    with pytest.raises(ValueError, match=match):
        multitaper_connectivity(
            time_series,
            SAMPLING_FREQUENCY,
            method="power",
            transform_kwargs={"artifact_samples": np.zeros(shape, dtype=bool)},
        )


def test_delay_significance_agrees_window_by_window(recording):
    """Each window's zero-coherence test, corrected within that window, selects
    the frequencies the kept trials alone select."""
    time_series, artifacts = recording
    connectivity = Connectivity.from_transform(
        _multitaper(time_series, artifacts, time_halfbandwidth_product=2)
    )
    coherency = connectivity.coherency()[..., [0], [1]]
    counts = connectivity.n_valid_observations[..., np.newaxis]
    for window in range(4):
        kept = _kept_window(
            time_series, artifacts, window * WINDOW, time_halfbandwidth_product=2
        )
        selected = _find_significant_frequencies(coherency[[window]], counts[[window]])
        assert selected.any()
        np.testing.assert_array_equal(
            selected,
            _find_significant_frequencies(
                kept.coherency()[..., [0], [1]], kept.n_observations
            ),
        )


@pytest.mark.parametrize(
    ("dims", "index"),
    [
        (("time",), (60,)),
        (("time", "trial"), (60, 1)),
        (("trial", "time"), (1, 60)),
        (("channel", "time"), (2, 60)),
    ],
)
def test_the_wrapper_shares_a_labeled_mask_over_the_dims_it_lacks(dims, index):
    """A DataArray mask of a (time, trial, channel) series that lacks the
    trial or channel dimension is shared by every trial or channel, whatever
    order its dimensions are in. (The oracles test (time, channel) where the
    trial and channel counts are equal.)"""
    xr = pytest.importorskip("xarray")
    time_series = np.random.default_rng(13).standard_normal((200, 4, 3))
    labeled = xr.DataArray(time_series, dims=("time", "trial", "channel"))
    mask = np.zeros([{"time": 200, "trial": 4, "channel": 3}[dim] for dim in dims], bool)
    mask[index] = True
    # The same artifacts as a (time, trial) mask: trial 1, or every trial.
    expected = np.zeros((200, 4), bool)
    expected[60, 1 if "trial" in dims else slice(None)] = True
    transform_kwargs = {"n_time_samples_per_window": WINDOW}

    def power(artifact_samples):
        return quietly(
            multitaper_connectivity,
            labeled,
            SAMPLING_FREQUENCY,
            method="power",
            transform_kwargs={**transform_kwargs, "artifact_samples": artifact_samples},
        ).values

    np.testing.assert_array_equal(power(xr.DataArray(mask, dims=dims)), power(expected))


@pytest.mark.parametrize(
    ("dims", "match"),
    [
        (("time", "epoch"), r"has dimension\(s\) \['epoch'\] that time_series"),
        (("channel",), r"not the time dimension 'time'"),
    ],
)
def test_the_wrapper_refuses_a_labeled_mask_without_the_series_dims(dims, match):
    xr = pytest.importorskip("xarray")
    labeled = xr.DataArray(
        np.random.default_rng(14).standard_normal((200, 4, 3)),
        dims=("time", "trial", "channel"),
    )
    mask = xr.DataArray(np.zeros((200, 3) if len(dims) == 2 else 3, bool), dims=dims)
    with pytest.raises(ValueError, match=match):
        multitaper_connectivity(
            labeled,
            SAMPLING_FREQUENCY,
            method="power",
            transform_kwargs={"artifact_samples": mask},
        )


@pytest.mark.parametrize(
    ("shape", "match"),
    [
        (
            (3, 200, 4),
            (
                r"for time_series with dimensions \('channel', 'time', 'trial'\) and shape "
                r"\(3, 200, 4\), an unlabeled mask is read by position in the order "
                r"\('time', 'trial', 'channel'\): it must be \(200,\) for \('time',\), "
                r"\(200, 4\) for \('time', 'trial'\), or \(200, 4, 3\) .*"
                r"pass xr.DataArray\(artifact_samples, dims=\('channel', 'time', 'trial'\)\)"
            ),
        ),
        ((4, 200), r"Did you transpose it\? Pass artifact_samples.T"),
    ],
)
def test_the_wrapper_states_the_mask_shapes_of_labeled_input_in_its_dims(shape, match):
    """An unlabeled mask of a labeled series is read in the order the series
    is transposed to; a wrong shape is reported in the series' own names."""
    xr = pytest.importorskip("xarray")
    labeled = xr.DataArray(
        np.random.default_rng(15).standard_normal((3, 200, 4)),
        dims=("channel", "time", "trial"),
    )
    with pytest.raises(ValueError, match=match):
        multitaper_connectivity(
            labeled,
            SAMPLING_FREQUENCY,
            method="power",
            transform_kwargs={"artifact_samples": np.zeros(shape, dtype=bool)},
        )


def test_the_wrapper_transposes_a_labeled_mask_like_its_labeled_input(recording):
    xr = pytest.importorskip("xarray")
    time_series, artifacts = recording
    labeled = xr.DataArray(
        np.transpose(time_series, (2, 0, 1)), dims=("channel", "time", "trial")
    )
    labeled_mask = xr.DataArray(artifacts.T.copy(), dims=("trial", "time"))
    transform_kwargs = {"n_time_samples_per_window": WINDOW}
    from_labels = quietly(
        multitaper_connectivity,
        labeled,
        SAMPLING_FREQUENCY,
        method="power",
        transform_kwargs={**transform_kwargs, "artifact_samples": labeled_mask},
    )
    from_arrays = quietly(
        multitaper_connectivity,
        time_series,
        SAMPLING_FREQUENCY,
        method="power",
        transform_kwargs={**transform_kwargs, "artifact_samples": artifacts},
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
def independent_transform():
    """The transform of two independent white-noise signals; window ``w``
    drops ``w % 8`` trials."""
    rng = np.random.default_rng(10)
    time_series = rng.standard_normal((N_REPLICATES * MC_WINDOW, N_TRIALS, 2))
    artifacts = np.zeros(time_series.shape[:2], dtype=bool)
    for window in range(N_REPLICATES):
        dropped = rng.choice(N_TRIALS, size=window % 8, replace=False)
        artifacts[window * MC_WINDOW + MC_WINDOW // 2, dropped] = True
    return Multitaper(
        time_series,
        sampling_frequency=MC_WINDOW,  # 1 Hz bins
        time_halfbandwidth_product=2,
        n_tapers=3,
        n_time_samples_per_window=MC_WINDOW,
        artifact_samples=artifacts,
    )


@pytest.fixture
def independent_signals(independent_transform):
    """A fresh Connectivity of the independent signals, so no test sees
    another's cached intermediates."""
    return Connectivity.from_transform(independent_transform)


def _replicate_mean_and_limit(values):
    """The mean over replicates (windows) of each window's mean over the
    tested frequencies, and five standard errors of it."""
    per_replicate = values[:, MC_FREQUENCIES, 0, 1].mean(axis=1)
    return per_replicate.mean(), 5 * per_replicate.std(ddof=1) / np.sqrt(N_REPLICATES)


@pytest.mark.parametrize("measure", COUNTING_MEASURES)
def test_counting_measures_are_unbiased_with_dropped_trials(
    independent_transform, independent_signals, measure, monkeypatch
):
    """Their mean under independence is 0 with each window's kept count; with
    the raw count it would lie many limits away."""
    counts = independent_signals.n_valid_observations[:, 0]
    assert sorted(set(counts.tolist())) == [3, 6, 9, 12, 15, 18, 21, 24]
    mean, limit = _replicate_mean_and_limit(getattr(independent_signals, measure)())
    assert abs(mean) < limit, (mean, limit)

    def raw_count(self, measure, dtype):
        return self.n_observations

    monkeypatch.setattr(Connectivity, "_counts_for_correction", raw_count)
    raw_counted = Connectivity.from_transform(independent_transform)
    biased, _ = _replicate_mean_and_limit(getattr(raw_counted, measure)())
    # Four limits: twenty standard errors.
    assert abs(biased) > 4 * limit, (biased, limit)


def test_stft_drops_the_trial_windows_with_artifacts(recording):
    """ShortTimeFourierTransform takes artifact_samples as Multitaper does."""
    time_series, artifacts = recording
    stft = ShortTimeFourierTransform(
        time_series,
        SAMPLING_FREQUENCY,
        n_time_samples_per_window=WINDOW,
        artifact_samples="nonfinite",
    )
    np.testing.assert_array_equal(to_host(stft.artifact_samples), artifacts)
    assert to_host(stft.observation_weights).shape == (4, 5, 1, stft.n_fft_samples, 1)
    connectivity = Connectivity.from_transform(stft)
    for measure in ("power", "magnitude_squared_coherence", "pairwise_phase_consistency"):
        masked = quietly(getattr(connectivity, measure))
        for window in range(4):
            kept = _kept_window(
                time_series, artifacts, window * WINDOW, transform=ShortTimeFourierTransform
            )
            np.testing.assert_allclose(
                masked[window], quietly(getattr(kept, measure))[0], rtol=1e-10, atol=1e-13
            )
    assert "artifact_samples='nonfinite' (16 of 20 trial windows kept)" in repr(stft)
    assert "Artifact samples:" in stft.summarize_parameters()
    assert stft._provenance_metadata()["n_dropped_trial_windows"] == 4
