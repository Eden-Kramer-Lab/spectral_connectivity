"""``Multitaper(artifact_handling="missing_data_tapers")``: estimating a trial
window that contains artifact samples from its valid samples.

Its tapers are the sequences zero at the artifacts that are most concentrated
in the bandwidth (Bronez 1988), the leading eigenvectors of the sinc kernel
restricted to the valid samples. Those concentrated above 0.9 are kept; a
window keeping fewer than 2 is dropped. Windows without artifacts are the
ordinary DPSS estimate, bit for bit.
"""

import inspect
import warnings

import numpy as np
import pytest
from scipy.linalg import eigh as scipy_eigh
from scipy.signal import detrend as scipy_detrend
from scipy.signal import lfilter
from scipy.signal.windows import dpss

from spectral_connectivity import (
    Connectivity,
    Multitaper,
    ShortTimeFourierTransform,
    multitaper_connectivity,
)
from spectral_connectivity.statistics import coherence_significance_pvalue
from spectral_connectivity.transforms import (
    _apply_adaptive_taper_weights,
    _detrend_valid_samples,
    _missing_data_slepian_tapers,
)
from tests._artifact_helpers import COUNTING_MEASURES, quietly
from tests._backend_helpers import to_host
from tests._expected_spectrum import _DETREND_BASIS, _expected_tapered_cross_spectrum

SAMPLING_FREQUENCY = 200.0
WINDOW = 64  # samples
NW = 3  # five DPSS tapers, the last of concentration 0.947


def _brute_force(valid, time_halfbandwidth_product):
    """Every eigenpair of the sinc kernel on the valid sample times, in
    decreasing order, from the kernel written out entry by entry."""
    half_bandwidth = time_halfbandwidth_product / valid.size
    times = np.flatnonzero(valid).astype(float)
    lag = times[:, None] - times[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        kernel = np.sin(2 * np.pi * half_bandwidth * lag) / (np.pi * lag)
    kernel[lag == 0] = 2 * half_bandwidth
    eigenvalues, eigenvectors = np.linalg.eigh(kernel)
    return eigenvalues[::-1], eigenvectors[:, ::-1]


def _valid(gaps, n_time=WINDOW):
    """``True`` but in the ``(start, length)`` gaps."""
    valid = np.ones(n_time, dtype=bool)
    for start, length in gaps:
        valid[start : start + length] = False
    return valid


# The warning that windows with artifacts were dropped all the same, which the
# fixtures do on purpose (tested in test_dropping_windows_with_artifacts_warns).
DROPPED_ANYWAY = "artifact_handling='missing_data_tapers': .* were dropped"


def _multitaper(time_series, artifacts, *, quiet=True, transform=Multitaper, **kwargs):
    """A ``transform`` of ``WINDOW``-sample windows at NW = ``NW`` with
    missing-data tapers, by default; ``quiet`` ignores ``DROPPED_ANYWAY``."""
    kwargs.setdefault("n_time_samples_per_window", WINDOW)
    kwargs.setdefault("time_halfbandwidth_product", NW)
    kwargs.setdefault("artifact_handling", "missing_data_tapers")
    with warnings.catch_warnings():
        if quiet:
            warnings.filterwarnings("ignore", DROPPED_ANYWAY, UserWarning)
        return transform(time_series, SAMPLING_FREQUENCY, artifact_samples=artifacts, **kwargs)


def _plain(time_series, **kwargs):
    """The same transform without a mask."""
    return Multitaper(
        time_series,
        SAMPLING_FREQUENCY,
        n_time_samples_per_window=WINDOW,
        time_halfbandwidth_product=NW,
        **kwargs,
    )


def _one_window(artifacts, seed):
    """Missing-data tapers of one 128-sample window at NW = 2, of noise with
    ``artifacts`` ``(128, n_trials)``."""
    rng = np.random.default_rng(seed)
    return _multitaper(
        rng.standard_normal((*artifacts.shape, 2)),
        artifacts,
        time_halfbandwidth_product=2,
        n_time_samples_per_window=artifacts.shape[0],
    )


@pytest.fixture(scope="module")
def recording():
    """Four windows of six trials of three signals, NaN at the artifacts.

    Window 0 is clean. Window 1: trial 1 has a 4-sample gap (it keeps 4
    tapers), trial 2 two 3-sample gaps (3 tapers). Window 2: trial 3 has every
    third sample missing (dropped), trial 4 a gap at the window's start (4
    tapers). Window 3: trial 5 has 50 of its 64 samples missing (dropped).
    """
    rng = np.random.default_rng(0)
    time_series = rng.standard_normal((4 * WINDOW, 6, 3))
    time_series[:, :, 1] += 0.6 * np.roll(time_series[:, :, 0], 2, axis=0)
    artifacts = np.zeros(time_series.shape[:2], dtype=bool)
    artifacts[WINDOW + 20 : WINDOW + 24, 1] = True
    artifacts[
        [WINDOW + 10, WINDOW + 11, WINDOW + 12, WINDOW + 45, WINDOW + 46, WINDOW + 47], 2
    ] = True
    artifacts[2 * WINDOW : 3 * WINDOW : 3, 3] = True
    artifacts[2 * WINDOW : 2 * WINDOW + 8, 4] = True
    artifacts[3 * WINDOW : 3 * WINDOW + 50, 5] = True
    time_series[artifacts] = np.nan
    return time_series, artifacts


EXPECTED_TAPER_COUNTS = [[5] * 6, [5, 4, 3, 5, 5, 5], [5, 5, 5, 0, 4, 5], [5] * 5 + [0]]


@pytest.fixture(scope="module")
def missing_data(recording):
    """The missing-data transform of the recording, for the tests that only read it."""
    return _multitaper(*recording)


def _clean_windows(artifacts):
    """``True`` at the trial windows without artifacts, ``(n_windows, n_trials)``."""
    n_trials = artifacts.shape[1]
    return ~artifacts.reshape(-1, WINDOW, n_trials).any(axis=1)


def test_each_window_keeps_its_concentrated_tapers(missing_data):
    multitaper = missing_data
    weights = to_host(multitaper.observation_weights)
    assert weights.shape == (4, 6, 5, multitaper.n_fft_samples, 1)
    assert (weights == weights[..., :1, :]).all()  # the same at every frequency
    counts = weights[..., 0, 0].sum(axis=-1)
    np.testing.assert_array_equal(counts, EXPECTED_TAPER_COUNTS)
    np.testing.assert_array_equal(to_host(multitaper.n_kept_tapers), EXPECTED_TAPER_COUNTS)
    # The kept tapers lead.
    np.testing.assert_array_equal(
        weights[..., 0, 0], np.arange(5) < counts[..., np.newaxis].astype(int)
    )
    np.testing.assert_array_equal(to_host(multitaper.valid_windows), counts > 0)
    connectivity = Connectivity.from_transform(multitaper)
    np.testing.assert_array_equal(connectivity.n_valid_observations[:, 0], counts.sum(axis=1))


@pytest.mark.parametrize(
    "gaps",
    [[(20, 4)], [(10, 3), (45, 3)], [(0, 8)], [(32, 1)], [(26, 12)]],
    ids=["one", "two", "at_start", "one_sample", "long"],
)
def test_the_tapers_are_the_brute_force_eigenvectors(gaps):
    """The concentrations and tapers are the eigenpairs of the restricted sinc
    kernel; the tapers have unit energy and are zero at the missing samples."""
    valid = _valid(gaps)
    orientation = dpss(WINDOW, NW, 5).T
    tapers, concentrations = _missing_data_slepian_tapers(valid, NW, 5, orientation)
    eigenvalues, eigenvectors = _brute_force(valid, NW)
    np.testing.assert_allclose(concentrations, eigenvalues[:5], atol=1e-12)
    assert (tapers[~valid] == 0).all()
    np.testing.assert_allclose(tapers.T @ tapers, np.eye(5), atol=1e-12)
    # Up to sign; these leading eigenvalues are distinct (gaps > 1e-5).
    assert np.diff(eigenvalues[:6]).max() < -1e-5
    alignment = np.abs(np.sum(tapers[valid] * eigenvectors[:, :5], axis=0))
    np.testing.assert_allclose(alignment, 1, atol=1e-10)
    # Each is oriented to agree with the DPSS of its order.
    assert (np.sum(tapers * orientation, axis=0) > 0).all()


def test_without_gaps_they_are_the_dpss():
    reference, ratios = dpss(WINDOW, NW, 5, return_ratios=True)
    tapers, concentrations = _missing_data_slepian_tapers(
        np.ones(WINDOW, dtype=bool), NW, 5, reference.T
    )
    np.testing.assert_allclose(tapers, reference.T, atol=1e-10)
    np.testing.assert_allclose(concentrations, ratios, atol=1e-12)


def test_the_dense_fallback_of_a_short_window_is_the_brute_force_solution():
    """20 valid samples or fewer are solved densely, as ARPACK cannot."""
    valid = _valid([(5, 4)], n_time=24)
    tapers, concentrations = _missing_data_slepian_tapers(valid, 2, 3, dpss(24, 2, 3).T)
    eigenvalues, eigenvectors = _brute_force(valid, 2)
    np.testing.assert_allclose(concentrations, eigenvalues[:3], atol=1e-12)
    alignment = np.abs(np.sum(tapers[valid] * eigenvectors[:, :3], axis=0))
    np.testing.assert_allclose(alignment, 1, atol=1e-10)


def test_tapers_of_equal_concentration_do_not_depend_on_the_solver(monkeypatch):
    """At NW = 8 the leading concentrations nearly coincide, so only their
    span is defined; that span is rotated to be closest to the DPSS, so
    the dense solver gives the same tapers, and without gaps they are the
    DPSS."""
    from spectral_connectivity import transforms

    n_time, nw, n_tapers = 256, 8, 15
    reference = dpss(n_time, nw, n_tapers).T
    valid = _valid([(40, 10), (150, 3)], n_time=n_time)
    eigenvalues, _ = _brute_force(valid, nw)
    assert (-np.diff(eigenvalues[:6])).min() < 1e-9  # the first two, 1.3e-10 apart
    sparse_tapers, sparse_concentrations = _missing_data_slepian_tapers(
        valid, nw, n_tapers, reference
    )

    def dense(first_row, indices, k):
        return scipy_eigh(first_row[np.abs(indices[:, None] - indices[None, :])])

    monkeypatch.setattr(transforms, "_leading_eigenpairs", lambda *a: _top(dense(*a), a[2]))
    dense_tapers, dense_concentrations = _missing_data_slepian_tapers(
        valid, nw, n_tapers, reference
    )
    np.testing.assert_allclose(sparse_concentrations, dense_concentrations, atol=1e-12)
    np.testing.assert_allclose(sparse_tapers, dense_tapers, atol=1e-8)
    monkeypatch.undo()
    gapless, _ = _missing_data_slepian_tapers(
        np.ones(n_time, dtype=bool), nw, n_tapers, reference
    )
    np.testing.assert_allclose(gapless, reference, atol=1e-8)


def _top(eigenpairs, k):
    """The ``k`` largest of ascending ``eigenpairs``."""
    eigenvalues, eigenvectors = eigenpairs
    return eigenvalues[-k:], eigenvectors[:, -k:]


def test_the_coefficients_are_those_of_the_brute_force_tapers(recording, missing_data):
    """A gapped window's coefficients: its valid samples, less their mean,
    times each brute-force taper scaled as the DPSS are, transformed."""
    time_series, artifacts = recording
    multitaper = missing_data
    coefficients = to_host(multitaper.fft())
    window, trial = 1, 2
    samples = slice(window * WINDOW, (window + 1) * WINDOW)
    valid = ~artifacts[samples, trial]
    _, eigenvectors = _brute_force(valid, NW)
    data = time_series[samples, trial][valid]  # (n_valid, n_signals)
    data = data - data.mean(axis=0)
    orientation = dpss(WINDOW, NW, 3)[:, valid]  # (3, n_valid)
    signs = np.sign(np.sum(eigenvectors[:, :3].T * orientation, axis=1))
    tapers = np.zeros((WINDOW, 3))
    tapers[valid] = eigenvectors[:, :3] * signs * np.sqrt(SAMPLING_FREQUENCY)
    projected = np.zeros((3, WINDOW, data.shape[1]))
    projected[:, valid] = tapers[valid].T[:, :, None] * data[None]
    expected = np.fft.fft(projected, n=multitaper.n_fft_samples, axis=1) / SAMPLING_FREQUENCY
    np.testing.assert_allclose(coefficients[window, trial, :3], expected, atol=1e-12)
    assert (coefficients[window, trial, 3:] == 0).all()  # the dropped tapers


@pytest.mark.parametrize("taper_weighting", ["uniform", "eigen", "adaptive"])
def test_a_window_without_artifacts_is_the_standard_estimate_bit_for_bit(
    recording, taper_weighting
):
    time_series, artifacts = recording
    clean = _clean_windows(artifacts)
    missing_data = to_host(
        _multitaper(time_series, artifacts, taper_weighting=taper_weighting).fft()
    )
    dropped = to_host(
        _multitaper(
            time_series, artifacts, taper_weighting=taper_weighting, artifact_handling="drop"
        ).fft()
    )
    if taper_weighting == "adaptive":
        # The iterations stop when every bin has converged, so the windows
        # with missing-data tapers can change their count, and the others'
        # weights within the tolerance (1e-8).
        np.testing.assert_allclose(missing_data[clean], dropped[clean], rtol=1e-6)
        return
    np.testing.assert_array_equal(missing_data[clean], dropped[clean])
    plain = to_host(_plain(np.nan_to_num(time_series), taper_weighting=taper_weighting).fft())
    np.testing.assert_array_equal(missing_data[clean], plain[clean])


def test_a_mask_without_artifacts_is_the_transform_without_one():
    time_series = np.random.default_rng(1).standard_normal((2 * WINDOW, 3, 2))
    masked = _multitaper(time_series, np.zeros((2 * WINDOW, 3), dtype=bool))
    assert masked.observation_weights is None
    np.testing.assert_array_equal(to_host(masked.fft()), to_host(_plain(time_series).fft()))


# Measures of each kind: plain averages, normalized and phase-lag families,
# those that count observations, and those that factor the cross-spectrum.
MEASURES = (
    "power",
    "magnitude_squared_coherence",
    "imaginary_coherence",
    "phase_locking_value",
    "weighted_phase_lag_index",
    "debiased_squared_weighted_phase_lag_index",
    "pairwise_phase_consistency",
    "partial_coherence",
    "pairwise_spectral_granger_prediction",
)


def test_every_measure_is_that_of_the_kept_observations(missing_data):
    """Each window's measures equal those of its kept (trial, taper)
    coefficients alone, the dropped tapers physically removed."""
    connectivity = Connectivity.from_transform(missing_data)
    coefficients = to_host(connectivity.fourier_coefficients)
    kept = np.arange(5) < np.array(EXPECTED_TAPER_COUNTS)[..., np.newaxis]
    for window in range(4):
        observations = coefficients[window][kept[window]]  # (n_kept, n_fft, n_signals)
        alone = Connectivity(observations[np.newaxis, :, np.newaxis])
        for measure in MEASURES:
            rtol = 1e-7 if measure == "pairwise_spectral_granger_prediction" else 1e-10
            np.testing.assert_allclose(
                quietly(getattr(connectivity, measure))[window],
                quietly(getattr(alone, measure))[0],
                rtol=rtol,
                atol=1e-13,
                err_msg=f"{measure}, window {window}",
            )
        jackknifed = quietly(connectivity.jackknife, "pairwise_phase_consistency")
        alone_jackknifed = quietly(alone.jackknife, "pairwise_phase_consistency")
        for field in ("bias_corrected", "standard_error"):
            np.testing.assert_allclose(
                getattr(jackknifed, field)[window],
                getattr(alone_jackknifed, field)[0],
                rtol=1e-9,
                atol=1e-13,
            )


@pytest.mark.parametrize("detrend_type", ["constant", "linear"])
def test_detrending_uses_the_valid_samples(detrend_type):
    rng = np.random.default_rng(2)
    windows = rng.standard_normal((3, 2, WINDOW))
    everywhere = np.ones(WINDOW, dtype=bool)
    np.testing.assert_allclose(
        _detrend_valid_samples(windows, everywhere, detrend_type),
        scipy_detrend(windows, type=detrend_type),
        atol=1e-12,
    )
    valid = _valid([(5, 6), (40, 10)])
    detrended = _detrend_valid_samples(windows, valid, detrend_type)
    assert (detrended[..., ~valid] == 0).all()
    # The least-squares fit to the valid samples alone.
    basis = _DETREND_BASIS[detrend_type](WINDOW)
    fit, *_ = np.linalg.lstsq(
        basis[valid], windows[..., valid].reshape(-1, valid.sum()).T, rcond=None
    )
    expected = windows[..., valid] - (basis[valid] @ fit).T.reshape(3, 2, -1)
    np.testing.assert_allclose(detrended[..., valid], expected, atol=1e-12)


@pytest.mark.parametrize("detrend_type", ["constant", "linear"])
def test_a_gapped_window_ignores_the_trend_it_removes(recording, detrend_type):
    """Adding a constant (and a line, when detrending one) to the samples
    leaves a gapped window's coefficients unchanged."""
    time_series, artifacts = recording
    trend = 5.0 + (detrend_type == "linear") * 0.3 * np.arange(time_series.shape[0])
    shifted = time_series + trend[:, None, None]
    base = to_host(_multitaper(time_series, artifacts, detrend_type=detrend_type).fft())
    moved = to_host(_multitaper(shifted, artifacts, detrend_type=detrend_type).fft())
    np.testing.assert_allclose(moved[1, 1:3], base[1, 1:3], atol=1e-9)


def test_a_short_gap_keeps_all_but_the_least_concentrated_taper():
    """A single missing sample in 64 already takes the fifth taper below 0.9
    (from 0.947); the other four stay well concentrated."""
    _, concentrations = _missing_data_slepian_tapers(
        _valid([(32, 1)]), NW, 5, dpss(WINDOW, NW, 5).T
    )
    assert (concentrations[:4] > 0.95).all()
    assert 0.85 < concentrations[4] < 0.9


@pytest.mark.parametrize(
    ("valid_pattern", "kept_fraction"),
    [([True, False], 1 / 2), ([True, False, False], 1 / 3), ([True, True, False], 2 / 3)],
)
def test_blanking_faster_than_the_bandwidth_caps_the_concentration(
    valid_pattern, kept_fraction
):
    """Blanking with a period shorter than 1 / (2 W) samples aliases every
    sequence into copies of its spectrum outside the band, so none can be
    concentrated above the fraction of samples kept: such a window is
    dropped."""
    n_time = 128
    valid = np.resize(valid_pattern, n_time)
    assert 1 / len(valid_pattern) > 2 * 2 / n_time  # faster than 2W at NW = 2
    eigenvalues, _ = _brute_force(valid, 2)
    assert eigenvalues[0] <= kept_fraction + 1e-12
    assert eigenvalues[0] > kept_fraction - 1e-3  # and the cap is reached
    artifacts = np.zeros((n_time, 2), dtype=bool)
    artifacts[:, 1] = ~valid
    np.testing.assert_array_equal(
        to_host(_one_window(artifacts, seed=3).valid_windows), [[True, False]]
    )


def _blanked_one_sample_in(period):
    """The window missing every ``period``-th sample."""
    artifacts = np.zeros((128, 1), dtype=bool)
    artifacts[::period] = True
    return _one_window(artifacts, seed=4)


def test_the_threshold_on_blanking_one_sample_in_a_period():
    """One sample missing in every 12 keeps 0.917 of the window, and its two
    leading tapers; one in every 9 keeps 0.889, below 0.9, and drops it."""
    weights = to_host(_blanked_one_sample_in(12).observation_weights)[0, 0, :, 0, 0]
    np.testing.assert_array_equal(weights, [1, 1, 0])
    with pytest.raises(ValueError, match=r"leaves no trial window.*fewer than 2"):
        _blanked_one_sample_in(9)


def test_a_window_keeping_one_concentrated_taper_is_dropped():
    """50 of 128 samples missing at NW = 2 leaves exactly one taper above 0.9."""
    artifacts = np.zeros((128, 2), dtype=bool)
    artifacts[:50, 1] = True
    eigenvalues, _ = _brute_force(~artifacts[:, 1], 2)
    assert eigenvalues[0] > 0.9 > eigenvalues[1]
    multitaper = _one_window(artifacts, seed=6)
    np.testing.assert_array_equal(to_host(multitaper.n_kept_tapers), [[3, 0]])
    np.testing.assert_array_equal(to_host(multitaper.valid_windows), [[True, False]])


def test_eigen_weighting_uses_each_windows_concentrations(recording, missing_data):
    time_series, artifacts = recording
    uniform = to_host(missing_data.fft())
    eigen = to_host(_multitaper(time_series, artifacts, taper_weighting="eigen").fft())
    window, trial = 1, 1  # four kept tapers
    valid = ~artifacts[window * WINDOW : (window + 1) * WINDOW, trial]
    concentrations = _brute_force(valid, NW)[0][:4]
    weights = np.sqrt(concentrations / concentrations.mean())
    np.testing.assert_allclose(
        eigen[window, trial, :4],
        uniform[window, trial, :4] * weights[:, None, None],
        rtol=1e-10,
    )


def test_adaptive_weighting_weights_a_gapped_window_as_a_window_of_its_own(
    recording, missing_data
):
    """Thomson's weights of a gapped window, from its own concentrations and the
    variance of its valid samples, as if its kept tapers were all it had."""
    time_series, artifacts = recording
    uniform = to_host(missing_data.fft())
    adaptive = to_host(_multitaper(time_series, artifacts, taper_weighting="adaptive").fft())
    window, trial = 1, 1
    samples = slice(window * WINDOW, (window + 1) * WINDOW)
    valid = ~artifacts[samples, trial]
    concentrations = _brute_force(valid, NW)[0][:4]
    noise = time_series[samples, trial][valid].var(axis=0) / SAMPLING_FREQUENCY
    alone = _apply_adaptive_taper_weights(
        uniform[window, trial, :4][None, None],
        concentrations,
        noise[None, None],
        max_iterations=50,
        tolerance=1e-8,
    )[0, 0]
    np.testing.assert_allclose(adaptive[window, trial, :4], alone, rtol=1e-6, atol=1e-12)
    assert (adaptive[window, trial, 4:] == 0).all()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {"artifact_handling": "interpolate"},
            r"must be 'drop' or 'missing_data_tapers', got 'interpolate'\. See",
        ),
        ({"artifact_handling": "missing_data_taper"}, "Did you mean 'missing_data_tapers'"),
        ({"artifact_samples": None}, "artifact_samples marks none"),
        ({"tapers": np.ones((WINDOW, 2)) / 8}, "custom tapers are not DPSS"),
        ({"is_low_bias": False}, "is_low_bias=False sets no such threshold"),
        ({"n_tapers": 1}, "uses 1, so it would drop"),
    ],
    ids=["unknown", "misspelled", "no_mask", "custom_tapers", "not_low_bias", "one_taper"],
)
def test_invalid_settings_are_rejected(recording, kwargs, match):
    time_series, artifacts = recording
    kwargs = {"artifacts": artifacts, **kwargs}
    kwargs["artifacts"] = kwargs.pop("artifact_samples", artifacts)
    with pytest.raises(ValueError, match=match):
        _multitaper(time_series, **kwargs)


def test_the_stft_refuses_missing_data_tapers(recording):
    time_series, artifacts = recording
    with pytest.raises(
        ValueError, match=r"supports only artifact_handling='drop'.*Multitaper"
    ):
        ShortTimeFourierTransform(
            time_series,
            SAMPLING_FREQUENCY,
            n_time_samples_per_window=WINDOW,
            artifact_samples=artifacts,
            artifact_handling="missing_data_tapers",
        )
    stft = ShortTimeFourierTransform(
        time_series,
        SAMPLING_FREQUENCY,
        n_time_samples_per_window=WINDOW,
        artifact_samples=artifacts,
    )
    assert stft.artifact_handling == "drop"
    with pytest.raises(
        ValueError, match=r"must be 'drop', got 'drp'\. Did you mean 'drop'\?.*ShortTime"
    ):
        ShortTimeFourierTransform(
            time_series,
            SAMPLING_FREQUENCY,
            artifact_samples=artifacts,
            artifact_handling="drp",
        )
    assert stft._provenance_metadata()["artifact_handling"] == "drop"


def test_artifact_handling_is_immutable_and_reported(recording, missing_data):
    time_series, artifacts = recording
    multitaper = missing_data
    assert multitaper.artifact_handling == "missing_data_tapers"
    with pytest.raises(AttributeError, match="immutable"):
        multitaper.artifact_handling = "drop"
    kept = (
        "<mask> (22 of 24 trial windows kept; 5 had artifacts, 3 rescued with "
        "missing-data tapers)"
    )
    assert (
        f"artifact_samples={kept},\n           artifact_handling='missing_data_tapers')"
        in (repr(multitaper))
    )
    assert f"Artifact samples:              {kept}" in multitaper.summarize_parameters()
    dropping = _multitaper(time_series, artifacts, artifact_handling="drop")
    assert "artifact_handling" not in repr(dropping)
    np.testing.assert_array_equal(
        to_host(dropping.n_kept_tapers), 5 * to_host(dropping.valid_windows)
    )
    metadata = multitaper._provenance_metadata()
    assert metadata["artifact_handling"] == "missing_data_tapers"
    assert metadata["n_dropped_trial_windows"] == 2
    assert metadata["n_missing_data_taper_trial_windows"] == 3


def test_dropping_windows_with_artifacts_warns(recording):
    """Once, with the count, what the dropped windows look like, the rule
    against the window's 1/(2W), and the remedies."""
    with pytest.warns(UserWarning, match=DROPPED_ANYWAY) as record:
        _multitaper(*recording, quiet=False)
    assert len(record) == 1
    message = str(record[0].message)
    for part in (
        "2 of 5 trial windows with artifacts kept fewer than 2 tapers",
        "enough of their energy (90%) inside the frequency band",
        "every 1/(2W) = 10.7 samples (0.0533 s",
        (
            "The dropped windows have 22% to 66% of their samples valid, and up to 22 "
            "separate artifacts in one window (2.91 samples apart on average)."
        ),
        "a larger time_halfbandwidth_product",
        "Multitaper.n_kept_tapers",
    ):
        assert part in message, part


@pytest.mark.parametrize(
    ("artifact_handling", "rule", "first"),
    [
        ("drop", "Alternatively, artifact_handling='missing_data_tapers'", "leaves 1 of 2"),
        ("missing_data_tapers", "fewer than 2 of its missing-data tapers", "1 of 1 trial"),
    ],
    ids=["drop", "missing_data_tapers"],
)
def test_the_drop_rule_is_stated_once(artifact_handling, rule, first):
    """Window 1, blanked every other sample, is dropped in its only trial. The
    first warning states the handling's rule: drop names missing-data tapers;
    with those, the dropped-anyway warning comes first, then the empty-window
    one without the rule."""
    time_series = np.random.default_rng(5).standard_normal((2 * WINDOW, 1, 2))
    artifacts = np.zeros((2 * WINDOW, 1), dtype=bool)
    artifacts[WINDOW::2] = True
    with pytest.warns(UserWarning, match="artifact") as record:
        _multitaper(time_series, artifacts, artifact_handling=artifact_handling, quiet=False)
    messages = [str(warning.message) for warning in record]
    assert first in messages[0]
    assert "leaves 1 of 2 time windows (0-based indices [1])" in messages[-1]
    assert [rule in message for message in messages] == [True] + [False] * (len(messages) - 1)


def test_a_multitaper_subclass_takes_missing_data_tapers(recording):
    class Subclass(Multitaper):
        pass

    subclass = _multitaper(*recording, transform=Subclass)
    np.testing.assert_array_equal(to_host(subclass.n_kept_tapers), EXPECTED_TAPER_COUNTS)


@pytest.mark.parametrize("set_first", [True, False], ids=["lines_first", "lines_last"])
def test_line_removal_is_refused(recording, set_first):
    """Line removal fits the shared DPSS tapers to every sample of a window,
    so it is refused whichever setting is stored first. Until Multitaper takes
    line_frequencies, a subclass that sets it stands in for it."""
    if "line_frequencies" in inspect.signature(Multitaper).parameters:
        if not set_first:
            pytest.skip("Multitaper sets line_frequencies itself")
        kwargs, transform = {"line_frequencies": [60.0]}, Multitaper
    else:

        class WithLines(Multitaper):
            def __init__(self, *args, **kwargs):
                if set_first:
                    self.line_frequencies = np.array([60.0])
                super().__init__(*args, **kwargs)
                if not set_first:
                    self.line_frequencies = np.array([60.0])

        kwargs, transform = {}, WithLines
    with pytest.raises(ValueError, match="line_frequencies cannot be combined with"):
        _multitaper(*recording, transform=transform, **kwargs)
    _multitaper(*recording, transform=transform, artifact_handling="drop", **kwargs)


def test_a_single_taper_set_is_required_where_a_feature_needs_one(recording, missing_data):
    with pytest.raises(ValueError, match="harmonic_f_test needs the same DPSS tapers"):
        missing_data._require_one_taper_set("harmonic_f_test")
    _multitaper(*recording, artifact_handling="drop")._require_one_taper_set("it")


@pytest.mark.skipif(
    not hasattr(Multitaper, "harmonic_f_test"), reason="harmonic_f_test is not merged"
)
def test_the_harmonic_f_test_is_refused(missing_data):
    with pytest.raises(ValueError, match="needs the same DPSS tapers"):
        missing_data.harmonic_f_test()


def test_the_wrapper_takes_artifact_handling_in_transform_kwargs(recording, missing_data):
    time_series, artifacts = recording
    transform_kwargs = {
        "n_time_samples_per_window": WINDOW,
        "artifact_samples": artifacts,
        "artifact_handling": "missing_data_tapers",
    }
    result = quietly(
        multitaper_connectivity,
        time_series,
        SAMPLING_FREQUENCY,
        method="pairwise_phase_consistency",
        time_halfbandwidth_product=NW,
        transform_kwargs=transform_kwargs,
    )
    direct = quietly(Connectivity.from_transform(missing_data).pairwise_phase_consistency)
    np.testing.assert_allclose(result.values, direct, rtol=1e-12, equal_nan=True)
    assert (
        '"artifact_handling":"missing_data_tapers"'
        in result.attrs["transform_parameters_json"]
    )


# Monte Carlo: replicate datasets on the window axis, one window each, of four
# trials: trial 0 clean, and the gaps of GAPS in the others in every window.
N_REPLICATES = 800
MC_TRIALS = 4
GAPS = {1: [(20, 4)], 2: [(10, 3), (45, 3)], 3: [(26, 12)]}
MC_TAPER_COUNTS = [5, 4, 3, 3]
# Interior bins half a bandwidth (6 Hz at NW = 3) apart, away from 0 Hz and
# Nyquist; the sampling frequency equals the window, so bins are 1 Hz apart.
MC_FREQUENCIES = [6, 9, 12, 15, 18, 21, 24, 27]


@pytest.fixture(scope="module")
def mc_artifacts():
    window = np.zeros((WINDOW, MC_TRIALS), dtype=bool)
    for trial, gaps in GAPS.items():
        window[:, trial] = ~_valid(gaps)
    return np.tile(window, (N_REPLICATES, 1))


def _mc_transforms(time_series, artifacts):
    """The Monte Carlo transforms by ``artifact_handling``, ``None`` without
    a mask; drop warns that it drops most trial windows."""
    settings = {
        "sampling_frequency": WINDOW,  # 1 Hz bins
        "time_halfbandwidth_product": NW,
        "n_time_samples_per_window": WINDOW,
    }
    transforms = {
        handling: quietly(
            Multitaper,
            time_series,
            artifact_samples=artifacts,
            artifact_handling=handling,
            **settings,
        )
        for handling in ("missing_data_tapers", "drop")
    }
    transforms[None] = Multitaper(time_series, **settings)
    return transforms


def _mean_and_limit(per_replicate):
    """The mean over replicates and five standard errors of it."""
    return per_replicate.mean(), 5 * per_replicate.std(ddof=1) / np.sqrt(per_replicate.size)


@pytest.fixture(scope="module")
def white_noise_transforms(mc_artifacts):
    """Two independent white-noise signals of unit variance: the missing-data,
    drop, and unmasked (``None``) transforms."""
    time_series = np.random.default_rng(10).standard_normal(
        (N_REPLICATES * WINDOW, MC_TRIALS, 2)
    )
    return _mc_transforms(time_series, mc_artifacts)


@pytest.fixture
def white_noise(white_noise_transforms):
    """A fresh Connectivity of each, so no test sees another's cached intermediates."""
    return {
        handling: Connectivity.from_transform(transform)
        for handling, transform in white_noise_transforms.items()
    }


def test_white_noise_power_is_unbiased_with_less_variance_than_drop(white_noise):
    """Unit-energy tapers on the valid samples give white noise its density,
    2 / sampling_frequency one-sided; keeping 15 of the 20 observations
    rather than the 5 drop keeps, the variance is about a third of drop's
    (measured: 0.066 against 0.20, and 0.049 without gaps)."""
    connectivity = white_noise["missing_data_tapers"]
    assert (connectivity.n_valid_observations == sum(MC_TAPER_COUNTS)).all()
    power = connectivity.power(one_sided=True)[:, MC_FREQUENCIES, 0]
    mean, limit = _mean_and_limit(power.mean(axis=1))
    truth = 2 / WINDOW
    assert abs(mean - truth) < limit, (mean, truth, limit)
    dropped = white_noise["drop"].power(one_sided=True)[:, MC_FREQUENCIES, 0]
    relative_variance = power.var(axis=0).mean() / truth**2
    dropped_variance = dropped.var(axis=0).mean() / truth**2
    # Near 1 / 15 and 1 / 5.
    assert 0.05 < relative_variance < 0.085, relative_variance
    assert dropped_variance > 2.5 * relative_variance, (dropped_variance, relative_variance)


def test_coherence_test_keeps_its_size(white_noise):
    connectivity = white_noise["missing_data_tapers"]
    coherency = connectivity.coherency()[:, MC_FREQUENCIES, 0, 1]
    counts = connectivity.n_valid_observations[:, MC_FREQUENCIES]
    rejected = coherence_significance_pvalue(coherency, counts) < 0.05
    limit = 5 * np.sqrt(0.05 * 0.95 / rejected.size)
    assert abs(rejected.mean() - 0.05) < limit, (rejected.mean(), limit)
    raw = coherence_significance_pvalue(coherency, connectivity.n_observations) < 0.05
    assert raw.mean() - 0.05 > 3 * limit, raw.mean()


@pytest.mark.parametrize("measure", COUNTING_MEASURES)
def test_counting_measures_are_unbiased(
    white_noise_transforms, white_noise, measure, monkeypatch
):
    """Their mean under independence is 0 with each window's count of kept
    tapers; with the raw count it would lie many limits away."""
    connectivity = white_noise["missing_data_tapers"]
    values = getattr(connectivity, measure)()[:, MC_FREQUENCIES, 0, 1]
    mean, limit = _mean_and_limit(values.mean(axis=1))
    assert abs(mean) < limit, (mean, limit)

    def raw_count(self, measure, dtype):
        return self.n_observations

    monkeypatch.setattr(Connectivity, "_counts_for_correction", raw_count)
    raw_counted = Connectivity.from_transform(white_noise_transforms["missing_data_tapers"])
    biased, _ = _mean_and_limit(
        getattr(raw_counted, measure)()[:, MC_FREQUENCIES, 0, 1].mean(axis=1)
    )
    # Two limits: ten standard errors (the counts are 15 and 20).
    assert abs(biased) > 2 * limit, (biased, limit)


def test_coupled_coherence_is_close_to_that_without_gaps(mc_artifacts):
    """The missing-data coherence of coupled signals (0.64) agrees with the
    coherence of the same data with nothing missing, replicate by replicate on
    average; drop's, from 5 observations, is biased upward and varies more
    (measured: 0.630, 0.629 without gaps, 0.655 for drop; variances 0.0115,
    0.0086, 0.031)."""
    rng = np.random.default_rng(11)
    source = rng.standard_normal((N_REPLICATES * WINDOW, MC_TRIALS))
    target = 0.8 * np.roll(source, 1, axis=0) + 0.6 * rng.standard_normal(source.shape)
    time_series = np.stack([source, target], axis=-1)
    coherence = {
        handling: Connectivity.from_transform(transform).magnitude_squared_coherence()[
            :, MC_FREQUENCIES, 0, 1
        ]
        for handling, transform in _mc_transforms(time_series, mc_artifacts).items()
    }
    difference, limit = _mean_and_limit(
        (coherence["missing_data_tapers"] - coherence[None]).mean(axis=1)
    )
    assert abs(difference) < limit, (difference, limit)
    dropped, _ = _mean_and_limit((coherence["drop"] - coherence[None]).mean(axis=1))
    assert dropped > 0.015, dropped
    assert (
        coherence["drop"].var(axis=0).mean()
        > 2 * coherence["missing_data_tapers"].var(axis=0).mean()
    )


def test_ar_power_averages_to_its_tapers_expectation_near_the_gapless_one(mc_artifacts):
    """AR(2) noise (poles of radius 0.77; a spectral range of 119): the power
    averages to the exact expectation of each trial's own tapers and detrending,
    which is within 3% of the expectation without gaps at the tested
    frequencies: the leakage of tapers concentrated above 0.9 rather than the
    DPSS's 0.947 or more, and the narrower smoothing of fewer tapers. (A
    spectral range of 2400 takes it to tens of percent.)"""
    ar = np.array([[[1.2]], [[-0.6]]])
    rng = np.random.default_rng(12)
    time_series = lfilter(
        [1],
        [1, -1.2, 0.6],
        rng.standard_normal((N_REPLICATES * WINDOW + 500, MC_TRIALS, 1)),
        axis=0,
    )[500:]
    reference = dpss(WINDOW, NW, 5).T
    gapless = _expected_tapered_cross_spectrum(ar, np.eye(1), reference, WINDOW, "constant")
    expected = MC_TAPER_COUNTS[0] * gapless
    for trial, gaps in GAPS.items():
        valid = _valid(gaps)
        tapers, _ = _missing_data_slepian_tapers(valid, NW, 5, reference)
        n_kept = MC_TAPER_COUNTS[trial]
        expected = expected + n_kept * _expected_tapered_cross_spectrum(
            ar, np.eye(1), tapers[:, :n_kept], WINDOW, "constant", valid=valid
        )
    expected = expected[:, 0, 0].real / sum(MC_TAPER_COUNTS) / WINDOW
    gapless = gapless[:, 0, 0].real / WINDOW
    transform = Multitaper(
        time_series,
        sampling_frequency=WINDOW,
        time_halfbandwidth_product=NW,
        n_time_samples_per_window=WINDOW,
        artifact_samples=mc_artifacts,
        artifact_handling="missing_data_tapers",
    )
    power = Connectivity.from_transform(transform).power(one_sided=False)[:, MC_FREQUENCIES, 0]
    mean, limit = _mean_and_limit((power / expected[MC_FREQUENCIES]).mean(axis=1))
    assert abs(mean - 1) < limit, (mean, limit)
    leakage = np.abs(expected[MC_FREQUENCIES] / gapless[MC_FREQUENCIES] - 1)
    assert leakage.max() < 0.03, leakage
