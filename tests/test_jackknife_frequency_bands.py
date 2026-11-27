"""``Connectivity.jackknife(frequency_bands=...)`` and ``JackknifeResult.replicates``.

A band statistic's interval is the jackknife of band-reduced replicates, each
recomputed without one whole trial. Leaving out one taper at a time is refused
for bands: the tapers' coefficients are correlated across nearby frequencies,
so its interval is too narrow (the Monte Carlo test below measures how much).
"""

import re
import warnings

import numpy as np
import pytest
import xarray as xr

from spectral_connectivity import Connectivity, Multitaper, frequency_band_reduce
from spectral_connectivity.simulate import simulate_MVAR
from spectral_connectivity.statistics import jackknife_confidence_interval
from tests._backend_helpers import to_host
from tests._expected_spectrum import _expected_multitaper_cross_spectrum
from tests._monte_carlo import replicate_mean_and_limit

_SAMPLING_FREQUENCY = 200
# 100-sample windows: 2 Hz bins. The "alpha" edges fall between bins, so its
# integral weighs two partial cells.
_BANDS = {"theta": (4, 8), "alpha": (8.3, 12.7), "beta": (13, 30)}
_N_TRIALS = 5
# Artifacts drop trial 1 from window 1 and trial 3 from window 0.
_ARTIFACTS = ((130, 1, 0), (20, 3, 1))
_TRANSFORMATIONS = {
    "magnitude_squared_coherence": "fisher_squared",
    "coherence_phase": "circular",
    "power": "log",
}
_DIMS = {
    "magnitude_squared_coherence": ("time", "frequency", "source", "target"),
    "coherence_phase": ("time", "frequency", "source", "target"),
    "power": ("time", "frequency", "source"),
}


@pytest.fixture(scope="module")
def time_series():
    """``(400, 5, 2)``: four 100-sample windows of five coupled trials."""
    rng = np.random.default_rng(20261127)
    signals = rng.standard_normal((400, _N_TRIALS, 2))
    signals[..., 1] += 0.6 * np.roll(signals[..., 0], 2, axis=0)
    return signals


def _with_artifacts(time_series):
    masked = time_series.copy()
    for sample, trial, signal in _ARTIFACTS:
        masked[sample, trial, signal] = np.nan
    return masked


def _connectivity(time_series, **multitaper_kwargs):
    """The default ``trials_tapers`` Connectivity of ``time_series`` (3 tapers)."""
    with warnings.catch_warnings():
        # Artifacts drop trial windows, with a warning.
        warnings.simplefilter("ignore", UserWarning)
        multitaper = Multitaper(
            time_series,
            sampling_frequency=_SAMPLING_FREQUENCY,
            n_time_samples_per_window=100,
            time_halfbandwidth_product=2,
            artifact_samples="nonfinite" if np.isnan(time_series).any() else None,
            **multitaper_kwargs,
        )
    return multitaper, Connectivity.from_transform(multitaper)


def _band_reduced(connectivity, method, frequency_reduction):
    """``method`` of ``connectivity`` reduced to ``_BANDS`` by the public
    ``frequency_band_reduce``, as an array ``(n_time, n_bands, ...)``."""
    labeled = xr.DataArray(
        getattr(connectivity, method)(),
        dims=_DIMS[method],
        coords={"frequency": connectivity.frequencies},
        attrs={"measure": method, "nyquist_frequency": _SAMPLING_FREQUENCY / 2},
    )
    return frequency_band_reduce(
        labeled, _BANDS, frequency_reduction=frequency_reduction
    ).values


def _without_trial(time_series, trial):
    return _connectivity(np.delete(time_series, trial, axis=1))[1]


@pytest.mark.parametrize("artifacts", [False, True], ids=["unweighted", "artifacts"])
@pytest.mark.parametrize(
    ("method", "frequency_reduction"),
    [
        ("magnitude_squared_coherence", "mean"),
        ("coherence_phase", "mean"),
        ("power", "mean"),
        ("power", "integral"),
    ],
)
def test_band_jackknife_is_the_jackknife_of_band_reduced_trial_replicates(
    time_series, artifacts, method, frequency_reduction
):
    """Each replicate is the measure recomputed from the data without one
    trial, then reduced to the bands by ``frequency_band_reduce``; the interval
    is ``jackknife_confidence_interval`` of those replicates. Where artifacts
    drop a trial from a window, that window's interval is the jackknife of the
    trials it keeps, and the replicate leaving out a trial it never had is NaN
    there."""
    data = _with_artifacts(time_series) if artifacts else time_series
    multitaper, connectivity = _connectivity(data)

    result = connectivity.jackknife(
        method,
        leave_out="trial",
        frequency_bands=_BANDS,
        frequency_reduction=frequency_reduction,
    )

    estimate = _band_reduced(connectivity, method, frequency_reduction)
    replicates = np.stack(
        [
            _band_reduced(_without_trial(data, trial), method, frequency_reduction)
            for trial in range(_N_TRIALS)
        ]
    )
    valid_windows = to_host(multitaper.valid_windows)  # (n_time, n_trials)
    assert result.estimate.shape == estimate.shape == (4, 3, *estimate.shape[2:])
    assert result.transformation == _TRANSFORMATIONS[method]
    np.testing.assert_allclose(result.estimate, estimate, rtol=1e-12, equal_nan=True)
    np.testing.assert_allclose(
        result.replicates,
        np.where(
            valid_windows.T.reshape(_N_TRIALS, 4, *(1,) * (estimate.ndim - 1)),
            replicates,
            np.nan,
        ),
        rtol=1e-12,
        equal_nan=True,
    )
    for window, kept in enumerate(valid_windows):
        expected = jackknife_confidence_interval(
            estimate[window],
            replicates[kept, window],
            transformation=_TRANSFORMATIONS[method],
        )
        counts = np.broadcast_to(result.n_observations, result.estimate.shape)
        assert np.all(counts[window] == kept.sum())
        for name in ("bias_corrected", "standard_error"):
            np.testing.assert_allclose(
                getattr(result, name)[window],
                getattr(expected, name),
                rtol=1e-11,
                atol=1e-14,
                equal_nan=True,
                err_msg=name,
            )
        for bound, expected_bound in zip(
            result.confidence_interval, expected.confidence_interval, strict=True
        ):
            np.testing.assert_allclose(
                bound[window], expected_bound, rtol=1e-11, atol=1e-14, equal_nan=True
            )
    if artifacts:
        assert np.isnan(result.replicates[1, 1]).all()  # trial 1 is not in window 1
        # Elsewhere a replicate is NaN only where the measure is.
        np.testing.assert_array_equal(
            np.isnan(result.replicates[1, 0]), np.isnan(result.estimate[0])
        )


def test_replicates_are_the_measure_without_each_trial_or_observation(time_series):
    """``replicates`` is the measure of the coefficients without one unit:
    with ``leave_out="observation"``, observation ``k`` of ``trials_tapers`` is
    taper ``k % n_tapers`` of trial ``k // n_tapers``; with ``"trial"``, every
    taper of one trial."""
    multitaper, connectivity = _connectivity(time_series)
    coefficients = to_host(multitaper.fft())
    n_time, n_trials, n_tapers = coefficients.shape[:3]
    # One trial-taper observation per "trial", trial-major.
    observations = coefficients.reshape(
        n_time, n_trials * n_tapers, 1, *coefficients.shape[3:]
    )

    by_observation = connectivity.jackknife("power", leave_out="observation")
    by_trial = connectivity.jackknife("power", leave_out="trial")

    expected_by_observation = [
        Connectivity(
            np.delete(observations, k, axis=1), frequencies=multitaper.frequencies
        ).power()
        for k in range(n_trials * n_tapers)
    ]
    expected_by_trial = [
        Connectivity(
            np.delete(coefficients, trial, axis=1), frequencies=multitaper.frequencies
        ).power()
        for trial in range(n_trials)
    ]
    np.testing.assert_allclose(
        by_observation.replicates, np.stack(expected_by_observation), rtol=1e-12
    )
    np.testing.assert_allclose(by_trial.replicates, np.stack(expected_by_trial), rtol=1e-12)
    # On the scale of the estimate, and the interval is theirs.
    assert by_trial.replicates.shape == (n_trials, *by_trial.estimate.shape)
    np.testing.assert_array_equal(
        by_trial.standard_error,
        jackknife_confidence_interval(
            by_trial.estimate, by_trial.replicates, transformation="log"
        ).standard_error,
    )


def test_replicates_give_an_interval_for_a_custom_statistic(time_series):
    """A statistic of the measure the interval does not cover, here the
    difference between two bands' mean coherence, gets an interval from the
    trial replicates: that is the jackknife of the statistic computed from
    the data without each trial."""
    _, connectivity = _connectivity(time_series)
    result = connectivity.jackknife(
        "magnitude_squared_coherence", leave_out="trial", frequency_bands=_BANDS
    )

    difference = jackknife_confidence_interval(
        result.estimate[:, 2] - result.estimate[:, 0],
        result.replicates[:, :, 2] - result.replicates[:, :, 0],
    )

    replicates = np.stack(
        [
            _band_reduced(
                _without_trial(time_series, trial), "magnitude_squared_coherence", "mean"
            )
            for trial in range(_N_TRIALS)
        ]
    )
    expected = jackknife_confidence_interval(
        result.estimate[:, 2] - result.estimate[:, 0],
        replicates[:, :, 2] - replicates[:, :, 0],
    )
    np.testing.assert_allclose(
        difference.standard_error, expected.standard_error, rtol=1e-10, equal_nan=True
    )


_TAPER_UNITS = [
    pytest.param("trials_tapers", _N_TRIALS, id="trials_tapers"),
    pytest.param("tapers", _N_TRIALS, id="tapers"),
    pytest.param("trials_tapers", 2, id="trials_tapers-2_trials"),
    pytest.param("tapers", 2, id="tapers-2_trials"),
]


@pytest.mark.parametrize(("expectation_type", "n_trials"), _TAPER_UNITS)
def test_bands_refuse_leaving_out_tapers(time_series, monkeypatch, expectation_type, n_trials):
    """Leaving out one taper at a time is refused for bands, before any measure
    is computed. The message gives the reason and the measured coverage, and
    the fix: leave out whole trials if there are 3, and otherwise that these
    data have no valid band jackknife."""
    multitaper, _ = _connectivity(time_series[:, :n_trials])
    connectivity = Connectivity.from_transform(multitaper, expectation_type=expectation_type)

    def not_computed(self, *args, **kwargs):
        msg = "the measure was computed before the refusal"
        raise AssertionError(msg)

    monkeypatch.setattr(Connectivity, "magnitude_squared_coherence", not_computed)
    with pytest.raises(ValueError, match=r"^jackknife\(frequency_bands=\.\.\.\)") as error:
        connectivity.jackknife("magnitude_squared_coherence", frequency_bands=_BANDS)

    message = str(error.value)
    assert "correlated at frequencies within their bandwidth" in message
    assert "43-81%" in message
    if n_trials < 3:
        assert "no valid jackknife for a band statistic on these data" in message
        assert f"these data have {n_trials}" in message
    else:
        assert "leave_out='trial'" in message
        assert "at least 3 trials" in message
        assert ("Build the Connectivity with expectation_type='trials_tapers'" in message) == (
            expectation_type == "tapers"
        )


@pytest.mark.parametrize(
    ("expectation_type", "n_tapers"),
    [("trials_tapers", 1), ("trials", 3)],
    ids=["one_taper_per_trial", "trials"],
)
def test_bands_accept_observations_that_are_whole_trials(
    time_series, expectation_type, n_tapers
):
    """With one taper per trial, or an expectation over trials only, each
    observation left out is a whole trial: ``leave_out="observation"`` is
    accepted and is the same jackknife as ``"trial"``."""
    multitaper, _ = _connectivity(time_series, n_tapers=n_tapers)
    connectivity = Connectivity.from_transform(multitaper, expectation_type=expectation_type)

    by_observation = connectivity.jackknife(
        "magnitude_squared_coherence", frequency_bands=_BANDS
    )
    by_trial = connectivity.jackknife(
        "magnitude_squared_coherence", leave_out="trial", frequency_bands=_BANDS
    )

    assert by_observation.replicates.shape[0] == _N_TRIALS
    for name in ("estimate", "standard_error", "replicates"):
        np.testing.assert_array_equal(getattr(by_observation, name), getattr(by_trial, name))


def _band_reduce_error(connectivity, frequency_bands, frequency_reduction, method):
    """The message ``frequency_band_reduce`` raises for the request."""
    labeled = xr.DataArray(
        np.zeros((1, connectivity.frequencies.size, 2, 2)),
        dims=("time", "frequency", "source", "target"),
        coords={"frequency": connectivity.frequencies},
        attrs={"measure": method},
    )
    with pytest.raises(ValueError, match=r".") as error:
        frequency_band_reduce(
            labeled, frequency_bands, frequency_reduction=frequency_reduction
        )
    return str(error.value)


@pytest.mark.parametrize(
    ("frequency_bands", "frequency_reduction", "method"),
    [
        ({"gamma": (150, 160)}, "mean", "magnitude_squared_coherence"),
        ({"reversed": (12, 8)}, "mean", "magnitude_squared_coherence"),
        ({}, "mean", "magnitude_squared_coherence"),
        ({1: (4, 8)}, "mean", "magnitude_squared_coherence"),
        (_BANDS, "median", "magnitude_squared_coherence"),
        (_BANDS, "integral", "magnitude_squared_coherence"),
    ],
    ids=["no_bin", "reversed", "empty", "unnamed", "median", "integral_of_coherence"],
)
def test_invalid_band_requests_raise_the_band_reduction_errors(
    time_series, monkeypatch, frequency_bands, frequency_reduction, method
):
    """A band request ``frequency_band_reduce`` refuses is refused with its
    message, before any measure is computed."""
    _, connectivity = _connectivity(time_series)
    expected = _band_reduce_error(connectivity, frequency_bands, frequency_reduction, method)

    def not_computed(self, *args, **kwargs):
        msg = "the measure was computed before the refusal"
        raise AssertionError(msg)

    monkeypatch.setattr(Connectivity, method, not_computed)
    with pytest.raises(ValueError, match=f"^{re.escape(expected)}$"):
        connectivity.jackknife(
            method,
            leave_out="trial",
            frequency_bands=frequency_bands,
            frequency_reduction=frequency_reduction,
        )


def test_bands_refuse_a_measure_that_selects_its_own_bins(time_series):
    _, connectivity = _connectivity(time_series)
    with pytest.raises(
        ValueError, match="selects its own frequency bins with frequency_range"
    ):
        connectivity.jackknife("phase_slope_index", leave_out="trial", frequency_bands=_BANDS)


def test_an_exact_spectrum_is_refused_first():
    """The exact-spectrum refusal comes before the band checks."""
    connectivity = Connectivity.from_var(
        np.array([[[0.5, 0.0], [0.3, 0.4]]]), sampling_frequency=100
    )
    with pytest.raises(ValueError, match=r"^jackknife is not defined"):
        connectivity.jackknife(
            "magnitude_squared_coherence", leave_out="trial", frequency_bands=_BANDS
        )


# A bivariate VAR(2): signal 0 oscillates and drives signal 1. Across the bands
# below the coherence of the expected spectrum varies from 0.47 to 0.56 and from
# 0.40 to 0.12, and power varies too.
_VAR_COEFFICIENTS = np.array([[[0.9, 0.0], [0.5, 0.6]], [[-0.5, 0.0], [0.0, -0.2]]])
_VAR_NOISE_COVARIANCE = np.eye(2)


def test_band_intervals_cover_with_trials_and_not_with_tapers():
    """Leaving out trials, the 95% intervals for band-mean magnitude-squared
    coherence and (on the log scale) band-mean power cover the band means of
    the spectrum the estimate averages to in about 95% of datasets; leaving
    out tapers, computed by hand from per-frequency replicates since
    ``frequency_bands`` refuses it, they cover far less.

    Each of 3000 time windows is one dataset of 4 trials of 64 samples (5
    tapers, NW = 3): independent VAR segments, so one call gives every
    dataset's interval. The target is the band mean of the measure of the
    exact expected multitaper cross-spectrum (``tests/_expected_spectrum.py``),
    so the window's and tapers' bias is not counted as error. Each band is 7
    bins (7 Hz) wide, a little more than the tapers' 6 Hz bandwidth.
    """
    n_time, n_trials, n_datasets = 64, 4, 3000
    segments = simulate_MVAR(
        _VAR_COEFFICIENTS,
        noise_covariance=_VAR_NOISE_COVARIANCE,
        n_time_samples=n_time,
        n_trials=n_datasets * n_trials,
        n_burnin_samples=200,
        random_state=20261127,
    )  # (n_time, n_datasets * n_trials, 2), each segment independent
    time_series = (
        segments.reshape(n_time, n_datasets, n_trials, 2)
        .transpose(1, 0, 2, 3)
        .reshape(n_datasets * n_time, n_trials, 2)
    )
    multitaper = Multitaper(
        time_series,
        sampling_frequency=n_time,
        n_time_samples_per_window=n_time,
        time_halfbandwidth_product=3,
    )
    connectivity = Connectivity.from_transform(multitaper)
    assert multitaper.n_tapers == 5

    # 1 Hz bins, so each band holds the 7 bins from low to high.
    bands = {"low": (4, 10), "high": (13, 19)}
    frequencies = connectivity.frequencies
    in_band = [(frequencies >= low) & (frequencies <= high) for low, high in bands.values()]
    expected = _expected_multitaper_cross_spectrum(
        multitaper, _VAR_COEFFICIENTS, _VAR_NOISE_COVARIANCE
    )[: frequencies.size]
    true_coherence = np.abs(expected[:, 0, 1]) ** 2 / (
        expected[:, 0, 0].real * expected[:, 1, 1].real
    )
    # One-sided density: the bins are interior, so doubled.
    true_power = 2 * expected[:, 0, 0].real

    def band_means(values, axis):
        return np.stack(
            [np.compress(mask, values, axis=axis).mean(axis) for mask in in_band], axis=axis
        )

    nominal = 0.95
    for method, truth, element in (
        ("magnitude_squared_coherence", true_coherence, (0, 1)),
        ("power", true_power, (0,)),
    ):
        target = np.array([truth[mask].mean() for mask in in_band])
        by_trial = connectivity.jackknife(method, leave_out="trial", frequency_bands=bands)
        # By hand: band-reduce the per-frequency leave-one-taper-out replicates.
        by_taper = connectivity.jackknife(method, leave_out="observation")
        taper_interval = jackknife_confidence_interval(
            band_means(by_taper.estimate, axis=1),
            band_means(by_taper.replicates, axis=2),
            transformation=by_trial.transformation,
        )
        assert by_taper.n_observations == n_trials * 5
        for band, name in enumerate(bands):
            covered = {}
            for units, interval in (("trial", by_trial), ("taper", taper_interval)):
                lower, upper = (
                    bound[:, band][(..., *element)] for bound in interval.confidence_interval
                )
                covered[units] = (lower <= target[band]) & (target[band] <= upper)
            coverage, limit = replicate_mean_and_limit(covered["trial"].astype(float))
            # Beyond sampling error: the jackknife and Student t approximations,
            # and at weak coherence the estimate's upward bias (13% in the high
            # band). Measured 94.5-95.3% over both measures and bands here, and
            # 93.1-95.7% in runs with other seeds, bands, window lengths
            # (64-128 samples), trial counts (4-8), and NW = 4.
            allowance = 0.01
            assert abs(coverage - nominal) <= limit + allowance, (method, name, coverage)
            # The guarded defect: leaving out tapers covers 74.6-79.1% here
            # (70-83% in those runs), 4 to 5 limits below nominal.
            taper_coverage, taper_limit = replicate_mean_and_limit(
                covered["taper"].astype(float)
            )
            assert nominal - taper_coverage > 3 * taper_limit + allowance, (
                method,
                name,
                taper_coverage,
            )
