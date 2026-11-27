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
import scipy.stats
import xarray as xr

from spectral_connectivity import (
    Connectivity,
    MorletWavelet,
    Multitaper,
    ShortTimeFourierTransform,
    Welch,
    connectivity_to_xarray,
    frequency_band_reduce,
    list_measures,
    multitaper_connectivity,
)
from spectral_connectivity.connectivity import _select_measure_frequencies
from spectral_connectivity.simulate import simulate_MVAR
from spectral_connectivity.statistics import JackknifeResult, jackknife_confidence_interval
from tests._artifact_helpers import quietly
from tests._backend_helpers import to_host
from tests._expected_spectrum import (
    _expected_multitaper_cross_spectrum,
    _expected_tapered_cross_spectrum,
)
from tests._monte_carlo import replicate_mean_and_limit
from tests._var_oracle import (
    _coherency,
    _dense_stable_var,
    _fourier_coefficients_with_cross_spectrum,
)

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
_DIMS = {measure.name: measure.dims for measure in list_measures()}

# A bivariate VAR(2): signal 0 oscillates and drives signal 1. Across the bands
# below the coherence of the expected spectrum varies from 0.47 to 0.56 and from
# 0.40 to 0.12, and power varies too.
_VAR_COEFFICIENTS = np.array([[[0.9, 0.0], [0.5, 0.6]], [[-0.5, 0.0], [0.0, -0.2]]])
_VAR_NOISE_COVARIANCE = np.eye(2)


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


def _connectivity(time_series, expectation_type="trials_tapers", **multitaper_kwargs):
    """A Multitaper of ``time_series`` (3 tapers; artifacts dropped, which
    warns) and its Connectivity."""
    multitaper = quietly(
        Multitaper,
        time_series,
        sampling_frequency=_SAMPLING_FREQUENCY,
        n_time_samples_per_window=100,
        time_halfbandwidth_product=2,
        artifact_samples="nonfinite" if np.isnan(time_series).any() else None,
        **multitaper_kwargs,
    )
    return multitaper, Connectivity.from_transform(
        multitaper, expectation_type=expectation_type
    )


def _assert_same_interval(result, expected, rtol=0.0, atol=0.0):
    """``result`` and ``expected`` have the same bias-corrected estimate,
    standard error and bounds (exactly, by default)."""
    for name in ("bias_corrected", "standard_error"):
        np.testing.assert_allclose(
            getattr(result, name),
            getattr(expected, name),
            rtol=rtol,
            atol=atol,
            equal_nan=True,
            err_msg=name,
        )
    for bound, expected_bound in zip(
        result.confidence_interval, expected.confidence_interval, strict=True
    ):
        np.testing.assert_allclose(bound, expected_bound, rtol=rtol, atol=atol, equal_nan=True)


def _datasets_as_windows(n_time, n_trials, n_datasets, random_state):
    """``(n_datasets * n_time, n_trials, 2)`` independent segments of the VAR:
    each window of ``n_time`` samples is one dataset of ``n_trials`` trials."""
    segments = simulate_MVAR(
        _VAR_COEFFICIENTS,
        noise_covariance=_VAR_NOISE_COVARIANCE,
        n_time_samples=n_time,
        n_trials=n_datasets * n_trials,
        n_burnin_samples=200,
        random_state=random_state,
    )
    return (
        segments.reshape(n_time, n_datasets, n_trials, 2)
        .transpose(1, 0, 2, 3)
        .reshape(n_datasets * n_time, n_trials, 2)
    )


def _magnitude_squared_coherence(spectrum):
    """Of the cross-spectral matrices ``(..., 2, 2)``, between the two signals."""
    return np.abs(_coherency(spectrum)[..., 0, 1]) ** 2


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


def _refuse_computing(monkeypatch, method):
    """Make ``Connectivity.<method>`` fail if a check lets it be computed."""

    def not_computed(self, *args, **kwargs):
        msg = "the measure was computed before the refusal"
        raise AssertionError(msg)

    if hasattr(Connectivity, method):
        monkeypatch.setattr(Connectivity, method, not_computed)


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
        keep_replicates=True,
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
        window_interval = JackknifeResult(
            estimate=result.estimate[window],
            bias_corrected=result.bias_corrected[window],
            standard_error=result.standard_error[window],
            confidence_interval=tuple(bound[window] for bound in result.confidence_interval),
            n_observations=counts[window],
            transformation=result.transformation,
        )
        _assert_same_interval(window_interval, expected, rtol=1e-11, atol=1e-14)
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

    by_observation = connectivity.jackknife(
        "power", leave_out="observation", keep_replicates=True
    )
    by_trial = connectivity.jackknife("power", leave_out="trial", keep_replicates=True)

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
    # With expectation_type="trials" the observations are the trials.
    by_trials_expectation = _connectivity(time_series, expectation_type="trials")[1].jackknife(
        "power", keep_replicates=True
    )
    np.testing.assert_allclose(
        by_trials_expectation.replicates,
        np.stack(
            [
                Connectivity(
                    np.delete(coefficients, trial, axis=1),
                    frequencies=multitaper.frequencies,
                    expectation_type="trials",
                ).power()
                for trial in range(n_trials)
            ]
        ),
        rtol=1e-12,
    )
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
        "magnitude_squared_coherence",
        leave_out="trial",
        frequency_bands=_BANDS,
        keep_replicates=True,
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


def test_replicates_are_kept_only_when_asked_for(time_series):
    """``replicates`` is None by default; asking for it changes nothing else.
    The result records the bands, in the order of its band axis, and the unit
    left out."""
    _, connectivity = _connectivity(time_series)
    kwargs = {"leave_out": "trial", "frequency_bands": _BANDS}
    default = connectivity.jackknife("magnitude_squared_coherence", **kwargs)
    kept = connectivity.jackknife(
        "magnitude_squared_coherence", keep_replicates=True, **kwargs
    )

    assert default.replicates is None
    assert kept.replicates.shape == (_N_TRIALS, *kept.estimate.shape)
    for name in ("estimate", "bias_corrected", "standard_error"):
        np.testing.assert_array_equal(getattr(default, name), getattr(kept, name))
    assert default.frequency_bands == _BANDS
    assert list(default.frequency_bands) == ["theta", "alpha", "beta"]
    assert default.frequency_reduction == "mean"
    assert default.leave_out == "trial"
    integral = connectivity.jackknife(
        "power", frequency_reduction="integral", one_sided=True, **kwargs
    )
    assert integral.frequency_reduction == "integral"
    unbanded = connectivity.jackknife("power", frequency_reduction="integral")
    assert (unbanded.frequency_bands, unbanded.frequency_reduction) == (None, None)
    assert unbanded.leave_out == "observation"
    for value in ("yes", 1):
        with pytest.raises(TypeError, match="keep_replicates must be a bool") as error:
            connectivity.jackknife("power", keep_replicates=value)
        assert "spectrum" not in str(error.value)


@pytest.mark.parametrize("transformation", ["circular", "fisher"])
def test_public_interval_warnings_point_at_the_caller(transformation):
    """The whole-circle and saturated-coherence warnings of
    jackknife_confidence_interval are attributed to the caller's line, through
    the interval core."""
    estimate = np.array([0.0, 1.0])
    replicates = np.array([[3.0, 1.0], [-3.0, 1.0], [2.9, 1.0]])
    with pytest.warns(UserWarning, match="Jackknife circular|Fisher jackknife") as caught:
        jackknife_confidence_interval(estimate, replicates, transformation=transformation)
    assert {record.filename for record in caught} == {__file__}


def test_two_replicates_with_a_nan_keep_elements_with_both():
    """With 2 replicates given, the too-few threshold is 2, not 3: an element
    with both keeps its interval, one with a single replicate is NaN with a
    warning."""
    estimate = np.array([0.5, 0.5])
    replicates = np.array([[0.4, 0.4], [0.6, np.nan]])
    with pytest.warns(UserWarning, match="1 of 2 elements have fewer than 2 non-NaN"):
        result = jackknife_confidence_interval(estimate, replicates)
    np.testing.assert_array_equal(result.n_observations, [2, 0])
    assert np.isfinite(result.standard_error[0])
    assert np.isnan(result.standard_error[1])
    plain = jackknife_confidence_interval(estimate[:1], replicates[:, :1])
    np.testing.assert_array_equal(result.standard_error[:1], plain.standard_error)


def test_jackknife_result_is_built_from_its_original_fields():
    """The fields added since the first 3.0 builds default to None."""
    estimate = np.ones(3)
    result = JackknifeResult(
        estimate=estimate,
        bias_corrected=estimate,
        standard_error=estimate,
        confidence_interval=(estimate, estimate),
        n_observations=5,
        transformation="identity",
    )
    assert (
        result.replicates,
        result.frequency_bands,
        result.frequency_reduction,
        result.leave_out,
        result.interval_center,
    ) == (None, None, None, None, None)
    replicates = np.ones((5, 3))
    direct = jackknife_confidence_interval(estimate, replicates)
    assert direct.replicates is replicates
    assert (direct.frequency_bands, direct.frequency_reduction, direct.leave_out) == (
        None,
        None,
        None,
    )
    assert direct.interval_center == "estimate"


@pytest.mark.parametrize("frequency_bands", [None, _BANDS], ids=["frequencies", "bands"])
@pytest.mark.parametrize("leave_out", ["observation", "trial"])
def test_replicates_passed_back_reproduce_the_interval_with_artifacts(
    time_series, leave_out, frequency_bands
):
    """``jackknife_confidence_interval`` skips the NaN replicates of units a
    window never had, so the estimate and replicates of a jackknife with
    artifacts give back its interval, window by window."""
    data = time_series.copy()
    data[110:118, 2, 0] = np.nan  # a gap in trial 2, window 1
    # Bands leave out tapers only with one taper per trial.
    one_taper = frequency_bands is not None and leave_out == "observation"
    _, connectivity = _connectivity(data, **({"n_tapers": 1} if one_taper else {}))
    result = connectivity.jackknife(
        "magnitude_squared_coherence",
        leave_out=leave_out,
        frequency_bands=frequency_bands,
        keep_replicates=True,
    )
    assert np.isnan(result.replicates).any()

    passed_back = jackknife_confidence_interval(
        result.estimate, result.replicates, transformation=result.transformation
    )

    _assert_same_interval(passed_back, result)
    defined = ~np.isnan(result.standard_error)
    np.testing.assert_array_equal(
        passed_back.n_observations[defined],
        np.broadcast_to(result.n_observations, result.estimate.shape)[defined],
    )


def test_a_replicate_the_measure_leaves_undefined_is_nan_only_in_the_jackknife():
    """The one difference between the two: where the measure is NaN without a
    unit the bin has (here coherence, when the only trial in which signal 1 has
    power is left out), the jackknife's interval is NaN, while the replicates
    passed back get the interval of the others."""
    rng = np.random.default_rng(0)
    shape = (1, 5, 1, 8, 2)
    coefficients = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    coefficients[:, 1:, :, :, 1] = 0  # signal 1 has power in trial 0 only
    with pytest.warns(UserWarning, match="near-.zero power"):
        result = Connectivity(coefficients).jackknife(
            "magnitude_squared_coherence", leave_out="trial", keep_replicates=True
        )
    element = (0, 1, 0, 1)
    assert np.isnan(result.replicates[(0, *element)])
    assert np.isnan(result.standard_error[element])

    passed_back = jackknife_confidence_interval(
        result.estimate, result.replicates, transformation=result.transformation
    )

    expected = jackknife_confidence_interval(
        result.estimate[element][np.newaxis],
        result.replicates[1:][(slice(None), *element)][:, np.newaxis],
        transformation=result.transformation,
    )
    assert passed_back.n_observations[element] == 4
    np.testing.assert_array_equal(
        passed_back.standard_error[element], expected.standard_error[0]
    )


def test_nan_replicates_are_skipped_element_by_element():
    """Each element's interval is that of its own non-NaN replicates; one
    left with fewer than 3 is NaN, with a warning, and one with none is NaN."""
    rng = np.random.default_rng(3)
    replicates = rng.uniform(0.2, 0.8, (6, 4))
    estimate = replicates.mean(axis=0)
    replicates[[0, 3], 1] = np.nan  # 4 left
    replicates[:4, 2] = np.nan  # 2 left
    replicates[:, 3] = np.nan  # none left

    with pytest.warns(UserWarning, match="1 of 4 elements have fewer than 3 non-NaN"):
        result = jackknife_confidence_interval(estimate, replicates, transformation="log")

    for element, kept in ((0, slice(None)), (1, [1, 2, 4, 5])):
        expected = jackknife_confidence_interval(
            estimate[element : element + 1],
            replicates[kept, element : element + 1],
            transformation="log",
        )
        np.testing.assert_allclose(
            result.standard_error[element], expected.standard_error[0], rtol=1e-14
        )
        np.testing.assert_allclose(
            result.confidence_interval[0][element],
            expected.confidence_interval[0][0],
            rtol=1e-14,
        )
    assert result.n_observations.tolist() == [6, 4, 0, 0]
    assert np.isnan(result.standard_error[2:]).all()
    assert np.isnan(result.confidence_interval[1][2:]).all()


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
    _, connectivity = _connectivity(time_series[:, :n_trials], expectation_type)

    _refuse_computing(monkeypatch, "magnitude_squared_coherence")
    with pytest.raises(ValueError, match=r"^jackknife\(frequency_bands=\.\.\.\)") as error:
        connectivity.jackknife("magnitude_squared_coherence", frequency_bands=_BANDS)

    message = str(error.value)
    assert "cannot leave out one taper at a time" in message
    assert "correlated with the other tapers' across the frequencies of a band" in message
    assert "see the Notes of Connectivity.jackknife" in message
    assert f"these data have {n_trials}" in message
    if n_trials < 3:
        assert "has no valid jackknife with fewer than 3 trials" in message
        assert "do not average those intervals or their replicates over a band" in message
    else:
        assert "Pass leave_out='trial'" in message or "pass leave_out='trial'" in message
        assert "at least 3 trials" in message
        assert ("per-trial band intervals have no valid jackknife" in message) == (
            expectation_type == "tapers"
        )


def test_one_taper_per_trial_is_a_trial_whatever_the_independence_flag():
    """The independence of observations is about the parts of one trial, so
    with one taper per trial leaving out an observation is leaving out a
    trial, valid as leave_out="trial" is, also with bands."""
    rng = np.random.default_rng(4)
    shape = (1, 5, 1, 16, 2)
    coefficients = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    connectivity = Connectivity(coefficients, observations_are_independent=False)
    bands = {"low": (0.05, 0.2)}
    by_observation = connectivity.jackknife(
        "magnitude_squared_coherence", frequency_bands=bands
    )
    by_trial = connectivity.jackknife(
        "magnitude_squared_coherence", leave_out="trial", frequency_bands=bands
    )
    np.testing.assert_array_equal(by_observation.standard_error, by_trial.standard_error)


def test_two_trials_have_no_band_jackknife(time_series):
    """With 2 trials, leaving out trials is refused for bands with advice that
    does not point back to leaving out tapers."""
    _, connectivity = _connectivity(time_series[:, :2])
    with pytest.raises(ValueError, match="requires at least 3 trials") as error:
        connectivity.jackknife(
            "magnitude_squared_coherence", leave_out="trial", frequency_bands=_BANDS
        )
    assert "has no valid jackknife with fewer than 3 trials" in str(error.value)
    assert "Keep leave_out='observation'" not in str(error.value)
    # Welch segments may be left out one at a time instead.
    welch = Connectivity.from_transform(
        Welch(time_series[:, :2], sampling_frequency=_SAMPLING_FREQUENCY, segment_duration=0.5)
    )
    with pytest.raises(ValueError, match="requires at least 3 trials") as error:
        welch.jackknife(
            "magnitude_squared_coherence", leave_out="trial", frequency_bands=_BANDS
        )
    assert "Keep leave_out='observation' to leave out one segment" in str(error.value)


@pytest.mark.parametrize(
    ("method", "frequency_bands", "frequency_reduction", "match"),
    [
        ("magnitude_squared_coherenc", _BANDS, "mean", "is not a connectivity measure"),
        ("phase_slope_index", _BANDS, "mean", "once per band with frequency_range"),
        ("group_delay", _BANDS, "mean", "does not resample group_delay"),
        ("magnitude_squared_coherence", _BANDS, "integral", "defined only for power"),
        (
            "magnitude_squared_coherence",
            {"x": (4.1, 4.2)},
            "mean",
            "contains no frequency bin",
        ),
    ],
    ids=["misspelled", "phase_slope_index", "group_delay", "integral", "empty_band"],
)
def test_the_method_and_bands_are_checked_before_the_taper_refusal(
    time_series, monkeypatch, method, frequency_bands, frequency_reduction, match
):
    """A request that fails for its method or bands says so, rather than
    giving the taper refusal (default leave_out), whose advice would not help;
    nothing is computed first."""
    _, connectivity = _connectivity(time_series)
    _refuse_computing(monkeypatch, method)
    with pytest.raises(ValueError, match=match):
        connectivity.jackknife(
            method, frequency_bands=frequency_bands, frequency_reduction=frequency_reduction
        )


@pytest.mark.parametrize("one_sided", [False, np.False_])
def test_a_two_sided_band_integral_is_refused(time_series, one_sided):
    """Integrating power(one_sided=False) would give half the band power."""
    _, connectivity = _connectivity(time_series)
    kwargs = {
        "leave_out": "trial",
        "frequency_bands": _BANDS,
        "frequency_reduction": "integral",
    }
    with pytest.raises(ValueError, match="Pass one_sided=True"):
        connectivity.jackknife("power", one_sided=one_sided, **kwargs)
    np.testing.assert_array_equal(
        connectivity.jackknife("power", one_sided=True, **kwargs).estimate,
        _band_reduced(connectivity, "power", "integral"),
    )


def test_bands_accept_observations_that_are_whole_trials(time_series):
    """With an expectation over trials only, each observation left out is a
    whole trial: ``leave_out="observation"`` is accepted and is the same
    jackknife as ``"trial"`` (one taper per trial is tested with the
    oracles below)."""
    _, connectivity = _connectivity(time_series, expectation_type="trials")

    by_observation = connectivity.jackknife(
        "magnitude_squared_coherence",
        frequency_bands=_BANDS,
        keep_replicates=True,
    )
    by_trial = connectivity.jackknife(
        "magnitude_squared_coherence",
        leave_out="trial",
        frequency_bands=_BANDS,
        keep_replicates=True,
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

    _refuse_computing(monkeypatch, method)
    with pytest.raises(ValueError, match=f"^{re.escape(expected)}$"):
        connectivity.jackknife(
            method,
            leave_out="trial",
            frequency_bands=frequency_bands,
            frequency_reduction=frequency_reduction,
        )


@pytest.mark.parametrize("segment_overlap", [0.0, 0.5])
def test_welch_segments_are_left_out_one_at_a_time_for_bands(time_series, segment_overlap):
    """Welch's observations are segments of each trial in time, which it
    records, so a band jackknife may leave out one segment at a time: each
    replicate is the band-reduced measure without one trial-segment."""
    welch = Welch(
        time_series,
        sampling_frequency=_SAMPLING_FREQUENCY,
        n_time_samples_per_segment=100,
        segment_overlap=segment_overlap,
    )
    connectivity = Connectivity.from_transform(welch)
    assert connectivity.observations_are_time_segments

    result = connectivity.jackknife(
        "magnitude_squared_coherence", frequency_bands=_BANDS, keep_replicates=True
    )

    coefficients = to_host(welch.fft())  # (1, n_trials, n_segments, n_fft, n_signals)
    observations = coefficients.reshape(1, -1, 1, *coefficients.shape[3:])
    replicates = np.stack(
        [
            _band_reduced(
                Connectivity(
                    np.delete(observations, k, axis=1), frequencies=welch.frequencies
                ),
                "magnitude_squared_coherence",
                "mean",
            )
            for k in range(observations.shape[1])
        ]
    )
    expected = jackknife_confidence_interval(
        _band_reduced(connectivity, "magnitude_squared_coherence", "mean"),
        replicates,
        transformation="fisher_squared",
    )
    assert result.n_observations == observations.shape[1] == _N_TRIALS * welch.n_segments
    np.testing.assert_allclose(result.replicates, replicates, rtol=1e-12, equal_nan=True)
    np.testing.assert_allclose(
        result.standard_error, expected.standard_error, rtol=1e-11, equal_nan=True
    )


def test_coefficients_passed_directly_are_taken_as_tapers_unless_stated(time_series):
    """Coefficients passed directly are taken as tapers, and the refusal says
    how to state that they are segments; stated, they are left out one at a
    time as from the transform."""
    welch = Welch(time_series, sampling_frequency=_SAMPLING_FREQUENCY, segment_duration=0.5)
    coefficients = to_host(welch.fft())
    direct = Connectivity(coefficients, frequencies=welch.frequencies)
    assert not direct.observations_are_time_segments
    with pytest.raises(ValueError, match="cannot leave out one taper at a time") as error:
        direct.jackknife("magnitude_squared_coherence", frequency_bands=_BANDS)
    assert "Connectivity.from_transform(Welch(...))" in str(error.value)
    assert "observations_are_time_segments=True" in str(error.value)

    stated = Connectivity(
        coefficients, frequencies=welch.frequencies, observations_are_time_segments=True
    )
    np.testing.assert_array_equal(
        stated.jackknife("magnitude_squared_coherence", frequency_bands=_BANDS).standard_error,
        Connectivity.from_transform(welch)
        .jackknife("magnitude_squared_coherence", frequency_bands=_BANDS)
        .standard_error,
    )


class _DatasetsAsTimeWindows:
    """Welch coefficients of independent datasets, one per time window.

    ``Welch`` has one time bin, so its trials (here, the datasets) are moved to
    the time axis; the segments stay on the taper axis and are recorded as
    segments, as ``Welch`` records them.
    """

    observations_are_time_segments = True

    def __init__(self, welch):
        self._coefficients = np.swapaxes(to_host(welch.fft()), 0, 1)
        self.frequencies = welch.frequencies
        self.time = np.arange(self._coefficients.shape[0])
        self.observations_are_independent = welch.observations_are_independent

    def fft(self):
        return self._coefficients.copy()


@pytest.mark.parametrize(
    ("segment_overlap", "allowance"),
    # Measured 94.5-94.8% with disjoint segments and 92.2-93.8% at half
    # overlap (seeds 1-3), where neighbouring segments are slightly
    # correlated and the interval a little narrow.
    [(0.0, 0.01), (0.5, 0.02)],
)
def test_welch_segment_band_intervals_cover(segment_overlap, allowance):
    """Leaving out one Welch segment at a time, the 95% interval for the
    band-mean magnitude-squared coherence covers the band mean of the
    expected spectrum of a segment in about 95% of 2000 datasets of 512
    samples (8 disjoint or 15 half-overlapping 64-sample segments)."""
    n_datasets, n_time, n_segment_samples = 2000, 512, 64
    time_series = simulate_MVAR(
        _VAR_COEFFICIENTS,
        noise_covariance=_VAR_NOISE_COVARIANCE,
        n_time_samples=n_time,
        n_trials=n_datasets,
        n_burnin_samples=200,
        random_state=1,
    )  # (n_time, n_datasets, 2)
    welch = Welch(
        time_series,
        sampling_frequency=n_segment_samples,  # 1 Hz bins
        n_time_samples_per_segment=n_segment_samples,
        segment_overlap=segment_overlap,
    )
    connectivity = Connectivity.from_transform(_DatasetsAsTimeWindows(welch))
    frequencies = connectivity.frequencies
    # Coherence is scale-free, so the Hann window's scaling does not matter.
    expected = _expected_tapered_cross_spectrum(
        _VAR_COEFFICIENTS,
        _VAR_NOISE_COVARIANCE,
        to_host(welch._stft._tapers_for_fft()),
        n_segment_samples,
        welch.detrend_type,
    )[: frequencies.size]
    coherence = _magnitude_squared_coherence(expected)
    in_band = (frequencies >= 4) & (frequencies <= 10)

    result = connectivity.jackknife(
        "magnitude_squared_coherence", frequency_bands={"low": (4, 10)}
    )

    target = coherence[in_band].mean()
    lower, upper = (bound[:, 0, 0, 1] for bound in result.confidence_interval)
    coverage, limit = replicate_mean_and_limit(
        ((lower <= target) & (target <= upper)).astype(float)
    )
    assert result.n_observations == welch.n_segments
    assert abs(coverage - 0.95) <= limit + allowance, coverage


def test_delay_bands_are_bands_of_its_frequency_range():
    """``delay`` is frequency-resolved over the bins of its ``frequency_range``,
    which the bands select from, as in the wrappers: the band jackknife's
    estimate is the wrapper's band value, and each replicate the band value of
    the data without one trial."""
    rng = np.random.default_rng(0)
    leader = rng.standard_normal((503, 6))
    signals = np.stack([leader[3:], leader[:-3]], axis=-1)  # signal 1 lags by 3 samples
    signals += 0.5 * rng.standard_normal(signals.shape)
    bands = {"low": (10, 20), "high": (30, 40)}
    kwargs = {"frequency_range": (5, 50)}

    def connectivity(data):
        return Connectivity.from_transform(Multitaper(data, sampling_frequency=500))

    result = connectivity(signals).jackknife(
        "delay", leave_out="trial", frequency_bands=bands, keep_replicates=True, **kwargs
    )

    labeled = multitaper_connectivity(
        signals, sampling_frequency=500, method="delay", frequency_bands=bands, **kwargs
    )
    np.testing.assert_allclose(
        result.estimate,
        labeled.transpose("time", "band", ...).values,
        rtol=1e-12,
        equal_nan=True,
    )
    assert np.isfinite(result.estimate[0, :, 3, 0, 1]).all()  # the zero-wrap candidate

    def band_values(data):
        full = connectivity(data)
        selection = _select_measure_frequencies(full.frequencies, (5, 50), None)
        return frequency_band_reduce(
            xr.DataArray(
                full.delay(**kwargs),
                dims=("time", "frequency", "candidate", "source", "target"),
                coords={"frequency": full.frequencies[selection.mask]},
                attrs={"measure": "delay"},
            ),
            bands,
        ).values

    np.testing.assert_allclose(
        result.replicates,
        np.stack([band_values(np.delete(signals, k, axis=1)) for k in range(6)]),
        rtol=1e-12,
        equal_nan=True,
    )


def test_an_exact_spectrum_is_refused_first():
    """The exact-spectrum refusal comes before the band checks."""
    connectivity = Connectivity.from_var(
        np.array([[[0.5, 0.0], [0.3, 0.4]]]), sampling_frequency=100
    )
    with pytest.raises(ValueError, match=r"^jackknife is not defined"):
        connectivity.jackknife(
            "magnitude_squared_coherence", leave_out="trial", frequency_bands=_BANDS
        )


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
    n_time, n_trials = 64, 4
    time_series = _datasets_as_windows(
        n_time, n_trials, n_datasets=3000, random_state=20261127
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
    true_coherence = _magnitude_squared_coherence(expected)
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
        by_taper = connectivity.jackknife(
            method, leave_out="observation", keep_replicates=True
        )
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


# Oracles written from the definitions, not from the package's helpers: a mean
# over the bins in [low, high] (edges included), a circular mean as the angle
# of the mean unit phasor, and the integral of one-sided power as that of the
# two-sided piecewise-constant density over [low, high] and [-high, -low].
# Each replicate is recomputed from the time series without the trial.

_ORACLE_N_WINDOWS = 3


def _oracle_data(n_fft, seed=7, anti_phase=False):
    """``(3 * n_fft, 5, 2)``: three windows of five coupled trials."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((_ORACLE_N_WINDOWS * n_fft, _N_TRIALS, 2))
    if anti_phase:
        # Phase near +/-pi everywhere: a linear mean would land near 0.
        x[..., 1] = -x[..., 0] + 0.4 * x[..., 1]
    else:
        x[..., 1] += 0.6 * np.roll(x[..., 0], 2, axis=0)
    return x


def _oracle_connectivity(x, n_fft, expectation_type="trials_tapers", **mt_kwargs):
    """A Multitaper of ``x`` in windows of ``n_fft`` samples, on 1 Hz bins."""
    mt = quietly(
        Multitaper,
        x,
        sampling_frequency=n_fft,  # 1 Hz bins
        n_time_samples_per_window=n_fft,
        n_fft_samples=n_fft,
        time_halfbandwidth_product=2,
        **mt_kwargs,
    )
    return mt, Connectivity.from_transform(mt, expectation_type=expectation_type)


def _brute_mean(values, frequencies, low, high, axis, circular=False):
    """Mean (or circular mean) of ``values`` over the bins in ``[low, high]``."""
    mask = (frequencies >= low - 1e-9) & (frequencies <= high + 1e-9)
    selected = np.compress(mask, values, axis=axis)
    if circular:
        return np.angle(np.mean(np.exp(1j * selected), axis=axis))
    return np.mean(selected, axis=axis)


def _overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def _brute_integral(power, n_fft, low, high, axis):
    """Integral over the band of the two-sided density behind one-sided ``power``.

    ``power`` is one-sided on bins 0..n_fft//2 (1 Hz apart): interior bins
    hold twice the two-sided density, DC and (even n_fft) Nyquist hold it
    once. Each two-sided bin owns a 1 Hz cell; the Nyquist cell wraps around
    to -Nyquist.
    """
    low = max(low, 0.0)
    nyquist_bin = n_fft // 2 if n_fft % 2 == 0 else None
    weights = np.zeros(n_fft // 2 + 1)
    for k in range(n_fft):
        f = k if k <= n_fft // 2 else k - n_fft
        # The Nyquist cell wraps around to -Nyquist.
        cells = [(f - 0.5, f), (-f, -f + 0.5)] if k == nyquist_bin else [(f - 0.5, f + 0.5)]
        covered = sum(
            _overlap(c0, c1, low, high) + _overlap(c0, c1, -high, -low) for c0, c1 in cells
        )
        density_scale = 1.0 if k == 0 or k == nyquist_bin else 0.5
        weights[abs(f)] += density_scale * covered
    return np.tensordot(np.moveaxis(power, axis, -1), weights, axes=([-1], [0]))


def _bands_for(n_fft):
    top = n_fft // 2
    return {
        "dc": (0, 2.5),
        "edges_between_bins": (3.3, 7.6),
        "single_bin": (10, 10),
        "overlap_a": (8, 14),
        "overlap_b": (12, 20),
        "top": (top - 4, top),
        "everything": (0, top),
    }


def _reduce_all(values, frequencies, bands, reduction, axis, n_fft, circular=False):
    """Every band of ``bands`` by the oracles, on a band axis at ``axis``."""
    reduced = []
    for low, high in bands.values():
        if reduction == "integral":
            reduced.append(_brute_integral(values, n_fft, low, high, axis))
        else:
            reduced.append(_brute_mean(values, frequencies, low, high, axis, circular))
    return np.stack(reduced, axis=axis)


def _oracle_band_values(
    x,
    method,
    bands,
    reduction="mean",
    *,
    n_fft=64,
    axis=1,
    circular=False,
    build=None,
    method_kwargs=None,
):
    """The oracles' band values of ``method`` for ``x`` and for ``x`` without
    each trial: ``(estimate, replicates)``, the replicates stacked first.

    ``build`` makes the Connectivity of a time series (the oracle Multitaper
    on ``n_fft``-sample windows by default).
    """
    build = build or (lambda data: _oracle_connectivity(data, n_fft)[1])

    def band_values(data):
        connectivity = build(data)
        values = getattr(connectivity, method)(**(method_kwargs or {}))
        return _reduce_all(
            values, connectivity.frequencies, bands, reduction, axis, n_fft, circular
        )

    return band_values(x), np.stack(
        [band_values(np.delete(x, k, axis=1)) for k in range(x.shape[1])]
    )


@pytest.mark.parametrize("n_fft", [64, 63], ids=["even_fft", "odd_fft"])
@pytest.mark.parametrize("reduction", ["integral", "mean"])
def test_power_band_jackknife_matches_brute_force_on_deleted_trials(n_fft, reduction):
    """Band means and integrals of power, with edges between bins, a one-bin
    and a zero-width band, overlapping bands, and the DC and Nyquist bins
    (folded as a one-sided density), on even and odd FFT lengths."""
    x = _oracle_data(n_fft)
    _, connectivity = _oracle_connectivity(x, n_fft)
    bands = _bands_for(n_fft)

    with warnings.catch_warnings():
        # The zero-width band (10, 10) integrates to 0, which has no log.
        warnings.filterwarnings("ignore", "Jackknife log transform", UserWarning)
        result = connectivity.jackknife(
            "power",
            leave_out="trial",
            frequency_bands=bands,
            frequency_reduction=reduction,
            one_sided=True,
            keep_replicates=True,
        )

    estimate, replicates = _oracle_band_values(
        x, "power", bands, reduction, n_fft=n_fft, method_kwargs={"one_sided": True}
    )
    assert result.estimate.shape == (_ORACLE_N_WINDOWS, len(bands), 2)
    np.testing.assert_allclose(result.estimate, estimate, rtol=1e-12)
    np.testing.assert_allclose(result.replicates, replicates, rtol=1e-12)
    if reduction == "integral" and n_fft % 2 == 0:
        # Parseval: the whole grid integrates to the sum of the two-sided
        # density over every FFT bin (power(one_sided=False) holds its
        # non-negative half; each interior bin has a mirror image).
        half = connectivity.power(one_sided=False)
        two_sided_total = 2 * half.sum(axis=1) - half[:, 0] - half[:, -1]
        np.testing.assert_allclose(result.estimate[:, -1], two_sided_total, rtol=1e-12)


def test_circular_band_mean_of_phase_near_pi_with_trials_expectation():
    """``expectation_type="trials"`` keeps the taper axis, so the frequency
    axis is third; leaving out an observation is leaving out a trial. The
    phase sits near +/-pi, where a linear mean would be near 0."""
    n_fft = 64
    x = _oracle_data(n_fft, anti_phase=True)
    _, connectivity = _oracle_connectivity(x, n_fft, expectation_type="trials")
    bands = _bands_for(n_fft)

    result = connectivity.jackknife(
        "coherence_phase", frequency_bands=bands, keep_replicates=True
    )

    estimate, replicates = _oracle_band_values(
        x,
        "coherence_phase",
        bands,
        axis=2,
        circular=True,
        build=lambda data: _oracle_connectivity(data, n_fft, expectation_type="trials")[1],
    )
    assert result.estimate.shape[:3] == (_ORACLE_N_WINDOWS, 3, len(bands))  # 3 tapers kept
    assert result.transformation == "circular"
    off_diagonal = (..., 0, 1)
    assert np.all(np.abs(result.estimate[off_diagonal]) > 2.5)  # near +/-pi
    np.testing.assert_allclose(
        np.exp(1j * result.estimate[off_diagonal]),
        np.exp(1j * estimate[off_diagonal]),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.exp(1j * result.replicates[off_diagonal]),
        np.exp(1j * replicates[off_diagonal]),
        atol=1e-12,
    )


def test_directed_band_jackknife_matches_deleted_trials():
    n_fft = 64
    x = _oracle_data(n_fft)
    _, connectivity = _oracle_connectivity(x, n_fft)
    bands = {"low": (2, 9), "mid": (9, 20)}
    method = "pairwise_spectral_granger_prediction"

    result = connectivity.jackknife(
        method, leave_out="trial", frequency_bands=bands, keep_replicates=True
    )

    estimate, replicates = _oracle_band_values(x, method, bands)
    np.testing.assert_allclose(result.estimate, estimate, rtol=1e-9, equal_nan=True)
    np.testing.assert_allclose(result.replicates, replicates, rtol=1e-9, equal_nan=True)
    # 0 -> 1 is driven in this simulation; its band estimate is positive.
    assert np.all(result.estimate[:, :, 0, 1] > 0)


def test_missing_data_tapers_replicates_use_the_kept_tapers_weights():
    """Under ``missing_data_tapers`` a trial window with an artifact keeps only
    its concentrated tapers (others weight 0). Each trial replicate is the
    measure of the same masked data without that trial."""
    n_fft = 64
    x = _oracle_data(n_fft)
    x[70:74, 1, 0] = np.nan  # trial 1, window 1
    x[5:8, 3, 1] = np.nan  # trial 3, window 0
    kwargs = {"artifact_samples": "nonfinite", "artifact_handling": "missing_data_tapers"}
    mt, connectivity = _oracle_connectivity(x, n_fft, **kwargs)
    assert (to_host(mt.observation_weights) == 0).any()  # some tapers are dropped
    bands = {"a": (3, 9), "b": (10, 25)}

    for method, reduction in (("magnitude_squared_coherence", "mean"), ("power", "integral")):
        result = connectivity.jackknife(
            method,
            leave_out="trial",
            frequency_bands=bands,
            frequency_reduction=reduction,
            keep_replicates=True,
        )
        _, replicates = _oracle_band_values(
            x,
            method,
            bands,
            reduction,
            build=lambda data: _oracle_connectivity(data, n_fft, **kwargs)[1],
        )
        np.testing.assert_allclose(result.replicates, replicates, rtol=1e-10, equal_nan=True)
        # MSC's diagonal is NaN by policy; the rest has an interval.
        assert np.isfinite(
            result.standard_error[..., 0, 1] if method != "power" else result.standard_error
        ).all()


def test_a_trial_dropped_from_every_window_gives_no_replicate():
    """``replicates`` omits a unit the weights drop from every bin; the
    others are the measure without each remaining trial."""
    n_fft = 64
    x = _oracle_data(n_fft)
    for window in range(_ORACLE_N_WINDOWS):
        x[window * n_fft + 3, 2, 0] = np.nan  # trial 2 dropped everywhere
    mt_kwargs = {"artifact_samples": "nonfinite"}
    _, connectivity = _oracle_connectivity(x, n_fft, **mt_kwargs)
    bands = {"a": (3, 9), "b": (10, 25)}

    result = connectivity.jackknife(
        "magnitude_squared_coherence",
        leave_out="trial",
        frequency_bands=bands,
        keep_replicates=True,
    )

    _, replicates = _oracle_band_values(
        np.delete(x, 2, axis=1), "magnitude_squared_coherence", bands
    )
    assert result.replicates.shape[0] == _N_TRIALS - 1
    assert np.all(np.asarray(result.n_observations) == _N_TRIALS - 1)
    np.testing.assert_allclose(result.replicates, replicates, rtol=1e-11, equal_nan=True)


# --- Monte Carlo: band power integral (log) and circular band phase --------------


@pytest.fixture(scope="module")
def monte_carlo_connectivity():
    """3000 datasets of 6 trials x 64 samples, one per time window."""
    n_time = 64
    time_series = _datasets_as_windows(
        n_time, n_trials=6, n_datasets=3000, random_state=20261128
    )
    multitaper = Multitaper(
        time_series,
        sampling_frequency=n_time,
        n_time_samples_per_window=n_time,
        time_halfbandwidth_product=3,
    )
    return multitaper, Connectivity.from_transform(multitaper)


def test_band_power_integral_interval_covers_the_expected_band_power(monte_carlo_connectivity):
    """95% log-scale intervals for band power (an integral with fractional
    edge cells) cover the integral of the expected multitaper spectrum."""
    multitaper, connectivity = monte_carlo_connectivity
    bands = {"low": (3.5, 10.5), "high": (12.25, 19.75)}
    frequencies = connectivity.frequencies
    expected = _expected_multitaper_cross_spectrum(
        multitaper, _VAR_COEFFICIENTS, _VAR_NOISE_COVARIANCE
    )
    true_power = 2 * expected[: frequencies.size, 0, 0].real  # interior bins
    n_fft = 2 * (frequencies.size - 1)
    result = connectivity.jackknife(
        "power", leave_out="trial", frequency_bands=bands, frequency_reduction="integral"
    )
    nominal = 0.95
    for band, (low, high) in enumerate(bands.values()):
        target = _brute_integral(true_power[np.newaxis], n_fft, low, high, axis=1)[0]
        lower, upper = (bound[:, band, 0] for bound in result.confidence_interval)
        covered = ((lower <= target) & (target <= upper)).astype(float)
        coverage, limit = replicate_mean_and_limit(covered)
        # Beyond sampling error, the jackknife and Student t approximations
        # with 6 trials: measured 93.7-95.3% over both bands and seeds
        # 20261128 and 1-3.
        allowance = 0.01
        assert abs(coverage - nominal) <= limit + allowance, (band, coverage, limit)


def test_circular_band_phase_interval_covers_the_expected_band_phase(monte_carlo_connectivity):
    """95% circular intervals of the band-mean coherence phase cover the
    circular band mean of the expected spectrum's phase where coherence is
    strong."""
    multitaper, connectivity = monte_carlo_connectivity
    bands = {"low": (4, 10)}
    frequencies = connectivity.frequencies
    expected = _expected_multitaper_cross_spectrum(
        multitaper, _VAR_COEFFICIENTS, _VAR_NOISE_COVARIANCE
    )[: frequencies.size]
    phase = np.angle(expected[:, 0, 1])
    in_band = (frequencies >= 4) & (frequencies <= 10)
    target = np.angle(np.mean(np.exp(1j * phase[in_band])))
    result = connectivity.jackknife(
        "coherence_phase", leave_out="trial", frequency_bands=bands
    )
    lower, upper = (bound[:, 0, 0, 1] for bound in result.confidence_interval)
    # The interval [lower, upper] on the circle (lower > upper wraps).
    covered = np.where(
        lower <= upper,
        (lower <= target) & (target <= upper),
        (target >= lower) | (target <= upper),
    ).astype(float)
    coverage, limit = replicate_mean_and_limit(covered)
    # Measured 94.6-94.9% over seeds 20261128 and 1-3.
    allowance = 0.01
    assert abs(coverage - 0.95) <= limit + allowance, (coverage, limit)


def test_a_band_with_an_edge_invalid_bin_is_nan_in_every_replicate():
    """``MorletWavelet(edge_mode="nan")`` leaves the low frequencies of the
    first and last time bins without support: a band containing any such bin
    is NaN there in the estimate and in every replicate (no NaN-skipping),
    and a band with full support has an interval."""
    rng = np.random.default_rng(3)
    x = rng.standard_normal((600, _N_TRIALS, 2))
    x[..., 1] += 0.6 * np.roll(x[..., 0], 2, axis=0)
    frequencies = np.arange(4.0, 40.0, 2.0)
    kwargs = {
        "sampling_frequency": 200.0,
        "frequencies": frequencies,
        "edge_mode": "nan",
        "decimation": 50,
    }
    connectivity = Connectivity.from_transform(MorletWavelet(x, **kwargs))
    bands = {"low": (4, 12), "high": (20, 38)}

    # A time bin without support keeps no trial: it is NaN, without a warning.
    result = connectivity.jackknife(
        "magnitude_squared_coherence", frequency_bands=bands, keep_replicates=True
    )

    estimate, replicates = _oracle_band_values(
        x,
        "magnitude_squared_coherence",
        bands,
        n_fft=None,
        build=lambda data: Connectivity.from_transform(MorletWavelet(data, **kwargs)),
    )
    off_diagonal = (..., 0, 1)
    # Some, not all, (time, band) cells are NaN.
    assert 0 < np.isnan(estimate[off_diagonal]).sum() < estimate[off_diagonal].size
    np.testing.assert_array_equal(
        np.isnan(result.estimate[off_diagonal]), np.isnan(estimate[off_diagonal])
    )
    np.testing.assert_allclose(result.estimate, estimate, rtol=1e-11, equal_nan=True)
    np.testing.assert_allclose(result.replicates, replicates, rtol=1e-11, equal_nan=True)
    finite = ~np.isnan(estimate[off_diagonal])
    assert np.isfinite(result.standard_error[off_diagonal][finite]).all()


def test_one_taper_per_trial_with_artifacts_observation_equals_trial():
    """With one taper per trial each observation is a trial, also when
    artifacts drop trials from some windows: ``leave_out="observation"``
    gives the same replicates (NaN where the window lacks the trial) and
    interval as ``"trial"``."""
    n_fft = 64
    x = _oracle_data(n_fft)
    x[70, 1, 0] = np.nan  # trial 1 dropped from window 1
    x[5, 3, 1] = np.nan  # trial 3 dropped from window 0
    _, connectivity = _oracle_connectivity(x, n_fft, artifact_samples="nonfinite", n_tapers=1)
    bands = {"a": (3, 9), "b": (10, 25)}
    kwargs = {
        "frequency_bands": bands,
        "frequency_reduction": "integral",
        "keep_replicates": True,
    }
    by_observation = connectivity.jackknife("power", **kwargs)
    by_trial = connectivity.jackknife("power", leave_out="trial", **kwargs)
    assert np.isnan(by_trial.replicates[1, 1]).all()
    assert np.isnan(by_trial.replicates[3, 0]).all()
    for name in ("estimate", "standard_error", "replicates", "bias_corrected"):
        np.testing.assert_array_equal(
            getattr(by_observation, name), getattr(by_trial, name), err_msg=name
        )


# --- The bias-corrected interval center ------------------------------------


@pytest.mark.parametrize(
    ("method", "transformation"),
    [("magnitude_squared_coherence", "fisher_squared"), ("coherence_phase", "circular")],
)
def test_bias_corrected_center_is_the_hand_computed_interval(
    time_series, method, transformation
):
    """``interval_center="bias_corrected"`` moves the interval, on the
    transformed scale, from ``g(estimate)`` to ``n g(estimate) - (n - 1)
    mean(g(replicates))``; the standard error and bias-corrected estimate are
    those of the default."""
    _, connectivity = _connectivity(time_series)
    kwargs = {"leave_out": "trial", "frequency_bands": _BANDS, "keep_replicates": True}
    default = connectivity.jackknife(method, **kwargs)
    centered = connectivity.jackknife(method, interval_center="bias_corrected", **kwargs)

    n = _N_TRIALS
    estimate, replicates = default.estimate, default.replicates
    if transformation == "fisher_squared":
        transformed_estimate = np.arctanh(np.sqrt(estimate))
        transformed = np.arctanh(np.sqrt(replicates))
        inverse = lambda value: np.clip(np.tanh(value), 0, 1) ** 2  # noqa: E731
    else:
        transformed_estimate = estimate
        transformed = estimate + np.angle(np.exp(1j * (replicates - estimate)))
        inverse = lambda value: np.angle(np.exp(1j * value))  # noqa: E731
    center = n * transformed_estimate - (n - 1) * transformed.mean(axis=0)
    half_width = scipy.stats.t.ppf(0.975, df=n - 1) * np.sqrt(
        (n - 1) / n * ((transformed - transformed.mean(axis=0)) ** 2).sum(axis=0)
    )

    assert (default.interval_center, centered.interval_center) == (
        "estimate",
        "bias_corrected",
    )
    np.testing.assert_array_equal(centered.standard_error, default.standard_error)
    np.testing.assert_array_equal(centered.bias_corrected, default.bias_corrected)
    off_diagonal = (..., 0, 1)
    for bound, expected in zip(
        centered.confidence_interval,
        (inverse(center - half_width), inverse(center + half_width)),
        strict=True,
    ):
        np.testing.assert_allclose(bound[off_diagonal], expected[off_diagonal], rtol=1e-12)
    # The public function centers the same way.
    _assert_same_interval(
        jackknife_confidence_interval(
            estimate,
            replicates,
            transformation=transformation,
            interval_center="bias_corrected",
        ),
        centered,
    )


_EPS = np.finfo(float).eps
_FORWARD = {
    "identity": lambda value: value,
    "log": np.log,
    "fisher": lambda value: np.arctanh(np.clip(value, -1 + _EPS, 1 - _EPS)),
    "fisher_squared": lambda value: np.arctanh(np.clip(np.sqrt(value), 0, 1 - _EPS)),
}
_INVERSE = {
    "identity": lambda value: value,
    "log": np.exp,
    "fisher": np.tanh,
    "fisher_squared": lambda value: np.clip(np.tanh(value), 0, 1) ** 2,
    "circular": lambda value: np.angle(np.exp(1j * value)),
}


@pytest.mark.parametrize("transformation", list(_INVERSE))
def test_bias_corrected_bounds_by_hand_for_every_transformation(transformation):
    """Through the public function, for every transformation (with a circular
    interval crossing +/-pi): the bounds are the back-transform of the
    bias-corrected center plus and minus the half-width, unclipped."""
    rng = np.random.default_rng(0)
    estimate = rng.uniform(0.05, 0.9, size=(8, 5))
    replicates = np.clip(estimate + 0.15 * rng.standard_normal((9, 8, 5)), 0.001, 0.999)
    if transformation == "circular":
        estimate = rng.uniform(-np.pi, np.pi, size=(8, 5))
        replicates = estimate + 0.2 * rng.standard_normal((9, 8, 5))
        estimate[0, 0] = 3.1
        replicates[:, 0, 0] = 3.1 + 0.3 * rng.standard_normal(9)
        transformed_estimate = estimate
        transformed = estimate + np.angle(np.exp(1j * (replicates - estimate)))
    else:
        transformed_estimate = _FORWARD[transformation](estimate)
        transformed = _FORWARD[transformation](replicates)
    n = replicates.shape[0]
    mean = transformed.mean(axis=0)
    center = n * transformed_estimate - (n - 1) * mean
    half_width = scipy.stats.t.ppf(0.975, df=n - 1) * np.sqrt(
        (n - 1) / n * ((transformed - mean) ** 2).sum(axis=0)
    )

    result = jackknife_confidence_interval(
        estimate, replicates, transformation=transformation, interval_center="bias_corrected"
    )

    assert result.interval_center == "bias_corrected"
    for bound, expected in zip(
        result.confidence_interval,
        (
            _INVERSE[transformation](center - half_width),
            _INVERSE[transformation](center + half_width),
        ),
        strict=True,
    ):
        np.testing.assert_allclose(bound, expected, rtol=1e-12, atol=1e-15)
    default = jackknife_confidence_interval(
        estimate, replicates, transformation=transformation
    )
    np.testing.assert_array_equal(result.standard_error, default.standard_error)
    np.testing.assert_array_equal(result.bias_corrected, default.bias_corrected)


@pytest.mark.parametrize("method", ["imaginary_coherence", "phase_locking_value"])
def test_a_magnitudes_bias_corrected_interval_is_never_inverted(method):
    """A bias-corrected center more than a half-width below 0 left the upper
    bound below the lower one clamped at 0 (seed 1, 5 uncoupled trials:
    imaginary_coherence at 44 and 60 Hz); both bounds are clamped now."""
    signals = np.random.default_rng(1).standard_normal((100, 5, 2))
    _, connectivity = _connectivity(signals)
    result = quietly(
        connectivity.jackknife, method, leave_out="trial", interval_center="bias_corrected"
    )
    lower, upper = (bound[..., 0, 1] for bound in result.confidence_interval)
    assert np.all(lower >= 0)
    assert np.all(upper >= lower)


def test_a_bias_corrected_interval_stays_in_the_measures_range():
    """For uncoupled signals the bias-corrected band Granger interval can lie
    wholly below 0 (seed 27, the low band); clipped into Granger's range
    [0, inf) it collapses to [0, 0], and nowhere does a bound leave it. The
    public function, which knows no measure, does not clip."""
    signals = np.random.default_rng(27).standard_normal((256, 10, 2))
    connectivity = Connectivity.from_transform(
        Multitaper(
            signals,
            sampling_frequency=1.0,
            n_time_samples_per_window=256,
            time_halfbandwidth_product=3,
        )
    )
    result = connectivity.jackknife(
        "pairwise_spectral_granger_prediction",
        leave_out="trial",
        frequency_bands={"low": (0.05, 0.15), "mid": (0.2, 0.3)},
        interval_center="bias_corrected",
        keep_replicates=True,
    )
    lower, upper = result.confidence_interval
    off_diagonal = ~np.eye(2, dtype=bool)
    assert np.all(lower[..., off_diagonal] >= 0)
    assert np.all(upper[..., off_diagonal] >= lower[..., off_diagonal])
    assert (lower[0, 0, 0, 1], upper[0, 0, 0, 1]) == (0.0, 0.0)
    unclipped = jackknife_confidence_interval(
        result.estimate,
        result.replicates,
        transformation=result.transformation,
        interval_center="bias_corrected",
    )
    assert unclipped.confidence_interval[1][0, 0, 0, 1] < 0
    np.testing.assert_array_equal(
        np.clip(unclipped.confidence_interval[0], 0, np.inf)[..., off_diagonal],
        lower[..., off_diagonal],
    )


@pytest.mark.parametrize("method", ["magnitude_squared_coherence", "power"])
def test_public_function_reproduces_a_weighted_band_bias_corrected_interval(
    time_series, method
):
    """With artifacts, bands and kept replicates, the public function given the
    method's estimate and replicates reproduces its bias-corrected interval."""
    _, connectivity = _connectivity(_with_artifacts(time_series))
    result = quietly(
        connectivity.jackknife,
        method,
        leave_out="trial",
        frequency_bands={"theta": (4, 8), "beta": (13, 30)},
        keep_replicates=True,
        interval_center="bias_corrected",
    )
    public = jackknife_confidence_interval(
        result.estimate,
        result.replicates,
        transformation=result.transformation,
        interval_center="bias_corrected",
    )
    assert result.interval_center == public.interval_center == "bias_corrected"
    for got, want in zip(public.confidence_interval, result.confidence_interval, strict=True):
        np.testing.assert_array_equal(got, want)


def test_an_unknown_interval_center_is_refused(time_series):
    _, connectivity = _connectivity(time_series)
    with pytest.raises(
        ValueError, match="interval_center must be 'estimate' or 'bias_corrected'"
    ):
        connectivity.jackknife("power", interval_center="median")
    with pytest.raises(ValueError, match="interval_center must be"):
        jackknife_confidence_interval(np.ones(2), np.ones((3, 2)), interval_center="median")


def test_bias_corrected_center_covers_a_biased_band_partial_coherence():
    """A band-mean partial coherence is biased upward by more than its spread,
    since averaging over the band's bins shrinks the spread but not the bias:
    the interval centered on the estimate covers the band mean of the expected
    spectrum's partial coherence far less than 95%, and the one centered on
    the bias-corrected estimate about 95%.

    A dense 5-signal VAR, 300 datasets of 10 trials of 512 samples (NW = 3, 50
    observations) on the time-window axis; the pair with the weakest partial
    coherence in the band (0.015), where the bias is largest relative to the
    spread (2.4 standard deviations).
    """
    coefficients, noise_covariance = _dense_stable_var(
        n_signals=5, n_lags=3, spectral_radius=0.95, seed=3
    )
    n_time, n_trials, n_datasets = 512, 10, 300
    segments = simulate_MVAR(
        coefficients,
        noise_covariance=noise_covariance,
        n_time_samples=n_time,
        n_trials=n_datasets * n_trials,
        n_burnin_samples=300,
        random_state=1,
    )
    multitaper = Multitaper(
        segments.reshape(n_time, n_datasets, n_trials, 5)
        .transpose(1, 0, 2, 3)
        .reshape(n_datasets * n_time, n_trials, 5),
        sampling_frequency=1.0,
        n_time_samples_per_window=n_time,
        time_halfbandwidth_product=3,
    )
    connectivity = Connectivity.from_transform(multitaper)
    band = (0.10, 0.20)
    frequencies = connectivity.frequencies
    in_band = (frequencies >= band[0]) & (frequencies <= band[1])
    exact = Connectivity(
        _fourier_coefficients_with_cross_spectrum(
            _expected_multitaper_cross_spectrum(multitaper, coefficients, noise_covariance),
            conjugate_symmetric=True,
        )
    )
    truth = exact.partial_coherence()[0][in_band].mean(axis=0)
    source, target = 0, 4
    assert truth[source, target] == truth[np.triu_indices(5, 1)].min()

    result = connectivity.jackknife(
        "partial_coherence",
        leave_out="trial",
        frequency_bands={"band": band},
        keep_replicates=True,
    )

    coverage = {}
    for center in ("estimate", "bias_corrected"):
        interval = jackknife_confidence_interval(
            result.estimate[:, 0, source, target],
            result.replicates[:, :, 0, source, target],
            transformation=result.transformation,
            interval_center=center,
        )
        lower, upper = interval.confidence_interval
        coverage[center] = replicate_mean_and_limit(
            ((lower <= truth[source, target]) & (truth[source, target] <= upper)).astype(float)
        )
    estimate_coverage, estimate_limit = coverage["estimate"]
    corrected, corrected_limit = coverage["bias_corrected"]
    # Measured 40-43% (seeds 1-3): more than 3 of its limits below nominal.
    assert 0.95 - estimate_coverage > 3 * estimate_limit, coverage
    # Measured 91.7-93.3% (seeds 1-3); the allowance is for the jackknife's
    # residual bias and the Student t approximation.
    assert abs(corrected - 0.95) <= corrected_limit + 0.02, coverage


def test_trial_band_mean_coherence_under_covers_at_weak_coherence():
    """With 5 trials and a band-mean coherence of 0.073 the estimate's upward
    bias makes the trial interval cover only about 87%; centering it on the
    bias-corrected estimate brings it to about 92%, and a moderate band of the
    same call covers about 94% either way.

    3000 datasets of 5 trials of 256 samples (NW = 3) on the time-window axis.
    Measured over seeds 11 and 1-3: weak band 86.6-88.0% (estimate) and
    91.6-93.1% (bias-corrected), moderate band 93.3-94.3% and 93.8-94.7%.
    """
    multitaper = Multitaper(
        _datasets_as_windows(256, 5, n_datasets=3000, random_state=11),
        sampling_frequency=1.0,
        n_time_samples_per_window=256,
        time_halfbandwidth_product=3,
    )
    connectivity = Connectivity.from_transform(multitaper)
    frequencies = connectivity.frequencies
    true_coherence = _magnitude_squared_coherence(
        _expected_multitaper_cross_spectrum(
            multitaper, _VAR_COEFFICIENTS, _VAR_NOISE_COVARIANCE
        )[: frequencies.size]
    )
    bands = {"weak": (0.30, 0.40), "moderate": (0.15, 0.25)}
    result = connectivity.jackknife(
        "magnitude_squared_coherence",
        leave_out="trial",
        frequency_bands=bands,
        keep_replicates=True,
    )
    coverage = {}
    for index, (name, (low, high)) in enumerate(bands.items()):
        target = true_coherence[(frequencies >= low) & (frequencies <= high)].mean()
        for center in ("estimate", "bias_corrected"):
            lower, upper = jackknife_confidence_interval(
                result.estimate[:, index, 0, 1],
                result.replicates[:, :, index, 0, 1],
                transformation="fisher_squared",
                interval_center=center,
            ).confidence_interval
            coverage[name, center] = replicate_mean_and_limit(
                ((lower <= target) & (target <= upper)).astype(float)
            )
    weak, weak_limit = coverage["weak", "estimate"]
    assert weak + weak_limit < 0.91, coverage
    for key, allowance in (
        (("weak", "bias_corrected"), 0.015),
        (("moderate", "estimate"), 0.01),
        (("moderate", "bias_corrected"), 0.01),
    ):
        value, limit = coverage[key]
        assert abs(value - 0.95) <= limit + allowance, (key, coverage)


# --- Figures the Notes cite --------------------------------------------------


@pytest.mark.parametrize("time_halfbandwidth_product", [3, 4])
def test_different_tapers_are_correlated_a_few_bins_apart(time_halfbandwidth_product):
    """For white noise the correlation of tapers j and k's coefficients at
    frequencies d bins apart is |sum_t h_j(t) h_k(t) exp(-2 pi i d t / N)| for
    unit-energy tapers: 0 at d = 0 (orthogonal tapers) but, for d of 1 to 3
    bins, up to 0.4-0.6 over pairs of different tapers (median 0.2-0.4), as
    the jackknife Notes say (measured: 0.42-0.59, median 0.25-0.39)."""
    multitaper = Multitaper(
        np.zeros((256, 1, 1)),
        sampling_frequency=1.0,
        time_halfbandwidth_product=time_halfbandwidth_product,
    )
    tapers = to_host(multitaper.tapers).T  # (n_tapers, n_time)
    tapers = tapers / np.linalg.norm(tapers, axis=1, keepdims=True)
    different = ~np.eye(len(tapers), dtype=bool)
    time = np.arange(tapers.shape[1])
    np.testing.assert_allclose(
        (tapers @ tapers.T)[different], 0, atol=1e-10
    )  # uncorrelated at one frequency
    for offset in (1, 2, 3):
        shifted = tapers * np.exp(-2j * np.pi * offset * time / tapers.shape[1])
        correlation = np.abs(shifted @ tapers.T)[different]
        assert 0.4 <= correlation.max() <= 0.6, (offset, correlation.max())
        assert 0.2 <= np.median(correlation) <= 0.4, (offset, np.median(correlation))


def test_phase_slope_index_intervals_cover_with_either_unit():
    """``phase_slope_index`` combines the bins of its ``frequency_range``
    inside its own fit, and its leave-one-taper-out interval is not too narrow
    the way a band mean's is: with 5 trials of 128 samples (NW = 3), both
    leave_out settings cover the PSI of the expected spectrum about 95% of
    1000 datasets (measured, seeds 1-3: 95.3-96.4% leaving out tapers,
    93.6-94.9% leaving out trials)."""
    coefficients = np.array([[[1.5, 0.0], [0.4, 0.5]], [[-0.9, 0.0], [0.0, 0.0]]])
    n_time, n_trials, n_datasets = 128, 5, 1000
    segments = simulate_MVAR(
        coefficients,
        noise_covariance=np.eye(2),
        n_time_samples=n_time,
        n_trials=n_datasets * n_trials,
        n_burnin_samples=300,
        random_state=1,
    )
    multitaper = Multitaper(
        segments.reshape(n_time, n_datasets, n_trials, 2)
        .transpose(1, 0, 2, 3)
        .reshape(n_datasets * n_time, n_trials, 2),
        sampling_frequency=1.0,
        n_time_samples_per_window=n_time,
        time_halfbandwidth_product=3,
    )
    connectivity = Connectivity.from_transform(multitaper)
    kwargs = {"frequency_range": (0.08, 0.13)}
    target = Connectivity(
        _fourier_coefficients_with_cross_spectrum(
            _expected_multitaper_cross_spectrum(multitaper, coefficients, np.eye(2)),
            conjugate_symmetric=True,
        )
    ).phase_slope_index(**kwargs)[0, 0, 1]
    for leave_out in ("observation", "trial"):
        lower, upper = (
            bound[:, 0, 1]
            for bound in connectivity.jackknife(
                "phase_slope_index", leave_out=leave_out, **kwargs
            ).confidence_interval
        )
        coverage, limit = replicate_mean_and_limit(
            ((lower <= target) & (target <= upper)).astype(float)
        )
        assert abs(coverage - 0.95) <= limit + 0.01, (leave_out, coverage)


# --- The band estimate is the labeled band value -----------------------------

_TRANSFORM_BANDS = {"theta": (4, 8), "alpha": (8.3, 12.7), "beta": (13, 30), "top": (80, 100)}


def _transforms(x):
    return {
        "multitaper": Multitaper(
            x,
            sampling_frequency=200,
            n_time_samples_per_window=100,
            time_halfbandwidth_product=2,
        ),
        "multitaper_odd": Multitaper(
            x,
            sampling_frequency=200,
            n_time_samples_per_window=99,
            time_halfbandwidth_product=2,
        ),
        "welch": Welch(x, sampling_frequency=200, segment_duration=0.25, segment_overlap=0.0),
        "stft": ShortTimeFourierTransform(x, sampling_frequency=200, time_window_duration=0.5),
        "morlet": MorletWavelet(
            x[:200], sampling_frequency=200, frequencies=np.arange(6, 99, 2.0)
        ),
    }


@pytest.mark.parametrize("name", ["multitaper", "multitaper_odd", "welch", "stft", "morlet"])
@pytest.mark.parametrize(
    ("method", "reduction", "kwargs"),
    [
        ("power", "integral", {"one_sided": True}),
        ("magnitude_squared_coherence", "mean", {}),
        ("coherence_phase", "mean", {}),
        ("delay", "mean", {"frequency_range": (5, 40)}),
    ],
)
def test_band_estimate_is_the_labeled_band_value_for_every_transform(
    time_series, name, method, reduction, kwargs
):
    """``jackknife(frequency_bands=...).estimate`` is what
    ``connectivity_to_xarray(transform, frequency_bands=...)`` returns, for
    each transform: two-sided even and odd FFT lengths, Welch, STFT, and the
    one-sided Morlet."""
    transform = quietly(lambda: _transforms(time_series)[name])
    bands = _TRANSFORM_BANDS if method != "delay" else {"a": (5, 12), "b": (12.5, 40)}
    leave_out = "observation" if name == "welch" else "trial"
    # The whole-circle warnings of phase intervals.
    result = quietly(
        Connectivity.from_transform(transform).jackknife,
        method,
        leave_out=leave_out,
        frequency_bands=bands,
        frequency_reduction=reduction,
        **kwargs,
    )
    wrapper_kwargs = (
        {"connectivity_kwargs": {"one_sided": kwargs["one_sided"]}}
        if "one_sided" in kwargs
        else kwargs
    )
    labeled = quietly(
        connectivity_to_xarray,
        transform,
        method,
        frequency_bands=bands,
        frequency_reduction=reduction,
        **wrapper_kwargs,
    )
    np.testing.assert_array_equal(result.estimate, labeled.values)
