"""Shrinkage of partial coherence toward the diagonal (``shrinkage=``).

The definition of ``"auto"``, the limits, and the error against the VAR's
partial coherence are in ``test_partial_coherence_oracles.py``; these are its
arguments, warnings, caching and behavior with few observations. The Monte Carlo tests
draw complex Gaussian eigencoefficients whose expected cross-spectral matrix
is the exact spectrum of a VAR (``tests/_var_oracle.py``), with replicate
datasets on the time-window axis, and compare against the VAR's exact partial
coherence, from the inverse of that spectrum. Limits are five standard errors
of the replicate mean; an allowance that is not sampling error is stated
beside its measured value.
"""

import re
import warnings
from unittest.mock import patch

import numpy as np
import pytest

from spectral_connectivity import Connectivity, Multitaper, multitaper_connectivity
from tests._var_oracle import (
    _analytic_var,
    _coherency,
    _companion_spectral_radius,
    _gaussian_coefficients,
    _mean_and_limit,
    _normalized_inverse,
)

# Bins of a 32-point FFT grid at which the VAR spectra are sampled.
_FREQUENCY_BINS = [1, 3, 5, 8]


def _chain_var(n_signals):
    """VAR(2) chain: each signal drives the next, each oscillates on its own."""
    coefficients = np.zeros((2, n_signals, n_signals))
    coefficients[0] = 0.5 * np.eye(n_signals) + 0.45 * np.eye(n_signals, k=-1)
    coefficients[1] = -0.3 * np.eye(n_signals)
    return coefficients


def _dense_var(n_signals):
    """VAR(1) with random couplings between about 30% of the pairs."""
    rng = np.random.default_rng(n_signals)
    while True:
        coefficients = rng.standard_normal((1, n_signals, n_signals)) * 0.5
        coefficients *= rng.random((1, n_signals, n_signals)) < 0.3
        coefficients[0] += np.diag(rng.uniform(0.2, 0.6, n_signals))
        if np.max(np.abs(np.linalg.eigvals(coefficients[0]))) < 0.9:
            return coefficients


_MODELS = {"chain": _chain_var, "dense": _dense_var}


def _var_replicates(model, n_signals, n_observations, n_replicates, seed=0):
    """``(connectivity, true_spectrum)`` for replicate datasets of a VAR.

    Each replicate (time window) holds ``n_observations`` complex Gaussian
    observations per frequency, drawn with the VAR's exact spectrum ``S(f)``
    at ``_FREQUENCY_BINS`` as their covariance.
    """
    coefficients = _MODELS[model](n_signals)
    assert _companion_spectral_radius(coefficients) < 1
    _, _, spectrum = _analytic_var(coefficients, np.eye(n_signals), 32)
    spectrum = spectrum[_FREQUENCY_BINS]
    observations = _gaussian_coefficients(spectrum, n_replicates, n_observations, seed=seed)
    return Connectivity(observations, is_one_sided=True), spectrum


def _squared_error(estimate, truth):
    """Mean squared error over pairs and frequencies, one per replicate."""
    off_diagonal = ~np.eye(truth.shape[-1], dtype=bool)
    return ((estimate[..., off_diagonal] - truth[..., off_diagonal]) ** 2).mean(axis=(-2, -1))


def _coefficients(rng, shape, mixing=0.6):
    """Complex Gaussian coefficients with correlated signals."""
    n_signals = shape[-1]
    white = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    return white @ (np.eye(n_signals) + mixing * rng.standard_normal((n_signals, n_signals)))


@pytest.mark.parametrize(
    ("shrinkage", "hint"),
    [
        (-0.1, ""),
        (1.5, ""),
        (np.nan, ""),
        (np.inf, ""),
        ([0.1], ""),
        (True, " Booleans are not accepted."),
        ("Auto", " Strings are case-sensitive."),
    ],
)
def test_rejects_invalid_shrinkage(shrinkage, hint):
    """The message says what each accepted value means."""
    coefficients = np.ones((1, 2, 2, 2, 2), dtype=complex)
    with pytest.raises(
        ValueError,
        match=r"^shrinkage must be None \(no shrinkage\), 'auto' \(estimate the "
        r"intensity for each time window and frequency\), or a number from 0 to 1 "
        rf"\(a fixed intensity\), got .*\.{re.escape(hint)}$",
    ):
        Connectivity(coefficients).partial_coherence(shrinkage=shrinkage)


def test_auto_intensity_is_the_same_for_reordered_signals():
    """Reordering the signals leaves the intensity unchanged, and the shrunk
    estimate is symmetric (rescaling is tested in test_connectivity.py)."""
    rng = np.random.default_rng(3)
    coefficients = _coefficients(rng, (2, 5, 2, 6, 5))
    order = rng.permutation(5)
    intensity = Connectivity(coefficients).partial_coherence_shrinkage()
    np.testing.assert_allclose(
        Connectivity(coefficients[..., order]).partial_coherence_shrinkage(),
        intensity,
        rtol=1e-12,
    )
    auto = Connectivity(coefficients).partial_coherence(shrinkage="auto")
    np.testing.assert_allclose(auto, np.swapaxes(auto, -1, -2), rtol=1e-12, equal_nan=True)


@pytest.mark.parametrize("model", ["chain", "dense"])
def test_auto_intensity_falls_as_one_over_n_and_results_approach_unshrunk(model):
    """With many observations the intensity is the first-order value from the
    true coherencies, ``sum v / sum (|rho|^2 + v)`` with
    ``v = (1 - |rho|^2) (1 - |rho|^2 / 2) / n``, which falls as ``1 / n``; and
    the shrunk estimate approaches the unshrunk one.

    The 3% allowance is for the second-order terms that value leaves out; the
    differences at 400 observations (8 signals), at most 1.4%, were within the
    sampling limits (1.3-2.6%). From 40 to 400 observations the RMS change
    made by shrinking fell 8.0-10.7 fold."""
    n_signals = 8
    changes = []
    for n_observations in (40, 400):
        connectivity, spectrum = _var_replicates(model, n_signals, n_observations, 200, seed=2)
        unshrunk = connectivity.partial_coherence()
        shrunk = connectivity.partial_coherence(shrinkage="auto")
        off_diagonal = ~np.eye(n_signals, dtype=bool)
        changes.append(np.sqrt(np.mean((shrunk - unshrunk)[..., off_diagonal] ** 2)))
    squared_rho = np.abs(_coherency(spectrum)[..., off_diagonal]) ** 2
    variance = (1 - squared_rho) * (1 - squared_rho / 2) / n_observations
    first_order = variance.sum(axis=-1) / (squared_rho + variance).sum(axis=-1)

    mean, limit = _mean_and_limit(connectivity.partial_coherence_shrinkage())
    assert np.all(np.abs(mean - first_order) <= 0.03 * first_order + limit)
    assert np.all(mean < 0.05)
    assert changes[1] < 0.15 * changes[0]


@pytest.mark.parametrize("model", ["chain", "dense"])
@pytest.mark.parametrize(("n_signals", "n_observations"), [(8, 7), (8, 4), (16, 15), (16, 8)])
def test_auto_shrinkage_is_usable_with_fewer_observations_than_signals(
    model, n_signals, n_observations
):
    """Without shrinkage the rank-deficient estimate is forced toward 1 (mean
    squared error 0.91-0.95 with one observation missing) or is unrelated to
    the data. With ``"auto"`` it is finite and its error is below that of
    estimating 0 for every pair, by 16 to 63 limits (0.0031-0.0100 against
    0.0052-0.0128), and below that of no shrinkage (0.0084-0.0377 with more
    than one observation missing). The warning says the estimate rests on the
    intensity."""
    connectivity, spectrum = _var_replicates(model, n_signals, n_observations, 400, seed=3)
    truth = _normalized_inverse(spectrum)
    with pytest.warns(
        UserWarning, match=r"see Connectivity\.partial_coherence_shrinkage\(\)\), not"
    ):
        shrunk = connectivity.partial_coherence(shrinkage="auto")
    off_diagonal = ~np.eye(n_signals, dtype=bool)
    assert np.isfinite(shrunk[..., off_diagonal]).all()
    error = _squared_error(shrunk, truth)
    mean, limit = _mean_and_limit(error - _squared_error(np.zeros_like(shrunk), truth))
    assert mean + limit < 0
    with pytest.warns(
        UserWarning, match="dominated by the null space: the estimate is forced"
    ):
        unshrunk = _squared_error(connectivity.partial_coherence(), truth)
    mean, limit = _mean_and_limit(error - unshrunk)
    assert mean + limit < 0


def test_zero_shrinkage_warns_as_no_shrinkage_with_few_observations():
    rng = np.random.default_rng(4)
    connectivity = Connectivity(_coefficients(rng, (1, 1, 5, 4, 6)))
    for shrinkage in (None, 0.0):
        with pytest.warns(UserWarning, match="the estimate is forced to 1"):
            connectivity.partial_coherence(shrinkage=shrinkage)
    with pytest.warns(
        UserWarning,
        match=r"only the shrinkage makes it invertible: the result is "
        r"set by the shrinkage intensity \(0\.2\), not conditioned",
    ):
        connectivity.partial_coherence(shrinkage=0.2)


def test_auto_shrinkage_needs_two_observations():
    """Each error names its call; only partial_coherence offers a fixed
    shrinkage. A fixed one runs, and its warning says the shrinkage, not the
    data, sets the result (which is the same for any data)."""
    rng = np.random.default_rng(5)
    connectivity = Connectivity(_coefficients(rng, (1, 1, 1, 4, 3)))
    with pytest.raises(
        ValueError,
        match=r'^partial_coherence\(shrinkage="auto"\) requires at least 2 .* or a fixed '
        "shrinkage between 0 and 1.$",
    ):
        connectivity.partial_coherence(shrinkage="auto")
    with pytest.raises(
        ValueError, match=r"^partial_coherence_shrinkage\(\) requires"
    ) as error:
        connectivity.partial_coherence_shrinkage()
    assert "fixed shrinkage" not in str(error.value)
    with pytest.warns(
        UserWarning,
        match=r"uses 1 observation .* set by the shrinkage intensity \(0\.5\), not "
        r"conditioned .* With one observation it does not depend on the data\.",
    ):
        shrunk = connectivity.partial_coherence(shrinkage=0.5)
    other = Connectivity(_coefficients(rng, (1, 1, 1, 4, 3)))
    with pytest.warns(UserWarning, match="uses 1 observation"):
        np.testing.assert_allclose(
            other.partial_coherence(shrinkage=0.5), shrunk, atol=1e-12, equal_nan=True
        )


def test_auto_intensity_is_nan_where_partial_coherence_is_undefined():
    """A frequency with zero power, or NaN coefficients, has no intensity;
    signals with zero power are left out of it."""
    rng = np.random.default_rng(6)
    coefficients = _coefficients(rng, (1, 6, 2, 4, 3))
    coefficients[..., 1, :] = 0.0  # a dead frequency
    coefficients[0, 2, 1, 2, 0] = np.nan  # one NaN coefficient at frequency 2
    with_dead_signals = np.concatenate([coefficients, np.zeros_like(coefficients)], axis=-1)
    with pytest.warns(UserWarning, match="NaN or Inf"):
        connectivity = Connectivity(coefficients, is_one_sided=True)
    with pytest.warns(UserWarning, match="NaN or Inf"):
        dead_signals = Connectivity(with_dead_signals, is_one_sided=True)
    intensity = connectivity.partial_coherence_shrinkage()
    assert np.isnan(intensity[0, [1, 2]]).all()
    assert np.isfinite(intensity[0, [0, 3]]).all()
    np.testing.assert_allclose(
        dead_signals.partial_coherence_shrinkage(), intensity, rtol=1e-14
    )

    with pytest.warns(UserWarning, match="zero power"):
        shrunk = connectivity.partial_coherence(shrinkage="auto")
    assert np.isnan(shrunk[0, [1, 2]]).all()
    assert np.isfinite(shrunk[0, [0, 3]][..., ~np.eye(3, dtype=bool)]).all()


def test_auto_shrinkage_refuses_weights_that_reweight_observations():
    rng = np.random.default_rng(7)
    coefficients = _coefficients(rng, (1, 6, 2, 4, 3))
    weights = rng.uniform(0.5, 1.5, (1, 6, 2, 4, 1))
    connectivity = Connectivity(coefficients, observation_weights=weights)
    with pytest.raises(
        ValueError,
        match=r'^partial_coherence\(shrinkage="auto"\) does not support non-uniform '
        r"observation_weights\. The intensity .* observations; use a fixed shrinkage "
        r"between 0 and 1 instead\. Non-uniform",
    ):
        connectivity.partial_coherence(shrinkage="auto")
    with pytest.raises(
        ValueError,
        match=r"^partial_coherence_shrinkage\(\) does not support non-uniform "
        r"observation_weights\. The intensity .* independent observations\. Non-uniform",
    ) as error:
        connectivity.partial_coherence_shrinkage()
    assert "fixed shrinkage" not in str(error.value)
    connectivity.partial_coherence(shrinkage=0.3)  # a fixed intensity counts nothing


def test_auto_intensity_counts_only_the_observations_weights_keep(monkeypatch):
    """Equal weights change nothing, and weights that only drop observations
    give the intensity of the observations they keep. Such weights are refused
    today by the guard against reweighting, which this bypasses: the intensity
    must still count each bin's kept observations once that guard admits them."""
    rng = np.random.default_rng(8)
    coefficients = _coefficients(rng, (2, 8, 2, 4, 3))
    intensity = Connectivity(coefficients).partial_coherence_shrinkage()
    equal = Connectivity(coefficients, observation_weights=np.full((2, 8, 2, 4, 1), 0.5))
    np.testing.assert_allclose(equal.partial_coherence_shrinkage(), intensity, rtol=1e-12)

    keep = np.ones((2, 8, 2, 4, 1))
    keep[0, :3] = 0.0  # drop three trials from window 0
    dropped = coefficients.copy()
    dropped[0, :3] = np.nan  # omitted by a zero weight, so never used
    monkeypatch.setattr(
        Connectivity, "_require_uniform_observation_weights", lambda *args, **kwargs: None
    )
    with pytest.warns(UserWarning, match="NaN or Inf"):
        weighted = Connectivity(dropped, observation_weights=keep)
    actual = weighted.partial_coherence_shrinkage()
    np.testing.assert_allclose(
        actual[0],
        Connectivity(coefficients[:1, 3:]).partial_coherence_shrinkage()[0],
        rtol=1e-12,
    )
    np.testing.assert_allclose(actual[1], intensity[1], rtol=1e-12)


def test_auto_shrinkage_warns_for_correlated_observations():
    rng = np.random.default_rng(9)
    connectivity = Connectivity(
        _coefficients(rng, (1, 6, 2, 4, 3)), observations_are_independent=False
    )
    with pytest.warns(UserWarning, match="assumes independent observations"):
        connectivity.partial_coherence(shrinkage="auto")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        connectivity.partial_coherence(shrinkage=0.3)


def test_wrappers_take_shrinkage_in_connectivity_kwargs():
    rng = np.random.default_rng(10)
    time_series = rng.standard_normal((256, 6, 4))
    time_series[:, :, 1] += time_series[:, :, 0]
    result = multitaper_connectivity(
        time_series,
        sampling_frequency=100,
        method="partial_coherence",
        connectivity_kwargs={"shrinkage": "auto"},
    )
    connectivity = Connectivity.from_transform(Multitaper(time_series, sampling_frequency=100))
    np.testing.assert_allclose(
        result.values, connectivity.partial_coherence(shrinkage="auto"), equal_nan=True
    )


def test_auto_intensity_is_computed_once_per_instance():
    """``partial_coherence_shrinkage()`` after ``partial_coherence(shrinkage=
    "auto")`` reuses the intensity, still checks the observations, and returns
    a copy; ``clear_cache`` and every jackknife replicate recompute it."""
    rng = np.random.default_rng(11)
    connectivity = Connectivity(
        _coefficients(rng, (1, 4, 2, 6, 3)), observations_are_independent=False
    )
    with patch.object(
        connectivity, "_coherency_influence", wraps=connectivity._coherency_influence
    ) as influence:
        with pytest.warns(UserWarning, match="assumes independent observations"):
            shrunk = connectivity.partial_coherence(shrinkage="auto")
        with pytest.warns(UserWarning, match="assumes independent observations"):
            intensity = connectivity.partial_coherence_shrinkage()
        assert influence.call_count == 1
        intensity[...] = 0.5  # the caller's copy
        with pytest.warns(UserWarning, match="assumes independent observations"):
            np.testing.assert_array_equal(
                connectivity.partial_coherence(shrinkage="auto"), shrunk
            )
        connectivity.clear_cache()
        with pytest.warns(UserWarning, match="assumes independent observations"):
            connectivity.partial_coherence_shrinkage()
        assert influence.call_count == 2

    independent = Connectivity(_coefficients(rng, (1, 4, 2, 6, 3)))
    with patch.object(
        Connectivity,
        "_coherency_influence",
        autospec=True,
        side_effect=Connectivity._coherency_influence,
    ) as influence:
        independent.jackknife("partial_coherence", shrinkage="auto")
    assert influence.call_count == 1 + independent.n_observations


def test_auto_intensity_cache_is_cleared_by_each_input_setter():
    """Assigning fourier_coefficients, expectation_type, or observation_weights
    gives the intensity of a new instance with those inputs, not the cached
    one (each assignment changes it, so a stale cache would fail)."""
    rng = np.random.default_rng(12)
    first, second = (_coefficients(rng, (2, 5, 3, 8, 4)) for _ in range(2))
    weights = np.ones((2, 5, 3, 8, 1))
    weights[:, :, :, :2] = 0.0  # no observation at the first two frequencies
    connectivity = Connectivity(first)
    previous = connectivity.partial_coherence_shrinkage()
    for name, value, inputs in [
        ("fourier_coefficients", second, {"fourier_coefficients": second}),
        (
            "expectation_type",
            "time_tapers",
            {"fourier_coefficients": second, "expectation_type": "time_tapers"},
        ),
        (
            "observation_weights",
            weights,
            {
                "fourier_coefficients": second,
                "expectation_type": "time_tapers",
                "observation_weights": weights,
            },
        ),
    ]:
        setattr(connectivity, name, value)
        intensity = connectivity.partial_coherence_shrinkage()
        np.testing.assert_array_equal(
            intensity, Connectivity(**inputs).partial_coherence_shrinkage()
        )
        assert intensity.shape != previous.shape or not np.array_equal(
            intensity, previous, equal_nan=True
        )
        previous = intensity
