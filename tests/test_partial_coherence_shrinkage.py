"""Shrinkage of partial coherence toward the diagonal (``shrinkage=``).

A fixed intensity is checked against its definition, and ``"auto"`` against
the same intensity computed one observation at a time. The Monte Carlo tests
draw complex Gaussian eigencoefficients whose expected cross-spectral matrix
is the exact spectrum of a VAR (``tests/_var_oracle.py``), with replicate
datasets on the time-window axis, and compare against the VAR's exact partial
coherence, from the inverse of that spectrum. Limits are five standard errors
of the replicate mean; an allowance that is not sampling error is stated
beside its measured value.
"""

import warnings

import numpy as np
import pytest

from spectral_connectivity import Connectivity, Multitaper, multitaper_connectivity
from tests._backend_helpers import to_host
from tests._var_oracle import _analytic_var, _companion_matrix

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


def _normalized_inverse(matrices):
    """``|P_ij|^2 / (P_ii P_jj)`` for ``P = inv(matrices)``."""
    precision = np.linalg.inv(matrices)
    diagonal = np.real(np.diagonal(precision, axis1=-2, axis2=-1))
    return np.abs(precision) ** 2 / (diagonal[..., :, None] * diagonal[..., None, :])


def _var_replicates(model, n_signals, n_observations, n_replicates, seed=0):
    """``(connectivity, true_spectrum)`` for replicate datasets of a VAR.

    Each replicate (time window) holds ``n_observations`` complex Gaussian
    observations per frequency, drawn with the VAR's exact spectrum ``S(f)``
    at ``_FREQUENCY_BINS`` as their covariance.
    """
    coefficients = _MODELS[model](n_signals)
    assert np.max(np.abs(np.linalg.eigvals(_companion_matrix(coefficients)))) < 1
    _, _, spectrum = _analytic_var(coefficients, np.eye(n_signals), 32)
    spectrum = spectrum[_FREQUENCY_BINS]
    cholesky = np.linalg.cholesky(spectrum)
    rng = np.random.default_rng(seed)
    shape = (n_replicates, n_observations, 1, len(_FREQUENCY_BINS), n_signals)
    white = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2)
    observations = np.einsum("fij,rnkfj->rnkfi", cholesky, white)
    return Connectivity(observations, is_one_sided=True), spectrum


def _mean_and_limit(values):
    """Replicate mean of ``values`` (replicates first) and five standard errors."""
    values = np.asarray(values)
    return values.mean(), 5 * values.std(ddof=1) / np.sqrt(len(values))


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
        (True, "Booleans are not accepted"),
        ("Auto", "Strings are case-sensitive"),
    ],
)
def test_rejects_invalid_shrinkage(shrinkage, hint):
    """The message says what each accepted value means."""
    coefficients = np.ones((1, 2, 2, 2, 2), dtype=complex)
    with pytest.raises(ValueError, match="shrinkage must be None") as error:
        Connectivity(coefficients).partial_coherence(shrinkage=shrinkage)
    message = str(error.value)
    assert message.startswith(
        "shrinkage must be None (no shrinkage), 'auto' (estimate the intensity "
        "for each time window and frequency), or a number from 0 to 1 (a fixed "
        "intensity), got "
    )
    assert hint in message


@pytest.mark.parametrize("intensity", [0.0, 0.3, 1.0])
def test_fixed_shrinkage_matches_its_definition(intensity):
    """``(1 - lambda) S + lambda diag(S)`` before inversion, the same as every
    coherency times ``1 - lambda``; 0 is no shrinkage and 1 gives 0."""
    rng = np.random.default_rng(1)
    connectivity = Connectivity(_coefficients(rng, (2, 6, 2, 5, 4)) * [1e-3, 1, 10, 1e4])
    actual = connectivity.partial_coherence(shrinkage=intensity)

    spectrum = to_host(connectivity._expectation_cross_spectral_matrix())[..., :3, :, :]
    power = np.real(np.diagonal(spectrum, axis1=-2, axis2=-1))
    coherency = spectrum / np.sqrt(power[..., :, None] * power[..., None, :])
    shrunk = (1 - intensity) * coherency + intensity * np.eye(4)
    expected = _normalized_inverse(shrunk + 1e-12 * np.eye(4))
    off_diagonal = ~np.eye(4, dtype=bool)
    np.testing.assert_allclose(
        actual[..., off_diagonal], expected[..., off_diagonal], rtol=1e-10, atol=1e-15
    )
    if intensity == 0:
        np.testing.assert_array_equal(actual, connectivity.partial_coherence())
    if intensity == 1:
        np.testing.assert_array_equal(actual[..., off_diagonal], 0.0)


def _intensity_one_observation_at_a_time(observations):
    """The ``"auto"`` intensity of one bin, ``observations`` ``(n, n_signals)``."""
    n_observations = observations.shape[0]
    spectrum = observations.T @ observations.conj() / n_observations
    power = np.real(np.diag(spectrum))
    scale = np.sqrt(np.outer(power, power))
    coherency = spectrum / scale
    products = observations[:, :, None] * observations.conj()[:, None, :] / scale
    relative_power = np.abs(observations) ** 2 / power
    influence = products - coherency / 2 * (
        relative_power[:, :, None] + relative_power[:, None, :]
    )
    variance = np.sum(np.abs(influence) ** 2, axis=0) / (n_observations * (n_observations - 1))
    off_diagonal = ~np.eye(len(power), dtype=bool)
    return np.clip(
        variance[off_diagonal].sum() / np.sum(np.abs(coherency[off_diagonal]) ** 2), 0, 1
    )


@pytest.mark.parametrize(
    ("expectation_type", "shape"),
    [("trials_tapers", (3, 4, 3, 7, 6)), ("time_tapers", (5, 2, 3, 7, 6))],
)
def test_auto_intensity_matches_the_estimator_one_observation_at_a_time(
    expectation_type, shape
):
    """The intensity, computed from batched means, equals its definition with
    each observation's influence formed explicitly, per kept index and bin."""
    rng = np.random.default_rng(2)
    coefficients = _coefficients(rng, shape)
    connectivity = Connectivity(coefficients, expectation_type=expectation_type)
    intensity = connectivity.partial_coherence_shrinkage()

    n_nonnegative = 4
    kept_axis = 0 if expectation_type == "trials_tapers" else 1
    expected = np.empty((shape[kept_axis], n_nonnegative))
    for kept in range(shape[kept_axis]):
        for frequency in range(n_nonnegative):
            selected = np.take(coefficients[..., frequency, :], kept, axis=kept_axis)
            expected[kept, frequency] = _intensity_one_observation_at_a_time(
                selected.reshape(-1, shape[-1])
            )
    np.testing.assert_allclose(intensity, expected, rtol=1e-12)
    assert np.all((expected > 0.05) & (expected < 0.95))  # not clipped

    # partial_coherence(shrinkage="auto") shrinks each bin by its intensity.
    auto = connectivity.partial_coherence(shrinkage="auto")
    for index, frequency in [(0, 0), (1, 3)]:
        fixed = connectivity.partial_coherence(shrinkage=float(intensity[index, frequency]))
        np.testing.assert_allclose(
            auto[index, frequency], fixed[index, frequency], rtol=1e-12, equal_nan=True
        )


def test_auto_intensity_is_the_same_for_rescaled_and_reordered_signals():
    """The intensity depends only on the coherencies: rescaling the signals
    (with arbitrary phases) or reordering them leaves it unchanged."""
    rng = np.random.default_rng(3)
    coefficients = _coefficients(rng, (2, 5, 2, 6, 5))
    scales = 10.0 ** np.linspace(-6, 6, 5) * np.exp(1j * rng.uniform(-np.pi, np.pi, 5))
    order = rng.permutation(5)
    intensity = Connectivity(coefficients).partial_coherence_shrinkage()
    np.testing.assert_allclose(
        Connectivity(coefficients * scales).partial_coherence_shrinkage(),
        intensity,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        Connectivity(coefficients[..., order]).partial_coherence_shrinkage(),
        intensity,
        rtol=1e-12,
    )
    auto = Connectivity(coefficients).partial_coherence(shrinkage="auto")
    np.testing.assert_allclose(auto, np.swapaxes(auto, -1, -2), rtol=1e-12, equal_nan=True)


@pytest.mark.parametrize(
    ("model", "n_signals", "n_observations"),
    [
        ("chain", 8, 10),
        ("dense", 8, 10),
        ("chain", 8, 20),
        ("dense", 8, 20),
        ("chain", 16, 20),
        ("dense", 16, 20),
        ("chain", 16, 40),
        ("dense", 16, 40),
    ],
)
def test_auto_shrinkage_lowers_the_error_with_few_observations(
    model, n_signals, n_observations
):
    """Against the VAR's exact partial coherence, ``"auto"`` has a lower mean
    squared error than no shrinkage with 1.25 to 2.5 observations per signal:
    the ratio was 0.06-0.08 with 10 observations for 8 signals, 0.28-0.42
    with 20, 0.06-0.09 with 20 for 16 signals and 0.40-0.58 with 40, each
    lower by 7.8 to 13.4 limits."""
    connectivity, spectrum = _var_replicates(model, n_signals, n_observations, 400)
    truth = _normalized_inverse(spectrum)
    unshrunk = _squared_error(connectivity.partial_coherence(), truth)
    shrunk = _squared_error(connectivity.partial_coherence(shrinkage="auto"), truth)
    mean, limit = _mean_and_limit(shrunk - unshrunk)
    assert mean + limit < 0


def _true_coherency(spectrum):
    power = np.real(np.diagonal(spectrum, axis1=-2, axis2=-1))
    return spectrum / np.sqrt(power[..., :, None] * power[..., None, :])


@pytest.mark.parametrize(
    ("model", "n_signals", "n_observations"),
    [
        ("chain", 8, 10),
        ("dense", 8, 10),
        ("chain", 8, 40),
        ("dense", 8, 40),
        ("chain", 16, 20),
        ("dense", 16, 20),
    ],
)
def test_auto_intensity_is_near_the_monte_carlo_optimum(model, n_signals, n_observations):
    """The intensity is near the one that minimizes the replicates' squared
    error of the shrunk coherencies, ``1 - sum Re(conj(rho) r) / sum |r|^2``
    over pairs and replicates, at each frequency.

    The estimator is first order in ``1 / n``. The allowance of 15% of the
    optimum covers what that leaves out: the mean intensity was 0.1-10% below
    or above it with 10 observations, 0.3-4.4% with 40 (8 signals), and 0.6-8%
    with 20 for 16 signals. Without the delta-method terms for the estimated
    powers (Schäfer & Strimmer's variance), it was 1.1-2.2 times the optimum
    in these cases, and fails this test for five of the six.
    """
    connectivity, spectrum = _var_replicates(model, n_signals, n_observations, 400, seed=1)
    rho = _true_coherency(spectrum)
    coherency = connectivity.coherency()
    off_diagonal = ~np.eye(n_signals, dtype=bool)
    squared = np.sum(np.abs(coherency[..., off_diagonal]) ** 2, axis=-1).mean(axis=0)
    aligned = np.sum(
        np.real(np.conj(rho[..., off_diagonal]) * coherency[..., off_diagonal]), axis=-1
    ).mean(axis=0)
    optimum = 1 - aligned / squared

    intensity = connectivity.partial_coherence_shrinkage()
    mean = intensity.mean(axis=0)
    limit = 5 * intensity.std(axis=0, ddof=1) / np.sqrt(intensity.shape[0])
    assert np.all(np.abs(mean - optimum) <= 0.15 * optimum + limit)


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
    squared_rho = np.abs(_true_coherency(spectrum)[..., off_diagonal]) ** 2
    variance = (1 - squared_rho) * (1 - squared_rho / 2) / n_observations
    first_order = variance.sum(axis=-1) / (squared_rho + variance).sum(axis=-1)

    intensity = connectivity.partial_coherence_shrinkage()
    mean = intensity.mean(axis=0)
    limit = 5 * intensity.std(axis=0, ddof=1) / np.sqrt(intensity.shape[0])
    assert np.all(np.abs(mean - first_order) <= 0.03 * first_order + limit)
    assert np.all(mean < 0.05)
    assert changes[1] < 0.15 * changes[0]


@pytest.mark.parametrize(
    ("model", "n_signals", "n_observations"),
    [
        ("chain", 8, 7),
        ("dense", 8, 7),
        ("chain", 8, 4),
        ("dense", 8, 4),
        ("chain", 16, 15),
        ("dense", 16, 15),
        ("chain", 16, 8),
        ("dense", 16, 8),
    ],
)
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
        match=r"single observation .* set by the shrinkage intensity \(0\.5\) and the "
        "number of signals regardless of the data",
    ):
        shrunk = connectivity.partial_coherence(shrinkage=0.5)
    other = Connectivity(_coefficients(rng, (1, 1, 1, 4, 3)))
    with pytest.warns(UserWarning, match="single observation"):
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
        match=r'^partial_coherence\(shrinkage="auto"\) does not support non-uniform .* '
        "'keep', an expectation_type that keeps the time axis, or a fixed shrinkage between 0 "
        r"and 1\.$",
    ):
        connectivity.partial_coherence(shrinkage="auto")
    with pytest.raises(
        ValueError,
        match=r"^partial_coherence_shrinkage\(\) does not support non-uniform .* "
        r"'keep', or an expectation_type that keeps the time axis\.$",
    ):
        connectivity.partial_coherence_shrinkage()
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
