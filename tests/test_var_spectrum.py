"""VARSpectrum: the exact cross-spectrum of a VAR model, as a transform.

The model's spectrum and its directed measures are known independently of the
package: the closed forms and the state-space (Riccati) Granger oracle in
``tests/_var_oracle.py`` factor nothing. Every comparison is exact up to the
Wilson factorization's tolerance, so no Monte Carlo is needed, except for one
check that a long multitaper estimate of the simulated model is a draw around
VARSpectrum's spectrum, which ties its units and its lag convention to
``simulate_MVAR``.
"""

import json
import warnings

import numpy as np
import pytest

from spectral_connectivity import (
    Connectivity,
    DirectedOrientationWarning,
    Multitaper,
    SpectralTransform,
    VARSpectrum,
)
from spectral_connectivity._measure_registry import _MEASURE_SPECS
from spectral_connectivity.minimum_phase_decomposition import _is_conjugate_symmetric
from spectral_connectivity.simulate import simulate_MVAR
from spectral_connectivity.wrapper import connectivity_to_xarray
from tests._backend_helpers import to_host
from tests._expected_spectrum import _assert_multitaper_estimates
from tests._var_oracle import (
    _analytic_directed_measures,
    _analytic_var,
    _block_granger_from_state_space,
    _dense_stable_var,
    _pairwise_granger_from_state_space,
    _state_space_conditional_granger,
    _time_reversed_var,
)

_SAMPLING_FREQUENCY = 250.0
# Every signal drives every other, with correlated innovations.
_DENSE = _dense_stable_var(n_signals=3, n_lags=2, spectral_radius=0.8, seed=0)
# A chain 0 -> 1 -> 2 with unequal innovation variances, which separate the
# noise-weighted transfer-function measures from the others.
_CHAIN = (
    np.stack(
        [np.array([[0.5, 0.0, 0.0], [0.4, 0.5, 0.0], [0.0, 0.4, 0.5]]), -0.6 * np.eye(3)]
    ),
    np.diag([1.0, 2.0, 0.5]),
)
_MODELS = pytest.mark.parametrize(
    ("coefficients", "noise_covariance"), [_DENSE, _CHAIN], ids=["dense", "chain"]
)
_OFF_DIAGONAL = ~np.eye(3, dtype=bool)


def _model_spectrum(coefficients, noise_covariance, n_fft):
    """The model's two-sided cross-spectral density in signal² / Hz."""
    return _analytic_var(coefficients, noise_covariance, n_fft)[2] / _SAMPLING_FREQUENCY


def _tightly_factored(var):
    """A Connectivity of ``var`` factored far below the tolerances compared."""
    return Connectivity.from_transform(
        var, minimum_phase_tolerance=1e-12, minimum_phase_max_iterations=5000
    )


def _directed(connectivity, measure, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warnings.filterwarnings("ignore", category=DirectedOrientationWarning)
        return getattr(connectivity, measure)(**kwargs)


@_MODELS
@pytest.mark.parametrize("n_fft_samples", [None, 501], ids=["default", "odd"])
def test_coefficients_reproduce_the_model_spectrum(
    coefficients, noise_covariance, n_fft_samples
):
    """The taper mean of the outer products is S(f) / sampling_frequency on
    every bin, exactly conjugate-symmetric (so the Wilson factorization takes
    its half-spectrum path) and real at 0 Hz and Nyquist."""
    var = VARSpectrum(
        coefficients,
        noise_covariance,
        sampling_frequency=_SAMPLING_FREQUENCY,
        n_fft_samples=n_fft_samples,
    )
    assert isinstance(var, SpectralTransform)
    coefficients_fft = var.fft()
    n_fft = var.n_fft_samples
    assert coefficients_fft.shape == (1, 1, 3, n_fft, 3)
    connectivity = Connectivity.from_transform(var)
    cross_spectrum = connectivity._expectation_cross_spectral_matrix()
    expected = _model_spectrum(coefficients, noise_covariance, n_fft)
    np.testing.assert_allclose(
        to_host(cross_spectrum)[0], expected, rtol=0, atol=1e-12 * np.abs(expected).max()
    )
    assert _is_conjugate_symmetric(cross_spectrum)
    real_bins = [0.0, _SAMPLING_FREQUENCY / 2] if n_fft % 2 == 0 else [0.0]
    np.testing.assert_allclose(
        connectivity.frequencies[connectivity.real_valued_frequencies], real_bins
    )
    np.testing.assert_allclose(
        to_host(var.frequencies), np.fft.fftfreq(n_fft, 1 / _SAMPLING_FREQUENCY)
    )


def test_power_is_the_models_one_sided_density():
    """power() doubles every bin between 0 Hz and Nyquist of S_ii / fs."""
    coefficients, noise_covariance = _DENSE
    var = VARSpectrum(coefficients, noise_covariance, sampling_frequency=_SAMPLING_FREQUENCY)
    power = Connectivity.from_transform(var).power(one_sided=True)[0]
    n_fft = var.n_fft_samples
    density = np.real(
        np.diagonal(_model_spectrum(coefficients, noise_covariance, n_fft), 0, -2, -1)
    )
    expected = density[: n_fft // 2 + 1].copy()
    expected[1 : (n_fft + 1) // 2] *= 2
    np.testing.assert_allclose(power, expected, rtol=1e-12)


def test_long_multitaper_estimate_of_the_simulated_model_is_a_draw_around_it():
    """A multitaper estimate of 1000 trials of simulate_MVAR output differs
    from VARSpectrum's spectrum by no more than sampling error (six standard
    errors per bin, five on average over frequency; see
    ``_assert_multitaper_estimates``). The model's poles have radius at most
    0.7, so a 256-sample window resolves its spectrum and the estimate's
    smoothing bias is small beside that error. Without detrending, 0 Hz is
    estimated too. Twice the spectrum (a one- vs two-sided scale), or the
    model with its coefficient matrices transposed, lies far outside it."""
    coefficients, noise_covariance = _dense_stable_var(
        n_signals=3, n_lags=2, spectral_radius=0.7, seed=3
    )
    time_series = simulate_MVAR(
        coefficients,
        noise_covariance,
        n_time_samples=256,
        n_trials=1000,
        n_burnin_samples=500,
        random_state=0,
    )
    multitaper = Multitaper(
        time_series,
        sampling_frequency=_SAMPLING_FREQUENCY,
        time_halfbandwidth_product=3,
        detrend_type=None,
    )
    estimate = to_host(
        Connectivity.from_transform(multitaper)._expectation_cross_spectral_matrix()
    )[0]

    def spectrum(var_coefficients):
        var = VARSpectrum(
            var_coefficients,
            noise_covariance,
            sampling_frequency=_SAMPLING_FREQUENCY,
            n_fft_samples=256,
        )
        return to_host(Connectivity.from_transform(var)._expectation_cross_spectral_matrix())[
            0
        ]

    _assert_multitaper_estimates(multitaper, estimate, spectrum(coefficients), "VARSpectrum")
    for wrong in (2 * spectrum(coefficients), spectrum(coefficients.swapaxes(-1, -2))):
        with pytest.raises(AssertionError, match="at frequency bin"):
            _assert_multitaper_estimates(multitaper, estimate, wrong, "wrong")


@_MODELS
def test_pairwise_conditional_and_time_reversed_granger_match_the_state_space_oracle(
    coefficients, noise_covariance
):
    """On the default grid the directed measures are those of the model to the
    Wilson tolerance: the state-space oracle factors nothing."""
    var = VARSpectrum(coefficients, noise_covariance, sampling_frequency=_SAMPLING_FREQUENCY)
    connectivity = _tightly_factored(var)
    n_fft = var.n_fft_samples

    pairwise = _directed(connectivity, "pairwise_spectral_granger_prediction")[0]
    oracle = _pairwise_granger_from_state_space(coefficients, noise_covariance, n_fft)
    assert oracle[:, _OFF_DIAGONAL].max() > 0.1
    np.testing.assert_allclose(
        pairwise[:, _OFF_DIAGONAL], oracle[:, _OFF_DIAGONAL], rtol=0, atol=1e-9
    )

    reversed_var = _time_reversed_var(coefficients, noise_covariance)
    time_reversed = _directed(connectivity, "time_reversed_spectral_granger_prediction")[0]
    np.testing.assert_allclose(
        time_reversed[:, _OFF_DIAGONAL],
        _pairwise_granger_from_state_space(*reversed_var, n_fft)[:, _OFF_DIAGONAL],
        rtol=0,
        atol=1e-9,
    )

    conditional = _directed(connectivity, "conditional_spectral_granger_prediction")[0]
    reversed_conditional = _directed(
        connectivity, "time_reversed_conditional_spectral_granger_prediction"
    )[0]
    for source, target in zip(*np.nonzero(_OFF_DIAGONAL), strict=True):
        for result, model in (
            (conditional, (coefficients, noise_covariance)),
            (reversed_conditional, reversed_var),
        ):
            np.testing.assert_allclose(
                result[:, source, target],
                _state_space_conditional_granger(*model, n_fft, target, source),
                rtol=0,
                atol=1e-9,
                err_msg=f"{source} -> {target}",
            )


def test_blockwise_granger_matches_the_state_space_oracle():
    """Between groups of a 5-signal model whose groups differ in size, forward
    and time-reversed (the state-space Granger of the VAR read backwards)."""
    coefficients, noise_covariance = _dense_stable_var(
        n_signals=5, n_lags=2, spectral_radius=0.8, seed=0
    )
    var = VARSpectrum(coefficients, noise_covariance, sampling_frequency=_SAMPLING_FREQUENCY)
    group_labels = ["a", "b", "b", "c", "c"]
    connectivity = _tightly_factored(var)
    for measure, model in (
        ("blockwise_spectral_granger_prediction", (coefficients, noise_covariance)),
        (
            "time_reversed_blockwise_spectral_granger_prediction",
            _time_reversed_var(coefficients, noise_covariance),
        ),
    ):
        blockwise, labels = _directed(connectivity, measure, group_labels=group_labels)
        blocks = [
            [i for i, label in enumerate(group_labels) if label == group] for group in labels
        ]
        oracle = _block_granger_from_state_space(*model, var.n_fft_samples, blocks)
        assert oracle[:, _OFF_DIAGONAL].max(axis=0).min() > 0.01
        np.testing.assert_allclose(
            blockwise[0][:, _OFF_DIAGONAL],
            oracle[:, _OFF_DIAGONAL],
            rtol=0,
            atol=1e-9,
            err_msg=measure,
        )


@pytest.mark.parametrize(
    "measure",
    [
        "directed_transfer_function",
        "partial_directed_coherence",
        "directed_coherence",
        "generalized_partial_directed_coherence",
        "direct_directed_transfer_function",
    ],
)
def test_transfer_function_measures_match_their_closed_forms(measure):
    """Every entry, the diagonal included, transposed from the closed forms'
    native [target, source] layout."""
    coefficients, noise_covariance = _CHAIN
    var = VARSpectrum(coefficients, noise_covariance, sampling_frequency=_SAMPLING_FREQUENCY)
    A, H, _ = _analytic_var(coefficients, noise_covariance, var.n_fft_samples)
    expected = np.swapaxes(
        _analytic_directed_measures(A, H, noise_covariance)[measure], -1, -2
    )
    result = np.asarray(_directed(_tightly_factored(var), measure))[0]
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-9)


@_MODELS
def test_measures_of_the_spectral_matrix_follow_its_formulas(coefficients, noise_covariance):
    """Coherency, partial coherence, canonical and global coherence of the
    exact S(f), from their definitions."""
    var = VARSpectrum(coefficients, noise_covariance, sampling_frequency=_SAMPLING_FREQUENCY)
    connectivity = Connectivity.from_transform(var)
    n_fft = var.n_fft_samples
    spectrum = _model_spectrum(coefficients, noise_covariance, n_fft)[: n_fft // 2 + 1]
    power = np.real(np.diagonal(spectrum, 0, -2, -1))
    coherency = spectrum / np.sqrt(power[..., :, None] * power[..., None, :])
    tolerance = {"rtol": 0, "atol": 1e-12}

    def off_diagonal(values):
        return np.asarray(values)[0][:, _OFF_DIAGONAL]

    np.testing.assert_allclose(
        off_diagonal(connectivity.coherency()), coherency[:, _OFF_DIAGONAL], **tolerance
    )
    np.testing.assert_allclose(
        off_diagonal(connectivity.magnitude_squared_coherence()),
        np.abs(coherency[:, _OFF_DIAGONAL]) ** 2,
        **tolerance,
    )
    np.testing.assert_allclose(
        off_diagonal(connectivity.imaginary_coherence()),
        np.abs(coherency[:, _OFF_DIAGONAL].imag),
        **tolerance,
    )
    precision = np.linalg.inv(spectrum)
    precision_power = np.real(np.diagonal(precision, 0, -2, -1))
    partial = np.abs(precision) ** 2 / (
        precision_power[..., :, None] * precision_power[..., None, :]
    )
    np.testing.assert_allclose(
        off_diagonal(connectivity.partial_coherence()), partial[:, _OFF_DIAGONAL], atol=1e-10
    )
    # Canonical coherence of groups {0, 1} and {2}: the largest squared
    # singular value of S_11^(-1/2) S_12 S_22^(-1/2).
    canonical, _ = connectivity.canonical_coherence(["a", "a", "b"])
    within = spectrum[:, :2, :2]
    eigenvalues, eigenvectors = np.linalg.eigh(within)
    inverse_root = eigenvectors @ (
        eigenvectors.conj().swapaxes(-1, -2) / np.sqrt(eigenvalues)[..., None]
    )
    whitened = inverse_root @ spectrum[:, :2, 2:] / np.sqrt(power[:, 2, None, None])
    expected_canonical = np.linalg.svd(whitened, compute_uv=False)[:, 0] ** 2
    np.testing.assert_allclose(canonical[0, :, 0, 1], expected_canonical, atol=1e-10)
    # Global coherence: the largest eigenvalue of S over its trace.
    fractions, _ = connectivity.global_coherence()
    eigenvalues = np.linalg.eigvalsh(spectrum)
    np.testing.assert_allclose(
        fractions[0, :, 0], eigenvalues[:, -1] / eigenvalues.sum(axis=-1), atol=1e-12
    )


def test_a_time_varying_model_stacks_the_models_of_its_windows():
    """Each window is the model of its own, with a shared or per-window
    covariance, on the time axis; times label it when given."""
    chain_coefficients, chain_noise = _CHAIN
    dense_coefficients, dense_noise = _DENSE
    coefficients = np.stack([chain_coefficients, dense_coefficients])
    noise = np.stack([chain_noise, dense_noise])
    stacked = VARSpectrum(
        coefficients, noise, sampling_frequency=_SAMPLING_FREQUENCY, time=[0.5, 1.5]
    )
    n_fft = stacked.n_fft_samples
    assert stacked.n_time_windows == 2
    connectivity = _tightly_factored(stacked)
    np.testing.assert_array_equal(connectivity.time, [0.5, 1.5])
    for window, (window_coefficients, window_noise) in enumerate([_CHAIN, _DENSE]):
        single = _tightly_factored(
            VARSpectrum(
                window_coefficients,
                window_noise,
                sampling_frequency=_SAMPLING_FREQUENCY,
                n_fft_samples=n_fft,
            )
        )
        np.testing.assert_array_equal(stacked.fft()[window], single._fourier_coefficients[0])
        np.testing.assert_allclose(
            _directed(connectivity, "pairwise_spectral_granger_prediction")[window],
            _directed(single, "pairwise_spectral_granger_prediction")[0],
            rtol=0,
            atol=1e-10,
        )
    shared = VARSpectrum(coefficients, chain_noise, sampling_frequency=_SAMPLING_FREQUENCY)
    np.testing.assert_array_equal(shared.noise_covariance, np.stack([chain_noise] * 2))
    assert shared.time is None
    np.testing.assert_array_equal(Connectivity.from_transform(shared).time, [0, 1])
    # A single model with per-window covariances is a model per window too.
    per_noise = VARSpectrum(chain_coefficients, noise, sampling_frequency=_SAMPLING_FREQUENCY)
    assert per_noise.coefficients.shape == (2, 2, 3, 3)


def test_the_minimum_grid_follows_the_decay_of_the_model():
    """Twice the first lag at which the companion matrix's power has norm
    1e-10, for the slowest window. Repeated eigenvalues (the chain's signals
    share their dynamics) decay more slowly than the spectral radius alone
    says; a nilpotent companion matrix reaches 0 at the length of its
    impulse response."""
    for coefficients, noise_covariance in (_DENSE, _CHAIN):
        var = VARSpectrum(
            coefficients, noise_covariance, sampling_frequency=_SAMPLING_FREQUENCY
        )
        companion = np.zeros((6, 6))
        companion[:3] = np.concatenate(list(coefficients), axis=1)
        companion[3:, :3] = np.eye(3)
        lag = var.minimum_n_fft_samples // 2
        assert np.linalg.norm(np.linalg.matrix_power(companion, lag)) <= 1e-10
        assert np.linalg.norm(np.linalg.matrix_power(companion, lag - 1)) > 1e-10
        rho = var.spectral_radius[0]
        np.testing.assert_allclose(rho, np.abs(np.linalg.eigvals(companion)).max())
        assert rho**lag <= 1e-10
        assert var.n_fft_samples >= var.minimum_n_fft_samples
    # The chain's eigenvalues are repeated: the radius alone would allow a
    # grid a quarter shorter.
    assert lag > 1.2 * np.log(1e-10) / np.log(rho)

    nilpotent = np.zeros((1, 3, 3))
    nilpotent[0, 1, 0] = nilpotent[0, 2, 1] = 0.8  # 0 -> 1 -> 2, no memory
    chain = VARSpectrum(nilpotent, sampling_frequency=_SAMPLING_FREQUENCY)
    assert chain.spectral_radius[0] == 0
    assert chain.minimum_n_fft_samples == 6
    granger = _directed(_tightly_factored(chain), "pairwise_spectral_granger_prediction")[0]
    oracle = _pairwise_granger_from_state_space(nilpotent, np.eye(3), chain.n_fft_samples)
    np.testing.assert_allclose(
        granger[:, _OFF_DIAGONAL], oracle[:, _OFF_DIAGONAL], rtol=0, atol=1e-9
    )
    white = VARSpectrum(np.zeros((0, 2, 2)), sampling_frequency=_SAMPLING_FREQUENCY)
    assert white.minimum_n_fft_samples == 2
    np.testing.assert_allclose(
        Connectivity.from_transform(white).power(one_sided=True)[0],
        [[1 / _SAMPLING_FREQUENCY] * 2] * white.n_fft_samples,
    )


@pytest.mark.parametrize(
    ("arguments", "error", "match"),
    [
        ({"coefficients": np.zeros((2, 2), dtype=complex)}, TypeError, "must be real"),
        ({"coefficients": np.zeros((1, 2, 3))}, ValueError, "one square matrix"),
        ({"coefficients": np.zeros((2, 2))}, ValueError, r"coefficients\[np\.newaxis\]"),
        ({"coefficients": np.full((1, 2, 2), np.nan)}, ValueError, "NaN or Inf"),
        ({"noise_covariance": np.eye(3)}, ValueError, "n_signals = 2"),
        ({"noise_covariance": [[1.0, 0.5], [0.4, 1.0]]}, ValueError, "must be symmetric"),
        (
            {"noise_covariance": [[1.0, 1.0], [1.0, 1.0]]},
            ValueError,
            "positive definite, but its smallest eigenvalue is",
        ),
        (
            {
                "coefficients": np.zeros((3, 1, 2, 2)),
                "noise_covariance": np.stack([np.eye(2)] * 2),
            },
            ValueError,
            "coefficients give 3 time windows and noise_covariance 2",
        ),
        (
            {"coefficients": [[[1.0, 0.0], [0.0, 0.5]]]},
            ValueError,
            "radius of their companion",
        ),
        ({"coefficients": [[[1.2, 0.0], [0.0, 0.5]]]}, ValueError, "is 1.2, not below 1"),
        ({"n_fft_samples": 2.5}, ValueError, "an integer of at least 2"),
        (
            {"coefficients": np.zeros((4, 2, 2)), "n_fft_samples": 4},
            ValueError,
            r"at least 5 \(2, and n_lags \+ 1 = 5",
        ),
        ({"time": [0.0, 1.0]}, ValueError, r"one time in seconds per window \(1\)"),
    ],
)
def test_invalid_models_are_refused(arguments, error, match):
    kwargs = {
        "coefficients": [[[0.5, 0.0], [0.2, 0.3]]],
        "sampling_frequency": _SAMPLING_FREQUENCY,
        **arguments,
    }
    with pytest.raises(error, match=match):
        VARSpectrum(**kwargs)


def test_messages_name_the_radius_and_the_grid():
    """Unstable: the radius, and the scaling that makes it 0.9. A grid below
    the minimum warns that only the directed measures lose exactness."""
    coefficients, noise_covariance = _DENSE
    with pytest.raises(
        ValueError, match=r"is 1\.6, not below 1.*c = 0\.5625 gives a radius of 0\.9"
    ):
        VARSpectrum(
            coefficients * 2.0 ** np.arange(1, 3)[:, None, None],
            noise_covariance,
            sampling_frequency=_SAMPLING_FREQUENCY,
        )
    var = VARSpectrum(coefficients, noise_covariance, sampling_frequency=_SAMPLING_FREQUENCY)
    minimum = var.minimum_n_fft_samples
    assert f"spectral_radius={var.spectral_radius[0]:.6g}" in repr(var)
    match = rf"below this model's minimum_n_fft_samples \({minimum}\)"
    with pytest.warns(UserWarning, match=match) as caught:
        short = VARSpectrum(
            coefficients,
            noise_covariance,
            sampling_frequency=_SAMPLING_FREQUENCY,
            n_fft_samples=minimum - 2,
        )
    message = str(caught[0].message)
    assert "the coherence family stay exact" in message
    assert f"use n_fft_samples >= {minimum}" in message
    # Power stays exact on the short grid.
    power = Connectivity.from_transform(short).power(one_sided=True)[0]
    density = _model_spectrum(coefficients, noise_covariance, minimum - 2)
    expected = np.real(np.diagonal(density, 0, -2, -1))[: (minimum - 2) // 2 + 1].copy()
    expected[1:-1] *= 2
    np.testing.assert_allclose(power, expected, rtol=1e-12)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        exact = VARSpectrum(
            coefficients,
            noise_covariance,
            sampling_frequency=_SAMPLING_FREQUENCY,
            n_fft_samples=minimum,
        )
    assert exact.n_fft_samples == minimum


def test_a_huge_default_grid_warns_of_its_size():
    """A pole this close to the unit circle needs a grid of millions of bins."""
    match = r"n_fft_samples=\d+ frequency bins.*0\.99999"
    with pytest.warns(UserWarning, match=match) as caught:
        var = VARSpectrum(np.array([[[0.99999]]]), sampling_frequency=_SAMPLING_FREQUENCY)
    assert var.n_fft_samples > 2**18
    assert "exact on any grid" in str(caught[0].message)


def test_the_sampling_frequency_is_required_by_its_name():
    with pytest.raises(TypeError, match="sampling_frequency"):
        VARSpectrum(np.zeros((1, 2, 2)))  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="sampling_frequency"):
        VARSpectrum(np.zeros((1, 2, 2)), fs=100)  # type: ignore[call-arg]


def test_the_model_is_a_read_only_snapshot():
    """Changing the caller's arrays afterwards changes nothing."""
    coefficients, noise_covariance = (array.copy() for array in _DENSE)
    var = VARSpectrum(coefficients, noise_covariance, sampling_frequency=_SAMPLING_FREQUENCY)
    before = var.fft()
    coefficients[:] = 0
    noise_covariance[:] = np.eye(3)
    np.testing.assert_array_equal(var.fft(), before)
    with pytest.raises(ValueError, match="read-only"):
        var.coefficients[0, 0, 0, 0] = 1.0
    assert var.fft() is not var.fft()


def test_labeled_results_record_the_model():
    """connectivity_to_xarray labels the signals and records the model, the
    grid, and that the windows are labeled by index."""
    coefficients, noise_covariance = _DENSE
    var = VARSpectrum(coefficients, noise_covariance, sampling_frequency=_SAMPLING_FREQUENCY)
    result = connectivity_to_xarray(
        var, "magnitude_squared_coherence", signal_names=["a", "b", "c"]
    )
    assert result.dims == ("time", "frequency", "source", "target")
    assert result.source.values.tolist() == ["a", "b", "c"]
    assert result.attrs["transform"] == "var_spectrum"
    assert result.attrs["sampling_frequency"] == _SAMPLING_FREQUENCY
    assert result.attrs["nyquist_frequency"] == _SAMPLING_FREQUENCY / 2
    assert result.attrs["n_signals"] == 3
    assert result.attrs["time_coordinate"] == "index"
    # Pseudo-observations are not samples: no count or independence recorded.
    assert result.attrs["is_exact_spectrum"] == 1
    assert "n_observations" not in result.attrs
    assert "observations_are_independent" not in result.attrs
    assert result.time.values.tolist() == [0]
    assert "units" not in result.time.attrs
    assert result.frequency.attrs["units"] == "Hz"
    estimator = json.loads(result.attrs["transform_parameters_json"])["estimator"]
    np.testing.assert_array_equal(estimator["coefficients"], var.coefficients)
    np.testing.assert_array_equal(estimator["noise_covariance"], var.noise_covariance)
    assert estimator["n_fft_samples"] == var.n_fft_samples
    assert estimator["n_lags"] == 2
    assert estimator["n_time_windows"] == 1
    # A larger model is recorded by shape and dtype, as large input attributes are.
    large, large_noise = _dense_stable_var(n_signals=6, n_lags=3, spectral_radius=0.5, seed=1)
    summarized = json.loads(
        connectivity_to_xarray(
            VARSpectrum(large, large_noise, sampling_frequency=_SAMPLING_FREQUENCY), "power"
        ).attrs["transform_parameters_json"]
    )["estimator"]
    assert summarized["coefficients"] == {
        "summarized_array": {"shape": [1, 3, 6, 6], "dtype": "float64"}
    }
    np.testing.assert_array_equal(summarized["noise_covariance"], [large_noise])

    timed = VARSpectrum(
        coefficients, noise_covariance, sampling_frequency=_SAMPLING_FREQUENCY, time=[2.0]
    )
    power = connectivity_to_xarray(timed, "power")
    assert power.attrs["time_coordinate"] == "provided"
    assert power.time.values.tolist() == [2.0]
    assert power.time.attrs["units"] == "s"
    assert power.attrs["long_name"] == _MEASURE_SPECS["power"].long_name


def test_refused_measures_are_refused_through_the_wrapper():
    coefficients, noise_covariance = _DENSE
    var = VARSpectrum(coefficients, noise_covariance, sampling_frequency=_SAMPLING_FREQUENCY)
    with pytest.raises(ValueError, match=r"^phase_locking_value is not defined for an exact"):
        connectivity_to_xarray(var, "phase_locking_value")


def test_the_docstring_names_every_refused_measure():
    """The class documentation lists what the registry refuses."""
    refused = [
        name for name, spec in _MEASURE_SPECS.items() if not spec.defined_for_exact_spectrum
    ]
    notes = " ".join(VARSpectrum.__doc__.split())
    assert all(name in notes for name in refused)


def test_from_var_is_from_transform_of_the_var_spectrum():
    """The shortcut passes every setting through and builds the same instance."""
    coefficients, noise_covariance = _DENSE
    model = {
        "sampling_frequency": _SAMPLING_FREQUENCY,
        "n_fft_samples": 300,
        "time": [4.0],
    }
    settings = {
        "expectation_type": "tapers",
        "minimum_phase_tolerance": 1e-11,
        "minimum_phase_max_iterations": 900,
    }
    shortcut = Connectivity.from_var(coefficients, noise_covariance, **model, **settings)
    explicit = Connectivity.from_transform(
        VARSpectrum(coefficients, noise_covariance, **model), **settings
    )
    assert shortcut.is_exact_spectrum
    assert shortcut.expectation_type == "tapers"
    np.testing.assert_array_equal(shortcut.time, [4.0])
    np.testing.assert_array_equal(shortcut.frequencies, explicit.frequencies)
    np.testing.assert_array_equal(
        shortcut.power(one_sided=True), explicit.power(one_sided=True)
    )
    for measure in ("coherency", "pairwise_spectral_granger_prediction"):
        np.testing.assert_array_equal(
            _directed(shortcut, measure), _directed(explicit, measure), err_msg=measure
        )


def test_from_var_refuses_what_varspectrum_and_the_flag_refuse():
    coefficients, noise_covariance = _DENSE
    with pytest.raises(ValueError, match="not stable"):
        Connectivity.from_var(3 * coefficients, sampling_frequency=_SAMPLING_FREQUENCY)
    with pytest.raises(ValueError, match="keeps the tapers"):
        Connectivity.from_var(
            coefficients, sampling_frequency=_SAMPLING_FREQUENCY, expectation_type="trials"
        )
    with pytest.raises(TypeError, match="positional arguments"):
        Connectivity.from_var(coefficients, noise_covariance, _SAMPLING_FREQUENCY)  # type: ignore[misc]


def _shared_dynamics_chain(rho):
    """A chain 0 -> 1 -> 2 of couplings 0.2 whose signals all have a pole at
    ``rho``: their powers grow by orders of magnitude along the chain."""
    coefficients = rho * np.eye(3)[np.newaxis]
    coefficients[0, 1, 0] = coefficients[0, 2, 1] = 0.2
    return coefficients, np.eye(3)


@pytest.mark.parametrize(
    "rho",
    [
        0.9,
        pytest.param(
            0.99,
            marks=pytest.mark.xfail(
                strict=True,
                reason="Wilson factorization of badly scaled spectra; fixed by "
                "fix/wilson-scaling",
            ),
        ),
    ],
)
def test_pairwise_granger_of_a_chain_with_shared_slow_dynamics(rho):
    """At rho 0.99 the signals' standard deviations are about 7, 100 and 1800,
    and the Wilson factorization of the (0, 2) pair converges to a factor
    that reproduces its spectrum but is not minimum phase: 2 -> 0, which is
    exactly 0 (signal 0 is driven by nothing), came out 3.9."""
    coefficients, noise_covariance = _shared_dynamics_chain(rho)
    var = VARSpectrum(coefficients, noise_covariance, sampling_frequency=_SAMPLING_FREQUENCY)
    granger = _directed(
        Connectivity.from_transform(var), "pairwise_spectral_granger_prediction"
    )[0]
    oracle = _pairwise_granger_from_state_space(
        coefficients, noise_covariance, var.n_fft_samples
    )
    np.testing.assert_allclose(
        granger[:, _OFF_DIAGONAL], oracle[:, _OFF_DIAGONAL], rtol=0, atol=1e-8
    )
