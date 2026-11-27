"""Published call signatures: released positional orders and keyword-only settings.

The released positional orders below are copied from the v2.0.1 sources; a
change to any of them would silently rebind existing positional calls.
"""

import inspect
import warnings

import numpy as np
import pytest

from spectral_connectivity import (
    Connectivity,
    MorletWavelet,
    Multitaper,
    ShortTimeFourierTransform,
    SidednessWarning,
    Welch,
    fourier_connectivity,
)
from spectral_connectivity.minimum_phase_decomposition import minimum_phase_decomposition
from spectral_connectivity.simulate import (
    simulate_lagged_broadband,
    simulate_shared_oscillation,
)
from spectral_connectivity.statistics import power_confidence_intervals
from spectral_connectivity.wrapper import connectivity_to_xarray, multitaper_connectivity
from tests._backend_helpers import to_host

_POSITIONAL = (
    inspect.Parameter.POSITIONAL_ONLY,
    inspect.Parameter.POSITIONAL_OR_KEYWORD,
)


def _positional_names(function):
    return [
        name
        for name, parameter in inspect.signature(function).parameters.items()
        if parameter.kind in _POSITIONAL and name != "self"
    ]


def _keyword_only_names(function):
    return [
        name
        for name, parameter in inspect.signature(function).parameters.items()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY
    ]


def test_multitaper_keeps_its_released_positional_order():
    assert _positional_names(Multitaper)[:13] == [
        "time_series",
        "sampling_frequency",
        "time_halfbandwidth_product",
        "detrend_type",
        "time_window_duration",
        "time_window_step",
        "n_tapers",
        "tapers",
        "start_time",
        "n_fft_samples",
        "n_time_samples_per_window",
        "n_time_samples_per_step",
        "is_low_bias",
    ]


def test_multitaper_positional_call_binds_the_released_parameters():
    time_series = np.random.default_rng(0).standard_normal((400, 2, 2))
    positional = Multitaper(time_series, 200, 2, "linear", 1.0, 0.5, 2)
    keyword = Multitaper(
        time_series,
        sampling_frequency=200,
        time_halfbandwidth_product=2,
        detrend_type="linear",
        time_window_duration=1.0,
        time_window_step=0.5,
        n_tapers=2,
    )
    assert positional.time_halfbandwidth_product == 2
    assert positional.detrend_type == "linear"
    assert positional.n_tapers == 2
    np.testing.assert_array_equal(to_host(positional.fft()), to_host(keyword.fft()))


@pytest.mark.parametrize(
    "transform", [ShortTimeFourierTransform, Welch, MorletWavelet], ids=lambda t: t.__name__
)
def test_new_transforms_take_only_data_and_rate_positionally(transform):
    assert _positional_names(transform) == ["time_series", "sampling_frequency"]
    assert _keyword_only_names(transform)


@pytest.mark.parametrize(
    ("make", "third"),
    [
        pytest.param(ShortTimeFourierTransform, "constant", id="ShortTimeFourierTransform"),
        pytest.param(Welch, 0.5, id="Welch"),
        pytest.param(MorletWavelet, [10.0, 20.0], id="MorletWavelet"),
    ],
)
def test_new_transforms_reject_a_positional_setting(make, third):
    time_series = np.zeros((200, 1, 2))
    with pytest.raises(TypeError, match="positional argument"):
        make(time_series, 100, third)


def test_fourier_connectivity_takes_settings_after_signal_names_by_keyword_only():
    """New in 3.0 (absent from v2.0.1), so its signature is set directly."""
    assert _positional_names(fourier_connectivity) == [
        "fourier_coefficients",
        "frequencies",
        "time",
        "method",
        "signal_names",
    ]
    assert {"squeeze", "connectivity_kwargs", "is_one_sided"} <= set(
        _keyword_only_names(fourier_connectivity)
    )
    coefficients = np.ones((4, 8, 2), dtype=complex)
    with pytest.raises(TypeError, match="positional argument"):
        fourier_connectivity(coefficients, None, None, "power", None, False)


def test_lagged_broadband_names_its_delays_in_samples():
    """New in 3.0: the delays are ``lags_samples`` (no ``lags`` alias), and
    ``n_trials`` defaults to one trial."""
    assert _positional_names(simulate_lagged_broadband) == [
        "lags_samples",
        "noise_levels",
        "n_time_samples",
        "n_trials",
        "random_state",
    ]
    assert inspect.signature(simulate_lagged_broadband).parameters["n_trials"].default == 1


def test_shared_oscillation_takes_its_amplitudes_by_keyword_only():
    """New in 3.0: ``n_trials`` defaults to one trial, so the required
    ``amplitudes`` after it is keyword-only."""
    assert _positional_names(simulate_shared_oscillation) == [
        "frequency",
        "sampling_frequency",
        "n_time_samples",
        "n_trials",
    ]
    parameters = inspect.signature(simulate_shared_oscillation).parameters
    assert parameters["n_trials"].default == 1
    assert "amplitudes" in _keyword_only_names(simulate_shared_oscillation)
    assert parameters["amplitudes"].default is inspect.Parameter.empty


@pytest.mark.parametrize(
    "name",
    [
        "canonical_coherency",
        "maximized_imaginary_coherency_components",
        "maximized_imaginary_coherency",
        "multivariate_interaction_measure",
        "partial_coherence",
    ],
)
def test_new_measures_take_tuning_parameters_by_keyword_only(name):
    method = getattr(Connectivity, name)
    accepted = {"rank", "regularization", "n_components"} & set(
        inspect.signature(method).parameters
    )
    assert accepted
    assert accepted <= set(_keyword_only_names(method))


def test_canonical_coherence_keeps_its_released_signature_without_tuning():
    assert _positional_names(Connectivity.canonical_coherence) == ["group_labels"]
    assert _keyword_only_names(Connectivity.canonical_coherence) == []


@pytest.fixture(scope="module")
def connectivity():
    time_series = np.random.default_rng(1).standard_normal((256, 8, 4))
    return Connectivity.from_transform(Multitaper(time_series, sampling_frequency=128))


def test_minimum_phase_decomposition_keeps_its_positional_tolerances():
    """``tolerance`` and ``max_iterations`` keep their released order and
    remain supported positionally, without a warning."""
    assert _positional_names(minimum_phase_decomposition) == [
        "cross_spectral_matrix",
        "tolerance",
        "max_iterations",
    ]
    rng = np.random.default_rng(2)
    mixing = rng.standard_normal((2, 2))
    spectrum = np.tile(mixing @ mixing.T + 2 * np.eye(2), (1, 16, 1, 1)).astype(complex)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        positional = minimum_phase_decomposition(spectrum, 1e-10, 50)
        keyword = minimum_phase_decomposition(spectrum, tolerance=1e-10, max_iterations=50)
    np.testing.assert_array_equal(to_host(positional), to_host(keyword))


def test_global_coherence_keeps_released_max_rank_and_workspace_is_keyword_only(
    connectivity,
):
    assert _positional_names(Connectivity.global_coherence) == ["max_rank"]
    assert _keyword_only_names(Connectivity.global_coherence) == ["max_workspace_elements"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # The 3.0 frequency-axis notice is not about the arguments.
        warnings.filterwarnings("ignore", category=SidednessWarning)
        positional = connectivity.global_coherence(2)
    keyword = connectivity.global_coherence(max_rank=2)
    assert positional[0].shape[-1] == 2
    for by_position, by_keyword in zip(positional, keyword, strict=True):
        np.testing.assert_array_equal(by_position, by_keyword)
    with pytest.raises(TypeError, match="positional argument"):
        connectivity.global_coherence(2, 1000)


def test_multitaper_connectivity_keeps_its_released_positional_order():
    assert _positional_names(multitaper_connectivity) == [
        "time_series",
        "sampling_frequency",
        "time_window_duration",
        "method",
        "signal_names",
        "squeeze",
        "connectivity_kwargs",
    ]
    assert {
        "time_window_step",
        "time_halfbandwidth_product",
        "n_tapers",
        "detrend_type",
        "transform_kwargs",
    } <= set(_keyword_only_names(multitaper_connectivity))


def test_connectivity_to_xarray_keeps_its_released_positional_order():
    assert _positional_names(connectivity_to_xarray) == [
        "m",
        "method",
        "signal_names",
        "squeeze",
    ]
    assert {"group_labels", "frequency_range", "connectivity_kwargs"} <= set(
        _keyword_only_names(connectivity_to_xarray)
    )
    assert "transform_kwargs" not in inspect.signature(connectivity_to_xarray).parameters


def test_power_confidence_intervals_keeps_its_released_argument_order():
    """The released positional order (count, power, level) is unchanged; only
    the names of the count and the level changed, and the old names are
    keyword-only aliases. The real-valued-bin mask is new and keyword-only."""
    assert _positional_names(power_confidence_intervals) == [
        "n_observations",
        "power",
        "confidence_level",
    ]
    assert _keyword_only_names(power_confidence_intervals) == [
        "n_tapers",
        "ci",
        "real_valued",
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        by_position = power_confidence_intervals(25, np.array([0.5, 2.0]), 0.9)
        by_keyword = power_confidence_intervals(
            n_observations=25, power=np.array([0.5, 2.0]), confidence_level=0.9
        )
    np.testing.assert_array_equal(by_position, by_keyword)


@pytest.mark.parametrize("constructor", ["VARSpectrum", "Connectivity.from_var"])
def test_var_constructors_take_only_the_model_positionally(constructor):
    """The model, as simulate_MVAR takes it; every setting by keyword."""
    from spectral_connectivity import VARSpectrum

    function = VARSpectrum if constructor == "VARSpectrum" else Connectivity.from_var
    assert _positional_names(function) == ["coefficients", "noise_covariance"]
    assert "sampling_frequency" in _keyword_only_names(function)
