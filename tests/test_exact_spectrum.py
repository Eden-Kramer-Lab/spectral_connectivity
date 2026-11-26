"""``is_exact_spectrum``: coefficients that factor a cross-spectral matrix.

Their taper-mean outer product is a population spectrum and the observations
are not samples, so only the measures the registry marks
``defined_for_exact_spectrum`` are computed; every other one, the jackknife,
the surrogate test and "auto" partial-coherence shrinkage raise.
"""

import warnings

import numpy as np
import pytest

from spectral_connectivity import Connectivity, list_measures
from spectral_connectivity._measure_registry import _EXACT_SPECTRUM_MEASURES
from spectral_connectivity.connectivity import EXPECTATION_AXES
from tests._measure_helpers import REFUSED_FOR_EXACT_SPECTRUM, measure_kwargs
from tests._var_oracle import (
    _STRONG_CHAIN_COEFFICIENTS,
    _analytic_var,
    _fourier_coefficients_with_cross_spectrum,
)

_GROUPS = ["a", "a", "b"]
_MEASURE_KWARGS = measure_kwargs(_GROUPS)


@pytest.fixture(scope="module")
def coefficients():
    """Coefficients whose taper mean is the exact spectrum of a 3-signal chain."""
    _, _, spectrum = _analytic_var(_STRONG_CHAIN_COEFFICIENTS, np.eye(3), 64)
    return _fourier_coefficients_with_cross_spectrum(spectrum, conjugate_symmetric=True)


class _ExactTransform:
    """A transform that declares its coefficients an exact spectrum."""

    is_exact_spectrum = True
    time = None
    frequencies = None

    def __init__(self, coefficients):
        self._coefficients = coefficients

    def fft(self):
        return self._coefficients.copy()


def _compute(connectivity, name):
    with warnings.catch_warnings():
        # Orientation and sidedness notices, and the degenerate-bin warnings of
        # some measures, are irrelevant here.
        warnings.simplefilter("ignore", UserWarning)
        return getattr(connectivity, name)(**_MEASURE_KWARGS.get(name, {}))


def test_the_refused_measures_are_the_ones_that_use_observations():
    """The registry's split, read from each implementation, as a list to review."""
    assert sorted(REFUSED_FOR_EXACT_SPECTRUM) == sorted(
        [
            "phase_locking_value",
            "corrected_imaginary_phase_locking_value",
            "phase_lag_index",
            "directed_phase_lag_index",
            "weighted_phase_lag_index",
            "debiased_squared_phase_lag_index",
            "debiased_squared_weighted_phase_lag_index",
            "pairwise_phase_consistency",
            "delay",
            "group_delay",
        ]
    )
    assert set(_EXACT_SPECTRUM_MEASURES) | set(REFUSED_FOR_EXACT_SPECTRUM) == {
        m.name for m in list_measures()
    }


@pytest.mark.parametrize("name", REFUSED_FOR_EXACT_SPECTRUM)
def test_measures_using_observations_refuse_an_exact_spectrum(coefficients, name):
    connectivity = Connectivity(coefficients, is_exact_spectrum=True)
    with pytest.raises(
        ValueError, match=rf"^{name} is not defined for an exact spectrum"
    ) as info:
        _compute(connectivity, name)
    message = str(info.value)
    assert "one factor of the model's cross-spectral matrix, not samples" in message
    assert "simulate_MVAR" in message
    reason = (
        "significance test whose degrees of freedom are n_observations"
        if name in {"delay", "group_delay"}
        else "averages a function of each trial/taper observation"
    )
    assert reason in message


@pytest.mark.parametrize("name", _EXACT_SPECTRUM_MEASURES)
def test_measures_of_the_spectral_matrix_are_unchanged_by_the_flag(coefficients, name):
    """The flag refuses measures; it changes none of the others' values.

    Compared to rounding rather than bitwise: the conditional Granger
    factorizations differed by 2.2e-16 (13 of 297 elements) between two
    identical computations on the Linux minimum-dependency CI job, where
    threaded BLAS can reorder sums; the flag itself changes no arithmetic.
    """
    exact = _compute(Connectivity(coefficients, is_exact_spectrum=True), name)
    plain = _compute(Connectivity(coefficients), name)
    if hasattr(exact, "scores"):  # MultivariateConnectivityResult
        exact, plain = exact.scores, plain.scores
    for exact_part, plain_part in zip(
        exact if isinstance(exact, tuple) else (exact,),
        plain if isinstance(plain, tuple) else (plain,),
        strict=True,
    ):
        if np.issubdtype(np.asarray(exact_part).dtype, np.number):
            np.testing.assert_allclose(exact_part, plain_part, rtol=1e-12, atol=1e-15)
        else:  # group labels
            np.testing.assert_array_equal(exact_part, plain_part)


def test_jackknife_refuses_an_exact_spectrum(coefficients):
    connectivity = Connectivity(coefficients, is_exact_spectrum=True)
    with pytest.raises(ValueError, match=r"^jackknife is not defined.*no sampling error"):
        connectivity.jackknife("magnitude_squared_coherence")


def test_auto_shrinkage_refuses_an_exact_spectrum(coefficients):
    """ "auto" estimates sampling variance; a fixed intensity is a function of
    the spectrum and is computed."""
    connectivity = Connectivity(coefficients, is_exact_spectrum=True)
    for call, name in [
        (lambda: connectivity.partial_coherence(shrinkage="auto"), "partial_coherence"),
        (connectivity.partial_coherence_shrinkage, "partial_coherence_shrinkage"),
    ]:
        with pytest.raises(
            ValueError, match=rf"^{name}\(.*is not defined for an exact spectrum"
        ) as info:
            call()
        assert "sampling variance" in str(info.value)
    np.testing.assert_allclose(
        connectivity.partial_coherence(shrinkage=0.3),
        Connectivity(coefficients).partial_coherence(shrinkage=0.3),
        rtol=1e-12,
        atol=1e-15,
    )


def test_observation_weights_are_refused_on_an_exact_spectrum(coefficients):
    """Weighting the pseudo-tapers would change the spectrum itself."""
    weights = np.ones((*coefficients.shape[:-1], 1))
    match = "^observation_weights cannot be set on an exact spectrum"
    with pytest.raises(ValueError, match=match):
        Connectivity(coefficients, is_exact_spectrum=True, observation_weights=weights)
    connectivity = Connectivity(coefficients, is_exact_spectrum=True)
    with pytest.raises(ValueError, match=match):
        connectivity.observation_weights = weights
    connectivity.observation_weights = None
    transform = _ExactTransform(coefficients)
    transform.observation_weights = weights
    with pytest.raises(ValueError, match=match):
        Connectivity.from_transform(transform)


@pytest.mark.parametrize("expectation_type", sorted(EXPECTATION_AXES))
def test_the_expectation_must_average_the_tapers(coefficients, expectation_type):
    """A single taper is one column of a factor, not a spectrum: refused at
    construction and on reassignment."""
    averages_tapers = 2 in EXPECTATION_AXES[expectation_type]
    if averages_tapers:
        connectivity = Connectivity(
            coefficients, expectation_type=expectation_type, is_exact_spectrum=True
        )
        assert connectivity.expectation_type == expectation_type
        return
    match = rf"^expectation_type={expectation_type!r} keeps the tapers"
    with pytest.raises(ValueError, match=match):
        Connectivity(coefficients, expectation_type=expectation_type, is_exact_spectrum=True)
    connectivity = Connectivity(coefficients, is_exact_spectrum=True)
    with pytest.raises(ValueError, match=match):
        connectivity.expectation_type = expectation_type
    assert connectivity.expectation_type == "trials_tapers"
    # Without the flag the same expectation is accepted.
    Connectivity(coefficients, expectation_type=expectation_type)


def test_from_transform_reads_the_flag_strictly(coefficients):
    """The transform's flag reaches the instance, and must be a bool there as
    on Connectivity itself."""
    exact = Connectivity.from_transform(_ExactTransform(coefficients))
    assert exact.is_exact_spectrum
    with pytest.raises(ValueError, match="not defined for an exact spectrum"):
        exact.phase_locking_value()
    assert not Connectivity(coefficients).is_exact_spectrum
    transform = _ExactTransform(coefficients)
    transform.is_exact_spectrum = "False"
    with pytest.raises(TypeError, match=r"^transform\.is_exact_spectrum must be a bool"):
        Connectivity.from_transform(transform)
    with pytest.raises(TypeError, match=r"^is_exact_spectrum must be a bool"):
        Connectivity(coefficients, is_exact_spectrum=1)
