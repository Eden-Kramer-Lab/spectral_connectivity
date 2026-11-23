"""``is_exact_spectrum``: coefficients that factor a cross-spectral matrix.

Their taper-mean outer product is a population spectrum and the observations
are not samples, so only the measures the registry marks
``defined_for_exact_spectrum`` are computed; every other one, and the
jackknife, raises.
"""

import warnings

import numpy as np
import pytest

from spectral_connectivity import Connectivity, list_measures
from spectral_connectivity._measure_registry import _EXACT_SPECTRUM_MEASURES, _MEASURE_SPECS
from spectral_connectivity.connectivity import EXPECTATION_AXES
from tests._var_oracle import (
    _STRONG_CHAIN_COEFFICIENTS,
    _analytic_var,
    _fourier_coefficients_with_cross_spectrum,
)

_GROUPS = ["a", "a", "b"]
_MEASURE_KWARGS = {
    "canonical_coherence": {"group_labels": _GROUPS},
    "canonical_coherency": {"group_labels": _GROUPS},
    "maximized_imaginary_coherency": {"group_labels": _GROUPS},
    "maximized_imaginary_coherency_components": {"group_labels": _GROUPS},
    "multivariate_interaction_measure": {"group_labels": _GROUPS},
    "blockwise_spectral_granger_prediction": {"group_labels": _GROUPS},
    "time_reversed_blockwise_spectral_granger_prediction": {"group_labels": _GROUPS},
    "subset_pairwise_spectral_granger_prediction": {"pairs": [(0, 1)]},
}
_REFUSED = [
    name for name, spec in _MEASURE_SPECS.items() if not spec.defined_for_exact_spectrum
]


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
    assert sorted(_REFUSED) == sorted(
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
    assert set(_EXACT_SPECTRUM_MEASURES) | set(_REFUSED) == {m.name for m in list_measures()}


@pytest.mark.parametrize("name", _REFUSED)
def test_measures_using_observations_refuse_an_exact_spectrum(coefficients, name):
    connectivity = Connectivity(coefficients, is_exact_spectrum=True)
    with pytest.raises(
        ValueError, match=rf"^{name} is not defined for an exact spectrum"
    ) as info:
        _compute(connectivity, name)
    message = str(info.value)
    assert "one factorization of the model's cross-spectral matrix" in message
    assert all(allowed in message for allowed in _EXACT_SPECTRUM_MEASURES)
    assert "simulate_MVAR" in message


@pytest.mark.parametrize("name", _EXACT_SPECTRUM_MEASURES)
def test_measures_of_the_spectral_matrix_are_unchanged_by_the_flag(coefficients, name):
    """The flag refuses measures; it changes none of the others' values."""
    exact = _compute(Connectivity(coefficients, is_exact_spectrum=True), name)
    plain = _compute(Connectivity(coefficients), name)
    if hasattr(exact, "scores"):  # MultivariateConnectivityResult
        exact, plain = exact.scores, plain.scores
    for exact_part, plain_part in zip(
        exact if isinstance(exact, tuple) else (exact,),
        plain if isinstance(plain, tuple) else (plain,),
        strict=True,
    ):
        np.testing.assert_array_equal(exact_part, plain_part)


def test_jackknife_refuses_an_exact_spectrum(coefficients):
    connectivity = Connectivity(coefficients, is_exact_spectrum=True)
    with pytest.raises(ValueError, match=r"^jackknife is not defined.*leaves out one"):
        connectivity.jackknife("magnitude_squared_coherence")


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


def test_from_transform_reads_the_flag(coefficients):
    exact = Connectivity.from_transform(_ExactTransform(coefficients))
    assert exact.is_exact_spectrum
    with pytest.raises(ValueError, match="not defined for an exact spectrum"):
        exact.phase_locking_value()
    assert not Connectivity(coefficients).is_exact_spectrum


def test_a_flag_that_is_not_a_bool_is_rejected(coefficients):
    transform = _ExactTransform(coefficients)
    transform.is_exact_spectrum = "False"
    with pytest.raises(TypeError, match=r"^transform\.is_exact_spectrum must be a bool"):
        Connectivity.from_transform(transform)
    with pytest.raises(TypeError, match=r"^is_exact_spectrum must be a bool"):
        Connectivity(coefficients, is_exact_spectrum=1)
