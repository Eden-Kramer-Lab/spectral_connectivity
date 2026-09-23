"""Tests for the ``observations_are_independent`` contract on ``Connectivity``.

Transforms whose observations are correlated (a Morlet transform whose
smoothing neighborhood is collected on the observation axis, or Welch segments
overlapping by more than half) expose ``observations_are_independent=False``.
``Connectivity`` records the flag so consumers that assume independent
trial-taper observations (finite-sample bias corrections, the zero-coherence
null, leave-one-out intervals) can warn or refuse.
"""

import numpy as np
import pytest

from spectral_connectivity import Connectivity


class _StubTransform:
    """Duck-typed transform exposing the ``from_transform`` contract."""

    is_one_sided = False

    def __init__(self, coefficients, *, observations_are_independent=None):
        self._coefficients = coefficients
        if observations_are_independent is not None:
            self.observations_are_independent = observations_are_independent
        n_frequencies = coefficients.shape[-2]
        self.frequencies = np.fft.fftfreq(n_frequencies)
        self.time = np.arange(coefficients.shape[0], dtype=float)

    def fft(self):
        return self._coefficients.copy()


def _coefficients(rng, shape=(1, 4, 3, 16, 3)):
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def test_connectivity_defaults_to_independent_observations():
    conn = Connectivity(_coefficients(np.random.default_rng(0)))
    assert conn.observations_are_independent is True


def test_connectivity_records_correlated_observations():
    conn = Connectivity(
        _coefficients(np.random.default_rng(0)), observations_are_independent=False
    )
    assert conn.observations_are_independent is False
    with pytest.raises(AttributeError):
        conn.observations_are_independent = True  # read-only


def test_from_transform_forwards_the_flag_from_a_stub_transform():
    rng = np.random.default_rng(1)
    correlated = _StubTransform(_coefficients(rng), observations_are_independent=False)
    assert Connectivity.from_transform(correlated).observations_are_independent is False
    independent = _StubTransform(_coefficients(rng), observations_are_independent=True)
    assert Connectivity.from_transform(independent).observations_are_independent is True


def test_from_transform_treats_a_missing_attribute_as_independent():
    stub = _StubTransform(_coefficients(np.random.default_rng(2)))
    assert not hasattr(stub, "observations_are_independent")
    assert Connectivity.from_transform(stub).observations_are_independent is True


def test_from_transform_does_not_pass_the_keyword_to_an_older_subclass():
    """A subclass mirroring the pre-flag signature must keep working with an
    independent-observation transform; the keyword is only sent when False."""

    class Legacy(Connectivity):
        def __init__(self, fourier_coefficients, expectation_type="trials_tapers", **kwargs):
            kwargs.pop("is_one_sided", None)
            assert "observations_are_independent" not in kwargs
            super().__init__(fourier_coefficients, expectation_type, **kwargs)

    stub = _StubTransform(_coefficients(np.random.default_rng(3)))
    assert Legacy.from_transform(stub).observations_are_independent is True


def test_flag_survives_coefficient_reassignment():
    rng = np.random.default_rng(4)
    conn = Connectivity(_coefficients(rng), observations_are_independent=False)
    conn.fourier_coefficients = _coefficients(rng)
    assert conn.observations_are_independent is False
