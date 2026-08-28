"""Helpful errors when an unknown connectivity method name is requested."""

import numpy as np
import pytest

from spectral_connectivity import (
    Connectivity,
    fourier_connectivity,
    multitaper_connectivity,
)


@pytest.fixture
def time_series():
    return np.random.default_rng(0).random((200, 5, 2))


def test_extension_measure_is_not_rejected_by_validation(time_series, monkeypatch):
    """A subclass/monkeypatched extension measure still reaches Connectivity."""

    def custom_similarity(connectivity):
        return np.zeros(
            (
                len(connectivity.time),
                len(connectivity.frequencies),
                connectivity.n_signals,
                connectivity.n_signals,
            )
        )

    monkeypatch.setattr(
        Connectivity, "custom_similarity", custom_similarity, raising=False
    )
    result = multitaper_connectivity(
        time_series, sampling_frequency=200, method="custom_similarity"
    )
    assert result.dims == ("time", "frequency", "source", "target")


def test_unknown_method_raises_valueerror_not_attributeerror(time_series):
    """A misspelled measure name fails as a ValueError, not AttributeError."""
    with pytest.raises(ValueError) as excinfo:
        multitaper_connectivity(time_series, sampling_frequency=200, method="granger")
    assert not isinstance(excinfo.value, AttributeError)
    assert "granger" in str(excinfo.value)


def test_unknown_method_suggests_close_match(time_series):
    """A near-miss name points the user at the real measure."""
    with pytest.raises(ValueError) as excinfo:
        multitaper_connectivity(time_series, sampling_frequency=200, method="coherence")
    assert "coherence_magnitude" in str(excinfo.value)


def test_unknown_method_points_to_list_measures(time_series):
    """The error tells the user how to enumerate valid measures."""
    with pytest.raises(ValueError) as excinfo:
        multitaper_connectivity(
            time_series, sampling_frequency=200, method="bogus_measure_xyz"
        )
    assert "list_measures" in str(excinfo.value)


def test_unknown_method_in_a_list_is_validated(time_series):
    """One bad name among valid ones is still caught before computing."""
    with pytest.raises(ValueError) as excinfo:
        multitaper_connectivity(
            time_series,
            sampling_frequency=200,
            method=["coherence_magnitude", "not_real"],
        )
    assert "not_real" in str(excinfo.value)


def test_fourier_connectivity_validates_method_names():
    """The Fourier entry point validates method names the same way."""
    rng = np.random.default_rng(0)
    # (observation, frequency, signal)
    fourier_coefficients = rng.random((5, 4, 2)) + 0j
    frequencies = np.linspace(0, 100, 4)
    with pytest.raises(ValueError) as excinfo:
        fourier_connectivity(
            fourier_coefficients,
            frequencies=frequencies,
            method="coherence",
        )
    assert "coherence_magnitude" in str(excinfo.value)
