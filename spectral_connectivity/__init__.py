"""Spectral connectivity analysis for electrophysiological data.

This package provides tools for computing frequency-domain functional and
directed connectivity measures from time series data using multitaper methods.
"""

# flake8: noqa
from spectral_connectivity.connectivity import Connectivity
from spectral_connectivity.statistics import (
    JackknifeResult,
    jackknife_confidence_interval,
)
from spectral_connectivity.transforms import (
    MorletWavelet,
    Multitaper,
    MultitaperParameters,
    ShortTimeFourierTransform,
    Welch,
    estimate_frequency_resolution,
    estimate_n_tapers,
    prepare_time_series,
    suggest_parameters,
)
from spectral_connectivity.utils import get_compute_backend
from spectral_connectivity.wrapper import (
    DEFAULT_METHODS,
    fourier_connectivity,
    frequency_band_reduce,
    multitaper_connectivity,
)

# Import version information
try:
    from spectral_connectivity._version import __version__
except ImportError:
    # Fallback for development installs
    from importlib.metadata import version

    __version__ = version("spectral_connectivity")

# Define the public API of the package
__all__ = [
    "Connectivity",
    "DEFAULT_METHODS",
    "JackknifeResult",
    "MorletWavelet",
    "Multitaper",
    "MultitaperParameters",
    "ShortTimeFourierTransform",
    "Welch",
    "fourier_connectivity",
    "frequency_band_reduce",
    "jackknife_confidence_interval",
    "multitaper_connectivity",
    "prepare_time_series",
    "estimate_frequency_resolution",
    "estimate_n_tapers",
    "suggest_parameters",
    "get_compute_backend",
]
