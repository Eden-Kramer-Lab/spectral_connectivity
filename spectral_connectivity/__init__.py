"""Spectral connectivity analysis for electrophysiological data.

This package provides tools for computing frequency-domain functional and
directed connectivity measures from time series data using multitaper methods.
"""

# flake8: noqa
from spectral_connectivity.connectivity import Connectivity
from spectral_connectivity.transforms import Multitaper, prepare_time_series
from spectral_connectivity.wrapper import multitaper_connectivity

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
    "Multitaper",
    "multitaper_connectivity",
    "prepare_time_series",
]
