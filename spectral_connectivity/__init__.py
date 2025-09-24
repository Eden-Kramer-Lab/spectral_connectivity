"""Spectral connectivity analysis for electrophysiological data.

This package provides tools for computing frequency-domain functional and
directed connectivity measures from time series data using multitaper methods.
"""

# flake8: noqa
from spectral_connectivity.connectivity import Connectivity
from spectral_connectivity.transforms import Multitaper
from spectral_connectivity.wrapper import multitaper_connectivity

__version__ = "1.1.2"
__all__ = ["Connectivity", "Multitaper", "multitaper_connectivity"]
