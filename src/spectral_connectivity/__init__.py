"""Spectral connectivity analysis for electrophysiological data.

This package provides tools for computing frequency-domain functional and
directed connectivity measures from time series data using multitaper methods.

Start with :func:`multitaper_connectivity`, which takes a time series shaped
``(n_time_samples, n_trials, n_signals)`` and returns a labeled xarray result,
and :func:`list_measures`, which lists every valid ``method`` with its units,
value range, and interpretation. In the wrapper's results,
``result.sel(source="a", target="b")`` is the influence of ``a`` on ``b``. The
lower-level :class:`Connectivity` methods return plain arrays whose signal axes
come in two orders; ``MeasureInfo.array_orientation`` names each measure's.

Guide for AI coding assistants:
https://spectral-connectivity.readthedocs.io/en/latest/llm_guide.html

Examples
--------
>>> import numpy as np
>>> from spectral_connectivity import multitaper_connectivity
>>> time_series = np.random.default_rng(0).standard_normal((1000, 5, 2))
>>> coherence = multitaper_connectivity(
...     time_series, sampling_frequency=500, method="coherence_magnitude"
... )
>>> coherence.dims
('time', 'frequency', 'source', 'target')
"""

from spectral_connectivity.connectivity import (
    Connectivity,
    MultivariateConnectivityResult,
)
from spectral_connectivity.minimum_phase_decomposition import (
    minimum_phase_reconstruction_error,
)
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
    MeasureInfo,
    fourier_connectivity,
    frequency_band_reduce,
    list_measures,
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
    "DEFAULT_METHODS",
    "Connectivity",
    "JackknifeResult",
    "MeasureInfo",
    "MorletWavelet",
    "Multitaper",
    "MultitaperParameters",
    "MultivariateConnectivityResult",
    "ShortTimeFourierTransform",
    "Welch",
    "estimate_frequency_resolution",
    "estimate_n_tapers",
    "fourier_connectivity",
    "frequency_band_reduce",
    "get_compute_backend",
    "jackknife_confidence_interval",
    "list_measures",
    "minimum_phase_reconstruction_error",
    "multitaper_connectivity",
    "prepare_time_series",
    "suggest_parameters",
]
