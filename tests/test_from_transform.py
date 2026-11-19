"""``Connectivity.from_transform`` and its 2.x name ``from_multitaper``."""

import numpy as np
import pytest

from spectral_connectivity import Connectivity, Multitaper, multitaper_connectivity
from spectral_connectivity.wrapper import connectivity_to_xarray


@pytest.fixture(scope="module")
def multitaper():
    time_series = np.random.default_rng(0).standard_normal((512, 6, 3))
    return Multitaper(time_series, sampling_frequency=256, time_window_duration=0.5)


def test_package_internals_construct_through_from_transform(monkeypatch, multitaper):
    """Normal use never reaches the 2.x constructor name."""

    def fail(*args, **kwargs):
        msg = "package code called Connectivity.from_multitaper"
        raise AssertionError(msg)

    monkeypatch.setattr(Connectivity, "from_multitaper", classmethod(fail))
    multitaper_connectivity(
        multitaper.time_series, sampling_frequency=256, method="magnitude_squared_coherence"
    )
    connectivity_to_xarray(multitaper, "magnitude_squared_coherence")
    Connectivity.from_transform(multitaper).magnitude_squared_coherence()
