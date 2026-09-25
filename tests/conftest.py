import importlib
import pkgutil
import sys

import numpy as np
import pytest

import spectral_connectivity


def pytest_configure(config):
    # NumPy < 2.3.1 with macOS Accelerate emits divide-by-zero/overflow/invalid
    # RuntimeWarnings from some matmul shapes, e.g. (1000, 3) @ (3, 3), although
    # the product is finite and exact. Python 3.10 cannot install a fixed NumPy.
    if sys.platform == "darwin" and np.lib.NumpyVersion(np.__version__) < "2.3.1":
        config.addinivalue_line(
            "filterwarnings", "ignore:.*encountered in matmul:RuntimeWarning"
        )


@pytest.fixture(scope="session")
def backend_modules():
    """Every package module that binds the array namespace ``xp``.

    Device-emulation tests swap ``xp`` in each of these. Discovering them, rather
    than listing them, keeps a new module from silently escaping the emulation.
    """
    modules = [
        importlib.import_module(info.name)
        for info in pkgutil.iter_modules(
            spectral_connectivity.__path__, "spectral_connectivity."
        )
    ]
    return tuple(module for module in modules if hasattr(module, "xp"))
