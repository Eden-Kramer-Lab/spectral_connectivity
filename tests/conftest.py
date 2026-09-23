import sys

import numpy as np


def pytest_configure(config):
    # NumPy < 2.3.1 with macOS Accelerate emits divide-by-zero/overflow/invalid
    # RuntimeWarnings from some matmul shapes, e.g. (1000, 3) @ (3, 3), although
    # the product is finite and exact. Python 3.10 cannot install a fixed NumPy.
    if sys.platform == "darwin" and np.lib.NumpyVersion(np.__version__) < "2.3.1":
        config.addinivalue_line(
            "filterwarnings", "ignore:.*encountered in matmul:RuntimeWarning"
        )
