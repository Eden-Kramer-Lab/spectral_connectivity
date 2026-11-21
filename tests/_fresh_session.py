"""Run a script in a fresh interpreter, to test what a whole session shows.

The migration warnings are shown once per Python session, which a test in
this process (where earlier tests have already issued them, under the
suite's own filters) cannot observe.
"""

import json
import os
import subprocess
import sys
from typing import Any

# Prepended to every script: ``shown`` records the message of each warning
# Python's filters let through, without changing the filters, and
# ``connectivity`` is ready to call.
_PREAMBLE = """
import json
import warnings

import numpy as np

shown = []
warnings.showwarning = lambda message, *args, **kwargs: shown.append(str(message))
from spectral_connectivity import (
    Connectivity,
    DirectedOrientationWarning,
    Multitaper,
    PowerSidednessWarning,
    multitaper_connectivity,
)

time_series = np.random.default_rng(0).standard_normal((64, 3, 2))
connectivity = Connectivity.from_transform(Multitaper(time_series, sampling_frequency=100))
"""


def run_fresh_session(body: str, *python_flags: str) -> Any:
    """Run the preamble and ``body`` in a new interpreter; return its JSON output.

    ``body`` must end by printing one JSON value. ``PYTHONWARNINGS`` is
    removed from the environment so the session starts with Python's default
    filters; ``python_flags`` (such as ``"-X", "context_aware_warnings=1"``)
    go before ``-c``.
    """
    environment = {k: v for k, v in os.environ.items() if k != "PYTHONWARNINGS"}
    output = subprocess.check_output(
        [sys.executable, *python_flags, "-c", _PREAMBLE + body], text=True, env=environment
    )
    return json.loads(output)
