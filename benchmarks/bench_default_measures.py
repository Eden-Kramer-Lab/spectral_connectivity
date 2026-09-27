"""Time the default connectivity measures and check their outputs for parity.

Two fixed cases (seed 0, 1000 Hz, white noise):

- (a) ``n_time=5000, n_trials=50, n_signals=8, time_window_duration=0.5``
- (b) ``n_time=2000, n_trials=100, n_signals=32, time_window_duration=None``

For each case this times the multitaper transform, then every measure in
``DEFAULT_METHODS`` (``coherence_magnitude`` among them) on a fresh
``Connectivity``, and the whole default set on one ``Connectivity`` (the way
the wrapper computes it, sharing cached intermediates), and likewise the
default phase-lag-index measures together. The measure timings
start from the precomputed coefficients, so they exclude the transform. Each
time is the fastest of ``--repeat`` runs, printed with the process's peak
resident memory so far.

Usage
-----
Capture a baseline before changing cached intermediates in ``connectivity.py``,
then check the change against it::

    uv run python benchmarks/bench_default_measures.py --save baseline.npz
    uv run python benchmarks/bench_default_measures.py --compare baseline.npz

``--save`` writes every default measure's array (from the default-set run) for
both cases to one ``.npz``. ``--compare`` prints the largest absolute
difference per measure against that file and exits non-zero when any exceeds
``1e-12`` or the NaN patterns differ.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np

from spectral_connectivity import DEFAULT_METHODS, Connectivity, Multitaper

try:
    import resource
except ImportError:  # Windows: no getrusage, so no peak-memory column
    resource = None  # type: ignore[assignment]

SAMPLING_FREQUENCY = 1000.0
TOLERANCE = 1e-12
PHASE_LAG_MEASURES = tuple(name for name in DEFAULT_METHODS if "phase_lag" in name)
CASES: dict[str, dict[str, Any]] = {
    "a": {"n_time": 5000, "n_trials": 50, "n_signals": 8, "time_window_duration": 0.5},
    "b": {"n_time": 2000, "n_trials": 100, "n_signals": 32, "time_window_duration": None},
}


def _peak_rss_mb() -> float:
    """Peak resident memory of this process so far, in MB (NaN without ``resource``)."""
    if resource is None:
        return np.nan
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is bytes on macOS and kilobytes on Linux.
    return peak / 2**20 if sys.platform == "darwin" else peak / 2**10


def _best_time(function: Callable[[], Any], repeat: int) -> tuple[float, Any]:
    """Fastest wall time of ``repeat`` calls and the last call's result."""
    best = np.inf
    result = None
    for _ in range(repeat):
        start = time.perf_counter()
        result = function()
        best = min(best, time.perf_counter() - start)
    return best, result


def _connectivity(multitaper: Multitaper, coefficients: np.ndarray) -> Connectivity:
    """A fresh ``Connectivity`` on precomputed coefficients (no FFT timed)."""
    return Connectivity(coefficients, frequencies=multitaper.frequencies, time=multitaper.time)


def _measure_set(
    multitaper: Multitaper, coefficients: np.ndarray, measures: tuple[str, ...]
) -> dict[str, np.ndarray]:
    """Compute ``measures`` in turn on one ``Connectivity``, sharing its caches."""
    connectivity = _connectivity(multitaper, coefficients)
    return {name: getattr(connectivity, name)() for name in measures}


def run_case(
    name: str, parameters: dict[str, Any], repeat: int
) -> tuple[list[tuple[str, float, float]], dict[str, np.ndarray]]:
    """Time one case; return ``(label, seconds, peak MB)`` rows and its outputs."""
    rng = np.random.default_rng(0)
    time_series = rng.standard_normal(
        (parameters["n_time"], parameters["n_trials"], parameters["n_signals"])
    )
    rows = []

    multitaper = Multitaper(
        time_series,
        sampling_frequency=SAMPLING_FREQUENCY,
        time_window_duration=parameters["time_window_duration"],
    )
    seconds, coefficients = _best_time(multitaper.fft, repeat)
    rows.append(("Multitaper.fft", seconds, _peak_rss_mb()))

    for measure in DEFAULT_METHODS:
        seconds, _ = _best_time(
            lambda measure=measure: getattr(
                _connectivity(multitaper, coefficients), measure
            )(),
            repeat,
        )
        rows.append((measure, seconds, _peak_rss_mb()))

    seconds, _ = _best_time(
        lambda: _measure_set(multitaper, coefficients, PHASE_LAG_MEASURES), repeat
    )
    rows.append(
        (f"phase-lag set ({len(PHASE_LAG_MEASURES)} measures)", seconds, _peak_rss_mb())
    )
    seconds, outputs = _best_time(
        lambda: _measure_set(multitaper, coefficients, DEFAULT_METHODS), repeat
    )
    rows.append(("default set", seconds, _peak_rss_mb()))
    print(f"\nCase ({name}): {parameters}, fourier_coefficients {coefficients.shape}")
    memory_header = f" {'peak MB':>9}" if resource is not None else ""
    print(f"{'measure':<45} {'seconds':>9}{memory_header}")
    for label, row_seconds, peak in rows:
        memory = f" {peak:>9.0f}" if resource is not None else ""
        print(f"{label:<45} {row_seconds:>9.3f}{memory}")
    return rows, outputs


def max_abs_difference(actual: np.ndarray, expected: np.ndarray) -> float:
    """Largest absolute difference; ``inf`` if shapes or NaN patterns differ."""
    if actual.shape != expected.shape:
        return np.inf
    actual_nan, expected_nan = np.isnan(actual), np.isnan(expected)
    if not np.array_equal(actual_nan, expected_nan):
        return np.inf
    finite = ~actual_nan
    if not finite.any():
        return 0.0
    return float(np.max(np.abs(actual[finite] - expected[finite])))


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        msg = f"must be at least 1, got {value}"
        raise argparse.ArgumentTypeError(msg)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--save", metavar="PATH", help="write the outputs to an .npz")
    group.add_argument("--compare", metavar="PATH", help="compare with a saved .npz")
    parser.add_argument(
        "--repeat", type=_positive_int, default=3, help="runs per timing; the fastest is kept"
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=sorted(CASES),
        default=sorted(CASES),
        help="which cases to run, e.g. '--cases b'",
    )
    args = parser.parse_args()

    outputs = {}
    with warnings.catch_warnings():
        # Diagnostics (e.g. Wilson-factorization convergence) are not timed data.
        warnings.simplefilter("ignore")
        for name in args.cases:
            _, case_outputs = run_case(name, CASES[name], args.repeat)
            outputs.update({f"{name}/{m}": value for m, value in case_outputs.items()})

    if args.save:
        np.savez(args.save, **outputs)
        print(f"\nSaved {len(outputs)} arrays to {args.save}")
    if args.compare:
        with np.load(args.compare) as baseline:
            failed = False
            print(f"\n{'case/measure':<50} {'max abs diff':>14}")
            for key, value in outputs.items():
                if key not in baseline.files:
                    failed = True
                    print(f"{key:<50} {'FAIL: missing':>14}")
                    continue
                difference = max_abs_difference(value, baseline[key])
                failed |= not difference <= TOLERANCE
                print(f"{key:<50} {difference:>14.3e}")
        print(f"\n{'FAIL' if failed else 'PASS'}: tolerance {TOLERANCE:g}")
        return int(failed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
