"""The paper-examples tutorial against the models it simulates.

Each example in ``examples/Tutorial_Using_Paper_Examples.py`` simulates a
published vector-autoregressive model and plots directed measures of it. The
models' true values are known, so the tutorial's answers can be checked
against them.

The examples and their settings are taken from the tutorial itself: its
imports, function definitions, and the top-level assignments that do not plot
are executed (see ``_Tutorial``), so a change to an example changes what
is tested. The only departure is a cap on the number of trials, which bounds
memory; it affects one example.

True values come from ``tests/_var_oracle.py``: DTF, PDC, generalized PDC, and
direct DTF in closed form from the coefficients, and pairwise spectral Granger
from a state-space oracle that factors no spectrum.

Two things are asserted of each example.

The factor behind its directed measures reproduces its estimated spectrum.
This is exact (1e-6), and it is what fails when the factorization is wrong:
with the lag ``n_fft / 2`` dropped from the causal projection, as before 3.0,
the factors of these examples missed their spectra by 0.07% to 3%, and the
tutorial showed a spectral Granger peak of 4.35 where the model has 3.70.

Its estimates are near the model's values (``_assert_within_budget``). These
are budgets, not tight tolerances, because the estimates carry sampling error
and the bias of their window and tapers: the examples with 100-sample windows
read sharp peaks 15-25% low. They fail when an example's settings do not suit
its model: with 5 s trials, the Wen example 2 of this tutorial gave a partial
directed coherence of 0.19 between two signals its model does not connect.
They would not notice a distortion of 10-20%; the first assertion is there
for the defect of that size this tutorial has had.
"""

import ast
import itertools
from pathlib import Path

import numpy as np
import pytest

import spectral_connectivity.simulate
from spectral_connectivity import Connectivity, Multitaper
from spectral_connectivity.transforms import prepare_time_series
from tests._backend_helpers import to_host
from tests._var_oracle import (
    _analytic_directed_measures,
    _analytic_var,
    _pairwise_granger_from_state_space,
)

_TUTORIAL = (
    Path(__file__).resolve().parents[1] / "examples" / "Tutorial_Using_Paper_Examples.py"
)
# Dhamala example 1 simulates 4000 trials, which take 3.7 GB to analyse; every
# other example has at most this many.
_MAX_TRIALS = 1000
# The measures the tutorial plots, by the name of the Connectivity method.
_MEASURES = (
    "pairwise_spectral_granger_prediction",
    "directed_transfer_function",
    "partial_directed_coherence",
    "generalized_partial_directed_coherence",
    "direct_directed_transfer_function",
)


def _mentions(node, name):
    return any(isinstance(n, ast.Name) and n.id == name for n in ast.walk(node))


def _tutorial_statements():
    """The tutorial's statements that define or compute something, and the
    keyword settings of each ``plot_directional(*example(), ...)`` call."""
    kept, settings = [], {}
    for node in ast.parse(_TUTORIAL.read_text()).body:
        if isinstance(node, ast.FunctionDef):
            kept.append(node)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            if "matplotlib" not in ast.unparse(node):
                kept.append(node)
        elif isinstance(node, ast.Assign):
            if not _mentions(node, "plt"):
                kept.append(node)
        elif isinstance(node.value, ast.Call) and _mentions(
            node.value.func, "plot_directional"
        ):
            example = node.value.args[0].value.func.id
            settings[example] = {
                keyword.arg: ast.literal_eval(keyword.value) for keyword in node.value.keywords
            }
    return kept, settings


_PLOTTED_EXAMPLES = _tutorial_statements()[1]


class _Tutorial:
    """The tutorial's namespace, with each simulation's model recorded."""

    def __init__(self):
        statements, self.settings = _tutorial_statements()
        simulate = spectral_connectivity.simulate.simulate_MVAR

        def recording_simulate(coefficients, **kwargs):
            noise_covariance = kwargs.get("noise_covariance")
            if noise_covariance is None:
                noise_covariance = np.eye(np.shape(coefficients)[-1])
            self._model = (
                np.asarray(coefficients, float),
                np.asarray(noise_covariance, float),
            )
            kwargs["n_trials"] = min(kwargs["n_trials"], _MAX_TRIALS)
            return simulate(coefficients, **kwargs)

        self.namespace = {}
        # The tutorial imports the simulator by name, so it binds this one.
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(spectral_connectivity.simulate, "simulate_MVAR", recording_simulate)
            code = compile(
                ast.Module(body=statements, type_ignores=[]), str(_TUTORIAL), "exec"
            )
            exec(code, self.namespace)

    def simulate(self, example):
        """``(time_series, sampling_frequency, coefficients, noise_covariance)``."""
        time_series, sampling_frequency = self.namespace[example]()
        return time_series, sampling_frequency, *self._model


@pytest.fixture(scope="module")
def tutorial():
    return _Tutorial()


def _true_measures(coefficients, noise_covariance, n_fft):
    """The model's measures on the non-negative FFT grid, ``[frequency, source, target]``."""
    A, H, _ = _analytic_var(coefficients, noise_covariance, n_fft)
    true = {
        name: np.swapaxes(value, -1, -2)
        for name, value in _analytic_directed_measures(A, H, noise_covariance).items()
    }
    true["pairwise_spectral_granger_prediction"] = _pairwise_granger_from_state_space(
        coefficients, noise_covariance, n_fft
    )
    return true


def _assert_within_budget(estimate, true, label):
    """An estimate, ``[frequency, source, target]``, agrees with the model.

    Where the model has no influence from one signal on another, the estimate
    stays below 0.05 at every frequency (the largest in these examples is
    0.02). Where it has one whose peak is at least 0.1, the estimate at that
    peak is within 30% of it, plus 0.05 (the furthest is the time-varying
    example, whose 100-sample windows read a Granger peak 23% low). The 0 Hz
    bin is left out of the peak: it is the least reliable bin, off by 20-50%
    in several examples.
    """
    n_signals = true.shape[-1]
    off_diagonal = ~np.eye(n_signals, dtype=bool)
    assert np.isfinite(estimate[:, off_diagonal]).all(), label
    for source, target in itertools.permutations(range(n_signals), 2):
        pair = f"{label}, x{source + 1} -> x{target + 1}"
        expected, measured = true[:, source, target], estimate[:, source, target]
        if expected.max() < 1e-8:
            assert measured.max() < 0.05, f"{pair}: the model has no such influence"
            continue
        peak = 1 + int(np.argmax(expected[1:]))
        if expected[peak] >= 0.1:
            np.testing.assert_allclose(
                measured[peak], expected[peak], rtol=0.3, atol=0.05, err_msg=pair
            )


def _assert_factor_reproduces_spectrum(connectivity):
    """``G Gᴴ`` equals the estimated cross-spectrum the directed measures use."""
    spectrum = to_host(connectivity._expectation_cross_spectral_matrix())
    factor = to_host(connectivity._minimum_phase_factor)
    residual = factor @ factor.conj().swapaxes(-1, -2) - spectrum
    scale = np.abs(spectrum).max(axis=(-3, -2, -1))
    np.testing.assert_array_less(np.abs(residual).max(axis=(-3, -2, -1)) / scale, 1e-6)


def _analyse_as_plotted(tutorial, example):
    """The example's ``Connectivity``, built as ``plot_directional`` builds it,
    and its model's true measures on the same frequencies."""
    time_series, sampling_frequency, coefficients, noise_covariance = tutorial.simulate(
        example
    )
    multitaper = Multitaper(
        prepare_time_series(time_series),
        sampling_frequency=sampling_frequency,
        start_time=0,
        **{"time_halfbandwidth_product": 2, **tutorial.settings[example]},
    )
    connectivity = Connectivity(
        fourier_coefficients=multitaper.fft(),
        frequencies=multitaper.frequencies,
        time=multitaper.time,
    )
    true = _true_measures(coefficients, noise_covariance, multitaper.n_fft_samples)
    return connectivity, true


@pytest.mark.parametrize("example", list(_PLOTTED_EXAMPLES))
def test_factor_behind_plotted_example_reproduces_its_spectrum(tutorial, example):
    """The directed measures of an example come from a factor of the spectrum
    estimated from its data, on the FFT length the tutorial uses."""
    connectivity, _ = _analyse_as_plotted(tutorial, example)
    _assert_factor_reproduces_spectrum(connectivity)


@pytest.mark.parametrize("example", list(_PLOTTED_EXAMPLES))
def test_plotted_example_agrees_with_its_model(tutorial, example):
    """What the tutorial plots for an example, computed as ``plot_directional``
    computes it, is the directed connectivity of the model it simulates."""
    connectivity, true = _analyse_as_plotted(tutorial, example)
    for measure in _MEASURES:
        estimate = getattr(connectivity, measure)()[0]
        _assert_within_budget(estimate, true[measure], f"{example} {measure}")


def test_time_varying_example_reverses_its_direction(tutorial):
    """The last example joins a recording in which x2 drives x1 to one in
    which x1 drives x2 and analyses it in sliding windows. The windows that lie
    wholly in each half show that half's model."""
    connectivity = tutorial.namespace["c"]
    multitaper = tutorial.namespace["m"]
    _assert_factor_reproduces_spectrum(connectivity)
    half_window = multitaper.time_window_duration / 2
    switch = tutorial.namespace["time_series1"].shape[0] / multitaper.sampling_frequency
    estimates = {
        "pairwise_spectral_granger_prediction": tutorial.namespace["granger"],
        "directed_transfer_function": tutorial.namespace["dtf"],
        "partial_directed_coherence": tutorial.namespace["pdc"],
    }
    halves = {
        "Dhamala_example2a": connectivity.time + half_window <= switch,
        "Dhamala_example2b": connectivity.time - half_window >= switch,
    }
    for example, windows in halves.items():
        assert windows.sum() >= 5
        _, _, coefficients, noise_covariance = tutorial.simulate(example)
        true = _true_measures(coefficients, noise_covariance, multitaper.n_fft_samples)
        for measure, estimate in estimates.items():
            _assert_within_budget(
                estimate[windows].mean(axis=0), true[measure], f"{example} {measure}"
            )
