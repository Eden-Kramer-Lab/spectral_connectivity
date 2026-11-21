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
from a state-space oracle that factors no spectrum. The cross-spectrum that an
example's window and tapers average its model's spectrum to comes from
``tests/_expected_spectrum.py``.

Four things are asserted of each plotted example and of each half of the
time-varying one. They fail for different reasons.

- The factors behind its directed measures reproduce its estimated spectrum,
  to 1e-6: the factor of the whole spectrum, which DTF, PDC, generalized PDC
  and direct DTF come from, and the model of each pair of signals, which
  pairwise Granger comes from. This fails when the factorization is wrong:
  with the lag ``n_fft / 2`` dropped from the causal projection, as before
  3.0, the factors of these examples missed their spectra by 0.07% to 3%,
  and the tutorial showed a spectral Granger peak of 4.35 where the model
  has 3.70.
- Its estimated cross-spectrum is a draw around the spectrum its window and
  tapers average the model's to, with no more than sampling error between
  them. This fails when the simulation or the transform is off by a few
  percent.
- The directed measures of that expected spectrum are near the model's
  (``_BIAS_BUDGET``). No simulated data enter, so this is the bias of the
  example's window and tapers, and it fails when they do not suit its model.
- Its estimates are near the model's (``_ESTIMATE_BUDGET``). This is the
  loosest of the four: it has to cover that bias and the sampling error of a
  directed measure, which nothing here bounds.
"""

import ast
import dataclasses
import itertools
import re
from pathlib import Path

import numpy as np
import pytest

import spectral_connectivity.simulate
from spectral_connectivity import Connectivity, Multitaper
from spectral_connectivity._granger import _var_model_from_spectrum
from spectral_connectivity.transforms import prepare_time_series
from tests._backend_helpers import to_device, to_host
from tests._expected_spectrum import (
    _assert_multitaper_estimates,
    _expected_multitaper_cross_spectrum,
)
from tests._var_oracle import (
    _analytic_directed_measures,
    _analytic_var,
    _fourier_coefficients_with_cross_spectrum,
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


@dataclasses.dataclass(eq=False)
class _Analysis:
    """An example as the tutorial analyses it, with the model its data follow
    in ``windows`` (a mask over the transform's time windows) and the measures
    the tutorial plots of it: by name, or already computed by the tutorial."""

    label: str
    multitaper: Multitaper
    connectivity: Connectivity
    windows: np.ndarray
    measures: tuple | dict
    coefficients: np.ndarray
    noise_covariance: np.ndarray

    def estimate(self, measure):
        """What the tutorial plots of ``measure``, averaged over ``windows``."""
        if isinstance(self.measures, dict):
            values = self.measures[measure]
        else:
            values = getattr(self.connectivity, measure)()
        return values[self.windows].mean(axis=0)

    def true_measures(self):
        """The model's measures on the non-negative FFT grid, ``[frequency, source, target]``."""
        A, H, _ = _analytic_var(
            self.coefficients, self.noise_covariance, self.multitaper.n_fft_samples
        )
        true = {
            name: np.swapaxes(value, -1, -2)
            for name, value in _analytic_directed_measures(A, H, self.noise_covariance).items()
        }
        true["pairwise_spectral_granger_prediction"] = _pairwise_granger_from_state_space(
            self.coefficients, self.noise_covariance, self.multitaper.n_fft_samples
        )
        return true

    def expected_spectrum(self):
        """The cross-spectrum the transform's estimate of the model averages to."""
        return _expected_multitaper_cross_spectrum(
            self.multitaper, self.coefficients, self.noise_covariance
        )


@pytest.fixture(scope="module")
def tutorial():
    return _Tutorial()


def _plotted(tutorial, example):
    """A plotted example, analysed as ``plot_directional`` analyses it."""
    time_series, sampling_frequency, *model = tutorial.simulate(example)
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
    return _Analysis(example, multitaper, connectivity, np.array([True]), _MEASURES, *model)


def _half_of_time_varying(tutorial, example):
    """One half of the last example, which joins a recording in which x2
    drives x1 to one in which x1 drives x2 and analyses it in sliding
    windows. The windows that lie wholly in a half show that half's model."""
    multitaper = tutorial.namespace["m"]
    n_first_half = tutorial.namespace["time_series1"].shape[0]
    window_start = multitaper.n_time_samples_per_step * np.arange(multitaper.time.size)
    if example == "Dhamala_example2a":
        windows = window_start + multitaper.n_time_samples_per_window <= n_first_half
    else:
        windows = window_start >= n_first_half
    assert windows.sum() >= 5
    _, _, *model = tutorial.simulate(example)
    plotted = {
        "pairwise_spectral_granger_prediction": tutorial.namespace["granger"],
        "directed_transfer_function": tutorial.namespace["dtf"],
        "partial_directed_coherence": tutorial.namespace["pdc"],
    }
    return _Analysis(
        f"{example} in the time-varying example",
        multitaper,
        tutorial.namespace["c"],
        windows,
        plotted,
        *model,
    )


_TIME_VARYING_HALVES = ("Dhamala_example2a", "Dhamala_example2b")


@pytest.fixture(scope="module", params=[*_PLOTTED_EXAMPLES, *_TIME_VARYING_HALVES])
def analysis(tutorial, request):
    """Each example in turn, simulated and transformed once for all of its
    tests: pytest runs every test of one example before it builds the next,
    so one example's data is held at a time."""
    if request.param in _TIME_VARYING_HALVES:
        return _half_of_time_varying(tutorial, request.param)
    return _plotted(tutorial, request.param)


# How far a measure may be from the model's, for each influence of one signal
# on another: (fraction of the peak of the model's values, absolute). Where the
# model has no influence, the peak is 0 and the absolute part is the limit.

# Estimates carry sampling error and bias. With the tutorial's seed the furthest
# uses 0.73 of this budget, and 0.82 over 31 seeds of every plotted example.
_ESTIMATE_BUDGET = (0.3, 0.05)
# The measures of an expected spectrum carry bias alone, and these limits are
# the largest the tutorial's settings give, with a margin: 23% of a peak, for
# spectral Granger in the 100-sample windows of the time-varying example, and
# 0.004 where a model has no influence. They hold the tutorial to the bias it
# has now. Wen example 2 with 5 s trials, as it once was, averaged to a partial
# directed coherence of 0.16 between two signals its model does not connect.
_BIAS_BUDGET = (0.25, 0.01)
# Direct DTF is normalized over every frequency, so a bin holds about
# 1 / n_frequencies of its target's inflow and the absolute parts above exceed
# it: an all-zero dDTF passed the estimate budget for 12 of the 14 plotted
# examples. Its sum over frequency, the share of the target's inflow it
# attributes to the source, is held to a budget relative to each influence's
# own share, (relative, absolute) as above. The absolute part is about twice
# the largest share estimated where a model has no direct influence (0.0027).
# With the tutorial's seed the furthest estimate uses 0.53 of its budget and
# the furthest expected spectrum 0.55 of its, while an all-zero dDTF uses 2.3
# or more of both in every example.
_SUMMED_DDTF_ESTIMATE_BUDGET = (0.3, 0.005)
_SUMMED_DDTF_BIAS_BUDGET = (0.25, 0.005)


def _explained_fraction(measure, values):
    """A measure on the scale its estimate is budgeted on, between 0 and 1.

    DTF, PDC, generalized PDC and direct DTF are normalized to that range
    already. Spectral Granger ``G`` is the logarithm of the target's power
    over the part its source does not explain, so the fraction of that power
    the source explains is ``1 - exp(-G)``. A tall Granger peak is the
    logarithm of a small remainder, and one simulation does not determine its
    height: over 40 seeds, the estimate at a model's peak has a standard
    deviation of 10% to 19% of that peak for the least determined influence
    of each example measured (Baccala 2 and 5, Ding 3, Wen 2), and with
    Granger as returned the estimate budget failed in 18 of 434 simulations
    (31 seeds of each plotted example), against none as fractions.

    On this scale a tall peak passes whenever the estimate is also tall (a
    true 4 read as 1.1 passes), so the height that one simulation returns is
    not held to the model here. What is held is what that height comes from:
    each pair's model reproduces its spectrum
    (``test_factors_behind_example_reproduce_its_spectrum``), the Granger of
    the expected spectrum is near the model's (the bias budget), and the
    arithmetic from a model to its Granger is checked on exact spectra in
    ``tests/test_directed_measures_oracle.py``.
    """
    if measure == "pairwise_spectral_granger_prediction":
        return -np.expm1(-values)
    return values


def _assert_near_model(measure, model, budget, label):
    """A measure, ``[frequency, source, target]``, is within ``budget`` of the
    model's for every influence of one signal on another, at every frequency
    above 0 Hz.

    The 0 Hz bin is left out: removing each window's mean distorts it (an
    influence that peaks at 0.1 or more by up to 26% of its peak, against 2%
    without detrending).
    """
    relative, absolute = budget
    off_diagonal = ~np.eye(model.shape[-1], dtype=bool)
    assert np.isfinite(measure[:, off_diagonal]).all(), label
    departure = np.abs(measure - model)[1:, off_diagonal]  # (frequency, influence)
    limit = relative * model[1:, off_diagonal].max(axis=0) + absolute
    frequency, influence = np.unravel_index(np.argmax(departure / limit), departure.shape)
    source, target = np.argwhere(off_diagonal)[influence]
    assert departure[frequency, influence] < limit[influence], (
        f"{label}, x{source + 1} -> x{target + 1}: {departure[frequency, influence]:.3g} "
        f"from the model at frequency bin {frequency + 1} (limit {limit[influence]:.3g})"
    )


def _assert_summed_ddtf_near_model(measure, model, budget, label):
    """Direct DTF, ``[frequency, source, target]``, summed over the
    frequencies above 0 Hz, is within ``budget`` of the model's for every
    influence of one signal on another, relative to that influence's own sum."""
    relative, absolute = budget
    off_diagonal = ~np.eye(model.shape[-1], dtype=bool)
    measure_share = measure[1:, off_diagonal].sum(axis=0)
    model_share = model[1:, off_diagonal].sum(axis=0)
    departure = np.abs(measure_share - model_share)
    limit = relative * model_share + absolute
    influence = np.argmax(departure / limit)
    source, target = np.argwhere(off_diagonal)[influence]
    assert departure[influence] < limit[influence], (
        f"{label}, x{source + 1} -> x{target + 1}: summed direct DTF "
        f"{measure_share[influence]:.3g}, model {model_share[influence]:.3g} "
        f"(limit {limit[influence]:.3g})"
    )


def _assert_reproduces(model, spectrum):
    """``model`` equals ``spectrum``, ``(..., frequency, signal, signal)``, to
    1e-6 of the spectrum's largest entry."""
    scale = np.abs(spectrum).max(axis=(-3, -2, -1))
    residual = np.abs(model - spectrum).max(axis=(-3, -2, -1))
    np.testing.assert_array_less(residual / scale, 1e-6)


def test_factors_behind_example_reproduce_its_spectrum(analysis):
    """The directed measures of an example come from factors of the spectrum
    estimated from its data, on the FFT length the tutorial uses, and each
    reproduces what it factors in every window.

    DTF, PDC, generalized PDC and direct DTF come from one factor ``G`` of the
    whole spectrum: ``G Gᴴ`` equals it. Pairwise Granger does not use that
    factor: it models each pair of signals on its own, by a transfer function
    ``H`` and an innovation covariance ``V``, and ``H V Hᴴ`` equals the pair's
    spectrum."""
    connectivity = analysis.connectivity
    spectrum = to_host(connectivity._expectation_cross_spectral_matrix())
    factor = to_host(connectivity._minimum_phase_factor)
    _assert_reproduces(factor @ factor.conj().swapaxes(-1, -2), spectrum)

    pairs = np.array(list(itertools.combinations(range(spectrum.shape[-1]), 2)))
    # (window, frequency, pair, 2, 2) -> (window, pair, frequency, 2, 2)
    pair_spectra = np.moveaxis(spectrum[..., pairs[:, :, None], pairs[:, None, :]], -3, -4)
    transfer, covariance = (
        to_host(part)
        for part in _var_model_from_spectrum(
            to_device(pair_spectra),
            minimum_phase_tolerance=connectivity._minimum_phase_tolerance,
            minimum_phase_max_iterations=connectivity._minimum_phase_max_iterations,
        )
    )
    n_non_negative = transfer.shape[-3]
    _assert_reproduces(
        transfer @ covariance[..., None, :, :] @ transfer.conj().swapaxes(-1, -2),
        pair_spectra[..., :n_non_negative, :, :],
    )


def test_example_estimates_the_expected_spectrum_of_its_model(analysis):
    """The cross-spectrum the tutorial estimates is, in each window, a draw
    around the one its window and tapers average the model's spectrum to."""
    spectra = to_host(analysis.connectivity._expectation_cross_spectral_matrix())
    expected = analysis.expected_spectrum()
    for window in np.flatnonzero(analysis.windows):
        _assert_multitaper_estimates(
            analysis.multitaper,
            spectra[window],
            expected,
            f"{analysis.label}, window {window}",
        )


def test_window_and_tapers_of_example_suit_its_model(analysis):
    """The directed measures of the spectrum an example's estimate averages
    to are near the model's: the bias of its window and tapers. The measures
    are the package's own, so an error in them smaller than the budget would
    pass as bias; their closed-form tests are what exclude one."""
    expected = Connectivity(
        fourier_coefficients=_fourier_coefficients_with_cross_spectrum(
            analysis.expected_spectrum(), conjugate_symmetric=True
        )
    )
    true = analysis.true_measures()
    for measure in analysis.measures:
        values = getattr(expected, measure)()[0]
        label = f"{analysis.label} {measure}"
        _assert_near_model(values, true[measure], _BIAS_BUDGET, label)
        if measure == "direct_directed_transfer_function":
            _assert_summed_ddtf_near_model(
                values, true[measure], _SUMMED_DDTF_BIAS_BUDGET, label
            )


def test_example_agrees_with_its_model(analysis):
    """What the tutorial plots for an example is the directed connectivity of
    the model it simulates (for the time-varying example, on average over the
    windows of each half)."""
    true = analysis.true_measures()
    for measure in analysis.measures:
        estimate = analysis.estimate(measure)
        label = f"{analysis.label} {measure}"
        _assert_near_model(
            _explained_fraction(measure, estimate),
            _explained_fraction(measure, true[measure]),
            _ESTIMATE_BUDGET,
            label,
        )
        if measure == "direct_directed_transfer_function":
            _assert_summed_ddtf_near_model(
                estimate, true[measure], _SUMMED_DDTF_ESTIMATE_BUDGET, label
            )


def _published_model(n_lags, n_signals, terms, noise_covariance):
    """``(coefficients, noise_covariance)`` from ``(target, source, lag,
    value)`` terms, 1-based as printed; coefficients are ``[lag - 1, target,
    source]`` as ``simulate_MVAR`` takes them."""
    coefficients = np.zeros((n_lags, n_signals, n_signals))
    for target, source, lag, value in terms:
        coefficients[lag - 1, target - 1, source - 1] = value
    return coefficients, np.asarray(noise_covariance, dtype=float)


# The models of Ding, Chen & Bressler (2006), "Granger causality: basic theory
# and application to neuroscience" (arXiv:q-bio/0608035, Eqs. 55-58; Handbook
# of Time Series Analysis, ch. 17), transcribed from the paper rather than the
# tutorial, so a transcription error in the tutorial is not also in its tests.
# x1, x2, x3 of Example 2 are the paper's X, Y, Z.
_ROOT2 = np.sqrt(2)
_EXAMPLE2_TERMS = [
    (1, 1, 1, 0.8),
    (1, 1, 2, -0.5),
    (1, 3, 1, 0.4),
    (2, 2, 1, 0.9),
    (2, 2, 2, -0.8),
    (3, 3, 1, 0.5),
    (3, 3, 2, -0.2),
    (3, 2, 1, 0.5),
]
_PUBLISHED_MODELS = {
    "ding_example1": _published_model(
        2,
        2,
        [
            (1, 1, 1, 0.9),
            (1, 1, 2, -0.5),
            (2, 2, 1, 0.8),
            (2, 2, 2, -0.5),
            (2, 1, 1, 0.16),
            (2, 1, 2, -0.2),
        ],
        [[1.0, 0.4], [0.4, 0.7]],
    ),
    "ding_example2a": _published_model(2, 3, _EXAMPLE2_TERMS, np.diag([0.3, 1.0, 0.2])),
    "ding_example2b": _published_model(
        2, 3, [*_EXAMPLE2_TERMS, (1, 2, 2, 0.2)], np.diag([0.3, 1.0, 0.2])
    ),
    "ding_example3": _published_model(
        3,
        5,
        [
            (1, 1, 1, 0.95 * _ROOT2),
            (1, 1, 2, -0.9025),
            (2, 1, 2, 0.5),
            (3, 1, 3, -0.4),
            (4, 1, 2, -0.5),
            (4, 4, 1, 0.25 * _ROOT2),
            (4, 5, 1, 0.25 * _ROOT2),
            (5, 4, 1, -0.25 * _ROOT2),
            (5, 5, 1, 0.25 * _ROOT2),
        ],
        np.diag([0.6, 0.5, 0.3, 0.3, 0.6]),
    ),
}


@pytest.mark.parametrize("example", list(_PUBLISHED_MODELS))
def test_example_simulates_the_published_model(tutorial, example):
    """The model an example simulates is the published one; every other test
    here takes the model from the tutorial itself."""
    _, _, coefficients, noise_covariance = tutorial.simulate(example)
    published_coefficients, published_noise = _PUBLISHED_MODELS[example]
    np.testing.assert_allclose(coefficients, published_coefficients, rtol=0, atol=1e-12)
    np.testing.assert_allclose(noise_covariance, published_noise, rtol=0, atol=1e-12)


def _displayed_coefficients(example, n_lags, n_signals):
    """Coefficients of the equations the tutorial displays above an example,
    such as ``$$ x_1 = 0.95\\sqrt2x_1(n-1) - 0.9025x_1(n-2) + w_1 $$``."""
    name = example.replace("_example", " Example ")
    heading = f"## {name[0].upper()}{name[1:]}"
    text = _TUTORIAL.read_text()
    cell = text[text.index(heading) :].split("# %%", 1)[0]
    coefficients = np.zeros((n_lags, n_signals, n_signals))
    for target, right_side in re.findall(r"\$\$\s*x_(\d)\s*=(.*?)\$\$", cell):
        for sign, value, root, source, lag in re.findall(
            r"([+-]?)\s*([0-9.]+)(\\sqrt2)?\s*x_(\d)\(n-(\d)\)", right_side
        ):
            coefficient = float(value) * (_ROOT2 if root else 1.0) * (-1 if sign == "-" else 1)
            coefficients[int(lag) - 1, int(target) - 1, int(source) - 1] = coefficient
    return coefficients


@pytest.mark.parametrize("example", list(_PUBLISHED_MODELS))
def test_example_displays_the_published_equations(example):
    """The equations the tutorial shows for an example are the published ones.
    Example 2a showed the lag-two x2 term that only 2b has, and Example 3 the
    x4 -> x5 coefficient with the wrong sign."""
    published_coefficients, _ = _PUBLISHED_MODELS[example]
    displayed = _displayed_coefficients(example, *published_coefficients.shape[:2])
    np.testing.assert_allclose(displayed, published_coefficients, rtol=0, atol=1e-12)
