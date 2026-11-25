"""Oracle tests of ``Multitaper(artifact_handling="missing_data_tapers")``.

Each test compares the missing-data tapers, or the estimates made with them,
with a quantity computed without the package's solver or estimator:

- the definition of the missing-data Slepian sequences (Bronez 1988; Chave
  2019), a dense eigendecomposition of the sinc kernel written out entry by
  entry on the valid sample times;
- the definition of a concentration, the fraction of a taper's energy in
  ``[-W, W]``, by Gauss-Legendre quadrature of its discrete-time Fourier
  transform;
- the gapless limit, SciPy's DPSS;
- the exact expected tapered cross-spectrum of a VAR over a window with
  missing samples, from its autocovariance (the inverse FFT of the analytic
  spectrum) and the valid-sample least-squares detrending written as a
  matrix; it also checks ``tests/_expected_spectrum.py`` with ``valid=``;
- an independent implementation of the missing-data multitaper estimator of
  a window (dense tapers, least-squares detrending, an explicit DFT matrix,
  and Thomson's adaptive iteration written from Percival & Walden 1993,
  eq. 370a), uniform, eigenvalue and adaptive weighting;
- a Monte Carlo of a VAR, replicates on the time-window axis, against that
  exact expectation, five standard errors of the replicate mean.

Tapers of nearly equal concentration are defined only up to a rotation, so
tapers are compared by the projector onto each group's span and estimates
by sums over tapers that do not depend on it.
"""

import warnings

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss
from scipy.linalg import eigh
from scipy.signal.windows import dpss

from spectral_connectivity import Connectivity, Multitaper
from spectral_connectivity.transforms import _missing_data_slepian_tapers
from tests._backend_helpers import to_host
from tests._expected_spectrum import _expected_tapered_cross_spectrum
from tests._var_oracle import _analytic_var

SAMPLING_FREQUENCY = 200.0
THRESHOLD = 0.9  # the low-bias rule: a taper is kept above this concentration
# Concentrations closer than this are one group, whose span alone is compared.
GROUP_GAP = 1e-6


def _dense_slepians(valid, time_halfbandwidth_product):
    """Every eigenpair, in decreasing order, of ``sin(2 pi W (t - s)) / (pi (t -
    s))`` (``2 W`` at ``t = s``) on the valid sample times ``t, s``."""
    half_bandwidth = time_halfbandwidth_product / valid.size
    times = np.flatnonzero(valid).astype(float)
    lag = times[:, None] - times[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        kernel = np.sin(2 * np.pi * half_bandwidth * lag) / (np.pi * lag)
    kernel[lag == 0] = 2 * half_bandwidth
    eigenvalues, eigenvectors = eigh(kernel)
    return eigenvalues[::-1], eigenvectors[:, ::-1]


def _expected_n_kept(eigenvalues, n_tapers):
    """The tapers the low-bias rule keeps of the ``n_tapers`` most
    concentrated, 0 if fewer than 2."""
    n_kept = int(np.count_nonzero(eigenvalues[:n_tapers] > THRESHOLD))
    return n_kept if n_kept >= 2 else 0


def _groups(eigenvalues):
    """Slices of decreasing ``eigenvalues`` closer than ``GROUP_GAP`` in a chain."""
    starts = np.flatnonzero(np.concatenate([[True], -np.diff(eigenvalues) > GROUP_GAP]))
    return [slice(a, b) for a, b in zip(starts, [*starts[1:], eigenvalues.size], strict=True)]


def _assert_same_spans(tapers, eigenvalues, eigenvectors, atol):
    """``tapers`` ``(n_valid, k)`` span, group by group, the eigenspaces of
    the ``k`` largest ``eigenvalues``; a group cut by the ``k``-th lies in the
    whole group's eigenspace."""
    k = tapers.shape[1]
    np.testing.assert_allclose(tapers.T @ tapers, np.eye(k), atol=atol)
    for group in _groups(eigenvalues):
        if group.start >= k:
            break
        span = eigenvectors[:, group]
        members = tapers[:, group.start : min(group.stop, k)]
        # Each taper lies in the group's eigenspace ...
        np.testing.assert_allclose(span @ (span.T @ members), members, atol=atol)
        if group.stop <= k:
            # ... and the group's tapers span all of it.
            np.testing.assert_allclose(members @ members.T, span @ span.T, atol=atol)


def _gaps(n_time, rng, max_gaps=3, max_fraction=0.125):
    """``True`` but at 1 to ``max_gaps`` random gaps of random length."""
    valid = np.ones(n_time, dtype=bool)
    for _ in range(rng.integers(1, max_gaps + 1)):
        length = int(rng.integers(1, max(2, int(max_fraction * n_time))))
        start = int(rng.integers(0, n_time - length + 1))
        valid[start : start + length] = False
    return valid


# (n_time_samples_per_window, NW): short windows, the dense fallback's
# neighborhood, and long ones where the leading concentrations coincide (NW = 8).
SOLVER_CASES = [(64, 3), (96, 2.5), (300, 4), (1000, 2), (1000, 8)]


@pytest.mark.parametrize(("n_time", "nw"), SOLVER_CASES)
def test_the_tapers_are_the_dense_eigenvectors_of_the_restricted_sinc_kernel(n_time, nw):
    """Missing-data Slepian sequences, by definition: concentrations and spans
    of the leading eigenpairs of the sinc kernel on the valid samples, over
    random gap patterns."""
    rng = np.random.default_rng(int(10 * nw) + n_time)
    n_tapers = int(np.floor(2 * nw)) - 1
    orientation = dpss(n_time, nw, n_tapers).T
    for _ in range(6):
        valid = _gaps(n_time, rng)
        tapers, concentrations = _missing_data_slepian_tapers(valid, nw, n_tapers, orientation)
        eigenvalues, eigenvectors = _dense_slepians(valid, nw)
        np.testing.assert_allclose(concentrations, eigenvalues[:n_tapers], atol=1e-12)
        assert (tapers[~valid] == 0).all()
        _assert_same_spans(tapers[valid], eigenvalues, eigenvectors, atol=1e-8)


def _recording_artifacts(n_time, n_windows, n_trials, rng):
    """Artifacts, ``(n_windows * n_time, n_trials)``: every trial window with
    random gaps except a clean one per window, with repeated patterns
    across windows and trials and some patterns that are dropped."""
    artifacts = np.zeros((n_windows * n_time, n_trials), dtype=bool)
    repeated = ~_gaps(n_time, rng)
    for window in range(n_windows):
        samples = slice(window * n_time, (window + 1) * n_time)
        for trial in range(1, n_trials):
            kind = (window + trial) % 4
            if kind == 0:
                artifacts[samples, trial] = repeated
            elif kind == 1:
                artifacts[samples, trial][::3] = True  # dropped: gaps every 3 samples
            else:
                artifacts[samples, trial] = ~_gaps(n_time, rng)
    return artifacts


@pytest.fixture(scope="module")
def gapped_recording():
    """Noise of 2 signals, 4 non-overlapping 96-sample windows, 5 trials."""
    n_time, n_windows, n_trials = 96, 4, 5
    rng = np.random.default_rng(2024)
    artifacts = _recording_artifacts(n_time, n_windows, n_trials, rng)
    time_series = rng.standard_normal((*artifacts.shape, 2))
    time_series[:, :, 1] += 0.5 * np.roll(time_series[:, :, 0], 3, axis=0)
    time_series += 0.01 * np.arange(len(time_series))[:, None, None]  # a trend
    return time_series, artifacts, n_time


def _transform(time_series, artifacts, n_time, nw=3, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return Multitaper(
            time_series,
            SAMPLING_FREQUENCY,
            n_time_samples_per_window=n_time,
            time_halfbandwidth_product=nw,
            artifact_samples=artifacts,
            artifact_handling="missing_data_tapers",
            **kwargs,
        )


def _windows(array, n_time):
    """``(n_windows * n_time, n_trials, ...)`` -> ``(n_windows, n_trials, n_time, ...)``."""
    return np.moveaxis(array.reshape(-1, n_time, *array.shape[1:]), 1, 2)


def test_each_trial_window_uses_its_own_patterns_tapers(gapped_recording):
    """Through the transform: each trial window keeps the tapers the dense
    oracle says its own gap pattern keeps, and its row of the taper table
    holds that pattern's tapers (catching a pattern given another's)."""
    time_series, artifacts, n_time = gapped_recording
    multitaper = _transform(time_series, artifacts, n_time)
    n_tapers = multitaper.n_tapers
    n_kept = to_host(multitaper.n_kept_tapers)
    # Each trial window's row of the transform's taper table, from which the
    # FFT takes the tapers of the windows it rescues.
    missing_data = multitaper._missing_data
    window_tapers = to_host(missing_data.tapers[missing_data.table_index])
    window_tapers = window_tapers / np.sqrt(SAMPLING_FREQUENCY)
    reference = dpss(n_time, 3, n_tapers).T
    patterns = _windows(artifacts, n_time)
    n_checked = 0
    for window, trial in np.ndindex(n_kept.shape):
        valid = ~patterns[window, trial]
        tapers = window_tapers[window, trial]
        if valid.all():
            assert n_kept[window, trial] == n_tapers
            np.testing.assert_allclose(tapers, reference, atol=1e-12)
            continue
        eigenvalues, eigenvectors = _dense_slepians(valid, 3)
        expected = _expected_n_kept(eigenvalues, n_tapers)
        assert n_kept[window, trial] == expected, (window, trial)
        if expected:
            assert (tapers[:, expected:] == 0).all()
            assert (tapers[~valid] == 0).all()
            _assert_same_spans(tapers[valid, :expected], eigenvalues, eigenvectors, atol=1e-8)
            n_checked += 1
    assert n_checked >= 6
    assert (n_kept == 0).sum() >= 3  # some patterns are dropped


def _band_energy_fraction(taper, half_bandwidth, n_nodes=512):
    """``int_{-W}^{W} |V(f)|^2 df / sum v^2`` by Gauss-Legendre quadrature of
    the taper's discrete-time Fourier transform ``V``."""
    nodes, weights = leggauss(n_nodes)
    frequencies = half_bandwidth * nodes
    transform = np.exp(-2j * np.pi * frequencies[:, None] * np.arange(taper.size)) @ taper
    return half_bandwidth * np.sum(weights * np.abs(transform) ** 2) / np.sum(taper**2)


@pytest.mark.parametrize(("n_time", "nw"), [(64, 3), (300, 4), (1000, 8)])
def test_each_concentration_is_its_tapers_energy_fraction_in_the_band(n_time, nw):
    """A missing-data taper's concentration is the fraction of its energy in
    ``[-W, W]``: the definition, integrated numerically. Within a group of
    nearly equal concentrations the tapers are rotated toward the DPSS, so
    each one's own fraction is only within the group's spread of the
    reported eigenvalue (7e-7 at NW = 8), while the group's sum is exact."""
    rng = np.random.default_rng(n_time)
    n_tapers = int(np.floor(2 * nw)) - 1
    for _ in range(3):
        valid = _gaps(n_time, rng)
        tapers, concentrations = _missing_data_slepian_tapers(
            valid, nw, n_tapers, dpss(n_time, nw, n_tapers).T
        )
        integrated = np.array([_band_energy_fraction(t, nw / n_time) for t in tapers.T])
        eigenvalues, _ = _dense_slepians(valid, nw)
        for group in _groups(eigenvalues):
            if group.start >= n_tapers:
                break
            members = slice(group.start, min(group.stop, n_tapers))
            spread = eigenvalues[group.start] - eigenvalues[group.stop - 1]
            np.testing.assert_allclose(
                integrated[members], concentrations[members], atol=spread + 1e-10
            )
            if group.stop <= n_tapers:
                np.testing.assert_allclose(
                    integrated[members].sum(), concentrations[members].sum(), atol=1e-10
                )


@pytest.mark.parametrize("n_time", [64, 65, 513, 1000])
@pytest.mark.parametrize("nw", [1.5, 3, 4, 8])
def test_without_gaps_they_are_scipys_dpss(n_time, nw):
    """The no-gap limit: SciPy's DPSS and its concentration ratios."""
    n_tapers = int(np.floor(2 * nw)) - 1
    reference, ratios = dpss(n_time, nw, n_tapers, return_ratios=True)
    tapers, concentrations = _missing_data_slepian_tapers(
        np.ones(n_time, dtype=bool), nw, n_tapers, reference.T
    )
    np.testing.assert_allclose(concentrations, ratios, atol=1e-12)
    np.testing.assert_allclose(tapers, reference.T, atol=1e-8)


# A VAR of 2 signals; x(t) = sum_k A_k x(t - k - 1) + e(t). Its spectrum spans
# a factor of about 30.
VAR_COEFFICIENTS = np.array([[[0.5, 0.0], [0.4, 0.3]], [[-0.3, 0.0], [0.0, -0.2]]])
VAR_NOISE = np.array([[1.0, 0.3], [0.3, 1.0]])


def _var_autocovariance_from_spectrum(max_lag, n_fft=8192):
    """``Gamma(h) = E[x(t + h) x(t)^T]``, the inverse FFT of the analytic
    spectrum ``S(w) = sum_h Gamma(h) exp(-i w h)``, ``h = 0 ... max_lag``."""
    _, _, spectrum = _analytic_var(VAR_COEFFICIENTS, VAR_NOISE, n_fft)
    return np.fft.ifft(spectrum, axis=0).real[: max_lag + 1]


def _detrending_matrix(valid, detrend_type):
    """``M`` ``(n_valid, n_valid)``: the valid samples' residuals from their
    least-squares constant or line are ``M @ x[valid]``."""
    times = np.flatnonzero(valid).astype(float)
    columns = {
        None: [],
        "constant": [np.ones_like(times)],
        "linear": [np.ones_like(times), times],
    }
    basis = np.array(columns[detrend_type]).reshape(-1, times.size).T
    if basis.shape[1] == 0:
        return np.eye(times.size)
    return np.eye(times.size) - basis @ np.linalg.solve(basis.T @ basis, basis.T)


def _brute_force_expected_cross_spectrum(valid, tapers, n_fft, detrend_type):
    """``E[J(f) J(f)^H]`` averaged over ``tapers`` ``(n_time, k)``, ``(n_fft,
    n_signals, n_signals)``: with ``J = sum_s b(s) x(s)``, ``b = M^T a`` and
    ``a(t) = taper(t) exp(-2 pi i f t)`` on the valid times, it is ``sum_{s,
    u} b(s) conj(b(u)) Gamma(s - u)``."""
    times = np.flatnonzero(valid)
    gamma = _var_autocovariance_from_spectrum(valid.size)
    lag = times[:, None] - times[None, :]
    covariance = np.where(
        (lag >= 0)[..., None, None],
        gamma[np.abs(lag)],
        gamma[np.abs(lag)].swapaxes(-1, -2),
    )  # (n_valid, n_valid, n_signals, n_signals): E[x(s) x(u)^T]
    detrending = _detrending_matrix(valid, detrend_type)
    phases = np.exp(-2j * np.pi * np.arange(n_fft)[:, None] * times[None, :] / n_fft)
    expected = 0
    for taper in tapers.T:
        b = (phases * taper[times]) @ detrending  # (n_fft, n_valid): (M^T a)^T
        expected = expected + np.einsum("fs,fu,suab->fab", b, b.conj(), covariance)
    return expected / tapers.shape[1]


@pytest.mark.parametrize("detrend_type", [None, "constant", "linear"])
def test_the_expected_spectrum_with_valid_samples_is_the_brute_force_one(detrend_type):
    """``tests/_expected_spectrum.py`` with ``valid=`` against the expectation
    written as a sum over pairs of valid samples, for the missing-data
    tapers of a window with two gaps."""
    n_time, nw = 64, 3
    valid = np.ones(n_time, dtype=bool)
    valid[[10, 11, 12, 40, 41, 42, 43]] = False
    eigenvalues, eigenvectors = _dense_slepians(valid, nw)
    n_kept = _expected_n_kept(eigenvalues, 5)
    tapers = np.zeros((n_time, n_kept))
    tapers[valid] = eigenvectors[:, :n_kept]
    expected = _expected_tapered_cross_spectrum(
        VAR_COEFFICIENTS, VAR_NOISE, tapers, n_time, detrend_type, valid=valid
    )
    brute_force = _brute_force_expected_cross_spectrum(valid, tapers, n_time, detrend_type)
    np.testing.assert_allclose(expected, brute_force, atol=1e-10 * np.abs(brute_force).max())


def _independent_estimate(window, valid, nw, n_tapers, n_fft, detrend_type, weighting):
    """The missing-data multitaper cross-spectrum of one trial window,
    ``window`` ``(n_time, n_signals)``, in per-sample units, ``(n_fft,
    n_signals, n_signals)``, or ``None`` if the window is dropped.

    The dense tapers kept by the low-bias rule; the valid samples' residuals
    from their least-squares fit; eigencoefficients ``J_k(f) = sum_t v_k(t)
    r(t) exp(-2 pi i f t)`` by an explicit DFT matrix; uniform (``mean_k J_k
    J_k^H``), eigenvalue (``sum_k l_k J_k J_k^H / sum_k l_k``) or adaptive
    weighting. Adaptive weighting is the diagonal only (Thomson's weights are
    per signal), iterated to a fixed point from the mean of the first two
    eigenspectra: ``d_k = sqrt(l_k) S / (l_k S + (1 - l_k) s2)``, ``S = sum_k
    d_k^2 |J_k|^2 / sum_k d_k^2``, ``s2`` the residuals' variance.
    """
    times = np.flatnonzero(valid)
    eigenvalues, eigenvectors = _dense_slepians(valid, nw)
    n_kept = _expected_n_kept(eigenvalues, n_tapers)
    if n_kept == 0:
        return None
    concentrations = eigenvalues[:n_kept]
    residuals = _detrending_matrix(valid, detrend_type) @ window[times]
    dft = np.exp(-2j * np.pi * np.arange(n_fft)[:, None] * times[None, :] / n_fft)
    eigencoefficients = np.einsum(
        "ft,tk,ts->kfs", dft, eigenvectors[:, :n_kept], residuals
    )  # (n_kept, n_fft, n_signals)
    outer = np.einsum("kfa,kfb->kfab", eigencoefficients, eigencoefficients.conj())
    if weighting == "uniform":
        return outer.mean(axis=0)
    if weighting == "eigen":
        return np.einsum("k,kfab->fab", concentrations, outer) / concentrations.sum()
    power = np.abs(eigencoefficients) ** 2  # (n_kept, n_fft, n_signals)
    variance = np.mean(residuals**2, axis=0) - np.mean(residuals, axis=0) ** 2
    lam = concentrations[:, None, None]
    spectrum = power[:2].mean(axis=0)
    for _ in range(500):
        d = np.sqrt(lam) * spectrum / (lam * spectrum + (1 - lam) * variance)
        spectrum = np.sum(d**2 * power, axis=0) / np.sum(d**2, axis=0)
    return spectrum


def _package_estimate(coefficients, n_kept, weighting):
    """A trial window's estimate from its coefficients ``(n_tapers, n_fft,
    n_signals)`` (taper weights applied), in per-sample units."""
    kept = coefficients[:n_kept] * np.sqrt(SAMPLING_FREQUENCY)
    if weighting == "adaptive":
        return np.mean(np.abs(kept) ** 2, axis=0)
    return np.einsum("kfa,kfb->fab", kept, kept.conj()) / n_kept


@pytest.mark.parametrize("detrend_type", ["constant", "linear"])
@pytest.mark.parametrize("weighting", ["uniform", "eigen", "adaptive"])
def test_each_window_is_the_independent_missing_data_estimate(
    gapped_recording, weighting, detrend_type
):
    """Every trial window's estimate, with or without gaps, is that of the
    independent implementation of the estimator (Bronez 1988; Chave 2019)."""
    time_series, artifacts, n_time = gapped_recording
    multitaper = _transform(
        time_series,
        artifacts,
        n_time,
        taper_weighting=weighting,
        detrend_type=detrend_type,
        adaptive_tolerance=1e-12,
        adaptive_max_iterations=500,
    )
    coefficients = to_host(multitaper.fft())
    n_kept = to_host(multitaper.n_kept_tapers)
    windows = _windows(time_series, n_time)
    patterns = _windows(artifacts, n_time)
    rtol = 1e-6 if weighting == "adaptive" else 1e-10
    n_compared = 0
    for window, trial in np.ndindex(n_kept.shape):
        valid = ~patterns[window, trial]
        expected = _independent_estimate(
            np.nan_to_num(windows[window, trial]),
            valid,
            3,
            multitaper.n_tapers,
            multitaper.n_fft_samples,
            detrend_type,
            weighting,
        )
        if expected is None:
            assert n_kept[window, trial] == 0
            continue
        estimate = _package_estimate(
            coefficients[window, trial], n_kept[window, trial], weighting
        )
        np.testing.assert_allclose(
            estimate, expected, atol=rtol * np.abs(expected).max(), err_msg=f"{window, trial}"
        )
        n_compared += 1
    assert n_compared >= 12


def test_the_power_averages_the_kept_tapers_of_the_kept_trial_windows(gapped_recording):
    """``Connectivity.power``: each window's power is the mean over its kept
    trial windows and their kept tapers of the independent eigenspectra."""
    time_series, artifacts, n_time = gapped_recording
    multitaper = _transform(time_series, artifacts, n_time)
    power = Connectivity.from_transform(multitaper).power(one_sided=False)
    windows = _windows(time_series, n_time)
    patterns = _windows(artifacts, n_time)
    for window in range(power.shape[0]):
        total, count = 0, 0
        for trial in range(windows.shape[1]):
            valid = ~patterns[window, trial]
            eigenvalues, _ = _dense_slepians(valid, 3)
            n_kept = _expected_n_kept(eigenvalues, multitaper.n_tapers)
            estimate = _independent_estimate(
                np.nan_to_num(windows[window, trial]),
                valid,
                3,
                multitaper.n_tapers,
                multitaper.n_fft_samples,
                "constant",
                "uniform",
            )
            if estimate is None:
                continue
            total = total + n_kept * np.einsum("faa->fa", estimate).real
            count += n_kept
        # The non-negative frequencies, undoubled with one_sided=False.
        expected = total[: power.shape[1]] / count / SAMPLING_FREQUENCY
        np.testing.assert_allclose(power[window], expected, rtol=1e-10)


def _simulate_var(n_segments, n_time, rng, burn_in=200):
    """``n_segments`` independent stationary stretches of the VAR, ``(n_segments,
    n_time, n_signals)``."""
    n_signals = VAR_NOISE.shape[0]
    noise = (
        rng.standard_normal((n_segments, burn_in + n_time, n_signals))
        @ np.linalg.cholesky(VAR_NOISE).T
    )
    x = np.zeros_like(noise)
    for t in range(burn_in + n_time):
        x[:, t] = noise[:, t]
        for lag, coefficient in enumerate(VAR_COEFFICIENTS, start=1):
            if t >= lag:
                x[:, t] += x[:, t - lag] @ coefficient.T
    return x[:, burn_in:]


MC_GAPS = {1: [(20, 4)], 2: [(10, 3), (45, 3)], 3: [(26, 12)], 4: [(0, 8)]}


def test_the_cross_spectrum_averages_to_its_exact_expectation():
    """Monte Carlo: 400 independent replicate windows of 5 trials of a VAR,
    one clean and four gapped (each trial's pattern the same in every
    window). Per window, the trial-averaged deviation of the power and of the
    real and imaginary cross-spectrum from the exact expectation (brute-force
    covariance and dense tapers), relative to the expected powers; the
    replicate mean is within 5 standard errors of 0 for each."""
    n_time, nw, n_replicates, n_trials = 64, 3, 400, 5
    rng = np.random.default_rng(7)
    segments = _simulate_var(n_replicates * n_trials, n_time, rng)
    time_series = (
        segments.reshape(n_replicates, n_trials, n_time, 2)
        .swapaxes(1, 2)
        .reshape(n_replicates * n_time, n_trials, 2)
    )
    valid = np.ones((n_trials, n_time), dtype=bool)
    for trial, gaps in MC_GAPS.items():
        for start, length in gaps:
            valid[trial, start : start + length] = False
    artifacts = np.tile(~valid.T, (n_replicates, 1))
    multitaper = _transform(time_series, artifacts, n_time, nw=nw)
    coefficients = to_host(multitaper.fft()) * np.sqrt(SAMPLING_FREQUENCY)
    n_kept = to_host(multitaper.n_kept_tapers)
    frequencies = np.arange(3, 29)  # away from 0 Hz, which detrending removes
    deviations = np.zeros((n_replicates, 4))
    for trial in range(n_trials):
        eigenvalues, eigenvectors = _dense_slepians(valid[trial], nw)
        kept = _expected_n_kept(eigenvalues, 5)
        assert (n_kept[:, trial] == kept).all()
        tapers = np.zeros((n_time, kept))
        tapers[valid[trial]] = eigenvectors[:, :kept]
        expected = _brute_force_expected_cross_spectrum(
            valid[trial], tapers, n_time, "constant"
        )[frequencies]
        c = coefficients[:, trial, :kept][:, :, frequencies]  # (n_replicates, k, f, 2)
        estimate = np.einsum("rkfa,rkfb->rfab", c, c.conj()) / kept
        scale = np.sqrt(expected[:, 0, 0].real * expected[:, 1, 1].real)
        deviation = (estimate - expected) / np.array(
            [[expected[:, 0, 0].real, scale], [scale, expected[:, 1, 1].real]]
        ).transpose(2, 0, 1)
        deviations += (
            np.stack(
                [
                    deviation[:, :, 0, 0].real.mean(axis=1),
                    deviation[:, :, 1, 1].real.mean(axis=1),
                    deviation[:, :, 0, 1].real.mean(axis=1),
                    deviation[:, :, 0, 1].imag.mean(axis=1),
                ],
                axis=1,
            )
            / n_trials
        )
    mean = deviations.mean(axis=0)
    limit = 5 * deviations.std(axis=0, ddof=1) / np.sqrt(n_replicates)
    assert (np.abs(mean) < limit).all(), (mean, limit)


def test_the_tapers_do_not_depend_on_the_lanczos_start_vector(monkeypatch):
    """Determinism: another start vector gives the same tapers, as the span of
    each group of nearly equal concentrations is rotated to the DPSS."""
    from spectral_connectivity import transforms

    n_time, nw, n_tapers = 1000, 8, 15
    reference = dpss(n_time, nw, n_tapers).T
    rng = np.random.default_rng(3)
    patterns = [_gaps(n_time, rng) for _ in range(4)]
    first = [_missing_data_slepian_tapers(v, nw, n_tapers, reference) for v in patterns]
    original = np.random.default_rng

    def other_start(seed=None):
        # Only the package's fixed integer seed is changed.
        return original(seed + 99 if isinstance(seed, int) else seed)

    monkeypatch.setattr(transforms.np.random, "default_rng", other_start)
    second = [_missing_data_slepian_tapers(v, nw, n_tapers, reference) for v in patterns]
    monkeypatch.undo()
    for (tapers_a, concentrations_a), (tapers_b, concentrations_b) in zip(
        first, second, strict=True
    ):
        np.testing.assert_allclose(concentrations_a, concentrations_b, atol=1e-12)
        np.testing.assert_allclose(tapers_a, tapers_b, atol=1e-8)
