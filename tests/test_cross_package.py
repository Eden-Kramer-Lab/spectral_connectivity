"""Cross-package agreement: the same estimator computed by independent code.

nitime is a dev dependency and is called directly. mne-connectivity is not a
dependency; its outputs are recorded once in
``tests/reference/mne_connectivity_reference.npz`` by the generator script
beside it.
"""

from pathlib import Path

import numpy as np
import pytest
from nitime.algorithms import coherence as nitime_coherence
from nitime.algorithms import multi_taper_csd
from scipy.signal.windows import dpss

from spectral_connectivity import Connectivity, Multitaper
from spectral_connectivity.simulate import simulate_lagged_broadband

_NITIME_SAMPLING_FREQUENCY = 200.0
_NITIME_NW = 3


@pytest.fixture(scope="module")
def nitime_series():
    """One window of three coherent broadband signals, shape (n_time, n_signals).

    500 samples (even, so the last one-sided bin is Nyquist). Signals 1 and 2
    are noisy copies of signal 0 delayed by 2 and 5 samples.
    """
    return simulate_lagged_broadband((0, 2, 5), 1.0, 500, random_state=0)


def test_coherence_matches_nitime(nitime_series):
    """Coherence and one-sided CSD equal nitime's non-adaptive multitaper estimate.

    nitime's ``multi_taper_csd(adaptive=False)`` weights taper ``k``'s
    eigenspectrum by its concentration eigenvalue: it multiplies each tapered
    spectrum by ``sqrt(lambda_k)`` and divides the taper sum by
    ``sum_k lambda_k``. That is this package's ``taper_weighting="eigen"``; the
    default ``"uniform"`` weighting differs from nitime by up to 2.5x per bin on
    this series, which is the whole of the difference between the two packages.
    The remaining conventions already agree:

    - Tapers: nitime asks for ``int(2 * NW) = 6`` DPSS tapers and this package
      for ``floor(2 * NW) - 1 = 5``, but ``low_bias=True`` drops the sixth
      (concentration 0.71 < 0.9), so both use the same five (equal up to sign;
      the sign cancels in every product ``X_i conj(X_j)``).
    - ``tapered_spectra`` is not a plain FFT of ``signal * taper``: it removes
      each signal's mean first, which is this package's default
      ``detrend_type="constant"``.
    - One-sided doubling of the interior bins (not DC or Nyquist) and the
      division by ``Fs`` match, so with ``Fs`` passed the CSD needs no
      rescaling. Without ``Fs`` nitime divides by its default ``2 * pi``.
      Every bin is compared, so the undoubled DC and Nyquist bins are checked
      too (mean removal does not zero the DC bin of a tapered series).
    - Orientation: nitime's ``csd[i, j]`` is ``E[X_i conj(X_j)]``, the same as
      ``cross_spectral_density()[..., i, j]``.
    - ``NFFT`` defaults to the series length; the transform is given the same
      ``n_fft_samples``. (nitime labels frequencies with
      ``linspace(0, Fs / 2, NFFT // 2 + 1)``, which is only the FFT grid for an
      even ``NFFT``; the series here is even.)
    """
    n_time, _ = nitime_series.shape
    transform = Multitaper(
        nitime_series[:, np.newaxis, :],
        sampling_frequency=_NITIME_SAMPLING_FREQUENCY,
        time_halfbandwidth_product=_NITIME_NW,
        n_fft_samples=n_time,
        taper_weighting="eigen",
    )
    connectivity = Connectivity.from_transform(transform)

    _, expected_coherence = nitime_coherence(
        nitime_series.T,
        csd_method={
            "this_method": "multi_taper_csd",
            "NW": _NITIME_NW,
            "adaptive": False,
            "low_bias": True,
            "sides": "onesided",
        },
    )  # (n_signals, n_signals, n_frequencies)
    coherence = connectivity.coherence_magnitude()[0]  # (n_frequencies, n_signals, n_signals)
    off_diagonal = ~np.eye(3, dtype=bool)
    np.testing.assert_allclose(
        coherence[:, off_diagonal],
        np.moveaxis(expected_coherence, -1, 0)[:, off_diagonal],
        rtol=0,
        atol=1e-8,
    )

    frequencies, expected_csd = multi_taper_csd(
        nitime_series.T,
        Fs=_NITIME_SAMPLING_FREQUENCY,
        NW=_NITIME_NW,
        adaptive=False,
        low_bias=True,
        sides="onesided",
    )
    np.testing.assert_allclose(connectivity.frequencies, frequencies)
    np.testing.assert_allclose(
        connectivity.cross_spectral_density()[0],
        np.moveaxis(expected_csd, -1, 0),
        rtol=1e-6,
    )


_MNE_REFERENCE = Path(__file__).parent / "reference" / "mne_connectivity_reference.npz"
# MNE method: (Connectivity method, element-wise transform, NW, atol). The
# comparison uses these values, and the fixture must record the same ones, so a
# regenerated fixture cannot loosen a tolerance or move a measure to another NW.
_MNE_EXPECTED = {
    "coh": ("coherence_magnitude", "sqrt", 3, 1e-12),
    "imcoh": ("imaginary_coherency", "identity", 3, 1e-12),
    "psi": ("phase_slope_index", "identity", 3, 1e-12),
    "cacoh": ("canonical_coherency", "abs", 3, 1e-7),
    "plv": ("phase_locking_value", "identity", 1, 1e-12),
    "ciplv": ("corrected_imaginary_phase_locking_value", "identity", 1, 1e-12),
    "ppc": ("pairwise_phase_consistency", "identity", 1, 1e-12),
    "pli": ("phase_lag_index", "abs", 1, 1e-12),
    "dpli": ("directed_phase_lag_index", "identity", 1, 1e-12),
    "wpli": ("weighted_phase_lag_index", "abs", 1, 1e-12),
    "wpli2_debiased": ("debiased_squared_weighted_phase_lag_index", "identity", 1, 1e-12),
}
_MNE_UNMATCHED = tuple(
    f"{measure} (NW 3)"
    for measure in ("plv", "ciplv", "ppc", "pli", "dpli", "wpli", "wpli2_debiased")
)
_TRANSFORMS = {"identity": np.asarray, "sqrt": np.sqrt, "abs": np.abs}


@pytest.fixture(scope="module")
def mne_reference():
    """The recorded mne-connectivity outputs and their settings."""
    with np.load(_MNE_REFERENCE) as reference:
        return {key: reference[key] for key in reference.files}


def _mne_tapers(n_time_samples, time_halfbandwidth_product):
    """mne-connectivity's taper set with its taper weights folded in.

    MNE uses periodic DPSS (``sym=False``), keeps those with concentration
    above 0.9 (``mt_low_bias=True``) and weights taper ``k`` by
    ``sqrt(eigenvalue_k)``. Scaling each taper by that weight and averaging
    uniformly gives MNE's weighted cross-spectrum times a constant, which every
    normalized measure cancels. Shape ``(n_time_samples, n_tapers)``.
    """
    tapers, eigenvalues = dpss(
        n_time_samples,
        time_halfbandwidth_product,
        int(2 * time_halfbandwidth_product),
        sym=False,
        return_ratios=True,
    )
    keep = eigenvalues > 0.9
    return (tapers[keep] * np.sqrt(eigenvalues[keep])[:, np.newaxis]).T


@pytest.fixture(scope="module")
def mne_matched_connectivity(mne_reference):
    """``Connectivity`` for each recorded ``NW``, from the recorded data and settings."""
    n_time_samples = int(mne_reference["n_time_samples"])
    time_series = simulate_lagged_broadband(
        mne_reference["lags"],
        float(mne_reference["noise_level"]),
        n_time_samples,
        n_trials=int(mne_reference["n_trials"]),
        random_state=int(mne_reference["seed"]),
    )
    return {
        nw: Connectivity.from_transform(
            Multitaper(
                time_series,
                sampling_frequency=float(mne_reference["sampling_frequency"]),
                time_halfbandwidth_product=nw,
                tapers=_mne_tapers(n_time_samples, nw),
            )
        )
        for nw in np.unique(mne_reference["time_halfbandwidth_products"]).tolist()
    }


@pytest.mark.parametrize("measure", list(_MNE_EXPECTED))
def test_measures_match_mne_connectivity_reference(
    measure, mne_reference, mne_matched_connectivity
):
    """Each measure equals mne-connectivity 0.9's on the same data and tapers.

    Per MNE method, the ``Connectivity`` method, element-wise transform,
    ``NW`` and tolerance (``_MNE_EXPECTED``; the fixture records the same, see
    ``tests/reference/generate_mne_connectivity_reference.py``):

    ============== ============================================= ==== =====
    MNE            this package                                  NW   atol
    ============== ============================================= ==== =====
    coh            sqrt(coherence_magnitude)                     3    1e-12
    imcoh          imaginary_coherency                           3    1e-12
    psi            phase_slope_index over the same (10, 60) Hz   3    1e-12
    cacoh          abs(canonical_coherency component 1)          3    1e-7
    plv            phase_locking_value                           1    1e-12
    ciplv          corrected_imaginary_phase_locking_value       1    1e-12
    ppc            pairwise_phase_consistency                    1    1e-12
    pli            abs(phase_lag_index)                          1    1e-12
    dpli           directed_phase_lag_index                      1    1e-12
    wpli           abs(weighted_phase_lag_index)                 1    1e-12
    wpli2_debiased debiased_squared_weighted_phase_lag_index     1    1e-12
    ============== ============================================= ==== =====

    MNE's connection ``seed -> target`` is ``[..., seed, target]`` here: both
    packages form the cross-spectrum as ``X_seed conj(X_target)``, so the
    signed measures (imcoh, psi, dpli) need no sign change. The comparison uses
    MNE's own tapers (see ``_mne_tapers``): with this package's symmetric DPSS
    and ``taper_weighting="eigen"`` instead, coh and imcoh differ by up to
    5e-4, and at NW 1 the sign-based PLI flips on bins whose imaginary part is
    near zero (one epoch in 30, 0.067). CaCoh is maximized over a phase by an iterative
    optimizer in each package, hence its looser tolerance; its phase is defined
    only modulo pi and is not compared.

    Unmatched, and recorded as such in the fixture: the phase measures with
    more than one taper -- ``plv``, ``ciplv``, ``ppc``, ``pli``, ``dpli``,
    ``wpli`` and ``wpli2_debiased`` at NW 3. MNE sums each epoch's tapers into
    one cross-spectrum before the phase non-linearity; this package treats
    every trial x taper as an observation. With one taper (NW 1) both reduce to
    the same per-epoch estimator, which is what is compared.
    """
    method, transform_name, time_halfbandwidth_product, atol = _MNE_EXPECTED[measure]
    transform = _TRANSFORMS[transform_name]
    connectivity = mne_matched_connectivity[time_halfbandwidth_product]
    frequency_index = np.searchsorted(connectivity.frequencies, mne_reference["freqs"])
    np.testing.assert_allclose(
        connectivity.frequencies[frequency_index], mne_reference["freqs"]
    )
    seeds, targets = mne_reference["seeds"], mne_reference["targets"]

    if method == "phase_slope_index":
        ours = connectivity.phase_slope_index(
            frequencies_of_interest=mne_reference["psi_band"]
        )[0][seeds, targets]
    elif method == "canonical_coherency":
        result = connectivity.canonical_coherency(mne_reference["cacoh_group_labels"])
        ours = result.scores[0, frequency_index, 0, 0]
    else:
        ours = getattr(connectivity, method)()[0][frequency_index][:, seeds, targets].T

    np.testing.assert_allclose(
        transform(ours),
        mne_reference[measure],
        rtol=0,
        atol=atol,
    )


def test_mne_connectivity_reference_records_expected_mapping_and_unmatched(mne_reference):
    """The fixture records the mapping, ``NW`` and tolerance the comparison uses,
    and names exactly the unmatched measures, each with a reason.

    Every recorded measure is compared. Each unmatched ``"<measure> (NW 3)"``
    is a phase measure the fixture records at NW 1 instead, so nothing is
    skipped silently.
    """
    measures = mne_reference["measures"].tolist()
    assert measures == list(_MNE_EXPECTED)
    recorded = zip(
        measures,
        mne_reference["our_methods"].tolist(),
        mne_reference["our_transforms"].tolist(),
        mne_reference["time_halfbandwidth_products"].tolist(),
        mne_reference["atol"].tolist(),
        strict=True,
    )
    for measure, method, transform, nw, atol in recorded:
        assert (method, transform, nw, atol) == _MNE_EXPECTED[measure], measure

    assert mne_reference["unmatched"].tolist() == list(_MNE_UNMATCHED)
    for unmatched in _MNE_UNMATCHED:
        measure = unmatched.removesuffix(" (NW 3)")
        assert _MNE_EXPECTED[measure][2] == 1, unmatched
        index = measures.index(measure)
        assert mne_reference["time_halfbandwidth_products"][index] == 1, unmatched
    assert all(reason.strip() for reason in mne_reference["unmatched_reasons"].tolist())
