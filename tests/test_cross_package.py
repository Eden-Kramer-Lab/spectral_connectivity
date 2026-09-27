"""Cross-package agreement: the same estimator computed by independent code.

nitime is a dev dependency and is called directly. mne-connectivity is not a
dependency; its outputs are recorded once in
``tests/reference/mne_connectivity_reference.npz`` by the generator script
beside it.
"""

import numpy as np
import pytest
from nitime.algorithms import coherence as nitime_coherence
from nitime.algorithms import multi_taper_csd

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
    - Orientation: nitime's ``csd[i, j]`` is ``E[X_i conj(X_j)]``, the same as
      ``cross_spectral_density()[..., i, j]``.
    - ``NFFT`` defaults to the series length; the transform is given the same
      ``n_fft_samples``. (nitime labels frequencies with
      ``linspace(0, Fs / 2, NFFT // 2 + 1)``, which is only the FFT grid for an
      even ``NFFT``; the series here is even.)

    DC is excluded from the comparison: after mean removal its power is
    roundoff-level.
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
    interior = slice(1, -1)

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
        coherence[interior][:, off_diagonal],
        np.moveaxis(expected_coherence, -1, 0)[interior][:, off_diagonal],
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
        connectivity.cross_spectral_density()[0][interior],
        np.moveaxis(expected_csd, -1, 0)[interior],
        rtol=1e-6,
    )
