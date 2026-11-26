"""How the tests match mne-connectivity's recorded outputs.

The recorded outputs and their settings are in
``tests/reference/mne_connectivity_reference.npz`` (the ``mne_reference``
fixture in ``tests/conftest.py``); the measures and tolerances are in
``tests/reference/_mne_measures.py``.
"""

from pathlib import Path

import numpy as np
from scipy.signal.windows import dpss

from spectral_connectivity.simulate import simulate_lagged_broadband

MNE_REFERENCE = Path(__file__).parent / "reference" / "mne_connectivity_reference.npz"
# The element-wise transforms named in ``_mne_measures.MEASURES``.
MNE_TRANSFORMS = {"identity": np.asarray, "sqrt": np.sqrt, "abs": np.abs}


def load_mne_reference() -> dict[str, np.ndarray]:
    """The recorded mne-connectivity outputs and their settings."""
    with np.load(MNE_REFERENCE) as reference:
        return {key: reference[key] for key in reference.files}


def mne_epochs(mne_reference: dict[str, np.ndarray]) -> np.ndarray:
    """The epochs MNE was given, ``(n_time_samples, n_trials, n_signals)``."""
    return simulate_lagged_broadband(
        mne_reference["lags"],
        float(mne_reference["noise_level"]),
        int(mne_reference["n_time_samples"]),
        n_trials=int(mne_reference["n_trials"]),
        random_state=int(mne_reference["seed"]),
    )


def mne_tapers(n_time_samples: int, time_halfbandwidth_product: float) -> np.ndarray:
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
