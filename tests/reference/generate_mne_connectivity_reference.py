"""Record mne-connectivity's outputs for ``tests/test_cross_package.py``.

mne-connectivity is not a dependency of this package, so its results are
recorded once into ``mne_connectivity_reference.npz`` beside this script. Run
it by hand, in a throwaway environment, only when this script changes::

    uv run --with mne-connectivity python tests/reference/generate_mne_connectivity_reference.py

The data are ``simulate_lagged_broadband(LAGS, NOISE_LEVEL, N_TIME_SAMPLES,
n_trials=N_TRIALS, random_state=SEED)`` at ``SAMPLING_FREQUENCY``: signal 0
leads signals 1 and 2 by 3 and 6 samples. The multitaper settings are
``mt_bandwidth = 2 * NW / T`` (MNE's full bandwidth for half-bandwidth product
``NW`` over a ``T``-second epoch), ``mt_adaptive=False``, ``mt_low_bias=True``,
``faverage=False``, and MNE's default ``fmin`` (5 cycles per epoch, 2.5 Hz).
MNE builds periodic DPSS tapers (SciPy's ``dpss(..., sym=False)``) and weights
taper ``k`` by ``sqrt(eigenvalue_k)``; the test feeds that taper set to
``Multitaper``.

Each recorded measure names the ``Connectivity`` method it is compared with,
the element-wise transform applied to that method's output, the ``NW`` of its
MNE call, and the absolute tolerance of the comparison:

- ``coh``, ``imcoh``, ``psi`` and ``cacoh`` are linear in the trial- and
  taper-averaged cross-spectrum and use ``NW = 3`` (five tapers).
- The phase measures take a non-linear function (sign, unit phasor, ...) of a
  per-epoch cross-spectrum. mne-connectivity first sums the tapers of each
  epoch into one cross-spectrum; this package treats every trial x taper as an
  observation. The two estimators agree only when there is one taper, so these
  are recorded with ``NW = 1`` (``low_bias`` keeps one taper) and listed in
  ``unmatched`` for the multitaper setting.

Pairwise measures are stored with shape ``(n_connections, n_freqs)`` for the
connections ``seeds[k] -> targets[k]`` (MNE's default lower-triangle indices),
``psi`` with shape ``(n_connections,)`` over ``PSI_BAND`` (exclusive edges in
both packages), and ``cacoh`` as the first component's magnitude, shape
``(n_freqs,)``, for the groups ``CACOH_GROUP_LABELS``.
"""

from pathlib import Path

import mne
import mne_connectivity
import numpy as np

from spectral_connectivity.simulate import simulate_lagged_broadband

OUTPUT = Path(__file__).with_name("mne_connectivity_reference.npz")
MAX_BYTES = 200_000

SEED = 0
SAMPLING_FREQUENCY = 500.0
N_TIME_SAMPLES = 1000
N_TRIALS = 30
LAGS = (0, 3, 6)
NOISE_LEVEL = 0.5
PSI_BAND = (10.0, 60.0)
CACOH_GROUP_LABELS = (0, 0, 1)

# MNE method: (NW, Connectivity method, element-wise transform, atol). The
# bivariate measures agree to roundoff (measured <= 2e-15). CaCoh maximizes
# over a phase with an iterative optimizer in each package (measured 5e-9).
MEASURES = {
    "coh": (3, "coherence_magnitude", "sqrt", 1e-12),
    "imcoh": (3, "imaginary_coherency", "identity", 1e-12),
    "psi": (3, "phase_slope_index", "identity", 1e-12),
    "cacoh": (3, "canonical_coherency", "abs", 1e-7),
    "plv": (1, "phase_locking_value", "identity", 1e-12),
    "ciplv": (1, "corrected_imaginary_phase_locking_value", "identity", 1e-12),
    "ppc": (1, "pairwise_phase_consistency", "identity", 1e-12),
    "pli": (1, "phase_lag_index", "abs", 1e-12),
    "dpli": (1, "directed_phase_lag_index", "identity", 1e-12),
    "wpli": (1, "weighted_phase_lag_index", "abs", 1e-12),
    "wpli2_debiased": (1, "debiased_squared_weighted_phase_lag_index", "identity", 1e-12),
}
PHASE_MEASURES = ("plv", "ciplv", "ppc", "pli", "dpli", "wpli", "wpli2_debiased")
UNMATCHED = {
    f"{measure} (NW 3)": (
        "mne-connectivity sums each epoch's tapers into one cross-spectrum before "
        "the phase non-linearity, while this package treats every trial x taper "
        "as an observation, so the estimators differ with more than one taper; "
        "recorded with NW 1 (one taper) instead"
    )
    for measure in PHASE_MEASURES
}


def _mt_kwargs(time_halfbandwidth_product: float) -> dict:
    duration = N_TIME_SAMPLES / SAMPLING_FREQUENCY
    return {
        "sfreq": SAMPLING_FREQUENCY,
        "mode": "multitaper",
        "mt_bandwidth": 2 * time_halfbandwidth_product / duration,
        "mt_adaptive": False,
        "mt_low_bias": True,
        "verbose": False,
    }


def main() -> None:
    """Compute the mne-connectivity outputs and write the reference file."""
    time_series = simulate_lagged_broadband(
        LAGS, NOISE_LEVEL, N_TIME_SAMPLES, n_trials=N_TRIALS, random_state=SEED
    )  # (n_time, n_trials, n_signals)
    epochs = time_series.transpose(1, 2, 0)  # (n_epochs, n_signals, n_time)
    seeds, targets = np.tril_indices(time_series.shape[-1], -1)

    arrays = {}
    freqs = []
    for time_halfbandwidth_product in (3, 1):
        bivariate = [
            name
            for name, (nw, *_) in MEASURES.items()
            if nw == time_halfbandwidth_product and name not in ("psi", "cacoh")
        ]
        results = mne_connectivity.spectral_connectivity_epochs(
            epochs,
            method=bivariate,
            indices=(seeds, targets),
            faverage=False,
            **_mt_kwargs(time_halfbandwidth_product),
        )
        for name, result in zip(bivariate, results, strict=True):
            arrays[name] = result.get_data()  # (n_connections, n_freqs)
            freqs.append(np.asarray(result.freqs))

    psi = mne_connectivity.phase_slope_index(
        epochs,
        indices=(seeds, targets),
        fmin=PSI_BAND[0],
        fmax=PSI_BAND[1],
        **_mt_kwargs(MEASURES["psi"][0]),
    )
    arrays["psi"] = psi.get_data()[:, 0]  # (n_connections,)

    groups = np.asarray(CACOH_GROUP_LABELS)
    cacoh = mne_connectivity.spectral_connectivity_epochs(
        epochs,
        method="cacoh",
        indices=([np.flatnonzero(groups == 0)], [np.flatnonzero(groups == 1)]),
        faverage=False,
        **_mt_kwargs(MEASURES["cacoh"][0]),
    )
    arrays["cacoh"] = np.abs(cacoh.get_data()[0])  # (n_freqs,)
    freqs.append(np.asarray(cacoh.freqs))

    for other in freqs[1:]:
        np.testing.assert_array_equal(other, freqs[0])
    if set(arrays) != set(MEASURES):
        msg = f"recorded {sorted(arrays)}, expected {sorted(MEASURES)}"
        raise RuntimeError(msg)

    names = list(MEASURES)
    np.savez_compressed(
        OUTPUT,
        **arrays,
        measures=np.array(names),
        our_methods=np.array([MEASURES[name][1] for name in names]),
        our_transforms=np.array([MEASURES[name][2] for name in names]),
        time_halfbandwidth_products=np.array([MEASURES[name][0] for name in names]),
        atol=np.array([MEASURES[name][3] for name in names]),
        unmatched=np.array(list(UNMATCHED)),
        unmatched_reasons=np.array(list(UNMATCHED.values())),
        freqs=freqs[0],
        seeds=seeds,
        targets=targets,
        seed=SEED,
        sampling_frequency=SAMPLING_FREQUENCY,
        n_time_samples=N_TIME_SAMPLES,
        n_trials=N_TRIALS,
        lags=np.array(LAGS),
        noise_level=NOISE_LEVEL,
        psi_band=np.array(PSI_BAND),
        cacoh_group_labels=groups,
        mt_adaptive=False,
        mt_low_bias=True,
        mne_connectivity_version=mne_connectivity.__version__,
        mne_version=mne.__version__,
    )
    size = OUTPUT.stat().st_size
    if size > MAX_BYTES:
        msg = f"{OUTPUT.name} is {size} bytes; keep it under {MAX_BYTES}"
        raise RuntimeError(msg)


if __name__ == "__main__":
    main()
