"""The mne-connectivity measures in the reference fixture and how each is compared.

Plain data shared by ``generate_mne_connectivity_reference.py`` (which records
it into the fixture) and ``tests/test_cross_package.py`` (which compares with
it and checks that the fixture recorded the same values).
"""

# MNE method: (Connectivity method, element-wise transform, NW, atol). The
# bivariate measures agree to roundoff (measured <= 2e-15). CaCoh maximizes
# over a phase with an iterative optimizer in each package (measured 5e-9).
MEASURES = {
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

# The phase measures are compared with one taper (NW 1); with several tapers
# the two packages' estimators differ, so their NW 3 variants are unmatched.
PHASE_MEASURES = tuple(name for name, (_, _, nw, _) in MEASURES.items() if nw == 1)
UNMATCHED = {
    f"{measure} (NW 3)": (
        "mne-connectivity sums each epoch's tapers into one cross-spectrum before "
        "the phase non-linearity, while this package treats every trial x taper "
        "as an observation, so the estimators differ with more than one taper; "
        "recorded with NW 1 (one taper) instead"
    )
    for measure in PHASE_MEASURES
}
