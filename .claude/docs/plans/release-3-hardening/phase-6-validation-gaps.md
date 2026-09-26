# Phase 6 — Planted-structure tests for the property-only measures, plus cross-package checks

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [contracts](shared-contracts.md)

**Inputs to read first:**

- [tests/test_directed_measures_oracle.py:42-117](../../../../tests/test_directed_measures_oracle.py#L42-L117) — `_analytic_var` (A, H, S on the FFT grid for a VAR), `_fourier_coefficients_with_cross_spectrum` (coefficients whose taper-mean cross-spectrum is exactly S), `var_oracle`; [:341-361](../../../../tests/test_directed_measures_oracle.py#L341-L361) the Geweke closed form (atol 1e-5); [:307-319](../../../../tests/test_directed_measures_oracle.py#L307-L319) the only blockwise test, scalar blocks; [:172-173, 417-422](../../../../tests/test_directed_measures_oracle.py#L417-L422) `_CHAIN_COEFFICIENTS` (0 -> 1 -> 2) and `_CHAIN_NOISE`.
- [tests/test_connectivity.py:210-224](../../../../tests/test_connectivity.py#L210-L224) — CSD's only test (Hermitian, diagonal = power); [:342-362](../../../../tests/test_connectivity.py#L342-L362) partial coherence vs its own inverse-spectrum definition on white noise; [:455-470, 916-940](../../../../tests/test_connectivity.py#L455-L470) MIC/MIM scalar-group and mixing-invariance tests.
- [src/spectral_connectivity/connectivity.py:1928](../../../../src/spectral_connectivity/connectivity.py#L1928) `cross_spectral_density`, [:2051](../../../../src/spectral_connectivity/connectivity.py#L2051) `coherence_phase`'s sign convention ("positive `[..., i, j]` means signal `i` leads `j`"), [:2236](../../../../src/spectral_connectivity/connectivity.py#L2236) `partial_coherence(regularization=...)`, [:2865, 2930](../../../../src/spectral_connectivity/connectivity.py#L2865) MIC/MIM, [:4151](../../../../src/spectral_connectivity/connectivity.py#L4151) blockwise Granger.
- Observed 2026-09-26 while probing nitime 0.12.1: our tapers equal `nitime.algorithms.dpss_windows(n, 3, 5)` up to sign (scaled by `sqrt(fs)`), yet `nitime.algorithms.multi_taper_csd(..., adaptive=False, sides="onesided")` differs from `cross_spectral_density` per bin by up to 2x with the same NW, and nitime's `tapered_spectra` output is not a plain `numpy.fft.fft` of `signal * taper` either. The cause was not determined; it is this phase's first task.
- `pyproject.toml:63, 83` — nitime is in both dev lists already; `testpaths = ["tests"]`, `python_files = ["test_*.py"]` (`:121-123`), so a generator script under `tests/reference/` with a non-`test_` name is not collected.
- User decision (2026-09-26): mne-connectivity comparisons use a recorded fixture, not a dependency.

**Contracts referenced:**

- [Directed-array orientation](shared-contracts.md#directed-array-orientation) — new tests assert `[..., 0, 1]` for `0 -> 1`.
- [Public simulators](shared-contracts.md#public-simulators) — the planted signals come from `simulate_shared_oscillation`, `simulate_lagged_broadband`, `simulate_MVAR`.

## Tasks

- **nitime cross-check** (`tests/test_cross_package.py`, new). First find the configuration under which `nitime.algorithms.coherence(ts, csd_method={"this_method": "multi_taper_csd", "NW": 3, "adaptive": False, "low_bias": True, "sides": "onesided"})` agrees with `coherence_magnitude` on a seeded 3-signal, single-window series (start by reading `nitime/algorithms/spectral.py::tapered_spectra` and `mtm_cross_spectrum` for their normalization and sign handling; candidates are eigenvalue weighting, `NFFT` handling, and one-sided doubling). Encode the matched settings and the reason in the test docstring, and assert coherence agreement at `atol=1e-8` on interior bins and one-sided CSD agreement at `rtol=1e-6` after the documented scaling. If no configuration agrees, the test still lands, comparing against `scipy.signal.csd` fed our DPSS tapers as explicit windows (same estimator, independent code), and the docstring records the nitime convention difference found.
- **mne-connectivity reference fixture.** `tests/reference/generate_mne_connectivity_reference.py` (run by hand: `uv run --with mne-connectivity python tests/reference/generate_mne_connectivity_reference.py`) builds seeded data with `simulate_lagged_broadband((0, 3, 6), 0.5, 1000, n_trials=30, random_state=0)` at 500 Hz, computes with `mne_connectivity.spectral_connectivity_epochs(data.transpose(1, 2, 0), method=["coh", "imcoh", "plv", "ciplv", "ppc", "pli", "dpli", "wpli", "wpli2_debiased"], mode="multitaper", sfreq=500, mt_bandwidth=<2 * NW / T>, mt_adaptive=False, mt_low_bias=True, faverage=False)` and `phase_slope_index`, and `cacoh` for groups `[0, 0, 1]` via `indices`, then saves `tests/reference/mne_connectivity_reference.npz` with the arrays, `freqs`, the seed, the settings, `mne_connectivity.__version__`, and an `unmatched` list. The test `test_measures_match_mne_connectivity_reference` regenerates our side with `Multitaper(..., time_halfbandwidth_product=NW, taper_weighting=<whatever matches>)` and compares, per measure, with the mapping and tolerance the generator recorded (expected: `coh` ↔ `coherence_magnitude` as `sqrt` or squared per MNE's definition, `imcoh` ↔ `imaginary_coherency`, `plv` ↔ `phase_locking_value`, `ciplv` ↔ `corrected_imaginary_phase_locking_value`, `ppc` ↔ `pairwise_phase_consistency`, `pli` ↔ `abs(phase_lag_index)`, `dpli` ↔ `directed_phase_lag_index`, `wpli` ↔ `abs(weighted_phase_lag_index)`, `wpli2_debiased` ↔ `debiased_squared_weighted_phase_lag_index`, `psi` ↔ `phase_slope_index` on the same band, `cacoh` ↔ `canonical_coherency` component 1 magnitude). Any measure that cannot be matched goes into `unmatched` with the reason and is asserted absent from the comparison, never silently skipped. Keep the fixture under 200 kB (store float32 if needed and widen tolerance accordingly).
- **Cross-spectral density planted test** (`tests/test_connectivity.py`, next to `:210`): `simulate_shared_oscillation(40, 500, 1000, n_trials=50, amplitudes=[1.0, 2.0], phase_offsets=[0.0, np.pi / 3], noise_levels=0.5, random_state=0)`, `Multitaper(..., time_halfbandwidth_product=2)`. Assert: argmax of `|csd[0, :, 0, 1]|` is the 40 Hz bin; `np.angle(csd[0, peak, 1, 0])` equals `+pi/3` within 0.05 (signal 1 leads by pi/3; confirm the sign against `coherence_phase`'s documented convention at `:2051` and assert `coherence_phase[0, peak, 1, 0]` has the same sign); the trapezoid integral of `csd[0, :, 0, 1]` over `peak ± 2 * frequency_resolution` has magnitude `A0 * A1 / 2 = 1.0` within 5% (the one-sided cross-power of two coherent sinusoids).
- **MIC / MIM multichannel planted test** (`tests/test_multivariate.py`): six signals, `simulate_shared_oscillation(20, 500, 1000, n_trials=100, amplitudes=[1, 0.8, 1.2, 1, 1.1, 0.9], phase_offsets=[0, 0, 0, pi/2, pi/2, pi/2], noise_levels=0.5, random_state=1)`, groups `[0, 0, 0, 1, 1, 1]`, NW 3 (500 observations). MIC has a positive finite-sample bias: with 500 × 20 samples and NW 2 (60 observations) the off-peak median measured 0.236 and the zero-lag value 0.208, so the smaller design cannot separate signal from bias. Measured with the sizes above: lagged MIC at 20 Hz 0.998, off-peak median 0.078, zero-lag MIC at 20 Hz 0.059, zero-lag canonical coherence 0.997. Assert: MIC at 20 Hz > 0.9; median MIC at `|f - 20| > 10` Hz < 0.15; `MIM >= MIC**2 - 1e-9` everywhere (MIM is the sum of squared singular values, MIC the largest). Zero-offset variant (volume conduction): MIC at 20 Hz < 0.15 and no larger than twice its own off-peak median (bias level, not coupling), while `canonical_coherence` at 20 Hz > 0.9. Mark `slow` only if the two cases together exceed two seconds.
- **Blockwise Granger with multichannel blocks** (`tests/test_directed_measures_oracle.py`): a 4-signal VAR with blocks {0, 1} -> {2, 3}:

  ```python
  _BLOCK_A1 = np.array(
      [[0.5, 0.1, 0.0, 0.0],
       [0.1, 0.5, 0.0, 0.0],
       [0.4, 0.2, 0.5, 0.1],
       [0.3, 0.3, 0.1, 0.5]]
  )
  _BLOCK_COEFFICIENTS = np.stack([_BLOCK_A1, -0.6 * np.eye(4)])
  _BLOCK_NOISE = np.eye(4)
  ```

  Assert stability first (companion-matrix spectral radius < 1). Geweke's block measure for uncorrelated innovations: `F_{X->Y}(f) = log det S_YY(f) - log det (S_YY(f) - H_YX(f) Sigma_XX H_YX(f)^H)` with X = {0, 1}, Y = {2, 3}; `F_{Y->X}` is analytically 0 because `H_XY = 0`. Feed the exact spectrum through `_fourier_coefficients_with_cross_spectrum(S)`, call `blockwise_spectral_granger_prediction([0, 0, 1, 1])`, and assert `result[0, :, 0, 1]` matches the closed form at `atol=1e-5` and `result[0, :, 1, 0]` is below 1e-5 (use `np.linalg.slogdet` for the determinants; the non-negative half of the grid). Add a simulated-data companion: `simulate_MVAR(_BLOCK_COEFFICIENTS, n_time_samples=2000, n_trials=30, random_state=0)` → mean over frequency of `[0, 1]` exceeds 5x that of `[1, 0]`.
- **Partial coherence planted tests** (`tests/test_connectivity.py`, next to `:342`). The existing chain (`_CHAIN_COEFFICIENTS`, coupling 0.4) is too weak for a simulated threshold: its analytic partial coherence peaks at 0.068 for (0, 1) and 0.817 for (1, 2), with pairwise coherence (0, 2) peaking at 0.31. Define a stronger chain in the oracle module and reuse it here:

  ```python
  _STRONG_CHAIN_COEFFICIENTS = np.stack(
      [np.array([[0.5, 0.0, 0.0], [0.8, 0.5, 0.0], [0.0, 0.8, 0.5]]), -0.6 * np.eye(3)]
  )  # companion-matrix spectral radius 0.775
  ```

  whose analytic partial coherence peaks at 0.250 for (0, 1) and 0.817 for (1, 2), is exactly 0 for (0, 2), and whose pairwise coherence (0, 2) peaks at 0.785. Analytic test — from `_analytic_var(_STRONG_CHAIN_COEFFICIENTS, np.eye(3), 256)`'s S, `P = inv(S)`, `pc_ij = |P_ij|^2 / (P_ii P_jj)`; assert `partial_coherence(regularization=1e-12)` matches at `atol=1e-8` on the exact-spectrum `Connectivity`. Simulated test — `simulate_MVAR(_STRONG_CHAIN_COEFFICIENTS, n_time_samples=2000, n_trials=60, random_state=0)` at 200 Hz, `Multitaper(..., time_halfbandwidth_product=3, n_fft_samples=2000)` so the estimate and `_analytic_var(..., n_fft=2000)` share a grid (300 observations). Do not assert on the peak value: the maximum over 1001 bins of a positively biased estimate overshoots (measured peak 0.396 at 30 trials and 0.323 at 60 against the analytic 0.250). Assert instead that the mean absolute deviation of each simulated partial-coherence curve from its analytic curve over the non-negative bins is below 0.05 (measured 0.023 for (0, 1), 0.021 for (1, 2), 0.003 for (0, 2)), that `partial_coherence[0, :, 0, 2]` stays below 0.05 everywhere (measured max 0.028; the 0-2 link is mediated by 1), and that the pairwise `coherence_magnitude[0, :, 0, 2]` peak exceeds 0.5 (measured 0.83).
- `CHANGELOG.md` "Added": one bullet for the planted and cross-package checks; `CLAUDE.md` Testing Strategy: one line on the reference fixture and how to regenerate it.

## Deliberately not in this phase

- Adding mne-connectivity or mne to any dependency list.
- Tests for measures that already have planted-structure evidence (see the audit summary in the overview's goals).
- Changing any measure implementation. If a planted test fails, that is a finding to report, not a threshold to tune; the exception is a wrong sign convention in the test's own expectation, which the analytic oracle settles.

## Validation slice

| Test | Asserts |
| --- | --- |
| `tests/test_cross_package.py::test_coherence_matches_nitime` | `atol=1e-8` interior-bin agreement under the documented settings (or the scipy fallback with the reason recorded) |
| `tests/test_cross_package.py::test_measures_match_mne_connectivity_reference[<measure>]` | each mapped measure within its recorded tolerance; `unmatched` measures are exactly the ones the docstring lists |
| `tests/test_connectivity.py::test_cross_spectral_density_recovers_planted_amplitude_and_phase` | peak bin, `+pi/3` phase within 0.05 with the sign matching `coherence_phase`, band integral magnitude 1.0 ± 5% |
| `tests/test_multivariate.py::test_mic_recovers_lagged_between_group_source` | MIC(20 Hz) > 0.9, off-peak median < 0.15, `MIM >= MIC**2` |
| `tests/test_multivariate.py::test_mic_rejects_zero_lag_shared_source` | MIC(20 Hz) < 0.15 and ≤ 2 × its off-peak median; canonical coherence(20 Hz) > 0.9 |
| `tests/test_directed_measures_oracle.py::test_blockwise_granger_matches_geweke_block_closed_form` | `[0, :, 0, 1]` within 1e-5 of the closed form; `[0, :, 1, 0]` < 1e-5 |
| `tests/test_directed_measures_oracle.py::test_blockwise_granger_direction_on_simulated_var` | planted block direction dominates 5x |
| `tests/test_connectivity.py::test_partial_coherence_matches_inverse_spectrum_oracle_on_chain` | `atol=1e-8` on the strong chain |
| `tests/test_connectivity.py::test_partial_coherence_removes_mediated_link_on_simulated_chain` | 0-2 partial coherence < 0.05 everywhere while pairwise 0-2 coherence peak > 0.5; mean absolute deviation of each simulated curve from the analytic curve < 0.05 |

None of these need the `slow` marker if the simulated cases stay at the sizes above (each under a second); measure and mark if not.

## Fixtures

`tests/reference/mne_connectivity_reference.npz` (checked in, generated once by the script beside it; regenerate only when the fixture generator changes). All other data is simulated inline with fixed seeds via the phase-5 simulators.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.
