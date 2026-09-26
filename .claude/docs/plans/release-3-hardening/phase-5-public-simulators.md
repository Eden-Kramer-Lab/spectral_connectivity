# Phase 5 — Public simulators, and a simulated-examples notebook that asserts

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [contracts](shared-contracts.md#public-simulators)

**Inputs to read first:**

- [src/spectral_connectivity/simulate.py](../../../../src/spectral_connectivity/simulate.py) — 90 lines, `simulate_MVAR` only; its `random_state` resolution at `:64-68` becomes the shared `_generator` helper.
- [tests/test_advanced_connectivity.py:97-111](../../../../tests/test_advanced_connectivity.py#L97-L111) — `_lagged_broadband` (circular `np.roll`, per-signal noise draws in a loop; used with threshold assertions at `:629`, `:662`, `:701`, `:736`, `:901`, `:939`). [:122-165](../../../../tests/test_advanced_connectivity.py#L122-L165) — canonical-coherence simulation (shared 20 Hz + private 40 Hz, groups of 3). [:268-290](../../../../tests/test_advanced_connectivity.py#L268-L290) — global-coherence simulation (five signals, amplitudes `0.5 + 0.2 i`, noise 0.3).
- [tests/test_notebooks.py:562-583](../../../../tests/test_notebooks.py#L562-L583) — `_lagged_broadband_pair` (the semantics the contract reproduces bit for bit; used at `:591`, `:626`, `:662`, `:700`, `:735`, all with syrupy snapshots).
- [tests/test_simulate.py](../../../../tests/test_simulate.py) — existing determinism/shape tests for `simulate_MVAR`; the new tests go here.
- [examples/Tutorial_On_Simulated_Examples.py](../../../../examples/Tutorial_On_Simulated_Examples.py) — 2481 lines, sections at `:31` power (200 Hz, 30 Hz), `:97` spectrogram (50 Hz onset at 25 s, trials, resolution), `:423` coherence, `:543` coherograms, `:670` imaginary coherence, `:797` PLV, `:924` PLI, `:1051` wPLI, `:1306` debiased wPLI, `:1435` PPC, `:1564` group delay (three sub-cases), `:1876` PSI, `:2177` canonical coherence, `:2391` global coherence, `:2459` xarray. Zero `assert` statements. Each section builds its sinusoids inline (e.g. `:428-440`).
- [tests/test_notebooks.py:1012-1065](../../../../tests/test_notebooks.py#L1012-L1065) — `test_tutorial_notebook_executes` (`slow`), the only check on the notebook; `tests/test_notebooks.py:150-200` shows the assertion style the tests use (peak within 0.1 Hz, off-peak noise floor `2σ²/fs` within 5%).
- `docs/api.rst` includes `spectral_connectivity.simulate` recursively, so new public functions get API pages without docs changes.

**Contracts referenced:**

- [Public simulators](shared-contracts.md#public-simulators) — implement exactly; the bit-identical semantics keep the notebook-test snapshots stable.

## Tasks

- `simulate.py`: add `_generator(random_state)` and use it in `simulate_MVAR` (behavior unchanged); add `simulate_lagged_broadband` and `simulate_shared_oscillation` per the contract, NumPy-style docstrings with shapes and a runnable example each (doctested in CI). Type-annotate; `uv run mypy src/` clean.
- `tests/test_simulate.py`: for `simulate_lagged_broadband` — negative and non-integer lags rejected; shape with and without trials; the cross-correlation of signal 1 against signal 0 peaks at exactly `lags[1] - lags[0]` samples (noise 0.1, 5000 samples); `noise_levels=0` reproduces the pure shifted source; seed determinism. For `simulate_shared_oscillation` — shape; the FFT bin at `frequency` dominates every signal's spectrum; the phase of `X_k conj(X_m)` at that bin equals `phase_offsets[k] - phase_offsets[m]` within 1e-6 when `noise_levels=0` and `random_phase_per_trial=False`; with `random_phase_per_trial=True` trial phases differ; amplitude 0 leaves a flat zero signal; seed determinism.
- Replace the private helpers: delete `tests/test_advanced_connectivity.py:97-111` and call `simulate_lagged_broadband(lags, noise_levels, n_time, n_trials, random_state=self.rng)` at the six sites (threshold tests; if any threshold now fails, report it rather than loosening it). Delete `tests/test_notebooks.py:562-583` and call `simulate_lagged_broadband((0, lag) if leader == 0 else (lag, 0), noise_sd, n_time_samples, n_trials, random_state=rng)` at the five sites; the snapshots must not change (that is the contract's bit-identity claim; if one does, diff the arrays and fix the simulator, not the snapshot). Rewrite the canonical (`:122-165`) and global (`:268-290`) simulations with `simulate_shared_oscillation` (thresholds only).
- Notebook rewrite of `Tutorial_On_Simulated_Examples.py`: each section builds its data with `simulate_shared_oscillation` (power, coherence, coherogram, PLV/PLI/wPLI/PPC, canonical with a second call for the private 40 Hz rhythm, global), `simulate_lagged_broadband` (group delay, PSI), and `simulate_MVAR` where Granger is shown, keeping every plot. Add one assert cell at the end of each section stating what the plot shows, mirroring `tests/test_notebooks.py`'s criteria: peak frequency within one frequency-resolution bin, coherence-family value at the peak > 0.95 and median off-peak < 0.1, PLI/wPLI sign follows the leader, group delay and PSI sign follow the leader with delay within 1 ms, canonical coherence > 0.95 at 20 Hz and < 0.5 at 40 Hz, global coherence > 0.95 at the shared frequency, Granger dominant in the planted direction. Regenerate with `uvx jupytext --sync examples/Tutorial_On_Simulated_Examples.py`.
- README/docs: one sentence in the README's tutorial list, "The simulated-examples notebook builds every case with `spectral_connectivity.simulate` and asserts what each plot shows." `CHANGELOG.md` "Added": the two simulators and the asserting notebook.

## Deliberately not in this phase

- New planted-structure tests for CSD, MIC/MIM, blockwise Granger, partial coherence (phase 6 uses these simulators).
- Rewriting the inline sinusoid construction inside `tests/test_notebooks.py` (snapshot churn for no coverage gain).
- Exporting the simulators from `spectral_connectivity/__init__.py`.

## Validation slice

| Test | Asserts |
| --- | --- |
| `tests/test_simulate.py::test_lagged_broadband_cross_correlation_peaks_at_lag` | argmax of the cross-correlation is `lags[1] - lags[0]` |
| `tests/test_simulate.py::test_lagged_broadband_reproduces_pair_helper_semantics` | `lags=(0, L)`, `n_trials=None`: signal 0 equals `source[L:]`, signal 1 equals `source[:n]` for a seeded generator replayed by hand |
| `tests/test_simulate.py::test_lagged_broadband_rejects_negative_or_fractional_lags` | `lags=(-3, 0)` and `lags=(0, 1.5)` raise `ValueError` matching `non-negative integers` |
| `tests/test_simulate.py::test_shared_oscillation_phase_offsets` | cross-spectrum phase at the planted bin equals the offset difference within 1e-6 |
| `tests/test_simulate.py::test_shared_oscillation_random_trial_phase` | trial-to-trial phases at the bin are not all equal; with the flag off they are |
| `tests/test_notebooks.py` (all) | pass with **unchanged** `__snapshots__/test_notebooks.ambr` |
| `tests/test_advanced_connectivity.py` | group-delay, PSI, canonical, global tests pass at their existing thresholds |
| `tests/test_notebooks.py::test_tutorial_notebook_executes[Tutorial_On_Simulated_Examples.ipynb]` (slow) | executes; because the notebook now asserts, a false plot fails this test |
| `uv run pytest --doctest-modules src/spectral_connectivity/simulate.py` | examples run |

## Fixtures

Seeded generators inline; no files.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind): `_lagged_broadband`, `_lagged_broadband_pair`.
- User-facing documentation listed as tasks is updated, not deferred.
