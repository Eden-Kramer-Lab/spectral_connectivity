# Phase 7 — Benchmark the default set, then compute the phase-lag moments in one pass

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#a-phase-lag-moments-in-one-pass)

**Inputs to read first:**

- [src/spectral_connectivity/connectivity.py:86-96](../../../../src/spectral_connectivity/connectivity.py#L86-L96) — `_IMAGINARY_MOMENTS` and `_IMAGINARY_MOMENT_PAIR_SYMMETRY`; [:163](../../../../src/spectral_connectivity/connectivity.py#L163) `PHASE_LAG_INDEX_MAX_WORKSPACE_ELEMENTS`; [:3355-3482](../../../../src/spectral_connectivity/connectivity.py#L3355-L3482) `_imaginary_moment_cache`, `_has_no_phase_lag`, `_imaginary_cross_spectrum_moments` (the tile loop at `:3423-3480`); the four consumers at [:3484](../../../../src/spectral_connectivity/connectivity.py#L3484) `phase_lag_index` (`sign`, `absolute`), [:3560-3614](../../../../src/spectral_connectivity/connectivity.py#L3560-L3614) `directed_phase_lag_index` (`sign`, `absolute`), [:3616](../../../../src/spectral_connectivity/connectivity.py#L3616) `weighted_phase_lag_index` (`imaginary`, `absolute`), [:3681](../../../../src/spectral_connectivity/connectivity.py#L3681) debiased PLI (`sign`, `absolute`), [:3740](../../../../src/spectral_connectivity/connectivity.py#L3740) debiased wPLI (`imaginary`, `squared`, `absolute`).
- [connectivity.py:1243-1259](../../../../src/spectral_connectivity/connectivity.py#L1243-L1259) — `_nonnegative_fourier_coefficients`, `_nonnegative_cross_spectral_matrix`; [:1292-1378](../../../../src/spectral_connectivity/connectivity.py#L1292-L1378) the weighted reduced CSM; [:1477-1500](../../../../src/spectral_connectivity/connectivity.py#L1477-L1500) `_expectation`. The `imaginary` moment equals the imaginary part of the reduced CSM under both weightings (same weights, same normalization).
- `Connectivity.clear_cache()` and the cache-invalidation path must keep clearing `_imaginary_moment_cache` (find where the cache dict is reset when inputs change).
- `tests/conftest.py` `backend_modules` / device-emulation fixture: this phase adds no module, but the new `einsum` / `count_nonzero(axis=tuple)` / ufunc `out=` calls must run under the emulated backend tests (`tests/test_emulated_device_backend.py`).
- Timings measured 2026-09-26 (18-core Mac, CPU) that the benchmark must reproduce as its baseline: case (b) default set 6.3 s, `coherence_magnitude` alone 0.46 s, PLI 2.6 s, debiased PLI 2.6 s, debiased wPLI 1.7 s, wPLI 1.5 s, Granger 2.3 s; case (a) default set 0.85 s. Prototype numbers are in [designs A](designs.md#a-phase-lag-moments-in-one-pass).

**Contracts referenced:** none.

**Designs referenced:** [designs.md#a-phase-lag-moments-in-one-pass](designs.md#a-phase-lag-moments-in-one-pass) (algorithm, reference loop, benchmark script spec).

## Tasks

- **Baseline capture (before touching `connectivity.py`).** Add `benchmarks/bench_default_measures.py` per designs A and run `uv run python benchmarks/bench_default_measures.py --save /tmp/baseline.npz` on master; paste the printed table into the PR description. Add the command to `CLAUDE.md` Development Commands with one line on when to run it (any change under `connectivity.py` that touches cached intermediates).
- **Restructure `_imaginary_cross_spectrum_moments`** per designs A: compute all four moments together at first use, keeping NaN propagation for the count-based `sign` moment via the per-tile `isnan` mask and keeping `imaginary` as a per-observation mean of the tile (not the CSM shortcut; see designs A item 2); reuse two tile buffers; single-pass reductions on the unweighted path, `_expectation` on the weighted path, both through `_reduce_phase_lag_tile`. Rewrite the method docstring (`:3389-3421`) so the rationale matches the new behavior. Nothing changes in the four public methods.
- **Replace the two structural cache tests whose premise the design changes** (`tests/test_connectivity.py:1659-1681` `test_phase_lag_index_moments_are_computed_lazily`, which asserts a lone PLI leaves only `{"sign", "absolute"}` cached, and `:1702-1717` `test_failed_phase_lag_reduction_caches_nothing`, which injects the failure through `_expectation`). New versions: `test_phase_lag_moments_are_computed_together_once` asserts that after any single phase-lag measure the cache holds exactly the four keys and that a second measure does not re-enter the tile loop (patch `_reduce_phase_lag_tile` with `wraps=` and assert `call_count`, the pattern already used at `tests/test_connectivity.py:2159` and `:2183` for the Granger kernels); `test_failed_phase_lag_reduction_caches_nothing` keeps its name and intent but patches `_reduce_phase_lag_tile` with `side_effect=MemoryError`, then asserts the cache is empty and the next measure equals a fresh instance's result. Keep `:1720-…` `test_phase_lag_family_uses_tiled_workspace_not_full_outer_product` as is; it must still pass.
- **Parity gate.** `uv run python benchmarks/bench_default_measures.py --compare /tmp/baseline.npz` must report every default measure within 1e-12 on both cases (expect exactly 0 for PLI and debiased PLI). Paste the timing table next to the baseline in the PR description; the default set on case (b) must be at least 1.4x faster and the four phase-lag measures at least 2x, or the PR is not ready.
- **GPU path.** Run `uv run pytest tests/test_emulated_device_backend.py tests/test_gpu_smoke.py` (the latter skips without CuPy); confirm the emulated backend exercises the new loop.
- `CHANGELOG.md` "Performance": one bullet with the measured before/after for the default set and the phase-lag family, and the statement that outputs are unchanged.

## Deliberately not in this phase

- Changing `DEFAULT_METHODS` or any measure's definition.
- Cropping frequencies before computation (overview non-goal).
- Multithreading the tile loop (measure first; overview non-goal).
- Granger-family performance.

## Validation slice

| Test | Asserts |
| --- | --- |
| `benchmarks/bench_default_measures.py --compare` (manual, in PR description) | max abs diff ≤ 1e-12 for all eleven default measures on cases (a) and (b); default set ≥ 1.4x, phase-lag family ≥ 2x faster on (b) |
| `tests/test_connectivity.py::test_weighted_phase_lag_index_is_exactly_one_at_a_tiny_constant_lag` (new) | coefficients `stack([X, X * exp(-1j * 1e-13)])` with `X` random `(1, 50, 5, 8)`: wPLI is exactly `1.0` at every bin (as on master) and PLI exactly `1.0`; the same on the weighted path via a `MorletWavelet` with `smoothing_time` |
| `tests/test_connectivity.py::test_phase_lag_moments_match_definition_under_observation_weights` (new) | on a `MorletWavelet` with `smoothing_time` (observation weights present), each of the four moments equals `_expectation(f(Im(X_i conj X_j)))` written out in the test at `atol=1e-12` |
| `tests/test_connectivity.py::test_phase_lag_moments_are_computed_together_once` / `test_failed_phase_lag_reduction_caches_nothing` (replacing `:1659-1681`, `:1702-1717`) | four keys cached after one measure and the tile loop not re-entered by a second; a failing tile leaves an empty cache and the next call recomputes correctly |
| `tests/test_connectivity.py::test_phase_lag_moments_propagate_nan_observations` (new) | coefficients `(1, 8, 3, 16, 2)` with one observation set to NaN at one bin: all four phase-lag measures and `directed_phase_lag_index` are NaN at that bin and finite at the others, exactly as on master (probed: PLI NaN at the bin, 0.0833 at its neighbor); repeat with a `MorletWavelet(..., edge_mode="nan")` transform |
| `tests/test_connectivity.py::test_phase_lag_family_single_measure_matches_batch` (new) | `phase_lag_index` from a fresh instance equals the one from an instance that first computed the other three measures, exactly |
| `tests/test_connectivity.py::test_phase_lag_moments_partial_last_block` (new) | `n_signals` not divisible by the block size (force a small `PHASE_LAG_INDEX_MAX_WORKSPACE_ELEMENTS` via monkeypatch) gives the same result as one block |
| existing PLI/wPLI/dPLI/debiased numerical tests and snapshots | unchanged (only the two structural tests above are rewritten) |
| `tests/test_emulated_device_backend.py` | passes with the new array calls |
| `uv run mypy src/` | clean |

## Fixtures

Seeded noise inline; `/tmp/baseline.npz` is produced by the baseline task and is not checked in.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind): the per-key tile re-formation and the two structural tests whose premise no longer holds.
- User-facing documentation listed as tasks is updated, not deferred.
