# Overview — Scope, dependencies, integration, risks

[← back to PLAN.md](PLAN.md)

## Current codebase integration points

All line numbers are master at `bcc16a8`.

- `.github/workflows/release.yml:20-21` — workflow-level `permissions: contents: read`; `:55-58` the zizmor step. Only the step's inputs change (phase 1).
- `src/spectral_connectivity/connectivity.py:420-440` — class docstring stating the two orientations; `:1655-1662` jackknife docstring; `:3863-4530` the ten Wilson-family directed methods whose return sites gain the swap (phase 2). `:86-96` `_IMAGINARY_MOMENTS`/symmetry table, `:163` workspace cap, `:3355-3482` moment cache and tile loop, `:3484`, `:3616`, `:3681`, `:3740` the four phase-lag measures (phase 7). `:1243-1259` non-negative coefficient/CSM views and `:1292-1378` weighted reduced CSM (phase 7 reuses them). `:1477-1500` `_expectation` (weighted path, phase 7).
- `src/spectral_connectivity/_granger.py:280`, `:516`, `:541`, `:570` — kernel docstrings and writes in `[..., target, source]`; kernels stay native, docstrings gain one sentence (phase 2).
- `src/spectral_connectivity/_measure_registry.py:14-67` `_MeasureSpec` (`transpose_output` at `:50`, `array_orientation` at `:59-67`), `:82-89` its validation, `:93-125` `_wilson_directed_spec` — field and property removed (phase 2). `:456-475` `_requested_methods` untouched.
- `src/spectral_connectivity/_result_formatting.py:37-58` `_check_method_accepts_kwargs` (phase 3 message rewrite), `:94-112` measure invocation (phase 3 `group_labels` routing), `:131-134`, `:185-186`, `:249-250` the transposes (phase 2 deletes).
- `src/spectral_connectivity/wrapper.py:71-131` `MeasureInfo` (`array_orientation` at `:109-112`, `:131`, `:205`; phase 2), `:502-519` and `:801-825` the two public signatures (phase 3 adds `group_labels`), `:584-620` and `:856-870` their `squeeze`/`connectivity_kwargs`/`**kwargs` docs (phase 3), `:681` orientation sentence (phase 2), `:412-499` `_format_and_reduce_measures` (phase 3 routing), `:745-756` `sampling_frequency is None` check (phase 3).
- `src/spectral_connectivity/transforms.py:46-66` `_validate_sampling_frequency` (phase 3 type check); `:860`, `:1769`, `:1900` the `= 1000` defaults and `:640`, `:1740`, `:1871` their docstrings (phase 3). `:91-120` `MultitaperParameters` (phase 4 checks whether its keys splat into the wrapper).
- `src/spectral_connectivity/_input_handling.py:352-361` — second `np.isfinite(sampling_frequency)` site (phase 3 reuses the shared validator).
- `src/spectral_connectivity/simulate.py` (90 lines, `simulate_MVAR` only) — gains two functions (phase 5).
- `docs/generate_measure_table.py:17`, `:41`, `:48` and `docs/CONNECTIVITY_METRIC_RANGES.md:11-14`, orientation column — removed (phase 2; `tests/test_docs.py` fails while stale).
- `docs/llm_guide.md:55-56`, `:84-87` (doctested by `tests/test_cookbook.py`) and `docs/cookbook.md:219` "Where to go next" (phase 4 inserts a recipe before it).
- `README.md:104-136` DataArray paragraphs, `:137-146` orientation paragraph, `:148-166` "Choosing parameters"; `docs/index.md:60-92`, `:93-101` mirror them.
- `examples/Intro_tutorial.py:17-27` intro, `:334-370` wrapper section (phase 4); `examples/Tutorial_On_Simulated_Examples.py` (2481 lines, 0 asserts; phase 5). Both are Jupytext-paired (`formats: ipynb,py:percent`); `jupytext` is not in the venv, `uvx jupytext --sync <file.py>` regenerates the `.ipynb`.
- `tests/test_advanced_connectivity.py:97-111` `_lagged_broadband` (used at `:629`, `:662`, `:701`, `:736`, `:901`, `:939`), `:122-165` canonical-coherence simulation, `:268-290` global-coherence simulation; `tests/test_notebooks.py:562-583` `_lagged_broadband_pair` (used at `:591`, `:626`, `:662`, `:700`, `:735`) — replaced by the public simulators (phase 5).
- `tests/test_directed_measures_oracle.py:42-57` `_analytic_var`, `:59-93` `_fourier_coefficients_with_cross_spectrum`, `:96-117` `var_oracle`, `:341-361` Geweke closed form, `:307-319` scalar blockwise, `:417-422` `_CHAIN_COEFFICIENTS` — reused by the block Geweke oracle and the partial-coherence oracle (phase 6).
- `tests/test_list_measures.py:381-405` orientation tests, `tests/test_wrapper.py:799-802` `transpose_output` validation tests, `:2964`; `tests/test_connectivity.py:1421`, `:1437`, `:1813`; `tests/test_notebooks.py:872` and the 4 Granger snapshots in `tests/__snapshots__/test_notebooks.ambr` — all encode the old orientation (phase 2).
- `CHANGELOG.md:13-33` migration table, `:35` Added, `:228` Changed, `:328` Fixed, `:491` Performance — each phase adds its rows/entries here.
- `CLAUDE.md:29-83` Development Commands — phase 7 adds the benchmark command.

## Scope and dependency policy

### Goals

- Master CI green so a 3.0 tag publishes.
- One orientation rule for every directed result in the package: `[..., i, j]` and `sel(source=i, target=j)` both mean `i -> j`.
- Every wrong or missing front-door input produces a WHAT/WHY/HOW error that names the argument.
- A first-time reader of the README runs a successful call before meeting any edge-case policy.
- Every wrapper measure has at least one test that plants a known structure in simulated data and recovers it, plus cross-package agreement where another package computes the same quantity.
- The default measure set gets measurably faster without changing a single output.

### Non-Goals

- Changing `DEFAULT_METHODS`. The set stays; it gets faster.
- Cropping `frequency_range` before computation. The wrapper computes every measure on the full spectrum and crops the output (`wrapper.py:485-491`, `_frequency_bands.py`). Revisit when a user reports a memory or time problem on long single-window epochs; the design would build a one-sided cropped `Connectivity` for the functional measures only, since Wilson factorization needs the full spectrum.
- Speeding up the Granger family. It is 27% of the default run and was optimized in #83/#88.
- Multithreading the phase-lag tiles. Memory-bandwidth bound; measure after phase 7 lands before adding a `workers` knob.
- Deprecation shims of any kind. 3.0 is already a numerical-break release (`CHANGELOG.md:9-11`).
- Splitting `connectivity.py` further or adding measures.
- Exporting the simulators from the package root. They live in `spectral_connectivity.simulate`, documented via `docs/api.rst`'s recursive autosummary.

### Dependency policy

No new runtime dependencies. No new dev dependencies: nitime is already in both dev lists (`pyproject.toml:63`, `:83`), and the mne-connectivity comparison is a recorded fixture generated by a manual script run under `uv run --with mne-connectivity` (user decision, 2026-09-26), so mne never enters `uv.lock`.

## Metrics

- Phase 1: the `Code Quality` job succeeds on the PR and on the next master push; `Build distribution` is no longer skipped.
- Phase 2: on the unidirectional VAR oracle (`test_directed_measures_oracle.py:96-117`), every directed `Connectivity` method puts the planted `0 -> 1` influence at `[..., 0, 1]`, and `multitaper_connectivity(...).sel(source=0, target=1)` equals the same element exactly.
- Phase 3: `multitaper_connectivity(ts, sampling_frequency="1000")` raises `TypeError` whose message contains `sampling_frequency`; `Multitaper(ts)` raises `TypeError` (missing argument); `multitaper_connectivity(ts, sampling_frequency=500, method="canonical_coherence")` raises `ValueError` naming `group_labels`.
- Phase 5: the executed notebook contains at least one `assert` per section and `test_tutorial_notebook_executes` passes.
- Phase 6: each measure listed in phase 6 has a planted-structure test; nitime coherence agrees to the tolerance the executor establishes and records in the test docstring; the mne-connectivity fixture comparison passes at the per-measure tolerances recorded in the fixture generator.
- Phase 7: on the benchmark's case (b) (2000 × 100 × 32, one window), the default set is at least 1.4x faster than the recorded baseline (6.3 s on the reference machine) and the four phase-lag measures at least 2x; all eleven default outputs match the baseline within 1e-12 absolute (`sign` and `absolute` moments exactly).

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Orientation flip silently breaks a user's low-level directed analysis | Migration-guide row with a one-line recipe (`np.swapaxes(old, -1, -2)`); every directed docstring example indexes the new way and is doctested in CI |
| A directed method is missed in the swap (e.g. `time_reversed`, `subset_pairwise`, `blockwise`) | Phase 2's parametrized test iterates `list_measures(directed=True)` minus the lead/lag measures and asserts `[..., 0, 1]` dominates on the oracle |
| Snapshot churn masks a real regression in the notebook tests | Phase 2 asserts the new arrays equal `swapaxes` of the stored ones before `--snapshot-update` |
| The faster phase-lag reductions change results at the rounding edge (near-zero phase lags, NaN observations) | The `imaginary` moment stays a per-observation mean (the CSM shortcut was probed and rejected: wPLI 0.9996 instead of exactly 1 at a `1e-13` rad lag); the `sign` counts carry an `isnan` mask; phase 7 has parity tests for a constant `1e-13` lag, for NaN observations, and the 1e-12 benchmark gate |
| nitime's estimator cannot be matched (per-taper spectra already differ beyond a scale factor while the tapers agree up to sign) | Phase 6 compares at the coherence/CSD level with explicit settings; if no configuration agrees, the test documents the convention difference found and the mne-connectivity fixture carries the coherence cross-check |
| mne-connectivity convention mismatches (signed vs magnitude imaginary coherency, `mt_bandwidth` vs `time_halfbandwidth_product`) | The generator script records the mapping used and the tolerance per measure; any measure that cannot be matched is listed in the fixture's `unmatched` field with the reason rather than dropped silently |
| Removing the 1000 Hz default breaks 20 test call sites in `tests/test_transforms.py` | Listed by line in phase 3; they get `sampling_frequency=1000` explicitly |

## Rollout Strategy

All at once, in the unreleased 3.0. Each phase is a PR on top of master; phases 1 and 2 first, 7 can run in parallel with 4-6. No feature flags. The user tags 3.0 after all seven land (their call, not the executor's).

## Open Questions

1. nitime agreement configuration — deferred to phase 6's first task; the executor records the finding in the test docstring.
2. Whether `MultitaperParameters` keys splat directly into `multitaper_connectivity(**params)` — phase 4 checks `transforms.py:91-120` and writes the README example accordingly.

## Estimated Effort

| Phase | Approx. diff |
| --- | --- |
| 1 | 2 lines |
| 2 | ~250 lines source/docs, ~150 lines tests, snapshot file |
| 3 | ~150 lines source, ~120 lines tests |
| 4 | ~200 lines Markdown, ~120 lines tutorial |
| 5 | ~150 lines `simulate.py`, ~80 lines tests, notebook rewrite (~400 lines touched) |
| 6 | ~400 lines tests, ~120-line generator script, one `.npz` fixture |
| 7 | ~120 lines benchmark, ~100 lines source, ~60 lines tests |
