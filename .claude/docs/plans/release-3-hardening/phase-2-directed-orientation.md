# Phase 2 — One orientation: `Connectivity` directed arrays read `[..., source, target]`

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#d-orientation-swap-sites)

**Inputs to read first:**

- [src/spectral_connectivity/connectivity.py:420-440](../../../../src/spectral_connectivity/connectivity.py#L420-L440) — class docstring that currently documents two orientations.
- [connectivity.py:3863-4530](../../../../src/spectral_connectivity/connectivity.py#L3863-L4530) — the ten Wilson-family directed methods; the swap sites are tabulated in [designs D](designs.md#d-orientation-swap-sites).
- [connectivity.py:1655-1662](../../../../src/spectral_connectivity/connectivity.py#L1655-L1662) — jackknife docstring's orientation note.
- [src/spectral_connectivity/_granger.py:270-300, 516, 541, 570](../../../../src/spectral_connectivity/_granger.py#L270-L300) — kernels stay native; `_warn_nan_granger_pairs` reads the native array.
- [src/spectral_connectivity/_measure_registry.py:14-125](../../../../src/spectral_connectivity/_measure_registry.py#L14-L125) — `transpose_output` (`:50`), `array_orientation` (`:59-67`), the two validation checks (`:82-89`), `_wilson_directed_spec` (`:93-125`).
- [src/spectral_connectivity/_result_formatting.py:131-134, 185-186, 249-250](../../../../src/spectral_connectivity/_result_formatting.py#L131-L134) — the wrapper's transposes.
- [src/spectral_connectivity/wrapper.py:109-131, 199-205, 681](../../../../src/spectral_connectivity/wrapper.py#L109-L131) — `MeasureInfo.array_orientation` and the docstring sentence.
- [docs/generate_measure_table.py:17, 41, 48](../../../../docs/generate_measure_table.py#L17) and [docs/CONNECTIVITY_METRIC_RANGES.md:11-14](../../../../docs/CONNECTIVITY_METRIC_RANGES.md#L11-L14) — the orientation column; `tests/test_docs.py` fails while the generated file is stale.
- [docs/llm_guide.md:55-56, 84-87](../../../../docs/llm_guide.md#L55-L56) — doctested (`tests/test_cookbook.py`).
- [README.md:137-146](../../../../README.md#L137-L146) and [docs/index.md:93-101](../../../../docs/index.md#L93-L101) — the two-orders paragraph.
- Tests encoding the old orientation: [tests/test_list_measures.py:381-405](../../../../tests/test_list_measures.py#L381-L405), [tests/test_wrapper.py:799-802, 2964](../../../../tests/test_wrapper.py#L799-L802), [tests/test_directed_measures_oracle.py:21, 179, 283, 356-361, 460-463](../../../../tests/test_directed_measures_oracle.py#L341-L361), [tests/test_connectivity.py:1421, 1437, 1813](../../../../tests/test_connectivity.py#L1813), [tests/test_notebooks.py:872](../../../../tests/test_notebooks.py#L872) and the four Granger entries in `tests/__snapshots__/test_notebooks.ambr`.

**Contracts referenced:**

- [Directed-array orientation](shared-contracts.md#directed-array-orientation) — this phase implements it; do not weaken.

**Designs referenced:** [designs.md#d-orientation-swap-sites](designs.md#d-orientation-swap-sites).

## Tasks

- Add `_source_target` (code in the contract) to `connectivity.py` next to `_asnumpy` (`:191`) and apply it at the ten return sites in designs D. Leave `_partial_directed_coherence` and `_transfer_function`-based private helpers native; verify `direct_directed_transfer_function` still composes native quantities before its single final swap. `time_reversed_spectral_granger_prediction` and `subset_pairwise_spectral_granger_prediction` must go through the swap too (`pairs` stays an unordered request).
- Rewrite orientation text: class docstring `:420-440` (one rule, no two families), jackknife `:1655-1662`, and each of the ten method docstrings' Returns sections ("Output `[..., i, j]` is the influence of signal `i` on signal `j` (`i -> j`)") and their doctest examples, which currently index `[..., 1, 0]` for `0 -> 1` (`:3910`, `:3973`, `:4117`, `:4190`, `:4253`, `:4315`, `:4399`, `:4452`, `:4517-4520`). The Notes sections state normalization axes in the old layout and must flip: `directed_transfer_function` at `:4231` says "`result.sum(axis=-1)` is 1" (sum over sources, now axis `-2`), `partial_directed_coherence` at `:4377` says "`result.sum(axis=-2)` is 1" (sum over targets, now axis `-1`); grep `sum(axis=` across the file for any other occurrence, and check `directed_coherence`'s and `generalized_partial_directed_coherence`'s Notes prose says "over sources"/"over targets" consistently with the new axes. Run `uv run pytest --doctest-modules src/spectral_connectivity/connectivity.py` until green.
- `examples/Tutorial_Using_Paper_Examples.py:55-64`: the plotting helper draws `measure[0, :, ind1, ind2]` and titles the panel `f"x{ind2 + 1} → x{ind1 + 1}"`; after this phase that title names the reverse direction. Change it to `f"x{ind1 + 1} → x{ind2 + 1}"`, regenerate the notebook with `uvx jupytext --sync examples/Tutorial_Using_Paper_Examples.py`, and check the panels against the planted structure of Baccalá & Sameshima (2001) example 2 (`:101`, `coefficients[0][i, j]` is the influence `j -> i` in `simulate_MVAR`'s convention): `coefficients[0][2, 0] == 0.0`, so the `x1 → x3` partial-directed-coherence panel must be flat while `x3 → x1` (`coefficients[0][0, 2] == 0.4`) is not; with the old label the flat panel would read `x3 → x1`. Grep both tutorials and `docs/` for `→` / `->` next to index expressions for any other hand-written direction label.
- `_granger.py`: kernels unchanged; add one sentence to the docstrings at `:280`, `:516`, `:570`: "The public `Connectivity` method returns the transpose, `[..., source, target]`."
- Registry: delete `transpose_output` and `array_orientation` from `_MeasureSpec`, the checks at `:82-89`, and the flag in `_wilson_directed_spec` (which then sets only `is_directed=True, requires_two_sided=True`; update its docstring). Delete the `transpose_output` branches in `_result_formatting.py` (`:131-134` collapse to `output_kind = "pairwise"` / `measure_spec.output_kind`; remove `:185-186`, `:249-250`). Delete `array_orientation` from `MeasureInfo` (`wrapper.py:109-112`, `:131`, `:205`) and fix the sentence at `:681`.
- `docs/generate_measure_table.py`: drop `_ORIENTATION` and the column; rewrite the intro of the generated file (`:11-14`) to one sentence stating the single rule; regenerate `docs/CONNECTIVITY_METRIC_RANGES.md` with `uv run python docs/generate_measure_table.py` and confirm `uv run pytest tests/test_docs.py` passes.
- `docs/llm_guide.md:55-56` — replace the `array_orientation` doctest with `granger.is_directed` only; `:84-87` — one rule. `README.md:137-146` and `docs/index.md:93-101` — replace the paragraph with: "For directed measures, `result.sel(source="a", target="b")` is the influence from `a` to `b`, and the lower-level `Connectivity` arrays use the same order: `result[..., i, j]` is `i -> j`. For lead/lag measures a positive value means the source leads."
- Tests: replace `test_array_orientation_matches_the_computed_direction` and `test_only_directed_measures_have_an_array_orientation` (`test_list_measures.py:381-405`) with one parametrized test over `list_measures(directed=True)` that, on the `zero_drives_one` fixture already used there, asserts `result[..., 0, 1].max() > 10 * result[..., 1, 0].max()` for the Wilson family and the documented lead/lag inequality for `directed_phase_lag_index`, `phase_slope_index`, `delay`, `group_delay`. Keep the existing test's special case (`test_list_measures.py:386-387`): `time_reversed_spectral_granger_prediction` is fed `time_series[::-1]`, because time reversal flips the apparent direction; on the reversed data the planted `0 -> 1` again lands at `[..., 0, 1]`. Delete `test_wrapper.py:799-802`. Flip the indices at `test_wrapper.py:2964`, `test_connectivity.py:1421`, `:1437`, `:1813`, `test_notebooks.py:872`, and throughout `tests/test_directed_measures_oracle.py`, which encodes the old layout in 23 indexed expressions: `:356-361` (`granger[:, source, target]`), `:460-463`, `test_non_causal_direction_is_zero` `:141-160` (`non_causal`/`causal` swap to `[..., 1, 0]`/`[..., 0, 1]`), `test_dtf_peak_matches_analytic_transfer_function_peak` `:269-277` (`dtf[:, 0, 2]` against `H[:, 2, 0]`), `test_conditional_granger_removes_mediated_influence` `:467-500` (`pairwise[..., 0, 2]`, `conditional[..., 0, 2]`), `test_time_reversed_granger_flips_unidirectional_oracle` `:504-515` (the reversed system's dominant direction is now `[..., 1, 0]`), `_analytic_directed_measures` `:176-214` (either transpose its outputs or index them as `[target, source]` where compared), and the docstrings at `:21`, `:179`, `:283`. Applying the transpose alone without these edits fails 16 parametrized cases in that module; run `uv run pytest tests/test_directed_measures_oracle.py -q` and fix every failure by re-indexing, never by loosening a tolerance. For the four Granger snapshots: first add a temporary assertion that the new array equals `np.swapaxes(stored, -1, -2)`, run it, then `uv run pytest tests/test_notebooks.py --snapshot-update` and remove the temporary assertion.
- `CHANGELOG.md`: migration-table row — "`Connectivity` Granger and directed-transfer-function methods returned `[..., target, source]` (`result[i, j]` was `j -> i`)" → "All directed `Connectivity` arrays are `[..., source, target]`, matching the wrapper's `sel(source, target)`; `np.swapaxes(old, -1, -2)` converts stored 2.x arrays. `MeasureInfo.array_orientation` is removed." Plus a "Changed" bullet.

## Deliberately not in this phase

- README/docs restructuring beyond the orientation paragraph (phase 4).
- New planted-direction tests on simulated time series beyond what the oracle fixture gives (phase 6).

## Validation slice

| Test | Asserts |
| --- | --- |
| `tests/test_list_measures.py::test_directed_measures_place_source_first[<measure>]` (new, parametrized) | `0 -> 1` planted → `[..., 0, 1]` dominates for every Wilson-family measure (time-reversed Granger on the reversed series); lead/lag measures satisfy their documented sign at `[..., 0, 1]` |
| `tests/test_directed_measures_oracle.py::test_pairwise_granger_matches_geweke_closed_form` | Geweke `0 -> 1` at `granger[:, 0, 1]` within 1e-5; `1 -> 0` at `[:, 1, 0]` ≈ 0 |
| `tests/test_directed_measures_oracle.py::test_wrapper_source_target_labels_follow_causal_direction` | wrapper `sel(source=0, target=1)` equals `Connectivity` `[..., 0, 1]` exactly (`np.testing.assert_array_equal`) |
| `tests/test_directed_measures_oracle.py::test_directed_measure_matches_analytic_closed_form[<measure>]` | closed forms compared after transposing the analytic `[target, source]` matrices |
| `tests/test_directed_measures_oracle.py` (whole module, 12 test functions) | every case passes; `test_non_causal_direction_is_zero` asserts `[..., 1, 0]` ≈ 0 and `[..., 0, 1]` > 0; `test_dtf_peak…` compares `dtf[:, 0, 2]` with `|H[:, 2, 0]|`; `test_conditional…` bounds `conditional[..., 0, 2]`; `test_time_reversed…` finds the flipped influence at `[..., 1, 0]` |
| `tests/test_notebooks.py` Granger tests | pass with updated snapshots that are the exact transpose of the old ones |
| `tests/test_notebooks.py::test_tutorial_notebook_executes[Tutorial_Using_Paper_Examples.ipynb]` (slow) | regenerated paper notebook executes; manual check that the `x1 → x3` PDC panel is flat and `x3 → x1` is not |
| `uv run pytest --doctest-modules src/spectral_connectivity/connectivity.py` | the DTF and PDC Notes' `sum(axis=…)` statements are turned into doctest lines (`bool(np.allclose(dtf.sum(axis=-2), 1))`) so the axis claim is executed, not just written |
| `uv run pytest --doctest-modules src/spectral_connectivity/` and `uv run pytest tests/test_docs.py tests/test_cookbook.py` | docstring examples, generated table, and guides consistent |
| `uv run mypy src/` | clean after the dataclass field removal |

## Fixtures

`zero_drives_one` (already in `tests/test_list_measures.py`) and `var_oracle` / `chain_oracle` (`tests/test_directed_measures_oracle.py:96-117`, `:215-229`).

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind): `transpose_output`, `array_orientation`, `_ORIENTATION`, the formatter's `swapaxes` calls.
- User-facing documentation listed as tasks is updated, not deferred.
