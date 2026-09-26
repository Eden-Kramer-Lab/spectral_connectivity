# Phase 3 — Actionable input errors, no silent 1000 Hz, a named `group_labels`

[← back to PLAN.md](PLAN.md) · [overview](overview.md) · [designs](designs.md#b-group_labels-routing-in-the-wrapper)

**Inputs to read first:**

- [src/spectral_connectivity/transforms.py:46-66](../../../../src/spectral_connectivity/transforms.py#L46-L66) — `_validate_sampling_frequency`, called at `:970`, `:1779`, `:1909`, `:2151`; `np.isfinite("1000")` here is the raw `ufunc 'isfinite'` TypeError a user sees today.
- [transforms.py:860, 1769, 1900](../../../../src/spectral_connectivity/transforms.py#L860) — `sampling_frequency: float = 1000` on `Multitaper`, `ShortTimeFourierTransform`, `Welch`; docstrings at `:640`, `:1740`, `:1871`. `MorletWavelet` (`:2130`) already requires it.
- [src/spectral_connectivity/_input_handling.py:352-361](../../../../src/spectral_connectivity/_input_handling.py#L352-L361) — the DataArray path's own `isfinite` check.
- [src/spectral_connectivity/wrapper.py:502-519, 584-620](../../../../src/spectral_connectivity/wrapper.py#L502-L519) and [:801-825, 856-870](../../../../src/spectral_connectivity/wrapper.py#L801-L825) — the two public signatures and their `squeeze`/`connectivity_kwargs`/`**kwargs` docs; [:412-499](../../../../src/spectral_connectivity/wrapper.py#L412-L499) `_format_and_reduce_measures`.
- [src/spectral_connectivity/_result_formatting.py:37-58, 94-112](../../../../src/spectral_connectivity/_result_formatting.py#L37-L58) — kwargs check and measure invocation.
- [src/spectral_connectivity/_measure_registry.py:450-453](../../../../src/spectral_connectivity/_measure_registry.py#L450-L453) — `_requires_two_sided`, the pattern for `_is_group_measure`.
- Call sites that rely on the 1000 Hz default (all in `tests/test_transforms.py`, found by AST scan): lines 224, 326, 333, 341, 355, 373, 653, 658, 725, 761, 769, 774, 778, 896, 900 and five more in the same file; re-run the scan (`ast.walk` for `Call` nodes named `Multitaper`/`Welch`/`ShortTimeFourierTransform` without a `sampling_frequency` keyword) to get the complete list.
- Observed today: `multitaper_connectivity(ts, sampling_frequency=1000, method="canonical_coherence")` raises bare `TypeError: Connectivity.canonical_coherence() missing 1 required positional argument: 'group_labels'`.

**Contracts referenced:** none.

**Designs referenced:** [designs.md#b-group_labels-routing-in-the-wrapper](designs.md#b-group_labels-routing-in-the-wrapper), [designs.md#c-sampling_frequency-validation](designs.md#c-sampling_frequency-validation).

## Tasks

- Replace `_validate_sampling_frequency` with the type-checking version in designs C; make `_input_handling.py:352-361` call it (import from `transforms`), keeping its own `None` handling. Confirm the import introduces no cycle by importing the package.
- Remove the `= 1000` defaults at `transforms.py:860`, `:1769`, `:1900` (parameter stays positional-or-keyword, now required); update the three docstrings to `sampling_frequency : float` with the sentence "Samples per second; required because it labels the frequency axis and scales power." Add `sampling_frequency=1000` at every call site the AST scan lists. Check `examples/*.py`, `docs/*.md` and the docstring examples with the same scan (the scan found none, but re-run after the change and run the doctests).
- Wrapper: add `group_labels` to both signatures, thread it into `_format_and_reduce_measures`, implement the routing and both error messages from designs B, add `_is_group_measure` to the registry, and rewrite the three docstring entries (`group_labels`, `connectivity_kwargs`, `**kwargs`) in both functions. Extend `_check_method_accepts_kwargs`'s message with the transform-settings hint.
- Cookbook (`docs/cookbook.md`): add a doctested recipe "## Group measures: canonical coherence between areas" between "Compute several measures at once" (`:116`) and "Collapse into frequency bands" (`:137`) showing `multitaper_connectivity(time_series, sampling_frequency=500, method="canonical_coherence", group_labels=["CA1", "CA1", "PFC"])` and the resulting `dims` `('time', 'frequency', 'source_group', 'target_group')`.
- `CHANGELOG.md`: migration rows — "`Multitaper`, `ShortTimeFourierTransform`, `Welch` defaulted `sampling_frequency` to 1000 Hz" → "`sampling_frequency` is required; pass your rate"; "group labels went through `connectivity_kwargs={"group_labels": ...}`" → "pass `group_labels=...` directly (the dict form still works for other measure arguments)". "Added" bullets for the typed error and the named argument.

## Deliberately not in this phase

- README/tutorial text about the two kwargs (phase 4 rewrites the front door; the docstrings are this phase's deliverable).
- Validating `group_labels` length/contents beyond what `Connectivity._validated_group_indices` already does.

## Validation slice

| Test | Asserts |
| --- | --- |
| `tests/test_error_messages.py::test_sampling_frequency_string_names_the_argument` | `multitaper_connectivity(ts, sampling_frequency="1000")` raises `TypeError` matching `sampling_frequency must be a number` and containing `'1000'` |
| `tests/test_error_messages.py::test_sampling_frequency_bool_is_rejected` | `Multitaper(ts, sampling_frequency=True)` raises `TypeError` |
| `tests/test_wrapper.py::test_dataarray_path_rejects_string_sampling_frequency` | a DataArray input with `sampling_frequency="500"` raises the same `TypeError` (covers `_input_handling`) |
| `tests/test_transforms.py::test_transforms_require_sampling_frequency[Multitaper\|Welch\|ShortTimeFourierTransform]` | constructing without it raises `TypeError` (missing argument) |
| `tests/test_wrapper.py::test_group_measure_without_group_labels_explains_the_argument` | `ValueError` matching `needs group_labels` and containing `group_labels=` |
| `tests/test_wrapper.py::test_group_labels_with_only_pairwise_measures_is_rejected` | `method="coherence_magnitude", group_labels=[0, 1]` raises `ValueError` matching `none of the requested measures compares groups` |
| `tests/test_wrapper.py::test_group_labels_reach_every_group_measure_in_a_batch` | `method=["canonical_coherence", "coherence_magnitude"], group_labels=[0,0,1,1]` returns a Dataset whose `canonical_coherence` has `source_group` size 2 and whose `coherence_magnitude` has `source` size 4 |
| `tests/test_wrapper.py::test_group_labels_recorded_in_provenance` | `result["canonical_coherence"].attrs["arg_group_labels_json"]` round-trips the labels |
| `tests/test_wrapper.py::test_connectivity_kwargs_group_labels_still_works` | the dict form alone (no `group_labels=` argument) produces an identical result (`xr.testing.assert_identical` ignoring `measure_kwargs_json`); the caller's dict is not mutated |
| `tests/test_wrapper.py::test_group_labels_given_twice_is_rejected` | `group_labels=[0,0,1,1]` together with `connectivity_kwargs={"group_labels": [0,1,0,1]}` raises `ValueError` matching `pass it once` |
| `tests/test_cookbook.py` | the new recipe's doctest passes |
| full suite, `uv run mypy src/` | green |

## Fixtures

Reuse the module-level `time_series` pattern of `tests/test_wrapper.py` (seeded noise, shape `(1000, 4, 4)`); a four-signal array so `group_labels=[0, 0, 1, 1]` is valid.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind): the duplicated `isfinite` check in `_input_handling.py`, the three `= 1000` defaults.
- User-facing documentation listed as tasks is updated, not deferred.
