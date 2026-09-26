# Phase 4 — Front door: README, docs index, cookbook recipe, tutorial order

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

**Inputs to read first:**

- [README.md:60-136](../../../../README.md#L60-L136) — "Usage Example": one code block, then the `list_measures` block, the cookbook pointer, the band block, and (`:104-136`) fifteen sentences on DataArray dimension inference, label types, datetime coordinates and dask. [README.md:148-166](../../../../README.md#L148-L166) — "Choosing parameters" with `suggest_parameters`, currently below the DataArray material.
- [docs/index.md:37-92](../../../../docs/index.md#L37-L92) — the same material, slightly shorter; keep the two files saying the same thing.
- [docs/cookbook.md](../../../../docs/cookbook.md) — recipes at `:24`, `:57`, `:79`, `:95`, `:116`, `:137`, `:157`, `:178`, and "Where to go next" at `:219`; every block is doctested by `tests/test_cookbook.py`.
- [examples/Intro_tutorial.py:17-27](../../../../examples/Intro_tutorial.py#L17-L27) — intro says "two main classes … There is also a function"; `:27-333` walk `Multitaper` then `Connectivity`; `:334-370` finally show `multitaper_connectivity`. The package docstring (`src/spectral_connectivity/__init__.py:6-9`) says to start with the wrapper. The `.ipynb` is Jupytext-paired; regenerate with `uvx jupytext --sync examples/Intro_tutorial.py`.
- [src/spectral_connectivity/transforms.py:91-120](../../../../src/spectral_connectivity/transforms.py#L91-L120) — `MultitaperParameters` TypedDict returned by `suggest_parameters` (`:257`); check which keys `multitaper_connectivity(**params)` would accept before writing the README example (open question 2 in the overview).
- Observed 2026-09-26: a 2-D `(n_time, n_signals)` array works directly in the wrapper (a singleton trial axis is inserted at `wrapper.py:766-767`), so the tutorial's `prepare_time_series` step is not needed on the quick-start path.

**Contracts referenced:**

- [Directed-array orientation](shared-contracts.md#directed-array-orientation) — phase 2 already rewrote the orientation paragraph; this phase leaves that paragraph alone.

## Tasks

- README "Usage Example" restructure (mirror in `docs/index.md`):
  1. Keep the first `multitaper_connectivity` block as is.
  2. Immediately after it, move "Choosing parameters" (`:148-166`) up as a short paragraph plus block. If `MultitaperParameters`' keys are all `Multitaper` keyword arguments, show `multitaper_connectivity(time_series, sampling_frequency=1000, method="coherence_magnitude", **params)`; otherwise pass `time_halfbandwidth_product=params["time_halfbandwidth_product"], time_window_duration=params["time_window_duration"]` explicitly.
  3. Keep `list_measures`, the multi-measure Dataset, and the frequency-bands block.
  4. Replace `:104-136` with three sentences: "`time_series` may also be an `xarray.DataArray`. Dimension names, not positions, define the roles: common names for time, trial and signal dimensions are recognized, `sampling_frequency` is inferred from a numeric `time` coordinate in seconds, and the signal index becomes the `source`/`target` labels. For other dimension names pass `time_dim`, `trial_dim` and `signal_dim`; the cookbook's DataArray recipe has the full contract (label types, datetime coordinates, dask)." Link to the recipe.
  5. Add one sentence after the wrapper block: "`**kwargs` configure the transform (window, tapers); `connectivity_kwargs` configure the measure; group measures take `group_labels`."
- Cookbook: add "## Pass a labeled DataArray" before "Where to go next" (`:219`), doctested: build `xr.DataArray(rng.standard_normal((1000, 4, 3)), dims=("time", "trial", "signal"), coords={"time": np.arange(1000) / 500, "signal": ["CA1", "CA3", "PFC"]})`, call `multitaper_connectivity(da, method="coherence_magnitude")` with no `sampling_frequency`, show `coherence.attrs["mt_sampling_frequency"]` is `500.0` and `coherence.source.values.tolist()`; then a second block with dims `("t", "epoch", "channel")` passed via `time_dim="t", trial_dim="epoch", signal_dim="channel"`. Carry the removed README sentences (ambiguity raises, elimination warns, datetime conversion snippet, dask rejection, label constraints) into this recipe's prose, condensed to a bullet list.
- Tutorial: rewrite `Intro_tutorial.py:17-27` to say the package has one entry point, `multitaper_connectivity`, and two classes underneath. Insert a "## Quick start" section right after it that reuses the existing simulated 200 Hz pair (`:44-58`) on the 2-D array directly: one call with `method="coherence_magnitude"`, `coherence.sel(source="0", target="1").plot(x="frequency")`, and `list_measures(default_only=True)`. Retitle `:27` to "## Under the hood: Multitaper" and `:188` to "## Under the hood: Connectivity". Fold `:334-370` into a "### Time-resolved connectivity with the wrapper" subsection placed after "Adding in time" (`:284`) so the wrapper appears twice with distinct purposes rather than as an afterthought. Regenerate the notebook with `uvx jupytext --sync examples/Intro_tutorial.py`.
- Put the intro notebook under test: `tests/test_notebooks.py:1013-1015` parametrizes `test_tutorial_notebook_executes` over only `Tutorial_On_Simulated_Examples.ipynb` and `Tutorial_Using_Paper_Examples.ipynb`; add `Intro_tutorial.ipynb` there, fix the module docstring at `:13-14` ("the two tutorial notebooks"), and add `examples/Intro_tutorial.ipynb` to the shell loop in `.github/workflows/release.yml:110` (the "Test notebooks" step) so CI executes it too.
- `CHANGELOG.md` "Changed": one bullet, "README and tutorial lead with `multitaper_connectivity`; the DataArray contract moved to a cookbook recipe."

## Deliberately not in this phase

- Adding asserts to or restructuring `Tutorial_On_Simulated_Examples.py` (phase 5, which rewrites it on the simulators).
- Touching `docs/llm_guide.md` beyond what phase 2 did; it is a reference for assistants, not the front door.
- Changing the orientation paragraph (phase 2 owns it).

## Validation slice

| Test | Asserts |
| --- | --- |
| `uv run pytest tests/test_cookbook.py` | new DataArray recipe doctests pass, including inferred `500.0` and the custom-dimension call |
| `uv run pytest tests/test_notebooks.py -k "executes and Intro" -m slow` (slow) | regenerated `Intro_tutorial.ipynb` executes end to end (the notebook must be in the parametrize list first; `--collect-only -q` shows the id) |
| manual: `uvx --from myst-parser python -c ...` is not needed; run `make -C docs html` (or `uv run sphinx-build -b html docs docs/_build/html`) | no new warnings from `index.md` |
| README diff review | the first successful call appears before any DataArray or orientation policy; the file shrinks by at least 20 lines |

## Fixtures

None beyond the doctest inputs written inline in the cookbook.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind): the long DataArray paragraphs exist only in the cookbook now.
- User-facing documentation listed as tasks is updated, not deferred.
