# Designs

[← back to PLAN.md](PLAN.md)

- [A. Phase-lag moments in one pass](#a-phase-lag-moments-in-one-pass) (phase 7)
- [B. `group_labels` routing in the wrapper](#b-group_labels-routing-in-the-wrapper) (phase 3)
- [C. `sampling_frequency` validation](#c-sampling_frequency-validation) (phase 3)
- [D. Orientation swap sites](#d-orientation-swap-sites) (phase 2)

## A. Phase-lag moments in one pass

### What the current code does

`Connectivity._imaginary_cross_spectrum_moments` (`connectivity.py:3383-3482`) forms, one source-row tile at a time, `Im(X_i conj X_j) = Im_i Re_j - Re_i Im_j` over every observation and non-negative frequency, applies one of four functions (`_IMAGINARY_MOMENTS`, `:86-96`: `sign`, identity, `abs`, square) and averages with `_expectation`. It computes only the keys the caller asks for and caches them (`_imaginary_moment_cache`, `:3355`). For a large case the workspace cap (`PHASE_LAG_INDEX_MAX_WORKSPACE_ELEMENTS = 16_000_000`, `:163`) makes the tile a single source row.

Measured on 2000 × 60 × 24 (5 tapers, 1001 bins), CPU, this session:

| Call pattern | Time |
| --- | --- |
| default-set order: `("sign","absolute")` then `("imaginary","squared","absolute")` — two tile passes | 1.39 s |
| all four keys in one call — one tile pass | 0.99 s |
| prototype of the loop below, but without the `imaginary` mean (it took `E[Im]` from the CSM, which the design below rejects) | 0.64 s |

Prototype outputs vs current: `sign` and `absolute` max abs diff 0.0, `squared` 3.2e-22. Adding the `imaginary` mean back costs one more single-pass reduction per tile (about 10% of the prototype's time), so expect roughly 0.7 s here, still about 2x the current default-set order.

### Why it is slow

Every tile is ~128 MB at case (b) sizes; the work is memory traffic. Each formation costs ~5 tile passes (two products, one subtraction, fresh allocations), each reduction 2 passes (materialize `f(tile)`, then `mean`). Re-forming the tile for a later key set is the largest avoidable cost, followed by fresh allocation of temporaries per tile.

### New design

1. **Compute all four moments at first use.** Whenever any key is missing, compute `sign`, `imaginary`, `absolute`, `squared` in one pass and cache them. A lone `phase_lag_index` call pays two extra single-pass reductions (~15-20%); every multi-measure call saves a full formation. This replaces the "only requested keys" rationale in the docstring at `:3405-3410`; rewrite it to say the formation dominates and the three reductions together cost less than a second formation.
2. **Keep the `imaginary` moment from the tile; do not take it from the cross-spectral matrix.** Mathematically `E[Im(X_i conj X_j)] = Im(E[X_i conj X_j])`, but `weighted_phase_lag_index` divides `E[Im]` by `E[|Im|]`, and at a near-zero phase lag both are tiny differences of large products. Today numerator and denominator come from the *same* per-observation numbers, so a constant tiny lag gives exactly ±1; the complex matmul that builds the CSM accumulates differently, and its imaginary part carries rounding of order `eps * |X|^2` relative to a signal of order `phi * |X|^2`. Probed with a constant `1e-13` rad offset (above the `_has_no_phase_lag` guard, `_ZERO_PHASE_LAG_EPSILONS = 16` at `:173`): current wPLI is exactly 1.0, the CSM shortcut gives 0.9996 to 1.0001. So `imaginary` stays a single-pass `mean` of the tile on the unweighted path and `_expectation(imaginary)` on the weighted path. All four keys stay in `_IMAGINARY_MOMENTS` and `_IMAGINARY_MOMENT_PAIR_SYMMETRY`.
3. **Reused buffers, no per-tile allocation.** Two float buffers of the maximal tile shape allocated once; the last partial block slices them.
4. **Single-pass reductions on the unweighted path.** `sign` from counts of positive and negative entries (bool tiles are 1/8 the traffic of float tiles), `imaginary` via `mean`, `squared` via `einsum` over the observation axes, `absolute` via `xp.abs(tile, out=buffer2)` then `mean`. **NaN must propagate as it does today**: `xp.mean(xp.sign(tile))` is NaN wherever any observation is NaN (Morlet `edge_mode="nan"` produces such coefficients; probed on master, one NaN observation makes `phase_lag_index` NaN at that bin), but `tile > 0` and `tile < 0` are both False for NaN, so the counts alone would silently treat a missing observation as a zero-sign one. The `einsum` and `abs`/`mean` reductions propagate NaN on their own. So the unweighted path computes `invalid = xp.any(xp.isnan(imaginary), axis=observation_axes)` once per tile (one more bool pass) and writes NaN into the `sign` moment where `invalid` is true. The weighted path (`self._observation_weights is not None`, only Morlet with edge weights today) keeps materializing `f(tile)` and calling `_expectation`, because the weights need the per-observation values. Both paths share formation.

5. **One injectable reduction step.** Both paths call a private method `_reduce_phase_lag_tile(imaginary, moments, start, stop)` that writes the four moments of one tile; the failure-recovery test patches it to raise, replacing the current test's `_expectation` injection (which the unweighted path no longer routes through).

Reference implementation of the tile loop (replace `:3423-3480`; keep the block-size computation and the lower-triangle fill; the body of the `if/else` below is `_reduce_phase_lag_tile`):

```python
real = xp.ascontiguousarray(coefficients.real)
imag = xp.ascontiguousarray(coefficients.imag)
tile_shape = (*coefficients.shape[:-1], signals_per_block, n_signals)
tile = xp.empty(tile_shape, dtype=real_dtype)
scratch = xp.empty(tile_shape, dtype=real_dtype)
observation_axes = tuple(self._expectation_axes)
n_observations = self.n_observations  # only used on the unweighted path
for start in range(0, n_signals, signals_per_block):
    stop = min(n_signals, start + signals_per_block)
    width = stop - start
    imaginary = tile[..., :width, start:]
    other = scratch[..., :width, start:]
    xp.multiply(imag[..., start:stop, xp.newaxis], real[..., xp.newaxis, start:], out=imaginary)
    xp.multiply(real[..., start:stop, xp.newaxis], imag[..., xp.newaxis, start:], out=other)
    xp.subtract(imaginary, other, out=imaginary)
    local_diagonal = xp.arange(width)
    imaginary[..., local_diagonal, local_diagonal] = 0
    if self._observation_weights is None:
        positive = xp.count_nonzero(imaginary > 0, axis=observation_axes)
        negative = xp.count_nonzero(imaginary < 0, axis=observation_axes)
        invalid = xp.any(xp.isnan(imaginary), axis=observation_axes)
        moments["sign"][..., start:stop, start:] = xp.where(
            invalid, xp.nan, (positive - negative) / n_observations
        )
        moments["imaginary"][..., start:stop, start:] = xp.mean(imaginary, axis=observation_axes)
        moments["squared"][..., start:stop, start:] = (
            xp.einsum(_SQUARED_SUBSCRIPTS, imaginary, imaginary) / n_observations
        )
        xp.abs(imaginary, out=other)
        moments["absolute"][..., start:stop, start:] = xp.mean(other, axis=observation_axes)
    else:
        for key, reduced in moments.items():
            reduced[..., start:stop, start:] = self._expectation(_IMAGINARY_MOMENTS[key](imaginary))
```

`_SQUARED_SUBSCRIPTS` contracts the three observation axes and keeps the rest; the executor derives it from `self._expectation_axes` (e.g. for `expectation_type="trials_tapers"` the kept axes are time, frequency, row, column: `"tabfrc,tabfrc->tfrc"`). Build the subscript string once per call from `_expectation_axes`. On CuPy every call used here exists (`count_nonzero` with a tuple axis, `einsum`, ufunc `out=`).

`n_observations` must be the count over the averaged axes actually present in `coefficients` (the existing `self.n_observations` property; verify it matches `_expectation`'s denominator on the unweighted path in the parity test).

### Benchmark script

`benchmarks/bench_default_measures.py` (new directory; not under `testpaths`, so pytest ignores it). Two fixed cases, seed 0:

- (a) `n_time=5000, n_trials=50, n_signals=8, time_window_duration=0.5`
- (b) `n_time=2000, n_trials=100, n_signals=32, time_window_duration=None`

For each: time `Multitaper(...).fft()`, `method="coherence_magnitude"`, each default measure alone (fresh `Connectivity`), and the full default set; report seconds and `resource.getrusage(RUSAGE_SELF).ru_maxrss`. `--save PATH` writes every default measure's array for both cases into one `.npz`; `--compare PATH` loads it and prints the max abs difference per measure, exiting non-zero above 1e-12. `uv run python benchmarks/bench_default_measures.py --save baseline.npz` is the baseline task; the same with `--compare` is the parity gate. Document the command in `CLAUDE.md` under Development Commands.

## B. `group_labels` routing in the wrapper

Group measures are the specs whose `output_kind` is `"group_pairwise"` (`canonical_coherence`, `maximized_imaginary_coherency`, `multivariate_interaction_measure`, `blockwise_spectral_granger_prediction`) or `"multivariate_components"` (`canonical_coherency`, `maximized_imaginary_coherency_components`). All six take a parameter literally named `group_labels` (`connectivity.py:2396`, `:2532`, `:2644`, `:2866`, `:2932`, `:4152`).

Signature change in both `multitaper_connectivity` (`wrapper.py:502-519`) and `fourier_connectivity` (`:801-825`): add, in the keyword-only block, `group_labels: Sequence[Hashable] | NDArray[Any] | None = None`, and thread it into `_format_and_reduce_measures` (`:412-426`) as a keyword.

In `_format_and_reduce_measures`, before the single/multi branch at `:448`:

```python
group_methods = [m for m in methods if _is_group_measure(m)]
if group_labels is not None and not group_methods:
    msg = (
        "group_labels was given, but none of the requested measures compares "
        f"groups of signals: {methods!r}. Request a group measure (see "
        "list_measures(category='group_pairwise') or "
        "list_measures(category='multivariate_components')) or drop group_labels."
    )
    raise ValueError(msg)
if group_methods and group_labels is None and "group_labels" not in connectivity_kwargs:
    n_signals = connectivity.n_signals
    msg = (
        f"{group_methods[0]} compares groups of signals and needs group_labels: "
        f"one label per signal ({n_signals} here) naming the group it belongs "
        "to, e.g. group_labels=['CA1', 'CA1', 'PFC', 'PFC'].\n"
        "Pass it as group_labels=... (the same labels apply to every group "
        "measure in this call)."
    )
    raise ValueError(msg)
```

`_is_group_measure(name)` is a one-line registry helper next to `_requires_two_sided` (`_measure_registry.py:450-453`). Resolve the labels once, before the checks above, so the dictionary form keeps working and a conflicting double specification is rejected rather than silently overwritten:

```python
legacy_labels = connectivity_kwargs.pop("group_labels", None)  # on a copy of the mapping
if group_labels is not None and legacy_labels is not None:
    msg = (
        "group_labels was given both as an argument and inside connectivity_kwargs; "
        "pass it once, as group_labels=..."
    )
    raise ValueError(msg)
group_labels = group_labels if group_labels is not None else legacy_labels
```

`connectivity_kwargs` must be copied (`dict(connectivity_kwargs or {})`) before the `pop`, since the caller's mapping is theirs. When formatting each method, the kwargs passed to `_connectivity_result_to_xarray` are `{**connectivity_kwargs, "group_labels": group_labels}` for group measures and the (now label-free) `connectivity_kwargs` otherwise, so the existing `_check_method_accepts_kwargs` (`_result_formatting.py:37-58`) no longer sees `group_labels` reach a pairwise measure. Provenance: `group_labels` lands in `attrs` through the existing `arg_<key>` mechanism (`:135-138`), which already handles sequences via the JSON path.

Docstring block that replaces `:594-620` (same text, adapted, at `:856-870`):

```
group_labels : sequence, optional
    One label per signal naming the group it belongs to; required by the
    group measures (``canonical_coherence``, ``canonical_coherency``,
    ``maximized_imaginary_coherency``, ``multivariate_interaction_measure``,
    ``blockwise_spectral_granger_prediction`` and
    ``maximized_imaginary_coherency_components``) and rejected otherwise.
connectivity_kwargs : dict, optional
    Extra keyword arguments for the *measure* (a ``Connectivity`` method),
    e.g. ``pairs`` for ``subset_pairwise_spectral_granger_prediction`` or
    ``n_components`` for ``canonical_coherency``; passed to every requested
    measure. Transform settings do not go here (see ``**kwargs``).
**kwargs
    Extra keyword arguments for the *transform* (``Multitaper``), e.g.
    ``time_halfbandwidth_product``, ``n_tapers``, ``time_window_step``,
    ``taper_weighting``. Measure settings do not go here (see
    ``connectivity_kwargs``).
```

`_check_method_accepts_kwargs` (`_result_formatting.py:37-58`) is shared by both wrappers, and only `multitaper_connectivity` takes transform keyword arguments (`fourier_connectivity`'s signature at `wrapper.py:801-825` has no `**kwargs`), so its message (`:49-56`) gains a hint that is true for both: "connectivity_kwargs configures the measure only. Transform settings such as time_halfbandwidth_product belong to the transform: pass them to multitaper_connectivity directly, or set them on the transform that produced the coefficients you give fourier_connectivity."

## C. `sampling_frequency` validation

Replace `_validate_sampling_frequency` (`transforms.py:46-66`) with a type check first:

```python
def _validate_sampling_frequency(sampling_frequency: Any) -> None:
    """Raise an actionable error for a non-numeric, non-finite or non-positive rate."""
    if isinstance(sampling_frequency, bool) or not isinstance(
        sampling_frequency, (int, float, np.integer, np.floating)
    ):
        msg = (
            "sampling_frequency must be a number (samples per second), got "
            f"{type(sampling_frequency).__name__} {sampling_frequency!r}.\n"
            "\n"
            "It labels the frequency axis and scales power, so it cannot be "
            "inferred from a plain array.\n"
            "\n"
            "Pass it as a number, e.g. sampling_frequency=1000 rather than '1000'."
        )
        raise TypeError(msg)
    if not np.isfinite(sampling_frequency) or sampling_frequency <= 0:
        ...  # existing message unchanged
```

`_input_handling.py:352-361` calls this function instead of its own `np.isfinite` check (import from `transforms`; there is no circular import since `_input_handling` already imports nothing from `transforms` that imports it back — verify with `uv run python -c "import spectral_connectivity"`). The wrapper's `None` check at `wrapper.py:750-756` stays as is (it runs first for array input).

## D. Orientation swap sites

Apply `_source_target` (see [shared-contracts](shared-contracts.md#directed-array-orientation)) exactly at these returns in `connectivity.py`; the private device-native helpers stay native because `direct_directed_transfer_function` composes them:

| Method (line of `def`) | Native producer | Return becomes |
| --- | --- | --- |
| `pairwise_spectral_granger_prediction` (3863) via `_pairwise_spectral_granger` (3918) | `_estimate_spectral_granger_prediction` | `return _source_target(result)` after `_warn_nan_granger_pairs(result, measure)` |
| `subset_pairwise_spectral_granger_prediction` (3941) | its kernel call | swap the array it returns (NaN-filled pairs stay NaN) |
| `time_reversed_spectral_granger_prediction` (3997) | `_pairwise_spectral_granger(time_reversed=True)` | covered by the shared helper |
| `conditional_spectral_granger_prediction` (4051) | `_granger` conditional loop (`_granger.py:541`) | swap at return |
| `blockwise_spectral_granger_prediction` (4151) | `_estimate_blockwise_spectral_granger` | `return to_numpy(_source_target(result)), to_numpy(labels)` (4209) |
| `directed_transfer_function` (4212) | `_squared_magnitude(H / _total_inflow(H))` (4255-4257) | wrap in `_source_target` |
| `directed_coherence` (4263) | expression at 4340-4345 | wrap |
| `partial_directed_coherence` (4357) | `self._partial_directed_coherence()` (4406) | `return _source_target(self._partial_directed_coherence())`; the private method stays native |
| `generalized_partial_directed_coherence` (4407) | expression at 4400-4404 | wrap |
| `direct_directed_transfer_function` (4465) | final product after 4500 | wrap the final expression only |

`_warn_nan_granger_pairs` (`_granger.py:270-300`) reads native `[..., target, source]` and prints `source -> target`; it keeps receiving the native array, so its message stays right.
