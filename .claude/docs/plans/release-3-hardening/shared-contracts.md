# Shared contracts

[← back to PLAN.md](PLAN.md)

- [Directed-array orientation](#directed-array-orientation) — owned by phase 2, relied on by phases 4, 5, 6
- [Public simulators](#public-simulators) — owned by phase 5, relied on by phase 6

## Directed-array orientation

Every directed result in the package is indexed **source first**:

- `Connectivity` methods: `result[..., i, j]` is the influence of signal `i` on signal `j` (`i -> j`). Group measures: `result[..., g, h]` is group `g -> h`, groups ordered as the returned `labels`.
- Lead/lag measures keep their existing meaning, which is already source-first: positive `result[..., i, j]` (above 0.5 for `directed_phase_lag_index`) means `i` leads `j`.
- Wrapper: `result.sel(source=a, target=b)` is `a -> b`, unchanged. The wrapper no longer transposes anything; `MeasureInfo` has no `array_orientation`.

Implementation rule (do not weaken): the Wilson-family kernels in `_granger.py` and the transfer-function properties keep their native `[..., target, source]` layout internally, because `_total_inflow` normalizes over sources on axis `-1` and the conditional/blockwise loops write `result[..., target, source]`. The reorder happens exactly once, at the public method's return, through one helper in `connectivity.py`:

```python
def _source_target(native: BackendArray) -> BackendArray:
    """Reorder a ``[..., target, source]`` matrix to ``[..., source, target]``.

    The Wilson-factorized kernels work in the transfer function's native
    layout, where row ``i`` collects the inflow to signal ``i``; every public
    directed measure returns the transpose so that ``[..., i, j]`` reads
    ``i -> j`` like the labeled wrapper's ``sel(source=i, target=j)``.
    """
    return xp.swapaxes(native, -1, -2)
```

Tests that plant a known direction assert on `[..., 0, 1]` for `0 -> 1`.

## Public simulators

Both live in `src/spectral_connectivity/simulate.py` next to `simulate_MVAR` (unchanged), return `float64` NumPy arrays, and take `random_state: int | np.random.Generator | None` resolved by one private helper `_generator(random_state)` shared with `simulate_MVAR`.

```python
def simulate_lagged_broadband(
    lags: Sequence[int],
    noise_levels: float | Sequence[float],
    n_time_samples: int,
    n_trials: int | None = None,
    random_state: int | np.random.Generator | None = None,
) -> NDArray[np.floating]:
    """Noisy copies of one white-noise source, each delayed by whole samples.

    Signal ``k`` is ``source[t - lags[k]] + noise_levels[k] * N(0, 1)``, so a
    signal with a smaller lag leads one with a larger lag by their difference
    in samples. The source is broadband on purpose: a sinusoid delayed by a
    whole number of cycles is indistinguishable from the original and carries
    no lag information for group delay or the phase slope index.

    Returns shape ``(n_time_samples, n_signals)``, or
    ``(n_time_samples, n_trials, n_signals)`` when ``n_trials`` is given.
    """
```

`lags` must be non-negative integers (`lags=(-3, 0)` would slice past the source and stack arrays of different lengths); validate with `ValueError("lags must be non-negative integers; express a lead as a smaller lag on the leading signal, e.g. lags=(0, 3)")`. Semantics that phase 5 must reproduce exactly (they keep `tests/test_notebooks.py`'s snapshots stable): with `lags=(0, L)` the source has `n_time_samples + L` samples, signal 0 is `source[L:]`, signal 1 is `source[:n_time_samples]`, and the noise is `noise_levels * rng.standard_normal(signals.shape)` drawn once after the source (this is bit-identical to the old `rng.normal(0, noise_sd, shape)`). Generally: `source = rng.standard_normal((n_time_samples + max(lags), *extra))`, signal `k` is `source[max_lag - lags[k] : max_lag - lags[k] + n_time_samples]`.

```python
def simulate_shared_oscillation(
    frequency: float,
    sampling_frequency: float,
    n_time_samples: int,
    n_trials: int,
    amplitudes: Sequence[float],
    *,
    phase_offsets: float | Sequence[float] = 0.0,
    noise_levels: float | Sequence[float] = 0.0,
    random_phase_per_trial: bool = True,
    random_state: int | np.random.Generator | None = None,
) -> NDArray[np.floating]:
    """One sinusoid seen by every signal, with per-signal amplitude and phase.

    ``signal[t, r, k] = amplitudes[k] * sin(2 pi frequency t / sampling_frequency
    + phi_r + phase_offsets[k]) + noise_levels[k] * N(0, 1)``. Signal ``k`` leads
    signal ``m`` by ``phase_offsets[k] - phase_offsets[m]`` radians. Each trial
    ``r`` draws an independent uniform phase ``phi_r`` unless
    ``random_phase_per_trial=False`` (then ``phi_r = 0``). Set an amplitude to
    0 to leave a signal out of the oscillation; add two calls (same
    ``random_state`` generator) to give one group a private rhythm.

    Returns shape ``(n_time_samples, n_trials, n_signals)``.
    """
```

`n_signals = len(amplitudes)`; `phase_offsets` and `noise_levels` broadcast to it. The random phase is drawn before the noise.
