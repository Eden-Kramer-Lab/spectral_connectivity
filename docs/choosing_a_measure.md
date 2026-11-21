# Choosing a measure

Start from the question you are asking of your data. Each row names the
measures that answer it (the `method` names of `multitaper_connectivity`) and
the mistake most often made with them. [Pitfalls](#pitfalls) below covers the
mistakes that apply to every measure, and
[Connectivity Metric Ranges](CONNECTIVITY_METRIC_RANGES.md) gives each
measure's range, units, diagonal, and interpretation.

<!-- Every code block on this page runs as a doctest in tests/test_cookbook.py. -->

| Question | Measures (**try first**) | Watch out for |
| --- | --- | --- |
| Are two signals coupled at a frequency? | **`magnitude_squared_coherence`** (amplitude and phase), `phase_locking_value` (phase only), `pairwise_phase_consistency` (an unbiased squared phase-locking value) | Coherence and the phase-locking value are biased upward when trials x tapers is small, and are 1 with a single observation, whatever the data. |
| Could the coupling be volume conduction or a shared reference (zero-lag mixing)? | `imaginary_coherence`, **`weighted_phase_lag_index`**, `debiased_squared_weighted_phase_lag_index`, `phase_lag_index`, `corrected_imaginary_phase_locking_value` | They are built on the imaginary part of the cross-spectrum, so they ignore all zero-lag coupling and all anti-phase (pi-lag) coupling, genuine or not: a value near 0 does not rule out coupling at either lag. |
| Is the coupling direct, or relayed through other recorded signals? | **`partial_coherence`**; for directed influence, `conditional_spectral_granger_prediction` | They remove only what passes through the signals you recorded. |
| Which signal leads? | From the phase: **`phase_slope_index`**, `directed_phase_lag_index`, `group_delay`, `delay`. Directed influence: `pairwise_spectral_granger_prediction`, `conditional_spectral_granger_prediction`, `directed_transfer_function`, `partial_directed_coherence`, and the rest of their families | The Granger and transfer-function measures need long windows and a two-sided transform (not Morlet wavelets). Check them against `time_reversed_spectral_granger_prediction`: a direction that does not reverse under time reversal suggests instantaneous mixing. `phase_slope_index` is unnormalized, so judge it against a null distribution, and check `group_delay` against its `group_delay_r_value`. |
| How are two groups of signals (two brain areas) coupled? | **`canonical_coherence`** and `canonical_coherency` (include zero-lag coupling), `maximized_imaginary_coherency` and `multivariate_interaction_measure` (these two ignore zero-lag coupling), `blockwise_spectral_granger_prediction` (directed) | Pass `group_labels`, one label per signal. A group pair with more signals than observations has a canonical coherence of 1 for any data (it warns). |
| Is one coherent network shared by all the signals? | **`global_coherence`** | Rescaling one signal changes the result: put the signals on a common scale first. |

For how to read direction and sign, see the
[direction convention](CONNECTIVITY_METRIC_RANGES.md#direction-convention).

## Pitfalls

- **Normalized measures need several observations.** Coherence, phase locking,
  and similar measures average over the observations, trials x tapers. With
  one trial and one taper, coherence is 1 at every frequency whatever the data
  (the package warns). Use several trials or `time_halfbandwidth_product >= 2`
  (3 or more tapers):

```python
>>> import warnings
>>> import numpy as np
>>> from spectral_connectivity import multitaper_connectivity
>>> rng = np.random.default_rng(0)
>>> unrelated = rng.standard_normal((1000, 1, 2))  # one trial of two independent signals
>>> with warnings.catch_warnings():
...     warnings.simplefilter("ignore")  # the warning this pitfall is about
...     coherence = multitaper_connectivity(
...         unrelated,
...         sampling_frequency=500,
...         method="magnitude_squared_coherence",
...         signal_names=["a", "b"],
...         time_halfbandwidth_product=1,  # one taper
...     )
>>> coherence.attrs["n_observations"]
1
>>> bool(np.allclose(coherence.sel(source="a", target="b"), 1.0))
True

```

- **Small samples bias the estimates.** Coherence and the phase-locking value
  are biased upward when trials x tapers is small: for unrelated signals
  coherence averages about `1 / n_observations`, not 0. The debiased measures
  (`pairwise_phase_consistency`, `debiased_squared_*`) are unbiased, so they
  can be slightly negative; that is noise, not negative coupling:

```python
>>> unrelated = rng.standard_normal((1000, 20, 2))  # 20 trials
>>> measures = multitaper_connectivity(
...     unrelated,
...     sampling_frequency=500,
...     method=["magnitude_squared_coherence", "pairwise_phase_consistency"],
...     signal_names=["a", "b"],
... )
>>> measures.attrs["n_observations"]  # 20 trials x 5 tapers
100
>>> pair = measures.sel(source="a", target="b")
>>> round(float(pair.magnitude_squared_coherence.mean()), 2)  # 1 / 100, not 0
0.01
>>> bool(pair.pairwise_phase_consistency.min() < 0)
True

```

- **Coherence is squared.** `magnitude_squared_coherence` is `|coherency|^2`;
  take `abs(coherency)` for the unsquared magnitude. A measure variable's
  `quantity` and `formula` attrs say which form a coherency-family value has.
- **Diagonals differ by measure.** Coherence masks each self-connection with
  NaN (its value would be 1), while `imaginary_coherence` gives 0 and
  `phase_locking_value` gives 1, to within rounding (NaN for a flat,
  zero-power signal). A variable's `diagonal_policy` attr says which, and the
  [diagonal values table](CONNECTIVITY_METRIC_RANGES.md#diagonal-values) lists
  every measure. Before averaging over pairs, mask each pairwise variable `v`
  with `v.where(v.source != v.target)`; a whole Dataset cannot be masked this
  way, because its `power` has no `target` axis:

```python
>>> measures = multitaper_connectivity(
...     unrelated,
...     sampling_frequency=500,
...     method=["magnitude_squared_coherence", "imaginary_coherence", "phase_locking_value"],
...     signal_names=["a", "b"],
... )
>>> self_connection = measures.sel(source="a", target="a")
>>> {name: self_connection[name].attrs["diagonal_policy"] for name in self_connection.data_vars}
{'magnitude_squared_coherence': 'masked_nan', 'imaginary_coherence': 'constant', 'phase_locking_value': 'constant'}
>>> bool(self_connection.magnitude_squared_coherence.isnull().all())
True
>>> bool(np.allclose(self_connection.imaginary_coherence, 0))
True
>>> bool(np.allclose(self_connection.phase_locking_value, 1))
True
>>> locking = measures.phase_locking_value
>>> locking.where(locking.source != locking.target).mean(["source", "target"]).dims
('time', 'frequency')

```

- **Directed measures need a two-sided spectrum.** The Granger and
  directed-transfer-function families work with `multitaper_connectivity`, but
  not with Morlet wavelets or other one-sided coefficients, which raise a
  `ValueError`.
- **Directed measures need long windows.** They come from a factorization of
  the estimated spectrum. A window shorter than the slowest correlation in the
  data gives unreliable values with no warning, and zero-padding does not
  repair that. The values also depend slightly on the FFT length; padding
  (`transform_kwargs={"n_fft_samples": n}`) removes that dependence but, in
  simulation, not the error against the true values.

The [Guide for AI coding assistants](llm_guide.md) lists the pitfalls of the
programming interface (deprecated spellings, group labels, band summaries,
frequency ranges, GPU setup).
