# Guide for AI coding assistants

This page is for AI coding assistants (and the people prompting them) writing
code that uses `spectral_connectivity`. It covers the one workflow to use, the
conventions that are easy to get wrong, and where to look things up. Every code
block runs as a doctest in the test suite (`tests/test_cookbook.py`).

## The workflow

Use `multitaper_connectivity`. It takes the time series and returns a labeled
`xarray` object; you do not need `Multitaper` or `Connectivity` unless you need
something the wrapper does not expose.

- Input shape is `(n_time_samples, n_trials, n_signals)`, or
  `(n_time_samples, n_signals)` for a single trial. An `xarray.DataArray` with
  named dimensions also works.
- `sampling_frequency` is in Hz. All durations are in seconds.
- One `method` name returns a `DataArray`; a list of names, or no `method`,
  returns a `Dataset` with one variable per measure.

```python
>>> import numpy as np
>>> from spectral_connectivity import list_measures, multitaper_connectivity
>>> rng = np.random.default_rng(0)
>>> leader = rng.standard_normal((1003, 20))
>>> # Signal "b" is signal "a" delayed by 3 samples (6 ms at 500 Hz), plus noise.
>>> time_series = np.stack([leader[3:], leader[:-3]], axis=-1)
>>> time_series += 0.5 * rng.standard_normal(time_series.shape)
>>> time_series.shape  # (n_time_samples, n_trials, n_signals)
(1000, 20, 2)
>>> coherence = multitaper_connectivity(
...     time_series,
...     sampling_frequency=500,
...     method="coherence_magnitude",
...     signal_names=["a", "b"],
... )
>>> coherence.dims
('time', 'frequency', 'source', 'target')
>>> bool(coherence.sel(source="a", target="b").mean() > 0.5)
True

```

## Find measures with `list_measures()`

Do not guess method names: `list_measures()` returns one `MeasureInfo` per
valid `method`, with what its values mean.

```python
>>> granger = next(
...     m for m in list_measures() if m.name == "pairwise_spectral_granger_prediction"
... )
>>> granger.value_range, granger.units, granger.dims
((0.0, inf), '1', ('time', 'frequency', 'source', 'target'))
>>> granger.is_directed, granger.array_orientation
(True, 'target_source')
>>> [m.name for m in list_measures(directed=True)][:3]
['pairwise_spectral_granger_prediction', 'directed_phase_lag_index', 'subset_pairwise_spectral_granger_prediction']

```

`interpretation` says how to read the values, and
[Connectivity Metric Ranges](CONNECTIVITY_METRIC_RANGES.md) tabulates all of
it. An unknown name raises an error listing the closest valid names.

## Direction

In the wrapper's results, `result.sel(source="a", target="b")` is always the
influence of `a` on `b`, and for phase measures, positive means `a` leads `b`.

```python
>>> granger = multitaper_connectivity(
...     time_series,
...     sampling_frequency=500,
...     method="pairwise_spectral_granger_prediction",
...     signal_names=["a", "b"],
... ).mean("frequency")
>>> bool(granger.sel(source="a", target="b") > granger.sel(source="b", target="a"))
True

```

The lower-level `Connectivity` methods return plain arrays in **two different
orders**; `MeasureInfo.array_orientation` names each measure's:

- `"target_source"` (Granger and directed-transfer-function families):
  `result[..., i, j]` is `j -> i`.
- `"source_target"` (`directed_phase_lag_index`, `phase_slope_index`,
  `delay`, `group_delay`): positive `result[..., i, j]` (above 0.5 for the
  directed phase lag index) means `i` leads `j`.

Prefer the wrapper's labeled results so you never index these by hand.

## Choosing parameters

- `time_window_duration` defaults to the whole recording (one window). Set it,
  and `time_window_step`, for a time-resolved result.
- Frequency resolution is `2 * time_halfbandwidth_product / time_window_duration`
  Hz, and there are `2 * time_halfbandwidth_product - 1` tapers (rounded
  down).
- `suggest_parameters` picks these from the sampling rate, recording length,
  and the resolution you need:

```python
>>> from spectral_connectivity import suggest_parameters
>>> parameters = suggest_parameters(
...     sampling_frequency=500, signal_duration=10.0, desired_freq_resolution=2.0
... )
>>> parameters["time_window_duration"], parameters["time_halfbandwidth_product"]
(3.0, 3.0)
>>> parameters["n_tapers"]
5

```

## Pitfalls

- **Normalized measures need several observations.** Coherence, phase locking,
  and similar measures average over trials × tapers. With one trial and one
  taper, coherence is exactly 1 at every frequency whatever the data (the
  package warns). Use several trials or `time_halfbandwidth_product >= 2`.
- **Small samples bias the estimates.** Coherence and the phase-locking value
  are biased upward when trials × tapers is small. The debiased measures
  (`pairwise_phase_consistency`, `debiased_squared_*`) are unbiased, so they
  can be slightly negative; that is noise, not negative coupling.
- **Directed measures need a two-sided spectrum.** The Granger and
  directed-transfer-function families work with `multitaper_connectivity`, but
  not with Morlet wavelets or other one-sided coefficients, which raise a
  `ValueError`.
- **Group measures need labels.** Pass `group_labels` (one label per signal)
  through `connectivity_kwargs` for `canonical_coherence`,
  `blockwise_spectral_granger_prediction`, and the other `group_pairwise` and
  `multivariate_components` measures.
- **Band summaries.** Use `frequency_bands={"theta": (4, 8)}` rather than
  averaging frequency bins by hand: phase measures get a circular mean and
  complex measures a complex mean.
- **GPU.** Set `SPECTRAL_CONNECTIVITY_ENABLE_GPU=true` *before* importing the
  package; setting it later has no effect.

```python
>>> bands = multitaper_connectivity(
...     time_series,
...     sampling_frequency=500,
...     method=["coherence_magnitude", "coherence_phase"],
...     signal_names=["a", "b"],
...     frequency_bands={"theta": (4, 8), "gamma": (30, 50)},
... )
>>> bands.coherence_phase.dims
('time', 'band', 'source', 'target')
>>> bool(bands.coherence_phase.sel(band="gamma", source="a", target="b") > 0)
True

```

## Uncertainty

`Connectivity.jackknife(method)` returns a bias-corrected estimate, standard
error, and confidence interval by leaving out one trial or taper at a time:

```python
>>> from spectral_connectivity import Connectivity, Multitaper
>>> connectivity = Connectivity.from_transform(
...     Multitaper(time_series, sampling_frequency=500, time_halfbandwidth_product=2)
... )
>>> result = connectivity.jackknife("coherence_magnitude")
>>> result.estimate.shape == result.standard_error.shape
True

```

## Where to look next

- `help(spectral_connectivity.multitaper_connectivity)` for every wrapper
  option (frequency cropping, decimation, `xarray` input dimensions).
- Each `Connectivity` measure's docstring has a runnable example.
- `help(spectral_connectivity.SpectralTransform)` for the interface a custom
  transform needs to work with `Connectivity.from_transform`.
- The [Cookbook](cookbook.md) has recipes for common tasks.
