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
...     method="magnitude_squared_coherence",
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
>>> granger.is_directed
True
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

The lower-level `Connectivity` methods return plain arrays in the same order:
`result[..., i, j]` is `i -> j` for every directed measure, and for
`directed_phase_lag_index`, `phase_slope_index`, `delay`, and `group_delay` a
positive `result[..., i, j]` (above 0.5 for the directed phase lag index)
means `i` leads `j`.

Prefer the wrapper's labeled results so you never index these by hand.

## Choosing parameters

- `time_window_duration` defaults to the whole recording (one window). Set it,
  and `time_window_step`, for a time-resolved result.
- The common settings `time_window_duration`, `time_window_step`,
  `time_halfbandwidth_product`, `n_tapers`, and `detrend_type` are named
  arguments of `multitaper_connectivity`. Put any other `Multitaper` setting in
  `transform_kwargs={...}` and measure settings in `connectivity_kwargs={...}`;
  a misspelled key raises instead of being ignored. Passing those as plain
  keyword arguments is the deprecated 2.x form.
- Multitaper frequency resolution is the full DPSS concentration bandwidth,
  `2 * time_halfbandwidth_product / time_window_duration` Hz. It describes
  spectral smoothing; FFT bin spacing also depends on the FFT length and
  zero-padding. There are `2 * time_halfbandwidth_product - 1` requested tapers
  (rounded down). Low-bias filtering drops any requested taper whose spectral
  concentration is at or below 0.9, with a warning, and `n_tapers` reports the
  count used: a larger `n_tapers` needs a larger `time_halfbandwidth_product`.
- `suggest_parameters` picks these from the sampling rate, recording length,
  and the resolution you need:

```python
>>> from spectral_connectivity import suggest_parameters
>>> parameters = suggest_parameters(
...     sampling_frequency=500, signal_duration=10.0, frequency_resolution=2.0
... )
>>> parameters["time_window_duration"], parameters["time_halfbandwidth_product"]
(3.0, 3.0)
>>> parameters["n_tapers"]
5

```

## Pitfalls

- **Use the shared result schema.** Common attrs include `transform`,
  `sampling_frequency` when known, `n_observations`, independence flags, and
  `output_schema_version=1`. Remaining settings are in
  `transform_parameters_json`, with `estimator`/`execution` records. The 15
  released `mt_*` attributes are compatibility copies scheduled for removal
  in 4.0. Multitaper's `frequency_resolution` property and helper names remain
  supported. See the [schema reference](cookbook.md#result-schema-and-transform-bandwidth).
- **Normalized measures need several observations.** Coherence, phase locking,
  and similar measures average over trials × tapers. With one trial and one
  taper, coherence is exactly 1 at every frequency whatever the data (the
  package warns). Use several trials or `time_halfbandwidth_product >= 2`.
- **Small samples bias the estimates.** Coherence and the phase-locking value
  are biased upward when trials × tapers is small. The debiased measures
  (`pairwise_phase_consistency`, `debiased_squared_*`) are unbiased, so they
  can be slightly negative; that is noise, not negative coupling.
- **Coherence is squared.** `magnitude_squared_coherence` is `|coherency|^2`;
  take `abs(coherency)` for the unsquared magnitude. Its 2.x name,
  `coherence_magnitude`, is deprecated: do not write new code with it, and
  index default results by the new name. A measure variable's `quantity` and
  `formula` attrs say which form a coherency-family value has.
- **Diagonals differ by measure.** A self-connection is NaN for coherence (a
  mask; the value would be 1) but 0 for `imaginary_coherence` and 1 for
  `phase_locking_value`, to within rounding (NaN for a flat, zero-power
  signal). Read a variable's
  `diagonal_policy` attr, and before averaging over pairs mask a pairwise
  variable `v` with
  `v.where(v.source != v.target)` (not a whole Dataset, whose `power` has no
  `target` axis).
- **Rate-adjusting spike-field coherence.** Call
  `statistics.coherence_rate_adjustment` with `Connectivity.power()` of the
  spike train and `is_one_sided=True` (that spectrum is one-sided), the spikes
  given to the transform as counts per sample, and
  `dt = 1 / sampling_frequency`. Leave out 0 Hz (and, with the default mean
  removal, the bins within a taper bandwidth of it); the Nyquist bin is valid
  only with a two-sided spectrum.
  For a two-sided spectrum
  (`Connectivity.power(one_sided=False)`; the wrappers do not return one) pass
  `is_one_sided=False`. Omitting the keyword warns and assumes one-sided.
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
- **Group measures need labels.** Pass `group_labels=...` (one label per
  signal, e.g. `["CA1", "CA1", "PFC"]`) directly to `multitaper_connectivity`
  or `fourier_connectivity` for `canonical_coherence`,
  `blockwise_spectral_granger_prediction`, and the other `group_pairwise` and
  `multivariate_components` measures. The labels reach only the measures that
  take them, so one call may mix group and pairwise measures.
- **Band summaries.** Use `frequency_bands={"theta": (4, 8)}` rather than
  averaging frequency bins by hand: phase measures get a circular mean and
  complex measures a complex mean.
- **Frequency ranges include both endpoints.** `frequency_range=(8, 12)` keeps
  the bins at 8 and 12 Hz. Pass it to the wrapper itself, also for
  `phase_slope_index`, `group_delay`, and `delay`, which then consume only
  those bins. Their old `frequencies_of_interest` excluded the edges and is
  deprecated; do not write new code with it.
- **Avoid the deprecated 2.x spellings** in new code: use
  `Connectivity.from_transform` (not `from_multitaper`),
  `power_confidence_intervals(n_observations=...)` (not `n_tapers=`), and
  the `transform_kwargs`/`connectivity_kwargs` mappings for advanced
  settings. Each old form still works through 3.x with a
  `DeprecationWarning`.
- **GPU.** Set `SPECTRAL_CONNECTIVITY_ENABLE_GPU=true` *before* importing the
  package; setting it later has no effect.

```python
>>> bands = multitaper_connectivity(
...     time_series,
...     sampling_frequency=500,
...     method=["magnitude_squared_coherence", "coherence_phase"],
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
>>> result = connectivity.jackknife("magnitude_squared_coherence")
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
