# Cookbook

Short, self-contained recipes for the most common tasks. Every code block on
this page is executed as a doctest in the test suite
(`tests/test_cookbook.py`), so the recipes are guaranteed to run against the
current release.

All recipes share this setup. `time_series` has shape
`(n_time_samples, n_trials, n_signals)`; a 2-D `(n_time_samples, n_signals)`
array works too.

```python
>>> import numpy as np
>>> from spectral_connectivity import (
...     multitaper_connectivity,
...     fourier_connectivity,
...     list_measures,
... )
>>> rng = np.random.default_rng(0)
>>> time_series = rng.standard_normal((1000, 4, 3))

```

## Discover the available measures

`list_measures()` enumerates every valid `method` name, with its output
category and a one-line description. Use it instead of guessing method strings.

```python
>>> len(list_measures())
37
>>> [m.name for m in list_measures(default_only=True)][:3]
['magnitude_squared_coherence', 'coherence_phase', 'debiased_squared_phase_lag_index']
>>> [m.name for m in list_measures(directed=True)][:2]
['pairwise_spectral_granger_prediction', 'directed_phase_lag_index']
>>> next(m for m in list_measures() if m.name == "phase_slope_index").requires_two_sided
False
>>> power = next(m for m in list_measures() if m.name == "power")
>>> power.category, power.description
('power', 'Return the power spectral density of the signal (one-sided by default).')

```

Passing an unknown name raises a helpful error rather than an obscure
`AttributeError`:

```python
>>> multitaper_connectivity(
...     time_series, sampling_frequency=500, method="coherence"
... )
Traceback (most recent call last):
    ...
ValueError: 'coherence' is not a known connectivity measure. Did you mean: 'magnitude_squared_coherence', 'coherence_phase', 'imaginary_coherence', 'partial_coherence', 'directed_coherence'? Call spectral_connectivity.list_measures() to see the 37 available measures.

```

## Functional connectivity: coherence

The high-level `multitaper_connectivity` runs the multitaper transform and
returns a labeled `xarray.DataArray` with `(time, frequency, source, target)`
axes.

```python
>>> coherence = multitaper_connectivity(
...     time_series,
...     sampling_frequency=500,
...     method="magnitude_squared_coherence",
...     time_halfbandwidth_product=3,
... )
>>> type(coherence).__name__
'DataArray'
>>> coherence.dims
('time', 'frequency', 'source', 'target')
>>> coherence.name
'magnitude_squared_coherence'
>>> coherence.attrs["quantity"]  # |coherency|^2, not |coherency|
'squared_magnitude'

```

Before 3.0 this measure was named `coherence_magnitude`. A default result now
holds `magnitude_squared_coherence` and no `coherence_magnitude`, so 2.x code
that indexes a default result by the old name fails loudly. Requesting the old
name explicitly still works through 3.x, with a `DeprecationWarning`: the
result keeps the old variable name and records `measure="coherence_magnitude"`,
the name asked for, with the same values. The old name is removed in 4.0.

```python
>>> import warnings
>>> default = multitaper_connectivity(time_series, sampling_frequency=500)
>>> "magnitude_squared_coherence" in default, "coherence_magnitude" in default
(True, False)
>>> with warnings.catch_warnings(record=True) as caught:
...     warnings.simplefilter("always")
...     old = multitaper_connectivity(
...         time_series, sampling_frequency=500, method="coherence_magnitude"
...     )
>>> [warning.category.__name__ for warning in caught]
['DeprecationWarning']
>>> old.name, old.attrs["measure"]
('coherence_magnitude', 'coherence_magnitude')
>>> bool(old.equals(default["magnitude_squared_coherence"]))
True

```

## Read and slice the result

Because the output is labeled, you select by name rather than by axis index.
Default signal labels are the string indices `"0"`, `"1"`, `"2"`; pass
`signal_names` to use your own.

```python
>>> pair = coherence.sel(source="0", target="1")
>>> pair.dims
('time', 'frequency')
>>> band = coherence.sel(frequency=slice(30, 50))
>>> float(band.frequency.min()) >= 30.0
True

```

## Mask or fill the diagonal

A self-connection (`source` equal to `target`) does not hold the same kind of
value for every measure, and each variable's `diagonal_policy` attribute says
which it is. Coherence is NaN there by the package's choice (`masked_nan`),
although self-coherence is 1 for a signal with nonzero power; the imaginary
part of coherency is 0 for a signal with itself, which `imaginary_coherence`
returns (`constant`) and the signed `imaginary_coherency` masks with NaN.
A constant holds to within rounding, and only for signals with power: where
`nan_for_zero_power_signal` is set, as for `imaginary_coherence`, a flat
signal's entries are NaN. The policy table in
`docs/CONNECTIVITY_METRIC_RANGES.md` defines each policy and lists its
measures.

```python
>>> default = multitaper_connectivity(time_series, sampling_frequency=500)
>>> default["magnitude_squared_coherence"].attrs["diagonal_policy"]
'masked_nan'
>>> default["imaginary_coherence"].attrs["diagonal_policy"]
'constant'
>>> imaginary = default["imaginary_coherence"]
>>> self_imaginary = imaginary.sel(source="0", target="0")
>>> bool(np.allclose(self_imaginary, imaginary.attrs["diagonal_value"]))  # to rounding
True

```

To treat every self-connection alike, compare the `source` and `target`
labels, which gives a `(source, target)` mask, and apply it to each variable
with both axes (not to `power`, which has only `source`). For a group measure
compare `source_group` with `target_group`.

```python
>>> self_connection = imaginary.source == imaginary.target
>>> masked = imaginary.where(~self_connection)  # NaN, like imaginary_coherency
>>> bool(masked.sel(source="0", target="0").isnull().all())
True
>>> coherence = default["magnitude_squared_coherence"]
>>> filled = coherence.where(~self_connection, 1.0)  # self-coherence at nonzero power
>>> float(filled.sel(source="1", target="1").min())
1.0
>>> bool(filled.sel(source="0", target="1").equals(coherence.sel(source="0", target="1")))
True

```

## Directed connectivity: spectral Granger

Directed measures are opt-in by name. The result reads `source -> target`:
`result.sel(source="A", target="B")` is the influence **from A to B**.

```python
>>> granger = multitaper_connectivity(
...     time_series,
...     sampling_frequency=500,
...     method="pairwise_spectral_granger_prediction",
...     signal_names=["A", "B", "C"],
...     time_halfbandwidth_product=3,
... )
>>> granger.coords["source"].values.tolist()
['A', 'B', 'C']
>>> a_to_b = granger.sel(source="A", target="B")
>>> a_to_b.dims
('time', 'frequency')

```

Directed measures come from a factorization of the estimated spectrum. Use
windows longer than the slowest correlation in the data: nothing warns when a
window is too short. The values also depend slightly on the FFT length
(`transform_kwargs={"n_fft_samples": ...}`). Zero-padding to twice the window
removes that dependence, at about twice the cost, but in simulation it did not
bring the values closer to the true ones unless the window was too short. The
Notes of `spectral_connectivity.minimum_phase_decomposition.minimum_phase_decomposition` give the sizes.

## Compute several measures at once

Pass a list of methods to get an `xarray.Dataset` with one variable per
measure. Shared spectra are cached, so this is cheaper than separate calls.

```python
>>> result = multitaper_connectivity(
...     time_series,
...     sampling_frequency=500,
...     method=["power", "magnitude_squared_coherence"],
...     time_halfbandwidth_product=3,
... )
>>> type(result).__name__
'Dataset'
>>> sorted(result.data_vars)
['magnitude_squared_coherence', 'power']
>>> result["power"].dims
('time', 'frequency', 'source')

```

`power` is indexed by `source` on purpose: in a Dataset it shares the `source`
axis of the pairwise measures, so `result["power"].sel(source="0")` is the
power of the signal whose row `result["magnitude_squared_coherence"].sel(
source="0")` holds. The component measures' filters and patterns use a
`signal` axis instead, because each holds weights over the same channels for
both sides of a connection (`side="source"` and `side="target"`). For a
standalone channel axis, rename it:

```python
>>> result["power"].rename(source="signal").dims
('time', 'frequency', 'signal')

```

## Set transform and measure options

The common multitaper settings (`sampling_frequency`, `time_window_duration`,
`time_window_step`, `time_halfbandwidth_product`, `n_tapers`, `detrend_type`)
and `group_labels` are named arguments of `multitaper_connectivity`. Any other
`Multitaper` setting goes in `transform_kwargs`, and settings for the measures
go in `connectivity_kwargs`:

```python
>>> components = multitaper_connectivity(
...     time_series,
...     sampling_frequency=500,
...     time_window_duration=0.5,
...     time_halfbandwidth_product=2,
...     n_tapers=3,
...     method="canonical_coherency",
...     group_labels=["a", "a", "b"],
...     transform_kwargs={"n_fft_samples": 512, "taper_weighting": "eigen"},
...     connectivity_kwargs={"n_components": 1},
... )
>>> components.attrs["mt_n_fft_samples"], components.attrs["mt_n_tapers"]
(512, 3)

```

Each setting has one route, so a common setting placed in `transform_kwargs`
raises `ValueError` instead of silently competing with its named argument, and
a misspelled key in either mapping raises `TypeError`. `connectivity_to_xarray`
takes an existing transform, so it has `connectivity_kwargs` and the named
`group_labels`, `frequency_range`, `frequency_decimation`, `frequency_bands`,
and `frequency_reduction`, but no `transform_kwargs`. The working
precision `dtype` and the Wilson factorization's `minimum_phase_tolerance` and
`minimum_phase_max_iterations`, used by the directed measures, are named
arguments of every wrapper; in `connectivity_kwargs` they raise `ValueError`.

Before 3.0, advanced settings were passed as plain keyword arguments, e.g.
`multitaper_connectivity(x, fs, n_fft_samples=512)`. That still works
through 3.x, with the same result and a `DeprecationWarning` naming the
mapping to use, and is removed in 4.0.

## Leave out artifacts and missing samples

Mark the samples to leave out with `artifact_samples`, a `Multitaper` (and
`ShortTimeFourierTransform`) setting, so in `transform_kwargs` for the wrapper. `True` marks an **artifact**, a
sample to leave out, as MNE's `BAD_` annotations, FieldTrip's artifact
definitions, and NumPy masked arrays do; a mask that marks the valid samples
as `True` must be inverted (`~mask`). `"nonfinite"` leaves out every NaN or
infinite sample:

```python
>>> with_gap = time_series.copy()
>>> with_gap[600:650, 1, :] = np.nan  # a gap in trial 1, 1.2-1.3 s
>>> ppc = multitaper_connectivity(
...     with_gap,
...     sampling_frequency=500,
...     time_window_duration=0.5,
...     method="pairwise_phase_consistency",
...     transform_kwargs={"artifact_samples": "nonfinite"},
... )
>>> bool(np.isfinite(ppc.isel(source=0, target=1)).all())
True

```

A trial's window is used only if none of its samples is an artifact, so choose
windows that fit between artifacts: here the 50-sample gap drops all 250
samples of trial 1's third 0.5 s window, a quarter of that window's data, and a
1 s window would drop half the recording of trial 1. Each time window averages
the trials it keeps, a window keeping none is NaN in every measure, and
dropping more than half of the trial windows warns. The outputs mark what is
kept: `Multitaper.valid_windows` is `True` at the kept trial windows, and the
result's `transform_parameters_json` records the kind of mask and the numbers
of artifact samples and dropped trial windows (`n_observations` stays the raw
count).

Artifact intervals given in seconds, such as an MNE `BAD_` annotation's onset
and duration, become a mask that is `True` from each onset for its duration,
with the time series' `(n_time_samples, n_trials)` layout:

```python
>>> sampling_frequency = 500
>>> times = np.arange(time_series.shape[0]) / sampling_frequency
>>> artifact_samples = np.zeros(time_series.shape[:2], dtype=bool)
>>> for trial, start, stop in [(1, 1.2, 1.3), (3, 0.1, 0.15)]:  # (trial, s, s)
...     artifact_samples[(times >= start) & (times < stop), trial] = True
>>> coherence = multitaper_connectivity(
...     time_series,
...     sampling_frequency=sampling_frequency,
...     time_window_duration=0.5,
...     method="magnitude_squared_coherence",
...     transform_kwargs={"artifact_samples": artifact_samples},
... )
>>> coherence.sizes["time"]
4

```

A FieldTrip artifact definition (`cfg.artfctdef.<type>.artifact`) gives each
interval as a row `[begsample, endsample]` of 1-based sample numbers that
include both ends: FieldTrip marks `begsample:endsample`
([`trl2boolvec`](https://github.com/fieldtrip/fieldtrip/blob/00ecb6e750c6e0d22e087be1874834e689a475d8/private/trl2boolvec.m)).
In Python that is the slice `begsample - 1:endsample`. The numbers count from
the start of the recording, so in an epoch that starts at recording sample
`trl_begsample` (the first column of FieldTrip's `trl`) the slice is
`begsample - trl_begsample:endsample - trl_begsample + 1`:

```python
>>> artifact = np.array([[601, 650]])  # one row, samples 601 to 650 of the recording
>>> recording_mask = np.zeros(1000, dtype=bool)
>>> for begsample, endsample in artifact:
...     recording_mask[begsample - 1 : endsample] = True
>>> int(recording_mask.sum()), int(np.flatnonzero(recording_mask)[0])  # 50 samples from index 600
(50, 600)

```

A `(n_time_samples,)` mask is shared by every trial. A mask may also mark each
channel, with the time series' own `(n_time_samples, n_trials, n_signals)`
shape: an artifact in one channel then leaves the sample out of every channel,
since every cross-spectrum needs both channels at the same samples. For a 2-D
`(n_time_samples, n_channels)` time series give the wrapper a
`(n_time_samples,)` or `(n_time_samples, n_channels)` mask. For an
`xarray.DataArray` time series, a `DataArray` mask is aligned by dimension
name: it needs the time dimension, and is shared by every trial or channel
when it lacks that dimension. An unlabeled mask is read in the time, trial,
channel order.

The measures that count observations (listed under
`Connectivity.n_valid_observations`) count each bin's kept observations.
`n_valid_observations` gives those counts, one per time window and
frequency, for the statistics functions:

```python
>>> from spectral_connectivity import Connectivity, Multitaper
>>> from spectral_connectivity.statistics import coherence_significance_pvalue
>>> multitaper = Multitaper(
...     time_series,
...     sampling_frequency=sampling_frequency,
...     time_window_duration=0.5,
...     artifact_samples=artifact_samples,
... )
>>> connectivity = Connectivity.from_transform(multitaper)
>>> connectivity.n_valid_observations[:, 0].tolist()  # of 4 trials x 5 tapers
[15, 20, 15, 20]
>>> p_values = coherence_significance_pvalue(
...     connectivity.coherency(),
...     connectivity.n_valid_observations[..., np.newaxis, np.newaxis],
... )

```

A bin that keeps fewer than two observations, such as a window that keeps no
trial, has no test, and its p-value is NaN. Every statistics function that
takes a count reads such an array the same way: an entry below the fewest
observations the statistic needs (two for the coherence statistics, one for
the power statistics) gives NaN there, while a single count below it raises.

Dropping a window throws away its valid samples too. With
`artifact_handling="missing_data_tapers"` (a `Multitaper` setting; for the
wrapper, in `transform_kwargs` beside `artifact_samples`) a trial window
containing artifacts is estimated from its valid samples instead, with
missing-data Slepian tapers: the sequences that are zero at its artifacts and
most concentrated in the bandwidth, the DPSS of a window with gaps (Bronez
1988; Chave 2019). It keeps those with over 90% of their energy in the band,
so it can keep fewer tapers than a clean window. A window keeping fewer than
2 is still dropped, with a warning: one with too few valid samples, or one
whose artifacts recur more often than every `1 / frequency_resolution`
seconds (here 1 / 12 Hz, 83 ms). `Multitaper.n_kept_tapers` says what each
trial window keeps; here the two windows with artifacts keep 4 and 3 of
their 5 tapers:

```python
>>> multitaper = Multitaper(
...     time_series,
...     sampling_frequency=sampling_frequency,
...     time_window_duration=0.5,
...     artifact_samples=artifact_samples,
...     artifact_handling="missing_data_tapers",
... )
>>> multitaper.n_kept_tapers.tolist()  # (n_time_windows, n_trials)
[[5, 5, 5, 4], [5, 5, 5, 5], [5, 3, 5, 5], [5, 5, 5, 5]]

```

Which to choose:

- `"drop"` when few windows have artifacts, or when the spectrum spans more
  than about 1000 in power (strong peaks over a low floor, or steep 1/f
  power), where the gapped windows' extra leakage matters. A spectrum's span
  is its largest true power over its smallest at every frequency from 0 Hz to
  Nyquist, those below the band of interest included, since leakage brings
  power from any of them;
- `"missing_data_tapers"` when `"drop"` would lose much of the data, the
  artifacts are short, and the spectrum spans less than about 100;
- in between, compute both on your data and compare.

:::{note}
The gapped windows' tapers are less concentrated than the DPSS (down to 0.9
rather than about 0.95), and fewer of them smooth less, so their expected
spectrum differs from that without gaps. For 64-sample windows (NW = 3) and
the gap patterns tested (a 4-, a 12-, and two 3-sample gaps), away from 0 Hz
and Nyquist, it differs by about 5% for a spectrum spanning about 120 (an
AR(2) with poles of radius 0.77) and by up to 37% for one spanning 2400
(poles of radius 0.95 at 0.1 cycles per sample, two 3-sample gaps), each
span taken over the window's 64 FFT frequencies. Other gap positions gave
1-13% (a single 4- to 12-sample gap at 165 positions, under 6% for 68% of
them) and 7-70% (two 3-sample gaps at 325, median 34%), and the difference
grows toward Nyquist. Several short gaps can leak far more than one long one,
depending on how far apart they are. For 500-sample windows (NW = 4) of an
AR(1) with coefficient 0.99 (spanning about 40000), a bandwidth or more above
0 Hz, one 20-sample gap at the window's center (samples 240-259) changed the
expected spectrum by 2.4% (1.5-10.3% over its positions, above 5% only within
25 samples of either end), and two 5-sample gaps at samples 200 and 300 by 52%.
Over a grid of positions two 5-sample gaps changed it by 6-70% (median 41%)
when 30 or more samples apart, but by 3-36% (median 8%) when closer, as they
then act much like one gap (4.9% at samples 240 and 255). In exchange the
valid samples are used: with three of four trials gapped in every window, the
variance of white-noise power is a third of that of `"drop"`. Building the
transform finds the tapers of each distinct pattern of artifacts in a trial
window, about 2 ms per pattern for 1000 samples and 65 ms for 60000. With
multithreaded OpenBLAS on many cores, as in the NumPy 1.24 and SciPy
1.11-era wheels, that eigensolver (ARPACK) can be 190-840 times slower per
pattern (600 ms against 3.2 ms for 1000 samples at NW = 4, on 18 threads). If
building the transform is slow, set `OPENBLAS_NUM_THREADS=1` before starting
Python, or build it within `threadpoolctl.threadpool_limits(1)`. Windows
without artifacts are unchanged.
:::

## Group measures: canonical coherence between areas

Group measures (`list_measures(category="group_pairwise")` and
`list_measures(category="multivariate_components")`) compare *groups* of
signals, such as all channels in one brain area against all channels in
another. Pass `group_labels`, one label per signal naming its group; a
`group_pairwise` result is indexed by `source_group` and `target_group`
instead of by signal.

```python
>>> between_areas = multitaper_connectivity(
...     time_series,
...     sampling_frequency=500,
...     method="canonical_coherence",
...     group_labels=["CA1", "CA1", "PFC"],
... )
>>> between_areas.dims
('time', 'frequency', 'source_group', 'target_group')
>>> between_areas.coords["source_group"].values.tolist()
['CA1', 'PFC']

```

When the input is a DataArray whose signals carry a coordinate naming their
group, pass that coordinate's name instead of the labels:

```python
>>> import xarray as xr
>>> labeled = xr.DataArray(
...     time_series,
...     dims=("time", "trial", "channel"),
...     coords={"channel": ["a", "b", "c"], "region": ("channel", ["CA1", "CA1", "PFC"])},
... )
>>> multitaper_connectivity(
...     labeled, sampling_frequency=500, method="canonical_coherence", group_labels="region"
... ).identical(between_areas)
True

```

The component measures (`canonical_coherency`,
`maximized_imaginary_coherency_components`) return a Dataset with one
`connection` per pair of groups. Each connection names its
`connection_source_group` and `connection_target_group`, and the spatial
filters and patterns have a `side` axis, `"source"` or `"target"`, over the
`signal` axis; a side's entries for signals outside its group are NaN.

```python
>>> components = multitaper_connectivity(
...     time_series,
...     sampling_frequency=500,
...     method="canonical_coherency",
...     group_labels=["CA1", "CA1", "PFC"],
... )
>>> components["canonical_coherency"].dims
('time', 'frequency', 'connection', 'component')
>>> components.connection_source_group.values.tolist()
['CA1']
>>> components.connection_target_group.values.tolist()
['PFC']
>>> filters = components["canonical_coherency_filters"]
>>> filters.dims
('time', 'frequency', 'connection', 'component', 'side', 'signal')
>>> filters.side.values.tolist()
['source', 'target']
>>> source_filters = filters.sel(side="source", connection=0, component=0)
>>> bool(source_filters.sel(signal="2").isnull().all())  # signal 2 is in PFC
True

```

`canonical_coherence` and `canonical_coherency` are different estimators, and
neither is the square or square root of the other. `canonical_coherence` is
the historical squared canonical correlation (complex channel weights);
`canonical_coherency` is the phase-optimized CaCoh (real weights and one
phase), whose magnitude is the unsquared maximized coherence. Their attributes
say so:

```python
>>> between_areas.attrs["quantity"], components["canonical_coherency"].attrs["quantity"]
('squared_magnitude', 'complex')
>>> print(between_areas.attrs["formula"])
max over complex weights a, b of |a^H S_st b|^2 / ((a^H S_ss a) * (b^H S_tt b)), with S_st the cross-spectral matrix from the source group's signals to the target group's

```

## Test against surrogate data

`Connectivity.surrogate_test` recomputes a measure on surrogate coefficients
built to satisfy a null hypothesis and returns Monte Carlo p-values. `kind`
sets the null, so choose it for the question:

| Question | `kind` | Needs | A shared stimulus-locked response |
| --- | --- | --- | --- |
| Do the signals covary from trial to trial? | `"trial_shuffle"` | 2+ trials | stays in the null |
| Are the signals independent at all? | `"phase_rotation"` | 2+ trials (or averaged non-overlapping time windows), the half bandwidth | is rejected |
| Is there more than zero-lag mixing? | `"conjugation"` | 2+ trials (or averaged non-overlapping time windows) | is rejected if lagged |

- `"trial_shuffle"` shuffles each signal's trials, keeping each trial whole.
  The expectation must average trials.
- `"phase_rotation"` rotates each trial (every taper alike) by a random
  phase. The frequencies within the transform's half bandwidth of 0 Hz and
  Nyquist are not tested; `Connectivity.from_transform` records it for a `Multitaper`
  (`frequency_resolution / 2`), and `half_bandwidth=` overrides it. A common
  stimulus-locked response is rejected too; to ask about interaction beyond
  it, use `"trial_shuffle"`. Subtracting each signal's mean across trials
  first leaves p-values too small unless there are many trials (5.5-7.0% of
  tests rejected at nominal 5% with 10 trials, about 5% with 40).
- `"conjugation"` conjugates (time-reverses) random trials. 0 Hz and Nyquist
  of a real signal, which it leaves unchanged, are not tested. Instantaneous mixing of
  independent sources is in its null, so pair it with
  a measure of lagged interaction: `imaginary_coherence`, the phase-lag
  indices, `phase_slope_index`, or pairwise spectral Granger. With
  `magnitude_squared_coherence` it asks only whether the coupling exceeds what
  a real coherency gives.
- `"phase_rotation"` and `"conjugation"` randomize whole trials, so each needs
  2 or more: a trial's tapers are correlated wherever the spectrum is not flat
  across the half bandwidth (near a peak, or on a steep 1/f spectrum), and
  randomizing them separately would reject independent signals far too
  often. With one recording, average non-overlapping time windows instead
  (`expectation_type="time_tapers"`, with `time_window_step` at least
  `time_window_duration`), which makes each window a unit; that holds only for
  windows longer than the signals' memory, which a narrow spectral peak makes
  long. Windows that overlap, even by half, have correlated coefficients, so
  every window of a trial shares its draw, and one recording in overlapping
  windows is refused. `Connectivity.from_transform` records whether the
  windows overlap; for coefficients passed directly, pass
  `time_bins_are_disjoint=True` if they do not.

Here signal 1 is signal 0 delayed by 3 samples (6 ms) and signal 2 is
independent:

```python
>>> from spectral_connectivity import Connectivity, Multitaper
>>> rng = np.random.default_rng(1)
>>> leader = rng.standard_normal((503, 10))
>>> lagged = np.stack([leader[3:], leader[:-3], rng.standard_normal((500, 10))], axis=-1)
>>> lagged += 0.5 * rng.standard_normal(lagged.shape)
>>> multitaper = Multitaper(lagged, sampling_frequency=500, time_halfbandwidth_product=2)
>>> connectivity = Connectivity.from_transform(multitaper)
>>> result = connectivity.surrogate_test(
...     "imaginary_coherence", kind="conjugation", n_surrogates=199, random_state=0
... )
>>> result.p_value.shape  # (n_time_windows, n_frequencies, n_signals, n_signals)
(1, 251, 3, 3)
>>> significant = result.fwer_p_value <= 0.05  # family-wise over every frequency and pair
>>> bool(significant[0, 40, 0, 1]), bool(significant[..., [0, 1], [2, 2]].any())
(True, False)

```

Report `fwer_p_value` to say *where* the measure is significant: it adjusts
for testing every element of a family. `p_value` is uncorrected, the test of
one element (one pair at one frequency) chosen before looking. The diagonal is
not tested and its p-values are NaN. By default every element is one family;
`fwer_axis` names the axes of one family, e.g. `(1, 2, 3)` for frequencies and
pairs within each time window, and the result stores the axes it used:

```python
>>> per_window = connectivity.surrogate_test(
...     "magnitude_squared_coherence",
...     kind="trial_shuffle",
...     fwer_axis=(1, 2, 3),
...     n_surrogates=199,
...     random_state=0,
... )
>>> per_window.fwer_axis, per_window.alternative
((1, 2, 3), 'greater')
>>> per_window.null_maxima.shape  # (n_surrogates, n_time_windows): one maximum per family
(199, 1)
>>> bool(per_window.fwer_p_value[0, 40, 0, 1] <= 0.05)
True
>>> rotated = connectivity.surrogate_test(
...     "magnitude_squared_coherence", kind="phase_rotation", n_surrogates=199, random_state=0
... )
>>> connectivity.half_bandwidth  # recorded from the Multitaper: 2 Hz
2.0
>>> bool(np.isnan(rotated.p_value[0, :2, 0, 1]).all())  # 0 and 1 Hz are within it of 0 Hz
True

```

## Global coherence pooled over time windows

`method="global_coherence"` decomposes each time window's trials and tapers
separately. To decompose one cross-spectral matrix per frequency, pooled over
every taper, trial, and window (the estimand of Chronux `CrossSpecMatc`'s
`Ctot`), build a `Connectivity` with `expectation_type="time_trials_tapers"`
from a transform whose windows do not overlap. Here eight channels of a
one-minute recording share a 10 Hz oscillation, and each 2 s window adds 3
tapers:

```python
>>> from spectral_connectivity import Connectivity, Multitaper
>>> t = np.arange(60 * 200) / 200  # 60 s at 200 Hz
>>> shared = np.sin(2 * np.pi * 10 * t)[:, np.newaxis] * np.linspace(0.5, 1.5, 8)
>>> recording = (shared + rng.standard_normal((t.size, 8)))[:, np.newaxis, :]
>>> multitaper = Multitaper(
...     recording,
...     sampling_frequency=200,
...     time_window_duration=2,  # the step defaults to the duration
...     time_halfbandwidth_product=2,
...     n_tapers=3,
... )
>>> pooled = Connectivity.from_transform(multitaper, expectation_type="time_trials_tapers")
>>> pooled.n_observations, pooled.n_signals
(90, 8)
>>> fractions, vectors = pooled.global_coherence()
>>> fractions.shape, vectors.shape  # one value per frequency, no time axis
((201, 1), (201, 8, 1))
>>> bool(fractions[20, 0] > 0.9)  # 10 Hz
True

```

`n_observations` should exceed `n_signals`, ideally by a wide margin: the value
is at least `1 / min(n_observations, n_signals)` and, without coupling, biased
above `1 / n_signals`, so compare values only at matched counts. This is not
the estimator of Purdon et al. (2013): it takes the mean over windows, not the
median; it decomposes the raw cross-spectrum, with no Laplacian unless you
re-reference the signals first; and it returns no significance test.

## Partial coherence with many signals and few trials

Partial coherence conditions each pair on every other signal, so it needs
several times as many observations (trials x tapers) as signals. With at most
about 2.5 per signal, shrink toward the diagonal: `"auto"` estimates the
intensity for each time window and frequency. The result records the setting,
not the intensities; compute those from the same transform to report them.

```python
>>> from spectral_connectivity import Connectivity, Multitaper
>>> rng = np.random.default_rng(1)
>>> recording = rng.standard_normal((1000, 4, 16))  # (time, trials, signals)
>>> recording[..., 1:] += 0.5 * recording[..., :-1]  # neighbors share a source
>>> partial = multitaper_connectivity(
...     recording,
...     sampling_frequency=500,
...     time_halfbandwidth_product=3,
...     method="partial_coherence",
...     connectivity_kwargs={"shrinkage": "auto"},
... )
>>> partial.attrs["n_observations"], partial.attrs["arg_shrinkage"]  # 4 trials x 5 tapers
(20, 'auto')
>>> transform = Multitaper(recording, sampling_frequency=500, time_halfbandwidth_product=3)
>>> intensity = Connectivity.from_transform(transform).partial_coherence_shrinkage()
>>> intensity.shape  # (n_time_windows, n_frequencies)
(1, 501)
>>> bool(0 < intensity.min() <= np.median(intensity) <= intensity.max() < 1)
True

```

Report the median and range of `intensity` with the result. Shrinkage biases
partial coherence toward 0, and `"auto"` shrinks less with more trials, so
compare conditions at matched trial counts or with one fixed `shrinkage`.

## Collapse into frequency bands

Pass `frequency_bands` to average (or integrate) each measure within named
bands. The `frequency` axis is replaced by a labeled `band` axis.

```python
>>> banded = multitaper_connectivity(
...     time_series,
...     sampling_frequency=500,
...     method="magnitude_squared_coherence",
...     frequency_bands={"theta": (4, 8), "gamma": (30, 50)},
...     time_halfbandwidth_product=3,
... )
>>> banded.dims
('time', 'band', 'source', 'target')
>>> banded.coords["band"].values.tolist()
['theta', 'gamma']

```

For `Welch`, `ShortTimeFourierTransform`, `MorletWavelet`, or your own
transform, `connectivity_to_xarray` takes the transform, a list of measures,
and the same frequency options:

```python
>>> from spectral_connectivity import Welch
>>> from spectral_connectivity import connectivity_to_xarray
>>> welch = Welch(time_series, sampling_frequency=500, segment_duration=0.5)
>>> welch_bands = connectivity_to_xarray(
...     welch,
...     ["power", "magnitude_squared_coherence"],
...     frequency_bands={"theta": (4, 8), "gamma": (30, 50)},
... )
>>> list(welch_bands.data_vars)
['power', 'magnitude_squared_coherence']
>>> welch_bands["power"].dims
('time', 'band', 'source')

```

For a confidence interval on a band value, see
[Intervals for band statistics](#intervals-for-band-statistics).

## Intervals for band statistics

`Connectivity.jackknife` takes the same `frequency_bands` and
`frequency_reduction`: it reduces the measure of the data and of every
leave-one-out replicate to the bands, then forms the interval (here on the
`atanh(sqrt(.))` scale of the band-mean coherence). Leave out whole trials;
leaving out trials needs at least 3. Here the two signals share a 20 Hz
rhythm, so they are coherent in the beta band only:

```python
>>> from spectral_connectivity import Connectivity, Multitaper
>>> rng = np.random.default_rng(2)
>>> t = np.arange(500) / 500  # 1 s at 500 Hz
>>> rhythm = np.sin(2 * np.pi * 20 * t[:, np.newaxis] + rng.uniform(0, 2 * np.pi, 10))
>>> trials = rng.standard_normal((500, 10, 2)) + rhythm[..., np.newaxis]  # (time, trials, signals)
>>> multitaper = Multitaper(trials, sampling_frequency=500, time_halfbandwidth_product=3)
>>> connectivity = Connectivity.from_transform(multitaper)
>>> result = connectivity.jackknife(
...     "magnitude_squared_coherence",
...     leave_out="trial",
...     frequency_bands={"theta": (4, 8), "beta": (16, 24)},
...     keep_replicates=True,
... )
>>> result.estimate.shape  # (n_time_windows, n_bands, n_signals, n_signals)
(1, 2, 2, 2)
>>> band = list(result.frequency_bands).index  # position of a band on the band axis
>>> lower, upper = result.confidence_interval
>>> beta = band("beta")
>>> print(lower[0, beta, 0, 1].round(2), result.estimate[0, beta, 0, 1].round(2), upper[0, beta, 0, 1].round(2))
0.61 0.64 0.66

```

Leaving out one taper at a time is refused for bands. Different tapers'
coefficients are uncorrelated at one frequency but correlated at frequencies
within the bandwidth, so the tapers of a band are not independent
observations and the interval would be too narrow: a nominal 95% interval
covered 43-81% of band means and maxima in simulation, against 91-96% for
band means leaving out trials. (Welch segments, which are not tapers of the
same samples, may be left out one at a time.)

```python
>>> connectivity.jackknife("magnitude_squared_coherence", frequency_bands={"beta": (16, 24)})
Traceback (most recent call last):
...
ValueError: jackknife(frequency_bands=...) cannot leave out one taper at a time (leave_out='observation' with expectation_type='trials_tapers'): ...

```

For any other statistic of the measure, compute it from the estimate and from
each of `result.replicates` (one per trial left out, untransformed, kept with
`keep_replicates=True`) and pass both to `jackknife_confidence_interval`.
Here, how much more coherent the signals are in beta than in theta. The
default `transformation="identity"` suits a difference: it can be negative,
so neither the log nor the `atanh` scale of a coherence applies to it.

```python
>>> from spectral_connectivity.statistics import jackknife_confidence_interval
>>> result.replicates.shape  # (n_trials, *result.estimate.shape)
(10, 1, 2, 2, 2)
>>> theta = band("theta")
>>> difference = jackknife_confidence_interval(
...     result.estimate[:, beta] - result.estimate[:, theta],
...     result.replicates[:, :, beta] - result.replicates[:, :, theta],
... )
>>> low, high = (bound[0, 0, 1] for bound in difference.confidence_interval)
>>> print(round(float(low), 2), round(float(high), 2))  # excludes 0
0.6 0.66

```

Do not average replicates from the default `leave_out="observation"` over
frequencies: they leave out tapers, and the interval comes out too narrow.

## Select a frequency range

`frequency_range=(low, high)` includes both endpoints. With 0.5-second windows
at 500 Hz the bins are 2 Hz apart, so `(10, 50)` keeps the 10 and 50 Hz bins:

```python
>>> settings = {"sampling_frequency": 500, "time_window_duration": 0.5}
>>> cropped = multitaper_connectivity(
...     time_series, method="magnitude_squared_coherence", frequency_range=(10, 50), **settings
... )
>>> cropped.frequency.values[[0, -1]].tolist()
[10.0, 50.0]

```

Bin `f` lies on bound `b` when `abs(f - b) <= 1e-6 * gap(f) + 2 * (ulp(f) +
ulp(b))`: within one part in a million of the distance to its nearest
neighbouring bin, plus two units in the last place of each stored value; a
bound farther from every bin is an ordinary edge between bins. A decimal bound
names a bin only when the bin's stored value has drifted from that decimal by
at most one part in a million of its spacing (plus rounding): float32
coordinates, `0.1 * k`, `np.linspace`, `rfftfreq`, and
`np.arange(1.0, 100.0, 0.1)` stay within that, while
`np.arange(1000.0, 1100.0, 0.001)` (drift up to 2.4e-6 of its spacing) and
float32 cumulative sums do not; take bounds from the coordinate for those.
Below, bin 142 of `np.arange(1.0, 100.0, 0.1)` is stored as
`15.200000000000014` and is still the 15.2 bin.

```python
>>> import xarray as xr
>>> frequencies = np.arange(1.0, 100.0, 0.1)
>>> coefficients = xr.DataArray(
...     rng.standard_normal((8, frequencies.size, 2))
...     + 1j * rng.standard_normal((8, frequencies.size, 2)),
...     dims=("trial", "frequency", "signal"),
...     coords={"frequency": frequencies},
... )
>>> edge = fourier_connectivity(
...     coefficients, method="magnitude_squared_coherence", frequency_range=(10, 15.2)
... )
>>> round(float(edge.frequency[-1]), 9), edge.sizes["frequency"]
(15.2, 53)

```

`delay`, `group_delay`, and `phase_slope_index` receive the wrapper's
`frequency_range` and consume only those bins; every other measure is computed
from the full spectrum (the directed measures factorize all of it) and then
cropped. Every variable the range applied to records the interval:

```python
>>> psi = multitaper_connectivity(
...     time_series, method="phase_slope_index", frequency_range=(10, 50), **settings
... )
>>> psi.attrs["selected_frequency_lower"], psi.attrs["selected_frequency_upper"]
(10.0, 50.0)
>>> psi.attrs["frequency_interval_closed"]
'both'

```

`requested_frequency_lower`/`_upper` hold the bounds as given and
`selected_frequency_lower`/`_upper` the lowest and highest bins in the
interval, before any `frequency_resolution` subsampling (with it,
`phase_slope_index` sums only every independent step of those bins, so its last
summed bin can lie below `selected_frequency_upper`). A Morlet result's
`valid_time` is likewise `True` only where every bin of the interval has full
support, which can be stricter than the bins `phase_slope_index` sums. Phase
slope and group delay have no frequency axis, so `frequency_bands` and
`frequency_decimation` leave them as computed over `frequency_range`; call once
per band for band values. `delay` keeps its frequency axis and is cropped,
decimated, and band-averaged like any other measure.

Adjacent bands such as `(10, 20)` and `(20, 30)` both average a bin lying at
exactly 20 Hz. Band integrals give each bin the part of its frequency cell
inside the band, so they stay additive and count no power twice:

```python
>>> from spectral_connectivity import frequency_band_reduce
>>> power = multitaper_connectivity(time_series, method="power", **settings)
>>> band_power = frequency_band_reduce(
...     power,
...     {"low": (10, 20), "high": (20, 30), "both": (10, 30)},
...     frequency_reduction="integral",
... )
>>> adjacent = band_power.sel(band="low") + band_power.sel(band="high")
>>> bool(np.allclose(adjacent, band_power.sel(band="both")))
True

```

Integrate before cropping. `frequency_band_reduce` covers the cells of the bins
its input has, so on a result already cropped by `frequency_range` or `sel`, a
band whose edge falls between the last kept bin and its dropped neighbour
comes out low by the part of the band in that neighbour's cell (up to half a
bin's cell at each end: 2.5% of a 20 Hz band per end on 1 Hz bins). A band
limits itself, so reduce the uncropped
result, or crop at least one bin wider than the bands. The wrappers
(`multitaper_connectivity`, `fourier_connectivity`, and `connectivity_to_xarray`,
the route for `ShortTimeFourierTransform`, `Welch`, and `MorletWavelet` results)
integrate the uncropped grid themselves when given `frequency_range` together
with `frequency_bands`, so pass the bands to the wrapper rather than reducing
its cropped result.

Before 3.0 these three measures took `frequencies_of_interest=(low, high)`,
which excludes a bin lying on either edge. It still works through 3.x, with
results recording `frequency_interval_closed="neither"` and a
`DeprecationWarning` naming the `frequency_range` that keeps the same bins:
on this grid `phase_slope_index(frequencies_of_interest=(10, 50))` becomes
`phase_slope_index(frequency_range=(12.0, 48.0))`. It is removed in 4.0.

## Compare spike-field coherence across firing rates

Spike-field coherence falls when a neuron fires less, even if its coupling to
the field is unchanged. `coherence_rate_adjustment` gives the factor by which
it falls when spikes are removed at random, so the coherence at a lower rate
can be compared with what the rate change alone predicts.

Give the transform spike counts per sample (0 or 1), not a rate, and pass the
bin width as `dt = 1 / sampling_frequency`. The spectrum to pass is
`Connectivity.power()` of the higher-rate spike train, which is a one-sided
density: say so with `is_one_sided=True`.

```python
>>> from spectral_connectivity import Connectivity, Multitaper
>>> from spectral_connectivity.statistics import coherence_rate_adjustment
>>> sampling_frequency = 500
>>> time = np.arange(250)[:, np.newaxis] / sampling_frequency
>>> oscillation = np.cos(2 * np.pi * 40 * time + rng.uniform(0, 2 * np.pi, (1, 400)))
>>> field = oscillation + 0.5 * rng.standard_normal(oscillation.shape)
>>> spikes = rng.poisson(100 * (1 + 0.9 * oscillation) / sampling_frequency)
>>> fewer_spikes = rng.binomial(spikes, 0.25)  # a quarter of the spikes, same coupling
>>> signals = np.stack([field, spikes, fewer_spikes], axis=-1).astype(float)
>>> connectivity = Connectivity.from_transform(
...     Multitaper(signals, sampling_frequency=sampling_frequency)
... )
>>> coherence = np.abs(connectivity.coherency()[0])  # (frequency, signal, signal)
>>> factor = coherence_rate_adjustment(
...     spikes.mean() * sampling_frequency,  # spikes per second
...     fewer_spikes.mean() * sampling_frequency,
...     connectivity.power(one_sided=True)[0, :, 1],
...     dt=1 / sampling_frequency,
...     is_one_sided=True,
... )
>>> at_40_hz = int(np.argmin(np.abs(connectivity.frequencies - 40)))
>>> expected = coherence[at_40_hz, 0, 1] * factor[at_40_hz]
>>> bool(abs(coherence[at_40_hz, 0, 2] - expected) < 0.05)
True

```

The factor multiplies the coherence magnitude, not its square. Leave out the
0 Hz bin and, with the default mean removal, the bins within a taper's
bandwidth of it, where removing each window's mean distorts the spike
spectrum. At
the Nyquist frequency the factor is valid only with a two-sided spectrum:
`connectivity.power(one_sided=False)` with `is_one_sided=False`. Without
`is_one_sided` the function assumes a one-sided spectrum and warns.

## Find line noise with the harmonic F-test

`Multitaper.harmonic_f_test()` tests every frequency of every window, trial,
and signal for a sinusoid that keeps its phase across the window (Thomson's
harmonic F-test), such as line noise or a stimulation artifact, and fits its
amplitude and phase. Each frequency is a test: Thomson's rule takes a line as
detected where the p-value is below `1 / n_time_samples_per_window`, which
also admits a frequency or two per window and signal by chance. A line present
throughout the recording is detected in most windows; a chance detection is
not.

```python
>>> from spectral_connectivity import Multitaper
>>> sampling_frequency = 1000
>>> time = np.arange(4000) / sampling_frequency
>>> rng = np.random.default_rng(2)
>>> recording = rng.standard_normal((4000, 1, 2))  # (time, trials, signals)
>>> # 60 Hz line noise and its harmonic, in signal 0 only
>>> recording[:, 0, 0] += 0.5 * np.cos(2 * np.pi * 60 * time)
>>> recording[:, 0, 0] += 0.5 * np.cos(2 * np.pi * 120 * time + 0.5)
>>> multitaper = Multitaper(
...     recording, sampling_frequency, time_window_duration=1.0,
...     time_halfbandwidth_product=4,
... )
>>> lines = multitaper.harmonic_f_test()
>>> lines.p_value.shape  # (n_time_windows, n_trials, n_frequencies, n_signals)
(4, 1, 2001, 2)
>>> detected = lines.p_value < 1 / multitaper.n_time_samples_per_window
>>> # The fraction of windows (and trials) in which each frequency is detected
>>> fraction = detected.mean(axis=(0, 1))  # (n_frequencies, n_signals)
>>> [lines.frequencies[fraction[:, signal] > 0.5].tolist() for signal in range(2)]
[[59.75, 60.0, 60.25, 119.75, 120.0, 120.25], []]
>>> # The bins next to a line, within the half-bandwidth, are significant too:
>>> # take each line at the local maximum of its typical F statistic
>>> typical_f = np.median(lines.f_statistic, axis=(0, 1))
>>> is_peak = np.zeros(typical_f.shape, dtype=bool)
>>> is_peak[1:-1] = (typical_f[1:-1] > typical_f[:-2]) & (typical_f[1:-1] >= typical_f[2:])
>>> [lines.frequencies[(fraction[:, s] > 0.5) & is_peak[:, s]].tolist() for s in range(2)]
[[60.0, 120.0], []]
>>> at_lines = np.isin(lines.frequencies, [60.0, 120.0])
>>> np.round(np.median(lines.amplitude[:, 0, at_lines, 0], axis=0), 1).tolist()
[0.5, 0.5]

```

The test pads each window to four times its length (`lines.frequencies` is
spaced 0.25 Hz here, against the window's 1 Hz bins), since a line between the
window's own bins can be missed; `harmonic_f_test(n_fft_samples=...)` sets
another grid. Frequencies within the half-bandwidth (`lines.half_bandwidth`)
of 0 Hz and Nyquist are NaN, because there a line overlaps its mirror image.
A narrow rhythm in the background also tests as line-like. The Notes of
`Multitaper.harmonic_f_test` give the sizes of these effects. For a false
discovery rate over the frequencies of each window, trial, and signal, pass
the p-values to `adjust_for_multiple_comparisons` with `axis=2`. To remove
the lines found, see the next recipe.

## Remove line noise before computing connectivity

`line_frequencies` makes the transform fit the named lines in every window,
trial, and signal (at exactly those frequencies, by the same harmonic
regression) and subtract them before the coefficients are computed, so every
measure uses the cleaned data. Through the wrappers it is a `transform_kwargs`
setting:

```python
>>> settings = dict(
...     sampling_frequency=sampling_frequency, time_window_duration=1.0,
...     time_halfbandwidth_product=4, method="power",
... )
>>> raw = multitaper_connectivity(recording, **settings)
>>> cleaned = multitaper_connectivity(
...     recording, transform_kwargs={"line_frequencies": [60, 120]}, **settings
... )
>>> def power_at(result, frequency):
...     return float(result.sel(frequency=frequency).isel(source=0).mean())
>>> # Before, 60 and 120 Hz stand far above the noise (at 90 Hz); after, they do not
>>> [power_at(raw, f) > 5 * power_at(raw, 90) for f in (60, 120)]
[True, True]
>>> [power_at(cleaned, f) < 1.5 * power_at(cleaned, 90) for f in (60, 120)]
[True, True]
>>> cleaned.attrs["mt_n_tapers"], "line_frequencies" in cleaned.attrs["transform_parameters_json"]
(7, True)

```

Name the lines the previous recipe finds. Three things to know, with their
sizes in the `line_frequencies` docs of `Multitaper`: a line must lie close to
its named frequency; the fit also takes about `1 / n_tapers` of the noise near
each line (a shallow notch in the spectrum); and a test there that counts
`n_observations`, such as `coherence_significance_pvalue`, is then too liberal,
so count `n_observations - n_trials` within the half-bandwidth of a removed
line. To subtract a line only where it is present, add
`line_significance_threshold` (a strict level such as `1e-3`, since each
window, trial, and signal is a test).

## Bring your own Fourier coefficients

If you already have Fourier coefficients (e.g. from a wavelet transform), skip
the multitaper step and use `fourier_connectivity`. NumPy inputs may use the
`(observation, frequency, signal)` layout shown here.

```python
>>> coefficients = rng.standard_normal((20, 16, 2)) + 1j * rng.standard_normal(
...     (20, 16, 2)
... )
>>> frequencies = np.linspace(0, 250, 16)
>>> byo = fourier_connectivity(
...     coefficients, frequencies=frequencies, method="magnitude_squared_coherence"
... )
>>> byo.dims
('time', 'frequency', 'source', 'target')
>>> byo.sizes["frequency"]
16

```

## Plug in your own transform

`Connectivity.from_transform` accepts any object that satisfies the
`SpectralTransform` protocol: an `fft()` method returning coefficients shaped
`(n_time_windows, n_trials, n_tapers, n_fft_samples, n_signals)`, plus
`frequencies` and `time`. No subclassing is needed. `help(SpectralTransform)`
lists the optional attributes such as `is_one_sided` (absent ones describe a
two-sided, unweighted, independent spectrum) and the scaling that makes
`power()` a density. `fft()` must return fresh, unshared storage on each call.
Neither the transform nor its caller may subsequently mutate it through any
alias, because `Connectivity` keeps it without copying. Read-only flags are
applied where the backend supports them. The example below is not scaled to a
density, so its `power()` is in arbitrary units; normalized measures such as
coherence are unaffected.

```python
>>> from spectral_connectivity import Connectivity, SpectralTransform
>>> class HannTransform:
...     """One Hann-windowed FFT per trial, non-negative frequencies only."""
...
...     is_one_sided = True  # optional; omit for a two-sided FFT-order spectrum
...
...     def __init__(self, time_series, sampling_frequency):
...         self.time_series = time_series  # (n_time_samples, n_trials, n_signals)
...         n_time_samples = time_series.shape[0]
...         self.frequencies = np.fft.rfftfreq(n_time_samples, d=1 / sampling_frequency)
...         self.time = np.array([n_time_samples / 2 / sampling_frequency])
...
...     def fft(self):
...         window = np.hanning(self.time_series.shape[0])[:, np.newaxis, np.newaxis]
...         coefficients = np.fft.rfft(window * self.time_series, axis=0)
...         # (frequency, trial, signal) -> (time, trial, taper, frequency, signal)
...         return coefficients.transpose(1, 0, 2)[np.newaxis, :, np.newaxis]
>>> transform = HannTransform(time_series, sampling_frequency=500)
>>> isinstance(transform, SpectralTransform)
True
>>> Connectivity.from_transform(transform).magnitude_squared_coherence().shape
(1, 501, 3, 3)

```

## Exact connectivity of a VAR model

`VARSpectrum` is a transform whose spectrum is that of a vector autoregressive
(VAR) model, `x(t) = sum_k A_k x(t - k) + e(t)`, so every measure of the
cross-spectral matrix it gives is the model's population value, with no
estimation error: what an estimate of data simulated from the model should
find, or the connectivity of a model fitted elsewhere. Label its results with
`connectivity_to_xarray(transform, method)`, one measure per call
(`multitaper_connectivity` estimates from a time series and does not take a
transform); `Connectivity.from_var(...)`, the shortcut for
`Connectivity.from_transform(VARSpectrum(...))`, gives the arrays.
`coefficients[k - 1]` is `A_k`, with the lag on the first axis (for MVGC's
MATLAB layout `A(:, :, k)`, pass `np.moveaxis(A, -1, 0)`), and its entry
`[i, j]` is the effect of signal `j` on signal `i`, as in `simulate_MVAR`: it
appears in the results as `source=j`, `target=i`. `noise_covariance` is the
covariance of `e(t)` (the identity by default).

```python
>>> from spectral_connectivity import Connectivity, VARSpectrum
>>> from spectral_connectivity.simulate import simulate_MVAR
>>> from spectral_connectivity.wrapper import connectivity_to_xarray
>>> coefficients = np.array(
...     [[[0.5, 0.0], [0.4, 0.5]], [[-0.6, 0.0], [0.0, -0.6]]]
... )  # (n_lags, n_signals, n_signals): signal "a" drives "b", never the reverse
>>> var = VARSpectrum(coefficients, sampling_frequency=500)
>>> granger = connectivity_to_xarray(
...     var, "pairwise_spectral_granger_prediction", signal_names=["a", "b"]
... )
>>> bool(granger.sel(source="a", target="b").max() > 0.1)
True
>>> bool(granger.sel(source="b", target="a").max() < 1e-9)  # exactly 0 up to rounding
True

```

`power` is the model's one-sided power spectral density, in the units a
multitaper estimate of the simulated process converges to. Give the model the
estimate's frequency grid (an `n_fft_samples` equal to the estimate's, or a
multiple of it), and, without detrending (which removes 0 Hz), 200 trials of
one-second windows agree with it to within their sampling error, 3% per bin:

```python
>>> simulated = simulate_MVAR(coefficients, n_time_samples=500, n_trials=200, random_state=0)
>>> estimate = multitaper_connectivity(
...     simulated, sampling_frequency=500, method="power", detrend_type=None
... ).isel(time=0)
>>> on_its_grid = VARSpectrum(coefficients, sampling_frequency=500, n_fft_samples=500)
>>> true_power = connectivity_to_xarray(on_its_grid, "power").isel(time=0)
>>> ratio = estimate / true_power
>>> bool(abs(float(ratio.mean()) - 1) < 0.02), bool(abs(ratio - 1).max() < 0.15)
(True, True)

```

The same comparison for spectral Granger prediction shows how far an estimate
of these 200 trials lies from the model's value (0.75 at its peak):

```python
>>> granger_estimate = multitaper_connectivity(
...     simulated,
...     sampling_frequency=500,
...     method="pairwise_spectral_granger_prediction",
...     detrend_type=None,
...     signal_names=["a", "b"],
... ).isel(time=0)
>>> true_granger = connectivity_to_xarray(
...     on_its_grid, "pairwise_spectral_granger_prediction", signal_names=["a", "b"]
... ).isel(time=0)
>>> error = abs(granger_estimate - true_granger)
>>> bool(error.sel(source="a", target="b").max() < 0.1)
True
>>> bool(error.sel(source="b", target="a").max() < 0.01)
True

```

The coefficients that `VARSpectrum.fft()` returns are one factorization of the
model's spectrum, not samples, so the measures computed from each trial and
taper or from their count (the phase-locking and phase-lag families, pairwise
phase consistency, `delay`, `group_delay`) raise rather than return values that
depend on that choice, as do `jackknife`, `surrogate_test`, and
`partial_coherence(shrinkage="auto")`, which estimate sampling error. Estimate those from simulated data.

```python
>>> Connectivity.from_var(coefficients, sampling_frequency=500).phase_locking_value()
Traceback (most recent call last):
...
ValueError: phase_locking_value is not defined for an exact spectrum ...

```

A model fitted by statsmodels (not a dependency of this package) has the same
convention. `fitted.coefs` is `(n_lags, n_signals, n_signals)` and
`fitted.sigma_u` the innovation covariance. The fitted intercept is ignored
(it sets the mean, not the spectrum), and `VARSpectrum` refuses an unstable
fit, which `fitted.is_stable()` reports beforehand:

```python
from statsmodels.tsa.api import VAR

fitted = VAR(simulated[:, 0, :]).fit(maxlags=2)  # one trial: (n_time_samples, n_signals)
assert fitted.is_stable()
connectivity = Connectivity.from_var(fitted.coefs, fitted.sigma_u, sampling_frequency=500)
```

Give `coefficients` a leading `n_time_windows` axis for a time-varying model
(one model per window, with `noise_covariance` shared or per window, and
`time=` the window times in seconds). The frequency grid defaults to the
shortest one that does not limit the directed measures' accuracy
(`VARSpectrum.minimum_n_fft_samples`), which grows as the model's poles near
the unit circle. A shorter `n_fft_samples` warns: power and the coherence
family stay exact on any grid, the directed measures do not. On the default
grid the directed measures are exact to the Wilson factorization's tolerance
(`minimum_phase_tolerance`, 1e-8 by default), and to float64 rounding where
the spectrum is nearly singular.

## Result schema and transform bandwidth

The xarray interfaces share `output_schema_version=1`. Common provenance is on
the Dataset and each measure variable, so extracting a variable retains the
facts needed to interpret it. Settings use one JSON record:

| Attribute | Meaning |
| --- | --- |
| `output_schema_version` | `1`: the version of the layout this table describes. |
| `package`, `package_version` | `spectral_connectivity` and the installed version that computed the result. |
| `input_attrs_json` | The attributes of an input `xarray.DataArray` (e.g. subject or session metadata), as one canonical JSON record, so arbitrary keys keep their names; an array value of more than 100 elements is recorded as its shape and dtype. Present only when the input had attributes. |
| `transform` | `multitaper`, `stft`, `welch`, `morlet`, `var_spectrum`, or `external_fourier`. |
| `sampling_frequency` | Physical sampling rate in Hz, included only when known. |
| `n_trials`, `n_signals` | Known trial and signal counts; an arbitrary external observation axis is not labeled as trials. |
| `n_observations` | Raw count averaged by the expectation, using the actual retained tapers/segments. Not recorded for an exact spectrum. |
| `expectation_type` | Axes averaged to form each spectral estimate. |
| `observations_are_independent`, `time_bins_are_independent` | Recorded independence assumptions, stored as 0/1 for NetCDF. The raw observation count is not an effective independent sample size. `observations_are_independent` is not recorded for an exact spectrum. |
| `observations_are_time_segments` | 1 when a trial's observations are segments of it in time (`Welch`), 0 for tapers of the same samples; stored as 0/1. Not recorded for an exact spectrum. |
| `is_exact_spectrum` | 1 when the coefficients are one factor of an exact cross-spectral matrix rather than samples (`VARSpectrum`): the values are a model's population values, and the result records no observation count or independence. 0 otherwise. |
| `backend` | `cpu` or `gpu`, reflecting the imported backend. |
| `transform_parameters_json` | Remaining settings in `estimator` and `execution` records; optional settings use JSON `null`. |
| `spectral_bandwidth`, `spectral_bandwidth_definition`, `spectral_bandwidth_units` | Qualified estimator bandwidth, when defined. |
| `frequency_bin_spacing`, `frequency_bin_spacing_units` | Spacing of the uniform frequency grid as returned, after the wrapper's cropping and decimation. Omitted for irregular/singleton grids and outputs without a frequency axis. xarray selections such as `.isel`/`.sel` keep attrs unchanged, so recompute spacing from the `frequency` coordinate after selecting bins yourself. |
| `nyquist_frequency` | Nyquist frequency of the computed grid when known: in Hz when the sampling rate is known, 0.5 cycles/sample for normalized external grids. Used to integrate the Nyquist bin after cropping. |
| `frequency_units` | `Hz` or `cycles/sample`, recorded for external Fourier inputs and for a transform of your own that leaves `frequencies` as `None`. |
| `frequency_coordinate`, `time_coordinate` | External Fourier inputs: whether each coordinate was `provided` or filled in (`normalized` frequencies, window `index` time). A transform of your own that leaves `frequencies` or `time` as `None` records the filled-in value. |
| `requested_frequency_lower`, `requested_frequency_upper` | The bounds given as `frequency_range` (or the deprecated `frequencies_of_interest`), in the frequency coordinate's units; an infinite bound of the deprecated keyword records nothing for that side. The interval record below is on every variable a `frequency_range` applied to: each frequency-resolved variable the wrappers cropped, and every variable of `delay`, `group_delay`, and `phase_slope_index`, which consume the interval and carry the record even without one (then every bin, with no requested bounds). Not on the Dataset itself. |
| `selected_frequency_lower`, `selected_frequency_upper` | The lowest and highest bins in that interval, before any `frequency_resolution` subsampling or `frequency_decimation`. |
| `frequency_interval_closed` | `"both"` when the bins on the bounds are included (`frequency_range`, or every bin), `"neither"` for the deprecated `frequencies_of_interest`. |
| `frequency_decimation`, `frequency_bands_json`, `frequency_reduction` | The wrapper's decimation and band reduction, on the frequency-resolved variables they were applied to. Decimation and band reduction keep the interval record, which still names the interval they were drawn from; the `frequency` (or `band`) coordinate shows what was returned. |
| `measure`, `measure_kwargs_json` | The measure name the caller requested and its keyword arguments, on each measure's variables. A result requested by a deprecated name (`coherence_magnitude`) records that name, and its variable has that name too. |
| `arg_<name>`, `arg_<name>_json` | Each of those keyword arguments on its own: a scalar as `arg_<name>`, a structured or non-finite value as canonical JSON under `arg_<name>_json` (e.g. `arg_frequency_range_json`, `arg_group_labels_json`). `measure_kwargs_json` remains the full record. |
| `long_name`, `units`, `interpretation` | How to read the measure variable, from `list_measures()`. With `sign_convention`, the value range, and `diagonal_policy` (below), these are on each measure variable and on `group_delay`'s other value variables, `group_delay_slope` and `group_delay_r_value` (range [-1, 1], with the slope's sign), which have their own; group membership and spatial vectors, filters, and patterns carry none. Band reduction names its reduction (mean, circular mean, complex mean, or integral) at the start of `interpretation` and `formula`, and an integral rewrites `long_name` and `units`; cropping and decimation change none of the measure attributes. |
| `sign_convention` | Which signal leads, for measures whose sign or angle shows it; omitted for the others. |
| `value_range_lower`, `value_range_upper`, `value_range_applies_to` | The estimator's theoretical range (see `docs/CONNECTIVITY_METRIC_RANGES.md` for when a computed value can lie outside it); an unbounded side is omitted. For complex measures `value_range_applies_to="magnitude"`: the range bounds the magnitude. |
| `quantity`, `formula` | For the coherency family, whether the value is a `magnitude`, `squared_magnitude`, `signed` real-valued component, `phase`, or `complex`, and its definition; omitted for other measures. |
| `diagonal_policy` | What a self-connection holds, one of the policies defined in the policy table of `docs/CONNECTIVITY_METRIC_RANGES.md` (see "Mask or fill the diagonal"). On each measure variable, and on `group_delay_slope` and `group_delay_r_value`, whose diagonals differ. |
| `diagonal_value` | Where the policy is `constant` or `rounding_noise`: the (nominal) value, a float. Omitted for the other policies. |
| `nan_for_zero_power_signal` | 1 if every entry involving a zero-power (flat) signal is NaN, else 0 (defined below the policy table). On each measure variable. |

```python
>>> schema_result = multitaper_connectivity(
...     time_series, sampling_frequency=500, time_window_duration=0.5,
...     time_halfbandwidth_product=2, method="power",
... )
>>> schema_result.attrs["output_schema_version"], schema_result.attrs["transform"]
(1, 'multitaper')
>>> schema_result.attrs["n_trials"], schema_result.attrs["n_observations"]
(4, 12)
>>> schema_result.attrs["spectral_bandwidth"], schema_result.attrs["frequency_bin_spacing"]
(8.0, 2.0)
>>> import json
>>> settings = json.loads(schema_result.attrs["transform_parameters_json"])
>>> settings["estimator"]["time_halfbandwidth_product"]
2
>>> settings["execution"] == {"fft_workers": None}
True

```

Multitaper's familiar `frequency_resolution` name describes **full DPSS
concentration bandwidth**, `2 * NW / window_duration` Hz. The property,
`estimate_frequency_resolution`, `suggest_parameters(frequency_resolution=...)`,
and the returned `frequency_resolution` key remain supported without warnings.
STFT and Welch use `equivalent_noise_bandwidth`: for periodic Hann windows of
at least three samples it is `1.5 / window_duration` Hz. Their output can have
the same FFT grid as Multitaper while describing different spectral smoothing.
For example, a 0.5-second window with NW=2 has 8 Hz DPSS concentration bandwidth
and 3 Hz Hann equivalent noise bandwidth. Zero-padding or frequency decimation
changes grid spacing while preserving estimator bandwidth.

Custom tapers and Morlet outputs omit `spectral_bandwidth`: no width
convention is defined for them. Power and cross-spectral density omit `units` when input
units are unknown. External Fourier defaults use `cycles/sample` frequencies
and window-index time; these do not establish a physical sampling rate.
Supplied Fourier coordinates preserve their attrs; a DataArray frequency
coordinate declaring `units="cycles/sample"` also stays normalized. Supported
frequency units are `Hz` and `cycles/sample`, compared ignoring case and
whitespace. `Hz` may also be spelled `Hertz`, `cps`, `1/s`, `cycles/s`, `s-1`,
`s^-1`, or `s**-1` (with `s`, `sec`, or `second`, and `cycle` or `cycles`), and
`cycles/sample` as `cycle/sample`. Convert other units, such as `kHz` or
`rad/s`, before calling the wrapper. An empty `units` is treated as unstated,
so the default applies. The top-level `frequency_units` is the canonical
spelling, while the returned `frequency` coordinate otherwise keeps the
supplied attrs as given.
Whether external coefficients were one-sided is an input setting, recorded as
`is_one_sided` in `transform_parameters_json`; it does not describe the
returned frequency grid. A labeled or explicitly declared trial axis, or the
trial axis of a 4-D/5-D NumPy array, supplies `n_trials`, while an arbitrary
observation axis supplies only the raw observation count.
Unambiguous scalar recording coordinates such as `subject`/`session` keep
their attrs; one named like a coordinate or variable of the result (e.g.
`frequency`, or `band` when `frequency_bands` is given) is dropped with a
warning. Coefficients without a time dimension that keep a numeric scalar
coordinate named `time` (or `times`), as after `isel(time=0)`, use it as the
time of their single window; a scalar under another name of a time dimension,
such as `window`, stays an identifier, since it may hold a window's length
(pass `time=[...]` to use it as the time).
Labels along averaged trial axes are omitted.

The result-schema migration is separate from the supported public property and
helper names. The [migration
guide](migration.md)
maps each 2.0.1 `mt_*` attribute, and each attribute renamed during 3.0
development, to the keys above.

The 15 scalar `mt_*` attributes released in 2.0.1 remain compatibility copies
through 3.x and will be removed in 4.0: `mt_detrend_type`, `mt_is_low_bias`,
`mt_sampling_frequency`, `mt_start_time`, `mt_time_halfbandwidth_product`,
`mt_n_fft_samples`, `mt_n_signals`, `mt_n_tapers`, `mt_n_time_samples_per_step`,
`mt_n_time_samples_per_window`, `mt_n_trials`, `mt_time_window_duration`,
`mt_time_window_step`, `mt_frequency_resolution`, and `mt_nyquist_frequency`.
Attribute dictionaries cannot warn on access. These keys are documented as
deprecated. New settings such as FFT workers or adaptive weighting do not get
legacy copies.

The copies equal their 2.0.1 values but are stored as scalars NetCDF can save,
so three representations differ: `mt_is_low_bias` is `1` or `0` instead of a
`bool`, `mt_start_time` and `mt_sampling_frequency` are floats, and a `None`
setting such as `mt_detrend_type` is the string `"None"`. Compare them with
`==`, not `is`. `mt_time_window_duration`, `mt_time_window_step`, and
`mt_frequency_resolution` describe the window actually analyzed, which is a
whole number of samples: a 0.505 s window at 100 Hz is 50 samples, so they
report 0.5 s and, for NW=3, 12 Hz where 2.0.1 reported the requested 0.505 s
and 11.88 Hz. `mt_n_time_samples_per_step` is the step rounded to the nearest
sample, where 2.0.1 truncated it (57, not 56, for a 0.57 s step at 100 Hz).
`mt_n_tapers` is the number of tapers used, where 2.0.1 reported the count
requested (`n_tapers` if given, else `floor(2 * NW - 1)`): with custom `tapers`
it is the number supplied (2.0.1: 5 for two tapers at the default NW=3), and
for DPSS tapers it excludes those low-bias filtering dropped (2.0.1: 50 for
`n_tapers=50` at NW=3, where 5 are used). The request is
`requested_n_tapers` in the `estimator` record of `transform_parameters_json`
(`null` when `n_tapers` was not given). `estimate_frequency_resolution` uses
the duration it is given.

## Pass a labeled DataArray

`time_series` may be an `xarray.DataArray`. Dimension names, not positions,
define the roles, so a `time` coordinate in seconds supplies the sampling
frequency and the signal labels become the `source`/`target` coordinates. A
coordinate in milliseconds would give a rate 1000 times too low; inference warns
when the rate is at most 1 Hz or the coordinate's `units` attribute says
milliseconds. A millisecond axis of data sampled faster than 1 kHz is not
detected without that attribute, so set `units` or pass `sampling_frequency`:

```python
>>> import xarray as xr
>>> da = xr.DataArray(
...     rng.standard_normal((1000, 4, 3)),
...     dims=("time", "trial", "signal"),
...     coords={"time": np.arange(1000) / 500, "signal": ["CA1", "CA3", "PFC"]},
... )
>>> coherence = multitaper_connectivity(da, method="magnitude_squared_coherence")
>>> coherence.attrs["sampling_frequency"]
500.0
>>> coherence.source.values.tolist()
['CA1', 'CA3', 'PFC']

```

For other dimension names, say which dimension plays which role:

```python
>>> custom = da.rename(time="t", trial="epoch", signal="channel")
>>> coherence = multitaper_connectivity(
...     custom,
...     method="magnitude_squared_coherence",
...     time_dim="t",
...     trial_dim="epoch",
...     signal_dim="channel",
... )
>>> coherence.attrs["sampling_frequency"], coherence.target.values.tolist()
(500.0, ['CA1', 'CA3', 'PFC'])

```

Common cases:

- Common time, trial and signal dimension names are recognized and transposed
  automatically; pass `time_dim`, `trial_dim` and `signal_dim` for others.
- A numeric `time` coordinate is elapsed seconds and supplies
  `sampling_frequency`. A numeric `sample` coordinate is sample numbers, which
  have no time scale, so pass `sampling_frequency` with it. Either one labels
  the output window centers, and a `sampling_frequency` you pass is checked
  against it.
- A 1-D index on the signal dimension, including its label type, becomes the
  `source`/`target` coordinates unless you pass `signal_names`.

If you hit an error or a warning:

- Ambiguous dimension names raise instead of falling back to axis position:
  name the roles with `time_dim`, `trial_dim` and `signal_dim`. When a single
  unrecognized dimension is left for the one remaining role, it is assigned by
  elimination and a warning names the mapping.
- The transform classes (`Multitaper`, `ShortTimeFourierTransform`, `Welch`,
  `MorletWavelet`) read a DataArray by position, as `(time, trial, signal)`,
  and take no `*_dim` arguments. One whose dimension names put time, trials or
  signals on another axis raises `ValueError` naming the misplaced dimensions
  and, when the names determine it, the transpose to apply;
  names that say nothing of a role (`dim_0`, `session`) are read by position.
- Inferring the rate needs enough coordinate precision: pass
  `sampling_frequency` for low-precision or large-offset time coordinates.
- Signal labels must be unique, non-missing, NetCDF-compatible scalars
  (strings, real numbers, datetimes or timedeltas); integer labels must fit the
  signed 32-bit range for portable NetCDF3 files.
- Datetime, timedelta and object-valued time coordinates are not supported
  yet; convert them to elapsed seconds as below. (Datetime and timedelta
  *signal labels* are fine.)
- A dask-backed DataArray is rejected; call `.compute()` (or `.load()`) first.

```python
>>> stamped = da.assign_coords(
...     time=np.datetime64("2024-01-01T00:00", "ns") + np.arange(1000) * np.timedelta64(2, "ms")
... )
>>> elapsed = stamped.assign_coords(
...     time=(stamped.time - stamped.time[0]) / np.timedelta64(1, "s")
... )
>>> multitaper_connectivity(elapsed, method="magnitude_squared_coherence").attrs["sampling_frequency"]
500.0

```

## Where to go next

- Value ranges for every measure: `docs/CONNECTIVITY_METRIC_RANGES.md`.
- The full lower-level API: the `Connectivity` class (each `method` above is a
  method on it).
- End-to-end tutorials: the notebooks under `examples/`.
