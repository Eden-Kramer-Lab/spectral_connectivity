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
35
>>> [m.name for m in list_measures(default_only=True)][:3]
['coherence_magnitude', 'coherence_phase', 'debiased_squared_phase_lag_index']
>>> [m.name for m in list_measures(directed=True)][:2]
['pairwise_spectral_granger_prediction', 'directed_phase_lag_index']
>>> next(m for m in list_measures() if m.name == "phase_slope_index").requires_two_sided
False
>>> power = next(m for m in list_measures() if m.name == "power")
>>> power.category, power.description
('power', 'Return the one-sided power spectral density of the signal.')

```

Passing an unknown name raises a helpful error rather than an obscure
`AttributeError`:

```python
>>> multitaper_connectivity(
...     time_series, sampling_frequency=500, method="coherence"
... )
Traceback (most recent call last):
    ...
ValueError: 'coherence' is not a known connectivity measure. Did you mean: 'coherence_magnitude', 'coherence_phase', 'imaginary_coherence', 'partial_coherence', 'directed_coherence'? Call spectral_connectivity.list_measures() to see the 35 available measures.

```

## Functional connectivity: coherence

The high-level `multitaper_connectivity` runs the multitaper transform and
returns a labeled `xarray.DataArray` with `(time, frequency, source, target)`
axes.

```python
>>> coherence = multitaper_connectivity(
...     time_series,
...     sampling_frequency=500,
...     method="coherence_magnitude",
...     time_halfbandwidth_product=3,
... )
>>> type(coherence).__name__
'DataArray'
>>> coherence.dims
('time', 'frequency', 'source', 'target')
>>> coherence.name
'coherence_magnitude'

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

## Compute several measures at once

Pass a list of methods to get an `xarray.Dataset` with one variable per
measure. Shared spectra are cached, so this is cheaper than separate calls.

```python
>>> result = multitaper_connectivity(
...     time_series,
...     sampling_frequency=500,
...     method=["power", "coherence_magnitude"],
...     time_halfbandwidth_product=3,
... )
>>> type(result).__name__
'Dataset'
>>> sorted(result.data_vars)
['coherence_magnitude', 'power']
>>> result["power"].dims
('time', 'frequency', 'source')

```

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

## Collapse into frequency bands

Pass `frequency_bands` to average (or integrate) each measure within named
bands. The `frequency` axis is replaced by a labeled `band` axis.

```python
>>> banded = multitaper_connectivity(
...     time_series,
...     sampling_frequency=500,
...     method="coherence_magnitude",
...     frequency_bands={"theta": (4, 8), "gamma": (30, 50)},
...     time_halfbandwidth_product=3,
... )
>>> banded.dims
('time', 'band', 'source', 'target')
>>> banded.coords["band"].values.tolist()
['theta', 'gamma']

```

## Select a frequency range

`frequency_range=(low, high)` includes both endpoints. With 0.5-second windows
at 500 Hz the bins are 2 Hz apart, so `(10, 50)` keeps the 10 and 50 Hz bins:

```python
>>> settings = {"sampling_frequency": 500, "time_window_duration": 0.5}
>>> cropped = multitaper_connectivity(
...     time_series, method="coherence_magnitude", frequency_range=(10, 50), **settings
... )
>>> cropped.frequency.values[[0, -1]].tolist()
[10.0, 50.0]

```

Bin `f` lies on bound `b` when `abs(f - b) <= 1e-6 * gap(f) + 2 * (ulp(f) +
ulp(b))`: within one part in a million of the distance to its nearest
neighbouring bin, plus two units in the last place of each stored value. So a
decimal bound names the bin printed as that decimal even where the stored value
has drifted from it, as on `np.arange` grids with a nonzero start (bin 142
below is stored as `15.200000000000014`); a bound farther from every bin is an
ordinary edge between bins.

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
...     coefficients, method="coherence_magnitude", frequency_range=(10, 15.2)
... )
>>> round(float(edge.frequency[-1]), 9), edge.sizes["frequency"]
(15.2, 53)

```

`delay`, `group_delay`, and `phase_slope_index` receive the wrapper's
`frequency_range` and consume only those bins; every other measure is computed
from the full spectrum (the directed measures factorize all of it) and then
cropped. The frequency-reduced measures record the interval on their variables:

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
`selected_frequency_lower`/`_upper` the lowest and highest bins consumed. Phase
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

Before 3.0 these three measures took `frequencies_of_interest=(low, high)`,
which excludes a bin lying on either edge. It still works through 3.x, with
results recording `frequency_interval_closed="neither"` and a
`DeprecationWarning` naming the `frequency_range` that keeps the same bins:
on this grid `phase_slope_index(frequencies_of_interest=(10, 50))` becomes
`phase_slope_index(frequency_range=(12.0, 48.0))`. It is removed in 4.0.

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
...     coefficients, frequencies=frequencies, method="coherence_magnitude"
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
>>> Connectivity.from_transform(transform).coherence_magnitude().shape
(1, 501, 3, 3)

```

## Result schema and transform bandwidth

The xarray interfaces share `output_schema_version=1`. Common provenance is on
the Dataset and each measure variable, so extracting a variable retains the
facts needed to interpret it. Settings use one JSON record:

| Attribute | Meaning |
| --- | --- |
| `transform` | `multitaper`, `stft`, `welch`, `morlet`, or `external_fourier`. |
| `sampling_frequency` | Physical sampling rate in Hz, included only when known. |
| `n_trials`, `n_signals` | Known trial and signal counts; an arbitrary external observation axis is not labeled as trials. |
| `n_observations` | Raw count averaged by the expectation, using the actual retained tapers/segments. |
| `expectation_type` | Axes averaged to form each spectral estimate. |
| `observations_are_independent`, `time_bins_are_independent` | Recorded independence assumptions, stored as 0/1 for NetCDF. The raw observation count is not an effective independent sample size. |
| `backend` | `cpu` or `gpu`, reflecting the imported backend. |
| `transform_parameters_json` | Remaining settings in `estimator` and `execution` records; optional settings use JSON `null`. |
| `spectral_bandwidth`, `spectral_bandwidth_definition`, `spectral_bandwidth_units` | Qualified estimator bandwidth, when defined. |
| `frequency_bin_spacing`, `frequency_bin_spacing_units` | Spacing of the uniform frequency grid as returned, after the wrapper's cropping and decimation. Omitted for irregular/singleton grids and outputs without a frequency axis. xarray selections such as `.isel`/`.sel` keep attrs unchanged, so recompute spacing from the `frequency` coordinate after selecting bins yourself. |
| `nyquist_frequency` | Nyquist frequency of the computed grid when known: in Hz when the sampling rate is known, 0.5 cycles/sample for normalized external grids. Used to integrate the Nyquist bin after cropping. |
| `frequency_units` | `Hz` or `cycles/sample`, recorded for external Fourier inputs and for a transform of your own that leaves `frequencies` as `None`. |
| `frequency_coordinate`, `time_coordinate` | External Fourier inputs: whether each coordinate was `provided` or filled in (`normalized` frequencies, window `index` time). A transform of your own that leaves `frequencies` or `time` as `None` records the filled-in value. |
| `frequency_range_json` | The wrapper's `frequency_range` as JSON `[low, high]`, on every variable it applied to: each frequency-resolved variable it cropped, and each variable of `delay`, `group_delay`, and `phase_slope_index`, which consumed it. Not on the Dataset itself. |
| `frequency_decimation`, `frequency_bands_json`, `frequency_reduction` | The wrapper's decimation and band reduction, on the frequency-resolved variables they were applied to. |
| `requested_frequency_lower`, `requested_frequency_upper` | `delay`, `group_delay`, and `phase_slope_index` only: the bounds given as `frequency_range` (or the deprecated `frequencies_of_interest`), in the frequency coordinate's units. Omitted when every bin is used. |
| `selected_frequency_lower`, `selected_frequency_upper` | The same three measures: the lowest and highest bins in that interval, before any `frequency_resolution` subsampling. |
| `frequency_interval_closed` | The same three measures: `"both"` when the bins on the bounds are included (`frequency_range`, or every bin), `"neither"` for the deprecated `frequencies_of_interest`. |

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
`estimate_frequency_resolution`, `suggest_parameters(desired_freq_resolution=...)`,
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
warning. Coefficients without a time dimension that keep a numeric scalar time
coordinate, as after `isel(time=0)`, use it as the time of their single window.
Labels along averaged trial axes are omitted.

The result-schema migration is separate from the supported public property and
helper names:

| Previous metadata | Schema version 1 |
| --- | --- |
| `mt_sampling_frequency`, `mt_n_trials`, `mt_n_signals` | `sampling_frequency`, `n_trials`, `n_signals`. |
| `mt_frequency_resolution` | `spectral_bandwidth`, qualified as `full_dpss_concentration_bandwidth`, for generated DPSS tapers. Custom tapers have no `spectral_bandwidth`; their `mt_frequency_resolution` stays the nominal NW-based value. |
| Other scalar `mt_*` settings | `transform_parameters_json`, with optional settings as JSON `null`. |

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
and 11.88 Hz. `estimate_frequency_resolution` uses the duration it is given.

## Pass a labeled DataArray

`time_series` may be an `xarray.DataArray`. Dimension names, not positions,
define the roles, so a `time` coordinate in seconds supplies the sampling
frequency and the signal labels become the `source`/`target` coordinates:

```python
>>> import xarray as xr
>>> da = xr.DataArray(
...     rng.standard_normal((1000, 4, 3)),
...     dims=("time", "trial", "signal"),
...     coords={"time": np.arange(1000) / 500, "signal": ["CA1", "CA3", "PFC"]},
... )
>>> coherence = multitaper_connectivity(da, method="coherence_magnitude")
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
...     method="coherence_magnitude",
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
>>> multitaper_connectivity(elapsed, method="coherence_magnitude").attrs["sampling_frequency"]
500.0

```

## Where to go next

- Value ranges for every measure: `docs/CONNECTIVITY_METRIC_RANGES.md`.
- The full lower-level API: the `Connectivity` class (each `method` above is a
  method on it).
- End-to-end tutorials: the notebooks under `examples/`.
