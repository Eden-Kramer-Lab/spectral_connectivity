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
`frequencies` and `time`. No subclassing is needed. Optional attributes such as
`is_one_sided` are honored when present and default to a two-sided,
unweighted, independent spectrum otherwise; `help(SpectralTransform)` lists
them. `fft()` must return a new array on each call, and nothing may modify it
afterwards, because `Connectivity` keeps it without copying. The example below
is not scaled to a power spectral density, so its `power()` is in arbitrary
units; normalized measures such as coherence are unaffected.
`help(SpectralTransform)` gives the scaling that makes `power()` a density.

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

## Where to go next

- Value ranges for every measure: `docs/CONNECTIVITY_METRIC_RANGES.md`.
- The full lower-level API: the `Connectivity` class (each `method` above is a
  method on it).
- End-to-end tutorials: the notebooks under `examples/`.
