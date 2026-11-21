# spectral_connectivity

[![Test, Build, and Publish](https://github.com/Eden-Kramer-Lab/spectral_connectivity/actions/workflows/release.yml/badge.svg)](https://github.com/Eden-Kramer-Lab/spectral_connectivity/actions/workflows/release.yml)
[![DOI](https://zenodo.org/badge/104382538.svg)](https://zenodo.org/badge/latestdoi/104382538)
[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/Eden-Kramer-Lab/spectral_connectivity/master)
[![status](https://joss.theoj.org/papers/27eb33e699c9ea723783c44576d081bb/status.svg)](https://joss.theoj.org/papers/27eb33e699c9ea723783c44576d081bb)
[![PyPI version](https://badge.fury.io/py/spectral_connectivity.svg)](https://badge.fury.io/py/spectral_connectivity)
[![Anaconda-Server Badge](https://anaconda.org/edeno/spectral_connectivity/badges/version.svg)](https://anaconda.org/edeno/spectral_connectivity)
[![Documentation Status](https://readthedocs.org/projects/spectral-connectivity/badge/?version=latest)](https://spectral-connectivity.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/Eden-Kramer-Lab/spectral_connectivity/branch/master/graph/badge.svg)](https://codecov.io/gh/Eden-Kramer-Lab/spectral_connectivity)

[**Installation**](#installation)
| [**Tutorials**](#tutorials)
| [**Usage Example**](#usage-example)
| [**Documentation**](#documentation)
| [**GPU Acceleration**](#gpu-acceleration)
| [**Developer Installation**](#developer-installation)
| [**Contributing**](#contributing)
| [**License**](#license)
| [**Citation**](#citation)

> [!WARNING]
> **Upgrading from 2.x? Two changes alter results without raising an error.**
>
> + **Directed measures changed direction.** `result[..., i, j]` from the
>   `Connectivity` methods of the spectral Granger family, directed transfer
>   function, directed coherence, (generalized) partial directed coherence, and
>   direct directed transfer function is now `i -> j` (2.x returned `j -> i`).
>   For pairwise and subset spectral Granger prediction,
>   `multitaper_connectivity(...).sel(source="a", target="b")` is now `a -> b`
>   (2.x returned `b -> a`). Swap the indices in code written for 2.x and
>   recompute stored results; `phase_slope_index`, `group_delay`, and `delay`
>   were already source first.
> + **Power is one-sided.** `Connectivity.power()` and the power that
>   `multitaper_connectivity` and `connectivity_to_xarray` return are twice the
>   2.x values between 0 Hz and Nyquist; `power(one_sided=False)` returns the
>   2.x values.
>
> Until 3.2 some of these calls emit a `DirectedOrientationWarning` or a
> `PowerSidednessWarning` as a reminder. The [migration guide](docs/migration.md)
> lists which ones, how to silence them, and every other change.

## What is spectral_connectivity?

`spectral_connectivity` is a Python software package that computes multitaper spectral estimates and frequency-domain brain connectivity measures such as coherence, spectral granger causality, and the phase lag index using the multitaper Fourier transform. Although there are other Python packages that do this (see [nitime](https://github.com/nipy/nitime) and [MNE-Python](https://github.com/mne-tools/mne-python)), `spectral_connectivity` has several differences:

+ it is designed to handle multiple time series at once
+ it caches frequently computed quantities such as the cross-spectral matrix and minimum-phase-decomposition, so that connectivity measures that use the same processing steps can be more quickly computed. Call `Connectivity.clear_cache()` to release these intermediates when memory is tight.
+ it decouples the time-frequency transform and the connectivity measures so that if you already have a preferred way of computing Fourier coefficients (i.e. from a wavelet transform), you can use that instead.
+ it implements the non-parametric version of the spectral granger causality in Python.
+ it implements the canonical coherence, which can
efficiently summarize brain-area level coherences from multielectrode recordings.
+ easier user interface for the multitaper fourier transform
+ core transforms and connectivity calculations support GPU acceleration when
  `cupy` is installed and `SPECTRAL_CONNECTIVITY_ENABLE_GPU=true` is set before
  importing the package. Public results are returned as NumPy arrays.

## Installation

Requires Python >= 3.10.

```bash
pip install spectral_connectivity
```

or

```bash
conda install -c edeno spectral_connectivity
```

GPU acceleration is optional; see [GPU Acceleration](#gpu-acceleration) to set
it up. To contribute, see [Developer Installation](#developer-installation).

## Tutorials

+ [Getting started: from time series to connectivity](examples/Intro_tutorial.ipynb):
  start here; coherence from a time series, and the settings that matter.
+ [Connectivity measures on simulated signals](examples/Tutorial_On_Simulated_Examples.ipynb):
  what each measure shows on signals with known, planted coupling.
+ [Directed connectivity on published autoregressive models](examples/Tutorial_Using_Paper_Examples.ipynb):
  Granger and transfer-function measures on models from the literature.

## Usage Example

The quickest way to get connectivity measures is the high-level
`multitaper_connectivity` function, which runs the multitaper transform and
returns labeled [xarray](https://docs.xarray.dev/) objects. This example runs
as is: it simulates four signals, three of which share a 10 Hz rhythm, and
plots the coherence of one pair.

```python
import matplotlib.pyplot as plt

from spectral_connectivity import multitaper_connectivity
from spectral_connectivity.simulate import simulate_shared_oscillation

sampling_frequency = 500  # samples per second, in Hz

# Your data goes in an array shaped (n_time_samples, n_trials, n_signals).
# Here: 20 trials of 10 s (5000 samples) from 4 signals. Signals a, b, and c
# share a 10 Hz oscillation; d is noise only.
time_series = simulate_shared_oscillation(
    frequency=10,  # Hz
    sampling_frequency=sampling_frequency,
    n_time_samples=5000,
    n_trials=20,
    amplitudes=[1.0, 1.0, 1.0, 0.0],
    noise_levels=1.0,
    random_state=0,
)
print(time_series.shape)  # (5000, 20, 4)

coherence = multitaper_connectivity(
    time_series,
    sampling_frequency=sampling_frequency,
    method="magnitude_squared_coherence",
    signal_names=["a", "b", "c", "d"],
    time_halfbandwidth_product=3,
)
print(coherence.dims)  # ('time', 'frequency', 'source', 'target')

# Select one pair. With no time_window_duration the whole trial is one time
# window, so take that window and plot coherence against frequency (Hz).
coherence.sel(source="a", target="b").isel(time=0).plot()
plt.xlim(0, 50)
plt.show()

# Ask for several measures at once (returns an xarray.Dataset)
measures = multitaper_connectivity(
    time_series,
    sampling_frequency=sampling_frequency,
    method=["magnitude_squared_coherence", "imaginary_coherence", "phase_locking_value"],
)
```

A few terms used throughout:

+ **Taper**: a window function that each stretch of data is multiplied by
  before its Fourier transform. The multitaper method uses several orthogonal
  tapers (Slepian sequences, or DPSS) and averages over them, which lowers the
  variance of the estimate.
+ **Time-halfbandwidth product** (`time_halfbandwidth_product`, NW): sets the
  frequency resolution, `2 * NW / time_window_duration` Hz (the full DPSS
  concentration bandwidth: the spectral smoothing, independent of FFT bin
  spacing and zero-padding), and the number of tapers, `2 * NW - 1` rounded
  down (5 for NW = 3). Larger NW: smoother, less noisy spectra but coarser
  frequency resolution.
+ **Observations**: the trials x tapers that each estimate averages over.
  Normalized measures such as coherence need several of them.

The common multitaper settings, like `time_halfbandwidth_product` here, are
named arguments: `time_window_duration` (seconds; default the whole trial),
`time_window_step` (seconds), `time_halfbandwidth_product`, `n_tapers`, and
`detrend_type`. Any other
`Multitaper` setting goes in `transform_kwargs={...}` (e.g.
`transform_kwargs={"n_fft_samples": 1024}`), options for the measure itself go
in `connectivity_kwargs={...}`, and group measures take `group_labels` (see
[setting transform and measure options](docs/cookbook.md#set-transform-and-measure-options)
and the
[group-measures recipe](docs/cookbook.md#group-measures-canonical-coherence-between-areas)).

The `3` above is a common default. To pick values from your recording's length
and the frequency resolution you need, use `suggest_parameters`, giving it the
duration of one trial. The window must fit within that duration: a resolution
that needs a longer window raises a `ValueError`, and a window that would leave
fewer than three per trial is shortened, keeping the resolution where possible
at the cost of fewer tapers (it warns when that leaves fewer than two):

```python
from spectral_connectivity import suggest_parameters

params = suggest_parameters(
    sampling_frequency=sampling_frequency,
    signal_duration=time_series.shape[0] / sampling_frequency,  # 10 s per trial
    frequency_resolution=2.0,  # Hz
)
# -> time_halfbandwidth_product=3.0, time_window_duration=3.0, n_tapers=5, ...
coherence = multitaper_connectivity(
    time_series,
    sampling_frequency=sampling_frequency,
    method="magnitude_squared_coherence",
    time_halfbandwidth_product=params["time_halfbandwidth_product"],
    time_window_duration=params["time_window_duration"],
)
```

Not sure which `method` names are valid? `list_measures()` enumerates every
supported measure with its output category and a one-line description, and an
unknown `method` name raises an error that suggests the closest matches. For a
measure `m`, `m.description` is that one-liner and `m.interpretation` says how
to read its values:

```python
from spectral_connectivity import list_measures

# Every valid method name, e.g. filter to the directed measures
directed = [m.name for m in list_measures(directed=True)]
```

### Choosing a measure

| Your question | Start with (**try first**) | Watch out for |
| --- | --- | --- |
| Are two signals coupled? | **`magnitude_squared_coherence`**, `phase_locking_value`, `pairwise_phase_consistency` | Few trials x tapers bias coherence and phase locking upward; one gives 1 for any data. |
| Could it be volume conduction (zero-lag mixing)? | `imaginary_coherence`, **`weighted_phase_lag_index`**, `debiased_squared_weighted_phase_lag_index` | They also ignore genuine zero-lag and anti-phase coupling. |
| Which signal leads? | **`phase_slope_index`**, `pairwise_spectral_granger_prediction`, `conditional_spectral_granger_prediction` | Granger measures need long windows and a two-sided transform. |
| How are two groups of signals coupled? | **`canonical_coherence`**, `blockwise_spectral_granger_prediction` | Pass `group_labels`, one label per signal. |

[Choosing a measure](docs/choosing_a_measure.md) has the full table and the
pitfalls common to every measure.

See the [**Cookbook**](docs/cookbook.md) for short, copy-pasteable recipes
covering functional and directed connectivity, reading the labeled output,
frequency bands, and using your own Fourier coefficients.

Crop, decimate, or aggregate the frequency coordinate without losing labels:

```python
band_connectivity = multitaper_connectivity(
    time_series,
    sampling_frequency=sampling_frequency,
    method="magnitude_squared_coherence",
    frequency_range=(4, 100),
    frequency_bands={"theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)},
)
```

Named-band means are arithmetic means of the computed score; complex measures
use a complex-vector mean and `coherence_phase` uses a circular mean. Band
integration is restricted to power and cross-spectral density, where it has the
physical interpretation of band power/covariance. The same reduction is
available for an already-computed result via
`frequency_band_reduce(result, frequency_bands, frequency_reduction="mean")`,
which takes the wrappers' keywords.

`time_series` may also be an `xarray.DataArray`. Dimension names, not
positions, define the roles: common names for time, trial and signal dimensions
are recognized, `sampling_frequency` is inferred from a numeric `time`
coordinate in seconds, and the signal index becomes the `source`/`target`
labels. For other dimension names pass `time_dim`, `trial_dim` and
`signal_dim`; the cookbook's
[DataArray recipe](docs/cookbook.md#pass-a-labeled-dataarray) has the full
contract (label types, datetime coordinates, dask).

For directed measures, `result.sel(source="a", target="b")` is the influence
from `a` to `b`, and the lower-level `Connectivity` arrays use the same order:
`result[..., i, j]` is `i -> j`. For lead/lag measures a positive value means
the source leads. The directed-transfer-function family is available by name
as an opt-in method. `list_measures()` reports each measure's value range,
units, and interpretation.

### Lower-level API

For finer control (or if you already have Fourier coefficients from another
method), use the `Multitaper` and `Connectivity` classes directly:

```python
from spectral_connectivity import Connectivity, Multitaper
from spectral_connectivity.simulate import simulate_shared_oscillation

sampling_frequency = 500  # Hz
# (n_time_samples, n_trials, n_signals): 20 trials of 10 s from 4 signals
time_series = simulate_shared_oscillation(
    frequency=10, sampling_frequency=sampling_frequency, n_time_samples=5000,
    n_trials=20, amplitudes=[1.0, 1.0, 1.0, 0.0], noise_levels=1.0, random_state=0,
)
brain_area_labels = ["CA1", "CA1", "PFC", "PFC"]  # one group label per signal

# Compute multitaper spectral estimate in 1 s windows every 0.5 s
m = Multitaper(
    time_series=time_series,
    sampling_frequency=sampling_frequency,
    time_halfbandwidth_product=3,
    time_window_duration=1.0,  # seconds
    time_window_step=0.5,  # seconds
    start_time=0.0,  # time of the first sample, in seconds
    # fft_workers=-1,  # optional: parallelize the CPU FFT across all cores
)

# Sets up computing connectivity measures/power from multitaper spectral estimate
# (`from_multitaper`, its 2.x name, is deprecated and removed in 4.0.) Its
# expectation_type says which axes the cross-spectrum is averaged over: the
# default "trials_tapers" keeps one estimate per time window.
c = Connectivity.from_transform(m)

# Here are a couple of examples. Pairwise measures return NumPy arrays shaped
# (n_time_windows, n_frequencies, n_signals, n_signals), (19, 251, 4, 4) here;
# power drops the last axis. c.time (window centers, s) and c.frequencies (Hz)
# label the first two axes.
power = c.power(one_sided=True)  # one-sided spectral power density
coherence = c.magnitude_squared_coherence()
weighted_phase_lag_index = c.weighted_phase_lag_index()
# Group measures return (values, group labels)
canonical_coherence, area_labels = c.canonical_coherence(brain_area_labels)
# Exact Vidaurre CaCoh is complex and component-resolved. Rich MIC uses the
# same result schema; both include filters, patterns, and group membership.
cacoh = c.canonical_coherency(brain_area_labels, n_components=2)
mic = c.maximized_imaginary_coherency_components(brain_area_labels, n_components=2)
```

The xarray interfaces preserve these non-pairwise contracts as labeled axes:
group measures use `source_group`/`target_group`; CaCoh and rich MIC use
`connection`, `component`, `side` (`"source"` or `"target"`, the connection's
`connection_source_group` or `connection_target_group`), and `signal` plus
group-membership metadata;
delay exposes a `candidate` axis; global coherence returns score and vector
variables; and group delay returns delay, slope, and correlation variables.
`frequency_range=(low, high)` includes both endpoints. Phase slope, group delay,
and delay consume only those bins; phase slope and group delay are already
frequency-reduced, so `frequency_bands` and `frequency_decimation` leave them
as computed over `frequency_range`.

The same connectivity core accepts Hann short-time Fourier, Welch, and Morlet
wavelet transforms. Morlet coefficients are one-sided, so they support
functional measures but deliberately reject the directed measures (the
spectral Granger and transfer-function families), which factor the spectrum
into a transfer function and noise covariance by Wilson's algorithm and need
both positive and negative frequencies:

```python
from spectral_connectivity import (
    Connectivity,
    MorletWavelet,
    ShortTimeFourierTransform,
    Welch,
)

stft = ShortTimeFourierTransform(time_series, sampling_frequency, time_window_duration=1)
welch = Welch(time_series, sampling_frequency, segment_duration=1)
# For a single continuous trial, collect a local time/frequency neighborhood so
# normalized measures are estimated over multiple observations. Invalid wavelet
# edges can be retained, marked NaN, or trimmed.
wavelet = MorletWavelet(
    time_series,
    sampling_frequency,
    frequencies=[4, 8, 16, 32],
    smoothing_time=0.5,
    smoothing_frequency=3,
    smoothing_kernel="hann",
    padding_mode="reflect",
    edge_mode="nan",
)

stft_coherence = Connectivity.from_transform(stft).magnitude_squared_coherence()
```

Which transform? Use `Multitaper` for stationary spectra with controlled
variance, `ShortTimeFourierTransform`/`Welch` for a simple single-window or
segment-averaged estimate, and `MorletWavelet` for time-resolved analysis at
specific frequencies. `Welch`'s spectral bandwidth
(`equivalent_noise_bandwidth`) is `1.5 / segment_duration` Hz for its Hann
window, so set `segment_duration` explicitly for electrophysiology data.
`MorletWavelet.valid_time_frequency` identifies bins with full in-record support;
the xarray wrapper carries it as a two-dimensional coordinate.

`Connectivity.from_transform` also accepts your own transform: any object
satisfying the `SpectralTransform` protocol (an `fft()` method plus
`frequencies` and `time`). `help(SpectralTransform)` gives its optional
attributes, the `fft()` ownership rule, and the scaling that makes `power()` a
density; the cookbook has a
[worked example](docs/cookbook.md#plug-in-your-own-transform).

For DPSS transforms, `taper_weighting="uniform"` preserves the historical
behavior; `"eigen"` weights by concentration ratio and `"adaptive"` applies
Thomson's frequency- and signal-specific iterative weights (useful when high
spectral dynamic range or line noise makes some tapers systematically leakier).

Already have full two-sided FFT coefficients? `fourier_connectivity` accepts
NumPy arrays or labeled DataArrays in the core
`(time, trial, taper, frequency, signal)` layout (with documented shorter
forms), preserving frequency, time, and signal coordinates, and returns the
same labeled results as `multitaper_connectivity`.

The labeled results share one schema, `output_schema_version=1`: common
transform and observation attributes, the estimator's qualified bandwidth, and
one `transform_parameters_json` record of the settings. See the
[result-schema reference and migration examples](docs/cookbook.md#result-schema-and-transform-bandwidth).

For uncertainty, `Connectivity.jackknife(method)` recomputes a real-valued
measure leaving out each trial/taper observation in turn and returns a
bias-corrected estimate, standard error, and confidence interval, applying an
automatic variance-stabilizing transformation (log for power, `atanh(sqrt(.))`
for magnitude-squared coherence, and circular for phase).

## Documentation

<!-- docs/index.md skips from here: a link to the docs site itself. -->
See the documentation on [ReadTheDocs](https://spectral-connectivity.readthedocs.io/en/latest/).
<!-- docs/index.md resumes here. -->

+ [Cookbook](docs/cookbook.md): short recipes for common tasks.
+ [Choosing a measure](docs/choosing_a_measure.md): which measure answers
  which question, and the pitfalls common to every measure.
+ [Connectivity Metric Ranges](docs/CONNECTIVITY_METRIC_RANGES.md): the range,
  units, diagonal, and interpretation of every measure.
+ [Migrating from 2.x to 3.0](docs/migration.md).

## Implemented Measures

The complete list, with each measure's `method` name, value range, units,
diagonal, and interpretation, is generated from the code into
[Connectivity Metric Ranges](docs/CONNECTIVITY_METRIC_RANGES.md);
`list_measures()` returns the same records in Python. By family:

+ **Spectra**: power and the cross-spectral density.
+ **Coherence**: coherency, its squared magnitude, phase, and imaginary part,
  and partial coherence.
+ **Phase synchrony**: the phase-locking value and pairwise phase consistency,
  and the phase lag index with its weighted and debiased forms.
+ **Between groups of signals**: canonical coherence and coherency, maximized
  imaginary coherency, the multivariate interaction measure, and global
  coherence.
+ **Directed influence**: spectral Granger prediction (pairwise, subset,
  conditional, blockwise, time-reversed) and the transfer-function family
  (directed transfer function, directed coherence, partial directed coherence
  and its generalized form, direct directed transfer function).
+ **Lead and lag**: the phase slope index, group delay, delay, and the directed
  phase lag index.

## Package Dependencies

`spectral_connectivity` requires:

+ python
+ numpy
+ matplotlib
+ scipy
+ xarray

See [pyproject.toml](https://github.com/Eden-Kramer-Lab/spectral_connectivity/blob/master/pyproject.toml) for the authoritative list of dependencies and their minimum versions.

## GPU Acceleration

GPU acceleration is optional. The core transforms and connectivity
calculations can run on an NVIDIA GPU through [CuPy](https://cupy.dev/), which
helps with large workloads (many signals, trials, or windows, or several
measures from the same data). For small datasets (fewer than 10 signals and
1000 time points) the CPU may be faster, because of the cost of moving data to
the GPU. Results are NumPy arrays either way.

1. Install CuPy. With pip on CUDA 12, the package's `gpu` extra installs
   `cupy-cuda12x>=13.0`:

   ```bash
   pip install "spectral-connectivity[gpu]"
   ```

   With conda, `conda install -c conda-forge cupy` picks the CUDA version for
   you. For another CUDA version (`nvidia-smi` shows yours), install the
   matching CuPy 13+ package instead of the extra, e.g.
   `pip install "cupy-cuda11x>=13.0"`; see the
   [CuPy installation guide](https://docs.cupy.dev/en/stable/install.html).

2. Set `SPECTRAL_CONNECTIVITY_ENABLE_GPU=true` **before** importing
   `spectral_connectivity`. The backend is chosen once, at import, so setting
   the variable later has no effect. In a shell, run
   `export SPECTRAL_CONNECTIVITY_ENABLE_GPU=true` before starting Python. In a
   Jupyter notebook, run `%env SPECTRAL_CONNECTIVITY_ENABLE_GPU=true` in the
   first cell (restart the kernel if the package was already imported). In a
   script, set it above the first import of the package:

   ```python
   import os

   os.environ["SPECTRAL_CONNECTIVITY_ENABLE_GPU"] = "true"

   import spectral_connectivity as sc
   ```

3. Check which backend is active:

   ```python
   backend = sc.get_compute_backend()
   print(backend["message"])
   ```

   On a GPU the message is `Using GPU backend with CuPy on <your GPU name>.`
   On the CPU it says the NumPy backend is in use and how to enable the GPU
   (install CuPy, set the variable before importing). The
   dictionary also holds `backend` (`"cpu"` or `"gpu"`), `gpu_enabled`
   (whether the variable requested the GPU), `gpu_available` (whether CuPy is
   installed), and `device_name`.

Troubleshooting:

+ **Importing the package raises `RuntimeError: GPU support was explicitly
  requested ... but CuPy is not installed`.** Install CuPy as in step 1,
  matching your CUDA version; `python -c "import cupy; print(cupy.__version__)"`
  checks the install and `nvidia-smi` the CUDA driver.
+ **The GPU is not used although CuPy is installed and the variable is set.**
  The variable must be set before `import spectral_connectivity`; set later, it
  has no effect (and the message may still suggest installing CuPy). Move it
  above the first import, or restart the notebook kernel.
+ **Out of GPU memory.** Process fewer trials or time windows per call, or run
  on the CPU: unset the variable and restart Python.

## Developer Installation

If you want to make contributions to this library, please use this installation.

1. Install miniconda (or anaconda) if it isn't already installed. Type into bash (or install from the anaconda website):

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
hash -r
```

1. Clone the repository to your local machine (`.../spectral_connectivity`) and create the development environment. Type into bash:

```bash
# uv: creates .venv from the committed uv.lock with the dev tools installed
uv sync
uv run pytest                     # any command runs in that environment
uvx pre-commit install            # optional: run ruff, codespell and mypy on each commit

# Or conda
conda env create -f environment.yml
conda activate spectral_connectivity
pip install -e .[dev]
```

`uv.lock` pins the development environment; `uv lock` refreshes it after a
dependency change, and CI checks that it is current.

## Contributing

We welcome contributions to `spectral_connectivity`! Please see our [Contributing Guidelines](https://github.com/Eden-Kramer-Lab/spectral_connectivity/blob/master/CONTRIBUTING.md) for details on:

+ How to report bugs and request features
+ Development workflow and coding standards
+ Testing requirements
+ Code review process

For questions or discussions, please open an issue on GitHub.

## License

This project is licensed under the **GPL-3.0** License - see the [LICENSE](https://github.com/Eden-Kramer-Lab/spectral_connectivity/blob/master/LICENSE) file for details.

## Citation

For citation, please use the following:

> Denovellis, E.L., Myroshnychenko, M., Sarmashghi, M., and Stephen, E.P. (2022). Spectral Connectivity: a python package for computing multitaper spectral estimates and frequency-domain brain connectivity measures on the CPU and GPU. JOSS 7, 4840. [10.21105/joss.04840](https://doi.org/10.21105/joss.04840).

### Recent publications and pre-prints that used this software

+ Detection of Directed Connectivities in Dynamic Systems for Different Excitation Signals using Spectral Granger Causality <https://doi.org/10.1007/978-3-662-58485-9_11>
+ Network Path Convergence Shapes Low-Level Processing in the Visual Cortex <https://doi.org/10.3389/fnsys.2021.645709>
+ Subthalamic–Cortical Network Reorganization during Parkinson's Tremor
<https://doi.org/10.1523/JNEUROSCI.0854-21.2021>
+ Unifying Pairwise Interactions in Complex Dynamics <https://doi.org/10.48550/arXiv.2201.11941>
+ Phencyclidine-induced psychosis causes hypersynchronization and
disruption of connectivity within prefrontal-hippocampal circuits
that is rescued by antipsychotic drugs <https://doi.org/10.1101/2021.02.03.429582>
+ The cerebellum regulates fear extinction through thalamo-prefrontal cortex interactions in male mice <https://doi.org/10.1038/s41467-023-36943-w>

<!-- docs/index.md includes this file up to this line. -->

## Releases

This package uses dynamic versioning with [Hatch](https://hatch.pypa.io/) based on git tags. The version is automatically determined from the repository state:

- **Tagged releases**: `1.2.0`
- **Development versions**: `1.2.0.dev5+g1a2b3c4` (5 commits since tag + git hash)

### Making a Release

Releases are published **automatically by CI**: pushing a `v*` tag runs the
*Test, Build, and Publish* workflow, which tests, builds, attests, and publishes
to PyPI via trusted publishing (after approval in the protected `pypi`
environment). Do **not** run `twine upload` by hand — that bypasses the tests,
attestations, and approval gate. See [CONTRIBUTING.md](https://github.com/Eden-Kramer-Lab/spectral_connectivity/blob/master/CONTRIBUTING.md) for the
full process and the required one-time setup.

```bash
# Update CHANGELOG.md, then tag and push -- CI does the rest.
git tag -a v1.2.0 -m "v1.2.0"
git push origin v1.2.0
```

Conda packages are published separately and manually:

```bash
conda build conda-recipe/ --output-folder ./conda-builds
anaconda upload ./conda-builds/noarch/spectral_connectivity-*.tar.bz2
```

The version number is automatically extracted from the git tag (without the 'v' prefix).

### Conda Package

This package is also available on conda via the `edeno` channel:

```bash
conda install -c edeno spectral_connectivity
```

**Not yet on conda-forge?** Help us get there! If you'd like this package on conda-forge for easier installation, please:
- 👍 React to [this issue](https://github.com/Eden-Kramer-Lab/spectral_connectivity/issues) requesting conda-forge support
- Or volunteer to help maintain the conda-forge feedstock
