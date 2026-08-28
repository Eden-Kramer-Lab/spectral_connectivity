# spectral_connectivity

[![Test, Build, and Publish](https://github.com/Eden-Kramer-Lab/spectral_connectivity/actions/workflows/release.yml/badge.svg)](https://github.com/Eden-Kramer-Lab/spectral_connectivity/actions/workflows/release.yml)
[![DOI](https://zenodo.org/badge/104382538.svg)](https://zenodo.org/badge/latestdoi/104382538)
[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/Eden-Kramer-Lab/spectral_connectivity/master)
[![status](https://joss.theoj.org/papers/27eb33e699c9ea723783c44576d081bb/status.svg)](https://joss.theoj.org/papers/27eb33e699c9ea723783c44576d081bb)
[![PyPI version](https://badge.fury.io/py/spectral_connectivity.svg)](https://badge.fury.io/py/spectral_connectivity)
[![Anaconda-Server Badge](https://anaconda.org/edeno/spectral_connectivity/badges/version.svg)](https://anaconda.org/edeno/spectral_connectivity)
[![Documentation Status](https://readthedocs.org/projects/spectral-connectivity/badge/?version=latest)](https://spectral-connectivity.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/Eden-Kramer-Lab/spectral_connectivity/branch/master/graph/badge.svg)](https://codecov.io/gh/Eden-Kramer-Lab/spectral_connectivity)

## What is spectral_connectivity?

`spectral_connectivity` is a Python software package that computes multitaper spectral estimates and frequency-domain brain connectivity measures such as coherence, spectral granger causality, and the phase lag index using the multitaper Fourier transform. Although there are other Python packages that do this (see [nitime](https://github.com/nipy/nitime) and [MNE-Python](https://github.com/mne-tools/mne-python)), `spectral_connectivity` has several differences:

+ it is designed to handle multiple time series at once
+ it caches frequently computed quantities such as the cross-spectral matrix and minimum-phase-decomposition, so that connectivity measures that use the same processing steps can be more quickly computed.
+ it decouples the time-frequency transform and the connectivity measures so that if you already have a preferred way of computing Fourier coefficients (i.e. from a wavelet transform), you can use that instead.
+ it implements the non-parametric version of the spectral granger causality in Python.
+ it implements the canonical coherence, which can
efficiently summarize brain-area level coherences from multielectrode recordings.
+ easier user interface for the multitaper fourier transform
+ core transforms and connectivity calculations support GPU acceleration when
  `cupy` is installed and `SPECTRAL_CONNECTIVITY_ENABLE_GPU=true` is set before
  importing the package. Public results are returned as NumPy arrays.

## Tutorials

See the following notebooks for more information on how to use the package:

+ [Tutorial](examples/Intro_tutorial.ipynb)
+ [Usage Examples](examples/Tutorial_On_Simulated_Examples.ipynb)
+ [More Usage Examples](examples/Tutorial_Using_Paper_Examples.ipynb)

## Usage Example

The high-level `multitaper_connectivity` function runs the multitaper transform
and returns a labeled xarray object:

```python
from spectral_connectivity import multitaper_connectivity

coherence = multitaper_connectivity(
    time_series,
    sampling_frequency=sampling_frequency,
    method="coherence_magnitude",
    time_halfbandwidth_product=3,
)
```

The wrapper also provides coordinate-aware frequency cropping, decimation, and
named-band reduction through `frequency_range`, `frequency_decimation`, and
`frequency_bands`. Phase bands use a circular mean, complex results use a
complex-vector mean, and integration is limited to power/cross-spectral density.

Use `fourier_connectivity` when coefficients were computed elsewhere. It accepts
NumPy arrays or labeled DataArrays, preserves their time/frequency/signal
coordinates, and uses the same result and provenance contract.

`time_series` may also be an `xarray.DataArray`. **For DataArray inputs, dimension
names define axis roles; positions do not.** Common dimension names are
inferred and transposed automatically; for domain-specific names, pass
`time_dim`, `trial_dim`, and `signal_dim` explicitly. Ambiguous dimensions raise
instead of falling back to axis position; when a single unrecognized dimension
is left for the one remaining role, it is assigned by elimination and a warning
names the assumed mapping. Numeric `time`
coordinates are interpreted as elapsed seconds and numeric `sample` coordinates
as sample numbers, and are used to label output window centers. When
`sampling_frequency` is given it is checked against the time index; when it is
omitted, a numeric elapsed-seconds `time` coordinate infers it (a `sample`
index cannot, having no time scale). Inference also requires enough coordinate
precision to resolve the rate reliably; pass `sampling_frequency` explicitly for
low-precision or large-offset time coordinates. A 1-D index on the signal
dimension is preserved—including its label type—as the result's `source` and
`target` coordinates unless `signal_names` is supplied. Signal labels must be
unique, non-missing, NetCDF-compatible scalar strings, real numbers, datetimes,
or timedeltas; integer labels must fit the signed 32-bit range for portable
NetCDF3 serialization.

Datetime, timedelta, and object-valued **time coordinates** are not yet
supported. Convert them to numeric elapsed seconds before calling
`multitaper_connectivity`, for example:

```python
da = da.assign_coords(time=(da.time - da.time[0]) / np.timedelta64(1, "s"))
```

datetime and timedelta **signal labels** remain valid.

A dask-backed DataArray is rejected; materialize it first with
`DataArray.compute()` (or `.load()`) and pass the result.

For directed measures, `result.sel(source="a", target="b")` means influence
from `a` to `b`. The directed-transfer-function family is available by name as
an opt-in method. The lower-level `Connectivity` methods retain their historical
array convention: `result[..., i, j]` represents `j -> i`.

For finer control, use the `Multitaper` and `Connectivity` classes directly:

```python
from spectral_connectivity import Multitaper, Connectivity

# Compute multitaper spectral estimate
m = Multitaper(
    time_series=signals,
    sampling_frequency=sampling_frequency,
    time_halfbandwidth_product=time_halfbandwidth_product,
    time_window_duration=0.060,
    time_window_step=0.060,
    start_time=time[0],
)

# Sets up computing connectivity measures/power from multitaper spectral estimate
# (`from_multitaper` is a backward-compatible alias for the transform-neutral
# `from_transform`.)
c = Connectivity.from_transform(m)

# Here are a couple of examples
power = c.power()  # spectral power
coherence = c.coherence_magnitude()
weighted_phase_lag_index = c.weighted_phase_lag_index()
canonical_coherence = c.canonical_coherence(brain_area_labels)
cacoh = c.canonical_coherency(brain_area_labels, n_components=2)
mic = c.maximized_imaginary_coherency_components(brain_area_labels, n_components=2)
```

The xarray wrappers retain nonstandard scientific shapes instead of flattening
them: group-pair matrices have `source_group`/`target_group`; rich CaCoh/MIC
results include `connection`, `component`, `side`, `signal`, filters, patterns,
and group membership; delay has a `candidate` axis; and global coherence and
group delay return multi-variable Datasets. Phase slope and group delay have no
frequency dimension because their band has already been reduced.

`ShortTimeFourierTransform`, `Welch`, and `MorletWavelet` provide alternative
spectral transforms with the same coefficient interface. Morlet output is
positive-frequency-only and is therefore limited to functional measures;
directed Wilson factorization requires a full two-sided FFT. For single-trial
Morlet data, `smoothing_time` and `smoothing_frequency` collect a local
time/frequency neighborhood; `smoothing_kernel` selects boxcar or Hann weights.
`padding_mode` controls convolution boundaries, while `edge_mode` retains,
masks, or trims estimates without full wavelet support. The strict
`valid_time_frequency` mask is also exposed by the xarray wrapper. `Welch`
resolves `1 / segment_duration` Hz, so set `segment_duration` explicitly for
electrophysiology data. Multitaper DPSS coefficients support uniform
(historical), eigenvalue, or Thomson adaptive taper weighting through
`taper_weighting`.

For uncertainty estimates, `Connectivity.jackknife(method)` recomputes a
real-valued measure while leaving out each trial/taper observation and returns a
bias-corrected estimate, standard error, and confidence interval, applying an
automatic variance-stabilizing transformation (log for power, `atanh(sqrt(.))`
for magnitude-squared coherence, and circular for phase).

## Citation

For citation, please use the following:

> Denovellis, E.L., Myroshnychenko, M., Sarmashghi, M., and Stephen, E.P. (2022). Spectral Connectivity: a python package for computing multitaper spectral estimates and frequency-domain brain connectivity measures on the CPU and GPU. JOSS 7, 4840. [10.21105/joss.04840](https://doi.org/10.21105/joss.04840).

## Implemented Measures

Functional

1. coherency
2. cross_spectral_density
3. coherence_magnitude and coherence_phase
4. imaginary_coherence and signed imaginary_coherency
5. partial_coherence
6. canonical_coherence and exact complex canonical_coherency (CaCoh)
7. maximized_imaginary_coherency (score-only or component-resolved) and multivariate_interaction_measure
8. phase_locking_value and corrected_imaginary_phase_locking_value
9. phase_lag_index, directed_phase_lag_index, and weighted_phase_lag_index
10. debiased_squared_phase_lag_index
11. debiased_squared_weighted_phase_lag_index
12. pairwise_phase_consistency
13. global_coherence

Directed

1. directed_transfer_function
2. directed_coherence
3. partial_directed_coherence
4. generalized_partial_directed_coherence
5. direct_directed_transfer_function
6. group_delay
7. pairwise_spectral_granger_prediction
8. conditional_spectral_granger_prediction
9. blockwise_spectral_granger_prediction
10. time_reversed_spectral_granger_prediction

## Package Dependencies

`spectral_connectivity` requires:

+ python
+ numpy
+ matplotlib
+ scipy
+ xarray

See the repository's `pyproject.toml` for the authoritative dependency list.

## Installation

```bash
pip install spectral_connectivity
```

or

```bash
conda install -c edeno spectral_connectivity
```

## Developer Installation

If you want to make contributions to this library, please use this installation.

1. Install miniconda (or anaconda) if it isn't already installed. Type into bash (or install from the anaconda website):

```bash
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
hash -r
```

2. Clone the repository to your local machine (`.../spectral_connectivity`) and install the anaconda environment for the repository. Type into bash:

```bash
conda env create -f environment.yml
conda activate spectral_connectivity
pip install -e .
```

## Recent publications and pre-prints that used this software

+ Detection of Directed Connectivities in Dynamic Systems for Different Excitation Signals using Spectral Granger Causality <https://doi.org/10.1007/978-3-662-58485-9_11>
+ Network Path Convergence Shapes Low-Level Processing in the Visual Cortex <https://doi.org/10.3389/fnsys.2021.645709>
+ Subthalamic–Cortical Network Reorganization during Parkinson's Tremor
<https://doi.org/10.1523/JNEUROSCI.0854-21.2021>
+ Unifying Pairwise Interactions in Complex Dynamics <https://doi.org/10.48550/arXiv.2201.11941>
+ Phencyclidine-induced psychosis causes hypersynchronization and
disruption of connectivity within prefrontal-hippocampal circuits
that is rescued by antipsychotic drugs <https://doi.org/10.1101/2021.02.03.429582>
+ The cerebellum regulates fear extinction through thalamo-prefrontal cortex interactions in male mice <https://doi.org/10.1038/s41467-023-36943-w>

```{toctree}
:caption: Guides
:hidden:
:maxdepth: 2

cookbook
CONNECTIVITY_METRIC_RANGES
STYLE
NOTEBOOK_SNAPSHOT_TESTS
```

```{toctree}
:caption: Tutorials
:hidden:
:maxdepth: 2

examples/Intro_tutorial
examples/Tutorial_On_Simulated_Examples
examples/Tutorial_Using_Paper_Examples

```

```{toctree}
:caption: Reference
:hidden:
:titlesonly:
:maxdepth: 1

api
contributing
```
