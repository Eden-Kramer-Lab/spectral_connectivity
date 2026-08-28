# spectral_connectivity

[![Test, Build, and Publish](https://github.com/Eden-Kramer-Lab/spectral_connectivity/actions/workflows/release.yml/badge.svg)](https://github.com/Eden-Kramer-Lab/spectral_connectivity/actions/workflows/release.yml)
[![DOI](https://zenodo.org/badge/104382538.svg)](https://zenodo.org/badge/latestdoi/104382538)
[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/Eden-Kramer-Lab/spectral_connectivity/master)
[![status](https://joss.theoj.org/papers/27eb33e699c9ea723783c44576d081bb/status.svg)](https://joss.theoj.org/papers/27eb33e699c9ea723783c44576d081bb)
[![PyPI version](https://badge.fury.io/py/spectral_connectivity.svg)](https://badge.fury.io/py/spectral_connectivity)
[![Anaconda-Server Badge](https://anaconda.org/edeno/spectral_connectivity/badges/version.svg)](https://anaconda.org/edeno/spectral_connectivity)
[![Documentation Status](https://readthedocs.org/projects/spectral-connectivity/badge/?version=latest)](https://spectral-connectivity.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/Eden-Kramer-Lab/spectral_connectivity/branch/master/graph/badge.svg)](https://codecov.io/gh/Eden-Kramer-Lab/spectral_connectivity)

[**Tutorials**](#tutorials)
| [**Documentation**](#documentation)
| [**Usage Example**](#usage-example)
| [**Installation**](#installation)
| [**Developer Installation**](#developer-installation)
| [**Contributing**](#contributing)
| [**License**](#license)
| [**Citation**](#citation)

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

### Tutorials

See the following notebooks for more information on how to use the package:

+ [Tutorial](examples/Intro_tutorial.ipynb)
+ [Usage Examples](examples/Tutorial_On_Simulated_Examples.ipynb)
+ [More Usage Examples](examples/Tutorial_Using_Paper_Examples.ipynb)

### Usage Example

The quickest way to get connectivity measures is the high-level
`multitaper_connectivity` function, which runs the multitaper transform and
returns labeled [xarray](https://docs.xarray.dev/) objects (dimensions such as
`time`, `frequency`, `source`, `target`):

```python
from spectral_connectivity import multitaper_connectivity

# time_series has shape (n_time_samples, n_trials, n_signals)
coherence = multitaper_connectivity(
    time_series,
    sampling_frequency=sampling_frequency,
    method="coherence_magnitude",
    time_halfbandwidth_product=3,
)

# Ask for several measures at once (returns an xarray.Dataset)
measures = multitaper_connectivity(
    time_series,
    sampling_frequency=sampling_frequency,
    method=["coherence_magnitude", "imaginary_coherence", "phase_locking_value"],
)
```

`time_series` may also be an `xarray.DataArray`. **For DataArray inputs, dimension
names define axis roles; positions do not.** Common dimension names are
inferred and transposed automatically; for domain-specific names, pass
`time_dim`, `trial_dim`, and `signal_dim` explicitly. Ambiguous dimensions raise
instead of falling back to axis position. Numeric `time`
coordinates are interpreted as elapsed seconds and numeric `sample` coordinates
as sample numbers; both are checked against `sampling_frequency` and used to
label output window centers. A 1-D index on the signal dimension is
preserved—including its label type—as the output's `source` and `target`
coordinates unless `signal_names` is passed explicitly. Signal labels must be
unique, non-missing, NetCDF-compatible scalar strings, real numbers, datetimes,
or timedeltas.

Datetime, timedelta, and object-valued **time coordinates** are not yet
supported. Convert them to numeric elapsed seconds before calling
`multitaper_connectivity`; datetime and timedelta **signal labels** remain valid.

For directed measures, `result.sel(source="a", target="b")` means influence
from `a` to `b`. The directed-transfer-function family is available by name as
an opt-in method. The lower-level `Connectivity` methods retain their historical
array convention: `result[..., i, j]` represents `j -> i`.

#### Choosing parameters

Not sure what `time_halfbandwidth_product`, number of tapers, or window
duration to use? `suggest_parameters` picks reasonable values from your
sampling rate, signal duration, and desired frequency resolution:

```python
from spectral_connectivity import suggest_parameters

params = suggest_parameters(
    sampling_frequency=1000,
    signal_duration=2.0,  # seconds
    desired_freq_resolution=4.0,  # Hz
)
print(params)  # -> time_halfbandwidth_product, time_window_duration, n_tapers, ...
```

See also `estimate_frequency_resolution`, `estimate_n_tapers`, and
`Multitaper.summarize_parameters()` for related helpers.

#### Lower-level API

For finer control (or if you already have Fourier coefficients from another
method), use the `Multitaper` and `Connectivity` classes directly:

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
    # fft_workers=-1,  # optional: parallelize the CPU FFT across all cores
)

# Sets up computing connectivity measures/power from multitaper spectral estimate
c = Connectivity.from_multitaper(m)

# Here are a couple of examples
power = c.power()  # spectral power
coherence = c.coherence_magnitude()
weighted_phase_lag_index = c.weighted_phase_lag_index()
canonical_coherence = c.canonical_coherence(brain_area_labels)
```

### Documentation

See the documentation on [ReadTheDocs](https://spectral-connectivity.readthedocs.io/en/latest/).

For a canonical reference of connectivity metric value ranges, see [Connectivity Metric Ranges](docs/CONNECTIVITY_METRIC_RANGES.md).

### Implemented Measures

Functional

1. coherency
2. canonical_coherence
3. imaginary_coherence
4. phase_locking_value
5. phase_lag_index
6. weighted_phase_lag_index
7. debiased_squared_phase_lag_index
8. debiased_squared_weighted_phase_lag_index
9. pairwise_phase_consistency
10. global coherence

Directed

1. directed_transfer_function
2. directed_coherence
3. partial_directed_coherence
4. generalized_partial_directed_coherence
5. direct_directed_transfer_function
6. group_delay
7. phase_lag_index
8. pairwise_spectral_granger_prediction

### Package Dependencies

`spectral_connectivity` requires:

+ python
+ numpy
+ matplotlib
+ scipy
+ xarray

See [environment.yml](environment.yml) for the most current list of dependencies.

### GPU Acceleration

`spectral_connectivity` supports GPU acceleration using
[CuPy](https://cupy.dev/), which can accelerate large workloads when a suitable
CUDA device is available.

#### GPU Setup Options

There are three ways to enable GPU acceleration:

**Option 1: Environment Variable (Shell)**

```bash
export SPECTRAL_CONNECTIVITY_ENABLE_GPU=true
python your_script.py
```

**Option 2: Environment Variable (Python Script)**

```python
import os

# IMPORTANT: Must set BEFORE importing spectral_connectivity
# (Python loads modules once; changing the variable after import has no effect)
os.environ["SPECTRAL_CONNECTIVITY_ENABLE_GPU"] = "true"

from spectral_connectivity import Multitaper, Connectivity

# Verify GPU is active
import spectral_connectivity as sc

backend = sc.get_compute_backend()
print(backend["message"])
# Should print: "Using GPU backend with CuPy on <your GPU name>"
```

**Option 3: Environment Variable (Jupyter Notebook)**

```python
# In first cell (before any imports):
%env SPECTRAL_CONNECTIVITY_ENABLE_GPU=true

# In second cell:
from spectral_connectivity import Multitaper, Connectivity
import spectral_connectivity as sc

# Verify GPU is active
backend = sc.get_compute_backend()
print(f"Backend: {backend['backend']}")
print(f"Device: {backend['device_name']}")
# Should show: Backend: gpu, Device: <your GPU name>

# Note: If you already imported spectral_connectivity before setting the
# environment variable, you must restart your kernel for changes to take effect:
# Kernel → Restart & Clear Output, then run cells again
```

#### Installing CuPy

**Recommended (conda - auto-detects CUDA version):**

```bash
conda install -c conda-forge cupy
```

**Recommended pip installation (CUDA 12):**

```bash
pip install "spectral-connectivity[gpu]"
```

The package's GPU extra installs `cupy-cuda12x>=13.0`. For another CUDA version,
install the corresponding CuPy 13+ distribution directly instead of the extra.

```bash
# Check your CUDA version first: nvidia-smi
pip install "cupy-cuda11x>=13.0"  # CUDA 11.x
pip install "cupy-cuda12x>=13.0"  # CUDA 12.x
```

See [CuPy Installation Guide](https://docs.cupy.dev/en/stable/install.html) for detailed instructions and GPU-specific requirements.

#### Checking GPU Status

Use `get_compute_backend()` to check if GPU acceleration is enabled:

```python
import spectral_connectivity as sc

backend = sc.get_compute_backend()
print(backend["message"])
# Example output (GPU enabled):
# "Using GPU backend with CuPy on NVIDIA Tesla V100-SXM2-16GB."

# Or if GPU not available:
# "Using CPU backend with NumPy. To enable GPU acceleration:
#   1. Install CuPy: 'conda install -c conda-forge cupy' or
#      'pip install "spectral-connectivity[gpu]"'
#   2. Set environment variable SPECTRAL_CONNECTIVITY_ENABLE_GPU='true' before importing
# See documentation for detailed setup instructions."

# Check all details
for key, value in backend.items():
    print(f"{key}: {value}")
```

Output fields:
- `backend`: Either "cpu" or "gpu"
- `gpu_enabled`: Whether GPU was requested via environment variable
- `gpu_available`: Whether CuPy is installed and importable
- `device_name`: Name of compute device (e.g., "CPU" or "GPU (Compute Capability 7.5)")
- `message`: Human-readable explanation of current configuration

#### Troubleshooting GPU Issues

**Issue: "GPU support was requested but CuPy is not installed"**

Solution: Install CuPy as shown above, ensuring the CUDA version matches your system.

**Issue: GPU not being used even after setting environment variable**

Possible causes:
1. Environment variable set *after* importing spectral_connectivity
   - Solution: Set `SPECTRAL_CONNECTIVITY_ENABLE_GPU=true` before any imports
   - In scripts: Move the `os.environ[...]` line to the very top, before all spectral_connectivity imports
   - In notebooks: Restart kernel (Kernel → Restart & Clear Output) and set variable in first cell
2. CuPy not installed or CUDA version mismatch
   - Solution: Run `python -c "import cupy; print(cupy.__version__)"` to verify installation
3. CUDA not available on system
   - Solution: Check CUDA installation with `nvidia-smi`
4. Environment variable set in script but after import statement
   - Solution: Ensure `os.environ['SPECTRAL_CONNECTIVITY_ENABLE_GPU'] = 'true'` appears before `from spectral_connectivity import ...`

**Issue: Out of memory errors on GPU**

Solution: Use smaller batch sizes or switch back to CPU for very large datasets:
```python
# Remove or unset the environment variable
os.environ.pop("SPECTRAL_CONNECTIVITY_ENABLE_GPU", None)
```

**Issue: Need to check GPU vs CPU performance**

Use `get_compute_backend()` to verify which backend is active, then compare timing:

```python
import time
import spectral_connectivity as sc

backend = sc.get_compute_backend()
print(f"Running on: {backend['backend']}")

start = time.time()
# Your spectral connectivity code here
elapsed = time.time() - start
print(f"Elapsed time: {elapsed:.2f}s")
```

#### When to Use GPU Acceleration

GPU acceleration is most beneficial for:
- Large datasets (many signals, long recordings, or many trials)
- High frequency resolution (small time windows, many tapers)
- Computing multiple connectivity measures from the same data

For small datasets (< 10 signals, < 1000 time points), CPU may be faster due to GPU transfer overhead.

### Installation

```bash
pip install spectral_connectivity
```

or

```bash
conda install -c edeno spectral_connectivity
```

### Developer Installation

If you want to make contributions to this library, please use this installation.

1. Install miniconda (or anaconda) if it isn't already installed. Type into bash (or install from the anaconda website):

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
hash -r
```

1. Clone the repository to your local machine (`.../spectral_connectivity`) and install the anaconda environment for the repository. Type into bash:

```bash
conda env create -f environment.yml
conda activate spectral_connectivity
pip install -e .
```

## Releases

This package uses dynamic versioning with [Hatch](https://hatch.pypa.io/) based on git tags. The version is automatically determined from the repository state:

- **Tagged releases**: `1.2.0`
- **Development versions**: `1.2.0.dev5+g1a2b3c4` (5 commits since tag + git hash)

### Making a Release

Releases are published **automatically by CI**: pushing a `v*` tag runs the
*Test, Build, and Publish* workflow, which tests, builds, attests, and publishes
to PyPI via trusted publishing (after approval in the protected `pypi`
environment). Do **not** run `twine upload` by hand — that bypasses the tests,
attestations, and approval gate. See [CONTRIBUTING.md](CONTRIBUTING.md) for the
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

## Contributing

We welcome contributions to `spectral_connectivity`! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on:

+ How to report bugs and request features
+ Development workflow and coding standards
+ Testing requirements
+ Code review process

For questions or discussions, please open an issue on GitHub.

## License

This project is licensed under the **GPL-3.0** License - see the [LICENSE](LICENSE) file for details.

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
