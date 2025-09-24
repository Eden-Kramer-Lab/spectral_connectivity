# spectral_connectivity

[![PR Test](https://github.com/Eden-Kramer-Lab/spectral_connectivity/actions/workflows/PR-test.yml/badge.svg)](https://github.com/Eden-Kramer-Lab/spectral_connectivity/actions/workflows/PR-test.yml)
[![DOI](https://zenodo.org/badge/104382538.svg)](https://zenodo.org/badge/latestdoi/104382538)
[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/Eden-Kramer-Lab/spectral_connectivity/master)
[![status](https://joss.theoj.org/papers/27eb33e699c9ea723783c44576d081bb/status.svg)](https://joss.theoj.org/papers/27eb33e699c9ea723783c44576d081bb)
[![PyPI version](https://badge.fury.io/py/spectral_connectivity.svg)](https://badge.fury.io/py/spectral_connectivity)
[![Anaconda-Server Badge](https://anaconda.org/edeno/spectral_connectivity/badges/version.svg)](https://anaconda.org/edeno/spectral_connectivity)
[![Documentation Status](https://readthedocs.org/projects/spectral-connectivity/badge/?version=latest)](https://spectral-connectivity.readthedocs.io/en/latest/?badge=latest)
[![Coverage Status](https://coveralls.io/repos/github/Eden-Kramer-Lab/spectral_connectivity/badge.svg?branch=master)](https://coveralls.io/github/Eden-Kramer-Lab/spectral_connectivity?branch=master)

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
+ all function are GPU-enabled if `cupy` is installed and the environmental variable `SPECTRAL_CONNECTIVITY_ENABLE_GPU` is set to 'true'.

### Tutorials

See the following notebooks for more information on how to use the package:

+ [Tutorial](examples/Intro_tutorial.ipynb)
+ [Usage Examples](examples/Tutorial_On_Simulated_Examples.ipynb)
+ [More Usage Examples](examples/Tutorial_Using_Paper_Examples.ipynb)

### Usage Example

```python
from spectral_connectivity import Multitaper, Connectivity

# Compute multitaper spectral estimate
m = Multitaper(time_series=signals,
               sampling_frequency=sampling_frequency,
               time_halfbandwidth_product=time_halfbandwidth_product,
               time_window_duration=0.060,
               time_window_step=0.060,
               start_time=time[0])

# Sets up computing connectivity measures/power from multitaper spectral estimate
c = Connectivity.from_multitaper(m)

# Here are a couple of examples
power = c.power() # spectral power
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

To create a new release:

```bash
# 1. Update version tag
git tag v1.2.0
git push origin v1.2.0

# 2. Build and publish to PyPI
hatch build
twine upload dist/*

# 3. Build and publish to conda
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
