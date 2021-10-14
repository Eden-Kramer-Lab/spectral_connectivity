# spectral_connectivity
![Build](https://travis-ci.org/Eden-Kramer-Lab/spectral_connectivity.svg?branch=master)[![Coverage Status](https://coveralls.io/repos/github/Eden-Kramer-Lab/spectral_connectivity/badge.svg?branch=master)](https://coveralls.io/github/Eden-Kramer-Lab/spectral_connectivity?branch=master) [![DOI](https://zenodo.org/badge/104382538.svg)](https://zenodo.org/badge/latestdoi/104382538)
[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/Eden-Kramer-Lab/spectral_connectivity/master)

`spectral_connectivity` is a python software package that computes multitaper spectral estimates and frequency-domain brain connectivity measures such as coherence, spectral granger causality, and the phase lag index using the multitaper Fourier transform. Although there are other python packages that do this (see [nitime](https://github.com/nipy/nitime) and [MNE-Python](https://github.com/mne-tools/mne-python)), spectral has several differences:

+ it is designed to handle multiple time series at once
+ it caches frequently computed quantities such as the cross-spectral matrix and minimum-phase-decomposition, so that connectivity measures that use the same processing steps can be more quickly computed.
+ it decouples the time-frequency transform and the connectivity measures so that if you already have a preferred way of computing Fourier coefficients (i.e. from a wavelet transform), you can use that instead.
+ it implements the non-parametric version of the spectral granger causality in Python.
+ it implements the canonical coherence, which can
efficiently summarize brain-area level coherences from multielectrode recordings.
+ easier user interface for the multitaper fourier transform

See the notebooks ([\#1](examples/Tutorial_On_Simulated_Examples.ipynb), [\#2](examples/Tutorial_Using_Paper_Examples.ipynb)) for more information on how to use the package.

### Usage Example ###
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

### Documentation ###
See the documentation [here](http://spectral-connectivity.readthedocs.io/en/latest/index.html).

### Spectral Estimation ###
1. Multitaper

### Implemented Measures ###
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

10. directed_transfer_function
11. directed_coherence
12. partial_directed_coherence
13. generalized_partial_directed_coherence
14. direct_directed_transfer_function
15. group_delay
16. phase_lag_index
17. pairwise_spectral_granger_prediction

### Package Dependencies ###
`spectral_connectivity` requires:
- python
- numpy
- matplotlib
- scipy
- xarray

See [environment.yml](environment.yml) for the most current list of dependencies.

### Installation ###
```python
pip install spectral_connectivity
```
or
```python
conda install -c edeno spectral_connectivity
```

### Developer Installation ###
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
conda update -q conda
conda info -a
conda env create -f environment.yml
source activate spectral_connectivity
python setup.py develop
```
