.. toctree::
   :hidden:

   Home page <self>
   Jupyter tutorials <tutorials>
   API reference <_autosummary/spectral_connectivity>

=====================
spectral_connectivity
=====================

**spectral_connectivity** is a python software package that computes frequency-domain brain connectivity measures such as coherence, spectral granger causality, and the phase lag index using the multitaper Fourier transform. Although there are other python packages that do this (see nitime and MNE-Python), spectral_connectivity has several differences:

* it is designed to handle multiple time series at once.
* it caches frequently computed quantities such as the cross-spectral matrix and minimum-phase-decomposition, so that connectivity measures that use the same processing steps can be more quickly computed.
* it decouples the time-frequency transform and the connectivity measures so that if you already have a preferred way of computing Fourier coefficients (i.e. from a wavelet transform), you can use that instead.
* it implements the non-parametric version of the spectral granger causality in Python.
* it implements the canonical coherence, which can efficiently summarize brain-area level coherences from multielectrode recordings.
* easier user interface for the multitaper fourier transform.
* all function are GPU-enabled if `cupy` is installed and the environmental variable `SPECTRAL_CONNECTIVITY_ENABLE_GPU` is set to 'true'.