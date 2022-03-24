---
title: 'Spectral Connectivity'
tags:
  - Python
  - neuroscience
  - multitaper analysis
  - spectral estimates
  - brain connectivity measures
  - fourier transform
authors:
  - name: Eric Denovellis
    orcid: 0000-0003-4606-087X
    affiliation: 1
  - name: Max Myroshnychenko
    affiliation: 2
  - name: Danylo Ulianych
    affiliation: 3
  - name: Mehrad Sarmashghi
    orcid: 0000-0002-7976-6636
    affiliation: 4
affiliations:
 - name: University of California, San Francisco
   index: 1
 - name:  University of California, San Francisco
   index: 2
 - name:  University of California, San Francisco
   index: 3
 - name: Boston University 
   index: 4
date: 10 March 2022
bibliography: paper.bib
---

# Summary

Characterizing oscillatory dynamics in brain is critical to understand functional connectivity among brain areas in system neuroscience. Spectral analysis techniques are widely used for analyzing the oscillatory behaviors and allow for understanding of brain functions. There are many spectral analysis methods such as wavelet, multitaper and P<sub>episods</sub> . Here, we focused on multitaper spectra technique which provides a smooth estimation of spectral density function of recorded time series signals based on their Fourier transform using multiple tapers with different shapes. Multitaper is a non-parametric estimation that allows for reducing the bias and variance of the estimation simultaneously [@dhamala2008analyzing]. 

`spectral_connectivity` is a python software package that computes multitaper spectral estimates and frequency-domain brain connectivity measures such as coherence, spectral granger causality, and the phase lag index using the multitaper Fourier transform. There are other python packages that have been already developed to do these analysis; however, spectral has several differences: (1) it is designed to handle multiple time series at once, (2) it caches frequently computed quantities such as the cross-spectral matrix and minimum-phase-decomposition, so that connectivity measures that use the same processing steps can be more quickly computed,
(4) it decouples the time-frequency transform and the connectivity measures so that if you already have a preferred way of computing Fourier coefficients (i.e. from a wavelet transform), you can use that instead, (5) it implements the non-parametric version of the spectral granger causality in Python, (6) it implements the canonical coherence, which can efficiently summarize brain-area level coherences from multielectrode recordings, and (7) easier user interface for the multitaper fourier transform 


# Acknowledgements


# References
