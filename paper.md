---
title: 'Spectral Connectivity'
tags:
  - Python
  - Neuroscience
  - Multitaper analysis
  - Spectral estimation
  - Brain connectivity measures
  - Fourier transform
authors:
  - name: Eric L. Denovellis
    orcid: 0000-0003-4606-087X
    affiliation: 1, 2, 3
  - name: Mehrad Sarmashghi
    orcid: 0000-0002-7976-6636
    affiliation: 4
  - name: Maxym Myroshnychenko
    affiliation: 5
    orcid: 0000-0001-7790-257X
  - name: Emily P. Stephen
    orcid: 0000-0003-1978-9622
    affiliation: 6
affiliations:
 - name: Howard Hughes Medical Institute, University of California, San Francisco, San Francisco, California
   index: 1
 - name: Departments of Physiology and Psychiatry, University of California, San Francisco, San Francisco, California
   index: 2
 - name: Kavli Institute for Fundamental Neuroscience, University of California, San Francisco, San Francisco, California
   index: 3
 - name: Division of Systems Engineering, Boston University
   index: 4
 - name: National Institutes of Health
   index: 5
 - name: Department of Mathematics and Statistics, Boston University
   index: 6
date: 10 September 2022
bibliography: paper.bib
---

# Summary

In neuroscience, characterizing the oscillatory dynamics of the brain is critical to understanding how brain areas interact and function. Neurons tend to fluctuate rhythmically in excitation – both through intrinsic currents at the cellular level and as groups. Brain oscillations and their relationships can indicate the difference between normal and pathological states such as in Alzheimer's and epilepsy. Spectral analysis techniques such as multitaper and wavelet analysis are widely used for decomposing signals into oscillatory components and connectivity measures are used to determine the relationships between those oscillatory components, indicating possible communication between brain areas. Because these analyses are so central to neuroscience, it is important to have a well-tested, standardized, and lightweight software package to compute these brain connectivity measures at scale.


# Statement of Need

`spectral_connectivity` is a python software package that computes multitaper spectral estimates and frequency-domain brain connectivity measures. Python is a programming language increasingly being used in the neurosciences, but there are relatively few software packages written in python. For computing spectral estimates and frequency-domain brain connectivity measures, there are two main packages: `nitime` and `mne-python`. `spectral_connectivity` has several differences. The neurosciences are also undergoing period of great technological development; more and more signals are being collected simultaneously and the duration of these signals is becoming longer as chronic recordings become possible. `spectral_connectivity` is designed to handle multiple time series, it can exploit GPUs for faster and more efficient computation. In addition, it can block compute important quantities such as the cross-spectral matrix in order to reduce memory burdens caused by large datasets. `spectral_connectivity` also implements several connectivity measures that have not previously been implemented in python such as the non-parametric version of the spectral granger causality and canonical coherence. Finally, `spectral_connectivity` is designed to be a lightweight package that has a simple API and can be easily be incorporated with other packages.

`spectral_connectivity` has already contributed to a number of publications and pre-prints in neuroscience ([@KuhnertDetectionDirectedConnectivities2019], [@VargaNetworkPathConvergence2021], [@LauroSubthalamicCorticalNetwork2021],
  , [@Delgado-SallentPhencyclidineinducedpsychosiscauses2021]) and one in physics [@CliffUnifyingPairwiseInteractions2022].

# Acknowledgements
We thank Uri T. Eden for support.
