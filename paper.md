---
title: 'Spectral Connectivity: A Python Package for Computing Multitaper Spectral Estimates and Frequency-Domain Brain Connectivity Measures on the CPU and GPU'
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
  - name: Maxym Myroshnychenko
    affiliation: 4
    orcid: 0000-0001-7790-257X
  - name: Mehrad Sarmashghi
    orcid: 0000-0002-7976-6636
    affiliation: 5
  - name: Emily P. Stephen
    orcid: 0000-0003-1978-9622
    affiliation: 6, 7
affiliations:
 - name: Howard Hughes Medical Institute, University of California, San Francisco, San Francisco, California
   index: 1
 - name: Departments of Physiology and Psychiatry, University of California, San Francisco, San Francisco, California
   index: 2
 - name: Kavli Institute for Fundamental Neuroscience, University of California, San Francisco, San Francisco, California
   index: 3
 - name: National Institute of Neurological Disorders and Stroke, Bethesda, Maryland
   index: 4
 - name: Division of Systems Engineering, Boston University, Boston, Massachusetts
   index: 5
 - name: Department of Mathematics and Statistics, Boston University, Boston, Massachusetts
   index: 6
 - name: Center for Systems Neuroscience, Boston University, Boston, Massachusetts
   index: 7
date: 10 September 2022
bibliography: paper.bib
---

# Summary

In neuroscience, characterizing the oscillatory dynamics of the brain is critical to understanding how brain areas interact and function. Neuronal activity tends to fluctuate rhythmically – both through intrinsic currents at the cellular level and through groups of neurons. Brain oscillations and their relationships can indicate the difference between normal and pathological brain states such as Alzheimer's and epilepsy. Spectral analysis techniques such as multitaper and wavelet analysis are widely used for decomposing signals into oscillatory components. Connectivity measures are used to determine the relationships between those oscillatory components, indicating possible communication between brain areas. Because these analyses are central to neuroscience and technological advances in recording are increasing the amount of simultaneously recorded signals, it is important to have a well-tested, standardized, and lightweight software package to compute these brain connectivity measures at scale.

# Statement of Need

`spectral_connectivity` is a Python software package that computes multitaper spectral estimates and frequency-domain brain connectivity measures. The programming language Python is increasingly being used in the neurosciences[@MullerPythonNeuroscience2015; @SchlaflyPythonpracticingneuroscientist2020], but the two main packages for spectral analysis in Python, `nitime` [@Rokem2020] and `mne-python` [@GramfortEtAl2013a], have issues that make them more difficult to use in many situations. For example, `nitime` implements several estimators of the power spectrum, but lacks spectrograms and windowed spectral estimators. `mne-python` is a much larger package designed as a full-featured analysis library for EEG and MEG data, and using the spectral analysis functions requires representing data using its ecosystem. Users of other data modalities, such as non-human electrophysiology data, may thus find much of `mne-python` inappropriate or unsuited for their analyses. This is an important problem because the non-human neurosciences are undergoing a period of great technological development; more and more signals are being collected simultaneously, and the duration of these signals is becoming longer as chronic recordings become possible. This rapid increase in the size and duration of datasets demands a lightweight, fast, and efficient spectral estimation package. `spectral_connectivity` is designed to handle multiple time series and can exploit GPUs for faster and more efficient computation. In addition, it can block compute important quantities such as the cross-spectral matrix in order to reduce memory burdens caused by large datasets. `spectral_connectivity` is also designed to be a lightweight package that has a simple user interface and can be easily be incorporated with other packages. Finally, `spectral_connectivity` also implements several connectivity measures that have not previously been implemented in Python such as the non-parametric version of the spectral granger causality and canonical coherence.

`spectral_connectivity` has already shown its utility to the neuroscience field. The package has already been used in a number of publications and pre-prints in neuroscience [@KuhnertDetectionDirectedConnectivities2019; @VargaNetworkPathConvergence2021; @LauroSubthalamicCorticalNetwork2021;
  , @Delgado-SallentPhencyclidineinducedpsychosiscauses2021]. Interestingly, it has also contributed to a publication in physics [@CliffUnifyingPairwiseInteractions2022], showing its versatility and easy of use. We hope this package will continue to be useful to the neuroscience community, particularly for non-human electrophysiology data.

# Acknowledgements

We thank Uri T. Eden for support and mentorship during the creation of this package.

# Citations
