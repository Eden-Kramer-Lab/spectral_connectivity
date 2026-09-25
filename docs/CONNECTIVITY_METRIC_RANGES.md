# Connectivity Metric Ranges

## Direction convention

For xarray results returned by `multitaper_connectivity` and
`fourier_connectivity`, `result.sel(source="a", target="b")` is the influence
from `a` to `b` (or, for phase measures, how much `a` leads `b`) for every
measure.

The lower-level `Connectivity` methods return plain arrays whose last two axes
are signals, in one of two orders (the *Low-level orientation* column below,
and `MeasureInfo.array_orientation` from `list_measures()`):

- **`j -> i`** (`"target_source"`): the Granger and directed-transfer-function
  families. `result[..., i, j]` is the influence of signal `j` on signal `i`.
- **`i` relative to `j`** (`"source_target"`): `directed_phase_lag_index`,
  `phase_slope_index`, `delay`, and `group_delay`. Positive `result[..., i, j]`
  (above 0.5 for the directed phase lag index) means signal `i` leads `j`.

The signed, non-directional phase measures (`coherence_phase`,
`imaginary_coherency`, `phase_lag_index`, `weighted_phase_lag_index`) follow
the same sign rule: positive `[..., i, j]` means `i` leads `j`.

## Measures

Generated from `list_measures()`. For complex measures the range bounds the
magnitude. "Observations" are trials × tapers.

<!-- measure-table:start -->
| Measure | Range | Units | Low-level orientation | Interpretation |
|---|---|---|---|---|
| `coherence_magnitude` | [0, 1] | 1 |  | Linear coupling at each frequency: 0 is none, 1 is a perfectly consistent amplitude and phase relationship. Biased upward when trials x tapers is small. |
| `coherence_phase` | [-π, π] | rad |  | Mean phase difference in radians. Positive (source=a, target=b) means a leads b. |
| `debiased_squared_phase_lag_index` | [-1, 1] | 1 |  | Bias-corrected squared phase lag index. Negative values are finite-sample noise around zero, not negative coupling. Lower bound is -1 / (n_observations - 1). |
| `debiased_squared_weighted_phase_lag_index` | [-1, 1] | 1 |  | Bias-corrected squared weighted phase lag index. Negative values are finite-sample noise around zero, not negative coupling. |
| `imaginary_coherence` | [0, 1] | 1 |  | Magnitude of the imaginary part of coherency; blind to zero-lag coupling such as volume conduction. |
| `pairwise_phase_consistency` | [-1, 1] | 1 |  | Bias-free estimate of the squared phase-locking value. Negative values are finite-sample noise around zero, not negative coupling. Lower bound is -1 / (n_observations - 1). |
| `pairwise_spectral_granger_prediction` | [0, ∞) | 1 | `[..., i, j]` is `j -> i` | Nonparametric spectral Granger causality from source to target: 0 is no directed influence; larger values mean more of the target's power is predicted by the source's past. Not conditioned on other signals. |
| `phase_lag_index` | [-1, 1] | 1 |  | Signed asymmetry of the phase-difference distribution; blind to zero-lag coupling. Take the absolute value for the unsigned index. Positive (source=a, target=b) means a leads b. |
| `phase_locking_value` | [0, 1] | 1 |  | Consistency of the phase difference across trials and tapers, ignoring amplitude: 0 is random, 1 is constant. Biased upward with few observations. |
| `power` | [0, ∞) | (input units)^2/Hz |  | One-sided power spectral density of each signal. |
| `weighted_phase_lag_index` | [-1, 1] | 1 |  | Phase lag index weighted by the magnitude of the imaginary cross-spectrum; less sensitive to noise than the unweighted index. Positive (source=a, target=b) means a leads b. |
| `coherency` | \|z\| in [0, 1] | 1 |  | Complex coherency: its squared magnitude is coherence_magnitude and its angle is coherence_phase. |
| `cross_spectral_density` | \|z\| in [0, ∞) | (input units)^2/Hz |  | Complex, Hermitian cross-spectrum; unnormalized, so it scales with signal power. Use coherency for a normalized version. |
| `imaginary_coherency` | [-1, 1] | 1 |  | Signed imaginary part of coherency; blind to zero-lag coupling. Positive (source=a, target=b) means a leads b. |
| `partial_coherence` | [0, 1] | 1 |  | Magnitude-squared coherence after removing the linear influence of every other signal; near 0 for pairs coupled only through other signals. |
| `corrected_imaginary_phase_locking_value` | [0, 1] | 1 |  | Phase locking with zero- and pi-lag contributions removed; insensitive to volume conduction. |
| `directed_phase_lag_index` | [0, 1] | 1 | `[..., i, j]` is `i` relative to `j` | Above 0.5, the source phase-leads the target; below 0.5 it lags; 0.5 is no preferred direction. |
| `subset_pairwise_spectral_granger_prediction` | [0, ∞) | 1 | `[..., i, j]` is `j -> i` | pairwise_spectral_granger_prediction for only the requested pairs; other entries are NaN. |
| `conditional_spectral_granger_prediction` | [0, ∞) | 1 | `[..., i, j]` is `j -> i` | Spectral Granger causality from source to target conditioned on every other signal, removing influence relayed through observed signals. |
| `time_reversed_spectral_granger_prediction` | [0, ∞) | 1 | `[..., i, j]` is `j -> i` | Pairwise spectral Granger causality of the time-reversed data. Genuine directed influence reverses under time reversal; directionality that does not reverse suggests instantaneous mixing. |
| `directed_transfer_function` | [0, 1] | 1 | `[..., i, j]` is `j -> i` | Fraction of the target's inflow at each frequency that comes from the source, including indirect paths; sums to 1 over sources. |
| `directed_coherence` | [0, 1] | 1 | `[..., i, j]` is `j -> i` | Noise-weighted directed transfer function: the fraction of the target's power attributable to the source; sums to 1 over sources. Assumes uncorrelated innovations. |
| `partial_directed_coherence` | [0, 1] | 1 | `[..., i, j]` is `j -> i` | Direct influence from source to target, normalized by the source's total outflow; sums to 1 over targets. |
| `generalized_partial_directed_coherence` | [0, 1] | 1 | `[..., i, j]` is `j -> i` | Partial directed coherence with each signal scaled by its innovation variance, making it insensitive to differences in signal scale. |
| `direct_directed_transfer_function` | [0, 1] | 1 | `[..., i, j]` is `j -> i` | Direct (not relayed) influence from source to target. Normalized over all frequencies, so values are small: compare pairs, not against 1. |
| `blockwise_spectral_granger_prediction` | [0, ∞) | 1 | `[..., i, j]` is `j -> i` | Spectral Granger causality between groups of signals set by group_labels, from source_group to target_group. |
| `canonical_coherence` | [0, 1] | 1 |  | Largest coherence between linear combinations of two groups of signals (historical estimator; see canonical_coherency). |
| `maximized_imaginary_coherency` | [0, 1] | 1 |  | Largest imaginary coherency between linear combinations of two groups; blind to zero-lag coupling. |
| `multivariate_interaction_measure` | [0, ∞) | 1 |  | Total phase-lagged interaction between two groups (the sum of squared imaginary-coherency components); at most the smaller group's rank. |
| `canonical_coherency` | \|z\| in [0, 1] | 1 |  | Complex canonical coherency per component between two groups, with the spatial filters and patterns that produce it. |
| `maximized_imaginary_coherency_components` | [0, 1] | 1 |  | maximized_imaginary_coherency resolved into components, with the spatial filters and patterns that produce them. |
| `delay` | (-∞, ∞) | s | `[..., i, j]` is `i` relative to `j` | Candidate delays in seconds, one per 2*pi phase ambiguity; the true delay is the candidate that is constant across frequency. Frequencies without significant coherence are NaN. Positive (source=a, target=b) means a leads b. |
| `global_coherence` | [0, 1] | 1 |  | Fraction of the total cross-spectral power in each component; a large leading component indicates one dominant coherent network. |
| `group_delay` | (-∞, ∞) | s | `[..., i, j]` is `i` relative to `j` | Delay in seconds from the slope of phase against frequency over the band; check group_delay_r_value for the quality of the fit. Positive (source=a, target=b) means a leads b. |
| `phase_slope_index` | (-∞, ∞) | 1 | `[..., i, j]` is `i` relative to `j` | Coherence-weighted slope of phase against frequency over the band. Unnormalized, so judge it against a null distribution rather than a fixed threshold. Positive (source=a, target=b) means a leads b. |
<!-- measure-table:end -->

## Notes

- For metrics theoretically bounded in [0, 1], numerical implementations should clamp to the interval after computation to avoid tiny overflows (e.g., 1 ± 1e−12). Unbiased estimators such as PPC and the debiased PLI variants must retain legitimate negative finite-sample values.
- For phase‑based metrics, choose an expectation (sample) axis (trials/tapers/segments) and aggregate **only** across that axis.
- Document the shapes/dtypes of inputs and outputs in each docstring per NumPy style.
