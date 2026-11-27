# Connectivity Metric Ranges

## Direction convention

For xarray results returned by `multitaper_connectivity` and
`fourier_connectivity`, `result.sel(source="a", target="b")` is the influence
from `a` to `b` (or, for phase measures, how much `a` leads `b`) for every
measure.

The lower-level `Connectivity` methods return plain arrays whose last two axes
are signals in the same source-first order: `result[..., i, j]` is the
influence of signal `i` on signal `j` (`i -> j`), and for the lead/lag and
signed phase measures a positive `result[..., i, j]` (above 0.5 for the
directed phase lag index) means `i` leads `j`.

## Measures

Generated from `list_measures()`. Ranges are the estimators' theoretical
bounds. A computed value can lie outside one by rounding, and by more on
degenerate input: with complex64 coefficients and fewer observations than a
group's channels, `canonical_coherency`'s magnitude has exceeded 1 by up to 2.8e-3.
For complex measures the range bounds the magnitude. "Observations" are trials × tapers. The Diagonal column is what a
self-connection holds (see [Diagonal values](#diagonal-values)); a number is
the constant every diagonal entry has, to within rounding. Flat signal is NaN
where every entry involving a zero-power signal is NaN.

<!-- measure-table:start -->
| Measure | Range | Units | Diagonal | Flat signal | Interpretation |
|---|---|---|---|---|---|
| `magnitude_squared_coherence` | [0, 1] | 1 | NaN (masked) | NaN | Linear coupling at each frequency: 0 is none, 1 is a perfectly consistent amplitude and phase relationship. Biased upward when trials x tapers is small. |
| `coherence_phase` | [-π, π] | rad | NaN (masked) | NaN | Mean phase difference in radians. Positive (source=a, target=b) means a leads b. |
| `debiased_squared_phase_lag_index` | [-1, 1] | 1 | 0 |  | Bias-corrected squared phase lag index. Negative values are finite-sample noise around zero, not negative coupling. Lower bound is -1 / (n_observations - 1). |
| `debiased_squared_weighted_phase_lag_index` | [-1, 1] | 1 | 0 |  | Bias-corrected squared weighted phase lag index. Negative values are finite-sample noise around zero, not negative coupling. |
| `imaginary_coherence` | [0, 1] | 1 | 0 | NaN | Magnitude of the imaginary part of coherency; blind to zero-lag coupling such as volume conduction. |
| `pairwise_phase_consistency` | [-1, 1] | 1 | 1 | NaN | Bias-free estimate of the squared phase-locking value. Negative values are finite-sample noise around zero, not negative coupling. Lower bound is -1 / (n_observations - 1). |
| `pairwise_spectral_granger_prediction` | [0, ∞) | 1 | NaN (undefined) |  | Nonparametric spectral Granger causality from source to target: 0 is no directed influence; larger values mean more of the target's power is predicted by the source's past. Not conditioned on other signals. |
| `phase_lag_index` | [-1, 1] | 1 | 0 |  | Signed asymmetry of the phase-difference distribution; blind to zero-lag coupling. Take the absolute value for the unsigned index. Positive (source=a, target=b) means a leads b. |
| `phase_locking_value` | [0, 1] | 1 | 1 | NaN | Consistency of the phase difference across trials and tapers, ignoring amplitude: 0 is random, 1 is constant. Biased upward with few observations. |
| `power` | [0, ∞) | (input units)^2/Hz | none |  | One-sided power spectral density of each signal. |
| `weighted_phase_lag_index` | [-1, 1] | 1 | 0 |  | Phase lag index weighted by the magnitude of the imaginary cross-spectrum; less sensitive to noise than the unweighted index. Positive (source=a, target=b) means a leads b. |
| `coherency` | \|z\| in [0, 1] | 1 | NaN (masked) | NaN | Complex coherency: its squared magnitude is magnitude_squared_coherence and its angle is coherence_phase. A positive angle at (source=a, target=b) means a leads b. |
| `cross_spectral_density` | \|z\| in [0, ∞) | (input units)^2/Hz | estimated |  | Complex, Hermitian cross-spectrum; unnormalized, so it scales with signal power. Use coherency for a normalized version. |
| `imaginary_coherency` | [-1, 1] | 1 | NaN (masked) |  | Signed imaginary part of coherency; blind to zero-lag coupling. Positive (source=a, target=b) means a leads b. |
| `partial_coherence` | [0, 1] | 1 | NaN (masked) |  | Magnitude-squared coherence after removing the linear influence of every other signal; near 0 for pairs coupled only through other signals. |
| `corrected_imaginary_phase_locking_value` | [0, 1] | 1 | 0, plus rounding noise | NaN | Phase locking with zero- and pi-lag contributions removed; insensitive to volume conduction. |
| `directed_phase_lag_index` | [0, 1] | 1 | 0.5 |  | Fraction of observations in which the source's phase leads the target's (a tie counts half), (1 + phase_lag_index) / 2; 0.5 is no preferred direction. Above 0.5 at (source=a, target=b) means a leads b; below 0.5, a lags b. |
| `subset_pairwise_spectral_granger_prediction` | [0, ∞) | 1 | NaN (undefined) |  | pairwise_spectral_granger_prediction for only the requested pairs; other entries are NaN. |
| `conditional_spectral_granger_prediction` | [0, ∞) | 1 | NaN (undefined) |  | Spectral Granger causality from source to target conditioned on every other signal, removing influence relayed through observed signals. |
| `time_reversed_spectral_granger_prediction` | [0, ∞) | 1 | NaN (undefined) |  | Pairwise spectral Granger causality of the time-reversed data. Genuine directed influence reverses under time reversal; directionality that does not reverse suggests instantaneous mixing. |
| `time_reversed_conditional_spectral_granger_prediction` | [0, ∞) | 1 | NaN (undefined) |  | Conditional spectral Granger causality of the time-reversed data. Genuine directed influence reverses under time reversal; directionality that does not reverse suggests instantaneous mixing. |
| `directed_transfer_function` | [0, 1] | 1 | estimated |  | Fraction of the target's inflow at each frequency that comes from the source, including indirect paths; sums to 1 over sources. |
| `directed_coherence` | [0, 1] | 1 | estimated |  | Noise-weighted directed transfer function: the fraction of the target's power attributable to the source; sums to 1 over sources. Assumes uncorrelated innovations. |
| `partial_directed_coherence` | [0, 1] | 1 | estimated |  | Direct influence from source to target, normalized by the source's total outflow; sums to 1 over targets. |
| `generalized_partial_directed_coherence` | [0, 1] | 1 | estimated |  | Partial directed coherence with each signal scaled by its innovation variance, making it insensitive to differences in signal scale. |
| `direct_directed_transfer_function` | [0, 1] | 1 | estimated |  | Influence from source to target, zero when relayed through observed signals; also positive when source and target drive a common observed signal, so it does not establish a direct link. Normalized over all frequencies, so values are small: compare pairs, not against 1. |
| `blockwise_spectral_granger_prediction` | [0, ∞) | 1 | NaN (undefined) |  | Spectral Granger causality between groups of signals set by group_labels, from source_group to target_group. |
| `time_reversed_blockwise_spectral_granger_prediction` | [0, ∞) | 1 | NaN (undefined) |  | Blockwise spectral Granger causality of the time-reversed data, from source_group to target_group. Genuine directed influence reverses under time reversal; directionality that does not reverse suggests instantaneous mixing. |
| `canonical_coherence` | [0, 1] | 1 | NaN (masked) |  | Largest magnitude-squared coherence between linear combinations of two groups of signals, the squared first canonical correlation (the historical estimator). A different estimator from canonical_coherency, not its square. |
| `maximized_imaginary_coherency` | [0, 1] | 1 | NaN (masked) |  | Largest imaginary coherency between linear combinations of two groups; blind to zero-lag coupling. |
| `multivariate_interaction_measure` | [0, ∞) | 1 | NaN (masked) |  | Total phase-lagged interaction between two groups (the sum of squared imaginary-coherency components); at most the smaller group's rank. |
| `canonical_coherency` | \|z\| in [0, 1] | 1 | none |  | Phase-optimized complex canonical coherency (Vidaurre et al. 2019) per component between two groups, with the spatial filters and patterns that produce it; its magnitude is the maximized coherence, not squared. A different estimator from canonical_coherence, not its square root. Conjugate convention: the score is magnitude * exp(-1j * phi), so with single-channel groups it is conj(coherency) and a negative angle means the source group leads. The phase is defined only up to the sign of the spatial filters. |
| `maximized_imaginary_coherency_components` | [0, 1] | 1 | none |  | maximized_imaginary_coherency resolved into components, with the spatial filters and patterns that produce them. |
| `delay` | (-∞, ∞) | s | NaN (masked) | NaN | Candidate delays in the reciprocal units of frequency (seconds for Hz), one per 2*pi phase ambiguity; the true delay is the candidate that is constant across frequency. Frequencies without significant coherence are NaN. Positive (source=a, target=b) means a leads b. |
| `global_coherence` | [0, 1] | 1 | none |  | Fraction of the total cross-spectral power in each component; a large leading component indicates one dominant coherent network. The leading value is at least 1/min(n_observations, n_signals) for any data and biased above 1/n_signals unless observations far outnumber signals, so compare values only at matched numbers of observations and signals. |
| `group_delay` | (-∞, ∞) | s | NaN (masked) | NaN | Delay in the reciprocal units of frequency (seconds for Hz), from the slope of phase against frequency over frequency_range (every bin by default); check group_delay_r_value for the quality of the fit. Positive (source=a, target=b) means a leads b. |
| `phase_slope_index` | (-∞, ∞) | 1 | NaN (masked) | NaN | Coherence-weighted slope of phase against frequency over frequency_range (every bin by default). Unnormalized, so judge it against a null distribution rather than a fixed threshold. Positive (source=a, target=b) means a leads b. |
<!-- measure-table:end -->

## Coherence and its relatives

Every measure in this family is computed from the cross-spectrum
`S_st = E[X_s * conj(X_t)]`, averaged over the expectation axes (trials ×
tapers by default). Their names say which quantity they hold:

<!-- formula-table:start -->
| Measure | Quantity | Formula |
|---|---|---|
| `magnitude_squared_coherence` | squared magnitude | \|S_st\|^2 / (S_ss * S_tt), where S_st = E[X_s * conj(X_t)], averaged over the expectation_type axes |
| `coherence_phase` | phase | angle(S_st), the angle of coherency, where S_st = E[X_s * conj(X_t)], averaged over the expectation_type axes |
| `imaginary_coherence` | magnitude | \|Im(S_st)\| / sqrt(S_ss * S_tt), where S_st = E[X_s * conj(X_t)], averaged over the expectation_type axes |
| `coherency` | complex | S_st / sqrt(S_ss * S_tt), where S_st = E[X_s * conj(X_t)], averaged over the expectation_type axes |
| `imaginary_coherency` | signed | Im(S_st) / sqrt(S_ss * S_tt), where S_st = E[X_s * conj(X_t)], averaged over the expectation_type axes |
| `partial_coherence` | squared magnitude | \|P_st\|^2 / (P_ss * P_tt), where P is the (regularized) inverse of the cross-spectral matrix, optionally shrunk toward its diagonal first (shrinkage) |
| `canonical_coherence` | squared magnitude | max over complex weights a, b of \|a^H S_st b\|^2 / ((a^H S_ss a) * (b^H S_tt b)), with S_st the cross-spectral matrix from the source group's signals to the target group's |
| `canonical_coherency` | complex | max over a phase phi and real weights a, b of a^T Re(exp(-1j * phi) * S_st) b / sqrt((a^T Re(S_ss) a) * (b^T Re(S_tt) b)), times exp(-1j * phi), with S_st the cross-spectral matrix from the source group's signals to the target group's |
<!-- formula-table:end -->

- `magnitude_squared_coherence` is the squared magnitude of `coherency`; it is
  what [SciPy calls coherence](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.coherence.html).
  Its 2.x name, `coherence_magnitude`, still works through 3.x with a
  `DeprecationWarning` (removed in 4.0). The magnitude of `coherency`, unsquared,
  is `abs(coherency)`.
- `imaginary_coherence` is the magnitude of `imaginary_coherency`: they differ
  only by `abs`, and only `imaginary_coherency` keeps the sign that says which
  signal leads.
- `canonical_coherence` and `canonical_coherency` are different estimators,
  neither the square nor the square root of the other. `canonical_coherence`
  is the historical estimator: the largest *squared* canonical correlation,
  optimized over complex channel weights. `canonical_coherency` is the
  phase-optimized CaCoh of Vidaurre et al. (2019): real channel weights and one
  phase, with the *unsquared* maximized coherence as its magnitude. For groups
  of one signal each they reduce to the pairwise measures,
  `canonical_coherence == magnitude_squared_coherence` and
  `canonical_coherency == conj(coherency)`; for larger groups they generally
  differ even after squaring.

Each result variable records these facts in its attributes: `quantity` and
`formula` (for this family), `sign_convention` (for signed measures and
angles), `interpretation`, `value_range_lower`/`value_range_upper` (an
unbounded side is omitted; `value_range_applies_to="magnitude"` for complex
measures), and `diagonal_policy`.

## Diagonal values

The diagonal of a pairwise result, `result.sel(source="a", target="a")` (or
`source_group`/`target_group` for group measures), holds a self-connection.
What it holds is recorded as each variable's `diagonal_policy` attribute and in
the Diagonal column above. This table, generated from the registry, is the
definition of each policy:

<!-- policy-table:start -->
| `diagonal_policy` | A self-connection holds | Variables |
|---|---|---|
| `masked_nan` | NaN by the package's choice, although the estimator defines a value there: self-coherence is 1 for a signal with nonzero power, its phase and imaginary part are 0, and partial and canonical coherence of a signal (group) with itself are 1; the maximized imaginary coherency and interaction measure do not compute a group with itself. The delay measures inherit coherency's mask. | `magnitude_squared_coherence`, `coherence_phase`, `coherency`, `imaginary_coherency`, `partial_coherence`, `canonical_coherence`, `maximized_imaginary_coherency`, `multivariate_interaction_measure`, `delay`, `group_delay`, `phase_slope_index`, `group_delay_slope` |
| `undefined_nan` | NaN: the estimator has no self-connection (a signal cannot Granger-cause itself in these models). | `pairwise_spectral_granger_prediction`, `subset_pairwise_spectral_granger_prediction`, `conditional_spectral_granger_prediction`, `time_reversed_spectral_granger_prediction`, `time_reversed_conditional_spectral_granger_prediction`, `blockwise_spectral_granger_prediction`, `time_reversed_blockwise_spectral_granger_prediction` |
| `constant` | diagonal_value, the same for any signal with nonzero power, to within the rounding of the averages it is computed from: at most 4 * (n_observations + 4) times machine epsilon away, since normalizing each averaged term leaves about 4 eps, the sum adds n_observations eps, and pairwise phase consistency multiplies the error by 2 n / (n - 1) <= 4. Exact on some platforms. | `debiased_squared_phase_lag_index` (0), `debiased_squared_weighted_phase_lag_index` (0), `imaginary_coherence` (0), `pairwise_phase_consistency` (1), `phase_lag_index` (0), `phase_locking_value` (1), `weighted_phase_lag_index` (0), `directed_phase_lag_index` (0.5), `group_delay_r_value` (1) |
| `rounding_noise` | Nominally diagonal_value (a defined 0/0 limit), but computed from rounding residues that the formula amplifies, so it can lie much farther from that value than rounding: the corrected imaginary PLV divides an imaginary part of order eps by sqrt(1 - Re^2) of at least sqrt(eps), so its diagonal is below sqrt(eps) (1.5e-8 in float64). Mask it rather than read it. | `corrected_imaginary_phase_locking_value` (0) |
| `estimated` | The estimator's own, data-dependent value: the power spectrum for the cross-spectral density (real to within rounding), each signal's self-transfer for the transfer-function measures. | `cross_spectral_density`, `directed_transfer_function`, `directed_coherence`, `partial_directed_coherence`, `generalized_partial_directed_coherence`, `direct_directed_transfer_function` |
| `not_applicable` | No self-connection entries. | `power`, `canonical_coherency`, `maximized_imaginary_coherency_components`, `global_coherence` |

`nan_for_zero_power_signal`: Every entry involving a zero-power (flat) signal, its self-connection included, is NaN: the measure normalizes by each signal's amplitude. Where it is not set, a flat signal can still give some NaN entries, or finite ones. Set for `magnitude_squared_coherence`, `coherence_phase`, `imaginary_coherence`, `pairwise_phase_consistency`, `phase_locking_value`, `coherency`, `corrected_imaginary_phase_locking_value`, `delay`, `group_delay`, `phase_slope_index`, `group_delay_slope`.
<!-- policy-table:end -->

Two differences are deliberate and are kept: `imaginary_coherence` is 0 (to
within rounding) on the diagonal where the signed `imaginary_coherency` is NaN,
and coherence is NaN where its value would be 1. To treat every
self-connection the same way, mask it yourself; the cookbook's "Mask or fill the diagonal" recipe shows how.
