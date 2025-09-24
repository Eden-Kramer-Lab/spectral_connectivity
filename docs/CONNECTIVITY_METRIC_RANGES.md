# Connectivity Metric Ranges

The following summarizes the theoretical value ranges for common connectivity metrics.

| Metric | Range | Notes |
|---|---|---|
| **Coherency (complex)** | \|C_{xy}(f)\| ∈ [0, 1]; phase ∈ [−π, π] | Values lie in the unit disk of the complex plane. |
| **Coherence (\|Cxy\|²)** | [0, 1] | 0: none; 1: perfect linear dependence at frequency f. |
| **Imaginary coherency** | [−1, 1] (raw Im) | Magnitude version in [0, 1]. Reduces zero‑lag mixing. |
| **Partial coherence** | [0, 1] | Coherence conditioned on other channels. |
| **PLV (Phase‑Locking Value)** | [0, 1] | 0: random phases; 1: constant phase diff. |
| **PLI (Phase‑Lag Index)** | [0, 1] | Robust to zero‑lag mixing (sign consistency). |
| **wPLI (Weighted PLI)** | [0, 1] | Weights by \|Im(Sxy)\|. |
| **dwPLI² (Debiased squared wPLI)** | [0, 1] | Bias‑corrected (Vinck et al., 2011). |
| **PPC (Pairwise Phase Consistency)** | [0, 1] | Unbiased phase consistency; bounded as PLV. |
| **DTF (Directed Transfer Function)** | [0, 1] (normalized) | Proportion of inflow via transfer function. |
| **PDC (Partial Directed Coherence)** | [0, 1] (normalized) | Directional influence via AR coefficients. |
| **Spectral Granger causality** | [0, ∞) | Non‑negative; no finite upper bound. |
| **PSI (Phase Slope Index)** | (−∞, ∞) | Signed; magnitude depends on phase slope. |
| **AEC (Amplitude Envelope Correlation)** | [−1, 1] | Pearson correlation of envelopes. |
| **Orthogonalized AEC** | [−1, 1] | Same bounds; reduced leakage bias. |
| **Spectral Mutual Information** | [0, ∞) | Non‑negative; unbounded above. |
| **Spectral Transfer Entropy** | [0, ∞) | Non‑negative; estimator‑dependent scale. |

## Notes

- For metrics bounded in [0, 1], numerical implementations should clamp to the interval after computation to avoid tiny overflows (e.g., 1 ± 1e−12).
- For phase‑based metrics, choose an expectation (sample) axis (trials/tapers/segments) and aggregate **only** across that axis.
- Document the shapes/dtypes of inputs and outputs in each docstring per NumPy style.
- Also document that for granger causality output [i,j] (row i, column j) corresponds to j --> i.

## Implementation Details

### Bounded Metrics [0, 1]
- **coherence_magnitude**: Magnitude-squared coherence
- **phase_locking_value**: Phase consistency measure
- **phase_lag_index**: Signed phase consistency (unsigned version)
- **weighted_phase_lag_index**: Weighted by imaginary coherency magnitude
- **debiased_squared_weighted_phase_lag_index**: Bias-corrected wPLI²
- **pairwise_phase_consistency**: Unbiased phase consistency
- **directed_transfer_function**: Normalized directional influence
- **partial_directed_coherence**: Normalized causal influence
- **generalized_partial_directed_coherence**: Scaled by noise variance

### Bounded Metrics [−1, 1]
- **imaginary_coherence**: Raw imaginary part of normalized cross-spectrum

### Unbounded Metrics [0, ∞)
- **pairwise_spectral_granger_prediction**: Spectral Granger causality

### Unbounded Metrics (−∞, ∞)
- **phase_slope_index**: Directional phase coupling measure

### Complex-valued Metrics
- **coherency**: Complex coherence with magnitude ∈ [0, 1] and phase ∈ [−π, π]