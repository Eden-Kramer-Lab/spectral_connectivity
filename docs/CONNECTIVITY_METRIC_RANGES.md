# Connectivity Metric Ranges

The following summarizes the theoretical value ranges for common connectivity metrics.

| Metric | Range | Notes |
|---|---|---|
| **Power** | [0, ∞) | Power spectral density; always non-negative. |
| **Coherency (complex)** | \|C_{xy}(f)\| ∈ [0, 1]; phase ∈ [−π, π] | Values lie in the unit disk of the complex plane. |
| **Coherence phase** | [−π, π] | Phase angles of complex coherency. |
| **Coherence magnitude** | [0, 1] | 0: none; 1: perfect linear dependence at frequency f. |
| **Imaginary coherence** | [0, 1] (magnitude) | Raw imaginary part ranges [−1, 1]; magnitude version. |
| **Canonical coherence** | [0, 1] | Maximal coherence between groups. |
| **Global coherence** | [0, ∞) | Squared singular values; no upper bound. |
| **PLV (Phase‑Locking Value)** | [0, 1] | 0: random phases; 1: constant phase diff. |
| **PLI (Phase‑Lag Index)** | [−1, 1] | Signed version; unsigned in [0, 1]. |
| **wPLI (Weighted PLI)** | [0, 1] | Weights by \|Im(Sxy)\|. |
| **dwPLI (Debiased PLI²)** | [0, 1] | Bias‑corrected squared PLI. |
| **dwPLI² (Debiased squared wPLI)** | [0, 1] | Bias‑corrected weighted PLI squared. |
| **PPC (Pairwise Phase Consistency)** | [0, 1] | Unbiased phase consistency; bounded as PLV. |
| **Spectral Granger causality** | [0, ∞) | Non‑negative; no finite upper bound. |
| **DTF (Directed Transfer Function)** | [0, 1] (normalized) | Proportion of inflow via transfer function. |
| **Directed coherence** | [0, 1] (normalized) | DTF scaled by noise variance. |
| **PDC (Partial Directed Coherence)** | [0, 1] (normalized) | Directional influence via AR coefficients. |
| **gPDC (Generalized PDC)** | [0, 1] (normalized) | PDC scaled by noise variance. |
| **dDTF (Direct DTF)** | [0, 1] (normalized) | DTF with partial coherence normalization. |
| **Group delay** | (−∞, ∞) | Time delays; can be positive or negative. |
| **PSI (Phase Slope Index)** | (−∞, ∞) | Signed; magnitude depends on phase slope. |

## Notes

- For metrics bounded in [0, 1], numerical implementations should clamp to the interval after computation to avoid tiny overflows (e.g., 1 ± 1e−12).
- For phase‑based metrics, choose an expectation (sample) axis (trials/tapers/segments) and aggregate **only** across that axis.
- Document the shapes/dtypes of inputs and outputs in each docstring per NumPy style.
- Also document that for granger causality output [i,j] (row i, column j) corresponds to j --> i.

## Implementation Details

### Bounded Metrics [0, 1]

- **coherence_magnitude**: Magnitude-squared coherence
- **imaginary_coherence**: Magnitude of imaginary coherency
- **canonical_coherence**: Maximal coherence between groups
- **phase_locking_value**: Phase consistency measure
- **weighted_phase_lag_index**: Weighted by imaginary coherency magnitude
- **debiased_squared_phase_lag_index**: Bias-corrected squared PLI
- **debiased_squared_weighted_phase_lag_index**: Bias-corrected wPLI²
- **pairwise_phase_consistency**: Unbiased phase consistency
- **directed_transfer_function**: Normalized directional influence
- **directed_coherence**: DTF scaled by noise variance
- **partial_directed_coherence**: Normalized causal influence
- **generalized_partial_directed_coherence**: PDC scaled by noise variance
- **direct_directed_transfer_function**: DTF with partial coherence normalization

### Bounded Metrics [−1, 1]

- **phase_lag_index**: Signed phase consistency (unsigned version in [0, 1])

### Phase Metrics [−π, π]

- **coherence_phase**: Phase angles of complex coherency

### Unbounded Metrics [0, ∞)

- **power**: Power spectral density
- **global_coherence**: Squared singular values
- **pairwise_spectral_granger_prediction**: Spectral Granger causality

### Unbounded Metrics (−∞, ∞)

- **group_delay**: Time delays between signals
- **phase_slope_index**: Directional phase coupling measure

### Complex-valued Metrics

- **coherency**: Complex coherence with magnitude ∈ [0, 1] and phase ∈ [−π, π]
