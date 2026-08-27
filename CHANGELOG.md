# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

> **Note on versioning:** this release corrects several measures whose numeric
> output changes (see **Changed — corrected numerics (BREAKING)** below). These
> are breaking changes and warrant a major version bump. Results computed for
> `global_coherence`, `power`, `phase_slope_index`, `delay`, and the time
> coordinate of multitaper/connectivity outputs will differ from earlier 2.x
> releases.

> **Upgrading — raised dependency floors:** this release requires `scipy>=1.11.1`
> (was 1.10). If you use the GPU backend, upgrade `cupy-cuda12x` to `>=13.0`
> (was 12.0) **before** upgrading `spectral_connectivity` — `pip install -U
> spectral_connectivity` will not force-upgrade an already-installed
> `cupy-cuda12x==12.x`, and the too-old CuPy is only detected at first GPU use
> (now with an explicit version error, no longer a misleading "CuPy is not
> installed" message). Upgrade with `pip install -U cupy-cuda12x`.

### Changed — corrected numerics (BREAKING)

- **`Connectivity.global_coherence`**: now returns each component's fraction of
  total coherent power (eigenvalue of the cross-spectral matrix over the sum of
  all eigenvalues, per Cimenser et al. 2011), bounded in `[0, 1]` and
  scale-invariant. It previously returned the raw squared singular values, which
  scaled with the square of the input amplitude (×10 input → ×100 output). The
  sparse (`svds`) branch is now sorted strongest-first explicitly (SciPy does
  not guarantee the order `svds` returns), matching the dense branch.
- **`Connectivity.power`**: now returns a true one-sided power spectral density —
  the interior positive-frequency bins are doubled (DC and, for even FFT length,
  Nyquist are not) so integrating over frequency recovers the full signal power.
  Previously the one-sided spectrum was half the correct magnitude. Connectivity
  measures are unaffected (they use the internal two-sided spectrum).
- **`Connectivity.phase_slope_index`**: now sums `conj(C(f)) * C(f + df)` over
  adjacent frequency bins (Nolte et al. 2008). It previously summed over all
  `i < j` frequency-pair combinations, a different statistic (both magnitude
  and, in general, sign differ).
- **`Connectivity.delay`**: now returns a time delay in seconds
  (`(phase + 2*pi*k) / (2*pi*f)`). It previously divided the phase by `2*pi`
  only, returning cycles, so a constant physical delay appeared to grow with
  frequency. The DC bin is returned as NaN. `group_delay` was already correct.
- **`Multitaper.time`**: windows are now labeled by their center time, as
  documented, instead of their first sample; the time coordinate of multitaper
  and connectivity results shifts later by ~half a window.

### Fixed

- **Wilson minimum-phase fallback is now deterministic (no global RNG state)**:
  when a sub-spectrum's zero-lag matrix is not positive-definite (rank-deficient
  / duplicated channels), `_get_initial_conditions` used to seed that unit's
  starting point from `np.random.standard_normal` — averaging 1000 random
  Wishart draws — so a pathological spectrum's factorization depended on
  unrelated random calls and required reseeding for reproducibility. It now uses
  the fixed positive-definite start `n_signals * I` (exactly the expectation of
  that average). Ordinary inputs whose zero-lag matrix is positive-definite never
  reach this fallback and are unchanged; the pathological fallback results are
  now deterministic instead of depending on the global NumPy random state
  (following the spirit of Scientific Python SPEC 7). This is a NumPy-backend
  concern only — CuPy's Cholesky returns NaN rather than raising, so the GPU path
  never enters this branch.
- **`statistics.Benjamini_Hochberg_procedure` — undefined tests no longer
  penalize the valid ones**: non-finite p-values (a coherence pair involving a
  dead/zero-power channel yields `NaN`) were counted toward the number of tests,
  which tightened the Benjamini-Hochberg threshold and made the *valid* pairs
  more conservative (fewer significant frequencies in `group_delay` / `delay`).
  They are now excluded from the family and returned as not-significant, so a
  dead channel no longer reduces the power of the healthy pairs. Results for
  finite p-values (the usual case, all channels live) are unchanged
  (bit-for-bit over 3000 randomized trials). The procedure now delegates to
  `scipy.stats.false_discovery_control`, which additionally rejects finite
  p-values outside `[0, 1]`. **Dependency floor raised:** `scipy>=1.11.1` (was
  `>=1.10`), the release that added `false_discovery_control`.
- **Wilson minimum-phase decomposition — one singular sub-spectrum no longer
  poisons the whole batch**: a single rank-deficient window (e.g. duplicated /
  linearly dependent channels) made the batched `linalg.solve` raise, which
  aborted the iteration and returned NaN for *every* sub-spectrum in the batch
  (and diverged from the GPU path, where CuPy returns NaN instead of raising).
  A new `_solve_isolating_singular` gives the NumPy path CuPy's semantics:
  singular/non-finite sub-matrices resolve to NaN while the rest are solved
  normally, so only the offending window is NaN, healthy windows still converge,
  the non-convergence warning reports the correct count, and CPU and GPU agree.
  `_get_initial_conditions` had the same all-or-nothing pattern — a single
  non-positive-definite window made the batched Cholesky raise and replaced
  *every* unit's deterministic Cholesky start with a random one (which could
  stop otherwise-convergent windows from converging). It now falls back to a
  random start only for the non-PD units, so the healthy units keep their exact
  Cholesky initialization.
- **`Connectivity.power` — preserves a float32 spectrum's dtype**: the one-sided
  doubling multiplied by a float64 scale array, silently upcasting a float32
  (complex64) spectrum back to float64 (doubling memory and defeating the
  precision choice). The scale now matches the spectrum's dtype.
- **Directed measures — scale-invariant regularization**: the Tikhonov diagonal
  loading used to stabilize the transfer-function inversions (`_MVAR_Fourier_coefficients`
  and `_estimate_transfer_function`) now scales with the RMS magnitude of the
  matrix (`λ = 1e-12 · √mean(|H|²)`) instead of its mean square. The old
  mean-square form gave `λ` amplitude-squared units while it is added to a
  matrix with amplitude units, so the effective regularization was not
  scale-invariant: rescaling the input by a large factor shifted spectral
  Granger and DTF by orders of magnitude (e.g. ×10¹² input changed DTF by
  thousands of percent). Results at ordinary scales are unchanged to ~1e-13.
- **`Connectivity.directed_coherence` — diagonal-covariance assumption**: this
  measure follows Baccalá et al. (1998), which assumes uncorrelated MVAR
  innovations (a diagonal noise covariance) so that the normalizing denominator
  equals the power spectral density `S_ii = Σ_k σ_kk|H_ik|²`. When the estimated
  innovation covariance has materially correlated off-diagonal terms (common for
  non-parametrically estimated MVARs), the true PSD `(H·Cov·Hᴴ)_ii` also contains
  cross-power that the diagonal formula omits. The assumption is now documented,
  and a `UserWarning` is emitted when the omitted cross-power is a material
  fraction of the true PSD — a dimension-aware criterion (max relative gap
  `|S_ii − Σ_k σ_kk|H_ik|²| / S_ii ≥ 10%`) rather than a pairwise-correlation
  threshold, so it also catches many weakly-but-jointly correlated sources whose
  cross terms still omit most of the power. The returned values are unchanged.
- **`Connectivity.power` / directed measures / statistics**: importing the
  package no longer executes a global `np.seterr(invalid="ignore")` that
  silenced NumPy invalid-operation warnings for all downstream caller code.
  Internal operations that legitimately produce NaN (the Granger log; the
  DTF/PDC inflow/outflow normalization when the Wilson decomposition does not
  converge) now scope the suppression locally instead.
- **`statistics.get_normal_distribution_p_values`**: uses `scipy.stats.norm.sf`
  instead of `1 - cdf`, so far-tail p-values keep full precision (e.g. `z=8.3`
  gives `~5.2e-17` rather than underflowing to `0`).
- **`wrapper.multitaper_connectivity`**: results are NetCDF-serializable again —
  only attributes of NetCDF-supported types are copied into the xarray `attrs`.
  Callable `Multitaper` members (e.g. the bound `summarize_parameters` method)
  are skipped and `None`-valued options (e.g. `detrend_type=None`) are encoded
  as the string `"None"`, both of which had made `to_netcdf` raise. The complex
  `coherency` measure is also excluded from the default `method=None` discovery
  (NetCDF cannot store complex arrays), so the documented default result is
  serializable; request `"coherency"` explicitly if you need it.
- **`Connectivity.phase_locking_value` / `pairwise_phase_consistency`**: a
  zero-magnitude cross-spectrum entry (dead/flat channel) is normalized to NaN
  via a guarded division with a `UserWarning`, instead of leaking a NumPy
  `RuntimeWarning` (now that the global warning suppression is removed).
- **`Connectivity.phase_slope_index`**: raises a clear error when fewer than two
  frequency bins remain in the band after subsampling, instead of returning an
  empty adjacent-product sum that NumPy reports as `0` (a false "no
  directionality" result).
- **`wrapper.multitaper_connectivity`**: accepts the documented
  `(n_times, n_channels)` 2-D input (promoted to a single-trial 3-D array)
  instead of forwarding it to `Multitaper` and raising.
- **`Connectivity`**: the constructor now creates the documented default
  coordinates (normalized frequencies, integer time indices) when they are
  omitted, so a directly-constructed instance no longer crashes in
  coordinate-dependent methods (`delay`, `group_delay`, `canonical_coherence`).
- **`minimum_phase_decomposition`**: Wilson-algorithm convergence uses a
  relative tolerance (normalized by the factor magnitude) so it is
  scale-invariant; an absolute tolerance was scale-dependent (accepting large
  relative reconstruction error for small-magnitude spectra). Because the
  relative criterion is stricter for typical data, the default `max_iterations`
  is raised to 500 (the loop still returns early on convergence, so this is
  cheap for well-conditioned inputs), and `minimum_phase_tolerance` /
  `minimum_phase_max_iterations` are now exposed on `Connectivity` /
  `Connectivity.from_multitaper` so callers can recover near-singular cases.
  A factor that becomes exactly singular during iteration (rank-deficient /
  duplicated channels) is now returned as NaN with the convergence warning
  instead of raising `LinAlgError`.
- **`transforms.dpss_windows`**: a singular pivot during inverse iteration no
  longer produces NaN tapers for valid parameters (e.g. `dpss_windows(8, 2, 3)`);
  the pivot magnitude is floored so non-degenerate results are unchanged.
  Invalid parameter combinations now raise: the window length must be `>= 2`
  (a single-sample window crashed inside `_fix_taper_sign`),
  `time_halfbandwidth_product` must satisfy `0 < NW < window_length / 2`
  (otherwise concentration ratios could exceed 1), and
  `1 <= n_tapers <= window_length`. A fractional `n_tapers` is rejected rather
  than silently truncated (`Multitaper` also validates it at construction, so
  the reported `n_tapers` cannot disagree with the taper count actually used).
- **`Connectivity`**: the constructor validates supplied `frequencies` / `time`
  coordinates — they must be exactly 1-D, finite, and match the data geometry.
  A mismatched length, a wrong shape (e.g. `(n, 1)`, which passed a length-only
  check), or a non-finite value previously misaligned/dropped bins silently, or
  crashed `delay` / `group_delay` / `canonical_coherence` with a broadcasting
  error.
- **`Connectivity.delay` / `group_delay` / `phase_slope_index`**: raise a clear
  error when the data has only one frequency bin, instead of a raw `IndexError`
  from reading `frequencies[1]` for the frequency step.
- **`Connectivity.delay` / `group_delay` / `phase_slope_index`**: a supplied
  `frequency_resolution` must be a finite positive number; zero, negative, NaN,
  and infinity are rejected (they caused step/slice errors or silent all-NaN
  results).
- **`statistics.coherence_significance_pvalue`**: `n_observations` must be a
  finite integer `>= 2` (non-finite or non-integer counts previously gave NaN or
  degenerate p-values).
- **`transforms.Multitaper`**: a `time_window_duration` / `time_window_step`
  that rounds/truncates to zero samples, or a window longer than the signal,
  now raises a clear error instead of a divide-by-zero or an empty transform.
  The validation runs on the resolved sample counts, so it also covers explicit
  `n_time_samples_per_window` / `n_time_samples_per_step` (including `0`), not
  just the duration-derived path.
- **`Connectivity.global_coherence`**: rescales the coefficients by their
  maximum magnitude before summing squares, so extreme magnitudes (e.g.
  `~1e-200` or `~1e200`) no longer underflow/overflow to NaN; the measure is a
  scale-invariant ratio, so the result is unchanged.
- **`statistics.coherence_significance_pvalue`**: requires `n_observations >= 2`
  (the Beta(1, n-1) null); smaller counts previously returned values outside
  `[0, 1]` (e.g. `1.33` for `n_observations=0`).
- **`Connectivity` phase-lag family** (`phase_lag_index`,
  `weighted_phase_lag_index`, `debiased_squared_phase_lag_index`,
  `debiased_squared_weighted_phase_lag_index`): raise a clear
  `NotImplementedError` under block mode (`blocks>=1`) instead of an
  `IndexError` — the Hermitian block assembly is incompatible with these
  measures' anti-symmetric transform and would otherwise return wrong-signed
  off-diagonals. The block accumulator also uses the active array namespace so
  it stays on-device under the CuPy backend.
- **`Connectivity.directed_coherence`**: corrected the noise-variance normalization. The per-source noise variance was broadcast on the target axis instead of the source axis and combined with a malformed `sqrt`/denominator, producing values greater than 1 for channels with unequal noise variances. It now returns the squared directed coherence `nv_j |H_ij|^2 / sum_k nv_k |H_ik|^2`, bounded in [0, 1] and summing to 1 over sources. **Values computed with earlier versions were incorrect whenever channel noise variances differed.**
- **`Connectivity.group_delay` / `Connectivity.delay`**: the frequency-significance test now uses the exact zero-coherence null distribution (`statistics.coherence_significance_pvalue`, magnitude-squared coherence ~ Beta(1, n-1)). The previous Fisher one-sample z-transform both returned all-NaN (so `group_delay` raised a `zero-size array` error) and, once that was patched, was badly miscalibrated at the zero-coherence boundary — it over-rejected the null by 3-4x (~16-22% actual rejection at a nominal 5%), yielding spurious "significant" frequencies. `coherence_fisher_z_transform` is retained for two-sample comparisons; it now validates that `n_obs1` is a finite integer `>= 2` and `n_obs2` is a finite integer equal to `0` (one-sample) or `>= 2` (`n_obs=1` previously raised `ZeroDivisionError`, and non-finite/fractional counts gave NaN with a runtime warning).
- **`statistics.power_confidence_intervals`**: split the tail mass evenly between the two tails. A requested 95% interval previously covered only ~90% (coverage was `2*ci - 1`). Added validation that `ci` is in `[0.5, 1.0)`, that `n_tapers` is a finite positive integer (zero/negative/NaN previously returned `(nan, nan)` and fractional values used meaningless non-integer degrees of freedom), and that `power` is finite and non-negative (a negative power previously returned negative, reversed bounds, e.g. `power=-1` -> ~`(-0.488, -3.080)`).
- **`statistics.power_bias` / `statistics.power_variance`**: evaluate the digamma/trigamma functions at the chi-squared shape parameter `n_observations` (`nu/2`) rather than `2*n_observations`, correcting an ~2x error in the log-power bias/variance used by `power_fisher_z_transform`.
- **`statistics.power_fisher_z_transform`**: the one-sample case (`n_obs2=0`) no longer hits the digamma/trigamma poles; the baseline default changed from 0 (`log(0)`) to 1.0. Observation counts must be finite integers and the spectra must be finite and strictly positive — non-finite/fractional counts and non-finite spectra previously produced silent NaN/Inf z-scores.
- **`transforms.Multitaper`**: raise a clear error when `n_fft_samples` is smaller than the window length instead of silently truncating the signal (the FFT crops rather than zero-pads).
- **`minimum_phase_decomposition`**: on Wilson-algorithm non-convergence, the unconverged sub-spectra are returned as NaN with a `UserWarning` instead of a silently partially-converged factor. Convergence is now tracked per independent sub-spectrum across all leading batch dimensions, so with expectation modes that retain a trial/taper axis a single failing sub-spectrum no longer NaNs the others at that time point.
- **`simulate.simulate_MVAR`**: fixed a crash for single-signal, multi-trial simulations (an unqualified `squeeze` collapsed the signal axis).
- **`utils.get_compute_backend`**: correctly reports the GPU backend (the previous `type(module)` check always reported CPU).
- Connectivity measures now raise an informative error for single-signal input instead of returning all-NaN, and `debiased_squared_phase_lag_index`, `debiased_squared_weighted_phase_lag_index`, and `pairwise_phase_consistency` raise when `n_observations < 2` instead of returning all-NaN / dividing by zero. `coherency` / `imaginary_coherence` return NaN (with a warning) for dead (zero-power) channels instead of dividing by a floored epsilon.
- **`statistics.get_normal_distribution_p_values`**: a genuine `TypeError` is now re-raised with its original traceback instead of being masked; the CuPy-to-host fallback only triggers for arrays that actually provide `.get()`.
- **`statistics.coherence_rate_adjustment`**: validates that **both** firing rates are finite and `> 0` (`firing_rate_condition2 <= 0` previously raised an unhandled `ZeroDivisionError`, and non-finite rates passed the old `<= 0` check) and returns NaN (with a single `UserWarning`) wherever the adjustment is undefined — non-positive spike power (invalid input, which could previously yield a finite adjustment `> 1`), a non-positive argument (`1 + adjusted_rate / spike_power_spectrum <= 0`), or zero spike power. The division and square root run under a scoped `errstate`, so a zero-power bin no longer leaks a `RuntimeWarning` and returns `0`, and an argument of exactly `0` no longer returns `inf`; both are now NaN as documented.
- **`Connectivity.global_coherence`**: clamps `max_rank` to the number of realizable components, `min(n_signals, n_trials * n_tapers)` (with a `UserWarning`). A one-trial/one-taper input previously crashed `svds` at the default `max_rank=1`, and an over-large `max_rank` could broadcast a single component into duplicate columns.
- **`minimum_phase_decomposition`**: rejects invalid convergence controls — a non-finite or non-positive `tolerance`, or a non-positive / non-integer `max_iterations` — with a clear `ValueError` instead of silently returning all-NaN with a misleading non-convergence warning.
- Corrected several docstring examples so their documented output matches (statistics `coherence_bias`, `coherence_rate_adjustment`, `get_normal_distribution_p_values`, `power_confidence_intervals`).
- Refreshed the notebook snapshot baselines (`tests/__snapshots__/test_notebooks.ambr`) for the corrected `power`, `group_delay` / `delay`, `phase_slope_index`, `global_coherence`, and directed-measure outputs; made the `simulate_MVAR`-based snapshot tests deterministic (they now pass `random_state`), fixed the `canonical_coherence` tests to the current group-label API, and skipped the conditional-Granger example (unimplemented). The `global_coherence` snapshot now records only the coherence fractions (the singular vectors have arbitrary sign/phase that is not stable across SciPy/BLAS versions). The custom snapshot extension now applies a true, array-wide numerical tolerance: it stores every full array (gzip-compressed float32) and compares every element with `np.allclose` (rtol=1e-6, atol=1e-9) in `matches`, so tiny floating-point differences (e.g. across BLAS/library versions) are tolerated while any real change is caught. Storing float32 (its ~1e-7 relative precision matches the tolerance) keeps the baseline small without weakening the check. Earlier iterations were unsound: the original `matches()` override was dead code (syrupy compares serialized strings, so comparisons were bit-exact); a significant-figure-rounding approach failed for values straddling a rounding boundary; a full-array (uncompressed) serialization bloated the baseline to ~40 MB; and a compact statistics-plus-samples fingerprint missed changes at unsampled positions or permutations of equal-magnitude values. CI now runs these snapshot tests (previously the whole file was excluded, hiding the stale baselines).
- Tooling: the mypy target is now Python 3.12 (matching the CI job) and the mypy floor is `>=1.11`, so mypy can parse NumPy 2.x's PEP 695 `type` statements in its stubs — the required type-checking gate previously failed to parse the stubs under a 3.10 target. Runtime compatibility with Python 3.10+ is still verified by the test matrix.

### Changed

- **`Connectivity.weighted_phase_lag_index`** is now documented as the *signed* index (range [-1, 1], like `phase_lag_index`); take the absolute value for the unsigned [0, 1] version. This is a documentation correction — the numeric output is unchanged.
- The `SPECTRAL_CONNECTIVITY_ENABLE_GPU` environment variable is parsed case-insensitively (`"true"`, `"1"`, `"yes"`, `"on"`) and warns on unrecognized values instead of silently falling back to CPU.
- Expensive directed-connectivity intermediates (minimum-phase factor, transfer function, noise covariance, MVAR coefficients) are cached per `Connectivity` instance and are automatically invalidated when `fourier_coefficients` or `expectation_type` is reassigned (they are now validated properties), so a reused instance cannot serve stale results. Reassigning `fourier_coefficients` with a different FFT-bin or time-window count now also resets the frequency/time coordinates to geometry-matching defaults (with a `UserWarning`), instead of leaving stale coordinates that silently dropped or misaligned bins in coordinate-dependent methods. Tikhonov regularization is scaled per (time-window, frequency) matrix rather than by a single global scalar.
- `multitaper_connectivity` gives an actionable error for `global_coherence` / `phase_slope_index` (which do not fit the xarray interface) instead of a cryptic xarray error.
- `multitaper_connectivity(method=None)` now selects an explicit default set of measures (the new `wrapper.DEFAULT_METHODS` allowlist) instead of discovering all public `Connectivity` methods and subtracting a denylist. The default set is unchanged (11 real-valued, NetCDF/xarray-compatible measures), but it is now stable and documented — a newly added `Connectivity` method can no longer silently join the default and break NetCDF serialization. The docstring no longer claims "all available methods": it names what the default excludes — `coherency` (complex; can still be requested by name), and `global_coherence` / `phase_slope_index` / `group_delay` / `canonical_coherence` / the directed-transfer-function family (which the wrapper does not support at all — request them via `Connectivity` directly). The default measure set and its variable order are unchanged (the allowlist is kept in the alphabetical order the previous discovery produced).
- **Docs**: `docs/CONNECTIVITY_METRIC_RANGES.md` now lists `global_coherence` as bounded in `[0, 1]` (fraction of total coherent power, Cimenser et al. 2011), matching the corrected measure — it previously described the old unbounded squared-singular-value output.
- `Connectivity` now stores its `fourier_coefficients` as a private immutable snapshot, so the per-instance caching cannot be silently corrupted by in-place edits (which would otherwise bypass cache invalidation and return stale power/coherence). The setter keeps a private copy that preserves the input's memory layout (`copy(order="K")`, so results are unchanged to the bit) and leaves the caller's original array untouched. The `fourier_coefficients` property never hands out a writable alias of that snapshot: it returns an independent, read-only copy. An in-place edit (`c.fourier_coefficients[...] = x`) therefore raises loudly instead of silently vanishing, and because the returned array is a copy (it owns its data) it is fully disconnected from the instance — re-enabling its `writeable` flag only affects the caller's throwaway copy. (A read-only *view* would not be safe: a caller could reach the owning base through `.base`, re-enable its `writeable` flag, and mutate the snapshot behind the caches.) On backends without a settable `writeable` flag (e.g. CuPy) the copy is returned writable but is still independent, so a write to it is harmless. Internal computations read the private snapshot directly, so the copy is paid only on explicit external access, not on the hot paths. To change the data, assign a new array to `fourier_coefficients`, which clears the caches, rather than editing in place.
- `Multitaper` now validates `fft_workers` at construction (must be `None` or a
  nonzero integer) and raises a message that names the parameter, instead of
  forwarding a bad value into an opaque SciPy `ValueError`/`TypeError`.
- `statistics.Benjamini_Hochberg_procedure` now emits a `UserWarning` when every
  p-value in the family is non-finite (the whole family is undefined, e.g. every
  tested pair involves a dead channel), so an all-not-significant result is not
  mistaken for "no true effects". It also raises a clear `RuntimeError` if the
  installed SciPy predates `false_discovery_control` (< 1.11), and its
  out-of-range error now reports how many values were out of range and their
  min/max.
- `transforms.detrend` now rejects a breakpoint at exactly the data length
  (validated against the documented `[0, N)` range) instead of silently
  collapsing it into an empty trailing segment.
- `multitaper_connectivity` now skips a measure that does not fit the xarray
  layout (`global_coherence`, `phase_slope_index`, the directed family, …) in a
  multi-measure call, logging the skip, instead of aborting the whole batch.
  These are flagged with a dedicated `wrapper.UnsupportedMeasureError` (a
  `ValueError` subclass), so a *genuine* computation error — e.g. a debiased
  measure raising `ValueError` because the data has too few observations — is
  **not** swallowed and still surfaces rather than silently dropping a requested
  measure.
- The `Connectivity`-sharing hook used by `multitaper_connectivity` is now an
  **internal, keyword-only `_connectivity` argument** to `connectivity_to_xarray`
  rather than a public `connectivity=` parameter. Tying results to a live,
  mutable `Multitaper` is a provenance footgun (the source's data or parameters
  can change after the `Connectivity` is built), so the injection is no longer
  part of the public API; reuse a transform by calling the `Connectivity`
  methods directly (the instance caches shared intermediates) or by requesting
  several measures from `multitaper_connectivity`. The internal hook is still
  validated defensively: only a `Connectivity.from_multitaper(m)` instance whose
  recorded source is that `Multitaper` (verified by identity, with the default
  `expectation_type`) is accepted, results are labeled from a parameter snapshot
  taken at build time (so a mutated source cannot mislabel them), and a mismatch
  raises an actionable error rather than a cryptic xarray dimension mismatch.
- `Connectivity` is now picklable and copyable when built via `from_multitaper`:
  the provenance weakref is dropped during serialization, and state is
  serialized as a `(__dict__, __slots__)` pair so attributes declared through a
  subclass' `__slots__` survive `pickle` / `copy.copy` / `copy.deepcopy` —
  including a string-form `__slots__` and name-mangled slot names. Legacy
  plain-dict pickles (written before this class defined `__getstate__`) are also
  accepted, with the new provenance fields initialized to `None`.

### Removed

- **`transforms.dpss_windows` no longer accepts `interp_from` / `interp_kind`.**
  The Slepian tapers are now solved directly with SciPy's LAPACK routine (see
  **Performance** below for the full rationale and equivalence evidence), so the
  public interpolation fast-path and its helper functions were removed. Code
  passing these keywords must drop them; the default exact computation is faster
  than the old interpolation path anyway. Relatedly, `dpss_windows(2, NW, 2)` (a
  two-sample window with two tapers) now raises a clear `ValueError` rather than
  returning degenerate tapers.

### Performance

- The expected cross-spectral matrix (used by coherence, spectral Granger, and
  most other measures) is now computed with a single batched matrix
  multiplication that contracts the averaged trial/taper/time axes directly,
  instead of materializing the full per-observation
  `(..., n_signals, n_signals)` outer product and then averaging. Results are
  unchanged to floating-point tolerance. On a representative case this was
  ~6× faster and cut peak memory ~9× (472 MB → 53 MB). As a consequence, the
  default computation now bypasses the `blocks` parameter entirely (it never
  forms the large intermediate that `blocks` was meant to chunk, so blocking it
  only added overhead). The coherence family and directed measures reduce the
  cross-spectral matrix directly and ignore `blocks`; the phase-lag-index family
  rejects it. `blocks` now affects only `phase_locking_value`, which applies a
  per-observation normalization before averaging and so must materialize the
  outer product.
- `multitaper_connectivity` now builds a single `Connectivity` from the
  multitaper transform and reuses it across every requested measure, instead of
  reconstructing one (and recomputing the uncached FFT) per measure. Results are
  bit-for-bit identical. The shared instance is passed via an internal,
  keyword-only `_connectivity` argument to `connectivity_to_xarray` (validated
  against the `Multitaper` to prevent mislabeled output); it is not part of the
  public API — reuse a transform yourself by calling `Connectivity` methods
  directly (the instance caches shared intermediates) or by requesting several
  measures from `multitaper_connectivity`. The FFT saving is modest on its
  own, because the tapers are already memoized on the `Multitaper` (so the
  dominant taper cost was not repeated); sharing one instance also lets the
  cached cross-spectrum and power (below) be reused across measures.
- `Connectivity._power` and the reduced expected cross-spectral matrix are now
  cached per instance (invalidated when `fourier_coefficients` or
  `expectation_type` is reassigned, alongside the existing directed-measure
  caches). `coherency` reads the power twice, and the coherence family plus
  pairwise spectral Granger each read the cross-spectrum, so caching avoids
  recomputing them — within a single measure and, for a shared instance, across
  measures. A repeated `coherency` on a warmed instance was ~60× faster in a
  representative case; the default multi-measure `multitaper_connectivity` call
  is correspondingly faster. Only the reduced `(..., n_signals, n_signals)`
  cross-spectrum is cached, never the large observation-resolved form.
- The phase-lag-index family (`phase_lag_index`, `weighted_phase_lag_index`,
  `debiased_squared_weighted_phase_lag_index`) now derives from shared reduced
  moments of the observation-level imaginary cross-spectrum (`E[sign(Im)]`,
  `E[Im]`, `E[|Im|]`, `E[Im**2]`), computed lazily per moment and cached, instead
  of re-forming that large intermediate once per transform function. Each moment
  is computed only when a measure needs it, from a single formation of the
  cross-spectrum, so a single-measure call does only its own reductions while a
  shared instance computing the whole family still avoids re-forming the
  cross-spectrum per transform function. Computing the family together was ~3.4×
  faster in a representative case; results are bit-for-bit identical. Only the
  reduced moments actually requested are cached (never the observation-level
  cross-spectrum), and they are invalidated with the other cached intermediates.
- `Connectivity.global_coherence` now computes its per-time-frequency-bin
  components with a single batched decomposition over all bins instead of a
  Python loop with one SVD per bin — a batched Hermitian eigendecomposition of
  the cross-spectral matrices (the measure's definition, Cimenser et al. 2011)
  when `n_estimates >= n_signals`, or the economy SVD of the thin coefficient
  matrix otherwise (which computes only the `n_estimates` non-trivial
  components). This was ~15× faster on a 1024-bin case and, on GPU, removes the
  two per-bin host syncs the old loop forced. Results match the previous per-bin
  path to floating-point tolerance (~5e-16 on the coherence fractions; the
  vectors, like singular vectors, are defined only up to a per-component phase
  for distinct components, and up to an arbitrary unitary rotation/permutation
  within any set of repeated/degenerate components, so they need not match the
  previous path). The batched path is used when the decomposition dimension
  `min(n_signals, n_estimates) <= 64` and is chunked over bins — from the
  original tensor, sized to the real per-bin working set — to bound peak memory;
  a per-bin `svds` fallback is retained for large square matrices, where
  computing every component would be wasteful.
- `Connectivity.group_delay` fits the unwrapped coherence phase against
  frequency with a vectorized masked regression (closed-form OLS slope and
  Pearson r from centered sums over the frequency axis) instead of
  `np.ma.apply_along_axis`, which called `scipy.stats.mstats.linregress` once per
  (time, signal pair). The regression kernel was ~500× faster in a
  representative case and matches the previous result to floating-point tolerance
  (~1e-15), with the same NaN for slices that have fewer than two significant
  frequencies. Masked (e.g. zero-power) frequencies are excluded rather than
  allowed to poison the sums, and the moments are computed from mean-centered
  residuals so the fit stays accurate when the absolute frequencies are large
  relative to their spacing. `scipy.stats.mstats.linregress` is no longer
  imported.
- Significant-frequency cluster selection (`_find_significant_frequencies`, used
  by `group_delay` and `delay`) now selects the largest independent significant
  cluster for all (time, signal-pair) slices with a vectorized pass instead of
  `np.apply_along_axis` (one Python call per slice). Results are bit-for-bit
  identical (it is boolean logic); the selection kernel was ~30× faster in a
  representative case, further speeding up a warmed `group_delay`. The slices are
  processed in bounded chunks (with `int32` run-length temporaries) so peak
  memory stays independent of their number.
- The Wilson minimum-phase iteration (used by the directed measures — spectral
  Granger, DTF, PDC and relatives) now synchronizes the device at most once per
  iteration instead of three times: the per-iteration convergence-count debug
  log is guarded behind the debug log level (so its reduction is not computed
  when debug logging is off), and the "all converged" and "all finished" early
  -exit tests are combined into a single reduction. Results are bit-for-bit
  identical; this is a GPU benefit (fewer host synchronizations) and a no-op on
  CPU.
- `transforms.dpss_windows` now computes the Slepian tapers via
  `scipy.signal.windows.dpss` (a LAPACK `eigh_tridiagonal` solve) instead of the
  package's own tridiagonal inverse-iteration solver. The eigenvalue problem and
  concentration-ratio formula are identical; SciPy solves it with compiled LAPACK
  rather than a Python-level iteration, so it is ~4.6–4.9× faster and removes a
  class of numerical edge cases (e.g. the singular-pivot NaN that previously
  required a guard). Tapers match the previous solver to ~1e-14 and eigenvalues
  to ~1e-15 (both up to the usual per-taper sign, which agrees here), and the
  independent `nitime` reference to floating-point tolerance. The validation,
  low-bias taper selection, and integer-`n_tapers` checks are unchanged. The now
  -unused custom solver helpers (`tridisolve`, `tridi_inverse_iteration`,
  `_find_tapers_from_optimization`, `_get_taper_eigenvalues`, `_fix_taper_sign`,
  `_auto_correlation`) and the interpolation fast-path (`interp_from` /
  `interp_kind`, `_find_tapers_from_interpolation`, `_interpolate_taper`) were
  removed; `dpss_windows` no longer accepts `interp_from` / `interp_kind`. A
  two-sample window with two tapers (`dpss_windows(2, NW, 2)`) now raises a clear
  `ValueError` — SciPy's antisymmetric-taper sign heuristic degenerates for a
  length-two window and raises a bare `IndexError` there (on both scipy 1.10.x
  and current releases), and such a window cannot support a second taper anyway.
- `transforms._sliding_window` now builds its windowed view with NumPy's
  `sliding_window_view` (subsampled to apply the step) instead of a hand-rolled
  `as_strided` call. NumPy recommends the higher-level helper: it validates
  bounds and shape rather than trusting manually computed strides. Output is
  bit-for-bit identical (verified on the multitaper transform and the window
  center-time coordinate); timings were neutral. With `is_copy=False` the view
  is now read-only (NumPy's default), which prevents the accidental in-place
  aliasing the previous writable view allowed; both production call sites copy.
  An out-of-range `axis` and a non-positive `step_size` now raise `ValueError`
  instead of silently windowing the wrong dimension or reversing the windows.
- `transforms.detrend` now delegates the actual mean/least-squares removal to
  `scipy.signal.detrend` (CPU) or `cupyx.scipy.signal.detrend` (GPU) instead of
  carrying a vendored copy of SciPy's algorithm, removing ~50 lines. The
  package's own `type`/`bp` validation and its actionable error messages are
  kept, so behavior — including the errors — is unchanged; CPU output is
  bit-for-bit identical (verified across constant/linear detrending, the
  `'l'`/`'c'` aliases, several axes, breakpoints, and int/float32 dtypes).
  **Dependency floor raised:** the GPU extra now requires `cupy-cuda12x>=13.0`
  (was `>=12.0`), because `cupyx.scipy.signal.detrend` was added in CuPy 13.
  GPU users on CuPy 12 must upgrade; CuPy 13 still targets CUDA 12.x.
- **`Connectivity.from_multitaper` adopts the transform without copying.** The
  immutable-snapshot setter copies the input Fourier coefficients (the largest
  array in the pipeline) to protect the caches from external in-place edits.
  `from_multitaper` passes the freshly built, unshared `Multitaper.fft()` output,
  which nothing else references, so it is now frozen in place instead of copied —
  removing a transient ~2x peak of that array during construction (which could
  push a memory-constrained GPU into OOM). Because `fft()` returns a `swapaxes`
  view whose base buffer is writable, the whole `.base` chain is frozen, so the
  read-only guarantee is unchanged; results are bit-for-bit identical. Direct
  construction (`Connectivity(fourier_coefficients=...)`) still copies
  defensively, since a caller-supplied array may be retained and mutated.
- **`Connectivity.global_coherence` gained a `max_workspace_elements` argument**
  bounding the batched decomposition's transient working set (default ~16M
  complex elements ≈ 256 MB, unchanged). Memory-constrained CPU/GPU callers can
  lower it to trade a little speed (more, smaller frequency chunks) for a smaller
  peak; the result is identical.

### Added

- **`multitaper_connectivity` / `connectivity_to_xarray` results are now
  self-describing.** The `time` and `frequency` coordinates carry CF-style
  `units` (`"s"` / `"Hz"`) and `long_name` attributes, so plotting libraries and
  NetCDF readers label axes correctly. Each result also records provenance
  attributes: `measure`, `package`, `package_version`, `backend` (`"CPU"` /
  `"GPU"`), and `expectation_type`, alongside the existing `mt_*` multitaper
  parameters and any measure keyword arguments (`arg_*`). All values are
  NetCDF-serializable and survive a `to_netcdf` round-trip.
- **`DEFAULT_METHODS`** — the default measure allowlist used by
  `multitaper_connectivity(method=None)` — is now exported from the top-level
  package (`from spectral_connectivity import DEFAULT_METHODS`), so callers can
  inspect or extend the default set without importing from the internal
  `wrapper` module.
- `Multitaper` gained an `fft_workers` argument (also accepted by
  `multitaper_connectivity` via `**kwargs`) that sets the number of parallel
  worker threads for SciPy's CPU FFT (`-1` uses all cores). It defaults to
  `None` (SciPy's single-threaded default), so existing behavior is unchanged;
  enabling it speeds up the FFT stage on multi-core CPUs (measured ~1.4× for the
  transform on an 18-core machine; larger on some SciPy builds). Results are
  numerically equivalent to the single-threaded result — a threaded FFT may
  differ at the floating-point-rounding level (~1e-16) from summation order, not
  bit-for-bit. It is opt-in to avoid oversubscribing CPUs when the analysis is
  already parallelized at a higher level, and is ignored on the GPU backend
  (whose FFT has no such parameter and is already parallel).

## [2.0.1] - 2026-05-12

### Fixed

- Broadcast identity matrix to batch dimensions of the LHS before calling `xp.linalg.solve` in `_estimate_transfer_function` and `_MVAR_Fourier_coefficients`, fixing CuPy batched-solve shape mismatch crashes that affected `directed_transfer_function`, `partial_directed_coherence`, `generalized_partial_directed_coherence`, and `direct_directed_transfer_function` on GPU.

## [2.0.0] - 2025-10-27

### BREAKING CHANGES

#### 3D Input Requirement for Multitaper Class

**BREAKING CHANGE**: `Multitaper` now requires 3D input arrays with explicit `(n_time_samples, n_trials, n_signals)` shape. Previously ambiguous 2D arrays now raise informative `ValueError`.

- **Migration Required**: Use the new `prepare_time_series()` helper function to convert 1D/2D arrays to 3D format
- **Why**: Eliminates dangerous ambiguity where `(n_time, n)` could mean either:
  - `(n_time, 1 trial, n signals)` OR
  - `(n_time, n trials, 1 signal)`

  Silent misinterpretation produces scientifically incorrect results.

**Migration Example**:
```python
# Before (ambiguous)
mt = Multitaper(eeg_data, sampling_frequency=1000)

# After (explicit)
from spectral_connectivity.transforms import prepare_time_series

eeg_3d = prepare_time_series(eeg_data, axis="signals")  # or axis='trials'
mt = Multitaper(eeg_3d, sampling_frequency=1000)
```

#### Nyquist Frequency Now Included for Even-Length FFTs

**BREAKING CHANGE**: Corrected frequency bin indexing to include Nyquist frequency for even-length FFTs

- **Impact on results**:
  - Even-length FFTs (N=1024): Now return 513 frequencies instead of 512 (adds Nyquist bin)
  - Odd-length FFTs (N=1023): No change (still return 512 frequencies)
- **Affected functions**: All connectivity measures using `@_non_negative_frequencies` decorator
- **Scientific justification**: The Nyquist frequency (sampling_rate/2) represents valid spectral information that should not be discarded
- See CHANGELOG "Unreleased" section for complete details and migration guidance

#### Development Tooling Changes

- **BREAKING**: Migrated from `black` to `ruff format` for code formatting
- **BREAKING**: Migrated from `flake8`/`pydocstyle` to `ruff` for linting
- Development workflows now use `ruff format` and `ruff check` commands
- All formatting/linting configurations moved to `pyproject.toml` under `[tool.ruff]`
- Developers must update their tooling: `pip install ruff` (100x faster than previous tools)

### Added
- `prepare_time_series()` helper function for safe dimension handling:
  - Converts 1D/2D arrays to required 3D format
  - Explicit `axis` parameter to clarify dimension meaning
  - Prevents ambiguous dimension interpretation
  - **Required for migration to v2.0.0**
- Comprehensive notebook snapshot testing with Syrupy:
  - 27 snapshot tests covering all tutorial notebook examples
  - Tests validate output shapes, data types, and numerical properties
  - Ensures tutorial examples remain accurate across releases
  - Automatically catches breaking changes in example code
  - Uses Syrupy's snapshot testing for efficient test maintenance
- **Test infrastructure** (`tests/conftest.py`):
  - Added pytest fixture to reset numpy random state before each test
  - Ensures consistent test reproducibility with fixed seed (42)
  - Helps prevent test ordering issues from shared random state
  - Auto-runs before every test without explicit fixture declaration
- **Physical constants with scientific rationale** (`spectral_connectivity/transforms.py`):
  - `MIN_EIGENVALUE_THRESHOLD = 0.9`: Minimum eigenvalue for low-bias tapers
    - Documents that 90% of taper energy must be in main lobe to reduce spectral leakage
    - Reference: Thomson (1982), "Spectrum estimation and harmonic analysis"
  - `TAPER_MULTIPLIER = 2.0`: Multiplier for calculating number of tapers
    - Documents the theoretical basis: ~2*NW orthogonal Slepian sequences are well-concentrated
    - Reference: Slepian (1978), "Prolate spheroidal wave functions"
  - All magic numbers replaced with named constants throughout codebase
  - Docstrings updated to reference constants while preserving readability
- Comprehensive test suite for advanced connectivity measures (`tests/test_advanced_connectivity.py`):
  - 18 test methods covering `canonical_coherence()`, `global_coherence()`, and `group_delay()`
  - Tests validate output shapes, value ranges, and mathematical properties (symmetry, antisymmetry)
  - Integration tests for complete Multitaper → Connectivity → advanced measures workflow
  - Tests handle edge cases: different group sizes, single signals, non-contiguous labels
  - Graceful handling of group_delay edge case when no frequencies are significant
- Parameter helper functions for guided multitaper analysis:
  - `estimate_frequency_resolution()`: Calculate frequency resolution from parameters
  - `estimate_n_tapers()`: Calculate number of tapers from time-halfbandwidth product
  - `suggest_parameters()`: Get parameter recommendations for your data
  - `MultitaperParameters` TypedDict for type-safe parameter handling
- `summarize_parameters()` method to `Multitaper` class:
  - Human-readable summary of all analysis parameters
  - Shows computed values (n_tapers, frequency_resolution, n_windows)
  - Displays overlap percentage for windowing
  - Formatted for terminal/notebook output
- Enhanced `time_halfbandwidth_product` docstring with formulas and practical guidance:
  - Mathematical relationship to frequency resolution and n_tapers
  - Typical values (NW=2,3,4,5+) with trade-off explanations
  - Examples for achieving target resolutions (1 Hz, 5 Hz, 10 Hz)
  - Cross-references to helper functions
- Comprehensive test suite for parameter helpers (`test_parameter_helpers.py`):
  - 22 tests covering all helper functions
  - Domain-specific tests (EEG, LFP typical parameters)
  - Edge cases (conflicting parameters, impossible resolutions)
  - Consistency checks with actual `Multitaper` behavior

### Changed
- **Complete type hint coverage** (`spectral_connectivity/transforms.py`, `spectral_connectivity/connectivity.py`):
  - Added type hints to ALL 28 previously untyped functions:
    - `transforms._make_tapers()`: Added parameter and return types
    - `connectivity.py`: 27 functions including all connectivity measures and helper functions
  - Used `TYPE_CHECKING` block to avoid circular imports for Multitaper type
  - Added `Literal` types for `multiple_comparisons_method` parameter
  - Fixed `Optional`/`None` handling in `_get_independent_frequency_step()` and `_bandpass()`
  - Fixed phase_lag_index return type (extract real part from complex result)
  - Enabled `disallow_untyped_defs = true` for transforms and connectivity modules
  - All 9 modules now have strict type checking enabled
  - 100% of public API is now fully type-annotated
- **Test coverage improvements**:
  - Overall coverage increased from 85% to **88%**
  - `connectivity.py` coverage improved from 71% to **93%**
  - Fixed snapshot tests after Nyquist frequency bin correction
  - Updated test assertions to use correct frequency bin calculation: `N//2 + 1` instead of `(N+1)//2`
- Improved method discovery in `multitaper_connectivity()` wrapper:
  - Replaced `dir()` with `inspect.getmembers(predicate=inspect.isfunction)` for type-safe method filtering
  - Automatically excludes properties and classmethods (more robust)
  - Renamed `bad_methods` to `excluded_methods` with clear categorization
  - Changed from list to set for O(1) membership testing
  - Added test `test_method_discovery_with_inspect()` to verify behavior
- Documented design decision for `adjust_for_multiple_comparisons()`:
  - Replaced TODO comment with clear explanation of current behavior
  - Explained why axis parameter is not implemented (standard approach treats all p-values as single family)
  - Left open for future enhancement if needed

### Removed
- All TODO comments from codebase (2 resolved)

### Changed
- **Development tooling modernization**:
  - Migrated from `black` to `ruff format` for code formatting (100x faster)
  - Migrated from `flake8`/`pydocstyle` to `ruff check` for linting
  - Updated GitHub Actions CI to use `ruff format --check` and `ruff check`
  - Updated [CLAUDE.md](CLAUDE.md) with new ruff commands
  - All formatting and linting now unified under single fast tool
- **Code quality improvements**:
  - Applied `ruff format` to all source files (10 files reformatted)
  - Fixed 19 auto-fixable ruff linting issues
  - All source code and tests now pass `ruff check` with zero warnings
  - Type hints improved with better union handling for `detrend()` function
- Updated tutorial notebooks to use `prepare_time_series()` for 3D input
- Improved random number generation in tests for better isolation

### Fixed
- **MyPy type annotation error** in `detrend()` function (`transforms.py:1867-1876`):
  - Fixed union type handling for `bp` parameter (now `int | list[int] | NDArray[np.integer]`)
  - Added support for list input in addition to int and ndarray
  - Eliminated unreachable code warnings from MyPy
  - All 9 source files now pass strict mypy type checking

#### **Nyquist Frequency Fix (from earlier release)**

- **Critical bug fix**: Corrected frequency bin indexing to include Nyquist frequency for even-length FFTs
  - **Affected code**: Changed frequency indexing from `(N+1)//2` to `N//2 + 1` in three locations:
    - `_non_negative_frequencies` decorator (line 107)
    - `canonical_coherence` method (line 638)
    - `_estimate_spectral_granger_prediction` function (line 2108)
  - **Impact on results**:
    - Even-length FFTs (N=1024): Now return 513 frequencies instead of 512 (adds Nyquist bin)
    - Odd-length FFTs (N=1023): No change (still return 512 frequencies)
  - **Affected functions**: All connectivity measures using `@_non_negative_frequencies` decorator:
    - `coherency()`, `coherence_magnitude()`, `coherence_phase()`, `imaginary_coherence()`
    - `phase_lag_index()`, `weighted_phase_lag_index()`, `debiased_squared_phase_lag_index()`
    - `debiased_squared_weighted_phase_lag_index()`, `phase_locking_value()`, `pairwise_phase_consistency()`
    - `power()`, all Granger causality measures (DTF, PDC, etc.)
  - **Scientific justification**: The Nyquist frequency (sampling_rate/2) represents the highest frequency
    that can be unambiguously represented in sampled data. For even-length FFTs, this frequency should be
    included once (not in negative frequencies). Excluding it discards valid spectral information and
    violates standard FFT conventions (numpy.fft.rfft, scipy.fft.rfft).
  - **Migration impact**:
    - New analyses: More accurate (includes previously missing frequency information)
    - Comparing to old results: Array shapes differ by 1 in frequency dimension for even-length FFTs
    - Published results: Document which version was used in methods section
  - **Tests added**:
    - `test_nyquist_bin_even_n()`: Validates N=1024 produces 513 frequencies
    - `test_nyquist_bin_odd_n()`: Validates N=1023 produces 512 frequencies
  - **Example**: With 1000 Hz sampling and 1024-sample FFT:
    - Old (incorrect): 512 bins, missing 500 Hz (Nyquist)
    - New (correct): 513 bins, includes 500 Hz (Nyquist)
  - See PR #71 for detailed analysis and discussion

- **Tikhonov regularization for MVAR matrix inversion stability**:
  - Replaced direct matrix inverse (`xp.linalg.inv()`) with Tikhonov-regularized solve in MVAR computations
  - Prevents `LinAlgError` exceptions when computing Granger causality with near-singular matrices
  - Affected functions: `_MVAR_Fourier_coefficients` property and `_estimate_transfer_function` function
  - Uses scale-aware regularization: λ = `TIKHONOV_REGULARIZATION_FACTOR` × mean(||H||²)
  - Added module-level constant `TIKHONOV_REGULARIZATION_FACTOR = 1e-12` for consistency
  - Solves `(H + λI)x = I` instead of computing `inv(H)` for better numerical stability
  - Added stress test `test_mvar_regularized_inverse_near_singular()` validating near-singular cases
  - All Granger causality measures now handle highly correlated signals gracefully
  - See PR #72 for detailed numerical analysis

- **Numerical stability improvements for coherence calculations**:
  - Replaced zero-clamping with epsilon-clamping in `coherency()` to prevent division by zero
  - Changed from `norm[norm == 0] = xp.nan` to `norm = xp.maximum(norm, xp.finfo(norm.dtype).eps)`
  - Added bounds clipping to `coherence_magnitude()` to ensure values stay in [0, 1] range
  - Added bounds clipping to `imaginary_coherence()` with epsilon protection
  - Prevents numerical artifacts from floating-point precision issues
  - More graceful degradation for low-power signals compared to NaN propagation
  - Added comprehensive test suite (`test_coherence_bounds.py`) with 3 tests covering edge cases
  - See PR #62 for detailed analysis

- CHANGELOG.md to track version changes following Keep a Changelog format
- Ruff linter configuration for faster, more comprehensive Python linting
- Enhanced package metadata with additional project URLs (Changelog, Source Code, Issue Tracker)
- Modern unified CI/CD workflow (`release.yml`) with automated PyPI publishing
- Support for Python 3.13
- Comprehensive parameter validation to `Multitaper` class:
  - Validates `sampling_frequency > 0` with domain-specific examples (EEG, LFP, fMRI)
  - Validates `time_halfbandwidth_product >= 1` with physical meaning explanation
  - Validates `time_window_duration > 0` (when provided) with frequency resolution formula
  - Validates `time_window_step > 0` (when provided) with overlap guidance
  - Warns when `time_halfbandwidth_product > 10` (unusually large, performance impact)
  - Warns when `time_window_step > time_window_duration` (creates data gaps)
  - Warns when data appears transposed (`n_time < n_signals`)
  - Warns when input contains NaN or Inf values with recovery suggestions
- Input shape validation to `Connectivity` class:
  - Requires 5D `fourier_coefficients` with clear error messages
  - Validates minimum 2 signals for connectivity analysis
  - Warns on NaN/Inf values in Fourier coefficients
- `prepare_time_series()` helper function for safe dimension handling:
  - Converts 1D/2D arrays to required 3D format
  - Explicit `axis` parameter to clarify dimension meaning
  - Prevents ambiguous dimension interpretation
- Enhanced error messages following WHAT/WHY/HOW pattern throughout
- 3D input requirement for `Multitaper` class to eliminate dimension ambiguity
- Intelligent error suggestion for `expectation_type` parameter:
  - Detects wrong word order (e.g., "tapers_trials" instead of "trials_tapers")
  - Suggests correct ordering with helpful explanation
  - Lists all valid options with most common choice highlighted
- Improved `detrend()` function error messages:
  - Clear explanation of linear vs constant detrending
  - Examples with domain-specific terminology (DC offset, best-fit line)
  - Actionable guidance for parameter selection
- Enhanced breakpoint validation in `detrend()`:
  - Shows specific invalid breakpoint values
  - Displays valid range based on actual data dimensions
  - Includes user's original input for easy debugging
- Comprehensive test suite for error message quality (`test_error_messages.py`)
- GPU status utility function `get_compute_backend()`:
  - Returns dict with backend type, GPU availability, device name, and helpful message
  - Shows actual GPU model name (e.g., "NVIDIA Tesla V100-SXM2-16GB") instead of compute capability
- Comprehensive test suite for block-wise computation:
  - 5 new tests validating correctness, symmetry, edge cases, and memory reduction
  - Empirical memory measurement showing 73% reduction for n_signals=50, blocks=5
  - Tests cover different expectation types and block configurations
- Enhanced documentation for `blocks` parameter in Connectivity class:
  - "When to use" and "When NOT to use" guidance with specific thresholds
  - Quick decision guide based on n_signals
  - Memory-speed tradeoff quantified (70-80% memory reduction, <10% speed penalty)
  - GPU VRAM considerations documented
  - Detects CuPy availability without side effects (uses `importlib.util.find_spec`)
  - Provides 4 different message variants for different GPU configurations
  - Includes comprehensive NumPy-style docstring with 3 usage examples
  - Example return value documented in docstring
  - Located in new `spectral_connectivity.utils` module
- Enhanced GPU device logging in `transforms.py` and `connectivity.py`:
  - Now shows actual GPU model name in log messages
  - Graceful fallback to compute capability if name unavailable
- Comprehensive GPU documentation in README.md:
  - 130+ line GPU Acceleration section with setup, troubleshooting, and usage guidance
  - 3 setup methods documented (shell, Python script, Jupyter notebook)
  - Verification steps included in all setup examples
  - Simplified CuPy installation instructions (conda recommended first)
  - Troubleshooting guide with 4 common issues and solutions
  - Clear explanation of import timing requirement (why "before importing" matters)
  - Example outputs shown for all code samples
  - Kernel restart guidance for Jupyter notebook users
  - Guidance on when GPU acceleration is beneficial
- Comprehensive test suite for GPU backend (`tests/test_gpu.py`):
  - 13 test methods covering all GPU configuration scenarios
  - Tests for CPU mode (default and explicit)
  - Tests for GPU mode (with and without CuPy available)
  - Validation of return value structure and types
  - Mock-based testing to avoid CuPy dependency
  - All tests pass (11 passed, 1 skipped when CuPy unavailable)

### Changed
- **BREAKING**: Minimum Python version raised from 3.9 to 3.10
- Migrated from flake8 to ruff for linting (100x faster, replaces flake8, isort, pydocstyle)
- Updated dependency pins: numpy>=1.24, scipy>=1.10, xarray>=2023.1, matplotlib>=3.7
- Improved mypy configuration with stricter type checking and per-module overrides
- Updated development documentation (CLAUDE.md, CONTRIBUTING.md) to reflect current tooling
- Expanded test matrix to Python 3.10, 3.11, 3.12, 3.13
- Consolidated CI workflows: removed redundant PR-test.yml and linting.yml in favor of release.yml
- Simplified CI from conda-based to pip-based installation (faster builds)
- Enhanced black configuration to target Python 3.10-3.13
- Updated ReadTheDocs to use Python 3.10

### Fixed
- Outdated release instructions in CONTRIBUTING.md (removed setup.py references)
- Deprecation warning in `minimum_phase_decomposition.py`: Changed `xp.linalg.linalg.LinAlgError` to `xp.linalg.LinAlgError` for compatibility with NumPy 2.0+
- **Critical bug in block-wise computation**: Fixed missing diagonal elements in cross-spectral matrix when using `blocks` parameter (changed `k=1` to `k=0` in `triu_indices`)
- **Critical bug in block-wise computation**: Fixed incorrect Hermitian symmetry in blocked cross-spectral matrix computation (added conjugate when filling transpose positions)

## [1.1.2] - 2023-10-17

### Added
- Conda packaging support with conda-recipe directory
- CLAUDE.md with development commands and architecture documentation
- Pinned coverage reporter version in CI workflow to avoid bugs

### Changed
- Updated module docstrings for clarity and context
- Updated README with Contributing and License sections

### Fixed
- Linting issues resolved

## [1.1.1] - 2023-09-15

### Changed
- Switch build system from setuptools to Hatch
- Add py.typed marker for type hint support
- Update and reorganize dependencies in environment.yml

### Fixed
- Resolve n_time_samples_per_window property logic error in transforms module
- Resolve mypy Optional[int] vs int return type errors
- Correct _fix_taper_sign return type annotation

## [1.1.0] - 2023-08-20

### Added
- GPU request guard feature to safely handle CUDA availability
- Complete audit of connectivity metric ranges documentation
- ValueError raised when window size parameters are unset

### Changed
- Updated GitHub Actions to latest versions
- Improved type hints throughout codebase

## [1.0.4] - 2023-03-15

### Fixed
- Bug fixes in connectivity calculations
- Improved numerical stability

## [1.0.3] - 2023-02-10

### Changed
- Performance improvements
- Documentation updates

## [1.0.2] - 2023-01-20

### Fixed
- Minor bug fixes
- Test coverage improvements

## [1.0.1] - 2022-12-15

### Fixed
- Package distribution fixes
- Documentation corrections

## [1.0.0] - 2022-12-01

### Added
- First stable release
- Full implementation of 15+ connectivity measures
- GPU acceleration support via CuPy
- Comprehensive test suite
- Complete documentation on ReadTheDocs

### Changed
- API stabilized for 1.0 release
- Performance optimizations

## [0.2.7] - 2022-06-15

### Added
- Additional connectivity measures
- Improved caching strategy

### Changed
- API improvements and refinements

## [0.2.6] - 2022-03-10

### Added
- Initial GPU support
- More connectivity measures

### Changed
- Refactored core architecture
- Improved documentation

---

[Unreleased]: https://github.com/Eden-Kramer-Lab/spectral_connectivity/compare/v2.0.0...HEAD
[2.0.0]: https://github.com/Eden-Kramer-Lab/spectral_connectivity/compare/v1.1.2...v2.0.0
[1.1.2]: https://github.com/Eden-Kramer-Lab/spectral_connectivity/compare/v1.1.1...v1.1.2
[1.1.1]: https://github.com/Eden-Kramer-Lab/spectral_connectivity/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/Eden-Kramer-Lab/spectral_connectivity/compare/v1.0.4...v1.1.0
[1.0.4]: https://github.com/Eden-Kramer-Lab/spectral_connectivity/compare/v1.0.3...v1.0.4
[1.0.3]: https://github.com/Eden-Kramer-Lab/spectral_connectivity/compare/v1.0.2...v1.0.3
[1.0.2]: https://github.com/Eden-Kramer-Lab/spectral_connectivity/compare/v1.0.1...v1.0.2
[1.0.1]: https://github.com/Eden-Kramer-Lab/spectral_connectivity/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/Eden-Kramer-Lab/spectral_connectivity/compare/v0.2.7...v1.0.0
[0.2.7]: https://github.com/Eden-Kramer-Lab/spectral_connectivity/compare/v0.2.6...v0.2.7
[0.2.6]: https://github.com/Eden-Kramer-Lab/spectral_connectivity/releases/tag/v0.2.6
