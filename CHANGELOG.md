# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

This release includes corrected numerical definitions and therefore requires a
major version bump. Recompute affected results rather than comparing them
directly with results from 2.x.

### Migration guide

| Previous behavior | New behavior / required action |
| --- | --- |
| `global_coherence` returned raw squared singular values | Returns the scale-invariant fraction of total coherent power in `[0, 1]` |
| One-sided `power` omitted the negative-frequency contribution | Interior positive-frequency bins are doubled; DC and Nyquist are unchanged |
| `phase_slope_index` combined every ordered frequency pair | Uses adjacent frequency bins, following Nolte et al. (2008) |
| `delay` returned cycles | Returns seconds; DC is `NaN` |
| Multitaper windows were labeled by their first sample | Windows are labeled by their center time |
| `xarray.DataArray` axes followed NumPy's positional `(time[, trial], signal)` order | **Dimension names now define DataArray axis roles**, and inputs are transposed automatically; pass `time_dim`, `trial_dim`, and `signal_dim` for custom names |
| `multitaper_connectivity` labeled directed measures with `source`/`target` transposed (`sel(source=a, target=b)` gave `b -> a`) | `sel(source=a, target=b)` is now `a -> b` — recompute any directed results (e.g. `pairwise_spectral_granger_prediction`) obtained through the wrapper |
| `directed_coherence` broadcast the noise variance on the wrong axis (values could exceed 1) | Uses the correct source-axis noise variance and is bounded in `[0, 1]` — recompute directed-coherence results |
| `group_delay` / `delay` frequency-significance test over-rejected the null ~3–4× | Uses the exact zero-coherence null distribution; the set of "significant" frequencies changes — recompute (a dead-channel pair also no longer penalizes valid pairs in the BH/Bonferroni family) |
| `power_confidence_intervals` covered ~90% at a nominal 95%; `power_bias` / `power_variance` were ~2× off | Corrected formulas — recompute power confidence intervals and log-power z-tests |
| Spectral Granger measures returned `NaN` wherever the estimate was `<= 0` | Exact zeros and roundoff-negative values are returned as `0.0` (no influence); only materially negative, degenerate bins are `NaN`. Replace `np.isnan(...)` checks for "no influence" with `== 0`, and expect `nanmean` over a direction to include those zeros |
| `Connectivity(..., blocks=...)` | Remove `blocks`; memory is bounded automatically |
| `dpss_windows(..., interp_from=..., interp_kind=...)` | Remove both arguments; the exact SciPy solver is faster |
| `partial_directed_coherence(keep_cupy=...)` | Remove `keep_cupy`; public measures consistently return NumPy arrays |
| SciPy 1.10 / CuPy 12 GPU extra | Upgrade to `scipy>=1.11.1` and, for GPU use, `cupy-cuda12x>=13.0` |

### Added

- Spectral primitives and pairwise measures: one-sided
  `cross_spectral_density`, signed `imaginary_coherency`, `partial_coherence`,
  corrected imaginary PLV, and directed PLI.
- Multivariate/group measures: maximized imaginary coherency (MIC),
  multivariate interaction measure (MIM), conditional spectral Granger
  (the Chen, Bressler & Ding 2006 frequency decomposition, computed from one
  full-system and one reduced-system factorization per source), blockwise
  spectral Granger, and time-reversed spectral Granger.
- Exact complex `canonical_coherency` (Vidaurre CaCoh) performs phase
  optimisation and component deflation and returns component scores, spatial
  filters, patterns, connections, and group membership. The new
  `maximized_imaginary_coherency_components` exposes the same rich result schema
  for multiple MIC singular components; the historical score-only APIs remain
  available unchanged. Both are vectorized over the time/frequency axes on the
  active backend (GPU-capable), with the CaCoh phase search done as a batched
  grid-and-Newton optimization rather than a per-bin loop.
- `minimum_phase_reconstruction_error`: an opt-in diagnostic returning the
  relative reconstruction error of the Wilson factorization per sub-spectrum, so
  callers can check whether a cross-spectrum is resolved finely enough in
  frequency to trust the directed measures built on it (an under-resolved
  spectrum can converge yet reconstruct poorly).
- `ShortTimeFourierTransform`, `Welch`, and `MorletWavelet` transforms. Morlet
  output is explicitly one-sided and rejects Wilson-factorized directed
  measures, which require a full two-sided spectrum. Morlet coefficients are
  scaled so that `power()` is the one-sided PSD in signal²/Hz, on the same
  scale as `Multitaper` (FieldTrip's convention; MNE's wavelet power is
  `2 * sampling_frequency` times larger).
- Morlet transforms now expose strict time-frequency edge validity and support
  constant/reflect/edge convolution padding, keep/NaN/trim edge policies,
  adjacent-frequency smoothing, and boxcar or Hann time-frequency kernels.
  Connectivity expectations consume the local weights directly, and xarray
  results carry the `valid_time_frequency` mask.
- Multitaper `taper_weighting` supports historical uniform weighting,
  eigenvalue weighting, and Thomson adaptive frequency/signal-specific
  weighting. Adaptive weighting compares the periodogram against the process
  noise on a matched power-spectral-density scale and warns if the iteration
  does not converge.
- `fourier_connectivity` accepts externally computed Fourier coefficients as
  NumPy arrays or labeled DataArrays. Functional measures support one-sided
  coefficients (inferred from a labeled non-negative frequency coordinate or
  declared with `is_one_sided=True`), while directed factorization still
  requires a full two-sided spectrum. High-level results can be cropped,
  decimated, or reduced into named frequency bands; phase uses a circular mean
  and integration is restricted to spectral densities. The standalone
  `frequency_band_reduce(result, bands, reduction=...)` applies the same
  band reduction to an already-computed labeled result. Datasets holding
  spatial filters, patterns, or global-coherence vectors are rejected, since
  their sign and phase are arbitrary per frequency; reduce the score variable
  instead.
- `Connectivity.jackknife` and `jackknife_confidence_interval` provide
  leave-one-trial/taper bias correction, standard errors, and confidence
  intervals with automatic variance-stabilizing transformations: log for power,
  `atanh(sqrt(.))` for magnitude-squared coherence (`transformation=
  "fisher_squared"`), and circular for phase.
- Magnitude-normalized measures (coherency, phase-locking value, and everything
  derived from them) warn when computed from a single observation, where they
  are mathematically forced to 1; set `smoothing_time` on `MorletWavelet` or
  provide multiple trials/tapers. `Welch` warns when its default segment length
  yields coarse frequency resolution. `fourier_connectivity` rejects directed
  measures on unlabeled coefficients whose two-sidedness cannot be verified.
- `multitaper_connectivity` now accepts the directed-transfer-function family
  (`directed_transfer_function`, `directed_coherence`,
  `partial_directed_coherence`, `generalized_partial_directed_coherence`,
  `direct_directed_transfer_function`) by name. They are opt-in (not in the
  default set) and, like the spectral Granger measures, are oriented so
  `sel(source=a, target=b)` is the influence from `a` to `b`.
- `list_measures()` enumerates every wrapper-supported `method` name with its
  output category, a one-line description sourced from the `Connectivity`
  method docstring, and whether it is a default and/or directed measure;
  filter with `category`, `default_only`, and `directed`. An unknown `method`
  passed to `multitaper_connectivity` or `fourier_connectivity` now raises a
  `ValueError` that suggests the closest measure names and points to
  `list_measures()`, instead of a bare `AttributeError`.
- A doctested `docs/cookbook.md` collects short, copy-pasteable recipes for the
  most common tasks (functional and directed connectivity, reading the labeled
  output, frequency bands, and bringing your own Fourier coefficients).
- A multi-measure `multitaper_connectivity` `Dataset` now carries the shared
  provenance (package, version, backend, expectation type, and the `mt_*`
  multitaper parameters) as top-level `Dataset.attrs`, not only on each
  variable.
- The xarray interfaces now expose every built-in nonstandard result contract:
  group-pair matrices, exact CaCoh/rich MIC components with filters, patterns,
  and membership, delay candidates, global-coherence scores/vectors,
  group-delay quantities, and frequency-reduced phase-slope matrices.
- Wrapper results carry descriptive time/frequency metadata plus NetCDF-safe
  measure, package-version, backend, expectation, transform, and method-argument
  provenance. Scalar method arguments are stored under `arg_<key>`; structured
  or non-finite values are stored as canonical, JSON-normalized data under
  `arg_<key>_json` (and `None` serializes as `"null"`), with the full mapping in
  `measure_kwargs_json`. Non-string mapping keys use a collision-safe tagged
  item representation. Attributes on an input `xarray.DataArray` are carried
  through in one canonical `input_attrs_json` record, preserving arbitrary keys
  without collisions or invalid NetCDF attribute names.
- `multitaper_connectivity` accepts an `xarray.DataArray`, infers common
  time/trial/signal dimension names, and transposes them into the numerical
  core's order. **This changes the DataArray contract from position-driven to
  name-driven:** domain-specific names must be assigned with `time_dim`,
  `trial_dim`, and `signal_dim`, and ambiguous dimensions raise instead of
  falling back to axis position; when a single unrecognized dimension is
  assigned to the one remaining role by elimination, a warning names the assumed
  mapping. Numeric time coordinates are treated as
  elapsed seconds, numeric sample coordinates as sample numbers, and are used
  for output window times. `sampling_frequency` is now optional for a DataArray:
  when omitted it is inferred from a numeric elapsed-seconds `time` coordinate
  with sufficient numerical precision (a `sample` index cannot supply one);
  low-precision or large-offset coordinates require an explicit rate. When given,
  the rate is validated against the index. It remains required for array input.
  Datetime, timedelta, and object-valued time coordinates are not yet supported
  and must first be converted to numeric elapsed seconds.
  Unless `signal_names` is supplied explicitly, the signal index is preserved
  as the result's `source` / `target` coordinates without coercing its type.
  Missing, duplicate, nested, and structured signal labels are rejected;
  integer labels must fit the signed 32-bit range required for portable NetCDF3
  serialization. Labels that cannot be inferred still produce a warning.
  Dask-backed DataArrays are rejected with a materialization hint.
- `DEFAULT_METHODS` and `get_compute_backend` are exported at package level.
- Independent analytic-oracle, failure-mode, backend-boundary, serialization,
  minimum-dependency, artifact, doctest, and notebook checks cover the corrected
  behavior.

### Changed

- `canonical_coherency` fixes the sign of each spatial filter by the dominant
  coefficient of its pattern, so the canonical phase is no longer ambiguous by
  pi; single-channel groups reproduce the conjugate pairwise coherency.
- `phase_slope_index`, `delay`, and `group_delay` require a uniformly spaced
  frequency grid, and `delay`/`group_delay` reject non-uniform observation
  weights, because their adjacent-bin combinations and coherence null assume
  equal spacing and equally weighted observations (wavelet grids and Hann
  smoothing violate these silently).
- `Connectivity.jackknife` requires at least three observations, rejects
  diagnostics, alternate constructors, and structured (component) results with
  a clear error, and documents the supported explicit `fisher_squared`
  transformation.
- `frequency_band_reduce` treats a band containing any NaN bin as undefined for
  both reductions (previously the mean skipped NaN) and records per-band
  validity in a `valid_time_band` coordinate when the input carries
  `valid_time_frequency`.
- `fourier_connectivity` warns when neither a frequency coordinate nor
  `is_one_sided` is given and a two-sided spectrum is assumed, and rejects a
  lone negative-frequency bin because it cannot be a complete FFT spectrum.
- Group measures require at least two groups and a non-missing label per signal
  (`canonical_coherence` previously returned an empty array for one group), and
  `Multitaper` reports non-positive or non-finite sampling rates as
  "sampling_frequency must be finite and positive".
- `global_coherence`, `power`, `phase_slope_index`, `delay`, and
  `Multitaper.time` now follow the definitions in the migration table.
- Public connectivity methods reject single-signal inputs with an actionable
  error; `power` remains available for one signal.
- Bias-corrected phase measures require at least two observations.
- `Multitaper` parameters and owned input snapshots are immutable after
  construction. Array accessors return detached copies so cached calculations
  cannot become stale through external mutation.
- `multitaper_connectivity(method=None)` uses the stable, exported
  `DEFAULT_METHODS` allowlist. Measures with incompatible result shapes point
  users to `Connectivity` directly.

### Fixed

- Directed measures use at least complex128 working precision, so complex64
  inputs can satisfy the Wilson factorization's default tolerance.
- Subset spectral Granger restores `NaN` on the global self-Granger diagonal.
- Wilson initialization and solves isolate rank-deficient sub-spectra instead of
  poisoning a whole batch. Failed Cholesky units use the deterministic
  `n_signals * I` fallback; healthy units retain their Cholesky starts.
- Wilson convergence is relative and scale-invariant, and non-converged units
  return `NaN` with a targeted warning.
- Directed-measure regularization is scale-invariant, directed coherence uses
  the correct source-axis noise variance, and it warns when correlated
  innovations materially violate its diagonal-covariance assumption.
- GPU boundaries for `group_delay`, `delay`, statistics, metadata, and public
  returns now use explicit device-to-host conversion.
- `power` preserves float32 precision, and global coherence avoids overflow or
  underflow for extreme input scales.
- Phase-locking and phase-lag measures handle dead channels without leaking
  runtime warnings; their documented finite-sample ranges are corrected.
- Group delay and delay use the exact zero-coherence significance distribution.
  Undefined p-values are excluded consistently from BH and Bonferroni families.
- Statistical helpers validate observation counts, confidence levels, spectra,
  firing rates, p-values, and correction methods. Power confidence intervals
  and log-power bias/variance use the corrected formulas.
- Multitaper, DPSS, detrending, coordinate, FFT-size, window-size, frequency-step,
  and `fft_workers` inputs now fail early with parameter-specific errors.
- The wrapper accepts documented 2-D `(time, channels)` input and produces
  NetCDF-serializable metadata. Unsupported batch measures are skipped without
  swallowing genuine computation errors.
- The documentation build installs the current checkout on Read the Docs,
  confines generated sources to ignored directories, and the introductory
  tutorial no longer uses the removed `blocks` argument.
- The xarray wrapper now labels directed measures (e.g.
  `pairwise_spectral_granger_prediction`) so that `sel(source=a, target=b)` is
  the influence *from* `a` *to* `b`; previously the `source`/`target` axes were
  transposed, silently returning the reverse direction. Recompute any directed
  results obtained through `multitaper_connectivity`. The underlying
  `Connectivity` methods are unchanged (they keep the `output[i, j] = j -> i`
  convention).
- The wrapper rejects an empty `method` list, duplicate `signal_names`, and
  requests where no method yields a compatible result, instead of silently
  returning an empty or mislabeled `Dataset`.
- `multitaper_connectivity(..., squeeze=True)` keeps the selected `source` and
  `target` as scalar coordinates instead of dropping them, so the squeezed
  `(time, frequency)` result still records which pair (and, for directed
  measures, which direction) it represents. `squeeze=True` is now honored only
  for single-method (DataArray) requests -- because those scalar coordinates
  are Dataset-wide, applying them in a multi-measure Dataset would collide with
  a sibling `power` variable's `source` dimension -- and is ignored, with a
  warning, for multi-measure requests. It is a no-op for `power`.
- Importing the package no longer changes NumPy's global floating-point warning
  state, and backend reporting now reflects the backend actually imported.
- `simulate_MVAR` preserves the signal axis for single-signal, multi-trial
  simulations.

### Performance

- Reduced cross-spectral matrices use a batched matrix multiplication rather
  than materializing the observation-level signal-by-signal outer product.
- Phase locking uses unit-normalized coefficients with the same batched
  reduction. Phase-lag measures use bounded signal-row tiles and cache only
  reduced moments.
- Compact subset spectral-Granger factors only the requested 2-by-2 spectra.
- Global coherence uses chunked batched eigendecomposition/SVD for modest
  decomposition dimensions and retains a per-bin sparse fallback for large
  square problems. `max_workspace_elements` controls its working-set target.
- Group delay regression and significant-frequency cluster selection are
  vectorized and processed in bounded chunks.
- Wilson iteration reduces GPU synchronization, DPSS delegates to SciPy's
  compiled solver, detrending delegates to SciPy/CuPy, and sliding windows use
  `sliding_window_view`.
- Multi-measure wrapper calls build one `Connectivity` and reuse its cached
  power, cross-spectrum, directed factors, and phase-lag moments.
- `Connectivity.from_multitaper` adopts the transform's fresh FFT output
  without a redundant full-size copy. CPU FFT parallelism is available through
  the opt-in `fft_workers` argument.

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

[Unreleased]: https://github.com/Eden-Kramer-Lab/spectral_connectivity/compare/v2.0.1...HEAD
[2.0.1]: https://github.com/Eden-Kramer-Lab/spectral_connectivity/compare/v2.0.0...v2.0.1
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
