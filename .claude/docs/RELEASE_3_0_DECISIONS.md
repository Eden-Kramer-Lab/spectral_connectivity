# 3.0 API and result consistency decisions

**Status: step 1 merged; step 2 prepared for implementation.**
[PR #102](https://github.com/Eden-Kramer-Lab/spectral_connectivity/pull/102)
merged into `master` as `138f3f7` on 2026-10-05. Step 2 starts from that commit
on `refactor/metadata-and-bandwidths`; its implementation checklist is below.
The accepted naming decision keeps Multitaper's `frequency_resolution` and
its existing helper names supported, with precise definitions and no synonyms.
The subsequent UX review also retains released function spellings, `ci`,
`global_coherence(max_rank=...)`, and positional minimum-phase tolerances.
Common wrapper settings remain explicit, flat keywords.
The original proposal was reviewed against `master` at `0682eae` and the released
`v2.0.1` tag on 2026-09-27.

**Project planning document. Records accepted decisions and implementation status.**

## Recommendation

Use 3.0 to establish a clear result schema, explicit measure names, and consistent
frequency selection. Preserve differences that reflect different estimators.
Keep compatibility paths for existing calls where they prevent silent changes to
an analysis.

The decisions with the greatest effect on users are **D1–D8, D12, D14, and
D23–D27**. The 31 decisions cover the original audit and the nine additional
items identified in review. “Later” means it should not hold up 3.0.

## Released compatibility boundary

Checked the source at `v2.0.1` using `git show`, including the wrapper's dynamic
attribute export. Compatibility is based on that release, not on intermediate
commits preparing 3.0.

| Released in v2.0.1 | New in 3.0 |
| --- | --- |
| `Multitaper`, its `frequency_resolution` property, and its exported `mt_*` metadata | STFT, Welch, Morlet, and their result metadata |
| `multitaper_connectivity`, `connectivity_to_xarray`, and power's `source` dimension | `fourier_connectivity`, `frequency_band_reduce`, and component-result schemas |
| `global_coherence(max_rank=)`, `from_multitaper`, and the original canonical-coherence estimator | MIC/MIM, phase-optimized CaCoh, and `from_transform` |
| `suggest_parameters`, `estimate_frequency_resolution`, and their existing arguments and `frequency_resolution` return key | Explicit estimator-bandwidth definitions in result metadata |
| `minimum_phase_decomposition` with positional tolerances, and `simulate_MVAR` | Reconstruction diagnostics and the two other simulators |

Rename unreleased interfaces directly. Add compatibility aliases only for
released interfaces, and remove those aliases in 4.0 under the policy below.

## Corrections to the audit's framing

- **Resolution is not one interchangeable number.** Multitaper's current value
  describes its concentration bandwidth; STFT's describes equivalent noise
  bandwidth. Their numerical difference is expected. Retain Multitaper's
  familiar released name, define its full DPSS concentration bandwidth clearly,
  and qualify the result metadata to distinguish estimator bandwidth from FFT
  bin spacing. Each quantity has one supported public name.
- **Different diagonals are not automatically bugs.** For a signal with nonzero
  spectral power, coherence with itself is mathematically 1; this package
  chooses to mask it with NaN. Other diagonals are algebraic identities or
  meaningful self-transfer values. A package mask and an undefined estimator
  must be described separately.
- **Unknown units should stay unknown.** Setting power's units to `"1"` would
  incorrectly describe it as dimensionless. Correct the documentation promise.
- **Morlet retains coordinate values.** The reproduced defect is loss of the
  `time` and `frequency` coordinate attributes, including units and descriptions.
- **Compatible additions can ship in 3.x.** New names, metadata, and helpers do
  not all require waiting until 4.0. Removing supported names or changing their
  contracts does. This distinction follows [Semantic Versioning](https://semver.org/).

## A. Result metadata and scientific meaning

| ID | Topic | Recommended choice | Compatibility and timing |
| --- | --- | --- | --- |
| D1 | Shared transform attributes | Use common keys for shared facts: `transform`, `sampling_frequency`, `n_trials`, `n_signals`, `n_observations`, `expectation_type`, `observations_are_independent`, and `time_bins_are_independent`. Include only facts that are known. Add `output_schema_version=1`. | Establish in 3.0. Preserve only the scalar `mt_*` compatibility keys that actually shipped in v2.0.1, until 4.0. Drop unreleased `stft_*`, `welch_*`, `morlet_*`, and `fourier_*` copies now. |
| D2 | Frequency resolution | Keep `Multitaper.frequency_resolution` as the single supported public name and define it as full DPSS concentration bandwidth (`2 * NW / time_window_duration`, in Hz). Use STFT/Welch `equivalent_noise_bandwidth` and `frequency_bin_spacing` for a uniform output grid. Results record a qualified `spectral_bandwidth` with `spectral_bandwidth_definition`; frequency-dependent bandwidth belongs on a frequency coordinate. | Retain Multitaper's released property and value without deprecation or a `concentration_bandwidth` synonym. Rename STFT's unreleased `frequency_resolution` directly, with no compatibility alias, and prevent inheritance of Multitaper's property with a different meaning. |
| D3 | Transform and execution settings | Keep D1's common facts at the top level and one `transform_parameters_json` record for remaining settings. Inside that record, separate `estimator` settings from `execution` settings such as `fft_workers`. Keep `backend="cpu"` or `"gpu"` at the top level. | Establish in 3.0. Do not repeat common facts inside JSON or add a second execution JSON record. The only extra compatibility copies are the released `mt_*` keys, removed in 4.0. |
| D4 | Frequency selection | Use `frequency_range=(low, high)` for a single selected interval, with both endpoints included. Retain `frequency_bands={name: (low, high)}` for named aggregation and `frequency_reduction` for mean/integral. Use one selection helper. | Prefer the canonical interface in 3.0. Keep `frequencies_of_interest` as a deprecated compatibility keyword with its historical exclusive edges. Reject calls supplying both spellings. See the routing rules below. |
| D5 | Diagonal values | Preserve and document current numerical diagonals. Describe coherence's NaN as an intentional self-connection mask despite mathematical self-coherence being 1 at nonzero power. Explicitly document `imaginary_coherence`'s 0 versus signed `imaginary_coherency`'s NaN; retain that difference. Add `diagonal_policy` metadata and an explicit masking recipe. | Documentation and metadata in 3.0. Choose documentation over changing values within the imaginary-coherence family. A convenience masking helper can follow. |
| D6 | Squared coherence | Make `magnitude_squared_coherence` the canonical name for the current `coherence_magnitude` calculation. Preserve `coherency` for the complex value. Keep `imaginary_coherence` and signed `imaginary_coherency`, with explicit formulas and magnitude/squared metadata. | Canonical name and default variable change in 3.0; add the migration row below. Old methods and method strings warn, preserve numerical results and explicitly requested variable names through 3.x, and are removed in 4.0. |
| D7 | Canonical coherence estimators | Keep `canonical_coherence` and `canonical_coherency` distinct. State the historical squared canonical-correlation definition and the phase-optimized CaCoh definition in their metadata and examples. | Clarify in 3.0. Do not implement one as the square or square root of the other. |
| D8 | Source terminology | Use `source`/`target` throughout labeled results: `connection_source_group`, `connection_target_group`, and `side=["source", "target"]`. | Rename these new 3.0 component-result coordinates before release; update selectors, examples, and serialization tests together. |

### Rules for the shared metadata

- Canonical metadata has two locations: common facts at the top level and one
  `transform_parameters_json` record for remaining settings. An example record is
  `{"estimator":{"time_halfbandwidth_product":3},"execution":{"fft_workers":4}}`.
  Do not also put `sampling_frequency`, counts, or `backend` in that record.
- The v2.0.1 scalar compatibility whitelist is `mt_detrend_type`,
  `mt_is_low_bias`, `mt_sampling_frequency`, `mt_start_time`,
  `mt_time_halfbandwidth_product`, `mt_n_fft_samples`, `mt_n_signals`,
  `mt_n_tapers`, `mt_n_time_samples_per_step`, `mt_n_time_samples_per_window`,
  `mt_n_trials`, `mt_time_window_duration`, `mt_time_window_step`,
  `mt_frequency_resolution`, and `mt_nyquist_frequency`. The old wrapper also
  attempted to export callable attributes; do not revive those invalid metadata
  entries. New options such as `fft_workers` and adaptive weighting do not get
  `mt_*` compatibility copies. Apply this whitelist only to Multitaper results.
- `transform` uses `"multitaper"`, `"stft"`, `"welch"`, `"morlet"`, or
  `"external_fourier"` for the built-in paths.
- `sampling_frequency` is in Hz when known. External Fourier coefficients do not
  always establish a physical sampling rate; omit it when it cannot be inferred
  reliably. A normalized frequency grid keeps its `cycles/sample` units.
- `n_observations` records the count used by the expectation. It is not a claim
  about effective independent sample size. Include the existing independence
  assumptions; any future effective-count estimate needs a separate definition.
  Do not call an arbitrary external observation axis “trials” without evidence.
- `frequency_bin_spacing` describes the returned grid, after decimation, and has
  explicit frequency units. Omit the scalar for irregular or singleton grids.
  It is not spectral resolving power.
- For built-in DPSS, identify `spectral_bandwidth` as the full concentration
  bandwidth. For Hann STFT/Welch, identify equivalent noise bandwidth. A future
  Morlet bandwidth must state whether it describes amplitude or power and which
  width convention it uses. Custom tapers also need an explicit definition.
- STFT currently subclasses Multitaper. Renaming the STFT property must also
  prevent it from inheriting Multitaper's supported `frequency_resolution`
  property with a different meaning. Include that in the D2 acceptance checks.
- Preserve units and `long_name` on copied coordinates. Keep recording identifiers
  such as scalar `subject`/`session` coordinates when unambiguous. Do not copy a
  trial-level condition label onto a result that averages several conditions.
- Put interpretation, sign convention, theoretical range, and diagonal policy
  on each measure variable. State whether a complex measure's range bounds its
  magnitude. Shared provenance also belongs on the Dataset; extracting a variable
  should retain enough provenance to interpret it.

The different bandwidth concepts are intentional. For example, SciPy's
[Welch documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html)
describes segment averaging and the window-dependent conversion between spectrum
and spectral density. Equal FFT grids do not imply equal smoothing.

### Rules for frequency selection

1. For delay, group delay, and PSI, the canonical `frequency_range` selects the
   bins consumed by the estimator. Record the requested bounds, actual selected
   bounds, and `frequency_interval_closed="both"` on its variables.
2. For Wilson-factorized measures, preserve the full two-sided spectrum during
   factorization. The wrapper's range crops the resulting frequency axis.
3. `frequency_bands` continues to mean aggregation of frequency-resolved results.
   Do not silently apply it a second time to an already frequency-reduced measure.
4. Inclusive adjacent bands share an endpoint for discrete means. Document this.
   Preserve the existing cell-overlap integration weights so neighboring band
   integrals remain additive; endpoint inclusion must not double-count power.
5. Old `frequencies_of_interest` calls record `frequency_interval_closed="neither"`
   and warn with an exact migration example. This compatibility path preserves
   previous bin selection instead of silently changing estimates.

For D6, [SciPy names and documents coherence as magnitude-squared coherence](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.coherence.html).
The proposed longer name makes that distinction explicit in saved variables.
Canonical defaults use the new variable name. An explicitly requested deprecated
name keeps its old output variable name during 3.x, so existing selectors work.
Aliases should not appear twice in defaults or count as different estimators.

### Required D6 migration row

| 2.x code | 3.0 migration |
| --- | --- |
| `multitaper_connectivity(x, fs).coherence_magnitude` or `result["coherence_magnitude"]` on a default result | Use `.magnitude_squared_coherence` or `result["magnitude_squared_coherence"]`. The old variable is absent from the new default Dataset, so old indexing fails loudly. During 3.x, explicitly requesting `method="coherence_magnitude"` retains the old variable name and emits a deprecation warning; that alias is removed in 4.0. |

Here `fs` is a local variable passed to the existing `sampling_frequency`
parameter. This proposal does not introduce an `fs=` keyword.

## B. Function signatures and result contracts

| ID | Topic | Recommended choice | Compatibility and timing |
| --- | --- | --- | --- |
| D9 | Group results and tuning parameters | Preserve score-only tuple returns and the richer `MultivariateConnectivityResult` where they describe different outputs. Make `rank`, `regularization`, and `n_components` keyword-only on new 3.0 APIs that support them. Use the xarray interface as the common labeled representation. | Before 3.0 for newly introduced signatures. Do not add unused tuning parameters to `canonical_coherence` or retrofit a different estimator solely for signature symmetry. |
| D10 | Welch window vocabulary | Keep `segment_duration` and `n_time_samples_per_segment`, because Welch averages segments into an output estimate. Prefer `segment_step` in seconds; retain `segment_overlap` explicitly as a fraction. Reject simultaneous step and overlap arguments. Report segment duration/step separately from output time support. | A step option can ship later. Keep Welch's current default 50% overlap and estimator behavior. Renaming its segments to output windows would obscure their role. |
| D11 | Positional transform arguments | Prefer `time_series, sampling_frequency` as the common positional prefix; make transform-specific arguments keyword-only in the new 3.0 transforms. Keep Multitaper's released positional argument order unchanged. | Set new signatures before 3.0. Multitaper's positional arguments remain a supported contract, not a second deprecated signature; do not silently reorder them. |
| D12 | Wrapper argument routing | Keep common high-level settings as explicit, flat keywords, including `sampling_frequency`, `time_window_duration`, `time_window_step`, `time_halfbandwidth_product`, `n_tapers`, `detrend_type`, and `group_labels` where applicable. Use `transform_kwargs` and `connectivity_kwargs` mappings for advanced settings. `connectivity_to_xarray` accepts an existing transform and therefore needs only `connectivity_kwargs`. | Introduce the explicit mappings in 3.0. Common named settings remain supported without warnings and preserve released positional order. Deprecate only legacy `**kwargs` forwarding for advanced options moved into the mappings. Reject duplicate keys across named, mapping, and legacy routes. Per-measure mappings can be a later addition. |
| D13 | “Method” versus “measure” | Keep `method=`, `DEFAULT_METHODS`, `list_measures()`, `measure` metadata, and `UnsupportedMeasureError`. Describe `method` as the selector for a connectivity measure. | No rename. The current distinction is understandable and does not justify changing several established entry points. |
| D14 | Statistics keywords | Prefer `power_confidence_intervals(n_observations=..., power=..., ci=...)`. Rename the count because it describes all averaged observations; retain `ci` and clearly document that it is a confidence level. Preserve the existing formula, coverage assumptions, argument order, and accepted confidence levels. | Add `n_observations` in 3.0 and keep `n_tapers` as a deprecated keyword alias with a conflict error and removal in 4.0. Keep `ci` fully supported without a `confidence_level` synonym or warning. The observation-count rename prevents a scientifically consequential mistake. |
| D15 | Function spelling | Keep the released `simulate_MVAR`, `Benjamini_Hochberg_procedure`, and `Bonferroni_correction` spellings, return values, and method-selector strings. Document their existing outputs and interpretation. | These released names remain supported without warnings or planned removal. Add no lowercase aliases solely to change capitalization. |
| D16 | Duplicate constructors | Make `Connectivity.from_transform` canonical and retain `from_multitaper` only as a deprecated compatibility alias. | Warn at each caller's use under Python's warning filters; remove `from_multitaper` in 4.0. Update internal calls to the canonical constructor so normal package use does not trigger the alias warning. |
| D17 | Simulator dimensions | Prefer `(time, trial, signal)` with `n_trials=1` for the two newly introduced simulators, matching `simulate_MVAR`. Make explicit `n_trials=None` the documented 2-D option for `simulate_lagged_broadband`. Make shared-oscillation amplitudes an explicit keyword when giving `n_trials` a default. | Both added simulators are absent from v2.0.1: set their defaults before 3.0 without aliases and update the asserting tutorials. Keep the released MVAR shape. |

Welch's segments have a different role from STFT output windows; preserving the
term is consistent with [the standard Welch interface](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html).
Likewise, missing `rank` on a different canonical estimator is not evidence that
adding it would preserve that estimator's definition.

## C. Additive fixes and documentation

| ID | Topic | Recommended choice | Timing |
| --- | --- | --- | --- |
| D18 | Morlet coordinate attributes | Preserve the existing `time` and `frequency` attrs when attaching `valid_time_frequency` or `valid_time`. | Fix before 3.0; small, confirmed defect. |
| D19 | Unknown power units | Omit `units` when physical input units are unknown. Document that omission; retain `long_name`. Use `"1"` only for dimensionless measures. | Fix the documentation before 3.0. No invented unit string. |
| D20 | Coordinate descriptions | Add `long_name` and applicable units to band edges, connection labels, and copied channel metadata. | Bundle with the result-schema work. |
| D21 | JSON encoding | Use the existing canonical serializer everywhere a field ends in `_json`. Structured optional values use JSON `null`; unknown optional scalar attributes are omitted in the canonical schema. Preserve legacy sentinel strings only in legacy attributes. | Bundle with D1–D3, with NetCDF round-trip checks. |
| D22 | Public exports | Add deliberate module `__all__` lists that retain documented public names. Document internal helpers as internal. | Later is fine. `__all__` controls star imports and some documentation tools; it does not prevent direct imports of helpers. |

## D. Additional decisions from review

| ID | Topic | Recommended choice | Compatibility and timing |
| --- | --- | --- | --- |
| D23 | Band-reduction keywords | Rename the helper's arguments to `frequency_band_reduce(result, frequency_bands=..., frequency_reduction="mean", circular=None)`, matching the wrapper. Keep the band mapping as the second positional argument and reduction/circular options keyword-only. | The helper is new in 3.0: replace `bands` and `reduction` now, update examples, and add no aliases. |
| D24 | Global coherence component count | Keep `global_coherence(max_rank=...)` and clearly document that it limits the number of returned components. Other component APIs retain their own `n_components` settings; distinguish output component counts from estimation subspace rank in their docs. Make the unreleased `max_workspace_elements` option keyword-only. | The released `max_rank` keyword and positional count remain supported without warnings or removal. Add no `n_components` synonym to `global_coherence`; keyword-only workspace control can be set before 3.0. |
| D25 | Power's `source` versus components' `signal` dimension | Keep `power(time, frequency, source)` and the component projections' `signal` axis. Power then aligns with the source dimension of pairwise results; `signal` in a projection indexes channels across its labeled sides. Document this deliberate distinction. | Power's `source` axis shipped in 2.x. No dimension rename or alias. A user wanting a standalone channel axis can explicitly call `power.rename(source="signal")`. |
| D26 | STFT representation and summary | Give STFT its own `repr` and `summarize_parameters()`: identify the Hann STFT, actual window/step, and equivalent noise bandwidth. Omit Multitaper-only settings such as NW and DPSS taper count from that description. | Fix before 3.0. The transform and these inherited descriptions are unreleased; no compatibility strings are needed. |
| D27 | Welch parameter inspection | Expose read-only scalar constructor settings on Welch, including resolved `segment_duration`, `detrend_type`, `start_time`, `n_fft_samples`, and `fft_workers`, alongside its existing rate, overlap, and sample counts. Expose resolved `segment_step` when D10 lands. | Add before 3.0 so callers need not inspect `_stft`. Document sample rounding and distinguish requested overlap from the realized step. Derive properties from the same state used for computation. |
| D28 | Parameter-helper terminology | Keep `suggest_parameters(desired_freq_resolution=...)`, `estimate_frequency_resolution(sampling_frequency, time_window_duration, time_halfbandwidth_product)`, and the returned `frequency_resolution` key. Explain that frequency resolution here is full DPSS concentration bandwidth in Hz, and distinguish it from FFT bin spacing in docs and examples. | These released names, signatures, and dictionary key remain supported without warnings or a 4.0 removal. Add no parallel `concentration_bandwidth` property/helper/keyword/key spellings. Documentation clarification belongs in step 2. |
| D29 | Minimum-phase tolerance calls | Keep `minimum_phase_decomposition`'s released positional and keyword forms for `tolerance` and `max_iterations`, including their order, defaults, and meanings. Prefer explicit keyword calls in documentation and examples. | Both forms remain supported without warnings or planned removal. A style preference for keyword calls does not change the released signature. |
| D30 | Invalid taper counts | Reject invalid explicit `n_tapers` at Multitaper construction, including zero, negative, fractional, boolean, or counts above the resolved window length. Keep `None` as automatic selection and check consistency with supplied tapers. | Fix before 3.0. Fail before FFT/allocation with a targeted message; preserve valid-call behavior. |
| D31 | Simulator lag and frequency units | Use `lags_samples` for the lagged-broadband simulator's whole-sample delays. Keep oscillation `frequency` and `sampling_frequency` in Hz, and document MVAR lags as sample steps. Include an example converting a desired delay in seconds to a rounded sample count. | The lagged simulator is new: rename `lags` before 3.0 without an alias. Keep MVAR's released semantics. Time delays and oscillation frequencies are different quantities, so consistency means explicit units, not giving them one unit. |

## Compatibility policy

- Use the verified release boundary above. Drop renamed, unreleased paths now;
  do not create compatibility copies for intermediate development versions.
- Every deprecated callable, keyword, positional route, or property emits
  `DeprecationWarning` at the external call site, naming the replacement and
  **removal in 4.0**. Keep its released numerical meaning during 3.x. Tests must
  enable and assert the warning explicitly because Python often filters this
  warning category by default. Internal package code uses canonical paths.
- **Plain-dictionary exception:** xarray attrs cannot warn on key access.
  Keep only the released legacy metadata keys described above,
  document them as deprecated, test their agreement, and remove them in 4.0.
  Do not add custom dict/container types just to intercept reads. This is an
  explicit warning-mechanism limitation, not an indefinite compatibility promise.
- Retain the estimator result when adding a spelling alias. A keyword with
  historical exclusive band edges needs the explicit compatibility behavior in D4.
- Track all these removals in one issue titled **“Remove deprecated 2.x API and
  metadata aliases in 4.0”**, following the role of #98 for the orientation
  warning. The issue must list old and new names, first deprecation release,
  warning tests, documentation, and the 4.0 removal checklist.
  [Issue #101](https://github.com/Eden-Kramer-Lab/spectral_connectivity/issues/101)
  was created and linked in PR #102. Update its entries in the implementation
  PR that introduces each deprecation.
- The 4.0 release removes the tracked aliases and legacy metadata dictionary
  keys. This policy applies to paths explicitly
  deprecated by this sheet; unchanged interfaces such as D11's Multitaper
  signature, D25's power dimension, and D2/D28's frequency-resolution property
  and helper names remain supported. D14's `ci`, D15's released spellings,
  D24's `max_rank` keyword/positional count, D29's positional tolerances, and
  D12's common named wrapper settings also remain supported.
- Keep DataArray returns for ordinary single measures and Dataset returns for
  multi-variable or multi-measure results. This pass does not require a universal
  `measure` dimension or a new container type.

## Suggested PR sequence

| PR | Scope | Required evidence |
| --- | --- | --- |
| 1 | Create the 4.0 removal issue; fix coordinate attrs/descriptions, transform inspection, and taper validation (D18, D20, D26, D27, D30). | Morlet coordinates keep attrs through serialization; STFT identifies itself; Welch properties match computation; bad taper counts fail at construction. |
| 2 | Define metadata, estimator bandwidths, existing helper terminology, and JSON encoding (D1–D3, D19, D21, D28). | One canonical representation plus only released compatibility keys; supported frequency-resolution names retain their values without warnings; no unreleased prefixed copies; NetCDF round-trips. |
| 3 | Add canonical frequency selection and align reduction keywords (D4, D23). | Exact-edge examples; old exclusive-edge calls warn and agree with historical selection; new helper keywords match wrappers; unchanged Wilson input spectra; additive band-power integrals. |
| 4 | Add coherence/source naming and interpretation metadata; document power's axis (D5–D8, D25). | Numerical identity of aliases; D6 default-variable migration example; explicitly requested old names still work with warnings; unchanged documented diagonals; component serialization. |
| 5 | Set new signatures and advanced wrapper routing; correct the statistics observation-count keyword; deprecate the constructor alias; document retained component-count and tolerance calls (D9, D11, D12, D14, D16, D24, D29). | Supported released forms and common wrapper keywords work without warnings; genuinely deprecated paths agree numerically and warn at callers; duplicate arguments fail; published positional order is preserved. |
| 6 | Set simulator shapes and lag units; finalize migration examples and the removal checklist (D17, D31). | All documented shapes and asserting tutorials pass; seconds-to-samples conversion is explicit; every deprecated path is tracked for 4.0. |

D10's step option and D22 can follow in 3.x. The released names and call forms
retained by D2, D11, D12, D14 (`ci`), D15, D24, D25, D28, and D29 deliberately
remain supported. Publish a short schema reference and migration table alongside
implementation. Keep this planning sheet updated with accepted decisions and
implementation status.

## Step 2 implementation preparation

**Scope:** D1–D3, D19, D21, and D28. **Branch:**
`refactor/metadata-and-bandwidths`, based on merged `master` at `138f3f7`.
Step 1, including its custom-taper and band-label follow-ups, is complete.

**Merged-base validation:** the full local CPU suite passed with 1,801 tests
and 16 skips on `138f3f7`. This is the baseline for step 2.

### Implementation order and affected code

1. **Bandwidth definitions and supported helper names** — `transforms.py`.
   Keep `Multitaper.frequency_resolution`, `estimate_frequency_resolution`,
   `suggest_parameters(desired_freq_resolution=...)`, their released positional
   signatures, and the returned `frequency_resolution` key. Document full DPSS
   concentration bandwidth, its formula, and units; add no synonyms or warnings.
   Rename STFT's unreleased property to `equivalent_noise_bandwidth`, expose
   that quantity on Welch, and prevent STFT from inheriting Multitaper's
   `frequency_resolution` property. Update STFT summaries and metadata callers
   to use its estimator-specific property.
2. **Canonical transform provenance** — `_provenance.py`, transform metadata
   builders, and the three wrapper entry points in `wrapper.py`.
   Record D1's common facts and `output_schema_version=1` once, with lowercase
   `backend`. Take `n_observations` and independence flags from the actual
   `Connectivity`; include `n_trials` only when its trial meaning is known.
   Put remaining native settings into `transform_parameters_json`, separating
   `estimator` and `execution`, and serialize through `_canonical_json`.
   Apply the 15-key released `mt_*` whitelist only to Multitaper results.
   Canonical optional settings use JSON `null`; omitted unknown scalar facts
   and legacy `"None"` sentinels follow the compatibility rules above.
3. **Metadata consumers and returned frequency grids** — `_result_formatting.py`,
   `_frequency_bands.py`, and the external-Fourier adapter in `wrapper.py`.
   Replace reads of prefixed sampling rates and coordinate-origin flags along
   with their writers. Preserve normalized `cycles/sample` grids and index time
   coordinates, and infer a physical sampling rate only from reliable supplied
   full FFT coordinates. Compute uniform `frequency_bin_spacing` with explicit
   units from the returned grid after cropping/decimation; omit it for singleton,
   irregular, or frequency-reduced outputs. Keep qualified estimator bandwidth
   separate from bin spacing, and omit undefined custom-taper/Morlet bandwidths.
   Ensure Dataset variables retain their own applicable provenance when selected.
   Preserve unambiguous scalar recording coordinates and copied coordinate attrs.
4. **Units, documentation, and migration** — wrapper docstrings, `docs/index.md`,
   `docs/cookbook.md`, `docs/llm_guide.md`, transform/helper doctests, and paired
   tutorial sources. Clarify the supported frequency-resolution names and their
   definitions. `_measure_label_attrs` already omits unknown power units;
   correct the promise that every variable has `units` and test both known and
   unknown units. Add a short result-schema reference and migration examples;
   update the relevant issue #101 entries when the deprecations are implemented.

### Acceptance checklist

- [ ] Multitaper, STFT, Welch, Morlet, and external Fourier results have the
  common schema on DataArrays and Datasets. There are no unreleased transform
  prefixes or unapproved `mt_*` keys; the released compatibility keys agree.
- [ ] An equal FFT grid can report different estimator bandwidths: at 100 Hz
  with a 0.5-second window and NW=2, DPSS concentration bandwidth is 8 Hz and
  Hann equivalent noise bandwidth is 3 Hz. Decimation changes reported bin
  spacing without changing the estimator bandwidth.
- [ ] Multitaper's frequency-resolution property, helper names, keyword, and
  returned key retain their values and positional order without warnings or
  synonym spellings. STFT exposes `equivalent_noise_bandwidth` and does not
  inherit Multitaper's `frequency_resolution` property.
- [ ] Metadata does not depend on whether tapers/FFT have already been cached;
  custom tapers and low-bias DPSS counts retain step 1's regression coverage.
- [ ] Structured settings decode as canonical JSON, including optional `null`
  values, and ordinary results round-trip with scipy, netCDF4, and h5netcdf.
  External normalized/irregular grids, band reductions, and extracted Dataset
  variables retain accurate units and applicable provenance.
- [ ] Update tests in `test_parameter_helpers.py`, `test_transforms.py`,
  `test_wrapper.py`, and `test_wrapper_netcdf.py`, plus doctested documentation
  and tutorials covering the clarified names and STFT's renamed property.
  Run the full suite, source doctests, Ruff formatting/lint, mypy, and CI's
  minimum-dependency checks.

## Verification performed for this sheet

Read the transform constructors and provenance builders, wrapper formatting and
band reduction, measure registry, group-method signatures, statistics helpers,
and simulators. Ran small CPU examples that reproduced the five-prefix metadata
problem, the 8 Hz versus 3 Hz bandwidth descriptions, Morlet's lost coordinate
attributes, exclusive core band selection, omitted unknown power units, and the
different diagonal values. For this revision, compared v2.0.1 definitions and
wrapper attribute export with the current source, including both parameter
helpers, `global_coherence`, the power axis, and the minimum-phase signature.
No estimator code was changed. These probes verify the listed behavior; they
are not a new validation of the statistical methods.
