# 3.0 API and result consistency decisions

**Status: all six steps, and the internal base-class step, are implemented and
merged into `master` (see the table below).**

| Step | Decisions | Branch | Status |
| --- | --- | --- | --- |
| 1 | D18, D20, D26, D27, D30 | `fix/coordinates-and-transforms` | Merged as PR #102 (`138f3f7`) |
| 2 | D1-D3, D19, D21, D28 | `refactor/metadata-and-bandwidths` | Merged as PR #108 (`6749e51`) |
| (internal) | windowed-FFT base | `refactor/windowed-fft-base` | Merged as PR #112 (`df37de2`) |
| 3 | D4, D23 | `refactor/frequency-selection` | Merged as PR #113 (`7c278d7`) |
| 4 | D5-D8, D25 | `refactor/coherence-naming` | Merged as PR #114 (`7a94abb`) |
| 5 | D9, D11, D12, D14, D16, D24, D29 | `refactor/signatures-and-routing` | Merged as PR #115 (`04200dc`) |
| 6 | D17, D31; migration table and removal checklist | `refactor/simulators-and-migration` | Merged as PR #116 (`518e17d`) |

D10's step option and D22 remain deferred to 3.x, as decided. D13 and D15
needed no code change.
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
| D2 | Frequency resolution | Keep `Multitaper.frequency_resolution` as the single supported public name and define it as full DPSS concentration bandwidth (`2 * NW / time_window_duration`, in Hz). Use STFT/Welch `equivalent_noise_bandwidth` and `frequency_bin_spacing` for a uniform output grid. Results record a qualified `spectral_bandwidth` with `spectral_bandwidth_definition`; frequency-dependent bandwidth belongs on a frequency coordinate. | Retain Multitaper's released property and value without deprecation or a `concentration_bandwidth` synonym. Rename STFT's unreleased `frequency_resolution` directly, with no compatibility alias, and prevent inheritance of Multitaper's property with a different meaning. Status: the property keeps its definition and its value for any window of a whole number of samples; since 3.0 it describes the realized (rounded) window, so a 0.505 s window at 100 Hz with NW=3 reports 12 Hz where 2.0.1 reported 11.88 Hz (CHANGELOG migration guide). |
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
- STFT must not inherit Multitaper's supported `frequency_resolution` property
  with a different meaning. Step 2 therefore stops STFT subclassing Multitaper:
  it holds one, so it has none of the DPSS-only members. Include that in the D2
  acceptance checks.
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
| D28 | Parameter-helper terminology | Keep `suggest_parameters(desired_freq_resolution=...)`, `estimate_frequency_resolution(sampling_frequency, time_window_duration, time_halfbandwidth_product)`, and the returned `frequency_resolution` key. Explain that frequency resolution here is full DPSS concentration bandwidth in Hz, and distinguish it from FFT bin spacing in docs and examples. | These released names, signatures, and dictionary key remain supported without warnings or a 4.0 removal. Add no parallel `concentration_bandwidth` property/helper/keyword/key spellings. Documentation clarification belongs in step 2. Status: the helpers compute from the duration they are given and keep their 2.0.1 values; only `Multitaper.frequency_resolution` follows the realized, rounded window (D2). |
| D29 | Minimum-phase tolerance calls | Keep `minimum_phase_decomposition`'s released positional and keyword forms for `tolerance` and `max_iterations`, including their order, defaults, and meanings. Prefer explicit keyword calls in documentation and examples. | Both forms remain supported without warnings or planned removal. A style preference for keyword calls does not change the released signature. Status: the order and meanings are kept, and both forms work without warnings; the `max_iterations` default is 500 from 3.0, where 2.0.1 used 60 (owner's ruling in step 5, recorded in the CHANGELOG; pass `max_iterations=60` for the 2.0.1 cap). |
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

Between PRs 2 and 3, a separate internal PR extracts a private windowed-FFT base
class shared by `Multitaper` and `ShortTimeFourierTransform`, replacing the
Multitaper that step 2 leaves STFT holding. It changes no public name, value,
or result attribute.

D10's step option and D22 can follow in 3.x. The released names and call forms
retained by D2, D11, D12, D14 (`ci`), D15, D24, D25, D28, and D29 deliberately
remain supported. Publish a short schema reference and migration table alongside
implementation. Keep this planning sheet updated with accepted decisions and
implementation status.

## Step 2 implementation

**Scope:** D1–D3, D19, D21, and D28. **Branch:**
`refactor/metadata-and-bandwidths`, based on merged `master` at `138f3f7`.
Step 1, including its custom-taper and band-label follow-ups, is complete.
Step 2 is implemented on this branch and ready for PR review.

**Merged-base validation:** the full local CPU suite passed with 1,801 tests
and 16 skips on `138f3f7`. This is the baseline for step 2.

**Step 2 validation (2026-10-06):** 1,840 tests passed with 16 skips on the
main CPU environment; 1,723 passed with 103 skips on Python 3.10 with the exact
NumPy/SciPy/xarray/Matplotlib floors. The floor run uses CI's `-p no:warnings`
for third-party deprecations and excludes notebook tests; optional NetCDF
engines are checked in the main environment. All 70 source doctests, Ruff
formatting/lint, mypy, and NumPy docstring checks for both new bandwidth
properties passed. The singular-solver regression fixture uses identical rows
to trigger an exact zero pivot consistently across LAPACK builds.

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

- [x] Multitaper, STFT, Welch, Morlet, and external Fourier results have the
  common schema on DataArrays and Datasets. There are no unreleased transform
  prefixes or unapproved `mt_*` keys; the released compatibility keys agree.
- [x] An equal FFT grid can report different estimator bandwidths: at 100 Hz
  with a 0.5-second window and NW=2, DPSS concentration bandwidth is 8 Hz and
  Hann equivalent noise bandwidth is 3 Hz. Decimation changes reported bin
  spacing without changing the estimator bandwidth.
- [x] Multitaper's frequency-resolution property, helper names, keyword, and
  returned key retain their values (for windows of a whole number of samples;
  a rounded window reports the realized window, see D2) and positional order
  without warnings or
  synonym spellings. STFT exposes `equivalent_noise_bandwidth` and does not
  inherit Multitaper's `frequency_resolution` property.
- [x] Metadata does not depend on whether tapers/FFT have already been cached;
  custom tapers and low-bias DPSS counts retain step 1's regression coverage.
- [x] Structured settings decode as canonical JSON, including optional `null`
  values, and ordinary results round-trip with scipy, netCDF4, and h5netcdf.
  External normalized/irregular grids, band reductions, and extracted Dataset
  variables retain accurate units and applicable provenance.
- [x] Existing parameter-helper and NetCDF coverage passes, and schema regressions
  are covered in `test_result_schema.py`, `test_transforms.py`, and `test_wrapper.py`,
  plus doctested documentation
  and tutorials covering the clarified names and STFT's renamed property.
  Run the full suite, source doctests, Ruff formatting/lint, mypy, and CI's
  minimum-dependency checks.

## Step 3 implementation

**Scope:** D4 and D23. **Branch:** `refactor/frequency-selection`, based on
the head of `refactor/windowed-fft-base` (now `39b4784`, on step 2's head
`1953a61`; the step was first built on that branch's earlier head `77e83ce`,
since rebased).
Implemented and validated locally on 2026-11-19: 2,142 tests passed with 16
skips on the main CPU environment; all 70 source doctests, Ruff
formatting/lint, and mypy passed.

Rulings where the rules above were silent:

- **Comparison rule.** One helper, `_frequency_interval_mask`, serves the
  wrappers' crop, each named band, and the three interval measures. Under
  `"both"` a bin `f` lies on a bound `b` when `|f - b| <= 1e-6 * gap(f) +
  2 * (ulp(f) + ulp(b))`: `gap(f)` is the distance to its nearest neighbouring
  bin (zero for a single bin) and each unit in the last place is in the
  value's own stored precision. These are `_uniform_grid_spacing`'s two
  allowances (one named `_GRID_SPACING_RTOL`, one `_last_place`), with the
  local gap in place of the mean step so irregular grids are judged bin by
  bin. A decimal bound names a bin when the bin has drifted from it by at
  most one part in a million of its spacing (plus rounding): float32, `0.1 *
  k`, `np.arange(1.0, 100.0, 0.1)` (bin 142 is `15.200000000000014`), and
  float64 cumulative-sum grids do; `np.arange(1000.0, 1100.0, 0.001)` (drift up
  to 2.4e-6 of its spacing) and float32 cumulative sums do not. A bound 1e-3 of
  the spacing from a bin does not select it.
  The deprecated keyword does not use that rule: its frozen selection,
  `_frequencies_of_interest_mask`, has its own validation, so no change to
  the canonical rule can alter what it accepts, and `_FrequencySelection`
  records it in a `deprecated` field from which `frequency_interval_closed`
  is derived. It evaluates v2.0.1's expression, `(band[0] < frequencies) &
  (frequencies < band[1])`, on the band as the caller gave it and the grid as
  stored, so the deprecated keyword selects the bins it selected in v2.0.1
  (rule 5) under the same NumPy: a Python-number band meets a float32 grid in
  float32, while an ndarray band's float64 scalar compares in float64 under
  NumPy 2 (NEP 50) but is cast to float32 under NumPy 1.x, as in v2.0.1
  itself (checked against the 2.0.1 wheel under both NumPy 2.5 and 1.24). On a rounded grid that can include a bin within rounding of
  an edge, as in v2.0.1. (Development builds before this step compared in
  float64 and gained edge bins on float32 grids.) It accepts infinite bounds
  and selects exactly the bins v2.0.1 selects for an open-ended band; it
  rejects a NaN bound, `low >= high`, and a band with no bin inside. A band
  that is not exactly two values also raises, deliberately: v2.0.1 silently
  used the first two elements of a longer band, which hides a caller's
  mistake. An infinite bound records no `requested_frequency_*` attribute for
  that side, so the interval attributes stay finite floats; the selected
  bounds and `frequency_interval_closed="neither"` still describe the
  interval, and `measure_kwargs_json` records the band as given.
- **Positional slot.** `frequencies_of_interest` was the first positional
  parameter in v2.0.1, so it stays there as the deprecated route (by keyword or
  position); `frequency_range` is keyword-only. A 2.x positional band therefore
  warns and keeps its exclusive meaning instead of silently gaining edge bins.
- **Recorded attributes.** `requested_frequency_lower`/`_upper` (omitted when
  every bin is used), `selected_frequency_lower`/`_upper` (lowest and highest
  bins in the interval, before any `frequency_resolution` subsampling), and
  `frequency_interval_closed` replace the unreleased
  `frequency_band_lower`/`_upper` on each variable of the three measures.
- **One interval record.** `_FrequencySelection.attrs()` is the single writer
  of those five attributes for every variable a `frequency_range` applied to,
  whether the wrapper cropped it or the estimator consumed the range. The
  unreleased `frequency_range_json` is retired without an alias;
  `measure_kwargs_json` and `arg_*` remain the generic record of a measure's
  keyword arguments. Decimation and band reduction keep the interval record,
  which names the interval they were drawn from, and add their own
  (`frequency_decimation`, `frequency_bands_json`, `frequency_reduction`).
- **Wrapper meaning.** `frequency_range` is selected once per call, on the
  full grid, before any measure runs. It is passed to the interval measures,
  which are not cut again, and that one selection crops every other result
  before the merge. It is rejected inside `connectivity_kwargs`.
  `frequency_bands` and `frequency_decimation` act on
  frequency-resolved variables, `delay` included, and leave
  `phase_slope_index`/`group_delay` as computed over `frequency_range`; a
  request made only of those two raises.
- **One selection, one record.** Each entry point
  (`multitaper_connectivity`/`fourier_connectivity` and
  `connectivity_to_xarray`) resolves a call's interval once, on the full grid,
  into one `_FrequencySelection`, and passes it to the formatter, so the
  coordinates, interval attributes, crop, and validity masks read the same
  object. The estimators still resolve their own public keywords from the
  caller's arguments as given (not floats re-derived from them, which would
  lose a float32 bound's precision). Which measures consume the range is one
  registry property, `_MeasureSpec.consumes_frequency_range`, derived from
  `output_kind`.
- **D23.** `frequency_band_reduce(result, frequency_bands, *,
  frequency_reduction="mean", circular=None)` with no aliases; its error
  messages name the canonical keywords.

### Step 3 acceptance checklist

- [x] Exact-edge examples: bounds on bins are included by the crop, bands, and
  all three measures, including float32 and decimal-rounded grids
  (`tests/test_frequency_selection.py`, cookbook "Select a frequency range",
  and the `delay` docstring example, now 91 bins for 5-50 Hz).
- [x] Old exclusive-edge calls warn at the caller with an exact migration
  example and agree with the released selection: v2.0.1 itself and this
  branch give the same `delay` bin counts for 30 tuple, list, and ndarray
  bands on float32 and float64 grids, and 38 of 40 recorded float64 and
  float32 outputs are bit-identical to the development builds before this
  step (`4266d7c`, and `77e83ce`, a base-class head since rebased, measured
  before the rebase); the other two are
  a float32 tuple band where those development builds had departed from
  v2.0.1. Open-ended bands (`(10.1, inf)`, `(-inf, 20.3)`, `(-inf, inf)`,
  `(0.3, inf)`) match the 2.0.1 wheel in all 48 tuple, list, ndarray,
  keyword, and positional cases; NaN, `(inf, inf)`, one-value, and
  three-value bands raise by design.
- [x] New helper keywords match the wrappers and give identical results.
- [x] Unchanged Wilson input spectra: the recorded factorization input is the
  same with and without `frequency_range`, and the cropped result equals the
  full result's slice.
- [x] Additive band-power integrals for adjacent inclusive bands; their means
  share the endpoint bin.
- [x] Issue #101 entry for the `frequencies_of_interest` alias.

## Step 4 implementation

**Scope:** D5-D8 and D25, the last bullet of "Rules for the shared metadata",
and the D6 migration row. **Branch:** `refactor/coherence-naming`, based on
step 3's final head (now `0480d01`). Implemented and validated locally on
2026-11-19: 2,239 tests passed with 16 skips on the main CPU environment, and
2,124 passed with 101 skips on Python 3.10 with the NumPy 1.24/SciPy
1.11.1/xarray 2023.1/Matplotlib 3.7 floors; all 70 source doctests (also on the
floors), Ruff formatting/lint, and mypy passed.

Rulings where the rules above were silent:

- **Alias resolution.** `_DEPRECATED_MEASURE_NAMES` in `connectivity.py` (next
  to `_NON_MEASURE_METHODS`, so the registry and `jackknife` share it) maps
  `coherence_magnitude` to `magnitude_squared_coherence`, and
  `_resolve_deprecated_measure_name` beside it writes the one warning and
  returns the canonical name, for the alias method, `jackknife`, and the
  wrappers' single entry point, `_requested_methods` (also used by
  `connectivity_to_xarray`). That entry point rejects both spellings,
  validates, and resolves once; everything downstream works with canonical
  names, and only the result variable's name and `measure` use the requested
  one. The registry has one record, under the canonical name.
- **Provenance.** A result requested by the old name keeps the variable name
  and records `measure="coherence_magnitude"`; every other attribute is the
  canonical measure's. No extra "canonical measure" attribute is added: the
  migration table maps the name.
- **Both spellings.** A `method` list naming both raises `ValueError`.
  `connectivity_to_xarray`'s default `method` becomes the canonical name, so
  its default result variable changes like the wrappers'.
- **jackknife.** It accepts the old name with one warning and uses the
  canonical name throughout, so its automatic `fisher_squared` scale is kept
  (an unknown name would fall to `identity`).
- **Diagonal policy vocabulary.** `masked_nan` (a package mask of a defined
  self-value), `undefined_nan` (no self-connection: the Granger family),
  `constant` (registry `diagonal_value`: 0, 0.5, or 1, to within the
  rounding of the averages behind it, which NumPy 2 often makes exact and
  NumPy 1.24 does not), `rounding_noise` (nominally `diagonal_value`, but
  computed from rounding residues the formula amplifies), `estimated`
  (cross-spectral density, transfer-function family), `not_applicable`
  (power, global coherence, component measures). A group with itself in
  `canonical_coherence`, MIC, and MIM is `masked_nan` (computable, not
  computed); `delay`, `group_delay`, and `phase_slope_index` inherit
  coherency's mask. `group_delay_r_value`'s diagonal is 1 (`constant`) where
  delay and slope are NaN, so each `group_delay` variable records its own.
- **Rounding of diagonals.** `corrected_imaginary_phase_locking_value` is
  `rounding_noise` with nominal value 0, not `constant`: its self-connection is
  the 0/0 it defines as 0, but computed as `|Im(PLV)|` (of order eps) over
  `sqrt(1 - Re(PLV)^2)` (of order sqrt(eps) when `Re` rounds just below 1), so
  it is noise below about sqrt(eps), measured up to 6e-10 in float64 and 2e-5
  at complex64 precision under NumPy 1.24. Tests bound a constant diagonal by
  `4 * (n_observations + 4) * eps` (about 4 eps per normalized term, n eps for
  the sum, and PPC's factor 2n/(n - 1) <= 4; it holds with one and two
  observations), the `rounding_noise` diagonal by `sqrt(eps)`, and the
  cross-spectral density diagonal's imaginary part by `n_observations * eps`
  times its real part. `value_range` is the theoretical range, which rounding
  can cross (PPC's self-value reaches 1 + 1e-15).
- **Flat signals.** `constant` holds for signals with nonzero power.
  `nan_for_zero_power_signal` (0/1 on every measure variable) marks the
  measures for which every entry involving a zero-power signal, the diagonal
  included, is NaN; a test derives it from a flat-channel run of every
  measure and companion. It holds for the coherence measures, PLV, PPC, the
  corrected imaginary PLV, and the delay measures; not for the phase lag
  indices, the cross-spectrum, `imaginary_coherency` (0, not NaN, off the
  diagonal), partial coherence, the Granger and transfer-function families,
  or measures without signal-pair axes.
- **One vocabulary.** `_DIAGONAL_POLICIES` in the registry defines the six
  policies (with the constant policy's rounding bound) and
  `_NAN_FOR_ZERO_POWER_SIGNAL` the flag; the ranges page's policy table is
  generated from them, and other docs refer to it.
- **Recorded nominal value.** Beside `constant` or `rounding_noise`, a
  variable records `diagonal_value` (a float).
- **Companion records.** A `_VariableSpec` holds what a result variable needs;
  `_MeasureSpec` extends it, and group delay's record declares
  `group_delay_slope` and `group_delay_r_value` as its companions:
  interpretation, group delay's sign convention, an unbounded slope and a
  [-1, 1] correlation. Group membership and spatial vectors, filters, and
  patterns carry no measure record.
- **dPLI.** Its lead/lag reading moved to `sign_convention`; `interpretation`
  says what the value is (the fraction of observations in which the source
  leads, `(1 + phase_lag_index) / 2`), so its row in the generated table
  changed.
- **Interpretation attributes.** `interpretation`, `sign_convention`,
  `value_range_lower`/`value_range_upper` (an infinite side omitted, since not
  every engine round-trips a bare infinity; `value_range_applies_to=
  "magnitude"` on complex measures), and `diagonal_policy` go on every
  measure variable from `_MeasureSpec.result_attrs`. The lead/lag sentence
  moved from `interpretation` into `sign_convention`; the generated table
  joins them, so its text is unchanged. `quantity` and `formula` are stated
  for the coherency family only (the coherence measures, partial coherence,
  and the two canonical estimators), where the decision rows ask for them.
- **Band-reduced attributes.** Cropping and decimation select bins and keep
  every measure attribute. Band reduction, where it rewrites `long_name` and
  `units`, prefixes `interpretation` and `formula` with the reduction that
  made each value ("Each value is the mean / circular mean / complex mean /
  integral over the band ..."), since both describe one bin; the value range,
  `value_range_applies_to`, `sign_convention`, `quantity`, and
  `diagonal_policy` hold of band values and are kept.
- **Commit order.** The canonical name lands first with the old name working
  silently (same alias resolution, no warning), then the mechanical move of
  tests and docs, then the warnings, so every commit passes on its own.
- **Snapshots and fixtures.** The notebook snapshot keys keep their stored
  names; the mne-connectivity fixture was regenerated (mne-connectivity
  0.9.0) because its table names our method, and only `our_methods` changed.
  The benchmark's `--compare` looks a renamed measure up under its old name.

### Step 4 acceptance checklist

- [x] Numerical identity of aliases: the method, the method strings of all
  three wrappers, and `jackknife` agree exactly with the canonical name
  (`tests/test_coherence_naming.py`); every registered measure's every
  variable is bit-identical to the parent `0480d01`, and the benchmark passes at 0.
- [x] D6 migration example: the default Dataset has no `coherence_magnitude`
  and old indexing raises (cookbook, CHANGELOG migration row, tests).
- [x] Explicitly requested old names work, keep their variable names, record
  them, and warn at the caller's line; package internals do not warn.
- [x] Unchanged documented diagonals: tests pin every measure's diagonal
  against its recorded policy and value, with rounding bounds that hold on the
  minimum-dependency floors too.
- [x] Component serialization: `connection_source_group`,
  `connection_target_group`, and `side=["source", "target"]` round-trip with
  every engine and select the same values.
- [x] Power's `source` axis and `power.rename(source="signal")` documented.
- [x] Measure attributes stay true after cropping, decimation, and band
  reduction (density, coherence, phase, and complex measures).
- [x] Issue #101 entry for the `coherence_magnitude` alias.

## Step 5 implementation

**Scope:** D9, D11, D12, D14, D16, D24, and D29. **Branch:**
`refactor/signatures-and-routing`, based on step 4's final head `23877e7`.
Implemented and validated locally on 2026-11-19, including the review fixes
and the final review's:
2,400 tests passed with 16 skips on the main CPU environment, and 2,285 passed
with 101 skips on Python 3.10 with the NumPy 1.24/SciPy 1.11.1/xarray
2023.1/Matplotlib 3.7 floors; all 70 source doctests (also on the floors),
Ruff formatting/lint, and mypy passed; the benchmark's 22 arrays are
bit-identical to `23877e7`.

Released forms checked against `v2.0.1`: `Multitaper` (13 positional
parameters, unchanged), `multitaper_connectivity(time_series,
sampling_frequency, time_window_duration, method, signal_names, squeeze,
connectivity_kwargs, **kwargs)`, `connectivity_to_xarray(m, method,
signal_names, squeeze, **kwargs)`, `power_confidence_intervals(n_tapers,
power, ci)`, `minimum_phase_decomposition(cross_spectral_matrix, tolerance,
max_iterations=60)`, `global_coherence(max_rank)`,
`canonical_coherence(group_labels)`, and `from_multitaper(multitaper_instance,
expectation_type, blocks, dtype)`. The STFT, Welch, Morlet, MIC, MIM,
`partial_coherence`, `canonical_coherency`, `max_workspace_elements`,
`fourier_connectivity`, and `from_transform` are new in 3.0.

Rulings where the rules above were silent:

- **One route per setting (owner's ruling).** The six common transform
  settings are named arguments only, with `Multitaper`'s real defaults and no
  sentinel defaults; one of them inside `transform_kwargs` raises
  `ValueError` whether or not it was also named. `group_labels` keeps its
  existing acceptance inside `connectivity_kwargs` (default `None` is
  unambiguous), with the existing both-ways `ValueError`. `frequency_range` is
  a named argument of all three wrappers, `connectivity_to_xarray` included
  (consumed by the interval measures, cropping any other), and is rejected
  inside `connectivity_kwargs`. A key both in a mapping and as a plain keyword
  raises `ValueError` naming both; an unknown key in either mapping or as a
  plain keyword raises `TypeError` with close matches (difflib plus names it
  begins), before any computation. A mapping argument that is not a mapping
  raises `TypeError`.
- **Legacy route.** Every plain keyword the target accepts warns (one
  warning per call listing them), including settings first added in 3.0
  such as `fft_workers`: the deprecated interface is the released forwarding
  route, not each key. The warning is raised just before the wrapper returns,
  so a call rejected for any reason raises only its error.
  `connectivity_to_xarray` gains a keyword-only `group_labels` so that common
  setting keeps a flat route.
- **from_multitaper (cleanup ruling).** Both names call one private
  builder, `_from_transform`, and neither calls the other. A subclass
  overriding `from_multitaper` warns once at its own `super()` call and is not
  reached by `from_transform`, so the migration row says to move such an
  override to `from_transform`; an override of `from_transform` that still
  calls `super().from_multitaper` works, with one warning, instead of
  recursing; and `Sub.from_multitaper` does not reach a `Sub.from_transform`
  override (a promise only this pull request had made). The 2.x `blocks`
  parameter (third position) was removed
  before this step and is not restored (owner's ruling). Every 2.x use of
  it, by keyword or in its released positional slot (third in
  `from_multitaper`, fifth in `Connectivity`), raises a `TypeError` that names
  `blocks`, says it had no effect on results, and says to drop it. Separately,
  `Connectivity` rejects any dtype that is not complex: a value bound there by
  position used to be read as float64 (`None`) and silently drop every
  coefficient's imaginary part.
- **One deprecation template.** `utils.warn_deprecated` writes the
  measure-name, `from_multitaper`, `n_tapers=`, and plain-keyword deprecation
  warnings (not step 3's `frequencies_of_interest` warning, which states an
  exact replacement interval) as "<old> is deprecated and will be
  removed in 4.0; use <new>, which <outcome>.", attributed with
  `stacklevel_outside_package()`.
- **fourier_connectivity (owner's ruling).** New in 3.0, so every parameter
  after `signal_names` is keyword-only, as for the other new 3.0 APIs.
- **n_tapers alias.** It matches the canonical 3.0 interval exactly; 2.0.1's
  numbers differ by the documented formula correction (2.0.1's `ci=c` interval
  is 3.0's `ci=2c-1`). Both spellings raise `ValueError`.
- **max_iterations default (owner's ruling).** It stays 500, a 3.0 change
  from 2.0.1's 60, recorded in the CHANGELOG (Changed and the migration
  guide) and the docstring, with `max_iterations=60` for the 2.0.1 cap.

### Step 5 acceptance checklist

- [x] Supported released forms and common wrapper keywords work without
  warnings (`tests/test_signatures.py`, `tests/test_wrapper_routing.py`).
- [x] Deprecated paths agree numerically and warn at the caller's line:
  `from_multitaper`, plain-keyword settings, `n_tapers=`; compared with the
  2.0.1 wheel under NumPy 2.5 and 1.24.
- [x] Duplicate arguments fail; misspelled mapping keys fail.
- [x] Published positional orders are preserved and pinned by tests.
- [x] Issue #101 entries for the three aliases (they cite #115).

## Step 6 implementation

**Scope:** D17 and D31, the migration table and schema reference for the whole
pass, and the issue #101 checklist. **Branch:**
`refactor/simulators-and-migration`, based on step 5's final head `91db050`
(PR #115).
Implemented and validated locally on 2026-11-19, including both final
reviews' fixes: 2,428 tests passed with 16 skips on the main CPU environment,
and 2,313 passed with 101 skips on Python 3.10 with the NumPy 1.24/SciPy
1.11.1/xarray 2023.1/Matplotlib 3.7 floors; all 70 source doctests (also on
the floors), Ruff formatting/lint, and mypy passed. No numerical code
changed, so no benchmark was run: the simulators' draws are bit-identical to
their first 3.0 form, and the notebook snapshot tests' stored values are
unchanged. All three tutorial notebooks execute without error.

Released forms checked against `v2.0.1`: its `simulate.py` defines only
`simulate_MVAR`, so `simulate_lagged_broadband` and
`simulate_shared_oscillation` are new in 3.0 and change without aliases, and
`simulate_MVAR` keeps its signature and `(time, trial, signal)` shape.

Rulings where the rules above were silent:

- **Positional order.** `simulate_lagged_broadband(lags_samples, noise_levels,
  n_time_samples, n_trials=1, random_state=None)` keeps its order, so the
  tutorial's positional calls still read naturally; `n_trials=None` is the
  2-D option and returns the default single trial's values bit for bit (the
  same draws in the same order). `simulate_shared_oscillation` keeps
  `n_trials` fourth with default 1 and makes `amplitudes` a required
  keyword-only argument (a required argument cannot follow a defaulted one
  positionally); a call that still passes it fifth raises `TypeError`. Only
  the lagged-broadband simulator accepts `n_trials=None`, as D17 says.
- **Units.** `lags_samples` is validated as before (non-negative integers);
  its error message names the `round(delay * sampling_frequency)`
  conversion. The seconds-to-samples example is a doctest in the
  `simulate_lagged_broadband` docstring (10.2 ms at 1500 Hz rounds to 15
  samples, a simulated delay of 10 ms) and a labeled line in each
  group-delay and phase-slope cell of the simulated-examples tutorial.
  `simulate_MVAR`'s docstring says its lags and time index count samples; no
  code changed.
- **One migration table.** The CHANGELOG migration guide is the single
  migration table. It gains a row for the 2.0.1 `mt_*` attributes (folded in
  from the cookbook's separate metadata table, which now links to it) and
  rows marked *Unreleased* for every interface renamed during 3.0
  development without a compatibility path (band-reduction keywords,
  component source/target coordinates, prefixed and unreleased attributes,
  the interval attributes, STFT's bandwidth property, keyword-only settings
  of the new APIs, and the simulators). The cookbook's "Result schema"
  section is the single schema reference; it gains `output_schema_version`,
  `package`/`package_version`, and the `arg_<name>` views. Each row was
  checked by running the old and new forms.
- **Counts.** Both simulators raise `ValueError` naming `n_time_samples` or
  `n_trials` for a count that is not a positive integer (zero, negative,
  fractional, boolean); NumPy integers are accepted. `n_trials=None` stays
  the lagged simulator's 2-D form, and the shared-oscillation message says
  it has none. `simulate_MVAR` is released and is not changed.
- **Rounding a delay.** The docstring says that Python's `round` sends an
  exact half to the even neighbour and that a delay of half a sample or less
  becomes 0, once, in the `lags_samples` parameter text; the example converts
  10.2 ms at 1500 Hz to 15 samples and prints the 10 ms that simulates.
- **Migration completeness.** The migration rows were checked against a
  diff of every public name and signature between the 2.0.1 wheel and this
  branch and against the same calls run under both, which added rows for the
  default result's `coherency`, window-step rounding and the realized-window
  properties, the corrected confidence interval behind both
  `power_confidence_intervals` spellings, `squeeze=True` with several
  measures, `power_fisher_z_transform`'s `spectrum2` default, the removed
  `tridisolve`/`tridi_inverse_iteration`, `connectivity.EXPECTATION` and the
  module's re-imports, and `mt_summarize_parameters`. D2 and D28 record that
  only `Multitaper.frequency_resolution` follows a rounded window.
- **Schema reference test.** A test in `tests/test_result_schema.py` reads
  the cookbook's attribute table and checks both directions: every
  attribute a result of any transform, measure, or wrapper option carries is
  named there (`input_attrs_json` was the one missing), and every name there
  is carried by some probed result. It reads the `mt_*` keys only from
  their list sentence, so dropping one from the list fails it. Generating
  the table from one code record would remove the check's need to parse it;
  that is left for 3.x, as is validating `simulate_MVAR`'s counts.
- **D20 and copied channel metadata.** D20 adds `long_name` (and units where
  they apply) to the coordinates the package creates: band edges, connection
  labels, and `source`/`target`. Channel metadata copied from an input
  DataArray (e.g. `area` becoming `source_area`/`target_area`) keeps the attrs
  it came with and gains none: the package cannot know what a user's
  coordinate means, and a name-derived label would add no information. A
  default description could be added in 3.x without breaking anything.
- **Final-review fixes to earlier steps (owner's ruling).** Two items in
  code from steps 4 and 5 are fixed here, as their own commits, rather than
  by restacking those branches: the wrappers now warn about
  `coherence_magnitude` only for a call that succeeds, like the other
  deprecated spellings, and a plain `frequencies_of_interest=` to
  `connectivity_to_xarray` gets only its own deprecation (naming
  `frequency_range`), not also the plain-keyword one advising
  `connectivity_kwargs`. One test runs every deprecated spelling in a 2.x
  sequence and checks one warning per spelling at its own line.
- **Further migration rows.** The guide adds the debiased PLI and wPLI values
  where there is no phase lag (`-1/(n-1)` and NaN in 2.0.1, 0 now, including
  dsWPLI's diagonal) and read-only `Multitaper` settings; the window-step
  advice reproduces 2.x's windows exactly (checked on the wheel); the
  `ci=2c-1` equivalence states its range.
- **Issue #101 checkboxes.** In the proposed text every alias and key stays
  unticked until 4.0 removes it; a box means "removed", not "listed".
- **Tutorial outputs.** The three tutorials were re-executed with
  nbconvert so their stored outputs show the final attributes. The stored
  Dataset reprs still list the 15 released `mt_*` compatibility copies,
  because every 3.x Multitaper result carries them; the tutorials are not
  changed to hide them. Cells that ended with a plotting call end with `;`,
  so no output is a `<... at 0x...>` repr, and the notebooks are executed
  without per-cell timing metadata. What remains run-specific is the
  development `package_version` (until a release is tagged) and the random
  element ids of xarray's HTML repr, which every execution regenerates.

### Step 6 acceptance checklist

- [x] Documented shapes: both new simulators return `(time, trial, signal)`
  with one trial by default, `simulate_lagged_broadband(n_trials=None)`
  returns `(time, signal)`, and `simulate_MVAR` is unchanged
  (`tests/test_simulate.py`); the asserting tutorials pass when executed.
- [x] Seconds-to-samples conversion is explicit in the docstring doctest,
  the tutorial, and the migration table; `lags=` raises `TypeError`.
- [x] Every deprecated path in the code (searched for `DeprecationWarning`)
  is in the proposed issue #101 checklist with its replacement, first
  deprecation release, and warning tests: the `coherence_magnitude` names,
  `frequencies_of_interest`, the wrappers' plain-keyword settings,
  `power_confidence_intervals(n_tapers=)`, `Connectivity.from_multitaper`,
  and the 15 `mt_*` attribute keys.
- [ ] Issue #101 updated with that text (orchestrator).

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
