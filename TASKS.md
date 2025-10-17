# Implementation Plan: spectral_connectivity Improvements

**Based on:** UX Review + Code Review (2025-10-17)
**Total Estimated Time:** ~30 hours across 3 sprints
**Goal:** Address all critical and high-priority issues to improve user experience

---

## Sprint 1: Critical Input Validation (Week 1)

**Goal:** Prevent silent failures and cryptic errors
**Estimated Time:** 8-10 hours

### Task 1.1: Add Shape Validation to Connectivity Class

- [x] Add validation in `Connectivity.__init__()` for 5D fourier_coefficients
  - [x] Check `ndim == 5`, raise `ValueError` with helpful message if not
  - [x] Include expected shape format in error message
  - [x] Suggest using `Multitaper.fft()` if user has wrong shape
- [x] Add validation for minimum number of signals (n_signals >= 2)
- [x] Add warning for NaN/Inf values in fourier_coefficients
  - [x] Use `np.isfinite()` to check
  - [x] Provide actionable hints (check input data, windowing params)
- [x] Write tests for validation:
  - [x] `test_connectivity_rejects_wrong_ndim()` - test 1D, 2D, 3D, 4D, 6D inputs
  - [x] `test_connectivity_warns_on_nan()` - test NaN/Inf warnings
  - [x] `test_connectivity_requires_multiple_signals()` - test n_signals < 2

**Files:**

- `spectral_connectivity/connectivity.py:215`
- `tests/test_connectivity.py`

**Acceptance Criteria:**

- [x] All invalid shapes raise clear `ValueError` before computation
- [x] Error messages follow WHAT/WHY/HOW pattern
- [x] Tests achieve 100% coverage of validation paths

**Completed:** 2025-10-17

---

### Task 1.2: Clarify Array Dimension Handling in Multitaper

- [x] **Decision:** Choose approach (recommend: require explicit 3D)
  - [x] Option B chosen: Require 3D input, provide helper function
- [x] **Option B Implementation (require 3D + helper):**
  - [x] Add validation requiring `ndim == 3` in `__init__`
  - [x] Create `prepare_time_series()` helper function
  - [x] Document reshaping patterns in error message
  - [x] Add examples showing np.newaxis usage
- [x] Add visual emphasis to docstrings:
  - [x] Use bold for dimension order explanations
  - [x] Add "Common mistake" warning boxes
  - [x] Include shape examples for typical use cases
  - [x] Add realistic neuroscience examples (EEG, LFP, channels, trials)
  - [x] Reorder examples to show helper function first (easier way)
- [x] Write tests:
  - [x] `test_multitaper_dimension_consistency()` - verify 3D behavior
  - [x] `test_prepare_time_series_single_trial()` - test 2D → 3D with axis='signals'
  - [x] `test_prepare_time_series_single_signal()` - test 2D → 3D with axis='trials'
  - [x] `test_prepare_time_series_1d()` - test 1D → 3D conversion
  - [x] `test_prepare_time_series_3d_passthrough()` - test 3D pass-through
  - [x] `test_prepare_time_series_invalid_axis()` - test invalid axis
  - [x] `test_prepare_time_series_requires_axis_for_2d()` - test missing axis
  - [x] `test_multitaper_requires_3d_input()` - test validation for 1D, 2D, 4D

**Files:**

- `spectral_connectivity/transforms.py` - Lines 42-50, 97-125, 143-182, 489-612
- `spectral_connectivity/__init__.py` - Added `prepare_time_series` to exports
- `tests/test_transforms.py` - Lines 165-177, 344-453

**Acceptance Criteria:**

- [x] No ambiguity in dimension interpretation
- [x] Clear error messages for wrong shapes
- [x] Documentation explicitly warns about common mistakes
- [x] Tests verify consistent behavior
- [x] Applied spectral-code-reviewer and scientific-ux-reviewer agents
- [x] Addressed all reviewer feedback

**Completed:** 2025-10-17

---

### Task 1.3: Add Parameter Validation to Multitaper

- [x] Validate `sampling_frequency > 0`
  - [x] Raise `ValueError` with suggested common values
- [x] Validate `time_halfbandwidth_product >= 1`
  - [x] Explain physical meaning in error message
  - [x] Suggest typical ranges (1-5)
  - [x] Add warning for unusually large values (> 10)
- [x] Validate `time_window_duration > 0` (if provided)
- [x] Validate `time_window_step > 0` (if provided)
  - [x] Add warning when step > duration (creates gaps)
- [x] Add warning for likely transposed data:
  - [x] Check if `shape[0] < shape[2]` for 3D arrays (n_time < n_signals)
  - [x] Warn user they may need to transpose
- [x] Add warning for NaN/Inf in input time_series
  - [x] Suggest interpolation, artifact removal, preprocessing checks
- [x] Write tests:
  - [x] `test_multitaper_rejects_negative_sampling_freq()`
  - [x] `test_multitaper_rejects_invalid_time_halfbandwidth()`
  - [x] `test_multitaper_rejects_negative_time_window_duration()`
  - [x] `test_multitaper_rejects_negative_time_window_step()`
  - [x] `test_multitaper_warns_likely_transposed()`
  - [x] `test_multitaper_warns_on_nan_input()`
  - [x] `test_multitaper_warns_on_large_time_halfbandwidth()`
  - [x] `test_multitaper_warns_on_step_larger_than_duration()`

**Files:**

- `spectral_connectivity/transforms.py` - Lines 201-342
- `tests/test_transforms.py` - Lines 457-577

**Acceptance Criteria:**

- [x] All invalid parameters caught before computation
- [x] Error messages provide context and suggestions
- [x] Warnings guide users to fix data issues
- [x] 100% coverage of validation paths
- [x] Applied spectral-code-reviewer and scientific-ux-reviewer agents
- [x] Addressed all reviewer feedback

**Completed:** 2025-10-17

---

## Sprint 2: Quick Wins + Testing (Week 2)

**Goal:** High-impact, low-effort improvements
**Estimated Time:** 8-10 hours

### Task 2.1: Fix Deprecation Warning

- [x] Change `xp.linalg.linalg.LinAlgError` to `xp.linalg.LinAlgError`
- [x] Test with NumPy (all tests pass: 6/6 in test_minimum_phase_decomposition.py, 168/168 overall)
- [x] Verify no warnings in test suite (ran with -W error::DeprecationWarning)

**Files:**

- `spectral_connectivity/minimum_phase_decomposition.py:78`

**Acceptance Criteria:**

- [x] No deprecation warnings with NumPy 2.0+
- [x] Works with both NumPy and CuPy

**Completed:** 2025-10-17

---

### Task 2.2: Improve Error Messages (WHAT/WHY/HOW Pattern)

- [x] Audit all error messages in codebase
- [x] Update error messages to answer:
  - [x] WHAT: Clear statement of problem
  - [x] WHY: Brief explanation of cause
  - [x] HOW: Specific, actionable recovery steps
- [x] Priority error messages to improve:
  - [x] Detrend type validation (`transforms.py:1296-1303`)
  - [x] Breakpoint validation (`transforms.py:1314-1335`)
  - [x] GPU import error (already good, verified completeness)
  - [x] expectation_type validation (enhanced with order checking)
- [x] Add order validation for `expectation_type`:
  - [x] Detect if user used wrong order (e.g., "tapers_trials" instead of "trials_tapers")
  - [x] Suggest correct order in error message
- [x] Write tests:
  - [x] `test_error_messages_are_helpful()` - verify message quality
  - [x] `test_expectation_type_suggests_correct_order()`

**Files:**

- `spectral_connectivity/transforms.py` - Lines 1296-1303, 1314-1335
- `spectral_connectivity/connectivity.py` - Lines 248-283
- `tests/test_error_messages.py` - New comprehensive test suite (199 lines, 9 tests)
- `tests/test_expectation_validation.py` - Updated test assertion

**Acceptance Criteria:**

- [x] All error messages follow WHAT/WHY/HOW pattern
- [x] Users can recover from errors without reading source code
- [x] expectation_type catches order mistakes
- [x] All tests pass (177/177)
- [x] Code quality gates pass (ruff, black, mypy)
- [x] Applied spectral-code-reviewer and scientific-ux-reviewer agents

**Completed:** 2025-10-17

---

### Task 2.3: Add GPU Status Utility Function

- [x] Create `get_compute_backend()` function in `utils.py`:
  - [x] Return dict with: backend, gpu_enabled, gpu_available, device_name, message
  - [x] Check if cupy is in sys.modules
  - [x] Check if cupy is importable (but don't import if not requested)
  - [x] Provide helpful message for each case (4 different scenarios)
  - [x] Show actual GPU model name (e.g., "NVIDIA Tesla V100") instead of compute capability
  - [x] Add example return value to docstring
- [x] Update GPU initialization code to log device info
  - [x] Enhanced logging in transforms.py to show GPU model name
  - [x] Enhanced logging in connectivity.py to show GPU model name
- [x] Improve README documentation:
  - [x] Add comprehensive GPU Acceleration section (130+ lines)
  - [x] Include all 3 setup options (env var shell, script, notebook)
  - [x] Add verification steps to each setup example
  - [x] Simplified CuPy installation instructions (conda recommended)
  - [x] Add troubleshooting for 4 common GPU issues
  - [x] Explain import timing requirement (why "before importing" matters)
  - [x] Add example outputs to code examples
  - [x] Include kernel restart guidance for notebooks
  - [x] Provide guidance on when to use GPU acceleration
- [x] Write comprehensive tests:
  - [x] 13 test methods covering all scenarios
  - [x] `test_cpu_mode_default()` - CPU mode default
  - [x] `test_cpu_mode_explicit_false()` - CPU mode explicit
  - [x] `test_gpu_mode_when_cupy_available()` - GPU mode with CuPy
  - [x] `test_gpu_mode_when_cupy_not_available()` - GPU mode without CuPy
  - [x] `test_return_value_structure()` - Validate return dict structure
  - [x] `test_backend_values()` - Validate backend field values
  - [x] `test_boolean_fields()` - Validate boolean types
  - [x] `test_device_name_present()` - Validate device_name field
  - [x] `test_message_is_helpful()` - Validate message quality
  - [x] `test_detect_cupy_import_state()` - Import state detection
  - [x] `test_environment_variable_respected()` - Env var handling
  - [x] `test_cpu_backend_details()` - CPU backend details
  - [x] Additional integration tests for consistency
- [x] Apply spectral-code-reviewer and scientific-ux-reviewer agents
- [x] Address all UX feedback (GPU model names, import timing, examples)

**Files:**

- `spectral_connectivity/utils.py` - NEW module with `get_compute_backend()`
- `spectral_connectivity/__init__.py` - Import and re-export from utils
- `spectral_connectivity/transforms.py` - Enhanced GPU device logging (lines 19-32)
- `spectral_connectivity/connectivity.py` - Enhanced GPU device logging (lines 34-47)
- `README.md` - Comprehensive GPU Acceleration section (lines 109-260)
- `tests/test_gpu.py` - NEW comprehensive test suite (151 lines, 13 tests)

**Acceptance Criteria:**

- [x] Users can check GPU status programmatically
- [x] Clear documentation for all GPU setup methods
- [x] Helpful error messages when GPU requested but unavailable
- [x] GPU device shows actual model name (e.g., "NVIDIA Tesla V100-SXM2-16GB")
- [x] Import timing requirement clearly explained
- [x] All setup examples include verification steps
- [x] All tests pass (11 passed, 1 skipped when CuPy unavailable)
- [x] Code quality gates pass (ruff, black, mypy)

**Completed:** 2025-10-17

---

### Task 2.4: Add Tests for Block-wise Computation

- [x] Create `test_expectation_cross_spectral_matrix_blocks()`:
  - [x] Generate test data (10 time windows, 10 signals)
  - [x] Compute connectivity with `blocks=None`
  - [x] Compute connectivity with `blocks=2`, `blocks=3`, etc.
  - [x] Verify all results match within floating-point tolerance
  - [x] Test with different expectation_types
- [x] Create `test_blocks_reduce_memory()`:
  - [x] Use memory profiling to verify blocks reduce peak memory
  - [x] Document memory reduction in test docstring
- [x] Add documentation to `Connectivity` class:
  - [x] Explain when to use `blocks` parameter
  - [x] Document memory tradeoff (speed vs memory)
  - [x] Provide rule of thumb (e.g., use blocks if n_signals > 50)

**Files:**

- `tests/test_connectivity.py` - Lines 725-976 (5 new tests)
- `spectral_connectivity/connectivity.py` - Lines 187-223 (enhanced docstring), Lines 475, 502 (bug fixes)

**Acceptance Criteria:**

- [x] Blocked and unblocked computation produce identical results
- [x] Memory reduction verified (73% reduction for n_signals=50, blocks=5)
- [x] Documentation guides users when to use blocks
- [x] Bug fixes: diagonal elements and Hermitian symmetry
- [x] Applied spectral-code-reviewer and scientific-ux-reviewer agents

**Completed:** 2025-10-17

---

## Sprint 3: Parameter Helpers + Advanced Testing (Weeks 3-4)

**Goal:** Reduce learning curve and validate untested features
**Estimated Time:** 12-14 hours

### Task 3.1: Add Parameter Helper Functions

- [x] Create utility functions in `transforms.py`:
  - [x] `estimate_frequency_resolution(sampling_freq, window_duration, time_halfbandwidth_product)`
  - [x] `estimate_n_tapers(time_halfbandwidth_product)`
  - [x] `suggest_parameters(sampling_freq, signal_duration, desired_freq_resolution=None, desired_n_tapers=None)`
  - [x] Added MultitaperParameters TypedDict for type-safe return values
- [x] Update `time_halfbandwidth_product` docstring:
  - [x] Add formula for frequency resolution
  - [x] Provide examples with common values (NW=2,3,4,5+)
  - [x] Explain formula to achieve target resolution
  - [x] Include practical guidance on parameter selection
- [x] Add `summarize_parameters()` method to `Multitaper`:
  - [x] Print human-readable summary of all parameters
  - [x] Show computed values (n_tapers, freq_resolution, n_windows)
  - [x] Format nicely for terminal/notebook output
  - [x] Show overlap percentage for windowing
- [x] Write comprehensive tests:
  - [x] `test_estimate_frequency_resolution()` - 6 tests covering basic calc, effects, consistency
  - [x] `test_estimate_n_tapers()` - 5 tests covering basic calc, different NW values, consistency
  - [x] `test_suggest_parameters()` - 8 tests covering defaults, targets, conflicts, errors, EEG/LFP scenarios
  - [x] `test_summarize_parameters()` - 3 tests covering method existence, format, content
  - [x] Total: 22 comprehensive tests, all passing
- [x] Add examples to docstrings showing parameter selection
  - [x] Domain-specific examples (EEG, LFP applications)
  - [x] Real-world use cases with expected outputs
- [x] Code quality checks:
  - [x] All tests pass (215 total, 22 new)
  - [x] Ruff linting passes
  - [x] Black formatting applied
  - [x] Type hints complete with TypedDict
- [x] Review agents applied:
  - [x] spectral-code-reviewer: Fixed critical TypedDict, MyPy, and Ruff issues
  - [x] scientific-ux-reviewer: Confirmed production-ready UX
- [x] Updated exports in `__init__.py`

**Files:**

- `spectral_connectivity/transforms.py` - Lines 15-43 (TypedDict), 45-123 (estimate_frequency_resolution), 126-178 (estimate_n_tapers), 181-384 (suggest_parameters), 407-430 (enhanced docstring), 790-903 (summarize_parameters)
- `tests/test_parameter_helpers.py` - NEW comprehensive test suite (294 lines, 22 tests)
- `spectral_connectivity/__init__.py` - Updated exports

**Acceptance Criteria:**

- [x] Users can estimate frequency resolution before computing
- [x] Helper functions guide parameter selection
- [x] Examples demonstrate common use cases (EEG, LFP)
- [x] All functions have tests with 100% coverage
- [x] Type-safe with MultitaperParameters TypedDict
- [x] Documentation follows NumPy style with practical examples
- [x] Error messages follow WHAT/WHY/HOW pattern

**Completed:** 2025-10-17

**Note:** Deferred alternative constructor `Multitaper.from_target_resolution()` - not needed given `suggest_parameters()` provides equivalent functionality with more flexibility.

---

### Task 3.2: Test Advanced Connectivity Measures

- [ ] Test `canonical_coherence()`:
  - [ ] Create synthetic data with known brain area structure
  - [ ] Verify output shape matches expectations
  - [ ] Test with different brain_area_labels configurations
  - [ ] Verify diagonal handling
  - [ ] Compare with manual computation for small example
- [ ] Test `global_coherence()`:
  - [ ] Create synthetic data with known coherence structure
  - [ ] Verify against manual SVD computation
  - [ ] Test output shape and value ranges
  - [ ] Test edge cases (single signal, perfect coherence)
- [ ] Test `group_delay()`:
  - [ ] Create signals with known phase relationships
  - [ ] Verify group delay calculation
  - [ ] Test linear regression component
  - [ ] Verify output ranges and shapes
- [ ] Add integration tests:
  - [ ] Test workflow: Multitaper � Connectivity � advanced measures
  - [ ] Verify xarray output from wrapper functions

**Files:**

- `tests/test_connectivity.py`
- `tests/test_wrapper.py`

**Acceptance Criteria:**

- [ ] All advanced measures have at least 2 tests each
- [ ] Tests verify correct computation with known inputs
- [ ] Edge cases covered (single signal, perfect sync, etc.)
- [ ] Coverage of connectivity.py increases to 80%+

---

### Task 3.3: Resolve TODO Comments

- [ ] **wrapper.py:230** - "TODO is there a better way to get all Connectivity methods?"
  - [ ] Replace `dir()` with `inspect.getmembers()`
  - [ ] Filter for public methods using `inspect.ismethod()`
  - [ ] Test that it finds all expected methods
- [ ] **statistics.py:139** - "TODO: add axis keyword?"
  - [ ] Add `axis` parameter to `adjust_for_multiple_comparisons()`
  - [ ] Implement using `np.apply_along_axis()` if axis specified
  - [ ] Update docstring with axis parameter
  - [ ] Add tests for multi-dimensional corrections
  - [ ] OR: Remove TODO if not needed, document decision
- [ ] Remove or implement all TODO comments:
  - [ ] Search codebase for remaining TODOs
  - [ ] Create issues for deferred features
  - [ ] Remove stale comments

**Files:**

- `spectral_connectivity/wrapper.py:230`
- `spectral_connectivity/statistics.py:139`
- All modules (search for TODO)

**Acceptance Criteria:**

- [ ] No TODO comments in codebase (or all documented in issues)
- [ ] `inspect.getmembers()` properly finds all methods
- [ ] axis parameter works correctly (if implemented)
- [ ] Tests verify new functionality

---

## Future TODOs (Low Priority - Not in Current Plan)

### Code Quality Improvements

- [ ] Increase test coverage to 85%+
  - Focus on `connectivity.py` (currently 71%)
  - Add edge case tests (empty arrays, single signals)
  - Test all error paths
- [ ] Enable stricter MyPy settings
  - Set `disallow_untyped_defs = true`
  - Add missing type hints to helper functions
  - Enable strict mode incrementally
- [ ] Extract magic numbers to named constants
  - `MIN_EIGENVALUE_THRESHOLD = 0.9`
  - `TAPER_MULTIPLIER = 2.0`
  - Document scientific rationale

### UX Enhancements

- [ ] Add memory estimation utility
  - Create `estimate_memory_usage()` function
  - Document memory requirements in README
  - Add warnings for large allocations
- [ ] Add progress indication for long operations
  - Add optional `progress` parameter to slow methods
  - Use `tqdm` for progress bars
  - Fall back to logging if tqdm unavailable
- [ ] Add "See Also" sections to all docstrings
  - Link related connectivity measures
  - Cross-reference alternative methods
- [ ] Add more usage examples
  - MNE integration example
  - pandas DataFrame example
  - Neo AnalogSignal example
- [ ] Improve NaN handling consistency
  - Document NaN behavior for all methods
  - Add sanity checks for value ranges
  - Clip coherence to [0,1] with warning if needed

### Refactoring

- [ ] Consider splitting `connectivity.py`
  - Create `functional_connectivity.py`
  - Create `directed_connectivity.py`
  - Keep base class in `connectivity.py`
  - **Note:** This is a breaking change, requires careful design
- [ ] Add performance benchmarks
  - CPU vs GPU comparisons
  - Memory profiling
  - Scaling analysis (N signals, M time points)
- [ ] Refactor long functions
  - Extract `_compute_phase_regression()` from `group_delay()`
  - Consider breaking up complex methods

---

## Success Metrics

### After Sprint 1 (Critical Validation)

- [ ] Zero silent failures due to wrong input shapes
- [ ] All parameter errors caught before computation starts
- [ ] Error messages guide users to solutions

### After Sprint 2 (Quick Wins + Testing)

- [ ] Zero deprecation warnings
- [ ] GPU setup clearly documented and testable
- [ ] Block-wise computation validated
- [ ] Error messages follow consistent pattern

### After Sprint 3 (Helpers + Advanced Tests)

- [ ] Users can easily select appropriate parameters
- [ ] All connectivity measures have tests
- [ ] Test coverage e 80%
- [ ] No TODO comments in codebase

### Overall Goal

- [ ] **Expert users:** Even better experience (already good)
- [ ] **Intermediate users:** Move from frustrated to productive
- [ ] **Novice users:** Move from blocked to learning successfully
- [ ] **Package rating:** Move from "good for experts" to "excellent for everyone"

---

## Notes

- **Testing Strategy:** Write tests alongside each implementation (not after)
- **Documentation:** Update docstrings as code changes, not as separate task
- **Review:** Each sprint should end with code review before merging
- **User Testing:** Consider getting feedback from real users after Sprint 1
- **Breaking Changes:** Dimension handling (Task 1.2) may be breaking - consider deprecation path

---

**Last Updated:** 2025-10-17
**Status:** Ready to begin Sprint 1
