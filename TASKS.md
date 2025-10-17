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

- [ ] Change `xp.linalg.linalg.LinAlgError` to `xp.linalg.LinAlgError`
- [ ] Test with both NumPy and CuPy
- [ ] Verify no warnings in test suite

**Files:**

- `spectral_connectivity/minimum_phase_decomposition.py:78`

**Acceptance Criteria:**

- [ ] No deprecation warnings with NumPy 2.0+
- [ ] Works with both NumPy and CuPy

---

### Task 2.2: Improve Error Messages (WHAT/WHY/HOW Pattern)

- [ ] Audit all error messages in codebase
- [ ] Update error messages to answer:
  - [ ] WHAT: Clear statement of problem
  - [ ] WHY: Brief explanation of cause
  - [ ] HOW: Specific, actionable recovery steps
- [ ] Priority error messages to improve:
  - [ ] Detrend type validation (`transforms.py:968`)
  - [ ] Breakpoint validation (`transforms.py:980-982`)
  - [ ] GPU import error (already good, verify completeness)
  - [ ] expectation_type validation (enhance with order checking)
- [ ] Add order validation for `expectation_type`:
  - [ ] Detect if user used wrong order (e.g., "trials_time" instead of "time_trials")
  - [ ] Suggest correct order in error message
- [ ] Write tests:
  - [ ] `test_error_messages_are_helpful()` - verify message quality
  - [ ] `test_expectation_type_suggests_correct_order()`

**Files:**

- `spectral_connectivity/transforms.py:968,980-982`
- `spectral_connectivity/connectivity.py:226-232`
- All modules (audit)

**Acceptance Criteria:**

- [ ] All error messages follow WHAT/WHY/HOW pattern
- [ ] Users can recover from errors without reading source code
- [ ] expectation_type catches order mistakes

---

### Task 2.3: Add GPU Status Utility Function

- [ ] Create `get_compute_backend()` function in `__init__.py`:
  - [ ] Return dict with: backend, gpu_enabled, gpu_available, device_name, message
  - [ ] Check if cupy is in sys.modules
  - [ ] Check if cupy is importable (but don't import if not requested)
  - [ ] Provide helpful message for each case
- [ ] Update GPU initialization code to log device info
- [ ] Improve README documentation:
  - [ ] Add section on GPU setup with all 3 options (env var, script, notebook)
  - [ ] Include CuPy installation instructions
  - [ ] Add troubleshooting for common GPU issues
- [ ] Write tests:
  - [ ] `test_get_compute_backend_cpu()` - CPU mode
  - [ ] `test_get_compute_backend_gpu()` - GPU mode (if available)

**Files:**

- `spectral_connectivity/__init__.py`
- `README.md`
- `tests/test_gpu.py` (new file)

**Acceptance Criteria:**

- [ ] Users can check GPU status programmatically
- [ ] Clear documentation for all GPU setup methods
- [ ] Helpful error messages when GPU requested but unavailable

---

### Task 2.4: Add Tests for Block-wise Computation

- [ ] Create `test_expectation_cross_spectral_matrix_blocks()`:
  - [ ] Generate test data (10 time windows, 10 signals)
  - [ ] Compute connectivity with `blocks=None`
  - [ ] Compute connectivity with `blocks=2`, `blocks=3`, etc.
  - [ ] Verify all results match within floating-point tolerance
  - [ ] Test with different expectation_types
- [ ] Create `test_blocks_reduce_memory()`:
  - [ ] Use memory profiling to verify blocks reduce peak memory
  - [ ] Document memory reduction in test docstring
- [ ] Add documentation to `Connectivity` class:
  - [ ] Explain when to use `blocks` parameter
  - [ ] Document memory tradeoff (speed vs memory)
  - [ ] Provide rule of thumb (e.g., use blocks if n_signals > 50)

**Files:**

- `tests/test_connectivity.py`
- `spectral_connectivity/connectivity.py` (docstring)

**Acceptance Criteria:**

- [ ] Blocked and unblocked computation produce identical results
- [ ] Memory reduction verified (at least qualitatively)
- [ ] Documentation guides users when to use blocks

---

## Sprint 3: Parameter Helpers + Advanced Testing (Weeks 3-4)

**Goal:** Reduce learning curve and validate untested features
**Estimated Time:** 12-14 hours

### Task 3.1: Add Parameter Helper Functions

- [ ] Create utility functions in `transforms.py`:
  - [ ] `estimate_frequency_resolution(sampling_freq, window_duration, time_halfbandwidth_product)`
  - [ ] `estimate_n_tapers(time_halfbandwidth_product)`
  - [ ] `suggest_parameters(sampling_freq, signal_duration, desired_freq_resolution=None, desired_n_tapers=None)`
- [ ] Update `time_halfbandwidth_product` docstring:
  - [ ] Add formula for frequency resolution
  - [ ] Provide examples with common values
  - [ ] Explain formula to achieve target resolution
- [ ] Add `summarize_parameters()` method to `Multitaper`:
  - [ ] Print human-readable summary of all parameters
  - [ ] Show computed values (n_tapers, freq_resolution, n_windows)
  - [ ] Format nicely for terminal/notebook output
- [ ] Create alternative constructor (optional):
  - [ ] `Multitaper.from_target_resolution()` classmethod
  - [ ] Automatically calculates time_halfbandwidth_product
- [ ] Write tests:
  - [ ] `test_estimate_frequency_resolution()`
  - [ ] `test_estimate_n_tapers()`
  - [ ] `test_suggest_parameters()`
  - [ ] `test_summarize_parameters_output()`
  - [ ] `test_from_target_resolution()` (if implemented)
- [ ] Add examples to docstrings showing parameter selection

**Files:**

- `spectral_connectivity/transforms.py`
- `tests/test_transforms.py`

**Acceptance Criteria:**

- [ ] Users can estimate frequency resolution before computing
- [ ] Helper functions guide parameter selection
- [ ] Examples demonstrate common use cases
- [ ] All functions have tests with 100% coverage

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
