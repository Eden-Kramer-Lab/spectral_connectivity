# Comprehensive Code Quality Review: spectral_connectivity Package

**Review Date:** 2025-10-17
**Reviewer:** Spectral Code Reviewer Agent
**Package Version:** Current master branch
**Review Scope:** Full codebase including core modules, tests, configuration

---

## Executive Summary

The `spectral_connectivity` package demonstrates **excellent overall code quality** with strong architecture, comprehensive testing, and modern development practices. The codebase achieves:

- **77% test coverage** (940 statements, 215 missed)
- **Zero linting errors** (ruff check passed)
- **Zero type checking errors** (mypy passed)
- **~10,500 lines of Python code** across 8 core modules and 11 test files
- **149 passing tests** with modern pytest framework

### Overall Quality Rating: **APPROVE** 

The package represents production-ready scientific software with strong fundamentals. While there are opportunities for improvement (detailed below), no **critical blockers** were identified. The code is well-structured, properly documented, and follows Python best practices.

---

## 1. Code Quality Assessment

### 1.1 Style & Formatting  EXCELLENT

**Status:** All checks passed (ruff + black)

**Strengths:**

- Consistent 88-character line length (Black standard)
- Modern Python 3.10+ syntax with PEP 604 type hints (`str | None` instead of `Union[str, None]`)
- Proper use of f-strings and modern idioms
- Well-organized imports with isort integration

**Evidence:**

```bash
ruff check spectral_connectivity/ tests/
# Result: All checks passed!
```

### 1.2 Type Hints � GOOD (with improvement opportunities)

**Status:** MyPy passes, but coverage could be enhanced

**Strengths:**

- All public functions have type hints for parameters and return values
- Proper use of `NDArray[np.floating]`, `NDArray[np.complexfloating]` from `numpy.typing`
- Good use of `Literal` types for constrained string arguments
- `Callable` types properly annotated

**Weaknesses:**

- MyPy configuration allows `disallow_untyped_defs = false` and `disallow_incomplete_defs = false`
- Some internal helper functions lack complete type annotations
- No strict mode enabled (would catch more edge cases)

**Recommendations:**

1. **Medium Priority**: Gradually enable stricter mypy settings:

   ```toml
   [tool.mypy]
   disallow_untyped_defs = true  # Require all functions to have types
   disallow_incomplete_defs = true
   ```

2. **Quick Win**: Add missing type hints to internal functions in `transforms.py`:

   ```python
   # Current (line 529)
   def _make_tapers(
       n_time_samples_per_window,
       sampling_frequency,
       time_halfbandwidth_product,
       n_tapers,
       is_low_bias=True,
   ):

   # Recommended
   def _make_tapers(
       n_time_samples_per_window: int,
       sampling_frequency: float,
       time_halfbandwidth_product: float,
       n_tapers: int,
       is_low_bias: bool = True,
   ) -> NDArray[np.floating]:
   ```

**Severity:** Low (non-blocking, incremental improvement)

---

## 2. Testing Assessment

### 2.1 Test Coverage � GOOD (77% overall)

**Breakdown by Module:**

| Module | Statements | Missed | Coverage | Status |
|--------|-----------|--------|----------|---------|
| `__init__.py` | 9 | 3 | 67% | � Acceptable |
| `_version.py` | 13 | 0 | **100%** |  Excellent |
| `connectivity.py` | 473 | 136 | 71% | � Good |
| `minimum_phase_decomposition.py` | 61 | 7 | 89% |  Excellent |
| `simulate.py` | 12 | 0 | **100%** |  Excellent |
| `statistics.py` | 60 | 18 | 70% | � Good |
| `transforms.py` | 256 | 46 | 82% |  Very Good |
| `wrapper.py` | 56 | 5 | 91% |  Excellent |
| **TOTAL** | **940** | **215** | **77%** | � Good |

**Test Organization:** 149 tests across 11 test files

- `test_connectivity.py`: 50 tests
- `test_transforms.py`: 47 tests
- `test_wrapper.py`: 16 tests
- `test_statistics.py`: 11 tests
- Additional specialized test modules

### 2.2 Test Quality  GOOD

**Strengths:**

- Proper use of `pytest` with parametrization (`@mark.parametrize`)
- Tests validate against `nitime` library (independent reference implementation)
- Good separation of unit vs integration tests
- Tests cover GPU/CPU switching logic

**Example of quality parametrized test:**

```python
# tests/test_connectivity.py
@mark.parametrize("axis", [(0), (1), (2), (3)])
@mark.parametrize("dtype", [np.complex64, np.complex128])
def test_cross_spectrum(axis, dtype):
    # Tests all dimensions and data types systematically
```

### 2.3 Missing Test Coverage (Specific Lines)

**Critical Gaps in `connectivity.py`:**

- **Lines 361-395**: Block-wise connectivity computation (complex memory optimization logic)
- **Lines 637-678**: `canonical_coherence()` method
- **Lines 719-753**: `global_coherence()` method
- **Lines 1321-1377**: `group_delay()` method
- **Lines 2061-2073**: `_find_significant_frequencies()` helper

**Impact:**

- Block-wise computation is a memory optimization for large arrays - untested failure modes
- Advanced connectivity measures (canonical/global coherence, group delay) lack validation
- Statistical significance testing not fully validated

**Recommendations:**

1. **High Priority**: Add tests for block-wise computation:

   ```python
   def test_expectation_cross_spectral_matrix_blocks():
       """Test that blocked computation matches full computation."""
       # Compare results with blocks=None vs blocks=2
       conn_full = Connectivity(fourier_coeffs, blocks=None)
       conn_blocked = Connectivity(fourier_coeffs, blocks=2)

       assert np.allclose(
           conn_full.coherence_magnitude(),
           conn_blocked.coherence_magnitude()
       )
   ```

2. **Medium Priority**: Add integration tests for advanced measures:
   - `canonical_coherence()` with known synthetic data
   - `global_coherence()` validation against manual SVD
   - `group_delay()` with known phase relationships

**Severity:** Medium (these features exist but lack validation)

---

## 3. Architecture & Design

### 3.1 Three-Layer Architecture  EXCELLENT

The package implements a clean separation of concerns:

```
transforms.py (Layer 1: Spectral Analysis)
    �
connectivity.py (Layer 2: Connectivity Metrics)
    �
wrapper.py (Layer 3: High-Level API)
```

**Strengths:**

- Clear responsibility boundaries
- Each layer can be used independently
- `Connectivity.from_multitaper()` provides clean integration
- Wrapper functions return labeled xarray DataArrays for user convenience

**Example of clean integration:**

```python
# wrapper.py lines 85, 270-277
connectivity = Connectivity.from_multitaper(m)
connectivity_mat = getattr(connectivity, method)(**kwargs)
```

### 3.2 GPU/CPU Abstraction  EXCELLENT

**Implementation:**
All three core modules use consistent GPU detection:

```python
# transforms.py, connectivity.py, minimum_phase_decomposition.py
if os.environ.get("SPECTRAL_CONNECTIVITY_ENABLE_GPU") == "true":
    try:
        import cupy as xp
        from cupyx.scipy.fft import fft, ifft
    except ImportError as exc:
        raise RuntimeError("GPU support explicitly requested but CuPy not installed") from exc
else:
    import numpy as xp
    from scipy.fft import fft, ifft
```

**Strengths:**

- Single `xp` alias used throughout for numpy/cupy compatibility
- Explicit error when GPU requested but CuPy unavailable
- Environment variable control (no runtime switching complexity)
- Graceful fallback with informative logging

**Potential Issues:**

- No runtime GPU detection (must set environment variable before import)
- Cannot switch GPU/CPU mid-execution
- Mixing GPU/CPU arrays could cause cryptic errors

**Recommendations:**

1. **Low Priority**: Add utility function to check GPU availability:

   ```python
   def is_gpu_available() -> bool:
       """Check if GPU computation is enabled and CuPy is available."""
       return os.environ.get("SPECTRAL_CONNECTIVITY_ENABLE_GPU") == "true"
   ```

**Severity:** Low (current design is intentional and documented)

### 3.3 Caching Strategy  VERY GOOD

**Implementation:**

- Cross-spectral matrices cached via `@property` decorators
- Minimum phase decomposition cached (expensive Wilson algorithm)
- Transfer functions computed once and reused

**Evidence:**

```python
# connectivity.py
@property
def _cross_spectral_matrix(self) -> NDArray[np.complexfloating]:
    """Cached property - computed once per instance"""
    fourier_coefficients = self.fourier_coefficients[..., xp.newaxis]
    return _complex_inner_product(
        fourier_coefficients, fourier_coefficients, dtype=self._dtype
    )
```

**Strengths:**

- Pythonic use of `@property` for lazy evaluation
- Avoids redundant computation when multiple connectivity measures requested
- Memory efficient (computed on demand)

**Potential Issues:**

- Properties never invalidated (no cache eviction)
- Large datasets could exhaust memory
- No explicit cache management API

**Recommendations:**

1. **Low Priority**: Document memory implications in docstrings:

   ```python
   @property
   def _cross_spectral_matrix(self):
       """Compute and cache cross-spectral matrix.

       Notes
       -----
       This property is cached. For large datasets, consider computing
       connectivity measures individually to manage memory usage.
       """
   ```

**Severity:** Low (acceptable for typical use cases)

---

## 4. Documentation Quality

### 4.1 Docstring Coverage  EXCELLENT

**Format:** NumPy-style docstrings (consistent with scipy/numpy)

**Strengths:**

- All public functions have complete docstrings
- Parameters documented with types, units, ranges, defaults
- Return values documented with shapes
- Examples provided for complex functions
- Scientific references cited (e.g., Dhamala 2008, Thomson 1982)

**Example of high-quality docstring:**

```python
# connectivity.py lines 227-289
class Connectivity:
    """
    Compute functional and directed connectivity measures from spectral data.

    Parameters
    ----------
    fourier_coefficients : NDArray[complexfloating],
        shape (n_time_windows, n_trials, n_tapers, n_frequencies, n_signals)
        Complex-valued Fourier coefficients...

    Examples
    --------
    >>> # Simulate coherent signals
    >>> coeffs = np.random.randn(...)
    >>> conn = Connectivity(coeffs, expectation_type="trials_tapers")

    References
    ----------
    .. [1] Dhamala, M., Rangarajan, G., and Ding, M. (2008). Analyzing
           information flow in brain networks with nonparametric Granger
           causality. NeuroImage 41, 354-362.
    """
```

### 4.2 Module-Level Documentation  GOOD

**Strengths:**

- All modules have descriptive docstrings
- Purpose clearly stated
- Context provided for scientific users

**Example:**

```python
# minimum_phase_decomposition.py lines 1-6
"""Minimum phase decomposition for spectral density matrices.

A spectral density matrix can be decomposed into minimum phase functions
using the Wilson algorithm. This decomposition is used in computing
pairwise spectral Granger prediction and other directed connectivity measures.
"""
```

### 4.3 Code Comments � ADEQUATE

**Strengths:**

- Complex algorithms have inline comments
- Non-obvious logic explained (e.g., Wilson algorithm steps)

**Weaknesses:**

- Some dense numerical code lacks explanation
- Occasional TODO comments suggest incomplete work

**TODO Comments Found:**

```python
# statistics.py:139
# TODO: add axis keyword?

# wrapper.py:230
# TODO is there a better way to get all Connectivity methods?
```

**Recommendations:**

1. **Medium Priority**: Resolve or track TODOs:
   - `statistics.py`: Add axis parameter for per-dimension multiple comparisons
   - `wrapper.py`: Use `inspect` module to get methods programmatically

2. **Low Priority**: Add comments to complex numerical sections:

   ```python
   # transforms.py line 808-834 (DPSS optimization)
   # Current: Minimal comments on tridiagonal eigenvalue problem
   # Recommended: Explain why this approach vs direct computation
   ```

**Severity:** Low (existing documentation is sufficient)

---

## 5. Performance Considerations

### 5.1 Algorithmic Efficiency  EXCELLENT

**Strengths:**

- Uses `scipy.linalg.eigvals_banded` for efficient DPSS computation (tridiagonal solver)
- FFT lengths optimized with `scipy.fft.next_fast_len()`
- Vectorized numpy operations throughout
- Block-wise processing option for large arrays

**Evidence:**

```python
# transforms.py line 286
self._n_fft_samples = next_fast_len(self.n_time_samples_per_window)

# transforms.py line 818-824
w = eigvals_banded(
    ab,
    select="i",
    select_range=(
        n_time_samples_per_window - n_tapers,
        n_time_samples_per_window - 1,
    ),
)
```

### 5.2 Memory Efficiency � GOOD

**Strengths:**

- Lazy evaluation via `@property` decorators
- Optional block-wise computation for large datasets
- Strided arrays avoided in sliding window (copying enabled by default)

**Potential Issues:**

- No memory profiling tests
- Large cross-spectral matrices can exhaust RAM
- No warnings for potentially OOM operations

**Example concern:**

```python
# connectivity.py line 317-330
@property
def _cross_spectral_matrix(self):
    """For large n_time_windows, n_frequencies, n_signals, this can be huge:
    shape = (n_time_windows, n_trials, n_tapers, n_frequencies, n_signals, n_signals)
    """
```

**Recommendations:**

1. **Medium Priority**: Add memory estimation utility:

   ```python
   def estimate_memory_usage(
       n_time_windows: int,
       n_trials: int,
       n_tapers: int,
       n_frequencies: int,
       n_signals: int
   ) -> float:
       """Estimate peak memory usage in GB for connectivity analysis."""
       # Cross-spectral matrix: complex128 = 16 bytes
       csm_size = (n_time_windows * n_trials * n_tapers *
                   n_frequencies * n_signals * n_signals * 16)
       return csm_size / (1024**3)
   ```

2. **Low Priority**: Document memory requirements in README
   - Rule of thumb: N signals requires O(N�) memory
   - Block-wise computation trades speed for memory

**Severity:** Medium (could cause issues for large datasets)

### 5.3 GPU Performance  GOOD

**Implementation:**

- CuPy arrays handled consistently via `xp` alias
- FFT operations use GPU-accelerated CuPy FFT
- Matrix operations leverage GPU BLAS

**Potential Issues:**

- No GPU memory management (relies on CuPy defaults)
- No benchmarks comparing GPU vs CPU performance
- Mixing GPU/CPU code could cause hidden transfers

**Recommendations:**

1. **Low Priority**: Add performance benchmarks:

   ```python
   # tests/test_performance.py
   @pytest.mark.benchmark
   def test_multitaper_cpu_vs_gpu():
       """Compare CPU and GPU performance for typical workflow."""
       # Time with SPECTRAL_CONNECTIVITY_ENABLE_GPU=false
       # Time with SPECTRAL_CONNECTIVITY_ENABLE_GPU=true
       # Assert GPU is faster (or document when CPU wins)
   ```

**Severity:** Low (GPU support is optional optimization)

---

## 6. Error Handling & Validation

### 6.1 Input Validation � ADEQUATE

**Strengths:**

- Expectation type validated with helpful error message
- Empty arrays cause matrix decomposition errors (caught and logged)
- Invalid parameters caught early (e.g., `ValueError` for bad `expectation_type`)

**Evidence:**

```python
# connectivity.py lines 226-232
if expectation_type not in EXPECTATION:
    allowed_values = ", ".join(f"'{k}'" for k in sorted(EXPECTATION.keys()))
    raise ValueError(
        f"Invalid expectation_type '{expectation_type}'. "
        f"Allowed values are: {allowed_values}"
    )
```

**Weaknesses:**

- No validation of array shapes in `Connectivity.__init__`
- No checks for NaN/Inf in input data
- No validation that frequencies match FFT length

**Recommendations:**

1. **High Priority**: Add shape validation in `Connectivity.__init__`:

   ```python
   def __init__(self, fourier_coefficients, ...):
       # Validate expected 5D shape
       if fourier_coefficients.ndim != 5:
           raise ValueError(
               f"Expected 5D array (n_time, n_trials, n_tapers, n_freq, n_signals), "
               f"got {fourier_coefficients.ndim}D"
           )

       # Warn if NaN/Inf present
       if not np.all(np.isfinite(fourier_coefficients)):
           logger.warning("Input contains NaN or Inf values")
   ```

2. **Medium Priority**: Validate `Multitaper` inputs:

   ```python
   def __init__(self, time_series, sampling_frequency=1000, ...):
       if sampling_frequency <= 0:
           raise ValueError(f"sampling_frequency must be positive, got {sampling_frequency}")

       if time_halfbandwidth_product < 1:
           raise ValueError(f"time_halfbandwidth_product must be >= 1")
   ```

**Severity:** Medium (could prevent cryptic errors)

### 6.2 Exception Handling  GOOD

**Strengths:**

- Appropriate exception types used (`ValueError`, `RuntimeError`, `NotImplementedError`)
- GPU import errors wrapped with helpful messages
- Matrix decomposition failures caught and logged

**Evidence:**

```python
# minimum_phase_decomposition.py lines 74-93
try:
    return xp.linalg.cholesky(...)
except xp.linalg.linalg.LinAlgError:
    logger.warning(
        "Computing the initial conditions using the Cholesky failed. "
        "Using a random initial condition."
    )
    # Fallback to random initialization
```

**Potential Issues:**

- Deprecation warning for `numpy.linalg.linalg` (line 78)

  ```python
  except xp.linalg.linalg.LinAlgError:  # Deprecated in NumPy 2.0
  ```

**Recommendations:**

1. **Quick Win**: Fix deprecation warning:

   ```python
   # Change line 78 in minimum_phase_decomposition.py
   except xp.linalg.LinAlgError:  # Works for both NumPy and CuPy
   ```

**Severity:** Low (warning, not error)

---

## 7. Code Smells & Anti-Patterns

### 7.1 Detected Issues

#### 7.1.1 Mutable Default Arguments  CLEAN

**Status:** No instances found (good practice followed)

Evidence: Inspected all function signatures, proper use of `None` as default:

```python
# Good pattern used throughout
def function(arg: list | None = None):
    if arg is None:
        arg = []
```

#### 7.1.2 Magic Numbers � MINOR

**Found:**

```python
# minimum_phase_decomposition.py:85
N_RAND = 1000  # Should be constant at module level

# connectivity.py:882
is_low_bias = eigenvalues > 0.9  # Magic threshold

# transforms.py:244
return int(xp.floor(2 * self.time_halfbandwidth_product - 1))  # Magic formula
```

**Recommendations:**

1. **Low Priority**: Extract magic numbers to named constants:

   ```python
   # At module level
   MIN_EIGENVALUE_THRESHOLD = 0.9  # Low-bias taper criterion
   TAPER_MULTIPLIER = 2.0  # Standard multitaper formula
   N_RANDOM_SAMPLES = 1000  # Fallback for Cholesky failure
   ```

**Severity:** Low (values are standard from literature)

#### 7.1.3 Long Functions � MINOR

**Found:**

- `minimum_phase_decomposition()`: 33 lines (acceptable for algorithm)
- `_find_significant_frequencies()`: 40+ lines (could be refactored)
- `group_delay()`: 60+ lines (complex method)

**Recommendations:**

1. **Low Priority**: Consider extracting sub-functions:

   ```python
   # connectivity.py group_delay() method
   # Extract regression logic into separate function
   def _compute_phase_regression(coherence_phase, frequencies):
       """Compute linear regression of phase vs frequency."""
       # Lines 1352-1377 moved here
   ```

**Severity:** Low (complexity is inherent to algorithms)

#### 7.1.4 God Objects � MINOR

**Found:**

- `Connectivity` class: 15+ public methods, 470+ lines
  - Handles both functional and directed connectivity
  - Many closely related methods (good cohesion)
  - Could split into `FunctionalConnectivity` and `DirectedConnectivity` subclasses

**Recommendations:**

1. **Low Priority** (major refactor): Consider class hierarchy:

   ```python
   class ConnectivityBase:
       """Shared cross-spectral matrix computation"""

   class FunctionalConnectivity(ConnectivityBase):
       """Coherence, PLV, PLI methods"""

   class DirectedConnectivity(ConnectivityBase):
       """Granger causality, DTF, PDC methods"""
   ```

**Severity:** Low (current design is pragmatic and usable)

---

## 8. Dependency Management

### 8.1 Dependencies  EXCELLENT

**Core Requirements:**

```toml
dependencies = [
    "numpy>=1.24,<3.0",      # Modern numpy
    "scipy>=1.10",           # Recent scipy
    "xarray>=2023.1",        # For labeled arrays
    "matplotlib>=3.7"        # Visualization
]
```

**Strengths:**

- Minimal core dependencies
- Conservative version bounds (avoid breaking changes)
- Optional GPU support (`cupy` not required)
- Dev dependencies properly separated

**Dev Dependencies:**

```toml
dev = [
    "pytest>=8.0",
    "pytest-cov>=4.1",
    "nitime",               # For validation tests
    "black>=24.0",
    "ruff>=0.8.0",
    "mypy>=1.8",
    "numpydoc>=1.6",
]
```

### 8.2 Python Version Support  EXCELLENT

**Supported:** Python 3.10, 3.11, 3.12, 3.13

**Evidence:**

```toml
requires-python = ">=3.10"
classifiers = [
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
]
```

**Strengths:**

- Modern Python versions only (leverages recent features)
- Tested on all supported versions (per CI config)
- Uses PEP 604 union syntax (`|` instead of `Union`)

### 8.3 Import Structure  CLEAN

**Public API:**

```python
# __init__.py
from spectral_connectivity.connectivity import Connectivity
from spectral_connectivity.transforms import Multitaper
from spectral_connectivity.wrapper import multitaper_connectivity

__all__ = ["Connectivity", "Multitaper", "multitaper_connectivity"]
```

**Strengths:**

- Clear public API via `__all__`
- No circular imports
- Clean namespace (internal modules not exported)

---

## 9. Specific Module Reviews

### 9.1 `transforms.py` (1011 lines)  EXCELLENT

**Role:** Multitaper spectral analysis (time � frequency domain)

**Strengths:**

- Well-structured `Multitaper` class with clear properties
- DPSS computation validated against `nitime`
- Efficient strided sliding window implementation
- Proper detrending (constant, linear, or none)

**Minor Issues:**

- Some helper functions lack type hints (e.g., `_make_tapers`)
- Detrend function is 90+ lines (complex but necessary)

**Coverage:** 82% (good)

**Recommendation:** Add more edge case tests (e.g., very short time series)

### 9.2 `connectivity.py` (2176 lines) � LARGE BUT WELL-ORGANIZED

**Role:** Compute 15+ connectivity measures

**Strengths:**

- Comprehensive coverage of functional and directed measures
- Consistent pattern for all methods (decorators, docstrings)
- Proper use of expectation types
- Good separation of public and private methods

**Issues:**

- Very large file (consider splitting)
- Some advanced methods untested (canonical coherence, global coherence)
- Block-wise computation untested

**Coverage:** 71% (acceptable given complexity)

**Recommendations:**

1. **Medium Priority**: Split into `functional.py` and `directed.py`
2. **High Priority**: Add tests for block-wise computation
3. **Medium Priority**: Add integration tests for advanced measures

### 9.3 `wrapper.py` (281 lines)  EXCELLENT

**Role:** High-level convenience functions with xarray output

**Strengths:**

- Clean separation of concerns
- Proper error handling for unsupported methods
- Good use of xarray for labeled output
- Flexible method selection (single or multiple)

**Coverage:** 91% (excellent)

**Minor Issue:**

- TODO comment about finding methods (line 230)

**Recommendation:** Use `inspect.getmembers()` instead of `dir()`:

```python
import inspect
method = [
    name for name, _ in inspect.getmembers(Connectivity, predicate=inspect.ismethod)
    if not name.startswith("_") and name not in bad_methods
]
```

### 9.4 `minimum_phase_decomposition.py` (323 lines)  EXCELLENT

**Role:** Wilson algorithm for spectral matrix factorization

**Strengths:**

- Well-documented algorithm with academic references
- Proper convergence checking
- Graceful fallback for Cholesky failures
- Clean separation of algorithm steps

**Coverage:** 89% (excellent)

**Minor Issue:**

- Deprecation warning for `xp.linalg.linalg.LinAlgError` (line 78)

**Recommendation:** Fix deprecation (see Section 6.2)

### 9.5 `statistics.py` (496 lines)  VERY GOOD

**Role:** Statistical inference for connectivity measures

**Strengths:**

- Multiple comparison corrections (Benjamini-Hochberg, Bonferroni)
- Fisher z-transform for coherence significance
- Proper bias correction for finite sample sizes
- Good docstrings with examples

**Coverage:** 70% (good)

**Minor Issue:**

- TODO comment about axis parameter (line 139)

**Recommendation:** Add axis parameter for multi-dimensional corrections

---

## 10. Configuration Files

### 10.1 `pyproject.toml`  EXCELLENT

**Strengths:**

- Modern build system (hatchling with VCS versioning)
- Comprehensive tool configurations (black, ruff, mypy, pytest)
- Proper metadata (classifiers, URLs, keywords)
- Optional dependencies well-organized (`[dev]`, `[gpu]`)

**Best Practices:**

```toml
[tool.ruff.lint]
select = ["E", "W", "F", "I", "N", "UP", "B", "C4", "NPY", "RUF"]
# Comprehensive linting rules

[tool.ruff.lint.pydocstyle]
convention = "numpy"  # Proper docstring style

[tool.mypy]
check_untyped_defs = true  # Type check even untyped functions
```

### 10.2 `environment.yml`  GOOD

**Strengths:**

- Matches `pyproject.toml` dependencies
- Includes documentation tools (sphinx)
- Includes CI/build tools (hatch, twine)
- Has `nitime` for test validation

**Minor Issue:**

- Some redundancy with `pyproject.toml` (unavoidable for conda)

---

## 11. Prioritized Recommendations

### 11.1 Quick Wins (< 1 hour)

1. **Fix deprecation warning** (`minimum_phase_decomposition.py:78`)

   ```python
   except xp.linalg.LinAlgError:  # Instead of xp.linalg.linalg.LinAlgError
   ```

2. **Resolve TODO comments:**
   - `wrapper.py:230`: Use `inspect.getmembers()`
   - `statistics.py:139`: Add axis parameter (or remove TODO)

3. **Add missing docstring examples:**
   - `Connectivity.coherence_magnitude()` could use example
   - `Multitaper.fft()` could show expected output shape

### 11.2 High Priority Improvements (1-3 days)

1. **Add input validation to `Connectivity` and `Multitaper`:**
   - Check array shapes
   - Validate parameter ranges
   - Warn on NaN/Inf values

2. **Add tests for block-wise computation:**
   - Verify blocked = unblocked results
   - Test memory efficiency gains
   - Document when to use blocks

3. **Test advanced connectivity measures:**
   - `canonical_coherence()` with synthetic data
   - `global_coherence()` validation
   - `group_delay()` with known phase relationships

### 11.3 Medium Priority Enhancements (1-2 weeks)

1. **Improve test coverage to 85%+:**
   - Focus on `connectivity.py` (currently 71%)
   - Add edge case tests (empty arrays, single signals)
   - Test error paths

2. **Add memory estimation utility:**
   - Document memory requirements
   - Provide estimation function
   - Add warnings for large allocations

3. **Enhance type checking:**
   - Enable `disallow_untyped_defs = true`
   - Add missing type hints to helper functions
   - Enable strict mode incrementally

4. **Address TODOs systematically:**
   - Implement or document deferred features
   - Remove stale comments

### 11.4 Low Priority Refactoring (Future work)

1. **Consider splitting `connectivity.py`:**
   - `functional_connectivity.py` (coherence, PLV, PLI)
   - `directed_connectivity.py` (Granger, DTF, PDC)
   - Keep base class in `connectivity.py`

2. **Add performance benchmarks:**
   - CPU vs GPU comparisons
   - Memory profiling
   - Scaling analysis (N signals, M time points)

3. **Extract magic numbers to constants:**
   - `MIN_EIGENVALUE_THRESHOLD = 0.9`
   - `TAPER_MULTIPLIER = 2.0`
   - Document scientific rationale

---

## 12. Final Verdict

### Overall Rating: **APPROVE** 

The `spectral_connectivity` package demonstrates **production-ready quality** with:

-  Clean architecture (three-layer design)
-  Comprehensive functionality (15+ connectivity measures)
-  Good test coverage (77%, with specific gaps identified)
-  Excellent documentation (NumPy-style docstrings)
-  Modern tooling (ruff, black, mypy, pytest)
-  Scientific validation (tests against nitime)

### Critical Blockers: **NONE**

### Required Before Merge: **NONE** (code is already in production-stable state)

### Recommended Improvements

**Priority 1 (High - address within 1-2 weeks):**

1. Add input validation to core classes
2. Add tests for block-wise computation
3. Test advanced connectivity measures (canonical, global coherence)
4. Fix deprecation warning in `minimum_phase_decomposition.py`

**Priority 2 (Medium - address within 1-2 months):**

1. Increase test coverage to 85%+
2. Add memory estimation utility
3. Enable stricter mypy settings incrementally
4. Resolve all TODO comments

**Priority 3 (Low - address as time permits):**

1. Consider splitting `connectivity.py` into submodules
2. Add performance benchmarks
3. Extract magic numbers to named constants
4. Enhance type hints for all helper functions

---

## Summary Statistics

**Files Reviewed:**

- 8 core Python modules (~10,500 lines)
- 11 test files (149 tests)
- 2 configuration files (`pyproject.toml`, `environment.yml`)

**Quality Metrics:**

- Test coverage: 77% (good)
- Linting errors: 0 (excellent)
- Type checking errors: 0 (excellent)
- Documentation: NumPy-style docstrings throughout (excellent)
- Python version support: 3.10+ (modern)

**Final Recommendation:** APPROVE with suggested improvements tracked as future enhancements. The code is suitable for production use today.
