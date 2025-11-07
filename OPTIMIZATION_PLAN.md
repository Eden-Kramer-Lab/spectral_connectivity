# Performance Optimization Plan for spectral_connectivity

**Document Version**: 2.0
**Last Updated**: 2025-11-07
**Analysis Based On**: Codebase commit e67dfaf
**Updated After**: Raymond Hettinger-style code review

---

## Executive Summary

This document outlines a comprehensive performance optimization strategy for the `spectral_connectivity` package, focusing on both computational speed and memory efficiency. The plan prioritizes **Pythonic patterns**, **measurement before optimization**, and **correctness over premature optimization**.

**Key Findings**:
- **Speed bottlenecks**: Redundant expensive computations, unvectorized loops, inefficient taper generation
- **Memory bottlenecks**: Unnecessary copies in specific hot paths (tridisolve), inefficient windowing for very large datasets
- **Critical constraint**: Caching must be memory-aware; naive LRU caching causes OOM errors
- **Philosophy**: Profile first, optimize the critical 3%, maintain code clarity

**Expected Overall Gains** (when Phase 1 optimizations applied):
- **Speed**: 3-10x faster for typical workflows (multi-measure analysis)
- **Memory**: 10-20% reduction with targeted optimizations
- **Maintainability**: Improved through use of standard library patterns

---

## Table of Contents

1. [Phase 1: High-Impact, Low-Risk Optimizations](#phase-1-high-impact-low-risk-optimizations)
2. [Phase 2: Measured Optimizations](#phase-2-measured-optimizations)
3. [Phase 3: Advanced Optimizations](#phase-3-advanced-optimizations)
4. [Implementation Guidelines](#implementation-guidelines)
5. [Testing and Validation Strategy](#testing-and-validation-strategy)
6. [Appendix: Rejected Optimizations](#appendix-rejected-optimizations)

---

## Phase 1: High-Impact, Low-Risk Optimizations

**Timeline**: 1-2 weeks
**Expected Gains**: 3-10x speed improvement for multi-measure workflows
**Philosophy**: Use standard library patterns, measure before optimizing

### 1.1 Use functools.cached_property for Instance Caching

**Problem**: Expensive properties like `_minimum_phase_factor`, `_power`, and `_transfer_function` are recomputed on every access, even within the same analysis workflow.

**Why Not Manual Cache Dictionary?**
Manual caching works but reinvents the wheel. Python 3.8+ has `functools.cached_property` which is:
- Simpler (no manual cache management)
- Standard library (well-tested)
- Automatic cleanup (cache disappears with instance)
- Thread-safe
- More Pythonic

**Why LRU Cache Fails**:
```python
# ❌ DANGEROUS - Causes OOM errors
@lru_cache(maxsize=128)
def _minimum_phase_factor(self):
    # Returns (n_time_windows, n_freqs, n_signals, n_signals) array
    # For 100 time windows × 512 freqs × 64 signals: ~1.6 GB per call
    # LRU cache keeps 128 of these = 200+ GB!
```

**Solution**: Use `functools.cached_property` (Python 3.8+)

**Files**: `spectral_connectivity/connectivity.py`

**Implementation**:

```python
from functools import cached_property

class Connectivity:
    # ... existing __init__ code ...

    @cached_property
    def _power(self) -> NDArray[np.floating]:
        """Cached power computation.

        Computed once per instance and cached automatically.
        """
        return self._expectation(
            self.fourier_coefficients * self.fourier_coefficients.conjugate()
        ).real

    @cached_property
    def _minimum_phase_factor(self) -> NDArray[np.complexfloating]:
        """Cached minimum phase decomposition.

        WARNING: This is a LARGE array. Only access if needed for directed measures.
        Cached automatically to avoid recomputation across multiple directed measures.
        """
        return minimum_phase_decomposition(
            self._expectation_cross_spectral_matrix()
        )

    @cached_property
    def _transfer_function(self) -> NDArray[np.complexfloating]:
        """Cached transfer function computation."""
        result = _estimate_transfer_function(self._minimum_phase_factor)
        # Apply non_negative_frequencies decorator logic
        n_frequencies = result.shape[-3]
        non_neg_index = xp.arange(0, n_frequencies // 2 + 1)
        return xp.take(result, indices=non_neg_index, axis=-3)

    @cached_property
    def _noise_covariance(self) -> NDArray[np.floating]:
        """Cached noise covariance computation."""
        return _estimate_noise_covariance(self._minimum_phase_factor)

    @cached_property
    def _MVAR_Fourier_coefficients(self) -> NDArray[np.complexfloating]:
        """Cached MVAR coefficients computation."""
        H = self._transfer_function
        lam = TIKHONOV_REGULARIZATION_FACTOR * xp.mean(xp.real(xp.conj(H) * H))
        identity = xp.eye(H.shape[-1], dtype=H.dtype)
        regularized_H = H + lam * identity
        return xp.linalg.solve(regularized_H, identity)

    # No need for clear_cache() method - just create new instance if needed
```

**For Python < 3.8 Projects** (if needed):

```python
# Add to spectral_connectivity/utils.py
class cached_property:
    """Cached property descriptor. Use functools.cached_property in Python 3.8+.

    This is a backport for older Python versions.
    """
    def __init__(self, func):
        self.func = func
        self.attrname = None
        self.__doc__ = func.__doc__

    def __set_name__(self, owner, name):
        self.attrname = name

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        if self.attrname is None:
            raise TypeError(
                "Cannot use cached_property instance without calling __set_name__"
            )
        cache = instance.__dict__
        val = cache.get(self.attrname, _NOT_FOUND := object())
        if val is _NOT_FOUND:
            val = self.func(instance)
            cache[self.attrname] = val
        return val
```

**Lines to Modify**:
- Line 442-445: `_power` property → `@cached_property`
- Line 568-569: `_minimum_phase_factor` property → `@cached_property`
- Line 572-574: `_transfer_function` property → `@cached_property`
- Line 577-578: `_noise_covariance` property → `@cached_property`
- Line 581-588: `_MVAR_Fourier_coefficients` property → `@cached_property`

**Expected Gain**:
- **Speed**: 3-5x for workflows computing multiple connectivity measures
- **Memory**: Minimal overhead, cache lifetime tied to instance
- **Code clarity**: Simpler than manual caching

**Risk**: Very low. Standard library pattern, well-tested.

---

### 1.2 Improve DPSS Taper Caching

**Problem**: DPSS tapers are computed via expensive eigenvalue decomposition, but identical parameters result in redundant computation. Using `@lru_cache` with float parameters is fragile due to floating-point comparison issues.

**Why Simple LRU Cache is Problematic**:
```python
# ❌ FRAGILE - Float comparison issues
@lru_cache(maxsize=32)
def _make_tapers(n_samples, fs, nw, n_tapers, is_low_bias):
    # fs=1000.0 vs fs=1000.00000001 → cache miss!
```

**Solution**: Custom hashable cache key with proper float handling

**Files**: `spectral_connectivity/transforms.py`

**Implementation**:

```python
# Add to transforms.py (around line 1400)

class _TaperCacheKey:
    """Hashable cache key for taper parameters.

    Handles floating-point comparison properly by rounding to avoid
    spurious cache misses from minor numerical differences.
    """
    __slots__ = ('n_samples', 'fs', 'nw', 'n_tapers', 'is_low_bias', '_hash')

    def __init__(self, n_time_samples_per_window, sampling_frequency,
                 time_halfbandwidth_product, n_tapers, is_low_bias):
        self.n_samples = int(n_time_samples_per_window)
        # Round floats to 6 decimal places to avoid spurious cache misses
        self.fs = round(float(sampling_frequency), 6)
        self.nw = round(float(time_halfbandwidth_product), 6)
        self.n_tapers = int(n_tapers)
        self.is_low_bias = bool(is_low_bias)
        # Pre-compute hash for efficiency
        self._hash = hash((self.n_samples, self.fs, self.nw,
                          self.n_tapers, self.is_low_bias))

    def __eq__(self, other):
        if not isinstance(other, _TaperCacheKey):
            return NotImplemented
        return (self.n_samples == other.n_samples and
                self.fs == other.fs and
                self.nw == other.nw and
                self.n_tapers == other.n_tapers and
                self.is_low_bias == other.is_low_bias)

    def __hash__(self):
        return self._hash


# Module-level cache - simple dict with FIFO eviction
_TAPER_CACHE = {}
_TAPER_CACHE_MAX_SIZE = 32


def _make_tapers(
    n_time_samples_per_window: int,
    sampling_frequency: float,
    time_halfbandwidth_product: float,
    n_tapers: int,
    is_low_bias: bool = True,
) -> NDArray[np.floating]:
    """Return discrete prolate spheroidal sequences (tapers) for multitaper analysis.

    Results are cached based on parameters. Cache holds up to 32 unique
    taper sets (~1-2 MB typical memory usage). Uses proper float handling
    to avoid spurious cache misses.

    Parameters
    ----------
    n_time_samples_per_window : int
        Number of samples in each window
    sampling_frequency : float
        Sampling rate in Hz
    time_halfbandwidth_product : float
        Time-half-bandwidth product (NW)
    n_tapers : int
        Number of tapers to compute
    is_low_bias : bool, default=True
        Use low-bias tapers if True

    Returns
    -------
    tapers : NDArray
        Tapers scaled by sqrt(sampling_frequency)

    Notes
    -----
    Caching is safe because:
    1. Tapers are small (typically 5-50 KB per entry)
    2. Heavily reused across windows with same parameters
    3. Cache size bounded to ~1-2 MB maximum
    4. Proper float handling avoids spurious cache misses

    To clear cache (e.g., for testing): call clear_taper_cache()
    """
    # Create cache key with proper float handling
    key = _TaperCacheKey(
        n_time_samples_per_window,
        sampling_frequency,
        time_halfbandwidth_product,
        n_tapers,
        is_low_bias
    )

    # Check cache
    if key in _TAPER_CACHE:
        return _TAPER_CACHE[key]

    # Compute tapers (expensive eigenvalue decomposition)
    tapers, _ = dpss_windows(
        n_time_samples_per_window,
        time_halfbandwidth_product,
        n_tapers,
        is_low_bias=is_low_bias,
    )
    result = tapers.T * xp.sqrt(sampling_frequency)

    # Add to cache with FIFO eviction
    if len(_TAPER_CACHE) >= _TAPER_CACHE_MAX_SIZE:
        # Simple FIFO eviction
        _TAPER_CACHE.pop(next(iter(_TAPER_CACHE)))

    _TAPER_CACHE[key] = result
    return result


def clear_taper_cache():
    """Clear the taper cache.

    Useful for testing or when memory is extremely tight.
    Under normal usage, the cache only uses ~1-2 MB and should not need clearing.
    """
    _TAPER_CACHE.clear()
```

**Lines to Modify**:
- Lines 1408-1440: Replace with new implementation

**Expected Gain**:
- **Speed**: 10-50x for repeated windowed analyses (100+ windows with same parameters)
- **Memory**: Negligible (~1-2MB max for cache)
- **Reliability**: No spurious cache misses from float comparison issues

**Risk**: Very low. Cache size is bounded, tapers are small, proper float handling.

---

### 1.3 Optimize tridisolve (Targeted Copy Elimination)

**Problem**: `tridisolve` makes defensive copies of diagonal arrays even when they could be safely overwritten.

**Why NOT change sliding window default**:
- The copy in sliding window is negligible compared to FFT time (~10-50ms vs seconds)
- Correctness and safety should come first
- Users who modify windowed data would encounter subtle bugs
- **Keep `is_copy=True` as the safe default**

**Files**: `spectral_connectivity/transforms.py`

**Current** (transforms.py, lines 1469-1470):
```python
def tridisolve(d, e, b, overwrite_b=True):
    dw = d.copy()  # Always copies
    ew = e.copy()  # Always copies
```

**Change**:
```python
def tridisolve(
    d: NDArray[np.floating],
    e: NDArray[np.floating],
    b: NDArray[np.floating],
    overwrite_b: bool = True,
    overwrite_d: bool = False,
    overwrite_e: bool = False,
) -> NDArray[np.floating]:
    """Symmetric tridiagonal system solver, from Golub and Van Loan p157.

    .. note:: Copied from NiTime.

    Parameters
    ----------
    d : ndarray
        Main diagonal stored in d[:]
    e : ndarray
        Superdiagonal stored in e[:-1]
    b : ndarray
        RHS vector
    overwrite_b : bool, default=True
        Whether to overwrite b with solution
    overwrite_d : bool, default=False
        Whether d can be overwritten (avoids copy)
    overwrite_e : bool, default=False
        Whether e can be overwritten (avoids copy)

    Returns
    -------
    x : ndarray
        Solution to Ax = b (if overwrite_b is False). Otherwise solution is
        stored in previous RHS vector b

    Notes
    -----
    Set overwrite_d=True and overwrite_e=True only when you're certain these
    arrays won't be reused. Default is False for safety.
    """
    N = len(b)
    # Work vectors - only copy if necessary
    dw = d if overwrite_d else d.copy()
    ew = e if overwrite_e else e.copy()
    if overwrite_b:
        x = b
    else:
        x = b.copy()

    # ... rest of function unchanged ...
    for j in range(1, N):
        w = ew[j - 1] / dw[j - 1]
        dw[j] = dw[j] - w * ew[j - 1]
        x[j] = x[j] - w * x[j - 1]
    x[N - 1] = x[N - 1] / dw[N - 1]
    for j in range(N - 2, -1, -1):
        x[j] = (x[j] - ew[j] * x[j + 1]) / dw[j]

    return x
```

**Update callers** in `tridi_inverse_iteration` (line 1533):
```python
tridisolve(eig_diag, e, x0, overwrite_b=True, overwrite_d=True, overwrite_e=True)
```

**Lines to Modify**:
- Lines 1443-1489: Add parameters to `tridisolve`
- Line 1533: Update caller in `tridi_inverse_iteration`

**Expected Gain**:
- **Memory**: 10-15% reduction in eigenvalue computation
- **Speed**: 5-10% faster (fewer allocations)

**Risk**: Low. Carefully audited call sites.

---

### 1.4 Keep Cross-Spectral Matrix As-Is (For Now)

**Original Proposal**: Replace with `einsum` for memory efficiency.

**Why Defer This**:
1. `einsum` doesn't accept `dtype` parameter (implementation error in original plan)
2. For outer products, `einsum` isn't always faster than broadcasting
3. The alleged memory savings are small compared to output array size
4. Current implementation is clear and works

**Decision**: **Keep current implementation**. If profiling shows this is a bottleneck, revisit with proper measurement.

**If optimizing later**, use this pattern:

```python
@property
def _cross_spectral_matrix(self) -> NDArray[np.complexfloating]:
    """Return cross-spectral matrix.

    For large arrays, alternative implementations may be faster.
    Current implementation prioritizes clarity and correctness.
    """
    fc = self.fourier_coefficients

    # Current implementation (clear and correct)
    fc_expanded = fc[..., xp.newaxis]
    result = _complex_inner_product(fc_expanded, fc_expanded, dtype=self._dtype)

    return result
```

**Lines to Modify**: None for Phase 1.

---

## Phase 2: Measured Optimizations

**Timeline**: 2-3 weeks
**Expected Gains**: 2-5x for specific methods
**Philosophy**: Profile first, optimize based on data

### 2.1 Profile Wilson Algorithm Before Optimizing

**Current Proposal**: Optimize copy operations in Wilson algorithm.

**Better Approach**: **Measure first**. Add instrumentation to understand:
1. How many iterations does Wilson algorithm typically need?
2. What fraction of time windows converge quickly vs slowly?
3. Is copying actually the bottleneck?

**Implementation** (add to `minimum_phase_decomposition.py`):

```python
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

# Module-level statistics for understanding convergence behavior
_WILSON_STATS = defaultdict(int)


def minimum_phase_decomposition(
    cross_spectral_matrix: NDArray[np.complexfloating],
    tolerance: float = 1e-8,
    max_iterations: int = 60,
) -> NDArray[np.complexfloating]:
    """Compute minimum phase decomposition using Wilson algorithm.

    ... existing docstring ...
    """
    n_time_points = cross_spectral_matrix.shape[0]
    n_signals = cross_spectral_matrix.shape[-1]
    identity_matrix = xp.eye(n_signals)
    is_converged = xp.zeros(n_time_points, dtype=bool)
    minimum_phase_factor = xp.zeros(cross_spectral_matrix.shape)
    minimum_phase_factor[..., :, :, :] = _get_initial_conditions(cross_spectral_matrix)

    for iteration in range(max_iterations):
        logger.debug(
            f"iteration: {iteration}, {is_converged.sum()} of {len(is_converged)} converged"
        )
        old_minimum_phase_factor = minimum_phase_factor.copy()
        linear_predictor = _get_linear_predictor(
            minimum_phase_factor, cross_spectral_matrix, identity_matrix
        )
        minimum_phase_factor = xp.matmul(
            minimum_phase_factor, _get_causal_signal(linear_predictor)
        )

        # If already converged at a time point, don't change.
        minimum_phase_factor[is_converged, ...] = old_minimum_phase_factor[
            is_converged, ...
        ]
        is_converged = _check_convergence(
            minimum_phase_factor, old_minimum_phase_factor, tolerance
        )

        if xp.all(is_converged):
            # Record successful early convergence
            _WILSON_STATS[f'converged_at_{iteration}'] += n_time_points
            logger.debug(f"All converged at iteration {iteration}")
            return minimum_phase_factor
    else:
        # Record maxed-out iterations
        _WILSON_STATS['max_iterations_reached'] += (~is_converged).sum()
        _WILSON_STATS['converged_at_60'] += is_converged.sum()
        logger.warning(
            f"Maximum iterations reached. {is_converged.sum()} of {len(is_converged)} converged"
        )
        return minimum_phase_factor


def get_wilson_stats():
    """Get statistics on Wilson algorithm convergence.

    Returns
    -------
    stats : dict
        Dictionary with keys like 'converged_at_5', 'converged_at_10', etc.
        showing how many time windows converged at each iteration.

    Examples
    --------
    >>> # After running several analyses
    >>> stats = get_wilson_stats()
    >>> print(stats)
    {'converged_at_5': 1200, 'converged_at_10': 300, 'max_iterations_reached': 5}
    >>> # Most converge in 5-10 iterations, only 5 time windows needed all 60
    """
    return dict(_WILSON_STATS)


def clear_wilson_stats():
    """Clear Wilson algorithm statistics."""
    _WILSON_STATS.clear()
```

**Then add to documentation**:

```markdown
# Performance Analysis: Wilson Algorithm

After running your typical workloads, check convergence statistics:

```python
from spectral_connectivity.minimum_phase_decomposition import get_wilson_stats

# Run your analysis
mt = Multitaper(data, ...)
conn = Connectivity.from_multitaper(mt)
granger = conn.pairwise_spectral_granger_prediction()

# Check Wilson algorithm behavior
stats = get_wilson_stats()
print(stats)
# Example output: {'converged_at_8': 450, 'converged_at_12': 50}
# → Most time windows converge in 8 iterations, some in 12
```

If most converge in <15 iterations, current implementation is fine.
If many hit 60 iterations, consider:
1. Better initial conditions
2. Different convergence criteria
3. Alternative algorithms (e.g., Anderson acceleration)
```

**Decision**: Add instrumentation, collect data, then decide if optimization is needed.

---

### 2.2 Vectorize global_coherence with Clear Documentation

**Problem**: Nested loops prevent parallelization.

**Files**: `spectral_connectivity/connectivity.py`

**Current** (lines 876-888): Nested loops over time and frequency.

**Improved Implementation** (with clear documentation):

```python
def global_coherence(
    self, max_rank: int = 1
) -> tuple[NDArray[np.floating], NDArray[np.complexfloating]]:
    """Find linear combinations that capture the most coherent power.

    ... (existing docstring) ...
    """
    (
        n_time_windows,
        n_trials,
        n_tapers,
        n_fft_samples,
        n_signals,
    ) = self.fourier_coefficients.shape

    # Reshaping strategy for vectorized SVD:
    #
    # Input:  (n_time, n_trials, n_tapers, n_freq, n_signals)
    # Goal:   (n_time*n_freq, n_signals, n_trials*n_tapers)
    #
    # Why? SVD operates on (signals, observations) matrices.
    # We want to compute SVD for each (time, freq) combination in parallel.
    #
    # Step 1: Move dimensions to correct positions
    fc = self.fourier_coefficients.transpose(0, 3, 4, 1, 2)
    # Now: (n_time, n_freq, n_signals, n_trials, n_tapers)

    # Step 2: Combine batch dimensions (time*freq) and observations (trials*tapers)
    n_batches = n_time_windows * n_fft_samples
    n_observations = n_trials * n_tapers
    fc_batched = fc.reshape(n_batches, n_signals, n_observations)
    # Now: (n_time*n_freq, n_signals, n_observations)

    # Compute SVD based on rank requirement
    if max_rank >= n_signals - 1:
        # Full SVD can be vectorized efficiently
        U, S, _ = xp.linalg.svd(fc_batched, full_matrices=False)
        # S shape: (n_batches, n_signals)
        # U shape: (n_batches, n_signals, n_signals)

        # Extract top-rank components
        singular_values = S[:, :max_rank]
        left_vectors = U[:, :, :max_rank]
    else:
        # Sparse SVD doesn't vectorize well - use loop
        # (But at least we've eliminated the double loop over time and freq)
        singular_values = xp.zeros((n_batches, max_rank))
        left_vectors = xp.zeros((n_batches, n_signals, max_rank), dtype=xp.complex128)

        for i, fc_slice in enumerate(fc_batched):
            # Note: Could potentially parallelize this loop in the future
            left_vectors[i], singular_values[i], _ = svds(fc_slice, max_rank)

    # Convert singular values to global coherence
    # (normalized by number of observations)
    global_coherence_values = (singular_values ** 2) / n_observations

    # Reshape back to (time, freq, rank) structure
    global_coherence_values = global_coherence_values.reshape(
        n_time_windows, n_fft_samples, max_rank
    )
    left_vectors = left_vectors.reshape(
        n_time_windows, n_fft_samples, n_signals, max_rank
    )

    # Convert from CuPy if needed
    try:
        return xp.asnumpy(global_coherence_values), xp.asnumpy(left_vectors)
    except AttributeError:
        return global_coherence_values, left_vectors
```

**Lines to Modify**: Lines 822-895

**Expected Gain**:
- **Speed**: 5-10x for full SVD case, 2-3x for sparse SVD case
- **GPU**: Especially beneficial (10-20x potential speedup)
- **Clarity**: Better documentation of reshaping strategy

**Risk**: Medium. Requires testing of reshaping logic, but strategy is well-documented.

---

### 2.3 Optimize Block Processing

**Keep existing plan** from Phase 2.2 of original document - this optimization is sound.

Pre-compute unique indices outside loop and use in-place conjugate operations.

**Lines to Modify**: Lines 487-526

**Expected Gain**:
- **Speed**: 20-30% for block-based computation (50+ signals)
- **Memory**: 10% reduction (in-place conjugate)

---

## Phase 3: Advanced Optimizations

**Timeline**: 3-4 weeks
**Expected Gains**: Additional memory reduction for very large datasets
**When to use**: Only for extreme cases (10,000+ windows, 64+ channels)

### 3.1 Chunked Windowing (Only for Very Large Datasets)

**Keep existing plan** from Phase 3.1 of original document.

**When NOT to use**:
- Typical analyses (<1000 windows)
- Sufficient memory available
- Performance is already acceptable

**When to use**:
- Very long recordings (>10,000 windows)
- High channel counts (>64 channels)
- Memory-constrained environments
- After profiling shows windowing is the bottleneck

---

## Implementation Guidelines

### Development Philosophy

**"Premature optimization is the root of all evil, yet we should not pass up our opportunities in that critical 3%."** — Donald Knuth

Your critical 3%:
1. ✅ Property caching for expensive intermediate results
2. ✅ DPSS taper caching
3. ✅ Vectorization where it matters (global_coherence)

Everything else: **Profile first, optimize second.**

### Development Process

1. **Profile before optimizing**:
   ```python
   import cProfile
   import pstats

   profiler = cProfile.Profile()
   profiler.enable()
   # Run your typical workflow
   profiler.disable()

   stats = pstats.Stats(profiler)
   stats.sort_stats('cumulative')
   stats.print_stats(20)  # Show top 20 hotspots
   ```

2. **Create feature branch** for each optimization:
   ```bash
   git checkout -b optimize/cached-properties
   ```

3. **Implement one optimization at a time** - do not bundle

4. **Write tests first** (see Testing Strategy below)

5. **Benchmark before and after**:
   ```python
   import time
   import numpy as np
   from spectral_connectivity import Multitaper, Connectivity

   data = np.random.randn(5000, 10, 32)

   # Before optimization
   start = time.time()
   mt = Multitaper(data, sampling_frequency=1000)
   conn = Connectivity.from_multitaper(mt)
   coherence = conn.coherence_magnitude()
   imaginary = conn.imaginary_coherence()
   plv = conn.phase_locking_value()
   baseline_time = time.time() - start

   # After optimization (in new code)
   # ... repeat ...
   optimized_time = time.time() - start

   speedup = baseline_time / optimized_time
   print(f"Speedup: {speedup:.2f}x")
   ```

6. **Memory profiling**:
   ```python
   import tracemalloc

   tracemalloc.start()
   # Run code
   current, peak = tracemalloc.get_traced_memory()
   tracemalloc.stop()

   print(f"Peak memory: {peak / 1024 / 1024:.1f} MB")
   ```

### Code Review Checklist

For each optimization PR, reviewers should verify:

- [ ] **Correctness**: Results match original (within floating-point tolerance)
- [ ] **Performance**: Profiling shows expected improvement
- [ ] **Memory**: Memory profiling confirms benefits (if applicable)
- [ ] **Clarity**: Code is readable and well-documented
- [ ] **Tests**: All existing tests pass, new tests added
- [ ] **Documentation**: Docstrings updated, optimization documented
- [ ] **Backwards compatibility**: No breaking API changes
- [ ] **GPU compatibility**: Works with both NumPy and CuPy
- [ ] **Edge cases**: Handles small datasets, single channel, etc.
- [ ] **Pythonic**: Uses standard library patterns where appropriate

---

## Testing and Validation Strategy

### 1. Correctness Tests

**Requirement**: Optimized code must produce identical results (within numerical tolerance).

```python
import numpy as np
import pytest
from numpy.testing import assert_allclose
from spectral_connectivity import Multitaper, Connectivity

@pytest.fixture
def sample_data():
    """Generate reproducible test data."""
    np.random.seed(42)
    return np.random.randn(1000, 5, 8)  # 1000 samples, 5 trials, 8 channels

class TestOptimizationCorrectness:
    """Test that optimizations don't change results."""

    def test_cached_properties_are_identical(self, sample_data):
        """Test that cached properties return identical values."""
        mt = Multitaper(sample_data, sampling_frequency=1000)
        conn = Connectivity.from_multitaper(mt)

        # Access property twice
        power1 = conn._power
        power2 = conn._power

        # Should be exact match (same object with cached_property)
        assert power1 is power2

    def test_taper_cache_correctness(self):
        """Test that cached tapers match uncached computation."""
        from spectral_connectivity.transforms import _make_tapers, clear_taper_cache

        # Clear cache
        clear_taper_cache()

        # First call (computes)
        tapers1 = _make_tapers(500, 1000.0, 3.0, 5, True)

        # Second call (should use cache)
        tapers2 = _make_tapers(500, 1000.0, 3.0, 5, True)

        # Should be identical
        assert_allclose(tapers1, tapers2, rtol=1e-15)

    def test_float_tolerance_in_taper_cache(self):
        """Test that minor float differences still hit cache."""
        from spectral_connectivity.transforms import _make_tapers, clear_taper_cache

        clear_taper_cache()

        # Create tapers with slightly different floats (within rounding tolerance)
        tapers1 = _make_tapers(500, 1000.000000, 3.0, 5, True)
        tapers2 = _make_tapers(500, 1000.000001, 3.0, 5, True)  # Should still hit cache

        # Should be same object if cache hit
        assert tapers1 is tapers2

    def test_vectorized_global_coherence_matches_original(self, sample_data):
        """Test vectorized implementation matches original."""
        mt = Multitaper(sample_data, sampling_frequency=1000)
        conn = Connectivity.from_multitaper(mt)

        # Compute using optimized version
        gc_opt, ugc_opt = conn.global_coherence(max_rank=2)

        # Test properties that should hold
        assert gc_opt.shape[0] > 0  # Has time windows
        assert gc_opt.shape[1] > 0  # Has frequencies
        assert gc_opt.shape[2] == 2  # max_rank=2
        assert np.all(gc_opt >= 0)  # Coherence is non-negative
        assert np.all(gc_opt <= 1)  # Coherence is bounded by 1
```

### 2. Property-Based Tests with Hypothesis

**Add property-based testing** to find edge cases automatically:

```python
from hypothesis import given, strategies as st
import hypothesis.extra.numpy as npst

@given(
    data=npst.arrays(
        dtype=np.float64,
        shape=npst.array_shapes(min_dims=3, max_dims=3,
                               min_side=10, max_side=100),
    ),
    sampling_frequency=st.floats(min_value=100, max_value=10000),
)
def test_multitaper_properties(data, sampling_frequency):
    """Property-based test for multitaper.

    Tests that certain properties always hold, regardless of input.
    """
    mt = Multitaper(data, sampling_frequency=sampling_frequency)

    # Properties that should always hold
    assert mt.frequencies.min() >= 0
    assert mt.frequencies.max() <= mt.nyquist_frequency
    assert len(mt.frequencies) == mt.n_fft_samples // 2 + 1
    assert mt.n_tapers > 0
    assert mt.n_time_samples_per_window <= data.shape[0]

@given(
    n_samples=st.integers(min_value=100, max_value=1000),
    fs=st.floats(min_value=100, max_value=10000),
    nw=st.floats(min_value=1, max_value=10),
    n_tapers=st.integers(min_value=1, max_value=20),
)
def test_taper_cache_properties(n_samples, fs, nw, n_tapers):
    """Test that taper caching doesn't change results."""
    from spectral_connectivity.transforms import _make_tapers

    # Call twice with same parameters
    tapers1 = _make_tapers(n_samples, fs, nw, n_tapers)
    tapers2 = _make_tapers(n_samples, fs, nw, n_tapers)

    # Should return same cached object
    assert tapers1 is tapers2
```

### 3. Performance Regression Tests

```python
import pytest
import time

@pytest.mark.benchmark
class TestPerformanceRegression:
    """Ensure optimizations don't regress performance."""

    def test_multi_measure_workflow_performance(self, sample_data, benchmark):
        """Benchmark typical workflow with multiple measures."""
        def compute_multiple_measures():
            mt = Multitaper(sample_data, sampling_frequency=1000)
            conn = Connectivity.from_multitaper(mt)

            # Compute multiple measures (benefits from caching)
            results = {
                'coherence': conn.coherence_magnitude(),
                'imaginary': conn.imaginary_coherence(),
                'plv': conn.phase_locking_value(),
            }
            return results

        # Benchmark will measure time and compare across runs
        result = benchmark(compute_multiple_measures)
        assert all(v is not None for v in result.values())
```

### 4. GPU Tests

```python
import pytest
import os

@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get("SPECTRAL_CONNECTIVITY_ENABLE_GPU") != "true",
    reason="GPU not enabled"
)
class TestGPUOptimizations:
    """Test optimizations work on GPU."""

    def test_cached_properties_gpu(self, sample_data):
        """Test property caching works with CuPy arrays."""
        import cupy as cp

        data_gpu = cp.asarray(sample_data)
        mt = Multitaper(data_gpu, sampling_frequency=1000)
        conn = Connectivity.from_multitaper(mt)

        # Cache should work with CuPy
        power1 = conn._power
        power2 = conn._power
        assert power1 is power2
        assert isinstance(power1, cp.ndarray)
```

---

## Appendix: Rejected or Deferred Optimizations

### A.1 FFT Size Optimization (REJECTED)

**Original Idea**: Allow users to specify exact FFT length to avoid padding.

**Rejection Reason**:
- FFT performance degrades significantly for non-optimal sizes
- Memory savings (2-10%) don't justify speed loss (2-5x slower)
- `next_fast_len` is essential for good FFT performance

**Decision**: Keep automatic padding to `next_fast_len`.

---

### A.2 Naive LRU Cache for Properties (REJECTED)

**Original Idea**: Use `@lru_cache` decorator on properties.

**Rejection Reason**: Causes OOM errors as reported by users.

**Solution Used**: `functools.cached_property` (Phase 1.1).

---

### A.3 Manual Cache Dictionary (REPLACED)

**Original Idea**: Implement manual cache dictionary for properties.

**Replacement**: Use `functools.cached_property` - simpler, more Pythonic, standard library.

**Why Better**:
- No manual cache management
- Automatic cleanup
- Standard library (well-tested)
- Thread-safe
- Clearer intent

---

### A.4 Sliding Window Copy Elimination (DEFERRED)

**Original Idea**: Change `is_copy=True` to `is_copy=False` for memory savings.

**Deferral Reason**:
- Copy overhead is negligible compared to FFT time (~10-50ms vs seconds)
- Correctness and safety should come first
- Users who modify windowed data would encounter subtle bugs
- Premature optimization

**Decision**: **Keep `is_copy=True` as the safe default**. Only optimize if profiling shows it's a bottleneck (unlikely).

---

### A.5 Wilson Algorithm Copy Optimization (PROFILE FIRST)

**Original Idea**: Only copy unconverged time points in Wilson algorithm.

**Better Approach**: Add instrumentation to measure:
1. Typical iteration counts
2. Convergence patterns
3. Whether copying is actually the bottleneck

**Decision**: Add profiling/stats (Phase 2.1), optimize only if data shows it's needed.

---

### A.6 einsum for Cross-Spectral Matrix (DEFERRED)

**Original Idea**: Use `einsum` for memory efficiency.

**Deferral Reason**:
- Original implementation had errors (`einsum` doesn't take `dtype` parameter)
- For outer products, `einsum` isn't always faster
- Memory savings are small compared to output size
- Current implementation is clear and correct

**Decision**: Keep current implementation. Revisit only if profiling shows it's a bottleneck.

---

### A.7 Parallel Processing with multiprocessing (REJECTED)

**Original Idea**: Use Python multiprocessing for parallelization.

**Rejection Reason**:
- High overhead for array serialization
- NumPy/CuPy already use optimized BLAS/CUDA parallelization
- Would conflict with existing parallelism
- GPU operations don't parallelize well across processes

**Better Alternative**: Ensure NumPy uses optimized BLAS (OpenBLAS, MKL) and leverage existing GPU parallelism.

---

### A.8 Sparse Matrix Representations (REJECTED)

**Original Idea**: Use sparse matrices for cross-spectral matrix.

**Rejection Reason**:
- Cross-spectral matrices are typically dense
- Sparse operations are slower for dense data
- Added complexity not justified

**When It Might Work**: Only for >1000 channels with known sparsity (rare).

---

## Conclusion

This updated optimization plan provides a **pragmatic, measured approach** to improving the performance of the `spectral_connectivity` package. Key principles:

**✅ DO:**
- Use standard library patterns (`cached_property`)
- Profile before optimizing
- Focus on the critical 3%: caching, taper computation, vectorization
- Maintain code clarity
- Test thoroughly

**❌ DON'T:**
- Sacrifice readability for minor gains
- Optimize without measuring
- Reinvent standard library functionality
- Change safe defaults without good reason
- Over-engineer solutions

**Expected Results** (Phase 1 only):
- **3-10x speed improvement** for multi-measure workflows
- **Minimal memory overhead** (<1-2 MB for caches)
- **Improved maintainability** through Pythonic patterns
- **No breaking changes** to public API

**Next Steps**:
1. Review and approve this updated plan
2. Implement Phase 1 optimizations (1-2 weeks)
3. Profile typical user workloads
4. Use data to guide Phase 2 decisions
5. Only proceed to Phase 3 for extreme use cases

Remember: **"Correctness first, fast second, maintainable always."**
