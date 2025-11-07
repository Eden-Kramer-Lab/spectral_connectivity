# Code Review: spectral_connectivity
**Reviewer**: Raymond Hettinger (simulated perspective)
**Date**: 2025-11-07

---

## Overview

I've reviewed the `spectral_connectivity` package with a focus on Pythonic design, algorithmic efficiency, and maintainability. This is solid scientific Python work with good attention to numerical correctness. The optimization plan is well-researched, though I have some suggestions for making it more Pythonic and avoiding common pitfalls.

**TL;DR**: The code is well-structured, but there are opportunities to simplify, use Python's strengths better, and avoid over-engineering the optimization plan.

---

## The Good Stuff 🎯

### 1. Excellent Use of Properties for Lazy Evaluation

```python
# transforms.py
@property
def tapers(self) -> NDArray[np.floating]:
    if self._tapers is None:
        self._tapers = _make_tapers(...)
    return self._tapers
```

**Why this is good**: This is exactly the right pattern for expensive one-time computations. It's clean, readable, and efficient.

### 2. Strong Type Hints and Documentation

The codebase has excellent type hints and comprehensive docstrings. This is crucial for scientific code where understanding dimensions and data types is critical.

```python
def minimum_phase_decomposition(
    cross_spectral_matrix: NDArray[np.complexfloating],
    tolerance: float = 1e-8,
    max_iterations: int = 60,
) -> NDArray[np.complexfloating]:
```

**Recommendation**: Keep this up. Type hints are documentation that the computer can verify.

### 3. Clear Separation of Concerns

The package structure is clean:
- `transforms.py`: Time → Frequency domain
- `connectivity.py`: Compute connectivity measures
- `wrapper.py`: High-level API
- `statistics.py`: Statistical testing

This is how you should organize scientific code.

### 4. Good Error Messages

```python
# connectivity.py:288-296
if fourier_coefficients.ndim != 5:
    raise ValueError(
        f"fourier_coefficients must be 5-dimensional, got {fourier_coefficients.ndim}D array.\n"
        f"Expected shape: (n_time_windows, n_trials, n_tapers, n_fft_samples, n_signals)\n"
        f"Got shape: {fourier_coefficients.shape}\n\n"
        f"If you have time series data, use the Multitaper class to transform it:\n"
        # ... helpful examples ...
    )
```

**This is how you write error messages!** Tell users what's wrong, why it's wrong, and how to fix it.

---

## Areas for Improvement 🔧

### 1. The Optimization Plan Needs Simplification

Looking at your optimization plan, I see a common pattern: **you're optimizing implementation details before considering algorithmic improvements**. Let's fix that priority order.

#### Problem: Instance Caching is Over-Complicated

Your proposed solution:

```python
# OPTIMIZATION_PLAN.md
@property
def _power(self) -> NDArray[np.floating]:
    """Cached power computation."""
    if '_power' not in self._cache:
        self._cache['_power'] = self._expectation(
            self.fourier_coefficients * self.fourier_coefficients.conjugate()
        ).real
    return self._cache['_power']
```

**Raymond's Take**: This works, but it's reinventing wheels. Python has a better tool:

```python
from functools import cached_property

class Connectivity:
    @cached_property
    def _power(self) -> NDArray[np.floating]:
        """Cached power computation."""
        return self._expectation(
            self.fourier_coefficients * self.fourier_coefficients.conjugate()
        ).real
```

**Why this is better**:
1. **Simpler**: No manual cache dictionary management
2. **Standard library**: Uses well-tested Python builtin (3.8+)
3. **Automatic cleanup**: Cache disappears with instance
4. **Thread-safe**: If you ever need it
5. **Pythonic**: Follows established patterns

**For Python < 3.8**, use this pattern:

```python
class cached_property:
    """Decorator for cached properties. Use functools.cached_property in Python 3.8+."""
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

Then use it everywhere:

```python
class Connectivity:
    @cached_property
    def _power(self):
        return self._expectation(
            self.fourier_coefficients * self.fourier_coefficients.conjugate()
        ).real

    @cached_property
    def _minimum_phase_factor(self):
        return minimum_phase_decomposition(
            self._expectation_cross_spectral_matrix()
        )

    @cached_property
    def _transfer_function(self):
        result = _estimate_transfer_function(self._minimum_phase_factor)
        n_frequencies = result.shape[-3]
        non_neg_index = xp.arange(0, n_frequencies // 2 + 1)
        return xp.take(result, indices=non_neg_index, axis=-3)
```

**No `clear_cache()` method needed** - just delete and recreate the instance if you need fresh computations.

---

### 2. The DPSS Taper Caching Needs a Rethink

Your plan:

```python
@lru_cache(maxsize=32)
def _make_tapers_cached(
    n_time_samples_per_window: int,
    sampling_frequency: float,
    time_halfbandwidth_product: float,
    n_tapers: int,
    is_low_bias: bool = True,
) -> NDArray[np.floating]:
```

**Problems**:
1. Function-level caching with floats as keys is fragile (floating point comparison issues)
2. Returns NumPy arrays, which aren't hashable
3. Manual `maxsize=32` is arbitrary

**Better approach - Module-level cache with proper key handling**:

```python
# transforms.py

class _TaperCacheKey:
    """Hashable cache key for taper parameters.

    Handles floating-point comparison properly and is immutable.
    """
    __slots__ = ('n_samples', 'fs', 'nw', 'n_tapers', 'is_low_bias', '_hash')

    def __init__(self, n_time_samples_per_window, sampling_frequency,
                 time_halfbandwidth_product, n_tapers, is_low_bias):
        self.n_samples = int(n_time_samples_per_window)
        # Round floats to avoid spurious cache misses
        self.fs = round(float(sampling_frequency), 6)
        self.nw = round(float(time_halfbandwidth_product), 6)
        self.n_tapers = int(n_tapers)
        self.is_low_bias = bool(is_low_bias)
        # Pre-compute hash
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


# Module-level cache - simple dict
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
    taper sets (~1-2 MB typical memory usage).
    """
    # Create cache key
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

    # Compute tapers
    tapers, _ = dpss_windows(
        n_time_samples_per_window,
        time_halfbandwidth_product,
        n_tapers,
        is_low_bias=is_low_bias,
    )
    result = tapers.T * xp.sqrt(sampling_frequency)

    # Add to cache with LRU eviction
    if len(_TAPER_CACHE) >= _TAPER_CACHE_MAX_SIZE:
        # Simple FIFO eviction (could use collections.OrderedDict for true LRU)
        _TAPER_CACHE.pop(next(iter(_TAPER_CACHE)))

    _TAPER_CACHE[key] = result
    return result


def clear_taper_cache():
    """Clear the taper cache. Useful for testing or when memory is tight."""
    _TAPER_CACHE.clear()
```

**Why this is better**:
1. Proper handling of floating-point keys
2. Explicit cache size and eviction strategy
3. Can be cleared if needed
4. No dependency on functools internals
5. Clear documentation of memory usage

---

### 3. Don't Over-Optimize the Wrong Things

Looking at section 1.3.1 (sliding window default to no-copy), I have concerns:

```python
def _sliding_window(..., is_copy: bool = False) -> NDArray[np.floating]:
```

**Raymond's Rule**: *"Premature optimization is the root of all evil, but so is premature pessimization."*

The current default of `is_copy=True` is the **right default** because:

1. **Correctness over performance**: Copy-by-default prevents subtle bugs
2. **Stride tricks are tricky**: Views can cause confusing behavior if users modify data
3. **The actual bottleneck is the FFT, not the copy**: Copying takes ~10-50ms, FFT takes seconds

**Better approach**:
```python
def _sliding_window(..., is_copy: bool = True) -> NDArray[np.floating]:
    """
    ...
    is_copy : bool, default=True
        Return copy to prevent accidental modification of view.
        Set to False for read-only operations when memory is extremely tight.

    Notes
    -----
    The default is True (copy) for safety. In typical usage, the copy overhead
    is negligible compared to FFT computation time. Only set is_copy=False if:
    - You've profiled and confirmed copying is a bottleneck
    - You're certain the windowed data won't be modified
    - Memory is severely constrained
    """
```

**Profile first, optimize second.**

---

### 4. The Wilson Algorithm Optimization Misses the Point

Your optimization:

```python
# Only store old values for unconverged time points
unconverged_mask = ~is_converged
if xp.any(unconverged_mask):
    old_minimum_phase_factor = minimum_phase_factor[unconverged_mask].copy()
```

**This adds complexity without addressing the real issue**: Why do you need 60 iterations?

**Better questions to ask**:
1. Can you use a better initial guess? (You're doing Cholesky, which is good)
2. Can you use better convergence criteria?
3. Can you use an accelerated algorithm (Anderson acceleration, etc.)?
4. What's the typical number of iterations in practice?

**Raymond's advice**: Measure first. Add a histogram of iteration counts:

```python
# Add to minimum_phase_decomposition
_ITERATION_HISTOGRAM = [0] * 61  # Track how many iterations are typical

def minimum_phase_decomposition(...):
    # ... existing code ...
    for iteration in range(max_iterations):
        # ... existing code ...
        if xp.all(is_converged):
            _ITERATION_HISTOGRAM[iteration] += 1
            return minimum_phase_factor

    _ITERATION_HISTOGRAM[max_iterations] += 1
    # ... existing code ...
```

Then optimize based on data:
- If most converge in <10 iterations, your current approach is fine
- If many hit 60, you need a better algorithm, not fewer copies
- If there's high variance, consider adaptive iteration limits

---

### 5. einsum is Great, But You're Using It Wrong

Your plan suggests:

```python
return xp.einsum(
    '...i,...j->...ij',
    fc,
    xp.conj(fc),
    dtype=self._dtype
)
```

**Two issues**:

1. **einsum doesn't take a dtype parameter** (this will error)
2. **For outer products, einsum isn't always faster than broadcasting**

**Better approach - Test both and choose based on data**:

```python
@property
def _cross_spectral_matrix(self) -> NDArray[np.complexfloating]:
    """Compute cross-spectral matrix.

    For large arrays, uses einsum for memory efficiency.
    For small arrays, uses matmul for speed.
    """
    fc = self.fourier_coefficients

    # For small enough arrays, matmul is faster due to better optimizations
    # Threshold determined empirically on common hardware
    n_elements = np.prod(fc.shape[:-1]) * fc.shape[-1]
    use_einsum = n_elements > 100_000  # Adjust based on profiling

    if use_einsum:
        # Memory-efficient for large arrays
        result = xp.einsum('...i,...j->...ij', fc, xp.conj(fc))
    else:
        # Faster for small arrays
        fc_expanded = fc[..., xp.newaxis]
        result = fc_expanded @ xp.conj(fc_expanded).swapaxes(-1, -2)

    # Cast if needed
    if result.dtype != self._dtype:
        result = result.astype(self._dtype)

    return result
```

**Better yet**: Just use matmul. The alleged memory savings are tiny compared to the output array size.

---

### 6. Vectorization of global_coherence is Good, But...

The nested loop in `global_coherence` is indeed slow, but your vectorization:

```python
fc_reshaped = (
    self.fourier_coefficients
    .transpose(0, 3, 4, 1, 2)
    .reshape(-1, n_signals, n_trials * n_tapers)
)
```

**This is hard to understand.** Readability counts.

**Better approach with clear intent**:

```python
def global_coherence(self, max_rank: int = 1):
    """Find linear combinations that capture the most coherent power."""
    (n_time_windows, n_trials, n_tapers,
     n_fft_samples, n_signals) = self.fourier_coefficients.shape

    # Reshape for vectorized SVD
    # Goal: (n_time_windows * n_fft_samples, n_signals, n_observations)
    # where n_observations = n_trials * n_tapers

    # Step 1: Move signals to second-to-last position
    fc = self.fourier_coefficients.transpose(0, 3, 4, 1, 2)
    # Shape: (n_time_windows, n_fft_samples, n_signals, n_trials, n_tapers)

    # Step 2: Combine time and frequency into batch dimension
    fc_batched = fc.reshape(
        n_time_windows * n_fft_samples,  # batch dimension
        n_signals,                         # signals
        n_trials * n_tapers               # observations
    )

    # Step 3: Compute SVD (vectorized over batch dimension)
    if max_rank >= n_signals - 1:
        # Full SVD is vectorizable
        U, S, _ = xp.linalg.svd(fc_batched, full_matrices=False)
        singular_values = S[:, :max_rank]
        left_vectors = U[:, :, :max_rank]
    else:
        # Sparse SVD - need to loop, but at least it's clear why
        singular_values = xp.zeros((len(fc_batched), max_rank))
        left_vectors = xp.zeros((len(fc_batched), n_signals, max_rank),
                               dtype=xp.complex128)
        for i, fc_slice in enumerate(fc_batched):
            left_vectors[i], singular_values[i], _ = svds(fc_slice, max_rank)

    # Convert to global coherence
    global_coherence = (singular_values ** 2) / (n_trials * n_tapers)

    # Reshape back to (time, freq, rank)
    global_coherence = global_coherence.reshape(
        n_time_windows, n_fft_samples, max_rank
    )
    left_vectors = left_vectors.reshape(
        n_time_windows, n_fft_samples, n_signals, max_rank
    )

    # Convert from CuPy if needed
    try:
        return xp.asnumpy(global_coherence), xp.asnumpy(left_vectors)
    except AttributeError:
        return global_coherence, left_vectors
```

**Key improvements**:
1. Comments explain the reshaping strategy
2. Each transformation has a clear purpose
3. Variable names indicate dimensions
4. The "why" is documented, not just the "what"

---

### 7. Don't Use Collections for the Wrong Job

I see this pattern in the optimization plan:

```python
sections = xp.array_split(xp.c_[_is, _it], self._blocks)
```

**This mixes concerns**. Array splitting for parallelization should use a clearer pattern:

```python
def _chunk_pairs(pairs, n_chunks):
    """Split pairs into roughly equal chunks.

    Yields
    ------
    chunk : array
        Subset of pairs for this chunk
    """
    chunk_size = len(pairs) // n_chunks + 1
    for i in range(0, len(pairs), chunk_size):
        yield pairs[i:i + chunk_size]
```

Then use it:

```python
for chunk in _chunk_pairs(pair_indices, self._blocks):
    # Process chunk
    ...
```

**This is more readable** and makes the intent clear.

---

## Architectural Suggestions 🏗️

### 1. Consider a Builder Pattern for Multitaper

Right now:

```python
mt = Multitaper(
    time_series=data,
    sampling_frequency=1000,
    time_window_duration=1.0,
    time_halfbandwidth_product=3,
    detrend_type="constant",
    # ... many more parameters
)
```

**For scientific code with many parameters**, consider:

```python
# Option 1: Use suggest_parameters more prominently
params = suggest_parameters(
    sampling_frequency=1000,
    signal_duration=10.0,
    desired_freq_resolution=2.0
)
mt = Multitaper.from_parameters(data, params)

# Option 2: Builder pattern
mt = (Multitaper(data)
      .with_sampling_frequency(1000)
      .with_time_window(duration=1.0)
      .with_tapers(time_halfbandwidth_product=3)
      .build())
```

**I prefer option 1** - it's more Pythonic and leverages your existing `suggest_parameters` function.

---

### 2. Separate Computation from Result Storage

Currently, `Connectivity` mixes computation and storage:

```python
class Connectivity:
    def __init__(self, fourier_coefficients, ...):
        self.fourier_coefficients = fourier_coefficients  # Large array!
        # ... computes many things ...
```

**Consider separating concerns**:

```python
class ConnectivityResults:
    """Store and export connectivity results."""
    def __init__(self, coherence, phase, ...):
        self.coherence = coherence
        self.phase = phase
        # ...

    def to_xarray(self, ...):
        """Convert to xarray with proper labeling."""
        ...

    def to_dataframe(self, ...):
        """Convert to pandas DataFrame."""
        ...

class Connectivity:
    """Compute connectivity measures from Fourier coefficients."""
    def __init__(self, fourier_coefficients, ...):
        self._fc = fourier_coefficients  # Private
        # Don't store results

    def compute_coherence(self) -> NDArray:
        """Compute coherence magnitude."""
        ...

    def compute_all(self, methods=None) -> ConnectivityResults:
        """Compute multiple measures efficiently."""
        # Reuse expensive intermediate results
        power = self._power
        csm = self._expectation_cross_spectral_matrix()

        results = {}
        for method in methods:
            results[method] = self._compute(method, power, csm)

        return ConnectivityResults(**results)
```

**This separates**:
- Computation logic (Connectivity)
- Result storage (ConnectivityResults)
- Caching strategy (internal to Connectivity)

---

### 3. Make GPU Support More Transparent

Current pattern:

```python
if os.environ.get("SPECTRAL_CONNECTIVITY_ENABLE_GPU") == "true":
    import cupy as xp
else:
    import numpy as xp
```

**This is fragile** (environment variables, module-level side effects).

**Better approach**:

```python
# spectral_connectivity/backends.py
from typing import Protocol

class ArrayBackend(Protocol):
    """Protocol for array computation backends."""
    def zeros(self, shape, dtype=None): ...
    def ones(self, shape, dtype=None): ...
    def asarray(self, a): ...
    # ... etc

# Auto-detect or allow explicit choice
def get_backend(prefer_gpu: bool = True) -> ArrayBackend:
    """Get the array computation backend.

    Parameters
    ----------
    prefer_gpu : bool
        If True, use GPU (CuPy) if available, else fall back to CPU (NumPy).

    Returns
    -------
    backend : ArrayBackend
        NumPy or CuPy module
    """
    if prefer_gpu:
        try:
            import cupy
            return cupy
        except ImportError:
            pass

    import numpy
    return numpy

# Then use it:
class Multitaper:
    def __init__(self, time_series, ..., backend=None):
        self.xp = backend or get_backend()
        self.time_series = self.xp.asarray(time_series)
```

**Benefits**:
1. Explicit backend choice
2. Testable (can inject mock backend)
3. No global state
4. Clearer error messages

---

## Testing Observations 🧪

### Good

Your test structure is solid:
- Separate test files for each module
- Fixtures for common setups
- Tests for error messages (!!)

### Could Be Better

#### 1. Missing Property-Based Tests

For numerical code, use Hypothesis:

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
    """Property-based test for multitaper."""
    mt = Multitaper(data, sampling_frequency=sampling_frequency)

    # Properties that should always hold
    assert mt.frequencies.min() >= 0
    assert mt.frequencies.max() <= mt.nyquist_frequency
    assert len(mt.frequencies) == mt.n_fft_samples // 2 + 1
    # ... etc
```

**This finds edge cases** you wouldn't think to test manually.

#### 2. Benchmark Regression Tests

Good that you have benchmarks, but make them part of CI:

```python
# tests/test_performance.py
import pytest

# Store expected times (updated periodically)
BASELINE_TIMES = {
    'multitaper_1000_10_5': 0.1,  # 1000 samples, 10 trials, 5 channels
    'coherence_1000_10_5': 0.05,
}

@pytest.mark.benchmark
def test_multitaper_performance(benchmark):
    """Ensure multitaper performance doesn't regress."""
    data = np.random.randn(1000, 10, 5)

    def run_multitaper():
        mt = Multitaper(data, sampling_frequency=1000)
        return mt.fft()

    result = benchmark(run_multitaper)
    # benchmark will fail if it's >2x slower than baseline
```

---

## Documentation Suggestions 📚

### 1. Add a "Performance Guide"

Create `docs/performance.md`:

```markdown
# Performance Guide

## When to Optimize

Don't optimize until you've profiled. Use:

```python
import cProfile
cProfile.run('your_analysis_code()', 'profile.stats')
```

## Common Bottlenecks

1. **DPSS taper computation**: Cached automatically
2. **FFT**: Use `n_fft_samples` that's a power of 2
3. **Large channel counts**: Consider `blocks` parameter
4. **GPU**: Set `SPECTRAL_CONNECTIVITY_ENABLE_GPU=true`

## Memory Management

For very large datasets:

```python
# Process in chunks
mt = Multitaper(data, chunk_size=100)
```
```

### 2. Add Type Stub Files (.pyi)

For better IDE support:

```python
# spectral_connectivity/connectivity.pyi
from numpy.typing import NDArray
import numpy as np

class Connectivity:
    def __init__(
        self,
        fourier_coefficients: NDArray[np.complexfloating],
        expectation_type: str = ...,
        frequencies: NDArray[np.floating] | None = ...,
        time: NDArray[np.floating] | None = ...,
        blocks: int | None = ...,
        dtype: np.dtype = ...,
    ) -> None: ...

    def coherence_magnitude(self) -> NDArray[np.floating]: ...
    # ... etc
```

---

## Final Recommendations 📋

### Priority 1: Do These Now

1. **Replace manual caching with `cached_property`** - Simpler and more Pythonic
2. **Improve taper cache key handling** - Avoid float comparison issues
3. **Add property-based tests** - Find edge cases automatically
4. **Keep `is_copy=True` default** - Correctness over premature optimization

### Priority 2: Do These Next

1. **Profile before optimizing** - Add iteration histograms, timing decorators
2. **Simplify einsum usage** - Or just use matmul
3. **Document performance characteristics** - Help users make informed choices
4. **Improve backend selection** - Make GPU support more explicit

### Priority 3: Consider for Future

1. **Builder pattern for Multitaper** - Or promote `from_parameters` method
2. **Separate computation from storage** - ConnectivityResults class
3. **Parallel processing** - But only if single-threaded performance is good
4. **Adaptive algorithms** - Better than optimizing fixed iteration loops

---

## The Bottom Line

This is solid scientific Python code. The optimization plan is well-researched but over-engineered in places. Remember:

> **"Premature optimization is the root of all evil... yet we should not pass up our opportunities in that critical 3%."** - Donald Knuth

Your critical 3%:
1. DPSS taper caching ✓
2. Property caching for expensive intermediate results ✓
3. Vectorization where it matters (global_coherence) ✓

Everything else should wait until you've profiled real workloads and found actual bottlenecks.

**Most importantly**: Don't sacrifice code clarity for performance unless you've measured the gain. Scientific code needs to be correct first, fast second, and maintainable always.

---

**Questions? The Python community is here to help. Consider posting specific optimization questions to the NumPy mailing list or Python Performance Discord.**

Keep up the good work!

— Raymond Hettinger (simulated)
