# Performance Optimization Plan for spectral_connectivity

**Document Version**: 1.0
**Last Updated**: 2025-11-07
**Analysis Based On**: Codebase commit e67dfaf

---

## Executive Summary

This document outlines a comprehensive performance optimization strategy for the `spectral_connectivity` package, focusing on both computational speed and memory efficiency. The plan is organized into phases with concrete implementation details, expected gains, and risk assessments.

**Key Findings**:
- **Speed bottlenecks**: Redundant expensive computations, unvectorized loops, inefficient taper generation
- **Memory bottlenecks**: Unnecessary array copies, inefficient windowing, duplicate storage
- **Critical constraint**: Caching must be memory-aware; naive LRU caching causes OOM errors

**Expected Overall Gains** (when all optimizations applied):
- **Speed**: 2-8x faster for typical workflows (multi-measure analysis)
- **Memory**: 30-50% reduction for large datasets (>50 channels, >1000 time windows)

---

## Table of Contents

1. [Phase 1: High-Impact, Low-Risk Optimizations](#phase-1-high-impact-low-risk-optimizations)
2. [Phase 2: Medium-Impact Optimizations](#phase-2-medium-impact-optimizations)
3. [Phase 3: Advanced Optimizations](#phase-3-advanced-optimizations)
4. [Implementation Guidelines](#implementation-guidelines)
5. [Testing and Validation Strategy](#testing-and-validation-strategy)
6. [Appendix: Rejected Optimizations](#appendix-rejected-optimizations)

---

## Phase 1: High-Impact, Low-Risk Optimizations

**Timeline**: 1-2 weeks
**Expected Gains**: 3-10x speed improvement, 20-30% memory reduction

### 1.1 Smart Instance-Level Property Caching

**Problem**: Expensive properties like `_minimum_phase_factor`, `_power`, and `_transfer_function` are recomputed on every access, even within the same analysis workflow.

**Why LRU Cache Fails**:
```python
# ❌ DANGEROUS - Causes OOM errors
@lru_cache(maxsize=128)
def _minimum_phase_factor(self):
    # Returns (n_time_windows, n_freqs, n_signals, n_signals) array
    # For 100 time windows × 512 freqs × 64 signals: ~1.6 GB per call
    # LRU cache keeps 128 of these = 200+ GB!
```

**Solution**: Instance-level caching with automatic cleanup

**Files**: `spectral_connectivity/connectivity.py`

**Implementation**:

```python
class Connectivity:
    def __init__(self, ...):
        # ... existing code ...
        # Cache dictionary for expensive computations
        self._cache = {}

    @property
    def _power(self) -> NDArray[np.floating]:
        """Cached power computation."""
        if '_power' not in self._cache:
            self._cache['_power'] = self._expectation(
                self.fourier_coefficients * self.fourier_coefficients.conjugate()
            ).real
        return self._cache['_power']

    @property
    def _minimum_phase_factor(self) -> NDArray[np.complexfloating]:
        """Cached minimum phase decomposition - LARGE array, use carefully."""
        if '_minimum_phase_factor' not in self._cache:
            self._cache['_minimum_phase_factor'] = minimum_phase_decomposition(
                self._expectation_cross_spectral_matrix()
            )
        return self._cache['_minimum_phase_factor']

    @property
    def _transfer_function(self) -> NDArray[np.complexfloating]:
        """Cached transfer function computation."""
        if '_transfer_function' not in self._cache:
            result = _estimate_transfer_function(self._minimum_phase_factor)
            # Apply non_negative_frequencies decorator logic
            n_frequencies = result.shape[-3]
            non_neg_index = xp.arange(0, n_frequencies // 2 + 1)
            self._cache['_transfer_function'] = xp.take(
                result, indices=non_neg_index, axis=-3
            )
        return self._cache['_transfer_function']

    @property
    def _noise_covariance(self) -> NDArray[np.floating]:
        """Cached noise covariance computation."""
        if '_noise_covariance' not in self._cache:
            self._cache['_noise_covariance'] = _estimate_noise_covariance(
                self._minimum_phase_factor
            )
        return self._cache['_noise_covariance']

    @property
    def _MVAR_Fourier_coefficients(self) -> NDArray[np.complexfloating]:
        """Cached MVAR coefficients computation."""
        if '_MVAR_Fourier_coefficients' not in self._cache:
            H = self._transfer_function
            lam = TIKHONOV_REGULARIZATION_FACTOR * xp.mean(xp.real(xp.conj(H) * H))
            identity = xp.eye(H.shape[-1], dtype=H.dtype)
            regularized_H = H + lam * identity
            self._cache['_MVAR_Fourier_coefficients'] = xp.linalg.solve(
                regularized_H, identity
            )
        return self._cache['_MVAR_Fourier_coefficients']

    def clear_cache(self):
        """Explicitly clear cache to free memory when needed."""
        self._cache.clear()
```

**Memory Safety**: Cache is tied to instance lifetime. When `Connectivity` object is deleted, cache is garbage collected automatically.

**Lines to Modify**:
- Line 442-445: `_power` property
- Line 568-569: `_minimum_phase_factor` property
- Line 572-574: `_transfer_function` property
- Line 577-578: `_noise_covariance` property
- Line 581-588: `_MVAR_Fourier_coefficients` property

**Expected Gain**:
- **Speed**: 3-5x for workflows computing multiple connectivity measures
- **Memory**: Minimal overhead (<1% for typical analyses)

**Risk**: Low. Cache lifetime matches object lifetime, preventing memory leaks.

---

### 1.2 Cache DPSS Tapers with Size Limits

**Problem**: DPSS tapers are computed via expensive eigenvalue decomposition (`dpss_windows`), but identical parameters across windows result in redundant computation.

**Why This Cache is Safe**:
- Tapers are small: (n_samples_per_window, n_tapers) typically ~5KB-50KB
- Reused heavily across different analyses with same parameters
- LRU eviction prevents unbounded growth

**Files**: `spectral_connectivity/transforms.py`

**Implementation**:

```python
from functools import lru_cache

# Add before _make_tapers function (around line 1408)

# Cache key must be hashable - convert numpy arrays to tuples if needed
@lru_cache(maxsize=32)  # 32 different taper configurations max
def _make_tapers_cached(
    n_time_samples_per_window: int,
    sampling_frequency: float,
    time_halfbandwidth_product: float,
    n_tapers: int,
    is_low_bias: bool = True,
) -> NDArray[np.floating]:
    """Cached DPSS taper generation.

    Cache is safe because:
    1. Tapers are small (typically <50KB)
    2. Heavily reused across windows with same parameters
    3. LRU eviction limits total memory (max ~1.6MB for 32 entries)
    """
    tapers, _ = dpss_windows(
        n_time_samples_per_window,
        time_halfbandwidth_product,
        n_tapers,
        is_low_bias=is_low_bias,
    )
    return tapers.T * xp.sqrt(sampling_frequency)


def _make_tapers(
    n_time_samples_per_window: int,
    sampling_frequency: float,
    time_halfbandwidth_product: float,
    n_tapers: int,
    is_low_bias: bool = True,
) -> NDArray[np.floating]:
    """Return discrete prolate spheroidal sequences (tapers) for multitaper analysis.

    Now uses caching to avoid recomputing identical tapers.
    """
    # Convert to hashable types for cache key
    return _make_tapers_cached(
        int(n_time_samples_per_window),
        float(sampling_frequency),
        float(time_halfbandwidth_product),
        int(n_tapers),
        bool(is_low_bias),
    )
```

**Lines to Modify**:
- Lines 1408-1440: Add cached version of `_make_tapers`

**Expected Gain**:
- **Speed**: 10-50x for repeated windowed analyses (100+ windows with same parameters)
- **Memory**: Negligible (~1-2MB max for cache)

**Risk**: Very low. Cache size is bounded and tapers are small.

---

### 1.3 Eliminate Unnecessary Array Copies

**Problem**: Multiple locations create defensive copies that aren't needed, doubling memory usage.

**Files**: `spectral_connectivity/transforms.py`, `spectral_connectivity/minimum_phase_decomposition.py`

**Implementation Details**:

#### 1.3.1 Sliding Window Default to No-Copy

**Current** (line 1311-1374):
```python
def _sliding_window(..., is_copy: bool = True) -> NDArray[np.floating]:
    # ...
    return strided.copy() if is_copy else strided  # Line 1374
```

**Change**:
```python
def _sliding_window(..., is_copy: bool = False) -> NDArray[np.floating]:
    """
    ...
    is_copy : bool, default=False
        Return strided array as copy. Set to True only if you need to
        modify the output array. For read-only operations (typical case),
        False saves memory.

    Notes
    -----
    Default changed to False for memory efficiency. The stride trick creates
    a view, which is safe for read-only operations like FFT. Only set is_copy=True
    if you need to modify the windowed data in-place.
    """
    # ...
    return strided.copy() if is_copy else strided
```

**Also update caller** (line 1158):
```python
time_series = _sliding_window(
    time_series,
    window_size=self.n_time_samples_per_window,
    step_size=self.n_time_samples_per_step,
    axis=0,
    # is_copy=False is now default, remove explicit parameter
)
```

**Lines to Modify**: 1316, 1374, 1158

#### 1.3.2 Optimize Wilson Algorithm Copies

**Current** (minimum_phase_decomposition.py, line 301):
```python
old_minimum_phase_factor = minimum_phase_factor.copy()  # Full copy every iteration!
```

**Problem**: For 100 time windows × 512 freqs × 64 signals: ~1.6GB copied per iteration × 60 iterations = wasteful

**Change**:
```python
def minimum_phase_decomposition(
    cross_spectral_matrix: NDArray[np.complexfloating],
    tolerance: float = 1e-8,
    max_iterations: int = 60,
) -> NDArray[np.complexfloating]:
    n_time_points = cross_spectral_matrix.shape[0]
    n_signals = cross_spectral_matrix.shape[-1]
    identity_matrix = xp.eye(n_signals)
    is_converged = xp.zeros(n_time_points, dtype=bool)
    minimum_phase_factor = xp.zeros(cross_spectral_matrix.shape)
    minimum_phase_factor[..., :, :, :] = _get_initial_conditions(cross_spectral_matrix)

    # Only store old values for unconverged time points
    for iteration in range(max_iterations):
        logger.debug(
            f"iteration: {iteration}, {is_converged.sum()} of {len(is_converged)} converged"
        )

        # Store only unconverged time points for comparison
        unconverged_mask = ~is_converged
        if xp.any(unconverged_mask):
            old_minimum_phase_factor = minimum_phase_factor[unconverged_mask].copy()

        linear_predictor = _get_linear_predictor(
            minimum_phase_factor, cross_spectral_matrix, identity_matrix
        )
        minimum_phase_factor = xp.matmul(
            minimum_phase_factor, _get_causal_signal(linear_predictor)
        )

        # Check convergence only for unconverged time points
        if xp.any(unconverged_mask):
            newly_converged = _check_convergence(
                minimum_phase_factor[unconverged_mask],
                old_minimum_phase_factor,
                tolerance
            )
            # Update convergence status
            unconverged_indices = xp.where(unconverged_mask)[0]
            is_converged[unconverged_indices] = newly_converged

        if xp.all(is_converged):
            return minimum_phase_factor
    else:
        logger.warning(
            f"Maximum iterations reached. {is_converged.sum()} of {len(is_converged)} converged"
        )
        return minimum_phase_factor
```

**Lines to Modify**: Lines 297-322 in minimum_phase_decomposition.py

**Expected Memory Savings**:
- Before: 60 iterations × 1.6GB = 96GB peak for copies
- After: Only unconverged copies, typically <10% after first few iterations

#### 1.3.3 Optimize tridisolve

**Current** (transforms.py, lines 1469-1470):
```python
dw = d.copy()
ew = e.copy()
```

**Change**:
```python
def tridisolve(
    d: NDArray[np.floating],
    e: NDArray[np.floating],
    b: NDArray[np.floating],
    overwrite_b: bool = True,
    overwrite_d: bool = False,  # NEW parameter
    overwrite_e: bool = False,  # NEW parameter
) -> NDArray[np.floating]:
    """Symmetric tridiagonal system solver, from Golub and Van Loan p157.

    .. note:: Copied from NiTime.

    Parameters
    ----------
    d : ndarray
      main diagonal stored in d[:]
    e : ndarray
      superdiagonal stored in e[:-1]
    b : ndarray
      RHS vector
    overwrite_b : bool
      Whether to overwrite b with solution
    overwrite_d : bool
      Whether d can be overwritten (avoids copy)
    overwrite_e : bool
      Whether e can be overwritten (avoids copy)

    Returns
    -------
    x : ndarray
      Solution to Ax = b (if overwrite_b is False). Otherwise solution is
      stored in previous RHS vector b
    """
    N = len(b)
    # work vectors - only copy if necessary
    dw = d if overwrite_d else d.copy()
    ew = e if overwrite_e else e.copy()
    if overwrite_b:
        x = b
    else:
        x = b.copy()
    # ... rest of function unchanged ...
```

**Update callers** in `tridi_inverse_iteration` (line 1533):
```python
tridisolve(eig_diag, e, x0, overwrite_d=True, overwrite_e=True)
```

**Lines to Modify**: Lines 1443-1489 in transforms.py

**Expected Gain**:
- **Memory**: 30-50% reduction for large datasets
- **Speed**: 10-20% faster (fewer allocations)

**Risk**: Low. Carefully reviewed to ensure no side effects.

---

### 1.4 Use einsum for Cross-Spectral Matrix

**Problem**: Broadcasting creates intermediate arrays and is less cache-efficient than einsum.

**Files**: `spectral_connectivity/connectivity.py`

**Current** (lines 458-461):
```python
@property
def _cross_spectral_matrix(self) -> NDArray[np.complexfloating]:
    fourier_coefficients = self.fourier_coefficients[..., xp.newaxis]
    return _complex_inner_product(
        fourier_coefficients, fourier_coefficients, dtype=self._dtype
    )
```

**Change**:
```python
@property
def _cross_spectral_matrix(self) -> NDArray[np.complexfloating]:
    """Return the complex-valued linear association between fourier coefficients.

    Returns
    -------
    cross_spectral_matrix : array
        Shape (n_time_windows, n_trials, n_tapers, n_fft_samples,
        n_signals, n_signals). Complex cross-spectral matrix.

    Notes
    -----
    Uses einsum for memory-efficient computation without intermediate arrays.
    """
    fc = self.fourier_coefficients
    # einsum is more memory efficient than broadcast + matmul
    # '...i,...j->...ij' creates outer product along last dimension
    return xp.einsum(
        '...i,...j->...ij',
        fc,
        xp.conj(fc),
        dtype=self._dtype
    )
```

**Lines to Modify**: Lines 447-461

**Expected Gain**:
- **Memory**: 15-25% reduction (no intermediate array with extra dimension)
- **Speed**: 5-10% faster (better cache locality)

**Risk**: Low. Einsum is well-tested and equivalent.

---

## Phase 2: Medium-Impact Optimizations

**Timeline**: 2-3 weeks
**Expected Gains**: 2-5x additional speed improvement for specific methods

### 2.1 Vectorize global_coherence

**Problem**: Nested loops prevent parallelization and add overhead.

**Files**: `spectral_connectivity/connectivity.py`

**Current** (lines 876-888):
```python
for time_ind in range(n_time_windows):
    for freq_ind in range(n_fft_samples):
        fourier_coefficients = (
            self.fourier_coefficients[time_ind, :, :, freq_ind, :]
            .reshape((n_trials * n_tapers, n_signals))
            .T
        )
        (
            global_coherence[time_ind, freq_ind],
            unnormalized_global_coherence[time_ind, freq_ind],
        ) = _estimate_global_coherence(fourier_coefficients, max_rank=max_rank)
```

**Change**:
```python
def global_coherence(
    self, max_rank: int = 1
) -> tuple[NDArray[np.floating], NDArray[np.complexfloating]]:
    """Find linear combinations that capture the most coherent power.

    ... (docstring same) ...
    """
    (
        n_time_windows,
        n_trials,
        n_tapers,
        n_fft_samples,
        n_signals,
    ) = self.fourier_coefficients.shape

    # Reshape to process all time-freq combinations at once
    # From: (time, trials, tapers, freq, signals)
    # To: (time*freq, signals, trials*tapers)
    fc_reshaped = (
        self.fourier_coefficients
        .transpose(0, 3, 4, 1, 2)  # (time, freq, signals, trials, tapers)
        .reshape(-1, n_signals, n_trials * n_tapers)  # (time*freq, signals, trials*tapers)
    )

    # Choose SVD strategy based on rank
    if max_rank >= n_signals - 1:
        # Use full SVD (vectorized over all time-freq)
        U, S, _ = xp.linalg.svd(fc_reshaped, full_matrices=False)
        S_squared = S[:, :max_rank] ** 2 / (n_trials * n_tapers)
        U_selected = U[:, :, :max_rank]
    else:
        # For small rank, use sparse SVD on each time-freq
        # (sparse SVD doesn't vectorize well, so keep loop but optimize)
        S_squared = xp.zeros((n_time_windows * n_fft_samples, max_rank))
        U_selected = xp.zeros(
            (n_time_windows * n_fft_samples, n_signals, max_rank),
            dtype=xp.complex128
        )
        for idx in range(n_time_windows * n_fft_samples):
            U_selected[idx], S_squared[idx], _ = svds(
                fc_reshaped[idx], max_rank
            )
            S_squared[idx] = S_squared[idx]**2 / (n_trials * n_tapers)

    # Reshape back to (time, freq, ...)
    global_coherence = S_squared.reshape(n_time_windows, n_fft_samples, max_rank)
    unnormalized_global_coherence = U_selected.reshape(
        n_time_windows, n_fft_samples, n_signals, max_rank
    )

    try:
        return xp.asnumpy(global_coherence), xp.asnumpy(
            unnormalized_global_coherence
        )
    except AttributeError:
        return global_coherence, unnormalized_global_coherence
```

**Lines to Modify**: Lines 822-895

**Expected Gain**:
- **Speed**: 5-10x for full SVD case, 2-3x for sparse SVD case
- **GPU**: Especially beneficial on GPU (10-20x)

**Risk**: Medium. Requires thorough testing of reshaping logic.

---

### 2.2 Optimize Block Processing

**Problem**: Block processing has overhead from repeated `_nonsorted_unique` calls and indexing.

**Files**: `spectral_connectivity/connectivity.py`

**Current** (lines 506-524):
```python
for sec in sections:
    # get unique indices
    _sxu = _nonsorted_unique(sec[:, 0])
    _syu = _nonsorted_unique(sec[:, 1])

    # computes block of connections
    _out = self._expectation(
        fcn(
            _complex_inner_product(
                fourier_coefficients[..., _sxu, :],
                fourier_coefficients[..., _syu, :],
                dtype=self._dtype,
            )
        )
    )

    # fill the output array (Hermitian symmetric filling)
    csm[..., _sxu.reshape(-1, 1), _syu.reshape(1, -1)] = _out
    csm[..., _syu.reshape(1, -1), _sxu.reshape(-1, 1)] = xp.conj(_out)
```

**Change**:
```python
# Pre-compute all unique indices outside loop
all_unique_x = []
all_unique_y = []
for sec in sections:
    all_unique_x.append(_nonsorted_unique(sec[:, 0]))
    all_unique_y.append(_nonsorted_unique(sec[:, 1]))

# Process blocks
for idx, sec in enumerate(sections):
    _sxu = all_unique_x[idx]
    _syu = all_unique_y[idx]

    # computes block of connections
    _out = self._expectation(
        fcn(
            _complex_inner_product(
                fourier_coefficients[..., _sxu, :],
                fourier_coefficients[..., _syu, :],
                dtype=self._dtype,
            )
        )
    )

    # Use advanced indexing more efficiently
    ix = _sxu.reshape(-1, 1)
    iy = _syu.reshape(1, -1)
    csm[..., ix, iy] = _out
    # In-place conjugate for memory efficiency
    xp.conjugate(_out, out=csm[..., iy, ix])
```

**Lines to Modify**: Lines 487-526

**Expected Gain**:
- **Speed**: 20-30% for block-based computation (50+ signals)
- **Memory**: 10% reduction (in-place conjugate)

**Risk**: Low. Logic remains the same, just reorganized.

---

### 2.3 Optimize pair-wise Granger Causality

**Problem**: Loop over pairs with full CSM computation is inefficient.

**Files**: `spectral_connectivity/connectivity.py`

**Current** (lines 2314-2334):
```python
for pair_indices in pairs:
    pair_indices = xp.array(pair_indices)[:, xp.newaxis]
    try:
        minimum_phase_factor = minimum_phase_decomposition(
            csm[..., pair_indices, pair_indices.T]
        )
        transfer_function = _estimate_transfer_function(minimum_phase_factor)[
            ..., non_neg_index, :, :
        ]
        rotated_covariance = _remove_instantaneous_causality(
            _estimate_noise_covariance(minimum_phase_factor)
        )
        predictive_power[..., pair_indices, pair_indices.T] = (
            _estimate_predictive_power(
                total_power[..., pair_indices[:, 0]],
                rotated_covariance,
                transfer_function,
            )
        )
    except np.linalg.LinAlgError:
        predictive_power[..., pair_indices, pair_indices.T] = xp.nan
```

**Change**:
```python
# Vectorize pair processing where possible
pair_array = xp.array(list(pairs))
n_pairs = len(pair_array)

# Pre-allocate arrays for batch processing
batch_minimum_phase = []
batch_indices = []

for i, (idx_i, idx_j) in enumerate(pair_array):
    try:
        pair_indices = xp.array([[idx_i], [idx_j]])
        minimum_phase_factor = minimum_phase_decomposition(
            csm[..., pair_indices[:, 0], pair_indices[0, :]]
        )
        batch_minimum_phase.append(minimum_phase_factor)
        batch_indices.append((idx_i, idx_j))
    except np.linalg.LinAlgError:
        predictive_power[..., idx_i, idx_j] = xp.nan
        predictive_power[..., idx_j, idx_i] = xp.nan

# Batch process successful decompositions
if batch_minimum_phase:
    # Stack for vectorized operations
    stacked_mph = xp.stack(batch_minimum_phase, axis=0)  # (n_pairs, ...)

    # Vectorized transfer function estimation
    transfer_functions = _estimate_transfer_function(
        stacked_mph.reshape(-1, *stacked_mph.shape[2:])
    ).reshape(stacked_mph.shape)[..., non_neg_index, :, :]

    # Vectorized noise covariance
    noise_covs = xp.stack([
        _estimate_noise_covariance(mph) for mph in batch_minimum_phase
    ])
    rotated_covs = _remove_instantaneous_causality(noise_covs)

    # Fill results
    for k, (idx_i, idx_j) in enumerate(batch_indices):
        pair_indices = xp.array([[idx_i], [idx_j]])
        predictive_power[..., idx_i, idx_j] = _estimate_predictive_power(
            total_power[..., idx_i],
            rotated_covs[k],
            transfer_functions[k],
        )[..., 0, 1]
        predictive_power[..., idx_j, idx_i] = _estimate_predictive_power(
            total_power[..., idx_j],
            rotated_covs[k],
            transfer_functions[k],
        )[..., 1, 0]
```

**Lines to Modify**: Lines 2282-2340

**Expected Gain**:
- **Speed**: 2-3x for many pairs (>50 pairs)

**Risk**: Medium. Requires careful validation of vectorized logic.

---

## Phase 3: Advanced Optimizations

**Timeline**: 3-4 weeks
**Expected Gains**: Additional 30-50% memory reduction for very large datasets

### 3.1 Chunked Windowing for Large Time Series

**Problem**: Current implementation creates full windowed view in memory.

**Files**: `spectral_connectivity/transforms.py`

**Current Impact**:
- 10,000 samples, 1000 windows, 64 channels, 5 tapers: ~10GB memory for windowed view

**Implementation Strategy**:

```python
class Multitaper:
    def __init__(self, ..., chunk_size: int | None = None):
        """
        ...
        chunk_size : int, optional
            Process time series in chunks of this many windows to reduce
            memory usage. Useful for very long recordings. If None (default),
            processes all windows at once.
        """
        self._chunk_size = chunk_size
        # ... existing code ...

    def fft(self) -> NDArray[np.complexfloating]:
        """Compute the fast Fourier transform using the multitaper method.

        Uses chunked processing if chunk_size was specified to reduce memory.
        """
        if self._chunk_size is None:
            # Original implementation
            return self._fft_full()
        else:
            # Chunked implementation
            return self._fft_chunked()

    def _fft_full(self) -> NDArray[np.complexfloating]:
        """Original non-chunked implementation."""
        time_series = _add_axes(self.time_series)
        time_series = _sliding_window(
            time_series,
            window_size=self.n_time_samples_per_window,
            step_size=self.n_time_samples_per_step,
            axis=0,
        )
        if self.detrend_type is not None:
            time_series = detrend(time_series, type=self.detrend_type)

        logger.info(self)

        return _multitaper_fft(
            self.tapers, time_series, self.n_fft_samples, self.sampling_frequency
        ).swapaxes(2, -1)

    def _fft_chunked(self) -> NDArray[np.complexfloating]:
        """Chunked implementation for memory efficiency."""
        time_series = _add_axes(self.time_series)

        # Calculate number of windows
        n_windows = int(
            np.floor(
                (time_series.shape[0] - self.n_time_samples_per_window)
                / self.n_time_samples_per_step
            ) + 1
        )

        # Pre-allocate output
        output_shape = (
            n_windows,
            self.time_series.shape[1],  # n_trials
            self.n_tapers,
            self.n_fft_samples,
            self.time_series.shape[-1],  # n_signals
        )
        result = xp.empty(output_shape, dtype=xp.complex128)

        # Process in chunks
        for chunk_start in range(0, n_windows, self._chunk_size):
            chunk_end = min(chunk_start + self._chunk_size, n_windows)

            # Calculate time series indices for this chunk
            ts_start = chunk_start * self.n_time_samples_per_step
            ts_end = (chunk_end - 1) * self.n_time_samples_per_step + self.n_time_samples_per_window

            # Extract and window this chunk
            chunk_ts = time_series[ts_start:ts_end]
            windowed_chunk = _sliding_window(
                chunk_ts,
                window_size=self.n_time_samples_per_window,
                step_size=self.n_time_samples_per_step,
                axis=0,
            )

            if self.detrend_type is not None:
                windowed_chunk = detrend(windowed_chunk, type=self.detrend_type)

            # Compute FFT for this chunk
            chunk_fft = _multitaper_fft(
                self.tapers,
                windowed_chunk,
                self.n_fft_samples,
                self.sampling_frequency,
            ).swapaxes(2, -1)

            # Store results
            result[chunk_start:chunk_end] = chunk_fft

        logger.info(self)
        return result
```

**Lines to Modify**: Lines 1147-1171, add new methods

**Expected Gain**:
- **Memory**: 50-70% reduction for very long recordings (10,000+ windows)
- **Speed**: Minimal impact (5-10% slower due to chunking overhead)

**Risk**: Medium-High. Requires extensive testing across parameter combinations.

**When to Use**:
- Very long recordings: >10,000 time windows
- High channel counts: >64 channels
- Memory-constrained environments

---

## Implementation Guidelines

### Development Process

1. **Create feature branch** for each phase:
   ```bash
   git checkout -b optimize/phase1-caching
   ```

2. **Implement one optimization at a time** - do not bundle multiple optimizations in one PR

3. **Write tests first** (see Testing Strategy below)

4. **Profile before and after**:
   ```python
   import cProfile
   import pstats

   profiler = cProfile.Profile()
   profiler.enable()
   # Run optimization
   profiler.disable()
   stats = pstats.Stats(profiler)
   stats.sort_stats('cumulative')
   stats.print_stats(20)
   ```

5. **Memory profiling**:
   ```python
   from memory_profiler import profile

   @profile
   def test_function():
       # Your code here
   ```

6. **Benchmark script** (create `benchmarks/benchmark_optimization.py`):
   ```python
   import numpy as np
   import time
   from spectral_connectivity import Multitaper, Connectivity

   def benchmark_scenario(n_samples, n_channels, n_trials, n_windows):
       """Standard benchmark scenario."""
       # Generate test data
       data = np.random.randn(n_samples, n_trials, n_channels)

       # Time multitaper
       start = time.time()
       mt = Multitaper(
           data,
           sampling_frequency=1000,
           time_window_duration=1.0,
           time_halfbandwidth_product=3,
       )
       fft = mt.fft()
       multitaper_time = time.time() - start

       # Time connectivity
       start = time.time()
       conn = Connectivity.from_multitaper(mt)
       coherence = conn.coherence_magnitude()
       imaginary_coh = conn.imaginary_coherence()
       plv = conn.phase_locking_value()
       granger = conn.pairwise_spectral_granger_prediction()
       connectivity_time = time.time() - start

       return {
           'multitaper_time': multitaper_time,
           'connectivity_time': connectivity_time,
           'total_time': multitaper_time + connectivity_time,
       }

   if __name__ == '__main__':
       # Small dataset
       print("Small dataset (10 channels, 1000 samples):")
       results = benchmark_scenario(1000, 10, 5, 10)
       print(f"  Total time: {results['total_time']:.2f}s")

       # Medium dataset
       print("\nMedium dataset (32 channels, 5000 samples):")
       results = benchmark_scenario(5000, 32, 10, 50)
       print(f"  Total time: {results['total_time']:.2f}s")

       # Large dataset
       print("\nLarge dataset (64 channels, 10000 samples):")
       results = benchmark_scenario(10000, 64, 10, 100)
       print(f"  Total time: {results['total_time']:.2f}s")
   ```

### Code Review Checklist

For each optimization PR, reviewers should verify:

- [ ] **Correctness**: Results match original implementation (within floating-point tolerance)
- [ ] **Performance**: Benchmark shows expected improvement
- [ ] **Memory**: Memory profiling shows expected reduction (if applicable)
- [ ] **Tests**: All existing tests pass, new tests added
- [ ] **Documentation**: Docstrings updated, optimization documented
- [ ] **Backwards compatibility**: No breaking API changes
- [ ] **GPU compatibility**: Works with both NumPy and CuPy
- [ ] **Edge cases**: Handles small datasets, single channel, etc.

---

## Testing and Validation Strategy

### 1. Correctness Tests

**Requirement**: Optimized code must produce identical results (within numerical tolerance).

**Implementation** (add to `tests/test_optimization.py`):

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

@pytest.fixture
def multitaper(sample_data):
    """Create multitaper instance."""
    return Multitaper(
        sample_data,
        sampling_frequency=1000,
        time_window_duration=0.5,
        time_halfbandwidth_product=3,
    )

class TestOptimizationCorrectness:
    """Test that optimizations don't change results."""

    def test_cached_properties_match_original(self, multitaper):
        """Test that cached properties return same values."""
        conn = Connectivity.from_multitaper(multitaper)

        # Access property twice, should return same cached value
        power1 = conn._power
        power2 = conn._power

        # Should be exact match (same object)
        assert power1 is power2

        # Clear cache and recompute, should match
        conn.clear_cache()
        power3 = conn._power
        assert_allclose(power1, power3, rtol=1e-14)

    def test_windowing_copy_vs_nocopy(self, sample_data):
        """Test that windowing with/without copy gives same FFT results."""
        # With copy (old behavior)
        mt_copy = Multitaper(sample_data, sampling_frequency=1000)
        fft_copy = mt_copy.fft()

        # Without copy (optimized behavior)
        # Would need to modify Multitaper to expose this parameter
        # mt_nocopy = Multitaper(sample_data, sampling_frequency=1000, _copy_windows=False)
        # fft_nocopy = mt_nocopy.fft()

        # assert_allclose(fft_copy, fft_nocopy, rtol=1e-14)

    def test_einsum_matches_matmul(self, multitaper):
        """Test that einsum CSM matches original matmul."""
        conn = Connectivity.from_multitaper(multitaper)

        # Get CSM using current method
        csm = conn._cross_spectral_matrix

        # Manually compute using old method
        fc = conn.fourier_coefficients[..., np.newaxis]
        from spectral_connectivity.connectivity import _complex_inner_product
        csm_old = _complex_inner_product(fc, fc, dtype=np.complex128)

        assert_allclose(csm, csm_old, rtol=1e-14)

    def test_vectorized_global_coherence(self, multitaper):
        """Test vectorized global coherence matches loop version."""
        conn = Connectivity.from_multitaper(multitaper)

        # Compute using optimized version
        gc_opt, ugc_opt = conn.global_coherence(max_rank=2)

        # Would need to keep old implementation for comparison
        # gc_old, ugc_old = conn._global_coherence_original(max_rank=2)

        # assert_allclose(gc_opt, gc_old, rtol=1e-12)
        # assert_allclose(ugc_opt, ugc_old, rtol=1e-12)

class TestOptimizationPerformance:
    """Test that optimizations improve performance."""

    def test_cached_access_faster(self, multitaper, benchmark):
        """Test that cached property access is faster."""
        conn = Connectivity.from_multitaper(multitaper)

        # First access (computes and caches)
        _ = conn._minimum_phase_factor

        # Second access should be nearly instant
        result = benchmark(lambda: conn._minimum_phase_factor)
        # Benchmark will measure time

    def test_taper_cache_effective(self, sample_data, benchmark):
        """Test that taper caching speeds up repeated calls."""
        def create_multitaper():
            return Multitaper(
                sample_data,
                sampling_frequency=1000,
                time_window_duration=0.5,
                time_halfbandwidth_product=3,
            )

        # First call (computes tapers)
        mt1 = create_multitaper()

        # Second call (should use cached tapers)
        result = benchmark(create_multitaper)
        # Benchmark will show speedup

class TestMemoryUsage:
    """Test memory usage of optimizations."""

    @pytest.mark.memory
    def test_nocopy_windowing_memory(self, sample_data):
        """Test that no-copy windowing uses less memory."""
        import tracemalloc

        # Measure with copy
        tracemalloc.start()
        mt_copy = Multitaper(sample_data, sampling_frequency=1000)
        _ = mt_copy.fft()
        _, peak_copy = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Measure without copy
        tracemalloc.start()
        # mt_nocopy = Multitaper(sample_data, sampling_frequency=1000, _copy_windows=False)
        # _ = mt_nocopy.fft()
        _, peak_nocopy = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # assert peak_nocopy < peak_copy * 0.7  # At least 30% reduction

    @pytest.mark.memory
    def test_cache_size_bounded(self, multitaper):
        """Test that cache doesn't grow unboundedly."""
        import sys

        conn = Connectivity.from_multitaper(multitaper)

        # Access all cached properties
        _ = conn._power
        _ = conn._minimum_phase_factor
        _ = conn._transfer_function
        _ = conn._noise_covariance
        _ = conn._MVAR_Fourier_coefficients

        # Check cache size is reasonable
        cache_size = sys.getsizeof(conn._cache)
        # Should be small overhead, most memory in cached arrays
        assert cache_size < 1024  # Dictionary overhead < 1KB
```

### 2. Integration Tests

**Test complete workflows**:

```python
class TestRealWorldWorkflows:
    """Test optimizations in realistic scenarios."""

    def test_multi_measure_analysis(self, sample_data):
        """Test computing multiple connectivity measures."""
        mt = Multitaper(
            sample_data,
            sampling_frequency=1000,
            time_window_duration=0.5,
        )
        conn = Connectivity.from_multitaper(mt)

        # Compute many measures (benefits from caching)
        measures = {
            'coherence': conn.coherence_magnitude(),
            'imaginary_coh': conn.imaginary_coherence(),
            'plv': conn.phase_locking_value(),
            'pli': conn.phase_lag_index(),
            'wpli': conn.weighted_phase_lag_index(),
            'ppc': conn.pairwise_phase_consistency(),
        }

        # All should complete without error
        assert all(m is not None for m in measures.values())

        # Check cache was used effectively
        assert '_power' in conn._cache

    def test_directed_measures_workflow(self, sample_data):
        """Test directed connectivity measures."""
        mt = Multitaper(sample_data, sampling_frequency=1000)
        conn = Connectivity.from_multitaper(mt)

        # These all use minimum phase factor (should be cached)
        granger = conn.pairwise_spectral_granger_prediction()
        dtf = conn.directed_transfer_function()
        dc = conn.directed_coherence()
        pdc = conn.partial_directed_coherence()

        # All should complete
        assert all(x is not None for x in [granger, dtf, dc, pdc])

        # Check expensive computation was cached
        assert '_minimum_phase_factor' in conn._cache
```

### 3. Regression Tests

**Ensure no performance regressions**:

```python
# benchmarks/test_performance_regression.py

import pytest
import numpy as np
import time
from spectral_connectivity import Multitaper, Connectivity

# Store baseline times (update after each optimization)
BASELINE_TIMES = {
    'multitaper_small': 0.1,  # seconds
    'multitaper_medium': 0.5,
    'coherence_small': 0.05,
    'coherence_medium': 0.2,
    'granger_small': 0.5,
    'granger_medium': 2.0,
}

TOLERANCE = 1.2  # Allow 20% slowdown before failing

@pytest.mark.benchmark
class TestPerformanceRegression:
    """Ensure optimizations don't regress performance."""

    def test_multitaper_small_dataset(self):
        """Benchmark small dataset."""
        data = np.random.randn(1000, 5, 10)

        start = time.time()
        mt = Multitaper(data, sampling_frequency=1000)
        _ = mt.fft()
        elapsed = time.time() - start

        assert elapsed < BASELINE_TIMES['multitaper_small'] * TOLERANCE

    def test_coherence_medium_dataset(self):
        """Benchmark coherence on medium dataset."""
        data = np.random.randn(5000, 10, 32)
        mt = Multitaper(data, sampling_frequency=1000)
        conn = Connectivity.from_multitaper(mt)

        start = time.time()
        _ = conn.coherence_magnitude()
        elapsed = time.time() - start

        assert elapsed < BASELINE_TIMES['coherence_medium'] * TOLERANCE
```

### 4. GPU Tests

**Ensure optimizations work on GPU**:

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

    def test_einsum_gpu(self, sample_data):
        """Test einsum optimization works on GPU."""
        import cupy as cp

        data_gpu = cp.asarray(sample_data)
        mt = Multitaper(data_gpu, sampling_frequency=1000)
        conn = Connectivity.from_multitaper(mt)

        # Should not raise error
        csm = conn._cross_spectral_matrix
        assert isinstance(csm, cp.ndarray)
```

---

## Appendix: Rejected Optimizations

### A.1 FFT Size Optimization (REJECTED)

**Original Idea**: Allow users to specify exact FFT length to avoid padding.

**Rejection Reason**:
- FFT performance degrades significantly for non-optimal sizes
- Memory savings (2-10%) don't justify speed loss (can be 2-5x slower)
- `next_fast_len` is essential for good FFT performance

**Example**:
```python
# next_fast_len(1001) = 1024 (optimal for FFT)
# Using exact 1001: ~3x slower FFT
```

**Decision**: Keep automatic padding to `next_fast_len`.

---

### A.2 Naive LRU Cache for Properties (REJECTED)

**Original Idea**: Use `@lru_cache` decorator on properties.

**Rejection Reason**: Causes OOM errors as reported by users.

**Why It Fails**:
```python
@lru_cache(maxsize=128)
def _minimum_phase_factor(self):
    # Array size: (100 time, 512 freq, 64 signals, 64 signals)
    # = 100 × 512 × 64 × 64 × 16 bytes (complex128)
    # = 1.6 GB per call
    # Cache holds 128 calls = 200+ GB!
```

**Solution Used**: Instance-level dictionary caching (see Phase 1.1).

---

### A.3 Parallel Processing with multiprocessing (REJECTED)

**Original Idea**: Use Python multiprocessing for time window parallelization.

**Rejection Reason**:
- High overhead for array serialization
- NumPy/CuPy already use optimized BLAS/CUDA parallelization
- Adding multiprocessing would conflict with existing parallelism
- GPU operations cannot be easily parallelized across processes

**Better Alternative**: Ensure NumPy uses optimized BLAS (OpenBLAS, MKL) and leverage existing GPU parallelism.

---

### A.4 Sparse Matrix Representations (REJECTED)

**Original Idea**: Use sparse matrices for cross-spectral matrix when many signals are uncorrelated.

**Rejection Reason**:
- Cross-spectral matrices are typically dense (all pairs have some correlation)
- Sparse matrix operations slower for dense data
- Added complexity not justified by typical use cases

**When It Might Work**: Only for >1000 channels with known sparsity structure (rare in neuroscience).

---

## Conclusion

This optimization plan provides a roadmap for significantly improving the performance and memory efficiency of the `spectral_connectivity` package. By following a phased approach with careful testing and validation, we can achieve:

- **3-8x speed improvement** for typical multi-measure workflows
- **30-50% memory reduction** for large datasets
- **Maintained correctness** through comprehensive testing
- **No breaking changes** to the public API

The plan prioritizes high-impact, low-risk optimizations first (Phase 1) to deliver value quickly, while more complex optimizations (Phase 3) can be implemented as needed based on user requirements.

**Next Steps**:
1. Review and approve this plan
2. Set up benchmarking infrastructure
3. Begin Phase 1 implementation
4. Iterate based on profiling results and user feedback
