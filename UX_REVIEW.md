# Spectral Connectivity Package - UX Review Report

**Review Date:** October 17, 2025
**Reviewer:** Claude (UX Specialist for Scientific Software)
**Scope:** User-facing API, error messages, documentation, and workflows

---

## Executive Summary

The `spectral_connectivity` package demonstrates strong foundational UX with comprehensive docstrings, clear API structure, and thoughtful design decisions. However, there are **critical validation gaps** and **confusion points** that will frustrate users, particularly neuroscientists learning computational methods.

### Key Findings

- **Critical Issues:** 3 blocking issues (input validation, unclear array dimension handling, silent failures)
- **High Priority:** 8 important improvements needed (parameter naming, error messages, workflow friction)
- **Medium Priority:** 6 enhancements (examples, defaults, progress feedback)
- **Overall Rating:** **NEEDS_POLISH** - Core functionality good, but requires refinement before being truly user-ready

**Primary concerns:**

1. Missing input validation allows invalid shapes to propagate and cause cryptic errors downstream
2. Array dimension ordering is confusing and under-documented (time/signals vs time/trials/signals)
3. Error messages lack context about actual vs. expected values
4. Parameter names mix scientific terminology with implementation details inconsistently

---

## Critical UX Issues

### 1. Missing Input Shape Validation (CRITICAL)

**File:** `/Users/edeno/Documents/GitHub/spectral_connectivity/spectral_connectivity/transforms.py`
**Lines:** 125-140 (Multitaper.**init**)

**Problem:**
The `Multitaper` class accepts any array shape without validation. Users can pass 1D arrays, 4D arrays, or incorrectly ordered dimensions, and the code won't fail until deep in the FFT computation with cryptic NumPy errors.

**Example failure scenario:**

```python
import numpy as np
from spectral_connectivity import Multitaper

# User accidentally transposes data: (n_signals, n_time) instead of (n_time, n_signals)
wrong_shape = np.random.randn(5, 1000)  # 5 signals, 1000 timepoints
mt = Multitaper(wrong_shape, sampling_frequency=1000)
# No error! But now thinks there are 1000 signals with 5 timepoints each
# FFT will fail or produce nonsensical results
```

**Why this matters:**
Neuroscientists often work with recordings from multiple electrodes and can easily transpose dimensions. Without clear validation, they'll spend hours debugging when the actual problem is input shape.

**Recommended fix:**

```python
def __init__(
    self,
    time_series: NDArray[np.floating],
    sampling_frequency: float = 1000,
    # ... other params ...
) -> None:
    # Validate input shape
    time_series = np.asarray(time_series)
    ndim = time_series.ndim

    if ndim < 1 or ndim > 3:
        raise ValueError(
            f"time_series must be 1D, 2D, or 3D array. "
            f"Got {ndim}D array with shape {time_series.shape}.\n\n"
            f"Expected shapes:\n"
            f"  - 1D: (n_time_samples,) for single signal\n"
            f"  - 2D: (n_time_samples, n_signals) for multiple signals\n"
            f"  - 3D: (n_time_samples, n_trials, n_signals) for trial-averaged data\n\n"
            f"If your data has shape {time_series.shape}, you may need to transpose or reshape it."
        )

    # Warn if first dimension is suspiciously small (likely transposed)
    if ndim >= 2 and time_series.shape[0] < time_series.shape[1]:
        logger.warning(
            f"time_series has shape {time_series.shape} where the first dimension "
            f"({time_series.shape[0]} samples) is smaller than the second dimension "
            f"({time_series.shape[1]}). This is unusual - the first dimension should be "
            f"time samples. Did you mean to transpose your array?"
        )

    self.time_series = xp.asarray(time_series)
    # ... rest of initialization ...
```

**Impact:** Critical blocker - prevents silent failures and hours of debugging

---

### 2. Confusing Array Dimension Documentation (CRITICAL)

**Files:**

- `/Users/edeno/Documents/GitHub/spectral_connectivity/spectral_connectivity/transforms.py` (lines 40-45)
- `/Users/edeno/Documents/GitHub/spectral_connectivity/spectral_connectivity/connectivity.py` (lines 147-150)

**Problem:**
The dimension ordering changes meaning depending on whether you have 2D or 3D input, but this is buried in prose. Users must read carefully to understand:

- 2D: `(n_time_samples, n_signals)`
- 3D: `(n_time_samples, n_trials, n_signals)`

The middle dimension **completely changes meaning** (signals vs trials), but this isn't visually emphasized.

**Example confusion:**

```python
# User has 100 timepoints, 5 trials, 3 channels
data_3d = np.random.randn(100, 5, 3)
mt = Multitaper(data_3d, sampling_frequency=500)

# User wants just 2 channels, single trial - might incorrectly do:
data_2d = np.random.randn(100, 2)  # Thinks this is 2 channels
# But package interprets this as 2 SIGNALS, not 2 trials
```

**Why this matters:**
This is a fundamental API decision that affects every use. Misunderstanding it means all downstream analyses are wrong, but code runs without error.

**Recommended fixes:**

1. **Add visual emphasis to docstring:**

```python
Parameters
----------
time_series : NDArray[floating]
    Input time series data. **IMPORTANT: Dimension order matters!**

    Supported shapes:

    - **1D: (n_time_samples,)**
      Single signal over time

    - **2D: (n_time_samples, n_signals)**
      Multiple signals (e.g., electrode channels), single trial

    - **3D: (n_time_samples, n_trials, n_signals)**
      Multiple signals across multiple trials/epochs

    ⚠️  **Common mistake**: The middle dimension in 3D data is TRIALS, not signals.
    Signals are always the last dimension.

    Examples:
      - Single 64-channel recording (5 sec, 1000 Hz): shape (5000, 64)
      - 64-channel recording with 10 trials: shape (5000, 10, 64)
```

2. **Add validation helper:**

```python
@staticmethod
def validate_time_series_shape(time_series: NDArray, verbose: bool = True) -> dict:
    """
    Check if time_series has a valid shape and print interpretation.

    Parameters
    ----------
    time_series : ndarray
        Array to validate
    verbose : bool, default=True
        If True, print interpretation of dimensions

    Returns
    -------
    dict with keys: 'n_time_samples', 'n_trials', 'n_signals', 'valid'

    Examples
    --------
    >>> data = np.random.randn(1000, 5, 8)
    >>> Multitaper.validate_time_series_shape(data)
    ✓ Valid 3D array: (1000, 5, 8)
      - Time samples: 1000
      - Trials: 5
      - Signals/channels: 8
    """
    # Implementation here
```

**Impact:** Critical blocker - fundamental API misunderstanding leads to incorrect analyses

---

### 3. Silent Failure with Invalid `expectation_type` Order (HIGH)

**File:** `/Users/edeno/Documents/GitHub/spectral_connectivity/spectral_connectivity/connectivity.py`
**Lines:** 226-232

**Problem:**
While the code validates that `expectation_type` contains valid keywords, it doesn't validate their **order**. Users can pass `"trials_time"` instead of `"time_trials"`, and it silently fails or produces wrong results.

**Current validation:**

```python
if expectation_type not in EXPECTATION:
    allowed_values = ", ".join(f"'{k}'" for k in sorted(EXPECTATION.keys()))
    raise ValueError(
        f"Invalid expectation_type '{expectation_type}'. "
        f"Allowed values are: {allowed_values}"
    )
```

**Example failure:**

```python
# User wants to average over trials and time
conn = Connectivity(fourier_coeffs, expectation_type="trials_time")
# KeyError! But message shows 'trials_time' isn't in the list
# User doesn't understand WHY - order matters but isn't explained
```

**Recommended fix:**

```python
if expectation_type not in EXPECTATION:
    allowed_values = ", ".join(f"'{k}'" for k in sorted(EXPECTATION.keys()))

    # Check if user used wrong order
    tokens = set(expectation_type.split('_'))
    valid_tokens = {'time', 'trials', 'tapers'}
    if tokens.issubset(valid_tokens):
        correct_order = '_'.join(sorted(tokens, key=lambda x: ['time', 'trials', 'tapers'].index(x)))
        raise ValueError(
            f"Invalid expectation_type '{expectation_type}'. "
            f"The dimensions must be in order: time, trials, tapers.\n\n"
            f"Did you mean '{correct_order}'?\n\n"
            f"All allowed values: {allowed_values}"
        )
    else:
        raise ValueError(
            f"Invalid expectation_type '{expectation_type}'. "
            f"Allowed values are: {allowed_values}\n\n"
            f"You can average over any combination of: 'time', 'trials', 'tapers'\n"
            f"Examples: 'trials_tapers', 'time_trials', 'time_trials_tapers'"
        )
```

**Impact:** High priority - causes errors with unclear recovery path

---

## Confusion Points

### 4. Parameter Naming Inconsistency (HIGH)

**Files:** Multiple

**Problem:**
Parameter names mix levels of abstraction inconsistently:

- Scientific terms: `time_halfbandwidth_product`, `sampling_frequency`
- Implementation details: `n_fft_samples`, `n_time_samples_per_window`
- Hybrid: `time_window_duration` (scientific) vs `n_time_samples_per_step` (implementation)

**Examples of confusion:**

| Current Name | Clearer Alternative | Reasoning |
|-------------|---------------------|-----------|
| `n_fft_samples` | `fft_length` or `n_fft_points` | "samples" ambiguous with time samples |
| `is_low_bias` | `exclude_high_bias_tapers` | "is" prefix suggests boolean state, but it's a filter |
| `n_time_samples_per_window` | `window_size_samples` | Shorter, parallel to `window_duration` |
| `n_time_samples_per_step` | `step_size_samples` | Parallel naming |

**Why this matters:**
Neuroscientists think in terms of experimental parameters (duration, frequency) but must toggle between that and implementation details. Inconsistent naming increases cognitive load.

**Recommendation:**
Standardize on user-facing scientific parameters as primary, with implementation details as secondary:

```python
def __init__(
    self,
    time_series: NDArray[np.floating],
    sampling_frequency: float = 1000,
    time_bandwidth_product: float = 3,  # Remove "half" - confusing
    detrend: str | None = "constant",  # Simpler than "detrend_type"
    window_duration: float | None = None,  # Shorter
    window_step: float | None = None,  # Shorter
    n_tapers: int | None = None,  # OK as is
    tapers: NDArray[np.floating] | None = None,  # OK as is
    start_time: float = 0,  # OK as is
    fft_length: int | None = None,  # Clearer than "n_fft_samples"
    # Advanced users only:
    window_size_samples: int | None = None,
    step_size_samples: int | None = None,
    exclude_high_bias_tapers: bool = True,  # Clearer intent
) -> None:
```

**Impact:** High - reduces learning curve and prevents parameter confusion

---

### 5. Method Return Shape Ambiguity (HIGH)

**File:** `/Users/edeno/Documents/GitHub/spectral_connectivity/spectral_connectivity/connectivity.py`

**Problem:**
Methods return different shapes depending on:

1. Whether frequencies are positive-only or two-sided
2. What dimensions were averaged over (controlled by `expectation_type`)
3. Whether it's a pairwise measure (returns matrix) or signal-wise (returns vector)

**This is documented per-method, but there's no systematic pattern.**

**Example confusion:**

```python
conn = Connectivity.from_multitaper(mt, expectation_type="trials_tapers")
power = conn.power()  # Returns (n_time, n_frequencies, n_signals)
coherence = conn.coherence_magnitude()  # Returns (n_time, n_frequencies, n_signals, n_signals)

# User switches expectation_type:
conn2 = Connectivity.from_multitaper(mt, expectation_type="time_trials_tapers")
power2 = conn2.power()  # Returns (n_frequencies, n_signals) - time dimension GONE
coherence2 = conn2.coherence_magnitude()  # Returns (n_frequencies, n_signals, n_signals)

# Shape changed! Downstream code may break
```

**Why this matters:**
Scientists write analysis pipelines that assume consistent output shapes. Changing `expectation_type` (a common operation when exploring data) breaks downstream code in non-obvious ways.

**Recommended fixes:**

1. **Add shape documentation to every method:**

```python
def coherence_magnitude(self) -> NDArray[np.floating]:
    """
    Return the magnitude squared of the complex coherency.

    Returns
    -------
    magnitude : NDArray[floating]
        Magnitude-squared coherence values.

        **Shape depends on expectation_type:**
        - "trials_tapers": (n_time, n_frequencies, n_signals, n_signals)
        - "time_trials_tapers": (n_frequencies, n_signals, n_signals)
        - "trials": (n_time, n_tapers, n_frequencies, n_signals, n_signals)

        Frequencies dimension always contains positive frequencies only.
        Diagonal elements (self-coherence) are set to NaN.

    Notes
    -----
    **Range**: [0, 1]. Implementation may produce tiny numerical excursions
    beyond bounds due to floating-point precision.
    """
```

2. **Add helper method to inspect output shapes:**

```python
def output_shape(self, method_name: str) -> tuple:
    """
    Get expected output shape for a connectivity method.

    Parameters
    ----------
    method_name : str
        Name of connectivity method (e.g., 'coherence_magnitude')

    Returns
    -------
    shape : tuple
        Expected shape of output array

    Examples
    --------
    >>> conn = Connectivity.from_multitaper(mt, expectation_type="trials_tapers")
    >>> conn.output_shape('power')
    (1, 500, 64)  # (n_time, n_frequencies, n_signals)
    >>> conn.output_shape('coherence_magnitude')
    (1, 500, 64, 64)  # (n_time, n_frequencies, n_signals, n_signals)
    """
```

**Impact:** High - prevents pipeline breakage and frustration

---

### 6. GPU Initialization Requirement Unclear (MEDIUM)

**Files:**

- `/Users/edeno/Documents/GitHub/spectral_connectivity/spectral_connectivity/transforms.py` (lines 13-29)
- `/Users/edeno/Documents/GitHub/spectral_connectivity/spectral_connectivity/connectivity.py` (lines 27-38)

**Problem:**
GPU support requires setting environment variable **before import**, but error message appears on import, not when users try to use GPU functionality. Users might miss the logging message.

**Current behavior:**

```python
# User terminal session:
$ python
>>> import spectral_connectivity  # Sees log message, but ignores it
>>> from spectral_connectivity import Multitaper
>>> # ... later wants GPU ...
>>> import os
>>> os.environ["SPECTRAL_CONNECTIVITY_ENABLE_GPU"] = "true"
>>> # Nothing happens! Already imported without GPU
>>> # User confused why GPU isn't working
```

**Why this matters:**
Environment variable setup before import is a Python-specific pattern that non-programmers find confusing. The workflow is brittle.

**Recommended fixes:**

1. **Add GPU status check function:**

```python
def gpu_status() -> dict:
    """
    Check GPU availability and configuration.

    Returns
    -------
    dict with keys:
        - 'available': bool, whether CuPy is installed
        - 'enabled': bool, whether GPU is currently enabled
        - 'device': str, GPU device info if available
        - 'message': str, human-readable status

    Examples
    --------
    >>> from spectral_connectivity import gpu_status
    >>> status = gpu_status()
    >>> print(status['message'])
    'GPU available but NOT enabled. To enable, restart Python with:
     import os
     os.environ["SPECTRAL_CONNECTIVITY_ENABLE_GPU"] = "true"
     import spectral_connectivity'
    """
```

2. **Improve error message with recovery steps:**

```python
if os.environ.get("SPECTRAL_CONNECTIVITY_ENABLE_GPU") == "true":
    try:
        import cupy as xp
        logger.info("✓ GPU acceleration enabled via CuPy")
    except ImportError as exc:
        raise RuntimeError(
            "GPU acceleration requested but CuPy is not installed.\n\n"
            "To fix this:\n"
            "  1. Install CuPy: conda install cupy  (or pip install cupy)\n"
            "  2. Verify your CUDA installation: conda list cudatoolkit\n"
            "  3. Restart Python and re-import spectral_connectivity\n\n"
            "Note: GPU acceleration requires NVIDIA GPU with CUDA support.\n"
            "Mac systems are not supported."
        ) from exc
else:
    logger.info("Using CPU (NumPy). To enable GPU: set SPECTRAL_CONNECTIVITY_ENABLE_GPU='true' before import")
```

**Impact:** Medium - improves GPU setup experience for users with hardware

---

### 7. Unclear Frequency Resolution vs Time-Bandwidth Product (MEDIUM)

**File:** `/Users/edeno/Documents/GitHub/spectral_connectivity/spectral_connectivity/transforms.py` (lines 48-51)

**Problem:**
The relationship between `time_halfbandwidth_product`, `time_window_duration`, and resulting `frequency_resolution` is documented but not intuitive. Users must do mental math or trial-and-error.

**Current docstring:**

```
time_halfbandwidth_product : float, default=3
    Time-bandwidth product controlling frequency resolution and number of
    tapers. Larger values give better frequency resolution but more spectral
    smoothing. Typical values are 2-4.
```

**Why this matters:**
Frequency resolution is what users care about ("Can I distinguish 10 Hz from 12 Hz?"), but they must set an abstract parameter.

**Example workflow:**

```python
# User wants 1 Hz frequency resolution on 2-second windows
# Must calculate: time_halfbandwidth_product = resolution * duration / 2
#                                             = 1 * 2 / 2 = 1
# But documentation doesn't give this formula!
```

**Recommended fix:**

1. **Add formula to docstring:**

```python
time_halfbandwidth_product : float, default=3
    Controls frequency resolution via the relationship:

        frequency_resolution = 2 * time_halfbandwidth_product / window_duration

    For example:
      - window_duration=2 sec, time_halfbandwidth_product=3 → ~3 Hz resolution
      - window_duration=1 sec, time_halfbandwidth_product=2 → ~4 Hz resolution

    **To achieve a specific frequency resolution:**
        time_halfbandwidth_product = (desired_resolution * window_duration) / 2

    Larger values give better frequency resolution but more spectral smoothing
    and longer computation time. Typical values are 2-4.

    Also determines number of tapers: n_tapers ≈ 2 * time_halfbandwidth_product - 1
```

2. **Add helper classmethod:**

```python
@classmethod
def from_target_resolution(
    cls,
    time_series: NDArray[np.floating],
    sampling_frequency: float,
    target_frequency_resolution: float,
    time_window_duration: float,
    **kwargs
) -> "Multitaper":
    """
    Create Multitaper with specific frequency resolution.

    Automatically calculates time_halfbandwidth_product to achieve
    the desired frequency resolution.

    Parameters
    ----------
    time_series : NDArray
        Input time series
    sampling_frequency : float
        Sampling rate in Hz
    target_frequency_resolution : float
        Desired frequency resolution in Hz
    time_window_duration : float
        Window duration in seconds
    **kwargs
        Other Multitaper parameters

    Returns
    -------
    Multitaper instance

    Examples
    --------
    >>> # Want 1 Hz resolution on 2-second windows
    >>> mt = Multitaper.from_target_resolution(
    ...     data, sampling_frequency=500,
    ...     target_frequency_resolution=1.0,
    ...     time_window_duration=2.0
    ... )
    >>> mt.frequency_resolution
    1.0
    """
    time_halfbandwidth_product = (target_frequency_resolution * time_window_duration) / 2
    return cls(
        time_series=time_series,
        sampling_frequency=sampling_frequency,
        time_window_duration=time_window_duration,
        time_halfbandwidth_product=time_halfbandwidth_product,
        **kwargs
    )
```

**Impact:** Medium - reduces trial-and-error for parameter selection

---

## Suggested Improvements

### 8. Add Progress Indication for Long Operations (MEDIUM)

**Files:** Multiple connectivity methods

**Problem:**
Many connectivity computations are slow (especially Granger causality with many signal pairs), but there's no progress indication. Users don't know if code is running or stuck.

**Example:**

```python
# Computing Granger causality on 64 channels = 2016 pairs
conn = Connectivity.from_multitaper(mt)
result = conn.pairwise_spectral_granger_prediction()
# ... silence for 5 minutes ...
# Is it working? Should I kill it?
```

**Recommended fix:**

```python
def pairwise_spectral_granger_prediction(
    self,
    progress: bool = True
) -> NDArray[np.floating]:
    """
    Return amount of power at a node explained by other nodes.

    Parameters
    ----------
    progress : bool, default=True
        Show progress bar for pair-wise computations.
        Set to False for batch scripts.

    ...
    """
    csm = self._expectation_cross_spectral_matrix()
    n_signals = csm.shape[-1]
    pairs = list(combinations(range(n_signals), 2))
    n_pairs = len(pairs)

    if progress and n_pairs > 10:  # Only show for non-trivial computations
        try:
            from tqdm import tqdm
            pairs_iter = tqdm(pairs, desc="Computing Granger prediction", unit="pairs")
        except ImportError:
            logger.info(f"Computing Granger prediction for {n_pairs} pairs...")
            pairs_iter = pairs
    else:
        pairs_iter = pairs

    total_power = self._power
    return _estimate_spectral_granger_prediction(total_power, csm, pairs_iter)
```

**Impact:** Medium - greatly improves experience for large datasets

---

### 9. Inconsistent NaN Handling in Output (MEDIUM)

**Problem:**
Some methods set invalid values to NaN (e.g., diagonal of coherence), others don't. This isn't consistently documented.

**Examples:**

```python
coherence = conn.coherence_magnitude()
# Diagonal is NaN (documented in line 519)

granger = conn.pairwise_spectral_granger_prediction()
# Diagonal is NaN (documented in line 2173)
# But what about failed decompositions? Also NaN (line 2169) but not documented
```

**Recommended fix:**
Add consistent documentation pattern:

```python
def coherence_magnitude(self) -> NDArray[np.floating]:
    """
    ...

    Returns
    -------
    magnitude : array
        Magnitude-squared coherence values.

        **Special values:**
        - Diagonal elements (i, i): NaN (self-coherence undefined)
        - Zero power cases: NaN (division by zero)

    Notes
    -----
    **Range**: [0, 1]. Implementation may produce tiny numerical excursions
    beyond bounds due to floating-point precision.
    """
```

**Impact:** Medium - prevents confusion about NaN values

---

### 10. Confusing FFT Method Name (LOW)

**File:** `/Users/edeno/Documents/GitHub/spectral_connectivity/spectral_connectivity/transforms.py` (line 394)

**Problem:**
`Multitaper.fft()` is a method, but users might think it's a cached property since it's called without arguments.

**Current:**

```python
fourier_coefficients = multitaper.fft()
```

**Why this matters:**
Calling `.fft()` multiple times recomputes the FFT each time (expensive!). Users might not realize this and call it repeatedly in loops.

**Recommended fixes:**

1. **Cache the result:**

```python
@property
def fourier_coefficients(self) -> NDArray[np.complexfloating]:
    """
    Compute (or retrieve cached) Fourier coefficients.

    Returns
    -------
    fourier_coefficients : array
        Shape (n_time_windows, n_trials, n_tapers, n_frequencies, n_signals).
        Complex-valued Fourier coefficients.

    Notes
    -----
    Result is cached after first computation.
    """
    if not hasattr(self, '_cached_fft'):
        self._cached_fft = self._compute_fft()
    return self._cached_fft

def _compute_fft(self) -> NDArray[np.complexfloating]:
    """Actual FFT computation."""
    time_series = _add_axes(self.time_series)
    # ... existing fft() code ...
```

2. **Keep fft() but add warning:**

```python
def fft(self) -> NDArray[np.complexfloating]:
    """
    Compute the fast Fourier transform using the multitaper method.

    **Performance note**: This method recomputes the FFT each time it's called.
    If you need the result multiple times, store it in a variable:

        coeffs = multitaper.fft()  # Compute once
        # Use coeffs multiple times

    Returns
    -------
    fourier_coefficients : array
        Shape (n_time_windows, n_trials, n_tapers, n_fft_samples, n_signals).
    """
```

**Impact:** Low - performance improvement and clarity

---

### 11. Missing Input Examples for Common Formats (MEDIUM)

**Problem:**
Neuroscientists use many data formats (MNE, Neo, NWB, etc.), but documentation only shows NumPy arrays. No guidance on converting from these formats.

**Recommended fix:**
Add examples section to main documentation:

```python
"""
Examples
--------
**From NumPy array:**
>>> import numpy as np
>>> data = np.random.randn(1000, 64)  # 1000 samples, 64 channels
>>> mt = Multitaper(data, sampling_frequency=500)

**From MNE Epochs:**
>>> import mne
>>> epochs = mne.read_epochs('data-epo.fif')
>>> # Extract data: (n_epochs, n_channels, n_times) -> transpose to (n_times, n_epochs, n_channels)
>>> data = epochs.get_data().transpose(2, 0, 1)
>>> mt = Multitaper(data, sampling_frequency=epochs.info['sfreq'])

**From pandas DataFrame:**
>>> import pandas as pd
>>> df = pd.read_csv('lfp_data.csv')  # Columns: time, ch1, ch2, ch3, ...
>>> data = df.iloc[:, 1:].values  # Skip time column
>>> mt = Multitaper(data, sampling_frequency=1000)

**From Neo AnalogSignal:**
>>> import neo
>>> import quantities as pq
>>> signal = neo.AnalogSignal(data, units='mV', sampling_rate=1000*pq.Hz)
>>> data = signal.magnitude.T  # Neo uses (time, channels)
>>> mt = Multitaper(data, sampling_frequency=signal.sampling_rate.magnitude)
"""
```

**Impact:** Medium - reduces friction for domain-specific users

---

### 12. No Sanity Checks on Computed Results (MEDIUM)

**Problem:**
Methods compute values that should fall in known ranges (e.g., coherence in [0,1]), but don't warn users if numerical precision issues cause violations.

**Example:**

```python
coherence = conn.coherence_magnitude()
# Due to floating point precision, might get 1.0000000002
# Or negative values near zero
```

**Recommended fix:**

```python
@_asnumpy
def coherence_magnitude(self) -> NDArray[np.floating]:
    """..."""
    result = _squared_magnitude(self.coherency())

    # Sanity check for numerical precision issues
    if xp.any(result > 1.0 + 1e-6) or xp.any(result < -1e-6):
        n_violations = xp.sum((result > 1.0 + 1e-6) | (result < -1e-6))
        logger.warning(
            f"Coherence magnitude should be in [0, 1], but {n_violations} values "
            f"are outside this range (max: {xp.max(result):.6f}, min: {xp.min(result):.6f}). "
            f"This is likely due to numerical precision issues. "
            f"Clipping to valid range."
        )
        result = xp.clip(result, 0.0, 1.0)

    return result
```

**Impact:** Medium - catches numerical issues early

---

## Good UX Patterns Found

The package demonstrates several excellent UX practices:

### 1. Comprehensive Docstrings (EXCELLENT)

- Every public method has detailed docstrings with parameter descriptions
- Includes scientific references for each connectivity measure
- **Range** notes specify expected output values (e.g., "Range: [0, 1]")
- Examples provided for key classes

**Example** (transforms.py, lines 32-123):

```python
"""
Multitaper spectral analysis for robust power spectral density estimation.

Transforms time-domain signals to frequency domain using multiple orthogonal
tapering windows (Slepian sequences). This approach reduces spectral leakage
and provides better spectral estimates than single-taper methods.

Parameters
----------
[detailed parameter documentation]

Examples
--------
>>> import numpy as np
>>> # Generate test signal: 50Hz + noise
>>> fs = 1000  # 1 kHz sampling
>>> t = np.arange(0, 1, 1/fs)
>>> signal = np.sin(2*np.pi*50*t) + 0.1*np.random.randn(len(t))
```

**Why this works:** Matches scientific software standards; users can learn from docstrings without reading tutorials.

---

### 2. Multiple API Levels (EXCELLENT)

The package provides three usage patterns for different expertise levels:

1. **Low-level:** Direct `Multitaper` → `Connectivity` workflow (expert users)
2. **Convenience:** `Connectivity.from_multitaper()` classmethod (intermediate)
3. **High-level:** `multitaper_connectivity()` wrapper with xarray output (beginners)

**Example** (wrapper.py, lines 137-224):

```python
def multitaper_connectivity(
    time_series: NDArray[np.floating],
    sampling_frequency: float,
    time_window_duration: float | None = None,
    method: str | list[str] | None = None,
    # ... simplified interface
) -> xr.DataArray | xr.Dataset:
    """
    Compute connectivity measures with multitaper spectral estimation.

    This is the main high-level function for connectivity analysis.
    """
```

**Why this works:** Beginner users get xarray outputs with labeled dimensions (easy plotting), experts get full control with NumPy arrays.

---

### 3. Sensible Defaults (GOOD)

Most parameters have reasonable defaults:

- `sampling_frequency=1000` (common in neuroscience)
- `time_halfbandwidth_product=3` (typical value)
- `detrend_type="constant"` (remove DC offset by default)
- `expectation_type="trials_tapers"` (most common averaging)

**Why this works:** Users can start with minimal parameters and refine later.

---

### 4. Informative **repr** (GOOD)

**Example** (transforms.py, lines 155-174):

```python
>>> mt = Multitaper(data, sampling_frequency=1000)
>>> mt
Multitaper(sampling_frequency=1000, time_halfbandwidth_product=3,
           time_window_duration=60.001, time_window_step=60.001,
           detrend_type='constant', start_time=0, n_tapers=5)
```

**Why this works:** Shows computed values (like `n_tapers`) alongside user-set parameters, helping with debugging.

---

### 5. Explicit GPU Opt-in (GOOD)

GPU acceleration requires explicit environment variable, preventing accidental GPU usage that might fail on some systems.

```python
if os.environ.get("SPECTRAL_CONNECTIVITY_ENABLE_GPU") == "true":
    import cupy as xp
else:
    import numpy as xp
```

**Why this works:** Conservative default (CPU) ensures compatibility; expert users can opt-in to performance.

---

### 6. Comprehensive Tutorials (EXCELLENT)

The `examples/Intro_tutorial.ipynb` provides step-by-step walkthrough with:

- Simulated data generation
- Explanations of each parameter
- Multiple examples building on each other
- Visualization of results

**Why this works:** Lower barrier to entry for scientists unfamiliar with spectral analysis.

---

### 7. Type Hints Throughout (GOOD)

```python
def __init__(
    self,
    time_series: NDArray[np.floating],
    sampling_frequency: float = 1000,
    # ...
) -> None:
```

**Why this works:** Modern Python practice; enables IDE autocomplete and static type checking.

---

## Overall Assessment

**Rating:** **NEEDS_POLISH**

**Rationale:**

The spectral_connectivity package has **strong foundational UX** with excellent documentation, thoughtful API design, and clear scientific grounding. However, **critical validation gaps** and **dimension handling confusion** will frustrate users, particularly neuroscientists learning computational methods.

### Can ship after addressing

1. **Critical:** Input shape validation (Issue #1)
2. **Critical:** Array dimension documentation improvements (Issue #2)
3. **High:** expectation_type order validation (Issue #3)
4. **High:** Output shape documentation (Issue #5)

### Strengths to maintain

- Comprehensive docstrings with scientific references
- Multiple API levels for different expertise
- Good default parameters
- Clear separation of spectral transform vs connectivity analysis

### Recommended Priority

1. **Week 1:** Fix critical validation issues (#1, #2, #3) - these are blockers
2. **Week 2:** Improve parameter naming consistency (#4) and output shape docs (#5)
3. **Week 3:** Add progress indication (#8) and better examples (#11)
4. **Week 4:** Polish GPU experience (#6) and edge case handling (#9, #12)

### User Readiness

- **Expert users (computational neuroscientists):** Ready with current docs, will work around issues
- **Intermediate users (experimentalists with Python experience):** Needs polish - dimension confusion will cause errors
- **Novice users (learning Python for analysis):** Not ready - needs clearer error messages and validation

The package is **close to user-ready** but needs one focused sprint on validation and documentation before broad release. Core functionality is solid; it's the edges that need smoothing.

---

## Appendix: Error Message Checklist

For every error message in the package, verify it answers:

1. **WHAT went wrong:** Clear statement of the problem
2. **WHY it happened:** Brief explanation of the cause
3. **HOW to fix it:** Specific, actionable recovery steps

### Current Error Messages Audit

✅ **GOOD** - GPU import error (transforms.py:20-24):

```python
raise RuntimeError(
    "GPU support was explicitly requested via SPECTRAL_CONNECTIVITY_ENABLE_GPU='true', "
    "but CuPy is not installed. Please install CuPy with: "
    "'pip install cupy' or 'conda install cupy'"
)
```

- WHAT: CuPy not installed
- WHY: GPU explicitly requested
- HOW: Install CuPy with specific commands

❌ **NEEDS WORK** - Detrend type error (transforms.py:968):

```python
raise ValueError("Trend type must be 'linear' or 'constant'.")
```

- WHAT: ✓ Invalid trend type
- WHY: ✗ Doesn't say what was provided
- HOW: ✗ Doesn't show valid options clearly

**Improved version:**

```python
raise ValueError(
    f"Invalid detrend type: '{type}'.\n"
    f"Must be 'linear', 'constant', or None.\n"
    f"  - 'constant': removes DC component (mean)\n"
    f"  - 'linear': removes linear trend\n"
    f"  - None: no detrending"
)
```

❌ **NEEDS WORK** - Breakpoint error (transforms.py:980-982):

```python
raise ValueError(
    "Breakpoints must be less than length " "of data along given axis."
)
```

- WHAT: ✓ Invalid breakpoints
- WHY: ✗ No context (what were the breakpoints? what was the length?)
- HOW: ✗ How to fix?

**Improved version:**

```python
raise ValueError(
    f"Invalid breakpoints: {bp_array[bp_array > N]}\n"
    f"All breakpoints must be less than data length ({N}) along axis {axis}.\n"
    f"Current data shape: {data.shape}"
)
```

---

## Testing Recommendations

To validate UX improvements, add tests for:

1. **Input validation:**

```python
def test_multitaper_rejects_wrong_dimensions():
    with pytest.raises(ValueError) as exc_info:
        Multitaper(np.random.randn(5, 10, 15, 20))  # 4D array
    assert "must be 1D, 2D, or 3D" in str(exc_info.value)
    assert "shape" in str(exc_info.value).lower()
```

2. **Transposition warning:**

```python
def test_multitaper_warns_likely_transposed():
    with pytest.warns(UserWarning) as warn_info:
        Multitaper(np.random.randn(5, 1000))  # Likely transposed
    assert "transpose" in str(warn_info[0].message).lower()
```

3. **Error message quality:**

```python
def test_error_messages_follow_what_why_how_pattern():
    """All error messages should answer: what, why, how."""
    # Test each ValueError and check message content
```

---

## Accessibility Notes

For users with visual impairments or using screen readers:

1. ✅ **Good:** Avoid emoji in error messages
2. ✅ **Good:** Use text-based progress indication (compatible with screen readers)
3. ⚠️ **Consider:** Visualization examples in tutorials should include text descriptions

For colorblind users:

1. ⚠️ **Check:** Tutorial examples use `viridis` colormap (good) but should note other colorblind-friendly options
2. ✅ **Good:** No use of color alone to convey information in docs

---

**End of Report**
