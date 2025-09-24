# Documentation Style Guide

## Overview

This document outlines the documentation standards for the spectral_connectivity package. We follow NumPy-style docstrings with precise array shape and dtype specifications.

## NumPy Docstring Format

All public functions and classes should use NumPy-style docstrings with the following sections:

### Standard Sections

1. **Summary**: Brief one-line description
2. **Parameters**: Input parameters with shapes and dtypes
3. **Returns**: Output values with shapes and dtypes
4. **Raises**: Exceptions that may be raised
5. **Notes**: Additional technical details
6. **References**: Citations and links
7. **Examples**: Executable code examples

### Array Documentation

For all array parameters and returns, specify:
- **Shape**: Use descriptive dimension names (e.g., `(n_times, n_channels)`)
- **Dtype**: Specify NumPy dtypes (e.g., `float64`, `complex128`)

#### Examples

```python
def coherence_magnitude(data: NDArray[np.complex128]) -> NDArray[np.float64]:
    """
    Calculate coherence magnitude between signals.

    Parameters
    ----------
    data : NDArray[complex128], shape (n_times, n_channels)
        Time series data for each channel.

    Returns
    -------
    coherence : NDArray[float64], shape (n_freqs, n_channels, n_channels)
        Magnitude of coherence between channel pairs.
    """
```

## Type Annotations

### Preferred Types

- Use `numpy.typing.NDArray` for array types
- Specify precise dtypes: `NDArray[np.float64]`, `NDArray[np.complex128]`
- For mixed precision, use `NDArray[np.floating]` or `NDArray[np.complexfloating]`
- Use standard Python types for scalars: `int`, `float`, `str`, `bool`

### GPU Compatibility

- Public APIs should use NumPy types in signatures
- Document CuPy compatibility in module docstrings
- Internal functions may use `xp` array module pattern

## Examples Requirements

- Include at least one executable example per public function
- Use realistic array shapes and data
- Show complete workflows where applicable
- Verify examples work with `doctest` when possible

## Documentation Checks

Run the following checks before committing:

```bash
# Style validation
pydocstyle --convention=numpy spectral_connectivity

# NumPy docstring validation (spot check)
numpydoc validate spectral_connectivity.connectivity.Connectivity

# Type checking
mypy spectral_connectivity/{connectivity,transforms,simulate,wrapper}.py
```

## Common Patterns

### Shape Specifications

```python
# Good - descriptive dimension names
shape (n_times, n_trials, n_channels)
shape (n_freqs, n_channels, n_channels)

# Less ideal - generic dimensions
shape (T, N, K)
```

### Dtype Specifications

```python
# Preferred - specific dtypes
NDArray[np.complex128]
NDArray[np.float64]

# Acceptable for flexibility
NDArray[np.floating]
NDArray[np.complexfloating]
```

### Optional Parameters

```python
# Document defaults and behavior
sampling_rate : float, default=1000.0
    Sampling rate in Hz. Used for frequency axis scaling.
```

This style guide ensures consistent, precise documentation that helps users understand array requirements and expected outputs.