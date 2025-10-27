# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`spectral_connectivity` is a Python package for computing multitaper spectral estimates and frequency-domain brain connectivity measures. It provides tools for functional and directed connectivity analysis of electrophysiological data, with optional GPU acceleration via CuPy.

## Core Architecture

The package follows a modular design with three main components:

1. **Transforms** (`spectral_connectivity/transforms.py`): Implements the `Multitaper` class for computing multitaper Fourier transforms
2. **Connectivity** (`spectral_connectivity/connectivity.py`): The `Connectivity` class computes various connectivity measures from spectral estimates
3. **Wrapper** (`spectral_connectivity/wrapper.py`): Provides `multitaper_connectivity()` function as a high-level interface

### Key Design Patterns

- **GPU/CPU Abstraction**: Uses `xp` namespace (numpy or cupy) controlled by `SPECTRAL_CONNECTIVITY_ENABLE_GPU` environment variable
- **Caching**: Frequently computed quantities like cross-spectral matrices are cached for performance
- **Expectation Framework**: Uses `EXPECTATION` dictionary to handle averaging over different dimensions (time, trials, tapers)

### Main Classes

- `Multitaper`: Handles multitaper spectral estimation
- `Connectivity`: Computes connectivity measures from spectral data
- Both classes support method chaining and lazy evaluation patterns

## Development Commands

### Environment Setup
```bash
# Create conda environment
conda env create -f environment.yml
conda activate spectral_connectivity
pip install -e .
```

### Testing
```bash
# Run all tests with coverage
pytest --cov=spectral_connectivity tests/ --cov-report=lcov:coverage.lcov -v

# Run specific test file
pytest tests/test_connectivity.py -v

# Run single test
pytest tests/test_connectivity.py::TestConnectivity::test_coherence -v
```

### Code Quality
```bash
# Format code
ruff format .

# Check formatting
ruff format --check .

# Lint code
ruff check .

# Fix auto-fixable linting issues
ruff check --fix .

# Type checking
mypy spectral_connectivity/
```

### Building and Release
```bash
# Build package
hatch build

# Clean build artifacts
rm -rf build/ dist/ *.egg-info/
```

## Important Configuration

### GPU Support
Set environment variable `SPECTRAL_CONNECTIVITY_ENABLE_GPU=true` to enable GPU acceleration via CuPy.

### Dependencies
- Core: numpy, scipy, xarray, matplotlib
- Dev tools: pytest, ruff, mypy, numpydoc
- Optional GPU: cupy-cuda12x

## Testing Strategy

- Unit tests in `tests/` directory mirror the source structure
- CI runs tests on Python 3.9+ and Ubuntu
- Coverage reporting via Coveralls
- Notebook integration tests execute tutorial examples
- Tests include both CPU and GPU code paths when available

## File Structure

```
spectral_connectivity/
├── __init__.py              # Main API exports
├── connectivity.py          # Connectivity measures
├── transforms.py           # Multitaper transforms
├── wrapper.py              # High-level interface
├── minimum_phase_decomposition.py
├── statistics.py           # Statistical utilities
└── simulate.py             # Data simulation utilities
```