# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Environment Setup
```bash
# Create conda environment and install dependencies
conda env create -f environment.yml
conda activate spectral_connectivity
pip install -e .
```

### Testing
```bash
# Run all tests (requires nitime to be installed)
pytest

# Run specific test file
pytest tests/test_connectivity.py

# Run tests with coverage
pytest --cov=spectral_connectivity
```

### Code Quality
```bash
# Format code with black
black spectral_connectivity/

# Lint with flake8
flake8 spectral_connectivity/

# Type checking with mypy
mypy spectral_connectivity/

# Check docstring style
pydocstyle spectral_connectivity/
```

### Documentation
```bash
# Build documentation (from docs/ directory)
cd docs/
make html
```

### Build and Release
```bash
# Build package
hatch build

# Upload to PyPI (requires twine and proper credentials)
twine upload dist/*
```

## Code Architecture

### Core Components

The package follows a modular architecture with three main components:

1. **transforms.py**: `Multitaper` class - Handles time-to-frequency domain transformation using multitaper methods
2. **connectivity.py**: `Connectivity` class - Computes various connectivity measures from spectral data
3. **wrapper.py**: High-level convenience functions that combine transforms and connectivity analysis

### Key Classes

- **`Multitaper`**: Primary class for spectral analysis using Slepian tapers
  - Transforms time series to frequency domain
  - Supports windowing, overlapping, and multiple trials
  - Caches computations for efficiency

- **`Connectivity`**: Main connectivity analysis class
  - Takes Multitaper output or raw Fourier coefficients
  - Implements 15+ connectivity measures (coherence, Granger causality, phase measures)
  - Handles both functional and directed connectivity

### GPU Support

The package supports GPU acceleration via CuPy:
- Set `SPECTRAL_CONNECTIVITY_ENABLE_GPU=true` environment variable
- Requires CuPy installation (`pip install cupy` or `conda install cupy`)
- GPU/CPU switching is handled automatically at import time in both `transforms.py` and `connectivity.py`
- Uses `xp` alias throughout codebase for numpy/cupy compatibility

### Data Flow

1. **Input**: Time series data (n_time, n_trials, n_signals)
2. **Transform**: `Multitaper` → Fourier coefficients + metadata
3. **Analysis**: `Connectivity.from_multitaper()` → Various connectivity measures
4. **Output**: Arrays or xarray DataArrays with proper labeling

### Caching Strategy

The package implements intelligent caching:
- Cross-spectral matrices are cached in `Connectivity` class
- Minimum phase decompositions are cached for Granger causality measures
- This allows fast computation of multiple connectivity measures from the same spectral data

### Testing Dependencies

Tests require additional dependencies not included in core package:
- `nitime`: For validating DPSS window implementations
- Install via: `pip install nitime` or use dev dependencies: `pip install -e .[dev]`