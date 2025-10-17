# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- CHANGELOG.md to track version changes following Keep a Changelog format
- Ruff linter configuration for faster, more comprehensive Python linting
- Enhanced package metadata with additional project URLs (Changelog, Source Code, Issue Tracker)
- Modern unified CI/CD workflow (`release.yml`) with automated PyPI publishing
- Support for Python 3.13
- Comprehensive parameter validation to `Multitaper` class:
  - Validates `sampling_frequency > 0` with domain-specific examples (EEG, LFP, fMRI)
  - Validates `time_halfbandwidth_product >= 1` with physical meaning explanation
  - Validates `time_window_duration > 0` (when provided) with frequency resolution formula
  - Validates `time_window_step > 0` (when provided) with overlap guidance
  - Warns when `time_halfbandwidth_product > 10` (unusually large, performance impact)
  - Warns when `time_window_step > time_window_duration` (creates data gaps)
  - Warns when data appears transposed (`n_time < n_signals`)
  - Warns when input contains NaN or Inf values with recovery suggestions
- Input shape validation to `Connectivity` class:
  - Requires 5D `fourier_coefficients` with clear error messages
  - Validates minimum 2 signals for connectivity analysis
  - Warns on NaN/Inf values in Fourier coefficients
- `prepare_time_series()` helper function for safe dimension handling:
  - Converts 1D/2D arrays to required 3D format
  - Explicit `axis` parameter to clarify dimension meaning
  - Prevents ambiguous dimension interpretation
- Enhanced error messages following WHAT/WHY/HOW pattern throughout
- 3D input requirement for `Multitaper` class to eliminate dimension ambiguity
- Intelligent error suggestion for `expectation_type` parameter:
  - Detects wrong word order (e.g., "tapers_trials" instead of "trials_tapers")
  - Suggests correct ordering with helpful explanation
  - Lists all valid options with most common choice highlighted
- Improved `detrend()` function error messages:
  - Clear explanation of linear vs constant detrending
  - Examples with domain-specific terminology (DC offset, best-fit line)
  - Actionable guidance for parameter selection
- Enhanced breakpoint validation in `detrend()`:
  - Shows specific invalid breakpoint values
  - Displays valid range based on actual data dimensions
  - Includes user's original input for easy debugging
- Comprehensive test suite for error message quality (`test_error_messages.py`)
- GPU status utility function `get_compute_backend()`:
  - Returns dict with backend type, GPU availability, device name, and helpful message
  - Shows actual GPU model name (e.g., "NVIDIA Tesla V100-SXM2-16GB") instead of compute capability
  - Detects CuPy availability without side effects (uses `importlib.util.find_spec`)
  - Provides 4 different message variants for different GPU configurations
  - Includes comprehensive NumPy-style docstring with 3 usage examples
  - Example return value documented in docstring
  - Located in new `spectral_connectivity.utils` module
- Enhanced GPU device logging in `transforms.py` and `connectivity.py`:
  - Now shows actual GPU model name in log messages
  - Graceful fallback to compute capability if name unavailable
- Comprehensive GPU documentation in README.md:
  - 130+ line GPU Acceleration section with setup, troubleshooting, and usage guidance
  - 3 setup methods documented (shell, Python script, Jupyter notebook)
  - Verification steps included in all setup examples
  - Simplified CuPy installation instructions (conda recommended first)
  - Troubleshooting guide with 4 common issues and solutions
  - Clear explanation of import timing requirement (why "before importing" matters)
  - Example outputs shown for all code samples
  - Kernel restart guidance for Jupyter notebook users
  - Guidance on when GPU acceleration is beneficial
- Comprehensive test suite for GPU backend (`tests/test_gpu.py`):
  - 13 test methods covering all GPU configuration scenarios
  - Tests for CPU mode (default and explicit)
  - Tests for GPU mode (with and without CuPy available)
  - Validation of return value structure and types
  - Mock-based testing to avoid CuPy dependency
  - All tests pass (11 passed, 1 skipped when CuPy unavailable)

### Changed
- **BREAKING**: Minimum Python version raised from 3.9 to 3.10
- Migrated from flake8 to ruff for linting (100x faster, replaces flake8, isort, pydocstyle)
- Updated dependency pins: numpy>=1.24, scipy>=1.10, xarray>=2023.1, matplotlib>=3.7
- Improved mypy configuration with stricter type checking and per-module overrides
- Updated development documentation (CLAUDE.md, CONTRIBUTING.md) to reflect current tooling
- Expanded test matrix to Python 3.10, 3.11, 3.12, 3.13
- Consolidated CI workflows: removed redundant PR-test.yml and linting.yml in favor of release.yml
- Simplified CI from conda-based to pip-based installation (faster builds)
- Enhanced black configuration to target Python 3.10-3.13
- Updated ReadTheDocs to use Python 3.10

### Fixed
- Outdated release instructions in CONTRIBUTING.md (removed setup.py references)
- Deprecation warning in `minimum_phase_decomposition.py`: Changed `xp.linalg.linalg.LinAlgError` to `xp.linalg.LinAlgError` for compatibility with NumPy 2.0+

## [1.1.2] - 2023-10-17

### Added
- Conda packaging support with conda-recipe directory
- CLAUDE.md with development commands and architecture documentation
- Pinned coverage reporter version in CI workflow to avoid bugs

### Changed
- Updated module docstrings for clarity and context
- Updated README with Contributing and License sections

### Fixed
- Linting issues resolved

## [1.1.1] - 2023-09-15

### Changed
- Switch build system from setuptools to Hatch
- Add py.typed marker for type hint support
- Update and reorganize dependencies in environment.yml

### Fixed
- Resolve n_time_samples_per_window property logic error in transforms module
- Resolve mypy Optional[int] vs int return type errors
- Correct _fix_taper_sign return type annotation

## [1.1.0] - 2023-08-20

### Added
- GPU request guard feature to safely handle CUDA availability
- Complete audit of connectivity metric ranges documentation
- ValueError raised when window size parameters are unset

### Changed
- Updated GitHub Actions to latest versions
- Improved type hints throughout codebase

## [1.0.4] - 2023-03-15

### Fixed
- Bug fixes in connectivity calculations
- Improved numerical stability

## [1.0.3] - 2023-02-10

### Changed
- Performance improvements
- Documentation updates

## [1.0.2] - 2023-01-20

### Fixed
- Minor bug fixes
- Test coverage improvements

## [1.0.1] - 2022-12-15

### Fixed
- Package distribution fixes
- Documentation corrections

## [1.0.0] - 2022-12-01

### Added
- First stable release
- Full implementation of 15+ connectivity measures
- GPU acceleration support via CuPy
- Comprehensive test suite
- Complete documentation on ReadTheDocs

### Changed
- API stabilized for 1.0 release
- Performance optimizations

## [0.2.7] - 2022-06-15

### Added
- Additional connectivity measures
- Improved caching strategy

### Changed
- API improvements and refinements

## [0.2.6] - 2022-03-10

### Added
- Initial GPU support
- More connectivity measures

### Changed
- Refactored core architecture
- Improved documentation

---

[Unreleased]: https://github.com/Eden-Kramer-Lab/spectral_connectivity/compare/v1.1.2...HEAD
[1.1.2]: https://github.com/Eden-Kramer-Lab/spectral_connectivity/compare/v1.1.1...v1.1.2
[1.1.1]: https://github.com/Eden-Kramer-Lab/spectral_connectivity/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/Eden-Kramer-Lab/spectral_connectivity/compare/v1.0.4...v1.1.0
[1.0.4]: https://github.com/Eden-Kramer-Lab/spectral_connectivity/compare/v1.0.3...v1.0.4
[1.0.3]: https://github.com/Eden-Kramer-Lab/spectral_connectivity/compare/v1.0.2...v1.0.3
[1.0.2]: https://github.com/Eden-Kramer-Lab/spectral_connectivity/compare/v1.0.1...v1.0.2
[1.0.1]: https://github.com/Eden-Kramer-Lab/spectral_connectivity/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/Eden-Kramer-Lab/spectral_connectivity/compare/v0.2.7...v1.0.0
[0.2.7]: https://github.com/Eden-Kramer-Lab/spectral_connectivity/compare/v0.2.6...v0.2.7
[0.2.6]: https://github.com/Eden-Kramer-Lab/spectral_connectivity/releases/tag/v0.2.6
