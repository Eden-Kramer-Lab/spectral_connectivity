# Notebook Snapshot Testing Plan

## Overview

Create snapshot tests for tutorial notebooks to verify:
1. **Execution:** Notebooks run without errors (slow tests)
2. **Numerical correctness:** Key outputs match expected values (fast tests)

## Scope

**Notebooks to test:**
- `examples/Tutorial_On_Simulated_Examples.ipynb`
- `examples/Tutorial_Using_Paper_Examples.ipynb`

**Excluded:**
- `examples/Intro_tutorial.ipynb` (too basic/introductory)

## Architecture

### Test Types

#### 1. Snapshot Tests (Fast* Always Run)
- Hand-written test functions in `tests/test_notebooks.py`
- Use `syrupy` fixture to snapshot numerical outputs
- Test key scenarios inspired by notebook examples
- Verify arrays match expected values using `np.allclose()` (rtol=1e-7, atol=1e-10)
- Run on every commit/PR

**Performance Note:** The full test suite (27 tests) takes ~2-3 minutes to run due to pytest/syrupy overhead with large NumPy arrays. Individual tests are fast (<1s each), but memory accumulation causes super-linear scaling when running all tests together. This is expected behavior and not a bug in the test code itself.

#### 2. Execution Tests (Slow, Marked)
- Verify notebooks execute without errors
- Use `jupyter nbconvert --execute`
- Marked with `@pytest.mark.slow`
- Run via `pytest -m slow` (optional in CI)
- Can be run only on PRs or releases

### Directory Structure

```
tests/
├── test_notebooks.py              # New: Hand-written snapshot + execution tests
├── __snapshots__/                 # Auto-created by syrupy
│   └── test_notebooks/
│       ├── test_power_spectrum_200hz.ambr
│       ├── test_coherence_phase_offset.ambr
│       └── ...
├── conftest.py                    # Update: Add slow marker
└── ... (existing tests)

examples/
├── Tutorial_On_Simulated_Examples.ipynb
├── Tutorial_On_Simulated_Examples.py     # Jupytext paired
├── Tutorial_Using_Paper_Examples.ipynb
└── Tutorial_Using_Paper_Examples.py      # Jupytext paired
```

## Implementation Steps

### Step 1: Add Dependencies

**File: `pyproject.toml`**

Add to `[project.optional-dependencies]`:
```toml
dev = [
    "syrupy>=4.0.0",
    # ... existing dev dependencies
]
```

### Step 2: Configure pytest Markers

**File: `pyproject.toml`**

Add to `[tool.pytest.ini_options]`:
```toml
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
]
```

### Step 3: Create Snapshot Tests

**File: `tests/test_notebooks.py`**

Create hand-written test functions covering key scenarios:

```python
"""Snapshot tests for tutorial notebooks.

These tests verify that key numerical outputs from the tutorial
notebooks remain stable across code changes. Tests are inspired
by notebook examples but hand-written for clarity and focus.
"""
import numpy as np
import pytest
from spectral_connectivity import Connectivity, Multitaper
from spectral_connectivity.transforms import prepare_time_series


def test_power_spectrum_200hz(snapshot):
    """Power spectrum of 200 Hz signal."""
    np.random.seed(42)
    sampling_frequency = 1500
    time = np.linspace(0, 50, 75001, endpoint=True)
    signal = np.sin(2 * np.pi * time * 200)
    noise = np.random.normal(0, 4, signal.shape)

    multitaper = Multitaper(
        prepare_time_series(signal + noise),
        sampling_frequency=sampling_frequency,
        time_halfbandwidth_product=3,
        start_time=time[0],
    )
    connectivity = Connectivity.from_multitaper(multitaper)

    assert connectivity.power() == snapshot
    assert connectivity.frequencies == snapshot


def test_coherence_magnitude_phase_offset(snapshot):
    """Coherence with fixed phase offset between signals."""
    np.random.seed(42)
    sampling_frequency = 1500
    time_extent = (0, 50)
    n_time_samples = int(((time_extent[1] - time_extent[0]) * sampling_frequency) + 1)
    time = np.linspace(time_extent[0], time_extent[1], num=n_time_samples, endpoint=True)

    # Create 2 signals with pi/2 phase offset
    frequency_of_interest = 200
    n_signals = 2
    signal = np.zeros((n_time_samples, n_signals))
    signal[:, 0] = np.sin(2 * np.pi * time * frequency_of_interest)
    phase_offset = np.pi / 2
    signal[:, 1] = np.sin((2 * np.pi * time * frequency_of_interest) + phase_offset)
    noise = np.random.normal(0, 4, signal.shape)

    multitaper = Multitaper(
        prepare_time_series(signal + noise, axis='signals'),
        sampling_frequency=sampling_frequency,
        time_halfbandwidth_product=5,
        start_time=time[0],
    )
    connectivity = Connectivity.from_multitaper(multitaper)

    assert connectivity.coherence_magnitude() == snapshot
    assert connectivity.frequencies == snapshot


def test_spectrogram_temporal_dynamics(snapshot):
    """Spectrogram showing 50 Hz turning on at t=25s."""
    np.random.seed(42)
    sampling_frequency = 1500
    time_extent = (0, 50)
    frequency_of_interest = [200, 50]
    n_time_samples = int(((time_extent[1] - time_extent[0]) * sampling_frequency) + 1)
    time = np.linspace(time_extent[0], time_extent[1], num=n_time_samples, endpoint=True)

    # Create signal with 200 Hz constant, 50 Hz turns on at t=25s
    signal = np.sin(2 * np.pi * time[:, np.newaxis] * frequency_of_interest)
    signal[: n_time_samples // 2, 1] = 0  # 50 Hz only in second half
    signal = signal.sum(axis=1)
    noise = np.random.normal(0, 4, signal.shape)

    multitaper = Multitaper(
        prepare_time_series(signal + noise),
        sampling_frequency=sampling_frequency,
        time_halfbandwidth_product=3,
        time_window_duration=0.600,
        time_window_step=0.300,
        start_time=time[0],
    )
    connectivity = Connectivity.from_multitaper(multitaper)

    assert connectivity.power() == snapshot
    assert connectivity.frequencies == snapshot
    assert connectivity.time == snapshot


def test_coherogram_phase_change(snapshot):
    """Coherogram showing phase offset changing at t=1.5s."""
    np.random.seed(42)
    sampling_frequency = 1500
    time_extent = (0, 2.400)
    n_trials = 100
    n_signals = 2
    frequency_of_interest = 200
    n_time_samples = int(((time_extent[1] - time_extent[0]) * sampling_frequency) + 1)
    time = np.linspace(time_extent[0], time_extent[1], num=n_time_samples, endpoint=True)

    # Random phase before t=1.5s, fixed phase after
    signal = np.zeros((n_time_samples, n_trials, n_signals))
    signal[:, :, 0] = np.sin(2 * np.pi * time[:, np.newaxis] * frequency_of_interest)
    phase_offset = np.random.uniform(-np.pi, np.pi, size=(n_time_samples, n_trials))
    phase_offset[np.where(time > 1.5), :] = np.pi / 2
    signal[:, :, 1] = np.sin(
        (2 * np.pi * time[:, np.newaxis] * frequency_of_interest) + phase_offset
    )
    noise = np.random.normal(0, 2, signal.shape)

    multitaper = Multitaper(
        prepare_time_series(signal + noise),
        sampling_frequency=sampling_frequency,
        time_halfbandwidth_product=1,
        time_window_duration=0.080,
        time_window_step=0.080,
        start_time=time[0],
    )
    connectivity = Connectivity.from_multitaper(multitaper)

    assert connectivity.coherence_magnitude() == snapshot
    assert connectivity.time == snapshot


# Add more snapshot tests for:
# - Granger causality
# - Phase locking value
# - Canonical coherence
# - Global coherence
# - Different connectivity measures from Tutorial_Using_Paper_Examples


@pytest.mark.slow
def test_tutorial_simulated_examples_executes():
    """Verify Tutorial_On_Simulated_Examples notebook executes without errors."""
    import subprocess

    result = subprocess.run(
        [
            "jupyter", "nbconvert",
            "--to", "notebook",
            "--execute",
            "--ExecutePreprocessor.timeout=600",
            "examples/Tutorial_On_Simulated_Examples.ipynb",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, (
        f"Notebook execution failed:\n"
        f"STDOUT: {result.stdout}\n"
        f"STDERR: {result.stderr}"
    )


@pytest.mark.slow
def test_tutorial_paper_examples_executes():
    """Verify Tutorial_Using_Paper_Examples notebook executes without errors."""
    import subprocess

    result = subprocess.run(
        [
            "jupyter", "nbconvert",
            "--to", "notebook",
            "--execute",
            "--ExecutePreprocessor.timeout=600",
            "examples/Tutorial_Using_Paper_Examples.ipynb",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, (
        f"Notebook execution failed:\n"
        f"STDOUT: {result.stdout}\n"
        f"STDERR: {result.stderr}"
    )
```

### Step 4: Generate Initial Snapshots

```bash
# Install syrupy
pip install syrupy

# Generate snapshots for the first time
pytest tests/test_notebooks.py --snapshot-update

# Verify snapshots were created
ls tests/__snapshots__/test_notebooks/

# Commit the snapshots
git add tests/__snapshots__/
git commit -m "Add initial notebook snapshots"
```

### Step 5: CI Configuration (Optional)

**Option A: Run slow tests only on PRs**

```yaml
# .github/workflows/tests.yml
- name: Run fast tests
  run: pytest -m "not slow"

- name: Run slow notebook tests (PR only)
  if: github.event_name == 'pull_request'
  run: pytest -m slow
```

**Option B: Run all tests always**

```yaml
- name: Run all tests
  run: pytest
```

## Usage

### Running Tests

```bash
# Run fast snapshot tests (default)
pytest tests/test_notebooks.py

# Run slow execution tests
pytest tests/test_notebooks.py -m slow

# Run all tests
pytest tests/test_notebooks.py -m ""

# Run specific test
pytest tests/test_notebooks.py::test_power_spectrum_200hz
```

### Updating Snapshots

When you intentionally change behavior:

```bash
# Update all snapshots
pytest tests/test_notebooks.py --snapshot-update

# Update specific snapshot
pytest tests/test_notebooks.py::test_power_spectrum_200hz --snapshot-update

# Review changes
git diff tests/__snapshots__/

# Commit updated snapshots
git add tests/__snapshots__/
git commit -m "Update snapshots after fixing X"
```

### Debugging Snapshot Failures

When a test fails:

1. **Review the diff:** Syrupy shows what changed
2. **Check if expected:** Is this an intentional behavior change?
3. **If expected:** Update snapshot with `--snapshot-update`
4. **If unexpected:** Fix the bug causing the regression

```bash
# See detailed diff
pytest tests/test_notebooks.py::test_power_spectrum_200hz -vv
```

## Maintenance

### Adding New Tests

1. Identify key scenario from notebooks
2. Write focused test function
3. Generate snapshot: `pytest path/to/test.py::test_name --snapshot-update`
4. Commit test and snapshot

### When Notebooks Change

- **If notebooks are documentation only:** Tests don't need to change
- **If tests fail after notebook update:** Decide if behavior change is intentional
  - Intentional: Update snapshots
  - Bug: Fix the code

### Random Seed Management

All tests use `np.random.seed(42)` for reproducibility. This ensures:
- Snapshots are deterministic
- Tests pass consistently
- Noise doesn't cause false failures

## Benefits

1. **Fast feedback:** Snapshot tests run quickly, catch regressions immediately
2. **Comprehensive coverage:** Test all key connectivity measures and scenarios
3. **Easy maintenance:** Hand-written tests are clear and focused
4. **Flexible:** Can run subset of tests (fast vs slow)
5. **Documented:** Tests serve as executable documentation of expected behavior
6. **Regression protection:** Numerical outputs are locked down

## Trade-offs

**Pros:**
- Catches numerical regressions
- Fast to run (snapshot tests)
- Clear test failures
- Version-controlled expected outputs

**Cons:**
- Snapshots must be updated when behavior changes
- Need to distinguish bugs from intentional changes
- Binary snapshots not human-readable (but syrupy shows diffs)

## Future Enhancements

- Add more connectivity measures (Granger, phase metrics)
- Test edge cases (odd N, single trials, etc.)
- Performance benchmarks (track execution time)
- Visual regression testing for plots (optional)
