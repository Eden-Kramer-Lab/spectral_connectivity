"""Tests for GPU backend detection and configuration."""

import os
import sys
from unittest.mock import patch

import pytest

from spectral_connectivity import get_compute_backend


class TestGetComputeBackend:
    """Test get_compute_backend() function."""

    def test_cpu_mode_default(self):
        """Test that CPU mode is default when GPU not enabled."""
        # Ensure env var is not set
        with patch.dict(os.environ, {}, clear=False):
            if "SPECTRAL_CONNECTIVITY_ENABLE_GPU" in os.environ:
                del os.environ["SPECTRAL_CONNECTIVITY_ENABLE_GPU"]

            result = get_compute_backend()

            assert result["backend"] == "cpu"
            assert result["gpu_enabled"] is False
            assert (
                result["gpu_available"] is not None
            )  # Should report if CuPy is available
            assert "device_name" in result
            assert "message" in result
            assert isinstance(result["message"], str)
            assert len(result["message"]) > 0

    def test_cpu_mode_explicit_false(self):
        """Test CPU mode when explicitly set to false."""
        with patch.dict(os.environ, {"SPECTRAL_CONNECTIVITY_ENABLE_GPU": "false"}):
            result = get_compute_backend()

            assert result["backend"] == "cpu"
            assert result["gpu_enabled"] is False

    def test_gpu_mode_when_cupy_available(self):
        """Test GPU mode when CuPy is available and enabled."""
        # Only run if cupy is actually installed
        try:
            import cupy  # noqa: F401

            cupy_available = True
        except ImportError:
            cupy_available = False

        if not cupy_available:
            pytest.skip("CuPy not installed, skipping GPU test")

        with patch.dict(os.environ, {"SPECTRAL_CONNECTIVITY_ENABLE_GPU": "true"}):
            # Need to reload modules to pick up env var change
            # This test checks the function, not the actual module imports
            result = get_compute_backend()

            # When cupy is available and enabled
            assert result["backend"] in ["cpu", "gpu"]  # Depends on current state
            assert "gpu_available" in result
            assert result["gpu_available"] is True
            assert "message" in result

    def test_gpu_mode_when_cupy_not_available(self):
        """Test GPU mode when CuPy is not available."""
        with patch.dict(os.environ, {"SPECTRAL_CONNECTIVITY_ENABLE_GPU": "true"}):
            # Mock cupy as not available
            with patch.dict(sys.modules, {"cupy": None}):
                result = get_compute_backend()

                # Should report that GPU was requested but not available
                assert "gpu_enabled" in result
                assert "gpu_available" in result
                assert result["gpu_available"] is False
                assert "message" in result
                assert (
                    "cupy" in result["message"].lower()
                    or "gpu" in result["message"].lower()
                )

    def test_return_value_structure(self):
        """Test that return value has all required keys."""
        result = get_compute_backend()

        required_keys = {
            "backend",
            "gpu_enabled",
            "gpu_available",
            "device_name",
            "message",
        }
        assert set(result.keys()) == required_keys

    def test_backend_values(self):
        """Test that backend field only has valid values."""
        result = get_compute_backend()

        assert result["backend"] in ["cpu", "gpu"]

    def test_boolean_fields(self):
        """Test that boolean fields are actually booleans."""
        result = get_compute_backend()

        assert isinstance(result["gpu_enabled"], bool)
        assert isinstance(result["gpu_available"], bool)

    def test_device_name_present(self):
        """Test that device_name is always a string."""
        result = get_compute_backend()

        assert isinstance(result["device_name"], str)

    def test_message_is_helpful(self):
        """Test that message provides useful information."""
        result = get_compute_backend()

        # Message should explain the current state
        assert len(result["message"]) > 20  # Should be a meaningful sentence
        assert isinstance(result["message"], str)

    def test_detect_cupy_import_state(self):
        """Test that function detects if cupy is already imported."""
        # Check if cupy is in sys.modules
        cupy_was_imported = "cupy" in sys.modules

        result = get_compute_backend()

        # If cupy was imported before calling get_compute_backend,
        # gpu_available should reflect that
        if cupy_was_imported:
            assert result["gpu_available"] is True


class TestGPUModeConsistency:
    """Test that GPU mode detection is consistent with actual module imports."""

    def test_environment_variable_respected(self):
        """Test that SPECTRAL_CONNECTIVITY_ENABLE_GPU env var is respected."""
        result = get_compute_backend()

        env_var = os.environ.get("SPECTRAL_CONNECTIVITY_ENABLE_GPU")

        if env_var == "true":
            assert result["gpu_enabled"] is True
        else:
            assert result["gpu_enabled"] is False

    def test_cpu_backend_details(self):
        """Test that CPU backend provides appropriate details."""
        with patch.dict(os.environ, {}, clear=False):
            if "SPECTRAL_CONNECTIVITY_ENABLE_GPU" in os.environ:
                del os.environ["SPECTRAL_CONNECTIVITY_ENABLE_GPU"]

            result = get_compute_backend()

            if result["backend"] == "cpu":
                # CPU backend should indicate numpy
                assert (
                    "cpu" in result["message"].lower()
                    or "numpy" in result["message"].lower()
                )
                # Device name should indicate CPU
                assert (
                    "cpu" in result["device_name"].lower()
                    or result["device_name"] == "CPU"
                )


class TestIsGpuEnabled:
    """Test is_gpu_enabled() environment-variable parsing."""

    @pytest.mark.parametrize(
        "value", ["true", "True", "TRUE", "1", "yes", "on", " true "]
    )
    def test_recognized_true_values(self, value):
        from spectral_connectivity.utils import is_gpu_enabled

        with patch.dict(os.environ, {"SPECTRAL_CONNECTIVITY_ENABLE_GPU": value}):
            assert is_gpu_enabled() is True

    @pytest.mark.parametrize("value", ["false", "0", "no", "off", ""])
    def test_recognized_false_values(self, value):
        from spectral_connectivity.utils import is_gpu_enabled

        with patch.dict(os.environ, {"SPECTRAL_CONNECTIVITY_ENABLE_GPU": value}):
            assert is_gpu_enabled() is False

    def test_unset_is_false(self):
        from spectral_connectivity.utils import is_gpu_enabled

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SPECTRAL_CONNECTIVITY_ENABLE_GPU", None)
            assert is_gpu_enabled() is False

    def test_unrecognized_value_warns_and_falls_back(self):
        from spectral_connectivity.utils import is_gpu_enabled

        with patch.dict(os.environ, {"SPECTRAL_CONNECTIVITY_ENABLE_GPU": "maybe"}):
            with pytest.warns(UserWarning, match="not a recognized value"):
                assert is_gpu_enabled() is False


class TestBackendDetection:
    """Test that get_compute_backend reports 'gpu' when xp is the cupy module."""

    def test_reports_gpu_when_transforms_xp_is_cupy(self):
        import types

        fake_cupy = types.ModuleType("cupy")  # __name__ == "cupy"
        fake_transforms = types.ModuleType("spectral_connectivity.transforms")
        fake_transforms.xp = fake_cupy

        with patch.dict(
            sys.modules,
            {"spectral_connectivity.transforms": fake_transforms},
        ):
            result = get_compute_backend()
            assert result["backend"] == "gpu"

    def test_reports_cpu_when_transforms_xp_is_numpy(self):
        import types

        import numpy as np

        fake_transforms = types.ModuleType("spectral_connectivity.transforms")
        fake_transforms.xp = np  # __name__ == "numpy"

        with patch.dict(
            sys.modules,
            {"spectral_connectivity.transforms": fake_transforms},
        ):
            result = get_compute_backend()
            assert result["backend"] == "cpu"
