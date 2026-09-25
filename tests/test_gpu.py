"""Tests for GPU backend detection and configuration.

The compute backend is fixed when ``spectral_connectivity`` is first imported
(``_backend.xp`` is numpy or cupy), so patching the environment afterwards only
changes what :func:`get_compute_backend` reports as *requested*
(``gpu_enabled``), never the imported ``backend``. Assertions about the imported
backend therefore depend on the session's actual ``_backend.xp``.
"""

import importlib.machinery
import importlib.util
import os
import subprocess
import sys
import types
from unittest.mock import patch

import numpy as np
import pytest

from spectral_connectivity import _backend, get_compute_backend
from spectral_connectivity.utils import GPU_ENV_VAR, is_gpu_enabled

_SESSION_IS_CPU = _backend.xp.__name__ == "numpy"
cpu_session_only = pytest.mark.skipif(
    not _SESSION_IS_CPU, reason="backend assertions assume a NumPy-backed import"
)


@pytest.fixture
def cpu_backend():
    """Report a NumPy-backed import regardless of the session's real backend."""
    fake_backend = types.ModuleType("spectral_connectivity._backend")
    fake_backend.xp = np
    with patch.dict(sys.modules, {"spectral_connectivity._backend": fake_backend}):
        yield


@pytest.fixture
def cupy_not_installed():
    """Make CuPy both absent from ``sys.modules`` and unimportable."""
    with patch.dict(sys.modules, {"cupy": None}):
        yield


@pytest.fixture
def cupy_installed_not_imported(monkeypatch):
    """Make CuPy discoverable by ``find_spec`` without importing it."""
    monkeypatch.delitem(sys.modules, "cupy", raising=False)
    real_find_spec = importlib.util.find_spec

    def find_spec(name, *args, **kwargs):
        if name == "cupy":
            return importlib.machinery.ModuleSpec("cupy", loader=None)
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)


class TestGetComputeBackend:
    """Test get_compute_backend() reporting."""

    def test_return_value_structure(self):
        """The report has exactly the documented keys and value types."""
        result = get_compute_backend()

        assert set(result) == {
            "backend",
            "gpu_enabled",
            "gpu_available",
            "device_name",
            "message",
        }
        assert isinstance(result["gpu_enabled"], bool)
        assert isinstance(result["gpu_available"], bool)
        assert isinstance(result["device_name"], str)
        assert isinstance(result["message"], str)
        assert result["backend"] == ("cpu" if _SESSION_IS_CPU else "gpu")

    @pytest.mark.parametrize("value", ["true", "1", "yes"])
    def test_gpu_enabled_reflects_env_request(self, value):
        """A truthy env value is reported as a GPU request."""
        with patch.dict(os.environ, {GPU_ENV_VAR: value}):
            assert get_compute_backend()["gpu_enabled"] is True

    @pytest.mark.parametrize("value", ["false", "0"])
    def test_gpu_enabled_false_for_falsy_env(self, value):
        with patch.dict(os.environ, {GPU_ENV_VAR: value}):
            assert get_compute_backend()["gpu_enabled"] is False

    def test_gpu_enabled_false_when_env_unset(self, monkeypatch):
        monkeypatch.delenv(GPU_ENV_VAR, raising=False)
        assert get_compute_backend()["gpu_enabled"] is False

    @pytest.mark.parametrize("value", ["true", "1", "false", "0", ""])
    def test_gpu_enabled_matches_env_parser(self, value):
        """gpu_enabled uses the same parser as the import-time backend switch."""
        with patch.dict(os.environ, {GPU_ENV_VAR: value}):
            assert get_compute_backend()["gpu_enabled"] is is_gpu_enabled()

    def test_requested_but_cupy_missing_message(self, cpu_backend, cupy_not_installed):
        """GPU requested without CuPy: say so and how to install it."""
        with patch.dict(os.environ, {GPU_ENV_VAR: "true"}):
            result = get_compute_backend()

        assert result["gpu_enabled"] is True
        assert result["gpu_available"] is False
        assert result["backend"] == "cpu"
        assert result["device_name"] == "CPU"
        assert "GPU acceleration was requested" in result["message"]
        assert "CuPy is not installed" in result["message"]
        assert "pip install cupy" in result["message"]

    def test_cupy_installed_but_not_requested_message(
        self, monkeypatch, cpu_backend, cupy_installed_not_imported
    ):
        """CuPy importable but not requested: explain how to turn the GPU on."""
        monkeypatch.delenv(GPU_ENV_VAR, raising=False)
        result = get_compute_backend()

        assert result["gpu_enabled"] is False
        assert result["gpu_available"] is True
        assert result["backend"] == "cpu"
        assert "CuPy is installed and GPU acceleration is available" in result["message"]
        assert f"{GPU_ENV_VAR}='true'" in result["message"]

    def test_cpu_only_message(self, monkeypatch, cpu_backend, cupy_not_installed):
        """Neither requested nor installed: CPU device and setup instructions."""
        monkeypatch.delenv(GPU_ENV_VAR, raising=False)
        result = get_compute_backend()

        assert result["gpu_enabled"] is False
        assert result["gpu_available"] is False
        assert result["backend"] == "cpu"
        assert result["device_name"] == "CPU"
        assert result["message"].startswith("Using CPU backend with NumPy.")
        assert "Install CuPy" in result["message"]

    def test_already_imported_cupy_counts_as_available(self, monkeypatch):
        """A CuPy already in ``sys.modules`` is reported as available.

        The fake module has no ``cuda`` attribute, so device probing fails and the
        generic ``"GPU"`` device name is reported.
        """
        monkeypatch.setitem(sys.modules, "cupy", types.ModuleType("cupy"))
        result = get_compute_backend()

        assert result["gpu_available"] is True
        assert result["device_name"] == "GPU"


@cpu_session_only
class TestCpuSessionBackend:
    """In a NumPy-backed session, env changes never switch the reported backend."""

    @pytest.mark.parametrize("value", ["true", "false"])
    def test_backend_stays_cpu_after_env_change(self, value):
        with patch.dict(os.environ, {GPU_ENV_VAR: value}):
            assert get_compute_backend()["backend"] == "cpu"


@pytest.mark.parametrize("value", ["true", "1"])
def test_gpu_request_without_cupy_fails_import(value):
    """Requesting the GPU without CuPy makes the package import fail loudly.

    Runs in a fresh interpreter because the backend is chosen at import time;
    CuPy is blocked before ``spectral_connectivity`` is imported. The message
    echoes the value that requested the GPU.
    """
    code = "import sys\nsys.modules['cupy'] = None\nimport spectral_connectivity\n"
    environment = {**os.environ, GPU_ENV_VAR: value}
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert result.returncode != 0
    assert "RuntimeError: GPU support was explicitly requested" in result.stderr
    assert f"{GPU_ENV_VAR}={value!r}" in result.stderr
    assert "CuPy is not installed" in result.stderr
    assert "pip install cupy" in result.stderr


class TestIsGpuEnabled:
    """Test is_gpu_enabled() environment-variable parsing."""

    @pytest.mark.parametrize("value", ["true", "True", "TRUE", "1", "yes", "on", " true "])
    def test_recognized_true_values(self, value):
        with patch.dict(os.environ, {GPU_ENV_VAR: value}):
            assert is_gpu_enabled() is True

    @pytest.mark.parametrize("value", ["false", "0", "no", "off", ""])
    def test_recognized_false_values(self, value):
        with patch.dict(os.environ, {GPU_ENV_VAR: value}):
            assert is_gpu_enabled() is False

    def test_unset_is_false(self, monkeypatch):
        monkeypatch.delenv(GPU_ENV_VAR, raising=False)
        assert is_gpu_enabled() is False

    def test_unrecognized_value_warns_and_falls_back(self):
        with (
            patch.dict(os.environ, {GPU_ENV_VAR: "maybe"}),
            pytest.warns(UserWarning, match="not a recognized value"),
        ):
            assert is_gpu_enabled() is False


class TestBackendDetection:
    """Test that get_compute_backend reports 'gpu' when xp is the cupy module."""

    def test_reports_gpu_when_backend_xp_is_cupy(self):
        fake_cupy = types.ModuleType("cupy")  # __name__ == "cupy"
        fake_backend = types.ModuleType("spectral_connectivity._backend")
        fake_backend.xp = fake_cupy

        with patch.dict(sys.modules, {"spectral_connectivity._backend": fake_backend}):
            result = get_compute_backend()
            assert result["backend"] == "gpu"

    def test_reports_cpu_when_backend_xp_is_numpy(self, cpu_backend):
        assert get_compute_backend()["backend"] == "cpu"
