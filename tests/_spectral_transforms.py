"""Transforms shared by the tests of the SpectralTransform interface."""

from typing import Any


class PrecomputedTransform:
    """Only the public SpectralTransform interface; a coordinate may be ``None``."""

    def __init__(self, coefficients: Any, frequencies: Any, time: Any) -> None:
        self._coefficients = coefficients
        self.frequencies = frequencies
        self.time = time

    def fft(self) -> Any:
        return self._coefficients.copy()
