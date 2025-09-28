"""Utility helpers for spectrogram analysis."""

from __future__ import annotations

import numpy as np

EPS = 1e-12


def dbfs(x: np.ndarray) -> np.ndarray:
    """Convert a linear magnitude array into dBFS."""
    return 20.0 * np.log10(np.maximum(x, EPS))


def hann_window(n: int) -> np.ndarray:
    """Return a Hann window of length ``n`` as ``float32``."""
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)).astype(np.float32)


__all__ = ["EPS", "dbfs", "hann_window"]
