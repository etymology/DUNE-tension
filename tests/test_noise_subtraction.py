"""Tests for stationary noise reduction utilities."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from src.spectrum_analysis.audio_processing import (
    compute_noise_profile,
    subtract_noise,
)
from src.spectrum_analysis.compare_pitch_cli import PitchCompareConfig


def test_stationary_noise_subtraction_removes_noise() -> None:
    """A stationary noise sample should be attenuated by Wiener filtering."""

    cfg = PitchCompareConfig(
        sample_rate=8000,
        min_frequency=40.0,
        min_oscillations_per_window=10.0,
        min_window_overlap=0.5,
        over_subtraction=1.0,
    )

    rng = np.random.default_rng(seed=1234)
    duration = int(0.5 * cfg.sample_rate)
    noise = rng.normal(scale=0.05, size=duration).astype(np.float32)

    profile = compute_noise_profile(noise, cfg)
    filtered, _, _, _ = subtract_noise(noise, profile, cfg)

    residual_rms = float(np.sqrt(np.mean(filtered**2) + 1e-12))
    original_rms = float(np.sqrt(np.mean(noise**2) + 1e-12))

    assert residual_rms < original_rms * 0.5
