"""Legacy compatibility shim for the package-native audio runtime adapter."""

from __future__ import annotations

from dune_tension.audio_runtime import (
    analyze_sample,
    calibrate_background_noise,
    get_noise_threshold,
    get_samplerate,
    record_audio_filtered,
    spoof_audio_sample,
)

__all__ = [
    "analyze_sample",
    "calibrate_background_noise",
    "get_noise_threshold",
    "get_samplerate",
    "record_audio_filtered",
    "spoof_audio_sample",
]
