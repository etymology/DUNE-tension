"""High-level workflows for capturing and classifying pitch in real time."""

from __future__ import annotations

import dataclasses
from typing import Optional, Tuple

import numpy as np

from .audio_processing import (
    acquire_audio,
    compute_noise_profile,
    record_noise_sample,
    subtract_noise,
)
from .crepe_analysis import activations_to_pitch, get_activations
from .pitch_compare_config import PitchCompareConfig


PitchEstimationResult = Tuple[
    float,
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]


def listen_for_trigger_and_classify(
    *,
    config: Optional[PitchCompareConfig] = None,
    expected_frequency: Optional[float] = None,
) -> PitchEstimationResult:
    """Capture audio via the harmonic comb trigger and estimate its pitch.

    The call blocks until the harmonic comb detector within
    :func:`~spectrum_analysis.audio_processing.acquire_audio` activates,
    capturing a single recording that is optionally noise-reduced before being
    classified with CREPE. The function returns the estimated pitch in Hertz,
    the corresponding confidence score, the activation frame times, CREPE's
    frequency axis, and the activation map itself.
    """

    if config is None:
        working_cfg = PitchCompareConfig()
    else:
        working_cfg = dataclasses.replace(config)

    if expected_frequency is not None:
        working_cfg = dataclasses.replace(working_cfg, expected_f0=expected_frequency)

    if working_cfg.input_mode != "mic":
        raise ValueError(
            "listen_for_trigger_and_classify requires a microphone input mode."
        )

    filtered_audio: np.ndarray
    if working_cfg.noise_duration > 0.0:
        noise = record_noise_sample(working_cfg)
        noise_profile = compute_noise_profile(noise, working_cfg)
        audio = acquire_audio(working_cfg, noise_profile.rms)
        filtered_audio, *_ = subtract_noise(audio, noise_profile, working_cfg)
    else:
        audio = acquire_audio(working_cfg, 0.0)
        filtered_audio = audio

    activation_result = get_activations(
        filtered_audio,
        working_cfg.sample_rate,
        expected_pitch=working_cfg.expected_f0,
        cfg=working_cfg,
        warn=False,
    )
    if activation_result is None:
        empty = np.empty(0, dtype=np.float32)
        empty_matrix = np.empty((0, 0), dtype=np.float32)
        return float("nan"), float("nan"), empty, empty, empty_matrix

    times, freq_axis, activation = activation_result
    pitch, confidence = activations_to_pitch(
        activation.T,
        times,
        freq_axis=freq_axis,
        expected_frequency=working_cfg.expected_f0,
    )
    return pitch, confidence, times, freq_axis, activation
