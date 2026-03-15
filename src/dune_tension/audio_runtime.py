from __future__ import annotations

import logging
import math
import os
from pathlib import Path
import random
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RUNTIME_NOISE_DIR = Path(
    os.environ.get("DUNE_TENSION_NOISE_DIR", _REPO_ROOT / "data" / "noise_filters")
)
_NOISE_FILTER_PATH = Path(
    os.environ.get(
        "DUNE_TENSION_NOISE_FILTER_PATH",
        _RUNTIME_NOISE_DIR / "noise_filter.npz",
    )
)
_LEGACY_NOISE_FILTER_PATH = Path(__file__).resolve().with_name("noise_filter.npz")

_cached_noise_profile: Any | None = None
_cached_noise_threshold = 0.0


def _pitch_compare_config(*, sample_rate: int, duration: float):
    from spectrum_analysis.pitch_compare_config import PitchCompareConfig

    return PitchCompareConfig(
        sample_rate=int(sample_rate),
        noise_duration=float(duration),
        input_mode="mic",
        show_plots=False,
    )


def _load_audio_processing_helpers():
    from spectrum_analysis.audio_processing import (
        apply_noise_filter,
        compute_noise_profile,
        load_noise_profile,
        record_noise_sample,
        save_noise_profile,
    )

    return {
        "apply_noise_filter": apply_noise_filter,
        "compute_noise_profile": compute_noise_profile,
        "load_noise_profile": load_noise_profile,
        "record_noise_sample": record_noise_sample,
        "save_noise_profile": save_noise_profile,
    }


def _load_noise_profile(*, sample_rate: int, duration: float = 1.0):
    global _cached_noise_profile, _cached_noise_threshold

    if _cached_noise_profile is not None:
        return _cached_noise_profile

    helpers = _load_audio_processing_helpers()
    cfg = _pitch_compare_config(sample_rate=sample_rate, duration=duration)

    for path in (_NOISE_FILTER_PATH, _LEGACY_NOISE_FILTER_PATH):
        if not path.exists():
            continue
        profile = helpers["load_noise_profile"](path, cfg)
        if profile is None:
            continue
        _cached_noise_profile = profile
        _cached_noise_threshold = float(getattr(profile, "rms", 0.0))
        return _cached_noise_profile

    _cached_noise_profile = None
    _cached_noise_threshold = 0.0
    return None


def get_noise_threshold() -> float:
    profile = _load_noise_profile(sample_rate=get_samplerate() or 44100)
    if profile is None:
        return 0.0
    return float(getattr(profile, "rms", 0.0))


def calibrate_background_noise(sample_rate: int, duration: float = 1.0) -> None:
    global _cached_noise_profile, _cached_noise_threshold

    helpers = _load_audio_processing_helpers()
    cfg = _pitch_compare_config(sample_rate=sample_rate, duration=duration)
    noise = helpers["record_noise_sample"](cfg)
    profile = helpers["compute_noise_profile"](noise, cfg)
    _NOISE_FILTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    helpers["save_noise_profile"](profile, _NOISE_FILTER_PATH, int(sample_rate))
    _cached_noise_profile = profile
    _cached_noise_threshold = float(getattr(profile, "rms", 0.0))


def record_audio_filtered(
    duration: float,
    sample_rate: int,
    plot: bool = False,
    normalize: bool = True,
):
    del plot

    helpers = _load_audio_processing_helpers()
    cfg = _pitch_compare_config(sample_rate=sample_rate, duration=duration)
    audio = np.asarray(helpers["record_noise_sample"](cfg), dtype=np.float32)

    profile = _load_noise_profile(sample_rate=sample_rate, duration=duration)
    if profile is not None:
        audio = np.asarray(
            helpers["apply_noise_filter"](audio, profile, over_subtraction=1.0),
            dtype=np.float32,
        )

    if normalize and audio.size:
        peak = float(np.max(np.abs(audio)))
        if peak > 0.0:
            audio = audio / peak

    amplitude = float(np.mean(np.abs(audio))) if audio.size else 0.0
    return audio, amplitude


def analyze_sample(audio_sample, sample_rate, wire_length):
    from dune_tension.tension_calculation import tension_pass, wire_equation
    from spectrum_analysis.pesto_analysis import estimate_pitch_from_audio

    frequency, confidence = estimate_pitch_from_audio(
        np.asarray(audio_sample, dtype=np.float32),
        int(sample_rate),
        expected_frequency=None,
    )
    tension = wire_equation(length=wire_length, frequency=frequency)["tension"]
    tension_ok = tension_pass(tension, wire_length)
    return frequency, confidence, tension, tension_ok


def get_samplerate():
    try:
        from spectrum_analysis.audio_sources import sd
    except Exception:
        return None

    if sd is None:
        return None

    try:
        device_info = sd.query_devices()
    except Exception as exc:
        LOGGER.warning("Failed to query audio devices: %s", exc)
        return None

    sound_device_index = next(
        (index for index, device in enumerate(device_info) if "default" in device["name"]),
        None,
    )
    if sound_device_index is None:
        LOGGER.warning("Couldn't find USB PnP Sound Device.")
        return None

    return device_info[sound_device_index]["default_samplerate"]


def spoof_audio_sample(npz_dir: str) -> np.ndarray:
    npz_files = [f for f in os.listdir(npz_dir) if f.endswith(".npz")]
    file_path = None
    if npz_files:
        file_path = os.path.join(npz_dir, random.choice(npz_files))

    data = None
    if file_path:
        try:
            with np.load(file_path) as loaded:
                if "audio" in loaded:
                    data = loaded["audio"]
        except Exception:
            data = None

    if data is None:
        sample_rate = 41000
        freq = 80
        total_samples = sample_rate
        wave = []
        for i in range(total_samples):
            phase = 2 * math.pi * freq * i / sample_rate
            wave.append(1.0 if math.sin(phase) >= 0 else -1.0)
        data = np.array(wave)

    return data


__all__ = [
    "analyze_sample",
    "calibrate_background_noise",
    "get_noise_threshold",
    "get_samplerate",
    "record_audio_filtered",
    "spoof_audio_sample",
]
