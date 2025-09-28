"""Audio processing utilities for the pitch comparison CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, TYPE_CHECKING

import numpy as np
from scipy import signal
from scipy.io import wavfile

try:  # Optional dependency - may not be available in CI
    import soundfile as sf  # type: ignore
except Exception:  # pragma: no cover - soundfile is optional
    sf = None  # type: ignore

try:  # Optional dependency - full audio analysis toolkit
    import librosa  # type: ignore
except Exception:  # pragma: no cover - dependency may be absent
    librosa = None  # type: ignore

if TYPE_CHECKING:  # pragma: no cover - only for type checking
    from .compare_pitch_cli import PitchCompareConfig


def load_audio(path: Path, target_sr: int) -> tuple[np.ndarray, int]:
    """Load an audio file and resample it to ``target_sr`` if needed."""

    if not path.exists():
        raise FileNotFoundError(path)

    if sf is not None:
        audio, sr = sf.read(path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
    else:
        sr, audio = wavfile.read(path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if audio.dtype != np.float32:
            max_val = (
                np.iinfo(audio.dtype).max
                if np.issubdtype(audio.dtype, np.integer)
                else 1.0
            )
            audio = audio.astype(np.float32) / max_val

    if sr != target_sr:
        if librosa is None:
            raise RuntimeError(
                "librosa is required to resample audio but is not available."
            )
        audio = librosa.resample(
            audio.astype(np.float32), orig_sr=sr, target_sr=target_sr
        )
        sr = target_sr

    return audio.astype(np.float32), sr


def determine_window_and_hop(
    cfg: "PitchCompareConfig", total_samples: Optional[int] = None
) -> tuple[int, int]:
    """Compute STFT window and hop sizes based on configuration constraints."""

    sample_rate = cfg.sample_rate
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive to determine window parameters.")

    min_frequency = max(cfg.min_frequency, 1e-12)
    min_oscillations = max(cfg.min_oscillations_per_window, 1e-12)

    min_overlap = float(cfg.min_window_overlap)
    if not np.isfinite(min_overlap):
        min_overlap = 0.0
    min_overlap = float(np.clip(min_overlap, 0.0, 0.999))

    if total_samples is None:
        desired_window_samples = int(
            round((min_oscillations / min_frequency) * sample_rate)
        )
        total_samples = max(desired_window_samples, 1)
    else:
        total_samples = max(int(total_samples), 1)

    total_duration = total_samples / sample_rate
    desired_window_sec = min_oscillations / min_frequency
    window_sec = min(desired_window_sec, total_duration)
    if not np.isfinite(window_sec) or window_sec <= 0:
        window_sec = total_duration if total_duration > 0 else 1.0 / sample_rate

    window_samples = int(round(window_sec * sample_rate))
    window_samples = max(min(window_samples, total_samples), 1)

    if total_samples >= 2 and window_samples < 2:
        window_samples = min(2, total_samples)

    if window_samples > 1 and window_samples % 2 == 1:
        if window_samples < total_samples:
            window_samples += 1
        else:
            window_samples = max(window_samples - 1, 1)

    max_step_fraction = max(1.0 - min_overlap, 0.0)
    hop_samples = int(np.floor(window_samples * max_step_fraction))
    hop_samples = max(min(hop_samples, window_samples), 1)

    return window_samples, hop_samples


def compute_noise_profile(
    noise: np.ndarray, cfg: "PitchCompareConfig"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate the noise profile for spectral subtraction."""

    win_len, hop_len = determine_window_and_hop(cfg, len(noise))
    freqs, times, stft = signal.stft(
        noise,
        fs=cfg.sample_rate,
        window="hann",
        nperseg=win_len,
        noverlap=win_len - hop_len,
        padded=False,
    )
    power = np.mean(np.abs(stft) ** 2, axis=1, keepdims=True)
    return freqs, times, power


def subtract_noise(
    audio: np.ndarray, noise_profile: np.ndarray, cfg: "PitchCompareConfig"
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Perform spectral subtraction to reduce noise in the signal."""

    win_len, hop_len = determine_window_and_hop(cfg, len(audio))
    freqs, times, stft = signal.stft(
        audio,
        fs=cfg.sample_rate,
        window="hann",
        nperseg=win_len,
        noverlap=win_len - hop_len,
        padded=False,
    )
    power = np.abs(stft) ** 2
    adjusted_noise = noise_profile * cfg.over_subtraction
    clean_power = np.maximum(power - adjusted_noise, 0.0)
    magnitude = np.sqrt(clean_power)
    cleaned_stft = magnitude * np.exp(1j * np.angle(stft))
    _, reconstructed = signal.istft(
        cleaned_stft,
        fs=cfg.sample_rate,
        window="hann",
        nperseg=win_len,
        noverlap=win_len - hop_len,
        input_onesided=True,
    )
    reconstructed = np.asarray(reconstructed, dtype=np.float32)
    return reconstructed, freqs, times, clean_power
