"""Audio processing utilities for the pitch comparison CLI."""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class NoiseProfile:
    """Stationary noise statistics cached for spectral subtraction."""

    freqs: np.ndarray
    spectrum: np.ndarray
    window_length: int
    hop_length: int
    rms: float


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


def compute_noise_profile(noise: np.ndarray, cfg: "PitchCompareConfig") -> NoiseProfile:
    """Estimate the stationary noise spectrum for later subtraction."""

    win_len, hop_len = determine_window_and_hop(cfg, len(noise))
    freqs, _, stft = signal.stft(
        noise,
        fs=cfg.sample_rate,
        window="hann",
        nperseg=win_len,
        noverlap=win_len - hop_len,
        padded=False,
    )
    spectrum = np.mean(stft, axis=1)
    rms = float(np.sqrt(np.mean(np.square(noise)) + 1e-12))
    return NoiseProfile(
        freqs=np.asarray(freqs, dtype=np.float32),
        spectrum=np.asarray(spectrum, dtype=np.complex64),
        window_length=int(win_len),
        hop_length=int(hop_len),
        rms=rms,
    )


def save_noise_profile(
    profile: NoiseProfile, cache_path: Path, sample_rate: int
) -> None:
    """Persist a noise profile to disk for reuse across runs."""

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        freqs=profile.freqs.astype(np.float32, copy=False),
        spectrum=profile.spectrum.astype(np.complex64, copy=False),
        window_length=int(profile.window_length),
        hop_length=int(profile.hop_length),
        rms=float(profile.rms),
        sample_rate=int(sample_rate),
    )


def load_noise_profile(
    cache_path: Path,
    cfg: "PitchCompareConfig",
    *,
    expected_window: Optional[int] = None,
    expected_hop: Optional[int] = None,
) -> Optional[NoiseProfile]:
    """Load a cached noise profile if it matches the current configuration."""

    if not cache_path.exists():
        return None

    try:
        with np.load(cache_path, allow_pickle=False) as data:
            sample_rate = int(data["sample_rate"])
            if sample_rate != int(cfg.sample_rate):
                return None

            window_length = int(data["window_length"])
            hop_length = int(data["hop_length"])

            if expected_window is not None and window_length != expected_window:
                return None
            if expected_hop is not None and hop_length != expected_hop:
                return None
            freqs = np.asarray(data["freqs"], dtype=np.float32)
            spectrum = np.asarray(data["spectrum"], dtype=np.complex64)
            rms = float(data["rms"])
    except (OSError, KeyError, ValueError):
        return None

    return NoiseProfile(
        freqs=freqs,
        spectrum=spectrum,
        window_length=window_length,
        hop_length=hop_length,
        rms=rms,
    )


def compute_spectrogram(
    audio: np.ndarray, cfg: "PitchCompareConfig"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a magnitude spectrogram for ``audio``."""

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
    return freqs, times, power


def subtract_noise(
    audio: np.ndarray, noise_profile: NoiseProfile, cfg: "PitchCompareConfig"
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Perform spectral subtraction using a stationary noise spectrum."""

    win_len = noise_profile.window_length
    hop_len = noise_profile.hop_length
    freqs, times, stft = signal.stft(
        audio,
        fs=cfg.sample_rate,
        window="hann",
        nperseg=win_len,
        noverlap=win_len - hop_len,
        padded=False,
    )
    if stft.shape[0] != noise_profile.spectrum.size:
        raise ValueError(
            "Noise profile frequency bins do not match the audio STFT dimensions."
        )

    noise_spectrum = noise_profile.spectrum.reshape(-1, 1)
    cleaned_stft = stft - noise_spectrum * complex(float(cfg.over_subtraction))
    _, reconstructed = signal.istft(
        cleaned_stft,
        fs=cfg.sample_rate,
        window="hann",
        nperseg=win_len,
        noverlap=win_len - hop_len,
        input_onesided=True,
    )
    reconstructed = np.asarray(reconstructed, dtype=np.float32)
    clean_power = np.abs(cleaned_stft) ** 2
    return reconstructed, freqs, times, clean_power
