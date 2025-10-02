"""Audio processing utilities for the pitch comparison CLI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
from scipy import signal
from scipy.io import wavfile
from scipy.signal import wiener

try:  # Optional dependency - may not be available in CI
    import soundfile as sf  # type: ignore
except Exception:  # pragma: no cover - soundfile is optional
    sf = None  # type: ignore

try:  # Optional dependency - full audio analysis toolkit
    import librosa  # type: ignore
except Exception:  # pragma: no cover - dependency may be absent
    librosa = None  # type: ignore

from spectrum_analysis.audio_sources import MicSource, sd

from spectrum_analysis.comb_trigger import record_with_harmonic_comb

if TYPE_CHECKING:  # pragma: no cover - only for type checking
    from spectrum_analysis.pitch_compare_config import PitchCompareConfig


@dataclass(frozen=True)
class NoiseProfile:
    """Stationary noise statistics cached for Wiener filtering."""

    freqs: np.ndarray
    spectrum: np.ndarray
    window_length: int
    hop_length: int
    rms: float
    variance: float


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


def _fit_stft_parameters(
    window_length: int, hop_length: int, signal_length: int
) -> tuple[int, int]:
    """Clamp STFT parameters so they are valid for ``signal_length`` samples."""

    signal_length = max(int(signal_length), 1)
    window_length = max(int(window_length), 1)
    hop_length = max(int(hop_length), 1)

    if window_length > signal_length:
        hop_fraction = hop_length / float(window_length)
        window_length = signal_length
        hop_length = max(int(round(window_length * hop_fraction)), 1)

    hop_length = min(hop_length, window_length)
    if window_length > 1 and hop_length >= window_length:
        hop_length = window_length - 1

    return window_length, hop_length


def compute_noise_profile(noise: np.ndarray, cfg: "PitchCompareConfig") -> NoiseProfile:
    """Estimate stationary noise statistics for Wiener filtering."""

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
    variance = float(np.var(noise, dtype=np.float64) + 1e-12)
    return NoiseProfile(
        freqs=np.asarray(freqs, dtype=np.float32),
        spectrum=np.asarray(spectrum, dtype=np.complex64),
        window_length=int(win_len),
        hop_length=int(hop_len),
        rms=rms,
        variance=variance,
    )


def apply_noise_filter(
    samples: np.ndarray,
    profile: NoiseProfile,
    *,
    over_subtraction: float = 1.0,
) -> np.ndarray:
    """Apply stationary-noise reduction to ``samples`` using ``profile``."""

    if samples.size == 0:
        return np.zeros(0, dtype=np.float32)

    shaped = np.asarray(samples, dtype=np.float32)
    return wiener_filter_signal(
        shaped,
        profile,
        over_subtraction=over_subtraction,
        window=_default_wiener_window(profile),
    )


def _default_wiener_window(profile: NoiseProfile) -> int:
    """Heuristic Wiener window size derived from the profiled STFT settings."""

    hop = max(int(profile.hop_length), 1)
    base = max(int(round(profile.window_length / hop)), 1)
    if base % 2 == 0:
        base += 1
    return max(base, 3)


def wiener_filter_signal(
    signal_in: np.ndarray,
    profile: NoiseProfile,
    *,
    over_subtraction: float = 1.0,
    window: Optional[int] = None,
) -> np.ndarray:
    """Apply a Wiener filter using statistics from ``profile``."""

    if signal_in.size == 0:
        return np.asarray(signal_in, dtype=np.float32)

    if window is None or window < 3:
        window = _default_wiener_window(profile)

    noise_var = float(profile.variance)
    if not np.isfinite(noise_var) or noise_var <= 0:
        noise_var = float(profile.rms**2)

    scale = max(float(over_subtraction), 0.0)
    noise_estimate = noise_var * (scale**2)
    if not np.isfinite(noise_estimate) or noise_estimate <= 0:
        noise_estimate = None

    filtered = wiener(signal_in, mysize=window, noise=noise_estimate)
    return np.asarray(filtered, dtype=np.float32)


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
        variance=float(profile.variance),
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
            variance = float(data["variance"]) if "variance" in data else rms**2
    except (OSError, KeyError, ValueError):
        return None

    return NoiseProfile(
        freqs=freqs,
        spectrum=spectrum,
        window_length=window_length,
        hop_length=hop_length,
        rms=rms,
        variance=variance,
    )


def compute_spectrogram(
    audio: np.ndarray, cfg: "PitchCompareConfig"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a magnitude spectrogram for ``audio``."""

    win_len, hop_len = determine_window_and_hop(cfg, len(audio))
    win_len, hop_len = _fit_stft_parameters(win_len, hop_len, len(audio))
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
    """Reduce stationary noise using a Wiener filter."""

    win_len, hop_len = _fit_stft_parameters(
        noise_profile.window_length, noise_profile.hop_length, len(audio)
    )

    filtered = wiener_filter_signal(
        audio,
        noise_profile,
        over_subtraction=float(cfg.over_subtraction),
        window=_default_wiener_window(noise_profile),
    )

    freqs, times, stft = signal.stft(
        filtered,
        fs=cfg.sample_rate,
        window="hann",
        nperseg=win_len,
        noverlap=win_len - hop_len,
        padded=False,
    )
    clean_power = np.abs(stft) ** 2
    return filtered, freqs, times, clean_power


def record_noise_sample(cfg: "PitchCompareConfig") -> np.ndarray:
    """Capture a noise sample for calibration or load it from disk."""

    duration_samples = int(cfg.noise_duration * cfg.sample_rate)

    if cfg.input_mode == "file" and cfg.noise_audio_path:
        noise, _ = load_audio(Path(cfg.noise_audio_path), cfg.sample_rate)
        if len(noise) > duration_samples:
            return noise[:duration_samples]
        if len(noise) < duration_samples:
            pad = duration_samples - len(noise)
            return np.pad(noise, (0, pad), mode="edge")
        return noise

    if cfg.input_mode == "file":
        if cfg.input_audio_path is None:
            raise ValueError(
                "input_audio_path must be provided when input_mode is 'file'"
            )
        audio, _ = load_audio(Path(cfg.input_audio_path), cfg.sample_rate)
        return audio[:duration_samples]

    if sd is None:
        raise RuntimeError(
            "sounddevice is required for microphone recording but is not available."
        )

    print(f"[INFO] Recording {cfg.noise_duration:.1f}s of background noise...")
    noise = sd.rec(
        duration_samples, samplerate=cfg.sample_rate, channels=1, dtype="float32"
    )
    sd.wait()
    return np.squeeze(noise).astype(np.float32)


def _acquire_audio_snr(cfg: "PitchCompareConfig", noise_rms: float) -> np.ndarray:
    _, hop = determine_window_and_hop(cfg)
    source = MicSource(cfg.sample_rate, hop)
    source.start()
    print("[INFO] Listening for audio events (RMS trigger)...")
    snr_threshold = 10 ** (cfg.snr_threshold_db / 20.0)
    collected: list[np.ndarray] = []
    above = False
    recording_started = False
    idle_samples = 0
    idle_limit = int(cfg.idle_timeout * cfg.sample_rate)
    max_samples = int(cfg.max_record_seconds * cfg.sample_rate)
    collected_samples = 0

    try:
        while collected_samples < max_samples:
            chunk = source.read()
            if chunk.size == 0:
                continue

            chunk_rms = np.sqrt(np.mean(np.square(chunk)) + 1e-12)
            ratio = chunk_rms / (noise_rms + 1e-12)

            if ratio >= snr_threshold:
                if not recording_started:
                    print("[INFO] Recording started.")
                    recording_started = True
                above = True
                idle_samples = 0
                collected.append(chunk)
                collected_samples += len(chunk)
            elif above:
                idle_samples += len(chunk)
                collected.append(chunk)
                collected_samples += len(chunk)
                if idle_samples >= idle_limit:
                    print("[INFO] Recording stopped (signal below threshold).")
                    break
        else:
            print("[WARN] Max recording length reached.")
    finally:
        source.stop()

    if not collected:
        raise RuntimeError("No audio captured above the SNR threshold.")

    return np.concatenate(collected).astype(np.float32)


def acquire_audio(cfg: "PitchCompareConfig", noise_rms: float) -> np.ndarray:
    """Record audio using the configured trigger or load from file."""

    if cfg.input_mode == "file":
        if cfg.input_audio_path is None:
            raise ValueError(
                "input_audio_path must be provided when input_mode is 'file'"
            )
        audio, _ = load_audio(Path(cfg.input_audio_path), cfg.sample_rate)
        return audio

    trigger_mode = getattr(cfg, "trigger_mode", "snr")

    if trigger_mode != "harmonic_comb":
        return _acquire_audio_snr(cfg, noise_rms)

    expected_f0 = cfg.expected_f0
    if expected_f0 is None or not np.isfinite(expected_f0) or expected_f0 <= 0.0:
        print("[WARN] expected_f0 missing; falling back to RMS trigger.")
        return _acquire_audio_snr(cfg, noise_rms)

    try:
        return record_with_harmonic_comb(
            expected_f0=expected_f0,
            sample_rate=cfg.sample_rate,
            max_record_seconds=cfg.max_record_seconds,
            comb_cfg=cfg.comb_trigger,
        )
    except ValueError:
        print("[WARN] Invalid frequency band; falling back to RMS trigger.")
        return _acquire_audio_snr(cfg, noise_rms)
