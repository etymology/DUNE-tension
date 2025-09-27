"""CLI for recording audio and comparing pitch detection methods."""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy import signal
from scipy.io import wavfile

from audio_sources import MicSource, sd

try:  # Optional dependency - may not be available in CI
    import soundfile as sf  # type: ignore
except Exception:  # pragma: no cover - soundfile is optional
    sf = None  # type: ignore

try:  # Optional dependency - heavy ML models
    import crepe  # type: ignore
except Exception:  # pragma: no cover - dependency may be absent
    crepe = None  # type: ignore

try:  # Optional dependency - full audio analysis toolkit
    import librosa  # type: ignore
except Exception:  # pragma: no cover - dependency may be absent
    librosa = None  # type: ignore


@dataclasses.dataclass
class PitchCompareConfig:
    """Configuration loaded from JSON for pitch comparison runs."""

    sample_rate: int = 44100
    noise_duration: float = 2.0
    snr_threshold_db: float = 3.0
    min_frequency: float = 55.0
    max_frequency: float = 2000.0
    idle_timeout: float = 1.0
    max_record_seconds: float = 30.0
    analysis_window_sec: float = 0.1
    step_size_sec: Optional[float] = None
    input_mode: str = "mic"
    input_audio_path: Optional[str] = None
    noise_audio_path: Optional[str] = None
    output_directory: str = "data"
    show_plots: bool = True
    crepe_model_capacity: str = "full"
    crepe_step_size_ms: Optional[float] = None
    over_subtraction: float = 1.0
    spoof_factor: float = 2.0

    @staticmethod
    def from_dict(raw: Dict[str, Any]) -> "PitchCompareConfig":
        alias_map = {"spoof factor": "spoof_factor"}
        normalized = {alias_map.get(k, k): v for k, v in raw.items()}
        known = {f.name for f in dataclasses.fields(PitchCompareConfig)}
        filtered = {k: v for k, v in normalized.items() if k in known}
        return PitchCompareConfig(**filtered)


def load_config(path: Path) -> PitchCompareConfig:
    data = json.loads(path.read_text())
    return PitchCompareConfig.from_dict(data)


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def record_noise_sample(cfg: PitchCompareConfig) -> np.ndarray:
    duration_samples = int(cfg.noise_duration * cfg.sample_rate)
    if cfg.input_mode == "file" and cfg.noise_audio_path:
        noise, sr = load_audio(Path(cfg.noise_audio_path), cfg.sample_rate)
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


def load_audio(path: Path, target_sr: int) -> tuple[np.ndarray, int]:
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


def acquire_audio(cfg: PitchCompareConfig, noise_rms: float) -> np.ndarray:
    if cfg.input_mode == "file":
        if cfg.input_audio_path is None:
            raise ValueError(
                "input_audio_path must be provided when input_mode is 'file'"
            )
        audio, _ = load_audio(Path(cfg.input_audio_path), cfg.sample_rate)
        return audio

    hop = determine_hop_length(cfg)
    source = MicSource(cfg.sample_rate, hop)
    source.start()
    print("[INFO] Listening for audio events...")
    snr_threshold = 10 ** (cfg.snr_threshold_db / 20.0)
    collected: list[np.ndarray] = []
    above = False
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


def compute_noise_profile(
    noise: np.ndarray, cfg: PitchCompareConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hop_len = determine_hop_length(cfg)
    win_len = determine_window_length(cfg, hop_len)
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


def determine_hop_length(cfg: PitchCompareConfig) -> int:
    min_step = 10.0 / max(cfg.min_frequency, 1e-6)
    step_sec = cfg.step_size_sec if cfg.step_size_sec is not None else min_step
    step_sec = max(step_sec, min_step)
    return max(int(round(step_sec * cfg.sample_rate)), 1)


def determine_window_length(cfg: PitchCompareConfig, hop_length: int) -> int:
    win_samples = int(round(cfg.analysis_window_sec * cfg.sample_rate))
    win_samples = max(win_samples, hop_length * 2)
    if win_samples % 2 == 1:
        win_samples += 1
    return win_samples


def subtract_noise(
    audio: np.ndarray, noise_profile: np.ndarray, cfg: PitchCompareConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hop_len = determine_hop_length(cfg)
    win_len = determine_window_length(cfg, hop_len)
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


def apply_volume_equalization(audio: np.ndarray, cfg: PitchCompareConfig) -> np.ndarray:
    if audio.size == 0:
        return audio

    sample_rate = float(cfg.sample_rate)
    min_freq = max(float(cfg.min_frequency), 1.0)
    max_freq = max(float(cfg.max_frequency), min_freq)
    cutoff_hz = max(min_freq * 0.25, 1.0)
    cutoff_hz = min(cutoff_hz, max_freq)

    rc = 1.0 / (2.0 * np.pi * cutoff_hz)
    alpha = np.exp(-1.0 / (sample_rate * rc))
    alpha = float(np.clip(alpha, 0.0, 0.9999))

    b = [1.0 - alpha]
    a = [1.0, -alpha]

    squared = np.square(audio, dtype=np.float32)
    envelope_sq, _ = signal.lfilter(b, a, squared, zi=[squared[0]])
    envelope = np.sqrt(envelope_sq + 1e-12).astype(np.float32, copy=False)

    desired_gain = 1.0 / np.clip(envelope, 1e-6, None)
    desired_gain = np.minimum(desired_gain, 10.0)
    smoothed_gain, _ = signal.lfilter(b, a, desired_gain, zi=[desired_gain[0]])

    return audio * smoothed_gain.astype(audio.dtype, copy=False)


def compute_crepe_activation(
    audio: np.ndarray,
    cfg: PitchCompareConfig,
    spoof_factor: Optional[float] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if crepe is None:
        print("[WARN] crepe is not installed; skipping CREPE activation plot.")
        return None
    crepe_sr = (
        cfg.sample_rate if spoof_factor is None else cfg.sample_rate / spoof_factor
    )
    if not np.isfinite(crepe_sr) or crepe_sr <= 0:
        raise ValueError("CREPE sample rate must be positive and finite.")
    if cfg.crepe_step_size_ms is not None:
        step_ms = float(cfg.crepe_step_size_ms)
    else:
        hop_samples = determine_hop_length(cfg)
        step_sec = hop_samples / crepe_sr
        step_ms = step_sec * 1000.0
    min_step_ms = (10.0 / cfg.min_frequency) * 1000.0
    step_ms = float(max(step_ms, min_step_ms))

    normalized_audio = np.asarray(audio, dtype=np.float32)
    normalized_audio = normalized_audio - float(np.mean(normalized_audio))
    normalized_audio = apply_volume_equalization(normalized_audio, cfg)
    rms = float(np.sqrt(np.mean(np.square(normalized_audio))))
    if rms > 0.0:
        normalized_audio = normalized_audio / rms
    else:
        normalized_audio = np.zeros_like(normalized_audio)

    activation = crepe.get_activation(
        normalized_audio,
        int(round(crepe_sr)),
        model_capacity=cfg.crepe_model_capacity,
        center=True,
        step_size=int(round(step_ms)),
        verbose=True,
    )
    if spoof_factor is not None and spoof_factor != 1.0:
        # Because we sampled at a different rate, we need to adjust the
        # activation bins to match the original frequency scale. When we
        # spoofed the sample rate by a factor of spoof_factor, the
        # frequencies were scaled by the same factor. In CREPE's 60-bin
        # per-octave scale, this corresponds to a shift of:
        #   bin_shift = log2(spoof_factor) * 60
        num_bins = activation.shape[1]
        bin_shift = int(round(np.log2(spoof_factor) * 60.0))
        if abs(bin_shift) < num_bins:
            if bin_shift > 0:
                activation = np.pad(
                    activation, ((0, 0), (bin_shift, 0)), mode="constant"
                )[:, :num_bins]
            elif bin_shift < 0:
                activation = np.pad(
                    activation, ((0, 0), (0, -bin_shift)), mode="constant"
                )[:, -bin_shift:]
        else:
            activation = np.zeros_like(activation)

    confidence = activation.max(axis=1)

    cents = crepe.core.to_local_average_cents(activation)

    frequency = 10 * 2 ** (cents / 1200)
    frequency[np.isnan(frequency)] = 0

    crepe_times = np.arange(confidence.shape[0]) * step_ms / 1000.0
    return (
        np.asarray(crepe_times, dtype=np.float32),
        np.asarray(activation, dtype=np.float32),
    )


def _crepe_like_frequency_axis(num_bins: int) -> np.ndarray:
    base_freq = 32.703195662574764  # C1 in Hz; matches CREPE/PESTO documentation
    bins_per_octave = 60.0  # 20 cents per bin
    return base_freq * (2.0 ** (np.arange(num_bins) / bins_per_octave))


def plot_results(
    timestamp: str,
    audio: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray,
    power: np.ndarray,
    crepe_results: List[Tuple[str, Optional[Tuple[np.ndarray, np.ndarray]]]],
    cfg: PitchCompareConfig,
    output_dir: Path,
) -> None:
    fig = plt.figure(figsize=(14, 8))
    grid = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.0])

    ax_wave = fig.add_subplot(grid[0, :])
    t = np.arange(len(audio)) / cfg.sample_rate
    ax_wave.plot(t, audio)
    ax_wave.set_title("Waveform")
    ax_wave.set_xlim(t[0], t[-1] if len(t) else 1.0)
    ax_wave.set_xlabel("Time (s)")
    ax_wave.set_ylabel("Amplitude")

    ax_spec = fig.add_subplot(grid[1, :])
    mesh = ax_spec.pcolormesh(
        times,
        freqs,
        10 * np.log10(power + 1e-12),
        shading="gouraud",
        cmap="magma",
    )
    ax_spec.set_ylim(cfg.min_frequency, cfg.max_frequency)
    ax_spec.set_ylabel("Frequency (Hz)")
    ax_spec.set_title("Spectrogram (Noise-Reduced)")
    fig.colorbar(mesh, ax=ax_spec, label="Power (dB)")

    crepe_axes = [fig.add_subplot(grid[2, 0]), fig.add_subplot(grid[2, 1])]
    for idx, ax in enumerate(crepe_axes):
        label, result = ("", None)
        if idx < len(crepe_results):
            label, result = crepe_results[idx]
        if result is not None:
            crepe_times, crepe_act = result
            freq_axis = _crepe_like_frequency_axis(crepe_act.shape[1])
            mask = (freq_axis >= cfg.min_frequency) & (freq_axis <= cfg.max_frequency)
            if mask.any():
                mesh_crepe = ax.pcolormesh(
                    crepe_times,
                    freq_axis[mask],
                    crepe_act[:, mask].T,
                    shading="nearest",
                    cmap="viridis",
                )
                fig.colorbar(mesh_crepe, ax=ax, label="Activation")
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No activation bins within frequency range.",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
            if crepe_act.size:
                average_activations = crepe_act.mean(axis=0, keepdims=True)
                cents = crepe.core.to_local_average_cents(average_activations)
                frequency = 10 * 2 ** (cents / 1200.0)
                confidence = average_activations.max(axis=1)

                freq_value = float(np.squeeze(frequency))
                conf_value = float(np.squeeze(confidence))

                if not np.isfinite(freq_value):
                    freq_value = 0.0
                if not np.isfinite(conf_value):
                    conf_value = 0.0

                legend_label = (
                    f"Fundamental: {freq_value:.2f} Hz\nConfidence: {conf_value:.3f}"
                )
                dummy_handle = Line2D([], [], color="none")
                ax.legend(
                    [dummy_handle], [legend_label], loc="upper right", frameon=True
                )
        else:
            ax.text(
                0.5,
                0.5,
                "CREPE activation unavailable.",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        ax.set_ylim(cfg.min_frequency, cfg.max_frequency)
        if idx == 0:
            ax.set_ylabel("Frequency (Hz)")
        else:
            ax.set_ylabel("")
        ax.set_xlabel("Time (s)")
        ax.set_title(label or "CREPE Activation")

    fig.tight_layout()
    fig_path = output_dir / f"{timestamp}_comparison.png"
    fig.savefig(fig_path, dpi=150)
    if cfg.show_plots:
        plt.show()
    else:
        plt.close(fig)


def save_audio(
    timestamp: str, audio: np.ndarray, cfg: PitchCompareConfig, output_dir: Path
) -> Path:
    audio_path = output_dir / f"{timestamp}_recording.wav"
    scaled = np.clip(audio, -1.0, 1.0)
    wavfile.write(audio_path, cfg.sample_rate, (scaled * 32767).astype(np.int16))
    return audio_path


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Compare pitch detection methods.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("pitch_compare_config.json"),
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    cfg = load_config(args.config)
    output_dir = Path(cfg.output_directory)
    ensure_output_dir(output_dir)
    timestamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")

    noise = record_noise_sample(cfg)
    noise_rms = float(np.sqrt(np.mean(np.square(noise)) + 1e-12))
    freqs, _, noise_profile = compute_noise_profile(noise, cfg)
    audio = acquire_audio(cfg, noise_rms)
    filtered_audio, freqs, times, power = subtract_noise(audio, noise_profile, cfg)

    audio_path = save_audio(timestamp, filtered_audio, cfg, output_dir)
    print(f"[INFO] Saved filtered audio to {audio_path}")

    crepe_results: List[Tuple[str, Optional[Tuple[np.ndarray, np.ndarray]]]] = []

    crepe_real = compute_crepe_activation(filtered_audio, cfg)
    real_label = f"CREPE Activation ({cfg.sample_rate} Hz Real)"
    crepe_results.append((real_label, crepe_real))

    spoof_factor = cfg.spoof_factor
    if not np.isfinite(spoof_factor) or spoof_factor <= 0:
        print("[WARN] Invalid spoof factor; defaulting to 1.0.")
        spoof_factor = 1.0
    spoof_label = f"CREPE Activation (Spoofed {int(round(cfg.sample_rate / spoof_factor))} Hz; ÷{spoof_factor:g})"
    crepe_spoofed = compute_crepe_activation(
        filtered_audio,
        cfg,
        spoof_factor=spoof_factor,
    )
    crepe_results.append((spoof_label, crepe_spoofed))

    plot_results(
        timestamp=timestamp,
        audio=filtered_audio,
        freqs=freqs,
        times=times,
        power=power,
        crepe_results=crepe_results,
        cfg=cfg,
        output_dir=output_dir,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
