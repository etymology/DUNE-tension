"""CLI for recording audio and comparing pitch detection methods."""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import numpy as np

from audio_sources import MicSource, sd

from .audio_processing import (
    compute_noise_profile,
    determine_window_and_hop,
    load_audio,
    subtract_noise,
)
from .crepe_analysis import (
    compute_crepe_activation,
    crepe_frequency_axis,
    summarize_activation,
)


@dataclasses.dataclass
class PitchCompareConfig:
    """Configuration loaded from JSON for pitch comparison runs."""

    sample_rate: int = 44100
    noise_duration: float = 2.0
    snr_threshold_db: float = 3.0
    min_frequency: float = 55.0
    max_frequency: float = 2000.0
    min_oscillations_per_window: float = 20.0
    min_window_overlap: float = 0.8
    idle_timeout: float = 1.0
    max_record_seconds: float = 30.0
    input_mode: str = "mic"
    input_audio_path: Optional[str] = None
    noise_audio_path: Optional[str] = None
    output_directory: str = "data"
    show_plots: bool = True
    crepe_model_capacity: str = "full"
    crepe_step_size_ms: Optional[float] = None
    over_subtraction: float = 1.0
    spoof_factor: float = 0.5

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


def acquire_audio(cfg: PitchCompareConfig, noise_rms: float) -> np.ndarray:
    if cfg.input_mode == "file":
        if cfg.input_audio_path is None:
            raise ValueError(
                "input_audio_path must be provided when input_mode is 'file'"
            )
        audio, _ = load_audio(Path(cfg.input_audio_path), cfg.sample_rate)
        return audio

    _, hop = determine_window_and_hop(cfg)
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

    _add_waveform_plot(fig, grid[0, :], audio, cfg)
    _add_spectrogram_plot(fig, grid[1, :], freqs, times, power, cfg)
    _populate_crepe_axes(fig, grid, 2, crepe_results, cfg)

    fig.tight_layout()
    fig_path = output_dir / f"{timestamp}_comparison.png"
    fig.savefig(fig_path, dpi=150)
    if cfg.show_plots:
        plt.show()
    else:
        plt.close(fig)


def _add_waveform_plot(
    fig: Figure, location: slice, audio: np.ndarray, cfg: PitchCompareConfig
) -> Axes:
    ax = fig.add_subplot(location)
    t = np.arange(len(audio)) / cfg.sample_rate if len(audio) else np.array([0.0])
    ax.plot(t, audio)
    ax.set_title("Waveform")
    ax.set_xlim(t[0], t[-1] if len(t) else 1.0)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    return ax


def _add_spectrogram_plot(
    fig: Figure,
    location: slice,
    freqs: np.ndarray,
    times: np.ndarray,
    power: np.ndarray,
    cfg: PitchCompareConfig,
) -> Axes:
    ax = fig.add_subplot(location)
    mesh = ax.pcolormesh(
        times, freqs, 10 * np.log10(power + 1e-12), shading="gouraud", cmap="magma"
    )
    ax.set_ylim(cfg.min_frequency, cfg.max_frequency)
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Spectrogram (Noise-Reduced)")
    fig.colorbar(mesh, ax=ax, label="Power (dB)")
    return ax


def _populate_crepe_axes(
    fig: Figure,
    grid: GridSpec,
    row: int,
    crepe_results: List[Tuple[str, Optional[Tuple[np.ndarray, np.ndarray]]]],
    cfg: PitchCompareConfig,
) -> None:
    axes = [fig.add_subplot(grid[row, col]) for col in range(2)]
    for idx, ax in enumerate(axes):
        label, result = ("", None)
        if idx < len(crepe_results):
            label, result = crepe_results[idx]
        _render_crepe_axis(fig, ax, label, result, cfg)


def _render_crepe_axis(
    fig: Figure,
    ax: Axes,
    label: str,
    result: Optional[Tuple[np.ndarray, np.ndarray]],
    cfg: PitchCompareConfig,
) -> None:
    if result is not None:
        crepe_times, crepe_act = result
        freq_axis = crepe_frequency_axis(crepe_act.shape[1])
        mask = (freq_axis >= cfg.min_frequency) & (freq_axis <= cfg.max_frequency)
        if mask.any():
            mesh = ax.pcolormesh(
                crepe_times,
                freq_axis[mask],
                crepe_act[:, mask].T,
                shading="nearest",
                cmap="viridis",
            )
            fig.colorbar(mesh, ax=ax, label="Activation")
        else:
            ax.text(
                0.5,
                0.5,
                "No activation bins within frequency range.",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

        legend_label = _activation_summary_label(crepe_act)
        dummy_handle = Line2D([], [], color="none")
        ax.legend([dummy_handle], [legend_label], loc="upper right", frameon=True)
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
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (s)")
    ax.set_title(label or "CREPE Activation")


def _activation_summary_label(activation: np.ndarray) -> str:
    freq_value, conf_value = summarize_activation(activation)
    if not np.isfinite(freq_value) or not np.isfinite(conf_value):
        return "Fundamental: N/A\nConfidence: N/A"
    return f"Fundamental: {freq_value:.2f} Hz\nConfidence: {conf_value:.3f}"


def save_audio(
    timestamp: str, audio: np.ndarray, cfg: PitchCompareConfig, output_dir: Path
) -> Path:
    audio_path = output_dir / f"{timestamp}_recording.wav"
    scaled = np.clip(audio, -1.0, 1.0)
    wav_data = (scaled * 32767).astype(np.int16)
    from scipy.io import wavfile

    wavfile.write(audio_path, cfg.sample_rate, wav_data)
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
    _, _, noise_profile = compute_noise_profile(noise, cfg)
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
    spoof_sr = (
        int(round(cfg.sample_rate / spoof_factor)) if spoof_factor else cfg.sample_rate
    )
    spoof_label = f"CREPE Activation (Spoofed {spoof_sr} Hz; ÷{spoof_factor:g})"
    crepe_spoofed = compute_crepe_activation(
        filtered_audio, cfg, spoof_factor=spoof_factor
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
