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
import numpy as np

try:  # Optional dependency: heavy torch/pesto models may be unavailable
    import torch  # type: ignore
except Exception:  # pragma: no cover - dependency may be absent
    torch = None  # type: ignore

try:  # Optional dependency - imported lazily when available
    import pesto  # type: ignore
except Exception:  # pragma: no cover - dependency may be absent
    pesto = None  # type: ignore

from audio_sources import MicSource, sd

from audio_processing import (
    compute_noise_profile,
    compute_spectrogram,
    determine_window_and_hop,
    load_audio,
    subtract_noise,
)
from crepe_analysis import (
    compute_crepe_activation,
    crepe_frequency_axis,
    activations_to_pitch,
    activation_map_to_pitch_track,
)

CREPE_FRAME_TARGET_RMS = 1
CREPE_IDEAL_PITCH = 600.0  # Hz


def _sr_augment_factor(expected_pitch: Optional[float], *, warn: bool = True) -> float:
    if expected_pitch is None:
        return 1.0
    if not np.isfinite(expected_pitch) or expected_pitch <= 0:
        if warn:
            print("[WARN] Invalid expected f0; defaulting augment factor to 1.0.")
        return 1.0
    return CREPE_IDEAL_PITCH / float(expected_pitch)


def get_activations(
    audio: np.ndarray,
    sample_rate: int,
    expected_pitch: Optional[float] = None,
    *,
    cfg: Optional[PitchCompareConfig] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return CREPE frequency axis, frame times, and activation map.

    Parameters
    ----------
    audio:
        Audio samples as a one-dimensional ``np.ndarray``.
    sample_rate:
        Sample rate associated with ``audio``.
    expected_pitch:
        Optional expected fundamental frequency used to derive the CREPE sample
        rate augmentation factor.
    cfg:
        Optional :class:`PitchCompareConfig` providing CREPE configuration
        details. If omitted, a default configuration is used with the provided
        ``sample_rate`` and ``expected_pitch`` values.

    Returns
    -------
    Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]
        ``(times, frequencies, activation)`` where ``times`` has shape ``(T,)``,
        ``frequencies`` has shape ``(F,)`` and ``activation`` has shape
        ``(F, T)``. Returns ``None`` when CREPE activations are unavailable.
    """

    if audio.ndim != 1:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    else:
        audio = np.asarray(audio, dtype=np.float32)

    base_cfg = cfg if cfg is not None else PitchCompareConfig()
    local_cfg = dataclasses.replace(
        base_cfg,
        sample_rate=int(sample_rate),
        expected_f0=expected_pitch,
    )

    sr_factor = _sr_augment_factor(expected_pitch, warn=True)
    crepe_result = compute_crepe_activation(
        audio,
        local_cfg,
        sr_augment_factor=sr_factor,
    )
    if crepe_result is None:
        return None

    times, activation = crepe_result
    freq_axis = crepe_frequency_axis(activation.shape[1])
    activation_ft = activation.T
    return (
        np.asarray(times, dtype=np.float32),
        np.asarray(freq_axis, dtype=np.float32),
        np.asarray(activation_ft, dtype=np.float32),
    )


def get_pesto_activations(
    audio: np.ndarray,
    sample_rate: int,
    *,
    cfg: Optional[PitchCompareConfig] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Return PESTO frame times, frequency axis, activation map, and pitch track.

    Parameters
    ----------
    audio:
        One-dimensional audio samples.
    sample_rate:
        Sampling rate associated with ``audio``.
    cfg:
        Optional :class:`PitchCompareConfig` providing PESTO configuration.

    Returns
    -------
    Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
        Tuple containing frame times in seconds, the frequency axis derived
        from MIDI bins, the activation map transposed to ``(F, T)``, and the
        per-frame pitch estimates in Hertz suitable for overlaying on the
        activation plot. ``None`` when pesto is unavailable or prediction
        fails.
    """

    if pesto is None or torch is None:
        print("[WARN] pesto or torch not available; skipping PESTO activation plot.")
        return None

    active_cfg = cfg if cfg is not None else PitchCompareConfig()
    buffer = np.asarray(audio, dtype=np.float32, order="C").reshape(-1)
    buffer_tensor = torch.from_numpy(buffer).to(dtype=torch.float32)
    if buffer_tensor.ndim == 1:
        buffer_tensor = buffer_tensor.unsqueeze(0)

    window_samples, hop_samples = determine_window_and_hop(active_cfg, buffer.size)
    if active_cfg.pesto_step_size_ms is not None:
        step_ms = float(active_cfg.pesto_step_size_ms)
    else:
        step_ms = hop_samples / float(sample_rate) * 1000.0

    min_overlap = float(np.clip(active_cfg.min_window_overlap, 0.0, 0.999))
    max_step_fraction = max(1.0 - min_overlap, 0.0)
    max_step_ms = max((window_samples * max_step_fraction) / sample_rate * 1000.0, 1.0)
    step_ms = float(np.clip(step_ms, 1.0, max_step_ms))

    model_name = getattr(active_cfg, "pesto_model_name", "mir-1k_g7")

    used_convert_to_freq = True
    try:
        result = pesto.predict(
            buffer_tensor,
            int(sample_rate),
            step_size=float(step_ms),
            model_name=model_name,
            convert_to_freq=True,
        )
    except TypeError:
        used_convert_to_freq = False
        try:
            result = pesto.predict(
                buffer_tensor,
                int(sample_rate),
                step_size=float(step_ms),
                model_name=model_name,
                convert_to_freq=False,
            )
        except Exception as exc:
            print(
                "[WARN] Unable to invoke pesto.predict with the provided arguments:",
                exc,
            )
            return None
    except Exception as exc:  # pragma: no cover - pesto failure is environment-specific
        print("[WARN] pesto.predict failed:", exc)
        return None

    if len(result) < 4:
        print("[WARN] pesto.predict did not return activation data; skipping plot.")
        return None

    times, predicted_freqs, _, activation = result[:4]
    activation_np = _to_numpy(activation)
    if activation_np.ndim == 3 and activation_np.shape[0] == 1:
        activation_np = activation_np[0]
    if activation_np.ndim != 2:
        print("[WARN] Unexpected activation shape from pesto.predict; skipping plot.")
        return None

    num_bins = activation_np.shape[1]
    midi_bins = np.arange(num_bins, dtype=np.float32)
    freq_axis = 440.0 * np.power(2.0, (midi_bins - 69.0) / 12.0)

    times_sec = _to_numpy(times).astype(np.float32, copy=False) * 1e-3
    activation_ft = activation_np.T.astype(np.float32, copy=False)
    overlay_freqs = _to_numpy(predicted_freqs).astype(np.float32, copy=False)
    if overlay_freqs.size and not used_convert_to_freq:
        overlay_freqs = 440.0 * np.power(2.0, (overlay_freqs - 69.0) / 12.0)
    return (
        times_sec,
        freq_axis.astype(np.float32, copy=False),
        activation_ft,
        overlay_freqs,
    )


def _to_numpy(value: Any) -> np.ndarray:
    if torch is not None and isinstance(value, torch.Tensor):  # type: ignore[union-attr]
        return value.detach().cpu().numpy()
    return np.asarray(value)


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
    over_subtraction: float = 1.0  # Noise reduction factor
    expected_f0: Optional[float] = (
        None  # Expected fundamental frequency for CREPE scaling
    )
    crepe_activation_coverage: float = 0.9
    pesto_model_name: str = "mir-1k_g7"
    pesto_step_size_ms: Optional[float] = None

    @staticmethod
    def from_dict(raw: Dict[str, Any]) -> "PitchCompareConfig":
        normalized = dict(raw)

        if "expected_f0" not in normalized and "sr_augment_factor" in normalized:
            try:
                sr_factor = float(normalized["sr_augment_factor"])
            except (TypeError, ValueError):
                sr_factor = float("nan")
            if np.isfinite(sr_factor) and sr_factor > 0:
                normalized["expected_f0"] = 1000.0 / sr_factor

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


ActivationResult = Tuple[np.ndarray, np.ndarray, np.ndarray]
ActivationPlotEntry = Tuple[str, Optional[ActivationResult], Optional[np.ndarray]]


def _crepe_pitch_overlay(result: ActivationResult) -> Optional[np.ndarray]:
    """Convert CREPE activations into a per-frame pitch contour."""

    _, freq_axis, activation = result
    try:
        overlay = activation_map_to_pitch_track(activation.T, freq_axis)
    except ValueError as exc:
        print(
            "[WARN] Unable to convert CREPE activations to pitch contour:",
            exc,
        )
        return None
    return overlay.astype(np.float32, copy=False)


def plot_results(
    timestamp: str,
    audio: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray,
    power: Optional[np.ndarray],
    activation_results: List[ActivationPlotEntry],
    cfg: PitchCompareConfig,
    output_dir: Path,
) -> None:
    num_activation_plots = max(1, len(activation_results))
    num_columns = max(2, num_activation_plots)

    fig = plt.figure(figsize=(16, 8))
    grid = fig.add_gridspec(3, num_columns, height_ratios=[1.0, 1.0, 1.0])

    _add_waveform_plot(fig, grid[0, :], audio, cfg)
    _add_spectrogram_plot(fig, grid[1, :], freqs, times, power, cfg)
    _populate_activation_axes(fig, grid, 2, activation_results, cfg)

    fig.tight_layout(rect=(0, 0, 0.92, 1))
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
    power: Optional[np.ndarray],
    cfg: PitchCompareConfig,
) -> Axes:
    ax = fig.add_subplot(location)
    if power is None:
        ax.text(
            0.5,
            0.5,
            "Spectrogram unavailable.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    else:
        mesh = ax.pcolormesh(
            times,
            freqs,
            10 * np.log10(np.asarray(power, dtype=float) + 1e-12),
            shading="gouraud",
            cmap="magma",
        )
        fig.colorbar(mesh, ax=ax, label="Power (dB)")
    ax.set_ylim(cfg.min_frequency, cfg.max_frequency)
    ax.set_ylabel("Frequency (Hz)")
    if cfg.input_mode == "file":
        title = "Spectrogram"
    else:
        title = "Spectrogram (Noise-Reduced)"
    ax.set_title(title)
    return ax


def _populate_activation_axes(
    fig: Figure,
    grid: GridSpec,
    row: int,
    activation_results: List[ActivationPlotEntry],
    cfg: PitchCompareConfig,
) -> None:
    ncols = getattr(grid, "ncols", len(activation_results) or 1)
    axes = [fig.add_subplot(grid[row, col]) for col in range(ncols)]
    for idx, ax in enumerate(axes):
        label: str
        result: Optional[ActivationResult]
        overlay: Optional[np.ndarray]
        label, result, overlay = ("", None, None)
        if idx < len(activation_results):
            label, result, overlay = activation_results[idx]
        _render_activation_axis(fig, ax, label, result, overlay, cfg)


def _render_activation_axis(
    fig: Figure,
    ax: Axes,
    label: str,
    result: Optional[ActivationResult],
    overlay: Optional[np.ndarray],
    cfg: PitchCompareConfig,
) -> None:
    x_limits: Optional[Tuple[float, float]] = None
    y_limits: Optional[Tuple[float, float]] = None
    if result is not None:
        frame_times, freq_axis, activation = result
        mask = (freq_axis >= cfg.min_frequency) & (freq_axis <= cfg.max_frequency)
        if mask.any():
            coverage = getattr(cfg, "crepe_activation_coverage", 0.9)
            if not np.isfinite(coverage):
                coverage = 1.0
            coverage = float(np.clip(coverage, 0.0, 1.0))
            mesh = ax.pcolormesh(
                frame_times,
                freq_axis[mask],
                activation[mask, :],
                shading="nearest",
                cmap="viridis",
            )
            fig.colorbar(mesh, ax=ax, label="Activation")

            masked_activation = activation.T[:, mask]
            freq_values = freq_axis[mask]
            limits = _compute_crepe_crop_limits(
                frame_times,
                freq_values,
                masked_activation,
                coverage,
                cfg.min_frequency,
                cfg.max_frequency,
            )
            if limits is not None:
                x_limits, y_limits = limits
        else:
            ax.text(
                0.5,
                0.5,
                "No activation bins within frequency range.",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

        if x_limits is None and frame_times.size:
            x_limits = (frame_times[0], frame_times[-1])
        if y_limits is None:
            y_limits = (cfg.min_frequency, cfg.max_frequency)

        if overlay is not None and frame_times.size:
            overlay_values = np.ravel(_to_numpy(overlay)).astype(np.float32, copy=False)
            min_len = min(frame_times.size, overlay_values.size)
            if min_len:
                frame_subset = frame_times[:min_len]
                overlay_subset = overlay_values[:min_len]
                overlay_mask = np.isfinite(overlay_subset)
                overlay_mask &= overlay_subset > 0
                if overlay_mask.any():
                    ax.plot(
                        frame_subset[overlay_mask],
                        overlay_subset[overlay_mask],
                        color="magenta",
                        linewidth=1.5,
                        label="Pitch Track",
                    )
                    ax.legend(loc="upper right", frameon=False)

        legend_label = _activation_summary_label(frame_times, freq_axis, activation.T)
        ax.text(
            1.02,
            0.0,
            legend_label,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            bbox={
                "boxstyle": "round,pad=0.4",
                "facecolor": "white",
                "edgecolor": "lightgray",
                "alpha": 0.9,
            },
        )
    else:
        ax.text(
            0.5,
            0.5,
            "Activation unavailable.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        y_limits = (cfg.min_frequency, cfg.max_frequency)

    if x_limits is not None:
        ax.set_xlim(*x_limits)
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (s)")
    ax.set_title(label or "Activation Map")


def _compute_crepe_crop_limits(
    times: np.ndarray,
    freqs: np.ndarray,
    activation: np.ndarray,
    coverage: float,
    min_frequency: float,
    max_frequency: float,
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    if activation.size == 0 or times.size == 0 or freqs.size == 0:
        return None

    coverage = float(np.clip(coverage, 0.0, 1.0))
    positive_activation = np.where(activation > 0.0, activation, 0.0)
    total_activation = float(positive_activation.sum())
    if total_activation <= 0.0 or coverage <= 0.0:
        return None

    min_time = float(times[0])
    max_time = float(times[-1]) if times.size else min_time
    min_freq = float(min_frequency)
    max_freq = float(max_frequency)

    if coverage >= 1.0:
        return (min_time, max_time), (min_freq, max_freq)

    cumulative = np.cumsum(np.cumsum(positive_activation, axis=0), axis=1)
    target = coverage * total_activation
    coverage_mask = cumulative >= target
    if not coverage_mask.any():
        return None

    candidate_indices = np.argwhere(coverage_mask)
    areas = (candidate_indices[:, 0] + 1) * (candidate_indices[:, 1] + 1)
    min_area = np.min(areas)
    smallest = candidate_indices[areas == min_area]
    order = np.lexsort((smallest[:, 1], smallest[:, 0]))
    best_idx = smallest[order[0]]
    time_idx = int(best_idx[0])
    freq_idx = int(best_idx[1])

    if times.size == 1:
        time_limit = float(times[0])
    else:
        time_limit = float(times[min(time_idx + 1, times.size - 1)])

    if freqs.size == 1:
        freq_limit = float(freqs[0])
    else:
        freq_limit = float(freqs[min(freq_idx + 1, freqs.size - 1)])

    time_limit = float(np.clip(time_limit, min_time, max_time))
    freq_limit = float(np.clip(freq_limit, min_freq, max_freq))

    if time_limit <= min_time and times.size > 1:
        time_limit = float(times[1])
    if freq_limit <= min_freq and freqs.size > 1:
        freq_limit = float(freqs[1])

    return (min_time, time_limit), (min_freq, freq_limit)


def _activation_summary_label(
    times: np.ndarray, freq_axis: np.ndarray, activation: np.ndarray
) -> str:
    freq_value, conf_value = activations_to_pitch(activation, times, freq_axis)
    if not np.isfinite(freq_value) or not np.isfinite(conf_value):
        return "Fundamental: N/A\nConfidence: N/A"
    return f"Fundamental: {freq_value:.2f} Hz\nConfidence: {conf_value:.3f}"


def find_activation_clusters(
    freqs: np.ndarray,
    times: np.ndarray,
    activation: np.ndarray,
) -> list[tuple[float, float]]:
    """Identify activation clusters and rank them by volume.

    Parameters
    ----------
    freqs:
        Frequency bin centers with shape ``(F,)``.
    times:
        Time bin centers with shape ``(T,)``.
    activation:
        Activation magnitudes with shape ``(F, T)``.

    Returns
    -------
    list[tuple[float, float]]
        A list of ``(average_frequency, volume)`` pairs sorted by descending
        volume. The ``average_frequency`` is the activation-weighted time
        average of the frequency for the cluster and ``volume`` is calculated as
        the integral of the activation over time using the frame durations.
    """

    freqs = np.asarray(freqs, dtype=float)
    times = np.asarray(times, dtype=float)
    activation = np.asarray(activation, dtype=float)

    if freqs.ndim != 1 or times.ndim != 1:
        raise ValueError("`freqs` and `times` must be one-dimensional arrays.")

    if activation.shape != (freqs.size, times.size):
        raise ValueError(
            "`activation` must have shape (freqs.size, times.size).",
        )

    if activation.size == 0:
        return []

    if times.size == 1:
        frame_durations = np.array([1.0], dtype=float)
    else:
        frame_steps = np.diff(times)
        if np.any(frame_steps <= 0.0):
            raise ValueError("`times` must be strictly increasing.")
        frame_durations = np.concatenate([frame_steps, frame_steps[-1:]])
        frame_durations = frame_durations.astype(float, copy=False)

    positive_mask = activation > 0.0
    if not positive_mask.any():
        return []

    visited = np.zeros_like(positive_mask, dtype=bool)
    clusters: list[tuple[float, float]] = []

    freq_values = freqs.reshape(-1, 1)

    for freq_idx in range(freqs.size):
        for time_idx in range(times.size):
            if not positive_mask[freq_idx, time_idx] or visited[freq_idx, time_idx]:
                continue

            stack = [(freq_idx, time_idx)]
            visited[freq_idx, time_idx] = True

            volume = 0.0
            freq_weight = 0.0

            while stack:
                f_idx, t_idx = stack.pop()
                weight = activation[f_idx, t_idx] * frame_durations[t_idx]
                volume += weight
                freq_weight += freq_values[f_idx, 0] * weight

                for n_f, n_t in (
                    (f_idx - 1, t_idx),
                    (f_idx + 1, t_idx),
                    (f_idx, t_idx - 1),
                    (f_idx, t_idx + 1),
                ):
                    if (
                        0 <= n_f < freqs.size
                        and 0 <= n_t < times.size
                        and positive_mask[n_f, n_t]
                        and not visited[n_f, n_t]
                    ):
                        visited[n_f, n_t] = True
                        stack.append((n_f, n_t))

            if volume > 0.0:
                clusters.append((freq_weight / volume, volume))

    clusters.sort(key=lambda item: item[1], reverse=True)
    return clusters


def save_audio(
    timestamp: str, audio: np.ndarray, cfg: PitchCompareConfig, output_dir: Path
) -> Path:
    audio_path = output_dir / f"{timestamp}_recording.wav"
    scaled = np.clip(audio, -1.0, 1.0)
    wav_data = (scaled * 32767).astype(np.int16)
    from scipy.io import wavfile

    wavfile.write(audio_path, cfg.sample_rate, wav_data)
    return audio_path


def _run_comparison(cfg: PitchCompareConfig, output_dir: Path) -> None:
    timestamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")

    is_file_input = cfg.input_mode == "file"

    if is_file_input:
        audio = acquire_audio(cfg, 0.0)
        filtered_audio = audio
        freqs, times, power = compute_spectrogram(filtered_audio, cfg)
    else:
        noise = record_noise_sample(cfg)
        noise_rms = float(np.sqrt(np.mean(np.square(noise)) + 1e-12))
        noise_profile = compute_noise_profile(noise, cfg)
        audio = acquire_audio(cfg, noise_rms)
        filtered_audio, freqs, times, power = subtract_noise(audio, noise_profile, cfg)

        audio_path = save_audio(timestamp, filtered_audio, cfg, output_dir)
        print(f"[INFO] Saved filtered audio to {audio_path}")

    activation_results: List[ActivationPlotEntry] = []

    crepe_real = get_activations(
        filtered_audio,
        cfg.sample_rate,
        expected_pitch=None,
        cfg=cfg,
    )
    real_label = f"CREPE Activation ({cfg.sample_rate} Hz Real)"
    real_overlay = _crepe_pitch_overlay(crepe_real) if crepe_real is not None else None
    activation_results.append((real_label, crepe_real, real_overlay))

    expected_f0 = cfg.expected_f0
    sr_augment_factor = _sr_augment_factor(expected_f0, warn=False)
    augmented_sr = (
        int(round(cfg.sample_rate * sr_augment_factor))
        if sr_augment_factor
        else cfg.sample_rate
    )
    sr_augmented_label = (
        f"CREPE Activation (augmented sr {augmented_sr} Hz; x{sr_augment_factor:g})"
    )
    crepe_scaled = get_activations(
        filtered_audio,
        cfg.sample_rate,
        expected_pitch=expected_f0,
        cfg=cfg,
    )
    scaled_overlay = (
        _crepe_pitch_overlay(crepe_scaled) if crepe_scaled is not None else None
    )
    activation_results.append((sr_augmented_label, crepe_scaled, scaled_overlay))

    pesto_label = f"PESTO Activation ({cfg.pesto_model_name})"
    pesto_result = get_pesto_activations(filtered_audio, cfg.sample_rate, cfg=cfg)
    if pesto_result is not None:
        times_sec, freq_axis, activation_ft, overlay = pesto_result
        activation_results.append(
            (pesto_label, (times_sec, freq_axis, activation_ft), overlay)
        )
    else:
        activation_results.append((pesto_label, None, None))

    plot_results(
        timestamp=timestamp,
        audio=filtered_audio,
        freqs=freqs,
        times=times,
        power=power,
        activation_results=activation_results,
        cfg=cfg,
        output_dir=output_dir,
    )


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

    if cfg.input_mode == "file" and cfg.input_audio_path is not None:
        input_path = Path(cfg.input_audio_path)
        if input_path.is_dir():
            wav_files = sorted(
                path
                for path in input_path.iterdir()
                if path.is_file() and path.suffix.lower() == ".wav"
            )
            if not wav_files:
                raise ValueError(
                    f"No .wav files found in directory: {input_path}"  # noqa: TRY003
                )

            for wav_file in wav_files:
                print(f"[INFO] Processing {wav_file}")
                _run_comparison(
                    dataclasses.replace(cfg, input_audio_path=str(wav_file)),
                    output_dir,
                )
            return

    if cfg.input_mode == "mic":
        print(
            "[INFO] Microphone mode active. Close the plot window to capture a new "
            "sample or press Ctrl+C to exit."
        )
        try:
            while True:
                _run_comparison(cfg, output_dir)
        except KeyboardInterrupt:
            print("\n[INFO] Exiting microphone capture loop.")
    else:
        _run_comparison(cfg, output_dir)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
