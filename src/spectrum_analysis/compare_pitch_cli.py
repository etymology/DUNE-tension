"""CLI for recording audio and comparing pitch detection methods."""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

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

from .audio_processing import (
    acquire_audio,
    compute_noise_profile,
    compute_spectrogram,
    determine_window_and_hop,
    load_audio,
    load_noise_profile,
    record_noise_sample,
    save_noise_profile,
    subtract_noise,
)
from .crepe_analysis import (
    activations_to_pitch,
    activation_map_to_pitch_track,
    get_activations,
)
from .pitch_compare_config import PitchCompareConfig, ensure_output_dir, load_config

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
    if coverage <= 0.0:
        return None

    positive_activation = np.where(activation > 0.0, activation, 0.0)
    total_activation = float(positive_activation.sum())
    if total_activation <= 0.0:
        return None

    min_time = float(times[0])
    max_time = float(times[-1]) if times.size else min_time
    min_freq = float(min_frequency)
    max_freq = float(max_frequency)

    if coverage >= 1.0:
        return (min_time, max_time), (min_freq, max_freq)

    freq_activation = positive_activation.sum(axis=0)
    if freq_activation.size == 0:
        return None

    target = coverage * total_activation
    best_start = 0
    best_end = freq_activation.size - 1
    best_width = float("inf")
    best_sum = 0.0
    current_sum = 0.0
    start = 0

    for end in range(freq_activation.size):
        current_sum += float(freq_activation[end])
        while start <= end and current_sum - float(freq_activation[start]) >= target:
            current_sum -= float(freq_activation[start])
            start += 1

        if current_sum >= target:
            lower_freq = float(freqs[start])
            upper_freq = float(freqs[end])
            width = upper_freq - lower_freq
            window_sum = current_sum
            if width < best_width or (
                np.isclose(width, best_width) and window_sum > best_sum
            ):
                best_width = width
                best_start = start
                best_end = end
                best_sum = window_sum

    if not np.isfinite(best_width):
        return (min_time, max_time), (min_freq, max_freq)

    lower_freq = float(freqs[best_start])
    upper_freq = float(freqs[best_end])

    lower_freq = float(np.clip(lower_freq, min_freq, max_freq))
    upper_freq = float(np.clip(upper_freq, min_freq, max_freq))

    if lower_freq == upper_freq:
        if best_start > 0:
            lower_freq = float(freqs[best_start - 1])
        elif best_end + 1 < freqs.size:
            upper_freq = float(freqs[best_end + 1])

    lower_freq = float(np.clip(lower_freq, min_freq, max_freq)) * 0.95
    upper_freq = float(np.clip(upper_freq, min_freq, max_freq)) * 1.05

    if upper_freq <= lower_freq:
        return (min_time, max_time), (min_freq, max_freq)

    return (min_time, max_time), (lower_freq, upper_freq)


def _activation_summary_label(
    times: np.ndarray, freq_axis: np.ndarray, activation: np.ndarray
) -> str:
    freq_value, conf_value = activations_to_pitch(activation, times, freq_axis)
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


def _run_comparison(cfg: PitchCompareConfig, output_dir: Path) -> None:
    timestamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")

    is_file_input = cfg.input_mode == "file"

    if is_file_input or cfg.noise_duration <= 0.0:
        audio = acquire_audio(cfg, 0.0)
        filtered_audio = audio
        freqs, times, power = compute_spectrogram(filtered_audio, cfg)
    else:
        duration_samples = int(cfg.noise_duration * cfg.sample_rate)
        win_len, hop_len = determine_window_and_hop(cfg, duration_samples)
        cache_dir = Path(cfg.output_directory) / "noise_filters"
        cache_name = (
            f"stationary_noise_sr{cfg.sample_rate}_n{duration_samples}"
            f"_win{win_len}_hop{hop_len}.npz"
        )
        cache_path = cache_dir / cache_name

        noise_profile = load_noise_profile(
            cache_path,
            cfg,
            expected_window=win_len,
            expected_hop=hop_len,
        )
        if noise_profile is None:
            noise = record_noise_sample(cfg)
            noise_profile = compute_noise_profile(noise, cfg)
            save_noise_profile(noise_profile, cache_path, cfg.sample_rate)

        audio = acquire_audio(cfg, noise_profile.rms)
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
    real_overlay = (
        _crepe_pitch_overlay(crepe_real)
        if cfg.show_pitch_overlay and crepe_real is not None
        else None
    )
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
        _crepe_pitch_overlay(crepe_scaled)
        if cfg.show_pitch_overlay and crepe_scaled is not None
        else None
    )
    activation_results.append((sr_augmented_label, crepe_scaled, scaled_overlay))

    pesto_label = f"PESTO Activation ({cfg.pesto_model_name})"
    pesto_result = get_pesto_activations(filtered_audio, cfg.sample_rate, cfg=cfg)
    if pesto_result is not None:
        times_sec, freq_axis, activation_ft, overlay = pesto_result
        if not cfg.show_pitch_overlay:
            overlay = None
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
    parser.add_argument(
        "--pitch-overlay",
        dest="pitch_overlay",
        action="store_true",
        default=None,
        help="Force-enable pitch contour overlays regardless of configuration.",
    )
    parser.add_argument(
        "--no-pitch-overlay",
        dest="pitch_overlay",
        action="store_false",
        default=None,
        help="Disable pitch contour overlays on activation plots.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    cfg = load_config(args.config)
    if args.pitch_overlay is not None:
        cfg = dataclasses.replace(cfg, show_pitch_overlay=args.pitch_overlay)
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
