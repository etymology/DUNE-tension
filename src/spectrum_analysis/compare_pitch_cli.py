"""CLI for recording audio and comparing pitch detection methods."""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import inspect
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import wavfile

from .audio import MicSource, sd

try:  # Optional dependency - may not be available in CI
    import soundfile as sf  # type: ignore
except Exception:  # pragma: no cover - soundfile is optional
    sf = None  # type: ignore

try:  # Optional dependency - heavy ML models
    import crepe  # type: ignore
except Exception:  # pragma: no cover - dependency may be absent
    crepe = None  # type: ignore

try:  # Optional dependency - heavy ML models
    from pesto import predict as pesto_predict  # type: ignore
except Exception:  # pragma: no cover - dependency may be absent
    pesto_predict = None  # type: ignore

try:  # Optional dependency - full audio analysis toolkit
    import librosa  # type: ignore
except Exception:  # pragma: no cover - dependency may be absent
    librosa = None  # type: ignore


@dataclasses.dataclass
class PitchCompareConfig:
    """Configuration loaded from JSON for pitch comparison runs."""

    sample_rate: int = 44100
    noise_duration: float = 2.0
    snr_threshold_db: float = 10.0
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
    pesto_model_name: str = "mir-1k_g7"
    pesto_step_size_ms: Optional[float] = None
    over_subtraction: float = 1.0

    @staticmethod
    def from_dict(raw: Dict[str, Any]) -> "PitchCompareConfig":
        known = {f.name for f in dataclasses.fields(PitchCompareConfig)}
        filtered = {k: v for k, v in raw.items() if k in known}
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
        boundary=None,
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
        boundary=None,
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


def compute_crepe_activation(
    audio: np.ndarray, cfg: PitchCompareConfig
) -> Optional[np.ndarray]:
    if crepe is None:
        print("[WARN] crepe is not installed; skipping CREPE activation plot.")
        return None
    step_ms = (
        cfg.crepe_step_size_ms
        if cfg.crepe_step_size_ms is not None
        else determine_hop_length(cfg) * 1000.0 / cfg.sample_rate
    )
    min_step_ms = (10.0 / cfg.min_frequency) * 1000.0
    step_ms = max(step_ms, min_step_ms)
    step_ms = float(step_ms)
    _, _, _, activation = crepe.predict(
        audio,
        cfg.sample_rate,
        model_capacity=cfg.crepe_model_capacity,
        step_size=int(round(step_ms)),
        viterbi=False,
    )
    return np.asarray(activation, dtype=np.float32)


def _crepe_like_frequency_axis(num_bins: int) -> np.ndarray:
    base_freq = 32.703195662574764  # C1 in Hz; matches CREPE/PESTO documentation
    bins_per_octave = 60.0  # 20 cents per bin
    return base_freq * (2.0 ** (np.arange(num_bins) / bins_per_octave))


def compute_pesto_activation(
    audio: np.ndarray, cfg: PitchCompareConfig
) -> Optional[np.ndarray]:
    if pesto_predict is None:
        print("[WARN] pesto is not installed; skipping Pesto activation plot.")
        return None
    step_ms = (
        cfg.pesto_step_size_ms
        if cfg.pesto_step_size_ms is not None
        else determine_hop_length(cfg) * 1000.0 / cfg.sample_rate
    )
    min_step_ms = (10.0 / cfg.min_frequency) * 1000.0
    step_ms = max(step_ms, min_step_ms)
    step_size = max(int(round(step_ms)), 1)

    kwargs: Dict[str, Any] = {
        "step_size": step_size,
        "viterbi": False,
    }

    if cfg.pesto_model_name:
        model_key = "model"
        try:
            signature = inspect.signature(pesto_predict)
        except (TypeError, ValueError):
            signature = None
        if signature is not None:
            params = signature.parameters
            if "model" in params:
                model_key = "model"
            elif "model_capacity" in params:
                model_key = "model_capacity"
            elif "checkpoint" in params:
                model_key = "checkpoint"
        kwargs[model_key] = cfg.pesto_model_name

    audio_buffer = np.asarray(audio, dtype=np.float32)

    try:
        _, _, _, activation = pesto_predict(
            audio_buffer,
            cfg.sample_rate,
            **kwargs,
        )
    except TypeError:
        # Fall back to minimal arguments for older pesto builds.
        fallback_kwargs = {
            k: v for k, v in kwargs.items() if k in {"step_size", "viterbi"}
        }
        _, _, _, activation = pesto_predict(
            audio_buffer,
            cfg.sample_rate,
            **fallback_kwargs,
        )

    activation = np.asarray(activation, dtype=np.float32)

    return activation


def _ensure_even(value: int) -> int:
    return value if value % 2 == 0 else value + 1


def _next_power_of_two(value: int) -> int:
    if value <= 0:
        return 1
    return 1 << (value - 1).bit_length()


def compute_pyin(
    audio: np.ndarray, cfg: PitchCompareConfig
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    if librosa is None:
        print("[WARN] librosa is not installed; skipping PYIN plot.")
        return None

    hop_len = max(determine_hop_length(cfg), 1)
    frame_length = determine_window_length(cfg, hop_len)
    frame_length = max(frame_length, hop_len * 2)
    frame_length = _ensure_even(frame_length)

    def run_pyin(hop: int, frame: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return librosa.pyin(
            audio,
            fmin=cfg.min_frequency,
            fmax=cfg.max_frequency,
            sr=cfg.sample_rate,
            frame_length=frame,
            hop_length=hop,
        )

    try:
        f0, voiced_flag, voiced_prob = run_pyin(hop_len, frame_length)
    except Exception as exc:  # pragma: no cover - depends on librosa internals
        from librosa.util.exceptions import ParameterError

        if not isinstance(exc, ParameterError):
            raise

        adjusted_hop = _ensure_even(_next_power_of_two(hop_len))
        adjusted_frame = _next_power_of_two(max(frame_length, len(audio)))
        adjusted_frame = max(adjusted_frame, adjusted_hop * 2)
        adjusted_frame = _ensure_even(adjusted_frame)

        print(
            "[WARN] librosa.pyin rejected hop/frame lengths; "
            "retrying with power-of-two sizes (hop=%d, frame=%d)."
            % (adjusted_hop, adjusted_frame)
        )

        f0, voiced_flag, voiced_prob = run_pyin(adjusted_hop, adjusted_frame)

        hop_len = adjusted_hop

    times = librosa.times_like(f0, sr=cfg.sample_rate, hop_length=hop_len)
    return times, f0


def plot_results(
    timestamp: str,
    audio: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray,
    power: np.ndarray,
    crepe_act: Optional[np.ndarray],
    pesto_act: Optional[np.ndarray],
    pyin_result: Optional[tuple[np.ndarray, np.ndarray]],
    cfg: PitchCompareConfig,
    output_dir: Path,
) -> None:
    plt.figure(figsize=(12, 10))
    ax_wave = plt.subplot(4, 1, 1)
    t = np.arange(len(audio)) / cfg.sample_rate
    ax_wave.plot(t, audio)
    ax_wave.set_title("Waveform")
    ax_wave.set_xlim(t[0], t[-1] if len(t) else 1.0)
    ax_wave.set_xlabel("Time (s)")
    ax_wave.set_ylabel("Amplitude")

    ax_spec = plt.subplot(4, 1, 2)
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
    plt.colorbar(mesh, ax=ax_spec, label="Power (dB)")

    ax_crepe = plt.subplot(4, 1, 3)
    if crepe_act is not None:
        crepe_times = np.linspace(0, len(audio) / cfg.sample_rate, crepe_act.shape[0])
        freq_axis = _crepe_like_frequency_axis(crepe_act.shape[1])
        mask = (freq_axis >= cfg.min_frequency) & (freq_axis <= cfg.max_frequency)
        if mask.any():
            mesh_crepe = ax_crepe.pcolormesh(
                crepe_times,
                freq_axis[mask],
                crepe_act[:, mask].T,
                shading="nearest",
                cmap="viridis",
            )
            plt.colorbar(mesh_crepe, ax=ax_crepe, label="Activation")
    ax_crepe.set_ylim(cfg.min_frequency, cfg.max_frequency)
    ax_crepe.set_ylabel("Frequency (Hz)")
    ax_crepe.set_title("CREPE Activation")

    ax_pesto = plt.subplot(4, 1, 4)
    if pesto_act is not None:
        pesto_times = np.linspace(0, len(audio) / cfg.sample_rate, pesto_act.shape[0])
        freq_axis = _crepe_like_frequency_axis(pesto_act.shape[1])
        mask = (freq_axis >= cfg.min_frequency) & (freq_axis <= cfg.max_frequency)
        if mask.any():
            mesh_pesto = ax_pesto.pcolormesh(
                pesto_times,
                freq_axis[mask],
                pesto_act[:, mask].T,
                shading="nearest",
                cmap="plasma",
            )
            plt.colorbar(mesh_pesto, ax=ax_pesto, label="Activation")
    ax_pesto.set_ylim(cfg.min_frequency, cfg.max_frequency)
    ax_pesto.set_ylabel("Frequency (Hz)")
    ax_pesto.set_xlabel("Time (s)")
    ax_pesto.set_title("Pesto Activation")

    plt.tight_layout()
    fig_path = output_dir / f"{timestamp}_comparison.png"
    plt.savefig(fig_path, dpi=150)
    if cfg.show_plots:
        plt.show()
    else:
        plt.close()

    if pyin_result is not None:
        times_pyin, f0 = pyin_result
        plt.figure(figsize=(10, 4))
        plt.plot(times_pyin, f0)
        plt.ylim(cfg.min_frequency, cfg.max_frequency)
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.title("librosa.pyin F0 Estimate")
        plt.tight_layout()
        pyin_path = output_dir / f"{timestamp}_pyin.png"
        plt.savefig(pyin_path, dpi=150)
        if cfg.show_plots:
            plt.show()
        else:
            plt.close()


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

    crepe_act = compute_crepe_activation(filtered_audio, cfg)
    pesto_act = compute_pesto_activation(filtered_audio, cfg)
    pyin_result = compute_pyin(filtered_audio, cfg)

    plot_results(
        timestamp=timestamp,
        audio=filtered_audio,
        freqs=freqs,
        times=times,
        power=power,
        crepe_act=crepe_act,
        pesto_act=pesto_act,
        pyin_result=pyin_result,
        cfg=cfg,
        output_dir=output_dir,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
