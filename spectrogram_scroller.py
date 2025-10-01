#!/usr/bin/env python3
"""
Scrolling Spectrogram Visualizer (now with PESTO pitch mode)

What's new
----------
- **--pesto**: Use the PESTO model (pip: `pesto-pitch`) to estimate pitch in real time.
  Instead of a spectrogram, show a scrolling **pitch track** where point color encodes
  **confidence** (0..1).
- **--pesto-step-ms**: step size (ms) for PESTO streaming inference (default 10 ms).
- Pitch mode coexists with the original spectrogram/noise-subtraction mode.

Install
-------
pip install numpy matplotlib sounddevice
pip install torch torchaudio pesto-pitch

Notes
-----
- PESTO mode requires PyTorch + torchaudio.
- For live audio, ensure your input device is selected correctly (Linux: `pavucontrol`).

Keyboard
--------
Common:
  p : pause/resume
  q / ESC : quit
  , / . : shrink/grow time window (both modes)
Spectrogram-only:
  [ / ] : decrease/increase dB range
  - / = : dec/inc max freq
  n : toggle noise filter
  r : re-profile noise

CLI Examples
------------
# PESTO pitch mode, 44.1 kHz, 10 ms step
python3 spectrogram_scroller.py --pesto

# PESTO on your USB mic
python3 spectrogram_scroller.py --pesto --device "USB PnP Sound Device"

# Classic spectrogram with noise profiling
python3 spectrogram_scroller.py --noise-sec 3

"""

import argparse
import sys
import time
import queue
import threading
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import matplotlib.pyplot as plt

# Try to import sounddevice. In demo mode or if missing, we can run without it.
try:
    import sounddevice as sd
except Exception:
    sd = None

# Optional PESTO imports are loaded lazily only if --pesto is set
torch = None
pesto = None
crepe = None

EPS = 1e-12

HERE = Path(__file__).resolve().parent
SRC_ROOT = HERE / "src"
NOISE_FILTER_DIR = HERE / "data" / "noise_filters"
SPECTRUM_SRC = SRC_ROOT / "spectrum_analysis"
for path in (SRC_ROOT, SPECTRUM_SRC):
    if path.exists():
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

SHARED_NOISE_TOOLS_AVAILABLE = True
try:
    from spectrum_analysis.audio_processing import (
        NoiseProfile as SharedNoiseProfile,
        compute_noise_profile as shared_compute_noise_profile,
        save_noise_profile as shared_save_noise_profile,
        wiener_filter_signal as shared_wiener_filter,
    )
    from spectrum_analysis.pitch_compare_config import PitchCompareConfig
except ImportError:
    SHARED_NOISE_TOOLS_AVAILABLE = False
    SharedNoiseProfile = None  # type: ignore[assignment]
    PitchCompareConfig = None  # type: ignore[assignment]
    shared_compute_noise_profile = None  # type: ignore[assignment]
    shared_save_noise_profile = None  # type: ignore[assignment]
    shared_wiener_filter = None  # type: ignore[assignment]


def dbfs(x: np.ndarray) -> np.ndarray:
    """Convert magnitude to dBFS (0 dB == amplitude 1.0)."""
    return 20.0 * np.log10(np.maximum(x, EPS))


def hann_window(n: int) -> np.ndarray:
    """Periodic Hann window suitable for STFT (sine window also works)."""
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)


class AudioSource:
    """Abstract audio source. Implement .start(), .read() -> np.ndarray, .stop()."""

    def start(self):
        raise NotImplementedError

    def read(self) -> np.ndarray:
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError


class MicSource(AudioSource):
    def __init__(self, samplerate: int, hop: int, device=None):
        if sd is None:
            raise RuntimeError(
                "sounddevice is not available. Install it or use --demo."
            )
        self.samplerate = samplerate
        self.hop = hop
        self.device = device
        self.q = queue.Queue(maxsize=64)
        self.stream = None
        self._stopped = threading.Event()

    def _callback(self, indata, frames, time_info, status):
        if status:
            # Avoid printing too often; you can log if desired
            pass
        # Mono: average channels if any
        if indata.ndim == 2 and indata.shape[1] > 1:
            mono = indata.mean(axis=1).copy()
        else:
            mono = indata[:, 0].copy() if indata.ndim == 2 else indata.copy()
        try:
            self.q.put_nowait(mono)
        except queue.Full:
            # Drop if we're falling behind
            pass

    def start(self):
        self._stopped.clear()
        self.stream = sd.InputStream(
            channels=1,
            samplerate=self.samplerate,
            blocksize=self.hop,
            device=self.device,
            callback=self._callback,
            dtype="float32",
        )
        self.stream.start()

    def read(self) -> np.ndarray:
        while not self._stopped.is_set():
            try:
                return self.q.get(timeout=0.1)
            except queue.Empty:
                continue
        return np.array([], dtype=np.float32)

    def stop(self):
        self._stopped.set()
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass


class DemoSource(AudioSource):
    """Generates a synthetic test signal: chirp + tones + noise."""

    def __init__(self, samplerate: int, hop: int):
        self.samplerate = samplerate
        self.hop = hop
        self.t = 0

    def start(self):
        pass

    def read(self) -> np.ndarray:
        n = self.hop
        sr = self.samplerate
        t = (self.t + np.arange(n)) / sr
        # Components: sweeping chirp, fixed tones, and some noise
        chirp = np.sin(2 * np.pi * (100 + (t * 0.5e3)) * t) * 0.25
        tone1 = 0.2 * np.sin(2 * np.pi * 440 * t)
        tone2 = 0.15 * np.sin(2 * np.pi * 880 * t + 0.3)
        noise = 0.02 * np.random.randn(n)
        y = chirp + tone1 + tone2 + noise
        self.t += n
        # light limiter to avoid clipping > 1.0
        y = np.tanh(1.5 * y)
        return y.astype(np.float32)

    def stop(self):
        pass


# ===================== Spectrogram implementation (unchanged, trimmed) =====================
class SpectrogramView:
    def __init__(
        self,
        samplerate: int,
        fft_size: int,
        hop: int,
        window_sec: float,
        max_freq: float,
        db_range: float,
    ):
        self.sr = samplerate
        self.fft_size = int(fft_size)
        self.hop = int(hop)
        self.window_sec = float(window_sec)
        self.max_freq = float(max_freq)
        self.db_range = float(db_range)

        self.win = hann_window(self.fft_size).astype(np.float32)
        self.freqs = np.fft.rfftfreq(self.fft_size, d=1.0 / self.sr)
        self.max_bin = np.searchsorted(self.freqs, self.max_freq, side="right")
        if self.max_bin < 2:
            self.max_bin = len(self.freqs)

        self.n_cols = max(10, int(round(self.window_sec * self.sr / self.hop)))
        self.S = np.full((self.max_bin, self.n_cols), -120.0, dtype=np.float32)

        self.sample_buf = np.zeros(0, dtype=np.float32)

        self.fig, self.ax = plt.subplots(figsize=(10, 5))
        extent = (-self.window_sec, 0.0, 0.0, self.freqs[self.max_bin - 1])
        self.im = self.ax.imshow(
            self.S,
            origin="lower",
            aspect="auto",
            extent=extent,
            interpolation="nearest",
        )
        self.cbar = self.fig.colorbar(self.im, ax=self.ax)
        self.cbar.set_label("dBFS")

        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Frequency (Hz)")
        self.ax.set_title("Scrolling Spectrogram")

        self._noise_filter: Optional[Callable[[np.ndarray], np.ndarray]] = None

    def set_noise_filter(
        self, fn: Optional[Callable[[np.ndarray], np.ndarray]]
    ) -> None:
        """Attach a noise filter to process each frame prior to display."""

        self._noise_filter = fn

    def process_frame(self, frame: np.ndarray):
        if self._noise_filter is not None:
            frame = self._noise_filter(frame)
        fw = frame * hann_window(len(frame))
        spec = np.fft.rfft(fw, n=self.fft_size)
        mag = np.abs(spec)[: self.max_bin]
        S_db = dbfs(mag)
        self.S = np.roll(self.S, shift=-1, axis=1)
        self.S[:, -1] = S_db

    def update_plot(self):
        vmax = np.max(self.S[:, -min(50, self.S.shape[1]) :])
        if not np.isfinite(vmax):
            vmax = -20.0
        self.im.set_data(self.S)
        self.im.set_clim(vmin=vmax - self.db_range, vmax=vmax)
        self.fig.canvas.draw_idle()


# ===================== PESTO Pitch implementation =====================
@dataclass
class PestoConfig:
    step_ms: float = 10.0
    model_name: str = "mir-1k_g7"
    sampling_rate: int = 44100
    max_batch_size: int = 4


class PestoPitchTracker:
    """
    Maintains a PESTO streaming model and turns audio buffers into a stream of
    (time, pitch_hz, confidence) tuples at step_ms resolution.
    """

    def __init__(self, cfg: PestoConfig):
        global torch, pesto
        if torch is None:
            import torch as _torch

            globals()["torch"] = _torch
        if pesto is None:
            import pesto as _pesto

            globals()["pesto"] = _pesto

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = pesto.load_model(
            cfg.model_name,
            step_size=cfg.step_ms,
            sampling_rate=cfg.sampling_rate,
            streaming=True,
            max_batch_size=cfg.max_batch_size,
        ).to(self.device)
        self.model.eval()

        self.sr = cfg.sampling_rate
        self.step_len = max(1, int(round(self.sr * cfg.step_ms / 1000.0)))
        self.buf = np.zeros(0, dtype=np.float32)
        self.t_ticks = 0  # number of emitted pesto steps

    def push_audio(self, samples: np.ndarray):
        if samples.size == 0:
            return None
        self.buf = np.concatenate([self.buf, samples])
        out = []
        # Feed model in fixed step_len chunks
        while self.buf.size >= self.step_len:
            chunk = self.buf[: self.step_len]
            self.buf = self.buf[self.step_len :]
            # Shape (batch=1, n)
            tensor = torch.from_numpy(chunk[None, :]).to(self.device)
            with torch.no_grad():
                # Ask for frequency directly
                f0, conf, amp = self.model(
                    tensor, convert_to_freq=True, return_activations=False
                )
            # f0, conf are tensors of shape (batch, 1) or (batch,) depending on impl
            f0_val = float(f0.reshape(-1)[0])
            c_val = float(conf.reshape(-1)[0])
            t_s = self.t_ticks * (self.step_len / self.sr)
            out.append((t_s, f0_val, c_val))
            self.t_ticks += 1
        return out


class CrepePitchTracker:
    """
    Streaming-ish wrapper for CREPE. We keep a small rolling buffer and call
    crepe.predict() to get an f0/conf sequence, then emit the last frame.
    """

    def __init__(
        self, sampling_rate: int = 44100, step_ms: float = 10.0, buffer_sec: float = 3.0
    ):
        global crepe
        if crepe is None:
            import crepe as _crepe

            globals()["crepe"] = _crepe
        self.in_sr = int(sampling_rate)
        self.step_ms = float(step_ms)
        self.buf = np.zeros(0, dtype=np.float32)
        self.buffer_sec = float(buffer_sec)
        self.target_sr = 16000  # CREPE expects 16 kHz
        self.ticks = 0

    def _resample_to_16k(self, x: np.ndarray) -> np.ndarray:
        if self.in_sr == self.target_sr:
            return x.astype(np.float32, copy=False)
        # Simple linear resampler (good enough for pitch tracking demo)
        n_src = x.size
        dur = n_src / self.in_sr
        n_tgt = int(round(dur * self.target_sr))
        if n_tgt <= 1:
            return np.zeros(0, dtype=np.float32)
        t_src = np.linspace(0.0, dur, num=n_src, endpoint=False)
        t_tgt = np.linspace(0.0, dur, num=n_tgt, endpoint=False)
        y = np.interp(t_tgt, t_src, x).astype(np.float32, copy=False)
        return y

    def push_audio(self, samples: np.ndarray):
        if samples.size == 0:
            return None
        self.buf = np.concatenate([self.buf, samples])
        # Keep a rolling buffer
        max_len = int(self.in_sr * self.buffer_sec)
        if self.buf.size > max_len:
            self.buf = self.buf[-max_len:]
        # Need enough audio to predict at least one frame
        if self.buf.size < int(self.in_sr * 0.25):
            return None
        # Resample to 16k for CREPE
        audio_16k = self._resample_to_16k(self.buf)
        if audio_16k.size < 160:  # ~10ms at 16k
            return None
        # step_size is in ms
        time_s, freq_hz, conf, _ = crepe.predict(
            audio_16k, self.target_sr, viterbi=False, step_size=self.step_ms, verbose=0
        )
        # Emit only the last estimate
        if freq_hz.size == 0:
            return None
        f0 = float(freq_hz[-1])
        c = float(conf[-1])
        # Approximate stream time as wall time offset similar to PESTO view; we return a tuple compatible with add_points
        return [(0.0, f0, c)]


class PestoPitchView:
    """
    Scrolling plot of pitch-vs-time; color indicates confidence (0..1).
    """

    def __init__(
        self,
        window_sec: float,
        ylim: tuple[float, float] | None,
        sr: int,
        step_len: int,
        conf_min: float = 0.1,
    ):
        self.window_sec = float(window_sec)
        self.sr = sr
        self.step_len = step_len
        self.t0 = time.time()
        self.conf_min = float(conf_min)

        self.times = []  # pesto: absolute seconds since start
        self.pitches = []  # pesto pitch (Hz)
        self.confs = []  # pesto confidence (0..1)
        self.ac_times = []  # autocorr: absolute seconds
        self.ac_pitches = []  # autocorr pitch (Hz)
        self.crepe_times = []  # CREPE absolute seconds
        self.crepe_pitches = []
        self.crepe_confs = []

        self.fig, self.ax = plt.subplots(figsize=(10, 5))
        self.scat = self.ax.scatter(
            [], [], c=[], cmap="viridis", vmin=0.0, vmax=1.0, s=18
        )
        self.cbar = self.fig.colorbar(self.scat, ax=self.ax)
        self.cbar.set_label("Confidence (0..1)")
        # Autocorr line (Hz)
        (self.ac_line,) = self.ax.plot([], [], linewidth=1.5)
        self.crepe_scat = self.ax.scatter(
            [], [], marker="s", s=18, cmap="viridis", vmin=0.0, vmax=1.0
        )
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Pitch (Hz)")
        self.ax.set_title("PESTO Pitch (Hz) & Autocorr (Hz)")

        if ylim is not None:
            self.ax.set_ylim(*ylim)

    def add_points(self, samples: list[tuple[float, float, float]]):
        # Shift model-relative times to wall time offset
        now = time.time() - self.t0
        for _, f0, conf in samples:
            if not np.isfinite(f0) or f0 <= 0:
                continue
            if conf < self.conf_min:
                continue
            self.times.append(now)  # approximate alignment
            self.pitches.append(float(f0))
            self.confs.append(float(np.clip(conf, 0.0, 1.0)))

        # Prune to window
        left = (time.time() - self.t0) - self.window_sec
        while self.times and self.times[0] < left:
            self.times.pop(0)
            self.pitches.pop(0)
            self.confs.pop(0)

    def add_crepe_points(self, samples):
        now = time.time() - self.t0
        for _, f0, conf in samples:
            if not np.isfinite(f0) or f0 <= 0 or conf < self.conf_min:
                continue
            self.crepe_times.append(now)
            self.crepe_pitches.append(float(f0))
            self.crepe_confs.append(float(np.clip(conf, 0.0, 1.0)))
        left = (time.time() - self.t0) - self.window_sec
        while self.crepe_times and self.crepe_times[0] < left:
            self.crepe_times.pop(0)
            self.crepe_pitches.pop(0)
            self.crepe_confs.pop(0)

    def add_ac_point(self, t_s, f0_hz):
        """Append an autocorr pitch point at current wall time (seconds since start)."""
        if f0_hz is None:
            return
        try:
            f0 = float(f0_hz)
        except Exception:
            return
        if not np.isfinite(f0) or f0 <= 0:
            return
        now = time.time() - self.t0
        self.ac_times.append(now)
        self.ac_pitches.append(f0)
        # Prune to window
        left = (time.time() - self.t0) - self.window_sec
        while self.ac_times and self.ac_times[0] < left:
            self.ac_times.pop(0)
            self.ac_pitches.pop(0)

    def update_plot(self):
        # Only plot confident points (optional)
        t = np.array(self.times, dtype=float)
        y = np.array(self.pitches, dtype=float)
        c = np.array(self.confs, dtype=float)

        # Build offsets for scatter
        mask = np.isfinite(y)
        offsets = (
            np.column_stack([t[mask], y[mask]]) if mask.any() else np.zeros((0, 2))
        )
        self.scat.set_offsets(offsets)
        self.scat.set_array(c[mask] if mask.any() else np.array([]))
        # Autocorr line data
        if len(self.ac_times) > 0:
            self.ac_line.set_data(
                np.array(self.ac_times, dtype=float),
                np.array(self.ac_pitches, dtype=float),
            )
        else:
            self.ac_line.set_data([], [])
        # Keep fixed horizontal window
        xmax = (
            t.max()
            if t.size
            else (self.ac_times[-1] if self.ac_times else self.window_sec)
        )
        self.ax.set_xlim(max(0, xmax - self.window_sec), xmax)
        # Update numeric readout
        latest_pesto = y[mask][-1] if mask.any() else float("nan")
        latest_ac = self.ac_pitches[-1] if self.ac_pitches else float("nan")
        self.fig.suptitle(
            f"PESTO: {latest_pesto:.1f} Hz | Autocorr: {latest_ac:.1f} Hz"
        )
        self.fig.canvas.draw_idle()


# ===================== App wrapper =====================


class App:
    def __init__(
        self,
        source: AudioSource,
        samplerate: int = 44100,
        fft_size: int = 2048,
        hop: int = 512,
        window_sec: float = 10.0,
        max_freq: float = 8000.0,
        db_range: float = 80.0,
        # Spectrogram noise filter params
        enable_noise_filter: bool = True,
        noise_sec: float = 2.0,
        over_sub: float = 1.0,
        # PESTO params
        pesto_enabled: bool = False,
        pesto_step_ms: float = 10.0,
        pesto_model: str = "mir-1k_g7",
        pesto_pitch_ylim: tuple[float, float] | None = None,
        pesto_conf_min: float = 0.1,
        crepe_enabled: bool = False,
        crepe_step_ms: float = 10.0,
    ):
        self.source = source
        self.sr = int(samplerate)
        self.hop = int(hop)
        self.window_sec = float(window_sec)
        self.paused = False
        self.mode = "pesto" if pesto_enabled else "spec"

        if self.mode == "spec":
            # Spectrogram view
            self.view = SpectrogramView(
                samplerate, fft_size, hop, window_sec, max_freq, db_range
            )
            # Spectrogram analysis buffers/params
            self.sample_buf = np.zeros(0, dtype=np.float32)
            self.noise_filter_enabled = bool(enable_noise_filter)
            if self.noise_filter_enabled and not SHARED_NOISE_TOOLS_AVAILABLE:
                raise RuntimeError(
                    "Shared noise-reduction utilities are unavailable; "
                    "install project dependencies or disable the noise filter."
                )
            self.noise_sec = float(noise_sec)
            self.over_sub = float(over_sub)
            self.fft_size = int(fft_size)
            self.win = hann_window(self.fft_size).astype(np.float32)
            self.freqs = np.fft.rfftfreq(self.fft_size, d=1.0 / self.sr)
            self.max_bin = np.searchsorted(self.freqs, max_freq, side="right")
            if self.max_bin < 2:
                self.max_bin = len(self.freqs)
            self.noise_profile: Optional[SharedNoiseProfile] = None
            self.noise_cfg: Optional[PitchCompareConfig] = (
                PitchCompareConfig(
                    sample_rate=int(self.sr),
                    noise_duration=float(self.noise_sec),
                    over_subtraction=float(self.over_sub),
                )
                if SHARED_NOISE_TOOLS_AVAILABLE
                else None
            )
        else:
            # PESTO: tracker + view
            cfg = PestoConfig(
                step_ms=pesto_step_ms, model_name=pesto_model, sampling_rate=samplerate
            )
            self.pesto = PestoPitchTracker(cfg)
            self.view = PestoPitchView(
                window_sec,
                pesto_pitch_ylim,
                samplerate,
                self.pesto.step_len,
                conf_min=pesto_conf_min,
            )
            # Keep FFT/noise info for autocorr resynthesis
            self.fft_size = int(fft_size)
            self.win = hann_window(self.fft_size).astype(np.float32)
            self.noise_sec = float(noise_sec)
            self.noise_filter_enabled = bool(enable_noise_filter)
            if self.noise_filter_enabled and not SHARED_NOISE_TOOLS_AVAILABLE:
                raise RuntimeError(
                    "Shared noise-reduction utilities are unavailable; "
                    "install project dependencies or disable the noise filter."
                )
            self.over_sub = float(over_sub)
            self.noise_profile: Optional[SharedNoiseProfile] = None
            self.noise_cfg: Optional[PitchCompareConfig] = (
                PitchCompareConfig(
                    sample_rate=int(self.sr),
                    noise_duration=float(self.noise_sec),
                    over_subtraction=float(self.over_sub),
                )
                if SHARED_NOISE_TOOLS_AVAILABLE
                else None
            )
            self.ac_buf = np.zeros(0, dtype=np.float32)

        # Connect keys
        self.fig = self.view.fig
        self.ax = self.view.ax
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        self.last_noise_profile_path: Optional[Path] = None

        if self.mode == "spec":
            self.view.set_noise_filter(self._apply_wiener_filter)

    def _profile_noise(self, notify: bool = False) -> bool:
        if notify:
            print("[INFO] Recording new noise profile...")
        if not SHARED_NOISE_TOOLS_AVAILABLE:
            if notify:
                print(
                    "[WARN] Noise-reduction utilities are unavailable; install project "
                    "dependencies to enable profiling."
                )
            return False
        target_samples = int(self.sr * max(0.2, self.noise_sec))
        collected = 0
        buf = np.zeros(0, dtype=np.float32)
        noise_frames: list[np.ndarray] = []
        t_start = time.time()
        timeout = max(3.0, self.noise_sec * 3.0)
        title0 = self.ax.get_title()
        self.ax.set_title("Profiling noise... keep quiet")
        self.fig.canvas.draw_idle()

        while collected < target_samples and (time.time() - t_start) < timeout:
            chunk = self.source.read()
            if chunk.size == 0:
                continue
            buf = np.concatenate([buf, chunk])
            collected += chunk.size
            while buf.size >= self.fft_size:
                frame = buf[: self.fft_size]
                buf = buf[self.hop :]
                noise_frames.append(frame.astype(np.float32))
                if len(noise_frames) > 2000:
                    break
        success = False
        if noise_frames and SHARED_NOISE_TOOLS_AVAILABLE:
            noise_samples = np.concatenate(noise_frames)
            if self.noise_cfg is not None and shared_compute_noise_profile is not None:
                self.noise_cfg.sample_rate = int(self.sr)
                self.noise_cfg.noise_duration = float(self.noise_sec)
                self.noise_cfg.over_subtraction = float(self.over_sub)
                self.noise_profile = shared_compute_noise_profile(
                    noise_samples.astype(np.float32, copy=False), self.noise_cfg
                )
                if self.noise_profile is not None:
                    self._persist_noise_profile(self.noise_profile)
                    success = True
            else:
                self.noise_profile = None
        else:
            self.noise_profile = None
        self.ax.set_title(title0)
        self.fig.canvas.draw_idle()
        if notify:
            if success:
                self.noise_filter_enabled = True
                print("[INFO] Noise profile captured; enabling filter.")
            else:
                print("[WARN] Unable to capture a noise profile.")
        return success

    def _persist_noise_profile(self, profile: SharedNoiseProfile) -> None:
        if not SHARED_NOISE_TOOLS_AVAILABLE or shared_save_noise_profile is None:
            return

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        duration_ms = int(round(self.noise_sec * 1000.0))
        file_name = (
            f"noise_sr{int(self.sr)}Hz_"
            f"win{int(profile.window_length)}_"
            f"hop{int(profile.hop_length)}_"
            f"dur{duration_ms}ms_"
            f"{timestamp}.npz"
        )

        cache_dir = NOISE_FILTER_DIR
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            return

        cache_path = cache_dir / file_name
        shared_save_noise_profile(profile, cache_path, int(self.sr))
        self.last_noise_profile_path = cache_path
        print(f"[INFO] Saved noise profile to {cache_path}")

    def on_key(self, event):
        if event.key in ("q", "escape"):
            plt.close(self.fig)
        elif event.key == "p":
            self.paused = not self.paused
        elif event.key == ",":
            self.window_sec = max(2.0, self.window_sec * 0.85)
        elif event.key == ".":
            self.window_sec = min(60.0, self.window_sec / 0.85)
        elif self.mode == "spec":
            if event.key == "n":
                self.noise_filter_enabled = not self.noise_filter_enabled
            elif event.key == "r":
                self._profile_noise(notify=True)

    # -------- Autocorr pitch (time-domain) --------
    def _autocorr_pitch(
        self, x: np.ndarray, fmin: float = 50.0, fmax: float = 1200.0
    ) -> float | None:
        """Estimate pitch via autocorrelation on window x (1D float). Returns Hz or None."""
        x = np.asarray(x, dtype=float)
        if x.size < 8:
            return None
        x = x - np.mean(x)
        if np.max(np.abs(x)) < 1e-8:
            return None
        # Fast autocorr via FFT
        n = int(1 << (int(np.ceil(np.log2(len(x)))) + 1))  # zero-pad
        X = np.fft.rfft(x, n=n)
        ac = np.fft.irfft(np.abs(X) ** 2)
        ac = ac[: len(x)]
        # Normalize
        if ac[0] <= 0:
            return None
        ac = ac / (ac[0] + 1e-12)
        # Lag bounds
        lag_min = int(self.sr / fmax)
        lag_max = int(self.sr / fmin)
        lag_min = max(1, lag_min)
        lag_max = min(lag_max, len(ac) - 1)
        if lag_max <= lag_min:
            return None
        # Peak search
        seg = ac[lag_min : lag_max + 1]
        k = int(np.argmax(seg)) + lag_min
        # Parabolic interpolation
        if 1 <= k < len(ac) - 1:
            denom = 2 * ac[k] - ac[k - 1] - ac[k + 1]
            if abs(denom) > 1e-12:
                delta = 0.5 * (ac[k - 1] - ac[k + 1]) / denom
                k = k + float(np.clip(delta, -1.0, 1.0))
        f0 = self.sr / float(k) if k > 0 else None
        return f0 if (f0 is not None and np.isfinite(f0)) else None

    def _apply_wiener_filter(self, frame: np.ndarray) -> np.ndarray:
        if (
            getattr(self, "noise_filter_enabled", False) is False
            or self.noise_profile is None
            or not SHARED_NOISE_TOOLS_AVAILABLE
            or shared_wiener_filter is None
        ):
            return frame
        if self.noise_cfg is not None:
            self.noise_cfg.over_subtraction = float(self.over_sub)
        filtered = shared_wiener_filter(
            np.asarray(frame, dtype=np.float32, copy=False),
            self.noise_profile,
            over_subtraction=float(self.over_sub),
        )
        return filtered

    # -------- Chunk processing --------
    def process_chunk(self, chunk: np.ndarray):
        if self.mode == "spec":
            if chunk.size == 0:
                return
            self.sample_buf = np.concatenate([self.sample_buf, chunk])
            while self.sample_buf.size >= self.fft_size:
                frame = self.sample_buf[: self.fft_size]
                self.sample_buf = self.sample_buf[self.hop :]
                self.view.process_frame(frame)
        else:
            # PESTO: push to tracker
            results = self.pesto.push_audio(chunk)
            if results:
                self.view.add_points(results)
            if getattr(self, "crepe", None) is not None:
                c_results = self.crepe.push_audio(chunk)
                if c_results:
                    self.view.add_crepe_points(c_results)
            # Autocorr: accumulate time domain, compute from last window
            if chunk.size > 0:
                self.ac_buf = np.concatenate([self.ac_buf, chunk])
                if self.ac_buf.size > self.fft_size * 3:
                    self.ac_buf = self.ac_buf[-self.fft_size * 3 :]
            if results and self.ac_buf.size >= self.fft_size:
                frame = self.ac_buf[-self.fft_size :]
                frame = self._apply_wiener_filter(frame)
                spec_clean = np.fft.rfft(frame * self.win, n=self.fft_size)
                td = np.fft.irfft(spec_clean, n=self.fft_size)
                f0_ac = self._autocorr_pitch(td)
                self.view.add_ac_point(0.0, f0_ac)

    def update_plot(self):
        self.view.update_plot()

    # -------- Main loop --------
    def run(self):
        self.source.start()
        try:
            # Noise profiling for both modes (optional in pesto)
            if self.mode == "spec" and getattr(self, "noise_filter_enabled", False):
                self._profile_noise()
            if self.mode == "pesto":
                try:
                    self._profile_noise()
                except Exception:
                    pass

            def _on_timer(event):
                if self.paused:
                    return
                drained = 0
                while True:
                    chunk = self.source.read()
                    if chunk.size == 0:
                        break
                    self.process_chunk(chunk)
                    drained += 1
                    if drained > 12:
                        break
                self.update_plot()

            timer = self.fig.canvas.new_timer(interval=30)
            timer.add_callback(_on_timer, None)
            timer.start()
            plt.show()
        finally:
            self.source.stop()


def parse_args():
    ap = argparse.ArgumentParser(
        description="Scrolling Spectrogram / PESTO Pitch Visualizer"
    )
    ap.add_argument("--samplerate", type=int, default=44100, help="Sample rate (Hz)")
    ap.add_argument(
        "--fft", type=int, default=2048, help="FFT size (power of 2 recommended)"
    )
    ap.add_argument("--hop", type=int, default=512, help="Hop size (samples)")
    ap.add_argument(
        "--window", type=float, default=10.0, help="Visible window (seconds)"
    )
    ap.add_argument(
        "--max-freq",
        type=float,
        default=8000.0,
        help="Max frequency shown (Hz) [spectrogram]",
    )
    ap.add_argument(
        "--db-range", type=float, default=80.0, help="Dynamic range (dB) [spectrogram]"
    )
    ap.add_argument(
        "--device", type=str, default=None, help="Input device name or index"
    )
    ap.add_argument(
        "--demo", action="store_true", help="Run in demo mode (no microphone)"
    )

    # Spectrogram noise filter
    ap.add_argument(
        "--noise-sec",
        type=float,
        default=2.0,
        help="Seconds to record for noise profiling at startup",
    )
    ap.add_argument(
        "--over-sub",
        type=float,
        default=1.0,
        help="Noise over-subtraction factor (1.0 = equal)",
    )
    ap.add_argument(
        "--no-noise-filter",
        action="store_true",
        help="Disable noise filter (no profiling/subtraction)",
    )

    # PESTO
    ap.add_argument(
        "--pesto",
        action="store_true",
        help="Enable PESTO pitch mode (requires pesto-pitch, torch, torchaudio)",
    )
    ap.add_argument(
        "--pesto-step-ms",
        type=float,
        default=10.0,
        help="PESTO step size in ms (streaming)",
    )
    ap.add_argument(
        "--pesto-model",
        type=str,
        default="mir-1k_g7",
        help="PESTO pretrained model name or path",
    )
    ap.add_argument(
        "--crepe",
        action="store_true",
        help="Overlay CREPE pitch estimation alongside PESTO",
    )
    ap.add_argument(
        "--crepe-step-ms", type=float, default=10.0, help="CREPE step size in ms"
    )
    ap.add_argument(
        "--pesto-min", type=float, default=50.0, help="Suggested Y-axis min pitch (Hz)"
    )
    ap.add_argument(
        "--pesto-max",
        type=float,
        default=2000.0,
        help="Suggested Y-axis max pitch (Hz)",
    )
    ap.add_argument(
        "--pesto-conf-min",
        type=float,
        default=0.1,
        help="Minimum PESTO confidence to plot a point (0..1)",
    )

    return ap.parse_args()


def main():
    args = parse_args()

    # Select audio source
    if args.demo or sd is None:
        source = DemoSource(args.samplerate, args.hop)
    else:
        try:
            source = MicSource(args.samplerate, args.hop, device=args.device)
        except Exception as e:
            print(f"[WARN] Could not initialize microphone input: {e}")
            print(
                "Falling back to demo mode. Use --device to select input or install sounddevice."
            )
            source = DemoSource(args.samplerate, args.hop)

    pesto_ylim = (args.pesto_min, args.pesto_max) if args.pesto else None

    app = App(
        source=source,
        samplerate=args.samplerate,
        fft_size=args.fft,
        hop=args.hop,
        window_sec=args.window,
        max_freq=args.max_freq,
        db_range=args.db_range,
        enable_noise_filter=(not args.no_noise_filter),
        noise_sec=args.noise_sec,
        over_sub=args.over_sub,
        pesto_enabled=args.pesto,
        pesto_step_ms=args.pesto_step_ms,
        pesto_model=args.pesto_model,
        pesto_pitch_ylim=pesto_ylim,
        pesto_conf_min=args.pesto_conf_min,
        crepe_enabled=args.crepe,
        crepe_step_ms=args.crepe_step_ms,
    )
    app.run()


if __name__ == "__main__":
    main()
