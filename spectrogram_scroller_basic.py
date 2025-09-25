#!/usr/bin/env python3
"""
Scrolling Spectrogram Visualizer with:
  • Noise Profiling
  • Live Spectrum Line (current frame, dBFS)
  • Autocorrelation vs Frequency (time-domain ACF, Hz = 1/lag)
  • Frequency-Domain Autocorrelation vs Δf
  • CREPE Pitch Track (now: frequency on X, time on Y, scrolling upward)
  • Spectrogram (now: frequency on X, time on Y, scrolling upward)

Keyboard:
p : pause/resume
q / ESC : quit
[ / ] : decrease/increase spectrogram dynamic range (dB span)
- / = : decrease/increase max frequency shown
, / . : decrease/increase time window (seconds)
n : toggle noise filter on/off
r : re-profile noise
"""

import argparse
import time
import queue
import threading
import numpy as np
import matplotlib.pyplot as plt

# Optional deps
try:
    import sounddevice as sd
except Exception:
    sd = None

try:
    import crepe  # https://github.com/marl/crepe

    try:
        from crepe.core import FREQUENCY_BINS as _CREPE_FREQ_BINS
    except Exception:
        _CREPE_FREQ_BINS = None

    _HAS_CREPE = True
except Exception:
    _HAS_CREPE = False
    _CREPE_FREQ_BINS = None

EPS = 1e-12

CREPE_DEFAULT_MIN_HZ = 32.70319566257483  # ~= C1
CREPE_DEFAULT_MAX_HZ = 1975.533205024496  # ~= B6


def _make_crepe_freq_axis(n_bins: int) -> np.ndarray:
    if n_bins <= 0:
        return np.zeros(0, dtype=np.float32)
    if _CREPE_FREQ_BINS is not None:
        base = np.asarray(_CREPE_FREQ_BINS, dtype=np.float32)
        if base.size == n_bins:
            return base.copy()
        x = np.linspace(0, base.size - 1, n_bins, dtype=np.float32)
        idx = np.arange(base.size, dtype=np.float32)
        return np.interp(x, idx, base).astype(np.float32)
    axis = np.geomspace(CREPE_DEFAULT_MIN_HZ, CREPE_DEFAULT_MAX_HZ, n_bins)
    return axis.astype(np.float32)


def dbfs(x: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(x, EPS))


def hann_window(n: int) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)


class AudioSource:
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
        if indata.ndim == 2 and indata.shape[1] > 1:
            mono = indata.mean(axis=1).copy()
        else:
            mono = indata[:, 0].copy() if indata.ndim == 2 else indata.copy()
        try:
            self.q.put_nowait(mono)
        except queue.Full:
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
        chirp = np.sin(2 * np.pi * (100 + (t * 0.5e3)) * t) * 0.4
        tone1 = 0.25 * np.sin(2 * np.pi * 440 * t)
        tone2 = 0.2 * np.sin(2 * np.pi * 880 * t + 0.3)
        noise = 0.02 * np.random.randn(n)
        y = chirp + tone1 + tone2 + noise
        self.t += n
        y = np.tanh(1.5 * y)
        return y.astype(np.float32)

    def stop(self):
        pass


class ScrollingSpectrogram:
    def __init__(
        self,
        source: AudioSource,
        samplerate: int = 44100,
        fft_size: int = 8192,
        hop: int = 512,
        window_sec: float = 5.0,
        max_freq: float = 2000.0,
        db_range: float = 80.0,
        enable_noise_filter: bool = True,
        noise_sec: float = 2.0,
        over_sub: float = 1.0,
        min_freq: float = 10.0,
        ac_win_sec: float = 0.5,
        # CREPE
        enable_pitch: bool = True,
        crepe_capacity: str = "small",
        crepe_step_ms: int = 20,
        crepe_win_sec: float = 0.5,
    ):
        self.source = source
        self.sr = samplerate
        self.fft_size = int(fft_size)
        self.hop = int(hop)
        self.window_sec = float(window_sec)
        self.max_freq = float(max_freq)
        self.db_range = float(db_range)

        # Noise filter
        self.noise_filter_enabled = bool(enable_noise_filter)
        self.noise_sec = float(noise_sec)
        self.over_sub = float(over_sub)
        self.noise_mag = None

        # FFT & freqs
        self.win = hann_window(self.fft_size).astype(np.float32)
        self.freqs = np.fft.rfftfreq(self.fft_size, d=1.0 / self.sr)
        self.max_bin = np.searchsorted(self.freqs, self.max_freq, side="right")
        if self.max_bin < 2:
            self.max_bin = len(self.freqs)

        # Time/freq buffer sizing
        self.n_rows = max(
            10, int(round(self.window_sec * self.sr / self.hop))
        )  # rows == time bins (scrolling)
        # IMPORTANT: Spectrogram buffer is now (time_rows, freq_bins)
        self.S = np.full((self.n_rows, self.max_bin), -120.0, dtype=np.float32)

        # Buffers
        self.sample_buf = np.zeros(0, dtype=np.float32)
        self.td_buf = np.zeros(0, dtype=np.float32)
        self.ac_win_sec = float(ac_win_sec)
        self.ac_win_samps = int(round(self.ac_win_sec * self.sr))
        self.min_freq = max(1e-6, float(min_freq))
        self.acf_max_lag_s = 1.0 / self.min_freq
        self.acf_max_lag_samps = max(1, int(round(self.acf_max_lag_s * self.sr)))

        # Figure: top row splits spectrogram & CREPE, remaining rows span both columns
        self.fig = plt.figure(figsize=(14, 15))
        gs = self.fig.add_gridspec(
            nrows=5,
            ncols=2,
            height_ratios=[3, 1, 1, 1, 1],
            width_ratios=[1, 1],
            hspace=0.55,
            wspace=0.3,
        )
        self.ax_spec = self.fig.add_subplot(gs[0, 0])
        self.ax_pitch = self.fig.add_subplot(gs[0, 1])
        self.ax_line = self.fig.add_subplot(gs[1, :])
        self.ax_acf = self.fig.add_subplot(gs[2, :])
        self.ax_facf = self.fig.add_subplot(
            gs[3, :]
        )  # frequency-domain ACF vs Δf
        self.ax_facf2 = self.fig.add_subplot(
            gs[4, :]
        )  # NEW: autocorr of (frequency-domain ACF) vs Δf

        # --- Spectrogram image: X=freq, Y=time (scroll UP)
        # extent: x [0..max_freq], y [-window..0]; origin lower => top is 0 (newest)
        self.im = self.ax_spec.imshow(
            self.S,
            origin="lower",
            aspect="auto",
            extent=(0.0, self.freqs[self.max_bin - 1], -self.window_sec, 0.0),
            interpolation="nearest",
        )
        self.cbar = self.fig.colorbar(self.im, ax=self.ax_spec)
        self.cbar.set_label("dBFS")
        self.ax_spec.set_xlabel("Frequency (Hz)")
        self.ax_spec.set_ylabel("Time (s)")

        # Spectrum line (still X=freq, Y=dB)
        self.line_freqs = self.freqs[: self.max_bin]
        self.last_mag_db = np.full_like(self.line_freqs, -120.0, dtype=np.float32)
        self.last_mag_lin = np.zeros_like(self.line_freqs, dtype=np.float32)
        (self.line_plot,) = self.ax_line.plot(self.line_freqs, self.last_mag_db, lw=1.0)
        self.ax_line.set_xlim(0, self.freqs[self.max_bin - 1])
        self.ax_line.set_ylim(-120, 0)
        self.ax_line.set_xlabel("Frequency (Hz)")
        self.ax_line.set_ylabel("dBFS")
        self.ax_line.grid(True, alpha=0.25, which="both")

        # Time-domain ACF → frequency axis (unchanged)
        lags = np.arange(1, self.acf_max_lag_samps + 1) / self.sr
        self.acf_freq_axis_hz = 1.0 / lags
        mask_acf = self.acf_freq_axis_hz <= self.max_freq
        self.acf_freq_axis_hz = self.acf_freq_axis_hz[mask_acf]
        self.acf_vals = np.zeros_like(self.acf_freq_axis_hz, dtype=np.float32)
        (self.acf_plot,) = self.ax_acf.plot(
            self.acf_freq_axis_hz, self.acf_vals, lw=1.0, label="ACF"
        )
        self.ax_acf.set_xlim(self.min_freq, self.max_freq)
        self.ax_acf.set_ylim(-1.0, 1.0)
        self.ax_acf.set_xlabel("Frequency (Hz) (from 1/lag)")
        self.ax_acf.set_ylabel("Autocorr (norm)")
        self.ax_acf.grid(True, alpha=0.25)
        self._acf_legend = self.ax_acf.legend(loc="upper right", framealpha=0.3)

        # Frequency-domain ACF (unchanged)
        self._rebuild_facf_axis()

        # --- CREPE pitch track: X=freq, Y=time, scrolling UP
        self.enable_pitch = bool(enable_pitch) and _HAS_CREPE
        self.crepe_capacity = str(crepe_capacity)
        self.crepe_step_ms = int(crepe_step_ms)
        self.crepe_win_sec = float(crepe_win_sec)
        self.pitch_freq_bins = _make_crepe_freq_axis(360)
        self.pitch_activation = np.zeros(
            (self.n_rows, self.pitch_freq_bins.size), dtype=np.float32
        )
        self.pitch_im = self.ax_pitch.imshow(
            self.pitch_activation,
            origin="lower",
            aspect="auto",
            extent=(
                self.pitch_freq_bins[0],
                self.pitch_freq_bins[-1],
                -self.window_sec,
                0.0,
            ),
            interpolation="nearest",
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
        )
        self.pitch_cbar = self.fig.colorbar(
            self.pitch_im, ax=self.ax_pitch, fraction=0.046, pad=0.04
        )
        self.pitch_cbar.set_label("CREPE confidence")
        self.ax_pitch.set_xlim(self.min_freq, self.max_freq)
        self.ax_pitch.set_ylim(-self.window_sec, 0.0)  # newest at top (0)
        self.ax_pitch.set_xlabel("Pitch (Hz)")
        self.ax_pitch.set_ylabel("Time (s)")
        self.ax_pitch.grid(True, alpha=0.25)
        if not _HAS_CREPE:
            self.ax_pitch.text(
                0.02,
                0.8,
                "Install 'crepe' to enable pitch",
                transform=self.ax_pitch.transAxes,
                fontsize=9,
                color="red",
                ha="left",
                va="center",
            )

        self._set_titles()
        self.paused = False
        self._last_crepe_run = 0.0
        self._crepe_min_interval = 0.10

        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

    def _set_titles(self):
        nf = (
            "ON"
            if (self.noise_filter_enabled and self.noise_mag is not None)
            else ("ON (profiling...)" if self.noise_filter_enabled else "OFF")
        )
        self.ax_spec.set_title(f"Spectrogram (X=freq, Y=time ↑)  |  Noise filter: {nf}")
        self.ax_line.set_title("Current Spectrum (latest frame)")
        self.ax_acf.set_title(
            f"Autocorrelation vs Frequency (last {self.ac_win_sec:.2f}s, {self.min_freq:.1f}–{self.max_freq:.0f} Hz)"
        )
        self.ax_facf.set_title(
            f"Frequency-domain Autocorrelation vs Δf (min {self.min_freq:.1f} Hz … max {self.max_freq:.0f} Hz)"
        )
        self.ax_pitch.set_title("Pitch (CREPE): X=freq, Y=time ↑, color=confidence")

    def on_key(self, event):
        if event.key in ("q", "escape"):
            plt.close(self.fig)
        elif event.key == "p":
            self.paused = not self.paused
            self._set_titles()
            self.fig.canvas.draw_idle()
        elif event.key == "[":
            self.db_range = max(20.0, self.db_range - 5.0)
            self._update_clim()
        elif event.key == "]":
            self.db_range = min(140.0, self.db_range + 5.0)
            self._update_clim()
        elif event.key == "-":
            self.max_freq = max(200.0, self.max_freq * 0.85)
            self._update_freq_axis()
        elif event.key in ("=", "+"):
            self.max_freq = min(self.sr / 2, self.max_freq / 0.85)
            self._update_freq_axis()
        elif event.key == ",":
            self.window_sec = max(2.0, self.window_sec * 0.85)
            self._update_time_axis()
        elif event.key == ".":
            self.window_sec = min(60.0, self.window_sec / 0.85)
            self._update_time_axis()
        elif event.key == "n":
            self.noise_filter_enabled = not self.noise_filter_enabled
            self._set_titles()
            self.fig.canvas.draw_idle()
        elif event.key == "r":
            self.profile_noise(self.noise_sec, show_status=True)

    def _update_freq_axis(self):
        # Spectrogram/Spectrum frequency dimension updates
        self.max_bin = np.searchsorted(self.freqs, self.max_freq, side="right")
        self.max_bin = max(2, min(self.max_bin, len(self.freqs)))
        # Resize spectrogram columns (freq dimension)
        new_S = np.full((self.S.shape[0], self.max_bin), -120.0, dtype=np.float32)
        copy_bins = min(self.S.shape[1], self.max_bin)
        new_S[:, :copy_bins] = self.S[:, :copy_bins]
        self.S = new_S
        # Update spectrogram extent (x axis)
        self.im.set_data(self.S)
        self.im.set_extent((0.0, self.freqs[self.max_bin - 1], -self.window_sec, 0.0))

        # Spectrum line arrays
        self.line_freqs = self.freqs[: self.max_bin]
        self.last_mag_db = (
            self.last_mag_db[: self.max_bin]
            if self.last_mag_db.size >= self.max_bin
            else np.pad(
                self.last_mag_db,
                (0, self.max_bin - self.last_mag_db.size),
                constant_values=-120.0,
            )
        )
        self.last_mag_lin = (
            self.last_mag_lin[: self.max_bin]
            if self.last_mag_lin.size >= self.max_bin
            else np.pad(
                self.last_mag_lin,
                (0, self.max_bin - self.last_mag_lin.size),
                constant_values=0.0,
            )
        )
        self.line_plot.set_data(self.line_freqs, self.last_mag_db)
        self.ax_line.set_xlim(0, self.freqs[self.max_bin - 1])

        # Rebuild Δf axis; update pitch x-limits
        self._rebuild_facf_axis()
        self.ax_pitch.set_xlim(self.min_freq, self.max_freq)

        self.fig.canvas.draw_idle()

    def _update_time_axis(self):
        # Recompute number of time rows and rebuild time-dependent arrays
        new_rows = max(10, int(round(self.window_sec * self.sr / self.hop)))
        if new_rows != self.S.shape[0]:
            new_S = np.full((new_rows, self.S.shape[1]), -120.0, dtype=np.float32)
            copy_rows = min(self.S.shape[0], new_rows)
            new_S[-copy_rows:, :] = self.S[-copy_rows:, :]  # keep newest at top
            self.S = new_S
        self.n_rows = self.S.shape[0]
        # Update spectrogram extent (y axis)
        self.im.set_extent((0.0, self.freqs[self.max_bin - 1], -self.window_sec, 0.0))

        # Rebuild pitch activation buffer to match rows
        if self.pitch_activation.shape[0] != self.n_rows:
            new_pitch = np.zeros(
                (self.n_rows, self.pitch_activation.shape[1]), dtype=np.float32
            )
            copy_rows = min(self.pitch_activation.shape[0], self.n_rows)
            new_pitch[-copy_rows:, :] = self.pitch_activation[-copy_rows:, :]
            self.pitch_activation = new_pitch
            self.pitch_im.set_data(self.pitch_activation)
        self.ax_pitch.set_ylim(-self.window_sec, 0.0)
        self.pitch_im.set_extent(
            (
                self.pitch_freq_bins[0],
                self.pitch_freq_bins[-1],
                -self.window_sec,
                0.0,
            )
        )

        self.fig.canvas.draw_idle()

    def _update_clim(self):
        vmax = np.max(self.S)
        vmax = -20.0 if not np.isfinite(vmax) else vmax
        self.im.set_clim(vmin=vmax - self.db_range, vmax=vmax)
        self.ax_line.set_ylim(vmax - self.db_range, vmax)
        self.fig.canvas.draw_idle()

    def _spec_mag(self, frame: np.ndarray) -> np.ndarray:
        fw = frame * self.win
        spec = np.fft.rfft(fw, n=self.fft_size)
        mag = np.abs(spec)
        return mag[: self.max_bin]

    def profile_noise(self, seconds: float, show_status: bool = False):
        if show_status:
            self.ax_spec.set_title("Profiling noise... Please keep the room quiet.")
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

        target_samples = int(self.sr * max(0.2, seconds))
        buf = np.zeros(0, dtype=np.float32)
        mags = []
        t0 = time.time()
        timeout = max(3.0, seconds * 3.0)
        collected = 0
        while collected < target_samples and (time.time() - t0) < timeout:
            chunk = self.source.read()
            if chunk.size == 0:
                continue
            buf = np.concatenate([buf, chunk])
            collected += chunk.size
            while buf.size >= self.fft_size:
                frame = buf[: self.fft_size]
                buf = buf[self.hop :]
                mags.append(self._spec_mag(frame))
                if len(mags) > 2000:
                    break

        self.noise_mag = (
            None
            if len(mags) == 0
            else np.median(np.vstack(mags), axis=0).astype(np.float32)
        )
        self._set_titles()
        self.fig.canvas.draw_idle()

    def _apply_noise_filter(self, mag: np.ndarray) -> np.ndarray:
        if not self.noise_filter_enabled or self.noise_mag is None:
            return mag
        return np.maximum(mag - self.over_sub * self.noise_mag, 0.0)

    # -------- ACF plots (unchanged logic) --------
    def _find_local_peaks(self, y: np.ndarray) -> np.ndarray:
        if y.size < 3:
            return np.array([], dtype=int)
        return np.where((y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:]))[0] + 1

    def _update_acf_plot(self):
        if self.td_buf.size < self.ac_win_samps:
            return
        x = self.td_buf[-self.ac_win_samps :].astype(np.float32)
        x = x - np.mean(x)
        power = np.sum(x * x)
        if power <= 1e-12:
            self.acf_vals[:] = 0.0
            peak_freq = float("nan")
            p2p = float("nan")
        else:
            acf_full = np.correlate(x, x, mode="full")
            acf_pos = acf_full[len(x) - 1 :]
            acf0 = acf_pos[0] if acf_pos[0] != 0 else 1.0
            max_lag = min(self.acf_max_lag_samps, len(acf_pos) - 1)
            acf_sel = acf_pos[1 : max_lag + 1] / acf0
            freqs = 1.0 / (np.arange(1, max_lag + 1) / self.sr)
            mask = freqs <= self.max_freq
            freqs_sel = freqs[mask]
            acf_sel = acf_sel[mask]
            order = np.argsort(freqs_sel)
            freqs_sel = freqs_sel[order]
            acf_sel = acf_sel[order]
            L = min(self.acf_vals.size, acf_sel.size)
            self.acf_vals[:L] = acf_sel[:L]
            if L < self.acf_vals.size:
                self.acf_vals[L:] = 0.0
            self.acf_freq_axis_hz = freqs_sel[:L].astype(np.float32)
            peaks = self._find_local_peaks(self.acf_vals[:L])
            if peaks.size == 0:
                gi = int(np.argmax(self.acf_vals[:L]))
                peak_freq = float(self.acf_freq_axis_hz[gi])
                p2p = float("nan")
            else:
                gi_rel = int(np.argmax(self.acf_vals[:L][peaks]))
                gi = int(peaks[gi_rel])
                peak_freq = float(self.acf_freq_axis_hz[gi])
                if peaks.size >= 2:
                    sort_idx = np.argsort(self.acf_vals[:L][peaks])[::-1]
                    p1 = int(peaks[sort_idx[0]])
                    p2 = int(peaks[sort_idx[1]])
                    p2p = abs(
                        float(self.acf_freq_axis_hz[p1] - self.acf_freq_axis_hz[p2])
                    )
                else:
                    p2p = float("nan")

        # ---- Legend metrics: global max frequency & height delta (0..2)
        if self.acf_freq_axis_hz.size > 0 and np.any(np.isfinite(self.acf_vals)):
            # restrict to the portion actually plotted (finite values)
            mask_finite = np.isfinite(self.acf_vals)
            vals = self.acf_vals[mask_finite]
            freqs_plot = self.acf_freq_axis_hz[mask_finite]
            if vals.size > 0:
                gi = int(np.argmax(vals))
                peak_freq = float(freqs_plot[gi])
                ymax = float(vals[gi])
                ymin = float(np.min(vals))
                dheight = max(0.0, min(2.0, ymax - ymin))  # clamp to [0, 2]
            else:
                peak_freq = float("nan")
                dheight = float("nan")
        else:
            peak_freq = float("nan")
            dheight = float("nan")

        # Update artist & legend
        self.acf_plot.set_data(self.acf_freq_axis_hz, self.acf_vals)
        label = f"ACF  |  max≈{peak_freq:.1f} Hz  |  Δheight≈{dheight:.2f}"
        self.acf_plot.set_label(label)
        if hasattr(self, "_acf_legend") and self._acf_legend:
            self._acf_legend.remove()
        self._acf_legend = self.ax_acf.legend(loc="upper right", framealpha=0.3)

    def _rebuild_facf_axis(self):
        """(Re)build the Δf axis and both frequency-domain ACF lines."""
        df = self.freqs[1] - self.freqs[0] if len(self.freqs) > 1 else 1.0
        shift_min_bins = max(1, int(np.ceil(self.min_freq / df)))
        shift_max_bins = max(shift_min_bins, int(np.floor(self.max_freq / df)))

        # Δf axis in Hz (positive shifts only)
        self.facf_shift_axis_hz = (
            np.arange(shift_min_bins, shift_max_bins + 1) * df
        ).astype(np.float32)
        if self.facf_shift_axis_hz.size == 0:
            self.facf_shift_axis_hz = np.array([self.min_freq], dtype=np.float32)

        # Main frequency-domain ACF (spectrum ⊗ spectrum)
        self.facf_vals = np.zeros_like(self.facf_shift_axis_hz, dtype=np.float32)
        if hasattr(self, "facf_plot"):
            self.facf_plot.set_data(self.facf_shift_axis_hz, self.facf_vals)
        else:
            (self.facf_plot,) = self.ax_facf.plot(
                self.facf_shift_axis_hz, self.facf_vals, lw=1.0
            )
        self.ax_facf.set_xlim(self.facf_shift_axis_hz[0], self.facf_shift_axis_hz[-1])
        self.ax_facf.set_ylim(-1.0, 1.0)
        self.ax_facf.set_xlabel("Frequency shift Δf (Hz)")
        self.ax_facf.set_ylabel("Autocorr (norm)")
        self.ax_facf.grid(True, alpha=0.25)

        # NEW: Autocorrelation of the FACF curve itself (FACF ⊗ FACF)
        self.facf2_vals = np.zeros_like(self.facf_shift_axis_hz, dtype=np.float32)
        if hasattr(self, "facf2_plot"):
            self.facf2_plot.set_data(self.facf_shift_axis_hz, self.facf2_vals)
        else:
            (self.facf2_plot,) = self.ax_facf2.plot(
                self.facf_shift_axis_hz, self.facf2_vals, lw=1.0
            )
        self.ax_facf2.set_xlim(self.facf_shift_axis_hz[0], self.facf_shift_axis_hz[-1])
        self.ax_facf2.set_ylim(-1.0, 1.0)
        self.ax_facf2.set_xlabel("Frequency shift Δf (Hz)")
        self.ax_facf2.set_ylabel("Autocorr of FACF (norm)")
        self.ax_facf2.grid(True, alpha=0.25)

    def _update_facf_plot(self):
        if self.last_mag_lin is None or self.last_mag_lin.size < 4:
            return
        x = self.last_mag_lin[: self.max_bin].astype(np.float64)
        x = x - x.mean()
        var = np.sum(x * x)
        if var <= 1e-18:
            self.facf_vals[:] = 0.0
        else:
            acf_full = np.correlate(x, x, mode="full")
            acf_pos = acf_full[len(x) - 1 :]
            acf0 = acf_pos[0] if acf_pos[0] != 0 else 1.0
            df = self.freqs[1] - self.freqs[0] if len(self.freqs) > 1 else 1.0
            shift_bins = np.round(self.facf_shift_axis_hz / df).astype(int)
            shift_bins = np.clip(shift_bins, 1, len(acf_pos) - 1)
            vals = acf_pos[shift_bins] / acf0
            if vals.size != self.facf_vals.size:
                self.facf_vals = np.resize(self.facf_vals, vals.shape)
                self.facf_plot.set_data(self.facf_shift_axis_hz, self.facf_vals)
            self.facf_vals[:] = vals
            self.facf_plot.set_data(self.facf_shift_axis_hz, self.facf_vals)

        # ---- Legend: first peak Δf
        peak_hz = float("nan")
        if self.facf_vals.size > 3:
            peaks = self._find_local_peaks(self.facf_vals)
            if peaks.size > 0:
                # take the first peak (smallest Δf > 0)
                peak_hz = float(self.facf_shift_axis_hz[peaks[0]])
        label = (
            f"FACF  |  1st peak≈{peak_hz:.1f} Hz" if np.isfinite(peak_hz) else "FACF"
        )
        self.facf_plot.set_label(label)
        if hasattr(self, "_facf_legend") and self._facf_legend:
            self._facf_legend.remove()
        self._facf_legend = self.ax_facf.legend(loc="upper right", framealpha=0.3)

    def _update_facf2_plot(self):
        """Update autocorrelation of the frequency-domain ACF (FACF ⊗ FACF) vs Δf."""
        if not hasattr(self, "facf_vals") or self.facf_vals.size < 3:
            return

        # Work on a finite slice (guard against NaNs)
        y = self.facf_vals.astype(np.float64)
        mask_finite = np.isfinite(y)
        if not np.any(mask_finite):
            self.facf2_vals[:] = 0.0
            self.facf2_plot.set_data(self.facf_shift_axis_hz, self.facf2_vals)
            return
        y = y[mask_finite]

        # Zero-mean before autocorrelation; normalize by lag-0
        y = y - y.mean()
        var = np.dot(y, y)
        if var <= 1e-18 or y.size < 3:
            self.facf2_vals[:] = 0.0
        else:
            acf_full = np.correlate(y, y, mode="full")
            acf_pos = acf_full[len(y) - 1 :]  # lags >= 0 (bin units along Δf axis)
            acf0 = acf_pos[0] if acf_pos[0] != 0 else 1.0

            # Map positive lag bins onto our Δf axis (skip 0 to show Δf>0 only)
            # facf_shift_axis_hz corresponds to positive-shift bins starting at shift_min_bins
            # We sample the autocorr at these same shift bins to share the x-axis.
            df = (
                self.facf_shift_axis_hz[1] - self.facf_shift_axis_hz[0]
                if len(self.facf_shift_axis_hz) > 1
                else 1.0
            )
            # Convert Δf to bin index (integer)
            shift_bins = np.round(self.facf_shift_axis_hz / df).astype(int)
            shift_bins = np.clip(
                shift_bins, 1, len(acf_pos) - 1
            )  # skip lag=0, stay within bounds

            vals = (acf_pos[shift_bins] / acf0).astype(np.float32)

            # Size-guard to our preallocated buffer
            if vals.size != self.facf2_vals.size:
                self.facf2_vals = np.resize(self.facf2_vals, vals.shape)
                self.facf2_plot.set_data(self.facf_shift_axis_hz, self.facf2_vals)
            self.facf2_vals[:] = vals

        self.facf2_plot.set_data(self.facf_shift_axis_hz, self.facf2_vals)
        # ---- Legend: first peak Δf
        peak_hz = float("nan")
        if self.facf2_vals.size > 3:
            peaks = self._find_local_peaks(self.facf2_vals)
            if peaks.size > 0:
                peak_hz = float(self.facf_shift_axis_hz[peaks[0]])
        label = (
            f"FACF²  |  1st peak≈{peak_hz:.1f} Hz" if np.isfinite(peak_hz) else "FACF²"
        )
        self.facf2_plot.set_label(label)
        if hasattr(self, "_facf2_legend") and self._facf2_legend:
            self._facf2_legend.remove()
        self._facf2_legend = self.ax_facf2.legend(loc="upper right", framealpha=0.3)

    # -------- CREPE pitch (X=freq, Y=time ↑) --------

    def _ensure_pitch_activation_bins(self, n_bins: int):
        n_bins = max(1, int(n_bins))
        if self.pitch_activation.shape[1] == n_bins:
            return
        new_freqs = _make_crepe_freq_axis(n_bins)
        new_pitch = np.zeros((self.n_rows, n_bins), dtype=np.float32)
        min_cols = min(self.pitch_activation.shape[1], n_bins)
        if min_cols > 0:
            new_pitch[:, :min_cols] = self.pitch_activation[:, :min_cols]
        self.pitch_activation = new_pitch
        self.pitch_freq_bins = new_freqs
        self.pitch_im.set_data(self.pitch_activation)
        self.pitch_im.set_extent(
            (
                self.pitch_freq_bins[0],
                self.pitch_freq_bins[-1],
                -self.window_sec,
                0.0,
            )
        )

    def _run_crepe_once(self):
        if not self.enable_pitch:
            return np.nan, 0.0, None
        now = time.time()
        if now - self._last_crepe_run < self._crepe_min_interval:
            return None
        if self.td_buf.size < int(self.crepe_win_sec * self.sr):
            return np.nan, 0.0, None
        self._last_crepe_run = now
        nwin = int(self.crepe_win_sec * self.sr)
        x = self.td_buf[-nwin:].astype(np.float32)
        try:
            _, f0, c, activation = crepe.predict(
                x,
                self.sr,
                step_size=self.crepe_step_ms,
                model_capacity=self.crepe_capacity,
                viterbi=True,
                verbose=0,
            )
            f0_last = float(f0[-1]) if f0.size else float("nan")
            c_last = float(c[-1]) if c.size else 0.0
            if not np.isfinite(f0_last) or f0_last <= 0:
                f0_last, c_last = np.nan, 0.0
            act_last = None
            if activation is not None:
                activation = np.asarray(activation)
                act_last = activation[-1] if activation.ndim >= 1 else activation
            return f0_last, c_last, act_last
        except Exception:
            self.enable_pitch = False
            return np.nan, 0.0, None

    def _update_pitch_plot(self):
        result = self._run_crepe_once()
        if result is None:  # throttled
            return
        _, _, activation = result
        if activation is None:
            activation = np.zeros(self.pitch_activation.shape[1], dtype=np.float32)
        activation = np.asarray(activation, dtype=np.float32).reshape(-1)
        if activation.size == 0:
            activation = np.zeros(self.pitch_activation.shape[1], dtype=np.float32)
        if activation.size != self.pitch_activation.shape[1]:
            self._ensure_pitch_activation_bins(activation.size)
        activation = np.nan_to_num(activation, nan=0.0, posinf=1.0, neginf=0.0)
        activation = np.clip(activation, 0.0, 1.0)
        # Scroll UP: roll rows left, newest row at end (top in plot via extent)
        self.pitch_activation = np.roll(self.pitch_activation, -1, axis=0)
        self.pitch_activation[-1, :] = activation
        self.pitch_im.set_data(self.pitch_activation)

    # -------- Main processing --------
    def process_new_samples(self, samples: np.ndarray):
        if samples.size == 0:
            return
        self.sample_buf = np.concatenate([self.sample_buf, samples])
        self.td_buf = np.concatenate([self.td_buf, samples])
        keep_td = max(self.ac_win_samps, int(self.crepe_win_sec * self.sr))
        if self.td_buf.size > keep_td:
            self.td_buf = self.td_buf[-keep_td:]

        while self.sample_buf.size >= self.fft_size:
            frame = self.sample_buf[: self.fft_size]
            self.sample_buf = self.sample_buf[self.hop :]  # advance by hop

            mag = self._spec_mag(frame)
            mag = self._apply_noise_filter(mag)
            S_db = dbfs(mag)

            # Spectrogram scroll UP: roll rows left, insert new row at end (top)
            self.S = np.roll(self.S, -1, axis=0)
            self.S[-1, : len(S_db)] = S_db
            if len(S_db) < self.S.shape[1]:
                self.S[-1, len(S_db) :] = -120.0

            self.last_mag_db = S_db.copy()
            self.last_mag_lin = mag.copy()

    def update_plot(self):
        # Spectrogram scaling
        recent = self.S[-min(50, self.S.shape[0]) :, :]
        vmax = np.max(recent)
        vmax = -20.0 if not np.isfinite(vmax) else vmax
        self.im.set_data(self.S)
        self.im.set_clim(vmin=vmax - self.db_range, vmax=vmax)

        # Spectrum
        if self.last_mag_db is not None and self.last_mag_db.size == self.max_bin:
            self.line_plot.set_data(self.line_freqs, self.last_mag_db)
            self.ax_line.set_ylim(vmax - self.db_range, vmax)

        # ACFs
        self._update_acf_plot()
        self._update_facf_plot()
        self._update_facf2_plot()

        # Pitch
        if self.enable_pitch:
            self._update_pitch_plot()

        self.fig.canvas.draw_idle()

    def run(self):
        self.source.start()
        if self.noise_filter_enabled:
            self.profile_noise(self.noise_sec, show_status=True)
        try:

            def _on_timer(_):
                if self.paused:
                    return
                drained = 0
                while True:
                    chunk = self.source.read()
                    if chunk.size == 0:
                        break
                    self.process_new_samples(chunk)
                    drained += 1
                    if drained > 12:
                        break
                self.update_plot()

            timer = self.fig.canvas.new_timer(interval=30)  # ~33 FPS
            timer.add_callback(_on_timer, None)
            timer.start()
            plt.show()
        finally:
            self.source.stop()


def parse_args():
    ap = argparse.ArgumentParser(
        description="Scrolling Spectrogram + Spectrum + ACFs + CREPE Pitch"
    )
    ap.add_argument("--samplerate", type=int, default=44100)
    ap.add_argument("--fft", type=int, default=8192)
    ap.add_argument("--hop", type=int, default=512)
    ap.add_argument("--window", type=float, default=5.0)
    ap.add_argument("--max-freq", type=float, default=2000.0)
    ap.add_argument("--db-range", type=float, default=80.0)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--noise-sec", type=float, default=2.0)
    ap.add_argument("--over-sub", type=float, default=1.0)
    ap.add_argument("--no-noise-filter", action="store_true")
    ap.add_argument("--min-freq", type=float, default=10.0)
    # CREPE
    ap.add_argument("--no-pitch", action="store_true")
    ap.add_argument(
        "--crepe-capacity",
        type=str,
        default="small",
        choices=["tiny", "small", "medium", "large", "full"],
    )
    ap.add_argument("--crepe-step-ms", type=int, default=20)
    ap.add_argument("--pitch-win", type=float, default=0.5)
    return ap.parse_args()


def main():
    args = parse_args()
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

    vis = ScrollingSpectrogram(
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
        min_freq=args.min_freq,
        ac_win_sec=0.5,
        enable_pitch=(not args.no_pitch),
        crepe_capacity=args.crepe_capacity,
        crepe_step_ms=args.crepe_step_ms,
        crepe_win_sec=args.pitch_win,
    )
    vis.run()


if __name__ == "__main__":
    main()
