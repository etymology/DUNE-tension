"""Matplotlib-based scrolling spectrogram visualizer."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from audio_sources import AudioSource
from utils import dbfs, hann_window


@dataclass
class SpectrogramConfig:
    samplerate: int = 44100
    fft_size: int = 8192
    hop: int = 512
    window_sec: float = 5.0
    max_freq: float = 2000.0
    db_range: float = 70.0
    enable_noise_filter: bool = True
    noise_sec: float = 2.0
    over_sub: float = 1.0
    min_freq: float = 10.0
    ac_win_sec: float = 0.5
    snr_trigger: float = 3.0
    snr_release: float = 3.0
    pre_record_sec: float = 0.1


class ScrollingSpectrogram:
    """Interactive matplotlib spectrogram visualizer."""

    def __init__(
        self, source: AudioSource, config: Optional[SpectrogramConfig] = None
    ) -> None:
        if config is None:
            config = SpectrogramConfig()
        self.config = config

        self.source = source
        self.sr = int(config.samplerate)
        self.fft_size = int(config.fft_size)
        self.hop = int(config.hop)
        self.window_sec = float(config.window_sec)
        self.max_freq = float(config.max_freq)
        self.db_range = float(config.db_range)

        self.noise_filter_enabled = bool(config.enable_noise_filter)
        self.noise_sec = float(config.noise_sec)
        self.over_sub = float(config.over_sub)
        self.noise_mag: Optional[np.ndarray] = None

        self.win = hann_window(self.fft_size)
        self.freqs = np.fft.rfftfreq(self.fft_size, d=1.0 / self.sr)
        self.max_bin = np.searchsorted(self.freqs, self.max_freq, side="right")
        if self.max_bin < 2:
            self.max_bin = len(self.freqs)

        self.n_rows = max(10, int(round(self.window_sec * self.sr / self.hop)))
        self.S = np.full((self.n_rows, self.max_bin), -120.0, dtype=np.float32)

        self.sample_buf = np.zeros(0, dtype=np.float32)
        self.td_buf = np.zeros(0, dtype=np.float32)
        self.ac_win_sec = float(config.ac_win_sec)
        self.ac_win_samps = int(round(self.ac_win_sec * self.sr))
        self.min_freq = max(1e-6, float(config.min_freq))
        self.acf_max_lag_s = 1.0 / self.min_freq
        self.acf_max_lag_samps = max(1, int(round(self.acf_max_lag_s * self.sr)))

        self.snr_trigger = float(getattr(config, "snr_trigger", 3.0))
        self.snr_release = float(getattr(config, "snr_release", self.snr_trigger))
        self.pre_record_sec = float(getattr(config, "pre_record_sec", 0.1))
        self.pre_record_samples = max(0, int(round(self.pre_record_sec * self.sr)))
        self.pre_buffer = np.zeros(0, dtype=np.float32)
        self.recording_active = False
        self.recorded_samples = np.zeros(0, dtype=np.float32)
        self._use_full_analysis = False
        self._mag_accum = np.zeros(self.max_bin, dtype=np.float64)
        self._frame_count = 0
        self.noise_rms = 1e-6

        self.fig = plt.figure(figsize=(14, 12))
        gs = self.fig.add_gridspec(
            nrows=5, ncols=1, height_ratios=[3, 1, 1, 1, 1], hspace=0.55
        )
        self.ax_spec = self.fig.add_subplot(gs[0, 0])
        self.ax_line = self.fig.add_subplot(gs[1, 0])
        self.ax_acf = self.fig.add_subplot(gs[2, 0])
        self.ax_facf = self.fig.add_subplot(gs[3, 0])
        self.ax_facf2 = self.fig.add_subplot(gs[4, 0])

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

        self.line_freqs = self.freqs[: self.max_bin]
        self.last_mag_db = np.full_like(self.line_freqs, -120.0, dtype=np.float32)
        self.last_mag_lin = np.zeros_like(self.line_freqs, dtype=np.float32)
        (self.line_plot,) = self.ax_line.plot(self.line_freqs, self.last_mag_db, lw=1.0)
        self.ax_line.set_xlim(0, self.freqs[self.max_bin - 1])
        self.ax_line.set_ylim(-120, 0)
        self.ax_line.set_xlabel("Frequency (Hz)")
        self.ax_line.set_ylabel("dBFS")
        self.ax_line.grid(True, alpha=0.25, which="both")

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

        self._rebuild_facf_axis()

        self._set_titles()
        self.paused = False
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self._reset_processing_state()

    # ------------------------------------------------------------------
    # Event handlers & UI updates
    # ------------------------------------------------------------------
    def _set_titles(self) -> None:
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

    def on_key(self, event) -> None:
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

    def _reset_processing_state(self) -> None:
        self.sample_buf = np.zeros(0, dtype=np.float32)
        self.td_buf = np.zeros(0, dtype=np.float32)
        if self.S.size:
            self.S[:, :] = -120.0
        self.last_mag_db = np.full_like(self.line_freqs, -120.0, dtype=np.float32)
        self.last_mag_lin = np.zeros_like(self.line_freqs, dtype=np.float32)
        if hasattr(self, "acf_vals") and self.acf_vals.size:
            self.acf_vals[:] = 0.0
        if hasattr(self, "facf_vals") and self.facf_vals.size:
            self.facf_vals[:] = 0.0
        if hasattr(self, "facf2_vals") and self.facf2_vals.size:
            self.facf2_vals[:] = 0.0
        self._mag_accum = np.zeros(self.max_bin, dtype=np.float64)
        self._frame_count = 0
        self._use_full_analysis = False

    def _update_pre_buffer(self, samples: np.ndarray) -> None:
        if self.pre_record_samples <= 0 or samples.size == 0:
            return
        joined = (
            samples.astype(np.float32, copy=False)
            if self.pre_buffer.size == 0
            else np.concatenate([self.pre_buffer, samples]).astype(
                np.float32, copy=False
            )
        )
        if joined.size > self.pre_record_samples:
            joined = joined[-self.pre_record_samples :]
        self.pre_buffer = joined

    def _compute_snr(self, samples: np.ndarray) -> float:
        if samples.size == 0:
            return 0.0
        rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
        noise = float(self.noise_rms) if self.noise_rms and self.noise_rms > 0 else 1e-6
        return rms / noise if noise > 0 else float("inf")

    def _start_recording(self) -> None:
        self.recording_active = True
        self._reset_processing_state()
        self.recorded_samples = np.zeros(0, dtype=np.float32)
        self._use_full_analysis = False

    def _append_recording(self, samples: np.ndarray) -> bool:
        if samples.size == 0:
            return False
        data = samples.astype(np.float32, copy=False)
        if self.recorded_samples.size == 0:
            self.recorded_samples = data.copy()
        else:
            self.recorded_samples = np.concatenate([self.recorded_samples, data])
        self.process_new_samples(data)
        return True

    def _finalize_recording(self) -> None:
        if self.recorded_samples.size == 0:
            self._use_full_analysis = False
            return
        if self._frame_count > 0:
            avg_mag = (self._mag_accum / max(1, self._frame_count)).astype(np.float32)
        else:
            frame = self.recorded_samples[: self.fft_size]
            if frame.size < self.fft_size:
                frame = np.pad(frame, (0, self.fft_size - frame.size))
            avg_mag = self._apply_noise_filter(self._spec_mag(frame)).astype(np.float32)
        avg_mag = np.maximum(avg_mag, 0.0)
        if avg_mag.size != self.max_bin:
            if avg_mag.size > self.max_bin:
                avg_mag = avg_mag[: self.max_bin]
            else:
                avg_mag = np.pad(avg_mag, (0, self.max_bin - avg_mag.size))
        self.last_mag_lin = avg_mag
        self.last_mag_db = dbfs(np.maximum(avg_mag, 1e-12))
        self._use_full_analysis = True
        self._mag_accum = np.zeros(self.max_bin, dtype=np.float64)
        self._frame_count = 0

    def _stop_recording(self) -> bool:
        if not self.recording_active:
            return False
        self.recording_active = False
        self._finalize_recording()
        return True

    def _handle_chunk(self, chunk: np.ndarray) -> bool:
        if chunk.size == 0:
            return False
        chunk = chunk.astype(np.float32, copy=False)
        pre_snapshot = self.pre_buffer.copy()
        self._update_pre_buffer(chunk)
        snr = self._compute_snr(chunk)
        changed = False
        if self.recording_active:
            changed = self._append_recording(chunk)
            if snr < self.snr_release:
                changed = self._stop_recording() or changed
        else:
            if snr >= self.snr_trigger:
                self._start_recording()
                changed = True
                if pre_snapshot.size:
                    changed = self._append_recording(pre_snapshot) or changed
                changed = self._append_recording(chunk) or changed
        return changed

    def _update_freq_axis(self) -> None:
        self.max_bin = np.searchsorted(self.freqs, self.max_freq, side="right")
        self.max_bin = max(2, min(self.max_bin, len(self.freqs)))

        new_S = np.full((self.S.shape[0], self.max_bin), -120.0, dtype=np.float32)
        copy_bins = min(self.S.shape[1], self.max_bin)
        new_S[:, :copy_bins] = self.S[:, :copy_bins]
        self.S = new_S

        self.im.set_data(self.S)
        self.im.set_extent((0.0, self.freqs[self.max_bin - 1], -self.window_sec, 0.0))

        self.line_freqs = self.freqs[: self.max_bin]
        if self.last_mag_db.size >= self.max_bin:
            self.last_mag_db = self.last_mag_db[: self.max_bin]
        else:
            self.last_mag_db = np.pad(
                self.last_mag_db,
                (0, self.max_bin - self.last_mag_db.size),
                constant_values=-120.0,
            )
        if self.last_mag_lin.size >= self.max_bin:
            self.last_mag_lin = self.last_mag_lin[: self.max_bin]
        else:
            self.last_mag_lin = np.pad(
                self.last_mag_lin,
                (0, self.max_bin - self.last_mag_lin.size),
                constant_values=0.0,
            )
        if hasattr(self, "_mag_accum"):
            if self._mag_accum.size >= self.max_bin:
                self._mag_accum = self._mag_accum[: self.max_bin]
            else:
                self._mag_accum = np.pad(
                    self._mag_accum,
                    (0, self.max_bin - self._mag_accum.size),
                    constant_values=0.0,
                )
        self.line_plot.set_data(self.line_freqs, self.last_mag_db)
        self.ax_line.set_xlim(0, self.freqs[self.max_bin - 1])

        self._rebuild_facf_axis()
        self.fig.canvas.draw_idle()

    def _update_time_axis(self) -> None:
        new_rows = max(10, int(round(self.window_sec * self.sr / self.hop)))
        if new_rows != self.S.shape[0]:
            new_S = np.full((new_rows, self.S.shape[1]), -120.0, dtype=np.float32)
            copy_rows = min(self.S.shape[0], new_rows)
            new_S[-copy_rows:, :] = self.S[-copy_rows:, :]
            self.S = new_S
        self.n_rows = self.S.shape[0]
        self.im.set_extent((0.0, self.freqs[self.max_bin - 1], -self.window_sec, 0.0))
        self.fig.canvas.draw_idle()

    def _update_clim(self) -> None:
        vmax = np.max(self.S)
        vmax = -20.0 if not np.isfinite(vmax) else vmax
        self.im.set_clim(vmin=vmax - self.db_range, vmax=vmax)
        self.ax_line.set_ylim(vmax - self.db_range, vmax)
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # DSP helpers
    # ------------------------------------------------------------------
    def _spec_mag(self, frame: np.ndarray) -> np.ndarray:
        fw = frame * self.win
        spec = np.fft.rfft(fw, n=self.fft_size)
        mag = np.abs(spec)
        return mag[: self.max_bin]

    def profile_noise(self, seconds: float, show_status: bool = False) -> None:
        if show_status:
            self.ax_spec.set_title("Profiling noise... Please keep the room quiet.")
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

        target_samples = int(self.sr * max(0.2, seconds))
        buf = np.zeros(0, dtype=np.float32)
        mags = []
        noise_chunks: list[np.ndarray] = []
        t0 = time.time()
        timeout = max(3.0, seconds * 3.0)
        collected = 0
        while collected < target_samples and (time.time() - t0) < timeout:
            chunk = self.source.read()
            if chunk.size == 0:
                continue
            buf = np.concatenate([buf, chunk])
            noise_chunks.append(chunk.astype(np.float32, copy=False))
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
        if noise_chunks:
            noise_td = np.concatenate(noise_chunks)
            self.noise_rms = float(np.sqrt(np.mean(noise_td.astype(np.float64) ** 2)))
        else:
            self.noise_rms = 1e-6
        self._set_titles()
        self.fig.canvas.draw_idle()

    def _apply_noise_filter(self, mag: np.ndarray) -> np.ndarray:
        if not self.noise_filter_enabled or self.noise_mag is None:
            return mag
        return np.maximum(mag - self.over_sub * self.noise_mag, 0.0)

    def _find_local_peaks(self, y: np.ndarray) -> np.ndarray:
        if y.size < 3:
            return np.array([], dtype=int)
        return np.where((y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:]))[0] + 1

    def _compute_prominence(self, y: np.ndarray, peaks: np.ndarray) -> np.ndarray:
        if peaks.size == 0:
            return np.zeros(0, dtype=np.float32)

        prominences = np.zeros(peaks.size, dtype=np.float32)
        n = y.size
        for i, peak_idx in enumerate(peaks):
            height = float(y[peak_idx])

            left_min = height
            j = int(peak_idx)
            while j > 0:
                j -= 1
                if y[j] > height:
                    break
                if y[j] < left_min:
                    left_min = float(y[j])

            right_min = height
            j = int(peak_idx)
            while j < n - 1:
                j += 1
                if y[j] > height:
                    break
                if y[j] < right_min:
                    right_min = float(y[j])

            prominences[i] = height - max(left_min, right_min)

        return prominences

    def _update_acf_plot(self) -> None:
        if self._use_full_analysis and self.recorded_samples.size >= 2:
            x = self.recorded_samples.astype(np.float32)
        else:
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
                prominences = self._compute_prominence(self.acf_vals[:L], peaks)
                order = np.argsort(prominences)[::-1]
                gi = int(peaks[order[0]])
                peak_freq = float(self.acf_freq_axis_hz[gi])
                if peaks.size >= 2:
                    p1 = int(peaks[order[0]])
                    p2 = int(peaks[order[1]])
                    p2p = abs(
                        float(self.acf_freq_axis_hz[p1] - self.acf_freq_axis_hz[p2])
                    )
                else:
                    p2p = float("nan")

        if self.acf_freq_axis_hz.size > 0 and np.any(np.isfinite(self.acf_vals)):
            mask_finite = np.isfinite(self.acf_vals)
            vals = self.acf_vals[mask_finite]
            freqs_plot = self.acf_freq_axis_hz[mask_finite]
            if vals.size > 0:
                peaks = self._find_local_peaks(vals)
                if peaks.size > 0:
                    prominences = self._compute_prominence(vals, peaks)
                    gi = int(peaks[int(np.argmax(prominences))])
                else:
                    gi = int(np.argmax(vals))
                peak_freq = float(freqs_plot[gi])
                ymax = float(vals[gi])
                ymin = float(np.min(vals))
                dheight = max(0.0, min(2.0, ymax - ymin))
            else:
                peak_freq = float("nan")
                dheight = float("nan")
        else:
            peak_freq = float("nan")
            dheight = float("nan")

        self.acf_plot.set_data(self.acf_freq_axis_hz, self.acf_vals)
        label = f"ACF  |  prom≈{peak_freq:.1f} Hz  |  Δheight≈{dheight:.2f}"
        self.acf_plot.set_label(label)
        if hasattr(self, "_acf_legend") and self._acf_legend:
            self._acf_legend.remove()
        self._acf_legend = self.ax_acf.legend(loc="upper right", framealpha=0.3)

    def _rebuild_facf_axis(self) -> None:
        df = self.freqs[1] - self.freqs[0] if len(self.freqs) > 1 else 1.0
        shift_min_bins = max(1, int(np.ceil(self.min_freq / df)))
        shift_max_bins = max(shift_min_bins, int(np.floor(self.max_freq / df)))

        self.facf_shift_axis_hz = (
            np.arange(shift_min_bins, shift_max_bins + 1) * df
        ).astype(np.float32)
        if self.facf_shift_axis_hz.size == 0:
            self.facf_shift_axis_hz = np.array([self.min_freq], dtype=np.float32)

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

    def _update_facf_plot(self) -> None:
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

        peak_hz = float("nan")
        peak_val = float("nan")
        if self.facf_vals.size > 0:
            mask = (
                np.isfinite(self.facf_vals)
                & (self.facf_shift_axis_hz >= self.min_freq)
                & (self.facf_shift_axis_hz <= self.max_freq)
            )
            if np.any(mask):
                vals = self.facf_vals[mask]
                freqs = self.facf_shift_axis_hz[mask]
                gi = int(np.argmax(vals))
                peak_hz = float(freqs[gi])
                peak_val = float(vals[gi])
        if np.isfinite(peak_hz):
            label = f"FACF  |  max≈{peak_hz:.1f} Hz  |  val≈{peak_val:.2f}"
        else:
            label = "FACF"
        self.facf_plot.set_label(label)
        if hasattr(self, "_facf_legend") and self._facf_legend:
            self._facf_legend.remove()
        self._facf_legend = self.ax_facf.legend(loc="upper right", framealpha=0.3)

    def _update_facf2_plot(self) -> None:
        if not hasattr(self, "facf_vals") or self.facf_vals.size < 3:
            return

        y = self.facf_vals.astype(np.float64)
        mask_finite = np.isfinite(y)
        if not np.any(mask_finite):
            self.facf2_vals[:] = 0.0
            self.facf2_plot.set_data(self.facf_shift_axis_hz, self.facf2_vals)
            return
        y = y[mask_finite]

        y = y - y.mean()
        var = np.dot(y, y)
        if var <= 1e-18 or y.size < 3:
            self.facf2_vals[:] = 0.0
        else:
            acf_full = np.correlate(y, y, mode="full")
            acf_pos = acf_full[len(y) - 1 :]
            acf0 = acf_pos[0] if acf_pos[0] != 0 else 1.0

            df = (
                self.facf_shift_axis_hz[1] - self.facf_shift_axis_hz[0]
                if len(self.facf_shift_axis_hz) > 1
                else 1.0
            )
            shift_bins = np.round(self.facf_shift_axis_hz / df).astype(int)
            shift_bins = np.clip(shift_bins, 1, len(acf_pos) - 1)

            vals = (acf_pos[shift_bins] / acf0).astype(np.float32)
            if vals.size != self.facf2_vals.size:
                self.facf2_vals = np.resize(self.facf2_vals, vals.shape)
                self.facf2_plot.set_data(self.facf_shift_axis_hz, self.facf2_vals)
            self.facf2_vals[:] = vals

        self.facf2_plot.set_data(self.facf_shift_axis_hz, self.facf2_vals)
        peak_hz = float("nan")
        peak_val = float("nan")
        if self.facf2_vals.size > 0:
            mask = (
                np.isfinite(self.facf2_vals)
                & (self.facf_shift_axis_hz >= self.min_freq)
                & (self.facf_shift_axis_hz <= self.max_freq)
            )
            if np.any(mask):
                vals = self.facf2_vals[mask]
                freqs = self.facf_shift_axis_hz[mask]
                gi = int(np.argmax(vals))
                peak_hz = float(freqs[gi])
                peak_val = float(vals[gi])
        if np.isfinite(peak_hz):
            label = f"FACF²  |  max≈{peak_hz:.1f} Hz  |  val≈{peak_val:.2f}"
        else:
            label = "FACF²"
        self.facf2_plot.set_label(label)
        if hasattr(self, "_facf2_legend") and self._facf2_legend:
            self._facf2_legend.remove()
        self._facf2_legend = self.ax_facf2.legend(loc="upper right", framealpha=0.3)

    # ------------------------------------------------------------------
    # Main processing loop
    # ------------------------------------------------------------------
    def process_new_samples(self, samples: np.ndarray) -> None:
        if samples.size == 0:
            return
        self.sample_buf = np.concatenate([self.sample_buf, samples])
        self.td_buf = np.concatenate([self.td_buf, samples])
        if self.td_buf.size > self.ac_win_samps:
            self.td_buf = self.td_buf[-self.ac_win_samps :]

        while self.sample_buf.size >= self.fft_size:
            frame = self.sample_buf[: self.fft_size]
            self.sample_buf = self.sample_buf[self.hop :]

            mag = self._spec_mag(frame)
            mag = self._apply_noise_filter(mag)
            S_db = dbfs(mag)

            if self.recording_active:
                if self._mag_accum.size != self.max_bin:
                    self._mag_accum = np.zeros(self.max_bin, dtype=np.float64)
                self._mag_accum[: len(mag)] += mag
                self._frame_count += 1

            self.S = np.roll(self.S, -1, axis=0)
            self.S[-1, : len(S_db)] = S_db
            if len(S_db) < self.S.shape[1]:
                self.S[-1, len(S_db) :] = -120.0

            self.last_mag_db = S_db.copy()
            self.last_mag_lin = mag.copy()

    def update_plot(self) -> None:
        recent = self.S[-min(50, self.S.shape[0]) :, :]
        vmax = np.max(recent)
        vmax = -20.0 if not np.isfinite(vmax) else vmax
        self.im.set_data(self.S)
        self.im.set_clim(vmin=vmax - self.db_range, vmax=vmax)

        if self.last_mag_db is not None and self.last_mag_db.size == self.max_bin:
            self.line_plot.set_data(self.line_freqs, self.last_mag_db)
            self.ax_line.set_ylim(vmax - self.db_range, vmax)

        self._update_acf_plot()
        self._update_facf_plot()
        self._update_facf2_plot()

        self.fig.canvas.draw_idle()

    def run(self) -> None:
        self.source.start()
        if self.noise_filter_enabled:
            self.profile_noise(self.noise_sec, show_status=True)
        try:

            def _on_timer(_):
                if self.paused:
                    return
                drained = 0
                updated = False
                while True:
                    chunk = self.source.read()
                    if chunk.size == 0:
                        break
                    updated = self._handle_chunk(chunk) or updated
                    drained += 1
                    if drained > 12:
                        break
                if updated or self.recording_active:
                    self.update_plot()

            timer = self.fig.canvas.new_timer(interval=30)
            timer.add_callback(_on_timer, None)
            timer.start()
            plt.show()
        finally:
            self.source.stop()


__all__ = ["SpectrogramConfig", "ScrollingSpectrogram"]
