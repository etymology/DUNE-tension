"""Embedded live plot support for the Tkinter GUI."""

from __future__ import annotations

import threading
from typing import Any
import tkinter as tk

import numpy as np
from matplotlib.figure import Figure

from dune_tension.summaries import build_summary_plot_figure_for_config
from spectrum_analysis.pitch_compare_config import PitchCompareConfig

try:  # pragma: no cover - backend availability depends on the runtime environment
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
except Exception:  # pragma: no cover - fall back to text placeholders in tests/headless
    FigureCanvasTkAgg = None  # type: ignore[assignment]

LIVE_SUMMARY_FIGSIZE = (7.8, 3.6)
LIVE_WAVEFORM_FIGSIZE = (7.2, 4.6)


class LivePlotManager:
    """Own the embedded summary and waveform plot canvases."""

    def __init__(
        self,
        root: tk.Misc,
        summary_parent: tk.Misc,
        waveform_parent: tk.Misc,
    ) -> None:
        self.root = root
        self.summary_parent = summary_parent
        self.waveform_parent = waveform_parent
        self.summary_canvas: Any | None = None
        self.waveform_canvas: Any | None = None
        self.summary_placeholder = tk.Label(
            summary_parent,
            text="Waiting for summary data...",
            anchor="w",
            justify="left",
        )
        self.summary_placeholder.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        self.waveform_placeholder = tk.Label(
            waveform_parent,
            text="Waiting for audio capture...",
            anchor="w",
            justify="left",
        )
        self.waveform_placeholder.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        self._summary_after_id: Any | None = None

    def publish_waveform(
        self,
        audio_sample: Any,
        samplerate: int,
        analysis: Any | None = None,
    ) -> None:
        """Render the latest captured waveform and diagnostics on the Tk thread."""

        try:
            waveform = np.asarray(audio_sample, dtype=float).reshape(-1)
        except Exception:
            return
        if waveform.size == 0:
            return

        self._on_tk_thread(
            lambda: self._render_waveform(waveform, int(samplerate), analysis)
        )

    def request_summary_refresh(self, config: Any, delay_ms: int = 100) -> None:
        """Debounce and redraw the current summary figure for ``config``."""

        def schedule() -> None:
            if self._summary_after_id is not None:
                try:
                    self.root.after_cancel(self._summary_after_id)
                except Exception:
                    pass
            self._summary_after_id = self.root.after(
                delay_ms,
                lambda: self._render_summary(config),
            )

        self._on_tk_thread(schedule)

    def _render_summary(self, config: Any) -> None:
        self._summary_after_id = None
        if FigureCanvasTkAgg is None:
            self._set_placeholder(
                self.summary_placeholder,
                "Matplotlib Tk backend unavailable.\nSummary plots cannot be embedded.",
            )
            return

        try:
            figure = build_summary_plot_figure_for_config(
                config,
                figsize=LIVE_SUMMARY_FIGSIZE,
            )
        except Exception as exc:
            self._set_placeholder(
                self.summary_placeholder,
                f"Failed to render summary plot:\n{exc}",
            )
            return

        if figure is None:
            self._set_placeholder(
                self.summary_placeholder,
                "No summary data available for the selected APA/layer.",
            )
            if self.summary_canvas is not None:
                self.summary_canvas.get_tk_widget().destroy()
                self.summary_canvas = None
            return

        self._show_canvas("summary", figure)

    def _render_waveform(
        self,
        waveform: np.ndarray,
        samplerate: int,
        analysis: Any | None,
    ) -> None:
        if FigureCanvasTkAgg is None:
            self._set_placeholder(
                self.waveform_placeholder,
                "Matplotlib Tk backend unavailable.\nWaveform plots cannot be embedded.",
            )
            return

        figure = self._build_audio_diagnostics_figure(waveform, samplerate, analysis)
        self._show_canvas("waveform", figure)

    def _show_canvas(self, kind: str, figure: Figure) -> None:
        parent = self.summary_parent if kind == "summary" else self.waveform_parent
        placeholder = (
            self.summary_placeholder if kind == "summary" else self.waveform_placeholder
        )
        current_canvas = self.summary_canvas if kind == "summary" else self.waveform_canvas

        if current_canvas is not None:
            try:
                current_canvas.get_tk_widget().destroy()
            except Exception:
                pass

        if hasattr(placeholder, "grid_remove"):
            placeholder.grid_remove()
        else:
            placeholder.grid_forget()

        canvas = FigureCanvasTkAgg(figure, master=parent)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        if kind == "summary":
            self.summary_canvas = canvas
        else:
            self.waveform_canvas = canvas

    def _set_placeholder(self, placeholder: tk.Label, text: str) -> None:
        try:
            placeholder.configure(text=text)
            placeholder.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        except Exception:
            return

    def _on_tk_thread(self, callback) -> None:
        try:
            if threading.current_thread() is threading.main_thread():
                callback()
            else:
                self.root.after(0, callback)
        except Exception:
            return

    @staticmethod
    def _build_audio_diagnostics_figure(
        waveform: np.ndarray,
        samplerate: int,
        analysis: Any | None,
    ) -> Figure:
        figure = Figure(figsize=LIVE_WAVEFORM_FIGSIZE, constrained_layout=True)
        grid = figure.add_gridspec(2, 2, height_ratios=[2.2, 1.6], hspace=0.16, wspace=0.12)
        waveform_axis = figure.add_subplot(grid[0, :])
        fft_axis = figure.add_subplot(grid[1, 0])
        activation_axis = figure.add_subplot(grid[1, 1])

        stride = max(1, waveform.size // 4000)
        shown = waveform[::stride]
        if samplerate > 0:
            x_axis = (np.arange(shown.size) * stride) / float(samplerate)
            x_label = "Time (s)"
        else:
            x_axis = np.arange(shown.size) * stride
            x_label = "Sample Index"

        waveform_axis.plot(x_axis, shown, linewidth=1.0, color="#1f77b4")
        waveform_axis.set_title("Latest Captured Waveform")
        waveform_axis.set_xlabel(x_label)
        waveform_axis.set_ylabel("Amplitude")
        waveform_axis.grid(True, linestyle=":", linewidth=0.5, color="gray")

        LivePlotManager._populate_fft_axis(fft_axis, waveform, samplerate)
        LivePlotManager._populate_pesto_axis(activation_axis, analysis)
        return figure

    @staticmethod
    def _populate_fft_axis(axis: Any, waveform: np.ndarray, samplerate: int) -> None:
        cfg = PitchCompareConfig()
        if waveform.size == 0:
            axis.text(0.5, 0.5, "No FFT data.", ha="center", va="center", transform=axis.transAxes)
            axis.set_title("FFT")
            return

        window = np.hanning(waveform.size)
        spectrum = np.fft.rfft(waveform * window)
        freqs = np.fft.rfftfreq(waveform.size, d=1.0 / max(float(samplerate), 1.0))
        magnitudes = np.abs(spectrum)
        magnitudes_db = 20.0 * np.log10(np.maximum(magnitudes, 1e-9))

        axis.plot(freqs, magnitudes_db, color="#d95f02", linewidth=1.0)
        axis.set_title("FFT")
        axis.set_xlabel("Frequency (Hz)")
        axis.set_ylabel("Magnitude (dB)")
        axis.grid(True, linestyle=":", linewidth=0.5, color="gray")
        if freqs.size:
            axis.set_xlim(0.0, min(float(freqs[-1]), float(cfg.max_frequency)))

    @staticmethod
    def _populate_pesto_axis(axis: Any, analysis: Any | None) -> None:
        cfg = PitchCompareConfig()
        activation_map = getattr(analysis, "activation_map", None)
        activation_freq_axis = getattr(analysis, "activation_freq_axis", None)
        frame_times = getattr(analysis, "frame_times", None)
        predicted_frequencies = getattr(analysis, "predicted_frequencies", None)
        expected_frequency = getattr(analysis, "expected_frequency", None)

        axis.set_title("PESTO Activations")
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("Frequency (Hz)")

        if (
            activation_map is None
            or activation_freq_axis is None
            or frame_times is None
            or np.asarray(activation_map).size == 0
            or np.asarray(activation_freq_axis).size == 0
            or np.asarray(frame_times).size == 0
        ):
            axis.text(
                0.5,
                0.5,
                "PESTO activation data unavailable.",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            return

        activation_map = np.asarray(activation_map, dtype=np.float32)
        activation_freq_axis = np.asarray(activation_freq_axis, dtype=np.float32)
        frame_times = np.asarray(frame_times, dtype=np.float32)
        mask = (activation_freq_axis >= cfg.min_frequency) & (
            activation_freq_axis <= cfg.max_frequency
        )
        if not np.any(mask):
            axis.text(
                0.5,
                0.5,
                "No activation bins within frequency range.",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            axis.set_ylim(cfg.min_frequency, cfg.max_frequency)
            return

        visible_freqs = activation_freq_axis[mask]
        visible_activation = activation_map[mask, :]
        visible_max_frequency = LivePlotManager._expected_frequency_max_frequency(
            expected_frequency,
            fallback_max=float(cfg.max_frequency),
        )
        axis.pcolormesh(
            frame_times,
            visible_freqs,
            visible_activation,
            shading="nearest",
            cmap="viridis",
        )
        if predicted_frequencies is not None:
            predicted_frequencies = np.asarray(predicted_frequencies, dtype=np.float32).reshape(-1)
            if predicted_frequencies.size == frame_times.size:
                valid = np.isfinite(predicted_frequencies) & (predicted_frequencies > 0.0)
                valid &= predicted_frequencies >= float(cfg.min_frequency)
                valid &= predicted_frequencies <= float(cfg.max_frequency)
                if np.any(valid):
                    axis.plot(
                        frame_times[valid],
                        predicted_frequencies[valid],
                        color="cyan",
                        linewidth=1.0,
                    )
        axis.set_ylim(cfg.min_frequency, visible_max_frequency)
        axis.grid(False)

    @staticmethod
    def _expected_frequency_max_frequency(
        expected_frequency: float | None,
        *,
        fallback_max: float,
    ) -> float:
        if expected_frequency is None:
            return float(fallback_max)
        try:
            limit = float(expected_frequency) * 2.0
        except (TypeError, ValueError):
            return float(fallback_max)
        if not np.isfinite(limit) or limit <= 0.0:
            return float(fallback_max)
        return float(min(fallback_max, limit))
