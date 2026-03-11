"""Embedded live plot support for the Tkinter GUI."""

from __future__ import annotations

import threading
from typing import Any
import tkinter as tk

import numpy as np
from matplotlib.figure import Figure

from dune_tension.summaries import build_summary_plot_figure_for_config

try:  # pragma: no cover - backend availability depends on the runtime environment
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
except Exception:  # pragma: no cover - fall back to text placeholders in tests/headless
    FigureCanvasTkAgg = None  # type: ignore[assignment]


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

    def publish_waveform(self, audio_sample: Any, samplerate: int) -> None:
        """Render the latest captured waveform on the Tk thread."""

        try:
            waveform = np.asarray(audio_sample, dtype=float).reshape(-1)
        except Exception:
            return
        if waveform.size == 0:
            return

        self._on_tk_thread(lambda: self._render_waveform(waveform, int(samplerate)))

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
            figure = build_summary_plot_figure_for_config(config)
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

    def _render_waveform(self, waveform: np.ndarray, samplerate: int) -> None:
        if FigureCanvasTkAgg is None:
            self._set_placeholder(
                self.waveform_placeholder,
                "Matplotlib Tk backend unavailable.\nWaveform plots cannot be embedded.",
            )
            return

        figure = self._build_waveform_figure(waveform, samplerate)
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
    def _build_waveform_figure(waveform: np.ndarray, samplerate: int) -> Figure:
        figure = Figure(figsize=(8, 2.8))
        axis = figure.add_subplot(1, 1, 1)

        stride = max(1, waveform.size // 4000)
        shown = waveform[::stride]
        if samplerate > 0:
            x_axis = (np.arange(shown.size) * stride) / float(samplerate)
            x_label = "Time (s)"
        else:
            x_axis = np.arange(shown.size) * stride
            x_label = "Sample Index"

        axis.plot(x_axis, shown, linewidth=1.0, color="#1f77b4")
        axis.set_title("Latest Captured Waveform")
        axis.set_xlabel(x_label)
        axis.set_ylabel("Amplitude")
        axis.grid(True, linestyle=":", linewidth=0.5, color="gray")
        figure.tight_layout()
        return figure
