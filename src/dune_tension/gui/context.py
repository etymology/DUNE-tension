"""Data structures describing the Tkinter GUI for the tensiometer."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
import threading
from threading import Event, Thread
from typing import Any, Callable
import tkinter as tk

from dune_tension.services import RuntimeBundle


@dataclass(slots=True)
class GUIWidgets:
    """Collection of Tkinter widgets used by the GUI."""

    entry_apa: tk.Entry
    measurement_mode_var: tk.StringVar
    layer_var: tk.StringVar
    side_var: tk.StringVar
    flipped_var: tk.BooleanVar
    a_taped_var: tk.BooleanVar
    b_taped_var: tk.BooleanVar
    entry_wire: tk.Entry
    entry_wire_list: tk.Entry
    entry_samples: tk.Entry
    entry_confidence: tk.Entry
    entry_record_duration: tk.Entry
    entry_measuring_duration: tk.Entry
    entry_wiggle_y_sigma: tk.Entry
    entry_focus_wiggle_sigma: tk.Entry
    plot_audio_var: tk.BooleanVar
    skip_measured_var: tk.BooleanVar
    entry_clear_range: tk.Entry
    entry_condition: tk.Entry
    entry_times_sigma: tk.Entry
    entry_set_tension: tk.Entry
    focus_slider: tk.Scale
    entry_xy: tk.Entry
    stream_segment_var: tk.StringVar
    stream_comb_var: tk.StringVar
    stream_focus_var: tk.StringVar
    stream_pitch_backlog_var: tk.StringVar
    stream_rescue_queue_var: tk.StringVar


@dataclass
class GUIContext:
    """Runtime information for the GUI and its callbacks."""

    root: tk.Misc
    widgets: GUIWidgets
    state_file: str
    runtime: RuntimeBundle
    stop_event: Event
    servo_controller: Any
    valve_controller: Any | None
    get_xy: Callable[[], tuple[float, float]]
    goto_xy: Callable[[float, float], bool]
    focus_command_var: tk.StringVar
    estimated_time_var: tk.StringVar
    focus_command_canvas: tk.Canvas | None = None
    focus_command_dot: Any | None = None
    monitor_last_path: str = ""
    monitor_last_mtime: float | None = None
    monitor_thread: Thread | None = None
    measurement_active: bool = False
    active_measurement_name: str = ""
    measurement_lock: Any = field(default_factory=threading.Lock)
    log_binding: Any | None = None
    strum: Callable[[], None] = field(default=lambda: None)
    live_plot_manager: Any | None = None


LOGGER = logging.getLogger(__name__)


def create_context(
    root: tk.Misc,
    widgets: GUIWidgets,
    state_file: str,
    *,
    runtime_bundle: RuntimeBundle,
    focus_command_var: tk.StringVar | None = None,
    estimated_time_var: tk.StringVar | None = None,
) -> GUIContext:
    """Create and return a :class:`GUIContext` for the GUI."""

    stop_event = Event()
    if focus_command_var is None:
        focus_command_var = tk.StringVar(master=root, value="4000")
    if estimated_time_var is None:
        estimated_time_var = tk.StringVar(master=root, value="Not running")

    return GUIContext(
        root=root,
        widgets=widgets,
        state_file=state_file,
        runtime=runtime_bundle,
        stop_event=stop_event,
        servo_controller=runtime_bundle.servo_controller,
        valve_controller=runtime_bundle.valve_controller,
        get_xy=runtime_bundle.motion.get_xy,
        goto_xy=runtime_bundle.motion.goto_xy,
        focus_command_var=focus_command_var,
        estimated_time_var=estimated_time_var,
        strum=runtime_bundle.strum,
    )
