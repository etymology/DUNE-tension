"""Data structures describing the Tkinter GUI for the tensiometer."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Event, Thread
from typing import Any, Callable, Protocol
import os
import tkinter as tk

from dune_tension.maestro import Controller, DummyController, ServoController
from valve_trigger import DeviceNotFoundError, ValveController

try:  # pragma: no cover - optional dependency
    from dune_tension.plc_io import (  # type: ignore
        get_xy as plc_get_xy,
        goto_xy as plc_goto_xy,
        spoof_get_xy,
        spoof_goto_xy,
    )
except Exception:  # pragma: no cover - graceful fallback when PLC IO is absent

    def plc_get_xy() -> tuple[float, float]:
        return (0.0, 0.0)

    def plc_goto_xy(x: float, y: float) -> bool:
        return True

    def spoof_get_xy() -> tuple[float, float]:
        return (0.0, 0.0)

    def spoof_goto_xy(x: float, y: float) -> bool:
        return True


class ValveControllerProtocol(Protocol):
    def start_strum(self) -> None: ...

    def stop_strum(self) -> None: ...

    def set_dwell_time(self, duration: float) -> None: ...

    def close(self) -> None: ...


class _DummyValveController:
    def start_strum(self) -> None:  # pragma: no cover - trivial behaviour
        return

    def stop_strum(self) -> None:  # pragma: no cover - trivial behaviour
        return

    def set_dwell_time(self, duration: float) -> None:
        return

    def close(self) -> None:  # pragma: no cover - trivial behaviour
        return


@dataclass(slots=True)
class GUIWidgets:
    """Collection of Tkinter widgets used by the GUI."""

    entry_apa: tk.Entry
    layer_var: tk.StringVar
    side_var: tk.StringVar
    flipped_var: tk.BooleanVar
    entry_wire: tk.Entry
    entry_wire_list: tk.Entry
    entry_samples: tk.Entry
    entry_confidence: tk.Entry
    entry_record_duration: tk.Entry
    entry_measuring_duration: tk.Entry
    plot_audio_var: tk.BooleanVar
    entry_clear_range: tk.Entry
    entry_condition: tk.Entry
    entry_set_tension: tk.Entry
    speed_slider: tk.Scale
    accel_slider: tk.Scale
    dwell_slider: tk.Scale
    focus_slider: tk.Scale
    entry_xy: tk.Entry


@dataclass
class GUIContext:
    """Runtime information for the GUI and its callbacks."""

    root: tk.Misc
    widgets: GUIWidgets
    state_file: str
    stop_event: Event
    servo_controller: ServoController
    valve_controller: ValveControllerProtocol
    get_xy: Callable[[], tuple[float, float]]
    goto_xy: Callable[[float, float], bool]
    focus_command_var: tk.StringVar
    focus_command_canvas: tk.Canvas | None = None
    focus_command_dot: Any | None = None
    monitor_last_path: str = ""
    monitor_last_mtime: float | None = None
    monitor_thread: Thread | None = None


def _create_servo_controller() -> ServoController:
    """Return a :class:`ServoController` respecting spoof settings."""

    if os.environ.get("SPOOF_SERVO"):
        return ServoController(servo=DummyController())
    return ServoController(Controller())


def _create_valve_controller() -> ValveControllerProtocol:
    """Return a valve controller, falling back to a dummy when unavailable."""

    if os.environ.get("SPOOF_VALVE"):
        return _DummyValveController()

    try:
        return ValveController()
    except (DeviceNotFoundError, RuntimeError) as exc:
        print(f"Valve controller unavailable: {exc}")
        return _DummyValveController()


def _resolve_plc_functions() -> tuple[
    Callable[[], tuple[float, float]], Callable[[float, float], bool]
]:
    """Return PLC helpers honoring the ``SPOOF_PLC`` flag."""

    if os.environ.get("SPOOF_PLC"):
        return spoof_get_xy, spoof_goto_xy
    return plc_get_xy, plc_goto_xy


def create_context(
    root: tk.Misc,
    widgets: GUIWidgets,
    state_file: str,
    *,
    focus_command_var: tk.StringVar | None = None,
) -> GUIContext:
    """Create and return a :class:`GUIContext` for the GUI."""

    stop_event = Event()
    servo_controller = _create_servo_controller()
    valve_controller = _create_valve_controller()
    try:
        valve_controller.set_dwell_time(servo_controller.dwell_time)
    except AttributeError:
        pass
    get_xy, goto_xy = _resolve_plc_functions()
    if focus_command_var is None:
        focus_command_var = tk.StringVar(master=root, value="4000")

    return GUIContext(
        root=root,
        widgets=widgets,
        state_file=state_file,
        stop_event=stop_event,
        servo_controller=servo_controller,
        valve_controller=valve_controller,
        get_xy=get_xy,
        goto_xy=goto_xy,
        focus_command_var=focus_command_var,
    )
