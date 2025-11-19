"""State persistence helpers for the tensiometer GUI."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import tkinter as tk
from typing import Any

from dune_tension.gui.context import GUIContext


@dataclass(slots=True)
class _PersistedState:
    apa_name: str
    layer: str
    side: str
    flipped: bool
    a_taped: bool
    b_taped: bool
    wire_number: str
    wire_list: str
    samples_per_wire: int
    confidence_threshold: float
    plot_audio: bool
    focus_target: int
    condition: str
    set_tension: str
    record_duration: str
    measuring_duration: str


def save_state(ctx: GUIContext) -> None:
    """Serialize the current GUI selections to ``ctx.state_file``."""

    w = ctx.widgets
    try:
        samples = int(w.entry_samples.get())
    except ValueError:
        samples = 3
    try:
        conf = float(w.entry_confidence.get())
    except ValueError:
        conf = 0.7

    state = _PersistedState(
        apa_name=w.entry_apa.get(),
        layer=w.layer_var.get(),
        side=w.side_var.get(),
        flipped=bool(w.flipped_var.get()),
        a_taped=bool(w.a_taped_var.get()),
        b_taped=bool(w.b_taped_var.get()),
        wire_number=w.entry_wire.get(),
        wire_list=w.entry_wire_list.get(),
        samples_per_wire=samples,
        confidence_threshold=conf,
        plot_audio=bool(w.plot_audio_var.get()),
        focus_target=int(w.focus_slider.get()),
        condition=w.entry_condition.get(),
        set_tension=w.entry_set_tension.get(),
        record_duration=w.entry_record_duration.get(),
        measuring_duration=w.entry_measuring_duration.get(),
    )

    with open(ctx.state_file, "w", encoding="utf-8") as handle:
        json.dump(asdict(state), handle)


def _set_entry(entry: tk.Entry, value: Any) -> None:
    entry.delete(0, tk.END)
    entry.insert(0, str(value))


def load_state(ctx: GUIContext) -> None:
    """Populate widgets from ``ctx.state_file`` if it exists."""

    try:
        with open(ctx.state_file, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return
    except json.JSONDecodeError as exc:  # pragma: no cover - corrupt state file
        print(f"Failed to load {ctx.state_file}: {exc}")
        return

    w = ctx.widgets
    _set_entry(w.entry_apa, data.get("apa_name", ""))
    w.layer_var.set(data.get("layer", "X"))
    w.side_var.set(data.get("side", "A"))
    w.flipped_var.set(bool(data.get("flipped", False)))
    w.a_taped_var.set(bool(data.get("a_taped", False)))
    w.b_taped_var.set(bool(data.get("b_taped", False)))
    _set_entry(w.entry_wire, data.get("wire_number", ""))
    _set_entry(w.entry_wire_list, data.get("wire_list", ""))
    _set_entry(w.entry_samples, data.get("samples_per_wire", 3))
    _set_entry(w.entry_confidence, data.get("confidence_threshold", 0.7))
    w.plot_audio_var.set(bool(data.get("plot_audio", False)))
    w.focus_slider.set(int(data.get("focus_target", 4000)))
    _set_entry(w.entry_condition, data.get("condition", ""))
    _set_entry(w.entry_set_tension, data.get("set_tension", ""))
    _set_entry(w.entry_record_duration, data.get("record_duration", 0.5))
    _set_entry(w.entry_measuring_duration, data.get("measuring_duration", 10.0))

    ctx.focus_command_var.set(str(w.focus_slider.get()))
