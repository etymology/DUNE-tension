"""Application bootstrap for the tensiometer GUI."""

from __future__ import annotations

from functools import partial
from typing import Any
import tkinter as tk

from dune_tension.gui.actions import (
    calibrate_background_noise,
    measure_outliers,
    clear_range,
    handle_close,
    interrupt,
    manual_goto,
    manual_increment,
    measure_auto,
    measure_calibrate,
    measure_condition,
    measure_list_button,
    monitor_tension_logs,
    set_manual_tension,
    update_focus_command_indicator,
)
from dune_tension.gui.context import GUIContext, GUIWidgets, create_context
from dune_tension.gui.state import load_state


def run_app(state_file: str = "gui_state.json", root: tk.Misc | None = None) -> None:
    """Launch the Tkinter GUI."""

    root = root or tk.Tk()
    root.title("Tensiometer GUI")
    if hasattr(root, "columnconfigure"):
        root.columnconfigure(0, weight=1)

    focus_command_var = tk.StringVar(master=root, value="4000")
    widgets, focus_canvas, focus_dot, buttons, pad_buttons = _create_widgets(
        root, focus_command_var
    )
    ctx = create_context(
        root,
        widgets,
        state_file,
        focus_command_var=focus_command_var,
    )
    ctx.focus_command_canvas = focus_canvas
    ctx.focus_command_dot = focus_dot

    _configure_commands(ctx, buttons, pad_buttons)

    load_state(ctx)
    _initialise_servo(ctx)

    ctx.root.protocol("WM_DELETE_WINDOW", lambda: handle_close(ctx))
    ctx.root.after(1000, lambda: monitor_tension_logs(ctx))
    ctx.root.mainloop()


def _initialise_servo(ctx: GUIContext) -> None:
    try:
        value = int(ctx.widgets.focus_slider.get())
    except Exception:
        value = 4000
    ctx.servo_controller.on_focus_command = partial(update_focus_command_indicator, ctx)
    ctx.servo_controller.focus_position = value
    try:
        ctx.servo_controller.focus_target(value)
    except Exception:
        update_focus_command_indicator(ctx, value)
    else:
        update_focus_command_indicator(ctx, value)


def _create_widgets(
    root: tk.Misc, focus_command_var: tk.StringVar
) -> tuple[
    GUIWidgets,
    tk.Canvas | None,
    Any | None,
    dict[str, tk.Button],
    list[tuple[tk.Button, int, int]],
]:
    """Build and layout the GUI widgets."""

    bottom_frame = tk.Frame(root)
    bottom_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
    if hasattr(bottom_frame, "columnconfigure"):
        bottom_frame.columnconfigure(0, weight=1)

    apa_frame = tk.LabelFrame(bottom_frame, text="APA")
    apa_frame.grid(row=0, column=0, sticky="ew", pady=5)

    measure_frame = tk.LabelFrame(bottom_frame, text="Measurement")
    measure_frame.grid(row=1, column=0, sticky="ew", pady=5)

    servo_frame = tk.LabelFrame(bottom_frame, text="Servo")
    servo_frame.grid(row=2, column=0, sticky="ew", pady=5)
    if hasattr(servo_frame, "columnconfigure"):
        servo_frame.columnconfigure(1, weight=1)

    manual_move_frame = tk.LabelFrame(bottom_frame, text="Manual Move")
    manual_move_frame.grid(row=3, column=0, sticky="ew", pady=5)

    tk.Label(apa_frame, text="APA Name:").grid(row=0, column=0, sticky="e")
    entry_apa = tk.Entry(apa_frame)
    entry_apa.grid(row=0, column=1)

    tk.Label(apa_frame, text="Layer:").grid(row=1, column=0, sticky="e")
    layer_var = tk.StringVar(apa_frame, value="X")
    tk.OptionMenu(apa_frame, layer_var, "X", "V", "U", "G").grid(row=1, column=1)

    tk.Label(apa_frame, text="Side:").grid(row=2, column=0, sticky="e")
    side_var = tk.StringVar(apa_frame, value="A")
    tk.OptionMenu(apa_frame, side_var, "A", "B").grid(row=2, column=1)

    flipped_var = tk.BooleanVar(value=False)
    tk.Checkbutton(apa_frame, text="Flipped", variable=flipped_var).grid(
        row=3, column=1, sticky="w"
    )

    tk.Label(measure_frame, text="Samples per Wire (≥1):").grid(
        row=0, column=0, sticky="e"
    )
    entry_samples = tk.Entry(measure_frame)
    entry_samples.grid(row=0, column=1)

    tk.Label(measure_frame, text="Confidence Threshold (0.0–1.0):").grid(
        row=1, column=0, sticky="e"
    )
    entry_confidence = tk.Entry(measure_frame)
    entry_confidence.grid(row=1, column=1)

    tk.Label(measure_frame, text="Record Duration (s):").grid(
        row=9, column=0, sticky="e"
    )
    entry_record_duration = tk.Entry(measure_frame)
    entry_record_duration.grid(row=9, column=1)

    tk.Label(measure_frame, text="Measuring Duration (s):").grid(
        row=10, column=0, sticky="e"
    )
    entry_measuring_duration = tk.Entry(measure_frame)
    entry_measuring_duration.grid(row=10, column=1)

    tk.Label(measure_frame, text="Wire Number:").grid(row=2, column=0, sticky="e")
    entry_wire = tk.Entry(measure_frame)
    entry_wire.grid(row=2, column=1)
    btn_calibrate = tk.Button(measure_frame, text="Calibrate")
    btn_calibrate.grid(row=2, column=2)

    tk.Label(measure_frame, text="Wire List:").grid(row=3, column=0, sticky="e")
    entry_wire_list = tk.Entry(measure_frame)
    entry_wire_list.grid(row=3, column=1)
    btn_measure_list = tk.Button(measure_frame, text="Seek Wire(s)")
    btn_measure_list.grid(row=3, column=2)

    plot_audio_var = tk.BooleanVar(value=False)
    tk.Checkbutton(measure_frame, text="Plot Audio", variable=plot_audio_var).grid(
        row=4, column=2, sticky="w"
    )

    btn_measure_auto = tk.Button(measure_frame, text="Measure Auto")
    btn_measure_auto.grid(row=4, column=0)
    btn_interrupt = tk.Button(measure_frame, text="Interrupt")
    btn_interrupt.grid(row=4, column=1)

    tk.Label(measure_frame, text="Clear Range:").grid(row=5, column=0, sticky="e")
    entry_clear_range = tk.Entry(measure_frame)
    entry_clear_range.grid(row=5, column=1)
    btn_clear_range = tk.Button(measure_frame, text="Clear")
    btn_clear_range.grid(row=5, column=2)

    tk.Label(measure_frame, text="Condition:").grid(row=6, column=0, sticky="e")
    entry_condition = tk.Entry(measure_frame)
    entry_condition.grid(row=6, column=1)
    btn_measure_condition = tk.Button(measure_frame, text="Measure Condition")
    btn_measure_condition.grid(row=6, column=2)

    btn_remeasure_outliers = tk.Button(measure_frame, text="Remeasure Outliers")
    btn_remeasure_outliers.grid(row=7, column=2)

    tk.Label(measure_frame, text="Set Tensions:").grid(row=8, column=0, sticky="e")
    entry_set_tension = tk.Entry(measure_frame)
    entry_set_tension.grid(row=8, column=1)
    btn_set_tension = tk.Button(measure_frame, text="Apply Tensions")
    btn_set_tension.grid(row=8, column=2)

    btn_calibrate_noise = tk.Button(measure_frame, text="Calibrate Noise")
    btn_calibrate_noise.grid(row=11, column=2)

    tk.Label(servo_frame, text="Focus:").grid(row=0, column=0, sticky="e")
    focus_slider = tk.Scale(servo_frame, from_=4000, to=8000, orient=tk.HORIZONTAL)
    focus_slider.set(4000)
    focus_slider.grid(row=0, column=1, sticky="ew")

    tk.Label(servo_frame, textvariable=focus_command_var).grid(
        row=1, column=0, sticky="e"
    )
    focus_command_canvas: tk.Canvas | None = None
    focus_command_dot: Any | None = None
    if hasattr(tk, "Canvas"):
        focus_command_canvas = tk.Canvas(servo_frame, height=10)
        focus_command_canvas.grid(row=1, column=1, sticky="ew")
        focus_command_canvas.create_line(0, 5, int(focus_slider.cget("length")), 5)
        focus_command_dot = focus_command_canvas.create_oval(
            0, 0, 0, 0, fill="blue", outline=""
        )

    tk.Label(manual_move_frame, text="X,Y:").grid(row=0, column=0, sticky="e")
    entry_xy = tk.Entry(manual_move_frame)
    entry_xy.grid(row=0, column=1)
    btn_manual_go = tk.Button(manual_move_frame, text="Go")
    btn_manual_go.grid(row=0, column=2)

    pad_frame = tk.Frame(manual_move_frame)
    pad_frame.grid(row=1, column=0, columnspan=3)

    pad_specs = [
        ("\u2196", -1, 1, 0, 0),
        ("\u2191", 0, 1, 0, 1),
        ("\u2197", 1, 1, 0, 2),
        ("\u2190", -1, 0, 1, 0),
        ("\u2192", 1, 0, 1, 2),
        ("\u2199", -1, -1, 2, 0),
        ("\u2193", 0, -1, 2, 1),
        ("\u2198", 1, -1, 2, 2),
    ]
    pad_buttons: list[tuple[tk.Button, int, int]] = []
    for label, dx, dy, row, col in pad_specs:
        button = tk.Button(pad_frame, text=label, width=2)
        button.grid(row=row, column=col)
        pad_buttons.append((button, dx, dy))

    widgets = GUIWidgets(
        entry_apa=entry_apa,
        layer_var=layer_var,
        side_var=side_var,
        flipped_var=flipped_var,
        entry_wire=entry_wire,
        entry_wire_list=entry_wire_list,
        entry_samples=entry_samples,
        entry_confidence=entry_confidence,
        entry_record_duration=entry_record_duration,
        entry_measuring_duration=entry_measuring_duration,
        plot_audio_var=plot_audio_var,
        entry_clear_range=entry_clear_range,
        entry_condition=entry_condition,
        entry_set_tension=entry_set_tension,
        focus_slider=focus_slider,
        entry_xy=entry_xy,
    )

    buttons = {
        "calibrate": btn_calibrate,
        "measure_list": btn_measure_list,
        "measure_auto": btn_measure_auto,
        "interrupt": btn_interrupt,
        "clear_range": btn_clear_range,
        "measure_condition": btn_measure_condition,
        "remeasure_outliers": btn_remeasure_outliers,
        "set_tension": btn_set_tension,
        "calibrate_noise": btn_calibrate_noise,
        "manual_go": btn_manual_go,
    }

    return widgets, focus_command_canvas, focus_command_dot, buttons, pad_buttons


def _configure_commands(
    ctx: GUIContext,
    buttons: dict[str, tk.Button],
    pad_buttons: list[tuple[tk.Button, int, int]],
) -> None:
    """Attach the GUI callbacks to the Tkinter widgets."""

    buttons["calibrate"].configure(command=lambda: measure_calibrate(ctx))
    buttons["measure_list"].configure(command=lambda: measure_list_button(ctx))
    buttons["measure_auto"].configure(command=lambda: measure_auto(ctx))
    buttons["interrupt"].configure(command=lambda: interrupt(ctx))
    buttons["clear_range"].configure(command=lambda: clear_range(ctx))
    buttons["measure_condition"].configure(command=lambda: measure_condition(ctx))
    buttons["remeasure_outliers"].configure(command=lambda: measure_outliers(ctx))
    buttons["set_tension"].configure(command=lambda: set_manual_tension(ctx))
    buttons["calibrate_noise"].configure(
        command=lambda: calibrate_background_noise(ctx)
    )
    buttons["manual_go"].configure(command=lambda: manual_goto(ctx))

    for button, dx, dy in pad_buttons:
        button.configure(command=partial(manual_increment, ctx, dx, dy))

    widgets = ctx.widgets
    widgets.focus_slider.configure(
        command=lambda val: ctx.servo_controller.focus_target(int(float(val)))
    )
