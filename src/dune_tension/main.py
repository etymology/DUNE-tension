import tkinter as tk
from tkinter import messagebox
from typing import Any
import json
import os
import re
import sounddevice as sd
from tensiometer import Tensiometer
from tensiometer_functions import make_config
from data_cache import (
    clear_wire_range,
    clear_outliers as cache_clear_outliers,
    get_dataframe,
    update_dataframe,
)
from results import EXPECTED_COLUMNS
from datetime import datetime
from threading import Event, Thread
from maestro import DummyController, ServoController, Controller

try:
    from plc_io import (
        get_xy as _get_xy,
        goto_xy as _goto_xy,
        spoof_get_xy,
        spoof_goto_xy,
    )
except Exception:  # pragma: no cover - fallback for missing deps

    def _get_xy():
        return (0.0, 0.0)

    def _goto_xy(x, y):
        return True

    def spoof_get_xy():
        return (0.0, 0.0)

    def spoof_goto_xy(x, y):
        return True


state_file = "gui_state.json"
stop_event = Event()


if os.environ.get("SPOOF_SERVO"):
    servo_controller = ServoController(servo=DummyController())
else:
    servo_controller = ServoController(Controller())

# Determine which PLC functions to use for manual movement
if os.environ.get("SPOOF_PLC"):
    _get_xy_func = spoof_get_xy
    _goto_xy_func = spoof_goto_xy
else:
    _get_xy_func = _get_xy
    _goto_xy_func = _goto_xy


def save_state():
    try:
        samples = int(entry_samples.get())
        conf = float(entry_confidence.get())
    except ValueError as e:
        print(f"{e}")
        samples = 3
        conf = 0.7

    state = {
        "apa_name": entry_apa.get(),
        "layer": layer_var.get(),
        "side": side_var.get(),
        "flipped": flipped_var.get(),
        "wire_number": entry_wire.get(),
        "wire_list": entry_wire_list.get(),
        "samples_per_wire": samples,
        "confidence_threshold": conf,
        "servo_speed": speed_slider.get(),
        "servo_accel": accel_slider.get(),
        "servo_dwell": dwell_slider.get(),
        "plot_audio": plot_audio_var.get(),
        "focus_target": focus_slider.get(),
        "condition": entry_condition.get(),
        "set_tension": entry_set_tension.get(),
        "record_duration": entry_record_duration.get(),
        "measuring_duration": entry_measuring_duration.get(),
    }
    with open(state_file, "w") as f:
        json.dump(state, f)


def load_state():
    if os.path.exists(state_file):
        try:
            with open(state_file, "r") as f:
                state = json.load(f)
        except json.JSONDecodeError as exc:  # pragma: no cover - corrupt state
            print(f"Failed to load {state_file}: {exc}")
            return
        entry_apa.insert(0, state.get("apa_name", ""))
        layer_var.set(state.get("layer", "X"))
        side_var.set(state.get("side", "A"))
        flipped_var.set(state.get("flipped", False))
        entry_wire.insert(0, state.get("wire_number", ""))
        entry_wire_list.insert(0, state.get("wire_list", ""))
        entry_samples.insert(0, str(state.get("samples_per_wire", 3)))
        entry_confidence.insert(0, str(state.get("confidence_threshold", 0.7)))
        speed_slider.set(state.get("servo_speed", 1))
        accel_slider.set(state.get("servo_accel", 1))
        dwell_slider.set(state.get("servo_dwell", 100))
        plot_audio_var.set(state.get("plot_audio", False))
        focus_slider.set(state.get("focus_target", 4000))
        entry_condition.insert(0, state.get("condition", ""))
        entry_set_tension.insert(0, state.get("set_tension", ""))
        entry_record_duration.insert(0, str(state.get("record_duration", 0.5)))
        entry_measuring_duration.insert(0, str(state.get("measuring_duration", 10.0)))


def create_tensiometer():
    try:
        samples = int(entry_samples.get())
        if samples < 1:
            raise ValueError("Samples per wire must be ≥ 1")

        conf = float(entry_confidence.get())
        if not (0.0 <= conf <= 1.0):
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")

        rec = float(entry_record_duration.get())
        meas = float(entry_measuring_duration.get())

    except ValueError as e:
        messagebox.showerror("Input Error", str(e))
        raise

    spoof_audio = bool(os.environ.get("SPOOF_AUDIO"))
    return Tensiometer(
        apa_name=entry_apa.get(),
        layer=layer_var.get(),
        side=side_var.get(),
        flipped=flipped_var.get(),
        spoof=spoof_audio,
        spoof_movement=bool(os.environ.get("SPOOF_PLC")),
        stop_event=stop_event,
        samples_per_wire=samples,
        confidence_threshold=conf,
        record_duration=rec,
        measuring_duration=meas,
        plot_audio=plot_audio_var.get(),
        start_servo_loop=servo_controller.start_loop,
        stop_servo_loop=servo_controller.stop_loop,
        focus_wiggle=servo_controller.nudge_focus,
    )


def measure_calibrate():
    def run():
        stop_event.clear()
        t = None
        try:
            t = create_tensiometer()
            wire_number = int(entry_wire.get())
            save_state()
            t.measure_calibrate(wire_number)
            print("Done calibrating wire", wire_number)
        finally:
            if t is not None:
                try:
                    t.close()
                except Exception:
                    pass
            stop_event.clear()

    Thread(target=run, daemon=True).start()


def measure_auto():
    def run():
        stop_event.clear()
        t = None
        try:
            t = create_tensiometer()
            save_state()
            t.measure_auto()
        finally:
            if t is not None:
                try:
                    t.close()
                except Exception:
                    pass
            stop_event.clear()

    Thread(target=run, daemon=True).start()


def measure_list():
    def run():
        stop_event.clear()
        t = None
        try:
            t = create_tensiometer()
            wire_list = [
                int(w.strip())
                for w in entry_wire_list.get().split(",")
                if w.strip().isdigit()
            ]
            save_state()
            print(f"Measuring wires: {wire_list}")
            t.measure_list(wire_list, preserve_order=False)
        finally:
            if t is not None:
                try:
                    t.close()
                except Exception:
                    pass
            stop_event.clear()

    Thread(target=run, daemon=True).start()


def calibrate_background_noise() -> None:
    """Record background noise for filtering future recordings."""

    def run() -> None:
        stop_event.clear()
        try:
            from audioProcessing import calibrate_background_noise, get_samplerate

            samplerate = get_samplerate()
            if samplerate is None:
                print("Unable to access audio device")
                return
            calibrate_background_noise(int(samplerate))
            print("Background noise calibrated")
        finally:
            stop_event.clear()

    Thread(target=run, daemon=True).start()


def measure_condition() -> None:
    """Measure wires whose tension satisfies ``entry_condition``."""

    def _get_wires(cfg, expr: str) -> list[int]:
        from data_cache import get_dataframe  # Local import for test stubs
        import pandas as pd

        df = get_dataframe(cfg.data_path)
        mask = (
            (df["apa_name"] == cfg.apa_name)
            & (df["layer"] == cfg.layer)
            & (df["side"] == cfg.side)
        )
        subset = df[mask].copy()
        subset["wire_number"] = pd.to_numeric(subset["wire_number"], errors="coerce")
        subset["tension"] = pd.to_numeric(subset["tension"], errors="coerce")
        subset = subset.dropna(subset=["wire_number", "tension"])
        subset = subset.sort_values("time").drop_duplicates(
            subset="wire_number", keep="last"
        )
        wires: list[int] = []
        for _, row in subset.iterrows():
            try:
                if eval(expr, {"t": float(row["tension"])}):
                    wires.append(int(row["wire_number"]))
            except Exception as exc:
                print(f"Invalid expression '{expr}': {exc}")
                return []
        return sorted(set(wires))

    def run() -> None:
        stop_event.clear()
        t = None
        expr = entry_condition.get().strip()
        if not expr:
            print("No condition specified")
            return
        try:
            t = create_tensiometer()
            save_state()
            wires = _get_wires(t.config, expr)
            if not wires:
                print(f"No wires satisfy: {expr}")
                return
            print(f"Measuring wires {wires} matching '{expr}'")
            t.measure_list(wires, preserve_order=False)
        finally:
            if t is not None:
                try:
                    t.close()
                except Exception:
                    pass
            stop_event.clear()

    Thread(target=run, daemon=True).start()


def _parse_ranges(text: str) -> list[tuple[int, int]]:
    """Return list of ``(start, end)`` tuples parsed from ``text``."""
    ranges: list[tuple[int, int]] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            try:
                a, b = part.split("-", 1)
                start = int(a)
                end = int(b)
            except ValueError:
                continue
        else:
            try:
                start = end = int(part)
            except ValueError:
                continue
        if start > end:
            start, end = end, start
        ranges.append((start, end))
    return ranges


_PAIR_RE = re.compile(r"\(?\s*(\d+)\s*[,:]\s*([+-]?\d+(?:\.\d*)?)\s*\)?")


def _parse_pairs(text: str) -> list[tuple[int, float]]:
    """Return list of ``(wire, tension)`` pairs parsed from ``text``."""
    pairs: list[tuple[int, float]] = []
    for m in _PAIR_RE.finditer(text):
        try:
            wire = int(m.group(1))
            tension = float(m.group(2))
        except ValueError:
            continue
        pairs.append((wire, tension))
    return pairs


def clear_range() -> None:
    ranges = _parse_ranges(entry_clear_range.get())
    if not ranges:
        print("No valid range specified")
        return

    try:
        samples = int(entry_samples.get())
    except Exception:
        samples = 3
    try:
        conf = float(entry_confidence.get())
    except Exception:
        conf = 0.7

    cfg = make_config(
        apa_name=entry_apa.get(),
        layer=layer_var.get(),
        side=side_var.get(),
        flipped=flipped_var.get(),
        samples_per_wire=samples,
        confidence_threshold=conf,
        plot_audio=plot_audio_var.get(),
    )

    for start, end in ranges:
        clear_wire_range(cfg.data_path, cfg.apa_name, cfg.layer, cfg.side, start, end)
    print(f"Cleared ranges: {entry_clear_range.get()}")


def clear_outliers() -> None:
    try:
        samples = int(entry_samples.get())
    except Exception:
        samples = 3
    try:
        conf = float(entry_confidence.get())
    except Exception:
        conf = 0.7

    cfg = make_config(
        apa_name=entry_apa.get(),
        layer=layer_var.get(),
        side=side_var.get(),
        flipped=flipped_var.get(),
        samples_per_wire=samples,
        confidence_threshold=conf,
        plot_audio=plot_audio_var.get(),
    )

    removed = cache_clear_outliers(
        cfg.data_path,
        cfg.apa_name,
        cfg.layer,
        cfg.side,
        2.0,
        conf,
    )
    if removed:
        print(f"Cleared outlier wires: {removed}")
    else:
        print("No outlier wires found")


def set_manual_tension() -> None:
    """Parse :data:`entry_set_tension` and update tension values."""
    pairs = _parse_pairs(entry_set_tension.get())
    if not pairs:
        print("No valid tension pairs specified")
        return

    try:
        samples = int(entry_samples.get())
    except Exception:
        samples = 3
    try:
        conf = float(entry_confidence.get())
    except Exception:
        conf = 0.7

    cfg = make_config(
        apa_name=entry_apa.get(),
        layer=layer_var.get(),
        side=side_var.get(),
        flipped=flipped_var.get(),
        samples_per_wire=samples,
        confidence_threshold=conf,
        plot_audio=plot_audio_var.get(),
    )

    df = get_dataframe(cfg.data_path)
    for wire, tension in pairs:
        mask = (
            (df["apa_name"] == cfg.apa_name)
            & (df["layer"] == cfg.layer)
            & (df["side"] == cfg.side)
            & (df["wire_number"].astype(int) == wire)
        )
        if mask.any():
            df.loc[mask, "tension"] = tension
            if "time" in df.columns:
                df.loc[mask, "time"] = datetime.now().isoformat()
        else:
            row = {col: "" for col in EXPECTED_COLUMNS}
            row.update(
                {
                    "apa_name": cfg.apa_name,
                    "layer": cfg.layer,
                    "side": cfg.side,
                    "wire_number": wire,
                    "frequency": 0.0,
                    "confidence": 1,
                    "tension": tension,
                    "tension_pass": 1,
                    "x": 0.0,
                    "y": 0.0,
                    "wires": [],
                    "ttf": 0.0,
                    "time": datetime.now().isoformat(),
                    "zone": 0,
                    "wire_length": 0.0,
                    "t_sigma": 0.0,
                }
            )
            df.loc[len(df)] = row
    update_dataframe(cfg.data_path, df)
    print(f"Updated tensions: {pairs}")


def interrupt():
    stop_event.set()
    servo_controller.stop_loop()


def monitor_tension_logs():
    """Check for updates to the tension data file and refresh logs."""
    try:
        samples = int(entry_samples.get())
        if samples < 1:
            raise ValueError
    except Exception:
        samples = 3

    try:
        conf = float(entry_confidence.get())
        if not (0.0 <= conf <= 1.0):
            raise ValueError
    except Exception:
        conf = 0.7

    config = make_config(
        apa_name=entry_apa.get(),
        layer=layer_var.get(),
        side=side_var.get(),
        flipped=flipped_var.get(),
        samples_per_wire=samples,
        confidence_threshold=conf,
        plot_audio=plot_audio_var.get(),
    )

    path = config.data_path
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = None

    if (
        monitor_tension_logs.last_path != path
        or monitor_tension_logs.last_mtime != mtime
    ):
        monitor_tension_logs.last_path = path
        monitor_tension_logs.last_mtime = mtime

        def run() -> None:
            try:
                from analyze import update_tension_logs

                update_tension_logs(config)
                print(
                    f"Updated tension logs for {config.apa_name} layer {config.layer}"
                )
            except Exception as exc:
                print(f"Failed to update logs: {exc}")

        if (
            monitor_tension_logs.update_thread is None
            or not monitor_tension_logs.update_thread.is_alive()
        ):
            monitor_tension_logs.update_thread = Thread(target=run, daemon=True)
            monitor_tension_logs.update_thread.start()

    root.after(2000, monitor_tension_logs)


monitor_tension_logs.last_path = ""
monitor_tension_logs.last_mtime = None
monitor_tension_logs.update_thread = None


def manual_goto():
    """Move the winder to the X,Y position entered in :data:`entry_xy`."""
    text = entry_xy.get()
    try:
        x_str, y_str = text.split(",")
        x_val = float(x_str.strip())
        y_val = float(y_str.strip())
    except ValueError:
        print(f"Invalid coordinates: {text}")
        return
    _goto_xy_func(x_val, y_val)


def manual_increment(dx: float, dy: float):
    """Move the winder by 0.1 mm increments in the specified direction.

    The winder's PLC can return positions with long floating point values.
    To keep movements predictable, always round the target coordinates to the
    nearest 0.1 mm before sending the command.
    """
    cur_x, cur_y = _get_xy_func()

    # Determine x-axis orientation based on side/flipped state
    if (side_var.get() == "A" and not flipped_var.get()) or (
        side_var.get() == "B" and flipped_var.get()
    ):
        x_sign = 1.0
    else:
        x_sign = -1.0

    new_x = cur_x + x_sign * dx * 0.1
    new_y = cur_y + dy * 0.1

    # Round to the nearest 0.1 mm to avoid accumulating floating point errors
    new_x = round(new_x, 1)
    new_y = round(new_y, 1)

    _goto_xy_func(new_x, new_y)


root = tk.Tk()
root.title("Tensiometer GUI")
if hasattr(root, "columnconfigure"):
    root.columnconfigure(0, weight=1)

focus_slider: tk.Scale  # defined later

# Track the most recent focus servo command.  This is updated via
# ``servo_controller.on_focus_command``.
focus_command_var = tk.StringVar(value="4000")
focus_command_canvas: Any | None = None
focus_command_dot: Any | None = None


def update_focus_command_indicator(val: int) -> None:
    """Display the last focus command as a blue dot under the slider."""

    def _update() -> None:
        focus_command_var.set(str(val))
        if not focus_command_canvas or not focus_command_dot:
            return
        length = focus_command_canvas.winfo_width()
        low = int(focus_slider["from"])
        high = int(focus_slider["to"])
        if high == low:
            x = 0
        else:
            x = (val - low) / (high - low) * length
        r = 3
        focus_command_canvas.coords(focus_command_dot, x - r, 5 - r, x + r, 5 + r)

    root.after(0, _update)


# Register the callback so that all focus movements update the indicator
servo_controller.on_focus_command = update_focus_command_indicator


def _on_close() -> None:
    """Gracefully shut down threads and destroy the root window."""
    stop_event.set()
    servo_controller.stop_loop()
    try:
        sd.stop()
    except Exception:
        pass
    try:
        root.destroy()
    except Exception:
        pass


root.protocol("WM_DELETE_WINDOW", _on_close)

# --- Layout Frames ---------------------------------------------------------
# The GUI is split into logical areas to make it easier to navigate.  "apa_frame"
# holds general information about the APA being measured.  "measure_frame"
# contains all measurement parameters and actions.  "servo_frame" groups the
# servo configuration widgets.  Finally, "bottom_frame" simply keeps the three
# main sections neatly stacked in the window.

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

# --- APA Info --------------------------------------------------------------
tk.Label(apa_frame, text="APA Name:").grid(row=0, column=0, sticky="e")
entry_apa = tk.Entry(apa_frame)
entry_apa.grid(row=0, column=1)

tk.Label(apa_frame, text="Layer:").grid(row=1, column=0, sticky="e")
layer_var = tk.StringVar(apa_frame)
layer_var.set("X")
tk.OptionMenu(apa_frame, layer_var, "X", "V", "U", "G").grid(row=1, column=1)

tk.Label(apa_frame, text="Side:").grid(row=2, column=0, sticky="e")
side_var = tk.StringVar(apa_frame)
side_var.set("A")
tk.OptionMenu(apa_frame, side_var, "A", "B").grid(row=2, column=1)

flipped_var = tk.BooleanVar()
tk.Checkbutton(apa_frame, text="Flipped", variable=flipped_var).grid(
    row=3, column=1, sticky="w"
)

# --- Measurement Parameters -----------------------------------------------
tk.Label(measure_frame, text="Samples per Wire (≥1):").grid(row=0, column=0, sticky="e")
entry_samples = tk.Entry(measure_frame)
entry_samples.grid(row=0, column=1)

tk.Label(measure_frame, text="Confidence Threshold (0.0–1.0):").grid(
    row=1, column=0, sticky="e"
)
entry_confidence = tk.Entry(measure_frame)
entry_confidence.grid(row=1, column=1)

tk.Label(measure_frame, text="Record Duration (s):").grid(row=9, column=0, sticky="e")
entry_record_duration = tk.Entry(measure_frame)
entry_record_duration.grid(row=9, column=1)

tk.Label(measure_frame, text="Measuring Duration (s):").grid(row=10, column=0, sticky="e")
entry_measuring_duration = tk.Entry(measure_frame)
entry_measuring_duration.grid(row=10, column=1)

tk.Label(measure_frame, text="Wire Number:").grid(row=2, column=0, sticky="e")
entry_wire = tk.Entry(measure_frame)
entry_wire.grid(row=2, column=1)
tk.Button(measure_frame, text="Calibrate", command=measure_calibrate).grid(
    row=2, column=2
)

tk.Label(measure_frame, text="Wire List:").grid(row=3, column=0, sticky="e")
entry_wire_list = tk.Entry(measure_frame)
entry_wire_list.grid(row=3, column=1)
tk.Button(measure_frame, text="Seek Wire(s)", command=measure_list).grid(
    row=3, column=2
)

plot_audio_var = tk.BooleanVar()
tk.Checkbutton(measure_frame, text="Plot Audio", variable=plot_audio_var).grid(
    row=4, column=2, sticky="w"
)

tk.Button(measure_frame, text="Measure Auto", command=measure_auto).grid(
    row=4, column=0
)
tk.Button(measure_frame, text="Interrupt", command=interrupt).grid(row=4, column=1)

tk.Label(measure_frame, text="Clear Range:").grid(row=5, column=0, sticky="e")
entry_clear_range = tk.Entry(measure_frame)
entry_clear_range.grid(row=5, column=1)
tk.Button(measure_frame, text="Clear", command=clear_range).grid(row=5, column=2)

tk.Label(measure_frame, text="Condition:").grid(row=6, column=0, sticky="e")
entry_condition = tk.Entry(measure_frame)
entry_condition.grid(row=6, column=1)
tk.Button(
    measure_frame,
    text="Measure Condition",
    command=measure_condition,
).grid(row=6, column=2)
tk.Button(
    measure_frame,
    text="Clear Outliers",
    command=clear_outliers,
).grid(row=7, column=2)

tk.Label(measure_frame, text="Set Tensions:").grid(row=8, column=0, sticky="e")
entry_set_tension = tk.Entry(measure_frame)
entry_set_tension.grid(row=8, column=1)
tk.Button(
    measure_frame,
    text="Apply Tensions",
    command=set_manual_tension,
).grid(row=8, column=2)



tk.Button(
    measure_frame,
    text="Calibrate Noise",
    command=calibrate_background_noise,
).grid(row=11, column=2)

# --- Servo Parameters ------------------------------------------------------
tk.Label(servo_frame, text="Servo Speed (1–255):").grid(row=0, column=0, sticky="e")

speed_slider = tk.Scale(
    servo_frame,
    from_=1,
    to=255,
    orient=tk.HORIZONTAL,
    command=servo_controller.set_speed,
)
speed_slider.set(1)
speed_slider.grid(row=0, column=1, sticky="ew")

tk.Label(servo_frame, text="Servo Acceleration (1–255):").grid(
    row=1, column=0, sticky="e"
)

accel_slider = tk.Scale(
    servo_frame,
    from_=1,
    to=255,
    orient=tk.HORIZONTAL,
    command=servo_controller.set_accel,
)
accel_slider.set(1)
accel_slider.grid(row=1, column=1, sticky="ew")

tk.Label(servo_frame, text="Dwell Time (0.00–2.00s):").grid(row=2, column=0, sticky="e")

dwell_slider = tk.Scale(
    servo_frame,
    from_=0,
    to=200,
    orient=tk.HORIZONTAL,
    command=lambda val: servo_controller.set_dwell_time(float(val) / 100),
)
dwell_slider.set(100)
dwell_slider.grid(row=2, column=1, sticky="ew")

tk.Label(servo_frame, text="Focus:").grid(row=3, column=0, sticky="e")

focus_slider = tk.Scale(
    servo_frame,
    from_=4000,
    to=8000,
    orient=tk.HORIZONTAL,
    command=lambda val: servo_controller.focus_target(int(val)),
)
focus_slider.set(4000)
focus_slider.grid(row=3, column=1, sticky="ew")

# Draw a line beneath the focus slider showing the last command sent to the
# servo controller.  The position is displayed numerically on the left.  If the
# ``Canvas`` widget is unavailable (as in some test environments), simply skip
# the indicator.
tk.Label(servo_frame, textvariable=focus_command_var).grid(row=4, column=0, sticky="e")
if hasattr(tk, "Canvas"):
    focus_command_canvas = tk.Canvas(servo_frame, height=10)
    focus_command_canvas.grid(row=4, column=1, sticky="ew")
    focus_command_canvas.create_line(0, 5, int(focus_slider.cget("length")), 5)
    focus_command_dot = focus_command_canvas.create_oval(
        0, 0, 0, 0, fill="blue", outline=""
    )
    update_focus_command_indicator(focus_slider.get())

# --- Manual Move -----------------------------------------------------------
manual_move_frame = tk.LabelFrame(bottom_frame, text="Manual Move")
manual_move_frame.grid(row=3, column=0, sticky="ew", pady=5)

tk.Label(manual_move_frame, text="X,Y:").grid(row=0, column=0, sticky="e")
entry_xy = tk.Entry(manual_move_frame)
entry_xy.grid(row=0, column=1)
tk.Button(manual_move_frame, text="Go", command=manual_goto).grid(row=0, column=2)

pad_frame = tk.Frame(manual_move_frame)
pad_frame.grid(row=1, column=0, columnspan=3)

btn_specs = [
    ("\u2196", -1, 1, 0, 0),
    ("\u2191", 0, 1, 0, 1),
    ("\u2197", 1, 1, 0, 2),
    ("\u2190", -1, 0, 1, 0),
    ("\u2192", 1, 0, 1, 2),
    ("\u2199", -1, -1, 2, 0),
    ("\u2193", 0, -1, 2, 1),
    ("\u2198", 1, -1, 2, 2),
]
for label, dx, dy, r, c in btn_specs:
    tk.Button(
        pad_frame,
        text=label,
        command=lambda dx=dx, dy=dy: manual_increment(dx, dy),
        width=2,
    ).grid(row=r, column=c)


load_state()
try:
    val = int(focus_slider.get())
except Exception:
    val = 4000
servo_controller.focus_position = val
try:
    servo_controller.focus_target(val)
except Exception:
    pass
root.after(1000, monitor_tension_logs)
root.mainloop()
