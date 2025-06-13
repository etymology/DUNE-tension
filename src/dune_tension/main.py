import tkinter as tk
from tkinter import messagebox
import json
import os
import sounddevice as sd
from tensiometer import Tensiometer
from tensiometer_functions import make_config
from data_cache import clear_wire_range
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
    }
    with open(state_file, "w") as f:
        json.dump(state, f)


def load_state():
    if os.path.exists(state_file):
        with open(state_file, "r") as f:
            state = json.load(f)
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


def create_tensiometer():
    try:
        samples = int(entry_samples.get())
        if samples < 1:
            raise ValueError("Samples per wire must be ≥ 1")

        conf = float(entry_confidence.get())
        if not (0.0 <= conf <= 1.0):
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")

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
        plot_audio=plot_audio_var.get(),
    )


def measure_calibrate():
    def run():
        stop_event.clear()
        servo_controller.start_loop()
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
            servo_controller.stop_loop()
            stop_event.clear()

    Thread(target=run, daemon=True).start()


def measure_auto():
    def run():
        stop_event.clear()
        servo_controller.start_loop()
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
            servo_controller.stop_loop()
            stop_event.clear()

    Thread(target=run, daemon=True).start()


def measure_list():
    def run():
        stop_event.clear()
        servo_controller.start_loop()
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
            servo_controller.stop_loop()
            stop_event.clear()

    Thread(target=run, daemon=True).start()


def _parse_ranges(text: str) -> list[tuple[int, int]]:
    """Return list of ``(start, end)`` tuples parsed from ``text``."""
    ranges: list[tuple[int, int]] = []
    for part in text.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            try:
                a, b = part.split('-', 1)
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
        try:
            from analyze import update_tension_logs

            update_tension_logs(config)
            print(f"Updated tension logs for {config.apa_name} layer {config.layer}")
        except Exception as exc:
            print(f"Failed to update logs: {exc}")

    root.after(50000, monitor_tension_logs)


monitor_tension_logs.last_path = ""
monitor_tension_logs.last_mtime = None


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

tk.Label(servo_frame, text="Dwell Time (0.00–1.00s):").grid(row=2, column=0, sticky="e")

dwell_slider = tk.Scale(
    servo_frame,
    from_=0,
    to=100,
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
root.after(1000, monitor_tension_logs)
root.mainloop()
