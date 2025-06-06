import tkinter as tk
from tkinter import messagebox
import json
import os
from tensiometer import Tensiometer
from threading import Event, Thread
import time
from maestro import Controller, DummyController

state_file = "gui_state.json"
stop_event = Event()


class ServoController:
    def __init__(self, servo=None):
        self.servo = servo or Controller()
        self.servo.setRange(0, 4000, 8000)
        self.running = Event()
        self.dwell_time = 1.0

    def set_speed(self, val):
        self.servo.setSpeed(0, int(val))

    def set_accel(self, val):
        self.servo.setAccel(0, int(val))

    def set_dwell_time(self, val):
        self.dwell_time = float(val)

    def start_loop(self):
        if not self.running.is_set():
            self.running.set()
            Thread(target=self.run_loop, daemon=True).start()

    def stop_loop(self):
        self.running.clear()

    def run_loop(self):
        while self.running.is_set():
            self.servo.setTarget(0, 4000)
            while self.servo.isMoving(0) and self.running.is_set():
                time.sleep(0.01)
            self.servo.setTarget(0, 8000)
            while self.servo.isMoving(0) and self.running.is_set():
                time.sleep(0.01)
            time.sleep(self.dwell_time)


if os.environ.get("SPOOF_SERVO") or os.environ.get("SPOOF_AUDIO"):
    servo_controller = ServoController(servo=DummyController())
else:
    servo_controller = ServoController()


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
        spoof_movement=bool(os.environ.get("SPOOF_PLC") or spoof_audio),
        stop_event=stop_event,
        samples_per_wire=samples,
        confidence_threshold=conf,
    )


def measure_calibrate():
    def run():
        stop_event.clear()
        servo_controller.start_loop()
        try:
            t = create_tensiometer()
            wire_number = int(entry_wire.get())
            save_state()
            t.measure_calibrate(wire_number)
            print("Done calibrating wire", wire_number)
        finally:
            servo_controller.stop_loop()
            stop_event.clear()

    Thread(target=run, daemon=True).start()


def measure_auto():
    def run():
        stop_event.clear()
        servo_controller.start_loop()
        try:
            t = create_tensiometer()
            save_state()
            t.measure_auto()
            print("Done measuring all wires")
        finally:
            servo_controller.stop_loop()
            stop_event.clear()

    Thread(target=run, daemon=True).start()


def measure_list():
    def run():
        stop_event.clear()
        servo_controller.start_loop()
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
            print("Done measuring wires", wire_list)
        finally:
            servo_controller.stop_loop()
            stop_event.clear()

    Thread(target=run, daemon=True).start()


def interrupt():
    stop_event.set()
    servo_controller.stop_loop()


root = tk.Tk()
root.title("Tensiometer GUI")

# Measurement frame
measurement_frame = tk.LabelFrame(root, text="Measurement")
measurement_frame.grid(row=4, column=0, columnspan=3, pady=5, sticky="we")

tk.Label(measurement_frame, text="Samples per Wire (≥1):").grid(row=0, column=0, sticky="e")
entry_samples = tk.Entry(measurement_frame)
entry_samples.grid(row=0, column=1)
tk.Label(measurement_frame, text="Confidence Threshold (0.0–1.0):").grid(row=1, column=0, sticky="e")
entry_confidence = tk.Entry(measurement_frame)
entry_confidence.grid(row=1, column=1)

tk.Label(measurement_frame, text="Wire Number:").grid(row=2, column=0, sticky="e")
entry_wire = tk.Entry(measurement_frame)
entry_wire.grid(row=2, column=1)
tk.Button(measurement_frame, text="Calibrate", command=measure_calibrate).grid(row=2, column=2)
tk.Label(root, text="Wire List:").grid(row=5, column=0, sticky="e")
entry_wire_list.grid(row=5, column=1)
tk.Button(root, text="Seek Wire(s)", command=measure_list).grid(row=5, column=2)
tk.Button(root, text="Measure Auto", command=measure_auto).grid(row=6, column=0)

tk.Button(root, text="Interrupt", command=interrupt).grid(row=6, column=1)
tk.Label(root, text="Servo Speed (1–255):").grid(row=7, column=0, sticky="e")
speed_slider.grid(row=7, column=1)
tk.Label(root, text="Servo Acceleration (1–255):").grid(row=8, column=0, sticky="e")
accel_slider.grid(row=8, column=1)
tk.Label(root, text="Dwell Time (0.00–2.00s):").grid(row=9, column=0, sticky="e")
dwell_slider.grid(row=9, column=1)

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

# Samples per wire
tk.Label(root, text="Samples per Wire (≥1):").grid(row=4, column=0, sticky="e")
entry_samples = tk.Entry(root)
entry_samples.grid(row=4, column=1)

# Confidence threshold
tk.Label(root, text="Confidence Threshold (0.0–1.0):").grid(row=5, column=0, sticky="e")
entry_confidence = tk.Entry(root)
entry_confidence.grid(row=5, column=1)

# Wire number
tk.Label(root, text="Wire Number:").grid(row=6, column=0, sticky="e")
entry_wire = tk.Entry(root)
entry_wire.grid(row=6, column=1)
tk.Button(root, text="Calibrate", command=measure_calibrate).grid(row=6, column=2)

tk.Label(measure_frame, text="Wire List:").grid(row=3, column=0, sticky="e")
entry_wire_list = tk.Entry(measure_frame)
entry_wire_list.grid(row=3, column=1)
tk.Button(measure_frame, text="Seek Wire(s)", command=measure_list).grid(
    row=3, column=2
)

tk.Button(measure_frame, text="Measure Auto", command=measure_auto).grid(
    row=4, column=0
)
tk.Button(measure_frame, text="Interrupt", command=interrupt).grid(row=4, column=1)

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
speed_slider.grid(row=0, column=1)

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
accel_slider.grid(row=1, column=1)

tk.Label(servo_frame, text="Dwell Time (0.00–2.00s):").grid(row=2, column=0, sticky="e")
dwell_slider = tk.Scale(
    servo_frame,
    from_=0,
    to=200,
    orient=tk.HORIZONTAL,
    command=lambda val: servo_controller.set_dwell_time(float(val) / 100),
)
dwell_slider.set(100)
dwell_slider.grid(row=2, column=1)

load_state()
root.mainloop()
