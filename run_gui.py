import tkinter as tk
from tkinter import messagebox
import json
import os
from tensiometer_functional import Tensiometer
from threading import Event, Thread

state_file = "gui_state.json"
stop_event = Event()

def save_state():
    try:
        samples = int(entry_samples.get())
        conf = float(entry_confidence.get())
    except ValueError as e:
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

def create_tensiometer():
    try:
        samples = int(entry_samples.get())
        if samples < 2:
            raise ValueError("Samples per wire must be ≥ 2")

        conf = float(entry_confidence.get())
        if not (0.0 <= conf <= 1.0):
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")

    except ValueError as e:
        messagebox.showerror("Input Error", str(e))
        raise

    return Tensiometer(
        apa_name=entry_apa.get(),
        layer=layer_var.get(),
        side=side_var.get(),
        flipped=flipped_var.get(),
        spoof=True,
        stop_event=stop_event,
        samples_per_wire=samples,
        confidence_threshold=conf,
    )

def measure_calibrate():
    def run():
        t = create_tensiometer()
        wire_number = int(entry_wire.get())
        save_state()
        t.measure_calibrate(wire_number)
        print("Done calibrating wire", wire_number)
    Thread(target=run, daemon=True).start()

def measure_auto():
    def run():
        t = create_tensiometer()
        save_state()
        t.measure_auto()
        print("Done measuring all wires")
    Thread(target=run, daemon=True).start()

def measure_list():
    def run():
        t = create_tensiometer()
        wire_list = [int(w.strip()) for w in entry_wire_list.get().split(",") if w.strip().isdigit()]
        save_state()
        print(f"Measuring wires: {wire_list}")
        t.measure_list(wire_list, preserve_order=False)
        print(" Done measuring wires", wire_list)
    Thread(target=run, daemon=True).start()

def interrupt():
    stop_event.set()
    stop_event.clear()

root = tk.Tk()
root.title("Tensiometer GUI")

# APA Name
tk.Label(root, text="APA Name:").grid(row=0, column=0, sticky="e")
entry_apa = tk.Entry(root)
entry_apa.grid(row=0, column=1)

# Layer
tk.Label(root, text="Layer:").grid(row=1, column=0, sticky="e")
layer_var = tk.StringVar(root)
layer_var.set("X")
tk.OptionMenu(root, layer_var, "X", "V", "U", "G").grid(row=1, column=1)

# Side
tk.Label(root, text="Side:").grid(row=2, column=0, sticky="e")
side_var = tk.StringVar(root)
side_var.set("A")
tk.OptionMenu(root, side_var, "A", "B").grid(row=2, column=1)

# Flipped
flipped_var = tk.BooleanVar()
tk.Checkbutton(root, text="Flipped", variable=flipped_var).grid(row=3, column=1, sticky="w")

# Samples per wire
tk.Label(root, text="Samples per Wire (≥2):").grid(row=4, column=0, sticky="e")
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

# Wire list
tk.Label(root, text="Wire List:").grid(row=7, column=0, sticky="e")
entry_wire_list = tk.Entry(root)
entry_wire_list.grid(row=7, column=1)
tk.Button(root, text="Seek Wire(s)", command=measure_list).grid(row=7, column=2)

# Measure Auto
tk.Button(root, text="Measure Auto", command=measure_auto).grid(row=8, column=0)

# Interrupt
tk.Button(root, text="Interrupt", command=interrupt).grid(row=8, column=1)

load_state()
root.mainloop()
