# tensiometer.py
import requests
from maestro import Controller
import sounddevice as sd
import numpy as np
import threading
import time
import os
import pandas as pd
from analyze_tension_data import analyze_tension_data
from datetime import datetime
import math
from typing import List
from logging_util import (
    log_data,
)

from geometry import (
    zone_lookup,
    length_lookup,
    refine_position,
)
from tension_calculation import (
    tension_lookup,
    tension_pass,
    has_cluster_dict,
    tension_plausible,
    calculate_kde_max,
)
from audioProcessing import record_audio, analyze_sample
from plc_io import get_xy, goto_xy, wiggle, TENSION_SERVER_URL


class Tensiometer:
    def __init__(
        self,
        apa_name,
        layer,
        side,
        ttyStr="/dev/ttyACM1",
        sound_card_name="default",
        record_duration=0.15,
        starting_wiggle_step=0.2,
        timeout=60,
        samples_per_wire=3,
        confidence_threshold=0.7,
        use_wiggle=True,
        save_audio=True,
        delay_after_plucking=0.2,
        wiggle_type="gaussian",
        use_servo=False,
        initial_wire_height=190,
        test_mode=False,
        wiggle_interval=2,
        flipped=False,
        bypass_data_collection=False,
        stop_event=None,
    ):
        """
        Initialize the controller, audio devices, and check web server connectivity more concisely.
        """
        self.apa_name = apa_name
        self.samples_per_wire = samples_per_wire
        self.record_duration = record_duration
        self.confidence_threshold = confidence_threshold
        self.delay_after_plucking = delay_after_plucking
        self.wiggle_type = wiggle_type
        self.starting_wiggle_step = starting_wiggle_step
        self.save_audio = save_audio
        self.timeout = timeout
        self.stop_servo_event = threading.Event()
        self.stop_wiggle_event = threading.Event()
        self.use_servo = use_servo
        self.use_wiggle = use_wiggle
        self.initial_wire_height = initial_wire_height
        self.layer = layer
        self.side = side
        self.wiggle_interval = wiggle_interval
        self.flipped = flipped
        self.bypass_data_collection = bypass_data_collection
        self.stop_event = stop_event

        if self.layer in ["X", "G"]:
            self.dx, self.dy = 0.0, 2300 / 480
            self.wire_min, self.wire_max = 1, 480
            if self.layer == "G":
                self.wire_max = 481
        else:
            self.dx, self.dy = 8.0, 5.75
            self.wire_min, self.wire_max = 8, 1146
            if (self.layer == "U" and self.side == "A") or (
                self.layer == "V" and self.side == "B"
            ):
                self.dy = -5.75
        if self.flipped:
            self.dy = -self.dy

        if not test_mode:
            if use_servo:
                try:
                    self.maestro = Controller(ttyStr)
                    self.servo_state = 0
                    self.maestro.runScriptSub(1)
                except Exception as e:
                    print(f"Failed to initialize Maestro controller: {e}")
                    exit(1)

            try:
                device_info = sd.query_devices()
                self.sound_device_index = next(
                    (
                        index
                        for index, d in enumerate(device_info)
                        if sound_card_name in d["name"]
                    ),
                    None,
                )
                if self.sound_device_index is not None:
                    self.sample_rate = device_info[self.sound_device_index][
                        "default_samplerate"
                    ]
                    print(
                        f"Using USB PnP Sound Device (hw:{self.sound_device_index},0)"
                    )
                else:
                    print("Couldn't find USB PnP Sound Device.")
                    print(device_info)
                    exit(1)
            except Exception as e:
                print(f"Failed to initialize audio devices: {e}")
                exit(1)

            if not self.is_web_server_active(TENSION_SERVER_URL):
                print(
                    "Failed to connect to the tension server.\nMake sure you are connected to Dunes and the server is running."
                )
                exit(1)
            print("Connected to the tension server.")

    def is_web_server_active(self, url):
        """
        Check if a web server is active by sending a HTTP GET request.
        """
        try:
            return 200 <= requests.get(url, timeout=3).status_code < 500
        except requests.RequestException as e:
            print(f"An error occurred while checking the server: {e}")
            return False

    def load_tension_summary(self):
        import pandas as pd

        file_path = (
            f"data/tension_summaries/tension_summary_{self.apa_name}_{self.layer}.csv"
        )
        try:
            df = pd.read_csv(file_path)
        except FileNotFoundError:
            return f"❌ File not found: {file_path}", [], []

        if "A" not in df.columns or "B" not in df.columns:
            return "⚠️ File missing required columns 'A' and 'B'", [], []

        # Convert columns to lists, preserving NaNs if present
        a_list = df["A"].tolist()
        b_list = df["B"].tolist()

        return a_list, b_list

    def get_uuid(self):
        """
        Get the UUID of the APA.
        """
        try:
            lut = pd.read_csv("data/uuid_lut.csv")
        except FileNotFoundError:
            return "❌ Lookup file not found at data/uuid_lut.csv"

        match = lut[lut["APA_name"] == self.apa_name]

        if match.empty:
            return f"⚠️ No UUID found for APA name: {self.apa_name}"

        return match.iloc[0]["UUID"]

    def has_data(self) -> bool:
        file_path = f"data/tension_data/tension_data_{self.apa_name}_{self.layer}.csv"

        if not os.path.exists(file_path):
            return False

        df = pd.read_csv(file_path, usecols=["side"])
        return any(df["side"].str.upper() == self.side.upper())

    def measure_calibrate(self, wire_number: int):
        x, y = get_xy()
        self.collect_wire_data(wire_number, x, y)

    def measure_list(self, wire_list: List[int]):
        # Step 1: Get valid (wire, x, y) triplets
        triplets = []
        for wire in wire_list:
            x, y = self.get_xy_from_file(wire)
            if x is not None and y is not None:
                triplets.append((wire, x, y))

        if not triplets:
            print("No valid wires with known coordinates.")
            return

        # Step 2: Sort greedily by minimizing distance
        current_x, current_y = get_xy()
        visited = []
        remaining = triplets.copy()

        while remaining:
            nearest = min(
                remaining,
                key=lambda triplet: math.hypot(
                    triplet[1] - current_x, triplet[2] - current_y
                ),
            )
            visited.append(nearest)
            remaining.remove(nearest)
            current_x, current_y = nearest[1], nearest[2]

        # Step 3: Measure in sorted order
        for wire, x, y in visited:
            if self.stop_event and self.stop_event.is_set():
                print("List measurement interrupted!")
                return
            self.collect_wire_data(wire, x, y)

    def measure_auto(self):
        if not self.has_data():
            print("No data available, please measure first.")
            return
        d = analyze_tension_data(self.apa_name, self.layer)
        wires_to_measure = d["missing_wires"]
        if len(wires_to_measure) == 0:
            print("All wires have been measured.")
            return
        else:
            print("Measuring missing wires...")
            # Measure the missing wires        print(f"Missing wires: {wires_to_measure}")
            self.measure_list(wires_to_measure)

    def collect_wire_data(self, wire_number: int, wire_x, wire_y):


        def save_audio_sample(audio_sample):
            if self.save_audio:
                np.savez(
                    f"audio/{self.layer}{self.side}{wire_number}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
                    audio_sample,
                )

        def collect_samples(start_time):
            nonlocal wire_x, wire_y
            wires = []
            wiggle_start_time = time.time()
            current_wiggle = self.starting_wiggle_step
            while (time.time() - start_time) < self.timeout:
                if self.stop_event and self.stop_event.is_set():
                    print("tension measurement interrupted!")
                    return
                audio_sample = record_audio(
                    self.record_duration, self.sample_rate, plot=False, normalize=True
                )
                save_audio_sample(audio_sample)
                if (
                    time.time() - wiggle_start_time > self.wiggle_interval
                    and self.use_wiggle
                ):
                    wiggle_start_time = time.time()
                    print(f"Wiggling {current_wiggle}mm")
                    wiggle(current_wiggle)
                if audio_sample is not None:
                    frequency, confidence, tension, tension_ok = analyze_sample(
                        audio_sample, self.sample_rate, length
                    )
                    x, y = get_xy()
                    if confidence > self.confidence_threshold and tension_plausible(
                        tension
                    ):
                        wiggle_start_time = time.time()
                        wires.append(
                            {
                                "tension": tension,
                                "tension_pass": tension_ok,
                                "frequency": frequency,
                                "confidence": confidence,
                                "x": x,
                                "y": y,
                            }
                        )
                        wire_y = np.average([d["y"] for d in wires])
                        current_wiggle = current_wiggle / 2 + 0.05

                        cluster = has_cluster_dict(
                            wires, "tension", self.samples_per_wire
                        )
                        if cluster != []:
                            return cluster
                        print(
                            f"tension: {tension:.1f}N, frequency: {frequency:.1f}Hz, "
                            f"confidence: {confidence * 100:.1f}%",
                            f"y: {y:.1f}",
                        )
            return []

        def generate_result(passingWires):
            nonlocal wire_x, wire_y
            result = {
                "layer": self.layer,
                "side": self.side,
                "wire_number": wire_number,
                "tension": 0,
                "tension_pass": False,
                "frequency": 0,
                "zone": zone_lookup(wire_x),
                "confidence": 0,
                "t_sigma": 0,
                "x": wire_x,
                "y": wire_y,
                "Gcode": f"X{round(wire_x, 1)} Y{round(wire_y, 1)}",
            }

            if len(passingWires) > 0:
                result["frequency"] = calculate_kde_max(
                    [d["frequency"] for d in passingWires]
                )
                result["tension"] = tension_lookup(
                    length=length, frequency=result["frequency"]
                )
                result["tension_pass"] = tension_pass(result["tension"], length)
                result["confidence"] = np.average(
                    [d["confidence"] for d in passingWires]
                )
                result["t_sigma"] = np.std([d["tension"] for d in passingWires])
                result["x"] = round(np.average([d["x"] for d in passingWires]), 1)
                result["y"] = round(np.average([d["y"] for d in passingWires]), 1)
                result["Gcode"] = f"X{round(result['x'], 1)} Y{round(result['y'], 1)}"
                result["wires"] = str([float(d["tension"]) for d in passingWires])

            return result

        if self.bypass_data_collection:
            result = {
                "layer": self.layer,
                "side": self.side,
                "wire_number": wire_number,
                "tension": 6,
                "tension_pass": True,
                "frequency": 90,
                "zone": zone_lookup(wire_x),
                "confidence": 100,
                "t_sigma": 0,
                "x": wire_x,
                "y": wire_y,
                "Gcode": f"X{round(wire_x, 1)} Y{round(wire_y, 1)}",
                "ttf": 0,
                "time": datetime.now(),
            }
            log_data(
                result,
                f"data/tension_data/tension_data_{self.apa_name}_{self.layer}.csv",
            )
            return result
        # Main logic
        length = length_lookup(self.layer, wire_number, zone_lookup(wire_x))
        start_time = time.time()

        goto_xy(wire_x, wire_y)

        wires = collect_samples(start_time)
        result = generate_result(wires)

        if result["tension"] == 0:
            print(f"measurement failed for wire number {wire_number}.")
        if not result["tension_pass"]:
            print(f"Tension failed for wire number {wire_number}.")
        ttf = time.time() - start_time
        print(
            f"Wire number {wire_number} has length {length * 1000:.1f}mm tension {result['tension']:.1f}N frequency {result['frequency']:.1f}Hz with confidence {result['confidence'] * 100:.1f}%.\n"
            f"Took {ttf} seconds to finish."
        )
        result["ttf"] = ttf
        log_data(
            result, f"data/tension_data/tension_data_{self.apa_name}_{self.layer}.csv"
        )

        return result

    def get_xy_from_file(self, wire_number: int) -> tuple[float, float] | None:
        apa_name = self.apa_name
        layer = self.layer
        side = self.side
        dx = self.dx
        dy = self.dy
        wire_min = self.wire_min
        wire_max = self.wire_max

        if wire_number < wire_min or wire_number > wire_max:
            print(f"Wire number {wire_number} is out of range for layer {layer}.")
            return None

        # Load the CSV file
        file_path = f"data/tension_data/tension_data_{apa_name}_{layer}.csv"
        expected_columns = [
            "layer",
            "side",
            "wire_number",
            "tension",
            "tension_pass",
            "frequency",
            "zone",
            "confidence",
            "t_sigma",
            "x",
            "y",
            "Gcode",
            "wires",
            "ttf",
            "time",
        ]

        try:
            df = pd.read_csv(file_path, skiprows=1, names=expected_columns)
        except FileNotFoundError:
            return f"❌ File not found: {file_path}"
        df_side = (
            df[df["side"].str.upper() == side.upper()]
            .sort_values("time")  # sort so latest is last
            .drop_duplicates(
                subset="wire_number", keep="last"
            )  # keep latest for each wire
            .sort_values("wire_number")  # optional: sort by wire number
            .reset_index(drop=True)
        )

        if df_side.empty:
            print(f"No data found for side '{side}' in file {file_path}")
            return None

        wire_numbers = df_side["wire_number"].values
        xs = df_side["x"].values
        ys = df_side["y"].values

        if wire_number in wire_numbers:
            idx = np.where(wire_numbers == wire_number)[0][0]
            x, y = xs[idx], ys[idx]
        elif wire_number < wire_numbers[0]:
            print("moving from first wire")
            delta_x = -dx * (wire_numbers[0] - wire_number)

            x, y = xs[0] + delta_x, ys[0]
        elif wire_number > wire_numbers[-1]:
            print("moving from last wire")
            delta_x = dx * (wire_number - wire_numbers[-1])

            x, y = xs[-1] + delta_x, ys[-1]
        else:
            print("moving by interpolation")
            lower_idx = np.max(np.where(wire_numbers < wire_number))
            upper_idx = np.min(np.where(wire_numbers > wire_number))
            wl, wu = wire_numbers[lower_idx], wire_numbers[upper_idx]
            xl, xu = xs[lower_idx], xs[upper_idx]
            yl, yu = ys[lower_idx], ys[upper_idx]
            fraction = (wire_number - wl) / (wu - wl)
            x = xl + fraction * (xu - xl)
            y = yl + fraction * (yu - yl)

        if layer in ["V", "U"]:
            refined = refine_position(x, y, dx, dy)
            return refined
        return (x, y)


if __name__ == "__main__":
    self = Tensiometer(
        apa_name="US_APA7",
        layer="V",
        side="B",
        starting_wiggle_step=0.3,
        samples_per_wire=3,
        confidence_threshold=0.6,
        use_wiggle=True,
        sound_card_name="default",
        timeout=100,
        save_audio=True,
        record_duration=0.15,
        wiggle_interval=1,
        test_mode=True,
    )
