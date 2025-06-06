import threading
import numpy as np
from datetime import datetime
import time
import pandas as pd
from tension_calculation import (
    calculate_kde_max,
    tension_lookup,
    tension_pass,
    has_cluster_dict,
    tension_plausible,
)
from tensiometer_functions import (
    make_config,
    measure_list,
    get_xy_from_file,
)
from geometry import (
    zone_lookup,
    length_lookup,
)
from audioProcessing import analyze_sample, get_samplerate
from plc_io import is_web_server_active, goto_xy
from data_cache import get_dataframe, update_dataframe, EXPECTED_COLUMNS


class Tensiometer:
    def __init__(
        self,
        apa_name,
        layer,
        side,
        flipped=False,
        stop_event=None,
        samples_per_wire=3,
        confidence_threshold=0.7,
        save_audio=True,
        spoof=False,
        spoof_movement=False,
    ):
        self.config = make_config(
            apa_name=apa_name,
            layer=layer,
            side=side,
            flipped=flipped,
            samples_per_wire=samples_per_wire,
            confidence_threshold=confidence_threshold,
            save_audio=save_audio,
            spoof=spoof,
        )
        self.stop_event = stop_event or threading.Event()
        try:
            web_ok = is_web_server_active()
        except Exception:
            web_ok = False

        if not spoof_movement and web_ok:
            from plc_io import get_xy, goto_xy, wiggle
        else:
            from plc_io import spoof_get_xy as get_xy, spoof_goto_xy as goto_xy, spoof_wiggle as wiggle
            print(
                "Web server is not active or spoof_movement enabled. Using dummy functions."
            )
        self.get_current_xy_position = get_xy
        self.goto_xy_func = goto_xy
        self.wiggle_func = wiggle

        self.samplerate = get_samplerate()
        if self.samplerate is None or spoof:
            print("Using spoofed audio sample for testing.")
            from audioProcessing import spoof_audio_sample

            self.samplerate = 44100  # Default samplerate for spoofing
            self.record_audio_func = lambda duration, sample_rate: spoof_audio_sample(
                "audio"
            )
        else:
            from audioProcessing import record_audio

            self.record_audio_func = lambda duration, sample_rate: record_audio(
                0.15, sample_rate=sample_rate, normalize=True
            )

    def measure_calibrate(self, wire_number):
        xy = self.get_current_xy_position()
        if xy is None:
            print(
                f"No position data found for wire {wire_number}. Using current position."
            )
            (
                x,
                y,
            ) = self.get_current_xy_position()
        else:
            x, y = xy
            self.goto_xy_func(x, y)

        return self.collect_wire_data(
            wire_number=wire_number,
            wire_x=x,
            wire_y=y,
        )

    def measure_auto(self):
        from analyze_tension_data import analyze_tension_data

        result = analyze_tension_data(self.config)
        wires_to_measure = result.get("missing_wires", [])[self.config.side]

        print(f"Missing wires: {wires_to_measure}")
        if not wires_to_measure:
            print("All wires are already measured.")
            return

        print("Measuring missing wires...")
        print(f"Missing wires: {wires_to_measure}")
        for wire_number in wires_to_measure:
            if self.stop_event and self.stop_event.is_set():
                print("Measurement interrupted.")
                return
            print(f"Measuring wire {wire_number}...")
            x, y = get_xy_from_file(self.config, wire_number)
            self.collect_wire_data(wire_number=wire_number, wire_x=x, wire_y=y)

    def measure_list(self, wire_list, preserve_order):
        measure_list(
            config=self.config,
            wire_list=wire_list,
            get_xy_from_file_func=get_xy_from_file,
            get_current_xy_func=self.get_current_xy_position,
            collect_func=lambda w, x, y: self.collect_wire_data(
                wire_number=w,
                wire_x=x,
                wire_y=y,
            ),
            stop_event=self.stop_event,
            preserve_order=preserve_order,
        )

    def collect_wire_data(self, wire_number: int, wire_x, wire_y):
        def collect_samples(start_time):
            nonlocal wire_x, wire_y
            wires = []
            wiggle_start_time = time.time()
            current_wiggle = 0.5
            while (time.time() - start_time) < 30:
                if self.stop_event and self.stop_event.is_set():
                    print("tension measurement interrupted!")
                    return None
                audio_sample = self.record_audio_func(
                    duration=0.15, sample_rate=self.samplerate
                )
                if self.stop_event and self.stop_event.is_set():
                    print("tension measurement interrupted!")
                    return None
                if self.config.save_audio and not self.config.spoof:
                    np.savez(
                        f"audio/{self.config.layer}{self.config.side}{wire_number}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
                        audio_sample,
                    )
                if time.time() - wiggle_start_time > 1:
                    wiggle_start_time = time.time()
                    print(f"Wiggling {current_wiggle}mm")
                    self.wiggle_func(current_wiggle)
                if audio_sample is not None:
                    frequency, confidence, tension, tension_ok = analyze_sample(
                        audio_sample, self.samplerate, length
                    )
                    if self.stop_event and self.stop_event.is_set():
                        print("tension measurement interrupted!")
                        return None
                    x, y = self.get_current_xy_position()
                    if (
                        confidence > self.config.confidence_threshold
                        and tension_plausible(tension)
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
                        current_wiggle = (current_wiggle + 0.1) / 1.5

                        cluster = has_cluster_dict(
                            wires, "tension", self.config.samples_per_wire
                        )
                        if cluster != []:
                            return cluster
                        print(
                            f"tension: {tension:.1f}N, frequency: {frequency:.1f}Hz, "
                            f"confidence: {confidence * 100:.1f}%",
                            f"y: {y:.1f}",
                        )
            return [] if not self.stop_event or not self.stop_event.is_set() else None

        def generate_result(passingWires):
            nonlocal wire_x, wire_y
            result = {
                "layer": self.config.layer,
                "side": self.config.side,
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

        # Main logic
        length = length_lookup(self.config.layer, wire_number, zone_lookup(wire_x))
        start_time = time.time()

        if self.stop_event and self.stop_event.is_set():
            print("Measurement interrupted.")
            return

        succeed = goto_xy(wire_x, wire_y)
        if self.stop_event and self.stop_event.is_set():
            print("Measurement interrupted.")
            return
        if not succeed:
            print(f"Failed to move to wire {wire_number} position {wire_x},{wire_y}.")
            return {
                "layer": self.config.layer,
                "side": self.config.side,
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

        wires = collect_samples(start_time)
        if wires is None:
            print("Measurement interrupted.")
            return
        result = generate_result(wires)

        if result["tension"] == 0:
            print(f"measurement failed for wire number {wire_number}.")
        if not result["tension_pass"]:
            print(f"Tension failed for wire number {wire_number}.")
        ttf = time.time() - start_time
        print(
            f"Wire number {wire_number} has length {length * 1000:.1f}mm tension {result['tension']:.1f}N frequency {result['frequency']:.1f}Hz with confidence {result['confidence'] * 100:.1f}%.\n at {result['x']},{result['y']}\n"
            f"Took {ttf} seconds to finish."
        )
        result["ttf"] = ttf

        df = get_dataframe(self.config.data_path)
        row = {col: result.get(col, None) for col in EXPECTED_COLUMNS}
        df.loc[len(df)] = row
        update_dataframe(self.config.data_path, df)

        return result

    def load_tension_summary(self):
        try:
            df = pd.read_csv(self.config.data_path)
        except FileNotFoundError:
            return f"❌ File not found: {self.config.data_path}", [], []

        if "A" not in df.columns or "B" not in df.columns:
            return "⚠️ File missing required columns 'A' and 'B'", [], []

        # Convert columns to lists, preserving NaNs if present
        a_list = df["A"].tolist()
        b_list = df["B"].tolist()

        return a_list, b_list
