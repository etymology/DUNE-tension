import time
import numpy as np
from datetime import datetime
from Tensiometer import Tensiometer
from audioProcessing import get_pitch_crepe, get_pitch_crepe_bandpass
import threading
from utilities import (
    log_data,
    zone_lookup,
    tension_lookup,
    length_lookup,
    tension_pass,
    has_cluster_dict,
    tension_plausible,
    calculate_kde_max,
)


def collect_wire_data(t: Tensiometer, wire_number: int, wire_x, wire_y):
    t.stop_servo_event.clear()
    t.stop_wiggle_event.clear()

    def start_servo_thread():
        servo_thread = threading.Thread(target=t.servo_loop)
        servo_thread.start()
        return servo_thread

    def save_audio_sample(audio_sample):
        if t.save_audio:
            np.savez(
                f"audio/{t.layer}{t.side}{wire_number}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
                audio_sample,
            )

    def analyze_sample(audio_sample):
        frequency, confidence = get_pitch_crepe_bandpass(audio_sample, t.sample_rate, length)
        tension = tension_lookup(length=length, frequency=frequency)
        tension_ok = tension_pass(tension, length)
        if (
            not tension_ok
            and confidence > t.confidence_threshold
            and tension_pass(tension / 4, length)
        ):
            tension /= 4
            frequency /= 2
            tension_ok = True
        return frequency, confidence, tension, tension_ok

    def collect_samples(start_time):
        nonlocal wire_x, wire_y
        wires = []
        wiggle_start_time = time.time()
        current_wiggle = t.starting_wiggle_step
        while (time.time() - start_time) < t.timeout:
            audio_sample = t.record_audio(t.record_duration, plot=False, normalize=True)
            save_audio_sample(audio_sample)
            if time.time() - wiggle_start_time > t.wiggle_interval and t.use_wiggle:
                wiggle_start_time = time.time()
                print("Wiggling")
                t.wiggle(wire_y, current_wiggle)
            if audio_sample is not None:
                frequency, confidence, tension, tension_ok = analyze_sample(
                    audio_sample
                )
                x, y = t.get_xy()
                if confidence > t.confidence_threshold and tension_plausible(tension):
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
                    print(
                        f"tension: {tension:.1f}N, frequency: {frequency:.1f}Hz, "
                        f"confidence: {confidence * 100:.1f}%",
                        f"x: {x:.1f} y: {y:.1f}",
                    )
                    wire_y = np.average([d["y"] for d in wires])
                    # current_wiggle/=1.5
                    if t.samples_per_wire == 1:
                        return wires
                    else:
                        cluster = has_cluster_dict(wires, "tension", t.samples_per_wire)
                        if cluster != []:
                            return cluster
        return []

    def generate_result(passingWires):
        nonlocal wire_x, wire_y
        result = {
            "layer": t.layer,
            "side": t.side,
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
            result["confidence"] = np.average([d["confidence"] for d in passingWires])
            result["t_sigma"] = np.std([d["tension"] for d in passingWires])
            result["x"] = round(np.average([d["x"] for d in passingWires]), 1)
            result["y"] = round(np.average([d["y"] for d in passingWires]), 1)
            result["Gcode"] = f"X{round(result['x'], 1)} Y{round(result['y'], 1)}"
            result["wires"] = str([float(d["tension"]) for d in passingWires])

        return result

    # Main logic
    length = length_lookup(t.layer, wire_number, zone_lookup(wire_x))
    start_time = time.time()

    if t.use_servo:
        servo_thread = start_servo_thread()
    t.goto_xy(wire_x, wire_y)

    wires = collect_samples(start_time)
    if t.use_servo:
        t.stop_servo_event.set()
        servo_thread.join()

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
    log_data(result, f"data/frequency_data_{t.apa_name}_{t.layer}.csv")

    return result
