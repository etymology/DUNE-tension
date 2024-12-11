import time
import numpy as np
from datetime import datetime
from Tensiometer import Tensiometer
from audioProcessing import (
    get_pitch_crepe,
)
import threading
from utilities import (
    log_data,
    zone_lookup,
    tension_lookup,
    length_lookup,
    calculate_kde_max,
    tension_pass,
)
from random import gauss
from time import sleep


def collect_wire_data(
    t: Tensiometer, wire_number: int, wire_x, wire_y
):
    t.stop_servo_event.clear()
    t.stop_wiggle_event.clear()

    def start_servo_thread():
        servo_thread = threading.Thread(target=t.servo_loop)
        servo_thread.start()
        return servo_thread

    def start_wiggle_thread():
        wiggle_thread = threading.Thread(target=t.wiggle_loop(wire_x, wire_y))
        wiggle_thread.start()
        return wiggle_thread

    def save_audio_sample(audio_sample):
        if t.save_audio:
            np.savez(
                f"audio/{t.layer}{t.side}{wire_number}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
                audio_sample,
            )

    def analyze_sample(audio_sample):
        frequency, confidence = get_pitch_crepe(audio_sample, t.sample_rate)
        tension = tension_lookup(length=length, frequency=frequency)
        tension_ok = tension_pass(tension, length)
        if not tension_ok and tension_pass(tension / 4, length):
            tension /= 4
            frequency /= 2
            tension_ok = True
        return frequency, confidence, tension, tension_ok

    def collect_samples(start_time, length):
        wires = []
        good_wire_count = 0
        while (
            time.time() - start_time
        ) < t.timeout and good_wire_count < t.samples_per_wire:
            t.set_xy_target(wire_x, gauss(wire_y, t.wiggle_step))
            audio_sample = t.record_audio(t.record_duration, plot=False, normalize=True)
            #  sd.rec(int(t.record_duration * t.sample_rate), samplerate=t.sample_rate, channels=1, dtype='float32')
            # # sd.wait()

            save_audio_sample(audio_sample)
            if audio_sample is not None:
                frequency, confidence, tension, tension_ok = analyze_sample(
                    audio_sample
                )
                if tension_ok and confidence > t.confidence_threshold:
                    good_wire_count += 1
                    wires.append(
                        {
                            "tension": tension,
                            "tension_pass": tension_ok,
                            "frequency": frequency,
                            "confidence": confidence,
                            "x": wire_x,
                            "y": wire_y,
                        }
                    )
                print(
                    f"Wire {wire_number} length: {length*1000:.1f}mm, tension: {tension:.1f}N, frequency: {frequency:.1f}Hz, confidence: {confidence*100:.1f}%"
                )
        return wires

    def calculate_passing_wires(wires):
        return [
            d
            for d in wires
            if d.get("tension_pass", False)
            and d.get("confidence", 0) > t.confidence_threshold
        ]

    def generate_result(passingWires):
        nonlocal wire_x, wire_y
        result = {
            "layer": t.layer,
            "side": t.side,
            "wire_number": wire_number,
            "tension": 0,
            "tension_pass": False,
            "zone": zone_lookup(wire_x),
            "frequency": 0,
            "confidence": 0,
            "x": wire_x,
            "y": wire_y,
            "Gcode": f"X{round(wire_x,1)} Y{round(wire_y,1)}",
        }
        if len(passingWires) == 1:
            result["frequency"] = passingWires[0]["frequency"]
            result["tension"] = passingWires[0]["tension"]
            result["tension_pass"] = passingWires[0]["tension_pass"]
            result["confidence"] = passingWires[0]["confidence"]
            result["x"] = passingWires[0]["x"]
            result["y"] = passingWires[0]["y"]
            result["Gcode"] = f"X{round(result['x'],1)} Y{round(result['y'],1)}"

        if len(passingWires) > 1:
            result["frequency"] = calculate_kde_max(
                [d["frequency"] for d in passingWires]
            )
            result["tension"] = tension_lookup(
                length=length, frequency=result["frequency"]
            )
            result["tension_pass"] = tension_pass(result["tension"], length)
            result["confidence"] = calculate_kde_max(
                [d["confidence"] for d in passingWires]
            )
            result["x"] = round(np.average([d["x"] for d in passingWires]), 1)
            result["y"] = round(np.average([d["y"] for d in passingWires]), 1)
            result["Gcode"] = f"X{round(result['x'],1)} Y{round(result['y'],1)}"

        return result

    # Main logic
    length = length_lookup(t.layer, wire_number, zone_lookup(wire_x))
    start_time = time.time()

    if t.use_servo:
        servo_thread = start_servo_thread()
    if t.use_wiggle:
        wiggle_thread = start_wiggle_thread()
    t.goto_xy(wire_x, wire_y)
    sleep(0.2)

    wires = collect_samples(start_time, length)
    if t.use_servo:
        t.stop_servo_event.set()
        servo_thread.join()
    if t.use_wiggle:
        t.stop_wiggle_event.set()
        wiggle_thread.join()

    passingWires = calculate_passing_wires(wires)
    result = generate_result(passingWires)

    if result["tension"] == 0:
        print(f"measurement failed for wire number {wire_number}.")
    if not result["tension_pass"]:
        print(f"Tension failed for wire number {wire_number}.")
    print(
        f"Wire number {wire_number} has length {length*1000:.1f}mm tension {result['tension']:.1f}N frequency {result['frequency']:.1f}Hz with confidence {result['confidence']*100:.1f}%.\n"
        f"Took {time.time() - start_time} seconds to finish."
    )
    log_data(result, f"data/frequency_data_{t.apa_name}_{t.layer}.csv")

    return result
