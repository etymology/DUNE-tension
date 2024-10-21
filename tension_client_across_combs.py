import time
import numpy as np
from datetime import datetime
from Tensiometer import Tensiometer
from audioProcessing import (
    save_wav,
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
    get_wire_coordinates,
    next_wire_target,
)
from random import gauss


def collect_wire_data(
    t: Tensiometer, layer: str, side: str, wire_number: int, wire_x, wire_y
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
            save_wav(
                audio_sample=audio_sample,
                filename=f"audio/{layer}{side}{wire_number}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.wav",
                sample_rate=t.sample_rate,
            )

    def analyze_sample(audio_sample):
        frequency, confidence = get_pitch_crepe(audio_sample, t.sample_rate)
        tension = tension_lookup(length=length, frequency=frequency)
        if layer in ["X", "G"]:
            tension_ok = tension > 4 and tension < 8.5
        else:
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
        ) < t.timeout and good_wire_count <= t.samples_per_wire:
            t.set_xy_target(wire_x, gauss(wire_y, t.wiggle_step))
            audio_sample = t.record_audio(
                t.record_duration, plot=False, normalize=False
            )
            save_audio_sample(audio_sample)
            if audio_sample is not None:
                frequency, confidence, tension, tension_ok = analyze_sample(
                    audio_sample
                )
                if tension_ok and confidence > t.confidence_threshold:
                    good_wire_count += 1
                print(
                    f"Wire {wire_number} length: {length*1000:.1f}mm, tension: {tension:.1f}N, frequency: {frequency:.1f}Hz, confidence: {confidence*100:.1f}%"
                )
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
            "layer": layer,
            "side": side,
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
        if passingWires is not None and len(passingWires) > 1:
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
            log_data(
                {
                    "side": side,
                    "wire_number": wire_number,
                    "x": result["x"],
                    "y": result["y"],
                },
                f"data/wireLUTs/{t.apa_name}_{layer}.csv",
            )
        return result

    # Main logic
    length = length_lookup(layer, wire_number, zone_lookup(wire_x))
    start_time = time.time()

    if t.use_servo:
        servo_thread = start_servo_thread()
    if t.use_wiggle:
        wiggle_thread = start_wiggle_thread()
    else:
        t.goto_xy(wire_x, wire_y)
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
    log_data(result, f"data/frequency_data_{t.apa_name}_{layer}.csv")

    return result


def measure_sequential_across_combs(
    t: Tensiometer,
    side: str,
    layer: str,
    initial_wire_number: int,
    direction: int = 1,
    use_relative_position: bool = False,
):
    # direction = 1 for increasing wire number, -1 for decreasing wire number

    if layer in ["X", "G"]:
        dx, dy = 0.0, 2300 / 480
        wire_min, wire_max = 1, 480
        if layer == "G":
            wire_max = 481
    else:
        dx, dy = 8.0, 5.75
        wire_min, wire_max = 8, 1151
        if (layer == "U" and side == "A") or (layer == "V" and side == "B"):
            dy = -5.75

    wire_number = initial_wire_number

    def measure_horizontal_layer():
        nonlocal wire_number
        y = 189.958 + dy * (wire_number - 1)  # for testing (199.1, 1XB)

        t.goto_xy(6300, y)

        while wire_number >= wire_min and wire_number <= wire_max:
            wire_data = collect_wire_data(t, layer, side, wire_number)
            wire_number += direction
            if use_relative_position:
                y = wire_data["y"]
            else:
                x, y = t.get_xy()
            y += dy
            t.goto_xy(6300, y)

    def measure_diagonal_layer():
        nonlocal wire_number
        wire_x, wire_y = t.get_xy()  # for testing
        while wire_number <= wire_max and wire_number >= wire_min:
            wire_data = collect_wire_data(t, layer, side, wire_number, wire_x, wire_y)
            if use_relative_position:
                wire_x, wire_y = wire_data["x"], wire_data["y"]
            wire_x, wire_y = next_wire_target(wire_x, wire_y, dx, dy, direction)
            wire_number += direction

    if layer in ["X", "G"]:
        measure_horizontal_layer()
    else:
        measure_diagonal_layer()


def measure_LUT(t: Tensiometer, layer: str, side: str, wire_numbers_to_measure: list):
    if layer in ["X", "G"]:
        for wire_number in wire_numbers_to_measure:
            collect_wire_data(
                t,
                layer,
                side,
                wire_number,
                6300,
                199.1 + 2300 / 480 * (wire_number - 1),
            )
    else:
        for wire_number in wire_numbers_to_measure:
            wire_x, wire_y = get_wire_coordinates(t.apa_name, layer, side, wire_number)
            if wire_x is not None and wire_y is not None:
                collect_wire_data(t, layer, side, wire_number, wire_x, wire_y)
            else:
                print(f"Wire {wire_number} not found in LUT.")


def seek_wire(t: Tensiometer, layer, side, wire_number):
    x, y = t.get_xy()
    max_confidence = 0.0
    best_y = y
    i = 0.0
    wiggle_step = 0.25
    while i < 20:
        wire_data = collect_wire_data(
            t,
            layer,
            side,
            wire_number,
            x,
            y + (-1) ** (i // 2 + 1) * (i // 2 + 1) * wiggle_step,
        )
        confidence = wire_data["confidence"]
        if confidence > max_confidence:
            max_confidence = confidence
            best_y = y + (-1) ** (i // 2 + 1) * (i // 2 + 1) * wiggle_step
        i += 1
    print(f"Best y: {best_y}, confidence: {max_confidence}")
    t.goto_xy(x, best_y)
    return x, best_y


if __name__ == "__main__":
    t = Tensiometer(
        apa_name="US_APA1",
        record_duration=0.2,
        wiggle_step=0.2,
        timeout=30,
        samples_per_wire=3,
        confidence_threshold=0.7,
        use_wiggle=False,
        save_audio=True,
        use_servo=False,
    )
    # print(seek_wire(t, "V", "B", 400))
    measure_sequential_across_combs(
        t,
        initial_wire_number=8,
        direction=1,
        side="B",
        layer="U",
        use_relative_position=False,
    )
    # measure_LUT(t, "V", "A", [749,7])
