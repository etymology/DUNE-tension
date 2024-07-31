import numpy as np
import time
from datetime import datetime
from typing import Tuple, Callable, Dict
from Tensiometer import Tensiometer
from audioProcessing import (
    save_audio_data,
    get_pitch_crepe,
    # get_pitch_naive_fft,
    # get_pitch_autocorrelation,
)
from utilities import log_frequency_data, zone_lookup, tension_lookup, length_lookup

AnalysisFuncType = Callable[[np.ndarray, int], Tuple[float, float]]


# @timeit
def analyze_wire(
    t: Tensiometer, layer: str, side: str, wire_number: int, wire_x, wire_y
):
    t.goto_xy(wire_x, wire_y)
    wiggle_generator = t.wiggle(wire_y, t.wiggle_step)
    analysis_methods: Dict[str, AnalysisFuncType] = {
        "crepe": get_pitch_crepe,
        # "naive_fft": get_pitch_naive_fft,
        # "autocorrelation": get_pitch_autocorrelation,
    }
    start_time = time.time()
    wires = []
    good_wire_count = 0
    while good_wire_count <= t.tries_per_wire:
        # t.goto_xy(wire_x, wire_y)
        # t.servo_toggle()
        # time.sleep(t.delay_after_plucking)
        audio_signal = t.record_audio_normalize(t.record_duration, plot=False)
        if t.save_audio:
            save_audio_data(
                audio_signal,
                f"audio/{layer}{side}{wire_number}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.npz",
            )
        if audio_signal is not None:
            analysis = {
                method: func(audio_signal, t.samplerate)
                for method, func in analysis_methods.items()
            }
            for method, (frequency, confidence) in analysis.items():
                length = length_lookup(layer, wire_number, zone_lookup(wire_x))
                if length < 150:
                    tension_min = 0.0258 * length + 0.232
                else:
                    tension_min = 4
                tension = tension_lookup(
                    length=length,
                    frequency=frequency,
                )
                tension_pass = tension < 8.5 and tension > tension_min
                if not tension_pass:
                    tension_2 = tension_lookup(
                        length=length,
                        frequency=frequency / 2,
                    )
                    if tension_2 < 8.5 and tension_2 > tension_min:
                        tension = tension_2
                        frequency = frequency / 2
                        tension_pass = True
                if tension_pass and confidence > t.confidence_threshold:
                    good_wire_count += 1
                time_to_finish = time.time() - start_time
                time_at_finish = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                print(
                    f"length: {length}, tension: {tension}, frequency: {frequency}, confidence: {confidence}"
                )
                wire_data = {
                    "layer": layer,
                    "side": side,
                    "wire_number": wire_number,
                    "tension": tension,
                    "tension_pass": tension_pass,
                    "zone": zone_lookup(wire_x),
                    "frequency": frequency,
                    "confidence": confidence,
                    "method": method,
                    "x": round(wire_x, 1),
                    "y": round(wire_y, 1),
                    "Gcode": f"X{round(wire_x,1)} Y{round(wire_y,1)}",
                    "tries": good_wire_count,
                    "time_to_finish": round(time_to_finish, 2),
                    "measured_at": time_at_finish,
                }
                wires.append(wire_data)
        wire_y = next(wiggle_generator)

    tensionPassingWires = [
        d
        for d in wires
        if d.get("tension_pass", False)
        and d.get("confidence", 0) > t.confidence_threshold
    ]

    if not tensionPassingWires:
        return {
            "layer": layer,
            "side": side,
            "wire_number": wire_number,
            "tension": 0,
            "tension_pass": False,
            "zone": zone_lookup(wire_x),
            "frequency": 0,
            "confidence": 0,
            "method": method,
            "x": round(wire_x, 1),
            "y": round(wire_y, 1),
            "Gcode": f"X{round(wire_x,1)} Y{round(wire_y,1)}",
            "tries": 0,
            "time_to_finish": round(time_to_finish, 2),
            "measured_at": time_at_finish,
        }
    time_to_finish = time.time() - start_time
    time_at_finish = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result = max(tensionPassingWires, key=lambda x: x.get("confidence", float("-inf")))
    result["frequency"] = np.average([d["frequency"] for d in tensionPassingWires])
    result["tension"] = np.average([d["tension"] for d in tensionPassingWires])
    result["confidence"] = np.average([d["confidence"] for d in tensionPassingWires])
    result["x"] = np.average([d["x"] for d in tensionPassingWires])
    result["y"] = np.average([d["y"] for d in tensionPassingWires])
    result["time_to_finish"] = round(time_to_finish, 2)
    result["time_at_finish"] = time_at_finish
    print(
        f"Wire number {wire_number} has tension {result['tension']} frequency {result['frequency']} Hz with confidence {result['confidence']} using {result['method']}. \nTook {result['time_to_finish']} seconds to finish."
    )
    return result


# @timeit
def measure_sequential(
    t: Tensiometer,
    initial_wire_number: int,
    final_wire_number: int,
    direction: str,
    side: str,
    layer: str,
    use_relative_position=True,
):
    logfilename = f"data/frequency_data_{t.apa_name}_{layer}.csv"  # {initial_wire_number}-{final_wire_number}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"

    if initial_wire_number > final_wire_number:
        step = -1
    else:
        step = 1
    if layer in ["X", "G"]:
        dx, dy = 0.0, 2300 / 480
    elif layer in ["V", "U"]:
        if direction == "diagonal":
            dx, dy = (
                2.724553916,
                3.791617987,
            )  # Orthogonal to a triangle of x=5.75 y=8.0
            if layer == "U":
                dy *= -1
            if side == "B":
                dx *= -1
        elif direction == "vertical":
            dx, dy = 0, 5.75 * step
        elif direction == "horizontal":
            dx, dy = 8.0 * step, 0
    else:
        print("Invalid layer.")
        exit(1)
    dx, dy = dx * step, dy * step
    movetype = t.get_movetype()
    state = t.get_state()
    target_x, target_y = t.get_xy()
    print(
        f"Current position: x={target_x}, y={target_y}\nstate={state}, movetype={movetype}"
    )
    wire_number = initial_wire_number
    for wire_number in range(initial_wire_number, final_wire_number + 1, step):
        wire_data = analyze_wire(t, layer, side, wire_number, target_x, target_y)
        log_frequency_data(wire_data, logfilename)
        if wire_data["tension"] == 0:
            print(f"measurement failed for wire number {wire_number}.")
        if not wire_data["tension_pass"]:
            print(f"Tension failed for wire number {wire_number}.")
        if use_relative_position and wire_data:
            target_x, target_y = wire_data["x"], wire_data["y"]
        target_x += dx
        target_y += dy
        # t.goto_xy(target_x, target_y)

    print(f"Finished scanning from wire {initial_wire_number} to {final_wire_number}.")


if __name__ == "__main__":
    t = Tensiometer(
        apa_name="US_APA3",
        tries_per_wire=5,
        wiggle_step=0.3,
        wiggle_type="gaussian",
        confidence_threshold=0.5,
        delay_after_plucking=0.0,
        record_duration=0.1,
        save_audio=False,
    )
    measure_sequential(
        t,
        initial_wire_number=400,
        final_wire_number=401,
        direction="horizontal",
        side="B",
        layer="V",
        use_relative_position=True,
    )
    # recheck_wires = [109, 120, 121, 124, 129, 133, 161, 162, 164, 166, 173, 176, 177, 179]
    # side = "B"
    # layer = "X"
    # logfilename = f"data/frequency_data_{t.apa_name}_{layer}.csv"  # {initial_wire_number}-{final_wire_number}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"

    # for wire in [466]:  # range(113,481):
    #     wire_data = analyze_wire(
    #         t, layer, side, wire, 6400, 191.1 + (wire - 1) * (2300 / 480)
    #     )
    #     log_frequency_data(wire_data, logfilename)


# TODO: create a LUT for the wires and their positions and a function to measure from LUT
# TODO: fix sequential measurement to move in the right direction when final_wire_number < initial_wire_number
# TODO:
