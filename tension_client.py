import numpy as np
import time
from datetime import datetime
from typing import Tuple, Callable, Dict
from Tensiometer import Tensiometer
from audioProcessing import (
    save_wav,
    get_pitch_crepe,
    # get_pitch_naive_fft,
    # get_pitch_autocorrelation,
)
from utilities import (
    log_data,
    zone_lookup,
    tension_lookup,
    length_lookup,
    calculate_kde_max,
    get_wire_coordinates,
    tension_pass,
)

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
    length = length_lookup(layer, wire_number, zone_lookup(wire_x))

    while good_wire_count <= t.samples_per_wire:
        t.goto_xy(wire_x, wire_y)
        t.servo_toggle()
        time.sleep(t.delay_after_plucking)
        audio_sample = t.record_audio(t.record_duration, plot=False)
        if t.save_audio:
            save_wav(
                audio_sample=audio_sample,
                filename=f"audio/{layer}{side}{wire_number}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.wav",
                sample_rate=t.sample_rate,
            )
        if audio_sample is not None:
            analysis = {
                method: func(audio_sample, t.sample_rate)
                for method, func in analysis_methods.items()
            }
            for method, (frequency, confidence) in analysis.items():
                tension = tension_lookup(
                    length=length,
                    frequency=frequency,
                )
                if tension_pass(tension, length):
                    tension_ok = True
                elif tension_pass(tension / 4, length):
                    tension = tension / 4
                    tension_ok = True
                else:
                    tension_ok = False
                if tension_ok and confidence > t.confidence_threshold:
                    good_wire_count += 1
                print(
                    f"length: {length}, tension: {tension}, frequency: {frequency}, confidence: {confidence}"
                )
                sample_analysis = {
                    "tension": tension,
                    "tension_pass": tension_ok,
                    "frequency": frequency,
                    "confidence": confidence,
                    "method": method,
                    "x": wire_x,
                    "y": wire_y,
                }
                wires.append(sample_analysis)
        wire_y = next(wiggle_generator)

    time_to_finish = time.time() - start_time
    time_at_finish = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    tensionPassingWires = [
        d
        for d in wires
        if d.get("tension_pass", False)
        and d.get("confidence", 0) > t.confidence_threshold
    ]
    result = {
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
    if not tensionPassingWires:
        return result

    result["frequency"] = calculate_kde_max(
        [d["frequency"] for d in tensionPassingWires]
    )
    result["tension"] = tension_lookup(length=length, frequency=result["frequency"])
    result["confidence"] = np.average([d["confidence"] for d in tensionPassingWires])
    best_x, best_y = (
        np.average([d["x"] for d in tensionPassingWires]),
        np.average([d["y"] for d in tensionPassingWires]),
    )
    result["x"] = best_x
    result["y"] = best_y
    result["Gcode"] = f"X{round(result['x'],1)} Y{round(result['y'],1)}"
    time_to_finish = time.time() - start_time
    time_at_finish = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result["time_to_finish"] = round(time_to_finish, 2)
    result["time_at_finish"] = time_at_finish
    log_data(
        {"wire_number": wire_number, "x": best_x, "y": best_y},
        f"data/wireLUTs/{t.apa_name}_{layer}",
    )
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
        log_data(wire_data, logfilename)
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

def measure_list(t: Tensiometer, wire_numbers_to_measure: list, layer: str, side: str):
    for wire_number in wire_numbers_to_measure:
        if get_wire_coordinates(t.apa_name, layer, side, wire_number) is not None:
            wire_x, wire_y = get_wire_coordinates(t.apa_name, layer, side, wire_number)
            wire_data = analyze_wire(t, layer, side, wire_number, wire_x, wire_y)
            if wire_data["tension"] == 0:
                print(f"measurement failed for wire number {wire_number}.")
            if not wire_data["tension_pass"]:
                print(f"Tension failed for wire number {wire_number}.")
            log_data(wire_data, f"data/frequency_data_{t.apa_name}_{layer}.csv")
            print(f"Finished scanning wire {wire_number}.")

if __name__ == "__main__":
    t = Tensiometer(
        apa_name="US_APA3",
        samples_per_wire=10,
        wiggle_step=0.3,
        wiggle_type="gaussian",
        confidence_threshold=0.5,
        delay_after_plucking=0.2,
        record_duration=0.2,
        save_audio=False,
    )
    measure_sequential(
        t,
        initial_wire_number=1146,
        final_wire_number=400,
        direction="vertical",
        side="B",
        layer="U",
        use_relative_position=True,
    )
