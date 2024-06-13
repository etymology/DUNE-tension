import numpy as np
import time
from datetime import datetime
from typing import Tuple, Callable, Dict, Any
from Tensiometer import Tensiometer
from audioProcessing import (
    save_audio_data,
    get_pitch_crepe,
)  # , get_pitch_naive_fft, get_pitch_autocorrelation
from utilities import log_frequency_data

AnalysisFuncType = Callable[[np.ndarray, int], Tuple[float, float]]


# @timeit
def analyze_wire(
    t: Tensiometer,
    layer: str,
    side: str,
    wire_number: int,
    wire_x: float,
    wire_y: float,
    filename: str,
):
    wiggle_generator = t.wiggle(wire_y)
    analysis_methods: Dict[str, AnalysisFuncType] = {
        "crepe": get_pitch_crepe,
        # "naive_fft": get_pitch_naive_fft,
        # "autocorrelation": get_pitch_autocorrelation
    }
    start_time = time.time()
    for i in range(t.tries_per_wire):
        x_target, y_target = wire_x, next(wiggle_generator)
        t.goto_xy(x_target, y_target)
        t.servo_toggle()
        time.sleep(t.delay_after_plucking)
        audio_signal = t.record_audio_normalize(t.record_duration, plot=False)
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
                if (
                    analysis["crepe"][1] > t.confidence_threshold
                    and analysis["crepe"][0] < t.max_frequency
                ):
                    time_to_finish = time.time() - start_time
                    time_at_finish = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    wire_data = {
                        "frequency": frequency,
                        "confidence": confidence,
                        "wire_number": wire_number,
                        "method": method,
                        "x": round(wire_x, 1),
                        "y": round(wire_y, 1),
                        "Gcode": f"X{round(wire_x,1)} Y{round(wire_y,1)}",
                        "tries": i,
                        "time_to_finish": round(time_to_finish, 2),
                        "measured_at": time_at_finish,
                        "success": True,
                    }
                    log_frequency_data(wire_data, filename=filename)
                    print(
                        f"Wire number {wire_number} has frequency {frequency} Hz with confidence {confidence} using {method}. \nTook {time_to_finish:.4f} seconds to finish."
                    )
                    return wire_data
    time_to_finish = time.time() - start_time
    time_at_finish = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    wire_data = {
        "frequency": 0,
        "confidence": 0,
        "wire_number": wire_number,
        "method": "FAILED",
        "x": round(wire_x, 1),
        "y": round(wire_y, 1),
        "Gcode": f"X{round(wire_x,1)} Y{round(wire_y,1)}",
        "tries": i,
        "time_to_finish": round(time_to_finish, 2),
        "measured_at": time_at_finish,
        "success": False,
    }
    log_frequency_data(wire_data, filename=filename)
    print(
        f"FAILED ---------------- Wire {wire_number} after {wire_data['tries']} tries and {wire_data['time_to_finish']:.4f} seconds. \nMoving on to the next wire."
    )
    return wire_data


# @timeit
def measure_sequential(
    t: Tensiometer,
    initial_wire_number: int,
    final_wire_number: int,
    diagonal: bool,
    side: str,
    layer: str,
    use_relative_position=True,
):
    logfilename = f"data/frequency_data_{layer}{side}_{t.apa_name}.csv"  # {initial_wire_number}-{final_wire_number}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"

    if initial_wire_number > final_wire_number:
        step = -1
    else:
        step = 1
    if layer in ["X", "G"]:
        dx, dy = 0.0, 2300 / 480
    elif layer in ["V", "U"]:
        if diagonal:
            dx, dy = (
                -2.724553916 * step,
                -3.791617987 * step,
            )  # Orthogonal to a triangle of x=5.75 y=8.0
        else:
            dx, dy = 0, 5.75 * step
    else:
        print("Invalid layer.")
        exit(1)
    if layer == "U":
        dy *= -1
    if side == "B":
        dx *= -1

    movetype = t.get_movetype()
    state = t.get_state()
    target_x, target_y = t.get_xy()
    print(
        f"Current position: x={target_x}, y={target_y}\nstate={state}, movetype={movetype}"
    )
    failed_wires = {}
    wire_number = initial_wire_number
    for wire_number in range(initial_wire_number, final_wire_number + 1, step):
        wire_data = analyze_wire(
            t, layer, side, wire_number, target_x, target_y, filename=logfilename
        )
        if not wire_data["success"]:
            failed_wires[wire_number] = (wire_data["x"], wire_data["y"])
        if use_relative_position and wire_data:
            target_x, target_y = wire_data["x"], wire_data["y"]
        target_x += dx
        target_y += dy

    print(
        f"Finished scanning from wire {initial_wire_number} to {final_wire_number} with {failed_wires}failed wires {failed_wires}."
    )
    return failed_wires


# @timeit
def measure_selected(t: Tensiometer, wires: Dict[str, Any], side: str, layer: str):
    start, end = min(wires.keys()), max(wires.keys())
    file_path = f"data/frequency_data_{layer}{side}_{t.apa_name}.csv"

    failed_wires = {}
    for wire_number in wires.keys():
        x, y = wires[wire_number]["x"], wires[wire_number]["y"]
        t.goto_xy(x, y)
        wire_data = analyze_wire(
            t, layer, side, int(wire_number), x, y, filename=file_path
        )
        if not wire_data["success"]:
            failed_wires[wire_number] = (wire_data["x"], wire_data["y"])
    print(
        f"Finished scanning from specified wired between {start} and {end} with failed wires {failed_wires}."
    )
    return failed_wires


if __name__ == "__main__":
    t = Tensiometer(apa_name="US_APA2")
    measure_sequential(
        t,
        initial_wire_number=8,
        final_wire_number=9999,
        diagonal=True,
        side="B",
        layer="U",
    )

    # wires_to_recheck = load_csv_to_dict('recheck/recheckB5.csv')
    # measure_selected(t, wires_to_recheck, side="B", layer="V",
    #                  tries_per_wire=50,
    #                  record_duration=0.2,
    #                  confidence_threshold=0.8,
    #                  wiggle_type='gaussian')
