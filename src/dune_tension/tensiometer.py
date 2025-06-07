import threading
import queue
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import time

import numpy as np
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
    check_stop_event,
)
from geometry import (
    zone_lookup,
    length_lookup,
)
from audioProcessing import analyze_sample, get_samplerate
from plc_io import is_web_server_active
from data_cache import get_dataframe, update_dataframe, EXPECTED_COLUMNS


@dataclass
class TensionResult:
    layer: str
    side: str
    wire_number: int
    frequency: float = 0.0
    confidence: float = 0.0
    x: float = 0.0
    y: float = 0.0
    wires: list[float] | None = None
    ttf: float = 0.0
    time: Optional[float] = None

    zone: int = field(init=False)
    wire_length: float = field(init=False)
    tension: float = field(init=False)
    tension_pass: bool = field(init=False)
    t_sigma: float = field(init=False)
    Gcode: str = field(init=False)

    def __post_init__(self) -> None:
        self.zone = zone_lookup(self.x)
        self.wire_length = length_lookup(self.layer, self.wire_number, self.zone)
        self.tension = tension_lookup(self.wire_length, self.frequency)
        self.tension_pass = tension_pass(self.tension, self.wire_length)
        wires_list = self.wires or []
        self.t_sigma = float(np.std(wires_list)) if hasattr(np, "std") else 0.0
        self.wires = str(wires_list)
        self.Gcode = f"X{round(self.x, 1)} Y{round(self.y, 1)}"


class Tensiometer:
    def __init__(
        self,
        apa_name: str,
        layer: str,
        side: str,
        flipped: bool = False,
        stop_event: Optional[threading.Event] = None,
        samples_per_wire: int = 3,
        confidence_threshold: float = 0.7,
        save_audio: bool = True,
        spoof: bool = False,
        spoof_movement: bool = False,
    ) -> None:
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
            from plc_io import (
                spoof_get_xy as get_xy,
                spoof_goto_xy as goto_xy,
                spoof_wiggle as wiggle,
            )

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

    def measure_calibrate(self, wire_number: int) -> Optional[TensionResult]:
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

    def measure_auto(self) -> None:
        from analyze import get_missing_wires

        wires_dict = get_missing_wires(self.config)
        wires_to_measure = wires_dict.get(self.config.side, [])

        print(f"Missing wires: {wires_to_measure}")
        if not wires_to_measure:
            print("All wires are already measured.")
            return

        print("Measuring missing wires...")
        print(f"Missing wires: {wires_to_measure}")
        for wire_number in wires_to_measure:
            if check_stop_event(self.stop_event):
                return
            print(f"Measuring wire {wire_number}...")
            x, y = get_xy_from_file(self.config, wire_number)
            self.collect_wire_data(wire_number=wire_number, wire_x=x, wire_y=y)

    def measure_list(self, wire_list: list[int], preserve_order: bool) -> None:
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

    def _collect_samples(
        self,
        wire_number: int,
        length: float,
        start_time: float,
        wire_y: float,
    ) -> tuple[list[TensionResult] | None, float]:
        wires: list[TensionResult] = []
        wiggle_start_time = time.time()
        current_wiggle = 0.5
        while (time.time() - start_time) < 30:
            if check_stop_event(self.stop_event, "tension measurement interrupted!"):
                return None, wire_y
            audio_sample = self.record_audio_func(
                duration=0.15, sample_rate=self.samplerate
            )
            if check_stop_event(self.stop_event, "tension measurement interrupted!"):
                return None, wire_y
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
                if check_stop_event(self.stop_event, "tension measurement interrupted!"):
                    return None, wire_y
                x, y = self.get_current_xy_position()
                if confidence > self.config.confidence_threshold and tension_plausible(
                    tension
                ):
                    wiggle_start_time = time.time()
                    wires.append(
                        TensionResult(
                            layer=self.config.layer,
                            side=self.config.side,
                            wire_number=wire_number,
                            frequency=frequency,
                            confidence=confidence,
                            x=x,
                            y=y,
                            wires=[tension],
                        )
                    )
                    wire_y = np.average([d.y for d in wires])
                    current_wiggle = (current_wiggle + 0.1) / 1.5
                    if self.config.samples_per_wire == 1:
                        return wires[:1], wire_y

                    cluster = has_cluster_dict(
                        wires, "tension", self.config.samples_per_wire
                    )
                    if cluster != []:
                        return cluster, wire_y
                    print(
                        f"tension: {tension:.1f}N, frequency: {frequency:.1f}Hz, "
                        f"confidence: {confidence * 100:.1f}%",
                        f"y: {y:.1f}",
                    )
        return (
            [] if not self.stop_event or not self.stop_event.is_set() else None
        ), wire_y

    def _generate_result(
        self,
        passing_wires: list[TensionResult],
        wire_number: int,
        wire_x: float,
        wire_y: float,
    ) -> TensionResult:
        if len(passing_wires) > 0:
            if self.config.samples_per_wire == 1:
                first = passing_wires[0]
                frequency = first.frequency
                confidence = first.confidence
                x = first.x
                y = first.y
                wires = [float(first.tension)]
            else:
                frequency = calculate_kde_max([d.frequency for d in passing_wires])
                confidence = np.average([d.confidence for d in passing_wires])
                x = round(np.average([d.x for d in passing_wires]), 1)
                y = round(np.average([d.y for d in passing_wires]), 1)
                wires = [float(d.tension) for d in passing_wires]
        else:
            frequency = 0.0
            confidence = 0.0
            x = wire_x
            y = wire_y
            wires = []

        result = TensionResult(
            layer=self.config.layer,
            side=self.config.side,
            wire_number=wire_number,
            frequency=frequency,
            confidence=confidence,
            x=x,
            y=y,
            wires=wires,
        )

        return result

    def collect_wire_data(
        self, wire_number: int, wire_x: float, wire_y: float
    ) -> Optional[TensionResult]:
        # Main logic
        length = length_lookup(self.config.layer, wire_number, zone_lookup(wire_x))
        start_time = time.time()

        if check_stop_event(self.stop_event):
            return

        # Check if this wire already has enough samples stored
        df = get_dataframe(self.config.data_path)
        df_wire = df[
            (df["layer"] == self.config.layer)
            & (df["side"] == self.config.side)
            & (df["wire_number"] == wire_number)
        ]
        df_wire = df_wire[
            (df_wire["confidence"] >= self.config.confidence_threshold)
            & df_wire["tension"].apply(tension_plausible)
        ]
        if not df_wire.empty:
            samples = [
                TensionResult(
                    layer=self.config.layer,
                    side=self.config.side,
                    wire_number=wire_number,
                    frequency=row["frequency"],
                    confidence=row["confidence"],
                    x=row["x"],
                    y=row["y"],
                    wires=[row["tension"]],
                )
                for _, row in df_wire.iterrows()
            ]
            cluster = has_cluster_dict(samples, "tension", self.config.samples_per_wire)
            if cluster:
                print(
                    f"Wire {wire_number} already has {len(cluster)} valid samples. Skipping collection."
                )
                return self._generate_result(cluster, wire_number, wire_x, wire_y)

        succeed = self.goto_xy_func(wire_x, wire_y)
        if check_stop_event(self.stop_event):
            return
        if not succeed:
            print(f"Failed to move to wire {wire_number} position {wire_x},{wire_y}.")
            return TensionResult(
                layer=self.config.layer,
                side=self.config.side,
                wire_number=wire_number,
                frequency=0.0,
                confidence=0.0,
                x=wire_x,
                y=wire_y,
                wires=[],
            )

        wires: list[TensionResult] = []
        sample_queue: queue.Queue = queue.Queue()
        record_stop = threading.Event()

        def record_loop() -> None:
            while (
                not record_stop.is_set()
                and not (self.stop_event and self.stop_event.is_set())
                and (time.time() - start_time) < 30
            ):
                audio_sample = self.record_audio_func(
                    duration=0.15, sample_rate=self.samplerate
                )
                if self.stop_event and self.stop_event.is_set():
                    break
                if audio_sample is not None:
                    if self.config.save_audio and not self.config.spoof:
                        np.savez(
                            f"audio/{self.config.layer}{self.config.side}{wire_number}_"
                            f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
                            audio_sample,
                        )
                    sample_queue.put(audio_sample)

        record_thread = threading.Thread(target=record_loop, daemon=True)
        record_thread.start()

        wiggle_start_time = time.time()
        current_wiggle = 0.5

        df = get_dataframe(self.config.data_path)
        while (time.time() - start_time) < 30:
            if check_stop_event(self.stop_event):
                record_stop.set()
                record_thread.join()
                return
            try:
                audio_sample = sample_queue.get(timeout=0.1)
            except queue.Empty:
                audio_sample = None

            if audio_sample is not None:
                frequency, confidence, tension, tension_ok = analyze_sample(
                    audio_sample, self.samplerate, length
                )
                if check_stop_event(self.stop_event):
                    record_stop.set()
                    record_thread.join()
                    return
                x, y = self.get_current_xy_position()
                if confidence > self.config.confidence_threshold and tension_plausible(
                    tension
                ):
                    wiggle_start_time = time.time()
                    sample = TensionResult(
                        layer=self.config.layer,
                        side=self.config.side,
                        wire_number=wire_number,
                        frequency=frequency,
                        confidence=confidence,
                        x=x,
                        y=y,
                        wires=[tension],
                    )
                    sample.time = time.time()
                    wires.append(sample)
                    row = {col: getattr(sample, col, None) for col in EXPECTED_COLUMNS}
                    df.loc[len(df)] = row
                    update_dataframe(self.config.data_path, df)
                    wire_y = np.average([d.y for d in wires])
                    current_wiggle = (current_wiggle + 0.1) / 1.5
                    df_wire = df[
                        (df["layer"] == self.config.layer)
                        & (df["side"] == self.config.side)
                        & (df["wire_number"] == wire_number)
                    ]
                    df_wire = df_wire[
                        (df_wire["confidence"] >= self.config.confidence_threshold)
                        & df_wire["tension"].apply(tension_plausible)
                    ]
                    samples = [
                        TensionResult(
                            layer=self.config.layer,
                            side=self.config.side,
                            wire_number=wire_number,
                            frequency=row["frequency"],
                            confidence=row["confidence"],
                            x=row["x"],
                            y=row["y"],
                            wires=[row["tension"]],
                        )
                        for _, row in df_wire.iterrows()
                    ]
                    cluster = has_cluster_dict(
                        samples, "tension", self.config.samples_per_wire
                    )
                    if cluster:
                        wires = cluster
                        break
                    print(
                        f"tension: {tension:.1f}N, frequency: {frequency:.1f}Hz, ",
                        f"confidence: {confidence * 100:.1f}%",
                        f"y: {y:.1f}",
                    )

            if time.time() - wiggle_start_time > 1:
                wiggle_start_time = time.time()
                print(f"Wiggling {current_wiggle}mm")
                self.wiggle_func(current_wiggle)

        record_stop.set()
        record_thread.join()
        if check_stop_event(self.stop_event):
            return

        result = self._generate_result(wires, wire_number, wire_x, wire_y)

        if result.tension == 0:
            print(f"measurement failed for wire number {wire_number}.")
        if not result.tension_pass:
            print(f"Tension failed for wire number {wire_number}.")
        ttf = time.time() - start_time
        print(
            f"Wire number {wire_number} has length {length * 1000:.1f}mm tension {result.tension:.1f}N frequency {result.frequency:.1f}Hz with confidence {result.confidence * 100:.1f}%.\n at {result.x},{result.y}\n"
            f"Took {ttf} seconds to finish."
        )
        result.ttf = ttf

        try:
            from analyze import update_tension_logs

            update_tension_logs(self.config)
        except Exception as exc:
            print(f"Failed to update logs: {exc}")

        return result

    def load_tension_summary(
        self,
    ) -> tuple[list, list] | tuple[str, list, list]:
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
