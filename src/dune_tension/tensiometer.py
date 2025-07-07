import threading
from datetime import datetime, timedelta
from typing import Optional, Callable
import time
import random
import numpy as np
import pandas as pd
from tension_calculation import calculate_kde_max, tension_plausible

try:
    from tension_calculation import has_cluster
except ImportError:  # fallback for older stubs
    from tension_calculation import has_cluster_dict as has_cluster
from tensiometer_functions import (
    make_config,
    measure_list,
    get_xy_from_file,
    check_stop_event,
)
from geometry import (
    zone_lookup,
    length_lookup,
    refine_position
)
from audioProcessing import analyze_sample, get_samplerate

from plc_io import is_web_server_active,increment
from data_cache import (
    get_dataframe,
    update_dataframe,
    get_samples_dataframe,
    update_samples_dataframe,
)
from results import TensionResult, RawSample, EXPECTED_COLUMNS


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
        plot_audio: bool = False,
        spoof: bool = False,
        spoof_movement: bool = False,
        start_servo_loop: Optional[Callable[[], None]] = None,
        stop_servo_loop: Optional[Callable[[], None]] = None,
        focus_wiggle: Optional[Callable[[float], None]] = None,
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
            plot_audio=plot_audio,
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
        self.wiggle_func = increment

        self.focus_wiggle_func = focus_wiggle or (lambda delta: None)

        self.start_servo_loop = start_servo_loop or (lambda: None)
        self.stop_servo_loop = stop_servo_loop or (lambda: None)

        # State tracking for winder wiggle thread
        self._wiggle_event: threading.Event | None = None
        self._wiggle_thread: threading.Thread | None = None

        self.samplerate = get_samplerate()
        if self.samplerate is None or spoof:
            print("Using spoofed audio sample for testing.")
            from audioProcessing import spoof_audio_sample

            self.samplerate = 44100  # Default samplerate for spoofing
            self.record_audio_func = (
                lambda duration, sample_rate: (spoof_audio_sample("audio"), 0.0)
            )
        else:
            from audioProcessing import record_audio

        self.record_audio_func = lambda duration, sample_rate: record_audio(
            duration, sample_rate=sample_rate, normalize=True
        )

    def start_wiggle(self) -> None:
        """Begin wiggling the winder in a background thread."""
        if self._wiggle_event and self._wiggle_event.is_set():
            return

        self._wiggle_event = threading.Event()
        self._wiggle_event.set()

        start_x, start_y = self.get_current_xy_position()

        def _run() -> None:
            while self._wiggle_event and self._wiggle_event.is_set():
                self.goto_xy_func(start_x, start_y + 1)
                if not self._wiggle_event.is_set():
                    break
                self.goto_xy_func(start_x, start_y - 1)
                time.sleep(0.01)

        self._wiggle_thread = threading.Thread(target=_run, daemon=True)
        self._wiggle_thread.start()

    def stop_wiggle(self) -> None:
        """Stop the background winder wiggle thread."""
        if not self._wiggle_event:
            return
        self._wiggle_event.clear()
        if self._wiggle_thread:
            self._wiggle_thread.join(timeout=0.1)
        self._wiggle_event = None
        self._wiggle_thread = None

    def _plot_audio(self, audio_sample) -> None:
        """Save a plot of the recorded audio sample to a temporary file."""
        try:
            import matplotlib.pyplot as plt  # Local import to avoid optional dep
        except Exception as exc:  # pragma: no cover - plotting is optional
            print(f"Failed to import matplotlib for plotting: {exc}")
            return

        try:
            from tempfile import NamedTemporaryFile

            plt.figure(figsize=(10, 4))
            plt.plot(audio_sample)
            plt.title("Recorded Audio Sample")
            plt.xlabel("Sample Index")
            plt.ylabel("Amplitude")
            plt.grid(True)
            plt.tight_layout()
            with NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                plt.savefig(tmp.name)
                print(f"Audio plot saved to {tmp.name}")
            plt.close()
        except Exception as exc:  # pragma: no cover - plotting is optional
            print(f"Failed to plot audio sample: {exc}")

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

        def low_numbered_wires_high(layer, side, flipped):
            """
            Low-numbered wires are high if (U xor B) matches
            whether we are in the normal configuration (not flipped).
            """
            return ((layer == "U") ^ (side == "B")) == (not flipped)

        low_numbered_high = low_numbered_wires_high(
            self.config.layer, self.config.side, self.config.flipped
        )

        wires_to_measure[:] = [
            x
            for x in wires_to_measure
            if (x >= 20 if low_numbered_high else x <= 1146 - 20)
        ]

        print("Measuring missing wires...")
        print(f"Missing wires: {wires_to_measure}")
        start_time = time.time()
        measured_count = 0
        for wire_number in wires_to_measure:

            if check_stop_event(self.stop_event):
                return

            xy = get_xy_from_file(self.config, wire_number)
            if xy is None:
                print(f"No position data found for wire {wire_number}")
            else:
                x, y = xy
                self.goto_xy_func(x, y)
                print(f"Measuring wire {wire_number} at position {x},{y}")
                wire_start = time.time()
                self.collect_wire_data(wire_number=wire_number, wire_x=x, wire_y=y)
                measured_count += 1
                elapsed = time.time() - start_time
                avg_time = elapsed / measured_count
                remaining = len(wires_to_measure) - measured_count
                est_remaining = avg_time * remaining
                print(
                    f"Estimated time remaining: {timedelta(seconds=int(est_remaining))}"
                )
        print("Done measuring all wires")

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
        wire_x: float,
        plc_direction: float,
        focus_direction: int,
    ) -> tuple[list[TensionResult] | None, float, float, int]:
        # Load any previously collected raw samples for this wire
        samples_df = get_samples_dataframe(self.config.data_path)
        mask = (
            (samples_df["apa_name"] == self.config.apa_name)
            & (samples_df["layer"] == self.config.layer)
            & (samples_df["side"] == self.config.side)
            & (samples_df["wire_number"] == wire_number)
            & (
                samples_df["confidence"].astype(float)
                >= self.config.confidence_threshold
            )
        )
        wires = [
            TensionResult(
                apa_name=row.apa_name,
                layer=row.layer,
                side=row.side,
                wire_number=int(row.wire_number),
                frequency=float(row.frequency),
                confidence=float(row.confidence),
                x=float(row.x),
                y=float(row.y),
                wires=[],
                time=(
                    datetime.fromisoformat(row.time)
                    if isinstance(row.time, str)
                    else row.time
                ),
            )
            for row in samples_df[mask].itertuples()
        ]
        cluster = has_cluster(wires, "tension", self.config.samples_per_wire)
        if cluster != []:
            print("already collected enough samples for this wire.")
            wire_y = np.average([d.y for d in wires])
            return cluster, wire_y, plc_direction, focus_direction
        wiggle_start_time = time.time()
        current_wiggle = 0.1
        last_amplitude = None
        measuring_timeout = 20  # seconds
        _,starting_amplitude = self.record_audio_func(
            duration=0.2, sample_rate=self.samplerate
        )
        self.goto_xy_func(wire_x, wire_y)
        while (time.time() - start_time) < measuring_timeout:
            if check_stop_event(self.stop_event, "tension measurement interrupted!"):
                return None, wire_y
            record_duration = .1 # seconds
            audio_sample,amplitude = self.record_audio_func(

                duration=record_duration, sample_rate=self.samplerate
            )
            if check_stop_event(self.stop_event, "tension measurement interrupted!"):
                return None, wire_y, plc_direction, focus_direction
            if audio_sample is not None and self.config.plot_audio:
                self._plot_audio(audio_sample)
            if self.config.save_audio and not self.config.spoof:
                np.savez(
                    f"audio/{self.config.layer}{self.config.side}{wire_number}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
                    audio_sample,
                )
            if audio_sample is not None and amplitude > starting_amplitude * 2:
                frequency, confidence, tension, tension_ok = analyze_sample(
                    audio_sample, self.samplerate, length
                )
                if check_stop_event(
                    self.stop_event, "tension measurement interrupted!"
                ):
                    return None, wire_y, plc_direction, focus_direction
                x, y = self.get_current_xy_position()
                
                if time.time() - wiggle_start_time > record_duration*3:
                    wiggle_start_time = time.time()
                    if random.choice([True, False]):  # wiggle PLC
                        if last_amplitude is not None and amplitude < last_amplitude:
                            plc_direction *= -1
                        increment = plc_direction * current_wiggle
                        wire_y += increment
                        self.goto_xy_func(wire_x, wire_y)
                        print(f"plc wiggle: {increment:.2f}mm confidence {confidence:.2f}")
                    else:  # wiggle focus
                        if last_amplitude is not None and amplitude < last_amplitude:
                            focus_direction *= -1
                        self.focus_wiggle_func(focus_direction * 100)
                        print(f"focus wiggle: {focus_direction * 100} confidence {confidence:.2f}")

                        last_amplitude = amplitude
                print(
                    f"tension: {tension:.1f}N, frequency: {frequency:.1f}Hz, "
                    f"confidence: {confidence * 100:.1f}%"
                    f"amplitude: {amplitude:.2f}"
                    f"y: {y:.1f}",
                )
                if confidence > self.config.confidence_threshold and tension_plausible(
                    tension
                ):
                    wires.append(
                        TensionResult(
                            apa_name=self.config.apa_name,
                            layer=self.config.layer,
                            side=self.config.side,
                            wire_number=wire_number,
                            frequency=frequency,
                            confidence=confidence,
                            x=x,
                            y=y,
                            wires=[tension],
                            time=datetime.now(),
                        )
                    )
                    # Store raw sample
                    samples_df = get_samples_dataframe(self.config.data_path)
                    raw = RawSample(
                        apa_name=self.config.apa_name,
                        layer=self.config.layer,
                        side=self.config.side,
                        wire_number=wire_number,
                        frequency=frequency,
                        confidence=confidence,
                        x=x,
                        y=y,
                        time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    )
                    samples_df.loc[len(samples_df)] = {
                        col: getattr(raw, col)
                        for col in raw.__dataclass_fields__.keys()
                    }
                    update_samples_dataframe(self.config.data_path, samples_df)
                    wire_y = np.average([d.y for d in wires])
                    current_wiggle = (current_wiggle + 0.1) / 1.5
                    if self.config.samples_per_wire == 1:
                        return wires[:1], wire_y, plc_direction, focus_direction

                    cluster = has_cluster(
                        wires, "tension", self.config.samples_per_wire
                    )
                    if cluster != []:
                        return cluster, wire_y, plc_direction, focus_direction
        return (
            [] if not self.stop_event or not self.stop_event.is_set() else None
        ), wire_y, plc_direction, focus_direction

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
                wires = [first.tension]
                x = first.x
                y = first.y
            else:
                frequency = calculate_kde_max([d.frequency for d in passing_wires])
                confidence = np.average([d.confidence for d in passing_wires])
                wires = [float(d.tension) for d in passing_wires]
                x = round(np.average([d.x for d in passing_wires]), 1)
                y = round(np.average([d.y for d in passing_wires]), 1)
        else:
            frequency = 0.0
            confidence = 0.0
            wires = []
            x = wire_x
            y = wire_y

        result = TensionResult(
            apa_name=self.config.apa_name,
            layer=self.config.layer,
            side=self.config.side,
            wire_number=wire_number,
            frequency=frequency,
            confidence=confidence,
            x=x,
            y=y,
            wires=wires,
            time=datetime.now(),
        )

        return result

    def collect_wire_data(
        self, wire_number: int, wire_x: float, wire_y: float
    ) -> Optional[TensionResult]:
        length = length_lookup(self.config.layer, wire_number, zone_lookup(wire_x))
        start_time = time.time()

        if check_stop_event(self.stop_event):
            return

        succeed = self.goto_xy_func(wire_x, wire_y)
        if check_stop_event(self.stop_event):
            return
        if not succeed:
            print(f"Failed to move to wire {wire_number} position {wire_x},{wire_y}.")
            return TensionResult(
                apa_name=self.config.apa_name,
                layer=self.config.layer,
                side=self.config.side,
                wire_number=wire_number,
                frequency=0.0,
                confidence=0.0,
                x=wire_x,
                y=wire_y,
                wires=[],
                time=datetime.now(),
            )

        self.start_servo_loop()
        try:
            plc_dir = getattr(self, "_plc_direction", 1.0)
            focus_dir = getattr(self, "_focus_direction", 1)
            wires, wire_y, plc_dir, focus_dir = self._collect_samples(
                wire_number=wire_number,
                length=length,
                start_time=start_time,
                wire_y=wire_y,
                wire_x=wire_x,
                plc_direction=plc_dir,
                focus_direction=focus_dir,
            )
            self._plc_direction = plc_dir
            self._focus_direction = focus_dir
        finally:
            self.stop_servo_loop()

        if wires is None:
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
        result.time = datetime.now()

        df = get_dataframe(self.config.data_path)
        row = {col: getattr(result, col, None) for col in EXPECTED_COLUMNS}
        if isinstance(row.get("time"), datetime):
            row["time"] = row["time"].isoformat()
        if isinstance(row.get("wires"), list):
            row["wires"] = str(row["wires"])
        df.loc[len(df)] = row
        update_dataframe(self.config.data_path, df)

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

    def close(self) -> None:
        """Stop any active audio streams used by the tensiometer."""
        try:
            import sounddevice as sd  # Local import to avoid mandatory dependency

            sd.stop()
        except Exception:
            pass
