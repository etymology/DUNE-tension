import threading
from datetime import datetime, timedelta
from typing import Optional, Callable
import time
import numpy as np
import pandas as pd
from random import gauss, choice

try:
    from tension_calculation import wire_equation, tension_plausible
except ImportError:  # fallback for older stubs
    pass
from spectrum_analysis.pitch_compare_config import PitchCompareConfig
from tensiometer_functions import (
    make_config,
    measure_list,
    get_xy_from_file,
    check_stop_event,
)
from geometry import zone_lookup, length_lookup
from audioProcessing import get_samplerate, get_noise_threshold
from spectrum_analysis.crepe_analysis import estimate_pitch_from_audio
from spectrum_analysis.audio_processing import acquire_audio

try:
    from plc_io import is_web_server_active, increment, set_speed, reset_plc
except Exception:  # pragma: no cover - fallback for older stubs
    from plc_io import is_web_server_active, increment

    def set_speed(*_args, **_kwargs):
        pass

    def reset_plc(*_args, **_kwargs):
        pass


from data_cache import (
    get_dataframe,
    update_dataframe,
)
from results import TensionResult, EXPECTED_COLUMNS


class Tensiometer:
    def __init__(
        self,
        apa_name: str,
        layer: str,
        side: str,
        flipped: bool = False,
        a_taped: bool = False,
        b_taped: bool = False,
        stop_event: Optional[threading.Event] = None,
        samples_per_wire: int = 1,
        confidence_threshold: float = 2,
        save_audio: bool = True,
        plot_audio: bool = False,
        record_duration: float = 0.5,
        measuring_duration: float = 10.0,
        snr: float = 1,
        spoof: bool = False,
        spoof_movement: bool = False,
        strum: Optional[Callable[[], None]] = None,
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
            record_duration=record_duration,
            measuring_duration=measuring_duration,
        )
        self.stop_event = stop_event or threading.Event()
        self.snr = snr
        self.noise_threshold = get_noise_threshold()
        try:
            web_ok = is_web_server_active()
        except Exception:
            web_ok = False

        if not spoof_movement and web_ok:
            from plc_io import get_xy, goto_xy
        else:
            from plc_io import (
                spoof_get_xy as get_xy,
                spoof_goto_xy as goto_xy,
            )

            print(
                "Web server is not active or spoof_movement enabled. Using dummy functions."
            )
        self.get_current_xy_position = get_xy
        self.goto_xy_func = goto_xy
        self.wiggle_func = increment

        self.focus_wiggle_func = focus_wiggle or (lambda delta: None)

        self.strum_func = strum or (lambda: None)

        self.a_taped = bool(a_taped)
        self.b_taped = bool(b_taped)

        # State tracking for winder wiggle thread
        self._wiggle_event: threading.Event | None = None
        self._wiggle_thread: threading.Thread | None = None

        self.samplerate = get_samplerate()
        if self.samplerate is None or spoof:
            print("Using spoofed audio sample for testing.")
            from audioProcessing import spoof_audio_sample

            self.samplerate = 44100  # Default samplerate for spoofing
            self.record_audio_func = lambda duration, sample_rate: (
                spoof_audio_sample("audio"),
                0.0,
            )
        else:
            from audioProcessing import record_audio_filtered

        self.record_audio_func = lambda duration, sample_rate: record_audio_filtered(
            duration, sample_rate=sample_rate, normalize=True
        )

    def _is_current_side_taped(self) -> bool:
        side = self.config.side.upper()
        if side == "A":
            return self.a_taped
        if side == "B":
            return self.b_taped
        return False

    def start_wiggle(self) -> None:
        """Begin wiggling the winder in a background thread."""
        if self._wiggle_event and self._wiggle_event.is_set():
            return

        self._wiggle_event = threading.Event()
        self._wiggle_event.set()

        start_x, start_y = self.get_current_xy_position()
        # Wiggle by roughly half the wire pitch to avoid hitting adjacent wires
        wiggle_width = 0  # abs(getattr(self.config, "dy", 5.0) / 20)

        def _run() -> None:
            while self._wiggle_event and self._wiggle_event.is_set():
                self.goto_xy_func(start_x, gauss(start_y, wiggle_width), speed=300)
                if self._wiggle_event is not None and not self._wiggle_event.is_set():
                    break
                time.sleep(0.01)

        self._wiggle_thread = threading.Thread(target=_run, daemon=True)
        self._wiggle_thread.start()

    def stop_wiggle(self) -> None:
        """Stop the background winder wiggle thread."""
        if not self._wiggle_event:
            return
        set_speed()
        self._wiggle_event.clear()
        if self._wiggle_thread:
            self._wiggle_thread.join(timeout=0.1)
        self._wiggle_event = None
        self._wiggle_thread = None
        reset_plc()

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

        return self.goto_collect_wire_data(
            wire_number=wire_number,
            wire_x=x,
            wire_y=y,
        )

    def measure_auto(self) -> None:
        from dune_tension.summaries import get_missing_wires

        wires_dict = get_missing_wires(self.config)
        wires_to_measure = wires_dict.get(self.config.side, [])

        if not wires_to_measure:
            print("All wires are already measured.")
            return

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
                self.goto_collect_wire_data(wire_number=wire_number, wire_x=x, wire_y=y)
                measured_count += 1
                elapsed = time.time() - start_time
                avg_time = elapsed / measured_count
                remaining = len(wires_to_measure) - measured_count
                est_remaining = avg_time * remaining
                print(
                    f"Estimated time remaining: {timedelta(seconds=int(est_remaining))}"
                )
        print("Done measuring all wires")

    def measure_list(
        self, wire_list: list[int], preserve_order: bool, profile: bool = False
    ) -> None:
        measure_list(
            config=self.config,
            wire_list=wire_list,
            get_xy_from_file_func=get_xy_from_file,
            get_current_xy_func=self.get_current_xy_position,
            collect_func=lambda w, x, y: self.goto_collect_wire_data(
                wire_number=w,
                wire_x=x,
                wire_y=y,
            ),
            stop_event=self.stop_event,
            preserve_order=preserve_order,
            profile=profile,
        )

    def _collect_samples(
        self,
        wire_number: int,
        length: float,
        start_time: float,
        wire_y: float,
        wire_x: float,
    ) -> list[TensionResult]:
        expected_frequency = wire_equation(length=length)["frequency"]
        measuring_timeout = self.config.measuring_duration
        passing_wires = []
        audio_acquisition_config = PitchCompareConfig(
            sample_rate=self.samplerate,
            max_record_seconds=self.config.record_duration,
            expected_f0=expected_frequency,
            snr_threshold_db=self.snr,
            trigger_mode="snr",
        )
        while (time.time() - start_time) < measuring_timeout:
            if check_stop_event(self.stop_event, "tension measurement interrupted!"):
                return None, wire_y
            x, y = self.get_current_xy_position()

            # trigger a valve pulse using pulse from valve_trigger.py
            self.strum_func()
            # record audio with harmonic comb

            audio_sample = acquire_audio(
                cfg=audio_acquisition_config, noise_rms=0.05, timeout=3
            )

            if audio_sample is not None:
                # estimate pitch from audio sample
                frequency, confidence = estimate_pitch_from_audio(
                    audio_sample,
                    self.samplerate,
                    expected_frequency,
                )
                print(
                    f"sample of wire {wire_number}: Measured frequency {frequency:.2f} Hz with confidence {confidence:.2f}"
                )
                wire_result = TensionResult(
                    apa_name=self.config.apa_name,
                    layer=self.config.layer,
                    side=self.config.side,
                    taped=self._is_current_side_taped(),
                    wire_number=wire_number,
                    frequency=frequency,
                    confidence=confidence,
                    x=x,
                    y=y,
                    time=datetime.now(),
                )

                # if the confidence is good, log the sample
                if (
                    wire_result.confidence >= self.config.confidence_threshold
                    and tension_plausible(wire_result.tension)
                ):
                    passing_wires.append(wire_result)
                else:
                    print("wiggling due to low confidence or implausible tension.")
                    self.wiggle_func(0, gauss(0, 1))
                    # self.focus_wiggle_func(choice([-100,100]))
                if len(passing_wires) >= self.config.samples_per_wire:
                    break
            else:
                print(f"sample of wire {wire_number}: No audio detected.")

        return passing_wires

    def _merge_results(
        self,
        passing_wires: list[TensionResult],
        wire_number: int,
        wire_x: float,
        wire_y: float,
    ) -> TensionResult:
        if passing_wires == []:
            return None
        if self.config.samples_per_wire == 1:
            return passing_wires[0]
        elif len(passing_wires) > 0:
            frequency = np.average([d.frequency for d in passing_wires])
            confidence = np.sum([d.confidence for d in passing_wires])
            x = round(np.average([d.x for d in passing_wires]), 1)
            y = round(np.average([d.y for d in passing_wires]), 1)
        else:
            frequency = 0.0
            confidence = 0.0
            x = wire_x
            y = wire_y

        return TensionResult(
            apa_name=self.config.apa_name,
            layer=self.config.layer,
            side=self.config.side,
            taped=self._is_current_side_taped(),
            wire_number=wire_number,
            frequency=frequency,
            confidence=confidence,
            x=x,
            y=y,
            time=datetime.now(),
        )

    def goto_collect_wire_data(
        self, wire_number: int, wire_x: float, wire_y: float
    ) -> Optional[TensionResult]:
        reset_plc()
        length = length_lookup(
            self.config.layer,
            wire_number,
            zone_lookup(wire_x),
            taped=self._is_current_side_taped(),
        )
        assert length != np.float64("nan"), "Length lookup returned NaN"
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
                taped=self._is_current_side_taped(),
                wire_number=wire_number,
                frequency=0.0,
                confidence=0.0,
                x=wire_x,
                y=wire_y,
                time=datetime.now(),
            )

        try:
            wires_results = self._collect_samples(
                wire_number=wire_number,
                length=length,
                start_time=start_time,
                wire_y=wire_y,
                wire_x=wire_x,
            )

        finally:
            reset_plc()

        if wires_results is None:
            return

        result = self._merge_results(wires_results, wire_number, wire_x, wire_y)

        if result is None:
            print(f"measurement failed for wire number {wire_number}.")
            return result
        if not result.tension_pass:
            print(f"Tension failed for wire number {wire_number}.")
        ttf = time.time() - start_time
        print(
            f"result: Wire number {wire_number} has length {length * 1000:.1f}mm tension {result.tension:.1f}N frequency {result.frequency:.1f}Hz with confidence {result.confidence:.2f}.\n at {result.x},{result.y}\n"
            f"Took {ttf} seconds to finish."
        )
        result.ttf = ttf
        result.time = datetime.now()
        reset_plc()

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
