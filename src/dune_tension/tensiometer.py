import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import math
from typing import Any, Optional, Callable
import time
import numpy as np
import pandas as pd
from random import gauss

try:
    from dune_tension.config import MEASUREMENT_WIGGLE_CONFIG
except ImportError:  # pragma: no cover - fallback for legacy test stubs
    from config import MEASUREMENT_WIGGLE_CONFIG

try:
    from dune_tension.geometry import zone_lookup, length_lookup
    from dune_tension.tension_calculation import wire_equation, tension_plausible
    from dune_tension.tensiometer_functions import (
        make_config,
        measure_list,
        plan_measurement_triplets,
        get_xy_from_file,
        check_stop_event,
    )
    from dune_tension.results import TensionResult
    from dune_tension.services import AudioCaptureService, MotionService, ResultRepository
except ImportError:  # pragma: no cover - fallback for legacy test stubs
    from geometry import zone_lookup, length_lookup
    try:
        from tension_calculation import wire_equation, tension_plausible
    except ImportError:
        from tension_calculation import tension_lookup, tension_plausible

        def wire_equation(*, length: float, frequency: float | None = None):
            active_frequency = 1.0 if frequency is None else float(frequency)
            return {
                "frequency": active_frequency,
                "tension": tension_lookup(length, active_frequency),
            }

    from tensiometer_functions import (
        make_config,
        measure_list,
        plan_measurement_triplets,
        get_xy_from_file,
        check_stop_event,
    )
    from results import TensionResult
    from dune_tension.services import AudioCaptureService, MotionService, ResultRepository

LOGGER = logging.getLogger(__name__)
FOCUS_MM_PER_QUARTER_US = 20.0 / 4000.0
FOCUS_X_MM_PER_QUARTER_US = FOCUS_MM_PER_QUARTER_US / math.sqrt(3.0)


@dataclass(frozen=True)
class AudioAcquisitionConfig:
    """Minimal runtime config passed into ``acquire_audio``."""

    sample_rate: int
    max_record_seconds: float
    expected_f0: float | None
    snr_threshold_db: float
    trigger_mode: str
    min_frequency: float = 30.0
    max_frequency: float = 2000.0
    min_oscillations_per_window: float = 10.0
    min_window_overlap: float = 0.5
    idle_timeout: float = 0.2
    input_mode: str = "mic"
    input_audio_path: str | None = None


def acquire_audio(*args, **kwargs):
    """Lazily import the runtime audio acquisition helper."""

    from spectrum_analysis.audio_processing import acquire_audio as _acquire_audio

    return _acquire_audio(*args, **kwargs)


def estimate_pitch_from_audio(*args, **kwargs):
    """Lazily import the runtime PESTO pitch estimator."""

    from spectrum_analysis.pesto_analysis import (
        estimate_pitch_from_audio as _estimate_pitch_from_audio,
    )

    return _estimate_pitch_from_audio(*args, **kwargs)


def analyze_audio_with_pesto(*args, **kwargs):
    """Lazily import the runtime PESTO diagnostics helper."""

    from spectrum_analysis.pesto_analysis import (
        analyze_audio_with_pesto as _analyze_audio_with_pesto,
    )

    return _analyze_audio_with_pesto(*args, **kwargs)


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
        wiggle_y_sigma_mm: float = MEASUREMENT_WIGGLE_CONFIG.y_sigma_mm,
        focus_wiggle_sigma_quarter_us: float = (
            MEASUREMENT_WIGGLE_CONFIG.focus_sigma_quarter_us
        ),
        strum: Optional[Callable[[], None]] = None,
        focus_wiggle: Optional[Callable[[float], None]] = None,
        focus_position_getter: Optional[Callable[[], int]] = None,
        estimated_time_callback: Optional[Callable[[str], None]] = None,
        audio_sample_callback: Optional[Callable[[Any, int, Any | None], None]] = None,
        summary_refresh_callback: Optional[Callable[[Any], None]] = None,
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
        self.wiggle_y_sigma_mm = float(wiggle_y_sigma_mm)
        self.focus_wiggle_sigma_quarter_us = float(focus_wiggle_sigma_quarter_us)
        if self.wiggle_y_sigma_mm < 0:
            raise ValueError("wiggle_y_sigma_mm must be non-negative")
        if self.focus_wiggle_sigma_quarter_us < 0:
            raise ValueError("focus_wiggle_sigma_quarter_us must be non-negative")
        self.motion = MotionService.build(spoof_movement=spoof_movement)
        self.audio = AudioCaptureService.build(spoof=spoof)
        self.repository = ResultRepository(self.config.data_path)
        self.noise_threshold = self.audio.noise_threshold
        self.samplerate = self.audio.samplerate
        self.record_audio_func = self.audio.record_audio

        self.get_current_xy_position = self.motion.get_xy
        self.goto_xy_func = self.motion.goto_xy
        self.wiggle_func = self.motion.increment

        self._has_focus_wiggle_callback = focus_wiggle is not None
        self.focus_wiggle_func = focus_wiggle or (lambda _delta: None)
        self.focus_position_getter = focus_position_getter or (lambda: 0)
        self.strum_func = strum or (lambda: None)
        self.estimated_time_callback = estimated_time_callback or (lambda _value: None)
        self.audio_sample_callback = (
            audio_sample_callback or (lambda _sample, _samplerate, _analysis: None)
        )
        self.summary_refresh_callback = summary_refresh_callback or (lambda _config: None)

        self.a_taped = bool(a_taped)
        self.b_taped = bool(b_taped)

        # State tracking for winder wiggle thread
        self._wiggle_event: threading.Event | None = None
        self._wiggle_thread: threading.Thread | None = None

    def _focus_wiggle_x_sign(self) -> float:
        """Return focus/X coupling sign: A is negative, B is positive."""

        return -1.0 if str(self.config.side).upper() == "A" else 1.0

    def _focus_to_x_delta_mm(self, delta_focus_units: float) -> float:
        """Convert a focus delta in quarter-us to the coupled X delta in mm."""

        return (
            self._focus_wiggle_x_sign()
            * float(delta_focus_units)
            * FOCUS_X_MM_PER_QUARTER_US
        )

    def _get_focus_position(self) -> int:
        """Return the latest commanded focus position in quarter-us units."""

        try:
            return int(self.focus_position_getter())
        except Exception:
            return 0

    def _apply_focus_wiggle_with_x_compensation(self, delta_focus: float) -> float | None:
        """Apply focus wiggle and X compensation for equivalent travel in mm."""

        if not self._has_focus_wiggle_callback:
            return None

        commanded_delta = int(float(delta_focus))
        self.focus_wiggle_func(commanded_delta)
        if commanded_delta == 0:
            return None

        delta_x_mm = self._focus_to_x_delta_mm(commanded_delta)
        try:
            cur_x, cur_y = self.get_current_xy_position()
        except Exception as exc:
            LOGGER.warning("Unable to read XY for focus wiggle compensation: %s", exc)
            return None

        new_x = round(cur_x + delta_x_mm, 1)
        try:
            moved = self.goto_xy_func(new_x, cur_y)
        except Exception as exc:
            LOGGER.warning("Focus wiggle compensation move failed: %s", exc)
            return None
        if moved is False:
            LOGGER.warning(
                "Focus wiggle compensation move to %s,%s failed.",
                new_x,
                cur_y,
            )
            return None

        try:
            compensated_x, _ = self.get_current_xy_position()
            return float(compensated_x)
        except Exception:
            return new_x

    @staticmethod
    def _sample_sort_key(result: TensionResult) -> tuple[float, datetime]:
        timestamp = getattr(result, "time", None)
        return (
            float(result.confidence),
            timestamp if timestamp is not None else datetime.min,
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
        wiggle_width = MEASUREMENT_WIGGLE_CONFIG.background_y_sigma_mm

        def _run() -> None:
            while self._wiggle_event and self._wiggle_event.is_set():
                self.goto_xy_func(
                    start_x,
                    gauss(start_y, wiggle_width),
                    speed=MEASUREMENT_WIGGLE_CONFIG.background_speed,
                )
                if self._wiggle_event is not None and not self._wiggle_event.is_set():
                    break
                time.sleep(MEASUREMENT_WIGGLE_CONFIG.background_interval_seconds)

        self._wiggle_thread = threading.Thread(target=_run, daemon=True)
        self._wiggle_thread.start()

    def stop_wiggle(self) -> None:
        """Stop the background winder wiggle thread."""
        if not self._wiggle_event:
            return
        self.motion.set_speed()
        self._wiggle_event.clear()
        if self._wiggle_thread:
            self._wiggle_thread.join(timeout=0.1)
        self._wiggle_event = None
        self._wiggle_thread = None
        self.motion.reset_plc()

    def _plot_audio(self, audio_sample) -> None:
        """Save a plot of the recorded audio sample to a temporary file."""
        try:
            import matplotlib.pyplot as plt  # Local import to avoid optional dep
        except Exception as exc:  # pragma: no cover - plotting is optional
            LOGGER.warning("Failed to import matplotlib for plotting: %s", exc)
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
                LOGGER.info("Audio plot saved to %s", tmp.name)
            plt.close()
        except Exception as exc:  # pragma: no cover - plotting is optional
            LOGGER.warning("Failed to plot audio sample: %s", exc)

    def measure_calibrate(self, wire_number: int) -> Optional[TensionResult]:
        xy = self.get_current_xy_position()
        if xy is None:
            LOGGER.warning(
                "No position data found for wire %s. Using current position.",
                wire_number,
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
            self.estimated_time_callback("0:00:00")
            LOGGER.info("All wires are already measured.")
            return

        LOGGER.info("Measuring missing wires...")
        LOGGER.info("Missing wires: %s", wires_to_measure)
        targets = plan_measurement_triplets(
            config=self.config,
            wire_list=wires_to_measure,
            get_xy_from_file_func=get_xy_from_file,
            get_current_xy_func=self.get_current_xy_position,
            preserve_order=False,
        )
        if not targets:
            self.estimated_time_callback("0:00:00")
            LOGGER.info("No measurable wires remain after planning motion targets.")
            return

        start_time = time.time()
        measured_count = 0
        did_report_zero = False
        for wire_number, x, y in targets:
            if check_stop_event(self.stop_event):
                return

            LOGGER.info(
                "Measuring wire %s at position %s,%s",
                wire_number,
                x,
                y,
            )
            self.goto_collect_wire_data(wire_number=wire_number, wire_x=x, wire_y=y)
            measured_count += 1
            remaining = len(targets) - measured_count
            if remaining > 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / measured_count
                est_remaining = avg_time * remaining
                eta_text = str(timedelta(seconds=int(est_remaining)))
                self.estimated_time_callback(eta_text)
                did_report_zero = eta_text == "0:00:00"
        if not did_report_zero:
            self.estimated_time_callback("0:00:00")
        LOGGER.info("Done measuring all wires")

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
    ) -> list[TensionResult] | None:
        expected_frequency = wire_equation(length=length)["frequency"]
        measuring_timeout = self.config.measuring_duration
        candidate_wires: list[TensionResult] = []
        audio_acquisition_config = AudioAcquisitionConfig(
            sample_rate=self.samplerate,
            max_record_seconds=self.config.record_duration,
            expected_f0=expected_frequency,
            snr_threshold_db=self.snr,
            trigger_mode="snr",
        )

        x_step_mm = max(
            0.1,
            min(
                length * MEASUREMENT_WIGGLE_CONFIG.xy_sigma_per_meter,
                MEASUREMENT_WIGGLE_CONFIG.xy_sigma_cap_mm,
            ),
        )
        y_step_mm = max(0.05, float(self.wiggle_y_sigma_mm))
        focus_step_quarter_us = max(10, int(abs(self.focus_wiggle_sigma_quarter_us)))

        min_x_step_mm = 0.1
        min_y_step_mm = 0.05
        min_focus_step_quarter_us = 5

        best_confidence = -1.0
        best_x = float(wire_x)
        best_y = float(wire_y)
        best_focus = self._get_focus_position()
        axis_index = 0

        def _move_to_pose(x_target: float, y_target: float, focus_target: int) -> None:
            current_focus = self._get_focus_position()
            delta_focus = int(focus_target - current_focus)
            if delta_focus != 0:
                self._apply_focus_wiggle_with_x_compensation(delta_focus)
            self.goto_xy_func(x_target, y_target)

        def _next_pose() -> tuple[float, float, int]:
            nonlocal axis_index, x_step_mm, y_step_mm, focus_step_quarter_us

            candidates: list[tuple[float, float, int]] = [
                (best_x + x_step_mm, best_y, best_focus),
                (best_x - x_step_mm, best_y, best_focus),
                (best_x, best_y + y_step_mm, best_focus),
                (best_x, best_y - y_step_mm, best_focus),
            ]

            if self._has_focus_wiggle_callback and focus_step_quarter_us > 0:
                for delta_focus in (focus_step_quarter_us, -focus_step_quarter_us):
                    candidates.append(
                        (
                            best_x + self._focus_to_x_delta_mm(delta_focus),
                            best_y,
                            best_focus + delta_focus,
                        )
                    )

            if axis_index >= len(candidates):
                axis_index = 0
                x_step_mm = max(min_x_step_mm, x_step_mm * 0.7)
                y_step_mm = max(min_y_step_mm, y_step_mm * 0.7)
                if self._has_focus_wiggle_callback:
                    focus_step_quarter_us = max(
                        min_focus_step_quarter_us,
                        int(focus_step_quarter_us * 0.7),
                    )

            return candidates[axis_index]

        while (time.time() - start_time) < measuring_timeout:
            if check_stop_event(self.stop_event, "tension measurement interrupted!"):
                return None
            x, y = self.get_current_xy_position()

            # trigger a valve pulse using pulse from valve_trigger.py
            self.strum_func()
            # record audio with harmonic comb

            audio_sample = acquire_audio(
                cfg=audio_acquisition_config,
                noise_rms=self.noise_threshold / 3,
                timeout=0.1,
            )

            if audio_sample is not None:
                analysis = None
                try:
                    analysis = analyze_audio_with_pesto(
                        audio_sample,
                        self.samplerate,
                        expected_frequency=expected_frequency,
                        include_activations=True,
                    )
                    frequency, confidence = analysis.frequency, analysis.confidence
                except Exception:
                    frequency, confidence = estimate_pitch_from_audio(
                        audio_sample,
                        self.samplerate,
                        expected_frequency,
                    )
                try:
                    self.audio_sample_callback(audio_sample, self.samplerate, analysis)
                except Exception as exc:
                    LOGGER.debug("Audio sample callback failed: %s", exc)

                LOGGER.info(
                    "Sample of wire %s: measured frequency %.2f Hz %s with confidence %.2f",
                    wire_number,
                    frequency,
                    wire_equation(length=length, frequency=frequency),
                    confidence,
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
                    focus_position=self._get_focus_position(),
                    time=datetime.now(),
                )
                self.repository.append_sample(wire_result)

                if tension_plausible(wire_result.tension):
                    candidate_wires.append(wire_result)
                    if wire_result.confidence > best_confidence:
                        best_confidence = wire_result.confidence
                        best_x = wire_result.x
                        best_y = wire_result.y
                        best_focus = (
                            wire_result.focus_position
                            if wire_result.focus_position is not None
                            else best_focus
                        )
                        axis_index = 0
                    if wire_result.confidence >= self.config.confidence_threshold:
                        break

            else:
                LOGGER.info("Sample of wire %s: no audio detected.", wire_number)
            if (time.time() - start_time) >= measuring_timeout:
                break

            target_x, target_y, target_focus = _next_pose()
            axis_index += 1
            LOGGER.info(
                "Optimizer next pose: x=%s y=%s focus=%s",
                target_x,
                target_y,
                target_focus,
            )
            _move_to_pose(target_x, target_y, target_focus)
        return candidate_wires

    def _merge_results(
        self,
        passing_wires: list[TensionResult],
        wire_number: int,
        wire_x: float,
        wire_y: float,
    ) -> TensionResult:
        if passing_wires == []:
            return None
        return max(passing_wires, key=self._sample_sort_key)

    def goto_collect_wire_data(
        self, wire_number: int, wire_x: float, wire_y: float
    ) -> Optional[TensionResult]:
        self.motion.reset_plc()
        length = length_lookup(
            self.config.layer,
            wire_number,
            zone_lookup(wire_x),
            taped=self._is_current_side_taped(),
        )
        if np.isnan(length):
            raise ValueError("Length lookup returned NaN")

        if check_stop_event(self.stop_event):
            return

        succeed = self.goto_xy_func(wire_x, wire_y)
        if check_stop_event(self.stop_event):
            return
        if not succeed:
            LOGGER.warning(
                "Failed to move to wire %s position %s,%s.",
                wire_number,
                wire_x,
                wire_y,
            )
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
                focus_position=self._get_focus_position(),
                time=datetime.now(),
            )
        start_time = time.time()
        try:
            wires_results = self._collect_samples(
                wire_number=wire_number,
                length=length,
                start_time=start_time,
                wire_y=wire_y,
                wire_x=wire_x,
            )

        finally:
            self.motion.reset_plc()

        if wires_results is None:
            return

        result = self._merge_results(wires_results, wire_number, wire_x, wire_y)

        if result is None:
            LOGGER.warning("Measurement failed for wire number %s.", wire_number)
            return result
        if not result.tension_pass:
            LOGGER.warning("Tension failed for wire number %s.", wire_number)
        ttf = time.time() - start_time
        LOGGER.info(
            "Result: wire %s length %.1f mm tension %.1f N frequency %.1f Hz confidence %.2f at %s,%s focus %s. Took %.2f seconds.",
            wire_number,
            length * 1000,
            result.tension,
            result.frequency,
            result.confidence,
            result.x,
            result.y,
            result.focus_position,
            ttf,
        )
        result.ttf = ttf
        result.time = datetime.now()
        self.motion.reset_plc()
        self.repository.append_result(result)
        try:
            self.summary_refresh_callback(self.config)
        except Exception as exc:
            LOGGER.debug("Summary refresh callback failed: %s", exc)

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
