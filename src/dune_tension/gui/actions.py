"""Tkinter command callbacks used by the tensiometer GUI."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from datetime import datetime
import logging
import os
import re
import threading
from threading import Thread
from typing import TYPE_CHECKING, Any

from tkinter import messagebox

from dune_tension.data_cache import (
    clear_wire_numbers,
    clear_wire_range,
    find_outliers,
    get_dataframe,
    update_dataframe,
)
from dune_tension.results import EXPECTED_COLUMNS
from dune_tension.tensiometer_functions import make_config
from dune_tension.gui.context import GUIContext
from dune_tension.gui.state import save_state

if TYPE_CHECKING:
    from dune_tension.tensiometer import Tensiometer

LOGGER = logging.getLogger(__name__)


def _set_estimated_time(ctx: GUIContext, value: str) -> None:
    """Update the ETA label on the Tk thread."""

    def apply() -> None:
        ctx.estimated_time_var.set(value)

    try:
        if threading.current_thread() is threading.main_thread():
            apply()
            return
        ctx.root.after(0, apply)
    except Exception:
        return


def _publish_live_waveform(
    ctx: GUIContext,
    audio_sample: Any,
    samplerate: int,
    analysis: Any | None,
) -> None:
    manager = getattr(ctx, "live_plot_manager", None)
    if manager is None:
        return
    manager.publish_waveform(audio_sample, samplerate, analysis)


def _request_live_summary_refresh(ctx: GUIContext, config: Any) -> None:
    manager = getattr(ctx, "live_plot_manager", None)
    if manager is None:
        return
    manager.request_summary_refresh(config)


@dataclass(frozen=True)
class WorkerInputs:
    apa_name: str
    layer: str
    side: str
    flipped: bool
    a_taped: bool
    b_taped: bool
    samples: int
    confidence: float
    record_duration: float
    measuring_duration: float
    plot_audio: bool
    wire_number: str
    wire_list: str
    condition: str
    times_sigma: str
    set_tension: str
    clear_range: str
    xy_text: str


def _show_input_error(ctx: GUIContext, message: str) -> None:
    messagebox.showerror("Input Error", message)


def _capture_worker_inputs(ctx: GUIContext) -> WorkerInputs:
    """Capture all widget values on the Tk thread before background work."""

    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("_capture_worker_inputs must run on the main Tk thread.")

    w = ctx.widgets
    try:
        samples = int(w.entry_samples.get())
        if samples < 1:
            raise ValueError("Samples per wire must be \u2265 1")
    except ValueError as exc:
        _show_input_error(ctx, str(exc))
        raise

    try:
        confidence = float(w.entry_confidence.get())
    except ValueError as exc:
        _show_input_error(ctx, str(exc))
        raise

    try:
        record_duration = float(w.entry_record_duration.get())
        measuring_duration = float(w.entry_measuring_duration.get())
    except ValueError as exc:
        _show_input_error(ctx, str(exc))
        raise

    return WorkerInputs(
        apa_name=w.entry_apa.get(),
        layer=w.layer_var.get(),
        side=w.side_var.get(),
        flipped=bool(w.flipped_var.get()),
        a_taped=bool(w.a_taped_var.get()),
        b_taped=bool(w.b_taped_var.get()),
        samples=samples,
        confidence=confidence,
        record_duration=record_duration,
        measuring_duration=measuring_duration,
        plot_audio=bool(w.plot_audio_var.get()),
        wire_number=w.entry_wire.get(),
        wire_list=w.entry_wire_list.get(),
        condition=w.entry_condition.get(),
        times_sigma=w.entry_times_sigma.get(),
        set_tension=w.entry_set_tension.get(),
        clear_range=w.entry_clear_range.get(),
        xy_text=w.entry_xy.get(),
    )


def create_tensiometer(ctx: GUIContext, inputs: WorkerInputs) -> "Tensiometer":
    """Instantiate a :class:`Tensiometer` from captured UI inputs."""

    from dune_tension.tensiometer import Tensiometer

    try:
        samples = int(inputs.samples)
        confidence = float(inputs.confidence)
        record_duration = float(inputs.record_duration)
        measuring_duration = float(inputs.measuring_duration)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid measurement inputs: {exc}") from exc

    spoof_audio = bool(os.environ.get("SPOOF_AUDIO"))
    return Tensiometer(
        apa_name=inputs.apa_name,
        layer=inputs.layer,
        side=inputs.side,
        flipped=inputs.flipped,
        a_taped=inputs.a_taped,
        b_taped=inputs.b_taped,
        spoof=spoof_audio,
        spoof_movement=bool(os.environ.get("SPOOF_PLC")),
        stop_event=ctx.stop_event,
        samples_per_wire=samples,
        confidence_threshold=confidence,
        record_duration=record_duration,
        measuring_duration=measuring_duration,
        plot_audio=inputs.plot_audio,
        strum=ctx.strum,
        focus_wiggle=ctx.servo_controller.nudge_focus,
        estimated_time_callback=lambda value: _set_estimated_time(ctx, value),
        audio_sample_callback=lambda audio_sample, samplerate, analysis: _publish_live_waveform(
            ctx,
            audio_sample,
            samplerate,
            analysis,
        ),
        summary_refresh_callback=lambda config: _request_live_summary_refresh(
            ctx,
            config,
        ),
    )


def _begin_measurement(ctx: GUIContext, name: str) -> bool:
    """Mark a measurement as active, returning ``False`` if one is already running."""

    with ctx.measurement_lock:
        if ctx.measurement_active:
            active_name = ctx.active_measurement_name or "another measurement"
            LOGGER.warning(
                "Ignoring %s request because %s is already running.",
                name,
                active_name,
            )
            return False
        ctx.measurement_active = True
        ctx.active_measurement_name = name
        return True


def _end_measurement(ctx: GUIContext) -> None:
    """Clear the active measurement marker."""

    with ctx.measurement_lock:
        ctx.measurement_active = False
        ctx.active_measurement_name = ""


def _run_in_thread(func=None, *, measurement: bool = False):
    """Decorator to execute ``func`` in a daemon thread."""

    def decorator(func):
        def wrapper(ctx: GUIContext, *args: Any, **kwargs: Any) -> None:
            try:
                inputs = _capture_worker_inputs(ctx)
            except ValueError:
                return

            save_state(ctx)

            measurement_name = func.__name__.replace("_", " ")
            if measurement and not _begin_measurement(ctx, measurement_name):
                return

            def run() -> None:
                ctx.stop_event.clear()
                try:
                    func(ctx, inputs, *args, **kwargs)
                finally:
                    ctx.stop_event.clear()
                    if measurement:
                        _end_measurement(ctx)

            try:
                Thread(target=run, daemon=True).start()
            except Exception:
                if measurement:
                    _end_measurement(ctx)
                raise

        return wrapper

    if func is None:
        return decorator
    return decorator(func)


@_run_in_thread(measurement=True)
def measure_calibrate(ctx: GUIContext, inputs: WorkerInputs) -> None:
    """Measure and calibrate a single wire."""

    tensiometer: Tensiometer | None = None
    try:
        tensiometer = create_tensiometer(ctx, inputs)
        wire_number = int(inputs.wire_number)
        tensiometer.measure_calibrate(wire_number)
        LOGGER.info("Done calibrating wire %s", wire_number)
    finally:
        _cleanup_after_measurement(ctx, tensiometer)


@_run_in_thread(measurement=True)
def measure_auto(ctx: GUIContext, inputs: WorkerInputs) -> None:
    """Automatically measure the full APA."""
    LOGGER.info("Starting automatic measurement of full APA")
    tensiometer: Tensiometer | None = None
    try:
        _set_estimated_time(ctx, "--")
        tensiometer = create_tensiometer(ctx, inputs)
        tensiometer.measure_auto()
        if ctx.stop_event.is_set():
            _set_estimated_time(ctx, "Interrupted")
    except Exception:
        _set_estimated_time(ctx, "Not running")
        raise
    finally:
        _cleanup_after_measurement(ctx, tensiometer, reset_estimated_time=False)


def _parse_ranges(text: str) -> list[tuple[int, int]]:
    """Return list of ``(start, end)`` tuples parsed from ``text``."""

    ranges: list[tuple[int, int]] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            try:
                start_str, end_str = part.split("-", 1)
                start = int(start_str)
                end = int(end_str)
            except ValueError:
                continue
        else:
            try:
                start = end = int(part)
            except ValueError:
                continue
        if start > end:
            start, end = end, start
        ranges.append((start, end))
    return ranges


@_run_in_thread(measurement=True)
def measure_list_button(ctx: GUIContext, inputs: WorkerInputs) -> None:
    """Measure a comma separated list of wire ranges."""

    tensiometer: Tensiometer | None = None
    try:
        tensiometer = create_tensiometer(ctx, inputs)
        ranges = _parse_ranges(inputs.wire_list)
        wire_list: list[int] = []
        for start, end in ranges:
            wire_list.extend(range(start, end + 1))
        LOGGER.info("Measuring wires: %s", wire_list)
        tensiometer.measure_list(wire_list, preserve_order=False)
    finally:
        _cleanup_after_measurement(ctx, tensiometer)


@_run_in_thread
def calibrate_background_noise(ctx: GUIContext, _inputs: WorkerInputs) -> None:
    """Record background noise for filtering future recordings."""

    try:
        from dune_tension.audioProcessing import (
            calibrate_background_noise,
            get_samplerate,
        )

        samplerate = get_samplerate()
        if samplerate is None:
            LOGGER.warning("Unable to access audio device")
            return
        calibrate_background_noise(int(samplerate))
        LOGGER.info("Background noise calibrated")
    finally:
        ctx.stop_event.clear()


@_run_in_thread(measurement=True)
def measure_condition(ctx: GUIContext, inputs: WorkerInputs) -> None:
    """Measure wires whose tension satisfies the configured expression."""

    def _get_wires(config, expr: str) -> list[int]:
        from dune_tension.summaries import get_tension_series

        try:
            predicate = _compile_tension_condition(expr)
        except ValueError as exc:
            LOGGER.warning("Invalid expression %r: %s", expr, exc)
            return []

        wires: list[int] = []
        tension_series = get_tension_series(config)
        for wire_number, tension in sorted(
            tension_series.get(str(config.side).upper(), {}).items()
        ):
            try:
                if predicate(float(tension)):
                    wires.append(int(wire_number))
            except Exception as exc:
                LOGGER.warning("Invalid expression %r: %s", expr, exc)
                return []
        return wires

    tensiometer: Tensiometer | None = None
    expr = inputs.condition.strip()
    if not expr:
        LOGGER.warning("No condition specified")
        return

    try:
        config = _make_config_from_inputs(inputs)
        wires = _get_wires(config, expr)
        if not wires:
            LOGGER.info("No wires satisfy: %s", expr)
            return
        tensiometer = create_tensiometer(ctx, inputs)
        LOGGER.info("Measuring wires %s matching %r", wires, expr)
        tensiometer.measure_list(wires, preserve_order=False)
    finally:
        _cleanup_after_measurement(ctx, tensiometer)


def _parse_pairs(text: str) -> list[tuple[int, float]]:
    pair_re = re.compile(r"\(?\s*(\d+)\s*[,:]\s*([+-]?\d+(?:\.\d*)?)\s*\)?")
    pairs: list[tuple[int, float]] = []
    for match in pair_re.finditer(text):
        try:
            wire = int(match.group(1))
            tension = float(match.group(2))
        except ValueError:
            continue
        pairs.append((wire, tension))
    return pairs


def _compile_tension_condition(expr: str):
    """Compile a safe condition expression using only variable ``t``."""

    allowed_nodes = (
        ast.Expression,
        ast.BoolOp,
        ast.BinOp,
        ast.UnaryOp,
        ast.Compare,
        ast.Name,
        ast.Load,
        ast.Constant,
        ast.And,
        ast.Or,
        ast.Not,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.Mod,
        ast.USub,
        ast.UAdd,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
    )

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"invalid syntax: {exc.msg}") from exc

    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            raise ValueError(f"disallowed expression node: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id != "t":
            raise ValueError("only variable 't' is allowed")

    code = compile(tree, "<tension-condition>", "eval")

    def predicate(tension: float) -> bool:
        result = eval(code, {"__builtins__": {}}, {"t": float(tension)})
        return bool(result)

    return predicate


def clear_range(ctx: GUIContext) -> None:
    ranges = _parse_ranges(ctx.widgets.entry_clear_range.get())
    if not ranges:
        LOGGER.warning("No valid range specified")
        return

    cfg = _make_config_from_widgets(ctx)
    for start, end in ranges:
        clear_wire_range(cfg.data_path, cfg.apa_name, cfg.layer, cfg.side, start, end)
    LOGGER.info("Cleared ranges: %s", ctx.widgets.entry_clear_range.get())
    _request_live_summary_refresh(ctx, cfg)


def erase_outliers(ctx: GUIContext) -> None:
    cfg = _make_config_from_widgets(ctx)
    try:
        conf = float(ctx.widgets.entry_confidence.get())
    except ValueError:
        conf = 0.7
    try:
        times_sigma = float(ctx.widgets.entry_times_sigma.get())
    except ValueError:
        times_sigma = 2.0

    outliers = sorted(
        find_outliers(
            cfg.data_path,
            cfg.apa_name,
            cfg.layer,
            cfg.side,
            times_sigma=times_sigma,
            confidence_threshold=conf,
        )
    )

    if outliers:
        clear_wire_numbers(
            cfg.data_path,
            cfg.apa_name,
            cfg.layer,
            cfg.side,
            outliers,
        )
        LOGGER.info("Erased outlier wires: %s", outliers)
        _request_live_summary_refresh(ctx, cfg)
    else:
        LOGGER.info("No outlier wires found")


def set_manual_tension(ctx: GUIContext) -> None:
    pairs = _parse_pairs(ctx.widgets.entry_set_tension.get())
    if not pairs:
        LOGGER.warning("No valid tension pairs specified")
        return

    cfg = _make_config_from_widgets(ctx)
    df = get_dataframe(cfg.data_path)
    for wire, tension in pairs:
        mask = (
            (df["apa_name"] == cfg.apa_name)
            & (df["layer"] == cfg.layer)
            & (df["side"] == cfg.side)
            & (df["wire_number"].astype(int) == wire)
        )
        if mask.any():
            df.loc[mask, "tension"] = tension
            if "time" in df.columns:
                df.loc[mask, "time"] = datetime.now().isoformat()
        else:
            row = {col: "" for col in EXPECTED_COLUMNS}
            row.update(
                {
                    "apa_name": cfg.apa_name,
                    "layer": cfg.layer,
                    "side": cfg.side,
                    "wire_number": wire,
                    "frequency": 0.0,
                    "confidence": 1,
                    "tension": tension,
                    "tension_pass": 1,
                    "x": 0.0,
                    "y": 0.0,
                    "time": datetime.now().isoformat(),
                    "zone": 1,
                    "wire_length": 0.0,
                }
            )
            df.loc[len(df)] = row
    update_dataframe(cfg.data_path, df)
    LOGGER.info("Updated tensions: %s", pairs)
    _request_live_summary_refresh(ctx, cfg)


def interrupt(ctx: GUIContext) -> None:
    ctx.stop_event.set()
    ctx.servo_controller.stop_loop()
    _set_estimated_time(ctx, "Interrupted")


def monitor_tension_logs(ctx: GUIContext) -> None:
    """Check for updates to the tension data file and refresh logs."""

    cfg = _make_config_from_widgets(ctx)

    try:
        mtime = os.path.getmtime(cfg.data_path)
    except OSError:
        mtime = None

    if ctx.monitor_last_path != cfg.data_path or ctx.monitor_last_mtime != mtime:
        ctx.monitor_last_path = cfg.data_path
        ctx.monitor_last_mtime = mtime

        def run() -> None:
            try:
                from dune_tension.summaries import update_tension_logs

                update_tension_logs(cfg)
                _request_live_summary_refresh(ctx, cfg)
                LOGGER.info(
                    "Updated tension logs for %s layer %s",
                    cfg.apa_name,
                    cfg.layer,
                )
            except Exception as exc:
                LOGGER.warning("Failed to update logs: %s", exc)

        if ctx.monitor_thread is None or not ctx.monitor_thread.is_alive():
            ctx.monitor_thread = Thread(target=run, daemon=True)
            ctx.monitor_thread.start()

    ctx.root.after(1000, lambda: monitor_tension_logs(ctx))


@_run_in_thread
def refresh_tension_logs(ctx: GUIContext, inputs: WorkerInputs) -> None:
    """Force an update of the tension logs regardless of file changes."""

    cfg = _make_config_from_inputs(inputs)

    try:
        from dune_tension.summaries import update_tension_logs

        update_tension_logs(cfg)
        _request_live_summary_refresh(ctx, cfg)
        try:
            ctx.monitor_last_path = cfg.data_path
            ctx.monitor_last_mtime = os.path.getmtime(cfg.data_path)
        except OSError:
            ctx.monitor_last_mtime = None
        LOGGER.info(
            "Manually refreshed tension logs for %s layer %s",
            cfg.apa_name,
            cfg.layer,
        )
    except Exception as exc:
        LOGGER.warning("Failed to refresh logs: %s", exc)


def manual_goto(ctx: GUIContext) -> None:
    text = ctx.widgets.entry_xy.get()
    try:
        x_str, y_str = text.split(",")
        x_val = float(x_str.strip())
        y_val = float(y_str.strip())
    except ValueError:
        LOGGER.warning("Invalid coordinates: %s", text)
        return
    ctx.goto_xy(x_val, y_val)


def manual_increment(ctx: GUIContext, dx: float, dy: float) -> None:
    cur_x, cur_y = ctx.get_xy()
    w = ctx.widgets
    if (w.side_var.get() == "A" and not w.flipped_var.get()) or (
        w.side_var.get() == "B" and w.flipped_var.get()
    ):
        x_sign = 1.0
    else:
        x_sign = -1.0

    new_x = cur_x + x_sign * dx * 0.1
    new_y = cur_y + dy * 0.1
    new_x = round(new_x, 1)
    new_y = round(new_y, 1)

    ctx.goto_xy(new_x, new_y)


def update_focus_command_indicator(ctx: GUIContext, value: int) -> None:
    ctx.focus_command_var.set(str(value))
    canvas = ctx.focus_command_canvas
    dot = ctx.focus_command_dot
    if not canvas or not dot:
        return
    length = canvas.winfo_width()
    slider = ctx.widgets.focus_slider
    low = int(slider["from"])
    high = int(slider["to"])
    if high == low:
        x_pos = 0
    else:
        x_pos = (value - low) / (high - low) * length
    radius = 3
    canvas.coords(dot, x_pos - radius, 5 - radius, x_pos + radius, 5 + radius)


def handle_close(ctx: GUIContext) -> None:
    ctx.stop_event.set()
    ctx.servo_controller.stop_loop()
    if ctx.log_binding is not None:
        try:
            ctx.log_binding.close()
        except Exception:
            pass
    if ctx.valve_controller is not None:
        try:
            ctx.valve_controller.close()
        except Exception:
            pass
    try:
        import sounddevice as sd  # type: ignore

        sd.stop()
    except Exception:
        pass
    try:
        ctx.root.destroy()
    except Exception:
        pass


def _make_config_from_widgets(ctx: GUIContext):
    w = ctx.widgets
    try:
        samples = int(w.entry_samples.get())
    except ValueError:
        samples = 3
    try:
        conf = float(w.entry_confidence.get())
    except ValueError:
        conf = 0.7

    return make_config(
        apa_name=w.entry_apa.get(),
        layer=w.layer_var.get(),
        side=w.side_var.get(),
        flipped=w.flipped_var.get(),
        samples_per_wire=samples,
        confidence_threshold=conf,
        plot_audio=w.plot_audio_var.get(),
    )


def _make_config_from_inputs(inputs: WorkerInputs):
    return make_config(
        apa_name=inputs.apa_name,
        layer=inputs.layer,
        side=inputs.side,
        flipped=inputs.flipped,
        samples_per_wire=inputs.samples,
        confidence_threshold=inputs.confidence,
        plot_audio=inputs.plot_audio,
    )


def _cleanup_after_measurement(
    ctx: GUIContext,
    tensiometer: "Tensiometer | None",
    *,
    reset_estimated_time: bool = True,
) -> None:
    if tensiometer is not None:
        try:
            tensiometer.close()
        except Exception:
            pass
    try:
        from dune_tension.plc_io import reset_plc

        reset_plc()
    except Exception:
        pass
    if reset_estimated_time:
        _set_estimated_time(ctx, "Not running")
    ctx.stop_event.clear()
