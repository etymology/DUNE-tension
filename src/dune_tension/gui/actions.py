"""Tkinter command callbacks used by the tensiometer GUI."""

from __future__ import annotations

from datetime import datetime
import os
import re
from threading import Thread
from typing import Any

import sounddevice as sd  # type: ignore
from tkinter import messagebox

from dune_tension.data_cache import (
    clear_outliers as cache_clear_outliers,
    clear_wire_range,
    get_dataframe,
    update_dataframe,
)
from dune_tension.results import EXPECTED_COLUMNS
from dune_tension.tensiometer import Tensiometer
from dune_tension.tensiometer_functions import make_config
from dune_tension.gui.context import GUIContext
from dune_tension.gui.state import save_state


def create_tensiometer(ctx: GUIContext) -> Tensiometer:
    """Instantiate a :class:`Tensiometer` from GUI selections."""

    w = ctx.widgets
    try:
        samples = int(w.entry_samples.get())
        if samples < 1:
            raise ValueError("Samples per wire must be ≥ 1")
    except ValueError as exc:
        messagebox.showerror("Input Error", str(exc))
        raise

    try:
        confidence = float(w.entry_confidence.get())
        if not (0.0 <= confidence <= 1.0):
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
    except ValueError as exc:
        messagebox.showerror("Input Error", str(exc))
        raise

    try:
        record_duration = float(w.entry_record_duration.get())
        measuring_duration = float(w.entry_measuring_duration.get())
    except ValueError as exc:
        messagebox.showerror("Input Error", str(exc))
        raise

    spoof_audio = bool(os.environ.get("SPOOF_AUDIO"))
    return Tensiometer(
        apa_name=w.entry_apa.get(),
        layer=w.layer_var.get(),
        side=w.side_var.get(),
        flipped=w.flipped_var.get(),
        spoof=spoof_audio,
        spoof_movement=bool(os.environ.get("SPOOF_PLC")),
        stop_event=ctx.stop_event,
        samples_per_wire=samples,
        confidence_threshold=confidence,
        record_duration=record_duration,
        measuring_duration=measuring_duration,
        plot_audio=w.plot_audio_var.get(),
        start_servo_loop=ctx.valve_controller.start_strum,
        stop_servo_loop=ctx.valve_controller.stop_strum,
        focus_wiggle=ctx.servo_controller.nudge_focus,
    )


def _run_in_thread(func):
    """Decorator to execute ``func`` in a daemon thread."""

    def wrapper(ctx: GUIContext, *args: Any, **kwargs: Any) -> None:
        def run() -> None:
            ctx.stop_event.clear()
            try:
                func(ctx, *args, **kwargs)
            finally:
                ctx.stop_event.clear()

        Thread(target=run, daemon=True).start()

    return wrapper


@_run_in_thread
def measure_calibrate(ctx: GUIContext) -> None:
    """Measure and calibrate a single wire."""

    tensiometer: Tensiometer | None = None
    try:
        tensiometer = create_tensiometer(ctx)
        wire_number = int(ctx.widgets.entry_wire.get())
        save_state(ctx)
        tensiometer.measure_calibrate(wire_number)
        print("Done calibrating wire", wire_number)
    finally:
        _cleanup_after_measurement(ctx, tensiometer)


@_run_in_thread
def measure_auto(ctx: GUIContext) -> None:
    """Automatically measure the full APA."""

    tensiometer: Tensiometer | None = None
    try:
        tensiometer = create_tensiometer(ctx)
        save_state(ctx)
        tensiometer.measure_auto()
    finally:
        _cleanup_after_measurement(ctx, tensiometer)


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


@_run_in_thread
def measure_list(ctx: GUIContext) -> None:
    """Measure a comma separated list of wire ranges."""

    tensiometer: Tensiometer | None = None
    try:
        tensiometer = create_tensiometer(ctx)
        ranges = _parse_ranges(ctx.widgets.entry_wire_list.get())
        wire_list: list[int] = []
        for start, end in ranges:
            wire_list.extend(range(start, end + 1))
        save_state(ctx)
        print(f"Measuring wires: {wire_list}")
        tensiometer.measure_list(wire_list, preserve_order=False)
    finally:
        _cleanup_after_measurement(ctx, tensiometer)


@_run_in_thread
def calibrate_background_noise(ctx: GUIContext) -> None:
    """Record background noise for filtering future recordings."""

    try:
        from dune_tension.audioProcessing import (
            calibrate_background_noise,
            get_samplerate,
        )

        samplerate = get_samplerate()
        if samplerate is None:
            print("Unable to access audio device")
            return
        calibrate_background_noise(int(samplerate))
        print("Background noise calibrated")
    finally:
        ctx.stop_event.clear()


@_run_in_thread
def measure_condition(ctx: GUIContext) -> None:
    """Measure wires whose tension satisfies the configured expression."""

    def _get_wires(config, expr: str) -> list[int]:
        from dune_tension.data_cache import get_dataframe  # local import for testing
        import pandas as pd

        df = get_dataframe(config.data_path)
        mask = (
            (df["apa_name"] == config.apa_name)
            & (df["layer"] == config.layer)
            & (df["side"] == config.side)
        )
        subset = df[mask].copy()
        subset["wire_number"] = pd.to_numeric(subset["wire_number"], errors="coerce")
        subset["tension"] = pd.to_numeric(subset["tension"], errors="coerce")
        subset = subset.dropna(subset=["wire_number", "tension"])
        subset = subset.sort_values("time").drop_duplicates(
            subset="wire_number", keep="last"
        )
        wires: list[int] = []
        for _, row in subset.iterrows():
            try:
                if eval(expr, {"t": float(row["tension"])}):
                    wires.append(int(row["wire_number"]))
            except Exception as exc:
                print(f"Invalid expression '{expr}': {exc}")
                return []
        return sorted(set(wires))

    tensiometer: Tensiometer | None = None
    expr = ctx.widgets.entry_condition.get().strip()
    if not expr:
        print("No condition specified")
        return

    try:
        tensiometer = create_tensiometer(ctx)
        save_state(ctx)
        wires = _get_wires(tensiometer.config, expr)
        if not wires:
            print(f"No wires satisfy: {expr}")
            return
        print(f"Measuring wires {wires} matching '{expr}'")
        for wire in wires:
            clear_wire_range(
                tensiometer.config.data_path,
                tensiometer.config.apa_name,
                tensiometer.config.layer,
                tensiometer.config.side,
                wire,
                wire,
            )
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


def clear_range(ctx: GUIContext) -> None:
    ranges = _parse_ranges(ctx.widgets.entry_clear_range.get())
    if not ranges:
        print("No valid range specified")
        return

    cfg = _make_config_from_widgets(ctx)
    for start, end in ranges:
        clear_wire_range(cfg.data_path, cfg.apa_name, cfg.layer, cfg.side, start, end)
    print(f"Cleared ranges: {ctx.widgets.entry_clear_range.get()}")


def clear_outliers(ctx: GUIContext) -> None:
    cfg = _make_config_from_widgets(ctx)
    try:
        conf = float(ctx.widgets.entry_confidence.get())
    except ValueError:
        conf = 0.7
    removed = cache_clear_outliers(
        cfg.data_path,
        cfg.apa_name,
        cfg.layer,
        cfg.side,
        2.0,
        conf,
    )
    if removed:
        print(f"Cleared outlier wires: {removed}")
    else:
        print("No outlier wires found")


def set_manual_tension(ctx: GUIContext) -> None:
    pairs = _parse_pairs(ctx.widgets.entry_set_tension.get())
    if not pairs:
        print("No valid tension pairs specified")
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
                    "wires": [],
                    "ttf": 0.0,
                    "time": datetime.now().isoformat(),
                    "zone": 0,
                    "wire_length": 0.0,
                    "t_sigma": 0.0,
                }
            )
            df.loc[len(df)] = row
    update_dataframe(cfg.data_path, df)
    print(f"Updated tensions: {pairs}")


def interrupt(ctx: GUIContext) -> None:
    ctx.stop_event.set()
    ctx.servo_controller.stop_loop()
    ctx.valve_controller.stop_strum()


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
                from dune_tension.analyze import update_tension_logs

                update_tension_logs(cfg)
                print(f"Updated tension logs for {cfg.apa_name} layer {cfg.layer}")
            except Exception as exc:
                print(f"Failed to update logs: {exc}")

        if ctx.monitor_thread is None or not ctx.monitor_thread.is_alive():
            ctx.monitor_thread = Thread(target=run, daemon=True)
            ctx.monitor_thread.start()

    ctx.root.after(2000, lambda: monitor_tension_logs(ctx))


def manual_goto(ctx: GUIContext) -> None:
    text = ctx.widgets.entry_xy.get()
    try:
        x_str, y_str = text.split(",")
        x_val = float(x_str.strip())
        y_val = float(y_str.strip())
    except ValueError:
        print(f"Invalid coordinates: {text}")
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
    ctx.valve_controller.stop_strum()
    try:
        ctx.valve_controller.close()
    except Exception:
        pass
    try:
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


def _cleanup_after_measurement(
    ctx: GUIContext, tensiometer: Tensiometer | None
) -> None:
    ctx.valve_controller.stop_strum()
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
    ctx.stop_event.clear()
