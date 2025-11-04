"""Utilities for triggering the pneumatic valve over the serial relay.

This module provides a high-level :func:`air_pulse` function that opens the
valve for a requested duration without blocking the caller.  It also exposes a
simple command-line interface that fires a pulse each time the user presses the
spacebar.
"""

from __future__ import annotations

import argparse
import sys
import termios
import threading
import time
import tty
from dataclasses import dataclass
from types import TracebackType
from typing import IO, Iterable

import serial
from serial import Serial
from serial.tools import list_ports

_OPEN_COMMAND = bytes([0xA0, 0x01, 0x01, 0xA2])
_CLOSE_COMMAND = bytes([0xA0, 0x01, 0x00, 0xA1])


class DeviceNotFoundError(RuntimeError):
    """Raised when the USB relay controlling the valve cannot be located."""


def _find_valve_port(*, vendor: str, product: str) -> str:
    """Return the system path for the configured valve relay.

    Args:
        vendor: USB vendor identifier expressed as a hexadecimal string.
        product: USB product identifier expressed as a hexadecimal string.

    Raises:
        DeviceNotFoundError: If no matching serial port can be located.
    """

    hardware_id = f"{vendor}:{product}"
    for port in list_ports.comports():
        if hardware_id in port.hwid:
            return port.device
    raise DeviceNotFoundError(
        "Unable to locate the USB relay controlling the air valve."
    )


@dataclass(slots=True)
class ValveConfig:
    """Configuration for the valve relay connection."""

    vendor_id: str = "1A86"
    product_id: str = "7523"
    baud_rate: int = 9600
    serial_timeout: float = 0


class ValveController:
    """Manage the serial connection to the valve relay."""

    def __init__(self, *, port: str | None = None, config: ValveConfig | None = None):
        self._config = config or ValveConfig()
        serial_port = port or _find_valve_port(
            vendor=self._config.vendor_id, product=self._config.product_id
        )
        try:
            self._serial = Serial(
                serial_port,
                self._config.baud_rate,
                timeout=self._config.serial_timeout,
                write_timeout=self._config.serial_timeout,
            )
        except serial.SerialException as exc:  # pragma: no cover - passthrough error
            raise RuntimeError("Unable to open the valve relay serial port.") from exc

        self._lock = threading.Lock()
        self._close_event: threading.Event | None = None
        self._close_thread: threading.Thread | None = None
        self._strum_stop: threading.Event | None = None
        self._strum_thread: threading.Thread | None = None
        self._strum_open_time: float = 0.05
        self._strum_closed_time: float = 0.05

    def pulse(self, duration: float) -> None:
        """Open the valve immediately and close it after *duration* seconds.

        The call returns as soon as the valve has been commanded to open; the
        closing action is handled by a dedicated background thread.
        """

        if duration <= 0:
            raise ValueError("Pulse duration must be positive.")

        start = time.perf_counter()

        with self._lock:
            if self._close_event is not None:
                self._close_event.set()

            self._write(_OPEN_COMMAND)

            cancel_event = threading.Event()
            thread = threading.Thread(
                target=self._close_after_delay,
                args=(start, duration, cancel_event),
                daemon=True,
                name="valve-close",
            )
            self._close_event = cancel_event
            self._close_thread = thread
            thread.start()

    def close(self) -> None:
        """Close the valve and tidy up background resources."""

        self.stop_strum()

        thread: threading.Thread | None
        with self._lock:
            if self._close_event is not None:
                self._close_event.set()
            thread = self._close_thread
            self._close_event = None
            self._close_thread = None

        if thread is not None:
            thread.join(timeout=0.1)

        with self._lock:
            if self._serial.is_open:
                self._write(_CLOSE_COMMAND)
                self._serial.close()

    def set_dwell_time(self, duration: float) -> None:
        """Adjust the open/closed dwell times used while strumming."""

        dwell = max(float(duration), 0.0)
        # Ensure we avoid a zero-duration busy loop; a short minimum delay keeps the
        # background thread cooperative while still responding quickly to updates.
        minimum = 0.005
        dwell = max(dwell, minimum)
        with self._lock:
            self._strum_open_time = dwell
            self._strum_closed_time = dwell

    def start_strum(self) -> None:
        """Begin continuously pulsing the valve in a background thread."""

        with self._lock:
            if self._strum_thread is not None and self._strum_thread.is_alive():
                return

            close_thread: threading.Thread | None = None
            if self._close_event is not None:
                self._close_event.set()
                close_thread = self._close_thread
            self._close_event = None
            self._close_thread = None

            stop_event = threading.Event()
            thread = threading.Thread(
                target=self._strum_loop,
                args=(stop_event,),
                daemon=True,
                name="valve-strum",
            )
            self._strum_stop = stop_event
            self._strum_thread = thread
            thread.start()

        if close_thread is not None:
            close_thread.join(timeout=0.1)

    def stop_strum(self) -> None:
        """Halt any active strumming thread and ensure the valve is closed."""

        stop_event: threading.Event | None
        thread: threading.Thread | None
        with self._lock:
            stop_event = self._strum_stop
            thread = self._strum_thread
            self._strum_stop = None
            self._strum_thread = None

        if stop_event is not None:
            stop_event.set()
        if thread is not None:
            thread.join(timeout=0.2)

        with self._lock:
            if self._serial.is_open:
                self._write(_CLOSE_COMMAND)

    def _strum_loop(self, stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            with self._lock:
                if not self._serial.is_open:
                    break
                self._write(_OPEN_COMMAND)

            if stop_event.wait(self._strum_open_time):
                break

            with self._lock:
                if not self._serial.is_open:
                    break
                self._write(_CLOSE_COMMAND)

            if stop_event.wait(self._strum_closed_time):
                break

        with self._lock:
            if self._serial.is_open:
                self._write(_CLOSE_COMMAND)

    def __enter__(self) -> "ValveController":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    def _write(self, payload: bytes) -> None:
        self._serial.write(payload)
        self._serial.flush()

    def _close_after_delay(
        self, start: float, duration: float, cancel_event: threading.Event
    ) -> None:
        target = start + duration
        while not cancel_event.is_set():
            remaining = target - time.perf_counter()
            if remaining <= 0:
                break
            sleep_interval = _sleep_interval(remaining)
            cancel_event.wait(sleep_interval)

        if cancel_event.is_set():
            return

        with self._lock:
            if self._close_event is cancel_event:
                self._write(_CLOSE_COMMAND)
                self._close_event = None
                self._close_thread = None


def _sleep_interval(remaining: float) -> float:
    if remaining > 0.01:
        return remaining - 0.005
    if remaining > 0.002:
        return 0.001
    return max(remaining / 2, 0.0005)


_DEFAULT_CONTROLLER: ValveController | None = None
_DEFAULT_CONTROLLER_LOCK = threading.Lock()


def _get_default_controller() -> ValveController:
    global _DEFAULT_CONTROLLER
    with _DEFAULT_CONTROLLER_LOCK:
        if _DEFAULT_CONTROLLER is None:
            _DEFAULT_CONTROLLER = ValveController()
    return _DEFAULT_CONTROLLER


def air_pulse(t_seconds: float) -> None:
    controller = _get_default_controller()
    controller.pulse(t_seconds)


def _ensure_tty(stream: IO[str]) -> None:
    if not stream.isatty():
        raise RuntimeError("The CLI requires an interactive terminal.")


def _cli(duration: float, port: str | None = None) -> int:
    try:
        _ensure_tty(sys.stdin)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    try:
        controller = ValveController(port=port)
    except DeviceNotFoundError:
        print("Error: Air valve controller not found.", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(
        "Press SPACE to trigger the valve pulse, or 'q' to quit."
        f" Pulse duration: {duration:.3f} s"
    )

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            char = sys.stdin.read(1)
            if char == " ":
                try:
                    controller.pulse(duration)
                    print(
                        f"Pulsed valve for {duration * 1_000:.1f} ms",
                        flush=True,
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    print(f"Error while pulsing valve: {exc}", file=sys.stderr)
                    return 1
            elif char in {"q", "Q", "\x04"}:  # Ctrl+D
                break
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        controller.close()

    print("Exiting.")
    return 0


def _parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pulse the air valve whenever the spacebar is pressed.",
    )
    parser.add_argument(
        "duration",
        type=float,
        help="Length of the valve pulse in seconds.",
    )
    parser.add_argument(
        "--port",
        help="Serial port path for the valve relay. Auto-detected when omitted.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    if args.duration <= 0:
        print("Error: duration must be a positive number.", file=sys.stderr)
        return 2
    return _cli(args.duration, port=args.port)


if __name__ == "__main__":
    raise SystemExit(main())
