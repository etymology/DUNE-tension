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
from math import ceil
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
        self._strum_stop_event: threading.Event | None = None
        self._strum_thread: threading.Thread | None = None

    def pulse(self, duration: float) -> None:
        """Open the valve immediately and close it after *duration* seconds.

        The call returns as soon as the valve has been commanded to open; the
        closing action is handled by a dedicated background thread.
        """
        if duration <= 0:
            raise ValueError("Pulse duration must be positive.")

        n_pad = ceil(duration * self._config.baud_rate / 10)  # 8N1
        padding = b'\x00' * n_pad                  # choose a byte the device ignores
        self._serial.write(_OPEN_COMMAND + padding + _CLOSE_COMMAND)
        self._serial.flush()


    def start_strum(self) -> None:
        """Begin emitting 10 ms air pulses every second on a background thread."""

        with self._lock:
            if self._strum_thread is not None and self._strum_thread.is_alive():
                return

            stop_event = threading.Event()
            thread = threading.Thread(
                target=self._strum_loop,
                args=(stop_event,),
                daemon=True,
                name="valve-strum",
            )
            self._strum_stop_event = stop_event
            self._strum_thread = thread
            thread.start()

    def stop_strum(self) -> None:
        """Stop the background strumming loop if it is running."""

        with self._lock:
            stop_event = self._strum_stop_event
            thread = self._strum_thread
            self._strum_stop_event = None
            self._strum_thread = None

        if stop_event is not None:
            stop_event.set()
        if thread is not None:
            thread.join(timeout=0.1)

    def __enter__(self) -> "ValveController":
        return self

    def close(self) -> None:
        """Close the serial connection to the valve relay."""
        self.stop_strum()
        self._serial.close()
        
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

    def _strum_loop(self, stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            self.pulse(0.003)
            if stop_event.wait(1):
                break

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
                    # controller.pulse(duration)
                    controller.start_strum()
                    print(f"Pulsed valve for {duration:.3f} seconds.")
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
