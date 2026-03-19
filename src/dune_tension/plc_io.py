import logging
import os
import threading
import time
from random import gauss
from typing import Any

import requests

from dune_tension.geometry import X_MAX, X_MIN, Y_MAX, Y_MIN, comb_positions
from dune_tension.plc_direct import (
    PLCCommunicationError,
    PLCTagReadError,
    PLCTagWriteError,
    is_direct_plc_available,
    read_tag_value,
    write_tag_value,
)

LOGGER = logging.getLogger(__name__)

# Lock used for individual HTTP request/response calls.
PLC_LOCK = threading.RLock()

# Separate lock to keep motion command sequences from interleaving.
_MOTION_LOCK = threading.Lock()

# Shared HTTP session for connection pooling and lower request latency.
_HTTP_SESSION: Any = None

_REQUEST_EXCEPTION = getattr(
    getattr(requests, "exceptions", object()),
    "RequestException",
    Exception,
)

# Amount of travel, in mm, assumed to be lost when reversing X direction
BACKLASH_DEADZONE = 0.5

# Track our best guess of the true position, accounting for backlash
_TRUE_XY = [None, None]

# Track the last X movement direction and remaining deadzone to take up
_LAST_X_DIR = 0
_X_DEADZONE_LEFT = 0.0

DEFAULT_TENSION_SERVER_URL = "http://192.168.137.1:5000"
TENSION_SERVER_URL = DEFAULT_TENSION_SERVER_URL
DEFAULT_PLC_IO_MODE = "server"
VALID_PLC_IO_MODES = {"server", "direct"}
_WARNED_PLC_IO_MODE_VALUES: set[str | None] = set()
IDLE_MOVE_TYPE = 0
IDLE_STATE = 1
XY_MOVE_TYPE = 2
XY_STATE = 3

HTTP_CONNECT_TIMEOUT = 0.35
HTTP_READ_TIMEOUT = 1.0
HTTP_RETRIES = 1
READ_RETRY_TIMEOUT = 5.0
READ_RETRY_INTERVAL = 0.05
STATE_POLL_INTERVAL = 0.05
IDLE_WAIT_TIMEOUT = 20.0
MOVE_WAIT_TIMEOUT = 120.0


def get_tension_server_url() -> str:
    """Return the configured tension server URL."""
    return os.getenv("TENSION_SERVER_URL", DEFAULT_TENSION_SERVER_URL).strip()


def _warn_plc_mode_once(raw_mode: str | None, message: str) -> None:
    """Log a PLC mode warning only once per raw env value."""
    if raw_mode in _WARNED_PLC_IO_MODE_VALUES:
        return
    _WARNED_PLC_IO_MODE_VALUES.add(raw_mode)
    LOGGER.warning(message)


def get_plc_io_mode() -> str:
    """Return the configured PLC transport mode."""
    raw_mode = os.getenv("PLC_IO_MODE")
    if raw_mode is None:
        _warn_plc_mode_once(
            None,
            "PLC_IO_MODE is not set. Falling back to 'server' mode.",
        )
        return DEFAULT_PLC_IO_MODE

    mode = raw_mode.strip().lower()
    if mode not in VALID_PLC_IO_MODES:
        _warn_plc_mode_once(
            raw_mode,
            f"Invalid PLC_IO_MODE={raw_mode!r}. Falling back to 'server' mode.",
        )
        return DEFAULT_PLC_IO_MODE

    return mode


def _get_http_session() -> Any:
    """Return a lazily-created requests session when available."""
    global _HTTP_SESSION
    if _HTTP_SESSION is None and hasattr(requests, "Session"):
        _HTTP_SESSION = requests.Session()
    return _HTTP_SESSION


def _request_with_retries(
    method: str,
    url: str,
    *,
    json_payload: Any = None,
    timeout: tuple[float, float] = (HTTP_CONNECT_TIMEOUT, HTTP_READ_TIMEOUT),
    retries: int = HTTP_RETRIES,
) -> Any:
    """Send an HTTP request with retry on network exceptions."""
    last_error: Exception | None = None

    for attempt in range(retries + 1):
        try:
            with PLC_LOCK:
                session = _get_http_session()
                if session is not None:
                    return session.request(
                        method=method,
                        url=url,
                        json=json_payload,
                        timeout=timeout,
                    )

                request_fn = getattr(requests, method.lower(), None)
                if request_fn is None:
                    raise RuntimeError(f"requests.{method.lower()} is unavailable")

                if json_payload is None:
                    return request_fn(url, timeout=timeout)
                return request_fn(url, json=json_payload, timeout=timeout)
        except _REQUEST_EXCEPTION as exc:
            last_error = exc
        except RuntimeError as exc:
            last_error = exc

        if attempt < retries:
            time.sleep(0.05 * (attempt + 1))

    raise RuntimeError(str(last_error) if last_error is not None else "Request failed")


def _parse_read_value(tag_name: str, payload: Any) -> Any:
    """Parse current and legacy tension server response payloads."""
    if not isinstance(payload, dict):
        raise TypeError("response is not a JSON object")

    if "value" in payload:
        return payload["value"]

    if tag_name not in payload:
        raise KeyError(f"missing key '{tag_name}'")

    value = payload[tag_name]
    if isinstance(value, (list, tuple)):
        if len(value) > 1:
            return value[1]
        raise IndexError(f"tag '{tag_name}' list missing value index")
    return value


def _read_numeric_tag(tag_name: str) -> float:
    """Read and coerce a numeric PLC tag, raising on communication/protocol errors."""
    value = read_tag(tag_name)
    if isinstance(value, dict) and "error" in value:
        raise RuntimeError(f"Failed to read {tag_name}: {value['error']}")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Invalid numeric value for {tag_name}: {value!r}") from exc


def _write_required(tag_name: str, value: Any) -> bool:
    """Write a tag and return False when the operation fails."""
    response = write_tag(tag_name, value)
    if isinstance(response, dict) and "error" in response:
        LOGGER.warning("Failed to write %s: %s", tag_name, response["error"])
        return False
    return True


def is_motion_target_in_bounds(x_target: float, y_target: float) -> bool:
    """Return whether the target is legal for PLC XY motion."""

    return X_MIN <= float(x_target) <= X_MAX and Y_MIN <= float(y_target) <= Y_MAX


def _read_tag_server(tag_name: str) -> Any:
    """Read a PLC tag through the HTTP tension server."""
    url = f"{get_tension_server_url()}/tags/{tag_name}"

    try:
        response = _request_with_retries("GET", url)
        if response.status_code == 200:
            try:
                return _parse_read_value(tag_name, response.json())
            except (KeyError, TypeError, ValueError, IndexError) as exc:
                return {"error": f"Malformed response: {exc}"}
        return {
            "error": "Failed to read tag",
            "status_code": response.status_code,
        }
    except Exception as exc:
        return {"error": str(exc)}


def _read_tag_direct(tag_name: str) -> Any:
    """Read a PLC tag directly through pycomm3."""
    try:
        return read_tag_value(tag_name)
    except (PLCCommunicationError, PLCTagReadError) as exc:
        return {"error": str(exc)}
    except Exception as exc:
        return {"error": str(exc)}


def _write_tag_server(tag_name: str, value: Any) -> dict[str, Any]:
    """Write a PLC tag through the HTTP tension server."""
    url = f"{get_tension_server_url()}/tags/{tag_name}"
    payload = {"value": value}

    try:
        response = _request_with_retries("POST", url, json_payload=payload)
    except Exception as exc:
        return {"error": str(exc)}

    if response.status_code != 200:
        return {"error": "Failed to write tag", "status_code": response.status_code}

    try:
        return response.json()
    except ValueError:
        return {tag_name: value, "value": value}


def _write_tag_direct(tag_name: str, value: Any) -> dict[str, Any]:
    """Write a PLC tag directly through pycomm3."""
    try:
        write_tag_value(tag_name, value)
    except (PLCCommunicationError, PLCTagWriteError) as exc:
        return {"error": str(exc)}
    except Exception as exc:
        return {"error": str(exc)}
    return {"tag": tag_name, "value": value, tag_name: value}


def read_tag(
    tag_name: str,
    *,
    timeout: float = READ_RETRY_TIMEOUT,
    retry_interval: float = READ_RETRY_INTERVAL,
) -> Any:
    """Read a PLC tag with retries until timeout expires."""
    end_time = time.monotonic() + timeout
    last_error: dict[str, Any] | None = None
    transport = get_plc_io_mode()

    while time.monotonic() < end_time:
        if transport == "direct":
            last_error = _read_tag_direct(tag_name)
        else:
            last_error = _read_tag_server(tag_name)

        if not (isinstance(last_error, dict) and "error" in last_error):
            return last_error

        time.sleep(retry_interval)

    return last_error if last_error is not None else {"error": "Read timeout"}


def get_xy() -> tuple[float, float]:
    """Get the current position of the tensioning system."""
    x = _read_numeric_tag("X_axis.ActualPosition")
    y = _read_numeric_tag("Y_axis.ActualPosition")
    return x, y


def _ensure_tracked_xy() -> tuple[float, float]:
    """Return the tracked XY position, seeding it from the PLC if needed."""

    if _TRUE_XY[0] is None or _TRUE_XY[1] is None:
        _TRUE_XY[0], _TRUE_XY[1] = get_xy()
    return tuple(_TRUE_XY)


def get_cached_xy() -> tuple[float, float]:
    """Return the internally tracked XY position.

    ``goto_xy`` keeps ``_TRUE_XY`` updated whenever a move command is issued.
    If no tracked state exists yet, seed it from the PLC once. Callers should
    prefer this over ``get_xy`` when doing motion math so backlash compensation
    stays internally consistent.
    """

    with PLC_LOCK:
        return _ensure_tracked_xy()


def get_state() -> int:
    """Get the current state of the tensioning system."""
    return int(_read_numeric_tag("STATE"))


def get_movetype() -> int:
    """Get the current move type of the tensioning system."""
    return int(_read_numeric_tag("MOVE_TYPE"))


def write_tag(tag_name: str, value: Any) -> dict[str, Any]:
    """Write a value to a PLC tag via the tension server."""
    if get_plc_io_mode() == "direct":
        return _write_tag_direct(tag_name, value)
    return _write_tag_server(tag_name, value)


def goto_xy(
    x_target: float,
    y_target: float,
    *,
    speed: float = 300,
    deadzone: float = BACKLASH_DEADZONE,
    check_comb: bool = True,
    idle_timeout: float = IDLE_WAIT_TIMEOUT,
    move_timeout: float = MOVE_WAIT_TIMEOUT,
) -> bool:
    """Move the winder to a given position with bounded waits."""

    global _TRUE_XY, _LAST_X_DIR, _X_DEADZONE_LEFT

    _ensure_tracked_xy()

    if check_comb:
        cur_x = _TRUE_XY[0]
        crosses = any(
            (cur_x < c < x_target) or (x_target < c < cur_x) for c in comb_positions
        )
        if crosses:
            transit_y = float(Y_MIN)
            if not goto_xy(
                cur_x,
                transit_y,
                speed=speed,
                deadzone=deadzone,
                check_comb=False,
                idle_timeout=idle_timeout,
                move_timeout=move_timeout,
            ):
                return False
            if not goto_xy(
                x_target,
                transit_y,
                speed=speed,
                deadzone=deadzone,
                check_comb=False,
                idle_timeout=idle_timeout,
                move_timeout=move_timeout,
            ):
                return False
            return goto_xy(
                x_target,
                y_target,
                speed=speed,
                deadzone=deadzone,
                check_comb=False,
                idle_timeout=idle_timeout,
                move_timeout=move_timeout,
            )

    with _MOTION_LOCK:
        if not is_motion_target_in_bounds(x_target, y_target):
            LOGGER.warning(
                "Motion target %s,%s out of bounds. Please enter a valid position.",
                x_target,
                y_target,
            )
            return False

        # Compute backlash-adjusted target state but only commit after successful move.
        delta_x = x_target - _TRUE_XY[0]
        direction = 1 if delta_x > 0 else -1 if delta_x < 0 else 0
        next_last_x_dir = _LAST_X_DIR
        next_deadzone_left = _X_DEADZONE_LEFT

        if direction != 0 and direction != next_last_x_dir:
            next_deadzone_left = deadzone
            next_last_x_dir = direction

        move_x = abs(delta_x)
        actual_move_x = 0.0
        if direction != 0:
            if next_deadzone_left > 0:
                if move_x <= next_deadzone_left:
                    next_deadzone_left -= move_x
                else:
                    actual_move_x = direction * (move_x - next_deadzone_left)
                    next_deadzone_left = 0.0
            else:
                actual_move_x = delta_x

        predicted_true_x = _TRUE_XY[0] + actual_move_x
        predicted_true_y = y_target

        idle_deadline = time.monotonic() + idle_timeout
        while True:
            try:
                current_state = get_state()
            except RuntimeError as exc:
                LOGGER.warning("Unable to read STATE before move: %s", exc)
                return False
            if current_state == IDLE_STATE:
                break
            if time.monotonic() >= idle_deadline:
                LOGGER.warning(
                    "Timed out waiting for idle state before move to %s,%s.",
                    x_target,
                    y_target,
                )
                return False
            time.sleep(STATE_POLL_INTERVAL)

        if not set_speed(speed):
            return False

        writes = (
            ("MOVE_TYPE", IDLE_MOVE_TYPE),
            ("STATE", IDLE_STATE),
            ("X_POSITION", x_target),
            ("Y_POSITION", y_target),
            ("MOVE_TYPE", XY_MOVE_TYPE),
        )
        for tag_name, value in writes:
            if not _write_required(tag_name, value):
                return False

        move_deadline = time.monotonic() + move_timeout
        while True:
            try:
                move_type = get_movetype()
            except RuntimeError as exc:
                LOGGER.warning(
                    "Unable to read MOVE_TYPE while waiting for move: %s", exc
                )
                return False
            if move_type != XY_MOVE_TYPE:
                break
            if time.monotonic() >= move_deadline:
                LOGGER.warning("Timed out waiting for move completion.")
                return False
            time.sleep(STATE_POLL_INTERVAL)

        _TRUE_XY[0] = predicted_true_x
        _TRUE_XY[1] = predicted_true_y
        _LAST_X_DIR = next_last_x_dir
        _X_DEADZONE_LEFT = next_deadzone_left
        return True


def reset_plc() -> bool:
    """Reset the PLC to its initial state."""
    with _MOTION_LOCK:
        return (
            _write_required("MOVE_TYPE", IDLE_MOVE_TYPE)
            and _write_required("STATE", IDLE_STATE)
            and set_speed(0)
        )


def increment(increment_x: float, increment_y: float) -> bool:
    """Move by an XY increment, preferring the cached position."""
    x, y = get_cached_xy()
    if x is None or y is None:
        try:
            x, y = get_xy()
        except RuntimeError as exc:
            LOGGER.warning("Unable to read XY for increment: %s", exc)
            return False
    return goto_xy(x + increment_x, y + increment_y)


def wiggle(step: float) -> bool:
    """Wiggle the winder by a given step size in a background thread."""

    def _do_wiggle() -> None:
        try:
            y_wiggle = gauss(0, step)
            increment(0, y_wiggle)
            LOGGER.info("Wiggling by %s mm", y_wiggle)
        finally:
            reset_plc()

    threading.Thread(target=_do_wiggle, daemon=True).start()
    return True


def set_speed(speed: float = 300) -> bool:
    """Set the speed of the winder."""
    if not (0 <= speed <= 1000):
        LOGGER.warning(
            "Speed %s out of bounds. Please enter a value between 0 and 1000.",
            speed,
        )
        return False

    response = write_tag("XY_SPEED", speed)
    if isinstance(response, dict) and "error" in response:
        LOGGER.warning("Failed to set speed: %s", response["error"])
        return False
    return True


def is_web_server_active() -> bool:
    """Check whether the HTTP tension server is active."""
    last_error: Exception | None = None

    for endpoint in ("/health", "/"):
        try:
            response = _request_with_retries(
                "GET",
                f"{get_tension_server_url()}{endpoint}",
                timeout=(HTTP_CONNECT_TIMEOUT, 0.75),
                retries=0,
            )
            if 200 <= response.status_code < 500:
                return True
        except Exception as exc:
            last_error = exc

    LOGGER.warning("An error occurred while checking the server: %s", last_error)
    return False


def is_plc_available() -> bool:
    """Check whether the configured PLC transport is available."""
    if get_plc_io_mode() == "direct":
        return is_direct_plc_available()
    return is_web_server_active()


# ---------------------------------------------------------------------------
# Spoofing utilities
# ---------------------------------------------------------------------------

# Track spoofed position so that movement functions can update it
_SPOOF_XY = [6300.0, 200.0]


def spoof_get_xy() -> tuple[float, float]:
    """Return the current spoofed XY position."""
    return tuple(_SPOOF_XY)


def spoof_goto_xy(x_target: float, y_target: float, **_: object) -> bool:
    """Pretend to move the winder and update the spoofed position."""
    if not is_motion_target_in_bounds(x_target, y_target):
        LOGGER.warning("[spoof] Motion target %s,%s out of bounds.", x_target, y_target)

    LOGGER.info("[spoof] Moving to %s,%s", x_target, y_target)
    _SPOOF_XY[0] = x_target
    _SPOOF_XY[1] = y_target
    return True


def spoof_wiggle(step: float) -> bool:
    """Pretend to wiggle the winder."""
    LOGGER.info("[spoof] Wiggling by +/- %s mm", step)
    return True
