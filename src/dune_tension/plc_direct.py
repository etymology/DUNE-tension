import atexit
import logging
import os
import threading
from typing import Any, Callable

LOGGER = logging.getLogger(__name__)

DEFAULT_PLC_IP_ADDRESS = "192.168.140.13"
DEFAULT_PLC_COMM_RETRIES = 2

_comm: Any = None
_comm_address: str | None = None
_comm_lock = threading.RLock()


class PLCCommunicationError(RuntimeError):
    """Raised when PLC communication cannot be established."""


class PLCTagReadError(RuntimeError):
    """Raised when a PLC tag cannot be read successfully."""


class PLCTagWriteError(RuntimeError):
    """Raised when a PLC tag cannot be written successfully."""


def get_plc_ip_address() -> str:
    """Return the configured PLC IP address."""
    return os.getenv("PLC_IP_ADDRESS", DEFAULT_PLC_IP_ADDRESS).strip()


def get_plc_comm_retries() -> int:
    """Return the configured PLC communication retry count."""
    raw = os.getenv("PLC_COMM_RETRIES", str(DEFAULT_PLC_COMM_RETRIES)).strip()
    try:
        return max(1, int(raw))
    except ValueError:
        LOGGER.warning(
            "Invalid PLC_COMM_RETRIES=%r. Falling back to %s.",
            raw,
            DEFAULT_PLC_COMM_RETRIES,
        )
        return DEFAULT_PLC_COMM_RETRIES


def _load_pycomm3() -> tuple[Any, type[BaseException]]:
    """Import pycomm3 lazily so server mode does not require it at import time."""
    try:
        from pycomm3 import logix_driver
        from pycomm3.exceptions import CommError
    except Exception as exc:
        raise RuntimeError(
            "pycomm3 is required when PLC_IO_MODE=direct or when running the tension server"
        ) from exc
    return logix_driver, CommError


def _close_comm() -> None:
    """Close and clear the PLC communication object."""
    global _comm, _comm_address
    if _comm is None:
        _comm_address = None
        return
    try:
        _comm.close()
    except Exception:
        pass
    _comm = None
    _comm_address = None


atexit.register(_close_comm)


def _ensure_comm() -> Any:
    """Return an open PLC connection, creating it when necessary."""
    global _comm, _comm_address
    address = get_plc_ip_address()
    logix_driver, _ = _load_pycomm3()

    if _comm is None or _comm_address != address:
        _close_comm()
        _comm = logix_driver.LogixDriver(address)
        _comm_address = address

    if not bool(getattr(_comm, "connected", False)):
        _comm.open()
    return _comm


def _run_plc_call(fn: Callable[[Any], Any]) -> Any:
    """Run a PLC operation with reconnect retries on communication failure."""
    last_error: BaseException | None = None
    _, comm_error_type = _load_pycomm3()

    for _ in range(get_plc_comm_retries()):
        with _comm_lock:
            try:
                return fn(_ensure_comm())
            except comm_error_type as exc:
                last_error = exc
                _close_comm()

    raise PLCCommunicationError("PLC communication error") from last_error


def _extract_value(plc_result: Any) -> tuple[Any, str | None]:
    """Normalize pycomm3 result objects into primitive values plus optional error."""
    error = getattr(plc_result, "error", None)
    value = getattr(plc_result, "value", plc_result)
    if error:
        return value, str(error)
    return value, None


def read_tag_value(tag_name: str) -> Any:
    """Read a PLC tag and return its primitive value."""
    result = _run_plc_call(lambda comm: comm.read(tag_name))
    if result is None:
        raise PLCTagReadError("Tag not found")

    value, error = _extract_value(result)
    if error:
        raise PLCTagReadError(error)
    return value


def write_tag_value(tag_name: str, value: Any) -> Any:
    """Write a PLC tag value and return the normalized pycomm3 response value."""
    result = _run_plc_call(lambda comm: comm.write((tag_name, value)))
    normalized_value, error = _extract_value(result)
    if error:
        raise PLCTagWriteError(error)
    return normalized_value


def is_direct_plc_available() -> bool:
    """Return whether a direct PLC connection can be opened."""
    try:
        _run_plc_call(lambda _comm: True)
    except Exception as exc:
        LOGGER.warning("Direct PLC is unavailable: %s", exc)
        return False
    return True
