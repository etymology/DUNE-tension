import atexit
import os
import threading
from typing import Any, Callable

from flask import Flask, jsonify, request
from pycomm3 import logix_driver
from pycomm3.exceptions import CommError

PLC_IP_ADDRESS = os.getenv("PLC_IP_ADDRESS", "192.168.140.13")
SERVER_PORT = int(os.getenv("SERVER_PORT", "5000"))
PLC_COMM_RETRIES = max(1, int(os.getenv("PLC_COMM_RETRIES", "2")))
TEST_SERVER = os.getenv("TEST_SERVER", "0").strip().lower() in {"1", "true", "yes"}
DEBUG_SERVER = os.getenv("TENSION_SERVER_DEBUG", "0").strip().lower() in {
    "1",
    "true",
    "yes",
}

app = Flask(__name__)

_comm: logix_driver.LogixDriver | None = None
_comm_lock = threading.RLock()


def _close_comm() -> None:
    """Close and clear the PLC communication object."""
    global _comm
    if _comm is None:
        return
    try:
        _comm.close()
    except Exception:
        pass
    _comm = None


atexit.register(_close_comm)


def _ensure_comm() -> logix_driver.LogixDriver:
    """Return an open PLC connection, creating it when necessary."""
    global _comm
    if _comm is None:
        _comm = logix_driver.LogixDriver(PLC_IP_ADDRESS)
    if not bool(getattr(_comm, "connected", False)):
        _comm.open()
    return _comm


def _run_plc_call(fn: Callable[[logix_driver.LogixDriver], Any]) -> Any:
    """Run a PLC operation with reconnect retries on communication failure."""
    last_error: Exception | None = None
    for _ in range(PLC_COMM_RETRIES):
        with _comm_lock:
            try:
                return fn(_ensure_comm())
            except CommError as exc:
                last_error = exc
                _close_comm()
    raise CommError("PLC communication error") from last_error


def _extract_value(plc_result: Any) -> tuple[Any, str | None]:
    """Normalize pycomm3 result objects into primitive values plus optional error."""
    error = getattr(plc_result, "error", None)
    value = getattr(plc_result, "value", plc_result)
    if error:
        return value, str(error)
    return value, None


@app.get("/health")
def health() -> tuple[Any, int]:
    """Lightweight health endpoint for client liveness checks."""
    return jsonify({"ok": True}), 200


@app.route("/tags/<tag_name>", methods=["GET"])
def read_tag(tag_name: str) -> tuple[Any, int]:
    """Read and return a PLC tag using a stable JSON response shape."""
    try:
        result = _run_plc_call(lambda comm: comm.read(tag_name))
    except CommError:
        return jsonify({"error": "PLC communication error"}), 502

    if result is None:
        return jsonify({"error": "Tag not found"}), 404

    value, error = _extract_value(result)
    if error:
        return jsonify({"error": error}), 404

    return jsonify({"tag": tag_name, "value": value, tag_name: value}), 200


@app.route("/tags/<tag_name>", methods=["POST"])
def write_tag(tag_name: str) -> tuple[Any, int]:
    """Write a PLC tag value with retry-on-reconnect behavior."""
    payload = request.get_json(silent=True) or {}
    if "value" not in payload:
        return jsonify({"error": "No value provided"}), 400

    value = payload["value"]
    if TEST_SERVER:
        return jsonify({"tag": tag_name, "value": value, tag_name: value}), 200

    try:
        result = _run_plc_call(lambda comm: comm.write((tag_name, value)))
    except CommError:
        return jsonify({"error": "PLC communication error"}), 502

    _, error = _extract_value(result)
    if error:
        return jsonify({"error": error}), 400

    return jsonify({"tag": tag_name, "value": value, tag_name: value}), 200


if __name__ == "__main__":
    if TEST_SERVER:
        print("Running in test mode. No tag writing allowed.\n")
    app.run(
        debug=DEBUG_SERVER,
        use_reloader=False,
        threaded=True,
        port=SERVER_PORT,
        host="0.0.0.0",
    )
