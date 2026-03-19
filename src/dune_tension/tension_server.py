import atexit
import os
from typing import Any

from flask import Flask, jsonify, request
from dune_tension.plc_direct import (
    PLCCommunicationError,
    PLCTagReadError,
    PLCTagWriteError,
    _close_comm,
    read_tag_value,
    write_tag_value,
)

SERVER_PORT = int(os.getenv("SERVER_PORT", "5000"))
TEST_SERVER = os.getenv("TEST_SERVER", "0").strip().lower() in {"1", "true", "yes"}
DEBUG_SERVER = os.getenv("TENSION_SERVER_DEBUG", "0").strip().lower() in {
    "1",
    "true",
    "yes",
}

app = Flask(__name__)


atexit.register(_close_comm)


@app.get("/health")
def health() -> tuple[Any, int]:
    """Lightweight health endpoint for client liveness checks."""
    return jsonify({"ok": True}), 200


@app.route("/tags/<tag_name>", methods=["GET"])
def read_tag(tag_name: str) -> tuple[Any, int]:
    """Read and return a PLC tag using a stable JSON response shape."""
    try:
        value = read_tag_value(tag_name)
    except PLCCommunicationError:
        return jsonify({"error": "PLC communication error"}), 502
    except PLCTagReadError as exc:
        return jsonify({"error": str(exc)}), 404

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
        write_tag_value(tag_name, value)
    except PLCCommunicationError:
        return jsonify({"error": "PLC communication error"}), 502
    except PLCTagWriteError as exc:
        return jsonify({"error": str(exc)}), 400

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
