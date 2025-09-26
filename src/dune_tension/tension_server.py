from flask import Flask, request, jsonify
from pycomm3 import logix_driver
from pycomm3.exceptions import CommError

PLC_IP_ADDRESS = "192.168.140.13"
SERVER_PORT = 5000

app = Flask(__name__)

TEST_SERVER = False


@app.route("/tags/<tag_name>", methods=["GET"])
def read_tag(tag_name):
    """Endpoint for reading the value of a given tag."""
    global comm
    for _ in range(2):
        try:
            value = comm.read(tag_name)
            break
        except CommError:
            try:
                comm.close()
            except Exception:
                pass
            try:
                comm.open()
            except Exception:
                pass
    else:
        return jsonify({"error": "PLC communication error"}), 500

    if value is None:
        return jsonify({"error": "Tag not found"}), 404
    return jsonify({tag_name: value})


@app.route("/tags/<tag_name>", methods=["POST"])
def write_tag(tag_name):
    """Endpoint for writing a value to a given tag."""
    # Expecting JSON data with a 'value' key
    value = request.json.get("value")
    if value is None:
        return jsonify({"error": "No value provided"}), 400

    global comm
    for _ in range(2):
        try:
            if not TEST_SERVER:
                comm.write((tag_name, value))
            break
        except CommError:
            try:
                comm.close()
            except Exception:
                pass
            try:
                comm.open()
            except Exception:
                pass
    else:
        return jsonify({"error": "PLC communication error"}), 500

    return jsonify({tag_name: value})


if __name__ == "__main__":
    if TEST_SERVER:
        print("Running in test mode. No tag writing allowed.\n")
    with logix_driver.LogixDriver(PLC_IP_ADDRESS) as comm:
        app.run(debug=True, port=SERVER_PORT, host="0.0.0.0")
