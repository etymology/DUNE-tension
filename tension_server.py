from flask import Flask, request, jsonify
from pycomm3 import logix_driver

PLC_IP_ADDRESS = "192.168.140.13"
SERVER_PORT = 5000

app = Flask(__name__)


@app.route('/tags/<tag_name>', methods=['GET'])
def read_tag(tag_name):
    """ Endpoint for reading the value of a given tag. """
    value = comm.read(tag_name)
    if value is None:
        return jsonify({'error': 'Tag not found'}), 404
    return jsonify({tag_name: value})


@app.route('/tags/<tag_name>', methods=['POST'])
def write_tag(tag_name):
    """ Endpoint for writing a value to a given tag. """
    # Expecting JSON data with a 'value' key
    value = request.json.get('value')
    if value is None:
        return jsonify({'error': 'No value provided'}), 400

    comm.write(tag_name, value)
    return jsonify({tag_name: value})


with logix_driver.LogixDriver(PLC_IP_ADDRESS) as comm:
    app.run(debug=True, port=SERVER_PORT)
