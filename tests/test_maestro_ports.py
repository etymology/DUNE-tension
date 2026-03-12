import importlib
import sys
from pathlib import Path
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_controller_falls_back_across_ttyacm_ports(monkeypatch):
    serial_stub = types.ModuleType("serial")
    attempted_ports = []

    class SerialException(Exception):
        pass

    class FakeSerial:
        def __init__(self, port, *_, **__):
            attempted_ports.append(port)
            if port in {"/dev/ttyACM0", "/dev/ttyACM1"}:
                raise SerialException(port)
            self.port = port

        def close(self):
            pass

        def write(self, _payload):
            pass

        def read(self):
            return b"\x00"

    serial_stub.Serial = FakeSerial
    serial_stub.SerialException = SerialException
    monkeypatch.setitem(sys.modules, "serial", serial_stub)

    maestro = importlib.import_module("dune_tension.maestro")
    maestro = importlib.reload(maestro)

    controller = maestro.Controller()

    assert attempted_ports == ["/dev/ttyACM0", "/dev/ttyACM1", "/dev/ttyACM2"]
    assert controller.faulted is False
    assert controller.usb.port == "/dev/ttyACM2"
