import sys
from pathlib import Path
import types
import time
from threading import Thread

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "dune_tension"))

serial_stub = types.ModuleType("serial")
class _Serial:
    def __init__(self, *a, **k):
        pass
serial_stub.Serial = _Serial
class _SerialException(Exception):
    pass
serial_stub.SerialException = _SerialException
sys.modules["serial"] = serial_stub

# Minimal tkinter stub to avoid display errors during import
tk_stub = types.ModuleType("tkinter")

class _Widget:
    def __init__(self, *a, **k):
        pass
    def grid(self, *a, **k):
        pass
    def insert(self, *a, **k):
        pass
    def set(self, *a, **k):
        pass
    def get(self):
        return ""
    def after(self, *a, **k):
        pass
    def mainloop(self):
        pass
    def title(self, *a, **k):
        pass

for cls in ["Tk", "Frame", "LabelFrame", "Label", "Entry", "OptionMenu", "Checkbutton", "Button", "Scale"]:
    setattr(tk_stub, cls, type(cls, (_Widget,), {}))

class _Var:
    def __init__(self, value=None):
        self._value = value
    def set(self, v):
        self._value = v
    def get(self):
        return self._value

tk_stub.StringVar = _Var
tk_stub.BooleanVar = _Var
tk_stub.HORIZONTAL = "HORIZONTAL"

messagebox_stub = types.ModuleType("tkinter.messagebox")
messagebox_stub.showerror = lambda *a, **k: None
sys.modules["tkinter"] = tk_stub
sys.modules["tkinter.messagebox"] = messagebox_stub

# Provide a minimal tensiometer stub so main can be imported without heavy deps
class _TensiometerStub:
    def __init__(self, *_, **__):
        pass

tensiometer_stub = types.ModuleType("tensiometer")
tensiometer_stub.Tensiometer = _TensiometerStub
sys.modules["tensiometer"] = tensiometer_stub

# Minimal tensiometer_functions stub with make_config
tfunc_stub = types.ModuleType("tensiometer_functions")
def _make_config(**kwargs):
    cfg = types.SimpleNamespace(**kwargs)
    cfg.data_path = f"{cfg.apa_name}_{cfg.layer}.csv"
    return cfg
tfunc_stub.make_config = _make_config
sys.modules["tensiometer_functions"] = tfunc_stub

import dune_tension.main as main
from dune_tension.maestro import DummyController


class DummyGetter:
    def __init__(self, value):
        self._value = value
    def get(self):
        return self._value


class DummyRoot:
    def __init__(self):
        self.after_args = None
    def after(self, delay, func):
        self.after_args = (delay, func)


class RecordController(DummyController):
    def __init__(self):
        super().__init__()
        self.calls = []
    def setTarget(self, chan, target):
        super().setTarget(chan, target)
        self.calls.append(target)

def test_create_tensiometer_flags(monkeypatch):
    called_args = {}
    class DummyTensiometer:
        def __init__(self, **kwargs):
            called_args.update(kwargs)
    monkeypatch.setattr(main, "Tensiometer", DummyTensiometer)
    monkeypatch.setattr(main, "entry_apa", DummyGetter("APA"))
    monkeypatch.setattr(main, "layer_var", DummyGetter("X"))
    monkeypatch.setattr(main, "side_var", DummyGetter("A"))
    monkeypatch.setattr(main, "flipped_var", DummyGetter(False))
    monkeypatch.setattr(main, "entry_samples", DummyGetter("2"))
    monkeypatch.setattr(main, "entry_confidence", DummyGetter("0.8"))
    monkeypatch.setattr(main.messagebox, "showerror", lambda *a, **k: None)
    monkeypatch.delenv("SPOOF_AUDIO", raising=False)
    monkeypatch.delenv("SPOOF_PLC", raising=False)
    main.create_tensiometer()
    assert called_args["spoof"] is False
    assert called_args["spoof_movement"] is False

    called_args.clear()
    monkeypatch.setenv("SPOOF_AUDIO", "1")
    main.create_tensiometer()
    assert called_args["spoof"] is True
    assert called_args["spoof_movement"] is True

    called_args.clear()
    monkeypatch.delenv("SPOOF_AUDIO")
    monkeypatch.setenv("SPOOF_PLC", "1")
    main.create_tensiometer()
    assert called_args["spoof"] is False
    assert called_args["spoof_movement"] is True


def test_servo_controller_run_loop():
    servo = RecordController()
    controller = main.ServoController(servo=servo)
    controller.dwell_time = 0
    controller.running.set()
    t = Thread(target=controller.run_loop)
    t.start()
    time.sleep(0.05)
    controller.running.clear()
    t.join(timeout=1)
    assert servo.calls[0] == 4000
    assert servo.calls[1] == 8000
    assert len(servo.calls) >= 2


def test_monitor_tension_logs(monkeypatch):
    updates = []
    class DummyConfig:
        apa_name = "APA"
        layer = "X"
        data_path = "dummy.csv"
    monkeypatch.setattr(main, "entry_apa", DummyGetter("APA"))
    monkeypatch.setattr(main, "layer_var", DummyGetter("X"))
    monkeypatch.setattr(main, "side_var", DummyGetter("A"))
    monkeypatch.setattr(main, "flipped_var", DummyGetter(False))
    monkeypatch.setattr(main, "root", DummyRoot())
    monkeypatch.setattr(main, "make_config", lambda **k: DummyConfig)
    monkeypatch.setattr(main.os.path, "getmtime", lambda p: 1)
    analyze_mod = types.ModuleType("analyze")
    analyze_mod.update_tension_logs = lambda conf: updates.append(conf)
    sys.modules["analyze"] = analyze_mod
    main.monitor_tension_logs.last_path = ""
    main.monitor_tension_logs.last_mtime = None
    main.monitor_tension_logs()
    assert updates and updates[-1] is DummyConfig
    assert main.monitor_tension_logs.last_mtime == 1
    main.monitor_tension_logs()
    assert len(updates) == 1
    monkeypatch.setattr(main.os.path, "getmtime", lambda p: 2)
    main.monitor_tension_logs()
    assert len(updates) == 2
