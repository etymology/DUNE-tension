import sys
from pathlib import Path
import types
import time
import pytest

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# ----- Stub external dependencies -----
# Minimal numpy
numpy_stub = types.ModuleType("numpy")


def _avg(lst):
    return sum(lst) / len(lst) if lst else 0.0


def _std(lst):
    m = _avg(lst)
    return (sum((x - m) ** 2 for x in lst) / len(lst)) ** 0.5 if lst else 0.0


def _linspace(a, b, n):
    if n == 1:
        return [a]
    step = (b - a) / (n - 1)
    return [a + i * step for i in range(n)]


def _argmax(lst):
    return max(range(len(lst)), key=lambda i: lst[i])


numpy_stub.average = _avg
numpy_stub.std = _std
numpy_stub.linspace = _linspace
numpy_stub.argmax = _argmax
numpy_stub.bool_ = bool
numpy_stub.isscalar = lambda x: not isinstance(x, (list, tuple))
numpy_stub.savez = lambda *a, **k: None
sys.modules["numpy"] = numpy_stub

# Minimal pandas
pandas_stub = types.ModuleType("pandas")


class _Column(list):
    def tolist(self):
        return list(self)


class _DataFrame:
    def __init__(self, data):
        self._data = {k: _Column(v) for k, v in data.items()}

    def __getitem__(self, key):
        return self._data[key]

    @property
    def columns(self):
        return list(self._data.keys())


def _read_csv(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    lines = [line.strip() for line in p.read_text().splitlines() if line.strip()]
    headers = lines[0].split(",")
    cols = {h: [] for h in headers}
    for line in lines[1:]:
        for h, val in zip(headers, line.split(",")):
            try:
                cols[h].append(float(val))
            except ValueError:
                cols[h].append(val)
    return _DataFrame(cols)


pandas_stub.read_csv = _read_csv
sys.modules["pandas"] = pandas_stub

# geometry
geo_stub = types.ModuleType("geometry")
geo_stub.zone_lookup = lambda x: 1
geo_stub.length_lookup = lambda layer, wire, zone: 1.0
geo_stub.refine_position = lambda layer, side, zone, wire: (0.0, 0.0)
sys.modules["geometry"] = geo_stub

# tension_calculation
tc_stub = types.ModuleType("tension_calculation")
tc_stub.calculate_kde_max = lambda freqs: max(freqs)
tc_stub.tension_lookup = lambda length, frequency: frequency * 0.1
tc_stub.tension_pass = lambda tension, length: True
tc_stub.has_cluster = lambda data, key, n: data[:n] if len(data) >= n else []
tc_stub.tension_plausible = lambda t: True
sys.modules["tension_calculation"] = tc_stub

# audioProcessing
ap_stub = types.ModuleType("audioProcessing")
ap_stub.get_samplerate = lambda: None
ap_stub.spoof_audio_sample = lambda p: []
ap_stub.analyze_sample = lambda sample, sr, length: (sr, 1.0, 2.0, True)
sys.modules["audioProcessing"] = ap_stub

# plc_io
plc_stub = types.ModuleType("plc_io")
plc_stub.is_web_server_active = lambda: False
plc_stub.spoof_get_xy = lambda: (0.0, 0.0)
plc_stub.spoof_goto_xy = lambda x, y: True
plc_stub.spoof_wiggle = lambda m: None
plc_stub.increment = lambda x, y: None
sys.modules["plc_io"] = plc_stub

# data_cache
dc_stub = types.ModuleType("data_cache")
dc_stub.get_dataframe = lambda path: None
dc_stub.update_dataframe = lambda path, df: None
dc_stub.get_samples_dataframe = lambda path: None
dc_stub.update_samples_dataframe = lambda path, df: None
dc_stub.EXPECTED_COLUMNS = []
sys.modules["data_cache"] = dc_stub

# results stub
results_stub = types.ModuleType("results")


class _Dummy:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        freq = kwargs.get("frequency", 0.0)
        if "tension" not in kwargs:
            self.tension = freq * 0.1
        self.tension_pass = kwargs.get("tension_pass", True)
        self.wires = kwargs.get("wires", [])
        self.zone = 1
        self.wire_length = 1.0
        self.t_sigma = _std(self.wires) if self.wires else 0.0


results_stub.TensionResult = _Dummy
results_stub.RawSample = _Dummy
results_stub.EXPECTED_COLUMNS = []
sys.modules["results"] = results_stub

# tensiometer_functions
tf_stub = types.ModuleType("tensiometer_functions")


def _make_config(**kwargs):
    cfg = types.SimpleNamespace(**kwargs)
    cfg.data_path = f"{cfg.apa_name}_{cfg.layer}.csv"
    return cfg


tf_stub.make_config = _make_config
tf_stub.measure_list = lambda **k: None
tf_stub.get_xy_from_file = lambda cfg, num: (0.0, 0.0)
tf_stub.check_stop_event = lambda evt, msg="": False
sys.modules["tensiometer_functions"] = tf_stub

from dune_tension.tensiometer import Tensiometer, TensionResult


def test_generate_result_single_sample():
    t = Tensiometer(apa_name="APA", layer="X", side="A", samples_per_wire=1)
    sample = TensionResult(
        apa_name="APA",
        layer="X",
        side="A",
        wire_number=1,
        frequency=5.0,
        confidence=0.9,
        x=1.0,
        y=2.0,
        wires=[2.0],
    )
    result = t._generate_result([sample], wire_number=1, wire_x=1.5, wire_y=2.5)
    assert result.tension == 0.5
    assert result.frequency == 5.0
    assert result.tension_pass
    assert result.confidence == 0.9
    assert result.x == 1.0
    assert result.y == 2.0
    assert result.zone == 1
    assert result.wires == [0.5]
    assert result.t_sigma == 0.0


def test_generate_result_multi_sample():
    t = Tensiometer(apa_name="APA", layer="X", side="A", samples_per_wire=3)
    wires = [
        TensionResult(
            apa_name="APA",
            layer="X",
            side="A",
            wire_number=1,
            frequency=1.0,
            confidence=0.5,
            x=0.0,
            y=0.0,
            wires=[2.0],
        ),
        TensionResult(
            apa_name="APA",
            layer="X",
            side="A",
            wire_number=1,
            frequency=2.0,
            confidence=0.6,
            x=0.2,
            y=0.2,
            wires=[2.2],
        ),
        TensionResult(
            apa_name="APA",
            layer="X",
            side="A",
            wire_number=1,
            frequency=3.0,
            confidence=0.7,
            x=0.4,
            y=0.4,
            wires=[1.8],
        ),
    ]
    result = t._generate_result(wires, wire_number=1, wire_x=2.0, wire_y=3.0)
    assert result.frequency == 3.0  # max frequency via stub
    assert result.tension == pytest.approx(0.3)  # frequency * 0.1 via stub
    assert result.tension_pass
    assert result.confidence == pytest.approx(_avg([0.5, 0.6, 0.7]))
    assert result.t_sigma == pytest.approx(_std([0.1, 0.2, 0.3]))
    assert result.x == pytest.approx(_avg([0.0, 0.2, 0.4]), rel=1e-7)
    assert result.y == pytest.approx(_avg([0.0, 0.2, 0.4]), rel=1e-7)
    for got, exp in zip(result.wires, [0.1, 0.2, 0.3]):
        assert got == pytest.approx(exp)


def test_load_tension_summary(tmp_path):
    csv = tmp_path / "test.csv"
    csv.write_text("A,B\n1,2\n3,4\n")
    t = Tensiometer(apa_name="APA", layer="X", side="A")
    t.config.data_path = str(csv)
    a, b = t.load_tension_summary()
    assert a == [1.0, 3.0]
    assert b == [2.0, 4.0]


def test_load_tension_summary_missing(tmp_path):
    csv = tmp_path / "missing.csv"
    t = Tensiometer(apa_name="APA", layer="X", side="A")
    t.config.data_path = str(csv)
    msg, a, b = t.load_tension_summary()
    assert "File not found" in msg
    assert a == [] and b == []


def test_load_tension_summary_bad_columns(tmp_path):
    csv = tmp_path / "bad.csv"
    csv.write_text("C,D\n1,2\n")
    t = Tensiometer(apa_name="APA", layer="X", side="A")
    t.config.data_path = str(csv)
    msg, a, b = t.load_tension_summary()
    assert "missing" in msg
    assert a == [] and b == []


def test_wiggle_start_stop(monkeypatch):
    moves = []
    plc = sys.modules["plc_io"]
    monkeypatch.setattr(plc, "spoof_get_xy", lambda: (1.0, 2.0))
    monkeypatch.setattr(plc, "spoof_goto_xy", lambda x, y: moves.append((x, y)) or True)

    t = Tensiometer(apa_name="APA", layer="X", side="A")

    t.start_wiggle()
    time.sleep(0.05)
    t.stop_wiggle()

    assert moves[0] == (1.0, 3.0)
    assert moves[1] == (1.0, 1.0)
