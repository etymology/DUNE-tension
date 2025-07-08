import sys
import types
from pathlib import Path

# Ensure src on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

_STUB_MODULES = [
    "matplotlib",
    "matplotlib.pyplot",
    "numpy",
    "pandas",
    "seaborn",
    "data_cache",
    "results",
    "tensiometer_functions",
    "tension_calculation",
]
_saved = {}


def setup_module(module):
    for name in _STUB_MODULES:
        _saved[name] = sys.modules.get(name)
        mod = types.ModuleType(name)
        if name == "data_cache":
            mod.get_samples_dataframe = lambda path: None
        elif name == "tensiometer_functions":

            class DummyConfig:
                pass

            mod.TensiometerConfig = DummyConfig
        elif name == "tension_calculation":
            mod.calculate_kde_max = lambda x: 0
            mod.has_cluster = lambda a, b, c: []
        elif name == "pandas":

            class _Column(list):
                def tolist(self):
                    return list(self)


            class DummyDF:
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
                return DummyDF(cols)

            mod.DataFrame = DummyDF
            mod.read_csv = _read_csv
        elif name == "results":

            class Dummy:
                pass

            mod.TensionResult = Dummy
        sys.modules[name] = mod
    global analyze
    from dune_tension import analyze as analyze_mod

    analyze = analyze_mod


def teardown_module(module):
    for name in _STUB_MODULES:
        if _saved[name] is None:
            del sys.modules[name]
        else:
            sys.modules[name] = _saved[name]


def test_order_missing_wires_basic():
    missing = [8, 1, 5, 7]
    measured = [10]
    ordered = analyze._order_missing_wires(missing, measured)
    assert ordered == [8, 7, 5, 1]


def test_order_missing_no_measured():
    missing = [3, 1, 2]
    ordered = analyze._order_missing_wires(missing, [])
    assert ordered == [1, 2, 3]


def test_get_missing_wires_from_summary(tmp_path, monkeypatch):
    # Create fake summary CSV in temporary directory
    base = tmp_path / "data" / "tension_summaries"
    base.mkdir(parents=True)
    csv = base / "tension_summary_APA_X.csv"
    csv.write_text("wire_number,A,B\n1,1,\n2,,2\n")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(analyze, "get_expected_range", lambda _l: range(1, 3))

    cfg = analyze.TensiometerConfig()
    cfg.apa_name = "APA"
    cfg.layer = "X"

    missing = analyze.get_missing_wires(cfg)

    assert missing["A"] == [2]
    assert missing["B"] == [1]
