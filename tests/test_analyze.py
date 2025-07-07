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

            class DummyDF:
                pass

            mod.DataFrame = DummyDF
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
