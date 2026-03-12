import importlib.util
from pathlib import Path
import sys
import threading
import time
import types


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "dune_tension"
    / "gui"
    / "actions.py"
)


def _load_actions_module(monkeypatch):
    dune_pkg = types.ModuleType("dune_tension")
    dune_pkg.__path__ = []
    gui_pkg = types.ModuleType("dune_tension.gui")
    gui_pkg.__path__ = []

    monkeypatch.setitem(sys.modules, "sounddevice", types.SimpleNamespace(stop=lambda: None))
    monkeypatch.setitem(sys.modules, "dune_tension", dune_pkg)
    monkeypatch.setitem(sys.modules, "dune_tension.gui", gui_pkg)

    data_cache = types.ModuleType("dune_tension.data_cache")
    data_cache.clear_wire_numbers = lambda *args, **kwargs: None
    data_cache.clear_wire_range = lambda *args, **kwargs: None
    data_cache.find_distribution_outliers = lambda *args, **kwargs: []
    data_cache.find_outliers = lambda *args, **kwargs: []
    data_cache.get_dataframe = lambda *args, **kwargs: None
    data_cache.update_dataframe = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dune_tension.data_cache", data_cache)

    results = types.ModuleType("dune_tension.results")
    results.EXPECTED_COLUMNS = []
    monkeypatch.setitem(sys.modules, "dune_tension.results", results)

    tensiometer = types.ModuleType("dune_tension.tensiometer")
    tensiometer.Tensiometer = object
    monkeypatch.setitem(sys.modules, "dune_tension.tensiometer", tensiometer)

    tensiometer_functions = types.ModuleType("dune_tension.tensiometer_functions")
    tensiometer_functions.make_config = lambda **kwargs: types.SimpleNamespace(**kwargs)
    monkeypatch.setitem(
        sys.modules,
        "dune_tension.tensiometer_functions",
        tensiometer_functions,
    )

    context = types.ModuleType("dune_tension.gui.context")
    context.GUIContext = object
    monkeypatch.setitem(sys.modules, "dune_tension.gui.context", context)

    state = types.ModuleType("dune_tension.gui.state")
    state.save_state = lambda _ctx: None
    monkeypatch.setitem(sys.modules, "dune_tension.gui.state", state)

    module_name = "gui_actions_under_test"
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def _wait_for(predicate, timeout=1.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def test_measurement_threads_are_serialized(monkeypatch):
    actions = _load_actions_module(monkeypatch)
    monkeypatch.setattr(actions, "_capture_worker_inputs", lambda _ctx: object())
    monkeypatch.setattr(actions, "save_state", lambda _ctx: None)

    calls = []
    started = threading.Event()
    release = threading.Event()

    @actions._run_in_thread(measurement=True)
    def fake_measurement(ctx, _inputs):
        calls.append(object())
        started.set()
        release.wait(timeout=1.0)

    ctx = types.SimpleNamespace(
        stop_event=threading.Event(),
        measurement_lock=threading.Lock(),
        measurement_active=False,
        active_measurement_name="",
    )

    fake_measurement(ctx)
    assert started.wait(timeout=1.0)

    fake_measurement(ctx)
    time.sleep(0.05)
    assert len(calls) == 1
    assert ctx.measurement_active is True

    release.set()
    assert _wait_for(lambda: ctx.measurement_active is False)
    assert ctx.active_measurement_name == ""

    fake_measurement(ctx)
    assert _wait_for(lambda: len(calls) == 2)


def test_erase_distribution_outliers_uses_bulk_detector(monkeypatch):
    actions = _load_actions_module(monkeypatch)

    class DummyGetter:
        def __init__(self, value):
            self._value = value

        def get(self):
            return self._value

    cfg = types.SimpleNamespace(
        data_path="db.sqlite",
        apa_name="APA",
        layer="G",
        side="A",
    )
    monkeypatch.setattr(actions, "_make_config_from_widgets", lambda _ctx: cfg)

    detector_calls = []
    clear_calls = []

    def fake_find(*args, **kwargs):
        detector_calls.append((args, kwargs))
        return [7, 9]

    def fake_clear(*args):
        clear_calls.append(args)

    monkeypatch.setattr(actions, "find_distribution_outliers", fake_find)
    monkeypatch.setattr(actions, "clear_wire_numbers", fake_clear)

    ctx = types.SimpleNamespace(
        widgets=types.SimpleNamespace(
            entry_confidence=DummyGetter("0.85"),
            entry_times_sigma=DummyGetter("2.5"),
        ),
        live_plot_manager=None,
    )

    actions.erase_distribution_outliers(ctx)

    assert detector_calls == [
        (
            ("db.sqlite", "APA", "G", "A"),
            {"times_sigma": 2.5, "confidence_threshold": 0.85},
        )
    ]
    assert clear_calls == [("db.sqlite", "APA", "G", "A", [7, 9])]


def test_measure_list_button_skips_already_measured_wires_when_enabled(monkeypatch):
    actions = _load_actions_module(monkeypatch)

    measured_wires = []

    class DummyTensiometer:
        def measure_list(self, wire_list, preserve_order=False):
            measured_wires.append((wire_list, preserve_order))

        def close(self):
            pass

    summaries = types.ModuleType("dune_tension.summaries")
    summaries.get_tension_series = lambda _config: {"A": {3: 5.8, 6: 6.1}}
    monkeypatch.setitem(sys.modules, "dune_tension.summaries", summaries)
    monkeypatch.setattr(actions, "create_tensiometer", lambda _ctx, _inputs: DummyTensiometer())
    monkeypatch.setattr(
        actions,
        "_make_config_from_inputs",
        lambda _inputs: types.SimpleNamespace(side="A"),
    )
    monkeypatch.setattr(actions, "_cleanup_after_measurement", lambda *_args, **_kwargs: None)

    inputs = types.SimpleNamespace(wire_list="3,5-7", skip_measured=True)

    actions.measure_list_button.__wrapped__(types.SimpleNamespace(), inputs)

    assert measured_wires == [([5, 7], False)]


def test_measure_list_button_keeps_requested_wires_when_skip_disabled(monkeypatch):
    actions = _load_actions_module(monkeypatch)

    measured_wires = []

    class DummyTensiometer:
        def measure_list(self, wire_list, preserve_order=False):
            measured_wires.append((wire_list, preserve_order))

        def close(self):
            pass

    monkeypatch.setattr(actions, "create_tensiometer", lambda _ctx, _inputs: DummyTensiometer())
    monkeypatch.setattr(actions, "_cleanup_after_measurement", lambda *_args, **_kwargs: None)

    inputs = types.SimpleNamespace(wire_list="3,5-7", skip_measured=False)

    actions.measure_list_button.__wrapped__(types.SimpleNamespace(), inputs)

    assert measured_wires == [([3, 5, 6, 7], False)]
