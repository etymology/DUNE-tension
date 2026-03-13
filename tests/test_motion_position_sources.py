import importlib.util
from pathlib import Path
import sys
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dune_tension.services import MotionService


def test_motion_service_prefers_backlash_aware_cached_xy(monkeypatch):
    plc = types.SimpleNamespace(
        is_web_server_active=lambda: True,
        get_xy=lambda: (10.0, 20.0),
        get_cached_xy=lambda: (30.0, 40.0),
        goto_xy=lambda *_args, **_kwargs: True,
        increment=lambda *_args, **_kwargs: None,
        reset_plc=lambda *_args, **_kwargs: None,
        set_speed=lambda *_args, **_kwargs: True,
    )
    monkeypatch.setitem(sys.modules, "plc_io", plc)

    motion = MotionService.build(spoof_movement=False)

    assert motion.get_xy() == (30.0, 40.0)


def test_gui_context_prefers_backlash_aware_cached_xy(monkeypatch):
    module_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "dune_tension"
        / "gui"
        / "context.py"
    )

    dune_pkg = types.ModuleType("dune_tension")
    dune_pkg.__path__ = []
    gui_pkg = types.ModuleType("dune_tension.gui")
    gui_pkg.__path__ = []

    maestro = types.ModuleType("dune_tension.maestro")

    class _DummyController:
        pass

    class _DummyServoController:
        def __init__(self, servo=None):
            self.servo = servo

    maestro.Controller = _DummyController
    maestro.DummyController = _DummyController
    maestro.ServoController = _DummyServoController

    valve_trigger = types.ModuleType("valve_trigger")
    valve_trigger.DeviceNotFoundError = RuntimeError
    valve_trigger.ValveController = type("ValveController", (), {})

    plc_io = types.ModuleType("dune_tension.plc_io")
    plc_io.get_xy = lambda: (10.0, 20.0)
    plc_io.get_cached_xy = lambda: (30.0, 40.0)
    plc_io.goto_xy = lambda *_args, **_kwargs: True
    plc_io.spoof_get_xy = lambda: (50.0, 60.0)
    plc_io.spoof_goto_xy = lambda *_args, **_kwargs: True

    monkeypatch.setitem(sys.modules, "dune_tension", dune_pkg)
    monkeypatch.setitem(sys.modules, "dune_tension.gui", gui_pkg)
    monkeypatch.setitem(sys.modules, "dune_tension.maestro", maestro)
    monkeypatch.setitem(sys.modules, "valve_trigger", valve_trigger)
    monkeypatch.setitem(sys.modules, "dune_tension.plc_io", plc_io)
    monkeypatch.delenv("SPOOF_PLC", raising=False)

    module_name = "gui_context_under_test"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None
    context = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, context)
    spec.loader.exec_module(context)

    get_xy, goto_xy = context._resolve_plc_functions()

    assert get_xy() == (30.0, 40.0)
    assert goto_xy is plc_io.goto_xy
