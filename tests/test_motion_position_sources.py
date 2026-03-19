import importlib.util
from pathlib import Path
import sys
import types

import dune_tension.services as services_module

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
    monkeypatch.setattr(services_module, "_import_plc_module", lambda: plc)

    motion = MotionService.build(spoof_movement=False)

    assert motion.get_xy() == (30.0, 40.0)


def test_motion_service_uses_generic_plc_availability_when_present(monkeypatch):
    plc = types.SimpleNamespace(
        is_plc_available=lambda: True,
        is_web_server_active=lambda: False,
        get_xy=lambda: (10.0, 20.0),
        get_cached_xy=lambda: (30.0, 40.0),
        goto_xy=lambda *_args, **_kwargs: True,
        increment=lambda *_args, **_kwargs: None,
        reset_plc=lambda *_args, **_kwargs: None,
        set_speed=lambda *_args, **_kwargs: True,
        spoof_get_xy=lambda: (-1.0, -1.0),
        spoof_goto_xy=lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(services_module, "_import_plc_module", lambda: plc)

    motion = MotionService.build(spoof_movement=False)

    assert motion.get_xy() == (30.0, 40.0)


def test_motion_service_falls_back_to_spoof_when_plc_unavailable(monkeypatch):
    plc = types.SimpleNamespace(
        is_plc_available=lambda: False,
        get_xy=lambda: (10.0, 20.0),
        get_cached_xy=lambda: (30.0, 40.0),
        goto_xy=lambda *_args, **_kwargs: True,
        increment=lambda *_args, **_kwargs: None,
        reset_plc=lambda *_args, **_kwargs: None,
        set_speed=lambda *_args, **_kwargs: True,
        spoof_get_xy=lambda: (-1.0, -2.0),
        spoof_goto_xy=lambda *_args, **_kwargs: "spoofed",
    )
    monkeypatch.setattr(services_module, "_import_plc_module", lambda: plc)

    motion = MotionService.build(spoof_movement=False)

    assert motion.get_xy() == (-1.0, -2.0)
    assert motion.goto_xy(1.0, 2.0) == "spoofed"


def test_motion_service_spoof_flag_overrides_live_plc(monkeypatch):
    plc = types.SimpleNamespace(
        is_plc_available=lambda: True,
        get_xy=lambda: (10.0, 20.0),
        get_cached_xy=lambda: (30.0, 40.0),
        goto_xy=lambda *_args, **_kwargs: True,
        increment=lambda *_args, **_kwargs: None,
        reset_plc=lambda *_args, **_kwargs: None,
        set_speed=lambda *_args, **_kwargs: True,
        spoof_get_xy=lambda: (-3.0, -4.0),
        spoof_goto_xy=lambda *_args, **_kwargs: "spoofed",
    )
    monkeypatch.setattr(services_module, "_import_plc_module", lambda: plc)

    motion = MotionService.build(spoof_movement=True)

    assert motion.get_xy() == (-3.0, -4.0)
    assert motion.goto_xy(1.0, 2.0) == "spoofed"


def test_gui_context_uses_runtime_bundle_motion(monkeypatch):
    module_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "dune_tension"
        / "gui"
        / "context.py"
    )

    dune_pkg = types.ModuleType("dune_tension")
    dune_pkg.__path__ = []
    services = types.ModuleType("dune_tension.services")
    services.RuntimeBundle = object

    monkeypatch.setitem(sys.modules, "dune_tension", dune_pkg)
    monkeypatch.setitem(sys.modules, "dune_tension.services", services)

    module_name = "gui_context_under_test"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None
    context = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, context)
    spec.loader.exec_module(context)

    runtime_bundle = types.SimpleNamespace(
        servo_controller=object(),
        valve_controller=object(),
        motion=types.SimpleNamespace(
            get_xy=lambda: (30.0, 40.0),
            goto_xy=lambda *_args, **_kwargs: True,
        ),
        strum=lambda: None,
    )
    focus_command_var = object()
    estimated_time_var = object()

    gui_context = context.create_context(
        root=object(),
        widgets=object(),
        state_file="gui_state.json",
        runtime_bundle=runtime_bundle,
        focus_command_var=focus_command_var,
        estimated_time_var=estimated_time_var,
    )

    assert gui_context.runtime is runtime_bundle
    assert gui_context.get_xy() == (30.0, 40.0)
    assert gui_context.goto_xy is runtime_bundle.motion.goto_xy
    assert gui_context.focus_command_var is focus_command_var
    assert gui_context.estimated_time_var is estimated_time_var
