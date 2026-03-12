import importlib.util
from pathlib import Path
import sys
import types


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "dune_tension"
    / "gui"
    / "app.py"
)


def _load_app_module(monkeypatch):
    tk_stub = types.ModuleType("tkinter")
    tk_stub.Misc = object
    tk_stub.StringVar = object
    tk_stub.BooleanVar = object
    tk_stub.Button = object
    tk_stub.Canvas = object
    monkeypatch.setitem(sys.modules, "tkinter", tk_stub)

    dune_pkg = types.ModuleType("dune_tension")
    dune_pkg.__path__ = []
    gui_pkg = types.ModuleType("dune_tension.gui")
    gui_pkg.__path__ = []
    monkeypatch.setitem(sys.modules, "dune_tension", dune_pkg)
    monkeypatch.setitem(sys.modules, "dune_tension.gui", gui_pkg)

    config = types.ModuleType("dune_tension.config")
    config.MEASUREMENT_WIGGLE_CONFIG = types.SimpleNamespace(
        y_sigma_mm=1.0,
        focus_sigma_quarter_us=2.0,
    )
    monkeypatch.setitem(sys.modules, "dune_tension.config", config)

    actions = types.ModuleType("dune_tension.gui.actions")
    for name in [
        "adjust_focus_with_x_compensation",
        "calibrate_background_noise",
        "clear_range",
        "erase_distribution_outliers",
        "erase_outliers",
        "handle_close",
        "interrupt",
        "manual_goto",
        "manual_increment",
        "measure_auto",
        "measure_calibrate",
        "measure_condition",
        "measure_list_button",
        "monitor_tension_logs",
        "refresh_tension_logs",
        "set_manual_tension",
        "update_focus_command_indicator",
    ]:
        setattr(actions, name, lambda *args, **kwargs: None)
    monkeypatch.setitem(sys.modules, "dune_tension.gui.actions", actions)

    context = types.ModuleType("dune_tension.gui.context")
    context.GUIContext = object
    context.GUIWidgets = object
    context.create_context = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dune_tension.gui.context", context)

    live_plots = types.ModuleType("dune_tension.gui.live_plots")
    live_plots.LIVE_SUMMARY_FIGSIZE = (7.8, 3.6)
    live_plots.LIVE_WAVEFORM_FIGSIZE = (7.2, 4.6)
    live_plots.LivePlotManager = object
    monkeypatch.setitem(sys.modules, "dune_tension.gui.live_plots", live_plots)

    logging_panel = types.ModuleType("dune_tension.gui.logging_panel")
    logging_panel.configure_gui_logging = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dune_tension.gui.logging_panel", logging_panel)

    state = types.ModuleType("dune_tension.gui.state")
    state.load_state = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dune_tension.gui.state", state)

    tensiometer_functions = types.ModuleType("dune_tension.tensiometer_functions")
    tensiometer_functions.make_config = lambda **kwargs: types.SimpleNamespace(**kwargs)
    monkeypatch.setitem(
        sys.modules,
        "dune_tension.tensiometer_functions",
        tensiometer_functions,
    )

    module_name = "gui_app_under_test"
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


class _FakeRoot:
    def __init__(self) -> None:
        self.columnconfigure_calls = []
        self.minsize_args = None
        self.update_calls = 0

    def update_idletasks(self) -> None:
        self.update_calls += 1

    def columnconfigure(self, index, **kwargs) -> None:
        self.columnconfigure_calls.append((index, kwargs))

    def minsize(self, width, height) -> None:
        self.minsize_args = (width, height)


class _FakeFrame:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height

    def winfo_reqwidth(self) -> int:
        return self.width

    def winfo_reqheight(self) -> int:
        return self.height


def test_configure_root_minimum_size_reserves_space_for_both_columns(monkeypatch):
    app = _load_app_module(monkeypatch)
    root = _FakeRoot()
    main_frame = _FakeFrame(420, 700)
    side_frame = _FakeFrame(760, 680)

    app._configure_root_minimum_size(root, main_frame, side_frame)

    assert root.update_calls == 1
    assert root.columnconfigure_calls == [
        (0, {"weight": 0, "minsize": 420}),
        (1, {"weight": 1, "minsize": 780}),
    ]
    assert root.minsize_args == (1230, 720)
