from pathlib import Path
import sys
import types
import importlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

def test_main_delegates_to_gui_entrypoint(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setitem(
        sys.modules,
        "dune_tension.gui",
        types.SimpleNamespace(run_app=lambda: calls.append("run")),
    )
    sys.modules.pop("dune_tension.main", None)
    main = importlib.import_module("dune_tension.main")

    main.main()

    assert calls == ["run"]
