from pathlib import Path
import sys
import types

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

sys.modules.setdefault("requests", types.ModuleType("requests"))

import dune_tension.plc_io as plc


@pytest.fixture(autouse=True)
def _reset_plc_mode_warnings():
    plc._WARNED_PLC_IO_MODE_VALUES.clear()


def test_read_tag_defaults_to_server_mode(monkeypatch):
    monkeypatch.delenv("PLC_IO_MODE", raising=False)
    monkeypatch.setattr(plc, "_read_tag_server", lambda tag_name: (tag_name, "server"))
    monkeypatch.setattr(
        plc,
        "_read_tag_direct",
        lambda _tag_name: pytest.fail("direct transport should not be used by default"),
    )

    assert plc.read_tag("STATE", timeout=0.01, retry_interval=0.0) == ("STATE", "server")


def test_direct_mode_routes_reads_and_writes_to_direct_transport(monkeypatch):
    monkeypatch.setenv("PLC_IO_MODE", "direct")
    monkeypatch.setattr(plc, "_read_tag_direct", lambda tag_name: (tag_name, "direct"))
    monkeypatch.setattr(
        plc,
        "_read_tag_server",
        lambda _tag_name: pytest.fail("server transport should not be used in direct mode"),
    )
    monkeypatch.setattr(
        plc,
        "_write_tag_direct",
        lambda tag_name, value: {"tag": tag_name, "value": value, "transport": "direct"},
    )
    monkeypatch.setattr(
        plc,
        "_write_tag_server",
        lambda _tag_name, _value: pytest.fail(
            "server transport should not be used in direct mode"
        ),
    )

    assert plc.read_tag("STATE", timeout=0.01, retry_interval=0.0) == ("STATE", "direct")
    assert plc.write_tag("STATE", 7) == {
        "tag": "STATE",
        "value": 7,
        "transport": "direct",
    }


def test_invalid_mode_falls_back_to_server_with_warning(monkeypatch, caplog):
    monkeypatch.setenv("PLC_IO_MODE", "banana")
    monkeypatch.setattr(plc, "_read_tag_server", lambda _tag_name: 123)
    monkeypatch.setattr(
        plc,
        "_read_tag_direct",
        lambda _tag_name: pytest.fail("direct transport should not be used for invalid mode"),
    )

    with caplog.at_level("WARNING"):
        value = plc.read_tag("STATE", timeout=0.01, retry_interval=0.0)

    assert value == 123
    assert "Invalid PLC_IO_MODE='banana'" in caplog.text


def test_is_plc_available_uses_direct_probe_in_direct_mode(monkeypatch):
    monkeypatch.setenv("PLC_IO_MODE", "direct")
    monkeypatch.setattr(plc, "is_direct_plc_available", lambda: True)
    monkeypatch.setattr(
        plc,
        "is_web_server_active",
        lambda: pytest.fail("server liveness check should not run in direct mode"),
    )

    assert plc.is_plc_available() is True


def test_is_plc_available_uses_server_probe_in_server_mode(monkeypatch):
    monkeypatch.setenv("PLC_IO_MODE", "server")
    monkeypatch.setattr(plc, "is_web_server_active", lambda: True)
    monkeypatch.setattr(
        plc,
        "is_direct_plc_available",
        lambda: pytest.fail("direct liveness check should not run in server mode"),
    )

    assert plc.is_plc_available() is True


class _FakeHttpResponse:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def test_write_tag_server_preserves_error_message(monkeypatch):
    monkeypatch.setattr(
        plc,
        "_request_with_retries",
        lambda *_args, **_kwargs: _FakeHttpResponse(
            400, {"error": "Tag STATE is read-only"}
        ),
    )

    assert plc._write_tag_server("STATE", 1) == {
        "error": "Tag STATE is read-only",
        "status_code": 400,
    }


def test_read_tag_server_preserves_error_message(monkeypatch):
    monkeypatch.setattr(
        plc,
        "_request_with_retries",
        lambda *_args, **_kwargs: _FakeHttpResponse(
            502, {"error": "PLC communication error"}
        ),
    )

    assert plc._read_tag_server("STATE") == {
        "error": "PLC communication error",
        "status_code": 502,
    }
