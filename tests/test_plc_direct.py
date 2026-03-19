from pathlib import Path
import sys
from types import SimpleNamespace
import types

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import dune_tension.plc_direct as plc_direct


class FakeCommError(Exception):
    pass


class FakeDriver:
    def __init__(self, address: str, *, read_results=None, write_results=None):
        self.address = address
        self.connected = False
        self.close_calls = 0
        self.read_results = list(read_results or [])
        self.write_results = list(write_results or [])

    def open(self) -> None:
        self.connected = True

    def close(self) -> None:
        self.connected = False
        self.close_calls += 1

    def read(self, _tag_name: str):
        result = self.read_results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def write(self, _payload):
        result = self.write_results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def _install_fake_pycomm3(monkeypatch, *, driver_plans):
    instances: list[FakeDriver] = []

    def make_driver(address: str) -> FakeDriver:
        plan = driver_plans[len(instances)]
        driver = FakeDriver(address, **plan)
        instances.append(driver)
        return driver

    monkeypatch.setattr(
        plc_direct,
        "_load_pycomm3",
        lambda: (types.SimpleNamespace(LogixDriver=make_driver), FakeCommError),
    )
    return instances


@pytest.fixture(autouse=True)
def _reset_direct_comm():
    plc_direct._close_comm()
    yield
    plc_direct._close_comm()


def test_read_tag_value_normalizes_pycomm3_result(monkeypatch):
    instances = _install_fake_pycomm3(
        monkeypatch,
        driver_plans=[
            {"read_results": [SimpleNamespace(value=42, error=None)]},
        ],
    )
    monkeypatch.setenv("PLC_IP_ADDRESS", "10.0.0.5")

    assert plc_direct.read_tag_value("STATE") == 42
    assert instances[0].address == "10.0.0.5"


def test_write_tag_value_raises_on_result_error(monkeypatch):
    _install_fake_pycomm3(
        monkeypatch,
        driver_plans=[
            {"write_results": [SimpleNamespace(value=None, error="bad write")]},
        ],
    )

    with pytest.raises(plc_direct.PLCTagWriteError, match="bad write"):
        plc_direct.write_tag_value("STATE", 1)


def test_run_plc_call_reconnects_after_comm_error(monkeypatch):
    instances = _install_fake_pycomm3(
        monkeypatch,
        driver_plans=[
            {"read_results": [FakeCommError("link dropped")]},
            {"read_results": [SimpleNamespace(value=7, error=None)]},
        ],
    )
    monkeypatch.setenv("PLC_COMM_RETRIES", "2")

    assert plc_direct.read_tag_value("STATE") == 7
    assert len(instances) == 2
    assert instances[0].close_calls == 1
