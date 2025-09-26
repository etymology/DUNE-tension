import sys
from pathlib import Path
import types
import math

# Insert src into path before importing
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Provide a minimal numpy stub so the module can be imported without the real package
sample_audio = [0.1, 0.2, 0.3]

numpy_stub = types.ModuleType("numpy")
numpy_stub.ndarray = object


def array(data):
    return list(data)


numpy_stub.array = array


class Loader(dict):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass


def load(path):
    return Loader({"audio": sample_audio})


numpy_stub.load = load
sys.modules.setdefault("numpy", numpy_stub)
sys.modules.setdefault("matplotlib", types.ModuleType("matplotlib"))
sys.modules.setdefault("matplotlib.pyplot", types.ModuleType("matplotlib.pyplot"))
sys.modules.setdefault("crepe", types.ModuleType("crepe"))
tc_module = types.ModuleType("tension_calculation")
tc_module.tension_lookup = lambda length, frequency: 0.0
tc_module.tension_pass = lambda t, length: True
sys.modules.setdefault("tension_calculation", tc_module)
sys.modules.setdefault("sounddevice", types.ModuleType("sounddevice"))

from dune_tension.audioProcessing import spoof_audio_sample


def test_spoof_audio_sample(tmp_path):
    (tmp_path / "sample.npz").write_bytes(b"fake")
    loaded = spoof_audio_sample(str(tmp_path))
    assert loaded == sample_audio


def test_spoof_audio_sample_fallback(tmp_path):
    """Should return fallback data when no npz files exist."""
    loaded = spoof_audio_sample(str(tmp_path))
    expected = []
    sr = 41000
    freq = 80
    for i in range(sr):
        phase = 2 * math.pi * freq * i / sr
        expected.append(1.0 if math.sin(phase) >= 0 else -1.0)
    assert loaded == expected
