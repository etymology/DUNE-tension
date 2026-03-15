from pathlib import Path
import math
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import dune_tension.audio_runtime as audio_runtime_module
from dune_tension.audioProcessing import spoof_audio_sample


sample_audio = [0.1, 0.2, 0.3]


class Loader(dict):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _fake_load(_path):
    return Loader({"audio": np.asarray(sample_audio, dtype=np.float32)})


def test_spoof_audio_sample(tmp_path, monkeypatch):
    (tmp_path / "sample.npz").write_bytes(b"fake")
    monkeypatch.setattr(audio_runtime_module.np, "load", _fake_load)
    loaded = spoof_audio_sample(str(tmp_path))
    assert list(loaded) == sample_audio


def test_spoof_audio_sample_fallback(tmp_path):
    """Should return fallback data when no npz files exist."""
    loaded = spoof_audio_sample(str(tmp_path))
    expected = []
    sr = 41000
    freq = 80
    for i in range(sr):
        phase = 2 * math.pi * freq * i / sr
        expected.append(1.0 if math.sin(phase) >= 0 else -1.0)
    assert list(loaded) == expected
