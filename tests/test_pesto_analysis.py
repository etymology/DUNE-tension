from contextlib import nullcontext
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spectrum_analysis import pesto_analysis


class _FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value, dtype=np.float32)

    def to(self, dtype=None):
        return self

    def unsqueeze(self, axis):
        return _FakeTensor(np.expand_dims(self.value, axis=axis))

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class _FakeTorch:
    Tensor = _FakeTensor
    float32 = np.float32

    @staticmethod
    def from_numpy(value):
        return _FakeTensor(value)

    @staticmethod
    def inference_mode():
        return nullcontext()


def test_estimate_pitch_from_audio_uses_expected_frequency_mask(monkeypatch):
    captured = {}

    def fake_load_model(name, step_size, sampling_rate, streaming, max_batch_size):
        captured["model_name"] = name
        captured["step_size"] = step_size
        captured["sampling_rate"] = sampling_rate
        captured["streaming"] = streaming
        captured["max_batch_size"] = max_batch_size

        def fake_model(audio_tensor, sr, convert_to_freq, return_activations):
            captured["audio_shape"] = audio_tensor.value.shape
            captured["sr"] = sr
            captured["convert_to_freq"] = convert_to_freq
            captured["return_activations"] = return_activations
            return (
                _FakeTensor([[110.0, 420.0]]),
                _FakeTensor([[0.7, 0.9]]),
                _FakeTensor([[0.0, 0.0]]),
            )

        return fake_model

    monkeypatch.setattr(pesto_analysis, "torch", _FakeTorch)
    monkeypatch.setattr(pesto_analysis, "load_model", fake_load_model)
    monkeypatch.setattr(pesto_analysis, "_RUNTIME_DEPS_LOADED", True)
    monkeypatch.setattr(pesto_analysis, "_MODEL_CACHE", {})
    monkeypatch.setattr(pesto_analysis, "_resolve_step_size_ms", lambda *_args: 5.0)

    frequency, confidence = pesto_analysis.estimate_pitch_from_audio(
        np.zeros(16, dtype=np.float32),
        sample_rate=16000,
        expected_frequency=100.0,
    )

    assert captured["model_name"] == "mir-1k_g7"
    assert captured["step_size"] == 5.0
    assert captured["sampling_rate"] == 16000
    assert captured["streaming"] is False
    assert captured["max_batch_size"] == 1
    assert captured["audio_shape"] == (1, 16)
    assert captured["sr"] == 16000
    assert captured["convert_to_freq"] is True
    assert captured["return_activations"] is False
    assert np.isclose(frequency, 110.0)
    assert np.isclose(confidence, 0.7)


def test_estimate_pitch_from_audio_returns_nan_without_pesto(monkeypatch):
    monkeypatch.setattr(pesto_analysis, "torch", None)
    monkeypatch.setattr(pesto_analysis, "load_model", None)
    monkeypatch.setattr(pesto_analysis, "_RUNTIME_DEPS_LOADED", True)

    frequency, confidence = pesto_analysis.estimate_pitch_from_audio(
        np.zeros(8, dtype=np.float32),
        sample_rate=44100,
    )

    assert np.isnan(frequency)
    assert np.isnan(confidence)
