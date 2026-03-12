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
                _FakeTensor([[660.0, 2520.0]]),
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
    assert captured["sampling_rate"] == 96000
    assert captured["streaming"] is False
    assert captured["max_batch_size"] == 1
    assert captured["audio_shape"] == (1, 16)
    assert captured["sr"] == 96000
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


def test_analyze_audio_with_pesto_returns_activation_map(monkeypatch):
    def fake_load_model(name, step_size, sampling_rate, streaming, max_batch_size):
        class _FakeModel:
            bins_per_semitone = 2
            preprocessor = type("Preprocessor", (), {"hcqt_kwargs": {"fmin": 55.0}})()

            def __call__(self, audio_tensor, sr, convert_to_freq, return_activations):
                assert audio_tensor.value.shape == (1, 16)
                assert sr == 16000
                assert convert_to_freq is True
                assert return_activations is True
                return (
                    _FakeTensor([[110.0, 120.0]]),
                    _FakeTensor([[0.7, 0.9]]),
                    _FakeTensor([[0.0, 0.0]]),
                    _FakeTensor([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]]),
                )

        return _FakeModel()

    monkeypatch.setattr(pesto_analysis, "torch", _FakeTorch)
    monkeypatch.setattr(pesto_analysis, "load_model", fake_load_model)
    monkeypatch.setattr(pesto_analysis, "_RUNTIME_DEPS_LOADED", True)
    monkeypatch.setattr(pesto_analysis, "_MODEL_CACHE", {})
    monkeypatch.setattr(pesto_analysis, "_resolve_step_size_ms", lambda *_args: 5.0)

    result = pesto_analysis.analyze_audio_with_pesto(
        np.zeros(16, dtype=np.float32),
        sample_rate=16000,
        include_activations=True,
    )

    assert np.isclose(result.frequency, 115.625)
    assert np.isclose(result.confidence, 0.8)
    assert result.expected_frequency is None
    assert result.activation_map is not None
    assert result.activation_map.shape == (3, 2)
    assert result.activation_freq_axis is not None
    assert result.activation_freq_axis.shape == (3,)
    assert np.all(np.diff(result.activation_freq_axis) > 0)


def test_analyze_audio_with_pesto_reverses_sr_augmentation(monkeypatch):
    activation = np.arange(60, dtype=np.float32).reshape(2, 30)

    def fake_load_model(name, step_size, sampling_rate, streaming, max_batch_size):
        class _FakeModel:
            bins_per_semitone = 2
            preprocessor = type("Preprocessor", (), {"hcqt_kwargs": {"fmin": 55.0}})()

            def __call__(self, _audio_tensor, sr, convert_to_freq, return_activations):
                assert sr == 32000
                assert convert_to_freq is True
                assert return_activations is True
                return (
                    _FakeTensor([[220.0, 240.0]]),
                    _FakeTensor([[0.5, 0.75]]),
                    _FakeTensor([[0.0, 0.0]]),
                    _FakeTensor([activation]),
                )

        return _FakeModel()

    monkeypatch.setattr(pesto_analysis, "torch", _FakeTorch)
    monkeypatch.setattr(pesto_analysis, "load_model", fake_load_model)
    monkeypatch.setattr(pesto_analysis, "_RUNTIME_DEPS_LOADED", True)
    monkeypatch.setattr(pesto_analysis, "_MODEL_CACHE", {})
    monkeypatch.setattr(pesto_analysis, "_resolve_step_size_ms", lambda *_args: 5.0)

    result = pesto_analysis.analyze_audio_with_pesto(
        np.zeros(16, dtype=np.float32),
        sample_rate=16000,
        expected_frequency=300.0,
        include_activations=True,
    )

    assert np.isclose(result.frequency, 116.0)
    assert np.isclose(result.confidence, 0.625)
    assert np.isclose(result.expected_frequency, 300.0)
    assert np.allclose(result.frame_times, np.array([0.0, 0.01], dtype=np.float32))
    assert np.allclose(
        result.predicted_frequencies,
        np.array([110.0, 120.0], dtype=np.float32),
    )
    assert result.activation_map is not None
    assert result.activation_map.shape == (30, 2)
    assert np.allclose(result.activation_map[:6, 0], np.array([24, 25, 26, 27, 28, 29], dtype=np.float32))
    assert np.allclose(result.activation_map[:6, 1], np.array([54, 55, 56, 57, 58, 59], dtype=np.float32))
    assert np.allclose(result.activation_map[6:, :], 0.0)
