from __future__ import annotations

import types
import time

import numpy as np

import dune_tension.results as results_module
import dune_tension.tensiometer as tensiometer_module
from dune_tension.results import TensionResult
from dune_tension.tensiometer import Tensiometer


def _stub_motion_service():
    return types.SimpleNamespace(
        get_xy=lambda: (1.0, 2.0),
        goto_xy=lambda *_args, **_kwargs: True,
        increment=lambda *_args, **_kwargs: None,
        reset_plc=lambda *_args, **_kwargs: None,
        set_speed=lambda *_args, **_kwargs: None,
    )


def _stub_audio_service(sample_rate: int = 8000):
    return types.SimpleNamespace(
        samplerate=sample_rate,
        noise_threshold=0.0,
        record_audio=lambda *_args, **_kwargs: (np.zeros(1, dtype=float), 0.0),
    )


def _patch_result_physics(monkeypatch) -> None:
    monkeypatch.setattr(results_module, "zone_lookup", lambda _x: 1)
    monkeypatch.setattr(
        results_module,
        "length_lookup",
        lambda _layer, _wire, _zone, taped=False: 1.0,
    )
    monkeypatch.setattr(
        results_module,
        "wire_equation",
        lambda length, frequency: {"tension": 6.0, "frequency": frequency},
    )
    monkeypatch.setattr(results_module, "tension_pass", lambda _tension, _length: True)


def test_collect_samples_invokes_audio_sample_callback(monkeypatch) -> None:
    published: list[tuple[list[float], int, object | None]] = []
    audio_sample = np.array([0.1, -0.2, 0.3], dtype=float)
    analysis = types.SimpleNamespace(
        frequency=80.0,
        confidence=0.95,
        activation_map=np.ones((4, 3), dtype=np.float32),
        activation_freq_axis=np.array([50.0, 60.0, 70.0, 80.0], dtype=np.float32),
        frame_times=np.array([0.0, 0.005, 0.01], dtype=np.float32),
        predicted_frequencies=np.array([78.0, 80.0, 79.0], dtype=np.float32),
    )

    monkeypatch.setattr(
        tensiometer_module.MotionService,
        "build",
        lambda spoof_movement=False: _stub_motion_service(),
    )
    monkeypatch.setattr(
        tensiometer_module.AudioCaptureService,
        "build",
        lambda spoof=False: _stub_audio_service(),
    )
    monkeypatch.setattr(tensiometer_module, "acquire_audio", lambda **_kwargs: audio_sample)
    monkeypatch.setattr(
        tensiometer_module,
        "wire_equation",
        lambda *, length, frequency=None: {"frequency": 80.0, "tension": 6.0},
    )
    monkeypatch.setattr(tensiometer_module, "tension_plausible", lambda _tension: True)
    monkeypatch.setattr(
        tensiometer_module,
        "analyze_audio_with_pesto",
        lambda *_args, **_kwargs: analysis,
    )
    _patch_result_physics(monkeypatch)

    tensiometer = Tensiometer(
        apa_name="APA",
        layer="X",
        side="A",
        confidence_threshold=0.9,
        measuring_duration=0.2,
        audio_sample_callback=lambda sample, samplerate, payload: published.append(
            (sample.tolist(), samplerate, payload)
        ),
    )
    tensiometer.repository.append_sample = lambda _result: None
    tensiometer.strum_func = lambda: None
    tensiometer.focus_wiggle_func = lambda _delta: None

    samples = tensiometer._collect_samples(
        wire_number=1,
        length=1.0,
        start_time=time.time(),
        wire_y=2.0,
        wire_x=1.0,
    )

    assert samples is not None
    assert len(samples) == 1
    assert published == [([0.1, -0.2, 0.3], 8000, analysis)]


def test_goto_collect_wire_data_invokes_summary_refresh_callback(monkeypatch) -> None:
    refreshes = []
    saved_results = []

    monkeypatch.setattr(
        tensiometer_module.MotionService,
        "build",
        lambda spoof_movement=False: _stub_motion_service(),
    )
    monkeypatch.setattr(
        tensiometer_module.AudioCaptureService,
        "build",
        lambda spoof=False: _stub_audio_service(),
    )
    monkeypatch.setattr(tensiometer_module, "zone_lookup", lambda _x: 1)
    monkeypatch.setattr(
        tensiometer_module,
        "length_lookup",
        lambda _layer, _wire, _zone, taped=False: 1.0,
    )
    _patch_result_physics(monkeypatch)

    result = TensionResult(
        apa_name="APA",
        layer="X",
        side="A",
        wire_number=1,
        frequency=80.0,
        confidence=0.95,
        x=1.0,
        y=2.0,
    )

    tensiometer = Tensiometer(
        apa_name="APA",
        layer="X",
        side="A",
        summary_refresh_callback=refreshes.append,
    )
    tensiometer._collect_samples = lambda **_kwargs: [result]
    tensiometer.repository.append_result = lambda saved: saved_results.append(saved)

    measured = tensiometer.goto_collect_wire_data(wire_number=1, wire_x=1.0, wire_y=2.0)

    assert measured is result
    assert saved_results == [result]
    assert refreshes == [tensiometer.config]
