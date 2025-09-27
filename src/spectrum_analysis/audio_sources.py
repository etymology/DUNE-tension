"""Audio source abstractions used by the spectrogram visualizer."""
from __future__ import annotations

import queue
import threading
from typing import Optional

import numpy as np

try:  # Optional dependency
    import sounddevice as sd
except Exception:  # pragma: no cover - optional dependency may be absent in CI
    sd = None  # type: ignore[assignment]


class AudioSource:
    """Abstract audio stream interface."""

    def start(self) -> None:  # pragma: no cover - interface method
        raise NotImplementedError

    def read(self) -> np.ndarray:  # pragma: no cover - interface method
        raise NotImplementedError

    def stop(self) -> None:  # pragma: no cover - interface method
        raise NotImplementedError


class MicSource(AudioSource):
    """Audio source backed by the default system microphone."""

    def __init__(self, samplerate: int, hop: int, device: Optional[str] = None) -> None:
        if sd is None:
            raise RuntimeError("sounddevice is not available. Install it or use --demo.")

        self.samplerate = samplerate
        self.hop = hop
        self.device = device
        self.q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=64)
        self.stream = None
        self._stopped = threading.Event()

    def _callback(self, indata, frames, time_info, status):  # pragma: no cover - sounddevice callback
        if indata.ndim == 2 and indata.shape[1] > 1:
            mono = indata.mean(axis=1).copy()
        else:
            mono = indata[:, 0].copy() if indata.ndim == 2 else indata.copy()
        try:
            self.q.put_nowait(mono)
        except queue.Full:
            pass

    def start(self) -> None:
        if sd is None:  # pragma: no cover - defensive
            raise RuntimeError("sounddevice is not available.")
        self._stopped.clear()
        self.stream = sd.InputStream(
            channels=1,
            samplerate=self.samplerate,
            blocksize=self.hop,
            device=self.device,
            callback=self._callback,
            dtype="float32",
        )
        self.stream.start()

    def read(self) -> np.ndarray:
        while not self._stopped.is_set():
            try:
                return self.q.get(timeout=0.1)
            except queue.Empty:
                continue
        return np.array([], dtype=np.float32)

    def stop(self) -> None:
        self._stopped.set()
        if self.stream is not None:
            try:  # pragma: no cover - depends on audio backend
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass


class DemoSource(AudioSource):
    """Synthetic audio source used when no microphone is available."""

    def __init__(self, samplerate: int, hop: int) -> None:
        self.samplerate = samplerate
        self.hop = hop
        self.t = 0

    def start(self) -> None:
        pass

    def read(self) -> np.ndarray:
        n = self.hop
        sr = self.samplerate
        t = (self.t + np.arange(n)) / sr
        chirp = np.sin(2 * np.pi * (100 + (t * 0.5e3)) * t) * 0.4
        tone1 = 0.25 * np.sin(2 * np.pi * 440 * t)
        tone2 = 0.2 * np.sin(2 * np.pi * 880 * t + 0.3)
        noise = 0.02 * np.random.randn(n)
        y = chirp + tone1 + tone2 + noise
        self.t += n
        y = np.tanh(1.5 * y)
        return y.astype(np.float32)

    def stop(self) -> None:
        pass


__all__ = ["AudioSource", "MicSource", "DemoSource", "sd"]
