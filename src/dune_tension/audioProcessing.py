# audioProcessing.py
import numpy as np

# Avoid using Tk based backends which can conflict with the Tkinter GUI when
# pitch detection runs in a background thread.
# Force matplotlib to use a non-interactive backend.  This avoids initialization
# of Tk when running in the GUI and keeps unit tests that stub out the
# ``matplotlib`` module from failing.  ``MPLBACKEND`` is respected by matplotlib
# if set before importing ``pyplot``.
import os
from pathlib import Path
import shutil
import logging

os.environ.setdefault("MPLBACKEND", "Agg")
try:  # pragma: no cover - fallback for legacy test stubs
    from dune_tension.tension_calculation import tension_pass, wire_equation
except ImportError:  # pragma: no cover
    from tension_calculation import tension_pass, wire_equation
import sounddevice as sd
import random
import math

# Lazily initialized default pesto model
_PESTO_MODEL = None
_CREPE = None
_CREPE_UNAVAILABLE = False
_PESTO_RUNTIME = None
_PESTO_RUNTIME_UNAVAILABLE = False
LOGGER = logging.getLogger(__name__)


def _get_pyplot():
    import matplotlib.pyplot as plt

    return plt


def _get_crepe_module():
    global _CREPE, _CREPE_UNAVAILABLE

    if _CREPE is not None:
        return _CREPE
    if _CREPE_UNAVAILABLE:
        return None

    try:  # pragma: no cover - optional dependency
        import crepe as crepe_module  # type: ignore
    except Exception:  # pragma: no cover - dependency may be absent
        _CREPE_UNAVAILABLE = True
        return None

    _CREPE = crepe_module
    return _CREPE


def _get_pesto_runtime():
    global _PESTO_RUNTIME, _PESTO_RUNTIME_UNAVAILABLE

    if _PESTO_RUNTIME is not None:
        return _PESTO_RUNTIME
    if _PESTO_RUNTIME_UNAVAILABLE:
        return None, None

    try:  # pragma: no cover - optional dependency
        import torch as torch_module
        from pesto import load_model as pesto_load_model
    except Exception:  # pragma: no cover - optional dependency may be absent
        _PESTO_RUNTIME_UNAVAILABLE = True
        return None, None

    _PESTO_RUNTIME = (torch_module, pesto_load_model)
    return _PESTO_RUNTIME


def load_audio_data(file_name):
    """Load audio data from a compressed .npz file."""
    try:
        with np.load(file_name) as data:
            audio_data = data["audio_data"]
        LOGGER.info("Audio data loaded from %s", file_name)
        return audio_data
    except Exception as e:
        LOGGER.warning("An error occurred while loading audio data: %s", e)
        return None


def get_pitch_autocorrelation(
    audio_data, samplerate, freq_low=20, freq_high=2000, show_plots=False
):
    """
    Analyzes an audio signal to find the dominant frequency using autocorrelation.

    Parameters:==
    - audio_data (np.ndarray): The audio signal data as a numpy array.
    - samplerate (int): The sample rate of the audio signal.
    - freq_low (int): The lower boundary of the frequency range to search (default 10 Hz).
    - freq_high (int): The higher boundary of the frequency range to search (default MAX_FREQUENCY Hz).

    Returns:
    - tuple: (dominant_frequency, confidence) where:
        - dominant_frequency is the detected frequency in Hz.
        - confidence is a measure of the amplitude of the autocorrelation peak relative to others.
    """
    audio_data = audio_data - np.mean(audio_data)

    # Compute the autocorrelation of the signal
    autocorr = np.correlate(audio_data, audio_data, mode="full")
    autocorr = autocorr[len(autocorr) // 2 :]  # Keep only the second half

    # Determine the maximum lag we consider by the highest frequency of interest
    min_lag = int(samplerate // freq_high)
    max_lag = int(samplerate // freq_low)
    autocorr = autocorr[min_lag : max_lag + 1]

    # Find the first peak
    # This simplistic peak finding assumes the first peak is the fundamental frequency
    peak_lag = np.argmax(autocorr) + min_lag

    # Calculate the dominant frequency
    dominant_frequency = samplerate / peak_lag

    # Confidence calculation (peak height relative to the max of the autocorrelation values)
    confidence = abs(autocorr[peak_lag - min_lag] / np.max(autocorr))

    # Plotting the autocorrelation function
    lags = np.arange(min_lag, max_lag + 1)
    if show_plots:
        plt = _get_pyplot()
        plt.figure(figsize=(12, 6))
        plt.plot(lags / samplerate, autocorr)
        plt.axvline(
            peak_lag / samplerate,
            color="r",
            linestyle="--",
            label=f"Dominant Frequency: {dominant_frequency:.2f} Hz",
        )
        plt.xlabel("Lag [s]")
        plt.ylabel("Autocorrelation")
        plt.title("Autocorrelation Function")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return dominant_frequency, confidence


def spectral_flatness(magnitude: np.ndarray) -> float:
    """Calculate the spectral flatness of the magnitude spectrum."""
    geometric_mean = np.exp(
        np.mean(np.log(magnitude + 1e-10))
    )  # Adding a small constant to avoid log(0)
    arithmetic_mean = np.mean(magnitude)
    return geometric_mean / arithmetic_mean


def get_pitch_naive_fft(
    audio_data: np.ndarray, samplerate: int, show_plots=False
) -> tuple[float, float]:
    """Estimate the pitch of the audio data using FFT and return the fundamental frequency f0 and a confidence based on spectral flatness."""

    # Compute the FFT of the audio data
    fft_spectrum = np.fft.rfft(audio_data)
    magnitude = np.abs(fft_spectrum)
    freqs = np.fft.rfftfreq(len(audio_data), d=1 / samplerate)

    # Consider only frequencies below MAX_FREQUENCY Hz
    valid_indices = freqs < 8000
    if not np.any(valid_indices):
        return 0.0, 0.0

    # Find the indices of the highest peaks in the magnitude spectrum
    valid_magnitudes = magnitude[valid_indices]
    valid_freqs = freqs[valid_indices]

    peak_indices = np.argpartition(valid_magnitudes, -10)[-10:]
    top_peaks = peak_indices[np.argsort(valid_magnitudes[peak_indices])[::-1]]

    # Get the frequencies of the highest peaks
    top_frequencies = valid_freqs[top_peaks]

    # Check for a fundamental frequency f0 such that other peaks are approximately multiples of f0
    f0 = top_frequencies[0]
    for _, candidate_f0 in enumerate(top_frequencies):
        multiples_found = False
        for f in top_frequencies:
            if f != candidate_f0:
                ratio = f / candidate_f0
                if np.abs(ratio - np.round(ratio)) <= 0.05:
                    multiples_found = True
                    break
        if multiples_found:
            f0 = candidate_f0
            break

    if show_plots:
        plt = _get_pyplot()
        # Plot the time-domain audio data
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(np.arange(len(audio_data)) / samplerate, audio_data)
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.title("Time-Domain Audio Data")

        # Plot the frequency-domain magnitude spectrum
        plt.subplot(2, 1, 2)
        plt.plot(valid_freqs, valid_magnitudes)
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Magnitude")
        plt.title("Frequency-Domain Magnitude Spectrum")
        plt.xscale("log")

        # Plot a vertical red line at the fundamental frequency f0
        plt.axvline(
            f0, color="r", linestyle="--", label=f"Fundamental Frequency: {f0:.2f} Hz"
        )
        plt.legend()

        plt.tight_layout()
        plt.show()

    confidence = 1.0 - spectral_flatness(valid_magnitudes)
    return f0, confidence  # Return the fundamental frequency and the confidence


def get_pitch_crepe(
    audio_data: np.ndarray, samplerate, model_capacity="tiny"
) -> tuple[float, float]:
    """Extract the pitch and confidence from the audio data using CREPE."""
    crepe = _get_crepe_module()
    if crepe is None:
        raise RuntimeError("crepe is not installed")

    _, frequencies, confidence, _ = crepe.predict(
        audio_data,
        samplerate,
        model_capacity=model_capacity,
        viterbi=False,
        verbose=0,
        step_size=50,
    )

    # Directly find the index of the maximum confidence
    if len(confidence) > 0:
        max_conf_idx = np.argmax(confidence)
        max_frequency = frequencies[max_conf_idx]
        max_confidence = confidence[max_conf_idx]
    else:
        # Handle the case where no confidence values are available
        LOGGER.warning("No confidence values available.")
        max_frequency = 0.0
        max_confidence = 0.0

    return max_frequency, max_confidence


def get_pitch_pesto(
    audio_data: np.ndarray,
    samplerate: int,
    model=None,
) -> tuple[float, float]:
    """Extract pitch and confidence using a Pesto pitch model.

    Parameters
    ----------
    audio_data:
        Array containing the audio waveform.  The data is converted to
        ``float32`` before being passed to the model.
    samplerate:
        The sample rate of ``audio_data``.  If ``model`` is ``None`` a default
        streaming model is created for this sample rate.
    model:
        Optional pre-loaded Pesto model.  When omitted, a default model is
        lazily instantiated using :func:`pesto.load_model`.
    """
    torch, load_model = _get_pesto_runtime()
    if torch is None or load_model is None:
        raise RuntimeError("pesto and torch are required for get_pitch_pesto")

    global _PESTO_MODEL
    if model is None:
        if _PESTO_MODEL is None:
            _PESTO_MODEL = load_model(
                "mir-1k_g7",
                step_size=5.0,
                sampling_rate=samplerate,
                streaming=False,
                max_batch_size=1,
            )
        model = _PESTO_MODEL

    buffer = np.asarray(audio_data, dtype=np.float32)
    if buffer.ndim == 1:
        buffer = buffer[None, :]
    elif buffer.ndim == 2 and buffer.shape[0] != 1:
        buffer = buffer.T

    buffer_tensor = torch.tensor(buffer, dtype=torch.float32)

    pitch, conf, _ = model(
        buffer_tensor, return_activations=False, convert_to_freq=True
    )

    pitch_val = pitch.mean().item() if pitch.numel() > 0 else 0.0
    conf_val = conf.mean().item() if conf.numel() > 0 else 0.1

    if not torch.isfinite(pitch.mean()):
        pitch_val = 0.0
    if not torch.isfinite(conf.mean()):
        conf_val = 0.1

    return pitch_val, conf_val


def record_audio(duration, sample_rate, plot=False, normalize=True):
    """Record audio and return the raw sample and its average amplitude."""
    try:
        audio_data = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype="float64",
        )
        sd.wait()  # Wait until recording is finished
        audio_data = audio_data.flatten()  # Flatten the audio data to a 1D array

        amplitude = float(np.mean(np.abs(audio_data)))

        if normalize:
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data = audio_data / max_val

        if plot:
            plt = _get_pyplot()
            plt.figure(figsize=(10, 4))
            plt.plot(audio_data)
            plt.title("Recorded Audio Waveform")
            plt.xlabel("Sample Index")
            plt.ylabel("Amplitude")
            plt.grid()
            plt.show()

        return audio_data, amplitude
    except Exception as e:
        LOGGER.warning("An error occurred while recording audio: %s", e)
        return None, 0.0
    finally:
        try:
            sd.stop()
        except Exception:
            pass

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RUNTIME_NOISE_DIR = Path(
    os.environ.get("DUNE_TENSION_NOISE_DIR", _REPO_ROOT / "data" / "noise_filters")
)
_NOISE_FILTER_PATH = Path(
    os.environ.get(
        "DUNE_TENSION_NOISE_FILTER_PATH",
        _RUNTIME_NOISE_DIR / "noise_filter.npz",
    )
)
_LEGACY_NOISE_FILTER_PATH = Path(__file__).resolve().with_name("noise_filter.npz")
_noise_filter: dict | None = None
_noise_threshold: float = 0.0


def _load_noise_filter() -> None:
    """Load the saved noise filter if available."""
    global _noise_filter, _noise_threshold
    if _noise_filter is not None:
        return

    for path in (_NOISE_FILTER_PATH, _LEGACY_NOISE_FILTER_PATH):
        if not path.exists():
            continue
        try:
            data = np.load(path)
            _noise_filter = {
                "sample_rate": int(data["sample_rate"]),
                "magnitude": data["magnitude"],
            }
            if "threshold" in data:
                _noise_threshold = float(data["threshold"])

            if (
                path == _LEGACY_NOISE_FILTER_PATH
                and _NOISE_FILTER_PATH != _LEGACY_NOISE_FILTER_PATH
                and not _NOISE_FILTER_PATH.exists()
            ):
                try:
                    _NOISE_FILTER_PATH.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, _NOISE_FILTER_PATH)
                except Exception:
                    pass
            return
        except Exception as exc:  # pragma: no cover - loading is optional
            LOGGER.warning("Failed to load noise filter from %s: %s", path, exc)
            _noise_filter = None


def get_noise_threshold() -> float:
    """Return the calibrated noise amplitude threshold."""
    _load_noise_filter()
    return _noise_threshold


def calibrate_background_noise(sample_rate: int, duration: float = 1.0) -> None:
    """Record ``duration`` seconds of background noise and create a spectral filter."""

    noise_sample, amp = record_audio(duration, sample_rate, plot=False, normalize=True)
    if noise_sample is not None:
        noise_fft = np.fft.rfft(noise_sample)
        noise_mag = np.abs(noise_fft)
        global _noise_filter, _noise_threshold
        _noise_filter = {
            "sample_rate": sample_rate,
            "magnitude": noise_mag,
        }
        _noise_threshold = np.mean(
            np.abs(noise_sample)
        )  # Set threshold to twice the mean amplitude
        try:
            _NOISE_FILTER_PATH.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                _NOISE_FILTER_PATH,
                sample_rate=sample_rate,
                magnitude=noise_mag,
                threshold=_noise_threshold,
            )
            LOGGER.info("Saved noise filter to %s", _NOISE_FILTER_PATH)
        except Exception as exc:  # pragma: no cover - saving is optional
            LOGGER.warning("Failed to save noise filter: %s", exc)


def record_audio_filtered(duration, sample_rate, plot=False, normalize=True):
    """Record audio and apply the calibrated stationary-noise filter if available."""

    _load_noise_filter()
    audio, amp = record_audio(duration, sample_rate, plot=plot, normalize=normalize)
    if audio is None:
        return None, 0.0

    if _noise_filter is not None:
        noise_sr = _noise_filter["sample_rate"]
        noise_mag = _noise_filter["magnitude"]

        fft = np.fft.rfft(audio)
        mag = np.abs(fft)
        phase = np.angle(fft)

        if noise_sr != sample_rate or len(noise_mag) != len(mag):
            x_old = np.linspace(0, sample_rate / 2, len(noise_mag))
            x_new = np.linspace(0, sample_rate / 2, len(mag))
            noise_mag = np.interp(x_new, x_old, noise_mag)

        if len(noise_mag) < len(mag):
            noise_mag = np.pad(noise_mag, (0, len(mag) - len(noise_mag)), mode="edge")
        else:
            noise_mag = noise_mag[: len(mag)]

        filtered_mag = np.maximum(0.0, mag - noise_mag)
        filtered_fft = filtered_mag * np.exp(1j * phase)
        audio = np.fft.irfft(filtered_fft, n=len(audio))
        if normalize:
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val
        amp = float(np.mean(np.abs(audio)))
    return audio, amp


def analyze_sample(audio_sample, sample_rate, wire_length):
    frequency, confidence = get_pitch_pesto(audio_sample, sample_rate)
    tension = wire_equation(length=wire_length, frequency=frequency)["tension"]
    tension_ok = tension_pass(tension, wire_length)
    if not tension_ok:
        for i in [2, 3, 4]:
            if tension_pass(tension / i**2, wire_length):
                tension /= i**2
                frequency /= i
                tension_ok = True
    return frequency, confidence, tension, tension_ok


def get_samplerate():
    try:
        device_info = sd.query_devices()
        sound_device_index = next(
            (index for index, d in enumerate(device_info) if "default" in d["name"]),
            None,
        )
        if sound_device_index is not None:
            sample_rate = device_info[sound_device_index]["default_samplerate"]
            LOGGER.debug("Available audio devices: %s", device_info)
            LOGGER.info(
                "Using (hw:%s,0),%s",
                sound_device_index,
                device_info[sound_device_index]["name"],
            )
        else:
            LOGGER.warning("Couldn't find USB PnP Sound Device.")
            LOGGER.debug("Available audio devices: %s", device_info)
            return None
        return sample_rate
    except Exception as e:
        LOGGER.error("Failed to initialize audio devices: %s", e)
        exit(1)


def spoof_audio_sample(npz_dir: str) -> np.ndarray:
    """
    Return an audio sample for spoofing.

    The function attempts to load a random ``.npz`` file from ``npz_dir`` and
    return the ``"audio"`` array inside.  If no valid audio sample can be
    loaded, a built in fallback sample containing a synthesized 80 Hz square
    wave sampled at ``41000`` Hz is returned instead.

    Parameters:
        npz_dir (str): Path to the directory containing ``.npz`` files.

    Returns:
        np.ndarray: The loaded audio array.
    """

    npz_files = [f for f in os.listdir(npz_dir) if f.endswith(".npz")]
    file_path = None
    if npz_files:
        file_path = os.path.join(npz_dir, random.choice(npz_files))

    data = None
    if file_path:
        try:
            with np.load(file_path) as loaded:
                if "audio" in loaded:
                    data = loaded["audio"]
        except Exception:
            data = None

    if data is None:
        # Generate a one second 80 Hz square wave at 41 kHz
        sample_rate = 41000
        freq = 80
        total_samples = sample_rate
        wave = []
        for i in range(total_samples):
            phase = 2 * math.pi * freq * i / sample_rate
            wave.append(1.0 if math.sin(phase) >= 0 else -1.0)
        data = np.array(wave)

    return data
