# audioProcessing.py
import numpy as np

# Avoid using Tk based backends which can conflict with the Tkinter GUI when
# pitch detection runs in a background thread.
# Force matplotlib to use a non-interactive backend.  This avoids initialization
# of Tk when running in the GUI and keeps unit tests that stub out the
# ``matplotlib`` module from failing.  ``MPLBACKEND`` is respected by matplotlib
# if set before importing ``pyplot``.
import os

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
import crepe
from tension_calculation import (
    tension_lookup,
    tension_pass,
)

# Optional dependencies used for alternative pitch detection
try:
    import torch
    from pesto import load_model
except Exception:  # pragma: no cover - optional
    torch = None
    load_model = None

# Lazily initialized default pesto model
_PESTO_MODEL = None
import sounddevice as sd
import os
import random
import math


def load_audio_data(file_name):
    """Load audio data from a compressed .npz file."""
    try:
        with np.load(file_name) as data:
            audio_data = data["audio_data"]
        print(f"Audio data loaded from {file_name}")
        return audio_data
    except Exception as e:
        print(f"An error occurred while loading audio data: {e}")
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
        print("No confidence values available.")
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


def record_audio(duration, sample_rate, plot=False, normalize=False):
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
            plt.figure(figsize=(10, 4))
            plt.plot(audio_data)
            plt.title("Recorded Audio Waveform")
            plt.xlabel("Sample Index")
            plt.ylabel("Amplitude")
            plt.grid()
            plt.show()

        return audio_data, amplitude
    except Exception as e:
        print(f"An error occurred while recording audio: {e}")
        return None, 0.0
    finally:
        try:
            sd.stop()
        except Exception:
            pass


def analyze_sample(audio_sample, sample_rate, wire_length):
    frequency, confidence = get_pitch_crepe(audio_sample, sample_rate)
    tension = tension_lookup(length=wire_length, frequency=frequency)
    tension_ok = tension_pass(tension, wire_length)
    if not tension_ok and tension_pass(tension / 4, wire_length):
        tension /= 4
        frequency /= 2
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
            print(device_info)
            print(
                f"Using (hw:{sound_device_index},0),{device_info[sound_device_index]['name']}"
            )
        else:
            print("Couldn't find USB PnP Sound Device.")
            print(device_info)
            return None
        return sample_rate
    except Exception as e:
        print(f"Failed to initialize audio devices: {e}")
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
