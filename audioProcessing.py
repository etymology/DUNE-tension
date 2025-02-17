# audioProcessing.py
import numpy as np
import matplotlib.pyplot as plt
import crepe
import soundfile as sf

def save_wav(audio_sample: np.ndarray, sample_rate: int, filename: str):
    # Save the audio sample to a WAV file
    sf.write(filename, audio_sample, int(sample_rate))


def load_wav(filename: str):
    # Load the WAV file
    audio_sample, sample_rate = sf.read(filename)
    return audio_sample, sample_rate


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

    Parameters:
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
