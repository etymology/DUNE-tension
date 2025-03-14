# audioProcessing.py
import numpy as np
import matplotlib.pyplot as plt
import crepe
import soundfile as sf

from scipy.signal import find_peaks
import librosa
import os


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


def get_pitch_crepe_bandpass(
    audio_data: np.ndarray, samplerate, length, model_capacity="tiny"
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


def get_pitch_fft_interpolated(audio, sample_rate):
    """
    Estimates the pitch (dominant frequency) of an audio signal.

    This function computes the magnitude spectrum via FFT, detects peaks using
    scipy.signal.find_peaks, and uses parabolic interpolation around the highest
    peak to estimate the frequency with sub-bin resolution.

    Parameters:
        audio (np.ndarray): 1D numpy array containing the audio signal.
        sample_rate (int): Sampling rate of the audio signal in Hz.

    Returns:
        float or None: The estimated pitch in Hz, or None if no peak is found.
    """
    # Number of samples in the audio signal.
    N = len(audio)

    # Compute the real FFT and corresponding frequency bins.
    spectrum = np.abs(np.fft.rfft(audio))

    # Find peaks in the magnitude spectrum.
    peaks, properties = find_peaks(spectrum, height=0)

    # If no peaks are found, return None.
    if len(peaks) == 0:
        return None

    # Select the peak with the highest magnitude.
    dominant_peak = peaks[np.argmax(spectrum[peaks])]

    # Parabolic interpolation around the peak to refine the peak position.
    if dominant_peak > 0 and dominant_peak < len(spectrum) - 1:
        # Magnitudes of the neighboring bins.
        alpha = spectrum[dominant_peak - 1]
        beta = spectrum[dominant_peak]
        gamma = spectrum[dominant_peak + 1]

        # Calculate the adjustment (p) using the parabolic formula.
        denominator = alpha - 2 * beta + gamma
        if denominator == 0:
            p = 0
        else:
            p = 0.5 * (alpha - gamma) / denominator
    else:
        p = 0

    # The refined index of the peak.
    refined_index = dominant_peak + p

    # Calculate the pitch using the refined index.
    # The frequency resolution of the FFT is sample_rate / N.
    pitch = refined_index * (sample_rate / N)

    return pitch,1

def generate_noise_filter(noise_audio,sample_rate, output_filter_path="filters/noise_filter.npz", n_fft=1024, hop_length=256):
    """
    Loads a .npz file containing background noise (keys: "audio", "sample_rate"),
    computes its STFT, and averages the magnitude spectrum over time to create a noise filter.
    The filter (and STFT parameters) is then saved to the specified output path.
    
    Parameters:
        noise_npz_path (str): Path to the .npz file containing background noise.
        output_filter_path (str): Path (including filename) to save the noise filter.
        n_fft (int): FFT window size.
        hop_length (int): Hop length for the STFT.
    """
    # # Load the background noise data
    # data = np.load(noise_npz_path)
    # noise_audio = data['audio']
    # sample_rate = int(data['sample_rate'])
    
    # Compute the STFT of the noise audio
    noise_stft = librosa.stft(noise_audio, n_fft=n_fft, hop_length=hop_length)
    noise_mag = np.abs(noise_stft)
    
    # Average the magnitude over time to obtain a noise spectrum (filter)
    noise_filter = np.mean(noise_mag, axis=1)  # shape: (n_fft//2+1,)
    
    noise_level = np.max(abs(noise_audio))
    # Ensure the filters directory exists
    os.makedirs(os.path.dirname(output_filter_path), exist_ok=True)
    
    # Save the filter along with STFT parameters and sample rate
    np.savez(output_filter_path, noise_filter=noise_filter, n_fft=n_fft, hop_length=hop_length, sample_rate=sample_rate,noise_level=noise_level)
    print(f"Noise filter saved to {output_filter_path}")
    return noise_filter, n_fft, hop_length, sample_rate,noise_level


def remove_noise(audio, sample_rate, filter_file_path="filters/noise_filter.npz"):
    """
    Removes background noise from the given audio signal using spectral subtraction.
    
    This function:
      1. Loads the saved noise filter.
      2. If the sample rate differs from the filter’s sample rate, resamples the audio.
      3. Computes the STFT of the input audio.
      4. Subtracts the noise spectrum from the magnitude spectrogram (ensuring no negative values).
      5. Reconstructs the time-domain signal using the inverse STFT.
    
    Parameters:
        audio (np.ndarray): Input noisy audio signal.
        sample_rate (int): Sample rate of the input audio.
        filter_file_path (str): Path to the saved noise filter (.npz file).
    
    Returns:
        np.ndarray: The denoised audio signal.
    """
    # Load the saved filter and parameters
    filter_data = np.load(filter_file_path)
    noise_filter = filter_data['noise_filter']
    n_fft = int(filter_data['n_fft'])
    hop_length = int(filter_data['hop_length'])
    filter_sr = int(filter_data['sample_rate'])
    
    # Resample audio if its sample rate does not match the filter's sample rate
    if sample_rate != filter_sr:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=filter_sr)
        sample_rate = filter_sr
        # print(f"Audio resampled to {filter_sr} Hz to match filter sample rate.")
    
    # Compute the STFT of the input audio
    audio_stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(audio_stft)
    phase = np.angle(audio_stft)
    
    # Subtract the noise filter (applied across each time frame)
    # Expand noise_filter to match the shape of magnitude spectrogram
    noise_filter_expanded = noise_filter[:, np.newaxis]
    cleaned_magnitude = magnitude - noise_filter_expanded
    cleaned_magnitude = np.maximum(cleaned_magnitude, 0)  # clip negative values
    
    # Reconstruct the STFT with the original phase
    cleaned_stft = cleaned_magnitude * np.exp(1j * phase)
    
    # Compute the inverse STFT to get the time-domain cleaned audio
    cleaned_audio = librosa.istft(cleaned_stft, hop_length=hop_length)
    
    return cleaned_audio


# Example usage:
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Generate a 1-second 440 Hz sine wave for demonstration.
    sample_rate = 44100
    t = np.linspace(0, 1, sample_rate, endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)

    estimated_pitch = get_pitch_fft_interpolated(audio, sample_rate)
    print(f"Estimated pitch: {estimated_pitch:.2f} Hz")

    # Optionally, visualize the spectrum and the detected peak.
    N = len(audio)
    spectrum = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(N, d=1 / sample_rate)

    plt.figure(figsize=(10, 4))
    plt.plot(freqs, spectrum, label="Spectrum")
    plt.axvline(
        estimated_pitch,
        color="r",
        linestyle="--",
        label=f"Estimated Pitch: {estimated_pitch:.2f} Hz",
    )
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title("Frequency Spectrum with Estimated Pitch")
    plt.legend()
    plt.show()
