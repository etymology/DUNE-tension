import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft, istft
import os
from datetime import datetime
import os
import numpy as np
import librosa
import soundfile as sf

# File paths
file1 = "/home/ben/DUNE-tension/audio/GA9_2025-02-17_17-15-04.npz"
file2 = "/home/ben/DUNE-tension/audio/GA9_2025-02-17_17-15-02.npz"

# Load the .npz files
data1 = np.load(file1)
data2 = np.load(file2)

# Check which keys are available in each file
print("Keys in {}:".format(file1), data1.files)
print("Keys in {}:".format(file2), data2.files)

# Assume the audio signal is stored under a key 'audio'.
# If not, we simply take the first array available.
if "audio" in data1.files:
    audio1 = data1["audio"]
else:
    audio1 = data1[data1.files[0]]

if "audio" in data2.files:
    audio2 = data2["audio"]
else:
    audio2 = data2[data2.files[0]]

fs = 44100  # Sampling rate in Hz

# # Create time axes for the signals
# t1 = np.linspace(0, len(audio1) / fs, len(audio1), endpoint=False)
# t2 = np.linspace(0, len(audio2) / fs, len(audio2), endpoint=False)

# # Plot the audio signals
# plt.figure(figsize=(12, 8))

# plt.subplot(2, 1, 1)
# plt.plot(t1, audio1)
# plt.title("Audio Signal from {}".format(file1))
# plt.xlabel("Time (s)")
# plt.ylabel("Amplitude")

# plt.subplot(2, 1, 2)
# plt.plot(t2, audio2)
# plt.title("Audio Signal from {}".format(file2))
# plt.xlabel("Time (s)")
# plt.ylabel("Amplitude")

# plt.tight_layout()
# plt.show()


# --------------------------
# 1. Load the audio signals
# --------------------------
# File "17-15-04.npz" contains only background noise
# File "17-15-02.npz" contains the plucked-string signal (signal + background noise)
# noise_file = "17-15-04.npz"
# noisy_file = "17-15-02.npz"
# fs = 44100  # Sampling rate in Hz


# # Helper: load the npz file and extract the first array if key 'audio' is not present.
# def load_audio(npz_file):
#     data = np.load(npz_file)
#     key = "audio" if "audio" in data.files else data.files[0]
#     return data[key]


# Load signals
noise_signal = audio1  # load_audio(noise_file)
noisy_signal = audio2  # load_audio(noisy_file)

# --------------------------
# 2. Set up STFT parameters
# --------------------------
n_fft = 1024  # FFT window size
hop = n_fft // 2  # 50% overlap
window = "hann"  # Window type

# --------------------------
# 3. Compute STFTs of noise and noisy signals
# --------------------------
# Background noise STFT
f_noise, t_noise, Zxx_noise = stft(
    noise_signal, fs=fs, window=window, nperseg=n_fft, noverlap=hop
)
# Compute an average noise magnitude and power spectrum across time frames (one value per frequency bin)
noise_mag = np.mean(np.abs(Zxx_noise), axis=1)  # for spectral subtraction
noise_psd = np.mean(np.abs(Zxx_noise) ** 2, axis=1)  # for Wiener filtering

# Noisy signal STFT
f_noisy, t_noisy, Zxx_noisy = stft(
    noisy_signal, fs=fs, window=window, nperseg=n_fft, noverlap=hop
)

# --------------------------
# 4. Method 1: Spectral Subtraction
# --------------------------
# For each time frame subtract the noise magnitude (ensure no negative values)
mag_noisy = np.abs(Zxx_noisy)
phase_noisy = np.angle(Zxx_noisy)
# Subtract the noise magnitude estimate (broadcast noise_mag over time frames)
mag_subtracted = np.maximum(mag_noisy - noise_mag[:, np.newaxis], 0)
# Reconstruct the STFT using the original phase
Zxx_specsub = mag_subtracted * np.exp(1j * phase_noisy)
# Inverse STFT to get the filtered time signal
_, specsub_signal = istft(
    Zxx_specsub, fs=fs, window=window, nperseg=n_fft, noverlap=hop
)

# --------------------------
# 5. Method 2: Wiener Filtering
# --------------------------
# Compute power spectrum of the noisy signal
P_noisy = np.abs(Zxx_noisy) ** 2
# Avoid division by zero
eps = 1e-8
# Compute Wiener gain: H = max(P_noisy - noise_psd, 0) / P_noisy (applied per frequency bin and time frame)
# Note: noise_psd is broadcast along time
H_wiener = np.maximum(P_noisy - noise_psd[:, np.newaxis], 0) / (P_noisy + eps)
# Apply the filter (multiplying both magnitude and phase)
Zxx_wiener = H_wiener * Zxx_noisy
# Inverse STFT to get time-domain signal
_, wiener_signal = istft(Zxx_wiener, fs=fs, window=window, nperseg=n_fft, noverlap=hop)


# --------------------------
# 6. Helper Functions for Plotting
# --------------------------
def plot_time_domain(signal, fs, title):
    t = np.linspace(0, len(signal) / fs, len(signal), endpoint=False)
    plt.plot(t, signal)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title(title)


def plot_frequency_spectrum(signal, fs, title):
    import numpy as np
    import matplotlib.pyplot as plt

    N = len(signal)
    freq = np.fft.rfftfreq(N, 1 / fs)
    spectrum = np.abs(np.fft.rfft(signal))

    # Create a mask for frequencies up to 3000 Hz
    mask = freq <= 3000
    freq_masked = freq[mask]
    spectrum_masked = spectrum[mask]

    plt.plot(freq_masked, spectrum_masked)
    plt.xscale("log")
    plt.xlim(1, 3000)  # Avoid 0 Hz in log scale by setting lower limit to 1 Hz
    plt.xlabel("Frequency [Hz] (log scale)")
    plt.ylabel("Magnitude")
    plt.title(title)


# --------------------------
# 7. Plot the Results
# --------------------------
plt.figure(figsize=(14, 10))

# (A) Original Noisy Signal: Time domain
plt.subplot(3, 2, 1)
plot_time_domain(noisy_signal, fs, "Noisy Signal (Time Domain)")

# (B) Original Noisy Signal: Frequency spectrum
plt.subplot(3, 2, 2)
plot_frequency_spectrum(noisy_signal, fs, "Noisy Signal (Frequency Spectrum)")

# (C) Spectral Subtraction Result: Time domain
plt.subplot(3, 2, 3)
plot_time_domain(specsub_signal, fs, "Spectral Subtraction (Time Domain)")

# (D) Spectral Subtraction Result: Frequency spectrum
plt.subplot(3, 2, 4)
plot_frequency_spectrum(specsub_signal, fs, "Spectral Subtraction (Frequency Spectrum)")

# (E) Wiener Filter Result: Time domain
plt.subplot(3, 2, 5)
plot_time_domain(wiener_signal, fs, "Wiener Filtering (Time Domain)")

# (F) Wiener Filter Result: Frequency spectrum
plt.subplot(3, 2, 6)
plot_frequency_spectrum(wiener_signal, fs, "Wiener Filtering (Frequency Spectrum)")

plt.tight_layout()
plt.show()


def save_background_filter(
    noise_signal,
    sample_rate,
    n_fft=1024,
    hop=None,
    directory="filters",
    file_prefix="noise_filter",
):
    """
    Compute the noise filter parameters from a background noise signal and save them in an .npz file.

    The function computes the STFT of the noise signal using a Hann window and then averages the
    magnitude and power spectra over time. It saves the following fields:
      - noise_mag: average noise magnitude (for spectral subtraction)
      - noise_psd: average noise power spectral density (for Wiener filtering)
      - frequencies: the frequency bins corresponding to the filter arrays
      - sample_rate: the sample rate of the original recording
      - n_fft: the FFT window size used
      - hop: the hop length (overlap) used

    The file is saved with a name of the form:
      f"{file_prefix}_{YYYY-MM-DD_HH-MM-SS}.npz" in the specified directory.

    Parameters:
        noise_signal (np.ndarray): 1D array containing the background noise signal.
        sample_rate (int): Sample rate of the noise signal in Hz.
        n_fft (int): FFT window size (default is 1024).
        hop (int or None): Hop size (overlap) for the STFT. If None, defaults to n_fft // 2.
        directory (str): Directory where the file will be saved (default "filters").
        file_prefix (str): Prefix for the saved file name (default "noise_filter").

    Returns:
        file_path (str): The full path to the saved .npz file.
    """
    # Set hop if not provided
    if hop is None:
        hop = n_fft // 2

    # Adjust n_fft if the noise signal is shorter than the specified window length.
    if len(noise_signal) < n_fft:
        n_fft = len(noise_signal)
        hop = max(1, n_fft // 2)

    # Compute the STFT of the noise signal
    f, t, Zxx = stft(
        noise_signal, fs=sample_rate, window="hann", nperseg=n_fft, noverlap=hop
    )

    # Average over time frames: these serve as our noise filter estimates.
    noise_mag = np.mean(np.abs(Zxx), axis=1)  # for spectral subtraction
    noise_psd = np.mean(np.abs(Zxx) ** 2, axis=1)  # for Wiener filtering

    # Ensure the output directory exists.
    os.makedirs(directory, exist_ok=True)

    # Create a file name with a timestamp.
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"{file_prefix}_{timestamp}.npz"
    file_path = os.path.join(directory, file_name)

    # Save the filter parameters.
    np.savez(
        file_path,
        noise_mag=noise_mag,
        noise_psd=noise_psd,
        frequencies=f,
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop=hop,
    )

    print(f"Saved background noise filter to: {file_path}")
    return file_path


def apply_saved_filter(audio_signal, filter_file, method="spectral_subtraction"):
    """
    Load a saved noise filter from an .npz file and apply it to a new audio signal.

    The filter file is expected to contain the following fields:
        - noise_mag: average noise magnitude (for spectral subtraction)
        - noise_psd: average noise power spectral density (for Wiener filtering)
        - frequencies: frequency bins (not directly used for filtering)
        - sample_rate: sample rate used to compute the filter
        - n_fft: FFT window length
        - hop: hop length (noverlap) used in the STFT

    Parameters:
        audio_signal (np.ndarray): 1D array of the new audio signal.
        filter_file (str): Path to the .npz file with the saved filter.
        method (str): Filtering method to use; either 'spectral_subtraction' or 'wiener'.

    Returns:
        filtered_signal (np.ndarray): The noise-reduced audio signal (time domain).
    """
    # Load filter parameters from file
    filter_data = np.load(filter_file)
    noise_mag = filter_data["noise_mag"]
    noise_psd = filter_data["noise_psd"]
    sample_rate = int(filter_data["sample_rate"])
    n_fft = int(filter_data["n_fft"])
    hop = int(filter_data["hop"])

    # Compute the STFT of the new audio signal using the same parameters
    f, t, Zxx = stft(
        audio_signal, fs=sample_rate, window="hann", nperseg=n_fft, noverlap=hop
    )

    if method == "spectral_subtraction":
        # Spectral Subtraction:
        # Subtract the average noise magnitude from each time frame (ensure no negatives)
        mag = np.abs(Zxx)
        phase = np.angle(Zxx)
        mag_filtered = np.maximum(mag - noise_mag[:, np.newaxis], 0)
        Zxx_filtered = mag_filtered * np.exp(1j * phase)

    elif method == "wiener":
        # Wiener Filtering:
        # Compute the power spectrum of the noisy signal
        P_noisy = np.abs(Zxx) ** 2
        eps = 1e-8  # to avoid division by zero
        # Calculate Wiener gain and apply it
        H = np.maximum(P_noisy - noise_psd[:, np.newaxis], 0) / (P_noisy + eps)
        Zxx_filtered = H * Zxx

    else:
        raise ValueError("Method must be either 'spectral_subtraction' or 'wiener'")

    # Inverse STFT to obtain the filtered time-domain signal
    _, filtered_signal = istft(
        Zxx_filtered, fs=sample_rate, window="hann", nperseg=n_fft, noverlap=hop
    )
    return filtered_signal





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
        print(f"Audio resampled to {filter_sr} Hz to match filter sample rate.")
    
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

# if __name__ == "__main__":
#     # --- Step 1: Generate and save noise filter ---
#     # Replace 'background_noise.npz' with your npz file containing background noise.
#     noise_npz_file = "background_noise.npz"
#     generate_noise_filter(noise_npz_file, output_filter_path="filters/noise_filter.npz", n_fft=1024, hop_length=256)
    
#     # --- Step 2: Remove noise from a new audio signal ---
#     # Replace 'noisy_audio.npz' with your npz file containing the new noisy audio.
#     new_audio_data = np.load("noisy_audio.npz")
#     noisy_audio = new_audio_data['audio']
#     sr = int(new_audio_data['sample_rate'])
    
#     # Apply the noise removal function
#     cleaned_audio = remove_noise(noisy_audio, sr, filter_file_path="filters/noise_filter.npz")
    
#     # Optionally, save the cleaned audio to a .wav file
#     output_wav = "cleaned_audio.wav"
#     sf.write(output_wav, cleaned_audio, sr)
#     print(f"Cleaned audio saved to {output_wav}")


generate_noise_filter("audio/GA9_2025-02-17_17-15-04.npz", output_filter_path="filters/noise_filter.npz", n_fft=1024, hop_length=256)
cleaned_audio = remove_noise(audio2, fs, filter_file_path="filters/noise_filter.npz")

plt.figure(figsize=(14, 10))

plot_time_domain(cleaned_audio, fs, "Cleaned Audio Signal (Time Domain)")
plt.show()