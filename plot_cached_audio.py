import os
import csv
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Use a non-interactive backend
import matplotlib.pyplot as plt
from audioProcessing import (
    get_pitch_crepe,
    get_pitch_naive_fft,
    get_pitch_autocorrelation,
)
import logging

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

# Specify the directory containing the .npz files
folder_path = "audio/"
sample_rate = 44100  # Sample rate in Hz
subsample_length = 0.2  # Length of the subsample in seconds

# List all files in the directory
files = os.listdir(folder_path)

# Filter out the .npz files
npz_files = [file for file in files if file.endswith(".npz")]


# Function to load .npz files safely
def safe_load_npz(file_path):
    try:
        return np.load(file_path, allow_pickle=True)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


# Open a CSV file to write the results
with open("results.csv", "w", newline="") as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(
        [
            "Filename",
            "Array Name",
            "Minimum",
            "Maximum",
            "Crepe Frequency",
            "Crepe Confidence",
        ]
    )

    # Iterate over each .npz file
    for npz_file in npz_files:
        # Load the .npz file with allow_pickle=True
        data = safe_load_npz(os.path.join(folder_path, npz_file))
        if data is None:
            continue

        # Assuming each .npz file contains arrays, plot the time domain and FFT of each array
        for array_name in data.files:
            try:
                array = data[array_name]

                # Ensure the array is 1-dimensional and not empty
                if isinstance(array, np.ndarray) and array.ndim == 1 and array.size > 0:
                    filename = f"images/{npz_file}_{array_name}_time_and_fft.png"
                    if os.path.exists(filename):
                        print(f"File {filename} already exists.")
                    else:
                        # Get pitch using different methods
                        autocorr_frequency, autocorr_confidence = (
                            get_pitch_autocorrelation(array, sample_rate)
                        )
                        crepe_frequency, crepe_confidence = get_pitch_crepe(
                            array, sample_rate
                        )
                        fft_frequency, fft_confidence = get_pitch_naive_fft(
                            array, sample_rate
                        )

                        # Compute the FFT of the array
                        fft_result = np.fft.fft(array)
                        fft_freq = np.fft.fftfreq(len(array), d=1 / sample_rate)

                        # Print FFT pitch for debugging
                        print(
                            f"FFT pitch is {fft_frequency} Hz with confidence {fft_confidence}"
                        )

                        # Filter to include only positive frequencies up to 10000 Hz
                        positive_freq_indices = (fft_freq > 0) & (fft_freq <= 10000)
                        positive_freqs = fft_freq[positive_freq_indices]
                        positive_magnitudes = np.abs(fft_result)[positive_freq_indices]

                        # Compute the autocorrelation of the array
                        autocorr = np.correlate(array, array, mode="full")
                        autocorr = autocorr[
                            autocorr.size // 2 :
                        ]  # Keep only the second half
                        # Create a plot with three subplots
                        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

                        # Plot the time domain signal
                        ax1.plot(array)
                        ax1.set_title(f"Time Domain of {npz_file} - {array_name}")
                        ax1.set_xlabel("Sample Index")
                        ax1.set_ylabel("Amplitude")

                        # Plot the positive frequencies of the FFT result up to 10000 Hz
                        ax2.plot(positive_freqs, positive_magnitudes)
                        ax2.set_title(
                            f"Positive Frequency Domain of {npz_file} - {array_name}"
                        )
                        ax2.set_xlabel("Frequency (Hz)")
                        ax2.set_ylabel("Magnitude")
                        ax2.set_xscale(
                            "log"
                        )  # Set the x-axis to a logarithmic scale

                        # Add vertical lines and labels for the detected frequencies
                        if autocorr_frequency <= 5000:
                            ax2.axvline(
                                x=autocorr_frequency, color="r", linestyle="--"
                            )
                            ax2.text(
                                autocorr_frequency,
                                max(positive_magnitudes) / 2,
                                f"{autocorr_frequency:.2f} Hz\nConf: {autocorr_confidence:.2f}",
                                color="r",
                                ha="center",
                            )
                        if crepe_frequency <= 5000:
                            ax2.axvline(
                                x=crepe_frequency, color="g", linestyle="--"
                            )
                            ax2.text(
                                crepe_frequency,
                                max(positive_magnitudes) / 2,
                                f"{crepe_frequency:.2f} Hz\nConf: {crepe_confidence:.2f}",
                                color="g",
                                ha="center",
                            )

                        # Plot the autocorrelation function
                        ax3.plot(autocorr)
                        ax3.set_title(
                            f"Autocorrelation Function of {npz_file} - {array_name}"
                        )
                        ax3.set_xlabel("Lag")
                        ax3.set_ylabel("Autocorrelation")

                        # Save the figure
                        plt.tight_layout()
                        plt.savefig(filename)
                        plt.show(fig)  # Close the figure to avoid display issues

            except Exception as e:
                print(f"Error processing array {array_name} in file {npz_file}: {e}")

print("Processing complete.")
