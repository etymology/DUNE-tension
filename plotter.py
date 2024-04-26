import matplotlib.pyplot as plt
import numpy as np

class Plotter:
    def plot_audio(self, audio_signal: np.ndarray, samplerate: int, fundamental_freq: float = None, confidence: float = None):
        """Plot both the waveform and the frequency spectrum of the audio signal in the same window."""
        # Create a figure with two subplots
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        # Plot the waveform
        times = np.linspace(0, len(audio_signal) / samplerate, num=len(audio_signal))
        axes[0].plot(times, audio_signal)
        axes[0].set_title('Recorded Waveform')
        axes[0].set_xlabel('Time (seconds)')
        axes[0].set_ylabel('Amplitude')
        axes[0].grid(True)

        # Plot the frequency spectrum
        fft_spectrum = np.abs(np.fft.fft(audio_signal))
        freqs = np.fft.fftfreq(len(audio_signal), 1 / samplerate)
        axes[1].plot(freqs[:len(freqs) // 2], fft_spectrum[:len(fft_spectrum) // 2])
        axes[1].set_title('Frequency Spectrum')
        axes[1].set_xlabel('Frequency (Hz)')
        axes[1].set_ylabel('Magnitude')

        if fundamental_freq and confidence:
            axes[1].axvline(fundamental_freq, color='r', linestyle='--', label=f'Fundamental Frequency at {fundamental_freq} Hz with confidence {confidence:.2f}')
            axes[1].legend()

        axes[1].grid(True)

        # Adjust layout to prevent overlap
        plt.tight_layout()
        plt.show()
