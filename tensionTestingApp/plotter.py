import matplotlib.pyplot as plt
import numpy as np

class Plotter:
    def plot_waveform(self, audio_signal: np.ndarray, samplerate: int):
        """Plot the waveform of the recorded audio signal."""
        plt.figure(figsize=(10, 4))
        times = np.linspace(0, len(audio_signal) / samplerate, num=len(audio_signal))
        plt.plot(times, audio_signal)
        plt.title('Recorded Waveform')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_frequency_spectrum(self, audio_signal: np.ndarray, samplerate: int, fundamental_freq: float = None, confidence: float = None):
        """Plot the frequency spectrum and highlight the fundamental frequency."""
        fft_spectrum = np.abs(np.fft.fft(audio_signal))
        freqs = np.fft.fftfreq(len(audio_signal), 1 / samplerate)

        plt.figure(figsize=(10, 4))
        plt.plot(freqs[:len(freqs) // 2], fft_spectrum[:len(fft_spectrum) // 2])
        plt.title('Frequency Spectrum')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        
        if fundamental_freq and confidence:
            plt.axvline(fundamental_freq, color='r', linestyle='--', label=f'Fundamental Frequency at {fundamental_freq} Hz with confidence {confidence:.2f}')
            plt.legend()

        plt.grid(True)
        plt.tight_layout()
        plt.show()
