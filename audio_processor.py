import numpy as np
from scipy.fft import rfftfreq, rfft
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import crepe
import sounddevice as sd


class AudioProcessor:
    def __init__(self, device_index, samplerate):
        self.device_index = device_index
        self.samplerate = samplerate

    def detect_sound(self, audio_signal, threshold):
        # Detect sound based on energy threshold
        return np.max(np.abs(audio_signal)) >= float(threshold)

    def record_audio_trigger(self, noise_threshold: float, sound_length: float):
        start_time = datetime.now()
        while True:
            audio_data = self.record_audio(0.005)
            if self.detect_sound(audio_data, noise_threshold):
                return self.record_audio(sound_length)
            elif datetime.now() > start_time + timedelta(seconds=2):
                print("No sound detected. Timed out. Quitting.")
                return None

    def record_audio(self, duration: float) -> np.ndarray:
        """Record audio from the microphone for a specified duration."""
        with sd.InputStream(device=self.device_index, samplerate=self.samplerate, dtype='float32') as stream:
            frames = int(duration * self.samplerate)
            audio_data, _ = stream.read(frames)
        # try:
        #     sd.play(audio_data)
        # except sd.PortAudioError as e:
        #     print(f"PortAudioError: {e}")
        return np.array(audio_data).flatten()

    def get_pitch_crepe(self, audio_data: np.ndarray) -> tuple[float, float]:
        """Extract the pitch and confidence from the audio data."""
        time, frequency, confidence, activation = crepe.predict(
            audio_data, self.samplerate, viterbi=False)
        max_confidence_idx = np.argmax(confidence)
        pitch = frequency[max_confidence_idx]
        confidence_level = confidence[max_confidence_idx]
        return pitch, confidence_level

    def get_pitch_scipy(self, audio_data: np.array, noise_threshold: float) -> tuple[float, float]:
        SampHz = self.samplerate
        fOmega = np.abs(rfft(audio_data))
        Omega = rfftfreq(len(audio_data), d=1/SampHz)

        # After obtaining raw signal and its fft, want to improve freq resolution
        # to do this, apply a filter and then perform cubic interpolation.
        funcOmega = interp1d(Omega[Omega < 1000],
                             fOmega[Omega < 1000], kind='cubic')

        # choose same frequency space as the labview plotter
        interpOmega = np.linspace(0.0, 250, 100000)
        interpfOmega = funcOmega(interpOmega)

        # Find the largest bumps
        ind = np.argsort(interpOmega)
        sortfOmega = interpfOmega[ind]
        sortOmega = interpOmega[ind]

        plt.title("audio amplitude")
        plt.plot(audio_data)
        plt.show()

        indPk, properties = find_peaks(sortfOmega)
        fOmegaPk = sortfOmega[indPk]
        OmegaPk = sortOmega[indPk]

        oldfOmegaPk = fOmegaPk
        fOmegaPk = fOmegaPk[oldfOmegaPk > noise_threshold]
        OmegaPk = OmegaPk[oldfOmegaPk > noise_threshold]

        hsortind = np.flip(np.argsort(fOmegaPk))
        hsortfOmegaPk = fOmegaPk[hsortind]
        hsortOmegaPk = OmegaPk[hsortind]

        if (len(hsortOmegaPk) > 10):
            for i in range(10):
                print(i, ": ")
                print("Peak Freq (Hz): ", hsortOmegaPk[i])
                print("Peak Height: ", hsortfOmegaPk[i])

        plt.title("audio freq")
        plt.plot(sortOmega, sortfOmega, label="Interpolated")
        plt.plot(Omega[np.argsort(Omega)], fOmega[np.argsort(
            Omega)], label="Original", linestyle='--')
        plt.scatter(hsortOmegaPk[:10], hsortfOmegaPk[:10],
                    linestyle="None", color='red', label="Peaks")
        plt.xlim(0, 250)
        plt.xlabel("Hz")
        plt.ylabel("f")
        plt.legend()
        plt.show()

        if (len(OmegaPk) > 0):
            qualstr = input("Good point? (num/n): ")
            if (qualstr.isnumeric()):
                qual = int(qualstr)
                frequency = hsortOmegaPk[qual]
                confidence = 1.0
            elif (qualstr == 'n'):
                frequency = 0.0
                confidence = 0.0

        return frequency, confidence

    def get_pitch_fft(self, data):
        """
        Estimate the pitch of the audio data using FFT.
        """
        # Convert audio data to frequency domain using FFT
        fft_spectrum = np.fft.rfft(data)
        # Get the magnitude
        magnitude = np.abs(fft_spectrum)
        # Find the peak frequency
        frequency = np.argmax(magnitude)
        # Convert the peak frequency index to actual frequency
        pitch = frequency * self.samplerate / len(data)
        return pitch, 1


if __name__ == "__main__":
    from plotter import Plotter
    from config_manager import ConfigManager
    from device_manager import DeviceManager
    # tensiometer = Tensiometer()
    config_manager = ConfigManager()
    device_manager = DeviceManager(config_manager.config)
    plotter = Plotter()
    audio_processor = AudioProcessor(device_index=0, samplerate=48000)

    device_manager.select_audio_device()

    def record_plot_log():
        audio_signal = audio_processor.record_audio(.5)
        # frequency, confidence = audio_processor.get_pitch_from_audio(audio_signal)
        # print(f"Detected pitch: {frequency:.2f} Hz with confidence: {confidence:.2f}")
        # plotter.plot_audio(audio_signal, audio_processor.samplerate, frequency, confidence)
        frequency = audio_processor.get_pitch_crepe(audio_signal)
        confidence = 0.0
        print(f"Detected pitch: {frequency} Hz with FFT.")
        plotter.plot_audio(
            audio_signal, audio_processor.samplerate, frequency, confidence)

    while True:
        # tensiometer.pluck_string()
        print("\nListening...")
        record_plot_log()
        input("Press Enter to continue...")
