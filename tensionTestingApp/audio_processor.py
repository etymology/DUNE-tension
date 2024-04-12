import numpy as np
import crepe
import sounddevice as sd

class AudioProcessor:
    def __init__(self, device_index, samplerate):
        self.device_index = device_index
        self.samplerate = samplerate

    def record_audio(self, duration: float) -> np.ndarray:
        """Record audio from the microphone for a specified duration."""
        with sd.InputStream(device=self.device_index, channels=1, samplerate=self.samplerate, dtype='float32') as stream:
            frames = int(duration * self.samplerate)
            audio_data, _ = stream.read(frames)
        return audio_data

    def get_pitch_from_audio(self, audio_data: np.ndarray) -> tuple[float, float]:
        """Extract the pitch and confidence from the audio data."""
        time, frequency, confidence, activation = crepe.predict(
            audio_data, self.samplerate, viterbi=False)
        max_confidence_idx = np.argmax(confidence)
        pitch = frequency[max_confidence_idx]
        confidence_level = confidence[max_confidence_idx]
        return pitch, confidence_level

    def detect_noise_threshold(self, audio_data: np.ndarray, threshold: float) -> bool:
        """Check if the maximum amplitude in the audio exceeds a specified threshold."""
        max_amplitude = np.max(np.abs(audio_data))
        return max_amplitude >= threshold
