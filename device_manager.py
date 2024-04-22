import sounddevice as sd
from typing import Tuple
from maestro import Controller

class DeviceManager:
    def __init__(self, config):
        self.sound_device_index = config.get('sound_device_index', 0)
        self.device_samplerate = config.get('device_samplerate', 44100)
        self.servo_controller = Controller()
        self.init_audio_devices()

    def init_audio_devices(self):
        """Initialize the audio devices based on current configuration."""
        try:
            devices = sd.query_devices()
            self.current_device = devices[self.sound_device_index]
        except Exception as e:
            print(f"Failed to initialize audio devices: {str(e)}")

    def select_audio_device(self):
        """Allow the user to select an audio device from available options."""
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            print(f"{i+1}. {device['name']} (default sr: {device['default_samplerate']} Hz)")
        try:
            selection = int(input("Select an audio device: "))
            if 1 <= selection <= len(devices):
                self.sound_device_index = selection - 1
                self.device_samplerate = devices[self.sound_device_index]['default_samplerate']
            else:
                print("Invalid selection. Please enter a number within the valid range.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    def record_audio(self, duration: float) -> Tuple[float, float]:
        """Record audio for a given duration using the selected audio device."""
        with sd.InputStream(device=self.sound_device_index, channels=1, samplerate=self.device_samplerate, dtype='float32') as stream:
            audio_data = stream.read(int(duration * self.device_samplerate))
        return audio_data