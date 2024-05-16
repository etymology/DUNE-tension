import sounddevice as sd
from typing import Tuple
from maestro import Controller

USB_SOUND_CARD_NAME = "USB PnP Sound Device"  #: Audio (hw:1,0)"


class DeviceManager:
    def __init__(self, config):
        self.servo_controller = Controller()
        self.init_audio_devices()

    def init_audio_devices(self):
        """Initialize the audio devices based on current configuration."""
        try:
            device_info = sd.query_devices()
            usb_pnp_index = next(
                index
                for index, d in enumerate(device_info)
                if USB_SOUND_CARD_NAME in d["name"]
            )
            if usb_pnp_index is not None:
                self.sound_device_index = usb_pnp_index
                print(f"Using USB PnP Sound Device (hw:{usb_pnp_index},0)")
            else:
                self.select_audio_device()
            self.device_samplerate = (
                device_info[usb_pnp_index]["default_samplerate"]
                if usb_pnp_index is not None
                else 44100
            )
            self.current_device = device_info[self.sound_device_index]
        except Exception as e:
            print("Couldn't find USB PnP Sound Device.")
            print(f"Failed to initialize audio devices: {str(e)}")
            self.select_audio_device()
            
    def select_audio_device(self):
        """Allow the user to select an audio device from available options."""
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            print(
                f"{i+1}. {device['name']} (default sr: {device['default_samplerate']} Hz)"
            )
        try:
            selection = int(input("Select an audio device: "))
            if 1 <= selection <= len(devices):
                self.sound_device_index = selection - 1
                self.device_samplerate = devices[self.sound_device_index][
                    "default_samplerate"
                ]
            else:
                print(
                    "Invalid selection. Please enter a number within the valid range."
                )
        except ValueError:
            print("Invalid input. Please enter a number.")

    def record_audio(self, duration: float) -> Tuple[float, float]:
        """Record audio for a given duration using the selected audio device."""
        with sd.InputStream(
            device=self.sound_device_index,
            channels=1,
            samplerate=self.device_samplerate,
            dtype="float32",
        ) as stream:
            audio_data = stream.read(int(duration * self.device_samplerate))
        return audio_data
