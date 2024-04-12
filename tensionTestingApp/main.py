
from config_manager import ConfigManager
from device_manager import DeviceManager
from audio_processor import AudioProcessor
from plotter import Plotter
from ui_manager import UIManager
from apa import APA
from tensiometer import Tensiometer

class TensionTestingApp:
    def __init__(self):
        self.config_manager = ConfigManager()
        self.apa = APA(self.config_manager.config["current_apa"])
        self.tensiometer = Tensiometer()
        self.device_manager = DeviceManager(self.config_manager.config)
        self.audio_processor = AudioProcessor(self.device_manager.sound_device_index, self.device_manager.device_samplerate)
        self.plotter = Plotter()
        self.ui_manager = UIManager(self)

    def run(self):
        self.ui_manager.run()

    # Additional methods to handle different functionalities
    def handle_select_device(self):
        self.device_manager.select_audio_device()

    def handle_record(self):
        audio_data = self.audio_processor.record_audio(0.5)
        frequency, confidence = self.audio_processor.get_pitch_from_audio(audio_data)
        self.plotter.plot_waveform(audio_data, self.audio_processor.samplerate)
        self.plotter.plot_frequency_spectrum(audio_data, self.audio_processor.samplerate, frequency, confidence)

    def handle_goto_wire(self):
        wire_number = int(input("Enter the wire number to go to: "))
        self.tensiometer.goto_xy(*self.apa.get_plucking_point(wire_number))

    def handle_calibration(self):
        self.apa.calibrate()

    def handle_change_variables(self):
        key = input("Enter the configuration key to change: ")
        value = input(f"Enter the new value for {key}: ")
        self.config_manager.update_config(key, value)

    def handle_quit(self):
        self.tensiometer.__exit__(None, None, None)
        print("Exiting the application.")
        exit()

if __name__ == "__main__":
    app = TensionTestingApp()
    app.run()
