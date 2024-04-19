
from config_manager import ConfigManager
from device_manager import DeviceManager
from audio_processor import AudioProcessor
from plotter import Plotter
from ui_manager import UIManager
from apa import APA
from tensiometer import Tensiometer


ZONE_BOUNDARIES = [2200, 3400, 4600, 5800]  # These are example values


class TensionTestingApp:
    def __init__(self):
        self.config_manager = ConfigManager()
        self.apa = APA(self.config_manager.config["current_apa"])
        self.tensiometer = Tensiometer()
        self.device_manager = DeviceManager(self.config_manager.config)
        self.audio_processor = AudioProcessor(
            self.device_manager.sound_device_index, self.device_manager.device_samplerate)
        self.plotter = Plotter()
        self.ui_manager = UIManager(self)

    def run(self):
        self.ui_manager.run()

    # Additional methods to handle different functionalities
    def handle_select_device(self):
        self.device_manager.select_audio_device()

    def handle_record(self):
        audio_data = self.audio_processor.record_audio(0.5)
        frequency, confidence = self.audio_processor.get_pitch_from_audio(
            audio_data)
        self.plotter.plot_waveform(audio_data, self.audio_processor.samplerate)
        self.plotter.plot_frequency_spectrum(
            audio_data, self.audio_processor.samplerate, frequency, confidence)

    def handle_goto_wire(self):
        wire_number = input("Enter the wire number to go to: ")
        curr_layer = self.config_manager.load_config()['current_layer']
        target_x, target_y = self.apa.get_plucking_point(wire_number, curr_layer)
        current_x, current_y = self.tensiometer.get_xy()

        # Determine the current and target zones
        current_zone = sum(
            current_x > boundary for boundary in ZONE_BOUNDARIES)
        target_zone = sum(target_x > boundary for boundary in ZONE_BOUNDARIES)

        if current_zone != target_zone:
            # Move to y=0 if not already there to ensure a clear path horizontally across zones
            if current_y != 0:
                print("Moving to y=0 to avoid collision...")
                self.tensiometer.goto_xy(current_x, 0)
                current_y = 0

            # If moving to a higher zone, move to the nearest boundary to the right; if lower, to the left
            if target_zone > current_zone:
                next_boundary_x = min(
                    boundary for boundary in ZONE_BOUNDARIES if boundary > current_x)
            else:
                next_boundary_x = max(
                    boundary for boundary in ZONE_BOUNDARIES if boundary < current_x)

            print(f"Moving along x to boundary at x={next_boundary_x}...")
            self.tensiometer.goto_xy(next_boundary_x, 0)

        # Move to the final x, y coordinates
        print(f"Moving to final position x={target_x}, y={target_y}...")
        self.tensiometer.goto_xy(target_x, target_y)
        print(
            f"Arrived at wire number {wire_number} at x={target_x}, y={target_y}.")

    def handle_calibration(self):
        self.apa.calibrate()

    def handle_change_variables(self):
        key = input("Enter the configuration key to change: ")
        value = input(f"Enter the new value for {key}: ")
        self.config_manager.update_config(key, value)

    def handle_quit(self):
        print("Saving config file.")
        self.config_manager.save_config()
        self.tensiometer.__exit__(None, None, None)
        print("Exiting the application.")
        exit()


if __name__ == "__main__":
    app = TensionTestingApp()
    app.run()
