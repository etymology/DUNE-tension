from scipy.fft import rfftfreq, fftfreq, rfft, fft
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

from config_manager import ConfigManager
from device_manager import DeviceManager
from datetime import datetime
from datetime import timedelta
import numpy as np
import csv
from time import sleep
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

    def log_frequency_and_wire_number(self, frequency, confidence, wire_number, filename):
        with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([wire_number, confidence, frequency])

    # Additional methods to handle different functionalities
    def handle_select_device(self):
        self.device_manager.select_audio_device()
        self.config_manager.config['sound_device_index'] = self.device_manager.sound_device_index
        self.config_manager.config['device_samplerate'] = self.device_manager.device_samplerate

    def handle_record(self):
        self.tensiometer.pluck_string()
        print("\nListening...")
        start_time = datetime.now()
        sleep(1.2)

        while True:
            # audio_data = self.audio_processor.record_audio(5.0)
            audio_data = self.audio_processor.record_audio(0.005)
            # plt.plot(audio_data)
            # plt.show()
            if self.audio_processor.detect_sound(audio_data, self.config_manager.config['noise_threshold']):
                audio_signal = self.audio_processor.record_audio(0.3)
                break
            elif datetime.now() > start_time + timedelta(seconds=10):
                print("No sound detected. Quitting")
                audio_signal = np.array([])
                break
        if(audio_signal.size > 0):
            # crepe pitch detection
            # frequency, confidence = self.audio_processor.get_pitch_from_audio(audio_signal)
            # self.plotter.plot_waveform(audio_signal, self.audio_processor.samplerate)
            # self.plotter.plot_frequency_spectrum(audio_signal, self.audio_processor.samplerate, 
                                                 frequency, confidence)
            # np pitch detection
            SampHz = self.audio_processor.samplerate
            fOmega = np.abs(rfft(audio_signal))
            Omega = rfftfreq(len(audio_signal), d=1/SampHz)
            ind = np.argsort(Omega)
            sortfOmega = fOmega[ind]
            sortOmega = Omega[ind]

            plt.title("audio amplitude")
            plt.plot(audio_signal)
            plt.show()

            indPk, properties = find_peaks(sortfOmega)
            fOmegaPk = sortfOmega[indPk]
            OmegaPk = sortOmega[indPk]

            globthresh = 0
            oldfOmegaPk = fOmegaPk
            fOmegaPk = fOmegaPk[oldfOmegaPk > globthresh]
            OmegaPk = OmegaPk[oldfOmegaPk > globthresh]

            plt.title("audio freq")
            plt.plot(sortOmega, sortfOmega)
            plt.scatter(OmegaPk, fOmegaPk, linestyle="None", color='red')
            plt.xlim(0, 1000)
            plt.xlabel("Hz")
            plt.ylabel("f")
            plt.show()

            hsortind = np.flip(np.argsort(fOmegaPk))
            hsortfOmegaPk = fOmegaPk[hsortind]
            hsortOmegaPk = OmegaPk[hsortind]

            if(len(hsortOmegaPk)>0):
                for i in range(10):
                    print(i, ": ")
                    print("Peak Freq (Hz): ", hsortOmegaPk[i])
                    print("Peak Height: ", hsortfOmegaPk[i])

            if (len(OmegaPk)>0):
                qualstr = input("Good point? (num): ")
                if (qualstr.isnumeric()):
                    qual = int(qualstr)
                    frequency = hsortOmegaPk[qual]
                    confidence = 1.0

            curr_wirenum = self.config_manager.config['current_wirenumber']
            log_prompt = input(f"Do you want to log the frequency? [wire number {curr_wirenum}](y/n): ")
            if log_prompt.lower() == 'y':
                self.log_frequency_and_wire_number(frequency, confidence, curr_wirenum, "Output.csv")
                print("Frequency logged.")
            elif log_prompt.lower() == 'n':
                print("Frequency not logged.")

    def handle_goto_spec_wire(self):
        wire_number = input("Enter the wire number to go to: ")
        sleep(1.0)
        curr_layer = self.config_manager.load_config()['current_layer']
        wire_loc = self.apa.get_plucking_point(wire_number, curr_layer)
        target_x = wire_loc['X']
        target_y = wire_loc['Y']
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

        # Update config
        self.config_manager.update_config('current_wirenumber', wire_number)

    def handle_goto_next_wire(self):
        wire_number = str(int(self.config_manager.load_config()['current_wirenumber'])+1)
        sleep(1.0)
        curr_layer = self.config_manager.load_config()['current_layer']
        wire_loc = self.apa.get_plucking_point(wire_number, curr_layer)
        target_x = wire_loc['X']
        target_y = wire_loc['Y']
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

        # Update config
        self.config_manager.update_config('current_wirenumber', wire_number)

    def handle_goto_prev_wire(self):
        wire_number = str(int(self.config_manager.load_config()['current_wirenumber'])-1)
        sleep(1.0)
        curr_layer = self.config_manager.load_config()['current_layer']
        wire_loc = self.apa.get_plucking_point(wire_number, curr_layer)
        target_x = wire_loc['X']
        target_y = wire_loc['Y']
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

        # Update config
        self.config_manager.update_config('current_wirenumber', wire_number)

    def handle_calibration(self):
        self.apa.calibrate()

    def handle_change_variables(self):
        print("Current config: ", self.config_manager.config)
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
