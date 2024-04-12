import sys
import termios
import tty

from tensiometer import Tensiometer
from apa import APA
import sounddevice as sd
import crepe
import numpy as np
import csv
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import tensorflow as tf
import json

tf.get_logger().setLevel('ERROR')

RIB_SPACING = 500  # in mm

DEFAULT_CONFIG = {
    "current_apa": "Wood",
    "current_wirenumber": 0,
    "current_layer": "V",
    "sound_device_index": 0,
    "device_samplerate": 44100,
    "noise_threshold": 0.01
}

class Application:
    def __init__(self):
        try:
            with open("config.json", "r") as config_file:
                config_data = json.load(config_file)
        except FileNotFoundError:
            config_data = {}


        # Merge default values with loaded values
        config_data = DEFAULT_CONFIG | config_data

        self.current_apa = APA(config_data["current_apa"])
        self.current_wire_number = config_data["current_wirenumber"]
        self.current_layer = config_data["current_layer"]
        self.sound_device_index = config_data["sound_device_index"]
        self.device_samplerate = config_data["device_samplerate"]
        self.noise_threshold = config_data["noise_threshold"]

        self.current_apa.load_calibration_from_json()
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.csv_filename = f"frequency_log_{timestamp}.csv"

    def save_to_config(self):
        config_data = {
            # save the name string, a unique identifier
            "current_apa": self.current_apa.name,
            "current_wirenumber": self.current_wire_number,
            "current_layer": self.current_layer,
            "sound_device_index": self.sound_device_index,
            "device_samplerate": self.device_samplerate,
            "noise-threshold": self.noise_threshold
        }
        with open("config.json", "w") as config_file:
            json.dump(config_data, config_file)

    def handle_change_variables(self):
        print("Select a variable to change:")
        variables = list(self.__dict__.keys())
        for i, var in enumerate(variables, 1):
            print(f"{i}. {var}")

        selection_index = input(
            "Enter the number corresponding to the variable you want to change: ")
        try:
            selection_index = int(selection_index)
            if 1 <= selection_index <= len(variables):
                selected_var = variables[selection_index - 1]
                new_value = input(f"Enter new value for {selected_var}: ")
                setattr(self, selected_var, new_value)
                self.save_to_config()
            else:
                print("Invalid selection. Please enter a number between 1 and",
                      len(variables))
        except ValueError:
            print("Invalid input. Please enter a number.")

    def handle_select_device(self):
        print("Available audio devices:")
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            print(f"{i+1}. {device['name']}")
        while True:
            try:
                selection = int(input("Select an audio device: "))
                if 1 <= selection <= len(sd.query_devices()):
                    self.sound_device_index = selection - 1
                    self.device_samplerate = devices[selection -
                                                     1]['default_samplerate']
                else:
                    print("Invalid selection. Please enter a number between 1 and", len(
                        sd.query_devices()))
            except ValueError:
                print("Invalid input. Please enter a number.")

    def _record_audio(self, duration):
        print("Recording...")
        with sd.InputStream(device=self.sound_device_index, channels=1, samplerate=self.device_samplerate, dtype='float32') as stream:
            audio_data = stream.read(int(duration * self.device_samplerate))
        print("Recording finished.")
        return audio_data

    def _get_pitch_from_audio(self, audio_data):
        time, frequency, confidence, activation = crepe.predict(
            audio_data, self.device_samplerate, viterbi=False)
        return frequency[np.argmax(confidence)], confidence

    def _plot_waveform_and_fft(self, audio_signal, fundamental_freq, fundamental_confidence):
        # Plot waveform and FFT
        plt.figure(figsize=(10, 6))

        # Plot waveform
        plt.subplot(2, 1, 1)
        plt.plot(audio_signal)
        plt.title('Recorded Waveform')
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')

        # Plot FFT
        plt.subplot(2, 1, 2)
        fft = np.abs(np.fft.fft(audio_signal))
        freqs = np.fft.fftfreq(len(audio_signal), 1/self.device_samplerate)
        plt.plot(freqs[:len(freqs)//2], fft[:len(fft)//2])

        # Add vertical line at the fundamental frequency
        plt.axvline(fundamental_freq, color='r', linestyle='--',
                    label='Fundamental Frequency')

        # Add confidence as a caption
        plt.text(0.95 * fundamental_freq, fft.max(), f'Confidence: {fundamental_confidence:.2f}',
                 color='r', fontsize=10, verticalalignment='bottom', horizontalalignment='right')
        plt.title('FFT')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.xlim(0, 3000)
        plt.legend()

        plt.tight_layout()
        plt.show(block=False)

    def handle_record(self):
        print("\nListening...")
        end_time = datetime.now() + timedelta(seconds=60)
        audio_signal = self._record_audio(0.05)
        while not np.max(np.abs(audio_signal)) >= self.noise_threshold:
            if datetime.now() <= end_time:
                print("Recording timed out.")
                break
            audio_signal = self._record_audio(0.05)
        audio_signal = self._record_audio(0.5)
        frequency, confidence = self._get_pitch_from_audio(audio_signal)
        self._plot_waveform_and_fft(audio_signal, frequency, confidence)
        with open(self.csv_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([self.current_wire_number, confidence, frequency])

    def _goto_wire(self, wire_number):
        current_x, current_y = tensiometer.get_xy()
        target_x, target_y = apa.calibration[self.current_layer][wire_number]
        if (current_x-target_x) > RIB_SPACING:
            tensiometer.goto_xy(current_x, 0)
            tensiometer.goto_xy(target_x, 0)
        tensiometer.goto_xy(target_x, target_y)
        self.current_wire_number = wire_number

    def handle_goto_wire(self):
        wire_number = input("Enter the wire you wish to go to.")
        self._goto_wire(wire_number)

    def handle_calibration(self):
        self.current_apa.calibrate()
        self.current_apa.save_calibration_to_json()

    def handle_quit(self):
        tensiometer.__exit__
        sys.exit()

    def getch(self):
        """
        Get a single character from the terminal without requiring the user to press Enter.
        """
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            char = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return char

    actions = {
        'd': handle_select_device,
        'r': handle_record,
        'w': handle_goto_wire,
        'c': handle_calibration,
        'p': handle_change_variables,
        'q': handle_quit
    }


if __name__ == "__main__":
    app = Application()
    app.initialize_from_config()
    with Tensiometer() as tensiometer:
        while True:
            print("\nPress 'd' to display available sound devices, 'r' to pluck the string and record audio, 'w' to go to a wire, 'c to calibrate', 'p' to change parameters, 'q' to quit.")
            key = app.getch().lower()
            if action := actions.get(key):
                action()
            else:
                print("Invalid input. Press 'd', 'r', 'w', 'c', 'p' or 'q'.")
