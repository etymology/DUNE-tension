# tensiometer.py
import requests
from maestro import Controller
import sounddevice as sd
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import utilities

IDLE_MOVE_TYPE = 0
IDLE_STATE = 1
XY_MOVE_TYPE = 2
XY_STATE = 3


class Tensiometer:
    def __init__(self, apa_name="",
                 tension_server_url='http://192.168.137.1:5000',
                 ttyStr='/dev/ttyACM0',
                 sound_card_name="USB PnP Sound Device",
                 tries_per_wire=20,
                 record_duration=0.1,
                 confidence_threshold=0.8,
                 max_frequency=3000,
                 delay_after_plucking=0.1,
                 wiggle_type = 'gaussian',
                 wiggle_step = 0.1):
        """
        Initialize the controller, audio devices, and check web server connectivity more concisely.
        """
        self.apa_name = apa_name
        self.tension_server_url = tension_server_url
        self.tries_per_wire = tries_per_wire
        self.record_duration = record_duration
        self.confidence_threshold = confidence_threshold
        self.max_frequency = max_frequency
        self.delay_after_plucking = delay_after_plucking
        self.wiggle_type = wiggle_type
        self.wiggle_step = wiggle_step

        if wiggle_type == 'gaussian':
            self.wiggle = utilities.gaussian_wiggle
        else:
            self.wiggle = utilities.stepwise_wiggle
        try:
            self.maestro = Controller(ttyStr=ttyStr)
            self.servo_state = 0
            self.maestro.runScriptSub(1)
        except Exception as e:
            print(f"Failed to initialize Maestro controller: {e}")
            exit(1)

        try:
            device_info = sd.query_devices()
            self.sound_device_index = next(
                (index for index, d in enumerate(device_info) if sound_card_name in d["name"]),
                None
            )
            if self.sound_device_index is not None:
                self.samplerate = device_info[self.sound_device_index]["default_samplerate"]
                print(f"Using USB PnP Sound Device (hw:{self.sound_device_index},0)")
            else:
                print("Couldn't find USB PnP Sound Device.")
                print(device_info)
                exit(1)
        except Exception as e:
            print(f"Failed to initialize audio devices: {e}")
            exit(1)

        if not self.is_web_server_active(tension_server_url):
            print("Failed to connect to the tension server.\nMake sure you are connected to Dunes and the server is running.")
            exit(1)
        print("Connected to the tension server.")

    def is_web_server_active(self, url):
        """
        Check if a web server is active by sending a HTTP GET request.
        """
        try:
            return 200 <= requests.get(url, timeout=3).status_code < 500
        except requests.RequestException as e:
            print(f"An error occurred while checking the server: {e}")
            return False

    def __exit__(self):
        self.maestro.close()

    def pluck_string(self):
        if not self.maestro.faulted:
            self.maestro.runScriptSub(0)
        else:
            print("Maestro is faulted. Cannot pluck the string.")

    def get_xy(self) -> tuple[float, float]:
        """ Get the current position of the tensioning system. """
        x = self.read_tag("X_axis.ActualPosition")
        y = self.read_tag("Y_axis.ActualPosition")
        return x, y

    def get_state(self) -> dict[str, list]:
        """ Get the current state of the tensioning system. """
        return self.read_tag("STATE")

    def enter_and_exit_state_movetype(self,target_state: int, target_move_type: int):
        """ Wait until the tensioning system is in the idle state. """
        while self.read_tag("STATE") != target_state and self.read_tag("MOVE_TYPE") != target_move_type:
            pass
        while self.read_tag("STATE") != IDLE_STATE and self.read_tag("MOVE_TYPE") != IDLE_MOVE_TYPE:
            pass

    def get_movetype(self) -> int:
        """ Get the current move type of the tensioning system. """
        movetype = self.read_tag("MOVE_TYPE")
        return movetype

    def goto_xy(self, x_target: float, y_target: float):
        """ Move the winder to a given position. """
        if x_target >=0 and x_target <=7174 and y_target >=0 and y_target <=2680:
            self.write_tag("MOVE_TYPE", IDLE_MOVE_TYPE)
            self.write_tag("STATE", IDLE_STATE)
            self.write_tag("X_POSITION", x_target)
            self.write_tag("Y_POSITION", y_target)
            self.write_tag("MOVE_TYPE", XY_MOVE_TYPE)
            current_x, current_y = self.get_xy()
            while (abs(current_x-x_target))>0.05 and (abs(current_y-y_target))>0.05:
                current_x, current_y = self.get_xy()
            return True
        else:
            print("Target out of bounds. Please enter a valid position.")
            return False
        # TODO: Add a timeout to prevent infinite loops, and return False if the timeout is reached.
        # TODO: Prevent moves across combs
        # TODO: Prevent moves outside of the valid range
        # TODO: IS this the right way to check if the move is done?
        
    
    def increment(self, increment_x, increment_y):
        x, y = self.get_xy()
        self.goto_xy(x+increment_x, y+increment_y)

    def detect_sound(self, audio_signal, threshold):
        # Detect sound based on energy threshold
        return np.max(np.abs(audio_signal)) >= float(threshold)

    def record_audio(self, duration: float):
        """Record audio for a given duration using the selected audio device."""
        with sd.InputStream(
            device=self.sound_device_index,
            channels=1,
            samplerate=self.samplerate,
            dtype="float32",
        ) as stream:
            audio_data = stream.read(int(duration * self.samplerate))[0]
        return audio_data.flatten()
    
    def record_audio_trigger(self, sound_length: float,noise_threshold = 0.01):
        start_time = datetime.now()
        while True:
            audio_data = self.record_audio(0.005)
            if self.detect_sound(audio_data, noise_threshold):
                recorded_audio =  self.record_audio(sound_length)
                return recorded_audio
            elif datetime.now() > start_time + timedelta(seconds=.5):
                print("No sound detected. Timed out.")
                return None

    def record_audio_normalize(self,duration, plot=False):
        """Record audio for a given duration and sample rate and normalize it to the range -1 to 1. Optionally plot the waveform."""
        try:
            audio_data = sd.rec(int(duration * self.samplerate), samplerate=self.samplerate, channels=1, dtype='float64')
            sd.wait()  # Wait until recording is finished
            audio_data = audio_data.flatten()   # Flatten the audio data to a 1D array
            # Normalize the audio data to the range -1 to 1
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data = audio_data / max_val

            # Plot the waveform if plot is True
            if plot:
                plt.figure(figsize=(10, 4))
                plt.plot(audio_data)
                plt.title("Recorded Audio Waveform")
                plt.xlabel("Sample Index")
                plt.ylabel("Amplitude")
                plt.grid()
                plt.show()

            return audio_data
        except Exception as e:
            print(f"An error occurred while recording audio: {e}")
            return None

        
    def servo_toggle(self):
        if self.servo_state == 0:
            self.servo_state = 1
            self.maestro.runScriptSub(0)
        elif self.servo_state == 1:
            self.servo_state = 0
            self.maestro.runScriptSub(1)

    def read_tag(self, tag_name):
        """
        Send a GET request to read the value of a PLC tag.
        """
        url = f"{self.tension_server_url}/tags/{tag_name}"
        # print(f"Attempting to read from URL: {url}")  # Debugging statement
        try:
            response = requests.get(url)
            if response.status_code == 200:
                return response.json()[tag_name][1]
            else:
                return {'error': 'Failed to read tag', 'status_code': response.status_code}
        except requests.exceptions.RequestException as e:
            return {'error': str(e)}
    def write_tag(self, tag_name, value):
        """
        Send a POST request to write a value to a PLC tag.
        """
        url = f"{self.tension_server_url}/tags/{tag_name}"
        # print(f"Attempting to write to URL: {url}")  # Debugging statement
        payload = {'value': value}
        try:
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                return response.json()
            else:
                return {'error': 'Failed to write tag', 'status_code': response.status_code}
        except requests.exceptions.RequestException as e:
            return {'error': str(e)}
