# tensiometer.py
import requests
from maestro import Controller
import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
import utilities
import threading
import time

IDLE_MOVE_TYPE = 0
IDLE_STATE = 1
XY_MOVE_TYPE = 2
XY_STATE = 3


class Tensiometer:
    def __init__(
        self,
        apa_name="",
        tension_server_url="http://192.168.137.1:5000",
        ttyStr="/dev/ttyACM1",
        sound_card_name="USB PnP Sound Device",
        samples_per_wire=10,
        record_duration=0.1,
        confidence_threshold=0.5,
        delay_after_plucking=0.2,
        wiggle_type="gaussian",
        wiggle_step=0.2,
        save_audio=True,
        timeout=10,
        use_servo=False,
        use_wiggle=False,
    ):
        """
        Initialize the controller, audio devices, and check web server connectivity more concisely.
        """
        self.apa_name = apa_name
        self.tension_server_url = tension_server_url
        self.samples_per_wire = samples_per_wire
        self.record_duration = record_duration
        self.confidence_threshold = confidence_threshold
        self.delay_after_plucking = delay_after_plucking
        self.wiggle_type = wiggle_type
        self.wiggle_step = wiggle_step
        self.save_audio = save_audio
        self.timeout = timeout
        self.stop_servo_event = threading.Event()
        self.stop_wiggle_event = threading.Event()
        self.use_servo = use_servo
        self.use_wiggle = use_wiggle

        if wiggle_type == "gaussian":
            self.wiggle = utilities.gaussian_wiggle
        else:
            self.wiggle = utilities.stepwise_wiggle
        if use_servo:
            try:
                self.maestro = Controller(ttyStr)
                self.servo_state = 0
                self.maestro.runScriptSub(1)
            except Exception as e:
                print(f"Failed to initialize Maestro controller: {e}")
                exit(1)

        try:
            device_info = sd.query_devices()
            self.sound_device_index = next(
                (
                    index
                    for index, d in enumerate(device_info)
                    if sound_card_name in d["name"]
                ),
                None,
            )
            if self.sound_device_index is not None:
                self.sample_rate = device_info[self.sound_device_index][
                    "default_samplerate"
                ]
                print(f"Using USB PnP Sound Device (hw:{self.sound_device_index},0)")
            else:
                print("Couldn't find USB PnP Sound Device.")
                print(device_info)
                exit(1)
        except Exception as e:
            print(f"Failed to initialize audio devices: {e}")
            exit(1)

        if not self.is_web_server_active(tension_server_url):
            print(
                "Failed to connect to the tension server.\nMake sure you are connected to Dunes and the server is running."
            )
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

    def get_xy(self):
        """Get the current position of the tensioning system."""
        x = self.read_tag("X_axis.ActualPosition")
        y = self.read_tag("Y_axis.ActualPosition")
        return x, y

    def get_state(self) -> dict[str, list]:
        """Get the current state of the tensioning system."""
        return self.read_tag("STATE")

    def get_movetype(self) -> int:
        """Get the current move type of the tensioning system."""
        movetype = self.read_tag("MOVE_TYPE")
        return movetype

    def goto_xy(self, x_target: float, y_target: float,speed= 50):
        """Move the winder to a given position."""
        # current_x, current_y = self.get_xy()
        if x_target < 0 or x_target > 7174 or y_target < 0 or y_target > 2680:
            print(
                f"Motion target {x_target},{y_target} out of bounds. Please enter a valid position."
            )
            return False

        self.write_tag("MOVE_TYPE", IDLE_MOVE_TYPE)
        self.write_tag("STATE", IDLE_STATE)
        self.write_tag("X_POSITION", x_target)
        self.write_tag("Y_POSITION", y_target)
        self.write_tag("MOVE_TYPE", XY_MOVE_TYPE)
        current_x, current_y = self.get_xy()
        while (abs(current_x - x_target)) > 0.02 and (abs(current_y - y_target)) > 0.02:
            current_x, current_y = self.get_xy()
        return True

    def increment(self, increment_x, increment_y):
        x, y = self.get_xy()
        self.goto_xy(x + increment_x, y + increment_y)

    def wiggle_loop(self):
        x,y = self.get_xy()
        while not self.stop_wiggle_event.is_set():
            self.goto_xy(x,y+self.wiggle_step)
            self.goto_xy(x, y-self.wiggle_step)

    def record_audio(self, duration, plot=False, normalize=True):
        """Record audio for a given duration and sample rate and normalize it to the range -1 to 1. Optionally plot the waveform."""
        try:
            audio_data = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype="float64",
            )
            sd.wait()  # Wait until recording is finished
            audio_data = audio_data.flatten()  # Flatten the audio data to a 1D array
            # Normalize the audio data to the range -1 to 1
            if normalize:
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

    def servo_loop(self):
        while not self.stop_servo_event.is_set():
            self.maestro.runScriptSub(0)
            time.sleep(0.4)

    def read_tag(self, tag_name):
        """
        Send a GET request to read the value of a PLC tag.
        """
        url = f"{self.tension_server_url}/tags/{tag_name}"
        # print(f"Attempting to read from URL: {url}")  # Debugging statement
        try:
            response = requests.get(url)
            if response.status_code == 200:
                # print(response.json())
                return response.json()[tag_name][1]
            else:
                return {
                    "error": "Failed to read tag",
                    "status_code": response.status_code,
                }
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}

    def write_tag(self, tag_name, value):
        """
        Send a POST request to write a value to a PLC tag.
        """
        url = f"{self.tension_server_url}/tags/{tag_name}"
        # print(f"Attempting to write to URL: {url}")  # Debugging statement
        payload = {"value": value}
        try:
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "error": "Failed to write tag",
                    "status_code": response.status_code,
                }
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
