import requests
from maestro import Controller
import sounddevice as sd
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import crepe
import os
import csv
from typing import Tuple, Callable, Dict
from time import sleep

TENSION_SERVER_URL = 'http://192.168.137.1:5000'
IDLE_MOVE_TYPE = 0
IDLE_STATE = 1
XY_MOVE_TYPE = 2
XY_STATE = 3
ZONE_1 = 1500
FIRST_WIRE_Y = 192.3
WIRE_SPACING = 2300/480
USB_SOUND_CARD_NAME = "USB PnP Sound Device"  #: Audio (hw:1,0)"

MAX_FREQUENCY = 2000  # Maximum frequency to consider for FFT
RECORDING_DURATION = 1.0  # Duration of audio recording in seconds
CONFIDENCE_THRESHOLD = 0.80
NOISE_THRESHOLD = 0.005
TRIES_PER_WIRE = 8
DELAY_AFTER_PLUCKING = 0.8 # Delay after plucking the string before recording audio to avoid noise due to servo movement
AnalysisFuncType = Callable[[np.ndarray, int], Tuple[float, float]]
MAX_FREQUENCY = 5000  # Example maximum frequency threshold
MULTIPLE_TOLERANCE = 0.03  # Tolerance for checking multiples (5%)
NUMBER_OF_PEAKS = 5  # Number of peaks to consider in the FFT spectrum

class Tensiometer:
    def __init__(self):
        """
        Initialize the controller, audio devices, and check web server connectivity more concisely.
        """
        try:
            self.maestro = Controller()
        except Exception as e:
            print(f"Failed to initialize Maestro controller: {e}")
            exit(1)

        try:
            device_info = sd.query_devices()
            self.sound_device_index = next(
                (index for index, d in enumerate(device_info) if USB_SOUND_CARD_NAME in d["name"]),
                None
            )
            if self.sound_device_index is not None:
                self.samplerate = device_info[self.sound_device_index]["default_samplerate"]
                print(f"Using USB PnP Sound Device (hw:{self.sound_device_index},0)")
            else:
                print("Couldn't find USB PnP Sound Device.")
                exit(1)
        except Exception as e:
            print(f"Failed to initialize audio devices: {e}")
            exit(1)

        if not self.is_web_server_active(TENSION_SERVER_URL):
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
        x = read_tag("X_axis.ActualPosition")
        y = read_tag("Y_axis.ActualPosition")
        return x, y

    def get_state(self) -> dict[str, list]:
        """ Get the current state of the tensioning system. """
        return read_tag("STATE")

    def enter_and_exit_state_movetype(self,target_state: int, target_move_type: int):
        """ Wait until the tensioning system is in the idle state. """
        while read_tag("STATE") != target_state and read_tag("MOVE_TYPE") != target_move_type:
            pass
        while read_tag("STATE") != IDLE_STATE and read_tag("MOVE_TYPE") != IDLE_MOVE_TYPE:
            pass

    def get_movetype(self) -> int:
        """ Get the current move type of the tensioning system. """
        movetype = read_tag("MOVE_TYPE")
        return movetype

    def goto_xy(self, x_target: float, y_target: float):
        state = self.get_state()
        movetype = self.get_movetype()
        if state == IDLE_STATE and movetype == IDLE_MOVE_TYPE:
            """ Move the winder to a given position. """
            write_tag("X_POSITION", x_target)
            write_tag("Y_POSITION", y_target)
            write_tag("MOVE_TYPE", XY_MOVE_TYPE)
            current_x, current_y = self.get_xy()
            while (abs(current_x-x_target))>0.1 and (abs(current_y-y_target))>0.1:
                current_x, current_y = self.get_xy()
        else:
            print(f"Cannot move the system in state {state} and movetype {movetype}.")

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
    
    def record_audio_trigger(self, sound_length: float):
        start_time = datetime.now()
        while True:
            audio_data = self.record_audio(0.005)
            if self.detect_sound(audio_data, NOISE_THRESHOLD):
                # recorded_audio =  normalize_audio(self.record_audio(sound_length))
                recorded_audio =  self.record_audio(sound_length)
                # plt.plot(recorded_audio)
                plt.show()
                # sd.play(recorded_audio, samplerate=self.samplerate)
                # sd.wait()
                return recorded_audio
            elif datetime.now() > start_time + timedelta(seconds=2):
                print("No sound detected. Timed out.")
                return None

def read_tag(tag_name,base_url = TENSION_SERVER_URL):
    """
    Send a GET request to read the value of a PLC tag.
    """
    url = f"{base_url}/tags/{tag_name}"
    # print(f"Attempting to read from URL: {url}")  # Debugging statement
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()[tag_name][1]
        else:
            return {'error': 'Failed to read tag', 'status_code': response.status_code}
    except requests.exceptions.RequestException as e:
        return {'error': str(e)}
    
def write_tag(tag_name, value,base_url = TENSION_SERVER_URL):
    """
    Send a POST request to write a value to a PLC tag.
    """
    url = f"{base_url}/tags/{tag_name}"
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

def analyze_wire(t: Tensiometer, wire_number, wire_x, wire_y):
    def wiggle_generator():
        i = 0  # Start with the first element index
        while True:  # Infinite loop
            yield (-1)**(i // 2 + 1) * (i // 2 + 1) * 0.1
            i += 1
    analysis_methods: Dict[str, AnalysisFuncType] = {
        "crepe": get_pitch_crepe,
        "naive_fft": get_pitch_naive_fft,
        "autocorrelation": get_pitch_autocorrelation,
    }
    wg = wiggle_generator()
    for _ in range(TRIES_PER_WIRE):
        t.goto_xy(wire_x, wire_y + next(wg))
        t.pluck_string()
        sleep(DELAY_AFTER_PLUCKING)
        audio_signal = t.record_audio_trigger(RECORDING_DURATION)
        if audio_signal is not None:
            analysis = {method: func(audio_signal, t.samplerate) for method, func in analysis_methods.items()}
            for method, (frequency, confidence) in analysis.items():
                if confidence != 0 and frequency < MAX_FREQUENCY:
                    print(f"Wire number {wire_number} has frequency {frequency} Hz with confidence {confidence} using {method}.")
                    log_frequency_data(frequency, confidence, wire_number, algorithm=method)
            if analysis["crepe"][1] > CONFIDENCE_THRESHOLD:
                return True
        print(f"Failed to detect a reliable frequency for wire {wire_number}. Retrying...")
    return False

def get_pitch_autocorrelation(audio_data, samplerate, freq_low=10, freq_high=MAX_FREQUENCY):
    """
    Analyzes an audio signal to find the dominant frequency using autocorrelation.

    Parameters:
    - audio_data (np.ndarray): The audio signal data as a numpy array.
    - samplerate (int): The sample rate of the audio signal.
    - freq_low (int): The lower boundary of the frequency range to search (default 10 Hz).
    - freq_high (int): The higher boundary of the frequency range to search (default MAX_FREQUENCY Hz).

    Returns:
    - tuple: (dominant_frequency, confidence) where:
        - dominant_frequency is the detected frequency in Hz.
        - confidence is a measure of the amplitude of the autocorrelation peak relative to others.
    """
    audio_data = audio_data - np.mean(audio_data)

    # Compute the autocorrelation of the signal
    autocorr = np.correlate(audio_data, audio_data, mode='full')
    autocorr = autocorr[len(autocorr)//2:]  # Keep only the second half

    # Determine the maximum lag we consider by the highest frequency of interest
    min_lag = int(samplerate // freq_high)
    max_lag = int(samplerate // freq_low)
    autocorr = autocorr[min_lag:max_lag+1]

    # Find the first peak
    # This simplistic peak finding assumes the first peak is the fundamental frequency
    peak_lag = np.argmax(autocorr) + min_lag

    # Calculate the dominant frequency
    dominant_frequency = samplerate / peak_lag

    # Confidence calculation (peak height relative to the mean of the autocorrelation values)
    confidence = abs(autocorr[peak_lag - min_lag] / np.max(autocorr))

    return dominant_frequency, confidence

def spectral_flatness(magnitude: np.ndarray) -> float:
    """Calculate the spectral flatness of the magnitude spectrum."""
    geometric_mean = np.exp(np.mean(np.log(magnitude + 1e-10)))  # Adding a small constant to avoid log(0)
    arithmetic_mean = np.mean(magnitude)
    return geometric_mean / arithmetic_mean

def get_pitch_naive_fft(audio_data: np.ndarray, samplerate: int) -> tuple[float, float]:
    """Estimate the pitch of the audio data using FFT and return the fundamental frequency f0 and a confidence based on spectral flatness."""
    
    # Compute the FFT of the audio data
    fft_spectrum = np.fft.rfft(audio_data)
    magnitude = np.abs(fft_spectrum)
    freqs = np.fft.rfftfreq(len(audio_data), d=1/samplerate)

    # Consider only frequencies below MAX_FREQUENCY Hz
    valid_indices = freqs < MAX_FREQUENCY
    if not np.any(valid_indices):
        return 0.0, 0.0

    # Find the indices of the highest peaks in the magnitude spectrum
    valid_magnitudes = magnitude[valid_indices]
    valid_freqs = freqs[valid_indices]

    peak_indices = np.argpartition(valid_magnitudes, -NUMBER_OF_PEAKS)[-NUMBER_OF_PEAKS:]
    top_peaks = peak_indices[np.argsort(valid_magnitudes[peak_indices])[::-1]]

    # Get the frequencies of the highest peaks
    top_frequencies = valid_freqs[top_peaks]

    # Plot the time-domain audio data
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(len(audio_data)) / samplerate, audio_data)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title("Time-Domain Audio Data")

    # Plot the frequency-domain magnitude spectrum
    plt.subplot(2, 1, 2)
    plt.plot(freqs, magnitude)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude")
    plt.title("Frequency-Domain Magnitude Spectrum")

    # Check for a fundamental frequency f0 such that other peaks are approximately multiples of f0
    f0 = top_frequencies[0]
    for _, candidate_f0 in enumerate(top_frequencies):
        multiples_found = True
        for f in top_frequencies:
            if f != candidate_f0:
                ratio = f / candidate_f0
                if not (np.abs(ratio - np.round(ratio)) <= MULTIPLE_TOLERANCE):
                    multiples_found = False
                    break
        if multiples_found:
            f0 = candidate_f0
            break

    # Plot a vertical red line at the fundamental frequency f0
    plt.axvline(f0, color='r', linestyle='--', label=f'Fundamental Frequency: {f0:.2f} Hz')
    plt.legend()

    plt.tight_layout()
    plt.show()

    confidence = 1.0 - spectral_flatness(valid_magnitudes)
    return f0, confidence  # Return the fundamental frequency and the confidence

def get_pitch_crepe(audio_data: np.ndarray, samplerate) -> tuple[float, float]:
    """Extract the pitch and confidence from the audio data using CREPE."""
    _, frequencies, confidence, _ = crepe.predict(audio_data, samplerate, model_capacity='medium', viterbi=False)
    
    # Directly find the index of the maximum confidence
    if len(confidence) > 0:
        max_conf_idx = np.argmax(confidence)
        max_frequency = frequencies[max_conf_idx]
        max_confidence = confidence[max_conf_idx]
    else:
        # Handle the case where no confidence values are available
        print("No confidence values available.")
        max_frequency = 0.0
        max_confidence = 0.0

    return max_frequency, max_confidence

def log_frequency_data(frequency, confidence, wire_number, algorithm = ""):
    """
    Log frequency, confidence, and wire number into a CSV file.
    """
    # If filename does not contain a path, use the current working directory
    if not os.path.dirname(filename):
        directory = os.getcwd()  # Get current directory
    else:
        directory = os.path.dirname(filename)
    
    # Ensure the directory exists
    os.makedirs(directory, exist_ok=True)

    # Full path for the file
    full_path = os.path.join(directory, os.path.basename(filename))

    # Open file safely with handling exceptions
    try:
        with open(full_path, 'a', newline='') as csvfile:
            fieldnames = ['wire_number', 'frequency', 'confidence', 'method']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # If file is empty, write header
            if os.stat(full_path).st_size == 0:
                writer.writeheader()
            
            writer.writerow({'wire_number': wire_number, 'frequency': frequency, 'confidence': confidence, 'method': algorithm})
            # print("Data logged successfully.")
    except IOError as e:
        print(f"Error opening or writing to file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
            




dx_diag, dy_diag = -2.724553916, 3.791617987
dx_vert, dy_vert = 0, 5.75

initial_wire_number = 8
final_wire_number = 220

step = 1

diagonal = True

failed_wires = []

if __name__ == "__main__":
    filename = f"data/frequency_data_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    # connect_to_wifi_linux("Dunes", "DunesPassword")
    t = Tensiometer()
    target_x, target_y = t.get_xy()
    movetype = t.get_movetype()
    state = t.get_state()
    print(f"Current position: x={target_x}, y={target_y}\nstate={state}, movetype={movetype}")
    
    wire_number = initial_wire_number
    while wire_number <= final_wire_number:
        if not analyze_wire(t, wire_number, target_x, target_y):
            print(f"Failed to detect a reliable frequency for wire {wire_number}. Moving on to the next wire.")
            log_frequency_data(0, 0, wire_number, algorithm="!!!Failed!!!")
            failed_wires.append(wire_number)
        if diagonal:
            target_x += dx_diag*step
            target_y += dy_diag*step
        else:
            target_x += dx_vert*step
            target_y += dy_vert*step
        wire_number += 1*step

    print(f"Finished scanning from wire {initial_wire_number} to {final_wire_number} with {len(failed_wires)} failed wires.")
    print(f"Failed wires: {failed_wires}")

    