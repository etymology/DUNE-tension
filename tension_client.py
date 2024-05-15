import requests
from maestro import Controller
import sounddevice as sd
from datetime import datetime, timedelta
import numpy as np
from scipy.interpolate import interp1d
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import crepe
import os
import csv
import aubio
import subprocess
from typing import Tuple, Callable, Dict
from time import sleep
import tensorflow as tf

TENSION_SERVER_URL = 'http://192.168.137.1:5000'
MAX_FREQUENCY = 160  # Maximum frequency to consider for FFT
RECORDING_DURATION = 1.0  # Duration of audio recording in seconds
FIRST_WIRE_Y = 192.3
WIRE_SPACING = 2300/480
CONFIDENCE_THRESHOLD = 0.90

IDLE_MOVE_TYPE = 0
IDLE_STATE = 1
XY_MOVE_TYPE = 2
XY_STATE = 3
ZONE_1 = 1500

USB_SOUND_CARD_NAME = "USB PnP Sound Device"  #: Audio (hw:1,0)"

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
-1
def wiggle_generator():
    i = 0  # Start with the first element index
    while True:  # Infinite loop
        yield (-1)**(i // 2 + 1) * (i // 2 + 1) * 0.1
        i += 1

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

def connect_to_wifi_linux(network_name, password):
    """
    Connect to a WiFi network on Linux using nmcli.

    Args:
    network_name (str): The SSID of the WiFi network.
    password (str): The password for the WiFi network.
    """
    try:
        # Command to add and connect to the WiFi network
        command = f"nmcli dev wifi connect '{network_name}' password '{password}'"
        subprocess.run(command, shell=True, check=True)

        print(f"Connected to WiFi network {network_name}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to connect to WiFi network: {str(e)}")

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

def get_pitch_autocorrelation(audio_data, samplerate, freq_low=10, freq_high=160):
    """
    Analyzes an audio signal to find the dominant frequency using autocorrelation.

    Parameters:
    - audio_data (np.ndarray): The audio signal data as a numpy array.
    - samplerate (int): The sample rate of the audio signal.
    - freq_low (int): The lower boundary of the frequency range to search (default 10 Hz).
    - freq_high (int): The higher boundary of the frequency range to search (default 160 Hz).

    Returns:
    - tuple: (dominant_frequency, confidence) where:
        - dominant_frequency is the detected frequency in Hz.
        - confidence is a measure of the amplitude of the autocorrelation peak relative to others.
    """
    # Compute the autocorrelation of the signal
    autocorr = np.correlate(audio_data, audio_data, mode='full')
    autocorr = autocorr[len(autocorr)//2:]  # Keep only the second half

    # Determine the maximum lag we consider by the highest frequency of interest
    max_lag = int(samplerate // freq_low)
    min_lag = int(samplerate // freq_high)
    # print(f"min_lag: {min_lag}, max_lag: {max_lag}")
    autocorr = autocorr[min_lag:max_lag+1]

    # Find the first peak
    # This simplistic peak finding assumes the first peak is the fundamental frequency
    peak_lag = np.argmax(autocorr) + min_lag

    # Calculate the dominant frequency
    dominant_frequency = samplerate / peak_lag

    # Confidence calculation (peak height relative to the mean of the autocorrelation values)
    confidence = autocorr[peak_lag - min_lag] / np.mean(autocorr)

    return dominant_frequency, confidence

def get_pitch_harmonic_series(audio_data, samplerate):
    """
    Detects a harmonic series in the frequency spectrum using the autocorrelation of the FFT magnitude spectrum.

    Parameters:
    - audio_data (np.ndarray): The audio signal data as a numpy array.
    - samplerate (int): The sample rate of the audio signal.

    Returns:
    - tuple: (fundamental_frequency, confidence) where:
        - fundamental_frequency is the detected fundamental frequency of the harmonic series.
        - confidence is a measure of the detectability of the harmonic series in the spectrum.
    """
    # Step 1: Compute FFT of the audio data
    fft_result = np.fft.fft(audio_data)
    magnitude_spectrum = np.abs(fft_result)

    # Step 2: Compute the autocorrelation of the magnitude spectrum
    autocorr = np.correlate(magnitude_spectrum, magnitude_spectrum, mode='full')
    autocorr = autocorr[len(autocorr)//2:]  # Keep only the second half

    # Step 3: Detect peaks in the autocorrelation to determine the fundamental frequency
    # Exclude the zero lag peak by starting from a small lag to skip the direct correlation peak at zero
    peak_lag = np.argmax(autocorr[1:]) + 1

    # Calculate the fundamental frequency
    fundamental_frequency = samplerate / peak_lag

    # Step 4: Confidence estimation
    # We use the peak value relative to the mean of autocorrelation values as confidence
    confidence = autocorr[peak_lag] / np.mean(autocorr)

    return fundamental_frequency, confidence

NOISE_THRESHOLD = 0.001

def get_pitch_scipy(audio_data: np.array, samplerate: int):
    SampHz = samplerate
    fOmega = np.abs(rfft(audio_data))
    Omega = rfftfreq(len(audio_data), d=1/SampHz)

    # Apply a filter and perform cubic interpolation to improve frequency resolution
    funcOmega = interp1d(Omega[Omega < 1000], fOmega[Omega < 1000], kind='cubic')
    interpOmega = np.linspace(0.0, 250, 100000)
    interpfOmega = funcOmega(interpOmega)

    # Find the peaks in the filtered and interpolated data
    indPk, properties = find_peaks(interpfOmega)
    fOmegaPk = interpfOmega[indPk]
    OmegaPk = interpOmega[indPk]

    # Filter peaks to only those above the noise threshold and within 10-160 Hz
    valid_peak_mask = (fOmegaPk > NOISE_THRESHOLD) & (OmegaPk >= 10) & (OmegaPk <= 160)
    filtered_fOmegaPk = fOmegaPk[valid_peak_mask]
    filtered_OmegaPk = OmegaPk[valid_peak_mask]

    # Find the highest peak that has a harmonic peak approximately twice its frequency
    frequency = 0.0  # Default to 0.0 if no valid peak with harmonic is found
    harmonic_tolerance = 5.0  # Frequency tolerance for harmonic detection

    for base_freq, base_amp in zip(filtered_OmegaPk, filtered_fOmegaPk):
        # Check for a harmonic peak near twice the base frequency
        harmonic_target = 2 * base_freq
        harmonic_mask = (filtered_OmegaPk >= harmonic_target - harmonic_tolerance) & \
                        (filtered_OmegaPk <= harmonic_target + harmonic_tolerance)

        if any(harmonic_mask):  # Check if there is any peak within the tolerance of the harmonic frequency
            if base_freq > frequency:  # Update to the highest frequency that meets the condition
                frequency = base_freq

    # Visualization
    # plt.figure(figsize=(12, 6))
    # plt.subplot(121)
    # plt.title("Audio Signal Amplitude")
    # plt.plot(audio_data)
    # plt.xlabel("Sample")
    # plt.ylabel("Amplitude")

    # plt.subplot(122)
    # plt.title("Frequency Spectrum")
    # plt.plot(interpOmega, interpfOmega, label="Interpolated Spectrum")
    # plt.scatter(filtered_OmegaPk, filtered_fOmegaPk, color='red', label="Valid Peaks")
    # if frequency > 0:
    #     plt.scatter([frequency, 2 * frequency], 
    #                 [funcOmega(frequency), funcOmega(2 * frequency)], 
    #                 color='green', label="Selected Peak and Harmonic", zorder=5)
    # plt.xlim(0, 250)
    # plt.xlabel("Frequency (Hz)")
    # plt.ylabel("Amplitude")
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    return frequency, 1.0


def get_pitch_fft(data: np.ndarray, samplerate:int) -> tuple[float, float]:
    """Estimate the pitch of the audio data using FFT and return pitch and a constant confidence, considering only frequencies below 160 Hz."""
    fft_spectrum = np.fft.rfft(data)
    magnitude = np.abs(fft_spectrum)
    freqs = np.fft.rfftfreq(len(data), d=1/samplerate)
    valid_indices = freqs < MAX_FREQUENCY  # Consider only frequencies below 160 Hz
    if not np.any(valid_indices):
        return 0.0, 0.0
    peak_freq_idx = np.argmax(magnitude[valid_indices])
    pitch = freqs[valid_indices][peak_freq_idx]
    return pitch, 1.0

def get_pitch_crepe(audio_data: np.ndarray, samplerate) -> tuple[float, float]:
    """Extract the pitch and confidence from the audio data using CREPE, considering only frequencies below 150 Hz."""
    _, frequency, confidence, _ = crepe.predict(audio_data, samplerate, model_capacity='medium', viterbi=False)
    valid_indices = frequency < 150  # Filter for frequencies below 150 Hz
    max_conf_idx = np.argmax(confidence[valid_indices])
    return frequency[valid_indices][max_conf_idx], confidence[valid_indices][max_conf_idx]

def get_pitch_aubio(audio_data: np.ndarray, sample_rate=44100):
    print(f"audio has shape {audio_data.shape}")
    """Detect the pitch of the audio using aubio and return both pitch and confidence."""
    sample_rate = int(sample_rate)  # Ensure sample_rate is an integer

    # Adjust buffer size as per the requirement from the error
    buffer_size = 512  # Buffer size for pitch detection
    hop_size = buffer_size  # Non-overlapping frames

    # Create aubio pitch detection object
    pitch_o = aubio.pitch("default", buffer_size, hop_size, sample_rate)
    pitch_o.set_unit("Hz")
    pitch_o.set_tolerance(0.8)

    # Analyze the pitch
    pitches = []
    confidences = []

    # Frame-wise analysis
    for i in range(0, len(audio_data), hop_size):
        samples = audio_data[i:i+hop_size]
        if len(samples) < hop_size:
            samples = np.pad(samples, (0, max(0, hop_size - len(samples))), mode='constant', constant_values=(0, 0))

        pitch = pitch_o(samples)[0]
        confidence = pitch_o.get_confidence()

        if confidence > 0.5:  # Consider only pitches with a confidence greater than 0.5
            pitches.append(pitch)
            confidences.append(confidence)

    if pitches:
        pitch_estimate = np.mean(pitches)
        confidence_estimate = np.mean(confidences)
    else:
        pitch_estimate = 0
        confidence_estimate = 0

    return pitch_estimate, confidence_estimate



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
            print("Failed to connect to the tension server. Please make sure the server is running and try again.")
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
            self.enter_and_exit_state_movetype(XY_STATE,XY_MOVE_TYPE)
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
            
def normalize_audio(audio):
    """Normalize the audio to the range -1 to 1."""
    max_amplitude = np.max(np.abs(audio))
    if max_amplitude == 0:
        return audio  # Return the original audio if it's silent to avoid division by zero
    return audio / max_amplitude



TRIES_PER_WIRE = 4
AnalysisFuncType = Callable[[np.ndarray, int], Tuple[float, float]]
DELAY_AFTER_PLUCKING = 0.8 # Delay after plucking the string before recording audio to avoid noise due to servo movement
def analyze_wire(t: Tensiometer,wire_number, wire_x, wire_y):
    analysis_methods: Dict[str, AnalysisFuncType] = {
        "crepe": get_pitch_crepe,
        "scipy": get_pitch_scipy,
        "fft": get_pitch_fft,
        "autocorrelation": get_pitch_autocorrelation,
        "harmonic": get_pitch_harmonic_series,
        "aubio": get_pitch_aubio
    }
    wg = wiggle_generator()
    for _ in range(TRIES_PER_WIRE):
        t.goto_xy(wire_x, wire_y + next(wg))
        t.pluck_string()
        sleep(DELAY_AFTER_PLUCKING)
        audio_signal = t.record_audio_trigger(RECORDING_DURATION)
        if audio_signal is not None:
            analysis = {method: func(audio_signal, t.samplerate) for method, func in analysis_methods.items()}
            crepe_confidence = log_and_display_results(analysis, wire_number)
            if crepe_confidence > 0.9:
                return True
        print(f"Failed to detect a reliable frequency for wire {wire_number}. Retrying...")
    return False

def log_and_display_results(analysis: Dict[str,tuple[float,float]], wire_number):
    for method, (frequency, confidence) in analysis.items():
        if confidence != 0 and frequency < 160:
            print(f"Wire number {wire_number} has frequency {frequency} Hz with confidence {confidence} using {method}.")
            log_frequency_data(frequency, confidence, wire_number, algorithm=method)
    return analysis["crepe"][1]

dx_diag, dy_diag = -2.724553916, 3.791617987
dx_vert, dy_vert = 0, -5.75

initial_wire_number = 8
final_wire_number = 220

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
            target_x += dx_diag
            target_y += dy_diag
        else:
            target_x += dx_vert
            target_y += dy_vert
        wire_number += 1

    print(f"Finished scanning from wire {initial_wire_number} to {final_wire_number} with {len(failed_wires)} failed wires.")
    print(f"Failed wires: {failed_wires}")

    