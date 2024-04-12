import sounddevice as sd
import crepe
import numpy as np
import csv
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
import tensorflow as tf
from maestro import Controller
from time import sleep
from gCodeDriver import APA
import sys
import tty
import termios

# Suppress TensorFlow messages except for errors
tf.get_logger().setLevel('ERROR')


class APA_frequencyTest(APA):
    def __init__(self, layer, selected_device, recording_duration, cfg="Untitled_cfg", ini_wirenum=None):
        current_apa.__init__(self, layer, cfg, ini_wirenum)
        self.selected_device = selected_device
        self.recording_duration = 0.0

    def set_recording_duration(self):
        try:
            duration = float(input("Enter recording duration in seconds: "))
            if duration <= 0:
                raise ValueError(
                    "Recording duration must be a positive number.")
            self.recording_duration = duration
        except ValueError as e:
            print("Invalid input:", e)


def find_audio_devices():
    devices = sd.query_devices()
    print("Available audio devices:")
    for i, device in enumerate(devices):
        print(f"{i + 1}. {device['name']}")

    while True:
        try:
            choice = int(
                input("Enter the number of the audio device you want to use: "))
            if 1 <= choice <= len(devices):
                print(devices[choice - 1])
                return devices[choice - 1]
            else:
                print("Invalid choice. Please enter a number within the range.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def get_pitch_from_audio(signal, sr):
    print("Getting pitch...")
    # Convert the list to a numpy array
    signal_np = np.array(signal)
    # Extract pitch using CREPE
    time, frequency, confidence, _ = crepe.predict(
        signal_np, sr, viterbi=False)

    # Find the fundamental frequency with the highest confidence
    max_confidence_index = np.argmax(confidence)
    fundamental_freq = frequency[max_confidence_index]
    fundamental_confidence = confidence[max_confidence_index]

    return fundamental_freq, fundamental_confidence


def record_audio(sr, duration):
    num_samples = int(duration * sr)
    audio = sd.rec(num_samples, samplerate=sr, channels=1, blocking=True)
    return audio.flatten()


def detect_sound(audio_signal, threshold):
    # Detect sound based on energy threshold
    return np.mean(audio_signal) >= threshold


def pluck_string(controller: Controller):
    """
    controller: an instance of maestro.Controller
    """
    try:
        controller.runScriptSub(0)  # move zip tie down
    except Exception:
        print("couldn't move servo")


def log_frequency_and_wire_number(frequency, confidence, wire_number, filename):
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([wire_number, confidence, frequency])


def set_recording_duration():
    try:
        duration = float(input("Enter recording duration in seconds: "))
        if duration <= 0:
            raise ValueError("Recording duration must be a positive number.")
        return duration
    except ValueError as e:
        print("Invalid input:", e)
        return None


def plot_waveform_and_fft(audio_signal, sr, fundamental_freq, fundamental_confidence):
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
    freqs = np.fft.fftfreq(len(audio_signal), 1/sr)
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


def getch():
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


if __name__ == "__main__":
    settings = load_settings()

    # Load the most recent audio device and wire number
    devices = list_audio_devices()
    selected_device = settings.get('selected_device', 0)
    recording_duration = settings.get('recording_duration', 0.5)
    # Get current wire number from settings
    current_wire_number = settings.get('current_wire_number', 0)
    noiseThreshold = 0.05  # Adjust the threshold as needed

    # Generate CSV filename with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_filename = f"frequency_log_{timestamp}.csv"

    # Initialize the maestro servo controller
    maestro6 = Controller()

    print(
        f"\nStarting with wire number {current_wire_number} and device {selected_device['name']}")

    while True:
        print("\nPress 'd' to display available sound devices, 'r' to pluck the string and record audio, 'w' to move servo to a wire number, '+' or '-' to move up or down, 'm' to set recording duration, 'q' to quit.")
        key = getch().lower()

        if key == 'd':  # 'd' key pressed
            current_apa.selected_device = find_audio_devices()
            print(
                f"Selected audio device: {current_apa.selected_device['name']}")

        elif key == 'r':  # 'r' key pressed
            pluck_string(maestro6)
            print("\nWaiting for audio... (press r to override)")
            while True:
                key = getch().lower()
                audio_signal = record_audio(
                    int(selected_device['default_samplerate']), .1)
                if detect_sound(audio_signal, noiseThreshold) or key == "r":
                    print("Recording...")
                    audio_signal = record_audio(
                        int(selected_device['default_samplerate']), recording_duration)
                    break  # Start recording when sound is detected
            audio_signal = record_audio(
                int(selected_device['default_samplerate']), recording_duration)
            fundamental_freq, fundamental_confidence = get_pitch_from_audio(
                audio_signal, int(selected_device['default_samplerate']))
            print(
                f"Fundamental Frequency: {fundamental_freq} Hz, Confidence: {fundamental_confidence}")
            plot_waveform_and_fft(audio_signal, int(
                selected_device['default_samplerate']), fundamental_freq, fundamental_confidence)
            log_prompt = input(
                f"Do you want to log the frequency? [wire number {current_wire_number}](y/n): ")
            if log_prompt.lower() == 'y':
                log_frequency_and_wire_number(
                    fundamental_freq, fundamental_confidence, current_wire_number, csv_filename)
                print("Frequency logged.")
            elif log_prompt.lower() == 'n':
                print("Frequency not logged.")
            plt.close()

        elif key == 'w':  # 'w' key pressed
            wire_number = int(input("Enter the wire number: "))
            move_servo_to_wire(wire_number)
            current_wire_number = wire_number
            # Update current wire number in settings
            update_settings('current_wire_number', current_wire_number)
            print(f"Robot moved to wire number {wire_number}.")

        elif key == '=':  # 'u' key pressed
            move_servo_to_wire(current_wire_number+1)
            current_wire_number = current_wire_number+1
            # Update current wire number in settings
            update_settings('current_wire_number', current_wire_number)
            print(f"Robot moved up one wire to {current_wire_number}.")

        elif key == '-':  # 'd' key pressed
            current_apa.move_to_wire(current_apa.wirenum-1)
            print(f"Robot moved up one wire to {current_apa.wirenum}.")

        elif key == 'm':  # 'm' key pressed
            current_apa.set_recording_duration()

        elif key == 'q':  # 'q' key pressed
            print("Quitting...")
            maestro6.close()
            break

        else:
            print("Invalid input. Press 'd', 'r', 'w', 'm', or 'q'.")

# save_settings(settings)
