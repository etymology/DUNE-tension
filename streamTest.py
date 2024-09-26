import pyaudio
import aubio
import numpy as np
import time
from Tensiometer import Tensiometer  # Import the Tensiometer class

# Set constants (as defined by user)
LARGE_BUFFER_SIZE = 2**12  # Larger buffer size to capture low-frequency signals
CHANNELS = 1  # Mono audio
RATE = 44100  # Sample rate in Hz
DEVICE_NAME = "USB PnP Sound Device"  # Name of the device to search for
PITCH_METHOD = "yin"  # Using 'yin' for pitch detection
HOP_SIZE = LARGE_BUFFER_SIZE // 2  # Increase hop size proportionally to buffer size
PITCH_PERIOD = 0.1  # Pitch output every 0.1 seconds
SILENCE_THRESHOLD = -60  # Silence threshold in dB
GAIN = 3.0  # Gain factor to amplify the signal (2.0 = 2x amplification)
N_CONSECUTIVE = 3  # Number of consecutive values within 5% to consider pitch stable
CONFIDENCE_THRESHOLD = 0.7  # Confidence threshold for stable pitch

# Initialize PyAudio
p = pyaudio.PyAudio()

# Find the USB PnP Sound Device
def find_input_device(device_name):
    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        if device_name in device_info.get('name'):
            return i
    return None

device_index = find_input_device(DEVICE_NAME)
if device_index is None:
    print(f"Error: Could not find device with name '{DEVICE_NAME}'")
    exit(1)

print(f"Using device: {DEVICE_NAME}")

# Open a stream with PyAudio using the USB PnP Sound Device
stream = p.open(format=pyaudio.paFloat32,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=LARGE_BUFFER_SIZE)

# Initialize aubio pitch detection with larger buffer size
pitch_o = aubio.pitch(PITCH_METHOD, LARGE_BUFFER_SIZE, HOP_SIZE, RATE)
pitch_o.set_unit("Hz")
pitch_o.set_silence(SILENCE_THRESHOLD)  # Set silence threshold

# Initialize Tensiometer instance
t = Tensiometer()  # Create instance of the Tensiometer class

# Buffer for storing the last few pitch values to check stability
pitch_buffer = []

def check_pitch_stability(pitch_buffer, tolerance=0.05):
    """
    Check if the last N_CONSECUTIVE values in the buffer are within tolerance.
    Tolerance is set to 5% (0.05).
    """
    if len(pitch_buffer) < N_CONSECUTIVE:
        return False
    # Compare the values in the buffer
    min_pitch = min(pitch_buffer)
    max_pitch = max(pitch_buffer)
    return (max_pitch - min_pitch) / min_pitch <= tolerance

try:
    while True:
        # Read from the audio stream
        audio_buffer = stream.read(LARGE_BUFFER_SIZE, exception_on_overflow=False)
        signal = np.frombuffer(audio_buffer, dtype=np.float32)

        # Amplify the signal
        amplified_signal = signal * GAIN

        # Process the amplified signal in chunks of HOP_SIZE
        for i in range(0, len(amplified_signal), HOP_SIZE):
            hop_signal = amplified_signal[i:i + HOP_SIZE]

            if len(hop_signal) == HOP_SIZE:  # Ensure the signal is of the correct size
                # Get pitch and confidence
                pitch = pitch_o(hop_signal)[0]
                confidence = pitch_o.get_confidence()

                # Only proceed if confidence is above the threshold
                if confidence > CONFIDENCE_THRESHOLD:
                    # Add the pitch to the buffer and keep only the last N_CONSECUTIVE values
                    pitch_buffer.append(pitch)
                    if len(pitch_buffer) > N_CONSECUTIVE:
                        pitch_buffer.pop(0)

                    # Check if the pitch values in the buffer are stable
                    if len(pitch_buffer) >= N_CONSECUTIVE and check_pitch_stability(pitch_buffer):
                        # Calculate the average pitch
                        avg_pitch = np.mean(pitch_buffer)

                        # Get x, y from the tensiometer
                        x, y = t.get_xy()

                        # Print the average pitch and Y value
                        print(f"Y: {y}, Frequency: {avg_pitch:.2f} Hz")

                        # Reset pitch buffer after printing
                        pitch_buffer.clear()

except KeyboardInterrupt:
    print("Stopping...")

# Cleanup
stream.stop_stream()
stream.close()
p.terminate()
