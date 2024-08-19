import sounddevice as sd
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt

import logging

from scipy.io import wavfile


from pydub import AudioSegment

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

print("tensorflow: %s" % tf.__version__)
# print("librosa: %s" % librosa.__version__)
MAX_ABS_INT16 = 32768.0
SPICE_EXPECTED_SAMPLE_RATE = 16000

spice_model = hub.load("https://tfhub.dev/google/spice/2")


def record_for_spice(
    duration=0.5,
    sample_rate=44100,
    filename="output.wav",
    output_file="converted_audio_file.wav",
):
    """
    Records an audio clip of the specified duration and saves it as a WAV file.

    Parameters:
    duration (int): Duration of the recording in seconds. Default is 1 second.
    sample_rate (int): Sample rate for the recording. Default is 44100 Hz.
    filename (str): The name of the output WAV file. Default is 'output.wav'.

    Returns:
    None
    """
    print("Recording...")
    audio_data = sd.rec(
        int(duration * sample_rate), samplerate=sample_rate, channels=2, dtype="int16"
    )
    sd.wait()  # Wait until the recording is finished
    # write(filename, sample_rate, audio_data)  # Save as WAV file
    audio = AudioSegment(
        audio_data[:, 0].tobytes(),
        frame_rate=sample_rate,
        sample_width=audio_data[:, 0].dtype.itemsize,
        channels=1,
    )
    # audio = AudioSegment.from_file(filename)
    audio = audio.set_frame_rate(SPICE_EXPECTED_SAMPLE_RATE).set_channels(1)
    audio.export(output_file, format="wav")
    return output_file


# converted_audio_file = record_for_spice()

# Loading audio samples from the wav file:
sample_rate, audio_samples = wavfile.read(record_for_spice(duration=.1), "rb")

print(audio_samples/float(MAX_ABS_INT16)
)
audio_samples = audio_samples / float(MAX_ABS_INT16)


# We now feed the audio to the SPICE tf.hub model to obtain pitch and uncertainty outputs as tensors.
model_output = spice_model.signatures["serving_default"](
    tf.constant(audio_samples, tf.float32)
)

pitch_outputs = model_output["pitch"]
uncertainty_outputs = model_output["uncertainty"]


fig, ax = plt.subplots()
fig.set_size_inches(20, 10)
plt.plot(pitch_outputs, label="pitch")
plt.plot(uncertainty_outputs, label="uncertainty")
plt.legend(loc="lower right")
plt.show()


def output2hz(pitch_output):
    # Constants taken from https://tfhub.dev/google/spice/2
    PT_OFFSET = 25.58
    PT_SLOPE = 63.07
    FMIN = 10.0
    BINS_PER_OCTAVE = 12.0
    cqt_bin = pitch_output * PT_SLOPE + PT_OFFSET
    return FMIN * 2.0 ** (1.0 * cqt_bin / BINS_PER_OCTAVE)


def pitch_with_min_uncertainty(pitch_outputs, uncertainty_outputs):
    # Convert tensors to NumPy arrays
    pitch_array = pitch_outputs.numpy()
    uncertainty_array = uncertainty_outputs.numpy()

    # Find the index of the minimum uncertainty
    min_uncertainty_index = np.argmin(uncertainty_array)

    # Retrieve the corresponding pitch value
    min_uncertainty_pitch = pitch_array[min_uncertainty_index]

    return min_uncertainty_pitch


print(output2hz(pitch_outputs))
print(uncertainty_outputs)
print(pitch_with_min_uncertainty(output2hz(pitch_outputs), uncertainty_outputs))
