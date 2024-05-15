import sounddevice as sd


def list_audio_devices():
    print("Available audio devices:")
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        print(
            f"Index: {i}, Name: {dev['name']}, Input Channels: {dev['max_input_channels']}, Output Channels: {dev['max_output_channels']}")


list_audio_devices()
