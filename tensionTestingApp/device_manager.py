import sounddevice as sd
from typing import Tuple
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from maestro import Controller

class DeviceManager:
    def __init__(self, config):
        self.sound_device_index = config.get('sound_device_index', 0)
        self.device_samplerate = config.get('device_samplerate', 44100)
        self.servo_controller = Controller()
        self.driver = None
        self.init_audio_devices()

    def init_audio_devices(self):
        """Initialize the audio devices based on current configuration."""
        try:
            devices = sd.query_devices()
            self.current_device = devices[self.sound_device_index]
        except Exception as e:
            print(f"Failed to initialize audio devices: {str(e)}")

    def select_audio_device(self):
        """Allow the user to select an audio device from available options."""
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            print(f"{i+1}. {device['name']} (default sr: {device['default_samplerate']} Hz)")
        try:
            selection = int(input("Select an audio device: "))
            if 1 <= selection <= len(devices):
                self.sound_device_index = selection - 1
                self.device_samplerate = devices[self.sound_device_index]['default_samplerate']
            else:
                print("Invalid selection. Please enter a number within the valid range.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    def record_audio(self, duration: float) -> Tuple[float, float]:
        """Record audio for a given duration using the selected audio device."""
        with sd.InputStream(device=self.sound_device_index, channels=1, samplerate=self.device_samplerate, dtype='float32') as stream:
            audio_data = stream.read(int(duration * self.device_samplerate))
        return audio_data

    def init_driver(self, webdriver_options):
        """Initialize a browser driver for interacting with the tensiometer's web interface."""
        self.driver = webdriver.Firefox(options=webdriver_options)
        self.driver.get('http://192.168.137.1/Desktop/index.html')

    def close_driver(self):
        """Close the browser driver."""
        if self.driver:
            self.driver.quit()

    def goto_xy(self, x: float, y: float):
        """Navigate the tensiometer to the specified x and y coordinates."""
        jog_button = WebDriverWait(self.driver, 2).until(
            EC.element_to_be_clickable((By.XPATH, '/html/body/footer/article[4]/button[2]'))
        )
        jog_button.click()

        gcode_enter_field = self.driver.find_element(By.XPATH, '//*[@id="manualGCode"]')
        gcode_enter_field.send_keys(f"X{round(x, 1)} Y{round(y, 1)}")

        execute_button = WebDriverWait(self.driver, 5).until(
            EC.element_to_be_clickable((By.XPATH, '/html/body/main/section[3]/article[4]/button'))
        )
        execute_button.click()

    def pluck_string(self):
        """Trigger the tensiometer to pluck a wire string."""
        self.servo_controller.runScriptSub(0) 