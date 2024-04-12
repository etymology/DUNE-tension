import json
import os.path
from typing import Tuple

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import sounddevice as sd

from maestro import Controller
from apa import APA

WINDER_URL = 'http://192.168.137.1/Desktop/index.html'
CONFIG_FILENAME = 'tensionConfig.json'
MAESTRO_SUBSCRIPTS = {'pluck_string': 0}
GCODE_XPATH = '//*[@id="manualGCode"]'
EXECUTE_BUTTON_XPATH = '/html/body/main/section[3]/article[4]/button'
JOG_BUTTON_XPATH = '/html/body/footer/article[4]/button[2]'

firefox_options = webdriver.FirefoxOptions()
firefox_options.add_argument("--headless")
firefox_options.add_argument("--width=2560")
firefox_options.add_argument("--height=1440")

class Tensiometer:
    def __init__(self, apaName: str):
        self.configuration = self.load_configuration()
        self.driver = self.init_driver()
        self.maestro = Controller()
        self.APA = APA(apaName)

    def load_configuration(self):
        if not os.path.exists(CONFIG_FILENAME):
            return {}
        with open(CONFIG_FILENAME, 'r') as file:
            return json.load(file)

    def save_configuration(self):
        with open(CONFIG_FILENAME, 'w') as file:
            json.dump(self.configuration, file)

    def load_default_config(self, tagname: str):
        self.configuration.setdefault(tagname, self.update_config(tagname))
        return self.configuration[tagname]

    def update_config(self, tagname: str, testType=any):
        new_value = input(f"Enter new value for {tagname}")
        if isinstance(new_value, testType):
            self.configuration[tagname] = new_value
        else:
            print("Wrong input type!")

    @staticmethod
    def init_driver():
        driver = webdriver.Firefox(options=firefox_options)
        driver.get(WINDER_URL)
        return driver

    def execute_script(self, script):
        return self.driver.execute_script(script)

    def xy_position(self) -> Tuple[float, float]:
        x_string, y_string = '', ''
        while not x_string or not y_string:
            x_string = self.execute_script(
                'return document.querySelector("td#xPositionCell").textContent').strip()
            y_string = self.execute_script(
                'return document.querySelector("td#yPositionCell").textContent').strip()
        return (float(x_string), float(y_string))

    def is_moving(self) -> bool:
        (x, y) = self.xy_position()
        x_target = float(self.execute_script(
            'return document.querySelector("td#xDesiredPosition").textContent').strip())
        y_target = float(self.execute_script(
            'return document.querySelector("td#yDesiredPosition").textContent').strip())
        return x != x_target or y != y_target

    def goto_xy(self, xy: Tuple[float, float]):
        gcode_command = f"X{round(xy[0],1)} Y{round(xy[1], 1)}"
        jog_button = WebDriverWait(self.driver, 2).until(
            EC.element_to_be_clickable((By.XPATH, JOG_BUTTON_XPATH))
        )
        jog_button.click()

        element_enter = self.driver.find_element(By.XPATH, GCODE_XPATH)
        element_enter.send_keys(gcode_command)

        ex_button = WebDriverWait(self.driver, 5).until(
            EC.element_to_be_clickable((By.XPATH, EXECUTE_BUTTON_XPATH))
        )
        ex_button.click()

    def quit_driver(self):
        self.driver.quit()

    def goto_wire(self, layer: str, wire_number: int):
        self.goto_xy(self.APA.calibration[layer][wire_number])

    def pluck_string(self):
        self.maestro.runScriptSub(MAESTRO_SUBSCRIPTS)

    def update_audio_device(self):
        devices = sd.query_devices()
        print("Available audio devices:")
        for i, device in enumerate(devices):
            print(f"{i + 1}. {device['name']}")

        while True:
            try:
                choice = int(input("Enter the number of the audio device you want to use: "))
                if 1 <= choice <= len(devices):
                    print(devices[choice - 1])
                    self.configuration['audioDevice']
                else:
                    print("Invalid choice. Please enter a number within the range.")
            except ValueError:
                print("Invalid input. Please enter a number.")
