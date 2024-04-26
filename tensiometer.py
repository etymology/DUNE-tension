from time import sleep
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from maestro import Controller

# URL of the webpage
WINDER_URL = 'http://192.168.137.1/Desktop/index.html'
MAESTRO_SUBSCRIPTS = {'pluck_string': 0}
GCODE_XPATH = '//*[@id="manualGCode"]'
EXECUTE_BUTTON_XPATH = '/html/body/main/section[3]/article[4]/button'
JOG_BUTTON_XPATH = '/html/body/footer/article[4]/button[2]'

# Initialize Firefox options
FIREFOX_OPTIONS = webdriver.FirefoxOptions()
FIREFOX_OPTIONS.add_argument("--headless")
FIREFOX_OPTIONS.add_argument("--width=2560")
FIREFOX_OPTIONS.add_argument("--height=1440")


class Tensiometer:
    """
    A tensiometer is a device affixed to an APA which moves to wires and plucks them.
    A Tensiometer must be attached to an APA to work. Its methods allow a user to
    inquire about the tensiometer's position, go to a position 
    or wire number, pluck a wire.
    """

    def __init__(self):
        # self driver set to None for remote testing
        # self.driver = None
        self.init_driver()
        self.servo_controller = Controller()

    def init_driver(self):
        firefox_options = FIREFOX_OPTIONS
        self.driver = webdriver.Firefox(options=firefox_options)
        self.driver.get(WINDER_URL)

    def __enter__(self):
        self.init_driver()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.driver.quit()

    def get_xy(self):
        x, y = '', ''
        while not x or not y:
            x = self.driver.execute_script(
                'return document.querySelector("td#xPositionCell").textContent').strip()
            y = self.driver.execute_script(
                'return document.querySelector("td#yPositionCell").textContent').strip()
        return float(x), float(y)

    def is_moving(self) -> bool:
        x, y = self.get_xy()
        x_target = float(self.driver.execute_script(
            'return document.querySelector("td#xDesiredPosition").textContent').strip())
        y_target = float(self.driver.execute_script(
            'return document.querySelector("td#yDesiredPosition").textContent').strip())
        return x != x_target or y != y_target

    def goto_xy(self, x, y):
        sleep(1.0)
        jog_button = WebDriverWait(self.driver, 2).until(
            EC.element_to_be_clickable((By.XPATH, JOG_BUTTON_XPATH))
        )
        jog_button.click()

        gcode_enter_field = self.driver.find_element(By.XPATH, GCODE_XPATH)
        gcode_enter_field.send_keys(f"X{round(x, 1)} Y{round(y, 1)}")

        execute_button = WebDriverWait(self.driver, 5).until(
            EC.element_to_be_clickable((By.XPATH, EXECUTE_BUTTON_XPATH))
        )
        execute_button.click()

    def pluck_string(self):
        self.servo_controller.runScriptSub(MAESTRO_SUBSCRIPTS['pluck_string'])

    def close_driver(self):
        """Close the browser driver."""
        if self.driver:
            self.driver.quit()
