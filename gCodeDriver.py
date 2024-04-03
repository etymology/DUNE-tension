import re
import platform
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys

# URL of the webpage
webpage_url = 'http://192.168.137.1/Desktop/index.html'

# Function to set the path to the Chrome executable based on the hostname


def get_chrome_path():
    if platform.system() == 'Linux':
        return '/usr/bin/google-chrome'  # Path to the Chrome executable on Linux
    else:
        return None


# Get the path to the Chrome executable
chrome_path = get_chrome_path()

# Initialize Chrome options
chrome_options = webdriver.ChromeOptions()
if chrome_path:
    chrome_options.binary_location = chrome_path
chrome_options.add_argument("--start-fullscreen")
# Initialize the Chrome webdriver with options
driver = webdriver.Chrome(options=chrome_options)

# Function to extract the wire number


def extract_wire_number():

    driver = webdriver.Chrome(options=chrome_options)
    try:
        # Open the webpage
        driver.get(webpage_url)
        time.sleep(5.0)
        # Use JavaScript to find the element by its path
        element_text = driver.execute_script(
            'return document.querySelector("#gCodeTable > tbody > tr.gCodeCurrentLine > td").textContent')
        # Use regular expression to extract the number after "WIRE"
        wire_number_match = re.search(r'WIRE (\d+)', element_text)

        if wire_number_match:
            wire_number = wire_number_match.group(1)
            return wire_number
        else:
            print("No wire number found in the line.")
            return None
    finally:
        # Close the webdriver
        driver.quit()


def click_step_button():
    driver = webdriver.Chrome(options=chrome_options)
    try:
        # Open the webpage
        driver.get(webpage_url)

        # Find the step button by its ID
        step_button = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.ID, 'stepButton'))
        )

        # Click the step button
        step_button.click()
        x_element = 100.0
        y_element = 100.0
        xd_element = -100.0
        yd_element = -100.0
        while not (x_element == xd_element) or not (y_element == yd_element):
            x_element = float(driver.execute_script(
                'return document.querySelector("td#xPositionCell").textContent').strip())
            xd_element = float(driver.execute_script(
                'return document.querySelector("td#xDesiredPosition").textContent').strip())
            y_element = float(driver.execute_script(
                'return document.querySelector("td#yPositionCell").textContent').strip())
            yd_element = float(driver.execute_script(
                'return document.querySelector("td#yDesiredPosition").textContent').strip())
            print("X-position:", x_element)
            print("Y-position:", y_element)
        # Sleep for 0.2 seconds to allow the action to take effect
        time.sleep(0.2)

    finally:
        # Close the webdriver
        driver.quit()


def manual_g_code():
    driver = webdriver.Chrome(options=chrome_options)
    try:
        driver.get(webpage_url)
        time.sleep(2)
        jog_button = WebDriverWait(driver, 2).until(
            EC.element_to_be_clickable(
                (By.XPATH, '/html/body/footer/article[4]/button[2]'))
        )
        jog_button.click()

        time.sleep(2)
        # cmd = "X1200 Y500"
        cmd = "X501.9 Y1295.5"

        element_enter = driver.find_element(By.XPATH, '//*[@id="manualGCode"]')
        element_enter.send_keys(cmd)

        # Find the execute button by its ID
        ex_button = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable(
                (By.XPATH, '/html/body/main/section[3]/article[4]/button'))
        )

        # Click the execute button
        ex_button.click()
        print("Click")
        x_element = 100.0
        y_element = 100.0
        xd_element = -100.0
        yd_element = -100.0
        while not (x_element == xd_element) or not (y_element == yd_element):
            x_element = float(driver.execute_script(
                'return document.querySelector("td#xPositionCell").textContent').strip())
            xd_element = float(driver.execute_script(
                'return document.querySelector("td#xDesiredPosition").textContent').strip())
            y_element = float(driver.execute_script(
                'return document.querySelector("td#yPositionCell").textContent').strip())
            yd_element = float(driver.execute_script(
                'return document.querySelector("td#yDesiredPosition").textContent').strip())
            print("X-position:", x_element)
            print("Y-position:", y_element)
        # Sleep for 0.2 seconds to allow the action to take effect
        time.sleep(0.2)

    finally:
        # Close the webdriver
        driver.quit()


if __name__ == "__main__":
    click_step_button()
    print(f"wire number {extract_wire_number()}")
    manual_g_code()
    print(f"wire number {extract_wire_number()}")
