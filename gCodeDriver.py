import re
import platform
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC



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

# Initialize the Chrome webdriver with options
driver = webdriver.Chrome(options=chrome_options)

# Function to extract the wire number
def extract_wire_number():

    driver = webdriver.Chrome(options=chrome_options)
    try:
        # Open the webpage
        driver.get(webpage_url)
        time.sleep(.3)
        # Use JavaScript to find the element by its path
        element_text = driver.execute_script('return document.querySelector("#gCodeTable > tbody > tr.gCodeCurrentLine > td").textContent')
        
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
    try:
        # Open the webpage
        driver.get(webpage_url)
        
        # Find the step button by its ID
        step_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.ID, 'stepButton'))
        )
        
        # Click the step button
        step_button.click()
        
        # Sleep for 0.2 seconds to allow the action to take effect
        time.sleep(0.2)
        
    finally:
        # Close the webdriver
        driver.quit()

if __name__ == "__main__":
    click_step_button()
    time.sleep(1)
    print(f"wire number {extract_wire_number()}")
