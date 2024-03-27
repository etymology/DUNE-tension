import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# URL of the webpage
webpage_url = 'http://192.168.137.1/Desktop/index.html'

def g_code_step():
    # Initialize the Firefox webdriver
    driver = webdriver.Firefox()
    
    try:
        # Open the webpage
        driver.get(webpage_url)
        
        # Find the button by its ID
        button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.ID, 'stepButton'))
        )
        
        # Click the button
        button.click()
        
        # Optional: Wait for some time to see the result
        driver.implicitly_wait(5)
        
    finally:
        # Close the webdriver
        driver.quit()

def extract_wire_number():
    # Initialize the Firefox webdriver
    driver = webdriver.Firefox()
    
    try:
        # Open the webpage
        driver.get(webpage_url)
        
        # Find the element by its class name
        element = driver.find_element_by_css_selector('tr.gCodeCurrentLine')
        
        # Get the text content of the element
        element_text = element.text
        
        # Use regular expression to extract the number after "WIRE"
        wire_number_match = re.search(r'WIRE (\d+)', element_text)
        
        if wire_number_match:
            wire_number = wire_number_match.group(1)
            print("Number after 'WIRE':", wire_number)
        else:
            print("No wire number found in the line.")
        
    finally:
        # Close the webdriver
        driver.quit()