import re
import platform
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys

def zone(x):
    if(x < 2400.0):
        return 1
    elif(2400.0 < x and x < 5000):
        return 2
    elif(5000 < x and x < 6500):
        return 4
    elif(6500 < x and x < 7000):
        return 5
        

def make_hdf():
    xlsx = ExcelFile('TensionGcodeOye.xlsx')
    for sheetind in range(len(xlsx.sheet_names)):
        df = xlsx.parse(xlsx.sheet_names[sheetind])
        sheetstr = xlsx.sheet_names[sheetind]
        df.to_hdf(sheetstr+".df", key="Misc") 

def find_wire_pos(wirenum, layer):
    # must be fixed, only works for Z1 vlayer now
    wiredf = pd.read_hdf("V_FULL_B_Z1.df")
    x = wiredf[wiredf.Wire==wirenum].X.to_numpy()[0]
    y = wiredf[wiredf.Wire==wirenum].Y.to_numpy()[0]
    return x, y

def find_wire_gcode(wirenum, layer):
    X, Y = find_wire_pos(wirenum, layer)
    return "X"+str(int(round(X, 4)))+" Y"+str(int(round(Y, 4)))

# start with laser class
class laser:
    def __init__(self, layer, wirenum):
        # Position Components, note: these do not auto update yet
        driver = webdriver.Chrome(options=chrome_options)
        driver.get(webpage_url)
        time.sleep(1.0)
        self.pos_x = float(driver.execute_script(\
                           'return document.querySelector("td#xPositionCell").textContent'\
                          ).strip())
        self.pos_y = float(driver.execute_script(\
                           'return document.querySelector("td#yPositionCell").textContent'\
                          ).strip())

        self.wirenum = wirenum
        self.layer = layer

# set attributes
    def set_pos(self):
        driver = webdriver.Chrome(options=chrome_options)
        driver.get(webpage_url)
        time.sleep(1.0)
        self.pos_x = float(driver.execute_script(\
                           'return document.querySelector("td#xPositionCell").textContent'\
                          ).strip())

        self.pos_y = float(driver.execute_script(\
                           'return document.querySelector("td#yPositionCell").textContent'\
                          ).strip())

# other funcs
    def is_moving(self):
        driver = webdriver.Chrome(options=chrome_options)
        driver.get(webpage_url)
        x_element = float(driver.execute_script(\
                           'return document.querySelector("td#xPositionCell").textContent'\
                          ).strip())
        xd_element = float(driver.execute_script(\
                            'return document.querySelector("td#xDesiredPosition").textContent'\
                          ).strip())

        y_element = float(driver.execute_script(\
                           'return document.querySelector("td#yPositionCell").textContent'\
                          ).strip())
        yd_element = float(driver.execute_script(\
                           'return document.querySelector("td#yDesiredPosition").textContent'\
                           ).strip())

        if (not(x_element == xd_element) or not(y_element == yd_element)):
            return True
        else:
            return False

    def move_to_wire(self, des_wire):
        ini_wire = self.wirenum
        cmd = find_wire_gcode(des_wire, "V")
        if(self.layer == "U" or self.layer == "V"):
            ini_zone = zone(self.pos_x)
            des_zone = zone(find_wire_pos(des_wire)[0])
            if(ini_zone == des_zone):
                manual_g_code(cmd)
            else:
                manual_g_code("X"+str(round(self.pos_x,4))+" Y190")
                manual_g_code(cmd)
        else:
            manual_g_code(some_cmd)
        self.wirenum = des_wire
        self.set_pos()

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
# chrome_options.add_argument("--headless")
# Initialize the Chrome webdriver with options
# driver = webdriver.Chrome(options=chrome_options)

# Function to extract the wire number
def extract_wirenum():
    driver = webdriver.Chrome(options=chrome_options)
    try:
        # Open the webpage
        driver.get(webpage_url)
        time.sleep(5.0)
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
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(webpage_url)
    try:
        # Open the webpage
        driver.get(webpage_url)
        
        # Find the step button by its ID
        step_button = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.ID, 'stepButton'))
        )
        
        time.sleep(0.2)
        # Click the step button
        step_button.click()

        # Sleep for 0.2 seconds to allow the action to take effect
        time.sleep(0.2)
        
    finally:
        # Close the webdriver
        driver.quit()

def manual_g_code(cmd):
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(webpage_url)
    try:
        driver.get(webpage_url)
        time.sleep(2)
        jog_button = WebDriverWait(driver, 2).until(
            EC.element_to_be_clickable((By.XPATH, '/html/body/footer/article[4]/button[2]'))
        )
        jog_button.click()

        time.sleep(2)
        
        element_enter = driver.find_element(By.XPATH, '//*[@id="manualGCode"]');
        element_enter.send_keys(cmd)

        # Find the execute button by its ID
        ex_button = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.XPATH, '/html/body/main/section[3]/article[4]/button'))
        )
        
        # Click the execute button
        ex_button.click()
        time.sleep(0.2)
        
    finally:
        # Close the webdriver
        driver.quit()


if __name__ == "__main__":
    manual_g_code("X2112 Y987.8")
    test = laser("V", 400)
    print(test.pos_x)
    print(test.pos_y)
    print(test.wirenum)
    print(test.layer)

    test.move_to_wire(399)
    print(test.pos_x)
    print(test.pos_y)
    print(test.wirenum)
    print(test.layer)
