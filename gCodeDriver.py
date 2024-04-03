import re
import sys
import json
import jsonpickle
import platform
import time
import pandas as pd
import os.path
from types import SimpleNamespace

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys

def make_config_comp(calx, caly, calwire, delx, dely, minwirenum, maxwirenum):
    currdict = {}

    for wire in range(minwirenum, maxwirenum+1):
        currdict[wire] = {}
        currdict[wire]["X"] = calx+(wire-calwire)*delx
        currdict[wire]["Y"] = caly+(wire-calwire)*dely
    
    return currdict

def make_config(APAstr):
    layer = None
    apa_dict = {}

    layer = input("Enter layer or quit (X, V, U, G, q): ")
    while(layer!="q"):
        apa_dict[layer]={}

        calbool = 'y'
        calx = []
        caly = []
        calwires = []
        print("Enter new calibration point(s): ")
   
        if(layer == "U" or layer == "V"):
           Ncalpts = 2
        else:
           Ncalpts = 1
 
        for n in range(Ncalpts):
            x = float(input("Enter calibration point X: "))
            y = float(input("Enter calibration point Y: "))
            calwire = int(input("Enter wire number of the calibration point: "))
            calx.append(x)
            caly.append(y)
            calwires.append(calwire)

        if layer == "X":
            layer_dict = make_config_comp(calx[0], caly[0], calwires[0], 0, -4.79166667, 1, 480)
        elif layer == "V":
            # Zone 1
            Vz1_p1 = make_config_comp(calx[0], caly[0], calwires[0], 2.72455392, -3.79161799, 8, 218)
            Vz1_p2 = make_config_comp(Vz1_p1[218]["X"], Vz1_p1[218]["Y"], 218, 0.0, -5.75, 219, 399)
            Vz1 = Vz1_p1 | Vz1_p2

            # Zone 2
            Vz2 = make_config_comp(2800, (Vz1[399]["Y"]-5.75)+(2800-Vz1[399]["X"])*5.75/8, 
                                   400, 0.0, -5.75, 400, 551)

            # Zone 4
            Vz4 = make_config_comp(5150, Vz2[551]["Y"]+(5150-Vz2[400]["X"])*5.75/8, 
                                   400, 0.0, -5.75, 552, 751)

            # Zone 5
            Vz5_p1 = make_config_comp(calx[1], caly[1], calwires[1], 2.72455392, -3.79161799, 992, 1146)
            Vz5_p2 = make_config_comp(Vz5_p1[992]["X"], Vz5_p1[992]["Y"], 992, 0.0, -5.75, 752, 991)
            Vz5 = Vz5_p1 | Vz5_p2

            layer_dict = Vz1 | Vz2 | Vz4 | Vz5

        elif layer == "U":
            # Zone 1
            Uz1 = make_config_comp(calx[0], caly[0], calwires[0], 0.0, 5.75, 150, 399)

            # Zone 2, not super sure about the numbers here
            # Uz2 = make_config_comp(2790, (Uz1[401]["Y"]-5.75)+(2790-Uz1[401]["X"])*5.75/8, 552, 0.0, -5.75, 402, 551)
            Uz2 = make_config_comp(2790, 2081.8, 552, 0.0, -5.75, 400, 551)

            # Zone 4, not super sure about the numbers here
            # Uz4 = make_config_comp(5150, Uz2[551]["Y"]+(5150-Uz2[402]["X"])*5.75/8, 553, 0.0, 5.75, 553, 751)
            Uz4 = make_config_comp(5150, 392.7, 553, 0.0, 5.75, 553, 751)

            # Zone 5
            Uz5 = make_config_comp(6300, 2033.8, 982, 0, 5.75, 752, 982)

            layer_dict = Uz1 | Uz2 | Uz4 | Uz5

        elif layer == "G":
            layer_dict = make_config_comp(calx[0], caly[0], calwires[0], 0, -4.79166667, 1, 481)

        apa_dict[layer] = layer_dict
        layer = input("Enter layer or quit (X, V, U, G, q): ")

    out_file = open(APAstr+".json", "w") 
    json.dump(apa_dict, out_file) 
    out_file.close() 

def zone(x):
    if(x < 2400.0):
        return 1
    elif(2400.0 < x and x < 5000.0):
        return 2
    elif(5000.0 < x and x < 6500.0):
        return 4
    elif(6500.0 < x and x < 7000.0):
        return 5

def find_wire_pos(wirenum, layer, cfg):
    wire_dict = load(cfg)
    x = wire_dict[layer][str(wirenum)]["X"]
    y = wire_dict[layer][str(wirenum)]["Y"]
    return x, y

def find_wire_gcode(wirenum, layer, cfg):
    X, Y = find_wire_pos(wirenum, layer, cfg)
    return "X"+str(int(round(X, 4)))+" Y"+str(int(round(Y, 4)))

def load(save_name):
    with open(save_name+'.json', 'r') as infile:
        obj = json.load(infile)
    return obj

# start with apa class
class apa(object):
    def __init__(self, layer, cfg="Untitled_cfg", ini_wirenum=None):
        # cfg used to determine locations of wires
        if not os.path.isfile(cfg+".json"):
            self.cfg = make_config(cfg)
            print(cfg)
        else:
            self.cfg = load(cfg)

        # Position Components, note: these do not auto update
        driver = webdriver.Chrome(options=chrome_options)
        driver.get(webpage_url)
        time.sleep(1.0)
        self.pos_x = float(driver.execute_script(\
                           'return document.querySelector("td#xPositionCell").textContent'\
                          ).strip())
        self.pos_y = float(driver.execute_script(\
                           'return document.querySelector("td#yPositionCell").textContent'\
                          ).strip())

        self.wirenum = ini_wirenum
        self.layer = layer

# save the entire state of the apa object
    def save_obj(self, save_name):
        jsonObj = jsonpickle.encode(self)
        with open(save_name+'.json', 'w') as outfile:
            json.dump(jsonObj, outfile)

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
        cmd = find_wire_gcode(des_wire, "V")
        if(self.layer == "U" or self.layer == "V"):
            ini_zone = zone(self.pos_x)
            des_zone = zone(find_wire_pos(des_wire, self.layer, self.cfg)[0])
            if(ini_zone == des_zone):
                manual_g_code(cmd)
            else:
                manual_g_code("X"+str(round(self.pos_x,4))+" Y190")
                manual_g_code(cmd)
        else:
            manual_g_code(some_cmd)
        self.wirenum = des_wire
        self.set_pos()

# save the entire state of the apa object
def load_obj(save_name):
    with open(save_name) as jsonfile:
        json_dict = json.load(jsonfile)

    json_obj = json.loads(json_dict, 
        object_hook=lambda d: SimpleNamespace(**d))
    return json_obj

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
# chrome_options.add_argument("--start-fullscreen")
chrome_options.add_argument("--headless")
chrome_options.add_argument("--window-size=1920,1080")

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
        driver.execute_script("document.body.style.zoom='75%'")   
        time.sleep(2)
        element_enter = driver.find_element(By.XPATH, '//*[@id="manualGCode"]');
        element_enter.send_keys(cmd)

        time.sleep(2)

        sys.exit()
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
    # test = apa("V" , "Wood_cfg")
    # test.save_obj("test")

    test = load_obj("test.json")
    print(test.pos_x)
    print(test.pos_y)
#    manual_g_code("X440 Y20")
#    print(test.pos_x)
#    print(test.pos_y)
#    print(test.wirenum)
#    print(test.layer)
#    print(zone(test.pos_x))
#
#    test.move_to_wire(399)
#    print(test.pos_x)
#    print(test.pos_y)
#    print(test.wirenum)
#    print(test.layer)
#    print(zone(test.pos_x))
