import time
import serial
import serial.tools.list_ports as port_list

# Features:
# 1. Onboard high-performance microcontrollers chip;
# 2. Onboard CH340 USB control chip;
# 3. Onboard power LED and relay status LED;
# 4. Onboard 2-way 5V, 10A / 250VAC, 10A / 30VDC relays, relay life can be a continuous pull 10 million times;
# 5. Module with overcurrent protection and relay diode freewheeling protection;

# Hardware introduction and description
# Board size: 50 x 40mm
# Board Interface Description:
# COM1: common;
# NC1: normally closed;
# NO1: normally open.
# COM2: common;
# NC2: normally closed;
# NO2: normally open.

# Communication protocol description:
# LC USB switch default communication baud rate: 9600BPS
# Open the first USB switch: A0 01 01 A2
# Turn off the first USB switch: A0 01 00 A1
# Open the second USB switch: A0 02 01 A3
# Turn off the second USB switch: A0 02 00 A2

# USB switch communication protocol
# Data (1) --- start flag (default is 0xA0)
# Data (2) --- switch address codes (0x01 and 0x02 represent the first and second switches, respectively)
# Data (3) --- operating data (0x00 is "off", 0x01 is "on")
# Data (4) --- check code

# Relay status query command:
# Send "FF" as hexadecimal (hex) to query.
# For example, if relays 1 and 2 are ON, and relays 3 and 4 are OFF, sending the relay query command "FF"will return:
# "CH1:ON \r\nCH2:ON \r\nCH3: OFF\r\nCH4:OFF\r\n"
# (Each channel relay is to return 10 byte sequence information)

# QinHeng Electronics CH340 serial converter variables
idVendor = "1A86"
idProduct = "7523"
open_valve = [0xA0, 0x01, 0x01, 0xA2]
close_valve = [0xA0, 0x01, 0x00, 0xA1]
devicePresent = False

ports = list(port_list.comports())

for p in ports:
    if str(idVendor + ":" + idProduct) in p.hwid:
        print("found it!")
        port = p.device
        devicePresent = True
        break

if devicePresent:
    print("Our device is connected")
    usb_relay_valve = serial.Serial(port, 9600, timeout=1)
    while True:
        usb_relay_valve.write(open_valve)
        time.sleep(0.01)
        usb_relay_valve.write(close_valve)
        time.sleep(2)