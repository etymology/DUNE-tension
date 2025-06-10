import serial
from sys import version_info
import platform
import time
from threading import Event, Thread, RLock

PY2 = version_info[0] == 2  # Running Python 2.x?

#
# ---------------------------
# Maestro Servo Controller
# ---------------------------
#
# Support for the Pololu Maestro line of servo controllers
#
# Steven Jacobs -- Aug 2013
# https://github.com/FRC4564/Maestro/
#
# These functions provide access to many of the Maestro's capabilities using the
# Pololu serial protocol
#


class Controller:
    # When connected via USB, the Maestro creates two virtual serial ports
    # /dev/ttyACM0 for commands and /dev/ttyACM1 for communications.
    # Be sure the Maestro is configured for "USB Dual Port" serial mode.
    # "USB Chained Mode" may work as well, but hasn't been tested.
    #
    # Pololu protocol allows for multiple Maestros to be connected to a single
    # serial port. Each connected device is then indexed by number.
    # This device number defaults to 0x0C (or 12 in decimal), which this module
    # assumes.  If two or more controllers are connected to different serial
    # ports, or you are using a Windows OS, you can provide the tty port.  For
    # example, '/dev/ttyACM2' or for Windows, something like 'COM3'.

    def __init__(self, ttyStr="/dev/ttyACM0", device=0x0C):
        self.faulted = False
        self.usb = None

        # Determine the appropriate port based on the operating system
        if platform.system() == "Windows":
            ttyStr = "COM3"

        # # # Search for the Micro Maestro 6-Servo Controller
        # ports = list_ports.comports()
        # maestro_port = None

        # for port in ports:
        #     if "Micro Maestro 6-Servo Controller" in port.description:
        #         maestro_port = port.device
        #         break

        # if maestro_port is not None:
        #     ttyStr = maestro_port

        # Open the command port
        try:
            self.usb = serial.Serial(ttyStr)
            self.faulted = False
            print(f"Connected to Micro Maestro on {ttyStr}")
        except serial.SerialException:
            print(
                f"Couldn't find Micro Maestro on {ttyStr}! Check the connection or port."
            )
            self.faulted = True

        # Command lead-in and device number are sent for each Pololu serial command.
        self.PololuCmd = chr(0xAA) + chr(device)
        # Track target position for each servo. Targets start at 0 for up to 24 servos.
        self.Targets = [0] * 24
        # Servo minimum and maximum targets can be restricted to protect components.
        self.Mins = [0] * 24
        self.Maxs = [0] * 24
        self.lock = RLock()

    # Cleanup by closing USB serial port
    def close(self):
        with self.lock:
            self.usb.close()

    # Send a Pololu command out the serial pnoort
    def sendCmd(self, cmd):
        with self.lock:
            cmdStr = self.PololuCmd + cmd
            if PY2:
                self.usb.write(cmdStr)
            else:
                self.usb.write(bytes(cmdStr, "latin-1"))

    # Set channels min and max value range.  Use this as a safety to protect
    # from accidentally moving outside known safe parameters. A setting of 0
    # allows unrestricted movement.
    #
    # ***Note that the Maestro itself is configured to limit the range of servo travel
    # which has precedence over these values.  Use the Maestro Control Center to configure
    # ranges that are saved to the controller.  Use setRange for software controllable ranges.
    def setRange(self, chan, min, max):
        self.Mins[chan] = min
        self.Maxs[chan] = max

    # Return Minimum channel range value
    def getMin(self, chan):
        return self.Mins[chan]

    # Return Maximum channel range value
    def getMax(self, chan):
        return self.Maxs[chan]

    # Set channel to a specified target value.  Servo will begin moving based
    # on Speed and Acceleration parameters previously set.
    # Target values will be constrained within Min and Max range, if set.
    # For servos, target represents the pulse width in of quarter-microseconds
    # Servo center is at 1500 microseconds, or 6000 quarter-microseconds
    # Typcially valid servo range is 3000 to 9000 quarter-microseconds
    # If channel is configured for digital output, values < 6000 = Low ouput
    def setTarget(self, chan, target):
        # if Min is defined and Target is below, force to Min
        with self.lock:
            if self.Mins[chan] > 0 and target < self.Mins[chan]:
                target = self.Mins[chan]
            if self.Maxs[chan] > 0 and target > self.Maxs[chan]:
                target = self.Maxs[chan]
            self.sendCmd(self._make_command(target, 0x04, chan))
            self.Targets[chan] = target

    # Set speed of channel
    # Speed is measured as 0.25microseconds/10milliseconds
    # For the standard 1ms pulse width change to move a servo between extremes, a speed
    # of 1 will take 1 minute, and a speed of 60 would take 1 second.
    # Speed of 0 is unrestricted.
    def setSpeed(self, chan, speed):
        with self.lock:
            self.sendCmd(self._make_command(speed, 0x07, chan))

    # Set acceleration of channel
    # This provide soft starts and finishes when servo moves to target position.
    # Valid values are from 0 to 255. 0=unrestricted, 1 is slowest start.
    # A value of 1 will take the servo about 3s to move between 1ms to 2ms range.
    def setAccel(self, chan, accel):
        with self.lock:
            self.sendCmd(self._make_command(accel, 0x09, chan))

    def _make_command(self, message, preface, chan):
        lsb = message & 127
        msb = message >> 7 & 127
        return chr(preface) + chr(chan) + chr(lsb) + chr(msb)

    # Get the current position of the device on the specified channel
    # The result is returned in a measure of quarter-microseconds, which mirrors
    # the Target parameter of setTarget.
    # This is not reading the true servo position, but the last target position sent
    # to the servo. If the Speed is set to below the top speed of the servo, then
    # the position result will align well with the acutal servo position, assuming
    # it is not stalled or slowed.
    def getPosition(self, chan):
        cmd = chr(0x10) + chr(chan)
        with self.lock:
            self.sendCmd(cmd)
            lsb = ord(self.usb.read())
            msb = ord(self.usb.read())
        return (msb << 8) + lsb

    # Test to see if a servo has reached the set target position.  This only provides
    # useful results if the Speed parameter is set slower than the maximum speed of
    # the servo.  Servo range must be defined first using setRange. See setRange comment.
    #
    # ***Note if target position goes outside of Maestro's allowable range for the
    # channel, then the target can never be reached, so it will appear to always be
    # moving to the target.
    def isMoving(self, chan):
        return self.Targets[chan] > 0 and self.getPosition(chan) != self.Targets[chan]

    # Have all servo outputs reached their targets? This is useful only if Speed and/or
    # Acceleration have been set on one or more of the channels. Returns True or False.
    # Not available with Micro Maestro.
    def getMovingState(self):
        cmd = chr(0x13)
        with self.lock:
            self.sendCmd(cmd)
            return self.usb.read() != chr(0)

    # Run a Maestro Script subroutine in the currently active script. Scripts can
    # have multiple subroutines, which get numbered sequentially from 0 on up. Code your
    # Maestro subroutine to either infinitely loop, or just end (return is not valid).
    def runScriptSub(self, subNumber):
        cmd = chr(0x27) + chr(subNumber)
        # can pass a param with command 0x28
        # cmd = chr(0x28) + chr(subNumber) + chr(lsb) + chr(msb)
        with self.lock:
            self.sendCmd(cmd)

    # Stop the current Maestro Script
    def stopScript(self):
        cmd = chr(0x24)
        with self.lock:
            self.sendCmd(cmd)


class DummyController:
    """Minimal stand-in for :class:`Controller` used during testing."""

    def __init__(self, *_, **__):
        self.Targets = [0] * 24
        self.Mins = [0] * 24
        self.Maxs = [0] * 24
        self.position = 0

    def close(self):
        pass

    def sendCmd(self, _):
        pass

    def setRange(self, chan, min_val, max_val):
        self.Mins[chan] = min_val
        self.Maxs[chan] = max_val

    def getMin(self, chan):
        return self.Mins[chan]

    def getMax(self, chan):
        return self.Maxs[chan]

    def setTarget(self, chan, target):
        self.Targets[chan] = target
        self.position = target

    def setSpeed(self, chan, speed):
        pass

    def setAccel(self, chan, accel):
        pass

    def getPosition(self, chan):
        return self.position

    def isMoving(self, chan):
        return False

    def getMovingState(self):
        return False

    def runScriptSub(self, subNumber):
        pass

    def stopScript(self):
        pass


class ServoController:
    """High-level servo helper used by the GUI."""

    def __init__(self, servo: Controller | DummyController | None = None) -> None:
        self.servo = servo or DummyController()
        self.servo.setRange(0, 4000, 8000)  # plucking servo
        self.running: Event = Event()
        self.dwell_time: float = 1.0

        self.servo.setRange(1, 4000, 8000)  # focus servo
        try:
            self.servo.setSpeed(1, 100)
            self.servo.setAccel(1, 100)
        except Exception:
            pass

    def set_speed(self, val: int) -> None:
        self.servo.setSpeed(0, int(val))

    def set_accel(self, val: int) -> None:
        self.servo.setAccel(0, int(val))

    def set_dwell_time(self, val: float) -> None:
        self.dwell_time = float(val)

    def start_loop(self) -> None:
        if not self.running.is_set():
            self.running.set()
            Thread(target=self.run_loop, daemon=True).start()

    def stop_loop(self) -> None:
        self.running.clear()

    def run_loop(self) -> None:
        while self.running.is_set():
            self.servo.setTarget(0, 4000)
            while self.servo.isMoving(0) and self.running.is_set():
                time.sleep(0.01)
            self.servo.setTarget(0, 8000)
            while self.servo.isMoving(0) and self.running.is_set():
                time.sleep(0.01)
            time.sleep(self.dwell_time)

    def focus_target(self, target: int) -> None:
        self.servo.setTarget(1, target)
        while self.servo.isMoving(1):
            time.sleep(0.01)


if __name__ == "__main__":
    import serial.tools.list_ports

    print([port.description for port in serial.tools.list_ports.comports()])

    servo = Controller()
    servo.setRange(0, 4000, 8000)
    servo.setSpeed(0, 1)
    servo.setAccel(0, 1)
    while True:
        time.sleep(1)
        servo.setTarget(0, 4000)
        while servo.isMoving(0):
            time.sleep(0.01)
        servo.setTarget(0, 8000)  # set servo to move to center position
        while servo.isMoving(0):
            time.sleep(0.01)
    servo.close()
