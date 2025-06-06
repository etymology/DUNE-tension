# servotest_gui.py

import tkinter as tk
from threading import Thread, Event
import time
from maestro.maestro import Controller


class ServoController:
    def __init__(self):
        self.servo = Controller()
        self.servo.setRange(0, 4000, 8000)
        self.running = Event()
        self.dwell_time = 1.0  # default dwell time in seconds

    def set_speed(self, speed: int):
        self.servo.setSpeed(0, speed)

    def set_accel(self, accel: int):
        self.servo.setAccel(0, accel)

    def set_dwell_time(self, val: float):
        self.dwell_time = float(val)

    def toggle_loop(self):
        if self.running.is_set():
            self.running.clear()
        else:
            self.running.set()
            Thread(target=self.run_loop, daemon=True).start()

    def run_loop(self):
        while self.running.is_set():
            self.servo.setTarget(0, 4000)
            while self.servo.isMoving(0) and self.running.is_set():
                time.sleep(0.01)
            time.sleep(self.dwell_time)

            self.servo.setTarget(0, 8000)
            while self.servo.isMoving(0) and self.running.is_set():
                time.sleep(0.01)
            time.sleep(self.dwell_time)


def build_gui(controller: ServoController):
    root = tk.Tk()
    root.title("Maestro Servo Controller")

    # Speed
    tk.Label(root, text="Speed").pack()
    speed_slider = tk.Scale(
        root,
        from_=1,
        to=255,
        orient=tk.HORIZONTAL,
        command=lambda val: controller.set_speed(int(val)),
    )
    speed_slider.set(1)
    speed_slider.pack()

    # Acceleration
    tk.Label(root, text="Acceleration").pack()
    accel_slider = tk.Scale(
        root,
        from_=1,
        to=255,
        orient=tk.HORIZONTAL,
        command=lambda val: controller.set_accel(int(val)),
    )
    accel_slider.set(1)
    accel_slider.pack()

    # Dwell time
    tk.Label(root, text="Dwell Time (seconds)").pack()
    dwell_slider = tk.Scale(
        root,
        from_=0,
        to=100,
        resolution=1,
        orient=tk.HORIZONTAL,
        command=lambda val: controller.set_dwell_time(float(val) / 100),
    )
    dwell_slider.set(100)  # Default to 1.0 second
    dwell_slider.pack()

    # Start/Stop button
    def toggle():
        if controller.running.is_set():
            start_stop_btn.config(text="Start")
        else:
            start_stop_btn.config(text="Stop")
        controller.toggle_loop()

    start_stop_btn = tk.Button(root, text="Start", command=toggle)
    start_stop_btn.pack(pady=10)

    root.mainloop()


if __name__ == "__main__":
    servo_controller = ServoController()
    build_gui(servo_controller)
