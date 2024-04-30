from main import TensionTestingApp
from datetime import datetime

def measure_x_layer(app: TensionTestingApp):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    for wire in range(480, 1, -1):
        app.goto_wire(wire)
        app.auto_log_frequency(output_file=f"X_tensions{timestamp}.csv")

if __name__ == "__main__":
    app = TensionTestingApp()
    app.handle_calibration()
    measure_x_layer(app)