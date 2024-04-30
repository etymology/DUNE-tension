from main import TensionTestingApp


def measure_one_wire(app: TensionTestingApp, wire_number: int):
    app.goto_wire(wire_number)
    for i in range(10):
        app.auto_log_frequency(output_file=f"{wire_number}_tensions.csv")

if __name__ == "__main__":
    app = TensionTestingApp()
    app.apa.calibrate(layer="X")
    measure_one_wire(app, 480)
    app.handle_quit()
