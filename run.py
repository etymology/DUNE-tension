from tensiometer_functional import Tensiometer
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

if __name__ == "__main__":
    t = Tensiometer(
        apa_name="US_APA999",
        layer="V",
        side="B",
        spoof=True
    )
    t.measure_calibrate(600)
    t.measure_auto()