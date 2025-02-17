from tension_client_across_combs import (
    measure_sequential_across_combs,measure_LUT
)
from Tensiometer import Tensiometer

# from process_wire_data import process_wire_data, find_tensions_outside_range
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

if __name__ == "__main__":
    t = Tensiometer(
        apa_name="US_APA5",
        layer="G",
        side="A",
        starting_wiggle_step=.2,
        samples_per_wire=5,
        confidence_threshold=0,
        use_wiggle=True,
        sound_card_name="default",
        timeout=120,
        save_audio=True,
        record_duration=.4,
        wiggle_interval=3,
        initial_wire_height=189.8
    )
    # process_wire_data(t)
    # lookup = find_tensions_outside_range(t)
    # measure_LUT(t, [1
    # measure_sequential_across_combs(
    #     t, initial_wire_number=431, direction=1, use_relative_position=True, use_LUT=False
    # )
    # measure_LUT(t,[396,423])
    measure_sequential_across_combs(
        t, initial_wire_number=8
            , direction=1, use_relative_position=True, use_LUT=False
    )

    # measure_LUT(t, [1089199])

    # x,y=t.get_xy()
    # t.goto_xy(x,y+0.1)
    # for wireno in range(20,30):
    #     measure_one_wire(t,wireno,10)
    #     x,y=t.get_xy()
    #     t.goto_xy(x,y+0.1)
    # print("done")
    # measure_sequential_across_combs(t, initial_wire_number=252, direction=1)

    # measure_sequential_across_combs(t, initial_wire_number=1, direction=1)
