from tension_client_across_combs import measure_sequential_across_combs, measure_LUT
from Tensiometer import Tensiometer

# from process_wire_data import process_wire_data, find_tensions_outside_range
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

if __name__ == "__main__":
    t = Tensiometer(
        apa_name="US_APA7",
        layer="G",
        side="B",
        starting_wiggle_step=0.2,
        samples_per_wire=3,
        confidence_threshold=0.7,
        use_wiggle=True,
        sound_card_name="default",
        timeout=100,
        save_audio=True,
        record_duration=0.15,
        wiggle_interval=2,
        flipped=False,
        initial_wire_height=193.6
    )

    measure_LUT(
        t,
        [48,]

    )



    # measure_LUT(
    #     t,
    #     [1105
    #      ]
    # )

    # x,y=t.get_xy()
    # t.goto_xy(x,y+0.1)
    # for wireno in range(20,30):
    #     measure_one_wire(t,wireno,10)
    #     x,y=t.get_xy()
    #     t.goto_xy(x,y+0.1)
    # print("done")
    # measure_sequential_across_combs(t, initial_wire_number=252, direction=1)

    # measure_sequential_across_combs(t, initial_wire_number=1, direction=1)
