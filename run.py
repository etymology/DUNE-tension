from tension_client_across_combs import measure_sequential_across_combs, measure_LUT
from Tensiometer import Tensiometer

# from process_wire_data import process_wire_data, find_tensions_outside_range
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

if __name__ == "__main__":
    t = Tensiometer(
        apa_name="US_APA6",
        layer="U",
        side="A",
        starting_wiggle_step=0.5,
        samples_per_wire=1,
        confidence_threshold=0.7,
        use_wiggle=True,
        sound_card_name="default",
        record_duration=0.15,
        wiggle_interval=.5,
        save_audio=True,
        timeout=30,
    )
    # process_wire_data(t)
    # lookup = find_tensions_outside_range(t)
    # measure_LUT(t, [1
    # measure_sequential_across_combs(
    #     t, initial_wire_number=431, direction=1, use_relative_position=True, use_LUT=FalseFalse
    # )
    # measure_LUT(t,[396,423])
    # measure_sequential_across_combs
    #     t, initial_wire_number=1
    #         , direction=1, use_relative_position=True, use_LUT=False
    # )


    measure_sequential_across_combs(
        t,
        initial_wire_number=959,
        direction=-1,
        use_relative_position=True,
        use_LUT=False,
    )


# x,y=t.get_xy()
# t.goto_xy(x,y+0.1)
# for wireno in range(20,30):
#     measure_one_wire(t,wireno,10)
#     x,y=t.get_xy()
#     t.goto_xy(x,y+0.1)
# print("done")
# measure_sequential_across_combs(t, initial_wire_number=252, direction=1)

# measure_sequential_across_combs(t, initial_wire_number=1, direction=1)
