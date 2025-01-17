from tension_client_across_combs import measure_sequential_across_combs, measure_LUT, measure_one_wire
from Tensiometer import Tensiometer
# from process_wire_data import process_wire_data, find_tensions_outside_range


if __name__ == "__main__":
    t = Tensiometer(
        apa_name="US_APA5",
        layer="X",
        side="Anojig",
        wiggle_step=0.0,
        samples_per_wire=3,
        confidence_threshold=0.7,
        initial_wire_height=189.3,
        use_wiggle=True,
        sound_card_name="default",
        save_audio=True,
        timeout=30
    )
    # process_wire_data(t)
    # lookup = find_tensions_outside_range(t)
    # measure_LUT(t, [1])
    # measure_sequential_across_combs(t, initial_wire_number=275, direction=1,final_wire_number=480)

    measure_LUT(t, [475,480])

    # x,y=t.get_xy()
    # t.goto_xy(x,y+0.1)
    # for wireno in range(20,30):
    #     measure_one_wire(t,wireno,10)
    #     x,y=t.get_xy()
    #     t.goto_xy(x,y+0.1)
    # print("done") 
    # measure_sequential_across_combs(t, initial_wire_number=252, direction=1)
    
    # measure_sequential_across_combs(t, initial_wire_number=1, direction=1)
