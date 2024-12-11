from tension_client_across_combs import measure_sequential_across_combs, measure_LUT
from Tensiometer import Tensiometer
# from process_wire_data import process_wire_data, find_tensions_outside_range


if __name__ == "__main__":
    t = Tensiometer(
        apa_name="US_APA4",
        layer="G",
        side="A",
        wiggle_step=0.0,
        samples_per_wire=1,
        confidence_threshold=0.65,
        initial_wire_height=187.9
    )

    # process_wire_data(t)
    # lookup = find_tensions_outside_range(t)

    # measure_sequential_across_combs(t, initial_wire_number=1, direction=1)

    measure_LUT(t, [460])