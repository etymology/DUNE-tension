from Tensiometer import Tensiometer
import pandas as pd

t = Tensiometer(
    apa_name="US_APA4",
    layer="U",
    side="A",
    wiggle_step=0.0,
    samples_per_wire=3,
    confidence_threshold=0.7,
    sound_card_name="",
    test_mode=True,
    )


def process_wire_data(t: Tensiometer):
    # Load the data
    input_file = f"data/frequency_data_{t.apa_name}_{t.layer}.csv"
    output_file = f"data/processed_wire_data_{t.apa_name}_{t.layer}.csv"
    data = pd.read_csv(input_file)

    # Initialize the full list of wires (A-8 through A-1151 and B-8 through B-1151)
    layers = ["A", "B"]
    wire_numbers = range(8, 1152)  # Inclusive range
    full_index = pd.MultiIndex.from_product(
        [layers, wire_numbers], names=["side", "wire_number"]
    )

    # Prepare the data: keep only the necessary columns and filter
    data = data[["layer", "side", "wire_number", "tension", "confidence"]]
    data["wire_number"] = data["wire_number"].astype(int)

    # Group by side and wire_number, keeping the row with the highest confidence
    grouped = (
        data.groupby(["side", "wire_number"])
        .apply(lambda x: x.loc[x["confidence"].idxmax()])
        .reset_index(drop=True)
    )

    # Reindex to include all wires and fill missing ones with tension 0
    grouped = (
        grouped.set_index(["side", "wire_number"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )

    # Add layer information back
    grouped["layer"] = "U"  # Assuming all rows have layer "U"
    grouped = grouped[["layer", "side", "wire_number", "tension"]]

    # Save to CSV
    grouped.to_csv(output_file, index=False)

def find_tensions_outside_range(t: Tensiometer, lower_bound=4, upper_bound=8.5):
    # Load the processed data
    input_file = f"data/processed_wire_data_{t.apa_name}_{t.layer}.csv"

    data = pd.read_csv(input_file)
    
    # Filter wires with tension outside the specified range
    outside_range = data[(data["tension"] < lower_bound) | (data["tension"] > upper_bound)]
    
    # Group by side and collect wire numbers for each side
    result = {
        side: group["wire_number"].tolist()
        for side, group in outside_range.groupby("side")
    }
    
    # Ensure keys for both "A" and "B" exist in the dictionary
    result.setdefault("A", [])
    result.setdefault("B", [])
    
    return result


# Process the file and save the output
output_path = "data/processed_wire_data.csv"
process_wire_data(t, output_path)


