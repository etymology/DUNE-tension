import os
import numpy as np
import pandas as pd
from datetime import datetime
from glob import glob

# Define directories
audio_dir = "audio"
csv_dir = "data/tension_data"
output_dir = "data/renamed_audio"
os.makedirs(output_dir, exist_ok=True)

# Set constant sample rate
samplerate = 44100


# Parse timestamp from filename
def parse_audio_filename(filename):
    # e.g., GB20_2025-05-23_14-18-33.npz
    base = os.path.basename(filename).replace(".npz", "")
    prefix, timestamp = base.split("_", 1)
    layer = prefix[0]
    side = prefix[1]
    wire_number = int(prefix[2:])
    dt = datetime.strptime(timestamp, "%Y-%m-%d_%H-%M-%S")
    return layer, side, wire_number, dt


expected_columns = [
    "layer",
    "side",
    "wire_number",
    "tension",
    "tension_pass",
    "frequency",
    "zone",
    "confidence",
    "t_sigma",
    "x",
    "y",
    "wires",
    "ttf",
    "time",
]

# Load all tension dataframes indexed by apa_name and layer
csv_files = glob(os.path.join(csv_dir, "tension_data_*.csv"))
csv_data = {}
for path in csv_files:
    base = os.path.basename(path)
    parts = base.replace(".csv", "").replace("tension_data_", "").rsplit("_", 1)
    apa_name = parts[0]
    layer = parts[1]
    df = pd.read_csv(path, usecols=["time", "wire_number", "side"])
    df["parsed_time"] = pd.to_datetime(
        df["time"], format="%Y-%m-%d_%H-%M-%S", errors="coerce"
    )
    csv_data[(apa_name, layer)] = df

# Process each audio file
audio_files = glob(os.path.join(audio_dir, "*.npz"))
for path in audio_files:
    layer, side, wire_number, audio_dt = parse_audio_filename(path)
    arr = np.load(path, allow_pickle=True)["arr_0"]

    best_match = None
    min_time_diff = pd.Timedelta.max

    # Search all csvs with matching layer
    for (apa_name, csv_layer), df in csv_data.items():
        if csv_layer != layer:
            continue
        # Filter to rows with matching side and wire_number
        matches = df[
            (df["side"] == side)
            & (df["wire_number"] == wire_number)
            & df["parsed_time"].notna()
        ]
        if matches.empty:
            continue
        # Find row with closest time
        time_diffs = (matches["parsed_time"] - audio_dt).abs()
        closest_idx = time_diffs.idxmin()
        if time_diffs[closest_idx] < min_time_diff:
            min_time_diff = time_diffs[closest_idx]
            best_match = (
                apa_name,
                layer,
                side,
                matches.loc[closest_idx, "parsed_time"],
            )

    if best_match:
        apa_name, layer, side, matched_time = best_match
        output_name = f"{apa_name}_{layer}_{side}_{wire_number}_{matched_time.strftime('%Y-%m-%d_%H-%M-%S')}.npz"
        output_path = os.path.join(output_dir, output_name)
        np.savez(output_path, audio=arr, samplerate=samplerate)

output_dir
