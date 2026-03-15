import os
import sqlite3
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
    "Gcode",
    "wires",
    "ttf",
    "time",
]

# Load tension data from the single SQLite database and index by apa_name/layer
csv_data = {}
db_path = os.path.join(csv_dir, "tension_data.db")
if os.path.exists(db_path):
    with sqlite3.connect(db_path) as conn:
        df_all = pd.read_sql_query(
            "SELECT apa_name, layer, time, wire_number, side FROM tension_data",
            conn,
        )
    df_all["parsed_time"] = pd.to_datetime(
        df_all["time"], format="%Y-%m-%d_%H-%M-%S", errors="coerce"
    )
    for (apa, lyr), group in df_all.groupby(["apa_name", "layer"]):
        csv_data[(apa, lyr)] = group

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
