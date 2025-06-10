import os
from pathlib import Path
import numpy as np
import pandas as pd

from dune_tension.audioProcessing import get_pitch_crepe, get_pitch_pesto


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    """Return audio array and sample rate from ``path``."""
    with np.load(path, allow_pickle=True) as data:
        files = list(data.files)
        audio = data["audio"] if "audio" in data.files else data[files[0]]
        sr = int(data["samplerate"]) if "samplerate" in files else 44100
    return audio, sr


def compare_methods(
    input_dir: str = "audio",
    output: str = "data/pitch_comparison.csv",
    diff_out: str = "data/pitch_diff.csv",
) -> None:
    """Analyze all npz files in ``input_dir`` with CREPE and Pesto."""
    rows = []
    folder = Path(input_dir)
    for npz_file in sorted(folder.glob("*.npz")):
        audio, sr = load_audio(npz_file)
        crepe_f, crepe_c = get_pitch_crepe(audio, sr)
        pesto_f, pesto_c = get_pitch_pesto(audio, sr)
        rows.append(
            {
                "file": npz_file.name,
                "method": "crepe",
                "frequency": crepe_f,
                "confidence": crepe_c,
            }
        )
        rows.append(
            {
                "file": npz_file.name,
                "method": "pesto",
                "frequency": pesto_f,
                "confidence": pesto_c,
            }
        )

    df = pd.DataFrame(rows)
    os.makedirs(Path(output).parent, exist_ok=True)
    df.to_csv(output, index=False)

    diffs = []
    for file, group in df.groupby("file"):
        if {"crepe", "pesto"} <= set(group["method"]):
            crepe_row = group[group["method"] == "crepe"].iloc[0]
            pesto_row = group[group["method"] == "pesto"].iloc[0]
            diffs.append(
                {
                    "file": file,
                    "frequency_difference": crepe_row["frequency"]
                    - pesto_row["frequency"],
                    "confidence_difference": crepe_row["confidence"]
                    - pesto_row["confidence"],
                }
            )
    if diffs:
        diff_df = pd.DataFrame(diffs)
        diff_df.to_csv(diff_out, index=False)


if __name__ == "__main__":
    compare_methods()
