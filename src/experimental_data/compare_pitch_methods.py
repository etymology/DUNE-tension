import os
import time
from pathlib import Path
import concurrent.futures
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
    timeout: float | None = 60.0,
    batch_timeout: float | None = None,
) -> None:
    """Analyze all npz files in ``input_dir`` with CREPE and Pesto.

    ``timeout`` sets the maximum number of seconds allowed for each pitch
    extraction method. If a method does not finish within the timeout, its
    results are recorded as ``NaN``. Results and per-file differences are
    appended to their CSV outputs as processing progresses so partial data is
    preserved if execution stops early.
    """
    folder = Path(input_dir)
    start = time.monotonic()
    out_path = Path(output)
    diff_path = Path(diff_out)
    os.makedirs(out_path.parent, exist_ok=True)
    os.makedirs(diff_path.parent, exist_ok=True)
    first_write = not out_path.exists()
    first_diff_write = not diff_path.exists()
    for npz_file in sorted(folder.glob("*.npz")):
        if batch_timeout is not None and time.monotonic() - start >= batch_timeout:
            break
        audio, sr = load_audio(npz_file)
        with concurrent.futures.ThreadPoolExecutor() as exe:
            f_crepe = exe.submit(get_pitch_crepe, audio, sr)
            f_pesto = exe.submit(get_pitch_pesto, audio, sr)
            rem = None
            if batch_timeout is not None:
                rem = batch_timeout - (time.monotonic() - start)
                if rem < 0:
                    rem = 0
            per_timeout = min(timeout, rem) if rem is not None else timeout
            try:
                crepe_f, crepe_c = f_crepe.result(timeout=per_timeout)
            except concurrent.futures.TimeoutError:
                crepe_f, crepe_c = np.nan, np.nan
            if per_timeout is not None:
                rem = batch_timeout - (time.monotonic() - start) if batch_timeout is not None else None
                per_timeout = min(timeout, rem) if rem is not None else timeout
            try:
                pesto_f, pesto_c = f_pesto.result(timeout=per_timeout)
            except concurrent.futures.TimeoutError:
                pesto_f, pesto_c = np.nan, np.nan
        rows = [
            {
                "file": npz_file.name,
                "method": "crepe",
                "frequency": crepe_f,
                "confidence": crepe_c,
            },
            {
                "file": npz_file.name,
                "method": "pesto",
                "frequency": pesto_f,
                "confidence": pesto_c,
            },
        ]
        pd.DataFrame(rows).to_csv(
            out_path,
            mode="a",
            index=False,
            header=first_write,
        )
        first_write = False

        if not (np.isnan(crepe_f) or np.isnan(pesto_f)):
            diff_row = {
                "file": npz_file.name,
                "frequency_difference": crepe_f - pesto_f,
                "confidence_difference": crepe_c - pesto_c,
            }
            pd.DataFrame([diff_row]).to_csv(
                diff_path,
                mode="a",
                index=False,
                header=first_diff_write,
            )
            first_diff_write = False



if __name__ == "__main__":
    compare_methods()
