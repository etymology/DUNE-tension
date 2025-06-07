import argparse
from pathlib import Path
import numpy as np


def rename_first_array(npz_file: Path) -> None:
    """Rename the first array in ``npz_file`` to ``'audio'`` and add ``samplerate=41000``."""
    with np.load(npz_file, allow_pickle=True) as data:
        if not data.files:
            print(f"{npz_file} contains no arrays, skipping")
            return
        arrays = {name: data[name] for name in data.files}
    first_key = data.files[0]
    audio_array = arrays.pop(first_key)
    new_arrays = {"audio": audio_array, "samplerate": np.array(41000)}
    new_arrays.update(arrays)
    tmp_path = npz_file.with_suffix(npz_file.suffix + ".tmp")
    with tmp_path.open("wb") as f:
        np.savez(f, **new_arrays)
    tmp_path.replace(npz_file)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=("Rename the first array field in npz files to 'audio' and add a samplerate field."),
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default="audio",
        help="Directory containing .npz files (default: 'audio')",
    )
    args = parser.parse_args(argv)

    folder = Path(args.directory)
    if not folder.is_dir():
        print(f"Directory {folder} does not exist")
        return 1

    for npz_file in sorted(folder.glob("*.npz")):
        print(f"Processing {npz_file}")
        rename_first_array(npz_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
