"""Entry point for the tensiometer GUI."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None or __package__ == "":  # pragma: no cover - script execution shim
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dune_tension.gui import run_app


def main() -> None:
    run_app()


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
