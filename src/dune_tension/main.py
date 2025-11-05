"""Entry point for the simplified tensiometer GUI."""

from __future__ import annotations

from dune_tension.simple_gui import run_app


def main() -> None:
    """Launch the redesigned GUI."""

    run_app()


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
