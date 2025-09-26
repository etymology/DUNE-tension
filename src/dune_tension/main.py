"""Entry point for the tensiometer GUI."""

from __future__ import annotations

from .gui import run_app


def main() -> None:
    run_app()


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
