"""Command-line utility to periodically call ``save_plot`` via ``update_tension_logs``."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import Dict

from dune_tension.summaries import update_tension_logs
from dune_tension.tensiometer_functions import make_config


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Periodically execute dune_tension.summaries.save_plot for the"
            " provided APA name and layer."
        )
    )
    parser.add_argument("apa_name", help="APA name, e.g. USAPA5")
    parser.add_argument(
        "layer",
        help="APA layer to summarize (U, V, X, or G)",
        choices=["U", "V", "X", "G"],
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=900.0,
        help="Number of seconds to wait between invocations (default: 900)",
    )
    parser.add_argument(
        "--side",
        default="A",
        choices=["A", "B"],
        help=(
            "Side passed to make_config. This does not change which samples"
            " are plotted but is required for configuration (default: A)."
        ),
    )
    parser.add_argument(
        "--samples-per-wire",
        type=int,
        default=3,
        help="Minimum number of samples per wire required to plot (default: 3)",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.7,
        help="Minimum confidence for a measurement to be included (default: 0.7)",
    )
    parser.add_argument(
        "--data-path",
        default="data/tension_data/tension_data.db",
        help="Path to the SQLite database that stores tension measurements.",
    )
    parser.add_argument(
        "--run-once",
        action="store_true",
        help="If set, run a single update and exit instead of looping forever.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level (default: INFO)",
    )
    return parser.parse_args(argv)


def _build_config(args: argparse.Namespace):
    config = make_config(
        apa_name=args.apa_name,
        layer=args.layer,
        side=args.side,
        samples_per_wire=args.samples_per_wire,
        confidence_threshold=args.confidence_threshold,
        save_audio=False,
        spoof=False,
    )
    config.data_path = args.data_path
    return config


def _log_result(result: Dict[str, str]) -> None:
    for label, path in result.items():
        logging.info("%s written to %s", label, path)


def run_periodic(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    config = _build_config(args)

    logging.info(
        "Starting periodic plot generation for APA %s layer %s (interval %.1f s)",
        config.apa_name,
        config.layer,
        args.interval,
    )

    while True:
        start_time = time.monotonic()
        try:
            result_paths = update_tension_logs(config)
            _log_result(result_paths)
        except Exception:  # noqa: BLE001 - ensure the loop continues on errors
            logging.exception("Failed to update tension logs")

        if args.run_once:
            break

        elapsed = time.monotonic() - start_time
        sleep_time = max(args.interval - elapsed, 0)
        logging.debug("Sleeping %.1f seconds before the next run", sleep_time)
        time.sleep(sleep_time)

    logging.info("Finished periodic plot generation")
    return 0


def main() -> None:
    raise SystemExit(run_periodic())


if __name__ == "__main__":
    main()
