from __future__ import annotations

import argparse
import json
from pathlib import Path

from dune_tension.streaming.controller import (
    StreamingControllerConfig,
    StreamingMeasurementController,
    SweepCorridor,
)
from dune_tension.streaming.focus_plane import FocusPlaneModel
from dune_tension.streaming.models import FocusAnchor
from dune_tension.streaming.replay import analyze_wav_paths, iter_wav_paths, write_summary_csv
from dune_tension.streaming.runtime import build_measurement_runtime


def _add_common_controller_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--apa-name", required=True)
    parser.add_argument("--layer", required=True)
    parser.add_argument("--side", required=True)
    parser.add_argument("--flipped", action="store_true")
    parser.add_argument("--sample-rate", type=int, default=44100)


def _controller_from_args(args: argparse.Namespace) -> StreamingMeasurementController:
    runtime = build_measurement_runtime()
    config = StreamingControllerConfig(
        apa_name=args.apa_name,
        layer=args.layer,
        side=args.side,
        flipped=bool(args.flipped),
        sample_rate=int(args.sample_rate),
    )
    return StreamingMeasurementController(runtime=runtime, config=config)


def main_stream_sweep(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="dune-tension-stream-sweep")
    _add_common_controller_args(parser)
    parser.add_argument(
        "--corridor",
        action="append",
        required=True,
        help="corridor_id:x0,y0,x1,y1,speed_mm_s[,focus_offset]",
    )
    args = parser.parse_args(argv)
    controller = _controller_from_args(args)
    corridors: list[SweepCorridor] = []
    for raw in args.corridor:
        corridor_id, values = raw.split(":", 1)
        parsed = [float(value) for value in values.split(",")]
        if len(parsed) not in {5, 6}:
            raise SystemExit(f"Invalid corridor specification: {raw}")
        corridors.append(
            SweepCorridor(
                corridor_id=corridor_id,
                x0=parsed[0],
                y0=parsed[1],
                x1=parsed[2],
                y1=parsed[3],
                speed_mm_s=parsed[4],
                focus_offset=parsed[5] if len(parsed) == 6 else 0.0,
            )
        )
    print(json.dumps(controller.run_sweep(corridors), indent=2, sort_keys=True))


def main_stream_rescue(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="dune-tension-stream-rescue")
    _add_common_controller_args(parser)
    parser.add_argument("--wire-number", required=True, type=int)
    args = parser.parse_args(argv)
    controller = _controller_from_args(args)
    print(
        json.dumps(
            controller.run_rescue(int(args.wire_number)),
            indent=2,
            sort_keys=True,
        )
    )


def main_fit_focus(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="dune-tension-fit-focus")
    parser.add_argument("json_path", help="JSON file containing [{'x_true','y_true','focus'}...]")
    args = parser.parse_args(argv)
    anchors_raw = json.loads(Path(args.json_path).read_text(encoding="utf-8"))
    anchors = [
        FocusAnchor(
            anchor_id=f"anchor-{index}",
            x_true=float(item["x_true"]),
            y_true=float(item["y_true"]),
            focus=float(item["focus"]),
            source=str(item.get("source", "file")),
        )
        for index, item in enumerate(anchors_raw)
    ]
    model = FocusPlaneModel.fit_from_anchors(anchors)
    print(json.dumps(model.coefficients(), indent=2, sort_keys=True))


def main_stream_replay(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="dune-tension-stream-replay")
    parser.add_argument("path", help="Path to one .wav file or a directory")
    parser.add_argument("--expected-frequency", type=float, default=None)
    parser.add_argument("--csv-out", default=None)
    args = parser.parse_args(argv)
    wav_paths = iter_wav_paths(args.path)
    summaries = analyze_wav_paths(
        wav_paths,
        expected_frequency_hz=args.expected_frequency,
    )
    if args.csv_out:
        write_summary_csv(summaries, args.csv_out)
    print(
        json.dumps(
            [summary.__dict__ for summary in summaries],
            indent=2,
            sort_keys=True,
        )
    )
