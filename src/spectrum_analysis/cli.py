"""Command-line entrypoints for the spectrogram scroller."""
from __future__ import annotations

import argparse

from .audio import DemoSource, MicSource, sd
from .visualizer import ScrollingSpectrogram, SpectrogramConfig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrolling Spectrogram + Spectrum + Autocorrelation views"
    )
    parser.add_argument("--samplerate", type=int, default=44100)
    parser.add_argument("--fft", type=int, default=8192)
    parser.add_argument("--hop", type=int, default=512)
    parser.add_argument("--window", type=float, default=5.0)
    parser.add_argument("--max-freq", type=float, default=2000.0)
    parser.add_argument("--db-range", type=float, default=80.0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--noise-sec", type=float, default=2.0)
    parser.add_argument("--over-sub", type=float, default=1.0)
    parser.add_argument("--no-noise-filter", action="store_true")
    parser.add_argument("--min-freq", type=float, default=10.0)
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> SpectrogramConfig:
    return SpectrogramConfig(
        samplerate=args.samplerate,
        fft_size=args.fft,
        hop=args.hop,
        window_sec=args.window,
        max_freq=args.max_freq,
        db_range=args.db_range,
        enable_noise_filter=not args.no_noise_filter,
        noise_sec=args.noise_sec,
        over_sub=args.over_sub,
        min_freq=args.min_freq,
        ac_win_sec=0.5,
    )


def create_source(args: argparse.Namespace):
    if args.demo or sd is None:
        return DemoSource(args.samplerate, args.hop)
    try:
        return MicSource(args.samplerate, args.hop, device=args.device)
    except Exception as exc:  # pragma: no cover - interactive fallback
        print(f"[WARN] Could not initialize microphone input: {exc}")
        print(
            "Falling back to demo mode. Use --device to select input or install sounddevice."
        )
        return DemoSource(args.samplerate, args.hop)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    source = create_source(args)
    config = build_config(args)
    vis = ScrollingSpectrogram(source=source, config=config)
    vis.run()


__all__ = ["parse_args", "build_config", "create_source", "main"]
