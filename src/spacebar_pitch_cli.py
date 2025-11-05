"""CLI to trigger valve pulses and pitch estimation on spacebar presses."""

from __future__ import annotations

import argparse
import math
import sys
import termios
import tty
from dataclasses import asdict
from typing import Iterable

from spectrum_analysis.comb_trigger import HarmonicCombConfig, record_with_harmonic_comb
from spectrum_analysis.crepe_analysis import estimate_pitch_from_audio
from valve_trigger import DeviceNotFoundError, ValveController


class TerminalMode:
    """Context manager enabling cbreak mode for the current terminal."""

    def __init__(self, fd: int):
        self._fd = fd
        self._original = termios.tcgetattr(fd)

    def __enter__(self) -> None:
        tty.setcbreak(self._fd)
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: BaseException | None,
    ) -> None:
        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._original)


def _ensure_tty() -> None:
    if not sys.stdin.isatty():
        raise RuntimeError("The CLI requires an interactive terminal.")


def _parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fire a 3 ms valve pulse, record with the harmonic comb trigger, "
            "and estimate pitch whenever the spacebar is pressed."
        )
    )
    parser.add_argument(
        "--expected-frequency",
        type=float,
        required=True,
        help="Expected fundamental frequency of the string in Hertz.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=44100,
        help="Sample rate to use for microphone recording (Hz).",
    )
    parser.add_argument(
        "--max-record-seconds",
        type=float,
        default=2.0,
        help="Maximum recording duration for each capture (seconds).",
    )
    parser.add_argument(
        "--port",
        help="Optional serial port path for the valve relay (auto-detected when omitted).",
    )
    parser.add_argument(
        "--comb-frame-size",
        type=int,
        default=HarmonicCombConfig.frame_size,
        help="Frame size for the harmonic comb trigger.",
    )
    parser.add_argument(
        "--comb-hop-size",
        type=int,
        default=HarmonicCombConfig.hop_size,
        help="Hop size for the harmonic comb trigger.",
    )
    parser.add_argument(
        "--comb-candidate-count",
        type=int,
        default=HarmonicCombConfig.candidate_count,
        help="Number of pitch candidates considered by the comb trigger.",
    )
    parser.add_argument(
        "--comb-harmonic-weight-count",
        type=int,
        default=HarmonicCombConfig.harmonic_weight_count,
        help="Number of harmonics weighted when scoring candidates.",
    )
    parser.add_argument(
        "--comb-min-harmonics",
        type=int,
        default=HarmonicCombConfig.min_harmonics,
        help="Minimum harmonics required to trigger recording.",
    )
    parser.add_argument(
        "--comb-on-rmax",
        type=float,
        default=HarmonicCombConfig.on_rmax,
        help="On-threshold for comb response (higher is more selective).",
    )
    parser.add_argument(
        "--comb-off-rmax",
        type=float,
        default=HarmonicCombConfig.off_rmax,
        help="Off-threshold for comb response (lower keeps recording longer).",
    )
    parser.add_argument(
        "--comb-sfm-max",
        type=float,
        default=HarmonicCombConfig.sfm_max,
        help="Maximum spectral flatness allowed during triggering.",
    )
    parser.add_argument(
        "--comb-on-frames",
        type=int,
        default=HarmonicCombConfig.on_frames,
        help="Number of consecutive frames above the on-threshold required to start recording.",
    )
    parser.add_argument(
        "--comb-off-frames",
        type=int,
        default=HarmonicCombConfig.off_frames,
        help="Number of consecutive frames below the off-threshold required to stop recording.",
    )
    return parser.parse_args(list(argv))


def _build_comb_config(args: argparse.Namespace) -> HarmonicCombConfig:
    return HarmonicCombConfig(
        frame_size=args.comb_frame_size,
        hop_size=args.comb_hop_size,
        candidate_count=args.comb_candidate_count,
        harmonic_weight_count=args.comb_harmonic_weight_count,
        min_harmonics=args.comb_min_harmonics,
        on_rmax=args.comb_on_rmax,
        off_rmax=args.comb_off_rmax,
        sfm_max=args.comb_sfm_max,
        on_frames=args.comb_on_frames,
        off_frames=args.comb_off_frames,
    )


def _validate_args(args: argparse.Namespace) -> bool:
    valid = True
    if args.expected_frequency <= 0:
        print("Error: --expected-frequency must be positive.", file=sys.stderr)
        valid = False
    if args.sample_rate <= 0:
        print("Error: --sample-rate must be positive.", file=sys.stderr)
        valid = False
    if args.max_record_seconds <= 0:
        print("Error: --max-record-seconds must be positive.", file=sys.stderr)
        valid = False
    return valid


def _format_value(label: str, value: float) -> str:
    if math.isnan(value) or not math.isfinite(value):
        return f"{label}: nan"
    if label == "confidence":
        return f"{label}: {value:.3f}"
    return f"{label}: {value:.2f}"


def _handle_trigger(
    controller: ValveController,
    comb_cfg: HarmonicCombConfig,
    *,
    sample_rate: int,
    expected_frequency: float,
    max_record_seconds: float,
) -> None:
    print("\n[INFO] Triggering valve pulse.", flush=True)
    try:
        controller.pulse(0.003)
    except Exception as exc:  # pragma: no cover - hardware interaction
        print(f"[ERROR] Failed to pulse valve: {exc}", file=sys.stderr)
        return

    try:
        audio = record_with_harmonic_comb(
            expected_f0=expected_frequency,
            sample_rate=sample_rate,
            max_record_seconds=max_record_seconds,
            comb_cfg=comb_cfg,
        )
    except Exception as exc:  # pragma: no cover - microphone interaction
        print(f"[ERROR] Recording failed: {exc}", file=sys.stderr)
        return

    if audio.size == 0:
        print("[WARN] No audio captured.")
        return

    try:
        frequency, confidence = estimate_pitch_from_audio(
            audio, sample_rate, expected_frequency
        )
    except Exception as exc:
        print(f"[ERROR] Pitch estimation failed: {exc}", file=sys.stderr)
        return

    freq_text = _format_value("frequency", frequency)
    confidence_text = _format_value("confidence", confidence)
    print(f"[RESULT] {freq_text} Hz, {confidence_text}")


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    if not _validate_args(args):
        return 2

    try:
        _ensure_tty()
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    try:
        controller = ValveController(port=args.port)
    except DeviceNotFoundError:
        print("Error: Air valve controller not found.", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    comb_cfg = _build_comb_config(args)
    config_summary = asdict(comb_cfg)
    print("Press SPACE to initiate a measurement, or 'q' to quit.")
    print(
        "Configured harmonic comb trigger:",
        ", ".join(f"{key}={value}" for key, value in config_summary.items()),
    )

    fd = sys.stdin.fileno()
    try:
        with TerminalMode(fd):
            while True:
                char = sys.stdin.read(1)
                if char == " ":
                    _handle_trigger(
                        controller,
                        comb_cfg,
                        sample_rate=args.sample_rate,
                        expected_frequency=args.expected_frequency,
                        max_record_seconds=args.max_record_seconds,
                    )
                elif char in {"q", "Q", "\x04"}:  # Ctrl+D
                    break
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        controller.close()

    print("Exiting.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
