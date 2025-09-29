"""Utility for generating harmonic waveforms augmented with noise.

The module exposes helpers to synthesise band-limited waveforms that decay to
the noise floor within the requested duration.  It supports a CLI for dataset
creation as well as a :func:`generate_waveform` function that can be used in
tests.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.io import wavfile


SUPPORTED_WAVEFORMS = {"sine", "square", "triangle", "comb"}


@dataclass(slots=True)
class WaveformParameters:
    """Parameters used to generate a single waveform sample."""

    sample_rate: int
    duration: float
    base_frequency: float
    waveform: str
    num_partials: int
    gain: float
    noise_level: float
    spectral_tilt_db_per_octave: float
    partial_decay_bias: float
    seed: int | None = None
    normalize: bool = True


@dataclass(slots=True)
class WaveformResult:
    """Result returned by :func:`generate_waveform`."""

    audio: np.ndarray
    deterministic: np.ndarray
    envelopes: list[np.ndarray]


def _normalised_weight(index: int, count: int) -> float:
    if count <= 1:
        return 0.5
    return index / (count - 1)


def _decay_exponent(bias: float, weight: float) -> float:
    exponent = float(np.exp(-bias * (weight - 0.5)))
    return float(np.clip(exponent, 0.25, 4.0))


def _spectral_tilt(partial_index: int, tilt_db_per_octave: float) -> float:
    if partial_index <= 1:
        return 1.0
    octaves = np.log2(partial_index)
    return float(10 ** (tilt_db_per_octave * octaves / 20))


def _partial_amplitude(waveform: str, partial_index: int) -> float:
    if waveform == "sine":
        return 1.0 if partial_index == 1 else 0.0
    if waveform == "square":
        if partial_index % 2 == 1:
            return 1.0 / partial_index
        return 0.0
    if waveform == "triangle":
        if partial_index % 2 == 1:
            sign = -1.0 if ((partial_index - 1) // 2) % 2 else 1.0
            return sign / (partial_index**2)
        return 0.0
    # Treat the comb waveform as a dense harmonic series with gentle decay.
    if waveform == "comb":
        return 1.0 / np.sqrt(partial_index)
    msg = ", ".join(sorted(SUPPORTED_WAVEFORMS))
    raise ValueError(f"Unsupported waveform '{waveform}'. Available: {msg}.")


def generate_waveform(
    params: WaveformParameters,
    *,
    include_noise: bool = True,
    return_envelopes: bool = False,
) -> WaveformResult:
    """Generate a waveform according to ``params``.

    Parameters
    ----------
    params:
        Generation parameters.
    include_noise:
        If ``False``, no stochastic noise is added to the waveform.  This is
        useful for unit tests where deterministic behaviour is required.
    return_envelopes:
        Whether to populate the :class:`WaveformResult` with per-partial
        envelopes for downstream analysis.
    """

    if params.waveform not in SUPPORTED_WAVEFORMS:
        msg = ", ".join(sorted(SUPPORTED_WAVEFORMS))
        raise ValueError(
            f"Unsupported waveform '{params.waveform}'. Available waveforms: {msg}."
        )

    rng = np.random.default_rng(params.seed)
    sample_count = int(params.sample_rate * params.duration)
    t = np.linspace(0.0, params.duration, sample_count, endpoint=False)
    progress = np.linspace(0.0, 1.0, sample_count, endpoint=False)
    if progress.size:
        progress[-1] = 1.0

    deterministic = np.zeros_like(t, dtype=np.float64)
    envelopes: list[np.ndarray] = []

    partial_count = max(1, params.num_partials)
    for partial_index in range(1, partial_count + 1):
        base_amplitude = _partial_amplitude(params.waveform, partial_index)
        if base_amplitude == 0.0:
            if return_envelopes:
                envelopes.append(np.zeros_like(t))
            continue

        amplitude = base_amplitude * _spectral_tilt(
            partial_index, params.spectral_tilt_db_per_octave
        )
        weight = _normalised_weight(len(envelopes), partial_count)
        exponent = _decay_exponent(params.partial_decay_bias, weight)
        decay_curve = 1.0 - np.power(progress, exponent)
        envelope = amplitude * np.power(np.clip(decay_curve, 0.0, 1.0), 2.0)
        if envelope.size:
            envelope[0] = amplitude
            envelope[-1] = 0.0
        angular_frequency = 2 * np.pi * params.base_frequency * partial_index
        phase = rng.uniform(0.0, 2 * np.pi)
        component = envelope * np.sin(angular_frequency * t + phase)
        deterministic += component
        if return_envelopes:
            envelopes.append(envelope.copy())

    deterministic *= params.gain

    audio = deterministic.astype(np.float64)
    if include_noise and params.noise_level > 0.0:
        noise = rng.normal(0.0, params.noise_level, size=audio.shape)
        audio = audio + noise

    max_magnitude = np.max(np.abs(audio))
    if params.normalize and max_magnitude > 1.0e-12:
        scale = max(1.0, max_magnitude)
        audio = audio / scale
        deterministic = deterministic / scale
        if return_envelopes:
            envelopes = [env / scale for env in envelopes]

    result_audio = audio.astype(np.float32)
    result_deterministic = deterministic.astype(np.float32)
    if return_envelopes:
        return WaveformResult(result_audio, result_deterministic, envelopes)
    return WaveformResult(result_audio, result_deterministic, [])


def _sanitise_value(value: float) -> str:
    return str(round(value, 6)).replace("-", "neg").replace(".", "p")


def _iter_waveform_configs(
    waveforms: Sequence[str],
    frequencies: Sequence[float],
    num_files: int,
) -> Iterable[tuple[str, float, int]]:
    for waveform in waveforms:
        for frequency in frequencies:
            for index in range(num_files):
                yield waveform, frequency, index


def generate_dataset(
    *,
    output_directory: Path,
    sample_rate: int,
    duration: float,
    base_frequencies: Sequence[float],
    waveforms: Sequence[str],
    num_files: int,
    num_partials: int,
    gain_range: tuple[float, float],
    noise_range: tuple[float, float],
    spectral_tilt_range: tuple[float, float],
    bias_range: tuple[float, float],
    seed: int | None = None,
) -> list[dict[str, float | str]]:
    """Generate a dataset of audio samples and return metadata for each file."""

    output_directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    metadata_rows: list[dict[str, float | str]] = []

    for waveform, frequency, index in _iter_waveform_configs(
        waveforms, base_frequencies, num_files
    ):
        gain = rng.uniform(*gain_range)
        noise_level = rng.uniform(*noise_range)
        spectral_tilt = rng.uniform(*spectral_tilt_range)
        bias = rng.uniform(*bias_range)

        params = WaveformParameters(
            sample_rate=sample_rate,
            duration=duration,
            base_frequency=frequency,
            waveform=waveform,
            num_partials=num_partials,
            gain=gain,
            noise_level=noise_level,
            spectral_tilt_db_per_octave=spectral_tilt,
            partial_decay_bias=bias,
            seed=rng.integers(0, 2**32 - 1).item(),
        )

        result = generate_waveform(params)
        filename = (
            f"{waveform}_f{_sanitise_value(frequency)}Hz_"
            f"gain{_sanitise_value(gain)}_"
            f"noise{_sanitise_value(noise_level)}_"
            f"tilt{_sanitise_value(spectral_tilt)}_"
            f"bias{_sanitise_value(bias)}_{index:03d}.wav"
        )

        path = output_directory / filename
        wavfile.write(path, sample_rate, result.audio)

        metadata_rows.append(
            {
                "filename": filename,
                "waveform": waveform,
                "frequency_hz": frequency,
                "gain": gain,
                "noise_level": noise_level,
                "spectral_tilt_db_per_octave": spectral_tilt,
                "partial_decay_bias": bias,
            }
        )

    metadata_path = output_directory / "metadata.csv"
    with metadata_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "filename",
                "waveform",
                "frequency_hz",
                "gain",
                "noise_level",
                "spectral_tilt_db_per_octave",
                "partial_decay_bias",
            ],
        )
        writer.writeheader()
        writer.writerows(metadata_rows)

    return metadata_rows


def _parse_float_list(values: Sequence[str]) -> list[float]:
    return [float(value) for value in values]


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Directory to store generated files.")
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument(
        "--waveforms",
        nargs="+",
        default=["triangle"],
        choices=sorted(SUPPORTED_WAVEFORMS),
    )
    parser.add_argument("--frequencies", nargs="+", type=float, required=True)
    parser.add_argument("--num-files", type=int, default=1)
    parser.add_argument("--num-partials", type=int, default=20)
    parser.add_argument("--gain-range", nargs=2, type=float, default=[0.1, 1.0])
    parser.add_argument("--noise-range", nargs=2, type=float, default=[0.001, 0.5])
    parser.add_argument(
        "--spectral-tilt-range",
        nargs=2,
        type=float,
        default=[-20, 20],
        help="Tilt in dB/octave applied to the harmonic series.",
    )
    parser.add_argument(
        "--bias-range",
        nargs=2,
        type=float,
        default=[-20, 20],
        help="Positive values favour faster decay for higher partials.",
    )
    parser.add_argument("--seed", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    gain_range = tuple(sorted(args.gain_range))
    noise_range = tuple(sorted(args.noise_range))
    spectral_tilt_range = tuple(sorted(args.spectral_tilt_range))
    bias_range = tuple(sorted(args.bias_range))

    generate_dataset(
        output_directory=args.output,
        sample_rate=args.sample_rate,
        duration=args.duration,
        base_frequencies=args.frequencies,
        waveforms=args.waveforms,
        num_files=args.num_files,
        num_partials=args.num_partials,
        gain_range=gain_range,  # type: ignore[arg-type]
        noise_range=noise_range,  # type: ignore[arg-type]
        spectral_tilt_range=spectral_tilt_range,  # type: ignore[arg-type]
        bias_range=bias_range,  # type: ignore[arg-type]
        seed=args.seed,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
