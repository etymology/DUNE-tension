# Streaming Overhaul With Harmonic-Comb Scoring

## Summary
- Add a new streaming measurement stack alongside the current episodic `Tensiometer` flow.
- Use `_harmonic_comb_response(...)` from [`src/spectrum_analysis/comb_trigger.py`](/Users/ben/DUNE-tension/src/spectrum_analysis/comb_trigger.py) as the v1 fast online voicedness or harmonicity scorer.
- Split streaming into two operational paths on one shared runtime:
  - `stream_sweep`: cover area quickly, collect provisional wire evidence, queue low-confidence regions or wires.
  - `stream_rescue`: optimize one wire locally for difficult or dubious cases.
- Keep pulsed air as the only v1 excitation mode; document continuous-air support as a later `ExcitationService` variant.
- Use commanded motion plus linear interpolation during constant-speed cruise windows for audio-to-pose alignment; explicitly exclude or mask accel/decel intervals.

## Key Changes
- Add a new `dune_tension.streaming` package with:
  - `MeasurementRuntime` for runtime assembly of motion, valve, audio, persistence, and analysis services.
  - `StreamingMeasurementController` with `run_sweep(...)` and `run_rescue(...)`.
  - `AudioStreamService` for continuous microphone capture on a background thread.
  - `PoseInterpolator` for timestamped XY and focus reconstruction from commanded segments.
  - `FocusPlaneModel` with initial planar form `focus = ax + by + c`.
  - `StreamingFrame`, `StreamingSegment`, and `WireCandidate` data models.
- Reuse `_harmonic_comb_response(...)` as the online frame scorer instead of inventing a new v1 voicedness metric.
  - Wrap it in a streaming analyzer that emits per-hop `comb_score`, `spectral_flatness`, `harmonic_valid`, RMS, and optional expected-band score.
  - Sweep mode uses mostly model-free harmonicity plus flatness gating.
  - Rescue mode adds target-wire weighting around the expected frequency band.
- Add an asynchronous slow pitch stage.
  - Buffer only frames or windows that pass the comb-based gate.
  - Run PESTO off the capture thread to produce pitch-lock confirmation and final frequency estimates.
  - Never let PESTO inference block motion or audio capture.
- Add sweep-session storage under `data/streaming_runs/<session_id>/`.
  - `manifest.json` for config, anchors, and focus-plane coefficients.
  - `streaming.db` for segments, pulses, aligned frames, provisional wire candidates, and rescue queue state.
  - chunked raw audio files referenced by timestamp and segment id.
- Keep final wire measurements in the existing tension DB, but extend the record shape with streaming provenance:
  - `measurement_mode`
  - `stream_session_id`
- Add additive GUI support:
  - measurement mode selector: `Legacy`, `Streaming Sweep`, `Streaming Rescue`
  - live status for current segment, comb score, pitch-lock status, and rescue-queue depth
  - keep existing legacy controls and workflow intact

## Sweep and Rescue Behavior
- `stream_sweep`
  - Begin with anchor calibration on selected wires or endpoints to seed the focus plane.
  - Execute constant-speed corridor scans with focus commanded from the current plane estimate.
  - Fire pulsed-air stimuli only during cruise windows.
  - Align comb-scored audio frames to interpolated pose and accumulate provisional evidence by nearby predicted wires.
  - Promote strong, pitch-locked candidates directly to final results.
  - Queue weak, conflicting, or ambiguous candidates for `stream_rescue`.
- `stream_rescue`
  - Start from the predicted pose or the best provisional pose from sweep.
  - Run short along-wire and focus micro-scans.
  - Maximize comb-based harmonicity first, then finalize with PESTO pitch-lock and tension plausibility.
  - Use strong rescue results to refine the focus-plane model online.

## Dataset and Evaluation
- Treat the real `.wav` corpus in `data/pitch_comparison/` and related fixture directories as a first-class v1 asset.
- Assume each file can be joined to expected-frequency labels, but not necessarily full pose metadata.
- Add an offline replay harness that:
  - streams `.wav` files through the same comb-based online analyzer used at runtime
  - benchmarks gate thresholds, false positives, trigger latency, and downstream pitch-lock success
  - produces summary reports for threshold tuning before machine testing
- Scope corpus use to evaluation plus parameter tuning only.
  - Do not train a new supervised classifier in v1.
  - Keep any learned model work out of the first implementation.
- Reuse the existing directory-processing capability in [`src/spectrum_analysis/compare_pitch_cli.py`](/Users/ben/DUNE-tension/src/spectrum_analysis/compare_pitch_cli.py) as the starting point for the replay workflow, but move streaming-specific evaluation into a dedicated replay script or module.

## Test Plan
- Unit-test the comb-wrapper analyzer around `_harmonic_comb_response(...)` for:
  - clean harmonic ringdown
  - broadband noise
  - adjacent-layer interference
  - weak pitched signals near threshold
- Unit-test `PoseInterpolator` on constant-speed segments with known accel/decel exclusion windows.
- Unit-test async buffering so delayed PESTO inference cannot stall frame ingestion or segment execution.
- Unit-test sweep aggregation from aligned frames into per-wire provisional candidates.
- Unit-test rescue convergence on simulated reward surfaces for along-wire position and focus.
- Add offline replay tests over a small locked fixture subset of real `.wav` files with expected frequencies.
- Add larger non-CI evaluation scripts for the full corpus to tune thresholds and compare scorer revisions.
- Regression-test the current legacy measurement and GUI paths to ensure fallback behavior is unchanged.

## Assumptions
- V1 reuses `_harmonic_comb_response(...)` as the fast online scorer rather than introducing a new classifier.
- The real `.wav` corpus is available locally and can be joined to expected-frequency labels.
- V1 uses the corpus for evaluation and threshold tuning, not supervised training.
- PLC pose is reconstructed from commanded trajectories plus interpolation during cruise, with accel/decel windows excluded from streaming decisions.
- The focus surface is approximated as planar initially and updated only from high-confidence rescue outcomes.
