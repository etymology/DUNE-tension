# Streaming Implementation Status

This document describes the streaming code that currently exists in the repo,
how it is wired into the application, and what still needs to be implemented or
validated before the streaming path should be treated as production-ready.

It complements `docs/streaming_implementation.md`:

- `streaming_implementation.md` is the design and target-state document.
- `streaming_status.md` is the current code-status document.

## Summary

The repo now contains a real additive streaming stack under
`src/dune_tension/streaming/`. The streaming path is no longer just a plan.

Implemented:

- shared streaming data models, pose math, evidence bins, and session storage
- a public harmonic-comb helper reused by streaming analysis
- a headless streaming runtime, sweep controller, rescue controller, replay
  harness, and CLI entry points
- tension-result provenance fields for streaming runs
- GUI mode selection and dispatch for `legacy`, `stream_sweep`, and
  `stream_rescue`
- basic automated tests for the streaming foundations, runtime, controller, and
  GUI dispatch

Not yet complete:

- focus-plane bootstrap from historical data or anchor measurements
- large-area comb-safe corridor planning
- real-time mid-segment decision-making
- robust hardware validation on the actual machine
- replay evaluation against labeled real `.wav` data at scale
- visualization of the evidence field and session artifacts

## Implemented Code

### Shared models and geometry

These modules now exist and are in use:

- `src/dune_tension/streaming/models.py`
  - `MeasurementMode`
  - `MeasurementPose`
  - `StreamingSegment`
  - `StreamingFrame`
  - `VoicedWindow`
  - `PitchObservation`
  - `PitchEvidenceBin`
  - `WireCandidate`
  - `StreamingManifest`
  - related session and queue records
- `src/dune_tension/streaming/pose.py`
  - `focus_side_sign(...)`
  - `focus_to_x_delta_mm(...)`
  - `stage_x_for_laser_target(...)`
  - `build_measurement_pose(...)`
  - `interpolate_segment_pose(...)`
- `src/dune_tension/streaming/focus_plane.py`
  - `FocusPlaneModel`
  - planar fit from anchors
  - focus prediction and clamp hooks
- `src/dune_tension/streaming/wire_positions.py`
  - `StreamingWirePositionProvider`
  - predicted wire lookup from the existing cached `WirePositionProvider`
  - active-layer wire direction and competing-layer direction helpers

These pieces encode the joint-pose model from the plan:

- stage XY is treated as the true position when focus is correct
- focus error produces a relative X correction
- the controller works with both the stage position and the derived laser
  position

### Harmonic-comb reuse

The harmonic comb response is now public:

- `src/spectrum_analysis/comb_trigger.py`
  - `harmonic_comb_response(...)`

That function is used by the streaming fast analyzer instead of depending on a
private underscore-only helper.

### Evidence aggregation

`src/dune_tension/streaming/evidence.py` now provides:

- `merge_pitch_confidence(...)`
- `PitchEvidenceField`

Current behavior:

- observations are binned in true-position space
- compatible pitch observations merge into one local hypothesis
- confidence grows monotonically as compatible measurements accumulate
- the bin retains a small `focus_response` summary over repeated focus offsets

This is the first concrete implementation of the intermediate evidence layer
called for by the plan. The controller does not jump directly from one pitch
result to one final wire.

### Session storage

`src/dune_tension/streaming/storage.py` now provides:

- `StreamingSessionRepository`
- `make_stream_session_id(...)`

Current session output:

- `data/streaming_runs/<session_id>/manifest.json`
- `data/streaming_runs/<session_id>/streaming.db`
- `data/streaming_runs/<session_id>/audio/*.wav`

Current `streaming.db` tables:

- `segments`
- `pulse_events`
- `frames`
- `voiced_windows`
- `pitch_results`
- `pitch_bins`
- `wire_candidates`
- `rescue_queue`
- `anchors`
- `audio_chunks`

The repository was made thread-safe because the pulse scheduler writes from a
background thread.

### Runtime and analysis services

These modules now exist:

- `src/dune_tension/streaming/runtime.py`
  - `TimedAudioChunk`
  - `AudioStreamService`
  - `MeasurementRuntime`
  - `build_measurement_runtime(...)`
- `src/dune_tension/streaming/analysis.py`
  - `StreamingAnalysisConfig`
  - `FastFrameAnalyzer`
  - `AsyncPitchWorker`

Current analysis behavior:

- microphone audio is read continuously into timestamped chunks
- each chunk is scored frame-by-frame with the harmonic comb
- voiced windows are built from adjacent passing frames
- shortlisted windows are sent to a background PESTO worker
- rescue windows are prioritized over sweep windows when the queue is full

### Sweep and rescue controller

`src/dune_tension/streaming/controller.py` now provides:

- `SweepCorridor`
- `StreamingControllerConfig`
- `build_corridors_for_wire_numbers(...)`
- `StreamingMeasurementController`
  - `run_sweep(...)`
  - `run_rescue(...)`
  - `close()`

Current `run_sweep(...)` behavior:

- creates a streaming session and manifest
- builds one `StreamingSegment` per requested corridor
- sets focus from the current `FocusPlaneModel`
- performs one stage move across the corridor at the requested speed
- schedules air pulses during the cruise window at a fixed interval
- drains captured audio after the segment
- analyzes chunks into frames, voiced windows, and pitch results
- updates pitch evidence bins
- aggregates local evidence into provisional `WireCandidate` objects
- directly accepts strong candidates
- queues weaker candidates into `rescue_queue`
- writes accepted results to the main tension DB with streaming provenance

Current `run_rescue(...)` behavior:

- starts from the cached historical wire position
- derives expected frequency from wire length
- runs a small local grid:
  - three along-wire offsets
  - three focus offsets
- records rescue session artifacts
- analyzes rescue audio with the same fast stage and async pitch stage
- accepts the best recovered candidate for the target wire
- writes the result to the main tension DB
- adds the accepted rescue point as a focus anchor and refits the in-memory
  plane

### Replay and CLI entry points

These modules now exist:

- `src/dune_tension/streaming/replay.py`
  - `.wav` reader
  - online-analyzer replay over files or directories
  - CSV summary writer
- `src/dune_tension/streaming/cli.py`
  - `main_stream_sweep(...)`
  - `main_stream_rescue(...)`
  - `main_fit_focus(...)`
  - `main_stream_replay(...)`

These scripts are registered in `pyproject.toml`:

- `dune-tension-stream-sweep`
- `dune-tension-stream-rescue`
- `dune-tension-fit-focus`
- `dune-tension-stream-replay`

### GUI integration

The existing Tk GUI now has streaming-mode support:

- `src/dune_tension/gui/context.py`
  - added `measurement_mode_var`
  - added live streaming status variables
- `src/dune_tension/gui/app.py`
  - added the mode selector
  - added status labels for segment, comb score, focus prediction, pitch
    backlog, and rescue queue depth
- `src/dune_tension/gui/state.py`
  - persists and reloads the selected measurement mode
- `src/dune_tension/gui/actions.py`
  - captures `measurement_mode`
  - constructs a `StreamingMeasurementController` when mode is not `legacy`
  - dispatches the existing measurement actions into legacy or streaming flows

Current GUI behavior:

- `legacy` mode uses the existing `Tensiometer`
- `stream_sweep` mode converts requested wires into short along-wire corridors
  and runs `run_sweep(...)`
- `stream_rescue` mode runs `run_rescue(...)` wire-by-wire using the existing
  buttons

This is intentionally minimal integration. The existing buttons are reused
rather than replaced with a fully new streaming-specific UI.

### Result-schema changes

`src/dune_tension/results.py` now adds:

- `measurement_mode`
- `stream_session_id`

These fields flow through the existing SQLite schema machinery in
`src/dune_tension/data_cache.py`, so legacy and streaming results can coexist in
the same DB.

## Automated Coverage Added

The repo now contains dedicated streaming tests:

- `tests/test_streaming_foundation.py`
  - result provenance defaults
  - pose interpolation and focus/X correction
  - evidence-bin confidence growth
  - focus-plane fitting
  - session-repository writes
  - harmonic-comb detection on synthetic audio
- `tests/test_streaming_runtime_and_controller.py`
  - fast analyzer voiced-window extraction
  - async pitch worker behavior
  - spoofed sweep-session success path

The GUI regression coverage was also extended:

- `tests/test_gui_actions.py`
  - streaming dispatch from the existing measurement actions

At the time of this update, the focused regression command used during
implementation was:

```bash
uv run --with pytest pytest -q \
  tests/test_gui_actions.py \
  tests/test_gui_app.py \
  tests/test_streaming_foundation.py \
  tests/test_streaming_runtime_and_controller.py \
  tests/test_results.py \
  tests/test_data_cache.py \
  tests/test_services.py
```

## What Is Still Simplified

The current implementation is a usable scaffold, but several parts are still
intentionally simple.

### Focus plane bootstrap is not wired

The `FocusPlaneModel` abstraction exists, but the production bootstrap path
from historical measurements or explicit anchor acquisition is not implemented.

Important consequence:

- the runtime currently creates an empty `FocusPlaneModel`
- the GUI does not yet seed it from historical `focus_position` rows
- the GUI does not yet run an anchor-calibration step before streaming

This is a major blocker for trusting the streaming path on hardware. The focus
model needs to be initialized before real use.

### Sweep planning is still per-wire and local

The design doc calls for corridor-level planning across comb-safe regions. The
current GUI path does not do that yet.

Current behavior:

- `build_corridors_for_wire_numbers(...)` creates one short along-wire corridor
  per requested wire
- default corridor extent is `1.0 mm` in each along-wire direction
- default speed is `5.0 mm/s`

Missing:

- comb-boundary-aware large-corridor planning
- automatic sweep tiling over regions
- grouping nearby requested wires into shared corridors

### Sweep analysis is segment-batched, not truly online

The code captures audio continuously, but the current controller drains and
analyzes the available chunks after each segment completes.

That means:

- motion is not yet adapting mid-segment to streaming evidence
- rescue queue decisions are effectively segment-level, not continuously online

This is acceptable for a first implementation, but it is not the full intended
streaming control loop from the design doc.

### Focus probing policy is manual, not automatic

The data model supports focus offsets and focus-response accumulation, but the
controller does not yet implement a corridor-level automatic `{-D, 0, +D}`
probe policy.

Current behavior:

- `SweepCorridor` supports `focus_offset`
- rescue explores focus offsets
- sweep does not yet automatically schedule repeated offset passes

### Candidate scoring is still heuristic

Current wire-candidate aggregation is deliberately simple:

- nearby-wire assignment by radius
- confidence from aggregated local pitch bins
- angle coherence from line-like scatter against active and competing layer
  directions
- focus consistency from the best local focus-response point

Still missing:

- stronger ambiguity resolution between nearby wires
- more principled rescue prioritization
- richer use of pulse timing and ringdown structure
- calibrated thresholds from the real corpus

### Rescue search is still a small local grid

Current rescue is useful, but it is not a sophisticated optimizer yet.

Current behavior:

- three along-wire offsets
- three focus offsets
- no explicit cross-wire offsets
- no adaptive search after the first local pass

### GUI integration is intentionally thin

The GUI can now launch streaming runs, but it is still a thin wrapper around the
controller.

Missing:

- dedicated streaming-specific parameter controls
- evidence-field visualization
- streaming-session browser and artifact inspector
- corridor editor or planner UI

## What Still Needs To Be Implemented

These are the highest-value remaining code tasks.

### Focus and geometry

- Load historical focus anchors from the tension DB and fit an initial focus
  plane.
- Add an explicit anchor-acquisition workflow before sweep.
- Decide how GUI-selected focus should seed the plane when no historical fit is
  available.
- Validate and enforce servo limits in the full runtime path.

### Sweep planning and motion

- Implement comb-safe corridor planning over larger regions.
- Split or reject sweep paths that would cross comb boundaries.
- Add bounds checking and planner validation before stage motion.
- Add a real sparse focus-probing policy during sweep.

### Streaming control loop

- Move from post-segment chunk draining to truly concurrent streaming analysis.
- Allow rescue queueing and status updates while a sweep is still in progress.
- Make pulse timing and refractory handling more realistic on the actual valve
  hardware.

### Evidence and candidate logic

- Tune bin size, merge tolerances, and acceptance thresholds on real data.
- Strengthen nearby-wire disambiguation.
- Use repeated independent sweeps to merge evidence more deliberately.
- Add session-analysis helpers that can inspect `streaming.db` without manual
  SQLite work.

### Replay and evaluation

- Connect replay to expected-frequency labels for the real `.wav` corpus.
- Add saved reports comparing threshold configurations.
- Add a locked fixture subset from the real corpus for regression testing.

### UX and operations

- Add dedicated streaming controls to the GUI.
- Add summary plots or evidence overlays for debugging runs.
- Document an operating procedure for:
  - fit focus
  - anchor
  - sweep
  - rescue
  - review session artifacts

## What Still Needs To Be Tested

### Real hardware tests

These cannot be replaced by the current spoofed tests.

- Verify the actual focus/X coupling during streaming moves.
- Verify that commanded stage speed and `cruise_margin_s` produce an honest
  cruise window.
- Verify that valve pulse timing is safe for the solenoid and does not retrigger
  too aggressively on already-ringing wires.
- Verify microphone throughput and queue behavior during longer runs.
- Verify GUI interrupt and cleanup against real PLC, servo, valve, and audio
  hardware.

### Real measurement tests

- Run rescue on wires with known good historical measurements and compare the
  final tension output.
- Run streaming sweep on a small wire group and inspect:
  - session artifacts
  - accepted wires
  - queued rescue wires
  - false nearby-wire assignments
- Check adjacent-layer discrimination on real cases where focus alone is not
  enough.

### Corpus-based offline tests

- Replay the real `.wav` corpus with labels and measure:
  - false positive rate
  - missed voiced windows
  - pitch error after PESTO
  - expected-frequency-aware versus model-free gating
- Keep representative failures as fixed regression fixtures.

## Recommended Next Step

The next code task should be focus-plane bootstrap and anchor handling. That is
the main gap between the current scaffold and a hardware-credible streaming
workflow. Without it, the joint focus and position model is present in code but
not yet initialized in a way that should be trusted on the machine.
