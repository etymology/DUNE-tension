# Streaming Audio Tension Measurement — Consolidated Implementation Design

This document supersedes the earlier concept note in `docs/streaming_plan.md`
and the intermediate draft in `docs/codex_streaming_PLAN.md`. It is the
implementation-facing plan for adding a streaming measurement architecture to
this repository without removing the current episodic workflow.

## Summary

The repo currently measures one wire at a time by moving to a predicted pose,
firing a pulse of air, recording one short audio event, estimating pitch, and
then stepping around x, y, and focus until confidence is good enough. That
works, but it is slow, couples acquisition to analysis, and does not use the
continuous structure of the search space.

The new design adds a streaming measurement stack alongside the current
`Tensiometer` flow. The streaming stack has two operational paths:

- `stream_sweep`: cover a corridor efficiently, collect provisional evidence for
  many wires, and identify weak or ambiguous regions.
- `stream_rescue`: optimize one wire locally when sweep evidence is weak, mixed,
  or suspicious.

V1 remains pulse-driven. It does not assume continuous air excitation. Audio is
captured continuously, but valve firing is scheduled deliberately during
constant-speed sweep segments or local rescue scans.

The fast online scorer is based on the existing harmonic comb logic in
`spectrum_analysis.comb_trigger`. Slow pitch confirmation is done asynchronously
with PESTO. Streaming artifacts are stored in a dedicated session directory, and
final accepted wire results continue to be written to the existing tension DB.
All streaming geometry is expressed as a joint measurement pose rather than bare
XY because focus changes are coupled to the X axis. The base XY coordinate is
the true laser position when focus is correct; the focus-induced X term is a
relative correction from that true position. The plan therefore treats wire
positioning and focusing as related but distinct models that must always be
evaluated together.

This means the efficiency gain over the current pulse-and-analyze loop is not
just "more XY samples per unit time". The intended gain is:

- replace per-wire random 3D search with a shared focus prior over a whole
  corridor
- probe focus sparsely and deliberately rather than independently at every wire
- aggregate repeated evidence at the same place across multiple passes
- reserve expensive local focus or position search for rescue only

## Why This Exists

The original `docs/streaming_plan.md` identifies four persistent measurement
problems:

- focus is found by random wiggle rather than by following a smooth reward
  surface
- x/y seeking is stochastic and can skip or duplicate wires
- adjacent layers are easy to confuse because planes are only a few millimeters
  apart
- pitch analysis is slow enough that the current workflow pauses hardware
  collection to run it

The current codebase reflects those limits:

- `src/dune_tension/tensiometer.py` still runs a strum, capture, analyze, step
  loop
- `src/dune_tension/services.py` already provides useful seams for motion,
  audio, and result persistence
- `src/dune_tension/data_cache.py` already separates raw sample rows from final
  summary rows
- `src/dune_tension/gui/actions.py` and `src/dune_tension/gui/context.py`
  already isolate GUI runtime assembly enough to add a second measurement mode
- `src/spectrum_analysis/comb_trigger.py` already contains the harmonic comb
  metric needed for fast online gating
- the repo already contains a large `.wav` corpus under `data/pitch_comparison/`
  and related fixture directories that can be used for replay evaluation and
  threshold tuning

## Chosen V1 Decisions

These decisions are fixed for the first implementation.

- The streaming stack is additive. `Tensiometer` remains available as the
  fallback mode.
- V1 supports `legacy`, `stream_sweep`, and `stream_rescue` measurement modes.
- V1 uses pulsed air only. Continuous air is deferred behind a future
  `ExcitationService` variant.
- The online reward is "harmonic or voiced enough to investigate", not final
  pitch confidence.
- The fast online reward reuses the harmonic comb response already implemented
  in `spectrum_analysis`.
- PESTO runs only on shortlisted windows or segments and never blocks motion or
  capture.
- Layer isolation uses two independent signals:
  - focus-plane consistency
  - pitch patterns that align with the current layer's wire angle
- Pose alignment uses commanded trajectories plus interpolation during
  constant-speed cruise windows.
- Acceleration and deceleration intervals are excluded from measurement scoring
  and pulse scheduling.
- Sweep segments must not cross comb boundaries. Streaming motion is constrained
  to comb-safe corridors so interpolation stays valid.
- Focus-plane calibration and wire-position calibration remain separate models.
  They are combined at runtime through a joint measurement pose.
- The focus surface is modeled as a plane `focus = ax + by + c` in V1.
- Focus optimization in V1 is two-level:
  - a global focus plane used as the default prediction
  - a local residual focus search estimated from repeated offset probes and
    rescue results
- All geometric reasoning uses a combined
  `(x_true, y_true, focus, focus_reference, x_focus_correction, x_laser)` state
  rather than raw XY alone.
- The `.wav` corpus is used for evaluation and parameter tuning only, not for
  training a supervised classifier in V1.

## V1 Scope and Non-Goals

### In scope

- a new streaming runtime under `src/dune_tension/streaming/`
- sweep plus rescue workflows
- online harmonic-comb scoring
- asynchronous PESTO confirmation
- pose interpolation and focus-plane prediction
- session-local persistence of streaming artifacts
- GUI mode selection and live streaming status
- CLI entry points for headless sweep, rescue, focus fitting, and offline replay
- replay-based evaluation using the real `.wav` corpus

### Out of scope

- removing the current `Tensiometer` path
- supervised model training for voiced or wire-quality classification
- continuous-air hardware control
- full 3D focus-surface fitting beyond a plane
- fully automatic large-area APA mapping in one pass without rescue
- replacing the existing summary plotting and tension DB schema beyond the
  minimal provenance additions described below

## Existing Code To Reuse

The new design should reuse these pieces rather than replacing them.

| Current code | Reuse in streaming stack |
| --- | --- |
| `src/dune_tension/services.py:MotionService` | motion commands, cached XY, speed control |
| `src/dune_tension/services.py:ResultRepository` | final accepted wire result persistence |
| `src/dune_tension/tensiometer.py` focus/X coupling math | focus compensation and rescue motion math |
| `src/dune_tension/tensiometer_functions.py` | `TensiometerConfig`, geometry-aware wire planning, future cached position provider |
| `src/dune_tension/summaries.py` | missing-wire lookup and GUI summary refresh |
| `src/dune_tension/gui/actions.py` | threading, stop-event handling, runtime assembly seam |
| `src/valve_trigger.py:ValveController` | pulse execution |
| `src/spectrum_analysis/audio_sources.py:MicSource` | continuous microphone capture |
| `src/spectrum_analysis/comb_trigger.py` | harmonic comb scoring logic |
| `src/spectrum_analysis/pesto_analysis.py` | slow pitch confirmation |
| `data/pitch_comparison/*.wav` and fixtures | replay evaluation and threshold tuning |

One cleanup should happen before streaming code depends on the comb metric:
extract `_harmonic_comb_response(...)` into a public pure helper so both the
existing comb trigger and the new streaming analyzer call the same function.
V1 should not build new logic around a private underscore function without
formalizing that reuse.

## High-Level Architecture

The streaming runtime is a coordinated set of background services with one
controller deciding where to move and when to fire air pulses.

```text
GUI or CLI
    |
    v
MeasurementRuntime
    |
    +-- Motion adapter
    +-- Valve adapter
    +-- Audio stream adapter
    +-- Streaming repository
    +-- Result repository
    +-- Wire position provider
    +-- Focus plane model
    |
    v
StreamingMeasurementController
    |
    +-- run_sweep(...)
    |      |
    |      +-- segment planner
    |      +-- pulse scheduler
    |      +-- online frame analyzer
    |      +-- async pitch worker
    |      +-- spatial evidence aggregator
    |      +-- provisional candidate aggregator
    |      +-- rescue queue
    |
    +-- run_rescue(...)
           |
           +-- local along-wire/focus scan
           +-- online reward maximization
           +-- pitch confirmation
           +-- final result writeback
```

The runtime separates three concerns cleanly:

- acquisition: audio stream, pulses, motion commands, timestamping
- analysis: fast harmonic-comb scoring plus asynchronous PESTO confirmation
- evidence accumulation: repeated pitch observations merged into spatial bins
- decision-making: sweep progression, candidate aggregation, rescue scheduling,
  and final acceptance

## Proposed Package Layout

Create a new package `src/dune_tension/streaming/` with these modules.

| Module | Responsibility |
| --- | --- |
| `runtime.py` | build a `MeasurementRuntime` from existing services and environment |
| `models.py` | dataclasses and typed records for segments, frames, candidates, and manifests |
| `pose.py` | segment timing, cruise-window masking, joint-pose interpolation, effective X derivation |
| `focus_plane.py` | focus-plane bootstrap, prediction, and online refinement |
| `wire_positions.py` | cached wire-position provider, wire-angle model, and sweep assignment helpers |
| `analysis.py` | fast comb-based frame metrics and async pitch worker |
| `evidence.py` | spatial pitch-bin aggregation, confidence fusion, and visualization helpers |
| `controller.py` | `StreamingMeasurementController`, sweep and rescue orchestration |
| `storage.py` | session directory creation, manifest writing, `streaming.db`, audio chunk indexing |
| `replay.py` | offline `.wav` replay harness using the same frame analyzer |
| `cli.py` | headless entry points |

The older `collector.py` sketch in the superseded draft should not be
implemented as written. V1 needs controller-driven sweep and rescue workflows,
not a single "run all threads and dump a pitch map" collector.

## Public Interfaces and Data Contracts

### Measurement mode

Add:

```python
MeasurementMode = Literal["legacy", "stream_sweep", "stream_rescue"]
```

This mode is selected in the GUI and passed through runtime assembly. The legacy
mode continues to construct `Tensiometer`. The streaming modes construct
`StreamingMeasurementController`.

### Runtime bundle

Add a `MeasurementRuntime` dataclass that owns all collaborators required by the
controller. It should contain:

- `motion`
- `valve`
- `audio_stream`
- `streaming_repo`
- `result_repo`
- `wire_positions`
- `focus_plane`
- `clock`
- `logger`

`MeasurementRuntime` is created by one outer bootstrap path. `Tensiometer`
should eventually be moved toward the same runtime-assembly pattern, but V1 only
needs to share enough bootstrap logic to avoid duplicating environment
resolution in GUI code.

### Core streaming models

The exact class names can vary, but V1 needs the following record shapes.

#### `MeasurementPose`

Represents the measurement state seen by the optical system.

Required fields:

- `x_true`
- `y_true`
- `focus`
- `focus_reference`
- `x_focus_correction`
- `x_laser`
- `side`

`x_true` and `y_true` are the reference coordinates where stage XY and laser
position agree. That equality holds when the instrument is correctly focused for
that location. `x_focus_correction` is the relative offset introduced by focus
error, and `x_laser = x_true + x_focus_correction`. The streaming stack must
not reason about XY without also carrying focus and the derived correction.

#### `StreamingSegment`

Represents one comb-safe motion segment.

Required fields:

- `segment_id`
- `mode`: `sweep` or `rescue`
- `pose0`
- `pose1`
- `speed_mm_s`
- `planned_start_time`
- `planned_end_time`
- `cruise_start_time`
- `cruise_end_time`
- `wire_hint`: optional predicted wire number for rescue
- `segment_status`

#### `StreamingFrame`

Represents one analyzed audio hop aligned to interpolated pose.

Required fields:

- `frame_id`
- `segment_id`
- `timestamp`
- `pose`
- `rms`
- `comb_score`
- `spectral_flatness`
- `harmonic_valid`
- `expected_band_score`: optional
- `voiced_gate_pass`
- `audio_chunk_ref`

#### `VoicedWindow`

Represents a short window selected for slow pitch analysis.

Required fields:

- `window_id`
- `segment_id`
- `start_time`
- `end_time`
- `pose_center`
- `wire_hint`: optional
- `audio_chunk_refs`

#### `PitchEvidenceBin`

Represents one small spatial bin in true-position space that accumulates pitch
evidence across repeated observations and independent sweeps.

Required fields:

- `bin_id`
- `x_bin`
- `y_bin`
- `bin_size_mm`
- `hypotheses`
- `source_window_count`
- `source_sweep_ids`
- `last_updated`

Each hypothesis inside a bin must track:

- `pitch_center_hz`
- `support_count`
- `weighted_pitch_hz`
- `combined_confidence`
- `max_pitch_confidence`
- `max_comb_score`
- `focus_response`

`focus_response` should summarize how confidence changes versus
`focus - focus_reference` for that local pitch hypothesis. V1 does not need a
full continuous model, but it should retain enough information to tell whether:

- the same local pitch improves when revisited
- one focus residual clearly dominates nearby alternatives
- local focus uncertainty is still large enough to justify rescue

Confidence should increase monotonically for repeated compatible observations.
V1 should use a simple fusion rule such as:

```text
combined_confidence = 1 - product(1 - clip(conf_i, 0, 1))
```

Pitch observations merge into the same hypothesis when they fall in the same
spatial bin and agree within a fixed pitch tolerance. V1 should use
`max(5 Hz, 1% relative)` as the merge tolerance.

#### `WireCandidate`

Represents provisional evidence accumulated for one predicted wire after one or
more `PitchEvidenceBin` objects have been correlated into a wire-level track.

Required fields:

- `wire_number`
- `source_mode`: `sweep` or `rescue`
- `support_count`
- `best_pose`
- `best_comb_score`
- `pitch_estimates`
- `pitch_confidences`
- `angle_coherence_score`
- `focus_consistency_score`
- `status`: `provisional`, `queued_for_rescue`, `accepted`, `rejected`

### Final wire result provenance

Extend `TensionResult` with two optional fields:

- `measurement_mode`
- `stream_session_id`

Default values:

- `measurement_mode="legacy"`
- `stream_session_id=None`

`EXPECTED_COLUMNS` is derived from `TensionResult`, so this automatically flows
into the existing SQLite append logic in `data_cache.py`. V1 should use that
mechanism rather than creating a separate final-results schema.

## Joint Pose Model

### Comb-safe segments

The current PLC path in `src/dune_tension/plc_io.py` can decompose long moves
that cross comb positions into multiple sub-moves. That behavior is fine for
legacy point moves but would make streaming interpolation wrong if ignored.

For V1:

- streaming sweep segments must stay within one comb interval
- the sweep planner must split large corridors at comb boundaries before motion
  commands are issued
- only the straight in-zone segment is considered a measurement segment
- transit moves to reposition between corridors are allowed, but no pulses are
  fired and no measurement frames are scored during those transits

### Constant-speed interpolation

The user-provided machine constraint is that PLC motion is precise and can run
at constant speed, but high-frequency actual-position polling is not required.
V1 should therefore reconstruct a joint pose from planned segments:

- record `planned_start_time` immediately before issuing the segment command
- compute expected cruise duration from segment length, speed, and known
  acceleration or jerk settings
- define explicit cruise start and end timestamps
- interpolate `x_true`, `y_true`, and `focus` only inside the cruise window
- derive `focus_reference` from the focus plane at `(x_true, y_true)`
- derive `x_focus_correction` from focus error relative to `focus_reference`
  using the same side-dependent transform as `_focus_to_x_delta_mm()` in
  `tensiometer.py`
- derive `x_laser = x_true + x_focus_correction`
- mark pre-cruise and post-cruise frames as `not_scored`

If future hardware work exposes reliable high-rate actual XY telemetry, that can
replace the interpolation path later. It is not required for V1.

### Focus-coupled X model

The recent focus/X synchronization work means focus cannot be treated as an
independent scalar attached to an XY point. The effective optical pose depends
on both stage motion and focus.

For V1:

- every segment stores full start and end measurement poses, not just XY plus a
  separate focus value
- every frame stores the derived `x_focus_correction` and `x_laser`
- when `focus == focus_reference`, stage XY and laser position are treated as
  equivalent and `x_focus_correction = 0`
- historical positions loaded from the tension DB may be treated as true wire
  positions when they come from accepted, correctly focused measurements
- rescue scans that nudge focus must account for the coupled X shift at every
  step, not only when writing final results

The existing focus/X coupling logic in `tensiometer.py` is the source of truth
for the side-dependent transform and should be reused rather than re-derived.

## Focus Plane Model

V1 uses a planar focus model:

```text
focus(x, y) = a*x + b*y + c
```

### Bootstrap

The initial focus plane is built in this order:

1. Load historical accepted points for the current APA, layer, and side from
   `tension_data.db` if there are at least three usable rows with
   `focus_position`.
2. Fit a least-squares plane to those points.
3. Run a short session anchor step before long sweeps:
   - prefer two endpoints plus one interior point, or three spatially separated
     rescue measurements
   - if the historical plane exists, use anchors to validate and optionally
     refit it
   - if no historical plane exists, anchors are mandatory

### Update policy

Only high-confidence rescue results may update the plane in V1. Sweep-only
observations are too noisy to retune focus geometry safely.

Update rules:

- require valid pitch-lock and plausible tension
- require comb score and pitch confidence above rescue thresholds
- refit plane incrementally or refit from the retained anchor plus rescue set
- clamp predicted focus to servo limits

If the plane is missing or unstable, sweep may proceed with a constant focus
value, but rescue remains responsible for final acceptance.

The focus plane is intentionally calibrated separately from wire positions. Its
job is to predict which layer is in focus for a given true XY area, not to
identify the wire on its own.

### How sweep addresses focus

Sweeping XY by itself does not solve the focus problem. It only increases the
rate at which spatial evidence is collected. Focus efficiency comes from using
the sweep as a vehicle for structured focus estimation rather than running a
fresh local focus search at every wire.

V1 should therefore treat focus as:

- a shared low-dimensional prior over the corridor, given by the focus plane
- a small local residual around that prior, estimated from sparse probes and
  rescue outcomes

The practical consequence is that most sweep segments run at the predicted focus
plane, while a smaller number of deliberately offset passes or windows sample
`focus_reference + delta_focus` for a few chosen `delta_focus` values.

Because focus is X-coupled, those offset probes are only valid if the motion is
X-compensated so the same true position is revisited while focus changes. This
is the key difference from the current random wiggle approach: the plan does not
intend to search focus and position independently.

### Focus probing policy

V1 should add a sparse focus-probing policy on top of sweep:

- default sweep uses `delta_focus = 0`
- some repeated passes over the same corridor use small offsets such as
  `delta_focus in {-D, +D}`
- commanded X is compensated during those offset passes so the same true path is
  sampled
- the resulting pitch and confidence are merged into the local evidence bins'
  `focus_response`

This makes focus estimation an amortized problem over a corridor instead of a
wire-by-wire random search.

## Wire Position Model

The streaming stack should not infer wire identity from x-position clustering
alone. The repo already has historical positions and layer geometry, and the
refactoring audit explicitly calls for a cached wire-position provider.

Add `WirePositionProvider` for V1:

- keyed by `(apa_name, layer, side, flipped)`
- loads the latest valid position per wire once per run
- exposes predicted center true pose for each wire
- exposes nearby candidate wires for an arbitrary measurement pose
- optionally exposes wire direction for along-wire rescue scans
- exposes the expected wire angle for the active layer and the competing angles
  for adjacent layers

Sweep assignment uses this provider to associate voiced windows with one or more
nearby candidate wires. Wire coordinates remain expressed in true-position
space; streaming observations compare those wire locations against the
focus-corrected `x_laser` while retaining `x_true` as the reference coordinate.
If multiple nearby wires are plausible, the candidate is marked ambiguous and
queued for rescue.

## Layer Isolation Strategy

Layer separation is not solved by focus alone.

V1 uses two orthogonal isolation mechanisms:

- a focus-plane model that predicts which layer should be optically sharp at a
  given area
- angle-coherent pitch patterns that should trace the direction of the current
  layer's wires when sweep windows are projected into measurement-pose space

Implications for implementation:

- a candidate is stronger when voiced windows with similar pitch form a track
  aligned with the active layer angle
- a candidate is weaker when the same pitch evidence aligns better with an
  adjacent layer angle
- focus consistency and angle coherence are both first-class scores on a
  `WireCandidate`
- sweep should include enough along-wire motion to observe angle coherence, not
  only isolated points
- rescue should explicitly separate two subproblems:
  - bring the correct layer into focus
  - center the measurement on the intended wire within that layer

## Audio Analysis Pipeline

### Fast online scorer

The fast scorer is a thin wrapper around the harmonic comb response logic from
`spectrum_analysis.comb_trigger`.

It must emit, per frame:

- `comb_score`
- `spectral_flatness`
- `harmonic_valid`
- RMS or equivalent amplitude metric
- optional `expected_band_score` when a target wire frequency is known

Use policy:

- sweep mode uses comb score plus flatness gating with no strong expected-pitch
  bias
- rescue mode uses the same scorer, but weights toward the expected band of the
  target wire
- the fast scorer decides whether to open or extend a voiced window
- it does not decide final wire acceptance on its own

### Slow asynchronous pitch stage

PESTO is used only after the fast stage says a window is worth checking.

Rules:

- the audio capture thread never runs PESTO
- the motion thread never waits for PESTO
- voiced windows are pushed to a background pitch worker queue
- PESTO outputs are joined back into the spatial evidence field before any
  wire-level candidate decision is made
- backpressure is handled by dropping low-priority sweep windows before rescue
  windows

The existing `analyze_audio_with_pesto(...)` helper is sufficient for V1. The
streaming stack does not need PESTO's streaming model as a hard dependency.

## Spatial Evidence Aggregation

The streaming plan should not jump directly from one `VoicedWindow` or one
PESTO result to a `WireCandidate`. There needs to be an intermediate evidence
layer that can:

- accumulate repeated measurements at the same place
- let confidence improve naturally as compatible observations arrive
- preserve conflicting pitch hypotheses instead of forcing an early wire choice
- correlate observations from independent sweeps before final wire assignment
- drive visualization even when wire identity is still ambiguous

### Evidence binning

V1 should introduce a `PitchEvidenceField` or equivalent abstraction built from
`PitchEvidenceBin` records.

Rules:

- bin in true-position space, not raw laser-offset space
- use a default spatial bin size of `0.5 mm`
- store all contributing voiced-window or pitch-result ids
- maintain multiple pitch hypotheses per bin when the local evidence is mixed
- merge compatible hypotheses using weighted pitch averaging and monotonic
  confidence fusion
- record how each hypothesis behaves across sampled focus residuals

### Cross-sweep correlation

Independent sweeps should update the same bin set whenever they revisit the same
true position. This is the mechanism by which repeated visits improve local
confidence and by which separate sweeps can be compared as the same or
different-wire evidence.

Wire-level inference should happen after binning:

- first build local pitch hypotheses per spatial bin
- first estimate whether any of those hypotheses prefer a specific local focus
  residual
- then stitch compatible bins into line-like tracks aligned with the current
  layer angle
- then compare those tracks against predicted wire positions

### Visualization

The evidence field is also the minimum useful visualization abstraction.

V1 visualization should support:

- dominant pitch per bin
- dominant combined confidence per bin
- multiple hypotheses in one bin when evidence conflicts
- overlays of current-layer and adjacent-layer angle tracks

This field is more useful than plotting only final wire candidates because it
shows where the system has repeatable local pitch evidence even before wire
identity is resolved.

## Excitation Model

V1 stays pulse-driven.

Do not implement sweep acquisition as a blind fixed-rate "start strum and forget
it" loop. The older draft suggested changing `ValveController.start_strum()` to
fire continuously at a configurable rate. That is too coarse for V1 because the
user explicitly noted that restimulating an already vibrating wire can damage a
good signal.

Instead add a pulse scheduler in the streaming controller:

- pulses are allowed only inside segment cruise windows
- sweep pulses are spaced by a configurable refractory interval
- rescue pulses may be slower and more selective than sweep pulses
- pulse timing is recorded in session storage
- frames are later interpreted relative to the last pulse time

Future continuous-air support can be introduced by swapping the excitation
service, not by rewriting the rest of the controller.

## Sweep Workflow

`StreamingMeasurementController.run_sweep(...)` is the default streaming
workflow.

### Inputs

- APA, layer, side, flip state
- one or more comb-safe sweep corridors
- sweep speed
- focus plane
- pulse scheduler settings
- candidate acceptance thresholds

### Flow

1. Bootstrap runtime, position provider, and focus plane.
2. Run anchor rescue measurements if needed to validate or build the plane.
3. Plan comb-safe sweep segments within the requested corridor.
4. For each segment:
   - move to the segment start pose
   - command the segment at constant speed
   - capture audio continuously in the background
   - fire pulses only during the cruise window
   - compute fast frame metrics
   - build voiced windows
   - push voiced windows to the async pitch worker
5. Update spatial pitch evidence bins from PESTO-confirmed windows, including
   local `focus_response` versus `delta_focus` where repeated offset probes
   exist.
6. Aggregate compatible bins into provisional wire candidates using:
   - focus consistency against the current focus plane
   - local focus-residual preference inferred from repeated offset probes
   - angle coherence against the active layer wire direction
   - ambiguity penalties for nearby competing wires or adjacent-layer angles
   - repeat-support strength from the evidence bins
7. Mark candidates as:
   - accepted directly when evidence is strong and unambiguous
   - queued for rescue when confidence is low, pitch is mixed, focus looks bad,
     or the window maps to multiple plausible wires
8. Persist final accepted results to the main tension DB with streaming
   provenance.
9. Persist rescue queue state and all streaming artifacts to the session
   directory.

### Sweep acceptance rules

A sweep candidate may be accepted without rescue only if:

- it maps cleanly to one wire
- harmonic-comb support is strong across multiple frames or windows
- PESTO frequency is stable enough for the target wire
- repeated visits to the same spatial bins strengthen one dominant local pitch
  hypothesis rather than creating unresolved local conflicts
- the supporting windows are consistent with the current focus plane
- if local offset probes were taken, they indicate a stable preferred focus
  residual
- the supporting windows form a pattern aligned with the active layer angle
- computed tension is plausible
- no nearby wire or adjacent-layer explanation is comparably likely

Anything less should go to rescue.

## Rescue Workflow

`StreamingMeasurementController.run_rescue(...)` handles difficult wires.

### Inputs

- target wire number
- optional seed pose from sweep
- optional expected frequency derived from wire length

### Flow

1. Choose the starting pose:
   - use the best sweep pose if available
   - otherwise use `WirePositionProvider` plus current focus plane
2. Build a local scan neighborhood:
   - along-wire offsets to test angle coherence
   - cross-wire offsets if needed for centering
   - small focus offsets to isolate the correct layer
3. Iterate local segments or point moves:
   - fire one pulse
   - capture ringdown
   - score harmonicity immediately
   - update best known measurement pose
4. Run PESTO on shortlisted rescue windows.
5. Accept the best result when:
   - pitch is locked
   - tension is plausible
   - confidence exceeds rescue threshold
6. Write the final `TensionResult`.
7. Use accepted rescue results to update the focus plane.

### Rescue objective

Rescue maximizes fast harmonicity first, then uses slow pitch confirmation to
decide final acceptance. It is not a pure amplitude search and not a blind
random wiggle.

Rescue is also where the remaining joint optimization happens explicitly. Sweep
reduces the search space by carrying a focus prior and sparse offset evidence
throughout a corridor; rescue spends concentrated pulses only where the local
focus residual or wire centering is still uncertain.

## Session Storage

All streaming artifacts are written under:

```text
data/streaming_runs/<session_id>/
```

Required contents:

- `manifest.json`
- `streaming.db`
- `audio/` chunk files
- optional derived plots or reports

### `manifest.json`

Must record:

- session id
- measurement mode
- APA, layer, side, flipped
- runtime config and thresholds
- initial focus plane coefficients
- anchor measurements used for bootstrap
- sweep corridors
- code version or git commit if available

### `streaming.db`

V1 should contain these tables at minimum:

| Table | Purpose |
| --- | --- |
| `segments` | planned and executed motion segments |
| `pulse_events` | pulse timestamps, duration, segment id |
| `frames` | aligned fast metrics per analyzed hop |
| `voiced_windows` | windows sent to async pitch |
| `pitch_results` | PESTO outputs for voiced windows |
| `pitch_bins` | aggregated spatial pitch evidence bins |
| `wire_candidates` | provisional and final candidate state |
| `rescue_queue` | queued rescue work and disposition |
| `anchors` | focus-plane bootstrap points |
| `audio_chunks` | filename, time span, sample rate, segment id |

Streaming session storage is separate from the main tension DB on purpose:

- session writes can be frequent and append-heavy
- final wire results are a much smaller, curated subset
- replay tooling should be able to analyze session data without touching the
  production summary tables

## GUI and CLI Integration

### GUI

Add a mode selector to the existing measurement UI:

- `Legacy`
- `Streaming Sweep`
- `Streaming Rescue`

Add live streaming status fields:

- current segment id
- current sweep corridor
- current comb score
- pitch worker backlog
- rescue queue depth
- current focus prediction

The existing live waveform panel may keep showing a recent capture, but V1 does
not need a full scrolling spectrogram embedded in the Tk UI before streaming can
ship.

### CLI

Add these entry points:

- `dune-tension-stream-sweep`
- `dune-tension-stream-rescue`
- `dune-tension-fit-focus`
- `dune-tension-stream-replay`

The replay CLI should accept either a single `.wav` file or a directory and run
the same fast analysis pipeline used online. It should output summary metrics and
optionally a CSV report for threshold tuning.

## Offline Replay and Dataset Evaluation

The repository already contains a substantial `.wav` corpus. V1 should use it as
part of implementation, not as a later cleanup.

### Dataset assumptions

- each file can be joined to expected-frequency labels
- full pose metadata may not exist for every file
- that is still enough for fast-scorer evaluation and pitch-lock benchmarking

### Replay goals

- tune comb-score and flatness thresholds
- measure false positives on noise and mixed signals
- compare expected-frequency-aware versus model-free gating
- estimate how often PESTO receives useful windows
- identify pathological recordings to keep as regression fixtures

### Replay outputs

Produce, per run:

- per-file summary CSV or SQLite table
- confusion-style counts for gate pass or fail
- expected versus measured frequency error statistics
- optional saved plots for representative failures

The replay harness should reuse the online frame-analysis code path as much as
possible. The only difference should be the audio source and the absence of live
motion or pulse scheduling.

## Recommended Implementation Order

Implement in this order so each step leaves a usable checkpoint.

### Phase 1: Shared primitives

- extract a public harmonic-comb metric helper
- add streaming models
- add session storage helpers
- add `PitchEvidenceField` and `PitchEvidenceBin` aggregation
- add `WirePositionProvider`
- add `FocusPlaneModel`
- add replay harness on existing `.wav` files

### Phase 2: Online analysis services

- add `AudioStreamService`
- add fast frame analyzer
- add voiced-window builder
- add async pitch worker
- add local focus-response accumulation inside evidence bins
- validate thresholds offline before hardware use

### Phase 3: Sweep controller

- add comb-safe segment planner
- add pose interpolation
- add pulse scheduler
- add sparse focus-offset sweep probing with X compensation
- add evidence-bin updates from pitch observations
- add provisional wire-candidate aggregation
- add sweep session writing

### Phase 4: Rescue controller

- add local along-wire and focus scans
- add rescue queue processing
- add focus-plane refinement from rescue outcomes
- add final accepted result writeback

### Phase 5: UI and operational polish

- add GUI mode selector and status fields
- add headless CLIs
- add summary and monitoring hooks
- document operating procedure for anchor, sweep, rescue

## Testing and Verification

### Unit tests

- harmonic-comb frame scoring on:
  - clean ringdown
  - broadband noise
  - adjacent-layer interference
  - weak voiced segments near threshold
- joint-pose interpolation over constant-speed segments with masked accel/decel
  windows
- focus-coupled X derivation using the same side-dependent transform as
  `tensiometer.py`, relative to the reference in-focus position
- pitch-hypothesis merge behavior inside one `PitchEvidenceBin`
- monotonic confidence growth for repeated compatible observations in the same
  spatial bin
- local focus-response accumulation across repeated offset probes at the same
  true position
- focus-plane fit and clamping behavior
- wire-position provider on historical measurement subsets interpreted as true
  positions from correctly focused measurements
- angle-coherence discrimination between current-layer and adjacent-layer track
  patterns
- backpressure behavior when pitch worker is slower than audio ingestion
- rescue optimization on synthetic reward surfaces

### Integration tests

- replay over a locked fixture subset of real `.wav` files
- spoofed sweep session that writes a valid session directory and DB
- spoofed rescue session that promotes one candidate to a final result
- repeated sweeps over the same corridor that increase bin-level confidence
  instead of duplicating unrelated candidates
- GUI action tests confirming measurement mode dispatch and cleanup behavior
- regression coverage for legacy `Tensiometer` behavior

### Manual validation

1. Fit or warm-start a focus plane for one APA side.
2. Run a short sweep in one comb-safe corridor.
3. Inspect `frames`, `pitch_results`, and `wire_candidates`.
4. Confirm obvious weak candidates entered the rescue queue.
5. Run rescue on a small set of wires and compare final tensions against known
   measurements in `tension_data.db`.

## Acceptance Criteria

The streaming implementation is ready for normal development use when all of the
following are true:

- a headless spoofed sweep session produces a valid session directory with audio,
  frame metrics, pitch results, and candidate rows
- replay on the real `.wav` corpus produces stable threshold reports
- sweep can accept some wires directly and queue others for rescue
- rescue can recover at least one ambiguous or low-confidence wire in testing
- repeated measurements at the same place increase local pitch confidence in the
  stored evidence field
- independent sweeps can be correlated through shared evidence bins before final
  wire assignment
- focus is improved through shared plane prediction plus sparse local residual
  probing, not by repeating a full focus search at every wire
- layer separation depends on both focus consistency and wire-angle coherence,
  not on either signal alone
- accepted streaming results are written to the existing tension DB with
  `measurement_mode` and `stream_session_id`
- the GUI can launch `Legacy`, `Streaming Sweep`, and `Streaming Rescue` without
  breaking existing measurement actions

## Deferred Work

These items are intentionally deferred after V1:

- continuous-air excitation
- non-planar focus surfaces
- end-to-end automatic full-APA mapping with no rescue phase
- learned voiced classifiers trained on the `.wav` corpus
- live embedded spectrogram or pitch-map visualization beyond simple status and
  recent waveform diagnostics

## Notes For The Implementer

- Preserve the existing `Tensiometer` path. Do not refactor it away as part of
  the first streaming implementation.
- Prefer explicit runtime assembly over new import-time globals.
- Keep session-local streaming writes separate from final curated tension
  results.
- Do not rely on x-only clustering for wire identity when historical positions
  and geometry are available.
- Do not jump directly from one pitch result to one wire decision. Aggregate
  local evidence in spatial pitch bins first.
- Do not reason about bare XY in streaming code. Always carry the true
  position, focus reference, and relative focus-induced X correction together.
- Do not implement V1 around blind continuous strumming.
- Tune thresholds with the existing `.wav` corpus before trusting hardware-only
  experiments.
