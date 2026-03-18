# dune-tension Refactoring Audit

## Scope

This audit covers the current `dune_tension` runtime for measurement, persistence,
and GUI orchestration, plus the test and compatibility layers that now shape how
easy the package is to refactor.

Primary code paths reviewed:

- `src/dune_tension/main.py`
- `src/dune_tension/tensiometer.py`
- `src/dune_tension/tensiometer_functions.py`
- `src/dune_tension/services.py`
- `src/dune_tension/data_cache.py`
- `src/dune_tension/results.py`
- `src/dune_tension/gui/app.py`
- `src/dune_tension/gui/actions.py`
- `src/dune_tension/gui/context.py`
- `src/dune_tension/gui/live_plots.py`
- `src/dune_tension/summaries.py`
- `src/dune_tension/audioProcessing.py`
- `src/spectrum_analysis/audio_processing.py`
- `src/spectrum_analysis/pesto_analysis.py`
- `tests/test_tensiometer.py`
- `tests/test_tensiometer_functions.py`
- `tests/test_tensiometer_live_callbacks.py`
- `tests/test_gui_actions.py`
- `tests/test_motion_position_sources.py`
- `tests/test_data_cache.py`
- `tests/test_main_dependencies.py`

## What Changed Since the Last Audit

The codebase has moved into an intermediate state. The runtime is not fully
decoupled, but several worthwhile seams already exist and should be treated as
real architecture, not just temporary cleanup.

### Already Landed

- `src/dune_tension/main.py` is now only a thin GUI entrypoint that delegates to
  `dune_tension.gui.run_app`.
- GUI responsibilities are now split across focused modules:
  `gui/app.py`, `gui/actions.py`, `gui/context.py`, `gui/live_plots.py`, and
  `gui/state.py`.
- Measurement planning is no longer duplicated inline. The shared planner
  `tensiometer_functions.plan_measurement_triplets(...)` is now used by both
  `Tensiometer.measure_auto()` and `tensiometer_functions.measure_list(...)`.
- Motion lookup now prefers cached PLC position in two places:
  `services.MotionService.build(...)` and `gui.context._resolve_plc_functions()`.
- Persistence already distinguishes between final summary rows and per-sample
  rows via the `tension_data` and `tension_samples` SQLite tables in
  `data_cache.py`.

### Partial Seams That Exist but Are Not Finished

- `services.MotionService`, `services.AudioCaptureService`, and
  `services.ResultRepository` are real adapters, but `Tensiometer.__init__()`
  still constructs them internally instead of receiving them from a runtime
  assembly layer.
- `gui.actions.WorkerInputs`, `gui.actions._capture_worker_inputs()`, and
  `gui.actions.create_tensiometer(ctx, inputs)` are useful seams for GUI-driven
  measurement setup, but bootstrap and environment resolution are still spread
  across both `gui.actions` and `gui.context`.
- `Tensiometer(..., audio_sample_callback=..., summary_refresh_callback=...)`
  exposes meaningful observer hooks, but those callbacks do not yet make the
  measurement loop fully injectable or easy to simulate end-to-end.

## Current Public Seams Worth Preserving

These interfaces are already valuable and should be preserved or formalized
rather than accidentally re-inlined during future refactors:

- `gui.actions.WorkerInputs`
- `gui.actions.create_tensiometer(ctx, inputs)`
- `tensiometer_functions.plan_measurement_triplets(...)`
- `services.MotionService`
- `services.AudioCaptureService`
- `services.ResultRepository`
- `Tensiometer(..., audio_sample_callback=..., summary_refresh_callback=...)`

## Highest-Value Remaining Refactors

### 1. Planner extraction is done, but coordinate lookup is still the hot-path bottleneck

The measurement planner is now centralized in
`tensiometer_functions.plan_measurement_triplets(...)`, which is a real
improvement. The remaining issue is that `tensiometer_functions.get_xy_from_file()`
still reloads cached measurement data, filters by APA and layer, filters again by
side, sorts by time, drops duplicates, sorts by wire number, and only then finds
the closest wire on every call.

That repeated cleanup still happens once per requested wire from both the auto and
list flows. The expensive part is no longer the planner itself; it is the
per-wire DataFrame reconstruction under the planner.

Recommended refactor:

- Introduce a cached `WirePositionProvider` keyed by
  `(apa_name, layer, side, flipped)`.
- Precompute the latest valid per-wire positions once per run.
- Use the provider from both `Tensiometer.measure_auto()` and the shared list
  measurement path.
- Keep the geometric refinement step separate so it remains testable.

Expected impact:

- Lower repeated `pandas` churn during long runs.
- Cleaner tests for coordinate lookup using small synthetic inputs.
- A better separation between planning and historical-data normalization.

### 2. SQLite table split is done, but batching and cache invalidation are still missing

`data_cache.py` now stores summary rows and sample rows separately, and
`ResultRepository.append_result()` / `append_sample()` reflect that split. That is
real progress compared with the older single-table shape.

The remaining hot-path problem is still in `data_cache._append_row()`:

- a new SQLite connection is opened for every row
- table existence and schema are rechecked on every append
- each row is committed immediately
- the in-process DataFrame cache is copied and extended on every append

Recommended refactor:

- Give `ResultRepository` an explicit connection or transaction scope for one
  measurement run.
- Buffer sample rows and flush them with `executemany`.
- Replace full-copy cache updates with cheaper invalidation or append-aware cache
  maintenance.
- Keep read helpers separate from append-heavy write paths.

Expected impact:

- Lower latency in the sampling loop.
- Less avoidable I/O and less DataFrame copying as the DB grows.
- Repository tests that can focus on transaction boundaries instead of repeated
  singleton writes.

### 3. Adapter wrappers exist, but `Tensiometer` still owns too much runtime assembly

`Tensiometer` is no longer reaching directly into every legacy module for every
operation; it now uses `MotionService`, `AudioCaptureService`, `ResultRepository`,
and observer callbacks. That makes the class easier to reason about than before.

Even so, `Tensiometer` still creates its own config, services, repository, and
core runtime behavior in one place. It still owns:

- runtime assembly
- stop-event handling
- optimizer and retry policy
- sample persistence
- final-result persistence
- ETA updates
- measurement loop orchestration

Recommended refactor:

- Keep `Tensiometer` as the workflow coordinator.
- Move construction into an explicit runtime factory such as
  `build_tensiometer(...)` or a `RuntimeBundle`.
- Inject motion, audio capture, pitch estimation, persistence, position lookup,
  and time/random helpers explicitly.
- Reuse the same runtime assembly path from the GUI instead of resolving pieces in
  multiple layers.

Expected impact:

- Smaller tests with local fakes instead of global import surgery.
- Cleaner separation between workflow code and environment bootstrap.
- Easier profiling and simulation of the measurement loop.

### 4. `TensionResult` is still an impure data object, and a stale CSV path remains visible

`results.TensionResult.__post_init__()` still computes zone, wire length,
tension, pass/fail, and a default timestamp. That keeps geometry and tension
calculation logic entangled with what is otherwise acting like a persisted record.

There is now a second stale signal in the same area:
`Tensiometer.load_tension_summary()` still assumes `config.data_path` points to a
CSV with `A` and `B` columns, but `tensiometer_functions.make_config()` now
hardcodes the runtime data path to the SQLite database
`data/tension_data/tension_data.db`.

Recommended refactor:

- Convert `TensionResult` into a plain persisted record or split creation into a
  pure derivation function plus a data container.
- Pass timestamps explicitly from callers.
- Remove or rewrite `Tensiometer.load_tension_summary()` so it matches the actual
  SQLite-backed summary model.

Expected impact:

- Simpler fixtures and fewer hidden calculations during object construction.
- Removal of a stale CSV-era code path that no longer matches the runtime.
- Clearer boundaries between domain math and storage schema.

### 5. GUI bootstrap is cleaner, but hardware and environment resolution are still too deep in the GUI layer

The GUI now has a much cleaner structure than the earlier single-file shape. The
remaining issue is specifically where runtime dependencies are still assembled.

`gui.context.create_context(...)` still constructs the servo controller, tries to
construct the valve controller, and resolves live-vs-spoof PLC functions from
environment variables. `gui.actions.create_tensiometer(...)` still resolves
`SPOOF_AUDIO` and `SPOOF_PLC` while building the measurement runtime.

Recommended refactor:

- Move environment-variable resolution to one outer bootstrap layer.
- Build a single `RuntimeBundle` or equivalent object for motion, audio, valve,
  servo, and persistence services.
- Let `create_context(...)` and `create_tensiometer(...)` receive already-built
  collaborators instead of deciding how to construct them.

Expected impact:

- GUI code that is easier to run in headless or semi-spoofed environments.
- Cleaner separation between Tk wiring and hardware/runtime assembly.
- More direct GUI tests that do not need to simulate environment state.

## Test Suite Realignment

The test suite now shows both the progress that has been made and the remaining
architectural debt.

### Focused seam tests that should be expanded

Newer tests already target specific seams instead of monolithic import-time
behavior:

- `tests/test_tensiometer_functions.py` covers
  `plan_measurement_triplets(...)` and the shared planner output used by list
  measurement.
- `tests/test_tensiometer_live_callbacks.py` covers
  `audio_sample_callback` and `summary_refresh_callback`.
- `tests/test_gui_actions.py` covers worker-thread serialization and GUI action
  seams such as list filtering and cleanup helpers.
- `tests/test_motion_position_sources.py` covers cached PLC position selection.
- `tests/test_data_cache.py` covers DB schema migration and split-table behavior.

These tests align with the current architecture and should be the model for
future refactors.

### Older import-surgery tests that now distort the architecture

`tests/test_tensiometer.py` still replaces broad dependency sets through
`sys.modules` before import. That remains a strong signal that the production
dependency graph is still too implicit.

`tests/test_main_dependencies.py` now appears stale against the current package
shape. It imports `dune_tension.main` but still expects GUI/bootstrap behaviors
that now live elsewhere, such as `create_tensiometer()` and other GUI-oriented
helpers. This file should be replaced with package-native GUI bootstrap tests that
target `gui.app.run_app(...)`, `gui.context.create_context(...)`, and the action
layer directly.

Recommended test workstream:

- Replace stale `main.py` tests with tests against the current GUI entrypoint and
  bootstrap modules.
- Shrink `sys.modules`-heavy tensiometer tests as collaborators become injectable.
- Add focused tests for a future `WirePositionProvider`.
- Add repository tests for batched inserts and read-after-write behavior under a
  shared transaction scope.

## Legacy Path Removal

The remaining cleanup work is no longer just about code style. There are still
multiple runtime shapes coexisting in the package.

### Split audio stack

The active measurement flow now uses `spectrum_analysis.audio_processing` and
`spectrum_analysis.pesto_analysis` through lazy imports in `tensiometer.py`, but
`services.AudioCaptureService` and GUI calibration paths still depend on
`dune_tension.audioProcessing`.

That split increases maintenance cost and keeps older library-hostile behavior
alive, including process-oriented failure handling inside the older audio module.

Recommended cleanup:

- Decide on one authoritative audio stack.
- Move any remaining required functionality into a small dedicated adapter layer.
- Retire or isolate `dune_tension.audioProcessing` once the active runtime no
  longer needs it.

### Compatibility fallbacks

Several modules still support both package-qualified imports and old top-level
imports, including `tensiometer.py`, `services.py`, `results.py`,
`tensiometer_functions.py`, and `summaries.py`.

Those fallbacks were useful during migration, but they now:

- hide dependency failures
- enlarge the number of runtime shapes that tests must support
- encourage tests to patch global module state rather than use explicit
  collaborators

Recommended cleanup:

- Remove package-vs-top-level fallback imports once the remaining legacy tests are
  replaced.
- Keep one canonical import path and one runtime shape.

### Stale CSV-era helpers

Not all CSV-era assumptions are gone. The stale `Tensiometer.load_tension_summary()`
path is the clearest example because it no longer matches the SQLite-backed
runtime. Similar helpers should either be rewritten around `summaries.py` or
relocated into explicit migration utilities if they are only needed for old data
flows.

## Recommended Refactor Order

### Phase 1: extend the seams that already exist

- Introduce a cached `WirePositionProvider`.
- Add an explicit runtime bundle or `build_tensiometer(...)` factory.
- Remove or rewrite the stale CSV summary path in `Tensiometer`.

### Phase 2: fix persistence hot paths

- Add transaction-scoped repository behavior.
- Batch sample and result inserts.
- Replace full-copy cache updates with cheaper invalidation or append-aware logic.

### Phase 3: consolidate bootstrap

- Move environment resolution and hardware construction out of
  `gui.context.create_context(...)`.
- Share one runtime assembly path between GUI bootstrap and measurement runtime
  construction.

### Phase 4: modernize the tests

- Replace stale `main.py` dependency tests.
- Reduce broad `sys.modules` stubbing in tensiometer tests.
- Add focused tests for injected collaborators, repository batching, and cached
  wire-position lookup.

### Phase 5: remove legacy paths

- Collapse duplicate audio paths.
- Remove compatibility import fallbacks after test migration.
- Retire or relocate stale CSV-oriented helpers that no longer match the
  SQLite-backed runtime.

## Document Update Notes

- This revision intentionally uses file and symbol references instead of brittle
  line ranges.
- The document reflects current code structure and intermediate refactor progress;
  it does not assume that partial seams are complete.
- Automated test execution could not be verified in this environment because
  `pytest` was not installed in the available interpreters or the local `.venv`.
