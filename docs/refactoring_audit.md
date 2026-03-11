# dune-tension Refactoring Audit

## Scope

This audit focuses on the current `dune_tension` runtime path for measurement, persistence, and GUI orchestration, with secondary attention to legacy modules that still affect maintainability and tests.

Primary files reviewed:

- `src/dune_tension/tensiometer.py`
- `src/dune_tension/tensiometer_functions.py`
- `src/dune_tension/data_cache.py`
- `src/dune_tension/services.py`
- `src/dune_tension/results.py`
- `src/dune_tension/gui/context.py`
- `src/dune_tension/audioProcessing.py`
- `src/spectrum_analysis/audio_processing.py`
- `src/spectrum_analysis/pesto_analysis.py`
- `tests/test_tensiometer.py`
- `tests/test_main_dependencies.py`

## Summary

The package has already moved in a better direction in a few places. `src/dune_tension/main.py` is now a thin entry point, the GUI is split into focused modules, and the newer `spectrum_analysis` code is noticeably easier to reason about than older `dune_tension.audioProcessing`.

The main remaining problems are:

1. The measurement path still mixes orchestration, hardware access, persistence, and retry policy in one class.
2. The hottest loops still do repeated DataFrame work and per-row SQLite writes.
3. Tests often have to stub entire modules through `sys.modules`, which is a strong signal that dependency boundaries are still too implicit.
4. Legacy fallback import patterns keep production code tolerant of many shapes, but they make the codebase harder to simplify and verify.

## Highest-Value Refactors

### 1. Cache coordinate lookups per run instead of rebuilding them per wire

`get_xy_from_file()` reloads cached measurement data, filters by APA and layer, filters again by side, sorts by time, drops duplicates, sorts by wire number, and only then finds the closest wire on every call (`src/dune_tension/tensiometer_functions.py:94-148`).

That work is repeated from:

- `Tensiometer.measure_auto()` for every missing wire (`src/dune_tension/tensiometer.py:240-284`)
- `measure_list()` for every requested wire (`src/dune_tension/tensiometer_functions.py:188-240`)

Why this matters:

- For long measurement runs, this turns coordinate lookup into repeated `pandas` reshaping work.
- The algorithm is effectively `O(number_of_requests * dataframe_cleanup)` when the cleanup result is mostly invariant for a given APA/layer/side.
- It also makes coordinate logic harder to test in isolation because the lookup logic is coupled directly to the global database accessor.

Recommended refactor:

- Introduce a `WirePositionProvider` that precomputes the latest per-wire position series once per `(apa_name, layer, side, flipped)` context.
- Build the provider at the start of `measure_auto()` and `measure_list()`.
- Expose a pure `get_position_for_wire(wire_number)` method that works on already-normalized arrays.
- Keep `refine_position()` separate so it can be tested independently.

Expected impact:

- Faster automatic runs.
- Less repeated DataFrame churn.
- Straightforward unit tests using in-memory data instead of stubbing `data_cache`.

### 2. Replace per-row SQLite writes and DataFrame cache copies with batched persistence

Every sample and final result is appended individually through `ResultRepository` (`src/dune_tension/services.py:136-148`), which ends up in `_append_row()` (`src/dune_tension/data_cache.py:100-118`).

Current behavior per append:

- Open a new SQLite connection.
- Ensure table schemas exist.
- Execute one insert.
- Commit immediately.
- Copy the full cached DataFrame and append one row in memory.

Why this matters:

- Sample collection is in the hot path (`src/dune_tension/tensiometer.py:347-394`).
- Opening and committing SQLite transactions per row adds avoidable I/O overhead.
- Copying the whole cached DataFrame on every append becomes increasingly expensive as the database grows.

Recommended refactor:

- Add a write-through repository with an explicit connection lifetime or transaction scope per measurement run.
- Buffer raw samples per wire and flush them with `executemany`.
- Either remove the in-process DataFrame cache for append-heavy paths or switch it to append-only invalidation instead of full-copy updates.
- Separate read models from write models: SQLite for writes, query helpers for reads.

Expected impact:

- Lower measurement latency during sampling.
- Better scaling as the DB grows.
- Cleaner repository tests around transaction boundaries and error handling.

### 3. Split `Tensiometer` into orchestration plus injectable adapters

`Tensiometer` currently constructs its own config, motion service, audio service, repository, and uses global helper functions for audio acquisition and pitch estimation (`src/dune_tension/tensiometer.py:84-135`, `src/dune_tension/tensiometer.py:66-81`).

It also owns:

- hardware movement
- stop-event handling
- sampling retry policy
- sample/result persistence
- ETA reporting
- wire selection flow

Why this matters:

- The class has too many reasons to change.
- Unit tests currently need broad monkeypatching because collaborators are implicit globals.
- It is difficult to benchmark or simulate the measurement loop without patching module-level functions.

Recommended refactor:

- Keep `Tensiometer` as the workflow coordinator only.
- Inject these collaborators explicitly:
  - `motion`
  - `audio_capture`
  - `pitch_estimator`
  - `result_repository`
  - `position_provider`
  - `clock` / `now`
  - `random_source` or `wiggle_policy`
- Move object construction into a factory such as `build_tensiometer(...)` used by the GUI.

Expected impact:

- Smaller tests with fake objects instead of `sys.modules` hacks.
- Easier profiling of just the measurement loop.
- Better control over timing-sensitive code in tests.

### 4. Make `TensionResult` a data object, not a calculation container

`TensionResult.__post_init__()` computes zone, wire length, tension, pass/fail, and default timestamp (`src/dune_tension/results.py:12-43`).

Why this matters:

- Simple object creation pulls in geometry and tension logic.
- Derived values depend on runtime lookups, which makes the model impure.
- Tests that want to build a result object for one purpose end up implicitly testing geometry and physics rules too.

Recommended refactor:

- Convert `TensionResult` into a plain persisted record.
- Create a separate pure function or builder, for example `derive_tension_result(...)`, to compute `zone`, `wire_length`, `tension`, and `tension_pass`.
- Pass timestamps explicitly from the caller instead of defaulting with `datetime.now()`.

Expected impact:

- Better separation between domain calculations and storage schema.
- Simpler fixtures in unit tests.
- Fewer surprises when constructing objects for UI or repository code.

## Testability Findings

### 5. Tests still need module-level stubbing to import core code

`tests/test_tensiometer.py` replaces `numpy`, `pandas`, `geometry`, `tension_calculation`, `audioProcessing`, `plc_io`, `data_cache`, `results`, and `tensiometer_functions` via `sys.modules` before import (`tests/test_tensiometer.py:1-170`).

`tests/test_main_dependencies.py` does similar broad import-time stubbing for `tkinter`, `serial`, and old module names (`tests/test_main_dependencies.py:1-136`).

This is the clearest evidence that production dependencies are not explicit enough.

Recommended refactor:

- Remove fallback imports that support alternate top-level module names once migration is complete.
- Depend on explicit interfaces passed at construction time.
- Keep import-time side effects minimal so modules can be imported without fake hardware modules installed.

Target outcome:

- Tests use local fakes and temporary SQLite files instead of global import surgery.
- CI failures become easier to interpret because they fail at object boundaries, not import order.

### 6. GUI context still constructs hardware at creation time

`create_context()` creates the servo controller, valve controller, and PLC functions directly (`src/dune_tension/gui/context.py:142-174`). The helper functions also read environment variables to pick live versus spoofed hardware (`src/dune_tension/gui/context.py:92-139`).

Why this matters:

- GUI tests must simulate environment state instead of simply passing collaborators.
- Importing the GUI stack is still coupled to hardware-oriented modules such as `valve_trigger`.
- The GUI is harder to run in alternate environments like headless integration tests.

Recommended refactor:

- Introduce a `HardwareBundle` or `RuntimeServices` object built in one place.
- Let `create_context()` accept those services as optional parameters.
- Keep environment-variable resolution in a top-level bootstrap layer only.

## Lower-Priority Cleanup

### 7. Legacy `audioProcessing.py` should be quarantined or retired

The current measurement path uses `spectrum_analysis.audio_processing.acquire_audio()` and `spectrum_analysis.pesto_analysis.estimate_pitch_from_audio()` through lazy imports in `tensiometer.py` (`src/dune_tension/tensiometer.py:66-81`).

However, `dune_tension.audioProcessing.py` still contains overlapping functionality and some library-hostile behavior such as `exit(1)` in `get_samplerate()` (`src/dune_tension/audioProcessing.py:488-510`).

Why this matters:

- Duplicate runtime paths increase maintenance cost.
- Process exit inside a library helper makes testing and embedding harder.
- `services.AudioCaptureService` still depends on this older module (`src/dune_tension/services.py:36-44`, `src/dune_tension/services.py:100-133`).

Recommended refactor:

- Decide whether `dune_tension.audioProcessing` is still authoritative.
- If not, move remaining required functionality into `spectrum_analysis` or a small dedicated audio adapter module.
- Replace `exit(1)` with an exception.

### 8. Compatibility fallbacks should be removed after migration

Several modules still support both package-qualified imports and old top-level imports:

- `src/dune_tension/tensiometer.py:11-43`
- `src/dune_tension/services.py:8-20`
- `src/dune_tension/results.py:4-9`
- `src/dune_tension/tensiometer_functions.py:8-11`

Why this matters:

- It hides dependency failures.
- It increases the number of code paths tests must account for.
- It encourages tests to rely on pre-import module patching instead of stable interfaces.

Recommended refactor:

- Remove compatibility imports once the package layout is final.
- Keep one import path and one runtime shape.

## Positive Patterns Worth Reusing

Two modules already show the direction the rest of the package should follow:

- `src/spectrum_analysis/pesto_analysis.py:29-143` keeps runtime dependency loading, model caching, and pitch estimation reasonably isolated.
- `src/spectrum_analysis/audio_processing.py:379-471` cleanly separates file input, RMS-trigger capture, harmonic-comb fallback, and noise handling.

These modules are not perfect, but they are closer to explicit, composable units than the older `dune_tension` runtime path.

## Suggested Refactor Order

### Phase 1: Improve seams without changing behavior

- Add constructor injection to `Tensiometer`.
- Add a position-provider abstraction.
- Move timestamp and tension derivation out of `TensionResult`.
- Convert GUI bootstrap to assemble dependencies once.

### Phase 2: Fix hot-path performance

- Batch SQLite inserts.
- Stop copying cached DataFrames on every append.
- Precompute normalized latest-wire positions per run.

### Phase 3: Simplify package shape

- Remove legacy import fallbacks.
- Retire or isolate `dune_tension.audioProcessing`.
- Keep environment-variable logic at the outer bootstrap layer only.

### Phase 4: Expand targeted tests

- Add unit tests for the measurement retry loop using fake motion, audio, repository, and clock objects.
- Add repository tests for batched inserts and read-after-write behavior.
- Add position-provider tests from small synthetic DataFrames.
- Add a benchmark for `measure_auto()` setup time and append throughput.

## Practical First Step

If only one refactor is started now, make it this:

1. Introduce explicit dependency injection for `Tensiometer`.
2. Build a cached `WirePositionProvider`.
3. Add a batched SQLite repository.

That combination addresses the largest current pain points in both performance and testability without requiring a full rewrite.
