# Architectural Change Proposals

Scope: audit of `src/` and key runtime entrypoints for unusual implementations, coupling, and performance inefficiencies.

## 1. Normalize package boundaries and imports (P0)
- Evidence:
  - `sys.path` shims in [`src/dune_tension/main.py`](src/dune_tension/main.py:8) and [`src/spectrum_analysis/compare_pitch_cli.py`](src/spectrum_analysis/compare_pitch_cli.py:13).
  - Non-package imports across runtime modules, e.g. [`src/dune_tension/tensiometer.py`](src/dune_tension/tensiometer.py:14), [`src/dune_tension/summaries.py`](src/dune_tension/summaries.py:9), [`src/dune_tension/results.py`](src/dune_tension/results.py:4), [`src/spectrum_analysis/spectrogram_scroller_basic.py`](src/spectrum_analysis/spectrogram_scroller_basic.py:4).
  - No declared script entry points in `pyproject.toml`.
- Problem:
  - Runtime behavior depends on invocation directory and import side effects.
- Proposal:
  - Convert all internal imports to fully-qualified package imports.
  - Remove `sys.path` insertion hacks.
  - Add `[project.scripts]` for GUI and CLIs.

## 2. Decompose `Tensiometer` into focused services (P0)
- Evidence:
  - [`src/dune_tension/tensiometer.py`](src/dune_tension/tensiometer.py) handles config, PLC movement, audio capture, retry/wiggle logic, result aggregation, and persistence in one class.
- Problem:
  - High coupling makes testing and changes risky; hardware and analysis concerns are intertwined.
- Proposal:
  - Split into `MotionController`, `AudioCaptureService`, `MeasurementEngine`, and `ResultRepository`.
  - Keep `Tensiometer` as thin orchestration layer.

## 3. Fix backend strategy bugs and invalid runtime guards (P0)
- Evidence:
  - Spoof audio lambda is overwritten unconditionally in [`src/dune_tension/tensiometer.py`](src/dune_tension/tensiometer.py:114) and [`src/dune_tension/tensiometer.py`](src/dune_tension/tensiometer.py:126).
  - Invalid NaN assert in [`src/dune_tension/tensiometer.py`](src/dune_tension/tensiometer.py:419) (`x != NaN` is always true).
- Problem:
  - Test/spoof mode does not behave as intended, and invalid values can pass silently.
- Proposal:
  - Replace lambda swapping with explicit strategy objects.
  - Use `np.isnan(length)` checks and typed validation.

## 4. Unify the two audio stacks and enforce stable APIs (P0)
- Evidence:
  - Two parallel audio modules: [`src/dune_tension/audioProcessing.py`](src/dune_tension/audioProcessing.py) and [`src/spectrum_analysis/audio_processing.py`](src/spectrum_analysis/audio_processing.py).
  - API drift bug: [`src/spectrum_analysis/workflow.py`](src/spectrum_analysis/workflow.py:64) calls `acquire_audio` without `timeout`, but [`src/spectrum_analysis/audio_processing.py`](src/spectrum_analysis/audio_processing.py:432) requires it.
- Problem:
  - Duplicate implementations diverge and break callsites.
- Proposal:
  - Select one canonical audio package and migrate callers.
  - Introduce versioned interfaces (or compatibility wrappers) during migration.

## 5. Redesign persistence around an explicit DB repository (P0)
- Evidence:
  - Global mutable cache in [`src/dune_tension/data_cache.py`](src/dune_tension/data_cache.py:11).
  - Full-table rewrite on update: [`src/dune_tension/data_cache.py`](src/dune_tension/data_cache.py:40) and [`src/dune_tension/data_cache.py`](src/dune_tension/data_cache.py:63).
  - Table mismatch: read from `tension_data` in [`src/dune_tension/data_cache.py`](src/dune_tension/data_cache.py:50) but write `tension_samples` in [`src/dune_tension/data_cache.py`](src/dune_tension/data_cache.py:63).
  - Return type mismatch (`list[int]` annotated, `set` returned) in [`src/dune_tension/data_cache.py`](src/dune_tension/data_cache.py:145).
- Problem:
  - Inefficient writes, schema drift risk, and inconsistent semantics.
- Proposal:
  - Create repository methods with explicit SQL for append/upsert/delete.
  - Separate `summary` and `samples` schemas with migrations.
  - Add thread/process-safe cache invalidation or remove global cache.

## 6. Cache wire-length LUTs and correct zone semantics (P1)
- Evidence:
  - `zone_lookup` can return `0` in [`src/dune_tension/geometry.py`](src/dune_tension/geometry.py:32), while `length_lookup` expects zones `1..5` at [`src/dune_tension/geometry.py`](src/dune_tension/geometry.py:108).
  - `length_lookup` reads CSV on every call in [`src/dune_tension/geometry.py`](src/dune_tension/geometry.py:102).
- Problem:
  - Repeated disk I/O in tight loops and boundary inconsistencies.
- Proposal:
  - Preload LUTs once (module cache or repository singleton).
  - Normalize zone mapping with explicit boundary tests.

## 7. Replace `eval` usage with safe parsers (P1)
- Evidence:
  - User expression eval in [`src/dune_tension/gui/actions.py`](src/dune_tension/gui/actions.py:205).
  - Data parsing eval in [`src/dune_tension/migrate_to_db.py`](src/dune_tension/migrate_to_db.py:28).
- Problem:
  - Security and correctness risk.
- Proposal:
  - Use `ast.literal_eval` for serialized list parsing.
  - For conditions, implement a constrained expression evaluator or predicate DSL.

## 8. Make GUI threading Tk-safe (P1)
- Evidence:
  - Worker thread launcher in [`src/dune_tension/gui/actions.py`](src/dune_tension/gui/actions.py:73).
  - Tk widget reads and `messagebox` calls inside worker path via [`src/dune_tension/gui/actions.py`](src/dune_tension/gui/actions.py:27).
- Problem:
  - Tkinter is not thread-safe; this can cause nondeterministic UI failures.
- Proposal:
  - Capture all widget inputs on main thread before spawning worker.
  - Route UI updates/errors via `root.after`/queue.

## 9. Consolidate M2M client into one typed module (P1)
- Evidence:
  - Duplicate client implementations: `common.py` (464 lines) and `common_v2.py` (543 lines).
  - Fragile token parsing by string split in [`src/dune_tension/m2m/common.py`](src/dune_tension/m2m/common.py:55) and [`src/dune_tension/m2m/common_v2.py`](src/dune_tension/m2m/common_v2.py:51).
  - Library-level `sys.exit` throughout both files (e.g. [`src/dune_tension/m2m/common.py`](src/dune_tension/m2m/common.py:82), [`src/dune_tension/m2m/common_v2.py`](src/dune_tension/m2m/common_v2.py:77)).
- Problem:
  - Hard to reason about behavior; callers cannot recover from errors cleanly.
- Proposal:
  - Merge into one client using response JSON decoding and exception-based errors.
  - Add typed request/response models and integration tests.

## 10. Fix schema/model drift between migration and result model (P1)
- Evidence:
  - Migration constructs `TensionResult(..., wires=..., ttf=...)` in [`src/dune_tension/migrate_to_db.py`](src/dune_tension/migrate_to_db.py:85), but `TensionResult` has no such fields in [`src/dune_tension/results.py`](src/dune_tension/results.py:9).
  - Manual tension updates inject non-model keys (`wires`, `ttf`, `t_sigma`) in [`src/dune_tension/gui/actions.py`](src/dune_tension/gui/actions.py:315).
- Problem:
  - Runtime failures and inconsistent persisted schema.
- Proposal:
  - Introduce separate dataclasses/tables for `MeasurementSample` vs `WireSummary`.
  - Validate writes against schema before persistence.

## 11. Reduce algorithmic inefficiencies in summarization and planning (P2)
- Evidence:
  - Nested per-wire DataFrame filtering in [`src/dune_tension/summaries.py`](src/dune_tension/summaries.py:42).
  - Duplicate coordinate lookup calls in list comprehension in [`src/dune_tension/tensiometer_functions.py`](src/dune_tension/tensiometer_functions.py:196).
- Problem:
  - Scales poorly with large APAs and repeated refreshes.
- Proposal:
  - Use grouped operations (`groupby`, `idxmax`) to compute latest per-wire measurements in one pass.
  - Compute `get_xy_from_file` once per wire.

## 12. Move runtime/generated artifacts out of package source (P2)
- Evidence:
  - Runtime data assets in package tree: `src/dune_tension/noise_filter.npz`, `src/dune_tension/noise_profile.npy`, `src/dune_tension/tensiometer_state.json`.
  - Root [`main.py`](main.py:1) is a placeholder unrelated to runtime package entrypoints.
- Problem:
  - Source tree mixes code with mutable runtime state and legacy entrypoints.
- Proposal:
  - Relocate runtime artifacts to `data/` or user cache dir.
  - Remove/replace placeholder root entrypoint with package script command.

## Suggested execution order
1. P0 items (1-5) to stabilize runtime boundaries and data flow.
2. P1 items (6-10) to remove high-risk behaviors and schema drift.
3. P2 items (11-12) for performance and maintainability cleanup.
