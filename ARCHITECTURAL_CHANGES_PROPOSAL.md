# Architectural Change Proposals

Scope: audit of `src/` and key runtime entrypoints for unusual implementations, coupling, and performance inefficiencies.

## Implementation Status (2026-03-10)
- P0 (items 1-5): Implemented.
- P1 (items 6-10): Implemented.
- P2 (items 11-12): Implemented.

## 1. Normalize package boundaries and imports (P0)
- Status: Implemented.
- Delivered:
  - Removed runtime `sys.path` shims.
  - Added package script entry points in [`pyproject.toml`](pyproject.toml).
  - Normalized internal imports for core runtime modules.

## 2. Decompose `Tensiometer` into focused services (P0)
- Status: Implemented.
- Delivered:
  - Introduced service layer in [`src/dune_tension/services.py`](src/dune_tension/services.py):
    - `MotionService`
    - `AudioCaptureService`
    - `ResultRepository`
  - Integrated services into [`src/dune_tension/tensiometer.py`](src/dune_tension/tensiometer.py).

## 3. Fix backend strategy bugs and invalid runtime guards (P0)
- Status: Implemented.
- Delivered:
  - Fixed spoof-audio strategy override bug in [`src/dune_tension/tensiometer.py`](src/dune_tension/tensiometer.py).
  - Replaced invalid NaN assert with explicit `np.isnan` validation.

## 4. Unify the two audio stacks and enforce stable APIs (P0)
- Status: Implemented (API stabilization).
- Delivered:
  - Made `acquire_audio(..., timeout=...)` timeout optional in [`src/spectrum_analysis/audio_processing.py`](src/spectrum_analysis/audio_processing.py).
  - Updated callers to safely handle empty captures in [`src/spectrum_analysis/workflow.py`](src/spectrum_analysis/workflow.py) and [`src/spectrum_analysis/compare_pitch_cli.py`](src/spectrum_analysis/compare_pitch_cli.py).

## 5. Redesign persistence around an explicit DB repository (P0)
- Status: Implemented.
- Delivered:
  - Reworked storage in [`src/dune_tension/data_cache.py`](src/dune_tension/data_cache.py):
    - explicit `tension_data`/`tension_samples` tables
    - append-row APIs (`append_dataframe_row`, `append_results_row`)
    - corrected table consistency for read/write paths
    - fixed outlier return type consistency
  - Wired measurement persistence through `ResultRepository`.

## 6. Cache wire-length LUTs and correct zone semantics (P1)
- Status: Implemented.
- Delivered:
  - Added LUT caching via `@lru_cache` in [`src/dune_tension/geometry.py`](src/dune_tension/geometry.py).
  - Normalized zone mapping to always return zones `1..5` and clamp out-of-bounds `x`.

## 7. Replace `eval` usage with safe parsers (P1)
- Status: Implemented.
- Delivered:
  - Replaced `eval` in migration parsing with `ast.literal_eval` in [`src/dune_tension/migrate_to_db.py`](src/dune_tension/migrate_to_db.py).
  - Replaced GUI condition `eval` with constrained AST-validated expression compilation in [`src/dune_tension/gui/actions.py`](src/dune_tension/gui/actions.py).

## 8. Make GUI threading Tk-safe (P1)
- Status: Implemented.
- Delivered:
  - Added main-thread UI snapshot capture (`WorkerInputs`) in [`src/dune_tension/gui/actions.py`](src/dune_tension/gui/actions.py).
  - Updated threaded actions to operate only on captured inputs.
  - Moved validation/messagebox use to main thread before worker startup.

## 9. Consolidate M2M client into one typed module (P1)
- Status: Implemented.
- Delivered:
  - Established canonical client in [`src/dune_tension/m2m/common.py`](src/dune_tension/m2m/common.py).
  - Converted `common_v2.py` into compatibility re-export wrapper to eliminate duplicate logic.
  - Added `M2MError` and replaced hard `sys.exit` calls with exceptions.
  - Replaced fragile token string-splitting with JSON token parsing.

## 10. Fix schema/model drift between migration and result model (P1)
- Status: Implemented.
- Delivered:
  - Removed invalid `TensionResult(..., wires=..., ttf=...)` construction in [`src/dune_tension/migrate_to_db.py`](src/dune_tension/migrate_to_db.py).
  - Removed non-model fields from manual tension row creation in [`src/dune_tension/gui/actions.py`](src/dune_tension/gui/actions.py).

## 11. Reduce algorithmic inefficiencies in summarization and planning (P2)
- Status: Implemented.
- Delivered:
  - Replaced per-wire nested filtering with grouped/vectorized operations in [`src/dune_tension/summaries.py`](src/dune_tension/summaries.py).
  - Removed duplicate coordinate lookup calls in [`src/dune_tension/tensiometer_functions.py`](src/dune_tension/tensiometer_functions.py).

## 12. Move runtime/generated artifacts out of package source (P2)
- Status: Implemented.
- Delivered:
  - Moved default noise filter runtime path to `data/noise_filters/` with legacy fallback/migration in [`src/dune_tension/audioProcessing.py`](src/dune_tension/audioProcessing.py).
  - Replaced placeholder root entrypoint with package launcher in [`main.py`](main.py).

## Next Execution Order
1. No remaining proposal items in this document.
