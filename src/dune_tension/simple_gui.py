"""Simplified GUI for collecting DUNE tension measurements."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import datetime as _dt
import json
from pathlib import Path
import re
import sqlite3
from typing import Sequence
import tkinter as tk
from tkinter import ttk

DEFAULT_LAYER_OPTIONS: tuple[str, ...] = ("X", "V", "U", "G")
DEFAULT_SIDE_OPTIONS: tuple[str, ...] = ("A", "B")


@dataclass(slots=True)
class MeasurementContext:
    """Persistent configuration for the measurement hardware."""

    apa_name: str
    layer: str
    side: str
    flipped: bool


@dataclass(slots=True)
class TensionSample:
    """Single tension sample collected from the tensiometer."""

    x: float
    y: float
    frequency: float


@dataclass(slots=True)
class TensionResult:
    """Aggregated result for a wire measurement."""

    tension: float
    confidence: float
    timestamp: _dt.datetime | None = None


@dataclass(slots=True)
class MeasurementOutcome:
    """Summary of a wire measurement operation."""

    wire_number: int
    coordinates: tuple[float, float] | None
    samples: list[TensionSample]
    result: TensionResult | None


@dataclass(slots=True)
class PersistentState:
    """User selections that should persist across sessions."""

    apa_name: str = ""
    layer: str = DEFAULT_LAYER_OPTIONS[0]
    side: str = DEFAULT_SIDE_OPTIONS[0]
    flipped: bool = False


class PersistentStateStore:
    """JSON-backed store for :class:`PersistentState`."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def load(self) -> PersistentState:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return PersistentState()
        except json.JSONDecodeError:
            return PersistentState()

        return PersistentState(
            apa_name=str(data.get("apa_name", "")),
            layer=str(data.get("layer", DEFAULT_LAYER_OPTIONS[0])),
            side=str(data.get("side", DEFAULT_SIDE_OPTIONS[0])),
            flipped=bool(data.get("flipped", False)),
        )

    def save(self, state: PersistentState) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(asdict(state), indent=2), encoding="utf-8")


class DatabaseManager:
    """SQLite helper around the measurement tables."""

    def __init__(self, db_path: Path) -> None:
        self._path = db_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self._path)
        self._initialise()

    def close(self) -> None:
        self._connection.close()

    def _initialise(self) -> None:
        cursor = self._connection.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS tension_results (
                apa_name TEXT NOT NULL,
                layer TEXT NOT NULL,
                side TEXT NOT NULL,
                wire_number INTEGER NOT NULL,
                tension REAL NOT NULL,
                confidence REAL NOT NULL,
                time_measured TEXT NOT NULL,
                PRIMARY KEY (apa_name, layer, side, wire_number)
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS tension_samples (
                apa_name TEXT NOT NULL,
                layer TEXT NOT NULL,
                side TEXT NOT NULL,
                x REAL NOT NULL,
                y REAL NOT NULL,
                frequency REAL NOT NULL,
                flipped INTEGER NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS apa_calibrations (
                apa_name TEXT NOT NULL,
                layer TEXT NOT NULL,
                side TEXT NOT NULL,
                wire_number INTEGER NOT NULL,
                x REAL NOT NULL,
                y REAL NOT NULL,
                PRIMARY KEY (apa_name, layer, side, wire_number)
            )
            """
        )
        self._connection.commit()

    def insert_samples(
        self, context: MeasurementContext, samples: Sequence[TensionSample]
    ) -> None:
        if not samples:
            return
        cursor = self._connection.cursor()
        cursor.executemany(
            """
            INSERT INTO tension_samples (
                apa_name, layer, side, x, y, frequency, flipped
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    context.apa_name,
                    context.layer,
                    context.side,
                    sample.x,
                    sample.y,
                    sample.frequency,
                    int(context.flipped),
                )
                for sample in samples
            ],
        )
        self._connection.commit()

    def insert_result(
        self,
        context: MeasurementContext,
        wire_number: int,
        result: TensionResult,
    ) -> None:
        timestamp = result.timestamp or _dt.datetime.now(tz=_dt.timezone.utc)
        cursor = self._connection.cursor()
        cursor.execute(
            """
            INSERT INTO tension_results (
                apa_name, layer, side, wire_number, tension, confidence, time_measured
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (apa_name, layer, side, wire_number)
            DO UPDATE SET
                tension=excluded.tension,
                confidence=excluded.confidence,
                time_measured=excluded.time_measured
            """,
            (
                context.apa_name,
                context.layer,
                context.side,
                wire_number,
                result.tension,
                result.confidence,
                timestamp.isoformat(),
            ),
        )
        self._connection.commit()

    def upsert_calibration(
        self, context: MeasurementContext, wire_number: int, x: float, y: float
    ) -> None:
        cursor = self._connection.cursor()
        cursor.execute(
            """
            INSERT INTO apa_calibrations (
                apa_name, layer, side, wire_number, x, y
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (apa_name, layer, side, wire_number)
            DO UPDATE SET
                x=excluded.x,
                y=excluded.y
            """,
            (context.apa_name, context.layer, context.side, wire_number, x, y),
        )
        self._connection.commit()

    def measured_wires(self, context: MeasurementContext) -> list[int]:
        cursor = self._connection.cursor()
        cursor.execute(
            """
            SELECT wire_number FROM tension_results
            WHERE apa_name = ? AND layer = ? AND side = ?
            ORDER BY wire_number
            """,
            (context.apa_name, context.layer, context.side),
        )
        return [row[0] for row in cursor.fetchall()]


class TensionApp:
    """Tkinter application that orchestrates tension measurements."""

    def __init__(
        self,
        root: tk.Misc | None = None,
        *,
        db_path: Path | None = None,
        state_file: Path | None = None,
    ) -> None:
        self.root = root or tk.Tk()
        self.root.title("DUNE Tension")

        repo_root = Path(__file__).resolve().parents[2]
        db_path = db_path or repo_root / "data" / "tension_measurements.db"
        state_file = state_file or repo_root / "simple_gui_state.json"

        self._database = DatabaseManager(db_path)
        self._state_store = PersistentStateStore(state_file)

        self.apa_var = tk.StringVar(master=self.root)
        self.layer_var = tk.StringVar(master=self.root, value=DEFAULT_LAYER_OPTIONS[0])
        self.side_var = tk.StringVar(master=self.root, value=DEFAULT_SIDE_OPTIONS[0])
        self.flipped_var = tk.BooleanVar(master=self.root, value=False)
        self.wire_number_var = tk.StringVar(master=self.root)
        self.wire_list_var = tk.StringVar(master=self.root)

        self._log_widget: tk.Text | None = None

        self._build_widgets()
        self._load_state()

    def run(self) -> None:
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.mainloop()

    def _build_widgets(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        main = ttk.Frame(self.root, padding=10)
        main.grid(row=0, column=0, sticky="nsew")
        main.columnconfigure(0, weight=1)

        apa_frame = ttk.LabelFrame(main, text="Configuration")
        apa_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        for i in range(2):
            apa_frame.columnconfigure(i, weight=1)

        ttk.Label(apa_frame, text="APA Name").grid(row=0, column=0, sticky="w")
        ttk.Entry(apa_frame, textvariable=self.apa_var).grid(
            row=0, column=1, sticky="ew"
        )

        ttk.Label(apa_frame, text="Layer").grid(row=1, column=0, sticky="w")
        layer_menu = ttk.OptionMenu(
            apa_frame,
            self.layer_var,
            self.layer_var.get(),
            *DEFAULT_LAYER_OPTIONS,
        )
        layer_menu.grid(row=1, column=1, sticky="ew")

        ttk.Label(apa_frame, text="Side").grid(row=2, column=0, sticky="w")
        side_menu = ttk.OptionMenu(
            apa_frame,
            self.side_var,
            self.side_var.get(),
            *DEFAULT_SIDE_OPTIONS,
        )
        side_menu.grid(row=2, column=1, sticky="ew")

        ttk.Checkbutton(apa_frame, text="Flipped", variable=self.flipped_var).grid(
            row=3, column=1, sticky="w"
        )

        measurement_frame = ttk.LabelFrame(main, text="Measurement")
        measurement_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        for i in range(3):
            measurement_frame.columnconfigure(i, weight=1)

        ttk.Label(measurement_frame, text="Wire Number").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Entry(measurement_frame, textvariable=self.wire_number_var).grid(
            row=0, column=1, sticky="ew"
        )
        ttk.Button(
            measurement_frame,
            text="Calibrate",
            command=self.calibrate_current_wire,
        ).grid(row=0, column=2, padx=(5, 0))

        ttk.Label(measurement_frame, text="Wire List").grid(row=1, column=0, sticky="w")
        ttk.Entry(measurement_frame, textvariable=self.wire_list_var).grid(
            row=1, column=1, sticky="ew"
        )
        ttk.Button(
            measurement_frame,
            text="Measure List",
            command=self.measure_wire_list,
        ).grid(row=1, column=2, padx=(5, 0))

        ttk.Button(
            measurement_frame,
            text="Measure Remaining",
            command=self.measure_remaining_wires,
        ).grid(row=2, column=2, padx=(5, 0), pady=(5, 0))

        log_frame = ttk.LabelFrame(main, text="Activity Log")
        log_frame.grid(row=2, column=0, sticky="nsew")
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self._log_widget = tk.Text(log_frame, height=12, state="disabled")
        self._log_widget.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(
            log_frame, orient="vertical", command=self._log_widget.yview
        )
        scrollbar.grid(row=0, column=1, sticky="ns")
        self._log_widget.configure(yscrollcommand=scrollbar.set)

    def _load_state(self) -> None:
        state = self._state_store.load()
        self.apa_var.set(state.apa_name)
        self.layer_var.set(state.layer)
        self.side_var.set(state.side)
        self.flipped_var.set(state.flipped)

    def _persist_state(self) -> None:
        state = PersistentState(
            apa_name=self.apa_var.get().strip(),
            layer=self.layer_var.get(),
            side=self.side_var.get(),
            flipped=bool(self.flipped_var.get()),
        )
        self._state_store.save(state)

    def _current_context(self) -> MeasurementContext:
        return MeasurementContext(
            apa_name=self.apa_var.get().strip(),
            layer=self.layer_var.get(),
            side=self.side_var.get(),
            flipped=bool(self.flipped_var.get()),
        )

    def _validate_context(self, context: MeasurementContext) -> bool:
        if not context.apa_name:
            self._log("Please provide an APA name before measuring.")
            return False
        return True

    def _parse_wire_list(self, raw: str) -> list[int]:
        tokens = re.split(r"[\s,]+", raw.strip())
        wires: list[int] = []
        for token in tokens:
            if not token:
                continue
            try:
                wires.append(int(token))
            except ValueError:
                self._log(f"Ignoring invalid wire identifier: {token}")
        return wires

    def _log(self, message: str) -> None:
        if self._log_widget is None:
            print(message)
            return
        timestamp = _dt.datetime.now().strftime("%H:%M:%S")
        self._log_widget.configure(state="normal")
        self._log_widget.insert("end", f"[{timestamp}] {message}\n")
        self._log_widget.configure(state="disabled")
        self._log_widget.see("end")

    def calibrate_current_wire(self) -> None:
        context = self._current_context()
        if not self._validate_context(context):
            return

        wire_text = self.wire_number_var.get().strip()
        if not wire_text:
            self._log("Enter a wire number to calibrate.")
            return
        try:
            wire_number = int(wire_text)
        except ValueError:
            self._log(f"Invalid wire number: {wire_text}")
            return

        outcome = self._perform_measurement(wire_number, context)
        if outcome is None:
            return

        position: tuple[float, float] | None
        try:
            position = get_current_position(context=context)
        except NotImplementedError:
            self._log(
                "Current position retrieval is not implemented; using target coordinates."
            )
            position = outcome.coordinates
        except Exception as exc:  # pragma: no cover - integration specific failures
            self._log(f"Failed to read current position: {exc}")
            position = outcome.coordinates

        if position is None:
            self._log("No position available to store calibration data.")
            return

        self._database.upsert_calibration(
            context, wire_number, position[0], position[1]
        )
        self._log(
            f"Stored calibration for wire {wire_number} at ({position[0]:.3f}, {position[1]:.3f})."
        )

    def measure_wire_list(self) -> None:
        context = self._current_context()
        if not self._validate_context(context):
            return

        wires = self._parse_wire_list(self.wire_list_var.get())
        if not wires:
            self._log("Provide at least one wire number in the list.")
            return

        for wire in wires:
            self._perform_measurement(wire, context)

    def measure_remaining_wires(self) -> None:
        context = self._current_context()
        if not self._validate_context(context):
            return

        try:
            wires = identify_unmeasured_wires(context=context, db=self._database)
        except NotImplementedError:
            self._log(
                "Determining unmeasured wires is not implemented; override identify_unmeasured_wires()."
            )
            return
        except Exception as exc:  # pragma: no cover - integration specific failures
            self._log(f"Failed to determine remaining wires: {exc}")
            return

        if not wires:
            self._log("All wires appear to have measurements.")
            return

        self._log(
            "Measuring remaining wires: " + ", ".join(str(wire) for wire in wires)
        )
        for wire in wires:
            self._perform_measurement(wire, context)

    def _perform_measurement(
        self, wire_number: int, context: MeasurementContext
    ) -> MeasurementOutcome | None:
        try:
            coordinates = determine_coordinates_for_wire(wire_number, context=context)
        except NotImplementedError:
            self._log(
                "Coordinate lookup is not implemented; override determine_coordinates_for_wire()."
            )
            return None
        except Exception as exc:  # pragma: no cover - integration specific failures
            self._log(f"Failed to determine coordinates for wire {wire_number}: {exc}")
            return None

        if coordinates is None:
            self._log(
                f"No coordinates returned for wire {wire_number}; skipping measurement."
            )
            return None

        try:
            move_to_coordinates(coordinates[0], coordinates[1], context=context)
        except NotImplementedError:
            self._log(
                "Motion control is not implemented; override move_to_coordinates() to enable measurements."
            )
            return None
        except Exception as exc:  # pragma: no cover - integration specific failures
            self._log(f"Failed to move to coordinates {coordinates}: {exc}")
            return None

        try:
            samples = collect_tension_samples(wire_number, context=context)
        except NotImplementedError:
            self._log(
                "Sample collection is not implemented; override collect_tension_samples()."
            )
            return None
        except Exception as exc:  # pragma: no cover - integration specific failures
            self._log(f"Sample collection failed for wire {wire_number}: {exc}")
            return None

        self._database.insert_samples(context, samples)
        self._log(f"Recorded {len(samples)} sample(s) for wire {wire_number}.")

        result: TensionResult | None
        try:
            result = evaluate_tension(samples, context=context)
        except NotImplementedError:
            self._log(
                "Tension evaluation is not implemented; override evaluate_tension() to record results."
            )
            result = None
        except Exception as exc:  # pragma: no cover - integration specific failures
            self._log(f"Failed to evaluate tension for wire {wire_number}: {exc}")
            result = None

        if result is not None:
            self._database.insert_result(context, wire_number, result)
            self._log(
                f"Stored tension {result.tension:.3f} with confidence {result.confidence:.3f} for wire {wire_number}."
            )
        else:
            self._log(f"No tension result stored for wire {wire_number}.")

        return MeasurementOutcome(
            wire_number=wire_number,
            coordinates=coordinates,
            samples=list(samples),
            result=result,
        )

    def _on_close(self) -> None:
        self._persist_state()
        self._database.close()
        self.root.destroy()


def run_app(
    *,
    db_path: Path | None = None,
    state_file: Path | None = None,
    root: tk.Misc | None = None,
) -> None:
    """Entry point used by ``python -m dune_tension``."""

    app = TensionApp(root=root, db_path=db_path, state_file=state_file)
    app.run()


def collect_tension_samples(
    wire_number: int, *, context: MeasurementContext
) -> Sequence[TensionSample]:  # pragma: no cover - stub function
    """Collect tensiometer samples for ``wire_number``.

    Override this stub to integrate with the tensiometer hardware.
    """

    raise NotImplementedError


def determine_coordinates_for_wire(
    wire_number: int, *, context: MeasurementContext
) -> tuple[float, float]:  # pragma: no cover - stub function
    """Return the stage coordinates for ``wire_number``.

    Override this stub to provide geometry lookup based on the APA configuration.
    """

    raise NotImplementedError


def move_to_coordinates(
    x: float, y: float, *, context: MeasurementContext
) -> None:  # pragma: no cover - stub function
    """Move the hardware to the provided ``(x, y)`` coordinates."""

    raise NotImplementedError


def evaluate_tension(
    samples: Sequence[TensionSample], *, context: MeasurementContext
) -> TensionResult:  # pragma: no cover - stub function
    """Return an aggregated tension result from ``samples``."""

    raise NotImplementedError


def identify_unmeasured_wires(
    *, context: MeasurementContext, db: DatabaseManager
) -> list[int]:  # pragma: no cover - stub function
    """Determine which wire numbers still need measurements."""

    raise NotImplementedError


def get_current_position(
    *, context: MeasurementContext
) -> tuple[float, float]:  # pragma: no cover - stub function
    """Return the current ``(x, y)`` stage position."""

    raise NotImplementedError


__all__ = [
    "DatabaseManager",
    "MeasurementContext",
    "MeasurementOutcome",
    "PersistentState",
    "PersistentStateStore",
    "TensionApp",
    "TensionResult",
    "TensionSample",
    "collect_tension_samples",
    "determine_coordinates_for_wire",
    "evaluate_tension",
    "get_current_position",
    "identify_unmeasured_wires",
    "move_to_coordinates",
    "run_app",
]
