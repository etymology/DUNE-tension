"""Tk-safe logging support for the tensiometer GUI."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from queue import Empty, SimpleQueue
from typing import Any

LOGGER_NAMES = ("dune_tension", "spectrum_analysis", "valve_trigger")


class TkTextLogHandler(logging.Handler):
    """Queue log records and append them to a Tk text widget on the UI thread."""

    def __init__(
        self,
        root: Any,
        text_widget: Any,
        *,
        poll_interval_ms: int = 100,
        max_lines: int = 1000,
    ) -> None:
        super().__init__(level=logging.INFO)
        self.root = root
        self.text_widget = text_widget
        self.poll_interval_ms = int(max(1, poll_interval_ms))
        self.max_lines = int(max(1, max_lines))
        self._queue: SimpleQueue[str] = SimpleQueue()
        self._after_id: Any | None = None
        self._closed = False
        self._schedule_drain()

    def emit(self, record: logging.LogRecord) -> None:
        if self._closed:
            return
        try:
            message = self.format(record)
        except Exception:
            self.handleError(record)
            return
        self._queue.put(message)

    def close(self) -> None:
        self._closed = True
        try:
            after_cancel = getattr(self.root, "after_cancel", None)
            if self._after_id is not None and callable(after_cancel):
                after_cancel(self._after_id)
        except Exception:
            pass
        self._after_id = None
        super().close()

    def _schedule_drain(self) -> None:
        if self._closed:
            return
        try:
            self._after_id = self.root.after(self.poll_interval_ms, self._drain_queue)
        except Exception:
            self._after_id = None

    def _drain_queue(self) -> None:
        if self._closed:
            return

        messages: list[str] = []
        while True:
            try:
                messages.append(self._queue.get_nowait())
            except Empty:
                break

        if messages:
            self._append_messages(messages)

        self._schedule_drain()

    def _append_messages(self, messages: list[str]) -> None:
        configure = getattr(self.text_widget, "configure", None)
        insert = getattr(self.text_widget, "insert", None)
        if not callable(insert):
            return

        try:
            if callable(configure):
                configure(state="normal")
            for message in messages:
                insert("end", f"{message}\n")
            self._trim_lines()
            see = getattr(self.text_widget, "see", None)
            if callable(see):
                see("end")
        finally:
            try:
                if callable(configure):
                    configure(state="disabled")
            except Exception:
                pass

    def _trim_lines(self) -> None:
        index = getattr(self.text_widget, "index", None)
        delete = getattr(self.text_widget, "delete", None)
        if not callable(index) or not callable(delete):
            return

        try:
            end_index = str(index("end-1c"))
            line_count = int(end_index.split(".", 1)[0])
        except Exception:
            return

        overflow = line_count - self.max_lines
        if overflow > 0:
            try:
                delete("1.0", f"{overflow + 1}.0")
            except Exception:
                pass


@dataclass
class GuiLogBinding:
    """Track logger state so GUI logging can be detached cleanly."""

    handler: TkTextLogHandler
    logger_states: tuple[tuple[logging.Logger, int, bool], ...]

    def close(self) -> None:
        for logger, level, propagate in self.logger_states:
            try:
                logger.removeHandler(self.handler)
            except Exception:
                pass
            logger.setLevel(level)
            logger.propagate = propagate
        self.handler.close()


def configure_gui_logging(root: Any, text_widget: Any) -> GuiLogBinding | None:
    """Attach a text-backed handler to the GUI-relevant logger namespaces."""

    if text_widget is None:
        return None

    handler = TkTextLogHandler(root, text_widget)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    )

    logger_states: list[tuple[logging.Logger, int, bool]] = []
    for name in LOGGER_NAMES:
        logger = logging.getLogger(name)
        logger_states.append((logger, logger.level, logger.propagate))
        logger.setLevel(logging.INFO)
        logger.propagate = False
        logger.addHandler(handler)

    return GuiLogBinding(handler=handler, logger_states=tuple(logger_states))
