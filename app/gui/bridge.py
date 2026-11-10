"""Thread-safe bridge: engine-thread events -> Qt main thread."""
from __future__ import annotations

from typing import Any

from PySide6.QtCore import QObject, Signal


class EventBridge(QObject):
    event = Signal(str, object)

    def emit(self, kind: str, payload: dict[str, Any]) -> None:
        # Emitted from the engine thread; delivered queued to GUI-thread slots.
        self.event.emit(kind, payload)
