"""AI Dev Orchestrator entry point.

    python -m app.main
"""
from __future__ import annotations

import sys


def main() -> int:
    from PySide6.QtWidgets import QApplication

    from app.config.discovery import autofill
    from app.config.settings import SettingsStore
    from app.gui.bridge import EventBridge
    from app.gui.main_window import MainWindow
    from app.orchestration.orchestrator import EngineHost

    app = QApplication(sys.argv)
    app.setApplicationName("AI Dev Orchestrator")
    app.setStyle("Fusion")

    store = SettingsStore()
    settings = store.load()
    notes = autofill(settings)
    store.save(settings)

    bridge = EventBridge()
    host = EngineHost(settings, bridge.emit)
    win = MainWindow(settings, store, bridge, host)  # connects to the bridge before any event is emitted
    for n in notes:
        win.terminal.append("SYSTEM", n)
    win.terminal.append("SYSTEM", f"settings: {store.path}")
    host.start()
    win.engine_started()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
