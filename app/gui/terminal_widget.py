"""Live activity console with per-source labels and filters."""
from __future__ import annotations

import html
import time
from collections import deque

from PySide6.QtCore import QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget

SOURCES = {
    "CLAUDE": "#d08770",
    "QWEN": "#88c0d0",
    "CLINE": "#8fc9a8",  # V5.1: the Cline worker runtime
    "SYSTEM": "#a3be8c",
    "REPAIR": "#bf616a",
    "VERIFY": "#ebcb8b",
    "GIT": "#b48ead",
    "MCP": "#81a1c1",
    "HUMAN": "#8fbcbb",  # V4: the user's answers to human verification requests
    "DISCOVERY": "#5e81ac",  # V4 pass 2: capability discovery
    "EVIDENCE": "#b3d18c",  # V4 pass 2: evidence plan and results
}
MAX_LINES = 20000


class TerminalWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._entries: deque[tuple[float, str, str]] = deque(maxlen=MAX_LINES)
        self._pending: list[tuple[float, str, str]] = []
        self._filters: dict[str, QCheckBox] = {}

        bar = QHBoxLayout()
        for src, color in SOURCES.items():
            cb = QCheckBox(src)
            cb.setChecked(True)
            cb.setStyleSheet(f"QCheckBox {{ color: {color}; font-weight: bold; }}")
            cb.toggled.connect(self._rerender)
            self._filters[src] = cb
            bar.addWidget(cb)
        bar.addStretch(1)
        self.autoscroll = QCheckBox("Auto-scroll")
        self.autoscroll.setChecked(True)
        bar.addWidget(self.autoscroll)
        clear = QPushButton("Clear")
        clear.clicked.connect(self.clear)
        bar.addWidget(clear)

        self.view = QPlainTextEdit()
        self.view.setReadOnly(True)
        self.view.setMaximumBlockCount(MAX_LINES)
        f = QFont("Cascadia Mono")
        f.setStyleHint(QFont.Monospace)
        f.setPointSize(9)
        self.view.setFont(f)
        self.view.setStyleSheet("QPlainTextEdit { background: #1e222a; color: #d8dee9; }")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(bar)
        lay.addWidget(self.view)

        # Batch appends so high-rate subprocess output doesn't stall the UI.
        self._timer = QTimer(self)
        self._timer.setInterval(80)
        self._timer.timeout.connect(self._flush)
        self._timer.start()

    def append(self, source: str, text: str, ts: float | None = None) -> None:
        e = (ts or time.time(), source, text)
        self._entries.append(e)
        self._pending.append(e)

    def clear(self) -> None:
        self._entries.clear()
        self._pending.clear()
        self.view.clear()

    def _visible(self, source: str) -> bool:
        cb = self._filters.get(source)
        return cb.isChecked() if cb else True

    def _fmt(self, e: tuple[float, str, str]) -> str:
        ts, src, text = e
        color = SOURCES.get(src, "#d8dee9")
        t = time.strftime("%H:%M:%S", time.localtime(ts))
        return (f'<span style="color:#616e88">{t}</span> '
                f'<span style="color:{color};font-weight:bold">[{src:<6}]</span> '
                f'<span style="white-space:pre">{html.escape(text)}</span>')

    def _flush(self) -> None:
        if not self._pending:
            return
        batch, self._pending = self._pending, []
        lines = [self._fmt(e) for e in batch if self._visible(e[1])]
        if not lines:
            return
        sb = self.view.verticalScrollBar()
        self.view.appendHtml("<br>".join(lines))
        if self.autoscroll.isChecked():
            sb.setValue(sb.maximum())

    def _rerender(self) -> None:
        self._pending.clear()
        self.view.clear()
        lines = [self._fmt(e) for e in self._entries if self._visible(e[1])]
        for i in range(0, len(lines), 500):
            self.view.appendHtml("<br>".join(lines[i:i + 500]))
