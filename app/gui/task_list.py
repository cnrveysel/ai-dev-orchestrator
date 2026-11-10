"""Center panel: the sequential task list."""
from __future__ import annotations

import time
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHeaderView, QTreeWidget, QTreeWidgetItem

ICONS = {
    "PENDING": "🔒", "READY": "▶️", "RUNNING": "🟡", "VERIFYING": "🔍", "REVIEWING": "🧠",
    "RETRY": "🔁", "PASSED": "✅", "FAILED": "❌", "BLOCKED": "⛔", "PAUSED": "⏸️", "SUPERSEDED": "↪️",
    "AWAITING_HUMAN": "🙋",
}
COLS = ["ID", "Title", "Status", "Attempt", "Duration", "Depends on"]


def fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return ""
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


class TaskList(QTreeWidget):
    task_selected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setColumnCount(len(COLS))
        self.setHeaderLabels(COLS)
        self.setRootIsDecorated(False)
        self.setUniformRowHeights(True)
        self.setAlternatingRowColors(True)
        h = self.header()
        h.setSectionResizeMode(1, QHeaderView.Stretch)
        for i in (0, 2, 3, 4, 5):
            h.setSectionResizeMode(i, QHeaderView.ResizeToContents)
        self._items: dict[str, QTreeWidgetItem] = {}
        self.tasks: dict[str, dict[str, Any]] = {}
        self.max_attempts = 3
        self.currentItemChanged.connect(self._on_current)

    def set_tasks(self, tasks: list[dict[str, Any]]) -> None:
        selected = self.selected_id()
        self.clear()
        self._items.clear()
        self.tasks.clear()
        # V2 phases show their internal subtasks as child rows.
        self.setRootIsDecorated(any(t.get("subtasks") for t in tasks))
        for t in sorted(tasks, key=lambda x: x.get("position", 0)):
            item = QTreeWidgetItem()
            item.setData(0, Qt.UserRole, t["id"])
            self.addTopLevelItem(item)
            self._items[t["id"]] = item
            self.update_task(t)
        if selected and selected in self._items:
            self.setCurrentItem(self._items[selected])

    def update_task(self, t: dict[str, Any]) -> None:
        self.tasks[t["id"]] = t
        item = self._items.get(t["id"])
        if item is None:
            return
        status = t.get("status", "PENDING")
        item.setText(0, f"{ICONS.get(status, '')} {t['id']}")
        item.setText(1, t.get("title", ""))
        item.setText(2, status)
        budget = self.max_attempts + int(t.get("extra_attempts", 0) or 0)
        att = int(t.get("attempts", 0) or 0)
        item.setText(3, f"{att}/{budget}" if att else "")
        item.setText(5, ", ".join(t.get("dependencies", [])))
        self._set_duration(item, t)
        bold = status in ("RUNNING", "VERIFYING", "REVIEWING", "RETRY", "AWAITING_HUMAN")
        f = item.font(0)
        f.setBold(bold)
        for c in range(len(COLS)):
            item.setFont(c, f)
        subtasks = t.get("subtasks") or []
        if subtasks:
            self._set_subtasks(item, t["id"], subtasks, status)

    def _set_subtasks(self, item: QTreeWidgetItem, phase_id: str, subtasks: list[dict[str, Any]], status: str) -> None:
        while item.childCount() > len(subtasks):
            item.removeChild(item.child(item.childCount() - 1))
        for i, st in enumerate(subtasks):
            child = item.child(i) if i < item.childCount() else None
            if child is None:
                child = QTreeWidgetItem()
                child.setData(0, Qt.UserRole, phase_id)  # selecting a subtask shows its phase
                item.addChild(child)
            child.setText(0, f"   {'☑' if st.get('done') else '☐'} {st.get('id', '')}")
            child.setText(1, st.get("title", ""))
            child.setText(2, "done" if st.get("done") else "")
        if status in ("RUNNING", "VERIFYING", "REVIEWING", "RETRY", "PAUSED", "BLOCKED", "AWAITING_HUMAN"):
            item.setExpanded(True)

    def _set_duration(self, item: QTreeWidgetItem, t: dict[str, Any]) -> None:
        start, end = t.get("started_at"), t.get("finished_at")
        if start is None:
            item.setText(4, "")
            return
        live = t.get("status") in ("RUNNING", "VERIFYING", "REVIEWING", "RETRY")
        item.setText(4, fmt_duration((end if end else (time.time() if live else start)) - start)
                     if (end or live) else "")

    def tick(self) -> None:
        for tid, t in self.tasks.items():
            if t.get("status") in ("RUNNING", "VERIFYING", "REVIEWING", "RETRY"):
                self._set_duration(self._items[tid], t)

    def selected_id(self) -> str | None:
        item = self.currentItem()
        return item.data(0, Qt.UserRole) if item else None

    def _on_current(self, cur, _prev) -> None:
        if cur is not None:
            self.task_selected.emit(cur.data(0, Qt.UserRole))
