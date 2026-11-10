"""Left sidebar: project, models, integrations, run history."""
from __future__ import annotations

import time
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QPushButton, QVBoxLayout, QWidget,
)


def _label() -> QLabel:
    lab = QLabel("…")
    lab.setWordWrap(True)
    lab.setTextInteractionFlags(Qt.TextSelectableByMouse)
    return lab


def _ok(flag: bool | None, text: str) -> str:
    icon = {True: "🟢", False: "🔴", None: "⚪"}[flag]
    return f"{icon} {text}"


class ProjectPanel(QWidget):
    repo_changed = Signal(str)
    run_activated = Signal(int)
    smoke_claude = Signal()
    smoke_worker = Signal()
    refresh_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)

        # Project
        g = QGroupBox("Project")
        f = QFormLayout(g)
        row = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setReadOnly(True)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        row.addWidget(self.path_edit, 1)
        row.addWidget(browse)
        f.addRow(row)
        self.branch = _label()
        self.repo_status = _label()
        f.addRow("Branch:", self.branch)
        f.addRow("Status:", self.repo_status)
        lay.addWidget(g)

        # Models
        g = QGroupBox("Models")
        f = QFormLayout(g)
        self.claude = _label()
        self.claude_model = _label()
        self.worker = _label()
        self.ninfer = _label()
        self.qwen = _label()
        f.addRow("Claude CLI:", self.claude)
        f.addRow("Orchestrator:", self.claude_model)
        f.addRow("Worker exe:", self.worker)
        f.addRow("NInfer:", self.ninfer)
        f.addRow("Qwen:", self.qwen)
        btns = QHBoxLayout()
        b1 = QPushButton("Smoke test Claude")
        b1.clicked.connect(self.smoke_claude)
        b2 = QPushButton("Smoke test worker")
        b2.clicked.connect(self.smoke_worker)
        btns.addWidget(b1)
        btns.addWidget(b2)
        f.addRow(btns)
        lay.addWidget(g)

        # Integrations
        g = QGroupBox("Integrations")
        f = QFormLayout(g)
        self.cbm = _label()
        self.unreal = _label()
        f.addRow("codebase-memory:", self.cbm)
        f.addRow("Unreal MCP:", self.unreal)
        refresh = QPushButton("Refresh status")
        refresh.clicked.connect(self.refresh_requested)
        f.addRow(refresh)
        lay.addWidget(g)

        # History
        g = QGroupBox("Run history")
        v = QVBoxLayout(g)
        self.history = QListWidget()
        self.history.itemDoubleClicked.connect(self._activate)
        v.addWidget(self.history)
        hint = QLabel("Double-click a run to load it.")
        hint.setStyleSheet("color:#888")
        v.addWidget(hint)
        lay.addWidget(g, 1)

    def _browse(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select project repository", self.path_edit.text())
        if d:
            self.set_repo(d)
            self.repo_changed.emit(d)

    def set_repo(self, path: str) -> None:
        self.path_edit.setText(path)
        self.path_edit.setToolTip(path)

    def repo(self) -> str:
        return self.path_edit.text()

    def _activate(self, item: QListWidgetItem) -> None:
        self.run_activated.emit(int(item.data(Qt.UserRole)))

    def set_runs(self, runs: list[dict[str, Any]]) -> None:
        self.history.clear()
        for r in runs:
            when = time.strftime("%m-%d %H:%M", time.localtime(r["created_at"]))
            goal = r["goal"].replace("\n", " ")
            tag = "V2 " if (r.get("pipeline_version") or "v1") == "v2" else ""
            text = f"#{r['id']} {tag}{when} [{r['state']}] {r['n_passed']}/{r['n_tasks']}  {goal[:60]}"
            it = QListWidgetItem(text)
            it.setToolTip(f"{r['goal']}\n{r['repo_path']}")
            it.setData(Qt.UserRole, r["id"])
            self.history.addItem(it)

    def set_status(self, st: dict[str, Any], orchestrator_mode: str, worker_mode: str) -> None:
        c = st.get("claude", {})
        if c.get("ok"):
            ver = "verified ✔" if c.get("verified") else "NOT smoke-tested"
            self.claude.setText(_ok(True, f"{c.get('version', '')} — {ver}\n{c.get('path')}"))
        else:
            self.claude.setText(_ok(False, c.get("error", "unavailable")))
        mode = "MOCK (no Claude calls)" if orchestrator_mode == "mock" else "Claude Code CLI"
        self.claude_model.setText(f"{mode}; model: {c.get('model', '?')}")
        w = st.get("worker", {})
        prof = (f"Profile: {w['profile']} (default, runtime {w.get('runtime', 'codex')})\n"
                if w.get("profile") else "")
        if w.get("ok"):
            ver = "verified ✔" if w.get("verified") else "NOT smoke-tested"
            self.worker.setText(_ok(True, f"{prof}{w.get('version', '')} — {ver}\n{w.get('path')}\n"
                                          f"CODEX_HOME (child only): {w.get('codex_home')}"))
        else:
            self.worker.setText(_ok(False, prof + w.get("error", "unavailable")))
        n = st.get("ninfer", {})
        self.ninfer.setText(_ok(n.get("ok", False), n.get("url", "") + ("" if n.get("ok") else f" — {n.get('error', '')}")))
        wm = "MOCK worker (no edits)" if worker_mode == "mock" else "Codex NInfer"
        if n.get("ok"):
            self.qwen.setText(_ok(bool(n.get("model_loaded")),
                                  f"{'served' if n.get('model_loaded') else 'NOT served'}: "
                                  f"{', '.join(n.get('models', []))} — mode: {wm}"))
        else:
            self.qwen.setText(_ok(False, f"unknown — mode: {wm}"))
        cm = st.get("codebase_memory", {})
        self.cbm.setText(_ok(cm.get("ok", False), cm.get("text", "")))
        u = st.get("unreal", {})
        self.unreal.setText(_ok(u.get("ok") if u.get("is_unreal") else None, u.get("text", "")))
        r = st.get("repo")
        if r:
            self.branch.setText(r.get("branch") or ("(detached)" if r.get("is_repo") else "—"))
            self.repo_status.setText(_ok(r.get("is_repo"), r.get("text", "")))
