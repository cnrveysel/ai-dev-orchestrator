"""Main window: wiring between widgets and the engine (via EngineHost)."""
from __future__ import annotations

import time
from typing import Any

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QFont
from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QInputDialog, QLabel, QMainWindow, QMessageBox, QPlainTextEdit, QPushButton, QSplitter,
    QVBoxLayout, QWidget,
)

from app.config.settings import Settings, SettingsStore
from app.gui.bridge import EventBridge
from app.gui.human_gate import HumanGatePanel
from app.gui.project_panel import ProjectPanel
from app.gui.settings_dialog import SettingsDialog
from app.gui.task_details import TaskDetails
from app.gui.task_list import TaskList, fmt_duration
from app.gui.terminal_widget import TerminalWidget
from app.orchestration.orchestrator import EngineHost
from app.workers.profile_registry import ProfileError, WorkerProfileRegistry
from app.workers.profiles import LEGACY_PROFILE_ID
from app.workers.runtimes import get_runtime

EXEC_STATES = {"STARTING_TASK", "WORKER_RUNNING", "VERIFYING", "REVIEWING", "RETRY_PENDING", "ADVANCING",
               "REPLANNING", "REPAIRING", "COLLECTING_EVIDENCE", "CONSISTENCY_REVIEW"}
PIPELINE_ITEMS = [("V1 · task by task", "v1"), ("V2 · phases", "v2")]
HALTED = {"PAUSED", "BLOCKED", "STOPPED", "ERROR"}


class MainWindow(QMainWindow):
    def __init__(self, settings: Settings, store: SettingsStore, bridge: EventBridge, host: EngineHost):
        super().__init__()
        self.settings = settings
        self.store = store
        self.host = host
        self.state = "IDLE"
        self.run_id: int | None = None
        self.run_started_at: float | None = None
        self.current_task_id: str | None = None
        self.plan: dict[str, Any] | None = None
        self.attempts: dict[str, dict[int, dict[str, Any]]] = {}
        self.tool_calls = 0
        self.claude_cost: float | None = None
        self.claude_calls = 0
        self.busy_op = False
        self.run_pipeline = "v1"  # pipeline of the loaded run (the selector applies to the next PLAN)
        self.human: dict[str, Any] | None = None  # V4: pending human verification request of the loaded run
        self.run_worker_profile: dict[str, Any] | None = None  # V5: the loaded run's locked worker profile
        self.setWindowTitle("AI Dev Orchestrator — Claude tech lead + local Qwen worker")
        self.resize(1600, 1000)
        self._build_ui()
        self._build_menu()
        bridge.event.connect(self._on_event)
        self._tick_timer = QTimer(self)
        self._tick_timer.setInterval(1000)
        self._tick_timer.timeout.connect(self._tick)
        self._tick_timer.start()
        if settings.last_repo_path:
            self.project.set_repo(settings.last_repo_path)
        self._update_buttons()

    def engine_started(self) -> None:
        """Called once the engine thread is up (the bridge was connected before it started)."""
        self._refresh_status()

    # ------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        # Top bar
        self.task_input = QPlainTextEdit()
        self.task_input.setPlaceholderText("Describe the development goal, e.g. \"Implement a complete inventory "
                                           "and crafting system for this Unreal game.\"")
        self.task_input.setFixedHeight(80)
        f = self.task_input.font()
        f.setPointSize(11)
        self.task_input.setFont(f)
        self.btn_plan = QPushButton("PLAN")
        self.btn_start = QPushButton("START")
        self.btn_pause = QPushButton("PAUSE")
        self.btn_resume = QPushButton("RESUME")
        self.btn_stop = QPushButton("STOP")
        for b, slot in [(self.btn_plan, self._plan), (self.btn_start, self._start), (self.btn_pause, self._pause),
                        (self.btn_resume, self._resume), (self.btn_stop, self._stop)]:
            b.setMinimumHeight(36)
            b.setMinimumWidth(90)
            bf = QFont(b.font())
            bf.setBold(True)
            b.setFont(bf)
            b.clicked.connect(slot)
        self.btn_stop.setStyleSheet("QPushButton:enabled { color: #b00; }")
        self.pipeline_combo = QComboBox()
        for text, value in PIPELINE_ITEMS:
            self.pipeline_combo.addItem(text, value)
        self.pipeline_combo.setToolTip("Pipeline for the next PLAN. V1: Claude reviews every task. V2: coherent "
                                       "phases, one Qwen session per phase, Claude checkpoints at phase boundaries.\n"
                                       "A loaded run always keeps the pipeline it was planned with.")
        self._set_pipeline_combo(self.settings.orchestration.pipeline_version)
        self.pipeline_combo.currentIndexChanged.connect(self._pipeline_changed)
        self.state_label = QLabel("IDLE")
        self.state_label.setStyleSheet("font-weight:bold; font-size:14px; padding:4px;")
        self.mode_label = QLabel()
        self.status_msg = QLabel()
        self.status_msg.setWordWrap(True)
        btns = QHBoxLayout()
        btns.addWidget(QLabel("Pipeline:"))
        btns.addWidget(self.pipeline_combo)
        for b in (self.btn_plan, self.btn_start, self.btn_pause, self.btn_resume, self.btn_stop):
            btns.addWidget(b)
        btns.addSpacing(12)
        btns.addWidget(self.state_label)
        btns.addWidget(self.mode_label)
        btns.addStretch(1)
        top = QWidget()
        tl = QVBoxLayout(top)
        tl.setContentsMargins(4, 4, 4, 0)
        tl.addWidget(self.task_input)
        tl.addLayout(btns)
        tl.addWidget(self.status_msg)

        self.project = ProjectPanel()
        self.project.setMinimumWidth(300)
        self.project.repo_changed.connect(self._repo_changed)
        self.project.run_activated.connect(self._load_run)
        self.project.smoke_claude.connect(self._smoke_claude)
        self.project.smoke_worker.connect(lambda: self._smoke_worker())
        self.project.refresh_requested.connect(self._refresh_status)

        self.tasks = TaskList()
        self.tasks.max_attempts = self.settings.orchestration.max_attempts
        self.tasks.task_selected.connect(self._show_task)
        self.details = TaskDetails()
        self.terminal = TerminalWidget()
        self.human_panel = HumanGatePanel()
        self.human_panel.feedback_submitted.connect(self._human_feedback)
        right = QSplitter(Qt.Vertical)  # V4: the human verification panel appears above the details
        right.addWidget(self.human_panel)
        right.addWidget(self.details)
        right.setSizes([330, 400])

        center = QSplitter(Qt.Horizontal)
        center.addWidget(self.tasks)
        center.addWidget(right)
        center.setSizes([650, 650])
        vert = QSplitter(Qt.Vertical)
        vert.addWidget(center)
        vert.addWidget(self.terminal)
        vert.setSizes([520, 360])
        main = QSplitter(Qt.Horizontal)
        main.addWidget(self.project)
        main.addWidget(vert)
        main.setSizes([340, 1260])

        root = QWidget()
        rl = QVBoxLayout(root)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(top)
        rl.addWidget(main, 1)
        self.setCentralWidget(root)

        sb = self.statusBar()
        self.m_speed = QLabel("Qwen: — tok/s")
        self.m_task = QLabel("Task: —")
        self.m_run = QLabel("Run: —")
        self.m_attempt = QLabel("Attempt: —")
        self.m_tools = QLabel("Tool calls: 0")
        self.m_claude = QLabel("Claude: —")
        self.m_phase = QLabel("")  # V2: phase / subtask / todo progress and last checkpoint
        for w in (self.m_speed, self.m_phase, self.m_task, self.m_run, self.m_attempt, self.m_tools, self.m_claude):
            w.setStyleSheet("padding: 0 10px;")
            sb.addPermanentWidget(w)
        self._update_mode_label()

    def _build_menu(self) -> None:
        m = self.menuBar().addMenu("&File")
        a = QAction("&Settings…", self)
        a.triggered.connect(self._open_settings)
        m.addAction(a)
        m.addSeparator()
        a = QAction("E&xit", self)
        a.triggered.connect(self.close)
        m.addAction(a)
        m = self.menuBar().addMenu("&Tools")
        for text, slot in [("Smoke test Claude Code CLI", self._smoke_claude),
                           ("Smoke test worker profile (full test)…", self._smoke_worker_choose),
                           ("Refresh status", self._refresh_status),
                           ("V2: start a fresh Qwen session for the current phase", self._fresh_session)]:
            a = QAction(text, self)
            a.triggered.connect(slot)
            m.addAction(a)

    def _update_mode_label(self) -> None:
        o = self.settings.orchestration
        orch = "MOCK" if o.orchestrator_mode == "mock" else "Claude CLI"
        work = "MOCK" if o.worker_mode == "mock" else f"{self._worker_runtime_name()} · {self._worker_profile_name()}"
        color = "#b60" if "MOCK" in (orch, work) else "#080"
        pipe = f" · Run pipeline: <b>{self.run_pipeline.upper()}</b>" if self.run_id is not None else ""
        self.mode_label.setText(f"<span style='color:{color}'>Orchestrator: <b>{orch}</b> · Worker: <b>{work}</b>"
                                f"{pipe}</span>")

    def _worker_profile(self):
        """(runtime id, profile) of the loaded run's locked profile, else of the default profile (V5.1)."""
        if self.run_worker_profile:
            return self.run_worker_profile.get("runtime") or "codex", None
        p = WorkerProfileRegistry(self.settings).default_profile()
        return (p.runtime if p else "codex"), p

    def _worker_runtime_name(self) -> str:
        runtime = get_runtime(self._worker_profile()[0])
        return runtime.display_name if runtime else self._worker_profile()[0]

    def _worker_profile_name(self) -> str:
        """The loaded run's locked worker profile, else the default profile a new run would lock (V5)."""
        if self.run_worker_profile:
            return self.run_worker_profile.get("name") or self.run_worker_profile.get("id", "?")
        p = WorkerProfileRegistry(self.settings).default_profile()
        return f"{p.name if p else self.settings.worker_profiles.default_profile_id} (default)"

    def _set_pipeline_combo(self, value: str) -> None:
        idx = max(0, self.pipeline_combo.findData((value or "v1").lower()))
        self.pipeline_combo.blockSignals(True)
        self.pipeline_combo.setCurrentIndex(idx)
        self.pipeline_combo.blockSignals(False)

    def _pipeline_changed(self, _index: int) -> None:
        self.settings.orchestration.pipeline_version = self.pipeline_combo.currentData() or "v1"
        self.store.save(self.settings)
        new = self.settings
        self._submit(lambda e: e.apply_settings(new))

    def _update_buttons(self) -> None:
        s = self.state
        has_plan = self.plan is not None
        self.btn_plan.setEnabled(not self.busy_op and s not in EXEC_STATES | {"PLANNING", "STOPPING"})
        self.pipeline_combo.setEnabled(s not in EXEC_STATES | {"PLANNING", "STOPPING"})
        self.btn_start.setEnabled(not self.busy_op and s == "PLAN_READY")
        self.btn_pause.setEnabled(s in EXEC_STATES)
        # AWAITING_HUMAN: an answer continues the run; RESUME only re-runs an interrupted follow-up.
        answered = s == "AWAITING_HUMAN" and (self.human or {}).get("status") == "feedback_received"
        self.btn_resume.setEnabled(not self.busy_op and has_plan and (s in HALTED or answered))
        self.btn_stop.setEnabled(s in EXEC_STATES | {"PLANNING", "PLAN_READY", "PAUSED", "BLOCKED", "ERROR",
                                                     "AWAITING_HUMAN"})
        colors = {"COMPLETED": "#080", "BLOCKED": "#b00", "ERROR": "#b00", "PAUSED": "#b60", "STOPPED": "#666",
                  "AWAITING_HUMAN": "#07a"}
        self.state_label.setText(s)
        self.state_label.setStyleSheet(f"font-weight:bold; font-size:14px; padding:4px; color:{colors.get(s, '#05a')}")

    # ------------------------------------------------------------ actions

    def _submit(self, fn) -> None:
        # Results come back as engine events through the bridge (queued onto the GUI thread).
        self.host.submit(fn)

    def _repo_changed(self, path: str) -> None:
        self.settings.last_repo_path = path
        self.store.save(self.settings)
        self._refresh_status()

    def _refresh_status(self) -> None:
        repo = self.project.repo() or None
        self._submit(lambda e: e.refresh_status(repo))

    def _plan(self) -> None:
        goal = self.task_input.toPlainText().strip()
        repo = self.project.repo()
        if not goal:
            QMessageBox.information(self, "PLAN", "Enter a development request first.")
            return
        if not repo:
            QMessageBox.information(self, "PLAN", "Select a project directory first.")
            return
        self.plan = None
        self.attempts.clear()
        self.tasks.set_tasks([])
        self.details.show_plan_overview(None)
        self.claude_cost, self.claude_calls, self.tool_calls = None, 0, 0
        pipeline = self.pipeline_combo.currentData() or "v1"
        self._submit(lambda e: e.plan(goal, repo, pipeline))

    def _start(self) -> None:
        if self.settings.orchestration.worker_mode == "ninfer":
            mock_note = ("\n\nPlanning and review are MOCK: the plan is placeholder tasks and every review "
                         "is simulated. Use a disposable repository."
                         if self.settings.orchestration.orchestrator_mode == "mock" else "")
            runtime = get_runtime(self._worker_profile()[0])
            if runtime is not None and not runtime.capabilities.sandbox:
                mock_note += (f"\n\nWorker sandbox: NONE - the {runtime.display_name} runtime has no command sandbox "
                              "(its commands are not confined to the repository).")
            elif self.settings.worker.sandbox == "danger-full-access":
                mock_note += "\n\nWorker sandbox: NONE (danger-full-access)."
            r = QMessageBox.question(self, "START", f"The REAL worker ({self._worker_runtime_name()} · "
                                                    f"{self._worker_profile_name()}) will EDIT "
                                                    f"files in\n{self.project.repo()}{mock_note}\n\nContinue?")
            if r != QMessageBox.Yes:
                return
        self._submit(lambda e: e.start())

    def _pause(self) -> None:
        self._submit(lambda e: e.pause())

    def _resume(self) -> None:
        guidance = ""
        if self.state == "BLOCKED":
            text, ok = QInputDialog.getMultiLineText(
                self, "Resume blocked run",
                f"Run is BLOCKED:\n{self.status_msg.text()}\n\nOptional guidance for the worker "
                "(sent with the next attempt). If the retry limit was reached, one more attempt is granted.")
            if not ok:
                return
            guidance = text
        self._submit(lambda e: e.resume(guidance))

    def _stop(self) -> None:
        self._submit(lambda e: e.stop())

    def _fresh_session(self) -> None:
        self._submit(lambda e: e.request_fresh_session())

    def _human_feedback(self, verdict: str, note: str) -> None:
        self._submit(lambda e: e.submit_human_feedback(verdict, note))

    def _load_run(self, run_id: int) -> None:
        if self.state in EXEC_STATES:
            QMessageBox.warning(self, "History", "Stop the active run before loading another one.")
            return
        self.attempts.clear()
        self.plan = None
        self.claude_cost, self.claude_calls, self.tool_calls = None, 0, 0
        self._submit(lambda e: e.load_run(run_id))

    def _smoke_claude(self) -> None:
        self.busy_op = True
        self._update_buttons()
        self._submit(lambda e: e.smoke_test_claude())

    def _smoke_worker(self, profile_id: str | None = None) -> None:
        """Full test of one worker profile (None = the default profile; on a V4 installation the legacy worker)."""
        self.busy_op = True
        self._update_buttons()
        self._submit(lambda e: e.smoke_test_worker(profile_id))

    def _smoke_worker_choose(self) -> None:
        reg = WorkerProfileRegistry(self.settings)
        profiles = sorted(reg.profiles(), key=lambda p: p.id != reg.default_id())
        items = [f"{p.name}  ({p.id})" + ("  [default]" if p.id == reg.default_id() else "") for p in profiles]
        item, ok = QInputDialog.getItem(self, "Worker profile full test", "Profile (the saved settings are tested):",
                                        items, 0, False)
        if ok:
            self._smoke_worker(profiles[items.index(item)].id)

    def _profiles_in_use(self) -> dict[str, list[int]] | None:
        """V5: which profiles resumable runs are bound to (asked from the engine thread, which owns the DB)."""
        try:
            return self.host.submit(lambda e: e.profiles_in_use()).result(timeout=10)
        except Exception:
            return None

    def _smoke_done(self, res: dict[str, Any] | None) -> None:
        self.busy_op = False
        self._update_buttons()
        if not res:
            return
        if res["target"] == "claude" and res["ok"]:
            self.settings.claude.verified_executable = res["executable"]
            self.settings.claude.verified_at = res["verified_at"]
        if res["target"] == "worker":
            try:  # the verification belongs to the tested profile (legacy = the V4 worker settings)
                WorkerProfileRegistry(self.settings).record_verification(
                    res.get("profile_id") or LEGACY_PROFILE_ID, bool(res["verified_executable"]),
                    res["verified_executable"], res["verified_codex_home"], res["verified_at"])
            except ProfileError:
                pass  # the profile was removed meanwhile
        self.store.save(self.settings)
        name = ("Claude Code CLI" if res["target"] == "claude"
                else f"Worker profile '{res.get('profile_name') or 'Codex NInfer worker'}'")
        extra = res.get("extra") or {}
        if res["target"] == "worker":
            where = (f"Session:\n{extra.get('session_id') or '(not found)'}  (provider {extra.get('provider')}, "
                     f"model {extra.get('model')})\n" if "session_id" in extra
                     else f"CODEX_HOME:\n{extra.get('codex_home', '(not set)')}\n")
            lines = [f"Worker executable:\n{res['executable']}\n",
                     where,
                     f"Response:\n{extra.get('reply', '')}\n",
                     f"Exit code: {extra.get('exit_code')}    Elapsed: {extra.get('elapsed')}s    "
                     f"Files created: {', '.join(extra.get('files_created') or []) or 'none'}"]
            if "ninfer_tokens_predicted_delta" in extra:
                lines.append(f"NInfer tokens_predicted delta: {extra['ninfer_tokens_predicted_delta']:+.0f}")
            lines.append("Normal `codex` launcher was NOT used.")
            lines.append("REAL worker mode is now allowed in Settings." if res["ok"]
                         else f"FAILED: {res['detail']}\nREAL worker mode stays disabled.")
        else:
            lines = [f"Executable used: {res['executable']}", res["detail"]]
        (QMessageBox.information if res["ok"] else QMessageBox.critical)(
            self, f"{name} smoke test {'PASSED' if res['ok'] else 'FAILED'}", "\n".join(lines))
        self._refresh_status()

    def _open_settings(self) -> None:
        dlg = SettingsDialog(self.settings, self, in_use_provider=self._profiles_in_use)
        if dlg.exec():
            self.settings = dlg.settings
            self.store.save(self.settings)
            self.tasks.max_attempts = self.settings.orchestration.max_attempts
            self._set_pipeline_combo(self.settings.orchestration.pipeline_version)
            self._update_mode_label()
            new = self.settings
            self._submit(lambda e: e.apply_settings(new))
            self._refresh_status()
            if dlg.full_test_profile:
                self._smoke_worker(dlg.full_test_profile)

    # ------------------------------------------------------------ events

    def _on_event(self, kind: str, p: dict[str, Any]) -> None:
        handler = getattr(self, f"_ev_{kind}", None)
        if handler:
            handler(p)

    def _ev_log(self, p):
        self.terminal.append(p["source"], p["text"], p.get("ts"))

    def _ev_history_logs(self, p):
        self.terminal.clear()
        self.terminal.append("SYSTEM", "---- persisted log of loaded run ----")
        for row in p["logs"]:
            self.terminal.append(row["source"], row["message"], row["ts"])

    def _ev_state(self, p):
        self.state = p["state"]
        self.human_panel.set_request(self.human, self.state)
        self._update_buttons()

    def _ev_run(self, p):
        self.run_id = p["run_id"]
        self.state = p["state"]
        if p.get("pipeline_version") and p["pipeline_version"] != self.run_pipeline:
            self.run_pipeline = p["pipeline_version"]
            self._update_mode_label()
        self.run_started_at = p.get("started_at")
        self.current_task_id = p.get("current_task_id")
        self.status_msg.setText(p.get("message") or "")
        self.human = p.get("human")
        self.human_panel.set_request(self.human, self.state)
        if p.get("worker_profile") != self.run_worker_profile:
            self.run_worker_profile = p.get("worker_profile")
            self._update_mode_label()
        self._update_buttons()

    def _ev_run_loaded(self, p):
        self.run_id = p["run_id"]
        self.run_pipeline = p.get("pipeline_version") or "v1"
        self._update_mode_label()
        self.task_input.setPlainText(p["goal"])
        if p.get("repo_path") and p["repo_path"] != self.project.repo():
            self.project.set_repo(p["repo_path"])

    def _ev_plan(self, p):
        self.plan = p["plan"]
        self.tasks.set_tasks(self.plan["tasks"])
        self.details.show_plan_overview(self.plan)
        self._update_buttons()

    def _ev_tasks(self, p):
        if self.plan is not None:
            self.plan["tasks"] = p["tasks"]
        self.tasks.set_tasks(p["tasks"])

    def _ev_task(self, p):
        t = p["task"]
        self.tasks.update_task(t)
        if self.tasks.selected_id() == t["id"]:
            self._show_task(t["id"])

    def _ev_attempt(self, p):
        a = p["attempt"]
        self.attempts.setdefault(a["task_id"], {})[a["number"]] = a
        if a.get("finished_at") is None:
            self.tool_calls = 0
        if self.tasks.selected_id() == a["task_id"]:
            self._show_task(a["task_id"])

    def _ev_tool_calls(self, p):
        self.tool_calls = p["count"]
        self.m_tools.setText(f"Tool calls: {self.tool_calls}")

    def _ev_orchestrator_call(self, p):
        self.claude_calls += 1
        if p.get("cost_usd") is not None:
            self.claude_cost = (self.claude_cost or 0.0) + p["cost_usd"]
        self._update_claude_label()

    def _ev_run_cost(self, p):
        self.claude_cost = p.get("cost_usd")
        self._update_claude_label()

    def _update_claude_label(self):
        cost = f" · CLI-reported cost ${self.claude_cost:.4f}" if self.claude_cost is not None else ""
        self.m_claude.setText(f"Claude calls: {self.claude_calls}{cost}")

    def _ev_metrics(self, p):
        if not p.get("ok"):
            self.m_speed.setText("Qwen: metrics unavailable")
            self.m_speed.setToolTip(p.get("error", ""))
        elif p.get("tokens_per_second") is None:
            self.m_speed.setText("Qwen: idle")
        else:
            self.m_speed.setText(f"Qwen: {p['tokens_per_second']:.1f} tok/s")

    def _ev_status(self, st):
        o = self.settings.orchestration
        self.project.set_status(st, o.orchestrator_mode, o.worker_mode)

    def _ev_runs(self, p):
        self.project.set_runs(p["runs"])

    def _ev_error(self, p):
        self.busy_op = False
        self.human_panel.set_request(self.human, self.state)  # e.g. a rejected answer: answerable again
        self._update_buttons()
        QMessageBox.warning(self, "AI Dev Orchestrator", p["message"])

    def _ev_smoke(self, p):
        self._smoke_done(p)

    # ------------------------------------------------------------ details / ticking

    def _show_task(self, task_id: str) -> None:
        t = self.tasks.tasks.get(task_id)
        if t is None:
            return
        attempts = [self.attempts[task_id][n] for n in sorted(self.attempts.get(task_id, {}))]
        self.details.show_task(t, attempts)

    def _tick(self) -> None:
        self.tasks.tick()
        now = time.time()
        self.m_run.setText(f"Run: {fmt_duration(now - self.run_started_at)}"
                           if self.run_started_at and self.state in EXEC_STATES else
                           ("Run: —" if not self.run_started_at else self.m_run.text()))
        tid = self.current_task_id
        v2 = self.run_pipeline == "v2"
        unit = "Phase" if v2 else "Task"
        if tid and tid in self.tasks.tasks:
            t = self.tasks.tasks[tid]
            budget = self.settings.orchestration.max_attempts + int(t.get("extra_attempts", 0) or 0)
            self.m_attempt.setText(f"Attempt: {tid} {t.get('attempts', 0)}/{budget}")
            atts = self.attempts.get(tid, {})
            if atts:
                last = atts[max(atts)]
                if last.get("finished_at") is None and self.state in EXEC_STATES:
                    self.m_task.setText(f"{unit}: {fmt_duration(now - last['started_at'])}")
            if v2 and t.get("started_at") and t.get("status") not in ("PASSED", "SUPERSEDED") and self.state in EXEC_STATES:
                self.m_task.setText(f"{unit}: {fmt_duration(now - t['started_at'])}")
        else:
            self.m_attempt.setText("Attempt: —")
        self.m_phase.setText(self._phase_progress_text(tid) if v2 else "")

    def _phase_progress_text(self, tid: str | None) -> str:
        phases = sorted(self.tasks.tasks.values(), key=lambda x: x.get("position", 0))
        if not phases:
            return ""
        done = sum(1 for x in phases if x.get("status") in ("PASSED", "SUPERSEDED"))
        t = self.tasks.tasks.get(tid) if tid else None
        if t is None:
            return f"Phases: {done}/{len(phases)} done"
        idx = next((i for i, x in enumerate(phases, 1) if x["id"] == tid), 0)
        subs = t.get("subtasks") or []
        todo = t.get("todo") or []
        parts = [f"Phase {idx}/{len(phases)} {tid}",
                 f"subtasks {sum(1 for x in subs if x.get('done'))}/{len(subs)}"]
        if todo:
            parts.append(f"todo {sum(1 for x in todo if x.get('completed'))}/{len(todo)}")
        if t.get("session_turns"):
            parts.append(f"session turns {t['session_turns']}")
        if t.get("last_checkpoint"):
            parts.append(f"checkpoint {t['last_checkpoint']}")
        return " · ".join(parts)

    def closeEvent(self, event) -> None:
        if self.state in EXEC_STATES:
            r = QMessageBox.question(self, "Exit", "A run is in progress. Stop it and exit? "
                                                   "(state is persisted; you can resume later)")
            if r != QMessageBox.Yes:
                event.ignore()
                return
        self.store.save(self.settings)
        self.host.shutdown()
        event.accept()
